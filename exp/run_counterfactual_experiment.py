from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
from datetime import datetime

import numpy as np
import yaml

# Ensure local imports work when run from repo root
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from data import make_cf_dataloader_from_cfg  # noqa: E402
from metrics.clinical import MetricsAggregator, ClinicalMetrics  # noqa: E402
from models import build_model_from_cfg, list_models  # noqa: E402
from models.losses import get_loss
from models.optim import build_optim_and_sched


def _as_bool(val, default=False):
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s in {"true", "1", "yes", "y", "on"}:
        return True
    if s in {"false", "0", "no", "n", "off", "none", "null"}:
        return False
    return default

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
import torch
from tqdm.auto import tqdm


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    try:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass
    return obj


def _unwrap_model(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def _sanitize_state_dict(state):
    if not state:
        return state
    cleaned = {}
    for k, v in state.items():
        if k.startswith("_orig_mod."):
            cleaned[k[len("_orig_mod."):]] = v
        elif k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v
    return cleaned


def _parse_compile_cfg(cfg):
    if cfg is None:
        return False, {}
    if isinstance(cfg, bool):
        return cfg, {}
    if isinstance(cfg, dict):
        enabled = _as_bool(cfg.get("enabled", True))
        kwargs = {}
        if "mode" in cfg:
            kwargs["mode"] = cfg["mode"]
        if "backend" in cfg:
            kwargs["backend"] = cfg["backend"]
        if "fullgraph" in cfg:
            kwargs["fullgraph"] = _as_bool(cfg.get("fullgraph", False))
        if "dynamic" in cfg:
            kwargs["dynamic"] = _as_bool(cfg.get("dynamic", False))
        return enabled, kwargs
    return False, {}


def _maybe_compile_model(model, compile_cfg, stage: str):
    enabled, kwargs = _parse_compile_cfg(compile_cfg)
    if not enabled:
        return model
    if hasattr(model, "_orig_mod"):
        return model
    if not hasattr(torch, "compile"):
        logging.warning("torch.compile not available; skipping compile for %s", stage)
        return model
    try:
        return torch.compile(model, **kwargs)
    except Exception as e:
        logging.warning("Could not compile model for %s: %s", stage, e)
        return model


def _ensure_same_shape(preds, targets, stage: str):
    if preds.shape != targets.shape:
        raise ValueError(
            f"{stage}: preds shape {preds.shape} does not match targets shape {targets.shape}. "
            "Set dataloader.target_horizon if the model predicts a single step, or adjust pred_len "
            "so model outputs and targets align."
        )


def _train_epochs(
    model: torch.nn.Module,
    dl_train,
    *,
    pred_len: int | None,
    criterion,
    optimizer,
    scheduler,
    device: torch.device,
    epochs: int,
    use_amp: bool,
    grad_clip_norm: float | None,
    progress_label: str | None = None,
):
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    for ep in range(epochs):
        epoch_desc = progress_label or f"Epoch {ep+1}/{epochs}"
        pbar = tqdm(dl_train, desc=f"{epoch_desc} ep{ep+1}/{epochs}", leave=False)
        for batch in pbar:
            X, Y, *_ = batch
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = model(X, pred_len) if pred_len is not None else model(X)
                _ensure_same_shape(preds, Y, stage="ga-train")
                loss = criterion(preds, Y)
            if loss.requires_grad:
                scaler.scale(loss).backward()
                if grad_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                scaler.step(optimizer)
                scaler.update()
            pbar.set_postfix({"loss": f"{float(loss.item()):.4f}"})
        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()


def _eval_loss(model: torch.nn.Module, dl_val, *, pred_len: int | None, criterion, device: torch.device) -> float:
    model.eval()
    v_losses = []
    with torch.no_grad():
        for batch in dl_val:
            Xv, Yv, *_ = batch
            Xv = Xv.to(device)
            Yv = Yv.to(device)
            preds_v = model(Xv, pred_len) if pred_len is not None else model(Xv)
            _ensure_same_shape(preds_v, Yv, stage="ga-val")
            lv = criterion(preds_v, Yv)
            v_losses.append(float(lv.item()))
    return float(np.mean(v_losses)) if v_losses else float("inf")


def _eval_rmse(model: torch.nn.Module, dl_val, *, pred_len: int | None, device: torch.device) -> float:
    model.eval()
    mse_vals = []
    with torch.no_grad():
        for batch in dl_val:
            Xv, Yv, *_ = batch
            Xv = Xv.to(device)
            Yv = Yv.to(device)
            preds_v = model(Xv, pred_len) if pred_len is not None else model(Xv)
            _ensure_same_shape(preds_v, Yv, stage="ga-val")
            mse_vals.append(torch.mean((preds_v - Yv) ** 2).item())
    mse = float(np.mean(mse_vals)) if mse_vals else float("inf")
    return float(np.sqrt(mse))


def _eval_gmse(model: torch.nn.Module, dl_val, *, pred_len: int | None, device: torch.device) -> float:
    gmse_fn = get_loss("gmse")
    vals = []
    model.eval()
    with torch.no_grad():
        for batch in dl_val:
            Xv, Yv, *_ = batch
            Xv = Xv.to(device)
            Yv = Yv.to(device)
            preds_v = model(Xv, pred_len) if pred_len is not None else model(Xv)
            _ensure_same_shape(preds_v, Yv, stage="ga-val")
            vals.append(float(gmse_fn(preds_v, Yv).item()))
    return float(np.mean(vals)) if vals else float("inf")


def _eval_seg(model: torch.nn.Module, dl_val, *, pred_len: int | None, device: torch.device) -> float:
    metrics = ClinicalMetrics.from_default()
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in dl_val:
            Xv, Yv, *_ = batch
            Xv = Xv.to(device)
            preds_v = model(Xv, pred_len) if pred_len is not None else model(Xv)
            _ensure_same_shape(preds_v, Yv, stage="ga-val")
            y_np = Yv.detach().cpu().numpy()
            p_np = preds_v.detach().cpu().numpy()
            total += float(metrics.SEG(y_np, p_np, reduction="sum"))
            count += y_np.size
    return total / count if count else float("inf")


def _ga_search_weights(
    *,
    model_name: str,
    resolved_model_cfg: dict,
    input_dim: int,
    history_len: int,
    pred_len: int,
    dl_train,
    dl_val,
    training_cfg: dict,
    device: torch.device,
) -> dict | None:
    ga_cfg = dict(training_cfg.get("ga_search") or {})
    if not _as_bool(ga_cfg.get("enabled", False)):
        return None

    loss_name = str(training_cfg.get("loss", "mse"))
    if loss_name.lower() != "glimmer_weighted":
        raise ValueError("GA search is only supported with loss='glimmer_weighted'.")

    pop_size = int(ga_cfg.get("population_size", 8))
    generations = int(ga_cfg.get("generations", 5))
    w_hypo_range = ga_cfg.get("w_hypo_range", [1.0, 5.0])
    w_hyper_range = ga_cfg.get("w_hyper_range", [1.0, 5.0])
    mutation_rate = float(ga_cfg.get("mutation_rate", 0.2))
    mutation_scale = float(ga_cfg.get("mutation_scale", 0.2))
    elite_fraction = float(ga_cfg.get("elite_fraction", 0.2))
    selection = str(ga_cfg.get("selection", "tournament"))
    tournament_size = int(ga_cfg.get("tournament_size", 3))
    inner_epochs = int(ga_cfg.get("inner_epochs", 2))
    ga_seed = int(ga_cfg.get("seed", training_cfg.get("seed", 42)))
    fitness_name = str(ga_cfg.get("fitness", "rmse")).lower()
    use_amp = _as_bool(ga_cfg.get("amp", False))
    grad_clip_norm = ga_cfg.get("grad_clip_norm", training_cfg.get("grad_clip_norm", None))
    if grad_clip_norm in ("null", "None"):
        grad_clip_norm = None

    if not isinstance(w_hypo_range, (list, tuple)) or len(w_hypo_range) != 2:
        raise ValueError("ga_search.w_hypo_range must be [min, max]")
    if not isinstance(w_hyper_range, (list, tuple)) or len(w_hyper_range) != 2:
        raise ValueError("ga_search.w_hyper_range must be [min, max]")

    rng = np.random.default_rng(ga_seed)

    def _sample_uniform(rng_, lo, hi):
        return float(rng_.uniform(float(lo), float(hi)))

    def _init_population():
        return [
            {"w_hypo": _sample_uniform(rng, w_hypo_range[0], w_hypo_range[1]),
             "w_hyper": _sample_uniform(rng, w_hyper_range[0], w_hyper_range[1])}
            for _ in range(pop_size)
        ]

    def _mutate(ind):
        for key, bounds in (("w_hypo", w_hypo_range), ("w_hyper", w_hyper_range)):
            if rng.random() < mutation_rate:
                span = float(bounds[1]) - float(bounds[0])
                ind[key] += float(rng.normal(0.0, mutation_scale * span))
                ind[key] = float(np.clip(ind[key], bounds[0], bounds[1]))
        return ind

    def _crossover(a, b):
        child = {
            "w_hypo": a["w_hypo"] if rng.random() < 0.5 else b["w_hypo"],
            "w_hyper": a["w_hyper"] if rng.random() < 0.5 else b["w_hyper"],
        }
        return _mutate(child)

    def _select(pop, fitness):
        if selection == "tournament":
            idx = rng.choice(len(pop), size=tournament_size, replace=False)
            best = min(idx, key=lambda i: fitness[i])
            return pop[best]
        # fallback to uniform random selection
        return pop[int(rng.integers(0, len(pop)))]

    base_loss_params = dict(training_cfg.get("loss_params") or {})

    def _eval_candidate(ind, gen_idx: int, cand_idx: int):
        model = build_model_from_cfg(
            resolved_model_cfg, input_dim=input_dim, history_len=history_len, pred_len=pred_len
        )
        model = model.to(device)
        loss_params = dict(base_loss_params)
        loss_params["w_hypo"] = float(ind["w_hypo"])
        loss_params["w_hyper"] = float(ind["w_hyper"])
        criterion = get_loss(loss_name, **loss_params)
        optimizer, scheduler = build_optim_and_sched(model, training_cfg)
        _train_epochs(
            model,
            dl_train,
            pred_len=pred_len,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epochs=inner_epochs,
            use_amp=use_amp,
            grad_clip_norm=grad_clip_norm,
            progress_label=f"GA g{gen_idx+1}/{generations} p{cand_idx+1}/{pop_size}",
        )
        if fitness_name == "rmse":
            return _eval_rmse(model, dl_val, pred_len=pred_len, device=device)
        if fitness_name == "seg":
            return _eval_seg(model, dl_val, pred_len=pred_len, device=device)
        if fitness_name == "gmse":
            return _eval_gmse(model, dl_val, pred_len=pred_len, device=device)
        raise ValueError("ga_search.fitness must be one of: rmse, seg, gmse")

    population = _init_population()
    best = None
    best_fit = float("inf")
    elite_count = max(1, int(round(pop_size * elite_fraction)))

    gen_bar = tqdm(range(generations), desc="GA generations", leave=True)
    for gen in gen_bar:
        fitness = []
        cand_bar = tqdm(population, desc=f"GA gen {gen+1}/{generations}", leave=False)
        for idx, ind in enumerate(cand_bar):
            val = _eval_candidate(ind, gen_idx=gen, cand_idx=idx)
            fitness.append(val)
            cand_bar.set_postfix({"val": f"{val:.4f}"})
        # rank
        order = np.argsort(fitness)
        population = [population[i] for i in order]
        fitness = [fitness[i] for i in order]
        if fitness[0] < best_fit:
            best_fit = fitness[0]
            best = dict(population[0])
        logging.info(
            "GA gen %s best: w_hypo=%.4f w_hyper=%.4f val=%.4f",
            gen + 1,
            population[0]["w_hypo"],
            population[0]["w_hyper"],
            fitness[0],
        )
        # build next generation
        elites = population[:elite_count]
        next_pop = list(elites)
        while len(next_pop) < pop_size:
            p1 = _select(population, fitness)
            p2 = _select(population, fitness)
            next_pop.append(_crossover(p1, p2))
        population = next_pop
    if best is None:
        return None
    logging.info("GA selected w_hypo=%.4f w_hyper=%.4f (val=%.4f)", best["w_hypo"], best["w_hyper"], best_fit)
    return best


def _filter_metrics(res: Dict[str, Any], keep: list[str] | None) -> Dict[str, Any]:
    if not keep:
        return res

    def filt(entry: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in entry.items() if k in keep}

    out = {"global": filt(res["global"])}
    if "per_horizon" in res:
        out["per_horizon"] = [filt(ph) for ph in res["per_horizon"]]
    if "per_patient" in res:
        out["per_patient"] = {pid: filt(v) for pid, v in res["per_patient"].items()}
    return out


def _compute_subset_metrics(
    gts: np.ndarray,
    preds: np.ndarray,
    pat_ids: np.ndarray,
    mask: np.ndarray,
    metrics_keep: list[str] | None,
) -> Dict[str, Any]:
    agg = MetricsAggregator()
    Y = gts[mask]
    P = preds[mask]
    pid = pat_ids[mask]
    valid = (~np.isnan(Y)) & (~np.isnan(P))
    agg.update(Y, P, subject_id=pid, mask=valid)
    return _filter_metrics(agg.compute(), metrics_keep)


def _macro_average_global(group_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Macro-average numeric global metrics across groups (equal weight per group)."""
    globals_list = []
    for entry in group_metrics.values():
        metrics = entry.get("metrics", {})
        global_metrics = metrics.get("global", {})
        if global_metrics:
            globals_list.append(global_metrics)
    if not globals_list:
        return {}

    keys = set().union(*[gm.keys() for gm in globals_list])
    avg: Dict[str, Any] = {}
    for key in sorted(keys):
        vals = []
        for gm in globals_list:
            if key not in gm:
                continue
            v = gm[key]
            if isinstance(v, (int, float, np.floating, np.integer)):
                vals.append(float(v))
            elif isinstance(v, (list, tuple, np.ndarray)):
                arr = np.asarray(v, dtype=float)
                vals.append(arr)
        if not vals:
            continue
        if isinstance(vals[0], np.ndarray):
            stacked = np.stack(vals, axis=0)
            avg[key] = np.nanmean(stacked, axis=0)
        else:
            avg[key] = float(np.nanmean(np.asarray(vals, dtype=float)))
    return avg


def _group_metrics(
    *,
    gts: np.ndarray,
    preds: np.ndarray,
    pat_ids: np.ndarray,
    labels: np.ndarray,
    metrics_keep: list[str] | None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    grouped: Dict[str, Any] = {}
    labels = np.asarray(labels, dtype=object)
    for val in np.unique(labels):
        mask = labels == val
        if not mask.any():
            continue
        grouped[str(val)] = {
            "count": int(mask.sum()),
            "metrics": _compute_subset_metrics(gts, preds, pat_ids, mask, metrics_keep),
        }
    macro = _macro_average_global(grouped)
    return grouped, macro


def _normalize_train_cfg(raw: Dict[str, Any]) -> Dict[str, Any]:
    raw = dict(raw or {})
    if "split" in raw or "training" in raw or "model_data" in raw:
        split = dict(raw.get("split") or {})
        training = dict(raw.get("training") or raw.get("train") or {})
        model_data = dict(raw.get("model_data") or {})
        model = dict(model_data.get("model") or {})
        dataloader = dict(model_data.get("dataloader") or {})
        # Backward-compat: allow some common keys at top-level
        if "split_name" in raw and "split_name" not in split:
            split["split_name"] = raw["split_name"]
        if "split_root" in raw and "split_root" not in split:
            split["split_root"] = raw["split_root"]
        if "processed_root" in raw and "processed_root" not in split:
            split["processed_root"] = raw["processed_root"]
        if "dataset" in raw and "split_name" not in split:
            split["split_name"] = raw["dataset"]
        if "dataset" in split and "split_name" not in split:
            split["split_name"] = split["dataset"]
        if "seed" in raw and "seed" not in model_data:
            model_data["seed"] = raw["seed"]
        if "model" in raw and "name" not in model:
            model["name"] = raw["model"]
        for k in ["history_len", "pred_len", "stride", "batch_size", "univariate", "num_workers", "shuffle"]:
            if k in raw and k not in dataloader:
                dataloader[k] = raw[k]
        if "shuffle" not in dataloader:
            dataloader["shuffle"] = True
        model_data["model"] = model
        model_data["dataloader"] = dataloader
        if "compile" not in training:
            training["compile"] = False
        return {"split": split, "training": training, "model_data": model_data, "notes": raw.get("notes")}

    # Legacy flat format -> normalized
    split = {
        "split_name": raw.get("split_name") or raw.get("dataset"),
        "split_root": raw.get("split_root", "configs/preprocessing"),
        "processed_root": raw.get("processed_root", "datasets_processed"),
        "train_part": raw.get("split_part", "train"),
        "val_part": raw.get("val_split_part", "val"),
    }
    training = {
        "epochs": raw.get("epochs", 1),
        "loss": raw.get("loss", "mse"),
        "grad_clip_norm": raw.get("grad_clip_norm", None),
        "amp": raw.get("use_mixed_precision", raw.get("amp", False)),
        "compile": raw.get("compile", False),
        "early_stopping": {
            "patience": raw.get("early_stopping_patience", None),
            "min_delta": raw.get("early_stopping_min_delta", 0.0),
        },
        "optimizer": {
            "name": raw.get("optimizer", "adamw"),
            "lr": raw.get("learning_rate", 1e-3),
            "weight_decay": raw.get("weight_decay", 0.0),
            "beta1": raw.get("adamw_beta1", 0.9),
            "beta2": raw.get("adamw_beta2", 0.999),
            "eps": raw.get("adamw_eps", 1e-8),
        },
        "scheduler": {
            "name": raw.get("scheduler", "constant"),
            "min_lr": raw.get("scheduler_min_lr", 0.0),
            "patience": raw.get("scheduler_patience", 2),
            "factor": raw.get("scheduler_factor", 0.5),
        },
    }
    model_data = {
        "seed": raw.get("seed", 42),
        "model": {"name": raw.get("model", None), "params": {}},
        "dataloader": {
            "history_len": raw.get("history_len", 288),
            "pred_len": raw.get("pred_len", 24),
            "stride": raw.get("stride", 1),
            "batch_size": raw.get("batch_size", 64),
            "univariate": raw.get("univariate", False),
            "num_workers": raw.get("num_workers", 0),
            "shuffle": raw.get("shuffle", True),
            "nocturnal_start_hour": raw.get("nocturnal_start_hour", 0),
            "nocturnal_end_hour": raw.get("nocturnal_end_hour", 6),
        },
    }
    return {"split": split, "training": training, "model_data": model_data, "notes": raw.get("notes")}


def _normalize_eval_cfg(raw: Dict[str, Any]) -> Dict[str, Any]:
    raw = dict(raw or {})
    if "split" in raw or "evaluation" in raw or "model_data" in raw:
        split = dict(raw.get("split") or {})
        evaluation = dict(raw.get("evaluation") or raw.get("eval") or {})
        model_data = dict(raw.get("model_data") or {})
        model = dict(model_data.get("model") or {})
        dataloader = dict(model_data.get("dataloader") or {})
        # Backward-compat: allow some common keys at top-level
        if "split_name" in raw and "split_name" not in split:
            split["split_name"] = raw["split_name"]
        if "split_part" in raw and "split_part" not in split:
            split["split_part"] = raw["split_part"]
        if "split_root" in raw and "split_root" not in split:
            split["split_root"] = raw["split_root"]
        if "processed_root" in raw and "processed_root" not in split:
            split["processed_root"] = raw["processed_root"]
        if "dataset" in raw and "split_name" not in split:
            split["split_name"] = raw["dataset"]
        if "dataset" in split and "split_name" not in split:
            split["split_name"] = split["dataset"]
        if "model" in raw and "name" not in model:
            model["name"] = raw["model"]
        for k in ["history_len", "pred_len", "stride", "batch_size", "univariate", "num_workers"]:
            if k in raw and k not in dataloader:
                dataloader[k] = raw[k]
        if "metrics" in raw and "metrics" not in evaluation:
            evaluation["metrics"] = raw.get("metrics")
        if "event_thresholds" in raw and "event_thresholds" not in evaluation:
            evaluation["event_thresholds"] = raw.get("event_thresholds")
        if "event_threshold" in raw and "event_threshold" not in evaluation:
            evaluation["event_threshold"] = raw.get("event_threshold")
        if "nocturnal_start_hour" in raw and "nocturnal_start_hour" not in dataloader:
            dataloader["nocturnal_start_hour"] = raw.get("nocturnal_start_hour")
        if "nocturnal_end_hour" in raw and "nocturnal_end_hour" not in dataloader:
            dataloader["nocturnal_end_hour"] = raw.get("nocturnal_end_hour")
        model_data["model"] = model
        model_data["dataloader"] = dataloader
        if "compile" not in evaluation:
            evaluation["compile"] = False
        return {"split": split, "evaluation": evaluation, "model_data": model_data, "notes": raw.get("notes")}

    # Legacy flat format -> normalized
    split = {
        "split_name": raw.get("split_name") or raw.get("dataset"),
        "split_part": raw.get("split_part", "test"),
        "split_root": raw.get("split_root", "configs/preprocessing"),
        "processed_root": raw.get("processed_root", "datasets_processed"),
    }
    evaluation = {
        "metrics": raw.get("metrics", None),
        "event_thresholds": raw.get("event_thresholds", {}),
        "event_threshold": raw.get("event_threshold", None),
        "compile": raw.get("compile", False),
    }
    model_data = {
        "seed": raw.get("seed", 42),
        "model": {"name": raw.get("model", None), "params": {}},
        "dataloader": {
            "history_len": raw.get("history_len", 288),
            "pred_len": raw.get("pred_len", 24),
            "stride": raw.get("stride", 1),
            "batch_size": raw.get("batch_size", 64),
            "univariate": raw.get("univariate", False),
            "num_workers": raw.get("num_workers", 0),
            "nocturnal_start_hour": raw.get("nocturnal_start_hour", 0),
            "nocturnal_end_hour": raw.get("nocturnal_end_hour", 6),
        },
    }
    return {"split": split, "evaluation": evaluation, "model_data": model_data, "notes": raw.get("notes")}


def _resolve_model_cfg(model_name: str, model_cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    model_cfg = dict(model_cfg or {})
    params = dict(model_cfg.get("params") or {})
    return {"name": model_name, "params": params}


def run_eval(
    model,
    dl,
    pred_len: int,
    args,
    metrics_keep: list[str] | None = None,
    event_thresholds: Dict[str, int] | None = None,
    compile_cfg: Any | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    agg = MetricsAggregator()
    preds_all = []
    gts_all = []
    X_all = []
    pat_ids = []
    seq_ids = []
    scenario_names = []
    scenario_types = []
    perturb_families = []
    evt_post_meal = []
    evt_post_bolus = []
    evt_nocturnal = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(model, torch.nn.Module):
        model = model.to(device)
        model.eval()
        model = _maybe_compile_model(model, compile_cfg, stage="eval")

    pbar = tqdm(dl, desc="Eval", leave=False)
    for batch in pbar:
        (
            X,
            Y,
            pid,
            sid,
            _,
            evt_tuple,
            meta,
            scenario_name,
            scenario_type,
            perturb_family,
        ) = batch
        Y_np = Y.numpy()
        preds = None
        if isinstance(model, torch.nn.Module):
            with torch.no_grad():
                preds_t = model(X.to(device), pred_len) if pred_len is not None else model(X.to(device))
                preds = preds_t.detach().cpu().numpy()
        elif hasattr(model, "predict"):
            X_np = X.numpy()
            preds = model.predict(X_np, pred_len)
        else:
            raise ValueError("Model must implement predict or be a torch.nn.Module with forward")
        _ensure_same_shape(preds, Y_np, stage="eval")
        mask = (~np.isnan(Y_np)) & (~np.isnan(preds))
        agg.update(Y_np, preds, subject_id=pid.numpy(), mask=mask)

        preds_all.append(preds)
        gts_all.append(Y_np)
        X_all.append(X.detach().cpu().numpy())
        pat_ids.append(pid.numpy())
        seq_ids.append(sid.numpy())
        scenario_names.append(np.asarray(scenario_name, dtype=object).reshape(-1))
        scenario_types.append(np.asarray(scenario_type, dtype=object).reshape(-1))
        perturb_families.append(np.asarray(perturb_family, dtype=object).reshape(-1))

        post_meal, post_bolus, nocturnal = evt_tuple
        evt_post_meal.append(np.asarray(post_meal, dtype=np.int64).reshape(-1))
        evt_post_bolus.append(np.asarray(post_bolus, dtype=np.int64).reshape(-1))
        evt_nocturnal.append(np.asarray(nocturnal, dtype=np.int64).reshape(-1))

    preds_all = np.concatenate(preds_all, axis=0) if preds_all else np.zeros((0, pred_len))
    gts_all = np.concatenate(gts_all, axis=0) if gts_all else np.zeros((0, pred_len))
    X_all = np.concatenate(X_all, axis=0) if X_all else np.zeros((0,))
    pat_ids = np.concatenate(pat_ids, axis=0) if pat_ids else np.zeros((0,))
    seq_ids = np.concatenate(seq_ids, axis=0) if seq_ids else np.zeros((0,))
    scenario_names = np.concatenate(scenario_names, axis=0) if scenario_names else np.zeros((0,), dtype=object)
    scenario_types = np.concatenate(scenario_types, axis=0) if scenario_types else np.zeros((0,), dtype=object)
    perturb_families = np.concatenate(perturb_families, axis=0) if perturb_families else np.zeros((0,), dtype=object)
    evt_post_meal = np.concatenate(evt_post_meal, axis=0).reshape(-1) if evt_post_meal else np.zeros((0,), dtype=np.int64)
    evt_post_bolus = np.concatenate(evt_post_bolus, axis=0).reshape(-1) if evt_post_bolus else np.zeros((0,), dtype=np.int64)
    evt_nocturnal = np.concatenate(evt_nocturnal, axis=0).reshape(-1) if evt_nocturnal else np.zeros((0,), dtype=np.int64)

    base_results = agg.compute()
    base_results = _filter_metrics(base_results, metrics_keep)

    by_scenario, scenario_macro = _group_metrics(
        gts=gts_all,
        preds=preds_all,
        pat_ids=pat_ids,
        labels=scenario_names,
        metrics_keep=metrics_keep,
    )
    by_type, type_macro = _group_metrics(
        gts=gts_all,
        preds=preds_all,
        pat_ids=pat_ids,
        labels=scenario_types,
        metrics_keep=metrics_keep,
    )
    by_family, family_macro = _group_metrics(
        gts=gts_all,
        preds=preds_all,
        pat_ids=pat_ids,
        labels=perturb_families,
        metrics_keep=metrics_keep,
    )

    artifacts = {
        "preds": preds_all,
        "gts": gts_all,
        "X": X_all,
        "pat_id": pat_ids,
        "seq_id": seq_ids,
        "scenario_name": scenario_names,
        "scenario_type": scenario_types,
        "perturb_family": perturb_families,
        "event_post_meal": evt_post_meal,
        "event_post_bolus": evt_post_bolus,
        "event_nocturnal": evt_nocturnal,
    }
    out = {
        "metrics": base_results,
        "feature_cols": dl.dataset.feature_cols,
        "by_scenario": {"groups": by_scenario, "macro_avg_global": scenario_macro},
        "by_scenario_type": {"groups": by_type, "macro_avg_global": type_macro},
        "by_perturb_family": {"groups": by_family, "macro_avg_global": family_macro},
    }

    if event_thresholds:
        out["event_conditioned"] = {}
        event_map = {
            "post_meal": evt_post_meal,
            "post_bolus": evt_post_bolus,
            "nocturnal": evt_nocturnal,
        }
        for name, thr in event_thresholds.items():
            if name not in event_map:
                continue
            if thr is None or thr < 0:
                continue
            arr = event_map[name]
            mask_evt = (arr >= 0) & (arr <= thr)
            if not mask_evt.any():
                out["event_conditioned"][name] = {"threshold": thr, "metrics": {}, "count": 0}
                continue
            agg_evt = MetricsAggregator()
            Y_evt = gts_all[mask_evt]
            P_evt = preds_all[mask_evt]
            pid_evt = pat_ids[mask_evt]
            mask_valid = (~np.isnan(Y_evt)) & (~np.isnan(P_evt))
            agg_evt.update(Y_evt, P_evt, subject_id=pid_evt, mask=mask_valid)
            evt_results = _filter_metrics(agg_evt.compute(), metrics_keep)
            out["event_conditioned"][name] = {
                "threshold": thr,
                "metrics": evt_results,
                "count": int(mask_evt.sum()),
            }

    return out, artifacts


def main():
    parser = argparse.ArgumentParser(
        description="Run counterfactual experiment/eval harness.",
        allow_abbrev=False,
    )
    parser.add_argument("--train", action="store_true", help="Run training stage")
    parser.add_argument("--test", action="store_true", help="Run evaluation stage")
    parser.add_argument("--train-config", type=Path, default=Path("configs/training/dclp3_private_linear.yaml"))
    parser.add_argument("--eval-config", type=Path, default=Path("configs/evaluation/dclp3_private_linear.yaml"))
    parser.add_argument("--model", type=str, default=None, choices=list_models(), help="Model to run (overrides config)")
    parser.add_argument("--run-id", type=str, default=None, help="Run identifier (default: timestamp)")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/results"),
        help="Root directory for run outputs",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Path to a directory containing checkpoint.pt to load model from before eval")
    args = parser.parse_args()

    if not args.train and not args.test:
        args.train = True
        args.test = True

    run_id = args.run_id or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_root = Path(args.output_root)
    if out_root.suffix:  # looks like a file path, not a directory
        logging.warning("output-root looks like a file; using its parent directory instead")
        out_root = out_root.parent if out_root.parent != Path("") else Path(".")
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "logs.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
    )
    logging.info("Run ID: %s", run_id)

    # Load configs
    train_cfg_raw = yaml.safe_load(args.train_config.read_text()) if args.train_config.exists() else {}
    eval_cfg_raw = yaml.safe_load(args.eval_config.read_text()) if args.eval_config.exists() else {}
    train_cfg = _normalize_train_cfg(train_cfg_raw)
    eval_cfg = _normalize_eval_cfg(eval_cfg_raw)

    train_seed = int(train_cfg.get("model_data", {}).get("seed", 42))
    eval_seed = int(eval_cfg.get("model_data", {}).get("seed", train_seed))
    seed = train_seed if args.train else eval_seed
    set_seed(seed)

    # Determine model name
    model_name = args.model or (
        train_cfg.get("model_data", {}).get("model", {}).get("name")
        or eval_cfg.get("model_data", {}).get("model", {}).get("name")
        or "zoh"
    )

    # Training stage (optional)
    model = None
    loaded_state = None
    loaded_model_cfg = None
    if args.checkpoint_dir:
        ckpt_path = Path(args.checkpoint_dir) / "checkpoint.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            loaded_state = _sanitize_state_dict(ckpt.get("state_dict", None))
            loaded_name = ckpt.get("model_name") or ckpt.get("model")
            loaded_model_cfg = ckpt.get("model_cfg") or ckpt.get("model_config")
            if (not args.model) and loaded_name:
                model_name = str(loaded_name)

    resolved_model_cfg = None
    if args.train:
        split_cfg = dict(train_cfg.get("split") or {})
        training_cfg = dict(train_cfg.get("training") or {})
        model_data_cfg = dict(train_cfg.get("model_data") or {})
        dl_cfg = dict(model_data_cfg.get("dataloader") or {})

        split_name = split_cfg.get("split_name")
        if not split_name:
            raise ValueError("train_config.split.split_name is required")
        train_part = split_cfg.get("train_part", split_cfg.get("split_part", "train"))
        val_part = split_cfg.get("val_part", "val")

        hist = int(dl_cfg.get("history_len", 288))
        pred = int(dl_cfg.get("pred_len", 24))

        dl_train = make_cf_dataloader_from_cfg(split_cfg, dl_cfg, split_part=str(train_part), seed=seed)
        input_dim = len(dl_train.dataset.feature_cols)

        resolved_model_cfg = _resolve_model_cfg(model_name, model_data_cfg.get("model"))
        model = build_model_from_cfg(resolved_model_cfg, input_dim=input_dim, history_len=hist, pred_len=pred)

        if loaded_state and hasattr(model, "load_state_dict"):
            try:
                model.load_state_dict(loaded_state)
                logging.info("Loaded checkpoint state from %s", args.checkpoint_dir)
            except Exception as e:
                logging.warning("Could not load state_dict from checkpoint: %s", e)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(model, torch.nn.Module):
            model = model.to(device)
            model.train()
            model = _maybe_compile_model(model, training_cfg.get("compile", None), stage="train")

            base_model = _unwrap_model(model)
            trainable_flag = getattr(base_model, "trainable", True)
            has_trainable = trainable_flag and any(p.requires_grad for p in model.parameters())
            optimizer = None
            scheduler = None
            criterion = None
            epochs = int(training_cfg.get("epochs", 1))
            loss_params = dict(training_cfg.get("loss_params") or {})
            if has_trainable:
                loss_name = str(training_cfg.get("loss", "mse"))
                criterion = get_loss(loss_name, **loss_params)
                optimizer, scheduler = build_optim_and_sched(model, training_cfg)

            # Early stopping params (monitors validation loss when available)
            es_cfg = dict(training_cfg.get("early_stopping") or {})
            es_raw = es_cfg.get("patience", None)
            es_patience = None if es_raw in (None, "null", "None", -1) else int(es_raw)
            es_min_delta = float(es_cfg.get("min_delta", 0.0) or 0.0)
            best_loss = float("inf")
            es_counter = 0

            # Track best-validation weights. Early stopping below only decides
            # *when* to stop; without this snapshot the model that gets
            # evaluated and checkpointed is the final epoch, which for models
            # that diverge after their best epoch is badly wrong.
            restore_best = _as_bool(training_cfg.get("restore_best_weights", True))
            best_state = None
            best_state_monitor = float("inf")
            best_state_epoch = None

            # Validation loader when needed (early stopping or ReduceLROnPlateau)
            dl_val = None
            ga_enabled = _as_bool((training_cfg.get("ga_search") or {}).get("enabled", False))
            needs_val = (
                es_patience is not None
                or isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
                or ga_enabled
            )
            if needs_val:
                dl_val = make_cf_dataloader_from_cfg(
                    split_cfg,
                    {**dl_cfg, "shuffle": False},
                    split_part=str(val_part),
                    seed=seed,
                )

            if hasattr(base_model, "prefit"):
                base_model.prefit(dl_train, dl_val)
                logging.info("Prefit completed for %s", model_name)

            if ga_enabled:
                if dl_val is None:
                    raise ValueError("GA search requires a validation split.")
                best = _ga_search_weights(
                    model_name=model_name,
                    resolved_model_cfg=resolved_model_cfg,
                    input_dim=input_dim,
                    history_len=hist,
                    pred_len=pred,
                    dl_train=dl_train,
                    dl_val=dl_val,
                    training_cfg=training_cfg,
                    device=device,
                )
                if best:
                    loss_params["w_hypo"] = best["w_hypo"]
                    loss_params["w_hyper"] = best["w_hyper"]
                    training_cfg["loss_params"] = dict(loss_params)
                    if has_trainable:
                        criterion = get_loss(loss_name, **loss_params)

            # Mixed precision
            use_amp = _as_bool(training_cfg.get("amp", training_cfg.get("use_mixed_precision", False)))
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

            grad_clip_norm = training_cfg.get("grad_clip_norm", None)
            if grad_clip_norm in ("null", "None"):
                grad_clip_norm = None

            if not has_trainable:
                logging.info("No trainable parameters; skipping gradient-based training loop.")
            else:
                for ep in range(epochs):
                    running = []
                    pbar = tqdm(dl_train, desc=f"Epoch {ep+1}/{epochs}", leave=False)
                    for batch in pbar:
                        X, Y, pid, sid, *_ = batch
                        X = X.to(device)
                        Y = Y.to(device)
                        optimizer.zero_grad()
                        with torch.amp.autocast("cuda", enabled=use_amp):
                            preds = model(X, pred) if pred is not None else model(X)
                            _ensure_same_shape(preds, Y, stage="train")
                            loss = criterion(preds, Y)
                        if loss.requires_grad:
                            scaler.scale(loss).backward()
                            if grad_clip_norm is not None:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                            scaler.step(optimizer)
                            scaler.update()
                        running.append(float(loss.item()))
                        pbar.set_postfix(
                            {"loss": f"{np.mean(running):.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
                        )
                    epoch_loss = float(np.mean(running)) if running else float("inf")

                    # Validation
                    val_loss = None
                    val_rmse = None
                    if dl_val is not None:
                        model.eval()
                        v_losses = []
                        v_mses = []
                        with torch.no_grad():
                            for batch in dl_val:
                                Xv, Yv, pid, sid, *_ = batch
                                Xv = Xv.to(device)
                                Yv = Yv.to(device)
                                with torch.amp.autocast("cuda", enabled=use_amp):
                                    preds_v = model(Xv, pred) if pred is not None else model(Xv)
                                    _ensure_same_shape(preds_v, Yv, stage="val")
                                    lv = criterion(preds_v, Yv)
                                v_losses.append(float(lv.item()))
                                v_mses.append(float(torch.mean((preds_v - Yv) ** 2).item()))
                        val_loss = float(np.mean(v_losses)) if v_losses else float("inf")
                        if v_mses:
                            val_rmse = float(np.sqrt(np.mean(v_mses)))
                        model.train()

                        if restore_best and val_loss < best_state_monitor:
                            best_state_monitor = val_loss
                            best_state_epoch = ep + 1
                            best_state = {
                                k: v.detach().cpu().clone()
                                for k, v in _unwrap_model(model).state_dict().items()
                            }

                    if scheduler:
                        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            scheduler.step(val_loss if val_loss is not None else epoch_loss)
                        else:
                            scheduler.step()

                    # Early stopping check
                    if es_patience is not None:
                        monitor = val_loss if val_loss is not None else epoch_loss
                        if best_loss - monitor > es_min_delta:
                            best_loss = monitor
                            es_counter = 0
                        else:
                            es_counter += 1
                        if es_counter >= es_patience:
                            logging.info("Early stopping triggered at epoch %s", ep + 1)
                            break
                    if val_rmse is not None:
                        wh = training_cfg.get("loss_params", {}).get("w_hypo")
                        wy = training_cfg.get("loss_params", {}).get("w_hyper")
                        if wh is not None or wy is not None:
                            logging.info(
                                "Epoch %s validation RMSE: %.6f (w_hypo=%s, w_hyper=%s)",
                                ep + 1,
                                val_rmse,
                                wh,
                                wy,
                            )
                        else:
                            logging.info("Epoch %s validation RMSE: %.6f", ep + 1, val_rmse)

                if best_state is not None:
                    _unwrap_model(model).load_state_dict(best_state)
                    logging.info(
                        "Restored best-validation weights from epoch %s (val loss %.6f)",
                        best_state_epoch,
                        best_state_monitor,
                    )

        logging.info("Training completed: model=%s split=%s/%s", model_name, split_name, train_part)

    # Evaluation stage
    results = {}
    artifacts = {}
    if args.test:
        split_cfg = dict(eval_cfg.get("split") or {})
        eval_block = dict(eval_cfg.get("evaluation") or {})
        model_data_cfg = dict(eval_cfg.get("model_data") or {})
        dl_cfg = dict(model_data_cfg.get("dataloader") or {})

        split_name = split_cfg.get("split_name")
        if not split_name:
            raise ValueError("eval_config.split.split_name is required")
        split_part = split_cfg.get("split_part", "test")

        hist = int(dl_cfg.get("history_len", 288))
        pred = int(dl_cfg.get("pred_len", 24))

        dl_eval = make_cf_dataloader_from_cfg(
            split_cfg,
            {**dl_cfg, "shuffle": False},
            split_part=str(split_part),
            seed=seed,
        )
        if model is None:
            input_dim = len(dl_eval.dataset.feature_cols)
            eval_model_cfg = model_data_cfg.get("model") or loaded_model_cfg or {}
            resolved_model_cfg = _resolve_model_cfg(model_name, eval_model_cfg)
            model = build_model_from_cfg(resolved_model_cfg, input_dim=input_dim, history_len=hist, pred_len=pred)
            if loaded_state and hasattr(model, "load_state_dict"):
                try:
                    model.load_state_dict(loaded_state)
                    logging.info("Loaded checkpoint state from %s", args.checkpoint_dir)
                except Exception as e:
                    logging.warning("Could not load state_dict from checkpoint: %s", e)
        metrics_keep = eval_block.get("metrics")
        evt_thrs = eval_block.get("event_thresholds")
        if not isinstance(evt_thrs, dict):
            evt_thrs = {}
        # Backward compatible single threshold
        evt_single = eval_block.get("event_threshold")
        if (not evt_thrs) and evt_single not in (None, -1, "null"):
            evt_thrs = {"post_meal": int(evt_single), "post_bolus": int(evt_single)}
        # Normalize thresholds
        evt_thrs = {
            str(k): (None if v in (None, -1, "null") else int(v))
            for k, v in evt_thrs.items()
        }
        results, artifacts = run_eval(
            model,
            dl_eval,
            pred,
            args,
            metrics_keep=metrics_keep,
            event_thresholds=evt_thrs,
            compile_cfg=eval_block.get("compile", None),
        )
        logging.info("Finished eval. Metrics keys: %s", list(results.get("metrics", {}).keys()))

    # Save outputs
    if artifacts:
        jsonable = to_jsonable({"train_config": train_cfg, "eval_config": eval_cfg, **results})
        (run_dir / "metrics.json").write_text(json.dumps(jsonable, indent=2))
        with open(run_dir / "config.yaml", "w") as f:
            yaml.safe_dump({"train": train_cfg, "eval": eval_cfg, "model": model_name}, f)
        np.savez(run_dir / "preds.npz", preds=artifacts["preds"], pat_id=artifacts["pat_id"], seq_id=artifacts["seq_id"])
        np.savez(run_dir / "GT.npz", gts=artifacts["gts"], pat_id=artifacts["pat_id"], seq_id=artifacts["seq_id"])
        np.savez(
            run_dir / "event.npz",
            post_meal=artifacts["event_post_meal"],
            post_bolus=artifacts["event_post_bolus"],
            nocturnal=artifacts["event_nocturnal"],
            pat_id=artifacts["pat_id"],
            seq_id=artifacts["seq_id"],
        )
        np.savez(
            run_dir / "cf_artifacts.npz",
            X=artifacts["X"],
            gts=artifacts["gts"],
            preds=artifacts["preds"],
            pat_id=artifacts["pat_id"],
            seq_id=artifacts["seq_id"],
            scenario_name=artifacts["scenario_name"],
            scenario_type=artifacts["scenario_type"],
            perturb_family=artifacts["perturb_family"],
        )
    # Save checkpoint (even for non-parametric models) for reproducibility
    if model is not None:
        try:
            base_model = _unwrap_model(model)
            state = base_model.state_dict() if hasattr(base_model, "state_dict") else {}
            ckpt_out = {"model_name": model_name, "state_dict": state}
            if resolved_model_cfg is not None:
                ckpt_out["model_cfg"] = resolved_model_cfg
            torch.save(ckpt_out, run_dir / "checkpoint.pt")
        except Exception as e:
            logging.warning("Could not save checkpoint: %s", e)


if __name__ == "__main__":
    main()
