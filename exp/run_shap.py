from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Ensure local imports work when run from repo root
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from data.dataloader import TimeSeriesWindowDataset  # noqa: E402
from exp.shap_utils import (  # noqa: E402
    SLICE_LABELS,
    apply_fixed_mask_inplace,
    apply_player_inplace,
    build_players,
    infer_modalities,
    load_shap_config,
    predict_mean_horizons,
    slice_label_from_events,
)
from exp.shap_reconstruction import compute_reconstruction_stats, reconstruct_inplace  # noqa: E402
from models import get_model  # noqa: E402


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    logging.warning("GPU not available; using CPU.")
    return torch.device("cpu")


def _load_checkpoint(checkpoint_dir: Path | None, checkpoint_path: Path | None) -> Dict[str, Any]:
    if checkpoint_path is not None:
        ckpt_path = Path(checkpoint_path)
    elif checkpoint_dir is not None:
        ckpt_path = Path(checkpoint_dir) / "checkpoint.pt"
    else:
        raise ValueError("Must provide --checkpoint_dir or --checkpoint_path")
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    return torch.load(ckpt_path, map_location="cpu")


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


def _maybe_compile_model(model, compile_cfg):
    enabled, kwargs = _parse_compile_cfg(compile_cfg)
    if not enabled:
        return model
    if hasattr(model, "_orig_mod"):
        return model
    if not hasattr(torch, "compile"):
        logging.warning("torch.compile not available; skipping compile")
        return model
    try:
        return torch.compile(model, **kwargs)
    except Exception as e:
        logging.warning("Could not compile model: %s", e)
        return model


def _sample_background_windows(
    pool: List[Tuple[int, np.ndarray]],
    k: int,
    rng: np.random.Generator,
    exclude_index: int | None,
    exclude_self: bool,
) -> np.ndarray:
    if not pool:
        raise ValueError("Empty background pool")
    if exclude_self and exclude_index is not None:
        filtered = [item for item in pool if item[0] != exclude_index]
        if filtered:
            pool = filtered
    idx = rng.integers(0, len(pool), size=k)
    return np.stack([pool[int(i)][1] for i in idx], axis=0)


def _choose_pool(
    pid: int,
    slice_label: str,
    pools_pid_slice: Dict[int, Dict[str, List[Tuple[int, np.ndarray]]]],
    pools_pid_all: Dict[int, List[Tuple[int, np.ndarray]]],
    pools_global_slice: Dict[str, List[Tuple[int, np.ndarray]]],
    pools_global_all: List[Tuple[int, np.ndarray]],
    cfg,
) -> List[Tuple[int, np.ndarray]]:
    bg = cfg.background

    pid_slice = pools_pid_slice.get(pid, {}).get(slice_label, [])
    if bg.prefer_same_slice and (len(pid_slice) >= bg.min_pool_size_same_slice):
        return pid_slice

    pid_any = pools_pid_all.get(pid, [])
    if pid_any:
        return pid_any

    if not bg.allow_global_fallback:
        return []

    if bg.global_prefer_same_slice:
        g_slice = pools_global_slice.get(slice_label, [])
        if g_slice and (len(g_slice) >= bg.min_pool_size_same_slice):
            return g_slice

    return pools_global_all


def _write_metadata_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _predict_batched_steps(
    model: torch.nn.Module,
    X_sequence_flat: torch.Tensor,
    n_steps: int,
    n_backgrounds: int,
    horizon_idxs: List[int],
    microbatch_size: int,
    pred_len: int,
    input_layout: str = "BTC",
) -> torch.Tensor:
    """
    Optimized inference for GPU.
    Takes a flattened tensor of shape ((P+1)*K, C, T) or ((P+1)*K, T, C).
    Returns tensor of shape (P+1, H) containing means over K.
    """
    # Run inference in microbatches to avoid OOM
    total_samples = X_sequence_flat.size(0)
    all_preds = []
    
    input_layout = str(input_layout).upper()
    if input_layout not in {"BTC", "BCT"}:
        raise ValueError("input_layout must be 'BTC' or 'BCT'")

    with torch.no_grad():
        for start_idx in range(0, total_samples, microbatch_size):
            end_idx = min(start_idx + microbatch_size, total_samples)
            batch_x = X_sequence_flat[start_idx:end_idx]
            
            if input_layout == "BCT":
                # Convert from [B, T, C] -> [B, C, T]
                batch_x = batch_x.permute(0, 2, 1).contiguous()

            # Direct model call (bypassing overhead)
            # Assuming model outputs (Batch, PredLen)
            try:
                preds = model(batch_x, pred_len)
            except TypeError:
                preds = model(batch_x)
            
            # Select specific horizons
            if horizon_idxs:
                preds = preds[:, horizon_idxs]
                
            all_preds.append(preds)
            
    # Concatenate all microbatches
    # Shape: ((P+1)*K, H)
    full_preds = torch.cat(all_preds, dim=0)
    
    # Reshape to (P+1, K, H) to average over K
    n_horizons = full_preds.size(1)
    reshaped = full_preds.view(n_steps, n_backgrounds, n_horizons)
    
    # Mean over the background dimension (dim 1)
    # Result: (P+1, H)
    return reshaped.mean(dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc permutation-SHAP for BG forecasting.", allow_abbrev=False)
    parser.add_argument(
        "--split_path",
        type=Path,
        default=None,
        help="Path to the split CSV file (overrides shap_config.data.split_path)",
    )
    parser.add_argument("--shap_config", type=Path, required=True, help="Path to SHAP config (.yaml/.json)")
    parser.add_argument("--checkpoint_dir", type=Path, default=None, help="Directory containing checkpoint.pt")
    parser.add_argument("--checkpoint_path", type=Path, default=None, help="Explicit path to checkpoint.pt")
    parser.add_argument("--model", type=str, default=None, help="Model name override (if checkpoint lacks it)")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory for SHAP artifacts")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling logs during SHAP run",
    )
    parser.add_argument(
        "--profile_every",
        type=int,
        default=10,
        help="Emit profiling stats every N permutations (when --profile is set)",
    )
    args = parser.parse_args()

    cfg = load_shap_config(args.shap_config)
    torch.manual_seed(cfg.shap.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.shap.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "logs.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_path)],
    )
    logging.info("Started SHAP run at %s", datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))

    data = cfg.data
    split_csv_path = args.split_path or (Path(data.split_path) if data.split_path else None)
    if split_csv_path is None:
        raise ValueError("Missing split CSV path. Provide --split_path or set data.split_path in the SHAP config.")
    split_csv_path = Path(split_csv_path)
    if not split_csv_path.exists():
        raise FileNotFoundError(split_csv_path)

    ds = TimeSeriesWindowDataset(
        split_name="__csv__",
        split_part=data.split_part,
        split_root=Path("configs/preprocessing"),
        processed_root=Path(data.processed_root),
        history_len=data.history_len,
        pred_len=data.pred_len,
        stride=data.stride,
        univariate=data.univariate,
        nocturnal_start_hour=data.nocturnal_start_hour,
        nocturnal_end_hour=data.nocturnal_end_hour,
        csv_path=split_csv_path,
        subset_size=data.subsample_size,
        subset_fraction=data.subsample_fraction,
        subset_seed=data.subsample_seed,
    )

    # DataLoader: use deterministic ordering for SHAP outputs.
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=data.batch_size,
        shuffle=False,
        num_workers=data.num_workers,
        drop_last=False,
    )

    players, fixed_mask = build_players(
        feature_cols=ds.feature_cols,
        history_len=data.history_len,
        windows_steps_by_modality=cfg.windows_steps_by_modality,
        modality_channels=cfg.modality_channels,
    )
    if not players:
        raise ValueError("No players constructed. Check modality/channel mapping and windows.")
    logging.info("Players: %s", len(players))

    horizon_idxs = [h - 1 for h in cfg.shap.horizons_steps]
    logging.info("Explaining horizons_steps=%s (indices=%s)", cfg.shap.horizons_steps, horizon_idxs)

    ckpt = _load_checkpoint(args.checkpoint_dir, args.checkpoint_path)
    model_name = args.model or ckpt.get("model_name") or ckpt.get("model") or None
    if not model_name:
        raise ValueError("Could not determine model name. Provide --model or use a checkpoint with model_name.")

    input_dim = 1 if data.univariate else len(ds.feature_cols)
    model_params = {}
    if isinstance(ckpt.get("model_cfg"), dict):
        model_params = dict(ckpt["model_cfg"].get("params") or {})
    model = get_model(
        model_name,
        input_dim=input_dim,
        history_len=data.history_len,
        pred_len=data.pred_len,
        **model_params,
    )
    if isinstance(model, torch.nn.Module):
        state = _sanitize_state_dict(ckpt.get("state_dict", {}) or {})
        try:
            model.load_state_dict(state, strict=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load state_dict for model '{model_name}': {e}") from e

    device = _select_device()
    recon_device = device
    
    # Check if we can use the fast GPU path
    is_torch_model = isinstance(model, torch.nn.Module)
    use_fast_path = is_torch_model and (device.type == "cuda")
    
    if is_torch_model:
        model = model.to(device)
        model.eval()
        model = _maybe_compile_model(model, cfg.model.compile)
        if use_fast_path:
            logging.info(f"Accelerated GPU path enabled for model {model_name}")
        else:
            logging.info(f"Using CPU path for Torch model {model_name} (No CUDA detected)")
    else:
        logging.info(f"Using standard CPU path for non-Torch model {model_name}")

    bg = cfg.background
    pools_pid_slice: Dict[int, Dict[str, List[Tuple[int, np.ndarray]]]] = {}
    pools_pid_all: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    pools_global_slice: Dict[str, List[Tuple[int, np.ndarray]]] = {k: [] for k in SLICE_LABELS}
    pools_global_all: List[Tuple[int, np.ndarray]] = []
    logging.info("Building background pools...")
    sample_counter = 0
    for batch in tqdm(dl, desc="Background", leave=False):
        X, Y, pid, sid, ctx_end_date, evt_tuple, meta = batch
        X_np = X.numpy()
        pid_np = pid.numpy().reshape(-1)
        post_meal_b, post_bolus_b, nocturnal_b = evt_tuple
        post_meal_np = np.asarray(post_meal_b, dtype=np.int64).reshape(-1)
        post_bolus_np = np.asarray(post_bolus_b, dtype=np.int64).reshape(-1)
        noct_np = np.asarray(nocturnal_b, dtype=np.int64).reshape(-1)
        for j in range(X_np.shape[0]):
            idx = sample_counter
            sample_counter += 1
            p = int(pid_np[j])
            label = slice_label_from_events(int(post_meal_np[j]), int(post_bolus_np[j]), int(noct_np[j]))
            entry = (idx, X_np[j].astype(np.float32, copy=False))

            pools_pid_slice.setdefault(p, {}).setdefault(label, [])
            if (bg.max_per_pid_slice is None) or (len(pools_pid_slice[p][label]) < bg.max_per_pid_slice):
                pools_pid_slice[p][label].append(entry)

            pools_pid_all.setdefault(p, [])
            if (bg.max_per_pid is None) or (len(pools_pid_all[p]) < bg.max_per_pid):
                pools_pid_all[p].append(entry)

            if bg.allow_global_fallback:
                if (bg.max_global is None) or (len(pools_global_all) < bg.max_global):
                    pools_global_all.append(entry)
                if (bg.max_global is None) or (len(pools_global_slice[label]) < bg.max_global):
                    pools_global_slice[label].append(entry)

    logging.info("Background pools built. Unique pids=%s total_windows=%s", len(pools_pid_all), sample_counter)

    modality_indices = infer_modalities(ds.feature_cols, modality_channels=cfg.modality_channels)
    all_windows = [w for _, w in pools_global_all] if pools_global_all else [w for v in pools_pid_all.values() for _, w in v]
    recon_stats = compute_reconstruction_stats(all_windows, modality_indices, cfg.reconstruction)

    # Save players + resolved config for reproducibility.
    (out_dir / "players.json").write_text(json.dumps([{
        "name": p.name, "modality": p.modality, "window_steps": list(p.window_steps),
        "time_slice": [p.time_slice.start, p.time_slice.stop], "channel_idxs": list(p.channel_idxs),
        "channel_names": list(p.channel_names)
    } for p in players], indent=2))
    
    with (out_dir / "config_resolved.json").open("w") as f:
        json.dump({
            "split_path": str(split_csv_path), "model": model_name,
            "config": {
                "data": asdict(cfg.data), "shap": asdict(cfg.shap), "background": asdict(cfg.background),
                "output": asdict(cfg.output), "model": asdict(cfg.model),
                "windows_steps_by_modality": {k: [list(x) for x in v] for k, v in cfg.windows_steps_by_modality.items()},
                "modality_channels": cfg.modality_channels,
            },
        }, f, indent=2)

    group_keys = list(SLICE_LABELS) if cfg.output.partition_by_slice else ["all"]
    per_slice = {k: {"shap": [], "base": [], "pred": [], "pid": [], "sid": [], "evt": [], "idx": []} for k in group_keys}
    meta_rows = {k: [] for k in group_keys}

    rng = np.random.default_rng(cfg.shap.seed)
    diag_rows = []
    diag_seen = 0
    rmse_sse = 0.0
    rmse_count = 0
    prof = {
        "base_recon": 0.0,
        "base_transfer": 0.0,
        "base_infer": 0.0,
        "perm_recon": 0.0,
        "perm_transfer": 0.0,
        "perm_infer": 0.0,
        "perm_count": 0,
        "recon_cgm": 0.0,
        "recon_basal": 0.0,
        "recon_meal_zero": 0.0,
        "recon_bolus_zero": 0.0,
        "recon_meal_coupling": 0.0,
        "recon_bolus_coupling": 0.0,
    }
    profile_enabled = bool(args.profile)
    profile_every = max(1, int(args.profile_every))

    logging.info("Computing SHAP values...")
    global_index = 0
    
    # --- MAIN LOOP ---
    for batch in tqdm(dl, desc="SHAP", leave=True):
        X, Y, pid, sid, ctx_end_date, evt_tuple, meta = batch
        X_np = X.numpy().astype(np.float32, copy=False)
        Y_np = Y.numpy().astype(np.float32, copy=False)
        pid_np = pid.numpy().reshape(-1)
        sid_np = sid.numpy().reshape(-1)
        post_meal_b, post_bolus_b, nocturnal_b = evt_tuple
        post_meal_np = np.asarray(post_meal_b, dtype=np.int64).reshape(-1)
        post_bolus_np = np.asarray(post_bolus_b, dtype=np.int64).reshape(-1)
        noct_np = np.asarray(nocturnal_b, dtype=np.int64).reshape(-1)

        for j in range(X_np.shape[0]):
            sample_idx = global_index
            global_index += 1

            p, s = int(pid_np[j]), int(sid_np[j])
            pm, pb, no = int(post_meal_np[j]), int(post_bolus_np[j]), int(noct_np[j])
            label = slice_label_from_events(pm, pb, no)
            store_key = label if cfg.output.partition_by_slice else "all"

            pool = _choose_pool(p, label, pools_pid_slice, pools_pid_all, pools_global_slice, pools_global_all, cfg)
            if not pool:
                logging.warning("No background pool found for pid=%s slice=%s; skipping sample=%s", p, label, sample_idx)
                continue

            # Draw background once for Base
            bg_X = _sample_background_windows(pool, cfg.shap.n_background_draws, rng, sample_idx, cfg.background.exclude_self)
            X_base = torch.as_tensor(bg_X, dtype=torch.float32, device=recon_device)
            x_orig = torch.as_tensor(X_np[j], dtype=torch.float32, device=recon_device)
            apply_fixed_mask_inplace(X_base, x_orig, fixed_mask)
            mask_base = ~fixed_mask
            t0 = time.perf_counter()
            reconstruct_inplace(
                X_base,
                mask_base,
                modality_indices,
                cfg.reconstruction,
                recon_stats,
                profiler=prof if profile_enabled else None,
            )
            prof["base_recon"] += time.perf_counter() - t0

            P, H = len(players), len(horizon_idxs)
            
            # --- PATH 1: Optimized GPU Path ---
            if use_fast_path:
                # 1. Compute Base (Empty) Value
                # FIX: Use flatten(0, 0) (no-op) or just pass X_base directly as it is already (K, ...)
                # effectively (Batch, ...) for the model.
                if X_base.device != device:
                    t0 = time.perf_counter()
                    X_base_gpu = X_base.to(device, non_blocking=True)
                    prof["base_transfer"] += time.perf_counter() - t0
                else:
                    X_base_gpu = X_base
                t0 = time.perf_counter()
                base = _predict_batched_steps(
                    model, X_base_gpu, 
                    n_steps=1, n_backgrounds=cfg.shap.n_background_draws, 
                    horizon_idxs=horizon_idxs, microbatch_size=cfg.shap.microbatch_size,
                    pred_len=data.pred_len, input_layout=data.input_layout,
                )[0].cpu().numpy()
                prof["base_infer"] += time.perf_counter() - t0

                shap_sum = torch.zeros((P, H), dtype=torch.float32, device=device)
                pred_values = None 

                perm_batch = int(cfg.shap.perm_batch_size)
                for r0 in range(0, cfg.shap.n_permutations, perm_batch):
                    r1 = min(cfg.shap.n_permutations, r0 + perm_batch)
                    batch_orders = []
                    batch_steps = []

                    for _ in range(r0, r1):
                        if cfg.shap.resample_background_each_permutation:
                            bg_X_r = _sample_background_windows(
                                pool,
                                cfg.shap.n_background_draws,
                                rng,
                                sample_idx,
                                cfg.background.exclude_self,
                            )
                            X_curr = torch.as_tensor(bg_X_r, dtype=torch.float32, device=recon_device)
                            apply_fixed_mask_inplace(X_curr, x_orig, fixed_mask)
                            mask_curr = mask_base.copy()
                            t0 = time.perf_counter()
                            reconstruct_inplace(
                                X_curr,
                                mask_curr,
                                modality_indices,
                                cfg.reconstruction,
                                recon_stats,
                                profiler=prof if profile_enabled else None,
                            )
                            prof["perm_recon"] += time.perf_counter() - t0
                        else:
                            X_curr = X_base.clone()
                            mask_curr = mask_base.copy()

                        order = rng.permutation(P)
                        batch_orders.append(order)

                        step_tensors = [X_curr.clone()]
                        t0 = time.perf_counter()
                        for pi in order:
                            apply_player_inplace(X_curr, x_orig, players[int(pi)])
                            ts = players[int(pi)].time_slice
                            ch = list(players[int(pi)].channel_idxs)
                            if ts.start != ts.stop and ch:
                                mask_curr[ts, ch] = False
                            touched = {players[int(pi)].modality}
                            if players[int(pi)].modality in {"meal", "bolus"}:
                                touched = {"meal", "bolus"}
                            reconstruct_inplace(
                                X_curr,
                                mask_curr,
                                modality_indices,
                                cfg.reconstruction,
                                recon_stats,
                                touched_modalities=tuple(touched),
                                profiler=prof if profile_enabled else None,
                            )
                            step_tensors.append(X_curr.clone())
                        prof["perm_recon"] += time.perf_counter() - t0

                        batch_steps.append(torch.stack(step_tensors, dim=0))
                        prof["perm_count"] += 1

                    X_sequence = torch.stack(batch_steps, dim=0)  # (R, P+1, K, T, C)
                    X_sequence_flat = X_sequence.flatten(0, 2)
                    if X_sequence_flat.device != device:
                        t0 = time.perf_counter()
                        X_sequence_flat = X_sequence_flat.to(device, non_blocking=True)
                        prof["perm_transfer"] += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    y_sequence = _predict_batched_steps(
                        model,
                        X_sequence_flat,
                        n_steps=(r1 - r0) * (P + 1),
                        n_backgrounds=cfg.shap.n_background_draws,
                        horizon_idxs=horizon_idxs,
                        microbatch_size=cfg.shap.microbatch_size,
                        pred_len=data.pred_len,
                        input_layout=data.input_layout,
                    )
                    prof["perm_infer"] += time.perf_counter() - t0

                    y_sequence = y_sequence.view(r1 - r0, P + 1, -1)
                    for bi, order in enumerate(batch_orders):
                        diffs = y_sequence[bi, 1:] - y_sequence[bi, :-1]
                        shap_sum[order] += diffs
                        if pred_values is None:
                            pred_values = y_sequence[bi, -1]

                    if profile_enabled and (prof["perm_count"] % profile_every == 0):
                        per = float(prof["perm_count"])
                        logging.info(
                            "PROFILE (perm=%d): recon=%.3fs transfer=%.3fs infer=%.3fs | total recon=%.3fs transfer=%.3fs infer=%.3fs",
                            int(per),
                            prof["perm_recon"] / per,
                            prof["perm_transfer"] / per,
                            prof["perm_infer"] / per,
                            prof["perm_recon"],
                            prof["perm_transfer"],
                            prof["perm_infer"],
                        )
                        logging.info(
                            "PROFILE recon breakdown (avg/perm): cgm=%.4fs basal=%.4fs "
                            "meal_zero=%.4fs bolus_zero=%.4fs meal_cpl=%.4fs bolus_cpl=%.4fs",
                            prof["recon_cgm"] / per,
                            prof["recon_basal"] / per,
                            prof["recon_meal_zero"] / per,
                            prof["recon_bolus_zero"] / per,
                            prof["recon_meal_coupling"] / per,
                            prof["recon_bolus_coupling"] / per,
                        )

                shap_values = (shap_sum / float(cfg.shap.n_permutations)).cpu().numpy()
                if pred_values is not None:
                    pred_values = pred_values.cpu().numpy()
                else:
                    pred_values = np.full((H,), np.nan, dtype=np.float32)
            else:
                base = predict_mean_horizons(
                    model=model, Xk=X_base, pred_len=data.pred_len, horizon_idxs=horizon_idxs,
                    device=device, microbatch_size=cfg.shap.microbatch_size, input_layout=data.input_layout,
                )
                shap_sum = np.zeros((P, H), dtype=np.float32)
                pred_values = None

                for r in range(cfg.shap.n_permutations):
                    if cfg.shap.resample_background_each_permutation:
                        bg_X = _sample_background_windows(
                            pool,
                            cfg.shap.n_background_draws,
                            rng,
                            sample_idx,
                            cfg.background.exclude_self,
                        )
                        X_curr = torch.as_tensor(bg_X, dtype=torch.float32, device=recon_device)
                        apply_fixed_mask_inplace(X_curr, x_orig, fixed_mask)
                        mask_curr = mask_base.copy()
                        reconstruct_inplace(
                            X_curr,
                            mask_curr,
                            modality_indices,
                            cfg.reconstruction,
                            recon_stats,
                            profiler=prof if profile_enabled else None,
                        )
                        y_prev = predict_mean_horizons(
                            model=model,
                            Xk=X_curr,
                            pred_len=data.pred_len,
                            horizon_idxs=horizon_idxs,
                            device=device,
                            microbatch_size=cfg.shap.microbatch_size,
                            input_layout=data.input_layout,
                        )
                    else:
                        X_curr = X_base.clone()
                        mask_curr = mask_base.copy()
                        y_prev = base

                    order = rng.permutation(P)
                    for pi in order:
                        apply_player_inplace(X_curr, x_orig, players[int(pi)])
                        ts = players[int(pi)].time_slice
                        ch = list(players[int(pi)].channel_idxs)
                        if ts.start != ts.stop and ch:
                            mask_curr[ts, ch] = False
                        touched = {players[int(pi)].modality}
                        if players[int(pi)].modality in {"meal", "bolus"}:
                            touched = {"meal", "bolus"}
                        reconstruct_inplace(
                            X_curr,
                            mask_curr,
                            modality_indices,
                            cfg.reconstruction,
                            recon_stats,
                            touched_modalities=tuple(touched),
                            profiler=prof if profile_enabled else None,
                        )
                        y_new = predict_mean_horizons(
                            model=model,
                            Xk=X_curr,
                            pred_len=data.pred_len,
                            horizon_idxs=horizon_idxs,
                            device=device,
                            microbatch_size=cfg.shap.microbatch_size,
                            input_layout=data.input_layout,
                        )
                        shap_sum[int(pi)] += (y_new - y_prev).astype(np.float32, copy=False)
                        y_prev = y_new

                    prof["perm_count"] += 1
                    if profile_enabled and (prof["perm_count"] % profile_every == 0):
                        per = float(prof["perm_count"])
                        logging.info(
                            "PROFILE (perm=%d): recon=%.3fs transfer=%.3fs infer=%.3fs | total recon=%.3fs transfer=%.3fs infer=%.3fs",
                            int(per),
                            prof["perm_recon"] / per if per else 0.0,
                            prof["perm_transfer"] / per if per else 0.0,
                            prof["perm_infer"] / per if per else 0.0,
                            prof["perm_recon"],
                            prof["perm_transfer"],
                            prof["perm_infer"],
                        )
                        logging.info(
                            "PROFILE recon breakdown (avg/perm): cgm=%.4fs basal=%.4fs "
                            "meal_zero=%.4fs bolus_zero=%.4fs meal_cpl=%.4fs bolus_cpl=%.4fs",
                            prof["recon_cgm"] / per,
                            prof["recon_basal"] / per,
                            prof["recon_meal_zero"] / per,
                            prof["recon_bolus_zero"] / per,
                            prof["recon_meal_coupling"] / per,
                            prof["recon_bolus_coupling"] / per,
                        )

                    if pred_values is None:
                        pred_values = y_prev
                
                shap_values = shap_sum / float(cfg.shap.n_permutations)
                if pred_values is None:
                    pred_values = np.full((H,), np.nan, dtype=np.float32)

            # --- END PATHS ---

            per_slice[store_key]["shap"].append(shap_values)
            per_slice[store_key]["base"].append(base.astype(np.float32, copy=False))
            per_slice[store_key]["pred"].append(np.asarray(pred_values, dtype=np.float32))
            per_slice[store_key]["pid"].append(p)
            per_slice[store_key]["sid"].append(s)
            per_slice[store_key]["evt"].append((pm, pb, no))
            per_slice[store_key]["idx"].append(sample_idx)

            # RMSE over selected horizons (pred_values corresponds to full coalition)
            if pred_values is not None and np.all(np.isfinite(pred_values)):
                y_true = Y_np[j][horizon_idxs]
                diff = pred_values - y_true
                rmse_sse += float(np.sum(diff ** 2))
                rmse_count += int(diff.size)

            meta_rows[store_key].append({
                "sample_index": sample_idx, "pid": p, "sid": s, "ctx_end_date": str(ctx_end_date[j]),
                "slice_label": label, "post_meal": pm, "post_bolus": pb, "nocturnal": no,
            })

            if cfg.output.diagnostics_subset_size:
                diag_seen += 1
                err = np.abs((pred_values - base) - shap_values.sum(axis=0))
                row = {"sample_index": sample_idx, **{f"err_h{h}": float(e) for h, e in zip(cfg.shap.horizons_steps, err)}}
                k = cfg.output.diagnostics_subset_size
                if len(diag_rows) < k:
                    diag_rows.append(row)
                else:
                    jrep = int(rng.integers(0, diag_seen))
                    if jrep < k:
                        diag_rows[jrep] = row

            if cfg.output.max_eval_samples is not None and global_index >= cfg.output.max_eval_samples:
                break
        if cfg.output.max_eval_samples is not None and global_index >= cfg.output.max_eval_samples:
            break

    horizons_steps = np.asarray(cfg.shap.horizons_steps, dtype=np.int64)
    diagnostics = {
        "horizons_steps": cfg.shap.horizons_steps, "n_players": len(players), "diagnostics_subset_size": len(diag_rows),
    }
    if diag_rows:
        err_keys = [k for k in diag_rows[0].keys() if k.startswith("err_h")]
        diagnostics["mean_abs_error_by_horizon"] = {k: float(np.mean([r[k] for r in diag_rows])) for k in err_keys}
        diagnostics["max_abs_error_by_horizon"] = {k: float(np.max([r[k] for r in diag_rows])) for k in err_keys}
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    for label in group_keys:
        n = len(per_slice[label]["idx"])
        if n == 0: continue
        np.savez(
            out_dir / f"shap_{label}.npz",
            shap_values=np.stack(per_slice[label]["shap"], axis=0),
            base_values=np.stack(per_slice[label]["base"], axis=0),
            pred_values=np.stack(per_slice[label]["pred"], axis=0),
            horizons_steps=horizons_steps,
            pid=np.asarray(per_slice[label]["pid"], dtype=np.int64),
            sid=np.asarray(per_slice[label]["sid"], dtype=np.int64),
            sample_index=np.asarray(per_slice[label]["idx"], dtype=np.int64),
            post_meal=np.asarray(per_slice[label]["evt"], dtype=np.int64)[:, 0],
            post_bolus=np.asarray(per_slice[label]["evt"], dtype=np.int64)[:, 1],
            nocturnal=np.asarray(per_slice[label]["evt"], dtype=np.int64)[:, 2],
        )
        _write_metadata_csv(out_dir / f"metadata_{label}.csv", meta_rows[label])

    if profile_enabled and prof["perm_count"] > 0:
        per = float(prof["perm_count"])
        logging.info(
            "SHAP timing (total): base_recon=%.3fs base_transfer=%.3fs base_infer=%.3fs "
            "perm_recon=%.3fs perm_transfer=%.3fs perm_infer=%.3fs perms=%d",
            prof["base_recon"],
            prof["base_transfer"],
            prof["base_infer"],
            prof["perm_recon"],
            prof["perm_transfer"],
            prof["perm_infer"],
            int(per),
        )
        logging.info(
            "SHAP timing (per‑perm): recon=%.3fs transfer=%.3fs infer=%.3fs",
            prof["perm_recon"] / per,
            prof["perm_transfer"] / per,
            prof["perm_infer"] / per,
        )
        logging.info(
            "SHAP recon breakdown (avg/perm): cgm=%.4fs basal=%.4fs "
            "meal_zero=%.4fs bolus_zero=%.4fs meal_cpl=%.4fs bolus_cpl=%.4fs",
            prof["recon_cgm"] / per,
            prof["recon_basal"] / per,
            prof["recon_meal_zero"] / per,
            prof["recon_bolus_zero"] / per,
            prof["recon_meal_coupling"] / per,
            prof["recon_bolus_coupling"] / per,
        )

    if rmse_count > 0:
        rmse = float(np.sqrt(rmse_sse / rmse_count))
        logging.info("RMSE (selected horizons): %.6f", rmse)
    else:
        logging.info("RMSE (selected horizons): N/A (no valid predictions)")
    logging.info("Done. Wrote outputs to %s", out_dir)


if __name__ == "__main__":
    main()
