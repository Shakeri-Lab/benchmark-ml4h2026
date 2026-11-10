from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

# Ensure repo root imports
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from data import make_dataloader_from_cfg  # noqa: E402
from metrics.clinical import MetricsAggregator  # noqa: E402

# EM models live in models/src
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "models"))
from src.models.mar import MAR  # type: ignore  # noqa: E402


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _unwrap_batch(batch):
    X, Y, pid, sid, ctx_end, evt_tuple, meta = batch
    return X, Y, pid, sid, ctx_end, evt_tuple, meta


def _reorder_target_last(x: torch.Tensor, target_idx: int = 0) -> torch.Tensor:
    if target_idx == x.shape[-1] - 1:
        return x
    idxs = [i for i in range(x.shape[-1]) if i != target_idx] + [target_idx]
    return x[..., idxs]


def _collect_train_tensors(dl_train, device: torch.device, target_idx: int = 0):
    feats = []
    targets = []
    for batch in dl_train:
        X, *_ = _unwrap_batch(batch)
        X = X.to(device)
        X = _reorder_target_last(X, target_idx=target_idx)
        feats.append(X)
        targets.append(X[..., -1])
    X_all = torch.cat(feats, dim=0) if feats else torch.empty(0, device=device)
    y_all = torch.cat(targets, dim=0) if targets else torch.empty(0, device=device)
    return {"features": X_all, "targets": y_all}


def _predict_batch(model: MAR, X: torch.Tensor, pred_len: int) -> torch.Tensor:
    if pred_len <= 0:
        raise ValueError("pred_len must be >= 1")
    if pred_len == 1:
        preds = model.predict(X, horizon=1)
        if preds.dim() == 2:
            preds = preds[:, -1:]
        return preds
    return model.predict(X, horizon=pred_len)


def _run_eval(model: MAR, dl_eval, pred_len: int):
    agg = MetricsAggregator()
    preds_all = []
    gts_all = []
    pat_ids = []
    seq_ids = []
    evt_post_meal = []
    evt_post_bolus = []
    evt_nocturnal = []

    for batch in dl_eval:
        X, Y, pid, sid, ctx_end, evt_tuple, meta = _unwrap_batch(batch)
        X = _reorder_target_last(X, target_idx=0)
        preds = _predict_batch(model, X, pred_len)
        preds_np = preds.detach().cpu().numpy()
        Y_np = Y.detach().cpu().numpy()
        agg.update(Y_np, preds_np, subject_id=pid.numpy())

        preds_all.append(preds_np)
        gts_all.append(Y_np)
        pat_ids.append(pid.numpy())
        seq_ids.append(sid.numpy())
        pm, pb, noct = evt_tuple
        evt_post_meal.append(pm.numpy())
        evt_post_bolus.append(pb.numpy())
        evt_nocturnal.append(noct.numpy())

    metrics = agg.compute()
    return metrics, {
        "preds": np.concatenate(preds_all, axis=0) if preds_all else np.empty((0, pred_len)),
        "gts": np.concatenate(gts_all, axis=0) if gts_all else np.empty((0, pred_len)),
        "pat_id": np.concatenate(pat_ids, axis=0) if pat_ids else np.empty((0,)),
        "seq_id": np.concatenate(seq_ids, axis=0) if seq_ids else np.empty((0,)),
        "event_post_meal": np.concatenate(evt_post_meal, axis=0) if evt_post_meal else np.empty((0,)),
        "event_post_bolus": np.concatenate(evt_post_bolus, axis=0) if evt_post_bolus else np.empty((0,)),
        "event_nocturnal": np.concatenate(evt_nocturnal, axis=0) if evt_nocturnal else np.empty((0,)),
    }


def main():
    parser = argparse.ArgumentParser(description="Train/test EM (MAR) models.")
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/results"))
    args = parser.parse_args()

    train_cfg = yaml.safe_load(args.train_config.read_text())
    eval_cfg = yaml.safe_load(args.eval_config.read_text())

    run_dir = args.output_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "logs.log")],
    )
    logging.info("Run ID: %s", args.run_id)

    split_cfg = dict(train_cfg.get("split") or {})
    model_data = dict(train_cfg.get("model_data") or {})
    model_cfg = dict(model_data.get("model") or {})
    dl_cfg = dict(model_data.get("dataloader") or {})
    train_part = split_cfg.get("train_part", "train")
    val_part = split_cfg.get("val_part", "val")

    history_len = int(dl_cfg.get("history_len", 48))
    pred_len = int(dl_cfg.get("pred_len", 12))

    dl_train = make_dataloader_from_cfg(split_cfg, dl_cfg, split_part=str(train_part), seed=model_data.get("seed", 42))
    dl_val = make_dataloader_from_cfg(split_cfg, {**dl_cfg, "shuffle": False}, split_part=str(val_part), seed=model_data.get("seed", 42))

    params = dict(model_cfg.get("params") or {})
    k = int(params.get("k", 3))
    p = int(params.get("p", 3))
    init_sigma = params.get("init_sigma", None)
    batch_size = int(params.get("batch_size", 128))
    tol = float(params.get("tol", 1e-5))
    max_iter = int(params.get("max_iter", 100))
    seed = int(model_data.get("seed", 42))
    univariate = bool(params.get("univariate", False))

    n_features = 1 if univariate else len(dl_train.dataset.feature_cols)
    # Build training tensors first so we can compute a data-driven init_sigma if needed.
    dummy_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = _collect_train_tensors(dl_train, device=dummy_device, target_idx=0)
    if init_sigma is None:
        if train_dataset["targets"].numel() == 0:
            raise ValueError("No training targets available to compute init_sigma.")
        init_sigma = float(train_dataset["targets"].std().item())

    model = MAR(n_features=n_features, p=p, k=k, random_seed=seed, init_sigma=init_sigma)
    device = model.device
    # Ensure tensors are on the model device
    train_dataset = {
        "features": train_dataset["features"].to(device),
        "targets": train_dataset["targets"].to(device),
    }
    model.train(train_dataset, batch_size=batch_size, tol=tol, max_iter=max_iter)

    # Evaluate on test split
    eval_split = dict(eval_cfg.get("split") or {})
    eval_model_data = dict(eval_cfg.get("model_data") or {})
    eval_dl_cfg = dict(eval_model_data.get("dataloader") or {})
    split_part = eval_split.get("split_part", "test")

    dl_eval = make_dataloader_from_cfg(eval_split, {**eval_dl_cfg, "shuffle": False}, split_part=str(split_part), seed=eval_model_data.get("seed", seed))
    metrics, artifacts = _run_eval(model, dl_eval, pred_len)

    # Save results
    jsonable = _to_jsonable({"train_config": train_cfg, "eval_config": eval_cfg, "metrics": metrics})
    (run_dir / "metrics.json").write_text(json.dumps(jsonable, indent=2))
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
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump({"train": train_cfg, "eval": eval_cfg, "model": "mar"}, f)

    logging.info("Finished EM evaluation.")


if __name__ == "__main__":
    main()
