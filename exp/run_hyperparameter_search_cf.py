from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import yaml

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from data import make_cf_dataloader_from_cfg  # noqa: E402
from models import build_model_from_cfg  # noqa: E402
from models.losses import get_loss  # noqa: E402
from models.optim import build_optim_and_sched  # noqa: E402

import torch
from tqdm.auto import tqdm


def _ensure_same_shape(preds, targets, stage: str):
    if preds.shape != targets.shape:
        raise ValueError(f"{stage}: preds shape {preds.shape} does not match targets shape {targets.shape}.")


def _get_xy(batch):
    # Counterfactual dataloader returns extra fields. We only need X, Y.
    return batch[0], batch[1]


def _eval_val_rmse(model, dl_val, pred_len: int):
    was_training = bool(getattr(model, "training", False))
    model.eval()
    mse_vals = []
    with torch.no_grad():
        for batch in dl_val:
            Xv, Yv = _get_xy(batch)
            Xv = Xv.to(next(model.parameters()).device) if isinstance(model, torch.nn.Module) else Xv
            Yv = Yv.to(Xv.device)
            preds = model(Xv, pred_len) if pred_len is not None else model(Xv)
            _ensure_same_shape(preds, Yv, stage="val")
            mse_vals.append(torch.mean((preds - Yv) ** 2).item())
    if was_training:
        model.train()
    mse = float(np.mean(mse_vals)) if mse_vals else float("inf")
    return float(np.sqrt(mse))


def _train_one(cfg: Dict[str, Any], out_dir: Path) -> float:
    split_cfg = dict(cfg.get("split") or {})
    training_cfg = dict(cfg.get("training") or {})
    model_data_cfg = dict(cfg.get("model_data") or {})
    dl_cfg = dict(model_data_cfg.get("dataloader") or {})

    train_part = split_cfg.get("train_part", "train")
    val_part = split_cfg.get("val_part", "val")

    pred_len = int(dl_cfg.get("pred_len", 6))
    history_len = int(dl_cfg.get("history_len", 48))

    dl_train = make_cf_dataloader_from_cfg(
        split_cfg,
        dl_cfg,
        split_part=str(train_part),
        seed=model_data_cfg.get("seed", 42),
    )
    dl_val = make_cf_dataloader_from_cfg(
        split_cfg,
        {**dl_cfg, "shuffle": False},
        split_part=str(val_part),
        seed=model_data_cfg.get("seed", 42),
    )

    input_dim = len(dl_train.dataset.feature_cols)
    model_cfg = dict(model_data_cfg.get("model") or {})
    model = build_model_from_cfg(model_cfg, input_dim=input_dim, history_len=history_len, pred_len=pred_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(model, torch.nn.Module):
        model = model.to(device)
        model.train()

    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    trainable = getattr(base_model, "trainable", True)
    has_trainable = trainable and any(p.requires_grad for p in model.parameters()) if isinstance(model, torch.nn.Module) else False

    if hasattr(base_model, "prefit"):
        base_model.prefit(dl_train, dl_val)

    if has_trainable:
        loss_name = str(training_cfg.get("loss", "mse"))
        loss_params = dict(training_cfg.get("loss_params") or {})
        criterion = get_loss(loss_name, **loss_params)
        optimizer, scheduler = build_optim_and_sched(model, training_cfg)

        epochs = int(training_cfg.get("epochs", 1))
        use_amp = bool(training_cfg.get("amp", False))
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        grad_clip_norm = training_cfg.get("grad_clip_norm", None)
        if grad_clip_norm in ("null", "None"):
            grad_clip_norm = None

        best_val = float("inf")
        progress_path = out_dir / "val_rmse.json"
        for ep in range(epochs):
            pbar = tqdm(dl_train, desc=f"Epoch {ep+1}/{epochs}", leave=False)
            for batch in pbar:
                X, Y = _get_xy(batch)
                X = X.to(device)
                Y = Y.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    preds = model(X, pred_len) if pred_len is not None else model(X)
                    _ensure_same_shape(preds, Y, stage="train")
                    loss = criterion(preds, Y)
                if loss.requires_grad:
                    scaler.scale(loss).backward()
                    if grad_clip_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                pbar.set_postfix({"loss": f"{float(loss.item()):.4f}"})

            val_rmse = _eval_val_rmse(model, dl_val, pred_len)
            best_val = min(best_val, val_rmse)
            progress_path.write_text(json.dumps({"val_rmse": best_val, "last_epoch": ep + 1}, indent=2))

            if scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_rmse)
                else:
                    scheduler.step()
        return best_val

    # Non-trainable: just evaluate
    return _eval_val_rmse(model, dl_val, pred_len)


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search runner for action_cf_benchmark.")
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/hyperparameters"))
    parser.add_argument("--arch", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--fail-if-exists", action="store_true")
    parser.add_argument("--shard", type=int, default=None,
                        help="0-based shard index; run only configs where i %% nshards == shard.")
    parser.add_argument("--nshards", type=int, default=None,
                        help="Total number of shards (for Slurm job arrays).")
    args = parser.parse_args()

    cfg_dir = args.config_dir
    if not cfg_dir.exists():
        raise FileNotFoundError(cfg_dir)

    out_root = args.output_root / f"{args.arch}_{args.dataset}"
    if args.fail_if_exists and out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing hyperparameter output directory: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)

    configs = sorted(cfg_dir.glob("*.yaml"))
    if not configs:
        raise FileNotFoundError(f"No configs found in {cfg_dir}")

    if args.shard is not None and args.nshards:
        total = len(configs)
        configs = [c for i, c in enumerate(configs) if i % int(args.nshards) == int(args.shard)]
        logging.info("shard %s/%s: running %s of %s configs",
                     args.shard, args.nshards, len(configs), total)
        if not configs:
            logging.info("nothing for this shard; exiting")
            return

    n_skipped = 0
    for cfg_path in configs:
        cfg = yaml.safe_load(cfg_path.read_text())
        run_id = cfg_path.stem
        run_dir = out_root / run_id
        if args.fail_if_exists and run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing hyperparameter run directory: {run_dir}")
        # Resume: a config already scored (or already recorded as failed) is not
        # redone. Without this, refilling a partially-completed sweep repeats
        # every finished config and burns the compute again.
        if (run_dir / "val_rmse.json").exists() or (run_dir / "config_failed.json").exists():
            n_skipped += 1
            continue
        run_dir.mkdir(parents=True, exist_ok=True)

        # setup per-run logging
        root = logging.getLogger()
        for h in list(root.handlers):
            root.removeHandler(h)
        root.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh = logging.FileHandler(run_dir / "logs.txt")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(fh)
        root.addHandler(sh)
        logging.info("Running %s", cfg_path.name)

        try:
            val_rmse = _train_one(cfg, run_dir)
        except Exception as exc:  # one bad parameterisation must not kill the sweep
            logging.exception("Config %s failed; recording and continuing", cfg_path.name)
            (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
            (run_dir / "config_failed.json").write_text(json.dumps(
                {"error": f"{type(exc).__name__}: {exc}"}, indent=2))
            continue

        # save config + metrics
        (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
        (run_dir / "val_rmse.json").write_text(json.dumps({"val_rmse": val_rmse}, indent=2))
        logging.info("Finished %s: val_rmse=%.6f", cfg_path.name, val_rmse)

    if n_skipped:
        logging.info("skipped %s already-completed configs", n_skipped)


if __name__ == "__main__":
    main()
