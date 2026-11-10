"""Isolated frozen-p12 inference. Run from the benchmark repository on HPC.

Dense predictions, patient identifiers and split specifications stay on HPC.
This script never fits models and never changes submitted source/artifacts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from analysis.evaluate_hypo_gating import _load_sequences, _load_split_spec, _sanitize_state_dict
from data import make_dataloader_from_cfg
from models import build_model_from_cfg


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def init_worker(model_cfg, input_dim, history, horizon, checkpoint):
    global WORKER_MODEL, WORKER_HORIZON
    torch.set_num_threads(1)
    WORKER_MODEL = build_model_from_cfg(model_cfg, input_dim=input_dim,
                                       history_len=history, pred_len=horizon)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    WORKER_MODEL.load_state_dict(_sanitize_state_dict(ckpt["state_dict"]), strict=True)
    WORKER_MODEL.eval()
    WORKER_HORIZON = horizon


def predict_worker(x):
    with torch.inference_mode():
        pred = WORKER_MODEL(torch.from_numpy(x), WORKER_HORIZON).numpy()
    if not np.isfinite(pred).all():
        raise RuntimeError("Nonfinite ARIMAX trajectory; do not silently drop windows.")
    return np.min(pred, axis=1)


def infer(args, part):
    start_time = time.monotonic()
    run_dir = Path(args.run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    ec = cfg.get("eval", cfg)
    split = ec["split"]
    md = ec["model_data"]
    dc = dict(md["dataloader"], stride=1, shuffle=False, num_workers=0)
    horizon = int(dc["pred_len"])
    if horizon != 12:
        raise ValueError("This preregistered analysis uses frozen 60-minute models only.")
    spec = _load_split_spec(Path(split.get("split_root", "configs/preprocessing")), split["split_name"])
    model_name = md["model"]["name"]
    source = Path(split.get("processed_root", "datasets_processed")) / spec["dataset"] / "timeseries.csv"
    out = args.out / part
    out.mkdir(parents=True, exist_ok=True)
    provenance = dict(run_dir=str(run_dir), checkpoint_sha256=sha(run_dir / "checkpoint.pt"),
                      config_sha256=sha(run_dir / "config.yaml"), source_sha256=sha(source),
                      split_sha256=hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
                      split_part=part, split_patients=spec["patients"][part], model=model_name,
                      dataset=spec["dataset"], horizon_min=60, min_context_steps=288,
                      inference_stride=1, strict_checkpoint_loading=model_name != "zoh",
                      script_sha256=sha(__file__))
    if (out / "dense_min_predictions.npz").exists():
        if json.loads((out / "provenance.json").read_text()) != provenance:
            raise RuntimeError("Existing cache has different provenance; never overwrite it.")
        print(f"Verified existing {model_name}/{spec['dataset']}/{part} cache", flush=True)
        return
    seqs = _load_sequences(source.parent.parent, spec["dataset"], spec, part)
    torch.manual_seed(int(md.get("seed", 42)))
    torch.set_num_threads(args.threads)
    dl = make_dataloader_from_cfg(split, dc, split_part=part, seed=md.get("seed", 42))
    model = build_model_from_cfg(md["model"], input_dim=len(dl.dataset.feature_cols),
                                history_len=dc["history_len"], pred_len=horizon)
    ckpt = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    state = _sanitize_state_dict(ckpt["state_dict"])
    if isinstance(model, torch.nn.Module):
        model.load_state_dict(state, strict=True)
    elif model_name != "zoh" or state:
        raise RuntimeError("Only parameter-free ZOH can omit state loading.")
    if args.workers > 1 and model_name != "arimax":
        raise ValueError("CPU process workers are for ARIMAX only.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but unavailable; do not fall back silently.")
    device = torch.device(args.device)
    if isinstance(model, torch.nn.Module):
        model = model.to(device).eval()
    executor = None
    if args.workers > 1:
        executor = ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"), initializer=init_worker,
            initargs=(md["model"], len(dl.dataset.feature_cols), dc["history_len"],
                      horizon, str(run_dir / "checkpoint.pt")))
    chunks = {k: [] for k in ("pid", "sid", "idx", "min_pred")}
    with torch.inference_mode():
        for bi, batch in enumerate(dl):
            x, _, pids, sids, dates, *_ = batch
            if executor is not None:
                parts = np.array_split(x.numpy(), min(args.workers, len(x)))
                mins = np.concatenate(list(executor.map(predict_worker, parts)))
                if bi == 0:
                    reference = np.min(model(x, horizon).numpy(), axis=1)
                    if not np.array_equal(reference, mins):
                        raise RuntimeError("Serial and parallel ARIMAX differ.")
                    (out / "parallel_check.json").write_text(json.dumps(
                        dict(windows=len(x), max_abs_error=0.0, workers=args.workers), indent=2))
            else:
                pred = (model(x.to(device), horizon).detach().cpu().numpy()
                        if isinstance(model, torch.nn.Module)
                        else model.predict(x.numpy(), horizon))
                if not np.isfinite(pred).all():
                    raise RuntimeError("Nonfinite predictions; do not silently omit windows.")
                mins = np.min(pred, axis=1)
            for i, (p, s) in enumerate(zip(pids, sids)):
                key = (int(p), int(s))
                idx = seqs[key]["date_to_idx"].get(str(dates[i]))
                if idx is None:
                    raise RuntimeError("Prediction origin is absent from source timeline.")
                if idx < 287:
                    continue
                for k, value in zip(chunks, (key[0], key[1], idx, mins[i])):
                    chunks[k].append(value)
            if bi % 500 == 0:
                print(f"{model_name}/{spec['dataset']}/{part}: batch {bi}/{len(dl)}, "
                      f"elapsed {time.monotonic()-start_time:.0f}s", flush=True)
    if executor is not None:
        executor.shutdown(wait=True)
    arr = {k: np.asarray(v, dtype=np.float64 if k == "min_pred" else np.int64)
           for k, v in chunks.items()}
    if not len(arr["idx"]):
        raise RuntimeError("No dense predictions survived the common context filter.")
    np.savez_compressed(out / "dense_min_predictions.npz", **arr)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (out / "completion.json").write_text(json.dumps(dict(
        n_origins=len(arr["idx"]), elapsed_seconds=time.monotonic()-start_time,
        min_prediction=float(arr["min_pred"].min()), max_prediction=float(arr["min_pred"].max()),
        device=str(device), feature_columns=list(dl.dataset.feature_cols)), indent=2))
    print(f"DONE {model_name}/{spec['dataset']}/{part}, {len(arr['idx'])} origins", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    for part in ("val", "test"):
        infer(args, part)
