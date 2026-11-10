#!/usr/bin/env python3
"""Retrain the arm-2 forecasters at a 4-hour prediction horizon (pred_len=48)
using each model's PREVIOUS best hyperparameters. No architecture search.

Configs are taken from each run directory's own saved config, which is the one
that actually produced the checkpoint (configs/ has drifted from it).
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml

MODELS = ["arimax","dlinear","gru","itransformer","patchmlp","patchmlp_glimmer","attengluco"]
OLD_RUN = "action_cf_controliq_best_{m}"
CPU_MODELS = {"arimax"}
LONG_MODELS = {"patchmlp_glimmer"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred-len", type=int, default=48)
    p.add_argument("--train-split", default="action_cf_benchmark_controliq")
    p.add_argument("--cfg-out", type=Path, default=Path("configs/counterfactual_4h"))
    p.add_argument("--run-prefix", default="cf4h_")
    p.add_argument("--models", default=",".join(MODELS))
    p.add_argument("--submit", action="store_true")
    a = p.parse_args()

    (a.cfg_out / "train").mkdir(parents=True, exist_ok=True)
    (a.cfg_out / "eval").mkdir(parents=True, exist_ok=True)
    made = []
    for m in [x.strip() for x in a.models.split(",") if x.strip()]:
        src = Path("outputs/results") / OLD_RUN.format(m=m) / "config.yaml"
        if not src.exists():
            print(f"  [skip] {m}: no run config"); continue
        c = yaml.safe_load(src.read_text())
        if "train" not in c or "eval" not in c:
            print(f"  [skip] {m}: config has no train/eval"); continue
        for kind in ("train","eval"):
            cfg = c[kind]
            dl = cfg.setdefault("model_data", {}).setdefault("dataloader", {})
            dl["pred_len"] = a.pred_len
            dl["stride"] = a.pred_len
            cfg.setdefault("split", {})["split_name"] = a.train_split
            # Glimmer: keep the GA search OFF (weights already selected previously)
            ga = cfg.get("training", {}).get("ga_search")
            if isinstance(ga, dict):
                ga["enabled"] = False
            (a.cfg_out / kind / f"{m}.yaml").write_text(yaml.safe_dump(cfg))
        made.append(m)
        print(f"  [cfg] {m}: pred_len={a.pred_len} stride={a.pred_len} split={a.train_split}")

    if not a.submit:
        print(f"\n[dry-run] {len(made)} configs written to {a.cfg_out}"); return

    for m in made:
        run_id = f"{a.run_prefix}{m}"
        if m in CPU_MODELS:
            res = ["--partition=standard","--time=12:00:00","--mem=64G","--cpus-per-task=8"]
        elif m in LONG_MODELS:
            res = ["--partition=standard","--time=24:00:00","--mem=128G","--cpus-per-task=8"]
        else:
            res = ["--partition=gpu","--time=12:00:00","--mem=64G","--cpus-per-task=8","--gres=gpu:1"]
        cmd = ["sbatch","--parsable","--job-name=cf4htrain","-A","shakeri-lab",
               "--output=logs/%x-%j.out","--error=logs/%x-%j.err", *res,
               "--wrap",
               ("cd /sfs/gpfs/tardis/project/shakeri-lab/Alireza_timeseries/benchmark && "
                "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1 && "
                # TRAIN ONLY. The eval phase would score 48-step predictions against the
                # 2h dataset's 24-step test suffixes and abort before the checkpoint
                # is written; inference runs later against the 4h dataset.
                ".conda_envs/bg-benchmark/bin/python exp/run_counterfactual_experiment.py "
                f"--train --train-config {a.cfg_out}/train/{m}.yaml "
                f"--eval-config {a.cfg_out}/eval/{m}.yaml "
                f"--output-root outputs/results --run-id {run_id}")]
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(f"  [submit] {m}: job={r.stdout.strip() or r.stderr.strip()[:60]}")


if __name__ == "__main__":
    main()
