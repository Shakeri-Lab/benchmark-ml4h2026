#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import yaml


DATASETS = ["dclp3", "dclp5", "pedap", "shanghait1dm", "iobp2"]
ARCHES = ["arimax", "dlinear", "gru", "itransformer", "patchmlp", "attengluco", "patchmlp_glimmer", "zoh"]
LAGS = {3: "15min", 6: "30min", 9: "45min", 12: "60min"}


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _set(cfg: Dict[str, Any], path: list[str], value: Any) -> None:
    cur = cfg
    for k in path[:-1]:
        cur = cur.setdefault(k, {})
    cur[path[-1]] = value


def _best_cfg_path(best_root: Path, arch: str, dataset: str) -> Path | None:
    cand = best_root / f"{arch}_{dataset}.yaml"
    if cand.exists():
        return cand
    return None


def _fallback_base_config(arch: str, dataset: str) -> Dict[str, Any]:
    # Fallbacks for zoh or missing hyperparameter best
    # Default to 1-hour multivariate configs if present.
    name = f"{dataset}_{arch}_mv.yaml"
    path = Path("configs/training") / name
    if path.exists():
        return _load_yaml(path)
    if arch == "zoh":
        return {
            "split": {
                "split_name": f"{dataset}_patients_70_10_20_seed42",
                "split_root": "configs/preprocessing",
                "processed_root": "datasets_processed",
                "train_part": "train",
                "val_part": "val",
            },
            "training": {
                "epochs": 1,
                "loss": "mse",
                "amp": False,
                "grad_clip_norm": None,
                "compile": False,
                "early_stopping": {"patience": 1, "min_delta": 0.0},
                "optimizer": {"name": "adam", "lr": 1e-4, "weight_decay": 0.0},
                "scheduler": {"name": "reduce_on_plateau", "patience": 2, "min_lr": 1.0e-6},
            },
            "model_data": {
                "seed": 42,
                "model": {"name": "zoh", "params": {"univariate": False}},
                "dataloader": {
                    "history_len": 48,
                    "pred_len": 12,
                    "stride": 12,
                    "batch_size": 128,
                    "univariate": False,
                    "num_workers": 0,
                    "shuffle": True,
                    "nocturnal_start_hour": 0,
                    "nocturnal_end_hour": 6,
                },
            },
            "notes": f"Fallback ZOH config for {dataset}",
        }
    # Last resort: use dclp3 config for the arch
    path = Path("configs/training") / f"dclp3_{arch}_mv.yaml"
    if path.exists():
        return _load_yaml(path)
    raise FileNotFoundError(f"No base config found for {arch} {dataset}")


def _make_patchmlp_glimmer(base_patchmlp: Dict[str, Any]) -> Dict[str, Any]:
    cfg = yaml.safe_load(yaml.safe_dump(base_patchmlp))
    _set(cfg, ["model_data", "model", "name"], "patchmlp_glimmer")
    # GLIMMER-style loss: weighted MAE + GA defaults from upstream
    _set(cfg, ["training", "loss"], "glimmer_weighted")
    _set(cfg, ["training", "loss_params"], {
        "w_hypo": 3.296363582,
        "w_hyper": 2.382397706,
        "hypo_threshold": 70.0,
        "hyper_threshold": 180.0,
        "mode": "mae",
    })
    # GA search is NOT re-run: the weights below are the values it selected
    # previously (recovered from those logs). Re-enabling it costs 25 generations
    # x 20 population x 30 inner epochs per run -- days, not minutes -- and it
    # timed out the IOBP2 runs at 12h.
    _set(cfg, ["training", "ga_search"], {
        "enabled": False,
        "fitness": "rmse",
        "population_size": 20,
        "generations": 25,
        "w_hypo_range": [1.0, 10.0],
        "w_hyper_range": [1.0, 10.0],
        "mutation_rate": 1.0,
        "mutation_scale": 0.0556,
        "selection": "uniform",
        "elite_fraction": 0.5,
        "inner_epochs": 30,
        "seed": 42,
    })
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Build best configs for all lags.")
    parser.add_argument("--best-root", type=Path, default=Path("configs/hyperparameter_best"))
    parser.add_argument("--out-root", type=Path, default=Path("configs/best_lag"))
    args = parser.parse_args()

    train_dir = args.out_root / "train"
    eval_dir = args.out_root / "eval"
    train_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    for dataset in DATASETS:
        for arch in ARCHES:
            base_cfg = None
            if arch != "patchmlp_glimmer":
                p = _best_cfg_path(args.best_root, arch, dataset)
                if p is not None:
                    base_cfg = _load_yaml(p)
            if base_cfg is None:
                base_cfg = _fallback_base_config(arch if arch != "patchmlp_glimmer" else "patchmlp", dataset)

            if arch == "patchmlp_glimmer":
                # Build from patchmlp best hyperparams
                patchmlp_best = _best_cfg_path(args.best_root, "patchmlp", dataset)
                if patchmlp_best is not None:
                    base_cfg = _load_yaml(patchmlp_best)
                base_cfg = _make_patchmlp_glimmer(base_cfg)

            for pred_len in LAGS.keys():
                cfg = yaml.safe_load(yaml.safe_dump(base_cfg))
                _set(cfg, ["model_data", "dataloader", "pred_len"], pred_len)
                _set(cfg, ["model_data", "dataloader", "stride"], pred_len)
                # annotate
                _set(cfg, ["notes"], f"Best {arch} on {dataset} ({pred_len} steps)")

                # Training config
                train_path = train_dir / f"{arch}_{dataset}_p{pred_len}.yaml"
                train_path.write_text(yaml.safe_dump(cfg))

                # Eval config
                eval_cfg = yaml.safe_load(yaml.safe_dump(cfg))
                _set(eval_cfg, ["split", "split_part"], "test")
                eval_path = eval_dir / f"{arch}_{dataset}_p{pred_len}.yaml"
                eval_path.write_text(yaml.safe_dump(eval_cfg))

    print(f"Wrote configs to {args.out_root}")


if __name__ == "__main__":
    main()
