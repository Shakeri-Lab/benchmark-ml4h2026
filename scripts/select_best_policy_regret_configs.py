#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict

import yaml


SOURCE_DATASET = "action_cf_benchmark"
TARGET_DATASET = "action_cf_benchmark_bolus"
EVAL_METRICS = ["rmse", "mae", "seg", "gmse", "grmse", "ceg_ratio", "peg_ratio", "seg_z_ratio"]
DEFAULT_EVENT_THRESHOLDS = {"post_meal": 3, "post_bolus": 3, "nocturnal": 0}
GLIMMER_DEFAULTS = {
    "loss": "glimmer_weighted",
    "loss_params": {
        "w_hypo": 2.0,
        "w_hyper": 2.0,
        "hypo_threshold": 70.0,
        "hyper_threshold": 180.0,
    },
    "ga_search": {
        "enabled": True,
        "fitness": "rmse",
        "population_size": 12,
        "generations": 12,
        "w_hypo_range": [1.0, 10.0],
        "w_hyper_range": [1.0, 10.0],
        "mutation_rate": 1.0,
        "mutation_scale": 0.70,
        "selection": "uniform",
        "elite_fraction": 0.33,
        "inner_epochs": 4,
        "seed": 42,
    },
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _load_val_rmse(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    val = data.get("val_rmse")
    return None if val is None else float(val)


def _best_run(pair_dir: Path, *, fallback_cfg_root: Path) -> tuple[Path, Dict[str, Any], float] | None:
    best: tuple[Path, Dict[str, Any], float] | None = None
    for run_dir in sorted(pair_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        rmse = _load_val_rmse(run_dir / "val_rmse.json")
        cfg_path = run_dir / "config.yaml"
        if not cfg_path.exists():
            alt_cfg = fallback_cfg_root / pair_dir.name / f"{run_dir.name}.yaml"
            if alt_cfg.exists():
                cfg_path = alt_cfg
        if rmse is None or not cfg_path.exists():
            continue
        cfg = _load_yaml(cfg_path)
        if best is None or rmse < best[2]:
            best = (run_dir, cfg, rmse)
    return best


def _set_split_to_target(cfg: Dict[str, Any], *, target_dataset: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    split = dict(cfg.get("split") or {})
    split["split_name"] = target_dataset
    split["processed_root"] = split.get("processed_root", "datasets_processed")
    split.setdefault("train_part", "train")
    split.setdefault("val_part", "val")
    cfg["split"] = split
    return cfg


def _make_eval_cfg(
    train_cfg: Dict[str, Any],
    *,
    model_name: str,
    source_run_id: str,
    source_val_rmse: float,
    target_dataset: str,
) -> Dict[str, Any]:
    split = dict(train_cfg.get("split") or {})
    split.pop("train_part", None)
    split.pop("val_part", None)
    split["split_part"] = "test"

    model_data = copy.deepcopy(train_cfg.get("model_data") or {})
    dl_cfg = dict(model_data.get("dataloader") or {})
    dl_cfg["shuffle"] = False
    model_data["dataloader"] = dl_cfg

    return {
        "split": split,
        "model_data": model_data,
        "evaluation": {
            "metrics": list(EVAL_METRICS),
            "event_thresholds": dict(DEFAULT_EVENT_THRESHOLDS),
            "compile": False,
        },
        "notes": (
            f"Policy-regret eval config for {model_name} on {target_dataset}; "
            f"selected from {source_run_id} with val_rmse={source_val_rmse:.6f}"
        ),
    }


def _make_glimmer_cfg(base_train_cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_train_cfg)
    model_block = dict(cfg.get("model_data", {}).get("model") or {})
    model_block["name"] = "patchmlp_glimmer"
    cfg["model_data"]["model"] = model_block

    training = dict(cfg.get("training") or {})
    training["loss"] = GLIMMER_DEFAULTS["loss"]
    training["loss_params"] = copy.deepcopy(GLIMMER_DEFAULTS["loss_params"])
    training["ga_search"] = copy.deepcopy(GLIMMER_DEFAULTS["ga_search"])
    cfg["training"] = training

    notes = str(cfg.get("notes", "")).strip()
    suffix = "using best PatchMLP hyperparameters and default GLIMMER GA settings"
    cfg["notes"] = f"{notes}; {suffix}" if notes else suffix
    return cfg


def _write_yaml(path: Path, data: Dict[str, Any], *, fail_if_exists: bool) -> None:
    if fail_if_exists and path.exists():
        raise FileExistsError(f"Refusing to overwrite existing config: {path}")
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select best action_cf hyperparameters and materialize policy-regret configs for the bolus CF benchmark."
    )
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs/hyperparameters"))
    parser.add_argument("--configs-root", type=Path, default=Path("configs/policy_regret_best"))
    parser.add_argument("--hyperparam-config-root", type=Path, default=Path("configs/hyperparameter"))
    parser.add_argument("--source-dataset", type=str, default=SOURCE_DATASET)
    parser.add_argument("--target-dataset", type=str, default=TARGET_DATASET)
    parser.add_argument("--fail-if-exists", action="store_true")
    args = parser.parse_args()

    train_root = args.configs_root / "train"
    eval_root = args.configs_root / "eval"
    train_root.mkdir(parents=True, exist_ok=True)
    eval_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {"source_dataset": args.source_dataset, "target_dataset": args.target_dataset, "models": {}}
    patchmlp_train_cfg: Dict[str, Any] | None = None
    patchmlp_meta: Dict[str, Any] | None = None

    suffix = f"_{args.source_dataset}"
    for pair_dir in sorted(args.outputs_root.iterdir()):
        if not pair_dir.is_dir() or not pair_dir.name.endswith(suffix):
            continue
        model_name = pair_dir.name[: -len(suffix)]
        best = _best_run(pair_dir, fallback_cfg_root=args.hyperparam_config_root)
        if best is None:
            continue
        run_dir, cfg, rmse = best
        train_cfg = _set_split_to_target(cfg, target_dataset=args.target_dataset)
        notes = str(train_cfg.get("notes", "")).strip()
        train_cfg["notes"] = (
            f"{notes}; policy-regret train config selected from {run_dir.name} with val_rmse={rmse:.6f}"
            if notes
            else f"Policy-regret train config selected from {run_dir.name} with val_rmse={rmse:.6f}"
        )
        eval_cfg = _make_eval_cfg(
            train_cfg,
            model_name=model_name,
            source_run_id=run_dir.name,
            source_val_rmse=rmse,
            target_dataset=args.target_dataset,
        )

        train_path = train_root / f"{model_name}_{args.target_dataset}.yaml"
        eval_path = eval_root / f"{model_name}_{args.target_dataset}.yaml"
        _write_yaml(train_path, train_cfg, fail_if_exists=args.fail_if_exists)
        _write_yaml(eval_path, eval_cfg, fail_if_exists=args.fail_if_exists)

        summary["models"][model_name] = {
            "source_pair_dir": pair_dir.name,
            "source_run_id": run_dir.name,
            "val_rmse": rmse,
            "train_config": str(train_path),
            "eval_config": str(eval_path),
        }

        if model_name == "patchmlp":
            patchmlp_train_cfg = train_cfg
            patchmlp_meta = {"source_run_id": run_dir.name, "val_rmse": rmse}

    if patchmlp_train_cfg is not None and patchmlp_meta is not None:
        glimmer_train = _make_glimmer_cfg(patchmlp_train_cfg)
        glimmer_eval = _make_eval_cfg(
            glimmer_train,
            model_name="patchmlp_glimmer",
            source_run_id=patchmlp_meta["source_run_id"],
            source_val_rmse=patchmlp_meta["val_rmse"],
            target_dataset=args.target_dataset,
        )
        glimmer_train_path = train_root / f"patchmlp_glimmer_{args.target_dataset}.yaml"
        glimmer_eval_path = eval_root / f"patchmlp_glimmer_{args.target_dataset}.yaml"
        _write_yaml(glimmer_train_path, glimmer_train, fail_if_exists=args.fail_if_exists)
        _write_yaml(glimmer_eval_path, glimmer_eval, fail_if_exists=args.fail_if_exists)
        summary["models"]["patchmlp_glimmer"] = {
            "source_pair_dir": f"patchmlp_{args.source_dataset}",
            "source_run_id": patchmlp_meta["source_run_id"],
            "val_rmse": patchmlp_meta["val_rmse"],
            "train_config": str(glimmer_train_path),
            "eval_config": str(glimmer_eval_path),
            "base_architecture": "patchmlp",
            "ga_defaults": copy.deepcopy(GLIMMER_DEFAULTS["ga_search"]),
        }

    summary_path = args.configs_root / "summary.json"
    if args.fail_if_exists and summary_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing summary: {summary_path}")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote policy-regret configs to {args.configs_root}")


if __name__ == "__main__":
    main()
