from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import yaml


DATASETS = ("dclp3", "dclp5", "pedap", "azt1d")
PRIMARY_DATASETS = ("dclp3", "pedap")
MODEL = "patchmlp"
SEEDS = (42,)
LOSSES: dict[str, dict[str, Any]] = {
    "mse": {"loss": "mse", "loss_params": None},
    "threshold_crossing_kovatchev_clinical_loss": {
        "loss": "threshold_crossing_kovatchev_clinical_loss",
        "loss_params": {
            "fitted_loss_config": "configs/surrogate/fitted_threshold_crossing_kovatchev_default.yaml",
            "value_scale": "mg/dL",
            "assert_mgdl_scale": True,
        },
    },
    "wolff_weighted_loss": {
        "loss": "wolff_weighted_loss",
        "loss_params": {
            "alpha_l1": 0.0,
            "assert_mgdl_scale": True,
            "source": "Wolff supplementary formula; alpha_l1 not specified in this repo, set to 0.0 for primary comparator.",
        },
    },
    "glimmer_fixed_transformer_weights": {
        "loss": "glimmer_fixed_transformer_weights",
        "loss_params": {
            "w_hypo": 4.67,
            "w_normal": 1.0,
            "w_hyper": 1.71,
            "hypo_threshold": 70.0,
            "hyper_threshold": 180.0,
            "mode": "mae",
            "assert_mgdl_scale": True,
            "source": "Fixed GLIMMER Table-3 Transformer-average weights; no GA search.",
        },
    },
}


def _load_base(dataset: str) -> dict[str, Any]:
    path = Path("configs/surrogate") / f"{MODEL}_{dataset}.yaml"
    if not path.exists():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text())


def _scale_patchmlp(cfg: dict[str, Any], multiplier: float) -> None:
    params = cfg["model_data"]["model"].setdefault("params", {})
    if "d_model" not in params:
        raise KeyError("PatchMLP capacity scaling expects model.params.d_model")
    base = int(params["d_model"])
    params["d_model"] = max(8, int(round(base * float(multiplier))))


def _make_cfg(
    *,
    base: dict[str, Any],
    dataset: str,
    loss_key: str,
    seed: int,
    capacity: str | None = None,
    capacity_multiplier: float | None = None,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    if capacity_multiplier is not None:
        _scale_patchmlp(cfg, capacity_multiplier)
    cfg["model_data"]["seed"] = int(seed)
    training = cfg.setdefault("training", {})
    loss_spec = LOSSES[loss_key]
    training["loss"] = loss_spec["loss"]
    if loss_spec["loss_params"] is None:
        training.pop("loss_params", None)
    else:
        training["loss_params"] = copy.deepcopy(loss_spec["loss_params"])
    training.pop("ga_search", None)
    training["compile"] = False
    training.setdefault("grad_clip_norm", 1.0)

    cfg["mechanism_frontier"] = {
        "stage": "mechanism_frontier_tests",
        "dataset": dataset.upper(),
        "model_architecture": MODEL,
        "loss_key": loss_key,
        "seed": int(seed),
        "history_hours": 12,
        "prediction_minutes": 60,
        "decision_horizon_minutes": 30,
        "decision_horizon_index_zero_based": 5,
        "loss_space": "mg/dL" if loss_key != "mse" else "repo_mse_convention",
    }
    if capacity is not None:
        cfg["mechanism_frontier"]["capacity"] = capacity
        cfg["mechanism_frontier"]["capacity_multiplier"] = float(capacity_multiplier or 1.0)
    cfg["surrogate"] = {
        "stage": "mechanism_frontier_tests",
        "dataset": dataset.upper(),
        "model_architecture": MODEL,
    }
    cfg["notes"] = (
        f"Mechanism-frontier PatchMLP config for {dataset.upper()} with {loss_key}. "
        "Uses 12 h history, 1 h prediction, stride 12, and single-lag 30 min evaluation downstream."
    )
    return cfg


def build_loss_comparator_configs(out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for dataset in DATASETS:
        base = _load_base(dataset)
        for loss_key in LOSSES:
            for seed in SEEDS:
                cfg = _make_cfg(base=base, dataset=dataset, loss_key=loss_key, seed=seed)
                path = out_dir / f"train_{loss_key}_{MODEL}_{dataset}_seed{seed}.yaml"
                path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                paths.append(path)
    return paths


def build_capacity_pilot_configs(out_dir: Path, dataset: str = "dclp3") -> list[Path]:
    paths: list[Path] = []
    base = _load_base(dataset)
    multipliers = [1.0, 0.75, 0.5, 0.25, 0.125]
    for mult in multipliers:
        cap = f"cap_{str(mult).replace('.', 'p')}x"
        cfg = _make_cfg(base=base, dataset=dataset, loss_key="mse", seed=42, capacity=cap, capacity_multiplier=mult)
        path = out_dir / f"capacity_pilot_mse_{MODEL}_{dataset}_{cap}_seed42.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        paths.append(path)
    return paths


def build_capacity_sweep_configs(out_dir: Path, capacities: dict[str, float] | None = None) -> list[Path]:
    paths: list[Path] = []
    capacities = capacities or {"default": 1.0, "medium_small": 0.5, "small": 0.25}
    for dataset in PRIMARY_DATASETS:
        base = _load_base(dataset)
        for cap, mult in capacities.items():
            for loss_key in LOSSES:
                cfg = _make_cfg(base=base, dataset=dataset, loss_key=loss_key, seed=42, capacity=cap, capacity_multiplier=mult)
                path = out_dir / f"capacity_sweep_{loss_key}_{MODEL}_{dataset}_{cap}_seed42.yaml"
                path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                paths.append(path)
    return paths


def build_eval_configs(out_dir: Path) -> None:
    base = {
        "model": MODEL,
        "seed": 42,
        "datasets": list(DATASETS),
        "detection_mode": "trajectory",
        "horizon_minutes": 30,
        "threshold_sweep_min": 40,
        "threshold_sweep_max": 120,
        "threshold_sweep_step": 1,
        "fa_day_targets": [0.5, 1.0, 2.0, 3.0],
        "true_hypo_threshold": 70.0,
        "min_event_duration_minutes": 15,
        "step_minutes": 5,
        "use_refractory": True,
        "methods": [
            {
                "name": "mse",
                "label": "MSE",
                "root": "outputs/surrogate/mechanism_frontier_tests/dense_predictions/mse",
                "fallback_root": "outputs/surrogate/mse_pair_generation",
                "template": "{root}/{model}_{dataset}/seed_{seed}",
                "fallback_template": "{root}/{model}_{dataset}",
            },
            *[
                {
                    "name": loss,
                    "label": label,
                    "root": f"outputs/surrogate/mechanism_frontier_tests/dense_predictions/{loss}",
                    "template": "{root}/{model}_{dataset}/seed_{seed}",
                }
                for loss, label in [
                    ("threshold_crossing_kovatchev_clinical_loss", "Threshold-crossing Kovatchev"),
                    ("wolff_weighted_loss", "Wolff weighted"),
                    ("glimmer_fixed_transformer_weights", "GLIMMER fixed"),
                ]
            ],
        ],
        "output_dir": "outputs/surrogate/mechanism_frontier_tests/decision_frontier",
    }
    (out_dir / "eval_single_lag_frontier.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
    point = dict(base)
    point.update(
        {
            "detection_mode": "point",
            "use_refractory": False,
            "output_dir": "outputs/surrogate/mechanism_frontier_tests/decision_frontier_point_diagnostic",
        }
    )
    (out_dir / "eval_single_lag_point_frontier_diagnostic.yaml").write_text(yaml.safe_dump(point, sort_keys=False))
    general = dict(base)
    general.pop("detection_mode", None)
    general.pop("true_hypo_threshold", None)
    general.pop("min_event_duration_minutes", None)
    general.pop("use_refractory", None)
    general["output_dir"] = "outputs/surrogate/mechanism_frontier_tests/general_metrics"
    (out_dir / "eval_general_metrics.yaml").write_text(yaml.safe_dump(general, sort_keys=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build configs for surrogate mechanism-frontier experiments.")
    parser.add_argument("--out-dir", type=Path, default=Path("configs/surrogate/mechanism_frontier_tests"))
    parser.add_argument("--capacity-pilot-dataset", default="dclp3")
    parser.add_argument("--selected-capacities-json", type=Path, default=None)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    capacities = None
    if args.selected_capacities_json is not None and args.selected_capacities_json.exists():
        selected = json.loads(args.selected_capacities_json.read_text())
        capacities = {
            "default": float(selected["default"]),
            "medium_small": float(selected["medium_small"]),
            "small": float(selected["small"]),
        }
    loss_paths = build_loss_comparator_configs(args.out_dir)
    pilot_paths = build_capacity_pilot_configs(args.out_dir, dataset=args.capacity_pilot_dataset)
    sweep_paths = build_capacity_sweep_configs(args.out_dir, capacities=capacities)
    build_eval_configs(args.out_dir)
    print(f"wrote {len(loss_paths)} loss-comparator configs")
    print(f"wrote {len(pilot_paths)} capacity-pilot configs")
    print(f"wrote {len(sweep_paths)} capacity-sweep configs")


if __name__ == "__main__":
    main()
