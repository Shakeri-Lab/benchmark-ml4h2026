#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import itertools
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml


DATASETS = {
    "dclp3": "dclp3_patients_70_10_20_seed42",
    "dclp5": "dclp5_patients_70_10_20_seed42",
    "pedap": "pedap_patients_70_10_20_seed42",
    "shanghait1dm": "shanghai_t1dm_patients_70_10_20_seed42",
    "iobp2": "iobp2_patients_70_10_20_seed42",
    "action_cf_benchmark": "action_cf_benchmark",
    "action_cf_benchmark_controliq": "action_cf_benchmark_controliq",
}


# Prefer 1-hour configs when available; fall back to p6 configs when missing.
BASE_CONFIGS = {
    "attengluco": {
        "dclp3": "configs/training/dclp3_attengluco_mv.yaml",
        "dclp5": "configs/training/dclp5_attengluco_mv.yaml",
        "pedap": "configs/training/pedap_attengluco_mv.yaml",
        "shanghait1dm": "configs/training/dclp3_attengluco_mv.yaml",
        "iobp2": "configs/training/dclp3_attengluco_mv.yaml",
    },
    "dlinear": {
        "dclp3": "configs/training/dclp3_dlinear_mv.yaml",
        "dclp5": "configs/training/dclp5_dlinear_mv.yaml",
        "pedap": "configs/training/pedap_dlinear_mv.yaml",
        "shanghait1dm": "configs/training/dclp3_dlinear_mv.yaml",
        "iobp2": "configs/training/dclp3_dlinear_mv.yaml",
    },
    "gru": {
        "dclp3": "configs/training/dclp3_gru_mv.yaml",
        "dclp5": "configs/training/dclp5_gru_mv.yaml",
        "pedap": "configs/training/pedap_gru_mv.yaml",
        "shanghait1dm": "configs/training/dclp3_gru_mv.yaml",
        "iobp2": "configs/training/dclp3_gru_mv.yaml",
    },
    "patchmlp": {
        "dclp3": "configs/training/dclp3_patchmlp_mv.yaml",
        "dclp5": "configs/training/dclp5_patchmlp_mv.yaml",
        "pedap": "configs/training/pedap_patchmlp_mv.yaml",
        "shanghait1dm": "configs/training/dclp3_patchmlp_mv.yaml",
        "iobp2": "configs/training/dclp3_patchmlp_mv.yaml",
    },
    "itransformer": {
        "dclp3": "configs/training/dclp3_itransformer_mv.yaml",
        "dclp5": "configs/training/dclp5_itransformer_mv.yaml",
        "pedap": "configs/training/pedap_itransformer_mv.yaml",
        "shanghait1dm": "configs/training/dclp3_itransformer_mv.yaml",
        "iobp2": "configs/training/dclp3_itransformer_mv.yaml",
    },
    "arimax": {
        "dclp3": "configs/training/dclp3_arimax_mv.yaml",
        "dclp5": "configs/training/dclp5_arimax_mv.yaml",
        "pedap": "configs/training/pedap_arimax_mv.yaml",
        "shanghait1dm": "configs/training/dclp3_arimax_mv.yaml",
        "iobp2": "configs/training/dclp3_arimax_mv.yaml",
    },
}


DATASET_OVERRIDES = {
    "default": {"pred_len": 6, "stride": 6, "history_len": 288},
    # IOBP2 is ~10x the size of the Jaeb cohorts (6.75M rows), so a stride of 6
    # makes the search prohibitively slow while adding little -- the models
    # plateau well before consuming that many windows, and epoch-based early
    # stopping then wastes a lot of compute before it triggers.
    #
    # The stride must be coprime to 288 (steps per day): gcd(stride, 288) = g
    # means window starts within a sequence only ever land on 288/g distinct
    # times of day. A stride of 60 shares a factor of 12 and would sample just
    # 24 phases, aliasing the diurnal cycle. 59 is prime, so starts cycle
    # through all 288 phases while still cutting cost ~10x.
    "iobp2": {"pred_len": 6, "stride": 59, "history_len": 288},
    "action_cf_benchmark": {"pred_len": 24, "stride": 24, "history_len": 288},
    "action_cf_benchmark_controliq": {"pred_len": 24, "stride": 24, "history_len": 288},
}

ACTION_CF_SOURCE_DATASETS = {"action_cf_benchmark", "action_cf_benchmark_controliq"}


def _apply_exog_regime(cfg: Dict[str, Any], model: str, dataset: str) -> None:
    """Observational cohorts get a strictly causal future-exog rule; the
    interventional datasets are handed the planned insulin sequence."""
    if model != "arimax":
        return
    regime = "provided" if dataset in ACTION_CF_SOURCE_DATASETS else "auto"
    _set(cfg, ["model_data", "model", "params", "future_exog_strategy"], regime)
    if regime == "provided":
        # The CF loader left-shifts basal and bolus by pred_len; those are exog
        # indices 0 and 1 in the [cgm, basal, bolus, meal] schema. The model
        # inverts the shift to recover the planned insulin.
        _set(cfg, ["model_data", "model", "params", "cf_shifted_exog_idx"], [0, 1])


def _apply_dataset_overrides(cfg: Dict[str, Any], dataset: str) -> None:
    overrides = DATASET_OVERRIDES.get(dataset, DATASET_OVERRIDES["default"])
    _set(cfg, ["model_data", "dataloader", "pred_len"], overrides["pred_len"])
    _set(cfg, ["model_data", "dataloader", "stride"], overrides["stride"])
    _set(cfg, ["model_data", "dataloader", "history_len"], overrides["history_len"])


def _set(cfg: Dict[str, Any], path: List[str], value: Any) -> None:
    cur = cfg
    for key in path[:-1]:
        cur = cur.setdefault(key, {})
    cur[path[-1]] = value


def _apply_bits(model: str, cfg: Dict[str, Any], bits: Tuple[int, ...]) -> None:
    b = list(bits)
    if model == "attengluco":
        _set(cfg, ["model_data", "model", "params", "d_model"], 32 if b[0] == 0 else 64)
        _set(cfg, ["model_data", "model", "params", "n_heads"], 4 if b[1] == 0 else 8)
        _set(cfg, ["model_data", "model", "params", "ff_units"], 128 if b[2] == 0 else 256)
        _set(cfg, ["training", "optimizer", "lr"], 1e-3 if b[3] == 0 else 5e-4)
        _set(cfg, ["training", "optimizer", "weight_decay"], 0.0 if b[4] == 0 else 1e-4)
        _set(cfg, ["model_data", "dataloader", "batch_size"], 64 if b[5] == 0 else 128)
        return
    if model == "dlinear":
        _set(cfg, ["model_data", "model", "params", "individual"], False if b[0] == 0 else True)
        _set(cfg, ["training", "optimizer", "lr"], 1e-4 if b[1] == 0 else 5e-4)
        _set(cfg, ["training", "optimizer", "weight_decay"], 0.0 if b[2] == 0 else 1e-3)
        _set(cfg, ["model_data", "dataloader", "batch_size"], 64 if b[3] == 0 else 128)
        _set(cfg, ["model_data", "dataloader", "history_len"], 48 if b[4] == 0 else 96)
        _set(cfg, ["training", "optimizer", "name"], "adamw" if b[5] == 0 else "adam")
        return
    if model == "gru":
        _set(cfg, ["model_data", "model", "params", "hidden_size"], 16 if b[0] == 0 else 32)
        _set(cfg, ["model_data", "model", "params", "num_layers"], 1 if b[1] == 0 else 2)
        _set(cfg, ["model_data", "model", "params", "dropout"], 0.0 if b[2] == 0 else 0.2)
        _set(cfg, ["model_data", "model", "params", "bidirectional"], False if b[3] == 0 else True)
        _set(cfg, ["training", "optimizer", "lr"], 1e-4 if b[4] == 0 else 5e-4)
        _set(cfg, ["model_data", "dataloader", "batch_size"], 64 if b[5] == 0 else 128)
        return
    if model == "patchmlp":
        _set(cfg, ["model_data", "model", "params", "d_model"], 512 if b[0] == 0 else 1024)
        _set(cfg, ["model_data", "model", "params", "e_layers"], 1 if b[1] == 0 else 2)
        _set(cfg, ["model_data", "model", "params", "use_norm"], True if b[2] == 0 else False)
        _set(cfg, ["training", "optimizer", "lr"], 1e-4 if b[3] == 0 else 5e-4)
        _set(cfg, ["training", "optimizer", "weight_decay"], 0.0 if b[4] == 0 else 1e-3)
        _set(cfg, ["model_data", "dataloader", "batch_size"], 64 if b[5] == 0 else 128)
        return
    if model == "itransformer":
        _set(cfg, ["model_data", "model", "params", "d_model"], 64 if b[0] == 0 else 128)
        _set(cfg, ["model_data", "model", "params", "n_heads"], 4 if b[1] == 0 else 8)
        _set(cfg, ["model_data", "model", "params", "e_layers"], 2 if b[2] == 0 else 3)
        _set(cfg, ["model_data", "model", "params", "d_ff"], 128 if b[3] == 0 else 256)
        _set(cfg, ["model_data", "model", "params", "dropout"], 0.1 if b[4] == 0 else 0.2)
        _set(cfg, ["training", "optimizer", "lr"], 1e-4 if b[5] == 0 else 5e-4)
        return
    if model == "arimax":
        p = 2 if b[0] == 0 else 3
        q = 1 if b[1] == 0 else 2
        d = 0 if b[2] == 0 else 1
        _set(cfg, ["model_data", "model", "params", "order"], [p, d, q])
        _set(cfg, ["model_data", "model", "params", "trend"], "c" if b[3] == 0 else None)
        _set(cfg, ["model_data", "dataloader", "history_len"], 48 if b[4] == 0 else 96)
        _set(cfg, ["model_data", "model", "params", "normalize_insulin_by_weight"], b[5] == 1)
        # Pinned, not searched: these were the defects behind the shipped model.
        _set(cfg, ["model_data", "model", "params", "enforce_invertibility"], True)
        _set(cfg, ["model_data", "model", "params", "enforce_stationarity"], True)
        _set(cfg, ["model_data", "model", "params", "drop_static_exog"], True)
        _set(cfg, ["model_data", "model", "params", "fit_sample_limit"], 1000)
        return
    raise ValueError(f"Unknown model {model}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("configs/hyperparameter"))
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset keys to generate (default: all).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model keys to generate (default: all).",
    )
    parser.add_argument(
        "--fail-if-exists",
        action="store_true",
        help="Refuse to write into an existing model/dataset config directory containing YAML files.",
    )
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product([0, 1], repeat=6))
    combos.sort(key=lambda b: (sum(b), b))

    dataset_keys = list(DATASETS.keys())
    if args.datasets:
        dataset_keys = [d.strip() for d in args.datasets.split(",") if d.strip()]
        missing = [d for d in dataset_keys if d not in DATASETS]
        if missing:
            raise ValueError(f"Unknown datasets: {missing}. Available: {sorted(DATASETS)}")

    model_keys = list(BASE_CONFIGS.keys())
    if args.models:
        model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
        missing = [m for m in model_keys if m not in BASE_CONFIGS]
        if missing:
            raise ValueError(f"Unknown models: {missing}. Available: {sorted(BASE_CONFIGS)}")

    for model in model_keys:
        ds_map = BASE_CONFIGS[model]
        for dataset in dataset_keys:
            base_path = ds_map.get(dataset)
            if base_path is None and dataset in ACTION_CF_SOURCE_DATASETS:
                base_path = ds_map.get("dclp3")
            if base_path is None:
                raise ValueError(f"No base config for model={model} dataset={dataset}")

            base_cfg = yaml.safe_load(Path(base_path).read_text())
            _apply_dataset_overrides(base_cfg, dataset)
            _apply_exog_regime(base_cfg, model, dataset)
            # update split name for dataset
            _set(base_cfg, ["split", "split_name"], DATASETS[dataset])
            _set(base_cfg, ["notes"], f"Hyperparameter search: {model} on {dataset}")

            out_dir = args.output_root / f"{model}_{dataset}"
            if args.fail_if_exists and out_dir.exists() and any(out_dir.glob("*.yaml")):
                raise FileExistsError(f"Refusing to overwrite existing configs in {out_dir}")
            out_dir.mkdir(parents=True, exist_ok=True)
            for idx, bits in enumerate(combos[:64]):
                cfg = copy.deepcopy(base_cfg)
                _apply_bits(model, cfg, bits)
                out_path = out_dir / f"{idx:03d}.yaml"
                out_path.write_text(yaml.safe_dump(cfg))

    print(f"Wrote configs to {args.output_root}")


if __name__ == "__main__":
    main()
