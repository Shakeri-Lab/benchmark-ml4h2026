from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.evaluate_hypo_gating import (  # noqa: E402
    _bootstrap_patient_metrics,
    _has_prediction_window,
    _load_sequences,
    _load_split_spec,
    _macro_average_patient_metrics,
    _patient_metric_record,
    _resolve_slice_thresholds,
    _run_inference,
    _sanitize_state_dict,
    build_alarm_indices,
    extract_event_onsets,
    match_alarms_to_events,
)
from data import make_dataloader_from_cfg  # noqa: E402
from models import build_model_from_cfg  # noqa: E402

DATASETS = ("dclp3", "dclp5", "pedap")
HORIZON_STEPS = (6, 12)
THRESHOLD_MINUTES = (15, 30, 46, 60)


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def _threshold_steps(minutes: int, step_minutes: int) -> int:
    # The slice marker is stored in 5-minute steps. A 46-minute threshold maps
    # to <=45 minutes, which is the exact <=46 criterion on a 5-minute grid.
    return int(math.floor(float(minutes) / float(step_minutes)))


def _parse_model_dir_name(name: str) -> tuple[str, str, int] | None:
    for dataset in DATASETS:
        for h in HORIZON_STEPS:
            suffix = f"_{dataset}_p{h}"
            if name.startswith("best_") and name.endswith(suffix):
                model_name = name[len("best_") : -len(suffix)]
                return model_name, dataset, h
    return None


def _find_model_dirs(results_root: Path, datasets: Iterable[str], horizons: Iterable[int]) -> list[Path]:
    dataset_set = set(datasets)
    horizon_set = set(int(h) for h in horizons)
    out: list[Path] = []
    for path in sorted(results_root.glob("best*")):
        if not path.is_dir():
            continue
        parsed = _parse_model_dir_name(path.name)
        if parsed is None:
            continue
        _, dataset, horizon_steps = parsed
        if dataset not in dataset_set or horizon_steps not in horizon_set:
            continue
        if (path / "config.yaml").exists() and (path / "checkpoint.pt").exists():
            out.append(path)
    return out


def _load_model_context(model_dir: Path, split_part: str, step_minutes: int):
    cfg = _load_yaml(model_dir / "config.yaml")
    eval_cfg = cfg.get("eval", cfg)
    train_cfg = cfg.get("train", {})

    split_cfg = dict(eval_cfg.get("split") or train_cfg.get("split") or {})
    model_data = dict(eval_cfg.get("model_data") or train_cfg.get("model_data") or {})
    dl_cfg = dict(model_data.get("dataloader") or {})
    model_cfg = dict(model_data.get("model") or {})

    split_root = Path(split_cfg.get("split_root", "configs/preprocessing"))
    split_name = split_cfg.get("split_name")
    if not split_name:
        raise ValueError(f"Missing split_name in {model_dir}/config.yaml")
    split_spec = _load_split_spec(split_root, split_name)
    dataset_name = split_spec.get("dataset") or split_name.split("_")[0]

    split_cfg = dict(split_cfg)
    split_cfg["split_part"] = split_part
    dl_cfg = dict(dl_cfg)
    dl_cfg["stride"] = 1
    dl_cfg["shuffle"] = False

    seed = model_data.get("seed", 42)
    dl = make_dataloader_from_cfg(split_cfg, dl_cfg, split_part=str(split_part), seed=seed)

    model_name = model_cfg.get("name") or cfg.get("model")
    if not model_name:
        raise ValueError(f"Missing model name in {model_dir}/config.yaml")
    hist = int(dl_cfg.get("history_len", 288))
    pred_len = int(dl_cfg.get("pred_len", 24))
    input_dim = len(dl.dataset.feature_cols)
    model = build_model_from_cfg(
        {"name": model_name, "params": model_cfg.get("params", {})},
        input_dim=input_dim,
        history_len=hist,
        pred_len=pred_len,
    )

    ckpt_path = model_dir / "checkpoint.pt"
    if ckpt_path.exists() and hasattr(model, "load_state_dict"):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = _sanitize_state_dict(ckpt.get("state_dict", None))
        if state:
            try:
                model.load_state_dict(state)
            except Exception:
                pass

    sequences = _load_sequences(Path(split_cfg.get("processed_root", "datasets_processed")), dataset_name, split_spec, split_part)
    return cfg, model, dl, pred_len, sequences, dataset_name


def _slice_metrics_from_predictions(
    *,
    model_dir: Path,
    model_name: str,
    dataset_name: str,
    pred_len: int,
    sequences: Dict[tuple[int, int], Dict[str, Any]],
    preds_by_seq: Dict[tuple[int, int], Dict[int, np.ndarray]],
    event_by_seq: Dict[tuple[int, int], Dict[int, tuple[int, int, int]]],
    slice_name: str,
    post_bolus_threshold_steps: int,
    nocturnal_threshold_steps: int,
    hypo_threshold: float,
    min_event_duration_minutes: int,
    step_minutes: int,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> Dict[str, Any]:
    min_samples = max(1, int(round(min_event_duration_minutes / step_minutes)))
    pred_len_samples = int(pred_len)

    patient_day_counts: Dict[int, float] = {}
    patient_true_events: Dict[int, int] = {}
    patient_detected_events: Dict[int, int] = {}
    patient_false_alarms: Dict[int, int] = {}
    patient_lead_times_samples: Dict[int, list[int]] = {}

    for key, seq in sequences.items():
        cgm = seq["cgm"]
        pred_dict_all = preds_by_seq.get(key, {})
        event_dict = event_by_seq.get(key, {})
        if slice_name == "post_bolus":
            pred_dict = {
                idx: pred
                for idx, pred in pred_dict_all.items()
                if idx in event_dict
                and int(event_dict[idx][1]) >= 0
                and int(event_dict[idx][1]) <= post_bolus_threshold_steps
            }
        elif slice_name == "nocturnal":
            pred_dict = {
                idx: pred
                for idx, pred in pred_dict_all.items()
                if idx in event_dict
                and int(event_dict[idx][2]) >= 0
                and int(event_dict[idx][2]) <= nocturnal_threshold_steps
            }
        else:
            raise ValueError(f"Unsupported slice for this sensitivity analysis: {slice_name}")
        pred_indices_sorted = sorted(pred_dict.keys())
        if not pred_indices_sorted:
            continue
        pid = key[0]
        coverage_samples = len(pred_indices_sorted)
        patient_day_counts[pid] = patient_day_counts.get(pid, 0.0) + (coverage_samples * step_minutes / 1440.0)

        events = extract_event_onsets(cgm, hypo_threshold, min_samples)
        valid_events = [t0 for t0 in events if _has_prediction_window(pred_indices_sorted, t0, pred_len_samples)]
        patient_true_events[pid] = patient_true_events.get(pid, 0) + len(valid_events)

        alarms = build_alarm_indices(pred_dict, hypo_threshold, pred_len_samples)
        detected, lead_times, false_alarms = match_alarms_to_events(
            alarms, valid_events, pred_len_samples, pred_indices_sorted
        )
        patient_detected_events[pid] = patient_detected_events.get(pid, 0) + int(detected)
        patient_false_alarms[pid] = patient_false_alarms.get(pid, 0) + int(false_alarms)
        patient_lead_times_samples.setdefault(pid, []).extend(int(x) for x in lead_times)

    pids = sorted(
        set(patient_day_counts)
        | set(patient_true_events)
        | set(patient_detected_events)
        | set(patient_false_alarms)
        | set(patient_lead_times_samples)
    )
    patient_metrics = [
        _patient_metric_record(
            pid=pid,
            true_events=patient_true_events.get(pid, 0),
            detected_events=patient_detected_events.get(pid, 0),
            false_alarms=patient_false_alarms.get(pid, 0),
            patient_days=patient_day_counts.get(pid, 0.0),
            lead_times_samples=patient_lead_times_samples.get(pid, []),
            step_minutes=step_minutes,
            point_count=0,
            point_sum_sq=0.0,
            point_peg_counts=np.zeros(9, dtype=float),
        )
        for pid in pids
    ]
    macro = _macro_average_patient_metrics(patient_metrics)
    ci = _bootstrap_patient_metrics(patient_metrics, n_bootstrap=n_bootstrap, seed=bootstrap_seed)
    return {
        "slice": slice_name,
        "model_name": str(model_name),
        "dataset_name": str(dataset_name),
        "prediction_horizon_minutes": int(pred_len * step_minutes),
        "event_recall": float(macro["event_recall"]),
        "event_recall_ci_low": float(ci.get("event_recall", {}).get("ci_low", float("nan"))),
        "event_recall_ci_high": float(ci.get("event_recall", {}).get("ci_high", float("nan"))),
        "false_alarms_per_patient_day": float(macro["false_alarms_per_patient_day"]),
        "false_alarms_per_patient_day_ci_low": float(
            ci.get("false_alarms_per_patient_day", {}).get("ci_low", float("nan"))
        ),
        "false_alarms_per_patient_day_ci_high": float(
            ci.get("false_alarms_per_patient_day", {}).get("ci_high", float("nan"))
        ),
        "median_warning_lead_time_minutes": float(macro["median_warning_lead_time_minutes"]),
        "median_warning_lead_time_minutes_ci_low": float(
            ci.get("median_warning_lead_time_minutes", {}).get("ci_low", float("nan"))
        ),
        "median_warning_lead_time_minutes_ci_high": float(
            ci.get("median_warning_lead_time_minutes", {}).get("ci_high", float("nan"))
        ),
        "mean_warning_lead_time_minutes": float(macro["mean_warning_lead_time_minutes"]),
        "mean_warning_lead_time_minutes_ci_low": float(
            ci.get("mean_warning_lead_time_minutes", {}).get("ci_low", float("nan"))
        ),
        "mean_warning_lead_time_minutes_ci_high": float(
            ci.get("mean_warning_lead_time_minutes", {}).get("ci_high", float("nan"))
        ),
        "num_true_events": float(macro["num_true_events"]),
        "num_detected_events": float(macro["num_detected_events"]),
        "num_false_alarms": float(macro["num_false_alarms"]),
        "num_patient_days_evaluated": float(macro["num_patient_days_evaluated"]),
        "n_patients": int(len(patient_metrics)),
        "model_dir": str(model_dir),
    }


def evaluate_model_dir_thresholds(
    model_dir: Path,
    *,
    threshold_minutes: Iterable[int],
    split_part: str,
    hypo_threshold: float,
    min_event_duration_minutes: int,
    step_minutes: int,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> list[Dict[str, Any]]:
    parsed = _parse_model_dir_name(model_dir.name)
    if parsed is None:
        raise ValueError(f"Could not parse model/dataset/horizon from {model_dir.name}")
    model_label, dataset_from_name, horizon_steps = parsed
    cfg, model, dl, pred_len, sequences, dataset_name = _load_model_context(model_dir, split_part, step_minutes)
    max_steps = max(_threshold_steps(int(m), step_minutes) for m in threshold_minutes)
    slice_thresholds = _resolve_slice_thresholds(cfg, overrides={"post_bolus": max_steps})
    nocturnal_threshold_steps = int(slice_thresholds.get("nocturnal", 0))
    preds_by_seq, event_by_seq, _ = _run_inference(
        model,
        dl,
        pred_len,
        sequences,
        slice_thresholds=slice_thresholds,
        cached_preds=None,
        cached_gts=None,
        cached_events=None,
    )

    rows: list[Dict[str, Any]] = []
    for minutes in threshold_minutes:
        steps = _threshold_steps(int(minutes), step_minutes)
        for slice_name in ("post_bolus", "nocturnal"):
            row = _slice_metrics_from_predictions(
                model_dir=model_dir,
                model_name=model_label,
                dataset_name=dataset_name or dataset_from_name,
                pred_len=pred_len,
                sequences=sequences,
                preds_by_seq=preds_by_seq,
                event_by_seq=event_by_seq,
                slice_name=slice_name,
                post_bolus_threshold_steps=steps,
                nocturnal_threshold_steps=nocturnal_threshold_steps,
                hypo_threshold=hypo_threshold,
                min_event_duration_minutes=min_event_duration_minutes,
                step_minutes=step_minutes,
                n_bootstrap=n_bootstrap,
                bootstrap_seed=bootstrap_seed,
            )
            row.update(
                {
                    "dataset_name": dataset_from_name,
                    "model_name": model_label,
                    "prediction_horizon_steps": int(horizon_steps),
                    "prediction_horizon_minutes": int(horizon_steps * step_minutes),
                    "post_bolus_threshold_minutes_requested": int(minutes),
                    "post_bolus_threshold_steps": int(steps),
                    "post_bolus_threshold_minutes_effective": int(steps * step_minutes),
                    "nocturnal_threshold_steps": int(nocturnal_threshold_steps),
                    "nocturnal_threshold_minutes_effective": int(nocturnal_threshold_steps * step_minutes),
                    "error": "",
                }
            )
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute post-bolus hypoglycemia-gating metrics over multiple post-bolus slice thresholds."
    )
    parser.add_argument("--results-root", type=Path, default=Path("outputs/results"))
    parser.add_argument("--model-dir", type=Path, default=None, help="Evaluate one model directory instead of scanning results-root.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/results/hypo_gating_sensitivity/post_bolus_threshold_sensitivity_p6_p12.csv"),
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--horizon-steps", nargs="+", type=int, default=list(HORIZON_STEPS), choices=list(HORIZON_STEPS))
    parser.add_argument("--threshold-minutes", nargs="+", type=int, default=list(THRESHOLD_MINUTES))
    parser.add_argument("--split", default="test")
    parser.add_argument("--hypo-threshold", type=float, default=70.0)
    parser.add_argument("--min-event-duration-minutes", type=int, default=15)
    parser.add_argument("--step-minutes", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    if args.model_dir is not None:
        model_dirs = [args.model_dir]
    else:
        model_dirs = _find_model_dirs(args.results_root, args.datasets, args.horizon_steps)
    if not model_dirs:
        raise SystemExit(f"No matching model directories found under {args.results_root}")

    rows: list[Dict[str, Any]] = []
    for model_dir in tqdm(model_dirs, desc="Models"):
        try:
            rows.extend(
                evaluate_model_dir_thresholds(
                    model_dir,
                    threshold_minutes=args.threshold_minutes,
                    split_part=args.split,
                    hypo_threshold=float(args.hypo_threshold),
                    min_event_duration_minutes=int(args.min_event_duration_minutes),
                    step_minutes=int(args.step_minutes),
                    n_bootstrap=int(args.n_bootstrap),
                    bootstrap_seed=int(args.bootstrap_seed),
                )
            )
        except Exception as exc:
            parsed = _parse_model_dir_name(model_dir.name)
            model_name, dataset, horizon_steps = parsed if parsed else (model_dir.name, "unknown", -1)
            for minutes in args.threshold_minutes:
                steps = _threshold_steps(int(minutes), args.step_minutes)
                for slice_name in ("post_bolus", "nocturnal"):
                    rows.append(
                        {
                            "dataset_name": dataset,
                            "model_name": model_name,
                            "model_dir": str(model_dir),
                            "prediction_horizon_steps": int(horizon_steps),
                            "prediction_horizon_minutes": int(horizon_steps * args.step_minutes)
                            if horizon_steps > 0
                            else np.nan,
                            "post_bolus_threshold_minutes_requested": int(minutes),
                            "post_bolus_threshold_steps": int(steps),
                            "post_bolus_threshold_minutes_effective": int(steps * args.step_minutes),
                            "slice": slice_name,
                            "nocturnal_threshold_steps": np.nan,
                            "nocturnal_threshold_minutes_effective": np.nan,
                            "event_recall": np.nan,
                            "event_recall_ci_low": np.nan,
                            "event_recall_ci_high": np.nan,
                            "false_alarms_per_patient_day": np.nan,
                            "false_alarms_per_patient_day_ci_low": np.nan,
                            "false_alarms_per_patient_day_ci_high": np.nan,
                            "median_warning_lead_time_minutes": np.nan,
                            "median_warning_lead_time_minutes_ci_low": np.nan,
                            "median_warning_lead_time_minutes_ci_high": np.nan,
                            "mean_warning_lead_time_minutes": np.nan,
                            "mean_warning_lead_time_minutes_ci_low": np.nan,
                            "mean_warning_lead_time_minutes_ci_high": np.nan,
                            "num_true_events": np.nan,
                            "num_detected_events": np.nan,
                            "num_false_alarms": np.nan,
                            "num_patient_days_evaluated": np.nan,
                            "n_patients": np.nan,
                            "error": str(exc),
                        }
                    )

    df = pd.DataFrame(rows)
    df = df.sort_values(
        [
            "dataset_name",
            "prediction_horizon_minutes",
            "post_bolus_threshold_minutes_requested",
            "slice",
            "model_name",
        ]
    ).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"[+] Wrote {len(df)} rows to {args.output_csv}")
    if "error" in df and df["error"].astype(str).str.len().gt(0).any():
        print("[!] Some rows had errors:")
        print(df[df["error"].astype(str).str.len().gt(0)][["model_dir", "post_bolus_threshold_minutes_requested", "error"]].to_string(index=False))


if __name__ == "__main__":
    main()
