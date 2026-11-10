from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

# Ensure repo root on path for local imports
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from data import make_dataloader_from_cfg  # noqa: E402
from metrics.clinical import ClinicalMetrics, zone_merge_counts  # noqa: E402
from models import build_model_from_cfg  # noqa: E402

SLICE_ORDER = ["overall", "nocturnal", "post_bolus", "post_meal"]


# ---------------------------------------------------------------------------
# Hypoglycemia safety-gating metrics for pretrained forecasting models.
#
# Event definition:
#   CGM < threshold for at least min_event_duration_minutes (>=3 samples @ 5min)
#   Event onset t0 is first sample in the run.
#   New event only after recovery >= threshold for >= min_event_duration_minutes.
#
# Alarm definition:
#   At time t (context end), if any predicted value within next PH minutes
#   is below threshold, an alarm can fire (subject to refractory).
#
# Metrics:
# - event_recall: detected_events / true_events
# - median_warning_lead_time_minutes: median(t0 - t_alarm) for detected events
# - false_alarms_per_patient_day: num_false_alarms / patient_days
# ---------------------------------------------------------------------------


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


def _sanitize_state_dict(state):
    if not state:
        return state
    cleaned = {}
    for k, v in state.items():
        if k.startswith("_orig_mod."):
            cleaned[k[len("_orig_mod.") :]] = v
        elif k.startswith("module."):
            cleaned[k[len("module.") :]] = v
        else:
            cleaned[k] = v
    return cleaned


def _load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def _load_split_spec(split_root: Path, split_name: str) -> Dict[str, Any]:
    path = Path(split_root) / split_name
    if path.is_dir():
        candidates = list(path.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"No split spec found in {path}")
        path = candidates[0]
    if not path.suffix:
        path = path.with_suffix(".json")
    if not path.exists():
        raise FileNotFoundError(f"Split spec not found: {path}")
    return json.loads(path.read_text())


def _canon_pid(val) -> str:
    try:
        f = float(val)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return str(val).strip()


def _timeseries_path(processed_root: Path, dataset_name: str) -> Path:
    candidates = [
        Path(processed_root) / dataset_name / "timeseries.csv",
        Path("dataset_processed") / dataset_name / "timeseries.csv",
        Path("datasets_preprocessed") / dataset_name / "timeseries.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"timeseries not found for {dataset_name}: tried {candidates}")


def _load_sequences(
    processed_root: Path,
    dataset_name: str,
    split_spec: Dict[str, Any],
    split_part: str,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    ts_path = _timeseries_path(processed_root, dataset_name)
    df = pd.read_csv(ts_path)
    if "pat_id" not in df.columns:
        if "pid" in df.columns:
            df = df.rename(columns={"pid": "pat_id"})
        else:
            raise KeyError(f"'pat_id' (or pid) not in dataframe {ts_path}")
    if "seq_id" not in df.columns:
        df = df.sort_values(["pat_id", "date"]).reset_index(drop=True)
        df["seq_id"] = df.groupby("pat_id").cumcount().mul(0)

    patients = split_spec.get("patients", {}).get(split_part)
    if patients is None:
        raise KeyError(f"Split part '{split_part}' not found in spec")
    pat_set = {_canon_pid(p) for p in patients}
    pat_key = df["pat_id"].apply(_canon_pid)
    df = df[pat_key.isin(pat_set)]

    df = df.dropna(subset=["cgm"]).copy()
    df = df.sort_values(["pat_id", "seq_id", "date"]).reset_index(drop=True)

    sequences: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for (pid, sid), g in df.groupby(["pat_id", "seq_id"], sort=False):
        dates = g["date"].astype(str).to_list()
        cgm = g["cgm"].to_numpy(dtype=float)
        date_to_idx = {d: i for i, d in enumerate(dates)}
        sequences[(int(pid), int(sid))] = {
            "dates": dates,
            "cgm": cgm,
            "date_to_idx": date_to_idx,
        }
    return sequences


def extract_event_onsets(
    cgm: np.ndarray,
    threshold: float,
    min_samples: int,
) -> List[int]:
    """Return onset indices for hypoglycemic events with recovery-based separation."""
    onsets = []
    n = len(cgm)
    below = cgm < threshold
    i = 0
    while i < n:
        if below[i]:
            start = i
            while i < n and below[i]:
                i += 1
            run_len = i - start
            if run_len >= min_samples:
                onsets.append(start)
                # Require recovery above threshold for min_samples to start a new event
                recovery = 0
                while i < n:
                    if cgm[i] >= threshold:
                        recovery += 1
                        if recovery >= min_samples:
                            i += 1
                            break
                    else:
                        recovery = 0
                    i += 1
            else:
                # Too short to be an event; continue scanning
                continue
        else:
            i += 1
    return onsets


def build_alarm_indices(
    pred_by_idx: Dict[int, np.ndarray],
    threshold: float,
    pred_len: int,
) -> List[int]:
    """Generate alarm indices with refractory period pred_len."""
    candidates = []
    for idx, pred in pred_by_idx.items():
        arr = np.asarray(pred).reshape(-1)
        alarm = bool(np.any(arr < threshold))
        if alarm:
            candidates.append(idx)
    if not candidates:
        return []
    candidates.sort()
    alarms = []
    last_alarm = -10**9
    for idx in candidates:
        if idx - last_alarm < pred_len:
            continue
        alarms.append(idx)
        last_alarm = idx
    return alarms


def _has_prediction_window(
    pred_indices_sorted: List[int],
    t0: int,
    pred_len: int,
) -> bool:
    if not pred_indices_sorted:
        return False
    # Require at least one warning step before onset, so only prediction times
    # strictly before t0 can make an event evaluable/detectable.
    lo = t0 - pred_len
    hi = t0 - 1
    # binary search
    import bisect

    left = bisect.bisect_left(pred_indices_sorted, lo)
    right = bisect.bisect_right(pred_indices_sorted, hi)
    return left < right


def match_alarms_to_events(
    alarms: List[int],
    event_onsets: List[int],
    pred_len: int,
    pred_indices_sorted: List[int],
) -> Tuple[int, List[int], int]:
    """Match alarms to events; alarms at onset do not count as detections."""
    used = set()
    detected = 0
    lead_times = []
    alarms_sorted = sorted(alarms)
    for t0 in event_onsets:
        if not _has_prediction_window(pred_indices_sorted, t0, pred_len):
            continue
        window_start = t0 - pred_len
        candidates = [a for a in alarms_sorted if (a >= window_start and a < t0 and a not in used)]
        if not candidates:
            continue
        a = min(candidates)
        used.add(a)
        detected += 1
        lead_times.append(t0 - a)
    false_alarms = len([a for a in alarms_sorted if a not in used])
    return detected, lead_times, false_alarms


def _patient_metric_record(
    pid: int,
    true_events: int,
    detected_events: int,
    false_alarms: int,
    patient_days: float,
    lead_times_samples: Sequence[int],
    step_minutes: int,
    point_count: int,
    point_sum_sq: float,
    point_peg_counts: Sequence[float],
) -> Dict[str, Any]:
    if lead_times_samples:
        lead_times_minutes = np.asarray(lead_times_samples, dtype=float) * step_minutes
        median_lead = float(np.median(lead_times_minutes))
        mean_lead = float(np.mean(lead_times_minutes))
    else:
        median_lead = float("nan")
        mean_lead = float("nan")

    recall = (
        float(detected_events) / float(true_events)
        if true_events > 0
        else float("nan")
    )
    false_per_day = (
        float(false_alarms) / float(patient_days)
        if patient_days > 0
        else float("nan")
    )
    rmse = float(np.sqrt(point_sum_sq / float(point_count))) if point_count > 0 else float("nan")
    peg_counts = np.asarray(point_peg_counts, dtype=float)
    peg_merged = zone_merge_counts(peg_counts) if peg_counts.size > 0 else np.zeros(5, dtype=float)
    peg_unsafe = (
        float(peg_merged[2:5].sum() / peg_merged.sum())
        if peg_merged.sum() > 0
        else float("nan")
    )

    return {
        "pat_id": int(pid),
        "event_recall": float(recall),
        "median_warning_lead_time_minutes": float(median_lead),
        "mean_warning_lead_time_minutes": float(mean_lead),
        "false_alarms_per_patient_day": float(false_per_day),
        "rmse": float(rmse),
        "peg_unsafe": float(peg_unsafe),
        "num_true_events": int(true_events),
        "num_detected_events": int(detected_events),
        "num_false_alarms": int(false_alarms),
        "num_patient_days_evaluated": float(patient_days),
        "num_points": int(point_count),
    }


def _nanmean_or_nan(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    finite = np.isfinite(arr)
    if not finite.any():
        return float("nan")
    return float(arr[finite].mean())


def _macro_average_patient_metrics(patient_metrics: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    return {
        "event_recall": _nanmean_or_nan([pm["event_recall"] for pm in patient_metrics]),
        "median_warning_lead_time_minutes": _nanmean_or_nan(
            [pm["median_warning_lead_time_minutes"] for pm in patient_metrics]
        ),
        "mean_warning_lead_time_minutes": _nanmean_or_nan(
            [pm["mean_warning_lead_time_minutes"] for pm in patient_metrics]
        ),
        "false_alarms_per_patient_day": _nanmean_or_nan(
            [pm["false_alarms_per_patient_day"] for pm in patient_metrics]
        ),
        "rmse": _nanmean_or_nan([pm["rmse"] for pm in patient_metrics]),
        "peg_unsafe": _nanmean_or_nan([pm["peg_unsafe"] for pm in patient_metrics]),
        "num_true_events": _nanmean_or_nan([pm["num_true_events"] for pm in patient_metrics]),
        "num_detected_events": _nanmean_or_nan([pm["num_detected_events"] for pm in patient_metrics]),
        "num_false_alarms": _nanmean_or_nan([pm["num_false_alarms"] for pm in patient_metrics]),
        "num_patient_days_evaluated": _nanmean_or_nan(
            [pm["num_patient_days_evaluated"] for pm in patient_metrics]
        ),
    }


def _bootstrap_patient_metrics(
    patient_metrics: Sequence[Dict[str, Any]],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    if not patient_metrics:
        return {}
    rng = np.random.default_rng(seed)
    n_patients = len(patient_metrics)
    metric_names = [
        "event_recall",
        "median_warning_lead_time_minutes",
        "mean_warning_lead_time_minutes",
        "false_alarms_per_patient_day",
        "rmse",
        "peg_unsafe",
        "num_true_events",
        "num_detected_events",
        "num_false_alarms",
        "num_patient_days_evaluated",
    ]
    samples: Dict[str, List[float]] = {name: [] for name in metric_names}

    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n_patients, size=n_patients)
        boot = [patient_metrics[i] for i in idx]
        agg = _macro_average_patient_metrics(boot)
        for name in metric_names:
            samples[name].append(agg[name])

    ci: Dict[str, Dict[str, float]] = {}
    for name, vals in samples.items():
        arr = np.asarray(vals, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            ci[name] = {"ci_low": float("nan"), "ci_high": float("nan")}
        else:
            ci[name] = {
                "ci_low": float(np.percentile(finite, 2.5)),
                "ci_high": float(np.percentile(finite, 97.5)),
            }
    return ci


def _default_slice_thresholds() -> Dict[str, int]:
    return {
        "post_meal": 3,
        "post_bolus": 3,
        "nocturnal": 0,
    }


def _resolve_slice_thresholds(
    cfg: Dict[str, Any],
    overrides: Optional[Dict[str, Optional[int]]] = None,
) -> Dict[str, int]:
    eval_cfg = cfg.get("eval", cfg)
    train_cfg = cfg.get("train", {})
    raw = dict(_default_slice_thresholds())
    raw.update((train_cfg.get("evaluation") or {}).get("event_thresholds") or {})
    raw.update((eval_cfg.get("evaluation") or {}).get("event_thresholds") or {})
    raw.update(eval_cfg.get("event_thresholds") or {})
    if overrides:
        raw.update({k: int(v) for k, v in overrides.items() if v is not None})
    return {
        "post_meal": int(raw.get("post_meal", 3)),
        "post_bolus": int(raw.get("post_bolus", 3)),
        "nocturnal": int(raw.get("nocturnal", 0)),
    }


def _slice_flags(post_meal: int, post_bolus: int, nocturnal: int, slice_thresholds: Dict[str, int]) -> Dict[str, bool]:
    return {
        "overall": True,
        "post_meal": bool(post_meal >= 0 and post_meal <= int(slice_thresholds["post_meal"])),
        "post_bolus": bool(post_bolus >= 0 and post_bolus <= int(slice_thresholds["post_bolus"])),
        "nocturnal": bool(nocturnal >= 0 and nocturnal <= int(slice_thresholds["nocturnal"])),
    }


def _predict_batch(model, X: torch.Tensor, pred_len: int) -> np.ndarray:
    if isinstance(model, torch.nn.Module):
        with torch.no_grad():
            preds_t = model(X, pred_len) if pred_len is not None else model(X)
        preds = preds_t.detach().cpu().numpy()
    elif hasattr(model, "predict"):
        preds = model.predict(X.numpy(), pred_len)
    else:
        raise ValueError("Model must implement predict or be a torch.nn.Module with forward")
    return np.asarray(preds)


def _run_inference(
    model,
    dl,
    pred_len: int,
    sequences: Dict[Tuple[int, int], Dict[str, Any]],
    slice_thresholds: Dict[str, int],
    cached_preds: Optional[np.ndarray] = None,
    cached_gts: Optional[np.ndarray] = None,
    cached_events: Optional[Dict[str, np.ndarray]] = None,
    min_context_steps: Optional[int] = None,
) -> Tuple[
    Dict[Tuple[int, int], Dict[int, np.ndarray]],
    Dict[Tuple[int, int], Dict[int, Tuple[int, int, int]]],
    Dict[str, Dict[int, Dict[str, Any]]],
]:
    pred_by_seq: Dict[Tuple[int, int], Dict[int, np.ndarray]] = {}
    event_by_seq: Dict[Tuple[int, int], Dict[int, Tuple[int, int, int]]] = {}
    patient_point_stats: Dict[str, Dict[int, Dict[str, Any]]] = {name: {} for name in SLICE_ORDER}
    cache_offset = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clinical = ClinicalMetrics.from_default()
    if isinstance(model, torch.nn.Module):
        model = model.to(device)
        model.eval()
    using_cached = cached_preds is not None
    batch_iter = tqdm(
        dl,
        desc=f"Hypo gating inference ({'cached' if using_cached else 'fresh'})",
        leave=False,
    )
    for batch in batch_iter:
        X, _, pid, sid, ctx_end_date, evt_tuple, *_ = batch
        batch_size = len(pid)
        if cached_preds is not None:
            lo = cache_offset
            hi = cache_offset + batch_size
            preds = np.asarray(cached_preds[lo:hi], dtype=float)
            y_true = np.asarray(
                cached_gts[lo:hi]
                if cached_gts is not None
                else (batch[1].detach().cpu().numpy() if torch.is_tensor(batch[1]) else batch[1]),
                dtype=float,
            )
            if cached_events is not None:
                cached_pid = np.asarray(cached_events["pat_id"][lo:hi], dtype=np.int64).reshape(-1)
                cached_sid = np.asarray(cached_events["seq_id"][lo:hi], dtype=np.int64).reshape(-1)
                batch_pid = np.asarray(pid, dtype=np.int64).reshape(-1)
                batch_sid = np.asarray(sid, dtype=np.int64).reshape(-1)
                if cached_pid.shape != batch_pid.shape or not np.array_equal(cached_pid, batch_pid) or not np.array_equal(cached_sid, batch_sid):
                    raise RuntimeError("Cached hypo-gating artifacts are misaligned with dataloader order.")
                post_meal = np.asarray(cached_events["post_meal"][lo:hi], dtype=np.int64).reshape(-1)
                post_bolus = np.asarray(cached_events["post_bolus"][lo:hi], dtype=np.int64).reshape(-1)
                nocturnal = np.asarray(cached_events["nocturnal"][lo:hi], dtype=np.int64).reshape(-1)
            else:
                post_meal, post_bolus, nocturnal = evt_tuple
                post_meal = np.asarray(post_meal, dtype=np.int64).reshape(-1)
                post_bolus = np.asarray(post_bolus, dtype=np.int64).reshape(-1)
                nocturnal = np.asarray(nocturnal, dtype=np.int64).reshape(-1)
            cache_offset = hi
        else:
            X = X.to(device) if isinstance(model, torch.nn.Module) else X
            preds = _predict_batch(model, X, pred_len)
            y_true = np.asarray(batch[1].detach().cpu().numpy() if torch.is_tensor(batch[1]) else batch[1], dtype=float)
            post_meal, post_bolus, nocturnal = evt_tuple
            post_meal = np.asarray(post_meal, dtype=np.int64).reshape(-1)
            post_bolus = np.asarray(post_bolus, dtype=np.int64).reshape(-1)
            nocturnal = np.asarray(nocturnal, dtype=np.int64).reshape(-1)
        peg_zones = clinical.peg_zone(y_true, preds)
        for i in range(len(pid)):
            key = (int(pid[i]), int(sid[i]))
            if key not in sequences:
                continue
            date_str = str(ctx_end_date[i])
            idx_map = sequences[key]["date_to_idx"]
            if date_str not in idx_map:
                continue
            idx = int(idx_map[date_str])
            # Common-window restriction: keep only targets that a model with
            # min_context_steps of history could also predict, so every model is
            # scored on an identical set of windows.
            if min_context_steps is not None and idx < int(min_context_steps) - 1:
                continue
            pred_vec = np.asarray(preds[i]).reshape(-1)
            pred_by_seq.setdefault(key, {})[idx] = pred_vec
            pm_i = int(post_meal[i])
            pb_i = int(post_bolus[i])
            no_i = int(nocturnal[i])
            event_by_seq.setdefault(key, {})[idx] = (pm_i, pb_i, no_i)
            flags = _slice_flags(pm_i, pb_i, no_i, slice_thresholds)
            pid_i = int(pid[i])
            y_i = np.asarray(y_true[i]).reshape(-1)
            pred_i = np.asarray(pred_vec).reshape(-1)
            finite = np.isfinite(y_i) & np.isfinite(pred_i)
            if finite.any():
                peg_counts = np.bincount(
                    np.asarray(peg_zones[i]).reshape(-1)[finite].astype(int),
                    minlength=9,
                )
                sum_sq = float(((y_i[finite] - pred_i[finite]) ** 2).sum())
                count = int(finite.sum())
                for slice_name, active in flags.items():
                    if not active:
                        continue
                    entry = patient_point_stats[slice_name].setdefault(
                        pid_i,
                        {"count": 0, "sum_sq": 0.0, "peg_counts": np.zeros(9, dtype=float)},
                    )
                    entry["count"] += count
                    entry["sum_sq"] += sum_sq
                    entry["peg_counts"] += peg_counts
    return pred_by_seq, event_by_seq, patient_point_stats


def evaluate_model_dir(
    model_dir: Path,
    split_part: str,
    hypo_threshold: float,
    min_event_duration_minutes: int,
    step_minutes: int,
    n_bootstrap: int,
    bootstrap_seed: int,
    slice_threshold_overrides: Optional[Dict[str, Optional[int]]] = None,
    min_context_steps: Optional[int] = None,
) -> Dict[str, Any]:
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

    # Override split part and stride for evaluation
    split_cfg = dict(split_cfg)
    split_cfg["split_part"] = split_part
    dl_cfg = dict(dl_cfg)
    dl_cfg["stride"] = 1
    dl_cfg["shuffle"] = False

    # Build dataloader
    seed = model_data.get("seed", 42)
    dl = make_dataloader_from_cfg(split_cfg, dl_cfg, split_part=str(split_part), seed=seed)

    # Build model
    model_name = model_cfg.get("name") or cfg.get("model")
    if not model_name:
        raise ValueError(f"Missing model name in {model_dir}/config.yaml")
    hist = int(dl_cfg.get("history_len", 288))
    pred_len = int(dl_cfg.get("pred_len", 24))
    input_dim = len(dl.dataset.feature_cols)
    model = build_model_from_cfg({"name": model_name, "params": model_cfg.get("params", {})},
                                 input_dim=input_dim, history_len=hist, pred_len=pred_len)

    # Load checkpoint
    ckpt_path = model_dir / "checkpoint.pt"
    if ckpt_path.exists() and hasattr(model, "load_state_dict"):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = _sanitize_state_dict(ckpt.get("state_dict", None))
        if state:
            try:
                model.load_state_dict(state)
            except Exception:
                pass

    cached_preds = cached_gts = None
    cached_events = None
    preds_path = model_dir / "preds.npz"
    gts_path = model_dir / "GT.npz"
    event_path = model_dir / "event.npz"
    if preds_path.exists() and gts_path.exists():
        preds_npz = np.load(preds_path, allow_pickle=True)
        gts_npz = np.load(gts_path, allow_pickle=True)
        candidate_preds = np.asarray(preds_npz["preds"], dtype=float)
        candidate_gts = np.asarray(gts_npz["gts"], dtype=float)
        if len(candidate_preds) == len(dl.dataset) and len(candidate_gts) == len(dl.dataset):
            cached_preds = candidate_preds
            cached_gts = candidate_gts
            if event_path.exists():
                evt_npz = np.load(event_path, allow_pickle=True)
                cached_events = {k: np.asarray(evt_npz[k]) for k in evt_npz.files}

    # Load sequences and run inference
    sequences = _load_sequences(Path(split_cfg.get("processed_root", "datasets_processed")), dataset_name, split_spec, split_part)
    slice_thresholds = _resolve_slice_thresholds(cfg, overrides=slice_threshold_overrides)
    preds_by_seq, event_by_seq, patient_point_stats = _run_inference(
        model,
        dl,
        pred_len,
        sequences,
        slice_thresholds=slice_thresholds,
        cached_preds=cached_preds,
        cached_gts=cached_gts,
        cached_events=cached_events,
        min_context_steps=min_context_steps,
    )

    min_samples = max(1, int(round(min_event_duration_minutes / step_minutes)))
    pred_len_samples = int(pred_len)

    by_slice: Dict[str, Dict[str, Any]] = {}
    for slice_name in SLICE_ORDER:
        patient_day_counts: Dict[int, float] = {}
        patient_true_events: Dict[int, int] = {}
        patient_detected_events: Dict[int, int] = {}
        patient_false_alarms: Dict[int, int] = {}
        patient_lead_times_samples: Dict[int, List[int]] = {}

        for key, seq in sequences.items():
            cgm = seq["cgm"]
            pred_dict_all = preds_by_seq.get(key, {})
            event_dict = event_by_seq.get(key, {})
            if slice_name == "overall":
                pred_dict = pred_dict_all
            else:
                pred_dict = {
                    idx: pred
                    for idx, pred in pred_dict_all.items()
                    if idx in event_dict
                    and _slice_flags(*event_dict[idx], slice_thresholds).get(slice_name, False)
                }
            pred_indices_sorted = sorted(pred_dict.keys())
            if not pred_indices_sorted:
                continue
            coverage_samples = len(pred_indices_sorted)
            pid = key[0]
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

        slice_point_stats = patient_point_stats.get(slice_name, {})
        pids = sorted(
            set(patient_day_counts)
            | set(patient_true_events)
            | set(patient_detected_events)
            | set(patient_false_alarms)
            | set(patient_lead_times_samples)
            | set(slice_point_stats)
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
                point_count=int(slice_point_stats.get(pid, {}).get("count", 0)),
                point_sum_sq=float(slice_point_stats.get(pid, {}).get("sum_sq", 0.0)),
                point_peg_counts=slice_point_stats.get(pid, {}).get("peg_counts", np.zeros(9, dtype=float)),
            )
            for pid in pids
        ]
        macro = _macro_average_patient_metrics(patient_metrics)
        ci = _bootstrap_patient_metrics(patient_metrics, n_bootstrap=n_bootstrap, seed=bootstrap_seed)
        all_lead_times_samples: List[int] = []
        for pid in pids:
            all_lead_times_samples.extend(patient_lead_times_samples.get(pid, []))
        lead_times_minutes = (
            np.asarray(all_lead_times_samples, dtype=float) * step_minutes
            if all_lead_times_samples
            else np.array([], dtype=float)
        )
        by_slice[slice_name] = {
            "slice": slice_name,
            "model_name": str(model_name),
            "dataset_name": str(dataset_name),
            "prediction_horizon_minutes": int(pred_len * step_minutes),
            "event_recall": float(macro["event_recall"]),
            "event_recall_ci_low": float(ci.get("event_recall", {}).get("ci_low", float("nan"))),
            "event_recall_ci_high": float(ci.get("event_recall", {}).get("ci_high", float("nan"))),
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
            "false_alarms_per_patient_day": float(macro["false_alarms_per_patient_day"]),
            "false_alarms_per_patient_day_ci_low": float(
                ci.get("false_alarms_per_patient_day", {}).get("ci_low", float("nan"))
            ),
            "false_alarms_per_patient_day_ci_high": float(
                ci.get("false_alarms_per_patient_day", {}).get("ci_high", float("nan"))
            ),
            "rmse": float(macro["rmse"]),
            "rmse_ci_low": float(ci.get("rmse", {}).get("ci_low", float("nan"))),
            "rmse_ci_high": float(ci.get("rmse", {}).get("ci_high", float("nan"))),
            "peg_unsafe": float(macro["peg_unsafe"]),
            "peg_unsafe_ci_low": float(ci.get("peg_unsafe", {}).get("ci_low", float("nan"))),
            "peg_unsafe_ci_high": float(ci.get("peg_unsafe", {}).get("ci_high", float("nan"))),
            "num_true_events": float(macro["num_true_events"]),
            "num_true_events_ci_low": float(ci.get("num_true_events", {}).get("ci_low", float("nan"))),
            "num_true_events_ci_high": float(ci.get("num_true_events", {}).get("ci_high", float("nan"))),
            "num_detected_events": float(macro["num_detected_events"]),
            "num_detected_events_ci_low": float(ci.get("num_detected_events", {}).get("ci_low", float("nan"))),
            "num_detected_events_ci_high": float(ci.get("num_detected_events", {}).get("ci_high", float("nan"))),
            "num_false_alarms": float(macro["num_false_alarms"]),
            "num_false_alarms_ci_low": float(ci.get("num_false_alarms", {}).get("ci_low", float("nan"))),
            "num_false_alarms_ci_high": float(ci.get("num_false_alarms", {}).get("ci_high", float("nan"))),
            "num_patient_days_evaluated": float(macro["num_patient_days_evaluated"]),
            "num_patient_days_evaluated_ci_low": float(
                ci.get("num_patient_days_evaluated", {}).get("ci_low", float("nan"))
            ),
            "num_patient_days_evaluated_ci_high": float(
                ci.get("num_patient_days_evaluated", {}).get("ci_high", float("nan"))
            ),
            "n_patients": int(len(patient_metrics)),
            "model_dir": str(model_dir),
            "lead_times_minutes": lead_times_minutes,
            "patient_metrics": patient_metrics,
        }

    overall = dict(by_slice["overall"])
    overall["by_slice"] = by_slice
    return overall


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate hypoglycemia gating metrics for pretrained models.")
    p.add_argument("--models_dir", type=Path, required=True, help="Directory containing model subdirectories.")
    p.add_argument("--split", type=str, default="test", help="Split to evaluate (default: test).")
    p.add_argument("--output_csv", type=Path, default=None, help="Optional output CSV path.")
    p.add_argument("--hypo_threshold", type=float, default=70.0, help="Hypoglycemia threshold (mg/dL).")
    p.add_argument(
        "--min_event_duration_minutes",
        type=int,
        default=15,
        help="Minimum event duration in minutes (default: 15).",
    )
    p.add_argument("--step_minutes", type=int, default=5, help="Sampling interval in minutes (default: 5).")
    p.add_argument("--n_bootstrap", type=int, default=1000, help="Patient-level bootstrap rounds (default: 1000).")
    p.add_argument("--bootstrap_seed", type=int, default=42, help="Bootstrap RNG seed (default: 42).")
    p.add_argument(
        "--post_meal_threshold",
        type=int,
        default=None,
        help="Optional override for the post-meal slice window in 5-minute steps.",
    )
    p.add_argument(
        "--post_bolus_threshold",
        type=int,
        default=None,
        help="Optional override for the post-bolus slice window in 5-minute steps.",
    )
    p.add_argument(
        "--min_context_steps",
        type=int,
        default=None,
        help=(
            "Restrict evaluation to windows whose sequence has at least this many "
            "history samples, so models with different history_len are scored on an "
            "identical window set (288 = the longest history_len in the suite)."
        ),
    )
    p.add_argument(
        "--nocturnal_threshold",
        type=int,
        default=None,
        help="Optional override for the nocturnal slice window in 5-minute steps.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.models_dir)
    slice_threshold_overrides = {
        "post_meal": args.post_meal_threshold,
        "post_bolus": args.post_bolus_threshold,
        "nocturnal": args.nocturnal_threshold,
    }
    # Allow passing a single model directory directly
    if (root / "config.yaml").exists() and (root / "checkpoint.pt").exists():
        model_dirs = [root]
    else:
        model_dirs = [p for p in root.iterdir() if p.is_dir()]
    results = []
    for md in tqdm(sorted(model_dirs), desc="Models", leave=False):
        cfg_path = md / "config.yaml"
        ckpt_path = md / "checkpoint.pt"
        if not cfg_path.exists() or not ckpt_path.exists():
            continue
        try:
            res = evaluate_model_dir(
                md,
                split_part=args.split,
                hypo_threshold=float(args.hypo_threshold),
                min_event_duration_minutes=int(args.min_event_duration_minutes),
                step_minutes=int(args.step_minutes),
                n_bootstrap=int(args.n_bootstrap),
                min_context_steps=args.min_context_steps,
                bootstrap_seed=int(args.bootstrap_seed),
                slice_threshold_overrides=slice_threshold_overrides,
            )
            # Write per-model CSV in model dir
            lead_times = res.get("lead_times_minutes", np.array([], dtype=float))
            patient_metrics = res.get("patient_metrics", [])
            by_slice = res.get("by_slice", {})
            res_out = dict(res)
            res_out.pop("lead_times_minutes", None)
            res_out.pop("patient_metrics", None)
            res_out.pop("by_slice", None)
            for slice_name in SLICE_ORDER:
                slice_row = dict(by_slice.get(slice_name, {}))
                if not slice_row:
                    continue
                slice_row.pop("lead_times_minutes", None)
                slice_row.pop("patient_metrics", None)
                results.append(slice_row)
            out_df = pd.DataFrame([res_out])
            out_df.to_csv(md / "hypo_gating_metrics.csv", index=False)
            by_slice_rows = []
            patient_slice_rows = []
            for slice_name in SLICE_ORDER:
                slice_res = dict(by_slice.get(slice_name, {}))
                if not slice_res:
                    continue
                slice_lead_times = slice_res.pop("lead_times_minutes", np.array([], dtype=float))
                slice_patient_metrics = slice_res.pop("patient_metrics", [])
                by_slice_rows.append(slice_res)
                for pm in slice_patient_metrics:
                    row = dict(pm)
                    row["slice"] = slice_name
                    patient_slice_rows.append(row)
                pd.DataFrame({"lead_time_minutes": slice_lead_times}).to_csv(
                    md / f"hypo_gating_lead_times_{slice_name}.csv", index=False
                )
            pd.DataFrame(by_slice_rows).to_csv(md / "hypo_gating_metrics_by_slice.csv", index=False)
            pd.DataFrame(patient_slice_rows).to_csv(md / "hypo_gating_patient_metrics_by_slice.csv", index=False)
            pd.DataFrame(patient_metrics).to_csv(md / "hypo_gating_patient_metrics.csv", index=False)
            # Save all lead times for detected events (one per line)
            pd.DataFrame({"lead_time_minutes": lead_times}).to_csv(md / "hypo_gating_lead_times.csv", index=False)
        except Exception as e:
            for slice_name in SLICE_ORDER:
                results.append(
                    {
                        "slice": slice_name,
                        "model_name": md.name,
                        "dataset_name": "unknown",
                        "prediction_horizon_minutes": math.nan,
                        "event_recall": math.nan,
                        "median_warning_lead_time_minutes": math.nan,
                        "mean_warning_lead_time_minutes": math.nan,
                        "false_alarms_per_patient_day": math.nan,
                        "rmse": math.nan,
                        "peg_unsafe": math.nan,
                        "num_true_events": 0,
                        "num_detected_events": 0,
                        "num_false_alarms": 0,
                        "num_patient_days_evaluated": 0.0,
                        "model_dir": str(md),
                        "error": str(e),
                    }
                )

    df = pd.DataFrame(results)
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)

    cols = [
        "slice",
        "model_name",
        "dataset_name",
        "prediction_horizon_minutes",
        "event_recall",
        "median_warning_lead_time_minutes",
        "mean_warning_lead_time_minutes",
        "false_alarms_per_patient_day",
        "rmse",
        "peg_unsafe",
        "num_true_events",
        "num_detected_events",
        "num_false_alarms",
        "num_patient_days_evaluated",
    ]
    for slice_name in SLICE_ORDER:
        sdf = df[df["slice"] == slice_name]
        if sdf.empty:
            continue
        print(f"\n=== {slice_name.upper()} ===")
        print(sdf[cols].to_string(index=False))


if __name__ == "__main__":
    main()
