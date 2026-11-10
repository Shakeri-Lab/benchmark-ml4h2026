"""Preprocessor for cohorts distributed in the DIAX per-subject JSON format.

DIAX ships one JSON per subject with a uniform schema::

    {"cgm":        {"time": [...], "value": [...]},   # mg/dL
     "basal_rate": {"time": [...], "value": [...]},   # U/hr, piecewise-constant
     "bolus":      {"time": [...], "value": [...]},   # U, sparse events
     "meal"/"carb":{"time": [...], "value": [...]},   # g, sparse events (optional)
     "weight":     {"date": [...], "value": [...]},   # kg, occasional
     "metadata":   {...}}

Because the schema is shared, this module serves every DIAX cohort; only the
folder name changes. Harmonization deliberately mirrors ``preprocessing/dclp3.py``
(15 s dedup, 30 min sequence gap, 5 min grid, 3 h basal lookback, 5 min bolus
bin, 12 h no-bolus filter, 312-step minimum) so cohorts stay comparable.

One intentional deviation: DCLP3 attaches the *nearest* bolus to each CGM
timestamp, which silently drops insulin when two boluses fall in the same 5 min
bin. Here boluses are *summed* within the bin, and the number of affected bins is
recorded in ``metadata.json`` under ``bolus_bins_with_multiple_events`` so the
difference is auditable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from preprocessing.common import ensure_io


@dataclass(frozen=True)
class DiaxConfig:
    dedup_window_seconds: int = 15
    sequence_gap_seconds: int = 1800  # 30 minutes
    resample_freq_minutes: int = 5
    basal_window_before_seconds: int = 10800  # 3 hours
    basal_window_after_seconds: int = 15
    bolus_window_before_seconds: int = 285
    bolus_window_after_seconds: int = 15
    max_bolus_gap_seconds: int = 43200  # 12 hours
    min_sequence_steps: int = 312  # 24 h history + 2 h horizon at 5 min
    cgm_min_mgdl: float = 20.0
    cgm_max_mgdl: float = 450.0
    # Physiological validity bounds. The DIAX export carries occasional corrupt
    # values -- 16 IOBP2 subjects have a literal `inf` basal_rate, and the column
    # also contains negatives and rates up to 730 U/hr against a p99 of 4.3.
    # Out-of-range samples are marked missing (not clipped) so the carry-forward
    # in _attach_basal falls back to the last physiologically valid rate.
    basal_max_u_per_hr: float = 15.0
    bolus_max_u: float = 50.0
    # Subjects whose metadata.subject.treatment_group matches are dropped.
    # The IOBP2 export labels every subject control_method="AID", but the
    # trial randomised 2:1 against standard care; the 11 "Control" subjects
    # bolus under user control, not under the iLet, and pooling them would
    # mix two dosing regimes in a cohort used to test controller transfer.
    exclude_treatment_groups: tuple = ()


# Channel aliases seen across DIAX cohorts.
_BASAL_KEYS = ("basal_rate", "basal")
_BOLUS_KEYS = ("bolus", "bolus_insulin")
_MEAL_KEYS = ("meal", "carbs", "carb", "carbohydrate")


def _first_present(d: dict, keys: Tuple[str, ...]) -> str | None:
    for k in keys:
        if isinstance(d.get(k), dict) and d[k].get("time"):
            return k
    return None


def _channel_frame(d: dict, key: str | None, value_name: str) -> pd.DataFrame:
    """Extract one time/value channel as a tidy frame, dropping non-numeric rows."""
    cols = ["date", value_name]
    if not key or key not in d:
        return pd.DataFrame(columns=cols)
    ch = d[key]
    t = ch.get("time") or []
    v = ch.get("value") or ch.get("values") or []
    n = min(len(t), len(v))
    if n == 0:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame({"date": pd.to_datetime(t[:n], errors="coerce"),
                       value_name: pd.to_numeric(v[:n], errors="coerce")})
    return df.dropna().sort_values("date").reset_index(drop=True)


def _numeric_pat_id(source_id, raw_id: str) -> int | None:
    """Derive an integer patient id from the DIAX identifiers."""
    for cand in (source_id, raw_id.rsplit("_", 1)[-1] if "_" in raw_id else raw_id):
        try:
            return int(str(cand).strip())
        except (TypeError, ValueError):
            continue
    return None


def _sanitize(df: pd.DataFrame, col: str, lo: float, hi: float) -> Tuple[pd.DataFrame, int]:
    """Drop non-finite and out-of-range samples, returning the count removed."""
    if df.empty:
        return df, 0
    v = pd.to_numeric(df[col], errors="coerce")
    ok = np.isfinite(v) & (v >= lo) & (v <= hi)
    return df.loc[ok].reset_index(drop=True), int((~ok).sum())


def _weight_frame(d: dict) -> pd.DataFrame:
    w = d.get("weight")
    if not isinstance(w, dict):
        return pd.DataFrame(columns=["date", "weight_kg"])
    dates = w.get("date") or w.get("time") or []
    vals = w.get("value") or []
    n = min(len(dates), len(vals))
    if n == 0:
        return pd.DataFrame(columns=["date", "weight_kg"])
    df = pd.DataFrame({"date": pd.to_datetime(dates[:n], errors="coerce"),
                       "weight_kg": pd.to_numeric(vals[:n], errors="coerce")})
    return df.dropna().sort_values("date").reset_index(drop=True)


def _dedup_within_window(df: pd.DataFrame, window_seconds: int) -> pd.DataFrame:
    """Keep the latest reading within each run of samples closer than the window."""
    if df.empty:
        return df
    g = df.sort_values("date")
    deltas = g["date"].diff().dt.total_seconds()
    new_group = deltas.isna() | (deltas > window_seconds)
    return g.groupby(new_group.cumsum()).tail(1).reset_index(drop=True)


def _resample_and_interpolate(df: pd.DataFrame, cfg: DiaxConfig) -> pd.DataFrame:
    """Split on gaps > sequence_gap_seconds, then put each segment on a 5 min grid."""
    if df.empty:
        return pd.DataFrame(columns=["date", "cgm"])
    max_gap = pd.Timedelta(seconds=cfg.sequence_gap_seconds)
    freq = f"{cfg.resample_freq_minutes}min"
    g = df.sort_values("date")
    boundaries = np.where((g["date"].diff() > max_gap).to_numpy())[0].tolist() + [len(g)]
    parts: List[pd.DataFrame] = []
    start = 0
    for end in boundaries:
        seg = g.iloc[start:end]
        start = end
        if seg.empty:
            continue
        res = seg.set_index("date").sort_index()[["cgm"]].resample(freq).mean()
        res["cgm"] = res["cgm"].interpolate(method="time")
        res = res.dropna(subset=["cgm"]).reset_index()
        if not res.empty:
            parts.append(res)
    if not parts:
        return pd.DataFrame(columns=["date", "cgm"])
    return pd.concat(parts, ignore_index=True)


def _attach_basal(cgm: pd.DataFrame, basal: pd.DataFrame, cfg: DiaxConfig) -> np.ndarray:
    """Carry the most recent basal rate forward, within a bounded lookback."""
    out = np.full(len(cgm), np.nan)
    if cgm.empty or basal.empty:
        return out
    bt = basal["date"].values.astype("datetime64[ns]").view("int64")
    bv = basal["basal"].to_numpy(dtype=float)
    ct = cgm["date"].values.astype("datetime64[ns]").view("int64")
    lower = ct - cfg.basal_window_before_seconds * 1_000_000_000
    upper = ct + cfg.basal_window_after_seconds * 1_000_000_000
    start = np.searchsorted(bt, lower, side="left")
    end = np.searchsorted(bt, upper, side="right")
    chosen = np.where(start < end, end - 1, -1)
    return np.where(chosen >= 0, bv[np.clip(chosen, 0, len(bv) - 1)], np.nan)


def _attach_event_sum(cgm: pd.DataFrame, events: pd.DataFrame, value_col: str,
                      cfg: DiaxConfig) -> Tuple[np.ndarray, int]:
    """Sum sparse events falling in each CGM sample's 5 min bin.

    The bin is (t - bolus_window_before, t + bolus_window_after], exactly one
    resample period wide, so no event is counted for two consecutive samples.
    """
    if cgm.empty or events.empty:
        return np.zeros(len(cgm)), 0
    et = events["date"].values.astype("datetime64[ns]").view("int64")
    ev = events[value_col].to_numpy(dtype=float)
    ct = cgm["date"].values.astype("datetime64[ns]").view("int64")
    lower = ct - cfg.bolus_window_before_seconds * 1_000_000_000
    upper = ct + cfg.bolus_window_after_seconds * 1_000_000_000
    start = np.searchsorted(et, lower, side="right")
    end = np.searchsorted(et, upper, side="right")
    cs = np.concatenate([[0.0], np.cumsum(ev)])
    totals = cs[end] - cs[start]
    multi_bins = int(np.sum((end - start) > 1))
    return totals, multi_bins


def _attach_weight(cgm: pd.DataFrame, weight: pd.DataFrame) -> np.ndarray:
    if cgm.empty:
        return np.array([])
    if weight.empty:
        return np.full(len(cgm), np.nan)
    s = weight.set_index("date")["weight_kg"]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    idx = pd.DatetimeIndex(cgm["date"])
    joined = s.reindex(s.index.union(idx)).interpolate(method="time").ffill().bfill()
    return joined.reindex(idx).to_numpy(dtype=float)


def _filter_long_bolus_gaps(df: pd.DataFrame, cfg: DiaxConfig) -> pd.DataFrame:
    """Drop spans that run longer than max_bolus_gap without any bolus."""
    if df.empty:
        return df
    th = cfg.max_bolus_gap_seconds * 1_000_000_000
    t = df["date"].values.astype("datetime64[ns]").view("int64")
    has_bolus = (df["bolus_standard"].to_numpy() + df["bolus_extended"].to_numpy()) != 0
    last = np.full(len(df), -1, dtype="int64")
    cur = -1
    for i in range(len(df)):
        if has_bolus[i]:
            cur = t[i]
        last[i] = cur
    keep = (last >= 0) & ((t - last) <= th)
    return df.loc[keep].reset_index(drop=True)


def _assign_sequences(df: pd.DataFrame, cfg: DiaxConfig, next_seq: int) -> Tuple[pd.DataFrame, int]:
    """Split into contiguous 5 min runs and drop runs shorter than the minimum."""
    if df.empty:
        return df, next_seq
    step = pd.Timedelta(minutes=cfg.resample_freq_minutes)
    d = df.sort_values("date").reset_index(drop=True)
    breaks = (d["date"].diff() != step).to_numpy()
    breaks[0] = True
    local = np.cumsum(breaks)
    d["_local_seq"] = local
    keep_parts = []
    for _, g in d.groupby("_local_seq", sort=True):
        if len(g) < cfg.min_sequence_steps:
            continue
        g = g.copy()
        g["seq_id"] = next_seq
        next_seq += 1
        keep_parts.append(g)
    if not keep_parts:
        return pd.DataFrame(columns=list(df.columns) + ["seq_id"]), next_seq
    out = pd.concat(keep_parts, ignore_index=True).drop(columns=["_local_seq"])
    return out, next_seq


def preprocess_diax(raw_dir: Path, output_dir: Path, dataset_name: str,
                    cfg: DiaxConfig | None = None) -> None:
    cfg = cfg or DiaxConfig()
    raw_dir, output_dir = ensure_io(raw_dir, output_dir)

    files = sorted(raw_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No subject JSON files found under {raw_dir}")

    frames: List[pd.DataFrame] = []
    next_seq = 1
    stats = {
        "subjects_seen": 0,
        "subjects_kept": 0,
        "subjects_no_cgm": 0,
        "subjects_no_sequence": 0,
        "bolus_bins_with_multiple_events": 0,
        "meal_channel_present": False,
        "carb_category_only": False,
        "subjects_without_weight": 0,
        "basal_samples_dropped_invalid": 0,
        "bolus_samples_dropped_invalid": 0,
        "subjects_with_invalid_basal": 0,
        "subjects_unparseable_id": 0,
        "subjects_excluded_treatment_group": 0,
    }
    id_map: Dict[int, str] = {}
    control_methods: Dict[str, int] = {}
    cohorts: Dict[str, int] = {}

    for f in tqdm(files, desc=f"Preprocess {dataset_name}"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        stats["subjects_seen"] += 1

        subject_meta = (d.get("metadata") or {}).get("subject") or {}
        tgroup = subject_meta.get("treatment_group")
        if cfg.exclude_treatment_groups and tgroup in cfg.exclude_treatment_groups:
            stats["subjects_excluded_treatment_group"] += 1
            continue
        # The shared dataloader packs pat_id into a numeric array alongside the
        # feature columns, so a string id (DIAX uses "<STUDY>_<n>") makes the
        # whole array object-dtype and breaks window indexing. Emit the numeric
        # part, matching DCLP3/DCLP5/PEDAP, and keep the full id in metadata.
        raw_id = str(d.get("subject_id") or d.get("unique_id") or f.stem)
        src_id = d.get("source_subject_id")
        pat_id = _numeric_pat_id(src_id, raw_id)
        if pat_id is None:
            stats["subjects_unparseable_id"] += 1
            continue
        id_map[pat_id] = raw_id

        cm = d.get("control_method")
        if cm:
            control_methods[cm] = control_methods.get(cm, 0) + 1
        ch = d.get("cohort")
        if ch:
            cohorts[ch] = cohorts.get(ch, 0) + 1
        if isinstance(d.get("carb_category"), dict) and d["carb_category"].get("time"):
            stats["carb_category_only"] = True

        cgm = _channel_frame(d, "cgm", "cgm")
        cgm = cgm[(cgm["cgm"] >= cfg.cgm_min_mgdl) & (cgm["cgm"] <= cfg.cgm_max_mgdl)]
        if cgm.empty:
            stats["subjects_no_cgm"] += 1
            continue
        cgm = _dedup_within_window(cgm, cfg.dedup_window_seconds)
        cgm = _resample_and_interpolate(cgm, cfg)
        if cgm.empty:
            stats["subjects_no_cgm"] += 1
            continue

        basal_key = _first_present(d, _BASAL_KEYS)
        bolus_key = _first_present(d, _BOLUS_KEYS)
        meal_key = _first_present(d, _MEAL_KEYS)
        if meal_key:
            stats["meal_channel_present"] = True

        basal = _channel_frame(d, basal_key, "basal")
        bolus = _channel_frame(d, bolus_key, "bolus")
        meal = _channel_frame(d, meal_key, "meal")

        basal, n_bad_basal = _sanitize(basal, "basal", 0.0, cfg.basal_max_u_per_hr)
        bolus, n_bad_bolus = _sanitize(bolus, "bolus", 0.0, cfg.bolus_max_u)
        stats["basal_samples_dropped_invalid"] += n_bad_basal
        stats["bolus_samples_dropped_invalid"] += n_bad_bolus
        if n_bad_basal:
            stats["subjects_with_invalid_basal"] += 1

        cgm["basal"] = _attach_basal(cgm, basal, cfg)
        bolus_vals, multi = _attach_event_sum(cgm, bolus, "bolus", cfg)
        stats["bolus_bins_with_multiple_events"] += multi
        cgm["bolus_standard"] = bolus_vals
        # DIAX does not distinguish extended boluses; the column is retained for
        # schema compatibility with the Jaeb cohorts.
        cgm["bolus_extended"] = 0.0
        meal_vals, _ = _attach_event_sum(cgm, meal, "meal", cfg)
        cgm["meal"] = meal_vals

        w = _attach_weight(cgm, _weight_frame(d))
        if np.all(~np.isfinite(w)):
            stats["subjects_without_weight"] += 1
        cgm["weight_kg"] = w

        cgm["basal"] = cgm["basal"].fillna(0.0)
        cgm = _filter_long_bolus_gaps(cgm, cfg)
        if cgm.empty:
            stats["subjects_no_sequence"] += 1
            continue

        cgm["pat_id"] = pat_id
        cgm, next_seq = _assign_sequences(cgm, cfg, next_seq)
        if cgm.empty:
            stats["subjects_no_sequence"] += 1
            continue

        stats["subjects_kept"] += 1
        frames.append(cgm)

    if not frames:
        raise RuntimeError(f"{dataset_name}: no subject produced a usable sequence")

    final_df = pd.concat(frames, ignore_index=True)
    final_df = final_df[
        ["pat_id", "seq_id", "date", "cgm", "basal", "bolus_standard",
         "bolus_extended", "weight_kg", "meal"]
    ].sort_values(["pat_id", "seq_id", "date"]).reset_index(drop=True)

    n_days = len(final_df) / 288.0
    metadata = {
        "dataset": dataset_name,
        "source_format": "DIAX per-subject JSON",
        "source_dir": str(raw_dir),
        "config": asdict(cfg),
        "channels": ["cgm", "basal", "bolus_standard", "bolus_extended", "weight_kg", "meal"],
        "units": {"cgm": "mg/dL", "basal": "U/hr", "bolus_standard": "U",
                  "bolus_extended": "U", "weight_kg": "kg", "meal": "g"},
        "patients": int(final_df["pat_id"].nunique()),
        "sequences": int(final_df["seq_id"].nunique()),
        "rows": int(len(final_df)),
        "approx_patient_days": round(n_days, 1),
        "cgm_mean_mgdl": round(float(final_df["cgm"].mean()), 3),
        "basal_u_per_day": round(float(final_df["basal"].sum() / n_days / 12.0), 3),
        "bolus_u_per_day": round(
            float((final_df["bolus_standard"] + final_df["bolus_extended"]).sum() / n_days), 3),
        "rows_with_bolus": int((final_df["bolus_standard"] != 0).sum()),
        "rows_with_meal": int((final_df["meal"] != 0).sum()),
        "patient_id_map": {str(k): v for k, v in sorted(id_map.items())},
        "excluded_treatment_groups": list(cfg.exclude_treatment_groups),
        "control_methods": control_methods,
        "cohorts": cohorts,
        **stats,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_dir / "timeseries.csv", index=False)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[+] {dataset_name}: {metadata['patients']} patients, "
          f"{metadata['sequences']} sequences, {metadata['rows']} rows "
          f"({metadata['approx_patient_days']} patient-days)")


def preprocess_iobp2(raw_dir: Path, output_dir: Path) -> None:
    """IOBP2 (iLet bionic pancreas). Meal announcements are categorical
    ('Less'/'Typical'/'More') with no gram amounts, so ``meal`` stays zero -
    matching DCLP3/DCLP5, which also carry no carbohydrate channel."""
    preprocess_diax(raw_dir, output_dir, "IOBP2",
                    DiaxConfig(exclude_treatment_groups=("Control",)))
