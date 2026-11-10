from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas._libs.tslibs import parsing
from tqdm.auto import tqdm

from preprocessing.common import ensure_io


@dataclass(frozen=True)
class DCLP5Config:
    dedup_window_seconds: int = 15   # keep latest within ~15 seconds
    sequence_gap_seconds: int = 1800  # 30 minutes gap splits sequences
    resample_freq_minutes: int = 5
    interp_min_gap_seconds: int = 420  # >7 minutes triggers interpolation during resample (informational)
    basal_window_before_seconds: int = 10800  # 3 hours before
    basal_window_after_seconds: int = 15      # slightly after
    bolus_window_before_seconds: int = 285
    bolus_window_after_seconds: int = 15
    max_bolus_gap_seconds: int = 43200  # 12 hours; drop data after this until next bolus
    min_sequence_steps: int = 312  # 24h history (288) + 2h horizon (24) at 5-min steps
    weight_columns: Tuple[str, ...] = ("weight_kg",)


def _parse_datetime(series: pd.Series) -> pd.Series:
    """Infer format from the first non-null value, then parse entire column with that format."""
    sample = next((x for x in series.dropna().astype(str) if x.strip()), None)
    if sample:
        fmt = parsing.guess_datetime_format(sample)
        if fmt:
            parsed = pd.to_datetime(series, format=fmt, errors="coerce")
            return parsed
    return pd.to_datetime(series, errors="coerce")


def _read_tandem_primary(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "TandemCGMDATAGXB.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing CGM file: {path}")
    df = pd.read_csv(path, sep="|", low_memory=False)
    df = df.rename(
        columns={
            "PtID": "pat_id",
            "DataDtTm": "date",
            "currentGlucoseDisplayValue": "cgm",
        }
    )[["pat_id", "date", "cgm"]]
    df["date"] = _parse_datetime(df["date"])
    df["source"] = "TandemCGMDATAGXB"
    return df


def _read_tandem_adjusted(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "DCLP5TandemCGMDATAGXB_b.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing CGM file: {path}")
    df = pd.read_csv(path, sep="|", low_memory=False)
    df = df.rename(
        columns={
            "PtID": "pat_id",
            "DataDtTm_adjusted": "date",
            "CGMValue": "cgm",
        }
    )[["pat_id", "date", "cgm"]]
    df["date"] = _parse_datetime(df["date"])
    df["source"] = "TandemCGMDATAGXB_b"
    return df


def _read_dexcom_clarity(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "DexcomClarityCGM.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing CGM file: {path}")
    df = pd.read_csv(path, sep="|", low_memory=False)
    df = df.rename(
        columns={
            "PtID": "pat_id",
            "DataDtTm": "date",
            "CGM": "cgm",
        }
    )[["pat_id", "date", "cgm"]]
    df["date"] = _parse_datetime(df["date"])
    df["source"] = "DexcomClarityCGM"
    return df


def _combine_sources(raw_dir: Path) -> Tuple[pd.DataFrame, int]:
    frames: List[pd.DataFrame] = [
        _read_tandem_primary(raw_dir),
        _read_tandem_adjusted(raw_dir),
        _read_dexcom_clarity(raw_dir),
    ]
    raw_total = sum(len(frame) for frame in frames)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined["pat_id"] = combined["pat_id"].astype(str).str.strip()
    combined["cgm"] = pd.to_numeric(combined["cgm"], errors="coerce")
    valid = combined.dropna(subset=["pat_id", "date", "cgm"])
    return valid, raw_total


def _dedup_within_window(df: pd.DataFrame, window_seconds: int = 10) -> pd.DataFrame:
    """
    Within each patient, keep only the latest record in any <= window window.
    Vectorized approach per patient with a progress bar.
    """
    if df.empty:
        return df

    df_sorted = df.sort_values(["pat_id", "date"])
    grouped = df_sorted.groupby("pat_id", sort=False)
    total_groups = len(grouped)

    deduped_parts: List[pd.DataFrame] = []
    for _, g in tqdm(grouped, total=total_groups, desc="Dedup DCLP5 patients"):
        deltas = g["date"].diff().dt.total_seconds()
        new_group = (deltas.isna()) | (deltas > window_seconds)
        group_ids = new_group.cumsum()
        deduped_parts.append(g.groupby(group_ids).tail(1))

    return pd.concat(deduped_parts, ignore_index=True)


def _resample_and_interpolate_cgm(
    df: pd.DataFrame, cfg: DCLP5Config
) -> Tuple[pd.DataFrame, int]:
    """
    Split on gaps > sequence_gap_seconds, resample each segment to fixed freq,
    and interpolate within segments. Returns resampled dataframe and number of segments.
    """
    resampled_parts: List[pd.DataFrame] = []
    max_gap = pd.Timedelta(seconds=cfg.sequence_gap_seconds)
    freq = f"{cfg.resample_freq_minutes}min"
    segment_count = 0

    for pat_id, group in df.groupby("pat_id", sort=False):
        group_sorted = group.sort_values("date")
        gaps = group_sorted["date"].diff() > max_gap
        boundaries = np.where(gaps)[0].tolist() + [len(group_sorted)]
        start = 0
        for end in boundaries:
            segment = group_sorted.iloc[start:end]
            start = end
            if segment.empty:
                continue
            seg_resampled = segment.set_index("date").sort_index()[["cgm"]].resample(freq).mean()
            seg_resampled["cgm"] = seg_resampled["cgm"].interpolate(method="time")
            seg_resampled["pat_id"] = pat_id
            seg_resampled["date"] = seg_resampled.index
            seg_resampled = seg_resampled.dropna(subset=["cgm"])
            if not seg_resampled.empty:
                resampled_parts.append(seg_resampled.reset_index(drop=True))
                segment_count += 1

    if not resampled_parts:
        return pd.DataFrame(columns=["pat_id", "date", "cgm"]), 0

    resampled = pd.concat(resampled_parts, ignore_index=True)
    return resampled[["pat_id", "date", "cgm"]], segment_count


def _read_weight_sources(raw_dir: Path, cgm_min_dates: Dict[str, pd.Timestamp]) -> pd.DataFrame:
    records: List[Dict[str, object]] = []

    def add_records(path: Path, date_col: str | None, weight_col: str, unit_col: str):
        if not path.exists():
            return
        df = pd.read_csv(path, sep="|", low_memory=False)
        if "PtID" not in df.columns or weight_col not in df.columns:
            return
        df = df.rename(columns={"PtID": "pat_id", weight_col: "weight", unit_col: "units"})
        if date_col and date_col in df.columns:
            df["date"] = _parse_datetime(df[date_col])
        else:
            df["date"] = pd.NaT
        df = df[["pat_id", "date", "weight", "units"]]
        df["pat_id"] = df["pat_id"].astype(str).str.strip()
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
        df["units"] = df["units"].astype(str).str.lower()
        for _, row in df.dropna(subset=["weight"]).iterrows():
            records.append(row.to_dict())

    # Follow-up visits with dates
    add_records(raw_dir / "FollowUpCTV.txt", "CGMUploadedDt", "Weight", "WeightUnits")
    # Screening baseline: no timestamp; anchor to earliest CGM date for that patient
    if (raw_dir / "DiabScreening.txt").exists():
        scr = pd.read_csv(raw_dir / "DiabScreening.txt", sep="|", low_memory=False)
        if "PtID" in scr.columns and "Weight" in scr.columns:
            scr = scr.rename(columns={"PtID": "pat_id", "Weight": "weight", "WeightUnits": "units"})
            scr["pat_id"] = scr["pat_id"].astype(str).str.strip()
            scr["weight"] = pd.to_numeric(scr["weight"], errors="coerce")
            scr["units"] = scr["units"].astype(str).str.lower()
            for _, row in scr.dropna(subset=["weight"]).iterrows():
                pat = row["pat_id"]
                anchor = cgm_min_dates.get(pat)
                if anchor is None:
                    continue
                records.append({"pat_id": pat, "date": anchor, "weight": row["weight"], "units": row["units"]})

    # Physical exam weights: no explicit datetime, so anchor to earliest CGM date and order by RecID
    phys_path = raw_dir / "DiabPhysExam.txt"
    if phys_path.exists():
        phys = pd.read_csv(phys_path, sep="|", low_memory=False)
        if {"PtID", "Weight"}.issubset(phys.columns):
            phys = phys.rename(columns={"PtID": "pat_id", "Weight": "weight", "WeightUnits": "units"})
            phys["pat_id"] = phys["pat_id"].astype(str).str.strip()
            phys["weight"] = pd.to_numeric(phys["weight"], errors="coerce")
            phys["units"] = phys.get("units", phys.get("WeightUnits", pd.Series(index=phys.index))).astype(str).str.lower()
            for pat, grp in phys.dropna(subset=["weight"]).groupby("pat_id"):
                anchor = cgm_min_dates.get(pat)
                if anchor is None:
                    continue
                grp_sorted = grp.sort_values("RecID") if "RecID" in grp.columns else grp
                for offset, (_, row) in enumerate(grp_sorted.iterrows()):
                    records.append(
                        {
                            "pat_id": pat,
                            "date": anchor + pd.Timedelta(days=offset),
                            "weight": row["weight"],
                            "units": row["units"],
                        }
                    )

    # Adverse events occasionally contain weight with dates
    adv_path = raw_dir / "AdvEvent.txt"
    if adv_path.exists():
        adv = pd.read_csv(adv_path, sep="|", low_memory=False)
        date_col = None
        for cand in ["AENotifiedDt", "AEOnsetDt", "AEResDt"]:
            if cand in adv.columns:
                date_col = cand
                break
        if {"PtID", "Weight"}.issubset(adv.columns) and date_col:
            adv = adv.rename(columns={"PtID": "pat_id", "Weight": "weight"})
            adv["pat_id"] = adv["pat_id"].astype(str).str.strip()
            adv["weight"] = pd.to_numeric(adv["weight"], errors="coerce")
            adv["date"] = _parse_datetime(adv[date_col])
            adv = adv.dropna(subset=["weight", "date"])
            for _, row in adv.iterrows():
                records.append({"pat_id": row["pat_id"], "date": row["date"], "weight": row["weight"], "units": "kg"})

    if not records:
        return pd.DataFrame(columns=["pat_id", "date", "weight_kg"])

    weight_df = pd.DataFrame(records)
    def to_kg(w, units):
        if pd.isna(w):
            return np.nan
        if isinstance(units, str) and "lb" in units:
            return float(w) * 0.45359237
        return float(w)

    weight_df["weight_kg"] = weight_df.apply(lambda r: to_kg(r["weight"], r["units"]), axis=1)
    weight_df = weight_df.dropna(subset=["pat_id", "date", "weight_kg"])
    weight_df = weight_df[["pat_id", "date", "weight_kg"]]
    return weight_df


def _attach_weight(
    cgm_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    interpolate: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Attach weight_kg to each CGM row using closest weight; if interpolate and bounding weights exist,
    linearly interpolate between them. If no weight for patient, leaves NaN.
    """
    cgm_df = cgm_df.copy()
    cgm_df["weight_kg"] = pd.NA
    stats = {"rows_with_weight": 0, "rows_weight_interpolated": 0}
    if cgm_df.empty or weight_df.empty:
        return cgm_df, stats

    for pat_id, group in cgm_df.groupby("pat_id", sort=False):
        w_group = weight_df[weight_df["pat_id"] == pat_id].sort_values("date")
        if w_group.empty:
            continue
        times = w_group["date"].values.astype("datetime64[ns]")
        weights = w_group["weight_kg"].to_numpy(dtype=float)
        times_ns = times.view("int64")

        def find_weight(ts: pd.Timestamp):
            ts_ns = ts.value
            idx = times_ns.searchsorted(ts_ns)
            if idx == 0:
                return weights[0], False
            if idx == len(times_ns):
                return weights[-1], False
            if not interpolate:
                # choose nearest
                before_diff = ts_ns - times_ns[idx - 1]
                after_diff = times_ns[idx] - ts_ns
                return (weights[idx - 1] if before_diff <= after_diff else weights[idx]), False
            # interpolate
            t0, t1 = times_ns[idx - 1], times_ns[idx]
            w0, w1 = weights[idx - 1], weights[idx]
            if t1 == t0:
                return w1, False
            frac = (ts_ns - t0) / (t1 - t0)
            return w0 + frac * (w1 - w0), True

        weight_vals = []
        interp_flags = []
        for ts in group["date"]:
            w, interp_flag = find_weight(ts)
            weight_vals.append(w)
            interp_flags.append(interp_flag)
        stats["rows_with_weight"] += len(group)
        stats["rows_weight_interpolated"] += sum(interp_flags)
        cgm_df.loc[group.index, "weight_kg"] = weight_vals

    return cgm_df, stats


def _read_basal(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "DCLP5TandemBASALRATECHG_b.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing basal file: {path}")
    df = pd.read_csv(path, sep="|", low_memory=False)
    has_adjusted = "DataDtTm_adjusted" in df.columns
    df = df.rename(
        columns={
            "PtID": "pat_id",
            "CommandedBasalRate": "basal",
            "DataDtTm": "date_raw",
        }
    )
    df["date"] = pd.NaT
    if has_adjusted:
        df["date"] = _parse_datetime(df["DataDtTm_adjusted"])
        # Fallback to raw timestamp if adjusted is missing for that row
        fallback = _parse_datetime(df["date_raw"])
        df["date"] = df["date"].fillna(fallback)
    else:
        df["date"] = _parse_datetime(df["date_raw"])

    df = df[["pat_id", "date", "basal"]]
    df["pat_id"] = df["pat_id"].astype(str).str.strip()
    df["basal"] = pd.to_numeric(df["basal"], errors="coerce")
    return df.dropna(subset=["pat_id", "date", "basal"])


def _attach_basal(
    cgm_df: pd.DataFrame,
    basal_df: pd.DataFrame,
    window_before_seconds: int,
    window_after_seconds: int,
) -> Tuple[pd.DataFrame, int]:
    """
    For each CGM row, attach the last basal rate within [-window_before, +window_after] seconds.
    Returns updated CGM frame and matched count.
    """
    if cgm_df.empty or basal_df.empty:
        cgm_df["basal"] = pd.NA
        return cgm_df, 0

    cgm_df = cgm_df.copy()
    cgm_df["basal"] = pd.NA

    matched = 0
    for pat_id, cgm_group in tqdm(
        cgm_df.groupby("pat_id", sort=False),
        total=cgm_df["pat_id"].nunique(),
        desc="Attach basal to CGM",
    ):
        basal_group = basal_df[basal_df["pat_id"] == pat_id].sort_values("date")
        if basal_group.empty:
            continue
        basal_times = basal_group["date"].values.astype("datetime64[ns]")
        basal_times_ns = basal_times.view("int64")
        basal_vals = basal_group["basal"].to_numpy()

        def find_basal(ts: pd.Timestamp):
            ts_ns = ts.value
            lower = ts_ns - window_before_seconds * 1_000_000_000
            upper = ts_ns + window_after_seconds * 1_000_000_000
            start = basal_times_ns.searchsorted(lower, side="left")
            end = basal_times_ns.searchsorted(upper, side="right")
            if start >= end:
                return pd.NA
            # pick the latest basal within the window (max timestamp)
            return basal_vals[end - 1]

        basal_match = cgm_group["date"].apply(find_basal)
        matched += basal_match.notna().sum()
        cgm_df.loc[cgm_group.index, "basal"] = basal_match.values

    return cgm_df, matched


def _read_bolus(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "DCLP5TandemBolus_Completed_Combined_b.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing bolus file: {path}")
    df = pd.read_csv(path, sep="|", low_memory=False)
    has_adjusted = "DataDtTm_adjusted" in df.columns
    df = df.rename(
        columns={
            "PtID": "pat_id",
            "BolusAmount": "bolus",
            "BolusType": "bolus_type",
            "DataDtTm": "date_raw",
        }
    )
    df["date"] = pd.NaT
    if has_adjusted:
        df["date"] = _parse_datetime(df["DataDtTm_adjusted"])
        fallback = _parse_datetime(df["date_raw"])
        df["date"] = df["date"].fillna(fallback)
    else:
        df["date"] = _parse_datetime(df["date_raw"])

    df = df[["pat_id", "date", "bolus", "bolus_type"]]
    df["pat_id"] = df["pat_id"].astype(str).str.strip()
    df["bolus"] = pd.to_numeric(df["bolus"], errors="coerce")
    df["bolus_type"] = df["bolus_type"].fillna("Unknown")
    return df.dropna(subset=["pat_id", "date", "bolus"])


def _attach_bolus(
    cgm_df: pd.DataFrame,
    bolus_df: pd.DataFrame,
    window_before_seconds: int,
    window_after_seconds: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    For each CGM row, attach the nearest bolus amount within [-window_before, +window_after] seconds.
    Defaults to 0 when no bolus is found. Tracks matched counts by bolus type.
    """
    cgm_df = cgm_df.copy()
    cgm_df["bolus"] = 0.0
    cgm_df["bolus_standard"] = 0.0
    cgm_df["bolus_extended"] = 0.0
    type_counts: Dict[str, int] = {}

    if cgm_df.empty or bolus_df.empty:
        return cgm_df, {"total": 0}

    for pat_id, cgm_group in tqdm(
        cgm_df.groupby("pat_id", sort=False),
        total=cgm_df["pat_id"].nunique(),
        desc="Attach bolus to CGM",
    ):
        bolus_group = bolus_df[bolus_df["pat_id"] == pat_id].sort_values("date")
        if bolus_group.empty:
            continue
        bolus_times = bolus_group["date"].to_numpy()
        bolus_vals = bolus_group["bolus"].to_numpy()
        bolus_types = bolus_group["bolus_type"].to_numpy()

        def find_bolus(ts: pd.Timestamp):
            idx = bolus_times.searchsorted(ts.to_datetime64())
            candidates = []
            if idx > 0:
                candidates.append(idx - 1)
            if idx < len(bolus_times):
                candidates.append(idx)
            best = None
            best_diff = None
            for i in candidates:
                delta = (bolus_times[i] - ts.to_datetime64()) / pd.Timedelta(seconds=1)
                if -window_before_seconds <= delta <= window_after_seconds:
                    adiff = abs(delta)
                    if best is None or adiff < best_diff or (adiff == best_diff and delta >= 0):
                        best = i
                        best_diff = adiff
            if best is None:
                return 0.0, None
            return float(bolus_vals[best]), bolus_types[best]

        bolus_match = cgm_group["date"].apply(find_bolus)
        # bolus_match is series of tuples
        amounts = []
        bol_std = []
        bol_ext = []
        for amt, btype in bolus_match:
            amounts.append(amt)
            is_extended = isinstance(btype, str) and ("extend" in btype.lower() or "square" in btype.lower() or "dual" in btype.lower())
            if is_extended:
                bol_ext.append(amt)
                bol_std.append(0.0)
            else:
                bol_std.append(amt)
                bol_ext.append(0.0)
            if btype:
                type_counts[btype] = type_counts.get(btype, 0) + (1 if amt != 0 else 0)
        cgm_df.loc[cgm_group.index, "bolus"] = amounts
        cgm_df.loc[cgm_group.index, "bolus_standard"] = bol_std
        cgm_df.loc[cgm_group.index, "bolus_extended"] = bol_ext

    type_counts["total"] = sum(type_counts.values())
    return cgm_df, type_counts


def _filter_long_bolus_gaps(df: pd.DataFrame, max_gap_seconds: int) -> pd.DataFrame:
    """Drop rows where time since last non-zero bolus exceeds max_gap_seconds."""
    if df.empty:
        return df
    kept_parts: List[pd.DataFrame] = []
    threshold = pd.Timedelta(seconds=max_gap_seconds)
    for pat_id, group in df.groupby("pat_id", sort=False):
        group = group.sort_values("date")
        last_bolus_time = None
        keep_mask = []
        for ts, bol_std, bol_ext in zip(group["date"], group.get("bolus_standard", group["bolus"]), group.get("bolus_extended", 0)):
            bol_total = (bol_std or 0) + (bol_ext or 0)
            if bol_total and bol_total != 0:
                last_bolus_time = ts
                keep_mask.append(True)
            else:
                if last_bolus_time is None:
                    keep_mask.append(False)
                else:
                    keep_mask.append((ts - last_bolus_time) <= threshold)
        kept = group.loc[keep_mask]
        if not kept.empty:
            kept_parts.append(kept)
    if not kept_parts:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(kept_parts, ignore_index=True)


def _read_meal(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "RocheMeter.txt"
    if not path.exists():
        return pd.DataFrame(columns=["pat_id", "date", "meal"])
    df = pd.read_csv(path, sep="|", low_memory=False)
    date_col = "DataDtTm_adjusted" if "DataDtTm_adjusted" in df.columns else "DataDtTm"
    df = df.rename(columns={"PtID": "pat_id", date_col: "date", "Carbs": "meal"})
    df["date"] = _parse_datetime(df["date"])
    df["pat_id"] = df["pat_id"].astype(str).str.strip()
    df["meal"] = pd.to_numeric(df["meal"], errors="coerce")
    df = df.dropna(subset=["pat_id", "date", "meal"])
    df = df[df["meal"] > 0]
    return df[["pat_id", "date", "meal"]]


def _attach_meal(df: pd.DataFrame, meal_df: pd.DataFrame, cfg: DCLP5Config) -> pd.DataFrame:
    if df.empty or meal_df.empty:
        df = df.copy()
        df["meal"] = 0.0
        return df
    freq = f"{cfg.resample_freq_minutes}min"
    meal_df = meal_df.copy()
    meal_df["date_round"] = meal_df["date"].dt.round(freq)
    meal_agg = (
        meal_df.groupby(["pat_id", "date_round"], as_index=False)["meal"]
        .sum()
        .rename(columns={"date_round": "date", "meal": "meal_val"})
    )
    out = df.merge(meal_agg, on=["pat_id", "date"], how="left")
    out["meal"] = out["meal_val"].fillna(0.0)
    out = out.drop(columns=["meal_val"])
    return out


def _assign_sequences(
    df: pd.DataFrame, cfg: DCLP5Config
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Split on gaps > sequence_gap_seconds, assign global seq_id, and drop short sequences.
    """
    if df.empty:
        cols = list(df.columns)
        if "seq_id" not in cols:
            cols.append("seq_id")
        return pd.DataFrame(columns=cols), {
            "sequences_total": 0,
            "sequences_kept": 0,
            "sequences_dropped_short": 0,
        }

    max_gap = pd.Timedelta(seconds=cfg.sequence_gap_seconds)
    seq_id = 1
    kept_parts: List[pd.DataFrame] = []
    total = 0
    dropped_short = 0

    for pat_id, group in df.groupby("pat_id", sort=False):
        group = group.sort_values("date")
        gaps = group["date"].diff() > max_gap
        boundaries = np.where(gaps)[0].tolist() + [len(group)]
        start = 0
        for end in boundaries:
            segment = group.iloc[start:end]
            start = end
            if segment.empty:
                continue
            total += 1
            if len(segment) < cfg.min_sequence_steps:
                dropped_short += 1
                continue
            segment = segment.copy()
            segment["seq_id"] = seq_id
            seq_id += 1
            kept_parts.append(segment)

    stats = {
        "sequences_total": total,
        "sequences_kept": len(kept_parts),
        "sequences_dropped_short": dropped_short,
    }
    if not kept_parts:
        return pd.DataFrame(columns=df.columns), stats
    result = pd.concat(kept_parts, ignore_index=True)
    return result, stats


def _write_outputs(df: pd.DataFrame, output_dir: Path, metadata: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # Sort with pat_id priority, then timestamp
    df = df.sort_values(["pat_id", "date"]).reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["basal"] = df.groupby("pat_id")["basal"].ffill().fillna(0.0)
    df["meal"] = df["meal"].fillna(0.0)
    if "weight_kg" not in df.columns:
        df["weight_kg"] = pd.NA
    if "bolus_standard" not in df.columns:
        df["bolus_standard"] = df.get("bolus", 0)
    if "bolus_extended" not in df.columns:
        df["bolus_extended"] = 0.0
    metadata["meals_found"] = int((df["meal"] > 0).sum())
    df = df[["pat_id", "seq_id", "date", "cgm", "basal", "bolus_standard", "bolus_extended", "weight_kg", "meal"]]

    df.to_csv(output_dir / "timeseries.csv", index=False)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def preprocess_dclp5(raw_dir: Path, output_dir: Path) -> None:
    raw_dir, output_dir = ensure_io(raw_dir, output_dir)

    combined, raw_total = _combine_sources(raw_dir)
    valid_count = len(combined)

    cfg = DCLP5Config()
    # Apply deduplication within configured window per patient
    deduped = _dedup_within_window(combined, window_seconds=cfg.dedup_window_seconds)
    resampled, segment_count = _resample_and_interpolate_cgm(deduped, cfg)
    basal_df = _read_basal(raw_dir)
    with_basal, basal_matched = _attach_basal(
        resampled,
        basal_df,
        window_before_seconds=cfg.basal_window_before_seconds,
        window_after_seconds=cfg.basal_window_after_seconds,
    )
    bolus_df = _read_bolus(raw_dir)
    with_bolus, bolus_counts = _attach_bolus(
        with_basal,
        bolus_df,
        window_before_seconds=cfg.bolus_window_before_seconds,
        window_after_seconds=cfg.bolus_window_after_seconds,
    )
    bolus_filtered = _filter_long_bolus_gaps(with_bolus, cfg.max_bolus_gap_seconds)
    # Collect weight data anchored to earliest CGM per patient (for screening weights)
    cgm_min_dates = (
        resampled.groupby("pat_id")["date"].min().to_dict()
        if not resampled.empty
        else {}
    )
    weight_df = _read_weight_sources(raw_dir, cgm_min_dates)
    with_weight, weight_stats = _attach_weight(bolus_filtered, weight_df, interpolate=True)

    meal_df = _read_meal(raw_dir)
    with_meal = _attach_meal(with_weight, meal_df, cfg)

    final_df, seq_stats = _assign_sequences(with_meal, cfg)

    metadata = {
        "dataset": "DCLP5",
        "source_files": [
            "TandemCGMDATAGXB.txt",
            "DCLP5TandemCGMDATAGXB_b.txt",
            "DexcomClarityCGM.txt",
            "DCLP5TandemBASALRATECHG_b.txt",
            "DCLP5TandemBolus_Completed_Combined_b.txt",
        ],
        "raw_rows": int(raw_total),
        "rows_after_validation": int(valid_count),
        "rows_after_dedup": int(len(deduped)),
        "rows_after_resample": int(len(resampled)),
        "rows_after_bolus_gap_filter": int(len(bolus_filtered)),
        "rows_final": int(len(final_df)),
        "rows_with_basal": int(basal_matched),
        "rows_with_bolus": int((final_df.get("bolus_standard", 0) != 0).sum() + (final_df.get("bolus_extended", 0) != 0).sum()),
        "patients_final": int(final_df["pat_id"].nunique()) if not final_df.empty else 0,
        "patients_with_weight": int(final_df.dropna(subset=["weight_kg"])["pat_id"].nunique()) if not final_df.empty else 0,
        "bolus_counts_by_type": bolus_counts,
        "weight_records": int(len(weight_df)),
        "rows_with_weight": int(weight_stats["rows_with_weight"]),
        "rows_weight_interpolated": int(weight_stats["rows_weight_interpolated"]),
        "segment_count_before_resample": int(segment_count),
        "sequence_stats": seq_stats,
        "channels": ["cgm", "basal", "bolus_standard", "bolus_extended", "weight_kg"],
        "basal_window_seconds_before": cfg.basal_window_before_seconds,
        "basal_window_seconds_after": cfg.basal_window_after_seconds,
        "bolus_window_seconds_before": cfg.bolus_window_before_seconds,
        "bolus_window_seconds_after": cfg.bolus_window_after_seconds,
        "dedup_window_seconds": cfg.dedup_window_seconds,
        "sequence_gap_seconds": cfg.sequence_gap_seconds,
        "resample_freq_minutes": cfg.resample_freq_minutes,
        "max_bolus_gap_seconds": cfg.max_bolus_gap_seconds,
        "min_sequence_steps": cfg.min_sequence_steps,
        "notes": (
            "Deduplicated per patient within configured window; kept latest record. "
            "Resampled CGM to fixed 5-minute grid with interpolation inside <=30-minute gaps. "
            "Attached basal as the latest rate within configured window of CGM timestamp. "
            "Attached bolus (default 0) if within configured window of CGM timestamp. "
            "Dropped spans with no bolus beyond configured gap and removed short sequences."
        ),
    }

    _write_outputs(final_df, output_dir, metadata)
