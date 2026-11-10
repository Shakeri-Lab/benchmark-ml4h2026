from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from pandas._libs.tslibs import parsing
from tqdm.auto import tqdm

from preprocessing.common import ensure_io


@dataclass(frozen=True)
class DCLP3Config:
    dedup_window_seconds: int = 15
    sequence_gap_seconds: int = 1800  # 30 minutes
    resample_freq_minutes: int = 5
    interp_min_gap_seconds: int = 420  # informational
    basal_window_before_seconds: int = 10800  # 3 hours
    basal_window_after_seconds: int = 15
    bolus_window_before_seconds: int = 285
    bolus_window_after_seconds: int = 15
    max_bolus_gap_seconds: int = 43200  # 12 hours
    min_sequence_steps: int = 312


def _parse_datetime(series: pd.Series) -> pd.Series:
    sample = next((x for x in series.dropna().astype(str) if x.strip()), None)
    if sample:
        fmt = parsing.guess_datetime_format(sample)
        if fmt:
            return pd.to_datetime(series, format=fmt, errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def _read_csv_fallback(path: Path, sep: str = "|") -> pd.DataFrame:
    """Read CSV trying UTF-8, then UTF-16 if needed."""
    try:
        return pd.read_csv(path, sep=sep, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=sep, low_memory=False, encoding="utf-16")


def _read_cgm_sources(raw_dir: Path) -> Tuple[pd.DataFrame, int]:
    base = raw_dir / "Data Files"
    frames: List[pd.DataFrame] = []

    # DexcomClarityCGM_a (UTF-16LE with BOM)
    p = base / "DexcomClarityCGM_a.txt"
    if p.exists():
        df = pd.read_csv(p, sep="|", low_memory=False, encoding="utf-16")
        df = df.rename(columns={"PtID": "pat_id", "DataDtTm": "date", "CGM": "cgm"})
        df = df[["pat_id", "date", "cgm"]]
        df["date"] = _parse_datetime(df["date"])
        df["pat_id"] = df["pat_id"].astype(str).str.strip()
        df["cgm"] = pd.to_numeric(df["cgm"], errors="coerce")
        frames.append(df)
    # cgm.txt (UTF-8)
    p = base / "cgm.txt"
    if p.exists():
        df = pd.read_csv(p, sep="|", low_memory=False)
        df = df.rename(columns={"PtID": "pat_id", "DataDtTm": "date", "CGM": "cgm"})
        df["date"] = _parse_datetime(df["date"])
        df["pat_id"] = df["pat_id"].astype(str).str.strip()
        df["cgm"] = pd.to_numeric(df["cgm"], errors="coerce")
        frames.append(df[["pat_id", "date", "cgm"]])
    # Pump CGM (UTF-8)
    p = base / "Pump_CGMGlucoseValue.txt"
    if p.exists():
        df = pd.read_csv(p, sep="|", low_memory=False)
        date_col = "DataDtTm_adjusted" if "DataDtTm_adjusted" in df.columns else "DataDtTm"
        df = df.rename(columns={"PtID": "pat_id", date_col: "date", "CGMValue": "cgm"})
        df["date"] = _parse_datetime(df["date"])
        df["pat_id"] = df["pat_id"].astype(str).str.strip()
        df["cgm"] = pd.to_numeric(df["cgm"], errors="coerce")
        frames.append(df[["pat_id", "date", "cgm"]])

    raw_total = sum(len(f) for f in frames)
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(columns=["pat_id", "date", "cgm"])
    combined = combined.dropna(subset=["pat_id", "date", "cgm"])
    return combined, raw_total


def _dedup_within_window(df: pd.DataFrame, window_seconds: int) -> pd.DataFrame:
    if df.empty:
        return df
    df_sorted = df.sort_values(["pat_id", "date"])
    grouped = df_sorted.groupby("pat_id", sort=False)
    parts: List[pd.DataFrame] = []
    for _, g in tqdm(grouped, total=len(grouped), desc="Dedup DCLP3 patients"):
        deltas = g["date"].diff().dt.total_seconds()
        new_group = (deltas.isna()) | (deltas > window_seconds)
        group_ids = new_group.cumsum()
        parts.append(g.groupby(group_ids).tail(1))
    return pd.concat(parts, ignore_index=True)


def _resample_and_interpolate_cgm(df: pd.DataFrame, cfg: DCLP3Config) -> Tuple[pd.DataFrame, int]:
    resampled_parts: List[pd.DataFrame] = []
    max_gap = pd.Timedelta(seconds=cfg.sequence_gap_seconds)
    freq = f"{cfg.resample_freq_minutes}min"
    segment_count = 0
    for pat_id, group in df.groupby("pat_id", sort=False):
        g = group.sort_values("date")
        gaps = g["date"].diff() > max_gap
        boundaries = np.where(gaps)[0].tolist() + [len(g)]
        start = 0
        for end in boundaries:
            seg = g.iloc[start:end]
            start = end
            if seg.empty:
                continue
            seg_indexed = seg.set_index("date").sort_index()
            res = seg_indexed[["cgm"]].resample(freq).mean()
            res["cgm"] = res["cgm"].interpolate(method="time")
            res["pat_id"] = pat_id
            res["date"] = res.index
            res = res.dropna(subset=["cgm"])
            if not res.empty:
                resampled_parts.append(res.reset_index(drop=True))
                segment_count += 1
    if not resampled_parts:
        return pd.DataFrame(columns=["pat_id", "date", "cgm"]), 0
    resampled = pd.concat(resampled_parts, ignore_index=True)
    return resampled[["pat_id", "date", "cgm"]], segment_count


def _read_basal(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "Data Files" / "Pump_BasalRateChange.txt"
    if not path.exists():
        return pd.DataFrame(columns=["pat_id", "date", "basal"])
    df = pd.read_csv(path, sep="|", low_memory=False)
    df = df.rename(
        columns={
            "PtID": "pat_id",
            "CommandedBasalRate": "basal",
            "DataDtTm": "date_raw",
        }
    )
    df["date"] = pd.NaT
    if "DataDtTm_adjusted" in df.columns:
        df["date"] = _parse_datetime(df["DataDtTm_adjusted"])
        fallback = _parse_datetime(df["date_raw"])
        df["date"] = df["date"].fillna(fallback)
    else:
        df["date"] = _parse_datetime(df["date_raw"])
    df["pat_id"] = df["pat_id"].astype(str).str.strip()
    df["basal"] = pd.to_numeric(df["basal"], errors="coerce")
    return df.dropna(subset=["pat_id", "date", "basal"])


def _attach_basal(cgm_df: pd.DataFrame, basal_df: pd.DataFrame, cfg: DCLP3Config) -> Tuple[pd.DataFrame, int]:
    if cgm_df.empty or basal_df.empty:
        cgm_df = cgm_df.copy()
        cgm_df["basal"] = pd.NA
        return cgm_df, 0
    result_parts: List[pd.DataFrame] = []
    matched = 0
    for pat_id, group in cgm_df.groupby("pat_id", sort=False):
        g = group.dropna(subset=["date"]).sort_values("date")
        bg = basal_df[basal_df["pat_id"] == pat_id].dropna(subset=["date"]).sort_values("date")
        if g.empty:
            continue
        if bg.empty:
            g = g.copy()
            g["basal"] = pd.NA
            result_parts.append(g)
            continue
        bt = bg["date"].values.astype("datetime64[ns]").view("int64")
        bv = bg["basal"].to_numpy()
        ct = g["date"].values.astype("datetime64[ns]").view("int64")
        lower = ct - cfg.basal_window_before_seconds * 1_000_000_000
        upper = ct + cfg.basal_window_after_seconds * 1_000_000_000
        start = np.searchsorted(bt, lower, side="left")
        end = np.searchsorted(bt, upper, side="right")
        chosen = np.where(start < end, end - 1, -1)
        basal_vals = np.where(chosen >= 0, bv[chosen], np.nan)
        matched += np.isfinite(basal_vals).sum()
        g = g.copy()
        g["basal"] = basal_vals
        result_parts.append(g)
    if not result_parts:
        cgm_df = cgm_df.copy()
        cgm_df["basal"] = pd.NA
        return cgm_df, 0
    out = pd.concat(result_parts, ignore_index=True)
    return out, int(matched)


def _read_meal(raw_dir: Path) -> pd.DataFrame:
    """Read meal/carbohydrate announcements from Roche meter logs."""
    path = raw_dir / "Data Files" / "RocheMeter_a.txt"
    if not path.exists():
        return pd.DataFrame(columns=["pat_id", "date", "meal"])
    df = _read_csv_fallback(path, sep="|")
    date_col = "DataDtTm_adjusted" if "DataDtTm_adjusted" in df.columns else "DataDtTm"
    df = df.rename(columns={"PtID": "pat_id", date_col: "date", "Carbs": "meal"})
    df["date"] = _parse_datetime(df["date"])
    df["pat_id"] = df["pat_id"].astype(str).str.strip()
    df["meal"] = pd.to_numeric(df["meal"], errors="coerce")
    df = df.dropna(subset=["pat_id", "date", "meal"])
    df = df[df["meal"] > 0]
    return df[["pat_id", "date", "meal"]]


def _attach_meal(df: pd.DataFrame, meal_df: pd.DataFrame, cfg: DCLP3Config) -> pd.DataFrame:
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


def _read_bolus(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "Data Files" / "Pump_BolusDelivered.txt"
    if not path.exists():
        return pd.DataFrame(columns=["pat_id", "date", "bolus", "bolus_type"])
    df = _read_csv_fallback(path, sep="|")
    df = df.rename(
        columns={
            "PtID": "pat_id",
            "BolusAmount": "bolus",
            "BolusType": "bolus_type",
            "DataDtTm": "date_raw",
        }
    )
    df["date"] = pd.NaT
    if "DataDtTm_adjusted" in df.columns:
        df["date"] = _parse_datetime(df["DataDtTm_adjusted"])
        fallback = _parse_datetime(df["date_raw"])
        df["date"] = df["date"].fillna(fallback)
    else:
        df["date"] = _parse_datetime(df["date_raw"])
    df["pat_id"] = df["pat_id"].astype(str).str.strip()
    df["bolus"] = pd.to_numeric(df["bolus"], errors="coerce")
    df["bolus_type"] = df["bolus_type"].fillna("Unknown")
    return df.dropna(subset=["pat_id", "date", "bolus"])


def _attach_bolus(cgm_df: pd.DataFrame, bolus_df: pd.DataFrame, cfg: DCLP3Config) -> Tuple[pd.DataFrame, Dict[str, int]]:
    cgm_df = cgm_df.copy()
    cgm_df["bolus"] = 0.0
    cgm_df["bolus_standard"] = 0.0
    cgm_df["bolus_extended"] = 0.0
    counts: Dict[str, int] = {}
    if cgm_df.empty or bolus_df.empty:
        return cgm_df, {"total": 0}
    for pat_id, group in tqdm(cgm_df.groupby("pat_id", sort=False), total=cgm_df["pat_id"].nunique(), desc="Attach bolus to CGM"):
        bg = bolus_df[bolus_df["pat_id"] == pat_id].sort_values("date")
        if bg.empty:
            continue
        bt = bg["date"].values.astype("datetime64[ns]").view("int64")
        bv = bg["bolus"].to_numpy()
        btype = bg["bolus_type"].to_numpy()
        ct = group["date"].values.astype("datetime64[ns]").view("int64")
        idx_right = np.searchsorted(bt, ct, side="left")
        prev_idx = idx_right - 1
        next_idx = idx_right
        valid_prev = (prev_idx >= 0)
        valid_next = (next_idx < len(bt))
        prev_dist = np.where(valid_prev, ct - bt[np.clip(prev_idx, 0, len(bt)-1)], np.inf)
        next_dist = np.where(valid_next, bt[np.clip(next_idx, 0, len(bt)-1)] - ct, np.inf)
        prev_dist = np.where(prev_dist <= cfg.bolus_window_before_seconds * 1_000_000_000, prev_dist, np.inf)
        next_dist = np.where(next_dist <= cfg.bolus_window_after_seconds * 1_000_000_000, next_dist, np.inf)
        choose_prev = prev_dist < next_dist
        choose_next = ~choose_prev
        best_idx = np.where(np.isfinite(prev_dist) | np.isfinite(next_dist),
                            np.where(choose_prev, prev_idx, next_idx),
                            -1)
        amounts = []
        bol_std = []
        bol_ext = []
        for bi in best_idx:
            if bi == -1:
                amounts.append(0.0)
                bol_std.append(0.0)
                bol_ext.append(0.0)
            else:
                amt = float(bv[bi])
                amounts.append(amt)
                t = btype[bi]
                is_extended = isinstance(t, str) and ("extend" in t.lower() or "square" in t.lower() or "dual" in t.lower())
                if is_extended:
                    bol_ext.append(amt)
                    bol_std.append(0.0)
                else:
                    bol_std.append(amt)
                    bol_ext.append(0.0)
                if t:
                    counts[t] = counts.get(t, 0) + (1 if amt != 0 else 0)
        cgm_df.loc[group.index, "bolus"] = amounts
        cgm_df.loc[group.index, "bolus_standard"] = bol_std
        cgm_df.loc[group.index, "bolus_extended"] = bol_ext
    counts["total"] = sum(counts.values())
    return cgm_df, counts


def _filter_long_bolus_gaps(df: pd.DataFrame, cfg: DCLP3Config) -> pd.DataFrame:
    if df.empty:
        return df
    out = []
    th = pd.Timedelta(seconds=cfg.max_bolus_gap_seconds)
    for pat_id, g in df.groupby("pat_id", sort=False):
        g = g.sort_values("date")
        last = None
        keep = []
        for ts, bol, bol_ext, bol_std in zip(g["date"], g["bolus"], g["bolus_extended"], g["bolus_standard"]):
            bol_total = (bol_ext or 0) + (bol_std or 0)
            if bol_total and bol_total != 0:
                last = ts
                keep.append(True)
            else:
                if last is None:
                    keep.append(False)
                else:
                    keep.append((ts - last) <= th)
        kept = g.loc[keep]
        if not kept.empty:
            out.append(kept)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=df.columns)


def _assign_sequences(df: pd.DataFrame, cfg: DCLP3Config) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if df.empty:
        cols = list(df.columns)
        if "seq_id" not in cols:
            cols.append("seq_id")
        return pd.DataFrame(columns=cols), {"sequences_total": 0, "sequences_kept": 0, "sequences_dropped_short": 0}
    max_gap = pd.Timedelta(seconds=cfg.sequence_gap_seconds)
    seq_id = 1
    kept: List[pd.DataFrame] = []
    total = dropped = 0
    for pat_id, g in df.groupby("pat_id", sort=False):
        g = g.sort_values("date")
        gaps = g["date"].diff() > max_gap
        bounds = np.where(gaps)[0].tolist() + [len(g)]
        s = 0
        for e in bounds:
            seg = g.iloc[s:e]
            s = e
            if seg.empty:
                continue
            total += 1
            if len(seg) < cfg.min_sequence_steps:
                dropped += 1
                continue
            seg = seg.copy()
            seg["seq_id"] = seq_id
            seq_id += 1
            kept.append(seg)
    stats = {"sequences_total": total, "sequences_kept": len(kept), "sequences_dropped_short": dropped}
    return (pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=df.columns)), stats


def _read_weight_sources(raw_dir: Path, cgm_min_dates: Dict[str, pd.Timestamp]) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    base = raw_dir / "Data Files"

    def add_records(path: Path, date_col: str | None, weight_col: str, unit_col: str):
        if not path.exists():
            return
        df = _read_csv_fallback(path, sep="|")
        if "PtID" not in df.columns or weight_col not in df.columns:
            return
        df = df.rename(columns={"PtID": "pat_id", weight_col: "weight", unit_col: "units"})
        df["pat_id"] = df["pat_id"].astype(str).str.strip()
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
        df["units"] = df["units"].astype(str).str.lower()
        if date_col and date_col in df.columns:
            df["date"] = _parse_datetime(df[date_col])
        else:
            df["date"] = pd.NaT
        for _, row in df.dropna(subset=["weight"]).iterrows():
            records.append(row.to_dict())

    add_records(base / "FollowUpCTV_a.txt", "CGMUploadedDt", "Weight", "WeightUnits")
    if (base / "DiabScreening_a.txt").exists():
        scr = _read_csv_fallback(base / "DiabScreening_a.txt", sep="|")
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

    phys_path = base / "DiabPhysExam_a.txt"
    if phys_path.exists():
        phys = _read_csv_fallback(phys_path, sep="|")
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

    adv_path = base / "AdvEvent_a.txt"
    if adv_path.exists():
        adv = _read_csv_fallback(adv_path, sep="|")
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
    return weight_df[["pat_id", "date", "weight_kg"]]


def _attach_weight(cgm_df: pd.DataFrame, weight_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    cgm_df = cgm_df.copy()
    cgm_df["weight_kg"] = pd.NA
    stats = {"rows_with_weight": 0, "rows_weight_interpolated": 0}
    if cgm_df.empty or weight_df.empty:
        return cgm_df, stats
    for pat_id, g in cgm_df.groupby("pat_id", sort=False):
        w = weight_df[weight_df["pat_id"] == pat_id].sort_values("date")
        if w.empty:
            continue
        times = w["date"].values.astype("datetime64[ns]").view("int64")
        weights = w["weight_kg"].to_numpy(dtype=float)
        def find(ts: pd.Timestamp):
            ts_ns = ts.value
            idx = times.searchsorted(ts_ns)
            if idx == 0:
                return weights[0], False
            if idx == len(times):
                return weights[-1], False
            t0, t1 = times[idx - 1], times[idx]
            w0, w1 = weights[idx - 1], weights[idx]
            if t1 == t0:
                return w1, False
            frac = (ts_ns - t0) / (t1 - t0)
            return w0 + frac * (w1 - w0), True
        vals = []
        interp_flags = []
        for ts in g["date"]:
            wv, inter = find(ts)
            vals.append(wv)
            interp_flags.append(inter)
        stats["rows_with_weight"] += len(g)
        stats["rows_weight_interpolated"] += sum(interp_flags)
        cgm_df.loc[g.index, "weight_kg"] = vals
    return cgm_df, stats


def preprocess_dclp3(raw_dir: Path, output_dir: Path) -> None:
    raw_dir, output_dir = ensure_io(raw_dir, output_dir)
    cfg = DCLP3Config()

    combined, raw_total = _read_cgm_sources(raw_dir)
    valid_count = len(combined)
    deduped = _dedup_within_window(combined, cfg.dedup_window_seconds)
    resampled, segment_count = _resample_and_interpolate_cgm(deduped, cfg)

    basal_df = _read_basal(raw_dir)
    with_basal, basal_matched = _attach_basal(resampled, basal_df, cfg)

    bolus_df = _read_bolus(raw_dir)
    with_bolus, bolus_counts = _attach_bolus(with_basal, bolus_df, cfg)

    bolus_filtered = _filter_long_bolus_gaps(with_bolus, cfg)
    cgm_min_dates = resampled.groupby("pat_id")["date"].min().to_dict() if not resampled.empty else {}
    weight_df = _read_weight_sources(raw_dir, cgm_min_dates)
    with_weight, weight_stats = _attach_weight(bolus_filtered, weight_df)

    meal_df = _read_meal(raw_dir)
    with_meal = _attach_meal(with_weight, meal_df, cfg)

    final_df, seq_stats = _assign_sequences(with_meal, cfg)

    metadata = {
        "dataset": "DCLP3",
        "source_files": [
            "Data Files/DexcomClarityCGM_a.txt",
            "Data Files/cgm.txt",
            "Data Files/Pump_CGMGlucoseValue.txt",
            "Data Files/Pump_BasalRateChange.txt",
            "Data Files/Pump_BolusDelivered.txt",
            "Data Files/DiabPhysExam_a.txt",
            "Data Files/DiabScreening_a.txt",
            "Data Files/FollowUpCTV_a.txt",
            "Data Files/AdvEvent_a.txt",
        ],
        "raw_rows": int(raw_total),
        "rows_after_validation": int(valid_count),
        "rows_after_dedup": int(len(deduped)),
        "rows_after_resample": int(len(resampled)),
        "rows_after_bolus_gap_filter": int(len(bolus_filtered)),
        "rows_final": int(len(final_df)),
        "rows_with_basal": int(basal_matched),
        "rows_with_bolus": int((final_df["bolus_standard"] != 0).sum() + (final_df["bolus_extended"] != 0).sum()),
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
            "Combined CGM sources, deduped within window, resampled/interpolated, "
            "attached basal/bolus, filtered long bolus gaps, attached weight, and filtered short sequences."
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    final_df = final_df.sort_values(["pat_id", "date"]).reset_index(drop=True)
    final_df["date"] = final_df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    # Basal: carry forward last known rate per patient; fallback to 0.0 if missing at start
    final_df["basal"] = (
        final_df.groupby("pat_id")["basal"].ffill().fillna(0.0)
    )
    final_df["meal"] = final_df["meal"].fillna(0.0)
    final_df["bolus_total"] = final_df.get("bolus_standard", 0) + final_df.get("bolus_extended", 0)
    metadata["meals_found"] = int((final_df["meal"] > 0).sum())
    final_df = final_df[
        ["pat_id", "seq_id", "date", "cgm", "basal", "bolus_standard", "bolus_extended", "weight_kg", "meal"]
    ]
    final_df.to_csv(output_dir / "timeseries.csv", index=False)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
