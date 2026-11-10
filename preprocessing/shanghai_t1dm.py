from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from pandas._libs.tslibs import parsing
from tqdm.auto import tqdm

from preprocessing.common import ensure_io


@dataclass(frozen=True)
class ShanghaiConfig:
    min_sequence_steps: int = 1  # keep all sequences by default; adjust if needed
    summary_weight_file: str = "Shanghai_T1DM_Summary.xlsx"
    resample_freq_minutes: int = 5
    insulin_index_file: str = "insulin_index_shanghai.json"


def _load_insulin_index() -> Dict[str, str]:
    """Load the curated insulin name -> class mapping."""
    path = Path(__file__).with_name("insulin_index_shanghai.json")
    mapping: Dict[str, str] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
            for name in data.get("short", []):
                mapping[name.strip().lower()] = "short"
            for name in data.get("long", []):
                mapping[name.strip().lower()] = "long"
        except Exception:
            pass
    return mapping


def _normalize_insulin_label(text: str) -> str:
    text = (text or "").replace("\xa0", " ").strip().lower()
    if not text:
        return ""
    base = re.split(r"[;,]", text, maxsplit=1)[0]
    base = re.sub(r"\s+", " ", base).strip()
    return base


def _classify_insulin_label(label: str, mapping: Dict[str, str]) -> str:
    if not label:
        return ""
    if label in mapping:
        return mapping[label]
    long_keywords = [
        "glargine",
        "detemir",
        "degludec",
        "nph",
        "ultralente",
        "lantus",
        "basaglar",
        "toujeo",
        "tresiba",
        "levemir",
        "protaphane",
    ]
    if any(k in label for k in long_keywords):
        return "long"
    short_keywords = ["rapid", "regular", "lispro", "aspart", "glulisine", "novolin", "novor", "humalog"]
    if any(k in label for k in short_keywords):
        return "short"
    return ""


def _parse_datetime_series(series: pd.Series) -> pd.Series:
    sample = next((x for x in series.dropna().astype(str) if x.strip()), None)
    if sample:
        fmt = parsing.guess_datetime_format(sample)
        if fmt:
            return pd.to_datetime(series, format=fmt, errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def _find_time_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if df[col].dtype.kind in {"M"}:
            return col
    for col in df.columns:
        lc = str(col).lower()
        if "time" in lc or "date" in lc:
            parsed = _parse_datetime_series(df[col])
            if parsed.notna().sum() > 0:
                df[col] = parsed
                return col
    col = df.columns[0]
    df[col] = _parse_datetime_series(df[col])
    return col


def _parse_sc_insulin_column(series: pd.Series, mapping: Dict[str, str], unknown_labels: Set[str]) -> Tuple[pd.Series, pd.Series]:
    short_vals: List[float] = []
    long_vals: List[float] = []
    for raw in series.fillna(""):
        val_str = str(raw).strip()
        if not val_str:
            short_vals.append(0.0)
            long_vals.append(0.0)
            continue
        label = _normalize_insulin_label(val_str)
        dose_match = re.findall(r"[-+]?[0-9]*\.?[0-9]+", val_str)
        dose = float(dose_match[0]) if dose_match else 0.0

        cls = _classify_insulin_label(label, mapping)
        if not cls:
            unknown_labels.add(label)
            cls = "short"  # fall back to short-acting to avoid dropping insulin
        if cls == "long":
            long_vals.append(dose)
            short_vals.append(0.0)
        else:
            short_vals.append(dose)
            long_vals.append(0.0)
    return pd.Series(short_vals, index=series.index, dtype=float), pd.Series(long_vals, index=series.index, dtype=float)


def _parse_iv_column(series: pd.Series, mapping: Dict[str, str], unknown_labels: Set[str]) -> Tuple[pd.Series, pd.Series]:
    """Parse IV insulin entries; merged cells appear as value followed by blank rows. Distribute dose evenly over the block."""
    short_vals = np.zeros(len(series), dtype=float)
    long_vals = np.zeros(len(series), dtype=float)
    raw = series.fillna("").astype(str)
    i = 0
    n = len(raw)
    while i < n:
        val = raw.iloc[i].strip()
        if not val:
            i += 1
            continue
        # find block length (value + subsequent blanks until next non-empty or end)
        j = i + 1
        while j < n and not raw.iloc[j].strip():
            j += 1
        block_len = max(1, j - i)

        label = _normalize_insulin_label(val)
        dose_match = re.findall(r"[-+]?[0-9]*\.?[0-9]+", val)
        dose = float(dose_match[0]) if dose_match else 0.0
        cls = _classify_insulin_label(label, mapping)
        if not cls:
            unknown_labels.add(label)
            cls = "short"
        per_row = dose / block_len if block_len > 0 else dose
        if cls == "long":
            long_vals[i:j] = per_row
        else:
            short_vals[i:j] = per_row
        i = j
    return pd.Series(short_vals, index=series.index, dtype=float), pd.Series(long_vals, index=series.index, dtype=float)


def _extract_glucose_grams(text: str) -> float:
    """Estimate glucose grams from IV description (e.g., '500 ml 5% glucose')."""
    lower = text.lower()
    vol_match = re.search(r"(\d+(?:\.\d+)?)\s*ml", lower)
    # Prefer a % that is tied to 'glucose'; otherwise take the first %
    perc_match = re.search(r"(\d+(?:\.\d+)?)\s*%[^,]*glucose", lower) or re.search(r"(\d+(?:\.\d+)?)\s*%", lower)
    if not vol_match or not perc_match:
        return 0.0
    vol_ml = float(vol_match.group(1))
    perc = float(perc_match.group(1))
    return vol_ml * (perc / 100.0)


def _parse_iv_glucose(series: pd.Series) -> pd.Series:
    """Parse IV glucose grams; distribute evenly across merged-cell blocks."""
    vals = np.zeros(len(series), dtype=float)
    raw = series.fillna("").astype(str)
    i = 0
    n = len(raw)
    while i < n:
        val = raw.iloc[i].strip()
        if not val:
            i += 1
            continue
        j = i + 1
        while j < n and not raw.iloc[j].strip():
            j += 1
        block_len = max(1, j - i)
        grams = _extract_glucose_grams(val)
        per_row = grams / block_len if block_len > 0 else grams
        vals[i:j] = per_row
        i = j
    return pd.Series(vals, index=series.index, dtype=float)


def _read_sequence_file(path: Path, insulin_mapping: Dict[str, str], unknown_labels: Set[str]) -> Tuple[int, int, pd.DataFrame]:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Unexpected filename format: {path.name}")
    pat_orig = int(parts[0])
    seq_local = int(parts[1])

    df = pd.read_excel(path)
    time_col = _find_time_column(df)
    df["date"] = _parse_datetime_series(df[time_col])

    cols_lower = {c.lower(): c for c in df.columns}

    def find_col(substr: str) -> str | None:
        for key, original in cols_lower.items():
            if substr in key:
                return original
        return None

    cgm_col = find_col("cgm")
    bolus_col = find_col("bolus insulin")
    basal_col = find_col("basal insulin")
    meal_col = find_col("dietary")
    sc_col = find_col("insulin dose - s.c.") or find_col("insulin dose")  # s.c. column
    iv_col = find_col("insulin dose - i.v.") or find_col("i.v.")

    rename_map = {
        cgm_col: "cgm",
        bolus_col: "bolus",
        basal_col: "basal",
        meal_col: "meal_raw",
    }
    if sc_col:
        rename_map[sc_col] = "bolus_sc_raw"
    if iv_col:
        rename_map[iv_col] = "bolus_iv_raw"
    df = df.rename(columns=rename_map)
    df["cgm"] = pd.to_numeric(df.get("cgm"), errors="coerce")
    df["bolus"] = pd.to_numeric(df.get("bolus"), errors="coerce").fillna(0.0)

    basal_raw = df.get("basal")
    basal_numeric = pd.to_numeric(basal_raw, errors="coerce")
    if basal_raw is not None:
        suspend_mask = basal_raw.astype(str).str.contains("suspend", case=False, na=False)
        basal_numeric = basal_numeric.mask(suspend_mask, 0.0)
    df["basal"] = basal_numeric.ffill().fillna(0.0)
    bolus_short = df.get("bolus", pd.Series(0.0, index=df.index)).fillna(0.0).copy()
    bolus_long = pd.Series(0.0, index=df.index)
    bolus_iv_short = pd.Series(0.0, index=df.index, dtype=float)
    bolus_iv_long = pd.Series(0.0, index=df.index, dtype=float)
    iv_glucose = pd.Series(0.0, index=df.index, dtype=float)
    if "bolus_sc_raw" in df.columns:
        sc_short, sc_long = _parse_sc_insulin_column(df["bolus_sc_raw"], insulin_mapping, unknown_labels)
        bolus_short = bolus_short.fillna(0.0) + sc_short
        bolus_long = bolus_long + sc_long
    if "bolus_iv_raw" in df.columns:
        iv_short, iv_long = _parse_iv_column(df["bolus_iv_raw"], insulin_mapping, unknown_labels)
        bolus_iv_short = bolus_iv_short + iv_short
        bolus_iv_long = bolus_iv_long + iv_long
        iv_glucose = iv_glucose + _parse_iv_glucose(df["bolus_iv_raw"])
    meal_raw = df.get("meal_raw")
    if meal_raw is not None:
        df["meal"] = meal_raw.fillna("").astype(str).str.strip().apply(lambda x: 1 if x else 0)
    else:
        df["meal"] = 0

    df["bolus_short"] = bolus_short.fillna(0.0)
    df["bolus_long"] = bolus_long.fillna(0.0)
    df["bolus_iv_short"] = bolus_iv_short.fillna(0.0)
    df["bolus_iv_long"] = bolus_iv_long.fillna(0.0)
    df["iv_glucose_g"] = iv_glucose.fillna(0.0)

    df = df[
        ["date", "cgm", "basal", "bolus_short", "bolus_long", "bolus_iv_short", "bolus_iv_long", "iv_glucose_g", "meal"]
    ]
    df = df.dropna(subset=["date", "cgm"])
    df["pat_orig"] = pat_orig
    df["seq_local"] = seq_local
    return pat_orig, seq_local, df


def preprocess_shanghai_t1dm(raw_dir: Path, output_dir: Path) -> None:
    raw_dir, output_dir = ensure_io(raw_dir, output_dir)
    cfg = ShanghaiConfig()
    insulin_mapping = _load_insulin_index()
    unknown_labels: Set[str] = set()

    files = sorted(
        [
            p
            for p in raw_dir.glob("*.xls*")
            if not p.name.startswith("~")
            and p.name != cfg.summary_weight_file
            and p.stem.split("_")[0].isdigit()
        ]
    )
    if not files:
        raise FileNotFoundError(f"No .xlsx files found in {raw_dir}")

    sequences: List[pd.DataFrame] = []
    seq_info: List[Tuple[int, int, pd.Timestamp, str]] = []
    for p in tqdm(files, desc="Read Shanghai sequences"):
        pat_orig, seq_local, df = _read_sequence_file(p, insulin_mapping, unknown_labels)
        if df.empty:
            continue
        first_ts = df["date"].min()
        seq_info.append((pat_orig, seq_local, first_ts, p.stem))
        sequences.append(df)

    if not sequences:
        raise ValueError("No usable sequences found in Shanghai dataset")

    pat_ids = sorted({info[0] for info in seq_info})
    pat_map = {pat: idx for idx, pat in enumerate(pat_ids)}

    seq_order = sorted(zip(sequences, seq_info), key=lambda x: (pat_map[x[1][0]], x[1][1], x[1][2]))
    final_parts: List[pd.DataFrame] = []
    seq_id_counter = 1
    for df, info in seq_order:
        pat_orig = info[0]
        stem = info[3]
        df = df.copy()
        df["pat_id"] = pat_map[pat_orig]
        df["seq_id"] = seq_id_counter
        df["seq_stem"] = stem
        seq_id_counter += 1
        final_parts.append(
            df[
                [
                    "pat_id",
                    "seq_id",
                    "seq_stem",
                    "date",
                    "cgm",
                    "basal",
                    "bolus_short",
                    "bolus_long",
                    "bolus_iv_short",
                    "bolus_iv_long",
                    "iv_glucose_g",
                    "meal",
                ]
            ]
        )

    # Resample each sequence to fixed 5-minute intervals
    resampled_parts: List[pd.DataFrame] = []
    freq = f"{cfg.resample_freq_minutes}min"
    for df in final_parts:
        df = df.sort_values("date")
        idx = pd.date_range(df["date"].min(), df["date"].max(), freq=freq)
        res = df.set_index("date").reindex(idx)
        res["cgm"] = res["cgm"].interpolate(method="time")
        res["basal"] = res["basal"].ffill().bfill().fillna(0.0)
        res["bolus_short"] = res["bolus_short"].fillna(0.0)
        res["bolus_long"] = res["bolus_long"].fillna(0.0)
        res["bolus_iv_short"] = res["bolus_iv_short"].ffill().fillna(0.0)
        res["bolus_iv_long"] = res["bolus_iv_long"].ffill().fillna(0.0)
        res["iv_glucose_g"] = res["iv_glucose_g"].ffill().fillna(0.0)
        res["meal"] = res["meal"].fillna(0).astype(int)
        res["pat_id"] = res["pat_id"].ffill().bfill()
        res["seq_id"] = res["seq_id"].ffill().bfill()
        res["seq_stem"] = res["seq_stem"].ffill().bfill()
        res["date"] = res.index
        res = res.dropna(subset=["cgm", "pat_id", "seq_id"])
        resampled_parts.append(res.reset_index(drop=True))

    final_df = pd.concat(resampled_parts, ignore_index=True)
    seq_lengths = final_df.groupby("seq_id").size()
    short_seqs = set(seq_lengths[seq_lengths < cfg.min_sequence_steps].index)
    if short_seqs:
        final_df = final_df[~final_df["seq_id"].isin(short_seqs)]

    # Attach weight per sequence from summary workbook
    weight_map: Dict[str, float] = {}
    summary_path = raw_dir / cfg.summary_weight_file
    if summary_path.exists():
        try:
            summary_df = pd.read_excel(summary_path)
            cols_lower = {c.lower(): c for c in summary_df.columns}
            pn_col = next((cols_lower[c] for c in cols_lower if "patient number" in c), None)
            w_col = next((cols_lower[c] for c in cols_lower if "weight" in c), None)
            if pn_col and w_col:
                for _, row in summary_df.iterrows():
                    pn = str(row[pn_col]).strip()
                    stem = Path(pn).stem
                    w = row[w_col]
                    weight_map[stem] = float(w) if pd.notna(w) else np.nan
        except Exception:
            pass

    final_df["weight_kg"] = final_df["seq_stem"].map(weight_map).astype(float)
    final_df = final_df.drop(columns=["seq_stem"])
    # ensure identifiers are ints (avoid float formatting in CSV)
    final_df["pat_id"] = final_df["pat_id"].astype(int)
    final_df["seq_id"] = final_df["seq_id"].astype(int)

    bolus_total = (
        final_df["bolus_short"] + final_df["bolus_long"] + final_df["bolus_iv_short"] + final_df["bolus_iv_long"]
    )

    metadata = {
        "dataset": "Shanghai_T1DM",
        "source_files": [p.name for p in files],
        "patients_final": int(final_df["pat_id"].nunique()),
        "sequences_final": int(final_df["seq_id"].nunique()),
        "rows_final": int(len(final_df)),
        "basal_points": int(len(final_df)),
        "bolus_points": int((bolus_total != 0).sum()),
        "bolus_short_points": int((final_df["bolus_short"] != 0).sum()),
        "bolus_long_points": int((final_df["bolus_long"] != 0).sum()),
        "bolus_iv_short_points": int((final_df["bolus_iv_short"] != 0).sum()),
        "bolus_iv_long_points": int((final_df["bolus_iv_long"] != 0).sum()),
        "iv_glucose_points": int((final_df["iv_glucose_g"] != 0).sum()),
        "min_sequence_steps": cfg.min_sequence_steps,
        "resample_freq_minutes": cfg.resample_freq_minutes,
        "weights_records": len(weight_map),
        "insulin_index_file": cfg.insulin_index_file,
        "unmapped_insulin_labels": sorted(label for label in unknown_labels if label),
        "channels": [
            "cgm",
            "basal",
            "bolus_short",
            "bolus_long",
            "bolus_iv_short",
            "bolus_iv_long",
            "iv_glucose_g",
            "weight_kg",
            "meal",
        ],
        "notes": "Parsed per-file sequences, mapped pat_id to zero-based, global seq_id assigned across all sequences; meal is binary from non-empty dietary intake; weight_kg attached per sequence from summary workbook when available; resampled to 5-minute grid. Free-text subcutaneous insulin is split into short vs long acting using insulin_index_shanghai.json; unmapped labels default to short acting.",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    final_df = final_df.sort_values(["pat_id", "seq_id", "date"]).reset_index(drop=True)
    final_df["date"] = final_df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    final_df = final_df[
        [
            "pat_id",
            "seq_id",
            "date",
            "cgm",
            "basal",
            "bolus_short",
            "bolus_long",
            "bolus_iv_short",
            "bolus_iv_long",
            "iv_glucose_g",
            "weight_kg",
            "meal",
        ]
    ]
    final_df.to_csv(output_dir / "timeseries.csv", index=False)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
