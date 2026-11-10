#!/usr/bin/env python
"""
Quick helper to build metadata.json for the private DCLP3 preprocessed CSV.

Expected columns: pid, date, cgm, bolus, meal (no basal).
Outputs metadata next to the CSV unless --out is provided.
This script is not tracked; keep it local for the private dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def make_metadata(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    if "pid" not in df.columns:
        raise ValueError("Expected a 'pid' column in the private CSV")
    df["pat_id"] = df["pid"].astype(str).str.strip()
    rows_final = len(df)
    patients_final = df["pat_id"].nunique()
    channels = [c for c in df.columns if c not in {"pid", "pat_id", "date", "seq_id"}]
    meta = {
        "dataset": "DCLP3-private",
        "source_files": ["datasets_processed/DCLP3-private/timeseries.csv"],
        "rows_final": int(rows_final),
        "patients_final": int(patients_final),
        "channels": channels,
        "notes": "Private preprocessed DCLP3 variant; columns pid,date,cgm,bolus,meal (no basal).",
    }
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("datasets_processed/DCLP3-private/timeseries.csv"),
        help="Path to the private preprocessed CSV.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to metadata.json (default: alongside CSV).",
    )
    args = ap.parse_args()

    meta = make_metadata(args.csv)
    out_path = args.out or args.csv.with_name("metadata.json")
    out_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote metadata to {out_path}")


if __name__ == "__main__":
    main()
