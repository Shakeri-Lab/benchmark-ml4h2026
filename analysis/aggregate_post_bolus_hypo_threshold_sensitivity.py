from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-model post-bolus hypo threshold sensitivity CSV parts.")
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    parts = sorted(args.parts_dir.glob("*.csv"))
    if not parts:
        raise SystemExit(f"No CSV parts found in {args.parts_dir}")
    frames = [pd.read_csv(p) for p in parts]
    df = pd.concat(frames, ignore_index=True)
    sort_cols = [
        "dataset_name",
        "prediction_horizon_minutes",
        "post_bolus_threshold_minutes_requested",
        "slice",
        "model_name",
    ]
    existing = [c for c in sort_cols if c in df.columns]
    if existing:
        df = df.sort_values(existing).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"[+] Aggregated {len(parts)} parts / {len(df)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
