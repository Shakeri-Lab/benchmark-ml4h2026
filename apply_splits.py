"""
Apply frozen split specs to produce per-split CSVs for each dataset.

Default behavior:
- Finds specs in preprocessing/splits/*.json
- Loads datasets_processed/<dataset>/timeseries.csv
- Writes split CSVs to datasets_processed/<dataset>/splits/<spec_stem>/{train,val,test}.csv

You can override the spec directory/file(s) and output root via CLI flags.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

import pandas as pd


def load_specs(spec_dir: Path, spec_files: List[Path]) -> List[Path]:
    if spec_files:
        return [p for p in spec_files if p.exists()]
    return sorted(spec_dir.glob("*.json"))


def normalize_id(val) -> str:
    try:
        f = float(val)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except Exception:
        pass
    return str(val).strip()


def normalize_ids(ids: Iterable) -> set:
    return {normalize_id(x) for x in ids if str(x).strip() != ""}


def apply_spec(
    spec_path: Path,
    timeseries_root: Path,
    out_root: Path,
    dataset_filter: str | None = None,
) -> None:
    spec = json.loads(spec_path.read_text())
    dataset = spec.get("dataset")
    if dataset_filter and dataset != dataset_filter:
        return

    # Try primary root, then fallbacks for private data
    ts_roots = [timeseries_root, Path("dataset_processed"), Path("datasets_preprocessed")]
    ts_path = None
    used_root = None
    for root in ts_roots:
        candidate = root / dataset / "timeseries.csv"
        if candidate.exists():
            ts_path = candidate
            used_root = root
            break
    if ts_path is None:
        raise FileNotFoundError(f"timeseries not found for {dataset} in {ts_roots}")

    df = pd.read_csv(ts_path)
    if "pat_id" not in df.columns:
        if "pid" in df.columns:
            df = df.rename(columns={"pid": "pat_id"})
        else:
            raise ValueError(f"'pat_id' column missing in {ts_path}")
    df["pat_id_norm"] = df["pat_id"].apply(normalize_id)

    patients: Dict[str, List] = spec.get("patients", {})
    spec_stem = spec_path.stem
    # If the caller left out_root at default datasets_processed but we used a different root, mirror the used root.
    out_base = out_root
    if out_root == Path("datasets_processed") and used_root is not None and out_root != used_root:
        out_base = used_root
    out_dir = out_base / dataset / "splits" / spec_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, ids in patients.items():
        ids_norm = normalize_ids(ids)
        split_df = df[df["pat_id_norm"].isin(ids_norm)].drop(columns=["pat_id_norm"])
        out_file = out_dir / f"{split_name}.csv"
        split_df.to_csv(out_file, index=False)
        print(f"[{dataset}] {split_name}: {len(split_df)} rows -> {out_file}")

    # Save a copy of the spec for provenance
    (out_dir / spec_path.name).write_text(json.dumps(spec, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply split specs to timeseries files.")
    parser.add_argument(
        "--spec-dir",
        type=Path,
        default=Path("configs/preprocessing"),
        help="Directory containing split spec JSON files (default: configs/preprocessing)",
    )
    parser.add_argument(
        "--spec-file",
        type=Path,
        action="append",
        help="Specific split spec file(s) to apply (can be repeated)",
    )
    parser.add_argument(
        "--dataset",
        help="Only apply specs for this dataset (default: all specs in the directory)",
    )
    parser.add_argument(
        "--timeseries-root",
        type=Path,
        default=Path("datasets_processed"),
        help="Root directory containing <dataset>/timeseries.csv files",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("datasets_processed"),
        help="Root directory where split CSVs will be written",
    )
    args = parser.parse_args()

    specs = load_specs(args.spec_dir, args.spec_file or [])
    if not specs:
        raise FileNotFoundError(f"No split spec files found in {args.spec_dir}")

    for spec_path in specs:
        apply_spec(spec_path, args.timeseries_root, args.out_root, dataset_filter=args.dataset)


if __name__ == "__main__":
    main()
