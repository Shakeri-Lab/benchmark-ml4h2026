"""
Ad-hoc helper to generate patient-level split specs (train/val/test).
Writes JSON files to preprocessing/splits and keeps them frozen for reproducibility.

This file is tracked. The split it produces decides which patient is in test for
every model in the benchmark, so a frozen split whose generator is not in the
repository is only reproducible for as long as one copy of the script survives
on one filesystem.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np


def canon_pid(pid):
    try:
        f = float(pid)
        if np.isfinite(f) and f.is_integer():
            return int(f)
    except Exception:
        pass
    return str(pid).strip()


def read_patients(timeseries_path: Path) -> list:
    patients = set()
    with timeseries_path.open() as f:
        reader = csv.reader(f)
        header = next(reader, [])
        col = None
        for cand in ["pat_id", "pid"]:
            if cand in header:
                col = cand
                break
        if not col:
            raise ValueError(f"'pat_id' or 'pid' column not found in {timeseries_path}")
        idx = header.index(col)
        for row in reader:
            if idx >= len(row):
                continue
            pid_raw = row[idx].strip()
            if pid_raw == "":
                continue
            patients.add(canon_pid(pid_raw))
    return sorted(patients, key=lambda x: (str(type(x)), x))


def make_split(patients: list, seed: int, ratios: tuple[float, float, float]) -> dict:
    rng = random.Random(seed)
    pats = patients[:]
    rng.shuffle(pats)
    n = len(pats)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    n_test = n - n_train - n_val
    return {
        "train": pats[:n_train],
        "val": pats[n_train : n_train + n_val],
        "test": pats[n_train + n_val :],
        "counts": {"train": n_train, "val": n_val, "test": n_test, "total": n},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate patient-level split specs.")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., DCLP3)")
    parser.add_argument(
        "--timeseries",
        type=Path,
        required=True,
        help="Path to datasets_processed/<dataset>/timeseries.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(0.7, 0.1, 0.2),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios (sum should be 1.0)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("preprocessing/splits"),
        help="Output directory for split specs",
    )
    args = parser.parse_args()

    patients = read_patients(args.timeseries)
    split = make_split(patients, args.seed, tuple(args.ratios))
    args.outdir.mkdir(parents=True, exist_ok=True)
    name = f"{args.dataset.lower()}_patients_{int(args.ratios[0]*100)}_{int(args.ratios[1]*100)}_{int(args.ratios[2]*100)}_seed{args.seed}.json"
    spec = {
        "dataset": args.dataset,
        "split_type": "patient",
        "seed": args.seed,
        "ratios": {"train": args.ratios[0], "val": args.ratios[1], "test": args.ratios[2]},
        "patients": {k: v for k, v in split.items() if k in {"train", "val", "test"}},
        "patient_counts": split["counts"],
    }
    out_path = args.outdir / name
    out_path.write_text(json.dumps(spec, indent=2))
    print(f"Wrote {out_path} ({split['counts']})")


if __name__ == "__main__":
    main()
