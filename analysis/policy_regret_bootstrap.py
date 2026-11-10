#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _bootstrap_means(values: np.ndarray, *, n_bootstrap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    n = values.shape[0]
    out = np.empty((n_bootstrap, values.shape[1]), dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        out[i] = values[idx].mean(axis=0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Macro-average and bootstrap policy-regret metrics over patients.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    patient_by_slice_path = args.run_dir / "policy_regret_patient_metrics_by_slice.csv"
    patient_path = args.run_dir / "policy_regret_patient_metrics.csv"
    if patient_by_slice_path.exists():
        df = pd.read_csv(patient_by_slice_path)
    elif patient_path.exists():
        df = pd.read_csv(patient_path)
        df.insert(0, "slice", "overall")
    else:
        raise FileNotFoundError(patient_by_slice_path)

    if df.empty:
        raise ValueError(f"No patient metrics found in {args.run_dir}")

    rows = []
    for slice_name, sub in df.groupby("slice", sort=False):
        numeric_cols = [
            c
            for c in sub.columns
            if c not in {"slice", "pat_id"} and pd.api.types.is_numeric_dtype(sub[c])
        ]
        if not numeric_cols:
            continue
        values = sub[numeric_cols].to_numpy(dtype=np.float64)
        boot = _bootstrap_means(values, n_bootstrap=args.n_bootstrap, seed=args.seed)
        for j, metric in enumerate(numeric_cols):
            point = float(values[:, j].mean())
            lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
            rows.append(
                {
                    "slice": str(slice_name),
                    "metric": metric,
                    "macro_mean": point,
                    "ci_lower": float(lo),
                    "ci_upper": float(hi),
                    "n_patients": int(len(sub)),
                }
            )

    out_df = pd.DataFrame(rows)
    out_csv = args.run_dir / "policy_regret_bootstrap_by_slice.csv"
    out_json = args.run_dir / "policy_regret_bootstrap_by_slice.json"
    out_df.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(rows, indent=2))

    overall_df = out_df.loc[out_df["slice"] == "overall"].drop(columns=["slice"], errors="ignore")
    overall_df.to_csv(args.run_dir / "policy_regret_bootstrap.csv", index=False)
    (args.run_dir / "policy_regret_bootstrap.json").write_text(json.dumps(overall_df.to_dict("records"), indent=2))

    for slice_name, sub in out_df.groupby("slice", sort=False):
        print(f"\n=== {slice_name} ===")
        print(sub.drop(columns=["slice"]).to_string(index=False))


if __name__ == "__main__":
    main()
