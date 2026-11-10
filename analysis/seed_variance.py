"""Quantify training (seed) variance in safety-gating metrics.

The gating evaluation bootstraps over patients, capturing sampling variance but
not variance from retraining. This aggregates seed replicates of the same
model/cohort/horizon and reports the spread in event recall and false alarms per
patient-day, so cross-model recall comparisons can be given honest error bars --
or dropped where the spread swamps the differences being claimed.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def collect(seedvar_dir: Path, common_dir: Path, cohort: str, horizon: int) -> pd.DataFrame:
    rows = []
    # seed 42 lives in the main common-window outputs
    for f in sorted(common_dir.glob("*.csv")):
        m = re.fullmatch(rf"(.+)_{cohort}_p(\d+)", f.stem)
        if not m:
            continue
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        d["model"] = m.group(1)
        d["seed"] = 42
        rows.append(d)
    # replicate seeds
    for f in sorted(seedvar_dir.glob("*.csv")):
        m = re.fullmatch(rf"(.+)_{cohort}_p(\d+)_s(\d+)", f.stem)
        if not m:
            continue
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        d["model"] = m.group(1)
        d["seed"] = int(m.group(3))
        rows.append(d)
    if not rows:
        raise SystemExit("no seed replicates found")
    df = pd.concat(rows, ignore_index=True)
    return df[df["prediction_horizon_minutes"] == horizon]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seedvar-dir", type=Path, default=Path("outputs/gating_seedvar"))
    p.add_argument("--common-dir", type=Path, default=Path("outputs/gating_common"))
    p.add_argument("--cohort", default="dclp3")
    p.add_argument("--horizon", type=int, default=30)
    a = p.parse_args()

    df = collect(a.seedvar_dir, a.common_dir, a.cohort, a.horizon)
    models_with_reps = (df.groupby("model")["seed"].nunique() > 1)
    keep = set(models_with_reps[models_with_reps].index)
    if not keep:
        raise SystemExit("no model has more than one seed yet")

    for sl in ["overall", "post_bolus"]:
        s = df[(df["slice"] == sl) & (df["model"].isin(keep))]
        if s.empty:
            continue
        print(f"\n=== {sl} ({a.cohort.upper()}, {a.horizon} min) ===")
        print(f"{'model':<18} {'n':>2} {'recall mean':>12} {'sd':>7} {'min':>7} {'max':>7} {'range':>7}")
        summary = []
        for m, g in s.groupby("model"):
            r = g["event_recall"].dropna()
            if len(r) < 2:
                continue
            summary.append((m, len(r), r.mean(), r.std(), r.min(), r.max(), r.max() - r.min()))
        for m, n, mu, sd, lo, hi, rng in sorted(summary, key=lambda x: -x[6]):
            print(f"{m:<18} {n:>2} {mu:>12.4f} {sd:>7.4f} {lo:>7.4f} {hi:>7.4f} {rng:>7.4f}")

        if summary:
            spreads = [x[6] for x in summary]
            means = [x[2] for x in summary]
            print(f"\n  largest within-model recall range : {max(spreads):.4f}")
            print(f"  spread of between-model means     : {max(means)-min(means):.4f}")
            if max(spreads) > 0.5 * (max(means) - min(means)):
                print("  -> WARNING: retraining noise is comparable to between-model differences;")
                print("     single-seed cross-model recall rankings are not supportable.")
            else:
                print("  -> between-model differences exceed retraining noise.")


if __name__ == "__main__":
    main()
