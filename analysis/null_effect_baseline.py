"""Null-effect reference for the counterfactual arm.

Effect RMSE alone cannot distinguish "predicts the intervention effect well"
from "predicts no intervention effect at all". A model that always outputs
delta_hat = 0 scores eRMSE = RMS(delta_true), which on this benchmark is the
*best* number in the table: on the Control-IQ bolus menu DLinear posts
eRMSE 14.22 with sign agreement 0.008, i.e. it ignores the insulin channel
entirely, yet beats every model that actually attempts an effect.

This script computes that reference by replacing each counterfactual scenario's
predictions with its own factual predictions -- forcing delta_hat = 0 -- and
running the unmodified metric code path, so the pairing, horizon resolution and
validity filtering are identical to the reported numbers. It then reports

    effect skill = 1 - eRMSE_model / eRMSE_null

which is > 0 only when a model beats predicting no effect at all, 0 when it
matches the null, and < 0 when it is worse than silence.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.counterfactual_metrics import (  # noqa: E402
    _load_artifacts,
    _load_valid_scenarios,
    _filter_valid,
    _resolve_horizon_indices,
    compute_effect_and_sign_metrics,
)


def _nullify_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Replace every counterfactual's preds with its episode's factual preds."""
    factual_preds = {
        (int(r.pat_id), int(r.seq_id)): np.asarray(r.preds)
        for r in df.itertuples(index=False)
        if str(r.scenario_name).lower() == "factual"
    }
    out = df.copy()
    new_preds = []
    for r in df.itertuples(index=False):
        key = (int(r.pat_id), int(r.seq_id))
        if str(r.scenario_name).lower() == "factual" or key not in factual_preds:
            new_preds.append(np.asarray(r.preds))
        else:
            new_preds.append(factual_preds[key].copy())
    out["preds"] = new_preds
    return out


def _records_to_frame(records, label: str) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df[df["perturbation_type"] == "__ALL__"]
    df = df.pivot_table(
        index=["perturbation_family", "horizon_minutes"],
        columns="metric_name",
        values="metric_value",
    ).reset_index()
    df.columns.name = None
    return df.rename(columns={"rmse_effect": f"eRMSE_{label}",
                              "sign_agreement": f"SA_{label}"})


def analyse(run_dir: Path, test_scenarios: Path, horizons, step_minutes: int,
            zero_tol: float) -> pd.DataFrame:
    npz = run_dir / "cf_artifacts.npz"
    df = _load_artifacts(npz)
    if test_scenarios.exists():
        df = _filter_valid(df, _load_valid_scenarios(test_scenarios))
    pred_len = int(np.asarray(df.iloc[0]["gts"]).reshape(-1).shape[0])
    hidx = _resolve_horizon_indices(horizons, step_minutes, pred_len)

    model = _records_to_frame(compute_effect_and_sign_metrics(df, hidx, zero_tol), "model")
    null = _records_to_frame(
        compute_effect_and_sign_metrics(_nullify_predictions(df), hidx, zero_tol), "null")

    out = model.merge(null, on=["perturbation_family", "horizon_minutes"])
    out.insert(0, "run", run_dir.name)
    out["effect_skill"] = 1.0 - out["eRMSE_model"] / out["eRMSE_null"]
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", type=Path, required=True,
                   help="Run directories containing cf_artifacts.npz")
    p.add_argument("--test-scenarios", type=Path,
                   default=Path("datasets_processed/action_cf_benchmark_controliq/test_scenarios.csv"))
    p.add_argument("--horizons", type=str, default="120")
    p.add_argument("--step-minutes", type=int, default=5)
    p.add_argument("--zero-tol", type=float, default=1e-6)
    p.add_argument("--out-csv", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    horizons = [int(x) for x in a.horizons.split(",") if x.strip()]
    frames = []
    for run in a.runs:
        try:
            frames.append(analyse(run, a.test_scenarios, horizons, a.step_minutes, a.zero_tol))
        except Exception as e:  # keep going; report which run failed
            print(f"[!] {run.name}: {e}", file=sys.stderr)
    if not frames:
        sys.exit("no runs produced metrics")
    res = pd.concat(frames, ignore_index=True)
    res = res.sort_values(["perturbation_family", "horizon_minutes", "eRMSE_model"])

    cols = ["run", "perturbation_family", "horizon_minutes",
            "eRMSE_model", "eRMSE_null", "effect_skill", "SA_model", "SA_null"]
    res = res[[c for c in cols if c in res.columns]]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(res.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    if a.out_csv:
        a.out_csv.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(a.out_csv, index=False)
        print(f"\n[+] wrote {a.out_csv}")


if __name__ == "__main__":
    main()
