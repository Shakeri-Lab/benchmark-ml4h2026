from __future__ import annotations

import numpy as np
import pandas as pd

# Ensure repo root is on sys.path
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from analysis.counterfactual_metrics import build_results  # noqa: E402


def _make_df() -> pd.DataFrame:
    # One episode with factual + two counterfactuals
    # Horizon steps = 3 (5,10,15 minutes)
    gts_f = np.array([100.0, 100.0, 100.0])
    preds_f = np.array([100.0, 100.0, 100.0])

    # Counterfactual 1: +10 effect
    gts_c1 = np.array([110.0, 110.0, 110.0])
    preds_c1 = np.array([108.0, 112.0, 110.0])

    # Counterfactual 2: -10 effect
    gts_c2 = np.array([90.0, 90.0, 90.0])
    preds_c2 = np.array([92.0, 88.0, 90.0])

    rows = [
        {
            "pat_id": 1,
            "seq_id": 1,
            "scenario_name": "factual",
            "scenario_type": "factual",
            "perturb_family": "none",
            "gts": gts_f,
            "preds": preds_f,
        },
        {
            "pat_id": 1,
            "seq_id": 1,
            "scenario_name": "basal_2x",
            "scenario_type": "counterfactual",
            "perturb_family": "basal",
            "gts": gts_c1,
            "preds": preds_c1,
        },
        {
            "pat_id": 1,
            "seq_id": 1,
            "scenario_name": "basal_0x",
            "scenario_type": "counterfactual",
            "perturb_family": "basal",
            "gts": gts_c2,
            "preds": preds_c2,
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    df = _make_df()
    results = build_results(
        df,
        horizon_minutes=[15],
        step_minutes=5,
        zero_tol=1e-6,
        tol_true=1e-6,
        tol_pred=1e-6,
        min_group_size=3,
        n_bootstrap=10,
        bootstrap_seed=42,
    )
    records = results["records"]

    # rmse_effect: true effect is +/-10; predicted effects are +/-10 at 15 minutes -> rmse 0
    rmse_rec = next(
        r for r in records if r["metric_name"] == "rmse_effect" and r["perturbation_type"] == "__ALL__"
    )
    assert abs(rmse_rec["metric_value"]) < 1e-6, f"Unexpected rmse_effect: {rmse_rec}"

    # sign agreement should be 1.0 for both counterfactuals
    sign_rec = next(
        r for r in records if r["metric_name"] == "sign_agreement" and r["perturbation_type"] == "__ALL__"
    )
    assert abs(sign_rec["metric_value"] - 1.0) < 1e-6, f"Unexpected sign_agreement: {sign_rec}"

    # ordinal metrics for basal group should be perfect
    kendall = next(r for r in records if r["metric_name"] == "kendall_rank_corr")
    exact = next(r for r in records if r["metric_name"] == "ordinal_exact_match")
    pair = next(r for r in records if r["metric_name"] == "ordinal_pairwise_accuracy")
    assert abs(kendall["metric_value"] - 1.0) < 1e-6, f"Unexpected kendall: {kendall}"
    assert abs(exact["metric_value"] - 1.0) < 1e-6, f"Unexpected exact: {exact}"
    assert abs(pair["metric_value"] - 1.0) < 1e-6, f"Unexpected pairwise: {pair}"

    print("Sanity check passed.")


if __name__ == "__main__":
    main()
