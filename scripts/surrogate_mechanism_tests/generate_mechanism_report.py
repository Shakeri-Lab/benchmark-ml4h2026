from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from scripts.surrogate_mechanism_tests.pareto_utils import dominated_hypervolume_grid, normalize_minmax, pareto_dominated  # noqa: E402


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _capacity_analysis(general: pd.DataFrame, frontier_ops: pd.DataFrame, out: Path) -> pd.DataFrame:
    if general.empty:
        return pd.DataFrame()
    overall = general[general["slice"].eq("overall")].copy()
    if "horizon_eval" in overall.columns:
        preferred = overall[overall["horizon_eval"].eq("through_30min")].copy()
        if not preferred.empty:
            overall = preferred
    if "capacity" not in overall.columns:
        overall["capacity"] = "unspecified"
    recall = frontier_ops[frontier_ops["target_fa_day"].eq(1.0)][["dataset", "method", "recall_at_target_fa"]].copy() if not frontier_ops.empty else pd.DataFrame()
    if not recall.empty:
        merged = overall.merge(recall, on=["dataset", "method"], how="left")
    else:
        merged = overall.copy()
        merged["recall_at_target_fa"] = np.nan
    metrics = merged[["rmse", "peg_unsafe", "recall_at_target_fa"]].copy()
    metrics["negative_recall_at_1fa"] = -metrics["recall_at_target_fa"].fillna(0.0)
    merged["negative_recall_at_1fa"] = metrics["negative_recall_at_1fa"]
    pts = normalize_minmax(metrics[["rmse", "peg_unsafe", "negative_recall_at_1fa"]].to_numpy())
    merged["pareto_dominated_3d"] = pareto_dominated(pts)
    hv_rows = []
    for loss, g in merged.groupby("method"):
        p = normalize_minmax(g[["rmse", "peg_unsafe", "negative_recall_at_1fa"]].to_numpy())
        hv_rows.append({"method": loss, "hypervolume_approx": dominated_hypervolume_grid(p, grid_size=40)})
    hv = pd.DataFrame(hv_rows)
    merged.to_csv(out / "all_pareto_metrics.csv", index=False)
    hv.to_csv(out / "hypervolume_summary.csv", index=False)
    return merged


def run(
    *,
    decision_dir: Path,
    general_dir: Path,
    capacity_dir: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(exist_ok=True)
    curves = _read_optional(decision_dir / "decision_frontier_curves.csv")
    ops = _read_optional(decision_dir / "decision_frontier_operating_points.csv")
    pauc = _read_optional(decision_dir / "decision_frontier_pauc.csv")
    deltas = _read_optional(decision_dir / "threshold_equivalent_deltas.csv")
    general = _read_optional(general_dir / "general_metrics_summary.csv")
    general_primary = general
    if not general.empty and "horizon_eval" in general.columns:
        preferred = general[general["horizon_eval"].eq("through_30min")].copy()
        if not preferred.empty:
            general_primary = preferred
    capacity_general = _read_optional(capacity_dir / "analysis" / "capacity_sweep_metrics.csv")
    if capacity_general.empty:
        capacity_general = general_primary
    pareto = _capacity_analysis(capacity_general, ops, output_dir)

    for name, df in [
        ("all_decision_frontier_metrics.csv", pd.concat([ops, pauc], ignore_index=True, sort=False) if not ops.empty or not pauc.empty else pd.DataFrame()),
        ("all_summary_metrics.csv", general_primary),
        ("threshold_equivalent_deltas.csv", deltas),
    ]:
        df.to_csv(output_dir / name, index=False)

    lines = [
        "# Mechanism Frontier Test Report",
        "",
        "This report separates two possible mechanisms for clinically structured losses: decision-threshold movement and capacity allocation. Wolff and GLIMMER are treated as fixed clinically motivated comparators; the goal is mechanism analysis, not criticism of those methods.",
        "",
        "## Hypothesis 1: Decision-Maker Effect",
        "",
        "The primary alarm analysis uses trajectory-based hypoglycemia detection, matching the MLHC protocol: true events require CGM <70 mg/dL for at least 15 minutes, alarms fire when any value in the prediction trajectory crosses the swept threshold, a prediction-horizon refractory period is applied, and lead time is measured from alarm time to event onset. Recall/false-alarm frontiers are compared against the MSE frontier.",
    ]
    if not deltas.empty and "delta_recall_vs_MSE_frontier" in deltas:
        mean_delta = deltas["delta_recall_vs_MSE_frontier"].mean()
        nonpos = int((deltas["delta_recall_vs_MSE_frontier"] <= 0).sum())
        lines.append(f"Mean threshold-equivalent recall delta versus the MSE frontier is {mean_delta:.4f}; {nonpos}/{len(deltas)} operating-point comparisons are non-positive.")
    else:
        lines.append("Decision-frontier deltas were not available yet. Run `evaluate_single_lag_frontier.py` after comparator training completes.")
    lines += [
        "",
        "## Hypothesis 2: Capacity Allocation",
        "",
        "The capacity analysis scales PatchMLP down from the default configuration and compares clinical-loss points against MSE points of equal or larger capacity. Dominance and hypervolume are computed with RMSE, PEG unsafe fraction, and negative recall@1FA/day as minimization objectives.",
    ]
    if not pareto.empty:
        dominated = int(pareto["pareto_dominated_3d"].sum())
        lines.append(f"Current Pareto table contains {len(pareto)} points; {dominated} are dominated in the 3D clinical frontier.")
    else:
        lines.append("Capacity-sweep metrics were not available yet. Run the capacity pilot/sweep and aggregation before interpreting this mechanism.")
    lines += [
        "",
        "## Hypothesis 3: Evaluation/Decision Utility",
        "",
        "If weighted-loss models do not improve the trajectory recall/FA frontier or are dominated by higher-capacity MSE models, the evidence supports using clinical losses as evaluation or decision utilities rather than standalone deterministic regression objectives.",
        "",
        "## Caveats",
        "",
        "- The primary alarm experiment uses thresholded point trajectories, not probabilistic forecasts.",
        "- Multi-horizon decisions may require horizon-dependent uncertainty or a variance head.",
        "- GLIMMER uses fixed Transformer-average weights from Table 3; no GA search is run.",
    ]
    (output_dir / "mechanism_report.md").write_text("\n".join(lines) + "\n")
    (output_dir / "mechanism_report_summary.json").write_text(
        json.dumps(
            {
                "has_decision_curves": bool(not curves.empty),
                "has_general_metrics": bool(not general.empty),
                "has_capacity_metrics": bool(not pareto.empty),
            },
            indent=2,
        )
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mechanism-frontier report tables and narrative.")
    parser.add_argument("--decision-dir", type=Path, default=Path("outputs/surrogate/mechanism_frontier_tests/decision_frontier"))
    parser.add_argument("--general-dir", type=Path, default=Path("outputs/surrogate/mechanism_frontier_tests/general_metrics"))
    parser.add_argument("--capacity-dir", type=Path, default=Path("outputs/surrogate/mechanism_frontier_tests/capacity_sweep"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/surrogate/mechanism_frontier_tests/final_report"))
    args = parser.parse_args()
    print(run(decision_dir=args.decision_dir, general_dir=args.general_dir, capacity_dir=args.capacity_dir, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
