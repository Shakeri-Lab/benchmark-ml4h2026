from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_val_rmse(run_dir: Path) -> float:
    metrics = run_dir / "metrics_val.json"
    if not metrics.exists():
        metrics = run_dir / "prediction_pairs_summary.json"
    if not metrics.exists():
        return float("nan")
    obj = json.loads(metrics.read_text())
    if "rmse" in obj:
        return float(obj["rmse"])
    return float(obj.get("metrics", {}).get("val", {}).get("rmse", float("nan")))


def run(pilot_root: Path, output_path: Path) -> Path:
    rows = []
    for run in sorted(pilot_root.glob("mse/cap_*x/patchmlp_*/seed_*")):
        parts = run.parts
        cap = parts[-3]
        mult = cap.replace("cap_", "").replace("p", ".").replace("x", "")
        rows.append({"capacity": cap, "multiplier": float(mult), "run_dir": str(run), "val_rmse": _read_val_rmse(run)})
    df = pd.DataFrame(rows).sort_values("multiplier", ascending=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path.with_suffix(".csv"), index=False)
    if df.empty:
        selected = {"default": 1.0, "medium_small": 0.5, "small": 0.25, "selection_status": "fallback_no_pilot_results"}
    else:
        finite = df[df["val_rmse"].notna()].copy()
        if finite.empty:
            selected = {"default": 1.0, "medium_small": 0.5, "small": 0.25, "selection_status": "fallback_no_finite_val_rmse"}
        else:
            default_rmse = float(finite.iloc[0]["val_rmse"])
            finite["relative_rmse"] = finite["val_rmse"] / max(default_rmse, 1e-12)
            medium = finite[finite["relative_rmse"].between(1.05, 1.35)]
            small = finite[finite["relative_rmse"].between(1.20, 2.50)]
            selected = {
                "default": float(finite.iloc[0]["multiplier"]),
                "medium_small": float((medium.iloc[-1] if not medium.empty else finite.iloc[min(len(finite) - 1, 2)])["multiplier"]),
                "small": float((small.iloc[-1] if not small.empty else finite.iloc[-1])["multiplier"]),
                "selection_status": "selected_from_validation_rmse",
                "default_val_rmse": default_rmse,
            }
    output_path.write_text(json.dumps(selected, indent=2))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Select PatchMLP capacity levels from MSE pilot validation RMSE.")
    parser.add_argument("--pilot-root", type=Path, default=Path("outputs/surrogate/mechanism_frontier_tests/capacity_sweep/pilot"))
    parser.add_argument("--output", type=Path, default=Path("outputs/surrogate/mechanism_frontier_tests/capacity_sweep/selected_capacities.json"))
    args = parser.parse_args()
    print(run(args.pilot_root, args.output))


if __name__ == "__main__":
    main()
