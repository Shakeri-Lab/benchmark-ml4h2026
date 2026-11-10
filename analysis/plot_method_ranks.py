#!/usr/bin/env python
"""
Create a boxplot of method ranks across datasets, horizons, and metrics.

Uses metrics.json from outputs/results/<run_id>/metrics.json where run_id follows:
  <dataset>_<method>_mv

Ranks are computed per (dataset, horizon, metric) across methods (lower is better).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _load_per_horizon_metric(run_dir: Path, metric: str) -> List[float] | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path, "r") as f:
        data = json.load(f)
    per_h = data.get("metrics", {}).get("per_horizon", None)
    if not isinstance(per_h, list):
        return None
    vals = []
    for h in per_h:
        if not isinstance(h, dict) or metric not in h:
            vals.append(np.nan)
        else:
            vals.append(h[metric])
    return vals


def _rank_with_ties(values: Dict[str, float]) -> Dict[str, float]:
    # Lower is better
    items = sorted(values.items(), key=lambda kv: kv[1])
    ranks: Dict[str, float] = {}
    i = 0
    while i < len(items):
        j = i + 1
        while j < len(items) and np.isclose(items[j][1], items[i][1]):
            j += 1
        # average rank for ties (1-indexed)
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[items[k][0]] = avg_rank
        i = j
    return ranks


def main():
    root = Path("outputs/results")
    out_dir = Path("analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["dclp3", "dclp5", "pedap"]
    methods = ["arimax", "patchmlp", "attengluco", "itransformer"]
    metrics = ["rmse", "grmse", "seg"]
    # 15, 30, 45, 60 minutes (steps of 5 minutes) => indices 2,5,8,11
    horizon_steps = [3, 6, 9, 12]
    horizon_indices = [h - 1 for h in horizon_steps]

    # Gather ranks
    ranks_by_method: Dict[str, List[float]] = {m: [] for m in methods}
    missing = []

    for dataset in datasets:
        for metric in metrics:
            for h_idx in horizon_indices:
                values: Dict[str, float] = {}
                for method in methods:
                    run_id = f"{dataset}_{method}_mv"
                    vals = _load_per_horizon_metric(root / run_id, metric)
                    if vals is None or h_idx >= len(vals) or np.isnan(vals[h_idx]):
                        missing.append((dataset, method, metric, h_idx))
                        continue
                    values[method] = float(vals[h_idx])
                if len(values) < 2:
                    continue
                ranks = _rank_with_ties(values)
                for method, r in ranks.items():
                    ranks_by_method[method].append(r)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [ranks_by_method[m] for m in methods]
    ax.boxplot(data, labels=methods, showmeans=True)
    ax.set_title("Method Rank Distribution Across Datasets / Horizons / Metrics")
    ax.set_ylabel("Rank (lower is better)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    legend_handles = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor="C0", markersize=8, label="Mean"),
        Line2D([0], [0], color="C1", linewidth=2, label="Median"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="C0", markersize=6, label="Outliers"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)

    out_path = out_dir / "method_rank_boxplot.pdf"
    fig.tight_layout()
    fig.savefig(out_path)

    print(f"[✓] Saved boxplot to {out_path}")
    if missing:
        print(f"[i] Missing values for {len(missing)} entries (skipped).")


if __name__ == "__main__":
    main()
