#!/usr/bin/env python
"""
Boxplot of method ranks across datasets, horizons, and metrics for 15/30/45/60 min.

Uses metrics.json from outputs/results/best_<model>_<dataset>_p{3,6,9,12}/metrics.json.
Metrics:
- RMSE (lower is better)
- PEG C+D+E (percentage; lower is better)
- Hypo sensitivity (higher is better)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path("outputs/results")
OUT_PATH = Path("analysis/method_rank_boxplot_all_lags.pdf")
DATASETS = ["dclp3", "dclp5", "pedap", "shanghait1dm"]
METHODS = ["arimax", "itransformer", "patchmlp", "attengluco", "gru", "zoh"]
HORIZON_STEPS = [3, 6, 9, 12]


def _load_per_horizon(run_dir: Path) -> List[dict] | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    data = json.loads(metrics_path.read_text())
    per_h = data.get("metrics", {}).get("per_horizon", None)
    if not isinstance(per_h, list):
        return None
    return per_h


def _peg_cde_from_row(row: dict) -> float | None:
    peg = row.get("peg_ratio")
    if isinstance(peg, list) and len(peg) >= 5:
        return 100.0 * sum(peg[2:5])
    return None


def _hypo_false_alarm_from_row(row: dict) -> float | None:
    if "hypo_precision" not in row:
        return None
    try:
        return 1.0 - float(row["hypo_precision"])
    except Exception:
        return None


def _rank_with_ties(values: Dict[str, float], higher_is_better: bool = False) -> Dict[str, float]:
    items = sorted(values.items(), key=lambda kv: kv[1], reverse=higher_is_better)
    ranks: Dict[str, float] = {}
    i = 0
    while i < len(items):
        j = i + 1
        while j < len(items) and np.isclose(items[j][1], items[i][1]):
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[items[k][0]] = avg_rank
        i = j
    return ranks


def main() -> None:
    ranks_by_method: Dict[str, List[float]] = {m: [] for m in METHODS}
    missing = []

    for dataset in DATASETS:
        for h in HORIZON_STEPS:
            h_idx = h - 1
            # Metrics: rmse, peg_cde (lower is better), hypo_sens (higher is better)
            # RMSE
            values: Dict[str, float] = {}
            for method in METHODS:
                run_id = f"best_{method}_{dataset}_p{h}"
                per_h = _load_per_horizon(ROOT / run_id)
                if per_h is None or h_idx >= len(per_h):
                    missing.append((dataset, method, h, "rmse"))
                    continue
                row = per_h[h_idx]
                if "rmse" not in row:
                    missing.append((dataset, method, h, "rmse"))
                    continue
                values[method] = float(row["rmse"])
            if len(values) >= 2:
                ranks = _rank_with_ties(values, higher_is_better=False)
                for m, r in ranks.items():
                    ranks_by_method[m].append(r)

            # PEG C+D+E
            values = {}
            for method in METHODS:
                run_id = f"best_{method}_{dataset}_p{h}"
                per_h = _load_per_horizon(ROOT / run_id)
                if per_h is None or h_idx >= len(per_h):
                    missing.append((dataset, method, h, "peg_cde"))
                    continue
                row = per_h[h_idx]
                val = _peg_cde_from_row(row)
                if val is None:
                    missing.append((dataset, method, h, "peg_cde"))
                    continue
                values[method] = float(val)
            if len(values) >= 2:
                ranks = _rank_with_ties(values, higher_is_better=False)
                for m, r in ranks.items():
                    ranks_by_method[m].append(r)

            # Hypo sensitivity (hypo_recall) - higher is better
            values = {}
            for method in METHODS:
                run_id = f"best_{method}_{dataset}_p{h}"
                per_h = _load_per_horizon(ROOT / run_id)
                if per_h is None or h_idx >= len(per_h):
                    missing.append((dataset, method, h, "hypo_sens"))
                    continue
                row = per_h[h_idx]
                if "hypo_recall" not in row:
                    missing.append((dataset, method, h, "hypo_sens"))
                    continue
                values[method] = float(row["hypo_recall"])
            if len(values) >= 2:
                ranks = _rank_with_ties(values, higher_is_better=True)
                for m, r in ranks.items():
                    ranks_by_method[m].append(r)

            # Hypo false alarm rate = 1 - hypo_precision (lower is better)
            values = {}
            for method in METHODS:
                run_id = f"best_{method}_{dataset}_p{h}"
                per_h = _load_per_horizon(ROOT / run_id)
                if per_h is None or h_idx >= len(per_h):
                    missing.append((dataset, method, h, "hypo_fa"))
                    continue
                row = per_h[h_idx]
                val = _hypo_false_alarm_from_row(row)
                if val is None:
                    missing.append((dataset, method, h, "hypo_fa"))
                    continue
                values[method] = float(val)
            if len(values) >= 2:
                ranks = _rank_with_ties(values, higher_is_better=False)
                for m, r in ranks.items():
                    ranks_by_method[m].append(r)

    # Plot
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    data = [ranks_by_method[m] for m in METHODS]
    ax.boxplot(data, labels=METHODS, showmeans=True)
    ax.set_title("Method Rank Distribution (15/30/45/60 min, 4 datasets, 3 metrics)")
    ax.set_ylabel("Rank (lower is better)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    legend_handles = [
        Line2D([0], [0], marker="^", color="none", markerfacecolor="C0", markersize=8, label="Mean"),
        Line2D([0], [0], color="C1", linewidth=2, label="Median"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="C0", markersize=6, label="Outliers"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(OUT_PATH)

    print(f"[✓] Saved boxplot to {OUT_PATH}")
    if missing:
        print(f"[i] Missing values for {len(missing)} entries (skipped).")


if __name__ == "__main__":
    main()
