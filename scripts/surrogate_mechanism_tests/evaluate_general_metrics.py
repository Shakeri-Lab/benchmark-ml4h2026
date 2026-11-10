from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.append(str(Path(__file__).resolve().parents[2]))

from metrics.clinical import ClinicalMetrics, zone_merge_counts  # noqa: E402
from scripts.fit_threshold_crossing_kovatchev import DKConstants, threshold_loss_np  # noqa: E402
from scripts.surrogate_mechanism_tests.evaluate_single_lag_frontier import (  # noqa: E402
    _horizon_index,
    _load_split,
    _resolve_run_dir,
)


TC_PARAMS = {
    "lambda_70": 16.03929692527233,
    "lambda_180": 15.446434228916164,
    "c": 0.17466771047389293,
}


def _slice_mask(y: np.ndarray, name: str) -> np.ndarray:
    if name == "overall":
        return np.ones_like(y, dtype=bool)
    if name == "hypo":
        return y < 70.0
    if name == "normal":
        return (y >= 70.0) & (y < 180.0)
    if name == "hyper":
        return y >= 180.0
    if name == "severe_hyper":
        return y >= 250.0
    raise ValueError(name)


def _metrics_for_arrays(y: np.ndarray, p: np.ndarray, clinical: ClinicalMetrics) -> dict[str, float]:
    mask = np.isfinite(y) & np.isfinite(p)
    y = y[mask]
    p = p[mask]
    if y.size == 0:
        return {k: float("nan") for k in ["rmse", "mae", "mean_seg", "seg95", "peg_unsafe", "tc_loss", "bias", "predicted_low_rate", "true_low_rate", "predicted_high_rate", "true_high_rate"]}
    diff = p - y
    seg = clinical.SEG(y, p, reduction="none")
    peg = clinical.peg_zone(y, p)
    counts = np.bincount(np.asarray(peg).reshape(-1).astype(int), minlength=9)
    merged = zone_merge_counts(counts)
    tc = threshold_loss_np(y, p, constants=DKConstants(glucose_min=40.0, glucose_max=400.0, delta=0.5, p_L=0.0, p_H=0.0, eps=1e-8), **TC_PARAMS)
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "mean_seg": float(np.mean(seg)),
        "seg95": float(np.percentile(seg, 95)),
        "peg_unsafe": float(merged[2:5].sum() / merged.sum()) if merged.sum() else float("nan"),
        "tc_loss": float(np.mean(tc)),
        "bias": float(np.mean(diff)),
        "predicted_low_rate": float(np.mean(p < 70.0)),
        "true_low_rate": float(np.mean(y < 70.0)),
        "predicted_high_rate": float(np.mean(p > 180.0)),
        "true_high_rate": float(np.mean(y > 180.0)),
    }


def run(config: Path) -> Path:
    cfg = yaml.safe_load(config.read_text())
    out = Path(cfg.get("output_dir", "outputs/surrogate/mechanism_frontier_tests/general_metrics"))
    out.mkdir(parents=True, exist_ok=True)
    model = str(cfg.get("model", "patchmlp")).lower()
    seed = int(cfg.get("seed", 42))
    horizon = int(cfg.get("horizon_minutes", 30))
    aggregate_horizons = bool(cfg.get("aggregate_horizons_through_target", True))
    clinical = ClinicalMetrics.from_default()

    rows = []
    patient_rows = []
    for dataset in cfg["datasets"]:
        dataset = str(dataset).lower()
        for method in cfg["methods"]:
            name = str(method["name"])
            run_dir = _resolve_run_dir(method, model=model, dataset=dataset, seed=seed)
            data = _load_split(run_dir, "test")
            idx = _horizon_index(data, horizon)
            y_all = data["y_gt"].astype(float)
            p_all = data["y_hat"].astype(float)
            pid = np.asarray(data.get("patient_id", np.arange(y_all.shape[0]))).astype(str)
            eval_sets = [(f"exact_{horizon}min", y_all[:, idx], p_all[:, idx])]
            if aggregate_horizons:
                eval_sets.append((f"through_{horizon}min", y_all[:, : idx + 1], p_all[:, : idx + 1]))
            for horizon_eval, y, p in eval_sets:
                if y.ndim == 1:
                    pid_eval = pid
                else:
                    pid_eval = np.repeat(pid, y.shape[1])
                    y = y.reshape(-1)
                    p = p.reshape(-1)
                for slice_name in ["overall", "hypo", "normal", "hyper", "severe_hyper"]:
                    m = _slice_mask(y, slice_name)
                    vals = _metrics_for_arrays(y[m], p[m], clinical)
                    rows.append(
                        {
                            "dataset": dataset.upper(),
                            "method": name,
                            "slice": slice_name,
                            "horizon_minutes": horizon,
                            "horizon_eval": horizon_eval,
                            **vals,
                        }
                    )
                for pat in np.unique(pid_eval):
                    pm = pid_eval == pat
                    vals = _metrics_for_arrays(y[pm], p[pm], clinical)
                    patient_rows.append(
                        {
                            "dataset": dataset.upper(),
                            "method": name,
                            "patient_id": pat,
                            "horizon_minutes": horizon,
                            "horizon_eval": horizon_eval,
                            **vals,
                        }
                    )

    summary = pd.DataFrame(rows)
    per_patient = pd.DataFrame(patient_rows)
    summary.to_csv(out / "general_metrics_summary.csv", index=False)
    per_patient.to_csv(out / "general_metrics_per_patient.csv", index=False)
    (out / "general_metrics_summary.json").write_text(json.dumps(json.loads(summary.to_json(orient="records")), indent=2))
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-mechanism-frontier")
        os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
        with contextlib.redirect_stderr(io.StringIO()):
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

        plots = out / "plots"
        plots.mkdir(exist_ok=True)
        overall = summary[summary["slice"].eq("overall") & summary["horizon_eval"].eq(f"through_{horizon}min")]
        for metric, ylabel in [("rmse", "RMSE at 30 min"), ("mean_seg", "Mean SEG at 30 min"), ("peg_unsafe", "PEG unsafe fraction")]:
            if metric not in overall:
                continue
            fig, ax = plt.subplots(figsize=(6.2, 3.6))
            labels = [f"{r.dataset}\n{r.method}" for r in overall.itertuples()]
            ax.bar(np.arange(len(overall)), overall[metric])
            ax.set_xticks(np.arange(len(overall)))
            ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
            ax.set_ylabel(ylabel)
            fig.tight_layout()
            fig.savefig(plots / f"{metric}_summary.pdf")
            plt.close(fig)
    except Exception as exc:
        plots = out / "plots"
        plots.mkdir(exist_ok=True)
        (plots / "plot_warning.txt").write_text(str(exc))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate general 30-minute clinical metrics for mechanism-frontier outputs.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
