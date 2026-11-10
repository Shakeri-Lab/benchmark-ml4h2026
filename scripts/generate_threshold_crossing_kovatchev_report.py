from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-threshold-crossing-report")

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parents[1]))

from metrics.clinical import ClinicalMetrics  # noqa: E402
from scripts.aggregate_threshold_crossing_kovatchev_results import (  # noqa: E402
    DATASETS,
    MODELS,
    _clinical_loss,
    _load_npz,
    _load_threshold_loss_config,
    _slice_mask,
)


MODEL_LABEL = {"gru": "GRU", "dlinear": "DLinear", "patchmlp": "PatchMLP", "itransformer": "iTransformer"}
DATASET_LABEL = {"dclp3": "DCLP3", "dclp5": "DCLP5", "pedap": "PEDAP", "azt1d": "AZT1D"}
SOURCE_LABEL = {"mse": "MSE-trained", "threshold_crossing": "TC-trained", "threshold_trained": "TC-trained"}


def _corr(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return {"pearson": np.nan, "pearson_p": np.nan, "spearman": np.nan, "spearman_p": np.nan, "n": int(mask.sum())}
    return {
        "pearson": float(pearsonr(x[mask], y[mask]).statistic),
        "pearson_p": float(pearsonr(x[mask], y[mask]).pvalue),
        "spearman": float(spearmanr(x[mask], y[mask]).statistic),
        "spearman_p": float(spearmanr(x[mask], y[mask]).pvalue),
        "n": int(mask.sum()),
    }


def _tex_num(x: Any, digits: int = 3) -> str:
    try:
        f = float(x)
    except Exception:
        return "--"
    if not math.isfinite(f):
        return "--"
    return f"{f:.{digits}f}"


def _delta_table(metrics: pd.DataFrame) -> pd.DataFrame:
    wide = metrics.pivot_table(
        index=["dataset", "model"],
        columns="source",
        values=[
            "rmse",
            "mae",
            "mean_seg_risk",
            "seg95",
            "peg_unsafe_fraction",
            "clinical_loss_mean",
            "bias",
            "abs_bias",
            "rmse_30min",
            "rmse_60min",
            "cross70_rate",
            "cross180_rate",
        ],
    )
    rows: list[dict[str, Any]] = []
    for dataset, model in wide.index:
        row = {"dataset": dataset, "model": model}
        for metric in wide.columns.get_level_values(0).unique():
            if (metric, "mse") not in wide or (metric, "threshold_crossing") not in wide:
                continue
            mse = float(wide.loc[(dataset, model), (metric, "mse")])
            new = float(wide.loc[(dataset, model), (metric, "threshold_crossing")])
            row[f"{metric}_mse"] = mse
            row[f"{metric}_tc"] = new
            row[f"delta_{metric}"] = new - mse
            row[f"pct_delta_{metric}"] = 100.0 * (new - mse) / mse if mse != 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "model"]).reset_index(drop=True)


def _slice_delta_table(slice_metrics: pd.DataFrame) -> pd.DataFrame:
    wide = slice_metrics.pivot_table(
        index=["dataset", "model", "glucose_slice"],
        columns="source",
        values=["rmse", "mean_seg_risk", "peg_unsafe_fraction", "clinical_loss_mean", "bias"],
    )
    rows: list[dict[str, Any]] = []
    for dataset, model, glucose_slice in wide.index:
        row = {"dataset": dataset, "model": model, "glucose_slice": glucose_slice}
        for metric in wide.columns.get_level_values(0).unique():
            mse = float(wide.loc[(dataset, model, glucose_slice), (metric, "mse")])
            new = float(wide.loc[(dataset, model, glucose_slice), (metric, "threshold_crossing")])
            row[f"{metric}_mse"] = mse
            row[f"{metric}_tc"] = new
            row[f"delta_{metric}"] = new - mse
        rows.append(row)
    return pd.DataFrame(rows)


def _hypo_delta_table(ops: pd.DataFrame) -> pd.DataFrame:
    op = ops[(ops["slice"] == "overall") & (ops["target_fa_day"] == 1.0)].copy()
    wide = op.pivot_table(
        index=["dataset", "model"],
        columns="source",
        values=["test_recall", "test_fa_day", "test_median_warning_lead_time_minutes", "theta_alarm"],
    )
    rows: list[dict[str, Any]] = []
    for dataset, model in wide.index:
        row = {"dataset": dataset, "model": model}
        for metric in ["test_recall", "test_fa_day", "test_median_warning_lead_time_minutes", "theta_alarm"]:
            mse = float(wide.loc[(dataset, model), (metric, "mse")])
            new = float(wide.loc[(dataset, model), (metric, "threshold_crossing")])
            row[f"{metric}_mse"] = mse
            row[f"{metric}_tc"] = new
            row[f"delta_{metric}"] = new - mse
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "model"]).reset_index(drop=True)


def _window_seg_correlations(
    *,
    mse_root: Path,
    new_loss_root: Path,
    loss_config: Path,
) -> pd.DataFrame:
    clinical = ClinicalMetrics.from_default()
    loss_cfg = _load_threshold_loss_config(loss_config)
    rows: list[dict[str, Any]] = []
    for source, base in (("mse", mse_root), ("threshold_crossing", new_loss_root)):
        for model in MODELS:
            for dataset in DATASETS:
                run_dir = base / f"{model}_{dataset}"
                if not (run_dir / "prediction_pairs_test.npz").exists():
                    continue
                data = _load_npz(run_dir, "test")
                y = np.asarray(data["y_gt"], dtype=float)
                p = np.asarray(data["y_hat"], dtype=float)
                mse = np.nanmean((p - y) ** 2, axis=1)
                rmse = np.sqrt(mse)
                seg = np.asarray(clinical.SEG(y.reshape(-1), p.reshape(-1), reduction="none"), dtype=float).reshape(y.shape)
                tc = _clinical_loss(y.reshape(-1), p.reshape(-1), loss_cfg).reshape(y.shape)
                scores = {"MSE": mse, "RMSE": rmse, "threshold_crossing_loss": np.nanmean(tc, axis=1)}
                mean_seg = np.nanmean(seg, axis=1)
                for slice_name in ("overall", "post_bolus", "nocturnal"):
                    mask = _slice_mask(data, slice_name)
                    for score_name, score in scores.items():
                        stats = _corr(score[mask], mean_seg[mask])
                        rows.append(
                            {
                                "section": "window_seg_correlation",
                                "source": source,
                                "dataset": dataset,
                                "model": model,
                                "slice": slice_name,
                                "score": score_name,
                                "target": "mean_SEG_risk",
                                **stats,
                            }
                        )
    return pd.DataFrame(rows)


def _aggregate_hypo_correlations(metrics: pd.DataFrame, ops: pd.DataFrame, per_patient: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    op = ops[ops["target_fa_day"] == 1.0].copy()
    joined = op.merge(
        metrics[
            [
                "source",
                "model",
                "dataset",
                "rmse",
                "mean_seg_risk",
                "clinical_loss_mean",
                "peg_unsafe_fraction",
                "abs_bias",
            ]
        ],
        on=["source", "model", "dataset"],
        how="left",
    )
    rows: list[dict[str, Any]] = []
    for slice_name, g_slice in joined.groupby("slice"):
        for source, group in g_slice.groupby("source"):
            for target in ["test_recall", "test_fa_day"]:
                for score in ["rmse", "mean_seg_risk", "clinical_loss_mean", "peg_unsafe_fraction", "abs_bias"]:
                    stats = _corr(group[score].to_numpy(), group[target].to_numpy())
                    rows.append(
                        {
                            "section": "run_hypogating_correlation",
                            "source": source,
                            "slice": slice_name,
                            "score": score,
                            "target": f"{target}_at_1fa",
                            **stats,
                        }
                    )

    piv = per_patient.pivot_table(index=["source", "dataset", "model", "patient_id"], columns="metric", values="value", aggfunc="mean").reset_index()
    prows: list[dict[str, Any]] = []
    for source, group in piv.groupby("source"):
        for target in ["hypo_recall_at_1fa", "hypo_fa_day_at_1fa"]:
            if target not in group:
                continue
            for score in ["rmse", "mean_seg_risk", "clinical_loss_mean", "peg_unsafe_fraction", "abs_bias"]:
                if score not in group:
                    continue
                stats = _corr(group[score].to_numpy(), group[target].to_numpy())
                prows.append(
                    {
                        "section": "patient_hypogating_correlation",
                        "source": source,
                        "score": score,
                        "target": target,
                        **stats,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(prows)


def _write_all_results(
    path: Path,
    *,
    delta: pd.DataFrame,
    slice_delta: pd.DataFrame,
    hypo_delta: pd.DataFrame,
    primary: pd.DataFrame,
    stats: pd.DataFrame,
    window_corr: pd.DataFrame,
    run_hypo_corr: pd.DataFrame,
    patient_hypo_corr: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    frames.append(delta.assign(section="performance_delta_dataset_model"))
    frames.append(slice_delta.assign(section="performance_delta_glucose_slice"))
    frames.append(hypo_delta.assign(section="hypogating_delta_overall_1fa"))
    frames.append(primary.assign(section="primary_patient_paired_endpoints"))
    frames.append(stats.assign(section="all_patient_paired_tests"))
    frames.append(window_corr)
    frames.append(run_hypo_corr)
    frames.append(patient_hypo_corr)
    all_results = pd.concat(frames, ignore_index=True, sort=False)
    cols = ["section"] + [c for c in all_results.columns if c != "section"]
    all_results = all_results[cols]
    all_results.to_csv(path, index=False)
    return all_results


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _plot_performance_delta(delta: pd.DataFrame, hypo_delta: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    specs = [
        ("rmse", "RMSE delta (mg/dL)", "lower is better"),
        ("mean_seg_risk", "Mean SEG delta", "lower is better"),
        ("peg_unsafe_fraction", "PEG unsafe delta", "lower is better"),
        ("test_recall", "Hypo recall @1 FA/day delta", "higher is better"),
    ]
    hypo = None
    if "delta_test_recall" not in delta:
        hypo = pd.DataFrame()
    for ax, (metric, title, subtitle) in zip(axes.ravel(), specs):
        if metric == "test_recall":
            values = hypo_delta["delta_test_recall"].to_numpy()
            labels = [f"{DATASET_LABEL[d]}\n{MODEL_LABEL[m]}" for d, m in zip(hypo_delta["dataset"], hypo_delta["model"])]
        else:
            values = delta[f"delta_{metric}"].to_numpy()
            labels = [f"{DATASET_LABEL[d]}\n{MODEL_LABEL[m]}" for d, m in zip(delta["dataset"], delta["model"])]
        color = np.where(values < 0, "#2a9d8f", "#c44536")
        if metric == "test_recall":
            color = np.where(values > 0, "#2a9d8f", "#c44536")
        x = np.arange(len(values))
        ax.bar(x, values, color=color, edgecolor="black", linewidth=0.35)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel("TC-trained minus MSE-trained")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=75, ha="right")
        ax.text(0.01, 0.96, subtitle, transform=ax.transAxes, va="top", color="0.35")
    fig.savefig(out / "fig_performance_deltas.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_seg_correlations(window_corr: pd.DataFrame, out: Path) -> None:
    summary = (
        window_corr.groupby(["source", "slice", "score"], as_index=False)
        .agg(pearson=("pearson", "mean"), spearman=("spearman", "mean"))
        .sort_values(["source", "slice", "score"])
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True, constrained_layout=True)
    score_order = ["MSE", "RMSE", "threshold_crossing_loss"]
    colors = {"MSE": "#6c757d", "RMSE": "#457b9d", "threshold_crossing_loss": "#e76f51"}
    for ax, slice_name in zip(axes, ["overall", "post_bolus", "nocturnal"]):
        xbase = np.arange(2)
        width = 0.22
        for i, score in enumerate(score_order):
            vals = []
            for source in ["mse", "threshold_crossing"]:
                hit = summary[(summary["source"] == source) & (summary["slice"] == slice_name) & (summary["score"] == score)]
                vals.append(float(hit["spearman"].iloc[0]) if not hit.empty else np.nan)
            ax.bar(xbase + (i - 1) * width, vals, width=width, color=colors[score], label=score.replace("_", " "), edgecolor="black", linewidth=0.35)
        ax.set_title(slice_name.replace("_", " ").title())
        ax.set_xticks(xbase)
        ax.set_xticklabels(["MSE preds", "TC preds"])
        ax.set_ylim(0.70, 0.95)
        ax.set_ylabel("Spearman with mean SEG")
    axes[0].legend(loc="lower left", frameon=False)
    fig.savefig(out / "fig_seg_correlation.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_hypo_correlations(patient_hypo_corr: pd.DataFrame, out: Path) -> None:
    sub = patient_hypo_corr[patient_hypo_corr["target"].isin(["hypo_recall_at_1fa", "hypo_fa_day_at_1fa"])].copy()
    score_order = ["rmse", "mean_seg_risk", "clinical_loss_mean", "peg_unsafe_fraction", "abs_bias"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3), sharey=True, constrained_layout=True)
    for ax, target in zip(axes, ["hypo_recall_at_1fa", "hypo_fa_day_at_1fa"]):
        g = sub[sub["target"] == target]
        x = np.arange(len(score_order))
        width = 0.35
        for j, source in enumerate(["mse", "threshold_crossing"]):
            vals = []
            for score in score_order:
                hit = g[(g["source"] == source) & (g["score"] == score)]
                vals.append(float(hit["spearman"].iloc[0]) if not hit.empty else np.nan)
            ax.bar(x + (j - 0.5) * width, vals, width=width, label=SOURCE_LABEL[source], edgecolor="black", linewidth=0.35)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(target.replace("_", " "))
        ax.set_xticks(x)
        ax.set_xticklabels(["RMSE", "SEG", "TC loss", "PEG unsafe", "|bias|"], rotation=30, ha="right")
        ax.set_ylabel("Patient-level Spearman")
    axes[0].legend(frameon=False)
    fig.savefig(out / "fig_hypogating_correlation.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_slice_deltas(slice_delta: pd.DataFrame, out: Path) -> None:
    summary = slice_delta.groupby("glucose_slice", as_index=False).agg(
        delta_rmse=("delta_rmse", "mean"),
        delta_seg=("delta_mean_seg_risk", "mean"),
        delta_peg=("delta_peg_unsafe_fraction", "mean"),
        delta_clin=("delta_clinical_loss_mean", "mean"),
    )
    order = ["lt70", "70to180", "ge180", "ge250"]
    summary["glucose_slice"] = pd.Categorical(summary["glucose_slice"], order, ordered=True)
    summary = summary.sort_values("glucose_slice")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    labels = ["y < 70", "70 <= y < 180", "y >= 180", "y >= 250"]
    for ax, col, title in zip(axes, ["delta_rmse", "delta_seg", "delta_peg"], ["RMSE", "Mean SEG", "PEG unsafe"]):
        vals = summary[col].to_numpy()
        ax.bar(np.arange(len(vals)), vals, color=np.where(vals < 0, "#2a9d8f", "#c44536"), edgecolor="black", linewidth=0.35)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(f"Mean delta by glucose slice: {title}")
        ax.set_ylabel("TC-trained minus MSE-trained")
    fig.savefig(out / "fig_glucose_slice_deltas.pdf", bbox_inches="tight")
    plt.close(fig)


def _latex_table_performance(pooled: pd.DataFrame) -> str:
    wide = pooled.set_index("source")
    rows = []
    for metric, label in [
        ("rmse", "RMSE"),
        ("mae", "MAE"),
        ("mean_seg_risk", "Mean SEG"),
        ("seg95", "SEG95"),
        ("peg_unsafe_fraction", "PEG unsafe"),
        ("clinical_loss_mean", "TC clinical loss"),
        ("bias", "Bias"),
        ("abs_bias", "|Bias|"),
    ]:
        mse = float(wide.loc["mse", metric])
        tc = float(wide.loc["threshold_crossing", metric])
        rows.append(f"{label} & {_tex_num(mse, 4)} & {_tex_num(tc, 4)} & {_tex_num(tc - mse, 4)} \\\\")
    return "\n".join(rows)


def _latex_table_correlations(window_corr: pd.DataFrame) -> str:
    summary = window_corr[window_corr["slice"] == "overall"].groupby(["source", "score"], as_index=False).agg(
        pearson=("pearson", "mean"), spearman=("spearman", "mean")
    )
    rows = []
    for source in ["mse", "threshold_crossing"]:
        for score in ["MSE", "RMSE", "threshold_crossing_loss"]:
            hit = summary[(summary["source"] == source) & (summary["score"] == score)]
            if hit.empty:
                continue
            label = score.replace("threshold_crossing_loss", "TC loss")
            rows.append(f"{SOURCE_LABEL[source]} & {label} & {_tex_num(hit['pearson'].iloc[0], 3)} & {_tex_num(hit['spearman'].iloc[0], 3)} \\\\")
    return "\n".join(rows)


def _write_tex(
    path: Path,
    *,
    pooled: pd.DataFrame,
    delta: pd.DataFrame,
    slice_delta: pd.DataFrame,
    hypo_delta: pd.DataFrame,
    primary: pd.DataFrame,
    window_corr: pd.DataFrame,
    patient_hypo_corr: pd.DataFrame,
) -> None:
    improved = {
        "rmse": int((delta["delta_rmse"] < 0).sum()),
        "seg": int((delta["delta_mean_seg_risk"] < 0).sum()),
        "peg": int((delta["delta_peg_unsafe_fraction"] < 0).sum()),
        "clin": int((delta["delta_clinical_loss_mean"] < 0).sum()),
        "hypo": int((hypo_delta["delta_test_recall"] > 0).sum()),
    }
    mean_delta = {
        "rmse": float(delta["delta_rmse"].mean()),
        "seg": float(delta["delta_mean_seg_risk"].mean()),
        "peg": float(delta["delta_peg_unsafe_fraction"].mean()),
        "clin": float(delta["delta_clinical_loss_mean"].mean()),
        "bias": float(delta["delta_bias"].mean()),
        "hypo": float(hypo_delta["delta_test_recall"].mean()),
        "fa": float(hypo_delta["delta_test_fa_day"].mean()),
    }
    hypo_slice = slice_delta[slice_delta["glucose_slice"] == "lt70"]
    hyper_slice = slice_delta[slice_delta["glucose_slice"] == "ge250"]
    tex = rf"""\section{{Threshold-Crossing Kovatchev Loss Report}}

\subsection{{What Was Evaluated}}
We evaluated a model-independent threshold-crossing Kovatchev clinical loss as both a training objective and a post-hoc scoring function. The loss combines signed Kovatchev transformed squared error with bounded smooth penalties for crossing 70 mg/dL and 180 mg/dL. The fitted clinical-loss parameters were derived from SEG only, not from model forecasts. We then retrained GRU, DLinear, PatchMLP, and iTransformer on DCLP3, DCLP5, PEDAP, and AZT1D with the clinical loss, and compared those models against the matched MSE-trained baselines.

The main question has two parts. First, does direct optimization of this loss improve forecasting and clinical endpoints? Second, independent of training, does the loss behave like a better test-set surrogate for SEG and hypoglycemia warning metrics than MSE/RMSE?

\subsection{{Optimization Result}}
Direct training with the pure threshold-crossing Kovatchev loss did not improve the forecasting models. Across the 16 dataset/model combinations, RMSE improved in {improved['rmse']}/16 runs, mean SEG improved in {improved['seg']}/16 runs, PEG unsafe improved in {improved['peg']}/16 runs, and the clinical loss itself improved in {improved['clin']}/16 runs. Hypoglycemia recall at 1 false alarm per day improved in {improved['hypo']}/16 runs.

\begin{{table}}[t]
\centering
\small
\caption{{Macro-average test-set performance over the 16 dataset/model runs. Lower is better except for signed bias. Delta is threshold-crossing trained minus MSE-trained.}}
\label{{tab:tc-kovatchev-macro-performance}}
\begin{{tabular}}{{lccc}}
\toprule
Metric & MSE-trained & TC-trained & Delta \\
\midrule
{_latex_table_performance(pooled)}
\bottomrule
\end{{tabular}}
\end{{table}}

The optimization failure is most visible as a calibration problem. The clinical-loss models shifted predictions downward: the macro-average bias changed by {_tex_num(mean_delta['bias'], 2)} mg/dL relative to the MSE baseline. This helped some true-hypoglycemia points but hurt general and high-glucose forecasting. In the true-hypoglycemia slice, the mean RMSE delta was {_tex_num(hypo_slice['delta_rmse'].mean(), 2)} mg/dL and the PEG unsafe delta was {_tex_num(hypo_slice['delta_peg_unsafe_fraction'].mean(), 4)}. In the severe high-glucose slice ($y \geq 250$), the mean RMSE delta was {_tex_num(hyper_slice['delta_rmse'].mean(), 2)} mg/dL and the mean SEG delta was {_tex_num(hyper_slice['delta_mean_seg_risk'].mean(), 4)}, showing substantial degradation.

\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figs/fig_performance_deltas.pdf}}
\caption{{Per-run deltas for RMSE, mean SEG, PEG unsafe, and hypoglycemia recall at 1 false alarm per day. Bars below zero are improvements for the first three metrics; bars above zero are improvements for recall.}}
\label{{fig:tc-kovatchev-performance-deltas}}
\end{{figure}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figs/fig_glucose_slice_deltas.pdf}}
\caption{{Mean performance deltas stratified by true-glucose slice. The pure clinical loss helped some low-glucose unsafe predictions but degraded high-glucose performance, especially for $y \geq 250$ mg/dL.}}
\label{{fig:tc-kovatchev-slice-deltas}}
\end{{figure}}

\subsection{{Metric Alignment Result}}
As a scoring function, the threshold-crossing Kovatchev loss is much better aligned with SEG than MSE or RMSE. On held-out test windows, the new loss had Spearman correlation around 0.92 with mean SEG, compared with about 0.80 for MSE/RMSE.

\begin{{table}}[t]
\centering
\small
\caption{{Overall test-window correlation with mean SEG risk. Scores are computed on the same test predictions.}}
\label{{tab:tc-kovatchev-seg-correlation}}
\begin{{tabular}}{{llcc}}
\toprule
Prediction source & Score & Pearson & Spearman \\
\midrule
{_latex_table_correlations(window_corr)}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figs/fig_seg_correlation.pdf}}
\caption{{Window-level Spearman correlation with SEG by slice. The threshold-crossing Kovatchev loss tracks SEG more closely than MSE/RMSE across overall, post-bolus, and nocturnal windows.}}
\label{{fig:tc-kovatchev-seg-correlation}}
\end{{figure}}

The correlation with hypoglycemia warning metrics is weaker than SEG alignment but still generally better than RMSE at patient level. For MSE-trained predictions, patient-level Spearman correlation with recall at 1 false alarm/day was -0.520 for RMSE, -0.573 for mean SEG, and -0.584 for the threshold-crossing loss. For threshold-crossing-trained predictions, the corresponding correlations were -0.366, -0.499, and -0.497. Thus the new loss is useful as a clinical scoring function, but direct optimization of the pure loss does not preserve calibration.

\begin{{figure}}[t]
\centering
\includegraphics[width=\textwidth]{{figs/fig_hypogating_correlation.pdf}}
\caption{{Patient-level Spearman correlation between test-set scores and hypoglycemia operating-point metrics. The threshold-crossing loss aligns with recall and false alarms better than RMSE in most comparisons, but the relationship is not strong enough to justify using the pure loss alone for training.}}
\label{{fig:tc-kovatchev-hypogating-correlation}}
\end{{figure}}

\subsection{{Interpretation}}
The result separates metric design from optimization. The threshold-crossing Kovatchev loss is a strong post-hoc surrogate for SEG and a reasonable clinical-risk score, but it is not a good standalone training objective in its current form. The pure clinical loss over-rewards downward prediction shifts because crossing 70 mg/dL is heavily penalized and there is no explicit calibration anchor. A defensible next step is to use this loss as an auxiliary term with an MSE or likelihood calibration anchor, or to constrain optimization with a bias/calibration penalty.
"""
    path.write_text(tex)


def run(args: argparse.Namespace) -> Path:
    comparison = Path(args.comparison_dir)
    out = Path(args.output_dir)
    figs = out / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()

    metrics = pd.read_csv(comparison / "mse_vs_threshold_crossing_metrics.csv")
    slice_metrics = pd.read_csv(comparison / "glucose_slice_metrics.csv")
    ops = pd.read_csv(comparison / "hypogating_operating_points.csv")
    primary = pd.read_csv(comparison / "primary_endpoint_summary.csv")
    stats = pd.read_csv(comparison / "statistical_tests.csv")
    pooled = pd.read_csv(comparison / "pooled_summary.csv")
    per_patient = pd.read_csv(comparison / "per_patient_metrics.csv")

    delta = _delta_table(metrics)
    slice_delta = _slice_delta_table(slice_metrics)
    hypo_delta = _hypo_delta_table(ops)
    window_corr = _window_seg_correlations(mse_root=Path(args.mse_root), new_loss_root=Path(args.new_loss_root), loss_config=Path(args.loss_config))
    run_hypo_corr, patient_hypo_corr = _aggregate_hypo_correlations(metrics, ops, per_patient)

    all_results = _write_all_results(
        out / "threshold_crossing_kovatchev_report_results.csv",
        delta=delta,
        slice_delta=slice_delta,
        hypo_delta=hypo_delta,
        primary=primary,
        stats=stats,
        window_corr=window_corr,
        run_hypo_corr=run_hypo_corr,
        patient_hypo_corr=patient_hypo_corr,
    )

    _plot_performance_delta(delta, hypo_delta, figs)
    _plot_seg_correlations(window_corr, figs)
    _plot_hypo_correlations(patient_hypo_corr, figs)
    _plot_slice_deltas(slice_delta, figs)
    _write_tex(
        out / "threshold_crossing_kovatchev_report.tex",
        pooled=pooled,
        delta=delta,
        slice_delta=slice_delta,
        hypo_delta=hypo_delta,
        primary=primary,
        window_corr=window_corr,
        patient_hypo_corr=patient_hypo_corr,
    )
    print(out)
    print(f"wrote {len(all_results)} result rows")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate threshold-crossing Kovatchev report CSV, LaTeX, and PDF figures.")
    p.add_argument("--comparison-dir", type=Path, default=Path("outputs/surrogate/loss_retraining/threshold_crossing_kovatchev/comparison"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/surrogate/report"))
    p.add_argument("--mse-root", type=Path, default=Path("outputs/surrogate/mse_pair_generation"))
    p.add_argument("--new-loss-root", type=Path, default=Path("outputs/surrogate/loss_retraining/threshold_crossing_kovatchev"))
    p.add_argument("--loss-config", type=Path, default=Path("configs/surrogate/fitted_threshold_crossing_kovatchev_default.yaml"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
