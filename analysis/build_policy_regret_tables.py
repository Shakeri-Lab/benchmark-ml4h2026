#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import pandas as pd


MODEL_LABELS = {
    "arimax": "ARIMAX",
    "attengluco": "AttenGluco",
    "dlinear": "DLinear",
    "gru": "GRU",
    "itransformer": "iTransformer",
    "patchmlp": "PatchMLP",
    "patchmlp_glimmer": "Glimmer\\textsuperscript{*}",
}
DEFAULT_MODELS = ["arimax", "attengluco", "dlinear", "gru", "itransformer", "patchmlp", "patchmlp_glimmer"]


def _fmt_ci(mean: float, lo: float, hi: float, decimals: int = 2) -> str:
    if any(math.isnan(float(x)) for x in (mean, lo, hi)):
        return "--"
    return f"{mean:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


def _maybe_bold(text: str, is_best: bool) -> str:
    return f"\\textbf{{{text}}}" if is_best and text != "--" else text


def _label_suffix(dataset_label: str) -> str:
    return (
        dataset_label.replace("\\_", "-")
        .replace("_", "-")
        .replace("\\", "")
        .replace("{", "")
        .replace("}", "")
        .lower()
    )


def _metric_row(df: pd.DataFrame, metric: str) -> pd.Series | None:
    sub = df.loc[df["metric"] == metric]
    if sub.empty:
        return None
    return sub.iloc[0]


def _cf_row(df: pd.DataFrame, *, family: str, metric: str, horizon: int) -> pd.Series | None:
    if metric == "kendall_rank_corr":
        sub = df.loc[
            (df["perturbation_family"] == family)
            & (df["ordinal_group"] == f"{family}_intensity")
            & (df["horizon_minutes"] == horizon)
            & (df["metric_name"] == metric)
        ]
    else:
        sub = df.loc[
            (df["perturbation_family"] == family)
            & (df["perturbation_type"] == "__ALL__")
            & (df["horizon_minutes"] == horizon)
            & (df["metric_name"] == metric)
        ]
    if sub.empty:
        return None
    return sub.iloc[0]


def _read_policy_metrics(policy_root: Path, prefix: str, models: Iterable[str]) -> dict[str, dict[str, tuple[float, float, float]]]:
    out: dict[str, dict[str, tuple[float, float, float]]] = {}
    for model in models:
        path = policy_root / f"{prefix}{model}" / "policy_regret_bootstrap.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        model_metrics: dict[str, tuple[float, float, float]] = {}
        for metric in ("action_match_rate", "mean_regret"):
            row = _metric_row(df, metric)
            if row is None:
                continue
            model_metrics[metric] = (float(row["macro_mean"]), float(row["ci_lower"]), float(row["ci_upper"]))
        if model_metrics:
            out[model] = model_metrics
    return out


def _read_cf_metrics(
    cf_root: Path,
    prefix: str,
    models: Iterable[str],
    horizon: int,
) -> dict[str, dict[tuple[str, str], tuple[float, float, float]]]:
    out: dict[str, dict[tuple[str, str], tuple[float, float, float]]] = {}
    for model in models:
        path = cf_root / f"{prefix}{model}" / "cf_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        model_metrics: dict[tuple[str, str], tuple[float, float, float]] = {}
        for family in ("basal", "bolus"):
            for metric in ("rmse_effect", "sign_agreement", "kendall_rank_corr"):
                row = _cf_row(df, family=family, metric=metric, horizon=horizon)
                if row is None:
                    continue
                model_metrics[(family, metric)] = (
                    float(row["metric_value"]),
                    float(row["ci_low"]),
                    float(row["ci_high"]),
                )
        if model_metrics:
            out[model] = model_metrics
    return out


def _best_flags(values: dict[str, tuple[float, float, float]], *, higher_is_better: bool) -> set[str]:
    valid = {k: v[0] for k, v in values.items() if not math.isnan(v[0])}
    if not valid:
        return set()
    best = max(valid.values()) if higher_is_better else min(valid.values())
    return {k for k, v in valid.items() if abs(v - best) <= 1e-12}


def _build_policy_table(
    policy_metrics: dict[str, dict[str, tuple[float, float, float]]],
    models: list[str],
    *,
    dataset_label: str,
) -> str:
    label_suffix = _label_suffix(dataset_label)
    amr_vals = {m: policy_metrics[m]["action_match_rate"] for m in models if m in policy_metrics and "action_match_rate" in policy_metrics[m]}
    regret_vals = {m: policy_metrics[m]["mean_regret"] for m in models if m in policy_metrics and "mean_regret" in policy_metrics[m]}
    best_amr = _best_flags(amr_vals, higher_is_better=True)
    best_regret = _best_flags(regret_vals, higher_is_better=False)

    rows = []
    for model in models:
        if model not in policy_metrics:
            continue
        amr = _fmt_ci(*policy_metrics[model].get("action_match_rate", (math.nan, math.nan, math.nan)))
        regret = _fmt_ci(*policy_metrics[model].get("mean_regret", (math.nan, math.nan, math.nan)))
        rows.append(
            f"{MODEL_LABELS.get(model, model)} & "
            f"{_maybe_bold(amr, model in best_amr)} & "
            f"{_maybe_bold(regret, model in best_regret)} \\\\"
        )

    body = "\n".join(rows) if rows else "\\multicolumn{3}{c}{No completed policy-regret runs found.} \\\\"
    return rf"""\begin{{table}}[t]
\centering
\small
\caption{{\textbf{{Policy-regret benchmark on {dataset_label} with 95\% confidence intervals.}} AMR is action match rate and Regret is mean regret. Each cell is reported as mean [95\% CI] from patient-level bootstrap resampling. Higher is better for AMR; lower is better for Regret. Best values in each column are bolded.}}
\label{{tab:policy-regret-amr-regret-ci-{label_suffix}}}
\begin{{tabular}}{{lcc}}
\toprule
\textbf{{Model}} & \textbf{{AMR}} & \textbf{{Regret}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def _build_cf_table(
    cf_metrics: dict[str, dict[tuple[str, str], tuple[float, float, float]]],
    models: list[str],
    *,
    horizon: int,
    dataset_label: str,
) -> str:
    label_suffix = _label_suffix(dataset_label)
    metric_order = [
        ("rmse_effect", "eRMSE", False),
        ("sign_agreement", "SA", True),
        ("kendall_rank_corr", "\\(\\mathbf{\\tau_b}\\)", True),
    ]
    family_blocks = []
    for family, family_label in (("basal", "Basal pert."), ("bolus", "Bolus pert.")):
        best_by_metric = {}
        for metric, _, higher in metric_order:
            vals = {
                m: cf_metrics[m][(family, metric)]
                for m in models
                if m in cf_metrics and (family, metric) in cf_metrics[m]
            }
            best_by_metric[metric] = _best_flags(vals, higher_is_better=higher)

        rows = []
        available = [m for m in models if m in cf_metrics and any((family, metric) in cf_metrics[m] for metric, _, _ in metric_order)]
        for idx, model in enumerate(available):
            cells = []
            for metric, _, _ in metric_order:
                val = cf_metrics[model].get((family, metric), (math.nan, math.nan, math.nan))
                cells.append(_maybe_bold(_fmt_ci(*val), model in best_by_metric[metric]))
            prefix = f"\\multirow{{{len(available)}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\small {family_label}}}}}\n& " if idx == 0 else "& "
            rows.append(f"{prefix}{MODEL_LABELS.get(model, model)} & {' & '.join(cells)} \\\\")
        if rows:
            family_blocks.append("\n".join(rows))
    body = "\n\\midrule\n".join(family_blocks) if family_blocks else "\\multicolumn{5}{c}{No completed counterfactual metric runs found.} \\\\"
    return rf"""\begin{{table}}[t]
\centering
\small
\setlength{{\tabcolsep}}{{4pt}}
\caption{{\textbf{{Counterfactual forecasting performance across the perturbation menu at {horizon} minutes with 95\% confidence intervals.}} eRMSE = effect RMSE, SA = sign agreement, and \(\tau_b\) = Kendall rank correlation over the action menu. Each cell is reported as mean [95\% CI] from patient-level bootstrap resampling. Lower is better for eRMSE; higher is better for SA and \(\tau_b\). Best values in each column are bolded.}}
\label{{tab:cf-menu-main-ci-{label_suffix}}}
\begin{{tabular}}{{llccc}}
\toprule
\textbf{{}} & \textbf{{Model}} & \textbf{{eRMSE}} & \textbf{{SA}} & \(\mathbf{{\tau_b}}\) \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LaTeX policy-regret and CF menu tables from saved run outputs.")
    parser.add_argument("--policy-root", type=Path, default=Path("outputs/results/policy_regret"))
    parser.add_argument("--policy-run-prefix", default="action_cf_benchmark_bolus_")
    parser.add_argument("--cf-root", type=Path, default=Path("outputs/results"))
    parser.add_argument("--cf-run-prefix", default="action_cf_best_")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated model run suffixes to include.")
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--dataset-label", default="action\\_cf\\_benchmark\\_bolus")
    parser.add_argument("--output-tex", type=Path, default=Path("policy_regret_amr_regret_ci.tex"))
    parser.add_argument("--fail-if-exists", action="store_true")
    args = parser.parse_args()

    if args.fail_if_exists and args.output_tex.exists():
        raise FileExistsError(f"Refusing to overwrite existing table: {args.output_tex}")

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    policy_metrics = _read_policy_metrics(args.policy_root, args.policy_run_prefix, models)
    cf_metrics = _read_cf_metrics(args.cf_root, args.cf_run_prefix, models, args.horizon)

    text = "\n\n".join(
        [
            _build_policy_table(policy_metrics, models, dataset_label=args.dataset_label),
            _build_cf_table(cf_metrics, models, horizon=args.horizon, dataset_label=args.dataset_label),
        ]
    )
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(text + "\n")
    print(f"[+] Wrote {args.output_tex}")


if __name__ == "__main__":
    main()
