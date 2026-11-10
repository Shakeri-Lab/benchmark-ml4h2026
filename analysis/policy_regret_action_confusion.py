from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def _architecture_from_run_dir(run_dir: Path, prefix: str) -> str:
    name = run_dir.name
    if name.startswith(prefix):
        return name[len(prefix) :]
    return name


def _format_scale(x: float) -> str:
    return f"{float(x):g}x"


def _available_action_labels(run_dir: Path, df: pd.DataFrame) -> list[str]:
    scales: set[float] = set()
    actions_path = run_dir / "policy_regret_actions.npz"
    if actions_path.exists():
        with np.load(actions_path, allow_pickle=True) as data:
            if "action_scale" in data.files:
                scales.update(float(x) for x in np.asarray(data["action_scale"], dtype=float).reshape(-1))
    scales.update(float(x) for x in df["oracle_scale"].dropna().to_numpy())
    scales.update(float(x) for x in df["selected_scale"].dropna().to_numpy())
    # Non-dose interventions (bolus_advance_30, bolus_delay_30, ...) carry a NaN
    # action_scale. NaN never compares equal, so a set keeps one entry per NaN and
    # they all format to the same label -- which makes the category list non-unique.
    finite = sorted({round(x, 6) for x in scales if np.isfinite(x)})
    labels, seen = [], set()
    for x in finite:
        lab = _format_scale(x)
        if lab not in seen:
            seen.add(lab)
            labels.append(lab)
    return labels


FAM = "bolus"   # action family being plotted; set from --family in main()


def _safe_plot_heatmap(counts: pd.DataFrame, architecture: str, out_png: Path, out_pdf: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    im = ax.imshow(counts.values, cmap="Blues")
    ax.set_xticks(range(len(counts.columns)))
    ax.set_xticklabels(counts.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(counts.index)))
    ax.set_yticklabels(counts.index)
    ax.set_xlabel(f"{architecture} selected {FAM} action")
    ax.set_ylabel(f"Oracle {FAM} action")
    ax.set_title(f"{architecture}: selected vs oracle {FAM} action")
    max_val = max(float(counts.values.max()), 1.0)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            v = int(counts.iloc[i, j])
            if v:
                color = "white" if v > 0.55 * max_val else "black"
                ax.text(j, i, str(v), ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, label="Episode count")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)


def _safe_plot_distribution(dist: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    arches = list(dict.fromkeys(dist["architecture"].tolist()))
    actions = list(dict.fromkeys(dist[f"{FAM}_action"].tolist()))
    x = np.arange(len(actions), dtype=float)
    width = 0.35
    fig, axes = plt.subplots(len(arches), 1, figsize=(9.5, max(2.2 * len(arches), 4)), sharex=True)
    if len(arches) == 1:
        axes = [axes]
    for ax, arch in zip(axes, arches):
        sub = dist[dist["architecture"] == arch].set_index(f"{FAM}_action").reindex(actions).fillna(0.0)
        ax.bar(x - width / 2, sub["oracle_fraction"].to_numpy(), width=width, label="Oracle")
        ax.bar(x + width / 2, sub["model_chosen_fraction"].to_numpy(), width=width, label="Model chosen")
        ax.set_ylabel(arch)
        ax.set_ylim(0, max(1.0, float(sub[["oracle_fraction", "model_chosen_fraction"]].to_numpy().max()) * 1.15))
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper right")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(actions, rotation=45, ha="right")
    axes[-1].set_xlabel(f"{FAM.capitalize()} action scale")
    fig.suptitle(f"Oracle and selected {FAM} action distributions by architecture", y=0.995)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build policy-regret bolus action confusion matrices by architecture.")
    parser.add_argument("--policy-regret-root", type=Path, default=Path("outputs/results/policy_regret"))
    parser.add_argument("--run-prefix", default="action_cf_benchmark_bolus_")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/results/policy_regret/action_cf_benchmark_bolus_action_confusion"),
    )
    parser.add_argument("--family", default="bolus",
                        help="Action family being summarised; sets axis labels, column names and filenames.")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    global FAM
    FAM = args.family

    args.output_dir.mkdir(parents=True, exist_ok=True)
    confusion_rows: list[pd.DataFrame] = []
    distribution_rows: list[dict] = []
    summary_rows: list[dict] = []

    run_dirs = sorted(args.policy_regret_root.glob(f"{args.run_prefix}*"))
    if not run_dirs:
        raise SystemExit(f"No run dirs found under {args.policy_regret_root} with prefix {args.run_prefix!r}")

    for run_dir in run_dirs:
        episode_path = run_dir / "policy_regret_episode_metrics.csv"
        if not episode_path.exists():
            continue
        architecture = _architecture_from_run_dir(run_dir, args.run_prefix)
        df = pd.read_csv(episode_path)
        required = {"oracle_scale", "selected_scale"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"{episode_path} missing columns: {sorted(missing)}")

        labels = _available_action_labels(run_dir, df)
        work = df.copy()
        work["oracle_action"] = pd.Categorical(
            work["oracle_scale"].map(_format_scale), categories=labels, ordered=True
        )
        work["selected_action"] = pd.Categorical(
            work["selected_scale"].map(_format_scale), categories=labels, ordered=True
        )

        counts = pd.crosstab(work["oracle_action"], work["selected_action"], dropna=False).reindex(
            index=labels, columns=labels, fill_value=0
        )
        counts.index.name = f"oracle_{FAM}_action"
        counts.columns.name = f"{architecture}_selected_{FAM}_action"
        row_norm = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

        # CSV columns are explicitly named so the matrix orientation is clear:
        # rows are oracle actions; columns are model-chosen actions.
        csv_columns = {label: f"model_chosen_{FAM}_action_{label}" for label in labels}
        counts_csv = counts.rename(columns=csv_columns)
        row_norm_csv = row_norm.rename(columns=csv_columns)
        counts_path = args.output_dir / f"{architecture}_{FAM}_oracle_confusion_matrix_counts.csv"
        row_norm_path = args.output_dir / f"{architecture}_{FAM}_oracle_confusion_matrix_row_normalized.csv"
        counts_csv.to_csv(counts_path)
        row_norm_csv.to_csv(row_norm_path)

        # Also keep a copy beside each original run for convenience/backward compatibility.
        counts_csv.to_csv(run_dir / f"{architecture}_{FAM}_oracle_confusion_matrix_counts.csv")
        row_norm_csv.to_csv(run_dir / f"{architecture}_{FAM}_oracle_confusion_matrix_row_normalized.csv")

        long = counts.stack().rename("count").reset_index()
        long = long.rename(columns={f"{architecture}_selected_{FAM}_action": f"model_chosen_{FAM}_action"})
        _oc, _mc = f"oracle_{FAM}_action", f"model_chosen_{FAM}_action"
        long["architecture"] = architecture
        long["row_fraction"] = [row_norm.loc[getattr(r, _oc), getattr(r, _mc)] for r in long.itertuples(index=False)]
        long["n_oracle_action"] = [counts.loc[getattr(r, _oc)].sum() for r in long.itertuples(index=False)]
        long["total_fraction"] = long["count"] / max(float(len(work)), 1.0)
        confusion_rows.append(long)

        oracle_counts = work["oracle_action"].value_counts(sort=False).reindex(labels, fill_value=0)
        selected_counts = work["selected_action"].value_counts(sort=False).reindex(labels, fill_value=0)
        for action in labels:
            distribution_rows.append(
                {
                    "architecture": architecture,
                    f"{FAM}_action": action,
                    "oracle_count": int(oracle_counts.loc[action]),
                    "oracle_fraction": float(oracle_counts.loc[action] / len(work)) if len(work) else 0.0,
                    "model_chosen_count": int(selected_counts.loc[action]),
                    "model_chosen_fraction": float(selected_counts.loc[action] / len(work)) if len(work) else 0.0,
                    "n_episodes": int(len(work)),
                }
            )

        exact = work["oracle_action"] == work["selected_action"]
        summary_rows.append(
            {
                "architecture": architecture,
                "run_dir": str(run_dir),
                "n_episodes": int(len(work)),
                "n_exact_matches": int(exact.sum()),
                "exact_match_rate": float(exact.mean()) if len(work) else 0.0,
                "mean_abs_scale_error": float((work["oracle_scale"] - work["selected_scale"]).abs().mean()),
                "median_abs_scale_error": float((work["oracle_scale"] - work["selected_scale"]).abs().median()),
            }
        )

        if not args.no_plots:
            _safe_plot_heatmap(
                counts,
                architecture,
                args.output_dir / f"{architecture}_{FAM}_oracle_confusion_matrix.png",
                args.output_dir / f"{architecture}_{FAM}_oracle_confusion_matrix.pdf",
            )

    confusion = pd.concat(confusion_rows, ignore_index=True) if confusion_rows else pd.DataFrame()
    distribution = pd.DataFrame(distribution_rows)
    summary = pd.DataFrame(summary_rows).sort_values("architecture").reset_index(drop=True)

    confusion.to_csv(args.output_dir / f"all_architectures_{FAM}_oracle_confusion_matrix_long.csv", index=False)
    distribution.to_csv(args.output_dir / f"all_architectures_{FAM}_action_distributions.csv", index=False)
    summary.to_csv(args.output_dir / f"all_architectures_{FAM}_oracle_confusion_summary.csv", index=False)

    if not args.no_plots and not distribution.empty:
        _safe_plot_distribution(
            distribution.sort_values(["architecture", f"{FAM}_action"]),
            args.output_dir / f"all_architectures_{FAM}_action_distributions.png",
            args.output_dir / f"all_architectures_{FAM}_action_distributions.pdf",
        )

    print(f"[+] Wrote outputs to {args.output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
