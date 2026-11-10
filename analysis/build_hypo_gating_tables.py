#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path("outputs/results")
OUT_FILE = Path("hypo_gating_tables.tex")
HYPO_RUN_SUFFIX = "p12"
HYPO_HORIZON_LABEL = "1-hour"
DATASETS = ["dclp3", "dclp5", "pedap"]
DATASET_LABELS = {
    "dclp3": "DCLP3",
    "dclp5": "DCLP5",
    "pedap": "PEDAP",
}
ARCH_ORDER = [
    "arimax",
    "dlinear",
    "gru",
    "itransformer",
    "patchmlp",
    "patchmlp_glimmer",
    "attengluco",
    "zoh",
]
ARCH_LABELS = {
    "arimax": "ARIMAX",
    "attengluco": "ATTENGLUCO",
    "dlinear": "DLINEAR",
    "gru": "GRU",
    "itransformer": "ITRANSFORMER",
    "patchmlp": "PATCHMLP",
    "patchmlp_glimmer": "PATCHMLP+GLIMMER",
    "zoh": "ZOH",
}
METRICS = [
    ("event_recall", "Recall", "max"),
    ("false_alarms_per_patient_day", "FA/day", "min"),
    ("rmse", "RMSE", "min"),
]
TABLE_SPECS = [
    (
        "overall",
        f"Hypo gating overall metrics for the {HYPO_HORIZON_LABEL} horizon (excluding Shanghai). "
        "Values are event recall, false alarms per patient-day, and RMSE. "
        "Best values in each dataset/metric are bolded.",
        "tab:hypo-gating-overall",
    ),
    (
        "post_bolus",
        f"Hypo gating post-bolus metrics for the {HYPO_HORIZON_LABEL} horizon (excluding Shanghai). "
        "Values are event recall, false alarms per patient-day, and RMSE. "
        "Best values in each dataset/metric are bolded.",
        "tab:hypo-gating-post-bolus",
    ),
]
CF_RUN_PREFIX = "action_cf_best_"
CF_HORIZON_MINUTES = 90
CF_FAMILIES = [("basal", "Basal"), ("bolus", "Bolus")]
CF_METRICS = [
    ("rmse_effect", "RMSE Effect", "min"),
    ("ordinal_pairwise_accuracy", "Ord. Pairwise Acc.", "max"),
]
CF_ORDINAL_GROUPS = {
    "basal": "basal_intensity",
    "bolus": "bolus_intensity",
}
POLICY_REGRET_ROOT = ROOT / "policy_regret"
POLICY_REGRET_PREFIX = "action_cf_benchmark_bolus_"
POLICY_REGRET_METRICS = [
    ("rmse", "RMSE", "min"),
    ("mean_regret", "Policy Regret", "min"),
]


def _infer_architecture_and_dataset(run_dir: Path) -> tuple[str, str] | None:
    name = run_dir.name.lower()
    if not name.startswith("best_") or not name.endswith(f"_{HYPO_RUN_SUFFIX}"):
        return None
    for dataset in DATASETS:
        suffix = f"_{dataset}_{HYPO_RUN_SUFFIX}"
        if name.endswith(suffix):
            architecture = name[len("best_") : -len(suffix)]
            return architecture, dataset
    return None


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _select_row(rows: list[dict[str, str]], slice_name: str) -> dict[str, str] | None:
    for row in rows:
        if row.get("slice", "").strip().lower() == slice_name:
            return row
    return None


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _infer_cf_architecture(run_dir: Path) -> str | None:
    name = run_dir.name.lower()
    if not name.startswith(CF_RUN_PREFIX):
        return None
    architecture = name[len(CF_RUN_PREFIX) :]
    return architecture or None


def _infer_policy_regret_architecture(run_dir: Path) -> str | None:
    name = run_dir.name.lower()
    if not name.startswith(POLICY_REGRET_PREFIX):
        return None
    architecture = name[len(POLICY_REGRET_PREFIX) :]
    return architecture or None


def _sort_architectures(architectures: set[str]) -> list[str]:
    order = {arch: idx for idx, arch in enumerate(ARCH_ORDER)}
    return sorted(architectures, key=lambda arch: (order.get(arch, len(order)), arch))


def _best_value(values: list[float], direction: str) -> float | None:
    if not values:
        return None
    if direction == "max":
        return max(values)
    return min(values)


def _format_value(value: float | None, best: float | None) -> str:
    if value is None:
        return "NA"
    formatted = f"{value:.3f}"
    if best is not None and math.isclose(value, best, rel_tol=1e-9, abs_tol=1e-12):
        return rf"\textbf{{{formatted}}}"
    return formatted


def _build_table(
    title: str,
    label: str,
    architectures: list[str],
    data: dict[str, dict[str, dict[str, float | None]]],
) -> list[str]:
    bests: dict[tuple[str, str], float | None] = {}
    for dataset in DATASETS:
        for metric_key, _, direction in METRICS:
            values = []
            for architecture in architectures:
                value = data.get(architecture, {}).get(dataset, {}).get(metric_key)
                if value is not None:
                    values.append(value)
            bests[(dataset, metric_key)] = _best_value(values, direction)

    colspec = "l" + "c" * (len(DATASETS) * len(METRICS))
    dataset_header = " & " + " & ".join(
        rf"\multicolumn{{{len(METRICS)}}}{{c}}{{{DATASET_LABELS[dataset]}}}" for dataset in DATASETS
    ) + r" \\"

    cmidrules = []
    for index, _dataset in enumerate(DATASETS):
        start = 2 + index * len(METRICS)
        end = start + len(METRICS) - 1
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")

    metric_header = "Model & " + " & ".join(metric_label for _dataset in DATASETS for _key, metric_label, _direction in METRICS)
    metric_header += r" \\"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{title}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        dataset_header,
        "".join(cmidrules),
        metric_header,
        r"\midrule",
    ]

    for architecture in architectures:
        cells = [ARCH_LABELS.get(architecture, architecture.upper())]
        for dataset in DATASETS:
            dataset_metrics = data.get(architecture, {}).get(dataset, {})
            for metric_key, _metric_label, _direction in METRICS:
                value = dataset_metrics.get(metric_key)
                best = bests[(dataset, metric_key)]
                cells.append(_format_value(value, best))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return lines


def _select_cf_row(
    rows: list[dict[str, str]],
    family: str,
    metric_name: str,
    horizon_minutes: int,
    *,
    perturbation_type: str | None = None,
    ordinal_group: str | None = None,
) -> dict[str, str] | None:
    for row in rows:
        if row.get("perturbation_family", "").strip().lower() != family:
            continue
        if row.get("metric_name", "").strip() != metric_name:
            continue
        try:
            if int(row.get("horizon_minutes", "").strip()) != horizon_minutes:
                continue
        except ValueError:
            continue
        if perturbation_type is not None and row.get("perturbation_type", "").strip() != perturbation_type:
            continue
        if ordinal_group is not None and row.get("ordinal_group", "").strip() != ordinal_group:
            continue
        return row
    return None


def _build_counterfactual_table(
    architectures: list[str],
    data: dict[str, dict[str, dict[str, float | None]]],
) -> list[str]:
    bests: dict[tuple[str, str], float | None] = {}
    for family, _family_label in CF_FAMILIES:
        for metric_name, _metric_label, direction in CF_METRICS:
            values = []
            for architecture in architectures:
                value = data.get(architecture, {}).get(family, {}).get(metric_name)
                if value is not None:
                    values.append(value)
            bests[(family, metric_name)] = _best_value(values, direction)

    colspec = "l" + "c" * (len(CF_FAMILIES) * len(CF_METRICS))
    family_header = " & " + " & ".join(
        rf"\multicolumn{{{len(CF_METRICS)}}}{{c}}{{{family_label}}}" for _family, family_label in CF_FAMILIES
    ) + r" \\"

    cmidrules = []
    for index, (_family, _family_label) in enumerate(CF_FAMILIES):
        start = 2 + index * len(CF_METRICS)
        end = start + len(CF_METRICS) - 1
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")

    metric_header = "Model & " + " & ".join(
        metric_label for _family, _family_label in CF_FAMILIES for _metric_name, metric_label, _direction in CF_METRICS
    )
    metric_header += r" \\"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Counterfactual action-benchmark metrics at 90 minutes. Values are RMSE effect and ordinal pairwise accuracy for basal and bolus interventions. Best values in each column are bolded.}",
        r"\label{tab:counterfactual-90min}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        family_header,
        "".join(cmidrules),
        metric_header,
        r"\midrule",
    ]

    for architecture in architectures:
        cells = [ARCH_LABELS.get(architecture, architecture.upper())]
        for family, _family_label in CF_FAMILIES:
            family_metrics = data.get(architecture, {}).get(family, {})
            for metric_name, _metric_label, _direction in CF_METRICS:
                value = family_metrics.get(metric_name)
                best = bests[(family, metric_name)]
                cells.append(_format_value(value, best))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return lines


def _build_policy_regret_table(
    architectures: list[str],
    data: dict[str, dict[str, float | None]],
) -> list[str]:
    bests: dict[str, float | None] = {}
    for metric_name, _metric_label, direction in POLICY_REGRET_METRICS:
        values = []
        for architecture in architectures:
            value = data.get(architecture, {}).get(metric_name)
            if value is not None:
                values.append(value)
        bests[metric_name] = _best_value(values, direction)

    colspec = "l" + "c" * len(POLICY_REGRET_METRICS)
    header = "Model & " + " & ".join(metric_label for _metric_name, metric_label, _direction in POLICY_REGRET_METRICS) + r" \\"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Policy-regret benchmark results on action\_cf\_benchmark\_bolus. RMSE is the run's overall forecast RMSE, and policy regret is macro mean patient regret. Best values in each column are bolded.}",
        r"\label{tab:policy-regret}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        header,
        r"\midrule",
    ]

    for architecture in architectures:
        cells = [ARCH_LABELS.get(architecture, architecture.upper())]
        for metric_name, _metric_label, _direction in POLICY_REGRET_METRICS:
            value = data.get(architecture, {}).get(metric_name)
            best = bests[metric_name]
            cells.append(_format_value(value, best))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return lines


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    if not ROOT.exists():
        raise SystemExit(f"Missing results root: {ROOT}")

    overall_data: dict[str, dict[str, dict[str, float | None]]] = {}
    post_bolus_data: dict[str, dict[str, dict[str, float | None]]] = {}
    architectures: set[str] = set()
    cf_data: dict[str, dict[str, dict[str, float | None]]] = {}
    cf_architectures: set[str] = set()
    policy_regret_data: dict[str, dict[str, float | None]] = {}
    policy_regret_architectures: set[str] = set()

    for run_dir in sorted(ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = _infer_architecture_and_dataset(run_dir)
        if parsed is not None:
            architecture, dataset = parsed
            architectures.add(architecture)

            metrics_path = run_dir / "hypo_gating_metrics.csv"
            by_slice_path = run_dir / "hypo_gating_metrics_by_slice.csv"
            if metrics_path.exists():
                overall_rows = _read_rows(metrics_path)
                by_slice_rows = _read_rows(by_slice_path) if by_slice_path.exists() else []

                overall_row = _select_row(overall_rows, "overall")
                if overall_row is None and len(overall_rows) == 1:
                    overall_row = overall_rows[0]
                post_bolus_row = _select_row(by_slice_rows, "post_bolus")

                for target, row in ((overall_data, overall_row), (post_bolus_data, post_bolus_row)):
                    target.setdefault(architecture, {}).setdefault(dataset, {})
                    for metric_key, _metric_label, _direction in METRICS:
                        target[architecture][dataset][metric_key] = _parse_float(None if row is None else row.get(metric_key))

        cf_architecture = _infer_cf_architecture(run_dir)
        if cf_architecture is not None:
            cf_metrics_path = run_dir / "cf_metrics.csv"
            if cf_metrics_path.exists():
                cf_architectures.add(cf_architecture)
                cf_rows = _read_rows(cf_metrics_path)
                for family, _family_label in CF_FAMILIES:
                    cf_data.setdefault(cf_architecture, {}).setdefault(family, {})
                    for metric_name, _metric_label, _direction in CF_METRICS:
                        ordinal_group = CF_ORDINAL_GROUPS[family] if metric_name == "ordinal_pairwise_accuracy" else None
                        perturbation_type = "__ALL__" if metric_name == "rmse_effect" else None
                        row = _select_cf_row(
                            cf_rows,
                            family,
                            metric_name,
                            CF_HORIZON_MINUTES,
                            perturbation_type=perturbation_type,
                            ordinal_group=ordinal_group,
                        )
                        cf_data[cf_architecture][family][metric_name] = _parse_float(
                            None if row is None else row.get("metric_value")
                        )

    if POLICY_REGRET_ROOT.exists():
        for run_dir in sorted(POLICY_REGRET_ROOT.iterdir()):
            if not run_dir.is_dir():
                continue
            architecture = _infer_policy_regret_architecture(run_dir)
            if architecture is None:
                continue
            metrics_path = run_dir / "metrics.json"
            summary_path = run_dir / "policy_regret_summary.json"
            if not metrics_path.exists() or not summary_path.exists():
                continue

            metrics_data = _load_json(metrics_path)
            summary_data = _load_json(summary_path)
            rmse = (
                metrics_data.get("metrics", {})
                .get("global", {})
                .get("rmse")
            )
            mean_regret = (
                summary_data.get("macro_avg_patient_metrics", {})
                .get("mean_regret")
            )

            policy_regret_architectures.add(architecture)
            policy_regret_data[architecture] = {
                "rmse": _parse_float(None if rmse is None else str(rmse)),
                "mean_regret": _parse_float(None if mean_regret is None else str(mean_regret)),
            }

    ordered_architectures = _sort_architectures(architectures)
    if not ordered_architectures:
        raise SystemExit(f"No matching best_*_<dataset>_{HYPO_RUN_SUFFIX} runs were found.")
    ordered_cf_architectures = _sort_architectures(cf_architectures)
    if not ordered_cf_architectures:
        raise SystemExit("No matching action_cf_best_* runs were found.")
    ordered_policy_regret_architectures = _sort_architectures(policy_regret_architectures)
    if not ordered_policy_regret_architectures:
        raise SystemExit("No matching policy_regret/action_cf_benchmark_bolus_* runs were found.")

    lines = [
        "% Auto-generated hypo gating and counterfactual tables from outputs/results",
        "",
    ]
    for slice_name, title, label in TABLE_SPECS:
        data = overall_data if slice_name == "overall" else post_bolus_data
        lines.extend(_build_table(title=title, label=label, architectures=ordered_architectures, data=data))
    lines.extend(_build_counterfactual_table(architectures=ordered_cf_architectures, data=cf_data))
    lines.extend(
        _build_policy_regret_table(
            architectures=ordered_policy_regret_architectures,
            data=policy_regret_data,
        )
    )

    OUT_FILE.write_text("\n".join(lines))
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
