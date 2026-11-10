#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path("outputs/results")
OUT_FILE = Path("hypo_results.csv")

DATASETS = [
    ("dclp3", "DCLP3"),
    ("dclp5", "DCLP5"),
    ("pedap", "PEDAP"),
]
HORIZONS = [
    (6, 30),
    (12, 60),
]
SLICES = ["overall", "nocturnal", "post_bolus"]
MODELS = [
    ("zoh", "ZOH"),
    ("arimax", "ARIMAX"),
    ("dlinear", "Dlinear"),
    ("gru", "GRU"),
    ("itransformer", "iTransformer"),
    ("patchmlp", "PatchMLP"),
    ("patchmlp_glimmer", "GLimmer"),
]

METRIC_SPECS = [
    ("rmse", "rmse"),
    ("peg_unsafe", "peg_unsafe"),
    ("hypo_recall", "event_recall"),
    ("hypo_false_alarms_per_patient_day", "false_alarms_per_patient_day"),
    ("hypo_median_lead_time_minutes", "median_warning_lead_time_minutes"),
]

OUT_COLUMNS = [
    "dataset",
    "horizon_minutes",
    "slice",
    "model",
]
for out_name, _src_name in METRIC_SPECS:
    OUT_COLUMNS.extend([out_name, f"{out_name}_ci_low", f"{out_name}_ci_high"])


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _clean_value(value: str | None) -> str:
    if value is None:
        return "NA"
    value = value.strip()
    if not value:
        return "NA"
    try:
        parsed = float(value)
    except ValueError:
        return value
    if math.isnan(parsed) or math.isinf(parsed):
        return "NA"
    return f"{parsed:.15g}"


def _select_overall_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    for row in rows:
        if row.get("slice", "").strip().lower() == "overall":
            return row
    if len(rows) == 1:
        return rows[0]
    if "slice" not in rows[0]:
        return rows[0]
    return None


def _select_slice_row(rows: list[dict[str, str]], slice_name: str) -> dict[str, str] | None:
    if slice_name == "overall":
        return _select_overall_row(rows)
    for row in rows:
        if row.get("slice", "").strip().lower() == slice_name:
            return row
    return None


def _build_output_row(
    dataset_label: str,
    horizon_minutes: int,
    slice_name: str,
    model_label: str,
    source_row: dict[str, str] | None,
) -> dict[str, str]:
    row = {
        "dataset": dataset_label,
        "horizon_minutes": str(horizon_minutes),
        "slice": slice_name,
        "model": model_label,
    }
    for out_name, src_name in METRIC_SPECS:
        row[out_name] = _clean_value(None if source_row is None else source_row.get(src_name))
        row[f"{out_name}_ci_low"] = _clean_value(None if source_row is None else source_row.get(f"{src_name}_ci_low"))
        row[f"{out_name}_ci_high"] = _clean_value(None if source_row is None else source_row.get(f"{src_name}_ci_high"))
    return row


def main() -> None:
    if not ROOT.exists():
        raise SystemExit(f"Missing results root: {ROOT}")

    out_rows: list[dict[str, str]] = []
    for dataset_key, dataset_label in DATASETS:
        for steps, horizon_minutes in HORIZONS:
            for slice_name in SLICES:
                for model_key, model_label in MODELS:
                    run_dir = ROOT / f"best_{model_key}_{dataset_key}_p{steps}"
                    overall_rows = _read_rows(run_dir / "hypo_gating_metrics.csv")
                    by_slice_rows = _read_rows(run_dir / "hypo_gating_metrics_by_slice.csv")

                    if slice_name == "overall":
                        source_row = _select_overall_row(overall_rows)
                        if source_row is None:
                            source_row = _select_slice_row(by_slice_rows, "overall")
                    else:
                        source_row = _select_slice_row(by_slice_rows, slice_name)

                    out_rows.append(
                        _build_output_row(
                            dataset_label=dataset_label,
                            horizon_minutes=horizon_minutes,
                            slice_name=slice_name,
                            model_label=model_label,
                            source_row=source_row,
                        )
                    )

    with OUT_FILE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUT_COLUMNS)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {OUT_FILE} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
