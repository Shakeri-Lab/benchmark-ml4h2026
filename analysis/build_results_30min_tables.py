#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml


ROOT = Path("outputs/results")
OUT_FILE = Path("results_30min.tex")
MODELS = ["arimax", "itransformer", "patchmlp", "attengluco", "gru", "zoh"]
DATASETS = ["dclp3", "dclp5", "pedap", "shanghait1dm"]
PRED_LEN = 6
HORIZON_IDX = PRED_LEN - 1  # 30 minutes at 5-min step


def _infer_dataset(cfg: dict[str, Any]) -> str | None:
    split = cfg.get("split", {})
    split_name = str(split.get("split_name", "")).lower()
    for ds in DATASETS:
        if ds in split_name:
            return ds
    notes = str(cfg.get("notes", "")).lower()
    for ds in DATASETS:
        if ds in notes:
            return ds
    return None


def _get_metric(metrics: dict[str, Any], key: str, idx: int) -> Any:
    if not isinstance(metrics, dict):
        return None
    per_h = metrics.get("per_horizon", {})
    if key in per_h:
        val = per_h[key]
        if isinstance(val, list):
            return val[min(idx, len(val) - 1)]
    return metrics.get("global", {}).get(key)


def _get_event_rmse(metrics: dict[str, Any], event_key: str, idx: int) -> Any:
    if not isinstance(metrics, dict):
        return None
    evt = metrics.get("event_conditioned", {}).get(event_key, {})
    vals = evt.get("metrics", {})
    if "rmse" in vals:
        v = vals["rmse"]
        if isinstance(v, list):
            return v[min(idx, len(v) - 1)]
        return v
    return None


def _get_hypo_sens(metrics: dict[str, Any], idx: int) -> Any:
    if not isinstance(metrics, dict):
        return None
    for key in ("hypo_recall", "hypo_sensitivity", "hypo_sens"):
        val = _get_metric(metrics, key, idx)
        if val is not None:
            return val
    return None


def _get_hypo_fa(metrics: dict[str, Any], idx: int) -> Any:
    if not isinstance(metrics, dict):
        return None
    # false alarm rate = 1 - hypo_precision
    val = _get_metric(metrics, "hypo_precision", idx)
    if val is None:
        return None
    try:
        return 1.0 - float(val)
    except Exception:
        return None


def _get_peg_cde(metrics: dict[str, Any], idx: int) -> Any:
    if not isinstance(metrics, dict):
        return None
    per_h = metrics.get("per_horizon", {})
    if not isinstance(per_h, dict):
        per_h = {}
    val = per_h.get("peg_ratio")
    if isinstance(val, list) and len(val) > 0:
        row = val[min(idx, len(val) - 1)]
        if isinstance(row, list) and len(row) >= 5:
            return 100.0 * sum(row[2:5])
    global_val = metrics.get("global", {}).get("peg_ratio")
    if isinstance(global_val, list) and len(global_val) >= 5:
        return 100.0 * sum(global_val[2:5])
    return None


def _fmt(x: Any) -> str:
    if x is None:
        return "NA"
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return "NA"
        return f"{x:.3f}"
    except Exception:
        return "NA"


def main() -> None:
    runs = []
    if not ROOT.exists():
        raise SystemExit(f"Missing results root: {ROOT}")
    for run_dir in ROOT.iterdir():
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.yaml"
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        raw = json.loads(metrics_path.read_text())
        metrics = raw.get("metrics", {})
        if not metrics:
            continue
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text())
        else:
            # Fall back to embedded configs in metrics.json
            cfg = raw.get("eval_config") or raw.get("train_config") or {}

        ds = _infer_dataset(cfg)
        if ds is None:
            # fallback: parse from run_dir name
            name = run_dir.name.lower()
            for d in DATASETS:
                if d in name:
                    ds = d
                    break
        if ds is None:
            continue

        model = str(cfg.get("model_data", {}).get("model", {}).get("name", "")).lower()
        if not model:
            # fallback: parse from run_dir name
            name = run_dir.name.lower()
            for m in MODELS:
                if m in name:
                    model = m
                    break
        if model not in MODELS:
            continue

        pred_len = cfg.get("model_data", {}).get("dataloader", {}).get("pred_len")
        if pred_len is None:
            # parse from run_dir name (e.g., _p6)
            name = run_dir.name.lower()
            if "_p6" in name:
                pred_len = 6
        if pred_len is None:
            continue
        if int(pred_len) != PRED_LEN:
            continue

        runs.append((ds, model, metrics))

    table = {ds: {m: None for m in MODELS} for ds in DATASETS}
    for ds, model, metrics in runs:
        table[ds][model] = metrics

    lines = []
    for ds in DATASETS:
        lines.append(r"\begin{table}[t]")
        lines.append(r"\centering")
        lines.append(rf"\caption{{{ds.upper()} 30-min forecast (6 steps)}}")
        lines.append(r"\begin{tabular}{l" + "c" * len(MODELS) + r"}")
        lines.append(r"\toprule")
        header = "Metric & " + " & ".join(m.upper() for m in MODELS) + r" \\"
        lines.append(header)
        lines.append(r"\midrule")

        def row(name: str, key_fn):
            vals = []
            for m in MODELS:
                metrics = table[ds].get(m)
                if metrics is None:
                    vals.append("NA")
                else:
                    vals.append(_fmt(key_fn(metrics)))
            lines.append(name + " & " + " & ".join(vals) + r" \\")

        row("RMSE", lambda metrics: _get_metric(metrics, "rmse", HORIZON_IDX))
        row("PEG C+D+E", lambda metrics: _get_peg_cde(metrics, HORIZON_IDX))
        row("Hypo Sens.", lambda metrics: _get_hypo_sens(metrics, HORIZON_IDX))
        row("Hypo False Alarm", lambda metrics: _get_hypo_fa(metrics, HORIZON_IDX))

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")
        lines.append("")

    OUT_FILE.write_text("\n".join(lines))
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
