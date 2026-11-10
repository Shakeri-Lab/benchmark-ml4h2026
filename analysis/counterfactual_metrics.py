from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Counterfactual metrics for action-conditional simulator benchmarks.
#
# Intuition:
# - rmse_effect: does the model predict the *magnitude* of the intervention
#   effect (counterfactual - factual) correctly?
# - sign_agreement: does the model predict the *direction* of the effect?
# - rmse_factual / rmse_counterfactual: how accurate are predictions on factual
#   and counterfactual rollouts themselves (not just their difference)?
# - ordinal metrics (Kendall / exact / pairwise): does the model preserve the
#   *relative ordering* across intervention intensities within the same episode?
# ---------------------------------------------------------------------------


def _to_str_array(arr: np.ndarray) -> np.ndarray:
    out = []
    for x in arr:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return np.asarray(out, dtype=object)


def _detect_family(name: str, fallback: Optional[str] = None) -> str:
    lower = (name or "").lower()
    if "basal" in lower:
        return "basal"
    if "bolus" in lower:
        return "bolus"
    return fallback or "unknown"


_INTENSITY_RE = re.compile(r"_(\d+(?:\.\d+)?)x$", re.IGNORECASE)


def _parse_intensity(name: str) -> Optional[float]:
    match = _INTENSITY_RE.search((name or "").lower())
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _sign_with_tol(x: float, tol: float) -> int:
    if abs(x) <= tol:
        return 0
    return 1 if x > 0 else -1


def _kendall_tau_b(
    true_vals: Sequence[float],
    pred_vals: Sequence[float],
    tol_true: float,
    tol_pred: float,
) -> float:
    n = len(true_vals)
    if n < 2:
        return float("nan")
    concordant = 0
    discordant = 0
    ties_true = 0
    ties_pred = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = true_vals[i] - true_vals[j]
            dy = pred_vals[i] - pred_vals[j]
            tie_x = abs(dx) <= tol_true
            tie_y = abs(dy) <= tol_pred
            if tie_x and tie_y:
                # tied in both => contributes to neither concordant nor discordant
                continue
            if tie_x:
                ties_true += 1
                continue
            if tie_y:
                ties_pred += 1
                continue
            if dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + ties_true) * (concordant + discordant + ties_pred))
    if denom == 0:
        return float("nan")
    return (concordant - discordant) / denom


def _pairwise_accuracy(
    true_vals: Sequence[float],
    pred_vals: Sequence[float],
    tol_true: float,
    tol_pred: float,
) -> Tuple[float, int]:
    n = len(true_vals)
    if n < 2:
        return float("nan"), 0
    correct = 0
    total = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = true_vals[i] - true_vals[j]
            if abs(dx) <= tol_true:
                # uninformative pair
                continue
            dy = pred_vals[i] - pred_vals[j]
            total += 1
            sign_true = _sign_with_tol(dx, tol_true)
            sign_pred = _sign_with_tol(dy, tol_pred)
            if sign_pred == sign_true:
                correct += 1
    if total == 0:
        return float("nan"), 0
    return correct / total, total


def _exact_match(
    true_vals: Sequence[float],
    pred_vals: Sequence[float],
    tol_true: float,
    tol_pred: float,
) -> Tuple[float, int]:
    acc, total = _pairwise_accuracy(true_vals, pred_vals, tol_true, tol_pred)
    if total == 0 or np.isnan(acc):
        return float("nan"), total
    return 1.0 if acc >= 1.0 else 0.0, total


def _load_artifacts(npz_path: Path) -> pd.DataFrame:
    data = np.load(npz_path, allow_pickle=True)
    gts = data["gts"]
    preds = data["preds"]
    pat_id = data["pat_id"].astype(int)
    seq_id = data["seq_id"].astype(int)
    scenario_name = _to_str_array(data["scenario_name"])
    scenario_type = _to_str_array(data["scenario_type"])
    perturb_family = _to_str_array(data["perturb_family"])

    records = []
    for i in range(len(pat_id)):
        records.append(
            {
                "pat_id": int(pat_id[i]),
                "seq_id": int(seq_id[i]),
                "scenario_name": scenario_name[i],
                "scenario_type": scenario_type[i],
                "perturb_family": perturb_family[i],
                "gts": np.asarray(gts[i]),
                "preds": np.asarray(preds[i]),
            }
        )
    return pd.DataFrame(records)


def _load_valid_scenarios(path: Path) -> set[tuple[int, int, str]]:
    df = pd.read_csv(path)
    required = {"pat_id", "seq_id", "scenario_name"}
    if not required.issubset(df.columns):
        raise KeyError(f"Missing columns in {path}: {required - set(df.columns)}")
    return {
        (int(r.pat_id), int(r.seq_id), str(r.scenario_name))
        for r in df.itertuples(index=False)
    }


def _filter_valid(df: pd.DataFrame, valid_set: set[tuple[int, int, str]]) -> pd.DataFrame:
    mask = [
        (int(row.pat_id), int(row.seq_id), str(row.scenario_name)) in valid_set
        for row in df.itertuples(index=False)
    ]
    return df.loc[mask].reset_index(drop=True)


def _resolve_horizon_indices(horizon_minutes: Sequence[int], step_minutes: int, pred_len: int) -> Dict[int, int]:
    idx_map = {}
    for h in horizon_minutes:
        steps_ahead = int(round(h / step_minutes))
        if abs(h - steps_ahead * step_minutes) > 1e-6:
            continue
        # Forecast arrays store the first future step at index 0, so horizon h
        # minutes corresponds to the (h / step_minutes)-th future step.
        idx = steps_ahead - 1
        if idx < 0 or idx >= pred_len:
            continue
        idx_map[h] = idx
    return idx_map


def compute_effect_and_sign_metrics(
    df: pd.DataFrame,
    horizon_idx: Dict[int, int],
    zero_tol: float,
) -> List[Dict[str, object]]:
    factual = {
        (int(r.pat_id), int(r.seq_id)): r
        for r in df.itertuples(index=False)
        if str(r.scenario_name).lower() == "factual"
    }

    # accumulators: (family, perturb_type, horizon) -> list of deltas / counts
    deltas = {}
    sign_counts = {}

    for r in df.itertuples(index=False):
        if str(r.scenario_name).lower() == "factual":
            continue
        key_ep = (int(r.pat_id), int(r.seq_id))
        if key_ep not in factual:
            continue
        fact = factual[key_ep]
        family = _detect_family(str(r.scenario_name), fallback=str(r.perturb_family))
        pert_type = str(r.scenario_name)
        gts_cf = np.asarray(r.gts).reshape(-1)
        preds_cf = np.asarray(r.preds).reshape(-1)
        gts_f = np.asarray(fact.gts).reshape(-1)
        preds_f = np.asarray(fact.preds).reshape(-1)
        for h_min, h_idx in horizon_idx.items():
            if h_idx >= len(gts_cf) or h_idx >= len(gts_f):
                continue
            dt = gts_cf[h_idx] - gts_f[h_idx]
            dp = preds_cf[h_idx] - preds_f[h_idx]
            if np.isnan(dt) or np.isnan(dp):
                continue
            key = (family, pert_type, h_min)
            deltas.setdefault(key, []).append((dp, dt))
            sign_key = (family, pert_type, h_min)
            sign_counts.setdefault(sign_key, []).append((dp, dt))

            # family aggregate
            fam_key = (family, "__ALL__", h_min)
            deltas.setdefault(fam_key, []).append((dp, dt))
            sign_counts.setdefault(fam_key, []).append((dp, dt))

    records = []
    for (family, pert_type, h_min), vals in deltas.items():
        if not vals:
            continue
        diff = [(dp - dt) for dp, dt in vals]
        rmse = float(np.sqrt(np.mean(np.square(diff)))) if diff else float("nan")
        records.append(
            {
                "perturbation_family": family,
                "perturbation_type": pert_type,
                "ordinal_group": None,
                "horizon_minutes": h_min,
                "metric_name": "rmse_effect",
                "metric_value": rmse,
                "number_of_valid_pairs": len(vals),
                "number_of_valid_groups": None,
            }
        )

    for (family, pert_type, h_min), vals in sign_counts.items():
        if not vals:
            continue
        agree = 0
        total = 0
        for dp, dt in vals:
            sign_true = _sign_with_tol(dt, zero_tol)
            sign_pred = _sign_with_tol(dp, zero_tol)
            total += 1
            if sign_true == sign_pred:
                agree += 1
        frac = float(agree / total) if total > 0 else float("nan")
        records.append(
            {
                "perturbation_family": family,
                "perturbation_type": pert_type,
                "ordinal_group": None,
                "horizon_minutes": h_min,
                "metric_name": "sign_agreement",
                "metric_value": frac,
                "number_of_valid_pairs": total,
                "number_of_valid_groups": None,
            }
        )

    return records


def compute_rmse_metrics(
    df: pd.DataFrame,
    horizon_idx: Dict[int, int],
) -> List[Dict[str, object]]:
    # Accumulate errors per (family, type, horizon)
    errs: Dict[Tuple[str, str, int], List[float]] = {}

    for r in df.itertuples(index=False):
        name = str(r.scenario_name)
        gts = np.asarray(r.gts).reshape(-1)
        preds = np.asarray(r.preds).reshape(-1)
        for h_min, h_idx in horizon_idx.items():
            if h_idx >= len(gts) or h_idx >= len(preds):
                continue
            if np.isnan(gts[h_idx]) or np.isnan(preds[h_idx]):
                continue
            err = float(preds[h_idx] - gts[h_idx])
            if name.lower() == "factual":
                key = ("factual", "factual", h_min)
                errs.setdefault(key, []).append(err)
            else:
                family = _detect_family(name, fallback=str(r.perturb_family))
                key = (family, name, h_min)
                errs.setdefault(key, []).append(err)
                fam_key = (family, "__ALL__", h_min)
                errs.setdefault(fam_key, []).append(err)

    records: List[Dict[str, object]] = []
    for (family, pert_type, h_min), vals in errs.items():
        if not vals:
            continue
        rmse = float(np.sqrt(np.mean(np.square(vals))))
        metric_name = "rmse_factual" if family == "factual" else "rmse_counterfactual"
        records.append(
            {
                "perturbation_family": family,
                "perturbation_type": pert_type,
                "ordinal_group": None,
                "horizon_minutes": h_min,
                "metric_name": metric_name,
                "metric_value": rmse,
                "number_of_valid_pairs": len(vals),
                "number_of_valid_groups": None,
            }
        )

    return records


def compute_ordinal_metrics(
    df: pd.DataFrame,
    horizon_idx: Dict[int, int],
    tol_true: float,
    tol_pred: float,
    min_group_size: int,
) -> List[Dict[str, object]]:
    # Build groups: (pat_id, seq_id, family) -> members
    groups: Dict[Tuple[int, int, str], List[Dict[str, object]]] = {}
    factual_rows = {
        (int(r.pat_id), int(r.seq_id)): r
        for r in df.itertuples(index=False)
        if str(r.scenario_name).lower() == "factual"
    }

    for r in df.itertuples(index=False):
        name = str(r.scenario_name)
        family = _detect_family(name, fallback=None)
        intensity = _parse_intensity(name)
        if intensity is None:
            continue
        if family not in {"basal", "bolus"}:
            continue
        key = (int(r.pat_id), int(r.seq_id), family)
        groups.setdefault(key, []).append(
            {
                "scenario_name": name,
                "intensity": intensity,
                "gts": np.asarray(r.gts).reshape(-1),
                "preds": np.asarray(r.preds).reshape(-1),
            }
        )

    # Insert factual as intensity 1.0 when available
    for (pat_id, seq_id), fact in factual_rows.items():
        for family in ["basal", "bolus"]:
            key = (pat_id, seq_id, family)
            if key not in groups:
                continue
            groups[key].append(
                {
                    "scenario_name": "factual",
                    "intensity": 1.0,
                    "gts": np.asarray(fact.gts).reshape(-1),
                    "preds": np.asarray(fact.preds).reshape(-1),
                }
            )

    # Aggregate per family and horizon
    per_family: Dict[Tuple[str, int], List[Tuple[float, float, float, int]]] = {}
    for (pat_id, seq_id, family), members in groups.items():
        # Deduplicate by scenario_name to avoid duplicates
        unique = {}
        for m in members:
            unique[m["scenario_name"]] = m
        members = list(unique.values())
        if len(members) < min_group_size:
            continue
        for h_min, h_idx in horizon_idx.items():
            vals_true = []
            vals_pred = []
            for m in members:
                gts = m["gts"]
                preds = m["preds"]
                if h_idx >= len(gts) or h_idx >= len(preds):
                    continue
                vals_true.append(float(gts[h_idx]))
                vals_pred.append(float(preds[h_idx]))
            if len(vals_true) < min_group_size:
                continue
            kendall = _kendall_tau_b(vals_true, vals_pred, tol_true, tol_pred)
            pair_acc, pair_count = _pairwise_accuracy(vals_true, vals_pred, tol_true, tol_pred)
            exact, _ = _exact_match(vals_true, vals_pred, tol_true, tol_pred)
            per_family.setdefault((family, h_min), []).append((kendall, exact, pair_acc, pair_count))

    records = []
    for (family, h_min), vals in per_family.items():
        if not vals:
            continue
        kendalls = [v[0] for v in vals if not np.isnan(v[0])]
        exacts = [v[1] for v in vals if not np.isnan(v[1])]
        pair_accs = [v[2] for v in vals if not np.isnan(v[2])]
        n_groups = len(vals)
        if kendalls:
            records.append(
                {
                    "perturbation_family": family,
                    "perturbation_type": None,
                    "ordinal_group": f"{family}_intensity",
                    "horizon_minutes": h_min,
                    "metric_name": "kendall_rank_corr",
                    "metric_value": float(np.nanmean(kendalls)),
                    "number_of_valid_pairs": None,
                    "number_of_valid_groups": n_groups,
                }
            )
        if exacts:
            records.append(
                {
                    "perturbation_family": family,
                    "perturbation_type": None,
                    "ordinal_group": f"{family}_intensity",
                    "horizon_minutes": h_min,
                    "metric_name": "ordinal_exact_match",
                    "metric_value": float(np.nanmean(exacts)),
                    "number_of_valid_pairs": None,
                    "number_of_valid_groups": n_groups,
                }
            )
        if pair_accs:
            records.append(
                {
                    "perturbation_family": family,
                    "perturbation_type": None,
                    "ordinal_group": f"{family}_intensity",
                    "horizon_minutes": h_min,
                    "metric_name": "ordinal_pairwise_accuracy",
                    "metric_value": float(np.nanmean(pair_accs)),
                    "number_of_valid_pairs": None,
                    "number_of_valid_groups": n_groups,
                }
            )

    return records


def _macro_average_patient_records(
    patient_records: pd.DataFrame,
    *,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    key_cols = [
        "perturbation_family",
        "perturbation_type",
        "ordinal_group",
        "horizon_minutes",
        "metric_name",
    ]
    out_cols = [
        *key_cols,
        "metric_value",
        "ci_low",
        "ci_high",
        "number_of_valid_pairs",
        "number_of_valid_groups",
        "number_of_patients",
    ]
    if patient_records.empty:
        return pd.DataFrame(columns=out_cols)

    rng = np.random.default_rng(int(bootstrap_seed))
    rows: List[Dict[str, Any]] = []
    for key_vals, sub in patient_records.groupby(key_cols, dropna=False, sort=True):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        metric_vals = pd.to_numeric(sub["metric_value"], errors="coerce").to_numpy(dtype=float)
        metric_vals = metric_vals[~np.isnan(metric_vals)]
        if metric_vals.size == 0:
            continue

        row = {col: val for col, val in zip(key_cols, key_vals)}
        row["metric_value"] = float(np.mean(metric_vals))
        row["number_of_patients"] = int(sub["pat_id"].nunique())

        valid_pairs = pd.to_numeric(sub["number_of_valid_pairs"], errors="coerce").to_numpy(dtype=float)
        row["number_of_valid_pairs"] = float(np.nansum(valid_pairs)) if np.any(~np.isnan(valid_pairs)) else float("nan")

        valid_groups = pd.to_numeric(sub["number_of_valid_groups"], errors="coerce").to_numpy(dtype=float)
        row["number_of_valid_groups"] = (
            float(np.nansum(valid_groups)) if np.any(~np.isnan(valid_groups)) else float("nan")
        )

        if int(n_bootstrap) > 0:
            boots = np.empty(int(n_bootstrap), dtype=float)
            n = metric_vals.size
            for i in range(int(n_bootstrap)):
                sample = metric_vals[rng.integers(0, n, size=n)]
                boots[i] = float(np.mean(sample))
            row["ci_low"] = float(np.percentile(boots, 2.5))
            row["ci_high"] = float(np.percentile(boots, 97.5))
        else:
            row["ci_low"] = float("nan")
            row["ci_high"] = float("nan")
        rows.append(row)

    return pd.DataFrame(rows, columns=out_cols)


def build_results(
    df: pd.DataFrame,
    horizon_minutes: Sequence[int],
    step_minutes: int,
    zero_tol: float,
    tol_true: float,
    tol_pred: float,
    min_group_size: int,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> Dict[str, object]:
    # Determine pred_len from first sample
    if df.empty:
        raise ValueError("No samples to evaluate.")
    pred_len = int(np.asarray(df.iloc[0].gts).reshape(-1).shape[0])
    horizon_idx = _resolve_horizon_indices(horizon_minutes, step_minutes, pred_len)
    if not horizon_idx:
        raise ValueError("No requested horizons are available in the predictions.")

    patient_rows: List[Dict[str, Any]] = []
    for pat_id, sub in df.groupby("pat_id", sort=True):
        effect_records = compute_effect_and_sign_metrics(sub, horizon_idx, zero_tol)
        rmse_records = compute_rmse_metrics(sub, horizon_idx)
        ordinal_records = compute_ordinal_metrics(sub, horizon_idx, tol_true, tol_pred, min_group_size)
        for rec in effect_records + rmse_records + ordinal_records:
            patient_rows.append({"pat_id": int(pat_id), **rec})

    patient_records = pd.DataFrame(patient_rows)
    records = _macro_average_patient_records(
        patient_records,
        n_bootstrap=int(n_bootstrap),
        bootstrap_seed=int(bootstrap_seed),
    )

    return {
        "horizons_minutes": list(horizon_idx.keys()),
        "step_minutes": step_minutes,
        "aggregation": "patient_macro_average",
        "bootstrap": {"n_rounds": int(n_bootstrap), "seed": int(bootstrap_seed), "unit": "patient"},
        "records": records.to_dict(orient="records"),
        "patient_records": patient_records.to_dict(orient="records"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Counterfactual benchmark metrics from saved artifacts.")
    p.add_argument("--artifacts", type=Path, required=True, help="Path to cf_artifacts.npz")
    p.add_argument(
        "--test-scenarios",
        type=Path,
        default=Path("datasets_processed/action_cf_benchmark/test_scenarios.csv"),
        help="Path to test_scenarios.csv (valid scenarios)",
    )
    p.add_argument(
        "--horizons",
        type=str,
        default="30,60,90,120",
        help="Comma-separated horizons in minutes (default: 30,60,90,120)",
    )
    p.add_argument("--step-minutes", type=int, default=5, help="Time step in minutes (default: 5)")
    p.add_argument("--zero-tol", type=float, default=1e-6, help="Tolerance for zero sign (default: 1e-6)")
    p.add_argument("--tol-true", type=float, default=1e-3, help="Tolerance for ties in true values")
    p.add_argument("--tol-pred", type=float, default=1e-3, help="Tolerance for ties in predicted values")
    p.add_argument("--min-group-size", type=int, default=3, help="Minimum ordinal group size (default: 3)")
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of patient-level bootstrap rounds for 95%% CIs (default: 1000)",
    )
    p.add_argument("--bootstrap-seed", type=int, default=42, help="Random seed for patient bootstrap")
    p.add_argument("--out-json", type=Path, default=None, help="Optional JSON output path")
    p.add_argument("--out-csv", type=Path, default=None, help="Optional CSV output path")
    p.add_argument("--out-patient-json", type=Path, default=None, help="Optional JSON output path for per-patient records")
    p.add_argument("--out-patient-csv", type=Path, default=None, help="Optional CSV output path for per-patient records")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = _load_artifacts(args.artifacts)
    if args.test_scenarios.exists():
        valid = _load_valid_scenarios(args.test_scenarios)
        df = _filter_valid(df, valid)

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    results = build_results(
        df,
        horizon_minutes=horizons,
        step_minutes=int(args.step_minutes),
        zero_tol=float(args.zero_tol),
        tol_true=float(args.tol_true),
        tol_pred=float(args.tol_pred),
        min_group_size=int(args.min_group_size),
        n_bootstrap=int(args.n_bootstrap),
        bootstrap_seed=int(args.bootstrap_seed),
    )

    if args.out_json:
        args.out_json.write_text(json.dumps(results, indent=2))
    if args.out_csv:
        pd.DataFrame(results["records"]).to_csv(args.out_csv, index=False)
    if args.out_patient_json:
        args.out_patient_json.write_text(json.dumps(results["patient_records"], indent=2))
    if args.out_patient_csv:
        pd.DataFrame(results["patient_records"]).to_csv(args.out_patient_csv, index=False)
    if not args.out_json and not args.out_csv and not args.out_patient_json and not args.out_patient_csv:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
