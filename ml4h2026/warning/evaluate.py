"""Calibrate validation FA/day, freeze thresholds, evaluate test lead curves.

Run via Slurm from the benchmark root. Only the aggregate directory is exportable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import yaml

from warning_metrics import (LEADS, BUDGETS, any_warning_window, choose_threshold,
    complete_warning_window, contiguous_onsets, lead_counts, macro_ci,
    match_alarms, retain_alarms)

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
from analysis.evaluate_hypo_gating import _load_split_spec, extract_event_onsets

BASE = Path("outputs/review_revision_20261009/warning_lead_budget")
MODELS = ("arimax", "dlinear", "gru", "itransformer", "patchmlp", "attengluco", "patchmlp_glimmer", "zoh")
COHORTS = ("dclp3", "dclp5", "pedap", "iobp2")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_json(obj):
    if isinstance(obj, dict):
        return {str(k): finite_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return finite_json(obj.tolist())
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


def write_json(path, obj):
    path.write_text(json.dumps(finite_json(obj), indent=2, allow_nan=False))


def load_part(model, cohort, part):
    run = Path("outputs/results") / f"bestck_{model}_{cohort}_p12"
    ec = yaml.safe_load((run / "config.yaml").read_text())["eval"]
    split = ec["split"]
    spec = _load_split_spec(Path(split.get("split_root", "configs/preprocessing")), split["split_name"])
    cache = BASE / "private/caches" / f"{model}_{cohort}_p12" / part
    provenance = json.loads((cache / "provenance.json").read_text())
    source = Path(split.get("processed_root", "datasets_processed")) / spec["dataset"] / "timeseries.csv"
    expected = dict(checkpoint_sha256=sha(run / "checkpoint.pt"), config_sha256=sha(run / "config.yaml"),
                    source_sha256=sha(source), split_part=part, split_patients=spec["patients"][part],
                    min_context_steps=288, horizon_min=60,
                    split_sha256=hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest())
    if any(provenance[k] != value for k, value in expected.items()):
        raise RuntimeError("Cache provenance mismatch.")
    dc = ec["model_data"]["dataloader"]
    if (dc.get("nocturnal_start_hour", 0), dc.get("nocturnal_end_hour", 6)) != (0, 6):
        raise RuntimeError("Unexpected nocturnal definition.")
    with np.load(cache / "dense_min_predictions.npz") as loaded:
        arr = {k: loaded[k] for k in loaded.files}
    # Identical patient/origin support across all model families after the
    # common 24-hour source-context filter. Fail rather than compare unmatched grids.
    ref = BASE / "private/caches" / f"patchmlp_glimmer_{cohort}_p12" / part / "dense_min_predictions.npz"
    with np.load(ref) as loaded:
        for field in ("pid", "sid", "idx"):
            if not np.array_equal(arr[field], loaded[field]):
                raise RuntimeError(f"{model}/{cohort}/{part}: origin grid differs from reference.")
    if not np.isfinite(arr["min_pred"]).all():
        raise RuntimeError("Nonfinite dense scores.")
    raw = pd.read_csv(source)
    raw = raw[raw.pat_id.isin(spec["patients"][part])].dropna(subset=["cgm"])
    raw = raw.sort_values(["pat_id", "seq_id", "date"])
    bolus_cols = [c for c in raw if "bolus" in c.lower()]
    meal_cols = [c for c in raw if "meal" in c.lower() or "carb" in c.lower()]
    has_meals = bool(meal_cols and (raw[meal_cols].fillna(0).to_numpy() > 0).any())
    slices = ["overall", "post_bolus_30", "not_post_bolus_30", "nocturnal", "daytime"]
    if has_meals:
        slices += ["post_meal_15", "not_post_meal_15"]
    groups = []
    support = dict(n_cached_origins=len(arr["idx"]), n_valid_origins=0,
                   n_unknown_start_onsets_excluded=0, n_matchable_events=0,
                   n_complete_warning_events=0, n_sequences=0,
                   meal_annotations_available=has_meals)
    # Group the already sorted prediction arrays once (avoid repeatedly scanning millions of rows).
    changes = np.r_[0, np.where((np.diff(arr["pid"]) != 0) | (np.diff(arr["sid"]) != 0))[0] + 1, len(arr["idx"])]
    lookup = {(int(arr["pid"][start]), int(arr["sid"][start])): (start, stop)
              for start, stop in zip(changes[:-1], changes[1:])}
    for key, frame in raw.groupby(["pat_id", "seq_id"], sort=False):
        key = (int(key[0]), int(key[1]))
        if key not in lookup:
            continue
        start, stop = lookup[key]
        origins_idx = arr["idx"][start:stop]
        dates = pd.to_datetime(frame.date)
        ns = dates.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        delta = np.diff(ns)
        if (delta <= 0).any() or (delta % (300 * 10**9) != 0).any():
            raise RuntimeError("Non-increasing dates or dates off the five-minute grid.")
        slots = ns // (300 * 10**9)
        if (origins_idx < 287).any() or (origins_idx + 12 >= len(slots)).any():
            raise RuntimeError("Unexpected context/forecast source coverage.")
        # No forecast may bridge a missing target or a gap in the common 24-hour input context.
        valid = ((slots[origins_idx + 12] - slots[origins_idx] == 12) &
                 (slots[origins_idx] - slots[origins_idx - 287] == 287))
        origins = slots[origins_idx[valid]]
        scores = arr["min_pred"][start:stop][valid]
        if not len(origins):
            continue
        if not (np.diff(origins) > 0).all():
            raise RuntimeError("Duplicate or unordered origins.")
        cgm = frame.cgm.to_numpy()
        # Compare copied event logic to original implementation on each segment.
        bounds = np.r_[0, np.where(np.diff(slots) != 1)[0] + 1, len(slots)]
        from warning_metrics import event_onsets
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if event_onsets(cgm[lo:hi]) != extract_event_onsets(cgm[lo:hi], 70, 3):
                raise RuntimeError("Copied event episode logic differs from source protocol.")
            if len(cgm[lo:hi]) >= 3 and (cgm[lo:lo+3] < 70).all():
                support["n_unknown_start_onsets_excluded"] += 1
        onset_indices = contiguous_onsets(cgm, slots)
        matchable = [(idx, int(slots[idx])) for idx in onset_indices
                     if any_warning_window(origins, int(slots[idx]))]
        complete = [(idx, slot) for idx, slot in matchable if complete_warning_window(origins, slot)]
        def since(columns):
            pos = (frame[columns].fillna(0).to_numpy() > 0).any(axis=1)
            last = np.maximum.accumulate(np.where(pos, slots, -10**12))
            return slots - last
        bolus = since(bolus_cols) if bolus_cols else np.full(len(frame), 10**12)
        meal = since(meal_cols) if has_meals else None
        night = dates.dt.hour.to_numpy() < 6
        masks = dict(overall=np.ones(len(frame), dtype=bool), post_bolus_30=bolus <= 6,
                     not_post_bolus_30=bolus > 6, nocturnal=night, daytime=~night)
        if has_meals:
            masks.update(post_meal_15=meal <= 3, not_post_meal_15=meal > 3)
        selected = {name: [slot for idx, slot in complete if masks[name][idx]] for name in slices}
        groups.append(dict(pid=key[0], origins=origins, scores=scores,
                           matchable=[slot for _, slot in matchable], selected=selected,
                           days=len(origins) / 288.0))
        support["n_valid_origins"] += len(origins)
        support["n_matchable_events"] += len(matchable)
        support["n_complete_warning_events"] += len(complete)
        support["n_sequences"] += 1
    if not groups:
        raise RuntimeError("No valid continuous timeline to evaluate.")
    support["n_patients"] = len({g["pid"] for g in groups})
    support["patient_days"] = sum(g["days"] for g in groups)
    support["slice_events"] = {s: sum(len(g["selected"][s]) for g in groups) for s in slices}
    return groups, slices, support, provenance


def evaluate_cutoff(groups, slices, cutoff):
    patients = {}
    for group in groups:
        p = patients.setdefault(group["pid"], dict(days=0.0, false_alarms=0,
            alarms=0, events={s: 0 for s in slices},
            detected={s: np.zeros(len(LEADS), dtype=int) for s in slices}))
        alarms = retain_alarms(group["origins"], group["scores"], cutoff)
        detected, used = match_alarms(alarms, group["matchable"])
        p["days"] += group["days"]
        p["false_alarms"] += len(alarms) - len(used)
        p["alarms"] += len(alarms)
        for name in slices:
            events = group["selected"][name]
            p["events"][name] += len(events)
            p["detected"][name] += lead_counts(events, detected)
    return patients


def score_cutoff(groups, cutoff):
    patients = evaluate_cutoff(groups, ["overall"], cutoff)
    fa = np.asarray([p["false_alarms"] / p["days"] for p in patients.values()])
    curves = np.asarray([p["detected"]["overall"] / p["events"]["overall"]
                        for p in patients.values() if p["events"]["overall"]])
    curve = curves.mean(axis=0) if len(curves) else np.full(len(LEADS), np.nan)
    area = np.trapezoid(curve, LEADS) / 60 if hasattr(np, "trapezoid") else np.trapz(curve, LEADS) / 60
    return dict(cutoff=cutoff, fa_day=float(fa.mean()), recall_0=float(curve[0]), recall_auc=float(area))


def summarize(patients, slices, metadata):
    rows = []
    fa = [p["false_alarms"] / p["days"] for p in patients.values()]
    fm, fl, fh = [float(v[0]) for v in macro_ci(fa)]
    total_days = sum(p["days"] for p in patients.values())
    total_fa = sum(p["false_alarms"] for p in patients.values())
    for name in slices:
        supported = [p for p in patients.values() if p["events"][name]]
        counts = np.asarray([p["detected"][name] for p in supported])
        denominators = np.asarray([p["events"][name] for p in supported])
        vals = counts / denominators[:, None] if len(supported) else np.empty((0, len(LEADS)))
        mean, lo, hi = macro_ci(vals)
        if len(supported) and (np.diff(mean) > 1e-12).any():
            raise RuntimeError("Recall must not increase as required warning time increases.")
        for j, lead in enumerate(LEADS):
            rows.append(dict(**metadata, slice=name, lead_min=int(lead), recall=mean[j],
                recall_lo=lo[j], recall_hi=hi[j], n_events=int(denominators.sum()),
                n_detected=int(counts[:, j].sum()) if len(counts) else 0,
                n_with_events=len(supported), n_patients=len(patients),
                recall_micro=float(counts[:, j].sum()/denominators.sum()) if denominators.sum() else np.nan,
                test_fa_day=fm, test_fa_day_lo=fl, test_fa_day_hi=fh,
                test_fa_day_micro=total_fa/total_days, patient_days=total_days,
                n_false_alarms=total_fa, n_alarms=sum(p["alarms"] for p in patients.values())))
    return rows


def paired_contrasts(patients, slices, metadata):
    rows = []
    pairs = [("post_bolus_30", "not_post_bolus_30"), ("nocturnal", "daytime")]
    if "post_meal_15" in slices:
        pairs.append(("post_meal_15", "not_post_meal_15"))
    for left, right in pairs:
        for p in patients.values():
            if p["events"][left] + p["events"][right] != p["events"]["overall"]:
                raise RuntimeError("Complementary slice denominators must sum to overall.")
            if not np.array_equal(p["detected"][left]+p["detected"][right], p["detected"]["overall"]):
                raise RuntimeError("Complementary slice detections must sum to overall.")
        complete = [p for p in patients.values() if p["events"][left] and p["events"][right]]
        values = np.asarray([p["detected"][left]/p["events"][left] -
                             p["detected"][right]/p["events"][right] for p in complete])
        values = values if len(values) else np.empty((0, len(LEADS)))
        mean, lo, hi = macro_ci(values)
        for j, lead in enumerate(LEADS):
            rows.append(dict(**metadata, left=left, right=right, lead_min=int(lead),
                             difference=mean[j], lo=lo[j], hi=hi[j], n_pairs=len(complete)))
    return rows


def run(model, cohort):
    start_time = time.monotonic()
    out = BASE / "aggregate" / f"{model}_{cohort}"
    if (out / "completion.json").exists():
        raise RuntimeError("Completed evaluation exists; use a new output directory for changes.")
    out.mkdir(parents=True, exist_ok=True)
    private = BASE / "private/evaluation" / f"{model}_{cohort}"
    private.mkdir(parents=True, exist_ok=True)
    val, val_slices, val_support, val_prov = load_part(model, cohort, "val")
    test, test_slices, test_support, test_prov = load_part(model, cohort, "test")
    if set(val_prov["split_patients"]) & set(test_prov["split_patients"]):
        raise RuntimeError("Validation and test patients overlap.")
    # Fixed broad grid plus a no-alarm candidate. Fine refinement uses validation only.
    cutoffs = [None] + list(np.arange(0.0, 180.01, 0.5))
    scores = []
    for i, cutoff in enumerate(cutoffs):
        scores.append(score_cutoff(val, cutoff))
        if i % 50 == 0:
            print(f"{model}/{cohort}: validation threshold {i}/{len(cutoffs)}, elapsed {time.monotonic()-start_time:.0f}s", flush=True)
    refine = set()
    for budget in BUDGETS:
        selected = choose_threshold(scores, budget)
        if selected["cutoff"] is not None:
            refine.update(round(float(c), 2) for c in np.arange(max(0, selected["cutoff"]-.5),
                           min(180, selected["cutoff"]+.5)+.025, .05))
    seen = {r["cutoff"] for r in scores}
    scores += [score_cutoff(val, c) for c in sorted(refine-seen)]
    write_json(out / "validation_threshold_scan.json", scores)
    curves, contrasts, selections, budget_contrasts = [], [], [], []
    test_patients = {}
    for budget in BUDGETS:
        selected = choose_threshold(scores, budget)
        # No use of test false alarms, recall, or slice membership in calibration.
        selection = dict(model=model, cohort=cohort, target_fa_day=budget,
            alarm_cutoff_mgdl=selected["cutoff"], validation_fa_day=selected["fa_day"],
            validation_recall_0=selected["recall_0"], validation_recall_auc=selected["recall_auc"],
            validation_target_shortfall=budget-selected["fa_day"])
        selections.append(selection)
        patients = evaluate_cutoff(test, test_slices, selected["cutoff"])
        test_patients[budget] = patients
        metadata = dict(model=model, cohort=cohort, target_fa_day=budget,
                        alarm_cutoff_mgdl=selected["cutoff"], validation_fa_day=selected["fa_day"])
        curves += summarize(patients, test_slices, metadata)
        contrasts += paired_contrasts(patients, test_slices, metadata)
        write_json(private / f"patients_budget_{budget:g}.json", patients)
    for a, b in zip(BUDGETS[:-1], BUDGETS[1:]):
        for name in test_slices:
            ids = [pid for pid in test_patients[a] if test_patients[a][pid]["events"][name]]
            differences = np.asarray([(test_patients[b][pid]["detected"][name] -
                test_patients[a][pid]["detected"][name]) / test_patients[a][pid]["events"][name] for pid in ids])
            differences = differences if len(differences) else np.empty((0, len(LEADS)))
            mean, lo, hi = macro_ci(differences)
            for j, lead in enumerate(LEADS):
                budget_contrasts.append(dict(model=model, cohort=cohort, slice=name, lead_min=int(lead),
                    lower_budget=a, higher_budget=b, difference=mean[j], lo=lo[j], hi=hi[j], n_pairs=len(ids)))
    write_json(out / "curves.json", curves)
    write_json(out / "slice_contrasts.json", contrasts)
    write_json(out / "budget_contrasts.json", budget_contrasts)
    write_json(out / "calibration.json", selections)
    write_json(out / "support.json", dict(model=model, cohort=cohort, validation=val_support, test=test_support))
    write_json(out / "completion.json", dict(model=model, cohort=cohort, elapsed_seconds=time.monotonic()-start_time,
        checkpoint_sha256=test_prov["checkpoint_sha256"], config_sha256=test_prov["config_sha256"],
        source_sha256=test_prov["source_sha256"], code_sha256=sha(__file__), metrics_sha256=sha(Path(__file__).with_name("warning_metrics.py")),
        validation_test_disjoint=True, common_model_origin_grid_verified=True,
        original_event_logic_equivalence_verified=True, monotone_lead_curves_verified=True,
        fixed_threshold_test=True))
    print(f"DONE {model}/{cohort}, elapsed {time.monotonic()-start_time:.0f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    args = parser.parse_args()
    run(args.model, args.cohort)
