"""Retrospective test-rate-matched curves, NOT validation-selected deployment thresholds.

The patient bootstrap reselects the cutoff in every draw. Private arrays stay HPC.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np

from evaluate import BASE, MODELS, COHORTS, load_part, evaluate_cutoff, write_json, sha
from warning_metrics import LEADS, BUDGETS, choose_threshold
from matched_metrics import bootstrap_selection, bootstrap_weights, weighted_selected_curves


def run(model, cohort):
    start_time = time.monotonic()
    out = BASE / "aggregate/test_rate_matched" / f"{model}_{cohort}"
    if (out / "completion.json").exists():
        raise RuntimeError("Completed descriptive result exists; do not overwrite.")
    out.mkdir(parents=True, exist_ok=True)
    groups, slices, support, provenance = load_part(model, cohort, "test")
    p0 = evaluate_cutoff(groups, slices, None)
    pids = sorted(p0)
    denominators = np.asarray([[p0[pid]["events"][s] for s in slices] for pid in pids])
    day = np.asarray([p0[pid]["days"] for pid in pids])
    cutoffs, counts, false, records = [], [], [], []
    def scan(cutoff):
        pp = evaluate_cutoff(groups, slices, cutoff)
        count = np.asarray([[pp[pid]["detected"][s] for s in slices] for pid in pids])
        false_counts = np.asarray([pp[pid]["false_alarms"] for pid in pids])
        overall = np.divide(count[:, 0], denominators[:, 0, None],
                            out=np.full((len(pids), len(LEADS)), np.nan), where=denominators[:, 0, None]>0)
        curve = np.nanmean(overall, axis=0)
        area = (np.trapezoid(curve, LEADS) if hasattr(np, "trapezoid") else np.trapz(curve, LEADS))/60
        record = dict(cutoff=cutoff, fa_day=float((false_counts/day).mean()),
                      recall_auc=float(area), recall_0=float(curve[0]))
        cutoffs.append(cutoff)
        counts.append(count)
        false.append(false_counts)
        records.append(record)
    for i, cutoff in enumerate([None] + list(np.arange(0, 180.01, .5))):
        scan(cutoff)
        if i % 50 == 0:
            print(f"{model}/{cohort}: descriptive test grid {i}/362, elapsed {time.monotonic()-start_time:.0f}s", flush=True)
    refine = set()
    for target in BUDGETS:
        cutoff = choose_threshold(records, target)["cutoff"]
        if cutoff is not None:
            refine.update(round(float(v), 2) for v in np.arange(max(0, cutoff-.5), min(180, cutoff+.5)+.025, .05))
    for cutoff in sorted(refine-set(cutoffs)):
        scan(cutoff)
    counts = np.asarray(counts)
    false = np.asarray(false)
    recall = np.divide(counts, denominators[None, :, :, None],
                       out=np.full_like(counts, np.nan, dtype=float), where=denominators[None, :, :, None]>0)
    weights = bootstrap_weights(len(pids))
    boot_selected = bootstrap_selection(false/day[None, :], recall[:, :, 0, :], cutoffs, BUDGETS, LEADS, weights)
    curve_rows, selections, contrast_rows, budget_rows = [], [], [], []
    boot_curves = {}
    main_indices = {}
    for target in BUDGETS:
        selection = choose_threshold(records, target)
        idx = cutoffs.index(selection["cutoff"])
        main_indices[target] = idx
        metadata = dict(model=model, cohort=cohort, target_fa_day=target,
            alarm_cutoff_mgdl=selection["cutoff"], operating_point="descriptive_test_rate_matched",
            test_fa_day=selection["fa_day"])
        selections.append(dict(**metadata, target_shortfall=target-selection["fa_day"],
                               threshold_chosen_on_test=True, deployment_validated=False))
        for si, name in enumerate(slices):
            values = recall[idx, :, si, :]
            mean = np.nanmean(values, axis=0)
            boot = weighted_selected_curves(recall[:, :, si, :], boot_selected[target], weights)
            boot_curves[target, name] = boot
            lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
            events = int(denominators[:, si].sum())
            for j, lead in enumerate(LEADS):
                curve_rows.append(dict(**metadata, slice=name, lead_min=int(lead), recall=mean[j],
                    recall_lo=lo[j], recall_hi=hi[j], n_events=events,
                    n_with_events=int((denominators[:, si]>0).sum()), n_patients=len(pids),
                    n_detected=int(counts[idx, :, si, j].sum()),
                    recall_micro=float(counts[idx, :, si, j].sum()/events) if events else np.nan,
                    patient_days=float(day.sum()), n_false_alarms=int(false[idx].sum()),
                    test_fa_day_micro=float(false[idx].sum()/day.sum()),
                    ci_method="patient bootstrap with within-draw cutoff reselection"))
        pairs = [("post_bolus_30", "not_post_bolus_30"), ("nocturnal", "daytime")]
        if "post_meal_15" in slices:
            pairs.append(("post_meal_15", "not_post_meal_15"))
        for left, right in pairs:
            li, ri = slices.index(left), slices.index(right)
            supported = (denominators[:, li]>0) & (denominators[:, ri]>0)
            difference = recall[:, :, li, :] - recall[:, :, ri, :]
            difference[:, ~supported, :] = np.nan
            point = np.nanmean(difference[idx], axis=0)
            boot = weighted_selected_curves(difference, boot_selected[target], weights)
            lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
            for j, lead in enumerate(LEADS):
                contrast_rows.append(dict(**metadata, left=left, right=right, lead_min=int(lead),
                    difference=point[j], lo=lo[j], hi=hi[j], n_pairs=int(supported.sum())))
    for lower, higher in zip(BUDGETS[:-1], BUDGETS[1:]):
        for si, name in enumerate(slices):
            point = np.nanmean(recall[main_indices[higher], :, si, :] - recall[main_indices[lower], :, si, :], axis=0)
            boot = boot_curves[higher, name] - boot_curves[lower, name]
            lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
            for j, lead in enumerate(LEADS):
                budget_rows.append(dict(model=model, cohort=cohort, slice=name, lead_min=int(lead),
                    lower_budget=lower, higher_budget=higher, difference=point[j], lo=lo[j], hi=hi[j],
                    n_pairs=int((denominators[:, si]>0).sum())))
    write_json(out / "curves.json", curve_rows)
    write_json(out / "calibration.json", selections)
    write_json(out / "slice_contrasts.json", contrast_rows)
    write_json(out / "budget_contrasts.json", budget_rows)
    write_json(out / "support.json", dict(model=model, cohort=cohort, test=support))
    write_json(out / "test_threshold_scan.json", records)
    private = BASE / "private/test_rate_matched" / f"{model}_{cohort}"
    private.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(private / "bootstrap_inputs.npz", patient_ids=pids, denominators=denominators,
                        detections=counts, false_alarms=false, patient_days=day,
                        cutoffs=np.asarray([np.nan if c is None else c for c in cutoffs]))
    write_json(out / "completion.json", dict(model=model, cohort=cohort,
        elapsed_seconds=time.monotonic()-start_time, n_boot=2000,
        threshold_reselected_in_bootstrap=True, threshold_chosen_on_test=True,
        deployment_validated=False, checkpoint_sha256=provenance["checkpoint_sha256"],
        config_sha256=provenance["config_sha256"], code_sha256=sha(__file__)))
    print(f"DONE descriptive {model}/{cohort}, elapsed {time.monotonic()-start_time:.0f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    args = parser.parse_args()
    run(args.model, args.cohort)
