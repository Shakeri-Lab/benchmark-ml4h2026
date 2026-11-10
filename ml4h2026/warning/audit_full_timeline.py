"""Independently reproduce saved point metrics without slicing alarm origins.

Runs on HPC; exports aggregate diagnostics only. Existing results are read-only.
"""
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path

import numpy as np

from evaluate import BASE, COHORTS, MODELS, load_part, sha, write_json
from warning_metrics import LEADS


def independent_match(origins, scores, cutoff, events):
    # Do not call the production alarm-suppression or matching functions.
    alarms = []
    if cutoff is not None:
        last_alarm = None
        for origin in origins[scores < cutoff]:
            origin = int(origin)
            if last_alarm is None or origin - last_alarm >= 12:
                alarms.append(origin)
                last_alarm = origin
    # Chronological alarm-first matching, rather than the production event-first
    # loop. Each alarm and event can be used at most once, strictly before onset.
    event_pos = 0
    detected = {}
    for alarm in alarms:
        while event_pos < len(events) and events[event_pos] <= alarm:
            event_pos += 1
        if event_pos < len(events) and events[event_pos] <= alarm + 12:
            onset = int(events[event_pos])
            detected[onset] = (onset - alarm) * 5
            event_pos += 1
    return detected, len(alarms) - len(detected)


def recount(groups, slices, cutoff):
    patients = {}
    for group in groups:
        p = patients.setdefault(group["pid"], dict(days=0., false=0,
            events={name: 0 for name in slices},
            counts={name: np.zeros(len(LEADS), dtype=int) for name in slices}))
        detected, false = independent_match(group["origins"], group["scores"], cutoff,
                                             group["matchable"])
        p["days"] += group["days"]
        p["false"] += false
        for name in slices:
            events = group["selected"][name]
            p["events"][name] += len(events)
            times = np.asarray([detected.get(int(event), -1) for event in events])
            p["counts"][name] += (times[:, None] >= LEADS[None, :]).sum(axis=0)
    for p in patients.values():
        for left, right in (("post_bolus_30", "not_post_bolus_30"), ("nocturnal", "daytime")):
            assert p["events"][left] + p["events"][right] == p["events"]["overall"]
            np.testing.assert_array_equal(p["counts"][left] + p["counts"][right], p["counts"]["overall"])
    return patients


def check_rows(rows, patients, slices):
    n_checked = 0
    expected_fa = np.mean([p["false"] / p["days"] for p in patients.values()])
    total_false = sum(p["false"] for p in patients.values())
    days = sum(p["days"] for p in patients.values())
    lookup = {(r["slice"], r["lead_min"]): r for r in rows}
    assert len(lookup) == len(slices) * len(LEADS)
    for name in slices:
        supported = [p for p in patients.values() if p["events"][name]]
        events = sum(p["events"][name] for p in supported)
        counts = np.asarray([p["counts"][name] for p in supported])
        denominators = np.asarray([p["events"][name] for p in supported])
        macro = (counts / denominators[:, None]).mean(axis=0)
        for j, lead in enumerate(LEADS):
            row = lookup[name, int(lead)]
            detected = int(counts[:, j].sum())
            assert row["n_events"] == events
            assert row["n_detected"] == detected
            assert row["n_with_events"] == len(supported)
            assert row["n_patients"] == len(patients)
            assert row["n_false_alarms"] == total_false
            for field, expected in (("recall", macro[j]), ("recall_micro", detected / events),
                                     ("test_fa_day", expected_fa), ("patient_days", days),
                                     ("test_fa_day_micro", total_false / days)):
                if not np.isclose(row[field], expected, rtol=1e-12, atol=1e-12):
                    raise AssertionError(f"{name}/{lead}/{field}: saved {row[field]}, fresh {expected}")
            n_checked += 1
    return n_checked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--cohort", choices=COHORTS, required=True)
    args = parser.parse_args()
    out = BASE / "aggregate" / "full_timeline_audit" / f"{args.model}_{args.cohort}.json"
    if out.exists():
        raise RuntimeError("Existing audit record will not be overwritten.")
    groups, slices, support, provenance = load_part(args.model, args.cohort, "test")
    operating_points = []
    n_checked = 0
    computed = {}
    for view, source in (("validation_calibrated", BASE / "aggregate"),
                         ("test_rate_matched", BASE / "aggregate" / "test_rate_matched")):
        folder = source / f"{args.model}_{args.cohort}"
        completion = json.loads((folder / "completion.json").read_text())
        if view == "validation_calibrated":
            assert completion["code_sha256"] == sha(Path(__file__).with_name("evaluate.py"))
            assert completion["metrics_sha256"] == sha(Path(__file__).with_name("warning_metrics.py"))
        else:
            assert completion["code_sha256"] == sha(Path(__file__).with_name("test_rate_matched.py"))
        assert completion["checkpoint_sha256"] == provenance["checkpoint_sha256"]
        assert completion["config_sha256"] == provenance["config_sha256"]
        rows = json.loads((folder / "curves.json").read_text())
        for target in (1., 1.5, 2., 2.5, 3.):
            chosen = [r for r in rows if r["target_fa_day"] == target]
            cutoffs = {r["alarm_cutoff_mgdl"] for r in chosen}
            assert len(cutoffs) == 1
            cutoff = cutoffs.pop()
            if cutoff not in computed:
                computed[cutoff] = recount(groups, slices, cutoff)
            checked = check_rows(chosen, computed[cutoff], slices)
            n_checked += checked
            # A post-bolus onset is ≤30 min after bolus. A matched warning ≥35 min
            # early must therefore precede that bolus, demonstrating that such
            # warnings are actually credited rather than removed by an origin mask.
            pre = next(r for r in chosen if r["slice"] == "post_bolus_30" and r["lead_min"] == 35)
            operating_points.append(dict(view=view, target_fa_day=target,
                alarm_cutoff_mgdl=cutoff, point_rows_checked=checked,
                post_bolus_events=pre["n_events"],
                post_bolus_detections_with_at_least_35min_warning=pre["n_detected"]))
            print(f"PASS {args.model}/{args.cohort}/{view}/{target:g}: {checked} rows", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, dict(model=args.model, cohort=args.cohort, passed=True,
        n_point_rows_checked=n_checked, n_operating_points=len(operating_points),
        independent_full_timeline_recount=True, slice_origin_filter_applied=False,
        artifacts_modified=False, cached_forecasts_provenance_verified=True,
        production_code_hashes_verified=True, operating_points=operating_points,
        audit_code_sha256=sha(__file__)))


if __name__ == "__main__":
    main()
