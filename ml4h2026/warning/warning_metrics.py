"""Pure alarm logic shared by synthetic tests and the isolated HPC evaluation."""
from __future__ import annotations

import bisect
import numpy as np

LEADS = np.arange(0, 61, 5, dtype=int)
BUDGETS = (1.0, 1.5, 2.0, 2.5, 3.0)
HORIZON_STEPS = 12
REFRACTORY_STEPS = 12


def event_onsets(cgm, duration=3):
    """Same 3-low-sample onset / 3-recovered-sample reset as submitted protocol."""
    out = []
    below = np.asarray(cgm) < 70
    n = len(below)
    i = 0
    while i < n:
        if not below[i]:
            i += 1
            continue
        start = i
        while i < n and below[i]:
            i += 1
        if i - start < duration:
            continue
        out.append(start)
        recovered = 0
        while i < n:
            recovered = 0 if below[i] else recovered + 1
            i += 1
            if recovered >= duration:
                break
    return out


def contiguous_onsets(cgm, slots):
    bounds = np.r_[0, np.where(np.diff(slots) != 1)[0] + 1, len(slots)]
    return [int(start) + t for start, stop in zip(bounds[:-1], bounds[1:])
            for t in event_onsets(cgm[start:stop]) if t > 0]


def complete_warning_window(origins, onset, horizon=HORIZON_STEPS):
    left, right = np.searchsorted(origins, [onset - horizon, onset])
    return right - left == horizon


def any_warning_window(origins, onset, horizon=HORIZON_STEPS):
    left, right = np.searchsorted(origins, [onset - horizon, onset])
    return right > left


def retain_alarms(origins, scores, cutoff, refractory=REFRACTORY_STEPS):
    if cutoff is None:
        return []
    candidates = origins[scores < cutoff]
    alarms = []
    pos = 0
    while pos < len(candidates):
        alarm = int(candidates[pos])
        alarms.append(alarm)
        pos = int(np.searchsorted(candidates, alarm + refractory))
    return alarms


def match_alarms(alarms, events, horizon=HORIZON_STEPS):
    """Full-timeline one-to-one matching, strictly before onset.

    Matching is performed once. Requiring a longer warning does not reclassify
    an otherwise true (but too-late) alarm as false.
    """
    used = set()
    detected = {}
    for onset in events:
        left = bisect.bisect_left(alarms, onset - horizon)
        right = bisect.bisect_left(alarms, onset)
        for alarm in alarms[left:right]:
            if alarm not in used:
                used.add(alarm)
                detected[int(onset)] = (int(onset) - alarm) * 5
                break
    return detected, used


def lead_counts(onsets, detected, leads=LEADS):
    times = np.asarray([detected.get(int(t), -1) for t in onsets])
    return (times[:, None] >= leads[None, :]).sum(axis=0)


def choose_threshold(rows, budget):
    """Closest validation macro FA/day not exceeding the specified target.

    Ties favor integrated warning recall, then recall at L=0, then lower cutoff.
    Scan all candidates: refractory alarm FA/day is not assumed monotonic.
    """
    feasible = [r for r in rows if r["fa_day"] <= budget + 1e-12]
    if not feasible:
        raise RuntimeError("No feasible threshold, including the no-alarm candidate.")
    def key(row):
        area = row.get("recall_auc", 0.0)
        recall = row.get("recall_0", 0.0)
        return (row["fa_day"], area if np.isfinite(area) else -1,
                recall if np.isfinite(recall) else -1,
                -(row["cutoff"] if row["cutoff"] is not None else -1e100))
    return max(feasible, key=key)


def macro_ci(values, n_boot=2000, seed=42):
    """Patient macro average, excluding patients without denominator support."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    supported = np.isfinite(values).any(axis=1)
    values = values[supported]
    if not len(values):
        empty = np.full(values.shape[1], np.nan)
        return empty, empty.copy(), empty.copy()
    if not np.isfinite(values).all():
        raise ValueError("All coordinates of a supported patient's curve must be finite.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), (n_boot, len(values)))
    # Avoid constructing boot x patients x leads for large cohorts.
    boot = np.stack([values[indices, j].mean(axis=1) for j in range(values.shape[1])], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    return values.mean(axis=0), lo, hi
