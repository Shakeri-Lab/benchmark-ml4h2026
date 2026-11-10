from __future__ import annotations

import numpy as np

# Ensure repo root on path
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from analysis.evaluate_hypo_gating import (  # noqa: E402
    build_alarm_indices,
    extract_event_onsets,
    match_alarms_to_events,
)


def main() -> None:
    # Synthetic sequence: 20 samples @5min
    # Events: one detected, one missed, with recovery separation.
    cgm = np.array(
        [
            100, 100, 100, 100, 100,
            65,  60,  60,  72,  75,  # event 1: onset at idx=5 (3 samples <70)
            80,  85,  90,  95,  100,
            65,  60,  60,  72,  75,  # event 2: onset at idx=15 (missed)
        ],
        dtype=float,
    )
    threshold = 70.0
    min_samples = 3
    pred_len = 3  # PH=15min

    events = extract_event_onsets(cgm, threshold, min_samples)
    assert events == [5, 15], f"Unexpected events: {events}"

    # Predictions: alarm for event 1 at idx=3 (within window [2,5])
    # A false alarm at idx=10 (no event in [7,10])
    # Alarm for event 2 suppressed due to refractory from idx=10
    pred_by_idx = {
        3: np.array([80, 60, 80]),   # triggers alarm
        10: np.array([60, 80, 80]),  # triggers false alarm
        12: np.array([60, 80, 80]),  # would trigger but suppressed by refractory (PH=3)
    }

    alarms = build_alarm_indices(pred_by_idx, threshold, pred_len)
    assert alarms == [3, 10], f"Unexpected alarms: {alarms}"

    detected, lead_times, false_alarms = match_alarms_to_events(
        alarms, events, pred_len, sorted(pred_by_idx.keys())
    )
    assert detected == 1, f"Expected 1 detected event, got {detected}"
    assert lead_times == [2], f"Expected lead time 2 samples, got {lead_times}"
    assert false_alarms == 1, f"Expected 1 false alarm, got {false_alarms}"

    print("Hypo gating sanity check passed.")


if __name__ == "__main__":
    main()
