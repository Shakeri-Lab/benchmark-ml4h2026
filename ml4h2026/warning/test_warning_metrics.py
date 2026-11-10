"""Synthetic tests for the exact alarm functions used in the HPC experiment."""
import ast
from pathlib import Path
import unittest

import numpy as np

from warning_metrics import (LEADS, any_warning_window, choose_threshold,
    complete_warning_window, contiguous_onsets, event_onsets, lead_counts,
    macro_ci, match_alarms, retain_alarms)


class AlarmTests(unittest.TestCase):
    def test_event_duration_and_recovery(self):
        self.assertEqual(event_onsets([90, 60, 60, 90, 60, 60, 60, 90, 90, 60, 60, 60,
                                      90, 90, 90, 60, 60, 60]), [4, 15])

    def test_gap_does_not_join_low_samples(self):
        self.assertEqual(contiguous_onsets(np.array([90, 60, 60, 60, 90]),
                                          np.array([0, 1, 2, 7, 8])), [])

    def test_low_at_segment_start_has_unknown_onset(self):
        self.assertEqual(contiguous_onsets(np.array([60, 60, 60, 90, 90, 90, 60, 60, 60]),
                                          np.arange(9)), [6])

    def test_complete_window_keeps_fixed_event_support(self):
        self.assertTrue(complete_warning_window(np.arange(0, 20), 12))
        self.assertFalse(complete_warning_window(np.array([0, 2, 3, 4]), 12))
        self.assertTrue(any_warning_window(np.array([0, 2, 3, 4]), 12))
        self.assertFalse(any_warning_window(np.array([12]), 12))

    def test_refractory_is_elapsed_time(self):
        origins = np.array([0, 2, 4, 8, 9, 11, 14, 26])
        self.assertEqual(retain_alarms(origins, np.ones(8), 2), [0, 14, 26])
        self.assertEqual(retain_alarms(origins, np.ones(8), None), [])

    def test_matching_is_one_to_one_and_pre_onset(self):
        det, used = match_alarms([0, 12, 24], [12, 17, 24])
        self.assertEqual(det, {12: 60, 17: 25})
        self.assertEqual(used, {0, 12})

    def test_late_true_alarm_not_reclassified_as_false(self):
        det, used = match_alarms([10], [12])
        counts = lead_counts([12, 20], det)
        self.assertEqual(counts.tolist(), [1, 1, 1] + [0] * 10)
        self.assertEqual(len([10]) - len(used), 0)
        self.assertTrue((np.diff(counts) <= 0).all())

    def test_slices_share_one_global_match(self):
        # Execute the actual pure evaluator function without importing HPC dependencies.
        path = Path(__file__).with_name("evaluate.py")
        tree = ast.parse(path.read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "evaluate_cutoff"]
        ns = dict(np=np, LEADS=LEADS, retain_alarms=retain_alarms,
                  match_alarms=match_alarms, lead_counts=lead_counts)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), ns)
        groups = [dict(pid=1, origins=np.array([0, 12]), scores=np.zeros(2),
                       matchable=[4, 16], selected=dict(overall=[4, 16], post=[4], other=[16]), days=1.0)]
        p = ns["evaluate_cutoff"](groups, ["overall", "post", "other"], 70)[1]
        self.assertEqual(p["false_alarms"], 0)
        np.testing.assert_array_equal(p["detected"]["post"]+p["detected"]["other"], p["detected"]["overall"])

    def test_threshold_grid_does_not_assume_monotonic_fa(self):
        rows = [dict(cutoff=None, fa_day=0, recall_auc=0, recall_0=0),
                dict(cutoff=65, fa_day=1.99, recall_auc=.2, recall_0=.6),
                dict(cutoff=70, fa_day=2.3, recall_auc=.4, recall_0=.7),
                dict(cutoff=75, fa_day=1.99, recall_auc=.3, recall_0=.65)]
        self.assertEqual(choose_threshold(rows, 2)["cutoff"], 75)
        self.assertIsNone(choose_threshold(rows, 1)["cutoff"])

    def test_macro_not_pooled_event_weighting(self):
        mean, lo, hi = macro_ci(np.array([[0, 0], [1, .5]]))
        np.testing.assert_array_equal(mean, [.5, .25])
        self.assertTrue((lo <= mean).all() and (hi >= mean).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
