import unittest
import numpy as np

from matched_metrics import choose_indices, bootstrap_selection, weighted_selected_curves
from warning_metrics import choose_threshold


class MatchedBootstrapTests(unittest.TestCase):
    def test_vectorized_selection_matches_declared_rule(self):
        rng = np.random.default_rng(42)
        fa = np.c_[np.zeros(50), np.round(rng.uniform(0, 4, (50, 40)), 1)]
        auc, rec0 = rng.random(fa.shape), rng.random(fa.shape)
        auc[:, 0] = rec0[:, 0] = 0
        cutoffs = [None] + list(range(40))
        for target in (1, 1.5, 2, 2.5, 3):
            found = choose_indices(fa, auc, rec0, cutoffs, target)
            for b in range(len(fa)):
                rows = [dict(cutoff=c, fa_day=fa[b, i], recall_auc=auc[b, i], recall_0=rec0[b, i])
                        for i, c in enumerate(cutoffs)]
                self.assertEqual(cutoffs[found[b]], choose_threshold(rows, target)["cutoff"])

    def test_bootstrap_reselects_not_freezes_cutoff(self):
        fa = np.array([[0, 0], [1, 3], [2, 4]], dtype=float)
        rec = np.array([[[0, 0], [0, 0]], [[1, .5], [.5, 0]], [[1, .7], [1, .8]]])
        weights = np.array([[2, 0], [0, 2]], dtype=float)
        chosen = bootstrap_selection(fa, rec, [None, 70, 80], [2], [0, 60], weights)[2]
        np.testing.assert_array_equal(chosen, [2, 0])
        curves = weighted_selected_curves(rec, chosen, weights)
        np.testing.assert_array_equal(curves, [[1, .7], [0, 0]])

    def test_slice_denominator_excludes_zero_event_patients(self):
        rec = np.array([[[1, .5], [np.nan, np.nan]], [[.5, 0], [np.nan, np.nan]]])
        got = weighted_selected_curves(rec, np.array([0, 1]), np.ones((2, 2)))
        np.testing.assert_array_equal(got, [[1, .5], [.5, 0]])


if __name__ == "__main__":
    unittest.main(verbosity=2)
