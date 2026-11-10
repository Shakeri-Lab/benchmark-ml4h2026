"""Synthetic timing, fitting, recursion, lag-response and no-leakage tests."""
import unittest
import numpy as np
from recursive_arx import Specification, RecursiveARX, MAX_LAG, design_rows, fit_moments


class Tests(unittest.TestCase):
    def test_known_recurrence(self):
        spec = Specification(1, 1, 0, 0.0)
        model = RecursiveARX(spec, [0.5, -2.0, -3.0], 10.0)
        hist = np.array([[100.0, 1.0, 0.0], [80.0, 1.0, 0.0]])
        plan = np.array([[2.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
        got = model.rollout(hist, plan)[0]
        np.testing.assert_allclose(got, [43.0, 29.5, 24.75])

    def test_impulse_propagates(self):
        model = RecursiveARX(Specification(1, 1, 0, 0), [.9, 0, -2], 10)
        hist = np.tile([100, 0, 0], (60, 1))
        zero = np.zeros((48, 2)); impulse = zero.copy(); impulse[0, 1] = 1
        delta = model.rollout(hist, impulse) - model.rollout(hist, zero)
        np.testing.assert_allclose(delta[0], -2 * .9 ** np.arange(48))

    def test_delayed_insulin(self):
        model = RecursiveARX(Specification(1, 3, 0, 0), [.9, 0, 0, 0, 0, 0, -2], 10)
        hist = np.tile([100, 0, 0], (60, 1)); zero = np.zeros((8, 2))
        impulse = zero.copy(); impulse[0, 1] = 1
        delta = model.rollout(hist, impulse) - model.rollout(hist, zero)
        np.testing.assert_allclose(delta[0, :3], [0, 0, -2])

    def test_future_plan_cannot_change_past(self):
        model = RecursiveARX(Specification(1, 1, 0, 0), [.9, -1, -2], 10)
        hist = np.tile([100, 0, 0], (60, 1)); plan = np.zeros((48, 2))
        modified = plan.copy(); modified[10:, :] = 10
        np.testing.assert_array_equal(model.rollout(hist, plan)[:, :10],
                                      model.rollout(hist, modified)[:, :10])

    def test_design_timestamps(self):
        t = np.arange(100, dtype=float)
        x, y = design_rows(np.column_stack((t ** 2, t, -t)), [60])
        self.assertEqual(x[0, 0], 59 ** 2)
        self.assertEqual(x[0, MAX_LAG], 59 ** 2 - 58 ** 2)
        self.assertEqual(x[0, 2 * MAX_LAG], 60)
        self.assertEqual(x[0, 3 * MAX_LAG + 1], -59)
        self.assertEqual(y[0, 1], 60 ** 2 - 59 ** 2)

    def test_fit_and_batch_equivalence(self):
        rng = np.random.default_rng(42); a = rng.normal(size=(800, 3))
        for t in range(1, len(a)):
            a[t, 0] = 7 + .8 * a[t - 1, 0] - 2 * a[t, 1] - 3 * a[t, 2]
        x, y = design_rows(a, np.arange(MAX_LAG + 1, len(a)))
        m = dict(n=len(x), sum_x=x.sum(0), sum_y=y.sum(0), xtx=x.T @ x, xty=x.T @ y)
        fitted = fit_moments(Specification(1, 1, 0, 0), m)
        np.testing.assert_allclose(fitted.coefficients, [.8, -2, -3], atol=1e-9)
        self.assertAlmostEqual(fitted.intercept, 7, places=9)
        h = np.stack([a[:60], a[60:120]])
        p = np.stack([a[60:108, 1:], a[120:168, 1:]])
        np.testing.assert_allclose(fitted.rollout(h, p),
                                  np.concatenate([fitted.rollout(h[i], p[i]) for i in range(2)]))

    def test_differenced_integration(self):
        model = RecursiveARX(Specification(1, 1, 1, 0), [.5, 0, -1], 0)
        hist = np.array([[100, 0, 0], [102, 0, 0]], dtype=float)
        plan = np.array([[0, 1], [0, 0]], dtype=float)
        np.testing.assert_allclose(model.rollout(hist, plan), [[102, 102]])


if __name__ == "__main__":
    unittest.main()
