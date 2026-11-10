import unittest
import numpy as np
from iob_forecaster import (IOBLinearForecaster, features_from_sequence,
                            insulin_on_board, remaining_fraction, valid_origins)


class IOBTests(unittest.TestCase):
    def test_published_impulse(self):
        ages = np.arange(40) * 5.0
        bolus = np.zeros(40); bolus[0] = 3.0
        result = insulin_on_board(np.arange(40), np.zeros(40), bolus)
        np.testing.assert_allclose(result, 3*(1+ages/100)*np.exp(-ages/100))
        self.assertEqual(result[0], 3)
        self.assertTrue((np.diff(result) < 0).all())

    def test_basal_unit_conversion(self):
        self.assertAlmostEqual(insulin_on_board([0], [1.2], [0])[0], .1)

    def test_superposition(self):
        doses = np.array([1., 0., 2., 0.])
        result = insulin_on_board(np.arange(4), np.ones(4), doses)
        expected = [sum((doses[j]+5/60)*remaining_fraction((i-j)*5)
                        for j in range(i+1)) for i in range(4)]
        np.testing.assert_allclose(result, expected)

    def test_gap_reset(self):
        f = features_from_sequence([100, 105, 120, 125], [0, 1, 10, 11],
                                   [0]*4, [2, 0, 0, 0])
        self.assertEqual(f[2, 2], 0)
        self.assertTrue(np.isnan(f[2, 1]))
        self.assertEqual(f[3, 1], 1)

    def test_history_truncation(self):
        self.assertEqual(insulin_on_board(np.arange(5), np.zeros(5), [1,0,0,0,0], history_steps=3)[3], 0)

    def test_causal_prefix_invariance(self):
        rng = np.random.default_rng(42)
        cgm, basal, bolus = rng.normal(140, 10, 40), rng.uniform(0, 2, 40), rng.uniform(0, 2, 40)
        before = features_from_sequence(cgm, np.arange(40), basal, bolus)[:20]
        cgm[20:] = 400; basal[20:] = 100; bolus[20:] = 100
        after = features_from_sequence(cgm, np.arange(40), basal, bolus)[:20]
        np.testing.assert_allclose(before, after)

    def test_origins_exclude_gaps(self):
        slots = np.r_[np.arange(20), np.arange(30, 50)]
        origins = valid_origins(slots, history_steps=5, horizon_steps=3)
        for idx in origins:
            self.assertTrue((np.diff(slots[idx-4:idx+4]) == 1).all())

    def test_short_sequence(self):
        self.assertEqual(len(valid_origins(np.arange(10))), 0)

    def test_glucose_trajectory_fit_and_serialization(self):
        rng = np.random.default_rng(42)
        x = np.column_stack((rng.uniform(70, 250, 500), rng.normal(size=500), rng.uniform(0, 5, 500)))
        coef = rng.normal(size=(4, 12))
        y = x[:, :1] + np.column_stack((np.ones(len(x)), x)) @ coef
        model = IOBLinearForecaster.fit(x, y)
        self.assertEqual(model.predict(x).shape, (500, 12))
        np.testing.assert_allclose(model.predict(x), y, atol=1e-8)
        restored = IOBLinearForecaster.from_dict(model.to_dict())
        np.testing.assert_array_equal(restored.predict(x), model.predict(x))

    def test_train_only_scaling(self):
        x = np.array([[100, 1, 2], [120, 2, 3], [130, 3, 4]], dtype=float)
        model = IOBLinearForecaster.fit(x, np.tile(x[:, :1], (1, 12)), .01)
        original = model.mean.copy()
        model.predict([[400, 20, 100]])
        np.testing.assert_array_equal(original, x.mean(axis=0))
        np.testing.assert_array_equal(model.mean, original)

    def test_constant_features(self):
        model = IOBLinearForecaster.fit(np.tile([120, 0, 0], (1, 1)), np.full((1, 12), 125.0))
        np.testing.assert_allclose(model.predict([[120, 0, 0]]), 125)

    def test_invalid_inputs_fail(self):
        for basal, bolus in [([np.nan], [0]), ([-1], [0]), ([0], [-1])]:
            with self.assertRaises(ValueError):
                insulin_on_board([0], basal, bolus)
        with self.assertRaises(ValueError):
            remaining_fraction([-1])


if __name__ == "__main__":
    unittest.main()
