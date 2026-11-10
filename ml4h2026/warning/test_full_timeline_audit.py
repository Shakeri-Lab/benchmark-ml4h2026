"""Independent-matcher equivalence and explicit pre-bolus warning fixtures."""
import unittest

import numpy as np

from audit_full_timeline import independent_match, recount
from warning_metrics import match_alarms, retain_alarms


class IndependentMatchingTests(unittest.TestCase):
    def test_warning_before_bolus_is_credited(self):
        # Alarm at minute 0, bolus at minute 30, hypoglycemia at minute 40.
        # The event is post-bolus, but its 40-minute warning is pre-bolus.
        group = dict(pid=1, origins=np.array([0]), scores=np.array([60.]),
                     matchable=[8], selected=dict(overall=[8], post_bolus_30=[8],
                     not_post_bolus_30=[], nocturnal=[], daytime=[8]), days=1.)
        patients = recount([group], list(group["selected"]), 70.)
        patient = patients[1]
        self.assertEqual(patient["false"], 0)
        self.assertEqual(patient["counts"]["post_bolus_30"][7], 1)  # ≥35 min
        self.assertEqual(patient["counts"]["post_bolus_30"][8], 1)  # ≥40 min
        self.assertEqual(patient["counts"]["post_bolus_30"][9], 0)  # ≥45 min

    def test_same_results_as_production_on_random_timelines(self):
        rng = np.random.default_rng(1042)
        for _ in range(1000):
            origins = np.sort(rng.choice(500, size=300, replace=False))
            scores = rng.normal(80., 20., len(origins))
            events = sorted(rng.choice(500, size=40, replace=False).tolist())
            cutoff = float(rng.uniform(50., 110.))
            actual, false = independent_match(origins, scores, cutoff, events)
            alarms = retain_alarms(origins, scores, cutoff)
            expected, used = match_alarms(alarms, events)
            self.assertEqual(actual, expected)
            self.assertEqual(false, len(alarms) - len(used))

    def test_no_alarm_policy(self):
        self.assertEqual(independent_match(np.array([0]), np.array([50.]), None, [4]), ({}, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
