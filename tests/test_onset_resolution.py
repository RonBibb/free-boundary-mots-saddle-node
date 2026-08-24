import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.onset_resolution import (
    endpoint_vector_difference,
    onset_summary,
    persistent_pair_transition,
)


class OnsetResolutionTests(unittest.TestCase):
    def test_persistent_pair_transition(self):
        self.assertEqual(persistent_pair_transition([0, 0, 2, 2]), 2)
        self.assertIsNone(persistent_pair_transition([2, 2]))
        self.assertIsNone(persistent_pair_transition([0, 2, 0]))
        self.assertIsNone(persistent_pair_transition([0, 1, 2]))

    def test_onset_summary_accepts_two_step_spread(self):
        times = [0.000625 + index * 0.0000625 for index in range(6)]
        summary = onset_summary(
            times,
            {"G7": [0, 2, 2, 2, 2, 2],
             "G8": [0, 0, 2, 2, 2, 2],
             "G9": [0, 0, 0, 2, 2, 2]},
            0.0000625,
        )
        self.assertTrue(summary["complete"])
        self.assertTrue(summary["spread_below_two_steps"])
        self.assertTrue(summary["G8_G9_lag_not_worse_than_G7_G8_plus_one_step"])

    def test_onset_summary_rejects_nonpersistent_history(self):
        summary = onset_summary(
            [0.0, 0.1, 0.2],
            {"G7": [0, 2, 0], "G8": [0, 0, 2], "G9": [0, 0, 2]},
            0.1,
        )
        self.assertFalse(summary["complete"])
        self.assertFalse(summary["spread_below_two_steps"])

    def test_endpoint_vector_difference(self):
        self.assertEqual(
            endpoint_vector_difference([[1.0, 2.0], [3.0, 4.0]],
                                       [[1.0, 2.0], [3.0, 4.0]]),
            0.0,
        )
        self.assertIsNone(endpoint_vector_difference([], []))


if __name__ == "__main__":
    unittest.main()
