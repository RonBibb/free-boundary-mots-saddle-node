import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.corrected_A790_test10e_high_z import (
    classify_test10e,
    manufactured_controls,
    scheme_spread,
    source_sequence,
    spread_gate,
    stage_history,
)


class Test10EHighZPredicates(unittest.TestCase):
    def test_manufactured_controls_pass(self):
        self.assertTrue(manufactured_controls()["passed"])

    def test_known_second_order_sequence(self):
        counts = np.asarray((80.0, 96.0, 112.0, 128.0))
        record = source_sequence("proper_ratio", 0.04 + 13.0 / counts**2)
        self.assertAlmostEqual(record["orders"]["fine_triplet_order"], 2.0, places=10)
        self.assertTrue(record["passes"])

    def test_nonmonotone_sequence_fails(self):
        record = source_sequence("proper_ratio", (0.04, 0.041, 0.0405, 0.043))
        self.assertFalse(record["fine_monotonic"])
        self.assertFalse(record["passes"])

    def test_stage_history_requires_ninety_percent_and_no_adverse(self):
        values = np.tile(np.asarray([[0.10], [0.08], [0.07], [0.065]]), (1, 10))
        self.assertTrue(stage_history("proper_ratio", values)["passes"])
        values[3, 0] = 0.12
        record = stage_history("proper_ratio", values)
        self.assertEqual(record["adverse_stage_count"], 1)
        self.assertFalse(record["passes"])

    def test_spread_and_classification_priorities(self):
        self.assertAlmostEqual(scheme_spread([1.0, 2.0], [1.01, 2.0]), 0.01 / 1.01)
        self.assertTrue(spread_gate(0.015, 0.01505)["passes"])
        self.assertFalse(spread_gate(0.015, 0.021)["passes"])
        self.assertEqual(
            classify_test10e(False, True, True, True, True, True, True, False),
            ("review", "invalid_high_z_boundary_audit"),
        )
        self.assertEqual(
            classify_test10e(True, True, False, True, True, True, True, False),
            ("fail", "uncontrolled_high_z_boundary_response"),
        )
        self.assertEqual(
            classify_test10e(True, False, False, True, True, True, True, True),
            ("pass", "high_z_converged_legacy_normalization_artifact"),
        )


if __name__ == "__main__":
    unittest.main()
