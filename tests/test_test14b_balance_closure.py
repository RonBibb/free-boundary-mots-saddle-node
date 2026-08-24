import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.test14b_balance_closure import (
    analytic_controls,
    finite_difference_weights,
    five_point_history_derivative,
    seam_distribution_controls,
    vaidya_ads5_control,
)


class Test14BBalanceClosureTests(unittest.TestCase):
    def test_irregular_finite_difference_weights_are_polynomial_exact(self):
        nodes = np.array((-0.7, -0.2, 0.0, 0.4, 1.1))
        target = 0.0
        weights = finite_difference_weights(nodes, target)
        for power in range(5):
            values = nodes**power
            expected = 1.0 if power == 1 else 0.0
            self.assertAlmostEqual(float(weights @ values), expected, places=11)

    def test_five_point_history_derivative_including_one_sided_edges(self):
        times = np.linspace(0.0, 1.0, 33)
        values = times**4 - 2.0 * times**3 + 0.5 * times
        expected = 4.0 * times**3 - 6.0 * times**2 + 0.5
        for stride in (1, 2, 4):
            actual = five_point_history_derivative(values, times, stride=stride)
            self.assertLess(float(np.max(np.abs(actual - expected))), 2e-11)

    def test_vaidya_and_stationary_controls(self):
        value = vaidya_ads5_control()
        self.assertTrue(value["analytic_below_1e_10"])
        self.assertTrue(value["sampled_below_0_2_percent"])
        self.assertTrue(value["stationary_below_1e_10"])

    def test_all_three_seam_distributions_converge(self):
        value = seam_distribution_controls()
        self.assertTrue(value["all_smoothed_below_2e_4"])
        self.assertTrue(value["all_omissions_fail_above_1_percent"])
        self.assertEqual(
            set(value["entries"]),
            {"curvature", "normal_connection", "brane_matter"},
        )

    def test_full_control_bundle_passes(self):
        self.assertTrue(analytic_controls()["passed"])


if __name__ == "__main__":
    unittest.main()

