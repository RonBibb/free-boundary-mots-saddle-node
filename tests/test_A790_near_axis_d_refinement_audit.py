import unittest

import numpy as np

from analyze_corrected_A790_near_axis_d_refinement import (
    difference_metrics,
    even_polynomial_fit,
    evaluate_fit,
)


class NearAxisDRefinementAuditTests(unittest.TestCase):
    def test_even_polynomial_fit_recovers_axis_and_profile(self):
        r = np.linspace(0.0, 0.8, 17)
        offsets = np.array([1.2, -0.4])
        x = r**2
        values = np.stack((
            offsets[0] + 0.3 * x - 0.2 * x**2 + 0.1 * x**3,
            offsets[1] - 0.7 * x + 0.4 * x**2 - 0.05 * x**3,
        ))
        fit = even_polynomial_fit(values, r, degree=3, window=0.5)
        self.assertTrue(np.allclose(fit["axis"], offsets, rtol=0.0, atol=1e-12))
        predicted = evaluate_fit(fit["coefficients"], np.array([0.0, 0.1, 0.3]))
        expected_x = np.array([0.0, 0.1, 0.3]) ** 2
        expected = np.stack((
            offsets[0] + 0.3 * expected_x - 0.2 * expected_x**2 + 0.1 * expected_x**3,
            offsets[1] - 0.7 * expected_x + 0.4 * expected_x**2 - 0.05 * expected_x**3,
        ))
        self.assertTrue(np.allclose(predicted, expected, rtol=0.0, atol=1e-12))

    def test_difference_metrics_recovers_positive_order(self):
        h = np.array([1.0 / 80.0, 1.0 / 96.0, 1.0 / 112.0])
        exact = np.array([1.0, -2.0, 0.5])
        states = [exact + value**3 * np.array([0.4, -0.2, 0.1]) for value in h]
        metrics = difference_metrics(*states)
        self.assertTrue(metrics["difference_decreases"])
        self.assertAlmostEqual(metrics["generalized_empirical_order"], 3.0, places=8)


if __name__ == "__main__":
    unittest.main()
