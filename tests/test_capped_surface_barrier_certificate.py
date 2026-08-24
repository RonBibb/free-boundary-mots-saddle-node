import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.anisotropic_capped_surface import _splines, anisotropic_rho_second
from bhps.capped_surface_barrier_certificate import (
    BilinearMetricEnclosure,
    Interval,
    barrier_from_values,
    cover_summary,
    initial_parameter_boxes,
    point_barrier_from_splines,
    process_cover_chunk,
)


class CappedSurfaceBarrierCertificateTests(unittest.TestCase):
    def test_interval_contains_sampled_arithmetic(self):
        left = Interval(-0.3, 0.7)
        right = Interval(1.2, 1.8)
        result = (left * right + 2.0) / right
        for x in np.linspace(left.lower, left.upper, 31):
            for y in np.linspace(right.lower, right.upper, 29):
                value = (x * y + 2.0) / y
                self.assertLessEqual(result.lower, value)
                self.assertGreaterEqual(result.upper, value)

    def test_manufactured_root_controls(self):
        x = Interval(0.49, 0.51)
        tangent = (x - 0.5) ** 2
        positive = tangent + 0.01
        sign_changing = x - 0.5
        self.assertLessEqual(tangent.lower, 0.0)
        self.assertGreaterEqual(tangent.upper, 0.0)
        self.assertGreater(positive.lower, 0.0)
        self.assertLessEqual(sign_changing.lower, 0.0)
        self.assertGreaterEqual(sign_changing.upper, 0.0)

    def test_flat_barrier_is_three_pointwise(self):
        values = {
            "A": 1.0, "B": 1.0, "C": 1.0,
            "Az": 0.0, "Ar": 0.0,
            "Bz": 0.0, "Br": 0.0,
            "Cz": 0.0, "Cr": 0.0,
        }
        for theta in np.linspace(0.0, math.pi / 2, 41):
            value = barrier_from_values(0.8, math.sin(theta), math.cos(theta), values)
            self.assertAlmostEqual(value, 3.0, places=13)

    def test_flat_interval_cover_is_complete_and_positive(self):
        metric = BilinearMetricEnclosure.flat()
        queue = initial_parameter_boxes(theta_count=8, rho_count=8)
        certified = []
        nonpositive = []
        unresolved = []
        while queue:
            state = process_cover_chunk(
                metric, queue, certified, nonpositive, unresolved,
                maximum_evaluations=19,
            )
            queue = state["queue"]
            certified = state["certified"]
            nonpositive = state["nonpositive"]
            unresolved = state["unresolved"]
        summary = cover_summary(
            queue, certified, nonpositive, unresolved,
            total_area=(math.pi / 2) * 1.57,
        )
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["nonpositive_box_count"], 0)
        self.assertEqual(summary["unresolved_box_count"], 0)
        self.assertAlmostEqual(summary["certified_area_fraction"], 1.0)
        self.assertGreater(summary["minimum_certified_lower_bound"], 0.0)

    def test_reduced_formula_matches_euler_lagrange_rhs(self):
        z = np.linspace(0.8, 2.8, 41)
        r = np.linspace(0.0, 2.0, 49)
        zz, rr = np.meshgrid(z, r, indexing="ij")
        psi = 1.0 + 0.025 * zz + 0.006 * rr**2
        a = 0.01 * zz - 0.002 * rr**2
        b = -0.008 * zz + 0.003 * rr**2
        c = 0.004 * zz - 0.001 * rr**2
        splines = _splines(z, r, psi, a, b, c)
        theta = np.linspace(0.08, math.pi / 2, 37)
        rho = np.linspace(0.35, 1.35, len(theta))
        reduced = point_barrier_from_splines(theta, rho, z[-1], splines)
        direct = anisotropic_rho_second(
            theta, rho, np.zeros_like(rho), z[-1], splines,
        ) / rho
        self.assertLess(np.max(np.abs(reduced - direct)), 2e-10)

    def test_flat_interval_contains_three(self):
        metric = BilinearMetricEnclosure.flat()
        for theta_lower, theta_upper in ((0.0, 0.01), (0.7, 0.72), (1.55, math.pi / 2)):
            enclosure = metric.barrier_interval(
                theta_lower, theta_upper, 0.4, 0.43,
            )
            self.assertLessEqual(enclosure.lower, 3.0)
            self.assertGreaterEqual(enclosure.upper, 3.0)
            self.assertGreater(enclosure.lower, 0.0)


if __name__ == "__main__":
    unittest.main()
