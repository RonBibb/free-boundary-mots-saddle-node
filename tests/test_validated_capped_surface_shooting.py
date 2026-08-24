import math
import sys
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.corrected_regular_axis_shooting import (
    corrected_regular_axis_initial_state,
)
from bhps.validated_capped_surface_shooting import (
    TensorBicubicIntervalSpline,
    VInterval,
    ValidatedBicubicMetric,
    axis_divergence_source_interval,
    axis_second_interval,
    exact_axis_weighted_ratio,
    interval_cos,
    interval_sin,
    regular_axis_cone,
    regularized_rhs_interval,
)


def flat_metric():
    z = np.linspace(0.8, 3.0, 9)
    r = np.linspace(0.0, 2.0, 11)
    ones = np.ones((len(z), len(r)))
    fields = {
        name: TensorBicubicIntervalSpline.from_scipy(
            RectBivariateSpline(z, r, ones, kx=3, ky=3)
        )
        for name in ("A", "B", "C")
    }
    return z, r, ValidatedBicubicMetric(z[-1], fields)


class ValidatedIntervalPrimitiveTests(unittest.TestCase):
    def test_binary_operations_contain_exact_rationals(self):
        left = VInterval.point(0.1)
        right = VInterval.point(0.3)
        exact_left = Fraction.from_float(0.1)
        exact_right = Fraction.from_float(0.3)
        for interval, exact in (
            (left + right, exact_left + exact_right),
            (left - right, exact_left - exact_right),
            (left * right, exact_left * exact_right),
            (left / right, exact_left / exact_right),
        ):
            value = float(exact)
            self.assertTrue(interval.contains(value), (interval, value))

    def test_trigonometric_intervals_contain_deterministic_points(self):
        box = VInterval(0.17, 1.23)
        sine = interval_sin(box)
        cosine = interval_cos(box)
        for value in np.linspace(box.lower, box.upper, 101):
            self.assertTrue(sine.contains(math.sin(value)))
            self.assertTrue(cosine.contains(math.cos(value)))


class ValidatedSplineTests(unittest.TestCase):
    def test_interval_spline_encloses_scipy_through_second_derivatives(self):
        z = np.linspace(1.0, 2.7, 12)
        r = np.linspace(0.0, 2.0, 14)
        zz, rr = np.meshgrid(z, r, indexing="ij")
        field = 2.0 + 0.2*zz + 0.1*rr + 0.03*zz*rr + 0.01*zz**2
        scipy_spline = RectBivariateSpline(z, r, field, kx=3, ky=3)
        interval_spline = TensorBicubicIntervalSpline.from_scipy(scipy_spline)
        for zvalue, rvalue in ((1.0, 0.0), (1.31, 0.27), (2.7, 2.0)):
            enclosure = interval_spline.evaluate(
                VInterval.point(zvalue), VInterval.point(rvalue), 2,
            )
            for derivative, interval in enclosure.items():
                expected = float(np.asarray(scipy_spline.ev(
                    zvalue, rvalue, dx=derivative[0], dy=derivative[1],
                )).reshape(-1)[0])
                self.assertTrue(interval.contains(expected), (
                    zvalue, rvalue, derivative, expected, interval,
                ))

    def test_box_crossing_knot_contains_dense_scipy_values(self):
        z = np.linspace(1.0, 2.7, 12)
        r = np.linspace(0.0, 2.0, 14)
        zz, rr = np.meshgrid(z, r, indexing="ij")
        field = np.exp(0.1*zz) * (1.0 + 0.03*rr**2)
        scipy_spline = RectBivariateSpline(z, r, field, kx=3, ky=3)
        interval_spline = TensorBicubicIntervalSpline.from_scipy(scipy_spline)
        zbox = VInterval(z[4] - 0.01, z[4] + 0.01)
        rbox = VInterval(r[5] - 0.01, r[5] + 0.01)
        enclosure = interval_spline.evaluate(zbox, rbox, 0)[(0, 0)]
        for zvalue in np.linspace(zbox.lower, zbox.upper, 9):
            for rvalue in np.linspace(rbox.lower, rbox.upper, 9):
                expected = float(scipy_spline.ev(zvalue, rvalue)[0])
                self.assertTrue(enclosure.contains(expected))


class RegularAxisTests(unittest.TestCase):
    def test_exact_axis_weighted_ratio_is_inside_sealed_coarse_bound(self):
        ratio = exact_axis_weighted_ratio(1e-3)
        self.assertGreaterEqual(ratio.lower, 1.0 / 3.0)
        self.assertLessEqual(ratio.upper, 0.334)
        self.assertLess(ratio.width, 1e-14)

    def test_flat_regular_axis_factor_is_one_not_three(self):
        z, r, metric = flat_metric()
        second = axis_second_interval(VInterval.point(0.7), metric)
        self.assertTrue(second.contains(0.7))
        state, audit = regular_axis_cone(
            VInterval.point(0.7), metric, theta_subdivisions=16,
        )
        self.assertTrue(audit["cone"].contains(0.7))
        self.assertGreater(state["slope"].lower, 0.0)

    def test_corrected_floating_initializer_uses_factor_three(self):
        z = np.linspace(0.8, 3.0, 9)
        r = np.linspace(0.0, 2.0, 11)
        ones = np.ones((len(z), len(r)))
        spline = RectBivariateSpline(z, r, ones, kx=3, ky=3)
        state, audit = corrected_regular_axis_initial_state(
            0.7, 1e-3, z[-1], {name: spline for name in ("A", "B", "C")},
        )
        self.assertAlmostEqual(audit["axis_barrier"], 3.0, places=11)
        self.assertAlmostEqual(
            audit["regular_axis_second_derivative"], 0.7, places=11,
        )
        self.assertAlmostEqual(state[1], 0.0007, places=13)

    def test_divergence_source_is_three_rho_in_flat_axis_limit(self):
        _, _, metric = flat_metric()
        rho = VInterval.point(0.7)
        source = axis_divergence_source_interval(
            VInterval.point(0.0), rho, VInterval.point(0.7), metric,
        )
        self.assertTrue(source.contains(2.1), source)

    def test_regularized_flat_zero_slope_point_value_is_three_rho(self):
        _, _, metric = flat_metric()
        result = regularized_rhs_interval(
            VInterval.point(0.4), VInterval.point(0.7),
            VInterval.point(0.0), metric,
        )
        self.assertTrue(result.contains(2.1), result)


if __name__ == "__main__":
    unittest.main()
