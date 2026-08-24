import math
import sys
import unittest
from pathlib import Path

import mpmath
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RectBivariateSpline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.anisotropic_capped_surface import anisotropic_rho_second
from bhps.correlated_validated_shooting import (
    ArchivedDop853Reference,
    HermiteReferenceStep,
    hermite_defect_bound,
    regular_axis_affine_error,
    validated_matrix_exponential,
)
from bhps.validated_capped_surface_shooting import (
    TensorBicubicIntervalSpline,
    VInterval,
    ValidatedBicubicMetric,
    axis_divergence_source_interval,
    axis_divergence_source_correlated_mean_value,
    axis_divergence_source_mean_value,
    regularized_divergence_rhs_interval,
    regular_axis_cone,
)


def flat_metrics():
    z = np.linspace(0.8, 3.0, 9)
    r = np.linspace(0.0, 2.0, 11)
    ones = np.ones((len(z), len(r)))
    scipy_fields = {
        name: RectBivariateSpline(z, r, ones, kx=3, ky=3)
        for name in ("A", "B", "C")
    }
    fields = {
        name: TensorBicubicIntervalSpline.from_scipy(spline)
        for name, spline in scipy_fields.items()
    }
    return scipy_fields, ValidatedBicubicMetric(z[-1], fields)


class MatrixExponentialTests(unittest.TestCase):
    def test_interval_exponential_contains_high_precision_values(self):
        matrices = (
            np.asarray([[-2.0, 0.3], [0.1, -0.7]]),
            np.asarray([[1.4, -0.2], [0.5, 0.8]]),
            np.asarray([[0.2, 2.0], [0.0, -0.4]]),
            np.asarray([[0.1, 1.0], [1e-12, 0.1]]),
        )
        mpmath.mp.dps = 100
        for matrix in matrices:
            enclosure, tail = validated_matrix_exponential(matrix, 1e-3, 28)
            expected = mpmath.expm(mpmath.matrix((matrix * 1e-3).tolist()))
            self.assertLessEqual(tail, 1e-18)
            for row in range(2):
                for column in range(2):
                    self.assertTrue(
                        enclosure[row, column].contains(
                            float(expected[row, column])
                        )
                    )


class ReferenceAndDefectTests(unittest.TestCase):
    def test_archived_dop853_dense_output_is_bitwise_identical(self):
        solved = solve_ivp(
            lambda theta, state: np.asarray([state[1], -state[0]]),
            (0.0, 1.0), np.asarray([1.0, 0.0]), method="DOP853",
            rtol=1e-12, atol=1e-14, max_step=0.1, dense_output=True,
        )
        archived = ArchivedDop853Reference.from_solve(solved)
        reconstructed = ArchivedDop853Reference(
            **archived.archive_payload()
        )
        for theta in np.linspace(0.0, 1.0, 101):
            self.assertTrue(np.array_equal(
                reconstructed.value(theta), solved.sol(theta),
            ))

    def test_flat_hermite_defect_bound_contains_dense_point_defects(self):
        scipy_fields, metric = flat_metrics()
        theta = 0.4
        step = 1e-3
        left = np.asarray([0.73, 0.08])
        right = np.asarray([0.7300802, 0.0804])

        def rhs(time, state):
            second = anisotropic_rho_second(
                np.asarray([time]), np.asarray([state[0]]),
                np.asarray([state[1]]), metric.z_brane, scipy_fields,
            )[0]
            return np.asarray([state[1], second])

        reference = HermiteReferenceStep.from_endpoints(
            theta, step, left, right, rhs(theta, left),
            rhs(theta + step, right),
        )
        bounds, _ = hermite_defect_bound(reference, metric, subdivisions=32)
        for fraction in np.linspace(0.0, 1.0, 257):
            time = theta + fraction * step
            value = reference.value(fraction)
            derivative = reference.derivative(fraction)
            defect = derivative - rhs(time, value)
            self.assertLessEqual(abs(defect[0]), bounds[0])
            self.assertLessEqual(abs(defect[1]), bounds[1])

    def test_divergence_rhs_matches_exact_slope_coordinate_transform(self):
        scipy_fields, metric = flat_metrics()
        theta = 0.4
        rho = 0.73
        slope = 0.08
        sine = math.sin(theta)
        momentum = sine**2 * slope
        enclosure = regularized_divergence_rhs_interval(
            VInterval.point(theta), VInterval.point(rho),
            VInterval.point(momentum), metric,
        )
        second = anisotropic_rho_second(
            np.asarray([theta]), np.asarray([rho]), np.asarray([slope]),
            metric.z_brane, scipy_fields,
        )[0]
        expected = (
            slope,
            sine**2 * (second + 2.0 * math.cos(theta) * slope / sine),
        )
        self.assertTrue(enclosure[0].contains(expected[0]))
        self.assertTrue(enclosure[1].contains(expected[1]))

    def test_divergence_hermite_defect_contains_dense_point_defects(self):
        scipy_fields, metric = flat_metrics()
        theta = 0.4
        step = 1e-3
        left = np.asarray([0.73, math.sin(theta)**2 * 0.08])
        right = np.asarray([0.7300802, math.sin(theta + step)**2 * 0.0804])

        def rhs(time, state):
            sine = math.sin(time)
            slope = state[1] / sine**2
            second = anisotropic_rho_second(
                np.asarray([time]), np.asarray([state[0]]),
                np.asarray([slope]), metric.z_brane, scipy_fields,
            )[0]
            return np.asarray([
                slope,
                sine**2 * (second + 2.0 * math.cos(time) * slope / sine),
            ])

        reference = HermiteReferenceStep.from_endpoints(
            theta, step, left, right, rhs(theta, left),
            rhs(theta + step, right),
        )
        bounds, _ = hermite_defect_bound(
            reference, metric, subdivisions=32,
            coordinate_system="divergence",
        )
        for fraction in np.linspace(0.0, 1.0, 257):
            time = theta + fraction * step
            value = reference.value(fraction)
            derivative = reference.derivative(fraction)
            defect = derivative - rhs(time, value)
            self.assertLessEqual(abs(defect[0]), bounds[0])
            self.assertLessEqual(abs(defect[1]), bounds[1])

    def test_axis_affine_generators_enclose_exact_axis_box(self):
        _, metric = flat_metrics()
        launch = VInterval(0.6999, 0.7001)
        axis_state, axis_audit = regular_axis_cone(
            launch, metric, theta_subdivisions=128,
        )
        reference_initial = np.asarray([0.7, 0.0007])
        error = regular_axis_affine_error(
            launch, axis_state, axis_audit, reference_initial, 1e-3,
        )
        intervals = error.intervals()
        self.assertEqual(error.generators.shape, (2, 2))
        self.assertTrue(intervals[0].contains(
            axis_state["rho"] - reference_initial[0]
        ))
        self.assertTrue(intervals[1].contains(
            axis_state["slope"] - reference_initial[1]
        ))

    def test_axis_source_mean_value_contains_dense_point_enclosures(self):
        _, metric = flat_metrics()
        theta = VInterval(2e-4, 2.5e-4)
        rho = VInterval(0.6999, 0.7001)
        axis_u = VInterval(0.68, 0.72)
        enclosure = axis_divergence_source_mean_value(
            theta, rho, axis_u, metric,
        )
        for theta_value in np.linspace(theta.lower, theta.upper, 5):
            for rho_value in np.linspace(rho.lower, rho.upper, 5):
                for u_value in np.linspace(axis_u.lower, axis_u.upper, 5):
                    point = axis_divergence_source_interval(
                        VInterval.point(theta_value),
                        VInterval.point(rho_value),
                        VInterval.point(u_value), metric,
                    )
                    self.assertTrue(enclosure.contains(point))

    def test_correlated_axis_source_contains_dense_regular_cone_points(self):
        _, metric = flat_metrics()
        theta = VInterval(2e-4, 2.5e-4)
        launch = VInterval(0.6999, 0.7001)
        axis_u = VInterval(0.68, 0.72)
        enclosure = axis_divergence_source_correlated_mean_value(
            theta, launch, axis_u, metric,
        )
        for theta_value in np.linspace(theta.lower, theta.upper, 5):
            for launch_value in np.linspace(launch.lower, launch.upper, 5):
                for u_value in np.linspace(axis_u.lower, axis_u.upper, 5):
                    rho_value = (
                        launch_value + (1.0 - math.cos(theta_value)) * u_value
                    )
                    point = axis_divergence_source_interval(
                        VInterval.point(theta_value),
                        VInterval.point(rho_value),
                        VInterval.point(u_value), metric,
                    )
                    self.assertTrue(enclosure.contains(point))


if __name__ == "__main__":
    unittest.main()
