import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.finite_archive_null_proxy import (
    ArchivedSpacetime,
    detect_synthetic_caustic,
    finite_terminal_classification,
    initialize_terminal_generators,
    integrate_coordinate_generators,
    integrate_hamiltonian_generators,
    polar_coordinates,
)


def flat_history(times, z, r, shift=0.0):
    position = np.zeros((len(times), len(z), len(r), 9))
    position[:, :, :, 0] = shift
    position[:, :, :, 2] = -1.0 + shift**2
    position[:, :, :, 3] = 1.0
    position[:, :, :, 6] = 1.0
    return position, np.zeros_like(position)


class FiniteArchiveNullProxyTests(unittest.TestCase):
    def setUp(self):
        self.times = np.linspace(0.0, 0.1, 11)
        self.z = np.linspace(0.0, 3.0, 49)
        self.r = np.linspace(0.0, 3.0, 49)
        self.theta = np.linspace(0.08, np.pi / 2.0 - 0.08, 17)

    def spacetime(self, shift=0.0):
        position, velocity = flat_history(
            self.times, self.z, self.r, shift=shift,
        )
        return ArchivedSpacetime(
            self.times, self.z, self.r, position, velocity,
        )

    def profile(self, radius=1.0):
        return {
            "theta": self.theta,
            "rho": np.full_like(self.theta, radius),
            "slope": np.zeros_like(self.theta),
        }

    def test_flat_backward_generators_follow_analytic_outgoing_rays(self):
        spacetime = self.spacetime()
        terminal, velocity = initialize_terminal_generators(
            spacetime, 0.1, self.profile(),
        )
        traced = integrate_coordinate_generators(
            spacetime, 0.1, 0.0, terminal, velocity,
            output_times=np.asarray((0.1, 0.05, 0.0)),
            rtol=1e-11, atol=1e-13,
        )
        angle, rho = polar_coordinates(spacetime.z[-1], traced["positions"])
        np.testing.assert_allclose(
            angle, np.broadcast_to(self.theta, angle.shape), atol=2e-11,
        )
        np.testing.assert_allclose(
            rho[:, 0], np.asarray((1.0, 0.95, 0.9)), atol=2e-10,
        )
        self.assertLess(traced["maximum_normalized_null_residual"], 2e-12)

    def test_constant_shift_analytic_ray_and_hamiltonian_agreement(self):
        spacetime = self.spacetime(shift=0.2)
        terminal, velocity = initialize_terminal_generators(
            spacetime, 0.1, self.profile(),
        )
        expected = terminal - 0.1 * velocity
        coordinate = integrate_coordinate_generators(
            spacetime, 0.1, 0.0, terminal, velocity,
            output_times=np.asarray((0.1, 0.0)), rtol=1e-11, atol=1e-13,
        )
        hamiltonian = integrate_hamiltonian_generators(
            spacetime, 0.1, 0.0, terminal, velocity,
            output_times=np.asarray((0.1, 0.0)), rtol=1e-11, atol=1e-13,
        )
        np.testing.assert_allclose(coordinate["positions"][-1], expected, atol=2e-10)
        np.testing.assert_allclose(
            hamiltonian["positions"], coordinate["positions"], atol=2e-10,
        )

    def test_non_lorentzian_metric_is_rejected(self):
        position, velocity = flat_history(self.times, self.z, self.r)
        position[:, :, :, 2] = 1.0
        spacetime = ArchivedSpacetime(
            self.times, self.z, self.r, position, velocity,
        )
        with self.assertRaises(RuntimeError):
            spacetime.metric_and_derivatives(0.05, np.asarray((1.0,)), np.asarray((1.0,)))

    def test_caustic_and_finite_terminal_classification(self):
        self.assertFalse(detect_synthetic_caustic(np.linspace(0.0, 1.0, 9)))
        self.assertTrue(detect_synthetic_caustic(np.asarray((0.0, 0.4, 0.3, 1.0))))
        profile = self.profile()
        zcoord = self.z[-1] - 1.01 * np.cos(self.theta)
        radius = 1.01 * np.sin(self.theta)
        classified = finite_terminal_classification(
            self.z[-1], np.stack((zcoord, radius), axis=1), profile, 1e-3,
        )
        self.assertEqual(classified["outside_count"], len(self.theta))
        self.assertEqual(classified["inside_count"], 0)


if __name__ == "__main__":
    unittest.main()
