import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice
from bhps.dynamical_capped_horizon_bvp import local_outgoing_expansion
from bhps.dynamical_multitopology import (
    closed_local_expansion,
    solve_dynamical_closed_surface_bvp,
    solve_dynamical_closed_surface_fd,
    solve_dynamical_spanning_surface_bvp,
    solve_dynamical_spanning_surface_fd,
    spanning_local_expansion,
)


def flat_state(z, r):
    position = np.zeros((len(z), len(r), 9))
    position[:, :, 2] = -1.0
    position[:, :, 3] = 1.0
    position[:, :, 6] = 1.0
    return position


class DynamicalMultitopologyTests(unittest.TestCase):
    def setUp(self):
        self.z = np.linspace(0.0, 4.0, 65)
        self.r = np.linspace(0.0, 4.0, 81)
        self.position = flat_state(self.z, self.r)
        self.zero = np.zeros_like(self.position)

    def test_closed_local_formula_matches_cap_special_case(self):
        prepared = prepare_capped_expansion_slice(
            self.position, self.zero, self.z, self.r,
        )
        theta = np.linspace(1e-3, np.pi / 2.0, 401)
        rho = 1.2 + 0.03 * np.cos(2.0 * theta)
        slope = -0.06 * np.sin(2.0 * theta)
        second = -0.12 * np.cos(2.0 * theta)
        closed = closed_local_expansion(
            prepared, self.z[-1], theta, rho, slope, second,
        )
        capped = local_outgoing_expansion(
            prepared, theta, rho, slope, second,
        )
        np.testing.assert_allclose(closed, capped, rtol=2e-12, atol=2e-12)

    def test_flat_closed_sphere_has_three_over_radius_expansion(self):
        prepared = prepare_capped_expansion_slice(
            self.position, self.zero, self.z, self.r,
        )
        theta = np.linspace(1e-3, np.pi - 1e-3, 401)
        radius = 0.8
        expansion = closed_local_expansion(
            prepared, 2.0, theta, np.full_like(theta, radius),
            np.zeros_like(theta), np.zeros_like(theta),
        )
        np.testing.assert_allclose(expansion, 3.0 / radius, rtol=2e-10, atol=2e-10)

    def test_closed_bvp_and_fd_recover_constant_curvature_sphere(self):
        k = 0.8
        velocity = np.zeros_like(self.position)
        velocity[:, :, 3] = -2.0 * k
        velocity[:, :, 6] = -2.0 * k
        target = 1.0 / k
        bvp = solve_dynamical_closed_surface_bvp(
            self.position, velocity, self.z, self.r, 2.0, 1.1,
            tolerance=1e-7, nodes=81, dense_nodes=401,
        )
        self.assertTrue(bvp["converged"], bvp["message"])
        self.assertAlmostEqual(bvp["radius_max"], target, places=5)
        self.assertLess(bvp["independent_expansion_interior_maximum"], 2e-4)
        fd = solve_dynamical_closed_surface_fd(
            self.position, velocity, self.z, self.r, 2.0, bvp,
            nodes=81, tolerance=1e-9,
        )
        self.assertTrue(fd["converged"], fd["message"])
        self.assertAlmostEqual(fd["radius_max"], target, places=4)

    def test_flat_spanning_cylinder_has_two_over_radius_expansion(self):
        prepared = prepare_capped_expansion_slice(
            self.position, self.zero, self.z, self.r,
        )
        radius = 1.25
        compact = np.linspace(self.z[0], self.z[-1], 301)
        expansion = spanning_local_expansion(
            prepared, compact, np.full_like(compact, radius),
            np.zeros_like(compact), np.zeros_like(compact),
        )
        np.testing.assert_allclose(expansion, 2.0 / radius, rtol=2e-10, atol=2e-10)

    def test_spanning_bvp_and_fd_recover_constant_curvature_cylinder(self):
        k = 0.4
        velocity = np.zeros_like(self.position)
        velocity[:, :, 3] = -2.0 * k
        velocity[:, :, 6] = -2.0 * k
        target = 2.0 / (3.0 * k)
        bvp = solve_dynamical_spanning_surface_bvp(
            self.position, velocity, self.z, self.r, 1.4,
            tolerance=1e-7, nodes=81, dense_nodes=401,
        )
        self.assertTrue(bvp["converged"], bvp["message"])
        self.assertAlmostEqual(bvp["radius_A"], target, places=6)
        self.assertLess(bvp["independent_expansion_interior_maximum"], 2e-5)
        fd = solve_dynamical_spanning_surface_fd(
            self.position, velocity, self.z, self.r, bvp,
            nodes=81, tolerance=1e-9,
        )
        self.assertTrue(fd["converged"], fd["message"])
        self.assertAlmostEqual(fd["radius_A"], target, places=5)


if __name__ == "__main__":
    unittest.main()
