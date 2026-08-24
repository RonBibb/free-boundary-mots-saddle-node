import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice
from bhps.dynamical_capped_horizon_bvp import (
    local_outgoing_expansion,
    solve_dynamical_capped_surface_bvp,
)


def flat_state(z, r):
    q = np.zeros((len(z), len(r), 9))
    q[:, :, 2] = -1.0
    q[:, :, 3] = 1.0
    q[:, :, 6] = 1.0
    return q


class DynamicalCappedHorizonBvpTests(unittest.TestCase):
    def test_local_formula_on_flat_half_sphere(self):
        z = np.linspace(0, 4, 49)
        r = np.linspace(0, 3, 65)
        q = flat_state(z, r)
        prepared = prepare_capped_expansion_slice(q, np.zeros_like(q), z, r)
        theta = np.linspace(1e-3, np.pi / 2, 301)
        radius = 1.2
        expansion = local_outgoing_expansion(
            prepared, theta, np.full_like(theta, radius),
            np.zeros_like(theta), np.zeros_like(theta),
        )
        np.testing.assert_allclose(expansion, 3.0 / radius, rtol=2e-10, atol=2e-10)

    def test_local_formula_matches_normal_derivative_evaluator_on_smooth_profile(self):
        from bhps.dynamical_capped_horizon import capped_outgoing_expansion

        z = np.linspace(0, 4, 65)
        r = np.linspace(0, 3, 81)
        q = flat_state(z, r)
        q[:, :, 3] = 0.9
        q[:, :, 6] = 1.3
        v = np.zeros_like(q)
        v[:, :, 3] = -0.12
        v[:, :, 6] = -0.08
        prepared = prepare_capped_expansion_slice(q, v, z, r)
        theta = np.linspace(1e-3, np.pi / 2, 1001)
        rho = 1.25 + 0.04 * np.cos(2.0 * theta)
        slope = -0.08 * np.sin(2.0 * theta)
        second = -0.16 * np.cos(2.0 * theta)
        local = local_outgoing_expansion(prepared, theta, rho, slope, second)
        sampled = capped_outgoing_expansion(
            q, v, z, r,
            {"theta": theta, "rho": rho, "slope": slope}, prepared=prepared,
        )
        keep = sampled["two_cell_interior_mask"]
        np.testing.assert_allclose(
            local[keep], sampled["outgoing_expansion"][keep],
            rtol=2e-5, atol=2e-5,
        )

    def test_bvp_finds_flat_constant_curvature_marginal_sphere(self):
        z = np.linspace(0, 4, 49)
        r = np.linspace(0, 3, 65)
        q = flat_state(z, r)
        k = 0.8
        v = np.zeros_like(q)
        v[:, :, 3] = -2.0 * k
        v[:, :, 6] = -2.0 * k
        result = solve_dynamical_capped_surface_bvp(
            q, v, z, r, 1.1, tolerance=1e-7, nodes=81, dense_nodes=301,
        )
        self.assertTrue(result["converged"], result["message"])
        self.assertAlmostEqual(result["rho_axis"], 1.0 / k, places=6)
        self.assertAlmostEqual(result["rho_brane"], 1.0 / k, places=6)
        self.assertLess(result["local_expansion_interior_maximum"], 1e-8)
        self.assertLess(
            result["primary_evaluator_crosscheck"]["two_cell_interior_maximum"],
            2e-5,
        )


if __name__ == "__main__":
    unittest.main()
