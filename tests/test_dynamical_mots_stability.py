import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.dynamical_capped_horizon_bvp import solve_dynamical_capped_surface_bvp
from bhps.dynamical_mots_stability import (
    finite_difference_matrix,
    mots_stability_matrix,
    neumann_extension,
)


def flat_state(z, r):
    q = np.zeros((len(z), len(r), 9))
    q[:, :, 2] = -1.0
    q[:, :, 3] = 1.0
    q[:, :, 6] = 1.0
    return q


class DynamicalMotsStabilityTests(unittest.TestCase):
    def test_neumann_elimination(self):
        theta = np.linspace(1e-4, np.pi / 2, 41)
        derivative = finite_difference_matrix(theta, 1, 7)
        extension = neumann_extension(derivative)
        np.testing.assert_allclose(derivative[0] @ extension, 0.0, atol=2e-13)
        np.testing.assert_allclose(derivative[-1] @ extension, 0.0, atol=2e-13)

    def test_constant_curvature_sphere_principal_spectrum(self):
        z = np.linspace(0, 4, 49)
        r = np.linspace(0, 3, 65)
        q = flat_state(z, r)
        k = 0.8
        v = np.zeros_like(q)
        v[:, :, 3] = -2.0 * k
        v[:, :, 6] = -2.0 * k
        surface = solve_dynamical_capped_surface_bvp(
            q, v, z, r, 1.1, tolerance=1e-8, nodes=101, dense_nodes=501,
        )
        result = mots_stability_matrix(
            q, v, z, r, surface, nodes=81, relative_step=1e-5,
        )
        radius = 1.0 / k
        expected = (-3.0 / radius**2, 5.0 / radius**2)
        measured = [item["real"] for item in result["leading_eigenvalues"][:2]]
        np.testing.assert_allclose(measured, expected, rtol=3e-3, atol=3e-3)
        self.assertLess(abs(result["principal_eigenvalue_imaginary"]), 1e-8)
        self.assertEqual(result["principal_eigenfunction_sign_changes"], 0)

    def test_frechet_matrix_matches_direct_manufactured_direction(self):
        z = np.linspace(0, 4, 49)
        r = np.linspace(0, 3, 65)
        q = flat_state(z, r)
        k = 0.8
        v = np.zeros_like(q)
        v[:, :, 3] = -2.0 * k
        v[:, :, 6] = -2.0 * k
        surface = solve_dynamical_capped_surface_bvp(
            q, v, z, r, 1.1, tolerance=1e-8, nodes=101, dense_nodes=501,
        )
        coarse = mots_stability_matrix(
            q, v, z, r, surface, nodes=49, relative_step=2e-5,
        )
        fine = mots_stability_matrix(
            q, v, z, r, surface, nodes=49, relative_step=1e-5,
        )
        difference = np.linalg.norm(coarse["matrix"] - fine["matrix"])
        scale = max(np.linalg.norm(fine["matrix"]), 1e-300)
        self.assertLess(difference / scale, 2e-6)


if __name__ == "__main__":
    unittest.main()
