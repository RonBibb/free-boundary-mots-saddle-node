import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.nonlinear_wall_acceleration import (
    scalar_wall_second_corner_fields,
    solve_scalar_wall_accelerations,
)


class NonlinearWallAccelerationTests(unittest.TestCase):
    def test_manufactured_scalar_endpoint_solve_closes_both_walls(self):
        z = np.linspace(1.0, 2.0, 17)
        r = np.linspace(0.0, 2.0, 13)
        zz, rr = np.meshgrid(z, r, indexing="ij")
        dz = derivative_matrix(z, 1, 7)
        A = 1.2 + 0.1 * zz + 0.03 * rr**2
        gzz_tt = 0.2 * np.cos(zz) * np.exp(-0.1 * rr**2)
        phi = 0.1 + 0.02 * zz + 0.01 * rr**2
        phi_tt = np.sin(1.3 * zz) * np.exp(-0.2 * rr**2)
        chi_tt = np.cos(0.7 * zz) * np.exp(-0.15 * rr**2)
        background = {"wall_stiffness": 3.0, "v0": 0.08, "v1": 0.15}
        solved = solve_scalar_wall_accelerations(
            dz, A, gzz_tt, phi, phi_tt, chi_tt, background,
        )
        rows = scalar_wall_second_corner_fields(
            dz, A, gzz_tt, phi, solved["phi_acceleration"],
            solved["chi_acceleration"], background, radial_buffer=0,
        )
        for wall in rows["walls"]:
            self.assertLess(np.max(np.abs(wall["phi_residual"])), 2e-12)
            self.assertLess(np.max(np.abs(wall["chi_residual"])), 2e-12)


if __name__ == "__main__":
    unittest.main()
