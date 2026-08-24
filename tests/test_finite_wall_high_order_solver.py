import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.finite_wall_high_order_solver import finite_wall_high_order_jacobian,finite_wall_high_order_residual,solve_finite_wall_high_order_slice
from bhps.scalar_pulse import scalar_pulse


class FiniteWallHighOrderSolverTests(unittest.TestCase):
    def test_zero_source_is_exact(self):
        solved=solve_finite_wall_high_order_slice(0,nz=13,nr=17,tolerance=1e-12)
        self.assertTrue(solved["converged"]);self.assertLess(solved["max_abs_residual"],1e-13)

    def test_protocol_125_can_select_seven_point_reference_operators(self):
        solved=solve_finite_wall_high_order_slice(
            0,nz=13,nr=17,tolerance=1e-12,stencil_width=7,
        )
        self.assertTrue(solved["converged"])
        self.assertLess(solved["max_abs_residual"],1e-12)
        self.assertEqual(solved["stencil_width"],7)
        self.assertTrue(solved["discretization"].startswith("7-point"))

    def test_block_jacobian_matches_directional_difference(self):
        solved=solve_finite_wall_high_order_slice(.15,nz=7,nr=9)
        self.assertTrue(solved["converged"]);q,phi,z,r=solved["q"],solved["phi"],solved["z"],solved["r"]
        _,chi_r,chi_z=scalar_pulse(z,r,.15);direction=np.sin(np.arange(2*q.size))
        dq=direction[:q.size].reshape(q.shape);dp=direction[q.size:].reshape(q.shape);step=1e-7
        finite=(
            finite_wall_high_order_residual(q+step*dq,phi+step*dp,z,r,solved["background"],chi_r,chi_z)
            -finite_wall_high_order_residual(q-step*dq,phi-step*dp,z,r,solved["background"],chi_r,chi_z)
        )/(2*step)
        exact=finite_wall_high_order_jacobian(q,phi,z,r,solved["background"],chi_r,chi_z)@direction
        self.assertLess(np.max(np.abs(finite-exact)),3e-5)


if __name__=="__main__":unittest.main()
