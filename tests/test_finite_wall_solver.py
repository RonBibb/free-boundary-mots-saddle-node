import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.finite_wall_solver import finite_wall_jacobian,finite_wall_residual,solve_finite_wall_slice
from bhps.scalar_pulse import scalar_pulse


class FiniteWallSolverTests(unittest.TestCase):
    def test_zero_source_is_exact_well_balanced_background(self):
        solved=solve_finite_wall_slice(0,nz=13,nr=17,wall_stiffness=20.,tolerance=1e-11)
        self.assertTrue(solved["converged"])
        self.assertLess(solved["max_abs_residual"],1e-12)
        self.assertLess(solved["max_relative_metric_deformation"],1e-12)
        self.assertLess(solved["max_stabilizer_deformation"],1e-12)

    def test_block_jacobian_matches_directional_difference(self):
        solved=solve_finite_wall_slice(.15,nz=7,nr=9,wall_stiffness=20.)
        self.assertTrue(solved["converged"])
        q,phi,z,r=solved["q"],solved["phi"],solved["z"],solved["r"]
        _,chi_r,chi_z=scalar_pulse(z,r,.15)
        direction=np.sin(np.arange(2*q.size));dq=direction[:q.size].reshape(q.shape);dp=direction[q.size:].reshape(q.shape)
        step=1e-7
        finite=(
            finite_wall_residual(q+step*dq,phi+step*dp,z,r,solved["background"],chi_r,chi_z)
            -finite_wall_residual(q-step*dq,phi-step*dp,z,r,solved["background"],chi_r,chi_z)
        )/(2*step)
        exact=finite_wall_jacobian(q,phi,z,r,solved["background"],chi_r,chi_z)@direction
        self.assertLess(np.max(np.abs(finite-exact)),3e-6)

    def test_sourced_selector_is_admissible_nonstationary_data(self):
        solved=solve_finite_wall_slice(
            .15,nz=9,nr=13,wall_stiffness=20.,stabilizer_forcing_amplitude=.1,
        )
        self.assertTrue(solved["converged"])
        self.assertGreater(solved["stabilizer_forcing_max"],0.)
        self.assertLess(solved["metric_junction_residual_max"],1e-9)
        self.assertLess(solved["scalar_wall_residual_max"],1e-9)

    def test_wall_flat_source_selector_converges(self):
        solved=solve_finite_wall_slice(
            .15,nz=9,nr=13,wall_stiffness=20.,stabilizer_forcing_amplitude=.1,
            stabilizer_forcing_profile="sin_squared",
        )
        self.assertTrue(solved["converged"])
        self.assertEqual(solved["stabilizer_forcing_profile"],"sin_squared")


if __name__=="__main__":unittest.main()
