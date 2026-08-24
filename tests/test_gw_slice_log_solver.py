import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.gw_slice_log_solver import gw_log_jacobian,gw_log_residual,solve_gw_log_slice
from bhps.scalar_pulse import scalar_pulse


class GWSliceLogSolverTests(unittest.TestCase):
    def test_zero_source_is_exact(self):
        solved=solve_gw_log_slice(0,nz=17,nr=25,tolerance=1e-12)
        self.assertTrue(solved["converged"])
        self.assertEqual(solved["max_abs_residual"],0.)
        self.assertEqual(solved["max_relative_deformation"],0.)

    def test_jacobian_matches_directional_difference(self):
        solved=solve_gw_log_slice(.2,nz=9,nr=13)
        u=solved["u"];z=solved["z"];r=solved["r"]
        _,chi_r,chi_z=scalar_pulse(z,r,.2);direction=np.sin(np.arange(u.size)).reshape(u.shape);step=1e-7
        finite=(
            gw_log_residual(u+step*direction,z,r,solved["background"],chi_r,chi_z)
            -gw_log_residual(u-step*direction,z,r,solved["background"],chi_r,chi_z)
        )/(2*step)
        exact=gw_log_jacobian(u,z,r,solved["background"],chi_r,chi_z)@direction.ravel()
        self.assertLess(np.max(np.abs(finite-exact)),2e-6)


if __name__=="__main__":unittest.main()
