import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.gw_slice_solver import gw_slice_jacobian,gw_slice_residual,solve_gw_slice
from bhps.scalar_pulse import scalar_pulse


class GWSliceSolverTests(unittest.TestCase):
    def test_zero_source_is_exact_well_balanced_background(self):
        solved=solve_gw_slice(0,nz=17,nr=25,backreaction=.01,tolerance=1e-11)
        self.assertTrue(solved["converged"])
        self.assertLess(solved["max_abs_residual"],1e-13)
        self.assertLess(solved["max_relative_deformation"],1e-13)
        self.assertEqual(solved["energy_dimensionless"],0.)
        self.assertEqual(solved["energy_quadrature_relative_difference"],0.)
        self.assertLessEqual(solved["residual_l2"],solved["max_abs_residual"])

    def test_jacobian_matches_directional_difference(self):
        solved=solve_gw_slice(.2,nz=9,nr=13,backreaction=.01)
        q=solved["q"];z=solved["z"];r=solved["r"]
        _,chi_r,chi_z=scalar_pulse(z,r,.2)
        direction=np.sin(np.arange(q.size)).reshape(q.shape);step=1e-7
        finite=(
            gw_slice_residual(q+step*direction,z,r,solved["background"],chi_r,chi_z)
            -gw_slice_residual(q-step*direction,z,r,solved["background"],chi_r,chi_z)
        )/(2*step)
        exact=gw_slice_jacobian(q,z,r,solved["background"],chi_r,chi_z)@direction.ravel()
        self.assertLess(np.max(np.abs(finite-exact)),2e-6)


if __name__=="__main__":unittest.main()
