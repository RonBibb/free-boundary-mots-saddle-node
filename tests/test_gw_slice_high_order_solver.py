import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.gw_slice_high_order_solver import derivative_matrix,gw_high_order_jacobian,gw_high_order_residual,solve_gw_high_order_slice
from bhps.scalar_pulse import scalar_pulse


class GWSliceHighOrderSolverTests(unittest.TestCase):
    def test_derivative_matrices_are_exact_on_quartics(self):
        x=np.linspace(.2,2,11);values=x**4-2*x**3+x
        np.testing.assert_allclose(derivative_matrix(x,1)@values,4*x**3-6*x**2+1,atol=2e-11)
        np.testing.assert_allclose(derivative_matrix(x,2)@values,12*x**2-12*x,atol=2e-10)

    def test_zero_source_is_exact(self):
        solved=solve_gw_high_order_slice(0,nz=17,nr=25,tolerance=1e-12)
        self.assertTrue(solved["converged"]);self.assertLess(solved["max_abs_residual"],1e-13)

    def test_jacobian_matches_directional_difference(self):
        solved=solve_gw_high_order_slice(.2,nz=9,nr=13)
        q=solved["q"];z=solved["z"];r=solved["r"]
        _,chi_r,chi_z=scalar_pulse(z,r,.2);direction=np.sin(np.arange(q.size)).reshape(q.shape);step=1e-7
        finite=(
            gw_high_order_residual(q+step*direction,z,r,solved["background"],chi_r,chi_z)
            -gw_high_order_residual(q-step*direction,z,r,solved["background"],chi_r,chi_z)
        )/(2*step)
        exact=gw_high_order_jacobian(q,z,r,solved["background"],chi_r,chi_z)@direction.ravel()
        self.assertLess(np.max(np.abs(finite-exact)),2e-5)


if __name__=="__main__":unittest.main()
