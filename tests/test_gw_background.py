import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from bhps.gw_background import solve_gw_background
from bhps.initial_data import make_grid
from bhps.stabilized_solver import solve_stabilized,stabilized_jacobian,stabilized_residual


class GWBackgroundTests(unittest.TestCase):
    def test_zero_backreaction_recovers_ads(self):
        z,_=make_grid(nz=25,nr=17)
        result=solve_gw_background(z,backreaction=0)
        self.assertTrue(result["converged"])
        self.assertLess(np.max(np.abs(result["psi"]-1/z)),1e-10)
        self.assertAlmostEqual(result["beta_b"],1,places=9)

    def test_weak_background_reports_tension_retuning(self):
        z,_=make_grid(nz=25,nr=17)
        result=solve_gw_background(z,epsilon=.1,backreaction=.01)
        self.assertTrue(result["converged"])
        self.assertLess(result["boundary_residual_max"],1e-10)
        self.assertGreater(result["beta_b"],1)
        self.assertLess(result["max_ads_relative_deformation"],.01)

    def test_background_is_exact_discrete_control(self):
        solved=solve_stabilized(0,nz=17,nr=25,backreaction=.01)
        self.assertTrue(solved["converged"])
        self.assertLess(solved["max_abs_residual"],1e-12)
        self.assertLess(solved["max_relative_deformation"],1e-12)

    def test_stabilized_jacobian_matches_directional_difference(self):
        solved=solve_stabilized(.2,nz=9,nr=13,backreaction=.01)
        psi=solved["psi"];z=solved["z"];r=solved["r"]
        from bhps.scalar_pulse import scalar_pulse
        _,chi_r,chi_z=scalar_pulse(z,r,.2)
        direction=np.sin(np.arange(psi.size)).reshape(psi.shape)
        step=1e-7
        finite=(stabilized_residual(psi+step*direction,z,r,solved["background"],chi_r,chi_z)-stabilized_residual(psi-step*direction,z,r,solved["background"],chi_r,chi_z))/(2*step)
        exact=stabilized_jacobian(psi,z,r,solved["background"],chi_r,chi_z)@direction.ravel()
        self.assertLess(np.max(np.abs(finite-exact)),2e-6)

    def test_finite_wall_background_satisfies_half_robin_conditions(self):
        z,_=make_grid(nz=25,nr=17)
        result=solve_gw_background(z,epsilon=.1,backreaction=.01,wall_stiffness=20.)
        self.assertTrue(result["converged"])
        self.assertLess(result["boundary_residual_max"],1e-9)
        self.assertEqual(result["wall_stiffness"],20.)


if __name__=="__main__":unittest.main()
