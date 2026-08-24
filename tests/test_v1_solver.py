import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from bhps.scalar_pulse import scalar_pulse
from bhps.v1_solver import solve_v1,continue_v1,v1_jacobian,v1_residual

class V1SolverTests(unittest.TestCase):
    def test_image_pulse_neumann_boundaries(self):
        z=np.exp(np.linspace(0,1,101));r=np.linspace(0,4,21);_,_,cz=scalar_pulse(z,r,sigma_y=.2)
        self.assertLess(np.max(np.abs(cz[[0,-1],:])),1e-8)
    def test_image_pulse_is_grid_independent_at_shared_points(self):
        z_coarse=np.exp(np.linspace(0,1,11));z_fine=np.exp(np.linspace(0,1,21));r=np.array([0.,1.])
        coarse=scalar_pulse(z_coarse,r,amplitude=.7)[0]
        fine=scalar_pulse(z_fine,r,amplitude=.7)[0]
        self.assertLess(np.max(np.abs(coarse-fine[::2])),1e-14)
    def test_zero_amplitude_is_ads(self):
        s=solve_v1(0,nz=9,nr=13);self.assertTrue(s["converged"]);self.assertEqual(s["energy_dimensionless"],0)
    def test_energy_scales_quadratically_at_fixed_geometry_weak_limit(self):
        a=solve_v1(.01,nz=9,nr=13);b=solve_v1(.02,nz=9,nr=13)
        self.assertTrue(a["converged"] and b["converged"])
        self.assertAlmostEqual(b["energy_dimensionless"]/a["energy_dimensionless"],4,delta=.02)
    def test_exact_jacobian_shape(self):
        s=solve_v1(.01,nz=9,nr=13);z,r=s["z"],s["r"]
        from bhps.scalar_pulse import scalar_pulse
        _,cr,cz=scalar_pulse(z,r,.01)
        self.assertEqual(v1_jacobian(s["psi"],z,r,cr,cz).shape,(117,117))
    def test_asymptotic_radion_boundary_keeps_ads_exact(self):
        solved=solve_v1(0,nz=17,nr=25,outer_boundary="asymptotic_radion")
        self.assertTrue(solved["converged"])
        self.assertLess(solved["max_abs_residual"],1e-12)
    def test_asymptotic_radion_jacobian_matches_difference(self):
        solved=solve_v1(.2,nz=9,nr=13,outer_boundary="asymptotic_radion")
        z,r,psi=solved["z"],solved["r"],solved["psi"]
        _,chi_r,chi_z=scalar_pulse(z,r,.2)
        direction=np.sin(np.arange(psi.size)).reshape(psi.shape);step=1e-7
        finite=(v1_residual(psi+step*direction,z,r,chi_r,chi_z,outer_boundary="asymptotic_radion")-v1_residual(psi-step*direction,z,r,chi_r,chi_z,outer_boundary="asymptotic_radion"))/(2*step)
        exact=v1_jacobian(psi,z,r,chi_r,chi_z,"asymptotic_radion")@direction.ravel()
        self.assertLess(np.max(np.abs(finite-exact)),2e-6)
    def test_continuation_crosses_direct_solve_difficulty(self):
        branch=continue_v1((0,.25,.5,.75,1.0),nz=9,nr=13,tolerance=1e-8)
        self.assertGreaterEqual(branch["accepted"][-1]["amplitude"],.75)

if __name__=="__main__":unittest.main()
