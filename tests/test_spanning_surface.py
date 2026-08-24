import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from bhps.spanning_surface import find_spanning_surfaces,solve_spanning_profile,spanning_maximum_principle_diagnostic
from bhps.spanning_surface_fd import find_spanning_surfaces_fd


class SpanningSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.z=np.linspace(1,2,33);self.r=np.linspace(0,6,121)

    def test_flat_control_has_no_spanning_surface(self):
        psi=np.ones((len(self.z),len(self.r)))
        result=find_spanning_surfaces(self.z,self.r,psi,guesses=(.5,1.,2.,3.))
        self.assertFalse(result["spanning_surface_found"])
        self.assertTrue(spanning_maximum_principle_diagnostic(self.z,self.r,psi)["strict_obstruction_on_sampled_domain"])
        self.assertFalse(find_spanning_surfaces_fd(self.z,self.r,psi,guesses=(.5,1.,2.,3.))["spanning_surface_found"])

    def test_manufactured_constant_radius_surface(self):
        radius=1.5
        psi=np.exp(-self.r[None,:]**2/(3*radius**2))*np.ones((len(self.z),1))
        result=find_spanning_surfaces(self.z,self.r,psi,guesses=(1.,1.5,2.))
        self.assertTrue(result["spanning_surface_found"])
        found=min(result["accepted"],key=lambda item:abs(item["radius_A"]-radius))
        self.assertAlmostEqual(found["radius_A"],radius,delta=2e-3)
        self.assertAlmostEqual(found["radius_B"],radius,delta=2e-3)
        self.assertIn("angular_mode_spectrum",found)
        followed=solve_spanning_profile(self.z,self.r,psi,radius,tolerance=1e-7)
        self.assertTrue(followed["converged"])
        self.assertAlmostEqual(followed["radius_A"],radius,delta=2e-3)
        independent=find_spanning_surfaces_fd(self.z,self.r,psi,guesses=(1.,1.5,2.))
        self.assertTrue(independent["spanning_surface_found"])
        fd=min(independent["accepted"],key=lambda item:abs(item["radius_A"]-radius))
        self.assertAlmostEqual(fd["radius_A"],radius,delta=3e-3)


if __name__=="__main__":unittest.main()
