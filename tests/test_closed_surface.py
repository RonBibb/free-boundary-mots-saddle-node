import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.closed_surface import find_closed_surfaces
from bhps.closed_surface_fd import find_closed_surfaces_fd


class ClosedSurfaceTests(unittest.TestCase):
    def test_tangherlini_sphere(self):
        z=np.linspace(-2.,2.,81);r=np.linspace(0.,3.,81);center=.25;strength=.36
        distance2=r[None,:]**2+(z[:,None]-center)**2
        # Replace the puncture only well inside the analytic horizon so the
        # tensor-product spline is not polluted by one enormous grid value.
        psi=1+strength/np.maximum(distance2,.3**2)
        found=find_closed_surfaces(z,r,psi,centers=(center,),guesses=(.45,.6,.8),tolerance=5e-5)
        self.assertTrue(found["closed_surface_found"])
        surface=found["accepted"][0];expected=np.sqrt(strength)
        self.assertAlmostEqual(surface["radius_max"],expected,delta=3e-2)
        self.assertAlmostEqual(surface["z_lower_tip"],center-expected,delta=3e-2)
        self.assertAlmostEqual(surface["z_upper_tip"],center+expected,delta=3e-2)

    def test_flat_control_has_no_closed_surface(self):
        z=np.linspace(-2.,2.,51);r=np.linspace(0.,3.,61);psi=np.ones((len(z),len(r)))
        found=find_closed_surfaces(z,r,psi,centers=(0.,),guesses=(.4,.8,1.2))
        self.assertFalse(found["closed_surface_found"])

    def test_fd_tangherlini_sphere(self):
        z=np.linspace(-2.,2.,81);r=np.linspace(0.,3.,81);center=.25;strength=.36
        distance2=r[None,:]**2+(z[:,None]-center)**2
        psi=1+strength/np.maximum(distance2,.3**2)
        found=find_closed_surfaces_fd(z,r,psi,centers=(center,),guesses=(.5,.6,.7),nodes=81)
        self.assertTrue(found["closed_surface_found"])
        self.assertAlmostEqual(found["accepted"][0]["radius_max"],np.sqrt(strength),delta=3e-2)


if __name__=="__main__":unittest.main()
