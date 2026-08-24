import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.capped_surface_fd import capped_fd_jacobian_diagnostic,solve_capped_surface_fd


class CappedSurfaceFiniteDifferenceTests(unittest.TestCase):
    def test_manufactured_tangherlini_cap(self):
        z=np.linspace(.2,2,49);r=np.linspace(0,3,81);strength=.36
        distance2=r[None,:]**2+(z[:,None]-z[-1])**2
        psi=1+strength/np.maximum(distance2,1e-8)
        solved=solve_capped_surface_fd(z,r,psi,.57,nodes=61,tolerance=1e-10)
        self.assertTrue(solved["converged"],solved["message"])
        self.assertAlmostEqual(solved["rho_brane"],np.sqrt(strength),delta=.025)
        self.assertLess(solved["discrete_residual_max"],1e-8)
        diagnostic=capped_fd_jacobian_diagnostic(z,r,psi,solved,relative_step=2e-6)
        self.assertGreater(diagnostic["next_singular_value"],diagnostic["smallest_singular_value"])
        self.assertAlmostEqual(diagnostic["null_residual"],diagnostic["smallest_singular_value"],delta=1e-8)


if __name__=="__main__":unittest.main()
