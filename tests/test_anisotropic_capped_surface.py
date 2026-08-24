import sys,unittest
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.anisotropic_capped_surface import _splines,anisotropic_capped_area_stability,anisotropic_rho_second,find_anisotropic_donor_capped_surfaces
from bhps.anisotropic_capped_surface_fd import solve_anisotropic_capped_surface_fd
from bhps.capped_surface import _rho_second


class AnisotropicCappedSurfaceTests(unittest.TestCase):
    def test_equation_reduces_to_conformal_equation(self):
        z=np.linspace(.2,2,49);r=np.linspace(0,3,65)
        psi=np.exp(.08*z[:,None]-.03*r[None,:]**2);zero=np.zeros_like(psi)
        splines=_splines(z,r,psi,zero,zero,zero)
        theta=np.linspace(.05,1.5,37);rho=1+.07*np.cos(2*theta);slope=-.14*np.sin(2*theta)
        radius=rho*np.sin(theta);zcoord=z[-1]-rho*np.cos(theta)
        spline=RectBivariateSpline(z,r,psi,kx=3,ky=3)
        expected=_rho_second(
            theta,rho,slope,spline.ev(zcoord,radius),
            spline.ev(zcoord,radius,dx=0,dy=1),
            spline.ev(zcoord,radius,dx=1,dy=0),
        )
        actual=anisotropic_rho_second(theta,rho,slope,z[-1],splines)
        self.assertLess(np.max(np.abs(actual-expected)),2e-9)

    def test_tangherlini_conformal_control(self):
        z=np.linspace(.2,2,45);r=np.linspace(0,3,65);strength=.36
        distance2=r[None,:]**2+(z[:,None]-z[-1])**2
        psi=1+strength/np.maximum(distance2,1e-8);zero=np.zeros_like(psi)
        result=find_anisotropic_donor_capped_surfaces(
            z,r,psi,zero,zero,zero,guesses=(.45,.6,.8),tolerance=5e-5,
        )
        self.assertTrue(result["capped_surface_found"])
        self.assertAlmostEqual(result["accepted"][0]["rho_brane"],np.sqrt(strength),delta=.03)

    def test_independent_nodal_tangherlini_control(self):
        z=np.linspace(.2,2,49);r=np.linspace(0,3,81);strength=.36
        distance2=r[None,:]**2+(z[:,None]-z[-1])**2
        psi=1+strength/np.maximum(distance2,1e-8);zero=np.zeros_like(psi)
        solved=solve_anisotropic_capped_surface_fd(
            z,r,psi,zero,zero,zero,.57,nodes=61,tolerance=1e-10,
        )
        self.assertTrue(solved["converged"],solved["message"])
        self.assertAlmostEqual(solved["rho_brane"],np.sqrt(strength),delta=.025)
        self.assertLess(solved["discrete_residual_max"],1e-8)
        stability=anisotropic_capped_area_stability(
            z,r,psi,zero,zero,zero,solved,nodes=25,maximum_angular_mode=2,
        )
        self.assertEqual(stability["negative_mode_count"],0)
        self.assertEqual([x["angular_mode"] for x in stability["angular_mode_spectrum"]],[0,1,2])


if __name__=="__main__":unittest.main()
