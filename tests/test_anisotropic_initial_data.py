import sys,unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.anisotropic_initial_data import anisotropic_initial_data_jacobian,anisotropic_initial_data_residual,solve_anisotropic_initial_data
from bhps.finite_wall_solver import solve_finite_wall_slice
from bhps.scalar_pulse import scalar_pulse


class AnisotropicInitialDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solved=solve_finite_wall_slice(.3,nz=13,nr=17,r_max=4.,wall_stiffness=20.,tolerance=1e-9,iterations=80)
        cls.chi,cls.chi_r,cls.chi_z=scalar_pulse(cls.solved["z"],cls.solved["r"],.3)

    def test_reference_is_exact_when_shape_is_zero(self):
        s=self.solved;zero=np.zeros_like(s["q"])
        residual=anisotropic_initial_data_residual(
            s["q"],s["phi"],s["z"],s["r"],zero,zero,zero,s["background"],
            self.chi_r,self.chi_z,s["q"],s["phi"],7,
        )
        # The localized chi source is already present in the reference slice.
        self.assertLess(np.max(np.abs(residual)),2e-8)

    def test_jacobian_matches_directional_difference(self):
        s=self.solved;z,r=s["z"],s["r"];zz,rr=np.meshgrid(z,r,indexing="ij")
        shape=.01*((zz-z[0])/(z[-1]-z[0]))**2*(1-(zz-z[0])/(z[-1]-z[0]))**2*np.exp(-(rr/2)**2)
        a=3*shape;b=-shape;c=-shape
        rng=np.random.default_rng(9);direction=rng.normal(size=2*s["q"].size);direction/=np.linalg.norm(direction)
        jacobian=anisotropic_initial_data_jacobian(
            s["q"],s["phi"],z,r,a,b,c,s["background"],self.chi_r,self.chi_z,s["q"],s["phi"],7,
        )
        step=1e-6;dq=direction[:s["q"].size].reshape(s["q"].shape);dp=direction[s["q"].size:].reshape(s["q"].shape)
        plus=anisotropic_initial_data_residual(s["q"]+step*dq,s["phi"]+step*dp,z,r,a,b,c,s["background"],self.chi_r,self.chi_z,s["q"],s["phi"],7)
        minus=anisotropic_initial_data_residual(s["q"]-step*dq,s["phi"]-step*dp,z,r,a,b,c,s["background"],self.chi_r,self.chi_z,s["q"],s["phi"],7)
        finite=(plus-minus)/(2*step);analytic=jacobian@direction
        self.assertLess(np.linalg.norm(finite-analytic)/np.linalg.norm(finite),3e-6)

    def test_small_fixed_shape_nonlinear_solve_converges(self):
        s=self.solved;z,r=s["z"],s["r"];zz,rr=np.meshgrid(z,r,indexing="ij")
        x=(zz-z[0])/(z[-1]-z[0]);shape=.003*x**2*(1-x)**2*np.exp(-(rr/2)**2)
        result=solve_anisotropic_initial_data(
            z,r,s["q"],s["phi"],3*shape,-shape,-shape,s["background"],
            self.chi_r,self.chi_z,stencil_width=7,tolerance=1e-9,iterations=12,
        )
        self.assertTrue(result["converged"])
        self.assertLess(result["maximum_residual"],1e-9)


if __name__=="__main__":unittest.main()
