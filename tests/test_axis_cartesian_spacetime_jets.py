import sys,unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.axis_cartesian_spacetime_jets import construct_time_symmetric_axis_spacetime_jets
from bhps.linearized_gh_einstein_scalar import (
    metric_geometry_from_jets,reduced_einstein_two_scalar_residual,
)


class AxisCartesianSpacetimeJetTests(unittest.TestCase):
    def manufactured_inputs(self):
        z=np.linspace(.7,2.3,33);r=np.linspace(0.,1.,17)
        zz=z[:,None];rr=r[None,:]
        alpha_squared=3+.2*zz+.3*zz**2+.4*rr**2
        compact=2+.5*zz+.7*zz**2+.6*rr**2
        transverse=4+.8*zz+.9*zz**2+1.1*rr**2
        radial=transverse+1.4*rr**2
        psi=np.ones_like(alpha_squared)
        a=.5*np.log(compact);b=.5*np.log(radial);c=.5*np.log(transverse)
        phi=.2+.3*zz+.4*zz**2+.5*rr**2
        chi=-.1+.6*zz-.2*zz**2+.7*rr**2
        zero=np.zeros_like(psi)
        acceleration={"zz":zero,"radial":zero,"transverse":zero,"zr":zero}
        return z,r,np.sqrt(alpha_squared),psi,a,b,c,phi,chi,acceleration,zero

    def test_polynomial_axis_and_scalar_jets_are_exact(self):
        z,r,alpha,psi,a,b,c,phi,chi,acceleration,zero=self.manufactured_inputs()
        location=1.37
        result=construct_time_symmetric_axis_spacetime_jets(
            z,r,alpha,psi,a,b,c,phi,chi,acceleration,zero,zero,zero,location,
        )
        data=result["background"]
        self.assertAlmostEqual(data["metric"][0,0],-(3+.2*location+.3*location**2),places=11)
        self.assertAlmostEqual(data["metric_first"][1,1,1],.5+1.4*location,places=11)
        self.assertAlmostEqual(data["metric_second"][1,1,1,1],1.4,places=10)
        self.assertAlmostEqual(data["metric_second"][2,2,1,1],1.2,places=10)
        self.assertAlmostEqual(data["metric_second"][2,2,2,2],5.0,places=10)
        self.assertAlmostEqual(data["metric_second"][3,3,2,2],2.2,places=10)
        self.assertAlmostEqual(data["phi_first"][1],.3+.8*location,places=11)
        self.assertAlmostEqual(data["phi_second"][1,1],.8,places=10)
        self.assertAlmostEqual(data["phi_second"][2,2],1.,places=10)

    def test_time_accelerations_are_inserted_without_coordinate_singularity(self):
        z,r,alpha,psi,a,b,c,phi,chi,acceleration,zero=self.manufactured_inputs()
        lapse=np.full_like(zero,.25)
        acceleration={
            "zz":np.full_like(zero,2.),"radial":np.full_like(zero,3.),
            "transverse":np.full_like(zero,3.),"zr":zero,
        }
        phi_tt=np.full_like(zero,4.);chi_tt=np.full_like(zero,-5.)
        result=construct_time_symmetric_axis_spacetime_jets(
            z,r,alpha,psi,a,b,c,phi,chi,acceleration,lapse,phi_tt,chi_tt,1.2,
        )
        data=result["background"]
        self.assertAlmostEqual(data["metric_second"][0,0,0,0],-2*alpha[:,0][10]*.25,delta=.02)
        self.assertAlmostEqual(data["metric_second"][0,0,1,1],2.)
        self.assertTrue(np.allclose(np.diag(data["metric_second"][0,0])[2:],3.))
        self.assertAlmostEqual(data["phi_second"][0,0],4.)
        self.assertAlmostEqual(data["chi_second"][0,0],-5.)
        self.assertAlmostEqual(result["regularity"]["mixed_zr_acceleration"],0.)

    def test_sampled_poincare_ads_closes_covariant_equation(self):
        z=np.linspace(1.,2.5,257);r=np.linspace(0.,1.,9)
        psi=np.broadcast_to(1/z[:,None],(len(z),len(r))).copy()
        zero=np.zeros_like(psi);acceleration={
            "zz":zero,"radial":zero,"transverse":zero,"zr":zero,
        }
        constructed=construct_time_symmetric_axis_spacetime_jets(
            z,r,psi,psi,zero,zero,zero,zero,zero,acceleration,zero,zero,zero,1.7,
        )["background"]
        geometry=metric_geometry_from_jets(
            constructed["metric"],constructed["metric_first"],constructed["metric_second"],
        )
        residual=reduced_einstein_two_scalar_residual(
            constructed["metric"],constructed["metric_first"],constructed["metric_second"],
            constructed["phi"],constructed["phi_first"],constructed["phi_second"],
            constructed["chi"],constructed["chi_first"],constructed["chi_second"],
            geometry["contracted_christoffel_covector"],
            geometry["contracted_christoffel_covector_first"],potential_offset=-6.,
        )
        # Cubic-spline second derivatives of the sampled rational conformal
        # factor dominate this error; the tensor construction itself is exact.
        np.testing.assert_allclose(residual["metric_residual"],0.,atol=5e-5)
        self.assertAlmostEqual(residual["phi_residual"],0.)


if __name__=="__main__":unittest.main()
