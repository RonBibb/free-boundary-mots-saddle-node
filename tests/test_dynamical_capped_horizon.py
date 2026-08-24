import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.dynamical_capped_horizon import (
    capped_outgoing_expansion,regular_so3_adm_slice,
)


def flat_state(z,r):
    q=np.zeros((len(z),len(r),9));q[:,:,2]=-1.;q[:,:,3]=1.;q[:,:,6]=1.
    return q


def half_sphere(radius,nodes=301):
    theta=np.linspace(1e-3,np.pi/2,nodes)
    return {"theta":theta,"rho":np.full_like(theta,radius),"slope":np.zeros_like(theta)}


class DynamicalCappedHorizonTests(unittest.TestCase):
    def test_flat_slice_adm_fields(self):
        z=np.linspace(0,4,33);r=np.linspace(0,3,49);q=flat_state(z,r)
        adm=regular_so3_adm_slice(q,np.zeros_like(q),z,r)
        np.testing.assert_allclose(adm["lapse"],1.,atol=1e-13)
        np.testing.assert_allclose(adm["extrinsic_base"],0.,atol=1e-12)
        np.testing.assert_allclose(adm["extrinsic_sphere_eigenvalue"],0.,atol=1e-12)

    def test_flat_half_sphere_has_three_over_radius_expansion(self):
        z=np.linspace(0,4,49);r=np.linspace(0,3,65);q=flat_state(z,r)
        radius=1.2
        result=capped_outgoing_expansion(q,np.zeros_like(q),z,r,half_sphere(radius))
        np.testing.assert_allclose(
            result["outgoing_expansion"][3:-3],3/radius,rtol=2e-5,atol=2e-5,
        )

    def test_isotropic_extrinsic_curvature_shifts_expansion_by_minus_three_k(self):
        z=np.linspace(0,4,49);r=np.linspace(0,3,65);q=flat_state(z,r)
        k=.17;v=np.zeros_like(q);v[:,:,3]=-2*k;v[:,:,6]=-2*k
        radius=1.2
        result=capped_outgoing_expansion(q,v,z,r,half_sphere(radius))
        np.testing.assert_allclose(
            result["outgoing_expansion"][3:-3],3/radius-3*k,rtol=2e-5,atol=2e-5,
        )

    def test_constant_tangential_shift_has_zero_extrinsic_curvature(self):
        z=np.linspace(0,4,33);r=np.linspace(0,3,49);q=flat_state(z,r)
        shift=.21;q[:,:,0]=shift;q[:,:,2]=-1+shift**2
        adm=regular_so3_adm_slice(q,np.zeros_like(q),z,r)
        np.testing.assert_allclose(adm["lapse"],1.,atol=1e-12)
        np.testing.assert_allclose(adm["extrinsic_base"],0.,atol=2e-11)
        np.testing.assert_allclose(adm["extrinsic_sphere_eigenvalue"],0.,atol=2e-11)


if __name__=="__main__":unittest.main()
