import unittest

import numpy as np

from bhps.anisotropic_capped_arclength import AnisotropicMetricFamily


class AnisotropicCappedArclengthTests(unittest.TestCase):
    def test_metric_family_interpolates_scale_factors_linearly(self):
        amplitudes=np.array([1.,2.]);z=np.linspace(1,2,7);r=np.linspace(0,3,9)
        shape=(2,len(z),len(r));psi=np.empty(shape)
        psi[0]=2.;psi[1]=4.;zero=np.zeros(shape)
        family=AnisotropicMetricFamily(amplitudes,z,r,psi,zero,zero,zero)
        zz=np.full(5,1.4);rr=np.linspace(.2,2.,5)
        np.testing.assert_allclose(family.evaluate("A",1.25,zz,rr),2.5)
        np.testing.assert_allclose(family.evaluate("A",1.25,zz,rr,1,0),0.,atol=1e-12)


if __name__=="__main__":unittest.main()
