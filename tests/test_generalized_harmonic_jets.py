import unittest

import numpy as np

from bhps.generalized_harmonic_jets import (
    diagonal_spatial_source_second_jets,
    initial_contracted_christoffel_time_jet,
    spatial_metric_acceleration_trace,
)


class GeneralizedHarmonicJetTests(unittest.TestCase):
    def test_time_jet_matches_direct_small_time_contraction(self):
        shape=(3,4);alpha=np.full(shape,1.7);alpha_tt=np.full(shape,.31)
        psi=np.full(shape,1.2);a=np.full(shape,.1);b=np.full(shape,-.05);c=np.zeros(shape)
        acceleration={
            "zz":np.full(shape,.7),"radial":np.full(shape,-.2),
            "transverse":np.full(shape,.15),
        }
        result=initial_contracted_christoffel_time_jet(
            acceleration,alpha,alpha_tt,psi,a,b,c,
        )
        eps=1e-7
        g00=-alpha**2-alpha*alpha_tt*eps**2
        g00_t=-2*alpha*alpha_tt*eps
        metrics=(psi**2*np.exp(2*a),psi**2*np.exp(2*b),psi**2*np.exp(2*c))
        accels=(acceleration["zz"],acceleration["radial"],acceleration["transverse"])
        direct=.5*(1/g00)*g00_t
        direct-=.5*sum(
            multiplicity*(accel*eps)/(metric+.5*accel*eps**2)
            for metric,accel,multiplicity in zip(metrics,accels,(1,1,2))
        )
        np.testing.assert_allclose(
            direct/eps,result["gamma_0_time_derivative"],rtol=2e-13,atol=2e-13,
        )

    def test_trace_and_spatial_source_jets_for_constant_accelerations(self):
        z=np.linspace(1.,2.,9);r=np.linspace(0.,1.,11);shape=(len(z),len(r))
        one=np.ones(shape);zero=np.zeros(shape)
        acceleration={"zz":2*one,"radial":4*one,"transverse":6*one}
        trace=spatial_metric_acceleration_trace(acceleration,one,zero,zero,zero)
        np.testing.assert_allclose(trace,18.)
        jets=diagonal_spatial_source_second_jets(
            z,r,acceleration,one,3*one,one,zero,zero,zero,
        )
        np.testing.assert_allclose(jets["gamma_z_second_time_derivative"],0.,atol=2e-12)
        np.testing.assert_allclose(jets["gamma_r_second_time_derivative"],0.,atol=2e-12)


if __name__=="__main__":unittest.main()
