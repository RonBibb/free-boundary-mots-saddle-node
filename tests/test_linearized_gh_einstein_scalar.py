import sys,unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.linearized_gh_einstein_scalar import (
    frozen_source_reduced_ricci,
    linearized_reduced_einstein_two_scalar_residual,
    metric_geometry_from_jets,
    reduced_einstein_two_scalar_residual,
    solve_reduced_einstein_two_scalar_acceleration,
    stationary_point_scalar_metric_mixing,
)


def zero_jets(metric,phi=0.,chi=0.):
    n=len(metric)
    return {
        "metric":np.asarray(metric,dtype=float),
        "metric_first":np.zeros((n,n,n)),"metric_second":np.zeros((n,n,n,n)),
        "phi":float(phi),"phi_first":np.zeros(n),"phi_second":np.zeros((n,n)),
        "chi":float(chi),"chi_first":np.zeros(n),"chi_second":np.zeros((n,n)),
    }


class LinearizedGeneralizedHarmonicEinsteinScalarTests(unittest.TestCase):
    def test_constraint_damping_tensor_has_project_convention_and_sign(self):
        eta=np.diag((-1.,1.,1.,1.,1.));first=np.zeros((5,5,5))
        second=np.zeros((5,5,5,5));source=np.array((.3,-.2,.1,.4,-.5))
        result=frozen_source_reduced_ricci(
            eta,first,second,source,np.zeros((5,5)),constraint_damping=2.,
        )
        # Gamma=0, hence C=-H.  The future normal covector is (-1,0,...).
        constraint=-source;normal=np.array((-1.,0.,0.,0.,0.))
        normal_upper=eta@normal
        expected=2*(
            .5*(np.outer(normal,constraint)+np.outer(constraint,normal))
            -.5*eta*np.dot(normal_upper,constraint)
        )
        np.testing.assert_allclose(result["constraint_damping_tensor"],expected,atol=1e-14)
        np.testing.assert_allclose(result["reduced_ricci"],expected,atol=1e-14)

    def test_constraint_damping_is_lower_order(self):
        eta=np.diag((-1.,1.,1.,1.,1.));background=zero_jets(eta)
        perturbation=zero_jets(np.zeros((5,5)))
        rng=np.random.default_rng(123);second=rng.normal(size=(5,5,5,5))
        second=.5*(second+second.swapaxes(0,1));second=.5*(second+second.swapaxes(2,3))
        perturbation["metric_second"]=second
        undamped=linearized_reduced_einstein_two_scalar_residual(
            background,perturbation,potential_offset=0.,constraint_damping=0.,
        )
        damped=linearized_reduced_einstein_two_scalar_residual(
            background,perturbation,potential_offset=0.,constraint_damping=3.,
        )
        np.testing.assert_allclose(
            damped["metric_residual"],undamped["metric_residual"],atol=2e-13,
        )

    def test_flat_principal_metric_operator_is_negative_half_wave(self):
        eta=np.diag((-1.,1.,1.,1.,1.));background=zero_jets(eta)
        perturbation=zero_jets(np.zeros((5,5)))
        rng=np.random.default_rng(8);second=rng.normal(size=(5,5,5,5))
        second=.5*(second+second.swapaxes(0,1));second=.5*(second+second.swapaxes(2,3))
        perturbation["metric_second"]=second
        result=linearized_reduced_einstein_two_scalar_residual(
            background,perturbation,potential_offset=0.,
        )
        expected=-.5*np.einsum("cd,cdab->ab",eta,second)
        np.testing.assert_allclose(result["metric_residual"],expected,atol=2e-13)

    def test_poincare_ads_background_satisfies_trace_reversed_equation(self):
        coordinate=1.7;eta=np.diag((-1.,1.,1.,1.,1.));metric=eta/coordinate**2
        background=zero_jets(metric);normal=4
        background["metric_first"][normal]=-2*eta/coordinate**3
        background["metric_second"][normal,normal]=6*eta/coordinate**4
        geometry=metric_geometry_from_jets(
            background["metric"],background["metric_first"],background["metric_second"],
        )
        residual=reduced_einstein_two_scalar_residual(
            background["metric"],background["metric_first"],background["metric_second"],
            0.,background["phi_first"],background["phi_second"],
            0.,background["chi_first"],background["chi_second"],
            geometry["contracted_christoffel_covector"],
            geometry["contracted_christoffel_covector_first"],
            potential_offset=-6.,
        )
        np.testing.assert_allclose(residual["metric_residual"],0.,atol=2e-13)
        self.assertAlmostEqual(residual["phi_residual"],0.)

    def test_pointwise_acceleration_solver_closes_reduced_equations(self):
        eta=np.diag((-1.,1.,1.,1.,1.));background=zero_jets(eta,phi=.2)
        geometry=metric_geometry_from_jets(
            background["metric"],background["metric_first"],background["metric_second"],
        )
        solved=solve_reduced_einstein_two_scalar_acceleration(
            background["metric"],background["metric_first"],background["metric_second"],
            background["phi"],background["phi_first"],background["phi_second"],
            background["chi"],background["chi_first"],background["chi_second"],
            geometry["contracted_christoffel_covector"],
            geometry["contracted_christoffel_covector_first"],
            mass_squared=1.5,potential_offset=0.,
        )
        self.assertAlmostEqual(solved["phi_acceleration"],-1.5*.2)
        metric_second=background["metric_second"].copy()
        phi_second=background["phi_second"].copy()
        chi_second=background["chi_second"].copy()
        metric_second[0,0]=solved["metric_acceleration"]
        phi_second[0,0]=solved["phi_acceleration"]
        chi_second[0,0]=solved["chi_acceleration"]
        residual=reduced_einstein_two_scalar_residual(
            background["metric"],background["metric_first"],metric_second,
            background["phi"],background["phi_first"],phi_second,
            background["chi"],background["chi_first"],chi_second,
            geometry["contracted_christoffel_covector"],
            geometry["contracted_christoffel_covector_first"],
            mass_squared=1.5,potential_offset=0.,
        )
        np.testing.assert_allclose(residual["metric_residual"],0.,atol=2e-13)
        self.assertAlmostEqual(residual["phi_residual"],0.,places=13)
        self.assertAlmostEqual(residual["chi_residual"],0.,places=13)
        expected_constraint_first=np.zeros((5,5));expected_constraint_first[0,0]=-.06
        np.testing.assert_allclose(
            residual["gauge_constraint_first_covector"],expected_constraint_first,
            atol=2e-13,
        )

    def test_stationary_scalar_mixing_is_finite_and_matches_direct_linearization(self):
        eta=np.diag((-1.,1.,1.,1.,1.));phi=.07;mass=1.3
        background=zero_jets(eta,phi=phi)
        background["phi_second"]=np.diag((0.,.12,.2,.2,.2))
        perturbation=zero_jets(np.zeros((5,5)),phi=1.)
        result=linearized_reduced_einstein_two_scalar_residual(
            background,perturbation,mass_squared=mass,potential_offset=0.,
        )
        mixing=stationary_point_scalar_metric_mixing(
            background["phi_second"],phi,mass,potential_offset=0.,
        )
        np.testing.assert_allclose(
            result["metric_residual"],mixing["metric_residual_per_delta_phi"],atol=2e-13,
        )
        self.assertTrue(mixing["finite"])
        self.assertFalse(mixing["depends_on_inverse_scalar_gradient"])

    def test_stationary_scalar_row_is_hessian_contraction(self):
        eta=np.diag((-1.,1.,1.,1.,1.));background=zero_jets(eta,phi=.2)
        background["phi_second"]=np.diag((0.,.1,.2,.3,.4))
        perturbation=zero_jets(np.diag((.5,.6,.7,.8,.9)))
        result=linearized_reduced_einstein_two_scalar_residual(
            background,perturbation,mass_squared=0.,potential_offset=0.,
        )
        expected=-sum(
            perturbation["metric"][i,i]*background["phi_second"][i,i]
            for i in range(5)
        )
        self.assertAlmostEqual(result["phi_residual"],expected,places=13)


if __name__=="__main__":unittest.main()
