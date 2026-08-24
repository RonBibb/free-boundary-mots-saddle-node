import sys,unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.regular_so3_gh_reduction import (
    FIELD_ORDER,RegularSO3BackgroundJetField,pack_regular_so3_residual,regular_so3_gh_coefficient_matrices,
    regular_so3_perturbation_jets,regular_so3_robin_matrix,
)
from bhps.israel_wave_matrix import coupled_robin_matrix
from bhps.linearized_gh_einstein_scalar import metric_geometry_from_jets,reduced_einstein_two_scalar_residual


def flat_background():
    return {
        "metric":np.diag((-1.,1.,1.,1.,1.)),
        "metric_first":np.zeros((5,5,5)),"metric_second":np.zeros((5,5,5,5)),
        "phi":0.,"phi_first":np.zeros(5),"phi_second":np.zeros((5,5)),
        "chi":0.,"chi_first":np.zeros(5),"chi_second":np.zeros((5,5)),
    }


class RegularSO3GeneralizedHarmonicReductionTests(unittest.TestCase):
    def test_regular_variables_have_expected_axis_values(self):
        values=np.arange(1.,10.)
        jets=regular_so3_perturbation_jets(0.,values)
        self.assertAlmostEqual(jets["metric"][1,0],values[0])
        self.assertAlmostEqual(jets["metric"][1,2],0.)
        self.assertAlmostEqual(jets["metric_first"][2,1,2],values[1])
        self.assertAlmostEqual(jets["metric"][2,2],values[3])
        self.assertAlmostEqual(jets["metric"][3,3],values[3])
        self.assertAlmostEqual(jets["metric_second"][2,2,2,2],2*values[4])
        self.assertAlmostEqual(jets["metric_second"][3,3,2,2],0.)
        self.assertAlmostEqual(jets["metric_first"][2,0,2],values[5])

    def test_off_axis_values_reconstruct_radial_components(self):
        radius=.8;values=np.arange(1.,10.)
        jets=regular_so3_perturbation_jets(radius,values)
        self.assertAlmostEqual(jets["metric"][1,2]/radius,values[1])
        self.assertAlmostEqual(jets["metric"][0,2]/radius,values[5])
        self.assertAlmostEqual(
            (jets["metric"][2,2]-jets["metric"][3,3])/radius**2,values[4],
        )

    def test_flat_principal_matrices_are_common_scalar_symbol(self):
        result=regular_so3_gh_coefficient_matrices(
            flat_background(),.7,potential_offset=0.,
        )
        self.assertEqual(len(FIELD_ORDER),9)
        self.assertLess(result["principal_identity_maximum_defect"],2e-13)
        self.assertTrue(result["finite"])

    def test_constraint_damping_changes_only_metric_lower_order_blocks(self):
        undamped=regular_so3_gh_coefficient_matrices(
            flat_background(),.7,potential_offset=0.,constraint_damping=0.,
        )
        damped=regular_so3_gh_coefficient_matrices(
            flat_background(),.7,potential_offset=0.,constraint_damping=1.3,
        )
        np.testing.assert_allclose(
            damped["pure_second_matrices"],undamped["pure_second_matrices"],atol=2e-13,
        )
        self.assertGreater(
            np.max(np.abs(damped["lower_first_matrices"]-undamped["lower_first_matrices"])),
            .1,
        )
        np.testing.assert_allclose(
            damped["lower_first_matrices"][:,7:,7:],
            undamped["lower_first_matrices"][:,7:,7:],atol=2e-13,
        )
        self.assertAlmostEqual(damped["constraint_damping_rate"],1.3)

    def test_flat_regular_reaction_contains_tensor_angular_coupling(self):
        radius=.9
        result=regular_so3_gh_coefficient_matrices(
            flat_background(),radius,potential_offset=0.,
        )
        p=FIELD_ORDER.index("h_perp");d=FIELD_ORDER.index("d=(h_rr-h_perp)/r^2")
        self.assertGreater(abs(result["zero_order_matrix"][p,d]),1.)
        self.assertTrue(np.all(np.isfinite(result["zero_order_matrix"])))

    def test_cartesian_jets_match_independent_finite_differences(self):
        rng=np.random.default_rng(31);radius=.8
        values=.2*rng.normal(size=9);first=.1*rng.normal(size=(3,9))
        second=.05*rng.normal(size=(3,3,9));second=.5*(second+second.swapaxes(0,1))
        jets=regular_so3_perturbation_jets(radius,values,first,second)

        def evaluate(displacement):
            t,z,x,y,w=displacement;radial=np.sqrt((radius+x)**2+y*y+w*w)
            reduced=np.array((t,z,radial-radius))
            local=values+reduced@first+.5*np.einsum("a,abf,b->f",reduced,second,reduced)
            point=np.array((radius+x,y,w));metric=np.zeros((5,5))
            metric[1,0]=metric[0,1]=local[0]
            for index in range(3):
                metric[1,index+2]=metric[index+2,1]=local[1]*point[index]
                metric[0,index+2]=metric[index+2,0]=local[5]*point[index]
            metric[0,0]=local[2];metric[1,1]=local[6]
            metric[2:,2:]=local[3]*np.eye(3)+local[4]*np.outer(point,point)
            return np.r_[metric.ravel(),local[7:]]

        step=2e-4;base=np.zeros(5);center=evaluate(base)
        numerical_first=np.empty((5,len(center)));numerical_second=np.empty((5,5,len(center)))
        for left in range(5):
            direction=np.zeros(5);direction[left]=step
            numerical_first[left]=(evaluate(direction)-evaluate(-direction))/(2*step)
            numerical_second[left,left]=(evaluate(direction)-2*center+evaluate(-direction))/step**2
            for right in range(left):
                other=np.zeros(5);other[right]=step
                mixed=(evaluate(direction+other)-evaluate(direction-other)-evaluate(-direction+other)+evaluate(-direction-other))/(4*step**2)
                numerical_second[left,right]=mixed;numerical_second[right,left]=mixed
        analytic_first=np.concatenate((jets["metric_first"].reshape(5,25),jets["phi_first"][:,None],jets["chi_first"][:,None]),axis=1)
        analytic_second=np.concatenate((jets["metric_second"].reshape(5,5,25),jets["phi_second"][:,:,None],jets["chi_second"][:,:,None]),axis=2)
        np.testing.assert_allclose(analytic_first,numerical_first,atol=2e-8)
        np.testing.assert_allclose(analytic_second,numerical_second,atol=2e-7)

    def test_sampled_ads_background_field_closes_off_axis(self):
        z=np.linspace(1.,2.5,129);r=np.linspace(0.,1.,33)
        psi=np.broadcast_to(1/z[:,None],(len(z),len(r))).copy();zero=np.zeros_like(psi)
        acceleration={"zz":zero,"radial":zero,"transverse":zero,"zr":zero}
        field=RegularSO3BackgroundJetField(
            z,r,psi,psi,zero,zero,zero,zero,zero,acceleration,zero,zero,zero,7,
        )
        background=field.at(1.7,.43)
        geometry=metric_geometry_from_jets(
            background["metric"],background["metric_first"],background["metric_second"],
        )
        residual=reduced_einstein_two_scalar_residual(
            background["metric"],background["metric_first"],background["metric_second"],
            background["phi"],background["phi_first"],background["phi_second"],
            background["chi"],background["chi_first"],background["chi_second"],
            geometry["contracted_christoffel_covector"],geometry["contracted_christoffel_covector_first"],
            potential_offset=-6.,
        )
        np.testing.assert_allclose(residual["metric_residual"],0.,atol=3e-4)
        coefficients=regular_so3_gh_coefficient_matrices(
            background,.43,potential_offset=-6.,
        )
        phi=FIELD_ORDER.index("delta_Phi");chi=FIELD_ORDER.index("delta_chi")
        np.testing.assert_allclose(coefficients["lower_first_matrices"][:,phi,phi],0.,atol=2e-12)
        np.testing.assert_allclose(coefficients["lower_first_matrices"][:,chi,chi],0.,atol=2e-12)

    def test_reduced_robin_matrix_is_exact_invariant_restriction(self):
        c=.3;cp=-.04;phix=-.12;gamma=20.;radius=.7
        full=coupled_robin_matrix(c,cp,phix,gamma)["matrix"]
        reduced=regular_so3_robin_matrix(c,cp,phix,gamma)["matrix"]
        rng=np.random.default_rng(2);u=rng.normal(size=7)
        embedded=np.zeros(13)
        embedded[0]=u[0];embedded[1]=u[1]+radius**2*u[2]
        embedded[2]=u[1];embedded[3]=u[1];embedded[4]=radius*u[3]
        embedded[10:]=u[4:]
        derivative=full@embedded
        projected=np.array((
            derivative[0],.5*(derivative[2]+derivative[3]),
            (derivative[1]-.5*(derivative[2]+derivative[3]))/radius**2,
            derivative[4]/radius,*derivative[10:],
        ))
        np.testing.assert_allclose(projected,reduced@u,atol=2e-13)


if __name__=="__main__":unittest.main()
