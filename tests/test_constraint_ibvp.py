import sys,unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

import numpy as np

from bhps.constraint_ibvp import (
    RegularSO3ConstraintBoundaryFeedback,
    RegularSO3OuterMetricConstraintFeedback,
    RegularSO3OuterMetricCharacteristicFeedback,
    RegularSO3OuterMetricWeakFluxFeedback,
    evaluate_regular_so3_constraint_field,
    flat_regular_so3_gauge_constraint,
    frozen_constraint_boundary_symbol,
    frozen_regular_so3_sommerfeld_symbol,
    frozen_constraint_mode_spectrum,
    gh_incoming_metric_constraint_extraction,
    gh_incoming_metric_constraint_lift,
    gh_metric_characteristic_projectors,
    israel_wave_boundary_count,
    linearized_regular_so3_gauge_constraint,
    regular_so3_metric_characteristic_projector_matrices,
    regular_so3_sommerfeld_energy_audit,
    regular_so3_incoming_metric_constraint_lift,
    regular_so3_radial_metric_derivative_matrices,
    regular_so3_boundary_characteristic_count,
    regular_so3_constraint_coefficient_matrices,
)
from bhps.regular_so3_gh_reduction import regular_so3_perturbation_jets


class ConstraintIBVPTests(unittest.TestCase):
    def test_regular_boundary_counts_close_without_adding_wall_constraints(self):
        count=regular_so3_boundary_characteristic_count(2)
        self.assertEqual(count["compact_wall_metric_rows"],{
            "mixed_normal_tangent_gauge":2,"tangential_israel":4,
            "normal_wave_coordinate":1,
        })
        self.assertEqual(count["outer_radial_metric_rows"],{
            "gauge_characteristics":3,"constraint_characteristics":3,
            "physical_characteristics":1,
        })
        self.assertEqual(count["compact_wall_total_rows"],9)
        self.assertEqual(count["outer_radial_total_rows"],9)
        self.assertTrue(count["constraint_rows_replace_incoming_data"])
        self.assertTrue(count["both_face_counts_close"])

    def test_five_dimensional_einstein_scalar_boundary_count_closes(self):
        count=israel_wave_boundary_count(5,2)
        self.assertEqual(count["metric_wave_fields"],15)
        self.assertEqual(count["israel_rows"],10)
        self.assertEqual(count["total_wave_fields"],17)
        self.assertEqual(count["total_boundary_rows"],17)
        self.assertTrue(count["count_closes"])

    def test_five_dimensional_metric_characteristic_projectors_close(self):
        metric=np.diag((-1.,1.,1.,1.,1.));time=np.array((-1.,0.,0.,0.,0.))
        normal=np.array((0.,1.,0.,0.,0.))
        result=gh_metric_characteristic_projectors(metric,time,normal)
        self.assertEqual(result["full_ranks"],{"gauge":5,"constraint":5,"physical":5})
        self.assertLess(result["completeness_defect"],1e-13)
        self.assertLess(result["idempotence_defect"],1e-13)
        self.assertLess(result["orthogonality_defect"],1e-13)
        # A pure transverse trace has no physical projection; this checks the
        # five-dimensional 1/3 coefficient rather than the 3+1 value 1/2.
        transverse_trace=np.diag((0.,0.,1.,1.,1.))
        physical=np.einsum("abcd,cd->ab",result["physical"],transverse_trace)
        np.testing.assert_allclose(physical,0.,atol=2e-14)

    def test_incoming_metric_constraint_lift_removes_constraint(self):
        metric=np.diag((-1.,1.,1.,1.,1.));time=np.array((-1.,0.,0.,0.,0.))
        normal=np.array((0.,1.,0.,0.,0.));constraint=np.array((.2,-.3,.1,.4,-.2))
        correction=gh_incoming_metric_constraint_lift(
            metric,time,normal,constraint,.8,.07,
        )
        extracted=gh_incoming_metric_constraint_extraction(
            metric,time,normal,correction,
        )
        np.testing.assert_allclose(extracted,-.87*constraint,atol=2e-14)
        projectors=gh_metric_characteristic_projectors(metric,time,normal)
        np.testing.assert_allclose(
            np.einsum("abcd,cd->ab",projectors["constraint"],correction),
            correction,atol=2e-14,
        )
        np.testing.assert_allclose(
            np.einsum("abcd,cd->ab",projectors["gauge"],correction),0.,atol=2e-14,
        )
        np.testing.assert_allclose(
            np.einsum("abcd,cd->ab",projectors["physical"],correction),0.,atol=2e-14,
        )

    def test_regular_so3_metric_characteristic_ranks_are_three_three_one(self):
        metric=np.diag((-1.,1.,1.,1.,1.))
        for direction in ("compact","radial"):
            for sign in (-1.,1.):
                result=regular_so3_metric_characteristic_projector_matrices(
                    metric,.7,direction,sign,
                )
                self.assertEqual(result["ranks"],{
                    "gauge":3,"constraint":3,"physical":1,
                })
                self.assertLess(result["completeness_defect"],1e-13)
                self.assertLess(result["idempotence_defect"],1e-13)
                self.assertLess(result["orthogonality_defect"],1e-13)

    def test_regular_constraint_lift_matches_full_five_dimensional_extraction(self):
        metric=np.diag((-1.3**2,1.1**2,.9**2,.9**2,.9**2));radius=.73
        constraint=np.array((.2,-.13,.17))
        for direction in ("compact","radial"):
            for sign in (-1.,1.):
                result=regular_so3_incoming_metric_constraint_lift(
                    metric,radius,direction,constraint,sign,
                )
                values=np.r_[result["correction"],0.,0.]
                full=regular_so3_perturbation_jets(radius,values)["metric"]
                extracted=gh_incoming_metric_constraint_extraction(
                    metric,result["time_normal_covector"],
                    result["boundary_normal_covector"],full,
                )
                np.testing.assert_allclose(
                    extracted,-result["lapse"]*result["full_constraint_covector"],
                    atol=2e-13,
                )
                projectors=regular_so3_metric_characteristic_projector_matrices(
                    metric,radius,direction,sign,
                )
                np.testing.assert_allclose(
                    projectors["constraint"]@result["correction"],
                    result["correction"],atol=2e-13,
                )

    def test_live_outer_metric_feedback_excludes_physical_wall_corners(self):
        z=np.linspace(.5,1.5,5);r=np.linspace(0.,2.,7)
        constraint=np.empty((len(z),len(r),3))
        zz,rr=np.meshgrid(z,r,indexing="ij")
        constraint[:,:,0]=.1+.02*zz;constraint[:,:,1]=-.07+.01*rr
        constraint[:,:,2]=.03-.01*zz
        class Feedback:
            def __init__(self):self.z=z;self.r=r
            def evaluate(self,position,velocity,source,time=0.):
                del position,velocity,source,time
                return constraint
        metrics=np.broadcast_to(np.diag((-1.,1.,1.,1.,1.)),(len(z),5,5)).copy()
        runtime=RegularSO3OuterMetricConstraintFeedback(Feedback(),metrics)
        result=runtime.evaluate(
            np.zeros((len(z),len(r),9)),np.zeros((len(z),len(r),9)),
            np.zeros((len(z),len(r),3)),
        )
        self.assertEqual(int(np.sum(result["incoming_mask"])),len(z)-2)
        self.assertFalse(result["incoming_mask"][0,-1])
        self.assertFalse(result["incoming_mask"][-1,-1])
        for i in range(1,len(z)-1):
            reduced=result["characteristic_correction"][i,-1]
            full=regular_so3_perturbation_jets(r[-1],reduced)["metric"]
            extracted=gh_incoming_metric_constraint_extraction(
                metrics[i],np.array((-1.,0.,0.,0.,0.)),np.array((0.,0.,1.,0.,0.)),full,
            )
            expected=np.array((constraint[i,-1,0],constraint[i,-1,1],r[-1]*constraint[i,-1,2],0.,0.))
            np.testing.assert_allclose(extracted,-expected,atol=2e-13)

    def test_complete_outer_feedback_respects_three_three_one_sectors(self):
        z=np.linspace(.5,1.5,5);r=np.linspace(0.,2.,9)
        metric=np.diag((-1.,1.,1.,1.,1.))
        metrics=np.broadcast_to(metric,(len(z),5,5)).copy()
        class ZeroConstraint:
            def __init__(self):self.z=z;self.r=r
            def evaluate(self,position,velocity,source,time=0.):
                del position,velocity,source,time
                return np.zeros((len(z),len(r),3))
        projector=regular_so3_metric_characteristic_projector_matrices(
            metric,r[-1],"radial",1.,
        )
        seed=np.array((.2,-.1,.3,.17,-.08,.11,-.14))
        derivative=projector["constraint"]@seed
        profile=(r**2-r[-1]**2)/(2*r[-1])
        q=np.zeros((len(z),len(r),9));v=np.zeros_like(q)
        q[:,:,:7]=profile[None,:,None]*derivative[None,None,:]
        feedback=RegularSO3OuterMetricCharacteristicFeedback(
            ZeroConstraint(),metrics,gauge_rate=.7,physical_rate=1.3,
        )
        result=feedback.evaluate(q,v,np.zeros((len(z),len(r),3)))
        for i in range(1,len(z)-1):
            incoming=result["incoming_characteristic"][i,-1,:7]
            np.testing.assert_allclose(incoming,-derivative,atol=2e-13)
            np.testing.assert_allclose(result["gauge_correction"][i,-1],0.,atol=2e-13)
            np.testing.assert_allclose(result["physical_correction"][i,-1],0.,atol=2e-13)
        self.assertFalse(result["incoming_mask"][0,-1])
        self.assertFalse(result["incoming_mask"][-1,-1])

    def test_complete_outer_gauge_and_physical_feedback_are_projector_pure(self):
        z=np.linspace(.5,1.5,5);r=np.linspace(0.,2.,9)
        metric=np.diag((-1.,1.,1.,1.,1.))
        metrics=np.broadcast_to(metric,(len(z),5,5)).copy()
        class ZeroConstraint:
            def __init__(self):self.z=z;self.r=r
            def evaluate(self,position,velocity,source,time=0.):
                del position,velocity,source,time
                return np.zeros((len(z),len(r),3))
        q=np.zeros((len(z),len(r),9));v=np.zeros_like(q)
        rng=np.random.default_rng(104);v[:,-1,:7]=rng.normal(size=(len(z),7))
        feedback=RegularSO3OuterMetricCharacteristicFeedback(
            ZeroConstraint(),metrics,gauge_rate=.7,physical_rate=1.3,
        )
        result=feedback.evaluate(q,v,np.zeros((len(z),len(r),3)))
        projector=regular_so3_metric_characteristic_projector_matrices(
            metric,r[-1],"radial",1.,
        )
        for i in range(1,len(z)-1):
            incoming=result["incoming_characteristic"][i,-1,:7]
            np.testing.assert_allclose(
                result["gauge_correction"][i,-1,:7],
                -.7*projector["gauge"]@incoming,atol=2e-13,
            )
            np.testing.assert_allclose(
                result["physical_correction"][i,-1,:7],
                -1.3*projector["physical"]@incoming,atol=2e-13,
            )
            np.testing.assert_allclose(
                projector["constraint"]@result["gauge_correction"][i,-1,:7],0.,atol=2e-13,
            )
            np.testing.assert_allclose(
                projector["constraint"]@result["physical_correction"][i,-1,:7],0.,atol=2e-13,
            )

    def test_weak_outer_feedback_uses_energy_surface_scaling_and_corner_precedence(self):
        z=np.linspace(.5,1.5,5);r=np.linspace(0.,2.,7)
        class Feedback:
            def __init__(self):self.z=z;self.r=r
            def evaluate(self,position,velocity,source,time=0.):
                del position,velocity,source,time
                correction=np.zeros((len(z),len(r),9));correction[:,-1,2]=2.
                mask=np.zeros((len(z),len(r)),dtype=bool);mask[1:-1,-1]=True
                return {"characteristic_correction":correction,"incoming_mask":mask,
                        "lapse":np.arange(len(z),dtype=float)+1.}
        surface=np.linspace(2.,3.,len(z))
        weak=RegularSO3OuterMetricWeakFluxFeedback(Feedback(),surface,penalty=.4)
        result=weak.evaluate(
            np.zeros((len(z),len(r),9)),np.zeros((len(z),len(r),9)),
            np.zeros((len(z),len(r),3)),
        )
        expected=.4*(np.arange(len(z))+1)*surface*2
        np.testing.assert_allclose(result["weak_radial_flux"][1:-1,2],expected[1:-1])
        np.testing.assert_allclose(result["weak_radial_flux"][[0,-1]],0.)

    def test_complete_weak_outer_feedback_uses_sommerfeld_complement(self):
        z=np.linspace(.5,1.5,5);r=np.linspace(0.,2.,7)
        metric=np.diag((-1.,1.,1.,1.,1.));metrics=np.broadcast_to(metric,(len(z),5,5)).copy()
        class ZeroConstraint:
            def __init__(self):self.z=z;self.r=r
            def evaluate(self,position,velocity,source,time=0.):
                del position,velocity,source,time
                return np.zeros((len(z),len(r),3))
        complete=RegularSO3OuterMetricCharacteristicFeedback(
            ZeroConstraint(),metrics,gauge_rate=.7,physical_rate=1.3,
        )
        weak=RegularSO3OuterMetricWeakFluxFeedback(
            complete,np.ones(len(z)),penalty=.4,
        )
        q=np.zeros((len(z),len(r),9));v=np.zeros_like(q)
        seed=np.array((.2,-.1,.3,.17,-.08,.11,-.14));v[:,-1,:7]=seed
        result=weak.evaluate(q,v,np.zeros((len(z),len(r),3)))
        projector=complete.projectors[2]
        expected=-.4*(.7*projector["gauge"]@seed+1.3*projector["physical"]@seed)
        np.testing.assert_allclose(result["weak_radial_flux"][2,:7],expected,atol=2e-13)
        self.assertEqual(
            result["weak_flux_policy"],
            "E27_constraint_plus_Sommerfeld_gauge_physical_weak_flux",
        )

    def test_constraint_sommerfeld_weak_flux_is_projector_pure(self):
        z=np.linspace(.5,1.5,5);r=np.linspace(0.,2.,7)
        metric=np.diag((-1.,1.,1.,1.,1.));metrics=np.broadcast_to(metric,(len(z),5,5)).copy()
        class ZeroConstraint:
            def __init__(self):self.z=z;self.r=r
            def evaluate(self,position,velocity,source,time=0.):
                del position,velocity,source,time
                return np.zeros((len(z),len(r),3))
        complete=RegularSO3OuterMetricCharacteristicFeedback(ZeroConstraint(),metrics)
        weak=RegularSO3OuterMetricWeakFluxFeedback(
            complete.constraint,np.ones(len(z)),penalty=.8,
            constraint_projectors=complete.projectors,constraint_sommerfeld=True,
        )
        q=np.zeros((len(z),len(r),9));v=np.zeros_like(q)
        seed=np.array((.2,-.1,.3,.17,-.08,.11,-.14));v[:,-1,:7]=seed
        result=weak.evaluate(q,v,np.zeros((len(z),len(r),3)))
        expected=-.8*complete.projectors[2]["constraint"]@seed
        np.testing.assert_allclose(result["weak_radial_flux"][2,:7],expected,atol=2e-13)
        np.testing.assert_allclose(
            complete.projectors[2]["gauge"]@result["weak_radial_flux"][2,:7],0.,atol=2e-13,
        )
        np.testing.assert_allclose(
            complete.projectors[2]["physical"]@result["weak_radial_flux"][2,:7],0.,atol=2e-13,
        )

    def test_direct_regular_incoming_characteristic_matches_full_tensor_map(self):
        metric=np.diag((-1.2**2,1.1**2,.9**2,.9**2,.9**2));radius=1.7
        z=np.linspace(.5,1.5,5);r=np.linspace(0.,radius,9)
        metrics=np.broadcast_to(metric,(len(z),5,5)).copy()
        class ZeroConstraint:
            def __init__(self):self.z=z;self.r=r
            def evaluate(self,position,velocity,source,time=0.):
                del position,velocity,source,time
                return np.zeros((len(z),len(r),3))
        rng=np.random.default_rng(206);q=np.zeros((len(z),len(r),9));v=np.zeros_like(q)
        q[:,:,:7]=rng.normal(size=(len(z),len(r),7));v[:,:,:7]=rng.normal(size=(len(z),len(r),7))
        feedback=RegularSO3OuterMetricCharacteristicFeedback(
            ZeroConstraint(),metrics,gamma2=.4,
        )
        result=feedback.evaluate(q,v,np.zeros((len(z),len(r),3)))
        radial=np.einsum("j,zjf->zf",feedback.radial_derivative_row,q)
        lapse=result["lapse"][2];normal=feedback.normal_upper_r[2]
        q_tensor=regular_so3_perturbation_jets(radius,q[2,-1])["metric"]
        v_tensor=regular_so3_perturbation_jets(radius,v[2,-1])["metric"]
        first=np.zeros((3,9));first[2]=radial[2]
        r_tensor=regular_so3_perturbation_jets(radius,q[2,-1],first)["metric_first"][2]
        full=-v_tensor/lapse-normal*r_tensor-.4*q_tensor
        transverse=.5*(full[3,3]+full[4,4])
        packed=np.array((
            full[1,0],full[1,2]/radius,full[0,0],transverse,
            (full[2,2]-transverse)/radius**2,full[0,2]/radius,full[1,1],
        ))
        np.testing.assert_allclose(
            result["incoming_characteristic"][2,-1,:7],packed,atol=3e-13,
        )

    def test_radial_metric_derivative_map_includes_regular_basis_derivatives(self):
        radius=1.4;rng=np.random.default_rng(918)
        values=rng.normal(size=7);radial=rng.normal(size=7)
        maps=regular_so3_radial_metric_derivative_matrices(radius)
        reconstructed=maps["value_matrix"]@values+maps["derivative_matrix"]@radial
        first=np.zeros((3,9));first[2,:7]=radial
        tensor=regular_so3_perturbation_jets(
            radius,np.r_[values,0.,0.],first,
        )["metric_first"][2]
        transverse=.5*(tensor[3,3]+tensor[4,4])
        direct=np.array((
            tensor[1,0],tensor[1,2]/radius,tensor[0,0],transverse,
            (tensor[2,2]-transverse)/radius**2,tensor[0,2]/radius,tensor[1,1],
        ))
        np.testing.assert_allclose(reconstructed,direct,atol=3e-13)
        self.assertGreater(np.linalg.norm(maps["value_matrix"]),.1)

    def test_frozen_constraint_symbols_have_no_unstable_root(self):
        for real in (.001,.1,1.,10.):
            for imag in (-20.,-1.,0.,1.,20.):
                for wave in (0.,.2,2.,20.):
                    result=frozen_constraint_boundary_symbol(real,imag,wave)
                    self.assertFalse(result["unstable_normal_root"])
                    self.assertFalse(result["unstable_tangential_root"])
                    self.assertGreater(result["decay_rate"].real,0.)

    def test_equal_rate_sommerfeld_flux_is_exactly_energy_dissipative(self):
        metrics=(
            np.diag((-1.,1.,1.,1.,1.)),
            np.diag((-1.2**2,1.1**2,.9**2,.9**2,.9**2)),
        )
        for metric in metrics:
            result=regular_so3_sommerfeld_energy_audit(metric,1.7)
            self.assertTrue(result["strictly_energy_dissipative"])
            self.assertLess(result["equal_rate_identity_defect"],3e-13)
            np.testing.assert_allclose(
                result["symmetric_eigenvalues"],np.ones(7),atol=3e-13,
            )

    def test_frozen_regular_sommerfeld_symbol_has_no_growing_root(self):
        minimum=np.inf
        for real in (1e-4,.01,.2,1.,10.):
            for imag in (-30.,-2.,0.,2.,30.):
                for wave in (0.,.1,1.,20.):
                    for penalty in (.25,.5,1.,2.,4.):
                        result=frozen_regular_so3_sommerfeld_symbol(
                            real,imag,wave,wave_speed=.73,
                            gauge_rate=penalty,constraint_rate=penalty,
                            physical_rate=penalty,
                        )
                        self.assertFalse(result["unstable_root"])
                        self.assertGreater(result["decay_rate"].real,0.)
                        minimum=min(minimum,result["minimum_normalized_gap"])
        self.assertGreater(minimum,1e-6)

    def test_regular_so3_constraint_matches_independent_flat_formula(self):
        rng=np.random.default_rng(90210);radius=.73
        eta=np.diag((-1.,1.,1.,1.,1.))
        background={
            "metric":eta,"metric_first":np.zeros((5,5,5)),
            "metric_second":np.zeros((5,5,5,5)),
        }
        values=rng.normal(size=9);first=rng.normal(size=(3,9))
        second=rng.normal(size=(3,3,9));second=.5*(second+second.swapaxes(0,1))
        source=rng.normal(size=3)
        perturbation=regular_so3_perturbation_jets(radius,values,first,second)
        direct=linearized_regular_so3_gauge_constraint(
            background,perturbation,radius,source,
        )
        closed=flat_regular_so3_gauge_constraint(radius,values,first,second,source)
        np.testing.assert_allclose(direct,closed,rtol=2e-13,atol=2e-13)

    def test_constraint_coefficient_matrices_reconstruct_direct_result(self):
        rng=np.random.default_rng(27);radius=.83;eta=np.diag((-1.,1.,1.,1.,1.))
        background={
            "metric":eta,"metric_first":np.zeros((5,5,5)),
            "metric_second":np.zeros((5,5,5,5)),
        }
        values=rng.normal(size=9);first=rng.normal(size=(3,9));source=rng.normal(size=3)
        matrices=regular_so3_constraint_coefficient_matrices(background,radius)
        reconstructed=(
            matrices["zero_matrix"]@values
            +np.einsum("dab,db->a",matrices["first_matrices"],first)-source
        )
        direct=linearized_regular_so3_gauge_constraint(
            background,regular_so3_perturbation_jets(radius,values,first),radius,source,
        )
        np.testing.assert_allclose(reconstructed,direct,atol=2e-13)

    def test_gridded_constraint_evaluator_matches_flat_closed_form(self):
        z=np.linspace(0,1,9);r=np.linspace(0,1,11);zz,rr=np.meshgrid(z,r,indexing="ij")
        eta=np.diag((-1.,1.,1.,1.,1.));background={
            "metric":eta,"metric_first":np.zeros((5,5,5)),
            "metric_second":np.zeros((5,5,5,5)),
        }
        # Polynomial fields make the five-point diagnostic derivatives exact.
        values=np.empty((len(z),len(r),9));velocity=np.empty_like(values)
        for field in range(9):
            values[:,:,field]=(field+1)*(.2+.3*zz+.1*zz**2+.4*rr**2)
            velocity[:,:,field]=(field+1)*(.07-.03*zz+.02*rr**2)
        source=np.stack((.1+.02*zz,-.2+.03*rr**2,.05+.01*zz*rr**2),axis=2)
        support=np.array((.05,.1,.2,.4,.7,1.))
        sampled=[]
        for radius in support:
            sampled.append(regular_so3_constraint_coefficient_matrices(background,radius))
        zero_samples=np.asarray([item["zero_matrix"] for item in sampled])
        first_samples=np.asarray([item["first_matrices"] for item in sampled])
        first_samples[:,2]*=support[:,None,None]
        zero=np.empty((len(z),len(r),3,9));first=np.empty((3,len(z),len(r),3,9))
        for i in range(len(z)):
            for index in np.ndindex((3,9)):
                zero[i,:,index[0],index[1]]=np.polynomial.polynomial.polyval(
                    r**2,np.polynomial.polynomial.polyfit(support**2,zero_samples[:,index[0],index[1]],3),
                )
            for direction in range(3):
                for index in np.ndindex((3,9)):
                    first[direction,i,:,index[0],index[1]]=np.polynomial.polynomial.polyval(
                        r**2,np.polynomial.polynomial.polyfit(support**2,first_samples[:,direction,index[0],index[1]],3),
                    )
        result=evaluate_regular_so3_constraint_field(
            z,r,values,velocity,source,zero,first,5,radial_first_is_scaled=True,
        )["constraint"]
        dz=np.empty_like(values);dr=np.empty_like(values)
        for field in range(9):
            dz[:,:,field]=(field+1)*(.3+.2*zz)
            dr[:,:,field]=(field+1)*.8*rr
        expected=np.empty_like(result)
        for i in range(len(z)):
            for j,radius in enumerate(r):
                jets=np.stack((velocity[i,j],dz[i,j],dr[i,j]))
                second=np.zeros((3,3,9));second[2,2]=.8*np.arange(1,10)
                expected[i,j]=flat_regular_so3_gauge_constraint(
                    radius,values[i,j],jets,second,source[i,j],
                )
        np.testing.assert_allclose(result,expected,atol=2e-9)

    def test_live_boundary_feedback_wraps_constraint_evaluator(self):
        z=np.linspace(0,1,7);r=np.linspace(0,1,9)
        values=np.zeros((7,9,9));velocity=np.zeros_like(values)
        source=np.zeros((7,9,3));source[...,0]=.2
        zero=np.zeros((7,9,3,9));first=np.zeros((3,7,9,3,9))
        feedback=RegularSO3ConstraintBoundaryFeedback(
            z,r,zero,first,radial_first_is_scaled=True,
        )
        result=feedback.evaluate(values,velocity,source)
        np.testing.assert_allclose(result[...,0],-.2)
        np.testing.assert_allclose(result[...,1:],0.)

    def test_regular_constraint_axis_formula_is_finite_even_limit(self):
        rng=np.random.default_rng(51);values=rng.normal(size=9)
        first=rng.normal(size=(3,9));first[2]=0.
        second=np.zeros((3,3,9));second[2,2]=rng.normal(size=9)
        source=rng.normal(size=3)
        axis=flat_regular_so3_gauge_constraint(0.,values,first,second,source)
        for radius in (1e-2,1e-4,1e-6):
            off_first=first.copy();off_first[2]=radius*second[2,2]
            off_axis=flat_regular_so3_gauge_constraint(
                radius,values,off_first,second,source,
            )
            self.assertLess(np.max(np.abs(off_axis-axis)),5*radius**2)
        self.assertTrue(np.all(np.isfinite(axis)))

    def test_nonconstant_flat_constraints_are_strictly_damped(self):
        for damping in (.1,1.,5.):
            for wave in (.02,.2,2.,20.):
                result=frozen_constraint_mode_spectrum(wave,damping)
                self.assertTrue(result["strictly_damped"])
                self.assertLess(result["maximum_real_part"],0.)
        constant=frozen_constraint_mode_spectrum(0.,1.)
        self.assertTrue(constant["constant_mode_exception"])
        self.assertFalse(constant["strictly_damped"])
        self.assertAlmostEqual(constant["maximum_real_part"],0.)

    def test_ads5_frozen_constraint_modes_are_damped(self):
        for ell in (.5,1.,3.):
            for wave in (0.,.2,2.,20.):
                result=frozen_constraint_mode_spectrum(
                    wave,1/ell,ricci_mixed_eigenvalue=-4/ell**2,
                )
                self.assertTrue(result["strictly_damped"])
                self.assertLess(result["maximum_real_part"],0.)


if __name__=="__main__":unittest.main()
