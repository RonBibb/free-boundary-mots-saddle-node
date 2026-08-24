import sys,unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.gh_source_driver import (
    AxisymmetricDrivenGHWaveIBVP,RegularSO3AnchoredDampedWaveTarget,
    bjorhus_constraint_boundary_rhs,bjorhus_source_boundary_rhs,
    first_order_driver_characteristic_speeds,
    integrate_source_driver,
    regular_so3_anchored_damped_wave_target,
    regular_so3_live_source_shift_advection,
    regular_so3_nonlinear_anchored_damped_wave_target,
    regular_so3_background_source_shift_advection,regular_so3_source_coupling_matrices,
    regular_so3_source_jets,source_driver_rhs,
)
from bhps.axisymmetric_reduced_wave_evolution import AxisymmetricVariableReducedWaveIBVP
from bhps.linearized_gh_einstein_scalar import linearized_reduced_einstein_two_scalar_residual
from bhps.regular_so3_gh_reduction import regular_so3_perturbation_jets


def flat_background():
    return {
        "metric":np.diag((-1.,1.,1.,1.,1.)),
        "metric_first":np.zeros((5,5,5)),"metric_second":np.zeros((5,5,5,5)),
        "phi":0.,"phi_first":np.zeros(5),"phi_second":np.zeros((5,5)),
        "chi":0.,"chi_first":np.zeros(5),"chi_second":np.zeros((5,5)),
    }


def zero_perturbation():
    result=flat_background();result["metric"]=np.zeros((5,5))
    return result


class GeneralizedHarmonicSourceDriverTests(unittest.TestCase):
    def test_metric_constraint_feedback_enters_only_selected_wave_acceleration(self):
        z=np.linspace(.5,1.5,5);r=np.linspace(0.,2.,7);one=np.ones((len(z),len(r)))
        wall=np.zeros((len(r),7,7));wave=AxisymmetricVariableReducedWaveIBVP(
            z,r,one,one,one,wall,wall,np.ones(len(r)),np.ones(len(r)),
            dirichlet_fields=2,
        )
        zero=np.zeros((len(z),len(r),9,3));first=np.zeros((3,*zero.shape))
        driver=AxisymmetricDrivenGHWaveIBVP(wave,zero,first,1.,2.)
        q=np.zeros((len(z),len(r),9));v=np.zeros_like(q)
        source=np.zeros((len(z),len(r),3));memory=np.zeros_like(source)
        target=np.zeros_like(source);characteristic=np.zeros_like(q)
        mask=np.zeros((len(z),len(r)),dtype=bool);mask[1:-1,-1]=True
        characteristic[mask,:7]=np.arange(1.,8.)
        lapse=np.linspace(.7,1.1,len(z))
        class Feedback:
            def evaluate(self,position,velocity,live_source,time=0.):
                del position,velocity,live_source,time
                return {
                    "characteristic_correction":characteristic,
                    "incoming_mask":mask,"lapse":lapse,
                }
        result=driver.rhs(
            0.,q,v,source,memory,target,
            metric_boundary_constraint_feedback=Feedback(),
        )
        expected=np.zeros_like(q)
        expected[mask]=-np.repeat(lapse[:,None],len(r),axis=1)[mask,None]*characteristic[mask]
        np.testing.assert_allclose(result[1],expected,atol=2e-14)
        np.testing.assert_allclose(result[0],0.,atol=0.)
        np.testing.assert_allclose(result[2],0.,atol=0.)
        np.testing.assert_allclose(result[3],0.,atol=0.)

    def test_regular_source_vector_has_finite_axis_jet(self):
        values=np.array((.2,-.4,.7));first=np.arange(9.).reshape(3,3)/10
        axis=regular_so3_source_jets(0.,values,first)
        np.testing.assert_allclose(axis["covector"],(.2,-.4,0.,0.,0.))
        np.testing.assert_allclose(np.diag(axis["first"])[2:],(.7,.7,.7))
        off=regular_so3_source_jets(.3,values,first)
        self.assertAlmostEqual(off["covector"][2],.21)
        self.assertAlmostEqual(off["first"][2,2],.7+.3*first[2,2])

    def test_pure_source_perturbation_enters_as_symmetric_gradient(self):
        background=flat_background();perturbation=zero_perturbation()
        rng=np.random.default_rng(17);source_first=rng.normal(size=(5,5))
        perturbation["gauge_source_covector"]=np.zeros(5)
        perturbation["gauge_source_first"]=source_first
        result=linearized_reduced_einstein_two_scalar_residual(
            background,perturbation,potential_offset=0.,
        )
        expected=.5*(source_first+source_first.T)
        np.testing.assert_allclose(result["metric_residual"],expected,atol=2e-13)
        self.assertAlmostEqual(result["phi_residual"],0.)
        self.assertAlmostEqual(result["chi_residual"],0.)

    def test_regular_source_coupling_is_finite_and_metric_only(self):
        result=regular_so3_source_coupling_matrices(
            flat_background(),.8,potential_offset=0.,
        )
        self.assertTrue(result["finite"])
        self.assertGreater(np.linalg.norm(result["evolution_first_matrices"]),1.)
        np.testing.assert_allclose(result["evolution_zero_matrix"][7:],0.,atol=1e-14)
        np.testing.assert_allclose(result["evolution_first_matrices"][:,7:],0.,atol=1e-14)

    def test_source_coupling_includes_constraint_damping_dependence(self):
        undamped=regular_so3_source_coupling_matrices(
            flat_background(),.8,potential_offset=0.,constraint_damping=0.,
        )
        damped=regular_so3_source_coupling_matrices(
            flat_background(),.8,potential_offset=0.,constraint_damping=1.4,
        )
        self.assertGreater(np.linalg.norm(
            damped["evolution_zero_matrix"]-undamped["evolution_zero_matrix"]
        ),.1)
        np.testing.assert_allclose(
            damped["evolution_first_matrices"],undamped["evolution_first_matrices"],atol=2e-13,
        )
        self.assertAlmostEqual(damped["constraint_damping_rate"],1.4)

    def test_driver_rhs_has_target_as_fixed_point(self):
        target=np.array((.2,-.1,.4))
        hdot,thetadot=source_driver_rhs(target,np.zeros(3),target,2.,3.)
        np.testing.assert_allclose(hdot,0.)
        np.testing.assert_allclose(thetadot,0.)

    def test_regular_shift_advection_matches_cartesian_directional_derivative(self):
        z=np.array((.4,.7));r=np.array((.3,.8));zz,rr=np.meshgrid(z,r,indexing="ij")
        q=np.zeros((2,2,9));q[:,:,0]=.2-.03*zz;q[:,:,5]=-.1+.04*rr**2
        inverse_z=1.1+.02*zz;inverse_r=.9-.03*rr**2
        source=np.empty((2,2,3));source_z=np.empty_like(source);scaled=np.empty_like(source)
        source[:,:,0]=.3+.2*zz+.1*rr**2
        source[:,:,1]=-.2+.05*zz-.07*rr**2
        source[:,:,2]=.4-.03*zz+.06*rr**2
        source_z[:,:,0]=.2;source_z[:,:,1]=.05;source_z[:,:,2]=-.03
        scaled[:,:,0]=.2*rr**2;scaled[:,:,1]=-.14*rr**2;scaled[:,:,2]=.12*rr**2
        result=regular_so3_background_source_shift_advection(
            q,r,source,source_z,scaled,inverse_z,inverse_r,
        )
        beta_z=inverse_z*q[:,:,0];beta_r=rr*inverse_r*q[:,:,5]
        expected=np.empty_like(result)
        expected[:,:,0]=beta_z*.2+beta_r*(.2*rr)
        expected[:,:,1]=beta_z*.05+beta_r*(-.14*rr)
        # H_x=r h_r, hence (beta^r d_r H_x)/r includes h_r/r basis term.
        expected[:,:,2]=beta_z*(-.03)+beta_r*(source[:,:,2]/rr+.12*rr)
        np.testing.assert_allclose(result,expected,atol=2e-14)

    def test_regular_shift_advection_has_finite_parity_axis_limit(self):
        radius=np.array((0.,.2));q=np.zeros((1,2,9))
        q[:,:,0]=.3;q[:,:,5]=-.2
        source=np.array([[[.4,-.1,.5],[.42,-.08,.49]]])
        source_z=np.array([[[.07,.03,-.02],[.07,.03,-.02]]])
        radial_scaled=np.array([[[0.,0.,0.],[.01,-.02,.03]]])
        inverse_z=np.full((1,2),1.1);inverse_r=np.full((1,2),.9)
        result=regular_so3_background_source_shift_advection(
            q,radius,source,source_z,radial_scaled,inverse_z,inverse_r,
        )
        expected_axis=1.1*.3*source_z[0,0]
        expected_axis[2]+=.9*(-.2)*source[0,0,2]
        self.assertTrue(np.all(np.isfinite(result)))
        np.testing.assert_allclose(result[0,0],expected_axis,atol=2e-14)

    def test_anchored_damped_wave_target_matches_full_adm_linearization(self):
        radius=np.array((.3,.8));lapse=np.array([[.72,.81]])
        compact=np.array([[1.15,1.08]]);radial=np.array([[.93,1.04]])
        angular=np.array([[.89,.98]])
        rng=np.random.default_rng(381);q=rng.normal(size=(1,2,9))
        result=regular_so3_anchored_damped_wave_target(
            q,radius,lapse,compact,radial,angular,.4,.7,.5,
        )
        expected=np.empty_like(result);step=2e-6
        for j,r in enumerate(radius):
            background=np.diag((
                -lapse[0,j]**2,compact[0,j]**2,radial[0,j]**2,
                angular[0,j]**2,angular[0,j]**2,
            ))
            perturbation=regular_so3_perturbation_jets(r,q[0,j])["metric"]
            def full_target(sign):
                metric=background+sign*step*perturbation
                spatial=metric[1:,1:];shift_covector=metric[0,1:]
                shift=np.linalg.solve(spatial,shift_covector)
                current_lapse=np.sqrt(-metric[0,0]+shift_covector@shift)
                logarithm=(
                    .5*(np.log(np.linalg.det(spatial))-np.log(np.linalg.det(background[1:,1:])))
                    -np.log(current_lapse/lapse[0,j])
                )
                normal=np.array((-current_lapse,0.,0.,0.,0.))
                # Sign reversal from H^LS=-Gamma to the project H=+Gamma.
                return -.4*logarithm*normal+.7/current_lapse*(metric[:,1:]@shift)
            derivative=(full_target(1)-full_target(-1))/(2*step)
            expected[0,j]=(derivative[0],derivative[1],derivative[2]/r)
        np.testing.assert_allclose(result,expected,rtol=3e-9,atol=3e-10)

    def test_state_dependent_target_recomputes_from_metric(self):
        radius=np.array((0.,.5));shape=(1,2);one=np.ones(shape)
        target=RegularSO3AnchoredDampedWaveTarget(
            radius,one,1.2*one,.9*one,1.1*one,.3,.4,
        )
        q=np.zeros((*shape,9));q[...,2]=.2;q[...,5]=-.1
        first=target.evaluate(q);second=target.evaluate(2*q)
        self.assertGreater(np.linalg.norm(first),0.)
        np.testing.assert_allclose(second,2*first,atol=2e-14)

    def test_nonlinear_anchored_target_has_linear_matrix_as_first_variation(self):
        radius=np.array((.3,.8));shape=(1,2)
        background=np.zeros((*shape,9))
        background[...,2]=-np.array(((.72,.81),))**2
        background[...,3]=np.array(((.89,.98),))**2
        background[...,6]=np.array(((1.15,1.08),))**2
        radial=np.array(((.93,1.04),))
        background[...,4]=(radial**2-background[...,3])/radius[None,:]**2
        source=np.array([[[.2,-.1,.04],[.15,-.08,.03]]])
        rng=np.random.default_rng(618);perturbation=.1*rng.normal(size=background.shape)
        step=1e-6
        plus=regular_so3_nonlinear_anchored_damped_wave_target(
            background+step*perturbation,background,source,radius,.4,.7,.5,
        )
        minus=regular_so3_nonlinear_anchored_damped_wave_target(
            background-step*perturbation,background,source,radius,.4,.7,.5,
        )
        derivative=(plus-minus)/(2*step)
        expected=regular_so3_anchored_damped_wave_target(
            perturbation,radius,np.array(((.72,.81),)),
            np.array(((1.15,1.08),)),radial,np.array(((.89,.98),)),.4,.7,.5,
        )
        np.testing.assert_allclose(derivative,expected,rtol=2e-8,atol=2e-9)
        np.testing.assert_allclose(
            regular_so3_nonlinear_anchored_damped_wave_target(
                background,background,source,radius,.4,.7,.5,
            ),source,atol=2e-14,
        )

    def test_live_source_shift_advection_includes_regular_axis_basis_term(self):
        radius=np.array((0.,.4));q=np.zeros((1,2,9))
        q[...,2]=-1.;q[...,3]=1.;q[...,6]=1.
        q[...,0]=.2;q[...,5]=-.3
        source=np.array([[[.4,-.2,.5],[.41,-.18,.48]]])
        source_z=np.array([[[.1,.03,-.02],[.1,.03,-.02]]])
        source_r=np.array([[[0.,0.,0.],[.04,.05,-.06]]])
        result=regular_so3_live_source_shift_advection(
            q,radius,source,source_z,source_r,
        )
        np.testing.assert_allclose(
            result[0,0],.2*source_z[0,0]-.3*np.array((0.,0.,source[0,0,2])),
            atol=2e-14,
        )
        expected=.2*source_z[0,1]-.12*source_r[0,1]
        expected[2]+=-.3*source[0,1,2]
        np.testing.assert_allclose(result[0,1],expected,atol=2e-14)

    def test_zero_shift_driver_matches_exact_decay_and_rk4_order(self):
        source=np.array((.7,-.2,.1));memory=np.array((.3,.4,-.5));target=np.array((.1,.1,.1))
        mu=1.7;eta=.8;final=.6
        exact_memory=memory*np.exp(-eta*final)
        exact_error=(source-target)*np.exp(-mu*final)+memory*(
            np.exp(-eta*final)-np.exp(-mu*final)
        )/(mu-eta)
        exact_source=target+exact_error
        errors=[]
        for steps in (4,8,16):
            result=integrate_source_driver(source,memory,target,final,final/steps,mu,eta)
            errors.append(np.linalg.norm(np.r_[
                result["source"]-exact_source,result["memory"]-exact_memory,
            ]))
        rates=np.log(np.array(errors[:-1])/errors[1:])/np.log(2.)
        self.assertGreater(min(rates),3.8)
        self.assertLess(errors[-1],2e-7)

    def test_driver_characteristic_speeds_are_real(self):
        result=first_order_driver_characteristic_speeds(.8,.13)
        np.testing.assert_allclose(result["metric_wave"],(-.93,.67))
        self.assertAlmostEqual(result["source"],-.13)
        self.assertAlmostEqual(result["memory"],0.)
        self.assertTrue(result["all_real"])

    def test_bjorhus_boundary_drives_only_source_toward_data(self):
        source=np.array((.7,-.2,.4));target=np.array((.1,.3,-.1))
        result=bjorhus_source_boundary_rhs(source,target,2.5)
        np.testing.assert_allclose(result,-2.5*(source-target))

    def test_project_constraint_feedback_has_decay_sign(self):
        constraint=np.array((.3,-.2,.1));rate=1.7
        source_dot=bjorhus_constraint_boundary_rhs(constraint,rate)
        # For frozen Gamma, C=Gamma-H and therefore Cdot=-Hdot.
        np.testing.assert_allclose(-source_dot,-rate*constraint)

    def test_driver_wrapper_accepts_nonhomogeneous_incoming_boundary_data(self):
        z=np.linspace(0,1,5);r=np.linspace(0,1,7);one=np.ones((5,7))
        wall=np.zeros((7,7,7))
        wave=AxisymmetricVariableReducedWaveIBVP(
            z,r,one,one,one,wall,wall,np.ones(7),np.ones(7),dirichlet_fields=2,
        )
        zero=np.zeros((5,7,9,3));first=np.zeros((3,5,7,9,3))
        driver=AxisymmetricDrivenGHWaveIBVP(wave,zero,first,1.2,.7)
        q=np.zeros((5,7,9));source=np.full((5,7,3),.6);memory=np.zeros_like(source)
        target=np.zeros_like(source);boundary_target=np.zeros_like(source)
        boundary_target[:,-1]=np.array((.2,-.1,.3));mask=np.zeros((5,7),dtype=bool);mask[:,-1]=True
        final=.2;mu_boundary=1.8
        result=driver.integrate(
            q,q,source,memory,target,final,.002,
            driver_boundary_target=boundary_target,
            driver_boundary_incoming_mask=mask,driver_boundary_rate=mu_boundary,
        )
        expected_interior=.6*np.exp(-1.2*final)
        expected_boundary=boundary_target[:,-1]+(
            source[:,-1]-boundary_target[:,-1]
        )*np.exp(-mu_boundary*final)
        np.testing.assert_allclose(result["source"][:,:-1],expected_interior,atol=2e-12)
        np.testing.assert_allclose(result["source"][:,-1],expected_boundary,atol=2e-12)
        np.testing.assert_allclose(result["memory"],0.,atol=1e-14)

    def test_axisymmetric_driver_wrapper_evolves_six_auxiliary_fields(self):
        z=np.linspace(0,1,5);r=np.linspace(0,1,7);one=np.ones((5,7))
        wall=np.zeros((7,7,7))
        wave=AxisymmetricVariableReducedWaveIBVP(
            z,r,one,one,one,wall,wall,np.ones(7),np.ones(7),dirichlet_fields=2,
        )
        zero=np.zeros((5,7,9,3));first=np.zeros((3,5,7,9,3))
        driver=AxisymmetricDrivenGHWaveIBVP(wave,zero,first,1.2,.7)
        q=np.zeros((5,7,9));h=np.ones((5,7,3));theta=.2*np.ones_like(h);target=.1*np.ones_like(h)
        result=driver.integrate(q,q,h,theta,target,.3,.01)
        exact_theta=theta*np.exp(-.7*.3)
        exact_h=target+(h-target)*np.exp(-1.2*.3)+theta*(
            np.exp(-.7*.3)-np.exp(-1.2*.3)
        )/(1.2-.7)
        np.testing.assert_allclose(result["source"],exact_h,atol=2e-9)
        np.testing.assert_allclose(result["memory"],exact_theta,atol=2e-9)
        np.testing.assert_allclose(result["position"],0.,atol=1e-14)


if __name__=="__main__":unittest.main()
