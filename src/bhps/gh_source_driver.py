"""Regular SO(3) generalized-harmonic source fields and first-order driver.

The independent source coefficients are ``(H_0,H_z,h_r)`` with
``H_i=h_r x_i``.  The Lindblom--Szilagyi first-order driver adds a memory
field ``theta_a`` and, in coordinate form, is

``H_t - N^k H_k = -mu (H-F) + theta`` and
``theta_t + eta theta = -eta N^k H_k``.

The project convention is ``H_a=+Gamma_a`` rather than the
Lindblom--Szilagyi convention ``H_a=-Gamma_a``.  Published target functions
must therefore be sign reversed before they are used here.  The module also
provides the regular linearization of a background-anchored damped-wave target
and constraint-preserving feedback on its incoming source characteristic.
"""

from __future__ import annotations

import numpy as np

from bhps.axisymmetric_reduced_wave_evolution import axisymmetric_rectangular_lower_order_matrices
from bhps.linearized_gh_einstein_scalar import linearized_reduced_einstein_two_scalar_residual
from bhps.regular_so3_gh_reduction import pack_regular_so3_residual


SOURCE_FIELD_ORDER=("delta_H_0","delta_H_z","delta_h_r=delta_H_r/r")
DRIVER_FIELD_ORDER=SOURCE_FIELD_ORDER+("theta_0","theta_z","theta_r/r")


def regular_so3_anchored_damped_wave_target_matrix(
    radius,lapse,compact_scale,radial_scale,angular_scale,
    mu_lapse,mu_shift,determinant_power=.5,
):
    """Return the ``3 x 9`` linear anchored damped-wave target map.

    The underlying Lindblom--Szilagyi target is

    ``F^LS_a=mu_L log(gamma**p/N) t_a-mu_S N^-1 g_ai N^i``.

    This code uses ``H_a=+Gamma_a=-H^LS_a``, so the target is sign reversed.
    The logarithm is normalized by its static background value, which keeps
    the corrected fold at the exact zero-perturbation fixed point.  The
    spatial determinant is four-dimensional: one compact direction and the
    three SO(3)-symmetric brane directions.
    """
    r=np.asarray(radius,dtype=float);alpha=np.asarray(lapse,dtype=float)
    compact=np.asarray(compact_scale,dtype=float)
    radial=np.asarray(radial_scale,dtype=float);angular=np.asarray(angular_scale,dtype=float)
    try:
        shape=np.broadcast_shapes(alpha.shape,compact.shape,radial.shape,angular.shape)
    except ValueError as error:
        raise ValueError("background target fields have incompatible shapes") from error
    alpha=np.broadcast_to(alpha,shape);compact=np.broadcast_to(compact,shape)
    radial=np.broadcast_to(radial,shape);angular=np.broadcast_to(angular,shape)
    if r.ndim!=1 or shape[-1:]!=(len(r),) or np.any(r<0):
        raise ValueError("radius must match the last background-grid dimension")
    mu_lapse=float(mu_lapse);mu_shift=float(mu_shift);power=float(determinant_power)
    if mu_lapse<0 or mu_shift<0 or power<0 or np.any(alpha<=0) or np.any(compact<=0) or np.any(radial<=0) or np.any(angular<=0):
        raise ValueError("invalid damped-wave target parameters")
    matrix=np.zeros((*shape,3,9))
    # Project-convention time target: +mu_L alpha delta log(gamma^p/N).
    matrix[...,0,2]=mu_lapse/(2*alpha)
    matrix[...,0,6]=mu_lapse*alpha*power/compact**2
    matrix[...,0,3]=mu_lapse*alpha*power*(radial**-2+2*angular**-2)
    matrix[...,0,4]=mu_lapse*alpha*power*r**2/radial**2
    # Project-convention sign reversal of the published shift target.
    matrix[...,1,0]=mu_shift/alpha
    matrix[...,2,5]=mu_shift/alpha
    return matrix


def regular_so3_anchored_damped_wave_target(
    metric_perturbation,radius,lapse,compact_scale,radial_scale,angular_scale,
    mu_lapse,mu_shift,determinant_power=.5,
):
    """Evaluate the regular linear background-anchored damped-wave target."""
    q=np.asarray(metric_perturbation,dtype=float)
    matrix=regular_so3_anchored_damped_wave_target_matrix(
        radius,lapse,compact_scale,radial_scale,angular_scale,
        mu_lapse,mu_shift,determinant_power,
    )
    if q.shape!=matrix.shape[:-2]+(9,):
        raise ValueError("metric perturbation and target background must match")
    return np.einsum("...ab,...b->...a",matrix,q)


def regular_so3_nonlinear_anchored_damped_wave_target(
    metric_state,background_metric_state,background_source,radius,
    mu_lapse,mu_shift,determinant_power=.5,
):
    """Return the exact nonlinear background-anchored damped-wave target.

    ``metric_state`` and ``background_metric_state`` use the nine regular
    fields in ``regular_so3_gh_reduction.FIELD_ORDER``.  The returned fields
    are the three regular covector coefficients ``(H_0,H_z,H_r/r)``.  The
    target equals ``background_source`` on the reference metric and its first
    variation is :func:`regular_so3_anchored_damped_wave_target_matrix`.

    The project convention is ``H=+Gamma``.  Thus the lapse part is the sign
    reverse of the published Lindblom--Szilagyi ``H=-Gamma`` target.
    """
    q=np.asarray(metric_state,dtype=float)
    q0=np.asarray(background_metric_state,dtype=float)
    source0=np.asarray(background_source,dtype=float)
    r=np.asarray(radius,dtype=float)
    if (
        q.shape!=q0.shape or q.shape[-1]!=9
        or source0.shape!=q.shape[:-1]+(3,)
        or r.ndim!=1 or q.shape[-2]!=len(r)
    ):
        raise ValueError("invalid nonlinear anchored-target fields")
    mu_lapse=float(mu_lapse);mu_shift=float(mu_shift)
    power=float(determinant_power)
    if mu_lapse<0 or mu_shift<0 or power<0 or np.any(r<0):
        raise ValueError("invalid nonlinear damped-wave target parameters")
    radius=r.reshape((1,)*(q.ndim-2)+(len(r),))

    def spatial_data(state):
        radial=state[...,3]+radius**2*state[...,4]
        compact_radial_determinant=(
            state[...,6]*radial-radius**2*state[...,1]**2
        )
        spatial_determinant=compact_radial_determinant*state[...,3]**2
        return radial,compact_radial_determinant,spatial_determinant

    radial,detector,spatial_determinant=spatial_data(q)
    _,background_detector,background_spatial_determinant=spatial_data(q0)
    if (
        np.any(detector<=0) or np.any(background_detector<=0)
        or np.any(spatial_determinant<=0)
        or np.any(background_spatial_determinant<=0)
    ):
        raise ValueError("damped-wave target requires a positive spatial metric")
    beta_z=(radial*q[...,0]-radius**2*q[...,1]*q[...,5])/detector
    beta_r_coefficient=(-q[...,1]*q[...,0]+q[...,6]*q[...,5])/detector
    shift_norm=q[...,0]*beta_z+radius**2*q[...,5]*beta_r_coefficient
    lapse_squared=-q[...,2]+shift_norm
    background_lapse_squared=-q0[...,2]
    if np.any(lapse_squared<=0) or np.any(background_lapse_squared<=0):
        raise ValueError("damped-wave target requires a spacelike time foliation")
    lapse=np.sqrt(lapse_squared);background_lapse=np.sqrt(background_lapse_squared)
    logarithm=(
        power*np.log(spatial_determinant/background_spatial_determinant)
        -np.log(lapse/background_lapse)
    )
    delta=np.empty_like(source0)
    delta[...,0]=mu_lapse*lapse*logarithm+mu_shift*shift_norm/lapse
    delta[...,1]=mu_shift*q[...,0]/lapse
    delta[...,2]=mu_shift*q[...,5]/lapse
    return source0+delta


def regular_so3_live_source_shift_advection(
    metric_state,radius,source,source_z,source_r,
):
    """Return exact nonlinear shift advection of a regular source covector.

    The third regular source coefficient represents ``H_i=h_r x_i``.  Its
    radial directional derivative therefore includes the Cartesian-basis
    contribution ``beta_r_coefficient*h_r`` at and away from the axis.
    """
    q=np.asarray(metric_state,dtype=float)
    h=np.asarray(source,dtype=float)
    hz=np.asarray(source_z,dtype=float)
    hr=np.asarray(source_r,dtype=float)
    r=np.asarray(radius,dtype=float)
    if (
        q.shape[-1]!=9 or h.shape!=q.shape[:-1]+(3,)
        or hz.shape!=h.shape or hr.shape!=h.shape
        or r.ndim!=1 or q.shape[-2]!=len(r)
    ):
        raise ValueError("invalid live source-advection fields")
    radius=r.reshape((1,)*(q.ndim-2)+(len(r),))
    radial=q[...,3]+radius**2*q[...,4]
    determinant=q[...,6]*radial-radius**2*q[...,1]**2
    if np.any(determinant<=0):
        raise ValueError("source advection requires a positive spatial metric")
    beta_z=(radial*q[...,0]-radius**2*q[...,1]*q[...,5])/determinant
    beta_r_coefficient=(-q[...,1]*q[...,0]+q[...,6]*q[...,5])/determinant
    beta_r=radius*beta_r_coefficient
    result=beta_z[...,None]*hz+beta_r[...,None]*hr
    result[...,2]+=beta_r_coefficient*h[...,2]
    return result


class RegularSO3AnchoredDampedWaveTarget:
    """State-dependent target callable for a fixed corrected background."""

    def __init__(
        self,radius,lapse,compact_scale,radial_scale,angular_scale,
        mu_lapse,mu_shift,determinant_power=.5,
    ):
        self.matrix=regular_so3_anchored_damped_wave_target_matrix(
            radius,lapse,compact_scale,radial_scale,angular_scale,
            mu_lapse,mu_shift,determinant_power,
        )
        self.mu_lapse=float(mu_lapse);self.mu_shift=float(mu_shift)
        self.determinant_power=float(determinant_power)

    def evaluate(self,metric_perturbation,time=0.):
        del time
        q=np.asarray(metric_perturbation,dtype=float)
        if q.shape!=self.matrix.shape[:-2]+(9,):
            raise ValueError("metric perturbation and target background must match")
        return np.einsum("...ab,...b->...a",self.matrix,q)


def regular_so3_source_jets(radius,values,first=None):
    """Expand three regular source coefficients into Cartesian covector jets."""
    radius=float(radius);values=np.asarray(values,dtype=float)
    first=np.zeros((3,3)) if first is None else np.asarray(first,dtype=float)
    if radius<0 or values.shape!=(3,) or first.shape!=(3,3):
        raise ValueError("invalid regular source jets")
    covector=np.array((values[0],values[1],radius*values[2],0.,0.))
    derivative=np.zeros((5,5))
    # Scalar covector components H_0 and H_z.
    for field,component in ((0,0),(1,1)):
        derivative[0,component]=first[0,field]
        derivative[1,component]=first[1,field]
        derivative[2,component]=first[2,field]
    # SO(3) vector covector H_i=h_r x_i.
    derivative[0,2]=radius*first[0,2]
    derivative[1,2]=radius*first[1,2]
    derivative[2,2]=values[2]+radius*first[2,2]
    derivative[3,3]=values[2];derivative[4,4]=values[2]
    return {"covector":covector,"first":derivative}


def _zero_perturbation():
    return {
        "metric":np.zeros((5,5)),"metric_first":np.zeros((5,5,5)),
        "metric_second":np.zeros((5,5,5,5)),
        "phi":0.,"phi_first":np.zeros(5),"phi_second":np.zeros((5,5)),
        "chi":0.,"chi_first":np.zeros(5),"chi_second":np.zeros((5,5)),
    }


def regular_so3_source_coupling_matrices(
    background,radius,mass_squared=0.,potential_offset=-6.,kappa5_squared=1.,
    constraint_damping=0.,constraint_damping_rho=0.,
):
    """Extract source-value and source-first coupling into nine wave rows.

    The returned evolution matrices enter the metric/scalar acceleration with
    a positive sign.  Their last two rows vanish because ``H_a`` couples only
    to the reduced Einstein equations.
    """
    radius=float(radius)
    if radius<=0:raise ValueError("extract off axis and take the regular parity limit")
    normalization=np.r_[np.full(7,-2.),np.ones(2)]
    zero=np.zeros((9,3));first=np.zeros((3,9,3))
    for column in range(3):
        values=np.zeros(3);values[column]=1.
        source=regular_so3_source_jets(radius,values)
        perturbation=_zero_perturbation()
        perturbation["gauge_source_covector"]=source["covector"]
        perturbation["gauge_source_first"]=source["first"]
        result=linearized_reduced_einstein_two_scalar_residual(
            background,perturbation,mass_squared=mass_squared,
            potential_offset=potential_offset,kappa5_squared=kappa5_squared,
            constraint_damping=constraint_damping,
            constraint_damping_rho=constraint_damping_rho,
        )
        zero[:,column]=normalization*pack_regular_so3_residual(
            result["metric_residual"],result["phi_residual"],result["chi_residual"],radius,
        )
        for direction in range(3):
            reduced_first=np.zeros((3,3));reduced_first[direction,column]=1.
            source=regular_so3_source_jets(radius,np.zeros(3),reduced_first)
            perturbation=_zero_perturbation()
            perturbation["gauge_source_covector"]=source["covector"]
            perturbation["gauge_source_first"]=source["first"]
            result=linearized_reduced_einstein_two_scalar_residual(
                background,perturbation,mass_squared=mass_squared,
                potential_offset=potential_offset,kappa5_squared=kappa5_squared,
                constraint_damping=constraint_damping,
                constraint_damping_rho=constraint_damping_rho,
            )
            first[direction,:,column]=normalization*pack_regular_so3_residual(
                result["metric_residual"],result["phi_residual"],result["chi_residual"],radius,
            )
    alpha_squared=-1/np.asarray(background["metric"],dtype=float)[0,0]
    return {
        "source_field_order":SOURCE_FIELD_ORDER,
        "operator_zero_matrix":zero,"operator_first_matrices":first,
        "evolution_zero_matrix":alpha_squared*zero,
        "evolution_first_matrices":alpha_squared*first,
        "constraint_damping_rate":float(constraint_damping),
        "constraint_damping_rho":float(constraint_damping_rho),
        "finite":bool(np.all(np.isfinite(zero)) and np.all(np.isfinite(first))),
    }


def source_driver_rhs(source,memory,target,mu,eta,shift_advection=None):
    """Evaluate the first-order driver for arrays ending in three fields."""
    source=np.asarray(source,dtype=float);memory=np.asarray(memory,dtype=float)
    target=np.asarray(target,dtype=float)
    advection=np.zeros_like(source) if shift_advection is None else np.asarray(shift_advection,dtype=float)
    if source.shape!=memory.shape or source.shape!=target.shape or source.shape!=advection.shape or source.shape[-1]!=3:
        raise ValueError("driver arrays must have matching trailing size three")
    mu=float(mu);eta=float(eta)
    if mu<=0 or eta<=0:raise ValueError("driver rates must be positive")
    return advection-mu*(source-target)+memory,-eta*(memory+advection)


def bjorhus_source_boundary_rhs(source,boundary_target,mu_boundary):
    """Return the incoming ``u^3_a=H_a`` Bjorhus boundary derivative.

    This is Eq. (E10) of Lindblom--Szilagyi.  Reversing both ``H`` and ``F``
    to the project's ``H=+Gamma`` convention leaves its form unchanged.
    The zero-speed ``u^4_a=theta_a+eta H_a`` field is not prescribed here.
    """
    source=np.asarray(source,dtype=float);target=np.asarray(boundary_target,dtype=float)
    rate=float(mu_boundary)
    if source.shape!=target.shape or source.shape[-1]!=3 or rate<=0:
        raise ValueError("invalid source-driver boundary data")
    return -rate*(source-target)


def bjorhus_constraint_boundary_rhs(constraint,mu_boundary):
    """Return project-convention constraint feedback for incoming ``H_a``.

    Lindblom--Szilagyi Eq. (E9) is written for ``H^LS=-Gamma``.  With the
    project variables ``H=-H^LS`` and ``C=Gamma-H``, it becomes
    ``partial_t H=+mu_B C`` and hence ``partial_t C=-mu_B C`` when the metric
    contribution is frozen.
    """
    constraint=np.asarray(constraint,dtype=float);rate=float(mu_boundary)
    if constraint.shape[-1]!=3 or rate<=0:
        raise ValueError("invalid constraint-feedback boundary data")
    return rate*constraint


def regular_so3_background_source_shift_advection(
    metric_perturbation,radius,background_source,background_source_z,
    background_source_radial_scaled,inverse_compact_metric,inverse_radial_metric,
):
    """Return ``delta N^k partial_k H_a`` in regular SO(3) variables.

    ``background_source_radial_scaled`` stores ``r partial_r(H_0,H_z,h_r)``.
    The shift coefficients follow from the zero-shift background:
    ``delta N^z=g^zz h_0z`` and
    ``delta N^r=r g^rr v_0``.  The final vector-source component includes the
    Cartesian basis derivative, giving ``g^rr v_0(h_r+r h_r,r)``.
    """
    q=np.asarray(metric_perturbation,dtype=float);r=np.asarray(radius,dtype=float)
    source=np.asarray(background_source,dtype=float);source_z=np.asarray(background_source_z,dtype=float)
    source_r_scaled=np.asarray(background_source_radial_scaled,dtype=float)
    inverse_z=np.asarray(inverse_compact_metric,dtype=float);inverse_r=np.asarray(inverse_radial_metric,dtype=float)
    if q.shape[-1]!=9 or source.shape!=q.shape[:-1]+(3,) or source_z.shape!=source.shape or source_r_scaled.shape!=source.shape:
        raise ValueError("invalid regular shift-advection fields")
    if inverse_z.shape!=q.shape[:-1] or inverse_r.shape!=q.shape[:-1] or r.shape!=(q.shape[-2],):
        raise ValueError("invalid metric or radial data for shift advection")
    beta_z=inverse_z*q[...,0];beta_r_coefficient=inverse_r*q[...,5]
    advection=beta_z[...,None]*source_z+beta_r_coefficient[...,None]*source_r_scaled
    advection[...,2]+=beta_r_coefficient*source[...,2]
    return advection


def integrate_source_driver(source,memory,target,final_time,time_step,mu,eta):
    """RK4 integration of the fixed-target, zero-shift source driver."""
    source=np.asarray(source,dtype=float).copy();memory=np.asarray(memory,dtype=float).copy()
    target=np.asarray(target,dtype=float)
    final=float(final_time);requested=float(time_step)
    if final<0 or requested<=0:raise ValueError("invalid integration interval")
    steps=max(1,int(np.ceil(final/requested)));dt=final/steps if final else 0.
    def rhs(h,theta):return source_driver_rhs(h,theta,target,mu,eta)
    for _ in range(steps):
        k1h,k1t=rhs(source,memory)
        k2h,k2t=rhs(source+.5*dt*k1h,memory+.5*dt*k1t)
        k3h,k3t=rhs(source+.5*dt*k2h,memory+.5*dt*k2t)
        k4h,k4t=rhs(source+dt*k3h,memory+dt*k3t)
        source+=dt*(k1h+2*k2h+2*k3h+k4h)/6
        memory+=dt*(k1t+2*k2t+2*k3t+k4t)/6
    return {"source":source,"memory":memory,"steps":steps,"time_step":dt}


def first_order_driver_characteristic_speeds(lapse,normal_shift):
    """Return the real normal speeds of the GH metric/source-driver blocks."""
    lapse=float(lapse);shift=float(normal_shift)
    if lapse<=0:raise ValueError("lapse must be positive")
    return {
        "metric_wave":np.array((-shift-lapse,-shift+lapse)),
        "source":float(-shift),"memory":0.,"all_real":True,
    }


class AxisymmetricDrivenGHWaveIBVP:
    """Couple a nine-field wave IBVP to the six-field first-order GH driver.

    The target may be prescribed in space and time or supplied as a live
    state-dependent object with an ``evaluate(metric,time)`` method. Source
    derivatives are coupled weakly through rectangular Q1 matrices, while the
    source and memory fields use the first-order driver at the same nodes.
    """

    def __init__(
        self,wave_system,source_zero_matrices,source_first_matrices,
        mu,eta,radial_first_is_scaled=False,background_source_shift_data=None,
    ):
        self.wave=wave_system;self.mu=float(mu);self.eta=float(eta)
        if self.wave.field_count!=9 or self.mu<=0 or self.eta<=0:
            raise ValueError("driver requires a nine-field wave system and positive rates")
        zero=np.asarray(source_zero_matrices,dtype=float)
        first=np.asarray(source_first_matrices,dtype=float)
        expected=(self.wave.nz,self.wave.nr,9,3)
        if zero.shape!=expected or first.shape!=(3,*expected):
            raise ValueError("source coupling fields have incompatible shapes")
        self.coupling=axisymmetric_rectangular_lower_order_matrices(
            self.wave.z,self.wave.r,self.wave.mass_weight,zero,first,
            radial_first_is_scaled=radial_first_is_scaled,
        )
        self.background_source_shift_data=None
        if background_source_shift_data is not None:
            data={key:np.asarray(value,dtype=float) for key,value in background_source_shift_data.items()}
            expected_source=(self.wave.nz,self.wave.nr,3);expected_scalar=(self.wave.nz,self.wave.nr)
            if (
                data.get("source",np.empty(0)).shape!=expected_source
                or data.get("z_first",np.empty(0)).shape!=expected_source
                or data.get("radial_first_scaled",np.empty(0)).shape!=expected_source
                or data.get("inverse_compact_metric",np.empty(0)).shape!=expected_scalar
                or data.get("inverse_radial_metric",np.empty(0)).shape!=expected_scalar
            ):raise ValueError("background source shift data have incompatible shapes")
            self.background_source_shift_data=data

    def _target(self,target,time,position):
        if hasattr(target,"evaluate"):
            values=np.asarray(target.evaluate(position,float(time)),dtype=float)
        elif callable(target):
            zz,rr=np.meshgrid(self.wave.z,self.wave.r,indexing="ij")
            values=np.asarray(target(float(time),zz,rr),dtype=float)
        else:values=np.asarray(target,dtype=float)
        if values.shape!=(self.wave.nz,self.wave.nr,3):
            raise ValueError("driver target has the wrong shape")
        return values

    def _driver_boundary_values(self,boundary_target,time):
        if callable(boundary_target):
            zz,rr=np.meshgrid(self.wave.z,self.wave.r,indexing="ij")
            values=np.asarray(boundary_target(float(time),zz,rr),dtype=float)
        else:values=np.asarray(boundary_target,dtype=float)
        if values.shape!=(self.wave.nz,self.wave.nr,3):
            raise ValueError("driver boundary target has the wrong shape")
        return values

    def _driver_boundary_constraints(
        self,boundary_constraint,time,position,velocity,source,
    ):
        if hasattr(boundary_constraint,"evaluate"):
            values=np.asarray(boundary_constraint.evaluate(
                position,velocity,source,float(time),
            ),dtype=float)
        elif callable(boundary_constraint):
            values=np.asarray(boundary_constraint(
                float(time),position,velocity,source,
            ),dtype=float)
        else:values=np.asarray(boundary_constraint,dtype=float)
        if values.shape!=(self.wave.nz,self.wave.nr,3):
            raise ValueError("driver boundary constraint has the wrong shape")
        return values

    def _solve_wave_load(self,flat_load):
        load=np.asarray(flat_load,dtype=float).reshape(self.wave.nodes,9)
        result=np.zeros_like(load);split=self.wave.dirichlet_fields
        if split:
            result[self.wave.gauge_free,:split]=self.wave._gauge_mass.solve(
                load[self.wave.gauge_free,:split]
            )
        result[self.wave.robin_free,split:]=self.wave._robin_mass.solve(
            load[self.wave.robin_free,split:]
        )
        return result.reshape(self.wave.nz,self.wave.nr,9)

    def rhs(
        self,time,position,velocity,source,memory,target,
        volume_source=None,left_boundary_data=None,right_boundary_data=None,
        driver_boundary_target=None,driver_boundary_incoming_mask=None,
        driver_boundary_rate=None,driver_boundary_constraint=None,
        metric_boundary_constraint_feedback=None,
    ):
        target_values=self._target(target,time,position)
        advection=None
        if self.background_source_shift_data is not None:
            data=self.background_source_shift_data
            advection=regular_so3_background_source_shift_advection(
                position,self.wave.r,data["source"],data["z_first"],
                data["radial_first_scaled"],data["inverse_compact_metric"],
                data["inverse_radial_metric"],
            )
        source_dot,memory_dot=source_driver_rhs(
            source,memory,target_values,self.mu,self.eta,advection,
        )
        if driver_boundary_target is not None and driver_boundary_constraint is not None:
            raise ValueError("choose target or constraint feedback, not both")
        if driver_boundary_target is not None or driver_boundary_constraint is not None:
            mask=np.asarray(driver_boundary_incoming_mask,dtype=bool)
            if mask.shape!=(self.wave.nz,self.wave.nr):
                raise ValueError("driver incoming mask has the wrong shape")
            if driver_boundary_target is not None:
                boundary_values=self._driver_boundary_values(driver_boundary_target,time)
                boundary_dot=bjorhus_source_boundary_rhs(
                    source,boundary_values,driver_boundary_rate,
                )
            else:
                constraints=self._driver_boundary_constraints(
                    driver_boundary_constraint,time,position,velocity,source,
                )
                boundary_dot=bjorhus_constraint_boundary_rhs(
                    constraints,driver_boundary_rate,
                )
            source_dot=np.array(source_dot,copy=True)
            source_dot[mask]=boundary_dot[mask]
        wave_acceleration=self.wave.acceleration(
            time,position,volume_source,left_boundary_data,right_boundary_data,
            velocity=velocity,
        )
        flat_source=np.asarray(source).reshape(-1)
        flat_source_dot=np.asarray(source_dot).reshape(-1)
        source_load=(
            self.coupling["reaction"]@flat_source
            +self.coupling["time_first"]@flat_source_dot
            +self.coupling["z_first"]@flat_source
            +self.coupling["r_first"]@flat_source
        )
        wave_acceleration+=self._solve_wave_load(source_load)
        if metric_boundary_constraint_feedback is not None:
            boundary=metric_boundary_constraint_feedback.evaluate(
                position,velocity,source,float(time),
            )
            characteristic=np.asarray(
                boundary["characteristic_correction"],dtype=float,
            )
            mask=np.asarray(boundary["incoming_mask"],dtype=bool)
            lapse=np.asarray(boundary["lapse"],dtype=float)
            if (
                characteristic.shape!=(self.wave.nz,self.wave.nr,9)
                or mask.shape!=(self.wave.nz,self.wave.nr)
                or lapse.shape!=(self.wave.nz,)
            ):
                raise ValueError("metric boundary feedback has incompatible shape")
            wave_acceleration=np.array(wave_acceleration,copy=True)
            if "weak_radial_flux" in boundary:
                flux=np.asarray(boundary["weak_radial_flux"],dtype=float)
                if flux.shape!=(self.wave.nz,9):
                    raise ValueError("weak radial flux has incompatible shape")
                weak_load=self.wave.outer_radial_flux_load(flux,mask[:,-1])
                # Use the diagonal Q1 norm for the SAT lift.  Unlike the
                # consistent inverse, it is local and cannot silently spread
                # an artificial-face penalty onto excluded wall-corner rows.
                weak_acceleration=(
                    weak_load/self.wave.lumped_mass[:,None]
                ).reshape(self.wave.nz,self.wave.nr,9)
                wave_acceleration+=weak_acceleration
            else:
                # For zero background shift, Pi_ab=-partial_t psi_ab/N.  Hence an
                # additive correction delta(partial_t u^-)=B_ab is implemented by
                # delta(partial_t^2 psi_ab)=-N B_ab.
                acceleration_correction=-lapse[:,None,None]*characteristic
                wave_acceleration[mask]+=acceleration_correction[mask]
        return velocity,wave_acceleration,source_dot,memory_dot

    def integrate(
        self,position,velocity,source,memory,target,final_time,time_step,
        volume_source=None,left_boundary_data=None,right_boundary_data=None,
        diagnostic=None,diagnostic_stride=1,
        driver_boundary_target=None,driver_boundary_incoming_mask=None,
        driver_boundary_rate=None,driver_boundary_constraint=None,
        metric_boundary_constraint_feedback=None,
    ):
        q=np.asarray(position,dtype=float).copy();v=np.asarray(velocity,dtype=float).copy()
        h=np.asarray(source,dtype=float).copy();theta=np.asarray(memory,dtype=float).copy()
        if q.shape!=(self.wave.nz,self.wave.nr,9) or v.shape!=q.shape:
            raise ValueError("wave state has the wrong shape")
        if h.shape!=(self.wave.nz,self.wave.nr,3) or theta.shape!=h.shape:
            raise ValueError("driver state has the wrong shape")
        final=float(final_time);requested=float(time_step)
        if final<0 or requested<=0:raise ValueError("invalid integration interval")
        steps=max(1,int(np.ceil(final/requested)));dt=final/steps if final else 0.
        stride=int(diagnostic_stride)
        if stride<1:raise ValueError("diagnostic stride must be positive")
        def evaluate(time,state):
            return self.rhs(
                time,*state,target,volume_source,left_boundary_data,right_boundary_data,
                driver_boundary_target,driver_boundary_incoming_mask,driver_boundary_rate,
                driver_boundary_constraint,metric_boundary_constraint_feedback,
            )
        state=(q,v,h,theta);time=0.;records=[]
        if diagnostic is not None:records.append(diagnostic(time,*state))
        for step_index in range(steps):
            k1=evaluate(time,state)
            stage=tuple(value+.5*dt*slope for value,slope in zip(state,k1))
            k2=evaluate(time+.5*dt,stage)
            stage=tuple(value+.5*dt*slope for value,slope in zip(state,k2))
            k3=evaluate(time+.5*dt,stage)
            stage=tuple(value+dt*slope for value,slope in zip(state,k3))
            k4=evaluate(time+dt,stage)
            state=tuple(
                value+dt*(a+2*b+2*c+d)/6
                for value,a,b,c,d in zip(state,k1,k2,k3,k4)
            )
            time+=dt
            if diagnostic is not None and ((step_index+1)%stride==0 or step_index+1==steps):
                records.append(diagnostic(time,*state))
        return {
            "position":state[0],"velocity":state[1],"source":state[2],
            "memory":state[3],"steps":steps,"time_step":dt,
            "diagnostics":records,
        }
