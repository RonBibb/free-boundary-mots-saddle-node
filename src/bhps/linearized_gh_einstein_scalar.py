"""Pointwise generalized-harmonic Einstein--two-scalar linearization.

This module represents a metric and two scalar fields by their coordinate jets
at one regular point.  It evaluates the trace-reversed reduced equations

    R^H_ab = kappa^2 [Phi_a Phi_b + chi_a chi_b + 2 V(Phi) g_ab/3],
    Box_g Phi - V'(Phi) = 0,
    Box_g chi = 0,

in five dimensions.  The generalized-harmonic source and its first derivative
are external fixed data.  Linearization uses the complex-step derivative, so
the result contains every connection, curvature, and metric--matter term
without symbolic cancellation or division by the background scalar gradient.

The optional Gundlach--Lindblom constraint addition uses this module's
constraint convention ``C_a=Gamma_a-H_a``:

    kappa [n_(a C_b) - (1+rho) g_ab n^c C_c/2].

It is lower order in the reduced metric equations and therefore leaves their
scalar wave principal symbol unchanged.  A dynamical gauge-source driver is
still a separate evolution system.
"""

from __future__ import annotations

import numpy as np


def metric_geometry_from_jets(metric,metric_first,metric_second):
    """Return inverse metric, connection, its derivative, and Ricci tensor."""
    metric=np.asarray(metric);first=np.asarray(metric_first);second=np.asarray(metric_second)
    dimension=metric.shape[0]
    if (
        metric.shape!=(dimension,dimension)
        or first.shape!=(dimension,dimension,dimension)
        or second.shape!=(dimension,dimension,dimension,dimension)
    ):
        raise ValueError("metric jets have incompatible shapes")
    inverse=np.linalg.inv(metric)
    dtype=np.result_type(metric,first,second)
    inverse_first=-np.einsum("ua,eab,bc->euc",inverse,first,inverse)
    # C[c,l,r] = d_l g_cr + d_r g_cl - d_c g_lr.
    first_combination=(
        first.transpose(1,0,2)+first.transpose(1,2,0)-first
    )
    connection=.5*np.einsum("uc,clr->ulr",inverse,first_combination)
    # The same combination with one leading derivative index.
    second_combination=(
        second.transpose(0,2,1,3)+second.transpose(0,2,3,1)-second
    )
    connection_first=.5*(
        np.einsum("euc,clr->eulr",inverse_first,first_combination)
        +np.einsum("uc,eclr->eulr",inverse,second_combination)
    ).astype(dtype,copy=False)

    connection_trace=np.einsum("aca->c",connection)
    ricci=(
        np.einsum("aalr->lr",connection_first)
        -np.einsum("rclc->lr",connection_first)
        +np.einsum("clr,c->lr",connection,connection_trace)
        -np.einsum("cla,arc->lr",connection,connection)
    )

    contracted_upper=np.einsum("bc,abc->a",inverse,connection)
    contracted_upper_first=np.empty((dimension,dimension),dtype=connection.dtype)
    for derivative in range(dimension):
        contracted_upper_first[derivative]=(
            np.einsum("bc,abc->a",inverse_first[derivative],connection)
            +np.einsum("bc,abc->a",inverse,connection_first[derivative])
        )
    contracted_covector=metric@contracted_upper
    contracted_covector_first=np.empty_like(contracted_upper_first)
    for derivative in range(dimension):
        contracted_covector_first[derivative]=(
            first[derivative]@contracted_upper
            +metric@contracted_upper_first[derivative]
        )
    return {
        "inverse_metric":inverse,
        "inverse_metric_first":inverse_first,
        "connection":connection,
        "connection_first":connection_first,
        "ricci":ricci,
        "contracted_christoffel_upper":contracted_upper,
        "contracted_christoffel_covector":contracted_covector,
        "contracted_christoffel_covector_first":contracted_covector_first,
    }


def frozen_source_reduced_ricci(
    metric,metric_first,metric_second,gauge_source_covector,gauge_source_first,
    constraint_damping=0.0,constraint_damping_rho=0.0,time_normal_covector=None,
):
    """Return the fixed-source, optionally constraint-damped reduced Ricci.

    The future unit normal is inferred from the coordinate-time foliation
    unless ``time_normal_covector`` is supplied.  ``constraint_damping`` is
    the nonnegative Gundlach/Lindblom damping rate and ``rho=0`` is the
    standard choice that damps every nonconstant frozen Minkowski mode.
    """
    geometry=metric_geometry_from_jets(metric,metric_first,metric_second)
    source=np.asarray(gauge_source_covector);source_first=np.asarray(gauge_source_first)
    dimension=metric.shape[0]
    if source.shape!=(dimension,) or source_first.shape!=(dimension,dimension):
        raise ValueError("gauge-source jets have incompatible shapes")
    constraint=geometry["contracted_christoffel_covector"]-source
    constraint_first=geometry["contracted_christoffel_covector_first"]-source_first
    derivative=np.empty_like(constraint_first)
    for left in range(dimension):
        for right in range(dimension):
            derivative[left,right]=constraint_first[left,right]
            for contracted in range(dimension):
                derivative[left,right]-=geometry["connection"][contracted,left,right]*constraint[contracted]
    reduced=geometry["ricci"]-.5*(derivative+derivative.T)
    damping_rate=float(constraint_damping);rho=float(constraint_damping_rho)
    if damping_rate<0 or rho<=-1:
        raise ValueError("require nonnegative constraint damping and rho > -1")
    if time_normal_covector is None:
        inverse=geometry["inverse_metric"]
        if np.real(inverse[0,0])>=0:
            raise ValueError("coordinate-time slices must be spacelike")
        lapse=1/np.sqrt(-inverse[0,0])
        normal_covector=np.zeros(dimension,dtype=reduced.dtype)
        normal_covector[0]=-lapse
    else:
        normal_covector=np.asarray(time_normal_covector,dtype=reduced.dtype)
        if normal_covector.shape!=(dimension,):
            raise ValueError("time normal has incompatible shape")
        norm=np.einsum("a,ab,b->",normal_covector,geometry["inverse_metric"],normal_covector)
        if not np.allclose(norm,-1.,rtol=1e-10,atol=1e-12):
            raise ValueError("time normal must be unit timelike")
    normal_upper=geometry["inverse_metric"]@normal_covector
    normal_constraint=np.dot(normal_upper,constraint)
    damping_tensor=damping_rate*(
        .5*(np.outer(normal_covector,constraint)+np.outer(constraint,normal_covector))
        -.5*(1+rho)*metric*normal_constraint
    )
    reduced=reduced+damping_tensor
    return {
        **geometry,"gauge_constraint_covector":constraint,
        "gauge_constraint_first_covector":constraint_first,
        "time_normal_covector":normal_covector,
        "constraint_damping_tensor":damping_tensor,"reduced_ricci":reduced,
    }


def _scalar_wave(inverse,connection,gradient,hessian):
    covariant_hessian=np.asarray(hessian).copy()
    dimension=len(gradient)
    for left in range(dimension):
        for right in range(dimension):
            for contracted in range(dimension):
                covariant_hessian[left,right]-=connection[contracted,left,right]*gradient[contracted]
    return np.einsum("ab,ab->",inverse,covariant_hessian),covariant_hessian


def reduced_einstein_two_scalar_residual(
    metric,metric_first,metric_second,
    phi,phi_first,phi_second,chi,chi_first,chi_second,
    gauge_source_covector,gauge_source_first,
    mass_squared=0.0,potential_offset=-6.0,kappa5_squared=1.0,
    constraint_damping=0.0,constraint_damping_rho=0.0,
):
    """Evaluate the five-dimensional reduced Einstein--two-scalar residual."""
    metric=np.asarray(metric);dimension=metric.shape[0]
    if dimension!=5:raise ValueError("the trace-reversed potential coefficient is specialized to five dimensions")
    phi_first=np.asarray(phi_first);phi_second=np.asarray(phi_second)
    chi_first=np.asarray(chi_first);chi_second=np.asarray(chi_second)
    if (
        phi_first.shape!=(dimension,) or chi_first.shape!=(dimension,)
        or phi_second.shape!=(dimension,dimension) or chi_second.shape!=(dimension,dimension)
    ):
        raise ValueError("scalar jets have incompatible shapes")
    mass_squared=float(mass_squared);kappa=float(kappa5_squared)
    if mass_squared<0 or kappa<=0:raise ValueError("invalid physical coefficients")
    geometry=frozen_source_reduced_ricci(
        metric,metric_first,metric_second,gauge_source_covector,gauge_source_first,
        constraint_damping=constraint_damping,
        constraint_damping_rho=constraint_damping_rho,
    )
    potential=potential_offset+.5*mass_squared*phi**2
    potential_prime=mass_squared*phi
    matter=(
        np.outer(phi_first,phi_first)+np.outer(chi_first,chi_first)
        +(2/3)*potential*metric
    )
    phi_wave,phi_covariant_hessian=_scalar_wave(
        geometry["inverse_metric"],geometry["connection"],phi_first,phi_second,
    )
    chi_wave,chi_covariant_hessian=_scalar_wave(
        geometry["inverse_metric"],geometry["connection"],chi_first,chi_second,
    )
    return {
        **geometry,
        "metric_residual":geometry["reduced_ricci"]-kappa*matter,
        "phi_residual":phi_wave-potential_prime,
        "chi_residual":chi_wave,
        "phi_covariant_hessian":phi_covariant_hessian,
        "chi_covariant_hessian":chi_covariant_hessian,
        "potential":potential,
        "potential_prime":potential_prime,
    }


def solve_reduced_einstein_two_scalar_acceleration(
    metric,metric_first,metric_second,
    phi,phi_first,phi_second,chi,chi_first,chi_second,
    gauge_source_covector,gauge_source_first,
    mass_squared=0.0,potential_offset=-6.0,kappa5_squared=1.0,
    constraint_damping=0.0,constraint_damping_rho=0.0,
):
    """Solve the reduced equations for the coordinate-time accelerations.

    The supplied second-jet arrays provide the spatial and mixed derivatives;
    their ``[0,0]`` entries are ignored and solved.  Gauge-source value and
    first jets are held fixed at the current evolution stage.  This is a
    pointwise nonlinear RHS primitive, not a spatial discretization or time
    integrator.
    """
    metric=np.asarray(metric);metric_first=np.asarray(metric_first)
    trial_metric_second=np.asarray(metric_second).copy()
    trial_phi_second=np.asarray(phi_second).copy()
    trial_chi_second=np.asarray(chi_second).copy()
    dimension=metric.shape[0]
    if (
        metric.shape!=(dimension,dimension)
        or metric_first.shape!=(dimension,dimension,dimension)
        or trial_metric_second.shape!=(dimension,dimension,dimension,dimension)
        or trial_phi_second.shape!=(dimension,dimension)
        or trial_chi_second.shape!=(dimension,dimension)
    ):
        raise ValueError("spacetime jets have incompatible shapes")
    trial_metric_second[0,0]=0.
    trial_phi_second[0,0]=0.
    trial_chi_second[0,0]=0.
    residual=reduced_einstein_two_scalar_residual(
        metric,metric_first,trial_metric_second,
        phi,phi_first,trial_phi_second,chi,chi_first,trial_chi_second,
        gauge_source_covector,gauge_source_first,
        mass_squared,potential_offset,kappa5_squared,
        constraint_damping,constraint_damping_rho,
    )
    inverse_time=float(np.real(residual["inverse_metric"][0,0]))
    if inverse_time>=0 or abs(inverse_time)<1e-15:
        raise ValueError("coordinate-time slices must be spacelike and nondegenerate")
    metric_acceleration=2*np.asarray(residual["metric_residual"])/inverse_time
    phi_acceleration=-residual["phi_residual"]/inverse_time
    chi_acceleration=-residual["chi_residual"]/inverse_time
    return {
        "metric_acceleration":metric_acceleration,
        "phi_acceleration":float(phi_acceleration),
        "chi_acceleration":float(chi_acceleration),
        "inverse_time_metric":inverse_time,
        "trial_residual":residual,
        "finite":bool(
            np.all(np.isfinite(metric_acceleration))
            and np.isfinite(phi_acceleration) and np.isfinite(chi_acceleration)
        ),
    }


def linearized_reduced_einstein_two_scalar_residual(
    background,perturbation,mass_squared=0.0,potential_offset=-6.0,
    kappa5_squared=1.0,complex_step=1e-30,
    constraint_damping=0.0,constraint_damping_rho=0.0,
):
    """Complex-step linearization about a background GH solution.

    The source is frozen unless the perturbation dictionary supplies
    ``gauge_source_covector`` and ``gauge_source_first``.  Those optional jets
    elevate ``H_a`` to an independent linearized field for source-driver
    coupling while leaving the background source equal to ``Gamma_a``.
    """
    required=(
        "metric","metric_first","metric_second","phi","phi_first","phi_second",
        "chi","chi_first","chi_second",
    )
    if any(key not in background for key in required) or any(key not in perturbation for key in required):
        raise ValueError("background and perturbation jets are incomplete")
    base_geometry=metric_geometry_from_jets(
        background["metric"],background["metric_first"],background["metric_second"],
    )
    source=base_geometry["contracted_christoffel_covector"]
    source_first=base_geometry["contracted_christoffel_covector_first"]
    step=float(complex_step)
    if step<=0:raise ValueError("complex_step must be positive")
    values={}
    for key in required:
        values[key]=np.asarray(background[key],dtype=complex)+1j*step*np.asarray(perturbation[key])
    source_perturbation=np.asarray(
        perturbation.get("gauge_source_covector",np.zeros_like(source)),dtype=float,
    )
    source_first_perturbation=np.asarray(
        perturbation.get("gauge_source_first",np.zeros_like(source_first)),dtype=float,
    )
    if source_perturbation.shape!=source.shape or source_first_perturbation.shape!=source_first.shape:
        raise ValueError("gauge-source perturbation jets have incompatible shapes")
    varied_source=source.astype(complex)+1j*step*source_perturbation
    varied_source_first=source_first.astype(complex)+1j*step*source_first_perturbation
    residual=reduced_einstein_two_scalar_residual(
        values["metric"],values["metric_first"],values["metric_second"],
        values["phi"],values["phi_first"],values["phi_second"],
        values["chi"],values["chi_first"],values["chi_second"],
        varied_source,varied_source_first,mass_squared,potential_offset,kappa5_squared,
        constraint_damping,constraint_damping_rho,
    )
    return {
        "metric_residual":np.imag(residual["metric_residual"])/step,
        "phi_residual":float(np.imag(residual["phi_residual"])/step),
        "chi_residual":float(np.imag(residual["chi_residual"])/step),
        "background_gauge_source_covector":source,
        "background_gauge_source_first":source_first,
        "finite":bool(
            np.all(np.isfinite(np.imag(residual["metric_residual"])/step))
            and np.isfinite(np.imag(residual["phi_residual"])/step)
            and np.isfinite(np.imag(residual["chi_residual"])/step)
        ),
    }


def stationary_point_scalar_metric_mixing(
    orthonormal_phi_hessian,phi,mass_squared,potential_offset=-6.,kappa5_squared=1.,
):
    """Return the algebraic scalar/metric mixing at ``nabla Phi=0``.

    The frame uses signature ``(-,+,+,+,+)``.  The scalar row acts on a
    covariant metric perturbation ``h_ab`` through
    ``-h^{ab} nabla_a nabla_b Phi``.  The metric rows act on ``delta Phi``
    through ``-(2 kappa^2/3) V'(Phi) eta_ab delta Phi``.
    """
    hessian=np.asarray(orthonormal_phi_hessian,dtype=float)
    if hessian.shape!=(5,5):raise ValueError("orthonormal Hessian must be 5 by 5")
    eta=np.diag((-1.,1.,1.,1.,1.));mass_squared=float(mass_squared)
    metric_from_scalar=-(2*float(kappa5_squared)/3)*(mass_squared*float(phi))*eta
    scalar_from_metric=np.empty((5,5))
    for left in range(5):
        for right in range(5):
            scalar_from_metric[left,right]=-(eta[left,left]*eta[right,right])*hessian[left,right]
    potential=float(potential_offset)+.5*mass_squared*float(phi)**2
    return {
        "metric_residual_per_delta_phi":metric_from_scalar,
        "scalar_residual_per_covariant_h":scalar_from_metric,
        "potential":potential,
        "finite":bool(np.all(np.isfinite(metric_from_scalar)) and np.all(np.isfinite(scalar_from_metric))),
        "depends_on_inverse_scalar_gradient":False,
    }
