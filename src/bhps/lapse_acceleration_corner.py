"""Time-time Israel second-corner completion by lapse acceleration."""

from __future__ import annotations

import numpy as np


def _wall_coefficients(phi_wall,target,background,upper):
    gamma=float(background["wall_stiffness"])
    potential=.5*gamma*(phi_wall-target)**2
    if upper:
        beta=float(background["beta_b"])-(potential-float(background["wall_potential_b"]))/6
        beta_phi=-gamma*(phi_wall-target)/6
    else:
        beta=float(background["beta_a"])+(potential-float(background["wall_potential_a"]))/6
        beta_phi=gamma*(phi_wall-target)/6
    return beta,beta_phi


def time_time_israel_second_corner_fields(
    acceleration,alpha,psi,a,phi,background,scalar_acceleration,
    lapse_acceleration=None,radial_buffer=7,
):
    """Evaluate the differentiated ``tt`` Israel row at both walls."""
    alpha=np.asarray(alpha);psi=np.asarray(psi);a=np.asarray(a);phi=np.asarray(phi)
    scalar_acceleration=np.asarray(scalar_acceleration)
    lapse_acceleration=(
        np.zeros_like(alpha) if lapse_acceleration is None
        else np.asarray(lapse_acceleration)
    )
    if not all(x.shape==alpha.shape for x in (psi,a,phi,scalar_acceleration,lapse_acceleration)):
        raise ValueError("all fields must share a grid shape")
    metric_tt=-alpha**2;acceleration_tt=-2*alpha*lapse_acceleration
    dz=acceleration["Dz"];A=psi*np.exp(a);buffer=int(radial_buffer)
    radial_slice=slice(None,-buffer) if buffer else slice(None);walls=[]
    for index,target,upper in (
        (0,float(background["v0"]),False),(-1,float(background["v1"]),True),
    ):
        beta,beta_phi=_wall_coefficients(phi[index],target,background,upper)
        terms=(
            (dz@acceleration_tt)[index],
            2*beta*A[index]*acceleration_tt[index],
            beta*metric_tt[index]*np.asarray(acceleration["zz"])[index]/A[index],
            2*beta_phi*scalar_acceleration[index]*A[index]*metric_tt[index],
        )
        residual=sum(terms);scale=np.maximum(1.,sum(np.abs(term) for term in terms))
        walls.append({
            "wall":"upper" if upper else "lower",
            "residual":np.asarray(residual[radial_slice]),
            "scale":np.asarray(scale[radial_slice]),
        })
    return {"walls":walls,"radial_buffer":buffer}


def construct_lapse_acceleration_completion(
    z,acceleration,alpha,psi,a,phi,background,scalar_acceleration,
):
    """Construct a smooth ``alpha_tt`` clearing both time-time wall rows.

    The spacetime metric acceleration ``g_tt,tt=-2 alpha alpha_tt`` is chosen
    to vanish at both walls.  Its two normal derivatives are supplied by a
    cubic Hermite interpolant, so no spatial ADM acceleration is altered.
    """
    z=np.asarray(z,dtype=float);alpha=np.asarray(alpha);psi=np.asarray(psi)
    a=np.asarray(a);phi=np.asarray(phi);scalar_acceleration=np.asarray(scalar_acceleration)
    A=psi*np.exp(a);metric_tt=-alpha**2;desired=[]
    for index,target,upper in (
        (0,float(background["v0"]),False),(-1,float(background["v1"]),True),
    ):
        beta,beta_phi=_wall_coefficients(phi[index],target,background,upper)
        desired.append(
            -beta*metric_tt[index]*np.asarray(acceleration["zz"])[index]/A[index]
            -2*beta_phi*scalar_acceleration[index]*A[index]*metric_tt[index]
        )
    length=z[-1]-z[0];x=(z-z[0])/length
    lower=x**3-2*x**2+x;upper=x**3-x**2
    metric_tt_acceleration=length*(
        lower[:,None]*desired[0][None,:]+upper[:,None]*desired[1][None,:]
    )
    lapse_acceleration=-metric_tt_acceleration/(2*alpha)
    return {
        "lapse_acceleration":lapse_acceleration,
        "metric_tt_acceleration":metric_tt_acceleration,
        "desired_wall_derivatives":desired,
        "maximum_absolute_lapse_acceleration":float(np.max(np.abs(lapse_acceleration))),
        "maximum_relative_lapse_acceleration":float(np.max(np.abs(lapse_acceleration/alpha))),
    }


def construct_minimum_norm_lapse_acceleration_completion(
    z,acceleration,alpha,psi,a,phi,background,scalar_acceleration,
    derivative_penalty=0.,
):
    """Choose free endpoint values to minimize the Hermite lapse gauge jet."""
    z=np.asarray(z,dtype=float);alpha=np.asarray(alpha);psi=np.asarray(psi)
    a=np.asarray(a);phi=np.asarray(phi);scalar_acceleration=np.asarray(scalar_acceleration)
    A=psi*np.exp(a);metric_tt=-alpha**2;desired=[];robin=[]
    for index,target,upper in (
        (0,float(background["v0"]),False),(-1,float(background["v1"]),True),
    ):
        beta,beta_phi=_wall_coefficients(phi[index],target,background,upper)
        desired.append(
            -beta*metric_tt[index]*np.asarray(acceleration["zz"])[index]/A[index]
            -2*beta_phi*scalar_acceleration[index]*A[index]*metric_tt[index]
        )
        robin.append(2*beta*A[index])
    length=z[-1]-z[0];x=(z-z[0])/length
    h00=2*x**3-3*x**2+1;h01=-2*x**3+3*x**2
    h10=x**3-2*x**2+x;h11=x**3-x**2
    fixed=length*(h10[:,None]*desired[0][None,:]+h11[:,None]*desired[1][None,:])
    column0=h00[:,None]-length*h10[:,None]*robin[0][None,:]
    column1=h01[:,None]-length*h11[:,None]*robin[1][None,:]
    metric_tt_acceleration=np.empty_like(alpha);endpoint_values=np.empty((2,alpha.shape[1]))
    dz=acceleration["Dz"]
    for radial_index in range(alpha.shape[1]):
        design=np.column_stack((
            column0[:,radial_index]/alpha[:,radial_index]**2,
            column1[:,radial_index]/alpha[:,radial_index]**2,
        ))
        # alpha_tt/alpha = -g_tt,tt/(2 alpha^2), so this directly
        # minimizes the dimensionless relative gauge acceleration.
        target=-fixed[:,radial_index]/alpha[:,radial_index]**2
        if float(derivative_penalty)>0:
            derivative_design=dz@np.column_stack((
                column0[:,radial_index],column1[:,radial_index],
            ))
            derivative_target=-(dz@fixed[:,radial_index])
            weight=np.sqrt(float(derivative_penalty))
            design=np.vstack((design,weight*derivative_design))
            target=np.concatenate((target,weight*derivative_target))
        values=np.linalg.lstsq(design,target,rcond=None)[0]
        endpoint_values[:,radial_index]=values
        metric_tt_acceleration[:,radial_index]=(
            fixed[:,radial_index]+column0[:,radial_index]*values[0]
            +column1[:,radial_index]*values[1]
        )
    lapse_acceleration=-metric_tt_acceleration/(2*alpha)
    return {
        "lapse_acceleration":lapse_acceleration,
        "metric_tt_acceleration":metric_tt_acceleration,
        "endpoint_metric_tt_accelerations":endpoint_values,
        "desired_wall_derivatives":desired,"robin_coefficients":robin,
        "derivative_penalty":float(derivative_penalty),
        "maximum_absolute_lapse_acceleration":float(np.max(np.abs(lapse_acceleration))),
        "maximum_relative_lapse_acceleration":float(np.max(np.abs(lapse_acceleration/alpha))),
        "lapse_acceleration_rms":float(np.sqrt(np.mean(lapse_acceleration**2))),
    }


def construct_target_relative_lapse_acceleration_completion(
    z,acceleration,alpha,psi,a,phi,background,scalar_acceleration,
    target_relative_acceleration,
):
    """Clear the ``tt`` corner while fitting a prescribed ``alpha_tt/alpha``.

    The differentiated Israel row fixes one Robin condition at each wall but
    leaves the two endpoint values of ``g_tt,tt`` free.  At each radial point
    those two values are chosen by least squares so that the Hermite completion
    is as close as possible to ``target_relative_acceleration`` throughout the
    interval.  This is useful for fitting a generalized-harmonic gauge jet,
    rather than minimizing the lapse acceleration without regard to the
    physical spatial-volume acceleration.
    """
    z=np.asarray(z,dtype=float);alpha=np.asarray(alpha);psi=np.asarray(psi)
    a=np.asarray(a);phi=np.asarray(phi);scalar_acceleration=np.asarray(scalar_acceleration)
    target_relative_acceleration=np.asarray(target_relative_acceleration,dtype=float)
    if target_relative_acceleration.shape!=alpha.shape:
        raise ValueError("target relative acceleration must share the field shape")
    A=psi*np.exp(a);metric_tt=-alpha**2;desired=[];robin=[]
    for index,target,upper in (
        (0,float(background["v0"]),False),(-1,float(background["v1"]),True),
    ):
        beta,beta_phi=_wall_coefficients(phi[index],target,background,upper)
        desired.append(
            -beta*metric_tt[index]*np.asarray(acceleration["zz"])[index]/A[index]
            -2*beta_phi*scalar_acceleration[index]*A[index]*metric_tt[index]
        )
        robin.append(2*beta*A[index])
    length=z[-1]-z[0];x=(z-z[0])/length
    h00=2*x**3-3*x**2+1;h01=-2*x**3+3*x**2
    h10=x**3-2*x**2+x;h11=x**3-x**2
    fixed=length*(h10[:,None]*desired[0][None,:]+h11[:,None]*desired[1][None,:])
    column0=h00[:,None]-length*h10[:,None]*robin[0][None,:]
    column1=h01[:,None]-length*h11[:,None]*robin[1][None,:]
    metric_tt_acceleration=np.empty_like(alpha);endpoint_values=np.empty((2,alpha.shape[1]))
    for radial_index in range(alpha.shape[1]):
        design=-.5*np.column_stack((
            column0[:,radial_index]/alpha[:,radial_index]**2,
            column1[:,radial_index]/alpha[:,radial_index]**2,
        ))
        fixed_relative=-.5*fixed[:,radial_index]/alpha[:,radial_index]**2
        target=(
            target_relative_acceleration[:,radial_index]-fixed_relative
        )
        values=np.linalg.lstsq(design,target,rcond=None)[0]
        endpoint_values[:,radial_index]=values
        metric_tt_acceleration[:,radial_index]=(
            fixed[:,radial_index]+column0[:,radial_index]*values[0]
            +column1[:,radial_index]*values[1]
        )
    lapse_acceleration=-metric_tt_acceleration/(2*alpha)
    relative=lapse_acceleration/alpha
    mismatch=relative-target_relative_acceleration
    return {
        "lapse_acceleration":lapse_acceleration,
        "metric_tt_acceleration":metric_tt_acceleration,
        "endpoint_metric_tt_accelerations":endpoint_values,
        "desired_wall_derivatives":desired,"robin_coefficients":robin,
        "target_relative_acceleration":target_relative_acceleration,
        "relative_acceleration_mismatch":mismatch,
        "maximum_absolute_lapse_acceleration":float(np.max(np.abs(lapse_acceleration))),
        "maximum_relative_lapse_acceleration":float(np.max(np.abs(relative))),
        "maximum_relative_acceleration_mismatch":float(np.max(np.abs(mismatch))),
        "relative_acceleration_mismatch_rms":float(np.sqrt(np.mean(mismatch**2))),
    }


def construct_projected_target_lapse_acceleration_completion(
    z,acceleration,alpha,psi,a,phi,background,scalar_acceleration,
    target_relative_acceleration,
):
    """Project a target lapse jet onto the two discrete ``tt`` corner rows.

    Unlike the cubic two-endpoint family above, this construction uses every
    normal-grid value as gauge freedom.  It returns the unique correction of
    minimum discrete L2 norm in ``alpha_tt/alpha`` subject to both Israel rows,
    using the same derivative operator as the corner audit.  Its grid and
    localization dependence therefore have to be audited explicitly.
    """
    z=np.asarray(z,dtype=float);alpha=np.asarray(alpha,dtype=float)
    psi=np.asarray(psi,dtype=float);a=np.asarray(a,dtype=float)
    phi=np.asarray(phi,dtype=float);scalar_acceleration=np.asarray(scalar_acceleration,dtype=float)
    target_relative_acceleration=np.asarray(target_relative_acceleration,dtype=float)
    if target_relative_acceleration.shape!=alpha.shape:
        raise ValueError("target relative acceleration must share the field shape")
    A=psi*np.exp(a);metric_tt=-alpha**2;desired=[];robin=[]
    for index,target,upper in (
        (0,float(background["v0"]),False),(-1,float(background["v1"]),True),
    ):
        beta,beta_phi=_wall_coefficients(phi[index],target,background,upper)
        desired.append(
            -beta*metric_tt[index]*np.asarray(acceleration["zz"])[index]/A[index]
            -2*beta_phi*scalar_acceleration[index]*A[index]*metric_tt[index]
        )
        robin.append(2*beta*A[index])
    derivative=acceleration["Dz"]
    derivative=derivative.toarray() if hasattr(derivative,"toarray") else np.asarray(derivative)
    target_metric_tt=-2*alpha**2*target_relative_acceleration
    metric_tt_acceleration=np.empty_like(alpha)
    for radial_index in range(alpha.shape[1]):
        constraint=np.vstack((derivative[0],derivative[-1])).copy()
        constraint[0,0]+=robin[0][radial_index]
        constraint[1,-1]+=robin[1][radial_index]
        # If delta g is parameterized by y=delta g/(2 alpha^2), then
        # minimizing ||y|| is exactly minimizing the relative-lapse mismatch.
        inverse_weight=2*alpha[:,radial_index]**2
        weighted_constraint=constraint*inverse_weight[None,:]
        right_hand_side=(
            np.array([desired[0][radial_index],desired[1][radial_index]])
            -constraint@target_metric_tt[:,radial_index]
        )
        gram=weighted_constraint@weighted_constraint.T
        correction_coordinates=(
            weighted_constraint.T@np.linalg.solve(gram,right_hand_side)
        )
        metric_tt_acceleration[:,radial_index]=(
            target_metric_tt[:,radial_index]+inverse_weight*correction_coordinates
        )
    lapse_acceleration=-metric_tt_acceleration/(2*alpha)
    relative=lapse_acceleration/alpha
    mismatch=relative-target_relative_acceleration
    return {
        "lapse_acceleration":lapse_acceleration,
        "metric_tt_acceleration":metric_tt_acceleration,
        "desired_wall_derivatives":desired,"robin_coefficients":robin,
        "target_relative_acceleration":target_relative_acceleration,
        "relative_acceleration_mismatch":mismatch,
        "maximum_absolute_lapse_acceleration":float(np.max(np.abs(lapse_acceleration))),
        "maximum_relative_lapse_acceleration":float(np.max(np.abs(relative))),
        "maximum_relative_acceleration_mismatch":float(np.max(np.abs(mismatch))),
        "relative_acceleration_mismatch_rms":float(np.sqrt(np.mean(mismatch**2))),
    }


def construct_localized_target_lapse_acceleration_completion(
    z,acceleration,alpha,psi,a,phi,background,scalar_acceleration,
    target_relative_acceleration,logarithmic_width=.15,
):
    """Use smooth fixed-physical-width wall layers to fit a target lapse jet.

    Two analytic profiles, ``s exp(-s/width)``, supply the missing lower and
    upper Robin data.  Their width is fixed in ``log(z)`` rather than in grid
    points, so the completion has a meaningful refinement limit.  The two
    amplitudes at every radius are solved using the audit derivative matrix,
    which leaves the differentiated Israel rows at roundoff.
    """
    z=np.asarray(z,dtype=float);alpha=np.asarray(alpha,dtype=float)
    psi=np.asarray(psi,dtype=float);a=np.asarray(a,dtype=float)
    phi=np.asarray(phi,dtype=float);scalar_acceleration=np.asarray(scalar_acceleration,dtype=float)
    target_relative_acceleration=np.asarray(target_relative_acceleration,dtype=float)
    width=float(logarithmic_width)
    if np.any(z<=0) or width<=0:
        raise ValueError("z and logarithmic width must be positive")
    if target_relative_acceleration.shape!=alpha.shape:
        raise ValueError("target relative acceleration must share the field shape")
    A=psi*np.exp(a);metric_tt=-alpha**2;desired=[];robin=[]
    for index,target,upper in (
        (0,float(background["v0"]),False),(-1,float(background["v1"]),True),
    ):
        beta,beta_phi=_wall_coefficients(phi[index],target,background,upper)
        desired.append(
            -beta*metric_tt[index]*np.asarray(acceleration["zz"])[index]/A[index]
            -2*beta_phi*scalar_acceleration[index]*A[index]*metric_tt[index]
        )
        robin.append(2*beta*A[index])
    derivative=acceleration["Dz"]
    derivative=derivative.toarray() if hasattr(derivative,"toarray") else np.asarray(derivative)
    y=np.log(z);lower_distance=y-y[0];upper_distance=y-y[-1]
    profiles=np.column_stack((
        lower_distance*np.exp(-lower_distance/width),
        upper_distance*np.exp(upper_distance/width),
    ))
    target_metric_tt=-2*alpha**2*target_relative_acceleration
    metric_tt_acceleration=np.empty_like(alpha);profile_coefficients=np.empty((2,alpha.shape[1]))
    for radial_index in range(alpha.shape[1]):
        constraint=np.vstack((derivative[0],derivative[-1])).copy()
        constraint[0,0]+=robin[0][radial_index]
        constraint[1,-1]+=robin[1][radial_index]
        right_hand_side=(
            np.array([desired[0][radial_index],desired[1][radial_index]])
            -constraint@target_metric_tt[:,radial_index]
        )
        coefficients=np.linalg.solve(constraint@profiles,right_hand_side)
        profile_coefficients[:,radial_index]=coefficients
        metric_tt_acceleration[:,radial_index]=(
            target_metric_tt[:,radial_index]+profiles@coefficients
        )
    lapse_acceleration=-metric_tt_acceleration/(2*alpha)
    relative=lapse_acceleration/alpha
    mismatch=relative-target_relative_acceleration
    return {
        "lapse_acceleration":lapse_acceleration,
        "metric_tt_acceleration":metric_tt_acceleration,
        "desired_wall_derivatives":desired,"robin_coefficients":robin,
        "profile_coefficients":profile_coefficients,
        "logarithmic_width":width,
        "target_relative_acceleration":target_relative_acceleration,
        "relative_acceleration_mismatch":mismatch,
        "maximum_absolute_lapse_acceleration":float(np.max(np.abs(lapse_acceleration))),
        "maximum_relative_lapse_acceleration":float(np.max(np.abs(relative))),
        "maximum_relative_acceleration_mismatch":float(np.max(np.abs(mismatch))),
        "relative_acceleration_mismatch_rms":float(np.sqrt(np.mean(mismatch**2))),
    }
