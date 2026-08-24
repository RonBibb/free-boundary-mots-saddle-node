"""Native-grid short-time evolution for the nonlinear regular SO(3) system.

This module turns the pointwise five-dimensional generalized-harmonic
Einstein--two-scalar acceleration solve into a method-of-lines right-hand
side for the nine regular fields used by :mod:`bhps.regular_so3_gh_reduction`.
It also supplies the live gauge, compact-wall, and outer-radial rows needed by
the short nonlinear promotion tests.
"""

from __future__ import annotations

import numpy as np

from bhps.adm_corner import _axisymmetric_derivatives
from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.gh_source_driver import (
    regular_so3_live_source_shift_advection,
    regular_so3_nonlinear_anchored_damped_wave_target,
    regular_so3_source_jets,
)
from bhps.linearized_gh_einstein_scalar import (
    metric_geometry_from_jets,
    solve_reduced_einstein_two_scalar_acceleration,
)
from bhps.joint_parent_selective_algebra import (
    solve_selective_wall_endpoint_block,
    summarize_selective_field_evidence,
)
from bhps.regular_so3_gh_reduction import FIELD_ORDER, regular_so3_perturbation_jets


FIELD_COUNT = 9
BOUNDARY_CLOSURE_MODES = (
    "legacy_wall_axis_outer",
    "wall_owner_last_experimental",
)


class CompactWallCoupledAlgebraicGateError(RuntimeError):
    """Structured rejection by the direct compact-wall 4x4 gate."""

    def __init__(self, message, *, radial_index, gate, diagnostics):
        super().__init__(message)
        self.radial_index = int(radial_index)
        self.gate = str(gate)
        self.diagnostics = dict(diagnostics)


def reduced_state_jets(position, velocity, z, r, stencil_width=7):
    """Return reduced value, first, and second jets on one native grid."""
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    expected = (len(z), len(r), FIELD_COUNT)
    if position.shape != expected or velocity.shape != expected or r[0] != 0:
        raise ValueError("invalid native regular state")
    first = np.zeros((3, *expected))
    second = np.zeros((3, 3, *expected))
    first[0] = velocity
    for field in range(FIELD_COUNT):
        spatial = _axisymmetric_derivatives(
            position[:, :, field], z, r, stencil_width,
        )
        velocity_spatial = _axisymmetric_derivatives(
            velocity[:, :, field], z, r, stencil_width,
        )
        first[1, :, :, field] = spatial["z"]
        first[2, :, :, field] = spatial["r"]
        second[1, 1, :, :, field] = spatial["zz"]
        second[1, 2, :, :, field] = spatial["zr"]
        second[2, 1, :, :, field] = spatial["zr"]
        second[2, 2, :, :, field] = spatial["rr"]
        second[0, 1, :, :, field] = velocity_spatial["z"]
        second[1, 0, :, :, field] = velocity_spatial["z"]
        second[0, 2, :, :, field] = velocity_spatial["r"]
        second[2, 0, :, :, field] = velocity_spatial["r"]
    return first, second


def _axis_fit(positive_values, r, window=0.5, degree=3):
    r = np.asarray(r, dtype=float)
    keep = (r[1:] <= float(window) + 1e-12)
    if np.count_nonzero(keep) < degree + 1:
        raise ValueError("axis window has too few points")
    values = np.asarray(positive_values)[..., keep, :]
    squared = r[1:][keep] ** 2
    flat = values.reshape((-1, len(squared), values.shape[-1]))
    result = np.empty((flat.shape[0], flat.shape[-1]))
    for row in range(flat.shape[0]):
        for field in range(flat.shape[-1]):
            result[row, field] = np.polynomial.polynomial.polyfit(
                squared, flat[row, :, field], degree,
            )[0]
    return result.reshape((*values.shape[:-2], values.shape[-1]))


def fill_regular_axis(field, r, window=0.5, degree=3):
    """Fill the removable even axis value of a regular native-grid field."""
    result = np.asarray(field, dtype=float).copy()
    if result.ndim != 3 or result.shape[1] != len(r):
        raise ValueError("axis field must have shape (z,r,components)")
    result[:, 0] = _axis_fit(result[:, 1:], r, window, degree)
    return result


def _reduced_covector_from_full(full, r, axis_window=0.5):
    """Reduce an SO(3)-symmetric Cartesian covector to (time,z,radial/r)."""
    full = np.asarray(full, dtype=float)
    r = np.asarray(r, dtype=float)
    reduced = np.empty((*full.shape[:2], 3))
    reduced[:, :, 0] = full[:, :, 0]
    reduced[:, :, 1] = full[:, :, 1]
    reduced[:, 1:, 2] = full[:, 1:, 2] / r[None, 1:]
    temporary = np.zeros((*full.shape[:2], 1))
    temporary[:, 1:, 0] = reduced[:, 1:, 2]
    temporary = fill_regular_axis(temporary, r, axis_window)
    reduced[:, 0, 2] = temporary[:, 0, 0]
    return reduced


def _full_covector_value(reduced, radius):
    value = np.zeros(5)
    value[0] = reduced[0]
    value[1] = reduced[1]
    value[2] = radius * reduced[2]
    return value


def _full_covector_spatial_first(reduced, z_first, r_first, radius):
    """Map spatial derivatives of a regular covector to Cartesian form."""
    result = np.zeros((5, 5))
    result[1, 0] = z_first[0]
    result[1, 1] = z_first[1]
    result[1, 2] = radius * z_first[2]
    if radius > 0:
        result[2, 0] = r_first[0]
        result[2, 1] = r_first[1]
        result[2, 2] = reduced[2] + radius * r_first[2]
    else:
        result[2, 2] = reduced[2]
    result[3, 3] = reduced[2]
    result[4, 4] = reduced[2]
    return result


class GaugeTaylorSource:
    """Prescribed first-order Taylor gauge source about one initial slice."""

    def __init__(self, source_value, source_first, z, r, stencil_width=7):
        self.source_value = np.asarray(source_value, dtype=float)
        self.source_first = np.asarray(source_first, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.r = np.asarray(r, dtype=float)
        expected = (len(z), len(r))
        if self.source_value.shape != (*expected, 5):
            raise ValueError("invalid gauge source values")
        if self.source_first.shape != (*expected, 5, 5):
            raise ValueError("invalid gauge source first jets")
        self.source_reduced = _reduced_covector_from_full(
            self.source_value, self.r,
        )
        source_time_full = self.source_first[:, :, 0, :]
        self.source_time_reduced = _reduced_covector_from_full(
            source_time_full, self.r,
        )
        self.source_time_z = np.empty_like(self.source_time_reduced)
        self.source_time_r = np.empty_like(self.source_time_reduced)
        for field in range(3):
            derivatives = _axisymmetric_derivatives(
                self.source_time_reduced[:, :, field], self.z, self.r,
                stencil_width,
            )
            self.source_time_z[:, :, field] = derivatives["z"]
            self.source_time_r[:, :, field] = derivatives["r"]

    def at(self, i, j, time):
        radius = float(self.r[j])
        source_time = _full_covector_value(
            self.source_time_reduced[i, j], radius,
        )
        value = self.source_value[i, j] + float(time) * source_time
        first = self.source_first[i, j].copy()
        first[0] = source_time
        first += float(time) * _full_covector_spatial_first(
            self.source_time_reduced[i, j], self.source_time_z[i, j],
            self.source_time_r[i, j], radius,
        )
        # The Taylor source has H_tt=0; do not add the spatial helper's zero
        # time row to the exact prescribed H_t row above.
        first[0] = source_time
        return value, first


def regular_source_spatial_derivatives(source, z, r, stencil_width=7):
    """Return native z and radial derivatives of three regular source fields."""
    source=np.asarray(source,dtype=float)
    z=np.asarray(z,dtype=float);r=np.asarray(r,dtype=float)
    if source.shape!=(len(z),len(r),3):
        raise ValueError("regular source must have shape (z,r,3)")
    z_first=np.empty_like(source);r_first=np.empty_like(source)
    for field in range(3):
        derivatives=_axisymmetric_derivatives(
            source[:,:,field],z,r,stencil_width,
        )
        z_first[:,:,field]=derivatives["z"]
        r_first[:,:,field]=derivatives["r"]
    return z_first,r_first


def regular_so3_outward_radial_speed(position,r):
    """Return the live outward radial characteristic speed at ``r[-1]``.

    The speed is the positive root of the inverse-metric radial principal
    symbol.  Using the inverse metric retains live shift and compact-radial
    mixing rather than assuming a diagonal background.
    """
    q=np.asarray(position,dtype=float);r=np.asarray(r,dtype=float)
    if q.ndim!=3 or q.shape[1]!=len(r) or q.shape[2]!=FIELD_COUNT or len(r)<2:
        raise ValueError("invalid regular metric state for radial speed")
    radius=float(r[-1]);speed=np.empty(q.shape[0])
    for i in range(q.shape[0]):
        metric=regular_so3_perturbation_jets(radius,q[i,-1])["metric"]
        inverse=np.linalg.inv(metric)
        gtt=float(inverse[0,0]);gtr=float(inverse[0,2]);grr=float(inverse[2,2])
        discriminant=gtr*gtr-gtt*grr
        if gtt>=0 or discriminant<=0 or not np.isfinite(discriminant):
            raise RuntimeError("outer radial face is not hyperbolic")
        speed[i]=(gtr-np.sqrt(discriminant))/gtt
    if np.any(speed<=0) or not np.all(np.isfinite(speed)):
        raise RuntimeError("outer radial characteristic is not outgoing")
    return speed


def _outer_radial_first(field,r,stencil_width=7):
    matrix=derivative_matrix(np.asarray(r,dtype=float),1,stencil_width)
    if hasattr(matrix,"toarray"):matrix=matrix.toarray()
    return np.einsum("j,ijf->if",matrix[-1],np.asarray(field,dtype=float))


def apply_outer_sommerfeld_acceleration(
    position,velocity,acceleration,reference_position,reference_acceleration,
    time,r,stencil_width=7,difference_step=1e-6,
):
    """Apply the live differentiated outgoing row at the artificial face.

    For ``delta q = q-(q0+t^2 a0/2)`` the boundary condition is
    ``delta q_t + c_+ delta q_r = 0``.  Its time derivative supplies the
    acceleration row.  Compact-wall corner nodes are excluded so the physical
    Israel/scalar and normal-gauge rows retain ownership there.
    """
    q=np.asarray(position,dtype=float);v=np.asarray(velocity,dtype=float)
    a=np.asarray(acceleration,dtype=float).copy()
    q0=np.asarray(reference_position,dtype=float)
    a0=np.asarray(reference_acceleration,dtype=float)
    time=float(time);step=float(difference_step)
    if q.shape!=v.shape or q.shape!=a.shape or q.shape!=q0.shape or q.shape!=a0.shape:
        raise ValueError("invalid outer Sommerfeld metric fields")
    if step<=0:raise ValueError("outer speed difference step must be positive")
    reference=q0+.5*time*time*a0
    reference_velocity=time*a0
    delta=q-reference;delta_velocity=v-reference_velocity
    speed=regular_so3_outward_radial_speed(q,r)
    speed_time=(
        regular_so3_outward_radial_speed(q+step*v,r)
        -regular_so3_outward_radial_speed(q-step*v,r)
    )/(2*step)
    delta_r=_outer_radial_first(delta,r,stencil_width)
    delta_velocity_r=_outer_radial_first(delta_velocity,r,stencil_width)
    target=a0[:,-1]-speed[:,None]*delta_velocity_r-speed_time[:,None]*delta_r
    before=a[:,-1].copy()
    a[1:-1,-1]=target[1:-1]
    residual=(
        a[:,-1]-a0[:,-1]+speed[:,None]*delta_velocity_r
        +speed_time[:,None]*delta_r
    )
    scale=np.maximum(
        1.,np.abs(a[:,-1]-a0[:,-1])
        +np.abs(speed[:,None]*delta_velocity_r)
        +np.abs(speed_time[:,None]*delta_r),
    )
    open_residual=residual[1:-1];open_scale=scale[1:-1]
    correction=a[1:-1,-1]-before[1:-1]
    return a,{
        "minimum_outward_speed":float(np.min(speed)),
        "maximum_outward_speed":float(np.max(speed)),
        "maximum_normalized_acceleration_residual":float(
            np.max(np.abs(open_residual)/open_scale)
        ),
        "maximum_absolute_acceleration_residual":float(np.max(np.abs(open_residual))),
        "metric_relative_correction":float(
            np.linalg.norm(correction[:,:7])
            /max(np.linalg.norm(a[1:-1,-1,:7]),np.linalg.norm(before[1:-1,:7]),1e-300)
        ),
        "scalar_relative_correction":float(
            np.linalg.norm(correction[:,7:])
            /max(np.linalg.norm(a[1:-1,-1,7:]),np.linalg.norm(before[1:-1,7:]),1e-300)
        ),
    }


def outer_sommerfeld_position_residuals(
    position,velocity,reference_position,reference_acceleration,time,r,
    stencil_width=7,
):
    """Evaluate ``delta q_t+c_+ delta q_r`` at open outer-face nodes."""
    q=np.asarray(position,dtype=float);v=np.asarray(velocity,dtype=float)
    q0=np.asarray(reference_position,dtype=float);a0=np.asarray(reference_acceleration,dtype=float)
    time=float(time);reference=q0+.5*time*time*a0
    delta=q-reference;delta_velocity=v-time*a0
    speed=regular_so3_outward_radial_speed(q,r)
    radial=_outer_radial_first(delta,r,stencil_width)
    terms=(delta_velocity[:,-1],speed[:,None]*radial)
    residual=sum(terms);scale=np.maximum(1.,np.abs(terms[0])+np.abs(terms[1]))
    open_residual=residual[1:-1];open_scale=scale[1:-1]
    return {
        "maximum_normalized":float(np.max(np.abs(open_residual)/open_scale)),
        "maximum_absolute":float(np.max(np.abs(open_residual))),
        "metric_maximum_normalized":float(np.max(np.abs(open_residual[:,:7])/open_scale[:,:7])),
        "scalar_maximum_normalized":float(np.max(np.abs(open_residual[:,7:])/open_scale[:,7:])),
    }


def apply_outer_source_sommerfeld(
    source,source_time,reference_source,reference_source_time,
    reference_source_second_time,position,time,r,stencil_width=7,
):
    """Apply an outgoing first-order row to the three live source fields."""
    h=np.asarray(source,dtype=float);ht=np.asarray(source_time,dtype=float).copy()
    h0=np.asarray(reference_source,dtype=float);ht0=np.asarray(reference_source_time,dtype=float)
    htt0=np.asarray(reference_source_second_time,dtype=float)
    if h.shape!=ht.shape or h.shape!=h0.shape or h.shape!=ht0.shape or h.shape!=htt0.shape:
        raise ValueError("invalid outer Sommerfeld source fields")
    time=float(time);reference=h0+time*ht0+.5*time*time*htt0
    reference_time=ht0+time*htt0
    speed=regular_so3_outward_radial_speed(position,r)
    radial=_outer_radial_first(h-reference,r,stencil_width)
    target=reference_time[:,-1]-speed[:,None]*radial
    before=ht[:,-1].copy();ht[1:-1,-1]=target[1:-1]
    residual=ht[:,-1]-reference_time[:,-1]+speed[:,None]*radial
    scale=np.maximum(1.,np.abs(ht[:,-1]-reference_time[:,-1])+np.abs(speed[:,None]*radial))
    return ht,{
        "maximum_normalized":float(np.max(np.abs(residual[1:-1])/scale[1:-1])),
        "maximum_absolute":float(np.max(np.abs(residual[1:-1]))),
        "relative_correction":float(
            np.linalg.norm(ht[1:-1,-1]-before[1:-1])
            /max(np.linalg.norm(ht[1:-1,-1]),np.linalg.norm(before[1:-1]),1e-300)
        ),
    }


def live_regular_source_second_time(
    position,velocity,background_position,background_source,
    source,source_time,memory_time,z,r,driver_mu,
    target_mu_lapse,target_mu_shift,target_determinant_power=.5,
    stencil_width=7,difference_step=1e-6,
):
    """Return ``H_tt`` from the live first-order driver at one stage.

    Only the directional time derivatives of the nonlinear target and shift
    advection are needed. They are evaluated by a centered directional
    difference along the current metric and source velocities. This avoids a
    third metric jet while retaining the exact driver equation
    ``H_tt=advection_t-mu(H_t-F_t)+theta_t``.
    """
    q=np.asarray(position,dtype=float);v=np.asarray(velocity,dtype=float)
    h=np.asarray(source,dtype=float);ht=np.asarray(source_time,dtype=float)
    theta_t=np.asarray(memory_time,dtype=float)
    step=float(difference_step);mu=float(driver_mu)
    if (
        q.shape!=v.shape or h.shape!=ht.shape or h.shape!=theta_t.shape
        or h.shape!=q.shape[:-1]+(3,) or step<=0 or mu<=0
    ):
        raise ValueError("invalid live source second-time fields")
    target_values=[];advection_values=[]
    for sign in (-1.,1.):
        q_stage=q+sign*step*v
        h_stage=h+sign*step*ht
        target_values.append(regular_so3_nonlinear_anchored_damped_wave_target(
            q_stage,background_position,background_source,r,
            target_mu_lapse,target_mu_shift,target_determinant_power,
        ))
        hz,hr=regular_source_spatial_derivatives(
            h_stage,z,r,stencil_width,
        )
        advection_values.append(regular_so3_live_source_shift_advection(
            q_stage,r,h_stage,hz,hr,
        ))
    target_time=(target_values[1]-target_values[0])/(2*step)
    advection_time=(advection_values[1]-advection_values[0])/(2*step)
    return advection_time-mu*(ht-target_time)+theta_t


class StageRegularGaugeSource:
    """Full Cartesian gauge-source jets from one live regular driver stage."""

    def __init__(self,source,source_time,z,r,stencil_width=7):
        self.source=np.asarray(source,dtype=float)
        self.source_time=np.asarray(source_time,dtype=float)
        self.z=np.asarray(z,dtype=float);self.r=np.asarray(r,dtype=float)
        expected=(len(self.z),len(self.r),3)
        if self.source.shape!=expected or self.source_time.shape!=expected:
            raise ValueError("invalid live regular gauge-source stage")
        self.source_z,self.source_r=regular_source_spatial_derivatives(
            self.source,self.z,self.r,stencil_width,
        )

    def at(self,i,j,time=0.):
        del time
        first=np.stack((
            self.source_time[i,j],self.source_z[i,j],self.source_r[i,j],
        ))
        jets=regular_so3_source_jets(
            float(self.r[j]),self.source[i,j],first,
        )
        return jets["covector"],jets["first"]


def gauge_taylor_source_from_initial_jets(jet_field, z, r):
    """Construct H=Gamma and its first jet from archived initial data."""
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    value = np.empty((len(z), len(r), 5))
    first = np.empty((len(z), len(r), 5, 5))
    for i in range(len(z)):
        for j in range(len(r)):
            jets = regular_so3_perturbation_jets(
                float(r[j]), jet_field.reduced_fields[i, j],
                jet_field.reduced_first[:, i, j],
                jet_field.reduced_second[:, :, i, j],
            )
            geometry = metric_geometry_from_jets(
                jets["metric"], jets["metric_first"], jets["metric_second"],
            )
            value[i, j] = geometry["contracted_christoffel_covector"]
            first[i, j] = geometry["contracted_christoffel_covector_first"]
    return GaugeTaylorSource(value, first, z, r)


def _pack_acceleration(metric_acceleration, phi_acceleration, chi_acceleration, radius):
    metric = np.asarray(metric_acceleration, dtype=float)
    transverse = 0.5 * (metric[3, 3] + metric[4, 4])
    return np.array((
        metric[1, 0], metric[1, 2] / radius, metric[0, 0], transverse,
        (metric[2, 2] - transverse) / radius**2, metric[0, 2] / radius,
        metric[1, 1], phi_acceleration, chi_acceleration,
    ))


def _wall_coefficients(phi, background, upper):
    gamma = float(background["wall_stiffness"])
    target = float(background["v1"] if upper else background["v0"])
    potential = 0.5 * gamma * (phi - target) ** 2
    if upper:
        beta = float(background["beta_b"]) - (
            potential - float(background["wall_potential_b"])
        ) / 6
        beta_phi = -gamma * (phi - target) / 6
        beta_phiphi = -gamma / 6
    else:
        beta = float(background["beta_a"]) + (
            potential - float(background["wall_potential_a"])
        ) / 6
        beta_phi = gamma * (phi - target) / 6
        beta_phiphi = gamma / 6
    return target, beta, beta_phi, beta_phiphi


def _solve_endpoint(dz, field, index, robin, forcing):
    diagonal = float(dz[index, index])
    without = (dz @ field)[index] - diagonal * field[index]
    denominator = diagonal + robin
    if np.any(np.abs(denominator) < 1e-12):
        raise RuntimeError("degenerate compact-wall acceleration row")
    field[index] = -(without + forcing) / denominator


def _finish_compact_wall_axis_fill(acceleration, r, normal_wall_acceleration=None):
    """Apply the legacy post-wall axis fill and restore owned ``g_zz`` data."""
    result = fill_regular_axis(acceleration, r)
    if normal_wall_acceleration is not None:
        normal = np.asarray(normal_wall_acceleration, dtype=float)
        result[0, 0, 6] = normal[0, 0]
        result[-1, 0, 6] = normal[1, 0]
    return result


def _native_regular_axis_quotient_images(acceleration, r):
    """Return the three reduced quotient limits from physical numerators."""
    source = np.asarray(acceleration, dtype=float)
    r = np.asarray(r, dtype=float)
    if source.ndim != 3 or source.shape[1] != len(r) or source.shape[2] != FIELD_COUNT:
        raise ValueError("axis acceleration must have shape (z,r,9)")
    if len(r) < 7 or r[0] != 0.0 or np.signbit(r[0]) or np.any(np.diff(r) <= 0.0):
        raise ValueError("native axis images require an increasing +0 radial grid")
    dr = derivative_matrix(r, 1, 7)
    parent_radius = float(r[-1])
    s = (r / parent_radius) ** 2
    ds = derivative_matrix(s, 1, 7)
    if hasattr(dr, "toarray"):
        dr = dr.toarray()
    if hasattr(ds, "toarray"):
        ds = ds.toarray()
    images = []
    for field in (1, 4, 5):
        numerator = np.zeros(source.shape[:2], dtype=float)
        if field == 4:
            numerator[:, 1:] = r[None, 1:] ** 2 * source[:, 1:, field]
            image = (ds @ numerator.T).T[:, 0] / parent_radius**2
        else:
            numerator[:, 1:] = r[None, 1:] * source[:, 1:, field]
            image = (dr @ numerator.T).T[:, 0]
        if np.any(numerator[:, 0] != 0.0) or np.any(np.signbit(numerator[:, 0])):
            raise AssertionError("physical parity numerator axis is not IEEE +0")
        images.append(image)
    result = np.ascontiguousarray(np.stack(images, axis=1))
    if not np.all(np.isfinite(result)):
        raise RuntimeError("native axis quotient image is nonfinite")
    return result


def reconcile_wall_owner_axis_null_channels(
    acceleration, r, window=0.5, degree=3,
):
    """Reconcile all three reduced quotient limits by native parity.

    At ``r=0`` the physical rows contain ``r*q1``, ``r**2*q4``, and
    ``r*q5``.  Their reduced axis values are supplied by the production
    seven-point derivative images of those physical numerators.  The legacy
    ``window`` and ``degree`` arguments remain accepted for API compatibility
    but no longer select a fixed physical fitting window.
    """
    result = np.asarray(acceleration, dtype=float).copy()
    r = np.asarray(r, dtype=float)
    if result.ndim != 3 or result.shape[1] != len(r) or result.shape[2] != FIELD_COUNT:
        raise ValueError("axis acceleration must have shape (z,r,9)")
    null_channels = (1, 4, 5)
    result[:, 0, null_channels] = _native_regular_axis_quotient_images(
        result, r,
    )
    return result


def apply_compact_wall_acceleration(
    position, velocity, acceleration, z, r, background,
    normal_wall_acceleration=None, stencil_width=7, *, fill_axis_after=True,
    impose_normal_tangential=True,
):
    """Solve the nonlinear twice-time-differentiated compact-wall rows.

    The normal-normal acceleration ``g_zz,tt`` is a gauge boundary datum.
    When supplied, ``normal_wall_acceleration`` has shape ``(2,nr)`` and is
    imposed at the lower and upper compact walls before the physical rows are
    solved.  ``fill_axis_after=False`` exposes the endpoint-solved state for
    staged diagnostics and for the opt-in wall-owner-last ordering.  The
    default retains the historical wall-solve then axis-fill operation.
    ``impose_normal_tangential=False`` is a diagnostic staging option: it
    exposes the physical metric/scalar wall solve before the separately owned
    normal-tangential Dirichlet gauge values are imposed.  Production callers
    retain the default behavior.
    """
    q = np.asarray(position, dtype=float)
    v = np.asarray(velocity, dtype=float)
    a = np.asarray(acceleration, dtype=float).copy()
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    dz = derivative_matrix(z, 1, stencil_width)
    if hasattr(dz, "toarray"):
        dz = dz.toarray()
    if normal_wall_acceleration is not None:
        normal = np.asarray(normal_wall_acceleration, dtype=float)
        if normal.shape != (2, len(r)):
            raise ValueError("normal wall gauge datum must have shape (2,nr)")
        a[0, :, 6] = normal[0]
        a[-1, :, 6] = normal[1]

    radius2 = r[None, :] ** 2
    fields = {
        "g00": q[:, :, 2].copy(),
        "perp": q[:, :, 3].copy(),
        "radial": (q[:, :, 3] + radius2 * q[:, :, 4]).copy(),
        "g0r": (r[None, :] * q[:, :, 5]).copy(),
    }
    rates = {
        "g00": v[:, :, 2].copy(),
        "perp": v[:, :, 3].copy(),
        "radial": (v[:, :, 3] + radius2 * v[:, :, 4]).copy(),
        "g0r": (r[None, :] * v[:, :, 5]).copy(),
    }
    accelerations = {
        "g00": a[:, :, 2].copy(),
        "perp": a[:, :, 3].copy(),
        "radial": (a[:, :, 3] + radius2 * a[:, :, 4]).copy(),
        "g0r": (r[None, :] * a[:, :, 5]).copy(),
    }
    corrections = []
    for wall, index, upper, sign in (
        ("lower", 0, False, -1.0), ("upper", -1, True, 1.0),
    ):
        A = np.sqrt(q[index, :, 6])
        A_t = v[index, :, 6] / (2 * A)
        A_tt = a[index, :, 6] / (2 * A) - v[index, :, 6] ** 2 / (4 * A**3)
        target, beta, beta_phi, beta_phiphi = _wall_coefficients(
            q[index, :, 7], background, upper,
        )

        old_phi = a[index, :, 7].copy()
        scalar_forcing = sign * 0.5 * float(background["wall_stiffness"]) * (
            (q[index, :, 7] - target) * A_tt
            + 2 * v[index, :, 7] * A_t
        )
        _solve_endpoint(
            dz, a[:, :, 7], index,
            sign * 0.5 * float(background["wall_stiffness"]) * A,
            scalar_forcing,
        )
        old_chi = a[index, :, 8].copy()
        _solve_endpoint(dz, a[:, :, 8], index, 0.0, 0.0)

        component_old = []
        component_new = []
        for name in ("g00", "perp", "radial", "g0r"):
            X = fields[name][index]
            X_t = rates[name][index]
            X_tt = accelerations[name]
            component_old.append(X_tt[index].copy())
            beta_t = beta_phi * v[index, :, 7]
            beta_tt = (
                beta_phi * a[index, :, 7]
                + beta_phiphi * v[index, :, 7] ** 2
            )
            forcing = 2 * (
                beta_tt * A * X + beta * A_tt * X
                + 2 * beta_t * A_t * X + 2 * beta_t * A * X_t
                + 2 * beta * A_t * X_t
            )
            _solve_endpoint(dz, X_tt, index, 2 * beta * A, forcing)
            component_new.append(X_tt[index].copy())

        a[index, :, 2] = accelerations["g00"][index]
        a[index, :, 3] = accelerations["perp"][index]
        a[index, 1:, 4] = (
            accelerations["radial"][index, 1:] - accelerations["perp"][index, 1:]
        ) / r[1:] ** 2
        a[index, 1:, 5] = accelerations["g0r"][index, 1:] / r[1:]
        # h_z0 and h_zr/r are normal-tangential Dirichlet gauge fields.  The
        # Phase-A2 diagnostic can persist the physical solve before this
        # separately owned gauge substep; production retains the default.
        if impose_normal_tangential:
            a[index, :, 0] = 0.0
            a[index, :, 1] = 0.0
        before = np.concatenate((old_phi, old_chi, *component_old))
        after = np.concatenate((a[index, :, 7], a[index, :, 8], *component_new))
        corrections.append({
            "wall": wall,
            "relative_norm": float(
                np.linalg.norm(after - before)
                / max(np.linalg.norm(after), np.linalg.norm(before), 1e-300)
            ),
            "maximum_absolute": float(np.max(np.abs(after - before))),
        })
    if fill_axis_after:
        a = _finish_compact_wall_axis_fill(a, r, normal_wall_acceleration)
    return a, corrections


def solve_compact_wall_tangential_chi_acceleration(
    position, velocity, acceleration, z, r, background, stencil_width=7,
    *, maximum_normalized_residual=1e-12, capture_profiles=False,
):
    """Solve only the time-symmetric normalized tangential and chi rows.

    This opt-in parent-construction helper assumes that the coupled compact-
    wall block has already supplied ``Phi_tt`` and ``g_zz,tt``.  Those two
    channels, together with the normal-tangential gauge channels, are treated
    as read-only.  For each physical tangential metric component ``f`` it
    solves the cancellation-exposed derivative of the *normalized* Israel
    functional,

    ``D_X^2 J_f = s [D_z f_tt/(2 A) - (D_z f) G_tt/(4 A^3)``
    ``                    + beta_Phi Phi_tt f + beta f_tt] = 0``,

    where ``A=sqrt(g_zz)`` and the exactly time-symmetric input makes the
    velocity-Hessian lane vanish.  This is not the raw ``Robin_tt=0`` row:
    the second term retains the off-manifold normalization derivative.

    The reflecting-scalar row is ``D_z chi_tt=0``.  Both compact endpoints
    are solved together at each radius.  The routine performs no generic axis
    fill, no normal-tangential gauge assignment, and no outer overwrite.
    Consequently the tensor-null axis coefficients ``q4`` and ``q5`` remain
    bitwise unchanged for a later, separately owned reconciliation step.
    """
    q = np.asarray(position, dtype=float)
    v = np.asarray(velocity, dtype=float)
    input_acceleration = np.asarray(acceleration, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    width = int(stencil_width)
    limit = float(maximum_normalized_residual)
    expected = (len(z), len(r), FIELD_COUNT)
    if (
        q.shape != expected or v.shape != expected
        or input_acceleration.shape != expected
    ):
        raise ValueError("invalid tangential/chi compact-wall fields")
    if (
        z.ndim != 1 or r.ndim != 1 or len(z) < width or len(r) < 2
        or np.any(np.diff(z) <= 0.0) or np.any(np.diff(r) <= 0.0)
        or r[0] != 0.0
    ):
        raise ValueError("invalid tangential/chi compact-wall grid")
    if width != stencil_width or width < 3:
        raise ValueError("invalid tangential/chi stencil width")
    if not all(np.all(np.isfinite(value)) for value in (
        q, v, input_acceleration, z, r,
    )):
        raise ValueError("tangential/chi compact-wall inputs must be finite")
    if np.any(v != 0.0):
        raise ValueError(
            "normalized parent-wall acceleration solve requires exact time symmetry"
        )
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("maximum_normalized_residual must be positive")
    if np.any(q[[0, -1], :, 6] <= 0.0):
        raise ValueError("compact-normal wall metric must be positive")

    dz = derivative_matrix(z, 1, width)
    if hasattr(dz, "toarray"):
        dz = dz.toarray()
    dz = np.asarray(dz, dtype=float)
    result = input_acceleration.copy()
    protected_channels = (0, 1, 6, 7)
    protected_before = np.ascontiguousarray(
        result[:, :, protected_channels]
    ).view(np.uint64).copy()
    null_axis_before = np.ascontiguousarray(
        result[:, 0, 4:6]
    ).view(np.uint64).copy()

    radius = r[None, :]
    radius2 = radius**2
    fields = {
        "tt": q[:, :, 2].copy(),
        "sphere": q[:, :, 3].copy(),
        "rr": (q[:, :, 3] + radius2 * q[:, :, 4]).copy(),
        "tr": (radius * q[:, :, 5]).copy(),
    }
    accelerations = {
        "tt": result[:, :, 2].copy(),
        "sphere": result[:, :, 3].copy(),
        "rr": (result[:, :, 3] + radius2 * result[:, :, 4]).copy(),
        "tr": (radius * result[:, :, 5]).copy(),
    }
    endpoints = (0, len(z) - 1)
    wall_specs = (("lower", 0, False, -1.0),
                  ("upper", len(z) - 1, True, 1.0))
    wall_data = []
    for wall_name, index, upper, orientation in wall_specs:
        _, beta, beta_phi, _ = _wall_coefficients(
            q[index, :, 7], background, upper,
        )
        wall_data.append({
            "wall": wall_name,
            "index": index,
            "orientation": orientation,
            "A": np.sqrt(q[index, :, 6]),
            "beta": np.asarray(beta, dtype=float),
            "beta_phi": np.asarray(beta_phi, dtype=float),
        })

    selective_field_names = ("h_00", "h_perp", "h_rr", "h_0r", "chi")
    metric_field_names = selective_field_names[:4]
    algebraic_records = {name: [] for name in selective_field_names}
    minimum_rank = 2
    maximum_condition = 0.0
    weakest_pivot = float("inf")
    for radial_index in range(len(r)):
        matrix = np.empty((2, 2))
        full_rows = np.empty((2, len(z)))
        right = np.empty((2, len(fields)))
        for wall_number, wall in enumerate(wall_data):
            index = wall["index"]
            A = wall["A"][radial_index]
            beta = wall["beta"][radial_index]
            beta_phi = wall["beta_phi"][radial_index]
            full_row = dz[index] / (2.0 * A)
            full_row = full_row.copy()
            full_row[index] += beta
            full_rows[wall_number] = full_row
            matrix[wall_number] = full_row[list(endpoints)]
            for component, (name, field) in enumerate(fields.items()):
                acceleration_field = accelerations[name]
                interior = float(
                    full_row[1:-1] @ acceleration_field[1:-1, radial_index]
                )
                field_z = float(dz[index] @ field[:, radial_index])
                normalization = (
                    -field_z
                    * result[index, radial_index, 6]
                    / (4.0 * A**3)
                )
                phi_coupling = (
                    beta_phi
                    * result[index, radial_index, 7]
                    * field[index, radial_index]
                )
                right[wall_number, component] = -(
                    interior + normalization + phi_coupling
                )
        solved, field_evidence = solve_selective_wall_endpoint_block(
            matrix,
            full_rows,
            right,
            metric_field_names,
            radial_index=radial_index,
        )
        for component, (name, evidence_name) in enumerate(zip(
            fields, metric_field_names,
        )):
            accelerations[name][0, radial_index] = solved[0, component]
            accelerations[name][-1, radial_index] = solved[1, component]
            evidence = field_evidence[evidence_name]
            algebraic_records[evidence_name].append(evidence)
            minimum_rank = min(minimum_rank, int(evidence["rank"]))
            maximum_condition = max(
                maximum_condition, float(evidence["equilibrated_condition"]),
            )
            weakest_pivot = min(
                weakest_pivot, float(evidence["normalized_pivot"]),
            )

    # The chi block has the same two compact endpoints at every radius.
    chi_matrix = dz[[0, -1]][:, [0, -1]]
    chi_full_rows = dz[[0, -1]]
    chi_right = -(chi_full_rows[:, 1:-1] @ result[1:-1, :, 8])
    for radial_index in range(len(r)):
        chi_solved, chi_evidence = solve_selective_wall_endpoint_block(
            chi_matrix,
            chi_full_rows,
            chi_right[:, radial_index, None],
            ("chi",),
            radial_index=radial_index,
        )
        result[0, radial_index, 8] = chi_solved[0, 0]
        result[-1, radial_index, 8] = chi_solved[1, 0]
        evidence = chi_evidence["chi"]
        algebraic_records["chi"].append(evidence)
    algebraic_evidence = summarize_selective_field_evidence(
        algebraic_records, selective_field_names,
    )
    chi_rank = algebraic_evidence["fields"]["chi"]["minimum_rank"]

    # Map the four physical tangential accelerations back to the regular
    # storage.  q4 and q5 at r=0 are invisible to the physical tensor and are
    # intentionally not inferred here.
    result[[0, -1], :, 2] = accelerations["tt"][[0, -1]]
    result[[0, -1], :, 3] = accelerations["sphere"][[0, -1]]
    result[0, 1:, 4] = (
        accelerations["rr"][0, 1:] - accelerations["sphere"][0, 1:]
    ) / r[1:]**2
    result[-1, 1:, 4] = (
        accelerations["rr"][-1, 1:] - accelerations["sphere"][-1, 1:]
    ) / r[1:]**2
    result[0, 1:, 5] = accelerations["tr"][0, 1:] / r[1:]
    result[-1, 1:, 5] = accelerations["tr"][-1, 1:] / r[1:]

    protected_after = np.ascontiguousarray(
        result[:, :, protected_channels]
    ).view(np.uint64)
    null_axis_after = np.ascontiguousarray(result[:, 0, 4:6]).view(np.uint64)
    protected_bitwise = bool(np.array_equal(protected_before, protected_after))
    null_axis_bitwise = bool(np.array_equal(null_axis_before, null_axis_after))
    if not protected_bitwise:
        raise AssertionError(
            "tangential/chi owner changed protected 0/1 or 6/7 channels"
        )
    if not null_axis_bitwise:
        raise AssertionError(
            "tangential/chi owner performed an implicit q4/q5 axis fill"
        )

    final_accelerations = {
        "tt": result[:, :, 2],
        "sphere": result[:, :, 3],
        "rr": result[:, :, 3] + radius2 * result[:, :, 4],
        "tr": radius * result[:, :, 5],
    }
    physical_acceleration = np.empty_like(result)
    physical_acceleration[:, :, 0] = result[:, :, 0]
    physical_acceleration[:, :, 1] = radius * result[:, :, 1]
    physical_acceleration[:, :, 2] = final_accelerations["tt"]
    physical_acceleration[:, :, 3] = final_accelerations["sphere"]
    physical_acceleration[:, :, 4] = final_accelerations["rr"]
    physical_acceleration[:, :, 5] = final_accelerations["tr"]
    physical_acceleration[:, :, 6:] = result[:, :, 6:]
    direct_physical_a_z = np.einsum(
        "ij,jrf->irf", dz, physical_acceleration,
    )[[0, -1]]
    row_implied_physical_a_z = direct_physical_a_z.copy()
    row_defined_mask = np.asarray(
        (False, False, True, True, True, True, False, False, True),
        dtype=bool,
    )
    records = []
    maximum_metric = 0.0
    maximum_chi = 0.0
    component_slots = {"tt": 2, "sphere": 3, "rr": 4, "tr": 5}
    for wall_number, wall in enumerate(wall_data):
        index = wall["index"]
        A = wall["A"]
        beta = wall["beta"]
        beta_phi = wall["beta_phi"]
        orientation = wall["orientation"]
        wall_components = {}
        for name, field in fields.items():
            acceleration_field = final_accelerations[name]
            field_z = (dz @ field)[index]
            acceleration_z = (dz @ acceleration_field)[index]
            terms = np.stack((
                acceleration_z / (2.0 * A),
                -field_z * result[index, :, 6] / (4.0 * A**3),
                beta_phi * result[index, :, 7] * field[index],
                beta * acceleration_field[index],
            ))
            residual = orientation * np.sum(terms, axis=0)
            derivative_absolute_sum = np.sum(
                np.abs(
                    dz[index, :, None]
                    * acceleration_field[:, :]
                    / (2.0 * A[None, :])
                ),
                axis=0,
            )
            scale = np.maximum(
                1.0,
                derivative_absolute_sum + np.sum(np.abs(terms[1:]), axis=0),
            )
            normalized = np.abs(residual) / scale
            maximum_metric = max(maximum_metric, float(np.max(normalized)))
            implied = (
                field_z * result[index, :, 6] / (2.0 * A**2)
                - 2.0 * A * beta_phi * result[index, :, 7] * field[index]
                - 2.0 * A * beta * acceleration_field[index]
            )
            row_implied_physical_a_z[
                wall_number, :, component_slots[name]
            ] = implied
            component_record = {
                "residual": residual,
                "scale": scale,
                "normalized": normalized,
                "maximum_normalized": float(np.max(normalized)),
            }
            if capture_profiles:
                component_record.update({
                    "terms": terms,
                    "direct_physical_a_z": acceleration_z,
                    "row_implied_physical_a_z": implied,
                })
            wall_components[name] = component_record

        chi_contributions = dz[index, :, None] * result[:, :, 8]
        chi_residual = np.sum(chi_contributions, axis=0)
        chi_scale = np.maximum(1.0, np.sum(np.abs(chi_contributions), axis=0))
        chi_normalized = np.abs(chi_residual) / chi_scale
        maximum_chi = max(maximum_chi, float(np.max(chi_normalized)))
        row_implied_physical_a_z[wall_number, :, 8] = 0.0
        chi_record = {
            "residual": chi_residual,
            "scale": chi_scale,
            "normalized": chi_normalized,
            "maximum_normalized": float(np.max(chi_normalized)),
        }
        if capture_profiles:
            chi_record.update({
                "contributions": chi_contributions,
                "direct_physical_a_z": direct_physical_a_z[
                    wall_number, :, 8
                ],
                "row_implied_physical_a_z": np.zeros(len(r)),
            })
        records.append({
            "wall": wall["wall"],
            "radial_indices": np.arange(len(r)),
            "components": wall_components,
            "chi": chi_record,
            "maximum_metric_normalized": max(
                item["maximum_normalized"] for item in wall_components.values()
            ),
            "maximum_chi_normalized": chi_record["maximum_normalized"],
        })

    scored = row_defined_mask[None, None, :]
    denominator = np.maximum.reduce((
        np.ones_like(direct_physical_a_z),
        np.abs(direct_physical_a_z),
        np.abs(row_implied_physical_a_z),
    ))
    row_defect = np.where(
        scored,
        np.abs(direct_physical_a_z - row_implied_physical_a_z) / denominator,
        0.0,
    )
    worst = max(maximum_metric, maximum_chi)
    if not np.isfinite(worst) or worst >= limit:
        raise RuntimeError(
            "normalized tangential/chi wall residual gate failed: "
            f"residual={worst}, limit={limit}"
        )
    return result, {
        "method": "direct_normalized_time_symmetric_DX2J_plus_chi",
        "time_symmetric": True,
        "maximum_allowed_normalized_residual": limit,
        "maximum_metric_normalized_residual": maximum_metric,
        "maximum_chi_normalized_residual": maximum_chi,
        "minimum_tangential_endpoint_rank": minimum_rank,
        "maximum_tangential_endpoint_condition": maximum_condition,
        "weakest_tangential_endpoint_pivot": weakest_pivot,
        "chi_endpoint_rank": chi_rank,
        "per_field_algebraic_evidence": algebraic_evidence,
        "protected_0_1_6_7_bitwise": protected_bitwise,
        "q4_q5_axis_bitwise": null_axis_bitwise,
        "walls": records,
        "direct_physical_a_z": direct_physical_a_z,
        "row_implied_physical_a_z": row_implied_physical_a_z,
        "row_defined_mask": row_defined_mask,
        "maximum_row_implied_scaled_defect": float(np.max(row_defect)),
        "passed": True,
    }


def impose_compact_wall_normal_tangential_acceleration(acceleration):
    """Impose only the two normal-tangential wall gauge accelerations."""
    result = np.asarray(acceleration, dtype=float).copy()
    if result.ndim != 3 or result.shape[2] != FIELD_COUNT:
        raise ValueError("invalid compact-wall acceleration fields")
    result[0, :, 0:2] = 0.0
    result[-1, :, 0:2] = 0.0
    return result


def compact_wall_normal_gauge_position_residuals(
    position,source,z,r,background,stencil_width=7,radial_buffer=7,
):
    """Evaluate the nonlinear normal GH constraint after Israel substitution.

    With normal-tangential metric components fixed to zero at a compact wall,
    ``C_z=0`` and the four tangential Israel rows reduce to

    ``g_zz,z + 8 beta g_zz^(3/2) - 2 H_z g_zz = 0``.
    """
    q=np.asarray(position,dtype=float);h=np.asarray(source,dtype=float)
    z=np.asarray(z,dtype=float);r=np.asarray(r,dtype=float)
    if q.shape[:2]!=(len(z),len(r)) or q.shape[-1]!=9 or h.shape!=(len(z),len(r),3):
        raise ValueError("invalid normal gauge wall fields")
    dz=derivative_matrix(z,1,stencil_width)
    if hasattr(dz,"toarray"):dz=dz.toarray()
    retained=slice(None,-int(radial_buffer)) if radial_buffer else slice(None)
    G=q[:,:,6];records=[]
    for wall,index,upper in (("lower",0,False),("upper",-1,True)):
        _,beta,_,_=_wall_coefficients(q[index,:,7],background,upper)
        terms=((dz@G)[index],8*beta*G[index]**1.5,-2*h[index,:,1]*G[index])
        residual=sum(terms);scale=np.maximum(1.,sum(np.abs(term) for term in terms))
        records.append({
            "wall":wall,
            "maximum_normalized":float(np.max(np.abs(residual[retained])/scale[retained])),
            "maximum_absolute":float(np.max(np.abs(residual[retained]))),
        })
    return {"walls":records,"maximum":max(item["maximum_normalized"] for item in records)}


def solve_compact_wall_normal_gauge_acceleration(
    position,velocity,acceleration,source,source_time,source_second_time,
    z,r,background,stencil_width=7,
):
    """Solve the twice-time-differentiated normal GH wall row for ``g_zz,tt``."""
    q=np.asarray(position,dtype=float);v=np.asarray(velocity,dtype=float)
    a=np.asarray(acceleration,dtype=float).copy()
    h=np.asarray(source,dtype=float);ht=np.asarray(source_time,dtype=float)
    htt=np.asarray(source_second_time,dtype=float)
    z=np.asarray(z,dtype=float);r=np.asarray(r,dtype=float)
    expected=(len(z),len(r),3)
    if (
        q.shape!=v.shape or q.shape!=a.shape or q.shape[-1]!=9
        or h.shape!=expected or ht.shape!=expected or htt.shape!=expected
    ):
        raise ValueError("invalid normal gauge acceleration fields")
    dz=derivative_matrix(z,1,stencil_width)
    if hasattr(dz,"toarray"):dz=dz.toarray()
    before=np.stack((a[0,:,6].copy(),a[-1,:,6].copy()))
    for _,index,upper in (("lower",0,False),("upper",-1,True)):
        G=q[index,:,6];Gt=v[index,:,6]
        _,beta,beta_phi,beta_phiphi=_wall_coefficients(
            q[index,:,7],background,upper,
        )
        beta_t=beta_phi*v[index,:,7]
        beta_tt=beta_phi*a[index,:,7]+beta_phiphi*v[index,:,7]**2
        robin=12*beta*np.sqrt(G)-2*h[index,:,1]
        forcing=(
            8*(
                beta_tt*G**1.5+3*beta_t*np.sqrt(G)*Gt
                +.75*beta*Gt**2/np.sqrt(G)
            )
            -2*(htt[index,:,1]*G+2*ht[index,:,1]*Gt)
        )
        _solve_endpoint(dz,a[:,:,6],index,robin,forcing)
    after=np.stack((a[0,:,6],a[-1,:,6]))
    return a,{
        "relative_correction":float(
            np.linalg.norm(after-before)/max(np.linalg.norm(after),np.linalg.norm(before),1e-300)
        ),
        "maximum_absolute_correction":float(np.max(np.abs(after-before))),
    }


def solve_compact_wall_coupled_phi_normal_acceleration(
    position,velocity,acceleration,source,source_time,source_second_time,
    z,r,background,stencil_width=7,maximum_condition=1e12,
    minimum_pivot_strength=1e-10,maximum_normalized_linear_residual=1e-12,
    *, capture_profiles=False,
):
    """Solve the coupled ``Phi_tt``/``g_zz,tt`` wall block directly.

    The stabilizer acceleration row contains ``g_zz,tt`` through
    ``sqrt(g_zz),tt`` while the normal GH row contains ``Phi_tt`` through
    ``beta,tt``.  The legacy closure alternates those two scalar rows four
    times.  This opt-in helper instead solves the exact coupled four-by-four
    block containing both compact-wall endpoints at every radial node.  It
    fails closed on deficient rank, poor row-equilibrated conditioning, or a
    weak endpoint pivot relative to the full compact-stencil rows.  Source and
    driver fields are frozen data; no source or interface equation is added.
    """
    q=np.asarray(position,dtype=float);v=np.asarray(velocity,dtype=float)
    a=np.asarray(acceleration,dtype=float).copy()
    h=np.asarray(source,dtype=float);ht=np.asarray(source_time,dtype=float)
    htt=np.asarray(source_second_time,dtype=float)
    z=np.asarray(z,dtype=float);r=np.asarray(r,dtype=float)
    expected=(len(z),len(r),3)
    if (
        q.shape!=v.shape or q.shape!=a.shape or q.shape!=(len(z),len(r),FIELD_COUNT)
        or h.shape!=expected or ht.shape!=expected or htt.shape!=expected
    ):
        raise ValueError("invalid coupled compact-wall acceleration fields")
    maximum_condition=float(maximum_condition)
    minimum_pivot_strength=float(minimum_pivot_strength)
    maximum_normalized_linear_residual=float(maximum_normalized_linear_residual)
    if (
        not np.isfinite(maximum_condition) or maximum_condition<=1.
        or not np.isfinite(minimum_pivot_strength)
        or not (0.<minimum_pivot_strength<1.)
        or not np.isfinite(maximum_normalized_linear_residual)
        or maximum_normalized_linear_residual<=0.
    ):
        raise ValueError("invalid coupled-wall algebraic threshold")
    dz=derivative_matrix(z,1,stencil_width)
    if hasattr(dz,"toarray"):dz=dz.toarray()
    endpoints=(0,-1)
    specifications=((0,False,-1.),(-1,True,1.))
    worst_condition=0.;worst_raw_condition=0.;worst_residual=0.
    minimum_rank=4
    weakest_pivot=float("inf");worst_condition_index=0;weakest_pivot_index=0
    maximum_correction=0.
    old_squared=0.;new_squared=0.;correction_squared=0.
    profiles={
        "rank":np.empty(len(r),dtype=int),
        "equilibrated_condition":np.empty(len(r)),
        "raw_condition":np.empty(len(r)),
        "pivot_strength":np.empty(len(r)),
        "normalized_linear_residual":np.empty(len(r)),
        "maximum_absolute_endpoint_correction":np.empty(len(r)),
    }
    for radial_index in range(len(r)):
        matrix=np.zeros((4,4));full=np.zeros((4,2*len(z)))
        right=np.zeros(4)
        old=np.asarray((
            a[0,radial_index,7],a[-1,radial_index,7],
            a[0,radial_index,6],a[-1,radial_index,6],
        ))
        for wall_number,(index,upper,sign) in enumerate(specifications):
            G=float(q[index,radial_index,6]);Gt=float(v[index,radial_index,6])
            if G<=0. or not np.isfinite(G):
                raise RuntimeError("nonpositive compact-normal metric in coupled wall solve")
            A=np.sqrt(G);At=Gt/(2.*A)
            phi=float(q[index,radial_index,7]);phit=float(v[index,radial_index,7])
            target,beta,beta_phi,beta_phiphi=_wall_coefficients(
                phi,background,upper,
            )
            delta=phi-target;gamma=float(background["wall_stiffness"])
            phi_factor=sign*.5*gamma
            phi_row=2*wall_number
            normal_row=phi_row+1
            own_phi=wall_number
            own_normal=2+wall_number
            full_endpoint=(0 if wall_number==0 else len(z)-1)

            matrix[phi_row,0:2]=dz[index,[0,-1]]
            matrix[phi_row,own_phi]+=phi_factor*A
            matrix[phi_row,own_normal]=phi_factor*delta/(2.*A)
            full[phi_row,:len(z)]=dz[index]
            full[phi_row,full_endpoint]+=phi_factor*A
            full[phi_row,len(z)+full_endpoint]=phi_factor*delta/(2.*A)

            matrix[normal_row,2:4]=dz[index,[0,-1]]
            matrix[normal_row,own_normal]+=(
                12.*beta*A-2.*float(h[index,radial_index,1])
            )
            matrix[normal_row,own_phi]=8.*beta_phi*G**1.5
            full[normal_row,len(z):]=dz[index]
            full[normal_row,len(z)+full_endpoint]+=(
                12.*beta*A-2.*float(h[index,radial_index,1])
            )
            full[normal_row,full_endpoint]=8.*beta_phi*G**1.5

            interior=np.arange(1,len(z)-1)
            phi_constant=float(
                dz[index,interior]@a[interior,radial_index,7]
            )+phi_factor*(
                -delta*Gt**2/(4.*A**3)+2.*phit*At
            )
            beta_t=beta_phi*phit
            normal_constant=float(
                dz[index,interior]@a[interior,radial_index,6]
            )+8.*(
                beta_phiphi*phit**2*G**1.5
                +3.*beta_t*A*Gt+.75*beta*Gt**2/A
            )-2.*(
                float(htt[index,radial_index,1])*G
                +2.*float(ht[index,radial_index,1])*Gt
            )
            right[phi_row]=-phi_constant
            right[normal_row]=-normal_constant

        row_norms=np.linalg.norm(full,axis=1)
        if np.any(row_norms<=0.) or not np.all(np.isfinite(row_norms)):
            raise CompactWallCoupledAlgebraicGateError(
                "degenerate full compact-wall row in coupled solve",
                radial_index=radial_index,
                gate="finite_positive_full_row_norm",
                diagnostics={"row_norms": row_norms.tolist()},
            )
        equilibrated=matrix/row_norms[:,None]
        singular=np.linalg.svd(equilibrated,compute_uv=False)
        rank=int(np.linalg.matrix_rank(equilibrated))
        minimum_rank=min(minimum_rank,rank)
        pivot_strength=float(singular[-1]) if len(singular) else 0.
        condition=float(singular[0]/singular[-1]) if pivot_strength>0. else float("inf")
        raw_condition=float(np.linalg.cond(matrix))
        if (
            rank<4 or not np.isfinite(condition) or condition>maximum_condition
            or pivot_strength<minimum_pivot_strength
        ):
            raise CompactWallCoupledAlgebraicGateError(
                "coupled Phi/g_zz compact-wall endpoint block failed its algebraic gate: "
                f"radial_index={radial_index}, rank={rank}, "
                f"equilibrated_condition={condition}, pivot_strength={pivot_strength}",
                radial_index=radial_index,
                gate="rank_condition_pivot",
                diagnostics={
                    "rank":rank,
                    "equilibrated_condition":condition,
                    "pivot_strength":pivot_strength,
                    "maximum_allowed_condition":maximum_condition,
                    "minimum_allowed_pivot_strength":minimum_pivot_strength,
                },
            )
        solved=np.linalg.solve(matrix,right)
        a[0,radial_index,7],a[-1,radial_index,7]=solved[0:2]
        a[0,radial_index,6],a[-1,radial_index,6]=solved[2:4]
        residual=matrix@solved-right
        normalized=float(
            np.max(np.abs(residual))
            /max(1.,np.linalg.norm(matrix,ord=np.inf)*np.max(np.abs(solved)),
                 np.max(np.abs(right)))
        )
        if (
            not np.isfinite(normalized)
            or normalized>=maximum_normalized_linear_residual
        ):
            raise CompactWallCoupledAlgebraicGateError(
                "coupled Phi/g_zz compact-wall endpoint block failed its "
                "normalized linear-residual gate: "
                f"radial_index={radial_index}, residual={normalized}, "
                f"limit={maximum_normalized_linear_residual}",
                radial_index=radial_index,
                gate="normalized_linear_residual",
                diagnostics={
                    "normalized_linear_residual":normalized,
                    "maximum_allowed_normalized_linear_residual":(
                        maximum_normalized_linear_residual
                    ),
                },
            )
        if condition>=worst_condition:
            worst_condition=condition;worst_condition_index=radial_index
        if pivot_strength<=weakest_pivot:
            weakest_pivot=pivot_strength;weakest_pivot_index=radial_index
        worst_raw_condition=max(worst_raw_condition,raw_condition)
        worst_residual=max(worst_residual,normalized)
        local_correction=float(np.max(np.abs(solved-old)))
        maximum_correction=max(maximum_correction,local_correction)
        profiles["rank"][radial_index]=rank
        profiles["equilibrated_condition"][radial_index]=condition
        profiles["raw_condition"][radial_index]=raw_condition
        profiles["pivot_strength"][radial_index]=pivot_strength
        profiles["normalized_linear_residual"][radial_index]=normalized
        profiles["maximum_absolute_endpoint_correction"][radial_index]=(
            local_correction
        )
        old_squared+=float(old@old)
        new_squared+=float(solved@solved)
        correction_squared+=float((solved-old)@(solved-old))
    record={
        "method":"direct_radial_4x4_both_walls_Phi_gzz",
        "maximum_allowed_condition":maximum_condition,
        "minimum_allowed_pivot_strength":minimum_pivot_strength,
        "maximum_allowed_normalized_linear_residual":(
            maximum_normalized_linear_residual
        ),
        "maximum_condition":float(worst_condition),
        "maximum_raw_condition":float(worst_raw_condition),
        "minimum_pivot_strength":float(weakest_pivot),
        "worst_condition_radial_index":int(worst_condition_index),
        "worst_condition_radius":float(r[worst_condition_index]),
        "weakest_pivot_radial_index":int(weakest_pivot_index),
        "weakest_pivot_radius":float(r[weakest_pivot_index]),
        "maximum_normalized_linear_residual":float(worst_residual),
        "minimum_rank":int(minimum_rank),
        "maximum_absolute_endpoint_correction":float(maximum_correction),
        "relative_correction":float(
            np.sqrt(correction_squared)
            /max(np.sqrt(old_squared),np.sqrt(new_squared),1e-300)
        ),
        "passed":bool(
            minimum_rank==4 and worst_condition<=maximum_condition
            and weakest_pivot>=minimum_pivot_strength
            and worst_residual<maximum_normalized_linear_residual
        ),
    }
    if capture_profiles:
        record["profiles"]={
            name:value.tolist() for name,value in profiles.items()
        }
    return a,record


def compact_wall_normal_gauge_acceleration_residuals(
    position,velocity,acceleration,source,source_time,source_second_time,
    z,r,background,stencil_width=7,radial_buffer=7,*,capture_profiles=False,
):
    """Evaluate the twice-time-differentiated normal GH wall rows."""
    q=np.asarray(position,dtype=float);v=np.asarray(velocity,dtype=float)
    a=np.asarray(acceleration,dtype=float);h=np.asarray(source,dtype=float)
    ht=np.asarray(source_time,dtype=float);htt=np.asarray(source_second_time,dtype=float)
    dz=derivative_matrix(np.asarray(z,dtype=float),1,stencil_width)
    if hasattr(dz,"toarray"):dz=dz.toarray()
    retained=slice(None,-int(radial_buffer)) if radial_buffer else slice(None)
    records=[]
    for wall,index,upper in (("lower",0,False),("upper",-1,True)):
        G=q[index,:,6];Gt=v[index,:,6]
        _,beta,beta_phi,beta_phiphi=_wall_coefficients(q[index,:,7],background,upper)
        beta_t=beta_phi*v[index,:,7]
        beta_tt=beta_phi*a[index,:,7]+beta_phiphi*v[index,:,7]**2
        terms=(
            (dz@a[:,:,6])[index],
            (12*beta*np.sqrt(G)-2*h[index,:,1])*a[index,:,6],
            8*(beta_tt*G**1.5+3*beta_t*np.sqrt(G)*Gt+.75*beta*Gt**2/np.sqrt(G)),
            -2*(htt[index,:,1]*G+2*ht[index,:,1]*Gt),
        )
        residual=sum(terms);scale=np.maximum(1.,sum(np.abs(term) for term in terms))
        record={
            "wall":wall,
            "maximum_normalized":float(np.max(np.abs(residual[retained])/scale[retained])),
            "maximum_absolute":float(np.max(np.abs(residual[retained]))),
        }
        if capture_profiles:
            normalized=np.abs(residual)/scale
            maximum_index=int(np.argmax(normalized[retained]))
            if radial_buffer:
                maximum_index=min(maximum_index,len(r)-int(radial_buffer)-1)
            record["profiles"]={
                "terms":[np.asarray(term).tolist() for term in terms],
                "residual":np.asarray(residual).tolist(),
                "scale":np.asarray(scale).tolist(),
                "normalized":np.asarray(normalized).tolist(),
                "maximum_index":maximum_index,
                "maximum_radius":float(r[maximum_index]),
            }
        records.append(record)
    return {"walls":records,"maximum":max(item["maximum_normalized"] for item in records)}


class NativeRegularSO3RHS:
    """Nonlinear native-grid RHS with a short-time gauge/wall closure."""

    def __init__(
        self, z, r, gauge_source, mass_squared, background,
        normal_wall_acceleration, stencil_width=7,live_normal_wall_gauge=False,
        live_outer_sommerfeld=False,
        boundary_closure_mode="legacy_wall_axis_outer",
    ):
        self.z = np.asarray(z, dtype=float)
        self.r = np.asarray(r, dtype=float)
        self.gauge_source = gauge_source
        self.mass_squared = float(mass_squared)
        self.background = background
        self.normal_wall_acceleration = np.asarray(
            normal_wall_acceleration, dtype=float,
        )
        self.stencil_width = int(stencil_width)
        self.live_normal_wall_gauge=bool(live_normal_wall_gauge)
        self.live_outer_sommerfeld=bool(live_outer_sommerfeld)
        if boundary_closure_mode not in BOUNDARY_CLOSURE_MODES:
            raise ValueError(
                "boundary_closure_mode must be one of "
                f"{BOUNDARY_CLOSURE_MODES}"
            )
        self.boundary_closure_mode=str(boundary_closure_mode)
        self.outer_reference_position=None
        self.outer_reference_acceleration=None

    def set_outer_sommerfeld_reference(self,position,acceleration):
        """Set the quadratic reference used by the live outer boundary."""
        position=np.asarray(position,dtype=float);acceleration=np.asarray(acceleration,dtype=float)
        expected=(len(self.z),len(self.r),FIELD_COUNT)
        if position.shape!=expected or acceleration.shape!=expected:
            raise ValueError("invalid outer Sommerfeld reference")
        self.outer_reference_position=position.copy()
        self.outer_reference_acceleration=acceleration.copy()
        self.live_outer_sommerfeld=True

    def acceleration(
        self,time,position,velocity,gauge_source=None,gauge_source_second_time=None,
        *,capture_boundary_stages=False,
    ):
        boundary_stages=[] if capture_boundary_stages else None

        def capture(name, acceleration, **metadata):
            if boundary_stages is not None:
                boundary_stages.append({
                    "name":str(name),
                    "acceleration":np.asarray(acceleration,dtype=float).copy(),
                    **metadata,
                })

        first, second = reduced_state_jets(
            position, velocity, self.z, self.r, self.stencil_width,
        )
        active_source=self.gauge_source if gauge_source is None else gauge_source
        result = np.zeros_like(position, dtype=float)
        finite = True
        inverse_time_maximum = 0.0
        # Positive-radius points provide the regular axis limit.  Physical
        # compact-wall endpoints are solved after this open-bulk pass.
        for i in range(len(self.z)):
            for j in range(1, len(self.r)):
                jets = regular_so3_perturbation_jets(
                    float(self.r[j]), position[i, j], first[:, i, j],
                    second[:, :, i, j],
                )
                source, source_first = active_source.at(i, j, time)
                solved = solve_reduced_einstein_two_scalar_acceleration(
                    jets["metric"], jets["metric_first"], jets["metric_second"],
                    jets["phi"], jets["phi_first"], jets["phi_second"],
                    jets["chi"], jets["chi_first"], jets["chi_second"],
                    source, source_first, mass_squared=self.mass_squared,
                    potential_offset=-6.0, kappa5_squared=1.0,
                )
                result[i, j] = _pack_acceleration(
                    solved["metric_acceleration"], solved["phi_acceleration"],
                    solved["chi_acceleration"], float(self.r[j]),
                )
                finite = finite and solved["finite"]
                inverse_time_maximum = max(
                    inverse_time_maximum, abs(solved["inverse_time_metric"]),
                )
        capture("bulk_positive_radius",result)
        # Preserve the historical pre-wall fill here.  The wall algebra owns
        # its endpoint solve and must see the same reduced input it was derived
        # with.  Operator-specific axis values are installed only after the
        # final wall/outer owners have completed below.
        result = fill_regular_axis(result, self.r)
        capture("initial_axis_fill",result)
        outer_diagnostic=None
        if self.live_outer_sommerfeld:
            if self.outer_reference_position is None or self.outer_reference_acceleration is None:
                raise ValueError("live outer Sommerfeld boundary requires a reference")
            if self.boundary_closure_mode=="wall_owner_last_experimental":
                result,outer_diagnostic=apply_outer_sommerfeld_acceleration(
                    position,velocity,result,self.outer_reference_position,
                    self.outer_reference_acceleration,time,self.r,self.stencil_width,
                )
                capture("outer_open_face_before_wall",result)

        legacy_axis_fill=(self.boundary_closure_mode=="legacy_wall_axis_outer")

        all_wall_corrections=[]

        def compact_pass(current,normal_data,label,iteration=None):
            current,pass_corrections=apply_compact_wall_acceleration(
                position,velocity,current,self.z,self.r,self.background,
                normal_data,self.stencil_width,fill_axis_after=False,
            )
            all_wall_corrections.extend({
                **record, "stage": str(label),
                **({} if iteration is None else {"iteration": int(iteration)}),
            } for record in pass_corrections)
            capture(
                f"{label}_wall_endpoint_solve",current,
                **({} if iteration is None else {"iteration":int(iteration)}),
            )
            if legacy_axis_fill:
                current=_finish_compact_wall_axis_fill(
                    current,self.r,normal_data,
                )
                capture(
                    f"{label}_post_wall_axis_fill",current,
                    **({} if iteration is None else {"iteration":int(iteration)}),
                )
            return current,pass_corrections

        normal_diagnostic=None
        if self.live_normal_wall_gauge:
            if (
                not hasattr(active_source,"source")
                or not hasattr(active_source,"source_time")
                or gauge_source_second_time is None
            ):
                raise ValueError("live normal wall gauge requires live source through second time")
            source_second=np.asarray(gauge_source_second_time,dtype=float)
            if self.boundary_closure_mode=="wall_owner_last_experimental":
                result,coupled_record=(
                    solve_compact_wall_coupled_phi_normal_acceleration(
                        position,velocity,result,active_source.source,
                        active_source.source_time,source_second,self.z,self.r,
                        self.background,self.stencil_width,
                    )
                )
                capture("coupled_Phi_gzz_wall_solve",result)
                normal=np.stack((result[0,:,6],result[-1,:,6]))
                result,corrections=compact_pass(
                    result,normal,"final_compact",
                )
                normal_residual=compact_wall_normal_gauge_acceleration_residuals(
                    position,velocity,result,active_source.source,
                    active_source.source_time,source_second,self.z,self.r,
                    self.background,self.stencil_width,radial_buffer=0,
                )
                if (
                    not np.isfinite(normal_residual["maximum"])
                    or normal_residual["maximum"]>=1e-10
                ):
                    raise RuntimeError(
                        "wall-owner-last closure failed its full unbuffered "
                        "normal-row residual gate: "
                        f"residual={normal_residual['maximum']}, limit=1e-10"
                    )
                normal_diagnostic={
                    "method":"direct_coupled_Phi_gzz_then_wall_rows",
                    "coupled_block":coupled_record,
                    "iterations":[],
                    "radial_buffer":0,
                    "final_residual":normal_residual,
                    "passed":bool(
                        coupled_record["passed"]
                        and normal_residual["maximum"]<1e-10
                    ),
                }
            else:
                normal=np.stack((result[0,:,6],result[-1,:,6]))
                normal_records=[]
                for iteration in range(4):
                    result,corrections=compact_pass(
                        result,normal,"normal_iteration",iteration,
                    )
                    result,record=solve_compact_wall_normal_gauge_acceleration(
                        position,velocity,result,active_source.source,
                        active_source.source_time,source_second,self.z,self.r,
                        self.background,self.stencil_width,
                    )
                    capture(
                        "normal_iteration_gzz_solve",result,
                        iteration=int(iteration),
                    )
                    normal=np.stack((result[0,:,6],result[-1,:,6]))
                    normal_records.append(record)
                result,corrections=compact_pass(
                    result,normal,"final_compact",
                )
                normal_residual=compact_wall_normal_gauge_acceleration_residuals(
                    position,velocity,result,active_source.source,
                    active_source.source_time,source_second,self.z,self.r,
                    self.background,self.stencil_width,
                )
                normal_diagnostic={
                    "method":"legacy_four_fixed_point_iterations",
                    "iterations":normal_records,
                    "radial_buffer":7,
                    "final_residual":normal_residual,
                }
        else:
            result, corrections = compact_pass(
                result,self.normal_wall_acceleration,"final_compact",
            )
        if legacy_axis_fill:
            capture("pre_outer",result)
        else:
            result=reconcile_wall_owner_axis_null_channels(result,self.r)
            capture(
                "post_wall_owner_reconciliation",result,
                reconciled_axis_channels=[1,4,5],
                reconciliation_reason="physical_parity_quotient_channels_at_r_zero",
            )
        if (
            self.live_outer_sommerfeld
            and self.boundary_closure_mode=="legacy_wall_axis_outer"
        ):
            result,outer_diagnostic=apply_outer_sommerfeld_acceleration(
                position,velocity,result,self.outer_reference_position,
                self.outer_reference_acceleration,time,self.r,self.stencil_width,
            )
            capture("post_outer",result)

        if legacy_axis_fill:
            result = reconcile_wall_owner_axis_null_channels(result, self.r)
        pointwise_fields = (0, 2, 3, 6, 7, 8)
        for i in range(1, len(self.z) - 1):
            jets = regular_so3_perturbation_jets(
                0.0, position[i, 0], first[:, i, 0], second[:, :, i, 0],
            )
            source, source_first = active_source.at(i, 0, time)
            solved = solve_reduced_einstein_two_scalar_acceleration(
                jets["metric"], jets["metric_first"], jets["metric_second"],
                jets["phi"], jets["phi_first"], jets["phi_second"],
                jets["chi"], jets["chi_first"], jets["chi_second"],
                source, source_first, mass_squared=self.mass_squared,
                potential_offset=-6.0, kappa5_squared=1.0,
            )
            metric = np.asarray(solved["metric_acceleration"], dtype=float)
            transverse = 0.5 * (metric[3, 3] + metric[4, 4])
            result[i, 0, pointwise_fields] = np.asarray((
                metric[1, 0], metric[0, 0], transverse, metric[1, 1],
                solved["phi_acceleration"], solved["chi_acceleration"],
            ))
            finite = finite and solved["finite"]
            inverse_time_maximum = max(
                inverse_time_maximum, abs(solved["inverse_time_metric"]),
            )
        capture(
            "post_axis_operator_repair", result,
            direct_open_z_channels=list(pointwise_fields),
            native_quotient_channels=[1, 4, 5],
        )

        fitted_axis=_axis_fit(result[:,1:],self.r)
        axis_difference=result[:,0]-fitted_axis
        axis_scale=max(
            float(np.max(np.abs(result[:,0]))),
            float(np.max(np.abs(fitted_axis))),1e-300,
        )
        return result, {
            "finite": bool(finite and np.all(np.isfinite(result))),
            "wall_corrections": all_wall_corrections,
            "maximum_absolute_inverse_time_metric": inverse_time_maximum,
            "normal_wall_gauge":normal_diagnostic,
            "outer_sommerfeld":outer_diagnostic,
            "boundary_closure_mode":self.boundary_closure_mode,
            "boundary_stages":boundary_stages,
            "boundary_stage_names":(
                [stage["name"] for stage in boundary_stages]
                if boundary_stages is not None else None
            ),
            "boundary_parameters":{
                "stencil_width":self.stencil_width,
                "axis_fit_window":0.5,
                "axis_fit_degree":3,
                "normal_wall_method":(
                    "direct_coupled_4x4_both_walls"
                    if (
                        self.live_normal_wall_gauge
                        and not legacy_axis_fill
                    ) else (
                        "four_fixed_point_iterations"
                        if self.live_normal_wall_gauge else "fixed_gauge_datum"
                    )
                ),
                "normal_wall_iterations":(
                    4 if self.live_normal_wall_gauge and legacy_axis_fill else 0
                ),
            },
            "axis_fit_preference_defect":{
                "maximum_absolute":float(np.max(np.abs(axis_difference))),
                "wall_maximum_absolute":float(np.max(np.abs(axis_difference[[0,-1]]))),
                "relative":float(np.max(np.abs(axis_difference))/axis_scale),
                "by_field":{
                    name:float(np.max(np.abs(axis_difference[:,index])))
                    for index,name in enumerate(FIELD_ORDER)
                },
                "wall_by_field":{
                    name:float(np.max(np.abs(axis_difference[[0,-1],index])))
                    for index,name in enumerate(FIELD_ORDER)
                },
            },
        }


def compact_wall_position_residuals(position, z, r, background, stencil_width=7, radial_buffer=7):
    """Return normalized residuals of the undifferentiated compact-wall rows."""
    q = np.asarray(position, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    dz = derivative_matrix(z, 1, stencil_width)
    if hasattr(dz, "toarray"):
        dz = dz.toarray()
    radius2 = r[None, :] ** 2
    fields = {
        "g00": q[:, :, 2],
        "perp": q[:, :, 3],
        "radial": q[:, :, 3] + radius2 * q[:, :, 4],
        "g0r": r[None, :] * q[:, :, 5],
    }
    retained = slice(None, -int(radial_buffer)) if radial_buffer else slice(None)
    records = []
    for wall, index, upper, sign in (
        ("lower", 0, False, -1.0), ("upper", -1, True, 1.0),
    ):
        A = np.sqrt(q[index, :, 6])
        target, beta, _, _ = _wall_coefficients(q[index, :, 7], background, upper)
        rows = {}
        for name, field in fields.items():
            terms = ((dz @ field)[index], 2 * beta * A * field[index])
            scale = np.maximum(1.0, np.abs(terms[0]) + np.abs(terms[1]))
            rows[name] = float(np.max(np.abs(sum(terms))[retained] / scale[retained]))
        gamma = float(background["wall_stiffness"])
        phi_terms = (
            (dz @ q[:, :, 7])[index],
            sign * 0.5 * gamma * (q[index, :, 7] - target) * A,
        )
        phi_scale = np.maximum(1.0, np.abs(phi_terms[0]) + np.abs(phi_terms[1]))
        chi = (dz @ q[:, :, 8])[index]
        rows["phi"] = float(np.max(np.abs(sum(phi_terms))[retained] / phi_scale[retained]))
        rows["chi"] = float(np.max(np.abs(chi)[retained]))
        rows["h_z0_dirichlet"] = float(np.max(np.abs(q[index, retained, 0])))
        rows["h_zr_dirichlet"] = float(np.max(np.abs(r[retained] * q[index, retained, 1])))
        records.append({"wall": wall, "rows": rows, "maximum": max(rows.values())})
    return {"walls": records, "maximum": max(record["maximum"] for record in records)}


def gauge_constraint_summary(
    position,velocity,time,rhs,radial_cut=None,gauge_source=None,
):
    """Evaluate the global and maximum GH constraint on one native state."""
    first, second = reduced_state_jets(
        position, velocity, rhs.z, rhs.r, rhs.stencil_width,
    )
    constraints = np.empty((len(rhs.z), len(rhs.r), 5))
    contracted = np.empty_like(constraints)
    sources = np.empty_like(constraints)
    for i in range(len(rhs.z)):
        for j in range(len(rhs.r)):
            jets = regular_so3_perturbation_jets(
                float(rhs.r[j]), position[i, j], first[:, i, j],
                second[:, :, i, j],
            )
            geometry = metric_geometry_from_jets(
                jets["metric"], jets["metric_first"], jets["metric_second"],
            )
            active_source=rhs.gauge_source if gauge_source is None else gauge_source
            source, _ = active_source.at(i,j,time)
            contracted[i, j] = geometry["contracted_christoffel_covector"]
            sources[i, j] = source
            constraints[i, j] = contracted[i, j] - source
    mask = np.ones((len(rhs.z), len(rhs.r)), dtype=bool)
    if radial_cut is not None:
        mask &= rhs.r[None, :] <= float(radial_cut) + 1e-12
    # Approximate spherical volume weights; the axis cell retains finite
    # volume instead of disappearing under a raw r^2 nodal weight.
    edges = np.empty(len(rhs.r) + 1)
    edges[0] = 0.0
    edges[1:-1] = 0.5 * (rhs.r[:-1] + rhs.r[1:])
    edges[-1] = rhs.r[-1] + 0.5 * (rhs.r[-1] - rhs.r[-2])
    radial_weight = (edges[1:] ** 3 - edges[:-1] ** 3) / 3
    z_weight = np.ones(len(rhs.z))
    z_weight[[0, -1]] = 0.5
    weight = z_weight[:, None] * radial_weight[None, :]
    weight = weight[mask]
    c = constraints[mask]
    gamma = contracted[mask]
    source = sources[mask]
    norm = lambda value: float(np.sqrt(np.sum(weight[:, None] * value**2)))
    local_scale = np.maximum(
        np.linalg.norm(gamma, axis=1), np.linalg.norm(source, axis=1),
    )
    local_relative = np.linalg.norm(c, axis=1) / np.maximum(local_scale, 1e-300)
    return {
        "global_relative": norm(c) / max(norm(gamma), norm(source), 1e-300),
        "maximum_local_relative": float(np.max(local_relative)),
        "maximum_absolute": float(np.max(np.linalg.norm(c, axis=1))),
        "finite": bool(np.all(np.isfinite(constraints))),
    }
