"""Coupled thin-seam identities and controls for Test 14C.

The primary formulas are fixed in immutable note 99.  This module deliberately
keeps geometric joint work separate from the Test-14B separate-delta
comparator; callers must replace, rather than add, the latter.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import RectBivariateSpline

from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice
from bhps.test14b_balance_closure import (
    _axis_regularize_even,
    _field_and_gradient_on_surface,
    _field_on_surface,
    _one_cap_integral,
    _proper_curve,
    _proper_scalar_derivatives,
    _sample,
)


def relative_scale_error(value, expected, scale=1.0):
    """Symmetric relative error with an explicit reference scale."""
    return float(
        abs(float(value) - float(expected))
        / max(abs(float(value)), abs(float(expected)), abs(float(scale)), 1e-300)
    )


def seam_intrinsic_integral(sphere_area, israel_coefficient):
    """Return ``I_delta = 8 c A_2`` in the note-99 orientation."""
    return float(8.0 * float(israel_coefficient) * float(sphere_area))


def cap_boundary_variation(
    sphere_area, israel_coefficient, h_meridional, h_sphere,
    h_sphere_meridional_derivative,
):
    """Doubled cap boundary term in the scalar-curvature first variation."""
    return float(
        8.0 * float(sphere_area)
        * (
            -float(h_sphere_meridional_derivative)
            + float(israel_coefficient)
            * (float(h_meridional) - float(h_sphere))
        )
    )


def seam_intrinsic_derivative(
    sphere_area, israel_coefficient, israel_rate, h_sphere,
):
    """Direct derivative of ``I_delta=8 c A_2``."""
    return float(
        8.0 * float(sphere_area)
        * (float(israel_rate) + 2.0 * float(israel_coefficient) * h_sphere)
    )


def compatible_israel_rate(
    h_sphere_meridional_derivative, israel_coefficient, h_meridional,
):
    """Kinematic rate implied by ``c=W_s/W`` for a normal deformation."""
    return float(
        float(h_sphere_meridional_derivative)
        - float(israel_coefficient) * float(h_meridional)
    )


def combined_intrinsic_joint_variation(
    sphere_area, israel_coefficient, h_sphere,
):
    """Return the coupled limit ``D_cap + dot I_delta``."""
    return float(
        8.0 * float(sphere_area) * float(israel_coefficient) * float(h_sphere)
    )


def coupled_seam_rate(
    area_radius, leaf_area_fractional_rate, sphere_area,
    israel_coefficient, h_sphere,
):
    """Geometric coupled seam contribution to ``dot Q_raw``.

    This is equation (C14C.2).  It does not consume a charge derivative or a
    closure residual.
    """
    seam_integral = seam_intrinsic_integral(sphere_area, israel_coefficient)
    global_radius_part = (
        float(area_radius) * seam_integral
        * float(leaf_area_fractional_rate) / 12.0
    )
    joint_work = (
        2.0 * float(area_radius) * float(sphere_area)
        * float(israel_coefficient) * float(h_sphere)
    )
    return {
        "seam_intrinsic_integral": seam_integral,
        "global_radius_part": float(global_radius_part),
        "joint_work": float(joint_work),
        "total": float(global_radius_part + joint_work),
    }


def coupled_from_uncombined(
    sphere_area, israel_coefficient, israel_rate, h_meridional, h_sphere,
    h_sphere_meridional_derivative,
):
    """Evaluate the cap and seam pieces before their compatibility reduction."""
    boundary = cap_boundary_variation(
        sphere_area, israel_coefficient, h_meridional, h_sphere,
        h_sphere_meridional_derivative,
    )
    seam_rate = seam_intrinsic_derivative(
        sphere_area, israel_coefficient, israel_rate, h_sphere,
    )
    combined = combined_intrinsic_joint_variation(
        sphere_area, israel_coefficient, h_sphere,
    )
    return {
        "cap_boundary_variation": boundary,
        "seam_intrinsic_derivative": seam_rate,
        "uncombined_sum": float(boundary + seam_rate),
        "combined_limit": combined,
        "compatibility_error": relative_scale_error(
            boundary + seam_rate, combined, max(abs(combined), 1.0),
        ),
    }


def seam_endpoint_transport(
    position, velocity, z, r, profile, rho_rate, wall_stiffness, wall_target,
):
    """Evaluate local material/normal seam transport without charge history.

    At the fixed-coordinate upper brane the profile endpoint has
    ``theta=pi/2``.  Its coordinate velocity is retained explicitly so this
    path is independent of the area and charge finite differences.
    """
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    theta = float(np.asarray(profile["theta"], dtype=float)[-1])
    rho = float(np.asarray(profile["rho"], dtype=float)[-1])
    rho_rate = float(np.asarray(rho_rate, dtype=float)[-1])
    zcoord = float(z[-1] - rho * math.cos(theta))
    radius = float(rho * math.sin(theta))
    z_rate = float(-rho_rate * math.cos(theta))
    radius_rate = float(rho_rate * math.sin(theta))

    def spline(field):
        return RectBivariateSpline(
            z, r, np.asarray(field, dtype=float),
            kx=min(3, len(z) - 1), ky=min(3, len(r) - 1), s=0,
        )

    def scalar(value):
        return float(np.asarray(value, dtype=float).reshape(-1)[0])

    transverse = spline(position[:, :, 3])
    transverse_rate = spline(velocity[:, :, 3])
    q = scalar(transverse.ev(zcoord, radius))
    q_material_rate = scalar(
        transverse_rate.ev(zcoord, radius)
        + z_rate * transverse.ev(zcoord, radius, dx=1, dy=0)
        + radius_rate * transverse.ev(zcoord, radius, dx=0, dy=1)
    )
    h_sphere = float(radius_rate / radius + 0.5 * q_material_rate / q)

    phi_spline = spline(position[:, :, 7])
    phi_rate_spline = spline(velocity[:, :, 7])
    phi = scalar(phi_spline.ev(zcoord, radius))
    x_phi = scalar(
        phi_rate_spline.ev(zcoord, radius)
        + z_rate * phi_spline.ev(zcoord, radius, dx=1, dy=0)
        + radius_rate * phi_spline.ev(zcoord, radius, dx=0, dy=1)
    )
    wall_c_rate = float(
        float(wall_stiffness) / 6.0
        * (phi - float(wall_target)) * x_phi
    )
    sphere_radius = float(radius * math.sqrt(q))
    sphere_area = float(4.0 * math.pi * sphere_radius**2)
    return {
        "coordinate_z": zcoord,
        "coordinate_radius": radius,
        "coordinate_z_rate": z_rate,
        "coordinate_radius_rate": radius_rate,
        "sphere_radius": sphere_radius,
        "sphere_area": sphere_area,
        "H_Omega": h_sphere,
        "wall_scalar": phi,
        "X_wall_scalar": x_phi,
        "wall_israel_coefficient_rate": wall_c_rate,
        "finite": bool(np.all(np.isfinite([
            zcoord, radius, z_rate, radius_rate, sphere_radius, sphere_area,
            h_sphere, phi, x_phi, wall_c_rate,
        ]))),
    }


def physical_coupled_record(
    balance_record, endpoint_transport, geometric_israel_rate,
):
    """Replace Test-14B's separate deltas by the note-99 coupled seam law."""
    record = balance_record
    seam = record["seam"]
    charge = record["charge"]
    area_fractional_rate = (
        float(record["area_rate_transport"])
        / float(charge["reflection_doubled_area"])
    )
    coefficient = float(seam["geometric_Ws_over_W"])
    sphere_area = float(seam["sphere_area_A2"])
    h_sphere = float(endpoint_transport["H_Omega"])
    theta_x = -float(seam["B"]) * float(seam["theta_n"])
    h_meridional = theta_x - 2.0 * h_sphere
    h_sphere_s = (
        float(geometric_israel_rate) + coefficient * h_meridional
    )
    uncombined = coupled_from_uncombined(
        sphere_area, coefficient, geometric_israel_rate, h_meridional,
        h_sphere, h_sphere_s,
    )
    coupled = coupled_seam_rate(
        charge["equivalent_area_radius"], area_fractional_rate,
        sphere_area, coefficient, h_sphere,
    )
    separate_names = (
        "curvature_seam", "brane_matter_seam",
        "normal_connection_seam",
    )
    separate = float(sum(record["rates"][name] for name in separate_names))
    corrected_rates = {
        name: float(value) for name, value in record["rates"].items()
        if name not in separate_names
    }
    corrected_rates.update({
        "coupled_seam_global_radius": float(coupled["global_radius_part"]),
        "coupled_seam_joint_work": float(coupled["joint_work"]),
    })
    corrected_total = float(sum(corrected_rates.values()))
    target = float(record["charge_rate_target"]["finite_difference_rate"])
    return {
        "grid": record["grid"],
        "branch": record["branch"],
        "stride": int(record["stride"]),
        "time": float(record["time"]),
        "charge": float(record["charge_rate_target"]["charge"]),
        "charge_rate_target": target,
        "area_fractional_rate": area_fractional_rate,
        "theta_X_at_seam": theta_x,
        "H_meridional_normal": float(h_meridional),
        "H_Omega": h_sphere,
        "H_Omega_s_normal": float(h_sphere_s),
        "geometric_israel_rate": float(geometric_israel_rate),
        "wall_israel_rate": float(
            endpoint_transport["wall_israel_coefficient_rate"]
        ),
        "israel_rate_relative_scale_error": relative_scale_error(
            geometric_israel_rate,
            endpoint_transport["wall_israel_coefficient_rate"], 0.005,
        ),
        "endpoint_transport": endpoint_transport,
        "uncombined": uncombined,
        "coupled_seam": coupled,
        "separate_delta_seam_rate": separate,
        "joint_correction": float(coupled["total"] - separate),
        "corrected_rates": corrected_rates,
        "test14b_total_rate": float(record["total_balance_rate"]),
        "corrected_total_rate": corrected_total,
        "corrected_pointwise_residual": float(target - corrected_total),
        "finite": bool(
            endpoint_transport["finite"]
            and np.all(np.isfinite([
                target, corrected_total, h_sphere, h_meridional,
                h_sphere_s, geometric_israel_rate,
            ]))
        ),
    }


def evaluate_geometric_bulk_leaf(
    position, velocity, z, r, profile, rho_rate, balance_record,
    *, prepared=None,
):
    """Independently vary the smooth-cap intrinsic curvature.

    The returned ``geometric_bulk_rate`` excludes the cap boundary term.  It
    is therefore the proper comparator to the doubled smooth Cao terms when
    the coupled seam law already contains the cap boundary variation.
    """
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    theta = np.asarray(profile["theta"], dtype=float)
    rho = np.asarray(profile["rho"], dtype=float)
    slope = np.asarray(profile["slope"], dtype=float)
    rho_rate = np.asarray(rho_rate, dtype=float)
    if not (theta.shape == rho.shape == slope.shape == rho_rate.shape):
        raise ValueError("profile and rate arrays must have the same shape")
    prepared = (
        prepare_capped_expansion_slice(position, velocity, z, r)
        if prepared is None else prepared
    )

    sine = np.sin(theta)
    cosine = np.cos(theta)
    zcoord = z[-1] - rho * cosine
    radius = rho * sine
    tangent_coordinate = np.stack((
        rho * sine - slope * cosine,
        rho * cosine + slope * sine,
    ), axis=1)
    embedding_velocity = np.stack((
        -rho_rate * cosine, rho_rate * sine,
    ), axis=1)

    metric = _sample(prepared, "base_metric", zcoord, radius)
    inverse = _sample(prepared, "base_inverse", zcoord, radius)
    connection = _sample(prepared, "base_connection", zcoord, radius)
    extrinsic = _sample(prepared, "extrinsic_base", zcoord, radius)
    extrinsic_sphere = _sample(
        prepared, "extrinsic_sphere_eigenvalue", zcoord, radius,
    )
    lapse = _sample(prepared, "lapse", zcoord, radius)
    shift = _sample(prepared, "shift", zcoord, radius)
    speed = np.sqrt(np.einsum(
        "...a,...ab,...b->...", tangent_coordinate, metric,
        tangent_coordinate,
    ))
    tangent = tangent_coordinate / speed[:, None]
    tangent_covector = np.einsum("...ab,...b->...a", metric, tangent)
    normal_covector = np.stack((
        -tangent_coordinate[:, 1], tangent_coordinate[:, 0],
    ), axis=1)
    normal_covector /= np.sqrt(np.einsum(
        "...a,...ab,...b->...", normal_covector, inverse,
        normal_covector,
    ))[:, None]
    normal = np.einsum("...ab,...b->...a", inverse, normal_covector)

    transverse, transverse_gradient = _field_and_gradient_on_surface(
        position[:, :, 3], z, r, zcoord, radius,
    )
    root_transverse = np.sqrt(transverse)
    sphere_radius = radius * root_transverse
    log_w_gradient = 0.5 * transverse_gradient / transverse[:, None]
    log_w_gradient[:, 1] += 1.0 / radius
    sphere_radius_normal = sphere_radius * np.einsum(
        "...a,...a->...", normal, log_w_gradient,
    )
    sphere_radius_meridional = sphere_radius * np.einsum(
        "...a,...a->...", tangent, log_w_gradient,
    )
    sphere_curvature = sphere_radius_normal / sphere_radius

    normal_covector_theta = np.gradient(
        normal_covector, theta, axis=0, edge_order=2,
    )
    curve_curvature = (
        np.einsum("...a,...a->...", tangent, normal_covector_theta) / speed
        - np.einsum(
            "...a,...b,...cab,...c->...", tangent, tangent, connection,
            normal_covector,
        )
    )
    extrinsic_meridional = np.einsum(
        "...a,...ab,...b->...", tangent, extrinsic, tangent,
    )
    spatial_transport = shift + embedding_velocity
    normal_speed = np.einsum(
        "...a,...a->...", normal_covector, spatial_transport,
    )
    tangential_speed = np.einsum(
        "...a,...a->...", tangent_covector, spatial_transport,
    )

    h_meridional = -lapse * extrinsic_meridional + normal_speed * curve_curvature
    h_sphere = -lapse * extrinsic_sphere + normal_speed * sphere_curvature
    h_sphere_s, _ = _proper_scalar_derivatives(
        h_sphere, theta, speed,
    )
    tangential_speed_s, _ = _proper_scalar_derivatives(
        tangential_speed, theta, speed,
    )

    _, warp_s, warp_ss = _proper_curve(
        theta, speed, sphere_radius, sphere_radius_meridional[-1],
    )
    intrinsic_einstein_meridional = -(
        1.0 - warp_s**2
    ) / sphere_radius**2
    intrinsic_einstein_sphere = warp_ss / sphere_radius
    intrinsic_ricci_meridional = -2.0 * warp_ss / sphere_radius
    intrinsic_ricci_sphere = (
        1.0 - warp_s**2 - sphere_radius * warp_ss
    ) / sphere_radius**2
    intrinsic_scalar = (
        intrinsic_ricci_meridional + 2.0 * intrinsic_ricci_sphere
    )
    intrinsic_einstein_meridional = _axis_regularize_even(
        intrinsic_einstein_meridional, theta, radius, r,
    )
    intrinsic_einstein_sphere = _axis_regularize_even(
        intrinsic_einstein_sphere, theta, radius, r,
    )
    intrinsic_ricci_meridional = _axis_regularize_even(
        intrinsic_ricci_meridional, theta, radius, r,
    )
    intrinsic_ricci_sphere = _axis_regularize_even(
        intrinsic_ricci_sphere, theta, radius, r,
    )
    intrinsic_scalar = _axis_regularize_even(
        intrinsic_scalar, theta, radius, r,
    )
    intrinsic_bulk_variation = 2.0 * _one_cap_integral(
        -2.0 * (
            intrinsic_einstein_meridional * h_meridional
            + 2.0 * intrinsic_einstein_sphere * h_sphere
        ),
        sphere_radius, speed, theta,
    )
    theta_x = h_meridional + 2.0 * h_sphere
    shear_meridional = h_meridional - theta_x / 3.0
    shear_sphere = h_sphere - theta_x / 3.0
    ricci_tf_meridional = (
        intrinsic_ricci_meridional - intrinsic_scalar / 3.0
    )
    ricci_tf_sphere = intrinsic_ricci_sphere - intrinsic_scalar / 3.0
    ricci_tf_shear_integral = 2.0 * _one_cap_integral(
        ricci_tf_meridional * shear_meridional
        + 2.0 * ricci_tf_sphere * shear_sphere,
        sphere_radius, speed, theta,
    )
    coefficient = float(warp_s[-1] / sphere_radius[-1])
    sphere_area = float(4.0 * math.pi * sphere_radius[-1] ** 2)
    boundary_variation = cap_boundary_variation(
        sphere_area, coefficient, h_meridional[-1], h_sphere[-1],
        h_sphere_s[-1],
    )

    charge = balance_record["charge"]
    area_radius = float(charge["equivalent_area_radius"])
    area_fractional_rate = (
        float(balance_record["area_rate_transport"])
        / float(charge["reflection_doubled_area"])
    )
    smooth_integral = float(
        charge["doubled_smooth_bulk_curvature_integral"]
    )
    global_radius_part = (
        area_radius * smooth_integral * area_fractional_rate / 12.0
    )
    intrinsic_bulk_rate = area_radius * intrinsic_bulk_variation / 4.0
    intrinsic_anisotropy_rate = (
        -area_radius * ricci_tf_shear_integral / 2.0
    )
    intrinsic_boundary_rate = area_radius * boundary_variation / 4.0
    geometric_bulk_rate = global_radius_part + intrinsic_bulk_rate
    geometric_bulk_plus_boundary_rate = (
        geometric_bulk_rate + intrinsic_boundary_rate
    )
    smooth_names = (
        "curvature_smooth", "matter_scalar_ll_smooth",
        "matter_scalar_ln_smooth", "shear_smooth",
        "normal_connection_smooth", "vacuum",
    )
    cao_smooth_rate = float(sum(
        balance_record["rates"][name] for name in smooth_names
    ))
    theta_x_direct = h_meridional + 2.0 * h_sphere
    theta_x_null = (
        -float(balance_record["seam"]["B"])
        * float(balance_record["seam"]["theta_n"])
    )

    # The material endpoint has zero tangential speed, but its derivative can
    # be nonzero.  This exposes the reparameterization contribution which is
    # individually large in the meridional deformation at the seam.
    material_h_meridional = h_meridional + tangential_speed_s
    material_h_sphere = h_sphere + tangential_speed * warp_s / sphere_radius
    material_h_sphere_s = (
        h_sphere_s
        + tangential_speed_s * warp_s / sphere_radius
        + tangential_speed * np.gradient(
            warp_s / sphere_radius, theta, edge_order=2,
        ) / speed
    )
    material_boundary_variation = cap_boundary_variation(
        sphere_area, coefficient, material_h_meridional[-1],
        material_h_sphere[-1], material_h_sphere_s[-1],
    )
    return {
        "intrinsic_bulk_variation": float(intrinsic_bulk_variation),
        "cap_boundary_variation": float(boundary_variation),
        "material_cap_boundary_variation": float(
            material_boundary_variation
        ),
        "global_radius_part": float(global_radius_part),
        "intrinsic_bulk_rate": float(intrinsic_bulk_rate),
        "ricci_tf_shear_integral": float(ricci_tf_shear_integral),
        "intrinsic_anisotropy_rate": float(intrinsic_anisotropy_rate),
        "intrinsic_boundary_rate": float(intrinsic_boundary_rate),
        "geometric_bulk_rate": float(geometric_bulk_rate),
        "geometric_bulk_plus_boundary_rate": float(
            geometric_bulk_plus_boundary_rate
        ),
        "cao_smooth_rate": cao_smooth_rate,
        "bulk_focusing_defect": float(geometric_bulk_rate - cao_smooth_rate),
        "theta_X_seam_direct": float(theta_x_direct[-1]),
        "theta_X_seam_null": theta_x_null,
        "theta_X_seam_error": relative_scale_error(
            theta_x_direct[-1], theta_x_null,
            max(abs(theta_x_null), 1.0),
        ),
        "H_meridional_seam": float(h_meridional[-1]),
        "H_Omega_seam": float(h_sphere[-1]),
        "H_Omega_s_seam": float(h_sphere_s[-1]),
        "tangential_speed_seam": float(tangential_speed[-1]),
        "tangential_speed_s_seam": float(tangential_speed_s[-1]),
        "maximum_absolute_tangential_speed": float(np.max(np.abs(
            tangential_speed
        ))),
        "finite": bool(np.all(np.isfinite([
            intrinsic_bulk_variation, boundary_variation,
            material_boundary_variation, global_radius_part,
            intrinsic_bulk_rate, intrinsic_boundary_rate,
            intrinsic_anisotropy_rate, geometric_bulk_rate, cao_smooth_rate,
        ]))),
    }


def leaf_marginal_transport_fields(
    position, velocity, z, r, profile, rho_rate, *, prepared=None,
):
    """Return the fields needed to audit ``L_X theta_l`` on one leaf."""
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    theta = np.asarray(profile["theta"], dtype=float)
    rho = np.asarray(profile["rho"], dtype=float)
    slope = np.asarray(profile["slope"], dtype=float)
    rho_rate = np.asarray(rho_rate, dtype=float)
    prepared = (
        prepare_capped_expansion_slice(position, velocity, z, r)
        if prepared is None else prepared
    )
    sine = np.sin(theta)
    cosine = np.cos(theta)
    zcoord = z[-1] - rho * cosine
    radius = rho * sine
    tangent_coordinate = np.stack((
        rho * sine - slope * cosine,
        rho * cosine + slope * sine,
    ), axis=1)
    embedding_velocity = np.stack((
        -rho_rate * cosine, rho_rate * sine,
    ), axis=1)
    metric = _sample(prepared, "base_metric", zcoord, radius)
    inverse = _sample(prepared, "base_inverse", zcoord, radius)
    connection = _sample(prepared, "base_connection", zcoord, radius)
    extrinsic = _sample(prepared, "extrinsic_base", zcoord, radius)
    extrinsic_sphere = _sample(
        prepared, "extrinsic_sphere_eigenvalue", zcoord, radius,
    )
    shift = _sample(prepared, "shift", zcoord, radius)
    speed = np.sqrt(np.einsum(
        "...a,...ab,...b->...", tangent_coordinate, metric,
        tangent_coordinate,
    ))
    tangent = tangent_coordinate / speed[:, None]
    tangent_covector = np.einsum("...ab,...b->...a", metric, tangent)
    normal_covector = np.stack((
        -tangent_coordinate[:, 1], tangent_coordinate[:, 0],
    ), axis=1)
    normal_covector /= np.sqrt(np.einsum(
        "...a,...ab,...b->...", normal_covector, inverse,
        normal_covector,
    ))[:, None]
    normal = np.einsum("...ab,...b->...a", inverse, normal_covector)
    transverse, transverse_gradient = _field_and_gradient_on_surface(
        position[:, :, 3], z, r, zcoord, radius,
    )
    sphere_radius = radius * np.sqrt(transverse)
    log_w_gradient = 0.5 * transverse_gradient / transverse[:, None]
    log_w_gradient[:, 1] += 1.0 / radius
    sphere_curvature = np.einsum(
        "...a,...a->...", normal, log_w_gradient,
    )
    normal_covector_theta = np.gradient(
        normal_covector, theta, axis=0, edge_order=2,
    )
    curve_curvature = (
        np.einsum("...a,...a->...", tangent, normal_covector_theta) / speed
        - np.einsum(
            "...a,...b,...cab,...c->...", tangent, tangent, connection,
            normal_covector,
        )
    )
    extrinsic_meridional = np.einsum(
        "...a,...ab,...b->...", tangent, extrinsic, tangent,
    )
    theta_l = (
        -extrinsic_meridional + curve_curvature
        + 2.0 * (-extrinsic_sphere + sphere_curvature)
    ) / math.sqrt(2.0)
    theta_n = (
        -curve_curvature - 2.0 * sphere_curvature
        - extrinsic_meridional - 2.0 * extrinsic_sphere
    ) / math.sqrt(2.0)
    theta_l = _axis_regularize_even(theta_l, theta, radius, r)
    theta_n = _axis_regularize_even(theta_n, theta, radius, r)
    theta_l_s, _ = _proper_scalar_derivatives(
        theta_l, theta, speed,
    )
    tangential_speed = np.einsum(
        "...a,...a->...", tangent_covector, shift + embedding_velocity,
    )
    return {
        "theta": theta,
        "theta_l": theta_l,
        "theta_n": theta_n,
        "theta_l_s": theta_l_s,
        "tangential_speed": tangential_speed,
        "sphere_radius": sphere_radius,
        "speed": speed,
        "finite": bool(np.all(np.isfinite([
            theta_l, theta_n, theta_l_s, tangential_speed,
            sphere_radius, speed,
        ]))),
    }


def marginal_tangency_rate(
    material_theta_l_rate, transport_fields, area_radius,
):
    """Convert material transport to ``L_X theta_l`` and its energy term."""
    material_theta_l_rate = np.asarray(material_theta_l_rate, dtype=float)
    normal_rate = (
        material_theta_l_rate
        - transport_fields["tangential_speed"]
        * transport_fields["theta_l_s"]
    )
    integral = 2.0 * _one_cap_integral(
        transport_fields["theta_n"] * normal_rate,
        transport_fields["sphere_radius"], transport_fields["speed"],
        transport_fields["theta"],
    )
    energy_term = float(float(area_radius) * integral / 3.0)
    return {
        "maximum_absolute_material_theta_l_rate": float(np.max(np.abs(
            material_theta_l_rate
        ))),
        "maximum_absolute_normal_theta_l_rate": float(np.max(np.abs(
            normal_rate
        ))),
        "theta_n_LX_theta_l_integral": float(integral),
        "hawking_product_derivative_term": energy_term,
        "finite": bool(np.all(np.isfinite(normal_rate))),
    }


def apply_intrinsic_anisotropy(physical_record, geometric_bulk_record):
    """Add note-100's independently evaluated anisotropy term exactly once."""
    corrected_rates = dict(physical_record["corrected_rates"])
    anisotropy = float(geometric_bulk_record["intrinsic_anisotropy_rate"])
    corrected_rates["intrinsic_anisotropy"] = anisotropy
    total = float(sum(corrected_rates.values()))
    target = float(physical_record["charge_rate_target"])
    defect = float(geometric_bulk_record["bulk_focusing_defect"])
    match_error = relative_scale_error(
        anisotropy, defect, max(abs(anisotropy), abs(defect), 1.0),
    )
    return {
        **physical_record,
        "corrected_rates": corrected_rates,
        "pre_anisotropy_total_rate": float(
            physical_record["corrected_total_rate"]
        ),
        "corrected_total_rate": total,
        "corrected_pointwise_residual": float(target - total),
        "intrinsic_anisotropy_rate": anisotropy,
        "geometric_bulk_rate": float(
            geometric_bulk_record["geometric_bulk_rate"]
        ),
        "cao_smooth_rate": float(geometric_bulk_record["cao_smooth_rate"]),
        "bulk_focusing_defect": defect,
        "anisotropy_to_bulk_defect_error": match_error,
        "geometric_bulk_diagnostics": geometric_bulk_record,
        "finite": bool(
            physical_record["finite"]
            and geometric_bulk_record["finite"]
            and np.isfinite(total)
        ),
    }
def _regularized_abs(x, epsilon, family):
    """Return a smooth absolute value and its first two derivatives."""
    x = np.asarray(x, dtype=float)
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if family == "softabs":
        root = np.sqrt(x * x + epsilon * epsilon)
        return (
            root - epsilon,
            x / root,
            epsilon * epsilon / root**3,
        )
    if family == "tanh":
        scaled = x / epsilon
        # log(cosh(y)) = logaddexp(y,-y)-log(2), without overflow.
        value = epsilon * (
            np.logaddexp(scaled, -scaled) - math.log(2.0)
        )
        derivative = np.tanh(scaled)
        second = (1.0 - derivative * derivative) / epsilon
        return value, derivative, second
    if family == "polynomial":
        absolute = np.abs(x)
        y = absolute / epsilon
        inside = y < 1.0
        value = absolute.copy()
        derivative = np.sign(x)
        second = np.zeros_like(x)
        yi = y[inside]
        polynomial = 3.0 * yi**2 - 3.0 * yi**3 + yi**4
        polynomial_prime = 6.0 * yi - 9.0 * yi**2 + 4.0 * yi**3
        polynomial_second = 6.0 - 18.0 * yi + 12.0 * yi**2
        value[inside] = epsilon * polynomial
        derivative[inside] = np.sign(x[inside]) * polynomial_prime
        second[inside] = polynomial_second / epsilon
        return value, derivative, second
    raise ValueError(f"unknown regularizer family {family!r}")


def _warped_scalar_integral(x, scale_factor, warp, warp_x, warp_xx):
    """Scalar-curvature integral for ``a^2 dx^2+a^2 warp^2 dOmega_2``."""
    density = 4.0 * math.pi * float(scale_factor) * (
        -4.0 * warp * warp_xx + 2.0 * (1.0 - warp_x**2)
    )
    return float(simpson(density, x=x))


def smoothing_layer_record(
    family, epsilon, nodes_per_epsilon, *, half_width=0.2,
    scale_factor=1.1, seam_warp=1.2, signed_slope=-0.8,
    fractional_scale_rate=0.3,
):
    """Resolve one coupled smoothing of a reflected manufactured seam.

    The scale factor multiplies both meridional and spherical metric factors.
    Hence ``H_m=H_Omega`` and the layer's exact rate is its scalar-curvature
    excess times the fractional scale rate.
    """
    epsilon = float(epsilon)
    nodes_per_epsilon = int(nodes_per_epsilon)
    intervals = int(math.ceil(
        2.0 * float(half_width) / epsilon * nodes_per_epsilon
    ))
    if intervals % 2:
        intervals += 1
    x = np.linspace(-float(half_width), float(half_width), intervals + 1)
    regularized, first, second = _regularized_abs(x, epsilon, family)
    warp = float(seam_warp) * (
        1.0 - float(signed_slope) * regularized
    )
    warp_x = -float(signed_slope) * float(seam_warp) * first
    warp_xx = -float(signed_slope) * float(seam_warp) * second
    regularized_integral = _warped_scalar_integral(
        x, scale_factor, warp, warp_x, warp_xx,
    )

    # Integrate the two sharp *smooth* sides separately.  Sampling a signed
    # derivative as zero at x=0 would introduce an artificial one-node defect
    # whose quadrature weight converges only linearly.
    positive = x[x >= 0.0]
    sharp_warp_positive = float(seam_warp) * (
        1.0 - float(signed_slope) * positive
    )
    sharp_warp_x_positive = np.full_like(
        positive, -float(signed_slope) * float(seam_warp),
    )
    sharp_smooth_integral = 2.0 * _warped_scalar_integral(
        positive, scale_factor, sharp_warp_positive,
        sharp_warp_x_positive, np.zeros_like(positive),
    )
    layer_excess = regularized_integral - sharp_smooth_integral
    expected_seam = (
        32.0 * math.pi * float(scale_factor)
        * float(signed_slope) * float(seam_warp) ** 2
    )
    layer_rate = float(fractional_scale_rate) * layer_excess
    expected_rate = float(fractional_scale_rate) * expected_seam
    return {
        "family": family,
        "epsilon": epsilon,
        "nodes_per_epsilon": nodes_per_epsilon,
        "node_count": len(x),
        "regularized_integral": regularized_integral,
        "sharp_smooth_integral": sharp_smooth_integral,
        "layer_excess": float(layer_excess),
        "expected_seam_integral": float(expected_seam),
        "layer_rate": layer_rate,
        "expected_coupled_rate": float(expected_rate),
        "integral_error": relative_scale_error(
            layer_excess, expected_seam, expected_seam,
        ),
        "rate_error": relative_scale_error(
            layer_rate, expected_rate, expected_rate,
        ),
        "finite": bool(np.all(np.isfinite([
            regularized_integral, sharp_smooth_integral, layer_excess,
            layer_rate,
        ]))),
    }


def smoothing_controls():
    """Run the three prospectively sealed coupled regularization families."""
    families = ("softabs", "tanh", "polynomial")
    epsilons = (0.04, 0.02, 0.01, 0.005)
    resolutions = (32, 64, 128)
    records = [
        smoothing_layer_record(family, epsilon, resolution)
        for family in families
        for epsilon in epsilons
        for resolution in resolutions
    ]
    finest = {
        family: next(
            item for item in records
            if item["family"] == family
            and item["epsilon"] == min(epsilons)
            and item["nodes_per_epsilon"] == max(resolutions)
        )
        for family in families
    }
    limiting_values = np.asarray([
        finest[family]["layer_rate"] for family in families
    ])
    expected = float(next(iter(finest.values()))["expected_coupled_rate"])
    family_spread = float(
        np.max(limiting_values) - np.min(limiting_values)
    )
    family_spread_error = float(
        family_spread / max(abs(expected), 1e-300)
    )
    spatial_convergence = []
    for family in families:
        for epsilon in epsilons:
            selected = [
                item for item in records
                if item["family"] == family and item["epsilon"] == epsilon
            ]
            selected.sort(key=lambda item: item["nodes_per_epsilon"])
            values = np.asarray([item["layer_rate"] for item in selected])
            scale = max(abs(values[-1]), 1.0)
            coarse_change = abs(values[1] - values[0]) / scale
            fine_change = abs(values[2] - values[1]) / scale
            reduction = float(
                coarse_change / max(fine_change, 1e-300)
            )
            # Node doubling must give at least second-order reduction.  A
            # fully resolved roundoff plateau is accepted separately.
            monotone = bool(
                reduction >= 3.5 or coarse_change < 2e-10
            )
            spatial_convergence.append({
                "family": family,
                "epsilon": epsilon,
                "relative_coarse_change": float(coarse_change),
                "relative_fine_change": float(fine_change),
                "node_doubling_reduction": reduction,
                "second_order_or_roundoff_plateau": monotone,
            })
    finest_error = max(item["rate_error"] for item in finest.values())
    passed = bool(
        all(item["finite"] for item in records)
        and finest_error < 0.01
        and family_spread_error < 0.01
        and all(
            item["second_order_or_roundoff_plateau"]
            for item in spatial_convergence
        )
    )
    return {
        "records": records,
        "finest": finest,
        "maximum_finest_rate_error": float(finest_error),
        "family_spread_relative_scale_error": family_spread_error,
        "spatial_convergence": spatial_convergence,
        "passed": passed,
    }


def _manufactured_warped_variation(parameters, nodes=4001, step=1e-6):
    """Check the scalar-curvature first variation on a smooth open cap."""
    a, b, p, q, rcoef = (float(value) for value in parameters)
    s = np.linspace(0.2, 1.0, int(nodes))

    def warp_data(time):
        base = 1.0 + a * s + b * s**2
        base_s = a + 2.0 * b * s
        base_ss = np.full_like(s, 2.0 * b)
        field = p + q * s + rcoef * s**2
        field_s = q + 2.0 * rcoef * s
        field_ss = np.full_like(s, 2.0 * rcoef)
        exponential = np.exp(float(time) * field)
        warp = exponential * base
        warp_s = exponential * (
            base_s + float(time) * field_s * base
        )
        warp_ss = exponential * (
            base_ss
            + 2.0 * float(time) * field_s * base_s
            + float(time) * field_ss * base
            + float(time) ** 2 * field_s**2 * base
        )
        return warp, warp_s, warp_ss, field, field_s

    def scalar_integral(time):
        warp, warp_s, warp_ss, _, _ = warp_data(time)
        density = 4.0 * math.pi * (
            -4.0 * warp * warp_ss + 2.0 * (1.0 - warp_s**2)
        )
        return float(simpson(density, x=s))

    warp, warp_s, warp_ss, h_sphere, h_sphere_s = warp_data(0.0)
    h_meridional = np.zeros_like(s)
    theta_x = 2.0 * h_sphere
    ricci_m = -2.0 * warp_ss / warp
    ricci_o = (1.0 - warp_s**2 - warp * warp_ss) / warp**2
    scalar = ricci_m + 2.0 * ricci_o
    einstein_m = ricci_m - 0.5 * scalar
    einstein_o = ricci_o - 0.5 * scalar
    volume_density = 4.0 * math.pi * warp**2 * (
        -2.0 * (einstein_m * h_meridional + 2.0 * einstein_o * h_sphere)
    )
    volume = float(simpson(volume_density, x=s))
    radial_boundary_density = 4.0 * (
        -h_sphere_s + warp_s / warp * (h_meridional - h_sphere)
    )
    areas = 4.0 * math.pi * warp**2
    boundary = float(
        areas[-1] * radial_boundary_density[-1]
        - areas[0] * radial_boundary_density[0]
    )
    finite_difference = float(
        (scalar_integral(step) - scalar_integral(-step)) / (2.0 * step)
    )

    shear_m = h_meridional - theta_x / 3.0
    shear_o = h_sphere - theta_x / 3.0
    ricci_tf_m = ricci_m - scalar / 3.0
    ricci_tf_o = ricci_o - scalar / 3.0
    trace_density = 4.0 * math.pi * warp**2 * scalar * theta_x / 3.0
    anisotropy_density = 4.0 * math.pi * warp**2 * (-2.0) * (
        ricci_tf_m * shear_m + 2.0 * ricci_tf_o * shear_o
    )
    trace_part = float(simpson(trace_density, x=s))
    anisotropy_part = float(simpson(anisotropy_density, x=s))
    direct_error = relative_scale_error(
        volume + boundary, finite_difference, finite_difference,
    )
    decomposition_error = relative_scale_error(
        trace_part + anisotropy_part, volume, volume,
    )
    omission_difference = relative_scale_error(
        trace_part + boundary, finite_difference, finite_difference,
    )
    double_difference = relative_scale_error(
        trace_part + 2.0 * anisotropy_part + boundary,
        finite_difference, finite_difference,
    )
    return {
        "parameters": list(parameters),
        "nodes": int(nodes),
        "finite_difference": finite_difference,
        "einstein_volume": volume,
        "boundary": boundary,
        "trace_part": trace_part,
        "anisotropy_part": anisotropy_part,
        "direct_first_variation_error": direct_error,
        "trace_anisotropy_decomposition_error": decomposition_error,
        "omission_difference": omission_difference,
        "double_counting_difference": double_difference,
        "finite": bool(np.all(np.isfinite([
            finite_difference, volume, boundary, trace_part,
            anisotropy_part,
        ]))),
    }


def anisotropy_controls():
    """Run the sealed note-100 intrinsic-anisotropy controls."""
    # Algebraic eigenvalue controls use arbitrary volume weights.
    ricci_m = np.asarray([2.0, 3.0, 4.0])
    ricci_o = np.asarray([2.0, 3.0, 4.0])
    h_m = np.asarray([-0.2, 0.4, 0.7])
    h_o = np.asarray([0.6, -0.3, 0.1])
    scalar = ricci_m + 2.0 * ricci_o
    theta = h_m + 2.0 * h_o
    einstein_tf_contraction = (
        (ricci_m - scalar / 3.0) * (h_m - theta / 3.0)
        + 2.0 * (ricci_o - scalar / 3.0) * (h_o - theta / 3.0)
    )
    einstein_leaf_zero = float(np.max(np.abs(einstein_tf_contraction)))

    ricci_m = np.asarray([1.0, 3.0, -0.5])
    ricci_o = np.asarray([2.0, -0.2, 4.0])
    h_m = np.asarray([0.4, -0.7, 1.2])
    h_o = h_m.copy()
    scalar = ricci_m + 2.0 * ricci_o
    theta = h_m + 2.0 * h_o
    isotropic_contraction = (
        (ricci_m - scalar / 3.0) * (h_m - theta / 3.0)
        + 2.0 * (ricci_o - scalar / 3.0) * (h_o - theta / 3.0)
    )
    isotropic_zero = float(np.max(np.abs(isotropic_contraction)))

    base_h_m = np.asarray([-0.3, 0.2, 0.9])
    base_h_o = np.asarray([0.7, -0.1, -0.4])
    trace_shift = 1.234

    def contraction(local_h_m, local_h_o):
        local_theta = local_h_m + 2.0 * local_h_o
        return (
            (ricci_m - scalar / 3.0)
            * (local_h_m - local_theta / 3.0)
            + 2.0 * (ricci_o - scalar / 3.0)
            * (local_h_o - local_theta / 3.0)
        )

    trace_invariance_error = float(np.max(np.abs(
        contraction(base_h_m + trace_shift, base_h_o + trace_shift)
        - contraction(base_h_m, base_h_o)
    )))
    manufactured = [
        _manufactured_warped_variation(values)
        for values in (
            (0.2, 0.08, 0.1, 0.4, -0.05),
            (-0.1, 0.12, -0.2, 0.25, 0.08),
            (0.35, -0.06, 0.3, -0.35, 0.11),
        )
    ]
    passed = bool(
        einstein_leaf_zero < 1e-12
        and isotropic_zero < 1e-12
        and trace_invariance_error < 1e-12
        and all(item["finite"] for item in manufactured)
        and max(
            item["direct_first_variation_error"] for item in manufactured
        ) < 2e-4
        and max(
            item["trace_anisotropy_decomposition_error"]
            for item in manufactured
        ) < 2e-4
        and min(
            item["omission_difference"] for item in manufactured
        ) > 0.01
        and min(
            item["double_counting_difference"] for item in manufactured
        ) > 0.01
    )
    return {
        "einstein_leaf_zero": einstein_leaf_zero,
        "isotropic_deformation_zero": isotropic_zero,
        "pure_trace_invariance_error": trace_invariance_error,
        "manufactured": manufactured,
        "passed": passed,
    }
def analytic_controls():
    """Run note-99 analytic, sign, and coupled-smoothing controls."""
    static = coupled_seam_rate(1.7, 0.0, 12.0, -0.8, 0.0)
    flat = coupled_seam_rate(1.7, 0.0, 12.0, 0.0, 0.4)

    sphere_area = 12.0
    coefficient = -0.8
    h_sphere = 0.3
    h_meridional = 0.3
    h_sphere_s = 0.0
    coefficient_rate = compatible_israel_rate(
        h_sphere_s, coefficient, h_meridional,
    )
    homothetic = coupled_from_uncombined(
        sphere_area, coefficient, coefficient_rate, h_meridional,
        h_sphere, h_sphere_s,
    )
    orientation_original = combined_intrinsic_joint_variation(
        sphere_area, coefficient, h_sphere,
    )
    # Normal reversal changes both c and the signed sphere deformation.
    orientation_reversed = combined_intrinsic_joint_variation(
        sphere_area, -coefficient, -h_sphere,
    )
    orientation_wrong = combined_intrinsic_joint_variation(
        sphere_area, -coefficient, h_sphere,
    )
    orientation_error = relative_scale_error(
        orientation_reversed, orientation_original, orientation_original,
    )
    wrong_orientation_difference = relative_scale_error(
        orientation_wrong, orientation_original, orientation_original,
    )

    smoothing = smoothing_controls()
    anisotropy = anisotropy_controls()
    passed = bool(
        abs(static["total"]) < 1e-12
        and abs(flat["total"]) < 1e-12
        and homothetic["compatibility_error"] < 1e-10
        and orientation_error < 1e-10
        and wrong_orientation_difference > 0.01
        and smoothing["passed"]
        and anisotropy["passed"]
    )
    return {
        "static": static,
        "flat_join": flat,
        "homothetic": homothetic,
        "orientation": {
            "original": orientation_original,
            "reversed": orientation_reversed,
            "incorrect_single_sign_reversal": orientation_wrong,
            "correct_reversal_error": orientation_error,
            "incorrect_reversal_difference": wrong_orientation_difference,
        },
        "smoothing": smoothing,
        "anisotropy": anisotropy,
        "passed": passed,
    }
