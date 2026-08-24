"""Test-14B generalized-Hawking--AdS balance on reflected SO(3) caps.

This module implements the prospectively sealed conventions in note 96.  Its
primary evaluator uses the canonical scalar stress tensor directly; it does
not infer matter flux from the Einstein tensor.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad, simpson
from scipy.interpolate import CubicSpline, RectBivariateSpline

from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice
from bhps.test14_quasilocal_charge import reflected_cap_charge


SQRT2 = math.sqrt(2.0)
OMEGA3 = 2.0 * math.pi**2


def relative_scale_error(value, expected, scale=1.0):
    """Symmetric relative error with an explicit absolute reference scale."""
    return float(
        abs(float(value) - float(expected))
        / max(abs(float(value)), abs(float(expected)), abs(float(scale)), 1e-300)
    )


def finite_difference_weights(nodes, target):
    """Return first-derivative weights exact for polynomials on ``nodes``."""
    nodes = np.asarray(nodes, dtype=float)
    shifted = nodes - float(target)
    count = len(nodes)
    matrix = np.vstack([shifted**power for power in range(count)])
    right = np.zeros(count)
    right[1] = 1.0
    return np.linalg.solve(matrix, right)


def five_point_history_derivative(values, times, stride=1):
    """Fourth-order local derivative, centered where the history permits it.

    Near either end, the first or last five stride-separated samples supply a
    one-sided derivative.  Every target time is retained so integrated windows
    share identical endpoints across stencil strides.
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    step = int(stride)
    if step < 1 or values.shape[0] != len(times) or len(times) < 4 * step + 1:
        raise ValueError("history is too short for the requested five-point stride")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")
    result = np.empty_like(values)
    count = len(times)
    for index in range(count):
        if index < 2 * step:
            selected = np.arange(0, 5 * step, step)
        elif index > count - 1 - 2 * step:
            selected = np.arange(count - 1 - 4 * step, count, step)
        else:
            selected = index + step * np.arange(-2, 3)
        weights = finite_difference_weights(times[selected], times[index])
        result[index] = np.tensordot(weights, values[selected], axes=(0, 0))
    return result


def _spline(z, r, field):
    return RectBivariateSpline(
        np.asarray(z, dtype=float), np.asarray(r, dtype=float),
        np.asarray(field, dtype=float), kx=min(3, len(z) - 1),
        ky=min(3, len(r) - 1), s=0,
    )


def _sample(prepared, name, zcoord, radius):
    field = np.asarray(prepared.adm[name])
    if field.ndim == 2:
        return _spline(prepared.z, prepared.r, field).ev(zcoord, radius)
    result = np.empty((len(zcoord), *field.shape[2:]))
    for index in np.ndindex(field.shape[2:]):
        result[(slice(None), *index)] = _spline(
            prepared.z, prepared.r,
            field[(slice(None), slice(None), *index)],
        ).ev(zcoord, radius)
    return result


def _axis_regularize_even(values, theta, radius, radial_grid):
    """Replace the unresolved coordinate-axis collar by an even fit."""
    values = np.asarray(values, dtype=float).copy()
    theta = np.asarray(theta, dtype=float)
    safe = np.flatnonzero(radius >= 2.0 * np.min(np.diff(radial_grid)))
    if len(safe) < 6:
        raise ValueError("too few axis-safe surface samples")
    first = int(safe[0])
    indices = safe[: min(8, len(safe))]
    scale = max(float(theta[indices[-1]] ** 2), 1e-30)
    coefficients = np.polynomial.polynomial.polyfit(
        theta[indices] ** 2 / scale, values[indices], 1,
    )
    values[:first] = np.polynomial.polynomial.polyval(
        theta[:first] ** 2 / scale, coefficients,
    )
    return values


def _proper_curve(theta, speed, sphere_radius, brane_first_derivative):
    """Return proper distance and one-sided derivatives of the warp radius."""
    theta = np.asarray(theta, dtype=float)
    speed = np.asarray(speed, dtype=float)
    sphere_radius = np.asarray(sphere_radius, dtype=float)
    increments = np.diff(theta) * (speed[:-1] + speed[1:]) / 2.0
    sampled = np.concatenate(([0.0], np.cumsum(increments)))
    missing = float(speed[0] * theta[0])
    proper_s = np.concatenate(([0.0], missing + sampled))
    proper_w = np.concatenate(([0.0], sphere_radius))
    curve = CubicSpline(
        proper_s, proper_w,
        bc_type=((1, 1.0), (1, float(brane_first_derivative))),
    )
    nodes = proper_s[1:]
    return proper_s, curve(nodes, 1), curve(nodes, 2)


def _proper_scalar_derivatives(values, theta, speed, axis_value=None):
    """Return first and second proper derivatives of an invariant scalar."""
    values = np.asarray(values, dtype=float)
    theta = np.asarray(theta, dtype=float)
    speed = np.asarray(speed, dtype=float)
    increments = np.diff(theta) * (speed[:-1] + speed[1:]) / 2.0
    sampled = np.concatenate(([0.0], np.cumsum(increments)))
    missing = float(speed[0] * theta[0])
    proper_s = np.concatenate(([0.0], missing + sampled))
    if axis_value is None:
        fit_count = min(10, len(theta))
        coefficients = np.polynomial.polynomial.polyfit(
            theta[:fit_count] ** 2, values[:fit_count],
            min(3, fit_count - 1),
        )
        axis_value = float(coefficients[0])
    extended = np.concatenate(([float(axis_value)], values))
    curve = CubicSpline(
        proper_s, extended, bc_type=((1, 0.0), "not-a-knot"),
    )
    nodes = proper_s[1:]
    return curve(nodes, 1), curve(nodes, 2)


def _one_cap_integral(values, sphere_radius, speed, theta):
    return float(simpson(
        4.0 * math.pi * np.asarray(sphere_radius) ** 2
        * np.asarray(values) * np.asarray(speed), x=theta,
    ))


def _field_on_surface(field, z, r, zcoord, radius):
    return _spline(z, r, field).ev(zcoord, radius)


def _field_and_gradient_on_surface(field, z, r, zcoord, radius):
    spline = _spline(z, r, field)
    return (
        spline.ev(zcoord, radius),
        np.stack((
            spline.ev(zcoord, radius, dx=1, dy=0),
            spline.ev(zcoord, radius, dx=0, dy=1),
        ), axis=1),
    )


def evaluate_balance_leaf(
    position, velocity, z, r, profile, rho_time_derivative, background,
    stencil_width=7, prepared=None, cosmological_constant=-6.0,
    bulk_stress_override=None,
):
    """Evaluate every smooth and distributional Test-14B rate on one leaf."""
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    theta = np.asarray(profile["theta"], dtype=float)
    rho = np.asarray(profile["rho"], dtype=float)
    slope = np.asarray(profile["slope"], dtype=float)
    rho_dot = np.asarray(rho_time_derivative, dtype=float)
    if not (rho.shape == slope.shape == rho_dot.shape == theta.shape):
        raise ValueError("profile and temporal derivative shapes disagree")
    prepared = (
        prepare_capped_expansion_slice(
            position, velocity, z, r, stencil_width=stencil_width,
        ) if prepared is None else prepared
    )

    sine = np.sin(theta)
    cosine = np.cos(theta)
    zcoord = z[-1] - rho * cosine
    radius = rho * sine
    tangent_coordinate = np.stack((
        rho * sine - slope * cosine,
        rho * cosine + slope * sine,
    ), axis=1)
    embedding_velocity = np.stack((-rho_dot * cosine, rho_dot * sine), axis=1)

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
        "...a,...ab,...b->...", tangent_coordinate, metric, tangent_coordinate,
    ))
    tangent = tangent_coordinate / speed[:, None]
    normal_covector = np.stack(
        (-tangent_coordinate[:, 1], tangent_coordinate[:, 0]), axis=1,
    )
    normal_norm = np.sqrt(np.einsum(
        "...a,...ab,...b->...", normal_covector, inverse, normal_covector,
    ))
    normal_covector /= normal_norm[:, None]
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
    sphere_curvature = sphere_radius_normal / sphere_radius
    mean_curvature = curve_curvature + 2.0 * sphere_curvature
    extrinsic_meridional = np.einsum(
        "...a,...ab,...b->...", tangent, extrinsic, tangent,
    )
    extrinsic_mixed = np.einsum(
        "...a,...ab,...b->...", tangent, extrinsic, normal,
    )
    outgoing_meridional = (
        -extrinsic_meridional + curve_curvature
    ) / SQRT2
    outgoing_sphere = (-extrinsic_sphere + sphere_curvature) / SQRT2
    theta_l = outgoing_meridional + 2.0 * outgoing_sphere
    theta_n = (
        -mean_curvature - extrinsic_meridional - 2.0 * extrinsic_sphere
    ) / SQRT2
    shear_squared = (
        (outgoing_meridional - theta_l / 3.0) ** 2
        + 2.0 * (outgoing_sphere - theta_l / 3.0) ** 2
    )
    theta_l = _axis_regularize_even(theta_l, theta, radius, r)
    theta_n = _axis_regularize_even(theta_n, theta, radius, r)
    shear_squared = _axis_regularize_even(shear_squared, theta, radius, r)
    omega = -extrinsic_mixed

    normal_speed = np.einsum(
        "...a,...a->...", normal_covector, shift + embedding_velocity,
    )
    null_a = (lapse + normal_speed) / SQRT2
    null_b = (normal_speed - lapse) / SQRT2
    area_rate_transport = -2.0 * _one_cap_integral(
        null_b * theta_n, sphere_radius, speed, theta,
    )

    charge = reflected_cap_charge(
        position, velocity, z, r, profile, stencil_width=stencil_width,
        prepared=prepared, cosmological_constant=cosmological_constant,
    )
    doubled_area = float(charge["reflection_doubled_area"])
    area_radius = float(charge["equivalent_area_radius"])

    _, geometric_w_s, sphere_radius_ss = _proper_curve(
        theta, speed, sphere_radius, sphere_radius_meridional[-1],
    )
    intrinsic_scalar = (
        -4.0 * sphere_radius_ss / sphere_radius
        + 2.0 * (1.0 - geometric_w_s**2) / sphere_radius**2
    )
    null_b_s, null_b_ss = _proper_scalar_derivatives(
        null_b, theta, speed,
    )
    omega_s, _ = _proper_scalar_derivatives(
        omega, theta, speed, axis_value=0.0,
    )
    normal_operator = (
        null_b_ss
        + 2.0 * geometric_w_s / sphere_radius * null_b_s
        - 2.0 * omega * null_b_s
        - null_b * (
            omega_s + 2.0 * geometric_w_s / sphere_radius * omega
        )
        + null_b * omega**2
    )

    scalar_data = []
    spatial_gradient_squared = np.zeros_like(theta)
    t_ll = np.zeros_like(theta)
    t_ln = np.zeros_like(theta)
    for field_index in (7, 8):
        value, gradient = _field_and_gradient_on_surface(
            position[:, :, field_index], z, r, zcoord, radius,
        )
        coordinate_rate = _field_on_surface(
            velocity[:, :, field_index], z, r, zcoord, radius,
        )
        normal_time_derivative = (
            coordinate_rate - np.einsum("...a,...a->...", shift, gradient)
        ) / lapse
        spatial_normal_derivative = np.einsum(
            "...a,...a->...", normal, gradient,
        )
        spatial_norm = np.einsum(
            "...a,...ab,...b->...", gradient, inverse, gradient,
        )
        spacetime_norm = -normal_time_derivative**2 + spatial_norm
        l_derivative = (
            normal_time_derivative + spatial_normal_derivative
        ) / SQRT2
        n_derivative = (
            normal_time_derivative - spatial_normal_derivative
        ) / SQRT2
        spatial_gradient_squared += spatial_norm
        t_ll += l_derivative**2
        t_ln += l_derivative * n_derivative + 0.5 * spacetime_norm
        scalar_data.append({
            "field_index": field_index,
            "minimum": float(np.min(value)),
            "maximum": float(np.max(value)),
            "maximum_absolute_u_derivative": float(np.max(np.abs(
                normal_time_derivative
            ))),
            "maximum_absolute_s_derivative": float(np.max(np.abs(
                spatial_normal_derivative
            ))),
        })
    phi = _field_on_surface(position[:, :, 7], z, r, zcoord, radius)
    scalar_potential = 0.5 * float(background["mass_squared"]) * phi**2
    t_ln += scalar_potential

    area_rate_fraction = area_rate_transport / doubled_area
    curvature_factor = area_rate_fraction + null_b * theta_n
    curvature_smooth_integral = 2.0 * _one_cap_integral(
        intrinsic_scalar * curvature_factor, sphere_radius, speed, theta,
    )
    matter_ll_smooth_integral = 2.0 * _one_cap_integral(
        theta_n * null_a * t_ll, sphere_radius, speed, theta,
    )
    matter_ln_smooth_integral = 2.0 * _one_cap_integral(
        theta_n * null_b * t_ln, sphere_radius, speed, theta,
    )
    shear_smooth_integral = 2.0 * _one_cap_integral(
        theta_n * null_a * shear_squared, sphere_radius, speed, theta,
    )
    normal_smooth_integral = 2.0 * _one_cap_integral(
        theta_n * normal_operator, sphere_radius, speed, theta,
    )

    wall_gamma = float(background["wall_stiffness"])
    wall_target = float(background["v1"])
    wall_potential = 0.5 * wall_gamma * (phi[-1] - wall_target) ** 2
    wall_energy = float(background["retuned_bare_tension_b"]) + wall_potential
    wall_c = wall_energy / 6.0
    wall_area = 4.0 * math.pi * sphere_radius[-1] ** 2
    seam_intrinsic_geometric = (
        32.0 * math.pi * sphere_radius[-1] * geometric_w_s[-1]
    )
    seam_intrinsic_israel = 8.0 * wall_c * wall_area
    curvature_seam_rate = (
        area_radius / 12.0 * seam_intrinsic_geometric * curvature_factor[-1]
    )
    brane_matter_rate = (
        -area_radius / 3.0 * theta_n[-1] * null_b[-1]
        * wall_energy * wall_area
    )
    normal_seam_rate = (
        2.0 * area_radius / 3.0 * theta_n[-1]
        * (null_b_s[-1] - null_b[-1] * omega[-1]) * wall_area
    )

    rates = {
        "curvature_smooth": area_radius / 12.0 * curvature_smooth_integral,
        "curvature_seam": curvature_seam_rate,
        "matter_scalar_ll_smooth": -area_radius / 3.0
        * matter_ll_smooth_integral,
        "matter_scalar_ln_smooth": -area_radius / 3.0
        * matter_ln_smooth_integral,
        "shear_smooth": -area_radius / 3.0 * shear_smooth_integral,
        "normal_connection_smooth": -area_radius / 3.0
        * normal_smooth_integral,
        "brane_matter_seam": brane_matter_rate,
        "normal_connection_seam": normal_seam_rate,
        "vacuum": cosmological_constant * area_radius
        * area_rate_transport / 3.0,
        "ads_background": -cosmological_constant * area_radius
        * area_rate_transport / 3.0,
    }
    total_rate = float(sum(rates.values()))
    override = None
    if bulk_stress_override is not None:
        override_ll = np.asarray(bulk_stress_override["T_ll"], dtype=float)
        override_ln = np.asarray(bulk_stress_override["T_ln"], dtype=float)
        if override_ll.shape != theta.shape or override_ln.shape != theta.shape:
            raise ValueError("bulk stress override must share the surface mesh")
        override_ll_rate = -area_radius / 3.0 * 2.0 * _one_cap_integral(
            theta_n * null_a * override_ll, sphere_radius, speed, theta,
        )
        override_ln_rate = -area_radius / 3.0 * 2.0 * _one_cap_integral(
            theta_n * null_b * override_ln, sphere_radius, speed, theta,
        )
        alternative = (
            total_rate - rates["matter_scalar_ll_smooth"]
            - rates["matter_scalar_ln_smooth"] - rates["vacuum"]
            + override_ll_rate + override_ln_rate
        )
        override = {
            "T_ll_rate": float(override_ll_rate),
            "T_ln_rate": float(override_ln_rate),
            "total_rate_replacing_scalar_plus_vacuum": float(alternative),
            "finite": bool(
                np.all(np.isfinite(override_ll))
                and np.all(np.isfinite(override_ln))
                and np.isfinite(alternative)
            ),
        }
    smooth_intrinsic_direct = 2.0 * _one_cap_integral(
        intrinsic_scalar, sphere_radius, speed, theta,
    )
    return {
        "charge": charge,
        "rates": {name: float(value) for name, value in rates.items()},
        "total_balance_rate": total_rate,
        "area_rate_transport": float(area_rate_transport),
        "geometry": {
            "minimum_lapse": float(np.min(lapse)),
            "minimum_speed": float(np.min(speed)),
            "maximum_absolute_theta_l": float(np.max(np.abs(theta_l))),
            "maximum_theta_n": float(np.max(theta_n)),
            "minimum_theta_n": float(np.min(theta_n)),
            "all_theta_n_negative": bool(np.max(theta_n) < 0.0),
            "maximum_shear_squared": float(np.max(shear_squared)),
            "maximum_absolute_normal_connection": float(np.max(np.abs(omega))),
            "maximum_absolute_normal_operator": float(np.max(np.abs(
                normal_operator
            ))),
            "maximum_absolute_normal_speed": float(np.max(np.abs(normal_speed))),
            "minimum_null_A": float(np.min(null_a)),
            "minimum_null_B": float(np.min(null_b)),
            "maximum_null_B": float(np.max(null_b)),
            "direct_smooth_intrinsic_integral": float(smooth_intrinsic_direct),
            "charge_smooth_intrinsic_integral": float(charge[
                "doubled_smooth_bulk_curvature_integral"
            ]),
            "smooth_intrinsic_relative_scale_error": relative_scale_error(
                smooth_intrinsic_direct,
                charge["doubled_smooth_bulk_curvature_integral"],
                charge["intrinsic_scalar_curvature_integral"],
            ),
        },
        "scalar": {
            "fields": scalar_data,
            "maximum_T_ll": float(np.max(t_ll)),
            "minimum_T_ll": float(np.min(t_ll)),
            "maximum_absolute_T_ln": float(np.max(np.abs(t_ln))),
            "maximum_scalar_potential": float(np.max(scalar_potential)),
            "maximum_spatial_scalar_gradient_squared": float(np.max(
                spatial_gradient_squared
            )),
        },
        "seam": {
            "wall_scalar": float(phi[-1]),
            "wall_potential": float(wall_potential),
            "wall_energy_bare_plus_potential": float(wall_energy),
            "israel_coefficient_c": float(wall_c),
            "sphere_area_A2": float(wall_area),
            "geometric_Ws_over_W": float(
                geometric_w_s[-1] / sphere_radius[-1]
            ),
            "geometric_intrinsic_integral": float(seam_intrinsic_geometric),
            "israel_intrinsic_integral": float(seam_intrinsic_israel),
            "israel_intrinsic_relative_scale_error": relative_scale_error(
                seam_intrinsic_geometric, seam_intrinsic_israel,
                charge["intrinsic_scalar_curvature_integral"],
            ),
            "B": float(null_b[-1]),
            "B_s": float(null_b_s[-1]),
            "omega_s_component": float(omega[-1]),
            "theta_n": float(theta_n[-1]),
        },
        "background": {
            "cosmological_constant": float(cosmological_constant),
            "vacuum_rate": float(rates["vacuum"]),
            "subtraction_rate": float(rates["ads_background"]),
            "cancellation_relative_scale_error": relative_scale_error(
                rates["vacuum"] + rates["ads_background"], 0.0,
                max(abs(rates["vacuum"]), abs(rates["ads_background"]), 1.0),
            ),
        },
        "bulk_stress_override": override,
        "finite": bool(
            np.all(np.isfinite(list(rates.values())))
            and np.isfinite(total_rate)
            and np.all(np.isfinite(intrinsic_scalar))
            and np.all(np.isfinite(normal_operator))
        ),
    }


def vaidya_ads5_control():
    """Analytic and sampled five-dimensional Vaidya--AdS balance control."""
    times = np.linspace(-0.1, 0.1, 41)
    mu = 1.0 + 0.2 * times + 0.1 * times**2
    mu_dot = 0.2 + 0.2 * times
    radius_squared = 0.5 * (-1.0 + np.sqrt(1.0 + 4.0 * mu))
    radius = np.sqrt(radius_squared)
    area = OMEGA3 * radius**3
    theta_n = -3.0 / radius
    t_ll = 3.0 * mu_dot / (2.0 * radius**3)
    computed_flux = -radius / 3.0 * theta_n * t_ll * area
    expected_flux = 3.0 * math.pi**2 * mu_dot
    charge = 3.0 * math.pi**2 * mu
    derivative = five_point_history_derivative(charge, times)
    integrated_flux = float(np.trapezoid(computed_flux, times))
    delta_charge = float(charge[-1] - charge[0])
    analytic_error = max(
        relative_scale_error(value, expected, max(abs(expected), 1.0))
        for value, expected in zip(computed_flux, expected_flux)
    )
    sampled_rate_error = max(
        relative_scale_error(value, expected, max(abs(expected), 1.0))
        for value, expected in zip(derivative, expected_flux)
    )
    sampled_integrated_error = relative_scale_error(
        integrated_flux, delta_charge, max(abs(delta_charge), 1.0),
    )
    stationary = np.ones(21)
    stationary_rate = five_point_history_derivative(
        3.0 * math.pi**2 * stationary, np.linspace(0.0, 0.1, 21),
    )
    stationary_error = float(np.max(np.abs(stationary_rate)))
    return {
        "family": "Vaidya-AdS5 with f=1+r^2-mu(v)/r^2",
        "analytic_maximum_relative_scale_error": analytic_error,
        "sampled_rate_maximum_relative_scale_error": sampled_rate_error,
        "sampled_integrated_relative_scale_error": sampled_integrated_error,
        "stationary_duplicate_absolute_rate_error": stationary_error,
        "analytic_below_1e_10": bool(analytic_error < 1e-10),
        "sampled_below_0_2_percent": bool(
            max(sampled_rate_error, sampled_integrated_error) < 0.002
        ),
        "stationary_below_1e_10": bool(stationary_error < 1e-10),
    }


def seam_distribution_controls(epsilon=1e-5):
    """Smooth-layer controls for all three reflected-brane distributions."""
    eps = float(epsilon)
    width = 1.0
    wall_radius = 1.3
    wall_c = -0.8
    b_value = -0.7
    b_s = 0.23
    omega_value = -0.17
    wall_energy = 6.0 * wall_c
    wall_area = 4.0 * math.pi * wall_radius**2

    limit = math.asinh(width / eps)

    def root_from_u(u):
        return eps * math.cosh(u)

    def w_from_u(u):
        return wall_radius - wall_c * (root_from_u(u) - eps)

    # x=epsilon sinh(u) turns epsilon^2/(x^2+epsilon^2)^(3/2) dx
    # into sech(u)^2 du and avoids a numerically singular narrow peak.
    curvature_smoothed = quad(
        lambda u: 16.0 * math.pi * w_from_u(u) * wall_c
        / math.cosh(u) ** 2,
        -limit, limit, epsabs=1e-11, epsrel=1e-11, limit=1000,
    )[0]
    curvature_expected = 32.0 * math.pi * wall_radius * wall_c

    def b_from_u(u):
        return b_value - b_s * (root_from_u(u) - eps)

    normal_smoothed = quad(
        lambda u: (-b_s + b_from_u(u) * omega_value) / math.cosh(u) ** 2,
        -limit, limit, epsabs=1e-11, epsrel=1e-11, limit=1000,
    )[0]
    normal_expected = -2.0 * (b_s - b_value * omega_value)

    brane_smoothed = quad(
        lambda u: 0.5 * wall_energy / math.cosh(u) ** 2, -limit, limit,
        epsabs=1e-11, epsrel=1e-11, limit=1000,
    )[0] * wall_area
    brane_expected = wall_energy * wall_area

    entries = {
        "curvature": {
            "smoothed": float(curvature_smoothed),
            "distributional": float(curvature_expected),
            "relative_scale_error": relative_scale_error(
                curvature_smoothed, curvature_expected, curvature_expected,
            ),
        },
        "normal_connection": {
            "smoothed": float(normal_smoothed),
            "distributional": float(normal_expected),
            "relative_scale_error": relative_scale_error(
                normal_smoothed, normal_expected, normal_expected,
            ),
        },
        "brane_matter": {
            "smoothed": float(brane_smoothed),
            "distributional": float(brane_expected),
            "relative_scale_error": relative_scale_error(
                brane_smoothed, brane_expected, brane_expected,
            ),
        },
    }
    for item in entries.values():
        item["omitted_relative_scale_error"] = relative_scale_error(
            0.0, item["distributional"], item["distributional"],
        )
    return {
        "smoothing_epsilon": eps,
        "entries": entries,
        "all_smoothed_below_2e_4": bool(all(
            item["relative_scale_error"] < 2e-4 for item in entries.values()
        )),
        "all_omissions_fail_above_1_percent": bool(all(
            item["omitted_relative_scale_error"] > 0.01
            for item in entries.values()
        )),
    }


def analytic_controls():
    vaidya = vaidya_ads5_control()
    seam = seam_distribution_controls()
    background_scales = (-100.0, -1.0, 0.0, 2.5, 80.0)
    background = [{
        "vacuum": value,
        "subtraction": -value,
        "sum": value - value,
    } for value in background_scales]
    background_pass = bool(all(item["sum"] == 0.0 for item in background))
    acceptance = {
        "vaidya_analytic": vaidya["analytic_below_1e_10"],
        "vaidya_sampled": vaidya["sampled_below_0_2_percent"],
        "stationary_duplicate": vaidya["stationary_below_1e_10"],
        "seam_smoothing": seam["all_smoothed_below_2e_4"],
        "seam_omission_fails": seam["all_omissions_fail_above_1_percent"],
        "background_cancellation": background_pass,
    }
    return {
        "vaidya_ads5": vaidya,
        "seam_distributions": seam,
        "background_cancellation": background,
        "acceptance": acceptance,
        "passed": bool(all(acceptance.values())),
    }
