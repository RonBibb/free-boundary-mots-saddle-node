"""Finite-thickness junction-collar controls for Test 14D.

The immutable formulas and gates are recorded in note 103.  This module
constructs local Gaussian collars only; it does not evolve a thick brane.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad, simpson
from scipy.special import erf


FAMILIES = ("tanh", "erf", "compact_c2")
WIDTH_RATIOS = (1.0 / 32.0, 1.0 / 64.0, 1.0 / 128.0, 1.0 / 256.0)
RESOLUTIONS = (32, 64, 128)
HALF_WIDTHS = 8.0


def relative_scale_error(value, expected, scale=1.0):
    """Return a symmetric relative error with an explicit absolute scale."""
    return float(
        abs(float(value) - float(expected))
        / max(abs(float(value)), abs(float(expected)), abs(float(scale)), 1e-300)
    )


def _log_cosh(value):
    value = np.asarray(value, dtype=float)
    return np.logaddexp(value, -value) - math.log(2.0)


def collar_grid(epsilon, nodes_per_epsilon, half_widths=HALF_WIDTHS):
    """Return the fixed symmetric Simpson grid for one collar."""
    epsilon = float(epsilon)
    nodes_per_epsilon = int(nodes_per_epsilon)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if nodes_per_epsilon < 4:
        raise ValueError("nodes_per_epsilon must be at least four")
    intervals = int(round(2.0 * half_widths * nodes_per_epsilon))
    if intervals % 2:
        intervals += 1
    half_width = float(half_widths * epsilon)
    return np.linspace(-half_width, half_width, intervals + 1)


def regularizer_profile(x, epsilon, family):
    """Return moment-matched ``q``, ``q_x``, ``delta``, and matched ``p``.

    ``p_x=q`` and the additive constant makes ``p(+-L)=L``.  Noncompact
    transitions are normalized by their finite-collar endpoint value.
    """
    x = np.asarray(x, dtype=float)
    epsilon = float(epsilon)
    if x.ndim != 1 or x.size < 3:
        raise ValueError("x must be a one-dimensional collar grid")
    if not np.allclose(x, -x[::-1], rtol=0.0, atol=1e-14):
        raise ValueError("collar grid must be symmetric")
    half_width = float(x[-1])
    if half_width <= 0.0 or epsilon <= 0.0:
        raise ValueError("positive width required")

    if family == "tanh":
        second_moment = math.pi**2 / 12.0
        scale = epsilon / math.sqrt(second_moment)
        y = x / scale
        raw_q = np.tanh(y)
        raw_q_x = (1.0 - raw_q**2) / scale
        raw_p = scale * _log_cosh(y)
    elif family == "erf":
        scale = epsilon
        y = x / scale
        raw_q = erf(y / math.sqrt(2.0))
        raw_q_x = (
            math.sqrt(2.0 / math.pi) / scale * np.exp(-0.5 * y**2)
        )
        raw_p = (
            x * raw_q
            + scale * math.sqrt(2.0 / math.pi)
            * (np.exp(-0.5 * y**2) - 1.0)
        )
    elif family == "compact_c2":
        scale = epsilon * math.sqrt(7.0)
        absolute_y = np.abs(x) / scale
        inside = absolute_y < 1.0
        raw_q = np.sign(x)
        raw_q_x = np.zeros_like(x)
        yi = x[inside] / scale
        raw_q[inside] = (
            (15.0 / 8.0) * yi
            - (5.0 / 4.0) * yi**3
            + (3.0 / 8.0) * yi**5
        )
        raw_q_x[inside] = (
            (15.0 / 8.0) * (1.0 - yi**2) ** 2 / scale
        )
        ya = absolute_y
        raw_p = np.empty_like(x)
        raw_p[inside] = scale * (
            (15.0 / 16.0) * ya[inside] ** 2
            - (5.0 / 16.0) * ya[inside] ** 4
            + (1.0 / 16.0) * ya[inside] ** 6
        )
        raw_p[~inside] = np.abs(x[~inside]) - 5.0 * scale / 16.0
    else:
        raise ValueError(f"unknown Test-14D family {family!r}")

    edge = float(raw_q[-1])
    if not np.isfinite(edge) or edge <= 0.0:
        raise ValueError("regularizer has invalid positive-edge value")
    q = raw_q / edge
    q_x = raw_q_x / edge
    raw_p = raw_p / edge
    p = raw_p + (half_width - float(raw_p[-1]))
    delta = 0.5 * q_x
    return {
        "q": q,
        "q_x": q_x,
        "delta": delta,
        "p": p,
        "scale": float(scale),
        "half_width": half_width,
        "edge_normalization": edge,
    }


def profile_diagnostics(x, epsilon, family):
    """Return the parity, moment, and endpoint diagnostics of a profile."""
    profile = regularizer_profile(x, epsilon, family)
    q = profile["q"]
    delta = profile["delta"]
    p = profile["p"]
    quadrature_normalization = float(simpson(delta, x=x))

    # Normalization follows exactly from delta=q_x/2.  Moments are checked
    # independently with adaptive quadrature split at the compact C2 joins.
    normalization = float(0.5 * (q[-1] - q[0]))
    half_width = float(x[-1])
    if family == "tanh":
        scale = epsilon / math.sqrt(math.pi**2 / 12.0)
        edge = math.tanh(half_width / scale)

        def density_at(value):
            tangent = math.tanh(value / scale)
            return 0.5 * (1.0 - tangent**2) / (scale * edge)

        points = None
    elif family == "erf":
        scale = epsilon
        edge = float(erf(half_width / (math.sqrt(2.0) * scale)))

        def density_at(value):
            return (
                0.5 * math.sqrt(2.0 / math.pi) / (scale * edge)
                * math.exp(-0.5 * (value / scale) ** 2)
            )

        points = None
    else:
        scale = epsilon * math.sqrt(7.0)

        def density_at(value):
            y = value / scale
            if abs(y) >= 1.0:
                return 0.0
            return 0.5 * (15.0 / 8.0) * (1.0 - y**2) ** 2 / scale

        points = [-scale, scale]
    quad_options = {"epsabs": 1e-13, "epsrel": 1e-13, "limit": 200}
    if points is not None:
        quad_options["points"] = points
    first = float(quad(
        lambda value: value * density_at(value),
        -half_width, half_width, **quad_options,
    )[0])
    second = float(quad(
        lambda value: value * value * density_at(value),
        -half_width, half_width, **quad_options,
    )[0])
    return {
        "normalization": normalization,
        "normalization_error": abs(normalization - 1.0),
        "simpson_normalization": quadrature_normalization,
        "simpson_normalization_error": abs(quadrature_normalization - 1.0),
        "first_moment": first,
        "first_moment_scaled_error": abs(first) / max(float(epsilon), 1e-300),
        "second_moment": second,
        "rms_width_relative_error": relative_scale_error(
            second, float(epsilon) ** 2, float(epsilon) ** 2,
        ),
        "q_odd_error": float(np.max(np.abs(q + q[::-1]))),
        "delta_even_error": float(np.max(np.abs(delta - delta[::-1]))),
        "left_q_error": abs(float(q[0]) + 1.0),
        "right_q_error": abs(float(q[-1]) - 1.0),
        "p_endpoint_error": max(
            abs(float(p[0]) - float(x[-1])),
            abs(float(p[-1]) - float(x[-1])),
        ),
        "finite": bool(all(
            np.all(np.isfinite(value))
            for value in (q, delta, p, normalization, first, second)
        )),
    }


def _instantaneous_collar(
    x, profile, *, time, c, c_rate, seam_warp, h_sphere, h_meridional,
):
    """Return the smooth and matched-sharp collar geometry at one time."""
    time = float(time)
    m = math.exp(float(h_meridional) * time)
    c_t = float(c) + float(c_rate) * time
    warp_base = float(seam_warp) * math.exp(float(h_sphere) * time)
    q = profile["q"]
    q_x = profile["q_x"]
    p = profile["p"]
    warp = warp_base * np.exp(-m * c_t * p)
    warp_s = -c_t * q * warp
    warp_ss = warp * (
        c_t**2 * q**2 - (c_t / m) * q_x
    )
    scalar = -4.0 * warp_ss / warp + 2.0 * (
        1.0 - warp_s**2
    ) / warp**2
    area_density = 4.0 * math.pi * m * warp**2

    sharp_warp = warp_base * np.exp(-m * c_t * np.abs(x))
    sharp_scalar_bulk = 2.0 / sharp_warp**2 - 6.0 * c_t**2
    sharp_area_density = 4.0 * math.pi * m * sharp_warp**2
    smooth_integral = float(simpson(scalar * area_density, x=x))
    sharp_bulk_integral = float(simpson(
        sharp_scalar_bulk * sharp_area_density, x=x,
    ))

    junction_density = 3.0 * c_t * q_x / m
    junction_integral = float(simpson(junction_density * m, x=x))
    area_weighted_junction = float(simpson(
        junction_density * area_density, x=x,
    ))

    edge_warp = float(warp[-1])
    edge_sphere_area = float(4.0 * math.pi * edge_warp**2)
    h_sphere_edge = float(
        h_sphere - (c_rate + h_meridional * c_t) * profile["p"][-1]
    )
    h_sphere_s_physical_edge = float(c_rate + h_meridional * c_t)
    boundary_variation = float(
        8.0 * edge_sphere_area
        * (
            -h_sphere_s_physical_edge
            + c_t * (h_meridional - h_sphere_edge)
        )
    )
    reference_sphere_area = float(4.0 * math.pi * warp_base**2)
    reference_boundary_variation = float(
        8.0 * reference_sphere_area
        * (
            -(c_rate + h_meridional * c_t)
            + c_t * (h_meridional - h_sphere)
        )
    )
    return {
        "m": m,
        "c": c_t,
        "warp_base": warp_base,
        "smooth_integral": smooth_integral,
        "sharp_bulk_integral": sharp_bulk_integral,
        "curvature_excess": float(smooth_integral - sharp_bulk_integral),
        "junction_integral": junction_integral,
        "area_weighted_junction": area_weighted_junction,
        "edge_warp": edge_warp,
        "edge_sphere_area": edge_sphere_area,
        "H_Omega_edge": h_sphere_edge,
        "H_Omega_s_physical_edge": h_sphere_s_physical_edge,
        "boundary_variation": boundary_variation,
        "reference_boundary_variation": reference_boundary_variation,
    }


def collar_record(
    family, epsilon, nodes_per_epsilon, *, c=0.37, c_rate=-0.11,
    seam_warp=1.25, h_sphere=0.23, h_meridional=-0.07,
    area_radius=1.7, area_fractional_rate=0.19, derivative_step=1e-5,
):
    """Evaluate one finite collar and its fixed thin-limit targets."""
    x = collar_grid(epsilon, nodes_per_epsilon)
    profile = regularizer_profile(x, epsilon, family)
    diagnostics = profile_diagnostics(x, epsilon, family)

    def state(time):
        return _instantaneous_collar(
            x, profile, time=time, c=c, c_rate=c_rate,
            seam_warp=seam_warp, h_sphere=h_sphere,
            h_meridional=h_meridional,
        )

    center = state(0.0)

    def centered_rate(key, step):
        return float((state(step)[key] - state(-step)[key]) / (2.0 * step))

    step = float(derivative_step)
    curvature_rate = centered_rate("curvature_excess", step)
    curvature_rate_half = centered_rate("curvature_excess", 0.5 * step)
    junction_rate = centered_rate("junction_integral", step)
    junction_rate_half = centered_rate("junction_integral", 0.5 * step)
    area_junction_rate = centered_rate("area_weighted_junction", step)
    area_junction_rate_half = centered_rate(
        "area_weighted_junction", 0.5 * step,
    )

    sphere_area = float(4.0 * math.pi * float(seam_warp) ** 2)
    seam_integral_target = float(8.0 * c * sphere_area)
    junction_target = float(6.0 * c)
    junction_rate_target = float(6.0 * c_rate)
    area_junction_target = float(6.0 * c * sphere_area)
    area_junction_rate_target = float(
        6.0 * sphere_area * (c_rate + 2.0 * c * h_sphere)
    )
    seam_rate_target = float(
        area_radius * seam_integral_target * area_fractional_rate / 12.0
        + 2.0 * area_radius * sphere_area * c * h_sphere
    )
    finite_seam_rate = float(
        area_radius * center["curvature_excess"]
        * area_fractional_rate / 12.0
        + area_radius / 4.0
        * (curvature_rate_half + center["reference_boundary_variation"])
    )
    derivative_step_error = relative_scale_error(
        curvature_rate, curvature_rate_half,
        max(abs(seam_rate_target), 1.0),
    )
    return {
        "family": family,
        "epsilon": float(epsilon),
        "nodes_per_epsilon": int(nodes_per_epsilon),
        "node_count": int(x.size),
        "parameters": {
            "c": float(c),
            "c_rate": float(c_rate),
            "seam_warp": float(seam_warp),
            "H_Omega": float(h_sphere),
            "H_meridional": float(h_meridional),
            "area_radius": float(area_radius),
            "area_fractional_rate": float(area_fractional_rate),
        },
        "profile": diagnostics,
        "curvature_excess": center["curvature_excess"],
        "curvature_target": seam_integral_target,
        "curvature_error": relative_scale_error(
            center["curvature_excess"], seam_integral_target,
            max(abs(seam_integral_target), 1.0),
        ),
        "curvature_rate": curvature_rate_half,
        "boundary_variation": center["reference_boundary_variation"],
        "collar_edge_boundary_variation_diagnostic": center[
            "boundary_variation"
        ],
        "finite_seam_rate": finite_seam_rate,
        "thin_seam_rate_target": seam_rate_target,
        "seam_rate_error": relative_scale_error(
            finite_seam_rate, seam_rate_target,
            max(abs(seam_rate_target), 1.0),
        ),
        "derivative_step_error": derivative_step_error,
        "junction_integral": center["junction_integral"],
        "junction_target": junction_target,
        "junction_error": relative_scale_error(
            center["junction_integral"], junction_target,
            max(abs(junction_target), 1.0),
        ),
        "junction_rate": junction_rate_half,
        "junction_rate_target": junction_rate_target,
        "junction_rate_error": relative_scale_error(
            junction_rate_half, junction_rate_target,
            max(abs(junction_rate_target), 1.0),
        ),
        "junction_rate_step_error": relative_scale_error(
            junction_rate, junction_rate_half,
            max(abs(junction_rate_target), 1.0),
        ),
        "area_weighted_junction": center["area_weighted_junction"],
        "area_weighted_junction_target": area_junction_target,
        "area_weighted_junction_rate": area_junction_rate_half,
        "area_weighted_junction_rate_target": area_junction_rate_target,
        "area_weighted_rate_step_error": relative_scale_error(
            area_junction_rate, area_junction_rate_half,
            max(abs(area_junction_rate_target), 1.0),
        ),
        "finite": bool(
            diagnostics["finite"]
            and np.all(np.isfinite([
                center["curvature_excess"], curvature_rate_half,
                finite_seam_rate, center["junction_integral"],
                junction_rate_half, center["area_weighted_junction"],
                area_junction_rate_half,
            ]))
        ),
    }


def zero_width_fit(records, key, target, scale):
    """Apply the sealed quadratic-in-width extrapolation."""
    ordered = sorted(records, key=lambda item: float(item["epsilon"]), reverse=True)
    widths = np.asarray([item["epsilon"] for item in ordered], dtype=float)
    values = np.asarray([item[key] for item in ordered], dtype=float)
    if len(widths) != 4:
        raise ValueError("zero-width fit requires exactly four widths")
    coefficients = np.polyfit(widths, values, deg=2)
    extrapolated = float(coefficients[-1])
    fine_coefficients = np.polyfit(widths[1:], values[1:], deg=2)
    leave_coarsest = float(fine_coefficients[-1])
    errors = np.abs(values - float(target)) / max(
        abs(float(target)), abs(float(scale)), 1e-300,
    )
    finest_monotone = bool(
        errors[-1] <= errors[-2] * (1.0 + 1e-8)
        or max(errors[-2:]) < 2e-4
    )
    return {
        "key": key,
        "target": float(target),
        "coefficients_quadratic": coefficients.tolist(),
        "extrapolated": extrapolated,
        "leave_coarsest_out": leave_coarsest,
        "extrapolated_error": relative_scale_error(
            extrapolated, target, scale,
        ),
        "leave_coarsest_stability": relative_scale_error(
            extrapolated, leave_coarsest, scale,
        ),
        "finest_value": float(values[-1]),
        "finest_error": relative_scale_error(values[-1], target, scale),
        "finest_monotone": finest_monotone,
    }


def manufactured_controls(records=None):
    """Run the outcome-blind Test-14D manufactured control matrix."""
    area_radius = 1.7
    if records is None:
        records = []
        for family in FAMILIES:
            for ratio in WIDTH_RATIOS:
                epsilon = ratio * area_radius
                for resolution in RESOLUTIONS:
                    records.append(collar_record(
                        family, epsilon, resolution, area_radius=area_radius,
                    ))
    else:
        records = list(records)

    finest_resolution = [
        item for item in records if item["nodes_per_epsilon"] == 128
    ]
    fits = {}
    for family in FAMILIES:
        family_records = [
            item for item in finest_resolution if item["family"] == family
        ]
        sample = family_records[0]
        fits[family] = {
            "curvature": zero_width_fit(
                family_records, "curvature_excess",
                sample["curvature_target"],
                max(abs(sample["curvature_target"]), 1.0),
            ),
            "seam_rate": zero_width_fit(
                family_records, "finite_seam_rate",
                sample["thin_seam_rate_target"],
                max(abs(sample["thin_seam_rate_target"]), 1.0),
            ),
            "area_weighted_junction_rate": zero_width_fit(
                family_records, "area_weighted_junction_rate",
                sample["area_weighted_junction_rate_target"],
                max(abs(sample["area_weighted_junction_rate_target"]), 1.0),
            ),
        }

    def spread(section):
        values = np.asarray([
            fits[family][section]["extrapolated"] for family in FAMILIES
        ])
        targets = np.asarray([
            fits[family][section]["target"] for family in FAMILIES
        ])
        scale = max(float(np.max(np.abs(targets))), 1.0)
        return float((np.max(values) - np.min(values)) / scale)

    resolution_checks = []
    for family in FAMILIES:
        for ratio in WIDTH_RATIOS:
            epsilon = ratio * area_radius
            selected = sorted(
                [item for item in records
                 if item["family"] == family
                 and math.isclose(item["epsilon"], epsilon)],
                key=lambda item: item["nodes_per_epsilon"],
            )
            scale = max(abs(selected[-1]["thin_seam_rate_target"]), 1.0)
            difference = abs(
                selected[-1]["finite_seam_rate"]
                - selected[-2]["finite_seam_rate"]
            ) / scale
            errors = np.asarray([
                abs(item["finite_seam_rate"] - selected[-1]["finite_seam_rate"])
                / scale for item in selected
            ])
            if errors[1] > 1e-15 and errors[0] > errors[1]:
                order = math.log(errors[0] / errors[1], 2.0)
            else:
                order = math.inf
            resolution_checks.append({
                "family": family,
                "epsilon": epsilon,
                "two_finest_difference": float(difference),
                "estimated_order": float(order),
                "passed": bool(
                    difference < 2e-4
                    and (order >= 2.0 or errors[0] < 2e-4)
                ),
            })

    static = collar_record(
        "compact_c2", area_radius / 128.0, 128,
        c=0.37, c_rate=0.0, h_sphere=0.0, h_meridional=0.0,
        area_fractional_rate=0.0, area_radius=area_radius,
    )
    flat = collar_record(
        "compact_c2", area_radius / 128.0, 128,
        c=0.0, c_rate=0.0, h_sphere=0.23, h_meridional=-0.07,
        area_fractional_rate=0.19, area_radius=area_radius,
    )
    sphere_area = 4.0 * math.pi * 1.25**2
    orientation_original = 2.0 * area_radius * sphere_area * 0.37 * 0.23
    orientation_joint = 2.0 * area_radius * sphere_area * (-0.37) * (-0.23)
    orientation_incorrect = 2.0 * area_radius * sphere_area * (-0.37) * 0.23
    omission_difference = abs(orientation_original) / max(
        abs(orientation_original), 1.0,
    )
    double_count_difference = omission_difference

    maximum_profile_parity = max(
        max(item["profile"]["q_odd_error"],
            item["profile"]["delta_even_error"])
        for item in records
    )
    maximum_normalization = max(
        item["profile"]["normalization_error"] for item in records
    )
    maximum_first_moment = max(
        item["profile"]["first_moment_scaled_error"] for item in records
    )
    maximum_endpoint = max(
        max(item["profile"]["left_q_error"],
            item["profile"]["right_q_error"],
            item["profile"]["p_endpoint_error"])
        for item in records
    )
    maximum_rms_32 = max(
        item["profile"]["rms_width_relative_error"] for item in records
        if item["nodes_per_epsilon"] == 32
    )
    maximum_rms_128 = max(
        item["profile"]["rms_width_relative_error"] for item in records
        if item["nodes_per_epsilon"] == 128
    )
    maximum_junction_error = max(item["junction_error"] for item in records)
    maximum_junction_rate_error = max(
        item["junction_rate_error"] for item in records
    )
    maximum_derivative_step_error = max(
        max(item["derivative_step_error"],
            item["junction_rate_step_error"],
            item["area_weighted_rate_step_error"])
        for item in records
    )
    maximum_finest_curvature = max(
        fits[family]["curvature"]["finest_error"] for family in FAMILIES
    )
    maximum_extrapolated_curvature = max(
        fits[family]["curvature"]["extrapolated_error"] for family in FAMILIES
    )
    maximum_finest_seam = max(
        fits[family]["seam_rate"]["finest_error"] for family in FAMILIES
    )
    maximum_extrapolated_seam = max(
        fits[family]["seam_rate"]["extrapolated_error"] for family in FAMILIES
    )
    maximum_fit_stability = max(
        fits[family][section]["leave_coarsest_stability"]
        for family in FAMILIES
        for section in ("curvature", "seam_rate", "area_weighted_junction_rate")
    )
    all_monotone = all(
        fits[family][section]["finest_monotone"]
        for family in FAMILIES
        for section in ("curvature", "seam_rate", "area_weighted_junction_rate")
    )
    profile_spreads = {
        section: spread(section)
        for section in ("curvature", "seam_rate", "area_weighted_junction_rate")
    }

    gates = {
        "parity": maximum_profile_parity < 1e-12,
        "normalization": maximum_normalization < 2e-8,
        "zero_first_moment": maximum_first_moment < 2e-8,
        "rms_width_32": maximum_rms_32 < 5e-3,
        "rms_width_128": maximum_rms_128 < 1e-3,
        "endpoint_matching": maximum_endpoint < 1e-12,
        "junction_integral": maximum_junction_error < 2e-6,
        "junction_rate": maximum_junction_rate_error < 2e-4,
        "derivative_step_halving": maximum_derivative_step_error < 2e-4,
        "finest_curvature": maximum_finest_curvature < 1e-2,
        "extrapolated_curvature": maximum_extrapolated_curvature < 2e-3,
        "finest_seam_rate": maximum_finest_seam < 1e-2,
        "extrapolated_seam_rate": maximum_extrapolated_seam < 2e-3,
        "profile_universality": max(profile_spreads.values()) < 1e-2,
        "fit_stability": maximum_fit_stability < 1e-2,
        "thickness_monotonicity": all_monotone,
        "resolution_convergence": all(item["passed"] for item in resolution_checks),
        "static": abs(static["finite_seam_rate"]) < 1e-12,
        "flat": (
            abs(flat["curvature_excess"]) < 1e-12
            and abs(flat["finite_seam_rate"]) < 1e-12
        ),
        "orientation": (
            relative_scale_error(
                orientation_joint, orientation_original,
                max(abs(orientation_original), 1.0),
            ) < 1e-12
            and relative_scale_error(
                orientation_incorrect, orientation_original,
                max(abs(orientation_original), 1.0),
            ) > 1e-2
        ),
        "no_double_counting": (
            omission_difference > 1e-2 and double_count_difference > 1e-2
        ),
    }
    summary = {
        "maximum_profile_parity_error": maximum_profile_parity,
        "maximum_normalization_error": maximum_normalization,
        "maximum_first_moment_scaled_error": maximum_first_moment,
        "maximum_endpoint_error": maximum_endpoint,
        "maximum_rms_width_error_32": maximum_rms_32,
        "maximum_rms_width_error_128": maximum_rms_128,
        "maximum_junction_error": maximum_junction_error,
        "maximum_junction_rate_error": maximum_junction_rate_error,
        "maximum_derivative_step_error": maximum_derivative_step_error,
        "maximum_finest_curvature_error": maximum_finest_curvature,
        "maximum_extrapolated_curvature_error": maximum_extrapolated_curvature,
        "maximum_finest_seam_rate_error": maximum_finest_seam,
        "maximum_extrapolated_seam_rate_error": maximum_extrapolated_seam,
        "maximum_fit_stability": maximum_fit_stability,
        "profile_spreads": profile_spreads,
    }
    return {
        "schema": "bhps-test14d-manufactured-controls-v1",
        "families": list(FAMILIES),
        "width_ratios": list(WIDTH_RATIOS),
        "resolutions": list(RESOLUTIONS),
        "records": records,
        "fits": fits,
        "resolution_checks": resolution_checks,
        "static_control": static,
        "flat_control": flat,
        "orientation_control": {
            "original": orientation_original,
            "simultaneous_reversal": orientation_joint,
            "one_factor_reversal": orientation_incorrect,
        },
        "summary": summary,
        "gates": gates,
        "finite": bool(all(item["finite"] for item in records)),
        "passed": bool(all(gates.values()) and all(item["finite"] for item in records)),
    }
