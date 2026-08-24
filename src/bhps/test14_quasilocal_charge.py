"""Five-dimensional generalized-Hawking charge for reflected SO(3) caps.

The physical computational domain contains one cap ending on a brane.  The
closed three-leaf used here is its Z2 reflection double.  The reflected
induced metric can be only Lipschitz at the brane, so the intrinsic scalar
curvature includes the distributional seam completion derived in note 95.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad, simpson
from scipy.interpolate import RectBivariateSpline

from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice


OMEGA3 = 2.0 * math.pi**2


def relative_difference(left, right):
    """Symmetric scale-relative difference used by the sealed gates."""
    return float(abs(left - right) / max(abs(left), abs(right), 1e-300))


def charge_from_integrals(
    doubled_area, intrinsic_scalar_integral, mean_curvature_squared_integral=0.0,
    cosmological_constant=-6.0,
):
    """Return ``kappa5^2 E`` in the convention ``kappa5^2=8 pi G5``.

    ``mean_curvature_squared_integral`` is ``integral K_perp^2 dA``.  It is
    zero on an exact marginal leaf but is retained for analytic controls.
    """
    area = float(doubled_area)
    intrinsic = float(intrinsic_scalar_integral)
    mean_squared = float(mean_curvature_squared_integral)
    cosmological = float(cosmological_constant)
    if not area > 0.0:
        raise ValueError("doubled area must be positive")
    area_radius = float((area / OMEGA3) ** (1.0 / 3.0))
    raw_integral = intrinsic - (2.0 / 3.0) * mean_squared
    raw_charge = 0.25 * area_radius * raw_integral
    background_charge = -0.25 * area_radius * cosmological * area
    ads_charge = raw_charge + background_charge
    round_charge = 3.0 * math.pi**2 * area_radius**2 * (
        1.0 + area_radius**2
    )
    shape_factor = intrinsic / (12.0 * math.pi**2 * area_radius)
    return {
        "reflection_doubled_area": area,
        "equivalent_area_radius": area_radius,
        "intrinsic_scalar_curvature_integral": intrinsic,
        "mean_curvature_squared_integral": mean_squared,
        "cosmological_constant": cosmological,
        "raw_generalized_hawking_charge_kappa5_squared_E": raw_charge,
        "ads_background_charge": background_charge,
        "generalized_hawking_ads_charge_kappa5_squared_E": ads_charge,
        "round_area_proxy_charge_kappa5_squared_E": round_charge,
        "intrinsic_curvature_shape_factor": shape_factor,
        "relative_nonround_charge_correction": (
            (ads_charge - round_charge) / max(abs(round_charge), 1e-300)
        ),
    }


def reflected_cap_charge(
    position, velocity, z, r, profile, stencil_width=7, prepared=None,
    cosmological_constant=-6.0,
):
    """Evaluate the seam-completed marginal charge of one reflected cap."""
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    theta = np.asarray(profile["theta"], dtype=float)
    rho = np.asarray(profile["rho"], dtype=float)
    slope = np.asarray(profile["slope"], dtype=float)
    if (
        theta.ndim != 1 or len(theta) < 5 or rho.shape != theta.shape
        or slope.shape != theta.shape or np.any(np.diff(theta) <= 0.0)
    ):
        raise ValueError("invalid capped profile")
    if position.shape != (len(z), len(r), 9) or velocity.shape != position.shape:
        raise ValueError("invalid evolved slice")

    zcoord = float(z[-1]) - rho * np.cos(theta)
    radius = rho * np.sin(theta)
    if (
        np.min(radius) < 0.0 or np.max(radius) > r[-1]
        or np.min(zcoord) < z[0] or np.max(zcoord) > z[-1]
    ):
        raise ValueError("capped profile leaves the numerical domain")

    prepared = (
        prepare_capped_expansion_slice(
            position, velocity, z, r, stencil_width=stencil_width,
        ) if prepared is None else prepared
    )
    if not (
        np.array_equal(prepared.z, z) and np.array_equal(prepared.r, r)
        and prepared.stencil_width == int(stencil_width)
    ):
        raise ValueError("prepared capped slice uses a different grid")

    base_metric = np.empty((len(theta), 2, 2))
    for left in range(2):
        for right in range(2):
            base_metric[:, left, right] = prepared.sample(
                ("base_metric", left, right), zcoord, radius,
            )
    transverse_spline = RectBivariateSpline(
        z, r, position[:, :, 3], kx=min(3, len(z) - 1),
        ky=min(3, len(r) - 1), s=0,
    )
    transverse = transverse_spline.ev(zcoord, radius)
    if np.any(transverse <= 0.0):
        raise RuntimeError("non-positive transverse metric on capped surface")

    tangent = np.stack((
        rho * np.sin(theta) - slope * np.cos(theta),
        rho * np.cos(theta) + slope * np.sin(theta),
    ), axis=1)
    speed = np.sqrt(np.einsum(
        "...a,...ab,...b->...", tangent, base_metric, tangent,
    ))
    if np.any(speed <= 0.0):
        raise RuntimeError("non-positive proper meridional speed")

    transverse_z = transverse_spline.ev(zcoord, radius, dx=1, dy=0)
    transverse_r = transverse_spline.ev(zcoord, radius, dx=0, dy=1)
    root_transverse = np.sqrt(transverse)
    sphere_radius = radius * root_transverse
    transverse_theta = transverse_z * tangent[:, 0] + transverse_r * tangent[:, 1]
    sphere_radius_theta = (
        tangent[:, 1] * root_transverse
        + 0.5 * radius * transverse_theta / root_transverse
    )
    sphere_radius_slope = sphere_radius_theta / speed

    one_sided_area = float(simpson(
        4.0 * math.pi * sphere_radius**2 * speed, x=theta,
    ))
    doubled_area = 2.0 * one_sided_area
    intrinsic_completed = float(simpson(
        16.0 * math.pi * (1.0 + sphere_radius_slope**2) * speed,
        x=theta,
    ))
    seam_integral = float(
        32.0 * math.pi * sphere_radius[-1] * sphere_radius_slope[-1]
    )
    doubled_smooth_bulk = intrinsic_completed - seam_integral
    charge = charge_from_integrals(
        doubled_area, intrinsic_completed,
        mean_curvature_squared_integral=0.0,
        cosmological_constant=cosmological_constant,
    )
    invalid_bulk_only_charge = charge_from_integrals(
        doubled_area, doubled_smooth_bulk,
        mean_curvature_squared_integral=0.0,
        cosmological_constant=cosmological_constant,
    )
    charge.update({
        "one_sided_cap_area": one_sided_area,
        "proper_meridional_length": float(simpson(speed, x=theta)),
        "maximum_sphere_radius": float(np.max(sphere_radius)),
        "axis_sphere_radius_slope": float(sphere_radius_slope[0]),
        "axis_regularity_defect": float(abs(sphere_radius_slope[0] - 1.0)),
        "brane_sphere_radius": float(sphere_radius[-1]),
        "brane_oriented_sphere_radius_slope": float(sphere_radius_slope[-1]),
        "distributional_seam_curvature_integral": seam_integral,
        "doubled_smooth_bulk_curvature_integral": doubled_smooth_bulk,
        "seam_fraction_of_completed_intrinsic_integral": float(
            seam_integral / max(abs(intrinsic_completed), 1e-300)
        ),
        "invalid_bulk_only_ads_charge": invalid_bulk_only_charge[
            "generalized_hawking_ads_charge_kappa5_squared_E"
        ],
        "invalid_omit_seam_charge_relative_difference": relative_difference(
            charge["generalized_hawking_ads_charge_kappa5_squared_E"],
            invalid_bulk_only_charge[
                "generalized_hawking_ads_charge_kappa5_squared_E"
            ],
        ),
        "minimum_curve_speed": float(np.min(speed)),
        "minimum_transverse_metric": float(np.min(transverse)),
        "finite": bool(
            np.all(np.isfinite(speed))
            and np.all(np.isfinite(sphere_radius_slope))
            and np.all(np.isfinite(list(charge.values())))
        ),
        "surface_type": "marginal leaf; K_perp^2 set to its exact zero limit",
        "closure": "Z2 reflection double with distributional brane-seam completion",
    })
    return charge


def analytic_controls():
    """Return the three prospectively sealed analytic/manufactured controls."""
    flat = []
    ads = []
    for radius in (0.5, 1.0, 2.0):
        area = OMEGA3 * radius**3
        intrinsic = 12.0 * math.pi**2 * radius
        mean_squared = 18.0 * math.pi**2 * radius
        flat_value = charge_from_integrals(
            area, intrinsic, mean_squared, cosmological_constant=0.0,
        )
        flat.append({
            "radius": radius,
            "computed_raw_charge": flat_value[
                "raw_generalized_hawking_charge_kappa5_squared_E"
            ],
            "expected_raw_charge": 0.0,
            "absolute_error": abs(flat_value[
                "raw_generalized_hawking_charge_kappa5_squared_E"
            ]),
        })
        ads_value = charge_from_integrals(
            area, intrinsic, 0.0, cosmological_constant=-6.0,
        )
        expected = 3.0 * math.pi**2 * radius**2 * (1.0 + radius**2)
        ads.append({
            "radius": radius,
            "computed_charge": ads_value[
                "generalized_hawking_ads_charge_kappa5_squared_E"
            ],
            "expected_charge": expected,
            "relative_error": relative_difference(
                ads_value["generalized_hawking_ads_charge_kappa5_squared_E"],
                expected,
            ),
        })

    length = 1.0
    epsilon = 1e-4
    normalization = math.sqrt(length**2 + epsilon**2)

    def smoothed_integrand(value):
        root = math.sqrt(value**2 + epsilon**2)
        sphere_radius = normalization - root
        first = -value / root
        second = -(epsilon**2) / root**3
        return 4.0 * math.pi * (
            -4.0 * sphere_radius * second + 2.0 * (1.0 - first**2)
        )

    smoothed = 2.0 * quad(
        smoothed_integrand, 0.0, length,
        epsabs=1e-11, epsrel=1e-11, limit=1000,
    )[0]
    distributional = 32.0 * math.pi * length
    omitted = 0.0
    seam = {
        "cap": "W(s)=s on 0<=s<=1 reflected at s=1",
        "smoothing_epsilon": epsilon,
        "smoothed_curvature_integral": smoothed,
        "distributional_completed_integral": distributional,
        "relative_error": relative_difference(smoothed, distributional),
        "bulk_only_integral": omitted,
        "bulk_only_relative_error": relative_difference(omitted, distributional),
    }
    acceptance = {
        "flat_round_raw_charge_below_1e_10": bool(all(
            item["absolute_error"] < 1e-10 for item in flat
        )),
        "schwarzschild_ads_round_relative_error_below_1e_10": bool(all(
            item["relative_error"] < 1e-10 for item in ads
        )),
        "smoothed_seam_relative_error_below_2e_4": bool(
            seam["relative_error"] < 2e-4
        ),
        "omitted_seam_fails_by_resolved_amount": bool(
            seam["bulk_only_relative_error"] > 0.1
        ),
    }
    return {
        "flat_round_s3": flat,
        "schwarzschild_ads5_horizon": ads,
        "reflected_seam": seam,
        "acceptance": acceptance,
        "passed": bool(all(acceptance.values())),
    }

