"""Basic proper geometry of a donor-capped surface on an evolved slice."""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import RectBivariateSpline

from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice


def capped_surface_geometry(
    position, velocity, z, r, profile, stencil_width=7, prepared=None,
):
    """Return proper cap area and simple shape invariants.

    The spatial metric is ``h_AB dx^A dx^B + r^2 g_perp dOmega_2^2``.
    The returned area is the one-sided cap area.  With a reflection-symmetric
    second bulk copy, the doubled area is twice this value.  The equivalent
    radius is normalized so a flat hemispherical S3 cap has its coordinate
    radius: ``A_cap = pi^2 R_area^3``.
    """
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    theta = np.asarray(profile["theta"], dtype=float)
    rho = np.asarray(profile["rho"], dtype=float)
    slope = np.asarray(profile["slope"], dtype=float)
    if (
        theta.ndim != 1 or len(theta) < 5 or rho.shape != theta.shape
        or slope.shape != theta.shape or np.any(np.diff(theta) <= 0)
    ):
        raise ValueError("invalid capped profile")
    if position.shape != (len(z), len(r), 9) or velocity.shape != position.shape:
        raise ValueError("invalid evolved slice")

    radius = rho * np.sin(theta)
    zcoord = float(z[-1]) - rho * np.cos(theta)
    if (
        np.min(radius) < 0 or np.max(radius) > r[-1]
        or np.min(zcoord) < z[0] or np.max(zcoord) > z[-1]
    ):
        raise ValueError("capped profile leaves the numerical domain")

    prepared = (
        prepare_capped_expansion_slice(
            position, velocity, z, r, stencil_width=stencil_width,
        )
        if prepared is None else prepared
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
    if np.any(transverse <= 0):
        raise RuntimeError("non-positive transverse metric on capped surface")

    tangent = np.stack((
        rho * np.sin(theta) - slope * np.cos(theta),
        rho * np.cos(theta) + slope * np.sin(theta),
    ), axis=1)
    speed = np.sqrt(np.einsum(
        "...a,...ab,...b->...", tangent, base_metric, tangent,
    ))
    sphere_radius = radius * np.sqrt(transverse)
    area_integrand = 4 * math.pi * sphere_radius**2 * speed
    cap_area = float(simpson(area_integrand, x=theta))
    meridional_length = float(simpson(speed, x=theta))
    equivalent_radius = float((cap_area / math.pi**2) ** (1 / 3))
    return {
        "one_sided_cap_area": cap_area,
        "reflection_doubled_area": 2 * cap_area,
        "equivalent_area_radius": equivalent_radius,
        "proper_meridional_length": meridional_length,
        "maximum_sphere_radius": float(np.max(sphere_radius)),
        "rho_axis": float(rho[0]),
        "rho_brane": float(rho[-1]),
        "endpoint_shape_ratio": float(rho[-1] / rho[0]),
        "meridional_shape_ratio": float(
            meridional_length / ((math.pi / 2) * equivalent_radius)
        ),
        "minimum_curve_speed": float(np.min(speed)),
        "minimum_transverse_metric": float(np.min(transverse)),
        "finite": bool(
            cap_area > 0 and np.all(np.isfinite(area_integrand))
            and np.isfinite(meridional_length)
        ),
        "area_convention": "one-sided donor cap; doubled value assumes reflection symmetry",
    }
