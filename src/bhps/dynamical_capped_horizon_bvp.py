"""Independent local-BVP detector for evolved donor-capped marginal surfaces.

This module deliberately does not use the cosine representation or the nodal
least-squares residual from :mod:`bhps.dynamical_capped_horizon`.  It writes
the outgoing expansion as a local second-order ODE for the polar radius
``rho(theta)`` and solves the two-point problem ``rho'(0)=rho'(pi/2)=0`` with
adaptive BVP collocation.

Only the already audited ADM slice reduction and its metric interpolation are
shared with the primary detector.  Surface curvature is evaluated here from
the covariant acceleration of the meridional curve.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_bvp

from bhps.dynamical_capped_horizon import (
    capped_outgoing_expansion,
    prepare_capped_expansion_slice,
)


def _sample_geometry(prepared, theta, rho, slope):
    """Sample evolved-slice tensors and construct tangent/normal geometry."""
    theta = np.asarray(theta, dtype=float)
    rho = np.asarray(rho, dtype=float)
    slope = np.asarray(slope, dtype=float)
    sine = np.sin(theta)
    cosine = np.cos(theta)
    radius = rho * sine
    zcoord = prepared.z[-1] - rho * cosine

    # The BVP iteration can briefly step beyond a boundary.  Clipping only the
    # interpolation coordinates keeps its residual finite; the final solution
    # is separately required to lie strictly inside the physical domain.
    sample_r = np.clip(radius, prepared.r[0], prepared.r[-1])
    sample_z = np.clip(zcoord, prepared.z[0], prepared.z[-1])
    count = len(theta)
    metric = np.empty((count, 2, 2))
    inverse = np.empty_like(metric)
    connection = np.empty((count, 2, 2, 2))
    log_gradient = np.empty((count, 2))
    extrinsic = np.empty_like(metric)
    for left in range(2):
        log_gradient[:, left] = prepared.sample(
            ("log_transverse_scale_gradient", left), sample_z, sample_r,
        )
        for right in range(2):
            metric[:, left, right] = prepared.sample(
                ("base_metric", left, right), sample_z, sample_r,
            )
            inverse[:, left, right] = prepared.sample(
                ("base_inverse", left, right), sample_z, sample_r,
            )
            extrinsic[:, left, right] = prepared.sample(
                ("extrinsic_base", left, right), sample_z, sample_r,
            )
            for upper in range(2):
                connection[:, upper, left, right] = prepared.sample(
                    ("base_connection", upper, left, right), sample_z, sample_r,
                )
    sphere_extrinsic = prepared.sample(
        ("extrinsic_sphere_eigenvalue",), sample_z, sample_r,
    )
    # The coordinate-radius contribution is analytic.  A strictly positive
    # floor is used only during nonlinear iteration; admitted profiles never
    # touch it.
    safe_radius = np.maximum(np.abs(radius), 1e-12)
    log_gradient[:, 1] += 1.0 / safe_radius

    tangent_coordinate = np.stack((
        rho * sine - slope * cosine,
        rho * cosine + slope * sine,
    ), axis=1)
    speed_squared = np.einsum(
        "...a,...ab,...b->...", tangent_coordinate, metric, tangent_coordinate,
    )
    speed_squared = np.maximum(speed_squared, 1e-24)
    speed = np.sqrt(speed_squared)
    tangent = tangent_coordinate / speed[:, None]
    normal_covector = np.stack(
        (-tangent_coordinate[:, 1], tangent_coordinate[:, 0]), axis=1,
    )
    normal_norm_squared = np.einsum(
        "...a,...ab,...b->...", normal_covector, inverse, normal_covector,
    )
    normal_covector /= np.sqrt(np.maximum(normal_norm_squared, 1e-24))[:, None]
    normal = np.einsum("...ab,...b->...a", inverse, normal_covector)
    return {
        "radius": radius,
        "zcoord": zcoord,
        "connection": connection,
        "log_gradient": log_gradient,
        "extrinsic": extrinsic,
        "sphere_extrinsic": sphere_extrinsic,
        "tangent": tangent,
        "normal": normal,
        "normal_covector": normal_covector,
        "speed_squared": speed_squared,
    }


def local_outgoing_expansion(prepared, theta, rho, slope, second):
    """Evaluate ``theta_+`` from local ``rho``, ``rho'``, and ``rho''``.

    Unlike the primary evaluator, this expression takes the curve's second
    derivative explicitly and never differentiates a sampled normal vector.
    """
    theta = np.asarray(theta, dtype=float)
    rho = np.asarray(rho, dtype=float)
    slope = np.asarray(slope, dtype=float)
    second = np.asarray(second, dtype=float)
    geometry = _sample_geometry(prepared, theta, rho, slope)
    sine = np.sin(theta)
    cosine = np.cos(theta)
    acceleration = np.stack((
        rho * cosine + 2.0 * slope * sine - second * cosine,
        -rho * sine + 2.0 * slope * cosine + second * sine,
    ), axis=1)
    curve = -np.einsum(
        "...a,...a->...", geometry["normal_covector"], acceleration,
    ) / geometry["speed_squared"]
    curve -= np.einsum(
        "...a,...b,...cab,...c->...",
        geometry["tangent"], geometry["tangent"], geometry["connection"],
        geometry["normal_covector"],
    )
    sphere = 2.0 * np.einsum(
        "...a,...a->...", geometry["normal"], geometry["log_gradient"],
    )
    tangent_extrinsic = np.einsum(
        "...a,...ab,...b->...",
        geometry["tangent"], geometry["extrinsic"], geometry["tangent"],
    )
    correction = -tangent_extrinsic - 2.0 * geometry["sphere_extrinsic"]
    return curve + sphere + correction


def dynamical_rho_second(prepared, theta, rho, slope):
    """Solve the affine local equation ``theta_+(rho'')=0`` for ``rho''``."""
    zero = np.zeros_like(np.asarray(rho, dtype=float))
    offset = local_outgoing_expansion(prepared, theta, rho, slope, zero)
    unit = local_outgoing_expansion(prepared, theta, rho, slope, np.ones_like(zero))
    coefficient = unit - offset
    return np.divide(
        -offset, coefficient,
        out=np.full_like(offset, np.nan), where=np.abs(coefficient) > 1e-12,
    )


def _boundary_conditions(left, right):
    return np.array((left[1], right[1]))


def solve_dynamical_capped_surface_bvp(
    position, velocity, z, r, initial, tolerance=2e-5, nodes=121,
    maximum_nodes=6000, stencil_width=7, dense_nodes=501, prepared=None,
):
    """Solve the evolved marginal-cap equation as a local two-point BVP."""
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    prepared = (
        prepare_capped_expansion_slice(position, velocity, z, r, stencil_width)
        if prepared is None else prepared
    )
    theta = np.linspace(1e-4, np.pi / 2.0, int(nodes))
    if np.isscalar(initial):
        state = np.vstack((np.full_like(theta, float(initial)), np.zeros_like(theta)))
    else:
        source_theta = np.asarray(initial["theta"], dtype=float)
        state = np.vstack((
            np.interp(theta, source_theta, np.asarray(initial["rho"], dtype=float)),
            np.interp(
                theta, source_theta,
                np.asarray(initial.get("slope", np.zeros_like(source_theta)), dtype=float),
            ),
        ))

    def equation(angle, values):
        return np.vstack((
            values[1],
            dynamical_rho_second(prepared, angle, values[0], values[1]),
        ))

    solved = solve_bvp(
        equation, _boundary_conditions, theta, state, tol=float(tolerance),
        max_nodes=int(maximum_nodes), verbose=0,
    )
    dense_theta = np.linspace(theta[0], theta[-1], int(dense_nodes))
    values = solved.sol(dense_theta)
    derivatives = solved.sol(dense_theta, 1)
    rho = values[0]
    slope = values[1]
    second = derivatives[1]
    equation_values = equation(dense_theta, values)
    local_expansion = local_outgoing_expansion(
        prepared, dense_theta, rho, slope, second,
    )
    radius = rho * np.sin(dense_theta)
    zcoord = z[-1] - rho * np.cos(dense_theta)
    dr = float(np.min(np.diff(r)))
    dz = float(np.min(np.diff(z)))
    interior = (radius >= 2.0 * dr) & (zcoord <= z[-1] - 2.0 * dz)
    in_domain = bool(
        np.min(rho) > 2.0 * r[1]
        and np.min(zcoord) > z[0] + 2.0 * dz
        and np.max(radius) < 0.9 * r[-1]
    )
    crosscheck = None
    try:
        checked = capped_outgoing_expansion(
            position, velocity, z, r,
            {"theta": dense_theta, "rho": rho, "slope": slope},
            stencil_width=stencil_width, prepared=prepared,
        )
        crosscheck = {
            "two_cell_interior_maximum": checked[
                "two_cell_interior_maximum_absolute"
            ],
            "two_cell_interior_normalized_maximum": checked[
                "two_cell_interior_maximum_normalized"
            ],
        }
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        crosscheck = {"error": f"{type(error).__name__}: {error}"}
    local_maximum = float(
        np.max(np.abs(local_expansion[interior])) if np.any(interior) else np.inf
    )
    boundary_error = float(max(abs(slope[0]), abs(slope[-1])))
    ode_defect = float(np.max(np.abs(derivatives - equation_values)))
    return {
        "converged": bool(
            solved.success and in_domain and np.any(interior)
            and local_maximum < max(1e-8, 10.0 * float(tolerance))
            and boundary_error < max(1e-8, 10.0 * float(tolerance))
        ),
        "solver_success": bool(solved.success),
        "message": str(solved.message),
        "in_domain": in_domain,
        "iterations": int(solved.niter),
        "mesh_nodes_used": int(len(solved.x)),
        "theta": dense_theta,
        "rho": rho,
        "slope": slope,
        "rho_axis": float(rho[0]),
        "rho_brane": float(rho[-1]),
        "rho_min": float(np.min(rho)),
        "rho_max": float(np.max(rho)),
        "boundary_slope_error": boundary_error,
        "local_expansion_interior_maximum": local_maximum,
        "local_expansion_full_maximum": float(np.max(np.abs(local_expansion))),
        "ode_defect_maximum": ode_defect,
        "primary_evaluator_crosscheck": crosscheck,
        "interior_point_count": int(np.count_nonzero(interior)),
    }
