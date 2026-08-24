"""Direct compact-wall Israel residual and preservation diagnostics.

The nonlinear SO(3) evolution uses wall-adapted coordinates.  Once the two
normal--tangential Dirichlet rows vanish, its four independent tangential
metric Robin rows are precisely the coordinate components of

    J_ab = K_ab - c(Phi) h_ab.

This module applies the required orientation and ``2*sqrt(g_zz)`` conversion
instead of treating the raw Robin rows as ``J_ab``.  It also differentiates
the *semi-discrete boundary functional* analytically:

    D_X J[q] := D J[q] . q_t = D J[q] . velocity.

No time-history difference and no acceleration snapshot is used.  The latter
would first enter a test of ``D_X**2 J``.  Scalar, reflecting-scalar, and gauge
rows are returned separately and are never folded into the Israel tensor.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import (
    CubicHermiteSpline,
    CubicSpline,
    PchipInterpolator,
    RectBivariateSpline,
)

from bhps.gw_slice_high_order_solver import derivative_matrix


COMPONENTS = ("tt", "rr", "sphere", "tr")
WALLS = ("lower", "upper")


def _dense_derivative(points, order=1, stencil_width=7):
    matrix = derivative_matrix(np.asarray(points, dtype=float), order, stencil_width)
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _validate_state(position, velocity, z, r):
    q = np.asarray(position, dtype=float)
    v = np.asarray(velocity, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    expected = (len(z), len(r), 9)
    if q.shape != expected or v.shape != expected:
        raise ValueError("position and velocity must have shape (z,r,9)")
    if not all(np.all(np.isfinite(item)) for item in (q, v, z, r)):
        raise ValueError("state and grids must be finite")
    if len(z) < 7 or len(r) < 7 or np.any(np.diff(z) <= 0) or np.any(np.diff(r) <= 0):
        raise ValueError("grids must be strictly increasing and support seven-point stencils")
    if not np.isclose(r[0], 0.0):
        raise ValueError("the regular radial grid must start at the axis")
    if np.any(q[:, :, 6] <= 0.0):
        raise ValueError("g_zz must remain positive")
    return q, v, z, r


def wall_source_coefficients(phi, background, wall):
    """Return coordinate Robin beta and outward Israel coefficient ``c``.

    ``beta`` is the coefficient in ``d_z h + 2 beta sqrt(g_zz) h=0``.
    The outward normals are ``-d_z`` at the lower wall and ``+d_z`` at the
    upper wall, hence ``c=-orientation*beta``.
    """
    phi = np.asarray(phi, dtype=float)
    if wall not in WALLS:
        raise ValueError("wall must be 'lower' or 'upper'")
    upper = wall == "upper"
    orientation = 1.0 if upper else -1.0
    required = (
        "wall_stiffness", "v0", "v1", "beta_a", "beta_b",
        "wall_potential_a", "wall_potential_b",
    )
    missing = [name for name in required if name not in background]
    if missing:
        raise KeyError(f"background lacks wall coefficients: {missing}")
    scalars = np.asarray([float(background[name]) for name in required])
    if not np.all(np.isfinite(scalars)) or not np.all(np.isfinite(phi)):
        raise ValueError("wall fields and background coefficients must be finite")
    gamma = float(background["wall_stiffness"])
    if gamma < 0.0:
        raise ValueError("wall stiffness must be nonnegative")
    target = float(background["v1"] if upper else background["v0"])
    bare_beta = float(background["beta_b"] if upper else background["beta_a"])
    reference_potential = float(
        background["wall_potential_b"] if upper else background["wall_potential_a"]
    )
    delta = phi - target
    potential = 0.5 * gamma * delta**2
    if upper:
        beta = bare_beta - (potential - reference_potential) / 6.0
        beta_phi = -gamma * delta / 6.0
    else:
        beta = bare_beta + (potential - reference_potential) / 6.0
        beta_phi = gamma * delta / 6.0
    return {
        "orientation": orientation,
        "target": target,
        "potential": potential,
        "beta": beta,
        "beta_phi": beta_phi,
        "c": -orientation * beta,
        "c_phi": -orientation * beta_phi,
    }


def _component_fields(state, r):
    radius2 = np.asarray(r, dtype=float)[None, :] ** 2
    radius = np.asarray(r, dtype=float)[None, :]
    return {
        "tt": state[:, :, 2],
        "rr": state[:, :, 3] + radius2 * state[:, :, 4],
        "sphere": state[:, :, 3],
        "tr": radius * state[:, :, 5],
    }


def _assemble_so3_tensor(components):
    sample = np.asarray(components["tt"])
    result = np.zeros((len(sample), 4, 4), dtype=float)
    result[:, 0, 0] = components["tt"]
    result[:, 1, 1] = components["rr"]
    result[:, 2, 2] = components["sphere"]
    result[:, 3, 3] = components["sphere"]
    result[:, 0, 1] = result[:, 1, 0] = components["tr"]
    return result


def _orthonormal_frames(metric):
    """Construct Eulerian orthonormal frames for Lorentzian wall metrics."""
    metric = np.asarray(metric, dtype=float)
    frames = np.empty_like(metric)
    eta = np.diag((-1.0, 1.0, 1.0, 1.0))
    defects = np.empty(len(metric))
    for index, local in enumerate(metric):
        inverse = np.linalg.inv(local)
        if inverse[0, 0] >= 0.0:
            raise ValueError("wall metric is not Lorentzian in the archived slicing")
        lapse = 1.0 / math.sqrt(-inverse[0, 0])
        time_covector = np.array((-lapse, 0.0, 0.0, 0.0))
        time_vector = inverse @ time_covector
        spatial = local[1:, 1:]
        cholesky = np.linalg.cholesky(spatial)
        spatial_frame = np.linalg.inv(cholesky.T)
        frame = np.zeros((4, 4))
        frame[:, 0] = time_vector
        frame[1:, 1:] = spatial_frame
        frames[index] = frame
        defects[index] = np.max(np.abs(frame.T @ local @ frame - eta))
    return frames, defects


def _tensor_norms(metric, junction, tangent, c):
    frames, frame_defect = _orthonormal_frames(metric)
    j_hat = np.einsum("nai,nab,nbj->nij", frames, junction, frames)
    dj_hat = np.einsum("nai,nab,nbj->nij", frames, tangent, frames)
    eta = np.diag((-1.0, 1.0, 1.0, 1.0))
    k_hat = j_hat + np.asarray(c)[:, None, None] * eta[None, :, :]
    source_hat = np.asarray(c)[:, None, None] * eta[None, :, :]
    j_frobenius = np.linalg.norm(j_hat, axis=(1, 2))
    k_frobenius = np.linalg.norm(k_hat, axis=(1, 2))
    source_frobenius = np.linalg.norm(source_hat, axis=(1, 2))
    relative = j_frobenius / np.maximum(
        np.maximum(k_frobenius, source_frobenius), 1e-300,
    )
    mixed_radius = np.empty(len(metric))
    anisotropy_radius = np.empty(len(metric))
    trace_absolute = np.empty(len(metric))
    tracefree_frobenius = np.empty(len(metric))
    tangent_trace_absolute = np.empty(len(metric))
    tangent_tracefree_frobenius = np.empty(len(metric))
    maximum_imaginary = 0.0
    for index, (local_metric, local_j, local_dj) in enumerate(
        zip(metric, junction, tangent)
    ):
        mixed = np.linalg.solve(local_metric, local_j)
        values = np.linalg.eigvals(mixed)
        maximum_imaginary = max(maximum_imaginary, float(np.max(np.abs(values.imag))))
        mixed_radius[index] = np.max(np.abs(values))
        trace = float(np.trace(mixed))
        trace_mean = trace / 4.0
        trace_absolute[index] = abs(trace)
        tracefree = local_j - trace_mean * local_metric
        local_frame = frames[index]
        tracefree_hat = local_frame.T @ tracefree @ local_frame
        tracefree_frobenius[index] = np.linalg.norm(tracefree_hat)
        anisotropy_radius[index] = np.max(np.abs(np.linalg.eigvals(
            mixed - trace_mean * np.eye(4)
        )))
        tangent_mixed = np.linalg.solve(local_metric, local_dj)
        tangent_trace = float(np.trace(tangent_mixed))
        tangent_trace_absolute[index] = abs(tangent_trace)
        tangent_tracefree = local_dj - tangent_trace / 4.0 * local_metric
        tangent_tracefree_hat = local_frame.T @ tangent_tracefree @ local_frame
        tangent_tracefree_frobenius[index] = np.linalg.norm(tangent_tracefree_hat)
    return {
        "frame_defect": frame_defect,
        "J_orthonormal_frobenius": j_frobenius,
        "J_orthonormal_max_component": np.max(np.abs(j_hat), axis=(1, 2)),
        "J_relative_orthonormal": relative,
        "DXJ_orthonormal_frobenius": np.linalg.norm(dj_hat, axis=(1, 2)),
        "DXJ_orthonormal_max_component": np.max(np.abs(dj_hat), axis=(1, 2)),
        "J_mixed_spectral_radius": mixed_radius,
        "J_mixed_anisotropy_radius": anisotropy_radius,
        "J_mixed_trace_absolute": trace_absolute,
        "J_tracefree_orthonormal_frobenius": tracefree_frobenius,
        "DXJ_mixed_trace_absolute": tangent_trace_absolute,
        "DXJ_tracefree_orthonormal_frobenius": tangent_tracefree_frobenius,
        "mixed_eigenvalue_maximum_imaginary": maximum_imaginary,
    }


def wall_junction_rows(
    position, velocity, z, r, background, wall, stencil_width=7,
):
    """Evaluate the native ``J_ab`` rows and ``D J[q].velocity``.

    The returned components reconstruct the full SO(3)-invariant tangential
    tensor in the basis ``(t,r,Omega_1,Omega_2)``.  ``DXJ`` is the exact
    derivative of the same semi-discrete row, including the Phi-dependence of
    the wall potential and the ``g_zz`` normalization.
    """
    q, v, z, r = _validate_state(position, velocity, z, r)
    index = -1 if wall == "upper" else 0
    dz = _dense_derivative(z, 1, stencil_width)
    dr = _dense_derivative(r, 1, stencil_width)
    q_fields = _component_fields(q, r)
    v_fields = _component_fields(v, r)
    source = wall_source_coefficients(q[index, :, 7], background, wall)
    orientation = source["orientation"]
    beta = source["beta"]
    beta_t = source["beta_phi"] * v[index, :, 7]
    A = np.sqrt(q[index, :, 6])
    A_t = v[index, :, 6] / (2.0 * A)
    components = {}
    for name in COMPONENTS:
        field = q_fields[name]
        field_t = v_fields[name]
        value = field[index]
        value_t = field_t[index]
        value_z = (dz @ field)[index]
        value_zt = (dz @ field_t)[index]
        source_term = 2.0 * beta * A * value
        robin = value_z + source_term
        source_term_t = 2.0 * (
            beta_t * A * value + beta * A_t * value + beta * A * value_t
        )
        robin_t = value_zt + source_term_t
        junction = orientation * robin / (2.0 * A)
        tangent = orientation / (2.0 * A) * (
            robin_t - robin * A_t / A
        )
        row_scale = np.maximum(1.0, np.abs(value_z) + np.abs(source_term))
        record = {
            "metric": value,
            "metric_t": value_t,
            "normal_derivative": value_z,
            "robin_source": source_term,
            "robin_residual": robin,
            "robin_normalized": np.abs(robin) / row_scale,
            "J": junction,
            "DXJ": tangent,
            "J_r": dr @ junction,
        }
        if name != "tr":
            coefficient = junction / value
            coefficient_t = (tangent * value - junction * value_t) / value**2
            record.update({
                "mixed_coefficient": coefficient,
                "DX_mixed_coefficient": coefficient_t,
                "mixed_coefficient_r": dr @ coefficient,
            })
        components[name] = record

    metric_tensor = _assemble_so3_tensor({
        name: components[name]["metric"] for name in COMPONENTS
    })
    junction_tensor = _assemble_so3_tensor({
        name: components[name]["J"] for name in COMPONENTS
    })
    tangent_tensor = _assemble_so3_tensor({
        name: components[name]["DXJ"] for name in COMPONENTS
    })
    tensor_norms = _tensor_norms(
        metric_tensor, junction_tensor, tangent_tensor, source["c"],
    )

    sign = orientation
    gamma = float(background["wall_stiffness"])
    phi = q[:, :, 7]
    phi_t = v[:, :, 7]
    delta = phi[index] - source["target"]
    scalar_robin = (dz @ phi)[index] + sign * 0.5 * gamma * delta * A
    scalar_tangent = (
        (dz @ phi_t)[index]
        + sign * 0.5 * gamma * (phi_t[index] * A + delta * A_t)
    )
    chi_robin = (dz @ q[:, :, 8])[index]
    chi_tangent = (dz @ v[:, :, 8])[index]
    gauge = {
        "h_z0": q[index, :, 0],
        "h_zr": r * q[index, :, 1],
        "h_z0_t": v[index, :, 0],
        "h_zr_t": r * v[index, :, 1],
    }
    return {
        "wall": wall,
        "orientation": orientation,
        "components": components,
        "metric_tensor": metric_tensor,
        "J_tensor": junction_tensor,
        "DXJ_tensor": tangent_tensor,
        "tensor_norms": tensor_norms,
        "source": source,
        "separate_rows": {
            "Phi_robin": scalar_robin,
            "DX_Phi_robin": scalar_tangent,
            "chi_neumann": chi_robin,
            "DX_chi_neumann": chi_tangent,
            "gauge": gauge,
        },
        "wall_adapted_gauge_maximum": float(max(
            np.max(np.abs(value)) for value in gauge.values()
        )),
        "finite": bool(
            np.all(np.isfinite(junction_tensor))
            and np.all(np.isfinite(tangent_tensor))
            and np.all(np.isfinite(tensor_norms["J_orthonormal_frobenius"]))
        ),
    }


def radial_zones(r, buffer_points=7):
    """Return disjoint axis, interior, and outer-corner index arrays."""
    r = np.asarray(r, dtype=float)
    count = int(buffer_points)
    if count < 1 or len(r) <= 2 * count:
        raise ValueError("radial grid is too short for the requested zones")
    return {
        "axis": np.arange(0, count),
        "interior": np.arange(count, len(r) - count),
        "outer_corner": np.arange(len(r) - count, len(r)),
    }


def _maximum_record(values, indices, r):
    values = np.asarray(values, dtype=float)
    local = int(indices[np.argmax(np.abs(values[indices]))])
    return {
        "maximum_absolute": float(abs(values[local])),
        "signed_value": float(values[local]),
        "radial_index": local,
        "radius": float(r[local]),
    }


def _nodal_widths(coordinate):
    coordinate = np.asarray(coordinate, dtype=float)
    edges = np.empty(len(coordinate) + 1)
    edges[1:-1] = 0.5 * (coordinate[:-1] + coordinate[1:])
    edges[0] = coordinate[0]
    edges[-1] = coordinate[-1]
    return np.diff(edges)


def _weighted_quantile(values, weights, quantile):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    total = float(np.sum(weights))
    if total <= 0.0:
        return float(np.max(values))
    cumulative = np.cumsum(weights)
    return float(values[min(np.searchsorted(cumulative, quantile * total), len(values) - 1)])


def _proper_profile_statistics(values, indices, r, metric_tensor):
    """Proper-wall L2/RMS/q95/Linf using ``4 pi R^2 dl`` weights."""
    values = np.abs(np.asarray(values, dtype=float)[indices])
    local_r = np.asarray(r, dtype=float)[indices]
    metric = np.asarray(metric_tensor, dtype=float)[indices]
    radial_metric = metric[:, 1, 1]
    sphere_metric = metric[:, 2, 2]
    if np.any(radial_metric <= 0.0) or np.any(sphere_metric <= 0.0):
        raise ValueError("wall spatial metric must be positive")
    density = 4.0 * math.pi * local_r**2 * sphere_metric * np.sqrt(radial_metric)
    weights = density * _nodal_widths(local_r)
    integral = float(np.sum(weights * values**2))
    volume = float(np.sum(weights))
    return {
        "proper_L2": math.sqrt(max(integral, 0.0)),
        "proper_RMS": math.sqrt(max(integral, 0.0) / max(volume, 1e-300)),
        "proper_weighted_q95": _weighted_quantile(values, weights, 0.95),
        "Linf": float(np.max(values)),
        "proper_measure": volume,
    }


def _proper_distance_from_outer_wall(r, metric_tensor):
    r = np.asarray(r, dtype=float)
    radial_speed = np.sqrt(np.asarray(metric_tensor)[:, 1, 1])
    increments = 0.5 * (radial_speed[:-1] + radial_speed[1:]) * np.diff(r)
    result = np.zeros(len(r))
    result[:-1] = np.cumsum(increments[::-1])[::-1]
    return result


def summarize_wall(record, r, buffer_points=7):
    """Reduce one wall record while preserving the three radial zones."""
    r = np.asarray(r, dtype=float)
    zones = radial_zones(r, buffer_points)
    summary = {}
    for zone, indices in zones.items():
        j_values = record["tensor_norms"]["J_orthonormal_frobenius"]
        dxj_values = record["tensor_norms"]["DXJ_orthonormal_frobenius"]
        summary[zone] = {
            "J_orthonormal_frobenius": _maximum_record(
                record["tensor_norms"]["J_orthonormal_frobenius"], indices, r,
            ),
            "J_relative_orthonormal": _maximum_record(
                record["tensor_norms"]["J_relative_orthonormal"], indices, r,
            ),
            "DXJ_orthonormal_frobenius": _maximum_record(
                dxj_values, indices, r,
            ),
            "proper_statistics": {
                "J": _proper_profile_statistics(
                    j_values, indices, r, record["metric_tensor"],
                ),
                "DXJ": _proper_profile_statistics(
                    dxj_values, indices, r, record["metric_tensor"],
                ),
                "J_trace": _proper_profile_statistics(
                    record["tensor_norms"]["J_mixed_trace_absolute"],
                    indices, r, record["metric_tensor"],
                ),
                "J_tracefree": _proper_profile_statistics(
                    record["tensor_norms"]["J_tracefree_orthonormal_frobenius"],
                    indices, r, record["metric_tensor"],
                ),
                "DXJ_trace": _proper_profile_statistics(
                    record["tensor_norms"]["DXJ_mixed_trace_absolute"],
                    indices, r, record["metric_tensor"],
                ),
                "DXJ_tracefree": _proper_profile_statistics(
                    record["tensor_norms"]["DXJ_tracefree_orthonormal_frobenius"],
                    indices, r, record["metric_tensor"],
                ),
            },
            "components": {
                name: {
                    "J": _maximum_record(record["components"][name]["J"], indices, r),
                    "DXJ": _maximum_record(record["components"][name]["DXJ"], indices, r),
                    "raw_robin_normalized": _maximum_record(
                        record["components"][name]["robin_normalized"], indices, r,
                    ),
                } for name in COMPONENTS
            },
        }
    proper_from_outer = _proper_distance_from_outer_wall(r, record["metric_tensor"])
    fixed_collars = {}
    for width in (0.10, 0.20):
        indices = np.flatnonzero(proper_from_outer <= width + 1e-14)
        fixed_collars[f"{width:.2f}"] = {
            "node_count": int(len(indices)),
            "realized_proper_width": float(np.max(proper_from_outer[indices])),
            "J": _proper_profile_statistics(
                record["tensor_norms"]["J_orthonormal_frobenius"],
                indices, r, record["metric_tensor"],
            ),
            "DXJ": _proper_profile_statistics(
                record["tensor_norms"]["DXJ_orthonormal_frobenius"],
                indices, r, record["metric_tensor"],
            ),
        }
    gauge_valid = bool(record["wall_adapted_gauge_maximum"] < 1e-10)
    return {
        "wall": record["wall"],
        "zones": summary,
        "wall_adapted_gauge_maximum": record["wall_adapted_gauge_maximum"],
        "frame_defect_maximum": float(np.max(record["tensor_norms"]["frame_defect"])),
        "mixed_eigenvalue_maximum_imaginary": record["tensor_norms"][
            "mixed_eigenvalue_maximum_imaginary"
        ],
        "fixed_proper_outer_collars": fixed_collars,
        "full_J_tensor_interpretation_valid": gauge_valid,
        "separate_rows": {
            "Phi_robin_maximum_absolute": float(np.max(np.abs(
                record["separate_rows"]["Phi_robin"]
            ))),
            "chi_neumann_maximum_absolute": float(np.max(np.abs(
                record["separate_rows"]["chi_neumann"]
            ))),
        },
        "finite": record["finite"],
    }


def compare_wall_records(reference, comparison, r, buffer_points=7):
    """Measure tensor differences in the reference wall's orthonormal frame."""
    if reference["wall"] != comparison["wall"]:
        raise ValueError("wall records must describe the same compact wall")
    r = np.asarray(r, dtype=float)
    metric = np.asarray(reference["metric_tensor"], dtype=float)
    if metric.shape != np.asarray(comparison["metric_tensor"]).shape:
        raise ValueError("wall records must share a radial grid")
    frames, frame_defect = _orthonormal_frames(metric)
    delta_j = np.asarray(reference["J_tensor"]) - np.asarray(
        comparison["J_tensor"]
    )
    delta_dxj = np.asarray(reference["DXJ_tensor"]) - np.asarray(
        comparison["DXJ_tensor"]
    )
    j_hat = np.einsum("nai,nab,nbj->nij", frames, delta_j, frames)
    dxj_hat = np.einsum("nai,nab,nbj->nij", frames, delta_dxj, frames)
    j_norm = np.linalg.norm(j_hat, axis=(1, 2))
    dxj_norm = np.linalg.norm(dxj_hat, axis=(1, 2))
    zones = {}
    for name, indices in radial_zones(r, buffer_points).items():
        zones[name] = {
            "J": _proper_profile_statistics(j_norm, indices, r, metric),
            "DXJ": _proper_profile_statistics(dxj_norm, indices, r, metric),
            "J_maximum": _maximum_record(j_norm, indices, r),
            "DXJ_maximum": _maximum_record(dxj_norm, indices, r),
        }
    return {
        "frame": "primary-dt reference wall orthonormal frame",
        "zones": zones,
        "frame_defect_maximum": float(np.max(frame_defect)),
        "finite": bool(
            np.all(np.isfinite(j_norm)) and np.all(np.isfinite(dxj_norm))
        ),
    }


def directional_derivative_ladder(
    position, direction, z, r, background,
    epsilons=(3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4), stencil_width=7,
    normalize_direction=False, maximum_state_fraction=1e-3,
):
    """Check analytic ``DJ[q].direction`` with a centered epsilon ladder."""
    q = np.asarray(position, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if q.shape != direction.shape:
        raise ValueError("direction must share the state shape")
    if not np.all(np.isfinite(direction)):
        raise ValueError("direction must be finite")
    epsilons = tuple(float(value) for value in epsilons)
    if any(value <= 0.0 for value in epsilons):
        raise ValueError("epsilon ladder must be positive")
    direction_rescale = 1.0
    if normalize_direction:
        fraction = float(maximum_state_fraction)
        if not (0.0 < fraction < 0.1):
            raise ValueError("maximum_state_fraction must lie in (0,0.1)")
        direction_maximum = float(np.max(np.abs(direction)))
        state_scale = max(float(np.max(np.abs(q))), 1.0)
        if direction_maximum > 0.0:
            direction_rescale = max(
                1.0,
                fraction * state_scale / (max(epsilons) * direction_maximum),
            )
        # Every trial state must retain positive g_zz and a Lorentzian wall
        # metric.  Back off deterministically if the target rescale is unsafe.
        zero = np.zeros_like(q)
        while direction_rescale > 1.0:
            try:
                for epsilon in epsilons:
                    for sign in (-1.0, 1.0):
                        trial = q + sign * epsilon * direction_rescale * direction
                        for wall in WALLS:
                            wall_junction_rows(
                                trial, zero, z, r, background, wall, stencil_width,
                            )
                break
            except (ValueError, np.linalg.LinAlgError):
                direction_rescale *= 0.5
        direction_rescale = max(direction_rescale, 1.0)
    evaluation_direction = direction_rescale * direction
    analytic = {
        wall: wall_junction_rows(
            q, direction, z, r, background, wall, stencil_width,
        ) for wall in WALLS
    }
    records = []
    for epsilon in epsilons:
        walls = {}
        for wall in WALLS:
            zero = np.zeros_like(q)
            plus_record = wall_junction_rows(
                q + epsilon * evaluation_direction, zero, z, r, background, wall,
                stencil_width,
            )
            minus_record = wall_junction_rows(
                q - epsilon * evaluation_direction, zero, z, r, background, wall,
                stencil_width,
            )
            plus = plus_record["J_tensor"]
            minus = minus_record["J_tensor"]
            numerical = (plus - minus) / (2.0 * epsilon * direction_rescale)
            error = numerical - analytic[wall]["DXJ_tensor"]
            scale = max(
                float(np.max(np.abs(numerical))),
                float(np.max(np.abs(analytic[wall]["DXJ_tensor"]))), 1e-14,
            )
            component_records = {}
            for name in COMPONENTS:
                numerical_component = (
                    plus_record["components"][name]["J"]
                    - minus_record["components"][name]["J"]
                ) / (2.0 * epsilon * direction_rescale)
                analytic_component = analytic[wall]["components"][name]["DXJ"]
                component_error = numerical_component - analytic_component
                component_scale = max(
                    float(np.max(np.abs(numerical_component))),
                    float(np.max(np.abs(analytic_component))), 1e-14,
                )
                component_records[name] = {
                    "maximum_absolute_error": float(np.max(np.abs(component_error))),
                    "maximum_relative_scale_error": float(
                        np.max(np.abs(component_error)) / component_scale
                    ),
                }
            walls[wall] = {
                "maximum_absolute_error": float(np.max(np.abs(error))),
                "maximum_relative_scale_error": float(np.max(np.abs(error)) / scale),
                "analytic_maximum_absolute": float(np.max(np.abs(
                    analytic[wall]["DXJ_tensor"]
                ))),
                "numerical_maximum_absolute": float(np.max(np.abs(numerical))),
                "components": component_records,
            }
        records.append({"epsilon": epsilon, "walls": walls})
    ladder_errors = np.asarray([
        max(item["walls"][wall]["maximum_relative_scale_error"] for wall in WALLS)
        for item in records
    ])
    ladder_absolute = np.asarray([
        max(item["walls"][wall]["maximum_absolute_error"] for wall in WALLS)
        for item in records
    ])
    accurate = (ladder_errors < 1e-7) & (ladder_absolute < 1e-8)
    adjacent_accurate = bool(np.any(accurate[:-1] & accurate[1:]))
    convergence_slopes = []
    for left, right in zip(records[:-1], records[1:]):
        left_error = max(
            left["walls"][wall]["maximum_absolute_error"] for wall in WALLS
        )
        right_error = max(
            right["walls"][wall]["maximum_absolute_error"] for wall in WALLS
        )
        if left_error > 0.0 and right_error > 0.0:
            convergence_slopes.append(float(
                math.log(left_error / right_error)
                / math.log(left["epsilon"] / right["epsilon"])
            ))
        else:
            convergence_slopes.append(float("nan"))
    second_order_pair = bool(any(
        1.5 <= slope <= 2.5 for slope in convergence_slopes[:3]
        if np.isfinite(slope)
    ))
    return {
        "records": records,
        "direction_rescale": float(direction_rescale),
        "maximum_state_fraction": (
            float(maximum_state_fraction) if normalize_direction else None
        ),
        "best_maximum_relative_scale_error": float(np.min(ladder_errors)),
        "best_maximum_absolute_error": float(np.min(ladder_absolute)),
        "adjacent_accurate_pair": adjacent_accurate,
        "second_order_pair_before_roundoff": second_order_pair,
        "convergence_slopes": convergence_slopes,
        "finite": bool(all(
            np.isfinite(item["walls"][wall]["maximum_relative_scale_error"])
            for item in records for wall in WALLS
        )),
    }


def high_precision_local_directional_ladder(
    position, direction, z, r, background,
    epsilons=(3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5),
    stencil_width=7, decimal_digits=80,
):
    """Audit ``DJ[q].direction`` without subtracting ``J[q]`` in float64.

    The production ``D_z`` matrix is linear, so it is applied once to the
    position and direction separately.  Only the local nonlinear Israel row
    is then centered with ``mpmath`` at the requested precision.  A separately
    evaluated high-precision analytic derivative is also compared with the
    production float64 formula.  This lane is intended for archived physical
    velocities whose change in ``J`` is below the float64 subtraction floor.
    """
    try:
        import mpmath as mp
    except ImportError as error:  # pragma: no cover - environment failure
        raise RuntimeError("mpmath is required for the high-precision audit") from error

    q, direction, z, r = _validate_state(position, direction, z, r)
    digits = int(decimal_digits)
    if digits < 40:
        raise ValueError("high-precision audit requires at least 40 decimal digits")
    epsilons = tuple(float(value) for value in epsilons)
    if any(value <= 0.0 for value in epsilons):
        raise ValueError("epsilon ladder must be positive")
    dz = _dense_derivative(z, 1, stencil_width)
    q_fields = _component_fields(q, r)
    d_fields = _component_fields(direction, r)
    q_z = {name: dz @ q_fields[name] for name in COMPONENTS}
    d_z = {name: dz @ d_fields[name] for name in COMPONENTS}
    analytic = {
        wall: wall_junction_rows(q, direction, z, r, background, wall, stencil_width)
        for wall in WALLS
    }

    records = []
    core_analytic_absolute = 0.0
    core_analytic_relative = 0.0
    with mp.workdps(digits):
        to_mp = lambda value: mp.mpf(float(value))
        two = mp.mpf(2)
        six = mp.mpf(6)
        half = mp.mpf("0.5")
        for epsilon in epsilons:
            walls = {}
            parameter = mp.mpf(str(epsilon))
            for wall in WALLS:
                index = -1 if wall == "upper" else 0
                upper = wall == "upper"
                orientation = mp.mpf(1 if upper else -1)
                gamma = to_mp(background["wall_stiffness"])
                target = to_mp(background["v1"] if upper else background["v0"])
                bare_beta = to_mp(
                    background["beta_b"] if upper else background["beta_a"]
                )
                reference_potential = to_mp(
                    background["wall_potential_b"] if upper
                    else background["wall_potential_a"]
                )
                components = {}
                wall_absolute = 0.0
                wall_analytic = 0.0
                wall_numerical = 0.0
                for name in COMPONENTS:
                    numerical = np.empty(len(r))
                    high_analytic = np.empty(len(r))
                    for radial_index in range(len(r)):
                        field = to_mp(q_fields[name][index, radial_index])
                        field_t = to_mp(d_fields[name][index, radial_index])
                        field_z = to_mp(q_z[name][index, radial_index])
                        field_zt = to_mp(d_z[name][index, radial_index])
                        phi = to_mp(q[index, radial_index, 7])
                        phi_t = to_mp(direction[index, radial_index, 7])
                        gzz = to_mp(q[index, radial_index, 6])
                        gzz_t = to_mp(direction[index, radial_index, 6])

                        def local_row(step):
                            trial_phi = phi + step * phi_t
                            delta = trial_phi - target
                            potential = half * gamma * delta**2
                            beta = (
                                bare_beta - (potential - reference_potential) / six
                                if upper else
                                bare_beta + (potential - reference_potential) / six
                            )
                            trial_gzz = gzz + step * gzz_t
                            if trial_gzz <= 0:
                                raise ValueError(
                                    "high-precision trial lost positive g_zz"
                                )
                            A = mp.sqrt(trial_gzz)
                            trial_field = field + step * field_t
                            trial_field_z = field_z + step * field_zt
                            return orientation * (
                                trial_field_z + two * beta * A * trial_field
                            ) / (two * A)

                        plus = local_row(parameter)
                        minus = local_row(-parameter)
                        numerical[radial_index] = float(
                            (plus - minus) / (two * parameter)
                        )

                        delta = phi - target
                        potential = half * gamma * delta**2
                        beta = (
                            bare_beta - (potential - reference_potential) / six
                            if upper else
                            bare_beta + (potential - reference_potential) / six
                        )
                        beta_phi = (
                            -gamma * delta / six if upper else gamma * delta / six
                        )
                        A = mp.sqrt(gzz)
                        A_t = gzz_t / (two * A)
                        robin = field_z + two * beta * A * field
                        robin_t = field_zt + two * (
                            beta_phi * phi_t * A * field
                            + beta * A_t * field
                            + beta * A * field_t
                        )
                        high_analytic[radial_index] = float(
                            orientation / (two * A)
                            * (robin_t - robin * A_t / A)
                        )

                    expected = analytic[wall]["components"][name]["DXJ"]
                    error = numerical - expected
                    analytic_error = high_analytic - expected
                    scale = max(
                        float(np.max(np.abs(numerical))),
                        float(np.max(np.abs(expected))), 1e-14,
                    )
                    absolute = float(np.max(np.abs(error)))
                    analytic_absolute = float(np.max(np.abs(analytic_error)))
                    analytic_scale = max(
                        float(np.max(np.abs(high_analytic))),
                        float(np.max(np.abs(expected))), 1e-14,
                    )
                    core_analytic_absolute = max(
                        core_analytic_absolute, analytic_absolute,
                    )
                    core_analytic_relative = max(
                        core_analytic_relative, analytic_absolute / analytic_scale,
                    )
                    components[name] = {
                        "maximum_absolute_error": absolute,
                        "maximum_relative_scale_error": absolute / scale,
                        "core_vs_high_precision_analytic_absolute": analytic_absolute,
                        "core_vs_high_precision_analytic_relative": (
                            analytic_absolute / analytic_scale
                        ),
                    }
                    wall_absolute = max(wall_absolute, absolute)
                    wall_analytic = max(
                        wall_analytic, float(np.max(np.abs(expected))),
                    )
                    wall_numerical = max(
                        wall_numerical, float(np.max(np.abs(numerical))),
                    )
                walls[wall] = {
                    "maximum_absolute_error": wall_absolute,
                    "maximum_relative_scale_error": wall_absolute / max(
                        wall_analytic, wall_numerical, 1e-14,
                    ),
                    "analytic_maximum_absolute": wall_analytic,
                    "numerical_maximum_absolute": wall_numerical,
                    "components": components,
                }
            records.append({"epsilon": epsilon, "walls": walls})

    relative = np.asarray([
        max(item["walls"][wall]["maximum_relative_scale_error"] for wall in WALLS)
        for item in records
    ])
    absolute = np.asarray([
        max(item["walls"][wall]["maximum_absolute_error"] for wall in WALLS)
        for item in records
    ])
    accurate = (relative < 1e-7) & (absolute < 1e-10)
    slopes = []
    for left, right, left_error, right_error in zip(
        records[:-1], records[1:], absolute[:-1], absolute[1:]
    ):
        slopes.append(float(
            math.log(left_error / right_error)
            / math.log(left["epsilon"] / right["epsilon"])
        ) if left_error > 0 and right_error > 0 else float("nan"))
    return {
        "method": (
            "production Dz factored by linearity; local nonlinear row centered "
            "with mpmath and independently differentiated at high precision"
        ),
        "mpmath_version": str(mp.__version__),
        "decimal_digits": digits,
        "records": records,
        "best_maximum_relative_scale_error": float(np.min(relative)),
        "best_maximum_absolute_error": float(np.min(absolute)),
        "core_vs_high_precision_analytic_maximum_absolute": (
            float(core_analytic_absolute)
        ),
        "core_vs_high_precision_analytic_maximum_relative": (
            float(core_analytic_relative)
        ),
        "adjacent_accurate_pair": bool(np.any(accurate[:-1] & accurate[1:])),
        "second_order_pair_before_roundoff": bool(any(
            1.5 <= slope <= 2.5 for slope in slopes[:5] if np.isfinite(slope)
        )),
        "convergence_slopes": slopes,
        "finite": bool(
            np.all(np.isfinite(relative)) and np.all(np.isfinite(absolute))
            and np.isfinite(core_analytic_absolute)
            and np.isfinite(core_analytic_relative)
        ),
    }


def manufactured_state(nz=17, nr=25):
    """Return an exactly wall-compatible cubic state and tangent."""
    z = np.linspace(1.0, 2.0, int(nz))
    r = np.linspace(0.0, 2.0, int(nr))
    background = {
        "wall_stiffness": 4.0,
        "v0": 0.10,
        "v1": 0.08,
        "beta_a": 0.15,
        "beta_b": 0.20,
        "wall_potential_a": 0.0,
        "wall_potential_b": 0.0,
    }
    base = CubicHermiteSpline(
        (z[0], z[-1]), (1.0, 1.08),
        (-2.0 * background["beta_a"], -2.0 * background["beta_b"] * 1.08),
    )(z)
    q = np.zeros((len(z), len(r), 9))
    rr = r[None, :]
    q[:, :, 2] = -base[:, None] * (1.0 + 0.01 * rr**2)
    q[:, :, 3] = base[:, None] * (1.0 + 0.02 * rr**2)
    q[:, :, 4] = 0.015 * base[:, None]
    q[:, :, 5] = 0.02 * base[:, None]
    q[:, :, 6] = 1.0
    phi_profile = CubicHermiteSpline(
        (z[0], z[-1]), (background["v0"], background["v1"]), (0.0, 0.0),
    )(z)
    q[:, :, 7] = phi_profile[:, None]
    multiplier = 0.06 * (1.0 + 0.03 * rr**2)
    v = np.zeros_like(q)
    for field in (2, 3, 4, 5):
        v[:, :, field] = multiplier * q[:, :, field]
    return {"z": z, "r": r, "position": q, "velocity": v, "background": background}


def manufactured_controls():
    """Run preservation, derivative, orientation, and adverse controls."""
    data = manufactured_state()
    q = data["position"]
    v = data["velocity"]
    z = data["z"]
    r = data["r"]
    background = data["background"]
    exact = {
        wall: wall_junction_rows(q, v, z, r, background, wall) for wall in WALLS
    }
    exact_j = max(
        np.max(np.abs(exact[wall]["J_tensor"])) for wall in WALLS
    )
    exact_tangent = max(
        np.max(np.abs(exact[wall]["DXJ_tensor"])) for wall in WALLS
    )
    exact_scalar = max(
        np.max(np.abs(exact[wall]["separate_rows"]["Phi_robin"]))
        for wall in WALLS
    )
    exact_chi = max(
        np.max(np.abs(exact[wall]["separate_rows"]["chi_neumann"]))
        for wall in WALLS
    )

    zz, rr = np.meshgrid(z, r, indexing="ij")
    direction = np.zeros_like(q)
    direction[:, :, 2] = 0.03 * (1.0 + zz + 0.1 * rr**2)
    direction[:, :, 3] = -0.02 * (1.0 + zz**2 + 0.05 * rr**2)
    direction[:, :, 4] = 0.004 * (1.0 + zz)
    direction[:, :, 5] = -0.003 * (1.0 + 0.2 * zz)
    direction[:, :, 6] = 0.01 * (1.0 + 0.1 * zz)
    direction[:, :, 7] = 0.02 * (1.0 + 0.1 * rr**2)
    ladder = directional_derivative_ladder(q, direction, z, r, background)

    fraction = (z - z[0]) / (z[-1] - z[0])
    upper_shape = fraction**2 * (fraction - 1.0)
    adverse = q.copy()
    adverse[:, :, 3] += 0.04 * upper_shape[:, None] * (1.0 + 0.02 * r[None, :] ** 2)
    adverse_records = {
        wall: wall_junction_rows(
            adverse, np.zeros_like(adverse), z, r, background, wall,
        ) for wall in WALLS
    }
    upper_adverse = float(np.max(np.abs(
        adverse_records["upper"]["components"]["sphere"]["J"]
    )))
    lower_leakage = float(np.max(np.abs(
        adverse_records["lower"]["components"]["sphere"]["J"]
    )))

    tangent_adverse = v.copy()
    tangent_adverse[:, :, 3] += (
        0.035 * upper_shape[:, None] * (1.0 + 0.02 * r[None, :] ** 2)
    )
    tangent_adverse_records = {
        wall: wall_junction_rows(q, tangent_adverse, z, r, background, wall)
        for wall in WALLS
    }
    tangent_adverse_j = max(
        np.max(np.abs(tangent_adverse_records[wall]["J_tensor"])) for wall in WALLS
    )
    tangent_adverse_upper = float(np.max(np.abs(
        tangent_adverse_records["upper"]["components"]["sphere"]["DXJ"]
    )))
    tangent_adverse_lower = float(np.max(np.abs(
        tangent_adverse_records["lower"]["components"]["sphere"]["DXJ"]
    )))

    wrong_orientation = {}
    for wall in WALLS:
        record = exact[wall]
        wrong = record["J_tensor"] + 2.0 * (
            record["source"]["c"][:, None, None] * record["metric_tensor"]
        )
        wrong_orientation[wall] = float(np.max(np.abs(wrong)))

    off_gauge = q.copy()
    off_gauge[0, :, 0] = 0.025
    off_gauge[-1, :, 0] = -0.025
    off_gauge_maximum = max(
        wall_junction_rows(
            off_gauge, np.zeros_like(off_gauge), z, r, background, wall,
        )["wall_adapted_gauge_maximum"] for wall in WALLS
    )
    gates = {
        "exact_J_below_1e_11": bool(exact_j < 1e-11),
        "exact_DXJ_below_1e_11": bool(exact_tangent < 1e-11),
        "scalar_compatibility_lane_below_1e_11": bool(exact_scalar < 1e-11),
        "chi_compatibility_lane_below_1e_11": bool(exact_chi < 1e-11),
        "directional_ladder_below_1e_7": bool(
            ladder["best_maximum_relative_scale_error"] < 1e-7
            and ladder["adjacent_accurate_pair"]
            and ladder["second_order_pair_before_roundoff"]
        ),
        "localized_adverse_detected": bool(upper_adverse > 1e-3),
        "localized_adverse_lower_leakage_below_1e_11": bool(lower_leakage < 1e-11),
        "value_small_tangent_adverse_detected": bool(
            tangent_adverse_j < 1e-11 and tangent_adverse_upper > 1e-3
        ),
        "tangent_adverse_lower_leakage_below_1e_11": bool(
            tangent_adverse_lower < 1e-11
        ),
        "wrong_orientation_detected_both_walls": bool(
            min(wrong_orientation.values()) > 0.1
        ),
        "off_gauge_adverse_detected": bool(off_gauge_maximum > 0.02),
    }
    return {
        "exact_J_maximum_absolute": float(exact_j),
        "exact_DXJ_maximum_absolute": float(exact_tangent),
        "exact_Phi_robin_maximum_absolute": float(exact_scalar),
        "exact_chi_neumann_maximum_absolute": float(exact_chi),
        "directional_derivative": ladder,
        "localized_upper_adverse_J_maximum": upper_adverse,
        "localized_adverse_lower_leakage": lower_leakage,
        "value_small_tangent_adverse_J_maximum": float(tangent_adverse_j),
        "tangent_adverse_upper_DXJ_maximum": tangent_adverse_upper,
        "tangent_adverse_lower_leakage": tangent_adverse_lower,
        "wrong_orientation_J_maximum": wrong_orientation,
        "off_gauge_adverse_maximum": float(off_gauge_maximum),
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def cap_spline_sphere_residual(position, z, r, radius, background, wall="upper"):
    """Reproduce the cap endpoint's cubic-spline ``W_s/W-c`` extraction."""
    q = np.asarray(position, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    radius = float(radius)
    if not (r[0] <= radius <= r[-1]):
        raise ValueError("cap endpoint lies outside the radial grid")
    index = -1 if wall == "upper" else 0
    orientation = 1.0 if wall == "upper" else -1.0

    def spline(field):
        return RectBivariateSpline(
            z, r, np.asarray(field, dtype=float),
            kx=min(3, len(z) - 1), ky=min(3, len(r) - 1), s=0,
        )

    sphere = spline(q[:, :, 3])
    normal = spline(q[:, :, 6])
    phi = spline(q[:, :, 7])
    z_value = float(z[index])
    h = float(np.asarray(sphere.ev(z_value, radius)).reshape(-1)[0])
    h_z = float(np.asarray(sphere.ev(z_value, radius, dx=1)).reshape(-1)[0])
    A = math.sqrt(float(np.asarray(normal.ev(z_value, radius)).reshape(-1)[0]))
    phi_value = float(np.asarray(phi.ev(z_value, radius)).reshape(-1)[0])
    source = wall_source_coefficients(phi_value, background, wall)
    geometric = orientation * h_z / (2.0 * A * h)
    return {
        "radius": radius,
        "geometric_sphere_coefficient": geometric,
        "wall_coefficient": float(source["c"]),
        "residual": float(geometric - source["c"]),
    }


def interpolate_native_sphere(record, r, radius, radius_rate=0.0):
    """Sample the native-wall sphere coefficient and its material tangent."""
    r = np.asarray(r, dtype=float)
    radius = float(radius)
    item = record["components"]["sphere"]
    residual = float(PchipInterpolator(r, item["mixed_coefficient"])(radius))
    fixed_rate = float(PchipInterpolator(r, item["DX_mixed_coefficient"])(radius))
    radial_rate = float(PchipInterpolator(r, item["mixed_coefficient_r"])(radius))
    cubic = float(CubicSpline(r, item["mixed_coefficient"])(radius))
    return {
        "residual": residual,
        "fixed_grid_DX_residual": fixed_rate,
        "radial_derivative": radial_rate,
        "material_DX_residual": fixed_rate + float(radius_rate) * radial_rate,
        "pchip_minus_cubic": residual - cubic,
    }
