"""Prospectively sealed interpolation tools for A790 Test 2D.

The production interpolant and the independent local polynomial evaluator do
not share evaluation code.  Leave-level-out estimates use only samples from a
single native grid, so they do not consume the physical G9/G10/G11
discretization differences that they are intended to qualify.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.interpolate import RectBivariateSpline


MINIMUM_NODES = 6
INTERPOLATION_ORDER_GATE = 2.5
SAFETY_FACTOR = 1.5


def _validated_inputs(field, z, r, target_z=None, target_r=None):
    values = np.asarray(field, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    if z.ndim != 1 or r.ndim != 1:
        raise ValueError("source coordinates must be one-dimensional")
    if len(z) < MINIMUM_NODES or len(r) < MINIMUM_NODES:
        raise ValueError("fifth-order interpolation requires six source nodes per axis")
    if np.any(np.diff(z) <= 0.0) or np.any(np.diff(r) <= 0.0):
        raise ValueError("source coordinates must strictly increase")
    if values.shape[:2] != (len(z), len(r)):
        raise ValueError("field and source grid do not match")
    if not np.all(np.isfinite(values)):
        raise ValueError("source field is nonfinite")
    if target_z is None or target_r is None:
        return values, z, r, None, None
    tz, tr = np.broadcast_arrays(
        np.asarray(target_z, dtype=float), np.asarray(target_r, dtype=float),
    )
    if not (np.all(np.isfinite(tz)) and np.all(np.isfinite(tr))):
        raise ValueError("target coordinates are nonfinite")
    scale_z = max(abs(z[0]), abs(z[-1]), 1.0)
    scale_r = max(abs(r[0]), abs(r[-1]), 1.0)
    if (
        np.min(tz) < z[0] - 1e-13 * scale_z
        or np.max(tz) > z[-1] + 1e-13 * scale_z
        or np.min(tr) < r[0] - 1e-13 * scale_r
        or np.max(tr) > r[-1] + 1e-13 * scale_r
    ):
        raise ValueError("interpolation target leaves the native grid")
    return values, z, r, np.clip(tz, z[0], z[-1]), np.clip(tr, r[0], r[-1])


def spline_interpolate(field, z, r, target_z, target_r, degree=5):
    """Evaluate a tensor-product zero-smoothing spline at paired targets."""
    values, z, r, tz, tr = _validated_inputs(field, z, r, target_z, target_r)
    degree = int(degree)
    if degree not in (1, 3, 5):
        raise ValueError("spline degree must be 1, 3, or 5")
    if len(z) <= degree or len(r) <= degree:
        raise ValueError("insufficient source nodes for requested spline degree")
    result = np.empty((*tz.shape, *values.shape[2:]), dtype=float)
    flattened_z, flattened_r = tz.ravel(), tr.ravel()
    for component in np.ndindex(values.shape[2:]):
        spline = RectBivariateSpline(
            z, r, values[(slice(None), slice(None), *component)],
            s=0, kx=degree, ky=degree,
        )
        result[(..., *component)] = spline.ev(
            flattened_z, flattened_r,
        ).reshape(tz.shape)
    return result


def _local_barycentric_weights(nodes, targets, width=6):
    """Return local-node indices and barycentric evaluation weights."""
    nodes = np.asarray(nodes, dtype=float)
    targets = np.asarray(targets, dtype=float).ravel()
    if width != 6 or len(nodes) < width:
        raise ValueError("the independent evaluator requires six-node panels")
    starts = np.searchsorted(nodes, targets, side="right") - width // 2
    starts = np.clip(starts, 0, len(nodes) - width)
    indices = starts[:, None] + np.arange(width)[None, :]
    panel = nodes[indices]
    denominators = np.ones_like(panel)
    for column in range(width):
        for other in range(width):
            if column != other:
                denominators[:, column] *= panel[:, column] - panel[:, other]
    barycentric = 1.0 / denominators
    delta = targets[:, None] - panel
    exact = np.isclose(
        delta, 0.0, rtol=0.0,
        atol=8.0 * np.finfo(float).eps * np.maximum(np.abs(panel), 1.0),
    )
    weights = np.empty_like(panel)
    exact_rows = np.any(exact, axis=1)
    if np.any(exact_rows):
        weights[exact_rows] = exact[exact_rows].astype(float)
    regular = ~exact_rows
    if np.any(regular):
        raw = barycentric[regular] / delta[regular]
        weights[regular] = raw / np.sum(raw, axis=1)[:, None]
    return indices, weights


def barycentric5_interpolate(field, z, r, target_z, target_r, chunk_size=4096):
    """Independent local degree-five tensor-product interpolation.

    This routine deliberately does not use SciPy's spline construction or
    evaluation path.  Chunking bounds temporary memory for tensor fields.
    """
    values, z, r, tz, tr = _validated_inputs(field, z, r, target_z, target_r)
    flat_z, flat_r = tz.ravel(), tr.ravel()
    iz, wz = _local_barycentric_weights(z, flat_z)
    ir, wr = _local_barycentric_weights(r, flat_r)
    components = values.shape[2:]
    flattened = np.empty((len(flat_z), *components), dtype=float)
    for start in range(0, len(flat_z), int(chunk_size)):
        stop = min(start + int(chunk_size), len(flat_z))
        block = values[
            iz[start:stop, :, None], ir[start:stop, None, :], ...
        ]
        flattened[start:stop] = np.einsum(
            "ni,nj,nij...->n...",
            wz[start:stop], wr[start:stop], block, optimize=True,
        )
    return flattened.reshape((*tz.shape, *components))


def interpolate(field, z, r, target_z, target_r, method="quintic"):
    """Dispatch the sealed production, independent, and adverse methods."""
    if method == "quintic":
        return spline_interpolate(field, z, r, target_z, target_r, degree=5)
    if method == "independent5":
        return barycentric5_interpolate(field, z, r, target_z, target_r)
    if method == "cubic":
        return spline_interpolate(field, z, r, target_z, target_r, degree=3)
    if method == "linear":
        return spline_interpolate(field, z, r, target_z, target_r, degree=1)
    raise ValueError("unknown sealed interpolation method")


def mapped_metric_fields(
    metric, sphere_factor, z, r, distance, arclength,
    native_z, native_r, method="quintic",
):
    """Map a native metric into the Test-2D `(D,S)` chart."""
    native_metric = interpolate(metric, z, r, native_z, native_r, method)
    x_D = np.stack((
        np.gradient(native_z, distance, axis=0, edge_order=2),
        np.gradient(native_r, distance, axis=0, edge_order=2),
    ), axis=-1)
    x_S = np.stack((
        np.gradient(native_z, arclength, axis=1, edge_order=2),
        np.gradient(native_r, arclength, axis=1, edge_order=2),
    ), axis=-1)
    covariant = np.empty((*native_z.shape, 2, 2))
    covariant[..., 0, 0] = np.einsum(
        "...a,...ab,...b->...", x_D, native_metric, x_D,
    )
    covariant[..., 0, 1] = covariant[..., 1, 0] = np.einsum(
        "...a,...ab,...b->...", x_D, native_metric, x_S,
    )
    covariant[..., 1, 1] = np.einsum(
        "...a,...ab,...b->...", x_S, native_metric, x_S,
    )
    inverse = np.linalg.inv(covariant)
    determinant = np.linalg.det(covariant)
    sphere = interpolate(sphere_factor, z, r, native_z, native_r, method)
    if np.min(sphere) <= 0.0 or np.min(determinant) <= 0.0:
        raise ValueError("mapped Test-2D metric is nonpositive")
    native_areal = native_r * np.sqrt(sphere)
    volume = 4.0 * math.pi * native_areal**2 * np.sqrt(determinant)
    return {
        "covariant": covariant,
        "q_DD": inverse[..., 0, 0], "q_DS": inverse[..., 0, 1],
        "q_SS": inverse[..., 1, 1], "volume_density": volume,
        "native_areal_radius": native_areal,
        "basis_D": x_D, "basis_S": x_S, "native_metric": native_metric,
    }


def mapped_extrinsic_fields(
    extrinsic, sphere_factor, z, r, native_z, native_r, mapped_metric,
    method="quintic",
):
    """Map ADM extrinsic curvature into the physical orthonormal frame."""
    K = interpolate(extrinsic, z, r, native_z, native_r, method)
    base_K = K[..., :2, :2]
    metric = mapped_metric["native_metric"]
    x_D, x_S = mapped_metric["basis_D"], mapped_metric["basis_S"]
    chart_inverse = np.linalg.inv(mapped_metric["covariant"])
    grad_D = (
        chart_inverse[..., 0, 0, None] * x_D
        + chart_inverse[..., 1, 0, None] * x_S
    )
    norm_D = np.sqrt(np.einsum("...a,...ab,...b->...", grad_D, metric, grad_D))
    e_D = grad_D / norm_D[..., None]
    projection = np.einsum("...a,...ab,...b->...", x_S, metric, e_D)
    tangent = x_S - projection[..., None] * e_D
    norm_S = np.sqrt(np.einsum("...a,...ab,...b->...", tangent, metric, tangent))
    e_S = tangent / norm_S[..., None]
    K_DD = np.einsum("...a,...ab,...b->...", e_D, base_K, e_D)
    K_DS = np.einsum("...a,...ab,...b->...", e_D, base_K, e_S)
    K_SS = np.einsum("...a,...ab,...b->...", e_S, base_K, e_S)
    K_tangent = 0.5 * (K[..., 2, 2] + K[..., 3, 3])
    mapped_sphere = interpolate(sphere_factor, z, r, native_z, native_r, method)
    K_Omega = K_tangent / mapped_sphere
    base_inverse = np.linalg.inv(metric)
    trace_base = np.einsum("...ab,...ab->...", base_inverse, base_K)
    K_base_squared = np.einsum(
        "...ac,...bd,...ab,...cd->...", base_inverse, base_inverse,
        base_K, base_K, optimize=True,
    )
    return {
        "K_DD": K_DD, "K_DS": K_DS, "K_SS": K_SS, "K_Omega": K_Omega,
        "trace_K": trace_base + 2.0 * K_Omega,
        "KijKij": K_base_squared + 2.0 * K_Omega**2,
    }


def endpoint_preserving_indices(length, stride, offset):
    """Indices for the sealed endpoint-preserving leave-out family."""
    length, stride, offset = int(length), int(stride), int(offset)
    if length < MINIMUM_NODES or stride not in (2, 4) or not 0 <= offset < stride:
        raise ValueError("invalid leave-out index request")
    interior = np.arange(1, length - 1)
    selected = interior[interior % stride == offset]
    indices = np.unique(np.concatenate(([0], selected, [length - 1]))).astype(int)
    if len(indices) < MINIMUM_NODES:
        raise ValueError("leave-out subgrid has fewer than six nodes")
    return indices


def _point_magnitude(values):
    values = np.asarray(values, dtype=float)
    if values.ndim <= 2:
        return np.abs(values)
    return np.sqrt(np.sum(values**2, axis=tuple(range(2, values.ndim))))


def _weighted_quantile(values, weights, quantile=0.95):
    values, weights = np.asarray(values).ravel(), np.asarray(weights).ravel()
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("positive leave-out weight is required")
    return float(values[np.searchsorted(np.cumsum(weights), quantile * total)])


def error_norms(error, weights=None, mask=None):
    """Return weighted RMS and weighted pointwise 95th-percentile errors."""
    magnitude = _point_magnitude(error)
    if weights is None:
        weights = np.ones(magnitude.shape, dtype=float)
    weights = np.broadcast_to(np.asarray(weights, dtype=float), magnitude.shape)
    if np.any(weights < 0.0) or not np.all(np.isfinite(weights)):
        raise ValueError("error weights must be finite and nonnegative")
    admitted = np.ones(magnitude.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if admitted.shape != magnitude.shape or not np.any(admitted):
        raise ValueError("leave-out mask is empty or has the wrong shape")
    selected_weight = weights[admitted]
    total = float(np.sum(selected_weight))
    if total <= 0.0:
        raise ValueError("leave-out mask has zero total weight")
    selected = magnitude[admitted]
    return {
        "L2": float(np.sqrt(np.sum(selected_weight * selected**2) / total)),
        "q95": _weighted_quantile(selected, selected_weight),
        "Linf": float(np.max(selected)),
    }


def leave_level_out(field, z, r, weights=None):
    """Run every sealed stride/offset leave-out fit on one native level."""
    values, z, r, _, _ = _validated_inputs(field, z, r)
    if weights is None:
        weights = np.ones((len(z), len(r)), dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(z), len(r)):
        raise ValueError("leave-out weights and native grid do not match")
    zz, rr = np.meshgrid(z, r, indexing="ij")
    records = {}
    envelopes = {2: {"L2": 0.0, "q95": 0.0, "Linf": 0.0},
                 4: {"L2": 0.0, "q95": 0.0, "Linf": 0.0}}
    for stride in (2, 4):
        for offset_z in range(stride):
            iz = endpoint_preserving_indices(len(z), stride, offset_z)
            for offset_r in range(stride):
                ir = endpoint_preserving_indices(len(r), stride, offset_r)
                predicted = spline_interpolate(
                    values[np.ix_(iz, ir)], z[iz], r[ir], zz, rr, degree=5,
                )
                retained = np.zeros((len(z), len(r)), dtype=bool)
                retained[np.ix_(iz, ir)] = True
                norms = error_norms(predicted - values, weights, ~retained)
                key = f"q{stride}_oz{offset_z}_or{offset_r}"
                records[key] = norms
                for norm in envelopes[stride]:
                    envelopes[stride][norm] = max(envelopes[stride][norm], norms[norm])
    scale = max(float(np.max(np.abs(values))), 1.0)
    floor = 512.0 * np.finfo(float).eps * scale
    orders, admissible = {}, True
    for norm in ("L2", "q95"):
        e2, e4 = envelopes[2][norm], envelopes[4][norm]
        resolved = bool(e2 > floor and e4 > floor)
        order = float(math.log(e4 / e2, 2.0)) if resolved and e4 > e2 else None
        orders[norm] = order
        admissible = bool(
            admissible and order is not None and order > INTERPOLATION_ORDER_GATE
        )
    return {
        "records": records,
        "envelopes": {str(key): value for key, value in envelopes.items()},
        "orders": orders,
        "roundoff_floor": floor,
        "admissible": admissible,
    }


def interpolation_allowance(
    primary, independent, cubic, leave_out, weights=None,
):
    """Construct the sealed per-state interpolation allowance."""
    delta55 = error_norms(np.asarray(primary) - np.asarray(independent), weights)
    delta53 = error_norms(np.asarray(primary) - np.asarray(cubic), weights)
    result = {"admissible": bool(leave_out["admissible"]), "norms": {}}
    for norm in ("L2", "q95"):
        e2 = float(leave_out["envelopes"]["2"][norm])
        order = leave_out["orders"][norm]
        richardson = (
            e2 / (2.0**float(order) - 1.0)
            if order is not None and order > INTERPOLATION_ORDER_GATE else math.inf
        )
        allowance = SAFETY_FACTOR * max(richardson, delta55[norm], delta53[norm])
        result["norms"][norm] = {
            "leave_out_e2": e2,
            "leave_out_e4": float(leave_out["envelopes"]["4"][norm]),
            "leave_out_order": order,
            "richardson": richardson,
            "production_vs_independent_degree5": delta55[norm],
            "quintic_vs_cubic": delta53[norm],
            "allowance": allowance,
        }
        result["admissible"] = bool(result["admissible"] and np.isfinite(allowance))
    return result


def manufactured_interpolation_controls():
    """Fast analytic qualification and false-pass controls."""
    z = np.linspace(-0.4, 1.1, 33)
    r = np.linspace(0.0, 1.8, 49)
    tz = np.linspace(z[0], z[-1], 41)[:, None]
    tr = np.linspace(r[0], r[-1], 57)[None, :]
    Z, R = np.broadcast_arrays(tz, tr)

    polynomial = (
        0.3 + 0.2 * z[:, None] - 0.1 * r[None, :]**2
        + 0.07 * z[:, None]**3 * r[None, :]**2
        - 0.01 * z[:, None]**5 * r[None, :]**5
    )
    polynomial_truth = (
        0.3 + 0.2 * Z - 0.1 * R**2 + 0.07 * Z**3 * R**2
        - 0.01 * Z**5 * R**5
    )
    polynomial_errors = {
        "production": float(np.max(np.abs(
            spline_interpolate(polynomial, z, r, Z, R, degree=5) - polynomial_truth
        ))),
        "independent": float(np.max(np.abs(
            barycentric5_interpolate(polynomial, z, r, Z, R) - polynomial_truth
        ))),
    }

    smooth = (
        np.sin(0.7 * z[:, None]) * np.cos(0.9 * r[None, :])
        + 0.2 * np.exp(-((z[:, None] - 0.31)**2 + (r[None, :] - 0.83)**2) / 0.08)
    )
    smooth_leave = leave_level_out(smooth, z, r)

    # Covariant tensor under the smooth relabeling
    # (u,v)=(z+0.08z^2, r+0.04r^3).  Pulling its interpolated components into
    # (u,v) must recover the Euclidean identity.
    derivative_u = 1.0 + 0.16 * z[:, None]
    derivative_v = 1.0 + 0.12 * r[None, :]**2
    relabeled_metric = np.zeros((len(z), len(r), 2, 2))
    relabeled_metric[..., 0, 0] = derivative_u**2
    relabeled_metric[..., 1, 1] = derivative_v**2
    interpolated_metric = barycentric5_interpolate(relabeled_metric, z, r, Z, R)
    target_du = 1.0 + 0.16 * Z
    target_dv = 1.0 + 0.12 * R**2
    recovered_metric = interpolated_metric.copy()
    recovered_metric[..., 0, 0] /= target_du**2
    recovered_metric[..., 0, 1] /= target_du * target_dv
    recovered_metric[..., 1, 0] /= target_du * target_dv
    recovered_metric[..., 1, 1] /= target_dv**2
    identity = np.broadcast_to(np.eye(2), recovered_metric.shape)
    relabeling_tensor_error = float(np.max(np.abs(recovered_metric - identity)))

    def adverse_record(source, truth):
        primary = spline_interpolate(source, z, r, Z, R, degree=5)
        independent = barycentric5_interpolate(source, z, r, Z, R)
        cubic = spline_interpolate(source, z, r, Z, R, degree=3)
        leave = leave_level_out(source, z, r)
        allowance = interpolation_allowance(primary, independent, cubic, leave)
        actual = error_norms(primary - truth)
        contained = bool(
            not allowance["admissible"]
            or all(allowance["norms"][norm]["allowance"] >= actual[norm]
                   for norm in ("L2", "q95"))
        )
        return {"leave_out": leave, "allowance": allowance,
                "actual_error": actual, "contained_or_rejected": contained}

    cusp_source = np.sqrt(
        (z[:, None] - 0.17)**2 + (r[None, :] - 0.91)**2
    )
    cusp_truth = np.sqrt((Z - 0.17)**2 + (R - 0.91)**2)
    checker_source = (
        np.sin(0.82 * np.pi * np.arange(len(z)))[:, None]
        * np.sin(0.78 * np.pi * np.arange(len(r)))[None, :]
    )
    zi = (Z - z[0]) / (z[-1] - z[0]) * (len(z) - 1)
    ri = (R - r[0]) / (r[-1] - r[0]) * (len(r) - 1)
    checker_truth = np.sin(0.82 * np.pi * zi) * np.sin(0.78 * np.pi * ri)
    layer_width = 0.40 * (r[1] - r[0])
    layer_source = np.exp(-((r[None, :] - 0.91) / layer_width)**2)
    layer_source = np.broadcast_to(layer_source, (len(z), len(r))).copy()
    layer_truth = np.exp(-((R - 0.91) / layer_width)**2)
    smooth_truth = (
        np.sin(0.7 * Z) * np.cos(0.9 * R)
        + 0.2 * np.exp(-((Z - 0.31)**2 + (R - 0.83)**2) / 0.08)
    )
    checkerboard_source = smooth + 1e-3 * (
        (-1.0)**(np.arange(len(z))[:, None] + np.arange(len(r))[None, :])
    )
    adverse = {
        "cusp": adverse_record(cusp_source, cusp_truth),
        "near_nyquist": adverse_record(checker_source, checker_truth),
        "one_cell_layer": adverse_record(layer_source, layer_truth),
        "injected_checkerboard": adverse_record(checkerboard_source, smooth_truth),
    }
    passed = bool(
        max(polynomial_errors.values()) < 1e-11
        and smooth_leave["admissible"]
        and relabeling_tensor_error < 1e-11
        and all(item["contained_or_rejected"] for item in adverse.values())
    )
    return {
        "passed": passed,
        "polynomial_errors": polynomial_errors,
        "smooth_leave_out": smooth_leave,
        "smooth_relabeling_tensor_error": relabeling_tensor_error,
        "adverse": adverse,
    }
