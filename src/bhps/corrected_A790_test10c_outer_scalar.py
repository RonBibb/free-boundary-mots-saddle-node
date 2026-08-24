"""Independent outer-scalar diagnostics for sealed Test 10C."""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import simpson

from bhps.regular_so3_gh_reduction import regular_so3_perturbation_jets


SCALARS = slice(7, 9)


def polynomial_endpoint_weights(nodes, derivative=1):
    """Polynomial-exact endpoint weights, independent of production stencils."""
    x = np.asarray(nodes, dtype=float)
    order = int(derivative)
    if x.ndim != 1 or len(x) < order + 1 or order < 0:
        raise ValueError("invalid polynomial derivative request")
    offset = x - x[-1]
    matrix = np.vstack([offset**power for power in range(len(x))])
    rhs = np.zeros(len(x))
    rhs[order] = float(math.factorial(order))
    return np.linalg.solve(matrix, rhs)


def endpoint_derivative(field, radius, stencil_width=7):
    values = np.asarray(field, dtype=float)
    r = np.asarray(radius, dtype=float)
    width = int(stencil_width)
    if values.shape[1] != len(r) or width < 2 or width > len(r):
        raise ValueError("invalid endpoint derivative fields")
    weights = polynomial_endpoint_weights(r[-width:])
    return np.einsum("j,ijf->if", weights, values[:, -width:])


def independent_outward_speed(position, radius):
    q = np.asarray(position, dtype=float)
    r = np.asarray(radius, dtype=float)
    speed = np.empty(q.shape[0])
    for i in range(q.shape[0]):
        metric = regular_so3_perturbation_jets(float(r[-1]), q[i, -1])["metric"]
        inverse = np.linalg.inv(metric)
        gtt = float(inverse[0, 0])
        gtr = float(inverse[0, 2])
        grr = float(inverse[2, 2])
        discriminant = gtr * gtr - gtt * grr
        if gtt >= 0.0 or discriminant <= 0.0 or not np.isfinite(discriminant):
            raise ValueError("outer face is not independently hyperbolic")
        speed[i] = (gtr - np.sqrt(discriminant)) / gtt
    if np.any(speed <= 0.0) or not np.all(np.isfinite(speed)):
        raise ValueError("invalid independent outgoing speed")
    return speed


def independent_characteristic_terms(
    position, velocity, before, reference_position, reference_acceleration,
    time_value, radius, stencil_width=7, difference_step=1e-6,
):
    q = np.asarray(position, dtype=float)
    v = np.asarray(velocity, dtype=float)
    b = np.asarray(before, dtype=float)
    q0 = np.asarray(reference_position, dtype=float)
    a0 = np.asarray(reference_acceleration, dtype=float)
    time_value = float(time_value)
    reference = q0 + 0.5 * time_value**2 * a0
    reference_velocity = time_value * a0
    delta = q - reference
    delta_velocity = v - reference_velocity
    speed = independent_outward_speed(q, radius)
    speed_time = (
        independent_outward_speed(q + difference_step * v, radius)
        - independent_outward_speed(q - difference_step * v, radius)
    ) / (2.0 * difference_step)
    delta_r = endpoint_derivative(delta, radius, stencil_width)
    delta_velocity_r = endpoint_derivative(
        delta_velocity, radius, stencil_width,
    )
    term_a = b[:, -1] - a0[:, -1]
    term_v = speed[:, None] * delta_velocity_r
    term_c = speed_time[:, None] * delta_r
    target = a0[:, -1] - term_v - term_c
    return {
        "target": target,
        "term_A": term_a,
        "term_V": term_v,
        "term_C": term_c,
        "speed": speed,
        "speed_time": speed_time,
    }


def proper_face_weight(position, z, radius):
    q = np.asarray(position, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(radius, dtype=float)
    if q.shape[:2] != (len(z), len(r)):
        raise ValueError("face geometry and grid do not match")
    perpendicular = q[1:-1, -1, 3]
    compact = q[1:-1, -1, 6]
    if np.any(perpendicular <= 0.0) or np.any(compact <= 0.0):
        raise ValueError("outer-face spatial metric is not positive")
    return 4.0 * math.pi * r[-1] ** 2 * perpendicular * np.sqrt(compact)


def _integral(values, coordinates, method):
    if method == "simpson":
        return float(simpson(values, x=coordinates))
    if method == "trapezoid":
        return float(np.trapezoid(values, x=coordinates))
    raise ValueError("unknown quadrature")


def weighted_norm(field, weight, z_open, method="simpson"):
    values = np.asarray(field, dtype=float)
    weight = np.asarray(weight, dtype=float)
    z_open = np.asarray(z_open, dtype=float)
    if values.shape[0] != len(weight) or len(weight) != len(z_open):
        raise ValueError("weighted norm fields are not aligned")
    squared = np.sum(values**2, axis=-1)
    return float(np.sqrt(max(_integral(weight * squared, z_open, method), 0.0)))


def symmetric_ratio(difference, left, right):
    numerator = float(np.linalg.norm(difference))
    denominator = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)))
    if denominator == 0.0:
        if numerator != 0.0:
            raise ValueError("nonzero difference has zero symmetric scale")
        return 0.0
    return numerator / denominator


def weighted_symmetric_ratio(difference, left, right, weight, z_open, method):
    numerator = weighted_norm(difference, weight, z_open, method)
    denominator = max(
        weighted_norm(left, weight, z_open, method),
        weighted_norm(right, weight, z_open, method),
    )
    if denominator == 0.0:
        if numerator != 0.0:
            raise ValueError("nonzero weighted difference has zero scale")
        return 0.0
    return numerator / denominator


def pointwise_symmetric_ratio(difference, left, right):
    difference = np.abs(np.asarray(difference, dtype=float))
    scale = np.maximum(np.abs(left), np.abs(right))
    ratio = np.zeros_like(difference)
    nonzero = scale > 0.0
    if np.any(~nonzero & (difference > 0.0)):
        raise ValueError("nonzero pointwise difference has zero scale")
    ratio[nonzero] = difference[nonzero] / scale[nonzero]
    index = np.unravel_index(np.argmax(ratio), ratio.shape)
    return {"maximum": float(ratio[index]), "index": [int(v) for v in index]}


def collar_rms(field, position, z, radius, stencil_width=7):
    values = np.asarray(field, dtype=float)[1:-1, -int(stencil_width):, SCALARS]
    q = np.asarray(position, dtype=float)[1:-1, -int(stencil_width):]
    local_r = np.asarray(radius, dtype=float)[-int(stencil_width):]
    local_z = np.asarray(z, dtype=float)[1:-1]
    compact = q[..., 6]
    radial = q[..., 3] + local_r[None, :] ** 2 * q[..., 4]
    cross = local_r[None, :] * q[..., 1]
    transverse = q[..., 3]
    determinant = compact * radial - cross**2
    if np.any(determinant <= 0.0) or np.any(transverse <= 0.0):
        raise ValueError("outer collar spatial metric is not positive")
    weight = 4.0 * math.pi * local_r[None, :] ** 2 * transverse * np.sqrt(determinant)
    density = weight * np.sum(values**2, axis=-1)
    volume_r = simpson(weight, x=local_r, axis=1)
    squared_r = simpson(density, x=local_r, axis=1)
    volume = float(simpson(volume_r, x=local_z))
    squared = float(simpson(squared_r, x=local_z))
    if volume <= 0.0:
        raise ValueError("outer collar volume is not positive")
    return float(np.sqrt(max(squared / volume, 0.0)))


def correction_metrics(
    position, before, after, reference_acceleration, terms, z, radius,
    stencil_width=7,
):
    b = np.asarray(before, dtype=float)
    a = np.asarray(after, dtype=float)
    open_before = b[1:-1, -1, SCALARS]
    open_after = a[1:-1, -1, SCALARS]
    correction = open_after - open_before
    independent_target = np.asarray(terms["target"])[1:-1, SCALARS]
    term_a = np.asarray(terms["term_A"])[1:-1, SCALARS]
    term_v = np.asarray(terms["term_V"])[1:-1, SCALARS]
    term_c = np.asarray(terms["term_C"])[1:-1, SCALARS]
    weight = proper_face_weight(position, z, radius)
    z_open = np.asarray(z, dtype=float)[1:-1]
    proper = {}
    for method in ("simpson", "trapezoid"):
        numerator = weighted_norm(correction, weight, z_open, method)
        before_norm = weighted_norm(open_before, weight, z_open, method)
        after_norm = weighted_norm(open_after, weight, z_open, method)
        term_norms = [
            weighted_norm(value, weight, z_open, method)
            for value in (term_a, term_v, term_c)
        ]
        term_denominator = sum(term_norms)
        term_numerator = weighted_norm(
            term_a + term_v + term_c, weight, z_open, method,
        )
        proper[method] = {
            "ratio": numerator / max(before_norm, after_norm, 1e-300),
            "numerator": numerator,
            "before_norm": before_norm,
            "after_norm": after_norm,
            "term_balance_ratio": (
                term_numerator / term_denominator
                if term_denominator > 0.0 else (0.0 if term_numerator == 0.0 else np.inf)
            ),
            "term_norms": term_norms,
        }
    quadrature_relative = abs(
        proper["simpson"]["ratio"] - proper["trapezoid"]["ratio"]
    ) / max(proper["simpson"]["ratio"], proper["trapezoid"]["ratio"], 1e-300)
    area = _integral(weight, z_open, "simpson")
    face_rms = proper["simpson"]["numerator"] / np.sqrt(area)
    before_face_rms = proper["simpson"]["before_norm"] / np.sqrt(area)
    after_face_rms = proper["simpson"]["after_norm"] / np.sqrt(area)
    reference_open = np.asarray(reference_acceleration)[1:-1, -1, SCALARS]
    reference_rms = weighted_norm(
        reference_open, weight, z_open, "simpson",
    ) / np.sqrt(area)
    collar = collar_rms(before, position, z, radius, stencil_width)
    collar_scale = max(before_face_rms, after_face_rms, reference_rms, collar)
    component_legacy = []
    component_proper = []
    component_proper_trapezoid = []
    for field in range(2):
        component_legacy.append(symmetric_ratio(
            correction[:, field], open_after[:, field], open_before[:, field],
        ))
        component_proper.append(weighted_symmetric_ratio(
            correction[:, field], open_after[:, field], open_before[:, field],
            weight, z_open, "simpson",
        ))
        component_proper_trapezoid.append(weighted_symmetric_ratio(
            correction[:, field], open_after[:, field], open_before[:, field],
            weight, z_open, "trapezoid",
        ))
    pointwise = pointwise_symmetric_ratio(correction, open_after, open_before)
    independent_difference = open_after - independent_target
    target_scale = max(np.linalg.norm(open_after), np.linalg.norm(independent_target), 1e-300)
    characteristic_closure = term_a + term_v + term_c + correction
    characteristic_scale = max(
        np.linalg.norm(term_a) + np.linalg.norm(term_v) + np.linalg.norm(term_c),
        np.linalg.norm(correction), 1e-300,
    )
    return {
        "legacy_ratio": symmetric_ratio(correction, open_after, open_before),
        "legacy_numerator": float(np.linalg.norm(correction)),
        "legacy_before_norm": float(np.linalg.norm(open_before)),
        "legacy_after_norm": float(np.linalg.norm(open_after)),
        "proper": proper,
        "quadrature_relative_difference": float(quadrature_relative),
        "component_legacy_ratios": component_legacy,
        "component_proper_ratios": component_proper,
        "component_proper_trapezoid_ratios": component_proper_trapezoid,
        "pointwise": pointwise,
        "maximum_absolute_correction": float(np.max(np.abs(correction))),
        "face_rms_correction": face_rms,
        "face_rms_before": before_face_rms,
        "face_rms_after": after_face_rms,
        "face_rms_reference": reference_rms,
        "collar_rms_before": collar,
        "collar_ratio": face_rms / max(collar_scale, 1e-300),
        "independent_target_relative_difference": float(
            np.linalg.norm(independent_difference) / target_scale
        ),
        "independent_target_maximum_absolute_difference": float(
            np.max(np.abs(independent_difference))
        ),
        "characteristic_closure_relative": float(
            np.linalg.norm(characteristic_closure) / characteristic_scale
        ),
        "post_correction_relative_residual": float(
            np.linalg.norm(independent_difference) / max(
                np.linalg.norm(open_after), np.linalg.norm(independent_target), 1e-300,
            )
        ),
        "weight_minimum": float(np.min(weight)),
        "weight_maximum": float(np.max(weight)),
    }


def normalized_radial_difference(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError("radial difference arrays are not aligned")
    numerator = np.max(np.abs(left - right), axis=(0, 2))
    scale = np.maximum(
        np.maximum(np.max(np.abs(left), axis=(0, 2)), np.max(np.abs(right), axis=(0, 2))),
        1.0,
    )
    return numerator / scale


def classify_test10c(valid, contamination, indeterminate, normalization, model_response):
    if not valid:
        return "review", "invalid_outer_scalar_audit"
    if contamination:
        return "fail", "resolved_common_interior_boundary_contamination"
    if indeterminate:
        return "review", "mixed_outer_scalar_diagnosis"
    if normalization and not model_response:
        return "pass", "legacy_normalization_inconsistency"
    if model_response and not normalization:
        return "pass", "genuine_boundary_local_response"
    return "review", "mixed_outer_scalar_diagnosis"
