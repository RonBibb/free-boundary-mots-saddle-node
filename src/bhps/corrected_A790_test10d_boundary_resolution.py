"""Prospectively sealed refined face integration for Test 10D."""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import romb
from scipy.interpolate import CubicSpline, PchipInterpolator


SCHEMES = ("pchip_gauss", "natural_cubic_romberg")
LEVELS = (4, 8, 16)
METRIC_KEYS = (
    "area",
    "norm_delta",
    "norm_before",
    "norm_after",
    "norm_reference",
    "norm_term_a",
    "norm_term_v",
    "norm_term_c",
    "norm_term_sum",
    "proper_ratio",
    "term_balance_ratio",
    "component_phi_ratio",
    "component_chi_ratio",
    "face_rms_delta",
    "face_rms_before",
    "face_rms_after",
    "face_rms_reference",
    "collar_ratio",
)


def dual_close(left, right, absolute=1e-13, relative=1e-6):
    """Symmetric absolute-plus-relative comparison without a scale floor."""
    left = float(left)
    right = float(right)
    return abs(left - right) <= absolute + relative * max(abs(left), abs(right))


def closure_gate(residual, scale, absolute=1e-18, relative=1e-10):
    residual = float(residual)
    scale = float(scale)
    if not np.isfinite(residual) or not np.isfinite(scale) or residual < 0.0 or scale < 0.0:
        return False
    return residual <= absolute + relative * scale


def _interpolator(scheme, z, values):
    if scheme == "pchip_gauss":
        return PchipInterpolator(z, values, axis=0, extrapolate=False)
    if scheme == "natural_cubic_romberg":
        return CubicSpline(z, values, axis=0, bc_type="natural", extrapolate=False)
    raise ValueError("unknown refined-face evaluator")


def _gauss_integral(interpolants, z, level, radius):
    nodes, weights = leggauss(int(level))
    total = None
    for left, right in zip(z[:-1], z[1:]):
        midpoint = 0.5 * (left + right)
        halfwidth = 0.5 * (right - left)
        x = midpoint + halfwidth * nodes
        values = _densities(interpolants, x, radius)
        local = halfwidth * np.tensordot(weights, values, axes=(0, 0))
        total = local if total is None else total + local
    return total


def _romberg_integral(interpolants, z, level, radius):
    total = None
    for left, right in zip(z[:-1], z[1:]):
        x = np.linspace(left, right, int(level) + 1)
        values = _densities(interpolants, x, radius)
        local = romb(values, dx=(right - left) / int(level), axis=0)
        total = local if total is None else total + local
    return total


FIELD_ORDER = (
    "delta", "before", "after", "reference", "term_a", "term_v",
    "term_c", "term_sum",
)


def _densities(interpolants, x, radius):
    perpendicular = np.asarray(interpolants["q_perp"](x), dtype=float)
    compact = np.asarray(interpolants["q_zz"](x), dtype=float)
    if (
        np.any(~np.isfinite(perpendicular)) or np.any(~np.isfinite(compact))
        or np.any(perpendicular <= 0.0) or np.any(compact <= 0.0)
    ):
        raise ValueError("refined outer-face measure is not positive and finite")
    weight = 4.0 * math.pi * float(radius) ** 2 * perpendicular * np.sqrt(compact)
    columns = [weight]
    for key in FIELD_ORDER:
        field = np.asarray(interpolants[key](x), dtype=float)
        if field.shape != (len(np.atleast_1d(x)), 2) or np.any(~np.isfinite(field)):
            raise ValueError("invalid refined scalar primitive")
        columns.extend((weight * field[:, 0] ** 2, weight * field[:, 1] ** 2))
    return np.column_stack(columns)


def _metrics_from_integrals(integrals, collar_rms):
    values = np.asarray(integrals, dtype=float)
    if values.shape != (1 + 2 * len(FIELD_ORDER),) or np.any(~np.isfinite(values)):
        raise ValueError("invalid refined face integrals")
    if values[0] <= 0.0 or np.any(values[1:] < -1e-24):
        raise ValueError("refined face integral is nonpositive")
    values[1:] = np.maximum(values[1:], 0.0)
    area = float(values[0])
    component_integrals = {
        key: values[1 + 2 * index: 3 + 2 * index]
        for index, key in enumerate(FIELD_ORDER)
    }
    component_norms = {
        key: np.sqrt(component_integrals[key]) for key in FIELD_ORDER
    }
    norms = {key: float(np.linalg.norm(component_norms[key])) for key in FIELD_ORDER}
    denominator = max(norms["before"], norms["after"])
    if denominator == 0.0:
        proper_ratio = 0.0 if norms["delta"] == 0.0 else np.inf
    else:
        proper_ratio = norms["delta"] / denominator
    term_denominator = norms["term_a"] + norms["term_v"] + norms["term_c"]
    if term_denominator == 0.0:
        term_balance = 0.0 if norms["term_sum"] == 0.0 else np.inf
    else:
        term_balance = norms["term_sum"] / term_denominator
    component_ratios = []
    for component in range(2):
        scale = max(component_norms["before"][component], component_norms["after"][component])
        numerator = component_norms["delta"][component]
        component_ratios.append(0.0 if scale == 0.0 and numerator == 0.0 else numerator / scale)
    root_area = math.sqrt(area)
    rms = {key: norms[key] / root_area for key in ("delta", "before", "after", "reference")}
    collar_scale = max(
        rms["before"], rms["after"], rms["reference"], float(collar_rms),
    )
    collar_ratio = 0.0 if collar_scale == 0.0 and rms["delta"] == 0.0 else rms["delta"] / collar_scale
    return {
        "area": area,
        **{f"norm_{key}": value for key, value in norms.items()},
        "proper_ratio": float(proper_ratio),
        "term_balance_ratio": float(term_balance),
        "component_phi_ratio": float(component_ratios[0]),
        "component_chi_ratio": float(component_ratios[1]),
        "face_rms_delta": float(rms["delta"]),
        "face_rms_before": float(rms["before"]),
        "face_rms_after": float(rms["after"]),
        "face_rms_reference": float(rms["reference"]),
        "collar_ratio": float(collar_ratio),
    }


def evaluate_refined_face(z_open, radius, q_perp, q_zz, fields, collar_rms, scheme, level):
    z = np.asarray(z_open, dtype=float)
    if z.ndim != 1 or len(z) < 5 or np.any(np.diff(z) <= 0.0):
        raise ValueError("invalid open-face coordinate")
    primitives = {
        "q_perp": np.asarray(q_perp, dtype=float),
        "q_zz": np.asarray(q_zz, dtype=float),
    }
    for key in FIELD_ORDER:
        primitives[key] = np.asarray(fields[key], dtype=float)
    if primitives["q_perp"].shape != z.shape or primitives["q_zz"].shape != z.shape:
        raise ValueError("face metric primitives are not aligned")
    for key in FIELD_ORDER:
        if primitives[key].shape != (len(z), 2):
            raise ValueError("face scalar primitives are not aligned")
    interpolants = {
        key: _interpolator(scheme, z, value) for key, value in primitives.items()
    }
    if scheme == "pchip_gauss":
        integrals = _gauss_integral(interpolants, z, level, radius)
    elif scheme == "natural_cubic_romberg":
        integrals = _romberg_integral(interpolants, z, level, radius)
    else:
        raise ValueError("unknown refined-face evaluator")
    result = _metrics_from_integrals(integrals, collar_rms)
    return {key: result[key] for key in METRIC_KEYS}


def evaluate_all_levels(z_open, radius, q_perp, q_zz, fields, collar_rms):
    records = {}
    for scheme in SCHEMES:
        records[scheme] = {
            level: evaluate_refined_face(
                z_open, radius, q_perp, q_zz, fields, collar_rms, scheme, level,
            ) for level in LEVELS
        }
    return records


def refinement_flags(records):
    flags = {}
    for scheme in SCHEMES:
        flags[scheme] = {
            key: dual_close(records[scheme][16][key], records[scheme][8][key])
            for key in METRIC_KEYS
        }
    flags["cross_scheme"] = {
        key: dual_close(
            records[SCHEMES[0]][16][key], records[SCHEMES[1]][16][key],
            absolute=1e-12, relative=0.02,
        ) for key in METRIC_KEYS
    }
    return flags


def ensemble_metrics(records):
    return {
        key: 0.5 * (
            records[SCHEMES[0]][16][key] + records[SCHEMES[1]][16][key]
        ) for key in METRIC_KEYS
    }


def pointwise_ratio(delta, before, after):
    delta = np.abs(np.asarray(delta, dtype=float))
    scale = np.maximum(np.abs(before), np.abs(after))
    ratio = np.zeros_like(delta)
    nonzero = scale > 0.0
    if np.any(~nonzero & (delta > 0.0)):
        raise ValueError("pointwise response has a zero scale")
    ratio[nonzero] = delta[nonzero] / scale[nonzero]
    return float(np.max(ratio))


def source_grid_close(left, right, pointwise=False):
    if pointwise:
        return dual_close(left, right, absolute=0.05, relative=0.10)
    return dual_close(left, right, absolute=0.005, relative=0.25)


def classify_test10d(
    valid, contamination, indeterminate, normalization, response,
    source_grid_consistent,
):
    if not valid:
        return "review", "invalid_refined_boundary_audit"
    if contamination:
        return "fail", "resolved_common_interior_boundary_contamination"
    if indeterminate:
        return "review", "mixed_refined_boundary_diagnosis"
    if normalization and source_grid_consistent:
        return "pass", "converged_legacy_normalization_artifact"
    if response and source_grid_consistent:
        return "pass", "converged_boundary_local_no_resolved_interior_effect"
    if response and not source_grid_consistent:
        return "review", "boundary_local_response_not_source_grid_converged"
    return "review", "mixed_refined_boundary_diagnosis"
