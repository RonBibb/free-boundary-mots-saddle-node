"""Pure helpers for Protocol 250's causal-signature comparison."""

from __future__ import annotations

import numpy as np


def embedded_curve(theta, rho, slope):
    theta = np.asarray(theta, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    slope = np.asarray(slope, dtype=np.float64)
    if theta.ndim != 1 or rho.shape != theta.shape or slope.shape != theta.shape:
        raise ValueError("profile arrays must be matching one-dimensional arrays")
    if theta.size < 3 or not np.all(np.isfinite(theta + rho + slope)):
        raise ValueError("profile arrays are invalid")
    coordinates = np.stack((-rho * np.cos(theta), rho * np.sin(theta)), axis=1)
    tangent = np.stack((
        rho * np.sin(theta) - slope * np.cos(theta),
        rho * np.cos(theta) + slope * np.sin(theta),
    ), axis=1)
    return coordinates, tangent


def projected_tube_norm(lapse, shift_covector, shift, metric, velocity, tangent):
    lapse = np.asarray(lapse, dtype=np.float64)
    shift_covector = np.asarray(shift_covector, dtype=np.float64)
    shift = np.asarray(shift, dtype=np.float64)
    metric = np.asarray(metric, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    tangent = np.asarray(tangent, dtype=np.float64)
    count = lapse.shape[0]
    if (
        lapse.ndim != 1 or shift_covector.shape != (count, 2)
        or shift.shape != (count, 2) or metric.shape != (count, 2, 2)
        or velocity.shape != (count, 2) or tangent.shape != (count, 2)
    ):
        raise ValueError("tube metric arrays have incompatible shapes")
    if not all(np.all(np.isfinite(value)) for value in (lapse, shift_covector, shift, metric, velocity, tangent)):
        raise ValueError("tube metric arrays must be finite")
    tangent_norm = np.einsum("...a,...ab,...b->...", tangent, metric, tangent)
    if np.any(lapse <= 0) or np.any(tangent_norm <= 0):
        raise ValueError("lapse and leaf tangent norm must be positive")
    g_tt = -lapse**2 + np.einsum("...a,...a->...", shift_covector, shift)
    v_norm = (
        g_tt + 2.0 * np.einsum("...a,...a->...", shift_covector, velocity)
        + np.einsum("...a,...ab,...b->...", velocity, metric, velocity)
    )
    v_dot_tangent = (
        np.einsum("...a,...a->...", shift_covector, tangent)
        + np.einsum("...a,...ab,...b->...", velocity, metric, tangent)
    )
    return v_norm - v_dot_tangent**2 / tangent_norm


def causal_resolution(backward, centered, forward):
    backward = np.asarray(backward, dtype=np.float64)
    centered = np.asarray(centered, dtype=np.float64)
    forward = np.asarray(forward, dtype=np.float64)
    if backward.ndim != 1 or centered.shape != backward.shape or forward.shape != backward.shape:
        raise ValueError("causal profiles must be matching one-dimensional arrays")
    if not all(np.all(np.isfinite(value)) for value in (backward, centered, forward)):
        raise ValueError("causal profiles must be finite")
    signs = np.stack((np.sign(backward), np.sign(centered), np.sign(forward)))
    sign_agreement = np.all(signs == signs[1:2], axis=0) & (signs[1] != 0)
    spread = np.maximum(np.abs(backward - centered), np.abs(forward - centered))
    resolved = sign_agreement & (np.abs(centered) > spread)
    if np.all(resolved) and np.all(centered > 0):
        label = "UNIFORMLY-SPACELIKE"
    elif np.all(resolved) and np.all(centered < 0):
        label = "UNIFORMLY-TIMELIKE"
    else:
        label = "MIXED-OR-UNRESOLVED"
    return {
        "label": label,
        "node_count": int(centered.size),
        "resolved_node_count": int(np.count_nonzero(resolved)),
        "resolved_fraction": float(np.mean(resolved)),
        "centered_minimum": float(np.min(centered)),
        "centered_maximum": float(np.max(centered)),
        "maximum_one_sided_spread": float(np.max(spread)),
        "minimum_centered_absolute_norm": float(np.min(np.abs(centered))),
        "minimum_resolution_margin": float(np.min(np.abs(centered) - spread)),
    }, resolved, spread


def relative_difference(first, second, floor=1e-12):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("comparison arrays must have matching one-dimensional shapes")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("comparison arrays must be finite")
    scale = np.maximum(np.maximum(np.abs(first), np.abs(second)), float(floor))
    return np.abs(first - second) / scale


def compare_norm_records(full, half, limit=0.01):
    if set(full) != {"backward", "centered", "forward"} or set(half) != set(full):
        raise ValueError("causal path inventory differs")
    paths = {}
    for name in ("backward", "centered", "forward"):
        values = relative_difference(full[name], half[name])
        paths[name] = {
            "maximum_relative_difference": float(np.max(values)),
            "rms_relative_difference": float(np.sqrt(np.mean(values**2))),
            "passed": bool(np.max(values) < limit),
        }
    return {
        "paths": paths,
        "maximum_relative_difference": float(max(item["maximum_relative_difference"] for item in paths.values())),
        "passed": bool(all(item["passed"] for item in paths.values())),
    }


def classify(full_spacelike, half_spacelike, temporal_consistency, inherited_spatial, inherited_balance):
    if not full_spacelike:
        return "FULL-DT-CAUSAL-SIGNATURE-FAIL"
    if not half_spacelike:
        return "HALF-DT-CAUSAL-SIGNATURE-FAIL"
    if not temporal_consistency:
        return "FULL-HALF-CAUSAL-SIGNATURE-INCONSISTENT"
    if not inherited_spatial:
        return "THREE-GRID-SPATIAL-PREREQUISITE-FAIL"
    if not inherited_balance:
        return "FINITE-SEGMENT-BALANCE-PREREQUISITE-FAIL"
    return "G10-FULL-HALF-CAUSAL-SIGNATURE-CONSISTENCY-PASS"
