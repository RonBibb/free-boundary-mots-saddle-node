"""Pure helpers for Protocol 234 null-expansion and area-law gates."""

from __future__ import annotations

import numpy as np


def null_expansions(mean_curvature, extrinsic_correction):
    """Return future outward and inward null expansions.

    The null normals are proportional to ``u+s`` and ``u-s``. Positive
    rescalings do not affect the sign test used by this protocol.
    """
    mean_curvature = np.asarray(mean_curvature, dtype=np.float64)
    extrinsic_correction = np.asarray(extrinsic_correction, dtype=np.float64)
    if mean_curvature.shape != extrinsic_correction.shape or mean_curvature.ndim != 1:
        raise ValueError("curvature arrays must be matching one-dimensional arrays")
    if not np.all(np.isfinite(mean_curvature)) or not np.all(np.isfinite(extrinsic_correction)):
        raise ValueError("curvature arrays must be finite")
    return mean_curvature + extrinsic_correction, -mean_curvature + extrinsic_correction


def negative_stencil_resolution(theta_minus_5, theta_minus_7, theta_minus_9, mask):
    """Apply the prospectively fixed resolved-negative stencil rule."""
    values = [np.asarray(value, dtype=np.float64) for value in (theta_minus_5, theta_minus_7, theta_minus_9)]
    mask = np.asarray(mask)
    if any(value.ndim != 1 for value in values) or not (values[0].shape == values[1].shape == values[2].shape):
        raise ValueError("inward-expansion arrays must have matching one-dimensional shapes")
    if mask.shape != values[0].shape or mask.dtype != np.bool_:
        raise ValueError("interior mask must be a matching Boolean array")
    if not np.any(mask) or not all(np.all(np.isfinite(value)) for value in values):
        raise ValueError("inward-expansion data are empty or nonfinite")
    spread = np.maximum(np.abs(values[0] - values[1]), np.abs(values[2] - values[1]))
    all_negative = (values[0] < 0) & (values[1] < 0) & (values[2] < 0)
    resolved = all_negative & (np.abs(values[1]) > spread)
    interior_resolved = resolved[mask]
    ratio = np.abs(values[1][mask]) / np.maximum(spread[mask], np.finfo(np.float64).tiny)
    summary = {
        "passed": bool(np.all(interior_resolved)),
        "interior_node_count": int(np.count_nonzero(mask)),
        "resolved_interior_node_count": int(np.count_nonzero(interior_resolved)),
        "minimum_theta_minus_7": float(np.min(values[1][mask])),
        "maximum_theta_minus_7": float(np.max(values[1][mask])),
        "maximum_theta_minus_any_stencil": float(max(np.max(value[mask]) for value in values)),
        "maximum_stencil_spread": float(np.max(spread[mask])),
        "minimum_pointwise_margin_ratio": float(np.min(ratio)),
    }
    return summary, resolved, spread


def strict_area_trend(times, areas):
    """Return strict adjacent area increments in increasing time order."""
    times = np.asarray(times, dtype=np.float64)
    areas = np.asarray(areas, dtype=np.float64)
    if times.ndim != 1 or areas.shape != times.shape or len(times) < 2:
        raise ValueError("times and areas must be matching one-dimensional arrays")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(areas)) or np.any(np.diff(times) <= 0):
        raise ValueError("times and areas must be finite and time ordered")
    increments = np.diff(areas)
    return {
        "strictly_increasing": bool(np.all(increments > 0)),
        "strictly_decreasing": bool(np.all(increments < 0)),
        "increments": [float(value) for value in increments],
        "total_change": float(areas[-1] - areas[0]),
        "relative_total_change": float((areas[-1] - areas[0]) / max(abs(areas[0]), abs(areas[-1]), 1e-300)),
    }
