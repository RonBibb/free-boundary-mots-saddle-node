"""Pure numerical helpers for the Protocol 233 sparse tube diagnostic."""

from __future__ import annotations

import numpy as np


def embedded_curve(theta, rho, slope):
    """Return ``(z offset, r)`` and the meridional tangent.

    The absolute bulk coordinate is ``z_b + z_offset``.  The fixed wall
    location cancels from time differences, so keeping the offset makes the
    helper independent of a particular compactification length.
    """
    theta = np.asarray(theta, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    slope = np.asarray(slope, dtype=np.float64)
    if theta.ndim != 1 or rho.shape != theta.shape or slope.shape != theta.shape:
        raise ValueError("profile arrays must be matching one-dimensional arrays")
    if len(theta) < 3 or not np.all(np.isfinite(theta + rho + slope)):
        raise ValueError("profile arrays are invalid")
    coordinates = np.stack((-rho * np.cos(theta), rho * np.sin(theta)), axis=1)
    tangent = np.stack((
        rho * np.sin(theta) - slope * np.cos(theta),
        rho * np.cos(theta) + slope * np.sin(theta),
    ), axis=1)
    return coordinates, tangent


def projected_tube_norm(lapse, shift_covector, shift, metric, velocity, tangent):
    """Return the normal-plane norm of a tube tangent.

    ``velocity`` is the spatial coordinate velocity at fixed leaf label and
    ``tangent`` is the spatial meridional leaf tangent.  The projection removes
    arbitrary tangential relabeling along the leaf.
    """
    lapse = np.asarray(lapse, dtype=np.float64)
    shift_covector = np.asarray(shift_covector, dtype=np.float64)
    shift = np.asarray(shift, dtype=np.float64)
    metric = np.asarray(metric, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    tangent = np.asarray(tangent, dtype=np.float64)
    count = lapse.shape[0]
    expected_vector = (count, 2)
    if (
        lapse.ndim != 1
        or shift_covector.shape != expected_vector
        or shift.shape != expected_vector
        or metric.shape != (count, 2, 2)
        or velocity.shape != expected_vector
        or tangent.shape != expected_vector
    ):
        raise ValueError("tube metric arrays have incompatible shapes")
    values = (lapse, shift_covector, shift, metric, velocity, tangent)
    if not all(np.all(np.isfinite(value)) for value in values):
        raise ValueError("tube metric arrays must be finite")
    spatial_tangent_norm = np.einsum("...a,...ab,...b->...", tangent, metric, tangent)
    if np.any(lapse <= 0) or np.any(spatial_tangent_norm <= 0):
        raise ValueError("lapse and leaf tangent norm must be positive")
    g_tt = -lapse**2 + np.einsum("...a,...a->...", shift_covector, shift)
    v_norm = (
        g_tt
        + 2.0 * np.einsum("...a,...a->...", shift_covector, velocity)
        + np.einsum("...a,...ab,...b->...", velocity, metric, velocity)
    )
    v_dot_tangent = (
        np.einsum("...a,...a->...", shift_covector, tangent)
        + np.einsum("...a,...ab,...b->...", velocity, metric, tangent)
    )
    return v_norm - v_dot_tangent**2 / spatial_tangent_norm


def causal_resolution(backward, centered, forward):
    """Classify a sparse three-estimate causal-norm comparison."""
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
        label = "UNIFORMLY-SPACELIKE-SPARSE-PILOT"
    elif np.all(resolved) and np.all(centered < 0):
        label = "UNIFORMLY-TIMELIKE-SPARSE-PILOT"
    else:
        label = "MIXED-OR-UNRESOLVED-SPARSE-PILOT"
    return {
        "label": label,
        "resolved_node_count": int(np.count_nonzero(resolved)),
        "node_count": int(centered.size),
        "resolved_fraction": float(np.mean(resolved)),
        "centered_minimum": float(np.min(centered)),
        "centered_maximum": float(np.max(centered)),
        "maximum_one_sided_spread": float(np.max(spread)),
        "minimum_centered_absolute_norm": float(np.min(np.abs(centered))),
    }, resolved, spread
