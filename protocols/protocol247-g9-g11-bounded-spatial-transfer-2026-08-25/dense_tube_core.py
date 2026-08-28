"""Pure classification helpers for Protocol 244."""

from __future__ import annotations

import numpy as np


def validate_step_schedule(checkpoint_steps, leaf_steps, control_steps):
    checkpoints = tuple(int(value) for value in checkpoint_steps)
    leaves = tuple(int(value) for value in leaf_steps)
    controls = tuple(int(value) for value in control_steps)
    if checkpoints != tuple(range(33, 49)):
        raise ValueError("checkpoint schedule differs")
    if leaves != tuple(range(38, 49)):
        raise ValueError("leaf schedule differs")
    if controls != (48,) or not set(controls) <= set(leaves):
        raise ValueError("control schedule differs")
    return True


def adjacent_profile_guard(profiles, ceiling=0.05):
    """Check time-ordered profile continuity without interpreting it as error."""
    values = [np.asarray(value, dtype=np.float64) for value in profiles]
    if len(values) < 2 or any(value.ndim != 1 for value in values):
        raise ValueError("profiles must be a nontrivial one-dimensional sequence")
    if any(value.shape != values[0].shape or not np.all(np.isfinite(value)) for value in values):
        raise ValueError("profile shapes or values differ")
    differences = np.asarray(
        [np.max(np.abs(right - left)) for left, right in zip(values[:-1], values[1:])],
        dtype=np.float64,
    )
    return {
        "passed": bool(np.all(differences < float(ceiling))),
        "ceiling": float(ceiling),
        "maximum_adjacent_absolute_difference": float(np.max(differences)),
        "adjacent_absolute_differences": [float(value) for value in differences],
    }


def strict_area_increase(steps, areas):
    steps = np.asarray(steps)
    areas = np.asarray(areas, dtype=np.float64)
    if steps.ndim != 1 or areas.shape != steps.shape or len(steps) < 2:
        raise ValueError("steps and areas must be matching sequences")
    if not np.issubdtype(steps.dtype, np.integer) or np.any(np.diff(steps) <= 0):
        raise ValueError("steps must be strictly increasing integers")
    if not np.all(np.isfinite(areas)):
        raise ValueError("areas must be finite")
    increments = np.diff(areas)
    return {
        "passed": bool(np.all(increments > 0.0)),
        "increments": [float(value) for value in increments],
        "minimum_increment": float(np.min(increments)),
        "total_change": float(areas[-1] - areas[0]),
        "relative_total_change": float(
            (areas[-1] - areas[0]) / max(abs(areas[0]), abs(areas[-1]), 1e-300)
        ),
    }


def classify_dense_tube(surface_stability, inward, area, signature):
    gates = {
        "surface_stability": bool(surface_stability),
        "inward_expansion": bool(inward),
        "area_increase": bool(area),
        "tube_signature": bool(signature),
    }
    if not gates["surface_stability"]:
        classification = "DENSE-OUTER-SURFACE-OR-STABILITY-FAIL"
    elif not gates["inward_expansion"]:
        classification = "DENSE-INWARD-EXPANSION-FAIL"
    elif not gates["area_increase"]:
        classification = "DENSE-AREA-INCREASE-FAIL"
    elif not gates["tube_signature"]:
        classification = "DENSE-TUBE-SIGNATURE-UNRESOLVED"
    else:
        classification = "FULL-DT-DENSE-G10-OUTER-MARGINAL-TUBE-PASS"
    return classification, gates
