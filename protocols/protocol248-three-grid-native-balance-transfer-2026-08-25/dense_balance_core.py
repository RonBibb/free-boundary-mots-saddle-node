"""Pure helpers and ordered gates for Protocol 245."""

from __future__ import annotations

import math

import numpy as np


SQRT2 = math.sqrt(2.0)
GATE_ORDER = ("orientation", "area", "seam", "native_wall_rate", "flux_ledger")


def three_point_rates(early, middle, late, dt):
    early = np.asarray(early, dtype=np.float64)
    middle = np.asarray(middle, dtype=np.float64)
    late = np.asarray(late, dtype=np.float64)
    dt = float(dt)
    if early.shape != middle.shape or middle.shape != late.shape:
        raise ValueError("three-point values must have matching shapes")
    if dt <= 0.0 or not all(np.all(np.isfinite(value)) for value in (early, middle, late)):
        raise ValueError("three-point data must be finite and dt positive")
    return {
        "backward": (middle - early) / dt,
        "centered": (late - early) / (2.0 * dt),
        "forward": (late - middle) / dt,
    }


def projected_norm(lapse, shift_covector, shift, metric, velocity, tangent):
    lapse = np.asarray(lapse, dtype=float)
    shift_covector = np.asarray(shift_covector, dtype=float)
    shift = np.asarray(shift, dtype=float)
    metric = np.asarray(metric, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    tangent = np.asarray(tangent, dtype=float)
    tangent_norm = np.einsum("...a,...ab,...b->...", tangent, metric, tangent)
    if np.any(tangent_norm <= 0.0):
        raise ValueError("tangent norm must be positive")
    gtt = -lapse**2 + np.einsum("...a,...a->...", shift_covector, shift)
    full = (
        gtt
        + 2.0 * np.einsum("...a,...a->...", shift_covector, velocity)
        + np.einsum("...a,...ab,...b->...", velocity, metric, velocity)
    )
    cross = (
        np.einsum("...a,...a->...", shift_covector, tangent)
        + np.einsum("...a,...ab,...b->...", velocity, metric, tangent)
    )
    return full - cross**2 / tangent_norm


def null_decomposition(lapse, shift, metric, velocity, tangent):
    lapse = np.asarray(lapse, dtype=float)
    shift = np.asarray(shift, dtype=float)
    metric = np.asarray(metric, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    tangent = np.asarray(tangent, dtype=float)
    inverse = np.linalg.inv(metric)
    conormal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
    conormal /= np.sqrt(np.einsum("...a,...ab,...b->...", conormal, inverse, conormal))[:, None]
    normal_speed = np.einsum("...a,...a->...", conormal, shift + velocity)
    a = (lapse + normal_speed) / SQRT2
    b = (normal_speed - lapse) / SQRT2
    return {"normal_speed": normal_speed, "A": a, "B": b, "norm": 2.0 * a * b}


def relative_scale_error(left, right, scale=0.0):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    scale = float(scale)
    return float(
        np.max(np.abs(left - right))
        / max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), abs(scale), 1e-300)
    )


def normalized_ledger_residual(target, terms):
    target = float(target)
    clean = {str(name): float(value) for name, value in terms.items()}
    if not math.isfinite(target) or not clean or not all(math.isfinite(value) for value in clean.values()):
        raise ValueError("ledger values must be finite and nonempty")
    total = float(sum(clean.values()))
    residual = float(target - total)
    norm = float(max(abs(target), sum(abs(value) for value in clean.values()), 1e-12))
    return {
        "target_rate": target,
        "terms": clean,
        "total_flux": total,
        "signed_residual": residual,
        "balance_norm": norm,
        "normalized_absolute_residual": float(abs(residual) / norm),
    }


def inherited_rate_pass(value, expected):
    value = float(value)
    expected = float(expected)
    error = abs(value - expected) / max(abs(value), abs(expected), 0.005, 1e-300)
    return bool(error < 0.05 or abs(value - expected) < 0.005)


def centered_directional(plus, minus, epsilon):
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    return float((float(plus) - float(minus)) / (2.0 * epsilon))


def classify(gates):
    if set(gates) != set(GATE_ORDER) or any(type(gates[name]) is not bool for name in GATE_ORDER):
        raise ValueError("exact Boolean gate inventory required")
    failure_labels = {
        "orientation": "FULL-DT-DENSE-LOCAL-ORIENTATION-FAIL",
        "area": "FULL-DT-DENSE-LOCAL-AREA-TRANSPORT-FAIL",
        "seam": "FULL-DT-DENSE-NATIVE-SEAM-GEOMETRY-FAIL",
        "native_wall_rate": "FULL-DT-DENSE-NATIVE-WALL-RATE-FAIL",
        "flux_ledger": "FULL-DT-DENSE-LOCAL-FLUX-LEDGER-FAIL",
    }
    for name in GATE_ORDER:
        if not gates[name]:
            return failure_labels[name]
    return "FULL-DT-DENSE-NATIVE-OPERATOR-LOCAL-BALANCE-PASS"
