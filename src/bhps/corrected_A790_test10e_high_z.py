"""Prospective convergence predicates for the sealed A=7.90 Test 10E audit."""

from __future__ import annotations

import numpy as np

from bhps.corrected_A790_physical_tensor_convergence import four_grid_orders


SOURCE_METRICS = (
    "proper_ratio", "collar_ratio", "component_phi_ratio",
    "pointwise_ratio", "maximum_absolute_correction",
)

NOISE_FLOORS = {
    "proper_ratio": 2.5e-4,
    "collar_ratio": 2.5e-4,
    "component_phi_ratio": 2.5e-4,
    "pointwise_ratio": 5e-3,
    "maximum_absolute_correction": 1e-11,
}

ORDER_MINIMUM = {
    "proper_ratio": 1.0,
    "collar_ratio": 1.0,
    "component_phi_ratio": 1.0,
    "pointwise_ratio": 0.5,
    "maximum_absolute_correction": 1.0,
}


def dual_close(left, right, absolute, relative):
    """Symmetric absolute-plus-relative gate used by all separation tests."""
    left = float(left)
    right = float(right)
    return bool(abs(left - right) <= absolute + relative * max(abs(left), abs(right)))


def consistency_close(metric, left, right, relative=0.15):
    absolute = {
        "proper_ratio": 0.0025,
        "collar_ratio": 0.0025,
        "component_phi_ratio": 0.0025,
        "pointwise_ratio": 0.05,
        "maximum_absolute_correction": 1e-11,
    }[metric]
    return dual_close(left, right, absolute, relative)


def source_sequence(metric, maxima, intervals=(80, 96, 112, 128)):
    """Score one G7/G8/G9/G10 run-maximum sequence prospectively."""
    values = np.asarray(maxima, dtype=float)
    if values.shape != (4,) or np.any(~np.isfinite(values)):
        raise ValueError("source sequence requires four finite maxima")
    differences = np.abs(np.diff(values))
    floor = NOISE_FLOORS[metric]
    resolved_at_floor = bool(differences[1] < floor and differences[2] < floor)
    orders = four_grid_orders(differences, intervals)
    fine_order = orders["fine_triplet_order"]
    monotonic = bool(differences[2] <= 1e-12 + differences[1])
    order_pass = bool(
        resolved_at_floor
        or (fine_order is not None and fine_order >= ORDER_MINIMUM[metric])
    )
    consistency = consistency_close(metric, values[2], values[3])
    return {
        "values": values.tolist(),
        "differences": {
            "D78": float(differences[0]),
            "D89": float(differences[1]),
            "D910": float(differences[2]),
        },
        "orders": orders,
        "noise_floor": floor,
        "resolved_at_floor": resolved_at_floor,
        "fine_monotonic": monotonic,
        "fine_order_pass": order_pass,
        "G9_G10_consistency": consistency,
        "passes": bool(monotonic and order_pass and consistency),
    }


def stage_history(metric, histories):
    """Apply the sealed active-stage monotonicity rule to four histories."""
    values = np.asarray(histories, dtype=float)
    if values.ndim != 2 or values.shape[0] != 4 or np.any(~np.isfinite(values)):
        raise ValueError("stage history must be four aligned finite histories")
    active = np.max(values, axis=0) >= 1e-12
    difference_89 = np.abs(values[2] - values[1])
    difference_910 = np.abs(values[3] - values[2])
    floor = NOISE_FLOORS[metric]
    ordinary = difference_910 <= floor + difference_89
    adverse = difference_910 > floor + 1.25 * difference_89
    active_count = int(np.sum(active))
    ordinary_fraction = (
        float(np.mean(ordinary[active])) if active_count else 1.0
    )
    adverse_count = int(np.sum(adverse & active))
    return {
        "active_stage_count": active_count,
        "ordinary_monotonic_fraction": ordinary_fraction,
        "adverse_stage_count": adverse_count,
        "passes": bool(ordinary_fraction >= 0.90 and adverse_count == 0),
    }


def scheme_spread(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-300)
    return float(np.max(np.abs(left - right) / scale))


def spread_gate(g9, g10):
    g9 = float(g9)
    g10 = float(g10)
    return {
        "G9": g9,
        "G10": g10,
        "nonincreasing_with_slack": bool(g10 <= g9 + 1e-4),
        "G10_below_two_percent": bool(g10 < 0.02),
        "passes": bool(g10 <= g9 + 1e-4 and g10 < 0.02),
    }


def separation_record(metric, full, control, temporal=False):
    relative = 0.10 if temporal else 0.15
    return {
        "full": float(full),
        "control": float(control),
        "absolute_difference": float(abs(float(full) - float(control))),
        "passes": consistency_close(metric, full, control, relative=relative),
    }


def classify_test10e(
    valid, uncontrolled, contamination, source_converged, radial_separated,
    temporal_separated, response, normalization,
):
    if not valid:
        return "review", "invalid_high_z_boundary_audit"
    if uncontrolled:
        return "fail", "uncontrolled_high_z_boundary_response"
    if contamination:
        return "fail", "resolved_high_z_common_interior_contamination"
    if source_converged and not radial_separated:
        return "review", "mixed_radial_source_grid_dependence"
    if source_converged and not temporal_separated:
        return "review", "mixed_temporal_source_grid_dependence"
    if source_converged and radial_separated and temporal_separated and normalization:
        return "pass", "high_z_converged_legacy_normalization_artifact"
    if source_converged and radial_separated and temporal_separated and response:
        return "pass", "high_z_converged_boundary_local_no_resolved_interior_effect"
    if not source_converged:
        return "review", "high_z_boundary_response_not_converged"
    return "review", "mixed_high_z_boundary_diagnosis"


def manufactured_controls():
    counts = np.asarray((80.0, 96.0, 112.0, 128.0))
    order_two = 0.04 + 13.0 / counts**2
    recovered = source_sequence("proper_ratio", order_two)
    nonmonotone = source_sequence(
        "proper_ratio", np.asarray((0.04, 0.041, 0.0405, 0.043)),
    )
    histories = np.vstack([
        order_two + 1e-5 * index for index in range(3)
        for _ in ()
    ]) if False else np.column_stack([
        order_two + 5e-6 * i for i in range(8)
    ])
    history = stage_history("proper_ratio", histories)
    radial_only = separation_record(
        "proper_ratio", 0.05, 0.08, temporal=False,
    )
    temporal_neutral = separation_record(
        "proper_ratio", 0.05, 0.0501, temporal=True,
    )
    fine_order = recovered["orders"]["fine_triplet_order"]
    gates = {
        "known_order_two": bool(fine_order is not None and abs(fine_order - 2.0) < 0.05),
        "known_sequence_passes": recovered["passes"],
        "stage_monotonicity": history["passes"],
        "nonmonotone_rejected": not nonmonotone["passes"],
        "radial_perturbation_assigned": not radial_only["passes"],
        "temporal_neutral_passes": temporal_neutral["passes"],
    }
    return {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "known_order": recovered,
        "nonmonotone": nonmonotone,
        "stage_history": history,
        "radial_only": radial_only,
        "temporal_neutral": temporal_neutral,
    }
