"""Pure three-grid balance comparison for Protocol 248."""

from __future__ import annotations

import math


GATE_ORDER = (
    "parent_admission",
    "local_balance_admission",
    "time_alignment",
    "sign_consistency",
    "geometry_consistency",
    "balance_consistency",
    "native_operator_consistency",
)


def symmetric_relative(left, right, floor=1e-300):
    left, right, floor = float(left), float(right), float(floor)
    if not all(math.isfinite(value) for value in (left, right, floor)) or floor <= 0.0:
        raise ValueError("finite values and a positive floor are required")
    return abs(left - right) / max(abs(left), abs(right), floor)


def inherited_native_rate_pass(left, right):
    return bool(symmetric_relative(left, right, 0.005) < 0.05 or abs(float(left) - float(right)) < 0.005)


def compare_centered(left_step, right_step, width):
    key = str(width)
    left = left_step["records"][key]["centered"]
    right = right_step["records"][key]["centered"]
    left_wall = left["native_wall_rate"]
    right_wall = right["native_wall_rate"]
    left_direct = left_wall["epsilon_records"]["5.0e-07"]["directional_rate"]
    right_direct = right_wall["epsilon_records"]["5.0e-07"]["directional_rate"]
    comparison = {
        "area_value_relative": symmetric_relative(
            left_step["area_values"][str(left_step["step"])],
            right_step["area_values"][str(right_step["step"])],
        ),
        "area_finite_rate_relative": symmetric_relative(
            left["area_transport"]["finite_difference_rate"], right["area_transport"]["finite_difference_rate"]
        ),
        "area_marginal_rate_relative": symmetric_relative(
            left["area_transport"]["marginal_integral_rate"], right["area_transport"]["marginal_integral_rate"]
        ),
        "seam_geometric_relative": symmetric_relative(left["seam"]["c_geometric"], right["seam"]["c_geometric"]),
        "seam_wall_relative": symmetric_relative(left["seam"]["c_wall"], right["seam"]["c_wall"]),
        "ledger_target_relative": symmetric_relative(left["ledger"]["target_rate"], right["ledger"]["target_rate"]),
        "ledger_total_relative": symmetric_relative(left["ledger"]["total_flux"], right["ledger"]["total_flux"]),
        "native_directional_relative": symmetric_relative(left_direct, right_direct, 0.005),
        "native_directional_absolute": abs(left_direct - right_direct),
        "native_history_relative": symmetric_relative(left_wall["history_rate"], right_wall["history_rate"], 0.005),
        "native_history_absolute": abs(left_wall["history_rate"] - right_wall["history_rate"]),
        "native_wall_relative": symmetric_relative(left_wall["wall_rate"], right_wall["wall_rate"], 0.005),
        "native_wall_absolute": abs(left_wall["wall_rate"] - right_wall["wall_rate"]),
    }
    terms = set(left["ledger"]["terms"])
    if terms != set(right["ledger"]["terms"]):
        raise ValueError("ledger term inventories differ")
    norm = max(left["ledger"]["balance_norm"], right["ledger"]["balance_norm"], 1e-12)
    term_errors = {
        name: abs(left["ledger"]["terms"][name] - right["ledger"]["terms"][name]) / norm
        for name in sorted(terms)
    }
    comparison["ledger_term_balance_norm_relative"] = max(term_errors.values())
    comparison["parent_normalized_residual"] = max(
        left["ledger"]["normalized_absolute_residual"], right["ledger"]["normalized_absolute_residual"]
    )
    sign_pass = bool(
        left["area_transport"]["finite_difference_rate"] > 0.0
        and right["area_transport"]["finite_difference_rate"] > 0.0
        and left["ledger"]["target_rate"] > 0.0 and right["ledger"]["target_rate"] > 0.0
        and left["ledger"]["total_flux"] > 0.0 and right["ledger"]["total_flux"] > 0.0
        and left_direct < 0.0 and right_direct < 0.0
    )
    geometry_pass = bool(
        comparison["area_value_relative"] < 0.01
        and comparison["area_finite_rate_relative"] < 0.02
        and comparison["area_marginal_rate_relative"] < 0.02
        and comparison["seam_geometric_relative"] < 0.01
        and comparison["seam_wall_relative"] < 0.01
    )
    balance_pass = bool(
        comparison["ledger_target_relative"] < 0.01
        and comparison["ledger_total_relative"] < 0.01
        and comparison["ledger_term_balance_norm_relative"] < 0.01
        and comparison["parent_normalized_residual"] < 0.01
    )
    native_pass = bool(
        inherited_native_rate_pass(left_direct, right_direct)
        and inherited_native_rate_pass(left_wall["history_rate"], right_wall["history_rate"])
        and inherited_native_rate_pass(left_wall["wall_rate"], right_wall["wall_rate"])
    )
    return {
        "comparison": comparison,
        "ledger_term_errors": term_errors,
        "sign_pass": sign_pass,
        "geometry_pass": geometry_pass,
        "balance_pass": balance_pass,
        "native_operator_pass": native_pass,
        "passed": bool(sign_pass and geometry_pass and balance_pass and native_pass),
    }


def classify(gates):
    if set(gates) != set(GATE_ORDER) or any(type(gates[name]) is not bool for name in GATE_ORDER):
        raise ValueError("exact Boolean gate inventory required")
    labels = {
        "parent_admission": "THREE-GRID-BALANCE-PARENT-ADMISSION-FAIL",
        "local_balance_admission": "THREE-GRID-LOCAL-BALANCE-ADMISSION-FAIL",
        "time_alignment": "THREE-GRID-BALANCE-TIME-ALIGNMENT-FAIL",
        "sign_consistency": "THREE-GRID-BALANCE-SIGN-CONSISTENCY-FAIL",
        "geometry_consistency": "THREE-GRID-BALANCE-GEOMETRY-CONSISTENCY-FAIL",
        "balance_consistency": "THREE-GRID-BALANCE-CONSISTENCY-FAIL",
        "native_operator_consistency": "THREE-GRID-NATIVE-OPERATOR-CONSISTENCY-FAIL",
    }
    for name in GATE_ORDER:
        if not gates[name]:
            return labels[name]
    return "G9-G10-G11-NATIVE-LOCAL-BALANCE-TRANSFER-PASS"
