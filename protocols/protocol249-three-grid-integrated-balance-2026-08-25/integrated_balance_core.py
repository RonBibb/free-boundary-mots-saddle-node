"""Pure finite-segment integrated-balance rules for Protocol 249."""

from __future__ import annotations

import math


SEGMENT_STEPS = (44, 45, 46, 47)
WIDTHS = (5, 7, 9)
DT = 3.125e-5
RELATIVE_LIMIT = 0.01
ROUNDING_LIMIT = 1e-12
BRANE_TERMS = ("coupled_seam_global_radius", "coupled_seam_joint_work")
GATE_ORDER = (
    "prerequisite_admission",
    "segment_orientation",
    "charge_quadrature_closure",
    "integrated_flux_closure",
    "brane_ledger_completeness",
    "stencil_robustness",
    "spatial_transfer",
    "temporal_control_admission",
)


def symmetric_relative(left, right, floor=1e-300):
    values = (float(left), float(right), float(floor))
    if not all(math.isfinite(value) for value in values) or values[2] <= 0.0:
        raise ValueError("finite values and a positive floor are required")
    return abs(values[0] - values[1]) / max(abs(values[0]), abs(values[1]), values[2])


def trapezoid(values, dt=DT):
    values = tuple(float(value) for value in values)
    if len(values) != len(SEGMENT_STEPS) or not all(math.isfinite(value) for value in values):
        raise ValueError("exact finite four-node segment required")
    return float(dt) * (0.5 * values[0] + values[1] + values[2] + 0.5 * values[3])


def segment_record(local_grid, charges, width):
    key = str(width)
    if set(charges) != set(SEGMENT_STEPS):
        raise ValueError("exact charge-step inventory required")
    records = [local_grid["steps"][str(step)]["records"][key]["centered"] for step in SEGMENT_STEPS]
    term_inventory = set(records[0]["ledger"]["terms"])
    if not term_inventory or any(set(record["ledger"]["terms"]) != term_inventory for record in records):
        raise ValueError("ledger-term inventory differs across the segment")
    if not set(BRANE_TERMS) <= term_inventory:
        raise ValueError("brane endpoint terms are absent")

    target_rates = [record["ledger"]["target_rate"] for record in records]
    total_fluxes = [record["ledger"]["total_flux"] for record in records]
    balance_norms = [record["ledger"]["balance_norm"] for record in records]
    integrated_terms = {
        name: trapezoid([record["ledger"]["terms"][name] for record in records])
        for name in sorted(term_inventory)
    }
    charge_change = float(charges[SEGMENT_STEPS[-1]] - charges[SEGMENT_STEPS[0]])
    integrated_target = trapezoid(target_rates)
    integrated_flux = trapezoid(total_fluxes)
    integrated_balance_norm = trapezoid(balance_norms)
    integrated_term_sum = float(math.fsum(integrated_terms.values()))
    charge_scale = max(abs(charge_change), abs(integrated_target), abs(integrated_flux), 1e-12)
    balance_scale = max(charge_scale, abs(integrated_balance_norm), 1e-12)
    residuals = {
        "charge_minus_target": charge_change - integrated_target,
        "charge_minus_flux": charge_change - integrated_flux,
        "target_minus_flux": integrated_target - integrated_flux,
        "term_sum_minus_flux": integrated_term_sum - integrated_flux,
        "charge_target_relative": abs(charge_change - integrated_target) / charge_scale,
        "charge_flux_relative": abs(charge_change - integrated_flux) / charge_scale,
        "target_flux_relative": abs(integrated_target - integrated_flux) / charge_scale,
        "charge_flux_balance_norm_relative": abs(charge_change - integrated_flux) / balance_scale,
        "term_sum_balance_norm_relative": abs(integrated_term_sum - integrated_flux) / balance_scale,
    }
    passes = {
        "orientation": bool(charge_change > 0.0 and integrated_target > 0.0 and integrated_flux > 0.0),
        "quadrature": bool(residuals["charge_target_relative"] < RELATIVE_LIMIT),
        "physical_flux": bool(
            residuals["charge_flux_relative"] < RELATIVE_LIMIT
            and residuals["target_flux_relative"] < RELATIVE_LIMIT
            and residuals["charge_flux_balance_norm_relative"] < RELATIVE_LIMIT
        ),
        "term_sum": bool(residuals["term_sum_balance_norm_relative"] < ROUNDING_LIMIT),
    }
    return {
        "width": int(width),
        "start_step": SEGMENT_STEPS[0],
        "end_step": SEGMENT_STEPS[-1],
        "duration": DT * (SEGMENT_STEPS[-1] - SEGMENT_STEPS[0]),
        "endpoint_charges": {str(step): float(charges[step]) for step in SEGMENT_STEPS},
        "charge_change": charge_change,
        "integrated_target_rate": integrated_target,
        "integrated_total_flux": integrated_flux,
        "integrated_balance_norm": integrated_balance_norm,
        "integrated_terms": integrated_terms,
        "integrated_brane_endpoint_flux": float(math.fsum(integrated_terms[name] for name in BRANE_TERMS)),
        "residuals": residuals,
        "passes": passes,
        "passed": bool(all(passes.values())),
    }


def compare_segments(left, right):
    left_terms, right_terms = left["integrated_terms"], right["integrated_terms"]
    if set(left_terms) != set(right_terms):
        raise ValueError("integrated-term inventories differ")
    norm = max(abs(left["integrated_balance_norm"]), abs(right["integrated_balance_norm"]), 1e-12)
    term_errors = {name: abs(left_terms[name] - right_terms[name]) / norm for name in sorted(left_terms)}
    metrics = {
        "start_charge_relative": symmetric_relative(left["endpoint_charges"]["44"], right["endpoint_charges"]["44"]),
        "end_charge_relative": symmetric_relative(left["endpoint_charges"]["47"], right["endpoint_charges"]["47"]),
        "charge_change_relative": symmetric_relative(left["charge_change"], right["charge_change"]),
        "integrated_target_relative": symmetric_relative(left["integrated_target_rate"], right["integrated_target_rate"]),
        "integrated_flux_relative": symmetric_relative(left["integrated_total_flux"], right["integrated_total_flux"]),
        "integrated_brane_endpoint_flux_balance_norm_relative": abs(
            left["integrated_brane_endpoint_flux"] - right["integrated_brane_endpoint_flux"]
        ) / norm,
        "maximum_term_balance_norm_relative": max(term_errors.values()),
    }
    sign_pass = bool(
        left["charge_change"] > 0.0 and right["charge_change"] > 0.0
        and left["integrated_total_flux"] > 0.0 and right["integrated_total_flux"] > 0.0
    )
    passed = bool(sign_pass and all(value < RELATIVE_LIMIT for value in metrics.values()))
    return {"metrics": metrics, "term_errors": term_errors, "sign_pass": sign_pass, "passed": passed}


def classify(gates):
    if set(gates) != set(GATE_ORDER) or any(type(gates[name]) is not bool for name in GATE_ORDER):
        raise ValueError("exact Boolean gate inventory required")
    failures = {
        "prerequisite_admission": "INTEGRATED-BALANCE-PREREQUISITE-FAIL",
        "segment_orientation": "INTEGRATED-BALANCE-ORIENTATION-FAIL",
        "charge_quadrature_closure": "INTEGRATED-BALANCE-QUADRATURE-FAIL",
        "integrated_flux_closure": "INTEGRATED-BALANCE-PHYSICAL-FLUX-FAIL",
        "brane_ledger_completeness": "INTEGRATED-BALANCE-BRANE-LEDGER-FAIL",
        "stencil_robustness": "INTEGRATED-BALANCE-STENCIL-ROBUSTNESS-FAIL",
        "spatial_transfer": "INTEGRATED-BALANCE-SPATIAL-TRANSFER-FAIL",
        "temporal_control_admission": "INTEGRATED-BALANCE-TEMPORAL-CONTROL-FAIL",
    }
    for name in GATE_ORDER:
        if not gates[name]:
            return failures[name]
    return "G9-G10-G11-FINITE-SEGMENT-INTEGRATED-BALANCE-PASS"
