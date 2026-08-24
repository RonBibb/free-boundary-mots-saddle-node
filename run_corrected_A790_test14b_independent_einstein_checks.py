#!/usr/bin/env python3
"""Independent Einstein/scalar and final-arithmetic audit for Test 14B."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.recovery_indexer import atomic_write_json, sha256_file
from bhps.test14b_balance_closure import (
    evaluate_balance_leaf,
    five_point_history_derivative,
)
from bhps.test14b_einstein_audit import einstein_null_contractions


PROTOCOL = Path("notes/96_A790_test14B_balance_closure_protocol.md")
PRIMARY = Path("results/corrected_A790_test14b_balance_closure.json")
STATE = Path("results/corrected_A790_t008_long_evolution_state.npz")
NOTE95_INDEPENDENT = Path(
    "results/corrected_A790_test14_independent_geometry_checks.json"
)
OUTPUT = Path(
    "results/corrected_A790_test14b_independent_einstein_checks.json"
)
ANCHOR_TIMES = (0.001, 0.002, 0.003, 0.004)


def integrate_window(records, left, right):
    selected = [
        item for item in records if left - 1e-12 <= item["time"] <= right + 1e-12
    ]
    times = np.asarray([item["time"] for item in selected])
    charge = np.asarray([
        item["charge_rate_target"]["charge"] for item in selected
    ])
    total = np.asarray([item["total_balance_rate"] for item in selected])
    named = np.stack([
        np.asarray(list(item["rates"].values()), dtype=float) for item in selected
    ])
    delta = float(charge[-1] - charge[0])
    integrated = float(np.trapezoid(total, times))
    norm = max(abs(delta), float(np.trapezoid(np.sum(abs(named), axis=1), times)), 1e-12)
    return {
        "delta_charge": delta, "integrated_total_rate": integrated,
        "closure_residual": delta - integrated,
        "balance_norm": norm,
        "normalized_absolute_residual": abs(delta - integrated) / norm,
    }


def main():
    primary = json.loads(PRIMARY.read_text())
    if primary["protocol_sha256"] != sha256_file(PROTOCOL):
        raise RuntimeError("primary Test-14B result has a different protocol hash")
    note95 = json.loads(NOTE95_INDEPENDENT.read_text())
    archive = np.load(STATE)
    absolute_indices = np.arange(5, len(archive["times"]))
    times = np.asarray(archive["times"])[absolute_indices]
    records = []
    for grid in ("G7", "G8"):
        z = np.asarray(archive[f"{grid}_z"])
        r = np.asarray(archive[f"{grid}_r"])
        positions = np.asarray(archive[f"{grid}_position_history"])[absolute_indices]
        velocities = np.asarray(archive[f"{grid}_velocity_history"])[absolute_indices]
        accelerations = five_point_history_derivative(velocities, times, stride=1)
        for branch in ("inner", "outer"):
            surface_history = primary["surface_records"][grid][branch]
            rho_history = np.stack([
                np.asarray(item["surface"]["rho"], dtype=float)
                for item in surface_history
            ])
            rho_rate = five_point_history_derivative(rho_history, times, stride=1)
            for current_time in ANCHOR_TIMES:
                local = int(np.argmin(abs(times - current_time)))
                source = surface_history[local]["surface"]
                profile = {
                    key: np.asarray(source[key], dtype=float)
                    for key in ("theta", "rho", "slope")
                }
                print(f"{grid} {branch} t={current_time:.3f}", flush=True)
                einstein = einstein_null_contractions(
                    positions[local], velocities[local], accelerations[local],
                    z, r, profile, primary["background"]["mass_squared"],
                    sample_nodes=129,
                )
                evaluated = evaluate_balance_leaf(
                    positions[local], velocities[local], z, r, profile,
                    rho_rate[local], primary["background"],
                    bulk_stress_override=einstein,
                )
                original = primary["balance_records"][grid][branch]["1"][local]
                records.append({
                    "grid": grid, "branch": branch, "time": current_time,
                    "sample_nodes": einstein["sample_nodes"],
                    "einstein_to_scalar_Tll_relative_L2": einstein[
                        "einstein_to_scalar_Tll_relative_L2"
                    ],
                    "einstein_to_scalar_total_Tln_relative_L2": einstein[
                        "einstein_to_scalar_total_Tln_relative_L2"
                    ],
                    "maximum_absolute_l_squared": einstein[
                        "maximum_absolute_l_squared"
                    ],
                    "maximum_absolute_n_squared": einstein[
                        "maximum_absolute_n_squared"
                    ],
                    "maximum_absolute_l_dot_n_plus_one": einstein[
                        "maximum_absolute_l_dot_n_plus_one"
                    ],
                    "primary_scalar_balance_rate": original["total_balance_rate"],
                    "independently_recomputed_scalar_balance_rate": evaluated[
                        "total_balance_rate"
                    ],
                    "einstein_replacement_balance_rate": evaluated[
                        "bulk_stress_override"
                    ]["total_rate_replacing_scalar_plus_vacuum"],
                    "charge_rate_target": original["charge_rate_target"][
                        "finite_difference_rate"
                    ],
                    "israel_seam_relative_scale_error": evaluated["seam"][
                        "israel_intrinsic_relative_scale_error"
                    ],
                    "area_transport_relative_scale_error": original[
                        "area_transport_check"
                    ]["relative_scale_error"],
                    "finite": bool(einstein["finite"] and evaluated["finite"]),
                })

    arithmetic = {}
    maximum_arithmetic_difference = 0.0
    for grid in ("G7", "G8"):
        arithmetic[grid] = {}
        for branch in ("inner", "outer"):
            direct = integrate_window(
                primary["balance_records"][grid][branch]["1"],
                float(times[0]), 0.004,
            )
            archived = primary["summaries"][grid][branch]["1"]["windows"][
                "primary"
            ]
            difference = max(
                abs(direct[key] - archived[key])
                for key in (
                    "delta_charge", "integrated_total_rate", "closure_residual",
                    "balance_norm", "normalized_absolute_residual",
                )
            )
            maximum_arithmetic_difference = max(
                maximum_arithmetic_difference, difference,
            )
            arithmetic[grid][branch] = {
                "independent": direct,
                "maximum_absolute_difference_from_primary_summary": difference,
            }

    summary = {
        "all_finite": bool(all(item["finite"] for item in records)),
        "maximum_einstein_to_scalar_Tll_relative_L2": max(
            item["einstein_to_scalar_Tll_relative_L2"] for item in records
        ),
        "maximum_einstein_to_scalar_total_Tln_relative_L2": max(
            item["einstein_to_scalar_total_Tln_relative_L2"] for item in records
        ),
        "maximum_null_normalization_error": max(
            max(
                item["maximum_absolute_l_squared"],
                item["maximum_absolute_n_squared"],
                item["maximum_absolute_l_dot_n_plus_one"],
            ) for item in records
        ),
        "maximum_einstein_replacement_rate_change": max(
            abs(
                item["einstein_replacement_balance_rate"]
                - item["primary_scalar_balance_rate"]
            ) for item in records
        ),
        "maximum_primary_rate_recomputation_difference": max(
            abs(
                item["independently_recomputed_scalar_balance_rate"]
                - item["primary_scalar_balance_rate"]
            ) for item in records
        ),
        "maximum_final_arithmetic_difference": maximum_arithmetic_difference,
        "note95_direct_intrinsic_maximum_relative_difference": note95[
            "summary"
        ]["maximum_intrinsic_crosscheck_relative_difference"],
    }
    scalar_einstein_pass = bool(
        summary["maximum_einstein_to_scalar_Tll_relative_L2"] < 0.01
        and summary["maximum_einstein_to_scalar_total_Tln_relative_L2"] < 0.01
    )
    passed = bool(
        summary["all_finite"] and scalar_einstein_pass
        and summary["maximum_null_normalization_error"] < 1e-10
        and summary["maximum_final_arithmetic_difference"] < 1e-12
    )
    payload = {
        "status": "PASS" if passed else "REVIEW",
        "classification": "independent_test14b_einstein_and_arithmetic_audit",
        "protocol": str(PROTOCOL), "protocol_sha256": sha256_file(PROTOCOL),
        "primary_result": str(PRIMARY),
        "primary_result_sha256": sha256_file(PRIMARY),
        "state_sha256": sha256_file(STATE),
        "note95_independent_result": str(NOTE95_INDEPENDENT),
        "note95_independent_sha256": sha256_file(NOTE95_INDEPENDENT),
        "records": records, "arithmetic": arithmetic, "summary": summary,
        "acceptance": {
            "scalar_and_full_einstein_contractions_agree_below_1_percent": (
                scalar_einstein_pass
            ),
            "null_normalization_below_1e_10": bool(
                summary["maximum_null_normalization_error"] < 1e-10
            ),
            "final_arithmetic_reproduced_below_1e_12": bool(
                summary["maximum_final_arithmetic_difference"] < 1e-12
            ),
        },
        "interpretation": (
            "Agreement between full geometric Einstein contractions and the "
            "canonical scalar-plus-vacuum stress makes an omitted smooth bulk "
            "matter term an implausible explanation of the outer-tube closure "
            "deficit. It does not repair that deficit or promote Test 14B."
        ),
    }
    atomic_write_json(OUTPUT, payload)
    print(json.dumps({"status": payload["status"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

