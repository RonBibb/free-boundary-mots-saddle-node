#!/usr/bin/env python3
"""Audit the initial GH-constraint tangent of the nonlinear bulk RHS."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.linearized_gh_einstein_scalar import (
    metric_geometry_from_jets,
    reduced_einstein_two_scalar_residual,
    solve_reduced_einstein_two_scalar_acceleration,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_g7
from run_corrected_fold_regular_so3_runtime import build_geometry


Z_FRACTIONS = (0.05, 0.25, 0.5, 0.75, 0.95)
R_VALUES = (0.25, 0.75, 1.5, 3.0, 4.0)
OUTPUT = Path("results/corrected_fold_nonlinear_rhs_constraint_tangent.json")


def local_record(geometry, z_value, r_value):
    background = geometry["jet_field"].at(z_value, r_value)
    archived_geometry = metric_geometry_from_jets(
        background["metric"], background["metric_first"], background["metric_second"],
    )
    source = archived_geometry["contracted_christoffel_covector"]
    source_first = archived_geometry["contracted_christoffel_covector_first"]
    solved = solve_reduced_einstein_two_scalar_acceleration(
        background["metric"], background["metric_first"], background["metric_second"],
        background["phi"], background["phi_first"], background["phi_second"],
        background["chi"], background["chi_first"], background["chi_second"],
        source, source_first, mass_squared=geometry["mass_squared"],
        potential_offset=-6.0, kappa5_squared=1.0,
    )
    metric_second = np.asarray(background["metric_second"]).copy()
    phi_second = np.asarray(background["phi_second"]).copy()
    chi_second = np.asarray(background["chi_second"]).copy()
    metric_second[0, 0] = solved["metric_acceleration"]
    phi_second[0, 0] = solved["phi_acceleration"]
    chi_second[0, 0] = solved["chi_acceleration"]
    residual = reduced_einstein_two_scalar_residual(
        background["metric"], background["metric_first"], metric_second,
        background["phi"], background["phi_first"], phi_second,
        background["chi"], background["chi_first"], chi_second,
        source, source_first, mass_squared=geometry["mass_squared"],
        potential_offset=-6.0, kappa5_squared=1.0,
    )
    tangent = np.asarray(residual["gauge_constraint_first_covector"])[0]
    new_gamma_t = np.asarray(residual["contracted_christoffel_covector_first"])[0]
    source_t = np.asarray(source_first)[0]
    scale = max(np.linalg.norm(new_gamma_t), np.linalg.norm(source_t), 1e-300)
    return {
        "z": float(z_value),
        "r": float(r_value),
        "constraint_tangent": tangent,
        "new_contracted_christoffel_time_derivative": new_gamma_t,
        "source_time_derivative": source_t,
        "constraint_tangent_norm": float(np.linalg.norm(tangent)),
        "constraint_tangent_relative": float(np.linalg.norm(tangent) / scale),
        "spatial_constraint_tangent_norm": float(np.linalg.norm(tangent[1:])),
    }


def build_case(geometry, label):
    z0 = float(geometry["z"][0])
    z1 = float(geometry["z"][-1])
    records = [
        local_record(geometry, z0 + fraction * (z1 - z0), r_value)
        for fraction in Z_FRACTIONS for r_value in R_VALUES
    ]
    tangent = np.asarray([record["constraint_tangent"] for record in records])
    gamma_t = np.asarray([
        record["new_contracted_christoffel_time_derivative"] for record in records
    ])
    source_t = np.asarray([record["source_time_derivative"] for record in records])
    return {
        "resolution": label,
        "source_grid": geometry["source_grid"],
        "selector_maximum": geometry["selector_maximum"],
        "records": records,
        "global_constraint_tangent_relative": float(
            np.linalg.norm(tangent)
            / max(np.linalg.norm(gamma_t), np.linalg.norm(source_t), 1e-300)
        ),
        "maximum_constraint_tangent_relative": max(
            record["constraint_tangent_relative"] for record in records
        ),
        "maximum_constraint_tangent_norm": max(
            record["constraint_tangent_norm"] for record in records
        ),
        "maximum_spatial_constraint_tangent_norm": max(
            record["spatial_constraint_tangent_norm"] for record in records
        ),
    }


def main():
    g5 = build_geometry("G5")
    g6 = build_geometry("G6")
    g7 = build_g7(g6)
    cases = [build_case(g5, "G5"), build_case(g6, "G6"), build_case(g7, "G7")]
    global_values = [case["global_constraint_tangent_relative"] for case in cases]
    maximum_values = [case["maximum_constraint_tangent_relative"] for case in cases]
    absolute_values = [case["maximum_constraint_tangent_norm"] for case in cases]
    decrease_global_g5_g6 = global_values[0] / max(global_values[1], 1e-300)
    decrease_global_g6_g7 = global_values[1] / max(global_values[2], 1e-300)
    decrease_absolute_g6_g7 = absolute_values[1] / max(absolute_values[2], 1e-300)
    maximum_spatial = max(case["maximum_spatial_constraint_tangent_norm"] for case in cases)
    acceptance = {
        "selectors_below_1e_8": bool(all(case["selector_maximum"] < 1e-8 for case in cases)),
        "G7_global_constraint_tangent_below_0_5_percent": bool(global_values[-1] < 0.005),
        "G7_maximum_constraint_tangent_below_0_5_percent": bool(maximum_values[-1] < 0.005),
        "global_constraint_tangent_decreases_G5_G6": bool(decrease_global_g5_g6 > 1.1),
        "global_constraint_tangent_decreases_G6_G7": bool(decrease_global_g6_g7 > 1.1),
        "absolute_constraint_tangent_decreases_G6_G7": bool(decrease_absolute_g6_g7 > 1.1),
        "spatial_constraint_tangent_below_1e_12": bool(maximum_spatial < 1e-12),
    }
    for case in cases:
        for record in case["records"]:
            for key in (
                "constraint_tangent", "new_contracted_christoffel_time_derivative",
                "source_time_derivative",
            ):
                record[key] = np.asarray(record[key]).tolist()
    summary = {
        "global_constraint_tangent_relative_by_resolution": dict(zip(("G5", "G6", "G7"), global_values)),
        "maximum_constraint_tangent_relative_by_resolution": dict(zip(("G5", "G6", "G7"), maximum_values)),
        "maximum_constraint_tangent_norm_by_resolution": dict(zip(("G5", "G6", "G7"), absolute_values)),
        "global_decrease_factor_G5_to_G6": decrease_global_g5_g6,
        "global_decrease_factor_G6_to_G7": decrease_global_g6_g7,
        "absolute_decrease_factor_G6_to_G7": decrease_absolute_g6_g7,
        "maximum_spatial_constraint_tangent_norm": maximum_spatial,
    }
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "scope": "open-bulk initial GH constraint-tangent convergence for the fixed-source nonlinear acceleration solve",
        "z_fractions": list(Z_FRACTIONS),
        "r_values": list(R_VALUES),
        "cases": cases,
        "summary": summary,
        "acceptance": acceptance,
        "interpretation": [
            "The metric and scalar values and first time derivatives are unchanged, so C_a=Gamma_a-H_a remains zero initially.",
            "Changing the metric acceleration changes Gamma_a,t; the reported tangent is Gamma_a,t-H_a,t with the archived source time jet.",
            "Convergence toward zero tests whether this is inherited initial-data discretization error rather than a new gauge-driver inconsistency.",
        ],
        "limitations": [
            "twenty-five matched open-bulk off-axis samples per resolution",
            "fixed archived gauge-source time jet",
            "compact-wall and regular-axis tangents require the assembled method-of-lines gate",
            "no finite-time evolution",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "summary": summary, "acceptance": acceptance}, indent=2), flush=True)


if __name__ == "__main__":
    main()
