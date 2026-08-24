#!/usr/bin/env python3
"""Admit the pointwise nonlinear RHS in the regular nine-field SO(3) basis."""

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
from bhps.regular_so3_gh_reduction import pack_regular_so3_residual
from run_corrected_fold_boundary_constraint_pulse import build_geometry


Z_FRACTIONS = (0.05, 0.25, 0.5, 0.75, 0.95)
R_SUPPORT = (0.0625, 0.125, 0.1875, 0.25, 0.375, 0.5, 1.0, 2.0, 3.0, 4.0)
AXIS_FIT_COUNTS = (4, 6)
OUTPUT = Path("results/corrected_fold_nonlinear_rhs_regular_assembly.json")


def relative_norm(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(left), np.linalg.norm(right), 1e-300))


def solve_at(geometry, z_value, r_value):
    background = geometry["jet_field"].at(z_value, r_value)
    full_geometry = metric_geometry_from_jets(
        background["metric"], background["metric_first"], background["metric_second"],
    )
    source = full_geometry["contracted_christoffel_covector"]
    source_first = full_geometry["contracted_christoffel_covector_first"]
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
    closure = reduced_einstein_two_scalar_residual(
        background["metric"], background["metric_first"], metric_second,
        background["phi"], background["phi_first"], phi_second,
        background["chi"], background["chi_first"], chi_second,
        source, source_first, mass_squared=geometry["mass_squared"],
        potential_offset=-6.0, kappa5_squared=1.0,
    )

    acceleration = np.asarray(solved["metric_acceleration"])
    regular = pack_regular_so3_residual(
        acceleration, solved["phi_acceleration"], solved["chi_acceleration"], r_value,
    )
    reconstructed = np.zeros((5, 5))
    reconstructed[1, 0] = reconstructed[0, 1] = regular[0]
    reconstructed[1, 2] = reconstructed[2, 1] = r_value * regular[1]
    reconstructed[0, 0] = regular[2]
    reconstructed[3, 3] = reconstructed[4, 4] = regular[3]
    reconstructed[2, 2] = regular[3] + r_value**2 * regular[4]
    reconstructed[0, 2] = reconstructed[2, 0] = r_value * regular[5]
    reconstructed[1, 1] = regular[6]
    symmetry_defect = relative_norm(reconstructed, acceleration)
    transverse_defect = abs(acceleration[3, 3] - acceleration[4, 4]) / max(
        abs(acceleration[3, 3]), abs(acceleration[4, 4]), 1e-300,
    )
    closure_maximum = max(
        float(np.max(np.abs(closure["metric_residual"]))),
        float(abs(closure["phi_residual"])),
        float(abs(closure["chi_residual"])),
    )
    return {
        "z": float(z_value),
        "r": float(r_value),
        "regular_acceleration": regular,
        "so3_reconstruction_relative_defect": symmetry_defect,
        "transverse_degeneracy_relative_defect": float(transverse_defect),
        "postsolve_closure_maximum": closure_maximum,
        "gauge_constraint_maximum": float(np.max(np.abs(closure["gauge_constraint_covector"]))),
        "finite": bool(solved["finite"] and np.all(np.isfinite(regular))),
    }


def axis_limits(records, z_value, fit_count):
    selected = sorted(
        (record for record in records if record["z"] == z_value), key=lambda item: item["r"],
    )[:fit_count]
    squared_radius = np.asarray([record["r"]**2 for record in selected])
    values = np.asarray([record["regular_acceleration"] for record in selected])
    result = np.empty(values.shape[1])
    for field in range(values.shape[1]):
        result[field] = np.polynomial.polynomial.polyfit(
            squared_radius, values[:, field], min(2, fit_count - 1),
        )[0]
    return result


def build_case(resolution):
    geometry = build_geometry(resolution)
    z0 = float(geometry["z"][0])
    z1 = float(geometry["z"][-1])
    z_values = [z0 + fraction * (z1 - z0) for fraction in Z_FRACTIONS]
    records = [
        solve_at(geometry, z_value, r_value)
        for z_value in z_values for r_value in R_SUPPORT
    ]
    axis = {}
    for fit_count in AXIS_FIT_COUNTS:
        axis[str(fit_count)] = np.asarray([
            axis_limits(records, z_value, fit_count) for z_value in z_values
        ])
    axis_fit_difference = relative_norm(axis[str(AXIS_FIT_COUNTS[0])], axis[str(AXIS_FIT_COUNTS[1])])
    return {
        "resolution": resolution,
        "source_grid": geometry["source_grid"],
        "selector_maximum": geometry["selector_maximum"],
        "z_values": z_values,
        "records": records,
        "axis_limits": axis,
        "axis_fit_relative_difference": axis_fit_difference,
    }


def main():
    cases = [build_case(resolution) for resolution in ("G5", "G6")]
    coarse, fine = cases
    coarse_vectors = np.asarray([record["regular_acceleration"] for record in coarse["records"]])
    fine_vectors = np.asarray([record["regular_acceleration"] for record in fine["records"]])
    bulk_transfer = relative_norm(coarse_vectors, fine_vectors)
    axis_transfer = relative_norm(
        coarse["axis_limits"][str(AXIS_FIT_COUNTS[-1])],
        fine["axis_limits"][str(AXIS_FIT_COUNTS[-1])],
    )
    maximum_symmetry = max(
        record["so3_reconstruction_relative_defect"] for case in cases for record in case["records"]
    )
    maximum_transverse = max(
        record["transverse_degeneracy_relative_defect"] for case in cases for record in case["records"]
    )
    maximum_closure = max(
        record["postsolve_closure_maximum"] for case in cases for record in case["records"]
    )
    maximum_constraint = max(
        record["gauge_constraint_maximum"] for case in cases for record in case["records"]
    )
    maximum_axis_fit = max(case["axis_fit_relative_difference"] for case in cases)
    acceptance = {
        "all_values_finite": bool(all(record["finite"] for case in cases for record in case["records"])),
        "selectors_below_1e_8": bool(all(case["selector_maximum"] < 1e-8 for case in cases)),
        "postsolve_closure_below_1e_9": bool(maximum_closure < 1e-9),
        "so3_forbidden_content_below_1e_10": bool(max(maximum_symmetry, maximum_transverse) < 1e-10),
        "gauge_constraint_below_1e_10": bool(maximum_constraint < 1e-10),
        "bulk_G5_G6_transfer_below_5_percent": bool(bulk_transfer < 0.05),
        "axis_fit_robustness_below_5_percent": bool(maximum_axis_fit < 0.05),
        "axis_G5_G6_transfer_below_5_percent": bool(axis_transfer < 0.05),
    }
    for case in cases:
        case["axis_limits"] = {
            key: np.asarray(value).tolist() for key, value in case["axis_limits"].items()
        }
        for record in case["records"]:
            record["regular_acceleration"] = np.asarray(record["regular_acceleration"]).tolist()
    summary = {
        "maximum_postsolve_closure": maximum_closure,
        "maximum_so3_reconstruction_relative_defect": maximum_symmetry,
        "maximum_transverse_degeneracy_relative_defect": maximum_transverse,
        "maximum_gauge_constraint": maximum_constraint,
        "bulk_G5_G6_relative_difference": bulk_transfer,
        "maximum_axis_fit_relative_difference": maximum_axis_fit,
        "axis_G5_G6_relative_difference": axis_transfer,
    }
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "scope": "regular nine-field SO(3) assembly of the corrected-fold pointwise nonlinear bulk-interior acceleration candidate",
        "z_fractions": list(Z_FRACTIONS),
        "r_support": list(R_SUPPORT),
        "axis_fit_counts": list(AXIS_FIT_COUNTS),
        "cases": cases,
        "summary": summary,
        "acceptance": acceptance,
        "limitations": [
            "matched off-axis support points with an extrapolated regular axis limit",
            "initial fixed-stage gauge-source jets anchored to the archived time-symmetric data",
            "open-bulk assembly only; nonlinear physical-wall rows are the next gate",
            "no spatial method-of-lines operator or finite-time evolution",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "summary": summary, "acceptance": acceptance}, indent=2), flush=True)


if __name__ == "__main__":
    main()
