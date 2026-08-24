#!/usr/bin/env python3
"""Admit a pointwise nonlinear reduced Einstein--scalar acceleration RHS."""

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
from run_corrected_fold_boundary_constraint_pulse import build_geometry


Z_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
R_VALUES = (0.25, 0.75, 1.5, 3.0, 4.0)
OUTPUT = Path("results/corrected_fold_nonlinear_rhs_admission.json")


def relative_norm(left, right):
    return float(
        np.linalg.norm(np.asarray(left) - np.asarray(right))
        / max(np.linalg.norm(left), np.linalg.norm(right), 1e-300)
    )


def local_record(geometry, z_value, r_value):
    background = geometry["jet_field"].at(z_value, r_value)
    full_geometry = metric_geometry_from_jets(
        background["metric"], background["metric_first"], background["metric_second"],
    )
    solved = solve_reduced_einstein_two_scalar_acceleration(
        background["metric"],
        background["metric_first"],
        background["metric_second"],
        background["phi"],
        background["phi_first"],
        background["phi_second"],
        background["chi"],
        background["chi_first"],
        background["chi_second"],
        full_geometry["contracted_christoffel_covector"],
        full_geometry["contracted_christoffel_covector_first"],
        mass_squared=geometry["mass_squared"],
        potential_offset=-6.0,
        kappa5_squared=1.0,
    )
    archived_metric = np.asarray(background["metric_second"])[0, 0]
    archived_phi = float(np.asarray(background["phi_second"])[0, 0])
    archived_chi = float(np.asarray(background["chi_second"])[0, 0])
    diagonal = np.diag(background["metric"])
    frame = 1 / np.sqrt(np.maximum(np.abs(diagonal), 1e-300))
    archived_orthonormal = archived_metric * frame[:, None] * frame[None, :]
    solved_orthonormal = solved["metric_acceleration"] * frame[:, None] * frame[None, :]

    metric_second = np.asarray(background["metric_second"]).copy()
    phi_second = np.asarray(background["phi_second"]).copy()
    chi_second = np.asarray(background["chi_second"]).copy()
    metric_second[0, 0] = solved["metric_acceleration"]
    phi_second[0, 0] = solved["phi_acceleration"]
    chi_second[0, 0] = solved["chi_acceleration"]
    closure = reduced_einstein_two_scalar_residual(
        background["metric"],
        background["metric_first"],
        metric_second,
        background["phi"],
        background["phi_first"],
        phi_second,
        background["chi"],
        background["chi_first"],
        chi_second,
        full_geometry["contracted_christoffel_covector"],
        full_geometry["contracted_christoffel_covector_first"],
        mass_squared=geometry["mass_squared"],
        potential_offset=-6.0,
        kappa5_squared=1.0,
    )
    metric_relative = relative_norm(solved_orthonormal, archived_orthonormal)
    phi_relative = abs(solved["phi_acceleration"] - archived_phi) / max(
        abs(solved["phi_acceleration"]), abs(archived_phi), 1e-300
    )
    chi_relative = abs(solved["chi_acceleration"] - archived_chi) / max(
        abs(solved["chi_acceleration"]), abs(archived_chi), 1e-300
    )
    return {
        "z": float(z_value),
        "r": float(r_value),
        "metric_acceleration_relative_difference": metric_relative,
        "phi_acceleration_relative_difference": float(phi_relative),
        "chi_acceleration_relative_difference": float(chi_relative),
        "archived_metric_acceleration_orthonormal_norm": float(np.linalg.norm(archived_orthonormal)),
        "solved_metric_acceleration_orthonormal_norm": float(np.linalg.norm(solved_orthonormal)),
        "archived_phi_acceleration": archived_phi,
        "solved_phi_acceleration": solved["phi_acceleration"],
        "archived_chi_acceleration": archived_chi,
        "solved_chi_acceleration": solved["chi_acceleration"],
        "closure_metric_residual_maximum": float(np.max(np.abs(closure["metric_residual"]))),
        "closure_phi_residual": float(abs(closure["phi_residual"])),
        "closure_chi_residual": float(abs(closure["chi_residual"])),
        "finite": solved["finite"],
        "_solved_vector": np.r_[
            solved_orthonormal.ravel(), solved["phi_acceleration"], solved["chi_acceleration"],
        ],
    }


def build_case(resolution):
    geometry = build_geometry(resolution)
    z0 = float(geometry["z"][0])
    z1 = float(geometry["z"][-1])
    z_values = [z0 + fraction * (z1 - z0) for fraction in Z_FRACTIONS]
    records = [
        local_record(geometry, z_value, r_value)
        for z_value in z_values for r_value in R_VALUES
    ]
    return {
        "resolution": resolution,
        "source_grid": geometry["source_grid"],
        "fold_amplitude": geometry["fold_amplitude"],
        "selector_maximum": geometry["selector_maximum"],
        "z_values": z_values,
        "records": records,
    }


def main():
    cases = [build_case(resolution) for resolution in ("G5", "G6")]
    coarse, fine = cases
    coarse_vectors = np.asarray([item["_solved_vector"] for item in coarse["records"]])
    fine_vectors = np.asarray([item["_solved_vector"] for item in fine["records"]])
    cross_grid = relative_norm(coarse_vectors, fine_vectors)
    fine_metric = max(item["metric_acceleration_relative_difference"] for item in fine["records"])
    fine_phi = max(item["phi_acceleration_relative_difference"] for item in fine["records"])
    fine_chi = max(item["chi_acceleration_relative_difference"] for item in fine["records"])
    maximum_closure = max(
        max(
            item["closure_metric_residual_maximum"],
            item["closure_phi_residual"],
            item["closure_chi_residual"],
        )
        for case in cases for item in case["records"]
    )
    acceptance = {
        "selectors_below_1e_8": bool(all(case["selector_maximum"] < 1e-8 for case in cases)),
        "all_solved_accelerations_finite": bool(all(
            item["finite"] for case in cases for item in case["records"]
        )),
        "solved_rhs_closes_reduced_equations_below_1e_9": bool(maximum_closure < 1e-9),
        "fine_metric_acceleration_difference_below_5_percent": bool(fine_metric < 0.05),
        "fine_phi_acceleration_difference_below_5_percent": bool(fine_phi < 0.05),
        "fine_chi_acceleration_difference_below_5_percent": bool(fine_chi < 0.05),
        "solved_rhs_cross_grid_difference_below_5_percent": bool(cross_grid < 0.05),
    }
    for case in cases:
        for item in case["records"]:
            item.pop("_solved_vector")
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "scope": "pointwise nonlinear reduced Einstein--two-scalar acceleration RHS admission on matched corrected G5/G6 points",
        "z_fractions": list(Z_FRACTIONS),
        "r_values": list(R_VALUES),
        "cases": cases,
        "summary": {
            "fine_maximum_metric_acceleration_relative_difference": fine_metric,
            "fine_maximum_phi_acceleration_relative_difference": fine_phi,
            "fine_maximum_chi_acceleration_relative_difference": fine_chi,
            "maximum_postsolve_reduced_residual": maximum_closure,
            "solved_rhs_cross_grid_relative_difference": cross_grid,
        },
        "acceptance": acceptance,
        "interpretation": [
            "The pointwise nonlinear primitive solves the fixed-stage generalized-harmonic reduced equations for metric and two-scalar accelerations rather than reusing a frozen linear perturbation operator.",
            "Agreement with the independently constructed ADM/scalar accelerations is the zero-step admission test for an evolved background.",
            "This does not yet provide spatial differencing, boundary updates, a gauge-driver step, constraint preservation, or time convergence.",
        ],
        "limitations": [
            "twenty-five matched off-axis points per corrected fold",
            "gauge-source value and first jets are fixed to the complete initial contracted-Christoffel jets",
            "pointwise RHS admission rather than a nonlinear method-of-lines evolution",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "summary": payload["summary"],
        "acceptance": acceptance,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
