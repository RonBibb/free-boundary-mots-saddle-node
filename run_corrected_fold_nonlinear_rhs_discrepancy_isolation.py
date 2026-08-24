#!/usr/bin/env python3
"""Isolate the failed nonlinear RHS admission metrics without moving its gate.

The original admission audit sampled the two physical compact walls and used
pointwise relative acceleration differences, including where an acceleration
crosses zero.  This diagnostic keeps that result intact and asks two narrower
questions:

1. Is the metric discrepancy a bulk defect or a physical-wall endpoint defect?
2. Is the large stabilizer percentage a large equation residual or division by
   a near-zero acceleration?
"""

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


WALL_Z_FRACTIONS = (0.0, 1.0)
INTERIOR_Z_FRACTIONS = (0.05, 0.25, 0.5, 0.75, 0.95)
R_VALUES = (0.25, 0.75, 1.5, 3.0, 4.0)
OUTPUT = Path("results/corrected_fold_nonlinear_rhs_discrepancy_isolation.json")


def relative_norm(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(left), np.linalg.norm(right), 1e-300))


def local_record(geometry, z_value, r_value, region):
    background = geometry["jet_field"].at(z_value, r_value)
    full_geometry = metric_geometry_from_jets(
        background["metric"], background["metric_first"], background["metric_second"],
    )
    source = full_geometry["contracted_christoffel_covector"]
    source_first = full_geometry["contracted_christoffel_covector_first"]
    archived = reduced_einstein_two_scalar_residual(
        background["metric"], background["metric_first"], background["metric_second"],
        background["phi"], background["phi_first"], background["phi_second"],
        background["chi"], background["chi_first"], background["chi_second"],
        source, source_first, mass_squared=geometry["mass_squared"],
        potential_offset=-6.0, kappa5_squared=1.0,
    )
    solved = solve_reduced_einstein_two_scalar_acceleration(
        background["metric"], background["metric_first"], background["metric_second"],
        background["phi"], background["phi_first"], background["phi_second"],
        background["chi"], background["chi_first"], background["chi_second"],
        source, source_first, mass_squared=geometry["mass_squared"],
        potential_offset=-6.0, kappa5_squared=1.0,
    )

    diagonal = np.diag(background["metric"])
    frame = 1 / np.sqrt(np.maximum(np.abs(diagonal), 1e-300))
    archived_metric_acceleration = np.asarray(background["metric_second"])[0, 0]
    archived_orthonormal = archived_metric_acceleration * frame[:, None] * frame[None, :]
    solved_orthonormal = solved["metric_acceleration"] * frame[:, None] * frame[None, :]
    correction = solved_orthonormal - archived_orthonormal

    residual_orthonormal = archived["metric_residual"] * frame[:, None] * frame[None, :]
    reduced_orthonormal = archived["reduced_ricci"] * frame[:, None] * frame[None, :]
    matter_orthonormal = reduced_orthonormal - residual_orthonormal
    metric_equation_scale = max(
        float(np.linalg.norm(reduced_orthonormal)),
        float(np.linalg.norm(matter_orthonormal)),
        1e-300,
    )
    inverse = archived["inverse_metric"]
    phi_term_scale = max(
        float(np.sum(np.abs(inverse * archived["phi_covariant_hessian"])) + abs(archived["potential_prime"])),
        1e-300,
    )
    chi_term_scale = max(
        float(np.sum(np.abs(inverse * archived["chi_covariant_hessian"]))),
        1e-300,
    )
    max_component = np.unravel_index(np.argmax(np.abs(correction)), correction.shape)
    archived_phi = float(np.asarray(background["phi_second"])[0, 0])
    archived_chi = float(np.asarray(background["chi_second"])[0, 0])
    return {
        "region": region,
        "z": float(z_value),
        "r": float(r_value),
        "metric_acceleration_relative_difference": relative_norm(solved_orthonormal, archived_orthonormal),
        "metric_acceleration_correction_orthonormal_norm": float(np.linalg.norm(correction)),
        "metric_acceleration_correction_maximum_component": [int(value) for value in max_component],
        "metric_acceleration_correction_maximum_absolute": float(np.max(np.abs(correction))),
        "metric_equation_residual_relative": float(np.linalg.norm(residual_orthonormal) / metric_equation_scale),
        "phi_acceleration_relative_difference": float(
            abs(solved["phi_acceleration"] - archived_phi)
            / max(abs(solved["phi_acceleration"]), abs(archived_phi), 1e-300)
        ),
        "phi_acceleration_absolute_difference": float(abs(solved["phi_acceleration"] - archived_phi)),
        "phi_equation_residual_relative": float(abs(archived["phi_residual"]) / phi_term_scale),
        "chi_acceleration_relative_difference": float(
            abs(solved["chi_acceleration"] - archived_chi)
            / max(abs(solved["chi_acceleration"]), abs(archived_chi), 1e-300)
        ),
        "chi_acceleration_absolute_difference": float(abs(solved["chi_acceleration"] - archived_chi)),
        "chi_equation_residual_relative": float(abs(archived["chi_residual"]) / chi_term_scale),
        "archived_phi_acceleration": archived_phi,
        "solved_phi_acceleration": solved["phi_acceleration"],
        "finite": solved["finite"],
        "_solved_vector": np.r_[solved_orthonormal.ravel(), solved["phi_acceleration"], solved["chi_acceleration"]],
    }


def build_case(resolution):
    geometry = build_geometry(resolution)
    z0 = float(geometry["z"][0])
    z1 = float(geometry["z"][-1])
    records = []
    for region, fractions in (("wall", WALL_Z_FRACTIONS), ("bulk_interior", INTERIOR_Z_FRACTIONS)):
        for fraction in fractions:
            z_value = z0 + fraction * (z1 - z0)
            records.extend(local_record(geometry, z_value, r_value, region) for r_value in R_VALUES)
    return {
        "resolution": resolution,
        "source_grid": geometry["source_grid"],
        "selector_maximum": geometry["selector_maximum"],
        "records": records,
    }


def maximum(case, key, region):
    return max(record[key] for record in case["records"] if record["region"] == region)


def main():
    cases = [build_case(resolution) for resolution in ("G5", "G6")]
    coarse, fine = cases
    coarse_bulk = np.asarray([record["_solved_vector"] for record in coarse["records"] if record["region"] == "bulk_interior"])
    fine_bulk = np.asarray([record["_solved_vector"] for record in fine["records"] if record["region"] == "bulk_interior"])
    summary = {
        "fine_wall_metric_acceleration_relative_difference": maximum(fine, "metric_acceleration_relative_difference", "wall"),
        "fine_bulk_metric_acceleration_relative_difference": maximum(fine, "metric_acceleration_relative_difference", "bulk_interior"),
        "fine_wall_metric_equation_residual_relative": maximum(fine, "metric_equation_residual_relative", "wall"),
        "fine_bulk_metric_equation_residual_relative": maximum(fine, "metric_equation_residual_relative", "bulk_interior"),
        "fine_bulk_phi_acceleration_relative_difference": maximum(fine, "phi_acceleration_relative_difference", "bulk_interior"),
        "fine_bulk_phi_acceleration_absolute_difference": maximum(fine, "phi_acceleration_absolute_difference", "bulk_interior"),
        "fine_bulk_phi_equation_residual_relative": maximum(fine, "phi_equation_residual_relative", "bulk_interior"),
        "fine_bulk_chi_equation_residual_relative": maximum(fine, "chi_equation_residual_relative", "bulk_interior"),
        "solved_bulk_rhs_cross_grid_relative_difference": relative_norm(coarse_bulk, fine_bulk),
    }
    for case in cases:
        for record in case["records"]:
            record.pop("_solved_vector")
    payload = {
        "status": "diagnostic",
        "scope": "root-cause isolation for the failed nonlinear pointwise RHS admission gate",
        "wall_z_fractions": list(WALL_Z_FRACTIONS),
        "interior_z_fractions": list(INTERIOR_Z_FRACTIONS),
        "r_values": list(R_VALUES),
        "cases": cases,
        "summary": summary,
        "interpretation_rules": [
            "The original sealed admission result remains failed and is not rescored by this diagnostic.",
            "Open-bulk Einstein equations and physical-wall Israel equations are distinct rows; a wall-localized mismatch does not by itself diagnose a bulk RHS failure.",
            "Relative acceleration differences are singular diagnostics when the reference acceleration crosses zero; equation-term-normalized residuals and absolute differences are reported separately.",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
