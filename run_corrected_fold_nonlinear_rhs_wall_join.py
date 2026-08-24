#!/usr/bin/env python3
"""Join the nonlinear open-bulk acceleration to compact-wall corner data."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.anisotropic_geometry import anisotropic_spatial_israel_second_corner_fields
from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.lapse_acceleration_corner import time_time_israel_second_corner_fields
from bhps.linearized_gh_einstein_scalar import (
    metric_geometry_from_jets,
    reduced_einstein_two_scalar_residual,
    solve_reduced_einstein_two_scalar_acceleration,
)
from bhps.regular_so3_gh_reduction import regular_so3_perturbation_jets
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_g7
from run_corrected_fold_regular_so3_runtime import build_geometry


STENCIL_WIDTH = 7
COLLAR_DEPTH = STENCIL_WIDTH - 1
RADIAL_BUFFER = 7
OUTPUT = Path("results/corrected_fold_nonlinear_rhs_wall_join.json")


def archived_fields(geometry):
    reduced_tt = np.asarray(geometry["jet_field"].reduced_second[0, 0])
    radius = np.asarray(geometry["r"])[None, :]
    transverse = reduced_tt[:, :, 3]
    acceleration = {
        "zz": reduced_tt[:, :, 6].copy(),
        "radial": (transverse + radius**2 * reduced_tt[:, :, 4]).copy(),
        "transverse": transverse.copy(),
        "zr": (radius * reduced_tt[:, :, 1]).copy(),
        "Dz": derivative_matrix(np.asarray(geometry["z"]), 1, STENCIL_WIDTH),
    }
    h00 = reduced_tt[:, :, 2]
    return {
        "acceleration": acceleration,
        "lapse_acceleration": -h00 / (2 * np.asarray(geometry["psi"])),
        "phi_acceleration": reduced_tt[:, :, 7].copy(),
        "chi_acceleration": reduced_tt[:, :, 8].copy(),
    }


def native_background(geometry, i, j):
    jet = geometry["jet_field"]
    return regular_so3_perturbation_jets(
        float(geometry["r"][j]),
        jet.reduced_fields[i, j],
        jet.reduced_first[:, i, j],
        jet.reduced_second[:, :, i, j],
    )


def solved_native_acceleration(geometry, i, j):
    background = native_background(geometry, i, j)
    full_geometry = metric_geometry_from_jets(
        background["metric"], background["metric_first"], background["metric_second"],
    )
    solved = solve_reduced_einstein_two_scalar_acceleration(
        background["metric"], background["metric_first"], background["metric_second"],
        background["phi"], background["phi_first"], background["phi_second"],
        background["chi"], background["chi_first"], background["chi_second"],
        full_geometry["contracted_christoffel_covector"],
        full_geometry["contracted_christoffel_covector_first"],
        mass_squared=geometry["mass_squared"], potential_offset=-6.0,
        kappa5_squared=1.0,
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
        full_geometry["contracted_christoffel_covector"],
        full_geometry["contracted_christoffel_covector_first"],
        mass_squared=geometry["mass_squared"], potential_offset=-6.0,
        kappa5_squared=1.0,
    )
    maximum_closure = max(
        float(np.max(np.abs(closure["metric_residual"]))),
        float(abs(closure["phi_residual"])),
        float(abs(closure["chi_residual"])),
    )
    return solved, maximum_closure


def fill_point(fields, geometry, i, j, solved):
    acceleration = np.asarray(solved["metric_acceleration"])
    transverse = 0.5 * (acceleration[3, 3] + acceleration[4, 4])
    fields["acceleration"]["zz"][i, j] = acceleration[1, 1]
    fields["acceleration"]["radial"][i, j] = acceleration[2, 2]
    fields["acceleration"]["transverse"][i, j] = transverse
    fields["acceleration"]["zr"][i, j] = acceleration[1, 2]
    fields["lapse_acceleration"][i, j] = -acceleration[0, 0] / (2 * geometry["psi"][i, j])
    fields["phi_acceleration"][i, j] = solved["phi_acceleration"]
    fields["chi_acceleration"][i, j] = solved["chi_acceleration"]


def assemble(geometry, include_wall):
    fields = archived_fields(geometry)
    nz = len(geometry["z"])
    nr = len(geometry["r"])
    collar_indices = list(range(0 if include_wall else 1, COLLAR_DEPTH + 1))
    collar_indices += list(range(nz - COLLAR_DEPTH - 1, nz if include_wall else nz - 1))
    maximum_closure = 0.0
    for i in collar_indices:
        for j in range(1, nr):
            solved, closure = solved_native_acceleration(geometry, i, j)
            fill_point(fields, geometry, i, j, solved)
            maximum_closure = max(maximum_closure, closure)
    return fields, maximum_closure


def scalar_wall_rows(geometry, fields):
    z = np.asarray(geometry["z"])
    dz = fields["acceleration"]["Dz"]
    psi = np.asarray(geometry["psi"])
    a = np.asarray(geometry["a"])
    phi = np.asarray(geometry["phi"])
    A = psi * np.exp(a)
    gamma = float(geometry["background"]["wall_stiffness"])
    phi_tt = fields["phi_acceleration"]
    chi_tt = fields["chi_acceleration"]
    zz_tt = fields["acceleration"]["zz"]
    rows = []
    for wall, index, target, sign in (
        ("lower", 0, float(geometry["background"]["v0"]), -1.0),
        ("upper", -1, float(geometry["background"]["v1"]), 1.0),
    ):
        phi_terms = (
            (dz @ phi_tt)[index],
            sign * 0.5 * gamma * A[index] * phi_tt[index],
            sign * 0.25 * gamma * (phi[index] - target) * zz_tt[index] / A[index],
        )
        phi_residual = sum(phi_terms)
        phi_scale = np.maximum(1.0, sum(np.abs(term) for term in phi_terms))
        chi_residual = (dz @ chi_tt)[index]
        chi_scale = np.maximum(1.0, np.abs(chi_residual))
        retained = slice(None, -RADIAL_BUFFER)
        rows.append({
            "wall": wall,
            "phi_maximum_normalized": float(np.max(np.abs(phi_residual[retained]) / phi_scale[retained])),
            "chi_maximum_normalized": float(np.max(np.abs(chi_residual[retained]) / chi_scale[retained])),
        })
    return rows


def summarize_rows(geometry, fields):
    spatial = anisotropic_spatial_israel_second_corner_fields(
        fields["acceleration"], geometry["psi"], geometry["a"], geometry["b"],
        geometry["c"], geometry["phi"], geometry["background"],
        fields["phi_acceleration"], RADIAL_BUFFER,
    )
    temporal = time_time_israel_second_corner_fields(
        fields["acceleration"], geometry["psi"], geometry["psi"], geometry["a"],
        geometry["phi"], geometry["background"], fields["phi_acceleration"],
        fields["lapse_acceleration"], RADIAL_BUFFER,
    )
    spatial_rows = []
    for wall in spatial["walls"]:
        for component in ("radial", "transverse"):
            item = wall["tangential_components"][component]
            spatial_rows.append({
                "wall": wall["wall"],
                "row": component,
                "maximum_normalized": float(np.max(np.abs(item["residual"]) / item["scale"])),
            })
        spatial_rows.append({
            "wall": wall["wall"],
            "row": "mixed_zr",
            "maximum_normalized": float(np.max(np.abs(wall["mixed_zr_residual"]) / wall["mixed_zr_scale"])),
        })
    temporal_rows = [
        {
            "wall": wall["wall"],
            "maximum_normalized": float(np.max(np.abs(wall["residual"]) / wall["scale"])),
        }
        for wall in temporal["walls"]
    ]
    scalar_rows = scalar_wall_rows(geometry, fields)
    return {
        "spatial_rows": spatial_rows,
        "time_time_rows": temporal_rows,
        "scalar_rows": scalar_rows,
        "maximum_spatial": max(row["maximum_normalized"] for row in spatial_rows),
        "maximum_time_time": max(row["maximum_normalized"] for row in temporal_rows),
        "maximum_phi": max(row["phi_maximum_normalized"] for row in scalar_rows),
        "maximum_chi": max(row["chi_maximum_normalized"] for row in scalar_rows),
    }


def build_case(geometry, label):
    archived = archived_fields(geometry)
    hybrid, hybrid_closure = assemble(geometry, include_wall=False)
    bulk_everywhere, bulk_closure = assemble(geometry, include_wall=True)
    return {
        "resolution": label,
        "source_grid": geometry["source_grid"],
        "selector_maximum": geometry["selector_maximum"],
        "archived_boundary_rows": summarize_rows(geometry, archived),
        "hybrid_boundary_rows": summarize_rows(geometry, hybrid),
        "bulk_everywhere_boundary_rows": summarize_rows(geometry, bulk_everywhere),
        "hybrid_maximum_postsolve_bulk_closure": hybrid_closure,
        "bulk_everywhere_maximum_postsolve_closure": bulk_closure,
    }


def main():
    g6 = build_geometry("G6")
    g7 = build_g7(g6)
    cases = [build_case(g6, "G6"), build_case(g7, "G7")]
    hybrid_spatial = max(case["hybrid_boundary_rows"]["maximum_spatial"] for case in cases)
    hybrid_time = max(case["hybrid_boundary_rows"]["maximum_time_time"] for case in cases)
    hybrid_phi = max(case["hybrid_boundary_rows"]["maximum_phi"] for case in cases)
    hybrid_chi = max(case["hybrid_boundary_rows"]["maximum_chi"] for case in cases)
    maximum_closure = max(case["hybrid_maximum_postsolve_bulk_closure"] for case in cases)
    acceptance = {
        "selectors_below_1e_8": bool(all(case["selector_maximum"] < 1e-8 for case in cases)),
        "hybrid_bulk_closure_below_1e_9": bool(maximum_closure < 1e-9),
        "hybrid_spatial_Israel_rows_below_0_025": bool(hybrid_spatial < 0.025),
        "hybrid_time_time_Israel_rows_below_0_025": bool(hybrid_time < 0.025),
        "hybrid_stabilizer_Robin_row_below_0_025": bool(hybrid_phi < 0.025),
        "hybrid_collapse_scalar_Neumann_row_below_0_025": bool(hybrid_chi < 0.025),
    }
    summary = {
        "hybrid_maximum_spatial_Israel_row": hybrid_spatial,
        "hybrid_maximum_time_time_Israel_row": hybrid_time,
        "hybrid_maximum_stabilizer_Robin_row": hybrid_phi,
        "hybrid_maximum_collapse_scalar_Neumann_row": hybrid_chi,
        "hybrid_maximum_postsolve_bulk_closure": maximum_closure,
    }
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "scope": "zero-step join of the native-grid nonlinear open-bulk acceleration to archived compact-wall-compatible endpoint accelerations",
        "stencil_width": STENCIL_WIDTH,
        "collar_depth": COLLAR_DEPTH,
        "radial_buffer": RADIAL_BUFFER,
        "cases": cases,
        "summary": summary,
        "acceptance": acceptance,
        "interpretation": [
            "The archived case is the existing boundary-compatible acceleration control.",
            "The hybrid case keeps the endpoint accelerations and replaces the open-bulk collar with the nonlinear pointwise solve.",
            "The bulk-everywhere case is a negative control that incorrectly evaluates the open-bulk RHS at physical-wall endpoints.",
        ],
        "limitations": [
            "zero-step second-corner compatibility rather than a nonlinear characteristic boundary integrator",
            "the regular axis node retains its archived acceleration while positive-radius collar nodes use the solved bulk RHS",
            "initial fixed-stage gauge-source jets",
            "finite-time evolution remains open",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "summary": summary, "acceptance": acceptance}, indent=2), flush=True)


if __name__ == "__main__":
    main()
