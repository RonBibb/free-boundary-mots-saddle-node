#!/usr/bin/env python3
"""Complete the zero-step nonlinear metric and scalar compact-wall solve."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.nonlinear_wall_acceleration import (
    scalar_wall_second_corner_fields,
    solve_scalar_wall_accelerations,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_g7
from run_corrected_fold_nonlinear_rhs_metric_wall_solve import (
    MATCHED_R,
    relative_norm,
    repair_metric_wall_rows,
)
from run_corrected_fold_nonlinear_rhs_wall_join import assemble, summarize_rows
from run_corrected_fold_regular_so3_runtime import build_geometry


OUTPUT = Path("results/corrected_fold_nonlinear_rhs_complete_wall_solve.json")


def scalar_summary(geometry, fields):
    rows = scalar_wall_second_corner_fields(
        fields["acceleration"]["Dz"],
        np.asarray(geometry["psi"]) * np.exp(np.asarray(geometry["a"])),
        fields["acceleration"]["zz"],
        geometry["phi"], fields["phi_acceleration"],
        fields["chi_acceleration"], geometry["background"], radial_buffer=7,
    )
    records = []
    for wall in rows["walls"]:
        records.append({
            "wall": wall["wall"],
            "phi_maximum_normalized": float(np.max(np.abs(wall["phi_residual"]) / wall["phi_scale"])),
            "chi_maximum_normalized": float(np.max(np.abs(wall["chi_residual"]) / wall["chi_scale"])),
        })
    return {
        "walls": records,
        "maximum_phi": max(record["phi_maximum_normalized"] for record in records),
        "maximum_chi": max(record["chi_maximum_normalized"] for record in records),
    }


def matched_scalar_wall_vector(geometry, fields):
    radius = np.asarray(geometry["r"])
    result = []
    for index in (0, -1):
        result.append(np.stack((
            CubicSpline(radius, fields["phi_acceleration"][index])(np.asarray(MATCHED_R)),
            CubicSpline(radius, fields["chi_acceleration"][index])(np.asarray(MATCHED_R)),
        ), axis=1))
    return np.asarray(result)


def build_case(geometry, label):
    hybrid, maximum_closure = assemble(geometry, include_wall=False)
    metric_repaired, metric_corrections = repair_metric_wall_rows(geometry, hybrid)
    scalar_control_before = scalar_summary(geometry, metric_repaired)
    scalar_solved = solve_scalar_wall_accelerations(
        metric_repaired["acceleration"]["Dz"],
        np.asarray(geometry["psi"]) * np.exp(np.asarray(geometry["a"])),
        metric_repaired["acceleration"]["zz"],
        geometry["phi"], metric_repaired["phi_acceleration"],
        metric_repaired["chi_acceleration"], geometry["background"],
    )
    scalar_repaired = {
        "acceleration": metric_repaired["acceleration"],
        "lapse_acceleration": metric_repaired["lapse_acceleration"],
        "phi_acceleration": scalar_solved["phi_acceleration"],
        "chi_acceleration": scalar_solved["chi_acceleration"],
    }
    # The scalar endpoint update changes the Phi_tt forcing in the Israel
    # rows.  Re-solve those metric endpoint rows once.  The scalar rows depend
    # on g_zz,tt, which is a compact-normal/gauge datum left unchanged by the
    # tangential/time-time metric solve, so this closes the coupled block.
    complete, coupled_metric_corrections = repair_metric_wall_rows(
        geometry, scalar_repaired,
    )
    metric_rows_after_scalar = summarize_rows(geometry, complete)
    scalar_rows_after = scalar_summary(geometry, complete)
    return {
        "resolution": label,
        "source_grid": geometry["source_grid"],
        "selector_maximum": geometry["selector_maximum"],
        "maximum_postsolve_bulk_closure": maximum_closure,
        "metric_wall_corrections": metric_corrections,
        "coupled_metric_wall_corrections": coupled_metric_corrections,
        "scalar_wall_corrections": scalar_solved["corrections"],
        "scalar_rows_before_solve": scalar_control_before,
        "scalar_rows_after_solve": scalar_rows_after,
        "metric_rows_after_scalar_solve": metric_rows_after_scalar,
        "_matched_scalar_wall_vector": matched_scalar_wall_vector(geometry, complete),
    }


def main():
    g6 = build_geometry("G6")
    g7 = build_g7(g6)
    cases = [build_case(g6, "G6"), build_case(g7, "G7")]
    maximum_phi = max(case["scalar_rows_after_solve"]["maximum_phi"] for case in cases)
    maximum_chi = max(case["scalar_rows_after_solve"]["maximum_chi"] for case in cases)
    maximum_spatial = max(case["metric_rows_after_scalar_solve"]["maximum_spatial"] for case in cases)
    maximum_time = max(case["metric_rows_after_scalar_solve"]["maximum_time_time"] for case in cases)
    maximum_closure = max(case["maximum_postsolve_bulk_closure"] for case in cases)
    scalar_transfer = relative_norm(
        cases[0]["_matched_scalar_wall_vector"], cases[1]["_matched_scalar_wall_vector"],
    )
    maximum_scalar_correction = max(
        correction["relative_norm"]
        for case in cases for correction in case["scalar_wall_corrections"]
    )
    acceptance = {
        "selectors_below_1e_8": bool(all(case["selector_maximum"] < 1e-8 for case in cases)),
        "bulk_closure_below_1e_9": bool(maximum_closure < 1e-9),
        "complete_spatial_Israel_rows_below_1e_9": bool(maximum_spatial < 1e-9),
        "complete_time_time_Israel_rows_below_1e_9": bool(maximum_time < 1e-9),
        "complete_stabilizer_Robin_rows_below_1e_9": bool(maximum_phi < 1e-9),
        "complete_collapse_scalar_Neumann_rows_below_1e_9": bool(maximum_chi < 1e-9),
        "solved_scalar_wall_G6_G7_transfer_below_5_percent": bool(scalar_transfer < 0.05),
    }
    summary = {
        "maximum_complete_spatial_Israel_row": maximum_spatial,
        "maximum_complete_time_time_Israel_row": maximum_time,
        "maximum_complete_stabilizer_Robin_row": maximum_phi,
        "maximum_complete_collapse_scalar_Neumann_row": maximum_chi,
        "maximum_postsolve_bulk_closure": maximum_closure,
        "solved_scalar_wall_G6_G7_relative_difference": scalar_transfer,
        "maximum_scalar_endpoint_correction_relative_norm": maximum_scalar_correction,
    }
    for case in cases:
        case.pop("_matched_scalar_wall_vector")
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "scope": "complete zero-step nonlinear compact-wall acceleration solve joined to the corrected-fold open-bulk RHS",
        "matched_r": list(MATCHED_R),
        "cases": cases,
        "summary": summary,
        "acceptance": acceptance,
        "interpretation": [
            "Metric endpoint accelerations satisfy differentiated Israel rows using the solved bulk collar.",
            "Stabilizer and collapse-scalar endpoint accelerations satisfy the twice-time-differentiated coordinate wall conditions.",
            "The coupled update is checked again against the metric rows because the Israel forcing contains the stabilizer acceleration.",
        ],
        "limitations": [
            "zero-step acceleration compatibility rather than a nonlinear characteristic boundary integrator",
            "regular axis collar acceleration remains archived at r=0",
            "fixed-stage initial gauge-source jets",
            "finite-time constraint and horizon evolution remain open",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "summary": summary, "acceptance": acceptance}, indent=2), flush=True)


if __name__ == "__main__":
    main()
