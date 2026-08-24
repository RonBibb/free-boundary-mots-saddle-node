#!/usr/bin/env python3
"""Solve the nonlinear metric wall-acceleration rows against the bulk collar."""

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_g7
from run_corrected_fold_nonlinear_rhs_wall_join import (
    assemble,
    archived_fields,
    summarize_rows,
)
from run_corrected_fold_regular_so3_runtime import build_geometry


MATCHED_R = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
OUTPUT = Path("results/corrected_fold_nonlinear_rhs_metric_wall_solve.json")


def relative_norm(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(left), np.linalg.norm(right), 1e-300))


def wall_coefficients(geometry, index, upper):
    phi = np.asarray(geometry["phi"])
    background = geometry["background"]
    gamma = float(background["wall_stiffness"])
    target = float(background["v1"] if upper else background["v0"])
    potential = 0.5 * gamma * (phi[index] - target) ** 2
    if upper:
        beta = float(background["beta_b"]) - (potential - float(background["wall_potential_b"])) / 6
        beta_phi = -gamma * (phi[index] - target) / 6
    else:
        beta = float(background["beta_a"]) + (potential - float(background["wall_potential_a"])) / 6
        beta_phi = gamma * (phi[index] - target) / 6
    return beta, beta_phi


def solve_robin_endpoint(dz, field, index, robin, forcing):
    diagonal = float(dz[index, index])
    derivative_without_endpoint = (dz @ field)[index] - diagonal * field[index]
    denominator = diagonal + robin
    if np.any(np.abs(denominator) < 1e-12):
        raise RuntimeError("degenerate wall acceleration row")
    field[index] = -(derivative_without_endpoint + forcing) / denominator


def repair_metric_wall_rows(geometry, hybrid):
    fields = {
        "acceleration": {
            key: value.copy() if key != "Dz" else value
            for key, value in hybrid["acceleration"].items()
        },
        "lapse_acceleration": hybrid["lapse_acceleration"].copy(),
        "phi_acceleration": hybrid["phi_acceleration"].copy(),
        "chi_acceleration": hybrid["chi_acceleration"].copy(),
    }
    dz = fields["acceleration"]["Dz"]
    psi = np.asarray(geometry["psi"])
    a = np.asarray(geometry["a"])
    b = np.asarray(geometry["b"])
    c = np.asarray(geometry["c"])
    alpha = psi
    A = psi * np.exp(a)
    metric = {
        "radial": psi**2 * np.exp(2 * b),
        "transverse": psi**2 * np.exp(2 * c),
    }
    wall_records = []
    for wall, index, upper in (("lower", 0, False), ("upper", -1, True)):
        beta, beta_phi = wall_coefficients(geometry, index, upper)
        before = {}
        for name in ("radial", "transverse"):
            value = fields["acceleration"][name]
            before[name] = value[index].copy()
            forcing = (
                beta * metric[name][index] * fields["acceleration"]["zz"][index] / A[index]
                + 2 * beta_phi * fields["phi_acceleration"][index] * A[index] * metric[name][index]
            )
            solve_robin_endpoint(dz, value, index, 2 * beta * A[index], forcing)
        before["lapse"] = fields["lapse_acceleration"][index].copy()
        metric_tt = -alpha**2
        metric_tt_acceleration = -2 * alpha * fields["lapse_acceleration"]
        forcing = (
            beta * metric_tt[index] * fields["acceleration"]["zz"][index] / A[index]
            + 2 * beta_phi * fields["phi_acceleration"][index] * A[index] * metric_tt[index]
        )
        solve_robin_endpoint(
            dz, metric_tt_acceleration, index, 2 * beta * A[index], forcing,
        )
        fields["lapse_acceleration"][index] = -metric_tt_acceleration[index] / (2 * alpha[index])
        before["mixed_zr"] = fields["acceleration"]["zr"][index].copy()
        fields["acceleration"]["zr"][index] = 0.0
        after = {
            "radial": fields["acceleration"]["radial"][index].copy(),
            "transverse": fields["acceleration"]["transverse"][index].copy(),
            "lapse": fields["lapse_acceleration"][index].copy(),
            "mixed_zr": fields["acceleration"]["zr"][index].copy(),
        }
        before_vector = np.concatenate([before[key] for key in ("radial", "transverse", "lapse", "mixed_zr")])
        after_vector = np.concatenate([after[key] for key in ("radial", "transverse", "lapse", "mixed_zr")])
        wall_records.append({
            "wall": wall,
            "endpoint_correction_relative_norm": relative_norm(after_vector, before_vector),
            "endpoint_correction_maximum_absolute": float(np.max(np.abs(after_vector - before_vector))),
        })
    return fields, wall_records


def matched_wall_vector(geometry, fields):
    radius = np.asarray(geometry["r"])
    result = []
    for index in (0, -1):
        components = []
        for values in (
            fields["acceleration"]["radial"],
            fields["acceleration"]["transverse"],
            fields["lapse_acceleration"],
            fields["acceleration"]["zr"],
        ):
            components.append(CubicSpline(radius, values[index])(np.asarray(MATCHED_R)))
        result.append(np.stack(components, axis=1))
    return np.asarray(result)


def build_case(geometry, label):
    archived = archived_fields(geometry)
    hybrid, maximum_closure = assemble(geometry, include_wall=False)
    repaired, wall_records = repair_metric_wall_rows(geometry, hybrid)
    return {
        "resolution": label,
        "source_grid": geometry["source_grid"],
        "selector_maximum": geometry["selector_maximum"],
        "archived_boundary_rows": summarize_rows(geometry, archived),
        "unrepaired_boundary_rows": summarize_rows(geometry, hybrid),
        "repaired_boundary_rows": summarize_rows(geometry, repaired),
        "wall_corrections": wall_records,
        "maximum_postsolve_bulk_closure": maximum_closure,
        "_matched_wall_vector": matched_wall_vector(geometry, repaired),
    }


def main():
    g6 = build_geometry("G6")
    g7 = build_g7(g6)
    cases = [build_case(g6, "G6"), build_case(g7, "G7")]
    maximum_spatial = max(case["repaired_boundary_rows"]["maximum_spatial"] for case in cases)
    maximum_time = max(case["repaired_boundary_rows"]["maximum_time_time"] for case in cases)
    maximum_closure = max(case["maximum_postsolve_bulk_closure"] for case in cases)
    wall_transfer = relative_norm(cases[0]["_matched_wall_vector"], cases[1]["_matched_wall_vector"])
    maximum_correction = max(
        wall["endpoint_correction_relative_norm"] for case in cases for wall in case["wall_corrections"]
    )
    acceptance = {
        "selectors_below_1e_8": bool(all(case["selector_maximum"] < 1e-8 for case in cases)),
        "bulk_closure_below_1e_9": bool(maximum_closure < 1e-9),
        "solved_spatial_Israel_rows_below_1e_9": bool(maximum_spatial < 1e-9),
        "solved_time_time_Israel_rows_below_1e_9": bool(maximum_time < 1e-9),
        "solved_metric_wall_G6_G7_transfer_below_5_percent": bool(wall_transfer < 0.05),
    }
    summary = {
        "maximum_repaired_spatial_Israel_row": maximum_spatial,
        "maximum_repaired_time_time_Israel_row": maximum_time,
        "maximum_postsolve_bulk_closure": maximum_closure,
        "repaired_metric_wall_G6_G7_relative_difference": wall_transfer,
        "maximum_endpoint_correction_relative_norm": maximum_correction,
    }
    for case in cases:
        case.pop("_matched_wall_vector")
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "scope": "zero-step nonlinear metric wall-acceleration solve joined to the G6/G7 open-bulk collar",
        "matched_r": list(MATCHED_R),
        "cases": cases,
        "summary": summary,
        "acceptance": acceptance,
        "interpretation": [
            "Tangential spatial and time-time endpoint accelerations are solved from the differentiated nonlinear Israel rows using the current bulk collar.",
            "The compact-normal and stabilizer endpoint accelerations remain separate gauge and scalar boundary data.",
            "The scalar rows reported inside each boundary summary are diagnostic only because their provisional formula fails the archived control and is not used for acceptance.",
        ],
        "limitations": [
            "zero-step boundary solve rather than a characteristic nonlinear boundary integrator",
            "nonlinear stabilizer and collapse-scalar wall rows remain open",
            "axis collar acceleration remains archived at r=0",
            "finite-time evolution remains open",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "summary": summary, "acceptance": acceptance}, indent=2), flush=True)


if __name__ == "__main__":
    main()
