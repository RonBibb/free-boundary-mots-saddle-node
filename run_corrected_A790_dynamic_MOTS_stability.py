#!/usr/bin/env python3
"""Sealed principal MOTS-stability audit of the formed A=7.90 cap pair."""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice
from bhps.dynamical_capped_horizon_bvp import solve_dynamical_capped_surface_bvp
from bhps.dynamical_mots_stability import mots_stability_matrix, public_stability
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry


OUTPUT = Path("results/corrected_A790_dynamic_MOTS_stability.json")
FINE_STATE = Path("results/corrected_A790_formation_time_refinement_state.npz")
LONG_STATE = Path("results/corrected_A790_two_grid_formation_search_state.npz")
PROTOCOL = "notes/84_A790_dynamic_MOTS_stability_protocol.md"
AMPLITUDE = 7.90
TIMES = (0.000625, 0.001, 0.004)
BRANCH_SEEDS = {"inner": 1.30, "outer": 1.55}
NODES = (33, 49, 65, 81)
PRIMARY_STEP = 1e-5
CHECK_STEP = 2e-5


def flat_state(z, r):
    position = np.zeros((len(z), len(r), 9))
    position[:, :, 2] = -1.0
    position[:, :, 3] = 1.0
    position[:, :, 6] = 1.0
    return position


def analytic_control():
    z = np.linspace(0.0, 4.0, 49)
    r = np.linspace(0.0, 3.0, 65)
    position = flat_state(z, r)
    k = 0.8
    velocity = np.zeros_like(position)
    velocity[:, :, 3] = -2.0 * k
    velocity[:, :, 6] = -2.0 * k
    surface = solve_dynamical_capped_surface_bvp(
        position, velocity, z, r, 1.1, tolerance=1e-8,
        nodes=101, dense_nodes=501,
    )
    spectra = {
        str(nodes): mots_stability_matrix(
            position, velocity, z, r, surface, nodes=nodes,
            relative_step=PRIMARY_STEP,
        ) for nodes in NODES
    }
    check = mots_stability_matrix(
        position, velocity, z, r, surface, nodes=81,
        relative_step=CHECK_STEP,
    )
    fine = spectra["81"]
    expected = (-1.92, 3.20)
    measured = tuple(item["real"] for item in fine["leading_eigenvalues"][:2])
    matrix_step_difference = float(
        np.linalg.norm(fine["matrix"] - check["matrix"])
        / max(np.linalg.norm(fine["matrix"]), 1e-300)
    )
    passed = bool(
        max(abs(value - target) / abs(target) for value, target in zip(measured, expected)) < .005
        and abs(fine["principal_eigenvalue_imaginary"]) < 1e-8
        and fine["principal_eigenfunction_sign_changes"] == 0
        and max(fine["left_neumann_defect"], fine["right_neumann_defect"]) < 1e-10
        and matrix_step_difference < 2e-6
    )
    return {
        "passed": passed,
        "expected_first_two_eigenvalues": list(expected),
        "measured_first_two_eigenvalues": list(measured),
        "matrix_step_relative_difference": matrix_step_difference,
        "spectra": {key: public_stability(value) for key, value in spectra.items()},
        "check_step_81": public_stability(check),
    }


def recover_surface(position, velocity, geometry, seed):
    return solve_dynamical_capped_surface_bvp(
        position, velocity, geometry["z"], geometry["r"], seed,
        tolerance=2e-5, nodes=121, maximum_nodes=6000, dense_nodes=501,
    )


def surface_passes(surface):
    check = surface.get("primary_evaluator_crosscheck", {})
    return bool(
        surface["converged"] and surface["in_domain"]
        and surface["local_expansion_interior_maximum"] < 2e-4
        and surface["boundary_slope_error"] < 2e-4
        and "error" not in check
        and check.get("two_cell_interior_maximum", np.inf) < .002
    )


def stability_series(position, velocity, geometry, surface):
    prepared = prepare_capped_expansion_slice(
        position, velocity, geometry["z"], geometry["r"],
    )
    spectra = {
        str(nodes): mots_stability_matrix(
            position, velocity, geometry["z"], geometry["r"], surface,
            nodes=nodes, relative_step=PRIMARY_STEP, prepared=prepared,
        ) for nodes in NODES
    }
    check = mots_stability_matrix(
        position, velocity, geometry["z"], geometry["r"], surface,
        nodes=81, relative_step=CHECK_STEP, prepared=prepared,
    )
    values = {
        int(nodes): record["principal_eigenvalue_real"]
        for nodes, record in spectra.items()
    }
    angular_error = abs(values[81] - values[65])
    step_error = abs(
        values[81] - check["principal_eigenvalue_real"]
    )
    numerical_error = max(angular_error, step_error)
    signs = [np.sign(values[nodes]) for nodes in (49, 65, 81)]
    sign_consistent = bool(signs[0] != 0 and signs[0] == signs[1] == signs[2])
    resolved = bool(
        sign_consistent and abs(values[81]) > 5.0 * max(numerical_error, 1e-14)
    )
    classification = (
        "outward_stable" if resolved and values[81] > 0
        else "outward_unstable" if resolved and values[81] < 0
        else "unresolved"
    )
    return {
        "classification": classification,
        "resolved": resolved,
        "fine_principal_eigenvalue": values[81],
        "angular_difference_65_81": angular_error,
        "Frechet_step_difference_81": step_error,
        "estimated_numerical_error": numerical_error,
        "sign_consistent_49_65_81": sign_consistent,
        "spectra": {
            key: public_stability(record) for key, record in spectra.items()
        },
        "check_step_81": public_stability(check),
    }


def state_at(label, time_value, geometry, fine, long):
    initial = np.asarray(geometry["jet_field"].reduced_fields)
    if np.isclose(time_value, 0.000625):
        increment = fine[f"{label}_fine_time_4_increment"]
        velocity = fine[f"{label}_fine_time_4_velocity"]
    elif np.isclose(time_value, 0.001):
        increment = fine[f"{label}_8step_increment"]
        velocity = fine[f"{label}_8step_velocity"]
    elif np.isclose(time_value, 0.004):
        increment = long[f"{label}_time_3_increment"]
        velocity = long[f"{label}_time_3_velocity"]
    else:
        raise ValueError("unsupported archived time")
    return initial + increment, velocity


def transfer(left, right):
    absolute = abs(left - right)
    relative = absolute / max(abs(left), abs(right), 1e-300)
    return {"absolute": float(absolute), "relative": float(relative)}


def main():
    if not FINE_STATE.exists() or not LONG_STATE.exists():
        raise FileNotFoundError("sealed note-77 and note-74 state archives are required")
    started = time.perf_counter()
    print("running analytic MOTS-stability control", flush=True)
    control = analytic_control()
    print("reconstructing corrected G7/G8 A=7.90 geometries", flush=True)
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": AMPLITUDE}
    geometries = {
        "G7": build_refined(
            seed, 81, 121, "G7A790-MOTS-stability",
            selector_iterations=40, slice_iterations=270,
        ),
    }
    geometries["G8"] = build_refined(
        geometries["G7"], 97, 145, "G8A790-MOTS-stability",
        selector_iterations=45, slice_iterations=280,
    )
    fine = np.load(FINE_STATE)
    long = np.load(LONG_STATE)
    records = {label: {} for label in geometries}
    for label, geometry in geometries.items():
        for time_value in TIMES:
            print(f"{label} t={time_value}: recovering branches", flush=True)
            position, velocity = state_at(label, time_value, geometry, fine, long)
            time_record = {}
            for branch, branch_seed in BRANCH_SEEDS.items():
                print(f"{label} t={time_value} {branch}: stability series", flush=True)
                surface = recover_surface(
                    position, velocity, geometry, branch_seed,
                )
                stability = stability_series(
                    position, velocity, geometry, surface,
                )
                time_record[branch] = {
                    "seed": branch_seed,
                    "surface": {
                        key: surface[key] for key in (
                            "converged", "solver_success", "in_domain",
                            "iterations", "mesh_nodes_used", "rho_axis",
                            "rho_brane", "boundary_slope_error",
                            "local_expansion_interior_maximum",
                            "primary_evaluator_crosscheck",
                        )
                    },
                    "surface_passes": surface_passes(surface),
                    "stability": stability,
                }
            records[label][str(time_value)] = time_record

    grid_transfer = {}
    for time_value in TIMES:
        grid_transfer[str(time_value)] = {}
        for branch in BRANCH_SEEDS:
            left = records["G7"][str(time_value)][branch]["stability"][
                "fine_principal_eigenvalue"
            ]
            right = records["G8"][str(time_value)][branch]["stability"][
                "fine_principal_eigenvalue"
            ]
            grid_transfer[str(time_value)][branch] = transfer(left, right)

    branches = [
        record
        for grid in records.values() for time_record in grid.values()
        for record in time_record.values()
    ]
    matrix_control_pass = all(
        spectrum["left_neumann_defect"] < 1e-10
        and spectrum["right_neumann_defect"] < 1e-10
        and spectrum["minimum_normal_factor"] > 0
        and np.isfinite(spectrum["operator_frobenius_norm"])
        and abs(spectrum["principal_eigenvalue_imaginary"])
        < 1e-6 * max(1.0, abs(spectrum["principal_eigenvalue_real"]))
        and spectrum["principal_eigenfunction_sign_changes"] == 0
        for branch in branches
        for spectrum in branch["stability"]["spectra"].values()
    )
    convergence_pass = all(
        branch["stability"]["angular_difference_65_81"]
        < max(.02, .05 * abs(branch["stability"]["fine_principal_eigenvalue"]))
        and branch["stability"]["Frechet_step_difference_81"]
        < max(.02, .05 * abs(branch["stability"]["fine_principal_eigenvalue"]))
        for branch in branches
    )
    classification_pass = all(
        branch["stability"]["resolved"] for branch in branches
    ) and all(
        records["G7"][str(time_value)][branch]["stability"]["classification"]
        == records["G8"][str(time_value)][branch]["stability"]["classification"]
        and (
            grid_transfer[str(time_value)][branch]["relative"] < .10
            or grid_transfer[str(time_value)][branch]["absolute"] < .02
        )
        for time_value in TIMES for branch in BRANCH_SEEDS
    )
    acceptance = {
        "analytic_and_manufactured_controls_pass": control["passed"],
        "all_physical_surfaces_pass_note80_rules": bool(all(
            branch["surface_passes"] for branch in branches
        )),
        "all_operator_boundary_reality_and_principal_mode_controls_pass": bool(
            matrix_control_pass
        ),
        "all_angular_and_Frechet_step_convergence_rules_pass": bool(
            convergence_pass
        ),
        "all_classifications_resolved_and_transfer_between_grids": bool(
            classification_pass
        ),
    }
    status = "pass" if all(acceptance.values()) else "review"
    classifications = {
        label: {
            time_label: {
                branch: record["stability"]["classification"]
                for branch, record in time_record.items()
            } for time_label, time_record in grid.items()
        } for label, grid in records.items()
    }
    payload = {
        "status": status,
        "classification": (
            "resolved_dynamic_MOTS_principal_stability"
            if status == "pass" else "dynamic_MOTS_stability_review"
        ),
        "scope": "sealed full-ADM principal MOTS-stability audit of corrected A=7.90 formed branches",
        "protocol": PROTOCOL,
        "operator_convention": "L f = delta_(f s) theta_plus; outward stable iff principal eigenvalue >= 0",
        "boundary_condition": "d(delta rho)/dtheta=0 at axis and wall, equivalently d(f/(s.e_rho))/dtheta=0",
        "amplitude": AMPLITUDE,
        "times": list(TIMES),
        "branch_seeds": BRANCH_SEEDS,
        "angular_nodes": list(NODES),
        "Frechet_steps": [PRIMARY_STEP, CHECK_STEP],
        "analytic_control": control,
        "records": records,
        "classifications": classifications,
        "cross_grid_principal_eigenvalue_transfer": grid_transfer,
        "acceptance": acceptance,
        "runtime_seconds": float(time.perf_counter() - started),
        "limitations": [
            "principal SO(3)-invariant sector only; non-principal angular spectrum not computed",
            "linear stability of each MOTS on a fixed foliation is not nonlinear dynamical selection",
            "orthogonal-wall preserving boundary condition within the donor-capped surface class",
            "not event-horizon stability, topology change, connected bulk geometry, dark matter, or mass transfer",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": status,
        "classification": payload["classification"],
        "classifications": classifications,
        "fine_principal_eigenvalues": {
            label: {
                time_label: {
                    branch: record["stability"]["fine_principal_eigenvalue"]
                    for branch, record in time_record.items()
                } for time_label, time_record in grid.items()
            } for label, grid in records.items()
        },
        "grid_transfer": grid_transfer,
        "acceptance": acceptance,
        "runtime_seconds": payload["runtime_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
