#!/usr/bin/env python3
"""Sealed nonlinear radial-profile perturbation pilot at corrected A=7.90."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import run_corrected_fold_live_nonlinear_gauge_source as live
from bhps.corrected_profile_perturbation_builder import build_profile_refined
from bhps.dynamical_capped_geometry import capped_surface_geometry
from bhps.nonlinear_regular_so3_evolution import (
    compact_wall_normal_gauge_position_residuals,
)
from run_corrected_A790_dynamic_MOTS_stability import (
    analytic_control,
    recover_surface,
    stability_series,
    surface_passes,
)
from run_corrected_A790_independent_dynamic_BVP_detector import (
    public_surface,
    search_slice,
    transfer as endpoint_transfer,
)
from run_corrected_A790_two_grid_formation_search import (
    evolution_pass,
    field_transfer,
)
from run_corrected_fold_regular_so3_runtime import build_geometry


PROTOCOL = "notes/87_A790_profile_perturbation_pilot_protocol.md"
SIGMA_VALUES = (.99, 1.01)
FINAL_TIME = .004
STEPS = 32
SAMPLE_STEPS = (4, 5, 6, 8, 32)
SAMPLE_TIMES = tuple(step * FINAL_TIME / STEPS for step in SAMPLE_STEPS)
DIAGNOSTIC_TIMES = (.001, .004)
BRANCH_SEEDS = {"inner": 1.30, "outer": 1.55}


def sigma_tag(sigma_r):
    return f"sigmaR{int(round(100 * sigma_r)):03d}"


def result_path(sigma_r):
    return Path(f"results/corrected_A790_{sigma_tag(sigma_r)}_profile_perturbation.json")


def state_path(sigma_r):
    return Path(f"results/corrected_A790_{sigma_tag(sigma_r)}_profile_perturbation_state.npz")


def sampled_state(run, step):
    return run if int(step) == STEPS else run["_checkpoints"][int(step)]


def compatibility_pass(record):
    return bool(
        record["finite_wall_converged"]
        and record["finite_wall_maximum_residual"] < 1e-9
        and record["anisotropic_selector_converged"]
        and record["anisotropic_selector_maximum_residual"] < 1e-8
        and record["balanced_hamiltonian_maximum_residual"] < 1e-7
        and record["zeroth_order_spatial_junction_maximum_residual"] < 1e-7
        and record["second_corner_maximum_intrinsic_residual"] < .025
        and record["scalar_wall_Neumann_relative_defect"] < 1e-10
        and record["extrinsic_curvature_and_scalar_momenta_zero"]
        and record["momentum_constraint_by_time_symmetry"] == 0.
    )


def initial_gauge_diagnostics(case):
    normal = compact_wall_normal_gauge_position_residuals(
        case["initial"], case["source0"], case["z"], case["r"],
        case["geometry"]["background"],
    )
    return {
        "global_GH_constraint": case["initial_constraint"]["global_relative"],
        "wall_position_residual": case["initial_wall"]["maximum"],
        "normal_wall_position_residual": normal["maximum"],
    }


def public_evolution(run):
    return {
        "finite": run["all_stages_finite"],
        "Lorentzian": run["signature"]["all_points_one_negative_direction"],
        "global_GH_constraint": run["final_constraint"]["global_relative"],
        "wall_position_residual": run["final_wall"]["maximum"],
        "normal_wall_position_residual": run[
            "final_normal_wall_position_residual"
        ]["maximum"],
        "maximum_outer_metric_correction": run["maximum_outer_metric_correction"],
        "maximum_outer_scalar_correction": run["maximum_outer_scalar_correction"],
        "maximum_outer_source_correction": run["maximum_outer_source_correction"],
    }


def transition_index(counts):
    positive = [index for index, count in enumerate(counts) if count > 0]
    if not positive:
        return None
    first = positive[0]
    if any(count != 0 for count in counts[:first]):
        return None
    if any(count != 2 for count in counts[first:]):
        return None
    return first


def stability_matrix_controls(stability):
    spectra = stability["spectra"].values()
    fine = stability["fine_principal_eigenvalue"]
    return bool(
        all(
            item["left_neumann_defect"] < 1e-10
            and item["right_neumann_defect"] < 1e-10
            and item["minimum_normal_factor"] > 0
            and np.isfinite(item["operator_frobenius_norm"])
            and abs(item["principal_eigenvalue_imaginary"])
            < 1e-6 * max(1., abs(item["principal_eigenvalue_real"]))
            and item["principal_eigenfunction_sign_changes"] == 0
            for item in spectra
        )
        and stability["angular_difference_65_81"] < max(.02, .05 * abs(fine))
        and stability["Frechet_step_difference_81"] < max(.02, .05 * abs(fine))
        and stability["resolved"]
    )


def run_sigma(fold, sigma_r, control):
    started = time.perf_counter()
    tag = sigma_tag(sigma_r)
    print(f"building constraint-solved G7/G8 A790 {tag} slices", flush=True)
    geometries = {
        "G7": build_profile_refined(
            fold, 81, 121, f"G7-A790-{tag}", sigma_r,
            selector_iterations=40, slice_iterations=270,
        ),
    }
    geometries["G8"] = build_profile_refined(
        geometries["G7"], 97, 145, f"G8-A790-{tag}", sigma_r,
        selector_iterations=45, slice_iterations=280,
    )
    cases = {
        label: live.setup_case(
            geometry, f"{label}-A790-{tag}",
            live_normal_wall_gauge=True, live_outer_sommerfeld=True,
        )
        for label, geometry in geometries.items()
    }
    live.FINAL_TIME = FINAL_TIME
    live.STEPS = STEPS
    runs = {}
    for label, case in cases.items():
        print(f"evolving {label} A790 {tag} through t={FINAL_TIME}", flush=True)
        runs[label] = live.integrate(case, checkpoint_steps=SAMPLE_STEPS[:-1])

    searches = {label: {} for label in geometries}
    for label, geometry in geometries.items():
        initial = cases[label]["initial"]
        searches[label]["0.0"] = search_slice(
            f"{label}-A790-{tag}-t0", initial, np.zeros_like(initial), geometry,
        )
        for step, sample_time in zip(SAMPLE_STEPS, SAMPLE_TIMES):
            state = sampled_state(runs[label], step)
            searches[label][str(sample_time)] = search_slice(
                f"{label}-A790-{tag}-t{sample_time:.6f}",
                state["_position"], state["_velocity"], geometry,
            )

    diagnostics = {label: {} for label in geometries}
    for label, geometry in geometries.items():
        for time_value in DIAGNOSTIC_TIMES:
            step = int(round(time_value * STEPS / FINAL_TIME))
            state = sampled_state(runs[label], step)
            branch_records = {}
            for branch, seed in BRANCH_SEEDS.items():
                print(
                    f"{label} A790 {tag} t={time_value} {branch}: area and stability",
                    flush=True,
                )
                surface_started = time.perf_counter()
                surface = recover_surface(
                    state["_position"], state["_velocity"], geometry, seed,
                )
                surface["runtime_seconds"] = float(
                    time.perf_counter() - surface_started
                )
                area = capped_surface_geometry(
                    state["_position"], state["_velocity"], geometry["z"],
                    geometry["r"], surface,
                )
                stability = stability_series(
                    state["_position"], state["_velocity"], geometry, surface,
                )
                branch_records[branch] = {
                    "seed": seed,
                    "surface": public_surface(surface),
                    "surface_passes": surface_passes(surface),
                    "geometry": area,
                    "stability": stability,
                }
            diagnostics[label][str(time_value)] = branch_records

    count_histories = {
        label: [
            searches[label][str(time_value)]["admitted_distinct_count"]
            for time_value in SAMPLE_TIMES
        ] for label in geometries
    }
    first_indices = {
        label: transition_index(counts) for label, counts in count_histories.items()
    }
    endpoint_transfers = {}
    for time_value in DIAGNOSTIC_TIMES:
        endpoint_transfers[str(time_value)] = endpoint_transfer(
            searches["G7"][str(time_value)]["admitted_signatures"],
            searches["G8"][str(time_value)]["admitted_signatures"],
        )
    area_transfers = {}
    stability_transfers = {}
    for time_value in DIAGNOSTIC_TIMES:
        time_label = str(time_value)
        area_transfers[time_label] = {}
        stability_transfers[time_label] = {}
        for branch in BRANCH_SEEDS:
            left = diagnostics["G7"][time_label][branch]
            right = diagnostics["G8"][time_label][branch]
            left_area = left["geometry"]["one_sided_cap_area"]
            right_area = right["geometry"]["one_sided_cap_area"]
            area_transfers[time_label][branch] = float(
                abs(left_area - right_area) / max(abs(left_area), abs(right_area), 1e-300)
            )
            left_value = left["stability"]["fine_principal_eigenvalue"]
            right_value = right["stability"]["fine_principal_eigenvalue"]
            absolute = abs(left_value - right_value)
            stability_transfers[time_label][branch] = {
                "absolute": float(absolute),
                "relative": float(
                    absolute / max(abs(left_value), abs(right_value), 1e-300)
                ),
            }

    compatibility = {
        label: geometry["compatibility"] for label, geometry in geometries.items()
    }
    profiles = {
        label: geometry["profile_perturbation"] for label, geometry in geometries.items()
    }
    initial_gauge = {
        label: initial_gauge_diagnostics(case) for label, case in cases.items()
    }
    final_field_transfer = {
        name: field_transfer(cases["G7"], runs["G7"], cases["G8"], runs["G8"], key)
        for name, key in (
            ("position_increment", "_increment"),
            ("velocity", "_velocity"),
            ("source_increment", "_source_increment"),
        )
    }
    all_searches = [record for grid in searches.values() for record in grid.values()]
    all_trials = [trial for record in all_searches for trial in record["trials"]]
    all_diagnostics = [
        record for grid in diagnostics.values() for time_record in grid.values()
        for record in time_record.values()
    ]
    initial_acceptance = {
        "all_initial_data_compatibility_gates_pass": bool(all(
            compatibility_pass(record) for record in compatibility.values()
        )),
        "all_initial_gauge_and_wall_gates_pass": bool(all(
            record["global_GH_constraint"] < .005
            and record["wall_position_residual"] < .0005
            and record["normal_wall_position_residual"] < .0005
            for record in initial_gauge.values()
        )),
        "profile_perturbation_is_nonzero_and_below_3_percent": bool(all(
            0 < record["pulse_relative_L2_difference"] < .03
            and 0 < record["gradient_relative_L2_difference"] < .03
            for record in profiles.values()
        )),
    }
    evolution_acceptance = {
        "both_evolutions_pass": bool(all(evolution_pass(run) for run in runs.values())),
        "final_field_transfer_below_5_percent": bool(
            max(final_field_transfer.values()) < .05
        ),
    }
    surface_acceptance = {
        "both_initial_BVP_searches_find_zero": bool(all(
            searches[label]["0.0"]["admitted_distinct_count"] == 0
            for label in geometries
        )),
        "both_grids_have_one_persistent_zero_to_two_transition": bool(
            all(index is not None for index in first_indices.values())
            and abs(first_indices["G7"] - first_indices["G8"]) <= 1
            and all(
                searches[label][str(time_value)]["admitted_distinct_count"] == 2
                for label in geometries for time_value in DIAGNOSTIC_TIMES
            )
        ),
        "all_admitted_BVP_surfaces_pass_note80_rules": bool(all(
            trial["admitted"] for trial in all_trials if trial["admitted"]
        )),
        "each_diagnostic_branch_has_at_least_two_BVP_seed_recoveries": bool(all(
            len(cluster["members"]) >= 2
            for label in geometries for time_value in DIAGNOSTIC_TIMES
            for cluster in searches[label][str(time_value)]["clusters"]
        )),
        "diagnostic_endpoint_transfer_below_1_percent": bool(all(
            item is not None and item["maximum"] < .01
            for item in endpoint_transfers.values()
        )),
    }
    geometry_acceptance = {
        "diagnostic_areas_are_finite_positive_and_ordered": bool(all(
            time_record["inner"]["geometry"]["finite"]
            and time_record["outer"]["geometry"]["finite"]
            and time_record["inner"]["geometry"]["one_sided_cap_area"] > 0
            and time_record["outer"]["geometry"]["one_sided_cap_area"]
            > time_record["inner"]["geometry"]["one_sided_cap_area"]
            for grid in diagnostics.values() for time_record in grid.values()
        )),
        "diagnostic_area_transfer_below_1_percent": bool(all(
            value < .01 for time_record in area_transfers.values()
            for value in time_record.values()
        )),
    }
    stability_acceptance = {
        "analytic_stability_control_passes": bool(control["passed"]),
        "all_diagnostic_surfaces_and_stability_controls_pass": bool(all(
            record["surface_passes"]
            and stability_matrix_controls(record["stability"])
            for record in all_diagnostics
        )),
        "inner_negative_outer_positive_everywhere": bool(all(
            time_record["inner"]["stability"]["classification"] == "outward_unstable"
            and time_record["outer"]["stability"]["classification"] == "outward_stable"
            for grid in diagnostics.values() for time_record in grid.values()
        )),
        "stability_eigenvalues_transfer_between_grids": bool(all(
            record["relative"] < .10 or record["absolute"] < .02
            for time_record in stability_transfers.values()
            for record in time_record.values()
        )),
    }
    acceptance = {
        **initial_acceptance,
        **evolution_acceptance,
        **surface_acceptance,
        **geometry_acceptance,
        **stability_acceptance,
    }
    hard_failure = not all((*initial_acceptance.values(), *evolution_acceptance.values()))
    status = "fail" if hard_failure else "pass" if all(acceptance.values()) else "review"
    payload = {
        "status": status,
        "classification": (
            "constraint_solved_profile_perturbation_preserves_pair_area_and_stability"
            if status == "pass" else
            "constraint_solved_profile_perturbation_hard_failure"
            if status == "fail" else
            "constraint_solved_profile_perturbation_review"
        ),
        "scope": f"sealed corrected A=7.90 nonlinear profile perturbation at sigma_r={sigma_r}",
        "protocol": PROTOCOL,
        "amplitude": 7.90,
        "sigma_r": sigma_r,
        "delta_sigma_r": sigma_r - 1.,
        "final_time": FINAL_TIME,
        "steps": STEPS,
        "time_step": FINAL_TIME / STEPS,
        "sample_times": list(SAMPLE_TIMES),
        "diagnostic_times": list(DIAGNOSTIC_TIMES),
        "analytic_stability_control": control,
        "compatibility": compatibility,
        "profile_perturbation": profiles,
        "initial_gauge_diagnostics": initial_gauge,
        "evolution_diagnostics": {
            label: public_evolution(run) for label, run in runs.items()
        },
        "final_field_transfer": final_field_transfer,
        "BVP_searches": searches,
        "count_histories": count_histories,
        "first_detection_indices": first_indices,
        "diagnostics": diagnostics,
        "endpoint_transfers": endpoint_transfers,
        "area_transfers": area_transfers,
        "stability_transfers": stability_transfers,
        "acceptance": acceptance,
        "runtime_seconds": float(time.perf_counter() - started),
        "limitations": [
            "finite symmetric one-percent radial-width perturbation, not proof of an open basin",
            "fixed corrected trace-free shape coefficients with compatibility re-audited after each elliptic solve",
            "principal SO(3)-invariant linear MOTS stability is not nonlinear branch selection",
            "finite twelve-seed star-shaped donor-capped BVP search",
            "apparent horizons remain foliation dependent",
            "not event-horizon, topology-change, connected-bulk, dark-matter, or mass-transfer evidence",
        ],
    }
    state_values = {
        "sample_steps": np.asarray(SAMPLE_STEPS),
        "sample_times": np.asarray(SAMPLE_TIMES),
    }
    for label, geometry in geometries.items():
        state_values[f"{label}_z"] = geometry["z"]
        state_values[f"{label}_r"] = geometry["r"]
        for index, step in enumerate(SAMPLE_STEPS):
            state = sampled_state(runs[label], step)
            state_values[f"{label}_time_{index}_increment"] = (
                state["_position"] - cases[label]["initial"]
            )
            state_values[f"{label}_time_{index}_velocity"] = state["_velocity"]
    np.savez_compressed(state_path(sigma_r), **state_values)
    result_path(sigma_r).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "sigma_r": sigma_r,
        "status": status,
        "classification": payload["classification"],
        "compatibility": compatibility,
        "profile_perturbation": profiles,
        "count_histories": count_histories,
        "areas": {
            label: {
                time_label: {
                    branch: record["geometry"]["one_sided_cap_area"]
                    for branch, record in time_record.items()
                } for time_label, time_record in grid.items()
            } for label, grid in diagnostics.items()
        },
        "eigenvalues": {
            label: {
                time_label: {
                    branch: record["stability"]["fine_principal_eigenvalue"]
                    for branch, record in time_record.items()
                } for time_label, time_record in grid.items()
            } for label, grid in diagnostics.items()
        },
        "acceptance": acceptance,
        "runtime_seconds": payload["runtime_seconds"],
    }, indent=2), flush=True)
    return payload


def write_combined(records, control):
    symmetry = {}
    for label in ("G7", "G8"):
        symmetry[label] = {}
        for name in ("pulse_relative_L2_difference", "gradient_relative_L2_difference"):
            left = records[0]["profile_perturbation"][label][name]
            right = records[1]["profile_perturbation"][label][name]
            symmetry[label][name] = {
                "narrow": left,
                "broad": right,
                "relative_magnitude_difference": abs(left - right) / max(left, right, 1e-300),
            }
    symmetry_pass = all(
        item["relative_magnitude_difference"] < .10
        for grid in symmetry.values() for item in grid.values()
    )
    both_pass = all(record["status"] == "pass" for record in records)
    status = "pass" if both_pass and symmetry_pass else "review"
    payload = {
        "status": status,
        "classification": (
            "symmetric_profile_perturbations_preserve_pair_area_and_stability"
            if status == "pass" else "symmetric_profile_perturbation_pilot_review"
        ),
        "scope": "sealed symmetric constraint-solved A=7.90 radial-profile perturbation pilot",
        "protocol": PROTOCOL,
        "sigma_r_values": list(SIGMA_VALUES),
        "analytic_stability_control_passes": bool(control["passed"]),
        "profile_norm_symmetry": symmetry,
        "profile_norm_symmetry_passes": symmetry_pass,
        "individual_results": [{
            "sigma_r": record["sigma_r"],
            "status": record["status"],
            "classification": record["classification"],
            "count_histories": record["count_histories"],
            "runtime_seconds": record["runtime_seconds"],
            "result_file": str(result_path(record["sigma_r"])),
            "state_file": str(state_path(record["sigma_r"])),
        } for record in records],
        "acceptance": {
            "both_symmetric_perturbations_pass": both_pass,
            "profile_norm_magnitudes_agree_within_10_percent": symmetry_pass,
        },
        "limitations": [
            "finite symmetric one-percent profile perturbation is local robustness evidence, not an open-basin proof",
            "inherits individual detector, slicing, linear-stability, and claim-boundary limitations",
        ],
    }
    path = Path("results/corrected_A790_profile_perturbation_pilot.json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma-r", type=float, choices=SIGMA_VALUES)
    parser.add_argument("--combine-only", action="store_true")
    args = parser.parse_args()
    print("running unchanged analytic MOTS-stability control", flush=True)
    control = analytic_control()
    if args.combine_only:
        records = [json.loads(result_path(value).read_text()) for value in SIGMA_VALUES]
        print(json.dumps(write_combined(records, control), indent=2), flush=True)
        return
    fold = build_geometry("G6")
    values = (args.sigma_r,) if args.sigma_r is not None else SIGMA_VALUES
    records = [run_sigma(fold, sigma_r, control) for sigma_r in values]
    if args.sigma_r is None:
        print(json.dumps(write_combined(records, control), indent=2), flush=True)


if __name__ == "__main__":
    main()
