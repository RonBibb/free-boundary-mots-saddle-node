#!/usr/bin/env python3
"""Sealed time localization and 2/4/8-step refinement of A=7.90 cap formation."""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import run_corrected_fold_live_nonlinear_gauge_source as live
from run_corrected_A790_t004_discrepancy_formation_confirmation import (
    admitted_rule_summary, scan_grid,
)
from run_corrected_A790_two_grid_formation_search import (
    blind_scan, endpoint_transfer, evolution_pass, field_transfer, static_search,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import relative_norm


OUTPUT = Path("results/corrected_A790_formation_time_refinement.json")
CHECKPOINT = Path("results/corrected_A790_formation_time_refinement_state.npz")
PROTOCOL = "notes/77_A790_formation_time_refinement_protocol.md"
AMPLITUDE = 7.90
FINAL_TIME = .001
STEP_COUNTS = (2, 4, 8)
FINE_TIMES = np.arange(1, 9, dtype=float) * (FINAL_TIME / 8)
FIXED_MODES = (40, 48)


def integrate_resolution(case, steps, checkpoints=()):
    live.FINAL_TIME = FINAL_TIME
    live.STEPS = int(steps)
    original_label = case["label"]
    case["label"] = f"{original_label}-{steps}step"
    try:
        return live.integrate(case, checkpoint_steps=checkpoints)
    finally:
        case["label"] = original_label


def median_signatures(record):
    if not record.get("clusters"):
        return []
    anchors = [np.asarray(cluster["signature"]) for cluster in record["clusters"]]
    members = [[] for _ in anchors]
    for trial in record["trials"]:
        if not trial.get("admitted"):
            continue
        pair = trial["adjacent_mode_pair"]
        signature = np.array((
            pair["representative_rho_axis"], pair["representative_rho_brane"],
        ))
        index = int(np.argmin([
            np.linalg.norm(signature - anchor) for anchor in anchors
        ]))
        members[index].append(signature)
    result = [
        np.median(np.stack(values), axis=0).tolist()
        for values in members if values
    ]
    return sorted(result, key=lambda value: value[1])


def same_grid_field_convergence(geometry, runs, key):
    mask = geometry["r"] <= 6. + 1e-12
    coarse, medium, fine = (run[key][:, mask] for run in runs)
    coarse_medium = relative_norm(coarse, medium)
    medium_fine = relative_norm(medium, fine)
    rate = float(
        math.log(coarse_medium / medium_fine, 2)
        if coarse_medium > 0 and medium_fine > 0 else np.inf
    )
    return {
        "coarse_medium_relative_difference": coarse_medium,
        "medium_fine_relative_difference": medium_fine,
        "convergence_rate": rate,
    }


def signature_convergence(records):
    vectors = [np.asarray(median_signatures(record)).ravel() for record in records]
    if any(vector.size != 4 for vector in vectors):
        return {
            "available": False,
            "coarse_medium_relative_difference": None,
            "medium_fine_relative_difference": None,
            "convergence_rate": None,
            "median_signatures": [vector.tolist() for vector in vectors],
        }
    coarse_medium = relative_norm(vectors[0], vectors[1])
    medium_fine = relative_norm(vectors[1], vectors[2])
    rate = float(
        math.log(coarse_medium / medium_fine, 2)
        if coarse_medium > 0 and medium_fine > 0 else np.inf
    )
    return {
        "available": True,
        "coarse_medium_relative_difference": coarse_medium,
        "medium_fine_relative_difference": medium_fine,
        "convergence_rate": rate,
        "median_signatures": [vector.reshape(2, 2).tolist() for vector in vectors],
    }


def transition_index(counts):
    positive = [index for index, count in enumerate(counts) if count > 0]
    if not positive:
        return None
    first = positive[0]
    if first == 0:
        return None
    if not all(count == 0 for count in counts[:first]):
        return None
    if not all(count == 2 for count in counts[first:]):
        return None
    return first


def public_diagnostics(run):
    return {
        "global_GH_constraint": run["final_constraint"]["global_relative"],
        "wall_position_residual": run["final_wall"]["maximum"],
        "normal_wall_position_residual": run[
            "final_normal_wall_position_residual"
        ]["maximum"],
        "maximum_outer_metric_correction": run["maximum_outer_metric_correction"],
        "maximum_outer_scalar_correction": run["maximum_outer_scalar_correction"],
        "maximum_outer_source_correction": run["maximum_outer_source_correction"],
        "finite": run["all_stages_finite"],
        "Lorentzian": run["signature"]["all_points_one_negative_direction"],
    }


def main():
    print("building fresh corrected G7/G8 A=7.90 refinement geometries", flush=True)
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": AMPLITUDE}
    geometries = {
        "G7": build_refined(
            seed, 81, 121, "G7A790-time-refine", selector_iterations=40,
            slice_iterations=270,
        ),
    }
    geometries["G8"] = build_refined(
        geometries["G7"], 97, 145, "G8A790-time-refine",
        selector_iterations=45, slice_iterations=280,
    )
    initial_static = {
        label: static_search(geometry) for label, geometry in geometries.items()
    }
    cases = {
        label: live.setup_case(
            geometry, f"{label}-A790-time-refine",
            live_normal_wall_gauge=True, live_outer_sommerfeld=True,
        )
        for label, geometry in geometries.items()
    }
    runs = {label: {} for label in geometries}
    for label, case in cases.items():
        for steps in STEP_COUNTS:
            print(f"evolving {label} A=7.90 to t=0.001 with {steps} steps", flush=True)
            checkpoints = range(1, steps) if steps == 8 else ()
            runs[label][steps] = integrate_resolution(case, steps, checkpoints)

    final_scans = {label: {} for label in geometries}
    trajectory = {label: [] for label in geometries}
    trigger_index = None
    for index, time in enumerate(FINE_TIMES):
        screens = {}
        states = {}
        for label in geometries:
            fine = runs[label][8]
            state = fine["_checkpoints"][index + 1] if index < 7 else fine
            states[label] = state
            screens[label] = blind_scan(
                f"{label}-A790_t{time:.6f}-screen", state["_position"],
                state["_velocity"], geometries[label],
                modes_to_run=FIXED_MODES,
            )
        if trigger_index is None and any(
            screen["admitted_distinct_count"] > 0 for screen in screens.values()
        ):
            trigger_index = index
        resolved = {}
        if trigger_index is not None:
            for label in geometries:
                resolved[label] = scan_grid(
                    f"{label}-A790_t{time:.6f}-discrepancy",
                    states[label]["_position"], states[label]["_velocity"],
                    geometries[label],
                )
        for label in geometries:
            selected = resolved.get(label, screens[label])
            trajectory[label].append({
                "time": float(time),
                "screen": screens[label],
                "discrepancy": resolved.get(label),
                "selected_detector": (
                    "discrepancy" if label in resolved else "fixed_40_48_screen"
                ),
                "admitted_distinct_count": selected["admitted_distinct_count"],
                "admitted_signatures": (
                    median_signatures(selected)
                    if label in resolved else selected["admitted_signatures"]
                ),
            })
        if index == 7 and resolved:
            for label in geometries:
                final_scans[label][8] = resolved[label]

    for label in geometries:
        for steps in (2, 4):
            run = runs[label][steps]
            final_scans[label][steps] = scan_grid(
                f"{label}-A790_t0.001-{steps}step-discrepancy",
                run["_position"], run["_velocity"], geometries[label],
            )
        if 8 not in final_scans[label]:
            run = runs[label][8]
            final_scans[label][8] = scan_grid(
                f"{label}-A790_t0.001-8step-discrepancy",
                run["_position"], run["_velocity"], geometries[label],
            )

    field_convergence = {}
    for label, geometry in geometries.items():
        ordered = [runs[label][steps] for steps in STEP_COUNTS]
        field_convergence[label] = {
            name: same_grid_field_convergence(geometry, ordered, key)
            for name, key in (
                ("position_increment", "_increment"),
                ("velocity", "_velocity"),
                ("source_increment", "_source_increment"),
            )
        }
    fine_field_transfer = {
        name: field_transfer(
            cases["G7"], runs["G7"][8], cases["G8"], runs["G8"][8], key,
        )
        for name, key in (
            ("position_increment", "_increment"),
            ("velocity", "_velocity"),
            ("source_increment", "_source_increment"),
        )
    }
    surface_convergence = {
        label: signature_convergence([
            final_scans[label][steps] for steps in STEP_COUNTS
        ])
        for label in geometries
    }
    fine_signatures = {
        label: median_signatures(final_scans[label][8]) for label in geometries
    }
    fine_endpoint_transfer = endpoint_transfer(
        fine_signatures["G7"], fine_signatures["G8"],
    )
    count_histories = {
        label: [item["admitted_distinct_count"] for item in records]
        for label, records in trajectory.items()
    }
    first_indices = {
        label: transition_index(counts) for label, counts in count_histories.items()
    }
    if all(value is not None for value in first_indices.values()):
        earliest = min(first_indices.values())
        latest = max(first_indices.values())
        bracket_lower = 0. if earliest == 0 else float(FINE_TIMES[earliest - 1])
        bracket_upper = float(FINE_TIMES[latest])
        formation_bracket = {
            "lower": bracket_lower,
            "upper": bracket_upper,
            "width": bracket_upper - bracket_lower,
            "first_detection_indices": first_indices,
            "first_detection_times": {
                label: float(FINE_TIMES[index])
                for label, index in first_indices.items()
            },
        }
    else:
        formation_bracket = None

    detected_summaries = []
    pretrigger_zero = True
    for label, records in trajectory.items():
        for index, record in enumerate(records):
            if trigger_index is None or index < trigger_index:
                pretrigger_zero = bool(
                    pretrigger_zero and record["screen"]["admitted_distinct_count"] == 0
                )
            elif record["discrepancy"] is not None:
                detected_summaries.append((
                    record["discrepancy"],
                    admitted_rule_summary(record["discrepancy"]),
                ))
    all_detector_rules = bool(
        pretrigger_zero and detected_summaries
        and all(
            record["admitted_distinct_count"] == 2
            and len(record["clusters"]) == 2
            and all(len(cluster["seeds"]) >= 2 for cluster in record["clusters"])
            and summary["all_confirmations_are_exactly_next_candidate"]
            and summary["maximum_condition_number"] < 1e5
            and summary["maximum_independent_expansion"] < .002
            and summary["maximum_mode_or_endpoint_difference"] < .002
            for record, summary in detected_summaries
        )
    )
    field_rate_pass = all(
        item["convergence_rate"] > 1.5
        for grid in field_convergence.values() for item in grid.values()
    )
    surface_convergence_pass = all(
        item["available"]
        and item["medium_fine_relative_difference"] < .002
        and (
            item["convergence_rate"] > 1.3
            or max(
                item["coarse_medium_relative_difference"],
                item["medium_fine_relative_difference"],
            ) < .0005
        )
        for item in surface_convergence.values()
    )
    localization_pass = bool(
        formation_bracket is not None
        and abs(first_indices["G7"] - first_indices["G8"]) <= 1
        and formation_bracket["width"] <= .00025 + 1e-15
    )
    acceptance = {
        "initial_zero_caps_and_all_six_evolutions_pass": bool(
            all(item["accepted_count"] == 0 for item in initial_static.values())
            and all(evolution_pass(run) for grid in runs.values() for run in grid.values())
        ),
        "fields_converge_above_1_5_and_fine_grids_transfer": bool(
            field_rate_pass and max(fine_field_transfer.values()) < .05
        ),
        "all_final_slices_have_two_time_convergent_candidates": bool(
            all(
                final_scans[label][steps]["admitted_distinct_count"] == 2
                for label in geometries for steps in STEP_COUNTS
            )
            and surface_convergence_pass
            and fine_endpoint_transfer is not None
            and fine_endpoint_transfer["maximum"] < .01
        ),
        "fine_histories_localize_one_persistent_pair_transition": localization_pass,
        "all_screen_and_post_trigger_detector_rules_pass": all_detector_rules,
    }
    result_status = "pass" if all(acceptance.values()) else "review"
    payload = {
        "status": result_status,
        "classification": (
            "localized_time_refined_paired_formation_candidate"
            if result_status == "pass" else "unresolved_formation_time_refinement"
        ),
        "scope": "sealed A=7.90 two-grid formation-time localization and 2/4/8-step refinement",
        "protocol": PROTOCOL,
        "amplitude": AMPLITUDE,
        "final_time": FINAL_TIME,
        "step_counts": list(STEP_COUNTS),
        "fine_times": FINE_TIMES.tolist(),
        "trigger_index": trigger_index,
        "initial_static_search": initial_static,
        "evolution_diagnostics": {
            label: {
                str(steps): public_diagnostics(run)
                for steps, run in grid.items()
            } for label, grid in runs.items()
        },
        "field_convergence": field_convergence,
        "fine_field_transfer": fine_field_transfer,
        "final_dynamic_search": {
            label: {str(steps): final_scans[label][steps] for steps in STEP_COUNTS}
            for label in geometries
        },
        "surface_convergence": surface_convergence,
        "fine_median_signatures": fine_signatures,
        "fine_endpoint_transfer": fine_endpoint_transfer,
        "trajectory": trajectory,
        "count_histories": count_histories,
        "formation_bracket": formation_bracket,
        "acceptance": acceptance,
        "limitations": [
            "formation bracket is detector- and foliation-dependent",
            "finite twelve-seed star-shaped donor-capped class",
            "short evolution through t=0.001",
            "not event-horizon location, continuum topology change, amplitude-basin evidence, long-time stability, connected bulk geometry, or mass transfer",
        ],
    }
    state_values = {
        "G7_z": geometries["G7"]["z"], "G7_r": geometries["G7"]["r"],
        "G8_z": geometries["G8"]["z"], "G8_r": geometries["G8"]["r"],
        "fine_times": FINE_TIMES,
    }
    for label in geometries:
        for steps in STEP_COUNTS:
            run = runs[label][steps]
            for name, key in (
                ("increment", "_increment"), ("velocity", "_velocity"),
                ("source_increment", "_source_increment"),
            ):
                state_values[f"{label}_{steps}step_{name}"] = run[key]
        fine = runs[label][8]
        for index in range(1, 8):
            checkpoint = fine["_checkpoints"][index]
            state_values[f"{label}_fine_time_{index - 1}_increment"] = (
                checkpoint["_position"] - cases[label]["initial"]
            )
            state_values[f"{label}_fine_time_{index - 1}_velocity"] = checkpoint[
                "_velocity"
            ]
    np.savez_compressed(CHECKPOINT, **state_values)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result_status,
        "classification": payload["classification"],
        "count_histories": count_histories,
        "formation_bracket": formation_bracket,
        "field_rates": {
            label: {name: item["convergence_rate"] for name, item in grid.items()}
            for label, grid in field_convergence.items()
        },
        "surface_convergence": surface_convergence,
        "fine_endpoint_transfer": fine_endpoint_transfer,
        "acceptance": acceptance,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
