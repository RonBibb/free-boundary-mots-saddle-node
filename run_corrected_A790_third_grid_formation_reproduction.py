#!/usr/bin/env python3
"""Sealed third-grid reproduction of localized A=7.90 cap-pair formation."""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import run_corrected_fold_live_nonlinear_gauge_source as live
from run_corrected_A790_formation_time_refinement import (
    median_signatures,
    public_diagnostics,
    transition_index,
)
from run_corrected_A790_t004_discrepancy_formation_confirmation import (
    admitted_rule_summary,
    scan_grid,
)
from run_corrected_A790_two_grid_formation_search import (
    blind_scan,
    endpoint_transfer,
    evolution_pass,
    static_search,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import interpolate_fields


OUTPUT = Path("results/corrected_A790_third_grid_formation_reproduction.json")
CHECKPOINT = Path(
    "results/corrected_A790_third_grid_formation_reproduction_state.npz"
)
BASELINE_RESULT = Path("results/corrected_A790_formation_time_refinement.json")
BASELINE_STATE = Path("results/corrected_A790_formation_time_refinement_state.npz")
PROTOCOL = "notes/79_A790_third_grid_formation_reproduction_protocol.md"
AMPLITUDE = 7.90
FINAL_TIME = 0.001
STEPS = 8
TIMES = np.arange(1, STEPS + 1, dtype=float) * (FINAL_TIME / STEPS)
FIXED_MODES = (40, 48)
GRID_INTERVALS = (80.0, 96.0, 112.0)
RADIAL_CUT = 6.0


def generalized_order(difference_78, difference_89):
    """Infer p from unequal h ratios, returning None if no positive root exists."""
    d78 = float(difference_78)
    d89 = float(difference_89)
    if not (d78 > 0.0 and d89 > 0.0):
        return None
    ratio = d78 / d89
    h7, h8, h9 = (1.0 / value for value in GRID_INTERVALS)

    def residual(order):
        return (
            (h7**order - h8**order) / (h8**order - h9**order)
            - ratio
        )

    lower = 1e-6
    upper = 16.0
    if residual(lower) * residual(upper) > 0.0:
        return None
    return float(brentq(residual, lower, upper))


def common_grid_field_diagnostics(baseline, g9_run, g9_case, key):
    z7 = np.asarray(baseline["G7_z"])
    r7 = np.asarray(baseline["G7_r"])
    mask = r7 <= RADIAL_CUT + 1e-12
    target_r = r7[mask]
    g7 = np.asarray(baseline[f"G7_8step_{key}"])[:, mask]
    g8 = interpolate_fields(
        baseline[f"G8_8step_{key}"], baseline["G8_z"], baseline["G8_r"],
        z7, target_r,
    )
    g9 = interpolate_fields(
        g9_run[f"_{key}"], g9_case["z"], g9_case["r"], z7, target_r,
    )
    d78 = float(np.linalg.norm(g7 - g8))
    d89 = float(np.linalg.norm(g8 - g9))
    scale78 = max(float(np.linalg.norm(g7)), float(np.linalg.norm(g8)), 1e-300)
    scale89 = max(float(np.linalg.norm(g8)), float(np.linalg.norm(g9)), 1e-300)
    return {
        "G7_G8_absolute_difference": d78,
        "G8_G9_absolute_difference": d89,
        "G7_G8_relative_difference": d78 / scale78,
        "G8_G9_relative_difference": d89 / scale89,
        "difference_ratio_G7G8_over_G8G9": d78 / max(d89, 1e-300),
        "generalized_empirical_order": generalized_order(d78, d89),
        "difference_decreases": bool(d89 < d78),
    }


def surface_grid_diagnostics(signatures):
    vectors = {
        label: np.asarray(values, dtype=float).ravel()
        for label, values in signatures.items()
    }
    if any(vector.size != 4 for vector in vectors.values()):
        return {
            "available": False,
            "G7_G8_absolute_difference": None,
            "G8_G9_absolute_difference": None,
            "generalized_empirical_order": None,
            "difference_decreases": False,
        }
    d78 = float(np.linalg.norm(vectors["G7"] - vectors["G8"]))
    d89 = float(np.linalg.norm(vectors["G8"] - vectors["G9"]))
    scale78 = max(
        float(np.linalg.norm(vectors["G7"])),
        float(np.linalg.norm(vectors["G8"])), 1e-300,
    )
    scale89 = max(
        float(np.linalg.norm(vectors["G8"])),
        float(np.linalg.norm(vectors["G9"])), 1e-300,
    )
    return {
        "available": True,
        "G7_G8_absolute_difference": d78,
        "G8_G9_absolute_difference": d89,
        "G7_G8_relative_difference": d78 / scale78,
        "G8_G9_relative_difference": d89 / scale89,
        "difference_ratio_G7G8_over_G8G9": d78 / max(d89, 1e-300),
        "generalized_empirical_order": generalized_order(d78, d89),
        "difference_decreases": bool(d89 < d78),
    }


def main():
    started = time.perf_counter()
    if not BASELINE_RESULT.exists() or not BASELINE_STATE.exists():
        raise FileNotFoundError("sealed note-77 result and state archives are required")
    baseline_result = json.loads(BASELINE_RESULT.read_text())
    if baseline_result.get("status") != "pass":
        raise RuntimeError("sealed note-77 baseline must have PASS status")
    baseline = np.load(BASELINE_STATE)

    print("reconstructing corrected G7/G8 and new G9 A=7.90 slice", flush=True)
    stage_started = time.perf_counter()
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": AMPLITUDE}
    g7 = build_refined(
        seed, 81, 121, "G7A790-third-grid-parent",
        selector_iterations=40, slice_iterations=270,
    )
    g8 = build_refined(
        g7, 97, 145, "G8A790-third-grid-parent",
        selector_iterations=45, slice_iterations=280,
    )
    g9 = build_refined(
        g8, 113, 169, "G9A790-third-grid",
        selector_iterations=50, slice_iterations=300,
    )
    build_seconds = time.perf_counter() - stage_started

    print("applying fresh static blind search on G9", flush=True)
    stage_started = time.perf_counter()
    initial_static = static_search(g9)
    static_seconds = time.perf_counter() - stage_started

    case = live.setup_case(
        g9, "G9-A790-third-grid", live_normal_wall_gauge=True,
        live_outer_sommerfeld=True,
    )
    live.FINAL_TIME = FINAL_TIME
    live.STEPS = STEPS
    print("evolving G9 A=7.90 to t=0.001 with eight steps", flush=True)
    stage_started = time.perf_counter()
    run = live.integrate(case, checkpoint_steps=range(1, STEPS))
    evolution_seconds = time.perf_counter() - stage_started

    stage_started = time.perf_counter()
    trajectory = []
    trigger_index = None
    for index, checkpoint_time in enumerate(TIMES):
        state = run["_checkpoints"][index + 1] if index < STEPS - 1 else run
        screen = blind_scan(
            f"G9-A790_t{checkpoint_time:.6f}-screen",
            state["_position"], state["_velocity"], g9,
            modes_to_run=FIXED_MODES,
        )
        if trigger_index is None and screen["admitted_distinct_count"] > 0:
            trigger_index = index
        discrepancy = None
        if trigger_index is not None:
            discrepancy = scan_grid(
                f"G9-A790_t{checkpoint_time:.6f}-discrepancy",
                state["_position"], state["_velocity"], g9,
            )
        selected = discrepancy if discrepancy is not None else screen
        trajectory.append({
            "time": float(checkpoint_time),
            "screen": screen,
            "discrepancy": discrepancy,
            "selected_detector": (
                "discrepancy" if discrepancy is not None
                else "fixed_40_48_screen"
            ),
            "admitted_distinct_count": selected["admitted_distinct_count"],
            "admitted_signatures": (
                median_signatures(selected)
                if discrepancy is not None else selected["admitted_signatures"]
            ),
        })
    detector_seconds = time.perf_counter() - stage_started

    counts = [record["admitted_distinct_count"] for record in trajectory]
    first_index = transition_index(counts)
    final_record = trajectory[-1]["discrepancy"]
    final_signatures = (
        median_signatures(final_record) if final_record is not None else []
    )
    baseline_signatures = baseline_result["fine_median_signatures"]
    signatures = {
        "G7": baseline_signatures["G7"],
        "G8": baseline_signatures["G8"],
        "G9": final_signatures,
    }

    field_diagnostics = {
        name: common_grid_field_diagnostics(baseline, run, case, key)
        for name, key in (
            ("position_increment", "increment"),
            ("velocity", "velocity"),
            ("source_increment", "source_increment"),
        )
    }
    surface_diagnostics = surface_grid_diagnostics(signatures)
    g8_g9_endpoint_transfer = endpoint_transfer(
        signatures["G8"], signatures["G9"],
    )
    g8_g9_field_transfer = {
        name: diagnostic["G8_G9_relative_difference"]
        for name, diagnostic in field_diagnostics.items()
    }

    post_trigger = [
        record["discrepancy"] for record in trajectory
        if record["discrepancy"] is not None
    ]
    detector_summaries = [
        admitted_rule_summary(record) for record in post_trigger
    ]
    pretrigger_zero = bool(all(
        record["screen"]["admitted_distinct_count"] == 0
        for index, record in enumerate(trajectory)
        if trigger_index is None or index < trigger_index
    ))
    detector_rules_pass = bool(
        pretrigger_zero and post_trigger
        and all(
            record["admitted_distinct_count"] == 2
            and len(record["clusters"]) == 2
            and all(len(cluster["seeds"]) >= 2 for cluster in record["clusters"])
            and summary["all_confirmations_are_exactly_next_candidate"]
            and summary["maximum_condition_number"] < 1e5
            and summary["maximum_independent_expansion"] < 0.002
            and summary["maximum_mode_or_endpoint_difference"] < 0.002
            for record, summary in zip(post_trigger, detector_summaries)
        )
    )
    acceptance = {
        "initial_zero_caps_and_evolution_passes": bool(
            initial_static["accepted_count"] == 0 and evolution_pass(run)
        ),
        "G9_history_matches_localized_two_grid_transition": bool(
            counts == [0, 0, 0, 0, 2, 2, 2, 2]
            and first_index == 4
        ),
        "all_screen_and_post_trigger_detector_rules_pass": detector_rules_pass,
        "G8_G9_field_and_endpoint_transfer_below_limits": bool(
            max(g8_g9_field_transfer.values(), default=math.inf) < 0.05
            and g8_g9_endpoint_transfer is not None
            and g8_g9_endpoint_transfer["maximum"] < 0.01
        ),
        "all_three_grid_differences_decrease": bool(
            all(item["difference_decreases"] for item in field_diagnostics.values())
            and surface_diagnostics["difference_decreases"]
        ),
    }
    status = "pass" if all(acceptance.values()) else "review"
    runtime = {
        "geometry_build_seconds": build_seconds,
        "static_search_seconds": static_seconds,
        "evolution_seconds": evolution_seconds,
        "detector_seconds": detector_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    payload = {
        "status": status,
        "classification": (
            "three_grid_localized_paired_formation_candidate"
            if status == "pass" else "unresolved_third_grid_reproduction"
        ),
        "scope": "sealed A=7.90 third-grid reproduction of localized paired marginal-surface formation",
        "protocol": PROTOCOL,
        "baseline_result": str(BASELINE_RESULT),
        "baseline_state": str(BASELINE_STATE),
        "amplitude": AMPLITUDE,
        "grid": {"label": "G9", "size": [113, 169], "r_max": 8.0},
        "time_step": FINAL_TIME / STEPS,
        "times": TIMES.tolist(),
        "initial_static_search": initial_static,
        "evolution_diagnostics": public_diagnostics(run),
        "trajectory": trajectory,
        "count_history": counts,
        "first_detection_index": first_index,
        "formation_bracket": (
            {"lower": float(TIMES[first_index - 1]),
             "upper": float(TIMES[first_index]),
             "width": float(TIMES[first_index] - TIMES[first_index - 1])}
            if first_index is not None else None
        ),
        "final_median_signatures": signatures,
        "G8_G9_endpoint_transfer": g8_g9_endpoint_transfer,
        "G8_G9_field_transfer": g8_g9_field_transfer,
        "three_grid_field_diagnostics": field_diagnostics,
        "three_grid_surface_diagnostics": surface_diagnostics,
        "post_trigger_detector_summaries": detector_summaries,
        "acceptance": acceptance,
        "runtime": runtime,
        "limitations": [
            "formation bracket is detector- and foliation-dependent",
            "finite twelve-seed star-shaped donor-capped surface class",
            "short evolution through t=0.001 at one time step on G9",
            "same evolution and surface algorithms as the two-grid baseline",
            "not event-horizon location, continuum topology change, amplitude-basin evidence, long-time stability, connected bulk geometry, or mass transfer",
        ],
    }

    state_values = {
        "G9_z": g9["z"],
        "G9_r": g9["r"],
        "times": TIMES,
        "G9_8step_increment": run["_increment"],
        "G9_8step_velocity": run["_velocity"],
        "G9_8step_source_increment": run["_source_increment"],
    }
    for index in range(STEPS - 1):
        state = run["_checkpoints"][index + 1]
        state_values[f"G9_time_{index}_increment"] = (
            state["_position"] - case["initial"]
        )
        state_values[f"G9_time_{index}_velocity"] = state["_velocity"]
    np.savez_compressed(CHECKPOINT, **state_values)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": status,
        "classification": payload["classification"],
        "count_history": counts,
        "formation_bracket": payload["formation_bracket"],
        "G8_G9_field_transfer": g8_g9_field_transfer,
        "G8_G9_endpoint_transfer": g8_g9_endpoint_transfer,
        "three_grid_field_orders": {
            name: item["generalized_empirical_order"]
            for name, item in field_diagnostics.items()
        },
        "surface_order": surface_diagnostics["generalized_empirical_order"],
        "acceptance": acceptance,
        "runtime": runtime,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
