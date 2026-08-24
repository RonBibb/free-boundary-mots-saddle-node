#!/usr/bin/env python3
"""Sealed fine-grid midpoint test of amplitude dependence in cap motion."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import run_corrected_fold_live_nonlinear_gauge_source as live
from run_corrected_fold_discrepancy_horizon_tracker import (
    public_record, select_surface, static_cap,
)
from run_corrected_fold_G7_cap_motion_history import history_metrics
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import relative_norm


OUTPUT = Path("results/corrected_fold_A797_midpoint_motion.json")
CHECKPOINT = Path("results/corrected_fold_A797_midpoint_motion_state.npz")
LOW_JSON = Path("results/corrected_fold_G8_cap_motion_replication.json")
LOW_STATE = Path("results/corrected_fold_G8_cap_motion_replication_state.npz")
HIGH_JSON = Path("results/corrected_fold_A8_cap_motion_robustness.json")
HIGH_STATE = Path("results/corrected_fold_A8_cap_motion_robustness_state.npz")
AMPLITUDE = 7.97
FINAL_TIME = 0.004
STEPS = 8
CHECKPOINT_STEPS = (2, 4, 6)
TIMES = np.array((0.001, 0.002, 0.003, 0.004))


def relative_scalar(left, right):
    return float(abs(left - right) / max(abs(left), abs(right), 1e-300))


def tracker_pass(records):
    return bool(
        all(
            record["first_index"] > 0
            and record["selected"]["converged"]
            and record["confirmation"]["converged"]
            and record["selected"]["jacobian_condition_number"] < 1e5
            and record["confirmation"]["jacobian_condition_number"] < 1e5
            for record in records
        )
        and max(
            max(
                record["comparison"]["profile_relative_difference"],
                record["comparison"]["axis_relative_difference"],
                record["comparison"]["brane_relative_difference"],
            ) for record in records
        ) < .0005
        and max(
            record["comparison"]["displacement_relative_difference"]
            for record in records
        ) < .01
    )


def main():
    low_json = json.loads(LOW_JSON.read_text())
    high_json = json.loads(HIGH_JSON.read_text())
    low_state = np.load(LOW_STATE)
    high_state = np.load(HIGH_STATE)
    print("building fresh corrected G8 A=7.97 midpoint geometry", flush=True)
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": AMPLITUDE}
    g7 = build_refined(
        seed, 81, 121, "G7A7970-seed", selector_iterations=40,
        slice_iterations=270,
    )
    g8 = build_refined(
        g7, 97, 145, "G8A7970", selector_iterations=45,
        slice_iterations=280,
    )
    initial = static_cap(g8)
    case = live.setup_case(
        g8, "G8-A797", live_normal_wall_gauge=True,
        live_outer_sommerfeld=True,
    )
    live.FINAL_TIME = FINAL_TIME
    live.STEPS = STEPS
    print("evolving midpoint G8 trajectory with three checkpoints", flush=True)
    run = live.integrate(case, checkpoint_steps=CHECKPOINT_STEPS)
    states = [run["_checkpoints"][step] for step in CHECKPOINT_STEPS] + [run]
    position0 = g8["jet_field"].reduced_fields
    records = [
        select_surface(
            f"G8-A797_t{time:.3f}", position0 + state["_increment"],
            state["_velocity"], g8, initial,
        )
        for time, state in zip(TIMES, states)
    ]
    selected = history_metrics(records, use_confirmation=False)
    confirmation = history_metrics(records, use_confirmation=True)

    low_exponents = {
        "selected": low_json["G8_selected_history"]["displacement_power_exponent"],
        "confirmation": low_json["G8_confirmation_history"]["displacement_power_exponent"],
    }
    high_exponents = {
        "selected": high_json["histories"]["G8"]["selected"]["displacement_power_exponent"],
        "confirmation": high_json["histories"]["G8"]["confirmation"]["displacement_power_exponent"],
    }
    midpoint_exponents = {
        "selected": selected["displacement_power_exponent"],
        "confirmation": confirmation["displacement_power_exponent"],
    }
    displacement_brackets = []
    for index, record in enumerate(records):
        midpoint_norm = float(np.linalg.norm(
            record["selected"]["rho"] - record["initial_on"],
        ))
        low_norm = float(np.linalg.norm(
            low_state[f"time_{index}_selected_horizon"]
            - low_state["initial_horizon"],
        ))
        high_norm = float(np.linalg.norm(
            high_state[f"G8_time_{index}_selected_horizon"]
            - high_state["G8_initial_horizon"],
        ))
        lower = .95 * min(low_norm, high_norm)
        upper = 1.05 * max(low_norm, high_norm)
        displacement_brackets.append({
            "time": float(TIMES[index]),
            "A7_94": low_norm,
            "A7_97": midpoint_norm,
            "A8_00": high_norm,
            "allowed_lower": lower,
            "allowed_upper": upper,
            "inside": bool(lower < midpoint_norm < upper),
        })
    initial_transfer = {
        "to_A7_94": {
            "profile": relative_norm(records[0]["initial_on"], low_state["initial_horizon"]),
            "axis": relative_scalar(records[0]["initial_on"][0], low_state["initial_horizon"][0]),
            "brane": relative_scalar(records[0]["initial_on"][-1], low_state["initial_horizon"][-1]),
        },
        "to_A8_00": {
            "profile": relative_norm(records[0]["initial_on"], high_state["G8_initial_horizon"]),
            "axis": relative_scalar(records[0]["initial_on"][0], high_state["G8_initial_horizon"][0]),
            "brane": relative_scalar(records[0]["initial_on"][-1], high_state["G8_initial_horizon"][-1]),
        },
    }
    endpoint_motion = all(
        record["selected"]["rho_axis"] > record["initial_on"][0]
        and record["selected"]["rho_brane"] > record["initial_on"][-1]
        for record in records
    )
    acceptance = {
        "initial_outer_cap_converges_below_1e_6": bool(
            initial["converged"] and initial["surface_residual_max"] < 1e-6
        ),
        "evolution_constraint_wall_boundary_pass": bool(
            run["all_stages_finite"]
            and run["signature"]["all_points_one_negative_direction"]
            and run["final_constraint"]["global_relative"] < .005
            and max(
                run["final_wall"]["maximum"],
                run["final_normal_wall_position_residual"]["maximum"],
            ) < .0005
            and max(
                run["maximum_normal_wall_acceleration_residual"],
                run["maximum_outer_acceleration_residual"],
                run["maximum_outer_source_residual"],
                run["final_outer_sommerfeld_position_residual"]["maximum_normalized"],
                run["final_outer_source_sommerfeld_residual"]["maximum_normalized"],
            ) < 1e-10
            and max(
                run["maximum_outer_metric_correction"],
                run["maximum_outer_scalar_correction"],
                run["maximum_outer_source_correction"],
            ) < .05
        ),
        "all_discrepancy_trackers_and_transfers_pass": tracker_pass(records),
        "selected_and_confirmation_match_endpoint_class": bool(
            selected["classification"]
            == confirmation["classification"]
            == low_json["classification"]
            == high_json["histories"]["G8"]["selected"]["classification"]
        ),
        "midpoint_exponents_inside_endpoint_brackets": bool(
            all(
                min(low_exponents[kind], high_exponents[kind]) - .03
                < midpoint_exponents[kind]
                < max(low_exponents[kind], high_exponents[kind]) + .03
                for kind in ("selected", "confirmation")
            ) and abs(midpoint_exponents["selected"] - midpoint_exponents["confirmation"]) < .02
        ),
        "all_displacement_norms_inside_endpoint_brackets": bool(all(
            item["inside"] for item in displacement_brackets
        )),
        "initial_profiles_close_and_endpoint_motion_positive": bool(
            max(value for item in initial_transfer.values() for value in item.values()) < .05
            and endpoint_motion
        ),
    }
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "classification": (
            selected["classification"] if all(acceptance.values()) else "unresolved_midpoint"
        ),
        "scope": "sealed fine-grid A=7.97 midpoint test of the short-time capped marginal-surface response",
        "protocol": "notes/72_A797_midpoint_motion_protocol.md",
        "amplitude": AMPLITUDE,
        "times": TIMES.tolist(),
        "time_step": FINAL_TIME / STEPS,
        "tracker_records": {
            f"t_{time:.3f}": public_record(record)
            for time, record in zip(TIMES, records)
        },
        "selected_history": selected,
        "confirmation_history": confirmation,
        "endpoint_exponents": {"A7_94": low_exponents, "A8_00": high_exponents},
        "midpoint_exponents": midpoint_exponents,
        "displacement_brackets": displacement_brackets,
        "initial_amplitude_transfer": initial_transfer,
        "all_endpoint_motion_positive": endpoint_motion,
        "final_diagnostics": {
            "global_GH_constraint": run["final_constraint"]["global_relative"],
            "wall_position_residual": run["final_wall"]["maximum"],
            "normal_wall_position_residual": run["final_normal_wall_position_residual"]["maximum"],
            "maximum_outer_metric_correction": run["maximum_outer_metric_correction"],
            "maximum_outer_scalar_correction": run["maximum_outer_scalar_correction"],
            "maximum_outer_source_correction": run["maximum_outer_source_correction"],
        },
        "acceptance": acceptance,
        "limitations": [
            "fine-grid midpoint test rather than a two-grid midpoint replication",
            "three amplitudes do not establish an open basin",
            "coordinate motion of pre-existing marginal caps through t=0.004",
            "not formation, event-horizon location, topology change, long-time stability, branch selection, or mass transfer",
        ],
    }
    np.savez_compressed(
        CHECKPOINT, z=g8["z"], r=g8["r"], times=TIMES,
        initial_horizon=records[0]["initial_on"],
        **{
            f"time_{index}_{name}": value
            for index, (state, record) in enumerate(zip(states, records))
            for name, value in (
                ("increment", state["_increment"]),
                ("velocity", state["_velocity"]),
                ("source_increment", state["_source_increment"]),
                ("selected_horizon", record["selected"]["rho"]),
                ("confirmation_horizon", record["confirmation"]["rho"]),
            )
        },
    )
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
