#!/usr/bin/env python3
"""Sealed G8 replication of the four-checkpoint capped-surface history."""

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
from run_corrected_fold_short_nonlinear_evolution import (
    RADIAL_COMPARISON_CUT, interpolate_fields, relative_norm,
)


OUTPUT = Path("results/corrected_fold_G8_cap_motion_replication.json")
CHECKPOINT = Path("results/corrected_fold_G8_cap_motion_replication_state.npz")
SHORT = Path("results/corrected_fold_G7_G8_spectral_horizon_refinement_state.npz")
G7_HISTORY_JSON = Path("results/corrected_fold_G7_cap_motion_history.json")
G7_HISTORY_STATE = Path("results/corrected_fold_G7_cap_motion_history_state.npz")
FINAL_TIME = 0.004
STEPS = 8
CHECKPOINT_STEPS = (2, 4, 6)
TIMES = np.array((0.001, 0.002, 0.003, 0.004))


def relative_scalar(left, right):
    return float(abs(left - right) / max(abs(left), abs(right), 1e-300))


def main():
    short = np.load(SHORT)
    g7_state = np.load(G7_HISTORY_STATE)
    g7_payload = json.loads(G7_HISTORY_JSON.read_text())
    print("building corrected G7/G8 A=7.94 replication geometries", flush=True)
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": 7.94}
    g7 = build_refined(
        seed, 81, 121, "G7A794-G8-history", selector_iterations=40,
        slice_iterations=270,
    )
    g8 = build_refined(
        g7, 97, 145, "G8A794-history", selector_iterations=45,
        slice_iterations=280,
    )
    initial = static_cap(g8)
    case = live.setup_case(
        g8, "G8-history", live_normal_wall_gauge=True,
        live_outer_sommerfeld=True,
    )
    live.FINAL_TIME = FINAL_TIME
    live.STEPS = STEPS
    print("evolving fresh G8 trajectory with three internal checkpoints", flush=True)
    run = live.integrate(case, checkpoint_steps=CHECKPOINT_STEPS)
    states = [run["_checkpoints"][step] for step in CHECKPOINT_STEPS] + [run]

    reproduction = {
        "position_increment": relative_norm(
            states[1]["_increment"], short["G8_increment"],
        ),
        "velocity": relative_norm(states[1]["_velocity"], short["G8_velocity"]),
    }
    position0 = g8["jet_field"].reduced_fields
    records = []
    for time, state in zip(TIMES, states):
        records.append(select_surface(
            f"G8_t{time:.3f}", position0 + state["_increment"],
            state["_velocity"], g8, initial,
        ))
    selected_history = history_metrics(records, use_confirmation=False)
    confirmation_history = history_metrics(records, use_confirmation=True)

    g7_initial = g7_state["initial_horizon"]
    cross_history = {}
    for index, (time, record) in enumerate(zip(TIMES, records)):
        g7_horizon = g7_state[f"time_{index}_selected_horizon"]
        g8_horizon = record["selected"]["rho"]
        cross_history[f"t_{time:.3f}"] = {
            "profile_relative_difference": relative_norm(g7_horizon, g8_horizon),
            "axis_relative_difference": relative_scalar(g7_horizon[0], g8_horizon[0]),
            "brane_relative_difference": relative_scalar(g7_horizon[-1], g8_horizon[-1]),
            "displacement_relative_difference": relative_norm(
                g7_horizon - g7_initial,
                g8_horizon - record["initial_on"],
            ),
        }

    mask = g7["r"] <= RADIAL_COMPARISON_CUT + 1e-12
    target_r = g7["r"][mask]
    g8_increment_on_g7 = interpolate_fields(
        run["_increment"], g8["z"], g8["r"], g7["z"], target_r,
    )
    g8_velocity_on_g7 = interpolate_fields(
        run["_velocity"], g8["z"], g8["r"], g7["z"], target_r,
    )
    final_spacetime_transfer = {
        "position_increment": relative_norm(
            g7_state["time_3_increment"][:, mask], g8_increment_on_g7,
        ),
        "velocity": relative_norm(
            g7_state["time_3_velocity"][:, mask], g8_velocity_on_g7,
        ),
    }
    tracker_pass = bool(
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
    acceptance = {
        "G8_t0_002_reproduces_note_63_below_1e_8": bool(
            max(reproduction.values()) < 1e-8
        ),
        "G8_evolution_constraint_wall_boundary_pass": bool(
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
        "all_G8_discrepancy_trackers_and_transfers_pass": tracker_pass,
        "G8_selected_confirmation_and_G7_classes_agree": bool(
            selected_history["classification"]
            == confirmation_history["classification"]
            == g7_payload["classification"]
        ),
        "G8_exponents_within_0_1_of_G7": bool(
            abs(
                selected_history["displacement_power_exponent"]
                - g7_payload["selected_history"]["displacement_power_exponent"]
            ) < .1
            and abs(
                confirmation_history["displacement_power_exponent"]
                - g7_payload["confirmation_history"]["displacement_power_exponent"]
            ) < .1
        ),
        "all_time_G7_G8_horizon_transfers_pass": bool(
            max(
                max(
                    item["profile_relative_difference"],
                    item["axis_relative_difference"],
                    item["brane_relative_difference"],
                ) for item in cross_history.values()
            ) < .002
            and max(
                item["displacement_relative_difference"]
                for item in cross_history.values()
            ) < .10
        ),
        "final_G7_G8_position_velocity_transfer_below_5_percent": bool(
            max(final_spacetime_transfer.values()) < .05
        ),
    }
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "classification": (
            selected_history["classification"]
            if all(acceptance.values()) else "unresolved_replication"
        ),
        "scope": "sealed G8 spacetime-grid replication of the four-checkpoint capped marginal-surface history through t=0.004",
        "protocol": "notes/70_G8_cap_motion_replication_protocol.md",
        "times": TIMES.tolist(),
        "time_step": FINAL_TIME / STEPS,
        "G8_t0_002_note_63_reproduction": reproduction,
        "tracker_records": {
            f"t_{time:.3f}": public_record(record)
            for time, record in zip(TIMES, records)
        },
        "G8_selected_history": selected_history,
        "G8_confirmation_history": confirmation_history,
        "G7_G8_selected_horizon_transfer": cross_history,
        "final_G7_G8_spacetime_transfer": final_spacetime_transfer,
        "acceptance": acceptance,
        "final_G8_diagnostics": {
            "global_GH_constraint": run["final_constraint"]["global_relative"],
            "wall_position_residual": run["final_wall"]["maximum"],
            "normal_wall_position_residual": run["final_normal_wall_position_residual"]["maximum"],
            "maximum_outer_metric_correction": run["maximum_outer_metric_correction"],
            "maximum_outer_scalar_correction": run["maximum_outer_scalar_correction"],
            "maximum_outer_source_correction": run["maximum_outer_source_correction"],
        },
        "limitations": [
            "coordinate motion of a pre-existing marginal cap through t=0.004",
            "kinematic classification rather than cause",
            "does not establish long-time convergence",
            "not formation, event-horizon location, topology change, branch selection, an open basin, or mass transfer",
        ],
    }
    np.savez_compressed(
        CHECKPOINT, z=case["z"], r=case["r"], times=TIMES,
        initial_horizon=records[0]["initial_on"],
        **{
            f"time_{index}_{name}": value
            for index, (state, record) in enumerate(zip(states, records))
            for name, value in (
                ("increment", state["_increment"]),
                ("velocity", state["_velocity"]),
                ("selected_horizon", record["selected"]["rho"]),
                ("confirmation_horizon", record["confirmation"]["rho"]),
            )
        },
    )
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
