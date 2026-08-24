#!/usr/bin/env python3
"""Sealed multi-checkpoint classification of G7 capped-surface motion."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import run_corrected_fold_live_nonlinear_gauge_source as live
from run_corrected_fold_discrepancy_horizon_tracker import (
    public_record, select_surface, static_cap,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import relative_norm


OUTPUT = Path("results/corrected_fold_G7_cap_motion_history.json")
CHECKPOINT = Path("results/corrected_fold_G7_cap_motion_history_state.npz")
BASELINE = Path("results/corrected_fold_G7_doubled_duration_horizon_state.npz")
FINAL_TIME = 0.004
STEPS = 8
CHECKPOINT_STEPS = (2, 4, 6)
TIMES = np.array((0.001, 0.002, 0.003, 0.004))


def history_metrics(records, use_confirmation=False):
    key = "confirmation" if use_confirmation else "selected"
    surfaces = [record[key] for record in records]
    initial = records[0]["initial_on"]
    displacements = np.array([
        np.linalg.norm(surface["rho"] - initial) for surface in surfaces
    ])
    axis = np.array([surface["rho_axis"] for surface in surfaces])
    brane = np.array([surface["rho_brane"] for surface in surfaces])
    exponent = float(np.polyfit(np.log(TIMES), np.log(displacements), 1)[0])
    monotone = bool(
        np.all(np.diff(displacements) > 0)
        and np.all(np.diff(axis) > 0)
        and np.all(np.diff(brane) > 0)
    )
    if not monotone:
        classification = "stalled_or_nonmonotone"
    elif exponent >= 1.5:
        classification = "accelerating_or_quadratic"
    elif exponent > 0:
        classification = "monotone_subquadratic"
    else:
        classification = "stalled_or_nonmonotone"
    return {
        "classification": classification,
        "displacement_power_exponent": exponent,
        "displacement_norms": displacements.tolist(),
        "successive_displacement_increments": np.diff(displacements).tolist(),
        "rho_axis": axis.tolist(),
        "rho_brane": brane.tolist(),
        "successive_axis_increments": np.diff(axis).tolist(),
        "successive_brane_increments": np.diff(brane).tolist(),
        "fractional_axis_changes": ((axis - initial[0]) / initial[0]).tolist(),
        "fractional_brane_changes": ((brane - initial[-1]) / initial[-1]).tolist(),
    }


def main():
    baseline = np.load(BASELINE)
    print("building corrected G7 A=7.94 history state", flush=True)
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": 7.94}
    geometry = build_refined(
        seed, 81, 121, "G7A794-history", selector_iterations=40,
        slice_iterations=270,
    )
    initial = static_cap(geometry)
    case = live.setup_case(
        geometry, "G7-history", live_normal_wall_gauge=True,
        live_outer_sommerfeld=True,
    )
    live.FINAL_TIME = FINAL_TIME
    live.STEPS = STEPS
    print("evolving one trajectory with three internal checkpoints", flush=True)
    run = live.integrate(case, checkpoint_steps=CHECKPOINT_STEPS)
    states = [run["_checkpoints"][step] for step in CHECKPOINT_STEPS] + [run]

    reproduction = {
        "t_0_002_position": relative_norm(
            states[1]["_increment"], baseline["halfway_increment"],
        ),
        "t_0_002_velocity": relative_norm(
            states[1]["_velocity"], baseline["halfway_velocity"],
        ),
        "t_0_004_position": relative_norm(
            states[3]["_increment"], baseline["final_increment"],
        ),
        "t_0_004_velocity": relative_norm(
            states[3]["_velocity"], baseline["final_velocity"],
        ),
    }
    position0 = geometry["jet_field"].reduced_fields
    records = []
    for time, state in zip(TIMES, states):
        records.append(select_surface(
            f"G7_t{time:.3f}", position0 + state["_increment"],
            state["_velocity"], geometry, initial,
        ))

    selected_history = history_metrics(records, use_confirmation=False)
    confirmation_history = history_metrics(records, use_confirmation=True)
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
    numerical_acceptance = {
        "archived_states_reproduce_below_1e_8": bool(max(reproduction.values()) < 1e-8),
        "evolution_constraint_wall_and_boundary_pass": bool(
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
        ),
        "all_discrepancy_trackers_and_transfers_pass": tracker_pass,
        "selected_and_confirmation_classes_agree": bool(
            selected_history["classification"]
            == confirmation_history["classification"]
        ),
    }
    classifiable = all(numerical_acceptance.values())
    classification = (
        selected_history["classification"] if classifiable else "unresolved"
    )
    payload = {
        "status": "pass" if classifiable else "review",
        "classification": classification,
        "scope": "sealed numerical classification of the resolved G7 capped marginal-surface motion through t=0.004",
        "protocol": "notes/69_G7_cap_motion_history_protocol.md",
        "times": TIMES.tolist(),
        "time_step": FINAL_TIME / STEPS,
        "archived_state_reproduction": reproduction,
        "tracker_records": {
            f"t_{time:.3f}": public_record(record)
            for time, record in zip(TIMES, records)
        },
        "selected_history": selected_history,
        "confirmation_history": confirmation_history,
        "numerical_acceptance": numerical_acceptance,
        "final_evolution_diagnostics": {
            "global_GH_constraint": run["final_constraint"]["global_relative"],
            "wall_position_residual": run["final_wall"]["maximum"],
            "normal_wall_position_residual": run["final_normal_wall_position_residual"]["maximum"],
            "maximum_outer_metric_correction": run["maximum_outer_metric_correction"],
            "maximum_outer_scalar_correction": run["maximum_outer_scalar_correction"],
            "maximum_outer_source_correction": run["maximum_outer_source_correction"],
        },
        "limitations": [
            "coordinate motion of a pre-existing marginal cap",
            "single G7 spacetime grid through t=0.004",
            "classification is kinematic and does not identify a cause",
            "not formation, event-horizon location, topology change, long-time stability, branch selection, an open basin, or mass transfer",
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
