#!/usr/bin/env python3
"""Sealed G7/G8 A=8.0 robustness of the four-checkpoint cap motion."""

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


OUTPUT = Path("results/corrected_fold_A8_cap_motion_robustness.json")
CHECKPOINT = Path("results/corrected_fold_A8_cap_motion_robustness_state.npz")
G7_REFERENCE_JSON = Path("results/corrected_fold_G7_cap_motion_history.json")
G7_REFERENCE_STATE = Path("results/corrected_fold_G7_cap_motion_history_state.npz")
G8_REFERENCE_JSON = Path("results/corrected_fold_G8_cap_motion_replication.json")
G8_REFERENCE_STATE = Path("results/corrected_fold_G8_cap_motion_replication_state.npz")
AMPLITUDE = 8.0
REFERENCE_AMPLITUDE = 7.94
FINAL_TIME = 0.004
STEPS = 8
CHECKPOINT_STEPS = (2, 4, 6)
TIMES = np.array((0.001, 0.002, 0.003, 0.004))


def relative_scalar(left, right):
    return float(abs(left - right) / max(abs(left), abs(right), 1e-300))


def evolve_history(geometry, label):
    initial = static_cap(geometry)
    case = live.setup_case(
        geometry, label, live_normal_wall_gauge=True,
        live_outer_sommerfeld=True,
    )
    live.FINAL_TIME = FINAL_TIME
    live.STEPS = STEPS
    print(f"{label}: evolving with three internal checkpoints", flush=True)
    run = live.integrate(case, checkpoint_steps=CHECKPOINT_STEPS)
    states = [run["_checkpoints"][step] for step in CHECKPOINT_STEPS] + [run]
    position0 = geometry["jet_field"].reduced_fields
    records = [
        select_surface(
            f"{label}_t{time:.3f}", position0 + state["_increment"],
            state["_velocity"], geometry, initial,
        )
        for time, state in zip(TIMES, states)
    ]
    return {
        "geometry": geometry,
        "case": case,
        "initial": initial,
        "run": run,
        "states": states,
        "records": records,
        "selected_history": history_metrics(records, use_confirmation=False),
        "confirmation_history": history_metrics(records, use_confirmation=True),
    }


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


def evolution_pass(item):
    run = item["run"]
    return bool(
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
    )


def main():
    reference_json = {
        "G7": json.loads(G7_REFERENCE_JSON.read_text()),
        "G8": json.loads(G8_REFERENCE_JSON.read_text()),
    }
    reference_state = {
        "G7": np.load(G7_REFERENCE_STATE),
        "G8": np.load(G8_REFERENCE_STATE),
    }
    print("building fresh corrected G7/G8 A=8.0 geometries", flush=True)
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": AMPLITUDE}
    g7 = build_refined(
        seed, 81, 121, "G7A8000", selector_iterations=40,
        slice_iterations=270,
    )
    g8 = build_refined(
        g7, 97, 145, "G8A8000", selector_iterations=45,
        slice_iterations=280,
    )
    histories = {
        "G7": evolve_history(g7, "G7-A8"),
        "G8": evolve_history(g8, "G8-A8"),
    }

    static_grid_transfer = {
        name: relative_scalar(
            histories["G7"]["initial"][name], histories["G8"]["initial"][name],
        ) for name in ("rho_axis", "rho_brane")
    }
    grid_horizon_transfer = {}
    for index, time in enumerate(TIMES):
        left = histories["G7"]["records"][index]
        right = histories["G8"]["records"][index]
        grid_horizon_transfer[f"t_{time:.3f}"] = {
            "profile_relative_difference": relative_norm(
                left["selected"]["rho"], right["selected"]["rho"],
            ),
            "axis_relative_difference": relative_scalar(
                left["selected"]["rho_axis"], right["selected"]["rho_axis"],
            ),
            "brane_relative_difference": relative_scalar(
                left["selected"]["rho_brane"], right["selected"]["rho_brane"],
            ),
            "displacement_relative_difference": relative_norm(
                left["selected"]["rho"] - left["initial_on"],
                right["selected"]["rho"] - right["initial_on"],
            ),
        }

    mask = g7["r"] <= RADIAL_COMPARISON_CUT + 1e-12
    target_r = g7["r"][mask]
    final_grid_transfer = {}
    for name, key in (
        ("position_increment", "_increment"),
        ("velocity", "_velocity"),
        ("source_increment", "_source_increment"),
    ):
        fine = interpolate_fields(
            histories["G8"]["run"][key], g8["z"], g8["r"],
            g7["z"], target_r,
        )
        final_grid_transfer[name] = relative_norm(
            histories["G7"]["run"][key][:, mask], fine,
        )

    amplitude_transfer = {}
    for label in ("G7", "G8"):
        current = histories[label]
        archived = reference_state[label]
        static_profile = relative_norm(
            current["records"][0]["initial_on"], archived["initial_horizon"],
        )
        static_radii = {
            "axis": relative_scalar(
                current["records"][0]["initial_on"][0], archived["initial_horizon"][0],
            ),
            "brane": relative_scalar(
                current["records"][0]["initial_on"][-1], archived["initial_horizon"][-1],
            ),
        }
        displacement_ratios = []
        endpoint_motion = []
        for index, record in enumerate(current["records"]):
            current_displacement = record["selected"]["rho"] - record["initial_on"]
            archived_displacement = (
                archived[f"time_{index}_selected_horizon"]
                - archived["initial_horizon"]
            )
            displacement_ratios.append(float(
                np.linalg.norm(current_displacement)
                / max(np.linalg.norm(archived_displacement), 1e-300)
            ))
            endpoint_motion.append(bool(
                record["selected"]["rho_axis"] > record["initial_on"][0]
                and record["selected"]["rho_brane"] > record["initial_on"][-1]
            ))
        amplitude_transfer[label] = {
            "initial_profile_relative_difference": static_profile,
            "initial_radius_relative_difference": static_radii,
            "displacement_ratios_A8_over_A794": displacement_ratios,
            "all_endpoint_motion_positive": all(endpoint_motion),
        }

    reference_exponents = {
        "G7": {
            "selected": reference_json["G7"]["selected_history"]["displacement_power_exponent"],
            "confirmation": reference_json["G7"]["confirmation_history"]["displacement_power_exponent"],
        },
        "G8": {
            "selected": reference_json["G8"]["G8_selected_history"]["displacement_power_exponent"],
            "confirmation": reference_json["G8"]["G8_confirmation_history"]["displacement_power_exponent"],
        },
    }
    current_classes = [
        histories[label][name]["classification"]
        for label in ("G7", "G8")
        for name in ("selected_history", "confirmation_history")
    ]
    current_exponents = {
        label: {
            "selected": histories[label]["selected_history"]["displacement_power_exponent"],
            "confirmation": histories[label]["confirmation_history"]["displacement_power_exponent"],
        } for label in ("G7", "G8")
    }
    acceptance = {
        "static_caps_converge_and_G7_G8_radii_transfer": bool(
            all(
                item["initial"]["converged"]
                and item["initial"]["surface_residual_max"] < 1e-6
                for item in histories.values()
            ) and max(static_grid_transfer.values()) < .002
        ),
        "both_evolutions_constraint_wall_boundary_pass": bool(all(
            evolution_pass(item) for item in histories.values()
        )),
        "all_discrepancy_trackers_and_transfers_pass": bool(all(
            tracker_pass(item["records"]) for item in histories.values()
        )),
        "all_histories_match_reference_class": bool(
            len(set(current_classes)) == 1
            and current_classes[0] == reference_json["G7"]["classification"]
        ),
        "exponents_transfer_across_grid_and_amplitude": bool(
            abs(current_exponents["G7"]["selected"] - current_exponents["G8"]["selected"]) < .1
            and abs(current_exponents["G7"]["confirmation"] - current_exponents["G8"]["confirmation"]) < .1
            and all(
                abs(current_exponents[label][kind] - reference_exponents[label][kind]) < .15
                for label in ("G7", "G8") for kind in ("selected", "confirmation")
            )
        ),
        "all_time_A8_G7_G8_horizon_transfers_pass": bool(
            max(
                max(
                    item["profile_relative_difference"],
                    item["axis_relative_difference"],
                    item["brane_relative_difference"],
                ) for item in grid_horizon_transfer.values()
            ) < .002
            and max(
                item["displacement_relative_difference"]
                for item in grid_horizon_transfer.values()
            ) < .10
        ),
        "final_A8_G7_G8_field_transfer_below_5_percent": bool(
            max(final_grid_transfer.values()) < .05
        ),
        "nearby_amplitude_profile_displacement_and_motion_pass": bool(all(
            record["initial_profile_relative_difference"] < .05
            and max(record["initial_radius_relative_difference"].values()) < .05
            and min(record["displacement_ratios_A8_over_A794"]) > .5
            and max(record["displacement_ratios_A8_over_A794"]) < 2.
            and record["all_endpoint_motion_positive"]
            for record in amplitude_transfer.values()
        )),
    }
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "classification": (
            current_classes[0] if all(acceptance.values()) else "unresolved_robustness"
        ),
        "scope": "sealed G7/G8 A=8.0 robustness of the four-checkpoint capped marginal-surface history",
        "protocol": "notes/71_A8_cap_motion_robustness_protocol.md",
        "amplitude": AMPLITUDE,
        "reference_amplitude": REFERENCE_AMPLITUDE,
        "relative_amplitude_change": (AMPLITUDE - REFERENCE_AMPLITUDE) / REFERENCE_AMPLITUDE,
        "times": TIMES.tolist(),
        "time_step": FINAL_TIME / STEPS,
        "static_G7_G8_radius_transfer": static_grid_transfer,
        "tracker_records": {
            label: {
                f"t_{time:.3f}": public_record(record)
                for time, record in zip(TIMES, histories[label]["records"])
            } for label in ("G7", "G8")
        },
        "histories": {
            label: {
                "selected": histories[label]["selected_history"],
                "confirmation": histories[label]["confirmation_history"],
            } for label in ("G7", "G8")
        },
        "reference_exponents": reference_exponents,
        "A8_G7_G8_horizon_transfer": grid_horizon_transfer,
        "final_A8_G7_G8_field_transfer": final_grid_transfer,
        "amplitude_transfer": amplitude_transfer,
        "final_diagnostics": {
            label: {
                "global_GH_constraint": item["run"]["final_constraint"]["global_relative"],
                "wall_position_residual": item["run"]["final_wall"]["maximum"],
                "normal_wall_position_residual": item["run"]["final_normal_wall_position_residual"]["maximum"],
                "maximum_outer_metric_correction": item["run"]["maximum_outer_metric_correction"],
                "maximum_outer_scalar_correction": item["run"]["maximum_outer_scalar_correction"],
                "maximum_outer_source_correction": item["run"]["maximum_outer_source_correction"],
            } for label, item in histories.items()
        },
        "acceptance": acceptance,
        "limitations": [
            "two tested amplitudes do not establish an open basin",
            "coordinate motion of pre-existing marginal caps through t=0.004",
            "kinematic classification rather than cause",
            "not formation, event-horizon location, topology change, long-time stability, branch selection, or mass transfer",
        ],
    }
    np.savez_compressed(
        CHECKPOINT,
        **{
            f"{label}_time_{index}_{name}": value
            for label in ("G7", "G8")
            for index, (state, record) in enumerate(zip(
                histories[label]["states"], histories[label]["records"],
            ))
            for name, value in (
                ("increment", state["_increment"]),
                ("velocity", state["_velocity"]),
                ("source_increment", state["_source_increment"]),
                ("selected_horizon", record["selected"]["rho"]),
                ("confirmation_horizon", record["confirmation"]["rho"]),
            )
        },
        G7_z=g7["z"], G7_r=g7["r"], G8_z=g8["z"], G8_r=g8["r"],
        G7_initial_horizon=histories["G7"]["records"][0]["initial_on"],
        G8_initial_horizon=histories["G8"]["records"][0]["initial_on"],
        times=TIMES,
    )
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
