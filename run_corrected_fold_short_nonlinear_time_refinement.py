#!/usr/bin/env python3
"""Sealed time-step refinement of the first short nonlinear evolution."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.nonlinear_regular_so3_evolution import (
    NativeRegularSO3RHS,
    compact_wall_position_residuals,
    gauge_constraint_summary,
    gauge_taylor_source_from_initial_jets,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_g7
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import (
    FINAL_TIME,
    RADIAL_COMPARISON_CUT,
    interpolate_fields,
    relative_norm,
    signature_summary,
)


OUTPUT = Path("results/corrected_fold_short_nonlinear_time_refinement.json")
CHECKPOINT = Path("results/corrected_fold_short_nonlinear_time_refinement_state.npz")
G6_STEPS = (2, 4, 8)
G7_STEPS = 4


def setup_case(geometry, label):
    z = np.asarray(geometry["z"], dtype=float)
    r = np.asarray(geometry["r"], dtype=float)
    jet = geometry["jet_field"]
    initial = np.asarray(jet.reduced_fields, dtype=float).copy()
    archived_acceleration = np.asarray(jet.reduced_second[0, 0], dtype=float)
    print(f"{label}: constructing one shared initial Taylor gauge source", flush=True)
    gauge = gauge_taylor_source_from_initial_jets(jet, z, r)
    normal_wall_acceleration = np.stack((
        archived_acceleration[0, :, 6], archived_acceleration[-1, :, 6],
    ))
    rhs = NativeRegularSO3RHS(
        z, r, gauge, geometry["mass_squared"], geometry["background"],
        normal_wall_acceleration,
    )
    zero_velocity = np.zeros_like(initial)
    print(f"{label}: evaluating shared zero-stage diagnostics", flush=True)
    initial_constraint = gauge_constraint_summary(
        initial, zero_velocity, 0.0, rhs, radial_cut=RADIAL_COMPARISON_CUT,
    )
    initial_wall = compact_wall_position_residuals(
        initial, z, r, geometry["background"],
    )
    return {
        "label": label, "geometry": geometry, "z": z, "r": r,
        "initial": initial, "rhs": rhs,
        "initial_constraint": initial_constraint, "initial_wall": initial_wall,
    }


def integrate(case, steps):
    steps = int(steps)
    position = case["initial"].copy()
    velocity = np.zeros_like(position)
    dt = FINAL_TIME / steps
    time = 0.0
    all_finite = True
    maximum_stage_change = 0.0
    maximum_wall_correction = 0.0
    for step in range(steps):
        print(
            f"{case['label']} {steps} steps: {step + 1}/{steps}, stage 1",
            flush=True,
        )
        acceleration, diagnostic = case["rhs"].acceleration(
            time, position, velocity,
        )
        midpoint_position = position + 0.5 * dt * velocity
        midpoint_velocity = velocity + 0.5 * dt * acceleration
        print(
            f"{case['label']} {steps} steps: {step + 1}/{steps}, stage 2",
            flush=True,
        )
        midpoint_acceleration, midpoint_diagnostic = case["rhs"].acceleration(
            time + 0.5 * dt, midpoint_position, midpoint_velocity,
        )
        all_finite = bool(
            all_finite and diagnostic["finite"] and midpoint_diagnostic["finite"]
        )
        maximum_stage_change = max(
            maximum_stage_change,
            relative_norm(acceleration, midpoint_acceleration),
        )
        for item in (
            *diagnostic["wall_corrections"],
            *midpoint_diagnostic["wall_corrections"],
        ):
            maximum_wall_correction = max(
                maximum_wall_correction, item["relative_norm"],
            )
        position = position + dt * midpoint_velocity
        velocity = velocity + dt * midpoint_acceleration
        time += dt
    print(f"{case['label']} {steps} steps: final diagnostics", flush=True)
    constraint = gauge_constraint_summary(
        position, velocity, time, case["rhs"],
        radial_cut=RADIAL_COMPARISON_CUT,
    )
    wall = compact_wall_position_residuals(
        position, case["z"], case["r"], case["geometry"]["background"],
    )
    signature = signature_summary(position, case["r"])
    speed = max(
        float(np.max(case["geometry"]["principal"]["z_coordinate_speed"])),
        float(np.max(case["geometry"]["principal"]["r_coordinate_speed"])),
    )
    courant = dt * speed / min(
        float(np.min(np.diff(case["z"]))),
        float(np.min(np.diff(case["r"]))),
    )
    return {
        "steps": steps,
        "time_step": dt,
        "final_time": time,
        "maximum_coordinate_courant": courant,
        "all_stages_finite": all_finite,
        "maximum_stage_acceleration_relative_change": maximum_stage_change,
        "maximum_wall_acceleration_correction_relative": maximum_wall_correction,
        "final_constraint": constraint,
        "final_wall_rows": wall,
        "signature": signature,
        "maximum_absolute_position_increment": float(
            np.max(np.abs(position - case["initial"]))
        ),
        "maximum_absolute_velocity": float(np.max(np.abs(velocity))),
        "_position": position,
        "_increment": position - case["initial"],
        "_velocity": velocity,
    }


def temporal_summary(runs):
    position_differences = np.array([
        np.linalg.norm(runs[index]["_position"] - runs[index + 1]["_position"])
        for index in range(2)
    ])
    velocity_differences = np.array([
        np.linalg.norm(runs[index]["_velocity"] - runs[index + 1]["_velocity"])
        for index in range(2)
    ])
    return {
        "position_successive_differences": position_differences.tolist(),
        "velocity_successive_differences": velocity_differences.tolist(),
        "position_convergence_rate": float(np.log2(
            position_differences[0] / max(position_differences[1], 1e-300)
        )),
        "velocity_convergence_rate": float(np.log2(
            velocity_differences[0] / max(velocity_differences[1], 1e-300)
        )),
        "coarse_to_fine_position_relative_difference": relative_norm(
            runs[0]["_increment"], runs[-1]["_increment"],
        ),
        "coarse_to_fine_velocity_relative_difference": relative_norm(
            runs[0]["_velocity"], runs[-1]["_velocity"],
        ),
    }


def grid_summary(g6_case, g6_run, g7_case, g7_run):
    mask = g6_case["r"] <= RADIAL_COMPARISON_CUT + 1e-12
    target_r = g6_case["r"][mask]
    fine_increment = interpolate_fields(
        g7_run["_increment"], g7_case["z"], g7_case["r"],
        g6_case["z"], target_r,
    )
    fine_velocity = interpolate_fields(
        g7_run["_velocity"], g7_case["z"], g7_case["r"],
        g6_case["z"], target_r,
    )
    coarse_increment = g6_run["_increment"][:, mask]
    coarse_velocity = g6_run["_velocity"][:, mask]
    per_field = []
    for field in range(9):
        increment_difference = float(np.linalg.norm(
            coarse_increment[:, :, field] - fine_increment[:, :, field]
        ))
        velocity_difference = float(np.linalg.norm(
            coarse_velocity[:, :, field] - fine_velocity[:, :, field]
        ))
        per_field.append({
            "field": field,
            "increment_relative_difference": relative_norm(
                coarse_increment[:, :, field], fine_increment[:, :, field],
            ),
            "increment_absolute_difference": increment_difference,
            "increment_signal_norm": float(max(
                np.linalg.norm(coarse_increment[:, :, field]),
                np.linalg.norm(fine_increment[:, :, field]),
            )),
            "velocity_relative_difference": relative_norm(
                coarse_velocity[:, :, field], fine_velocity[:, :, field],
            ),
            "velocity_absolute_difference": velocity_difference,
            "velocity_signal_norm": float(max(
                np.linalg.norm(coarse_velocity[:, :, field]),
                np.linalg.norm(fine_velocity[:, :, field]),
            )),
        })
    return {
        "position_increment_relative_difference": relative_norm(
            coarse_increment, fine_increment,
        ),
        "velocity_relative_difference": relative_norm(
            coarse_velocity, fine_velocity,
        ),
        "per_field": per_field,
    }


def public_run(run):
    return {key: value for key, value in run.items() if not key.startswith("_")}


def main():
    print("building G6 corrected-fold state", flush=True)
    g6_geometry = build_geometry("G6")
    print("building G7 corrected-fold state", flush=True)
    g7_geometry = build_g7(g6_geometry)
    g6 = setup_case(g6_geometry, "G6")
    g6_runs = [integrate(g6, steps) for steps in G6_STEPS]
    g7 = setup_case(g7_geometry, "G7")
    g7_run = integrate(g7, G7_STEPS)

    temporal = temporal_summary(g6_runs)
    grid = grid_summary(g6, g6_runs[1], g7, g7_run)
    constraints = [run["final_constraint"]["global_relative"] for run in g6_runs]
    all_constraints = constraints + [g7_run["final_constraint"]["global_relative"]]
    wall_values = [run["final_wall_rows"]["maximum"] for run in g6_runs]
    initial_wall = g6["initial_wall"]["maximum"]
    constraint_spread = (
        max(constraints) - min(constraints)
    ) / max(max(constraints), 1e-300)
    acceptance = {
        "all_stages_and_constraints_finite": bool(
            all(run["all_stages_finite"] and run["final_constraint"]["finite"] for run in (*g6_runs, g7_run))
        ),
        "Lorentzian_signature_preserved": bool(
            all(run["signature"]["all_points_one_negative_direction"] for run in (*g6_runs, g7_run))
        ),
        "maximum_Courant_below_0_1": bool(
            max(run["maximum_coordinate_courant"] for run in (*g6_runs, g7_run)) < 0.1
        ),
        "position_time_convergence_rate_at_least_1_5": bool(
            temporal["position_convergence_rate"] >= 1.5
        ),
        "velocity_time_convergence_rate_at_least_1_5": bool(
            temporal["velocity_convergence_rate"] >= 1.5
        ),
        "all_final_global_GH_constraints_below_0_5_percent": bool(
            max(all_constraints) < 0.005
        ),
        "G6_constraint_time_step_spread_below_25_percent": bool(
            constraint_spread < 0.25
        ),
        "finest_G6_wall_residual_below_0_05_percent": bool(
            wall_values[-1] < 0.0005
        ),
        "no_G6_wall_residual_growth_beyond_allowance": bool(
            max(wall_values) <= 1.01 * initial_wall + 1e-8
        ),
        "four_step_G6_G7_position_transfer_below_5_percent": bool(
            grid["position_increment_relative_difference"] < 0.05
        ),
        "four_step_G6_G7_velocity_transfer_below_5_percent": bool(
            grid["velocity_relative_difference"] < 0.05
        ),
    }
    summary = {
        "temporal": temporal,
        "four_step_grid_transfer": grid,
        "G6_final_global_GH_constraints": constraints,
        "G7_four_step_final_global_GH_constraint": g7_run["final_constraint"]["global_relative"],
        "G6_constraint_time_step_spread": constraint_spread,
        "G6_initial_wall_residual": initial_wall,
        "G6_final_wall_residuals": wall_values,
        "G7_four_step_final_wall_residual": g7_run["final_wall_rows"]["maximum"],
    }
    np.savez_compressed(
        CHECKPOINT,
        G6_z=g6["z"], G6_r=g6["r"], G7_z=g7["z"], G7_r=g7["r"],
        **{
            f"G6_steps_{run['steps']}_{name}": run[key]
            for run in g6_runs
            for name, key in (("increment", "_increment"), ("velocity", "_velocity"))
        },
        G7_steps_4_increment=g7_run["_increment"],
        G7_steps_4_velocity=g7_run["_velocity"],
    )
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "scope": "sealed time-step refinement and four-step G6/G7 transfer for the first short nonlinear evolution",
        "protocol": "notes/54_short_nonlinear_time_refinement_protocol.md",
        "final_time": FINAL_TIME,
        "G6_steps": list(G6_STEPS),
        "G7_steps": G7_STEPS,
        "radial_comparison_cut": RADIAL_COMPARISON_CUT,
        "G6_initial_constraint": g6["initial_constraint"],
        "G6_initial_wall_rows": g6["initial_wall"],
        "G7_initial_constraint": g7["initial_constraint"],
        "G7_initial_wall_rows": g7["initial_wall"],
        "G6_runs": [public_run(run) for run in g6_runs],
        "G7_run": public_run(g7_run),
        "summary": summary,
        "acceptance": acceptance,
        "limitations": [
            "t=0.002 short evolution",
            "first-order Taylor gauge source about the initial slice",
            "frozen compact-normal wall acceleration as a gauge datum",
            "one-sided open-bulk artificial radial boundary",
            "not nonlinear stability, collapse, horizon formation, or mass transfer",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"], "summary": summary,
        "acceptance": acceptance,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
