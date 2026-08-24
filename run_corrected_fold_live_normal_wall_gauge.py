#!/usr/bin/env python3
"""Sealed G7/G8 short evolution with a live compact-normal wall gauge row."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.nonlinear_regular_so3_evolution import (
    compact_wall_normal_gauge_position_residuals,
)
from run_corrected_fold_live_nonlinear_gauge_source import (
    DRIVER_ETA,DRIVER_MU,STEPS,TARGET_MU_LAPSE,TARGET_MU_SHIFT,TARGET_POWER,
    common_grid_vectors,integrate,public_case,public_run,setup_case,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_g7,build_g8
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import (
    FINAL_TIME,RADIAL_COMPARISON_CUT,relative_norm,
)


OUTPUT=Path("results/corrected_fold_live_normal_wall_gauge.json")
CHECKPOINT=Path("results/corrected_fold_live_normal_wall_gauge_state.npz")


def wall_acceleration_transfer(g7,g8):
    keep=g7["r"]<=RADIAL_COMPARISON_CUT+1e-12
    target_r=g7["r"][keep]
    coarse=np.stack((
        g7["_initial_live_acceleration"][0,keep,6],
        g7["_initial_live_acceleration"][-1,keep,6],
    ))
    fine=np.empty_like(coarse)
    for wall,index in enumerate((0,-1)):
        fine[wall]=CubicSpline(
            g8["r"],g8["_initial_live_acceleration"][index,:,6],
        )(target_r)
    return {
        "relative_difference":relative_norm(coarse,fine),
        "absolute_difference":float(np.linalg.norm(coarse-fine)),
        "G7_norm":float(np.linalg.norm(coarse)),
        "G8_interpolated_norm":float(np.linalg.norm(fine)),
    }


def main():
    print("building corrected G7/G8 states",flush=True)
    g6_geometry=build_geometry("G6")
    g7_geometry=build_g7(g6_geometry)
    g8_geometry=build_g8(g7_geometry)
    g7=setup_case(g7_geometry,"G7-wall",live_normal_wall_gauge=True)
    g8=setup_case(g8_geometry,"G8-wall",live_normal_wall_gauge=True)
    cases=(g7,g8)
    initial_normal_position={
        case["label"]:compact_wall_normal_gauge_position_residuals(
            case["initial"],case["source0"],case["z"],case["r"],
            case["geometry"]["background"],
        ) for case in cases
    }
    normal_transfer=wall_acceleration_transfer(g7,g8)
    print(json.dumps({
        "initial_normal_position":initial_normal_position,
        "initial_normal_acceleration_transfer":normal_transfer,
    },indent=2),flush=True)
    g7_run=integrate(g7);g8_run=integrate(g8);runs=(g7_run,g8_run)
    grid={}
    for name,key in (
        ("position_increment","_increment"),("velocity","_velocity"),
        ("source_increment","_source_increment"),
    ):
        coarse,fine=common_grid_vectors(g7,g7_run,g8,g8_run,key)
        grid[f"{name}_relative_difference"]=relative_norm(coarse,fine)
    initial_acceleration_residuals={
        case["label"]:case["initial_live_normal_wall_gauge"]["final_residual"]["maximum"]
        for case in cases
    }
    final_iteration_corrections={
        case["label"]:case["initial_live_normal_wall_gauge"]["iterations"][-1]["relative_correction"]
        for case in cases
    }
    acceptance={
        "initial_normal_position_rows_below_0_05_percent":bool(max(
            item["maximum"] for item in initial_normal_position.values()
        )<.0005),
        "initial_normal_position_row_decreases_G7_G8":bool(
            initial_normal_position["G8-wall"]["maximum"]
            <initial_normal_position["G7-wall"]["maximum"]
        ),
        "initial_normal_acceleration_rows_below_1e_10":bool(max(
            initial_acceleration_residuals.values()
        )<1e-10),
        "finite_time_normal_acceleration_rows_below_1e_10":bool(max(
            run["maximum_normal_wall_acceleration_residual"] for run in runs
        )<1e-10),
        "fourth_wall_iteration_correction_below_1e_10":bool(max(
            final_iteration_corrections.values()
        )<1e-10),
        "live_frozen_initial_acceleration_difference_below_5_percent":bool(max(
            case["initial_live_Taylor_acceleration_relative_difference"] for case in cases
        )<.05),
        "G7_G8_normal_wall_acceleration_transfer_below_5_percent":bool(
            normal_transfer["relative_difference"]<.05
        ),
        "all_stages_finite":bool(all(run["all_stages_finite"] for run in runs)),
        "Lorentzian_signature_preserved":bool(all(
            run["signature"]["all_points_one_negative_direction"] for run in runs
        )),
        "final_global_GH_constraints_below_0_5_percent":bool(max(
            run["final_constraint"]["global_relative"] for run in runs
        )<.005),
        "final_physical_wall_rows_below_0_05_percent":bool(max(
            run["final_wall"]["maximum"] for run in runs
        )<.0005),
        "final_normal_position_rows_below_0_05_percent":bool(max(
            run["final_normal_wall_position_residual"]["maximum"] for run in runs
        )<.0005),
        "physical_and_normal_position_rows_do_not_grow":bool(all(
            run["final_wall"]["maximum"]<=1.01*case["initial_wall"]["maximum"]+1e-8
            and run["final_normal_wall_position_residual"]["maximum"]
            <=1.01*initial_normal_position[case["label"]]["maximum"]+1e-8
            for case,run in zip(cases,runs)
        )),
        "G7_G8_position_transfer_below_5_percent":bool(
            grid["position_increment_relative_difference"]<.05
        ),
        "G7_G8_velocity_transfer_below_5_percent":bool(
            grid["velocity_relative_difference"]<.05
        ),
        "G7_G8_source_increment_transfer_below_5_percent":bool(
            grid["source_increment_relative_difference"]<.05
        ),
    }
    summary={
        "initial_normal_position_rows":initial_normal_position,
        "initial_normal_acceleration_residuals":initial_acceleration_residuals,
        "fourth_iteration_relative_corrections":final_iteration_corrections,
        "normal_wall_acceleration_transfer":normal_transfer,
        "live_grid_transfer":grid,
        "final_global_GH_constraint_by_resolution":{
            case["label"]:run["final_constraint"]["global_relative"]
            for case,run in zip(cases,runs)
        },
        "final_normal_position_residual_by_resolution":{
            case["label"]:run["final_normal_wall_position_residual"]["maximum"]
            for case,run in zip(cases,runs)
        },
        "maximum_finite_time_normal_acceleration_residual":max(
            run["maximum_normal_wall_acceleration_residual"] for run in runs
        ),
    }
    np.savez_compressed(
        CHECKPOINT,G7_z=g7["z"],G7_r=g7["r"],G8_z=g8["z"],G8_r=g8["r"],
        G7_increment=g7_run["_increment"],G7_velocity=g7_run["_velocity"],
        G7_source=g7_run["_source"],G8_increment=g8_run["_increment"],
        G8_velocity=g8_run["_velocity"],G8_source=g8_run["_source"],
    )
    payload={
        "status":"pass" if all(acceptance.values()) else "review",
        "scope":"sealed G7/G8 short live nonlinear evolution with constraint-compatible compact-normal wall gauge acceleration",
        "protocol":"notes/57_live_compact_normal_wall_gauge_protocol.md",
        "driver_parameters":{"mu":DRIVER_MU,"eta":DRIVER_ETA},
        "target_parameters":{
            "mu_lapse":TARGET_MU_LAPSE,"mu_shift":TARGET_MU_SHIFT,
            "determinant_power":TARGET_POWER,
        },
        "final_time":FINAL_TIME,"steps":STEPS,
        "cases":[
            {"initial":public_case(case),"run":public_run(run)}
            for case,run in zip(cases,runs)
        ],
        "summary":summary,"acceptance":acceptance,
        "limitations":[
            "very short physical-wall gauge test",
            "driver and target parameters remain numerical controls",
            "one-sided open-bulk artificial radial boundary remains",
            "no long-time well-posedness, stability, collapse, horizon, topology, or mass-transfer claim",
        ],
    }
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":payload["status"],"summary":summary,"acceptance":acceptance},indent=2),flush=True)


if __name__=="__main__":main()
