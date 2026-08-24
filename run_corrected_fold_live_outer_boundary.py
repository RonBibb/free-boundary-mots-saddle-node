#!/usr/bin/env python3
"""Sealed G7/G8 live nonlinear outer-radial Sommerfeld boundary gate."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.nonlinear_regular_so3_evolution import regular_so3_outward_radial_speed
from run_corrected_fold_live_nonlinear_gauge_source import (
    DRIVER_ETA,DRIVER_MU,STEPS,TARGET_MU_LAPSE,TARGET_MU_SHIFT,TARGET_POWER,
    common_grid_vectors,integrate,public_case,public_run,setup_case,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_g7,build_g8
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import (
    FINAL_TIME,RADIAL_COMPARISON_CUT,relative_norm,
)


OUTPUT=Path("results/corrected_fold_live_outer_boundary.json")
CHECKPOINT=Path("results/corrected_fold_live_outer_boundary_state.npz")
BASELINE=Path("results/corrected_fold_live_normal_wall_gauge_state.npz")


def inner_baseline_comparison(case,run,baseline,prefix):
    keep=case["r"]<=RADIAL_COMPARISON_CUT+1e-12
    return {
        "position_increment_relative_difference":relative_norm(
            run["_increment"][:,keep],baseline[f"{prefix}_increment"][:,keep],
        ),
        "velocity_relative_difference":relative_norm(
            run["_velocity"][:,keep],baseline[f"{prefix}_velocity"][:,keep],
        ),
        "source_relative_difference":relative_norm(
            run["_source"][:,keep],baseline[f"{prefix}_source"][:,keep],
        ),
    }


def main():
    if not BASELINE.exists():raise FileNotFoundError("passing live-wall checkpoint is required")
    print("building corrected G7/G8 states",flush=True)
    g6_geometry=build_geometry("G6")
    g7_geometry=build_g7(g6_geometry);g8_geometry=build_g8(g7_geometry)
    g7=setup_case(
        g7_geometry,"G7-outer",live_normal_wall_gauge=True,
        live_outer_sommerfeld=True,
    )
    g8=setup_case(
        g8_geometry,"G8-outer",live_normal_wall_gauge=True,
        live_outer_sommerfeld=True,
    )
    cases=(g7,g8)
    initial_speed={
        case["label"]:(
            case["initial_live_outer_sommerfeld"]["minimum_outward_speed"],
            case["initial_live_outer_sommerfeld"]["maximum_outward_speed"],
        ) for case in cases
    }
    speed7=regular_so3_outward_radial_speed(g7["initial"],g7["r"])
    speed8=regular_so3_outward_radial_speed(g8["initial"],g8["r"])
    speed8_on_g7=CubicSpline(g8["z"],speed8)(g7["z"])
    speed_transfer=relative_norm(speed7,speed8_on_g7)
    g7_run=integrate(g7);g8_run=integrate(g8);runs=(g7_run,g8_run)
    baseline=np.load(BASELINE)
    inner={
        "G7-outer":inner_baseline_comparison(g7,g7_run,baseline,"G7"),
        "G8-outer":inner_baseline_comparison(g8,g8_run,baseline,"G8"),
    }
    grid={}
    for name,key in (
        ("position_increment","_increment"),("velocity","_velocity"),
        ("source_increment","_source_increment"),
    ):
        coarse,fine=common_grid_vectors(g7,g7_run,g8,g8_run,key)
        grid[f"{name}_relative_difference"]=relative_norm(coarse,fine)
    all_corrections=[
        run[key] for run in runs for key in (
            "maximum_outer_metric_correction","maximum_outer_scalar_correction",
            "maximum_outer_source_correction",
        )
    ]
    initial_acceleration_residual=max(
        case["initial_live_outer_sommerfeld"]["maximum_normalized_acceleration_residual"]
        for case in cases
    )
    acceptance={
        "outward_speeds_finite_and_positive":bool(all(
            np.isfinite(value) and value>0 for pair in initial_speed.values() for value in pair
        )),
        "G7_G8_speed_range_transfer_below_5_percent":bool(speed_transfer<.05),
        "initial_outer_acceleration_residual_below_1e_10":bool(initial_acceleration_residual<1e-10),
        "finite_time_outer_acceleration_residual_below_1e_10":bool(max(
            run["maximum_outer_acceleration_residual"] for run in runs
        )<1e-10),
        "finite_time_outer_source_residual_below_1e_10":bool(max(
            run["maximum_outer_source_residual"] for run in runs
        )<1e-10),
        "final_outer_position_residual_below_1e_8":bool(max(
            run["final_outer_sommerfeld_position_residual"]["maximum_normalized"]
            for run in runs
        )<1e-8),
        "final_outer_source_residual_below_1e_8":bool(max(
            run["final_outer_source_sommerfeld_residual"]["maximum_normalized"]
            for run in runs
        )<1e-8),
        "outer_boundary_is_active_but_below_5_percent":bool(
            max(all_corrections)>1e-8 and max(all_corrections)<.05
        ),
        "inner_solution_matches_wall_only_checkpoint_below_1e_8":bool(max(
            value for record in inner.values() for value in record.values()
        )<1e-8),
        "all_stages_finite":bool(all(run["all_stages_finite"] for run in runs)),
        "Lorentzian_signature_preserved":bool(all(
            run["signature"]["all_points_one_negative_direction"] for run in runs
        )),
        "final_global_GH_constraints_below_0_5_percent":bool(max(
            run["final_constraint"]["global_relative"] for run in runs
        )<.005),
        "physical_and_normal_wall_rows_below_0_05_percent":bool(max(
            max(run["final_wall"]["maximum"],run["final_normal_wall_position_residual"]["maximum"])
            for run in runs
        )<.0005),
        "physical_and_normal_wall_rows_do_not_grow":bool(all(
            run["final_wall"]["maximum"]<=1.01*case["initial_wall"]["maximum"]+1e-8
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
        "initial_outward_speed_ranges":initial_speed,
        "speed_range_transfer":speed_transfer,
        "initial_outer_acceleration_residual":initial_acceleration_residual,
        "maximum_finite_time_outer_acceleration_residual":max(
            run["maximum_outer_acceleration_residual"] for run in runs
        ),
        "maximum_finite_time_outer_source_residual":max(
            run["maximum_outer_source_residual"] for run in runs
        ),
        "maximum_outer_corrections":{
            case["label"]:{
                "metric":run["maximum_outer_metric_correction"],
                "scalar":run["maximum_outer_scalar_correction"],
                "source":run["maximum_outer_source_correction"],
            } for case,run in zip(cases,runs)
        },
        "final_outer_position_residual":{
            case["label"]:run["final_outer_sommerfeld_position_residual"]
            for case,run in zip(cases,runs)
        },
        "inner_wall_only_checkpoint_comparison":inner,
        "live_grid_transfer":grid,
        "final_global_GH_constraint":{
            case["label"]:run["final_constraint"]["global_relative"]
            for case,run in zip(cases,runs)
        },
    }
    np.savez_compressed(
        CHECKPOINT,G7_z=g7["z"],G7_r=g7["r"],G8_z=g8["z"],G8_r=g8["r"],
        G7_initial_outward_speed=speed7,G8_initial_outward_speed=speed8,
        G7_increment=g7_run["_increment"],G7_velocity=g7_run["_velocity"],
        G7_source=g7_run["_source"],G8_increment=g8_run["_increment"],
        G8_velocity=g8_run["_velocity"],G8_source=g8_run["_source"],
    )
    payload={
        "status":"pass" if all(acceptance.values()) else "review",
        "scope":"sealed G7/G8 short live nonlinear complete-field outer Sommerfeld boundary",
        "protocol":"notes/58_live_nonlinear_outer_boundary_protocol.md",
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
            "very short nonlinear boundary-compatibility test",
            "absorption evidence is inherited from the admitted linear energy, symbol, and pulse controls",
            "quadratic initial reference is not a long-duration asymptotic background model",
            "no nonlinear continuum well-posedness, long-time stability, collapse, horizon, topology, or mass-transfer claim",
        ],
    }
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":payload["status"],"summary":summary,"acceptance":acceptance},indent=2),flush=True)


if __name__=="__main__":main()
