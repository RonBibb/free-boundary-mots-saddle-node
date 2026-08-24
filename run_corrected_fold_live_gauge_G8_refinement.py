#!/usr/bin/env python3
"""Fresh sealed G7/G8 refinement of the short live nonlinear gauge driver."""

import json
from pathlib import Path

import numpy as np

from run_corrected_fold_live_nonlinear_gauge_source import (
    DRIVER_ETA,DRIVER_MU,STEPS,TARGET_MU_LAPSE,TARGET_MU_SHIFT,TARGET_POWER,
    common_grid_vectors,integrate,public_case,public_run,setup_case,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_g7,build_g8
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import (
    FINAL_TIME,RADIAL_COMPARISON_CUT,interpolate_fields,relative_norm,
)


OUTPUT=Path("results/corrected_fold_live_gauge_G8_refinement.json")
CHECKPOINT=Path("results/corrected_fold_live_gauge_G8_refinement_state.npz")
PREVIOUS_SOURCE_MISMATCH=0.08870115640640658


def initial_source_time_transfer(g7,g8):
    mask=g7["r"]<=RADIAL_COMPARISON_CUT+1e-12
    target_r=g7["r"][mask]
    fine=interpolate_fields(
        g8["source_time0"],g8["z"],g8["r"],g7["z"],target_r,
    )
    coarse=g7["source_time0"][:,mask]
    return {
        "combined_relative_difference":relative_norm(coarse,fine),
        "H0_relative_difference":relative_norm(coarse[:,:,0],fine[:,:,0]),
        "H0_absolute_difference":float(np.linalg.norm(coarse[:,:,0]-fine[:,:,0])),
        "H0_coarse_norm":float(np.linalg.norm(coarse[:,:,0])),
        "H0_fine_interpolated_norm":float(np.linalg.norm(fine[:,:,0])),
    }


def main():
    print("building G6 seed and corrected G7 state",flush=True)
    g6_geometry=build_geometry("G6")
    g7_geometry=build_g7(g6_geometry)
    print("building fresh corrected G8 state",flush=True)
    g8_geometry=build_g8(g7_geometry)
    g7=setup_case(g7_geometry,"G7")
    g8=setup_case(g8_geometry,"G8")
    source_time=initial_source_time_transfer(g7,g8)
    print(json.dumps({"initial_G7_G8_source_time":source_time},indent=2),flush=True)
    g7_run=integrate(g7);g8_run=integrate(g8)
    grid={}
    for name,key in (
        ("position_increment","_increment"),("velocity","_velocity"),
        ("source_increment","_source_increment"),
    ):
        coarse,fine=common_grid_vectors(g7,g7_run,g8,g8_run,key)
        grid[f"{name}_relative_difference"]=relative_norm(coarse,fine)
    cases=(g7,g8);runs=(g7_run,g8_run)
    acceptance={
        "G8_selector_below_1e_8":bool(g8_geometry["selector_maximum"]<1e-8),
        "G7_G8_initial_H0_time_transfer_below_5_percent":bool(
            source_time["H0_relative_difference"]<.05
        ),
        "all_stages_finite":bool(all(run["all_stages_finite"] for run in runs)),
        "Lorentzian_signature_preserved":bool(all(
            run["signature"]["all_points_one_negative_direction"] for run in runs
        )),
        "final_global_GH_constraints_below_0_5_percent":bool(max(
            run["final_constraint"]["global_relative"] for run in runs
        )<.005),
        "final_wall_rows_below_0_05_percent":bool(max(
            run["final_wall"]["maximum"] for run in runs
        )<.0005),
        "wall_rows_do_not_grow_beyond_allowance":bool(all(
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
        "source_increment_mismatch_decreases_from_G6_G7":bool(
            grid["source_increment_relative_difference"]<PREVIOUS_SOURCE_MISMATCH
        ),
    }
    summary={
        "G7_G8_initial_source_time_transfer":source_time,
        "G7_G8_live_grid_transfer":grid,
        "previous_G6_G7_source_increment_relative_difference":PREVIOUS_SOURCE_MISMATCH,
        "source_increment_mismatch_decrease_factor":float(
            PREVIOUS_SOURCE_MISMATCH/grid["source_increment_relative_difference"]
        ),
        "final_global_GH_constraint_by_resolution":{
            case["label"]:run["final_constraint"]["global_relative"]
            for case,run in zip(cases,runs)
        },
        "final_wall_residual_by_resolution":{
            case["label"]:run["final_wall"]["maximum"]
            for case,run in zip(cases,runs)
        },
    }
    np.savez_compressed(
        CHECKPOINT,G7_z=g7["z"],G7_r=g7["r"],G8_z=g8["z"],G8_r=g8["r"],
        G7_increment=g7_run["_increment"],G7_velocity=g7_run["_velocity"],
        G7_source=g7_run["_source"],G7_memory=g7_run["_memory"],
        G8_increment=g8_run["_increment"],G8_velocity=g8_run["_velocity"],
        G8_source=g8_run["_source"],G8_memory=g8_run["_memory"],
    )
    payload={
        "status":"pass" if all(acceptance.values()) else "review",
        "scope":"fresh sealed G7/G8 spatial refinement of the short live nonlinear gauge-source evolution",
        "protocol":"notes/56_live_gauge_G8_refinement_protocol.md",
        "source_grids":{"G7":g7_geometry["source_grid"],"G8":g8_geometry["source_grid"]},
        "selector_maxima":{"G7":g7_geometry["selector_maximum"],"G8":g8_geometry["selector_maximum"]},
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
            "very short live-driver refinement",
            "G8 is a fresh nonlinear solve but uses interpolated G7 fields only as its initial guess",
            "driver and target parameters remain numerical controls",
            "frozen compact-normal wall gauge datum",
            "one-sided open-bulk artificial radial boundary",
            "no long-time stability, collapse, horizon, or mass-transfer claim",
        ],
    }
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":payload["status"],"summary":summary,"acceptance":acceptance},indent=2),flush=True)


if __name__=="__main__":main()
