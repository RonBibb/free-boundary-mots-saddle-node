#!/usr/bin/env python3
"""First sealed live nonlinear gauge-source evolution on G6/G7."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.gh_source_driver import (
    regular_so3_live_source_shift_advection,
    regular_so3_nonlinear_anchored_damped_wave_target,
    source_driver_rhs,
)
from bhps.nonlinear_regular_so3_evolution import (
    NativeRegularSO3RHS,
    StageRegularGaugeSource,
    apply_outer_source_sommerfeld,
    compact_wall_normal_gauge_position_residuals,
    compact_wall_position_residuals,
    gauge_constraint_summary,
    gauge_taylor_source_from_initial_jets,
    live_regular_source_second_time,
    outer_sommerfeld_position_residuals,
    regular_source_spatial_derivatives,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_g7
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import (
    FINAL_TIME,RADIAL_COMPARISON_CUT,interpolate_fields,relative_norm,
    signature_summary,
)


OUTPUT=Path("results/corrected_fold_live_nonlinear_gauge_source.json")
CHECKPOINT=Path("results/corrected_fold_live_nonlinear_gauge_source_state.npz")
BASELINE=Path("results/corrected_fold_short_nonlinear_time_refinement_state.npz")
STEPS=4
DRIVER_MU=2.0
DRIVER_ETA=1.25
TARGET_MU_LAPSE=0.4
TARGET_MU_SHIFT=0.6
TARGET_POWER=0.5


def setup_case(
    geometry,label,live_normal_wall_gauge=False,live_outer_sommerfeld=False,
):
    z=np.asarray(geometry["z"],dtype=float)
    r=np.asarray(geometry["r"],dtype=float)
    jet=geometry["jet_field"]
    initial=np.asarray(jet.reduced_fields,dtype=float).copy()
    archived_acceleration=np.asarray(jet.reduced_second[0,0],dtype=float)
    print(f"{label}: constructing initial H=Gamma data",flush=True)
    taylor=gauge_taylor_source_from_initial_jets(jet,z,r)
    source=taylor.source_reduced.copy()
    source_time=taylor.source_time_reduced.copy()
    target=regular_so3_nonlinear_anchored_damped_wave_target(
        initial,initial,source,r,TARGET_MU_LAPSE,TARGET_MU_SHIFT,TARGET_POWER,
    )
    source_z,source_r=regular_source_spatial_derivatives(source,z,r)
    advection=regular_so3_live_source_shift_advection(
        initial,r,source,source_z,source_r,
    )
    memory=source_time-advection+DRIVER_MU*(source-target)
    source_dot,memory_dot=source_driver_rhs(
        source,memory,target,DRIVER_MU,DRIVER_ETA,advection,
    )
    normal=np.stack((archived_acceleration[0,:,6],archived_acceleration[-1,:,6]))
    rhs=NativeRegularSO3RHS(
        z,r,taylor,geometry["mass_squared"],geometry["background"],normal,
        live_normal_wall_gauge=live_normal_wall_gauge,
    )
    live_stage=StageRegularGaugeSource(source,source_dot,z,r)
    zero_velocity=np.zeros_like(initial)
    source_second=live_regular_source_second_time(
        initial,zero_velocity,initial,source,source,source_dot,memory_dot,
        z,r,DRIVER_MU,TARGET_MU_LAPSE,TARGET_MU_SHIFT,TARGET_POWER,
    )
    print(f"{label}: comparing live and Taylor zero-stage accelerations",flush=True)
    live_acceleration,live_diagnostic=rhs.acceleration(
        0.,initial,zero_velocity,live_stage,
        source_second if live_normal_wall_gauge else None,
    )
    if live_outer_sommerfeld:
        rhs.set_outer_sommerfeld_reference(initial,live_acceleration)
        live_acceleration,live_diagnostic=rhs.acceleration(
            0.,initial,zero_velocity,live_stage,
            source_second if live_normal_wall_gauge else None,
        )
    comparison_rhs=rhs
    if live_normal_wall_gauge:
        comparison_rhs=NativeRegularSO3RHS(
            z,r,taylor,geometry["mass_squared"],geometry["background"],normal,
        )
    taylor_acceleration,taylor_diagnostic=comparison_rhs.acceleration(
        0.,initial,zero_velocity,
    )
    initial_constraint=gauge_constraint_summary(
        initial,zero_velocity,0.,rhs,RADIAL_COMPARISON_CUT,live_stage,
    )
    return {
        "label":label,"geometry":geometry,"z":z,"r":r,"initial":initial,
        "rhs":rhs,"taylor":taylor,"source0":source,"memory0":memory,
        "source_time0":source_time,
        "initial_target_anchoring_relative":relative_norm(target,source),
        "initial_source_time_relative_error":relative_norm(source_dot,source_time),
        "initial_memory_dot_norm":float(np.linalg.norm(memory_dot)),
        "initial_live_Taylor_acceleration_relative_difference":relative_norm(
            live_acceleration,taylor_acceleration,
        ),
        "initial_live_finite":bool(live_diagnostic["finite"]),
        "initial_Taylor_finite":bool(taylor_diagnostic["finite"]),
        "initial_constraint":initial_constraint,
        "initial_live_normal_wall_gauge":live_diagnostic["normal_wall_gauge"],
        "initial_live_outer_sommerfeld":live_diagnostic["outer_sommerfeld"],
        "_initial_live_acceleration":live_acceleration,
        "_initial_source_second_time":source_second,
        "initial_wall":compact_wall_position_residuals(
            initial,z,r,geometry["background"],
        ),
    }


def driver_stage(case,time,position,velocity,source,memory):
    source_z,source_r=regular_source_spatial_derivatives(
        source,case["z"],case["r"],
    )
    target=regular_so3_nonlinear_anchored_damped_wave_target(
        position,case["initial"],case["source0"],case["r"],
        TARGET_MU_LAPSE,TARGET_MU_SHIFT,TARGET_POWER,
    )
    advection=regular_so3_live_source_shift_advection(
        position,case["r"],source,source_z,source_r,
    )
    source_dot,memory_dot=source_driver_rhs(
        source,memory,target,DRIVER_MU,DRIVER_ETA,advection,
    )
    outer_source=None
    if case["rhs"].live_outer_sommerfeld:
        source_dot,outer_source=apply_outer_source_sommerfeld(
            source,source_dot,case["source0"],case["source_time0"],
            case["_initial_source_second_time"],position,time,case["r"],
            case["rhs"].stencil_width,
        )
    gauge=StageRegularGaugeSource(
        source,source_dot,case["z"],case["r"],
    )
    source_second=live_regular_source_second_time(
        position,velocity,case["initial"],case["source0"],source,source_dot,
        memory_dot,case["z"],case["r"],DRIVER_MU,TARGET_MU_LAPSE,
        TARGET_MU_SHIFT,TARGET_POWER,
    )
    acceleration,diagnostic=case["rhs"].acceleration(
        time,position,velocity,gauge,
        source_second if case["rhs"].live_normal_wall_gauge else None,
    )
    return (
        velocity,acceleration,source_dot,memory_dot,
    ),{
        "finite":bool(
            diagnostic["finite"] and np.all(np.isfinite(source_dot))
            and np.all(np.isfinite(memory_dot)) and np.all(np.isfinite(target))
        ),
        "target":target,"advection":advection,
        "source_second_time":source_second,
        "wall_corrections":diagnostic["wall_corrections"],
        "normal_wall_gauge":diagnostic["normal_wall_gauge"],
        "outer_sommerfeld":diagnostic["outer_sommerfeld"],
        "outer_source_sommerfeld":outer_source,
    }


def integrate(case,checkpoint_steps=()):
    dt=FINAL_TIME/STEPS
    checkpoint_steps={int(value) for value in checkpoint_steps}
    if any(value<1 or value>STEPS for value in checkpoint_steps):
        raise ValueError("checkpoint steps must lie within the integration")
    checkpoints={}
    state=(
        case["initial"].copy(),np.zeros_like(case["initial"]),
        case["source0"].copy(),case["memory0"].copy(),
    )
    time=0.;all_finite=True;maximum_stage_change=0.
    maximum_normal_wall_acceleration_residual=0.
    maximum_outer_acceleration_residual=0.
    maximum_outer_source_residual=0.
    maximum_outer_metric_correction=0.
    maximum_outer_scalar_correction=0.
    maximum_outer_source_correction=0.
    for step in range(STEPS):
        print(f"{case['label']}: live step {step+1}/{STEPS}, stage 1",flush=True)
        k1,d1=driver_stage(case,time,*state)
        midpoint=tuple(value+.5*dt*slope for value,slope in zip(state,k1))
        print(f"{case['label']}: live step {step+1}/{STEPS}, stage 2",flush=True)
        k2,d2=driver_stage(case,time+.5*dt,*midpoint)
        all_finite=bool(all_finite and d1["finite"] and d2["finite"])
        for diagnostic in (d1,d2):
            normal=diagnostic["normal_wall_gauge"]
            if normal is not None:
                maximum_normal_wall_acceleration_residual=max(
                    maximum_normal_wall_acceleration_residual,
                    normal["final_residual"]["maximum"],
                )
            outer=diagnostic["outer_sommerfeld"]
            if outer is not None:
                maximum_outer_acceleration_residual=max(
                    maximum_outer_acceleration_residual,
                    outer["maximum_normalized_acceleration_residual"],
                )
                maximum_outer_metric_correction=max(
                    maximum_outer_metric_correction,
                    outer["metric_relative_correction"],
                )
                maximum_outer_scalar_correction=max(
                    maximum_outer_scalar_correction,
                    outer["scalar_relative_correction"],
                )
            outer_source=diagnostic["outer_source_sommerfeld"]
            if outer_source is not None:
                maximum_outer_source_residual=max(
                    maximum_outer_source_residual,
                    outer_source["maximum_normalized"],
                )
                maximum_outer_source_correction=max(
                    maximum_outer_source_correction,
                    outer_source["relative_correction"],
                )
        maximum_stage_change=max(
            maximum_stage_change,relative_norm(k1[1],k2[1]),
        )
        state=tuple(value+dt*slope for value,slope in zip(state,k2))
        time+=dt
        if step+1 in checkpoint_steps:
            checkpoints[step+1]={
                "time":time,
                "_position":state[0].copy(),
                "_increment":state[0]-case["initial"],
                "_velocity":state[1].copy(),
                "_source_increment":state[2]-case["source0"],
            }
    position,velocity,source,memory=state
    source_z,source_r=regular_source_spatial_derivatives(
        source,case["z"],case["r"],
    )
    target=regular_so3_nonlinear_anchored_damped_wave_target(
        position,case["initial"],case["source0"],case["r"],
        TARGET_MU_LAPSE,TARGET_MU_SHIFT,TARGET_POWER,
    )
    advection=regular_so3_live_source_shift_advection(
        position,case["r"],source,source_z,source_r,
    )
    source_dot,memory_dot=source_driver_rhs(
        source,memory,target,DRIVER_MU,DRIVER_ETA,advection,
    )
    final_outer_source=None
    if case["rhs"].live_outer_sommerfeld:
        source_dot,final_outer_source=apply_outer_source_sommerfeld(
            source,source_dot,case["source0"],case["source_time0"],
            case["_initial_source_second_time"],position,time,case["r"],
            case["rhs"].stencil_width,
        )
    final_gauge=StageRegularGaugeSource(
        source,source_dot,case["z"],case["r"],
    )
    print(f"{case['label']}: live final diagnostics",flush=True)
    constraint=gauge_constraint_summary(
        position,velocity,time,case["rhs"],RADIAL_COMPARISON_CUT,final_gauge,
    )
    wall=compact_wall_position_residuals(
        position,case["z"],case["r"],case["geometry"]["background"],
    )
    return {
        "final_time":time,"steps":STEPS,"time_step":dt,
        "all_stages_finite":all_finite,
        "maximum_stage_acceleration_relative_change":maximum_stage_change,
        "maximum_normal_wall_acceleration_residual":maximum_normal_wall_acceleration_residual,
        "maximum_outer_acceleration_residual":maximum_outer_acceleration_residual,
        "maximum_outer_source_residual":maximum_outer_source_residual,
        "maximum_outer_metric_correction":maximum_outer_metric_correction,
        "maximum_outer_scalar_correction":maximum_outer_scalar_correction,
        "maximum_outer_source_correction":maximum_outer_source_correction,
        "final_constraint":constraint,"final_wall":wall,
        "signature":signature_summary(position,case["r"]),
        "final_source_target_relative_difference":relative_norm(source,target),
        "final_source_increment_norm":float(np.linalg.norm(source-case["source0"])),
        "final_memory_increment_norm":float(np.linalg.norm(memory-case["memory0"])),
        "final_source_dot_norm":float(np.linalg.norm(source_dot)),
        "final_memory_dot_norm":float(np.linalg.norm(memory_dot)),
        "final_normal_wall_position_residual":compact_wall_normal_gauge_position_residuals(
            position,source,case["z"],case["r"],case["geometry"]["background"],
        ),
        "final_outer_sommerfeld_position_residual":(
            outer_sommerfeld_position_residuals(
                position,velocity,case["rhs"].outer_reference_position,
                case["rhs"].outer_reference_acceleration,time,case["r"],
                case["rhs"].stencil_width,
            ) if case["rhs"].live_outer_sommerfeld else None
        ),
        "final_outer_source_sommerfeld_residual":final_outer_source,
        "_position":position,"_increment":position-case["initial"],
        "_velocity":velocity,"_source":source,
        "_source_increment":source-case["source0"],"_memory":memory,
        "_checkpoints":checkpoints,
    }


def common_grid_vectors(coarse_case,coarse,fine_case,fine,key):
    mask=coarse_case["r"]<=RADIAL_COMPARISON_CUT+1e-12
    target_r=coarse_case["r"][mask]
    coarse_values=coarse[key][:,mask]
    fine_values=interpolate_fields(
        fine[key],fine_case["z"],fine_case["r"],coarse_case["z"],target_r,
    )
    return coarse_values,fine_values


def public_run(run):
    return {key:value for key,value in run.items() if not key.startswith("_")}


def public_case(case):
    return {
        key:value for key,value in case.items()
        if key in (
            "label","initial_target_anchoring_relative",
            "initial_source_time_relative_error","initial_memory_dot_norm",
            "initial_live_Taylor_acceleration_relative_difference",
            "initial_live_finite","initial_Taylor_finite","initial_constraint",
            "initial_wall","initial_live_normal_wall_gauge",
            "initial_live_outer_sommerfeld",
        )
    }


def main():
    if not BASELINE.exists():
        raise FileNotFoundError("sealed Taylor-source checkpoint is required")
    baseline=np.load(BASELINE)
    print("building G6 corrected-fold state",flush=True)
    g6_geometry=build_geometry("G6")
    print("building G7 corrected-fold state",flush=True)
    g7_geometry=build_g7(g6_geometry)
    g6=setup_case(g6_geometry,"G6");g7=setup_case(g7_geometry,"G7")
    g6_run=integrate(g6);g7_run=integrate(g7)

    grid={}
    for name,key in (
        ("position_increment","_increment"),("velocity","_velocity"),
        ("source_increment","_source_increment"),
    ):
        coarse,fine=common_grid_vectors(g6,g6_run,g7,g7_run,key)
        grid[f"{name}_relative_difference"]=relative_norm(coarse,fine)
    baseline_comparison={}
    for case,run,label in ((g6,g6_run,"G6"),(g7,g7_run,"G7")):
        mask=case["r"]<=RADIAL_COMPARISON_CUT+1e-12
        baseline_increment=baseline[f"{label}_steps_4_increment"][:,mask]
        baseline_velocity=baseline[f"{label}_steps_4_velocity"][:,mask]
        baseline_comparison[label]={
            "position_increment_relative_difference":relative_norm(
                run["_increment"][:,mask],baseline_increment,
            ),
            "velocity_relative_difference":relative_norm(
                run["_velocity"][:,mask],baseline_velocity,
            ),
        }
    cases=(g6,g7);runs=(g6_run,g7_run)
    acceptance={
        "initial_target_anchoring_below_1e_10":bool(max(
            case["initial_target_anchoring_relative"] for case in cases
        )<1e-10),
        "initial_source_time_error_below_1e_10":bool(max(
            case["initial_source_time_relative_error"] for case in cases
        )<1e-10),
        "initial_live_Taylor_accelerations_within_5_percent":bool(max(
            case["initial_live_Taylor_acceleration_relative_difference"] for case in cases
        )<.05),
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
        "G6_G7_live_position_transfer_below_5_percent":bool(
            grid["position_increment_relative_difference"]<.05
        ),
        "G6_G7_live_velocity_transfer_below_5_percent":bool(
            grid["velocity_relative_difference"]<.05
        ),
        "G6_G7_live_source_increment_transfer_below_5_percent":bool(
            grid["source_increment_relative_difference"]<.05
        ),
        "live_Taylor_position_and_velocity_within_5_percent":bool(all(
            value<.05 for record in baseline_comparison.values() for value in record.values()
        )),
    }
    summary={
        "grid_transfer":grid,
        "Taylor_source_baseline_comparison":baseline_comparison,
        "final_global_GH_constraint_by_resolution":{
            case["label"]:run["final_constraint"]["global_relative"]
            for case,run in zip(cases,runs)
        },
        "final_wall_residual_by_resolution":{
            case["label"]:run["final_wall"]["maximum"]
            for case,run in zip(cases,runs)
        },
        "initial_live_Taylor_acceleration_difference_by_resolution":{
            case["label"]:case["initial_live_Taylor_acceleration_relative_difference"]
            for case in cases
        },
    }
    np.savez_compressed(
        CHECKPOINT,G6_z=g6["z"],G6_r=g6["r"],G7_z=g7["z"],G7_r=g7["r"],
        G6_increment=g6_run["_increment"],G6_velocity=g6_run["_velocity"],
        G6_source=g6_run["_source"],G6_memory=g6_run["_memory"],
        G7_increment=g7_run["_increment"],G7_velocity=g7_run["_velocity"],
        G7_source=g7_run["_source"],G7_memory=g7_run["_memory"],
    )
    payload={
        "status":"pass" if all(acceptance.values()) else "review",
        "scope":"first sealed live nonlinear regular gauge-source evolution on corrected G6/G7 folds",
        "protocol":"notes/55_live_nonlinear_gauge_source_protocol.md",
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
            "very short live-driver test",
            "driver and target parameters are numerical controls rather than a production selection",
            "frozen compact-normal wall acceleration remains a gauge datum",
            "one-sided open-bulk artificial radial boundary",
            "no long-time constraint, stability, collapse, horizon, or mass-transfer claim",
        ],
    }
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":payload["status"],"summary":summary,"acceptance":acceptance},indent=2),flush=True)


if __name__=="__main__":main()
