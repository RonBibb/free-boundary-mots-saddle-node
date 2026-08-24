#!/usr/bin/env python3
"""Sealed G7/G8 short evolved-horizon refinement with the spectral tracker."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_capped_surface import solve_anisotropic_capped_profile
from bhps.dynamical_capped_horizon import solve_spectral_dynamical_capped_surface
from run_corrected_fold_live_nonlinear_gauge_source import (
    DRIVER_ETA,DRIVER_MU,STEPS,TARGET_MU_LAPSE,TARGET_MU_SHIFT,TARGET_POWER,
    common_grid_vectors,integrate,public_case,public_run,setup_case,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import FINAL_TIME,relative_norm


OUTPUT=Path("results/corrected_fold_G7_G8_spectral_horizon_refinement.json")
CHECKPOINT=Path("results/corrected_fold_G7_G8_spectral_horizon_refinement_state.npz")
MODES=(40,48);COLLOCATION=257


def static_cap(geometry):
    result=solve_anisotropic_capped_profile(
        geometry["z"],geometry["r"],geometry["psi"],geometry["a"],geometry["b"],
        geometry["c"],1.53,tolerance=1e-8,nodes=220,max_nodes=12000,
    )
    if not result["converged"]:raise RuntimeError("initial outer cap failed")
    return result


def spectral_set(case,run,initial):
    return [solve_spectral_dynamical_capped_surface(
        run["_position"],run["_velocity"],case["z"],case["r"],initial,
        tolerance=5e-4,collocation_nodes=COLLOCATION,cosine_modes=modes,
        maximum_evaluations=200,
    ) for modes in MODES]


def relative_scalar(left,right):
    return float(abs(left-right)/max(abs(left),abs(right),1e-300))


def main():
    print("building corrected G7/G8 A=7.94 states",flush=True)
    fold=build_geometry("G6");seed={**fold,"fold_amplitude":7.94}
    g7_geometry=build_refined(seed,81,121,"G7A794",selector_iterations=40,slice_iterations=270)
    g8_geometry=build_refined(g7_geometry,97,145,"G8A794",selector_iterations=45,slice_iterations=280)
    geometries=(g7_geometry,g8_geometry);labels=("G7-spectral","G8-spectral")
    static=tuple(static_cap(item) for item in geometries)
    cases=tuple(setup_case(
        geometry,label,live_normal_wall_gauge=True,live_outer_sommerfeld=True,
    ) for geometry,label in zip(geometries,labels))
    runs=tuple(integrate(case) for case in cases)
    surfaces=[]
    for case,run,initial in zip(cases,runs,static):
        print(f"{case['label']}: solving 40/48-mode final caps",flush=True)
        surfaces.append(spectral_set(case,run,initial))
    grid={}
    for name,key in (("position_increment","_increment"),("velocity","_velocity"),("source_increment","_source_increment")):
        coarse,fine=common_grid_vectors(cases[0],runs[0],cases[1],runs[1],key)
        grid[f"{name}_relative_difference"]=relative_norm(coarse,fine)
    within={}
    displacement=[]
    for label,pair,initial in zip(labels,surfaces,static):
        coarse,fine=pair
        initial_on=np.interp(fine["theta"],initial["theta"],initial["rho"])
        within[label]={
            "profile_relative_difference":relative_norm(coarse["rho"],fine["rho"]),
            "axis_relative_difference":relative_scalar(coarse["rho_axis"],fine["rho_axis"]),
            "brane_relative_difference":relative_scalar(coarse["rho_brane"],fine["rho_brane"]),
            "displacement_relative_difference":relative_norm(coarse["rho"]-initial_on,fine["rho"]-initial_on),
        }
        displacement.append(fine["rho"]-initial_on)
    selected=(surfaces[0][-1],surfaces[1][-1])
    selected_profile_transfer=relative_norm(selected[0]["rho"],selected[1]["rho"])
    selected_radius_transfer={
        name:relative_scalar(selected[0][name],selected[1][name])
        for name in ("rho_axis","rho_brane")
    }
    selected_displacement_transfer=relative_norm(*displacement)
    changes={
        label:{
            name:float((surface[name]-initial[name])/initial[name])
            for name in ("rho_axis","rho_brane")
        } for label,surface,initial in zip(labels,selected,static)
    }
    static_radius_transfer={
        name:relative_scalar(static[0][name],static[1][name])
        for name in ("rho_axis","rho_brane")
    }
    acceptance={
        "initial_static_caps_converge_and_transfer_below_0_2_percent":bool(
            max(static_radius_transfer.values())<.002 and all(
                item["surface_residual_max"]<1e-6 and item["boundary_slope_error"]<1e-8 for item in static
            )
        ),
        "all_spectral_surfaces_converge_below_5e_4":bool(all(
            surface["converged"] and surface["in_domain"] and surface["interior_expansion_maximum"]<.0005
            for pair in surfaces for surface in pair
        )),
        "within_grid_angular_transfer_passes":bool(max(
            max(record["profile_relative_difference"],record["axis_relative_difference"],record["brane_relative_difference"])
            for record in within.values()
        )<.0005 and max(record["displacement_relative_difference"] for record in within.values())<.01),
        "all_stages_finite_and_Lorentzian":bool(all(
            run["all_stages_finite"] and run["signature"]["all_points_one_negative_direction"] for run in runs
        )),
        "constraint_wall_and_boundary_rows_pass":bool(
            max(run["final_constraint"]["global_relative"] for run in runs)<.005
            and max(max(run["final_wall"]["maximum"],run["final_normal_wall_position_residual"]["maximum"]) for run in runs)<.0005
            and max(max(run["maximum_normal_wall_acceleration_residual"],run["maximum_outer_acceleration_residual"]) for run in runs)<1e-10
        ),
        "G7_G8_position_velocity_source_transfer_below_5_percent":bool(max(grid.values())<.05),
        "selected_horizon_profile_and_radii_transfer_below_0_2_percent":bool(
            max(selected_profile_transfer,*selected_radius_transfer.values())<.002
        ),
        "selected_horizon_displacement_transfer_below_10_percent":bool(selected_displacement_transfer<.10),
        "selected_motion_directions_agree_and_are_nonzero":bool(
            all(record["rho_axis"]>0 and record["rho_brane"]>0 for record in changes.values())
            and max(abs(value) for record in changes.values() for value in record.values())>.001
        ),
    }
    summary={
        "evolution_grid_transfer":grid,"static_horizon_radius_transfer":static_radius_transfer,
        "within_grid_angular_transfer":within,
        "selected_G7_G8_profile_transfer":selected_profile_transfer,
        "selected_G7_G8_radius_transfer":selected_radius_transfer,
        "selected_G7_G8_displacement_transfer":selected_displacement_transfer,
        "selected_fractional_radius_changes":changes,
        "selected_expansion_residuals":{
            label:surface["interior_expansion_maximum"] for label,surface in zip(labels,selected)
        },
        "final_global_GH_constraints":{
            label:run["final_constraint"]["global_relative"] for label,run in zip(labels,runs)
        },
    }
    np.savez_compressed(
        CHECKPOINT,G7_z=cases[0]["z"],G7_r=cases[0]["r"],G8_z=cases[1]["z"],G8_r=cases[1]["r"],
        G7_increment=runs[0]["_increment"],G7_velocity=runs[0]["_velocity"],
        G8_increment=runs[1]["_increment"],G8_velocity=runs[1]["_velocity"],
        horizon_theta=selected[0]["theta"],G7_final_horizon=selected[0]["rho"],G8_final_horizon=selected[1]["rho"],
        G7_initial_horizon=np.interp(selected[0]["theta"],static[0]["theta"],static[0]["rho"]),
        G8_initial_horizon=np.interp(selected[1]["theta"],static[1]["theta"],static[1]["rho"]),
    )
    payload={
        "status":"pass" if all(acceptance.values()) else "review",
        "scope":"sealed G7/G8 short live nonlinear refinement of the persistent A=7.94 outer cap with the smooth spectral tracker",
        "protocol":"notes/63_G7_G8_spectral_horizon_refinement_protocol.md",
        "driver_parameters":{"mu":DRIVER_MU,"eta":DRIVER_ETA},
        "target_parameters":{"mu_lapse":TARGET_MU_LAPSE,"mu_shift":TARGET_MU_SHIFT,"determinant_power":TARGET_POWER},
        "final_time":FINAL_TIME,"steps":STEPS,"summary":summary,"acceptance":acceptance,
        "limitations":[
            "t=0.002 persistence of a pre-existing cap","note-60 G6/G7 review is not rescored",
            "not formation, event-horizon location, topology change, long-time stability, branch selection, or mass transfer",
        ],
    }
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps(payload,indent=2),flush=True)


if __name__=="__main__":main()
