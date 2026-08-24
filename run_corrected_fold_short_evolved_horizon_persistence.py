#!/usr/bin/env python3
"""Sealed G6/G7 short persistence and motion of the A=7.94 outer cap."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_capped_surface import solve_anisotropic_capped_profile
from bhps.dynamical_capped_horizon import solve_dynamical_capped_surface
from run_corrected_fold_live_nonlinear_gauge_source import (
    DRIVER_ETA,DRIVER_MU,STEPS,TARGET_MU_LAPSE,TARGET_MU_SHIFT,TARGET_POWER,
    common_grid_vectors,integrate,public_case,public_run,setup_case,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import (
    FINAL_TIME,relative_norm,
)


OUTPUT=Path("results/corrected_fold_short_evolved_horizon_persistence.json")
CHECKPOINT=Path("results/corrected_fold_short_evolved_horizon_persistence_state.npz")


def static_cap(geometry):
    result=solve_anisotropic_capped_profile(
        geometry["z"],geometry["r"],geometry["psi"],geometry["a"],geometry["b"],
        geometry["c"],1.53,tolerance=1e-8,nodes=200,max_nodes=10000,
    )
    if not result["converged"]:raise RuntimeError("initial outer cap did not converge")
    return result


def horizon_summary(surface):
    return {
        key:surface[key] for key in (
            "converged","optimizer_success","in_domain","function_evaluations",
            "rho_axis","rho_brane","boundary_slope_error",
            "regularized_expansion_maximum","regularized_expansion_l2",
            "raw_two_cell_interior_maximum","minimum_expansion","maximum_expansion",
        )
    }


def main():
    print("building corrected G6/G7 A=7.94 states",flush=True)
    fold=build_geometry("G6");seed={**fold,"fold_amplitude":7.94}
    g6_geometry=build_refined(seed,65,97,"G6A794",selector_iterations=35,slice_iterations=260)
    g7_geometry=build_refined(g6_geometry,81,121,"G7A794",selector_iterations=40,slice_iterations=270)
    geometries=(g6_geometry,g7_geometry);labels=("G6-horizon","G7-horizon")
    static=(static_cap(g6_geometry),static_cap(g7_geometry))
    cases=tuple(
        setup_case(
            geometry,label,live_normal_wall_gauge=True,live_outer_sommerfeld=True,
        ) for geometry,label in zip(geometries,labels)
    )
    runs=tuple(integrate(case) for case in cases)
    final=[]
    for case,run,initial in zip(cases,runs,static):
        print(f"{case['label']}: solving final marginal cap",flush=True)
        final.append(solve_dynamical_capped_surface(
            run["_position"],run["_velocity"],case["z"],case["r"],initial,
            tolerance=5e-5,nodes=81,maximum_evaluations=180,
        ))
    final=tuple(final)
    grid={}
    for name,key in (
        ("position_increment","_increment"),("velocity","_velocity"),
        ("source_increment","_source_increment"),
    ):
        coarse,fine=common_grid_vectors(cases[0],runs[0],cases[1],runs[1],key)
        grid[f"{name}_relative_difference"]=relative_norm(coarse,fine)
    final_profile_transfer=relative_norm(final[0]["rho"],final[1]["rho"])
    initial_on_final=tuple(
        np.interp(surface["theta"],initial["theta"],initial["rho"])
        for surface,initial in zip(final,static)
    )
    displacement=tuple(
        surface["rho"]-initial_values
        for surface,initial_values in zip(final,initial_on_final)
    )
    displacement_transfer=relative_norm(*displacement)
    radius_transfer={
        name:float(abs(final[0][name]-final[1][name])/max(abs(final[0][name]),abs(final[1][name])))
        for name in ("rho_axis","rho_brane")
    }
    fractional_changes={
        label:{
            name:float((surface[name]-initial[name])/initial[name])
            for name in ("rho_axis","rho_brane")
        } for label,surface,initial in zip(labels,final,static)
    }
    acceptance={
        "initial_static_caps_converge_below_1e_6":bool(all(
            item["surface_residual_max"]<1e-6 and item["boundary_slope_error"]<1e-8
            for item in static
        )),
        "final_dynamic_caps_converge_below_1e_8":bool(all(
            item["converged"] and item["regularized_expansion_maximum"]<1e-8
            and item["boundary_slope_error"]<1e-8 and item["function_evaluations"]<50
            for item in final
        )),
        "all_stages_finite":bool(all(run["all_stages_finite"] for run in runs)),
        "Lorentzian_signature_preserved":bool(all(
            run["signature"]["all_points_one_negative_direction"] for run in runs
        )),
        "final_global_GH_constraints_below_0_5_percent":bool(max(
            run["final_constraint"]["global_relative"] for run in runs
        )<.005),
        "compact_wall_rows_below_0_05_percent":bool(max(
            max(run["final_wall"]["maximum"],run["final_normal_wall_position_residual"]["maximum"])
            for run in runs
        )<.0005),
        "live_wall_and_outer_acceleration_rows_below_1e_10":bool(max(
            max(run["maximum_normal_wall_acceleration_residual"],run["maximum_outer_acceleration_residual"])
            for run in runs
        )<1e-10),
        "G6_G7_position_velocity_source_transfer_below_5_percent":bool(max(grid.values())<.05),
        "final_horizon_radii_transfer_below_0_2_percent":bool(max(radius_transfer.values())<.002),
        "final_horizon_profile_transfer_below_0_2_percent":bool(final_profile_transfer<.002),
        "horizon_displacement_transfer_below_10_percent":bool(displacement_transfer<.10),
        "horizon_motion_exceeds_0_1_percent":bool(max(
            abs(value) for record in fractional_changes.values() for value in record.values()
        )>.001),
    }
    summary={
        "evolution_grid_transfer":grid,"final_horizon_radius_transfer":radius_transfer,
        "final_horizon_profile_transfer":final_profile_transfer,
        "horizon_displacement_transfer":displacement_transfer,
        "fractional_radius_changes":fractional_changes,
        "final_horizons":{
            label:horizon_summary(surface) for label,surface in zip(labels,final)
        },
        "final_global_GH_constraints":{
            label:run["final_constraint"]["global_relative"]
            for label,run in zip(labels,runs)
        },
    }
    np.savez_compressed(
        CHECKPOINT,
        G6_z=cases[0]["z"],G6_r=cases[0]["r"],G7_z=cases[1]["z"],G7_r=cases[1]["r"],
        G6_increment=runs[0]["_increment"],G6_velocity=runs[0]["_velocity"],
        G7_increment=runs[1]["_increment"],G7_velocity=runs[1]["_velocity"],
        horizon_theta=final[0]["theta"],G6_initial_horizon=initial_on_final[0],
        G7_initial_horizon=initial_on_final[1],G6_final_horizon=final[0]["rho"],
        G7_final_horizon=final[1]["rho"],
    )
    payload={
        "status":"pass" if all(acceptance.values()) else "review",
        "scope":"sealed G6/G7 short live nonlinear persistence and motion of the pre-existing A=7.94 outer donor cap",
        "protocol":"notes/60_short_evolved_horizon_persistence_protocol.md",
        "driver_parameters":{"mu":DRIVER_MU,"eta":DRIVER_ETA},
        "target_parameters":{"mu_lapse":TARGET_MU_LAPSE,"mu_shift":TARGET_MU_SHIFT,"determinant_power":TARGET_POWER},
        "final_time":FINAL_TIME,"steps":STEPS,
        "cases":[
            {"initial":public_case(case),"run":public_run(run),
             "initial_static_horizon":{
                 key:initial[key] for key in ("rho_axis","rho_brane","surface_residual_max","boundary_slope_error")
             },"final_dynamic_horizon":horizon_summary(surface)}
            for case,run,initial,surface in zip(cases,runs,static,final)
        ],
        "summary":summary,"acceptance":acceptance,
        "limitations":[
            "t=0.002 persistence test of a pre-existing cap",
            "not horizon formation, an event horizon, topology change, long-time stability, branch selection, or mass transfer",
        ],
    }
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":payload["status"],"summary":summary,"acceptance":acceptance},indent=2),flush=True)


if __name__=="__main__":main()
