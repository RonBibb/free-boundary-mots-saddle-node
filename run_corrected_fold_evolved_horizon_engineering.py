#!/usr/bin/env python3
"""First unscored marginal-cap solve on a fully evolved corrected slice."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_capped_surface import solve_anisotropic_capped_profile
from bhps.dynamical_capped_horizon import solve_dynamical_capped_surface
from run_corrected_fold_live_nonlinear_gauge_source import integrate,setup_case
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry


OUTPUT=Path("results/corrected_fold_evolved_horizon_engineering.json")


def main():
    print("building corrected G6 A=7.94 state",flush=True)
    fold=build_geometry("G6");seed={**fold,"fold_amplitude":7.94}
    geometry=build_refined(seed,65,97,"G6A794",selector_iterations=35,slice_iterations=260)
    static=solve_anisotropic_capped_profile(
        geometry["z"],geometry["r"],geometry["psi"],geometry["a"],geometry["b"],
        geometry["c"],1.53,tolerance=1e-8,nodes=200,max_nodes=10000,
    )
    if not static["converged"]:raise RuntimeError("static outer cap failed")
    case=setup_case(
        geometry,"G6-A794-horizon",live_normal_wall_gauge=True,
        live_outer_sommerfeld=True,
    )
    run=integrate(case)
    print("solving final evolved marginal cap",flush=True)
    final=solve_dynamical_capped_surface(
        run["_position"],run["_velocity"],case["z"],case["r"],static,
        tolerance=5e-4,nodes=61,maximum_evaluations=160,
    )
    summary={
        "final_time":run["final_time"],
        "initial_static_rho_axis":static["rho_axis"],
        "initial_static_rho_brane":static["rho_brane"],
        "final_dynamic_rho_axis":final["rho_axis"],
        "final_dynamic_rho_brane":final["rho_brane"],
        "axis_fractional_change":float((final["rho_axis"]-static["rho_axis"])/static["rho_axis"]),
        "brane_fractional_change":float((final["rho_brane"]-static["rho_brane"])/static["rho_brane"]),
        "final_regularized_expansion_maximum":final["regularized_expansion_maximum"],
        "final_boundary_slope_error":final["boundary_slope_error"],
        "final_surface_function_evaluations":final["function_evaluations"],
        "final_surface_converged":final["converged"],
        "final_global_GH_constraint":run["final_constraint"]["global_relative"],
        "Lorentzian_signature_preserved":run["signature"]["all_points_one_negative_direction"],
        "maximum_normal_wall_acceleration_residual":run["maximum_normal_wall_acceleration_residual"],
        "maximum_outer_acceleration_residual":run["maximum_outer_acceleration_residual"],
    }
    payload={
        "status":"engineering_only",
        "scope":"first unscored full theta-plus marginal-cap solve after live nonlinear G6 A=7.94 evolution to t=0.002",
        "summary":summary,
        "limitations":[
            "G6 only","single short final time","no prospective acceptance rules",
            "tracks persistence of a pre-existing outer marginal cap rather than horizon formation",
        ],
    }
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps(payload,indent=2),flush=True)


if __name__=="__main__":main()
