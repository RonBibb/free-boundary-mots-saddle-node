#!/usr/bin/env python3
"""Unscored G6 t=0 engineering test of the independent dynamic MOTS solve."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_capped_surface import solve_anisotropic_capped_profile
from bhps.dynamical_capped_horizon import solve_dynamical_capped_surface
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry


OUTPUT=Path("results/corrected_fold_dynamical_surface_solver_engineering.json")


def main():
    print("building corrected G6 A=7.94 state",flush=True)
    fold=build_geometry("G6");seed={**fold,"fold_amplitude":7.94}
    geometry=build_refined(seed,65,97,"G6A794",selector_iterations=35,slice_iterations=260)
    static=solve_anisotropic_capped_profile(
        geometry["z"],geometry["r"],geometry["psi"],geometry["a"],geometry["b"],
        geometry["c"],1.53,tolerance=1e-8,nodes=200,max_nodes=10000,
    )
    if not static["converged"]:raise RuntimeError("static outer cap failed")
    q=np.asarray(geometry["jet_field"].reduced_fields,dtype=float)
    print("solving full dynamical theta-plus equation",flush=True)
    dynamic=solve_dynamical_capped_surface(
        q,np.zeros_like(q),geometry["z"],geometry["r"],static,
        tolerance=5e-4,nodes=61,maximum_evaluations=120,
    )
    summary={
        "static_rho_axis":static["rho_axis"],"static_rho_brane":static["rho_brane"],
        "dynamic_rho_axis":dynamic["rho_axis"],"dynamic_rho_brane":dynamic["rho_brane"],
        "axis_relative_difference":float(abs(dynamic["rho_axis"]-static["rho_axis"])/static["rho_axis"]),
        "brane_relative_difference":float(abs(dynamic["rho_brane"]-static["rho_brane"])/static["rho_brane"]),
        "dynamic_regularized_expansion_maximum":dynamic["regularized_expansion_maximum"],
        "dynamic_boundary_slope_error":dynamic["boundary_slope_error"],
        "function_evaluations":dynamic["function_evaluations"],
        "dynamic_converged":dynamic["converged"],
    }
    payload={
        "status":"engineering_only","scope":"unscored independent nodal solve of full theta-plus=0 on the corrected G6 A=7.94 time-symmetric slice",
        "summary":summary,
        "limitations":["G6 only","t=0 only","no prospective acceptance rules"],
    }
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps(payload,indent=2),flush=True)


if __name__=="__main__":main()
