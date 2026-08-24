#!/usr/bin/env python3
"""Unscored cosine-mode scan for the repaired G7 dynamical cap solver."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_capped_surface import solve_anisotropic_capped_profile
from bhps.dynamical_capped_horizon import solve_spectral_dynamical_capped_surface
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import relative_norm


OUTPUT=Path("results/corrected_fold_spectral_horizon_high_mode_engineering.json")
CHECKPOINT=Path("results/corrected_fold_short_evolved_horizon_persistence_state.npz")
MODES=(24,32,40,48)


def main():
    print("reconstructing final G7 A=7.94 slice",flush=True)
    fold=build_geometry("G6");seed={**fold,"fold_amplitude":7.94}
    g6=build_refined(seed,65,97,"G6A794",selector_iterations=35,slice_iterations=260)
    g7=build_refined(g6,81,121,"G7A794",selector_iterations=40,slice_iterations=270)
    archive=np.load(CHECKPOINT);q=g7["jet_field"].reduced_fields+archive["G7_increment"]
    v=archive["G7_velocity"]
    static=solve_anisotropic_capped_profile(
        g7["z"],g7["r"],g7["psi"],g7["a"],g7["b"],g7["c"],1.53,
        tolerance=1e-8,nodes=200,max_nodes=10000,
    )
    surfaces=[]
    for modes in MODES:
        print(f"solving with {modes} cosine modes",flush=True)
        surfaces.append(solve_spectral_dynamical_capped_surface(
            q,v,g7["z"],g7["r"],static,tolerance=5e-5,
            collocation_nodes=257,cosine_modes=modes,maximum_evaluations=160,
        ))
    comparisons=[]
    for coarse,fine in zip(surfaces[:-1],surfaces[1:]):
        coarse_on_fine=np.interp(fine["theta"],coarse["theta"],coarse["rho"])
        static_on_fine=np.interp(fine["theta"],static["theta"],static["rho"])
        comparisons.append({
            "modes":[coarse["cosine_modes"],fine["cosine_modes"]],
            "profile_relative_difference":relative_norm(coarse_on_fine,fine["rho"]),
            "displacement_relative_difference":relative_norm(
                coarse_on_fine-static_on_fine,fine["rho"]-static_on_fine,
            ),
            "axis_relative_difference":float(abs(coarse["rho_axis"]-fine["rho_axis"])/max(abs(coarse["rho_axis"]),abs(fine["rho_axis"]))),
            "brane_relative_difference":float(abs(coarse["rho_brane"]-fine["rho_brane"])/max(abs(coarse["rho_brane"]),abs(fine["rho_brane"]))),
        })
    payload={
        "status":"engineering_only",
        "scope":"unscored cosine-mode scan of the repaired spectral dynamical cap solver on the archived final G7 A=7.94 slice",
        "solves":[{
            "modes":item["cosine_modes"],"rho_axis":item["rho_axis"],"rho_brane":item["rho_brane"],
            "axis_change":item["rho_axis"]-static["rho_axis"],
            "brane_change":item["rho_brane"]-static["rho_brane"],
            "expansion_maximum":item["interior_expansion_maximum"],
            "jacobian_minimum_singular_value":item["minimum_jacobian_singular_value"],
            "jacobian_condition_number":item["jacobian_condition_number"],
            "function_evaluations":item["function_evaluations"],"converged":item["converged"],
        } for item in surfaces],
        "comparisons":comparisons,
        "limitations":["engineering only","single archived G7 final slice","no prospective acceptance rules"],
    }
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps(payload,indent=2),flush=True)


if __name__=="__main__":main()
