#!/usr/bin/env python3
"""Unscored t=0 control of the repaired spectral dynamic cap solver."""

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


OUTPUT=Path("results/corrected_fold_spectral_horizon_t0_engineering.json")
MODES=(24,32,40,48)


def main():
    print("building corrected G7 A=7.94 initial slice",flush=True)
    fold=build_geometry("G6");seed={**fold,"fold_amplitude":7.94}
    g6=build_refined(seed,65,97,"G6A794",selector_iterations=35,slice_iterations=260)
    g7=build_refined(g6,81,121,"G7A794",selector_iterations=40,slice_iterations=270)
    static=solve_anisotropic_capped_profile(
        g7["z"],g7["r"],g7["psi"],g7["a"],g7["b"],g7["c"],1.53,
        tolerance=1e-8,nodes=200,max_nodes=10000,
    )
    q=np.asarray(g7["jet_field"].reduced_fields);v=np.zeros_like(q);surfaces=[]
    for modes in MODES:
        print(f"solving t=0 cap with {modes} cosine modes",flush=True)
        surfaces.append(solve_spectral_dynamical_capped_surface(
            q,v,g7["z"],g7["r"],static,tolerance=5e-4,
            collocation_nodes=257,cosine_modes=modes,maximum_evaluations=160,
        ))
    records=[]
    for surface in surfaces:
        static_on=np.interp(surface["theta"],static["theta"],static["rho"])
        records.append({
            "modes":surface["cosine_modes"],"converged":surface["converged"],
            "profile_relative_difference_from_static":relative_norm(surface["rho"],static_on),
            "rho_axis_relative_difference_from_static":float(abs(surface["rho_axis"]-static["rho_axis"])/static["rho_axis"]),
            "rho_brane_relative_difference_from_static":float(abs(surface["rho_brane"]-static["rho_brane"])/static["rho_brane"]),
            "expansion_maximum":surface["interior_expansion_maximum"],
            "jacobian_condition_number":surface["jacobian_condition_number"],
        })
    payload={
        "status":"engineering_only","scope":"unscored t=0 recovery of the known G7 A=7.94 static outer cap by the repaired spectral dynamic solver",
        "records":records,
        "limitations":["engineering only","G7 only","no prospective acceptance rules"],
    }
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps(payload,indent=2),flush=True)


if __name__=="__main__":main()
