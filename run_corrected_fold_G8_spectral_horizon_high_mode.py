#!/usr/bin/env python3
"""Fresh sealed 48/56/64-mode refinement of the archived final G8 cap."""

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


OUTPUT=Path("results/corrected_fold_G8_spectral_horizon_high_mode.json")
CHECKPOINT=Path("results/corrected_fold_G7_G8_spectral_horizon_refinement_state.npz")
MODES=(48,56,64);COLLOCATION=257


def relative_scalar(left,right):
    return float(abs(left-right)/max(abs(left),abs(right),1e-300))


def main():
    if not CHECKPOINT.exists():raise FileNotFoundError("note-63 checkpoint is required")
    print("reconstructing archived final G8 A=7.94 slice",flush=True)
    fold=build_geometry("G6");seed={**fold,"fold_amplitude":7.94}
    g7=build_refined(seed,81,121,"G7A794",selector_iterations=40,slice_iterations=270)
    g8=build_refined(g7,97,145,"G8A794",selector_iterations=45,slice_iterations=280)
    archive=np.load(CHECKPOINT);q0=np.asarray(g8["jet_field"].reduced_fields)
    q=q0+archive["G8_increment"];v=archive["G8_velocity"]
    static=solve_anisotropic_capped_profile(
        g8["z"],g8["r"],g8["psi"],g8["a"],g8["b"],g8["c"],1.53,
        tolerance=1e-8,nodes=220,max_nodes=12000,
    )
    surfaces=[]
    for modes in MODES:
        print(f"solving final G8 cap with {modes} modes",flush=True)
        surfaces.append(solve_spectral_dynamical_capped_surface(
            q,v,g8["z"],g8["r"],static,tolerance=5e-4,
            collocation_nodes=COLLOCATION,cosine_modes=modes,maximum_evaluations=220,
        ))
    comparisons=[]
    for coarse,fine in zip(surfaces[:-1],surfaces[1:]):
        initial=np.interp(fine["theta"],static["theta"],static["rho"])
        comparisons.append({
            "modes":[coarse["cosine_modes"],fine["cosine_modes"]],
            "profile_relative_difference":relative_norm(coarse["rho"],fine["rho"]),
            "axis_relative_difference":relative_scalar(coarse["rho_axis"],fine["rho_axis"]),
            "brane_relative_difference":relative_scalar(coarse["rho_brane"],fine["rho_brane"]),
            "displacement_relative_difference":relative_norm(coarse["rho"]-initial,fine["rho"]-initial),
        })
    records=[{
        "modes":surface["cosine_modes"],"converged":surface["converged"],"in_domain":surface["in_domain"],
        "expansion_maximum":surface["interior_expansion_maximum"],
        "rho_axis":surface["rho_axis"],"rho_brane":surface["rho_brane"],
        "axis_change":surface["rho_axis"]-static["rho_axis"],
        "brane_change":surface["rho_brane"]-static["rho_brane"],
        "jacobian_condition_number":surface["jacobian_condition_number"],
    } for surface in surfaces]
    acceptance={
        "all_surfaces_converge_below_5e_4":bool(all(
            item["converged"] and item["in_domain"] and item["expansion_maximum"]<.0005 for item in records
        )),
        "adjacent_profiles_and_radii_transfer_below_0_05_percent":bool(max(
            max(item["profile_relative_difference"],item["axis_relative_difference"],item["brane_relative_difference"])
            for item in comparisons
        )<.0005),
        "adjacent_displacements_transfer_below_1_percent":bool(max(
            item["displacement_relative_difference"] for item in comparisons
        )<.01),
        "all_motion_directions_positive":bool(all(
            item["axis_change"]>0 and item["brane_change"]>0 for item in records
        )),
    }
    payload={
        "status":"pass" if all(acceptance.values()) else "review",
        "scope":"fresh sealed 48/56/64-mode refinement of the archived final G8 A=7.94 smooth marginal cap",
        "protocol":"notes/64_G8_spectral_horizon_high_mode_protocol.md",
        "records":records,"comparisons":comparisons,"acceptance":acceptance,
        "limitations":[
            "reuses archived note-63 G8 spacetime","does not rescore note 63",
            "short-time persistence rather than formation or long-time stability",
        ],
    }
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps(payload,indent=2),flush=True)


if __name__=="__main__":main()
