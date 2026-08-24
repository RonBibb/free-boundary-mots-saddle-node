#!/usr/bin/env python3
"""Sealed t=0 and evolved-slice repair of the G7 dynamic cap tracker."""

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


OUTPUT=Path("results/corrected_fold_spectral_horizon_repair.json")
CHECKPOINT=Path("results/corrected_fold_short_evolved_horizon_persistence_state.npz")
MODES=(32,40,48);COLLOCATION=257


def solve_set(q,v,geometry,static):
    return [solve_spectral_dynamical_capped_surface(
        q,v,geometry["z"],geometry["r"],static,tolerance=5e-4,
        collocation_nodes=COLLOCATION,cosine_modes=modes,maximum_evaluations=180,
    ) for modes in MODES]


def pair_comparisons(surfaces,static):
    records=[]
    for coarse,fine in zip(surfaces[:-1],surfaces[1:]):
        coarse_on=np.interp(fine["theta"],coarse["theta"],coarse["rho"])
        initial=np.interp(fine["theta"],static["theta"],static["rho"])
        records.append({
            "modes":[coarse["cosine_modes"],fine["cosine_modes"]],
            "profile_relative_difference":relative_norm(coarse_on,fine["rho"]),
            "displacement_relative_difference":relative_norm(coarse_on-initial,fine["rho"]-initial),
            "axis_relative_difference":float(abs(coarse["rho_axis"]-fine["rho_axis"])/max(abs(coarse["rho_axis"]),abs(fine["rho_axis"]))),
            "brane_relative_difference":float(abs(coarse["rho_brane"]-fine["rho_brane"])/max(abs(coarse["rho_brane"]),abs(fine["rho_brane"]))),
        })
    return records


def main():
    if not CHECKPOINT.exists():raise FileNotFoundError("note-60 checkpoint is required")
    print("reconstructing initial and final G7 A=7.94 slices",flush=True)
    fold=build_geometry("G6");seed={**fold,"fold_amplitude":7.94}
    g6=build_refined(seed,65,97,"G6A794",selector_iterations=35,slice_iterations=260)
    g7=build_refined(g6,81,121,"G7A794",selector_iterations=40,slice_iterations=270)
    static=solve_anisotropic_capped_profile(
        g7["z"],g7["r"],g7["psi"],g7["a"],g7["b"],g7["c"],1.53,
        tolerance=1e-8,nodes=200,max_nodes=10000,
    )
    archive=np.load(CHECKPOINT);q0=np.asarray(g7["jet_field"].reduced_fields)
    print("solving t=0 spectral controls",flush=True)
    initial=solve_set(q0,np.zeros_like(q0),g7,static)
    print("solving evolved spectral caps",flush=True)
    evolved=solve_set(q0+archive["G7_increment"],archive["G7_velocity"],g7,static)
    initial_records=[]
    for surface in initial:
        reference=np.interp(surface["theta"],static["theta"],static["rho"])
        initial_records.append({
            "modes":surface["cosine_modes"],"expansion_maximum":surface["interior_expansion_maximum"],
            "profile_difference_from_static":relative_norm(surface["rho"],reference),
            "axis_difference_from_static":float(abs(surface["rho_axis"]-static["rho_axis"])/static["rho_axis"]),
            "brane_difference_from_static":float(abs(surface["rho_brane"]-static["rho_brane"])/static["rho_brane"]),
            "converged":surface["converged"],"in_domain":surface["in_domain"],
        })
    comparisons=pair_comparisons(evolved,static)
    evolved_records=[{
        "modes":surface["cosine_modes"],"rho_axis":surface["rho_axis"],"rho_brane":surface["rho_brane"],
        "axis_change":surface["rho_axis"]-static["rho_axis"],
        "brane_change":surface["rho_brane"]-static["rho_brane"],
        "expansion_maximum":surface["interior_expansion_maximum"],
        "converged":surface["converged"],"in_domain":surface["in_domain"],
        "jacobian_condition_number":surface["jacobian_condition_number"],
    } for surface in evolved]
    selected_initial=initial_records[1:];selected_evolved=evolved_records[1:]
    acceptance={
        "fine_t0_recovery_below_0_05_percent":bool(max(
            max(item["expansion_maximum"],item["profile_difference_from_static"],item["axis_difference_from_static"],item["brane_difference_from_static"])
            for item in selected_initial
        )<.0005),
        "fine_evolved_expansions_below_5e_4":bool(max(
            item["expansion_maximum"] for item in selected_evolved
        )<.0005),
        "evolved_profiles_and_radii_transfer_below_0_05_percent":bool(max(
            max(item["profile_relative_difference"],item["axis_relative_difference"],item["brane_relative_difference"])
            for item in comparisons
        )<.0005),
        "evolved_displacements_transfer_below_1_percent":bool(max(
            item["displacement_relative_difference"] for item in comparisons
        )<.01),
        "evolved_motion_directions_agree":bool(all(
            item["axis_change"]>0 and item["brane_change"]>0 for item in evolved_records
        )),
        "all_selected_solves_finite_and_in_domain":bool(all(
            item["converged"] and item["in_domain"] for item in selected_initial+selected_evolved
        )),
    }
    summary={"t0_recovery":initial_records,"evolved":evolved_records,"evolved_mode_comparisons":comparisons}
    payload={
        "status":"pass" if all(acceptance.values()) else "review",
        "scope":"sealed smooth spectral repair of the G7 A=7.94 dynamic cap tracker at t=0 and t=0.002",
        "protocol":"notes/62_spectral_dynamical_horizon_repair_protocol.md",
        "summary":summary,"acceptance":acceptance,
        "limitations":[
            "single G7 spacetime grid","does not rescore note-60 source-grid failure",
            "short-time persistence rather than formation, topology change, or long-time stability",
        ],
    }
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps(payload,indent=2),flush=True)


if __name__=="__main__":main()
