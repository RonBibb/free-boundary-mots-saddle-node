#!/usr/bin/env python3
"""Sealed angular-node refinement of the final G7 dynamical cap."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_capped_surface import solve_anisotropic_capped_profile
from bhps.dynamical_capped_horizon import solve_dynamical_capped_surface
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import relative_norm


OUTPUT=Path("results/corrected_fold_dynamic_horizon_node_refinement.json")
CHECKPOINT=Path("results/corrected_fold_short_evolved_horizon_persistence_state.npz")
NODES=(61,81,101,121)


def main():
    if not CHECKPOINT.exists():raise FileNotFoundError("note-60 checkpoint is required")
    print("reconstructing corrected G7 A=7.94 final slice",flush=True)
    fold=build_geometry("G6");seed={**fold,"fold_amplitude":7.94}
    g6=build_refined(seed,65,97,"G6A794",selector_iterations=35,slice_iterations=260)
    g7=build_refined(g6,81,121,"G7A794",selector_iterations=40,slice_iterations=270)
    archive=np.load(CHECKPOINT)
    q=np.asarray(g7["jet_field"].reduced_fields)+archive["G7_increment"]
    v=archive["G7_velocity"]
    static=solve_anisotropic_capped_profile(
        g7["z"],g7["r"],g7["psi"],g7["a"],g7["b"],g7["c"],1.53,
        tolerance=1e-8,nodes=200,max_nodes=10000,
    )
    surfaces=[]
    for nodes in NODES:
        print(f"solving final cap with {nodes} nodes",flush=True)
        surfaces.append(solve_dynamical_capped_surface(
            q,v,g7["z"],g7["r"],static,tolerance=5e-5,nodes=nodes,
            maximum_evaluations=180,
        ))
    comparisons=[]
    for coarse,fine in zip(surfaces[:-1],surfaces[1:]):
        coarse_final=np.interp(fine["theta"],coarse["theta"],coarse["rho"])
        coarse_initial=np.interp(fine["theta"],static["theta"],static["rho"])
        fine_initial=np.interp(fine["theta"],static["theta"],static["rho"])
        comparisons.append({
            "nodes":[len(coarse["theta"]),len(fine["theta"])],
            "final_profile_relative_difference":relative_norm(coarse_final,fine["rho"]),
            "displacement_relative_difference":relative_norm(
                coarse_final-coarse_initial,fine["rho"]-fine_initial,
            ),
            "rho_axis_relative_difference":float(abs(coarse["rho_axis"]-fine["rho_axis"])/max(abs(coarse["rho_axis"]),abs(fine["rho_axis"]))),
            "rho_brane_relative_difference":float(abs(coarse["rho_brane"]-fine["rho_brane"])/max(abs(coarse["rho_brane"]),abs(fine["rho_brane"]))),
        })
    fine=comparisons[1:]
    motion=[{
        "nodes":nodes,
        "axis_change":float(surface["rho_axis"]-static["rho_axis"]),
        "brane_change":float(surface["rho_brane"]-static["rho_brane"]),
    } for nodes,surface in zip(NODES,surfaces)]
    acceptance={
        "all_solves_converge_below_1e_8":bool(all(
            item["converged"] and item["regularized_expansion_maximum"]<1e-8
            and item["boundary_slope_error"]<1e-8 and item["function_evaluations"]<50
            for item in surfaces
        )),
        "fine_profiles_and_radii_transfer_below_0_05_percent":bool(max(
            max(item["final_profile_relative_difference"],item["rho_axis_relative_difference"],item["rho_brane_relative_difference"])
            for item in fine
        )<.0005),
        "fine_displacements_transfer_below_10_percent":bool(max(
            item["displacement_relative_difference"] for item in fine
        )<.10),
        "fine_motion_directions_agree":bool(all(
            item["axis_change"]>0 and item["brane_change"]<0 for item in motion[1:]
        )),
    }
    summary={
        "comparisons":comparisons,"motion":motion,
        "solves":[{
            "nodes":nodes,"rho_axis":surface["rho_axis"],"rho_brane":surface["rho_brane"],
            "expansion_residual":surface["regularized_expansion_maximum"],
            "boundary_slope_error":surface["boundary_slope_error"],
            "function_evaluations":surface["function_evaluations"],
        } for nodes,surface in zip(NODES,surfaces)],
    }
    payload={
        "status":"pass" if all(acceptance.values()) else "review",
        "scope":"sealed angular-node refinement of the final G7 A=7.94 dynamical marginal cap from note 60",
        "protocol":"notes/61_dynamical_horizon_node_refinement_protocol.md",
        "summary":summary,"acceptance":acceptance,
        "limitations":[
            "reuses archived G7 final spacetime","does not rescore note-60 source-grid failure",
            "short-time persistence rather than formation or long-time stability",
        ],
    }
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps(payload,indent=2),flush=True)


if __name__=="__main__":main()
