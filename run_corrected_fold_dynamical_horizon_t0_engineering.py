#!/usr/bin/env python3
"""Unscored t=0 reduction of the dynamical expansion to the static cap."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_capped_surface import solve_anisotropic_capped_profile
from bhps.dynamical_capped_horizon import capped_outgoing_expansion
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry


OUTPUT=Path("results/corrected_fold_dynamical_horizon_t0_engineering.json")


def evaluate(geometry,label):
    profile=solve_anisotropic_capped_profile(
        geometry["z"],geometry["r"],geometry["psi"],geometry["a"],geometry["b"],
        geometry["c"],1.53,tolerance=1e-8,nodes=200,max_nodes=10000,
    )
    if not profile["converged"]:
        raise RuntimeError(f"{label} static cap did not converge: {profile['message']}")
    q=np.asarray(geometry["jet_field"].reduced_fields,dtype=float)
    expansion=capped_outgoing_expansion(
        q,np.zeros_like(q),geometry["z"],geometry["r"],profile,
    )
    # The first two points are most sensitive to differentiating the normal
    # beside the polar coordinate tip. Report both complete and trimmed norms.
    keep=slice(2,-2)
    collar={}
    values=expansion["outgoing_expansion"]
    for count in (2,4,8,16,24,32):
        collar[str(count)]=float(np.max(np.abs(values[count:-count])))
    radial_clear=expansion["r"]>=2*(geometry["r"][1]-geometry["r"][0])
    wall_clear=expansion["z"]<=geometry["z"][-1]-2*(geometry["z"][1]-geometry["z"][0])
    physical_interior=radial_clear&wall_clear
    return {
        "label":label,"grid_size":[len(geometry["z"]),len(geometry["r"])],
        "fold_amplitude":geometry["fold_amplitude"],
        "static_profile":{
            "rho_axis":profile["rho_axis"],"rho_brane":profile["rho_brane"],
            "surface_residual_max":profile["surface_residual_max"],
            "boundary_slope_error":profile["boundary_slope_error"],
        },
        "dynamical_expansion":{
            "complete_maximum_absolute":expansion["maximum_absolute_expansion"],
            "trimmed_maximum_absolute":float(np.max(np.abs(expansion["outgoing_expansion"][keep]))),
            "trimmed_l2":float(np.linalg.norm(expansion["outgoing_expansion"][keep])),
            "angular_collar_maxima":collar,
            "two_native_cell_interior_maximum":float(np.max(np.abs(values[physical_interior]))),
            "two_native_cell_interior_count":int(np.count_nonzero(physical_interior)),
            "maximum_extrinsic_correction":float(np.max(np.abs(expansion["extrinsic_curvature_correction"]))),
            "minimum":expansion["minimum_expansion"],"maximum":expansion["maximum_expansion"],
            "finite":expansion["finite"],
        },
    }


def main():
    print("building corrected G6/G7 states at A=7.94",flush=True)
    fold=build_geometry("G6");seed={**fold,"fold_amplitude":7.94}
    g6=build_refined(seed,65,97,"G6A794",selector_iterations=35,slice_iterations=260)
    g7=build_refined(g6,81,121,"G7A794",selector_iterations=40,slice_iterations=270)
    records=[evaluate(g6,"G6"),evaluate(g7,"G7")]
    payload={
        "status":"engineering_only",
        "scope":"unscored reduction of the full dynamical outgoing expansion to the static corrected-fold cap at t=0",
        "records":records,
        "limitations":[
            "no prospective acceptance rules",
            "time-symmetric slice only",
            "tests expansion evaluation on a known profile, not a dynamical surface solve",
        ],
    }
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps(payload,indent=2),flush=True)


if __name__=="__main__":main()
