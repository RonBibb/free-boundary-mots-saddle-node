#!/usr/bin/env python3
"""Check whether the shared corner-corrected fold retains a capped surface."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_capped_surface import find_anisotropic_donor_capped_surfaces
from bhps.capped_surface import find_donor_capped_surfaces
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice


amplitude=8.572038845434301
reference=solve_finite_wall_high_order_slice(
    amplitude,nz=49,nr=73,r_max=8.,wall_stiffness=20.,epsilon=.1,
    backreaction=.01,tolerance=1e-10,iterations=240,
)
archive=np.load("results/high_order_physical_corner_shared_grid_domain_solve_state.npz")
zero=np.zeros_like(reference["q"])
guesses=tuple(np.linspace(.9,1.68,17))
conformal=find_donor_capped_surfaces(
    reference["z"],reference["r"],reference["psi"],guesses=guesses,
    tolerance=2e-5,stability_nodes=31,
)
conformal_general=find_anisotropic_donor_capped_surfaces(
    reference["z"],reference["r"],reference["psi"],zero,zero,zero,
    guesses=guesses,tolerance=2e-5,
)
corrected_psi=1/(reference["z"][:,None]+archive["q_G5R8"])
corrected=find_anisotropic_donor_capped_surfaces(
    reference["z"],reference["r"],corrected_psi,
    archive["a_G5R8"],archive["b_G5R8"],archive["c_G5R8"],
    guesses=guesses,tolerance=2e-5,
)

def surfaces(result):
    return [{
        key:item[key] for key in (
            "rho_axis","rho_brane","rho_min","rho_max","z_tip",
            "brane_radius","surface_residual_max","area"
        )
    } for item in result["accepted"]]

old=surfaces(conformal);general=surfaces(conformal_general);new=surfaces(corrected)
agreement=[]
for item in old:
    if general:
        nearest=min(general,key=lambda candidate:abs(candidate["rho_brane"]-item["rho_brane"]))
        agreement.append({
            "old_rho_brane":item["rho_brane"],"general_rho_brane":nearest["rho_brane"],
            "absolute_difference":abs(nearest["rho_brane"]-item["rho_brane"]),
        })
payload={
    "status":"corrected_anisotropic_slice_contains_cap_pair" if len(corrected["accepted"])==2 else "corrected_anisotropic_cap_search_incomplete",
    "slice":{
        "solver":"independent five-point coupled finite-wall solver",
        "grid_size":[49,73],"r_max":8.,"fold_amplitude":amplitude,
        "corner_state":"results/high_order_physical_corner_shared_grid_domain_solve_state.npz",
    },
    "settings":{"guesses":list(guesses),"tolerance":2e-5},
    "conformal_reference":{
        "found":conformal["capped_surface_found"],"surfaces":old,
        "successful_trials":conformal["successful_trials"],
    },
    "anisotropic_equation_conformal_control":{
        "found":conformal_general["capped_surface_found"],"surfaces":general,
        "successful_trials":conformal_general["successful_trials"],
        "rho_brane_agreement":agreement,
    },
    "corrected_anisotropic":{
        "found":corrected["capped_surface_found"],"surfaces":new,
        "successful_trials":corrected["successful_trials"],
        "in_domain_successful_trials":corrected["in_domain_successful_trials"],
    },
    "limitations":[
        "single corrected fold grid",
        "collocation finder only",
        "anisotropic stability operator not yet evaluated",
        "corrected fold location has not been recontinued",
    ],
}
Path("results/anisotropic_corrected_cap_audit.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps(payload,indent=2))
