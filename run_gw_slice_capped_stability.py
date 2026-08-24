#!/usr/bin/env python3
"""Full-angular stability classification of equilibrium-GW capped pairs."""

import json,sys,warnings
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from bhps.capped_surface import find_donor_capped_surfaces
from bhps.gw_slice_solver import solve_gw_slice

cases=[
    {"id":"G3_R8","nz":33,"nr":49,"r_max":8.,"amplitude":8.49},
    {"id":"G4_R8","nz":49,"nr":73,"r_max":8.,"amplitude":8.58},
    {"id":"G5_R8","nz":65,"nr":97,"r_max":8.,"amplitude":8.61},
    {"id":"G6_R8","nz":81,"nr":121,"r_max":8.,"amplitude":8.625},
    {"id":"G4_R10","nz":49,"nr":91,"r_max":10.,"amplitude":8.60},
    {"id":"G4_R12","nz":49,"nr":109,"r_max":12.,"amplitude":8.61},
]
output=[]
for case in cases:
    geometry=solve_gw_slice(
        case["amplitude"],nz=case["nz"],nr=case["nr"],r_max=case["r_max"],
        epsilon=.1,backreaction=.01,tolerance=1e-10,iterations=180,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        found=find_donor_capped_surfaces(
            geometry["z"],geometry["r"],geometry["psi"],guesses=(1.3,1.4,1.5,1.6,1.7),
            tolerance=1e-8,stability_nodes=51,stability_step=2.5e-5,
        )
    surfaces=sorted(found["accepted"],key=lambda item:item["rho_brane"])
    output.append({
        **case,"energy_dimensionless":geometry["energy_dimensionless"],
        "constraint_residual":geometry["max_abs_residual"],"surfaces":surfaces,
    })

classification=[]
for case in output:
    classification.append({
        "id":case["id"],"surface_count":len(case["surfaces"]),
        "angular_mode_negative_counts_l0_through_l3":[
            [mode["negative_mode_count"] for mode in surface["angular_mode_spectrum"]]
            for surface in case["surfaces"]
        ],
        "angular_mode_lowest_eigenvalues_l0_through_l3":[
            [mode["lowest_normalized_eigenvalue"] for mode in surface["angular_mode_spectrum"]]
            for surface in case["surfaces"]
        ],
    })
expected=[[1,0,0,0],[0,0,0,0]]
payload={
    "status":"equilibrium_GW_initial_slice_full_angular_stability_classified",
    "epsilon":.1,"backreaction_b0":.01,"cases":output,"classification":classification,
    "all_cases_match_inner_one_mode_outer_zero_modes":all(item["angular_mode_negative_counts_l0_through_l3"]==expected for item in classification),
    "limitations":["finite-difference meridional second variation", "equilibrium-profile initial-data subfamily"],
}
Path("results/gw_slice_capped_stability.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps({"all_cases_match":payload["all_cases_match_inner_one_mode_outer_zero_modes"],"classification":classification},indent=2))
