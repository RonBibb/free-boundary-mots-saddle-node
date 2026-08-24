#!/usr/bin/env python3
"""Replicate and classify the corrected anisotropic cap pair on two grids."""

import json,os,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_capped_surface import anisotropic_capped_area_stability,find_anisotropic_donor_capped_surfaces
from bhps.anisotropic_capped_surface_fd import solve_anisotropic_capped_surface_fd
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice


state_path=os.environ.get(
    "BHPS_CORNER_STATE","results/high_order_physical_corner_shared_grid_domain_solve_state.npz"
)
output_path=os.environ.get(
    "BHPS_CAP_OUTPUT","results/anisotropic_corrected_cap_grid_stability.json"
)
archive=np.load(state_path)
cases=[]
for name,nz,nr,amplitude in (
    ("G5R8",49,73,8.572038845434301),
    ("G6R8",65,97,8.572368541895216),
):
    reference=solve_finite_wall_high_order_slice(
        amplitude,nz=nz,nr=nr,r_max=8.,wall_stiffness=20.,epsilon=.1,
        backreaction=.01,tolerance=1e-10,iterations=240,
    )
    psi=1/(reference["z"][:,None]+archive[f"q_{name}"])
    a,b,c=(archive[f"{field}_{name}"] for field in "abc")
    collocation=find_anisotropic_donor_capped_surfaces(
        reference["z"],reference["r"],psi,a,b,c,
        guesses=tuple(np.linspace(.9,1.95,22)),tolerance=2e-5,
    )
    surfaces=[]
    for item in collocation["accepted"]:
        finite_difference=solve_anisotropic_capped_surface_fd(
            reference["z"],reference["r"],psi,a,b,c,item["rho_brane"],
            nodes=121,tolerance=1e-10,max_evaluations=10000,
        )
        stability=(
            anisotropic_capped_area_stability(
                reference["z"],reference["r"],psi,a,b,c,finite_difference,
                nodes=41,relative_step=2.5e-5,maximum_angular_mode=3,
            ) if finite_difference["converged"] else None
        )
        surfaces.append({
            "collocation":{key:item[key] for key in (
                "rho_axis","rho_brane","surface_residual_max","area"
            )},
            "finite_difference":{key:finite_difference[key] for key in (
                "converged","rho_axis","rho_brane","discrete_residual_max",
                "continuous_defect_interior_max","nodes"
            )},
            "stability":stability,
        })
    cases.append({
        "name":name,"grid_size":[nz,nr],"fold_amplitude":amplitude,
        "collocation_found":collocation["capped_surface_found"],
        "collocation_successful_trials":collocation["successful_trials"],
        "surfaces":surfaces,
    })

acceptance={
    "both_grids_have_two_caps":all(len(case["surfaces"])==2 for case in cases),
    "all_fd_solves_converged":all(
        surface["finite_difference"]["converged"]
        for case in cases for surface in case["surfaces"]
    ),
    "expected_inner_unstable_outer_stable":all(
        [surface["stability"]["negative_mode_count"] for surface in case["surfaces"]]==[1,0]
        for case in cases
    ),
}
payload={
    "status":"corrected_caps_replicated_and_classified_on_two_grids" if all(acceptance.values()) else "corrected_cap_two_grid_replication_incomplete",
    "corner_state":state_path,
    "cases":cases,"acceptance":acceptance,
}
Path(output_path).write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps({"status":payload["status"],"acceptance":acceptance},indent=2))
