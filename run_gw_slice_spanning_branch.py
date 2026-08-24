#!/usr/bin/env python3
"""Branch-follow and classify the finite-box spanning-surface candidate."""

import json,sys,warnings
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from bhps.capped_continuation import fit_fold_normal_form
from bhps.gw_slice_solver import solve_gw_slice
from bhps.spanning_surface import find_spanning_surfaces,solve_spanning_profile

cases=[
    {"id":"G5_R8","nz":65,"nr":97,"r_max":8.,"start":20.2,"stop":19.1},
    {"id":"G6_R8","nz":81,"nr":121,"r_max":8.,"start":19.5,"stop":18.45},
    {"id":"G6_R10","nz":81,"nr":151,"r_max":10.,"start":22.5,"stop":20.8},
    {"id":"G6_R12","nz":81,"nr":181,"r_max":12.,"start":24.5,"stop":23.3},
]
outputs=[]
for case in cases:
    first_geometry=solve_gw_slice(
        case["start"],nz=case["nz"],nr=case["nr"],r_max=case["r_max"],
        epsilon=.1,backreaction=.01,tolerance=1e-10,iterations=180,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        found=find_spanning_surfaces(
            first_geometry["z"],first_geometry["r"],first_geometry["psi"],
            guesses=tuple(np.linspace(1.5,3.5,15)),tolerance=1e-8,stability_nodes=31,
        )
    seed=sorted(found["accepted"],key=lambda item:item["radius_B"])
    if len(seed)!=2:raise RuntimeError(f"expected two seed surfaces for {case['id']}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        inner=solve_spanning_profile(first_geometry["z"],first_geometry["r"],first_geometry["psi"],seed[0]["radius_A"],tolerance=1e-8)
        outer=solve_spanning_profile(first_geometry["z"],first_geometry["r"],first_geometry["psi"],seed[1]["radius_A"],tolerance=1e-8)
    previous=first_geometry["q"];records=[]
    for amplitude in np.arange(case["start"],case["stop"]-.025,-.05):
        amplitude=float(round(amplitude,6))
        geometry=first_geometry if amplitude==case["start"] else solve_gw_slice(
            amplitude,nz=case["nz"],nr=case["nr"],r_max=case["r_max"],
            epsilon=.1,backreaction=.01,initial=previous,tolerance=1e-10,iterations=180,
        )
        previous=geometry["q"] if geometry["converged"] else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            one=solve_spanning_profile(geometry["z"],geometry["r"],geometry["psi"],inner,tolerance=1e-8)
            two=solve_spanning_profile(geometry["z"],geometry["r"],geometry["psi"],outer,tolerance=1e-8)
        ordered=sorted((one,two),key=lambda item:item["radius_B"]);pair=bool(geometry["converged"] and all(item["converged"] for item in ordered))
        separation=float(ordered[1]["radius_B"]-ordered[0]["radius_B"]) if pair else 0.
        records.append({
            "amplitude":amplitude,"energy_dimensionless":geometry["energy_dimensionless"],
            "constraint_residual":geometry["max_abs_residual"],"pair_converged":pair,
            "radius_B_separation":separation,"radius_separation":separation,
            "surfaces":[{key:item[key] for key in (
                "radius_A","radius_B","radius_min","radius_max","area","surface_residual_max","boundary_slope_error"
            )} for item in ordered],
        })
        if pair and separation>1e-4:inner,outer=ordered
        else:break
    close=[item for item in records if item["pair_converged"] and item["radius_separation"]<.5]
    fits=[fit_fold_normal_form(close,count) for count in (4,6,8,10) if len(close)>=count]
    selected=next(item for item in fits if item["fit_point_count"]==6)
    fold_geometry=solve_gw_slice(
        selected["fold_amplitude"],nz=case["nz"],nr=case["nr"],r_max=case["r_max"],
        epsilon=.1,backreaction=.01,tolerance=1e-10,iterations=180,
    )
    outputs.append({
        **{key:value for key,value in case.items() if key not in {"start","stop"}},"amplitude_step":.05,
        "records":records,"normal_form_fits":fits,"selected_fold_amplitude":selected["fold_amplitude"],
        "selected_fold_energy_dimensionless":fold_geometry["energy_dimensionless"],
        "seed_pair_angular_negative_counts_l0_through_l3":[
            [mode["negative_mode_count"] for mode in item["angular_mode_spectrum"]] for item in seed
        ],
        "fit_amplitude_systematic_span":max(item["fold_amplitude"] for item in fits)-min(item["fold_amplitude"] for item in fits),
    })

by_id={item["id"]:item for item in outputs}
summary={
    "G6_R8_R10_fold_energy_relative_change":abs(by_id["G6_R10"]["selected_fold_energy_dimensionless"]-by_id["G6_R8"]["selected_fold_energy_dimensionless"])/by_id["G6_R8"]["selected_fold_energy_dimensionless"],
    "G6_R10_R12_fold_energy_relative_change":abs(by_id["G6_R12"]["selected_fold_energy_dimensionless"]-by_id["G6_R10"]["selected_fold_energy_dimensionless"])/by_id["G6_R10"]["selected_fold_energy_dimensionless"],
    "all_seed_pairs_match_inner_one_mode_outer_zero_modes":all(item["seed_pair_angular_negative_counts_l0_through_l3"]==[[1,0,0,0],[0,0,0,0]] for item in outputs),
}
payload={
    "status":"finite_box_spanning_fold_candidate_rejected_for_domain_nonconvergence",
    "epsilon":.1,"backreaction_b0":.01,"cases":outputs,"summary":summary,
    "limitations":["strong outer-domain drift", "normal-form rather than pseudo-arclength crossing", "very high source energy", "equilibrium stabilizer profile"],
}
Path("results/gw_slice_spanning_branch.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps({"folds":{item["id"]:[item["selected_fold_amplitude"],item["selected_fold_energy_dimensionless"]] for item in outputs},"summary":summary},indent=2))
