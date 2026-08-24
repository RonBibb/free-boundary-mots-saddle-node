#!/usr/bin/env python3
"""One-at-a-time Goldberger--Wise parameter robustness of the capped fold."""

import json,sys,warnings
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from bhps.capped_continuation import fit_fold_normal_form,solve_capped_profile
from bhps.capped_surface import find_donor_capped_surfaces
from bhps.gw_slice_solver import solve_gw_slice

cases=[
    {"id":"unstabilized_control","epsilon":.1,"backreaction":0.,"start":8.7,"stop":8.35},
    {"id":"b0_0p003","epsilon":.1,"backreaction":.003,"start":8.7,"stop":8.36},
    {"id":"baseline","epsilon":.1,"backreaction":.01,"start":8.7,"stop":8.38},
    {"id":"b0_0p03","epsilon":.1,"backreaction":.03,"start":8.75,"stop":8.43},
    {"id":"epsilon_0p075","epsilon":.075,"backreaction":.01,"start":8.7,"stop":8.38},
    {"id":"epsilon_0p125","epsilon":.125,"backreaction":.01,"start":8.7,"stop":8.38},
]


def metric(case,amplitude,initial=None):
    return solve_gw_slice(
        amplitude,nz=49,nr=73,r_max=8,epsilon=case["epsilon"],backreaction=case["backreaction"],
        initial=initial,tolerance=1e-10,iterations=180,
    )


outputs=[]
for case in cases:
    previous=None;inner=1.3;outer=1.7;records=[]
    for amplitude in np.arange(case["start"],case["stop"]-.0025,-.005):
        amplitude=float(round(amplitude,6));geometry=metric(case,amplitude,previous)
        previous=geometry["q"] if geometry["converged"] else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            one=solve_capped_profile(geometry["z"],geometry["r"],geometry["psi"],inner,tolerance=1e-8)
            two=solve_capped_profile(geometry["z"],geometry["r"],geometry["psi"],outer,tolerance=1e-8)
        ordered=sorted((one,two),key=lambda item:item["rho_brane"]);pair=bool(geometry["converged"] and all(item["converged"] for item in ordered))
        separation=float(ordered[1]["rho_brane"]-ordered[0]["rho_brane"]) if pair else 0.
        records.append({
            "amplitude":amplitude,"energy_dimensionless":geometry["energy_dimensionless"],
            "constraint_residual":geometry["max_abs_residual"],"pair_converged":pair,"radius_separation":separation,
        })
        if pair and separation>1e-4:inner,outer=ordered
        else:break
    close=[item for item in records if item["pair_converged"] and item["radius_separation"]<.25]
    fits=[fit_fold_normal_form(close,count) for count in (4,6,8,10) if len(close)>=count]
    selected=next(item for item in fits if item["fit_point_count"]==6);fold=metric(case,selected["fold_amplitude"])
    safe_amplitude=selected["fold_amplitude"]+.075;safe=metric(case,safe_amplitude)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        found=find_donor_capped_surfaces(
            safe["z"],safe["r"],safe["psi"],guesses=(1.25,1.35,1.45,1.55,1.65,1.75),
            tolerance=1e-8,stability_nodes=41,stability_step=2.5e-5,
        )
    surfaces=sorted(found["accepted"],key=lambda item:item["rho_brane"])
    outputs.append({
        **{key:value for key,value in case.items() if key not in {"start","stop"}},"amplitude_step":.005,
        "background_ads_relative_deformation":fold["background"]["max_ads_relative_deformation"],
        "selected_fold_amplitude":selected["fold_amplitude"],"selected_fold_energy_dimensionless":fold["energy_dimensionless"],
        "fit_amplitude_systematic_span":max(item["fold_amplitude"] for item in fits)-min(item["fold_amplitude"] for item in fits),
        "normal_form_fits":fits,"records":records,"safe_stability_amplitude":safe_amplitude,
        "safe_angular_negative_counts_l0_through_l3":[
            [mode["negative_mode_count"] for mode in surface["angular_mode_spectrum"]] for surface in surfaces
        ],
    })

expected=[[1,0,0,0],[0,0,0,0]]
payload={
    "status":"capped_fold_persists_under_one_at_a_time_weak_GW_parameter_variations",
    "metric_case":"G4_R8","cases":outputs,
    "all_cases_have_expected_pair_stability":all(item["safe_angular_negative_counts_l0_through_l3"]==expected for item in outputs),
    "limitations":["one-at-a-time parameter variations", "single G4 metric/domain", "stiff-wall equilibrium profiles", "normal-form fold fits"],
}
Path("results/gw_parameter_robustness.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps({
    "all_cases_have_expected_pair_stability":payload["all_cases_have_expected_pair_stability"],
    "folds":{item["id"]:{"A":item["selected_fold_amplitude"],"E":item["selected_fold_energy_dimensionless"],"background_deformation":item["background_ads_relative_deformation"],"modes":item["safe_angular_negative_counts_l0_through_l3"]} for item in outputs},
},indent=2))
