#!/usr/bin/env python3
"""One-at-a-time pulse-profile robustness scan of the equilibrium-GW fold."""

import json,sys,warnings
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.capped_continuation import fit_fold_normal_form,solve_capped_profile
from bhps.capped_surface import find_donor_capped_surfaces
from bhps.gw_slice_solver import solve_gw_slice

cases=[
    {"id":"baseline","sigma_r":1.,"sigma_y":.2,"center_fraction":.9,"start":8.70,"stop":8.30},
    {"id":"center_0p85","sigma_r":1.,"sigma_y":.2,"center_fraction":.85,"start":9.10,"stop":8.40},
    {"id":"center_0p95","sigma_r":1.,"sigma_y":.2,"center_fraction":.95,"start":8.80,"stop":7.70},
    {"id":"sigma_y_0p175","sigma_r":1.,"sigma_y":.175,"center_fraction":.9,"start":8.30,"stop":7.20},
    {"id":"sigma_y_0p225","sigma_r":1.,"sigma_y":.225,"center_fraction":.9,"start":10.20,"stop":9.00},
    {"id":"sigma_r_0p9","sigma_r":.9,"sigma_y":.2,"center_fraction":.9,"start":9.10,"stop":7.90},
    {"id":"sigma_r_1p1","sigma_r":1.1,"sigma_y":.2,"center_fraction":.9,"start":9.20,"stop":8.40},
]


def metric(case,amplitude,initial=None):
    return solve_gw_slice(
        amplitude,nz=49,nr=73,r_max=8,epsilon=.1,backreaction=.01,
        sigma_r=case["sigma_r"],sigma_y=case["sigma_y"],center_fraction=case["center_fraction"],
        initial=initial,tolerance=1e-10,iterations=180,
    )


def starting_pair(geometry):
    found=[]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        for guess in np.linspace(.75,1.75,9):
            item=solve_capped_profile(geometry["z"],geometry["r"],geometry["psi"],float(guess),tolerance=1e-8)
            if item["converged"] and not any(abs(item["rho_brane"]-old["rho_brane"])<2e-3 for old in found):
                found.append(item)
    found=sorted(found,key=lambda item:item["rho_brane"])
    if len(found)<2:raise RuntimeError("starting amplitude did not yield a cap pair")
    return found[0],found[-1]


outputs=[]
for case in cases:
    first_geometry=metric(case,case["start"]);inner,outer=starting_pair(first_geometry)
    previous=first_geometry["q"];records=[]
    for amplitude in np.arange(case["start"],case["stop"]-.005,-.01):
        amplitude=float(round(amplitude,6))
        geometry=first_geometry if amplitude==case["start"] else metric(case,amplitude,previous)
        previous=geometry["q"] if geometry["converged"] else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            one=solve_capped_profile(geometry["z"],geometry["r"],geometry["psi"],inner,tolerance=1e-8)
            two=solve_capped_profile(geometry["z"],geometry["r"],geometry["psi"],outer,tolerance=1e-8)
        ordered=sorted((one,two),key=lambda item:item["rho_brane"]);pair=bool(geometry["converged"] and all(item["converged"] for item in ordered))
        separation=float(ordered[1]["rho_brane"]-ordered[0]["rho_brane"]) if pair else 0.
        records.append({
            "amplitude":amplitude,"energy_dimensionless":geometry["energy_dimensionless"],
            "constraint_residual":geometry["max_abs_residual"],"pair_converged":pair,
            "radius_separation":separation,
        })
        if pair and separation>1e-4:inner,outer=ordered
        else:break
    close=[item for item in records if item["pair_converged"] and item["radius_separation"]<.3]
    fits=[fit_fold_normal_form(close,count) for count in (4,6,8,10) if len(close)>=count]
    selected=next(item for item in fits if item["fit_point_count"]==6)
    fold_geometry=metric(case,selected["fold_amplitude"])
    safe_amplitude=min(case["start"],selected["fold_amplitude"]+.08);safe_geometry=metric(case,safe_amplitude)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        stable=find_donor_capped_surfaces(
            safe_geometry["z"],safe_geometry["r"],safe_geometry["psi"],
            guesses=tuple(np.linspace(.8,1.75,8)),tolerance=1e-8,stability_nodes=41,stability_step=2.5e-5,
        )
    surfaces=sorted(stable["accepted"],key=lambda item:item["rho_brane"])
    angular_counts=[
        [mode["negative_mode_count"] for mode in surface["angular_mode_spectrum"]]
        for surface in surfaces
    ]
    outputs.append({
        **{key:value for key,value in case.items() if key not in {"start","stop"}},
        "amplitude_step":.01,"records":records,"normal_form_fits":fits,
        "selected_fold_amplitude":selected["fold_amplitude"],
        "selected_fold_energy_dimensionless":fold_geometry["energy_dimensionless"],
        "fit_amplitude_systematic_span":max(item["fold_amplitude"] for item in fits)-min(item["fold_amplitude"] for item in fits),
        "safe_stability_amplitude":safe_amplitude,"safe_surface_count":len(surfaces),
        "safe_angular_negative_counts_l0_through_l3":angular_counts,
    })

expected=[[1,0,0,0],[0,0,0,0]]
payload={
    "status":"open_one_at_a_time_pulse_neighborhood_has_same_capped_fold_structure",
    "metric_case":"G4_R8","epsilon":.1,"backreaction_b0":.01,"cases":outputs,
    "all_cases_have_pair_and_expected_stability":all(item["safe_angular_negative_counts_l0_through_l3"]==expected for item in outputs),
    "limitations":["one-at-a-time variations", "single metric grid/domain", "normal-form rather than arclength crossings", "finite parameter intervals only"],
}
Path("results/gw_slice_profile_robustness.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps({
    "all_cases_have_pair_and_expected_stability":payload["all_cases_have_pair_and_expected_stability"],
    "folds":{item["id"]:{"A":item["selected_fold_amplitude"],"E":item["selected_fold_energy_dimensionless"],"modes":item["safe_angular_negative_counts_l0_through_l3"]} for item in outputs},
},indent=2))
