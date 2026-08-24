#!/usr/bin/env python3
"""Refinement scan of the capped fold on equilibrium-GW initial slices."""

import json,sys,warnings
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.capped_continuation import fit_fold_normal_form,solve_capped_profile
from bhps.gw_slice_solver import solve_gw_slice


epsilon=.1;backreaction=.01
cases=[
    {"id":"G3_R8","nz":33,"nr":49,"r_max":8.,"start":8.7,"stop":8.36},
    {"id":"G4_R8","nz":49,"nr":73,"r_max":8.,"start":8.7,"stop":8.44},
    {"id":"G5_R8","nz":65,"nr":97,"r_max":8.,"start":8.75,"stop":8.48},
    {"id":"G6_R8","nz":81,"nr":121,"r_max":8.,"start":8.7,"stop":8.50},
    {"id":"G4_R10","nz":49,"nr":91,"r_max":10.,"start":8.7,"stop":8.44},
    {"id":"G4_R12","nz":49,"nr":109,"r_max":12.,"start":8.75,"stop":8.44},
]


def public(surface):
    return {key:surface[key] for key in (
        "rho_axis","rho_brane","rho_min","rho_max","area",
        "surface_residual_max","boundary_slope_error",
    )}


outputs=[]
for case in cases:
    previous_q=None;inner=1.3;outer=1.75;records=[]
    for amplitude in np.arange(case["start"],case["stop"]-.0025,-.005):
        amplitude=float(round(amplitude,6))
        geometry=solve_gw_slice(
            amplitude,nz=case["nz"],nr=case["nr"],r_max=case["r_max"],
            epsilon=epsilon,backreaction=backreaction,initial=previous_q,
            tolerance=1e-10,iterations=180,
        )
        previous_q=geometry["q"] if geometry["converged"] else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            first=solve_capped_profile(geometry["z"],geometry["r"],geometry["psi"],inner,tolerance=1e-8)
            second=solve_capped_profile(geometry["z"],geometry["r"],geometry["psi"],outer,tolerance=1e-8)
        ordered=sorted((first,second),key=lambda item:item["rho_brane"])
        pair=bool(geometry["converged"] and all(item["converged"] for item in ordered))
        separation=float(ordered[1]["rho_brane"]-ordered[0]["rho_brane"]) if pair else 0.
        records.append({
            "amplitude":amplitude,"energy_dimensionless":geometry["energy_dimensionless"],
            "constraint_residual":geometry["max_abs_residual"],"pair_converged":pair,
            "radius_separation":separation,"surfaces":[public(item) for item in ordered],
        })
        if pair and separation>1e-4:inner,outer=ordered
        else:break
    close=[item for item in records if item["pair_converged"] and item["radius_separation"]<.25]
    fits=[fit_fold_normal_form(close,tail) for tail in (4,6,8,10) if len(close)>=tail]
    selected=next(item for item in fits if item["fit_point_count"]==6)
    fold_geometry=solve_gw_slice(
        selected["fold_amplitude"],nz=case["nz"],nr=case["nr"],r_max=case["r_max"],
        epsilon=epsilon,backreaction=backreaction,tolerance=1e-10,iterations=180,
    )
    outputs.append({
        **{key:value for key,value in case.items() if key not in {"start","stop"}},"step":.005,
        "background_ads_relative_deformation":fold_geometry["background"]["max_ads_relative_deformation"],
        "records":records,"normal_form_fits":fits,
        "selected_fold_amplitude":selected["fold_amplitude"],
        "selected_fold_energy_dimensionless":fold_geometry["energy_dimensionless"],
        "selected_fold_energy_dimensionless_trapezoid":fold_geometry["energy_dimensionless_trapezoid"],
        "fold_energy_quadrature_relative_difference":fold_geometry["energy_quadrature_relative_difference"],
        "fit_amplitude_systematic_span":max(item["fold_amplitude"] for item in fits)-min(item["fold_amplitude"] for item in fits),
    })


def extrapolate(items,key):
    h=np.array([item["r_max"]/(item["nr"]-1) for item in items]);y=np.array([item[key] for item in items])
    ratio=(y[0]-y[1])/(y[1]-y[2])
    equation=lambda order:(h[0]**order-h[1]**order)/(h[1]**order-h[2]**order)-ratio
    order=float(brentq(equation,.1,8.));limit=float(y[2]+(y[2]-y[1])/((h[1]/h[2])**order-1))
    return {
        "observed_order":order,"extrapolated_value":limit,
        "two_finest_relative_change":float(abs(y[2]-y[1])/abs(y[2])),
        "finest_relative_extrapolation_error":float(abs(y[2]-limit)/abs(limit)),
    }


by_id={item["id"]:item for item in outputs};fine=[by_id[x] for x in ("G4_R8","G5_R8","G6_R8")]
summary={
    "fold_amplitude_convergence":extrapolate(fine,"selected_fold_amplitude"),
    "fold_energy_convergence":extrapolate(fine,"selected_fold_energy_dimensionless"),
    "G4_R8_R10_relative_domain_difference":{
        "fold_amplitude":abs(by_id["G4_R10"]["selected_fold_amplitude"]-by_id["G4_R8"]["selected_fold_amplitude"])/by_id["G4_R8"]["selected_fold_amplitude"],
        "fold_energy":abs(by_id["G4_R10"]["selected_fold_energy_dimensionless"]-by_id["G4_R8"]["selected_fold_energy_dimensionless"])/by_id["G4_R8"]["selected_fold_energy_dimensionless"],
    },
    "G4_R10_R12_relative_domain_difference":{
        "fold_amplitude":abs(by_id["G4_R12"]["selected_fold_amplitude"]-by_id["G4_R10"]["selected_fold_amplitude"])/by_id["G4_R10"]["selected_fold_amplitude"],
        "fold_energy":abs(by_id["G4_R12"]["selected_fold_energy_dimensionless"]-by_id["G4_R10"]["selected_fold_energy_dimensionless"])/by_id["G4_R10"]["selected_fold_energy_dimensionless"],
    },
    "maximum_fold_energy_quadrature_relative_difference":max(item["fold_energy_quadrature_relative_difference"] for item in outputs),
}
payload={
    "status":"equilibrium_GW_initial_slice_fold_branch_followed_not_arclength_crossed",
    "epsilon":epsilon,"backreaction_b0":backreaction,
    "initial_data_interpretation":"Phi on equilibrium background with zero momentum; metric responds through Hamiltonian constraint",
    "cases":outputs,"summary":summary,
    "limitations":["normal-form extrapolation rather than pseudo-arclength crossing", "single metric implementation", "no quasi-static delta-Phi solve"],
}
Path("results/gw_slice_capped_branch.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps({"cases":{item["id"]:[item["selected_fold_amplitude"],item["selected_fold_energy_dimensionless"]] for item in outputs},"summary":summary},indent=2))
