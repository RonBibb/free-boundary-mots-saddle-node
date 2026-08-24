#!/usr/bin/env python3
import json,sys,warnings
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from bhps.capped_continuation import fit_fold_normal_form,solve_capped_profile
from bhps.radion_variable_solver import solve_q

cases=[
    {"id":"G3_R8","nz":33,"nr":49,"r_max":8.,"start":8.7,"stop":8.32},
    {"id":"G4_R8","nz":49,"nr":73,"r_max":8.,"start":8.7,"stop":8.42},
    {"id":"G5_R8","nz":65,"nr":97,"r_max":8.,"start":8.75,"stop":8.47},
    {"id":"G6_R8","nz":81,"nr":121,"r_max":8.,"start":8.65,"stop":8.48},
    {"id":"G4_R10","nz":49,"nr":91,"r_max":10.,"start":8.7,"stop":8.42},
]

def public_surface(surface):
    return {key:surface[key] for key in (
        "rho_axis","rho_brane","rho_min","rho_max","area",
        "surface_residual_max","boundary_slope_error",
    )}

outputs=[]
for case in cases:
    q_previous=None;inner=1.3;outer=1.75;records=[]
    amplitudes=np.arange(case["start"],case["stop"]-.0025,-.005)
    for amplitude in amplitudes:
        amplitude=float(round(amplitude,6))
        geometry=solve_q(amplitude,nz=case["nz"],nr=case["nr"],r_max=case["r_max"],initial=q_previous,tolerance=1e-10,iterations=180)
        q_previous=geometry["q"] if geometry["converged"] else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            first=solve_capped_profile(geometry["z"],geometry["r"],geometry["psi"],inner,tolerance=1e-8)
            second=solve_capped_profile(geometry["z"],geometry["r"],geometry["psi"],outer,tolerance=1e-8)
        ordered=sorted((first,second),key=lambda item:item["rho_brane"])
        pair=bool(geometry["converged"] and all(item["converged"] for item in ordered))
        separation=float(ordered[1]["rho_brane"]-ordered[0]["rho_brane"]) if pair else 0.
        records.append({
            "amplitude":amplitude,
            "energy_dimensionless":geometry["energy_dimensionless"],
            "constraint_residual":geometry["max_abs_residual"],
            "pair_converged":pair,
            "radius_separation":separation,
            "surfaces":[public_surface(item) for item in ordered],
        })
        if pair and separation>1e-4:
            inner,outer=ordered
        else:
            break
    close=[item for item in records if item["pair_converged"] and item["radius_separation"]<.25]
    fits=[fit_fold_normal_form(close,tail) for tail in (4,6,8,10) if len(close)>=tail]
    selected=next(item for item in fits if item["fit_point_count"]==6) if len(close)>=6 else fits[0]
    fold_geometry=solve_q(selected["fold_amplitude"],nz=case["nz"],nr=case["nr"],r_max=case["r_max"],tolerance=1e-10,iterations=180)
    outputs.append({
        **{key:value for key,value in case.items() if key not in {"start","stop"}},
        "step":.005,
        "records":records,
        "normal_form_fits":fits,
        "selected_fold_amplitude":selected["fold_amplitude"],
        "selected_fold_energy_dimensionless":fold_geometry["energy_dimensionless"],
        "fit_amplitude_systematic_span":max(item["fold_amplitude"] for item in fits)-min(item["fold_amplitude"] for item in fits),
    })

def three_grid_extrapolation(items,key):
    hs=np.array([item["r_max"]/(item["nr"]-1) for item in items])
    ys=np.array([item[key] for item in items])
    ratio=(ys[0]-ys[1])/(ys[1]-ys[2])
    def equation(order):
        return (hs[0]**order-hs[1]**order)/(hs[1]**order-hs[2]**order)-ratio
    order=float(brentq(equation,.1,8.))
    extrapolated=float(ys[2]+(ys[2]-ys[1])/((hs[1]/hs[2])**order-1.))
    return {
        "observed_order":order,
        "extrapolated_value":extrapolated,
        "two_finest_relative_change":float(abs(ys[2]-ys[1])/abs(ys[2])),
        "finest_relative_extrapolation_error":float(abs(ys[2]-extrapolated)/abs(extrapolated)),
    }

by_id={item["id"]:item for item in outputs}
fine=[by_id[item] for item in ("G4_R8","G5_R8","G6_R8")]
summary={
    "fold_amplitude_convergence":three_grid_extrapolation(fine,"selected_fold_amplitude"),
    "fold_energy_convergence":three_grid_extrapolation(fine,"selected_fold_energy_dimensionless"),
    "G4_R8_R10_relative_domain_difference":{
        "fold_amplitude":abs(by_id["G4_R10"]["selected_fold_amplitude"]-by_id["G4_R8"]["selected_fold_amplitude"])/by_id["G4_R8"]["selected_fold_amplitude"],
        "fold_energy":abs(by_id["G4_R10"]["selected_fold_energy_dimensionless"]-by_id["G4_R8"]["selected_fold_energy_dimensionless"])/by_id["G4_R8"]["selected_fold_energy_dimensionless"],
    },
}

payload={
    "status":"branch_followed_fold_estimate_not_pseudo_arclength",
    "cases":outputs,
    "summary":summary,
    "limitations":[
        "normal-form extrapolation rather than pseudo-arclength crossing",
        "metric-grid threshold extrapolation still required",
        "unstabilized C3 control"
    ],
}
path=Path("results/c3_capped_branch_follow.json");path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps({
    "cases":{case["id"]:{"A_fold":case["selected_fold_amplitude"],"E_fold":case["selected_fold_energy_dimensionless"],"fit_span":case["fit_amplitude_systematic_span"]} for case in outputs},
    "summary":summary,
},indent=2))
