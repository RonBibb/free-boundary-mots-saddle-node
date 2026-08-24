#!/usr/bin/env python3
"""Grid and radial-domain refinement of the coupled finite-wall capped fold."""

import json,sys,warnings
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.capped_continuation import fit_fold_normal_form,solve_capped_profile
from bhps.finite_wall_solver import solve_finite_wall_slice


wall_stiffness=20.;epsilon=.1;backreaction=.01
cases=[
    {"id":"G3_R8","nz":33,"nr":49,"r_max":8.,"start":8.65,"stop":8.35},
    {"id":"G4_R8","nz":49,"nr":73,"r_max":8.,"start":8.70,"stop":8.45},
    {"id":"G5_R8","nz":65,"nr":97,"r_max":8.,"start":8.70,"stop":8.48},
    {"id":"G6_R8","nz":81,"nr":121,"r_max":8.,"start":8.70,"stop":8.50},
    {"id":"G4_R10","nz":49,"nr":91,"r_max":10.,"start":8.72,"stop":8.46},
    {"id":"G4_R12","nz":49,"nr":109,"r_max":12.,"start":8.72,"stop":8.47},
]


def metric(case,amplitude,initial=None):
    return solve_finite_wall_slice(
        amplitude,wall_stiffness=wall_stiffness,nz=case["nz"],nr=case["nr"],r_max=case["r_max"],
        epsilon=epsilon,backreaction=backreaction,initial=initial,tolerance=1e-10,iterations=180,
    )


outputs=[]
for case in cases:
    previous=None;inner=1.3;outer=1.7;records=[]
    for amplitude in np.arange(case["start"],case["stop"]-.0025,-.005):
        amplitude=float(round(amplitude,6));geometry=metric(case,amplitude,previous)
        previous=geometry if geometry["converged"] else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            one=solve_capped_profile(geometry["z"],geometry["r"],geometry["psi"],inner,tolerance=1e-8)
            two=solve_capped_profile(geometry["z"],geometry["r"],geometry["psi"],outer,tolerance=1e-8)
        ordered=sorted((one,two),key=lambda item:item["rho_brane"])
        pair=bool(geometry["converged"] and all(item["converged"] for item in ordered))
        separation=float(ordered[1]["rho_brane"]-ordered[0]["rho_brane"]) if pair else 0.
        records.append({
            "amplitude":amplitude,"energy_dimensionless":geometry["energy_dimensionless"],
            "coupled_residual":geometry["max_abs_residual"],"pair_converged":pair,
            "radius_separation":separation,
        })
        if pair and separation>1e-4:inner,outer=ordered
        else:break
    close=[item for item in records if item["pair_converged"] and item["radius_separation"]<.25]
    fits=[fit_fold_normal_form(close,count) for count in (4,6,8,10) if len(close)>=count]
    selected=next(item for item in fits if item["fit_point_count"]==6);fold=metric(case,selected["fold_amplitude"])
    outputs.append({
        **{key:value for key,value in case.items() if key not in {"start","stop"}},"amplitude_step":.005,
        "selected_fold_amplitude":selected["fold_amplitude"],
        "selected_fold_energy_dimensionless":fold["energy_dimensionless"],
        "fold_energy_quadrature_relative_difference":fold["energy_quadrature_relative_difference"],
        "fold_coupled_residual":fold["max_abs_residual"],
        "fit_amplitude_systematic_span":max(item["fold_amplitude"] for item in fits)-min(item["fold_amplitude"] for item in fits),
        "normal_form_fits":fits,"records":records,
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


by_id={item["id"]:item for item in outputs};grids=[by_id[x] for x in ("G4_R8","G5_R8","G6_R8")]
summary={
    "fold_amplitude_convergence":extrapolate(grids,"selected_fold_amplitude"),
    "fold_energy_convergence":extrapolate(grids,"selected_fold_energy_dimensionless"),
    "G4_R8_R10_relative_domain_difference":{
        "fold_amplitude":abs(by_id["G4_R10"]["selected_fold_amplitude"]-by_id["G4_R8"]["selected_fold_amplitude"])/by_id["G4_R8"]["selected_fold_amplitude"],
        "fold_energy":abs(by_id["G4_R10"]["selected_fold_energy_dimensionless"]-by_id["G4_R8"]["selected_fold_energy_dimensionless"])/by_id["G4_R8"]["selected_fold_energy_dimensionless"],
    },
    "G4_R10_R12_relative_domain_difference":{
        "fold_amplitude":abs(by_id["G4_R12"]["selected_fold_amplitude"]-by_id["G4_R10"]["selected_fold_amplitude"])/by_id["G4_R10"]["selected_fold_amplitude"],
        "fold_energy":abs(by_id["G4_R12"]["selected_fold_energy_dimensionless"]-by_id["G4_R10"]["selected_fold_energy_dimensionless"])/by_id["G4_R10"]["selected_fold_energy_dimensionless"],
    },
    "maximum_fold_energy_quadrature_relative_difference":max(item["fold_energy_quadrature_relative_difference"] for item in outputs),
    "maximum_fold_coupled_residual":max(item["fold_coupled_residual"] for item in outputs),
}
payload={
    "status":"finite_wall_coupled_capped_fold_grid_and_domain_refined",
    "wall_stiffness":wall_stiffness,"epsilon":epsilon,"backreaction_b0":backreaction,
    "cases":outputs,"summary":summary,
    "limitations":["four grid levels ending at G6","normal-form fold fits","one coupled discretization","momentarily stationary scalar selector"],
}
Path("results/finite_wall_refinement.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps({
    "status":payload["status"],
    "folds":{item["id"]:[item["selected_fold_amplitude"],item["selected_fold_energy_dimensionless"]] for item in outputs},
    "summary":summary,
},indent=2))
