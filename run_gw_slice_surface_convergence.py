#!/usr/bin/env python3
"""Capped-surface convergence at fixed invariant collapse-field energy."""

import json,sys,warnings
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.capped_continuation import solve_capped_profile
from bhps.gw_slice_solver import solve_gw_slice

target_energy=900.
cases=[
    {"id":"G3_R8","nz":33,"nr":49,"r_max":8.},
    {"id":"G4_R8","nz":49,"nr":73,"r_max":8.},
    {"id":"G5_R8","nz":65,"nr":97,"r_max":8.},
    {"id":"G6_R8","nz":81,"nr":121,"r_max":8.},
    {"id":"G4_R10","nz":49,"nr":91,"r_max":10.},
    {"id":"G4_R12","nz":49,"nr":109,"r_max":12.},
]


def metric(case,amplitude):
    return solve_gw_slice(
        amplitude,nz=case["nz"],nr=case["nr"],r_max=case["r_max"],
        epsilon=.1,backreaction=.01,tolerance=1e-10,iterations=180,
    )


outputs=[]
for case in cases:
    cache={}
    def difference(amplitude):
        key=float(amplitude)
        if key not in cache:cache[key]=metric(case,key)
        return cache[key]["energy_dimensionless"]-target_energy
    amplitude=float(brentq(difference,8.5,9.0,xtol=2e-10));geometry=metric(case,amplitude)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        inner=solve_capped_profile(geometry["z"],geometry["r"],geometry["psi"],1.3,tolerance=1e-8)
        outer=solve_capped_profile(geometry["z"],geometry["r"],geometry["psi"],1.7,tolerance=1e-8)
    surfaces=sorted((inner,outer),key=lambda item:item["rho_brane"])
    if not all(item["converged"] for item in surfaces):raise RuntimeError(f"surface failure in {case['id']}")
    outputs.append({
        **case,"amplitude_at_target_energy":amplitude,"energy_dimensionless":geometry["energy_dimensionless"],
        "energy_target_relative_error":abs(geometry["energy_dimensionless"]-target_energy)/target_energy,
        "constraint_residual":geometry["max_abs_residual"],
        "surfaces":[{key:item[key] for key in (
            "rho_axis","rho_brane","rho_min","rho_max","area","surface_residual_max","boundary_slope_error"
        )} for item in surfaces],
    })


def extrapolate(items,branch,key):
    h=np.array([item["r_max"]/(item["nr"]-1) for item in items]);y=np.array([item["surfaces"][branch][key] for item in items])
    ratio=(y[0]-y[1])/(y[1]-y[2]);equation=lambda p:(h[0]**p-h[1]**p)/(h[1]**p-h[2]**p)-ratio
    order=float(brentq(equation,.1,8.));limit=float(y[2]+(y[2]-y[1])/((h[1]/h[2])**order-1))
    return {
        "observed_order":order,"extrapolated_value":limit,
        "two_finest_relative_change":abs(y[2]-y[1])/abs(y[2]),
        "finest_relative_extrapolation_error":abs(y[2]-limit)/abs(limit),
    }


by_id={item["id"]:item for item in outputs};fine=[by_id[x] for x in ("G4_R8","G5_R8","G6_R8")]
convergence={}
for branch,name in ((0,"inner"),(1,"outer")):
    convergence[name]={key:extrapolate(fine,branch,key) for key in ("rho_axis","rho_brane","area")}

domain={}
for branch,name in ((0,"inner"),(1,"outer")):
    domain[name]={}
    for key in ("rho_axis","rho_brane","area"):
        first=by_id["G4_R10"]["surfaces"][branch][key];second=by_id["G4_R12"]["surfaces"][branch][key]
        domain[name][key]=abs(second-first)/abs(second)

payload={
    "status":"fixed_invariant_energy_surface_convergence_passed","target_energy_dimensionless":target_energy,
    "epsilon":.1,"backreaction_b0":.01,"cases":outputs,
    "three_grid_convergence_G4_G6_R8":convergence,
    "G4_R10_R12_relative_domain_differences":domain,
    "limitations":["single invariant energy above fold", "equilibrium stabilizer profile", "collocation surface solver"],
}
Path("results/gw_slice_surface_convergence.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps({"amplitudes":{item["id"]:item["amplitude_at_target_energy"] for item in outputs},"convergence":convergence,"domain":domain},indent=2))
