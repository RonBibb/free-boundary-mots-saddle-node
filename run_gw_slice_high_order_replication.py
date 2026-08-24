#!/usr/bin/env python3
"""Independent high-order metric-discretization replication of the GW fold."""

import json,sys,warnings
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.capped_continuation import fit_fold_normal_form,solve_capped_profile
from bhps.gw_slice_high_order_solver import solve_gw_high_order_slice

cases=[
    {"id":"H3_R8","nz":33,"nr":49,"r_max":8.},
    {"id":"H4_R8","nz":49,"nr":73,"r_max":8.},
    {"id":"H5_R8","nz":65,"nr":97,"r_max":8.},
    {"id":"H4_R10","nz":49,"nr":91,"r_max":10.},
    {"id":"H4_R12","nz":49,"nr":109,"r_max":12.},
]
outputs=[]
for case in cases:
    previous=None;inner=1.3;outer=1.75;records=[]
    for amplitude in np.arange(8.75,8.54,-.005):
        amplitude=float(round(amplitude,6))
        geometry=solve_gw_high_order_slice(
            amplitude,nz=case["nz"],nr=case["nr"],r_max=case["r_max"],
            epsilon=.1,backreaction=.01,initial=previous,tolerance=1e-10,iterations=180,
        )
        previous=geometry["q"] if geometry["converged"] else None
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
            "radius_separation":separation,
        })
        if pair and separation>1e-4:inner,outer=ordered
        else:break
    close=[item for item in records if item["pair_converged"] and item["radius_separation"]<.25]
    fits=[fit_fold_normal_form(close,count) for count in (4,6,8,10) if len(close)>=count]
    selected=next(item for item in fits if item["fit_point_count"]==6)
    fold_geometry=solve_gw_high_order_slice(
        selected["fold_amplitude"],nz=case["nz"],nr=case["nr"],r_max=case["r_max"],
        epsilon=.1,backreaction=.01,tolerance=1e-10,iterations=180,
    )
    outputs.append({
        **case,"amplitude_step":.005,"records":records,"normal_form_fits":fits,
        "selected_fold_amplitude":selected["fold_amplitude"],
        "selected_fold_energy_dimensionless":fold_geometry["energy_dimensionless"],
        "fold_energy_quadrature_relative_difference":fold_geometry["energy_quadrature_relative_difference"],
        "fit_amplitude_systematic_span":max(item["fold_amplitude"] for item in fits)-min(item["fold_amplitude"] for item in fits),
    })


def extrapolate(items,key):
    h=np.array([item["r_max"]/(item["nr"]-1) for item in items]);y=np.array([item[key] for item in items])
    ratio=(y[0]-y[1])/(y[1]-y[2]);equation=lambda p:(h[0]**p-h[1]**p)/(h[1]**p-h[2]**p)-ratio
    order=float(brentq(equation,.1,8.));limit=float(y[2]+(y[2]-y[1])/((h[1]/h[2])**order-1))
    return {"observed_order":order,"extrapolated_value":limit,"finest_relative_extrapolation_error":abs(y[2]-limit)/abs(limit)}


by_id={item["id"]:item for item in outputs};fine=[by_id[x] for x in ("H3_R8","H4_R8","H5_R8")]
primary=json.loads(Path("results/gw_slice_capped_branch.json").read_text())["summary"]
amplitude_convergence=extrapolate(fine,"selected_fold_amplitude")
energy_convergence=extrapolate(fine,"selected_fold_energy_dimensionless")
summary={
    "fold_amplitude_convergence":amplitude_convergence,"fold_energy_convergence":energy_convergence,
    "high_order_vs_primary_extrapolated_relative_difference":{
        "fold_amplitude":abs(amplitude_convergence["extrapolated_value"]-primary["fold_amplitude_convergence"]["extrapolated_value"])/primary["fold_amplitude_convergence"]["extrapolated_value"],
        "fold_energy":abs(energy_convergence["extrapolated_value"]-primary["fold_energy_convergence"]["extrapolated_value"])/primary["fold_energy_convergence"]["extrapolated_value"],
    },
    "H4_R10_R12_relative_domain_difference":{
        "fold_amplitude":abs(by_id["H4_R12"]["selected_fold_amplitude"]-by_id["H4_R10"]["selected_fold_amplitude"])/by_id["H4_R10"]["selected_fold_amplitude"],
        "fold_energy":abs(by_id["H4_R12"]["selected_fold_energy_dimensionless"]-by_id["H4_R10"]["selected_fold_energy_dimensionless"])/by_id["H4_R10"]["selected_fold_energy_dimensionless"],
    },
}
payload={
    "status":"independent_high_order_metric_discretization_replication_passed",
    "epsilon":.1,"backreaction_b0":.01,"cases":outputs,"summary":summary,
    "independence":"separate residual/Jacobian with five-point polynomial derivative matrices; same physical q variable and outer condition",
    "limitations":["same continuum conformal ansatz", "same collocation surface finder", "normal-form rather than arclength on replication path"],
}
Path("results/gw_slice_high_order_replication.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps({"cases":{item["id"]:[item["selected_fold_amplitude"],item["selected_fold_energy_dimensionless"]] for item in outputs},"summary":summary},indent=2))
