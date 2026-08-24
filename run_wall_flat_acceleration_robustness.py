#!/usr/bin/env python3
"""Fold robustness under wall-flat initial stabilizer acceleration."""

import json,sys,warnings
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.capped_continuation import fit_fold_normal_form,solve_capped_profile
from bhps.capped_surface import find_donor_capped_surfaces
from bhps.finite_wall_solver import solve_finite_wall_slice


cases=[
    {"id":"mode1_minus10","mode":1,"strength":-10.},
    {"id":"stationary","mode":1,"strength":0.},
    {"id":"mode1_plus10","mode":1,"strength":10.},
    {"id":"mode2_minus10","mode":2,"strength":-10.},
    {"id":"mode2_plus10","mode":2,"strength":10.},
]


def metric(case,amplitude,initial=None):
    return solve_finite_wall_slice(
        amplitude,wall_stiffness=20.,nz=49,nr=73,r_max=8.,epsilon=.1,backreaction=.01,
        stabilizer_forcing_amplitude=case["strength"],stabilizer_forcing_mode=case["mode"],
        stabilizer_forcing_profile="sin_squared",initial=initial,tolerance=1e-10,iterations=180,
    )


outputs=[]
for case in cases:
    previous=None;inner=1.3;outer=1.7;records=[]
    for amplitude in np.arange(8.72,8.45,-.005):
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
            "radius_separation":separation,"max_stabilizer_deformation":geometry["max_stabilizer_deformation"],
        })
        if pair and separation>1e-4:inner,outer=ordered
        else:break
    close=[item for item in records if item["pair_converged"] and item["radius_separation"]<.25]
    fits=[fit_fold_normal_form(close,count) for count in (4,6,8,10) if len(close)>=count]
    selected=next(item for item in fits if item["fit_point_count"]==6);fold=metric(case,selected["fold_amplitude"])
    safe=metric(case,selected["fold_amplitude"]+.075)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        found=find_donor_capped_surfaces(
            safe["z"],safe["r"],safe["psi"],guesses=(1.25,1.35,1.45,1.55,1.65,1.75),
            tolerance=1e-8,stability_nodes=41,stability_step=2.5e-5,
        )
    surfaces=sorted(found["accepted"],key=lambda item:item["rho_brane"])
    outputs.append({
        **case,"selected_fold_amplitude":selected["fold_amplitude"],
        "selected_fold_energy_dimensionless":fold["energy_dimensionless"],
        "fold_max_stabilizer_deformation":fold["max_stabilizer_deformation"],
        "fold_stabilizer_forcing_max":fold["stabilizer_forcing_max"],
        "fold_coupled_residual":fold["max_abs_residual"],
        "fit_amplitude_systematic_span":max(item["fold_amplitude"] for item in fits)-min(item["fold_amplitude"] for item in fits),
        "normal_form_fits":fits,"records":records,
        "safe_angular_negative_counts_l0_through_l3":[
            [mode["negative_mode_count"] for mode in surface["angular_mode_spectrum"]] for surface in surfaces
        ],
    })

expected=[[1,0,0,0],[0,0,0,0]];baseline=next(item for item in outputs if item["id"]=="stationary")
payload={
    "status":"capped_fold_persists_under_wall_flat_initial_stabilizer_acceleration",
    "metric_case":"G4_R8","wall_stiffness":20.,"forcing_profile":"sin_squared","cases":outputs,
    "all_cases_have_expected_pair_stability":all(item["safe_angular_negative_counts_l0_through_l3"]==expected for item in outputs),
    "maximum_fold_energy_relative_shift_from_stationary":max(
        abs(item["selected_fold_energy_dimensionless"]-baseline["selected_fold_energy_dimensionless"])/baseline["selected_fold_energy_dimensionless"] for item in outputs
    ),
    "scalar_corner_property":"S and its coordinate-normal derivative vanish at both walls, satisfying the scalar-only part of second-order Robin compatibility",
    "limitations":[
        "single G4 grid/domain","two prescribed acceleration shapes and finite strength samples",
        "normal-form fold fits","metric-normal acceleration part of second-order wall compatibility remains open",
        "not a full dynamical evolution",
    ],
}
Path("results/wall_flat_acceleration_robustness.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps({
    "status":payload["status"],"all_cases_have_expected_pair_stability":payload["all_cases_have_expected_pair_stability"],
    "maximum_fold_energy_relative_shift_from_stationary":payload["maximum_fold_energy_relative_shift_from_stationary"],
    "folds":{item["id"]:{"A":item["selected_fold_amplitude"],"E":item["selected_fold_energy_dimensionless"],"dPhi":item["fold_max_stabilizer_deformation"],"modes":item["safe_angular_negative_counts_l0_through_l3"]} for item in outputs},
},indent=2))
