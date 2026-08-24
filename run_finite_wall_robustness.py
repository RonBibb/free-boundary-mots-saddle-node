#!/usr/bin/env python3
"""Finite-wall coupled-response robustness of the capped fold."""

import json,sys,warnings
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.capped_continuation import fit_fold_normal_form,solve_capped_profile
from bhps.capped_surface import find_donor_capped_surfaces
from bhps.finite_wall_solver import solve_finite_wall_slice


configuration={
    "metric_case":"G4_R8","nz":49,"nr":73,"r_max":8.,
    "epsilon":.1,"backreaction_b0":.01,"wall_stiffnesses":[2.,5.,20.,100.],
    "amplitude_start":8.7,"amplitude_stop":8.45,"amplitude_step":.005,
}


def metric(gamma,amplitude,initial=None):
    return solve_finite_wall_slice(
        amplitude,wall_stiffness=gamma,nz=configuration["nz"],nr=configuration["nr"],
        r_max=configuration["r_max"],epsilon=configuration["epsilon"],
        backreaction=configuration["backreaction_b0"],initial=initial,
        tolerance=1e-10,iterations=180,
    )


outputs=[]
for gamma in configuration["wall_stiffnesses"]:
    previous=None;inner=1.3;outer=1.7;records=[]
    stop=configuration["amplitude_stop"]-.5*configuration["amplitude_step"]
    for amplitude in np.arange(configuration["amplitude_start"],stop,-configuration["amplitude_step"]):
        amplitude=float(round(amplitude,6));geometry=metric(gamma,amplitude,previous)
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
    selected=next(item for item in fits if item["fit_point_count"]==6)
    fold=metric(gamma,selected["fold_amplitude"])
    safe_amplitude=selected["fold_amplitude"]+.075;safe=metric(gamma,safe_amplitude)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        found=find_donor_capped_surfaces(
            safe["z"],safe["r"],safe["psi"],guesses=(1.25,1.35,1.45,1.55,1.65,1.75),
            tolerance=1e-8,stability_nodes=41,stability_step=2.5e-5,
        )
    surfaces=sorted(found["accepted"],key=lambda item:item["rho_brane"])
    outputs.append({
        "wall_stiffness":gamma,"selected_fold_amplitude":selected["fold_amplitude"],
        "selected_fold_energy_dimensionless":fold["energy_dimensionless"],
        "fold_coupled_residual":fold["max_abs_residual"],
        "fold_scalar_wall_residual":fold["scalar_wall_residual_max"],
        "fold_max_stabilizer_deformation":fold["max_stabilizer_deformation"],
        "background_ads_relative_deformation":fold["background"]["max_ads_relative_deformation"],
        "fit_amplitude_systematic_span":max(item["fold_amplitude"] for item in fits)-min(item["fold_amplitude"] for item in fits),
        "normal_form_fits":fits,"records":records,"safe_stability_amplitude":safe_amplitude,
        "safe_angular_negative_counts_l0_through_l3":[
            [mode["negative_mode_count"] for mode in surface["angular_mode_spectrum"]] for surface in surfaces
        ],
    })

expected=[[1,0,0,0],[0,0,0,0]]
fold_amplitudes=np.array([item["selected_fold_amplitude"] for item in outputs])
fold_energies=np.array([item["selected_fold_energy_dimensionless"] for item in outputs])
payload={
    "status":"capped_fold_persists_with_coupled_finite_wall_stabilizer_response",
    "configuration":configuration,"cases":outputs,
    "all_cases_have_expected_pair_stability":all(item["safe_angular_negative_counts_l0_through_l3"]==expected for item in outputs),
    "wall_scan_fold_amplitude_relative_span":float(np.ptp(fold_amplitudes)/np.mean(fold_amplitudes)),
    "wall_scan_fold_energy_relative_span":float(np.ptp(fold_energies)/np.mean(fold_energies)),
    "limitations":[
        "single G4 metric grid and radial domain","normal-form fold fits",
        "N=psi quasi-static stabilizer selector, not a complete static-spacetime solution",
        "one weak-backreaction and epsilon choice",
    ],
}
Path("results/finite_wall_robustness.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps({
    "status":payload["status"],"all_cases_have_expected_pair_stability":payload["all_cases_have_expected_pair_stability"],
    "wall_scan_fold_amplitude_relative_span":payload["wall_scan_fold_amplitude_relative_span"],
    "wall_scan_fold_energy_relative_span":payload["wall_scan_fold_energy_relative_span"],
    "folds":{str(item["wall_stiffness"]):{
        "A":item["selected_fold_amplitude"],"E":item["selected_fold_energy_dimensionless"],
        "dPhi":item["fold_max_stabilizer_deformation"],"modes":item["safe_angular_negative_counts_l0_through_l3"],
    } for item in outputs},
},indent=2))
