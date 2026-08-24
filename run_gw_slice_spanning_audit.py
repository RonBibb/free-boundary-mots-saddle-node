#!/usr/bin/env python3
"""Two-solver and maximum-principle audit of interval-spanning horizons."""

import json,sys,warnings
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from bhps.gw_slice_solver import solve_gw_slice
from bhps.spanning_surface import find_spanning_surfaces,spanning_maximum_principle_diagnostic
from bhps.spanning_surface_fd import find_spanning_surfaces_fd

cases=[
    {"id":"G4_R8","nz":49,"nr":73,"r_max":8.},
    {"id":"G5_R8","nz":65,"nr":97,"r_max":8.},
    {"id":"G4_R12","nz":49,"nr":109,"r_max":12.},
    {"id":"G6_R8","nz":81,"nr":121,"r_max":8.},
]
amplitudes=(0.,3.,5.,5.5,6.,8.6,10.,12.,16.,18.,19.,20.,22.,24.)
outputs=[]
for case in cases:
    records=[];previous=None
    for amplitude in amplitudes:
        geometry=solve_gw_slice(
            amplitude,nz=case["nz"],nr=case["nr"],r_max=case["r_max"],
            epsilon=.1,backreaction=.01,initial=previous,tolerance=1e-10,iterations=180,
        )
        previous=geometry["q"] if geometry["converged"] else None
        guesses=tuple(np.linspace(max(3*geometry["r"][1],.12),.82*geometry["r"][-1],24))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            collocation=find_spanning_surfaces(
                geometry["z"],geometry["r"],geometry["psi"],guesses=guesses,tolerance=1e-8,stability_nodes=31,
            )
            finite_difference=find_spanning_surfaces_fd(
                geometry["z"],geometry["r"],geometry["psi"],guesses=guesses,nodes=101,tolerance=1e-10,
            )
        obstruction=spanning_maximum_principle_diagnostic(geometry["z"],geometry["r"],geometry["psi"],refinement=5)
        records.append({
            "amplitude":amplitude,"energy_dimensionless":geometry["energy_dimensionless"],
            "constraint_residual":geometry["max_abs_residual"],"maximum_principle":obstruction,
            "collocation_accepted_count":len(collocation["accepted"]),
            "collocation_successful_trial_count":collocation["successful_trials"],
            "finite_difference_accepted_count":len(finite_difference["accepted"]),
            "finite_difference_successful_trial_count":sum(item["solver_success"] for item in finite_difference["trials"]),
            "collocation_accepted":[{key:item[key] for key in (
                "radius_A","radius_B","radius_min","radius_max","area","surface_residual_max","angular_mode_spectrum"
            )} for item in collocation["accepted"]],
            "finite_difference_accepted":[{key:item[key] for key in (
                "radius_A","radius_B","radius_min","radius_max","discrete_residual_max","continuous_defect_interior_max"
            )} for item in finite_difference["accepted"]],
        })
    outputs.append({**case,"records":records})

payload={
    "status":"low_amplitude_sampled_maximum_principle_obstruction_high_energy_two_solver_spanning_pair_candidate_not_domain_converged",
    "epsilon":.1,"backreaction_b0":.01,"cases":outputs,
    "limitations":[
        "maximum-principle statement is a sampled numerical certificate and loses sign above the low-amplitude range",
        "failure of both nonlinear finders is not a nonexistence proof where the obstruction is absent",
        "finite seed sets and three grid/domain cases",
        "equilibrium stabilizer profile",
    ],
}
Path("results/gw_slice_spanning_audit.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps({case["id"]:[{
    "A":item["amplitude"],"E":item["energy_dimensionless"],
    "obstruction":item["maximum_principle"]["strict_obstruction_on_sampled_domain"],
    "margin":item["maximum_principle"]["minimum_2_over_R_plus_3_dR_logpsi"],
    "collocation":item["collocation_accepted_count"],"fd":item["finite_difference_accepted_count"],
} for item in case["records"]] for case in outputs},indent=2))
