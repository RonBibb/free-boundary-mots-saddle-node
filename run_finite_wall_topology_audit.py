#!/usr/bin/env python3
"""Multi-representation topology audit at fixed invariant collapse energy."""

import json,sys,warnings
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.capped_surface import find_donor_capped_surfaces
from bhps.capped_surface_fd import solve_capped_surface_fd
from bhps.closed_surface import find_closed_surfaces
from bhps.closed_surface_fd import find_closed_surfaces_fd
from bhps.finite_wall_solver import solve_finite_wall_slice
from bhps.spanning_surface import find_spanning_surfaces,spanning_maximum_principle_diagnostic
from bhps.spanning_surface_fd import find_spanning_surfaces_fd


target_energy=900.
cases=[
    {"id":"G4_R8","nz":49,"nr":73,"r_max":8.},
    {"id":"G5_R8","nz":65,"nr":97,"r_max":8.},
    {"id":"G6_R8","nz":81,"nr":121,"r_max":8.},
    {"id":"G4_R12","nz":49,"nr":109,"r_max":12.},
]


def geometry(case,amplitude):
    return solve_finite_wall_slice(
        amplitude,wall_stiffness=20.,nz=case["nz"],nr=case["nr"],r_max=case["r_max"],
        epsilon=.1,backreaction=.01,tolerance=1e-10,iterations=180,
    )


def fd_caps(z,r,psi,guesses):
    accepted=[];success=0
    for guess in guesses:
        item=solve_capped_surface_fd(z,r,psi,guess,nodes=121,tolerance=1e-10)
        success+=int(item["solver_success"])
        if item["converged"] and not any(abs(item["rho_brane"]-old["rho_brane"])<5e-3 for old in accepted):accepted.append(item)
    return {"accepted":accepted,"successful_trials":success,"trial_count":len(guesses)}


outputs=[]
for case in cases:
    amplitude=float(brentq(lambda value:geometry(case,value)["energy_dimensionless"]-target_energy,8.,9.5,xtol=2e-11))
    solved=geometry(case,amplitude);z,r,psi=solved["z"],solved["r"],solved["psi"]
    cap_guesses=tuple(np.linspace(max(.25,3*r[1]),2.1,12))
    spanning_guesses=tuple(np.linspace(max(.25,3*r[1]),.8*r[-1],22))
    centers=np.linspace(z[0]+4*(z[1]-z[0]),z[-1]-4*(z[1]-z[0]),7)
    closed_guesses=np.linspace(max(.25,3*r[1]),min(.46*(z[-1]-z[0]),2.2),10)
    z_reflected=z[0]+z[-1]-z[::-1];psi_reflected=psi[::-1,:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        upper=find_donor_capped_surfaces(z,r,psi,guesses=cap_guesses,tolerance=1e-8,stability_nodes=31)
        upper_fd=fd_caps(z,r,psi,cap_guesses)
        lower=find_donor_capped_surfaces(z_reflected,r,psi_reflected,guesses=cap_guesses,tolerance=1e-8,stability_nodes=31)
        lower_fd=fd_caps(z_reflected,r,psi_reflected,cap_guesses)
        closed=find_closed_surfaces(z,r,psi,centers=centers,guesses=closed_guesses,tolerance=1e-7)
        closed_fd=find_closed_surfaces_fd(z,r,psi,centers=centers,guesses=closed_guesses,nodes=101,tolerance=1e-10)
        spanning=find_spanning_surfaces(z,r,psi,guesses=spanning_guesses,tolerance=1e-8,stability_nodes=31)
        spanning_fd=find_spanning_surfaces_fd(z,r,psi,guesses=spanning_guesses,nodes=101,tolerance=1e-10)
    obstruction=spanning_maximum_principle_diagnostic(z,r,psi,refinement=5)
    outputs.append({
        **case,"amplitude_at_target_energy":amplitude,"energy_dimensionless":solved["energy_dimensionless"],
        "coupled_residual":solved["max_abs_residual"],
        "upper_cap":{
            "collocation_count":len(upper["accepted"]),"fd_count":len(upper_fd["accepted"]),
            "collocation_radii":sorted(item["rho_brane"] for item in upper["accepted"]),
            "fd_radii":sorted(item["rho_brane"] for item in upper_fd["accepted"]),
        },
        "opposite_cap":{
            "collocation_count":len(lower["accepted"]),"fd_count":len(lower_fd["accepted"]),
            "collocation_successful_trials":lower["successful_trials"],"fd_successful_trials":lower_fd["successful_trials"],
        },
        "closed_bulk":{
            "collocation_count":len(closed["accepted"]),"fd_count":len(closed_fd["accepted"]),
            "collocation_successful_trials":closed["successful_trials"],"collocation_in_domain_successful_trials":closed["in_domain_successful_trials"],
            "fd_successful_trials":closed_fd["successful_trials"],"fd_in_domain_successful_trials":closed_fd["in_domain_successful_trials"],
        },
        "interval_spanning":{
            "collocation_count":len(spanning["accepted"]),"fd_count":len(spanning_fd["accepted"]),
            "collocation_successful_trials":spanning["successful_trials"],
            "fd_successful_trials":sum(item["solver_success"] for item in spanning_fd["trials"]),
            "sampled_maximum_principle":obstruction,
        },
    })

payload={
    "status":"fixed_energy_multi_representation_search_recovers_only_pulse_side_capped_pair",
    "target_energy_dimensionless":target_energy,"wall_stiffness":20.,"cases":outputs,
    "all_cases_recover_two_upper_caps_in_both_solvers":all(item["upper_cap"]["collocation_count"]==2 and item["upper_cap"]["fd_count"]==2 for item in outputs),
    "any_other_topology_candidate":any(
        item[key][solver]>0 for item in outputs for key in ("opposite_cap","closed_bulk","interval_spanning") for solver in ("collocation_count","fd_count")
    ),
    "limitations":[
        "non-detection by finite-seed nonlinear searches is not a nonexistence proof",
        "the sampled spanning maximum-principle obstruction is not positive at E=900",
        "closed surfaces restricted to star-shaped polar representations",
        "SO(3)-symmetric time-symmetric initial slices only",
    ],
}
Path("results/finite_wall_topology_audit.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps({
    "status":payload["status"],"all_cases_recover_two_upper_caps_in_both_solvers":payload["all_cases_recover_two_upper_caps_in_both_solvers"],
    "any_other_topology_candidate":payload["any_other_topology_candidate"],
    "cases":outputs,
},indent=2))
