#!/usr/bin/env python3
import json,sys,warnings
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from bhps.capped_surface import find_donor_capped_surfaces
from bhps.radion_variable_solver import solve_q

cases=[
    {"id":"G2_R8","nz":25,"nr":37,"r_max":8.,"amplitudes":[8.2,8.25,8.3,8.325,8.35]},
    {"id":"G3_R8","nz":33,"nr":49,"r_max":8.,"amplitudes":[8.375,8.4]},
    {"id":"G4_R8","nz":49,"nr":73,"r_max":8.,"amplitudes":[8.475,8.5]},
    {"id":"G5_R8","nz":65,"nr":97,"r_max":8.,"amplitudes":[8.5,8.525,8.55]},
    {"id":"G4_R10","nz":49,"nr":91,"r_max":10.,"amplitudes":[8.475,8.5]},
]
seeds=tuple(np.linspace(.9,1.8,25));output=[]
for case in cases:
    previous=None;records=[]
    for amplitude in case["amplitudes"]:
        solved=solve_q(amplitude,nz=case["nz"],nr=case["nr"],r_max=case["r_max"],initial=previous,tolerance=1e-9,iterations=160)
        previous=solved["q"] if solved["converged"] else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            surfaces=find_donor_capped_surfaces(solved["z"],solved["r"],solved["psi"],guesses=seeds,tolerance=5e-6)
        records.append({
            "amplitude":amplitude,
            "energy_dimensionless":solved["energy_dimensionless"],
            "constraint_converged":solved["converged"],
            "constraint_residual":solved["max_abs_residual"],
            "surface_count":len(surfaces["accepted"]),
            "surfaces":surfaces["accepted"],
        })
    output.append({key:value for key,value in case.items() if key!="amplitudes"}|{"records":records})

payload={
    "status":"candidate_fold_not_accepted",
    "topology":"B-brane-capped half-S3",
    "cases":output,
    "limitations":[
        "first-appearance bracket drifts with resolution",
        "no pseudo-arclength continuation of the surface fold",
        "stability spectrum not yet computed",
        "C3 is an unstabilized control rather than the physical V1 model"
    ],
}
path=Path("results/c3_capped_bracket.json");path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps({case["id"]:[(item["amplitude"],item["surface_count"]) for item in case["records"]] for case in output},indent=2))
