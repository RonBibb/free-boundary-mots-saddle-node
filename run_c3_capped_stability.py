#!/usr/bin/env python3
import json,sys,warnings
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from bhps.capped_surface import find_donor_capped_surfaces
from bhps.radion_variable_solver import solve_q

cases=[
    {"id":"G3_R8","nz":33,"nr":49,"r_max":8.,"amplitude":8.5},
    {"id":"G4_R8","nz":49,"nr":73,"r_max":8.,"amplitude":8.55},
    {"id":"G5_R8","nz":65,"nr":97,"r_max":8.,"amplitude":8.6},
    {"id":"G4_R10","nz":49,"nr":91,"r_max":10.,"amplitude":8.55},
]
output=[]
for case in cases:
    solved=solve_q(case["amplitude"],nz=case["nz"],nr=case["nr"],r_max=case["r_max"],tolerance=1e-10,iterations=180)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore",RuntimeWarning)
        surfaces=find_donor_capped_surfaces(solved["z"],solved["r"],solved["psi"],guesses=(1.3,1.4,1.5,1.6,1.7),tolerance=1e-8,stability_nodes=51,stability_step=2.5e-5)
    output.append({
        **case,
        "energy_dimensionless":solved["energy_dimensionless"],
        "constraint_residual":solved["max_abs_residual"],
        "surfaces":sorted(surfaces["accepted"],key=lambda item:item["rho_brane"]),
    })

classification=[]
for case in output:
    modes=[surface["negative_mode_count"] for surface in case["surfaces"]]
    angular_modes=[
        [item["negative_mode_count"] for item in surface["angular_mode_spectrum"]]
        for surface in case["surfaces"]
    ]
    angular_lowest=[
        [item["lowest_normalized_eigenvalue"] for item in surface["angular_mode_spectrum"]]
        for surface in case["surfaces"]
    ]
    classification.append({
        "id":case["id"],"surface_count":len(modes),"negative_mode_counts":modes,
        "angular_mode_negative_counts_l0_through_l3":angular_modes,
        "angular_mode_lowest_eigenvalues_l0_through_l3":angular_lowest,
    })
payload={
    "status":"strong_C3_candidate_not_physical_V1_result",
    "topology":"B-brane-capped half-S3 pair",
    "cases":output,
    "classification":classification,
    "all_cases_show_one_unstable_and_one_stable":all(item["negative_mode_counts"]==[1,0] for item in classification),
    "all_cases_show_no_nonspherical_negative_mode":all(
        all(counts[1:]==[0,0,0] for counts in item["angular_mode_negative_counts_l0_through_l3"])
        for item in classification
    ),
    "limitations":[
        "finite-difference meridional second variation",
        "angular sectors use the warped-product harmonic decomposition",
        "unstabilized C3 control"
    ],
}
path=Path("results/c3_capped_stability.json");path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps({
    "all_cases_show_one_unstable_and_one_stable":payload["all_cases_show_one_unstable_and_one_stable"],
    "all_cases_show_no_nonspherical_negative_mode":payload["all_cases_show_no_nonspherical_negative_mode"],
    "classification":classification,
},indent=2))
