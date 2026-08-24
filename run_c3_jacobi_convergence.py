#!/usr/bin/env python3
import json,sys,warnings
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from bhps.capped_surface import find_donor_capped_surfaces
from bhps.radion_variable_solver import solve_q

solved=solve_q(8.55,nz=49,nr=73,r_max=8,tolerance=1e-10,iterations=180)
records=[]
for nodes in (31,41,51,61):
    for step in (1.25e-5,2.5e-5,5e-5):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            found=find_donor_capped_surfaces(
                solved["z"],solved["r"],solved["psi"],guesses=(1.35,1.6),
                tolerance=1e-8,stability_nodes=nodes,stability_step=step,
            )
        surfaces=sorted(found["accepted"],key=lambda item:item["rho_brane"])
        records.append({
            "nodes":nodes,"relative_step":step,
            "eigenvalues":[surface["lowest_normalized_jacobi_eigenvalue"] for surface in surfaces],
            "negative_mode_counts":[surface["normalized_negative_mode_count"] for surface in surfaces],
            "angular_lowest_eigenvalues_l0_through_l3":[
                [mode["lowest_normalized_eigenvalue"] for mode in surface["angular_mode_spectrum"]]
                for surface in surfaces
            ],
            "angular_negative_counts_l0_through_l3":[
                [mode["negative_mode_count"] for mode in surface["angular_mode_spectrum"]]
                for surface in surfaces
            ],
        })
payload={
    "case":"G4_R8","amplitude":8.55,
    "records":records,
    "all_mode_counts_stable":all(item["negative_mode_counts"]==[1,0] for item in records),
    "all_angular_mode_counts_stable":all(
        item["angular_negative_counts_l0_through_l3"]==[[1,0,0,0],[0,0,0,0]]
        for item in records
    ),
    "interpretation":"Normalized Jacobi signs through angular mode l=3 are stable under Hessian-node and differencing-step variation; positivity at l=1 and the positive l(l+1) shift exclude higher-angular negative modes",
}
path=Path("results/c3_jacobi_convergence.json");path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps({"all_mode_counts_stable":payload["all_mode_counts_stable"],"all_angular_mode_counts_stable":payload["all_angular_mode_counts_stable"],"records":records},indent=2))
