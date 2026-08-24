#!/usr/bin/env python3
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from scipy.sparse.linalg import eigs
from bhps.radion_variable_solver import continue_q,q_jacobian,solve_q
from bhps.scalar_pulse import scalar_pulse

amplitudes=[0,.5,.75,.82,.84,.85,.86,.865,.87,.875,.88]
cases=[
    {"id":"G3_R8","nz":33,"nr":49,"r_max":8.},
    {"id":"G4_R8","nz":49,"nr":73,"r_max":8.},
    {"id":"G5_R8","nz":65,"nr":97,"r_max":8.},
    {"id":"G3_R10","nz":33,"nr":61,"r_max":10.},
    {"id":"G4_R10","nz":49,"nr":91,"r_max":10.},
    {"id":"G3_R12","nz":33,"nr":73,"r_max":12.},
    {"id":"G4_R12","nz":49,"nr":109,"r_max":12.},
]
output=[]
for case in cases:
    branch=continue_q(amplitudes,nz=case["nz"],nr=case["nr"],r_max=case["r_max"],tolerance=1e-9,iterations=120)
    weak=solve_q(.5,nz=case["nz"],nr=case["nr"],r_max=case["r_max"],tolerance=1e-9,iterations=120)
    _,chi_r,chi_z=scalar_pulse(weak["z"],weak["r"],.5)
    eigenvalue=eigs(q_jacobian(weak["q"],weak["z"],weak["r"],chi_r,chi_z),k=1,sigma=0,which="LM",return_eigenvectors=False)[0]
    output.append({
        **case,
        "accepted":branch["accepted"],
        "rejected":branch["rejected"],
        "near_zero_eigenvalue_A0p5":float(eigenvalue.real),
        "near_zero_eigenvalue_imag_A0p5":float(eigenvalue.imag),
    })

def point(case,amplitude):
    return next(item for item in case["accepted"] if item["amplitude"]==amplitude)

weak=[point(case,.5) for case in output]
energies=[item["energy_dimensionless"] for item in weak]
deformations=[abs(item["interpolated_axis_extremum"]) for item in weak]
summary={
    "all_cases_reach_A0p88":all(case["accepted"][-1]["amplitude"]==.88 and case["rejected"] is None for case in output),
    "energy_all_case_relative_spread_A0p5":(max(energies)-min(energies))/(sum(energies)/len(energies)),
    "deformation_all_case_relative_spread_A0p5":(max(deformations)-min(deformations))/(sum(deformations)/len(deformations)),
    "energy_G4_R8_R12_relative_difference_A0p5":abs(point(output[1],.5)["energy_dimensionless"]-point(output[-1],.5)["energy_dimensionless"])/point(output[-1],.5)["energy_dimensionless"],
    "deformation_G4_R8_R12_relative_difference_A0p5":abs(point(output[1],.5)["interpolated_axis_extremum"]-point(output[-1],.5)["interpolated_axis_extremum"])/abs(point(output[-1],.5)["interpolated_axis_extremum"]),
}
payload={
    "variable":"q=1/psi-z",
    "outer_boundary":"q_r+q/r=0",
    "amplitudes":amplitudes,
    "acceptance_residual":1e-9,
    "cases":output,
    "summary":summary,
}
path=Path("results/c3_q_replication.json");path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(summary,indent=2,sort_keys=True))
