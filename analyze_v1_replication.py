#!/usr/bin/env python3
import json
from pathlib import Path

data=json.loads(Path("results/v1_replication.json").read_text())
cases={case["id"]:case for case in data["cases"]}
def point(case_id,amplitude):
    return next(x for x in cases[case_id]["accepted"] if x["amplitude"]==amplitude)
def rel(a,b):return abs(a-b)/abs(b)
weak=[point(case_id,.5) for case_id in cases]
energies=[item["energy_dimensionless"] for item in weak]
deformations=[item["max_relative_deformation"] for item in weak]
middle_rejection=cases["G2_R8"]["rejected"]
summary={
    "all_except_middle_grid_reach_A_0p88":all(point(cid,.88) for cid in ("G1_R8","G3_R8","G2_R10","G3_R10")),
    "middle_grid_rejected_amplitude":middle_rejection["amplitude"],
    "energy_G2_G3_R8_relative_difference_A0p5":rel(point("G2_R8",.5)["energy_dimensionless"],point("G3_R8",.5)["energy_dimensionless"]),
    "energy_R8_R10_G3_relative_difference_A0p5":rel(point("G3_R8",.5)["energy_dimensionless"],point("G3_R10",.5)["energy_dimensionless"]),
    "energy_all_case_relative_spread_A0p5":(max(energies)-min(energies))/(sum(energies)/len(energies)),
    "deformation_G2_G3_R8_relative_difference_A0p5":rel(point("G2_R8",.5)["max_relative_deformation"],point("G3_R8",.5)["max_relative_deformation"]),
    "deformation_all_case_max_to_min_A0p5":max(deformations)/min(deformations),
    "endpoint_interpretation":"single-grid termination tracks an unstabilized radion near-null mode; it is not a physical fold candidate",
    "energy_preliminary_acceptance_A0p5":False,
    "energy_acceptance_reason":"all-case spread exceeds 1% and no monotone grid/domain convergence is established",
    "pointwise_deformation_accepted":False
}
Path("results/v1_replication_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
print(json.dumps(summary,indent=2,sort_keys=True))
