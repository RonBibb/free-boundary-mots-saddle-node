#!/usr/bin/env python3
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from bhps.radion_effective import gw_control_point

points=[]
for epsilon in (.01,.025,.05,.075,.1):
    for b0 in (1e-4,1e-3,.01,.03,.1):
        points.append(gw_control_point(epsilon,b0,1.,1.))
maximum=max(points,key=lambda item:item["mu_rad_times_source_width"])
payload={
    "model":"leading stiff-wall probe Goldberger-Wise effective potential",
    "separation_over_ell":1.,
    "source_width_over_ell":1.,
    "scan_bounds":{"epsilon_max":.1,"b0_max":.1},
    "maximum":maximum,
    "all_points_fail_even_mu_sigma_gt_1":not any(item["necessary_heavy_condition_mu_sigma_gt_1"] for item in points),
    "interpretation":"the frozen-radion approximation is not justified on the V1 source-width scale within this leading weak-backreaction scan",
    "limitations":[
        "leading small-epsilon potential",
        "stiff scalar boundary potentials",
        "no finite-backreaction correction",
        "no coupled 5D radion-scalar-metric spectrum"
    ],
    "points":points,
}
path=Path("results/radion_control_scan.json")
path.parent.mkdir(exist_ok=True)
path.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2,sort_keys=True))
