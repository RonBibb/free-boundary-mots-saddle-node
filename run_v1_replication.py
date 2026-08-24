#!/usr/bin/env python3
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from bhps.v1_solver import continue_v1

amplitudes=[0,.5,.75,.82,.84,.85,.86,.865,.87,.875,.88]
cases=[
    {"id":"G1_R8","nz":17,"nr":25,"r_max":8.0},
    {"id":"G2_R8","nz":25,"nr":37,"r_max":8.0},
    {"id":"G3_R8","nz":33,"nr":49,"r_max":8.0},
    {"id":"G2_R10","nz":25,"nr":46,"r_max":10.0},
    {"id":"G3_R10","nz":33,"nr":61,"r_max":10.0}
]
output=[]
for case in cases:
    branch=continue_v1(amplitudes,nz=case["nz"],nr=case["nr"],r_max=case["r_max"],tolerance=1e-9,iterations=120)
    output.append({**case,"accepted":branch["accepted"],"rejected":branch["rejected"]})
payload={"amplitudes":amplitudes,"acceptance_residual":1e-9,"cases":output}
p=Path("results/v1_replication.json");p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");print(json.dumps(payload,indent=2,sort_keys=True))
