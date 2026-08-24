#!/usr/bin/env python3
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from bhps.initial_data import solve
from bhps.capped_surface import find_donor_capped_surfaces

amps=[0,1,2,3,3.4,3.5,3.54,3.56]
previous=None; records=[]
for amplitude in amps:
    data=solve(amplitude,initial=previous)
    if not data["converged"]:
        records.append({"amplitude":amplitude,"constraint_converged":False,"residual":data["max_abs_residual"]});break
    previous=data["psi"]
    caps=find_donor_capped_surfaces(data["z"],data["r"],data["psi"])
    records.append({"amplitude":amplitude,"constraint_converged":True,"residual":data["max_abs_residual"],"capped_surface_found":caps["capped_surface_found"],"accepted":caps["accepted"],"successful_trials":caps["successful_trials"],"in_domain_successful_trials":caps["in_domain_successful_trials"]})
payload={"grid":[17,25],"d_over_ell":1,"donor_width":.75,"records":records}
out=Path("results/capped_scan.json");out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");print(json.dumps(payload,indent=2,sort_keys=True))
