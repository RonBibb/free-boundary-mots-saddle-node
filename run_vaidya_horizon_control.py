#!/usr/bin/env python3
"""Generate event-versus-apparent-horizon controls for the pipeline."""

import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.vaidya_horizon_control import outgoing_expansion,thin_shell_apparent_horizon,thin_shell_event_horizon,thin_shell_mass,trace_smooth_event_horizon


mass=1.;shell=0.;v=np.linspace(-4.5,2.,651)
event=thin_shell_event_horizon(v,mass,shell);apparent=thin_shell_apparent_horizon(v,mass,shell)
pre=(v>=-4*mass)&(v<shell)
thin={
    "final_mass":mass,"shell_time":shell,"event_horizon_birth_time":-4*mass,
    "event_horizon_preexists_local_collapse_by":4*mass,
    "minimum_pre_shell_event_horizon_outgoing_expansion":float(np.min(outgoing_expansion(event[pre][1:],thin_shell_mass(v[pre][1:],mass,shell)))),
    "pre_shell_apparent_horizon_exists":bool(np.any(np.isfinite(apparent[pre]))),
    "post_shell_horizons_coincide":bool(np.allclose(event[v>shell],apparent[v>shell])),
}
smooth=trace_smooth_event_horizon(final_mass=1.,start=0.,duration=1.)
payload={
    "status":"Vaidya_control_distinguishes_global_event_horizon_from_slice_local_marginal_surface",
    "thin_shell":thin,
    "smooth_collapse":{
        "event_horizon_birth_time":smooth["birth_time"],"collapse_start":smooth["collapse_start"],
        "collapse_end":smooth["collapse_end"],"minimum_event_minus_apparent":smooth["minimum_event_minus_apparent"],
        "function_evaluations":smooth["function_evaluations"],
        "sample":[
            {"v":float(vv),"mass":float(mm),"event_radius":float(ee),"apparent_radius":None if not np.isfinite(aa) else float(aa)}
            for vv,mm,ee,aa in zip(smooth["v"][::50],smooth["mass"][::50],smooth["event_radius"][::50],smooth["apparent_radius"][::50])
        ],
    },
    "pipeline_rule":"forward evolution tracks marginal/apparent horizons; an event horizon is reconstructed backward only after a sufficiently late final spacetime is available",
    "limitations":["four-dimensional spherical control spacetime","prescribed Vaidya matter","not a braneworld evolution"],
}
Path("results/vaidya_horizon_control.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2))
