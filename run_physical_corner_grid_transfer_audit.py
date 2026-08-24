#!/usr/bin/env python3
"""Transfer one physical correction across grids without refitting its shape."""

import json,sys,time
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_initial_data import solve_anisotropic_initial_data
from bhps.finite_wall_solver import solve_finite_wall_slice
from bhps.physical_corner_corrector import combine_shape_modes,physical_corner_state,tracefree_shape_basis
from bhps.scalar_pulse import scalar_pulse


amplitude=8.415541903059392
source=np.load("results/full_spatial_corner_repair_fine_continuation_state.npz")
coefficients=source["coefficients"]


def fixed_row_maxima(state,base_scales):
    answer={};offset=0
    for wall in state["fields"]["walls"]:
        answer[wall["wall"]]={}
        for name in ("radial","transverse"):
            values=wall["tangential_components"][name]["residual"]
            scale=base_scales[offset:offset+values.size];offset+=values.size
            answer[wall["wall"]][name]=float(np.max(np.abs(values)/scale))
        values=wall["mixed_zr_residual"]
        scale=base_scales[offset:offset+values.size];offset+=values.size
        answer[wall["wall"]]["mixed_zr"]=float(np.max(np.abs(values)/scale))
    return answer


records=[]
for nz,nr in ((33,49),(49,73),(65,97)):
    started=time.time()
    reference=solve_finite_wall_slice(
        amplitude,nz=nz,nr=nr,r_max=8.,wall_stiffness=20.,epsilon=.1,
        backreaction=.01,tolerance=1e-10,iterations=240,
    )
    z=reference["z"];r=reference["r"]
    chi,chi_r,chi_z=scalar_pulse(z,r,amplitude)
    modes=tracefree_shape_basis(z,r,6)["modes"]
    a,b,c=combine_shape_modes(coefficients,modes)
    selector=solve_anisotropic_initial_data(
        z,r,reference["q"],reference["phi"],a,b,c,reference["background"],
        chi_r,chi_z,stencil_width=7,tolerance=1e-9,iterations=35,
    )
    if not selector["converged"]:
        raise RuntimeError(f"selector failed on {nz}x{nr}: {selector['maximum_residual']}")
    zero=np.zeros_like(reference["q"])
    base=physical_corner_state(
        z,r,reference["q"],reference["phi"],zero,zero,zero,
        reference["background"],chi_r,chi_z,chi,None,7,7,True,
    )
    final=physical_corner_state(
        z,r,selector["q"],selector["phi"],a,b,c,reference["background"],
        chi_r,chi_z,chi,base["scales"],7,7,True,
    )
    record={
        "grid_size":[nz,nr],"selector_maximum_residual":selector["maximum_residual"],
        "reference_full_spatial_maximum":base["maximum_fixed_scaled_residual"],
        "transferred_full_spatial_maximum":final["maximum_fixed_scaled_residual"],
        "transferred_full_spatial_l2":final["fixed_scaled_residual_l2"],
        "row_maxima":fixed_row_maxima(final,base["scales"]),
        "maximum_shape_logarithm":float(max(np.max(np.abs(a)),np.max(np.abs(b)),np.max(np.abs(c)))),
        "maximum_log_conformal_change":float(np.max(np.abs(np.log(selector["psi"]/base["psi"])))),
        "maximum_stabilizer_change":float(np.max(np.abs(selector["phi"]-reference["phi"]))),
        "elapsed_seconds":time.time()-started,
    }
    records.append(record);print(record,flush=True)

payload={
    "status":"fixed_shape_grid_transfer_fails" if max(item["transferred_full_spatial_maximum"] for item in records)>.02 else "fixed_shape_grid_transfer_passes_0.02",
    "source_state":"results/full_spatial_corner_repair_fine_continuation_state.npz",
    "settings":{"radial_modes":6,"stencil_width":7,"radial_buffer":7,"fixed_shape_coefficients":True},
    "records":records,
    "summary":{
        "all_selector_residuals_below_1e-8":all(item["selector_maximum_residual"]<1e-8 for item in records),
        "maximum_transferred_residual":max(item["transferred_full_spatial_maximum"] for item in records),
        "minimum_transferred_residual":min(item["transferred_full_spatial_maximum"] for item in records),
    },
    "interpretation":"A grid-portable physical correction must retain decreasing or acceptably small residuals without refitting its shape coefficients.",
}
Path("results/physical_corner_grid_transfer_audit.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps({"status":payload["status"],"summary":payload["summary"]},indent=2))
