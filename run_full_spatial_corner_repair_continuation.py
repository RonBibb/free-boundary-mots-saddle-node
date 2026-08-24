#!/usr/bin/env python3
"""Continue the physical repair with tangential and mixed spatial rows."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_geometry import anisotropic_spatial_junction_fields
from bhps.finite_wall_solver import solve_finite_wall_slice
from bhps.physical_corner_corrector import physical_corner_state,solve_relinearized_physical_corner
from bhps.scalar_pulse import scalar_pulse


amplitude=8.415541903059392
reference=solve_finite_wall_slice(
    amplitude,nz=33,nr=49,r_max=8.,wall_stiffness=20.,epsilon=.1,
    backreaction=.01,tolerance=1e-10,iterations=180,
)
parent_path=Path("results/relinearized_corner_repair_continuation.json")
parent=json.loads(parent_path.read_text())
archive=np.load("results/relinearized_corner_repair_continuation_state.npz")
chi,chi_r,chi_z=scalar_pulse(reference["z"],reference["r"],amplitude)
result=solve_relinearized_physical_corner(
    reference["z"],reference["r"],reference["q"],reference["phi"],
    reference["background"],chi_r,chi_z,chi=chi,radial_modes=6,
    stencil_width=7,radial_buffer=7,finite_difference_step=2e-4,
    maximum_iterations=10,corner_tolerance=.01,shape_bound=.5,
    initial_trust_radius=.0168,regularization=1.,selector_tolerance=1e-9,
    difference_scheme="forward",maximum_row_weight=4.,verbose=True,
    initial_coefficients=archive["coefficients"],initial_q=archive["q"],
    initial_phi=archive["phi"],include_mixed=True,
)

zero=np.zeros_like(reference["q"]);z=reference["z"];r=reference["r"]
base_state=physical_corner_state(
    z,r,reference["q"],reference["phi"],zero,zero,zero,
    reference["background"],chi_r,chi_z,chi,None,7,7,True,
)
final_state=physical_corner_state(
    z,r,result["q"],result["phi"],result["a"],result["b"],result["c"],
    reference["background"],chi_r,chi_z,chi,base_state["scales"],7,7,True,
)

def row_maxima(state):
    answer={}
    for wall in state["fields"]["walls"]:
        answer[wall["wall"]]={
            name:float(np.max(np.abs(item["residual"])/item["scale"]))
            for name,item in wall["tangential_components"].items()
        }
        answer[wall["wall"]]["mixed_zr"]=float(np.max(
            np.abs(wall["mixed_zr_residual"])/wall["mixed_zr_scale"]
        ))
    return answer

base_junction=anisotropic_spatial_junction_fields(
    z,r,base_state["psi"],zero,zero,zero,reference["phi"],reference["background"],7,
)
final_junction=anisotropic_spatial_junction_fields(
    z,r,result["psi"],result["a"],result["b"],result["c"],result["phi"],
    reference["background"],7,
)
junction={}
for base_wall,final_wall in zip(base_junction["walls"],final_junction["walls"]):
    junction[base_wall["wall"]]={}
    for name in ("radial","transverse"):
        base_values=base_wall[name][:-7];final_values=final_wall[name][:-7]
        junction[base_wall["wall"]][name]={
            "reference_maximum_absolute":float(np.max(np.abs(base_values))),
            "final_maximum_absolute":float(np.max(np.abs(final_values))),
            "maximum_absolute_change":float(np.max(np.abs(final_values-base_values))),
        }

payload={
    "status":"full_spatial_corner_pilot_passes" if result["converged"] else "full_spatial_corner_pilot_does_not_yet_pass",
    "parent_result":str(parent_path),
    "slice":parent["slice"],"settings":result["settings"],
    "history":result["history"],"linearizations":result["linearizations"],
    "summary":{
        "converged":result["converged"],
        "reference_full_spatial_maximum":base_state["maximum_fixed_scaled_residual"],
        "continuation_initial_full_spatial_maximum":result["history"][0]["maximum_fixed_scaled_corner_residual"],
        "final_full_spatial_maximum":final_state["maximum_fixed_scaled_residual"],
        "final_full_spatial_l2":final_state["fixed_scaled_residual_l2"],
        "final_selector_maximum_residual":result["final_selector_maximum_residual"],
        "maximum_shape_logarithm":result["maximum_shape_logarithm"],
        "maximum_log_conformal_change":result["maximum_log_conformal_change"],
        "maximum_stabilizer_change":result["maximum_stabilizer_change"],
        "coefficient_l2":float(np.linalg.norm(result["coefficients"])),
        "maximum_absolute_coefficient":float(np.max(np.abs(result["coefficients"]))),
    },
    "row_maxima":{"reference":row_maxima(base_state),"final":row_maxima(final_state)},
    "zeroth_order_junction":junction,
    "acceptance":{
        "all_tangential_and_mixed_rows_below_0.01":bool(final_state["maximum_fixed_scaled_residual"]<.01),
        "selector_residual_below_1e-8":bool(result["final_selector_maximum_residual"]<1e-8),
        "shape_logarithm_below_0.5":bool(result["maximum_shape_logarithm"]<.5),
    },
    "limitations":[
        "single coarse fold slice",
        "finite 48-mode physical basis",
        "fixed reference-grid normalization",
        "time-time and harmonic corner rows remain",
        "independent discretization and grid/domain refinement remain",
        "the capped surface has not yet been recomputed in the anisotropic metric",
    ],
}
Path("results/full_spatial_corner_repair_continuation.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
np.savez_compressed(
    "results/full_spatial_corner_repair_continuation_state.npz",
    coefficients=result["coefficients"],q=result["q"],phi=result["phi"],
    psi=result["psi"],a=result["a"],b=result["b"],c=result["c"],z=z,r=r,
)
print(json.dumps({"status":payload["status"],"summary":payload["summary"],"row_maxima":payload["row_maxima"],"junction":junction},indent=2))
