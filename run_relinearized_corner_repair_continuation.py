#!/usr/bin/env python3
"""Continue the relinearized corner repair from its archived nonlinear state."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.finite_wall_solver import solve_finite_wall_slice
from bhps.physical_corner_corrector import solve_relinearized_physical_corner
from bhps.scalar_pulse import scalar_pulse


amplitude=8.415541903059392
reference=solve_finite_wall_slice(
    amplitude,nz=33,nr=49,r_max=8.,wall_stiffness=20.,epsilon=.1,
    backreaction=.01,tolerance=1e-10,iterations=180,
)
continued_state=Path("results/relinearized_corner_repair_continuation_state.npz")
if continued_state.exists():
    archive=np.load(continued_state)
    parent_path=Path("results/relinearized_corner_repair_continuation.json")
    parent=json.loads(parent_path.read_text())
    prior_history=parent["combined_history"]
    trust_radius=float(parent["continuation_history"][-1]["trust_radius"])
else:
    archive=np.load("results/relinearized_corner_repair_state.npz")
    parent_path=Path("results/relinearized_corner_repair.json")
    parent=json.loads(parent_path.read_text())
    prior_history=parent["history"]
    trust_radius=.0258
chi,chi_r,chi_z=scalar_pulse(reference["z"],reference["r"],amplitude)
result=solve_relinearized_physical_corner(
    reference["z"],reference["r"],reference["q"],reference["phi"],
    reference["background"],chi_r,chi_z,chi=chi,radial_modes=6,
    stencil_width=7,radial_buffer=7,finite_difference_step=2e-4,
    maximum_iterations=8,corner_tolerance=.02,shape_bound=.5,
    initial_trust_radius=trust_radius,regularization=1.,selector_tolerance=1e-9,
    difference_scheme="forward",maximum_row_weight=4.,verbose=True,
    initial_coefficients=archive["coefficients"],initial_q=archive["q"],
    initial_phi=archive["phi"],
)
combined_history=prior_history+[
    {**item,"iteration":item["iteration"]+len(prior_history)}
    for item in result["history"]
]
payload={
    "status":"continued_relinearized_physical_corner_repair_passes" if result["converged"] else "continued_relinearized_physical_corner_repair_does_not_yet_pass",
    "parent_result":str(parent_path),
    "slice":parent["slice"],"settings":result["settings"],
    "continuation_history":result["history"],"combined_history":combined_history,
    "linearizations":result["linearizations"],
    "summary":{
        "converged":result["converged"],
        "original_corner_maximum":prior_history[0]["maximum_fixed_scaled_corner_residual"],
        "continuation_initial_corner_maximum":result["history"][0]["maximum_fixed_scaled_corner_residual"],
        "final_corner_maximum":result["final_maximum_fixed_scaled_corner_residual"],
        "final_corner_l2":result["final_corner_residual_l2"],
        "final_intrinsic_corner_maximum":result["final_maximum_intrinsic_corner_residual"],
        "final_selector_maximum_residual":result["final_selector_maximum_residual"],
        "maximum_shape_logarithm":result["maximum_shape_logarithm"],
        "maximum_log_conformal_change":result["maximum_log_conformal_change"],
        "maximum_stabilizer_change":result["maximum_stabilizer_change"],
        "coefficient_l2":float(np.linalg.norm(result["coefficients"])),
        "maximum_absolute_coefficient":float(np.max(np.abs(result["coefficients"]))),
        "new_accepted_shape_steps":sum(bool(item.get("step_accepted",False)) for item in result["history"]),
    },
    "acceptance":{
        "corner_maximum_below_0.02":bool(result["final_maximum_fixed_scaled_corner_residual"]<.02),
        "selector_residual_below_1e-8":bool(result["final_selector_maximum_residual"]<1e-8),
        "shape_logarithm_below_0.5":bool(result["maximum_shape_logarithm"]<.5),
    },
    "limitations":parent["limitations"],
}
Path("results/relinearized_corner_repair_continuation.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
np.savez_compressed(
    "results/relinearized_corner_repair_continuation_state.npz",
    coefficients=result["coefficients"],q=result["q"],phi=result["phi"],
    psi=result["psi"],a=result["a"],b=result["b"],c=result["c"],
    z=reference["z"],r=reference["r"],
)
print(json.dumps({"status":payload["status"],"summary":payload["summary"],"history":result["history"]},indent=2))
