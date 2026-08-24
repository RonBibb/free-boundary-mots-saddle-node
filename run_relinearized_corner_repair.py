#!/usr/bin/env python3
"""Relinearized nonlinear variable-projection repair of Israel corners."""

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
chi,chi_r,chi_z=scalar_pulse(reference["z"],reference["r"],amplitude)
result=solve_relinearized_physical_corner(
    reference["z"],reference["r"],reference["q"],reference["phi"],
    reference["background"],chi_r,chi_z,chi=chi,radial_modes=6,
    stencil_width=7,radial_buffer=7,finite_difference_step=2e-4,
    maximum_iterations=10,corner_tolerance=.02,shape_bound=.5,
    initial_trust_radius=.08,regularization=.1,selector_tolerance=1e-9,
    difference_scheme="forward",maximum_row_weight=4.,verbose=True,
)

payload={
    "status":"relinearized_physical_corner_repair_passes" if result["converged"] else "relinearized_physical_corner_repair_does_not_yet_pass",
    "slice":{
        "grid_size":[len(reference["z"]),len(reference["r"])],"r_max":8.,
        "amplitude":amplitude,"energy_dimensionless":reference["energy_dimensionless"],
        "reference_selector_residual":reference["max_abs_residual"],
    },
    "settings":result["settings"],
    "history":result["history"],"linearizations":result["linearizations"],
    "summary":{
        "converged":result["converged"],
        "initial_corner_maximum":result["history"][0]["maximum_fixed_scaled_corner_residual"],
        "final_corner_maximum":result["final_maximum_fixed_scaled_corner_residual"],
        "final_corner_l2":result["final_corner_residual_l2"],
        "final_intrinsic_corner_maximum":result["final_maximum_intrinsic_corner_residual"],
        "final_selector_maximum_residual":result["final_selector_maximum_residual"],
        "maximum_shape_logarithm":result["maximum_shape_logarithm"],
        "maximum_log_conformal_change":result["maximum_log_conformal_change"],
        "maximum_stabilizer_change":result["maximum_stabilizer_change"],
        "coefficient_l2":float(np.linalg.norm(result["coefficients"])),
        "maximum_absolute_coefficient":float(np.max(np.abs(result["coefficients"]))),
        "accepted_shape_steps":sum(bool(item.get("step_accepted",False)) for item in result["history"]),
    },
    "acceptance":{
        "corner_maximum_below_0.02":bool(result["final_maximum_fixed_scaled_corner_residual"]<.02),
        "selector_residual_below_1e-8":bool(result["final_selector_maximum_residual"]<1e-8),
        "shape_logarithm_below_0.5":bool(result["maximum_shape_logarithm"]<.5),
    },
    "interpretation":[
        "Every accepted shape update is followed by a converged nonlinear Hamiltonian--stabilizer solve.",
        "The constraint-projected corner Jacobian is rebuilt on every corrected geometry.",
        "Passing this pilot would establish a finite spatial-corner-compatible slice on one coarse production grid, not nonlinear evolution or continuum robustness.",
    ],
    "limitations":[
        "single coarse fold slice","finite 48-mode physical basis",
        "spatial tangential Israel second corner only","time-time and harmonic corner rows remain",
        "independent discretization and grid/domain refinement would be required after a pass",
    ],
}
Path("results/relinearized_corner_repair.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
np.savez_compressed(
    "results/relinearized_corner_repair_state.npz",
    coefficients=result["coefficients"],q=result["q"],phi=result["phi"],
    psi=result["psi"],a=result["a"],b=result["b"],c=result["c"],
    z=reference["z"],r=reference["r"],
)
print(json.dumps({"status":payload["status"],"summary":payload["summary"],"history":payload["history"]},indent=2))
