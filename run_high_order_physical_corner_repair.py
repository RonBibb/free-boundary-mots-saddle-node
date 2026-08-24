#!/usr/bin/env python3
"""Fit the physical spatial-corner correction on the independent fold slice."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_initial_data import solve_anisotropic_initial_data
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice
from bhps.physical_corner_corrector import combine_shape_modes,solve_relinearized_physical_corner,tracefree_shape_basis
from bhps.scalar_pulse import scalar_pulse


amplitude=8.572038845434301
reference=solve_finite_wall_high_order_slice(
    amplitude,nz=49,nr=73,r_max=8.,wall_stiffness=20.,epsilon=.1,
    backreaction=.01,tolerance=1e-10,iterations=220,
)
chi,chi_r,chi_z=scalar_pulse(reference["z"],reference["r"],amplitude)
seed=np.load("results/full_spatial_corner_repair_fine_continuation_state.npz")
modes=tracefree_shape_basis(reference["z"],reference["r"],6)["modes"]
a,b,c=combine_shape_modes(seed["coefficients"],modes)
initial=solve_anisotropic_initial_data(
    reference["z"],reference["r"],reference["q"],reference["phi"],a,b,c,
    reference["background"],chi_r,chi_z,stencil_width=7,tolerance=1e-9,iterations=40,
)
if not initial["converged"]:raise RuntimeError("seed selector solve failed")
result=solve_relinearized_physical_corner(
    reference["z"],reference["r"],reference["q"],reference["phi"],
    reference["background"],chi_r,chi_z,chi=chi,radial_modes=6,
    stencil_width=7,radial_buffer=7,finite_difference_step=2e-4,
    maximum_iterations=10,corner_tolerance=.02,shape_bound=.5,
    initial_trust_radius=.04,regularization=1.,selector_tolerance=1e-9,
    difference_scheme="forward",maximum_row_weight=4.,verbose=True,
    initial_coefficients=seed["coefficients"],initial_q=initial["q"],
    initial_phi=initial["phi"],include_mixed=True,
)
payload={
    "status":"independent_high_order_physical_corner_pilot_passes" if result["converged"] else "independent_high_order_physical_corner_pilot_does_not_yet_pass",
    "slice":{
        "solver":"independent five-point coupled finite-wall solver",
        "grid_size":[49,73],"r_max":8.,"fold_amplitude":amplitude,
        "reference_elliptic_residual":reference["max_abs_residual"],
    },
    "settings":result["settings"],"history":result["history"],
    "linearizations":result["linearizations"],
    "summary":{
        "converged":result["converged"],
        "initial_full_spatial_maximum":result["history"][0]["maximum_fixed_scaled_corner_residual"],
        "final_full_spatial_maximum":result["final_maximum_fixed_scaled_corner_residual"],
        "final_full_spatial_l2":result["final_corner_residual_l2"],
        "final_selector_maximum_residual":result["final_selector_maximum_residual"],
        "maximum_shape_logarithm":result["maximum_shape_logarithm"],
        "maximum_log_conformal_change":result["maximum_log_conformal_change"],
        "maximum_stabilizer_change":result["maximum_stabilizer_change"],
        "coefficient_l2":float(np.linalg.norm(result["coefficients"])),
        "maximum_absolute_coefficient":float(np.max(np.abs(result["coefficients"]))),
    },
    "limitations":[
        "single independent fold grid",
        "finite 48-mode physical basis",
        "grid transfer and an independent anisotropic discretization remain",
        "time-time and harmonic corner rows remain",
        "the capped surface has not yet been recomputed in the anisotropic metric",
    ],
}
Path("results/high_order_physical_corner_repair.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
np.savez_compressed(
    "results/high_order_physical_corner_repair_state.npz",
    coefficients=result["coefficients"],q=result["q"],phi=result["phi"],
    psi=result["psi"],a=result["a"],b=result["b"],c=result["c"],
    z=reference["z"],r=reference["r"],
)
print(json.dumps({"status":payload["status"],"summary":payload["summary"]},indent=2))
