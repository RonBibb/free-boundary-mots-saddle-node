#!/usr/bin/env python3
"""Bounded lapse-only pilot for the spatial Israel second-corner failure."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.finite_wall_solver import solve_finite_wall_slice
from bhps.lapse_corner_repair import repair_spatial_corners_with_lapse
from bhps.scalar_pulse import scalar_pulse


amplitude=8.415541903059392
solved=solve_finite_wall_slice(
    amplitude,nz=33,nr=49,r_max=8.,wall_stiffness=20.,epsilon=.1,
    backreaction=.01,tolerance=1e-10,iterations=180,
)
chi,chi_r,chi_z=scalar_pulse(solved["z"],solved["r"],amplitude)
settings=(
    {"id":"conservative","regularization":.1,"coefficient_bound":.1},
    {"id":"moderate","regularization":.01,"coefficient_bound":.15},
)
cases=[]
for setting in settings:
    result=repair_spatial_corners_with_lapse(
        solved["z"],solved["r"],solved["psi"],solved["phi"],chi_r,chi_z,
        solved["background"],chi=chi,compact_modes=4,radial_modes=8,
        regularization=setting["regularization"],
        coefficient_bound=setting["coefficient_bound"],
        maximum_function_evaluations=15,
    )
    cases.append({
        **setting,
        "optimizer_success":result["success"],
        "optimizer_message":result["message"],
        "function_evaluations":result["function_evaluations"],
        "coefficient_count":result["coefficient_count"],
        "maximum_absolute_coefficient":float(np.max(np.abs(result["coefficients"]))),
        "initial_physical_residual_l2":result["initial_physical_residual_l2"],
        "final_physical_residual_l2":result["final_physical_residual_l2"],
        "initial_maximum_fixed_scaled_residual":result["initial_maximum_fixed_scaled_residual"],
        "final_maximum_fixed_scaled_residual":result["final_maximum_fixed_scaled_residual"],
        "maximum_absolute_log_lapse_correction":result["maximum_absolute_log_lapse_correction"],
        "minimum_lapse":result["minimum_lapse"],"maximum_lapse":result["maximum_lapse"],
        "maximum_discrete_wall_derivative_of_log_correction":result["maximum_discrete_wall_derivative_of_log_correction"],
        "final_intrinsic_tangential_normalized_residual":result["corner_audit"]["maximum_tangential_normalized_residual"],
        "final_intrinsic_mixed_normalized_acceleration":result["corner_audit"]["maximum_mixed_zr_normalized_acceleration"],
        "accepted":bool(
            result["final_maximum_fixed_scaled_residual"]<.02
            and result["maximum_absolute_log_lapse_correction"]<.5
            and result["maximum_discrete_wall_derivative_of_log_correction"]<.01
        ),
    })

payload={
    "status":"bounded_lapse_only_pilot_rescales_but_cannot_repair_geometric_second_corner",
    "slice":{
        "solver":"primary second-order finite-wall solver","grid_size":[33,49],
        "r_max":8.,"amplitude":amplitude,
        "amplitude_source":"G3_R8 fold in results/finite_wall_refinement.json",
        "energy_dimensionless":solved["energy_dimensionless"],
    },
    "lapse_family":"alpha=psi exp(u), u expanded in compact Neumann cosines and radial half-integer cosines",
    "fixed_scale_rule":"all optimization residual scales are frozen at the uncorrected slice",
    "acceptance":"maximum fixed-scaled corner residual <0.02, max abs log lapse correction <0.5, wall derivative defect <0.01",
    "cases":cases,
    "any_case_accepted":bool(any(case["accepted"] for case in cases)),
    "interpretation":[
        "Both bounded lapse families reduce the spatial corner residual but neither closes it.",
        "The fixed baseline normalization prevents an optimizer from appearing successful by generating enormous metric accelerations.",
        "The subsequent tensor-covariance audit shows that this reduction is local coordinate-time slowing rather than partial compatibility repair.",
        "A regular lapse and wall-preserving shift cannot remove a nonzero geometric Israel second-corner tensor.",
        "The next repair must add compatibility-corrected gravitational initial data and re-solve the constraints and junction conditions.",
    ],
    "limitations":[
        "single coarse fold-scale slice",
        "finite smooth lapse basis",
        "optimizer evaluation limit reached in admitted cases",
        "time-time Israel and normal harmonic rows not included",
    ],
}
Path("results/lapse_corner_repair.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps(payload,indent=2))
