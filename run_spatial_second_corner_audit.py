#!/usr/bin/env python3
"""Spatial nonlinear second-corner audit for selected finite-wall slices."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.adm_corner import spatial_israel_second_corner_audit,time_symmetric_adm_metric_acceleration
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice
from bhps.finite_wall_solver import solve_finite_wall_slice
from bhps.scalar_pulse import scalar_pulse


def solve_and_audit(solver,amplitude,size,solver_name,radial_size=None,r_max=8.):
    # Match the production refinement grids exactly: 33x49, 49x73, ...
    radial_size=int(1.5*size) if radial_size is None else int(radial_size)
    solved=solver(
        amplitude,nz=size,nr=radial_size,r_max=float(r_max),wall_stiffness=20.,
        epsilon=.1,backreaction=.01,tolerance=1e-10,iterations=180,
    )
    chi,chi_r,chi_z=scalar_pulse(solved["z"],solved["r"],amplitude)
    acceleration=time_symmetric_adm_metric_acceleration(
        solved["z"],solved["r"],solved["psi"],solved["phi"],chi_r,chi_z,
        solved["background"]["mass_squared"],chi=chi,stencil_width=7,
    )
    corner=spatial_israel_second_corner_audit(
        acceleration,solved["psi"],solved["phi"],solved["background"],
        stabilizer_acceleration=np.zeros_like(solved["psi"]),radial_buffer=7,
    )
    for wall in corner["walls"]:
        for component in wall["tangential_components"].values():
            component["r_at_maximum_normalized_residual"]=float(
                solved["r"][component["maximum_index_within_retained_radial_grid"]]
            )
    return {
        "solver":solver_name,"grid_size":[size,radial_size],"r_max":float(r_max),
        "amplitude":float(amplitude),"energy_dimensionless":solved["energy_dimensionless"],
        "elliptic_residual":solved["max_abs_residual"],
        "maximum_absolute_metric_acceleration":acceleration["maximum_absolute_acceleration"],
        **corner,
    }


background_controls=[
    solve_and_audit(solve_finite_wall_slice,0.,size,"primary_second_order")
    for size in (33,49,65,81)
]
weak_pulse=[
    solve_and_audit(solve_finite_wall_slice,.5,size,"primary_second_order")
    for size in (33,49,65,81)
]
primary_fold=[
    solve_and_audit(solve_finite_wall_slice,amplitude,size,"primary_second_order")
    for size,amplitude in (
        (33,8.415541903059392),(49,8.501156129278852),
        (65,8.53235655905242),(81,8.546887772929086),
    )
]
independent_fold=[
    solve_and_audit(solve_finite_wall_high_order_slice,amplitude,size,"independent_five_point")
    for size,amplitude in (
        (33,8.57071355055347),(49,8.572038845434301),(65,8.572368541895216),
    )
]
primary_domain_fold=[
    solve_and_audit(solve_finite_wall_slice,amplitude,49,"primary_second_order",radial_size,r_max)
    for radial_size,r_max,amplitude in (
        (73,8.,8.501156129278852),(91,10.,8.518814140892566),(109,12.,8.531582358007602),
    )
]
independent_domain_fold=[
    solve_and_audit(solve_finite_wall_high_order_slice,amplitude,49,"independent_five_point",radial_size,r_max)
    for radial_size,r_max,amplitude in (
        (73,8.,8.572038845434301),(91,10.,8.591525244593445),(109,12.,8.605669112233226),
    )
]

def finest_max(cases):return cases[-1]["maximum_tangential_normalized_residual"]

payload={
    "status":"existing_localized_slice_family_fails_spatial_israel_second_corner_in_alpha_equals_psi_zero_shift_adm_gauge",
    "assumptions":[
        "four spatial dimensions",
        "time symmetry K_ij=0 and zero scalar momenta",
        "zero shift and initial lapse alpha=psi",
        "bulk ADM acceleration from Einstein--two-scalar equations",
        "spatial tangential Israel rows and mixed z-r wall gauge only",
    ],
    "fold_amplitudes_source":{
        "primary":"results/finite_wall_refinement.json, R8 cases",
        "independent":"results/finite_wall_high_order_replication.json, R8 cases",
    },
    "background_controls":background_controls,
    "weak_pulse_controls":weak_pulse,
    "primary_fold_scale":primary_fold,
    "independent_high_order_fold_scale":independent_fold,
    "primary_domain_fold_scale":primary_domain_fold,
    "independent_domain_fold_scale":independent_domain_fold,
    "summary":{
        "background_finest_normalized_residual":finest_max(background_controls),
        "weak_pulse_finest_normalized_residual":finest_max(weak_pulse),
        "primary_fold_finest_normalized_residual":finest_max(primary_fold),
        "independent_fold_finest_normalized_residual":finest_max(independent_fold),
        "primary_largest_domain_normalized_residual":finest_max(primary_domain_fold),
        "independent_largest_domain_normalized_residual":finest_max(independent_domain_fold),
        "background_control_converges_toward_zero":bool(
            finest_max(background_controls)<background_controls[0]["maximum_tangential_normalized_residual"]/5
        ),
        "fold_corner_failure_replicated":bool(
            finest_max(primary_fold)>.05 and finest_max(independent_fold)>.05
        ),
    },
    "interpretation":[
        "The Hamiltonian-plus-stabilizer selector supplies valid constraint data and zeroth-order wall junction data, but not smooth second-corner evolution data in this lapse/shift choice.",
        "The mixed z-r gauge acceleration decreases under refinement, while the tangential Israel residual remains order 0.1 at fold scale.",
        "The scoped static apparent-horizon fold is unchanged; the failure blocks direct use of these slices as smooth nonlinear evolution data.",
        "Repair requires a boundary-compatible lapse/gauge solve or additional non-conformal gravitational initial-data freedom, followed by re-solving the constraints.",
    ],
    "limitations":[
        "does not audit the time-time Israel row or normal-normal harmonic gauge row",
        "diagnostic uses high-order numerical derivatives of discrete elliptic solutions",
        "a different lapse/gauge or compatibility-corrected initial-data family may pass",
        "no evolution was attempted",
    ],
}
Path("results/spatial_second_corner_audit.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps({"status":payload["status"],"summary":payload["summary"]},indent=2))
