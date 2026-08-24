#!/usr/bin/env python3
"""Transfer the corrected fold across domains and reclassify its cap pair."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_capped_surface import anisotropic_capped_area_stability,find_anisotropic_donor_capped_surfaces
from bhps.anisotropic_capped_surface_fd import solve_anisotropic_capped_surface_fd
from bhps.anisotropic_geometry import anisotropic_scalar_gradient_energy
from bhps.anisotropic_initial_data import solve_anisotropic_initial_data
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice
from bhps.physical_corner_corrector import combine_shape_modes,physical_corner_state,tracefree_shape_basis
from bhps.scalar_pulse import scalar_pulse


g5=json.loads(Path("results/corrected_anisotropic_arclength.json").read_text())
g6=json.loads(Path("results/corrected_anisotropic_arclength_G6.json").read_text())
fold_g5=float(g5["summary"]["fine_fold_amplitude"])
fold_g6=float(g6["summary"]["fine_fold_amplitude"])
state_path="results/corrected_family_knot_A8_state.npz"
archive=np.load(state_path);coefficients=archive["coefficients"]


def selected_geometry(name,nz,nr,r_max,amplitude):
    reference=solve_finite_wall_high_order_slice(
        amplitude,nz=nz,nr=nr,r_max=r_max,wall_stiffness=20.,epsilon=.1,
        backreaction=.01,tolerance=1e-10,iterations=240,
    )
    chi,chi_r,chi_z=scalar_pulse(reference["z"],reference["r"],amplitude)
    modes=tracefree_shape_basis(
        reference["z"],reference["r"],6,(.5,1.),8.,
        ((7.5,1.5),(7.5,3.0)),
    )["modes"]
    a,b,c=combine_shape_modes(coefficients,modes)
    selected=solve_anisotropic_initial_data(
        reference["z"],reference["r"],reference["q"],reference["phi"],
        a,b,c,reference["background"],chi_r,chi_z,
        initial_q=archive.get(f"q_{name}"),initial_phi=archive.get(f"phi_{name}"),
        stencil_width=7,tolerance=1e-9,iterations=30,
    )
    corner=physical_corner_state(
        reference["z"],reference["r"],selected["q"],selected["phi"],
        a,b,c,reference["background"],chi_r,chi_z,chi,None,7,7,True,
    )
    return reference,selected,corner,a,b,c,chi_r,chi_z


transfer=[]
for name,nz,nr,r_max,amplitude in (
    ("G5R8",49,73,8.,fold_g5),("G6R8",65,97,8.,fold_g6),
    ("G5R10",49,91,10.,fold_g5),("G5R12",49,109,12.,fold_g5),
):
    reference,selected,corner,a,b,c,chi_r,chi_z=selected_geometry(
        name,nz,nr,r_max,amplitude,
    )
    transfer.append({
        "name":name,"grid_size":[nz,nr],"r_max":r_max,"amplitude":amplitude,
        "last_audited_radius":float(reference["r"][-8]),
        "selector_maximum":selected["maximum_residual"],
        "corner_maximum":corner["maximum_intrinsic_residual"],
        "energy_dimensionless":anisotropic_scalar_gradient_energy(
            reference["z"],reference["r"],corner["psi"],a,b,c,chi_r,chi_z,
        ),
    })


stability=[]
for name,nz,nr in (("G5R8",49,73),("G6R8",65,97)):
    amplitude=7.94
    reference,selected,corner,a,b,c,chi_r,chi_z=selected_geometry(
        name,nz,nr,8.,amplitude,
    )
    caps=find_anisotropic_donor_capped_surfaces(
        reference["z"],reference["r"],corner["psi"],a,b,c,
        guesses=tuple(np.linspace(1.25,1.68,18)),tolerance=2e-5,
    )
    surfaces=[]
    for item in sorted(caps["accepted"],key=lambda value:value["rho_brane"]):
        finite_difference=solve_anisotropic_capped_surface_fd(
            reference["z"],reference["r"],corner["psi"],a,b,c,
            item["rho_brane"],nodes=121,tolerance=1e-10,max_evaluations=10000,
        )
        classification=anisotropic_capped_area_stability(
            reference["z"],reference["r"],corner["psi"],a,b,c,
            finite_difference,nodes=41,relative_step=2.5e-5,maximum_angular_mode=3,
        )
        surfaces.append({
            "rho_brane":item["rho_brane"],
            "collocation_residual":item["surface_residual_max"],
            "fd_converged":finite_difference["converged"],
            "fd_residual":finite_difference["discrete_residual_max"],
            "negative_mode_count":classification["negative_mode_count"],
            "lowest_normalized_jacobi_eigenvalue":classification["lowest_normalized_jacobi_eigenvalue"],
            "angular_mode_spectrum":classification["angular_mode_spectrum"],
        })
    stability.append({
        "name":name,"grid_size":[nz,nr],"amplitude":amplitude,
        "distance_above_own_fold":amplitude-(fold_g5 if name=="G5R8" else fold_g6),
        "corner_maximum":corner["maximum_intrinsic_residual"],"surfaces":surfaces,
    })

acceptance={
    "all_transfer_corners_pass_0.025":all(item["corner_maximum"]<.025 for item in transfer),
    "both_grids_have_two_caps":all(len(item["surfaces"])==2 for item in stability),
    "expected_inner_unstable_outer_stable":all(
        [surface["negative_mode_count"] for surface in item["surfaces"]]==[1,0]
        for item in stability
    ),
    "no_extra_negative_modes_l1_to_l3":all(
        all(
            all(mode["negative_mode_count"]==0 for mode in surface["angular_mode_spectrum"][1:])
            for surface in item["surfaces"]
        ) for item in stability
    ),
}
payload={
    "status":"corrected_fold_grid_domain_transfer_and_stability_pass"
    if all(acceptance.values()) else "corrected_fold_transfer_or_stability_incomplete",
    "shape_state":state_path,
    "arclength_results":[
        "results/corrected_anisotropic_arclength.json",
        "results/corrected_anisotropic_arclength_G6.json",
    ],
    "fold_amplitudes":{"G5R8":fold_g5,"G6R8":fold_g6},
    "transfer":transfer,"stability":stability,"acceptance":acceptance,
    "interpretation":"The A=8 compatible shape knot transfers to both larger domains at the pseudo-arclength fold amplitude and remains below the 0.025 spatial-corner threshold. Immediately above the fold, both grids retain the one-negative-mode inner cap and stable outer cap, with no extra negative angular mode through l=3.",
    "limitations":[
        "fixed local A=8 shape knot rather than a densely continued coefficient curve",
        "stability sampled at A=7.94 rather than exactly at the degenerate fold",
        "last seven radial stencil points excluded",
        "not a nonlinear spacetime evolution",
    ],
}
Path("results/corrected_fold_transfer_stability.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps({"status":payload["status"],"acceptance":acceptance,"transfer":transfer},indent=2))
