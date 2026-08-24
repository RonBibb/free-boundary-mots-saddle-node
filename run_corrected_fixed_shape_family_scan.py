#!/usr/bin/env python3
"""Bracket the corrected cap fold while holding the promoted shape fixed."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_capped_surface import find_anisotropic_donor_capped_surfaces
from bhps.anisotropic_geometry import anisotropic_scalar_gradient_energy
from bhps.anisotropic_initial_data import solve_anisotropic_initial_data
from bhps.capped_continuation import fit_fold_normal_form
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice
from bhps.physical_corner_corrector import combine_shape_modes,physical_corner_state,tracefree_shape_basis
from bhps.scalar_pulse import scalar_pulse


amplitudes=(7.85,7.90,7.925,7.95,7.975,8.00,8.025,8.05,8.075,8.10,8.15,8.20)
grids=(("G5R8",49,73),("G6R8",65,97))
state_path="results/high_order_physical_corner_full_exterior_solve_state.npz"
archive=np.load(state_path);coefficients=archive["coefficients"]
runs=[]
for name,nz,nr in grids:
    records=[];initial_q=initial_phi=None
    for amplitude in amplitudes:
        reference=solve_finite_wall_high_order_slice(
            amplitude,nz=nz,nr=nr,r_max=8.,wall_stiffness=20.,epsilon=.1,
            backreaction=.01,tolerance=1e-10,iterations=240,
        )
        chi,chi_r,chi_z=scalar_pulse(reference["z"],reference["r"],amplitude)
        basis=tracefree_shape_basis(
            reference["z"],reference["r"],6,(.5,1.),8.,
            ((7.5,1.5),(7.5,3.0)),
        )
        a,b,c=combine_shape_modes(coefficients,basis["modes"])
        selected=solve_anisotropic_initial_data(
            reference["z"],reference["r"],reference["q"],reference["phi"],
            a,b,c,reference["background"],chi_r,chi_z,
            initial_q=initial_q,initial_phi=initial_phi,stencil_width=7,
            tolerance=1e-9,iterations=30,
        )
        initial_q,initial_phi=selected["q"],selected["phi"]
        corner=physical_corner_state(
            reference["z"],reference["r"],selected["q"],selected["phi"],
            a,b,c,reference["background"],chi_r,chi_z,chi,None,7,7,True,
        )
        caps=find_anisotropic_donor_capped_surfaces(
            reference["z"],reference["r"],corner["psi"],a,b,c,
            guesses=tuple(np.linspace(.9,1.9,21)),tolerance=3e-5,
        )
        radii=sorted(float(item["rho_brane"]) for item in caps["accepted"])
        pair=len(radii)==2
        records.append({
            "amplitude":amplitude,
            "energy_dimensionless":anisotropic_scalar_gradient_energy(
                reference["z"],reference["r"],corner["psi"],a,b,c,chi_r,chi_z,
            ),
            "selector_maximum":selected["maximum_residual"],
            "corner_maximum":corner["maximum_intrinsic_residual"],
            "corner_passes_0.025":corner["maximum_intrinsic_residual"]<.025,
            "cap_count":len(radii),"cap_rho_brane":radii,
            "pair_converged":pair,
            "radius_separation":radii[1]-radii[0] if pair else 0.,
        })
        print({
            "grid":name,"amplitude":amplitude,"caps":radii,
            "corner":corner["maximum_intrinsic_residual"],
        },flush=True)
    paired=[item for item in records if item["pair_converged"]]
    no_cap=[item for item in records if item["cap_count"]==0]
    fit=fit_fold_normal_form(records,tail=min(6,len(paired))) if len(paired)>=3 else None
    runs.append({
        "name":name,"grid_size":[nz,nr],"records":records,"fold_fit":fit,
        "cap_bracket":{
            "largest_no_cap_amplitude":max(item["amplitude"] for item in no_cap) if no_cap else None,
            "smallest_pair_amplitude":min(item["amplitude"] for item in paired) if paired else None,
        },
    })

folds=[run["fold_fit"]["fold_amplitude"] for run in runs if run["fold_fit"]]
fold_corner_pass=all(
    min(
        (item for item in run["records"] if item["pair_converged"]),
        key=lambda item:item["amplitude"],
    )["corner_passes_0.025"] for run in runs
)
payload={
    "status":"fixed_shape_brackets_fold_but_corner_gate_fails_near_fold"
    if folds and not fold_corner_pass else
    "fixed_shape_fold_and_corner_pilot_passes" if folds else "fixed_shape_fold_not_bracketed",
    "corner_state":state_path,"amplitudes":list(amplitudes),"runs":runs,
    "cross_grid":{
        "fold_amplitudes":folds,
        "relative_fold_difference":abs(folds[1]-folds[0])/max(abs(folds[1]),1e-300)
        if len(folds)==2 else None,
    },
    "interpretation":"The fixed promoted shape gives a clean two-cap/no-cap fold bracket, but its spatial second-corner residual rises just above 0.025 near the fold. The fixed-shape scan is therefore a direction finder; a small amplitude-dependent coefficient continuation is required before the corrected fold can be promoted.",
    "limitations":[
        "fixed 80 coefficients rather than a corrected shape curve",
        "two R8 grids",
        "direct cap multiplicity scan rather than pseudo-arclength",
    ],
}
Path("results/corrected_fixed_shape_family_scan.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps({
    "status":payload["status"],"cross_grid":payload["cross_grid"],
    "brackets":[run["cap_bracket"] for run in runs],
},indent=2))
