#!/usr/bin/env python3
"""Locate the local cap fold using the full-exterior A=8 shape knot."""

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


amplitudes=(7.90,7.94,7.96,7.975,7.985,7.995,8.00,8.005,8.015,8.03,8.05)
state_path="results/corrected_family_knot_A8_state.npz"
archive=np.load(state_path);coefficients=archive["coefficients"]
runs=[]
for name,nz,nr in (("G5R8",49,73),("G6R8",65,97)):
    records=[];initial_q=initial_phi=None
    for amplitude in amplitudes:
        reference=solve_finite_wall_high_order_slice(
            amplitude,nz=nz,nr=nr,r_max=8.,wall_stiffness=20.,epsilon=.1,
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
            guesses=tuple(np.linspace(1.1,1.75,20)),tolerance=2e-5,
        )
        radii=sorted(float(item["rho_brane"]) for item in caps["accepted"])
        energy=anisotropic_scalar_gradient_energy(
            reference["z"],reference["r"],corner["psi"],a,b,c,chi_r,chi_z,
        )
        pair=len(radii)==2
        records.append({
            "amplitude":amplitude,"energy_dimensionless":energy,
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
    paired=sorted(
        (item for item in records if item["pair_converged"]),
        key=lambda item:item["amplitude"],
    )
    no_cap=[item for item in records if item["cap_count"]==0]
    fit=fit_fold_normal_form(records,tail=min(6,len(paired)))
    fit_records=paired[:min(6,len(paired))]
    energy=np.array([item["energy_dimensionless"] for item in fit_records])
    squared=np.array([item["radius_separation"]**2 for item in fit_records])
    energy_slope,energy_intercept=np.polyfit(energy,squared,1)
    fit["fold_energy_dimensionless"]=-energy_intercept/energy_slope
    fit["energy_normal_form_slope"]=energy_slope
    runs.append({
        "name":name,"grid_size":[nz,nr],"records":records,"fold_fit":fit,
        "cap_bracket":{
            "largest_no_cap_amplitude":max(item["amplitude"] for item in no_cap),
            "smallest_pair_amplitude":min(item["amplitude"] for item in paired),
        },
    })

folds=[run["fold_fit"]["fold_amplitude"] for run in runs]
energies=[run["fold_fit"]["fold_energy_dimensionless"] for run in runs]
near_fold_corner_pass=all(
    all(item["corner_passes_0.025"] for item in run["records"] if 7.94<=item["amplitude"]<=8.03)
    for run in runs
)
payload={
    "status":"corrected_A8_knot_fold_bracket_and_corner_gate_pass"
    if near_fold_corner_pass else "corrected_A8_knot_fold_corner_incomplete",
    "shape_state":state_path,"amplitudes":list(amplitudes),"runs":runs,
    "cross_grid":{
        "fold_amplitudes":folds,
        "fold_energies_dimensionless":energies,
        "relative_fold_amplitude_difference":abs(folds[1]-folds[0])/abs(folds[1]),
        "relative_fold_energy_difference":abs(energies[1]-energies[0])/abs(energies[1]),
    },
    "interpretation":"The second shape knot moves the corrected cap fold only slightly while bringing the full spatial corner below 0.025 throughout the sampled fold neighborhood on both R8 grids. This supplies a compatible direct fold bracket; pseudo-arclength and interpolation between coefficient knots remain separate gates.",
    "limitations":[
        "one local shape knot held fixed over the narrow fold scan",
        "two R8 grids",
        "normal-form fit rather than pseudo-arclength crossing",
        "large-domain transfer of the exact fold slice remains",
    ],
}
Path("results/corrected_A8_knot_local_fold_scan.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps({
    "status":payload["status"],"cross_grid":payload["cross_grid"],
    "brackets":[run["cap_bracket"] for run in runs],
},indent=2))
