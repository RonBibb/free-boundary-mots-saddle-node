#!/usr/bin/env python3
"""Audit stabilizer-gradient topology on the corrected G5/G6 folds."""

import json,sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_initial_data import solve_anisotropic_initial_data
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice
from bhps.localized_stabilizer_gradient import localized_stabilizer_gradient_diagnostics
from bhps.linearized_gh_einstein_scalar import stationary_point_scalar_metric_mixing
from bhps.physical_corner_corrector import combine_shape_modes,tracefree_shape_basis
from bhps.scalar_pulse import scalar_pulse


def corrected_geometry(name,nz,nr,amplitude,archive,coefficients):
    reference=solve_finite_wall_high_order_slice(
        amplitude,nz=nz,nr=nr,r_max=8.,wall_stiffness=20.,epsilon=.1,
        backreaction=.01,tolerance=1e-10,iterations=240,
    )
    _,chi_r,chi_z=scalar_pulse(reference["z"],reference["r"],amplitude)
    modes=tracefree_shape_basis(
        reference["z"],reference["r"],6,(.5,1.),8.,
        ((7.5,1.5),(7.5,3.0)),
    )["modes"]
    a,b,c=combine_shape_modes(coefficients,modes)
    selected=solve_anisotropic_initial_data(
        reference["z"],reference["r"],reference["q"],reference["phi"],a,b,c,
        reference["background"],chi_r,chi_z,
        initial_q=archive[f"q_{name}"],initial_phi=archive[f"phi_{name}"],
        stencil_width=7,tolerance=1e-9,iterations=30,
    )
    psi=1/(reference["z"][:,None]+selected["q"])
    return reference["z"],reference["r"],selected["phi"],psi,a,b,selected


def json_record(name,z,r,phi,psi,a,b,selected,amplitude):
    audit=localized_stabilizer_gradient_diagnostics(z,r,phi,psi,a,b)
    for point in audit["axis_stationary_points"]:
        location=point["z"]
        psi_value=float(CubicSpline(z,psi[:,0])(location))
        a_value=float(CubicSpline(z,a[:,0])(location))
        b_value=float(CubicSpline(z,b[:,0])(location))
        compact_factor=psi_value*np.exp(a_value)
        radial_factor=psi_value*np.exp(b_value)
        compact_hessian=point["phi_zz"]/compact_factor**2
        radial_hessian=point["phi_rr"]/radial_factor**2
        orthonormal_hessian=np.diag((0.,compact_hessian,radial_hessian,radial_hessian,radial_hessian))
        mixing=stationary_point_scalar_metric_mixing(
            orthonormal_hessian,point["phi"],.1*(.1+4),potential_offset=-6.,
        )
        point["orthonormal_compact_hessian"]=float(compact_hessian)
        point["orthonormal_radial_hessian"]=float(radial_hessian)
        point["metric_residual_per_delta_phi_diagonal"]=[
            float(mixing["metric_residual_per_delta_phi"][i,i]) for i in range(5)
        ]
        point["scalar_residual_per_diagonal_covariant_h"]=[
            float(mixing["scalar_residual_per_covariant_h"][i,i]) for i in range(5)
        ]
        point["regular_mixing_finite"]=mixing["finite"]
        point["mixing_depends_on_inverse_scalar_gradient"]=mixing["depends_on_inverse_scalar_gradient"]
    finite=np.isfinite(audit["turn_locations"])
    return {
        "name":name,"source_grid":[len(z),len(r)],"fold_amplitude":amplitude,
        "selector_maximum":float(selected["maximum_residual"]),
        "rays_with_compact_turn":audit["rays_with_compact_turn"],
        "fraction_of_rays_with_compact_turn":audit["fraction_of_rays_with_compact_turn"],
        "multiple_turn_ray_count":audit["multiple_turn_ray_count"],
        "turn_z_range":audit["turn_z_range"],
        "minimum_sampled_invariant_gradient_magnitude":audit["minimum_sampled_invariant_gradient_magnitude"],
        "maximum_abs_phi_z":audit["maximum_abs_phi_z"],
        "maximum_abs_phi_r":audit["maximum_abs_phi_r"],
        "divided_compact_gradient_variable_admissible":audit["divided_compact_gradient_variable_admissible"],
        "axis_stationary_points":audit["axis_stationary_points"],
        "turn_curve":{
            "r":[float(value) for value in r[finite]],
            "z":[float(value) for value in audit["turn_locations"][finite]],
            "phi_r":[float(value) for value in audit["transverse_gradient_at_turn"][finite]],
        },
    },audit


g5=json.loads(Path("results/corrected_anisotropic_arclength.json").read_text())
g6=json.loads(Path("results/corrected_anisotropic_arclength_G6.json").read_text())
archive=np.load("results/corrected_family_knot_A8_state.npz")
coefficients=archive["coefficients"]
specifications=(
    ("G5R8",49,73,float(g5["summary"]["fine_fold_amplitude"])),
    ("G6R8",65,97,float(g6["summary"]["fine_fold_amplitude"])),
)

records=[];raw=[]
for specification in specifications:
    name,nz,nr,amplitude=specification
    geometry=corrected_geometry(name,nz,nr,amplitude,archive,coefficients)
    record,audit=json_record(name,*geometry,amplitude)
    records.append(record);raw.append((geometry,audit))

point5=records[0]["axis_stationary_points"][0]
point6=records[1]["axis_stationary_points"][0]
r5=np.asarray(records[0]["turn_curve"]["r"]);z5=np.asarray(records[0]["turn_curve"]["z"])
r6=np.asarray(records[1]["turn_curve"]["r"]);z6=np.asarray(records[1]["turn_curve"]["z"])
common_max=min(r5[-1],r6[-1],6.75);common=r6[r6<=common_max]
curve_difference=np.abs(np.interp(common,r5,z5)-np.interp(common,r6,z6))
comparison={
    "axis_turn_z_relative_difference":float(abs(point5["z"]-point6["z"])/abs(point6["z"])),
    "axis_turn_phi_relative_difference":float(abs(point5["phi"]-point6["phi"])/abs(point6["phi"])),
    "axis_phi_zz_relative_difference":float(abs(point5["phi_zz"]-point6["phi_zz"])/abs(point6["phi_zz"])),
    "axis_phi_rr_relative_difference":float(abs(point5["phi_rr"]-point6["phi_rr"])/abs(point6["phi_rr"])),
    "turn_curve_common_radius_maximum":float(common_max),
    "turn_curve_maximum_abs_z_difference":float(np.max(curve_difference)),
    "turn_curve_rms_z_difference":float(np.sqrt(np.mean(curve_difference**2))),
}
acceptance={
    "both_selector_residuals_below_1e_8":all(item["selector_maximum"]<1e-8 for item in records),
    "both_divided_compact_gradient_variables_rejected":all(not item["divided_compact_gradient_variable_admissible"] for item in records),
    "both_have_one_axis_stationary_minimum":all(
        len(item["axis_stationary_points"])==1
        and item["axis_stationary_points"][0]["classification"]=="minimum"
        for item in records
    ),
    "both_stationary_mixings_finite_and_undivided":all(
        item["axis_stationary_points"][0]["regular_mixing_finite"]
        and not item["axis_stationary_points"][0]["mixing_depends_on_inverse_scalar_gradient"]
        for item in records
    ),
    "axis_turn_z_cross_grid_relative_difference_below_1e_3":comparison["axis_turn_z_relative_difference"]<1e-3,
    "turn_curve_cross_grid_maximum_z_difference_below_5e_3":comparison["turn_curve_maximum_abs_z_difference"]<5e-3,
}
payload={
    "status":"pass" if all(acceptance.values()) else "review",
    "scope":"stabilizer-gradient topology on the exact corrected gamma ell=20 G5/G6 folds",
    "cases":records,"cross_grid_comparison":comparison,"acceptance":acceptance,
    "interpretation":[
        "The localized production folds are not compact-direction-monotone even though their one-dimensional gamma ell=20 background is stable and monotone.",
        "An isolated stabilizer minimum lies on the symmetry axis, so any perturbation variable that divides by Phi_z or by the full stabilizer gradient is singular on the physical source-region geometry.",
        "This is a formulation obstruction, not a localized stability eigenvalue calculation.",
    ],
    "limitations":[
        "time-symmetric corrected initial slices rather than evolved spacetimes",
        "gradient topology only; no localized coupled perturbation spectrum",
        "axis radial Hessian obtained from an even polynomial fit in r squared",
        "turn-curve comparison restricted to the common r<=6.75 source neighborhood",
    ],
}
Path("results/corrected_fold_stabilizer_gradient_audit.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps({
    "status":payload["status"],"acceptance":acceptance,
    "cases":[{
        "name":item["name"],"selector_maximum":item["selector_maximum"],
        "rays_with_turn":item["rays_with_compact_turn"],
        "ray_fraction":item["fraction_of_rays_with_compact_turn"],
        "axis_stationary_points":item["axis_stationary_points"],
    } for item in records],
    "cross_grid_comparison":comparison,
},indent=2))
