#!/usr/bin/env python3
"""Audit the regular nine-field SO(3) GH operator on corrected folds."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_geometry import anisotropic_metric_acceleration,anisotropic_scalar_acceleration
from bhps.anisotropic_initial_data import solve_anisotropic_initial_data
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice
from bhps.generalized_harmonic_jets import spatial_metric_acceleration_trace
from bhps.lapse_acceleration_corner import construct_localized_target_lapse_acceleration_completion
from bhps.linearized_gh_einstein_scalar import metric_geometry_from_jets,reduced_einstein_two_scalar_residual
from bhps.physical_corner_corrector import combine_shape_modes,tracefree_shape_basis
from bhps.regular_so3_gh_reduction import FIELD_ORDER,RegularSO3BackgroundJetField,regular_so3_gh_coefficient_matrices
from bhps.scalar_pulse import scalar_pulse


def build_case(name,nz,nr,amplitude,archive,coefficients):
    reference=solve_finite_wall_high_order_slice(
        amplitude,nz=nz,nr=nr,r_max=8.,wall_stiffness=20.,epsilon=.1,
        backreaction=.01,tolerance=1e-10,iterations=240,
    )
    chi,chi_r,chi_z=scalar_pulse(reference["z"],reference["r"],amplitude)
    modes=tracefree_shape_basis(
        reference["z"],reference["r"],6,(.5,1.),8.,((7.5,1.5),(7.5,3.0)),
    )["modes"]
    a,b,c=combine_shape_modes(coefficients,modes)
    selected=solve_anisotropic_initial_data(
        reference["z"],reference["r"],reference["q"],reference["phi"],a,b,c,
        reference["background"],chi_r,chi_z,
        initial_q=archive[f"q_{name}"],initial_phi=archive[f"phi_{name}"],
        stencil_width=7,tolerance=1e-9,iterations=30,
    )
    z=reference["z"];r=reference["r"];phi=selected["phi"]
    psi=1/(z[:,None]+selected["q"]);mass=float(reference["background"]["mass_squared"])
    acceleration=anisotropic_metric_acceleration(
        z,r,psi,a,b,c,phi,chi_r,chi_z,mass,chi=chi,stencil_width=7,lapse=psi,
    )
    phi_tt=anisotropic_scalar_acceleration(z,r,psi,a,b,c,phi,mass,lapse=psi,stencil_width=7)
    chi_tt=anisotropic_scalar_acceleration(z,r,psi,a,b,c,chi,0.,lapse=psi,stencil_width=7)
    trace=spatial_metric_acceleration_trace(acceleration,psi,a,b,c)
    completion=construct_localized_target_lapse_acceleration_completion(
        z,acceleration,psi,psi,a,phi,reference["background"],phi_tt,.5*trace,.15,
    )
    field=RegularSO3BackgroundJetField(
        z,r,psi,psi,a,b,c,phi,chi,acceleration,completion["lapse_acceleration"],
        phi_tt,chi_tt,7,
    )
    return {
        "name":name,"source_grid":[nz,nr],"fold_amplitude":amplitude,
        "selector_maximum":float(selected["maximum_residual"]),
        "mass_squared":mass,"z":z,"field":field,
    }


def closure_at(background,mass):
    geometry=metric_geometry_from_jets(
        background["metric"],background["metric_first"],background["metric_second"],
    )
    residual=reduced_einstein_two_scalar_residual(
        background["metric"],background["metric_first"],background["metric_second"],
        background["phi"],background["phi_first"],background["phi_second"],
        background["chi"],background["chi_first"],background["chi_second"],
        geometry["contracted_christoffel_covector"],geometry["contracted_christoffel_covector_first"],
        mass_squared=mass,potential_offset=-6.,kappa5_squared=1.,
    )
    frame=1/np.sqrt(np.abs(np.diag(background["metric"])))
    metric_residual=residual["metric_residual"]*frame[:,None]*frame[None,:]
    ricci=residual["ricci"]*frame[:,None]*frame[None,:]
    matter=ricci-metric_residual
    metric_scale=max(float(np.max(np.abs(ricci))),float(np.max(np.abs(matter))),1e-300)
    inverse=residual["inverse_metric"]
    phi_scale=max(float(np.max(np.abs(inverse*residual["phi_covariant_hessian"]))),abs(residual["potential_prime"]),1e-300)
    chi_scale=max(float(np.max(np.abs(inverse*residual["chi_covariant_hessian"]))),1e-300)
    return {
        "metric_orthonormal_maximum":float(np.max(np.abs(metric_residual))),
        "metric_relative":float(np.max(np.abs(metric_residual))/metric_scale),
        "phi_absolute":float(abs(residual["phi_residual"])),
        "phi_relative":float(abs(residual["phi_residual"])/phi_scale),
        "chi_absolute":float(abs(residual["chi_residual"])),
        "chi_relative":float(abs(residual["chi_residual"])/chi_scale),
        "gauge_constraint_maximum":float(np.max(np.abs(residual["gauge_constraint_covector"]))),
    }


g5=json.loads(Path("results/corrected_anisotropic_arclength.json").read_text())
g6=json.loads(Path("results/corrected_anisotropic_arclength_G6.json").read_text())
archive=np.load("results/corrected_family_knot_A8_state.npz");coefficients=archive["coefficients"]
specifications=(
    ("G5R8",49,73,float(g5["summary"]["fine_fold_amplitude"])),
    ("G6R8",65,97,float(g6["summary"]["fine_fold_amplitude"])),
)
cases=[build_case(*specification,archive,coefficients) for specification in specifications]
closure_z=np.geomspace(cases[0]["z"][0],cases[0]["z"][-1],5)
closure_r=np.array((.25,.75,1.5,3.,4.))
operator_z=np.geomspace(cases[0]["z"][0],cases[0]["z"][-1],3)
operator_r=np.array((.25,.75,1.5,3.))

reaction_arrays=[];first_arrays=[];records=[]
for case in cases:
    closure=[]
    for z_value in closure_z:
        for r_value in closure_r:
            closure.append({
                "z":float(z_value),"r":float(r_value),
                **closure_at(case["field"].at(z_value,r_value),case["mass_squared"]),
            })
    operators=[];case_reaction=[];case_first=[]
    for z_value in operator_z:
        for r_value in operator_r:
            coefficients_at_point=regular_so3_gh_coefficient_matrices(
                case["field"].at(z_value,r_value),r_value,
                mass_squared=case["mass_squared"],potential_offset=-6.,
            )
            reaction=coefficients_at_point["evolution_reaction_matrix"]
            first=coefficients_at_point["evolution_first_matrices"]
            case_reaction.append(reaction);case_first.append(first)
            maximum_index=np.unravel_index(np.argmax(np.abs(reaction)),reaction.shape)
            eigenvalues=np.linalg.eigvals(reaction)
            operators.append({
                "z":float(z_value),"r":float(r_value),
                "principal_identity_maximum_defect":coefficients_at_point["principal_identity_maximum_defect"],
                "reaction_frobenius_norm":float(np.linalg.norm(reaction)),
                "reaction_maximum_absolute_entry":float(np.max(np.abs(reaction))),
                "reaction_maximum_entry":{
                    "row":FIELD_ORDER[maximum_index[0]],"column":FIELD_ORDER[maximum_index[1]],
                    "value":float(reaction[maximum_index]),
                },
                "reaction_spectral_radius":float(np.max(np.abs(eigenvalues))),
                "reaction_spectral_abscissa":float(np.max(np.real(eigenvalues))),
                "first_frobenius_norms":[float(np.linalg.norm(first[index])) for index in range(3)],
                "reaction_entries_above_1e_10":int(np.count_nonzero(np.abs(reaction)>1e-10)),
                "finite":coefficients_at_point["finite"],
            })
    reaction_arrays.append(np.asarray(case_reaction));first_arrays.append(np.asarray(case_first))
    records.append({
        "name":case["name"],"source_grid":case["source_grid"],
        "fold_amplitude":case["fold_amplitude"],"selector_maximum":case["selector_maximum"],
        "closure_samples":closure,"operator_samples":operators,
        "closure_summary":{
            "metric_relative_maximum":max(item["metric_relative"] for item in closure),
            "phi_relative_maximum":max(item["phi_relative"] for item in closure),
            "chi_relative_maximum":max(item["chi_relative"] for item in closure),
            "gauge_constraint_maximum":max(item["gauge_constraint_maximum"] for item in closure),
        },
        "operator_summary":{
            "principal_identity_maximum_defect":max(item["principal_identity_maximum_defect"] for item in operators),
            "reaction_norm_range":[min(item["reaction_frobenius_norm"] for item in operators),max(item["reaction_frobenius_norm"] for item in operators)],
            "first_norm_maxima":[max(item["first_frobenius_norms"][index] for item in operators) for index in range(3)],
            "all_finite":all(item["finite"] for item in operators),
        },
    })

def relative_field_difference(left,right):
    return float(np.linalg.norm(left-right)/max(np.linalg.norm(right),1e-300))

comparison={
    "reaction_coefficient_field_relative_difference":relative_field_difference(reaction_arrays[0],reaction_arrays[1]),
    "first_coefficient_field_relative_difference":relative_field_difference(first_arrays[0],first_arrays[1]),
}
fine=records[1]
acceptance={
    "selectors_below_1e_8":all(item["selector_maximum"]<1e-8 for item in records),
    "fine_metric_closure_relative_below_2_percent":fine["closure_summary"]["metric_relative_maximum"]<.02,
    "fine_phi_closure_relative_below_2_percent":fine["closure_summary"]["phi_relative_maximum"]<.02,
    "fine_chi_closure_relative_below_2_percent":fine["closure_summary"]["chi_relative_maximum"]<.02,
    "all_operator_coefficients_finite":all(item["operator_summary"]["all_finite"] for item in records),
    "all_principal_identity_defects_below_1e_10":all(item["operator_summary"]["principal_identity_maximum_defect"]<1e-10 for item in records),
    "reaction_field_cross_grid_difference_below_5_percent":comparison["reaction_coefficient_field_relative_difference"]<.05,
    "first_field_cross_grid_difference_below_5_percent":comparison["first_coefficient_field_relative_difference"]<.05,
}
payload={
    "status":"pass" if all(acceptance.values()) else "review",
    "scope":"regular nine-field SO(3)-symmetric frozen-source GH coefficient audit on matched corrected-fold points",
    "field_order":list(FIELD_ORDER),
    "closure_sample_coordinates":{"z":[float(value) for value in closure_z],"r":[float(value) for value in closure_r]},
    "operator_sample_coordinates":{"z":[float(value) for value in operator_z],"r":[float(value) for value in operator_r]},
    "cases":records,"cross_grid_comparison":comparison,"acceptance":acceptance,
    "interpretation":[
        "The physical SO(3)-invariant sector has nine regular fields, not seventeen independent radial scalar components.",
        "The extracted lower-order matrices include the angular tensor connection terms while retaining one common scalar-wave principal symbol.",
        "The separate Cartesian stationary-point audit supplies the r=0 closure; this sampled field audit begins at r=0.25.",
    ],
    "limitations":[
        "twelve matched coefficient points and twenty-five closure points per corrected fold rather than every evolution node",
        "frozen background generalized-harmonic source",
        "constraint damping and evolved source driver absent",
        "coefficient matrices not yet assembled into Q1/RK4 time evolution",
    ],
}
Path("results/corrected_fold_regular_so3_operator_audit.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps({
    "status":payload["status"],"acceptance":acceptance,
    "case_summaries":[{
        "name":item["name"],"closure":item["closure_summary"],
        "operator":item["operator_summary"],
    } for item in records],"cross_grid_comparison":comparison,
},indent=2))
