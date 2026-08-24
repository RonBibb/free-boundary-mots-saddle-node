#!/usr/bin/env python3
"""Close the local covariant Einstein--two-scalar jets on corrected folds."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_geometry import anisotropic_metric_acceleration,anisotropic_scalar_acceleration
from bhps.anisotropic_initial_data import solve_anisotropic_initial_data
from bhps.axis_cartesian_spacetime_jets import construct_time_symmetric_axis_spacetime_jets
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice
from bhps.generalized_harmonic_jets import spatial_metric_acceleration_trace
from bhps.gh_operator_coefficients import FIELD_ORDER,frozen_source_gh_coefficient_matrices
from bhps.lapse_acceleration_corner import construct_localized_target_lapse_acceleration_completion
from bhps.linearized_gh_einstein_scalar import metric_geometry_from_jets,reduced_einstein_two_scalar_residual
from bhps.localized_stabilizer_gradient import localized_stabilizer_gradient_diagnostics
from bhps.physical_corner_corrector import combine_shape_modes,tracefree_shape_basis
from bhps.scalar_pulse import scalar_pulse


def corrected_case(name,nz,nr,amplitude,archive,coefficients):
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
        reference["z"],reference["r"],reference["q"],reference["phi"],a,b,c,
        reference["background"],chi_r,chi_z,
        initial_q=archive[f"q_{name}"],initial_phi=archive[f"phi_{name}"],
        stencil_width=7,tolerance=1e-9,iterations=30,
    )
    z=reference["z"];r=reference["r"]
    psi=1/(z[:,None]+selected["q"]);phi=selected["phi"]
    mass=float(reference["background"]["mass_squared"])
    metric_acceleration=anisotropic_metric_acceleration(
        z,r,psi,a,b,c,phi,chi_r,chi_z,mass,chi=chi,
        stencil_width=7,lapse=psi,
    )
    phi_acceleration=anisotropic_scalar_acceleration(
        z,r,psi,a,b,c,phi,mass,lapse=psi,stencil_width=7,
    )
    chi_acceleration=anisotropic_scalar_acceleration(
        z,r,psi,a,b,c,chi,0.,lapse=psi,stencil_width=7,
    )
    trace=spatial_metric_acceleration_trace(metric_acceleration,psi,a,b,c)
    completion=construct_localized_target_lapse_acceleration_completion(
        z,metric_acceleration,psi,psi,a,phi,reference["background"],
        phi_acceleration,.5*trace,.15,
    )
    topology=localized_stabilizer_gradient_diagnostics(z,r,phi,psi,a,b)
    if len(topology["axis_stationary_points"])!=1:
        raise RuntimeError("expected exactly one corrected-fold axis stationary point")
    location=topology["axis_stationary_points"][0]["z"]
    constructed=construct_time_symmetric_axis_spacetime_jets(
        z,r,psi,psi,a,b,c,phi,chi,metric_acceleration,
        completion["lapse_acceleration"],phi_acceleration,chi_acceleration,location,
    )
    background=constructed["background"]
    geometry=metric_geometry_from_jets(
        background["metric"],background["metric_first"],background["metric_second"],
    )
    residual=reduced_einstein_two_scalar_residual(
        background["metric"],background["metric_first"],background["metric_second"],
        background["phi"],background["phi_first"],background["phi_second"],
        background["chi"],background["chi_first"],background["chi_second"],
        geometry["contracted_christoffel_covector"],
        geometry["contracted_christoffel_covector_first"],
        mass_squared=mass,potential_offset=-6.,kappa5_squared=1.,
    )
    diagonal=np.diag(background["metric"])
    frame=1/np.sqrt(np.abs(diagonal))
    orthonormal_residual=residual["metric_residual"]*frame[:,None]*frame[None,:]
    orthonormal_ricci=residual["ricci"]*frame[:,None]*frame[None,:]
    orthonormal_reduced=residual["reduced_ricci"]*frame[:,None]*frame[None,:]
    metric_scale=max(
        float(np.max(np.abs(orthonormal_ricci))),
        float(np.max(np.abs(orthonormal_reduced-orthonormal_residual))),1e-300,
    )
    inverse=residual["inverse_metric"]
    phi_terms=np.abs(inverse*residual["phi_covariant_hessian"])
    chi_terms=np.abs(inverse*residual["chi_covariant_hessian"])
    phi_scale=max(float(np.max(phi_terms)),abs(residual["potential_prime"]),1e-300)
    chi_scale=max(float(np.max(chi_terms)),1e-300)
    coefficients=frozen_source_gh_coefficient_matrices(
        background,mass_squared=mass,potential_offset=-6.,kappa5_squared=1.,
    )
    zero_matrix=coefficients["zero_order_matrix"]
    first_matrices=coefficients["scalar_wave_adjusted_first_matrices"]
    reaction_matrix=coefficients["evolution_reaction_matrix"]
    phi_index=FIELD_ORDER.index("delta_Phi")
    return {
        "name":name,"grid_size":[nz,nr],"fold_amplitude":amplitude,
        "selector_maximum":float(selected["maximum_residual"]),
        "stationary_location":constructed["location"],
        "stationary_phi":float(background["phi"]),
        "stationary_coordinate_gradient":[float(value) for value in background["phi_first"]],
        "regularity":{key:float(value) for key,value in constructed["regularity"].items()},
        "metric_residual_orthonormal_maximum":float(np.max(np.abs(orthonormal_residual))),
        "metric_residual_relative_maximum":float(np.max(np.abs(orthonormal_residual))/metric_scale),
        "metric_residual_orthonormal_diagonal":[float(value) for value in np.diag(orthonormal_residual)],
        "phi_residual":float(residual["phi_residual"]),
        "phi_residual_relative":float(abs(residual["phi_residual"])/phi_scale),
        "chi_residual":float(residual["chi_residual"]),
        "chi_residual_relative":float(abs(residual["chi_residual"])/chi_scale),
        "gauge_constraint_maximum":float(np.max(np.abs(residual["gauge_constraint_covector"]))),
        "lapse_relative_acceleration":float(
            np.interp(location,z,(completion["lapse_acceleration"]/psi)[:,0])
        ),
        "metric_acceleration_trace_half":float(np.interp(location,z,(.5*trace)[:,0])),
        "operator_summary":{
            "field_order":list(FIELD_ORDER),
            "finite":coefficients["finite"],
            "zero_order_frobenius_norm":float(np.linalg.norm(zero_matrix)),
            "scalar_wave_adjusted_first_frobenius_norms":[
                float(np.linalg.norm(first_matrices[index])) for index in range(5)
            ],
            "evolution_reaction_frobenius_norm":float(np.linalg.norm(reaction_matrix)),
            "evolution_reaction_maximum_absolute_entry":float(np.max(np.abs(reaction_matrix))),
            "entries_above_1e_10":int(np.count_nonzero(np.abs(reaction_matrix)>1e-10)),
            "metric_rows_coupled_from_delta_phi":[
                float(reaction_matrix[index,phi_index]) for index in range(15)
            ],
            "delta_phi_row_coupled_from_metric":[
                float(reaction_matrix[phi_index,index]) for index in range(15)
            ],
        },
        "_operator_arrays":{
            "zero":zero_matrix,"first":first_matrices,"reaction":reaction_matrix,
        },
    }


g5=json.loads(Path("results/corrected_anisotropic_arclength.json").read_text())
g6=json.loads(Path("results/corrected_anisotropic_arclength_G6.json").read_text())
archive=np.load("results/corrected_family_knot_A8_state.npz")
coefficients=archive["coefficients"]
specifications=(
    ("G5R8",49,73,float(g5["summary"]["fine_fold_amplitude"])),
    ("G6R8",65,97,float(g6["summary"]["fine_fold_amplitude"])),
)
cases=[corrected_case(*specification,archive,coefficients) for specification in specifications]
coarse,fine=cases
def relative_matrix_difference(left,right):
    return float(np.linalg.norm(left-right)/max(np.linalg.norm(right),1e-300))

comparison={
    "stationary_z_relative_difference":abs(
        coarse["stationary_location"]["z"]-fine["stationary_location"]["z"]
    )/abs(fine["stationary_location"]["z"]),
    "metric_residual_cross_grid_ratio":coarse["metric_residual_orthonormal_maximum"]/fine["metric_residual_orthonormal_maximum"],
    "phi_residual_cross_grid_ratio":abs(coarse["phi_residual"])/max(abs(fine["phi_residual"]),1e-300),
    "chi_residual_cross_grid_ratio":abs(coarse["chi_residual"])/max(abs(fine["chi_residual"]),1e-300),
    "zero_order_matrix_relative_difference":relative_matrix_difference(
        coarse["_operator_arrays"]["zero"],fine["_operator_arrays"]["zero"],
    ),
    "compact_first_matrix_relative_difference":relative_matrix_difference(
        coarse["_operator_arrays"]["first"][1],fine["_operator_arrays"]["first"][1],
    ),
    "evolution_reaction_matrix_relative_difference":relative_matrix_difference(
        coarse["_operator_arrays"]["reaction"],fine["_operator_arrays"]["reaction"],
    ),
}
acceptance={
    "selectors_below_1e_8":all(case["selector_maximum"]<1e-8 for case in cases),
    "axis_metric_regularity_below_1e_10":all(case["regularity"]["relative_metric_mismatch"]<1e-10 for case in cases),
    "axis_acceleration_regularity_below_1e_5":all(case["regularity"]["relative_acceleration_mismatch"]<1e-5 for case in cases),
    "mixed_axis_acceleration_at_roundoff":all(abs(case["regularity"]["mixed_zr_acceleration"])<1e-12 for case in cases),
    "gauge_constraints_at_roundoff":all(case["gauge_constraint_maximum"]<1e-12 for case in cases),
    "fine_metric_residual_relative_below_5e_3":fine["metric_residual_relative_maximum"]<5e-3,
    "fine_phi_residual_relative_below_2e_3":fine["phi_residual_relative"]<2e-3,
    "fine_chi_residual_relative_below_2e_3":fine["chi_residual_relative"]<2e-3,
    "metric_residual_decreases_by_at_least_three":comparison["metric_residual_cross_grid_ratio"]>3.,
    "both_scalar_residuals_decrease_by_at_least_two":comparison["phi_residual_cross_grid_ratio"]>2. and comparison["chi_residual_cross_grid_ratio"]>2.,
    "both_operator_matrices_finite":all(case["operator_summary"]["finite"] for case in cases),
    "three_transverse_first_matrix_norms_respect_axis_SO3":all(
        np.ptp(case["operator_summary"]["scalar_wave_adjusted_first_frobenius_norms"][2:5])
        <1e-12*max(case["operator_summary"]["scalar_wave_adjusted_first_frobenius_norms"][2:5])
        for case in cases
    ),
    "reaction_matrix_cross_grid_difference_below_2_percent":comparison["evolution_reaction_matrix_relative_difference"]<.02,
    "compact_first_matrix_cross_grid_difference_below_2_percent":comparison["compact_first_matrix_relative_difference"]<.02,
}
for case in cases:case.pop("_operator_arrays")
payload={
    "status":"pass" if all(acceptance.values()) else "review",
    "scope":"complete regular Cartesian spacetime jets at the corrected-fold stabilizer minimum",
    "cases":cases,"cross_grid_comparison":comparison,"acceptance":acceptance,
    "interpretation":[
        "The ADM metric accelerations and both scalar accelerations complete the time-symmetric data into finite covariant spacetime jets at the point where the stabilizer gradient vanishes.",
        "Choosing the background generalized-harmonic source and first source jet equal to the contracted-Christoffel jets makes the reduced residual identical to the physical trace-reversed Einstein residual at the audit point.",
        "Passing this local closure test removes the stationary-point formulation obstruction; it does not establish global hyperbolic evolution or nonlinear stability.",
    ],
    "limitations":[
        "single symmetry-axis stationary point on each of the G5/G6 corrected folds",
        "radial Hessians use a six-point even polynomial fit in r squared",
        "zero shift acceleration and a fixed background generalized-harmonic source jet",
        "local jet closure rather than an assembled grid operator or evolved gauge driver",
    ],
}
Path("results/corrected_fold_spacetime_jet_audit.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps(payload,indent=2))
