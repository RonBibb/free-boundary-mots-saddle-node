#!/usr/bin/env python3
"""Machine-readable adjudication of the prospective static acceptance gates."""

import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from bhps.gw_slice_solver import solve_gw_slice


def load(name):return json.loads(Path("results",name).read_text())

branch=load("gw_slice_capped_branch.json");arclength=load("gw_slice_capped_arclength.json")
stability=load("gw_slice_capped_stability.json");surface_fd=load("gw_slice_capped_fd_replication.json")
metric_high=load("gw_slice_high_order_replication.json");profiles=load("gw_slice_profile_robustness.json")
surfaces=load("gw_slice_surface_convergence.json");gw_parameters=load("gw_parameter_robustness.json")
spanning=load("gw_slice_spanning_branch.json")

fine_amplitude=next(item["selected_fold_amplitude"] for item in branch["cases"] if item["id"]=="G6_R8")
fine=solve_gw_slice(fine_amplitude,nz=81,nr=121,r_max=8,epsilon=.1,backreaction=.01,tolerance=1e-10,iterations=180)
surface_records=[surface for case in surfaces["cases"] for surface in case["surfaces"]]
surface_convergence=[entry for branch_data in surfaces["three_grid_convergence_G4_G6_R8"].values() for entry in branch_data.values()]
surface_domain=[value for branch_data in surfaces["G4_R10_R12_relative_domain_differences"].values() for value in branch_data.values()]

checks={
    "constraint_max_below_1e_minus_8":fine["max_abs_residual"]<1e-8,
    "constraint_l2_below_1e_minus_9":fine["residual_l2"]<1e-9,
    "junction_residual_below_1e_minus_9":fine["junction_residual_max"]<1e-9,
    "positive_metric":fine["min_psi"]>0,
    "three_grid_fold_convergence":branch["summary"]["fold_energy_convergence"]["observed_order"]>1,
    "largest_domain_fold_energy_change_below_2_percent":branch["summary"]["G4_R10_R12_relative_domain_difference"]["fold_energy"]<.02,
    "energy_quadrature_difference_below_0p5_percent":fine["energy_quadrature_relative_difference"]<.005,
    "surface_residual_below_1e_minus_7":max(item["surface_residual_max"] for item in surface_records)<1e-7,
    "surface_boundary_residual_below_1e_minus_8":max(item["boundary_slope_error"] for item in surface_records)<1e-8,
    "surface_observables_have_measured_positive_order":min(item["observed_order"] for item in surface_convergence)>1,
    "surface_finest_extrapolation_error_below_2_percent":max(item["finest_relative_extrapolation_error"] for item in surface_convergence)<.02,
    "surface_largest_domain_change_below_1_percent":max(surface_domain)<.01,
    "fold_crossed_by_pseudo_arclength":arclength["status"]=="equilibrium_GW_initial_slice_pseudo_arclength_fold_crossed",
    "independent_surface_solver_agrees_below_2_percent":surface_fd["summary"]["finest_fd_collocation_fold_relative_difference"]<.02,
    "independent_metric_discretization_agrees_below_2_percent":metric_high["summary"]["high_order_vs_primary_extrapolated_relative_difference"]["fold_energy"]<.02,
    "near_null_direction_exhibited":min(surface_fd["summary"]["finest_near_fold_branch_difference_overlaps"])>.99,
    "full_angular_stability_classified":stability["all_cases_match_inner_one_mode_outer_zero_modes"],
    "pulse_profile_neighborhood_persists":profiles["all_cases_have_pair_and_expected_stability"],
    "weak_GW_parameter_neighborhood_persists":gw_parameters["all_cases_have_expected_pair_stability"],
    "spanning_candidate_rejected_by_domain_gate":spanning["summary"]["G6_R10_R12_fold_energy_relative_change"]>.02,
}
payload={
    "status":"all_prospective_static_numerical_gates_pass_for_scoped_capped_fold" if all(checks.values()) else "one_or_more_static_numerical_gates_fail",
    "claim_scope":"capped marginal-surface fold in the stiff-wall equilibrium-GW time-symmetric initial-data subfamily and tested one-at-a-time neighborhood",
    "checks":checks,
    "headline_metrics":{
        "fine_constraint_max":fine["max_abs_residual"],"fine_constraint_l2":fine["residual_l2"],
        "fine_junction_residual_max":fine["junction_residual_max"],
        "fine_energy_quadrature_relative_difference":fine["energy_quadrature_relative_difference"],
        "primary_extrapolated_fold_energy":branch["summary"]["fold_energy_convergence"]["extrapolated_value"],
        "independent_high_order_extrapolated_fold_energy":metric_high["summary"]["fold_energy_convergence"]["extrapolated_value"],
        "metric_solver_fold_energy_relative_difference":metric_high["summary"]["high_order_vs_primary_extrapolated_relative_difference"]["fold_energy"],
        "maximum_surface_finest_extrapolation_error":max(item["finest_relative_extrapolation_error"] for item in surface_convergence),
        "maximum_surface_domain_change":max(surface_domain),
    },
    "not_adjudicated_or_still_open":[
        "full action variation and hyperbolic well-posedness",
        "finite stabilizer wall stiffness",
        "multidimensional parameter map",
        "generic non-equilibrium stabilizer initial data",
        "dynamical horizon formation and branch selection",
        "continuum existence of the high-energy spanning candidate",
    ],
}
Path("results/static_acceptance_adjudication.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2))
