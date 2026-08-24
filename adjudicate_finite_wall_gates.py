#!/usr/bin/env python3
"""Machine-readable acceptance-gate audit for the finite-wall capped fold."""

import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.finite_wall_solver import solve_finite_wall_slice


def load(name):return json.loads(Path("results",name).read_text())


refinement=load("finite_wall_refinement.json")
arclength=load("finite_wall_arclength.json")
surface_fd=load("finite_wall_surface_fd_replication.json")
high_order=load("finite_wall_high_order_replication.json")
robustness=load("finite_wall_robustness.json")
fine_amplitude=next(item["selected_fold_amplitude"] for item in refinement["cases"] if item["id"]=="G6_R8")
fine=solve_finite_wall_slice(
    fine_amplitude,wall_stiffness=20.,nz=81,nr=121,r_max=8.,epsilon=.1,backreaction=.01,
    tolerance=1e-10,iterations=180,
)

checks={
    "coupled_residual_max_below_1e_minus_8":fine["max_abs_residual"]<1e-8,
    "coupled_residual_l2_below_1e_minus_9":fine["residual_l2"]<1e-9,
    "metric_junction_residual_below_1e_minus_9":fine["metric_junction_residual_max"]<1e-9,
    "scalar_wall_residual_below_1e_minus_9":fine["scalar_wall_residual_max"]<1e-9,
    "positive_metric":fine["min_psi"]>0,
    "three_grid_fold_convergence":refinement["summary"]["fold_energy_convergence"]["observed_order"]>1,
    "two_finest_fold_energy_change_below_2_percent":refinement["summary"]["fold_energy_convergence"]["two_finest_relative_change"]<.02,
    "largest_domain_fold_energy_change_below_2_percent":refinement["summary"]["G4_R10_R12_relative_domain_difference"]["fold_energy"]<.02,
    "energy_quadrature_difference_below_0p5_percent":fine["energy_quadrature_relative_difference"]<.005,
    "fold_crossed_by_pseudo_arclength":arclength["status"]=="finite_wall_coupled_pseudo_arclength_fold_crossed",
    "independent_surface_solver_agrees_below_2_percent":surface_fd["summary"]["finest_fd_collocation_fold_relative_difference"]<.02,
    "near_null_direction_exhibited":min(surface_fd["summary"]["finest_near_fold_branch_difference_overlaps"])>.99,
    "independent_coupled_discretization_agrees_below_2_percent":high_order["summary"]["high_order_vs_primary_extrapolated_relative_difference"]["fold_energy"]<.02,
    "independent_coupled_discretization_domain_change_below_2_percent":high_order["summary"]["H4_R10_R12_relative_domain_difference"]["fold_energy"]<.02,
    "finite_wall_stiffness_interval_persists":robustness["all_cases_have_expected_pair_stability"],
}
payload={
    "status":"all_finite_wall_static_numerical_gates_pass_for_scoped_capped_fold" if all(checks.values()) else "one_or_more_finite_wall_static_numerical_gates_fail",
    "claim_scope":"capped marginal-surface fold in the gamma ell=20 momentarily stationary finite-wall stabilizer initial-data selector, with G4 robustness over gamma ell=2...100",
    "checks":checks,
    "headline_metrics":{
        "fine_coupled_residual_max":fine["max_abs_residual"],
        "fine_coupled_residual_l2":fine["residual_l2"],
        "fine_metric_junction_residual_max":fine["metric_junction_residual_max"],
        "fine_scalar_wall_residual_max":fine["scalar_wall_residual_max"],
        "fine_energy_quadrature_relative_difference":fine["energy_quadrature_relative_difference"],
        "primary_extrapolated_fold_energy":refinement["summary"]["fold_energy_convergence"]["extrapolated_value"],
        "independent_high_order_extrapolated_fold_energy":high_order["summary"]["fold_energy_convergence"]["extrapolated_value"],
        "coupled_metric_solver_fold_energy_relative_difference":high_order["summary"]["high_order_vs_primary_extrapolated_relative_difference"]["fold_energy"],
        "primary_largest_domain_fold_energy_change":refinement["summary"]["G4_R10_R12_relative_domain_difference"]["fold_energy"],
        "independent_largest_domain_fold_energy_change":high_order["summary"]["H4_R10_R12_relative_domain_difference"]["fold_energy"],
    },
    "not_adjudicated_or_still_open":[
        "full spatial Einstein equations for a static spacetime",
        "hyperbolic evolution and nonlinear boundary well-posedness",
        "generic non-equilibrium stabilizer initial data",
        "multidimensional parameter map",
        "dynamical horizon formation and branch selection",
        "continuum existence of the high-energy spanning candidate",
    ],
}
Path("results/finite_wall_acceptance_adjudication.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2))
