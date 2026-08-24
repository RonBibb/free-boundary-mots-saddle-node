#!/usr/bin/env python3
"""Neighborhood audit for the explicit Israel--scalar Robin symmetrizer."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.gw_background import solve_gw_background
from bhps.israel_wave_matrix import analytic_robin_symmetrizer,physical_block_separation_determinant


def wall_parameters(background,gamma,wall_index):
    index=0 if wall_index==0 else -1
    target=background["v0"] if wall_index==0 else background["v1"]
    derivative=float(gamma*(background["phi"][index]-target))
    coefficient=float(1. if wall_index==0 else -background["beta_b"])
    return coefficient,derivative


def scan_neighborhood(background,gamma,wall_index):
    center_c,center_uprime=wall_parameters(background,gamma,wall_index)
    records=[]
    for c_factor in np.linspace(.9,1.1,9):
        for uprime_factor in np.linspace(.5,1.5,9):
            for gamma_factor in np.linspace(.8,1.2,9):
                c=center_c*c_factor
                uprime=center_uprime*uprime_factor
                stiffness=gamma*gamma_factor
                audit=analytic_robin_symmetrizer(
                    c,uprime/6,uprime/2,stiffness,
                )
                records.append({
                    "c":c,"wall_potential_derivative":uprime,
                    "wall_stiffness":stiffness,
                    "separation":audit["spectral_separation_singular_value"],
                    "minimum_symmetrizer_eigenvalue":audit["minimum_eigenvalue"],
                    "condition_number":audit["condition_number"],
                    "symmetry_defect":audit["symmetry_defect"],
                    "block_diagonalization_defect":audit["block_diagonalization_defect"],
                    "separation_determinant":physical_block_separation_determinant(c,uprime,stiffness),
                })
    c_abs_min=.9*abs(center_c);c_abs_max=1.1*abs(center_c)
    gamma_min=.8*gamma;uprime_abs_max=1.5*abs(center_uprime)
    if center_c>0:
        determinant_abs_lower_bound=(
            5*c_abs_min*gamma_min-20*c_abs_max*c_abs_max-uprime_abs_max*uprime_abs_max/3
        )
        determinant_sign="positive"
    else:
        determinant_abs_lower_bound=5*c_abs_min*gamma_min+20*c_abs_min*c_abs_min
        determinant_sign="negative"
    return {
        "wall":"lower" if wall_index==0 else "upper",
        "center_umbilic_coefficient":center_c,
        "center_wall_potential_derivative":center_uprime,
        "relative_ranges":{
            "umbilic_coefficient":[.9,1.1],
            "wall_potential_derivative":[.5,1.5],
            "wall_stiffness":[.8,1.2],
        },
        "sample_count":len(records),
        "analytic_separation_determinant_sign":determinant_sign,
        "analytic_absolute_determinant_lower_bound":float(determinant_abs_lower_bound),
        "minimum_sampled_absolute_determinant":float(min(abs(x["separation_determinant"]) for x in records)),
        "minimum_block_separation":float(min(x["separation"] for x in records)),
        "minimum_symmetrizer_eigenvalue":float(min(x["minimum_symmetrizer_eigenvalue"] for x in records)),
        "maximum_symmetrizer_condition_number":float(max(x["condition_number"] for x in records)),
        "maximum_symmetry_defect":float(max(x["symmetry_defect"] for x in records)),
        "maximum_block_diagonalization_defect":float(max(x["block_diagonalization_defect"] for x in records)),
    }


cases=[]
for gamma in (20.,100.):
    background=solve_gw_background(
        np.linspace(1,np.e,257),epsilon=.1,backreaction=.01,
        wall_stiffness=gamma,tolerance=1e-11,
    )
    cases.append({
        "wall_stiffness":gamma,
        "walls":[scan_neighborhood(background,gamma,wall) for wall in (0,1)],
    })

walls=[wall for case in cases for wall in case["walls"]]
payload={
    "status":"explicit_smooth_robin_symmetrizer_uniform_on_sampled_production_neighborhoods",
    "construction":{
        "tangential_eigenvalue":"a=-2c with multiplicity ten",
        "normal_scalar_symmetrizer":"diag(1,16*kappa5_squared/3)",
        "shear_equation":"X(a I-D)=-B",
        "full_symmetrizer":"W=T^{-T} diag(I_10,H_D,1) T^{-1}",
        "smoothness_condition":"a I-D remains invertible",
    },
    "cases":cases,
    "total_neighborhood_samples":sum(wall["sample_count"] for wall in walls),
    "all_sampled_neighborhoods_uniform":bool(all(
        wall["minimum_block_separation"]>1e-3
        and wall["analytic_absolute_determinant_lower_bound"]>0
        and wall["minimum_symmetrizer_eigenvalue"]>1e-3
        and wall["maximum_symmetry_defect"]<1e-10
        and wall["maximum_block_diagonalization_defect"]<1e-10
        for wall in walls
    )),
    "energy_identity":{
        "boundary_flux":"u_t^T W partial_n u = (1/2) partial_t(u^T W R u) - (1/2) u^T partial_t(W R) u",
        "variable_terms":"derivatives of W and W R are lower order and controlled locally by trace and Gronwall estimates",
        "meaning":"supplies the boundary algebra for a local variable-coefficient linear energy estimate",
    },
    "limitations":[
        "finite neighborhood scan supports but does not replace coefficient bounds in a written theorem",
        "quasilinear iteration and differentiated estimates remain to be written",
        "higher corner compatibility remains open",
        "no nonlinear evolution has been run",
    ],
}
Path("results/variable_symmetrizer_audit.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps(payload,indent=2))
