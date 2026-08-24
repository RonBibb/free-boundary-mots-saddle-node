#!/usr/bin/env python3
"""Coupled gauge-invariant scalar--radion spectrum audit."""

import json,sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.gw_background import solve_gw_background
from bhps.radion_effective import finite_interval_weak_radion_mass_squared,leading_radion_mass
from bhps.scalar_radion import coupled_scalar_radion_spectrum,frozen_wentzell_boundary_symbol,shoot_lowest_scalar_radion_mode


def solve_case(size,epsilon,backreaction,wall_stiffness,shoot=False):
    z=np.linspace(1,np.e,int(size))
    background=solve_gw_background(
        z,epsilon=epsilon,backreaction=backreaction,
        wall_stiffness=wall_stiffness,tolerance=1e-11,
    )
    record={
        "grid_size":int(size),"epsilon":float(epsilon),
        "backreaction_b0":float(backreaction),
        "wall_stiffness":None if wall_stiffness is None else float(wall_stiffness),
        "proper_separation":background["proper_separation"],
        "background_residual":background["boundary_residual_max"],
        "max_ads_relative_deformation":background["max_ads_relative_deformation"],
        "phi_at_walls":[float(background["phi"][0]),float(background["phi"][-1])],
        "phi_z_at_walls":[float(background["phi_z"][0]),float(background["phi_z"][-1])],
        "minimum_abs_phi_z":float(np.min(np.abs(background["phi_z"]))),
        "phi_z_sign_change":bool(np.any(background["phi_z"][:-1]*background["phi_z"][1:]<0)),
    }
    try:
        spectrum=coupled_scalar_radion_spectrum(
            z,background["psi"],background["psi_z"],background["phi"],
            background["phi_z"],background["mass_squared"],wall_stiffness,count=6,
        )
    except ValueError as error:
        record["master_variable_admissible"]=False
        record["status"]="open_requires_regular_undivided_coupled_formulation"
        record["reason"]=str(error)
        return record
    record.update({
        "master_variable_admissible":True,
        "status":"positive_coupled_spectrum",
        "wall_alphas":[float(x) for x in spectrum["wall_alphas"]],
        "boundary_mass_weights":[float(x) for x in spectrum["boundary_mass_weights"]],
        "mu_squared":[float(x) for x in spectrum["mu_squared"]],
        "minimum_mu_squared":spectrum["minimum_mu_squared"],
        "all_positive":spectrum["all_positive"],
        "positive_wall_weights":spectrum["positive_wall_weights"],
        "bulk_gradient_weights_at_walls":[
            float(1/(background["psi"][0]*background["phi_z"][0]**2)),
            float(1/(background["psi"][-1]*background["phi_z"][-1]**2)),
        ],
        "bulk_master_mass_squared_at_walls":[
            float(2*background["phi_z"][0]**2/3),
            float(2*background["phi_z"][-1]**2/3),
        ],
    })
    if shoot:
        shooting=shoot_lowest_scalar_radion_mode(
            z,background["psi"],background["psi_z"],background["phi"],
            background["phi_z"],background["mass_squared"],wall_stiffness,
            eigenvalue_hint=spectrum["minimum_mu_squared"],
        )
        record["shooting_mu_squared"]=shooting["mu_squared"]
        record["shooting_boundary_residual"]=shooting["boundary_residual"]
        record["shooting_to_finite_element_relative_difference"]=float(
            abs(shooting["mu_squared"]/spectrum["minimum_mu_squared"]-1)
        )
    return record


finite_wall=[solve_case(257,.1,.01,gamma,shoot=True) for gamma in (2.,5.,20.,100.)]
stiff_grid=[solve_case(size,.1,.01,None) for size in (65,129,257,513)]
finite_grid=[solve_case(size,.1,.01,20.) for size in (65,129,257,513)]

transition_grid=np.linspace(1,np.e,257)
def upper_wall_phi_z(gamma):
    return float(solve_gw_background(
        transition_grid,epsilon=.1,backreaction=.01,wall_stiffness=float(gamma),
        tolerance=1e-11,
    )["phi_z"][-1])

transition_samples=[
    {"wall_stiffness":float(gamma),"upper_wall_phi_z":upper_wall_phi_z(gamma)}
    for gamma in (2.,5.,7.5,10.,20.)
]
turning_threshold=float(brentq(upper_wall_phi_z,7.5,10.,xtol=1e-11))

weak_controls=[]
for epsilon in (.02,.05,.1):
    for backreaction in (1e-4,1e-3,1e-2):
        case=solve_case(513,epsilon,backreaction,None)
        ratio=np.exp(-epsilon)
        leading=leading_radion_mass(epsilon,backreaction*ratio**2,1.)**2
        finite=finite_interval_weak_radion_mass_squared(epsilon,backreaction,1.)
        case["leading_effective_mu_squared"]=float(leading)
        case["full_to_leading_ratio"]=float(case["minimum_mu_squared"]/leading)
        case["finite_interval_weak_mu_squared"]=float(finite)
        case["full_to_finite_interval_weak_ratio"]=float(case["minimum_mu_squared"]/finite)
        weak_controls.append(case)

boundary_symbol_audits=[]
for case in finite_wall:
    if not case["master_variable_admissible"]:continue
    for wall in (0,1):
        magnitudes=[];unstable=[]
        for real in np.logspace(-3,2,16):
            for imag in np.linspace(-20,20,25):
                for wave in np.linspace(0,20,17):
                    symbol=frozen_wentzell_boundary_symbol(
                        real,imag,wave,
                        case["bulk_gradient_weights_at_walls"][wall],
                        case["boundary_mass_weights"][wall],
                        case["bulk_master_mass_squared_at_walls"][wall],
                    )
                    magnitudes.append(symbol["normalized_determinant_magnitude"])
                    unstable.append(symbol["unstable_root"])
        boundary_symbol_audits.append({
            "wall_stiffness":case["wall_stiffness"],"wall_index":wall,
            "sample_count":len(magnitudes),
            "minimum_normalized_determinant_magnitude":float(min(magnitudes)),
            "unstable_root_detected":bool(any(unstable)),
        })

admissible=[case for case in finite_wall+stiff_grid+finite_grid+weak_controls if case["master_variable_admissible"]]
payload={
    "status":"coupled_scalar_radion_positive_for_monotone_profiles_turning_profile_cases_open",
    "literature_anchor":{
        "paper":"Boos, Mikhailov, Smolyakov, Volobuev, Mod. Phys. Lett. A 21 (2006) 1431",
        "arxiv_version":"hep-th/0511185v4",
        "source_sha256":"5d005947eab89245ce370e576bd293db72ff301991e09d803d4f3602720658c1",
        "mapped_equations":"source labels ug0, eq_bc--eq_bc2, eq_sc, and parconstr",
    },
    "finite_wall_scan":finite_wall,
    "turning_profile_transition":{
        "criterion":"upper-wall Phi_z=0 while lower-wall Phi_z<0",
        "critical_wall_stiffness":turning_threshold,
        "samples":transition_samples,
    },
    "stiff_wall_grid_refinement":stiff_grid,
    "gamma20_grid_refinement":finite_grid,
    "weak_backreaction_effective_comparison":weak_controls,
    "finite_wall_wentzell_boundary_symbols":boundary_symbol_audits,
    "all_admissible_cases_positive":all(
        case["all_positive"] and case["positive_wall_weights"]
        for case in admissible
    ),
    "all_requested_finite_wall_cases_resolved":all(case["master_variable_admissible"] for case in finite_wall),
    "all_admissible_wentzell_symbol_samples_stable":not any(case["unstable_root_detected"] for case in boundary_symbol_audits),
    "interpretation":[
        "The physical scalar tower mixes the stabilizer and scalar metric/radion perturbations.",
        "Finite wall potentials enter the generalized mass matrix through eigenvalue-dependent Israel data.",
        "Those terms are positive boundary kinetic energies and their frozen Wentzell symbols have no sampled unstable root.",
        "The gamma=2 and gamma=5 backgrounds turn in field space, making the divided master variable singular; no spectral conclusion is drawn for them.",
        "Positivity here is a linear one-dimensional spectral result, not a nonlinear IBVP proof.",
    ],
    "limitations":[
        "uses the Poincare-invariant one-dimensional background",
        "does not yet test vector, gauge, constraint-propagation, or nonlinear Israel boundary sectors",
        "agreement with the leading effective radion formula must be judged only in its small-epsilon and small-backreaction regime",
    ],
}
Path("results/scalar_radion_audit.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2))
