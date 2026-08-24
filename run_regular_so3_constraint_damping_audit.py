#!/usr/bin/env python3
"""Audit regular SO(3) GH constraints and constraint damping controls."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_geometry import anisotropic_metric_acceleration,anisotropic_scalar_acceleration
from bhps.anisotropic_initial_data import solve_anisotropic_initial_data
from bhps.constraint_ibvp import (
    flat_regular_so3_gauge_constraint,frozen_constraint_mode_spectrum,
    linearized_regular_so3_gauge_constraint,
)
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice
from bhps.generalized_harmonic_jets import spatial_metric_acceleration_trace
from bhps.lapse_acceleration_corner import construct_localized_target_lapse_acceleration_completion
from bhps.physical_corner_corrector import combine_shape_modes,tracefree_shape_basis
from bhps.regular_so3_gh_reduction import RegularSO3BackgroundJetField,regular_so3_gh_coefficient_matrices,regular_so3_perturbation_jets
from bhps.scalar_pulse import scalar_pulse


def build_corrected_g6_field():
    fold=json.loads(Path("results/corrected_anisotropic_arclength_G6.json").read_text())
    amplitude=float(fold["summary"]["fine_fold_amplitude"]);name="G6R8";nz,nr=65,97
    archive=np.load("results/corrected_family_knot_A8_state.npz");coefficients=archive["coefficients"]
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
        "field":field,"z":z,"mass_squared":mass,"fold_amplitude":amplitude,
        "selector_maximum":float(selected["maximum_residual"]),
    }


def independent_constraint_control():
    rng=np.random.default_rng(20260813);radius=.73
    values=rng.normal(size=9);first=rng.normal(size=(3,9))
    second=rng.normal(size=(3,3,9));second=.5*(second+second.swapaxes(0,1))
    source=rng.normal(size=3);eta=np.diag((-1.,1.,1.,1.,1.))
    background={
        "metric":eta,"metric_first":np.zeros((5,5,5)),
        "metric_second":np.zeros((5,5,5,5)),
    }
    perturbation=regular_so3_perturbation_jets(radius,values,first,second)
    full=linearized_regular_so3_gauge_constraint(background,perturbation,radius,source)
    closed=flat_regular_so3_gauge_constraint(radius,values,first,second,source)
    axis_first=first.copy();axis_first[2]=0.
    axis=flat_regular_so3_gauge_constraint(0.,values,axis_first,second,source)
    errors=[]
    for test_radius in (1e-2,1e-3,1e-4):
        off_first=axis_first.copy();off_first[2]=test_radius*second[2,2]
        off=flat_regular_so3_gauge_constraint(test_radius,values,off_first,second,source)
        errors.append(float(np.max(np.abs(off-axis))))
    return {
        "full_cartesian_vs_closed_form_maximum_error":float(np.max(np.abs(full-closed))),
        "axis_limit_errors":errors,"axis_values":[float(value) for value in axis],
    }


def modal_controls():
    flat=[]
    for damping in (.25,1.,4.):
        for wave in (0.,.02,.2,2.,20.):
            result=frozen_constraint_mode_spectrum(wave,damping)
            flat.append({
                "damping_rate":damping,"wavenumber":wave,
                "maximum_real_part":result["maximum_real_part"],
                "strictly_damped":result["strictly_damped"],
                "constant_mode_exception":result["constant_mode_exception"],
            })
    ads=[]
    for ell in (.5,1.,3.):
        for wave in (0.,.2,2.,20.):
            damping=1/ell
            result=frozen_constraint_mode_spectrum(
                wave,damping,ricci_mixed_eigenvalue=-4/ell**2,
            )
            ads.append({
                "ads_radius":ell,"damping_rate":damping,"wavenumber":wave,
                "maximum_real_part":result["maximum_real_part"],
                "strictly_damped":result["strictly_damped"],
            })
    return flat,ads


def corrected_fold_coefficient_control(case,damping_rate=1.):
    coordinates=[
        (float(value),radius)
        for value in np.geomspace(case["z"][0],case["z"][-1],3)
        for radius in (.25,1.,3.)
    ]
    samples=[]
    for index,(z_value,radius) in enumerate(coordinates):
        print(f"damped coefficient sample {index+1}/{len(coordinates)}",flush=True)
        background=case["field"].at(z_value,radius)
        undamped=regular_so3_gh_coefficient_matrices(
            background,radius,mass_squared=case["mass_squared"],potential_offset=-6.,
        )
        damped=regular_so3_gh_coefficient_matrices(
            background,radius,mass_squared=case["mass_squared"],potential_offset=-6.,
            constraint_damping=damping_rate,
        )
        principal_change=float(np.max(np.abs(
            damped["pure_second_matrices"]-undamped["pure_second_matrices"]
        )))
        reaction_change=float(np.linalg.norm(
            damped["evolution_reaction_matrix"]-undamped["evolution_reaction_matrix"]
        ))
        first_change=float(np.linalg.norm(
            damped["evolution_first_matrices"]-undamped["evolution_first_matrices"]
        ))
        scalar_change=max(
            float(np.max(np.abs(
                damped["evolution_reaction_matrix"][7:]-undamped["evolution_reaction_matrix"][7:]
            ))),
            float(np.max(np.abs(
                damped["evolution_first_matrices"][:,7:]-undamped["evolution_first_matrices"][:,7:]
            ))),
        )
        samples.append({
            "z":z_value,"r":radius,"principal_change":principal_change,
            "reaction_change_frobenius":reaction_change,
            "first_change_frobenius":first_change,
            "scalar_row_change_maximum":scalar_change,
            "damped_principal_identity_defect":damped["principal_identity_maximum_defect"],
            "all_finite":bool(damped["finite"]),
        })
    return samples


identity=independent_constraint_control();flat,ads=modal_controls()
case=build_corrected_g6_field();samples=corrected_fold_coefficient_control(case)
acceptance={
    "independent_constraint_identity_below_1e_11":identity["full_cartesian_vs_closed_form_maximum_error"]<1e-11,
    "regular_axis_limit_converges":identity["axis_limit_errors"][-1]<1e-7,
    "all_nonconstant_flat_modes_strictly_damped":all(
        item["strictly_damped"] for item in flat if item["wavenumber"]>0
    ),
    "flat_constant_exception_reproduced":all(
        item["constant_mode_exception"] and not item["strictly_damped"]
        for item in flat if item["wavenumber"]==0
    ),
    "all_frozen_ads5_modes_strictly_damped":all(item["strictly_damped"] for item in ads),
    "corrected_fold_damping_leaves_principal_symbol_unchanged":max(item["principal_change"] for item in samples)<1e-11,
    "corrected_fold_damping_changes_constraint_lower_order_terms":min(item["first_change_frobenius"] for item in samples)>1e-6,
    "corrected_fold_scalar_rows_unchanged":max(item["scalar_row_change_maximum"] for item in samples)<1e-11,
    "corrected_fold_coefficients_finite":all(item["all_finite"] for item in samples),
    "selector_below_1e_8":case["selector_maximum"]<1e-8,
}
payload={
    "status":"pass" if all(acceptance.values()) else "review",
    "scope":"regular SO(3) GH constraint identity, frozen subsidiary modes, and corrected-fold damping coefficient audit",
    "constraint_convention":"C_a = Gamma_a - H_a",
    "damping_addition":"kappa [n_(a C_b) - (1+rho) g_ab n^c C_c/2], rho=0",
    "regular_constraint_order":["C_0","C_z","c_r=C_r/r"],
    "independent_constraint_control":identity,
    "flat_modal_controls":flat,"frozen_ads5_modal_controls":ads,
    "corrected_fold":{
        "case":"G6R8","fold_amplitude":case["fold_amplitude"],
        "selector_maximum":case["selector_maximum"],"damping_rate":1.,
        "coefficient_samples":samples,
    },
    "acceptance":acceptance,
    "limitations":[
        "flat and AdS5 subsidiary spectra are frozen orthonormal-frame linear controls",
        "the constant Minkowski constraint mode is not damped by the standard rho=0 addition",
        "corrected-fold checks sample nine off-axis points rather than every runtime node",
        "a damping-parameter window has not yet been selected by a freely evolving constraint pulse",
        "the generalized-harmonic source remains fixed in this audit",
    ],
}
Path("results/regular_so3_constraint_damping_audit.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps({"status":payload["status"],"acceptance":acceptance},indent=2))
