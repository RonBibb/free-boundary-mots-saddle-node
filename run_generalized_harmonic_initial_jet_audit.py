#!/usr/bin/env python3
"""Audit generalized-harmonic source jets after all second-corner repairs."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_geometry import anisotropic_metric_acceleration,anisotropic_scalar_acceleration
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice
from bhps.generalized_harmonic_jets import diagonal_spatial_source_second_jets,initial_contracted_christoffel_time_jet,initial_normal_contracted_christoffel,spatial_metric_acceleration_trace
from bhps.lapse_acceleration_corner import construct_minimum_norm_lapse_acceleration_completion,construct_projected_target_lapse_acceleration_completion,time_time_israel_second_corner_fields
from bhps.physical_corner_corrector import radial_buffer_for_cutoff
from bhps.scalar_pulse import scalar_pulse


settings=(
    ("G5R8",49,73,8.,8.572038845434301),
    ("G6R8",65,97,8.,8.572368541895216),
    ("G5R10",49,91,10.,8.591525244593445),
    ("G5R12",49,109,12.,8.605669112233226),
)
archive=np.load("results/high_order_physical_corner_shared_grid_domain_solve_state.npz")
cases=[]
for name,nz,nr,r_max,amplitude in settings:
    reference=solve_finite_wall_high_order_slice(
        amplitude,nz=nz,nr=nr,r_max=r_max,wall_stiffness=20.,epsilon=.1,
        backreaction=.01,tolerance=1e-10,iterations=240,
    )
    chi,chi_r,chi_z=scalar_pulse(reference["z"],reference["r"],amplitude)
    phi=archive[f"phi_{name}"]
    psi=1/(reference["z"][:,None]+archive[f"q_{name}"])
    a,b,c=(archive[f"{field}_{name}"] for field in "abc")
    acceleration=anisotropic_metric_acceleration(
        reference["z"],reference["r"],psi,a,b,c,phi,chi_r,chi_z,
        float(reference["background"]["mass_squared"]),chi=chi,
        stencil_width=7,lapse=psi,
    )
    scalar_acceleration=anisotropic_scalar_acceleration(
        reference["z"],reference["r"],psi,a,b,c,phi,
        float(reference["background"]["mass_squared"]),lapse=psi,
        stencil_width=7,
    )
    trace=spatial_metric_acceleration_trace(acceleration,psi,a,b,c)
    target_relative=.5*trace
    old_completion=construct_minimum_norm_lapse_acceleration_completion(
        reference["z"],acceleration,psi,psi,a,phi,reference["background"],
        scalar_acceleration,
    )
    old_jet=initial_contracted_christoffel_time_jet(
        acceleration,psi,old_completion["lapse_acceleration"],psi,a,b,c,
    )
    completion=construct_projected_target_lapse_acceleration_completion(
        reference["z"],acceleration,psi,psi,a,phi,reference["background"],
        scalar_acceleration,target_relative,
    )
    jet=initial_contracted_christoffel_time_jet(
        acceleration,psi,completion["lapse_acceleration"],psi,a,b,c,
    )
    spatial_jets=diagonal_spatial_source_second_jets(
        reference["z"],reference["r"],acceleration,psi,
        completion["lapse_acceleration"],psi,a,b,c,7,
    )
    gamma_z=initial_normal_contracted_christoffel(
        reference["z"],reference["r"],psi,psi,a,b,c,7,
    )
    buffer=radial_buffer_for_cutoff(reference["r"],6.75)
    domain=(slice(None),slice(None,-buffer) if buffer else slice(None))
    completed=time_time_israel_second_corner_fields(
        acceleration,psi,psi,a,phi,reference["background"],scalar_acceleration,
        completion["lapse_acceleration"],buffer,
    )
    gamma0_t=jet["gamma_0_time_derivative"][domain]
    old_gamma0_t=old_jet["gamma_0_time_derivative"][domain]
    relative=jet["relative_lapse_acceleration"][domain]
    half_trace=.5*trace[domain]
    cases.append({
        "name":name,"grid_size":[nz,nr],"r_max":r_max,
        "radial_buffer":buffer,"audit_radius_maximum":6.75,
        "time_time_corner_maximum_normalized":float(max(
            np.max(np.abs(wall["residual"])/wall["scale"])
            for wall in completed["walls"]
        )),
        "minimum_lapse_choice_gamma0_time_jet_maximum":float(np.max(np.abs(old_gamma0_t))),
        "projected_harmonic_gamma0_time_jet_maximum":float(np.max(np.abs(gamma0_t))),
        "projected_harmonic_gamma0_time_jet_rms":float(np.sqrt(np.mean(gamma0_t**2))),
        "projected_harmonic_gamma0_relative_to_half_trace":float(
            np.max(np.abs(gamma0_t))/np.max(np.abs(half_trace))
        ),
        "maximum_relative_lapse_acceleration":float(np.max(np.abs(relative))),
        "maximum_half_spatial_metric_trace_acceleration":float(np.max(np.abs(half_trace))),
        "nonzero_projection_fraction":float(np.mean(np.abs(gamma0_t)>1e-10)),
        "initial_normal_gamma_z_maximum":float(np.max(np.abs(gamma_z[domain]))),
        "normal_gamma_z_second_time_jet_maximum":float(np.max(np.abs(
            spatial_jets["gamma_z_second_time_derivative"][domain]
        ))),
        "radial_gamma_r_second_time_jet_maximum":float(np.max(np.abs(
            spatial_jets["gamma_r_second_time_derivative"][domain]
        ))),
    })

maximum_corner=max(case["time_time_corner_maximum_normalized"] for case in cases)
maximum_source_ratio=max(case["projected_harmonic_gamma0_relative_to_half_trace"] for case in cases)
payload={
    "status":"initial_harmonic_source_jet_completion_passes_discrete_audit"
    if maximum_corner<1e-10 and maximum_source_ratio<2e-3
    else "initial_harmonic_source_jet_completion_requires_revision",
    "corner_state":"results/high_order_physical_corner_shared_grid_domain_solve_state.npz",
    "cases":cases,
    "derivation":{
        "time_component":"Gamma_0,t = alpha_tt/alpha - (gamma^ij gamma_ij,tt)/2 at zero shift and vanishing first time jets",
        "normal_component":"Gamma_z,tt = d_z[(gamma_zz,tt)/(2 A^2) - (gamma_rr,tt)/(2 B^2) - (gamma_perp,tt)/(C^2) - alpha_tt/alpha]",
        "projection":"minimize the discrete L2 Gamma_0,t correction subject to the exact two time-time Israel rows",
    },
    "interpretation":"The raw lapse acceleration is large because it tracks the physical spatial-volume acceleration. After projection onto the exact time-time corner rows, the required lower contracted-Christoffel time-source jet is about one part in one thousand of that physical trace scale. The normal and radial spatial source second jets are finite but remain data-dependent gauge-source prescriptions, not an IBVP theorem.",
    "limitations":[
        "the minimum correction is defined with the same discrete normal derivative used by the corner audit",
        "the correction is localized to the boundary derivative stencil and needs continuum/localization testing",
        "large alpha_tt/alpha reflects large physical trace acceleration and may still constrain evolution timesteps",
        "matching finite initial source jets does not prove nonlinear generalized-harmonic well posedness",
        "common physical interval r <= 6.75",
    ],
}
Path("results/generalized_harmonic_initial_jet_audit.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps(payload,indent=2))
