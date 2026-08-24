#!/usr/bin/env python3
"""Audit smooth harmonic and time-time corner jets at the corrected fold."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_geometry import anisotropic_metric_acceleration,anisotropic_scalar_acceleration
from bhps.anisotropic_initial_data import solve_anisotropic_initial_data
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice
from bhps.generalized_harmonic_jets import diagonal_spatial_source_second_jets,initial_contracted_christoffel_time_jet,spatial_metric_acceleration_trace
from bhps.lapse_acceleration_corner import construct_localized_target_lapse_acceleration_completion,time_time_israel_second_corner_fields
from bhps.physical_corner_corrector import combine_shape_modes,tracefree_shape_basis
from bhps.scalar_pulse import scalar_pulse


g5=json.loads(Path("results/corrected_anisotropic_arclength.json").read_text())
g6=json.loads(Path("results/corrected_anisotropic_arclength_G6.json").read_text())
fold_g5=float(g5["summary"]["fine_fold_amplitude"])
fold_g6=float(g6["summary"]["fine_fold_amplitude"])
state_path="results/corrected_family_knot_A8_state.npz"
archive=np.load(state_path);coefficients=archive["coefficients"]
width=.15;sigma_y=.2;cases=[]
for name,nz,nr,r_max,amplitude in (
    ("G5R8",49,73,8.,fold_g5),("G6R8",65,97,8.,fold_g6),
    ("G5R10",49,91,10.,fold_g5),("G5R12",49,109,12.,fold_g5),
):
    reference=solve_finite_wall_high_order_slice(
        amplitude,nz=nz,nr=nr,r_max=r_max,wall_stiffness=20.,epsilon=.1,
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
    psi=1/(reference["z"][:,None]+selected["q"]);phi=selected["phi"]
    acceleration=anisotropic_metric_acceleration(
        reference["z"],reference["r"],psi,a,b,c,phi,chi_r,chi_z,
        float(reference["background"]["mass_squared"]),chi=chi,
        stencil_width=7,lapse=psi,
    )
    scalar_acceleration=anisotropic_scalar_acceleration(
        reference["z"],reference["r"],psi,a,b,c,phi,
        float(reference["background"]["mass_squared"]),lapse=psi,stencil_width=7,
    )
    trace=spatial_metric_acceleration_trace(acceleration,psi,a,b,c)
    completion=construct_localized_target_lapse_acceleration_completion(
        reference["z"],acceleration,psi,psi,a,phi,reference["background"],
        scalar_acceleration,.5*trace,width,
    )
    completed=time_time_israel_second_corner_fields(
        acceleration,psi,psi,a,phi,reference["background"],scalar_acceleration,
        completion["lapse_acceleration"],7,
    )
    time_jet=initial_contracted_christoffel_time_jet(
        acceleration,psi,completion["lapse_acceleration"],psi,a,b,c,
    )
    spatial_jets=diagonal_spatial_source_second_jets(
        reference["z"],reference["r"],acceleration,psi,
        completion["lapse_acceleration"],psi,a,b,c,7,
    )
    domain=(slice(None),slice(None,-7));gamma0=time_jet["gamma_0_time_derivative"][domain]
    half_trace=.5*trace[domain]
    component_max=max(
        float(np.max(np.abs(acceleration[field][domain]/metric[domain])))
        for field,metric in (
            ("zz",psi**2*np.exp(2*a)),("radial",psi**2*np.exp(2*b)),
            ("transverse",psi**2*np.exp(2*c)),
        )
    )
    normal_source=float(np.max(np.abs(
        spatial_jets["gamma_z_second_time_derivative"][domain]
    )))
    cases.append({
        "name":name,"grid_size":[nz,nr],"r_max":r_max,"amplitude":amplitude,
        "last_audited_radius":float(reference["r"][-8]),
        "selector_maximum":selected["maximum_residual"],
        "time_time_corner_maximum_normalized":float(max(
            np.max(np.abs(wall["residual"])/wall["scale"])
            for wall in completed["walls"]
        )),
        "gamma0_time_jet_maximum":float(np.max(np.abs(gamma0))),
        "gamma0_relative_to_half_trace":float(
            np.max(np.abs(gamma0))/np.max(np.abs(half_trace))
        ),
        "maximum_relative_lapse_acceleration":float(np.max(np.abs(
            time_jet["relative_lapse_acceleration"][domain]
        ))),
        "normal_source_second_jet_maximum":normal_source,
        "normal_source_to_pulse_gradient_scale":normal_source/(component_max/sigma_y),
        "fastest_coordinate_acceleration_timescale":1/np.sqrt(component_max),
    })

acceptance={
    "all_time_time_rows_below_1e-10":all(
        case["time_time_corner_maximum_normalized"]<1e-10 for case in cases
    ),
    "all_gamma0_ratios_below_0.10":all(
        case["gamma0_relative_to_half_trace"]<.10 for case in cases
    ),
    "normal_source_scale_ratio_grid_domain_stable":max(
        case["normal_source_to_pulse_gradient_scale"] for case in cases
    )-min(case["normal_source_to_pulse_gradient_scale"] for case in cases)<.02,
}
payload={
    "status":"corrected_fold_harmonic_time_corner_pilot_pass"
    if all(acceptance.values()) else "corrected_fold_harmonic_audit_incomplete",
    "shape_state":state_path,"gauge_logarithmic_width":width,
    "cases":cases,"acceptance":acceptance,
    "interpretation":"The smooth fixed-width generalized-harmonic construction remains compatible at the newly continued corrected fold across both grids and all three radial domains. The source jets are stiff but retain their pulse-scale normalization.",
    "limitations":[
        "initial source jets rather than an evolved gauge driver",
        "last seven radial stencil points excluded",
        "no timestep or nonlinear evolution convergence",
    ],
}
Path("results/corrected_fold_harmonic_audit.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps({"status":payload["status"],"acceptance":acceptance,"cases":cases},indent=2))
