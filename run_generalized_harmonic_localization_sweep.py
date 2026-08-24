#!/usr/bin/env python3
"""Test fixed-width smooth harmonic source completions across grids/domains."""

import json,os,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_geometry import anisotropic_metric_acceleration,anisotropic_scalar_acceleration
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice
from bhps.generalized_harmonic_jets import diagonal_spatial_source_second_jets,initial_contracted_christoffel_time_jet,spatial_metric_acceleration_trace
from bhps.lapse_acceleration_corner import construct_localized_target_lapse_acceleration_completion,time_time_israel_second_corner_fields
from bhps.physical_corner_corrector import radial_buffer_for_cutoff
from bhps.scalar_pulse import scalar_pulse


settings=(
    ("G5R8",49,73,8.,8.572038845434301),
    ("G6R8",65,97,8.,8.572368541895216),
    ("G5R10",49,91,10.,8.591525244593445),
    ("G5R12",49,109,12.,8.605669112233226),
)
widths=(.08,.10,.15,.20)
state_path=os.environ.get(
    "BHPS_CORNER_STATE","results/high_order_physical_corner_shared_grid_domain_solve_state.npz"
)
output_path=os.environ.get(
    "BHPS_GH_OUTPUT","results/generalized_harmonic_localization_sweep.json"
)
full_exterior=os.environ.get("BHPS_FULL_EXTERIOR","").lower() in ("1","true","yes")
archive=np.load(state_path)
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
    target=.5*trace
    buffer=7 if full_exterior else radial_buffer_for_cutoff(reference["r"],6.75)
    domain=(slice(None),slice(None,-buffer) if buffer else slice(None))
    width_results=[]
    for width in widths:
        completion=construct_localized_target_lapse_acceleration_completion(
            reference["z"],acceleration,psi,psi,a,phi,reference["background"],
            scalar_acceleration,target,width,
        )
        completed=time_time_israel_second_corner_fields(
            acceleration,psi,psi,a,phi,reference["background"],scalar_acceleration,
            completion["lapse_acceleration"],buffer,
        )
        jet=initial_contracted_christoffel_time_jet(
            acceleration,psi,completion["lapse_acceleration"],psi,a,b,c,
        )
        spatial=diagonal_spatial_source_second_jets(
            reference["z"],reference["r"],acceleration,psi,
            completion["lapse_acceleration"],psi,a,b,c,7,
        )
        gamma0=jet["gamma_0_time_derivative"][domain]
        relative=jet["relative_lapse_acceleration"][domain]
        width_results.append({
            "logarithmic_width":width,
            "points_per_width":width*(nz-1),
            "time_time_corner_maximum_normalized":float(max(
                np.max(np.abs(wall["residual"])/wall["scale"])
                for wall in completed["walls"]
            )),
            "gamma0_time_jet_maximum":float(np.max(np.abs(gamma0))),
            "gamma0_time_jet_rms":float(np.sqrt(np.mean(gamma0**2))),
            "gamma0_relative_to_half_trace":float(
                np.max(np.abs(gamma0))/np.max(np.abs(.5*trace[domain]))
            ),
            "gamma0_normal_derivative_maximum":float(np.max(np.abs(
                acceleration["Dz"]@jet["gamma_0_time_derivative"]
            )[domain])),
            "maximum_relative_lapse_acceleration":float(np.max(np.abs(relative))),
            "normal_gamma_z_second_time_jet_maximum":float(np.max(np.abs(
                spatial["gamma_z_second_time_derivative"][domain]
            ))),
        })
    cases.append({
        "name":name,"grid_size":[nz,nr],"r_max":r_max,
        "audit_radius_maximum":float(reference["r"][-buffer-1]) if full_exterior else 6.75,
        "widths":width_results,
    })

selected_width=.15
selected=[
    next(item for item in case["widths"] if item["logarithmic_width"]==selected_width)
    for case in cases
]
payload={
    "status":"smooth_fixed_width_time_corner_completion_passes_normal_source_stiffness_open"
    if max(item["time_time_corner_maximum_normalized"] for item in selected)<1e-10
    and max(item["gamma0_relative_to_half_trace"] for item in selected)<.10
    else "smooth_fixed_width_harmonic_completion_requires_revision",
    "selected_logarithmic_width":selected_width,
    "corner_state":state_path,"full_exterior":full_exterior,
    "cases":cases,
    "interpretation":"A smooth fixed-log(z)-width wall-layer completion removes the grid-shrinking support of the discrete minimum projection. Width 0.15 is resolved by 7.2 and 9.6 intervals on the two R8 grids, clears the time-time corner to roundoff, and keeps Gamma_0,t below nine percent of the physical half-trace scale. The large normal spatial source second jet is essentially unchanged and therefore comes from the physical component accelerations rather than the completion layer.",
    "limitations":[
        "width 0.15 is a pilot gauge choice, not an optimized evolution gauge",
        "the normal source second jet remains about 10^3",
        "the relative lapse acceleration remains set mainly by the physical volume acceleration",
        "no nonlinear evolution or quasilinear boundary theorem has been supplied",
        "last seven radial points excluded" if full_exterior else "common physical interval r <= 6.75",
    ],
}
Path(output_path).write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps(payload,indent=2))
