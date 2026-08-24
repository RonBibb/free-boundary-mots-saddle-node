#!/usr/bin/env python3
"""Complete the time-time Israel second corner on the shared corrected cases."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_geometry import anisotropic_metric_acceleration,anisotropic_scalar_acceleration
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice
from bhps.lapse_acceleration_corner import construct_minimum_norm_lapse_acceleration_completion,time_time_israel_second_corner_fields
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
    phi=archive[f"phi_{name}"];psi=1/(reference["z"][:,None]+archive[f"q_{name}"])
    a,b,c=(archive[f"{field}_{name}"] for field in "abc")
    acceleration=anisotropic_metric_acceleration(
        reference["z"],reference["r"],psi,a,b,c,phi,chi_r,chi_z,
        float(reference["background"]["mass_squared"]),chi=chi,
        stencil_width=7,lapse=psi,
    )
    scalar_acceleration=anisotropic_scalar_acceleration(
        reference["z"],reference["r"],psi,a,b,c,phi,
        float(reference["background"]["mass_squared"]),lapse=psi,stencil_width=7,
    )
    buffer=radial_buffer_for_cutoff(reference["r"],6.75)
    baseline=time_time_israel_second_corner_fields(
        acceleration,psi,psi,a,phi,reference["background"],scalar_acceleration,
        None,buffer,
    )
    completion=construct_minimum_norm_lapse_acceleration_completion(
        reference["z"],acceleration,psi,psi,a,phi,reference["background"],
        scalar_acceleration,
    )
    completed=time_time_israel_second_corner_fields(
        acceleration,psi,psi,a,phi,reference["background"],scalar_acceleration,
        completion["lapse_acceleration"],buffer,
    )
    cases.append({
        "name":name,"grid_size":[nz,nr],"r_max":r_max,
        "scalar_acceleration_maximum":float(np.max(np.abs(scalar_acceleration))),
        "baseline_maximum_normalized":float(max(
            np.max(np.abs(wall["residual"])/wall["scale"]) for wall in baseline["walls"]
        )),
        "completed_maximum_normalized":float(max(
            np.max(np.abs(wall["residual"])/wall["scale"]) for wall in completed["walls"]
        )),
        "maximum_absolute_lapse_acceleration":completion["maximum_absolute_lapse_acceleration"],
        "maximum_relative_lapse_acceleration":completion["maximum_relative_lapse_acceleration"],
        "lapse_acceleration_rms":completion["lapse_acceleration_rms"],
    })

payload={
    "status":"minimum_norm_time_time_israel_completion_passes" if max(
        case["completed_maximum_normalized"] for case in cases
    )<1e-8 else "time_time_israel_second_corner_completion_fails",
    "corner_state":"results/high_order_physical_corner_shared_grid_domain_solve_state.npz",
    "cases":cases,
    "interpretation":"The time-time row is linearly soluble by a lapse-acceleration gauge jet after the physical spatial correction. Endpoint values minimize the discrete L2 relative acceleration; bounded generalized-harmonic source compatibility remains separate.",
    "limitations":[
        "relative lapse acceleration remains about five",
        "generalized-harmonic gauge-source compatibility remains",
        "common physical interval r <= 6.75",
        "not an evolution",
    ],
}
Path("results/time_time_lapse_acceleration_corner.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps(payload,indent=2))
