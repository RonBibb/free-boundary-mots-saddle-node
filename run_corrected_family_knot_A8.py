#!/usr/bin/env python3
"""Solve a second shared full-exterior corner-compatible family knot at A=8."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice
from bhps.multicase_corner import solve_shared_physical_corner
from bhps.scalar_pulse import scalar_pulse


amplitude=8.0
settings=(
    ("G5R8",49,73,8.),("G6R8",65,97,8.),
    ("G5R10",49,91,10.),("G5R12",49,109,12.),
)
source_path="results/high_order_physical_corner_full_exterior_solve_state.npz"
source=np.load(source_path);cases=[]
for name,nz,nr,r_max in settings:
    reference=solve_finite_wall_high_order_slice(
        amplitude,nz=nz,nr=nr,r_max=r_max,wall_stiffness=20.,epsilon=.1,
        backreaction=.01,tolerance=1e-10,iterations=240,
    )
    chi,chi_r,chi_z=scalar_pulse(reference["z"],reference["r"],amplitude)
    cases.append({
        "name":name,"z":reference["z"],"r":reference["r"],
        "reference_q":reference["q"],"reference_phi":reference["phi"],
        "background":reference["background"],"chi":chi,"chi_r":chi_r,"chi_z":chi_z,
        "radial_buffer":7,"initial_q":source[f"q_{name}"],
        "initial_phi":source[f"phi_{name}"],
    })
result=solve_shared_physical_corner(
    cases,source["coefficients"],radial_modes=6,axis_widths=(.5,1.),basis_radius=8.,
    annular_profiles=((7.5,1.5),(7.5,3.0)),
    stencil_width=7,finite_difference_step=2e-4,maximum_iterations=2,
    corner_tolerance=.025,shape_bound=.5,initial_trust_radius=.03,
    regularization=1.,selector_tolerance=1e-9,difference_scheme="forward",
    maximum_row_weight=8.,include_mixed=True,verbose=True,
)
payload={
    "status":"corrected_family_A8_shared_knot_passes_0.025"
    if result["converged"] else "corrected_family_A8_shared_knot_incomplete",
    "amplitude":amplitude,"source_state":source_path,
    "settings":result["settings"],"history":result["history"],
    "linearizations":result["linearizations"],
    "cases":[{
        "name":case["name"],"grid_size":[len(case["z"]),len(case["r"])],
        "r_max":float(case["r"][-1]),"last_audited_radius":float(case["r"][-8]),
    } for case in cases],
    "summary":{
        "final_case_maxima":result["final_case_maxima"],
        "final_worst_case_maximum":result["final_worst_case_maximum"],
        "selector_maxima":[item["maximum_residual"] for item in result["selectors"]],
        "coefficient_change_l2":float(np.linalg.norm(
            result["coefficients"]-source["coefficients"]
        )),
        "coefficient_l2":float(np.linalg.norm(result["coefficients"])),
        "maximum_shape_logarithm":float(max(
            max(np.max(np.abs(state[field])) for field in ("a","b","c"))
            for state in result["states"]
        )),
    },
    "interpretation":"This is the second full-exterior knot of an amplitude-dependent physical shape curve. It is solved jointly on two grids and three radial domains at A=8, immediately above the fixed-shape cap-fold bracket.",
    "limitations":[
        "single amplitude knot rather than a completed coefficient continuation",
        "cap fold must be re-scanned on the new local shape",
        "last seven radial stencil points excluded",
    ],
}
Path("results/corrected_family_knot_A8.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
state={"coefficients":result["coefficients"]}
for case,item in zip(cases,result["states"]):
    for field in ("q","phi","a","b","c"):
        state[f"{field}_{case['name']}"]=item[field]
np.savez_compressed("results/corrected_family_knot_A8_state.npz",**state)
print(json.dumps({"status":payload["status"],"summary":payload["summary"]},indent=2))
