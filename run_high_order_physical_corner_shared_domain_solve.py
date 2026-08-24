#!/usr/bin/env python3
"""Fit one nonlinear physical-corner shape across R8, R10, and R12."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice
from bhps.multicase_corner import solve_shared_physical_corner
from bhps.physical_corner_corrector import radial_buffer_for_cutoff
from bhps.scalar_pulse import scalar_pulse


case_settings=((8.,73,8.572038845434301),(10.,91,8.591525244593445),(12.,109,8.605669112233226))
cases=[]
for r_max,nr,amplitude in case_settings:
    reference=solve_finite_wall_high_order_slice(
        amplitude,nz=49,nr=nr,r_max=r_max,wall_stiffness=20.,epsilon=.1,
        backreaction=.01,tolerance=1e-10,iterations=240,
    )
    chi,chi_r,chi_z=scalar_pulse(reference["z"],reference["r"],amplitude)
    cases.append({
        "name":f"R{int(r_max)}","z":reference["z"],"r":reference["r"],
        "reference_q":reference["q"],"reference_phi":reference["phi"],
        "background":reference["background"],"chi":chi,"chi_r":chi_r,"chi_z":chi_z,
        "radial_buffer":radial_buffer_for_cutoff(reference["r"],6.75),
        "fold_amplitude":amplitude,"reference_residual":reference["max_abs_residual"],
    })
seed=np.load("results/high_order_physical_corner_shared_domain_path_state.npz")["coefficients"]
result=solve_shared_physical_corner(
    cases,seed,radial_modes=6,axis_widths=(.5,1.),basis_radius=8.,
    stencil_width=7,finite_difference_step=2e-4,maximum_iterations=3,
    corner_tolerance=.025,shape_bound=.5,initial_trust_radius=.04,
    regularization=1.,selector_tolerance=1e-9,difference_scheme="forward",
    maximum_row_weight=6.,include_mixed=True,verbose=True,
)
payload={
    "status":"shared_three_domain_nonlinear_pilot_passes_0.025" if result["converged"] else "shared_three_domain_nonlinear_pilot_does_not_yet_pass_0.025",
    "retained_r_max":6.75,"settings":result["settings"],
    "cases":[{
        "name":case["name"],"grid_size":[len(case["z"]),len(case["r"])],
        "r_max":float(case["r"][-1]),"fold_amplitude":case["fold_amplitude"],
        "reference_residual":case["reference_residual"],
    } for case in cases],
    "history":result["history"],"linearizations":result["linearizations"],
    "summary":{
        "converged":result["converged"],
        "final_case_maxima":result["final_case_maxima"],
        "final_worst_case_maximum":result["final_worst_case_maximum"],
        "selector_maxima":[item["maximum_residual"] for item in result["selectors"]],
        "coefficient_l2":float(np.linalg.norm(result["coefficients"])),
        "maximum_shape_logarithm":float(max(
            max(np.max(np.abs(state[name])) for name in ("a","b","c"))
            for state in result["states"]
        )),
    },
    "limitations":[
        "three high-order fold domains at one z resolution",
        "common physical interval r <= 6.75",
        "finite 64-mode diagonal anisotropic basis",
        "time-time, harmonic, and anisotropic-horizon gates remain",
    ],
}
Path("results/high_order_physical_corner_shared_domain_solve.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
archive={"coefficients":result["coefficients"]}
for case,state in zip(cases,result["states"]):
    archive[f"q_{case['name']}"]=state["q"]
    archive[f"phi_{case['name']}"]=state["phi"]
    archive[f"a_{case['name']}"]=state["a"]
    archive[f"b_{case['name']}"]=state["b"]
    archive[f"c_{case['name']}"]=state["c"]
np.savez_compressed("results/high_order_physical_corner_shared_domain_solve_state.npz",**archive)
print(json.dumps({"status":payload["status"],"summary":payload["summary"]},indent=2))
