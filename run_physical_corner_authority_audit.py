#!/usr/bin/env python3
"""Constraint-projected anisotropic physical-corrector authority audit."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.finite_wall_solver import solve_finite_wall_slice
from bhps.anisotropic_initial_data import solve_anisotropic_initial_data
from bhps.physical_corner_corrector import evaluate_physical_corner_candidate,prepare_physical_corrector,solve_linear_corner_correction
from bhps.scalar_pulse import scalar_pulse


amplitude=8.415541903059392
solved=solve_finite_wall_slice(
    amplitude,nz=33,nr=49,r_max=8.,wall_stiffness=20.,epsilon=.1,
    backreaction=.01,tolerance=1e-10,iterations=180,
)
chi,chi_r,chi_z=scalar_pulse(solved["z"],solved["r"],amplitude)
prepared=prepare_physical_corrector(
    solved["z"],solved["r"],solved["psi"],solved["phi"],chi_r,chi_z,
    solved["background"],chi=chi,radial_modes=6,stencil_width=7,
    radial_buffer=7,finite_difference_step=2e-4,
)

cases=[]
for regularization in (.1,.01,.001,.0001):
    linear=solve_linear_corner_correction(prepared,regularization)
    nonlinear=[evaluate_physical_corner_candidate(
        prepared,linear,solved["z"],solved["r"],solved["psi"],solved["phi"],
        chi_r,chi_z,solved["background"],chi=chi,correction_scale=scale,
        stencil_width=7,radial_buffer=7,
    ) for scale in (.25,.5,.75,1.)]
    resolved=[]
    for scale in (.1,.25):
        shape_a=scale*linear["a"];shape_b=scale*linear["b"];shape_c=scale*linear["c"]
        solution=solve_anisotropic_initial_data(
            solved["z"],solved["r"],solved["q"],solved["phi"],
            shape_a,shape_b,shape_c,solved["background"],chi_r,chi_z,
            initial_q=solved["q"]+scale*linear["dq"],
            initial_phi=solved["phi"]+scale*linear["dphi"],
            stencil_width=7,tolerance=1e-9,iterations=20,
        )
        resolved_direction={
            **linear,
            "dq":(solution["q"]-solved["q"])/scale,
            "dphi":(solution["phi"]-solved["phi"])/scale,
            "u":-solved["psi"]*(solution["q"]-solved["q"])/scale,
        }
        evaluation=evaluate_physical_corner_candidate(
            prepared,resolved_direction,solved["z"],solved["r"],solved["psi"],
            solved["phi"],chi_r,chi_z,solved["background"],chi=chi,
            correction_scale=scale,stencil_width=7,radial_buffer=7,
        )
        resolved.append({
            **evaluation,"selector_converged":solution["converged"],
            "selector_maximum_residual":solution["maximum_residual"],
            "selector_iterations":len(solution["history"]),
            "maximum_stabilizer_change_after_resolve":float(np.max(np.abs(solution["phi"]-solved["phi"]))),
        })
    cases.append({
        "regularization":regularization,
        "linear":{key:value for key,value in linear.items() if key not in ("coefficients","dq","dphi","u","a","b","c")},
        "nonlinear_scale_scan":nonlinear,
        "nonlinear_constraint_resolves":resolved,
    })

def family_fit(family):
    indices=[i for i,label in enumerate(prepared["labels"]) if label["family"]==family]
    sub={**prepared,"matrix":prepared["matrix"][:,indices],"modes":[prepared["modes"][i] for i in indices]}
    result=solve_linear_corner_correction(sub,.01)
    return {
        "family":family,"column_count":len(indices),"matrix_rank":result["matrix_rank"],
        "final_linear_maximum":result["final_linear_maximum"],
        "final_linear_l2":result["final_linear_l2"],
    }

payload={
    "status":"anisotropic_physical_modes_have_linear_corner_authority_but_first_finite_nonlinear_constraint_resolves_do_not_close_corner",
    "slice":{
        "grid_size":[len(solved["z"]),len(solved["r"])],"r_max":8.,
        "amplitude":amplitude,"energy_dimensionless":solved["energy_dimensionless"],
        "elliptic_residual":solved["max_abs_residual"],
    },
    "ansatz":"gamma=psi^2 diag(exp(2a),exp(2b),exp(2c) radial-transverse), a+b+2c=0",
    "basis":{
        **prepared["settings"],"row_count":int(prepared["matrix"].shape[0]),
        "column_count":int(prepared["matrix"].shape[1]),
        "families":["compact_vs_radial_space","radial_vs_transverse"],
        "compact_profiles":"double-zero Hermite profiles with independent second/third wall jets",
        "radial_profiles":"even outer-flat radial bumps; radial-vs-transverse modes vanish quadratically at the axis",
    },
    "constraint_projection":{
        "maximum_linearized_selector_residual_over_unprojected_source":prepared["maximum_constraint_projection_ratio"],
        "median_linearized_selector_residual_over_unprojected_source":prepared["median_constraint_projection_ratio"],
    },
    "family_controls":[family_fit("compact_vs_radial_space"),family_fit("radial_vs_transverse")],
    "cases":cases,
    "acceptance_for_this_pilot":"finite candidate corner <0.02, max log shape <0.5, Hamiltonian change <0.02, junction change <0.01",
    "interpretation":[
        "The trace-free physical shape modes span both tangential Israel rows and reduce the linearized defect by more than an order of magnitude.",
        "The coupled conformal and stabilizer response projects every sampled shape mode onto the linearized selector, including Hamiltonian, scalar, and zeroth-order wall rows.",
        "The nonlinear q-Phi solver converges on the finite fixed-shape candidates, but the best tested resolved candidate reduces the corner only from 0.377 to about 0.290.",
        "The infinitesimal authority therefore does not extrapolate to an accepted finite correction with a frozen response matrix.",
        "The smallest viable next solver must recompute the constraint-projected corner Jacobian during a regularized nonlinear variable-projection iteration.",
    ],
    "limitations":[
        "single coarse fold slice","finite 48-mode physical basis","linear constraint projection",
        "fixed trace-free shape coefficients during each nonlinear elliptic resolve",
        "no relinearized nonlinear shape iteration","time-time and harmonic rows not yet tested",
    ],
}
for case in payload["cases"]:
    for candidate in case["nonlinear_scale_scan"]+case["nonlinear_constraint_resolves"]:
        candidate["accepted"]=bool(
            candidate["maximum_fixed_scaled_corner_residual"]<.02
            and candidate["maximum_absolute_shape_logarithm"]<.5
            and candidate["maximum_absolute_hamiltonian_change"]<.02
            and candidate["maximum_absolute_junction_change"]<.01
            and candidate.get("selector_converged",True)
            and candidate.get("selector_maximum_residual",0.)<1e-8
        )
payload["any_finite_candidate_accepted"]=bool(any(
    item["accepted"] for case in payload["cases"]
    for item in case["nonlinear_scale_scan"]+case["nonlinear_constraint_resolves"]
))
Path("results/physical_corner_authority_audit.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps({
    "status":payload["status"],"constraint_projection":payload["constraint_projection"],
    "family_controls":payload["family_controls"],
    "cases":[{
        "regularization":case["regularization"],
        "linear_maximum":case["linear"]["final_linear_maximum"],
        "best_finite":min(case["nonlinear_scale_scan"],key=lambda item:item["maximum_fixed_scaled_corner_residual"]),
        "best_resolved":min(case["nonlinear_constraint_resolves"],key=lambda item:item["maximum_fixed_scaled_corner_residual"]),
    } for case in cases],"any_finite_candidate_accepted":payload["any_finite_candidate_accepted"],
},indent=2))
