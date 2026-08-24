#!/usr/bin/env python3
"""Independent FD/Newton surface replication of the finite-wall cap fold."""

import json,sys,warnings
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from bhps.capped_continuation import fit_fold_normal_form,solve_capped_profile
from bhps.capped_surface_fd import capped_fd_jacobian_diagnostic,solve_capped_surface_fd
from bhps.finite_wall_solver import solve_finite_wall_slice


amplitudes=np.arange(8.70,8.47,-.005);geometries=[];previous=None
for amplitude in amplitudes:
    item=solve_finite_wall_slice(
        float(round(amplitude,6)),wall_stiffness=20.,nz=49,nr=73,r_max=8,epsilon=.1,backreaction=.01,
        initial=previous,tolerance=1e-10,iterations=180,
    )
    if not item["converged"]:raise RuntimeError(f"metric failed at A={amplitude}")
    geometries.append(item);previous=item

reference_geometry=next(item for amplitude,item in zip(amplitudes,geometries) if abs(amplitude-8.58)<1e-9)
with warnings.catch_warnings():
    warnings.simplefilter("ignore",RuntimeWarning)
    reference=sorted((
        solve_capped_profile(reference_geometry["z"],reference_geometry["r"],reference_geometry["psi"],1.39,tolerance=1e-8),
        solve_capped_profile(reference_geometry["z"],reference_geometry["r"],reference_geometry["psi"],1.59,tolerance=1e-8),
    ),key=lambda item:item["rho_brane"])
reference_radii=[item["rho_brane"] for item in reference]

runs=[]
for nodes in (41,61,81,101,121):
    inner=1.3;outer=1.7;records=[];pairs={}
    for amplitude,geometry in zip(amplitudes,geometries):
        one=solve_capped_surface_fd(geometry["z"],geometry["r"],geometry["psi"],inner,nodes=nodes,tolerance=1e-10)
        two=solve_capped_surface_fd(geometry["z"],geometry["r"],geometry["psi"],outer,nodes=nodes,tolerance=1e-10)
        ordered=sorted((one,two),key=lambda item:item["rho_brane"]);pair=all(item["converged"] for item in ordered)
        separation=float(ordered[1]["rho_brane"]-ordered[0]["rho_brane"]) if pair else 0.
        key=float(round(amplitude,6));records.append({
            "amplitude":key,"pair_converged":pair,"radius_separation":separation,
            "surfaces":[{name:item[name] for name in ("converged","rho_axis","rho_brane","discrete_residual_max","function_evaluations")} for item in ordered],
        })
        if pair and separation>1e-4:pairs[key]=ordered;inner,outer=ordered
        else:break
    close=[item for item in records if item["pair_converged"] and item["radius_separation"]<.25]
    fits=[fit_fold_normal_form(close,count) for count in (4,6,8,10) if len(close)>=count]
    selected=next(item for item in fits if item["fit_point_count"]==6)
    at_reference=next(item for item in records if item["amplitude"]==8.58);radii=[item["rho_brane"] for item in at_reference["surfaces"]]
    diagnostics=[]
    for target in (8.525,8.515,8.51,8.505):
        pair=pairs[target];metric=next(item for amplitude,item in zip(amplitudes,geometries) if abs(amplitude-target)<1e-9)
        difference=pair[1]["nodal_rho"]-pair[0]["nodal_rho"];difference/=np.linalg.norm(difference);branches=[]
        for surface in pair:
            diagnostic=capped_fd_jacobian_diagnostic(metric["z"],metric["r"],metric["psi"],surface)
            vector=diagnostic.pop("right_singular_vector");diagnostic["branch_difference_overlap"]=float(abs(np.dot(vector,difference)))
            branches.append(diagnostic)
        diagnostics.append({"amplitude":target,"branches":branches})
    runs.append({
        "nodes":nodes,"records":records,"normal_form_fits":fits,
        "selected_fold_amplitude":selected["fold_amplitude"],
        "A8p58_rho_brane":radii,"A8p58_collocation_absolute_differences":[abs(a-b) for a,b in zip(radii,reference_radii)],
        "jacobian_diagnostics":diagnostics,
    })

collocation_fold=8.501241145637987;near=runs[-1]["jacobian_diagnostics"][-1]["branches"]
summary={
    "collocation_pseudo_arclength_fold_amplitude":collocation_fold,
    "finest_fd_fold_amplitude":runs[-1]["selected_fold_amplitude"],
    "finest_fd_collocation_fold_relative_difference":abs(runs[-1]["selected_fold_amplitude"]-collocation_fold)/collocation_fold,
    "two_finest_fd_fold_relative_change":abs(runs[-1]["selected_fold_amplitude"]-runs[-2]["selected_fold_amplitude"])/runs[-1]["selected_fold_amplitude"],
    "finest_A8p58_collocation_radius_differences":runs[-1]["A8p58_collocation_absolute_differences"],
    "maximum_finest_discrete_residual":max(surface["discrete_residual_max"] for record in runs[-1]["records"] for surface in record["surfaces"] if surface["converged"]),
    "finest_near_fold_smallest_to_next_singular_ratios":[item["smallest_singular_value"]/item["next_singular_value"] for item in near],
    "finest_near_fold_branch_difference_overlaps":[item["branch_difference_overlap"] for item in near],
}
payload={
    "status":"finite_wall_coupled_independent_surface_replication_passed",
    "metric_case":"G4_R8","wall_stiffness":20.,"epsilon":.1,"backreaction_b0":.01,"amplitude_step":.005,
    "surface_solver":"second-order finite differences plus scipy root hybr","reference_solver":"solve_bvp collocation",
    "runs":runs,"summary":summary,
    "limitations":["same coupled G4 metric solver","fixed-amplitude normal-form FD fit","momentarily stationary scalar selector"],
}
Path("results/finite_wall_surface_fd_replication.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(summary,indent=2))
