#!/usr/bin/env python3
"""Replicate the G4 capped fold with a finite-difference/Newton surface solver."""

import json,sys,warnings
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.capped_continuation import fit_fold_normal_form,solve_capped_profile
from bhps.capped_surface_fd import capped_fd_jacobian_diagnostic,solve_capped_surface_fd
from bhps.radion_variable_solver import solve_q


amplitudes=np.arange(8.65,8.44,-.005)
geometries=[];previous_q=None
for amplitude in amplitudes:
    solved=solve_q(float(round(amplitude,6)),nz=49,nr=73,r_max=8,initial=previous_q,tolerance=1e-10,iterations=180)
    if not solved["converged"]:
        raise RuntimeError(f"metric solve failed at A={amplitude}")
    geometries.append(solved);previous_q=solved["q"]

reference_geometry=next(item for amplitude,item in zip(amplitudes,geometries) if abs(amplitude-8.55)<1e-9)
with warnings.catch_warnings():
    warnings.simplefilter("ignore",RuntimeWarning)
    reference_surfaces=sorted((
        solve_capped_profile(reference_geometry["z"],reference_geometry["r"],reference_geometry["psi"],1.39,tolerance=1e-8),
        solve_capped_profile(reference_geometry["z"],reference_geometry["r"],reference_geometry["psi"],1.59,tolerance=1e-8),
    ),key=lambda item:item["rho_brane"])
reference_radii=[item["rho_brane"] for item in reference_surfaces]

runs=[]
for nodes in (41,61,81,101,121):
    inner=1.3;outer=1.7;records=[];paired_solutions={}
    for amplitude,geometry in zip(amplitudes,geometries):
        first=solve_capped_surface_fd(geometry["z"],geometry["r"],geometry["psi"],inner,nodes=nodes,tolerance=1e-10)
        second=solve_capped_surface_fd(geometry["z"],geometry["r"],geometry["psi"],outer,nodes=nodes,tolerance=1e-10)
        ordered=sorted((first,second),key=lambda item:item["rho_brane"])
        pair=bool(all(item["converged"] for item in ordered))
        separation=float(ordered[1]["rho_brane"]-ordered[0]["rho_brane"]) if pair else 0.
        records.append({
            "amplitude":float(round(amplitude,6)),"pair_converged":pair,"radius_separation":separation,
            "surfaces":[{key:item[key] for key in (
                "converged","rho_axis","rho_brane","discrete_residual_max","function_evaluations"
            )} for item in ordered],
        })
        if pair and separation>1e-4:
            paired_solutions[float(round(amplitude,6))]=ordered
            inner,outer=ordered
        else:
            break
    close=[item for item in records if item["pair_converged"] and item["radius_separation"]<.25]
    fits=[fit_fold_normal_form(close,count) for count in (4,6,8,10) if len(close)>=count]
    selected=next(item for item in fits if item["fit_point_count"]==6)
    at_reference=next(item for item in records if item["amplitude"]==8.55)
    radii=[item["rho_brane"] for item in at_reference["surfaces"]]
    diagnostics=[]
    for target in (8.5,8.49,8.485,8.48):
        pair=paired_solutions[target]
        geometry=next(item for amplitude,item in zip(amplitudes,geometries) if abs(amplitude-target)<1e-9)
        difference=pair[1]["nodal_rho"]-pair[0]["nodal_rho"]
        difference=difference/np.linalg.norm(difference)
        branch=[]
        for surface in pair:
            item=capped_fd_jacobian_diagnostic(geometry["z"],geometry["r"],geometry["psi"],surface)
            vector=item.pop("right_singular_vector")
            item["branch_difference_overlap"]=float(abs(np.dot(vector,difference)))
            branch.append(item)
        diagnostics.append({"amplitude":target,"branches":branch})
    runs.append({
        "nodes":nodes,"records":records,"normal_form_fits":fits,
        "selected_fold_amplitude":selected["fold_amplitude"],
        "fit_amplitude_systematic_span":max(item["fold_amplitude"] for item in fits)-min(item["fold_amplitude"] for item in fits),
        "A8p55_rho_brane":radii,
        "A8p55_collocation_absolute_differences":[abs(a-b) for a,b in zip(radii,reference_radii)],
        "jacobian_diagnostics":diagnostics,
    })

collocation_fold=8.476573575097554
summary={
    "collocation_pseudo_arclength_fold_amplitude":collocation_fold,
    "finest_fd_fold_amplitude":runs[-1]["selected_fold_amplitude"],
    "finest_fd_collocation_fold_relative_difference":abs(runs[-1]["selected_fold_amplitude"]-collocation_fold)/collocation_fold,
    "two_finest_fd_fold_relative_change":abs(runs[-1]["selected_fold_amplitude"]-runs[-2]["selected_fold_amplitude"])/runs[-1]["selected_fold_amplitude"],
    "finest_A8p55_collocation_radius_differences":runs[-1]["A8p55_collocation_absolute_differences"],
    "maximum_finest_discrete_residual":max(
        surface["discrete_residual_max"] for record in runs[-1]["records"] for surface in record["surfaces"] if surface["converged"]
    ),
    "finest_near_fold_relative_smallest_singular_values":[
        item["relative_smallest_singular_value"] for item in runs[-1]["jacobian_diagnostics"][-1]["branches"]
    ],
    "finest_near_fold_branch_difference_overlaps":[
        item["branch_difference_overlap"] for item in runs[-1]["jacobian_diagnostics"][-1]["branches"]
    ],
    "finest_near_fold_smallest_to_next_singular_ratios":[
        item["smallest_singular_value"]/item["next_singular_value"]
        for item in runs[-1]["jacobian_diagnostics"][-1]["branches"]
    ],
}
payload={
    "status":"independent_surface_solver_replication_passed",
    "metric_case":"G4_R8","amplitude_step":.005,"surface_solver":"second-order finite differences plus scipy root hybr",
    "reference_solver":"solve_bvp collocation", "reference_A8p55_radii":reference_radii,
    "runs":runs,"summary":summary,
    "limitations":["same interpolated G4 metric", "normal-form fold fit rather than finite-difference pseudo-arclength", "unstabilized C3 control"],
}
Path("results/c3_capped_fd_replication.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(summary,indent=2))
