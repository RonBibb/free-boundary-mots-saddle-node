#!/usr/bin/env python3
"""Pseudo-arclength crossing of the coupled finite-wall G4 capped fold."""

import json,sys,warnings
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.capped_arclength import MetricFamily,continue_capped_arclength
from bhps.capped_continuation import solve_capped_profile
from bhps.finite_wall_solver import solve_finite_wall_slice


config={
    "nz":49,"nr":73,"r_max":8.,"epsilon":.1,"backreaction_b0":.01,"wall_stiffness":20.,
    "metric_amplitude_min":8.47,"metric_amplitude_max":8.70,
    "metric_amplitude_steps":[.005,.0025],"first_amplitude":8.57,
    "second_amplitude":8.56,"arclength_step":.008,"maximum_steps":34,
}


def geometry(amplitude,initial=None):
    return solve_finite_wall_slice(
        amplitude,nz=config["nz"],nr=config["nr"],r_max=config["r_max"],
        epsilon=config["epsilon"],backreaction=config["backreaction_b0"],
        wall_stiffness=config["wall_stiffness"],initial=initial,tolerance=1e-10,iterations=180,
    )


def fit_turn(points,count):
    amplitude=np.array([item["amplitude"] for item in points]);radius=np.array([item["rho_brane"] for item in points])
    center=int(np.argmin(amplitude));half=count//2;lo=center-half;hi=lo+count
    coefficients=np.polyfit(radius[lo:hi],amplitude[lo:hi],2);fold_radius=float(-coefficients[1]/(2*coefficients[0]))
    return {
        "fit_point_count":count,"fold_amplitude":float(np.polyval(coefficients,fold_radius)),
        "fold_rho_brane":fold_radius,
        "fit_amplitude_max_residual":float(np.max(np.abs(np.polyval(coefficients,radius[lo:hi])-amplitude[lo:hi]))),
    }


def public(item):
    return {key:item[key] for key in (
        "converged","amplitude","rho_axis","rho_brane","rho_min","rho_max",
        "boundary_slope_error","arclength_residual","mesh_nodes",
    ) if key in item}


def run(spacing):
    amplitudes=np.arange(config["metric_amplitude_min"],config["metric_amplitude_max"]+.5*spacing,spacing)
    geometries=[];previous=None
    for amplitude in amplitudes:
        item=geometry(float(amplitude),previous)
        if not item["converged"]:raise RuntimeError(f"metric failed at A={amplitude}")
        geometries.append(item);previous=item
    z=geometries[0]["z"];r=geometries[0]["r"]
    family=MetricFamily(amplitudes,z,r,np.array([item["psi"] for item in geometries]))
    starting=[]
    for amplitude,guess in ((config["first_amplitude"],1.40),(config["second_amplitude"],1.405)):
        item=geometry(amplitude)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            profile=solve_capped_profile(z,r,item["psi"],guess,tolerance=1e-8)
        if not profile["converged"]:raise RuntimeError(f"surface failed at A={amplitude}")
        profile["amplitude"]=amplitude;starting.append(profile)
    points=continue_capped_arclength(
        family,*starting,step=config["arclength_step"],count=config["maximum_steps"],
        tolerance=2e-6,nodes=180,max_nodes=10000,
    )
    failure=next((i for i,item in enumerate(points) if not item.get("converged",False)),len(points));converged=points[:failure]
    turn=int(np.argmin([item["amplitude"] for item in converged]));fits=[fit_turn(converged,n) for n in (3,5,7)]
    selected=next(item for item in fits if item["fit_point_count"]==5);fold_geometry=geometry(selected["fold_amplitude"])
    post=[(i,item) for i,item in enumerate(converged) if i>turn and item["amplitude"]>=8.55]
    validation=None
    if post:
        index,continued=min(post,key=lambda pair:abs(pair[1]["amplitude"]-8.56));exact=geometry(continued["amplitude"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            corrected=solve_capped_profile(z,r,exact["psi"],continued,tolerance=1e-8)
            direct=solve_capped_profile(z,r,exact["psi"],1.58,tolerance=1e-8)
        validation={
            "point_index":index,"amplitude":continued["amplitude"],
            "interpolated_family_rho_brane":continued["rho_brane"],
            "exact_metric_corrected_rho_brane":corrected["rho_brane"],"direct_outer_rho_brane":direct["rho_brane"],
            "interpolation_correction":abs(corrected["rho_brane"]-continued["rho_brane"]),
            "direct_outer_difference":abs(corrected["rho_brane"]-direct["rho_brane"]),
            "corrected_surface_residual":corrected["surface_residual_max"],
            "corrected_converged":corrected["converged"],"direct_outer_converged":direct["converged"],
        }
    values=np.array([item["amplitude"] for item in converged])
    crossed=bool(
        1<turn<len(converged)-2 and np.all(np.diff(values[:turn+1])<0) and np.all(np.diff(values[turn:])>0)
        and validation and validation["corrected_converged"] and validation["direct_outer_converged"]
        and validation["direct_outer_difference"]<1e-4
    )
    return {
        "metric_amplitude_step":spacing,"crossed":crossed,"turn_index":turn,
        "turn_amplitude_sample":converged[turn]["amplitude"],"fold_fits":fits,
        "selected_fold_amplitude":selected["fold_amplitude"],
        "selected_fold_energy_dimensionless":fold_geometry["energy_dimensionless"],
        "metric_family_max_coupled_residual":max(item["max_abs_residual"] for item in geometries),
        "points":[public(item) for item in points],"validation":validation,
    }


runs=[run(spacing) for spacing in config["metric_amplitude_steps"]];coarse,fine=runs
summary={
    "fine_fold_amplitude":fine["selected_fold_amplitude"],
    "fine_fold_energy_dimensionless":fine["selected_fold_energy_dimensionless"],
    "fold_amplitude_spacing_relative_change":abs(fine["selected_fold_amplitude"]-coarse["selected_fold_amplitude"])/fine["selected_fold_amplitude"],
    "fold_energy_spacing_relative_change":abs(fine["selected_fold_energy_dimensionless"]-coarse["selected_fold_energy_dimensionless"])/fine["selected_fold_energy_dimensionless"],
    "maximum_arclength_residual":max(item.get("arclength_residual",0.) for run_item in runs for item in run_item["points"]),
}
payload={
    "status":"finite_wall_coupled_pseudo_arclength_fold_crossed" if all(item["crossed"] for item in runs) else "incomplete",
    "configuration":config,"runs":runs,"summary":summary,
    "limitations":["G4 metric and one domain","local quadratic turn fit","one coupled metric discretization","momentarily stationary scalar selector"],
}
Path("results/finite_wall_arclength.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":payload["status"],"summary":summary,"fine_validation":fine["validation"]},indent=2))
