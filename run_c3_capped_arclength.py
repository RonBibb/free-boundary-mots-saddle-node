#!/usr/bin/env python3
"""Cross and audit the G4 C3 capped-surface fold by pseudo-arclength."""

import json,sys,warnings
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from bhps.capped_arclength import MetricFamily,continue_capped_arclength
from bhps.capped_continuation import solve_capped_profile
from bhps.radion_variable_solver import solve_q


configuration={
    "nz":49,"nr":73,"r_max":8.,
    "metric_amplitude_min":8.44,"metric_amplitude_max":8.66,
    "metric_amplitude_steps":[.005,.0025],
    "first_amplitude":8.55,"second_amplitude":8.54,
    "arclength_step":.008,"maximum_steps":32,
}


def exact_geometry(amplitude):
    return solve_q(
        amplitude,nz=configuration["nz"],nr=configuration["nr"],r_max=configuration["r_max"],
        tolerance=1e-10,iterations=180,
    )


def public(item):
    result={key:item[key] for key in (
        "converged","amplitude","rho_axis","rho_brane","rho_min","rho_max",
        "boundary_slope_error",
    ) if key in item}
    if "arclength_residual" in item:
        result.update(arclength_residual=item["arclength_residual"],mesh_nodes=item["mesh_nodes"])
    return result


def fold_fit(points,fit_points):
    amplitude=np.array([item["amplitude"] for item in points])
    radius=np.array([item["rho_brane"] for item in points]);center=int(np.argmin(amplitude))
    half=fit_points//2;lo=max(0,center-half);hi=min(len(points),lo+fit_points);lo=hi-fit_points
    coefficients=np.polyfit(radius[lo:hi],amplitude[lo:hi],2)
    fold_radius=float(-coefficients[1]/(2*coefficients[0]))
    fold_amplitude=float(np.polyval(coefficients,fold_radius))
    return {
        "fit_point_count":fit_points,"fold_amplitude":fold_amplitude,"fold_rho_brane":fold_radius,
        "fit_amplitude_max_residual":float(np.max(np.abs(np.polyval(coefficients,radius[lo:hi])-amplitude[lo:hi]))),
    }


def run_family(metric_step):
    amplitudes=np.arange(
        configuration["metric_amplitude_min"],
        configuration["metric_amplitude_max"]+.5*metric_step,metric_step,
    )
    geometries=[];previous_q=None
    for amplitude in amplitudes:
        solved=solve_q(
            float(amplitude),nz=configuration["nz"],nr=configuration["nr"],
            r_max=configuration["r_max"],initial=previous_q,tolerance=1e-10,iterations=180,
        )
        if not solved["converged"]:
            raise RuntimeError(f"metric solve failed at A={amplitude}")
        geometries.append(solved);previous_q=solved["q"]
    z=geometries[0]["z"];r=geometries[0]["r"]
    family=MetricFamily(amplitudes,z,r,np.array([item["psi"] for item in geometries]))

    starting=[]
    for amplitude,guess in ((configuration["first_amplitude"],1.39),(configuration["second_amplitude"],1.40)):
        geometry=exact_geometry(amplitude)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            profile=solve_capped_profile(z,r,geometry["psi"],guess,tolerance=1e-8)
        if not profile["converged"]:
            raise RuntimeError(f"starting surface failed at A={amplitude}")
        profile["amplitude"]=amplitude;starting.append(profile)

    points=continue_capped_arclength(
        family,*starting,step=configuration["arclength_step"],count=configuration["maximum_steps"],
        tolerance=2e-6,nodes=180,max_nodes=10000,
    )
    failure=next((i for i,item in enumerate(points) if not item.get("converged",False)),len(points))
    converged=points[:failure];turn_index=int(np.argmin([item["amplitude"] for item in converged]))
    fits=[fold_fit(converged,count) for count in (3,5,7)]
    selected=next(item for item in fits if item["fit_point_count"]==5)
    fold_geometry=exact_geometry(selected["fold_amplitude"])

    post_turn=[(i,item) for i,item in enumerate(converged) if i>turn_index and item["amplitude"]>=8.53]
    validation=None
    if post_turn:
        validation_index,continued=min(post_turn,key=lambda pair:abs(pair[1]["amplitude"]-8.54))
        geometry=exact_geometry(continued["amplitude"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            corrected=solve_capped_profile(z,r,geometry["psi"],continued,tolerance=1e-8)
            direct_outer=solve_capped_profile(z,r,geometry["psi"],1.58,tolerance=1e-8)
        validation={
            "point_index":validation_index,"amplitude":continued["amplitude"],
            "interpolated_family_rho_brane":continued["rho_brane"],
            "exact_metric_corrected_rho_brane":corrected["rho_brane"],
            "direct_outer_rho_brane":direct_outer["rho_brane"],
            "interpolation_correction":abs(corrected["rho_brane"]-continued["rho_brane"]),
            "direct_outer_difference":abs(corrected["rho_brane"]-direct_outer["rho_brane"]),
            "corrected_surface_residual":corrected["surface_residual_max"],
            "corrected_converged":corrected["converged"],"direct_outer_converged":direct_outer["converged"],
        }
    amplitude_values=np.array([item["amplitude"] for item in converged])
    crossed=bool(
        len(converged)>=4 and 1<turn_index<len(converged)-2
        and np.all(np.diff(amplitude_values[:turn_index+1])<0)
        and np.all(np.diff(amplitude_values[turn_index:])>0)
        and validation is not None and validation["corrected_converged"] and validation["direct_outer_converged"]
        and validation["direct_outer_difference"]<1e-4
    )
    return {
        "metric_amplitude_step":metric_step,"crossed":crossed,
        "metric_family_max_constraint_residual":max(item["max_abs_residual"] for item in geometries),
        "turn_index":turn_index,"turn_amplitude_sample":converged[turn_index]["amplitude"],
        "fold_fits":fits,"selected_fold_amplitude":selected["fold_amplitude"],
        "selected_fold_energy_dimensionless":fold_geometry["energy_dimensionless"],
        "fit_amplitude_systematic_span":max(item["fold_amplitude"] for item in fits)-min(item["fold_amplitude"] for item in fits),
        "points":[public(item) for item in points],"validation":validation,
    }


runs=[run_family(step) for step in configuration["metric_amplitude_steps"]]
coarse,fine=runs
summary={
    "fold_amplitude_spacing_relative_change":abs(fine["selected_fold_amplitude"]-coarse["selected_fold_amplitude"])/abs(fine["selected_fold_amplitude"]),
    "fold_energy_spacing_relative_change":abs(fine["selected_fold_energy_dimensionless"]-coarse["selected_fold_energy_dimensionless"])/abs(fine["selected_fold_energy_dimensionless"]),
    "fine_fold_amplitude":fine["selected_fold_amplitude"],
    "fine_fold_energy_dimensionless":fine["selected_fold_energy_dimensionless"],
    "maximum_arclength_residual":max(item.get("arclength_residual",0.) for run in runs for item in run["points"]),
}
payload={
    "status":"pseudo_arclength_fold_crossed" if all(item["crossed"] for item in runs) else "pseudo_arclength_incomplete",
    "configuration":configuration,"runs":runs,"summary":summary,
    "limitations":[
        "G4 metric and one radial domain",
        "quadratic local fit used to locate the turn between arclength samples",
        "unstabilized C3 control",
    ],
}
Path("results/c3_capped_arclength.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":payload["status"],"summary":summary,"fine_validation":fine["validation"]},indent=2))
