#!/usr/bin/env python3
"""Pseudo-arclength crossing of the corrected anisotropic capped fold."""

import json,os,sys,warnings
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_capped_arclength import AnisotropicMetricFamily,continue_capped_arclength
from bhps.anisotropic_capped_surface import solve_anisotropic_capped_profile
from bhps.anisotropic_geometry import anisotropic_scalar_gradient_energy
from bhps.anisotropic_initial_data import solve_anisotropic_initial_data
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice
from bhps.physical_corner_corrector import combine_shape_modes,physical_corner_state,tracefree_shape_basis
from bhps.scalar_pulse import scalar_pulse


config={
    "nz":int(os.environ.get("BHPS_ARC_NZ","49")),
    "nr":int(os.environ.get("BHPS_ARC_NR","73")),
    "r_max":8.,"wall_stiffness":20.,"epsilon":.1,
    "backreaction":.01,"metric_amplitude_min":7.88,"metric_amplitude_max":8.08,
    "metric_amplitude_steps":[.01,.005],"first_amplitude":7.98,
    "second_amplitude":7.96,"arclength_step":.01,"maximum_steps":38,
}
archive=np.load("results/corrected_family_knot_A8_state.npz")
coefficients=archive["coefficients"]


def geometry(amplitude,initial=None):
    reference=solve_finite_wall_high_order_slice(
        amplitude,nz=config["nz"],nr=config["nr"],r_max=config["r_max"],
        wall_stiffness=config["wall_stiffness"],epsilon=config["epsilon"],
        backreaction=config["backreaction"],tolerance=1e-10,iterations=240,
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
        initial_q=None if initial is None else initial["q"],
        initial_phi=None if initial is None else initial["phi"],
        stencil_width=7,tolerance=1e-9,iterations=30,
    )
    corner=physical_corner_state(
        reference["z"],reference["r"],selected["q"],selected["phi"],a,b,c,
        reference["background"],chi_r,chi_z,chi,None,7,7,True,
    )
    return {
        "z":reference["z"],"r":reference["r"],"q":selected["q"],
        "phi":selected["phi"],"psi":corner["psi"],"a":a,"b":b,"c":c,
        "selector_maximum":selected["maximum_residual"],
        "corner_maximum":corner["maximum_intrinsic_residual"],
        "energy_dimensionless":anisotropic_scalar_gradient_energy(
            reference["z"],reference["r"],corner["psi"],a,b,c,chi_r,chi_z,
        ),
    }


def fit_turn(points,count):
    amplitude=np.array([item["amplitude"] for item in points])
    radius=np.array([item["rho_brane"] for item in points])
    center=int(np.argmin(amplitude));half=count//2;lo=center-half;hi=lo+count
    coefficients=np.polyfit(radius[lo:hi],amplitude[lo:hi],2)
    fold_radius=float(-coefficients[1]/(2*coefficients[0]))
    return {
        "fit_point_count":count,
        "fold_amplitude":float(np.polyval(coefficients,fold_radius)),
        "fold_rho_brane":fold_radius,
        "fit_amplitude_max_residual":float(np.max(np.abs(
            np.polyval(coefficients,radius[lo:hi])-amplitude[lo:hi]
        ))),
    }


def public(item):
    return {key:item[key] for key in (
        "converged","amplitude","rho_axis","rho_brane","rho_min","rho_max",
        "boundary_slope_error","arclength_residual","mesh_nodes",
    ) if key in item}


def run(spacing):
    amplitudes=np.arange(
        config["metric_amplitude_min"],config["metric_amplitude_max"]+.5*spacing,spacing,
    )
    geometries=[];previous=None
    for amplitude in amplitudes:
        item=geometry(float(amplitude),previous)
        geometries.append(item);previous=item
    z=geometries[0]["z"];r=geometries[0]["r"]
    family=AnisotropicMetricFamily(
        amplitudes,z,r,np.array([item["psi"] for item in geometries]),
        np.array([item["a"] for item in geometries]),
        np.array([item["b"] for item in geometries]),
        np.array([item["c"] for item in geometries]),
    )
    starting=[]
    for amplitude,guess in (
        (config["first_amplitude"],1.375),(config["second_amplitude"],1.393),
    ):
        exact=geometry(amplitude)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            profile=solve_anisotropic_capped_profile(
                z,r,exact["psi"],exact["a"],exact["b"],exact["c"],guess,
                tolerance=1e-8,
            )
        if not profile["converged"]:raise RuntimeError(f"surface failed at A={amplitude}")
        profile["amplitude"]=amplitude;starting.append(profile)
    points=continue_capped_arclength(
        family,*starting,step=config["arclength_step"],count=config["maximum_steps"],
        tolerance=2e-6,nodes=180,max_nodes=10000,
    )
    failure=next((i for i,item in enumerate(points) if not item.get("converged",False)),len(points))
    converged=points[:failure];turn=int(np.argmin([item["amplitude"] for item in converged]))
    fits=[fit_turn(converged,count) for count in (3,5,7)]
    selected=next(item for item in fits if item["fit_point_count"]==5)
    fold_geometry=geometry(selected["fold_amplitude"])
    post=[(i,item) for i,item in enumerate(converged) if i>turn and item["amplitude"]>=7.95]
    validation=None
    if post:
        index,continued=min(post,key=lambda pair:abs(pair[1]["amplitude"]-7.96))
        exact=geometry(continued["amplitude"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            corrected=solve_anisotropic_capped_profile(
                z,r,exact["psi"],exact["a"],exact["b"],exact["c"],continued,
                tolerance=1e-8,
            )
            direct=solve_anisotropic_capped_profile(
                z,r,exact["psi"],exact["a"],exact["b"],exact["c"],1.56,
                tolerance=1e-8,
            )
        validation={
            "point_index":index,"amplitude":continued["amplitude"],
            "interpolated_family_rho_brane":continued["rho_brane"],
            "exact_metric_corrected_rho_brane":corrected["rho_brane"],
            "direct_outer_rho_brane":direct["rho_brane"],
            "interpolation_correction":abs(corrected["rho_brane"]-continued["rho_brane"]),
            "direct_outer_difference":abs(corrected["rho_brane"]-direct["rho_brane"]),
            "corrected_surface_residual":corrected["surface_residual_max"],
            "corrected_converged":corrected["converged"],
            "direct_outer_converged":direct["converged"],
        }
    values=np.array([item["amplitude"] for item in converged])
    crossed=bool(
        1<turn<len(converged)-2 and np.all(np.diff(values[:turn+1])<0)
        and np.all(np.diff(values[turn:])>0) and validation
        and validation["corrected_converged"] and validation["direct_outer_converged"]
        and validation["direct_outer_difference"]<1e-4
    )
    return {
        "metric_amplitude_step":spacing,"crossed":crossed,"turn_index":turn,
        "turn_amplitude_sample":converged[turn]["amplitude"],"fold_fits":fits,
        "selected_fold_amplitude":selected["fold_amplitude"],
        "selected_fold_energy_dimensionless":fold_geometry["energy_dimensionless"],
        "metric_family_max_selector_residual":max(item["selector_maximum"] for item in geometries),
        "metric_family_max_corner_residual":max(item["corner_maximum"] for item in geometries),
        "points":[public(item) for item in points],"validation":validation,
    }


runs=[run(spacing) for spacing in config["metric_amplitude_steps"]]
coarse,fine=runs
summary={
    "fine_fold_amplitude":fine["selected_fold_amplitude"],
    "fine_fold_energy_dimensionless":fine["selected_fold_energy_dimensionless"],
    "fold_amplitude_spacing_relative_change":abs(
        fine["selected_fold_amplitude"]-coarse["selected_fold_amplitude"]
    )/fine["selected_fold_amplitude"],
    "fold_energy_spacing_relative_change":abs(
        fine["selected_fold_energy_dimensionless"]-coarse["selected_fold_energy_dimensionless"]
    )/fine["selected_fold_energy_dimensionless"],
    "maximum_arclength_residual":max(
        item.get("arclength_residual",0.) for run_item in runs for item in run_item["points"]
    ),
}
payload={
    "status":"corrected_anisotropic_pseudo_arclength_fold_crossed"
    if all(item["crossed"] for item in runs) else "corrected_anisotropic_arclength_incomplete",
    "configuration":config,"shape_state":"results/corrected_family_knot_A8_state.npz",
    "runs":runs,"summary":summary,
    "limitations":[
        "G5 R8 metric family",
        "fixed A=8 shape knot over the local amplitude interval",
        "piecewise-linear metric interpolation",
        "local quadratic turn fit",
    ],
}
output_path=os.environ.get(
    "BHPS_ARC_OUTPUT","results/corrected_anisotropic_arclength.json"
)
Path(output_path).write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps({
    "status":payload["status"],"summary":summary,"fine_validation":fine["validation"],
},indent=2))
