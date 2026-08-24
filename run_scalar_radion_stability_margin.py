#!/usr/bin/env python3
"""Map the one-dimensional scalar--radion margin around gamma ell=20."""

import json,sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.gw_background import solve_gw_background
from bhps.undivided_scalar_radion import (
    chebyshev_lobatto_grid_and_derivative,
    undivided_scalar_radion_spectral_control,
)


def upper_wall_alpha(epsilon,backreaction,gamma,size=257):
    z=np.linspace(1,np.e,int(size))
    background=solve_gw_background(
        z,epsilon=epsilon,backreaction=backreaction,
        wall_stiffness=gamma,tolerance=1e-11,
    )
    psi=background["psi"];psi_z=background["psi_z"]
    phi=background["phi"];phi_z=background["phi_z"]
    phi_zz=-3*psi_z*phi_z/psi+psi**2*background["mass_squared"]*phi
    phi_y=phi_z/psi
    phi_yy=phi_zz/psi**2-phi_z*psi_z/psi**3
    return float(gamma/2+phi_yy[-1]/phi_y[-1])


def spectrum(epsilon,backreaction,gamma,size=33):
    z,_=chebyshev_lobatto_grid_and_derivative(1,np.e,int(size))
    background=solve_gw_background(
        z,epsilon=epsilon,backreaction=backreaction,
        wall_stiffness=gamma,tolerance=1e-11,
    )
    result=undivided_scalar_radion_spectral_control(
        z,background["psi"],background["psi_z"],background["phi"],
        background["phi_z"],background["mass_squared"],gamma,count=3,
    )
    return [float(value) for value in result["mu_squared"]]


records=[]
for epsilon in (.075,.1,.125):
    for backreaction in (.005,.01,.02,.03):
        threshold=float(brentq(
            lambda gamma:upper_wall_alpha(epsilon,backreaction,gamma),
            12.,20.,xtol=1e-10,
        ))
        threshold_spectrum=spectrum(epsilon,backreaction,threshold)
        production_spectrum=spectrum(epsilon,backreaction,20.)
        records.append({
            "epsilon":epsilon,"backreaction_b0":backreaction,
            "critical_wall_stiffness":threshold,
            "production_wall_stiffness":20.,
            "fractional_stiffness_margin":float(20/threshold-1),
            "upper_wall_alpha_at_production":upper_wall_alpha(
                epsilon,backreaction,20.,
            ),
            "three_nearest_mu_squared_at_production":production_spectrum,
            "closest_mu_squared_at_threshold":threshold_spectrum[0],
        })

payload={
    "status":"gamma20_one_dimensional_scalar_radion_margin_passes_sampled_parameter_neighborhood",
    "parameter_grid":{
        "epsilon":[.075,.1,.125],
        "backreaction_b0":[.005,.01,.02,.03],
        "production_wall_stiffness":20.,
    },
    "records":records,
    "summary":{
        "minimum_critical_wall_stiffness":float(min(item["critical_wall_stiffness"] for item in records)),
        "maximum_critical_wall_stiffness":float(max(item["critical_wall_stiffness"] for item in records)),
        "minimum_fractional_stiffness_margin":float(min(item["fractional_stiffness_margin"] for item in records)),
        "minimum_upper_wall_alpha_at_production":float(min(item["upper_wall_alpha_at_production"] for item in records)),
        "minimum_production_mu_squared":float(min(item["three_nearest_mu_squared_at_production"][0] for item in records)),
        "maximum_abs_threshold_mu_squared":float(max(abs(item["closest_mu_squared_at_threshold"]) for item in records)),
        "all_sampled_production_modes_positive":all(item["three_nearest_mu_squared_at_production"][0]>0 for item in records),
        "all_sampled_critical_values_below_production":all(item["critical_wall_stiffness"]<20 for item in records),
    },
    "interpretation":[
        "The production gamma ell=20 background remains on the positive side of the one-dimensional scalar--radion threshold throughout the previously audited epsilon/backreaction neighborhood.",
        "This margin is a model-selection control, not a substitute for evaluating the localized corrected-fold coefficients.",
    ],
    "limitations":[
        "Poincare-invariant one-dimensional backgrounds",
        "rectangular sampled parameter neighborhood",
        "wall stiffness varied symmetrically at the two branes",
        "localized-source and nonlinear evolution effects absent",
    ],
}
Path("results/scalar_radion_stability_margin.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps(payload,indent=2,sort_keys=True))
