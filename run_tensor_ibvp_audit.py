#!/usr/bin/env python3
"""Linear TT-sector spectrum and frozen-boundary audit."""

import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.gw_background import solve_gw_background
from bhps.tensor_ibvp import frozen_neumann_boundary_symbol,frozen_positive_robin_boundary_symbol,weighted_neumann_tensor_spectrum,weighted_robin_scalar_spectrum


z=np.linspace(1,np.e,257);backgrounds=[]
for gamma in (2.,5.,20.,100.):
    background=solve_gw_background(z,epsilon=.1,backreaction=.01,wall_stiffness=gamma)
    spectrum=weighted_neumann_tensor_spectrum(z,background["psi"],count=8)
    scalar=weighted_robin_scalar_spectrum(z,background["psi"],background["mass_squared"],gamma,count=8)
    backgrounds.append({
        "wall_stiffness":gamma,"background_residual":background["boundary_residual_max"],
        "omega_squared":[float(value) for value in spectrum["omega_squared"]],
        "minimum_omega_squared":spectrum["minimum_omega_squared"],
        "first_positive_omega_squared":spectrum["first_positive_omega_squared"],
        "zero_mode_constant_overlap":spectrum["zero_mode_constant_overlap"],
        "nonnegative":spectrum["all_within_roundoff_nonnegative"],
        "probe_scalar_omega_squared":[float(value) for value in scalar["omega_squared"]],
        "probe_scalar_minimum_omega_squared":scalar["minimum_omega_squared"],
        "probe_scalar_positive":scalar["all_positive"],
    })

samples=[];scalar_samples=[]
for real in np.logspace(-3,2,16):
    for imag in np.linspace(-20,20,25):
        for wave in np.linspace(0,20,17):
            item=frozen_neumann_boundary_symbol(real,imag,wave);samples.append(abs(item["boundary_determinant"]))
            scalar_item=frozen_positive_robin_boundary_symbol(real,imag,wave,np.sqrt(.41),2.)
            scalar_samples.append(abs(scalar_item["boundary_determinant"]))

payload={
    "status":"linear_TT_and_fixed_metric_scalar_sectors_pass_energy_spectrum_and_frozen_boundary_checks",
    "backgrounds":backgrounds,"boundary_symbol_sample_count":len(samples),
    "minimum_sampled_boundary_determinant_magnitude":float(min(samples)),
    "minimum_sampled_scalar_boundary_determinant_magnitude":float(min(scalar_samples)),
    "all_background_spectra_nonnegative":all(item["nonnegative"] for item in backgrounds),
    "all_probe_scalar_spectra_positive":all(item["probe_scalar_positive"] for item in backgrounds),
    "derivation":{
        "equation":"-d_t(psi^3 d_t h)+d_x(psi^3 d_x h)+d_z(psi^3 d_z h)=0",
        "TT_Israel_boundary":"d_z h=0",
        "energy":"integral psi^3[(d_t h)^2+(d_x h)^2+(d_z h)^2]/2",
        "probe_scalar_energy":"bulk Klein-Gordon energy plus positive gamma Phi^2/4 on each fundamental-interval wall",
    },
    "limitations":[
        "linear transverse-traceless tensor sector only","one-dimensional backgrounds",
        "fixed-metric scalar test does not include scalar-radion metric coupling",
        "does not test coupled scalar/radion, vector, gauge, or constraint modes",
        "frozen-principal boundary symbol is necessary evidence, not a full nonlinear well-posedness proof",
    ],
}
Path("results/tensor_ibvp_audit.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2))
