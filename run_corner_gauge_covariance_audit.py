#!/usr/bin/env python3
"""Gauge-covariance audit of the nonlinear Israel second-corner defect."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.adm_corner import add_metric_accelerations,shift_acceleration_correction,time_symmetric_adm_metric_acceleration
from bhps.finite_wall_solver import solve_finite_wall_slice
from bhps.gauge_corner import compare_mixed_residual_fields,compare_tangential_residual_fields,corner_fields,maximum_tangential_residual
from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.scalar_pulse import scalar_pulse


def manufactured_case(size):
    z=np.linspace(1,np.e,size);r=np.linspace(0,8,size+16)
    zz,rr=np.meshgrid(z,r,indexing="ij");psi=1/zz;phi=np.zeros_like(psi)
    compact=(zz-1)/(np.e-1)
    chi=.4*np.cos(np.pi*compact)*np.exp(-(rr/2)**2)
    chi_z=-.4*np.pi/(np.e-1)*np.sin(np.pi*compact)*np.exp(-(rr/2)**2)
    chi_r=-rr*chi/2
    background={
        "wall_stiffness":0.,"v0":0.,"v1":0.,"beta_a":1.,"beta_b":1.,
        "wall_potential_a":0.,"wall_potential_b":0.,
    }
    base=time_symmetric_adm_metric_acceleration(
        z,r,psi,phi,chi_r,chi_z,0.,m_chi_squared=.3,chi=chi,
        stencil_width=9,lapse=psi,
    )
    reference=corner_fields(base,psi,phi,background,np.zeros_like(psi),7)
    factor=np.exp(.25*np.cos(np.pi*rr/16))
    lapse_changed=time_symmetric_adm_metric_acceleration(
        z,r,psi,phi,chi_r,chi_z,0.,m_chi_squared=.3,chi=chi,
        stencil_width=9,lapse=factor*psi,
    )
    lapse_fields=corner_fields(lapse_changed,psi,phi,background,np.zeros_like(psi),7)
    lapse_comparison=compare_tangential_residual_fields(
        reference,lapse_fields,[factor[0,:-7]**2,factor[-1,:-7]**2],
    )
    shift_r=.1*np.cos(2*np.pi*compact)*np.sin(np.pi*rr/8)
    correction=shift_acceleration_correction(
        z,r,psi,np.zeros_like(psi),shift_r,stencil_width=9,
    )
    shift_fields=corner_fields(
        add_metric_accelerations(base,correction),psi,phi,background,
        np.zeros_like(psi),7,
    )
    shift_comparison=compare_tangential_residual_fields(reference,shift_fields)
    return {
        "grid_size":[len(z),len(r)],
        "reference_maximum_normalized_residual":maximum_tangential_residual(reference),
        "variable_lapse":lapse_comparison,
        "tangential_shift":shift_comparison,
    }


def fold_case(size,amplitude):
    radial_size=int(1.5*size)
    solved=solve_finite_wall_slice(
        amplitude,nz=size,nr=radial_size,r_max=8.,wall_stiffness=20.,
        epsilon=.1,backreaction=.01,tolerance=1e-10,iterations=180,
    )
    z,r=solved["z"],solved["r"]
    psi,phi=solved["psi"],solved["phi"]
    zz,rr=np.meshgrid(z,r,indexing="ij")
    chi,chi_r,chi_z=scalar_pulse(z,r,amplitude)
    base=time_symmetric_adm_metric_acceleration(
        z,r,psi,phi,chi_r,chi_z,solved["background"]["mass_squared"],
        chi=chi,stencil_width=7,lapse=psi,
    )
    zero=np.zeros_like(psi)
    reference=corner_fields(base,psi,phi,solved["background"],zero,7)
    constant_lapse=[]
    for factor in (.5,1.5):
        changed=time_symmetric_adm_metric_acceleration(
            z,r,psi,phi,chi_r,chi_z,solved["background"]["mass_squared"],
            chi=chi,stencil_width=7,lapse=factor*psi,
        )
        fields=corner_fields(changed,psi,phi,solved["background"],zero,7)
        constant_lapse.append({
            "factor":factor,
            **compare_tangential_residual_fields(reference,fields,factor**2),
        })
    phi_r=phi@derivative_matrix(r,1,7).T
    tangential_shifts=[]
    compact=(zz-z[0])/(z[-1]-z[0])
    for n,m in ((0,1),(1,2),(2,3)):
        shift_r=.1*np.cos(n*np.pi*compact)*np.sin(m*np.pi*rr/r[-1])
        correction=shift_acceleration_correction(
            z,r,psi,zero,shift_r,stencil_width=7,
        )
        fields=corner_fields(
            add_metric_accelerations(base,correction),psi,phi,solved["background"],
            shift_r*phi_r,7,
        )
        tangential_shifts.append({
            "compact_mode":n,"radial_mode":m,"maximum_shift_acceleration":.1,
            **compare_tangential_residual_fields(reference,fields),
            **compare_mixed_residual_fields(reference,fields),
        })
    return {
        "grid_size":[len(z),len(r)],"amplitude":float(amplitude),
        "energy_dimensionless":solved["energy_dimensionless"],
        "elliptic_residual":solved["max_abs_residual"],
        "reference_maximum_normalized_residual":maximum_tangential_residual(reference),
        "constant_lapse_rescalings":constant_lapse,
        "tangential_shift_accelerations":tangential_shifts,
    }


manufactured=[manufactured_case(size) for size in (33,65,97)]
fold=[fold_case(size,amplitude) for size,amplitude in (
    (33,8.415541903059392),(49,8.501156129278852),(65,8.53235655905242),
)]
payload={
    "status":"gauge_only_cannot_remove_nonzero_geometric_israel_second_corner",
    "geometric_identity":"For C_ab=0 and L_T C_ab=0 at the corner, (L_(fT+V))^2 C_ab=f^2 (L_T)^2 C_ab when V is tangent to the wall.",
    "manufactured_covariance_control":manufactured,
    "production_fold_gauge_response":fold,
    "summary":{
        "manufactured_finest_variable_lapse_covariance_defect":manufactured[-1]["variable_lapse"]["maximum_fixed_scaled_covariance_defect"],
        "manufactured_finest_shift_response":manufactured[-1]["tangential_shift"]["maximum_fixed_scaled_covariance_defect"],
        "fold_finest_maximum_shift_response":max(
            item["maximum_fixed_scaled_covariance_defect"]
            for item in fold[-1]["tangential_shift_accelerations"]
        ),
        "fold_finest_reference_residual":fold[-1]["reference_maximum_normalized_residual"],
        "constant_lapse_scaling_maximum_defect":max(
            item["maximum_fixed_scaled_covariance_defect"]
            for case in fold for item in case["constant_lapse_rescalings"]
        ),
    },
    "decision":[
        "A regular positive lapse only rescales the geometric second-corner tensor and cannot make a nonzero tensor vanish.",
        "A wall-preserving tangential shift is a null direction for the Israel defect up to numerical differentiation error.",
        "The apparent reduction in the bounded lapse-only pilot is local time slowing, not a compatibility repair.",
        "The next admissible repair must change the physical initial data while re-solving the constraints and zeroth-order junction conditions.",
    ],
    "limitations":[
        "numerical production check uses three primary-solver fold grids",
        "shift modes are a finite sample; the geometric tensor identity supplies the general wall-tangent statement",
        "normal shifts are excluded because they move the fixed wall",
        "time-time and generalized-harmonic rows remain to be tested after physical data are corrected",
    ],
}
Path("results/corner_gauge_covariance_audit.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps({"status":payload["status"],"summary":payload["summary"]},indent=2))
