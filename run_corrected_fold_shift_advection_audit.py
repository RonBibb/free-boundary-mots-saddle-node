#!/usr/bin/env python3
"""Audit linear shift advection of the corrected-fold background GH source."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.gh_source_driver import AxisymmetricDrivenGHWaveIBVP,regular_so3_background_source_shift_advection
from run_corrected_fold_gh_driver_runtime import (
    CONSTRAINT_DAMPING,DRIVER_ETA,DRIVER_FREQUENCY,DRIVER_MU,
    driver_spatial_fields,interpolate_background_source_shift_data,
    interpolate_source_coefficients,sample_background_source_shift_data,
    sample_source_coefficients,state_norm,
)
from run_corrected_fold_regular_so3_runtime import (
    OMEGA,build_geometry,manufactured_problem,sample_coefficients,
)


GRID_SIZES=((17,25),(33,49),(49,73))
WAVE_SCALE=.02


def shifted_problem(geometry,wave_coefficients,source_coefficients,shift_support,nz,nr):
    wave,principal,base_values,base_source,base_left,base_right=manufactured_problem(
        geometry,wave_coefficients,nz,nr,
    )
    source_zero,source_first=interpolate_source_coefficients(
        source_coefficients,wave.z,wave.r,
    )
    shift_data=interpolate_background_source_shift_data(shift_support,wave.z,wave.r)
    driver=AxisymmetricDrivenGHWaveIBVP(
        wave,source_zero,source_first,DRIVER_MU,DRIVER_ETA,
        radial_first_is_scaled=True,background_source_shift_data=shift_data,
    )
    values=WAVE_SCALE*base_values;spatial=driver_spatial_fields(wave.z,wave.r)
    advection_amplitude=regular_so3_background_source_shift_advection(
        values,wave.r,shift_data["source"],shift_data["z_first"],
        shift_data["radial_first_scaled"],shift_data["inverse_compact_metric"],
        shift_data["inverse_radial_metric"],
    )
    denominator=DRIVER_ETA**2+OMEGA**2
    theta_cos=-DRIVER_ETA**2/denominator*advection_amplitude
    theta_sin=-DRIVER_ETA*OMEGA/denominator*advection_amplitude

    def exact_driver(time):
        source_cosine=np.cos(DRIVER_FREQUENCY*time)
        source_sine=np.sin(DRIVER_FREQUENCY*time)
        wave_cosine=np.cos(OMEGA*time);wave_sine=np.sin(OMEGA*time)
        source=source_cosine*spatial["source"]
        source_t=-DRIVER_FREQUENCY*source_sine*spatial["source"]
        source_z=source_cosine*spatial["source_z"]
        source_r=source_cosine*spatial["source_r"]
        advection=wave_cosine*advection_amplitude
        memory=wave_cosine*theta_cos+wave_sine*theta_sin
        target=source+(source_t-advection-memory)/DRIVER_MU
        memory_t=-OMEGA*wave_sine*theta_cos+OMEGA*wave_cosine*theta_sin
        return source,source_t,source_z,source_r,memory,memory_t,target,advection

    def source_acceleration(time):
        source,source_t,source_z,source_r,*_=exact_driver(time)
        radial=np.zeros((nz,nr,9));rr=np.broadcast_to(wave.r[None,:],(nz,nr))
        radial[:,1:]=np.einsum(
            "ijab,ijb->ija",source_first[2,:,1:]/rr[:,1:,None,None],source_r[:,1:],
        )
        return (
            np.einsum("ijab,ijb->ija",source_zero,source)
            +np.einsum("ijab,ijb->ija",source_first[0],source_t)
            +np.einsum("ijab,ijb->ija",source_first[1],source_z)+radial
        )

    def volume_source(time,zq,rq):return WAVE_SCALE*base_source(time,zq,rq)-source_acceleration(time)
    def target(time,zq,rq):return exact_driver(time)[6]
    left=lambda time,rq:WAVE_SCALE*base_left(time,rq)
    right=lambda time,rq:WAVE_SCALE*base_right(time,rq)
    return driver,principal,values,exact_driver,target,volume_source,left,right


def exact_driver_residual(exact_driver):
    maximum=0.
    for time in (0.,.017,.061,.1):
        source,source_t,_,_,memory,memory_t,target,advection=exact_driver(time)
        h_residual=source_t-(advection-DRIVER_MU*(source-target)+memory)
        theta_residual=memory_t+DRIVER_ETA*(memory+advection)
        maximum=max(maximum,float(np.max(np.abs(h_residual))),float(np.max(np.abs(theta_residual))))
    return maximum


def grid_audit(geometry,wave_coefficients,source_coefficients,shift_support,final_time=.012):
    records=[]
    for nz,nr in GRID_SIZES:
        print(f"shift-advection grid {nz} x {nr}",flush=True)
        driver,principal,values,exact_driver,target,volume,left,right=shifted_problem(
            geometry,wave_coefficients,source_coefficients,shift_support,nz,nr,
        )
        source0,_,_,_,memory0,*_=exact_driver(0.)
        speed=max(np.max(principal["z_speed"]),np.max(principal["r_speed"]))
        spacing=min(np.min(np.diff(driver.wave.z)),np.min(np.diff(driver.wave.r)))
        result=driver.integrate(
            values,np.zeros_like(values),source0,memory0,target,final_time,
            .025*spacing/speed,volume,left,right,
        )
        exact_q=np.cos(OMEGA*final_time)*values
        exact_v=-OMEGA*np.sin(OMEGA*final_time)*values
        exact_h,_,_,_,exact_theta,*_=exact_driver(final_time)
        error=np.hypot(
            state_norm(driver.wave,result["position"]-exact_q,result["source"]-exact_h),
            state_norm(driver.wave,result["velocity"]-exact_v,result["memory"]-exact_theta),
        )
        scale=np.hypot(
            state_norm(driver.wave,exact_q,exact_h),state_norm(driver.wave,exact_v,exact_theta),
        )
        records.append({
            "grid_size":[nz,nr],"maximum_grid_spacing":float(max(
                np.max(np.diff(driver.wave.z)),np.max(np.diff(driver.wave.r)),
            )),"combined_relative_error":float(error/scale),
            "time_step":result["time_step"],"steps":result["steps"],
            "exact_driver_residual":exact_driver_residual(exact_driver),
            "advection_amplitude_maximum":float(np.max(np.abs(exact_driver(0.)[-1]))),
        })
    errors=np.array([item["combined_relative_error"] for item in records])
    spacing=np.array([item["maximum_grid_spacing"] for item in records])
    return {
        "records":records,
        "convergence_rates":[float(value) for value in np.log(errors[:-1]/errors[1:])/np.log(spacing[:-1]/spacing[1:])],
    }


def timestep_audit(geometry,wave_coefficients,source_coefficients,shift_support,final_time=.08):
    driver,_,values,exact_driver,target,volume,left,right=shifted_problem(
        geometry,wave_coefficients,source_coefficients,shift_support,13,17,
    )
    source0,_,_,_,memory0,*_=exact_driver(0.);solutions=[];records=[]
    for steps in (8,16,32,64):
        result=driver.integrate(
            values,np.zeros_like(values),source0,memory0,target,final_time,
            final_time/steps,volume,left,right,
        )
        solutions.append(result);records.append({"steps":result["steps"],"time_step":result["time_step"]})
    differences=[]
    for coarse,fine in zip(solutions[:-1],solutions[1:]):
        differences.append(np.hypot(
            state_norm(driver.wave,coarse["position"]-fine["position"],coarse["source"]-fine["source"]),
            state_norm(driver.wave,coarse["velocity"]-fine["velocity"],coarse["memory"]-fine["memory"]),
        ))
    differences=np.asarray(differences)
    return {
        "records":records,"successive_solution_differences":[float(value) for value in differences],
        "convergence_rates":[float(value) for value in np.log(differences[:-1]/differences[1:])/np.log(2.)],
    }


geometry=build_geometry()
print("sampling damped wave and source coefficients",flush=True)
wave_coefficients=sample_coefficients(geometry,constraint_damping=CONSTRAINT_DAMPING)
source_coefficients=sample_source_coefficients(geometry,CONSTRAINT_DAMPING)
shift_support=sample_background_source_shift_data(geometry)
grid=grid_audit(geometry,wave_coefficients,source_coefficients,shift_support)
time=timestep_audit(geometry,wave_coefficients,source_coefficients,shift_support)
acceptance={
    "shift_advection_is_nonzero":min(item["advection_amplitude_maximum"] for item in grid["records"])>1e-5,
    "exact_driver_residual_below_1e_11":max(item["exact_driver_residual"] for item in grid["records"])<1e-11,
    "all_errors_finite":all(np.isfinite(item["combined_relative_error"]) for item in grid["records"]),
    "fine_spatial_rate_above_1p6":grid["convergence_rates"][-1]>1.6,
    "both_timestep_rates_above_3p5":min(time["convergence_rates"])>3.5,
    "background_source_axis_variants_below_5_percent":max(
        shift_support["axis_extrapolation_relative_differences"].values()
    )<.05,
}
payload={
    "status":"pass" if all(acceptance.values()) else "review",
    "scope":"corrected G6-fold manufactured GH driver with linear metric-shift advection of the nonuniform background source",
    "wave_scale":WAVE_SCALE,"constraint_damping_rate":CONSTRAINT_DAMPING,
    "driver_rates":{"mu":DRIVER_MU,"eta":DRIVER_ETA},
    "background_source_axis_extrapolation_relative_differences":shift_support["axis_extrapolation_relative_differences"],
    "radial_first_scaled_axis_diagnostics":shift_support["radial_first_scaled_axis_diagnostics"],
    "grid_convergence":grid,"timestep_convergence":time,"acceptance":acceptance,
    "limitations":[
        "manufactured linear metric/source fields rather than a freely selected gauge target",
        "zero background shift; the new term is the perturbative shift acting on nonuniform background H_a",
        "five-by-seven wave/source support and five-by-nine background-shift support at r<=4",
        "homogeneous driver boundary values",
        "nonlinear advection and nonlinear Einstein evolution remain open",
    ],
}
Path("results/corrected_fold_shift_advection_audit.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps({"status":payload["status"],"grid":grid,"time":time,"acceptance":acceptance},indent=2))
