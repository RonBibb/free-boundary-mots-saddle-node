#!/usr/bin/env python3
"""Manufactured corrected-fold runtime with damping and evolved GH source."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.gh_source_driver import AxisymmetricDrivenGHWaveIBVP,DRIVER_FIELD_ORDER,regular_so3_source_coupling_matrices
from bhps.linearized_gh_einstein_scalar import metric_geometry_from_jets
from run_corrected_fold_regular_so3_runtime import (
    GRID_SIZES,OMEGA,R_MAX,SUPPORT_R,SUPPORT_Z_COUNT,build_geometry,
    even_axis_extension,interpolate_tensor,manufactured_problem,sample_coefficients,
)


DRIVER_FREQUENCY=.9
CONSTRAINT_DAMPING=1.
DRIVER_MU=2.
DRIVER_ETA=1.25
BACKGROUND_SHIFT_SUPPORT_R=np.array((1/12,1/6,.25,1/3,.5,1.,2.,3.,4.))


def sample_source_coefficients(
    geometry,constraint_damping=CONSTRAINT_DAMPING,support_r=SUPPORT_R,
):
    support_r=np.asarray(support_r,dtype=float)
    z_support=np.geomspace(geometry["z"][0],geometry["z"][-1],SUPPORT_Z_COUNT)
    zero=np.empty((len(z_support),len(support_r),9,3))
    first=np.empty((3,len(z_support),len(support_r),9,3))
    for i,z_value in enumerate(z_support):
        print(f"source coupling support z {i+1}/{len(z_support)}",flush=True)
        for j,r_value in enumerate(support_r):
            result=regular_so3_source_coupling_matrices(
                geometry["jet_field"].at(z_value,r_value),r_value,
                mass_squared=geometry["mass_squared"],potential_offset=-6.,
                constraint_damping=constraint_damping,
            )
            zero[i,j]=result["evolution_zero_matrix"]
            first[:,i,j]=result["evolution_first_matrices"]
    zero_ext,zero_axis=even_axis_extension(zero,support_r)
    time_ext,time_axis=even_axis_extension(first[0],support_r)
    z_ext,z_axis=even_axis_extension(first[1],support_r)
    radial_ext,radial_axis=even_axis_extension(first[2]*support_r[None,:,None,None],support_r)
    return {
        "z":z_support,"r":np.r_[0.,support_r],"zero":zero_ext,
        "first":np.stack((time_ext,z_ext,radial_ext)),
        "radial_first_is_scaled":True,
        "constraint_damping_rate":float(constraint_damping),
        "axis_extrapolation_relative_differences":{
            "zero":zero_axis,"time_first":time_axis,"z_first":z_axis,
            "scaled_radial_first":radial_axis,
        },
    }


def driver_spatial_fields(z,r):
    zz,rr=np.meshgrid(z,r,indexing="ij");x=(zz-z[0])/(z[-1]-z[0])
    radial=(1-(rr/R_MAX)**2)**2;base=np.sin(np.pi*x)*radial
    base_z=np.pi/(z[-1]-z[0])*np.cos(np.pi*x)*radial
    base_r=-4*rr/R_MAX**2*(1-(rr/R_MAX)**2)*np.sin(np.pi*x)
    source_vector=np.array((.12,-.08,.05));memory_vector=np.array((.03,.02,-.025))
    return {
        "source":base[:,:,None]*source_vector,
        "source_z":base_z[:,:,None]*source_vector,
        "source_r":base_r[:,:,None]*source_vector,
        "memory":base[:,:,None]*memory_vector,
    }


def interpolate_source_coefficients(coefficients,z,r):
    zero=interpolate_tensor(coefficients["zero"],coefficients["z"],coefficients["r"],z,r)
    first=np.empty((3,len(z),len(r),9,3))
    for direction in range(3):
        first[direction]=interpolate_tensor(
            coefficients["first"][direction],coefficients["z"],coefficients["r"],z,r,
        )
    return zero,first


def sample_background_source_shift_data(
    geometry,support_r=BACKGROUND_SHIFT_SUPPORT_R,
):
    """Sample regular background ``H=Gamma`` fields and spatial derivatives."""
    support_r=np.asarray(support_r,dtype=float)
    z_support=np.geomspace(geometry["z"][0],geometry["z"][-1],SUPPORT_Z_COUNT)
    source=np.empty((len(z_support),len(support_r),3))
    z_first=np.empty_like(source);radial_scaled=np.empty_like(source)
    inverse_compact=np.empty((len(z_support),len(support_r)))
    inverse_radial=np.empty_like(inverse_compact)
    for i,z_value in enumerate(z_support):
        print(f"background source-shift support z {i+1}/{len(z_support)}",flush=True)
        for j,radius in enumerate(support_r):
            background=geometry["jet_field"].at(z_value,radius)
            metric_geometry=metric_geometry_from_jets(
                background["metric"],background["metric_first"],background["metric_second"],
            )
            h=metric_geometry["contracted_christoffel_covector"]
            hf=metric_geometry["contracted_christoffel_covector_first"]
            source[i,j]=(h[0],h[1],h[2]/radius)
            z_first[i,j]=(hf[1,0],hf[1,1],hf[1,2]/radius)
            radial_scaled[i,j]=(radius*hf[2,0],radius*hf[2,1],hf[2,2]-h[2]/radius)
            inverse_compact[i,j]=metric_geometry["inverse_metric"][1,1]
            inverse_radial[i,j]=metric_geometry["inverse_metric"][2,2]
    fields={};axis={}
    for name,values in (
        ("source",source),("z_first",z_first),
        ("inverse_compact_metric",inverse_compact),("inverse_radial_metric",inverse_radial),
    ):
        fields[name],axis[name]=even_axis_extension(values,support_r)
    # r d_r of an even regular coefficient vanishes exactly at the axis.  A
    # relative comparison between two fits to that zero is ill-conditioned,
    # so impose parity and normalize the residual fit by the off-axis field.
    radial_extension,raw_radial_fit_difference=even_axis_extension(
        radial_scaled,support_r,
    )
    radial_axis_fit=radial_extension[:,0].copy()
    radial_extension[:,0]=0.
    fields["radial_first_scaled"]=radial_extension
    axis["radial_first_scaled_parity_fit_relative_to_support"]=float(
        np.linalg.norm(radial_axis_fit)/max(np.linalg.norm(radial_scaled),1e-300)
    )
    return {
        "z":z_support,"r":np.r_[0.,support_r],**fields,
        "axis_extrapolation_relative_differences":axis,
        "radial_first_scaled_axis_diagnostics":{
            "imposed_axis_maximum":float(np.max(np.abs(radial_extension[:,0]))),
            "unconstrained_axis_fit_maximum":float(np.max(np.abs(radial_axis_fit))),
            "two_fit_difference_relative_to_unconstrained_fit":raw_radial_fit_difference,
        },
    }


def interpolate_background_source_shift_data(data,z,r):
    return {
        key:interpolate_tensor(data[key],data["z"],data["r"],z,r)
        for key in (
            "source","z_first","radial_first_scaled",
            "inverse_compact_metric","inverse_radial_metric",
        )
    }


def driven_problem(geometry,wave_coefficients,source_coefficients,nz,nr):
    wave,principal,values,wave_source,left,upper=manufactured_problem(
        geometry,wave_coefficients,nz,nr,
    )
    zero,first=interpolate_source_coefficients(source_coefficients,wave.z,wave.r)
    driver=AxisymmetricDrivenGHWaveIBVP(
        wave,zero,first,DRIVER_MU,DRIVER_ETA,radial_first_is_scaled=True,
    )
    spatial=driver_spatial_fields(wave.z,wave.r)

    def exact_driver(time):
        cosine=np.cos(DRIVER_FREQUENCY*time);sine=np.sin(DRIVER_FREQUENCY*time)
        source=cosine*spatial["source"]
        source_t=-DRIVER_FREQUENCY*sine*spatial["source"]
        source_z=cosine*spatial["source_z"]
        source_r=cosine*spatial["source_r"]
        memory=np.exp(-DRIVER_ETA*time)*spatial["memory"]
        target=source+(source_t-memory)/DRIVER_MU
        return source,source_t,source_z,source_r,memory,target

    def source_acceleration(time):
        source,source_t,source_z,source_r,_,_=exact_driver(time)
        radial=np.zeros((nz,nr,9));rr=np.broadcast_to(wave.r[None,:],(nz,nr))
        radial[:,1:]=np.einsum(
            "ijab,ijb->ija",first[2,:,1:]/rr[:,1:,None,None],source_r[:,1:],
        )
        return (
            np.einsum("ijab,ijb->ija",zero,source)
            +np.einsum("ijab,ijb->ija",first[0],source_t)
            +np.einsum("ijab,ijb->ija",first[1],source_z)+radial
        )

    def volume_source(time,zq,rq):return wave_source(time,zq,rq)-source_acceleration(time)
    def target(time,zq,rq):return exact_driver(time)[-1]
    return driver,principal,values,spatial,exact_driver,target,volume_source,left,upper


def state_norm(wave,wave_values,source_values):
    wave_norm=wave.l2_norm(wave_values)
    flat=np.asarray(source_values).reshape(wave.nodes,3)
    source_norm=float(np.sqrt(max(0.,np.sum((wave.mass@flat)*flat))))
    return float(np.hypot(wave_norm,source_norm))


def grid_audit(geometry,wave_coefficients,source_coefficients,final_time=.012):
    records=[]
    for nz,nr in GRID_SIZES:
        print(f"driven runtime grid {nz} x {nr}",flush=True)
        driver,principal,values,spatial,exact_driver,target,volume,left,upper=driven_problem(
            geometry,wave_coefficients,source_coefficients,nz,nr,
        )
        source0,_,_,_,memory0,_=exact_driver(0.)
        speed=max(np.max(principal["z_speed"]),np.max(principal["r_speed"]))
        spacing=min(np.min(np.diff(driver.wave.z)),np.min(np.diff(driver.wave.r)))
        result=driver.integrate(
            values,np.zeros_like(values),source0,memory0,target,final_time,
            .025*spacing/speed,volume,left,upper,
        )
        exact_q=np.cos(OMEGA*final_time)*values
        exact_v=-OMEGA*np.sin(OMEGA*final_time)*values
        exact_h,_,_,_,exact_theta,_=exact_driver(final_time)
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
        })
    errors=np.array([item["combined_relative_error"] for item in records])
    spacing=np.array([item["maximum_grid_spacing"] for item in records])
    return {
        "records":records,
        "convergence_rates":[float(value) for value in np.log(errors[:-1]/errors[1:])/np.log(spacing[:-1]/spacing[1:])],
    }


def timestep_audit(geometry,wave_coefficients,source_coefficients,final_time=.08,nz=13,nr=17):
    driver,_,values,_,exact_driver,target,volume,left,upper=driven_problem(
        geometry,wave_coefficients,source_coefficients,nz,nr,
    )
    source0,_,_,_,memory0,_=exact_driver(0.);solutions=[];records=[]
    for steps in (8,16,32,64):
        result=driver.integrate(
            values,np.zeros_like(values),source0,memory0,target,final_time,
            final_time/steps,volume,left,upper,
        )
        solutions.append(result)
        records.append({"steps":result["steps"],"time_step":result["time_step"]})
    differences=[]
    for left_result,right_result in zip(solutions[:-1],solutions[1:]):
        differences.append(np.hypot(
            state_norm(
                driver.wave,left_result["position"]-right_result["position"],
                left_result["source"]-right_result["source"],
            ),
            state_norm(
                driver.wave,left_result["velocity"]-right_result["velocity"],
                left_result["memory"]-right_result["memory"],
            ),
        ))
    differences=np.asarray(differences)
    return {
        "records":records,"successive_solution_differences":[float(value) for value in differences],
        "convergence_rates":[float(value) for value in np.log(differences[:-1]/differences[1:])/np.log(2.)],
    }


def main():
    geometry=build_geometry()
    print("sampling damped wave coefficients",flush=True)
    wave_coefficients=sample_coefficients(geometry,constraint_damping=CONSTRAINT_DAMPING)
    source_coefficients=sample_source_coefficients(geometry)
    grid=grid_audit(geometry,wave_coefficients,source_coefficients)
    time=timestep_audit(geometry,wave_coefficients,source_coefficients)
    axis={
        **{f"wave_{key}":value for key,value in wave_coefficients["axis_extrapolation_relative_differences"].items()},
        **{f"source_{key}":value for key,value in source_coefficients["axis_extrapolation_relative_differences"].items()},
    }
    acceptance={
        "selector_below_1e_8":geometry["selector_maximum"]<1e-8,
        "damping_is_active":wave_coefficients["constraint_damping_rate"]==CONSTRAINT_DAMPING,
        "all_axis_extrapolation_variants_below_5_percent":max(axis.values())<.05,
        "all_grid_errors_finite":all(np.isfinite(item["combined_relative_error"]) for item in grid["records"]),
        "fine_grid_rate_above_1p6":grid["convergence_rates"][-1]>1.6,
        "both_timestep_rates_above_3p5":min(time["convergence_rates"])>3.5,
    }
    payload={
        "status":"pass" if all(acceptance.values()) else "review",
        "scope":"corrected G6-fold regular nine-field wave runtime with constraint damping and six-field evolved GH source driver",
        "wave_field_count":9,"driver_field_order":list(DRIVER_FIELD_ORDER),
        "total_position_like_field_count":15,"constraint_damping_rate":CONSTRAINT_DAMPING,
        "driver_rates":{"mu":DRIVER_MU,"eta":DRIVER_ETA},
        "fold_amplitude":geometry["fold_amplitude"],"selector_maximum":geometry["selector_maximum"],
        "coefficient_support":{
            "z":[float(value) for value in source_coefficients["z"]],
            "r":[float(value) for value in source_coefficients["r"]],
        },
        "axis_extrapolation_relative_differences":axis,
        "grid_convergence":grid,"timestep_convergence":time,"acceptance":acceptance,
        "limitations":[
            "manufactured linear target and perturbations rather than a freely evolving constraint pulse",
            "zero-background-shift specialization; linear shift-advection coupling is not yet included",
            "driver target is prescribed rather than selected by a dynamical damped-wave gauge functional",
            "source fields vanish at the compact and outer radial boundaries, so general driver boundary data remain unaudited",
            "G6 fold and r<=4 runtime; radial-domain and G5/G6 driver transfer remain open",
            "nonlinear Einstein evolution and apparent-horizon tracking remain downstream",
        ],
    }
    Path("results/corrected_fold_gh_driver_runtime.json").write_text(
        json.dumps(payload,indent=2,sort_keys=True)+"\n"
    )
    print(json.dumps({"status":payload["status"],"acceptance":acceptance,"axis":axis,"grid":grid,"time":time},indent=2))


if __name__=="__main__":main()
