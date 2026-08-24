#!/usr/bin/env python3
"""Audit a live geometry-dependent GH target on the corrected fold."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.gh_source_driver import (
    RegularSO3AnchoredDampedWaveTarget,
    regular_so3_anchored_damped_wave_target_matrix,
    regular_so3_background_source_shift_advection,source_driver_rhs,
)
from bhps.regular_so3_gh_reduction import regular_so3_perturbation_jets
from run_corrected_fold_gh_driver_runtime import (
    CONSTRAINT_DAMPING,DRIVER_ETA,DRIVER_MU,
    interpolate_background_source_shift_data,interpolate_source_coefficients,
    sample_background_source_shift_data,sample_source_coefficients,state_norm,
)
from run_corrected_fold_regular_so3_runtime import (
    OMEGA,build_geometry,manufactured_problem,sample_coefficients,spline,
)


TARGET_MU_LAPSE=.4
TARGET_MU_SHIFT=.6
TARGET_DETERMINANT_POWER=.5
WAVE_SCALE=.02


def background_target_fields(geometry,z,r):
    zz,rr=np.meshgrid(z,r,indexing="ij")
    fields={}
    for name in ("psi","a","b","c"):
        fields[name]=spline(
            geometry[name],geometry["z"],geometry["r"],
        ).ev(zz.ravel(),rr.ravel()).reshape(len(z),len(r))
    lapse=fields["psi"]
    return {
        "lapse":lapse,
        "compact_scale":lapse*np.exp(fields["a"]),
        "radial_scale":lapse*np.exp(fields["b"]),
        "angular_scale":lapse*np.exp(fields["c"]),
    }


def make_target(geometry,z,r):
    data=background_target_fields(geometry,z,r)
    return RegularSO3AnchoredDampedWaveTarget(
        r,data["lapse"],data["compact_scale"],data["radial_scale"],
        data["angular_scale"],TARGET_MU_LAPSE,TARGET_MU_SHIFT,
        TARGET_DETERMINANT_POWER,
    )


def full_project_convention_target(metric,background_metric):
    spatial=metric[1:,1:];background_spatial=background_metric[1:,1:]
    shift_covector=metric[0,1:];shift=np.linalg.solve(spatial,shift_covector)
    lapse=np.sqrt(-metric[0,0]+shift_covector@shift)
    background_lapse=np.sqrt(-background_metric[0,0])
    logarithm=(
        TARGET_DETERMINANT_POWER*(
            np.log(np.linalg.det(spatial))-np.log(np.linalg.det(background_spatial))
        )-np.log(lapse/background_lapse)
    )
    normal=np.array((-lapse,0.,0.,0.,0.))
    # Project H=+Gamma is the negative of the published H^LS=-Gamma target.
    return (
        -TARGET_MU_LAPSE*logarithm*normal
        +TARGET_MU_SHIFT/lapse*(metric[:,1:]@shift)
    )


def pointwise_linearization_audit(geometry):
    z_values=np.geomspace(geometry["z"][0],geometry["z"][-1],5)
    r_values=np.array((.125,.25,.5,1.,2.,3.,4.))
    rng=np.random.default_rng(260813);records=[]
    for z in z_values:
        for radius in r_values:
            background=geometry["jet_field"].at(z,radius)["metric"]
            lapse=np.sqrt(-background[0,0]);compact=np.sqrt(background[1,1])
            radial=np.sqrt(background[2,2]);angular=np.sqrt(background[3,3])
            q=.15*rng.normal(size=9)
            matrix=regular_so3_anchored_damped_wave_target_matrix(
                np.array((radius,)),np.array((lapse,)),np.array((compact,)),
                np.array((radial,)),np.array((angular,)),TARGET_MU_LAPSE,
                TARGET_MU_SHIFT,TARGET_DETERMINANT_POWER,
            )[0]
            reduced=matrix@q
            perturbation=regular_so3_perturbation_jets(radius,q)["metric"]
            step=2e-6
            derivative=(
                full_project_convention_target(background+step*perturbation,background)
                -full_project_convention_target(background-step*perturbation,background)
            )/(2*step)
            direct=np.array((derivative[0],derivative[1],derivative[2]/radius))
            records.append({
                "z":float(z),"r":float(radius),
                "relative_difference":float(
                    np.linalg.norm(reduced-direct)/max(np.linalg.norm(direct),1e-300)
                ),
                "target_norm":float(np.linalg.norm(reduced)),
            })
    return {
        "records":records,
        "maximum_relative_difference":max(item["relative_difference"] for item in records),
        "minimum_nonzero_target_norm":min(item["target_norm"] for item in records),
    }


def target_matrix_support(geometry):
    z=np.geomspace(geometry["z"][0],geometry["z"][-1],5)
    r=np.array((0.,.125,.25,.5,1.,2.,3.,4.));data=background_target_fields(geometry,z,r)
    matrix=regular_so3_anchored_damped_wave_target_matrix(
        r,data["lapse"],data["compact_scale"],data["radial_scale"],
        data["angular_scale"],TARGET_MU_LAPSE,TARGET_MU_SHIFT,
        TARGET_DETERMINANT_POWER,
    )
    return z,r,matrix


def transfer_audit(g5,g6):
    z5,r5,m5=target_matrix_support(g5);z6,r6,m6=target_matrix_support(g6)
    if not np.allclose(z5,z6) or not np.array_equal(r5,r6):
        raise RuntimeError("G5/G6 target supports do not match")
    return {
        "g5_g6_relative_difference":float(
            np.linalg.norm(m5-m6)/max(np.linalg.norm(m6),1e-300)
        ),
        "axis_finite":bool(np.all(np.isfinite(m6[:,0]))),
        "axis_anisotropy_coefficient_maximum":float(np.max(np.abs(m6[:,0,:,4]))),
        "g6_target_matrix_norm":float(np.linalg.norm(m6)),
    }


def manufactured_problem_with_live_target(
    geometry,wave_coefficients,source_coefficients,shift_support,nz=13,nr=17,
):
    wave,_,base_values,_,base_left,base_right=manufactured_problem(
        geometry,wave_coefficients,nz,nr,
    )
    source_zero,source_first=interpolate_source_coefficients(
        source_coefficients,wave.z,wave.r,
    )
    shift_data=interpolate_background_source_shift_data(shift_support,wave.z,wave.r)
    from bhps.gh_source_driver import AxisymmetricDrivenGHWaveIBVP
    driver=AxisymmetricDrivenGHWaveIBVP(
        wave,source_zero,source_first,DRIVER_MU,DRIVER_ETA,
        radial_first_is_scaled=True,background_source_shift_data=shift_data,
    )
    target=make_target(geometry,wave.z,wave.r)
    q_amplitude=WAVE_SCALE*base_values
    target_amplitude=target.evaluate(q_amplitude)
    advection_amplitude=regular_so3_background_source_shift_advection(
        q_amplitude,wave.r,shift_data["source"],shift_data["z_first"],
        shift_data["radial_first_scaled"],shift_data["inverse_compact_metric"],
        shift_data["inverse_radial_metric"],
    )

    theta_denominator=DRIVER_ETA**2+OMEGA**2
    theta_cos=-DRIVER_ETA**2/theta_denominator*advection_amplitude
    theta_sin=-DRIVER_ETA*OMEGA/theta_denominator*advection_amplitude
    forcing_cos=DRIVER_MU*target_amplitude+advection_amplitude+theta_cos
    forcing_sin=theta_sin
    source_denominator=DRIVER_MU**2+OMEGA**2
    source_cos=(DRIVER_MU*forcing_cos-OMEGA*forcing_sin)/source_denominator
    source_sin=(OMEGA*forcing_cos+DRIVER_MU*forcing_sin)/source_denominator

    def exact(time):
        cosine=np.cos(OMEGA*time);sine=np.sin(OMEGA*time)
        q=cosine*q_amplitude;v=-OMEGA*sine*q_amplitude
        acceleration=-OMEGA**2*cosine*q_amplitude
        source=cosine*source_cos+sine*source_sin
        source_t=-OMEGA*sine*source_cos+OMEGA*cosine*source_sin
        memory=cosine*theta_cos+sine*theta_sin
        memory_t=-OMEGA*sine*theta_cos+OMEGA*cosine*theta_sin
        return q,v,acceleration,source,source_t,memory,memory_t

    left=lambda time,rq:WAVE_SCALE*base_left(time,rq)
    right=lambda time,rq:WAVE_SCALE*base_right(time,rq)

    def volume_source(time,zq,rq):
        del zq,rq
        q,v,desired,source,_,memory,_=exact(time)
        unforced=driver.rhs(
            time,q,v,source,memory,target,None,left,right,
        )[1]
        return desired-unforced

    return driver,target,exact,volume_source,left,right,target_amplitude,advection_amplitude


def exact_driver_residual(driver,target,exact,shift_data):
    maximum=0.
    for time in (0.,.017,.061,.1):
        q,_,_,source,source_t,memory,memory_t=exact(time)
        target_value=target.evaluate(q,time)
        advection=regular_so3_background_source_shift_advection(
            q,driver.wave.r,shift_data["source"],shift_data["z_first"],
            shift_data["radial_first_scaled"],shift_data["inverse_compact_metric"],
            shift_data["inverse_radial_metric"],
        )
        rhs_source,rhs_memory=source_driver_rhs(
            source,memory,target_value,DRIVER_MU,DRIVER_ETA,advection,
        )
        maximum=max(
            maximum,float(np.max(np.abs(source_t-rhs_source))),
            float(np.max(np.abs(memory_t-rhs_memory))),
        )
    return maximum


def timestep_audit(geometry,wave_coefficients,source_coefficients,shift_support,final_time=.08):
    setup=manufactured_problem_with_live_target(
        geometry,wave_coefficients,source_coefficients,shift_support,
    )
    driver,target,exact,volume,left,right,target_amplitude,advection_amplitude=setup
    q0,v0,_,source0,_,memory0,_=exact(0.);solutions=[];records=[]
    q_exact,v_exact,_,source_exact,_,memory_exact,_=exact(final_time)
    for steps in (8,16,32,64):
        result=driver.integrate(
            q0,v0,source0,memory0,target,final_time,final_time/steps,
            volume,left,right,
        )
        solutions.append(result)
        error=np.hypot(
            state_norm(driver.wave,result["position"]-q_exact,result["source"]-source_exact),
            state_norm(driver.wave,result["velocity"]-v_exact,result["memory"]-memory_exact),
        )
        scale=np.hypot(
            state_norm(driver.wave,q_exact,source_exact),
            state_norm(driver.wave,v_exact,memory_exact),
        )
        records.append({
            "steps":result["steps"],"time_step":result["time_step"],
            "combined_relative_error":float(error/scale),
        })
    errors=np.array([item["combined_relative_error"] for item in records])
    shift_data=interpolate_background_source_shift_data(shift_support,driver.wave.z,driver.wave.r)
    return {
        "records":records,
        "exact_error_convergence_rates":[
            float(value) for value in np.log(errors[:-1]/errors[1:])/np.log(2.)
        ],
        "exact_driver_residual":exact_driver_residual(driver,target,exact,shift_data),
        "target_amplitude_maximum":float(np.max(np.abs(target_amplitude))),
        "shift_advection_amplitude_maximum":float(np.max(np.abs(advection_amplitude))),
    }


def main():
    print("building corrected G5/G6 geometries",flush=True)
    g6=build_geometry("G6");g5=build_geometry("G5")
    pointwise=pointwise_linearization_audit(g6);transfer=transfer_audit(g5,g6)
    print("sampling corrected-G6 wave, source, and shift coefficients",flush=True)
    wave_coefficients=sample_coefficients(g6,constraint_damping=CONSTRAINT_DAMPING)
    source_coefficients=sample_source_coefficients(g6,CONSTRAINT_DAMPING)
    shift_support=sample_background_source_shift_data(g6)
    time=timestep_audit(g6,wave_coefficients,source_coefficients,shift_support)
    acceptance={
        "full_adm_linearization_relative_difference_below_1e_7":pointwise["maximum_relative_difference"]<1e-7,
        "target_is_nonzero":time["target_amplitude_maximum"]>1e-5,
        "shift_advection_remains_nonzero":time["shift_advection_amplitude_maximum"]>1e-5,
        "axis_target_is_finite":transfer["axis_finite"],
        "axis_anisotropy_target_coefficient_is_zero":transfer["axis_anisotropy_coefficient_maximum"]<1e-14,
        "g5_g6_target_matrix_difference_below_2_percent":transfer["g5_g6_relative_difference"]<.02,
        "exact_driver_residual_below_1e_11":time["exact_driver_residual"]<1e-11,
        "all_timestep_rates_above_3p5":min(time["exact_error_convergence_rates"])>3.5,
    }
    payload={
        "status":"pass" if all(acceptance.values()) else "review",
        "scope":"linear corrected-fold GH runtime with a live background-anchored damped-wave target recomputed from the metric",
        "source_sign_convention":"project H_a=+Gamma_a; target is the negative of the Lindblom--Szilagyi H_a=-Gamma_a convention",
        "target_parameters":{
            "mu_lapse":TARGET_MU_LAPSE,"mu_shift":TARGET_MU_SHIFT,
            "determinant_power":TARGET_DETERMINANT_POWER,
        },
        "driver_rates":{"mu":DRIVER_MU,"eta":DRIVER_ETA},
        "constraint_damping_rate":CONSTRAINT_DAMPING,
        "pointwise_full_adm_linearization":pointwise,
        "target_matrix_transfer":transfer,"timestep_convergence":time,
        "acceptance":acceptance,
        "limitations":[
            "background-anchored linearization rather than the nonlinear damped-wave target",
            "semidiscrete manufactured wave forcing isolates live target evaluation and RK4 coupling rather than repeating spatial convergence",
            "target and driver parameters are numerical controls, not a selected production window",
            "general driver and incoming-constraint boundary data remain open",
            "nonlinear Einstein evolution and horizon selection remain open",
        ],
    }
    Path("results/corrected_fold_geometry_target_audit.json").write_text(
        json.dumps(payload,indent=2,sort_keys=True)+"\n"
    )
    print(json.dumps({
        "status":payload["status"],"pointwise":pointwise,
        "transfer":transfer,"time":time,"acceptance":acceptance,
    },indent=2))


if __name__=="__main__":main()
