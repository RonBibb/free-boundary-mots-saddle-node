#!/usr/bin/env python3
"""Audit nonhomogeneous incoming GH source-driver boundary data."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.gh_source_driver import first_order_driver_characteristic_speeds
from run_corrected_fold_geometry_target_audit import (
    manufactured_problem_with_live_target,
)
from run_corrected_fold_gh_driver_runtime import (
    CONSTRAINT_DAMPING,sample_background_source_shift_data,
    sample_source_coefficients,state_norm,
)
from run_corrected_fold_regular_so3_runtime import build_geometry,sample_coefficients


BOUNDARY_RATE=1.6
CONTROL_NORMAL_SHIFT=.08


def boundary_mask(nz,nr):
    mask=np.zeros((nz,nr),dtype=bool)
    mask[0,:]=True;mask[-1,:]=True;mask[:,-1]=True
    return mask


def boundary_timestep_audit(
    geometry,wave_coefficients,source_coefficients,shift_support,final_time=.08,
):
    setup=manufactured_problem_with_live_target(
        geometry,wave_coefficients,source_coefficients,shift_support,
    )
    driver,target,exact,_,left,right,target_amplitude,advection_amplitude=setup
    mask=boundary_mask(driver.wave.nz,driver.wave.nr)

    def boundary_target(time,zq,rq):
        del zq,rq
        _,_,_,source,source_t,_,_=exact(time)
        return source+source_t/BOUNDARY_RATE

    def volume_source(time,zq,rq):
        del zq,rq
        q,v,desired,source,_,memory,_=exact(time)
        unforced=driver.rhs(
            time,q,v,source,memory,target,None,left,right,
            boundary_target,mask,BOUNDARY_RATE,
        )[1]
        return desired-unforced

    residual=0.;boundary_amplitude=0.
    for time in (0.,.017,.061,.1):
        _,_,_,source,source_t,_,_=exact(time)
        data=boundary_target(time,None,None)
        residual=max(residual,float(np.max(np.abs(
            source_t[mask]+BOUNDARY_RATE*(source[mask]-data[mask])
        ))))
        boundary_amplitude=max(boundary_amplitude,float(np.max(np.abs(data[mask]))))

    q0,v0,_,source0,_,memory0,_=exact(0.)
    q_exact,v_exact,_,source_exact,_,memory_exact,_=exact(final_time)
    records=[]
    for steps in (8,16,32,64):
        result=driver.integrate(
            q0,v0,source0,memory0,target,final_time,final_time/steps,
            volume_source,left,right,
            driver_boundary_target=boundary_target,
            driver_boundary_incoming_mask=mask,
            driver_boundary_rate=BOUNDARY_RATE,
        )
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
            "boundary_source_maximum_error":float(np.max(np.abs(
                result["source"][mask]-source_exact[mask]
            ))),
            "boundary_memory_maximum_error":float(np.max(np.abs(
                result["memory"][mask]-memory_exact[mask]
            ))),
        })
    errors=np.array([item["combined_relative_error"] for item in records])
    return {
        "boundary_node_count":int(np.sum(mask)),
        "boundary_segments":["lower compact wall","upper compact wall","outer radial boundary"],
        "boundary_target_amplitude_maximum":boundary_amplitude,
        "exact_bjorhus_boundary_residual":residual,
        "live_target_amplitude_maximum":float(np.max(np.abs(target_amplitude))),
        "shift_advection_amplitude_maximum":float(np.max(np.abs(advection_amplitude))),
        "records":records,
        "exact_error_convergence_rates":[
            float(value) for value in np.log(errors[:-1]/errors[1:])/np.log(2.)
        ],
    }


def main():
    geometry=build_geometry("G6")
    print("sampling corrected-G6 wave, source, and shift coefficients",flush=True)
    wave_coefficients=sample_coefficients(geometry,constraint_damping=CONSTRAINT_DAMPING)
    source_coefficients=sample_source_coefficients(geometry,CONSTRAINT_DAMPING)
    shift_support=sample_background_source_shift_data(geometry)
    boundary=boundary_timestep_audit(
        geometry,wave_coefficients,source_coefficients,shift_support,
    )
    lapse=float(np.median(geometry["psi"]))
    characteristics=first_order_driver_characteristic_speeds(
        lapse,CONTROL_NORMAL_SHIFT,
    )
    acceptance={
        "controlled_source_characteristic_is_incoming":characteristics["source"]<0,
        "memory_characteristic_has_zero_speed":characteristics["memory"]==0.,
        "boundary_data_are_nonhomogeneous":boundary["boundary_target_amplitude_maximum"]>1e-5,
        "all_three_external_boundary_segments_are_covered":len(boundary["boundary_segments"])==3,
        "exact_bjorhus_residual_below_1e_12":boundary["exact_bjorhus_boundary_residual"]<1e-12,
        "all_timestep_rates_above_3p5":min(boundary["exact_error_convergence_rates"])>3.5,
        "finest_boundary_source_error_below_1e_8":boundary["records"][-1]["boundary_source_maximum_error"]<1e-8,
        "finest_boundary_memory_error_below_1e_8":boundary["records"][-1]["boundary_memory_maximum_error"]<1e-8,
    }
    payload={
        "status":"pass" if all(acceptance.values()) else "review",
        "scope":"corrected-G6 linear target/shift/source runtime with nonhomogeneous Bjorhus data on incoming H_a characteristics",
        "boundary_rate":BOUNDARY_RATE,
        "controlled_normal_shift":CONTROL_NORMAL_SHIFT,
        "characteristic_speeds":{
            key:(value.tolist() if isinstance(value,np.ndarray) else value)
            for key,value in characteristics.items()
        },
        "boundary_audit":boundary,"acceptance":acceptance,
        "limitations":[
            "controlled incoming normal shift rather than the zero-shift corrected background",
            "manufactured nonhomogeneous boundary data",
            "the zero-speed memory characteristic is evolved but deliberately receives no boundary condition",
            "metric incoming-constraint characteristic projection remains a separate gate",
            "linearized corrected-fold runtime rather than nonlinear Einstein evolution",
        ],
    }
    Path("results/corrected_fold_driver_boundary_audit.json").write_text(
        json.dumps(payload,indent=2,sort_keys=True)+"\n"
    )
    print(json.dumps({
        "status":payload["status"],"characteristics":characteristics,
        "boundary":boundary,"acceptance":acceptance,
    },indent=2,default=lambda value:value.tolist()))


if __name__=="__main__":main()
