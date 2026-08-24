#!/usr/bin/env python3
"""Audit incoming source-characteristic feedback from the live GH constraint."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.constraint_ibvp import (
    RegularSO3ConstraintBoundaryFeedback,evaluate_regular_so3_constraint_field,
)
from bhps.gh_source_driver import AxisymmetricDrivenGHWaveIBVP
from run_corrected_fold_driver_boundary_audit import BOUNDARY_RATE,boundary_mask
from run_corrected_fold_free_constraint_pulse import (
    interpolate_constraint_coefficients,sample_constraint_coefficients,
)
from run_corrected_fold_gh_driver_runtime import (
    CONSTRAINT_DAMPING,DRIVER_ETA,DRIVER_MU,interpolate_source_coefficients,
    sample_source_coefficients,state_norm,
)
from run_corrected_fold_regular_so3_runtime import (
    build_geometry,manufactured_problem,sample_coefficients,
)


WAVE_SCALE=.01
OFFSET_VECTOR=np.array((.003,-.002,.0015))


def feedback_problem(
    geometry,wave_coefficients,source_coefficients,constraint_coefficients,
    nz=13,nr=17,
):
    wave,_,base_values,_,base_left,base_right=manufactured_problem(
        geometry,wave_coefficients,nz,nr,
    )
    source_zero,source_first=interpolate_source_coefficients(
        source_coefficients,wave.z,wave.r,
    )
    driver=AxisymmetricDrivenGHWaveIBVP(
        wave,source_zero,source_first,DRIVER_MU,DRIVER_ETA,
        radial_first_is_scaled=True,
    )
    constraint_zero,constraint_first=interpolate_constraint_coefficients(
        constraint_coefficients,wave.z,wave.r,
    )
    feedback=RegularSO3ConstraintBoundaryFeedback(
        wave.z,wave.r,constraint_zero,constraint_first,5,
        radial_first_is_scaled=True,
    )
    mask=boundary_mask(nz,nr);position=WAVE_SCALE*base_values
    velocity=np.zeros_like(position);zero_source=np.zeros((nz,nr,3))
    gamma=feedback.evaluate(position,velocity,zero_source)
    zz,rr=np.meshgrid(wave.z,wave.r,indexing="ij")
    profile=(1+.15*(zz-wave.z[0])/(wave.z[-1]-wave.z[0]))*(1-.1*(rr/wave.r[-1])**2)
    offset=profile[:,:,None]*OFFSET_VECTOR
    source0=gamma+offset;memory0=np.zeros_like(source0);target=source0.copy()

    def exact(time):
        decay=np.exp(-BOUNDARY_RATE*time)
        source=source0.copy();source_t=np.zeros_like(source)
        source[mask]=gamma[mask]+decay*offset[mask]
        source_t[mask]=-BOUNDARY_RATE*decay*offset[mask]
        return position,velocity,source,source_t,memory0

    left_values=WAVE_SCALE*base_left(0.,wave.r)
    right_values=WAVE_SCALE*base_right(0.,wave.r)
    left=lambda time,rq:left_values
    right=lambda time,rq:right_values

    def volume_source(time,zq,rq):
        del zq,rq
        q,v,source,_,memory=exact(time)
        unforced=driver.rhs(
            time,q,v,source,memory,target,None,left,right,
            None,mask,BOUNDARY_RATE,feedback,
        )[1]
        return -unforced

    return driver,feedback,mask,exact,target,volume_source,left,right,offset


def timestep_audit(
    geometry,wave_coefficients,source_coefficients,constraint_coefficients,
    final_time=.2,
):
    setup=feedback_problem(
        geometry,wave_coefficients,source_coefficients,constraint_coefficients,
    )
    driver,feedback,mask,exact,target,volume,left,right,offset=setup
    q0,v0,source0,_,memory0=exact(0.)
    q_exact,v_exact,source_exact,_,memory_exact=exact(final_time)
    initial_constraint=feedback.evaluate(q0,v0,source0)[mask]
    exact_final_constraint=feedback.evaluate(q_exact,v_exact,source_exact)[mask]
    records=[]
    for steps in (10,20,40,80,160,320):
        result=driver.integrate(
            q0,v0,source0,memory0,target,final_time,final_time/steps,
            volume,left,right,
            driver_boundary_incoming_mask=mask,
            driver_boundary_rate=BOUNDARY_RATE,
            driver_boundary_constraint=feedback,
        )
        numerical_constraint=feedback.evaluate(
            result["position"],result["velocity"],result["source"],
        )[mask]
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
            "boundary_constraint_maximum_error":float(np.max(np.abs(
                numerical_constraint-exact_final_constraint
            ))),
            "boundary_constraint_l2_ratio":float(
                np.linalg.norm(numerical_constraint)/np.linalg.norm(initial_constraint)
            ),
        })
    errors=np.array([item["combined_relative_error"] for item in records])
    exact_residual=0.
    for time in (0.,.037,.113,final_time):
        q,v,source,source_t,_=exact(time);constraint=feedback.evaluate(q,v,source)
        exact_residual=max(exact_residual,float(np.max(np.abs(
            source_t[mask]-BOUNDARY_RATE*constraint[mask]
        ))))
    return {
        "boundary_node_count":int(np.sum(mask)),
        "initial_boundary_constraint_l2":float(np.linalg.norm(initial_constraint)),
        "exact_final_to_initial_constraint_ratio":float(np.exp(-BOUNDARY_RATE*final_time)),
        "exact_feedback_residual":exact_residual,
        "offset_amplitude_maximum":float(np.max(np.abs(offset[mask]))),
        "records":records,
        "exact_error_convergence_rates":[
            float(value) for value in np.log(errors[:-1]/errors[1:])/np.log(2.)
        ],
    }


def main():
    geometry=build_geometry("G6")
    print("sampling corrected-G6 damped wave/source and constraint coefficients",flush=True)
    wave_coefficients=sample_coefficients(geometry,constraint_damping=CONSTRAINT_DAMPING)
    source_coefficients=sample_source_coefficients(geometry,CONSTRAINT_DAMPING)
    constraint_coefficients=sample_constraint_coefficients(geometry)
    result=timestep_audit(
        geometry,wave_coefficients,source_coefficients,constraint_coefficients,
    )
    finest=result["records"][-1];expected=result["exact_final_to_initial_constraint_ratio"]
    acceptance={
        "initial_boundary_constraint_is_nonzero":result["initial_boundary_constraint_l2"]>1e-5,
        "boundary_feedback_data_are_nonhomogeneous":result["offset_amplitude_maximum"]>1e-5,
        "exact_feedback_residual_below_1e_12":result["exact_feedback_residual"]<1e-12,
        "all_timestep_rates_above_3p5":min(result["exact_error_convergence_rates"])>3.5,
        "finest_boundary_constraint_error_below_1e_8":finest["boundary_constraint_maximum_error"]<1e-8,
        "finest_constraint_decay_ratio_matches_exact":abs(
            finest["boundary_constraint_l2_ratio"]-expected
        )<1e-8,
    }
    payload={
        "status":"pass" if all(acceptance.values()) else "review",
        "scope":"corrected-G6 live C_a=Gamma_a-H_a feedback on controlled incoming H_a boundary characteristics",
        "source_sign_mapping":"Lindblom--Szilagyi E9 maps to project partial_t H_a=+mu_B C_a",
        "boundary_rate":BOUNDARY_RATE,"timestep_convergence":result,
        "acceptance":acceptance,
        "limitations":[
            "source-characteristic constraint feedback only",
            "the incoming metric-wave constraint projection remains open",
            "static manufactured metric with volume forcing",
            "controlled incoming characteristic rather than zero-shift background boundary flow",
            "linear corrected-fold runtime rather than nonlinear Einstein evolution",
        ],
    }
    Path("results/corrected_fold_source_constraint_boundary_audit.json").write_text(
        json.dumps(payload,indent=2,sort_keys=True)+"\n"
    )
    print(json.dumps({
        "status":payload["status"],"result":result,"acceptance":acceptance,
    },indent=2))


if __name__=="__main__":main()
