#!/usr/bin/env python3
"""Evolve a free, boundary-isolated GH constraint pulse on the corrected fold."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.constraint_ibvp import (
    REGULAR_SO3_CONSTRAINT_ORDER,evaluate_regular_so3_constraint_field,
    regular_so3_constraint_coefficient_matrices,
)
from bhps.gh_source_driver import AxisymmetricDrivenGHWaveIBVP
from run_corrected_fold_gh_driver_runtime import (
    DRIVER_ETA,DRIVER_MU,interpolate_source_coefficients,sample_source_coefficients,
)
from run_corrected_fold_regular_so3_runtime import (
    R_MAX,SUPPORT_R,SUPPORT_Z_COUNT,build_geometry,even_axis_extension,
    interpolate_tensor,sample_coefficients,sampled_system,
)


FINAL_TIME=.10
AMPLITUDE=1e-4
Z_CENTER=1.86
Z_HALF_WIDTH=.34
R_CENTER=1.45
R_HALF_WIDTH=.72
DAMPING_SWEEP=(0.,.5,1.,2.,4.)
SWEEP_GRID=(33,49)
REFINEMENT_GRIDS=((17,25),(33,49),(49,73))


def sample_constraint_coefficients(geometry,support_r=SUPPORT_R):
    support_r=np.asarray(support_r,dtype=float)
    z_support=np.geomspace(geometry["z"][0],geometry["z"][-1],SUPPORT_Z_COUNT)
    zero=np.empty((len(z_support),len(support_r),3,9))
    first=np.empty((3,len(z_support),len(support_r),3,9))
    for i,z_value in enumerate(z_support):
        print(f"constraint diagnostic support z {i+1}/{len(z_support)}",flush=True)
        for j,radius in enumerate(support_r):
            result=regular_so3_constraint_coefficient_matrices(
                geometry["jet_field"].at(z_value,radius),radius,
            )
            zero[i,j]=result["zero_matrix"];first[:,i,j]=result["first_matrices"]
    zero_ext,zero_axis=even_axis_extension(zero,support_r)
    time_ext,time_axis=even_axis_extension(first[0],support_r)
    z_ext,z_axis=even_axis_extension(first[1],support_r)
    radial_ext,radial_axis=even_axis_extension(
        first[2]*support_r[None,:,None,None],support_r,
    )
    return {
        "z":z_support,"r":np.r_[0.,support_r],"zero":zero_ext,
        "first":np.stack((time_ext,z_ext,radial_ext)),
        "radial_first_is_scaled":True,
        "axis_extrapolation_relative_differences":{
            "zero":zero_axis,"time_first":time_axis,"z_first":z_axis,
            "scaled_radial_first":radial_axis,
        },
    }


def interpolate_constraint_coefficients(coefficients,z,r):
    zero=interpolate_tensor(coefficients["zero"],coefficients["z"],coefficients["r"],z,r)
    first=np.empty((3,len(z),len(r),3,9))
    for direction in range(3):
        first[direction]=interpolate_tensor(
            coefficients["first"][direction],coefficients["z"],coefficients["r"],z,r,
        )
    return zero,first


def affine_coefficients(zero,unit,damping_rate):
    """Use exact linearity of every damping-dependent coefficient in kappa."""
    rate=float(damping_rate);result={}
    for key in ("reaction","first"):
        result[key]=zero[key]+rate*(unit[key]-zero[key])
    result.update({
        "z":zero["z"],"r":zero["r"],"radial_first_is_scaled":True,
        "principal_defect":max(zero["principal_defect"],unit["principal_defect"]),
        "constraint_damping_rate":rate,
    })
    return result


def affine_source_coefficients(zero,unit,damping_rate):
    rate=float(damping_rate)
    return {
        "z":zero["z"],"r":zero["r"],
        "zero":zero["zero"]+rate*(unit["zero"]-zero["zero"]),
        "first":zero["first"]+rate*(unit["first"]-zero["first"]),
        "radial_first_is_scaled":True,"constraint_damping_rate":rate,
    }


def smooth_bump(coordinate,center,half_width):
    scaled=(np.asarray(coordinate)-float(center))/float(half_width)
    result=np.zeros_like(scaled,dtype=float);inside=np.abs(scaled)<1
    result[inside]=np.exp(1-1/(1-scaled[inside]**2))
    return result


def initial_pulse(z,r):
    profile=smooth_bump(z[:,None],Z_CENTER,Z_HALF_WIDTH)*smooth_bump(
        r[None,:],R_CENTER,R_HALF_WIDTH,
    )
    position=np.zeros((len(z),len(r),9));position[:,:,2]=AMPLITUDE*profile
    return position,np.zeros_like(position)


def constraint_energy(wave,constraint,constraint_time):
    c=np.asarray(constraint).reshape(wave.nodes,3)
    ct=np.asarray(constraint_time).reshape(wave.nodes,3)
    l2_squared=float(np.sum((wave.mass@c)*c))
    kinetic=float(np.sum((wave.mass@ct)*ct))
    gradient=float(np.sum((wave.stiffness@c)*c))
    return {
        "l2_norm":float(np.sqrt(max(0.,l2_squared))),
        "energy":float(.5*max(0.,kinetic+gradient)),
        "kinetic":float(.5*max(0.,kinetic)),
        "gradient":float(.5*max(0.,gradient)),
    }


def boundary_amplitude_fraction(values,constraint):
    q=np.asarray(values);c=np.asarray(constraint);mask=np.zeros(q.shape[:2],dtype=bool)
    mask[:2,:]=True;mask[-2:,:]=True;mask[:,-2:]=True;mask[:,:2]=True
    return float(max(
        np.max(np.abs(q[mask]))/max(np.max(np.abs(q)),1e-300),
        np.max(np.abs(c[mask]))/max(np.max(np.abs(c)),1e-300),
    ))


def boundary_constraint_l2_fraction(wave,constraint):
    c=np.asarray(constraint);mask=np.zeros(c.shape[:2],dtype=bool)
    mask[:2,:]=True;mask[-2:,:]=True;mask[:,-2:]=True;mask[:,:2]=True
    lumped=np.asarray(wave.mass.sum(axis=1)).ravel().reshape(wave.nz,wave.nr)
    density=lumped[:,:,None]*c*c
    return float(np.sum(density[mask])/max(np.sum(density),1e-300))


def run_case(
    geometry,wave_coefficients,source_coefficients,constraint_coefficients,
    nz,nr,damping_rate,r_max=R_MAX,
):
    r_max=float(r_max)
    wave,principal,_,_,_,_=sampled_system(geometry,wave_coefficients,nz,nr,r_max=r_max)
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
    position,velocity=initial_pulse(wave.z,wave.r)
    source=np.zeros((nz,nr,3));memory=np.zeros_like(source);target=np.zeros_like(source)
    speed=max(float(np.max(principal["z_speed"])),float(np.max(principal["r_speed"])))
    support_clearance=min(
        Z_CENTER-Z_HALF_WIDTH-wave.z[0],wave.z[-1]-(Z_CENTER+Z_HALF_WIDTH),
        R_CENTER-R_HALF_WIDTH,r_max-(R_CENTER+R_HALF_WIDTH),
    )
    causal_arrival_lower_bound=support_clearance/speed
    spacing=min(np.min(np.diff(wave.z)),np.min(np.diff(wave.r)))
    requested_step=.022*spacing/speed
    estimated_steps=max(1,int(np.ceil(FINAL_TIME/requested_step)))
    stride=max(1,estimated_steps//24)

    def diagnostic(time,q,v,h,theta):
        _,acceleration,h_dot,_=driver.rhs(time,q,v,h,theta,target)
        c=evaluate_regular_so3_constraint_field(
            wave.z,wave.r,q,v,h,constraint_zero,constraint_first,5,
            radial_first_is_scaled=True,
        )["constraint"]
        c_time=evaluate_regular_so3_constraint_field(
            wave.z,wave.r,v,acceleration,h_dot,constraint_zero,constraint_first,5,
            radial_first_is_scaled=True,
        )["constraint"]
        energy=constraint_energy(wave,c,c_time)
        return {
            "time":float(time),**energy,
            "boundary_amplitude_fraction":boundary_amplitude_fraction(q,c),
            "boundary_constraint_l2_fraction":boundary_constraint_l2_fraction(wave,c),
            "metric_l2_norm":wave.l2_norm(q),
            "source_maximum":float(np.max(np.abs(h))),
        }

    result=driver.integrate(
        position,velocity,source,memory,target,FINAL_TIME,requested_step,
        diagnostic=diagnostic,diagnostic_stride=stride,
    )
    records=result["diagnostics"];initial=records[0];final=records[-1]
    peak=max(item["energy"] for item in records)
    return {
        "grid_size":[nz,nr],"damping_rate":float(damping_rate),
        "radial_domain_maximum":r_max,
        "time_step":result["time_step"],"steps":result["steps"],
        "maximum_coordinate_speed":speed,"support_clearance":support_clearance,
        "causal_boundary_arrival_lower_bound":causal_arrival_lower_bound,
        "stopped_before_boundary_arrival":bool(FINAL_TIME<causal_arrival_lower_bound),
        "initial_constraint_l2":initial["l2_norm"],
        "final_constraint_l2":final["l2_norm"],
        "constraint_l2_ratio":final["l2_norm"]/max(initial["l2_norm"],1e-300),
        "initial_constraint_energy":initial["energy"],
        "final_constraint_energy":final["energy"],
        "constraint_energy_ratio":final["energy"]/max(initial["energy"],1e-300),
        "peak_constraint_energy_ratio":peak/max(initial["energy"],1e-300),
        "final_boundary_amplitude_fraction":final["boundary_amplitude_fraction"],
        "maximum_boundary_amplitude_fraction":max(item["boundary_amplitude_fraction"] for item in records),
        "maximum_boundary_constraint_l2_fraction":max(item["boundary_constraint_l2_fraction"] for item in records),
        "source_remains_zero":bool(max(item["source_maximum"] for item in records)<1e-14),
        "diagnostics":records,
    }


def main():
    geometry=build_geometry()
    print("sampling undamped and unit-damped wave coefficients",flush=True)
    wave_zero=sample_coefficients(geometry,constraint_damping=0.)
    wave_unit=sample_coefficients(geometry,constraint_damping=1.)
    print("sampling undamped and unit-damped source couplings",flush=True)
    source_zero=sample_source_coefficients(geometry,0.)
    source_unit=sample_source_coefficients(geometry,1.)
    constraint_coefficients=sample_constraint_coefficients(geometry)

    sweep=[]
    for damping in DAMPING_SWEEP:
        print(f"free constraint sweep kappa={damping:g}",flush=True)
        sweep.append(run_case(
            geometry,affine_coefficients(wave_zero,wave_unit,damping),
            affine_source_coefficients(source_zero,source_unit,damping),
            constraint_coefficients,*SWEEP_GRID,damping,
        ))
    positive=[item for item in sweep if item["damping_rate"]>0]
    selected=min(positive,key=lambda item:item["constraint_energy_ratio"])["damping_rate"]
    print(f"selected damping kappa={selected:g} for refinement",flush=True)
    refinement=[]
    for grid in REFINEMENT_GRIDS:
        if list(grid)==list(SWEEP_GRID):
            item=next(item for item in sweep if item["damping_rate"]==selected)
        else:
            print(f"free constraint refinement grid {grid[0]} x {grid[1]}",flush=True)
            item=run_case(
                geometry,affine_coefficients(wave_zero,wave_unit,selected),
                affine_source_coefficients(source_zero,source_unit,selected),
                constraint_coefficients,*grid,selected,
            )
        refinement.append(item)

    ratios=np.array([item["constraint_energy_ratio"] for item in refinement])
    two_finest_relative_difference=float(abs(ratios[-1]-ratios[-2])/max(abs(ratios[-1]),1e-300))
    undamped=next(item for item in sweep if item["damping_rate"]==0)
    chosen=next(item for item in sweep if item["damping_rate"]==selected)
    axis_max=max(constraint_coefficients["axis_extrapolation_relative_differences"].values())
    acceptance={
        "initial_pulse_is_nonzero":min(item["initial_constraint_l2"] for item in refinement)>1e-10,
        "all_runs_stop_before_causal_boundary_arrival":all(item["stopped_before_boundary_arrival"] for item in sweep+refinement),
        "sweep_and_two_finest_near_boundary_amplitudes_below_0p5_percent":max(
            item["maximum_boundary_amplitude_fraction"] for item in sweep+refinement[-2:]
        )<.005,
        "sweep_and_two_finest_near_boundary_constraint_l2_fractions_below_1e_6":max(
            item["maximum_boundary_constraint_l2_fraction"] for item in sweep+refinement[-2:]
        )<1e-6,
        "coarsest_refinement_is_rejected_by_boundary_isolation":refinement[0]["maximum_boundary_amplitude_fraction"]>.005,
        "undamped_control_does_not_mimic_selected_decay":undamped["constraint_energy_ratio"]>1.35*chosen["constraint_energy_ratio"],
        "selected_damping_reduces_constraint_energy_by_20_percent":chosen["constraint_energy_ratio"]<.8,
        "selected_damping_has_no_energy_overshoot_above_10_percent":chosen["peak_constraint_energy_ratio"]<1.1,
        "selected_decay_ratio_two_finest_agrees_within_8_percent":two_finest_relative_difference<.08,
        "constraint_axis_extrapolation_below_5_percent":axis_max<.05,
        "source_zero_fixed_point_is_preserved":all(item["source_remains_zero"] for item in sweep+refinement),
    }
    payload={
        "status":"pass" if all(acceptance.values()) else "review",
        "scope":"free boundary-isolated regular SO(3) GH metric-constraint pulse on the corrected G6 fold",
        "constraint_order":list(REGULAR_SO3_CONSTRAINT_ORDER),
        "pulse":{
            "field":"h00","amplitude":AMPLITUDE,"z_center":Z_CENTER,
            "z_half_width":Z_HALF_WIDTH,"r_center":R_CENTER,"r_half_width":R_HALF_WIDTH,
            "volume_forcing":False,"initial_source":0.,"driver_target":0.,
        },
        "final_time":FINAL_TIME,"damping_sweep":sweep,
        "selected_damping_rate":selected,"selected_refinement":refinement,
        "selected_decay_ratio_two_finest_relative_difference":two_finest_relative_difference,
        "constraint_axis_extrapolation_relative_differences":constraint_coefficients["axis_extrapolation_relative_differences"],
        "acceptance":acceptance,
        "limitations":[
            "linearized corrected G6-fold coefficients and a small metric pulse",
            "zero gauge-source fixed point does not exercise a nonzero dynamical gauge target",
            "constraint energy is a positive finite-element diagnostic norm, not a curved-background exact conserved energy",
            "the 17 x 25 pulse is under-resolved and rejected by the boundary-isolation criterion; decay convergence uses 33 x 49 and 49 x 73",
            "zero-background-shift specialization and homogeneous driver boundary data",
            "G5 transfer and radial-domain enlargement remain separate gates",
            "nonlinear constraint growth and apparent-horizon dynamics remain open",
        ],
    }
    Path("results/corrected_fold_free_constraint_pulse.json").write_text(
        json.dumps(payload,indent=2,sort_keys=True)+"\n"
    )
    print(json.dumps({
        "status":payload["status"],"selected_damping_rate":selected,
        "sweep":[{
            "kappa":item["damping_rate"],"energy_ratio":item["constraint_energy_ratio"],
            "l2_ratio":item["constraint_l2_ratio"],"boundary":item["maximum_boundary_amplitude_fraction"],
        } for item in sweep],
        "refinement":[{
            "grid":item["grid_size"],"energy_ratio":item["constraint_energy_ratio"],
        } for item in refinement],
        "two_finest_relative_difference":two_finest_relative_difference,
        "acceptance":acceptance,
    },indent=2))


if __name__=="__main__":main()
