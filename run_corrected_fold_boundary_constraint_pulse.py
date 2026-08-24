#!/usr/bin/env python3
"""Send a free GH constraint pulse into the live outer metric projector."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.constraint_ibvp import (
    REGULAR_SO3_CONSTRAINT_ORDER,RegularSO3ConstraintBoundaryFeedback,
    RegularSO3OuterMetricConstraintFeedback,
    RegularSO3OuterMetricCharacteristicFeedback,RegularSO3OuterMetricWeakFluxFeedback,
    evaluate_regular_so3_constraint_field,
    regular_so3_metric_characteristic_projector_matrices,
)
from bhps.gh_source_driver import AxisymmetricDrivenGHWaveIBVP
from run_corrected_fold_free_constraint_pulse import (
    boundary_amplitude_fraction,constraint_energy,
    interpolate_constraint_coefficients,sample_constraint_coefficients,smooth_bump,
)
from run_corrected_fold_gh_driver_runtime import (
    DRIVER_ETA,DRIVER_MU,
    interpolate_source_coefficients,sample_source_coefficients,
)
from run_corrected_fold_regular_so3_runtime import (
    build_geometry,sample_coefficients,sampled_system,
)


FINAL_TIME=.55
DAMPING_RATE=4.
AMPLITUDE=1e-4
Z_CENTER=(1.+np.e)/2
Z_HALF_WIDTH=.15
R_CENTER=3.53
R_HALF_WIDTH=.12
GRIDS=((49,73),(57,85))


def smooth_bump_derivative(coordinate,center,half_width):
    coordinate=np.asarray(coordinate,dtype=float);width=float(half_width)
    scaled=(coordinate-float(center))/width;value=smooth_bump(coordinate,center,width)
    result=np.zeros_like(value);inside=np.abs(scaled)<1
    result[inside]=value[inside]*(-2*scaled[inside])/(width*(1-scaled[inside]**2)**2)
    return result


def outgoing_initial_pulse(
    z,r,radial_speed,constraint_basis,r_center=R_CENTER,r_half_width=R_HALF_WIDTH,
):
    z_profile=smooth_bump(z,Z_CENTER,Z_HALF_WIDTH)
    radial=smooth_bump(r,r_center,r_half_width)
    radial_first=smooth_bump_derivative(r,r_center,r_half_width)
    basis=np.asarray(constraint_basis,dtype=float)
    if basis.shape!=(len(z),7):raise ValueError("constraint pulse basis has the wrong shape")
    position=np.zeros((len(z),len(r),9));velocity=np.zeros_like(position)
    position[:,:,:7]=(
        AMPLITUDE*z_profile[:,None,None]*radial[None,:,None]*basis[:,None,:]
    )
    velocity[:,:,:7]=(
        -AMPLITUDE*radial_speed[:,:,None]*z_profile[:,None,None]
        *radial_first[None,:,None]*basis[:,None,:]
    )
    return position,velocity


def regional_constraint_l2(wave,constraint,mask):
    lumped=np.asarray(wave.mass.sum(axis=1)).ravel().reshape(wave.nz,wave.nr)
    density=lumped[:,:,None]*np.asarray(constraint)**2
    return float(np.sqrt(max(0.,np.sum(density[np.asarray(mask,dtype=bool)]))))


def build_case(
    geometry,wave_coefficients,source_coefficients,constraint_coefficients,nz,nr,
    pulse_sector="constraint",r_max=4.,r_center=R_CENTER,
    r_half_width=R_HALF_WIDTH,sommerfeld_penalty=1.,
):
    wave,principal,_,_,_,_=sampled_system(
        geometry,wave_coefficients,nz,nr,r_max=r_max,outer_dirichlet=False,
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
    live_constraint=RegularSO3ConstraintBoundaryFeedback(
        wave.z,wave.r,constraint_zero,constraint_first,5,
        radial_first_is_scaled=True,
    )
    metrics=np.asarray([
        geometry["jet_field"].at(z_value,wave.r[-1])["metric"]
        for z_value in wave.z
    ])
    outer=RegularSO3OuterMetricConstraintFeedback(
        live_constraint,metrics,include_corners=False,
    )
    complete_outer=RegularSO3OuterMetricCharacteristicFeedback(
        live_constraint,metrics,gauge_rate=1.,physical_rate=1.,gamma2=0.,
        stencil_width=5,include_corners=False,
    )
    outer_surface=np.sqrt(
        principal["mass_weight"][:,-1]*principal["r_gradient_weight"][:,-1]
    )
    weak_outer=RegularSO3OuterMetricWeakFluxFeedback(outer,outer_surface)
    complete_weak_outer=RegularSO3OuterMetricWeakFluxFeedback(
        complete_outer,outer_surface,
    )
    sommerfeld_constraint_outer=RegularSO3OuterMetricWeakFluxFeedback(
        outer,outer_surface,constraint_projectors=complete_outer.projectors,
        constraint_sommerfeld=True,penalty=sommerfeld_penalty,
    )
    complete_sommerfeld_outer=RegularSO3OuterMetricWeakFluxFeedback(
        complete_outer,outer_surface,constraint_projectors=complete_outer.projectors,
        constraint_sommerfeld=True,penalty=sommerfeld_penalty,
    )
    sector=str(pulse_sector)
    if sector not in ("gauge","constraint","physical","mixed"):
        raise ValueError("pulse sector must be gauge, constraint, physical, or mixed")
    seed=np.array((.31,-.22,.17,-.29,.13,.23,-.19))
    sector_basis=np.empty((len(wave.z),7))
    for i,metric in enumerate(metrics):
        projectors=regular_so3_metric_characteristic_projector_matrices(
            metric,wave.r[-1],"radial",1.,
        )
        sector_basis[i]=(
            seed if sector=="mixed" else projectors[sector]@seed
        )
    basis_scale=max(np.max(np.linalg.norm(sector_basis,axis=1)),1e-300)
    if basis_scale<1e-12:raise RuntimeError("selected pulse seed misses its characteristic sector")
    sector_basis/=basis_scale
    position,velocity=outgoing_initial_pulse(
        wave.z,wave.r,principal["r_speed"],sector_basis,r_center,r_half_width,
    )
    source=np.zeros((nz,nr,3));memory=np.zeros_like(source);target=np.zeros_like(source)
    speed_r=float(np.max(principal["r_speed"]));speed_z=float(np.max(principal["z_speed"]))
    radial_clearance=wave.r[-1]-(float(r_center)+float(r_half_width))
    compact_clearance=min(
        Z_CENTER-Z_HALF_WIDTH-wave.z[0],wave.z[-1]-(Z_CENTER+Z_HALF_WIDTH),
    )
    return {
        "wave":wave,"driver":driver,"constraint_zero":constraint_zero,
        "constraint_first":constraint_first,"outer":outer,
        "complete_outer":complete_outer,"weak_outer":weak_outer,
        "complete_weak_outer":complete_weak_outer,"position":position,
        "sommerfeld_constraint_outer":sommerfeld_constraint_outer,
        "complete_sommerfeld_outer":complete_sommerfeld_outer,
        "velocity":velocity,"source":source,"memory":memory,"target":target,
        "radial_arrival_lower_bound":radial_clearance/speed_r,
        "compact_arrival_lower_bound":compact_clearance/speed_z,
        "maximum_radial_speed":speed_r,"maximum_compact_speed":speed_z,
        "pulse_sector":sector,"radial_domain_maximum":float(r_max),
        "radial_pulse_center":float(r_center),
        "radial_pulse_half_width":float(r_half_width),
        "outer_band_lower":float(r_max)-.35,
        "sommerfeld_penalty":float(sommerfeld_penalty),
    }


def run_case(setup,feedback_mode,time_step_factor=.026):
    wave=setup["wave"];driver=setup["driver"]
    spacing=min(np.min(np.diff(wave.z)),np.min(np.diff(wave.r)))
    requested=time_step_factor*spacing/max(
        setup["maximum_radial_speed"],setup["maximum_compact_speed"],
    )
    estimated=max(1,int(np.ceil(FINAL_TIME/requested)));stride=max(1,estimated//36)
    mode=str(feedback_mode)
    if mode=="off":outer_feedback=None
    elif mode=="constraint_only":outer_feedback=setup["outer"]
    elif mode=="complete_3+3+1":outer_feedback=setup["complete_outer"]
    elif mode=="weak_constraint_only":outer_feedback=setup["weak_outer"]
    elif mode=="weak_complete_3+3+1":outer_feedback=setup["complete_weak_outer"]
    elif mode=="sommerfeld_constraint_only":outer_feedback=setup["sommerfeld_constraint_outer"]
    elif mode=="sommerfeld_complete_3+3+1":outer_feedback=setup["complete_sommerfeld_outer"]
    else:raise ValueError("invalid outer feedback mode")
    radial_arrival=setup["radial_arrival_lower_bound"]
    zz,rr=np.meshgrid(wave.z,wave.r,indexing="ij")
    outer_band=rr>=setup["outer_band_lower"];interior=rr<setup["outer_band_lower"]

    def diagnostic(time,q,v,h,theta):
        _,acceleration,h_dot,_=driver.rhs(
            time,q,v,h,theta,setup["target"],
            metric_boundary_constraint_feedback=outer_feedback,
        )
        constraint=evaluate_regular_so3_constraint_field(
            wave.z,wave.r,q,v,h,setup["constraint_zero"],setup["constraint_first"],5,
            radial_first_is_scaled=True,
        )["constraint"]
        constraint_time=evaluate_regular_so3_constraint_field(
            wave.z,wave.r,v,acceleration,h_dot,setup["constraint_zero"],
            setup["constraint_first"],5,radial_first_is_scaled=True,
        )["constraint"]
        energy=constraint_energy(wave,constraint,constraint_time)
        return {
            "time":float(time),**energy,
            "outer_band_constraint_l2":regional_constraint_l2(wave,constraint,outer_band),
            "interior_constraint_l2":regional_constraint_l2(wave,constraint,interior),
            "boundary_amplitude_fraction":boundary_amplitude_fraction(q,constraint),
            "corner_position_maximum":float(max(
                np.max(np.abs(q[0,-1])),np.max(np.abs(q[-1,-1])),
            )),
        }

    result=driver.integrate(
        setup["position"],setup["velocity"],setup["source"],setup["memory"],
        setup["target"],FINAL_TIME,requested,diagnostic=diagnostic,
        diagnostic_stride=stride,metric_boundary_constraint_feedback=outer_feedback,
    )
    records=result["diagnostics"];initial=records[0];final=records[-1]
    post=[item for item in records if item["time"]>=radial_arrival]
    return {
        "feedback_mode":mode,"grid_size":[wave.nz,wave.nr],
        "steps":result["steps"],"time_step":result["time_step"],
        "radial_arrival_lower_bound":setup["radial_arrival_lower_bound"],
        "compact_arrival_lower_bound":setup["compact_arrival_lower_bound"],
        "initial_constraint_l2":initial["l2_norm"],
        "final_constraint_l2":final["l2_norm"],
        "final_constraint_energy":final["energy"],
        "final_outer_band_constraint_l2":final["outer_band_constraint_l2"],
        "final_interior_constraint_l2":final["interior_constraint_l2"],
        "postcontact_peak_outer_band_constraint_l2":max(
            item["outer_band_constraint_l2"] for item in post
        ),
        "postcontact_peak_constraint_energy":max(item["energy"] for item in post),
        "initial_boundary_amplitude_fraction":initial["boundary_amplitude_fraction"],
        "maximum_corner_position":max(item["corner_position_maximum"] for item in records),
        "finite":bool(all(np.isfinite(item["energy"]) for item in records)),
        "diagnostics":records,
    }


def main():
    geometry=build_geometry("G6")
    print(f"sampling kappa={DAMPING_RATE:g} wave/source/constraint coefficients",flush=True)
    wave_coefficients=sample_coefficients(
        geometry,constraint_damping=DAMPING_RATE,
    )
    source_coefficients=sample_source_coefficients(geometry,DAMPING_RATE)
    constraint_coefficients=sample_constraint_coefficients(geometry)
    pairs=[]
    for grid in GRIDS:
        print(f"boundary-reaching pulse grid {grid[0]} x {grid[1]}",flush=True)
        setup=build_case(
            geometry,wave_coefficients,source_coefficients,constraint_coefficients,*grid,
        )
        off=run_case(setup,"off");on=run_case(setup,"constraint_only")
        complete=run_case(setup,"complete_3+3+1")
        pairs.append({
            "grid_size":list(grid),"feedback_off":off,"feedback_on":on,
            "complete_characteristic":complete,
            "final_outer_constraint_ratio_on_to_off":on["final_outer_band_constraint_l2"]
                /max(off["final_outer_band_constraint_l2"],1e-300),
            "postcontact_outer_peak_ratio_on_to_off":on["postcontact_peak_outer_band_constraint_l2"]
                /max(off["postcontact_peak_outer_band_constraint_l2"],1e-300),
            "final_total_energy_ratio_on_to_off":on["final_constraint_energy"]
                /max(off["final_constraint_energy"],1e-300),
            "final_outer_constraint_ratio_complete_to_off":complete["final_outer_band_constraint_l2"]
                /max(off["final_outer_band_constraint_l2"],1e-300),
            "final_outer_constraint_ratio_complete_to_constraint_only":complete["final_outer_band_constraint_l2"]
                /max(on["final_outer_band_constraint_l2"],1e-300),
            "final_total_energy_ratio_complete_to_off":complete["final_constraint_energy"]
                /max(off["final_constraint_energy"],1e-300),
        })
    ratios=np.array([item["final_outer_constraint_ratio_on_to_off"] for item in pairs])
    reductions=1-ratios
    fine_difference=(
        float(abs(reductions[-1]-reductions[-2])/max(abs(reductions[-1]),1e-300))
        if len(ratios)>=2 and reductions[-1]>0 and reductions[-2]>0 else None
    )
    complete_ratios=np.array([
        item["final_outer_constraint_ratio_complete_to_off"] for item in pairs
    ])
    complete_reductions=1-complete_ratios
    complete_fine_difference=(
        float(abs(complete_reductions[-1]-complete_reductions[-2])
              /max(abs(complete_reductions[-1]),1e-300))
        if len(complete_ratios)>=2 and complete_reductions[-1]>0
        and complete_reductions[-2]>0 else None
    )
    acceptance={
        "pulse_initially_boundary_isolated_below_1e_8":bool(max(
            item[mode]["initial_boundary_amplitude_fraction"]
            for item in pairs for mode in (
                "feedback_off","feedback_on","complete_characteristic",
            )
        )<1e-8),
        "radial_face_reached_before_final_time":bool(max(
            item["feedback_on"]["radial_arrival_lower_bound"] for item in pairs
        )<FINAL_TIME),
        "compact_walls_not_reached_before_final_time":bool(min(
            item["feedback_on"]["compact_arrival_lower_bound"] for item in pairs
        )>FINAL_TIME),
        "all_runs_finite":bool(all(
            item[mode]["finite"] for item in pairs for mode in (
                "feedback_off","feedback_on","complete_characteristic",
            )
        )),
        "physical_wall_corners_remain_zero_below_1e_13":bool(max(
            item[mode]["maximum_corner_position"]
            for item in pairs for mode in ("feedback_on","complete_characteristic")
        )<1e-13),
        "feedback_reduces_fine_final_outer_constraint":bool(ratios[-1]<1.),
        "feedback_reduces_fine_postcontact_outer_peak":bool(pairs[-1]["postcontact_outer_peak_ratio_on_to_off"]<1.),
        "feedback_effect_two_finest_agrees_within_15_percent":bool(
            fine_difference is not None and fine_difference<.15
        ),
        "complete_characteristic_reduces_fine_outer_constraint_by_5_percent":bool(
            complete_ratios[-1]<.95
        ),
        "complete_characteristic_outperforms_constraint_only_on_fine_grid":bool(
            pairs[-1]["final_outer_constraint_ratio_complete_to_constraint_only"]<1.
        ),
        "complete_characteristic_effect_two_finest_agrees_within_20_percent":bool(
            complete_fine_difference is not None and complete_fine_difference<.20
        ),
    }
    payload={
        "status":"pass" if all(acceptance.values()) else "review",
        "scope":"free outward GH constraint pulse reaching the live corrected-G6 outer metric boundary",
        "constraint_order":list(REGULAR_SO3_CONSTRAINT_ORDER),
        "pulse":{
            "field":"outer-constraint-projector image of an h00 seed",
            "amplitude":AMPLITUDE,"z_center":Z_CENTER,
            "z_half_width":Z_HALF_WIDTH,"r_center":R_CENTER,
            "r_half_width":R_HALF_WIDTH,"outgoing_velocity":True,
            "volume_forcing":False,"initial_source":0.,"driver_target":0.,
        },
        "final_time":FINAL_TIME,"damping_rate":DAMPING_RATE,
        "grid_pairs":pairs,"two_finest_feedback_effect_relative_difference":fine_difference,
        "two_finest_complete_effect_relative_difference":complete_fine_difference,
        "acceptance":acceptance,
        "limitations":[
            "linear corrected-G6 coefficients and a small metric pulse",
            "causal arrival bounds use maximum coordinate speeds",
            "constraint energy is a positive diagnostic norm rather than an exact curved-background energy",
            "complete 3+3+1 mode uses zero-incoming relaxation with unit gauge/physical rates rather than a nonlinear radiation target",
            "r=6 transfer and nonlinear collapse remain open",
        ],
    }
    Path("results/corrected_fold_boundary_constraint_pulse.json").write_text(
        json.dumps(payload,indent=2,sort_keys=True)+"\n"
    )
    print(json.dumps({
        "status":payload["status"],"summary":[{
            "grid":item["grid_size"],
            "final_outer_ratio":item["final_outer_constraint_ratio_on_to_off"],
            "complete_outer_ratio":item["final_outer_constraint_ratio_complete_to_off"],
            "complete_to_constraint_only":item["final_outer_constraint_ratio_complete_to_constraint_only"],
            "postcontact_peak_ratio":item["postcontact_outer_peak_ratio_on_to_off"],
            "energy_ratio":item["final_total_energy_ratio_on_to_off"],
        } for item in pairs],"two_finest_difference":fine_difference,
        "two_finest_complete_difference":complete_fine_difference,
        "acceptance":acceptance,
    },indent=2))


if __name__=="__main__":main()
