#!/usr/bin/env python3
"""Audit reflected gauge and physical characteristics at the outer face."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.constraint_ibvp import evaluate_regular_so3_constraint_field
from run_corrected_fold_boundary_constraint_pulse import (
    DAMPING_RATE,FINAL_TIME,build_case,regional_constraint_l2,
)
from run_corrected_fold_free_constraint_pulse import (
    constraint_energy,interpolate_constraint_coefficients,
    sample_constraint_coefficients,
)
from run_corrected_fold_gh_driver_runtime import (
    DRIVER_ETA,DRIVER_MU,interpolate_source_coefficients,sample_source_coefficients,
)
from run_corrected_fold_regular_so3_runtime import (
    build_geometry,sample_coefficients,
)


GRIDS=((41,61),(49,73))
SECTORS=("gauge","physical")
OUTPUT_PATH=Path("results/corrected_fold_outer_sector_pulses.json")


def characteristic_norms(setup,position,velocity,source,time):
    data=setup["complete_outer"].evaluate(position,velocity,source,time)
    incoming=data["incoming_characteristic"][:,-1,:7];values={}
    for sector in ("gauge","constraint","physical"):
        projected=np.asarray([
            setup["complete_outer"].projectors[i][sector]@incoming[i]
            for i in range(len(setup["wave"].z))
        ])
        density=np.sum(projected**2,axis=1)
        values[sector]=float(np.sqrt(max(0.,np.trapezoid(
            density,setup["wave"].z,
        ))))
    values["total"]=float(np.sqrt(sum(values[key]**2 for key in (
        "gauge","constraint","physical",
    ))))
    return values


def run_sector_case(setup,mode,time_step_factor=.026):
    wave=setup["wave"];driver=setup["driver"]
    if mode=="off":feedback=None
    elif mode=="constraint_only":feedback=setup["outer"]
    elif mode=="complete_3+3+1":feedback=setup["complete_outer"]
    elif mode=="sommerfeld_complete_3+3+1":feedback=setup["complete_sommerfeld_outer"]
    else:raise ValueError("invalid outer feedback mode")
    spacing=min(np.min(np.diff(wave.z)),np.min(np.diff(wave.r)))
    requested=time_step_factor*spacing/max(
        setup["maximum_radial_speed"],setup["maximum_compact_speed"],
    )
    estimated=max(1,int(np.ceil(FINAL_TIME/requested)));stride=max(1,estimated//36)
    zz,rr=np.meshgrid(wave.z,wave.r,indexing="ij");outer_band=rr>=3.65

    def diagnostic(time,q,v,h,theta):
        _,acceleration,h_dot,_=driver.rhs(
            time,q,v,h,theta,setup["target"],
            metric_boundary_constraint_feedback=feedback,
        )
        constraint=evaluate_regular_so3_constraint_field(
            wave.z,wave.r,q,v,h,setup["constraint_zero"],setup["constraint_first"],5,
            radial_first_is_scaled=True,
        )["constraint"]
        constraint_time=evaluate_regular_so3_constraint_field(
            wave.z,wave.r,v,acceleration,h_dot,setup["constraint_zero"],
            setup["constraint_first"],5,radial_first_is_scaled=True,
        )["constraint"]
        return {
            "time":float(time),"characteristic":characteristic_norms(setup,q,v,h,time),
            "constraint_energy":constraint_energy(wave,constraint,constraint_time)["energy"],
            "outer_constraint_l2":regional_constraint_l2(wave,constraint,outer_band),
            "corner_position_maximum":float(max(
                np.max(np.abs(q[0,-1])),np.max(np.abs(q[-1,-1])),
            )),
        }

    result=driver.integrate(
        setup["position"],setup["velocity"],setup["source"],setup["memory"],
        setup["target"],FINAL_TIME,requested,diagnostic=diagnostic,
        diagnostic_stride=stride,metric_boundary_constraint_feedback=feedback,
    )
    records=result["diagnostics"];selected=setup["pulse_sector"]
    post=[item for item in records if item["time"]>=setup["radial_arrival_lower_bound"]]
    return {
        "mode":mode,"grid_size":[wave.nz,wave.nr],"steps":result["steps"],
        "time_step":result["time_step"],"finite":bool(all(
            np.isfinite(item["characteristic"]["total"]) for item in records
        )),
        "initial_selected_incoming":records[0]["characteristic"][selected],
        "final_selected_incoming":records[-1]["characteristic"][selected],
        "postcontact_peak_selected_incoming":max(
            item["characteristic"][selected] for item in post
        ),
        "final_characteristic_norms":records[-1]["characteristic"],
        "final_outer_constraint_l2":records[-1]["outer_constraint_l2"],
        "final_constraint_energy":records[-1]["constraint_energy"],
        "maximum_corner_position":max(item["corner_position_maximum"] for item in records),
        "radial_arrival_lower_bound":setup["radial_arrival_lower_bound"],
        "compact_arrival_lower_bound":setup["compact_arrival_lower_bound"],
        "diagnostics":records,
    }


def sector_audit(
    geometry,wave_coefficients,source_coefficients,constraint_coefficients,sector,
):
    records=[]
    for grid in GRIDS:
        print(f"{sector} pulse grid {grid[0]} x {grid[1]}",flush=True)
        setup=build_case(
            geometry,wave_coefficients,source_coefficients,constraint_coefficients,
            *grid,pulse_sector=sector,
        )
        modes={mode:run_sector_case(setup,mode) for mode in (
            "off","constraint_only","complete_3+3+1",
        )}
        off=modes["off"];constraint=modes["constraint_only"];complete=modes["complete_3+3+1"]
        records.append({
            "grid_size":list(grid),"modes":modes,
            "complete_to_off_final_selected_ratio":complete["final_selected_incoming"]
                /max(off["final_selected_incoming"],1e-300),
            "complete_to_constraint_only_final_selected_ratio":complete["final_selected_incoming"]
                /max(constraint["final_selected_incoming"],1e-300),
            "complete_to_off_postcontact_peak_ratio":complete["postcontact_peak_selected_incoming"]
                /max(off["postcontact_peak_selected_incoming"],1e-300),
            "complete_to_off_outer_constraint_ratio":complete["final_outer_constraint_l2"]
                /max(off["final_outer_constraint_l2"],1e-300),
        })
    reductions=np.array([
        1-item["complete_to_off_final_selected_ratio"] for item in records
    ])
    agreement=(
        float(abs(reductions[-1]-reductions[-2])/max(abs(reductions[-1]),1e-300))
        if np.all(reductions>0) else None
    )
    acceptance={
        "initial_selected_incoming_below_1e_10":bool(max(
            item["modes"][mode]["initial_selected_incoming"]
            for item in records for mode in ("off","constraint_only","complete_3+3+1")
        )<1e-10),
        "radial_face_reached_before_final_time":bool(max(
            item["modes"]["complete_3+3+1"]["radial_arrival_lower_bound"]
            for item in records
        )<FINAL_TIME),
        "compact_walls_not_reached_before_final_time":bool(min(
            item["modes"]["complete_3+3+1"]["compact_arrival_lower_bound"]
            for item in records
        )>FINAL_TIME),
        "all_runs_finite":bool(all(
            item["modes"][mode]["finite"] for item in records
            for mode in ("off","constraint_only","complete_3+3+1")
        )),
        "fine_complete_corner_below_1_percent_of_pulse_amplitude":bool(
            records[-1]["modes"]["complete_3+3+1"]["maximum_corner_position"]<1e-6
        ),
        "fine_complete_corner_not_amplified_above_off_by_5_percent":bool(
            records[-1]["modes"]["complete_3+3+1"]["maximum_corner_position"]
            <1.05*max(records[-1]["modes"]["off"]["maximum_corner_position"],1e-300)
        ),
        "complete_reduces_fine_selected_reflection_by_2_percent":bool(
            records[-1]["complete_to_off_final_selected_ratio"]<.98
        ),
        "complete_outperforms_constraint_only_by_2_percent":bool(
            records[-1]["complete_to_constraint_only_final_selected_ratio"]<.98
        ),
        "complete_reduces_fine_postcontact_peak_by_2_percent":bool(
            records[-1]["complete_to_off_postcontact_peak_ratio"]<.98
        ),
        "two_grid_reduction_agrees_within_25_percent":bool(
            agreement is not None and agreement<.25
        ),
        "fine_outer_constraint_not_increased_by_10_percent":bool(
            records[-1]["complete_to_off_outer_constraint_ratio"]<1.1
        ),
    }
    return {
        "sector":sector,"records":records,
        "two_grid_reduction_relative_difference":agreement,
        "status":"pass" if all(acceptance.values()) else "review",
        "acceptance":acceptance,
    }


def main():
    geometry=build_geometry("G6")
    print(f"sampling kappa={DAMPING_RATE:g} coefficients",flush=True)
    wave_coefficients=sample_coefficients(geometry,constraint_damping=DAMPING_RATE)
    source_coefficients=sample_source_coefficients(geometry,DAMPING_RATE)
    constraint_coefficients=sample_constraint_coefficients(geometry)
    sectors=[sector_audit(
        geometry,wave_coefficients,source_coefficients,constraint_coefficients,sector,
    ) for sector in SECTORS]
    payload={
        "status":"pass" if all(item["status"]=="pass" for item in sectors) else "review",
        "scope":"sector-specific reflected incoming gauge and physical characteristics at corrected-G6 r=4",
        "damping_rate":DAMPING_RATE,"final_time":FINAL_TIME,"grids":[list(x) for x in GRIDS],
        "sectors":sectors,
        "limitations":[
            "linear corrected-G6 coefficients and small projector-pure metric pulses",
            "zero-incoming relaxation with unit gauge/physical rates",
            "coordinate regular-field characteristic norm rather than a covariant radiation norm",
            "physical condition is zero incoming linear radiation rather than a curvature-based nonlinear target",
            "r=6 transfer and nonlinear collapse remain open",
        ],
    }
    OUTPUT_PATH.write_text(
        json.dumps(payload,indent=2,sort_keys=True)+"\n"
    )
    print(json.dumps({
        "status":payload["status"],"sectors":[{
            "sector":item["sector"],"status":item["status"],
            "ratios":[{
                "grid":record["grid_size"],
                "complete_to_off":record["complete_to_off_final_selected_ratio"],
                "complete_to_constraint_only":record["complete_to_constraint_only_final_selected_ratio"],
                "peak_ratio":record["complete_to_off_postcontact_peak_ratio"],
            } for record in item["records"]],
            "agreement":item["two_grid_reduction_relative_difference"],
            "acceptance":item["acceptance"],
        } for item in sectors],
    },indent=2))


if __name__=="__main__":main()
