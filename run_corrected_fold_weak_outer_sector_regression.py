#!/usr/bin/env python3
"""Check gauge/physical non-regression for the passing weak 3+3+1 flux."""

import json
from pathlib import Path

import numpy as np

from run_corrected_fold_boundary_constraint_pulse import (
    AMPLITUDE,DAMPING_RATE,FINAL_TIME,build_case,build_geometry,sample_coefficients,
)
from run_corrected_fold_free_constraint_pulse import sample_constraint_coefficients
from run_corrected_fold_gh_driver_runtime import sample_source_coefficients
from run_corrected_fold_outer_sector_pulses import run_sector_case


OUTPUT=Path("results/corrected_fold_weak_outer_sector_regression.json")
BASELINES={
    "gauge":Path("results/corrected_fold_outer_sector_pulses.json"),
    "physical":Path("results/corrected_fold_outer_physical_refinement.json"),
}


def baseline_sector(path,sector):
    payload=json.loads(path.read_text())
    return next(item for item in payload["sectors"] if item["sector"]==sector)


def main():
    baselines={key:baseline_sector(path,key) for key,path in BASELINES.items()}
    geometry=build_geometry("G6")
    print(f"sampling kappa={DAMPING_RATE:g} weak-sector coefficients",flush=True)
    wave_coefficients=sample_coefficients(geometry,constraint_damping=DAMPING_RATE)
    source_coefficients=sample_source_coefficients(geometry,DAMPING_RATE)
    constraint_coefficients=sample_constraint_coefficients(geometry)
    sectors=[]
    for sector in ("gauge","physical"):
        records=[]
        for baseline in baselines[sector]["records"]:
            grid=tuple(baseline["grid_size"])
            print(f"weak {sector} pulse grid {grid[0]} x {grid[1]}",flush=True)
            setup=build_case(
                geometry,wave_coefficients,source_coefficients,constraint_coefficients,
                *grid,pulse_sector=sector,
            )
            result=run_sector_case(setup,"sommerfeld_complete_3+3+1")
            off=baseline["modes"]["off"];point=baseline["modes"]["complete_3+3+1"]
            records.append({
                "grid_size":list(grid),"weak_complete":result,
                "baseline_off":off,"baseline_point_complete":point,
                "final_selected_ratio_to_off":result["final_selected_incoming"]
                    /max(off["final_selected_incoming"],1e-300),
                "postcontact_peak_ratio_to_off":result["postcontact_peak_selected_incoming"]
                    /max(off["postcontact_peak_selected_incoming"],1e-300),
                "final_selected_ratio_to_point_complete":result["final_selected_incoming"]
                    /max(point["final_selected_incoming"],1e-300),
                "outer_constraint_ratio_to_off":result["final_outer_constraint_l2"]
                    /max(off["final_outer_constraint_l2"],1e-300),
            })
        ratios=np.array([item["final_selected_ratio_to_off"] for item in records])
        reductions=1-ratios
        agreement=(
            float(abs(reductions[-1]-reductions[-2])/max(abs(reductions[-1]),1e-300))
            if np.all(reductions>0) else None
        )
        acceptance={
            "all_runs_finite":bool(all(item["weak_complete"]["finite"] for item in records)),
            "initial_selected_incoming_below_1e_10":bool(max(
                item["weak_complete"]["initial_selected_incoming"] for item in records
            )<1e-10),
            "reduces_selected_reflection_on_both_grids":bool(np.all(ratios<1)),
            "reduces_fine_selected_reflection_by_2_percent":bool(ratios[-1]<.98),
            "reduces_fine_postcontact_peak_by_2_percent":bool(
                records[-1]["postcontact_peak_ratio_to_off"]<.98
            ),
            "two_grid_reduction_agrees_within_25_percent":bool(
                agreement is not None and agreement<.25
            ),
            "fine_not_worse_than_point_complete_by_10_percent":bool(
                records[-1]["final_selected_ratio_to_point_complete"]<1.10
            ),
            "fine_outer_constraint_not_increased_by_10_percent":bool(
                records[-1]["outer_constraint_ratio_to_off"]<1.10
            ),
            "fine_corner_below_1_percent_of_pulse_amplitude":bool(
                records[-1]["weak_complete"]["maximum_corner_position"]<.01*AMPLITUDE
            ),
        }
        sectors.append({
            "sector":sector,"status":"pass" if all(acceptance.values()) else "review",
            "records":records,"two_grid_reduction_relative_difference":agreement,
            "acceptance":acceptance,
        })
    payload={
        "status":"pass" if all(item["status"]=="pass" for item in sectors) else "review",
        "scope":"gauge and physical non-regression for the weak homogeneous 3+3+1 outer flux",
        "final_time":FINAL_TIME,"damping_rate":DAMPING_RATE,"sectors":sectors,
        "limitations":[
            "linear corrected-G6 coefficients and small projector-pure metric pulses",
            "gauge uses 41x61 and 49x73 while physical uses 49x73 and 57x85",
            "this numerical non-regression audit is not a continuum energy estimate",
        ],
    }
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "status":payload["status"],"sectors":[{
            "sector":item["sector"],"status":item["status"],
            "ratios":[record["final_selected_ratio_to_off"] for record in item["records"]],
            "agreement":item["two_grid_reduction_relative_difference"],
            "acceptance":item["acceptance"],
        } for item in sectors],
    },indent=2),flush=True)


if __name__=="__main__":main()
