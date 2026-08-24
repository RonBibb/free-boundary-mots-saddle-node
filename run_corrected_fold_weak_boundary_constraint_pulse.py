#!/usr/bin/env python3
"""Compare weak E27 and homogeneous constraint-sector outer fluxes."""

import json
from pathlib import Path

import numpy as np

from run_corrected_fold_boundary_constraint_pulse import (
    DAMPING_RATE,FINAL_TIME,GRIDS,build_case,build_geometry,run_case,
    sample_coefficients,
)
from run_corrected_fold_free_constraint_pulse import sample_constraint_coefficients
from run_corrected_fold_gh_driver_runtime import sample_source_coefficients


OUTPUT=Path("results/corrected_fold_weak_boundary_constraint_pulse.json")
BASELINE=Path("results/corrected_fold_boundary_constraint_pulse.json")
MODES=(
    "weak_constraint_only","weak_complete_3+3+1",
    "sommerfeld_constraint_only","sommerfeld_complete_3+3+1",
)


def relative_difference(values):
    reductions=1-np.asarray(values,dtype=float)
    if np.any(reductions<=0):return None
    return float(abs(reductions[-1]-reductions[-2])/max(abs(reductions[-1]),1e-300))


def main():
    baseline=json.loads(BASELINE.read_text())
    off_by_grid={tuple(item["grid_size"]):item["feedback_off"] for item in baseline["grid_pairs"]}
    point_by_grid={tuple(item["grid_size"]):item["complete_characteristic"] for item in baseline["grid_pairs"]}
    geometry=build_geometry("G6")
    print(f"sampling kappa={DAMPING_RATE:g} weak-boundary coefficients",flush=True)
    wave_coefficients=sample_coefficients(geometry,constraint_damping=DAMPING_RATE)
    source_coefficients=sample_source_coefficients(geometry,DAMPING_RATE)
    constraint_coefficients=sample_constraint_coefficients(geometry)
    records=[]
    for grid in GRIDS:
        key=tuple(grid)
        if key not in off_by_grid:raise RuntimeError("baseline grid is missing")
        print(f"weak boundary pulse grid {grid[0]} x {grid[1]}",flush=True)
        setup=build_case(
            geometry,wave_coefficients,source_coefficients,constraint_coefficients,*grid,
        )
        candidates={}
        for mode in MODES:
            print(f"  {mode}",flush=True)
            result=run_case(setup,mode)
            result["final_outer_ratio_to_off"]=(
                result["final_outer_band_constraint_l2"]
                /max(off_by_grid[key]["final_outer_band_constraint_l2"],1e-300)
            )
            result["final_outer_ratio_to_point_complete"]=(
                result["final_outer_band_constraint_l2"]
                /max(point_by_grid[key]["final_outer_band_constraint_l2"],1e-300)
            )
            candidates[mode]=result
        records.append({
            "grid_size":list(grid),"baseline_off":off_by_grid[key],
            "baseline_point_complete":point_by_grid[key],"candidates":candidates,
        })
    ratios={mode:[
        item["candidates"][mode]["final_outer_ratio_to_off"] for item in records
    ] for mode in MODES}
    agreements={mode:relative_difference(value) for mode,value in ratios.items()}
    candidate_gates={}
    for mode in MODES:
        candidate_gates[mode]={
            "all_runs_finite":bool(all(item["candidates"][mode]["finite"] for item in records)),
            "reduces_outer_constraint_on_both_grids":bool(all(value<1 for value in ratios[mode])),
            "reduces_fine_outer_constraint_by_5_percent":bool(ratios[mode][-1]<.95),
            "two_grid_reduction_agrees_within_20_percent":bool(
                agreements[mode] is not None and agreements[mode]<.20
            ),
            "outperforms_point_complete_on_fine_grid":bool(
                records[-1]["candidates"][mode]["final_outer_ratio_to_point_complete"]<1
            ),
        }
    complete_modes=("weak_complete_3+3+1","sommerfeld_complete_3+3+1")
    passed=[mode for mode in complete_modes if all(candidate_gates[mode].values())]
    payload={
        "status":"pass" if passed else "review",
        "scope":"weak outer-face constraint-sector flux on the corrected-G6 linear runtime",
        "final_time":FINAL_TIME,"damping_rate":DAMPING_RATE,
        "penalty":1.,"mass_lift":"diagonal Q1 lumped norm",
        "candidate_policies":{
            "weak_constraint_only":"E27 constraint lift as an energy-scaled weak face flux",
            "weak_complete_3+3+1":"E27 constraint weak flux plus gauge/physical Sommerfeld flux",
            "sommerfeld_constraint_only":"homogeneous Sommerfeld flux on the constraint projector",
            "sommerfeld_complete_3+3+1":"homogeneous Sommerfeld flux on all 3+3+1 projectors",
        },
        "records":records,"ratios_to_off":ratios,
        "two_grid_reduction_relative_difference":agreements,
        "candidate_acceptance":candidate_gates,"passing_complete_candidates":passed,
        "limitations":[
            "linear corrected-G6 coefficients and a small projector-pure metric pulse",
            "the homogeneous constraint-Sommerfeld candidate is exploratory and is not labeled constraint preserving",
            "a numerical weak-flux pass would not by itself prove continuum well posedness",
            "gauge and physical sector regression is required after any constraint-sector pass",
        ],
    }
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "status":payload["status"],"ratios_to_off":ratios,
        "two_grid_reduction_relative_difference":agreements,
        "passing_complete_candidates":passed,
    },indent=2),flush=True)


if __name__=="__main__":main()
