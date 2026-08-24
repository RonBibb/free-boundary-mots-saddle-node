#!/usr/bin/env python3
"""Penalty-neighborhood and r=6 transfer controls for the weak absorber."""

import json
from pathlib import Path

import numpy as np

from run_corrected_fold_boundary_constraint_pulse import (
    AMPLITUDE,DAMPING_RATE,FINAL_TIME,build_case,build_geometry,run_case,
    sample_coefficients,
)
from run_corrected_fold_free_constraint_pulse import sample_constraint_coefficients
from run_corrected_fold_gh_driver_runtime import sample_source_coefficients


OUTPUT=Path("results/corrected_fold_sommerfeld_robustness.json")
BASELINE=Path("results/corrected_fold_weak_boundary_constraint_pulse.json")
SUPPORT_R4=np.array((.125,.25,.5,1.,2.,3.,4.))
SUPPORT_R6=np.array((.125,.25,.5,1.,2.,3.,4.,5.,6.))


def coefficients(geometry,support):
    return (
        sample_coefficients(geometry,constraint_damping=DAMPING_RATE,support_r=support),
        sample_source_coefficients(geometry,DAMPING_RATE,support),
        sample_constraint_coefficients(geometry,support),
    )


def main():
    stored=json.loads(BASELINE.read_text());base_record=stored["records"][0]
    baseline_off=base_record["baseline_off"]
    baseline_unit=base_record["candidates"]["sommerfeld_complete_3+3+1"]
    baseline_ratio=baseline_unit["final_outer_ratio_to_off"]
    geometry=build_geometry("G6")

    print("sampling r=4 penalty-neighborhood coefficients",flush=True)
    wave4,source4,constraint4=coefficients(geometry,SUPPORT_R4)
    penalty_records=[]
    for penalty in (.5,2.):
        print(f"r=4 penalty {penalty:g}",flush=True)
        setup=build_case(
            geometry,wave4,source4,constraint4,49,73,
            sommerfeld_penalty=penalty,
        )
        result=run_case(setup,"sommerfeld_complete_3+3+1")
        ratio=result["final_outer_band_constraint_l2"]/max(
            baseline_off["final_outer_band_constraint_l2"],1e-300,
        )
        penalty_records.append({
            "penalty":penalty,"result":result,"ratio_to_off":ratio,
            "reduction":1-ratio,
            "normal_incidence_scalar_reflection_magnitude":abs((1-penalty)/(1+penalty)),
        })
    penalty_records.insert(1,{
        "penalty":1.,"result":baseline_unit,"ratio_to_off":baseline_ratio,
        "reduction":1-baseline_ratio,
        "normal_incidence_scalar_reflection_magnitude":0.,
        "source":"results/corrected_fold_weak_boundary_constraint_pulse.json",
    })

    print("sampling r=6 transfer coefficients",flush=True)
    wave6,source6,constraint6=coefficients(geometry,SUPPORT_R6)
    setup6=build_case(
        geometry,wave6,source6,constraint6,49,97,r_max=6.,r_center=5.53,
        sommerfeld_penalty=1.,
    )
    print("r=6 uncontrolled pulse",flush=True)
    off6=run_case(setup6,"off")
    print("r=6 complete Sommerfeld pulse",flush=True)
    complete6=run_case(setup6,"sommerfeld_complete_3+3+1")
    ratio6=complete6["final_outer_band_constraint_l2"]/max(
        off6["final_outer_band_constraint_l2"],1e-300,
    )
    reduction4=1-baseline_ratio;reduction6=1-ratio6
    transfer_difference=float(
        abs(reduction6-reduction4)/max(abs(reduction4),abs(reduction6),1e-300)
    )
    acceptance={
        "all_penalty_runs_finite":bool(all(item["result"]["finite"] for item in penalty_records)),
        "all_positive_penalties_reduce_r4_outer_constraint_by_5_percent":bool(all(
            item["ratio_to_off"]<.95 for item in penalty_records
        )),
        "unit_penalty_has_smallest_r4_outer_constraint":bool(
            baseline_ratio<=min(item["ratio_to_off"] for item in penalty_records)+1e-12
        ),
        "r6_runs_are_finite":bool(off6["finite"] and complete6["finite"]),
        "r6_reduces_outer_constraint_by_5_percent":bool(ratio6<.95),
        "r4_r6_reduction_agrees_within_20_percent":bool(transfer_difference<.20),
        "r6_pulse_reaches_outer_face_before_final_time":bool(
            complete6["radial_arrival_lower_bound"]<FINAL_TIME
        ),
        "r6_pulse_avoids_compact_walls_before_final_time":bool(
            complete6["compact_arrival_lower_bound"]>FINAL_TIME
        ),
        "r6_fine_corner_below_1_percent_of_pulse_amplitude":bool(
            complete6["maximum_corner_position"]<.01*AMPLITUDE
        ),
    }
    payload={
        "status":"pass" if all(acceptance.values()) else "review",
        "scope":"penalty-neighborhood and outer-location transfer for the complete weak homogeneous regular 3+3+1 Sommerfeld boundary",
        "damping_rate":DAMPING_RATE,"final_time":FINAL_TIME,
        "penalty_records":penalty_records,
        "outer_location_transfer":{
            "r4_grid":[49,73],"r4_ratio_to_off":baseline_ratio,
            "r4_reduction":reduction4,
            "r6_grid":[49,97],"r6_ratio_to_off":ratio6,
            "r6_reduction":reduction6,
            "reduction_relative_difference":transfer_difference,
            "r6_off":off6,"r6_complete":complete6,
        },
        "acceptance":acceptance,
        "limitations":[
            "linear corrected-G6 coefficients and a small constraint-projector-pure pulse",
            "penalty neighborhood is checked at one production grid because the unit penalty already has two-grid refinement",
            "r4 and r6 use the same z grid but radial spacings differ by 12.5 percent",
            "the pulse remains close to each artificial face rather than propagating from a common physical radius",
            "nonlinear stability remains open",
        ],
    }
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "status":payload["status"],
        "penalties":[{
            "penalty":item["penalty"],"ratio":item["ratio_to_off"],
            "reduction":item["reduction"],
        } for item in penalty_records],
        "r6_ratio":ratio6,"r6_reduction":reduction6,
        "r4_r6_reduction_relative_difference":transfer_difference,
        "acceptance":acceptance,
    },indent=2),flush=True)


if __name__=="__main__":main()
