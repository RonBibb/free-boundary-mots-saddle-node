#!/usr/bin/env python3
"""Matched-radial-spacing refinement of the r=6 Sommerfeld transfer."""

import json
from pathlib import Path

import numpy as np

from run_corrected_fold_boundary_constraint_pulse import (
    AMPLITUDE,DAMPING_RATE,FINAL_TIME,build_case,build_geometry,run_case,
    sample_coefficients,
)
from run_corrected_fold_free_constraint_pulse import sample_constraint_coefficients
from run_corrected_fold_gh_driver_runtime import sample_source_coefficients


OUTPUT=Path("results/corrected_fold_sommerfeld_r6_matched_refinement.json")
BASELINE=Path("results/corrected_fold_weak_boundary_constraint_pulse.json")
SUPPORT=np.array((.125,.25,.5,1.,2.,3.,4.,5.,6.))


def main():
    baseline=json.loads(BASELINE.read_text())["records"][0]
    ratio4=baseline["candidates"]["sommerfeld_complete_3+3+1"]["final_outer_ratio_to_off"]
    geometry=build_geometry("G6")
    print("sampling matched-spacing r=6 coefficients",flush=True)
    wave=sample_coefficients(geometry,constraint_damping=DAMPING_RATE,support_r=SUPPORT)
    source=sample_source_coefficients(geometry,DAMPING_RATE,SUPPORT)
    constraint=sample_constraint_coefficients(geometry,SUPPORT)
    setup=build_case(
        geometry,wave,source,constraint,49,109,r_max=6.,r_center=5.53,
        sommerfeld_penalty=1.,
    )
    print("matched-spacing r=6 uncontrolled",flush=True);off=run_case(setup,"off")
    print("matched-spacing r=6 complete",flush=True)
    complete=run_case(setup,"sommerfeld_complete_3+3+1")
    ratio6=complete["final_outer_band_constraint_l2"]/max(
        off["final_outer_band_constraint_l2"],1e-300,
    )
    reduction4=1-ratio4;reduction6=1-ratio6
    difference=float(abs(reduction6-reduction4)/max(reduction4,reduction6,1e-300))
    coarse=json.loads(Path("results/corrected_fold_sommerfeld_robustness.json").read_text())[
        "outer_location_transfer"
    ]
    coarse_ratio=coarse["r6_ratio_to_off"]
    r6_refinement_difference=float(
        abs((1-ratio6)-(1-coarse_ratio))/max(1-ratio6,1-coarse_ratio,1e-300)
    )
    acceptance={
        "runs_are_finite":bool(off["finite"] and complete["finite"]),
        "matched_r6_reduces_outer_constraint_by_5_percent":bool(ratio6<.95),
        "r6_reduction_refines_within_15_percent":bool(r6_refinement_difference<.15),
        "r4_r6_reduction_agrees_within_20_percent":bool(difference<.20),
        "pulse_reaches_outer_face_before_final_time":bool(
            complete["radial_arrival_lower_bound"]<FINAL_TIME
        ),
        "pulse_avoids_compact_walls_before_final_time":bool(
            complete["compact_arrival_lower_bound"]>FINAL_TIME
        ),
        "fine_corner_below_1_percent_of_pulse_amplitude":bool(
            complete["maximum_corner_position"]<.01*AMPLITUDE
        ),
    }
    payload={
        "status":"pass" if all(acceptance.values()) else "review",
        "scope":"matched-radial-spacing r=6 refinement of the weak homogeneous 3+3+1 Sommerfeld constraint pulse",
        "r4_reference":{"grid":[49,73],"spacing":4/72,"ratio":ratio4,"reduction":reduction4},
        "r6_coarse":{"grid":[49,97],"spacing":6/96,"ratio":coarse_ratio,"reduction":1-coarse_ratio},
        "r6_matched":{"grid":[49,109],"spacing":6/108,"ratio":ratio6,"reduction":reduction6,"off":off,"complete":complete},
        "r6_refinement_reduction_relative_difference":r6_refinement_difference,
        "r4_r6_reduction_relative_difference":difference,"acceptance":acceptance,
        "limitations":[
            "linear corrected-G6 coefficients and a small near-boundary constraint-projector-pure pulse",
            "matching radial spacing does not match the dimensionless radius-dependent regular tensor basis",
            "the final constraint norm is not a covariant reflection coefficient",
            "nonlinear stability remains open",
        ],
    }
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "status":payload["status"],"r4_reduction":reduction4,
        "r6_coarse_reduction":1-coarse_ratio,"r6_matched_reduction":reduction6,
        "r6_refinement_difference":r6_refinement_difference,
        "r4_r6_difference":difference,"acceptance":acceptance,
    },indent=2),flush=True)


if __name__=="__main__":main()
