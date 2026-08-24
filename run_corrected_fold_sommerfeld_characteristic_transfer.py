#!/usr/bin/env python3
"""Matched-spacing r=4/6 transfer of constraint-characteristic reflection."""

import json
from pathlib import Path

import numpy as np

from run_corrected_fold_boundary_constraint_pulse import (
    AMPLITUDE,DAMPING_RATE,FINAL_TIME,build_case,build_geometry,sample_coefficients,
)
from run_corrected_fold_free_constraint_pulse import sample_constraint_coefficients
from run_corrected_fold_gh_driver_runtime import sample_source_coefficients
from run_corrected_fold_outer_sector_pulses import run_sector_case


OUTPUT=Path("results/corrected_fold_sommerfeld_characteristic_transfer.json")
SUPPORT=np.array((.125,.25,.5,1.,2.,3.,4.,5.,6.))
DOMAINS=((4.,49,73,3.53),(6.,49,109,5.53))


def main():
    geometry=build_geometry("G6")
    print("sampling r=4/6 characteristic-transfer coefficients",flush=True)
    wave=sample_coefficients(geometry,constraint_damping=DAMPING_RATE,support_r=SUPPORT)
    source=sample_source_coefficients(geometry,DAMPING_RATE,SUPPORT)
    constraint=sample_constraint_coefficients(geometry,SUPPORT)
    records=[]
    for radius,nz,nr,center in DOMAINS:
        print(f"r={radius:g} matched-spacing characteristic baseline",flush=True)
        setup=build_case(
            geometry,wave,source,constraint,nz,nr,r_max=radius,r_center=center,
            pulse_sector="constraint",sommerfeld_penalty=1.,
        )
        off=run_sector_case(setup,"off")
        print(f"r={radius:g} matched-spacing characteristic control",flush=True)
        complete=run_sector_case(setup,"sommerfeld_complete_3+3+1")
        ratio=complete["final_selected_incoming"]/max(
            off["final_selected_incoming"],1e-300,
        )
        peak_ratio=complete["postcontact_peak_selected_incoming"]/max(
            off["postcontact_peak_selected_incoming"],1e-300,
        )
        records.append({
            "radius":radius,"grid_size":[nz,nr],"radial_spacing":radius/(nr-1),
            "off":off,"complete":complete,"final_reflection_ratio":ratio,
            "final_reflection_reduction":1-ratio,
            "postcontact_peak_ratio":peak_ratio,
        })
    reductions=np.array([item["final_reflection_reduction"] for item in records])
    difference=float(abs(reductions[1]-reductions[0])/max(abs(reductions).max(),1e-300))
    acceptance={
        "all_runs_finite":bool(all(
            item[mode]["finite"] for item in records for mode in ("off","complete")
        )),
        "initial_incoming_below_1e_10":bool(max(
            item[mode]["initial_selected_incoming"]
            for item in records for mode in ("off","complete")
        )<1e-10),
        "both_domains_reduce_final_reflection_by_2_percent":bool(all(
            item["final_reflection_ratio"]<.98 for item in records
        )),
        "both_domains_reduce_postcontact_peak_by_2_percent":bool(all(
            item["postcontact_peak_ratio"]<.98 for item in records
        )),
        "r4_r6_reflection_reduction_agrees_within_25_percent":bool(difference<.25),
        "both_domains_avoid_compact_walls":bool(all(
            item["complete"]["compact_arrival_lower_bound"]>FINAL_TIME for item in records
        )),
        "both_domains_reach_outer_face":bool(all(
            item["complete"]["radial_arrival_lower_bound"]<FINAL_TIME for item in records
        )),
        "both_fine_corners_below_1_percent_of_pulse_amplitude":bool(max(
            item["complete"]["maximum_corner_position"] for item in records
        )<.01*AMPLITUDE),
    }
    payload={
        "status":"pass" if all(acceptance.values()) else "review",
        "scope":"matched-spacing transfer of reflected incoming constraint characteristics for the complete weak homogeneous 3+3+1 Sommerfeld boundary",
        "damping_rate":DAMPING_RATE,"final_time":FINAL_TIME,
        "records":records,"r4_r6_reduction_relative_difference":difference,
        "acceptance":acceptance,
        "limitations":[
            "linear corrected-G6 coefficients and a small near-boundary constraint-projector-pure pulse",
            "the same reduced-coordinate seed is projected at each radius",
            "the characteristic norm is a regular-coordinate norm rather than a full covariant symmetrizer norm",
            "nonlinear stability remains open",
        ],
    }
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "status":payload["status"],"records":[{
            "radius":item["radius"],"ratio":item["final_reflection_ratio"],
            "reduction":item["final_reflection_reduction"],
            "peak_ratio":item["postcontact_peak_ratio"],
        } for item in records],"r4_r6_reduction_difference":difference,
        "acceptance":acceptance,
    },indent=2),flush=True)


if __name__=="__main__":main()
