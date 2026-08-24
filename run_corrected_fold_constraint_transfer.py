#!/usr/bin/env python3
"""Transfer the selected free-constraint pulse across folds and domains."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from run_corrected_fold_free_constraint_pulse import run_case,sample_constraint_coefficients
from run_corrected_fold_regular_so3_runtime import build_geometry,sample_coefficients


DAMPING_RATE=4.
SUPPORT_BY_DOMAIN={
    4.:np.array((.125,.25,.5,1.,2.,3.,4.)),
    6.:np.array((.125,.25,.5,1.,2.,3.,4.,5.,6.)),
}


def zero_source_coefficients(geometry,support_r):
    z=np.geomspace(geometry["z"][0],geometry["z"][-1],5)
    return {
        "z":z,"r":np.r_[0.,support_r],
        "zero":np.zeros((len(z),len(support_r)+1,9,3)),
        "first":np.zeros((3,len(z),len(support_r)+1,9,3)),
        "radial_first_is_scaled":True,"constraint_damping_rate":DAMPING_RATE,
    }


def transfer_case(resolution,r_max,nz,nr):
    print(f"building corrected {resolution} geometry",flush=True)
    geometry=build_geometry(resolution);support=SUPPORT_BY_DOMAIN[float(r_max)]
    print(f"sampling {resolution} damped coefficients through r={r_max:g}",flush=True)
    wave=sample_coefficients(
        geometry,constraint_damping=DAMPING_RATE,support_r=support,
    )
    constraint=sample_constraint_coefficients(geometry,support)
    source=zero_source_coefficients(geometry,support)
    result=run_case(
        geometry,wave,source,constraint,nz,nr,DAMPING_RATE,r_max=r_max,
    )
    return {
        "corrected_fold_resolution":resolution,
        "fold_amplitude":geometry["fold_amplitude"],
        "selector_maximum":geometry["selector_maximum"],
        "coefficient_support_r":[0.,*[float(value) for value in support]],
        "constraint_axis_extrapolation_relative_differences":constraint["axis_extrapolation_relative_differences"],
        **result,
    }


baseline_payload=json.loads(Path("results/corrected_fold_free_constraint_pulse.json").read_text())
baseline=next(
    item for item in baseline_payload["selected_refinement"] if item["grid_size"]==[33,49]
)
cases=[
    {
        "corrected_fold_resolution":"G6","fold_amplitude":None,
        "selector_maximum":None,"coefficient_support_r":[0.,.125,.25,.5,1.,2.,3.,4.],
        **baseline,
        "radial_domain_maximum":4.,
    },
    transfer_case("G5",4.,33,49),
    transfer_case("G6",6.,33,73),
]
baseline_ratio=cases[0]["constraint_energy_ratio"]
for item in cases:
    item["energy_ratio_relative_difference_from_G6_R4"]=float(
        abs(item["constraint_energy_ratio"]-baseline_ratio)/baseline_ratio
    )
fold_difference=cases[1]["energy_ratio_relative_difference_from_G6_R4"]
domain_difference=cases[2]["energy_ratio_relative_difference_from_G6_R4"]
new_cases=cases[1:]
acceptance={
    "all_new_selectors_below_1e_8":all(item["selector_maximum"]<1e-8 for item in new_cases),
    "all_new_runs_stop_before_causal_boundary_arrival":all(item["stopped_before_boundary_arrival"] for item in new_cases),
    "all_new_constraint_energy_ratios_below_0p8":all(item["constraint_energy_ratio"]<.8 for item in new_cases),
    "G5_G6_energy_ratio_difference_below_5_percent":fold_difference<.05,
    "R4_R6_energy_ratio_difference_below_5_percent":domain_difference<.05,
    "all_new_near_boundary_amplitudes_below_0p5_percent":max(item["maximum_boundary_amplitude_fraction"] for item in new_cases)<.005,
    "all_new_near_boundary_constraint_l2_fractions_below_1e_6":max(item["maximum_boundary_constraint_l2_fraction"] for item in new_cases)<1e-6,
    "all_constraint_axis_variants_below_5_percent":max(
        value for item in new_cases
        for value in item["constraint_axis_extrapolation_relative_differences"].values()
    )<.05,
    "source_zero_fixed_point_is_preserved":all(item["source_remains_zero"] for item in new_cases),
}
payload={
    "status":"pass" if all(acceptance.values()) else "review",
    "scope":"selected kappa=4 free GH constraint-pulse transfer across corrected G5/G6 folds and r=4/6 runtime domains",
    "baseline_source":"results/corrected_fold_free_constraint_pulse.json",
    "damping_rate":DAMPING_RATE,"cases":cases,
    "G5_G6_energy_ratio_relative_difference":fold_difference,
    "R4_R6_energy_ratio_relative_difference":domain_difference,
    "acceptance":acceptance,
    "limitations":[
        "one accepted runtime grid spacing for each transfer comparison",
        "linear small h00 constraint pulse and zero source target",
        "G5 and G6 backgrounds share the same corrected physical-shape family",
        "r=6 remains inside the source initial-data domain r=8",
        "shift advection, nonzero driver targets, nonlinear evolution, and horizon tracking remain open",
    ],
}
Path("results/corrected_fold_constraint_transfer.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps({
    "status":payload["status"],
    "cases":[{
        "fold":item["corrected_fold_resolution"],"r_max":item["radial_domain_maximum"],
        "grid":item["grid_size"],"energy_ratio":item["constraint_energy_ratio"],
        "boundary_l2_fraction":item["maximum_boundary_constraint_l2_fraction"],
    } for item in cases],
    "G5_G6_relative_difference":fold_difference,
    "R4_R6_relative_difference":domain_difference,"acceptance":acceptance,
},indent=2))
