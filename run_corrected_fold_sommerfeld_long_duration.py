#!/usr/bin/env python3
"""Long-duration mixed-sector audit of the corrected-fold outer absorber."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.constraint_ibvp import evaluate_regular_so3_constraint_field
from run_corrected_fold_boundary_constraint_pulse import (
    AMPLITUDE,DAMPING_RATE,build_case,build_geometry,regional_constraint_l2,
    sample_coefficients,
)
from run_corrected_fold_free_constraint_pulse import (
    constraint_energy,sample_constraint_coefficients,
)
from run_corrected_fold_gh_driver_runtime import sample_source_coefficients
from run_corrected_fold_outer_sector_pulses import characteristic_norms


FINAL_TIME=3.
GRIDS=((25,37),(33,49))
OUTPUT=Path("results/corrected_fold_sommerfeld_long_duration.json")


def bulk_principal_energy(wave,q,v):
    flat_q=np.asarray(q).reshape(wave.nodes,wave.field_count)
    flat_v=np.asarray(v).reshape(wave.nodes,wave.field_count)
    return float(.5*np.sum((wave.mass@flat_v)*flat_v)+.5*np.sum((wave.stiffness@flat_q)*flat_q))


def run_long_case(setup,mode,time_step_factor=.026):
    wave=setup["wave"];driver=setup["driver"]
    feedback=None if mode=="off" else setup["complete_sommerfeld_outer"]
    spacing=min(np.min(np.diff(wave.z)),np.min(np.diff(wave.r)))
    requested=time_step_factor*spacing/max(
        setup["maximum_radial_speed"],setup["maximum_compact_speed"],
    )
    estimated=max(1,int(np.ceil(FINAL_TIME/requested)));stride=max(1,estimated//120)
    zz,rr=np.meshgrid(wave.z,wave.r,indexing="ij")
    outer=rr>=setup["outer_band_lower"]

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
        q_norm=wave.l2_norm(q);v_norm=wave.l2_norm(v)
        return {
            "time":float(time),"characteristic":characteristic_norms(setup,q,v,h,time),
            "position_l2":q_norm,"velocity_l2":v_norm,
            "state_l2":float(np.hypot(q_norm,v_norm)),
            "bulk_principal_energy":bulk_principal_energy(wave,q,v),
            "constraint_energy":constraint_energy(wave,constraint,constraint_time)["energy"],
            "outer_constraint_l2":regional_constraint_l2(wave,constraint,outer),
            "source_l2":float(np.linalg.norm(h)),
            "corner_position_maximum":float(max(
                np.max(np.abs(q[0,-1])),np.max(np.abs(q[-1,-1])),
            )),
        }

    result=driver.integrate(
        setup["position"],setup["velocity"],setup["source"],setup["memory"],
        setup["target"],FINAL_TIME,requested,diagnostic=diagnostic,
        diagnostic_stride=stride,metric_boundary_constraint_feedback=feedback,
    )
    records=result["diagnostics"];initial=records[0]
    late=[item for item in records if item["time"]>=.75*FINAL_TIME]
    post=[item for item in records if item["time"]>=setup["radial_arrival_lower_bound"]]
    integrals={sector:float(np.trapezoid(
        [item["characteristic"][sector]**2 for item in post],
        [item["time"] for item in post],
    )) for sector in ("gauge","constraint","physical","total")}
    return {
        "mode":mode,"grid_size":[wave.nz,wave.nr],"steps":result["steps"],
        "time_step":result["time_step"],"final_time":FINAL_TIME,
        "radial_arrival_lower_bound":setup["radial_arrival_lower_bound"],
        "compact_arrival_lower_bound":setup["compact_arrival_lower_bound"],
        "compact_crossing_count_lower_bound":float(
            FINAL_TIME/setup["compact_arrival_lower_bound"]
        ),
        "initial_state_l2":initial["state_l2"],
        "final_state_l2":records[-1]["state_l2"],
        "maximum_state_l2":max(item["state_l2"] for item in records),
        "late_maximum_state_l2":max(item["state_l2"] for item in late),
        "initial_bulk_principal_energy":initial["bulk_principal_energy"],
        "final_bulk_principal_energy":records[-1]["bulk_principal_energy"],
        "maximum_bulk_principal_energy":max(item["bulk_principal_energy"] for item in records),
        "late_maximum_bulk_principal_energy":max(item["bulk_principal_energy"] for item in late),
        "initial_constraint_energy":initial["constraint_energy"],
        "final_constraint_energy":records[-1]["constraint_energy"],
        "maximum_constraint_energy":max(item["constraint_energy"] for item in records),
        "late_maximum_constraint_energy":max(item["constraint_energy"] for item in late),
        "integrated_postcontact_characteristic_squared":integrals,
        "maximum_corner_position":max(item["corner_position_maximum"] for item in records),
        "maximum_source_l2":max(item["source_l2"] for item in records),
        "finite":bool(all(all(np.isfinite(value) for value in (
            item["state_l2"],item["bulk_principal_energy"],item["constraint_energy"],
            item["characteristic"]["total"],
        )) for item in records)),
        "diagnostics":records,
    }


def main():
    geometry=build_geometry("G6")
    print(f"sampling kappa={DAMPING_RATE:g} long-duration coefficients",flush=True)
    wave_coefficients=sample_coefficients(geometry,constraint_damping=DAMPING_RATE)
    source_coefficients=sample_source_coefficients(geometry,DAMPING_RATE)
    constraint_coefficients=sample_constraint_coefficients(geometry)
    records=[]
    for grid in GRIDS:
        print(f"mixed long-duration grid {grid[0]} x {grid[1]}",flush=True)
        setup=build_case(
            geometry,wave_coefficients,source_coefficients,constraint_coefficients,
            *grid,pulse_sector="mixed",r_center=3.45,r_half_width=.20,
        )
        print("  uncontrolled",flush=True);off=run_long_case(setup,"off")
        print("  complete Sommerfeld",flush=True);complete=run_long_case(setup,"complete")
        ratios={
            "final_state":complete["final_state_l2"]/max(off["final_state_l2"],1e-300),
            "late_maximum_state":complete["late_maximum_state_l2"]
                /max(off["late_maximum_state_l2"],1e-300),
            "final_bulk_principal_energy":complete["final_bulk_principal_energy"]
                /max(off["final_bulk_principal_energy"],1e-300),
            "late_maximum_constraint_energy":complete["late_maximum_constraint_energy"]
                /max(off["late_maximum_constraint_energy"],1e-300),
            **{f"integrated_{sector}_characteristic":
                complete["integrated_postcontact_characteristic_squared"][sector]
                /max(off["integrated_postcontact_characteristic_squared"][sector],1e-300)
                for sector in ("gauge","constraint","physical","total")},
        }
        records.append({"grid_size":list(grid),"off":off,"complete":complete,"ratios":ratios})
    fine=records[-1];controlled_growth=np.array([
        item["complete"]["late_maximum_state_l2"]
        /max(item["complete"]["initial_state_l2"],1e-300) for item in records
    ])
    sector_ratios={sector:np.array([
        item["ratios"][f"integrated_{sector}_characteristic"] for item in records
    ]) for sector in ("gauge","constraint","physical")}
    acceptance={
        "all_runs_finite":bool(all(item[mode]["finite"] for item in records for mode in ("off","complete"))),
        "at_least_four_compact_crossing_times":bool(min(
            item["complete"]["compact_crossing_count_lower_bound"] for item in records
        )>4),
        "controlled_state_stays_below_10_times_initial":bool(max(controlled_growth)<10),
        "controlled_fine_late_state_below_80_percent_of_off":bool(
            fine["ratios"]["late_maximum_state"]<.8
        ),
        "controlled_fine_final_energy_below_80_percent_of_off":bool(
            fine["ratios"]["final_bulk_principal_energy"]<.8
        ),
        "controlled_fine_late_constraint_energy_not_above_off":bool(
            fine["ratios"]["late_maximum_constraint_energy"]<1.
        ),
        "all_fine_integrated_sector_reflections_reduced_by_20_percent":bool(all(
            fine["ratios"][f"integrated_{sector}_characteristic"]<.8
            for sector in ("gauge","constraint","physical")
        )),
        "sector_reflection_ratios_agree_between_grids_within_25_percent":bool(all(
            abs(values[-1]-values[-2])/max(abs(values[-1]),abs(values[-2]),1e-300)<.25
            for values in sector_ratios.values()
        )),
        "fine_corner_below_5_percent_of_pulse_amplitude":bool(
            fine["complete"]["maximum_corner_position"]<.05*AMPLITUDE
        ),
        "source_driver_zero_fixed_point_preserved":bool(max(
            item[mode]["maximum_source_l2"] for item in records for mode in ("off","complete")
        )<1e-13),
    }
    payload={
        "status":"pass" if all(acceptance.values()) else "review",
        "scope":"long-duration mixed gauge-constraint-physical pulse in the full corrected-G6 variable-coefficient linear wave/driver runtime",
        "damping_rate":DAMPING_RATE,"final_time":FINAL_TIME,
        "grids":[list(grid) for grid in GRIDS],"records":records,
        "controlled_late_state_growth_from_initial":controlled_growth.tolist(),
        "acceptance":acceptance,
        "limitations":[
            "linear corrected-G6 coefficients and a small mixed projector pulse",
            "the positive bulk principal energy omits lower-order and compact-wall boundary-energy corrections",
            "the run crosses compact causal times but does not model nonlinear horizon formation",
            "the scalar wave fields begin at zero and the outer Sommerfeld feedback acts on the seven metric fields",
        ],
    }
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "status":payload["status"],"summary":[{
            "grid":item["grid_size"],"ratios":item["ratios"],
            "controlled_growth":item["complete"]["late_maximum_state_l2"]
                /max(item["complete"]["initial_state_l2"],1e-300),
            "crossings":item["complete"]["compact_crossing_count_lower_bound"],
        } for item in records],"acceptance":acceptance,
    },indent=2),flush=True)


if __name__=="__main__":main()
