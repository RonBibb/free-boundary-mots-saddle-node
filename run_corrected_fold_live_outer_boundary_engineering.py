#!/usr/bin/env python3
"""Unscored G7 engineering run for the nonlinear outer Sommerfeld row."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from run_corrected_fold_live_nonlinear_gauge_source import integrate,setup_case
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_g7
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import RADIAL_COMPARISON_CUT,relative_norm


OUTPUT=Path("results/corrected_fold_live_outer_boundary_engineering.json")
BASELINE=Path("results/corrected_fold_live_normal_wall_gauge_state.npz")


def main():
    if not BASELINE.exists():
        raise FileNotFoundError("passing live-wall checkpoint is required")
    print("building corrected G7 state",flush=True)
    geometry=build_g7(build_geometry("G6"))
    case=setup_case(
        geometry,"G7-outer-engineering",live_normal_wall_gauge=True,
        live_outer_sommerfeld=True,
    )
    run=integrate(case)
    baseline=np.load(BASELINE)
    keep=case["r"]<=RADIAL_COMPARISON_CUT+1e-12
    inner={
        "position_increment_relative_difference":relative_norm(
            run["_increment"][:,keep],baseline["G7_increment"][:,keep],
        ),
        "velocity_relative_difference":relative_norm(
            run["_velocity"][:,keep],baseline["G7_velocity"][:,keep],
        ),
        "source_relative_difference":relative_norm(
            run["_source"][:,keep],baseline["G7_source"][:,keep],
        ),
    }
    payload={
        "status":"engineering_only",
        "scope":"unscored G7 nonlinear complete-field Sommerfeld outer-boundary non-regression run",
        "initial_outer_diagnostic":case["initial_live_outer_sommerfeld"],
        "maximum_outer_acceleration_residual":run["maximum_outer_acceleration_residual"],
        "maximum_outer_source_residual":run["maximum_outer_source_residual"],
        "final_outer_position_residual":run["final_outer_sommerfeld_position_residual"],
        "final_outer_source_residual":run["final_outer_source_sommerfeld_residual"],
        "inner_comparison_to_passing_live_wall_run":inner,
        "final_constraint":run["final_constraint"],
        "signature":run["signature"],
        "limitations":[
            "engineering result; no acceptance rules were sealed",
            "G7 only",
            "t=0.002 background non-regression rather than an injected-pulse reflection test",
        ],
    }
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps(payload,indent=2),flush=True)


if __name__=="__main__":main()
