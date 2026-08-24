#!/usr/bin/env python3
"""Sealed 48/56-mode confirmation of the A=7.90 final cap pair."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from run_corrected_A790_two_grid_formation_search import (
    blind_scan, endpoint_transfer, static_search,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry


OUTPUT = Path("results/corrected_A790_t004_high_mode_formation_confirmation.json")
STATE = Path("results/corrected_A790_two_grid_formation_search_state.npz")
PROTOCOL = "notes/75_A790_t004_high_mode_formation_confirmation_protocol.md"
AMPLITUDE = 7.90
MODES = (48, 56)


def admitted_diagnostics(record):
    trials = [
        trial for trial in record["trials"]
        if trial["adjacent_mode_pair"] is not None
        and trial["adjacent_mode_pair"]["admitted"]
    ]
    mode_differences = [
        value
        for trial in trials
        for value in (
            trial["adjacent_mode_pair"]["profile_relative_difference"],
            trial["adjacent_mode_pair"]["axis_relative_difference"],
            trial["adjacent_mode_pair"]["brane_relative_difference"],
        )
    ]
    conditions = [
        surface["jacobian_condition_number"]
        for trial in trials for surface in trial["surfaces"]
    ]
    validations = [
        surface["independent_expansion"]["two_cell_interior_maximum"]
        for trial in trials for surface in trial["surfaces"]
    ]
    return {
        "admitted_seed_count": len(trials),
        "maximum_adjacent_mode_difference": float(max(mode_differences, default=0.)),
        "maximum_condition_number": float(max(conditions, default=0.)),
        "maximum_independent_expansion": float(max(validations, default=0.)),
    }


def main():
    if not STATE.exists():
        raise FileNotFoundError("the sealed note-74 final-slice archive is required")
    archive = np.load(STATE)
    print("reconstructing corrected G7/G8 A=7.90 geometries", flush=True)
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": AMPLITUDE}
    g7 = build_refined(
        seed, 81, 121, "G7A790-high-mode", selector_iterations=40,
        slice_iterations=270,
    )
    g8 = build_refined(
        g7, 97, 145, "G8A790-high-mode", selector_iterations=45,
        slice_iterations=280,
    )
    static = {"G7": static_search(g7), "G8": static_search(g8)}
    records = {}
    for label, geometry in (("G7", g7), ("G8", g8)):
        position = (
            np.asarray(geometry["jet_field"].reduced_fields)
            + archive[f"{label}_time_3_increment"]
        )
        velocity = archive[f"{label}_time_3_velocity"]
        records[label] = blind_scan(
            f"{label}-A790_t0.004-high-mode", position, velocity, geometry,
            modes_to_run=MODES,
        )
    transfer = endpoint_transfer(
        records["G7"]["admitted_signatures"],
        records["G8"]["admitted_signatures"],
    )
    diagnostics = {
        label: admitted_diagnostics(record) for label, record in records.items()
    }
    acceptance = {
        "fresh_initial_static_searches_find_zero_caps": bool(all(
            item["accepted_count"] == 0 for item in static.values()
        )),
        "both_final_slices_have_exactly_two_blind_candidates": bool(all(
            record["admitted_distinct_count"] == 2 for record in records.values()
        )),
        "two_grid_endpoint_transfer_below_1_percent": bool(
            transfer is not None and transfer["maximum"] < .01
        ),
        "all_admitted_high_mode_pairs_pass_fixed_detector_rules": bool(all(
            item["admitted_seed_count"] > 0
            and item["maximum_condition_number"] < 1e5
            and item["maximum_adjacent_mode_difference"] < .002
            and item["maximum_independent_expansion"] < .002
            for item in diagnostics.values()
        )),
    }
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "classification": (
            "paired_formation_candidate_confirmed_at_t0.004"
            if all(acceptance.values()) else "unresolved_high_mode_confirmation"
        ),
        "scope": "sealed blind 48/56-mode confirmation of the corrected A=7.90 final marginal-surface pair",
        "protocol": PROTOCOL,
        "source_state": str(STATE),
        "amplitude": AMPLITUDE,
        "time": .004,
        "cosine_modes": list(MODES),
        "initial_static_search": static,
        "dynamic_search": records,
        "cross_grid_endpoint_transfer": transfer,
        "admitted_diagnostics": diagnostics,
        "acceptance": acceptance,
        "limitations": [
            "fresh high-mode confirmation of an archived final slice, not a rerun of note 74",
            "note 74 remains REVIEW under its original 40/48-mode rule",
            "finite twelve-seed star-shaped donor-capped search",
            "does not fully localize formation between t=0 and t=0.001",
            "not event-horizon location, topology change, long-time stability, an open basin, connected bulk geometry, or mass transfer",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "classification": payload["classification"],
        "counts": {
            label: record["admitted_distinct_count"]
            for label, record in records.items()
        },
        "signatures": {
            label: record["admitted_signatures"]
            for label, record in records.items()
        },
        "transfer": transfer,
        "diagnostics": diagnostics,
        "acceptance": acceptance,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
