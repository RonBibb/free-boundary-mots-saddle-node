#!/usr/bin/env python3
"""Sealed resolution-aware blind confirmation of the A=7.90 final cap pair."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice
from run_corrected_A790_blind_horizon_detector_engineering import (
    SEEDS, public_surface, solve_seed, stable_pair,
)
from run_corrected_A790_two_grid_formation_search import endpoint_transfer
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry


OUTPUT = Path("results/corrected_A790_t004_discrepancy_formation_confirmation.json")
STATE = Path("results/corrected_A790_two_grid_formation_search_state.npz")
PROTOCOL = "notes/76_A790_t004_discrepancy_formation_protocol.md"
AMPLITUDE = 7.90
CANDIDATE_MODES = (24, 28, 32, 36, 40, 48, 56, 64)
CONDITION_LIMIT = 1e5
VALIDATION_LIMIT = .002


def surface_passes(surface):
    return bool(
        "error" not in surface
        and surface["converged"]
        and surface["jacobian_condition_number"] < CONDITION_LIMIT
        and surface.get("independent_expansion") is not None
        and surface["independent_expansion"][
            "two_cell_interior_maximum"
        ] < VALIDATION_LIMIT
    )


def scan_seed(label, position, velocity, geometry, seed, prepared):
    surfaces = []
    selected_index = None
    for modes in CANDIDATE_MODES:
        print(f"{label}, seed={seed:.2f}: solving {modes} modes", flush=True)
        surface = solve_seed(
            position, velocity, geometry, seed, modes, prepared,
        )
        surfaces.append(surface)
        if selected_index is None and surface_passes(surface):
            selected_index = len(surfaces) - 1
        elif selected_index is not None:
            break
    pair = None
    if selected_index is not None and selected_index + 1 < len(surfaces):
        pair = stable_pair(surfaces[selected_index], surfaces[selected_index + 1])
    return {
        "seed": float(seed),
        "selected_index": selected_index,
        "selected_modes": (
            surfaces[selected_index]["cosine_modes"]
            if selected_index is not None else None
        ),
        "confirmation_modes": (
            surfaces[selected_index + 1]["cosine_modes"]
            if selected_index is not None and selected_index + 1 < len(surfaces)
            else None
        ),
        "admitted": bool(pair is not None and pair["admitted"]),
        "adjacent_mode_pair": pair,
        "surfaces": [public_surface(surface) for surface in surfaces],
    }


def cluster_trials(trials):
    clusters = []
    for trial in trials:
        if not trial["admitted"]:
            continue
        pair = trial["adjacent_mode_pair"]
        signature = np.array((
            pair["representative_rho_axis"],
            pair["representative_rho_brane"],
        ))
        destination = None
        for cluster in clusters:
            if np.linalg.norm(signature - np.asarray(cluster["signature"])) < .005:
                destination = cluster
                break
        if destination is None:
            destination = {
                "signature": signature.tolist(),
                "seeds": [],
                "selected_confirmation_modes": [],
            }
            clusters.append(destination)
        destination["seeds"].append(trial["seed"])
        destination["selected_confirmation_modes"].append([
            trial["selected_modes"], trial["confirmation_modes"],
        ])
    return sorted(clusters, key=lambda item: item["signature"][1])


def scan_grid(label, position, velocity, geometry):
    prepared = prepare_capped_expansion_slice(
        position, velocity, geometry["z"], geometry["r"],
    )
    trials = [
        scan_seed(label, position, velocity, geometry, seed, prepared)
        for seed in SEEDS
    ]
    clusters = cluster_trials(trials)
    return {
        "label": label,
        "trial_count": len(trials),
        "admitted_distinct_count": len(clusters),
        "clusters": clusters,
        "admitted_signatures": [item["signature"] for item in clusters],
        "trials": trials,
    }


def admitted_rule_summary(record):
    admitted = [trial for trial in record["trials"] if trial["admitted"]]
    pairs = [trial["adjacent_mode_pair"] for trial in admitted]
    surfaces = []
    for trial in admitted:
        index = trial["selected_index"]
        surfaces.extend(trial["surfaces"][index:index + 2])
    return {
        "admitted_seed_count": len(admitted),
        "maximum_mode_or_endpoint_difference": float(max((
            value for pair in pairs for value in (
                pair["profile_relative_difference"],
                pair["axis_relative_difference"],
                pair["brane_relative_difference"],
            )
        ), default=0.)),
        "maximum_condition_number": float(max((
            surface["jacobian_condition_number"] for surface in surfaces
        ), default=0.)),
        "maximum_independent_expansion": float(max((
            surface["independent_expansion"]["two_cell_interior_maximum"]
            for surface in surfaces
        ), default=0.)),
        "all_confirmations_are_exactly_next_candidate": bool(all(
            CANDIDATE_MODES.index(trial["confirmation_modes"])
            == CANDIDATE_MODES.index(trial["selected_modes"]) + 1
            for trial in admitted
        )),
    }


def main():
    if not STATE.exists():
        raise FileNotFoundError("the sealed note-74 final-slice archive is required")
    archive = np.load(STATE)
    print("reconstructing corrected G7/G8 A=7.90 geometries", flush=True)
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": AMPLITUDE}
    g7 = build_refined(
        seed, 81, 121, "G7A790-discrepancy", selector_iterations=40,
        slice_iterations=270,
    )
    g8 = build_refined(
        g7, 97, 145, "G8A790-discrepancy", selector_iterations=45,
        slice_iterations=280,
    )
    records = {}
    for label, geometry in (("G7", g7), ("G8", g8)):
        position = (
            np.asarray(geometry["jet_field"].reduced_fields)
            + archive[f"{label}_time_3_increment"]
        )
        velocity = archive[f"{label}_time_3_velocity"]
        records[label] = scan_grid(
            f"{label}-A790_t0.004-discrepancy", position, velocity, geometry,
        )
    transfer = endpoint_transfer(
        records["G7"]["admitted_signatures"],
        records["G8"]["admitted_signatures"],
    )
    summaries = {
        label: admitted_rule_summary(record) for label, record in records.items()
    }
    acceptance = {
        "both_grids_have_exactly_two_distinct_candidates": bool(all(
            record["admitted_distinct_count"] == 2 for record in records.values()
        )),
        "all_candidates_pass_discrepancy_selection_and_confirmation": bool(all(
            summary["admitted_seed_count"] > 0
            and summary["all_confirmations_are_exactly_next_candidate"]
            and summary["maximum_condition_number"] < CONDITION_LIMIT
            and summary["maximum_independent_expansion"] < VALIDATION_LIMIT
            and summary["maximum_mode_or_endpoint_difference"] < .002
            for summary in summaries.values()
        )),
        "two_grid_endpoint_transfer_below_1_percent": bool(
            transfer is not None and transfer["maximum"] < .01
        ),
        "each_branch_recovered_from_at_least_two_seeds": bool(all(
            len(cluster["seeds"]) >= 2
            for record in records.values() for cluster in record["clusters"]
        )),
    }
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "classification": (
            "paired_formation_candidate_confirmed_at_t0.004"
            if all(acceptance.values()) else "unresolved_discrepancy_confirmation"
        ),
        "scope": "sealed blind resolution-aware two-grid confirmation of the corrected A=7.90 final marginal-surface pair",
        "protocol": PROTOCOL,
        "source_state": str(STATE),
        "amplitude": AMPLITUDE,
        "time": .004,
        "candidate_modes": list(CANDIDATE_MODES),
        "condition_limit": CONDITION_LIMIT,
        "independent_expansion_limit": VALIDATION_LIMIT,
        "dynamic_search": records,
        "cross_grid_endpoint_transfer": transfer,
        "admitted_rule_summary": summaries,
        "acceptance": acceptance,
        "limitations": [
            "resolution-aware confirmation of archived final slices; notes 74 and 75 remain REVIEW",
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
        "clusters": {
            label: record["clusters"] for label, record in records.items()
        },
        "transfer": transfer,
        "summaries": summaries,
        "acceptance": acceptance,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
