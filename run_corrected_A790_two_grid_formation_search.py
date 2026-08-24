#!/usr/bin/env python3
"""Sealed two-grid short-time formation search from horizonless A=7.90 data."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import run_corrected_fold_live_nonlinear_gauge_source as live
from bhps.anisotropic_capped_surface import find_anisotropic_donor_capped_surfaces
from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice
from run_corrected_A790_blind_horizon_detector_engineering import (
    MODES, SEEDS, deduplicate, public_surface, solve_seed, stable_pair,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import (
    RADIAL_COMPARISON_CUT, interpolate_fields, relative_norm,
)


OUTPUT = Path("results/corrected_A790_two_grid_formation_search.json")
CHECKPOINT = Path("results/corrected_A790_two_grid_formation_search_state.npz")
PROTOCOL = "notes/74_A790_two_grid_formation_protocol.md"
AMPLITUDE = 7.90
FINAL_TIME = .004
STEPS = 8
CHECKPOINT_STEPS = (2, 4, 6)
TIMES = (.001, .002, .003, .004)


def static_search(geometry):
    result = find_anisotropic_donor_capped_surfaces(
        geometry["z"], geometry["r"], geometry["psi"], geometry["a"],
        geometry["b"], geometry["c"], guesses=SEEDS, tolerance=2e-5,
    )
    return {
        "trial_count": result["trial_count"],
        "successful_trials": result["successful_trials"],
        "in_domain_successful_trials": result["in_domain_successful_trials"],
        "accepted_count": len(result["accepted"]),
        "accepted_signatures": [
            [item["rho_axis"], item["rho_brane"]] for item in result["accepted"]
        ],
    }


def blind_scan(label, position, velocity, geometry, modes_to_run=MODES):
    prepared = prepare_capped_expansion_slice(
        position, velocity, geometry["z"], geometry["r"],
    )
    trials = []
    pairs = []
    for seed in SEEDS:
        surfaces = []
        for modes in modes_to_run:
            print(f"{label}, seed={seed:.2f}: solving {modes} modes", flush=True)
            surfaces.append(solve_seed(
                position, velocity, geometry, seed, modes, prepared,
            ))
        pair = stable_pair(*surfaces)
        pairs.append(pair)
        trials.append({
            "seed": float(seed),
            "surfaces": [public_surface(surface) for surface in surfaces],
            "adjacent_mode_pair": pair,
        })
    accepted = deduplicate(pairs)
    signatures = sorted(
        ([item["representative_rho_axis"], item["representative_rho_brane"]]
         for item in accepted),
        key=lambda value: value[1],
    )
    return {
        "label": label,
        "trial_count": len(trials),
        "admitted_distinct_count": len(signatures),
        "admitted_signatures": signatures,
        "trials": trials,
    }


def field_transfer(coarse_case, coarse, fine_case, fine, key):
    mask = coarse_case["r"] <= RADIAL_COMPARISON_CUT + 1e-12
    fine_on = interpolate_fields(
        fine[key], fine_case["z"], fine_case["r"],
        coarse_case["z"], coarse_case["r"][mask],
    )
    return relative_norm(coarse[key][:, mask], fine_on)


def endpoint_transfer(left, right):
    if len(left) != len(right):
        return None
    values = []
    pairs = []
    for coarse, fine in zip(left, right):
        axis = abs(coarse[0] - fine[0]) / max(abs(coarse[0]), abs(fine[0]), 1e-300)
        brane = abs(coarse[1] - fine[1]) / max(abs(coarse[1]), abs(fine[1]), 1e-300)
        values.extend((axis, brane))
        pairs.append({"axis": float(axis), "brane": float(brane)})
    return {"pairs": pairs, "maximum": float(max(values, default=0.))}


def evolution_pass(run):
    return bool(
        run["all_stages_finite"]
        and run["signature"]["all_points_one_negative_direction"]
        and run["final_constraint"]["global_relative"] < .005
        and max(
            run["final_wall"]["maximum"],
            run["final_normal_wall_position_residual"]["maximum"],
        ) < .0005
        and max(
            run["maximum_normal_wall_acceleration_residual"],
            run["maximum_outer_acceleration_residual"],
            run["maximum_outer_source_residual"],
            run["final_outer_sommerfeld_position_residual"]["maximum_normalized"],
            run["final_outer_source_sommerfeld_residual"]["maximum_normalized"],
        ) < 1e-10
        and max(
            run["maximum_outer_metric_correction"],
            run["maximum_outer_scalar_correction"],
            run["maximum_outer_source_correction"],
        ) < .05
    )


def classify(counts7, counts8, transfers):
    if counts7 == counts8 == [0, 0, 0, 0]:
        return "no_detected_formation"
    if counts7 != counts8:
        return "unresolved"
    positive = [index for index, count in enumerate(counts7) if count > 0]
    if not positive:
        return "unresolved"
    first = positive[0]
    paired_history = bool(
        all(count == 0 for count in counts7[:first])
        and all(count == 2 for count in counts7[first:])
    )
    transfer_pass = bool(all(
        item is not None and item["maximum"] < .01
        for item in transfers[first:]
    ))
    return "paired_formation_candidate" if paired_history and transfer_pass else "unresolved"


def main():
    print("building corrected G7/G8 A=7.90 initial slices", flush=True)
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": AMPLITUDE}
    g7_geometry = build_refined(
        seed, 81, 121, "G7A790-formation", selector_iterations=40,
        slice_iterations=270,
    )
    g8_geometry = build_refined(
        g7_geometry, 97, 145, "G8A790-formation", selector_iterations=45,
        slice_iterations=280,
    )
    initial_static = {
        "G7": static_search(g7_geometry),
        "G8": static_search(g8_geometry),
    }

    g7 = live.setup_case(
        g7_geometry, "G7-A790", live_normal_wall_gauge=True,
        live_outer_sommerfeld=True,
    )
    g8 = live.setup_case(
        g8_geometry, "G8-A790", live_normal_wall_gauge=True,
        live_outer_sommerfeld=True,
    )
    live.FINAL_TIME = FINAL_TIME
    live.STEPS = STEPS
    print("evolving G7 A=7.90 with four formation checkpoints", flush=True)
    g7_run = live.integrate(g7, checkpoint_steps=CHECKPOINT_STEPS)
    print("evolving G8 A=7.90 with four formation checkpoints", flush=True)
    g8_run = live.integrate(g8, checkpoint_steps=CHECKPOINT_STEPS)

    records = {}
    state_archive = {}
    for label, geometry, run in (
        ("G7", g7_geometry, g7_run), ("G8", g8_geometry, g8_run),
    ):
        states = [run["_checkpoints"][step] for step in CHECKPOINT_STEPS] + [run]
        records[label] = []
        for index, (time, state) in enumerate(zip(TIMES, states)):
            records[label].append(blind_scan(
                f"{label}-A790_t{time:.3f}", state["_position"],
                state["_velocity"], geometry,
            ))
            state_archive[f"{label}_time_{index}_increment"] = state["_position"] - g7["initial"] if label == "G7" else state["_position"] - g8["initial"]
            state_archive[f"{label}_time_{index}_velocity"] = state["_velocity"]

    counts7 = [record["admitted_distinct_count"] for record in records["G7"]]
    counts8 = [record["admitted_distinct_count"] for record in records["G8"]]
    transfers = [
        endpoint_transfer(left["admitted_signatures"], right["admitted_signatures"])
        for left, right in zip(records["G7"], records["G8"])
    ]
    classification = classify(counts7, counts8, transfers)
    final_transfer = {
        "position_increment": field_transfer(g7, g7_run, g8, g8_run, "_increment"),
        "velocity": field_transfer(g7, g7_run, g8, g8_run, "_velocity"),
        "source_increment": field_transfer(
            g7, g7_run, g8, g8_run, "_source_increment",
        ),
    }
    acceptance = {
        "both_initial_static_searches_find_zero_caps": bool(all(
            item["accepted_count"] == 0 for item in initial_static.values()
        )),
        "both_evolutions_pass": bool(evolution_pass(g7_run) and evolution_pass(g8_run)),
        "final_field_transfer_below_5_percent": bool(
            max(final_transfer.values()) < .05
        ),
        "detector_outcome_is_prospectively_decisive": bool(
            classification != "unresolved"
        ),
    }
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "classification": classification,
        "scope": "sealed two-grid short-time blind formation search from corrected horizonless A=7.90 initial data",
        "protocol": PROTOCOL,
        "amplitude": AMPLITUDE,
        "times": list(TIMES),
        "time_step": FINAL_TIME / STEPS,
        "initial_static_search": initial_static,
        "dynamic_search": records,
        "count_histories": {"G7": counts7, "G8": counts8},
        "cross_grid_endpoint_transfer": transfers,
        "final_field_transfer": final_transfer,
        "final_diagnostics": {
            label: {
                "global_GH_constraint": run["final_constraint"]["global_relative"],
                "wall_position_residual": run["final_wall"]["maximum"],
                "normal_wall_position_residual": run[
                    "final_normal_wall_position_residual"
                ]["maximum"],
                "maximum_outer_metric_correction": run[
                    "maximum_outer_metric_correction"
                ],
                "maximum_outer_scalar_correction": run[
                    "maximum_outer_scalar_correction"
                ],
                "maximum_outer_source_correction": run[
                    "maximum_outer_source_correction"
                ],
            }
            for label, run in (("G7", g7_run), ("G8", g8_run))
        },
        "acceptance": acceptance,
        "limitations": [
            "finite twelve-seed search in the star-shaped donor-capped class",
            "four positive-time checkpoints through t=0.004",
            "a no-detection result is not a proof of global nonexistence",
            "not event-horizon location, topology change, long-time stability, an open basin, connected bulk geometry, or mass transfer",
        ],
    }
    np.savez_compressed(
        CHECKPOINT, G7_z=g7_geometry["z"], G7_r=g7_geometry["r"],
        G8_z=g8_geometry["z"], G8_r=g8_geometry["r"], times=np.asarray(TIMES),
        **state_archive,
    )
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "classification": classification,
        "counts": payload["count_histories"],
        "acceptance": acceptance,
        "field_transfer": final_transfer,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
