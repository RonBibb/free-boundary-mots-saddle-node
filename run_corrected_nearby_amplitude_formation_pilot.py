#!/usr/bin/env python3
"""Sealed two-grid formation pilot at the nearby amplitudes A=7.88 and 7.92."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import run_corrected_fold_live_nonlinear_gauge_source as live
from run_corrected_A790_formation_time_refinement import (
    median_signatures,
    public_diagnostics,
)
from run_corrected_A790_t004_discrepancy_formation_confirmation import (
    admitted_rule_summary,
    scan_grid,
)
from run_corrected_A790_two_grid_formation_search import (
    blind_scan,
    endpoint_transfer,
    evolution_pass,
    field_transfer,
    static_search,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry


PROTOCOL = "notes/78_nearby_amplitude_formation_pilot_protocol.md"
AMPLITUDES = (7.88, 7.92)
FINAL_TIME = .004
STEPS = 32
SAMPLE_STEPS = tuple(range(1, 9)) + tuple(range(12, 33, 4))
FIXED_MODES = (40, 48)


def tag(amplitude):
    return f"A{int(round(100 * amplitude))}"


def output_path(amplitude):
    return Path(f"results/corrected_{tag(amplitude)}_formation_pilot.json")


def state_path(amplitude):
    return Path(f"results/corrected_{tag(amplitude)}_formation_pilot_state.npz")


def sampled_state(run, step):
    return run if step == STEPS else run["_checkpoints"][step]


def detector_rules_pass(record):
    summary = admitted_rule_summary(record)
    return bool(
        record["admitted_distinct_count"] in (0, 2)
        and (
            record["admitted_distinct_count"] == 0
            or (
                len(record["clusters"]) == 2
                and all(len(cluster["seeds"]) >= 2 for cluster in record["clusters"])
                and summary["all_confirmations_are_exactly_next_candidate"]
                and summary["maximum_condition_number"] < 1e5
                and summary["maximum_independent_expansion"] < .002
                and summary["maximum_mode_or_endpoint_difference"] < .002
            )
        )
    )


def run_amplitude(fold, amplitude):
    started = time.perf_counter()
    amplitude_tag = tag(amplitude)
    print(f"building corrected G7/G8 {amplitude_tag} pilot geometries", flush=True)
    seed = {**fold, "fold_amplitude": amplitude}
    geometries = {
        "G7": build_refined(
            seed, 81, 121, f"G7{amplitude_tag}-formation-pilot",
            selector_iterations=40, slice_iterations=270,
        ),
    }
    geometries["G8"] = build_refined(
        geometries["G7"], 97, 145, f"G8{amplitude_tag}-formation-pilot",
        selector_iterations=45, slice_iterations=280,
    )
    initial_static = {
        label: static_search(geometry) for label, geometry in geometries.items()
    }
    cases = {
        label: live.setup_case(
            geometry, f"{label}-{amplitude_tag}-formation-pilot",
            live_normal_wall_gauge=True, live_outer_sommerfeld=True,
        )
        for label, geometry in geometries.items()
    }
    live.FINAL_TIME = FINAL_TIME
    live.STEPS = STEPS
    checkpoint_steps = SAMPLE_STEPS[:-1]
    runs = {}
    for label, case in cases.items():
        print(
            f"evolving {label} {amplitude_tag} to t={FINAL_TIME:.3f} "
            f"with {STEPS} steps",
            flush=True,
        )
        runs[label] = live.integrate(case, checkpoint_steps=checkpoint_steps)

    times = [step * FINAL_TIME / STEPS for step in SAMPLE_STEPS]
    screens = []
    trigger_index = None
    for index, (step, sample_time) in enumerate(zip(SAMPLE_STEPS, times)):
        grid_screens = {}
        for label in geometries:
            state = sampled_state(runs[label], step)
            grid_screens[label] = blind_scan(
                f"{label}-{amplitude_tag}_t{sample_time:.6f}-screen",
                state["_position"], state["_velocity"], geometries[label],
                modes_to_run=FIXED_MODES,
            )
        screens.append({
            "step": step,
            "time": sample_time,
            "counts": {
                label: record["admitted_distinct_count"]
                for label, record in grid_screens.items()
            },
            "records": grid_screens,
        })
        if any(
            record["admitted_distinct_count"] > 0
            for record in grid_screens.values()
        ):
            trigger_index = index
            break

    if trigger_index is None:
        full_indices = [len(SAMPLE_STEPS) - 1]
    else:
        full_indices = sorted(set((
            trigger_index,
            min(trigger_index + 1, len(SAMPLE_STEPS) - 1),
            len(SAMPLE_STEPS) - 1,
        )))

    full_scans = []
    for index in full_indices:
        step = SAMPLE_STEPS[index]
        sample_time = times[index]
        grid_records = {}
        for label in geometries:
            state = sampled_state(runs[label], step)
            grid_records[label] = scan_grid(
                f"{label}-{amplitude_tag}_t{sample_time:.6f}-discrepancy",
                state["_position"], state["_velocity"], geometries[label],
            )
        signatures = {
            label: median_signatures(record)
            for label, record in grid_records.items()
        }
        full_scans.append({
            "step": step,
            "time": sample_time,
            "counts": {
                label: record["admitted_distinct_count"]
                for label, record in grid_records.items()
            },
            "median_signatures": signatures,
            "cross_grid_endpoint_transfer": endpoint_transfer(
                signatures["G7"], signatures["G8"],
            ),
            "rule_summaries": {
                label: admitted_rule_summary(record)
                for label, record in grid_records.items()
            },
            "records": grid_records,
        })

    final_field_transfer = {
        name: field_transfer(cases["G7"], runs["G7"], cases["G8"], runs["G8"], key)
        for name, key in (
            ("position_increment", "_increment"),
            ("velocity", "_velocity"),
            ("source_increment", "_source_increment"),
        )
    }
    numerical_acceptance = {
        "both_initial_static_searches_find_zero_caps": bool(all(
            item["accepted_count"] == 0 for item in initial_static.values()
        )),
        "both_evolutions_pass": bool(all(evolution_pass(run) for run in runs.values())),
        "final_field_transfer_below_5_percent": bool(
            max(final_field_transfer.values()) < .05
        ),
    }
    scan_rules = bool(all(
        detector_rules_pass(record)
        for scan in full_scans for record in scan["records"].values()
    ))
    transfer_rules = bool(all(
        scan["cross_grid_endpoint_transfer"] is not None
        and scan["cross_grid_endpoint_transfer"]["maximum"] < .01
        for scan in full_scans
    ))
    if trigger_index is None:
        decisive = bool(
            all(
                all(count == 0 for count in item["counts"].values())
                for item in screens
            )
            and all(count == 0 for count in full_scans[-1]["counts"].values())
            and scan_rules and transfer_rules
        )
        classification = (
            "no_detected_formation" if decisive else "unresolved"
        )
        formation_bracket = None
    else:
        pretrigger_zero = all(
            all(count == 0 for count in item["counts"].values())
            for item in screens[:trigger_index]
        )
        all_paired = all(
            all(count == 2 for count in scan["counts"].values())
            for scan in full_scans
        )
        decisive = bool(pretrigger_zero and all_paired and scan_rules and transfer_rules)
        classification = (
            "paired_formation_candidate" if decisive else "unresolved"
        )
        lower = 0. if trigger_index == 0 else times[trigger_index - 1]
        formation_bracket = {
            "lower": lower,
            "upper": times[trigger_index],
            "width": times[trigger_index] - lower,
        }

    acceptance = {
        **numerical_acceptance,
        "detector_outcome_is_prospectively_decisive": decisive,
    }
    if not all(numerical_acceptance.values()):
        status = "fail"
    elif decisive:
        status = "pass"
    else:
        status = "review"
    elapsed = time.perf_counter() - started
    payload = {
        "status": status,
        "classification": classification,
        "scope": f"sealed two-grid nearby-amplitude formation pilot at A={amplitude:.2f}",
        "protocol": PROTOCOL,
        "amplitude": amplitude,
        "final_time": FINAL_TIME,
        "steps": STEPS,
        "time_step": FINAL_TIME / STEPS,
        "sample_steps": list(SAMPLE_STEPS),
        "sample_times": times,
        "initial_static_search": initial_static,
        "evolution_diagnostics": {
            label: public_diagnostics(run) for label, run in runs.items()
        },
        "screen_history": screens,
        "trigger_index": trigger_index,
        "formation_bracket": formation_bracket,
        "full_discrepancy_scans": full_scans,
        "final_field_transfer": final_field_transfer,
        "acceptance": acceptance,
        "elapsed_seconds": elapsed,
        "limitations": [
            "finite twelve-seed star-shaped donor-capped search",
            "formation bracket is detector- and foliation-dependent",
            "finite amplitude samples do not prove an open basin",
            "short evolution through t=0.004",
            "not event-horizon location, continuum topology change, long-time stability, connected bulk geometry, or mass transfer",
        ],
    }

    state_values = {
        "sample_steps": np.asarray(SAMPLE_STEPS),
        "sample_times": np.asarray(times),
    }
    for label, geometry in geometries.items():
        state_values[f"{label}_z"] = geometry["z"]
        state_values[f"{label}_r"] = geometry["r"]
        for index, step in enumerate(SAMPLE_STEPS):
            state = sampled_state(runs[label], step)
            state_values[f"{label}_time_{index}_increment"] = (
                state["_position"] - cases[label]["initial"]
            )
            state_values[f"{label}_time_{index}_velocity"] = state["_velocity"]
    np.savez_compressed(state_path(amplitude), **state_values)
    output_path(amplitude).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({
        "amplitude": amplitude,
        "status": status,
        "classification": classification,
        "screen_counts": [item["counts"] for item in screens],
        "full_scan_counts": [item["counts"] for item in full_scans],
        "formation_bracket": formation_bracket,
        "final_field_transfer": final_field_transfer,
        "acceptance": acceptance,
        "elapsed_seconds": elapsed,
    }, indent=2), flush=True)
    return payload


def write_combined(records):
    all_decisive = all(record["status"] == "pass" for record in records)
    classes = [record["classification"] for record in records]
    if classes == ["paired_formation_candidate", "paired_formation_candidate"]:
        classification = "two_amplitude_paired_formation_pilot"
    elif all_decisive:
        classification = "decisive_nearby_amplitude_boundary_pilot"
    else:
        classification = "unresolved_nearby_amplitude_pilot"
    combined = {
        "status": "pass" if all_decisive else "review",
        "classification": classification,
        "scope": "sealed nearby-amplitude formation pilot at A=7.88 and A=7.92",
        "protocol": PROTOCOL,
        "amplitudes": [record["amplitude"] for record in records],
        "individual_results": [
            {
                "amplitude": record["amplitude"],
                "status": record["status"],
                "classification": record["classification"],
                "formation_bracket": record["formation_bracket"],
                "final_field_transfer": record["final_field_transfer"],
                "elapsed_seconds": record["elapsed_seconds"],
                "result_file": str(output_path(record["amplitude"])),
                "state_file": str(state_path(record["amplitude"])),
            }
            for record in records
        ],
        "supports_nearby_amplitude_basin_evidence": bool(
            classification == "two_amplitude_paired_formation_pilot"
        ),
        "limitations": [
            "three successful amplitudes including prior A=7.90 would still be finite sampled evidence, not proof of an open basin",
            "inherits each individual result's detector, slicing, surface-class, and short-time limitations",
        ],
    }
    combined_path = Path("results/corrected_nearby_amplitude_formation_pilot.json")
    combined_path.write_text(json.dumps(combined, indent=2, sort_keys=True) + "\n")
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--amplitude", type=float, choices=AMPLITUDES,
        help="run only one sealed amplitude (default: run both and combine)",
    )
    parser.add_argument(
        "--combine-only", action="store_true",
        help="combine existing individual result files without rerunning",
    )
    args = parser.parse_args()
    if args.combine_only:
        records = [json.loads(output_path(value).read_text()) for value in AMPLITUDES]
        print(json.dumps(write_combined(records), indent=2), flush=True)
        return
    fold = build_geometry("G6")
    values = (args.amplitude,) if args.amplitude is not None else AMPLITUDES
    records = [run_amplitude(fold, amplitude) for amplitude in values]
    if args.amplitude is None:
        print(json.dumps(write_combined(records), indent=2), flush=True)


if __name__ == "__main__":
    main()
