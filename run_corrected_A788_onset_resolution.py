#!/usr/bin/env python3
"""Sealed, resumable three-grid time refinement of the A=7.88 onset."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import run_corrected_fold_live_nonlinear_gauge_source as live
from bhps.onset_resolution import endpoint_vector_difference, onset_summary
from bhps.recovery_indexer import (
    RecoveryIndex,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
    validate_npz,
)
from run_corrected_A790_formation_time_refinement import public_diagnostics
from run_corrected_A790_independent_dynamic_BVP_detector import (
    SEEDS,
    analytic_controls,
    search_slice,
)
from run_corrected_A790_two_grid_formation_search import (
    endpoint_transfer,
    evolution_pass,
    field_transfer,
    static_search,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import relative_norm


PROTOCOL = Path("notes/90_A788_onset_resolution_protocol.md")
OUTPUT = Path("results/corrected_A788_onset_resolution.json")
STATE_OUTPUT = Path("results/corrected_A788_onset_resolution_state.npz")
RECOVERY_ROOT = Path("results/corrected_A788_onset_resolution_recovery")
MANIFEST = RECOVERY_ROOT / "index.json"
COARSE_RESULT = Path("results/corrected_A788_formation_pilot.json")
COARSE_STATE = Path("results/corrected_A788_formation_pilot_state.npz")
AMPLITUDE = 7.88
FINAL_TIME = 0.0015
STEPS = 24
DT = FINAL_TIME / STEPS
SEGMENT_STEPS = 4
SEARCH_STEPS = tuple(range(10, STEPS + 1))
GRID_SPECS = {
    "G7": (81, 121, 40, 270),
    "G8": (97, 145, 45, 280),
    "G9": (113, 169, 50, 300),
}
FIXED_INPUT_HASHES = {
    "notes/78_nearby_amplitude_formation_pilot_protocol.md":
        "a2da898d4a7e874dc5c277b12c53ead6dcc5da31db4b699f9ae1d5c5f6439454",
    "results/corrected_A788_formation_pilot.json":
        "669fcefcf5254130ed0197e5b90e69aeba379f6517ef590ab545ae4ac665eb00",
    "results/corrected_A788_formation_pilot_state.npz":
        "d7650fa3128094c903cb9c843ce1b1f018bcc84a14c1558828d6379f259f169b",
    "run_corrected_nearby_amplitude_formation_pilot.py":
        "7d8866e7115ff0fe9f208542e9d81dcb2163d5effdf5e9a7e579e4ee04685e18",
    "run_corrected_A790_independent_dynamic_BVP_detector.py":
        "4c71239822a5674cb9166483b647effdccc5b2c629d7e5b936ac7933cbc2a1f4",
}


def recovery_inputs():
    paths = (
        Path(__file__), Path("src/bhps/onset_resolution.py"),
        Path("src/bhps/recovery_indexer.py"),
        Path("src/bhps/dynamical_capped_horizon_bvp.py"),
    )
    return {**FIXED_INPUT_HASHES, **{str(path): sha256_file(path) for path in paths}}


def stage_json(index, stage_id, path, kind, metadata, producer, expected=600.0):
    index.register(stage_id, kind, expected, metadata)
    cached = index.validated_path(stage_id)
    if cached is not None:
        payload = json.loads(cached.read_text())
        if payload.get("protocol_sha256") != index.protocol_sha256:
            raise RuntimeError(f"cached protocol mismatch in {stage_id}")
        return payload, True
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        payload = producer()
        wrapped = {
            "stage_id": stage_id,
            "protocol_sha256": index.protocol_sha256,
            **payload,
        }
        atomic_write_json(path, wrapped)
        check = json.loads(path.read_text())
        if check.get("stage_id") != stage_id:
            raise RuntimeError(f"atomic JSON validation failed for {stage_id}")
        elapsed = time.perf_counter() - started
        index.mark_complete(stage_id, path, elapsed)
        return wrapped, False
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def build_geometries():
    print("constructing fresh corrected A=7.88 G7/G8/G9 geometries", flush=True)
    fold = build_geometry("G6")
    parent = {**fold, "fold_amplitude": AMPLITUDE}
    result = {}
    for label, (nz, nr, selector_iterations, slice_iterations) in GRID_SPECS.items():
        result[label] = build_refined(
            parent, nz, nr, f"{label}A788-onset-resolution",
            selector_iterations=selector_iterations,
            slice_iterations=slice_iterations,
        )
        parent = result[label]
    return result


def empty_segment_diagnostics():
    return {
        "all_stages_finite": True,
        "maximum_stage_acceleration_relative_change": 0.0,
        "maximum_normal_wall_acceleration_residual": 0.0,
        "maximum_outer_acceleration_residual": 0.0,
        "maximum_outer_source_residual": 0.0,
        "maximum_outer_metric_correction": 0.0,
        "maximum_outer_scalar_correction": 0.0,
        "maximum_outer_source_correction": 0.0,
    }


def update_segment_diagnostics(summary, k1, k2, diagnostics):
    summary["maximum_stage_acceleration_relative_change"] = max(
        summary["maximum_stage_acceleration_relative_change"],
        relative_norm(k1[1], k2[1]),
    )
    for diagnostic in diagnostics:
        summary["all_stages_finite"] = bool(
            summary["all_stages_finite"] and diagnostic["finite"]
        )
        normal = diagnostic["normal_wall_gauge"]
        if normal is not None:
            summary["maximum_normal_wall_acceleration_residual"] = max(
                summary["maximum_normal_wall_acceleration_residual"],
                normal["final_residual"]["maximum"],
            )
        outer = diagnostic["outer_sommerfeld"]
        if outer is not None:
            summary["maximum_outer_acceleration_residual"] = max(
                summary["maximum_outer_acceleration_residual"],
                outer["maximum_normalized_acceleration_residual"],
            )
            summary["maximum_outer_metric_correction"] = max(
                summary["maximum_outer_metric_correction"],
                outer["metric_relative_correction"],
            )
            summary["maximum_outer_scalar_correction"] = max(
                summary["maximum_outer_scalar_correction"],
                outer["scalar_relative_correction"],
            )
        outer_source = diagnostic["outer_source_sommerfeld"]
        if outer_source is not None:
            summary["maximum_outer_source_residual"] = max(
                summary["maximum_outer_source_residual"],
                outer_source["maximum_normalized"],
            )
            summary["maximum_outer_source_correction"] = max(
                summary["maximum_outer_source_correction"],
                outer_source["relative_correction"],
            )


def integrate_segment(case, state, start_step, end_step):
    position, velocity, source, memory = (np.asarray(value).copy() for value in state)
    diagnostic = empty_segment_diagnostics()
    snapshots = {}
    for step in range(start_step + 1, end_step + 1):
        current_time = (step - 1) * DT
        print(
            f"{case['label']}: restartable live step {step}/{STEPS}, stage 1",
            flush=True,
        )
        k1, d1 = live.driver_stage(
            case, current_time, position, velocity, source, memory,
        )
        midpoint = tuple(
            value + 0.5 * DT * slope
            for value, slope in zip((position, velocity, source, memory), k1)
        )
        print(
            f"{case['label']}: restartable live step {step}/{STEPS}, stage 2",
            flush=True,
        )
        k2, d2 = live.driver_stage(case, current_time + 0.5 * DT, *midpoint)
        update_segment_diagnostics(diagnostic, k1, k2, (d1, d2))
        position, velocity, source, memory = tuple(
            value + DT * slope
            for value, slope in zip((position, velocity, source, memory), k2)
        )
        snapshots[f"step_{step:03d}_increment"] = position - case["initial"]
        snapshots[f"step_{step:03d}_velocity"] = velocity.copy()
    return (position, velocity, source, memory), snapshots, diagnostic


def segment_path(label, start_step, end_step):
    return RECOVERY_ROOT / f"evolution_{label}_steps_{start_step + 1:03d}_{end_step:03d}.npz"


def diagnostic_arrays(diagnostic):
    return {
        f"diag_{key}": np.asarray(value)
        for key, value in diagnostic.items()
    }


def diagnostic_from_archive(archive):
    return {
        key.removeprefix("diag_"): archive[key].item()
        for key in archive.files if key.startswith("diag_")
    }


def validate_segment(path, shape, source_shape, start_step, end_step):
    required = {
        "end_position": shape, "end_velocity": shape,
        "end_source": source_shape, "end_memory": source_shape,
        "start_step": (), "end_step": (),
    }
    for step in range(start_step + 1, end_step + 1):
        required[f"step_{step:03d}_increment"] = shape
        required[f"step_{step:03d}_velocity"] = shape
    validate_npz(path, required)
    with np.load(path) as archive:
        if int(archive["start_step"]) != start_step or int(archive["end_step"]) != end_step:
            raise RuntimeError("segment step-index mismatch")


def run_grid_evolution(index, label, geometry, case):
    state = (
        case["initial"].copy(), np.zeros_like(case["initial"]),
        case["source0"].copy(), case["memory0"].copy(),
    )
    diagnostics = []
    paths = []
    shape = tuple(case["initial"].shape)
    source_shape = tuple(case["source0"].shape)
    parent_sha256 = "initial-data"
    for start_step in range(0, STEPS, SEGMENT_STEPS):
        end_step = min(start_step + SEGMENT_STEPS, STEPS)
        stage_id = f"evolution/{label}/steps_{start_step + 1:03d}_{end_step:03d}"
        path = segment_path(label, start_step, end_step)
        metadata = {
            "grid": label, "start_step": start_step, "end_step": end_step,
            "start_time": start_step * DT, "end_time": end_step * DT,
            "parent_sha256": parent_sha256,
        }
        index.register(stage_id, "evolution-segment", 1800.0, metadata)
        cached = index.validated_path(stage_id)
        if cached is None:
            index.mark_running(stage_id)
            started = time.perf_counter()
            try:
                state, snapshots, segment_diagnostic = integrate_segment(
                    case, state, start_step, end_step,
                )
                atomic_write_npz(
                    path,
                    start_step=np.asarray(start_step), end_step=np.asarray(end_step),
                    end_position=state[0], end_velocity=state[1],
                    end_source=state[2], end_memory=state[3],
                    **snapshots, **diagnostic_arrays(segment_diagnostic),
                )
                validate_segment(path, shape, source_shape, start_step, end_step)
                elapsed = time.perf_counter() - started
                index.mark_complete(stage_id, path, elapsed)
                cached = path
            except Exception as error:
                index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
                raise
        else:
            validate_segment(cached, shape, source_shape, start_step, end_step)
        with np.load(cached) as archive:
            state = tuple(
                np.asarray(archive[key])
                for key in ("end_position", "end_velocity", "end_source", "end_memory")
            )
            diagnostics.append(diagnostic_from_archive(archive))
        paths.append(cached)
        parent_sha256 = sha256_file(cached)
    return state, diagnostics, paths


def combine_segment_diagnostics(records):
    combined = empty_segment_diagnostics()
    for record in records:
        combined["all_stages_finite"] = bool(
            combined["all_stages_finite"] and record["all_stages_finite"]
        )
        for key in combined:
            if key != "all_stages_finite":
                combined[key] = max(float(combined[key]), float(record[key]))
    return combined


def finalize_run(case, state, segment_diagnostics):
    position, velocity, source, memory = state
    time_value = FINAL_TIME
    source_z, source_r = live.regular_source_spatial_derivatives(
        source, case["z"], case["r"],
    )
    target = live.regular_so3_nonlinear_anchored_damped_wave_target(
        position, case["initial"], case["source0"], case["r"],
        live.TARGET_MU_LAPSE, live.TARGET_MU_SHIFT, live.TARGET_POWER,
    )
    advection = live.regular_so3_live_source_shift_advection(
        position, case["r"], source, source_z, source_r,
    )
    source_dot, memory_dot = live.source_driver_rhs(
        source, memory, target, live.DRIVER_MU, live.DRIVER_ETA, advection,
    )
    final_outer_source = None
    if case["rhs"].live_outer_sommerfeld:
        source_dot, final_outer_source = live.apply_outer_source_sommerfeld(
            source, source_dot, case["source0"], case["source_time0"],
            case["_initial_source_second_time"], position, time_value,
            case["r"], case["rhs"].stencil_width,
        )
    final_gauge = live.StageRegularGaugeSource(
        source, source_dot, case["z"], case["r"],
    )
    print(f"{case['label']}: resumed-run final diagnostics", flush=True)
    constraint = live.gauge_constraint_summary(
        position, velocity, time_value, case["rhs"],
        live.RADIAL_COMPARISON_CUT, final_gauge,
    )
    wall = live.compact_wall_position_residuals(
        position, case["z"], case["r"], case["geometry"]["background"],
    )
    run = {
        "final_time": time_value, "steps": STEPS, "time_step": DT,
        **combine_segment_diagnostics(segment_diagnostics),
        "final_constraint": constraint, "final_wall": wall,
        "signature": live.signature_summary(position, case["r"]),
        "final_normal_wall_position_residual":
            live.compact_wall_normal_gauge_position_residuals(
                position, source, case["z"], case["r"],
                case["geometry"]["background"],
            ),
        "final_outer_sommerfeld_position_residual":
            live.outer_sommerfeld_position_residuals(
                position, velocity, case["rhs"].outer_reference_position,
                case["rhs"].outer_reference_acceleration, time_value,
                case["r"], case["rhs"].stencil_width,
            ),
        "final_outer_source_sommerfeld_residual": final_outer_source,
        "_position": position, "_increment": position - case["initial"],
        "_velocity": velocity, "_source": source,
        "_source_increment": source - case["source0"], "_memory": memory,
    }
    return run


def load_evolved_step(label, step, initial):
    start_step = ((step - 1) // SEGMENT_STEPS) * SEGMENT_STEPS
    end_step = min(start_step + SEGMENT_STEPS, STEPS)
    path = segment_path(label, start_step, end_step)
    with np.load(path) as archive:
        increment = np.asarray(archive[f"step_{step:03d}_increment"])
        velocity = np.asarray(archive[f"step_{step:03d}_velocity"])
    return initial + increment, velocity


def search_passes(record):
    count = record["admitted_distinct_count"]
    return bool(
        count in (0, 2)
        and (count == 0 or (
            len(record["clusters"]) == 2
            and all(len(cluster["members"]) >= 2 for cluster in record["clusters"])
        ))
    )


def bvp_stage(index, label, step, position, velocity, geometry, coarse=False):
    family = "coarse_bvp" if coarse else "bvp"
    time_value = step * (0.000125 if coarse else DT)
    stage_id = f"{family}/{label}/step_{step:03d}"
    path = RECOVERY_ROOT / f"{family}_{label}_step_{step:03d}.json"
    payload, _ = stage_json(
        index, stage_id, path, "independent-BVP-search",
        {"grid": label, "step": step, "time": time_value, "coarse": coarse},
        lambda: {
            "grid": label, "step": step, "time": time_value,
            "coarse": coarse,
            "search": search_slice(
                f"{label}-A788-{'coarse' if coarse else 'fine'}-t{time_value:.7f}",
                position, velocity, geometry,
            ),
        },
        expected=900.0,
    )
    return payload["search"]


def initial_stage(index, label, geometry):
    position = np.asarray(geometry["jet_field"].reduced_fields)
    velocity = np.zeros_like(position)
    stage_id = f"initial/{label}"
    path = RECOVERY_ROOT / f"initial_{label}.json"
    payload, _ = stage_json(
        index, stage_id, path, "initial-surface-search",
        {"grid": label, "time": 0.0},
        lambda: {
            "grid": label, "time": 0.0,
            "static": static_search(geometry),
            "BVP": search_slice(f"{label}-A788-initial", position, velocity, geometry),
        },
        expected=900.0,
    )
    return payload


def cross_grid_endpoint_records(searches):
    result = {}
    common_steps = [
        step for step in SEARCH_STEPS
        if all(searches[label][step]["admitted_distinct_count"] == 2 for label in GRID_SPECS)
    ]
    for step in common_steps:
        signatures = {
            label: searches[label][step]["admitted_signatures"]
            for label in GRID_SPECS
        }
        result[str(step)] = {
            "time": step * DT,
            "G7_G8": endpoint_transfer(signatures["G7"], signatures["G8"]),
            "G8_G9": endpoint_transfer(signatures["G8"], signatures["G9"]),
            "G7_G8_vector_difference": endpoint_vector_difference(
                signatures["G7"], signatures["G8"],
            ),
            "G8_G9_vector_difference": endpoint_vector_difference(
                signatures["G8"], signatures["G9"],
            ),
        }
    return result


def assemble_state_archive(index, geometries, segment_paths):
    arrays = {"times": np.arange(1, STEPS + 1, dtype=float) * DT}
    for label, geometry in geometries.items():
        arrays[f"{label}_z"] = geometry["z"]
        arrays[f"{label}_r"] = geometry["r"]
        for path in segment_paths[label]:
            with np.load(path) as archive:
                for key in archive.files:
                    if key.startswith("step_"):
                        arrays[f"{label}_{key}"] = np.asarray(archive[key])
        final = segment_paths[label][-1]
        with np.load(final) as archive:
            for name in ("end_source", "end_memory"):
                arrays[f"{label}_{name}"] = np.asarray(archive[name])
    stage_id = "final/state_archive"
    index.register(stage_id, "combined-state-archive", 900.0, {"steps": STEPS})
    cached = index.validated_path(stage_id)
    if cached is None:
        index.mark_running(stage_id)
        started = time.perf_counter()
        try:
            atomic_write_npz(STATE_OUTPUT, **arrays)
            validate_npz(STATE_OUTPUT)
            index.mark_complete(stage_id, STATE_OUTPUT, time.perf_counter() - started)
        except Exception as error:
            index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
            raise
    else:
        validate_npz(cached)


def combine(index, geometries, cases, runs, initial, searches, coarse_searches, segment_paths):
    histories = {
        label: [searches[label][step]["admitted_distinct_count"] for step in SEARCH_STEPS]
        for label in GRID_SPECS
    }
    search_times = [step * DT for step in SEARCH_STEPS]
    onset = onset_summary(search_times, histories, DT)
    endpoint_records = cross_grid_endpoint_records(searches)
    final_step_record = endpoint_records.get(str(STEPS))
    final_endpoint_gate = bool(
        final_step_record is not None
        and (
            final_step_record["G8_G9_vector_difference"]
            < final_step_record["G7_G8_vector_difference"]
            or max(
                final_step_record["G7_G8"]["maximum"],
                final_step_record["G8_G9"]["maximum"],
            ) < 0.001
        )
    )
    endpoint_transfer_gate = bool(
        endpoint_records
        and all(
            record[pair] is not None and record[pair]["maximum"] < 0.01
            for record in endpoint_records.values() for pair in ("G7_G8", "G8_G9")
        )
        and final_endpoint_gate
    )
    temporal = {}
    with np.load(COARSE_STATE) as archive:
        coarse_times = np.asarray(archive["sample_times"])
        for label in ("G7", "G8"):
            temporal[label] = {}
            for time_value in (0.001, 0.0015):
                coarse_index = int(np.flatnonzero(np.isclose(coarse_times, time_value))[0])
                fine_step = int(round(time_value / DT))
                fine_position, fine_velocity = load_evolved_step(
                    label, fine_step, cases[label]["initial"],
                )
                coarse_increment = np.asarray(
                    archive[f"{label}_time_{coarse_index}_increment"]
                )
                coarse_velocity = np.asarray(
                    archive[f"{label}_time_{coarse_index}_velocity"]
                )
                coarse_step = int(round(time_value / 0.000125))
                coarse_record = coarse_searches[label][coarse_step]
                fine_record = searches[label][fine_step]
                temporal[label][f"{time_value:.6f}"] = {
                    "coarse_step": coarse_step, "fine_step": fine_step,
                    "coarse_count": coarse_record["admitted_distinct_count"],
                    "fine_count": fine_record["admitted_distinct_count"],
                    "position_increment_relative_difference": relative_norm(
                        coarse_increment, fine_position - cases[label]["initial"],
                    ),
                    "velocity_relative_difference": relative_norm(
                        coarse_velocity, fine_velocity,
                    ),
                    "coarse_fine_endpoint_transfer": endpoint_transfer(
                        coarse_record["admitted_signatures"],
                        fine_record["admitted_signatures"],
                    ),
                }
    temporal_gate = bool(
        all(
            record[name] < 0.01
            for grid in temporal.values() for record in grid.values()
            for name in (
                "position_increment_relative_difference",
                "velocity_relative_difference",
            )
        )
        and all(
            temporal[label]["0.001500"][count] == 2
            for label in temporal for count in ("coarse_count", "fine_count")
        )
    )
    final_field_transfer = {
        "G7_G8": {
            name: field_transfer(cases["G7"], runs["G7"], cases["G8"], runs["G8"], key)
            for name, key in (("position_increment", "_increment"), ("velocity", "_velocity"))
        },
        "G8_G9": {
            name: field_transfer(cases["G8"], runs["G8"], cases["G9"], runs["G9"], key)
            for name, key in (("position_increment", "_increment"), ("velocity", "_velocity"))
        },
    }
    initial_gate = bool(all(
        initial[label]["static"]["accepted_count"] == 0
        and initial[label]["BVP"]["admitted_distinct_count"] == 0
        for label in GRID_SPECS
    ))
    detector_gate = bool(all(
        search_passes(record)
        for grid in searches.values() for record in grid.values()
    ))
    persistent_gate = bool(
        onset["complete"] and all(counts[-1] == 2 for counts in histories.values())
    )
    localization_gate = bool(
        onset["complete"]
        and onset["spread_below_two_steps"]
        and onset["G8_G9_lag_not_worse_than_G7_G8_plus_one_step"]
        and all(histories[label][0] == 0 for label in GRID_SPECS)
    )
    acceptance = {
        "analytic_BVP_controls_pass": True,
        "all_three_initial_static_and_BVP_searches_find_zero": initial_gate,
        "all_three_evolutions_pass": bool(all(evolution_pass(run) for run in runs.values())),
        "all_independent_BVP_searches_pass_admission_rules": detector_gate,
        "all_three_histories_have_one_persistent_zero_to_two_transition": persistent_gate,
        "three_grid_onset_spread_and_delayed_threshold_gate": localization_gate,
        "cross_grid_endpoints_transfer_and_final_spatial_gate": endpoint_transfer_gate,
        "coarse_fine_time_refinement_gate": temporal_gate,
        "final_adjacent_grid_fields_transfer_below_5_percent": bool(
            max(value for pair in final_field_transfer.values() for value in pair.values()) < 0.05
        ),
        "provenance_and_recovery_artifacts_validate": True,
    }
    hard_keys = (
        "analytic_BVP_controls_pass",
        "all_three_initial_static_and_BVP_searches_find_zero",
        "all_three_evolutions_pass",
        "provenance_and_recovery_artifacts_validate",
    )
    if not all(acceptance[key] for key in hard_keys):
        status = "fail"
        classification = "A788_onset_resolution_hard_failure"
    elif all(acceptance.values()):
        status = "pass"
        classification = "three_grid_time_refined_delayed_formation"
    else:
        status = "review"
        classification = "unresolved_A788_onset_boundary"
    stage_runtime = float(sum(
        stage.get("elapsed_seconds", 0.0)
        for stage in index.data["stages"].values()
    ))
    payload = {
        "status": status, "classification": classification,
        "scope": "sealed A=7.88 three-grid half-step onset resolution",
        "protocol": str(PROTOCOL), "protocol_sha256": index.protocol_sha256,
        "amplitude": AMPLITUDE, "grids": {
            label: {"size": list(GRID_SPECS[label][:2]), "r_max": 8.0}
            for label in GRID_SPECS
        },
        "final_time": FINAL_TIME, "steps": STEPS, "time_step": DT,
        "evolution_checkpoint_steps": list(range(SEGMENT_STEPS, STEPS + 1, SEGMENT_STEPS)),
        "search_steps": list(SEARCH_STEPS), "search_times": search_times,
        "BVP_seeds": list(SEEDS),
        "provenance": {
            "inputs": recovery_inputs(), "recovery_manifest": str(MANIFEST),
            "recovery_manifest_sha256_before_final_result": sha256_file(MANIFEST),
        },
        "initial_searches": initial,
        "evolution_diagnostics": {
            label: public_diagnostics(run) for label, run in runs.items()
        },
        "fine_BVP_searches": {
            label: {str(step): record for step, record in grid.items()}
            for label, grid in searches.items()
        },
        "fine_count_histories": histories,
        "onset": onset,
        "cross_grid_endpoint_records": endpoint_records,
        "coarse_fine_BVP_and_field_comparison": temporal,
        "final_field_transfer": final_field_transfer,
        "acceptance": acceptance,
        "runtime": {
            "new_stage_compute_seconds": stage_runtime,
            "recovery_stage_count": len(index.data["stages"]),
        },
        "limitations": [
            "finite twelve-seed star-shaped donor-capped BVP search",
            "same-domain three-grid sequence and one half-size time step",
            "formation time is sampled, detector dependent, and foliation dependent",
            "initial zero searches are not a global nonexistence proof",
            "not an event horizon, continuum critical time, open basin, topology change, nonlinear stability, connected geometry, dark matter, or mass-transfer result",
        ],
    }
    assemble_state_archive(index, geometries, segment_paths)
    atomic_write_json(OUTPUT, payload)
    if json.loads(OUTPUT.read_text()).get("status") != status:
        raise RuntimeError("combined result validation failed")
    final_stage = "final/result"
    index.register(final_stage, "combined-result", 300.0, {"status": status})
    if index.validated_path(final_stage) is None:
        index.mark_running(final_stage)
        index.mark_complete(final_stage, OUTPUT, 0.0)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", choices=tuple(GRID_SPECS), help="run/resume one grid only")
    parser.add_argument("--combine-only", action="store_true")
    args = parser.parse_args()
    overall_started = time.perf_counter()
    index = RecoveryIndex(MANIFEST, PROTOCOL, recovery_inputs())
    controls_payload, _ = stage_json(
        index, "controls/analytic_BVP", RECOVERY_ROOT / "analytic_BVP.json",
        "analytic-control", {}, lambda: {"controls": analytic_controls()},
        expected=300.0,
    )
    if not controls_payload["controls"]["passed"]:
        raise RuntimeError("sealed analytic BVP control failed")
    geometries = build_geometries()
    cases = {
        label: live.setup_case(
            geometry, f"{label}-A788-onset-resolution",
            live_normal_wall_gauge=True, live_outer_sommerfeld=True,
        )
        for label, geometry in geometries.items()
    }
    labels = (args.grid,) if args.grid else tuple(GRID_SPECS)
    if args.combine_only:
        labels = ()
    initial = {}
    runs = {}
    segment_paths = {}
    searches = {}
    for label in GRID_SPECS:
        initial[label] = initial_stage(index, label, geometries[label])
        if (
            initial[label]["static"]["accepted_count"] != 0
            or initial[label]["BVP"]["admitted_distinct_count"] != 0
        ):
            raise RuntimeError(f"sealed initial-zero gate failed on {label}")
    for label in labels:
        print(f"running/resuming {label} A=7.88 half-step evolution", flush=True)
        state, diagnostics, paths = run_grid_evolution(
            index, label, geometries[label], cases[label],
        )
        runs[label] = finalize_run(cases[label], state, diagnostics)
        segment_paths[label] = paths
        if not evolution_pass(runs[label]):
            raise RuntimeError(f"sealed evolution-validity gate failed on {label}")
        searches[label] = {}
        for step in SEARCH_STEPS:
            position, velocity = load_evolved_step(
                label, step, cases[label]["initial"],
            )
            searches[label][step] = bvp_stage(
                index, label, step, position, velocity, geometries[label],
            )
    if args.grid:
        print(json.dumps({
            "grid": args.grid,
            "status": "stage_complete",
            "counts": [searches[args.grid][step]["admitted_distinct_count"] for step in SEARCH_STEPS],
            "elapsed_seconds": time.perf_counter() - overall_started,
        }, indent=2), flush=True)
        return
    if args.combine_only:
        for label in GRID_SPECS:
            paths = [
                segment_path(label, start, min(start + SEGMENT_STEPS, STEPS))
                for start in range(0, STEPS, SEGMENT_STEPS)
            ]
            if not all(path.exists() for path in paths):
                raise FileNotFoundError(f"missing evolution segments for {label}")
            segment_paths[label] = paths
            diagnostics = []
            for path in paths:
                with np.load(path) as archive:
                    diagnostics.append(diagnostic_from_archive(archive))
            with np.load(paths[-1]) as archive:
                state = tuple(np.asarray(archive[key]) for key in (
                    "end_position", "end_velocity", "end_source", "end_memory",
                ))
            runs[label] = finalize_run(cases[label], state, diagnostics)
            searches[label] = {}
            for step in SEARCH_STEPS:
                stage_id = f"bvp/{label}/step_{step:03d}"
                path = index.validated_path(stage_id)
                if path is None:
                    raise FileNotFoundError(f"missing validated {stage_id}")
                searches[label][step] = json.loads(path.read_text())["search"]
    coarse_searches = {label: {} for label in ("G7", "G8")}
    with np.load(COARSE_STATE) as archive:
        coarse_times = np.asarray(archive["sample_times"])
        for label in coarse_searches:
            for time_value in (0.001, 0.0015):
                index_value = int(np.flatnonzero(np.isclose(coarse_times, time_value))[0])
                coarse_step = int(round(time_value / 0.000125))
                position = (
                    cases[label]["initial"]
                    + np.asarray(archive[f"{label}_time_{index_value}_increment"])
                )
                velocity = np.asarray(archive[f"{label}_time_{index_value}_velocity"])
                coarse_searches[label][coarse_step] = bvp_stage(
                    index, label, coarse_step, position, velocity,
                    geometries[label], coarse=True,
                )
    payload = combine(
        index, geometries, cases, runs, initial, searches,
        coarse_searches, segment_paths,
    )
    print(json.dumps({
        "status": payload["status"],
        "classification": payload["classification"],
        "fine_count_histories": payload["fine_count_histories"],
        "onset": payload["onset"],
        "final_field_transfer": payload["final_field_transfer"],
        "acceptance": payload["acceptance"],
        "elapsed_seconds": time.perf_counter() - overall_started,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
