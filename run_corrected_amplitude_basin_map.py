#!/usr/bin/env python3
"""Sealed Test-6 finite-amplitude, domain-qualified basin map."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import run_corrected_A788_onset_resolution as resumable
import run_corrected_fold_live_nonlinear_gauge_source as live
from bhps.amplitude_basin import (
    CANDIDATE_AMPLITUDES,
    DOMAIN_ANCHORS,
    aggregate_basin_status,
    amplitude_tag,
    build_R12_pair,
    monotone_onset_diagnostic,
    persistent_late_pair,
    sampled_onset,
)
from bhps.recovery_indexer import (
    RecoveryIndex,
    atomic_write_json,
    sha256_file,
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


PROTOCOL = Path("notes/94_amplitude_basin_map_protocol.md")
OUTPUT = Path("results/corrected_amplitude_basin_map.json")
RECOVERY_ROOT = Path("results/corrected_amplitude_basin_recovery")
R8_FRESH = (7.84, 7.86)
R8_ARCHIVED = (7.88, 7.92)
R12_FRESH = (7.84, 7.92)
R8_FINAL_TIME = 0.004
R8_STEPS = 32
R8_SEARCH_STEPS = tuple(range(4, R8_STEPS + 1, 4))
R12_FINAL_TIME = 0.004
R12_STEPS = 8
R12_SEARCH_STEPS = tuple(range(1, R12_STEPS + 1))
LATE_TIMES = (0.003, 0.004)

FROZEN_INPUT_HASHES = {
    "results/corrected_family_knot_A8_state.npz":
        "e293184299baf5b791f6a49a91f6e6266ad656072f2e8fd2e39ce21d14c5416e",
    "results/corrected_A788_formation_pilot.json":
        "669fcefcf5254130ed0197e5b90e69aeba379f6517ef590ab545ae4ac665eb00",
    "results/corrected_A788_formation_pilot_state.npz":
        "d7650fa3128094c903cb9c843ce1b1f018bcc84a14c1558828d6379f259f169b",
    "results/corrected_A792_formation_pilot.json":
        "c787a27014484965471b68aa4fe006069ffaaab37ddbc5dbb164f2c7143742bf",
    "results/corrected_A792_formation_pilot_state.npz":
        "8868a947724f651761c2603d76e4f6361d30ed23073c9c21feb73196aab6f5a5",
    "results/corrected_A788_onset_resolution.json":
        "4740db4f670993e04a93144f71a41488d652b550216c29f503c3e923c05d9afe",
    "results/corrected_A790_two_grid_formation_search.json":
        "7dfbe900b0733a62684749c6383aea2e4788816ec8715eae5c0a1778023979b2",
    "results/corrected_A790_independent_dynamic_BVP_detector.json":
        "d4d8b4771c0c33e32008fdf27b7d2d2477d9701f42b70d06f336a0905ee19127",
    "results/corrected_A790_R12_domain_sequence.json":
        "8dd9119315d5b62113dfc5c541884d1bc696baba73bf9ee7bbbaa6ae1e4389a1",
    "results/corrected_A790_initial_no_horizon_certificate.json":
        "3e14bd9d4f300963b4e771acb8d82bf7e46109995a07808ce29d8c178c56898a",
}


def worker_name(domain, amplitude):
    return f"{domain.lower()}_{amplitude_tag(amplitude).lower()}"


def worker_output(domain, amplitude):
    return RECOVERY_ROOT / worker_name(domain, amplitude) / "result.json"


def worker_root(domain, amplitude):
    return RECOVERY_ROOT / worker_name(domain, amplitude)


def recovery_inputs():
    current = (
        Path(__file__), Path("src/bhps/amplitude_basin.py"),
        Path("src/bhps/recovery_indexer.py"),
        Path("src/bhps/dynamical_capped_horizon_bvp.py"),
        Path("run_corrected_A788_onset_resolution.py"),
        Path("run_corrected_A790_independent_dynamic_BVP_detector.py"),
    )
    return {
        **FROZEN_INPUT_HASHES,
        **{str(path): sha256_file(path) for path in current},
    }


def verify_frozen_inputs():
    for path, expected in FROZEN_INPUT_HASHES.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"frozen input mismatch for {path}: {actual} != {expected}")


def stage_json(index, root, stage_id, kind, metadata, producer, expected=900.0):
    safe = stage_id.replace("/", "_")
    path = root / f"{safe}.json"
    index.register(stage_id, kind, expected, metadata)
    cached = index.validated_path(stage_id)
    if cached is not None:
        payload = json.loads(cached.read_text())
        if payload.get("protocol_sha256") != index.protocol_sha256:
            raise RuntimeError(f"cached protocol mismatch in {stage_id}")
        return payload
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        payload = {
            "stage_id": stage_id,
            "protocol_sha256": index.protocol_sha256,
            **producer(),
        }
        atomic_write_json(path, payload)
        if json.loads(path.read_text()).get("stage_id") != stage_id:
            raise RuntimeError(f"atomic JSON validation failed for {stage_id}")
        index.mark_complete(stage_id, path, time.perf_counter() - started)
        return payload
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def search_passes(record):
    count = int(record["admitted_distinct_count"])
    return bool(
        count in (0, 2)
        and (count == 0 or (
            len(record["clusters"]) == 2
            and all(len(cluster["members"]) >= 2 for cluster in record["clusters"])
        ))
    )


def construction_pass(geometries):
    return bool(all(
        float(geometry["reference_maximum_residual"]) < 1e-9
        and float(geometry["selector_maximum"]) < 1e-9
        and np.all(np.isfinite(geometry["jet_field"].reduced_fields))
        for geometry in geometries.values()
    ))


def build_R8_pair(amplitude):
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": float(amplitude)}
    tag = amplitude_tag(amplitude)
    g7 = build_refined(
        seed, 81, 121, f"G7{tag}R8-basin", selector_iterations=40,
        slice_iterations=270,
    )
    g8 = build_refined(
        g7, 97, 145, f"G8{tag}R8-basin", selector_iterations=45,
        slice_iterations=280,
    )
    return g7, g8


def initial_search_stage(index, root, domain, amplitude, label, geometry):
    position = np.asarray(geometry["jet_field"].reduced_fields)
    velocity = np.zeros_like(position)
    return stage_json(
        index, root, f"initial/{label}", "initial-static-and-BVP-search",
        {"domain": domain, "amplitude": amplitude, "grid": label, "time": 0.0},
        lambda: {
            "domain": domain, "amplitude": amplitude, "grid": label,
            "static": static_search(geometry),
            "BVP": search_slice(
                f"{label}-{amplitude_tag(amplitude)}-{domain}-basin-t0",
                position, velocity, geometry,
            ),
        },
    )


def bvp_stage(index, root, domain, amplitude, label, step, time_value,
              position, velocity, geometry):
    return stage_json(
        index, root, f"bvp/{label}/step_{step:03d}", "independent-BVP-search",
        {
            "domain": domain, "amplitude": amplitude, "grid": label,
            "step": step, "time": time_value,
        },
        lambda: {
            "domain": domain, "amplitude": amplitude, "grid": label,
            "step": step, "time": time_value,
            "search": search_slice(
                f"{label}-{amplitude_tag(amplitude)}-{domain}-basin-t{time_value:.7f}",
                position, velocity, geometry,
            ),
        },
    )["search"]


def summarize_searches(amplitude, times, searches):
    counts = {
        f"{label}_t{time_value:.3f}": int(
            searches[label][time_value]["admitted_distinct_count"]
        )
        for label in ("G7", "G8") for time_value in times
    }
    transfers = {
        f"t{time_value:.3f}": endpoint_transfer(
            searches["G7"][time_value]["admitted_signatures"],
            searches["G8"][time_value]["admitted_signatures"],
        ) for time_value in times
    }
    combined_counts = []
    for time_value in times:
        pair = [
            searches[label][time_value]["admitted_distinct_count"]
            for label in ("G7", "G8")
        ]
        combined_counts.append(pair[0] if pair[0] == pair[1] else 1)
    onset = sampled_onset(times, combined_counts)
    valid = bool(all(
        search_passes(record) for records in searches.values()
        for record in records.values()
    ))
    late_pair = persistent_late_pair(counts)
    late_transfer = bool(all(
        transfers[f"t{time_value:.3f}"] is not None
        and transfers[f"t{time_value:.3f}"]["maximum"] < 0.01
        for time_value in LATE_TIMES
    ))
    return {
        "amplitude": amplitude,
        "counts": counts,
        "cross_grid_endpoint_transfer": transfers,
        "sampled_onset": onset,
        "all_searches_admissible": valid,
        "late_pair_persistent": late_pair,
        "late_endpoint_transfer_below_1_percent": late_transfer,
    }


def fresh_worker(domain, amplitude):
    domain = domain.upper()
    amplitude = float(amplitude)
    if domain == "R8" and amplitude not in R8_FRESH:
        raise ValueError("fresh R8 worker is sealed only for A=7.84 or 7.86")
    if domain == "R12" and amplitude not in R12_FRESH:
        raise ValueError("fresh R12 worker is sealed only for A=7.84 or 7.92")
    verify_frozen_inputs()
    root = worker_root(domain, amplitude)
    root.mkdir(parents=True, exist_ok=True)
    index = RecoveryIndex(root / "index.json", PROTOCOL, recovery_inputs())
    controls = stage_json(
        index, root, "controls/analytic_BVP", "analytic-control", {},
        lambda: {"control": analytic_controls()}, expected=300.0,
    )["control"]
    print(f"building fresh {domain} {amplitude_tag(amplitude)} G7/G8", flush=True)
    g7, g8 = build_R8_pair(amplitude) if domain == "R8" else build_R12_pair(amplitude)
    geometries = {"G7": g7, "G8": g8}
    initial = {
        label: initial_search_stage(index, root, domain, amplitude, label, geometry)
        for label, geometry in geometries.items()
    }
    final_time = R8_FINAL_TIME if domain == "R8" else R12_FINAL_TIME
    steps = R8_STEPS if domain == "R8" else R12_STEPS
    search_steps = R8_SEARCH_STEPS if domain == "R8" else R12_SEARCH_STEPS
    resumable.FINAL_TIME = final_time
    resumable.STEPS = steps
    resumable.DT = final_time / steps
    resumable.SEGMENT_STEPS = 4 if domain == "R8" else 2
    resumable.RECOVERY_ROOT = root
    cases, runs, segment_paths = {}, {}, {}
    for label, geometry in geometries.items():
        cases[label] = live.setup_case(
            geometry, f"{label}-{amplitude_tag(amplitude)}-{domain}-basin",
            live_normal_wall_gauge=True, live_outer_sommerfeld=True,
        )
        state, diagnostics, paths = resumable.run_grid_evolution(
            index, label, geometry, cases[label],
        )
        runs[label] = resumable.finalize_run(cases[label], state, diagnostics)
        segment_paths[label] = [str(path) for path in paths]
    times = [step * final_time / steps for step in search_steps]
    searches = {label: {} for label in geometries}
    for label, geometry in geometries.items():
        initial_position = np.asarray(geometry["jet_field"].reduced_fields)
        for step, time_value in zip(search_steps, times):
            position, velocity = resumable.load_evolved_step(
                label, step, initial_position,
            )
            searches[label][time_value] = bvp_stage(
                index, root, domain, amplitude, label, step, time_value,
                position, velocity, geometry,
            )
    summary = summarize_searches(amplitude, times, searches)
    initial_zero = bool(all(
        record["static"]["accepted_count"] == 0
        and record["BVP"]["admitted_distinct_count"] == 0
        for record in initial.values()
    ))
    evolution_valid = bool(all(evolution_pass(run) for run in runs.values()))
    final_transfer = {
        name: field_transfer(cases["G7"], runs["G7"], cases["G8"], runs["G8"], key)
        for name, key in (
            ("position_increment", "_increment"), ("velocity", "_velocity"),
            ("source_increment", "_source_increment"),
        )
    }
    field_gate = bool(max(final_transfer.values()) < 0.05)
    construction_valid = construction_pass(geometries)
    primary_pass = bool(
        controls["passed"] and construction_valid and initial_zero
        and evolution_valid and field_gate and summary["all_searches_admissible"]
        and summary["late_pair_persistent"]
        and summary["late_endpoint_transfer_below_1_percent"]
    )
    hard_failure = bool(
        not controls["passed"] or not construction_valid or not evolution_valid
        or not field_gate or not summary["all_searches_admissible"]
    )
    acceptance = {
        "analytic_controls_pass": bool(controls["passed"]),
        "construction_pass": construction_valid,
        "both_initial_static_and_BVP_searches_zero": initial_zero,
        "both_evolutions_pass": evolution_valid,
        "final_cross_grid_fields_below_5_percent": field_gate,
        "all_BVP_searches_admissible": summary["all_searches_admissible"],
        "paired_at_t003_and_t004_on_both_grids": summary["late_pair_persistent"],
        "late_cross_grid_endpoints_below_1_percent":
            summary["late_endpoint_transfer_below_1_percent"],
    }
    payload = {
        "status": "pass" if primary_pass else ("fail" if hard_failure else "review"),
        "classification": (
            "late_pair_persistence_sample" if primary_pass
            else ("invalid_worker" if hard_failure else "sampled_boundary")
        ),
        "protocol": str(PROTOCOL),
        "protocol_sha256": index.protocol_sha256,
        "domain": domain, "amplitude": amplitude,
        "grids": {
            label: {
                "shape": geometry["source_grid"],
                "r_max": float(geometry["r"][-1]),
                "reference_residual": geometry["reference_maximum_residual"],
                "selector_residual": geometry["selector_maximum"],
            } for label, geometry in geometries.items()
        },
        "time_step": final_time / steps, "steps": steps,
        "search_times": times, "BVP_seeds": list(SEEDS),
        "initial_searches": initial, "BVP_searches": searches,
        "search_summary": summary,
        "evolution_diagnostics": {
            label: public_diagnostics(run) for label, run in runs.items()
        },
        "final_field_transfer": final_transfer,
        "acceptance": acceptance,
        "primary_pass": primary_pass, "hard_failure": hard_failure,
        "segment_paths": segment_paths,
        "provenance": recovery_inputs(),
    }
    atomic_write_json(worker_output(domain, amplitude), payload)
    index.register("final/result", "worker-result", 300.0, {
        "domain": domain, "amplitude": amplitude,
    })
    index.mark_running("final/result")
    index.mark_complete("final/result", worker_output(domain, amplitude), 0.0)
    print(json.dumps({
        "worker": worker_name(domain, amplitude), "status": payload["status"],
        "counts": summary["counts"], "sampled_onset": summary["sampled_onset"],
        "acceptance": acceptance,
    }, indent=2), flush=True)
    return payload


def archived_R8_worker(amplitude):
    amplitude = float(amplitude)
    if amplitude not in R8_ARCHIVED:
        raise ValueError("archive worker is sealed only for A=7.88 or 7.92")
    verify_frozen_inputs()
    domain = "R8"
    root = worker_root(domain, amplitude)
    root.mkdir(parents=True, exist_ok=True)
    index = RecoveryIndex(root / "index.json", PROTOCOL, recovery_inputs())
    controls = stage_json(
        index, root, "controls/analytic_BVP", "analytic-control", {},
        lambda: {"control": analytic_controls()}, expected=300.0,
    )["control"]
    g7, g8 = build_R8_pair(amplitude)
    geometries = {"G7": g7, "G8": g8}
    initial = {
        label: initial_search_stage(index, root, domain, amplitude, label, geometry)
        for label, geometry in geometries.items()
    }
    tag = amplitude_tag(amplitude)
    state_path = Path(f"results/corrected_{tag}_formation_pilot_state.npz")
    result_path = Path(f"results/corrected_{tag}_formation_pilot.json")
    archived_result = json.loads(result_path.read_text())
    searches = {label: {} for label in geometries}
    times = list(LATE_TIMES)
    with np.load(state_path) as archive:
        archived_times = np.asarray(archive["sample_times"])
        for label, geometry in geometries.items():
            initial_position = np.asarray(geometry["jet_field"].reduced_fields)
            for time_value in times:
                matches = np.flatnonzero(np.isclose(archived_times, time_value))
                if len(matches) != 1:
                    raise RuntimeError(f"archive has no unique t={time_value}")
                offset = int(matches[0])
                position = initial_position + archive[f"{label}_time_{offset}_increment"]
                velocity = archive[f"{label}_time_{offset}_velocity"]
                step = int(round(time_value / 0.000125))
                searches[label][time_value] = bvp_stage(
                    index, root, domain, amplitude, label, step, time_value,
                    position, velocity, geometry,
                )
    summary = summarize_searches(amplitude, times, searches)
    initial_zero = bool(all(
        record["static"]["accepted_count"] == 0
        and record["BVP"]["admitted_distinct_count"] == 0
        for record in initial.values()
    ))
    prior_numerical = archived_result["acceptance"]
    prior_evolution_valid = bool(
        prior_numerical["both_evolutions_pass"]
        and prior_numerical["final_field_transfer_below_5_percent"]
    )
    prior_onset = (
        json.loads(Path("results/corrected_A788_onset_resolution.json").read_text())["onset"]
        if amplitude == 7.88 else {
            "classification": "sampled_persistent_zero_to_two",
            "bracket": archived_result["formation_bracket"],
        }
    )
    if amplitude == 7.88:
        brackets = prior_onset.get("brackets", {})
        prior_onset = {
            "classification": "sampled_persistent_zero_to_two",
            "bracket": brackets.get("G9") or brackets.get("G8"),
        }
    primary_pass = bool(
        controls["passed"] and construction_pass(geometries) and initial_zero
        and prior_evolution_valid and summary["all_searches_admissible"]
        and summary["late_pair_persistent"]
        and summary["late_endpoint_transfer_below_1_percent"]
    )
    hard_failure = bool(
        not controls["passed"] or not construction_pass(geometries)
        or not prior_evolution_valid or not summary["all_searches_admissible"]
    )
    payload = {
        "status": "pass" if primary_pass else ("fail" if hard_failure else "review"),
        "classification": "archived_evolution_fresh_late_BVP_audit",
        "protocol": str(PROTOCOL), "protocol_sha256": index.protocol_sha256,
        "domain": domain, "amplitude": amplitude,
        "grids": {
            label: {"shape": geometry["source_grid"], "r_max": float(geometry["r"][-1])}
            for label, geometry in geometries.items()
        },
        "search_times": times, "BVP_seeds": list(SEEDS),
        "initial_searches": initial, "BVP_searches": searches,
        "search_summary": {**summary, "sampled_onset": prior_onset},
        "prior_evolution_acceptance": prior_numerical,
        "primary_pass": primary_pass, "hard_failure": hard_failure,
        "provenance": recovery_inputs(),
    }
    atomic_write_json(worker_output(domain, amplitude), payload)
    index.register("final/result", "worker-result", 300.0, {
        "domain": domain, "amplitude": amplitude,
    })
    index.mark_running("final/result")
    index.mark_complete("final/result", worker_output(domain, amplitude), 0.0)
    print(json.dumps({
        "worker": worker_name(domain, amplitude), "status": payload["status"],
        "counts": summary["counts"], "sampled_onset": prior_onset,
    }, indent=2), flush=True)
    return payload


def existing_A790_records():
    fixed = json.loads(Path("results/corrected_A790_two_grid_formation_search.json").read_text())
    bvp = json.loads(Path("results/corrected_A790_independent_dynamic_BVP_detector.json").read_text())
    domain = json.loads(Path("results/corrected_A790_R12_domain_sequence.json").read_text())
    fixed_searches = domain["all_domain_surface_searches"]["R8"]
    fixed_counts = {
        f"{label}_t{time_value:.3f}": fixed_searches[label][f"{time_value:.3f}"]["admitted_distinct_count"]
        for label in ("G7", "G8") for time_value in LATE_TIMES
    }
    fixed_transfers = {
        f"t{time_value:.3f}": endpoint_transfer(
            fixed_searches["G7"][f"{time_value:.3f}"]["admitted_signatures"],
            fixed_searches["G8"][f"{time_value:.3f}"]["admitted_signatures"],
        ) for time_value in LATE_TIMES
    }
    fixed_record = {
        "status": "pass", "classification": "reused_sealed_A790_anchor",
        "domain": "R8", "amplitude": 7.90,
        "search_summary": {
            "counts": fixed_counts,
            "sampled_onset": {
                "classification": "sampled_persistent_zero_to_two",
                "bracket": {"lower": 0.0005, "upper": 0.000625},
            },
            "late_pair_persistent": persistent_late_pair(fixed_counts),
            "late_endpoint_transfer_below_1_percent": all(
                item is not None and item["maximum"] < 0.01
                for item in fixed_transfers.values()
            ),
            "cross_grid_endpoint_transfer": fixed_transfers,
        },
        "primary_pass": bool(
            fixed["acceptance"]["both_initial_static_searches_find_zero_caps"]
            and bvp["acceptance"]["both_initial_slices_admit_zero_candidates"]
            and persistent_late_pair(fixed_counts)
            and all(item is not None and item["maximum"] < 0.01 for item in fixed_transfers.values())
        ),
        "hard_failure": False,
    }
    r12_searches = domain["all_domain_surface_searches"]["R12"]
    r12_counts = {
        f"{label}_t{time_value:.3f}": r12_searches[label][f"{time_value:.3f}"]["admitted_distinct_count"]
        for label in ("G7", "G8") for time_value in LATE_TIMES
    }
    domain_record = {
        "status": "pass", "classification": "reused_sealed_A790_R12_anchor",
        "domain": "R12", "amplitude": 7.90,
        "search_summary": {"counts": r12_counts},
        "primary_pass": bool(
            domain["claim_support"]["pair_existence_and_persistence_domain_asymptotic_support"]
            and persistent_late_pair(r12_counts)
        ),
        "hard_failure": False,
    }
    return fixed_record, domain_record


def aggregate():
    verify_frozen_inputs()
    fixed = {}
    for amplitude in (7.84, 7.86, 7.88, 7.92):
        path = worker_output("R8", amplitude)
        if not path.exists():
            raise FileNotFoundError(f"missing completed worker: {path}")
        fixed[amplitude_tag(amplitude)] = json.loads(path.read_text())
    anchors = {}
    for amplitude in (7.84, 7.92):
        path = worker_output("R12", amplitude)
        if not path.exists():
            raise FileNotFoundError(f"missing completed worker: {path}")
        anchors[amplitude_tag(amplitude)] = json.loads(path.read_text())
    fixed_790, anchor_790 = existing_A790_records()
    fixed["A790"] = fixed_790
    anchors["A790"] = anchor_790
    control = json.loads(
        Path("results/corrected_A790_initial_no_horizon_certificate.json").read_text()
    )["A794_adverse_control"]
    verdict = aggregate_basin_status(fixed, anchors, bool(control["passed"]))
    onset_records = {
        tag: {"sampled_onset": record["search_summary"]["sampled_onset"]}
        for tag, record in fixed.items()
    }
    payload = {
        **verdict,
        "scope": "sealed domain-qualified five-point finite amplitude basin map",
        "protocol": str(PROTOCOL), "protocol_sha256": sha256_file(PROTOCOL),
        "candidate_amplitudes": list(CANDIDATE_AMPLITUDES),
        "adverse_control_amplitude": 7.94,
        "domain_anchors": list(DOMAIN_ANCHORS),
        "fixed_domain_records": fixed,
        "domain_anchor_records": anchors,
        "A794_adverse_control": control,
        "onset_ordering_diagnostic": monotone_onset_diagnostic(onset_records),
        "claim": {
            "if_pass": (
                "five finite Rmax=8 samples show horizonless-start paired late-time "
                "persistence, with direct Rmax=12 support at the lower, center, and upper samples"
            ),
            "prohibited": [
                "continuum or mathematically open basin",
                "domain-converged formation time",
                "global initial horizon nonexistence",
                "event horizon or topology change",
                "connected bulk throat, halo, or mass transfer",
            ],
        },
        "provenance": recovery_inputs(),
        "worker_outputs": {
            f"{domain}_{amplitude_tag(amplitude)}": {
                "path": str(worker_output(domain, amplitude)),
                "sha256": sha256_file(worker_output(domain, amplitude)),
            }
            for domain, amplitudes in (("R8", (7.84, 7.86, 7.88, 7.92)),
                                       ("R12", (7.84, 7.92)))
            for amplitude in amplitudes
        },
    }
    atomic_write_json(OUTPUT, payload)
    print(json.dumps({
        "status": payload["status"], "classification": payload["classification"],
        "fixed": {tag: record["primary_pass"] for tag, record in fixed.items()},
        "anchors": {tag: record["primary_pass"] for tag, record in anchors.items()},
        "onset": payload["onset_ordering_diagnostic"],
    }, indent=2), flush=True)
    return payload


def run_worker(name):
    mapping = {
        "r8_a784": lambda: fresh_worker("R8", 7.84),
        "r8_a786": lambda: fresh_worker("R8", 7.86),
        "r8_a788": lambda: archived_R8_worker(7.88),
        "r8_a792": lambda: archived_R8_worker(7.92),
        "r12_a784": lambda: fresh_worker("R12", 7.84),
        "r12_a792": lambda: fresh_worker("R12", 7.92),
    }
    return mapping[name]()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=(
        "r8_a784", "r8_a786", "r8_a788", "r8_a792",
        "r12_a784", "r12_a792",
    ))
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.plan:
        print(json.dumps({
            "candidate_amplitudes": CANDIDATE_AMPLITUDES,
            "adverse_control": 7.94, "domain_anchors": DOMAIN_ANCHORS,
            "workers": [
                "r8_a784", "r8_a786", "r8_a788", "r8_a792",
                "r12_a784", "r12_a792",
            ],
        }, indent=2))
        return
    if args.worker:
        run_worker(args.worker)
        return
    if args.aggregate:
        aggregate()
        return
    if args.all:
        for name in (
            "r8_a784", "r8_a786", "r8_a788", "r8_a792",
            "r12_a784", "r12_a792",
        ):
            run_worker(name)
        aggregate()
        return
    parser.error("select --worker, --aggregate, --all, or --plan")


if __name__ == "__main__":
    main()
