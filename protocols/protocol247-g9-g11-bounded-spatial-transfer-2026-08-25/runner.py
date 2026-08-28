#!/usr/bin/env python3
"""Recoverable bounded G9/G11 spatial transfer of the outer marginal tube."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

import engine244 as engine
from authority import file_record, sha256, verify_freeze
from transfer_core import adjacent_leaf_transfer, classify


SCHEMA = "protocol247-g9-g11-bounded-spatial-transfer-result-v1"
CHECKPOINT_SCHEMA = "protocol247-grid-checkpoint-v1"
LEAF_SCHEMA = "protocol247-grid-outer-leaf-v1"
DT = 3.125e-5
START_STEP = 32
CHECKPOINT_STEPS = tuple(range(43, 49))
LEAF_STEPS = tuple(range(43, 49))
SOLVE_ORDER = tuple(range(48, 42, -1))
CONTROL_STEP = 48
GRIDS = ("G9", "G11")
FIELDS = ("q", "v", "source", "memory")
OUTPUT = ROOT / "candidate-output"
P228 = ROOT / "sealed-inputs/protocol228"
P244 = ROOT / "sealed-inputs/protocol244"
P246 = ROOT / "sealed-inputs/protocol246"
THREAD_VARS = engine.THREAD_VARS


class Protocol247Error(RuntimeError):
    pass


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def fingerprint(value, prefix):
    return hashlib.sha256(prefix + canonical(value)).hexdigest()


def fingerprinted(value, prefix):
    result = dict(value)
    result["fingerprint"] = fingerprint(value, prefix)
    return result


def runtime_preflight():
    if not sys.dont_write_bytecode or any(os.environ.get(name) != "1" for name in THREAD_VARS):
        raise Protocol247Error("runtime controls differ")
    observed = Path(getattr(sys.modules.get("__main__"), "__file__", "")).resolve()
    if observed != (ROOT / "bootstrap.py").resolve():
        raise Protocol247Error("authorized bootstrap was bypassed")


def grid_bundle(context, label):
    return context["g11_bundle"] if label == "G11" else context["bundles"][label]


def state_shapes(label):
    spatial = {"G9": (113, 211), "G11": (145, 271)}[label]
    return (spatial + (9,), spatial + (9,), spatial + (3,), spatial + (3,))


def load_start(label):
    path = P228 / f"{label}_start_step0032.npz"
    if not engine.regular_immutable_file(path):
        raise Protocol247Error(f"unsafe start input: {label}")
    with np.load(path, allow_pickle=False) as archive:
        if not set(FIELDS) <= set(archive.files):
            raise Protocol247Error(f"start-state inventory differs: {label}")
        arrays = {name: np.ascontiguousarray(archive[name], dtype=np.float64) for name in FIELDS}
    if any(array.shape != shape or not np.all(np.isfinite(array)) for array, shape in zip(arrays.values(), state_shapes(label))):
        raise Protocol247Error(f"start-state shape or values differ: {label}")
    return arrays


def load_terminal(label):
    stem = f"{label}_step0048"
    arrays = engine.load_npz(P228 / f"{stem}.npz", FIELDS)
    receipt = read_json(P228 / f"{stem}.json")
    observed = receipt.pop("fingerprint", None)
    expected = fingerprint(receipt, b"protocol228-checkpoint\0")
    receipt["fingerprint"] = observed
    if not (
        observed == expected
        and receipt.get("authority_sha256") == sha256(P228 / "freeze_record.json")
        and receipt.get("grid") == label
        and receipt.get("end_step") == CONTROL_STEP
        and receipt.get("dt") == DT
        and receipt.get("archive", {}).get("sha256") == sha256(P228 / f"{stem}.npz")
        and receipt.get("passed") is True
    ):
        raise Protocol247Error(f"terminal control receipt differs: {label}")
    return arrays


def load_inputs():
    p228 = read_json(P228 / "protocol228_result.json")
    p244 = read_json(P244 / "protocol244_result.json")
    p246 = read_json(P246 / "protocol246_result.json")
    if p228.get("classification") != "REPAIRED-PARENT-FORMATION-CLOSURE-PASS":
        raise Protocol247Error("Protocol 228 prerequisite differs")
    if p244.get("scientific", {}).get("classification") != "FULL-DT-DENSE-G10-OUTER-MARGINAL-TUBE-PASS":
        raise Protocol247Error("Protocol 244 prerequisite differs")
    if p246.get("scientific", {}).get("classification") != "FULL-HALF-NATIVE-BALANCE-CONSISTENCY-PASS":
        raise Protocol247Error("Protocol 246 prerequisite differs")
    with np.load(P228 / "protocol228_profiles.npz", allow_pickle=False) as archive:
        profiles = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    for label in GRIDS:
        for field in ("theta", "rho", "slope"):
            value = profiles.get(f"{label}_outer_{field}")
            if value is None or value.dtype != np.float64 or value.shape != (501,) or not np.all(np.isfinite(value)):
                raise Protocol247Error(f"terminal profile differs: {label}/{field}")
    starts = {label: load_start(label) for label in GRIDS}
    terminals = {label: load_terminal(label) for label in GRIDS}
    return starts, terminals, profiles, p244


def endpoint_semantics(context, label, step, arrays):
    bundle = grid_bundle(context, label)
    state = tuple(arrays[name] for name in FIELDS)
    records = []
    for _ in range(2):
        stage = context["runner"].reconstruct_driver_stage(
            bundle, context["p228"].MODE, step * DT, state, capture=True,
        )
        audit = context["runner"]._technical_stage_audit(bundle, context["p228"].MODE, stage)
        gates = context["p220"].P190.jsonable(audit["gates"])
        records.append((bool(stage["finite"] and gates["all_technical_gates_pass"]), gates))
    if records[0] != records[1]:
        raise Protocol247Error(f"endpoint replay differs: {label}/{step}")
    return records[0]


def checkpoint_name(label, step):
    return f"{label}_full_step{step:04d}"


def checkpoint_prefix():
    return b"protocol247-grid-checkpoint-v1\0"


def validate_checkpoint(context, authority_sha, label, step, previous_step, previous_sha, terminal):
    stem = checkpoint_name(label, step)
    archive_path = OUTPUT / f"{stem}.npz"
    receipt_path = OUTPUT / f"{stem}.json"
    arrays = engine.load_npz(archive_path, FIELDS)
    passed, gates = endpoint_semantics(context, label, step, arrays)
    bitwise = step != CONTROL_STEP or all(np.array_equal(arrays[name], terminal[name]) for name in FIELDS)
    bare = {
        "schema": CHECKPOINT_SCHEMA,
        "authority_sha256": authority_sha,
        "grid": label,
        "start_step": previous_step,
        "end_step": step,
        "start_time": previous_step * DT,
        "end_time": step * DT,
        "dt": DT,
        "mode": "legacy_wall_axis_outer",
        "previous_checkpoint_sha256": previous_sha,
        "archive": file_record(archive_path, ROOT),
        "arrays": {name: engine.array_record(arrays[name]) for name in sorted(arrays)},
        "endpoint_gates": gates,
        "endpoint_repeat_exact": True,
        "published_parent_control": step == CONTROL_STEP,
        "published_parent_control_bitwise_replay": bitwise,
        "passed": bool(passed and bitwise),
    }
    expected = fingerprinted(bare, checkpoint_prefix())
    if not receipt_path.exists():
        if not bare["passed"]:
            raise Protocol247Error(f"checkpoint semantics failed: {label}/{step}")
        engine.atomic_json(receipt_path, expected)
    if read_json(receipt_path) != expected or not bare["passed"]:
        raise Protocol247Error(f"checkpoint receipt differs: {label}/{step}")
    return arrays, sha256(receipt_path), bitwise


def publish_checkpoint(context, authority_sha, label, step, previous_step, previous_state, previous_sha, terminal):
    stem = checkpoint_name(label, step)
    archive_path = OUTPUT / f"{stem}.npz"
    receipt_path = OUTPUT / f"{stem}.json"
    if receipt_path.exists() and not archive_path.exists():
        raise Protocol247Error(f"checkpoint receipt without archive: {label}/{step}")
    if not archive_path.exists():
        bundle = grid_bundle(context, label)
        validator = lambda stage: context["runner"]._technical_stage_audit(bundle, context["p228"].MODE, stage)
        print(f"{label}: evolving {previous_step * DT:.7f} -> {step * DT:.7f}", flush=True)
        segment = context["p220"].P216.run_half(
            bundle, context["runner"], context["p228"].MODE, DT,
            step - previous_step, tuple(previous_state[name] for name in FIELDS),
            previous_step + 1, validator,
        )
        if not (
            segment["completed"] and len(segment["records"]) == 2 * (step - previous_step)
            and all(item["finite"] for item in segment["records"])
            and all(item["technical_audit"]["gates"]["all_technical_gates_pass"] for item in segment["records"])
        ):
            raise Protocol247Error(f"evolution segment failed: {label}/{step}")
        arrays = {name: np.ascontiguousarray(segment["end_state"][index]) for index, name in enumerate(FIELDS)}
        engine.atomic_npz(archive_path, arrays)
    return validate_checkpoint(context, authority_sha, label, step, previous_step, previous_sha, terminal)


def profile_name(label, step):
    return f"{label}_outer_step{step:04d}"


def parent_profile(label, profiles):
    return {field: np.ascontiguousarray(profiles[f"{label}_outer_{field}"]) for field in ("theta", "rho", "slope")}


def control_comparison(label, step, profile, profiles):
    if step != CONTROL_STEP:
        return {"is_control": False, "passed": True}
    parent = parent_profile(label, profiles)
    rho = float(np.max(np.abs(profile["rho"] - parent["rho"])))
    slope = float(np.max(np.abs(profile["slope"] - parent["slope"])))
    endpoint = float(np.hypot(profile["rho"][0] - parent["rho"][0], profile["rho"][-1] - parent["rho"][-1]))
    return {
        "is_control": True,
        "rho_maximum_absolute_difference": rho,
        "slope_maximum_absolute_difference": slope,
        "endpoint_euclidean_difference": endpoint,
        "passed": bool(rho < 1e-4 and slope < 1e-3 and endpoint < 1e-4),
    }


def leaf_prefix():
    return b"protocol247-grid-outer-leaf-v1\0"


def evaluate_leaf(label, step, state, profile, profiles, solver_public):
    old = engine.control_comparison
    try:
        engine.control_comparison = lambda observed_step, observed_profile, _ignored: control_comparison(
            label, observed_step, observed_profile, profiles,
        )
        return engine.evaluate_leaf(step, state, profile, profiles, solver_public)
    finally:
        engine.control_comparison = old


def validate_leaf(label, step, state, authority_sha, profiles):
    stem = profile_name(label, step)
    archive_path = OUTPUT / f"{stem}.npz"
    receipt_path = OUTPUT / f"{stem}.json"
    profile = engine.profile_arrays(archive_path)
    existing = read_json(receipt_path) if receipt_path.exists() else None
    solver_public = None if existing is None else existing.get("evaluation", {}).get("surface")
    if solver_public is None:
        z = np.linspace(1.0, np.e, state["q"].shape[0])
        r = np.linspace(0.0, 10.0, state["q"].shape[1])
        prepared = engine.prepare_capped_expansion_slice(state["q"], state["v"], z, r, stencil_width=7)
        first = engine.solve_dynamical_capped_surface_bvp(
            state["q"], state["v"], z, r, profile, tolerance=2e-6,
            nodes=121, maximum_nodes=6000, dense_nodes=501, prepared=prepared,
        )
        second = engine.solve_dynamical_capped_surface_bvp(
            state["q"], state["v"], z, r, profile, tolerance=2e-6,
            nodes=121, maximum_nodes=6000, dense_nodes=501, prepared=prepared,
        )
        if any(not np.array_equal(first[field], second[field]) for field in ("theta", "rho", "slope")):
            raise Protocol247Error(f"leaf replay differs: {label}/{step}")
        solver_public = engine.surface_tools.public_surface(first)
    metadata, derived = evaluate_leaf(label, step, state, profile, profiles, solver_public)
    bare = {
        "schema": LEAF_SCHEMA,
        "authority_sha256": authority_sha,
        "grid": label,
        "step": step,
        "time_over_ell": step * DT,
        "solve_repeat_exact": True,
        "archive": file_record(archive_path, ROOT),
        "profile_arrays": {name: engine.array_record(profile[name]) for name in sorted(profile)},
        "derived_array_records": {name: engine.array_record(value) for name, value in sorted(derived.items())},
        "evaluation": metadata,
        "passed": metadata["passed"],
    }
    expected = fingerprinted(bare, leaf_prefix())
    if not receipt_path.exists():
        engine.atomic_json(receipt_path, expected)
    if read_json(receipt_path) != expected or not metadata["passed"]:
        raise Protocol247Error(f"leaf receipt differs: {label}/{step}")
    return metadata, derived


def publish_leaf(label, step, state, seed, authority_sha, profiles):
    stem = profile_name(label, step)
    archive_path = OUTPUT / f"{stem}.npz"
    receipt_path = OUTPUT / f"{stem}.json"
    if receipt_path.exists() and not archive_path.exists():
        raise Protocol247Error(f"leaf receipt without archive: {label}/{step}")
    if not archive_path.exists():
        z = np.linspace(1.0, np.e, state["q"].shape[0])
        r = np.linspace(0.0, 10.0, state["q"].shape[1])
        prepared = engine.prepare_capped_expansion_slice(state["q"], state["v"], z, r, stencil_width=7)
        print(f"{label}: outer leaf step {step} at t/ell={step * DT:.7f}", flush=True)
        first = engine.solve_dynamical_capped_surface_bvp(
            state["q"], state["v"], z, r, seed, tolerance=2e-6,
            nodes=121, maximum_nodes=6000, dense_nodes=501, prepared=prepared,
        )
        second = engine.solve_dynamical_capped_surface_bvp(
            state["q"], state["v"], z, r, seed, tolerance=2e-6,
            nodes=121, maximum_nodes=6000, dense_nodes=501, prepared=prepared,
        )
        arrays = {field: np.ascontiguousarray(first[field]) for field in ("theta", "rho", "slope")}
        if any(not np.array_equal(arrays[field], second[field]) for field in arrays):
            raise Protocol247Error(f"leaf solve repeat differs: {label}/{step}")
        engine.atomic_npz(archive_path, arrays)
    return validate_leaf(label, step, state, authority_sha, profiles)


def local_evaluation(label, leaves, derived, states):
    old_steps, old_dt = engine.LEAF_STEPS, engine.FULL_DT
    try:
        engine.LEAF_STEPS = LEAF_STEPS
        engine.FULL_DT = DT
        evaluation, arrays = engine.dense_evaluation(leaves, derived, states)
    finally:
        engine.LEAF_STEPS, engine.FULL_DT = old_steps, old_dt
    local_pass = bool(
        evaluation["all_surfaces_and_stability_pass"]
        and evaluation["all_inward_expansions_resolved_negative"]
        and evaluation["area_increase"]["passed"]
        and evaluation["all_interior_tube_signatures_resolved_spacelike"]
    )
    return {"grid": label, "passed": local_pass, **evaluation}, arrays


def g10_reference(p244):
    evaluation = p244["scientific"]["evaluation"]
    return {
        "grid": "G10",
        "passed": all(p244["scientific"]["gates"].values()),
        "leaves": {str(step): evaluation["leaves"][str(step)] for step in LEAF_STEPS},
        "tube": {str(step): evaluation["tube"][str(step)] for step in range(44, 48)},
        "area_increase": {
            "passed": all(
                evaluation["leaves"][str(right)]["geometry"]["one_sided_cap_area"]
                > evaluation["leaves"][str(left)]["geometry"]["one_sided_cap_area"]
                for left, right in zip(LEAF_STEPS[:-1], LEAF_STEPS[1:])
            ),
        },
        "all_interior_tube_signatures_resolved_spacelike": all(
            evaluation["tube"][str(step)]["label"] == "UNIFORMLY-SPACELIKE-SPARSE-PILOT"
            and evaluation["tube"][str(step)]["resolved_fraction"] == 1.0
            for step in range(44, 48)
        ),
    }


def final_prefix():
    return b"protocol247-result-v1\0"


def finalize(authority_sha, local, derived_by_grid, tube_arrays_by_grid, p244, terminal_replays, peak, started):
    all_grids = {"G9": local["G9"], "G10": g10_reference(p244), "G11": local["G11"]}
    transfers = {}
    for pair in (("G9", "G10"), ("G10", "G11")):
        key = "-".join(pair)
        transfers[key] = {
            str(step): adjacent_leaf_transfer(
                all_grids[pair[0]]["leaves"][str(step)],
                all_grids[pair[1]]["leaves"][str(step)],
            ) for step in LEAF_STEPS
        }
    local_pass = all(local[label]["passed"] for label in GRIDS)
    terminal_pass = all(terminal_replays.values()) and all(
        local[label]["leaves"][str(CONTROL_STEP)]["control_comparison"]["passed"] for label in GRIDS
    )
    transfer_pass = all(item["passed"] for pair in transfers.values() for item in pair.values())
    causal_pass = all(
        all_grids[label]["all_interior_tube_signatures_resolved_spacelike"] for label in ("G9", "G10", "G11")
    )
    classification, gates = classify(local_pass, transfer_pass, terminal_pass, causal_pass)
    arrays = {}
    for label in GRIDS:
        for step in LEAF_STEPS:
            for name, value in derived_by_grid[label][step].items():
                arrays[f"{label}_step{step:04d}_{name}"] = value
        for name, value in tube_arrays_by_grid[label].items():
            arrays[f"{label}_{name}"] = value
    archive_path = OUTPUT / "protocol247_spatial_transfer_arrays.npz"
    if not archive_path.exists():
        engine.atomic_npz(archive_path, arrays)
    else:
        observed = engine.load_npz(archive_path, tuple(arrays))
        if any(not np.array_equal(observed[name], arrays[name]) for name in arrays):
            raise Protocol247Error("final array archive differs")
    scientific = {
        "classification": classification,
        "gates": gates,
        "dt": DT,
        "start_step": START_STEP,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "leaf_steps": list(LEAF_STEPS),
        "interior_tube_steps": list(range(44, 48)),
        "terminal_checkpoint_bitwise_replays": terminal_replays,
        "grids": all_grids,
        "adjacent_grid_transfers": transfers,
    }
    bare = {
        "schema": SCHEMA,
        "authority_sha256": authority_sha,
        "scientific": scientific,
        "archive": file_record(archive_path, ROOT),
        "array_records": {name: engine.array_record(value) for name, value in sorted(arrays.items())},
        "resource": {
            "peak_rss_bytes": max(peak, engine.memory_gate()),
            "soft_rss_stop_bytes": engine.SOFT_RSS_BYTES,
            "elapsed_wall_seconds": float(time.monotonic() - started),
        },
        "spacetime_evolution_executed": True,
        "new_parent_solve_executed": False,
        "submitted_paper_edited": False,
        "parent_or_published_artifact_modified": False,
        "archive_only_balance_transfer_authorized": classification == "G9-G10-G11-BOUNDED-OUTER-TUBE-SPATIAL-TRANSFER-PASS",
        "continuum_dynamical_horizon_claim_authorized": False,
        "event_horizon_claim_authorized": False,
        "source_ownership_claim_authorized": False,
    }
    expected = fingerprinted(bare, final_prefix())
    result_path = OUTPUT / "protocol247_result.json"
    if not result_path.exists():
        engine.atomic_json(result_path, expected)
    if read_json(result_path) != expected:
        raise Protocol247Error("completed result differs")
    return expected


def execute(academic_root, visual_root, project_root):
    started = time.monotonic()
    runtime_preflight()
    verify_freeze(ROOT)
    authority_sha = sha256(ROOT / "authority/freeze_record.json")
    starts, terminals, profiles, p244 = load_inputs()
    context, old_crossfit = engine.load_evolution_context(academic_root, visual_root, project_root)
    peak = engine.memory_gate()
    try:
        with context["p220"].exclusive_lock(ROOT):
            if not OUTPUT.exists():
                OUTPUT.mkdir(mode=0o755)
                engine.fsync_directory(ROOT)
            states_by_grid = {}
            leaves_by_grid = {}
            derived_by_grid = {}
            tube_arrays_by_grid = {}
            terminal_replays = {}
            for label in GRIDS:
                states = {START_STEP: starts[label]}
                previous_step = START_STEP
                previous_state = starts[label]
                previous_sha = sha256(P228 / f"{label}_start_step0032.npz")
                replay = False
                for step in CHECKPOINT_STEPS:
                    state, previous_sha, bitwise = publish_checkpoint(
                        context, authority_sha, label, step, previous_step,
                        previous_state, previous_sha, terminals[label],
                    )
                    states[step] = state
                    previous_step, previous_state = step, state
                    if step == CONTROL_STEP:
                        replay = bitwise
                    peak = max(peak, engine.memory_gate())
                leaves, derived = {}, {}
                for step in SOLVE_ORDER:
                    seed = parent_profile(label, profiles) if step == CONTROL_STEP else engine.profile_arrays(
                        OUTPUT / f"{profile_name(label, step + 1)}.npz"
                    )
                    leaves[step], derived[step] = publish_leaf(
                        label, step, states[step], seed, authority_sha, profiles,
                    )
                    peak = max(peak, engine.memory_gate())
                local, tube_arrays = local_evaluation(label, leaves, derived, states)
                states_by_grid[label] = states
                leaves_by_grid[label] = local
                derived_by_grid[label] = derived
                tube_arrays_by_grid[label] = tube_arrays
                terminal_replays[label] = replay
            result = finalize(
                authority_sha, leaves_by_grid, derived_by_grid, tube_arrays_by_grid,
                p244, terminal_replays, peak, started,
            )
            verify_freeze(ROOT)
            return result
    finally:
        context["runner"].axis_even_crossfit_audit = old_crossfit


def status():
    result_path = OUTPUT / "protocol247_result.json"
    checkpoints = {
        label: [step for step in CHECKPOINT_STEPS if (OUTPUT / f"{checkpoint_name(label, step)}.json").is_file()]
        for label in GRIDS
    } if OUTPUT.exists() else {label: [] for label in GRIDS}
    leaves = {
        label: [step for step in SOLVE_ORDER if (OUTPUT / f"{profile_name(label, step)}.json").is_file()]
        for label in GRIDS
    } if OUTPUT.exists() else {label: [] for label in GRIDS}
    result = read_json(result_path) if result_path.is_file() else None
    return {
        "schema": "protocol247-status-v1",
        "completed_checkpoint_steps": checkpoints,
        "completed_leaf_steps_in_solve_order": leaves,
        "final_result_present": result is not None,
        "classification": None if result is None else result["scientific"]["classification"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--academic-root", required=True)
    run.add_argument("--visual-root", required=True)
    run.add_argument("--project-root", required=True)
    commands.add_parser("verify")
    commands.add_parser("status")
    values = vars(parser.parse_args(argv))
    command = values.pop("command")
    if command == "run":
        result = execute(**values)
    elif command == "verify":
        runtime_preflight()
        authority = verify_freeze(ROOT)
        load_inputs()
        result = {"status": "VERIFIED", "authority_sha256": sha256(ROOT / "authority/freeze_record.json"), "fingerprint": authority["fingerprint"]}
    else:
        result = status()
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
