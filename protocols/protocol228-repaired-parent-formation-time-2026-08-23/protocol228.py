#!/usr/bin/env python3
"""Recoverable repaired-parent later-time MOTS formation search."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SCHEMA = "protocol228-repaired-parent-formation-time-v1"
CHECKPOINT_SCHEMA = "protocol228-evolution-checkpoint-v1"
DETECTOR_SCHEMA = "protocol228-pilot-detector-v1"
MODE = "legacy_wall_axis_outer"
DT = 0.00003125
START_STEP = 32
SEGMENT_STEPS = 16
SAMPLE_STEPS = tuple(range(48, 257, 16))
FIELDS = ("q", "v", "source", "memory")
GRIDS = ("G9", "G10", "G11")
THREAD_VARS = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)


class Protocol228Error(RuntimeError):
    pass


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path, root=None):
    path = Path(path).absolute()
    return {
        "path": path.relative_to(Path(root).absolute()).as_posix() if root else str(path),
        "byte_count": path.stat().st_size,
        "sha256": sha256(path),
    }


def same(path, item):
    path = Path(path)
    return bool(
        set(item) == {"path", "byte_count", "sha256"}
        and path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
        and type(item["byte_count"]) is int and path.stat().st_size == item["byte_count"]
        and type(item["sha256"]) is str and sha256(path) == item["sha256"]
    )


def fsync_dir(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    if path.exists() or path.is_symlink():
        raise Protocol228Error("output exists: " + path.name)
    with temporary.open("xb") as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    os.replace(temporary, path); fsync_dir(path.parent)


def atomic_npz(path, arrays):
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    if path.exists() or path.is_symlink():
        raise Protocol228Error("output exists: " + path.name)
    with temporary.open("xb") as stream:
        np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    os.replace(temporary, path); fsync_dir(path.parent)


def protocol_roots(academic_root):
    base = Path(academic_root).absolute() / "Discussion/protocols"
    return {
        "p226": base / "protocol226-corrected-canonical-g11-2026-08-23",
        "p227": base / "protocol227-direct-g11-mots-observables-2026-08-23",
    }


def input_paths(academic_root):
    roots = protocol_roots(academic_root)
    return {
        "p226/protocol": roots["p226"] / "PROTOCOL.md",
        "p226/source": roots["p226"] / "protocol226.py",
        "p226/authority": roots["p226"] / "authority/freeze_record.json",
        "p226/result": roots["p226"] / "candidate-output/protocol226_result.json",
        "p227/protocol": roots["p227"] / "PROTOCOL.md",
        "p227/source": roots["p227"] / "protocol227.py",
        "p227/authority": roots["p227"] / "authority/freeze_record.json",
        "p227/result": roots["p227"] / "candidate-output/protocol227_result.json",
        "G9/endpoint": roots["p227"].parents[0] / "protocol220-recoverable-canonical-spatial-2026-08-22/candidate-output/G9_repeat1.npz",
        "G10/endpoint": roots["p227"].parents[0] / "protocol216-doubled-horizon-canonical-2026-08-22/linux-output/g10_doubled_horizon_canonical.npz",
        "G11/endpoint": roots["p226"] / "candidate-output/G11_repeat1.npz",
    }


def freeze(root, academic_root):
    root = Path(root).absolute()
    authority_dir = root / "authority"
    if any(authority_dir.iterdir()) or (root / "candidate-output").exists():
        raise Protocol228Error("prospective namespace differs")
    inputs = input_paths(academic_root)
    for name, path in inputs.items():
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
            raise Protocol228Error("unsafe input: " + name)
    p226_result = read_json(inputs["p226/result"])
    p227_result = read_json(inputs["p227/result"])
    if p226_result.get("classification") != "CANONICAL-G11-SPATIAL-CLOSURE-PASS":
        raise Protocol228Error("Protocol226 prerequisite differs")
    if not (
        p227_result.get("classification") == "DIRECT-MOTS-OBSERVABLE-REVIEW"
        and p227_result.get("time_over_ell") == 0.001
        and all(item.get("distinct_cluster_count") == 0 for item in p227_result.get("records", {}).values())
    ):
        raise Protocol228Error("Protocol227 prerequisite differs")
    authority = {
        "schema": SCHEMA,
        "status": "FROZEN",
        "mode": MODE,
        "dt": DT,
        "start_step": START_STEP,
        "sample_steps": list(SAMPLE_STEPS),
        "sample_times": [step * DT for step in SAMPLE_STEPS],
        "pilot_grid": "G10",
        "candidate_rule": "earliest prescribed G10 checkpoint with exactly two admitted clusters",
        "common_grid_plan": list(GRIDS),
        "detector_inherited_without_threshold_change": True,
        "sources": {
            name: record(root / name, root)
            for name in ("PROTOCOL.md", "protocol228.py", "bootstrap.py", "tests/test_protocol228.py")
        },
        "inputs": {name: record(path) for name, path in inputs.items()},
        "candidate_output_absent_at_freeze": True,
        "later_checkpoint_substitution_authorized": False,
        "parent_solve_or_repair_authorized": False,
        "event_horizon_claim_authorized": False,
        "continuum_order_claim_authorized": False,
        "phase_selection_claim_authorized": False,
        "source_ownership_claim_authorized": False,
    }
    authority["fingerprint"] = hashlib.sha256(b"protocol228-freeze\0" + canonical(authority)).hexdigest()
    atomic_json(authority_dir / "freeze_record.json", authority)
    return authority


def verify(root, academic_root):
    root = Path(root).absolute()
    authority = read_json(root / "authority/freeze_record.json")
    fingerprint = authority.pop("fingerprint", None)
    expected = hashlib.sha256(b"protocol228-freeze\0" + canonical(authority)).hexdigest()
    authority["fingerprint"] = fingerprint
    if authority.get("schema") != SCHEMA or fingerprint != expected:
        raise Protocol228Error("authority differs")
    for name, item in authority["sources"].items():
        if not same(root / item["path"], item):
            raise Protocol228Error("source differs: " + name)
    inputs = input_paths(academic_root)
    if set(inputs) != set(authority["inputs"]):
        raise Protocol228Error("input inventory differs")
    for name, item in authority["inputs"].items():
        if not same(inputs[name], item):
            raise Protocol228Error("input differs: " + name)
    return authority, inputs


def runtime_preflight(root):
    if not sys.dont_write_bytecode:
        raise Protocol228Error("Python bytecode must be disabled")
    if any(os.environ.get(name) != "1" for name in THREAD_VARS):
        raise Protocol228Error("thread controls differ")
    bootstrap = (Path(root) / "bootstrap.py").resolve()
    main_file = Path(getattr(sys.modules.get("__main__"), "__file__", "")).resolve()
    if main_file != bootstrap:
        raise Protocol228Error("authorized bootstrap was bypassed")


def load_modules_and_context(root, academic_root, visual_root, project_root, inputs):
    p226 = load(inputs["p226/source"], "protocol226_bound228")
    p227 = load(inputs["p227/source"], "protocol227_bound228")
    _, p226_inputs = p226.verify(protocol_roots(academic_root)["p226"], academic_root)
    p227.verify_frozen_inputs()
    context = p226.load_context(root, academic_root, visual_root, project_root, p226_inputs)
    context.update({"p226": p226, "p227": p227})
    return context


def load_initial_state(label, path):
    with np.load(path, allow_pickle=False) as archive:
        prefix = "endpoint_" if label == "G10" else ""
        required = {prefix + name for name in FIELDS}
        if not required <= set(archive.files):
            raise Protocol228Error(label + " endpoint inventory differs")
        state = tuple(np.ascontiguousarray(archive[prefix + name], dtype=np.float64) for name in FIELDS)
    expected = expected_state_shapes(label)
    if any(array.shape != shape or not np.all(np.isfinite(array)) for array, shape in zip(state, expected)):
        raise Protocol228Error(label + " endpoint differs")
    return state


def expected_state_shapes(label):
    spatial = {"G9": (113, 211), "G10": (129, 241), "G11": (145, 271)}[label]
    return (spatial + (9,), spatial + (9,), spatial + (3,), spatial + (3,))


def grid_coordinates(state):
    shape = state[0].shape
    return np.linspace(1.0, math.e, shape[0]), np.linspace(0.0, 10.0, shape[1])


def array_manifest(context, arrays):
    return {name: context["p220"].P211.array_record(value) for name, value in sorted(arrays.items())}


def checkpoint_name(label, step):
    return f"{label}_step{step:04d}"


def checkpoint_fingerprint(value):
    return hashlib.sha256(b"protocol228-checkpoint\0" + canonical(value)).hexdigest()


def detector_fingerprint(value):
    return hashlib.sha256(b"protocol228-detector\0" + canonical(value)).hexdigest()


def result_fingerprint(value):
    return hashlib.sha256(b"protocol228-result\0" + canonical(value)).hexdigest()


def load_checkpoint_archive(path):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(FIELDS):
            raise Protocol228Error("checkpoint array inventory differs")
        return {name: np.ascontiguousarray(archive[name]) for name in FIELDS}


def endpoint_semantics(context, label, step, arrays):
    bundle = context["g11_bundle"] if label == "G11" else context["bundles"][label]
    state = tuple(arrays[name] for name in FIELDS)
    endpoint = context["runner"].reconstruct_driver_stage(bundle, MODE, step * DT, state, capture=True)
    audit = context["runner"]._technical_stage_audit(bundle, MODE, endpoint)
    return bool(endpoint["finite"] and audit["gates"]["all_technical_gates_pass"]), context["p220"].P190.jsonable(audit["gates"])


def validate_checkpoint(output, name, context, authority_sha, label, start_step, end_step, previous_sha):
    archive_path, receipt_path = output / (name + ".npz"), output / (name + ".json")
    if not archive_path.is_file() or archive_path.is_symlink() or archive_path.stat().st_nlink != 1:
        raise Protocol228Error("checkpoint archive differs: " + name)
    arrays = load_checkpoint_archive(archive_path)
    passed, gates = endpoint_semantics(context, label, end_step, arrays)
    if not passed:
        raise Protocol228Error("recovered endpoint gate failed: " + name)
    if not receipt_path.exists():
        receipt = {
            "schema": CHECKPOINT_SCHEMA, "authority_sha256": authority_sha,
            "grid": label, "start_step": start_step, "end_step": end_step,
            "start_time": start_step * DT, "end_time": end_step * DT,
            "mode": MODE, "dt": DT, "previous_checkpoint_sha256": previous_sha,
            "archive": record(archive_path, ROOT), "arrays": array_manifest(context, arrays),
            "endpoint_gates": gates, "passed": True,
        }
        receipt["fingerprint"] = checkpoint_fingerprint(receipt)
        atomic_json(receipt_path, receipt)
    receipt = read_json(receipt_path)
    fingerprint = receipt.pop("fingerprint", None)
    expected = checkpoint_fingerprint(receipt); receipt["fingerprint"] = fingerprint
    if not (
        fingerprint == expected and receipt.get("schema") == CHECKPOINT_SCHEMA
        and receipt.get("authority_sha256") == authority_sha and receipt.get("grid") == label
        and receipt.get("start_step") == start_step and receipt.get("end_step") == end_step
        and receipt.get("previous_checkpoint_sha256") == previous_sha
        and receipt.get("archive") == record(archive_path, ROOT)
        and receipt.get("arrays") == array_manifest(context, arrays) and receipt.get("passed") is True
    ):
        raise Protocol228Error("checkpoint receipt differs: " + name)
    return tuple(arrays[name] for name in FIELDS), sha256(receipt_path)


def publish_checkpoint(output, context, authority_sha, label, start_step, end_step, start_state, previous_sha):
    name = checkpoint_name(label, end_step)
    archive_path, receipt_path = output / (name + ".npz"), output / (name + ".json")
    if receipt_path.exists() and not archive_path.exists():
        raise Protocol228Error("receipt without checkpoint: " + name)
    if not archive_path.exists():
        bundle = context["g11_bundle"] if label == "G11" else context["bundles"][label]
        validator = lambda stage: context["runner"]._technical_stage_audit(bundle, MODE, stage)
        print(f"{label}: evolving {start_step * DT:.7f} -> {end_step * DT:.7f}", flush=True)
        segment = context["p220"].P216.run_half(
            bundle, context["runner"], MODE, DT, end_step - start_step,
            start_state, start_step + 1, validator,
        )
        if not (
            segment["completed"] and len(segment["records"]) == 2 * (end_step - start_step)
            and all(item["finite"] for item in segment["records"])
            and all(item["technical_audit"]["gates"]["all_technical_gates_pass"] for item in segment["records"])
        ):
            raise Protocol228Error("evolution segment gate failed: " + name)
        arrays = {field: np.ascontiguousarray(segment["end_state"][index]) for index, field in enumerate(FIELDS)}
        atomic_npz(archive_path, arrays)
    return validate_checkpoint(output, name, context, authority_sha, label, start_step, end_step, previous_sha)


def detector_scan(context, label, state):
    p227 = context["p227"]
    position, velocity = state[0], state[1]
    z, r = grid_coordinates(state)
    prepared = p227.prepare_capped_expansion_slice(position, velocity, z, r)
    trials = []
    for seed in p227.SEEDS:
        print(f"{label}: BVP seed {seed:.2f}", flush=True)
        try:
            surface = p227.solve_dynamical_capped_surface_bvp(
                position, velocity, z, r, seed, tolerance=2e-5,
                nodes=121, maximum_nodes=6000, dense_nodes=501, prepared=prepared,
            )
            trials.append({"seed": seed, "admitted": p227.admitted(surface), "surface": surface})
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            trials.append({"seed": seed, "admitted": False, "error": f"{type(error).__name__}: {error}"})
    clusters = p227.cluster_trials(trials)
    return {
        "trial_count": len(trials),
        "admitted_trial_count": sum(item["admitted"] for item in trials),
        "distinct_cluster_count": len(clusters),
        "cluster_signatures": [item["signature"] for item in clusters],
        "trials": [
            ({"seed": item["seed"], "admitted": item["admitted"], "surface": p227.public_surface(item["surface"])}
             if "surface" in item else item)
            for item in trials
        ],
    }


def publish_detector(output, context, authority_sha, step, state, checkpoint_receipt_sha):
    path = output / f"G10_step{step:04d}_detector.json"
    if not path.exists():
        scan = context["p220"].P190.jsonable(detector_scan(context, "G10", state))
        value = {
            "schema": DETECTOR_SCHEMA, "authority_sha256": authority_sha,
            "grid": "G10", "step": step, "time_over_ell": step * DT,
            "checkpoint_receipt_sha256": checkpoint_receipt_sha,
            "scan": scan, "candidate": scan["distinct_cluster_count"] == 2,
        }
        value["fingerprint"] = detector_fingerprint(value)
        atomic_json(path, value)
    value = read_json(path); fingerprint = value.pop("fingerprint", None)
    expected = detector_fingerprint(value); value["fingerprint"] = fingerprint
    if not (
        fingerprint == expected and value.get("schema") == DETECTOR_SCHEMA
        and value.get("authority_sha256") == authority_sha and value.get("step") == step
        and value.get("checkpoint_receipt_sha256") == checkpoint_receipt_sha
        and value.get("candidate") is (value.get("scan", {}).get("distinct_cluster_count") == 2)
    ):
        raise Protocol228Error("detector receipt differs at step " + str(step))
    return value


def full_observables(context, label, state):
    p227 = context["p227"]
    position, velocity = state[0], state[1]
    z, r = grid_coordinates(state)
    prepared = p227.prepare_capped_expansion_slice(position, velocity, z, r)
    trials = []
    for seed in p227.SEEDS:
        print(f"{label}: final BVP seed {seed:.2f}", flush=True)
        try:
            surface = p227.solve_dynamical_capped_surface_bvp(
                position, velocity, z, r, seed, tolerance=2e-5,
                nodes=121, maximum_nodes=6000, dense_nodes=501, prepared=prepared,
            )
            trials.append({"seed": seed, "admitted": p227.admitted(surface), "surface": surface})
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            trials.append({"seed": seed, "admitted": False, "error": f"{type(error).__name__}: {error}"})
    clusters = p227.cluster_trials(trials)
    branches, arrays = {}, {}
    for index, branch in enumerate(("inner", "outer")):
        if index >= len(clusters):
            continue
        representative = min(clusters[index]["members"], key=lambda item: item["surface"]["local_expansion_interior_maximum"])
        surface = representative["surface"]
        geometry = p227.capped_surface_geometry(position, velocity, z, r, surface, prepared=prepared)
        stability = p227.stability_series(position, velocity, z, r, surface, prepared)
        branches[branch] = {
            "representative_seed": representative["seed"],
            "cluster_signature": clusters[index]["signature"],
            "cluster_seed_count": len(clusters[index]["members"]),
            "surface": p227.public_surface(surface), "geometry": geometry, "stability": stability,
        }
        arrays[f"{label}_{branch}_theta"] = np.asarray(surface["theta"])
        arrays[f"{label}_{branch}_rho"] = np.asarray(surface["rho"])
        arrays[f"{label}_{branch}_slope"] = np.asarray(surface["slope"])
    return context["p220"].P190.jsonable({
        "trial_count": len(trials), "admitted_trial_count": sum(item["admitted"] for item in trials),
        "distinct_cluster_count": len(clusters), "branches": branches,
    }), arrays


def acceptance(records, transfers):
    count = all(records[label]["distinct_cluster_count"] == 2 for label in GRIDS)
    geometry = count and all(
        branch["geometry"]["finite"] and branch["geometry"]["one_sided_cap_area"] > 0
        for grid in records.values() for branch in grid["branches"].values()
    ) and all(
        grid["branches"]["outer"]["geometry"]["one_sided_cap_area"]
        > grid["branches"]["inner"]["geometry"]["one_sided_cap_area"]
        for grid in records.values()
    )
    classification = count and all(
        grid["branches"]["inner"]["stability"]["classification"] == "outward-unstable"
        and grid["branches"]["outer"]["stability"]["classification"] == "outward-stable"
        for grid in records.values()
    )
    geometry_transfer = count and all(
        item["available"] and max(
            *item["endpoint_relative_differences"].values(),
            *item["geometry_relative_differences"].values(),
        ) < 0.01 for transfer in transfers.values() for item in transfer.values()
    )
    stability_transfer = count and all(
        item["available"] and item["classification_agrees"] and (
            item["principal_eigenvalue_relative_difference"] < 0.10
            or item["principal_eigenvalue_absolute_difference"] < 0.02
        ) for transfer in transfers.values() for item in transfer.values()
    )
    operators = count and all(
        branch["stability"]["resolved"] and branch["stability"]["controls_pass"]
        for grid in records.values() for branch in grid["branches"].values()
    )
    return {
        "exactly_two_admitted_clusters_on_every_grid": bool(count),
        "finite_positive_ordered_geometry_on_every_grid": bool(geometry),
        "opposite_resolved_inherited_stability_signs_on_every_grid": bool(classification),
        "all_adjacent_endpoint_and_geometry_transfers_below_1_percent": bool(geometry_transfer),
        "all_adjacent_stability_transfers_pass": bool(stability_transfer),
        "all_operator_resolution_and_boundary_controls_pass": bool(operators),
    }


def finalize_negative(output, authority_sha, detectors):
    value = {
        "schema": SCHEMA, "authority_sha256": authority_sha, "status": "REVIEW",
        "classification": "NO-REPAIRED-PARENT-PAIR-THROUGH-0.008",
        "pilot_grid": "G10", "searched_steps": list(SAMPLE_STEPS),
        "searched_times": [step * DT for step in SAMPLE_STEPS],
        "detector_artifacts": detectors, "candidate_step": None, "candidate_time_over_ell": None,
        "bounded_negative_only": True, "nonexistence_claim_authorized": False,
        "event_horizon_claim_authorized": False, "phase_selection_claim_authorized": False,
    }
    value["fingerprint"] = result_fingerprint(value)
    atomic_json(output / "protocol228_result.json", value)
    return value


def finalize_candidate(output, context, authority_sha, candidate_step, states, detectors):
    records, profile_arrays = {}, {}
    for label in GRIDS:
        records[label], arrays = full_observables(context, label, states[label])
        profile_arrays.update(arrays)
    transfers = {
        "G9-G10": context["p227"].adjacent_transfer(records["G9"], records["G10"]),
        "G10-G11": context["p227"].adjacent_transfer(records["G10"], records["G11"]),
    }
    checks = acceptance(records, transfers)
    passed = all(checks.values())
    if profile_arrays:
        atomic_npz(output / "protocol228_profiles.npz", profile_arrays)
    value = {
        "schema": SCHEMA, "authority_sha256": authority_sha,
        "status": "PASS" if passed else "REVIEW",
        "classification": ("REPAIRED-PARENT-FORMATION-CLOSURE-PASS" if passed
                           else "FORMATION-CANDIDATE-NOT-DIRECTLY-CLOSED"),
        "pilot_grid": "G10", "candidate_step": candidate_step,
        "candidate_time_over_ell": candidate_step * DT,
        "formation_sampling_bracket": [(candidate_step - SEGMENT_STEPS) * DT, candidate_step * DT],
        "detector_artifacts": detectors, "records": records,
        "adjacent_grid_transfers": transfers, "acceptance": checks,
        "event_horizon_claim_authorized": False, "continuum_order_claim_authorized": False,
        "phase_selection_claim_authorized": False, "source_ownership_claim_authorized": False,
    }
    value["fingerprint"] = result_fingerprint(value)
    atomic_json(output / "protocol228_result.json", value)
    return value


def validate_final(path, authority_sha):
    value = read_json(path); fingerprint = value.pop("fingerprint", None)
    expected = result_fingerprint(value); value["fingerprint"] = fingerprint
    if fingerprint != expected or value.get("schema") != SCHEMA or value.get("authority_sha256") != authority_sha:
        raise Protocol228Error("final result differs")
    return value


def run(root, academic_root, visual_root, project_root):
    root = Path(root).absolute(); runtime_preflight(root)
    _, inputs = verify(root, academic_root); authority_sha = sha256(root / "authority/freeze_record.json")
    p220_path = protocol_roots(academic_root)["p226"].parent / "protocol220-recoverable-canonical-spatial-2026-08-22/protocol220.py"
    p220 = load(p220_path, "protocol220_lock_bound228")
    with p220.exclusive_lock(root):
        output = root / "candidate-output"
        if not output.exists():
            output.mkdir(); fsync_dir(root)
        if not output.is_dir() or output.is_symlink():
            raise Protocol228Error("candidate output is unsafe")
        final_path = output / "protocol228_result.json"
        if final_path.exists():
            return validate_final(final_path, authority_sha)
        context = load_modules_and_context(root, academic_root, visual_root, project_root, inputs)
        old = context["runner"].axis_even_crossfit_audit
        context["runner"].axis_even_crossfit_audit = context["p220"].P190.node_crossfit
        try:
            initial = {
                label: load_initial_state(label, inputs[label + "/endpoint"])
                for label in GRIDS
            }
            detectors = []
            state = initial["G10"]; previous_sha = sha256(inputs["G10/endpoint"]); previous_step = START_STEP
            candidate_step = None; states_at_candidate = {}
            for step in SAMPLE_STEPS:
                state, receipt_sha = publish_checkpoint(
                    output, context, authority_sha, "G10", previous_step, step, state, previous_sha,
                )
                detector = publish_detector(output, context, authority_sha, step, state, receipt_sha)
                detectors.append(record(output / f"G10_step{step:04d}_detector.json", root))
                if detector["candidate"]:
                    candidate_step = step; states_at_candidate["G10"] = state; break
                previous_step, previous_sha = step, receipt_sha
            if candidate_step is None:
                result = finalize_negative(output, authority_sha, detectors)
            else:
                for label in ("G9", "G11"):
                    state = initial[label]; previous_sha = sha256(inputs[label + "/endpoint"]); previous_step = START_STEP
                    for step in range(START_STEP + SEGMENT_STEPS, candidate_step + 1, SEGMENT_STEPS):
                        state, receipt_sha = publish_checkpoint(
                            output, context, authority_sha, label, previous_step, step, state, previous_sha,
                        )
                        previous_step, previous_sha = step, receipt_sha
                    states_at_candidate[label] = state
                result = finalize_candidate(output, context, authority_sha, candidate_step, states_at_candidate, detectors)
        finally:
            context["runner"].axis_even_crossfit_audit = old
        result = validate_final(final_path, authority_sha)
        verify(root, academic_root)
        return result


def status(root):
    output = Path(root) / "candidate-output"
    final = output / "protocol228_result.json"
    result = read_json(final) if final.is_file() else None
    checkpoints = sorted(path.stem for path in output.glob("*_step*.json") if not path.name.endswith("_detector.json")) if output.exists() else []
    detectors = sorted(path.stem for path in output.glob("G10_step*_detector.json")) if output.exists() else []
    return {
        "schema": "protocol228-status-v1", "checkpoint_count": len(checkpoints),
        "pilot_detector_count": len(detectors), "latest_checkpoint": checkpoints[-1] if checkpoints else None,
        "final_result_present": result is not None,
        "classification": None if result is None else result.get("classification"),
        "candidate_time_over_ell": None if result is None else result.get("candidate_time_over_ell"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(); commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "verify"):
        command = commands.add_parser(name); command.add_argument("--root", required=True); command.add_argument("--academic-root", required=True)
    command = commands.add_parser("run"); command.add_argument("--root", required=True); command.add_argument("--academic-root", required=True)
    command.add_argument("--visual-root", required=True); command.add_argument("--project-root", required=True)
    command = commands.add_parser("status"); command.add_argument("--root", required=True)
    values = vars(parser.parse_args(argv)); selected = values.pop("command")
    if selected == "freeze": result = freeze(**values)
    elif selected == "verify": authority, _ = verify(**values); result = {"verified": True, "fingerprint": authority["fingerprint"]}
    elif selected == "run": result = run(**values)
    else: result = status(**values)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
