#!/usr/bin/env python3
"""Recoverable dense G10 outer marginal-tube continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

import parent_protocol232 as p232
import parent_surface as surface_tools
from authority import file_record, sha256, verify_freeze
from dense_tube_core import (
    adjacent_profile_guard,
    classify_dense_tube,
    strict_area_increase,
    validate_step_schedule,
)
from horizon_core import negative_stencil_resolution, null_expansions
from tube_core import causal_resolution, embedded_curve, projected_tube_norm
from bhps.dynamical_capped_geometry import capped_surface_geometry
from bhps.dynamical_capped_horizon import (
    capped_outgoing_expansion,
    prepare_capped_expansion_slice,
)
from bhps.dynamical_capped_horizon_bvp import solve_dynamical_capped_surface_bvp


SCHEMA = "protocol244-full-dt-g10-dense-tube-result-v1"
CHECKPOINT_SCHEMA = "protocol244-full-dt-g10-checkpoint-v1"
LEAF_SCHEMA = "protocol244-full-dt-g10-outer-leaf-v1"
FULL_DT = 3.125e-5
START_STEP = 32
CHECKPOINT_STEPS = tuple(range(33, 49))
LEAF_STEPS = tuple(range(38, 49))
CONTROL_STEPS = (48,)
SOLVE_ORDER = tuple(range(48, 37, -1))
WIDTHS = (5, 7, 9)
FIELDS = ("q", "v", "source", "memory")
OUTPUT = ROOT / "candidate-output"
P232_INPUT = ROOT / "sealed-inputs/protocol232"
P228_INPUT = ROOT / "sealed-inputs/protocol228"
P243_INPUT = ROOT / "sealed-inputs/protocol243"
SOFT_RSS_BYTES = 16 * 1024**3
THREAD_VARS = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)


class Protocol244Error(RuntimeError):
    pass


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def fingerprinted(value, fingerprint_function):
    """Return a fingerprinted copy while preserving the canonical bare record."""
    if "fingerprint" in value:
        raise Protocol244Error("bare record already contains a fingerprint")
    result = dict(value)
    result["fingerprint"] = fingerprint_function(value)
    return result


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def array_record(value):
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "byte_count": int(array.nbytes),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def protocol211_array_record(value):
    """Exact array-record convention used by the Protocol 232 receipts."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(b"protocol211-array-v1\0")
    digest.update(array.dtype.str.encode())
    digest.update(b"\0")
    digest.update(canonical(list(array.shape)))
    digest.update(array.tobytes(order="C"))
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "byte_count": int(array.nbytes),
        "sha256": digest.hexdigest(),
    }


def regular_immutable_file(path):
    path = Path(path)
    return bool(
        path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
        and (path.stat().st_mode & 0o222) == 0
    )


def fsync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path, payload):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise Protocol244Error(f"output path is not fresh: {path.name}")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o444)
    os.replace(temporary, path)
    fsync_directory(path.parent)
    if not regular_immutable_file(path):
        raise Protocol244Error(f"published path is not immutable: {path.name}")


def atomic_json(path, value):
    atomic_bytes(path, canonical(value))
    if read_json(path) != value:
        raise Protocol244Error(f"JSON immediate replay failed: {Path(path).name}")


def atomic_npz(path, arrays):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise Protocol244Error(f"output path is not fresh: {path.name}")
    with temporary.open("xb") as stream:
        np.savez(stream, **{name: np.ascontiguousarray(value) for name, value in arrays.items()})
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o444)
    os.replace(temporary, path)
    fsync_directory(path.parent)
    if not regular_immutable_file(path):
        raise Protocol244Error(f"published path is not immutable: {path.name}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(arrays) or any(
            not np.array_equal(archive[name], np.asarray(arrays[name])) for name in arrays
        ):
            raise Protocol244Error(f"NPZ immediate replay failed: {path.name}")


def current_peak_rss_bytes():
    values = []
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    values.append(int(usage if sys.platform == "darwin" else usage * 1024))
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                fields = line.split()
                if len(fields) >= 2 and fields[1].isascii() and fields[1].isdigit():
                    values.append(int(fields[1]) * 1024)
    return max(values)


def memory_gate():
    peak = current_peak_rss_bytes()
    if peak >= SOFT_RSS_BYTES:
        raise Protocol244Error("INCOMPLETE-RESOURCE: process RSS reached the 16 GiB stop")
    return peak


def runtime_preflight():
    if not sys.dont_write_bytecode or any(os.environ.get(name) != "1" for name in THREAD_VARS):
        raise Protocol244Error("runtime controls differ")
    expected = (ROOT / "bootstrap.py").resolve()
    observed = Path(getattr(sys.modules.get("__main__"), "__file__", "")).resolve()
    if observed != expected:
        raise Protocol244Error("authorized bootstrap was bypassed")
    validate_step_schedule(CHECKPOINT_STEPS, LEAF_STEPS, CONTROL_STEPS)


def load_npz(path, required):
    path = Path(path)
    if not regular_immutable_file(path):
        raise Protocol244Error(f"unsafe NPZ: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(required):
            raise Protocol244Error(f"NPZ inventory differs: {path.name}")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in required}
    if any(not np.all(np.isfinite(value)) for value in arrays.values()):
        raise Protocol244Error(f"nonfinite NPZ array: {path.name}")
    return arrays


def _terminal_control():
    step = 48
    stem = "G10_step0048"
    arrays = load_npz(P228_INPUT / f"{stem}.npz", FIELDS)
    receipt = read_json(P228_INPUT / f"{stem}.json")
    fingerprint = receipt.pop("fingerprint", None)
    expected = hashlib.sha256(b"protocol228-checkpoint\0" + canonical(receipt)).hexdigest()
    receipt["fingerprint"] = fingerprint
    if not (
        fingerprint == expected
        and receipt.get("authority_sha256") == sha256(P228_INPUT / "freeze_record.json")
        and receipt.get("end_step") == step
        and receipt.get("end_time") == step * FULL_DT
        and receipt.get("dt") == FULL_DT
        and receipt.get("archive", {}).get("byte_count") == (P228_INPUT / f"{stem}.npz").stat().st_size
        and receipt.get("archive", {}).get("sha256") == sha256(P228_INPUT / f"{stem}.npz")
        and receipt.get("arrays") == {name: protocol211_array_record(arrays[name]) for name in sorted(arrays)}
        and receipt.get("passed") is True
    ):
        raise Protocol244Error("copied Protocol 228 terminal checkpoint differs")
    return arrays, receipt


def validate_copied_inputs():
    with np.load(P228_INPUT / "G10_start_step0032.npz", allow_pickle=False) as archive:
        required = {"endpoint_" + name for name in FIELDS}
        if not required <= set(archive.files):
            raise Protocol244Error("copied Protocol 228 start-state inventory differs")
        start_state = {
            name: np.ascontiguousarray(archive["endpoint_" + name], dtype=np.float64)
            for name in FIELDS
        }
    terminal, _ = _terminal_control()
    controls = {48: terminal}
    p228 = read_json(P228_INPUT / "protocol228_result.json")
    if not (
        p228.get("classification") == "REPAIRED-PARENT-FORMATION-CLOSURE-PASS"
        and p228.get("candidate_step") == 48
    ):
        raise Protocol244Error("copied Protocol 228 result differs")
    with np.load(P228_INPUT / "protocol228_profiles.npz", allow_pickle=False) as archive:
        profiles = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    for key in ("theta", "rho", "slope"):
        name = f"G10_outer_{key}"
        if name not in profiles or profiles[name].dtype != np.float64 or profiles[name].shape != (501,):
            raise Protocol244Error(f"copied terminal profile differs: {name}")
    p243 = read_json(P243_INPUT / "protocol243_result.json")
    if not (
        p243.get("classification") == "DENSE-NATIVE-OPERATOR-LOCAL-BALANCE-PASS"
        and p243.get("temporal_or_cross_grid_native_balance_study_authorized") is True
        and p243.get("submitted_paper_edited") is False
    ):
        raise Protocol244Error("Protocol 243 authorization differs")
    return start_state, controls, p228, profiles


def original_protocol232_root(academic_root):
    return Path(academic_root).absolute() / "Discussion/protocols/protocol232-g10-half-timestep-saddle-node-2026-08-24"


def bind_inherited_bhps_path(project_root):
    """Expose authenticated parent-only bhps modules to inherited loaders."""
    project_package = Path(project_root).absolute() / "src/bhps"
    if not project_package.is_dir() or project_package.is_symlink():
        raise Protocol244Error("inherited project bhps package is unsafe")
    package = sys.modules.get("bhps")
    package_path = getattr(package, "__path__", None)
    if package is None or package_path is None:
        raise Protocol244Error("local bhps package was not initialized")
    resolved = str(project_package.resolve())
    observed = [str(Path(item).resolve()) for item in package_path]
    if resolved not in observed:
        package_path.insert(0, resolved)
    rebound = [str(Path(item).resolve()) for item in package_path]
    if not rebound or rebound[0] != resolved:
        raise Protocol244Error("inherited project bhps precedence differs")
    return resolved


def load_evolution_context(academic_root, visual_root, project_root):
    parent_root = original_protocol232_root(academic_root)
    _, inputs = p232.verify(parent_root, academic_root)
    if sha256(parent_root / "authority/freeze_record.json") != sha256(P232_INPUT / "freeze_record.json"):
        raise Protocol244Error("live original Protocol 232 authority differs from copied parent")
    p229 = p232.load(inputs["p229/source"], "protocol229_bound244")
    p229.verify(p232.protocol_roots(academic_root)["p229"], academic_root)
    p228 = p232.load(inputs["p228/source"], "protocol228_bound244")
    _, p228_inputs = p228.verify(p232.protocol_roots(academic_root)["p228"], academic_root)
    p220 = p232.load(inputs["p220/source"], "protocol220_bound244")
    bind_inherited_bhps_path(project_root)
    context = p228.load_modules_and_context(ROOT, academic_root, visual_root, project_root, p228_inputs)
    context.update({"p228": p228, "p220": context["p220"], "p229": p229})
    old_crossfit = context["runner"].axis_even_crossfit_audit
    context["runner"].axis_even_crossfit_audit = context["p220"].P190.node_crossfit
    return context, old_crossfit


def checkpoint_name(step):
    return f"G10_full_step{step:04d}"


def checkpoint_fingerprint(value):
    return hashlib.sha256(b"protocol244-checkpoint-v1\0" + canonical(value)).hexdigest()


def endpoint_semantics(context, step, arrays):
    state = tuple(arrays[name] for name in FIELDS)
    first_pass, first_gates = p232.endpoint_gate(context, state, step * FULL_DT)
    second_pass, second_gates = p232.endpoint_gate(context, state, step * FULL_DT)
    if first_pass != second_pass or first_gates != second_gates:
        raise Protocol244Error(f"endpoint repeat differs: {step}")
    return first_pass, first_gates


def validate_checkpoint(context, authority_sha, step, previous_step, previous_sha, controls):
    stem = checkpoint_name(step)
    archive_path = OUTPUT / f"{stem}.npz"
    receipt_path = OUTPUT / f"{stem}.json"
    arrays = load_npz(archive_path, FIELDS)
    passed, gates = endpoint_semantics(context, step, arrays)
    control_replay = step not in controls or all(
        np.array_equal(arrays[name], controls[step][name]) for name in FIELDS
    )
    metadata = {
        "schema": CHECKPOINT_SCHEMA,
        "authority_sha256": authority_sha,
        "grid": "G10",
        "start_step": previous_step,
        "end_step": step,
        "start_time": previous_step * FULL_DT,
        "end_time": step * FULL_DT,
        "dt": FULL_DT,
        "mode": "legacy_wall_axis_outer",
        "previous_checkpoint_sha256": previous_sha,
        "archive": file_record(archive_path, ROOT),
        "arrays": {name: array_record(arrays[name]) for name in sorted(arrays)},
        "endpoint_gates": gates,
        "endpoint_repeat_exact": True,
        "published_parent_control": step in controls,
        "published_parent_control_bitwise_replay": control_replay,
        "passed": bool(passed and control_replay),
    }
    if not receipt_path.exists():
        if not metadata["passed"]:
            raise Protocol244Error(f"checkpoint publication gap failed semantics: {step}")
        atomic_json(receipt_path, fingerprinted(metadata, checkpoint_fingerprint))
    receipt = read_json(receipt_path)
    fingerprint = receipt.pop("fingerprint", None)
    expected = checkpoint_fingerprint(receipt)
    receipt["fingerprint"] = fingerprint
    expected_metadata = fingerprinted(metadata, checkpoint_fingerprint)
    if receipt != expected_metadata or fingerprint != expected or not metadata["passed"]:
        raise Protocol244Error(f"checkpoint receipt differs: {step}")
    return arrays, sha256(receipt_path), control_replay


def publish_checkpoint(context, authority_sha, step, previous_step, previous_state, previous_sha, controls):
    stem = checkpoint_name(step)
    archive_path = OUTPUT / f"{stem}.npz"
    receipt_path = OUTPUT / f"{stem}.json"
    if receipt_path.exists() and not archive_path.exists():
        raise Protocol244Error(f"checkpoint receipt without archive: {step}")
    if not archive_path.exists():
        bundle = context["bundles"]["G10"]
        validator = lambda stage: context["runner"]._technical_stage_audit(bundle, context["p228"].MODE, stage)
        print(f"G10 dense evolution {previous_step * FULL_DT:.7f} -> {step * FULL_DT:.7f}", flush=True)
        state_tuple = tuple(previous_state[name] for name in FIELDS)
        segment = context["p220"].P216.run_half(
            bundle, context["runner"], context["p228"].MODE, FULL_DT,
            step - previous_step, state_tuple, previous_step + 1, validator,
        )
        if not (
            segment["completed"]
            and len(segment["records"]) == 2 * (step - previous_step)
            and all(item["finite"] for item in segment["records"])
            and all(item["technical_audit"]["gates"]["all_technical_gates_pass"] for item in segment["records"])
        ):
            raise Protocol244Error(f"evolution segment failed: {step}")
        arrays = {name: np.ascontiguousarray(segment["end_state"][index]) for index, name in enumerate(FIELDS)}
        atomic_npz(archive_path, arrays)
    arrays, receipt_sha, replay = validate_checkpoint(
        context, authority_sha, step, previous_step, previous_sha, controls,
    )
    memory_gate()
    return arrays, receipt_sha, replay


def profile_name(step):
    return f"G10_outer_step{step:04d}"


def profile_fingerprint(value):
    return hashlib.sha256(b"protocol244-outer-leaf-v1\0" + canonical(value)).hexdigest()


def profile_arrays(path):
    return load_npz(path, ("theta", "rho", "slope"))


def parent_profile(step, copied_profiles):
    if step != 48:
        raise Protocol244Error("only the terminal Protocol 228 profile is a parent control")
    return {key: np.ascontiguousarray(copied_profiles[f"G10_outer_{key}"]) for key in ("theta", "rho", "slope")}


def control_comparison(step, profile, copied_profiles):
    if step not in CONTROL_STEPS:
        return {"is_control": False, "passed": True}
    parent = parent_profile(step, copied_profiles)
    rho_difference = float(np.max(np.abs(profile["rho"] - parent["rho"])))
    slope_difference = float(np.max(np.abs(profile["slope"] - parent["slope"])))
    endpoint_difference = float(math.hypot(
        profile["rho"][0] - parent["rho"][0],
        profile["rho"][-1] - parent["rho"][-1],
    ))
    return {
        "is_control": True,
        "rho_maximum_absolute_difference": rho_difference,
        "slope_maximum_absolute_difference": slope_difference,
        "endpoint_euclidean_difference": endpoint_difference,
        "passed": bool(rho_difference < 1e-4 and slope_difference < 1e-3 and endpoint_difference < 1e-4),
    }


def solver_public_admitted(surface):
    cross = surface.get("primary_evaluator_crosscheck", {})
    return bool(
        surface.get("converged")
        and surface.get("in_domain")
        and surface.get("local_expansion_interior_maximum", math.inf) < 2e-4
        and surface.get("boundary_slope_error", math.inf) < 2e-4
        and "error" not in cross
        and cross.get("two_cell_interior_maximum", math.inf) < 0.002
    )


def evaluate_leaf(step, state, profile, copied_profiles, solver_public):
    position, velocity = state["q"], state["v"]
    z = np.linspace(1.0, math.e, position.shape[0])
    r = np.linspace(0.0, 10.0, position.shape[1])
    prepared = prepare_capped_expansion_slice(position, velocity, z, r, stencil_width=7)
    admitted = solver_public_admitted(solver_public)
    geometry = capped_surface_geometry(position, velocity, z, r, profile, stencil_width=7, prepared=prepared)
    stability = surface_tools.stability_series(position, velocity, z, r, profile, prepared)
    width_records = {}
    masks = []
    inward = {}
    arrays = {key: np.ascontiguousarray(profile[key]) for key in ("theta", "rho", "slope")}
    for width in WIDTHS:
        width_prepared = prepare_capped_expansion_slice(position, velocity, z, r, stencil_width=width)
        evaluated = capped_outgoing_expansion(
            position, velocity, z, r, profile,
            stencil_width=width, prepared=width_prepared,
        )
        outward, theta_minus = null_expansions(
            evaluated["mean_curvature"], evaluated["extrinsic_curvature_correction"],
        )
        if not np.array_equal(outward, evaluated["outgoing_expansion"]):
            raise Protocol244Error(f"outgoing expansion identity differs: {step}/{width}")
        mask = np.asarray(evaluated["two_cell_interior_mask"], dtype=bool)
        masks.append(mask)
        inward[width] = theta_minus
        outgoing_maximum = float(np.max(np.abs(outward[mask])))
        width_records[str(width)] = {
            "outgoing_interior_maximum_absolute": outgoing_maximum,
            "outgoing_parent_ceiling_pass": bool(outgoing_maximum < 0.002),
            "inward_interior_minimum": float(np.min(theta_minus[mask])),
            "inward_interior_maximum": float(np.max(theta_minus[mask])),
        }
        arrays[f"w{width}_theta_plus"] = outward
        arrays[f"w{width}_theta_minus"] = theta_minus
    if not all(np.array_equal(masks[0], mask) for mask in masks[1:]):
        raise Protocol244Error(f"expansion masks differ: {step}")
    resolution, resolved, spread = negative_stencil_resolution(
        inward[5], inward[7], inward[9], masks[0],
    )
    arrays["interior_mask"] = masks[0].astype(np.uint8)
    arrays["inward_resolved"] = resolved.astype(np.uint8)
    arrays["inward_stencil_spread"] = spread
    control = control_comparison(step, profile, copied_profiles)
    metadata = {
        "step": step,
        "time_over_ell": step * FULL_DT,
        "surface": solver_public,
        "surface_admission_pass": bool(admitted),
        "geometry": geometry,
        "stability": stability,
        "stencils": width_records,
        "negative_inward_resolution": resolution,
        "all_outgoing_parent_ceilings_pass": bool(all(item["outgoing_parent_ceiling_pass"] for item in width_records.values())),
        "control_comparison": control,
    }
    metadata["passed"] = bool(
        metadata["surface_admission_pass"]
        and stability["classification"] == "outward-stable"
        and stability["resolved"] and stability["controls_pass"]
        and metadata["all_outgoing_parent_ceilings_pass"]
        and resolution["passed"]
        and control["passed"]
    )
    return metadata, arrays


def validate_leaf(step, state, authority_sha, copied_profiles):
    stem = profile_name(step)
    archive_path = OUTPUT / f"{stem}.npz"
    receipt_path = OUTPUT / f"{stem}.json"
    arrays = profile_arrays(archive_path)
    existing_receipt = read_json(receipt_path) if receipt_path.exists() else None
    solver_public = None if existing_receipt is None else existing_receipt.get("evaluation", {}).get("surface")
    if not receipt_path.exists():
        z = np.linspace(1.0, math.e, state["q"].shape[0])
        r = np.linspace(0.0, 10.0, state["q"].shape[1])
        prepared = prepare_capped_expansion_slice(state["q"], state["v"], z, r, stencil_width=7)
        first = solve_dynamical_capped_surface_bvp(
            state["q"], state["v"], z, r,
            arrays, tolerance=2e-6, nodes=121, maximum_nodes=6000, dense_nodes=501,
            prepared=prepared,
        )
        second = solve_dynamical_capped_surface_bvp(
            state["q"], state["v"], z, r,
            arrays, tolerance=2e-6, nodes=121, maximum_nodes=6000, dense_nodes=501,
            prepared=prepared,
        )
        if any(not np.array_equal(first[key], second[key]) for key in ("theta", "rho", "slope")):
            raise Protocol244Error(f"publication-gap leaf repeat differs: {step}")
        solver_public = surface_tools.public_surface(first)
    if not isinstance(solver_public, dict):
        raise Protocol244Error(f"leaf solver provenance is missing: {step}")
    metadata, derived = evaluate_leaf(step, state, arrays, copied_profiles, solver_public)
    expected_arrays = {name: array_record(value) for name, value in sorted(derived.items())}
    if not receipt_path.exists():
        receipt = {
            "schema": LEAF_SCHEMA,
            "authority_sha256": authority_sha,
            "step": step,
            "time_over_ell": step * FULL_DT,
            "solve_repeat_exact": True,
            "archive": file_record(archive_path, ROOT),
            "profile_arrays": {name: array_record(arrays[name]) for name in sorted(arrays)},
            "derived_array_records": expected_arrays,
            "evaluation": metadata,
            "passed": metadata["passed"],
        }
        atomic_json(receipt_path, fingerprinted(receipt, profile_fingerprint))
    receipt = read_json(receipt_path)
    fingerprint = receipt.pop("fingerprint", None)
    expected_fingerprint = profile_fingerprint(receipt)
    receipt["fingerprint"] = fingerprint
    if not (
        fingerprint == expected_fingerprint
        and receipt.get("schema") == LEAF_SCHEMA
        and receipt.get("authority_sha256") == authority_sha
        and receipt.get("step") == step
        and receipt.get("archive") == file_record(archive_path, ROOT)
        and receipt.get("profile_arrays") == {name: array_record(arrays[name]) for name in sorted(arrays)}
        and receipt.get("derived_array_records") == expected_arrays
        and receipt.get("evaluation") == metadata
        and receipt.get("solve_repeat_exact") is True
        and receipt.get("passed") is metadata["passed"]
    ):
        raise Protocol244Error(f"leaf receipt differs: {step}")
    return metadata, derived


def publish_leaf(step, state, seed, authority_sha, copied_profiles):
    stem = profile_name(step)
    archive_path = OUTPUT / f"{stem}.npz"
    receipt_path = OUTPUT / f"{stem}.json"
    if receipt_path.exists() and not archive_path.exists():
        raise Protocol244Error(f"leaf receipt without archive: {step}")
    if not archive_path.exists():
        z = np.linspace(1.0, math.e, state["q"].shape[0])
        r = np.linspace(0.0, 10.0, state["q"].shape[1])
        prepared = prepare_capped_expansion_slice(state["q"], state["v"], z, r)
        print(f"G10 outer leaf step {step} at t/ell={step * FULL_DT:.7f}", flush=True)
        first = solve_dynamical_capped_surface_bvp(
            state["q"], state["v"], z, r, seed, tolerance=2e-6,
            nodes=121, maximum_nodes=6000, dense_nodes=501, prepared=prepared,
        )
        second = solve_dynamical_capped_surface_bvp(
            state["q"], state["v"], z, r, seed, tolerance=2e-6,
            nodes=121, maximum_nodes=6000, dense_nodes=501, prepared=prepared,
        )
        arrays = {key: np.ascontiguousarray(first[key]) for key in ("theta", "rho", "slope")}
        if any(not np.array_equal(arrays[key], second[key]) for key in arrays):
            raise Protocol244Error(f"leaf solve repeat differs: {step}")
        atomic_npz(archive_path, arrays)
    metadata, derived = validate_leaf(step, state, authority_sha, copied_profiles)
    memory_gate()
    return metadata, derived


def validate_inventory():
    if not OUTPUT.exists():
        return
    if OUTPUT.is_symlink() or not OUTPUT.is_dir():
        raise Protocol244Error("candidate-output is unsafe")
    allowed = set()
    for step in CHECKPOINT_STEPS:
        stem = checkpoint_name(step)
        allowed.update({f"{stem}.npz", f"{stem}.json"})
    for step in SOLVE_ORDER:
        stem = profile_name(step)
        allowed.update({f"{stem}.npz", f"{stem}.json"})
    allowed.update({"protocol244_dense_tube_arrays.npz", "protocol244_result.json"})
    names = {path.name for path in OUTPUT.iterdir()}
    if not names <= allowed:
        raise Protocol244Error(f"unexpected output inventory: {sorted(names - allowed)}")
    gap = False
    for step in CHECKPOINT_STEPS:
        stem = checkpoint_name(step)
        archive = (OUTPUT / f"{stem}.npz").exists()
        receipt = (OUTPUT / f"{stem}.json").exists()
        if receipt and not archive:
            raise Protocol244Error("checkpoint receipt without archive")
        if gap and (archive or receipt):
            raise Protocol244Error("checkpoint inventory is not a prefix")
        if not (archive and receipt):
            gap = True
    any_leaf = any((OUTPUT / f"{profile_name(step)}.npz").exists() for step in SOLVE_ORDER)
    if gap and any_leaf:
        raise Protocol244Error("leaf precedes complete evolution")
    leaf_gap = False
    for step in SOLVE_ORDER:
        stem = profile_name(step)
        archive = (OUTPUT / f"{stem}.npz").exists()
        receipt = (OUTPUT / f"{stem}.json").exists()
        if receipt and not archive:
            raise Protocol244Error("leaf receipt without archive")
        if leaf_gap and (archive or receipt):
            raise Protocol244Error("leaf inventory is not the authorized solve-order prefix")
        if not (archive and receipt):
            leaf_gap = True
    final_archive = (OUTPUT / "protocol244_dense_tube_arrays.npz").exists()
    final_result = (OUTPUT / "protocol244_result.json").exists()
    if final_result and not final_archive:
        raise Protocol244Error("final result without final archive")
    if (final_archive or final_result) and (gap or leaf_gap):
        raise Protocol244Error("final artifact precedes all stages")


def result_fingerprint(value):
    return hashlib.sha256(b"protocol244-result-v1\0" + canonical(value)).hexdigest()


def dense_evaluation(leaves, derived, states):
    ordered = [leaves[step] for step in LEAF_STEPS]
    profile_guard = adjacent_profile_guard([derived[step]["rho"] for step in LEAF_STEPS])
    area = strict_area_increase(
        np.asarray(LEAF_STEPS, dtype=np.int64),
        np.asarray([item["geometry"]["one_sided_cap_area"] for item in ordered]),
    )
    surface_stability = bool(all(
        item["surface_admission_pass"]
        and item["stability"]["classification"] == "outward-stable"
        and item["stability"]["resolved"] and item["stability"]["controls_pass"]
        and item["control_comparison"]["passed"]
        for item in ordered
    ) and profile_guard["passed"])
    inward = bool(all(
        item["negative_inward_resolution"]["passed"]
        and item["all_outgoing_parent_ceilings_pass"]
        for item in ordered
    ))
    tube = {}
    arrays = {}
    for index in range(1, len(LEAF_STEPS) - 1):
        early_step, middle_step, late_step = LEAF_STEPS[index - 1:index + 2]
        theta = derived[middle_step]["theta"]
        if not (
            np.array_equal(theta, derived[early_step]["theta"])
            and np.array_equal(theta, derived[late_step]["theta"])
        ):
            raise Protocol244Error(f"theta grid differs around step {middle_step}")
        early_coordinates, _ = embedded_curve(theta, derived[early_step]["rho"], derived[early_step]["slope"])
        middle_coordinates, _ = embedded_curve(theta, derived[middle_step]["rho"], derived[middle_step]["slope"])
        late_coordinates, _ = embedded_curve(theta, derived[late_step]["rho"], derived[late_step]["slope"])
        spacing = (middle_step - early_step) * FULL_DT
        velocities = {
            "backward": (middle_coordinates - early_coordinates) / spacing,
            "centered": (late_coordinates - early_coordinates) / (2.0 * spacing),
            "forward": (late_coordinates - middle_coordinates) / spacing,
        }
        metric = surface_tools.sample_mid_metric(
            states[middle_step], theta, derived[middle_step]["rho"], derived[middle_step]["slope"],
        )
        norms = {
            name: projected_tube_norm(
                metric["lapse"], metric["shift_covector"], metric["shift"],
                metric["metric"], velocity, metric["tangent"],
            )
            for name, velocity in velocities.items()
        }
        summary, resolved, spread = causal_resolution(
            norms["backward"], norms["centered"], norms["forward"],
        )
        summary.update({
            "step": middle_step,
            "time_over_ell": middle_step * FULL_DT,
            "time_spacing": spacing,
        })
        tube[str(middle_step)] = summary
        for name, value in norms.items():
            arrays[f"step{middle_step:04d}_norm_{name}"] = value
        arrays[f"step{middle_step:04d}_resolved"] = resolved.astype(np.uint8)
        arrays[f"step{middle_step:04d}_one_sided_spread"] = spread
    signature = bool(all(
        item["label"] == "UNIFORMLY-SPACELIKE-SPARSE-PILOT"
        and item["resolved_fraction"] == 1.0
        for item in tube.values()
    ))
    return {
        "profile_continuity": profile_guard,
        "area_increase": area,
        "all_surfaces_and_stability_pass": surface_stability,
        "all_inward_expansions_resolved_negative": inward,
        "all_interior_tube_signatures_resolved_spacelike": signature,
        "tube": tube,
        "leaves": {str(step): leaves[step] for step in LEAF_STEPS},
    }, arrays


def finalize(authority_sha, checkpoint_replay, leaves, derived, states, peak, started):
    evaluation, tube_arrays = dense_evaluation(leaves, derived, states)
    if not checkpoint_replay:
        raise Protocol244Error("published Protocol 228 terminal checkpoint replay differs")
    classification, gates = classify_dense_tube(
        evaluation["all_surfaces_and_stability_pass"],
        evaluation["all_inward_expansions_resolved_negative"],
        evaluation["area_increase"]["passed"],
        evaluation["all_interior_tube_signatures_resolved_spacelike"],
    )
    all_arrays = {}
    for step in LEAF_STEPS:
        for name, value in derived[step].items():
            all_arrays[f"step{step:04d}_{name}"] = value
    all_arrays.update(tube_arrays)
    archive_path = OUTPUT / "protocol244_dense_tube_arrays.npz"
    if not archive_path.exists():
        atomic_npz(archive_path, all_arrays)
    else:
        observed = load_npz(archive_path, tuple(all_arrays))
        if any(not np.array_equal(observed[name], all_arrays[name]) for name in all_arrays):
            raise Protocol244Error("final archive publication gap differs")
    result_path = OUTPUT / "protocol244_result.json"
    scientific = {
        "classification": classification,
        "gates": gates,
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "leaf_steps": list(LEAF_STEPS),
        "control_steps": list(CONTROL_STEPS),
        "all_parent_checkpoint_controls_bitwise_exact": bool(checkpoint_replay),
        "evaluation": evaluation,
    }
    if result_path.exists():
        result = read_json(result_path)
        fingerprint = result.pop("fingerprint", None)
        expected = result_fingerprint(result)
        result["fingerprint"] = fingerprint
        if not (
            fingerprint == expected
            and result.get("authority_sha256") == authority_sha
            and result.get("scientific") == scientific
            and result.get("archive") == file_record(archive_path, ROOT)
            and result.get("array_records") == {name: array_record(value) for name, value in sorted(all_arrays.items())}
        ):
            raise Protocol244Error("completed result differs on recovery")
        return result
    peak = max(peak, memory_gate())
    result = {
        "schema": SCHEMA,
        "authority_sha256": authority_sha,
        "scientific": scientific,
        "archive": file_record(archive_path, ROOT),
        "array_records": {name: array_record(value) for name, value in sorted(all_arrays.items())},
        "resource": {
            "peak_rss_bytes": peak,
            "soft_rss_stop_bytes": SOFT_RSS_BYTES,
            "elapsed_wall_seconds": float(time.monotonic() - started),
        },
        "spacetime_evolution_executed": True,
        "new_parent_solve_executed": False,
        "submitted_paper_edited": False,
        "parent_or_published_artifact_modified": False,
        "full_dt_native_balance_replay_authorized": classification == "FULL-DT-DENSE-G10-OUTER-MARGINAL-TUBE-PASS",
        "continuum_dynamical_horizon_claim_authorized": False,
        "event_horizon_claim_authorized": False,
        "source_ownership_claim_authorized": False,
    }
    result = fingerprinted(result, result_fingerprint)
    atomic_json(result_path, result)
    return result


def execute(academic_root, visual_root, project_root):
    started = time.monotonic()
    runtime_preflight()
    authority = verify_freeze(ROOT)
    authority_sha = sha256(ROOT / "authority/freeze_record.json")
    start_state, controls, _, copied_profiles = validate_copied_inputs()
    context, old_crossfit = load_evolution_context(academic_root, visual_root, project_root)
    p220 = context["p220"]
    peak = memory_gate()
    try:
        with p220.exclusive_lock(ROOT):
            if not OUTPUT.exists():
                OUTPUT.mkdir(mode=0o755)
                fsync_directory(ROOT)
            validate_inventory()
            states = {START_STEP: start_state}
            previous_step = START_STEP
            previous_state = start_state
            previous_sha = sha256(P228_INPUT / "G10_start_step0032.npz")
            replay_flags = []
            for step in CHECKPOINT_STEPS:
                state, previous_sha, replay = publish_checkpoint(
                    context, authority_sha, step, previous_step, previous_state, previous_sha, controls,
                )
                states[step] = state
                previous_step, previous_state = step, state
                if step in CONTROL_STEPS:
                    replay_flags.append(replay)
                peak = max(peak, memory_gate())
            leaves = {}
            derived = {}
            for step in SOLVE_ORDER:
                if step in CONTROL_STEPS:
                    seed = parent_profile(step, copied_profiles)
                else:
                    seed = profile_arrays(OUTPUT / f"{profile_name(step + 1)}.npz")
                metadata, arrays = publish_leaf(step, states[step], seed, authority_sha, copied_profiles)
                leaves[step] = metadata
                derived[step] = arrays
                peak = max(peak, memory_gate())
            validate_inventory()
            result = finalize(
                authority_sha, bool(len(replay_flags) == len(CONTROL_STEPS) and all(replay_flags)),
                leaves, derived, states, peak, started,
            )
            validate_inventory()
            verify_freeze(ROOT)
            return result
    finally:
        context["runner"].axis_even_crossfit_audit = old_crossfit


def status():
    completed_checkpoints = [
        step for step in CHECKPOINT_STEPS
        if (OUTPUT / f"{checkpoint_name(step)}.npz").is_file()
        and (OUTPUT / f"{checkpoint_name(step)}.json").is_file()
    ] if OUTPUT.exists() else []
    completed_leaves = [
        step for step in SOLVE_ORDER
        if (OUTPUT / f"{profile_name(step)}.npz").is_file()
        and (OUTPUT / f"{profile_name(step)}.json").is_file()
    ] if OUTPUT.exists() else []
    result_path = OUTPUT / "protocol244_result.json"
    result = read_json(result_path) if result_path.is_file() else None
    return {
        "schema": "protocol244-status-v1",
        "completed_checkpoint_steps": completed_checkpoints,
        "completed_leaf_steps_in_solve_order": completed_leaves,
        "final_result_present": result is not None,
        "classification": None if result is None else result.get("scientific", {}).get("classification"),
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
        value = execute(**values)
    elif command == "verify":
        runtime_preflight()
        authority = verify_freeze(ROOT)
        validate_copied_inputs()
        value = {"status": "VERIFIED", "authority_sha256": sha256(ROOT / "authority/freeze_record.json"), "fingerprint": authority["fingerprint"]}
    else:
        value = status()
    print(json.dumps(value, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
