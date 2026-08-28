#!/usr/bin/env python3
"""Archive-only three-grid native local-balance transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import engine245 as engine
from authority import file_record, sha256, verify_freeze
from spatial_balance_core import compare_centered, classify


SCHEMA = "protocol248-three-grid-native-balance-transfer-result-v1"
P245 = ROOT / "sealed-inputs/protocol245"
P246 = ROOT / "sealed-inputs/protocol246"
P247 = ROOT / "sealed-inputs/protocol247"
P247_OUTPUT = P247 / "candidate-output"
OUTPUT = ROOT / "candidate-output"
DT = 3.125e-5
LEAF_STEPS = tuple(range(43, 49))
INTERIOR_STEPS = tuple(range(44, 48))
GRIDS = ("G9", "G11")
WIDTHS = (5, 7, 9)
FIELDS = ("q", "v", "source", "memory")
THREAD_VARS = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)
EXPECTED_RUNTIME = {
    "python": "3.8.10",
    "numpy": "1.24.4",
    "scipy": "1.10.1",
    "system": "Linux",
    "machine": "aarch64",
}


class Protocol248Error(RuntimeError):
    pass


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def check_fingerprint(record, prefix):
    bare = dict(record); observed = bare.pop("fingerprint", None)
    return observed == hashlib.sha256(prefix + canonical(bare)).hexdigest()


def array_record(value):
    array = np.ascontiguousarray(value)
    return {"dtype": array.dtype.str, "shape": list(array.shape), "byte_count": int(array.nbytes), "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def regular(path):
    path = Path(path)
    return path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1 and (path.stat().st_mode & 0o222) == 0


def runtime_preflight():
    if not sys.dont_write_bytecode or any(os.environ.get(name) != "1" for name in THREAD_VARS):
        raise Protocol248Error("runtime controls differ")
    observed = Path(getattr(sys.modules.get("__main__"), "__file__", "")).resolve()
    if observed != (ROOT / "bootstrap.py").resolve():
        raise Protocol248Error("authorized bootstrap was bypassed")
    runtime = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "system": platform.system(),
        "machine": platform.machine(),
    }
    if runtime != EXPECTED_RUNTIME:
        raise Protocol248Error(f"live numerical runtime differs: {runtime}")


def load_npz(path):
    if not regular(path):
        raise Protocol248Error(f"unsafe archive: {Path(path).name}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def validate_p245_p246():
    p245 = read_json(P245 / "protocol245_result.json")
    p246 = read_json(P246 / "protocol246_result.json")
    if not (
        check_fingerprint(p245, b"protocol245-result-v1\0")
        and p245.get("classification") == "FULL-DT-DENSE-NATIVE-OPERATOR-LOCAL-BALANCE-PASS"
        and p245.get("complete_physical_repeat_exact") is True
        and p245.get("scientific", {}).get("interior_steps") == list(range(39, 48))
        and all(p245.get("scientific", {}).get("gates", {}).values())
        and p245.get("spacetime_evolution_executed") is False
        and p245.get("surface_solve_executed") is False
    ):
        raise Protocol248Error("Protocol 245 prerequisite differs")
    if not (
        check_fingerprint(p246, b"protocol246-result-v1\0")
        and p246.get("classification") == "FULL-HALF-NATIVE-BALANCE-CONSISTENCY-PASS"
        and p246.get("complete_comparison_repeat_exact") is True
        and all(p246.get("scientific", {}).get("gates", {}).values())
    ):
        raise Protocol248Error("Protocol 246 prerequisite differs")
    return p245, p246


def expected_shape(label, field):
    spatial = {"G9": (113, 211), "G11": (145, 271)}[label]
    return spatial + (9 if field in {"q", "v"} else 3,)


def validate_p247():
    result = read_json(P247_OUTPUT / "protocol247_result.json")
    if not (
        check_fingerprint(result, b"protocol247-result-v1\0")
        and result.get("schema") == "protocol247-g9-g11-bounded-spatial-transfer-result-v1"
        and result.get("authority_sha256") == sha256(P247 / "freeze_record.json")
        and result.get("scientific", {}).get("classification") == "G9-G10-G11-BOUNDED-OUTER-TUBE-SPATIAL-TRANSFER-PASS"
        and all(result.get("scientific", {}).get("gates", {}).values())
        and result.get("archive_only_balance_transfer_authorized") is True
        and result.get("parent_or_published_artifact_modified") is False
        and result.get("submitted_paper_edited") is False
    ):
        raise Protocol248Error("Protocol 247 prerequisite differs")
    states_by_grid, profiles_by_grid, leaves_by_grid = {}, {}, {}
    for label in GRIDS:
        states, profiles, leaves = {}, {}, {}
        for step in LEAF_STEPS:
            checkpoint_stem = f"{label}_full_step{step:04d}"
            checkpoint_path = P247_OUTPUT / f"{checkpoint_stem}.npz"
            checkpoint = read_json(P247_OUTPUT / f"{checkpoint_stem}.json")
            arrays = load_npz(checkpoint_path)
            if not (
                check_fingerprint(checkpoint, b"protocol247-grid-checkpoint-v1\0")
                and checkpoint.get("schema") == "protocol247-grid-checkpoint-v1"
                and checkpoint.get("authority_sha256") == result["authority_sha256"]
                and checkpoint.get("grid") == label and checkpoint.get("end_step") == step
                and checkpoint.get("dt") == DT and checkpoint.get("passed") is True
                and checkpoint.get("endpoint_repeat_exact") is True
                and checkpoint.get("archive") == file_record(checkpoint_path, P247)
                and set(arrays) == set(FIELDS)
                and all(
                    arrays[field].dtype == np.float64 and arrays[field].shape == expected_shape(label, field)
                    and np.all(np.isfinite(arrays[field])) and array_record(arrays[field]) == checkpoint["arrays"][field]
                    for field in FIELDS
                )
            ):
                raise Protocol248Error(f"Protocol 247 checkpoint differs: {label}/{step}")
            leaf_stem = f"{label}_outer_step{step:04d}"
            profile_path = P247_OUTPUT / f"{leaf_stem}.npz"
            leaf = read_json(P247_OUTPUT / f"{leaf_stem}.json")
            profile = load_npz(profile_path)
            if not (
                check_fingerprint(leaf, b"protocol247-grid-outer-leaf-v1\0")
                and leaf.get("schema") == "protocol247-grid-outer-leaf-v1"
                and leaf.get("authority_sha256") == result["authority_sha256"]
                and leaf.get("grid") == label and leaf.get("step") == step
                and leaf.get("passed") is True and leaf.get("solve_repeat_exact") is True
                and leaf.get("archive") == file_record(profile_path, P247)
                and set(profile) == {"theta", "rho", "slope"}
                and all(
                    profile[field].dtype == np.float64 and profile[field].shape == (501,)
                    and np.all(np.isfinite(profile[field])) and array_record(profile[field]) == leaf["profile_arrays"][field]
                    for field in profile
                )
                and leaf.get("evaluation") == result["scientific"]["grids"][label]["leaves"][str(step)]
            ):
                raise Protocol248Error(f"Protocol 247 leaf differs: {label}/{step}")
            states[step], profiles[step], leaves[step] = arrays, profile, leaf["evaluation"]
        if not all(np.array_equal(profiles[LEAF_STEPS[0]]["theta"], profiles[step]["theta"]) for step in LEAF_STEPS[1:]):
            raise Protocol248Error(f"theta grids differ: {label}")
        states_by_grid[label], profiles_by_grid[label], leaves_by_grid[label] = states, profiles, leaves
    return result, states_by_grid, profiles_by_grid, leaves_by_grid


def evaluate_grid(label, states, profiles, leaves, background):
    old = (engine.LEAF_STEPS, engine.INTERIOR_STEPS, engine.GRID_LABEL)
    try:
        engine.LEAF_STEPS = LEAF_STEPS
        engine.INTERIOR_STEPS = INTERIOR_STEPS
        engine.GRID_LABEL = label
        first, first_arrays = engine.evaluate(states, profiles, leaves, background)
        second, second_arrays = engine.evaluate(states, profiles, leaves, background)
    finally:
        engine.LEAF_STEPS, engine.INTERIOR_STEPS, engine.GRID_LABEL = old
    if first != second or set(first_arrays) != set(second_arrays) or any(
        not np.array_equal(first_arrays[name], second_arrays[name]) for name in first_arrays
    ):
        raise Protocol248Error(f"complete balance repeat differs: {label}")
    return first, first_arrays


def finite_record(value):
    if isinstance(value, dict):
        return all(finite_record(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_record(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def compare_grids(grid_results, p245):
    records = {"G9": grid_results["G9"], "G10": p245["scientific"], "G11": grid_results["G11"]}
    comparisons = {}
    maxima = {
        name: 0.0 for name in (
            "area_value_relative", "area_finite_rate_relative", "area_marginal_rate_relative",
            "seam_geometric_relative", "seam_wall_relative", "ledger_target_relative",
            "ledger_total_relative", "ledger_term_balance_norm_relative", "native_directional_relative",
            "native_directional_absolute", "native_history_relative", "native_history_absolute",
            "native_wall_relative", "native_wall_absolute", "parent_normalized_residual",
        )
    }
    signs = geometry = balance = native = alignment = True
    for left_label, right_label in (("G9", "G10"), ("G10", "G11")):
        pair = f"{left_label}-{right_label}"
        comparisons[pair] = {}
        for step in INTERIOR_STEPS:
            left_step = records[left_label]["steps"][str(step)]
            right_step = records[right_label]["steps"][str(step)]
            time_equal = bool(left_step["time_over_ell"] == right_step["time_over_ell"] == step * DT)
            alignment = alignment and time_equal
            widths = {}
            for width in WIDTHS:
                item = compare_centered(left_step, right_step, width)
                widths[str(width)] = item
                signs = signs and item["sign_pass"]
                geometry = geometry and item["geometry_pass"]
                balance = balance and item["balance_pass"]
                native = native and item["native_operator_pass"]
                for name, value in item["comparison"].items():
                    maxima[name] = max(maxima[name], float(value))
            comparisons[pair][str(step)] = {"step": step, "time_over_ell": step * DT, "time_alignment_exact": time_equal, "widths": widths}
    parent = bool(
        p245.get("classification") == "FULL-DT-DENSE-NATIVE-OPERATOR-LOCAL-BALANCE-PASS"
        and all(grid_results[label]["classification"] == "FULL-DT-DENSE-NATIVE-OPERATOR-LOCAL-BALANCE-PASS" for label in GRIDS)
    )
    local = bool(all(records[label]["classification"] == "FULL-DT-DENSE-NATIVE-OPERATOR-LOCAL-BALANCE-PASS" for label in ("G9", "G10", "G11")))
    gates = {
        "parent_admission": parent,
        "local_balance_admission": local,
        "time_alignment": bool(alignment),
        "sign_consistency": bool(signs),
        "geometry_consistency": bool(geometry),
        "balance_consistency": bool(balance),
        "native_operator_consistency": bool(native),
    }
    result = {
        "classification": classify(gates),
        "gates": gates,
        "matched_steps": list(INTERIOR_STEPS),
        "matched_time_count": len(INTERIOR_STEPS),
        "classification_driving_path": "centered",
        "stencil_widths": list(WIDTHS),
        "native_epsilon": 5e-7,
        "maximum_discrepancies": maxima,
        "adjacent_grid_comparisons": comparisons,
        "local_grid_results": {label: records[label] for label in ("G9", "G10", "G11")},
    }
    if not finite_record(result):
        raise Protocol248Error("nonfinite comparison record")
    return result


def current_peak_rss_bytes():
    usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return usage if sys.platform == "darwin" else usage * 1024


def atomic_bytes(path, payload):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise Protocol248Error(f"output namespace is not fresh: {path.name}")
    with temporary.open("xb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    os.replace(temporary, path)
    engine.fsync_directory(path.parent)


def atomic_json(path, value):
    atomic_bytes(path, canonical(value))
    if read_json(path) != value:
        raise Protocol248Error("JSON immediate replay failed")


def atomic_npz(path, arrays):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise Protocol248Error(f"output namespace is not fresh: {path.name}")
    with temporary.open("xb") as stream:
        np.savez(stream, **{name: np.ascontiguousarray(value) for name, value in arrays.items()})
        stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    os.replace(temporary, path); engine.fsync_directory(path.parent)
    observed = load_npz(path)
    if set(observed) != set(arrays) or any(not np.array_equal(observed[name], arrays[name]) for name in arrays):
        raise Protocol248Error("NPZ immediate replay failed")


def execute():
    started = time.monotonic()
    runtime_preflight()
    verify_freeze(ROOT)
    p245, _ = validate_p245_p246()
    _, states, profiles, leaves = validate_p247()
    background = engine.load_background()
    controls = engine.manufactured_controls()
    if controls.get("passed") is not True:
        raise Protocol248Error("manufactured controls failed")
    first_results, first_arrays = {}, {}
    for label in GRIDS:
        first_results[label], first_arrays[label] = evaluate_grid(label, states[label], profiles[label], leaves[label], background)
    first = compare_grids(first_results, p245)
    second_results, second_arrays = {}, {}
    for label in GRIDS:
        second_results[label], second_arrays[label] = evaluate_grid(label, states[label], profiles[label], leaves[label], background)
    second = compare_grids(second_results, p245)
    if first != second or first_results != second_results:
        raise Protocol248Error("complete transfer repeat differs")
    all_arrays = {}
    for label in GRIDS:
        for name, value in first_arrays[label].items():
            all_arrays[f"{label}_{name}"] = value
        if set(first_arrays[label]) != set(second_arrays[label]) or any(
            not np.array_equal(first_arrays[label][name], second_arrays[label][name]) for name in first_arrays[label]
        ):
            raise Protocol248Error(f"balance arrays repeat differs: {label}")
    if not OUTPUT.exists():
        OUTPUT.mkdir(mode=0o755); engine.fsync_directory(ROOT)
    if OUTPUT.is_symlink() or not OUTPUT.is_dir():
        raise Protocol248Error("candidate-output is unsafe")
    allowed = {"protocol248_balance_transfer_arrays.npz", "protocol248_result.json"}
    if not {path.name for path in OUTPUT.iterdir()} <= allowed:
        raise Protocol248Error("candidate-output inventory differs")
    archive_path = OUTPUT / "protocol248_balance_transfer_arrays.npz"
    if not archive_path.exists():
        atomic_npz(archive_path, all_arrays)
    else:
        observed_arrays = load_npz(archive_path)
        if set(observed_arrays) != set(all_arrays) or any(not np.array_equal(observed_arrays[name], all_arrays[name]) for name in all_arrays):
            raise Protocol248Error("recovered array archive differs")
    authority_sha = sha256(ROOT / "authority/freeze_record.json")
    bare = {
        "schema": SCHEMA,
        "authority_sha256": authority_sha,
        "classification": first["classification"],
        "scientific": first,
        "complete_evaluation_and_comparison_repeat_exact": True,
        "manufactured_controls": controls,
        "archive": file_record(archive_path, ROOT),
        "array_records": {name: array_record(value) for name, value in sorted(all_arrays.items())},
        "runtime": {
            "python": platform.python_version(), "platform": platform.platform(),
            "elapsed_wall_seconds": float(time.monotonic() - started), "peak_rss_bytes": current_peak_rss_bytes(),
        },
        "spacetime_evolution_executed": False,
        "surface_solve_executed": False,
        "parent_or_published_artifact_modified": False,
        "submitted_paper_edited": False,
        "bounded_three_grid_local_balance_transfer_established": first["classification"] == "G9-G10-G11-NATIVE-LOCAL-BALANCE-TRANSFER-PASS",
        "continuum_dynamical_horizon_claim_authorized": False,
        "integrated_or_global_balance_claim_authorized": False,
        "event_horizon_claim_authorized": False,
        "connected_topology_claim_authorized": False,
        "source_ownership_claim_authorized": False,
    }
    result = dict(bare)
    result["fingerprint"] = hashlib.sha256(b"protocol248-result-v1\0" + canonical(bare)).hexdigest()
    result_path = OUTPUT / "protocol248_result.json"
    if not result_path.exists():
        atomic_json(result_path, result)
    else:
        recovered = read_json(result_path)
        if not (
            check_fingerprint(recovered, b"protocol248-result-v1\0")
            and recovered.get("schema") == SCHEMA
            and recovered.get("authority_sha256") == authority_sha
            and recovered.get("classification") == first["classification"]
            and recovered.get("scientific") == first
            and recovered.get("complete_evaluation_and_comparison_repeat_exact") is True
            and recovered.get("manufactured_controls") == controls
            and recovered.get("archive") == file_record(archive_path, ROOT)
            and recovered.get("array_records") == {name: array_record(value) for name, value in sorted(all_arrays.items())}
        ):
            raise Protocol248Error("recovered result differs")
        result = recovered
    verify_freeze(ROOT)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "verify", "status"))
    args = parser.parse_args(argv)
    if args.command == "run":
        result = execute()
    elif args.command == "verify":
        runtime_preflight()
        verify_freeze(ROOT); validate_p245_p246(); validate_p247(); engine.load_background()
        result = {"status": "VERIFIED", "authority_sha256": sha256(ROOT / "authority/freeze_record.json")}
    else:
        path = OUTPUT / "protocol248_result.json"
        value = read_json(path) if path.is_file() else None
        result = {"status": "COMPLETE" if value else "NOT-STARTED", "classification": None if value is None else value["classification"]}
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
