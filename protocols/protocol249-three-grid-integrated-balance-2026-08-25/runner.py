#!/usr/bin/env python3
"""Archive-only three-grid finite-segment integrated balance."""

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
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT))

import engine248 as engine
from authority import file_record, sha256, verify_freeze
from integrated_balance_core import BRANE_TERMS, SEGMENT_STEPS, WIDTHS, compare_segments, classify, segment_record


SCHEMA = "protocol249-three-grid-integrated-balance-result-v1"
P244 = ROOT / "sealed-inputs/protocol244"
P246 = ROOT / "sealed-inputs/protocol246"
P247 = ROOT / "sealed-inputs/protocol247"
P248 = ROOT / "sealed-inputs/protocol248"
OUTPUT = ROOT / "candidate-output"
DT = 3.125e-5
GRIDS = ("G9", "G10", "G11")
FIELDS = ("q", "v", "source", "memory")
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
EXPECTED_RUNTIME = {"python": "3.8.10", "numpy": "1.24.4", "scipy": "1.10.1", "system": "Linux", "machine": "aarch64"}


class Protocol249Error(RuntimeError): pass


def canonical(value): return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def check_fingerprint(record, prefix):
    bare = dict(record); observed = bare.pop("fingerprint", None)
    return observed == hashlib.sha256(prefix + canonical(bare)).hexdigest()


def array_record(value):
    array = np.ascontiguousarray(value)
    return {"dtype": array.dtype.str, "shape": list(array.shape), "byte_count": int(array.nbytes), "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest()}


def regular(path):
    path = Path(path); return path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1 and (path.stat().st_mode & 0o222) == 0


def load_npz(path):
    if not regular(path): raise Protocol249Error(f"unsafe archive: {Path(path).name}")
    with np.load(path, allow_pickle=False) as archive: return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def runtime_preflight():
    if not sys.dont_write_bytecode or any(os.environ.get(name) != "1" for name in THREAD_VARS): raise Protocol249Error("runtime controls differ")
    if Path(getattr(sys.modules.get("__main__"), "__file__", "")).resolve() != (ROOT / "bootstrap.py").resolve(): raise Protocol249Error("authorized bootstrap was bypassed")
    observed = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "system": platform.system(), "machine": platform.machine()}
    if observed != EXPECTED_RUNTIME: raise Protocol249Error(f"live numerical runtime differs: {observed}")


def validate_prerequisites():
    p246 = read_json(P246 / "protocol246_result.json")
    p248_result_path = P248 / "candidate-output/protocol248_result.json"
    p248_arrays_path = P248 / "candidate-output/protocol248_balance_transfer_arrays.npz"
    p248 = read_json(p248_result_path)
    if not (check_fingerprint(p246, b"protocol246-result-v1\0") and p246.get("classification") == "FULL-HALF-NATIVE-BALANCE-CONSISTENCY-PASS" and p246.get("complete_comparison_repeat_exact") is True and all(p246.get("scientific", {}).get("gates", {}).values())):
        raise Protocol249Error("Protocol 246 prerequisite differs")
    if not (
        check_fingerprint(p248, b"protocol248-result-v1\0")
        and p248.get("schema") == "protocol248-three-grid-native-balance-transfer-result-v1"
        and p248.get("authority_sha256") == sha256(P248 / "freeze_record.json")
        and p248.get("classification") == "G9-G10-G11-NATIVE-LOCAL-BALANCE-TRANSFER-PASS"
        and all(p248.get("scientific", {}).get("gates", {}).values())
        and p248.get("complete_evaluation_and_comparison_repeat_exact") is True
        and p248.get("bounded_three_grid_local_balance_transfer_established") is True
        and p248.get("archive") == file_record(p248_arrays_path, P248)
        and p248.get("parent_or_published_artifact_modified") is False
        and p248.get("submitted_paper_edited") is False
    ):
        raise Protocol249Error("Protocol 248 prerequisite differs")
    load_npz(p248_arrays_path)
    return p246, p248


def expected_shape(label, field):
    spatial = {"G9": (113, 211), "G10": (129, 241), "G11": (145, 271)}[label]
    return spatial + (9 if field in {"q", "v"} else 3,)


def validate_grid_inputs():
    p244 = read_json(P244 / "candidate-output/protocol244_result.json")
    p244_arrays_path = P244 / "candidate-output/protocol244_dense_tube_arrays.npz"
    p247 = read_json(P247 / "candidate-output/protocol247_result.json")
    if not (
        check_fingerprint(p244, b"protocol244-result-v1\0")
        and p244.get("authority_sha256") == sha256(P244 / "freeze_record.json")
        and p244.get("scientific", {}).get("classification") == "FULL-DT-DENSE-G10-OUTER-MARGINAL-TUBE-PASS"
        and p244.get("full_dt_native_balance_replay_authorized") is True
        and p244.get("archive") == file_record(p244_arrays_path, P244)
    ): raise Protocol249Error("Protocol 244 prerequisite differs")
    load_npz(p244_arrays_path)
    if not (
        check_fingerprint(p247, b"protocol247-result-v1\0")
        and p247.get("authority_sha256") == sha256(P247 / "freeze_record.json")
        and p247.get("scientific", {}).get("classification") == "G9-G10-G11-BOUNDED-OUTER-TUBE-SPATIAL-TRANSFER-PASS"
        and p247.get("archive_only_balance_transfer_authorized") is True
    ): raise Protocol249Error("Protocol 247 prerequisite differs")

    states_by_grid, profiles_by_grid = {}, {}
    for label in GRIDS:
        parent = P244 if label == "G10" else P247
        output = parent / "candidate-output"
        parent_result = p244 if label == "G10" else p247
        states, profiles = {}, {}
        for step in SEGMENT_STEPS:
            checkpoint_path = output / f"{label}_full_step{step:04d}.npz"
            checkpoint = read_json(output / f"{label}_full_step{step:04d}.json")
            arrays = load_npz(checkpoint_path)
            checkpoint_prefix = b"protocol244-checkpoint-v1\0" if label == "G10" else b"protocol247-grid-checkpoint-v1\0"
            checkpoint_schema = "protocol244-full-dt-g10-checkpoint-v1" if label == "G10" else "protocol247-grid-checkpoint-v1"
            if not (
                check_fingerprint(checkpoint, checkpoint_prefix) and checkpoint.get("schema") == checkpoint_schema
                and checkpoint.get("authority_sha256") == parent_result["authority_sha256"]
                and checkpoint.get("grid") == label and checkpoint.get("end_step") == step and checkpoint.get("dt") == DT
                and checkpoint.get("passed") is True and checkpoint.get("endpoint_repeat_exact") is True
                and checkpoint.get("archive") == file_record(checkpoint_path, parent)
                and set(arrays) == set(FIELDS)
                and all(arrays[field].dtype == np.float64 and arrays[field].shape == expected_shape(label, field) and np.all(np.isfinite(arrays[field])) and array_record(arrays[field]) == checkpoint["arrays"][field] for field in FIELDS)
            ): raise Protocol249Error(f"checkpoint differs: {label}/{step}")

            profile_path = output / f"{label}_outer_step{step:04d}.npz"
            leaf = read_json(output / f"{label}_outer_step{step:04d}.json"); profile = load_npz(profile_path)
            leaf_prefix = b"protocol244-outer-leaf-v1\0" if label == "G10" else b"protocol247-grid-outer-leaf-v1\0"
            leaf_schema = "protocol244-full-dt-g10-outer-leaf-v1" if label == "G10" else "protocol247-grid-outer-leaf-v1"
            expected_evaluation = parent_result["scientific"]["evaluation"]["leaves"][str(step)] if label == "G10" else parent_result["scientific"]["grids"][label]["leaves"][str(step)]
            if not (
                check_fingerprint(leaf, leaf_prefix) and leaf.get("schema") == leaf_schema
                and leaf.get("authority_sha256") == parent_result["authority_sha256"] and leaf.get("step") == step
                and leaf.get("passed") is True and leaf.get("solve_repeat_exact") is True
                and leaf.get("archive") == file_record(profile_path, parent)
                and set(profile) == {"theta", "rho", "slope"}
                and all(profile[field].dtype == np.float64 and profile[field].shape == (501,) and np.all(np.isfinite(profile[field])) and array_record(profile[field]) == leaf["profile_arrays"][field] for field in profile)
                and leaf.get("evaluation") == expected_evaluation
            ): raise Protocol249Error(f"outer leaf differs: {label}/{step}")
            states[step], profiles[step] = arrays, profile
        if not all(np.array_equal(profiles[SEGMENT_STEPS[0]]["theta"], profiles[step]["theta"]) for step in SEGMENT_STEPS[1:]): raise Protocol249Error(f"theta grids differ: {label}")
        states_by_grid[label], profiles_by_grid[label] = states, profiles
    return states_by_grid, profiles_by_grid


def compute_charges(states_by_grid, profiles_by_grid):
    result, arrays = {}, {}
    for label in GRIDS:
        sample = states_by_grid[label][SEGMENT_STEPS[0]]["q"]
        z = np.linspace(1.0, math.e, sample.shape[0]); r = np.linspace(0.0, 10.0, sample.shape[1])
        result[label] = {}
        table = np.empty((len(WIDTHS), len(SEGMENT_STEPS)), dtype=np.float64)
        for iw, width in enumerate(WIDTHS):
            result[label][width] = {}
            for it, step in enumerate(SEGMENT_STEPS):
                state = states_by_grid[label][step]
                prepared = engine.prepare_capped_expansion_slice(state["q"], state["v"], z, r, stencil_width=width)
                charge = engine.reflected_cap_charge(state["q"], state["v"], z, r, profiles_by_grid[label][step], stencil_width=width, prepared=prepared)
                value = float(charge["generalized_hawking_ads_charge_kappa5_squared_E"])
                if not math.isfinite(value): raise Protocol249Error("nonfinite charge")
                result[label][width][step] = value; table[iw, it] = value
        arrays[f"{label}_charges_width_by_step"] = table
    return result, arrays


def finite_record(value):
    if isinstance(value, dict): return all(finite_record(item) for item in value.values())
    if isinstance(value, list): return all(finite_record(item) for item in value)
    if isinstance(value, float): return math.isfinite(value)
    return True


def build_science(p246, p248, charges):
    local = p248["scientific"]["local_grid_results"]
    segments = {label: {str(width): segment_record(local[label], charges[label][width], width) for width in WIDTHS} for label in GRIDS}
    stencil = {}
    for label in GRIDS:
        stencil[label] = {}
        for left, right in ((5, 7), (7, 9), (5, 9)):
            stencil[label][f"{left}-{right}"] = compare_segments(segments[label][str(left)], segments[label][str(right)])
    spatial = {}
    for left, right in (("G9", "G10"), ("G10", "G11")):
        spatial[f"{left}-{right}"] = {str(width): compare_segments(segments[left][str(width)], segments[right][str(width)]) for width in WIDTHS}
    all_segments = [segments[label][str(width)] for label in GRIDS for width in WIDTHS]
    gates = {
        "prerequisite_admission": True,
        "segment_orientation": bool(all(item["passes"]["orientation"] for item in all_segments)),
        "charge_quadrature_closure": bool(all(item["passes"]["quadrature"] for item in all_segments)),
        "integrated_flux_closure": bool(all(item["passes"]["physical_flux"] for item in all_segments)),
        "brane_ledger_completeness": bool(all(item["passes"]["term_sum"] and set(BRANE_TERMS) <= set(item["integrated_terms"]) for item in all_segments)),
        "stencil_robustness": bool(all(item["passed"] for pairs in stencil.values() for item in pairs.values())),
        "spatial_transfer": bool(all(item["passed"] for widths in spatial.values() for item in widths.values())),
        "temporal_control_admission": bool(p246.get("classification") == "FULL-HALF-NATIVE-BALANCE-CONSISTENCY-PASS" and all(p246.get("scientific", {}).get("gates", {}).values())),
    }
    maxima = {
        "charge_target_relative": max(item["residuals"]["charge_target_relative"] for item in all_segments),
        "charge_flux_relative": max(item["residuals"]["charge_flux_relative"] for item in all_segments),
        "target_flux_relative": max(item["residuals"]["target_flux_relative"] for item in all_segments),
        "term_sum_balance_norm_relative": max(item["residuals"]["term_sum_balance_norm_relative"] for item in all_segments),
        "stencil_metric": max(max(item["metrics"].values()) for pairs in stencil.values() for item in pairs.values()),
        "spatial_metric": max(max(item["metrics"].values()) for widths in spatial.values() for item in widths.values()),
    }
    science = {
        "classification": classify(gates), "gates": gates, "segment_steps": list(SEGMENT_STEPS),
        "segment_times": [step * DT for step in SEGMENT_STEPS], "quadrature": "four-node composite trapezoid",
        "classification_driving_path": "centered", "stencil_widths": list(WIDTHS),
        "brane_endpoint_terms": list(BRANE_TERMS), "maximum_discrepancies": maxima,
        "grids": segments, "within_grid_stencil_comparisons": stencil, "adjacent_grid_comparisons": spatial,
        "protocol246_temporal_control_classification": p246["classification"],
    }
    if not finite_record(science): raise Protocol249Error("nonfinite science record")
    return science


def current_peak_rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss); return value if sys.platform == "darwin" else value * 1024


def fsync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def atomic_bytes(path, payload):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink(): raise Protocol249Error("output namespace is not fresh")
    with temporary.open("xb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    os.replace(temporary, path); fsync_directory(path.parent)


def atomic_json(path, value):
    atomic_bytes(path, canonical(value))
    if read_json(path) != value: raise Protocol249Error("JSON immediate replay failed")


def atomic_npz(path, arrays):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink(): raise Protocol249Error("output namespace is not fresh")
    with temporary.open("xb") as stream:
        np.savez(stream, **{name: np.ascontiguousarray(value) for name, value in arrays.items()}); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    os.replace(temporary, path); fsync_directory(path.parent)
    observed = load_npz(path)
    if set(observed) != set(arrays) or any(not np.array_equal(observed[name], arrays[name]) for name in arrays): raise Protocol249Error("NPZ immediate replay failed")


def execute():
    started = time.monotonic(); runtime_preflight(); verify_freeze(ROOT)
    p246, p248 = validate_prerequisites(); states, profiles = validate_grid_inputs()
    first_charges, first_arrays = compute_charges(states, profiles); second_charges, second_arrays = compute_charges(states, profiles)
    if first_charges != second_charges or set(first_arrays) != set(second_arrays) or any(not np.array_equal(first_arrays[name], second_arrays[name]) for name in first_arrays): raise Protocol249Error("charge repeat differs")
    first = build_science(p246, p248, first_charges); second = build_science(p246, p248, second_charges)
    if first != second: raise Protocol249Error("integration/comparison repeat differs")
    if not OUTPUT.exists(): OUTPUT.mkdir(mode=0o755); fsync_directory(ROOT)
    if OUTPUT.is_symlink() or not OUTPUT.is_dir(): raise Protocol249Error("candidate-output is unsafe")
    allowed = {"protocol249_charge_arrays.npz", "protocol249_result.json"}
    if not {path.name for path in OUTPUT.iterdir()} <= allowed: raise Protocol249Error("candidate-output inventory differs")
    archive_path = OUTPUT / "protocol249_charge_arrays.npz"
    if not archive_path.exists(): atomic_npz(archive_path, first_arrays)
    else:
        observed = load_npz(archive_path)
        if set(observed) != set(first_arrays) or any(not np.array_equal(observed[name], first_arrays[name]) for name in first_arrays): raise Protocol249Error("recovered charge archive differs")
    authority_sha = sha256(ROOT / "authority/freeze_record.json")
    bare = {
        "schema": SCHEMA, "authority_sha256": authority_sha, "classification": first["classification"],
        "scientific": first, "complete_charge_and_integration_repeat_exact": True,
        "archive": file_record(archive_path, ROOT), "array_records": {name: array_record(value) for name, value in sorted(first_arrays.items())},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(), "elapsed_wall_seconds": float(time.monotonic() - started), "peak_rss_bytes": current_peak_rss_bytes()},
        "spacetime_evolution_executed": False, "surface_solve_executed": False,
        "parent_or_published_artifact_modified": False, "submitted_paper_edited": False,
        "finite_segment_integrated_balance_established": first["classification"] == "G9-G10-G11-FINITE-SEGMENT-INTEGRATED-BALANCE-PASS",
        "continuum_dynamical_horizon_claim_authorized": False, "event_horizon_claim_authorized": False,
        "connected_topology_claim_authorized": False, "global_intersector_charge_claim_authorized": False,
        "source_ownership_claim_authorized": False,
    }
    result = dict(bare); result["fingerprint"] = hashlib.sha256(b"protocol249-result-v1\0" + canonical(bare)).hexdigest()
    result_path = OUTPUT / "protocol249_result.json"
    if not result_path.exists(): atomic_json(result_path, result)
    else:
        recovered = read_json(result_path)
        if not (
            check_fingerprint(recovered, b"protocol249-result-v1\0")
            and recovered.get("schema") == SCHEMA and recovered.get("authority_sha256") == authority_sha
            and recovered.get("classification") == first["classification"] and recovered.get("scientific") == first
            and recovered.get("complete_charge_and_integration_repeat_exact") is True
            and recovered.get("archive") == file_record(archive_path, ROOT)
            and recovered.get("array_records") == {name: array_record(value) for name, value in sorted(first_arrays.items())}
            and recovered.get("finite_segment_integrated_balance_established") == bare["finite_segment_integrated_balance_established"]
        ): raise Protocol249Error("recovered result differs")
        result = recovered
    verify_freeze(ROOT); return result


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("run", "verify", "status")); args = parser.parse_args(argv)
    if args.command == "run": result = execute()
    elif args.command == "verify": runtime_preflight(); verify_freeze(ROOT); validate_prerequisites(); validate_grid_inputs(); result = {"status": "VERIFIED", "authority_sha256": sha256(ROOT / "authority/freeze_record.json")}
    else:
        path = OUTPUT / "protocol249_result.json"; value = read_json(path) if path.is_file() else None
        result = {"status": "COMPLETE" if value else "NOT-STARTED", "classification": None if value is None else value["classification"]}
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__": main()
