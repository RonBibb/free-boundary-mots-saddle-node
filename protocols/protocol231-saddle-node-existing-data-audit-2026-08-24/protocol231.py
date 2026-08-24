#!/usr/bin/env python3
"""Read-only precision audit of the sealed Protocol-229-v4 records."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCHEMA = "protocol231-saddle-node-existing-data-audit-v1"
GRIDS = ("G9", "G10", "G11")


class Protocol231Error(RuntimeError):
    pass


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


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


def roots(academic_root):
    protocols = Path(academic_root).absolute() / "Discussion/protocols"
    return (
        protocols / "protocol229-free-boundary-mots-saddle-node-v4-2026-08-24",
        protocols / "protocol230-protocol229-archive-finalization-2026-08-24",
    )


def input_paths(academic_root):
    p229, p230 = roots(academic_root)
    protocols = p229.parent
    paths = {
        "p229/protocol": p229 / "PROTOCOL.md",
        "p229/source": p229 / "protocol229.py",
        "p229/core": p229 / "continuation_core.py",
        "p229/authority": p229 / "authority/freeze_record.json",
        "p230/result": p230 / "candidate-output/protocol230_result.json",
        "p227/source": protocols / "protocol227-direct-g11-mots-observables-2026-08-23/protocol227.py",
    }
    for grid in GRIDS:
        paths[f"p229/{grid}-json"] = p229 / f"candidate-output/protocol229_{grid}.json"
        paths[f"p229/{grid}-npz"] = p229 / f"candidate-output/protocol229_{grid}.npz"
    return paths


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    if path.exists() or path.is_symlink():
        raise Protocol231Error("output exists: " + path.name)
    with temporary.open("xb") as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def freeze(root, academic_root):
    root = Path(root).absolute(); inputs = input_paths(academic_root)
    if any((root / "authority").iterdir()) or (root / "candidate-output").exists():
        raise Protocol231Error("prospective namespace differs")
    for name, path in inputs.items():
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
            raise Protocol231Error("unsafe input: " + name)
    p230 = read_json(inputs["p230/result"])
    if p230.get("classification") != "FREE-BOUNDARY-MOTS-SADDLE-NODE-CLOSURE-PASS":
        raise Protocol231Error("Protocol230 prerequisite differs")
    authority = {
        "schema": SCHEMA,
        "status": "FROZEN",
        "diagnostic_only": True,
        "grids": list(GRIDS),
        "fixed_continuation_midpoint_for_log_fit": True,
        "fixed_half_exponent_comparison": True,
        "leave_one_out_required": True,
        "condition_trace_manufacture_authorized": False,
        "protocol229_reclassification_authorized": False,
        "sources": {name: record(root / name, root) for name in ("PROTOCOL.md", "protocol231.py", "tests/test_protocol231.py")},
        "inputs": {name: record(path) for name, path in inputs.items()},
        "candidate_output_absent_at_freeze": True,
    }
    authority["fingerprint"] = hashlib.sha256(b"protocol231-freeze\0" + canonical(authority)).hexdigest()
    atomic_json(root / "authority/freeze_record.json", authority)
    return authority


def verify(root, academic_root):
    root = Path(root).absolute(); authority = read_json(root / "authority/freeze_record.json")
    fingerprint = authority.pop("fingerprint", None)
    expected = hashlib.sha256(b"protocol231-freeze\0" + canonical(authority)).hexdigest()
    authority["fingerprint"] = fingerprint
    if authority.get("schema") != SCHEMA or fingerprint != expected:
        raise Protocol231Error("authority differs")
    for name, item in authority["sources"].items():
        if not same(root / item["path"], item):
            raise Protocol231Error("source differs: " + name)
    inputs = input_paths(academic_root)
    if set(inputs) != set(authority["inputs"]):
        raise Protocol231Error("input inventory differs")
    for name, item in authority["inputs"].items():
        if not same(inputs[name], item):
            raise Protocol231Error("input differs: " + name)
    return authority, inputs


def linear_fit(x, y):
    n = len(x); xm = sum(x) / n; ym = sum(y) / n
    denominator = sum((v - xm) ** 2 for v in x)
    if n < 2 or denominator <= 0:
        raise Protocol231Error("degenerate fit")
    slope = sum((a - xm) * (b - ym) for a, b in zip(x, y)) / denominator
    intercept = ym - slope * xm
    predictions = [intercept + slope * v for v in x]
    total = sum((v - ym) ** 2 for v in y)
    residual = sum((a - b) ** 2 for a, b in zip(y, predictions))
    return slope, intercept, 1.0 - residual / total if total > 0 else 1.0


def fixed_time_fit(times, separations, critical_time):
    offsets = [time - critical_time for time in times]
    if any(value <= 0 for value in offsets) or any(value <= 0 for value in separations):
        raise Protocol231Error("nonpositive scaling datum")
    exponent, intercept, r_squared = linear_fit([math.log(v) for v in offsets], [math.log(v) for v in separations])
    amplitude = math.exp(sum(math.log(y) - 0.5 * math.log(x) for x, y in zip(offsets, separations)) / len(offsets))
    fixed_predictions = [amplitude * math.sqrt(x) for x in offsets]
    maximum_relative = max(abs(a - b) / a for a, b in zip(separations, fixed_predictions))
    return {
        "free_log_exponent": exponent,
        "free_log_intercept": intercept,
        "free_log_R_squared": r_squared,
        "fixed_half_amplitude": amplitude,
        "fixed_half_maximum_relative_residual": maximum_relative,
    }


def scaling_diagnostic(result):
    pairs = result["square_root_pairs"]
    times = [float(item["time_over_ell"]) for item in pairs]
    separations = [float(item["area_separation"]) for item in pairs]
    critical = float(result["critical_time_estimate"])
    full = fixed_time_fit(times, separations, critical)
    leave_one_out = []
    for omitted in range(len(times)):
        fit = fixed_time_fit(times[:omitted] + times[omitted + 1:], separations[:omitted] + separations[omitted + 1:], critical)
        leave_one_out.append({"omitted_multiplier": pairs[omitted]["multiplier"], **fit})
    exponents = [item["free_log_exponent"] for item in leave_one_out]
    return {
        "continuation_midpoint_time": critical,
        "archived_coupled_fit": result["square_root_fit"],
        "archived_R_squared_semantics": "linear fit of squared area separation versus time",
        "archived_log_exponent_semantics": "separate log fit using the critical time inferred by the squared-area fit",
        "independent_midpoint_fit": full,
        "leave_one_out": leave_one_out,
        "leave_one_out_exponent_range": [min(exponents), max(exponents)],
        "drop_nearest_exponent": leave_one_out[0]["free_log_exponent"],
        "drop_farthest_exponent": leave_one_out[-1]["free_log_exponent"],
    }


def sequence_diagnostic(values):
    first = values[1] - values[0]; second = values[2] - values[1]
    monotone = bool(first == 0 or second == 0 or first * second > 0)
    scale = max(max(abs(value) for value in values), 1e-300)
    return {
        "values": values,
        "adjacent_differences": [first, second],
        "monotone": monotone,
        "absolute_spread": max(values) - min(values),
        "relative_spread": (max(values) - min(values)) / scale,
        "real_richardson_order_defined": bool(first * second > 0 and abs(second) > 0),
        "interpretation": "ordered sequence" if monotone else "nonmonotone three-grid scatter; no real Richardson order",
    }


def run(root, academic_root):
    root = Path(root).absolute(); _, inputs = verify(root, academic_root)
    output = root / "candidate-output"
    if output.exists() or output.is_symlink():
        raise Protocol231Error("candidate output already exists")
    output.mkdir()
    records = {}
    for grid in GRIDS:
        wrapper = read_json(inputs[f"p229/{grid}-json"])
        if wrapper.get("result", {}).get("passed") is not True:
            raise Protocol231Error(grid + " result differs")
        records[grid] = wrapper["result"]
    core_source = inputs["p229/core"].read_text(encoding="utf-8")
    p229_source = inputs["p229/source"].read_text(encoding="utf-8")
    p227_source = inputs["p227/source"].read_text(encoding="utf-8")
    normalization = {
        "right_phase_largest_component_positive_real": "pivot = int(np.argmax(np.abs(right)))" in core_source and "right *= phase" in core_source,
        "right_infinity_norm_one": "right /= max(float(np.max(np.abs(right)))" in core_source,
        "left_scaled_after_right_to_unit_overlap": "left /= np.conj(overlap)" in core_source,
        "adjoint_kind": "Euclidean conjugate-transpose eigenvector of the discrete collocation matrix",
        "post_normalization_overlap_is_independent_evidence": False,
    }
    condition_provenance = {
        "primary_spectral_condition_limit": 1.0e5,
        "primary_spectral_gate_belongs_to_protocol229_continuation": False,
        "protocol229_load_bearing_solver": "adaptive local second-order BVP with inherited residual/domain/cross-evaluator admission",
        "protocol229_source_mentions_jacobian_condition_number": "jacobian_condition_number" in p229_source,
        "protocol227_admission_mentions_condition": "condition" in p227_source[p227_source.index("def admitted"):p227_source.index("def cluster_trials")],
        "condition_number_trace_archived": False,
        "conclusion": "the 1e5 spectral condition gate is not load-bearing on Protocol229 continuation; no critical spectral-condition trace was archived",
    }
    sequences = {
        "critical_time": sequence_diagnostic([float(records[g]["critical_time_estimate"]) for g in GRIDS]),
        "critical_area": sequence_diagnostic([float(records[g]["critical_geometry"]["one_sided_cap_area"]) for g in GRIDS]),
        "transversality": sequence_diagnostic([float(records[g]["critical_coefficients"]["transversality_values"][1]) for g in GRIDS]),
        "quadratic": sequence_diagnostic([float(records[g]["critical_coefficients"]["quadratic_values"][1]) for g in GRIDS]),
    }
    result = {
        "schema": SCHEMA,
        "classification": "EXISTING-DATA-PRECISION-AUDIT-COMPLETE",
        "status": "COMPLETE",
        "authority_sha256": sha256(root / "authority/freeze_record.json"),
        "scaling": {grid: scaling_diagnostic(records[grid]) for grid in GRIDS},
        "three_grid_sequences": sequences,
        "mode_normalization": normalization,
        "condition_gate_provenance": condition_provenance,
        "protocol229_reclassified": False,
        "new_evolution_executed": False,
        "new_surface_or_operator_evaluation_executed": False,
        "continuum_order_claim_authorized": False,
    }
    result["fingerprint"] = hashlib.sha256(b"protocol231-result\0" + canonical(result)).hexdigest()
    atomic_json(output / "protocol231_result.json", result)
    verify(root, academic_root)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "verify", "run"):
        command = sub.add_parser(name); command.add_argument("--root", required=True); command.add_argument("--academic-root", required=True)
    args = vars(parser.parse_args(argv)); command = args.pop("command")
    if command == "freeze": value = freeze(**args)
    elif command == "verify": value = {"verified": True, "fingerprint": verify(**args)[0]["fingerprint"]}
    else: value = run(**args)
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
