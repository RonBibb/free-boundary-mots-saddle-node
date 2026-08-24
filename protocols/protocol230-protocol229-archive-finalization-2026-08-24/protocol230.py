#!/usr/bin/env python3
"""Archive-only finalization of the completed Protocol 229 v4 grid artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
SCHEMA = "protocol230-protocol229-archive-finalization-v1"
P229_SCHEMA = "protocol229-free-boundary-mots-saddle-node-v4"
P229_GRID_SCHEMA = "protocol229-grid-continuation-v4"
P229_ROOT_NAME = "protocol229-free-boundary-mots-saddle-node-v4-2026-08-24"
P229_AUTHORITY_SHA256 = "ea24614eb1962375c4e96fcbeaca9a40e292b1c33361f5959db92495f2ca75bc"
DT = 0.00003125
GRIDS = ("G9", "G10", "G11")
ADJACENT_PAIRS = (("G9", "G10"), ("G10", "G11"))
GRID_CHECKS = {
    "Protocol228_endpoint_replay_bitwise_exact",
    "adjoint_transversality_resolved_nonzero",
    "critical_principal_magnitude_below_0_02",
    "invariant_area_square_root_scaling_pass",
    "opposite_principal_sign_bracket_found",
    "outer_and_inner_anchor_signs_exact",
    "quadratic_nondegeneracy_resolved_nonzero",
    "simple_real_sign_definite_principal_mode",
    "zero_time_bracket_below_dt_over_64",
}


class Protocol230Error(RuntimeError):
    pass


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


def regular_file(path):
    path = Path(path)
    return bool(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1)


def file_record(path, relative_to=None):
    path = Path(path).resolve()
    if not regular_file(path):
        raise Protocol230Error("unsafe file: " + str(path))
    return {
        "path": path.relative_to(Path(relative_to).resolve()).as_posix() if relative_to else str(path),
        "byte_count": path.stat().st_size,
        "sha256": sha256(path),
    }


def validate_file_record(path, item):
    path = Path(path)
    return bool(
        set(item) == {"path", "byte_count", "sha256"}
        and type(item["path"]) is str
        and type(item["byte_count"]) is int
        and type(item["sha256"]) is str
        and regular_file(path)
        and path.stat().st_size == item["byte_count"]
        and sha256(path) == item["sha256"]
    )


def atomic_json(path, value):
    path = Path(path)
    payload = canonical(value)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise Protocol230Error("output path already exists")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def p229_root(academic_root):
    return Path(academic_root).resolve() / "Discussion/protocols" / P229_ROOT_NAME


def p229_direct_inputs(academic_root):
    upstream = p229_root(academic_root)
    paths = {
        "p229/authority": upstream / "authority/freeze_record.json",
        "p229/source": upstream / "protocol229.py",
    }
    for label in GRIDS:
        paths[f"p229/{label}/json"] = upstream / "candidate-output" / f"protocol229_{label}.json"
        paths[f"p229/{label}/npz"] = upstream / "candidate-output" / f"protocol229_{label}.npz"
    return paths


def p229_freeze_fingerprint(value):
    return hashlib.sha256(b"protocol229-freeze\0" + canonical(value)).hexdigest()


def p229_grid_fingerprint(value):
    return hashlib.sha256(b"protocol229-grid\0" + canonical(value)).hexdigest()


def validate_p229_authority(academic_root):
    upstream = p229_root(academic_root)
    authority_path = upstream / "authority/freeze_record.json"
    if sha256(authority_path) != P229_AUTHORITY_SHA256:
        raise Protocol230Error("Protocol 229 authority SHA-256 differs")
    authority = read_json(authority_path)
    fingerprint = authority.pop("fingerprint", None)
    expected = p229_freeze_fingerprint(authority)
    authority["fingerprint"] = fingerprint
    if authority.get("schema") != P229_SCHEMA or authority.get("status") != "FROZEN" or fingerprint != expected:
        raise Protocol230Error("Protocol 229 authority fingerprint differs")
    if authority.get("dt") != DT or tuple(authority.get("grids", ())) != ("G10", "G9", "G11"):
        raise Protocol230Error("Protocol 229 frozen grid contract differs")
    if authority.get("candidate_output_absent_at_freeze") is not True:
        raise Protocol230Error("Protocol 229 freeze lifecycle differs")
    for key in (
        "parent_solve_or_repair_authorized", "continuum_theorem_claim_authorized",
        "event_horizon_claim_authorized", "phase_selection_claim_authorized",
        "source_ownership_claim_authorized",
    ):
        if authority.get(key) is not False:
            raise Protocol230Error("Protocol 229 claim firewall differs: " + key)
    for name, item in authority.get("sources", {}).items():
        path = upstream / item.get("path", "")
        if not validate_file_record(path, item):
            raise Protocol230Error("Protocol 229 source differs: " + name)
    for name, item in authority.get("inputs", {}).items():
        path = Path(item.get("path", ""))
        if not validate_file_record(path, item):
            raise Protocol230Error("Protocol 229 inherited input differs: " + name)
    return authority


def require_finite_number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise Protocol230Error("nonfinite or nonnumeric field: " + name)
    return float(value)


def validate_npz_inventory(path, label):
    expected = {f"{label}_critical_theta.npy", f"{label}_critical_rho.npy", f"{label}_critical_slope.npy"}
    try:
        with ZipFile(path, "r") as archive:
            if set(archive.namelist()) != expected or archive.testzip() is not None:
                raise Protocol230Error("Protocol 229 NPZ inventory differs: " + label)
    except Protocol230Error:
        raise
    except Exception as error:
        raise Protocol230Error("Protocol 229 NPZ is unreadable: " + label) from error


def validate_grid_artifact(academic_root, label, expected_inputs=None):
    upstream = p229_root(academic_root)
    json_path = upstream / "candidate-output" / f"protocol229_{label}.json"
    npz_path = upstream / "candidate-output" / f"protocol229_{label}.npz"
    if expected_inputs is not None:
        if not validate_file_record(json_path, expected_inputs[f"p229/{label}/json"]):
            raise Protocol230Error("bound grid JSON differs: " + label)
        if not validate_file_record(npz_path, expected_inputs[f"p229/{label}/npz"]):
            raise Protocol230Error("bound grid NPZ differs: " + label)
    value = read_json(json_path)
    if set(value) != {"schema", "authority_sha256", "grid", "archive", "result", "fingerprint"}:
        raise Protocol230Error("grid wrapper schema differs: " + label)
    fingerprint = value.pop("fingerprint")
    expected = p229_grid_fingerprint(value)
    value["fingerprint"] = fingerprint
    if (
        fingerprint != expected or value["schema"] != P229_GRID_SCHEMA or value["grid"] != label
        or value["authority_sha256"] != P229_AUTHORITY_SHA256
    ):
        raise Protocol230Error("grid wrapper provenance differs: " + label)
    archive_record = value["archive"]
    if archive_record.get("path") != f"candidate-output/protocol229_{label}.npz":
        raise Protocol230Error("grid NPZ path differs: " + label)
    if not validate_file_record(npz_path, archive_record):
        raise Protocol230Error("grid NPZ record differs: " + label)
    validate_npz_inventory(npz_path, label)
    result = value["result"]
    required = {
        "schema", "grid", "passed", "checks", "continuation_trace", "zero_bracket",
        "critical_time_estimate", "critical_principal_eigenvalue", "critical_geometry",
        "critical_coefficients", "square_root_pairs", "square_root_fit",
    }
    if set(result) != required or result["schema"] != P229_GRID_SCHEMA or result["grid"] != label:
        raise Protocol230Error("grid result schema differs: " + label)
    if type(result["passed"]) is not bool or result["passed"] is not True:
        raise Protocol230Error("grid did not pass: " + label)
    checks = result["checks"]
    if set(checks) != GRID_CHECKS or any(type(checks[name]) is not bool or checks[name] is not True for name in GRID_CHECKS):
        raise Protocol230Error("per-grid gates differ: " + label)
    critical_time = require_finite_number(result["critical_time_estimate"], label + " critical time")
    critical_lambda = require_finite_number(result["critical_principal_eigenvalue"], label + " critical eigenvalue")
    area = require_finite_number(result["critical_geometry"].get("one_sided_cap_area"), label + " critical area")
    transversality = result["critical_coefficients"].get("transversality_values")
    quadratic = result["critical_coefficients"].get("quadratic_values")
    if not isinstance(transversality, list) or len(transversality) != 2:
        raise Protocol230Error("transversality record differs: " + label)
    if not isinstance(quadratic, list) or len(quadratic) != 2:
        raise Protocol230Error("quadratic record differs: " + label)
    transversality = [require_finite_number(v, label + " transversality") for v in transversality]
    quadratic = [require_finite_number(v, label + " quadratic") for v in quadratic]
    fit = result["square_root_fit"]
    fit_summary = {
        "R_squared": require_finite_number(fit.get("R_squared"), label + " R squared"),
        "critical_time": require_finite_number(fit.get("critical_time"), label + " fit critical time"),
        "log_exponent": require_finite_number(fit.get("log_exponent"), label + " exponent"),
    }
    if not (0.001 <= critical_time <= 0.0015 and abs(critical_lambda) < 0.02 and area > 0):
        raise Protocol230Error("grid scalar gate differs: " + label)
    if not (fit_summary["R_squared"] >= 0.98 and 0.40 <= fit_summary["log_exponent"] <= 0.60):
        raise Protocol230Error("square-root fit gate differs: " + label)
    return result, {
        "passed": True,
        "critical_time_estimate": critical_time,
        "critical_principal_eigenvalue": critical_lambda,
        "one_sided_cap_area": area,
        "square_root_fit": fit_summary,
        "transversality_values": transversality,
        "quadratic_values": quadratic,
        "grid_json": file_record(json_path),
        "grid_npz": file_record(npz_path),
    }


def relative_difference(left, right):
    return float(abs(left - right) / max(abs(left), abs(right), 1e-300))


def same_nonzero_sign(left, right):
    return bool(left != 0.0 and right != 0.0 and ((left > 0.0) == (right > 0.0)))


def cross_grid_checks(records):
    transfers = {}
    for left, right in ADJACENT_PAIRS:
        a, b = records[left], records[right]
        trans_left = a["critical_coefficients"]["transversality_values"][1]
        trans_right = b["critical_coefficients"]["transversality_values"][1]
        quad_left = a["critical_coefficients"]["quadratic_values"][1]
        quad_right = b["critical_coefficients"]["quadratic_values"][1]
        item = {
            "critical_time_absolute_difference": float(abs(a["critical_time_estimate"] - b["critical_time_estimate"])),
            "critical_area_relative_difference": relative_difference(
                a["critical_geometry"]["one_sided_cap_area"], b["critical_geometry"]["one_sided_cap_area"]
            ),
            "transversality_relative_difference": relative_difference(trans_left, trans_right),
            "quadratic_relative_difference": relative_difference(quad_left, quad_right),
            "transversality_sign_agrees": same_nonzero_sign(trans_left, trans_right),
            "quadratic_sign_agrees": same_nonzero_sign(quad_left, quad_right),
        }
        item["passed"] = bool(
            item["critical_time_absolute_difference"] <= DT / 4.0
            and item["critical_area_relative_difference"] < 0.01
            and item["transversality_sign_agrees"]
            and item["quadratic_sign_agrees"]
            and item["transversality_relative_difference"] < 0.20
            and item["quadratic_relative_difference"] < 0.20
        )
        transfers[left + "-" + right] = item
    return transfers, bool(all(item["passed"] for item in transfers.values()))


def freeze_fingerprint(value):
    return hashlib.sha256(b"protocol230-freeze\0" + canonical(value)).hexdigest()


def result_fingerprint(value):
    return hashlib.sha256(b"protocol230-result\0" + canonical(value)).hexdigest()


def freeze(root, academic_root):
    root = Path(root).resolve()
    authority_dir = root / "authority"
    candidate = root / "candidate-output"
    if candidate.exists() or not authority_dir.is_dir() or any(authority_dir.iterdir()):
        raise Protocol230Error("prospective output lifecycle differs")
    validate_p229_authority(academic_root)
    direct_inputs = p229_direct_inputs(academic_root)
    for label in GRIDS:
        validate_grid_artifact(academic_root, label)
    sources = ("PROTOCOL.md", "protocol230.py", "tests/test_protocol230.py")
    authority = {
        "schema": SCHEMA,
        "status": "FROZEN",
        "scope": "archive-only-finalization-without-scientific-reexecution",
        "protocol229_authority_sha256": P229_AUTHORITY_SHA256,
        "dt": DT,
        "grids": list(GRIDS),
        "adjacent_pairs": [list(pair) for pair in ADJACENT_PAIRS],
        "thresholds": {
            "critical_time_absolute_difference_max": DT / 4.0,
            "critical_area_relative_difference_strict_max": 0.01,
            "transversality_relative_difference_strict_max": 0.20,
            "quadratic_relative_difference_strict_max": 0.20,
            "matching_nonzero_coefficient_signs_required": True,
        },
        "sources": {name: file_record(root / name, root) for name in sources},
        "inputs": {name: file_record(path) for name, path in direct_inputs.items()},
        "candidate_output_absent_at_freeze": True,
        "scientific_reexecution_authorized": False,
        "continuum_theorem_claim_authorized": False,
        "event_horizon_claim_authorized": False,
        "phase_selection_claim_authorized": False,
        "source_ownership_claim_authorized": False,
    }
    authority["fingerprint"] = freeze_fingerprint(authority)
    atomic_json(authority_dir / "freeze_record.json", authority)
    return authority


def verify_authority(root, academic_root):
    root = Path(root).resolve()
    path = root / "authority/freeze_record.json"
    if not regular_file(path):
        raise Protocol230Error("freeze authority missing or unsafe")
    authority = read_json(path)
    fingerprint = authority.pop("fingerprint", None)
    expected = freeze_fingerprint(authority)
    authority["fingerprint"] = fingerprint
    if authority.get("schema") != SCHEMA or authority.get("status") != "FROZEN" or fingerprint != expected:
        raise Protocol230Error("freeze authority differs")
    for name, item in authority.get("sources", {}).items():
        if not validate_file_record(root / item.get("path", ""), item):
            raise Protocol230Error("frozen source differs: " + name)
    direct_inputs = p229_direct_inputs(academic_root)
    if set(direct_inputs) != set(authority.get("inputs", {})):
        raise Protocol230Error("direct input inventory differs")
    for name, path_value in direct_inputs.items():
        if not validate_file_record(path_value, authority["inputs"][name]):
            raise Protocol230Error("frozen input differs: " + name)
    validate_p229_authority(academic_root)
    records = {}
    summaries = {}
    for label in GRIDS:
        records[label], summaries[label] = validate_grid_artifact(academic_root, label, authority["inputs"])
    return authority, records, summaries


def validate_result(path, authority_sha256):
    if not regular_file(path):
        raise Protocol230Error("final result missing or unsafe")
    value = read_json(path)
    fingerprint = value.pop("fingerprint", None)
    expected = result_fingerprint(value)
    value["fingerprint"] = fingerprint
    if fingerprint != expected or value.get("schema") != SCHEMA or value.get("authority_sha256") != authority_sha256:
        raise Protocol230Error("final result fingerprint or authority differs")
    required_false = (
        "scientific_reexecution_performed", "continuum_theorem_claim_authorized",
        "event_horizon_claim_authorized", "phase_selection_claim_authorized",
        "source_ownership_claim_authorized",
    )
    if any(value.get(name) is not False for name in required_false):
        raise Protocol230Error("final claim firewall differs")
    if value.get("archive_only_finalization") is not True:
        raise Protocol230Error("finalization scope differs")
    return value


def build_result(authority, authority_sha256, records, summaries):
    transfers, transfer_pass = cross_grid_checks(records)
    all_grid_pass = bool(all(records[label]["passed"] is True for label in GRIDS))
    passed = bool(all_grid_pass and transfer_pass)
    value = {
        "schema": SCHEMA,
        "authority_sha256": authority_sha256,
        "protocol229_authority_sha256": authority["protocol229_authority_sha256"],
        "status": "PASS" if passed else "REVIEW",
        "classification": (
            "FREE-BOUNDARY-MOTS-SADDLE-NODE-CLOSURE-PASS"
            if passed else "SADDLE-NODE-CONDITIONS-NOT-SATISFIED"
        ),
        "archive_only_finalization": True,
        "scientific_reexecution_performed": False,
        "grid_summaries": summaries,
        "adjacent_grid_transfers": transfers,
        "all_grid_gates_pass": all_grid_pass,
        "all_cross_grid_gates_pass": transfer_pass,
        "resolved_free_boundary_mots_saddle_node_claim_authorized": passed,
        "continuum_theorem_claim_authorized": False,
        "event_horizon_claim_authorized": False,
        "phase_selection_claim_authorized": False,
        "source_ownership_claim_authorized": False,
    }
    value["fingerprint"] = result_fingerprint(value)
    return value


def finalize(root, academic_root):
    root = Path(root).resolve()
    authority, records, summaries = verify_authority(root, academic_root)
    authority_path = root / "authority/freeze_record.json"
    authority_sha256 = sha256(authority_path)
    output = root / "candidate-output"
    final_path = output / "protocol230_result.json"
    expected = build_result(authority, authority_sha256, records, summaries)
    if output.exists():
        if not output.is_dir() or output.is_symlink() or set(path.name for path in output.iterdir()) != {final_path.name}:
            raise Protocol230Error("candidate output inventory differs")
        adopted = validate_result(final_path, authority_sha256)
        if adopted != expected:
            raise Protocol230Error("existing final result differs from recomputation")
        return adopted
    output.mkdir(mode=0o755)
    value = expected
    atomic_json(final_path, value)
    validated = validate_result(final_path, authority_sha256)
    if validated != value:
        raise Protocol230Error("fresh final result reload differs")
    return validated


def main(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "verify", "finalize"):
        command = commands.add_parser(name)
        command.add_argument("--root", required=True)
        command.add_argument("--academic-root", required=True)
    values = vars(parser.parse_args(argv))
    command = values.pop("command")
    if command == "freeze":
        result = freeze(**values)
    elif command == "verify":
        authority, records, summaries = verify_authority(**values)
        result = {
            "verified": True,
            "authority_fingerprint": authority["fingerprint"],
            "grids": sorted(records),
            "summaries": summaries,
        }
    else:
        result = finalize(**values)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
