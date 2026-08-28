#!/usr/bin/env python3
"""Freeze and verify Protocol 248 source and inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AUTHORITY = ROOT / "authority/freeze_record.json"
SCHEMA = "protocol248-three-grid-native-balance-transfer-freeze-v1"
PREFIX = b"protocol248-freeze-v1\0"
SOURCES = (
    "PROTOCOL.md", "README.md", "authority.py", "bootstrap.py", "runner.py", "engine245.py",
    "dense_balance_core.py", "spatial_balance_core.py", "tests/test_spatial_balance_core.py",
    "src/bhps/__init__.py", "src/bhps/adm_corner.py", "src/bhps/dynamical_capped_geometry.py",
    "src/bhps/dynamical_capped_horizon.py", "src/bhps/dynamical_capped_horizon_bvp.py",
    "src/bhps/dynamical_mots_stability.py", "src/bhps/gw_background.py",
    "src/bhps/gw_slice_high_order_solver.py", "src/bhps/initial_data.py", "src/bhps/scalar_pulse.py",
    "src/bhps/test14_quasilocal_charge.py", "src/bhps/test14b_balance_closure.py",
    "src/bhps/test14c_coupled_seam.py",
)


class AuthorityError(RuntimeError):
    pass


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path, immutable=False):
    path = Path(path)
    return bool(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1 and (not immutable or (path.stat().st_mode & 0o222) == 0))


def file_record(path, root=None):
    path = Path(path).absolute()
    base = Path(root).absolute() if root is not None else None
    return {"path": path.relative_to(base).as_posix() if base else str(path), "byte_count": path.stat().st_size, "sha256": sha256(path)}


def input_paths(root=ROOT):
    return tuple(sorted(path for path in (Path(root) / "sealed-inputs").rglob("*") if path.is_file()))


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise AuthorityError("authority namespace is not fresh")
    with temporary.open("xb") as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    os.replace(temporary, path)
    descriptor = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare_freeze():
    if AUTHORITY.exists() or (ROOT / "candidate-output").exists():
        raise AuthorityError("prospective namespace differs")
    sources = tuple(ROOT / name for name in SOURCES)
    inputs = input_paths()
    if not inputs or not all(regular(path) for path in (*sources, *inputs)):
        raise AuthorityError("source or input is missing or unsafe")
    p245 = json.loads((ROOT / "sealed-inputs/protocol245/protocol245_result.json").read_text())
    p246 = json.loads((ROOT / "sealed-inputs/protocol246/protocol246_result.json").read_text())
    p247 = json.loads((ROOT / "sealed-inputs/protocol247/candidate-output/protocol247_result.json").read_text())
    if p245.get("classification") != "FULL-DT-DENSE-NATIVE-OPERATOR-LOCAL-BALANCE-PASS":
        raise AuthorityError("Protocol 245 prerequisite differs")
    if p246.get("classification") != "FULL-HALF-NATIVE-BALANCE-CONSISTENCY-PASS":
        raise AuthorityError("Protocol 246 prerequisite differs")
    if not (
        p247.get("scientific", {}).get("classification") == "G9-G10-G11-BOUNDED-OUTER-TUBE-SPATIAL-TRANSFER-PASS"
        and p247.get("archive_only_balance_transfer_authorized") is True
    ):
        raise AuthorityError("Protocol 247 authorization differs")
    value = {
        "schema": SCHEMA,
        "status": "FROZEN",
        "grids": ["G9", "G10", "G11"],
        "matched_steps": list(range(44, 48)),
        "classification_driving_path": "centered",
        "stencil_widths": [5, 7, 9],
        "native_epsilon": 5e-7,
        "area_value_relative_limit": 0.01,
        "area_rate_relative_limit": 0.02,
        "seam_relative_limit": 0.01,
        "ledger_rate_relative_limit": 0.01,
        "ledger_term_balance_norm_relative_limit": 0.01,
        "parent_normalized_residual_limit": 0.01,
        "native_rate_relative_limit": 0.05,
        "native_rate_absolute_limit": 0.005,
        "required_runtime": {
            "python": "3.8.10", "numpy": "1.24.4", "scipy": "1.10.1",
            "system": "Linux", "machine": "aarch64",
        },
        "sources": {path.relative_to(ROOT).as_posix(): file_record(path, ROOT) for path in sources},
        "inputs": {path.relative_to(ROOT).as_posix(): file_record(path, ROOT) for path in inputs},
        "candidate_output_absent_at_freeze": True,
        "spacetime_evolution_authorized": False,
        "surface_solve_authorized": False,
        "parent_or_published_artifact_modification_authorized": False,
        "submitted_paper_edit_authorized": False,
        "continuum_dynamical_horizon_claim_authorized": False,
        "integrated_or_global_balance_claim_authorized": False,
        "event_horizon_claim_authorized": False,
        "connected_topology_claim_authorized": False,
        "source_ownership_claim_authorized": False,
    }
    value["fingerprint"] = hashlib.sha256(PREFIX + canonical(value)).hexdigest()
    atomic_json(AUTHORITY, value)
    for path in (*sources, *inputs):
        path.chmod(0o444)
    return value


def verify_freeze(root=ROOT):
    root = Path(root).absolute()
    path = root / "authority/freeze_record.json"
    if not regular(path, immutable=True):
        raise AuthorityError("freeze authority is missing or unsafe")
    value = json.loads(path.read_text())
    bare = dict(value); observed = bare.pop("fingerprint", None)
    if value.get("schema") != SCHEMA or value.get("status") != "FROZEN" or observed != hashlib.sha256(PREFIX + canonical(bare)).hexdigest():
        raise AuthorityError("freeze semantics differ")
    if set(value.get("sources", {})) != set(SOURCES):
        raise AuthorityError("source inventory differs")
    inputs = {path.relative_to(root).as_posix() for path in (root / "sealed-inputs").rglob("*") if path.is_file()}
    if set(value.get("inputs", {})) != inputs:
        raise AuthorityError("input inventory differs")
    for section in ("sources", "inputs"):
        for name, expected in value[section].items():
            local = root / name
            if not regular(local, immutable=True) or file_record(local, root) != expected:
                raise AuthorityError(f"frozen file differs: {name}")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.verify:
        parser.error("select exactly one mode")
    print(json.dumps(prepare_freeze() if args.prepare else verify_freeze(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
