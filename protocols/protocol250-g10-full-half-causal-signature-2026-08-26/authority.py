#!/usr/bin/env python3
"""Freeze and verify Protocol 250 source and immutable inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AUTHORITY = ROOT / "authority/freeze_record.json"
SCHEMA = "protocol250-g10-full-half-causal-signature-freeze-v1"
PREFIX = b"protocol250-freeze-v1\0"
SOURCES = (
    "PROTOCOL.md", "README.md", "authority.py", "bootstrap.py", "runner.py",
    "causal_signature_core.py", "tests/test_causal_signature_core.py",
    "src/bhps/__init__.py", "src/bhps/adm_corner.py",
    "src/bhps/dynamical_capped_geometry.py", "src/bhps/dynamical_capped_horizon.py",
    "src/bhps/dynamical_capped_horizon_bvp.py", "src/bhps/dynamical_mots_stability.py",
    "src/bhps/gw_background.py", "src/bhps/gw_slice_high_order_solver.py",
    "src/bhps/initial_data.py", "src/bhps/scalar_pulse.py",
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
    return bool(
        path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
        and (not immutable or (path.stat().st_mode & 0o222) == 0)
    )


def file_record(path, root=None):
    path = Path(path).absolute()
    base = Path(root).absolute() if root is not None else None
    return {
        "path": path.relative_to(base).as_posix() if base else str(path),
        "byte_count": path.stat().st_size,
        "sha256": sha256(path),
    }


def input_paths(root=ROOT):
    return tuple(sorted(path for path in (Path(root) / "sealed-inputs").rglob("*") if path.is_file()))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path.parent.mkdir(mode=0o755)
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


def prerequisite_semantics():
    p240 = read_json(ROOT / "sealed-inputs/protocol240/candidate-output/protocol240_result.json")
    p244 = read_json(ROOT / "sealed-inputs/protocol244/candidate-output/protocol244_result.json")
    p246 = read_json(ROOT / "sealed-inputs/protocol246/candidate-output/protocol246_result.json")
    p247 = read_json(ROOT / "sealed-inputs/protocol247/candidate-output/protocol247_result.json")
    p249 = read_json(ROOT / "sealed-inputs/protocol249/candidate-output/protocol249_result.json")
    if p240.get("scientific", {}).get("classification") != "DENSE-G10-OUTER-MARGINAL-TUBE-PASS":
        raise AuthorityError("Protocol 240-v3 prerequisite differs")
    if p244.get("scientific", {}).get("classification") != "FULL-DT-DENSE-G10-OUTER-MARGINAL-TUBE-PASS":
        raise AuthorityError("Protocol 244 prerequisite differs")
    if p246.get("classification") != "FULL-HALF-NATIVE-BALANCE-CONSISTENCY-PASS":
        raise AuthorityError("Protocol 246 prerequisite differs")
    if p247.get("scientific", {}).get("classification") != "G9-G10-G11-BOUNDED-OUTER-TUBE-SPATIAL-TRANSFER-PASS":
        raise AuthorityError("Protocol 247 prerequisite differs")
    if not (
        p249.get("classification") == "G9-G10-G11-FINITE-SEGMENT-INTEGRATED-BALANCE-PASS"
        and p249.get("finite_segment_integrated_balance_established") is True
    ):
        raise AuthorityError("Protocol 249 prerequisite differs")


def prepare_freeze():
    if AUTHORITY.exists() or (ROOT / "candidate-output").exists():
        raise AuthorityError("prospective namespace differs")
    sources = tuple(ROOT / name for name in SOURCES)
    inputs = input_paths()
    if not inputs or not all(regular(path) for path in (*sources, *inputs)):
        raise AuthorityError("source or input is missing or unsafe")
    prerequisite_semantics()
    value = {
        "schema": SCHEMA,
        "status": "FROZEN",
        "grid": "G10",
        "full_steps": list(range(39, 48)),
        "half_steps": list(range(78, 96, 2)),
        "full_leaf_steps": list(range(38, 49)),
        "half_leaf_steps": list(range(76, 98, 2)),
        "full_dt": 3.125e-5,
        "half_dt": 1.5625e-5,
        "physical_leaf_spacing": 3.125e-5,
        "projection": "g(V,V)-g(V,S)^2/g(S,S)",
        "signature_convention": "(-,+,+,+,+)",
        "causal_paths": ["backward", "centered", "forward"],
        "full_half_relative_limit": 0.01,
        "relative_floor": 1e-12,
        "orthogonal_contact_residual_limit": 2e-4,
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
        "event_horizon_claim_authorized": False,
        "connected_topology_claim_authorized": False,
        "global_intersector_charge_claim_authorized": False,
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
    value = read_json(path)
    bare = dict(value); observed = bare.pop("fingerprint", None)
    if (
        value.get("schema") != SCHEMA or value.get("status") != "FROZEN"
        or observed != hashlib.sha256(PREFIX + canonical(bare)).hexdigest()
    ):
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
    result = prepare_freeze() if args.prepare else verify_freeze()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
