#!/usr/bin/env python3
"""Freeze and verify Protocol 247."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AUTHORITY = ROOT / "authority/freeze_record.json"
SCHEMA = "protocol247-g9-g11-bounded-spatial-transfer-freeze-v1"
PREFIX = b"protocol247-freeze-v1\0"
SOURCES = (
    "PROTOCOL.md", "README.md", "authority.py", "bootstrap.py", "runner.py",
    "engine244.py", "transfer_core.py", "dense_tube_core.py", "horizon_core.py", "tube_core.py",
    "tests/test_transfer_core.py", "src/parent_protocol232.py", "src/parent_surface.py",
    "src/bhps/__init__.py", "src/bhps/adm_corner.py", "src/bhps/dynamical_capped_geometry.py",
    "src/bhps/dynamical_capped_horizon.py", "src/bhps/dynamical_capped_horizon_bvp.py",
    "src/bhps/dynamical_mots_stability.py", "src/bhps/gw_background.py",
    "src/bhps/gw_slice_high_order_solver.py", "src/bhps/initial_data.py", "src/bhps/scalar_pulse.py",
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


def record(path, root):
    path = Path(path).absolute()
    return {"path": path.relative_to(Path(root).absolute()).as_posix(), "byte_count": path.stat().st_size, "sha256": sha256(path)}


def file_record(path, root=None):
    """Compatibility name used by the inherited Protocol 244 source closure."""
    path = Path(path).absolute()
    if root is None:
        return {"path": str(path), "byte_count": path.stat().st_size, "sha256": sha256(path)}
    return record(path, root)


def input_paths(root=ROOT):
    return tuple(sorted(path for path in (Path(root) / "sealed-inputs").rglob("*") if path.is_file()))


def atomic_json(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise AuthorityError("authority namespace is not fresh")
    with temporary.open("xb") as stream:
        stream.write(canonical(value))
        stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
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
    p228 = json.loads((ROOT / "sealed-inputs/protocol228/protocol228_result.json").read_text())
    p244 = json.loads((ROOT / "sealed-inputs/protocol244/protocol244_result.json").read_text())
    p246 = json.loads((ROOT / "sealed-inputs/protocol246/protocol246_result.json").read_text())
    if p228.get("classification") != "REPAIRED-PARENT-FORMATION-CLOSURE-PASS":
        raise AuthorityError("Protocol 228 prerequisite differs")
    if p244.get("scientific", {}).get("classification") != "FULL-DT-DENSE-G10-OUTER-MARGINAL-TUBE-PASS":
        raise AuthorityError("Protocol 244 prerequisite differs")
    if p246.get("scientific", {}).get("classification") != "FULL-HALF-NATIVE-BALANCE-CONSISTENCY-PASS":
        raise AuthorityError("Protocol 246 prerequisite differs")
    value = {
        "schema": SCHEMA,
        "status": "FROZEN",
        "grids": ["G9", "G11"],
        "full_dt": 3.125e-5,
        "start_step": 32,
        "checkpoint_steps": list(range(43, 49)),
        "leaf_steps": list(range(43, 49)),
        "interior_tube_steps": list(range(44, 48)),
        "terminal_step": 48,
        "terminal_checkpoint_bitwise_replay_required": True,
        "terminal_profile_limits": {"rho_absolute": 1e-4, "slope_absolute": 1e-3, "endpoint_euclidean": 1e-4},
        "geometry_adjacent_relative_limit": 0.01,
        "stability_adjacent_relative_limit": 0.10,
        "stability_adjacent_absolute_limit": 0.02,
        "sources": {path.relative_to(ROOT).as_posix(): record(path, ROOT) for path in sources},
        "inputs": {path.relative_to(ROOT).as_posix(): record(path, ROOT) for path in inputs},
        "candidate_output_absent_at_freeze": True,
        "spacetime_evolution_authorized": True,
        "surface_solve_authorized": True,
        "new_parent_solve_authorized": False,
        "parent_or_published_artifact_modification_authorized": False,
        "submitted_paper_edit_authorized": False,
        "archive_only_balance_transfer_authorized_only_after_pass": True,
        "continuum_dynamical_horizon_claim_authorized": False,
        "event_horizon_claim_authorized": False,
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
            if not regular(local, immutable=True) or record(local, root) != expected:
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
