"""Small atomic, hash-validated recovery index for long numerical runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _temporary_path(destination: Path, suffix: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=suffix,
        dir=str(destination.parent),
    )
    os.close(descriptor)
    return Path(name)


def atomic_write_json(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    temporary = _temporary_path(destination, ".json.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_npz(path: str | Path, **arrays) -> None:
    destination = Path(path)
    temporary = _temporary_path(destination, ".npz.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def validate_npz(
    path: str | Path, required_shapes: dict[str, tuple[int, ...]] | None = None,
    require_finite: bool = True,
) -> dict:
    with np.load(path) as archive:
        keys = list(archive.files)
        if required_shapes:
            missing = sorted(set(required_shapes) - set(keys))
            if missing:
                raise ValueError(f"missing NPZ keys: {missing}")
            for key, shape in required_shapes.items():
                if tuple(archive[key].shape) != tuple(shape):
                    raise ValueError(
                        f"{key} has shape {archive[key].shape}, expected {shape}"
                    )
        if require_finite:
            # Finiteness is defined only for numeric arrays.  Recovery files
            # also carry immutable Unicode metadata (schema, fingerprints,
            # configuration JSON); applying ``np.isfinite`` to those arrays
            # raises instead of validating the archive.
            nonfinite = [
                key for key in keys
                if np.asarray(archive[key]).dtype.kind in "fc"
                and not np.all(np.isfinite(archive[key]))
            ]
            if nonfinite:
                raise ValueError(f"nonfinite NPZ arrays: {nonfinite}")
    return {"keys": keys, "byte_count": Path(path).stat().st_size}


class RecoveryIndex:
    """Atomic manifest whose completed stages are content-addressed files."""

    def __init__(
        self, path: str | Path, protocol_path: str | Path,
        expected_inputs: dict[str, str], maximum_stage_seconds: float = 3600.0,
    ):
        self.path = Path(path)
        self.protocol_path = str(protocol_path)
        self.protocol_sha256 = sha256_file(protocol_path)
        self.expected_inputs = dict(expected_inputs)
        self.maximum_stage_seconds = float(maximum_stage_seconds)
        for source, expected in self.expected_inputs.items():
            actual = sha256_file(source)
            if actual != expected:
                raise RuntimeError(
                    f"provenance hash mismatch for {source}: {actual} != {expected}"
                )
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
            if self.data.get("protocol_sha256") != self.protocol_sha256:
                raise RuntimeError("recovery manifest protocol hash mismatch")
            if self.data.get("expected_inputs") != self.expected_inputs:
                raise RuntimeError("recovery manifest input provenance mismatch")
            changed = False
            for stage in self.data.get("stages", {}).values():
                if stage.get("status") == "running":
                    stage["status"] = "pending"
                    stage["interrupted_at"] = self._timestamp()
                    changed = True
            if changed:
                self._save()
        else:
            self.data = {
                "schema": "bhps-recovery-index-v1",
                "protocol_path": self.protocol_path,
                "protocol_sha256": self.protocol_sha256,
                "expected_inputs": self.expected_inputs,
                "maximum_stage_seconds": self.maximum_stage_seconds,
                "created_at": self._timestamp(),
                "stages": {},
            }
            self._save()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _save(self) -> None:
        self.data["updated_at"] = self._timestamp()
        atomic_write_json(self.path, self.data)

    def register(
        self, stage_id: str, kind: str, expected_max_seconds: float,
        metadata: dict | None = None,
    ) -> dict:
        expected = float(expected_max_seconds)
        if expected <= 0.0 or expected > self.maximum_stage_seconds:
            raise ValueError(
                f"stage {stage_id} expected duration {expected}s violates "
                f"0 < duration <= {self.maximum_stage_seconds}s"
            )
        stages = self.data["stages"]
        if stage_id not in stages:
            stages[stage_id] = {
                "status": "pending", "kind": str(kind),
                "expected_max_seconds": expected,
                "metadata": dict(metadata or {}),
            }
            self._save()
        else:
            stage = stages[stage_id]
            if stage.get("kind") != str(kind):
                raise RuntimeError(f"stage-kind mismatch for {stage_id}")
            if stage.get("metadata", {}) != dict(metadata or {}):
                raise RuntimeError(f"stage-metadata mismatch for {stage_id}")
        return stages[stage_id]

    def mark_running(self, stage_id: str) -> None:
        stage = self.data["stages"][stage_id]
        stage["status"] = "running"
        stage["started_at"] = self._timestamp()
        self._save()

    def mark_failed(self, stage_id: str, message: str) -> None:
        stage = self.data["stages"][stage_id]
        stage["status"] = "failed"
        stage["failure"] = str(message)
        stage["failed_at"] = self._timestamp()
        self._save()

    def mark_complete(
        self, stage_id: str, output_path: str | Path, elapsed_seconds: float,
        metadata: dict | None = None,
    ) -> None:
        output = Path(output_path)
        if not output.is_file():
            raise FileNotFoundError(output)
        stage = self.data["stages"][stage_id]
        stage.update({
            "status": "complete",
            "output_path": str(output),
            "sha256": sha256_file(output),
            "byte_count": output.stat().st_size,
            "elapsed_seconds": float(elapsed_seconds),
            "completed_at": self._timestamp(),
        })
        if metadata is not None:
            stage["completion_metadata"] = dict(metadata)
        self._save()

    def validated_path(self, stage_id: str) -> Path | None:
        stage = self.data["stages"].get(stage_id)
        if not stage or stage.get("status") != "complete":
            return None
        path = Path(stage["output_path"])
        if (
            not path.is_file()
            or path.stat().st_size != stage.get("byte_count")
            or sha256_file(path) != stage.get("sha256")
        ):
            return None
        return path
