#!/usr/bin/env python3
"""Run the sealed Protocol-131 archive-only N0/N1 postmortem.

The runner has no parent-construction, parent-update, projection, acceleration,
RHS/RK, or evolution entry point.  It reads the two immutable Protocol-128
terminal checkpoints through :mod:`bhps.protocol131_postmortem`, writes one
atomic NPZ diagnostic artifact per parent, and then writes one ordered final
classification.  A first run requires a raw prospective freeze record; a
restart requires the validated authority snapshot created by that first run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bhps import protocol131_postmortem as postmortem  # noqa: E402
from bhps.protocol131_freeze_authority import (  # noqa: E402
    AUTHORIZATION_KIND,
    AUTHORIZATION_SCOPE,
    revalidate_protocol131_freeze_authority_snapshot,
    validate_protocol131_freeze_authority,
)
from bhps.protocol131_environment_contract import (  # noqa: E402
    validate_protocol131_environment_contract,
)
from bhps.protocol131_precision import extended_precision_residual  # noqa: E402
from bhps.protocol131_source_inventory import (  # noqa: E402
    validate_protocol131_input_manifest,
    validate_protocol131_source_manifest,
)
from bhps.recovery_indexer import (  # noqa: E402
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
    validate_npz,
)


DEFAULT_FREEZE_RECORD = ROOT / "protocol131_freeze_record.json"
AUTHORITY_SNAPSHOT_SUFFIX = "_freeze_authority_snapshot.json"
LEDGER_NAME = "stage_ledger.json"
LEDGER_SCHEMA = "Protocol-131-archive-only-stage-ledger-v1"
PARENT_STAGE_SCHEMA = "Protocol-131-parent-diagnostic-NPZ-v1"
FINAL_SCHEMA = "Protocol-131-final-classification-v1"
PARENT_LABELS = ("N0", "N1")
STAGE_ORDER = ("diagnostic/N0", "diagnostic/N1", "classification/final")
_RESERVED_ARRAY_KEY = "__protocol131_stage_metadata_json"
_ARRAY_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_ALLOWED_CLASSIFICATIONS = frozenset({
    "INVALID-AUDIT",
    "INVALID-JACOBIAN",
    "ARITHMETIC-LIMITED",
    "SOLVER-STAGNATION",
    "DISCRETE-COMPATIBILITY-OBSTRUCTION",
    "ILL-CONDITIONED",
    "NONLINEAR/GLOBALIZATION-UNRESOLVED",
    "INCONCLUSIVE-MIXED",
})


class Protocol131RunnerError(RuntimeError):
    """Raised when execution or recovery cannot remain within the freeze."""


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Protocol131RunnerError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise Protocol131RunnerError(f"nonfinite JSON constant: {value}")


def _load_json_file(path, label):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise Protocol131RunnerError(f"{label} is not a regular non-symlink file")
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except Protocol131RunnerError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Protocol131RunnerError(f"{label} is not valid UTF-8 JSON") from error


def _jsonable(value, label="value"):
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if type(key) is not str:
                raise Protocol131RunnerError(f"{label} has a non-string key")
            result[key] = _jsonable(item, f"{label}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_jsonable(item, f"{label}[]") for item in value]
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise Protocol131RunnerError(
                f"{label} contains an array; scientific arrays belong in NPZ"
            )
        return _jsonable(value.item(), label)
    if isinstance(value, np.generic):
        return _jsonable(value.item(), label)
    if value is None or type(value) in (str, bool, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Protocol131RunnerError(f"{label} contains a nonfinite float")
        return float(value)
    raise Protocol131RunnerError(
        f"{label} contains an unsupported {type(value).__name__}"
    )


def _canonical_bytes(value):
    return json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value):
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _authority_binding(authority):
    return {
        "authorization_kind": str(authority["authorization_kind"]),
        "authorization_scope": str(authority["authorization_scope"]),
        "protocol_sha256": str(authority["protocol_sha256"]),
        "runtime_contract_sha256": str(authority["runtime_contract_sha256"]),
        "source_manifest_sha256": str(authority["source_manifest_sha256"]),
        "input_manifest_sha256": str(authority["input_manifest_sha256"]),
        "candidate_output_directory": str(authority["candidate_output_directory"]),
    }


def _validate_authority_scope(authority):
    if not (
        authority.get("authorization_kind") == AUTHORIZATION_KIND
        and authority.get("authorization_scope") == AUTHORIZATION_SCOPE
        and authority.get("status") == "FROZEN"
        and authority.get("archive_only_postmortem_authorized") is True
        and authority.get("new_parent_construction_authorized") is False
        and authority.get("evolution_authorized") is False
    ):
        raise Protocol131RunnerError("freeze authority does not authorize this scope")


def _preflight_default_callbacks():
    if not callable(extended_precision_residual):
        raise Protocol131RunnerError(
            "the frozen extended-precision diagnostic is unavailable"
        )
    if not callable(getattr(postmortem, "classify_protocol131", None)):
        raise Protocol131RunnerError(
            "the frozen Protocol-131 classifier is unavailable"
        )


def _normalize_precision_result(result):
    if isinstance(result, tuple) and len(result) == 2:
        summary, arrays = result
    elif isinstance(result, Mapping) and set(result) == {"summary", "arrays"}:
        summary, arrays = result["summary"], result["arrays"]
    elif isinstance(result, Mapping):
        summary, arrays = result, {}
    else:
        raise Protocol131RunnerError("extended-precision diagnostic returned invalid data")
    if not isinstance(summary, Mapping) or not isinstance(arrays, Mapping):
        raise Protocol131RunnerError("extended-precision result has invalid containers")
    return dict(summary), dict(arrays)


def _base_arrays(residual, localization, cancellation):
    arrays = {
        "residual": np.asarray(residual),
        "binned_energy_16x16": np.asarray(localization["binned_energy_16x16"]),
        "wall_even": np.asarray(localization["wall_even"]),
        "wall_odd": np.asarray(localization["wall_odd"]),
        "raw": np.asarray(cancellation["raw"]),
        "reference_defect": np.asarray(cancellation["reference_defect"]),
        "balanced_before_wall_override": np.asarray(
            cancellation["balanced_before_wall_override"]
        ),
        "final_subtraction_roundoff_bound": np.asarray(
            cancellation["final_subtraction_roundoff_bound"]
        ),
        "final_row_roundoff_bound": np.asarray(
            cancellation["final_row_roundoff_bound"]
        ),
    }
    for wall_name, wall in cancellation["wall_terms"].items():
        for name, value in wall.items():
            arrays[f"wall_{wall_name}_{name}"] = np.asarray(value)
    return arrays


def _default_parent_analyzer(label):
    """Evaluate one immutable terminal state without accepting any update."""
    parent = postmortem.load_terminal_parent(label)
    residual, jacobian, replay = postmortem.replay_residual_and_jacobian(parent)
    localization = postmortem.residual_localization(parent, residual)
    cancellation = postmortem.residual_cancellation_terms(parent, residual)
    jacobian_audit = postmortem.audit_analytic_jacobian(
        parent, residual, jacobian,
    )
    linear = None
    merit = {"available": False, "samples": []}
    trust = {
        "not_reached": True,
        "blocked_by": "jacobian_audit",
        "rho_linear": 0.0,
        "direction_count": 0,
        "samples": [],
    }
    if jacobian_audit["passed"]:
        linear = postmortem.linear_range_analysis(parent, residual, jacobian)
        if linear.get("analysis_complete", False):
            trust = postmortem.certify_linear_trust_radius(
                parent, residual, jacobian, linear,
            )
            postmortem.annotate_dual_certificates(
                parent, jacobian, linear, trust,
            )
            merit = postmortem.frozen_newton_merit_curve(
                parent, residual, jacobian, linear,
            )
        else:
            trust = {
                "not_reached": True,
                "blocked_by": str(linear.get("failure_stage", "spectrum")),
                "rho_linear": 0.0,
                "direction_count": 0,
                "samples": [],
            }
    if linear is None:
        precision_summary = {
            "complete": False,
            "not_reached": True,
            "blocked_by": "jacobian_audit",
        }
        precision_arrays = {}
    else:
        precision_result = extended_precision_residual(
            parent, residual, localization, cancellation, linear,
        )
        precision_summary, precision_arrays = _normalize_precision_result(
            precision_result
        )

    if linear is None:
        summary = {
            "protocol_identifier": postmortem.PROTOCOL_IDENTIFIER,
            "parent_label": label,
            "parent_identity": str(parent["record"]["parent_identity"]),
            "generated_input_sha256": parent["generated_input_sha256"],
            "replay": replay,
            "localization": {
                "atoms": localization["atoms"],
                "blocks": localization["blocks"],
                "dominant_atom_by_Linf": localization["dominant_atom_by_Linf"],
            },
            "jacobian_audit": jacobian_audit,
            "linear": {"not_reached": True, "blocked_by": "jacobian_audit"},
            "merit_curve": merit,
        }
        arrays = _base_arrays(residual, localization, cancellation)
    else:
        summary = postmortem.compact_parent_summary(
            parent, replay, localization, jacobian_audit, linear, merit,
        )
        arrays = postmortem.parent_array_payload(
            residual, localization, cancellation, linear, parent=parent,
        )
    summary["trust_radius"] = trust
    summary["precision"] = precision_summary
    for name, value in precision_arrays.items():
        target = f"precision_{name}"
        if target in arrays:
            raise Protocol131RunnerError(f"duplicate precision array name: {target}")
        arrays[target] = np.asarray(value)
    summary["archive_only"] = True
    summary["parent_update_accepted"] = False
    summary["evolution_executed"] = False
    return {"summary": summary, "arrays": arrays}


def _normalize_parent_result(result, label):
    if not isinstance(result, Mapping) or set(result) != {"summary", "arrays"}:
        raise Protocol131RunnerError(
            f"{label} analyzer must return exactly summary and arrays"
        )
    summary = _jsonable(result["summary"], f"{label}.summary")
    arrays = result["arrays"]
    if not isinstance(arrays, Mapping) or not arrays:
        raise Protocol131RunnerError(f"{label} array payload must be nonempty")
    normalized = {}
    for name, value in arrays.items():
        if type(name) is not str or _ARRAY_NAME.fullmatch(name) is None:
            raise Protocol131RunnerError(f"{label} has an invalid array name")
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.kind not in "biufc":
            raise Protocol131RunnerError(f"{label}.{name} is not a numeric array")
        if not np.all(np.isfinite(array)):
            raise Protocol131RunnerError(f"{label}.{name} is nonfinite")
        normalized[name] = np.ascontiguousarray(array)
    if summary.get("parent_label", label) != label:
        raise Protocol131RunnerError(f"{label} summary identity differs")
    summary["parent_label"] = label
    return summary, normalized


def _array_sha256(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    for value in (array.dtype.str, json.dumps(list(array.shape))):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array_manifest(arrays):
    return {
        name: {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "sha256": _array_sha256(array),
        }
        for name, array in sorted(arrays.items())
    }


def _parent_path(candidate, label):
    return candidate / f"diagnostic_{label}.npz"


def _snapshot_path(candidate):
    candidate = Path(candidate)
    return candidate.with_name(candidate.name + AUTHORITY_SNAPSHOT_SUFFIX)


def _ledger_path(candidate):
    return candidate / LEDGER_NAME


def _new_ledger(authority, snapshot):
    created = _timestamp()
    return {
        "schema": LEDGER_SCHEMA,
        "protocol_identifier": postmortem.PROTOCOL_IDENTIFIER,
        "authority_binding": _authority_binding(authority),
        "authority_snapshot": {
            "path": str(snapshot),
            "sha256": sha256_file(snapshot),
            "byte_count": snapshot.stat().st_size,
        },
        "created_at": created,
        "updated_at": created,
        "stage_order": list(STAGE_ORDER),
        "stages": {
            stage_id: {
                "order": index,
                "kind": "parent-diagnostic" if index < 2 else "classification",
                "status": "pending",
            }
            for index, stage_id in enumerate(STAGE_ORDER)
        },
    }


def _save_ledger(path, ledger):
    ledger["updated_at"] = _timestamp()
    atomic_write_json(path, _jsonable(ledger, "ledger"))


def _validate_snapshot_file(candidate, expected_authority=None):
    path = _snapshot_path(candidate)
    snapshot = _load_json_file(path, "Protocol-131 authority snapshot")
    authority = revalidate_protocol131_freeze_authority_snapshot(snapshot)
    _validate_authority_scope(authority)
    if Path(authority["candidate_output_directory"]).resolve() != candidate.resolve():
        raise Protocol131RunnerError("authority snapshot candidate path differs")
    if expected_authority is not None and _canonical_bytes(authority) != _canonical_bytes(
        expected_authority
    ):
        raise Protocol131RunnerError("authority snapshot differs from issued authority")
    return authority


def _validate_ledger(candidate, authority):
    path = _ledger_path(candidate)
    ledger = _load_json_file(path, "Protocol-131 stage ledger")
    required = {
        "schema", "protocol_identifier", "authority_binding",
        "authority_snapshot", "created_at", "updated_at", "stage_order", "stages",
    }
    if set(ledger) != required:
        raise Protocol131RunnerError("stage ledger schema differs")
    if not (
        ledger["schema"] == LEDGER_SCHEMA
        and ledger["protocol_identifier"] == postmortem.PROTOCOL_IDENTIFIER
        and ledger["authority_binding"] == _authority_binding(authority)
        and ledger["stage_order"] == list(STAGE_ORDER)
        and set(ledger["stages"]) == set(STAGE_ORDER)
    ):
        raise Protocol131RunnerError("stage ledger authority or order differs")
    snapshot = _snapshot_path(candidate)
    expected_snapshot = {
        "path": str(snapshot),
        "sha256": sha256_file(snapshot),
        "byte_count": snapshot.stat().st_size,
    }
    if ledger["authority_snapshot"] != expected_snapshot:
        raise Protocol131RunnerError("stage ledger authority-snapshot binding differs")
    for index, stage_id in enumerate(STAGE_ORDER):
        stage = ledger["stages"][stage_id]
        if not isinstance(stage, Mapping) or not (
            stage.get("order") == index
            and stage.get("kind") == (
                "parent-diagnostic" if index < 2 else "classification"
            )
            and stage.get("status") in {
                "pending", "running", "failed", "complete",
            }
        ):
            raise Protocol131RunnerError(f"invalid ledger entry for {stage_id}")
    return ledger


def _load_or_initialize_ledger(candidate, authority):
    """Load a ledger or create one after a crash before ledger creation."""
    path = _ledger_path(candidate)
    if path.is_symlink():
        raise Protocol131RunnerError("Protocol-131 stage ledger may not be a symlink")
    if path.exists():
        return _validate_ledger(candidate, authority)
    ledger = _new_ledger(authority, _snapshot_path(candidate))
    _save_ledger(path, ledger)
    return _validate_ledger(candidate, authority)


def _parent_metadata(authority, label, summary, arrays):
    metadata = {
        "schema": PARENT_STAGE_SCHEMA,
        "protocol_identifier": postmortem.PROTOCOL_IDENTIFIER,
        "stage_id": f"diagnostic/{label}",
        "parent_label": label,
        "authority_binding": _authority_binding(authority),
        "scientific_summary": summary,
        "scientific_summary_sha256": _canonical_sha256(summary),
        "array_manifest": _array_manifest(arrays),
        "archive_only": True,
        "parent_update_accepted": False,
        "evolution_executed": False,
    }
    # This is both a strict-JSON check and the exact text embedded in the NPZ.
    _canonical_bytes(metadata)
    return metadata


def _reload_parent_artifact(candidate, authority, label, stage=None):
    path = _parent_path(candidate, label)
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise Protocol131RunnerError(f"{label} parent artifact is not a regular file")
    if stage is not None:
        artifact = stage.get("artifact")
        if not isinstance(artifact, Mapping) or set(artifact) != {
            "path", "sha256", "byte_count",
        }:
            raise Protocol131RunnerError(f"{label} ledger artifact record differs")
        if not (
            artifact["path"] == str(path)
            and artifact["byte_count"] == path.stat().st_size
            and artifact["sha256"] == sha256_file(path)
        ):
            raise Protocol131RunnerError(f"{label} artifact changed")
    validate_npz(path, require_finite=True)
    try:
        with np.load(path, allow_pickle=False) as archive:
            if _RESERVED_ARRAY_KEY not in archive.files:
                raise Protocol131RunnerError(f"{label} parent metadata is absent")
            metadata = json.loads(
                str(archive[_RESERVED_ARRAY_KEY]),
                object_pairs_hook=_reject_duplicate_json_keys,
            )
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in archive.files if name != _RESERVED_ARRAY_KEY
            }
    except Protocol131RunnerError:
        raise
    except (KeyError, ValueError, json.JSONDecodeError) as error:
        raise Protocol131RunnerError(f"{label} parent artifact is malformed") from error
    required = {
        "schema", "protocol_identifier", "stage_id", "parent_label",
        "authority_binding", "scientific_summary",
        "scientific_summary_sha256", "array_manifest", "archive_only",
        "parent_update_accepted", "evolution_executed",
    }
    summary = metadata.get("scientific_summary")
    if not (
        set(metadata) == required
        and metadata.get("schema") == PARENT_STAGE_SCHEMA
        and metadata.get("protocol_identifier") == postmortem.PROTOCOL_IDENTIFIER
        and metadata.get("stage_id") == f"diagnostic/{label}"
        and metadata.get("parent_label") == label
        and metadata.get("authority_binding") == _authority_binding(authority)
        and isinstance(summary, Mapping)
        and summary.get("parent_label") == label
        and metadata.get("scientific_summary_sha256") == _canonical_sha256(summary)
        and metadata.get("archive_only") is True
        and metadata.get("parent_update_accepted") is False
        and metadata.get("evolution_executed") is False
    ):
        raise Protocol131RunnerError(f"{label} parent metadata differs")
    if _array_manifest(arrays) != metadata.get("array_manifest"):
        raise Protocol131RunnerError(f"{label} parent array payload hash differs")
    for array in arrays.values():
        array.flags.writeable = False
    return metadata, arrays


def _write_parent_stage(candidate, authority, label, result):
    summary, arrays = _normalize_parent_result(result, label)
    path = _parent_path(candidate, label)
    if path.exists() or path.is_symlink():
        raise Protocol131RunnerError(f"immutable {label} output already exists")
    metadata = _parent_metadata(authority, label, summary, arrays)
    atomic_write_npz(
        path,
        **{
            _RESERVED_ARRAY_KEY: np.asarray(
                _canonical_bytes(metadata).decode("utf-8")
            ),
            **arrays,
        },
    )
    restored_metadata, restored_arrays = _reload_parent_artifact(
        candidate, authority, label,
    )
    if restored_metadata != metadata:
        raise Protocol131RunnerError(f"immediate {label} metadata reload differs")
    if any(
        _array_sha256(restored_arrays[name]) != _array_sha256(arrays[name])
        for name in arrays
    ):
        raise Protocol131RunnerError(f"immediate {label} NPZ reload differs")
    return restored_metadata, restored_arrays


def _record_parent_completion(
    ledger_path, ledger, label, *, adopted_after_interruption=False,
):
    stage_id = f"diagnostic/{label}"
    path = _parent_path(ledger_path.parent, label)
    stage = ledger["stages"][stage_id]
    stage.update({
        "status": "complete",
        "completed_at": _timestamp(),
        "adopted_after_interruption": bool(adopted_after_interruption),
        "artifact": {
            "path": str(path),
            "sha256": sha256_file(path),
            "byte_count": path.stat().st_size,
        },
    })
    _save_ledger(ledger_path, ledger)


def _materialize_parent(
    candidate, authority, ledger, label, analyzer,
):
    stage_id = f"diagnostic/{label}"
    ledger_path = _ledger_path(candidate)
    stage = ledger["stages"][stage_id]
    if stage["status"] == "complete":
        return _reload_parent_artifact(candidate, authority, label, stage)
    path = _parent_path(candidate, label)
    if path.exists() or path.is_symlink():
        # The atomic artifact is the durable commit.  A crash can happen after
        # its rename but before the ledger update; validate and adopt it.
        restored = _reload_parent_artifact(candidate, authority, label)
        _record_parent_completion(
            ledger_path, ledger, label, adopted_after_interruption=True,
        )
        return _reload_parent_artifact(
            candidate, authority, label, ledger["stages"][stage_id],
        )
    if stage["status"] in {"running", "failed"}:
        stage.clear()
        stage.update({
            "order": PARENT_LABELS.index(label),
            "kind": "parent-diagnostic",
            "status": "pending",
            "recovered_atomic_absence_at": _timestamp(),
        })
        _save_ledger(ledger_path, ledger)
    stage["status"] = "running"
    stage["started_at"] = _timestamp()
    _save_ledger(ledger_path, ledger)
    started = time.perf_counter()
    try:
        result = analyzer(label)
        metadata, arrays = _write_parent_stage(
            candidate, authority, label, result,
        )
        stage["elapsed_seconds"] = float(time.perf_counter() - started)
        _record_parent_completion(ledger_path, ledger, label)
        return _reload_parent_artifact(
            candidate, authority, label, ledger["stages"][stage_id],
        )
    except Exception as error:
        stage["status"] = "failed"
        stage["failed_at"] = _timestamp()
        stage["failure"] = f"{type(error).__name__}: {error}"
        _save_ledger(ledger_path, ledger)
        raise


def _default_finalizer(parent_summaries, parent_arrays):
    return postmortem.classify_protocol131(parent_summaries, parent_arrays)


def _normalize_classification(record):
    if not isinstance(record, Mapping):
        raise Protocol131RunnerError("classifier did not return a mapping")
    normalized = _jsonable(record, "classification")
    classification = normalized.get("classification")
    if classification not in _ALLOWED_CLASSIFICATIONS:
        raise Protocol131RunnerError(
            "classifier did not issue one frozen Protocol-131 classification"
        )
    if not (
        normalized.get("placeholder") is not True
        and normalized.get("complete") is True
        and normalized.get("provenance_valid") is True
    ):
        raise Protocol131RunnerError(
            "an incomplete, invalid-provenance, or placeholder record is not "
            "a scientific classification"
        )
    return normalized


def _reload_final(candidate, authority, stage=None):
    path = candidate / "classification_final.json"
    if path.is_symlink() or not path.is_file():
        raise Protocol131RunnerError("final classification is not a regular file")
    if stage is not None:
        artifact = stage.get("artifact")
        if not isinstance(artifact, Mapping) or set(artifact) != {
            "path", "sha256", "byte_count",
        }:
            raise Protocol131RunnerError("final ledger artifact record differs")
        if not (
            artifact["path"] == str(path)
            and artifact["byte_count"] == path.stat().st_size
            and artifact["sha256"] == sha256_file(path)
        ):
            raise Protocol131RunnerError("final classification artifact changed")
    payload = _load_json_file(path, "Protocol-131 final classification")
    required = {
        "schema", "protocol_identifier", "authority_binding", "classification",
        "classification_record", "parent_stage_sha256", "archive_only",
        "parent_construction_authorized", "parent_update_authorized",
        "phase_a_authorized", "evolution_authorized",
        "interface_physics_authorized",
    }
    classification_record = payload.get("classification_record")
    expected_parent_hashes = {
        label: sha256_file(_parent_path(candidate, label))
        for label in PARENT_LABELS
    }
    if not (
        set(payload) == required
        and payload.get("schema") == FINAL_SCHEMA
        and payload.get("protocol_identifier") == postmortem.PROTOCOL_IDENTIFIER
        and payload.get("authority_binding") == _authority_binding(authority)
        and payload.get("classification") in _ALLOWED_CLASSIFICATIONS
        and _normalize_classification(classification_record) == classification_record
        and payload.get("classification") == classification_record["classification"]
        and payload.get("parent_stage_sha256") == expected_parent_hashes
        and payload.get("archive_only") is True
        and payload.get("parent_construction_authorized") is False
        and payload.get("parent_update_authorized") is False
        and payload.get("phase_a_authorized") is False
        and payload.get("evolution_authorized") is False
        and payload.get("interface_physics_authorized") is False
    ):
        raise Protocol131RunnerError("final classification semantics differ")
    return payload


def _write_final_artifact(
    candidate, authority, parent_metadata, parent_arrays, finalizer,
):
    path = candidate / "classification_final.json"
    if path.exists() or path.is_symlink():
        raise Protocol131RunnerError("immutable final artifact already exists")
    summaries = {
        label: parent_metadata[label]["scientific_summary"]
        for label in PARENT_LABELS
    }
    classification_record = _normalize_classification(
        finalizer(summaries, parent_arrays)
    )
    payload = {
        "schema": FINAL_SCHEMA,
        "protocol_identifier": postmortem.PROTOCOL_IDENTIFIER,
        "authority_binding": _authority_binding(authority),
        "classification": classification_record["classification"],
        "classification_record": classification_record,
        "parent_stage_sha256": {
            label: sha256_file(_parent_path(candidate, label))
            for label in PARENT_LABELS
        },
        "archive_only": True,
        "parent_construction_authorized": False,
        "parent_update_authorized": False,
        "phase_a_authorized": False,
        "evolution_authorized": False,
        "interface_physics_authorized": False,
    }
    atomic_write_json(path, _jsonable(payload, "final payload"))
    restored = _reload_final(candidate, authority)
    if restored != payload:
        raise Protocol131RunnerError("immediate final-classification reload differs")
    return restored


def _record_final_completion(
    candidate, ledger, payload, *, adopted_after_interruption=False,
):
    path = candidate / "classification_final.json"
    stage = ledger["stages"]["classification/final"]
    stage.update({
        "status": "complete",
        "completed_at": _timestamp(),
        "classification": payload["classification"],
        "adopted_after_interruption": bool(adopted_after_interruption),
        "artifact": {
            "path": str(path),
            "sha256": sha256_file(path),
            "byte_count": path.stat().st_size,
        },
    })
    _save_ledger(_ledger_path(candidate), ledger)


def _materialize_final(
    candidate, authority, ledger, parent_metadata, parent_arrays, finalizer,
):
    stage_id = "classification/final"
    stage = ledger["stages"][stage_id]
    if stage["status"] == "complete":
        return _reload_final(candidate, authority, stage)
    path = candidate / "classification_final.json"
    if path.exists() or path.is_symlink():
        payload = _reload_final(candidate, authority)
        _record_final_completion(
            candidate, ledger, payload, adopted_after_interruption=True,
        )
        return _reload_final(candidate, authority, ledger["stages"][stage_id])
    if stage["status"] in {"running", "failed"}:
        stage.clear()
        stage.update({
            "order": 2,
            "kind": "classification",
            "status": "pending",
            "recovered_atomic_absence_at": _timestamp(),
        })
        _save_ledger(_ledger_path(candidate), ledger)
    stage["status"] = "running"
    stage["started_at"] = _timestamp()
    _save_ledger(_ledger_path(candidate), ledger)
    try:
        payload = _write_final_artifact(
            candidate, authority, parent_metadata, parent_arrays, finalizer,
        )
        _record_final_completion(candidate, ledger, payload)
        return _reload_final(candidate, authority, stage)
    except Exception as error:
        stage["status"] = "failed"
        stage["failed_at"] = _timestamp()
        stage["failure"] = f"{type(error).__name__}: {error}"
        _save_ledger(_ledger_path(candidate), ledger)
        raise


def run_protocol131_postmortem(
    *,
    freeze_record=None,
    freeze_authority=None,
    output_directory=None,
):
    """Execute or resume the fixed N0 -> N1 -> classification sequence.

    Production execution uses only the source-manifest-bound functions in
    :mod:`bhps.protocol131_postmortem`; arbitrary callback injection is not a
    public runner capability.
    """
    if (freeze_record is None) == (freeze_authority is None):
        raise Protocol131RunnerError(
            "supply exactly one of freeze_record for first run or "
            "freeze_authority for recovery"
        )
    recovery = freeze_authority is not None
    authority = (
        revalidate_protocol131_freeze_authority_snapshot(freeze_authority)
        if recovery else validate_protocol131_freeze_authority(freeze_record)
    )
    _validate_authority_scope(authority)
    validate_protocol131_environment_contract(authority["runtime_contract_path"])
    validate_protocol131_source_manifest(authority["source_manifest"])
    validate_protocol131_input_manifest(authority["input_manifest"])
    candidate = Path(authority["candidate_output_directory"])
    if output_directory is not None and Path(output_directory).resolve() != candidate.resolve():
        raise Protocol131RunnerError("requested output differs from frozen candidate")

    # Freeze validation is complete before callback preflight and before the
    # first candidate-directory byte is created.
    _preflight_default_callbacks()
    analyzer = _default_parent_analyzer
    classify = _default_finalizer

    if recovery:
        authority = _validate_snapshot_file(candidate, authority)
        if candidate.is_symlink() or (candidate.exists() and not candidate.is_dir()):
            raise Protocol131RunnerError("recovery candidate directory is invalid")
        candidate.mkdir(parents=True, exist_ok=True)
        ledger = _load_or_initialize_ledger(candidate, authority)
    else:
        snapshot_path = _snapshot_path(candidate)
        if snapshot_path.exists() or snapshot_path.is_symlink():
            raise Protocol131RunnerError(
                "external freeze-authority snapshot already exists; resume from it"
            )
        atomic_write_json(snapshot_path, _jsonable(authority, "authority"))
        authority = _validate_snapshot_file(candidate, authority)
        candidate.mkdir(parents=True, exist_ok=False)
        ledger = _load_or_initialize_ledger(candidate, authority)

    parent_metadata = {}
    parent_arrays = {}
    for label in PARENT_LABELS:
        # Re-hash the complete source/input freeze immediately before each
        # scientific producer.  A mid-run source change invalidates execution.
        authority = _validate_snapshot_file(candidate, authority)
        validate_protocol131_environment_contract(
            authority["runtime_contract_path"]
        )
        validate_protocol131_source_manifest(authority["source_manifest"])
        validate_protocol131_input_manifest(authority["input_manifest"])
        ledger = _validate_ledger(candidate, authority)
        metadata, arrays = _materialize_parent(
            candidate, authority, ledger, label, analyzer,
        )
        parent_metadata[label] = metadata
        parent_arrays[label] = arrays

    authority = _validate_snapshot_file(candidate, authority)
    validate_protocol131_environment_contract(authority["runtime_contract_path"])
    validate_protocol131_source_manifest(authority["source_manifest"])
    validate_protocol131_input_manifest(authority["input_manifest"])
    ledger = _validate_ledger(candidate, authority)
    return _materialize_final(
        candidate, authority, ledger, parent_metadata, parent_arrays, classify,
    )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--freeze-record", type=Path, default=None,
        help=f"raw first-run freeze record (default: {DEFAULT_FREEZE_RECORD.name})",
    )
    source.add_argument(
        "--resume-snapshot", type=Path,
        help="validated authority snapshot from an existing candidate directory",
    )
    parser.add_argument("--output-directory", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.resume_snapshot is not None:
        authority = _load_json_file(args.resume_snapshot, "resume authority snapshot")
        result = run_protocol131_postmortem(
            freeze_authority=authority,
            output_directory=args.output_directory,
        )
    else:
        record_path = args.freeze_record or DEFAULT_FREEZE_RECORD
        record = _load_json_file(record_path, "raw Protocol-131 freeze record")
        result = run_protocol131_postmortem(
            freeze_record=record,
            output_directory=args.output_directory,
        )
    print(result["classification"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
