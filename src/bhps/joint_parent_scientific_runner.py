"""Fail-closed scientific runner and recovery contract for Protocol 125.

This module owns operation ordering and durable checkpointing; it does not
invent any missing scientific evidence.  A run can start only after
``validate_protocol125_freeze_authority`` has verified a prospective freeze
whose source manifest contains both this runner and the concrete adapter
implementation.  The adapter must expose the complete parent -> prerequisite
-> acceleration -> two-parent data path and a restart codec for every stage.

There is deliberately no default scientific adapter.  The concrete explicit
adapter lives in ``joint_parent_production_adapter`` and supplies the complete
composition, transitive freeze inventory, and lossless NPZ reload path.  It
must still be supplied explicitly after its exact bytes have been included in
a prospective freeze.  Supplying no adapter therefore fails before an output
directory or an N0/N1 construction callback can be reached.

The runner never executes Phase A or an evolution call.  Its strongest output
is the ordered adjudicator's permission to prepare a separately frozen Phase
A.  Phase B, the RHS/RK matrix, and interface physics are always forbidden.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from bhps.joint_parent_freeze_authority import (
    AUTHORIZATION_KIND,
    AUTHORIZATION_SCOPE,
    revalidate_protocol125_freeze_authority_snapshot,
    validate_protocol125_freeze_authority,
)
from bhps.joint_parent_ordered_adjudicator import (
    POST_ACCELERATION_GROUPS,
    PRE_ACCELERATION_GROUPS,
    TWO_PARENT_GROUPS,
    adjudicate_protocol125_ordered,
)
from bhps.recovery_indexer import (
    RecoveryIndex,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
    validate_npz,
)


PROTOCOL_IDENTIFIER = "Protocol-125-scientific-runner-recovery-v1"
RUNNER_MANIFEST_LOGICAL_NAME = "protocol125_scientific_runner"
PARENT_LABELS = ("N0", "N1")
STAGE_SCHEMA = "Protocol-125-restartable-stage-NPZ-v1"
FINAL_SCHEMA = "Protocol-125-ordered-adjudication-JSON-v1"
REQUIRED_ADAPTER_CAPABILITIES = (
    "independent_parent_construction",
    "pre_acceleration_composition",
    "post_acceleration_composition",
    "two_parent_composition",
    "lossless_checkpoint_restore",
)
_STAGE_KINDS = ("parent", "pre-acceleration", "post-acceleration", "two-parent")
_CORE_GATE_KEYS = ("complete", "provenance_valid", "passed", "fingerprint")
_RESERVED_NPZ_PREFIX = "__protocol125_"

# The five implementation blockers recorded by the earlier readiness audit are
# closed by the explicit production adapter.  This empty tuple is kept as a
# stable audit surface; prospective freezing is a separate authorization gate.
PRODUCTION_ADAPTER_BLOCKERS = ()
PRODUCTION_FREEZE_BLOCKERS = (
    "freeze Protocol 125 only after its status is FROZEN and independent review passes",
    "include the adapter's complete transitive code, environment, and immutable-input inventory",
    "supply the explicit frozen adapter; no implicit/default candidate execution exists",
)
PRODUCTION_ADAPTER_COMPONENTS = MappingProxyType({
    "parent_construction": (
        "bhps.joint_parent_construction.construct_joint_parent_position"
    ),
    "position_state": (
        "bhps.joint_parent_position_state.build_joint_parent_position_state"
    ),
    "native_evidence": (
        "bhps.joint_parent_native_evidence."
        "build_protocol125_native_position_tangent_evidence"
    ),
    "pre_acceleration_composer": (
        "bhps.joint_parent_preacceleration."
        "evaluate_protocol125_preacceleration"
    ),
    "acceleration_fixed_point": (
        "bhps.joint_parent_acceleration."
        "solve_joint_parent_acceleration_fixed_point"
    ),
    "shared_representation": (
        "bhps.joint_parent_shared_representation."
        "build_protocol125_shared_representation"
    ),
    "append_only_lineage": (
        "bhps.joint_parent_lineage_adapter."
        "build_protocol125_append_only_position_lineage"
    ),
    "final_matrix": (
        "bhps.joint_parent_final_matrix."
        "evaluate_protocol125_final_representation_matrix"
    ),
    "post_acceleration_composer": (
        "bhps.joint_parent_postacceleration."
        "compose_protocol125_postacceleration_records"
    ),
    "two_parent_composer": (
        "bhps.joint_parent_two_parent.compose_protocol125_two_parent_records"
    ),
})


class Protocol125RunnerError(RuntimeError):
    """Base class for runner ordering, checkpoint, or adapter failures."""


class Protocol125AdapterBlocker(Protocol125RunnerError):
    """Raised before output when no complete frozen scientific adapter exists."""


class Protocol125RecoveryError(Protocol125RunnerError):
    """Raised when an immutable checkpoint cannot be reloaded exactly."""


def _valid_sha256(value):
    value = str(value)
    return bool(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _immutable_array(value):
    array = np.ascontiguousarray(np.asarray(value))
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({str(name): _freeze(item) for name, item in value.items()})
    if isinstance(value, np.ndarray):
        return _immutable_array(value)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _json_value(value, path="record"):
    """Return strict JSON data, rejecting NaN, arrays, and opaque objects."""
    if isinstance(value, Mapping):
        if any(type(name) is not str or not name for name in value):
            raise TypeError(f"{path} has a non-string or empty mapping key")
        return {
            name: _json_value(value[name], f"{path}/{name}")
            for name in sorted(value)
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item, f"{path}/{index}") for index, item in enumerate(value)]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a nonfinite float")
        return value
    raise TypeError(f"{path} contains unsupported JSON value {type(value).__name__}")


def _digest_piece(digest, tag, payload):
    tag = str(tag).encode("utf-8")
    payload = bytes(payload)
    digest.update(len(tag).to_bytes(8, "little"))
    digest.update(tag)
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def _tree_sha256(value, root="record"):
    """Hash nested string mappings, sequences, scalars, and non-object arrays."""
    digest = hashlib.sha256()

    def visit(item, path):
        if isinstance(item, Mapping):
            if any(type(name) is not str or not name for name in item):
                raise TypeError(f"{path} has an invalid mapping key")
            _digest_piece(digest, f"mapping:{path}", str(len(item)).encode("ascii"))
            for name in sorted(item):
                visit(item[name], f"{path}/{name}")
            return
        if isinstance(item, (tuple, list)):
            _digest_piece(digest, f"sequence:{path}", str(len(item)).encode("ascii"))
            for index, child in enumerate(item):
                visit(child, f"{path}/{index}")
            return
        if isinstance(item, np.generic):
            item = item.item()
        if item is None:
            _digest_piece(digest, f"none:{path}", b"")
            return
        if isinstance(item, bool):
            _digest_piece(digest, f"bool:{path}", b"1" if item else b"0")
            return
        if isinstance(item, int):
            _digest_piece(digest, f"int:{path}", str(item).encode("ascii"))
            return
        if isinstance(item, float):
            # Scientific failure evidence may legitimately contain +inf
            # condition estimates.  Hash the IEEE payload exactly; finiteness
            # is a scorer concern, not a checkpoint-integrity concern.
            _digest_piece(digest, f"float:{path}", struct.pack("!d", item))
            return
        if isinstance(item, str):
            _digest_piece(digest, f"str:{path}", item.encode("utf-8"))
            return
        array = np.ascontiguousarray(np.asarray(item))
        if array.dtype == object:
            raise TypeError(f"{path} has object dtype")
        _digest_piece(digest, f"array-shape:{path}", repr(array.shape).encode("ascii"))
        _digest_piece(digest, f"array-dtype:{path}", array.dtype.str.encode("ascii"))
        _digest_piece(digest, f"array-data:{path}", array.tobytes())

    visit(value, str(root))
    return digest.hexdigest()


@dataclass(frozen=True)
class Protocol125CheckpointPayload:
    """Complete restart state supplied by a prospectively frozen adapter."""

    arrays: Mapping
    metadata: Mapping


@dataclass(frozen=True)
class Protocol125RunnerStage:
    """One in-memory stage plus its lossless checkpoint representation."""

    runtime: object
    checkpoint: Protocol125CheckpointPayload
    bindings: Mapping
    gate_records: Mapping | None = None


@dataclass(frozen=True)
class Protocol125ScientificAdapters:
    """The exact callable surface that a frozen production adapter must expose."""

    identifier: str
    implementation_path: str
    implementation_sha256: str
    source_manifest_name: str
    capabilities: tuple[str, ...]
    construct_parent: Callable
    compose_pre_acceleration: Callable
    compose_post_acceleration: Callable
    compose_two_parent: Callable
    restore_checkpoint: Callable
    # Optional only for manufactured adapters.  A production adapter supplies
    # the complete transitive local-code and immutable-input inventory; every
    # declared entry is then required in the prospective freeze authority.
    source_manifest_inventory: Mapping | None = None
    runtime_environment_verifier: Callable | None = None


def production_adapter_blockers():
    """Return the immutable blocker inventory; this function performs no I/O."""
    return PRODUCTION_ADAPTER_BLOCKERS


def production_adapter_readiness():
    """Distinguish implemented adapter readiness from scientific authorization."""
    return MappingProxyType({
        "ready": True,
        "adapter_implementation_ready": True,
        "default_adapter_registered": False,
        "explicit_adapter_required": True,
        "component_callables": PRODUCTION_ADAPTER_COMPONENTS,
        "blockers": PRODUCTION_ADAPTER_BLOCKERS,
        "prospective_freeze_blockers": PRODUCTION_FREEZE_BLOCKERS,
        "prospectively_frozen": False,
        "scientific_run_ready": False,
        "scientific_candidate_execution_authorized": False,
    })


def _validated_adapter(adapters, authority):
    if adapters is None:
        raise Protocol125AdapterBlocker(
            "no Protocol-125 default production adapter is registered; supply the "
            "explicit adapter only after its complete inventory is prospectively frozen"
        )
    if not isinstance(adapters, Protocol125ScientificAdapters):
        raise TypeError("runner adapters require Protocol125ScientificAdapters")
    if type(adapters.identifier) is not str or not adapters.identifier:
        raise Protocol125AdapterBlocker("adapter identifier is missing")
    if tuple(adapters.capabilities) != REQUIRED_ADAPTER_CAPABILITIES:
        raise Protocol125AdapterBlocker("adapter capability inventory is incomplete or reordered")
    for name in (
        "construct_parent",
        "compose_pre_acceleration",
        "compose_post_acceleration",
        "compose_two_parent",
        "restore_checkpoint",
    ):
        if not callable(getattr(adapters, name)):
            raise Protocol125AdapterBlocker(f"adapter callback {name} is not callable")
    if not _valid_sha256(adapters.implementation_sha256):
        raise Protocol125AdapterBlocker("adapter implementation digest is invalid")
    source_manifest = authority.get("source_manifest")
    entry = source_manifest.get(adapters.source_manifest_name) if isinstance(
        source_manifest, Mapping
    ) else None
    if not isinstance(entry, Mapping):
        raise Protocol125AdapterBlocker("adapter implementation is absent from the freeze manifest")
    implementation = Path(adapters.implementation_path)
    try:
        resolved = implementation.resolve(strict=True)
    except OSError as error:
        raise Protocol125AdapterBlocker("adapter implementation file cannot be resolved") from error
    if (
        resolved.is_symlink()
        or not resolved.is_file()
        or str(resolved) != str(entry.get("path", ""))
        or adapters.implementation_sha256 != str(entry.get("sha256", ""))
        or sha256_file(resolved) != adapters.implementation_sha256
    ):
        raise Protocol125AdapterBlocker("adapter implementation differs from the frozen manifest")
    runner_entry = source_manifest.get(RUNNER_MANIFEST_LOGICAL_NAME) if isinstance(
        source_manifest, Mapping
    ) else None
    runner_path = Path(__file__).resolve()
    if not isinstance(runner_entry, Mapping) or (
        str(runner_entry.get("path", "")) != str(runner_path)
        or str(runner_entry.get("sha256", "")) != sha256_file(runner_path)
    ):
        raise Protocol125AdapterBlocker("scientific runner differs from or is absent from the freeze manifest")
    inventory = adapters.source_manifest_inventory
    if inventory is not None:
        if not isinstance(inventory, Mapping) or not inventory:
            raise Protocol125AdapterBlocker(
                "adapter transitive source-manifest inventory is empty"
            )
        if any(type(name) is not str or not name for name in inventory):
            raise Protocol125AdapterBlocker(
                "adapter transitive source-manifest names are invalid"
            )
        if (
            str(inventory.get(adapters.source_manifest_name, "")) != str(resolved)
            or str(inventory.get(RUNNER_MANIFEST_LOGICAL_NAME, ""))
            != str(runner_path)
        ):
            raise Protocol125AdapterBlocker(
                "adapter transitive inventory does not bind adapter and runner"
            )
        occupied = set()
        for logical_name, raw_path in inventory.items():
            try:
                unresolved_path = Path(raw_path)
                if unresolved_path.is_symlink():
                    raise Protocol125AdapterBlocker(
                        f"transitive freeze input {logical_name} may not be a symlink"
                    )
                expected_path = unresolved_path.resolve(strict=True)
            except Protocol125AdapterBlocker:
                raise
            except (OSError, TypeError) as error:
                raise Protocol125AdapterBlocker(
                    f"transitive freeze input {logical_name} cannot be resolved"
                ) from error
            if not expected_path.is_file():
                raise Protocol125AdapterBlocker(
                    f"transitive freeze input {logical_name} is not a regular file"
                )
            if expected_path in occupied:
                raise Protocol125AdapterBlocker(
                    "adapter transitive source-manifest inventory reuses a path"
                )
            occupied.add(expected_path)
            manifest_entry = source_manifest.get(logical_name)
            if not isinstance(manifest_entry, Mapping):
                # The adjudicator is already a separately hashed freeze input
                # and the freeze schema correctly forbids duplicating it in
                # source_manifest.
                if str(expected_path) == str(authority.get("adjudicator_path", "")):
                    recorded_path = str(authority["adjudicator_path"])
                    recorded_sha256 = str(authority["adjudicator_sha256"])
                else:
                    raise Protocol125AdapterBlocker(
                        f"transitive freeze input {logical_name} is absent from the manifest"
                    )
            else:
                recorded_path = str(manifest_entry.get("path", ""))
                recorded_sha256 = str(manifest_entry.get("sha256", ""))
            if (
                recorded_path != str(expected_path)
                or recorded_sha256 != sha256_file(expected_path)
            ):
                raise Protocol125AdapterBlocker(
                    f"transitive freeze input {logical_name} differs from frozen bytes"
                )
        environment_path = inventory.get("environment:runtime-contract")
        if environment_path is not None:
            verifier = adapters.runtime_environment_verifier
            if not callable(verifier):
                raise Protocol125AdapterBlocker(
                    "adapter runtime environment contract lacks a verifier"
                )
            try:
                environment_record = verifier(Path(environment_path))
            except Exception as error:
                raise Protocol125AdapterBlocker(
                    "active runtime differs from the frozen environment contract"
                ) from error
            if (
                not isinstance(environment_record, Mapping)
                or not _valid_sha256(environment_record.get("fingerprint", ""))
            ):
                raise Protocol125AdapterBlocker(
                    "runtime environment verifier returned an invalid record"
                )
    return adapters


def _validate_bindings(bindings, expected, label):
    if not isinstance(bindings, Mapping) or tuple(bindings) != tuple(expected):
        raise Protocol125RunnerError(f"{label} bindings differ from the required parent order")
    for parent, identity in bindings.items():
        if parent not in PARENT_LABELS or not _valid_sha256(identity):
            raise Protocol125RunnerError(f"{label} has an invalid parent binding")
    if len(bindings) == 2 and str(bindings["N0"]) == str(bindings["N1"]):
        raise Protocol125RunnerError("N0 and N1 identities are not independent")


def _validate_gate_records(records, groups, bindings, label):
    if not isinstance(records, Mapping) or tuple(records) != tuple(groups):
        raise Protocol125RunnerError(f"{label} gate record set is incomplete or reordered")
    single_parent = len(bindings) == 1
    parent_label = next(iter(bindings)) if single_parent else None
    for group in groups:
        record = records[group]
        if not isinstance(record, Mapping):
            raise Protocol125RunnerError(f"{label}:{group} gate is not a mapping")
        if any(name not in record for name in _CORE_GATE_KEYS):
            raise Protocol125RunnerError(f"{label}:{group} gate lacks its fail-closed core")
        if any(type(record[name]) is not bool for name in _CORE_GATE_KEYS[:3]):
            raise Protocol125RunnerError(f"{label}:{group} gate core is not boolean")
        if not _valid_sha256(record["fingerprint"]):
            raise Protocol125RunnerError(f"{label}:{group} fingerprint is invalid")
        not_reached = record.get("not_reached", False)
        if type(not_reached) is not bool:
            raise Protocol125RunnerError(
                f"{label}:{group} not_reached flag is not boolean"
            )
        if not_reached:
            # A downstream stop is itself reached audit evidence.  It may
            # legitimately have passed=False, but its provenance and stop
            # declaration must be complete and valid.
            if (
                record["complete"] is not True
                or record["provenance_valid"] is not True
                or record["passed"] is not False
                or not isinstance(record.get("blocked_by"), str)
                or not record["blocked_by"]
            ):
                raise Protocol125RunnerError(
                    f"{label}:{group} not-reached gate lacks complete valid stop evidence"
                )
        else:
            # INVALID audit evidence is a technical stop, not a recoverable
            # scientific classification.  Enforce this before checkpointing
            # on fresh execution and again after adapter reconstruction.
            if record["complete"] is not True:
                raise Protocol125RunnerError(
                    f"{label}:{group} reached gate is incomplete"
                )
            if record["provenance_valid"] is not True:
                raise Protocol125RunnerError(
                    f"{label}:{group} reached gate has invalid provenance"
                )
        if single_parent:
            if (
                str(record.get("parent_label", "")) != parent_label
                or str(record.get("parent_identity", "")) != str(bindings[parent_label])
            ):
                raise Protocol125RunnerError(f"{label}:{group} parent binding differs")
        else:
            supplied = record.get("parent_identities")
            if not isinstance(supplied, Mapping) or any(
                str(supplied.get(parent, "")) != str(bindings[parent])
                for parent in PARENT_LABELS
            ):
                raise Protocol125RunnerError(f"{label}:{group} two-parent binding differs")


def _checkpoint_arrays(payload, *, expected_kind, record_sha256):
    if not isinstance(payload, Protocol125CheckpointPayload):
        raise TypeError("stage checkpoint requires Protocol125CheckpointPayload")
    if not isinstance(payload.arrays, Mapping) or not payload.arrays:
        raise Protocol125RunnerError("stage checkpoint array record is empty")
    arrays = {}
    for name, value in payload.arrays.items():
        if type(name) is not str or not name or name.startswith(_RESERVED_NPZ_PREFIX):
            raise Protocol125RunnerError("stage checkpoint contains an invalid array name")
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype == object or array.dtype.kind not in "biufcSU":
            raise Protocol125RunnerError(f"checkpoint array {name} has an unsupported dtype")
        if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
            raise Protocol125RunnerError(f"checkpoint array {name} is nonfinite")
        arrays[name] = _immutable_array(array)
    metadata = _json_value(payload.metadata, "checkpoint-metadata")
    required_metadata = {
        "stage_kind": expected_kind,
        "complete_state": True,
        "restartable_without_unrecorded_state": True,
        "record_sha256": record_sha256,
    }
    for name, expected in required_metadata.items():
        if metadata.get(name) != expected:
            raise Protocol125RunnerError(f"checkpoint metadata {name} is not {expected!r}")
    return arrays, metadata


def _validate_stage(stage, *, stage_kind, label, expected_bindings):
    if not isinstance(stage, Protocol125RunnerStage):
        raise TypeError(f"{label} callback did not return Protocol125RunnerStage")
    _validate_bindings(stage.bindings, expected_bindings, label)
    if stage_kind == "parent":
        if stage.gate_records is not None:
            raise Protocol125RunnerError(f"{label} parent stage may not claim gate records")
        record_sha256 = protocol125_stage_record_sha256(
            stage_kind, stage.bindings, None,
        )
    else:
        groups = {
            "pre-acceleration": PRE_ACCELERATION_GROUPS,
            "post-acceleration": POST_ACCELERATION_GROUPS,
            "two-parent": TWO_PARENT_GROUPS,
        }[stage_kind]
        _validate_gate_records(stage.gate_records, groups, stage.bindings, label)
        record_sha256 = protocol125_stage_record_sha256(
            stage_kind, stage.bindings, stage.gate_records,
        )
    arrays, metadata = _checkpoint_arrays(
        stage.checkpoint,
        expected_kind=stage_kind,
        record_sha256=record_sha256,
    )
    return record_sha256, arrays, metadata


def protocol125_stage_record_sha256(stage_kind, bindings, gate_records):
    """Return the canonical record digest required in adapter checkpoints."""
    stage_kind = str(stage_kind)
    if stage_kind == "parent":
        if gate_records is not None:
            raise ValueError("parent checkpoint digest cannot include gate records")
        return _tree_sha256(
            {"bindings": bindings, "stage_kind": stage_kind},
            root="parent-stage-record",
        )
    if stage_kind not in _STAGE_KINDS[1:]:
        raise ValueError("unknown Protocol-125 checkpoint stage kind")
    if gate_records is None:
        raise ValueError("gate checkpoint digest requires gate records")
    return _tree_sha256(gate_records, root="gate-records")


def _checkpoint_digest(arrays, metadata, envelope):
    return _tree_sha256(
        {"arrays": arrays, "metadata": metadata, "envelope": envelope},
        root="checkpoint-payload",
    )


def _stage_filename(stage_id):
    safe = stage_id.replace("/", "_")
    if not safe or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in safe):
        raise ValueError("stage identifier is not path safe")
    return f"{safe}.npz"


def _write_stage_checkpoint(
    index,
    output_directory,
    stage_id,
    stage_kind,
    stage,
    *,
    authority,
    adapters,
    expected_bindings,
    elapsed_seconds,
):
    record_sha256, arrays, metadata = _validate_stage(
        stage,
        stage_kind=stage_kind,
        label=stage_id,
        expected_bindings=expected_bindings,
    )
    destination = Path(output_directory)/_stage_filename(stage_id)
    if destination.exists():
        raise Protocol125RecoveryError(
            f"immutable checkpoint already exists before completion: {destination}"
        )
    envelope = {
        "schema": STAGE_SCHEMA,
        "stage_id": stage_id,
        "stage_kind": stage_kind,
        "adapter_identifier": adapters.identifier,
        "adapter_implementation_sha256": adapters.implementation_sha256,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": str(authority["protocol_sha256"]),
        "adjudicator_sha256": str(authority["adjudicator_sha256"]),
        "record_sha256": record_sha256,
        "bindings": _json_value(stage.bindings, "bindings"),
    }
    payload_sha256 = _checkpoint_digest(arrays, metadata, envelope)
    embedded = {
        f"{_RESERVED_NPZ_PREFIX}schema": np.asarray(STAGE_SCHEMA),
        f"{_RESERVED_NPZ_PREFIX}stage_id": np.asarray(stage_id),
        f"{_RESERVED_NPZ_PREFIX}stage_kind": np.asarray(stage_kind),
        f"{_RESERVED_NPZ_PREFIX}envelope_json": np.asarray(
            json.dumps(envelope, sort_keys=True, separators=(",", ":"))
        ),
        f"{_RESERVED_NPZ_PREFIX}metadata_json": np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        ),
        f"{_RESERVED_NPZ_PREFIX}payload_sha256": np.asarray(payload_sha256),
    }
    atomic_write_npz(destination, **embedded, **arrays)
    _reload_stage_payload(
        destination,
        expected_stage_id=stage_id,
        expected_kind=stage_kind,
        expected_envelope=envelope,
        expected_arrays=arrays,
        expected_metadata=metadata,
    )
    index.mark_complete(
        stage_id,
        destination,
        elapsed_seconds,
        {
            "stage_kind": stage_kind,
            "record_sha256": record_sha256,
            "payload_sha256": payload_sha256,
            "adapter_identifier": adapters.identifier,
        },
    )
    validated = index.validated_path(stage_id)
    if validated != destination:
        raise Protocol125RecoveryError(f"recovery index did not validate {stage_id}")
    return destination


def _reload_stage_payload(
    path,
    *,
    expected_stage_id,
    expected_kind,
    expected_envelope=None,
    expected_arrays=None,
    expected_metadata=None,
):
    path = Path(path)
    validate_npz(path, require_finite=True)
    try:
        with np.load(path, allow_pickle=False) as archive:
            schema = str(archive[f"{_RESERVED_NPZ_PREFIX}schema"])
            stage_id = str(archive[f"{_RESERVED_NPZ_PREFIX}stage_id"])
            stage_kind = str(archive[f"{_RESERVED_NPZ_PREFIX}stage_kind"])
            envelope = json.loads(str(archive[f"{_RESERVED_NPZ_PREFIX}envelope_json"]))
            metadata = json.loads(str(archive[f"{_RESERVED_NPZ_PREFIX}metadata_json"]))
            recorded_digest = str(archive[f"{_RESERVED_NPZ_PREFIX}payload_sha256"])
            arrays = {
                name: _immutable_array(archive[name])
                for name in archive.files if not name.startswith(_RESERVED_NPZ_PREFIX)
            }
    except (KeyError, ValueError, json.JSONDecodeError) as error:
        raise Protocol125RecoveryError(f"checkpoint {path} is incomplete") from error
    if (
        schema != STAGE_SCHEMA
        or stage_id != expected_stage_id
        or stage_kind != expected_kind
        or envelope.get("schema") != STAGE_SCHEMA
        or envelope.get("stage_id") != expected_stage_id
        or envelope.get("stage_kind") != expected_kind
    ):
        raise Protocol125RecoveryError(f"checkpoint identity mismatch for {expected_stage_id}")
    found_digest = _checkpoint_digest(arrays, metadata, envelope)
    if not _valid_sha256(recorded_digest) or recorded_digest != found_digest:
        raise Protocol125RecoveryError(f"checkpoint payload hash mismatch for {expected_stage_id}")
    if expected_envelope is not None and envelope != expected_envelope:
        raise Protocol125RecoveryError(f"checkpoint envelope changed for {expected_stage_id}")
    if expected_metadata is not None and metadata != expected_metadata:
        raise Protocol125RecoveryError(f"checkpoint metadata changed for {expected_stage_id}")
    if expected_arrays is not None:
        if set(arrays) != set(expected_arrays) or any(
            np.ascontiguousarray(arrays[name]).tobytes()
            != np.ascontiguousarray(expected_arrays[name]).tobytes()
            or arrays[name].shape != np.asarray(expected_arrays[name]).shape
            or arrays[name].dtype != np.asarray(expected_arrays[name]).dtype
            for name in arrays
        ):
            raise Protocol125RecoveryError(f"checkpoint arrays changed for {expected_stage_id}")
    return MappingProxyType({
        "path": path,
        "envelope": _freeze(envelope),
        "metadata": _freeze(metadata),
        "arrays": MappingProxyType(arrays),
        "payload_sha256": recorded_digest,
        "byte_count": path.stat().st_size,
        "sha256": sha256_file(path),
    })


def reload_protocol125_recovery_checkpoint(index, stage_id):
    """Hash-validate and load one complete stage from a RecoveryIndex.

    A manifest entry that says ``complete`` but whose file was changed is an
    audit failure, never a request to recompute or overwrite the artifact.
    """
    stage = index.data.get("stages", {}).get(stage_id)
    if not isinstance(stage, Mapping) or stage.get("status") != "complete":
        raise Protocol125RecoveryError(f"stage {stage_id} is not complete")
    path = index.validated_path(stage_id)
    if path is None:
        raise Protocol125RecoveryError(f"stage {stage_id} failed recovery hash/byte validation")
    metadata = stage.get("completion_metadata", {})
    return _reload_stage_payload(
        path,
        expected_stage_id=stage_id,
        expected_kind=str(metadata.get("stage_kind", "")),
    )


def _stage_all_pass(stage, groups):
    # Full structural validation has already run.  Incomplete or invalid
    # evidence never unlocks a downstream producer.
    return bool(all(
        stage.gate_records[group]["complete"]
        and stage.gate_records[group]["provenance_valid"]
        and stage.gate_records[group]["passed"]
        for group in groups
    ))


def _ordered_parent_records(stages):
    return {
        parent: {
            group: stages[parent].gate_records[group]
            for group in stages[parent].gate_records
        }
        for parent in PARENT_LABELS
    }


def _authority_inputs(authority):
    entries = [
        (authority["adjudicator_path"], authority["adjudicator_sha256"]),
        (
            authority["independent_review"]["path"],
            authority["independent_review"]["sha256"],
        ),
    ]
    entries.extend(
        (entry["path"], entry["sha256"])
        for entry in authority["source_manifest"].values()
    )
    result = {}
    for path, digest in entries:
        if path in result and result[path] != digest:
            raise Protocol125RunnerError("freeze authority records conflicting file hashes")
        result[str(path)] = str(digest)
    return result


def _materialize_stage(
    index,
    output_directory,
    *,
    stage_id,
    stage_kind,
    expected_bindings,
    authority,
    adapters,
    producer,
    restore_context,
    expected_max_seconds,
):
    metadata = {
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "stage_kind": stage_kind,
        "adapter_identifier": adapters.identifier,
        "adapter_implementation_sha256": adapters.implementation_sha256,
        "binding_labels": list(expected_bindings),
    }
    index.register(stage_id, stage_kind, expected_max_seconds, metadata)
    registered = index.data["stages"][stage_id]
    if registered.get("status") == "complete":
        archived = reload_protocol125_recovery_checkpoint(index, stage_id)
        restored = adapters.restore_checkpoint(
            stage_id,
            archived,
            context=MappingProxyType(dict(restore_context)),
        )
        _validate_stage(
            restored,
            stage_kind=stage_kind,
            label=f"restored:{stage_id}",
            expected_bindings=expected_bindings,
        )
        found = _tree_sha256(
            restored.checkpoint.arrays, root=f"restored:{stage_id}:arrays"
        )
        expected = _tree_sha256(
            archived["arrays"], root=f"restored:{stage_id}:arrays"
        )
        if found != expected:
            raise Protocol125RecoveryError(f"restored checkpoint arrays differ for {stage_id}")
        return restored
    destination = Path(output_directory)/_stage_filename(stage_id)
    if destination.exists():
        raise Protocol125RecoveryError(
            f"unindexed or interrupted immutable artifact exists for {stage_id}"
        )
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        stage = producer()
        path = _write_stage_checkpoint(
            index,
            output_directory,
            stage_id,
            stage_kind,
            stage,
            authority=authority,
            adapters=adapters,
            expected_bindings=expected_bindings,
            elapsed_seconds=time.perf_counter()-started,
        )
        archived = reload_protocol125_recovery_checkpoint(index, stage_id)
        restored = adapters.restore_checkpoint(
            stage_id,
            archived,
            context=MappingProxyType(dict(restore_context)),
        )
        restored_record_sha, restored_arrays, restored_metadata = _validate_stage(
            restored,
            stage_kind=stage_kind,
            label=f"restored:{stage_id}",
            expected_bindings=expected_bindings,
        )
        if (
            restored_record_sha != archived["envelope"]["record_sha256"]
            or _tree_sha256(restored_arrays, root="checkpoint-arrays")
            != _tree_sha256(archived["arrays"], root="checkpoint-arrays")
            # Reloaded archive data are recursively frozen for safe recovery,
            # so JSON arrays become tuples.  Compare strict JSON content here
            # rather than the mutable/immutable container representation.
            or _json_value(restored_metadata, "restored-checkpoint-metadata")
            != _json_value(archived["metadata"], "archived-checkpoint-metadata")
        ):
            raise Protocol125RecoveryError(f"lossless stage restore failed for {stage_id}")
        if path != archived["path"]:
            raise AssertionError("new checkpoint path changed during immediate reload")
        return restored
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def _write_final_adjudication(index, output_directory, adjudication, authority, adapters):
    stage_id = "adjudication/final"
    metadata = {
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "schema": FINAL_SCHEMA,
        "adapter_identifier": adapters.identifier,
    }
    index.register(stage_id, "ordered-adjudication", 60.0, metadata)
    stage = index.data["stages"][stage_id]
    if stage.get("status") == "complete":
        path = index.validated_path(stage_id)
        if path is None:
            raise Protocol125RecoveryError("final adjudication failed recovery hash validation")
        payload = json.loads(path.read_text())
        if payload.get("schema") != FINAL_SCHEMA:
            raise Protocol125RecoveryError("final adjudication schema differs")
        return _freeze(payload)
    path = Path(output_directory)/"adjudication_final.json"
    if path.exists():
        raise Protocol125RecoveryError("unindexed final adjudication artifact already exists")
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        ordered = _json_value(adjudication, "ordered-adjudication")
        if bool(ordered.get("rhs_rk_phase_b_full_matrix_authorized")):
            raise Protocol125RunnerError("ordered adjudication attempted to authorize Phase B")
        if bool(ordered.get("interface_physics_authorized")):
            raise Protocol125RunnerError("ordered adjudication attempted to authorize interface physics")
        payload_without_hash = {
            "schema": FINAL_SCHEMA,
            "protocol_identifier": PROTOCOL_IDENTIFIER,
            "protocol_sha256": str(authority["protocol_sha256"]),
            "adjudicator_sha256": str(authority["adjudicator_sha256"]),
            "adapter_identifier": adapters.identifier,
            "adapter_implementation_sha256": adapters.implementation_sha256,
            "ordered_adjudication": ordered,
            "phase_a_authorized": bool(ordered.get("phase_a_authorized", False)),
            "phase_a_executed": False,
            "rhs_rk_phase_b_full_matrix_authorized": False,
            "rhs_rk_phase_b_full_matrix_executed": False,
            "interface_physics_authorized": False,
            "candidate_science_beyond_parent_qualification_executed": False,
        }
        payload = {
            **payload_without_hash,
            "fingerprint": _tree_sha256(payload_without_hash, root="runner-final"),
        }
        atomic_write_json(path, payload)
        reloaded = json.loads(path.read_text())
        if reloaded != payload:
            raise Protocol125RecoveryError("final adjudication JSON did not reload exactly")
        index.mark_complete(
            stage_id,
            path,
            time.perf_counter()-started,
            {
                "classification": ordered.get("classification", "INVALID-audit"),
                "phase_a_authorized": bool(ordered.get("phase_a_authorized", False)),
                "fingerprint": payload["fingerprint"],
            },
        )
        if index.validated_path(stage_id) != path:
            raise Protocol125RecoveryError("final adjudication index validation failed")
        return _freeze(payload)
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def run_protocol125_scientific(
    *,
    freeze_record=None,
    freeze_authority=None,
    adapters=None,
    output_directory=None,
):
    """Run only the frozen Protocol-125 parent qualification sequence.

    Freeze validation is the first operation.  In particular, neither adapter
    validation, parent construction, nor directory creation precedes it.
    Every producer is followed by an atomic checkpoint, recovery-index hash
    binding, immediate allow-pickle-free reload, and adapter reconstruction.
    The reconstructed stage—not the uncheckpointed object—is consumed next.
    """
    if (freeze_record is None) == (freeze_authority is None):
        raise Protocol125RunnerError(
            "supply exactly one of freeze_record for a first run or "
            "freeze_authority for recovery"
        )
    recovery_mode = freeze_authority is not None
    authority = (
        revalidate_protocol125_freeze_authority_snapshot(freeze_authority)
        if recovery_mode
        else validate_protocol125_freeze_authority(freeze_record)
    )
    if (
        authority.get("authorization_kind") != AUTHORIZATION_KIND
        or authority.get("authorization_scope") != AUTHORIZATION_SCOPE
        or authority.get("status") != "FROZEN"
    ):
        raise Protocol125RunnerError("freeze validator returned an invalid authority scope")
    adapters = _validated_adapter(adapters, authority)
    candidate = Path(authority["candidate_output_directory"])
    if output_directory is not None and Path(output_directory).resolve() != candidate.resolve():
        raise Protocol125RunnerError("requested output differs from the frozen candidate directory")

    if recovery_mode:
        if (
            candidate.is_symlink()
            or not candidate.is_dir()
            or not (candidate/"recovery_index.json").is_file()
            or (candidate/"recovery_index.json").is_symlink()
        ):
            raise Protocol125RecoveryError(
                "freeze-authorized recovery requires an existing regular recovery index"
            )
    else:
        # All prior checks are read-only.  This is the first filesystem mutation.
        candidate.mkdir(parents=True, exist_ok=True)
    index = RecoveryIndex(
        candidate/"recovery_index.json",
        authority["protocol_path"],
        _authority_inputs(authority),
        maximum_stage_seconds=43200.0,
    )

    parents = {}
    for label in PARENT_LABELS:
        # Each call receives only the common verified authority, never the
        # other parent.  Cross-parent state is unavailable by construction.
        stage = _materialize_stage(
            index,
            candidate,
            stage_id=f"parent/{label}",
            stage_kind="parent",
            expected_bindings=(label,),
            authority=authority,
            adapters=adapters,
            producer=lambda label=label: adapters.construct_parent(
                label,
                freeze_authority=authority,
            ),
            restore_context={"freeze_authority": authority},
            expected_max_seconds=21600.0,
        )
        parents[label] = stage
    parent_identities = {
        label: str(parents[label].bindings[label]) for label in PARENT_LABELS
    }
    _validate_bindings(parent_identities, PARENT_LABELS, "parent identities")

    pre = {}
    for label in PARENT_LABELS:
        pre[label] = _materialize_stage(
            index,
            candidate,
            stage_id=f"pre-acceleration/{label}",
            stage_kind="pre-acceleration",
            expected_bindings=(label,),
            authority=authority,
            adapters=adapters,
            producer=lambda label=label: adapters.compose_pre_acceleration(
                label,
                parents[label],
                freeze_authority=authority,
            ),
            restore_context={
                "freeze_authority": authority,
                "parent_stage": parents[label],
            },
            expected_max_seconds=7200.0,
        )
    pre_records = _ordered_parent_records(pre)
    if not all(_stage_all_pass(pre[label], PRE_ACCELERATION_GROUPS) for label in PARENT_LABELS):
        adjudication = adjudicate_protocol125_ordered(
            pre_records,
            parent_identities=parent_identities,
            protocol_freeze_record=authority,
        )
        return _write_final_adjudication(
            index, candidate, adjudication, authority, adapters,
        )

    post = {}
    for label in PARENT_LABELS:
        post[label] = _materialize_stage(
            index,
            candidate,
            stage_id=f"post-acceleration/{label}",
            stage_kind="post-acceleration",
            expected_bindings=(label,),
            authority=authority,
            adapters=adapters,
            producer=lambda label=label: adapters.compose_post_acceleration(
                label,
                parents[label],
                pre[label],
                freeze_authority=authority,
            ),
            restore_context={
                "freeze_authority": authority,
                "parent_stage": parents[label],
                "pre_acceleration_stage": pre[label],
            },
            expected_max_seconds=21600.0,
        )
    post_records = _ordered_parent_records(post)
    if not all(_stage_all_pass(post[label], POST_ACCELERATION_GROUPS) for label in PARENT_LABELS):
        adjudication = adjudicate_protocol125_ordered(
            pre_records,
            parent_identities=parent_identities,
            protocol_freeze_record=authority,
            post_acceleration_records=post_records,
        )
        return _write_final_adjudication(
            index, candidate, adjudication, authority, adapters,
        )

    two_parent = _materialize_stage(
        index,
        candidate,
        stage_id="two-parent/comparison",
        stage_kind="two-parent",
        expected_bindings=PARENT_LABELS,
        authority=authority,
        adapters=adapters,
        producer=lambda: adapters.compose_two_parent(
            MappingProxyType(dict(parents)),
            MappingProxyType(dict(pre)),
            MappingProxyType(dict(post)),
            parent_identities=MappingProxyType(dict(parent_identities)),
            freeze_authority=authority,
        ),
        restore_context={
            "freeze_authority": authority,
            "parent_stages": MappingProxyType(dict(parents)),
            "pre_acceleration_stages": MappingProxyType(dict(pre)),
            "post_acceleration_stages": MappingProxyType(dict(post)),
        },
        expected_max_seconds=7200.0,
    )
    adjudication = adjudicate_protocol125_ordered(
        pre_records,
        parent_identities=parent_identities,
        protocol_freeze_record=authority,
        post_acceleration_records=post_records,
        two_parent_records={
            group: two_parent.gate_records[group] for group in TWO_PARENT_GROUPS
        },
    )
    return _write_final_adjudication(
        index, candidate, adjudication, authority, adapters,
    )
