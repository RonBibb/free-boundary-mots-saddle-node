"""Concrete, unregistered Protocol-125 scientific-runner adapter.

This module is the sole production composition route from one independently
constructed parent through the pre-acceleration, post-acceleration, and
two-parent scorers.  It installs no global/default adapter and performs no
work at import time.  The runner can use an explicit instance only after this
file and the runner have been included in a prospectively validated freeze.

Every checkpoint uses a lossless, ``allow_pickle=False`` tree codec.  Numeric
and Unicode ndarrays are stored as raw uint8 payloads with explicit dtype and
shape descriptors; mappings, sequence kinds, Python scalar types, signed
zero, infinities, and bytes are preserved.  Reload reconstructs the opaque
Hermite/reference/shared objects only through their canonical builders and
requires exact coefficient and evidence equality before returning a stage.
No acceleration solve is repeated during reload.
"""

from __future__ import annotations

import ast
import base64
import binascii
import hashlib
import json
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from bhps.joint_parent_acceleration import (
    Protocol125AccelerationScientificFailure,
    solve_joint_parent_acceleration_fixed_point,
    validate_protocol125_acceleration_failure_record,
)
from bhps.joint_parent_bulk_reference import FiniteWallReferenceHermitePair
from bhps.joint_parent_bulk_validation import run_protocol125_bulk_validation
from bhps.joint_parent_construction import (
    Protocol125ScientificConstructionFailure,
    construct_joint_parent_position,
    validate_protocol125_construction_failure_record,
)
from bhps.joint_parent_endpoint_audits import (
    build_protocol125_wall_profile_evidence,
)
from bhps.joint_parent_environment_contract import (
    CONTRACT_PATH as RUNTIME_ENVIRONMENT_CONTRACT_PATH,
    validate_protocol125_environment_contract,
)
from bhps.joint_parent_fields import bulk_acceleration_from_completed_position
from bhps.joint_parent_final_inputs import (
    build_protocol125_final_matrix_inputs,
)
from bhps.joint_parent_final_matrix import (
    evaluate_protocol125_final_representation_matrix,
)
from bhps.joint_parent_legacy_holdout import (
    build_protocol125_preacceleration_legacy_position_inputs,
)
from bhps.joint_parent_lineage_adapter import (
    build_protocol125_append_only_position_lineage,
)
from bhps.joint_parent_native_evidence import (
    build_protocol125_native_position_tangent_evidence,
)
from bhps.joint_parent_position_pair import PositionOnlyConstrainedHermitePair
from bhps.joint_parent_position_state import (
    build_joint_parent_position_state,
    derive_joint_parent_position_outer_contract,
)
from bhps.joint_parent_postacceleration import (
    Protocol125PostAccelerationInputs,
    Protocol125PostAccelerationFailureInputs,
    capture_protocol125_bulk_sampler_provenance,
    capture_protocol125_acceleration_failure_provenance,
    capture_protocol125_postacceleration_provenance,
    compose_protocol125_acceleration_failure_records,
    compose_protocol125_postacceleration_records,
)
from bhps.joint_parent_preacceleration import (
    Protocol125PreAccelerationCoreInputs,
    Protocol125PreAccelerationInputs,
    Protocol125PreAccelerationRepresentationInputs,
    capture_protocol125_bulk_prerequisite_provenance,
    capture_protocol125_legacy_sampling_provenance,
    capture_protocol125_position_prefix_provenance,
    compose_protocol125_construction_failure_records,
    compose_protocol125_representation_coefficient_failure_records,
    evaluate_protocol125_position_prefix,
    extend_protocol125_legacy_sampling,
    finalize_protocol125_preacceleration_stop,
    finish_protocol125_bulk_prerequisite,
)
from bhps.joint_parent_representation import (
    Protocol125RepresentationCoefficientFailure,
    bind_protocol125_representation_coefficient_failure,
    validate_protocol125_representation_coefficient_failure,
)
from bhps.joint_parent_refinement_diagnostics import (
    axis_acceleration_derivative_image_profile,
    correction_profile,
    frozen_validation_meshes,
)
from bhps.joint_parent_scientific_runner import (
    POST_ACCELERATION_GROUPS,
    PRE_ACCELERATION_GROUPS,
    REQUIRED_ADAPTER_CAPABILITIES,
    TWO_PARENT_GROUPS,
    Protocol125CheckpointPayload,
    Protocol125RunnerStage,
    Protocol125ScientificAdapters,
    _validate_gate_records,
    protocol125_stage_record_sha256,
)
from bhps.joint_parent_shared_representation import (
    build_protocol125_shared_representation,
)
from bhps.joint_parent_two_parent import (
    Protocol125TwoParentInputs,
    compose_protocol125_two_parent_records,
    protocol125_two_parent_input_hashes,
)
from bhps.matched_staged_continuum import hash_arrays
from bhps.recovery_indexer import sha256_file


PROTOCOL_IDENTIFIER = "Protocol-125-concrete-production-adapter-v1"
CODEC_IDENTIFIER = "Protocol-125-lossless-tree-uint8-codec-v1"
SOURCE_MANIFEST_NAME = "protocol125_production_adapter"
PARENT_LABELS = ("N0", "N1")
V_MESH_NAMES = ("V0", "V1", "V2")

# Exact local Python closure reached from the runner and this adapter.  The
# inventory is checked against the import syntax before it is exposed, so a
# new local dependency cannot silently escape the next prospective freeze.
TRANSITIVE_LOCAL_MODULES = (
    "bhps.adm_corner",
    "bhps.anisotropic_geometry",
    "bhps.anisotropic_initial_data",
    "bhps.axisymmetric_reduced_wave_evolution",
    "bhps.corrected_A790_R12_builder",
    "bhps.finite_wall_high_order_solver",
    "bhps.generalized_harmonic_jets",
    "bhps.gh_source_driver",
    "bhps.gw_background",
    "bhps.gw_slice_high_order_solver",
    "bhps.initial_data",
    "bhps.joint_parent_acceleration",
    "bhps.joint_parent_adjudication",
    "bhps.joint_parent_boundary_contracts",
    "bhps.joint_parent_builder",
    "bhps.joint_parent_bulk_audit",
    "bhps.joint_parent_bulk_reference",
    "bhps.joint_parent_bulk_validation",
    "bhps.joint_parent_construction",
    "bhps.joint_parent_endpoint_audits",
    "bhps.joint_parent_environment_contract",
    "bhps.joint_parent_fields",
    "bhps.joint_parent_final_inputs",
    "bhps.joint_parent_final_matrix",
    "bhps.joint_parent_freeze_authority",
    "bhps.joint_parent_legacy_holdout",
    "bhps.joint_parent_lineage_adapter",
    "bhps.joint_parent_native_completion",
    "bhps.joint_parent_native_evidence",
    "bhps.joint_parent_ordered_adjudicator",
    "bhps.joint_parent_position_audits",
    "bhps.joint_parent_position_pair",
    "bhps.joint_parent_position_state",
    "bhps.joint_parent_postacceleration",
    "bhps.joint_parent_preacceleration",
    "bhps.joint_parent_production_adapter",
    "bhps.joint_parent_protocol125_sampling_lineage",
    "bhps.joint_parent_refinement_diagnostics",
    "bhps.joint_parent_representation",
    "bhps.joint_parent_scientific_runner",
    "bhps.joint_parent_selective_algebra",
    "bhps.joint_parent_shape",
    "bhps.joint_parent_shared_representation",
    "bhps.joint_parent_source_closure",
    "bhps.joint_parent_two_parent",
    "bhps.junction_preservation_diagnostic",
    "bhps.junction_second_preservation_diagnostic",
    "bhps.lapse_acceleration_corner",
    "bhps.linearized_gh_einstein_scalar",
    "bhps.matched_staged_continuum",
    "bhps.nonlinear_regular_so3_evolution",
    "bhps.physical_corner_corrector",
    "bhps.recovery_indexer",
    "bhps.regular_so3_gh_reduction",
    "bhps.scalar_pulse",
    "bhps.staged_boundary_preservation",
)


class Protocol125ProductionAdapterError(RuntimeError):
    """Raised when production composition or checkpoint reload differs."""


def _module_path(module_name):
    package_root = Path(__file__).resolve().parent
    if module_name == "bhps":
        return package_root/"__init__.py"
    prefix = "bhps."
    if not str(module_name).startswith(prefix):
        raise ValueError("local Protocol-125 module is outside bhps")
    return package_root.joinpath(*str(module_name)[len(prefix):].split(".")).with_suffix(
        ".py"
    )


def _imported_local_modules(module_name, path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as error:
        raise Protocol125ProductionAdapterError(
            f"cannot parse transitive source module {module_name}"
        ) from error
    imported = set()
    package_parts = module_name.split(".")[:-1]
    for node in ast.walk(tree):
        candidates = []
        if isinstance(node, ast.Import):
            candidates.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if node.level > len(package_parts):
                    raise Protocol125ProductionAdapterError(
                        f"invalid relative import in {module_name}"
                    )
                base = package_parts[:len(package_parts)-node.level+1]
                if node.module:
                    base.extend(node.module.split("."))
                candidates.append(".".join(base))
            elif node.module:
                candidates.append(node.module)
            if candidates == ["bhps"]:
                candidates.extend(
                    f"bhps.{alias.name}" for alias in node.names
                    if alias.name != "*"
                )
        for candidate in candidates:
            if candidate == "bhps":
                imported.add("bhps")
                continue
            if candidate.startswith("bhps.") and _module_path(candidate).is_file():
                imported.add(candidate)
    return imported


def _validate_transitive_local_module_inventory():
    declared = set(TRANSITIVE_LOCAL_MODULES)
    if len(declared) != len(TRANSITIVE_LOCAL_MODULES):
        raise Protocol125ProductionAdapterError(
            "transitive local-module inventory contains duplicates"
        )
    roots = {
        "bhps.joint_parent_scientific_runner",
        "bhps.joint_parent_production_adapter",
    }
    reached = set()
    pending = list(roots)
    while pending:
        module_name = pending.pop()
        if module_name in reached:
            continue
        if module_name not in declared:
            raise Protocol125ProductionAdapterError(
                f"transitive local module {module_name} is undeclared"
            )
        path = _module_path(module_name)
        if path.is_symlink() or not path.is_file():
            raise Protocol125ProductionAdapterError(
                f"transitive local module {module_name} is unavailable"
            )
        reached.add(module_name)
        pending.extend(
            imported for imported in _imported_local_modules(module_name, path)
            if imported != "bhps" and imported not in reached
        )
    if reached != declared:
        extras = tuple(sorted(declared-reached))
        raise Protocol125ProductionAdapterError(
            f"transitive local-module inventory is not exact: unreachable={extras}"
        )


def protocol125_production_source_inventory():
    """Return the exact code, environment, and immutable-input freeze closure."""
    _validate_transitive_local_module_inventory()
    validate_protocol125_environment_contract(
        RUNTIME_ENVIRONMENT_CONTRACT_PATH,
    )
    project_root = Path(__file__).resolve().parents[2]
    inventory = {
        "source:bhps-package-init": str(_module_path("bhps").resolve()),
        SOURCE_MANIFEST_NAME: str(Path(__file__).resolve()),
        "protocol125_scientific_runner": str(
            _module_path("bhps.joint_parent_scientific_runner").resolve()
        ),
    }
    for module_name in TRANSITIVE_LOCAL_MODULES:
        if module_name in {
            "bhps.joint_parent_production_adapter",
            "bhps.joint_parent_scientific_runner",
        }:
            continue
        inventory[f"source:{module_name}"] = str(_module_path(module_name).resolve())
    inventory.update({
        "environment:pyproject": str((project_root/"pyproject.toml").resolve()),
        "environment:uv-lock": str((project_root/"uv.lock").resolve()),
        "environment:runtime-contract": str(
            Path(RUNTIME_ENVIRONMENT_CONTRACT_PATH).resolve()
        ),
        "input:sealed-protocol120-parent": str((
            project_root
            / "results/corrected_A790_matched_staged_continuum_recovery"
            / "phase_a_parent_projection.npz"
        ).resolve()),
        "input:frozen-family-knot-state": str((
            project_root/"results/corrected_family_knot_A8_state.npz"
        ).resolve()),
    })
    for logical_name, raw_path in inventory.items():
        path = Path(raw_path)
        if path.is_symlink() or not path.is_file():
            raise Protocol125ProductionAdapterError(
                f"required freeze inventory entry {logical_name} is unavailable"
            )
    if len(set(inventory.values())) != len(inventory):
        raise Protocol125ProductionAdapterError(
            "production freeze inventory reuses a file path"
        )
    return MappingProxyType(inventory)


def _valid_sha256(value):
    value = str(value)
    return bool(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _immutable(value):
    source = np.asarray(value)
    array = np.ascontiguousarray(source) if source.ndim else source.copy()
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _encoded_arrays_sha256(arrays):
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(np.asarray(arrays[name]))
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


class _LosslessTreeEncoder:
    def __init__(self):
        self.arrays = {}
        self.counter = 0

    def _raw(self, value):
        source = np.asarray(value)
        array = np.ascontiguousarray(source) if source.ndim else source.copy()
        if array.dtype == object or array.dtype.fields is not None:
            raise TypeError(
                "Protocol-125 checkpoint tree forbids object or structured arrays"
            )
        name = f"codec_raw_{self.counter:08d}"
        self.counter += 1
        raw = np.frombuffer(array.tobytes(), dtype=np.uint8).copy()
        self.arrays[name] = raw
        return name, array.dtype.str, tuple(int(item) for item in array.shape)

    def encode(self, value, path):
        if isinstance(value, Mapping):
            if any(type(name) is not str or not name for name in value):
                raise TypeError(f"checkpoint mapping {path} has an invalid key")
            return {
                "type": "mapping",
                "items": [
                    [name, self.encode(value[name], f"{path}/{name}")]
                    for name in value
                ],
            }
        if isinstance(value, tuple):
            return {
                "type": "tuple",
                "items": [
                    self.encode(item, f"{path}/{index}")
                    for index, item in enumerate(value)
                ],
            }
        if isinstance(value, list):
            return {
                "type": "list",
                "items": [
                    self.encode(item, f"{path}/{index}")
                    for index, item in enumerate(value)
                ],
            }
        if isinstance(value, np.ndarray):
            name, dtype, shape = self._raw(value)
            return {
                "type": "ndarray", "array": name,
                "dtype": dtype, "shape": list(shape),
            }
        if isinstance(value, np.generic):
            name, dtype, shape = self._raw(np.asarray(value))
            return {
                "type": "numpy-scalar", "array": name,
                "dtype": dtype, "shape": list(shape),
            }
        if value is None:
            return {"type": "none"}
        if type(value) is bool:
            return {"type": "bool", "value": value}
        if type(value) is int:
            return {"type": "int", "value": str(value)}
        if type(value) is float:
            return {
                "type": "float64-bits",
                "value": base64.b64encode(struct.pack("!d", value)).decode("ascii"),
            }
        if type(value) is complex:
            return {
                "type": "complex128-bits",
                "value": base64.b64encode(
                    struct.pack("!dd", value.real, value.imag)
                ).decode("ascii"),
            }
        if type(value) is str:
            return {"type": "str", "value": value}
        if type(value) is bytes:
            return {
                "type": "bytes",
                "value": base64.b64encode(value).decode("ascii"),
            }
        raise TypeError(
            f"checkpoint tree {path} contains unsupported {type(value).__name__}"
        )


def _pack_roots(roots):
    if not isinstance(roots, Mapping) or not roots:
        raise ValueError("checkpoint codec requires named roots")
    if any(type(name) is not str or not name for name in roots):
        raise TypeError("checkpoint root names must be nonempty strings")
    encoder = _LosslessTreeEncoder()
    descriptor = {
        "codec": CODEC_IDENTIFIER,
        "roots": [
            [name, encoder.encode(roots[name], f"root/{name}")]
            for name in roots
        ],
    }
    manifest = json.dumps(
        descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    arrays = dict(encoder.arrays)
    arrays["codec_manifest_utf8"] = np.frombuffer(manifest, dtype=np.uint8).copy()
    return arrays, {
        "codec_identifier": CODEC_IDENTIFIER,
        "codec_payload_sha256": _encoded_arrays_sha256(arrays),
        # Checkpoint metadata is serialized through strict JSON by the runner.
        # Keep it in that canonical representation before the first write so
        # an immediate restore does not see tuple -> list type drift.
        "codec_root_order": list(roots),
        "codec_array_count": len(arrays),
    }


def _decode_raw(node, arrays):
    name = node.get("array")
    if type(name) is not str or name not in arrays:
        raise Protocol125ProductionAdapterError("checkpoint tree raw array is missing")
    raw = np.ascontiguousarray(np.asarray(arrays[name]))
    if raw.dtype != np.uint8 or raw.ndim != 1:
        raise Protocol125ProductionAdapterError("checkpoint tree raw payload is not uint8")
    try:
        dtype = np.dtype(node["dtype"])
        shape = tuple(int(item) for item in node["shape"])
    except (KeyError, TypeError, ValueError) as error:
        raise Protocol125ProductionAdapterError(
            "checkpoint tree dtype/shape descriptor is invalid"
        ) from error
    if dtype.hasobject or dtype.fields is not None or any(item < 0 for item in shape):
        raise Protocol125ProductionAdapterError(
            "checkpoint tree dtype/shape descriptor is unsafe"
        )
    expected = math.prod(shape)*dtype.itemsize
    if raw.nbytes != expected:
        raise Protocol125ProductionAdapterError("checkpoint tree raw byte count differs")
    return np.frombuffer(raw.tobytes(), dtype=dtype).reshape(shape)


def _decode_node(node, arrays, path):
    if not isinstance(node, Mapping) or type(node.get("type")) is not str:
        raise Protocol125ProductionAdapterError(f"checkpoint descriptor {path} is invalid")
    kind = node["type"]
    if kind == "mapping":
        items = node.get("items")
        if not isinstance(items, list):
            raise Protocol125ProductionAdapterError(f"checkpoint mapping {path} is invalid")
        output = {}
        for item in items:
            if (
                not isinstance(item, list) or len(item) != 2
                or type(item[0]) is not str or not item[0]
                or item[0] in output
            ):
                raise Protocol125ProductionAdapterError(
                    f"checkpoint mapping item {path} is invalid"
                )
            output[item[0]] = _decode_node(item[1], arrays, f"{path}/{item[0]}")
        return output
    if kind in ("tuple", "list"):
        items = node.get("items")
        if not isinstance(items, list):
            raise Protocol125ProductionAdapterError(f"checkpoint sequence {path} is invalid")
        values = [
            _decode_node(item, arrays, f"{path}/{index}")
            for index, item in enumerate(items)
        ]
        return tuple(values) if kind == "tuple" else values
    if kind == "ndarray":
        return _immutable(_decode_raw(node, arrays))
    if kind == "numpy-scalar":
        value = _decode_raw(node, arrays)
        if value.shape != ():
            raise Protocol125ProductionAdapterError("numpy scalar descriptor is not scalar")
        return value[()]
    if kind == "none":
        return None
    if kind == "bool" and type(node.get("value")) is bool:
        return node["value"]
    if kind == "int" and type(node.get("value")) is str:
        try:
            return int(node["value"])
        except ValueError as error:
            raise Protocol125ProductionAdapterError(
                "integer checkpoint payload is invalid"
            ) from error
    if kind == "float64-bits" and type(node.get("value")) is str:
        try:
            raw = base64.b64decode(node["value"], validate=True)
        except (binascii.Error, ValueError) as error:
            raise Protocol125ProductionAdapterError(
                "float checkpoint payload is invalid"
            ) from error
        if len(raw) != 8:
            raise Protocol125ProductionAdapterError("float checkpoint width differs")
        return struct.unpack("!d", raw)[0]
    if kind == "complex128-bits" and type(node.get("value")) is str:
        try:
            raw = base64.b64decode(node["value"], validate=True)
        except (binascii.Error, ValueError) as error:
            raise Protocol125ProductionAdapterError(
                "complex checkpoint payload is invalid"
            ) from error
        if len(raw) != 16:
            raise Protocol125ProductionAdapterError("complex checkpoint width differs")
        real, imaginary = struct.unpack("!dd", raw)
        return complex(real, imaginary)
    if kind == "str" and type(node.get("value")) is str:
        return node["value"]
    if kind == "bytes" and type(node.get("value")) is str:
        try:
            return base64.b64decode(node["value"], validate=True)
        except (binascii.Error, ValueError) as error:
            raise Protocol125ProductionAdapterError(
                "bytes checkpoint payload is invalid"
            ) from error
    raise Protocol125ProductionAdapterError(
        f"checkpoint descriptor {path} has unsupported or malformed type {kind}"
    )


def _unpack_roots(arrays, metadata):
    if not isinstance(arrays, Mapping) or "codec_manifest_utf8" not in arrays:
        raise Protocol125ProductionAdapterError("checkpoint codec manifest is absent")
    if str(metadata.get("codec_identifier", "")) != CODEC_IDENTIFIER:
        raise Protocol125ProductionAdapterError("checkpoint codec identifier differs")
    if any(
        np.asarray(value).dtype != np.uint8 or np.asarray(value).ndim != 1
        for value in arrays.values()
    ):
        raise Protocol125ProductionAdapterError(
            "checkpoint codec payload contains a non-raw array"
        )
    found_hash = _encoded_arrays_sha256(arrays)
    if str(metadata.get("codec_payload_sha256", "")) != found_hash:
        raise Protocol125ProductionAdapterError("checkpoint codec payload hash differs")
    if int(metadata.get("codec_array_count", -1)) != len(arrays):
        raise Protocol125ProductionAdapterError("checkpoint codec array count differs")
    try:
        manifest = bytes(np.asarray(arrays["codec_manifest_utf8"]))
        descriptor = json.loads(manifest.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise Protocol125ProductionAdapterError("checkpoint codec manifest is invalid") from error
    if descriptor.get("codec") != CODEC_IDENTIFIER:
        raise Protocol125ProductionAdapterError("checkpoint manifest codec differs")
    roots = descriptor.get("roots")
    if not isinstance(roots, list):
        raise Protocol125ProductionAdapterError("checkpoint root descriptor is invalid")
    result = {}
    for item in roots:
        if (
            not isinstance(item, list) or len(item) != 2
            or type(item[0]) is not str or not item[0] or item[0] in result
        ):
            raise Protocol125ProductionAdapterError("checkpoint root entry is invalid")
        result[item[0]] = _decode_node(item[1], arrays, f"root/{item[0]}")
    if tuple(result) != tuple(metadata.get("codec_root_order", ())):
        raise Protocol125ProductionAdapterError("checkpoint root order differs")
    referenced = {"codec_manifest_utf8"}

    def collect(node):
        if isinstance(node, Mapping):
            if node.get("type") in ("ndarray", "numpy-scalar"):
                referenced.add(node["array"])
            for value in node.values():
                collect(value)
        elif isinstance(node, list):
            for value in node:
                collect(value)

    collect(descriptor)
    if referenced != set(arrays):
        raise Protocol125ProductionAdapterError("checkpoint contains unreferenced raw arrays")
    return result


def _tree_bitwise_digest(value):
    arrays, metadata = _pack_roots({"value": value})
    digest = hashlib.sha256()
    digest.update(metadata["codec_payload_sha256"].encode("ascii"))
    digest.update(json.dumps(
        metadata, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8"))
    for name in sorted(arrays):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(arrays[name]).tobytes())
    return digest.hexdigest()


def _require_tree_equal(left, right, label):
    if _tree_bitwise_digest(left) != _tree_bitwise_digest(right):
        raise Protocol125ProductionAdapterError(f"{label} changed across checkpoint reload")


def _parent_identity(parent):
    return hash_arrays(
        np.asarray(parent["label"]),
        np.asarray(parent["z"]),
        np.asarray(parent["r"]),
        np.asarray(parent["position"]),
        np.asarray(parent["selector_q"]),
        np.asarray(parent["phi"]),
        np.asarray(parent["reference_q"]),
        np.asarray(parent["reference_phi"]),
    )


def _parent_payload(parent):
    if not isinstance(parent, Mapping):
        raise TypeError("production parent runtime is not a mapping")
    payload = {
        name: value for name, value in parent.items()
        if name != "freeze_authority"
    }
    identity = str(payload.get("parent_identity", ""))
    if not _valid_sha256(identity) or _parent_identity(payload) != identity:
        raise Protocol125ProductionAdapterError("production parent identity differs")
    return payload


@dataclass(frozen=True)
class Protocol125ProductionParentState:
    parent: Mapping | None
    construction_failure: Mapping | None = None


@dataclass(frozen=True)
class Protocol125ProductionConstructionFailurePreState:
    construction_failure: Mapping
    pre_result: Mapping


@dataclass(frozen=True)
class Protocol125ProductionRepresentationFailurePreState:
    representation_coefficient_failure: Mapping
    pre_result: Mapping


@dataclass(frozen=True)
class Protocol125ProductionPreState:
    position_pair: PositionOnlyConstrainedHermitePair
    position_state_record: Mapping
    reference_pair: FiniteWallReferenceHermitePair
    native_evidence: Mapping
    bulk_audit: Mapping
    legacy_inputs: Mapping
    pre_inputs: Protocol125PreAccelerationInputs
    pre_provenance: Mapping
    pre_result: Mapping


@dataclass(frozen=True)
class Protocol125ProductionStoppedPreState:
    position_pair: PositionOnlyConstrainedHermitePair
    position_state_record: Mapping
    reference_pair: FiniteWallReferenceHermitePair
    native_evidence: Mapping
    legacy_inputs: Mapping | None
    staged_provenance: Mapping
    staged_result: Mapping
    pre_result: Mapping


@dataclass(frozen=True)
class Protocol125ProductionPostState:
    bulk_acceleration: np.ndarray
    bulk_record: Mapping
    compatible_acceleration: np.ndarray
    fixed_point_record: Mapping
    shared_build: object
    wall_profile_evidence: Mapping
    final_matrix_adapter_record: Mapping
    final_matrix_result: Mapping
    append_only_lineage: Mapping
    correction_profile: Mapping
    axis_image_profile: Mapping
    bulk_sampler_provenance: Mapping
    post_provenance: Mapping
    post_result: Mapping
    position_v2: np.ndarray
    hzz_zz_v2: np.ndarray
    a_hzz_v2: np.ndarray


@dataclass(frozen=True)
class Protocol125ProductionAccelerationFailurePostState:
    bulk_acceleration: np.ndarray
    bulk_record: Mapping
    acceleration_failure: Mapping
    failure_inputs: Protocol125PostAccelerationFailureInputs
    failure_provenance: Mapping
    post_result: Mapping


@dataclass(frozen=True)
class Protocol125ProductionTwoParentState:
    input_hashes: Mapping
    records: Mapping


def _checkpoint(stage_kind, bindings, gate_records, roots):
    if gate_records is not None:
        groups = {
            "pre-acceleration": PRE_ACCELERATION_GROUPS,
            "post-acceleration": POST_ACCELERATION_GROUPS,
            "two-parent": TWO_PARENT_GROUPS,
        }.get(str(stage_kind))
        if groups is None:
            raise Protocol125ProductionAdapterError(
                "production gate checkpoint has an unknown stage kind"
            )
        _validate_gate_records(
            gate_records,
            groups,
            bindings,
            f"production:{stage_kind}",
        )
    arrays, codec_metadata = _pack_roots(roots)
    record_sha256 = protocol125_stage_record_sha256(
        stage_kind, bindings, gate_records,
    )
    return Protocol125CheckpointPayload(
        arrays=arrays,
        metadata={
            "stage_kind": stage_kind,
            "complete_state": True,
            "restartable_without_unrecorded_state": True,
            "record_sha256": record_sha256,
            "adapter_protocol_identifier": PROTOCOL_IDENTIFIER,
            **codec_metadata,
        },
    )


def _source_parent(stage, label):
    if not isinstance(stage, Protocol125RunnerStage) or not isinstance(
        stage.runtime, Protocol125ProductionParentState,
    ):
        raise TypeError("production adapter requires its restored parent stage")
    if stage.runtime.construction_failure is not None or stage.runtime.parent is None:
        raise Protocol125ProductionAdapterError(
            "a construction-failure parent has no represented position"
        )
    parent = stage.runtime.parent
    if (
        str(parent.get("label", "")) != label
        or str(parent.get("parent_identity", "")) != str(stage.bindings[label])
        or _parent_identity(parent) != str(stage.bindings[label])
    ):
        raise Protocol125ProductionAdapterError("parent stage binding differs")
    return parent


def _source_construction_failure(stage, label):
    if not isinstance(stage, Protocol125RunnerStage) or not isinstance(
        stage.runtime, Protocol125ProductionParentState,
    ):
        raise TypeError("production adapter requires its restored parent stage")
    failure = stage.runtime.construction_failure
    if failure is None:
        return None
    failure = validate_protocol125_construction_failure_record(failure)
    if (
        stage.runtime.parent is not None
        or str(failure["parent_label"]) != label
        or str(failure["parent_identity"]) != str(stage.bindings.get(label, ""))
    ):
        raise Protocol125ProductionAdapterError(
            "construction-failure parent binding differs"
        )
    return failure


def _source_pre(stage, label, identity):
    if not isinstance(stage, Protocol125RunnerStage) or not isinstance(
        stage.runtime, Protocol125ProductionPreState,
    ):
        raise TypeError("production adapter requires its restored pre-acceleration stage")
    runtime = stage.runtime
    if (
        str(stage.bindings.get(label, "")) != identity
        or str(runtime.pre_result.get("parent_identity", "")) != identity
    ):
        raise Protocol125ProductionAdapterError("pre-acceleration binding differs")
    return runtime


def _source_post(stage, label, identity):
    if not isinstance(stage, Protocol125RunnerStage) or not isinstance(
        stage.runtime, Protocol125ProductionPostState,
    ):
        raise TypeError("production adapter requires its restored post-acceleration stage")
    runtime = stage.runtime
    if (
        str(stage.bindings.get(label, "")) != identity
        or str(runtime.post_result.get("parent_identity", "")) != identity
    ):
        raise Protocol125ProductionAdapterError("post-acceleration binding differs")
    return runtime


class Protocol125ProductionAdapter:
    """Explicit production adapter; never registered or executed implicitly."""

    def runner_adapters(self):
        path = Path(__file__).resolve()
        return Protocol125ScientificAdapters(
            identifier=PROTOCOL_IDENTIFIER,
            implementation_path=str(path),
            implementation_sha256=sha256_file(path),
            source_manifest_name=SOURCE_MANIFEST_NAME,
            capabilities=REQUIRED_ADAPTER_CAPABILITIES,
            construct_parent=self.construct_parent,
            compose_pre_acceleration=self.compose_pre_acceleration,
            compose_post_acceleration=self.compose_post_acceleration,
            compose_two_parent=self.compose_two_parent,
            restore_checkpoint=self.restore_checkpoint,
            source_manifest_inventory=protocol125_production_source_inventory(),
            runtime_environment_verifier=(
                validate_protocol125_environment_contract
            ),
        )

    def construct_parent(self, label, *, freeze_authority):
        label = str(label)
        if label not in PARENT_LABELS:
            raise ValueError("production parent label must be N0 or N1")
        try:
            parent = _parent_payload(construct_joint_parent_position(
                label, freeze_authority=freeze_authority,
            ))
        except Protocol125ScientificConstructionFailure as error:
            failure = validate_protocol125_construction_failure_record(error.record)
            identity = str(failure["parent_identity"])
            bindings = {label: identity}
            runtime = Protocol125ProductionParentState(None, failure)
            roots = {"construction_failure": failure}
            return Protocol125RunnerStage(
                runtime,
                _checkpoint("parent", bindings, None, roots),
                bindings,
                None,
            )
        identity = str(parent["parent_identity"])
        bindings = {label: identity}
        runtime = Protocol125ProductionParentState(MappingProxyType(parent), None)
        roots = {"parent": parent}
        return Protocol125RunnerStage(
            runtime,
            _checkpoint("parent", bindings, None, roots),
            bindings,
            None,
        )

    def compose_pre_acceleration(
        self, label, parent_stage, *, freeze_authority,
    ):
        del freeze_authority
        label = str(label)
        construction_failure = _source_construction_failure(parent_stage, label)
        if construction_failure is not None:
            result = compose_protocol125_construction_failure_records(
                construction_failure,
            )
            identity = str(construction_failure["parent_identity"])
            bindings = {label: identity}
            runtime = Protocol125ProductionConstructionFailurePreState(
                construction_failure,
                result,
            )
            roots = {
                "construction_failure": construction_failure,
                "pre_result": result,
            }
            return Protocol125RunnerStage(
                runtime,
                _checkpoint(
                    "pre-acceleration", bindings, result["groups"], roots,
                ),
                bindings,
                result["groups"],
            )
        parent = _source_parent(parent_stage, label)
        identity = str(parent["parent_identity"])
        background = parent["background"]
        z = np.asarray(parent["z"], dtype=float)
        r = np.asarray(parent["r"], dtype=float)
        try:
            outer = derive_joint_parent_position_outer_contract(parent)
            state, position_record = build_joint_parent_position_state(
                parent["position"],
                z,
                r,
                background,
                outer_open_face_contract=outer,
                parent_r_max=float(r[-1]),
            )
            pair = PositionOnlyConstrainedHermitePair.from_primary(state)
            reference = FiniteWallReferenceHermitePair.build(
                z, r, parent["reference_q"], parent["reference_phi"],
            )
        except Protocol125RepresentationCoefficientFailure as error:
            evidence = bind_protocol125_representation_coefficient_failure(
                error.evidence,
                identity,
            )
            result = (
                compose_protocol125_representation_coefficient_failure_records(
                    parent,
                    parent["construction_provenance_record"],
                    evidence,
                )
            )
            bindings = {label: identity}
            runtime = Protocol125ProductionRepresentationFailurePreState(
                evidence,
                result,
            )
            roots = {
                "representation_coefficient_failure": evidence,
                "pre_result": result,
            }
            return Protocol125RunnerStage(
                runtime,
                _checkpoint(
                    "pre-acceleration", bindings, result["groups"], roots,
                ),
                bindings,
                result["groups"],
            )
        native = build_protocol125_native_position_tangent_evidence(
            parent, pair, position_record,
        )
        core_inputs = Protocol125PreAccelerationCoreInputs(
            parent_mapping=parent,
            position_pair=pair,
            reference_pair=reference,
            construction_provenance=parent["construction_provenance_record"],
            native_position_tangent_evidence=native,
        )
        position_provenance = capture_protocol125_position_prefix_provenance(
            core_inputs,
        )
        position_prefix = evaluate_protocol125_position_prefix(
            core_inputs, position_provenance,
        )
        bindings = {label: identity}
        common_roots = {
            "position_pair_arrays": pair.coefficient_arrays("position_pair"),
            "position_state_record": position_record,
            "reference_pair_arrays": reference.coefficient_arrays("reference_pair"),
            "native_evidence": native,
        }
        if not position_prefix["passed"]:
            result = finalize_protocol125_preacceleration_stop(position_prefix)
            roots = {
                **common_roots,
                "position_prefix_provenance": position_provenance,
                "position_prefix_result": position_prefix,
                "pre_result": result,
            }
            runtime = Protocol125ProductionStoppedPreState(
                pair, position_record, reference, native, None,
                position_provenance, position_prefix, result,
            )
            return Protocol125RunnerStage(
                runtime,
                _checkpoint(
                    "pre-acceleration", bindings, result["groups"], roots,
                ),
                bindings,
                result["groups"],
            )
        legacy = build_protocol125_preacceleration_legacy_position_inputs(
            pair, parent_identity=identity,
        )
        representation_inputs = Protocol125PreAccelerationRepresentationInputs(
            parent_mapping=parent,
            position_pair=pair,
            reference_pair=reference,
            construction_provenance=parent["construction_provenance_record"],
            native_position_tangent_evidence=native,
            legacy_Q33_by_mesh=legacy["legacy_Q33_by_mesh"],
            legacy_Q55_by_mesh=legacy["legacy_Q55_by_mesh"],
            legacy_component_orders=legacy["component_orders"],
        )
        representation_provenance = (
            capture_protocol125_legacy_sampling_provenance(
                representation_inputs, position_prefix,
            )
        )
        representation_prefix = extend_protocol125_legacy_sampling(
            representation_inputs, position_prefix, representation_provenance,
        )
        if not representation_prefix["passed"]:
            result = finalize_protocol125_preacceleration_stop(
                representation_prefix,
            )
            roots = {
                **common_roots,
                "legacy_inputs": legacy,
                "representation_prefix_provenance": representation_provenance,
                "representation_prefix_result": representation_prefix,
                "pre_result": result,
            }
            runtime = Protocol125ProductionStoppedPreState(
                pair, position_record, reference, native, legacy,
                representation_provenance, representation_prefix, result,
            )
            return Protocol125RunnerStage(
                runtime,
                _checkpoint(
                    "pre-acceleration", bindings, result["groups"], roots,
                ),
                bindings,
                result["groups"],
            )
        bulk = run_protocol125_bulk_validation(
            label, pair, reference, background,
        )
        inputs = Protocol125PreAccelerationInputs(
            parent_mapping=parent,
            position_pair=pair,
            reference_pair=reference,
            construction_provenance=parent["construction_provenance_record"],
            native_position_tangent_evidence=native,
            legacy_Q33_by_mesh=legacy["legacy_Q33_by_mesh"],
            legacy_Q55_by_mesh=legacy["legacy_Q55_by_mesh"],
            legacy_component_orders=legacy["component_orders"],
            bulk_validation_audit=bulk,
        )
        provenance = capture_protocol125_bulk_prerequisite_provenance(
            inputs, representation_prefix,
        )
        result = finish_protocol125_bulk_prerequisite(
            inputs, representation_prefix, provenance,
        )
        runtime = Protocol125ProductionPreState(
            pair, position_record, reference, native, bulk, legacy,
            inputs, provenance, result,
        )
        roots = {
            **common_roots,
            "legacy_inputs": legacy,
            "bulk_audit": bulk,
            "pre_provenance": provenance,
            "pre_result": result,
        }
        return Protocol125RunnerStage(
            runtime,
            _checkpoint("pre-acceleration", bindings, result["groups"], roots),
            bindings,
            result["groups"],
        )

    @staticmethod
    def _post_products(parent, pre, bulk_acceleration, compatible, fixed):
        label = str(parent["label"])
        identity = str(parent["parent_identity"])
        background = parent["background"]
        z = np.asarray(parent["z"], dtype=float)
        r = np.asarray(parent["r"], dtype=float)
        shared = build_protocol125_shared_representation(
            parent,
            background,
            pre.position_pair,
            pre.position_state_record,
            compatible,
            bulk_acceleration,
            fixed["source_triplet"],
        )
        velocity = np.zeros_like(np.asarray(parent["position"], dtype=float))
        if np.any(np.signbit(velocity)):
            raise AssertionError("production time-symmetric velocity lost positive zero")
        wall = build_protocol125_wall_profile_evidence(
            parent,
            velocity,
            compatible,
            fixed["source_triplet"],
            shared.final_pair,
        )
        final_bundle = build_protocol125_final_matrix_inputs(
            shared,
            pre.reference_pair,
            background,
            parent_label=label,
            parent_identity=identity,
        )
        final_matrix = evaluate_protocol125_final_representation_matrix(
            final_bundle.inputs, final_bundle.provenance,
        )
        lineage = build_protocol125_append_only_position_lineage(
            pre.position_pair, shared.final_pair, pre.reference_pair,
        )
        meshes = frozen_validation_meshes()
        dense_r = np.asarray(meshes["dense_wall"]["r"], dtype=float)
        walls = z[[0, -1]]
        dense_position = shared.final_pair.primary.position.evaluate_reduced(
            walls, dense_r,
        )
        dense_bulk = shared.bulk_sampler.evaluate_wall_reduced(dense_r)
        dense_compatible = (
            shared.final_pair.primary.acceleration.evaluate_reduced(walls, dense_r)
        )
        correction = correction_profile(
            dense_position, dense_bulk, dense_compatible, dense_r,
        )
        v2_z = np.asarray(meshes["V2"]["z"], dtype=float)
        v2_r = np.asarray(meshes["V2"]["r"], dtype=float)
        bulk_axis = shared.bulk_sampler.v2_axis_reduced()
        compatible_axis = shared.final_pair.primary.acceleration.evaluate_reduced(
            v2_z, np.asarray([0.0]),
        )[:, 0]
        axis = axis_acceleration_derivative_image_profile(
            bulk_axis, compatible_axis, v2_z,
        )
        sampler_provenance = capture_protocol125_bulk_sampler_provenance(
            shared.bulk_sampler,
            parent_label=label,
            parent_identity=identity,
            correction_profile=correction,
            axis_image_profile=axis,
        )
        post_inputs = Protocol125PostAccelerationInputs(
            pre_acceleration_result=pre.pre_result,
            fixed_point_record=fixed,
            normalized_wall_profile_score=wall,
            final_representation_matrix=final_matrix,
            append_only_lineage=lineage["append_only_validation"],
            correction_profile=correction,
            axis_image_profile=axis,
            bulk_sampler_provenance=sampler_provenance,
        )
        post_provenance = capture_protocol125_postacceleration_provenance(
            post_inputs,
        )
        post_result = compose_protocol125_postacceleration_records(
            post_inputs, post_provenance,
        )
        # The refinement diagnostic owns the reduced-to-coordinate conversion.
        # Persisting coordinate components here would transform h_rr/h_0r twice.
        position_v2 = shared.final_pair.primary.position.evaluate_reduced(
            v2_z, v2_r,
        )
        hzz_zz_v2 = shared.final_pair.primary.position.evaluate_coordinate_components(
            v2_z, v2_r, z_order=2,
        )[:, :, 6]
        a_hzz_v2 = shared.final_pair.primary.acceleration.evaluate_coordinate_components(
            v2_z, v2_r,
        )[:, :, 6]
        return {
            "shared": shared,
            "wall": wall,
            "final_bundle": final_bundle,
            "final_matrix": final_matrix,
            "lineage": lineage,
            "correction": correction,
            "axis": axis,
            "sampler_provenance": sampler_provenance,
            "post_provenance": post_provenance,
            "post_result": post_result,
            "position_v2": position_v2,
            "hzz_zz_v2": hzz_zz_v2,
            "a_hzz_v2": a_hzz_v2,
        }

    def compose_post_acceleration(
        self,
        label,
        parent_stage,
        pre_acceleration_stage,
        *,
        freeze_authority,
    ):
        del freeze_authority
        label = str(label)
        parent = _source_parent(parent_stage, label)
        identity = str(parent["parent_identity"])
        pre = _source_pre(pre_acceleration_stage, label, identity)
        if not bool(pre.pre_result.get("passed", False)):
            raise Protocol125ProductionAdapterError(
                "post-acceleration adapter received a failed prerequisite"
            )
        z = np.asarray(parent["z"], dtype=float)
        r = np.asarray(parent["r"], dtype=float)
        bulk_acceleration, bulk_record = bulk_acceleration_from_completed_position(
            parent["position"], z, r, parent["background"], stencil_width=7,
        )
        try:
            compatible, fixed = solve_joint_parent_acceleration_fixed_point(
                pre.position_pair.primary,
                parent["position"],
                bulk_acceleration,
                z,
                r,
                parent["background"],
                parent_label=label,
                parent_identity=identity,
            )
        except Protocol125AccelerationScientificFailure as error:
            failure = validate_protocol125_acceleration_failure_record(
                error.record,
            )
            failure_inputs = Protocol125PostAccelerationFailureInputs(
                pre_acceleration_result=pre.pre_result,
                acceleration_failure_record=failure,
            )
            provenance = capture_protocol125_acceleration_failure_provenance(
                failure_inputs,
            )
            result = compose_protocol125_acceleration_failure_records(
                failure_inputs, provenance,
            )
            runtime = Protocol125ProductionAccelerationFailurePostState(
                _immutable(bulk_acceleration),
                bulk_record,
                failure,
                failure_inputs,
                provenance,
                result,
            )
            roots = {
                "bulk_acceleration": runtime.bulk_acceleration,
                "bulk_record": runtime.bulk_record,
                "acceleration_failure": runtime.acceleration_failure,
                "failure_provenance": runtime.failure_provenance,
                "post_result": runtime.post_result,
            }
            bindings = {label: identity}
            return Protocol125RunnerStage(
                runtime,
                _checkpoint(
                    "post-acceleration", bindings, result["groups"], roots,
                ),
                bindings,
                result["groups"],
            )
        products = self._post_products(
            parent, pre, bulk_acceleration, compatible, fixed,
        )
        runtime = Protocol125ProductionPostState(
            _immutable(bulk_acceleration), bulk_record, _immutable(compatible), fixed,
            products["shared"], products["wall"],
            products["final_bundle"].adapter_record,
            products["final_matrix"], products["lineage"],
            products["correction"], products["axis"],
            products["sampler_provenance"], products["post_provenance"],
            products["post_result"], _immutable(products["position_v2"]),
            _immutable(products["hzz_zz_v2"]), _immutable(products["a_hzz_v2"]),
        )
        roots = self._post_roots(runtime)
        bindings = {label: identity}
        return Protocol125RunnerStage(
            runtime,
            _checkpoint(
                "post-acceleration", bindings,
                runtime.post_result["groups"], roots,
            ),
            bindings,
            runtime.post_result["groups"],
        )

    @staticmethod
    def _post_roots(runtime):
        return {
            "bulk_acceleration": runtime.bulk_acceleration,
            "bulk_record": runtime.bulk_record,
            "compatible_acceleration": runtime.compatible_acceleration,
            "fixed_point_record": runtime.fixed_point_record,
            "shared_build_arrays": runtime.shared_build.coefficient_arrays("shared"),
            "wall_profile_evidence": runtime.wall_profile_evidence,
            "final_matrix_adapter_record": runtime.final_matrix_adapter_record,
            "final_matrix_result": runtime.final_matrix_result,
            "append_only_lineage": runtime.append_only_lineage,
            "correction_profile": runtime.correction_profile,
            "axis_image_profile": runtime.axis_image_profile,
            "bulk_sampler_provenance": runtime.bulk_sampler_provenance,
            "post_provenance": runtime.post_provenance,
            "post_result": runtime.post_result,
            "position_v2": runtime.position_v2,
            "hzz_zz_v2": runtime.hzz_zz_v2,
            "a_hzz_v2": runtime.a_hzz_v2,
        }

    def compose_two_parent(
        self,
        parent_stages,
        pre_acceleration_stages,
        post_acceleration_stages,
        *,
        parent_identities,
        freeze_authority,
    ):
        del freeze_authority
        if any(tuple(stages) != PARENT_LABELS for stages in (
            parent_stages, pre_acceleration_stages, post_acceleration_stages,
        )):
            raise ValueError("two-parent production stage order differs")
        identities = {label: str(parent_identities[label]) for label in PARENT_LABELS}
        if (
            not all(_valid_sha256(value) for value in identities.values())
            or identities["N0"] == identities["N1"]
        ):
            raise ValueError("two-parent production identities are invalid")
        parents = {
            label: _source_parent(parent_stages[label], label)
            for label in PARENT_LABELS
        }
        pres = {
            label: _source_pre(pre_acceleration_stages[label], label, identities[label])
            for label in PARENT_LABELS
        }
        posts = {
            label: _source_post(post_acceleration_stages[label], label, identities[label])
            for label in PARENT_LABELS
        }
        del parents
        inputs = self._two_parent_inputs(posts, pres)
        input_hashes = protocol125_two_parent_input_hashes(inputs)
        records = compose_protocol125_two_parent_records(
            inputs, parent_identities=identities,
        )
        runtime = Protocol125ProductionTwoParentState(input_hashes, records)
        roots = {"two_parent_input_hashes": input_hashes, "two_parent_records": records}
        return Protocol125RunnerStage(
            runtime,
            _checkpoint("two-parent", identities, records, roots),
            identities,
            records,
        )

    @staticmethod
    def _two_parent_inputs(posts, pres):
        frozen = frozen_validation_meshes()
        v_meshes = MappingProxyType({
            name: MappingProxyType({
                "z": _immutable(frozen[name]["z"]),
                "r": _immutable(frozen[name]["r"]),
                "sha256": str(frozen[name]["sha256"]),
            })
            for name in V_MESH_NAMES
        })
        n0 = posts["N0"]
        n1 = posts["N1"]
        return Protocol125TwoParentInputs(
            n0_position_state=n0.shared_build.final_pair.primary.position,
            n1_position_state=n1.shared_build.final_pair.primary.position,
            n0_acceleration_state=n0.shared_build.final_pair.primary.acceleration,
            n1_acceleration_state=n1.shared_build.final_pair.primary.acceleration,
            n0_native_completion_evidence=pres["N0"].native_evidence,
            n1_native_completion_evidence=pres["N1"].native_evidence,
            n0_construction_provenance=(
                pres["N0"].pre_inputs.construction_provenance
            ),
            n1_construction_provenance=(
                pres["N1"].pre_inputs.construction_provenance
            ),
            v_meshes=v_meshes,
            dense_wall_r=_immutable(frozen["dense_wall"]["r"]),
            n0_bulk_audit=pres["N0"].bulk_audit,
            n1_bulk_audit=pres["N1"].bulk_audit,
            n0_correction_profile=n0.correction_profile,
            n1_correction_profile=n1.correction_profile,
            n0_position_v2=n0.position_v2,
            n1_position_v2=n1.position_v2,
            n0_hzz_zz_v2=n0.hzz_zz_v2,
            n1_hzz_zz_v2=n1.hzz_zz_v2,
            n0_a_hzz_v2=n0.a_hzz_v2,
            n1_a_hzz_v2=n1.a_hzz_v2,
            n0_axis_image_profile=n0.axis_image_profile,
            n1_axis_image_profile=n1.axis_image_profile,
        )

    def restore_checkpoint(self, stage_id, archived, *, context):
        if not isinstance(archived, Mapping):
            raise TypeError("production checkpoint archive must be a mapping")
        metadata = archived["metadata"]
        if str(metadata.get("adapter_protocol_identifier", "")) != PROTOCOL_IDENTIFIER:
            raise Protocol125ProductionAdapterError("checkpoint adapter protocol differs")
        roots = _unpack_roots(archived["arrays"], metadata)
        kind = str(archived["envelope"]["stage_kind"])
        bindings = {
            name: str(value)
            for name, value in archived["envelope"]["bindings"].items()
        }
        if kind == "parent":
            return self._restore_parent(roots, bindings, context)
        if kind == "pre-acceleration":
            return self._restore_pre(roots, bindings, context)
        if kind == "post-acceleration":
            return self._restore_post(roots, bindings, context)
        if kind == "two-parent":
            return self._restore_two(roots, bindings, context)
        raise Protocol125ProductionAdapterError(
            f"unsupported production checkpoint stage {stage_id}:{kind}"
        )

    @staticmethod
    def _restore_parent(roots, bindings, context):
        if len(bindings) != 1 or tuple(roots) not in (
            ("parent",), ("construction_failure",),
        ):
            raise Protocol125ProductionAdapterError("parent checkpoint roots differ")
        label = next(iter(bindings))
        if tuple(roots) == ("construction_failure",):
            failure = validate_protocol125_construction_failure_record(
                roots["construction_failure"],
            )
            if (
                str(failure["parent_label"]) != label
                or str(failure["parent_identity"]) != bindings[label]
            ):
                raise Protocol125ProductionAdapterError(
                    "restored construction-failure binding differs"
                )
            if "freeze_authority" not in context:
                raise Protocol125ProductionAdapterError(
                    "restored parent lacks freeze authority context"
                )
            runtime = Protocol125ProductionParentState(None, failure)
            return Protocol125RunnerStage(
                runtime,
                _checkpoint("parent", bindings, None, roots),
                bindings,
                None,
            )
        parent = _parent_payload(roots["parent"])
        if str(parent["label"]) != label or str(parent["parent_identity"]) != bindings[label]:
            raise Protocol125ProductionAdapterError("restored parent binding differs")
        # The authority is deliberately not serialized into the scientific
        # parent payload; it is re-supplied by the validated runner context.
        if "freeze_authority" not in context:
            raise Protocol125ProductionAdapterError(
                "restored parent lacks freeze authority context"
            )
        runtime = Protocol125ProductionParentState(MappingProxyType(parent), None)
        return Protocol125RunnerStage(
            runtime,
            _checkpoint("parent", bindings, None, roots),
            bindings,
            None,
        )

    def _restore_pre(self, roots, bindings, context):
        if tuple(roots) == ("construction_failure", "pre_result"):
            if len(bindings) != 1:
                raise Protocol125ProductionAdapterError(
                    "construction-failure pre bindings differ"
                )
            label = next(iter(bindings))
            failure = _source_construction_failure(
                context.get("parent_stage"), label,
            )
            if failure is None:
                raise Protocol125ProductionAdapterError(
                    "construction-failure pre checkpoint lacks its failed parent"
                )
            _require_tree_equal(
                failure, roots["construction_failure"],
                "construction failure evidence",
            )
            result = compose_protocol125_construction_failure_records(failure)
            _require_tree_equal(
                result, roots["pre_result"], "construction failure pre result",
            )
            runtime = Protocol125ProductionConstructionFailurePreState(
                failure, result,
            )
            return Protocol125RunnerStage(
                runtime,
                _checkpoint(
                    "pre-acceleration", bindings, result["groups"], roots,
                ),
                bindings,
                result["groups"],
            )
        if tuple(roots) == (
            "representation_coefficient_failure", "pre_result",
        ):
            if len(bindings) != 1:
                raise Protocol125ProductionAdapterError(
                    "representation-failure pre bindings differ"
                )
            label = next(iter(bindings))
            parent = _source_parent(context.get("parent_stage"), label)
            if str(parent["parent_identity"]) != str(bindings[label]):
                raise Protocol125ProductionAdapterError(
                    "representation-failure parent binding differs"
                )
            evidence = validate_protocol125_representation_coefficient_failure(
                roots["representation_coefficient_failure"],
            )
            result = (
                compose_protocol125_representation_coefficient_failure_records(
                    parent,
                    parent["construction_provenance_record"],
                    evidence,
                )
            )
            _require_tree_equal(
                result,
                roots["pre_result"],
                "representation coefficient failure pre result",
            )
            runtime = Protocol125ProductionRepresentationFailurePreState(
                evidence,
                result,
            )
            return Protocol125RunnerStage(
                runtime,
                _checkpoint(
                    "pre-acceleration", bindings, result["groups"], roots,
                ),
                bindings,
                result["groups"],
            )
        common_root_names = (
            "position_pair_arrays", "position_state_record",
            "reference_pair_arrays", "native_evidence",
        )
        position_stop_roots = common_root_names + (
            "position_prefix_provenance", "position_prefix_result",
            "pre_result",
        )
        representation_stop_roots = common_root_names + (
            "legacy_inputs", "representation_prefix_provenance",
            "representation_prefix_result", "pre_result",
        )
        full_roots = common_root_names + (
            "legacy_inputs", "bulk_audit", "pre_provenance", "pre_result",
        )
        root_order = tuple(roots)
        if root_order not in (
            position_stop_roots, representation_stop_roots, full_roots,
        ) or len(bindings) != 1:
            raise Protocol125ProductionAdapterError("pre-acceleration checkpoint roots differ")
        label = next(iter(bindings))
        parent_stage = context.get("parent_stage")
        parent = _source_parent(parent_stage, label)
        outer = derive_joint_parent_position_outer_contract(parent)
        state, position_record = build_joint_parent_position_state(
            parent["position"], parent["z"], parent["r"], parent["background"],
            outer_open_face_contract=outer, parent_r_max=float(parent["r"][-1]),
        )
        pair = PositionOnlyConstrainedHermitePair.from_primary(state)
        reference = FiniteWallReferenceHermitePair.build(
            parent["z"], parent["r"], parent["reference_q"], parent["reference_phi"],
        )
        _require_tree_equal(
            pair.coefficient_arrays("position_pair"), roots["position_pair_arrays"],
            "position pair coefficients",
        )
        _require_tree_equal(
            position_record, roots["position_state_record"],
            "position state record",
        )
        _require_tree_equal(
            reference.coefficient_arrays("reference_pair"), roots["reference_pair_arrays"],
            "finite-wall reference coefficients",
        )
        core_inputs = Protocol125PreAccelerationCoreInputs(
            parent_mapping=parent,
            position_pair=pair,
            reference_pair=reference,
            construction_provenance=parent["construction_provenance_record"],
            native_position_tangent_evidence=roots["native_evidence"],
        )
        position_provenance = capture_protocol125_position_prefix_provenance(
            core_inputs,
        )
        position_prefix = evaluate_protocol125_position_prefix(
            core_inputs, position_provenance,
        )
        if root_order == position_stop_roots:
            result = finalize_protocol125_preacceleration_stop(position_prefix)
            _require_tree_equal(
                position_provenance, roots["position_prefix_provenance"],
                "position prefix provenance",
            )
            _require_tree_equal(
                position_prefix, roots["position_prefix_result"],
                "position prefix result",
            )
            _require_tree_equal(result, roots["pre_result"], "position-stop result")
            runtime = Protocol125ProductionStoppedPreState(
                pair, position_record, reference, roots["native_evidence"],
                None, position_provenance, position_prefix, result,
            )
            return Protocol125RunnerStage(
                runtime,
                _checkpoint("pre-acceleration", bindings, result["groups"], roots),
                bindings,
                result["groups"],
            )
        if not position_prefix["passed"]:
            raise Protocol125ProductionAdapterError(
                "later pre checkpoint exists after a position-prefix failure"
            )
        legacy = roots["legacy_inputs"]
        representation_inputs = Protocol125PreAccelerationRepresentationInputs(
            parent_mapping=parent,
            position_pair=pair,
            reference_pair=reference,
            construction_provenance=parent["construction_provenance_record"],
            native_position_tangent_evidence=roots["native_evidence"],
            legacy_Q33_by_mesh=legacy["legacy_Q33_by_mesh"],
            legacy_Q55_by_mesh=legacy["legacy_Q55_by_mesh"],
            legacy_component_orders=legacy["component_orders"],
        )
        representation_provenance = (
            capture_protocol125_legacy_sampling_provenance(
                representation_inputs, position_prefix,
            )
        )
        representation_prefix = extend_protocol125_legacy_sampling(
            representation_inputs, position_prefix, representation_provenance,
        )
        if root_order == representation_stop_roots:
            result = finalize_protocol125_preacceleration_stop(
                representation_prefix,
            )
            _require_tree_equal(
                representation_provenance,
                roots["representation_prefix_provenance"],
                "representation prefix provenance",
            )
            _require_tree_equal(
                representation_prefix, roots["representation_prefix_result"],
                "representation prefix result",
            )
            _require_tree_equal(
                result, roots["pre_result"], "representation-stop result",
            )
            runtime = Protocol125ProductionStoppedPreState(
                pair, position_record, reference, roots["native_evidence"],
                legacy, representation_provenance, representation_prefix,
                result,
            )
            return Protocol125RunnerStage(
                runtime,
                _checkpoint("pre-acceleration", bindings, result["groups"], roots),
                bindings,
                result["groups"],
            )
        if not representation_prefix["passed"]:
            raise Protocol125ProductionAdapterError(
                "bulk checkpoint exists after a representation-prefix failure"
            )
        inputs = Protocol125PreAccelerationInputs(
            parent_mapping=parent,
            position_pair=pair,
            reference_pair=reference,
            construction_provenance=parent["construction_provenance_record"],
            native_position_tangent_evidence=roots["native_evidence"],
            legacy_Q33_by_mesh=legacy["legacy_Q33_by_mesh"],
            legacy_Q55_by_mesh=legacy["legacy_Q55_by_mesh"],
            legacy_component_orders=legacy["component_orders"],
            bulk_validation_audit=roots["bulk_audit"],
        )
        provenance = capture_protocol125_bulk_prerequisite_provenance(
            inputs, representation_prefix,
        )
        result = finish_protocol125_bulk_prerequisite(
            inputs, representation_prefix, provenance,
        )
        _require_tree_equal(provenance, roots["pre_provenance"], "pre provenance")
        _require_tree_equal(result, roots["pre_result"], "pre result")
        runtime = Protocol125ProductionPreState(
            pair, position_record, reference, roots["native_evidence"],
            roots["bulk_audit"], legacy, inputs, provenance, result,
        )
        return Protocol125RunnerStage(
            runtime,
            _checkpoint("pre-acceleration", bindings, result["groups"], roots),
            bindings,
            result["groups"],
        )

    def _restore_post(self, roots, bindings, context):
        if tuple(roots) == (
            "bulk_acceleration", "bulk_record", "acceleration_failure",
            "failure_provenance", "post_result",
        ):
            if len(bindings) != 1:
                raise Protocol125ProductionAdapterError(
                    "acceleration-failure post bindings differ"
                )
            label = next(iter(bindings))
            parent = _source_parent(context.get("parent_stage"), label)
            pre = _source_pre(
                context.get("pre_acceleration_stage"), label, bindings[label],
            )
            recomputed_bulk, recomputed_bulk_record = (
                bulk_acceleration_from_completed_position(
                    parent["position"], parent["z"], parent["r"],
                    parent["background"], stencil_width=7,
                )
            )
            _require_tree_equal(
                recomputed_bulk, roots["bulk_acceleration"],
                "failed-stage bulk acceleration",
            )
            _require_tree_equal(
                recomputed_bulk_record, roots["bulk_record"],
                "failed-stage bulk record",
            )
            failure = validate_protocol125_acceleration_failure_record(
                roots["acceleration_failure"],
            )
            if (
                str(failure["parent_label"]) != label
                or str(failure["parent_identity"]) != bindings[label]
            ):
                raise Protocol125ProductionAdapterError(
                    "restored acceleration-failure binding differs"
                )
            failure_inputs = Protocol125PostAccelerationFailureInputs(
                pre_acceleration_result=pre.pre_result,
                acceleration_failure_record=failure,
            )
            provenance = capture_protocol125_acceleration_failure_provenance(
                failure_inputs,
            )
            result = compose_protocol125_acceleration_failure_records(
                failure_inputs, provenance,
            )
            _require_tree_equal(
                provenance, roots["failure_provenance"],
                "acceleration-failure provenance",
            )
            _require_tree_equal(
                result, roots["post_result"],
                "acceleration-failure post result",
            )
            runtime = Protocol125ProductionAccelerationFailurePostState(
                _immutable(recomputed_bulk),
                recomputed_bulk_record,
                failure,
                failure_inputs,
                provenance,
                result,
            )
            return Protocol125RunnerStage(
                runtime,
                _checkpoint(
                    "post-acceleration", bindings, result["groups"], roots,
                ),
                bindings,
                result["groups"],
            )
        expected_roots = tuple(self._post_roots_fields())
        if tuple(roots) != expected_roots or len(bindings) != 1:
            raise Protocol125ProductionAdapterError("post-acceleration checkpoint roots differ")
        label = next(iter(bindings))
        parent = _source_parent(context.get("parent_stage"), label)
        pre = _source_pre(
            context.get("pre_acceleration_stage"), label, bindings[label],
        )
        recomputed_bulk, recomputed_bulk_record = bulk_acceleration_from_completed_position(
            parent["position"], parent["z"], parent["r"], parent["background"],
            stencil_width=7,
        )
        _require_tree_equal(recomputed_bulk, roots["bulk_acceleration"], "bulk acceleration")
        _require_tree_equal(recomputed_bulk_record, roots["bulk_record"], "bulk record")
        compatible = np.asarray(roots["compatible_acceleration"])
        fixed = roots["fixed_point_record"]
        products = self._post_products(parent, pre, recomputed_bulk, compatible, fixed)
        reconstructed = Protocol125ProductionPostState(
            _immutable(recomputed_bulk), recomputed_bulk_record, _immutable(compatible), fixed,
            products["shared"], products["wall"],
            products["final_bundle"].adapter_record,
            products["final_matrix"], products["lineage"],
            products["correction"], products["axis"],
            products["sampler_provenance"], products["post_provenance"],
            products["post_result"], _immutable(products["position_v2"]),
            _immutable(products["hzz_zz_v2"]), _immutable(products["a_hzz_v2"]),
        )
        regenerated_roots = self._post_roots(reconstructed)
        for name in expected_roots:
            _require_tree_equal(regenerated_roots[name], roots[name], f"post root {name}")
        return Protocol125RunnerStage(
            reconstructed,
            _checkpoint(
                "post-acceleration", bindings,
                reconstructed.post_result["groups"], roots,
            ),
            bindings,
            reconstructed.post_result["groups"],
        )

    @staticmethod
    def _post_roots_fields():
        return (
            "bulk_acceleration", "bulk_record", "compatible_acceleration",
            "fixed_point_record", "shared_build_arrays", "wall_profile_evidence",
            "final_matrix_adapter_record", "final_matrix_result",
            "append_only_lineage", "correction_profile", "axis_image_profile",
            "bulk_sampler_provenance", "post_provenance", "post_result",
            "position_v2", "hzz_zz_v2", "a_hzz_v2",
        )

    def _restore_two(self, roots, bindings, context):
        if tuple(roots) != ("two_parent_input_hashes", "two_parent_records"):
            raise Protocol125ProductionAdapterError("two-parent checkpoint roots differ")
        if tuple(bindings) != PARENT_LABELS:
            raise Protocol125ProductionAdapterError("two-parent checkpoint bindings differ")
        post_stages = context.get("post_acceleration_stages")
        if not isinstance(post_stages, Mapping) or tuple(post_stages) != PARENT_LABELS:
            raise Protocol125ProductionAdapterError("two-parent restore context differs")
        posts = {
            label: _source_post(post_stages[label], label, bindings[label])
            for label in PARENT_LABELS
        }
        # Bind the bulk audits from their corresponding restored pre stages.
        pre_stages = context.get("pre_acceleration_stages")
        if not isinstance(pre_stages, Mapping) or tuple(pre_stages) != PARENT_LABELS:
            raise Protocol125ProductionAdapterError("two-parent pre context differs")
        pres = {
            label: _source_pre(pre_stages[label], label, bindings[label])
            for label in PARENT_LABELS
        }
        inputs = self._two_parent_inputs(posts, pres)
        hashes = protocol125_two_parent_input_hashes(inputs)
        records = compose_protocol125_two_parent_records(
            inputs, parent_identities=bindings,
        )
        _require_tree_equal(hashes, roots["two_parent_input_hashes"], "two-parent hashes")
        _require_tree_equal(records, roots["two_parent_records"], "two-parent records")
        runtime = Protocol125ProductionTwoParentState(hashes, records)
        return Protocol125RunnerStage(
            runtime,
            _checkpoint("two-parent", bindings, records, roots),
            bindings,
            records,
        )
