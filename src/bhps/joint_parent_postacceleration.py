"""Pure per-parent post-acceleration composition for Protocol 125.

The composer in this module consumes only evidence that has already been
calculated.  It does not construct a parent, form an acceleration, solve a
wall row, evaluate a representation, write an artifact, or authorize Phase A
or any evolution call.  Its five outputs are the exact per-parent records
expected by :class:`bhps.joint_parent_gate_ledger.Protocol125GateLedger`.

Numerical failures remain complete, provenance-valid failures.  Missing,
internally inconsistent, identity-mismatched, owner-violating, or changed
evidence is instead rejected as ``INVALID-audit``.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from bhps.joint_parent_acceleration import (
    ACCELERATION_FAILURE_PROTOCOL_IDENTIFIER,
    validate_protocol125_acceleration_failure_record,
)
from bhps.joint_parent_endpoint_audits import (
    ACCELERATION_ENDPOINT_CONVERSION_LANES,
    DENSE_WALL_PROFILE_DERIVATIVE_RECIPE,
    DENSE_WALL_SOURCE_RECIPE,
    SOURCE_WALL_PROFILE_DERIVATIVE_RECIPE,
    WALL_PROFILE_EVIDENCE_PROTOCOL_IDENTIFIER,
    WALL_PROFILE_INPUT_HASH_KEYS,
    WALL_PROFILE_MESHES,
    WALL_ORDER,
    WALL_PROFILE_ROWS,
    WALL_PROFILE_STAGES,
    wall_profile_evidence_fingerprint,
)
from bhps.joint_parent_final_matrix import (
    INPUT_HASH_KEYS as FINAL_MATRIX_INPUT_HASH_KEYS,
    PROTOCOL_IDENTIFIER as FINAL_MATRIX_PROTOCOL_IDENTIFIER,
    REQUIRED_FINAL_MATRIX_LANES,
)
from bhps.joint_parent_preacceleration import (
    INPUT_HASH_KEYS as PRE_ACCELERATION_INPUT_HASH_KEYS,
    PRE_ACCELERATION_GROUPS,
    PROTOCOL_IDENTIFIER as PRE_ACCELERATION_PROTOCOL_IDENTIFIER,
    _hash_tree as _preacceleration_hash_tree,
)
from bhps.joint_parent_protocol125_sampling_lineage import (
    Protocol125BulkAccelerationSampler,
)
from bhps.joint_parent_refinement_diagnostics import (
    AXIS_ACCELERATION_IMAGE_ORDER,
    AXIS_IMAGE_ORDER_ONE_LIMIT,
    AXIS_IMAGE_SMALL_LIMIT,
    COLLAR_EDGES,
    DENSE_OUTER_SHA256,
    DENSE_WALL_SHA256,
    HZZ_INDEX,
    PHYSICAL_COMPONENT_ORDER,
    frozen_validation_meshes,
)
from bhps.matched_staged_continuum import DriverConfiguration, hash_arrays


PROTOCOL_IDENTIFIER = "Protocol-125-post-acceleration-composer-v1"
BULK_SAMPLER_PROVENANCE_IDENTIFIER = (
    "Protocol-125-restricted-bulk-sampler-provenance-v1"
)
ACCELERATION_FAILURE_PROVENANCE_IDENTIFIER = (
    "Protocol-125-post-acceleration-failure-provenance-v1"
)
POST_ACCELERATION_GROUPS = (
    "acceleration_closure",
    "wall_algebra",
    "final_representation",
    "endpoint_derivatives",
    "correction_size",
)
FINAL_REPRESENTATION_LANES = (
    "Q53_Q33_position_spatial",
    "Q53_Q33_position_q4_q5_images",
    "Q53_Q33_acceleration",
    "Q53_Q33_acceleration_q4_q5_images",
    "Q53_Q33_source_triplet",
    "independent_dense_wall_position",
    "sealed_legacy_Q33_Q55_position_spatial_acceleration",
    "sealed_legacy_Q33_Q55_source_triplet",
)
ENDPOINT_DERIVATIVE_LANES = (
    "Q53_position_endpoint_z",
    "Q33_position_endpoint_z",
    "Q53_acceleration_endpoint_z",
    "Q33_acceleration_endpoint_z",
    *ACCELERATION_ENDPOINT_CONVERSION_LANES,
    "time_symmetric_velocity_endpoint_z",
    "Q53_position_outer_derivative",
    "Q33_position_outer_derivative",
    "Q53_acceleration_outer_derivative",
    "Q33_acceleration_outer_derivative",
    "independent_dense_outer_position",
)
if set(FINAL_REPRESENTATION_LANES).isdisjoint(ENDPOINT_DERIVATIVE_LANES) is False:
    raise RuntimeError("post-acceleration final-matrix lane partition overlaps")
if set(FINAL_REPRESENTATION_LANES) | set(ENDPOINT_DERIVATIVE_LANES) != set(
    REQUIRED_FINAL_MATRIX_LANES
):
    raise RuntimeError("post-acceleration final-matrix lane partition is incomplete")

SELECTIVE_FIELD_ORDER = ("h_00", "h_perp", "h_rr", "h_0r", "chi")
WALL_COMPONENT_ORDER = ("tt", "sphere", "rr", "tr")
JUNCTION_COMPONENT_ORDER = ("tt", "rr", "sphere", "tr")
ROW_DEFINED_PHYSICAL_A_Z_MASK = (
    False, False, True, True, True, True, False, False, True,
)

INPUT_HASH_KEYS = (
    "pre_acceleration_result_sha256",
    "fixed_point_record_sha256",
    "normalized_wall_profile_evidence_sha256",
    "final_representation_matrix_sha256",
    "append_only_lineage_sha256",
    "correction_profile_sha256",
    "axis_image_profile_sha256",
    "bulk_sampler_provenance_sha256",
)
ACCELERATION_FAILURE_INPUT_HASH_KEYS = (
    "pre_acceleration_result_sha256",
    "acceleration_failure_record_sha256",
)
PROVENANCE_KEYS = (
    "protocol_identifier",
    "parent_label",
    "parent_identity",
    "required_group_order",
    "input_hashes",
)
BULK_SAMPLER_PROVENANCE_KEYS = (
    "protocol_identifier",
    "parent_label",
    "parent_identity",
    "source_fingerprint",
    "sampler_fingerprint",
    "sampler_archive_sha256",
    "dense_wall_sha256",
    "dense_wall_bulk_physical_sha256",
    "V2_coordinate_sha256",
    "V2_axis_bulk_reduced_sha256",
    "correction_profile_sha256",
    "axis_image_profile_sha256",
    "wall_recipe",
    "axis_recipe",
    "wall_values_direct_from_clamped_Q3",
    "axis_values_direct_from_clamped_Q5",
    "stored_correction_interpolated",
    "compatible_contract_used_for_bulk",
)


@dataclass(frozen=True)
class Protocol125PostAccelerationInputs:
    """Explicit, already-computed evidence for one independent parent."""

    pre_acceleration_result: Mapping
    fixed_point_record: Mapping
    normalized_wall_profile_score: Mapping
    final_representation_matrix: Mapping
    append_only_lineage: Mapping
    correction_profile: Mapping
    axis_image_profile: Mapping
    bulk_sampler_provenance: Mapping


@dataclass(frozen=True)
class Protocol125PostAccelerationFailureInputs:
    """A passed pre-acceleration result and one sealed scientific stop."""

    pre_acceleration_result: Mapping
    acceleration_failure_record: Mapping


def _valid_sha256(value):
    value = str(value)
    return bool(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _digest_bytes(digest, tag, payload):
    tag = str(tag).encode("utf-8")
    payload = bytes(payload)
    digest.update(len(tag).to_bytes(8, "little"))
    digest.update(tag)
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def _update_tree_digest(digest, value, path):
    if isinstance(value, Mapping):
        if any(not isinstance(name, str) or not name for name in value):
            raise ValueError(f"hashed mapping {path} contains an invalid key")
        _digest_bytes(digest, f"mapping:{path}", str(len(value)).encode("ascii"))
        for name in sorted(value):
            _digest_bytes(digest, f"key:{path}", name.encode("utf-8"))
            _update_tree_digest(digest, value[name], f"{path}/{name}")
        return
    if isinstance(value, (tuple, list)):
        _digest_bytes(digest, f"sequence:{path}", str(len(value)).encode("ascii"))
        for index, item in enumerate(value):
            _update_tree_digest(digest, item, f"{path}/{index}")
        return
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype == object:
            raise ValueError(f"hashed array {path} has object dtype")
        if array.dtype.kind in "fc" and np.any(np.isnan(array)):
            raise ValueError(f"hashed array {path} contains NaN")
        _digest_bytes(digest, f"array-shape:{path}", repr(array.shape).encode("ascii"))
        _digest_bytes(digest, f"array-dtype:{path}", array.dtype.str.encode("ascii"))
        _digest_bytes(digest, f"array-data:{path}", array.tobytes())
        return
    if value is None:
        _digest_bytes(digest, f"none:{path}", b"")
        return
    if isinstance(value, bool):
        _digest_bytes(digest, f"bool:{path}", b"1" if value else b"0")
        return
    if isinstance(value, int):
        _digest_bytes(digest, f"int:{path}", str(value).encode("ascii"))
        return
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError(f"hashed scalar {path} is NaN")
        _digest_bytes(digest, f"float:{path}", struct.pack("!d", value))
        return
    if isinstance(value, str):
        _digest_bytes(digest, f"str:{path}", value.encode("utf-8"))
        return
    if isinstance(value, bytes):
        _digest_bytes(digest, f"bytes:{path}", value)
        return
    raise TypeError(f"unsupported hashed evidence at {path}: {type(value).__name__}")


def _fingerprint_tree(value, *, root):
    digest = hashlib.sha256()
    _update_tree_digest(digest, value, str(root))
    return digest.hexdigest()


def _immutable_array(value):
    array = np.ascontiguousarray(np.asarray(value))
    if not array.flags.writeable:
        return array
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


def _exact_keys(record, expected, label):
    if not isinstance(record, Mapping) or set(record) != set(expected):
        raise ValueError(f"{label} schema must be exactly {tuple(expected)}")


def _strict_bool(value, label):
    if type(value) is not bool:
        raise TypeError(f"{label} must be a bool")
    return value


def _number(value, label, *, allow_infinite=False, nonnegative=False):
    result = float(value)
    if math.isnan(result) or (not allow_infinite and not math.isfinite(result)):
        raise ValueError(f"{label} is not finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _same_number(left, right):
    left = float(left)
    right = float(right)
    if math.isinf(left) or math.isinf(right):
        return left == right
    return left == right


def _array(value, label, *, shape=None, finite=True, dtype=float):
    result = np.asarray(value, dtype=dtype)
    if shape is not None and result.shape != tuple(shape):
        raise ValueError(f"{label} has shape {result.shape}, expected {tuple(shape)}")
    if finite and not np.all(np.isfinite(result)):
        raise ValueError(f"{label} is nonfinite")
    if not finite and np.any(np.isnan(result)):
        raise ValueError(f"{label} contains NaN")
    return result


def _named_array_sha256(values):
    digest = hashlib.sha256()
    for name, value in values:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(name).encode())
        digest.update(b"\0")
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _basic_context(inputs):
    if not isinstance(inputs, Protocol125PostAccelerationInputs):
        raise TypeError("post-acceleration inputs require their data-only bundle")
    pre = inputs.pre_acceleration_result
    if not isinstance(pre, Mapping):
        raise TypeError("pre-acceleration result must be a mapping")
    label = str(pre.get("parent_label", ""))
    identity = str(pre.get("parent_identity", ""))
    if label not in ("N0", "N1"):
        raise ValueError("post-acceleration parent label must be N0 or N1")
    if not _valid_sha256(identity):
        raise ValueError("post-acceleration parent identity must be a SHA-256 digest")
    return MappingProxyType({"label": label, "identity": identity})


def _input_hashes(inputs):
    context = _basic_context(inputs)
    values = (
        ("pre_acceleration_result_sha256", inputs.pre_acceleration_result),
        ("fixed_point_record_sha256", inputs.fixed_point_record),
        (
            "normalized_wall_profile_evidence_sha256",
            inputs.normalized_wall_profile_score,
        ),
        ("final_representation_matrix_sha256", inputs.final_representation_matrix),
        ("append_only_lineage_sha256", inputs.append_only_lineage),
        ("correction_profile_sha256", inputs.correction_profile),
        ("axis_image_profile_sha256", inputs.axis_image_profile),
        ("bulk_sampler_provenance_sha256", inputs.bulk_sampler_provenance),
    )
    hashes = {
        name: _fingerprint_tree(value, root=name.removesuffix("_sha256"))
        for name, value in values
    }
    if tuple(hashes) != INPUT_HASH_KEYS or not all(
        _valid_sha256(value) for value in hashes.values()
    ):
        raise RuntimeError("post-acceleration input hash inventory is incomplete")
    return context, MappingProxyType(hashes)


def capture_protocol125_postacceleration_provenance(inputs):
    """Seal the complete evidence inventory before composing any gate."""
    context, hashes = _input_hashes(inputs)
    return MappingProxyType({
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "parent_label": context["label"],
        "parent_identity": context["identity"],
        "required_group_order": POST_ACCELERATION_GROUPS,
        "input_hashes": hashes,
    })


def _acceleration_failure_input_hashes(inputs):
    if not isinstance(inputs, Protocol125PostAccelerationFailureInputs):
        raise TypeError(
            "acceleration-failure composition requires its data-only bundle"
        )
    pre = inputs.pre_acceleration_result
    if not isinstance(pre, Mapping):
        raise TypeError("acceleration-failure prerequisite must be a mapping")
    label = str(pre.get("parent_label", ""))
    identity = str(pre.get("parent_identity", ""))
    if label not in ("N0", "N1") or not _valid_sha256(identity):
        raise ValueError("acceleration-failure parent binding is invalid")
    context = MappingProxyType({"label": label, "identity": identity})
    hashes = MappingProxyType({
        "pre_acceleration_result_sha256": _fingerprint_tree(
            pre, root="failure_pre_acceleration_result",
        ),
        "acceleration_failure_record_sha256": _fingerprint_tree(
            inputs.acceleration_failure_record,
            root="acceleration_failure_record",
        ),
    })
    if tuple(hashes) != ACCELERATION_FAILURE_INPUT_HASH_KEYS:
        raise RuntimeError("acceleration-failure input hash inventory differs")
    return context, hashes


def capture_protocol125_acceleration_failure_provenance(inputs):
    """Seal a passed prerequisite and scientific failure before composition."""
    context, hashes = _acceleration_failure_input_hashes(inputs)
    return MappingProxyType({
        "protocol_identifier": ACCELERATION_FAILURE_PROVENANCE_IDENTIFIER,
        "parent_label": context["label"],
        "parent_identity": context["identity"],
        "required_group_order": POST_ACCELERATION_GROUPS,
        "input_hashes": hashes,
    })


def _validate_acceleration_failure_provenance(
    provenance, context, found_hashes,
):
    _exact_keys(
        provenance,
        (
            "protocol_identifier", "parent_label", "parent_identity",
            "required_group_order", "input_hashes",
        ),
        "acceleration-failure provenance",
    )
    if (
        str(provenance["protocol_identifier"])
        != ACCELERATION_FAILURE_PROVENANCE_IDENTIFIER
        or str(provenance["parent_label"]) != context["label"]
        or str(provenance["parent_identity"]) != context["identity"]
        or tuple(provenance["required_group_order"]) != POST_ACCELERATION_GROUPS
    ):
        raise ValueError("acceleration-failure provenance binding differs")
    expected = provenance["input_hashes"]
    if (
        not isinstance(expected, Mapping)
        or tuple(expected) != ACCELERATION_FAILURE_INPUT_HASH_KEYS
    ):
        raise ValueError("acceleration-failure input hash inventory differs")
    changed = tuple(
        name for name in ACCELERATION_FAILURE_INPUT_HASH_KEYS
        if not _valid_sha256(expected[name])
        or str(expected[name]) != str(found_hashes[name])
    )
    if changed:
        raise ValueError(f"acceleration-failure input hash mismatch: {changed}")


def capture_protocol125_bulk_sampler_provenance(
    sampler,
    *,
    parent_label,
    parent_identity,
    correction_profile,
    axis_image_profile,
):
    """Bind one reconstructed restricted sampler to its two scored products."""
    if not isinstance(sampler, Protocol125BulkAccelerationSampler):
        raise TypeError("bulk-sampler provenance requires the frozen sampler type")
    label = str(parent_label)
    identity = str(parent_identity)
    if label not in ("N0", "N1") or not _valid_sha256(identity):
        raise ValueError("bulk-sampler provenance parent identity is invalid")
    frozen = frozen_validation_meshes()
    record = {
        "protocol_identifier": BULK_SAMPLER_PROVENANCE_IDENTIFIER,
        "parent_label": label,
        "parent_identity": identity,
        "source_fingerprint": str(sampler.source_fingerprint),
        "sampler_fingerprint": str(sampler.sampler_fingerprint),
        "sampler_archive_sha256": _fingerprint_tree(
            sampler.coefficient_arrays(), root="bulk_sampler_archive",
        ),
        "dense_wall_sha256": DENSE_WALL_SHA256,
        "dense_wall_bulk_physical_sha256": hash_arrays(
            sampler.dense_wall_physical(),
        ),
        "V2_coordinate_sha256": frozen["V2"]["sha256"],
        "V2_axis_bulk_reduced_sha256": hash_arrays(
            sampler.v2_axis_reduced(),
        ),
        "correction_profile_sha256": _fingerprint_tree(
            correction_profile, root="correction_profile",
        ),
        "axis_image_profile_sha256": _fingerprint_tree(
            axis_image_profile, root="axis_image_profile",
        ),
        "wall_recipe": "clamped-Q3-in-u-with-native-Du7-endpoints",
        "axis_recipe": "clamped-Q5-in-z-with-native-Dz7-endpoints",
        "wall_values_direct_from_clamped_Q3": True,
        "axis_values_direct_from_clamped_Q5": True,
        "stored_correction_interpolated": False,
        "compatible_contract_used_for_bulk": False,
    }
    return _freeze(record)


def _validate_provenance(provenance, context, found_hashes):
    _exact_keys(provenance, PROVENANCE_KEYS, "post-acceleration provenance")
    if str(provenance["protocol_identifier"]) != PROTOCOL_IDENTIFIER:
        raise ValueError("post-acceleration protocol identifier differs")
    if str(provenance["parent_label"]) != context["label"]:
        raise ValueError("post-acceleration parent label provenance differs")
    if str(provenance["parent_identity"]) != context["identity"]:
        raise ValueError("post-acceleration parent identity provenance differs")
    if tuple(provenance["required_group_order"]) != POST_ACCELERATION_GROUPS:
        raise ValueError("post-acceleration gate inventory differs")
    expected = provenance["input_hashes"]
    if not isinstance(expected, Mapping) or tuple(expected) != INPUT_HASH_KEYS:
        raise ValueError("post-acceleration input hash inventory differs")
    changed = tuple(
        name for name in INPUT_HASH_KEYS
        if not _valid_sha256(expected[name])
        or str(expected[name]) != str(found_hashes[name])
    )
    if changed:
        raise ValueError(f"post-acceleration input hash mismatch: {changed}")


def _validate_pre_acceleration(record, context):
    expected_top = (
        "protocol_identifier", "classification", "complete",
        "provenance_valid", "passed", "parent_label", "parent_identity",
        "required_group_order", "groups", "invalid_reasons",
        "input_hashes_before", "input_hashes_after",
        "inputs_stable_while_scoring", "single_parent_only",
        "second_parent_and_common_V2_still_required", "acceleration_evaluated",
        "acceleration_authorized", "scientific_execution_authorized",
        "artifact_written",
    )
    _exact_keys(record, expected_top, "pre-acceleration result")
    if not (
        str(record["protocol_identifier"]) == PRE_ACCELERATION_PROTOCOL_IDENTIFIER
        and str(record["parent_label"]) == context["label"]
        and str(record["parent_identity"]) == context["identity"]
        and str(record["classification"]) == "PASS-single-parent-pre-acceleration"
        and _strict_bool(record["complete"], "pre-acceleration complete")
        and _strict_bool(record["provenance_valid"], "pre-acceleration provenance")
        and _strict_bool(record["passed"], "pre-acceleration passed")
    ):
        raise ValueError("post-acceleration evidence lacks a successful prerequisite")
    if tuple(record["required_group_order"]) != PRE_ACCELERATION_GROUPS:
        raise ValueError("pre-acceleration group inventory differs")
    groups = record["groups"]
    if not isinstance(groups, Mapping) or tuple(groups) != PRE_ACCELERATION_GROUPS:
        raise ValueError("pre-acceleration group records are incomplete or reordered")
    expected_gate = (
        "complete", "provenance_valid", "passed", "fingerprint",
        "parent_label", "parent_identity", "group_name", "invalid_reasons",
        "details",
    )
    for name in PRE_ACCELERATION_GROUPS:
        gate = groups[name]
        _exact_keys(gate, expected_gate, f"pre-acceleration gate {name}")
        if not (
            str(gate["group_name"]) == name
            and str(gate["parent_label"]) == context["label"]
            and str(gate["parent_identity"]) == context["identity"]
            and _strict_bool(gate["complete"], f"{name} complete")
            and _strict_bool(gate["provenance_valid"], f"{name} provenance")
            and _strict_bool(gate["passed"], f"{name} passed")
            and tuple(gate["invalid_reasons"]) == ()
        ):
            raise ValueError(f"pre-acceleration gate {name} did not pass")
        payload = {
            "protocol_identifier": PRE_ACCELERATION_PROTOCOL_IDENTIFIER,
            "group_name": name,
            "parent_label": context["label"],
            "parent_identity": context["identity"],
            "complete": True,
            "provenance_valid": True,
            "passed": True,
            "invalid_reasons": (),
            "details": gate["details"],
        }
        expected_fingerprint = _preacceleration_hash_tree(
            payload, root=f"gate/{name}",
        )
        if str(gate["fingerprint"]) != expected_fingerprint:
            raise ValueError(f"pre-acceleration gate {name} fingerprint differs")
    before = record["input_hashes_before"]
    after = record["input_hashes_after"]
    if (
        not isinstance(before, Mapping)
        or not isinstance(after, Mapping)
        or tuple(before) != PRE_ACCELERATION_INPUT_HASH_KEYS
        or tuple(after) != PRE_ACCELERATION_INPUT_HASH_KEYS
        or any(not _valid_sha256(before[name]) for name in before)
        or dict(before) != dict(after)
    ):
        raise ValueError("pre-acceleration input hashes are incomplete or changed")
    required_flags = {
        "inputs_stable_while_scoring": True,
        "single_parent_only": True,
        "second_parent_and_common_V2_still_required": True,
        "acceleration_evaluated": False,
        "acceleration_authorized": False,
        "scientific_execution_authorized": False,
        "artifact_written": False,
    }
    for name, expected in required_flags.items():
        if _strict_bool(record[name], f"pre-acceleration {name}") is not expected:
            raise ValueError(f"pre-acceleration flag {name} differs")
    if tuple(record["invalid_reasons"]) != ():
        raise ValueError("successful pre-acceleration result retains invalid reasons")
    return {
        "classification": str(record["classification"]),
        "group_fingerprints": {
            name: str(groups[name]["fingerprint"]) for name in PRE_ACCELERATION_GROUPS
        },
    }


def _validate_source_triplet(record, shape):
    keys = (
        "source", "source_time", "source_second_time", "memory",
        "memory_time", "target", "advection", "raw_geometric_source",
        "normal_wall_completion", "Hdot_reassembly_scaled_Linf",
        "difference_step", "driver", "outer_source_overwrite_applied",
        "memory_carried_from_previous_iterate",
    )
    _exact_keys(record, keys, "fixed-point source triplet")
    source_shape = (shape[0], shape[1], 3)
    for name in keys[:8]:
        _array(record[name], f"source triplet {name}", shape=source_shape)
    hdot = _number(
        record["Hdot_reassembly_scaled_Linf"], "source Hdot defect",
        nonnegative=True,
    )
    if float(record["difference_step"]) != 1e-6:
        raise ValueError("source second-time difference step differs")
    driver = record["driver"]
    expected_driver = DriverConfiguration().public()
    if not isinstance(driver, Mapping) or dict(driver) != expected_driver:
        raise ValueError("source driver configuration differs")
    if (
        _strict_bool(
            record["outer_source_overwrite_applied"], "outer source overwrite flag",
        )
        or _strict_bool(
            record["memory_carried_from_previous_iterate"], "carried memory flag",
        )
    ):
        raise ValueError("fixed-point source map violated an owner prohibition")
    completion = record["normal_wall_completion"]
    _exact_keys(
        completion,
        ("normal_gauge", "ownership_pass", "changed_value_count", "finite"),
        "normal-source wall completion",
    )
    if not (
        _strict_bool(completion["ownership_pass"], "source completion ownership")
        and _strict_bool(completion["finite"], "source completion finite")
        and int(completion["changed_value_count"]) >= 0
    ):
        raise ValueError("normal-source wall completion violated ownership")
    normal = completion["normal_gauge"]
    if not isinstance(normal, Mapping) or "maximum" not in normal:
        raise ValueError("source normal-gauge audit is incomplete")
    normal_maximum = _number(
        normal["maximum"], "source normal-gauge maximum", nonnegative=True,
    )
    return {
        "Hdot_reassembly_scaled_Linf": hdot,
        "Hdot_gate": bool(hdot <= 1e-12),
        "source_normal_gauge_maximum": normal_maximum,
        "source_normal_gauge_gate": bool(normal_maximum < 1e-10),
    }


def _validate_axis_reconciliation(record):
    keys = (
        "method", "stencil_width", "parent_radius",
        "anisotropy_axis_positive_zero", "time_radial_axis_positive_zero",
        "positive_radius_reassembly_bitwise", "only_q4_q5_axis_changed_bitwise",
        "polynomial_fit_applied",
    )
    _exact_keys(record, keys, "axis reconciliation")
    if not (
        str(record["method"])
        == "native-seven-point-physical-numerator-axis-parity"
        and int(record["stencil_width"]) == 7
        and float(record["parent_radius"]) == 12.0
        and _strict_bool(record["anisotropy_axis_positive_zero"], "q4 numerator zero")
        and _strict_bool(record["time_radial_axis_positive_zero"], "q5 numerator zero")
        and _strict_bool(record["positive_radius_reassembly_bitwise"], "axis reassembly")
        and _strict_bool(record["only_q4_q5_axis_changed_bitwise"], "axis owner")
        and not _strict_bool(record["polynomial_fit_applied"], "axis fit flag")
    ):
        raise ValueError("native q4/q5 parity or ownership record differs")


def _validate_coupled_algebra(record, radial_count):
    keys = (
        "method", "maximum_allowed_condition", "minimum_allowed_pivot_strength",
        "maximum_allowed_normalized_linear_residual", "maximum_condition",
        "maximum_raw_condition", "minimum_pivot_strength",
        "worst_condition_radial_index", "worst_condition_radius",
        "weakest_pivot_radial_index", "weakest_pivot_radius",
        "maximum_normalized_linear_residual", "minimum_rank",
        "maximum_absolute_endpoint_correction", "relative_correction",
        "passed", "profiles",
    )
    _exact_keys(record, keys, "coupled Phi/g_zz algebra")
    if not (
        str(record["method"]) == "direct_radial_4x4_both_walls_Phi_gzz"
        and float(record["maximum_allowed_condition"]) == 1e12
        and float(record["minimum_allowed_pivot_strength"]) == 1e-10
        and float(record["maximum_allowed_normalized_linear_residual"]) == 1e-12
    ):
        raise ValueError("coupled algebra method or thresholds differ")
    profiles = record["profiles"]
    profile_keys = (
        "rank", "equilibrated_condition", "raw_condition", "pivot_strength",
        "normalized_linear_residual", "maximum_absolute_endpoint_correction",
    )
    _exact_keys(profiles, profile_keys, "coupled algebra profiles")
    rank = _array(profiles["rank"], "coupled ranks", shape=(radial_count,), dtype=int)
    condition = _array(
        profiles["equilibrated_condition"], "coupled conditions",
        shape=(radial_count,), finite=False,
    )
    raw_condition = _array(
        profiles["raw_condition"], "coupled raw conditions",
        shape=(radial_count,), finite=False,
    )
    pivot = _array(
        profiles["pivot_strength"], "coupled pivots", shape=(radial_count,),
    )
    residual = _array(
        profiles["normalized_linear_residual"], "coupled residuals",
        shape=(radial_count,), finite=False,
    )
    correction = _array(
        profiles["maximum_absolute_endpoint_correction"],
        "coupled endpoint corrections", shape=(radial_count,), finite=False,
    )
    if any(np.any(value < 0.0) for value in (condition, raw_condition, pivot, residual, correction)):
        raise ValueError("coupled algebra profile contains a negative diagnostic")
    summaries = {
        "minimum_rank": int(np.min(rank)),
        "maximum_condition": float(np.max(condition)),
        "maximum_raw_condition": float(np.max(raw_condition)),
        "minimum_pivot_strength": float(np.min(pivot)),
        "maximum_normalized_linear_residual": float(np.max(residual)),
        "maximum_absolute_endpoint_correction": float(np.max(correction)),
    }
    if any(not _same_number(record[name], value) for name, value in summaries.items()):
        raise ValueError("coupled algebra summaries do not reproduce their profiles")
    for name in (
        "worst_condition_radial_index", "weakest_pivot_radial_index",
    ):
        if not 0 <= int(record[name]) < radial_count:
            raise ValueError(f"coupled algebra {name} lies outside the radial grid")
    for name in (
        "worst_condition_radius", "weakest_pivot_radius", "relative_correction",
    ):
        _number(record[name], f"coupled algebra {name}", nonnegative=True)
    passed = bool(
        np.all(rank == 4)
        and np.all(condition <= 1e12)
        and np.all(pivot >= 1e-10)
        and np.all(residual < 1e-12)
    )
    if _strict_bool(record["passed"], "coupled algebra passed") is not passed:
        raise ValueError("coupled algebra pass flag does not match its profiles")
    return {
        **summaries,
        "every_radius_rank_four": bool(np.all(rank == 4)),
        "passed": passed,
    }


def _validate_selective_field_summary(record, field, radial_count):
    keys = (
        "required_rank", "maximum_allowed_condition",
        "minimum_allowed_normalized_pivot",
        "maximum_allowed_normalized_linear_residual", "minimum_rank",
        "maximum_equilibrated_condition", "maximum_raw_condition",
        "minimum_normalized_pivot", "maximum_normalized_linear_residual",
        "profiles", "passed",
    )
    _exact_keys(record, keys, f"selective field {field}")
    if not (
        int(record["required_rank"]) == 2
        and float(record["maximum_allowed_condition"]) == 1e12
        and float(record["minimum_allowed_normalized_pivot"]) == 1e-10
        and float(record["maximum_allowed_normalized_linear_residual"]) == 1e-12
    ):
        raise ValueError(f"selective field {field} thresholds differ")
    profiles = record["profiles"]
    profile_keys = (
        "rank", "equilibrated_condition", "raw_condition",
        "normalized_pivot", "normalized_linear_residual",
    )
    _exact_keys(profiles, profile_keys, f"selective field {field} profiles")
    rank = _array(profiles["rank"], f"{field} ranks", shape=(radial_count,), dtype=int)
    condition = _array(
        profiles["equilibrated_condition"], f"{field} conditions",
        shape=(radial_count,), finite=False,
    )
    raw_condition = _array(
        profiles["raw_condition"], f"{field} raw conditions",
        shape=(radial_count,), finite=False,
    )
    pivot = _array(
        profiles["normalized_pivot"], f"{field} pivots", shape=(radial_count,),
    )
    residual = _array(
        profiles["normalized_linear_residual"], f"{field} residuals",
        shape=(radial_count,), finite=False,
    )
    if any(np.any(value < 0.0) for value in (condition, raw_condition, pivot, residual)):
        raise ValueError(f"selective field {field} has a negative diagnostic")
    summaries = {
        "minimum_rank": int(np.min(rank)),
        "maximum_equilibrated_condition": float(np.max(condition)),
        "maximum_raw_condition": float(np.max(raw_condition)),
        "minimum_normalized_pivot": float(np.min(pivot)),
        "maximum_normalized_linear_residual": float(np.max(residual)),
    }
    if any(not _same_number(record[name], value) for name, value in summaries.items()):
        raise ValueError(f"selective field {field} summaries differ from profiles")
    passed = bool(
        np.all(rank == 2)
        and np.all(condition <= 1e12)
        and np.all(pivot >= 1e-10)
        and np.all(residual < 1e-12)
    )
    if _strict_bool(record["passed"], f"selective field {field} passed") is not passed:
        raise ValueError(f"selective field {field} pass flag differs")
    return {**summaries, "passed": passed}


def _validate_selective_wall_records(walls, radial_count):
    if not isinstance(walls, (tuple, list)) or len(walls) != 2:
        raise ValueError("selective wall profiles must contain both walls")
    maximum_metric = 0.0
    maximum_chi = 0.0
    for wall_index, (expected_wall, wall) in enumerate(zip(WALL_ORDER, walls)):
        keys = (
            "wall", "radial_indices", "components", "chi",
            "maximum_metric_normalized", "maximum_chi_normalized",
        )
        _exact_keys(wall, keys, f"selective {expected_wall} wall")
        if str(wall["wall"]) != expected_wall:
            raise ValueError("selective wall order differs")
        radial_indices = _array(
            wall["radial_indices"], f"{expected_wall} radial indices",
            shape=(radial_count,), dtype=int,
        )
        if not np.array_equal(radial_indices, np.arange(radial_count)):
            raise ValueError(f"{expected_wall} selective radial coverage is incomplete")
        components = wall["components"]
        if not isinstance(components, Mapping) or tuple(components) != WALL_COMPONENT_ORDER:
            raise ValueError(f"{expected_wall} selective component order differs")
        local_metric = []
        for name in WALL_COMPONENT_ORDER:
            component = components[name]
            keys = (
                "residual", "scale", "normalized", "maximum_normalized",
                "terms", "direct_physical_a_z", "row_implied_physical_a_z",
            )
            _exact_keys(component, keys, f"{expected_wall} {name} wall profile")
            residual = _array(
                component["residual"], f"{expected_wall} {name} residual",
                shape=(radial_count,),
            )
            scale = _array(
                component["scale"], f"{expected_wall} {name} scale",
                shape=(radial_count,),
            )
            normalized = _array(
                component["normalized"], f"{expected_wall} {name} normalized",
                shape=(radial_count,),
            )
            if np.any(scale < 1.0) or not np.array_equal(normalized, np.abs(residual)/scale):
                raise ValueError(f"{expected_wall} {name} wall normalization differs")
            maximum = float(np.max(normalized))
            if not _same_number(component["maximum_normalized"], maximum):
                raise ValueError(f"{expected_wall} {name} wall maximum differs")
            _array(component["terms"], f"{expected_wall} {name} terms")
            _array(
                component["direct_physical_a_z"], f"{expected_wall} {name} direct a_z",
                shape=(radial_count,),
            )
            _array(
                component["row_implied_physical_a_z"],
                f"{expected_wall} {name} row-implied a_z", shape=(radial_count,),
            )
            local_metric.append(maximum)
        chi = wall["chi"]
        chi_keys = (
            "residual", "scale", "normalized", "maximum_normalized",
            "contributions", "direct_physical_a_z", "row_implied_physical_a_z",
        )
        _exact_keys(chi, chi_keys, f"{expected_wall} chi wall profile")
        residual = _array(
            chi["residual"], f"{expected_wall} chi residual", shape=(radial_count,),
        )
        scale = _array(chi["scale"], f"{expected_wall} chi scale", shape=(radial_count,))
        normalized = _array(
            chi["normalized"], f"{expected_wall} chi normalized",
            shape=(radial_count,),
        )
        if np.any(scale < 1.0) or not np.array_equal(normalized, np.abs(residual)/scale):
            raise ValueError(f"{expected_wall} chi wall normalization differs")
        chi_maximum = float(np.max(normalized))
        if not _same_number(chi["maximum_normalized"], chi_maximum):
            raise ValueError(f"{expected_wall} chi wall maximum differs")
        _array(chi["contributions"], f"{expected_wall} chi contributions")
        _array(
            chi["direct_physical_a_z"], f"{expected_wall} chi direct a_z",
            shape=(radial_count,),
        )
        _array(
            chi["row_implied_physical_a_z"], f"{expected_wall} chi row a_z",
            shape=(radial_count,),
        )
        metric_maximum = max(local_metric)
        if not (
            _same_number(wall["maximum_metric_normalized"], metric_maximum)
            and _same_number(wall["maximum_chi_normalized"], chi_maximum)
        ):
            raise ValueError(f"{expected_wall} selective wall summary differs")
        maximum_metric = max(maximum_metric, metric_maximum)
        maximum_chi = max(maximum_chi, chi_maximum)
    return maximum_metric, maximum_chi


def _validate_selective_algebra(record, radial_count):
    keys = (
        "method", "time_symmetric", "maximum_allowed_normalized_residual",
        "maximum_metric_normalized_residual", "maximum_chi_normalized_residual",
        "minimum_tangential_endpoint_rank", "maximum_tangential_endpoint_condition",
        "weakest_tangential_endpoint_pivot", "chi_endpoint_rank",
        "per_field_algebraic_evidence", "protected_0_1_6_7_bitwise",
        "q4_q5_axis_bitwise", "walls", "direct_physical_a_z",
        "row_implied_physical_a_z", "row_defined_mask",
        "maximum_row_implied_scaled_defect", "passed",
    )
    _exact_keys(record, keys, "selective tangential/chi algebra")
    if not (
        str(record["method"]) == "direct_normalized_time_symmetric_DX2J_plus_chi"
        and _strict_bool(record["time_symmetric"], "selective time symmetry")
        and float(record["maximum_allowed_normalized_residual"]) == 1e-12
    ):
        raise ValueError("selective wall method or thresholds differ")
    if not (
        _strict_bool(record["protected_0_1_6_7_bitwise"], "selective protected owners")
        and _strict_bool(record["q4_q5_axis_bitwise"], "selective q4/q5 owner")
    ):
        raise ValueError("selective wall solve violated channel ownership")
    algebra = record["per_field_algebraic_evidence"]
    _exact_keys(
        algebra,
        ("field_order", "fields", "each_field_gated_separately",
         "chi_credited_only_with_chi_block", "passed"),
        "selective per-field evidence",
    )
    if tuple(algebra["field_order"]) != SELECTIVE_FIELD_ORDER:
        raise ValueError("selective field order differs")
    fields = algebra["fields"]
    if not isinstance(fields, Mapping) or tuple(fields) != SELECTIVE_FIELD_ORDER:
        raise ValueError("selective field evidence is incomplete or reordered")
    summaries = {
        name: _validate_selective_field_summary(fields[name], name, radial_count)
        for name in SELECTIVE_FIELD_ORDER
    }
    field_pass = all(item["passed"] for item in summaries.values())
    if not (
        _strict_bool(algebra["each_field_gated_separately"], "selective field gate flag")
        and _strict_bool(
            algebra["chi_credited_only_with_chi_block"], "selective chi owner flag",
        )
        and _strict_bool(algebra["passed"], "selective algebra summary pass") is field_pass
    ):
        raise ValueError("selective per-field summary is inconsistent")
    metric_fields = SELECTIVE_FIELD_ORDER[:4]
    expected_summary = {
        "minimum_tangential_endpoint_rank": min(
            summaries[name]["minimum_rank"] for name in metric_fields
        ),
        "maximum_tangential_endpoint_condition": max(
            summaries[name]["maximum_equilibrated_condition"] for name in metric_fields
        ),
        "weakest_tangential_endpoint_pivot": min(
            summaries[name]["minimum_normalized_pivot"] for name in metric_fields
        ),
        "chi_endpoint_rank": summaries["chi"]["minimum_rank"],
    }
    if any(not _same_number(record[name], value) for name, value in expected_summary.items()):
        raise ValueError("selective top-level algebra summaries differ")
    maximum_metric, maximum_chi = _validate_selective_wall_records(
        record["walls"], radial_count,
    )
    if not (
        _same_number(record["maximum_metric_normalized_residual"], maximum_metric)
        and _same_number(record["maximum_chi_normalized_residual"], maximum_chi)
    ):
        raise ValueError("selective wall residual summaries differ")
    direct = _array(
        record["direct_physical_a_z"], "direct physical acceleration-z",
        shape=(2, radial_count, 9),
    )
    implied = _array(
        record["row_implied_physical_a_z"], "row-implied physical acceleration-z",
        shape=(2, radial_count, 9),
    )
    mask = _array(record["row_defined_mask"], "row-defined acceleration-z mask", dtype=bool)
    if tuple(bool(value) for value in mask) != ROW_DEFINED_PHYSICAL_A_Z_MASK:
        raise ValueError("row-implied acceleration-z ownership mask differs")
    denominator = np.maximum.reduce((np.ones_like(direct), np.abs(direct), np.abs(implied)))
    defect = np.where(mask[None, None, :], np.abs(direct-implied)/denominator, 0.0)
    maximum_defect = float(np.max(defect))
    if not _same_number(record["maximum_row_implied_scaled_defect"], maximum_defect):
        raise ValueError("row-implied acceleration-z defect summary differs")
    passed = bool(
        field_pass
        and maximum_metric < 1e-10
        and maximum_chi < 1e-10
        and maximum_defect < 1e-12
    )
    if _strict_bool(record["passed"], "selective wall passed") is not passed:
        raise ValueError("selective wall pass flag differs")
    return {
        "fields": summaries,
        "maximum_metric_normalized_residual": maximum_metric,
        "maximum_chi_normalized_residual": maximum_chi,
        "maximum_row_implied_scaled_defect": maximum_defect,
        "passed": passed,
    }


def _validate_normal_gauge(record):
    _exact_keys(record, ("walls", "maximum"), "final normal-GH audit")
    walls = record["walls"]
    if not isinstance(walls, (tuple, list)) or len(walls) != 2:
        raise ValueError("final normal-GH audit omits a wall")
    maxima = []
    for expected_wall, wall in zip(WALL_ORDER, walls):
        allowed = (
            {"wall", "maximum_normalized", "maximum_absolute"},
            {"wall", "maximum_normalized", "maximum_absolute", "profiles"},
        )
        if not isinstance(wall, Mapping) or set(wall) not in allowed:
            raise ValueError(f"final normal-GH {expected_wall} wall schema differs")
        if str(wall["wall"]) != expected_wall:
            raise ValueError("final normal-GH wall order differs")
        maximum = _number(
            wall["maximum_normalized"], f"{expected_wall} normal-GH maximum",
            nonnegative=True,
        )
        _number(
            wall["maximum_absolute"], f"{expected_wall} normal-GH absolute maximum",
            nonnegative=True,
        )
        maxima.append(maximum)
    expected = max(maxima)
    if not _same_number(record["maximum"], expected):
        raise ValueError("final normal-GH maximum differs from its walls")
    return {"maximum": expected, "passed": bool(expected < 1e-10)}


def _validate_wall_second_tangent(record, radial_count, expected_wall):
    keys = (
        "wall", "orientation", "scope", "components", "metric_tensor",
        "J_tensor", "DXJ_tensor", "DJ_acceleration_tensor",
        "D2J_velocity_velocity_tensor", "DX2J_tensor",
        "decomposition_maximum_absolute_defect",
        "raw_vs_cancellation_exposed_maximum_absolute_defect",
        "separate_rows", "source", "finite",
    )
    _exact_keys(record, keys, f"{expected_wall} second-junction tangent")
    if str(record["wall"]) != expected_wall:
        raise ValueError("second-junction tangent wall identity differs")
    tensor_shape = (radial_count, 4, 4)
    tensors = {
        name: _array(record[name], f"{expected_wall} {name}", shape=tensor_shape)
        for name in (
            "metric_tensor", "J_tensor", "DXJ_tensor",
            "DJ_acceleration_tensor", "D2J_velocity_velocity_tensor",
            "DX2J_tensor",
        )
    }
    defect = float(np.max(np.abs(
        tensors["DX2J_tensor"]
        - tensors["DJ_acceleration_tensor"]
        - tensors["D2J_velocity_velocity_tensor"]
    )))
    if not _same_number(record["decomposition_maximum_absolute_defect"], defect):
        raise ValueError(f"{expected_wall} second-junction decomposition differs")
    components = record["components"]
    if not isinstance(components, Mapping) or tuple(components) != JUNCTION_COMPONENT_ORDER:
        raise ValueError(f"{expected_wall} second-junction component order differs")
    component_keys = (
        "metric", "metric_t", "metric_tt", "normal_derivative",
        "normal_derivative_t", "normal_derivative_tt", "robin_residual",
        "DX_robin_residual", "DX2_robin_residual", "J", "DXJ",
        "DJ_acceleration", "D2J_velocity_velocity", "DX2J",
        "DX2J_raw_robin_form", "second_form_maximum_absolute_defect",
    )
    raw_defects = []
    component_arrays = {}
    for name in JUNCTION_COMPONENT_ORDER:
        component = components[name]
        _exact_keys(component, component_keys, f"{expected_wall} junction component {name}")
        arrays = {
            key: _array(
                component[key], f"{expected_wall} {name} {key}",
                shape=(radial_count,),
            )
            for key in component_keys[:-1]
        }
        component_arrays[name] = arrays
        local = float(np.max(np.abs(
            arrays["DX2J"]-arrays["DX2J_raw_robin_form"]
        )))
        if not _same_number(component["second_form_maximum_absolute_defect"], local):
            raise ValueError(f"{expected_wall} {name} second-form defect differs")
        raw_defects.append(local)
    raw_defect = max(raw_defects)
    if not _same_number(
        record["raw_vs_cancellation_exposed_maximum_absolute_defect"], raw_defect,
    ):
        raise ValueError(f"{expected_wall} raw/cancellation defect differs")
    def assemble(key):
        result = np.zeros(tensor_shape)
        result[:, 0, 0] = component_arrays["tt"][key]
        result[:, 1, 1] = component_arrays["rr"][key]
        result[:, 2, 2] = component_arrays["sphere"][key]
        result[:, 3, 3] = component_arrays["sphere"][key]
        result[:, 0, 1] = result[:, 1, 0] = component_arrays["tr"][key]
        return result

    tensor_component_keys = {
        "metric_tensor": "metric",
        "J_tensor": "J",
        "DXJ_tensor": "DXJ",
        "DJ_acceleration_tensor": "DJ_acceleration",
        "D2J_velocity_velocity_tensor": "D2J_velocity_velocity",
        "DX2J_tensor": "DX2J",
    }
    tensor_reassembly_defect = max(
        float(np.max(np.abs(tensors[tensor_name]-assemble(component_key))))
        for tensor_name, component_key in tensor_component_keys.items()
    )
    separate = record["separate_rows"]
    separate_keys = (
        "Phi_robin", "DX_Phi_robin", "DJ_Phi_robin_acceleration",
        "D2_Phi_robin_velocity_velocity", "DX2_Phi_robin", "chi_neumann",
        "DX_chi_neumann", "DJ_chi_neumann_acceleration",
        "D2_chi_neumann_velocity_velocity", "DX2_chi_neumann",
    )
    _exact_keys(separate, separate_keys, f"{expected_wall} separate wall rows")
    separate_arrays = {
        name: _array(
            separate[name], f"{expected_wall} separate row {name}",
            shape=(radial_count,),
        )
        for name in separate_keys
    }
    phi_reassembly = float(np.max(np.abs(
        separate_arrays["DX2_Phi_robin"]
        - separate_arrays["DJ_Phi_robin_acceleration"]
        - separate_arrays["D2_Phi_robin_velocity_velocity"]
    )))
    chi_reassembly = float(np.max(np.abs(
        separate_arrays["DX2_chi_neumann"]
        - separate_arrays["DJ_chi_neumann_acceleration"]
        - separate_arrays["D2_chi_neumann_velocity_velocity"]
    )))
    source = record["source"]
    source_keys = (
        "beta", "beta_phi", "beta_phiphi", "beta_t", "beta_tt",
        "sqrt_gzz", "sqrt_gzz_t", "sqrt_gzz_tt",
    )
    _exact_keys(source, source_keys, f"{expected_wall} wall source")
    source_arrays = {
        name: _array(
            source[name],
            f"{expected_wall} source {name}",
            shape=(() if name == "beta_phiphi" else (radial_count,)),
        )
        for name in source_keys
    }
    if np.any(source_arrays["sqrt_gzz"] <= 0.0):
        raise ValueError(f"{expected_wall} wall source has nonpositive sqrt(g_zz)")
    finite = bool(
        all(np.all(np.isfinite(value)) for value in tensors.values())
        and all(np.all(np.isfinite(value)) for value in separate_arrays.values())
        and all(np.all(np.isfinite(value)) for value in source_arrays.values())
    )
    if _strict_bool(record["finite"], f"{expected_wall} junction finite") is not finite:
        raise ValueError(f"{expected_wall} junction finite flag differs")
    return {
        "decomposition_defect": defect,
        "raw_vs_cancellation_defect": raw_defect,
        "tensor_component_reassembly_defect": tensor_reassembly_defect,
        "Phi_row_reassembly_defect": phi_reassembly,
        "chi_row_reassembly_defect": chi_reassembly,
        "passed": bool(
            finite
            and defect < 1e-12
            and raw_defect < 1e-12
            and tensor_reassembly_defect < 1e-12
            and phi_reassembly < 1e-12
            and chi_reassembly < 1e-12
        ),
    }


def _validate_fixed_point(record):
    keys = (
        "method", "history", "maps_used", "consecutive_converged_maps",
        "source_triplet", "coupled", "selective", "axis_reconciliation",
        "normal_gauge", "wall_second_tangent", "delta_acceleration",
        "outer_overwrite_applied", "generic_axis_fill_applied",
        "endpoint_history_carried",
    )
    _exact_keys(record, keys, "fixed-point record")
    if str(record["method"]) != "Protocol-125-full-update-eight-map-fixed-point":
        raise ValueError("fixed-point method differs")
    delta = _array(record["delta_acceleration"], "fixed-point delta acceleration")
    if delta.ndim != 3 or delta.shape[-1] != 9 or min(delta.shape[:2]) < 2:
        raise ValueError("fixed-point delta acceleration has the wrong shape")
    shape = delta.shape
    for name in (
        "outer_overwrite_applied", "generic_axis_fill_applied",
        "endpoint_history_carried",
    ):
        if _strict_bool(record[name], f"fixed-point {name}"):
            raise ValueError(f"fixed-point owner prohibition failed: {name}")
    history = record["history"]
    maps_used = int(record["maps_used"])
    if not isinstance(history, (tuple, list)) or not 1 <= len(history) <= 8:
        raise ValueError("fixed-point history must contain one through eight maps")
    if maps_used != len(history):
        raise ValueError("fixed-point map count differs from history")
    history_keys = (
        "map", "acceleration_scaled_Linf_change",
        "source_triplet_scaled_Linf_change", "consecutive_converged_maps",
        "coupled", "selective", "normal_tangential_correction_scaled_Linf",
        "axis_reconciliation_scaled_Linf", "axis_reconciliation",
    )
    consecutive = 0
    for map_index, item in enumerate(history, start=1):
        _exact_keys(item, history_keys, f"fixed-point map {map_index}")
        if int(item["map"]) != map_index:
            raise ValueError("fixed-point map numbers are not consecutive")
        acceleration_change = _number(
            item["acceleration_scaled_Linf_change"], "fixed-point acceleration change",
            nonnegative=True,
        )
        source_change = _number(
            item["source_triplet_scaled_Linf_change"], "fixed-point source change",
            nonnegative=True,
        )
        _number(
            item["normal_tangential_correction_scaled_Linf"],
            "normal-tangential correction", nonnegative=True,
        )
        _number(
            item["axis_reconciliation_scaled_Linf"], "axis reconciliation change",
            nonnegative=True,
        )
        consecutive = consecutive+1 if (
            acceleration_change < 1e-12 and source_change < 1e-12
        ) else 0
        if int(item["consecutive_converged_maps"]) != consecutive:
            raise ValueError("fixed-point consecutive-map counter is inconsistent")
        _validate_axis_reconciliation(item["axis_reconciliation"])
        selective = item["selective"]
        if not isinstance(selective, Mapping) or not (
            _strict_bool(selective.get("protected_0_1_6_7_bitwise"), "map selective owner")
            and _strict_bool(selective.get("q4_q5_axis_bitwise"), "map q4/q5 owner")
        ):
            raise ValueError("a fixed-point map violated selective ownership")
        if consecutive >= 2 and map_index != len(history):
            raise ValueError("fixed-point history continued after two-map convergence")
    if int(record["consecutive_converged_maps"]) != consecutive:
        raise ValueError("fixed-point final consecutive-map count differs")
    converged = bool(consecutive >= 2)
    if not converged and maps_used != 8:
        raise ValueError("unconverged fixed-point evidence stopped before eight maps")
    if _fingerprint_tree(record["coupled"], root="coupled") != _fingerprint_tree(
        history[-1]["coupled"], root="coupled",
    ):
        raise ValueError("final coupled algebra differs from the final map")
    if _fingerprint_tree(record["selective"], root="selective") != _fingerprint_tree(
        history[-1]["selective"], root="selective",
    ):
        raise ValueError("final selective algebra differs from the final map")
    if _fingerprint_tree(
        record["axis_reconciliation"], root="axis_reconciliation",
    ) != _fingerprint_tree(
        history[-1]["axis_reconciliation"], root="axis_reconciliation",
    ):
        raise ValueError("final axis reconciliation differs from the final map")
    _validate_axis_reconciliation(record["axis_reconciliation"])
    source = _validate_source_triplet(record["source_triplet"], shape)
    coupled = _validate_coupled_algebra(record["coupled"], shape[1])
    selective = _validate_selective_algebra(record["selective"], shape[1])
    normal = _validate_normal_gauge(record["normal_gauge"])
    wall = record["wall_second_tangent"]
    if not isinstance(wall, Mapping) or tuple(wall) != WALL_ORDER:
        raise ValueError("fixed-point second-junction walls are incomplete or reordered")
    tangents = {
        name: _validate_wall_second_tangent(wall[name], shape[1], name)
        for name in WALL_ORDER
    }
    return {
        "maps_used": maps_used,
        "consecutive_converged_maps": consecutive,
        "two_consecutive_map_convergence": converged,
        "source": source,
        "coupled": coupled,
        "selective": selective,
        "normal_gauge": normal,
        "wall_second_tangent": tangents,
        "acceleration_closure_pass": bool(
            converged
            and source["Hdot_gate"]
            and source["source_normal_gauge_gate"]
            and normal["passed"]
        ),
        "wall_algebra_pass": bool(
            coupled["passed"]
            and selective["passed"]
            and normal["passed"]
            and all(item["passed"] for item in tangents.values())
        ),
    }


def _validate_normalized_wall_score(
    record, *, expected_radial_count, evidence_label,
):
    keys = (
        "wall_order", "row_order", "records", "gates", "passed",
        "fingerprint", "gate",
    )
    _exact_keys(record, keys, f"normalized {evidence_label} wall profile score")
    if tuple(record["wall_order"]) != WALL_ORDER or tuple(record["row_order"]) != WALL_PROFILE_ROWS:
        raise ValueError("normalized wall profile row or wall order differs")
    records = record["records"]
    if not isinstance(records, Mapping) or tuple(records) != ("position", "acceleration"):
        raise ValueError("normalized wall profile stages differ")
    expected_gates = {}
    fingerprint_items = []
    radial_count = None
    row_keys = (
        "signed_terms", "signed_residual", "positive_scale", "signed_profile",
        "absolute_profile", "wall_Linf", "wall_pass",
    )
    for stage in ("position", "acceleration"):
        stage_record = records[stage]
        if not isinstance(stage_record, Mapping) or tuple(stage_record) != WALL_PROFILE_ROWS:
            raise ValueError(f"normalized wall {stage} rows differ")
        for row in WALL_PROFILE_ROWS:
            row_record = stage_record[row]
            _exact_keys(row_record, row_keys, f"normalized wall {stage} {row}")
            terms = _array(row_record["signed_terms"], f"{stage} {row} terms")
            if terms.ndim != 3 or terms.shape[1] != 2:
                raise ValueError(f"normalized wall {stage} {row} terms have wrong shape")
            if radial_count is None:
                radial_count = terms.shape[2]
            if terms.shape[2] != radial_count:
                raise ValueError("normalized wall rows use different radial meshes")
            residual = _array(
                row_record["signed_residual"], f"{stage} {row} residual",
                shape=(2, radial_count),
            )
            scale = _array(
                row_record["positive_scale"], f"{stage} {row} scale",
                shape=(2, radial_count),
            )
            profile = _array(
                row_record["signed_profile"], f"{stage} {row} profile",
                shape=(2, radial_count),
            )
            absolute = _array(
                row_record["absolute_profile"], f"{stage} {row} absolute profile",
                shape=(2, radial_count),
            )
            if not (
                np.array_equal(residual, np.sum(terms, axis=0))
                and np.all(scale >= 1.0)
                and np.array_equal(profile, residual/scale)
                and np.array_equal(absolute, np.abs(profile))
            ):
                raise ValueError(f"normalized wall {stage} {row} algebra differs")
            maxima = tuple(float(np.max(absolute[index])) for index in range(2))
            passes = tuple(maximum < 1e-10 for maximum in maxima)
            if tuple(row_record["wall_Linf"]) != maxima or tuple(row_record["wall_pass"]) != passes:
                raise ValueError(f"normalized wall {stage} {row} summary differs")
            for wall_index, wall in enumerate(WALL_ORDER):
                expected_gates[f"{stage}_{row}_{wall}"] = passes[wall_index]
            fingerprint_items.extend((
                (f"{stage}_{row}_terms", terms),
                (f"{stage}_{row}_scale", scale),
                (f"{stage}_{row}_profile", profile),
            ))
    if radial_count != int(expected_radial_count):
        raise ValueError(
            f"normalized {evidence_label} wall score has the wrong radial count"
        )
    gates = record["gates"]
    if not isinstance(gates, Mapping) or tuple(gates) != tuple(expected_gates):
        raise ValueError("normalized wall gate inventory differs")
    for name, expected in expected_gates.items():
        if _strict_bool(gates[name], f"normalized wall gate {name}") is not expected:
            raise ValueError(f"normalized wall gate {name} differs from its profile")
    passed = all(expected_gates.values())
    if _strict_bool(record["passed"], "normalized wall passed") is not passed:
        raise ValueError("normalized wall pass flag differs")
    if str(record["fingerprint"]) != _named_array_sha256(fingerprint_items):
        raise ValueError("normalized wall profile fingerprint differs")
    if str(record["gate"]) != "each named row and wall has Linf < 1e-10":
        raise ValueError("normalized wall profile gate description differs")
    return {
        "radial_point_count": radial_count,
        "maximum_by_gate": {
            f"{stage}_{row}_{wall}": float(
                records[stage][row]["wall_Linf"][wall_index]
            )
            for stage in ("position", "acceleration")
            for row in WALL_PROFILE_ROWS
            for wall_index, wall in enumerate(WALL_ORDER)
        },
        "passed": passed,
        "fingerprint": str(record["fingerprint"]),
    }


def _validate_normalized_wall_evidence(record, context):
    keys = (
        "protocol_identifier", "parent_label", "parent_identity",
        "source_fingerprint", "endpoint_fingerprint", "wall_order",
        "row_order", "mesh_order", "coordinates", "derivative_recipes",
        "input_hashes", "time_symmetry", "live_compact_context", "meshes",
        "named_gate_order", "named_row_wall_gates",
        "named_row_wall_passed", "constituent_logical_AND", "complete",
        "provenance_valid", "passed", "scientific_execution_authorized",
        "artifact_written", "fingerprint",
    )
    _exact_keys(record, keys, "source+dense normalized wall-profile evidence")
    if str(record["protocol_identifier"]) != WALL_PROFILE_EVIDENCE_PROTOCOL_IDENTIFIER:
        raise ValueError("normalized wall-profile evidence protocol differs")
    if (
        str(record["parent_label"]) != context["label"]
        or str(record["parent_identity"]) != context["identity"]
    ):
        raise ValueError("normalized wall-profile evidence parent identity differs")
    if not all(
        _valid_sha256(record[name])
        for name in ("source_fingerprint", "endpoint_fingerprint", "fingerprint")
    ):
        raise ValueError("normalized wall-profile evidence fingerprint is invalid")
    if (
        tuple(record["wall_order"]) != WALL_ORDER
        or tuple(record["row_order"]) != WALL_PROFILE_ROWS
        or tuple(record["mesh_order"]) != WALL_PROFILE_MESHES
    ):
        raise ValueError("normalized wall-profile evidence ordering differs")
    for name, expected in (
        ("complete", True),
        ("provenance_valid", True),
        ("constituent_logical_AND", True),
        ("scientific_execution_authorized", False),
        ("artifact_written", False),
    ):
        if _strict_bool(record[name], f"normalized wall-profile {name}") is not expected:
            raise ValueError(f"normalized wall-profile {name} differs")

    coordinate_keys = (
        "source_z", "source_r", "dense_r", "source_z_sha256",
        "source_r_sha256", "source_pair_sha256",
        "source_wall_coordinate_sha256", "dense_r_sha256",
        "dense_wall_coordinate_sha256",
    )
    coordinates = record["coordinates"]
    _exact_keys(coordinates, coordinate_keys, "normalized wall-profile coordinates")
    source_z = _array(coordinates["source_z"], "wall-profile source z")
    source_r = _array(coordinates["source_r"], "wall-profile source r")
    dense_r = _array(coordinates["dense_r"], "wall-profile dense r")
    frozen_dense = frozen_validation_meshes()["dense_wall"]["r"]
    if (
        source_z.ndim != 1
        or source_r.ndim != 1
        or len(source_z) < 7
        or len(source_r) < 7
        or np.any(np.diff(source_z) <= 0.0)
        or np.any(np.diff(source_r) <= 0.0)
        or source_r[0] != 0.0
        or np.signbit(source_r[0])
        or source_z[0] != 1.0
        or source_z[-1] != np.e
        or source_r[-1] != 12.0
        or not np.array_equal(dense_r, frozen_dense)
    ):
        raise ValueError("normalized wall-profile coordinates are not Protocol-125 meshes")
    walls = source_z[[0, -1]]
    coordinate_hashes = {
        "source_z_sha256": hash_arrays(source_z),
        "source_r_sha256": hash_arrays(source_r),
        "source_pair_sha256": hash_arrays(source_z, source_r),
        "source_wall_coordinate_sha256": hash_arrays(walls, source_r),
        "dense_r_sha256": hash_arrays(dense_r),
        "dense_wall_coordinate_sha256": hash_arrays(walls, dense_r),
    }
    if any(str(coordinates[name]) != value for name, value in coordinate_hashes.items()):
        raise ValueError("normalized wall-profile coordinate hash differs")
    if coordinate_hashes["dense_r_sha256"] != DENSE_WALL_SHA256:
        raise ValueError("normalized wall-profile dense mesh hash is not frozen")

    recipe_keys = (
        "source_recipe", "source_stencil_width", "dense_recipe",
        "dense_source_recipe",
    )
    recipes = record["derivative_recipes"]
    _exact_keys(recipes, recipe_keys, "normalized wall-profile derivative recipes")
    if (
        str(recipes["source_recipe"]) != SOURCE_WALL_PROFILE_DERIVATIVE_RECIPE
        or int(recipes["source_stencil_width"]) != 7
        or str(recipes["dense_recipe"]) != DENSE_WALL_PROFILE_DERIVATIVE_RECIPE
        or str(recipes["dense_source_recipe"]) != DENSE_WALL_SOURCE_RECIPE
    ):
        raise ValueError("normalized wall-profile derivative recipe differs")

    inputs = record["input_hashes"]
    _exact_keys(inputs, WALL_PROFILE_INPUT_HASH_KEYS, "normalized wall-profile inputs")
    if not all(_valid_sha256(inputs[name]) for name in WALL_PROFILE_INPUT_HASH_KEYS):
        raise ValueError("normalized wall-profile input hash is invalid")

    symmetry_keys = (
        "source_velocity_shape", "source_velocity_sha256",
        "source_positive_zero_reference_sha256",
        "source_bitwise_positive_zero", "dense_velocity_shape",
        "dense_velocity_sha256", "dense_positive_zero_reference_sha256",
        "dense_bitwise_positive_zero", "dense_velocity_recipe", "passed",
    )
    symmetry = record["time_symmetry"]
    _exact_keys(symmetry, symmetry_keys, "normalized wall-profile time symmetry")
    source_shape = (len(source_z), len(source_r), 9)
    dense_shape = (2, len(dense_r), 9)
    if tuple(symmetry["source_velocity_shape"]) != source_shape or tuple(
        symmetry["dense_velocity_shape"]
    ) != dense_shape:
        raise ValueError("normalized wall-profile velocity shape differs")
    source_reference = hash_arrays(np.zeros(source_shape, dtype=float))
    dense_reference = hash_arrays(np.zeros(dense_shape, dtype=float))
    if (
        str(symmetry["source_positive_zero_reference_sha256"]) != source_reference
        or str(symmetry["dense_positive_zero_reference_sha256"]) != dense_reference
        or str(symmetry["dense_velocity_recipe"])
        != "exact-positive-zero-time-symmetric-extension"
        or str(inputs["completed_velocity_sha256"])
        != str(symmetry["source_velocity_sha256"])
    ):
        raise ValueError("normalized wall-profile time-symmetry provenance differs")
    source_positive = str(symmetry["source_velocity_sha256"]) == source_reference
    dense_positive = str(symmetry["dense_velocity_sha256"]) == dense_reference
    if (
        _strict_bool(
            symmetry["source_bitwise_positive_zero"],
            "source wall-profile positive zero",
        ) is not source_positive
        or _strict_bool(
            symmetry["dense_bitwise_positive_zero"],
            "dense wall-profile positive zero",
        ) is not dense_positive
    ):
        raise ValueError("normalized wall-profile positive-zero flag differs")
    symmetry_pass = bool(source_positive and dense_positive)
    if _strict_bool(symmetry["passed"], "wall-profile time symmetry passed") is not symmetry_pass:
        raise ValueError("normalized wall-profile time-symmetry pass flag differs")

    live_keys = (
        "contract_identifier", "contract_fingerprint",
        "position_and_acceleration_share_live_contract",
        "source_normal_context_present", "source_second_normal_context_present",
        "source_position_reproduction_scaled_Linf",
        "source_acceleration_reproduction_scaled_Linf",
        "source_normal_context_reproduction_scaled_Linf",
        "source_second_normal_context_reproduction_scaled_Linf",
        "dense_source_normal_sha256", "dense_source_second_normal_sha256",
        "passed",
    )
    live = record["live_compact_context"]
    _exact_keys(live, live_keys, "normalized wall-profile live compact context")
    if not str(live["contract_identifier"]) or not _valid_sha256(
        live["contract_fingerprint"]
    ) or not all(_valid_sha256(live[name]) for name in (
        "dense_source_normal_sha256", "dense_source_second_normal_sha256",
    )):
        raise ValueError("normalized wall-profile compact-context provenance is invalid")
    live_booleans = tuple(
        _strict_bool(live[name], f"normalized wall-profile {name}")
        for name in (
            "position_and_acceleration_share_live_contract",
            "source_normal_context_present", "source_second_normal_context_present",
        )
    )
    live_defects = tuple(
        _number(live[name], f"normalized wall-profile {name}", nonnegative=True)
        for name in (
            "source_position_reproduction_scaled_Linf",
            "source_acceleration_reproduction_scaled_Linf",
            "source_normal_context_reproduction_scaled_Linf",
            "source_second_normal_context_reproduction_scaled_Linf",
        )
    )
    live_pass = bool(all(live_booleans) and all(value <= 1e-12 for value in live_defects))
    if _strict_bool(live["passed"], "wall-profile live context passed") is not live_pass:
        raise ValueError("normalized wall-profile live-context pass flag differs")

    meshes = record["meshes"]
    if not isinstance(meshes, Mapping) or tuple(meshes) != WALL_PROFILE_MESHES:
        raise ValueError("normalized wall-profile mesh inventory differs")
    validated_meshes = {
        "source": _validate_normalized_wall_score(
            meshes["source"],
            expected_radial_count=len(source_r),
            evidence_label="source",
        ),
        "dense": _validate_normalized_wall_score(
            meshes["dense"],
            expected_radial_count=len(dense_r),
            evidence_label="dense",
        ),
    }
    expected_gate_order = tuple(
        f"{mesh}_{stage}_{row}_{wall}"
        for mesh in WALL_PROFILE_MESHES
        for stage in WALL_PROFILE_STAGES
        for row in WALL_PROFILE_ROWS
        for wall in WALL_ORDER
    )
    if tuple(record["named_gate_order"]) != expected_gate_order:
        raise ValueError("normalized wall-profile named gate order differs")
    named = record["named_row_wall_gates"]
    if not isinstance(named, Mapping) or tuple(named) != expected_gate_order:
        raise ValueError("normalized wall-profile named gate inventory differs")
    expected_named = {
        f"{mesh}_{name}": bool(meshes[mesh]["gates"][name])
        for mesh in WALL_PROFILE_MESHES
        for name in meshes[mesh]["gates"]
    }
    for name, expected in expected_named.items():
        if _strict_bool(named[name], f"normalized wall-profile gate {name}") is not expected:
            raise ValueError(f"normalized wall-profile gate {name} differs")
    named_pass = bool(all(expected_named.values()))
    if _strict_bool(
        record["named_row_wall_passed"], "normalized wall-profile named pass",
    ) is not named_pass:
        raise ValueError("normalized wall-profile named-row pass flag differs")
    passed = bool(named_pass and symmetry_pass and live_pass)
    if _strict_bool(record["passed"], "normalized wall-profile evidence passed") is not passed:
        raise ValueError("normalized wall-profile evidence pass flag differs")
    if str(record["fingerprint"]) != wall_profile_evidence_fingerprint(record):
        raise ValueError("normalized wall-profile evidence fingerprint differs")
    return {
        "source": validated_meshes["source"],
        "dense": validated_meshes["dense"],
        "source_fingerprint": str(record["source_fingerprint"]),
        "endpoint_fingerprint": str(record["endpoint_fingerprint"]),
        "coordinate_hashes": coordinate_hashes,
        "derivative_recipes": dict(recipes),
        "time_symmetry_passed": symmetry_pass,
        "live_compact_context_passed": live_pass,
        "named_row_wall_passed": named_pass,
        "named_row_wall_gate_count": len(expected_named),
        "passed": passed,
        "fingerprint": str(record["fingerprint"]),
    }


def _validate_final_matrix(record, context):
    keys = (
        "protocol_identifier", "parent_label", "classification", "complete",
        "provenance_valid", "passed", "required_lane_order", "lanes",
        "failed_lanes", "invalid_reasons", "source_fingerprint",
        "endpoint_fingerprint", "input_hashes_before", "input_hashes_after",
        "inputs_stable_while_scoring", "constituent_logical_AND",
        "phase_a_authorized", "scientific_execution_authorized",
        "artifact_written",
    )
    _exact_keys(record, keys, "final representation matrix result")
    if not (
        str(record["protocol_identifier"]) == FINAL_MATRIX_PROTOCOL_IDENTIFIER
        and str(record["parent_label"]) == context["label"]
        and _strict_bool(record["complete"], "final matrix complete")
        and _strict_bool(record["provenance_valid"], "final matrix provenance")
    ):
        raise ValueError("final representation matrix is incomplete or mislabeled")
    if tuple(record["required_lane_order"]) != REQUIRED_FINAL_MATRIX_LANES:
        raise ValueError("final representation matrix lane inventory differs")
    lanes = record["lanes"]
    if not isinstance(lanes, Mapping) or tuple(lanes) != REQUIRED_FINAL_MATRIX_LANES:
        raise ValueError("final representation matrix lanes are incomplete or reordered")
    lane_pass = {}
    for name in REQUIRED_FINAL_MATRIX_LANES:
        lane = lanes[name]
        if not isinstance(lane, Mapping) or "passed" not in lane:
            raise ValueError(f"final representation lane {name} lacks a pass result")
        lane_pass[name] = _strict_bool(lane["passed"], f"final matrix lane {name}")
    failed = tuple(name for name in REQUIRED_FINAL_MATRIX_LANES if not lane_pass[name])
    passed = not failed
    if not (
        _strict_bool(record["passed"], "final matrix passed") is passed
        and tuple(record["failed_lanes"]) == failed
        and str(record["classification"]) == (
            "PASS-final-representation-matrix" if passed
            else "FAIL-final-representation-matrix"
        )
        and tuple(record["invalid_reasons"]) == ()
    ):
        raise ValueError("final representation matrix classification is inconsistent")
    before = record["input_hashes_before"]
    after = record["input_hashes_after"]
    if (
        not isinstance(before, Mapping)
        or not isinstance(after, Mapping)
        or tuple(before) != FINAL_MATRIX_INPUT_HASH_KEYS
        or tuple(after) != FINAL_MATRIX_INPUT_HASH_KEYS
        or any(not _valid_sha256(before[name]) for name in before)
        or dict(before) != dict(after)
    ):
        raise ValueError("final representation matrix inputs changed or are incomplete")
    for name in ("source_fingerprint", "endpoint_fingerprint"):
        if not _valid_sha256(record[name]):
            raise ValueError(f"final representation matrix {name} is invalid")
    required_flags = {
        "inputs_stable_while_scoring": True,
        "constituent_logical_AND": True,
        "phase_a_authorized": False,
        "scientific_execution_authorized": False,
        "artifact_written": False,
    }
    for name, expected in required_flags.items():
        if _strict_bool(record[name], f"final matrix {name}") is not expected:
            raise ValueError(f"final representation matrix flag {name} differs")
    expected_endpoint_states = {
        "Q53_position_endpoint_z": "position",
        "Q33_position_endpoint_z": "position",
        "Q53_acceleration_endpoint_z": "acceleration",
        "Q33_acceleration_endpoint_z": "acceleration",
    }
    for lane_name, state_name in expected_endpoint_states.items():
        lane = lanes[lane_name]
        if (
            str(lane.get("state_name", "")) != state_name
            or str(lane.get("dense_r_sha256", "")) != DENSE_WALL_SHA256
        ):
            raise ValueError(f"final endpoint lane {lane_name} provenance differs")
    velocity = lanes["time_symmetric_velocity_endpoint_z"]
    if not (
        _strict_bool(velocity.get("bitwise_positive_zero"), "velocity endpoint +0")
        and _valid_sha256(velocity.get("fingerprint", ""))
    ):
        raise ValueError("time-symmetric velocity endpoint-z provenance differs")
    for lane_name in (
        "Q53_position_outer_derivative", "Q33_position_outer_derivative",
        "Q53_acceleration_outer_derivative", "Q33_acceleration_outer_derivative",
    ):
        lane = lanes[lane_name]
        if not (
            _strict_bool(lane.get("complete"), f"{lane_name} complete")
            and _strict_bool(lane.get("provenance_valid"), f"{lane_name} provenance")
            and _strict_bool(
                lane.get("compact_endpoints_excluded_from_score"),
                f"{lane_name} endpoint ownership",
            )
            and not _strict_bool(
                lane.get("contract_target_query_called"), f"{lane_name} target query",
            )
            and _strict_bool(
                lane.get("fresh_degree_five_target_reconstruction"),
                f"{lane_name} fresh target",
            )
            and str(lane.get("dense_outer_sha256", "")) == DENSE_OUTER_SHA256
            and _valid_sha256(lane.get("fingerprint", ""))
        ):
            raise ValueError(f"final outer derivative lane {lane_name} is not independent")
    dense_wall = lanes["independent_dense_wall_position"]
    if _strict_bool(
        dense_wall.get("stored_residual_or_target_array_comparison_used"),
        "dense-wall stored comparison flag",
    ):
        raise ValueError("final dense-wall position audit used a stored comparator")
    dense_outer = lanes["independent_dense_outer_position"]
    if _strict_bool(
        dense_outer.get("contract_target_query_called"),
        "dense-outer target-query flag",
    ):
        raise ValueError("final dense-outer position audit queried its contract target")
    return {
        "classification": str(record["classification"]),
        "lane_pass": lane_pass,
        "failed_lanes": failed,
        "source_fingerprint": str(record["source_fingerprint"]),
        "endpoint_fingerprint": str(record["endpoint_fingerprint"]),
        "passed": passed,
    }


def _validate_append_only_lineage(record):
    keys = ("passed", "gates", "position_payload_hash", "position_only", "shared")
    _exact_keys(record, keys, "append-only position lineage")
    gate_order = (
        "payload_hash_invariant", "payload_names_invariant",
        "payload_arrays_bitwise", "direct_parent_payload",
        "direct_parent_compact", "direct_parent_outer", "direct_parent_archive",
        "compact_identifier_evolved", "compact_fingerprint_evolved",
        "outer_identifier_evolved", "outer_fingerprint_evolved",
        "archive_fingerprint_evolved", "appended_child_present",
    )
    gates = record["gates"]
    if not isinstance(gates, Mapping) or tuple(gates) != gate_order:
        raise ValueError("append-only lineage gate inventory differs")
    if not all(_strict_bool(gates[name], f"lineage gate {name}") for name in gate_order):
        raise ValueError("position/shared transition is not append-only")
    if not _strict_bool(record["passed"], "append-only lineage passed"):
        raise ValueError("append-only position lineage did not pass")
    if not _valid_sha256(record["position_payload_hash"]):
        raise ValueError("append-only position payload hash is invalid")
    position_keys = (
        "compact_identifier", "compact_fingerprint", "outer_identifier",
        "outer_fingerprint", "archive_fingerprint",
    )
    shared_keys = position_keys+("appended_child_hash",)
    _exact_keys(record["position_only"], position_keys, "position-only lineage identity")
    _exact_keys(record["shared"], shared_keys, "shared lineage identity")
    for stage, names in (("position_only", position_keys), ("shared", shared_keys)):
        values = record[stage]
        for name in names:
            value = str(values[name])
            if not value:
                raise ValueError(f"{stage} lineage {name} is empty")
            if "fingerprint" in name or "hash" in name:
                if not _valid_sha256(value):
                    raise ValueError(f"{stage} lineage {name} is invalid")
    return {
        "position_payload_hash": str(record["position_payload_hash"]),
        "position_only_archive_fingerprint": str(
            record["position_only"]["archive_fingerprint"]
        ),
        "shared_archive_fingerprint": str(record["shared"]["archive_fingerprint"]),
        "appended_child_hash": str(record["shared"]["appended_child_hash"]),
        "passed": True,
    }


def _localization_from_profile(profile, proper_radius, weights):
    energy = weights*profile**2
    total = float(np.sum(energy))
    fractions = []
    for lower, upper in zip(COLLAR_EDGES[:-1], COLLAR_EDGES[1:]):
        mask = proper_radius >= lower
        if np.isfinite(upper):
            mask &= proper_radius < upper
        fractions.append(float(np.sum(energy[mask])/total) if total > 0.0 else 0.0)
    if total > 0.0:
        index = int(np.searchsorted(np.cumsum(energy), 0.9*total))
        radius_90 = float(proper_radius[min(index, len(proper_radius)-1)])
    else:
        radius_90 = 0.0
    integral2 = float(np.trapezoid(profile**2, proper_radius))
    integral4 = float(np.trapezoid(profile**4, proper_radius))
    support = integral2**2/integral4 if integral4 > 0.0 else 0.0
    return {
        "collar_edges": COLLAR_EDGES.tolist(),
        "area_energy_fractions": fractions,
        "area_energy_fraction_sum": float(sum(fractions)),
        "proper_radius_90": radius_90,
        "effective_proper_support_length": float(support),
        "effective_support_quadrature": (
            "(trapezoid(c_hzz^2,dell)^2)/trapezoid(c_hzz^4,dell)"
        ),
    }


def _validate_correction_profile(record):
    keys = (
        "physical_component_order", "signed_normalized_correction",
        "proper_radius", "proper_wall_weights", "full_physical_Linf",
        "full_physical_weighted_RMS", "hzz_C", "hzz_W", "localization",
        "small_full_Linf_gate", "small_full_RMS_gate", "order_one_failure",
    )
    _exact_keys(record, keys, "dense-wall correction profile")
    if tuple(record["physical_component_order"]) != PHYSICAL_COMPONENT_ORDER:
        raise ValueError("correction physical-component order differs")
    correction = _array(
        record["signed_normalized_correction"], "signed correction profile",
        shape=(2, 1025, len(PHYSICAL_COMPONENT_ORDER)),
    )
    proper = _array(record["proper_radius"], "proper radius", shape=(2, 1025))
    weights = _array(record["proper_wall_weights"], "proper-wall weights", shape=(2, 1025))
    if (
        np.any(weights < 0.0)
        or float(np.sum(weights)) <= 0.0
        or np.any(proper[:, 0] != 0.0)
        or np.any(np.signbit(proper[:, 0]))
        or np.any(np.diff(proper, axis=1) <= 0.0)
    ):
        raise ValueError("proper-wall correction geometry is invalid")
    total_weight = float(np.sum(weights))
    full_linf = float(np.max(np.abs(correction)))
    full_rms = float(np.sqrt(
        np.sum(weights[:, :, None]*correction**2)
        /(len(PHYSICAL_COMPONENT_ORDER)*total_weight)
    ))
    hzz = correction[:, :, HZZ_INDEX]
    hzz_c = float(np.max(np.abs(hzz)))
    hzz_w = float(np.sqrt(np.sum(weights*hzz**2)/total_weight))
    summaries = {
        "full_physical_Linf": full_linf,
        "full_physical_weighted_RMS": full_rms,
        "hzz_C": hzz_c,
        "hzz_W": hzz_w,
    }
    if any(not _same_number(record[name], value) for name, value in summaries.items()):
        raise ValueError("correction-size summaries differ from their profile")
    expected_flags = {
        "small_full_Linf_gate": bool(full_linf <= 0.05),
        "small_full_RMS_gate": bool(full_rms <= 0.01),
        "order_one_failure": bool(full_linf > 0.5),
    }
    for name, expected in expected_flags.items():
        if _strict_bool(record[name], f"correction {name}") is not expected:
            raise ValueError(f"correction flag {name} differs from its profile")
    localization = record["localization"]
    if not isinstance(localization, Mapping) or tuple(localization) != WALL_ORDER:
        raise ValueError("correction localization walls differ")
    for wall_index, wall in enumerate(WALL_ORDER):
        expected = _localization_from_profile(
            hzz[wall_index], proper[wall_index], weights[wall_index],
        )
        found = localization[wall]
        _exact_keys(found, expected, f"{wall} correction localization")
        for name, value in expected.items():
            if isinstance(value, list):
                if not np.array_equal(np.asarray(found[name]), np.asarray(value)):
                    raise ValueError(f"{wall} localization {name} differs")
            elif isinstance(value, float):
                if not _same_number(found[name], value):
                    raise ValueError(f"{wall} localization {name} differs")
            elif str(found[name]) != value:
                raise ValueError(f"{wall} localization {name} differs")
    passed = bool(
        expected_flags["small_full_Linf_gate"]
        and expected_flags["small_full_RMS_gate"]
        and not expected_flags["order_one_failure"]
    )
    return {**summaries, **expected_flags, "passed": passed}


def _validate_axis_image_profile(record):
    keys = (
        "method", "image_order", "z", "z_sha256", "input_form",
        "raw_q4_bulk", "raw_q4_compatible", "raw_q4_change", "raw_q5_bulk",
        "raw_q5_compatible", "raw_q5_change", "images", "small_limit",
        "order_one_limit", "all_small", "any_order_one",
    )
    _exact_keys(record, keys, "V2 axis acceleration-image profile")
    if not (
        str(record["method"])
        == "Protocol-125-analytic-axis-acceleration-derivative-images"
        and tuple(record["image_order"]) == AXIS_ACCELERATION_IMAGE_ORDER
        and float(record["small_limit"]) == AXIS_IMAGE_SMALL_LIMIT
        and float(record["order_one_limit"]) == AXIS_IMAGE_ORDER_ONE_LIMIT
        and str(record["input_form"])
        in ("direct-continuous-axis", "full-continuous-mesh-axis-column")
    ):
        raise ValueError("axis acceleration-image method or limits differ")
    frozen_z = frozen_validation_meshes()["V2"]["z"]
    z = _array(record["z"], "axis-image compact coordinate", shape=frozen_z.shape)
    if not np.array_equal(z, frozen_z) or str(record["z_sha256"]) != hash_arrays(z):
        raise ValueError("axis-image compact coordinate differs from frozen V2")
    raw = {
        name: _array(record[name], f"axis image {name}", shape=z.shape)
        for name in (
            "raw_q4_bulk", "raw_q4_compatible", "raw_q4_change",
            "raw_q5_bulk", "raw_q5_compatible", "raw_q5_change",
        )
    }
    if not (
        np.array_equal(raw["raw_q4_change"], raw["raw_q4_compatible"]-raw["raw_q4_bulk"])
        and np.array_equal(raw["raw_q5_change"], raw["raw_q5_compatible"]-raw["raw_q5_bulk"])
    ):
        raise ValueError("axis-image raw q4/q5 changes differ")
    images = record["images"]
    if not isinstance(images, Mapping) or tuple(images) != AXIS_ACCELERATION_IMAGE_ORDER:
        raise ValueError("axis acceleration-image inventory differs")
    expected_inputs = {
        AXIS_ACCELERATION_IMAGE_ORDER[0]: (
            2.0*raw["raw_q4_bulk"], 2.0*raw["raw_q4_compatible"],
        ),
        AXIS_ACCELERATION_IMAGE_ORDER[1]: (
            raw["raw_q5_bulk"], raw["raw_q5_compatible"],
        ),
    }
    small_flags = []
    order_one_flags = []
    summaries = {}
    image_keys = (
        "bulk_image", "compatible_image", "signed_normalized_correction", "K",
        "maximum_index", "maximum_z", "small_image_gate", "order_one_failure",
    )
    for name in AXIS_ACCELERATION_IMAGE_ORDER:
        image = images[name]
        _exact_keys(image, image_keys, f"axis acceleration image {name}")
        bulk = _array(image["bulk_image"], f"{name} bulk image", shape=z.shape)
        compatible = _array(
            image["compatible_image"], f"{name} compatible image", shape=z.shape,
        )
        if not (
            np.array_equal(bulk, expected_inputs[name][0])
            and np.array_equal(compatible, expected_inputs[name][1])
        ):
            raise ValueError(f"axis acceleration image {name} does not derive from q4/q5")
        denominator = np.maximum.reduce((np.ones_like(bulk), np.abs(bulk), np.abs(compatible)))
        correction = (compatible-bulk)/denominator
        found = _array(
            image["signed_normalized_correction"], f"{name} correction", shape=z.shape,
        )
        if not np.array_equal(found, correction):
            raise ValueError(f"axis acceleration image {name} correction differs")
        index = int(np.argmax(np.abs(correction)))
        maximum = float(np.abs(correction[index]))
        small = bool(maximum <= AXIS_IMAGE_SMALL_LIMIT)
        order_one = bool(maximum > AXIS_IMAGE_ORDER_ONE_LIMIT)
        if not (
            _same_number(image["K"], maximum)
            and int(image["maximum_index"]) == index
            and _same_number(image["maximum_z"], z[index])
            and _strict_bool(image["small_image_gate"], f"{name} small gate") is small
            and _strict_bool(image["order_one_failure"], f"{name} order-one gate") is order_one
        ):
            raise ValueError(f"axis acceleration image {name} summary differs")
        small_flags.append(small)
        order_one_flags.append(order_one)
        summaries[name] = {"K": maximum, "small": small, "order_one": order_one}
    all_small = all(small_flags)
    any_order_one = any(order_one_flags)
    if not (
        _strict_bool(record["all_small"], "axis all-small gate") is all_small
        and _strict_bool(record["any_order_one"], "axis order-one flag") is any_order_one
    ):
        raise ValueError("axis acceleration-image aggregate flags differ")
    return {
        "images": summaries,
        "all_small": all_small,
        "any_order_one": any_order_one,
        "passed": bool(all_small and not any_order_one),
    }


def _validate_bulk_sampler_provenance(record, context, correction, axis):
    _exact_keys(record, BULK_SAMPLER_PROVENANCE_KEYS, "bulk sampler provenance")
    if not (
        str(record["protocol_identifier"]) == BULK_SAMPLER_PROVENANCE_IDENTIFIER
        and str(record["parent_label"]) == context["label"]
        and str(record["parent_identity"]) == context["identity"]
    ):
        raise ValueError("bulk sampler provenance parent binding differs")
    for name in (
        "source_fingerprint", "sampler_fingerprint", "sampler_archive_sha256",
        "dense_wall_bulk_physical_sha256", "V2_axis_bulk_reduced_sha256",
    ):
        if not _valid_sha256(record[name]):
            raise ValueError(f"bulk sampler provenance {name} is invalid")
    if not (
        str(record["dense_wall_sha256"]) == DENSE_WALL_SHA256
        and str(record["V2_coordinate_sha256"])
        == frozen_validation_meshes()["V2"]["sha256"]
        and str(record["correction_profile_sha256"])
        == _fingerprint_tree(correction, root="correction_profile")
        and str(record["axis_image_profile_sha256"])
        == _fingerprint_tree(axis, root="axis_image_profile")
        and str(record["wall_recipe"])
        == "clamped-Q3-in-u-with-native-Du7-endpoints"
        and str(record["axis_recipe"])
        == "clamped-Q5-in-z-with-native-Dz7-endpoints"
    ):
        raise ValueError("bulk sampler recipe, mesh, or product binding differs")
    required_flags = {
        "wall_values_direct_from_clamped_Q3": True,
        "axis_values_direct_from_clamped_Q5": True,
        "stored_correction_interpolated": False,
        "compatible_contract_used_for_bulk": False,
    }
    for name, expected in required_flags.items():
        if _strict_bool(record[name], f"bulk sampler {name}") is not expected:
            raise ValueError(f"bulk sampler provenance flag {name} differs")
    return {
        "source_fingerprint": str(record["source_fingerprint"]),
        "sampler_fingerprint": str(record["sampler_fingerprint"]),
        "sampler_archive_sha256": str(record["sampler_archive_sha256"]),
        "dense_wall_bulk_physical_sha256": str(
            record["dense_wall_bulk_physical_sha256"]
        ),
        "V2_axis_bulk_reduced_sha256": str(record["V2_axis_bulk_reduced_sha256"]),
        "passed": True,
    }


def _gate_record(group_name, context, *, complete, provenance_valid, passed, details, reasons=()):
    complete = bool(complete)
    provenance_valid = bool(provenance_valid)
    passed = bool(passed and complete and provenance_valid)
    reasons = tuple(str(reason) for reason in reasons)
    frozen_details = _freeze(details)
    payload = {
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "group_name": str(group_name),
        "parent_label": context["label"],
        "parent_identity": context["identity"],
        "complete": complete,
        "provenance_valid": provenance_valid,
        "passed": passed,
        "invalid_reasons": reasons,
        "details": frozen_details,
    }
    return MappingProxyType({
        "complete": complete,
        "provenance_valid": provenance_valid,
        "passed": passed,
        "fingerprint": _fingerprint_tree(payload, root=f"gate/{group_name}"),
        "parent_label": context["label"],
        "parent_identity": context["identity"],
        "group_name": str(group_name),
        "invalid_reasons": reasons,
        "details": frozen_details,
    })


def _invalid_groups(context, reason):
    return MappingProxyType({
        name: _gate_record(
            name,
            context,
            complete=False,
            provenance_valid=False,
            passed=False,
            details={"scorer_executed": False, "failure_kind": "invalid-audit"},
            reasons=(reason,),
        )
        for name in POST_ACCELERATION_GROUPS
    })


def _invalid_result(context, reason, *, hashes_before=None):
    groups = MappingProxyType({}) if context is None else _invalid_groups(context, reason)
    result = {
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "classification": "INVALID-audit",
        "complete": False,
        "provenance_valid": False,
        "passed": False,
        "required_group_order": POST_ACCELERATION_GROUPS,
        "groups": groups,
        "invalid_reasons": (str(reason),),
        "input_hashes_before": hashes_before,
        "input_hashes_after": None,
        "inputs_stable_while_scoring": False,
        "single_parent_only": True,
        "two_parent_refinement_still_required": True,
        "phase_a_authorized": False,
        "scientific_execution_authorized": False,
        "artifact_written": False,
    }
    if context is not None:
        result.update({
            "parent_label": context["label"],
            "parent_identity": context["identity"],
        })
    return _freeze(result)


def _not_reached_gate_record(group_name, context, *, blocked_by, details):
    blocked_by = str(blocked_by)
    if blocked_by not in POST_ACCELERATION_GROUPS:
        raise ValueError("not-reached post-acceleration blocker is unknown")
    frozen_details = _freeze(details)
    payload = {
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "group_name": str(group_name),
        "parent_label": context["label"],
        "parent_identity": context["identity"],
        "complete": True,
        "provenance_valid": True,
        "passed": False,
        "not_reached": True,
        "blocked_by": blocked_by,
        "invalid_reasons": (),
        "details": frozen_details,
    }
    return MappingProxyType({
        "complete": True,
        "provenance_valid": True,
        "passed": False,
        "fingerprint": _fingerprint_tree(
            payload, root=f"gate/{group_name}/not_reached",
        ),
        "parent_label": context["label"],
        "parent_identity": context["identity"],
        "group_name": str(group_name),
        "not_reached": True,
        "blocked_by": blocked_by,
        "invalid_reasons": (),
        "details": frozen_details,
    })


def compose_protocol125_acceleration_failure_records(inputs, provenance):
    """Compose one sealed acceleration stop into ordered ledger records.

    Closure failures stop at ``acceleration_closure``.  A measured wall-owner
    rejection records both the consequent closure failure and the directly
    measured ``wall_algebra`` failure; only groups after that measured stop
    are marked not reached.  No scorer or numerical solve is called here.
    """
    context = None
    hashes_before = None
    try:
        context, hashes_before = _acceleration_failure_input_hashes(inputs)
        _validate_acceleration_failure_provenance(
            provenance, context, hashes_before,
        )
        pre = _validate_pre_acceleration(
            inputs.pre_acceleration_result, context,
        )
        failure = validate_protocol125_acceleration_failure_record(
            inputs.acceleration_failure_record,
        )
        if (
            str(failure["protocol_identifier"])
            != ACCELERATION_FAILURE_PROTOCOL_IDENTIFIER
            or str(failure["parent_label"]) != context["label"]
            or str(failure["parent_identity"]) != context["identity"]
        ):
            raise ValueError(
                "acceleration failure differs from its passed prerequisite"
            )
        hashes_after = _acceleration_failure_input_hashes(inputs)[1]
        changed = tuple(
            name for name in ACCELERATION_FAILURE_INPUT_HASH_KEYS
            if str(hashes_before[name]) != str(hashes_after[name])
        )
        if changed:
            raise ValueError(
                f"acceleration-failure inputs changed while scoring: {changed}"
            )
    except (TypeError, ValueError, RuntimeError, KeyError, AttributeError) as error:
        return _invalid_result(
            context,
            f"acceleration-failure evidence: {error}",
            hashes_before=hashes_before,
        )

    failure_group = str(failure["failure_group"])
    failure_fingerprint = str(failure["fingerprint"])
    binding_details = {
        "pre_acceleration": pre,
        "scientific_failure_record": failure,
        "failure_group": failure_group,
        "failure_reason": str(failure["failure_reason"]),
        "failure_fingerprint": failure_fingerprint,
        "attempt_fingerprint": str(failure["attempt_fingerprint"]),
        "maps_completed": int(failure["maps_completed"]),
        "consecutive_converged_maps": int(
            failure["consecutive_converged_maps"]
        ),
        "acceleration_returned": False,
        "input_fingerprints": {
            "pre_acceleration": hashes_before[
                "pre_acceleration_result_sha256"
            ],
            "acceleration_failure": hashes_before[
                "acceleration_failure_record_sha256"
            ],
        },
    }
    closure = _gate_record(
        "acceleration_closure",
        context,
        complete=True,
        provenance_valid=True,
        passed=False,
        details={
            **binding_details,
            "fixed_point_or_final_closure_passed": False,
            "measured_wall_failure_prevented_closure": bool(
                failure_group == "wall_algebra"
            ),
        },
    )
    groups = {"acceleration_closure": closure}
    if failure_group == "wall_algebra":
        groups["wall_algebra"] = _gate_record(
            "wall_algebra",
            context,
            complete=True,
            provenance_valid=True,
            passed=False,
            details={
                "failure_reason": str(failure["failure_reason"]),
                "failure_event": failure["failure_event"],
                "last_iterate_evidence": failure["last_iterate_evidence"],
                "failure_fingerprint": failure_fingerprint,
                "direct_measured_wall_gate_failure": True,
                "input_fingerprints": binding_details["input_fingerprints"],
            },
        )
        downstream = POST_ACCELERATION_GROUPS[2:]
    else:
        downstream = POST_ACCELERATION_GROUPS[1:]
    for name in downstream:
        groups[name] = _not_reached_gate_record(
            name,
            context,
            blocked_by="acceleration_closure",
            details={
                "scorer_executed": False,
                "failure_kind": "ordered-scientific-stop",
                "upstream_failure_group": failure_group,
                "upstream_failure_reason": str(failure["failure_reason"]),
                "upstream_failure_fingerprint": failure_fingerprint,
            },
        )
    ordered_groups = MappingProxyType({
        name: groups[name] for name in POST_ACCELERATION_GROUPS
    })
    failed_groups = tuple(
        name for name in POST_ACCELERATION_GROUPS
        if not ordered_groups[name].get("not_reached", False)
        and not ordered_groups[name]["passed"]
    )
    not_reached_groups = tuple(
        name for name in POST_ACCELERATION_GROUPS
        if ordered_groups[name].get("not_reached", False)
    )
    return _freeze({
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "classification": "FAIL-acceleration",
        "complete": True,
        "provenance_valid": True,
        "passed": False,
        "parent_label": context["label"],
        "parent_identity": context["identity"],
        "required_group_order": POST_ACCELERATION_GROUPS,
        "groups": ordered_groups,
        "failed_groups": failed_groups,
        "not_reached_groups": not_reached_groups,
        "scientific_failure_fingerprint": failure_fingerprint,
        "invalid_reasons": (),
        "input_hashes_before": hashes_before,
        "input_hashes_after": hashes_after,
        "inputs_stable_while_scoring": True,
        "single_parent_only": True,
        "two_parent_refinement_still_required": True,
        "phase_a_authorized": False,
        "scientific_execution_authorized": False,
        "artifact_written": False,
    })


def compose_protocol125_postacceleration_records(inputs, provenance):
    """Return five immutable ledger-ready records and a fail-closed summary."""
    context = None
    hashes_before = None
    try:
        context, hashes_before = _input_hashes(inputs)
        _validate_provenance(provenance, context, hashes_before)
        pre = _validate_pre_acceleration(inputs.pre_acceleration_result, context)
        fixed = _validate_fixed_point(inputs.fixed_point_record)
        wall = _validate_normalized_wall_evidence(
            inputs.normalized_wall_profile_score, context,
        )
        matrix = _validate_final_matrix(inputs.final_representation_matrix, context)
        if (
            wall["source_fingerprint"] != matrix["source_fingerprint"]
            or wall["endpoint_fingerprint"] != matrix["endpoint_fingerprint"]
        ):
            raise ValueError(
                "normalized wall-profile evidence differs from the final matrix"
            )
        lineage = _validate_append_only_lineage(inputs.append_only_lineage)
        correction = _validate_correction_profile(inputs.correction_profile)
        axis = _validate_axis_image_profile(inputs.axis_image_profile)
        sampler = _validate_bulk_sampler_provenance(
            inputs.bulk_sampler_provenance,
            context,
            inputs.correction_profile,
            inputs.axis_image_profile,
        )
        hashes_after = _input_hashes(inputs)[1]
        changed = tuple(
            name for name in INPUT_HASH_KEYS
            if str(hashes_before[name]) != str(hashes_after[name])
        )
        if changed:
            raise ValueError(f"post-acceleration inputs changed while scoring: {changed}")
    except (TypeError, ValueError, RuntimeError, KeyError, AttributeError) as error:
        return _invalid_result(
            context,
            f"post-acceleration evidence: {error}",
            hashes_before=hashes_before,
        )

    groups = MappingProxyType({
        "acceleration_closure": _gate_record(
            "acceleration_closure",
            context,
            complete=True,
            provenance_valid=True,
            passed=fixed["acceleration_closure_pass"],
            details={
                "pre_acceleration": pre,
                "fixed_point_maps_used": fixed["maps_used"],
                "consecutive_converged_maps": fixed["consecutive_converged_maps"],
                "two_consecutive_map_convergence": fixed[
                    "two_consecutive_map_convergence"
                ],
                "source_closure": fixed["source"],
                "normal_gauge": fixed["normal_gauge"],
                "owner_prohibitions_verified": True,
                "input_fingerprints": {
                    "pre_acceleration": hashes_before[
                        "pre_acceleration_result_sha256"
                    ],
                    "fixed_point": hashes_before["fixed_point_record_sha256"],
                },
            },
        ),
        "wall_algebra": _gate_record(
            "wall_algebra",
            context,
            complete=True,
            provenance_valid=True,
            passed=bool(fixed["wall_algebra_pass"] and wall["passed"]),
            details={
                "coupled_4x4": fixed["coupled"],
                "selective_per_field": fixed["selective"],
                "normal_gauge": fixed["normal_gauge"],
                "second_junction_decomposition": fixed["wall_second_tangent"],
                "source_normalized_wall_profiles": wall["source"],
                "dense_normalized_wall_profiles": wall["dense"],
                "wall_profile_evidence": {
                    name: wall[name]
                    for name in (
                        "coordinate_hashes", "derivative_recipes",
                        "time_symmetry_passed", "live_compact_context_passed",
                        "named_row_wall_passed", "named_row_wall_gate_count",
                        "fingerprint",
                    )
                },
                "ownership_parity_and_reassembly_verified": True,
                "input_fingerprints": {
                    "fixed_point": hashes_before["fixed_point_record_sha256"],
                    "wall_profiles": hashes_before[
                        "normalized_wall_profile_evidence_sha256"
                    ],
                },
            },
        ),
        "final_representation": _gate_record(
            "final_representation",
            context,
            complete=True,
            provenance_valid=True,
            passed=bool(
                lineage["passed"]
                and all(matrix["lane_pass"][name] for name in FINAL_REPRESENTATION_LANES)
            ),
            details={
                "required_lane_order": FINAL_REPRESENTATION_LANES,
                "lane_pass": {
                    name: matrix["lane_pass"][name] for name in FINAL_REPRESENTATION_LANES
                },
                "append_only_position_lineage": lineage,
                "source_fingerprint": matrix["source_fingerprint"],
                "endpoint_fingerprint": matrix["endpoint_fingerprint"],
                "input_fingerprints": {
                    "final_matrix": hashes_before[
                        "final_representation_matrix_sha256"
                    ],
                    "lineage": hashes_before["append_only_lineage_sha256"],
                },
            },
        ),
        "endpoint_derivatives": _gate_record(
            "endpoint_derivatives",
            context,
            complete=True,
            provenance_valid=True,
            passed=all(matrix["lane_pass"][name] for name in ENDPOINT_DERIVATIVE_LANES),
            details={
                "required_lane_order": ENDPOINT_DERIVATIVE_LANES,
                "lane_pass": {
                    name: matrix["lane_pass"][name] for name in ENDPOINT_DERIVATIVE_LANES
                },
                "row_implied_acceleration_z_defect": fixed["selective"][
                    "maximum_row_implied_scaled_defect"
                ],
                "independent_outer_replay_required": True,
                "input_fingerprints": {
                    "final_matrix": hashes_before[
                        "final_representation_matrix_sha256"
                    ],
                    "fixed_point": hashes_before["fixed_point_record_sha256"],
                },
            },
        ),
        "correction_size": _gate_record(
            "correction_size",
            context,
            complete=True,
            provenance_valid=True,
            passed=bool(correction["passed"] and axis["passed"] and sampler["passed"]),
            details={
                "dense_wall_correction": correction,
                "V2_axis_acceleration_derivative_images": axis,
                "restricted_bulk_sampler": sampler,
                "two_parent_decrease_envelope_and_localization_still_required": True,
                "input_fingerprints": {
                    "correction_profile": hashes_before["correction_profile_sha256"],
                    "axis_image_profile": hashes_before["axis_image_profile_sha256"],
                    "bulk_sampler_provenance": hashes_before[
                        "bulk_sampler_provenance_sha256"
                    ],
                },
            },
        ),
    })
    passed = all(record["passed"] for record in groups.values())
    return _freeze({
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "classification": (
            "PASS-single-parent-post-acceleration"
            if passed else "FAIL-single-parent-post-acceleration"
        ),
        "complete": True,
        "provenance_valid": True,
        "passed": bool(passed),
        "parent_label": context["label"],
        "parent_identity": context["identity"],
        "required_group_order": POST_ACCELERATION_GROUPS,
        "groups": groups,
        "failed_groups": tuple(name for name in POST_ACCELERATION_GROUPS if not groups[name]["passed"]),
        "invalid_reasons": (),
        "input_hashes_before": hashes_before,
        "input_hashes_after": hashes_after,
        "inputs_stable_while_scoring": True,
        "single_parent_only": True,
        "two_parent_refinement_still_required": True,
        "phase_a_authorized": False,
        "scientific_execution_authorized": False,
        "artifact_written": False,
    })
