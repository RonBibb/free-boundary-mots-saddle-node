"""Pure two-parent composition for the Protocol-125 gate ledger.

This module consumes two continuous, already-built representation pairs and
already-scored bulk/correction evidence.  It neither constructs nor repairs a
parent, performs a solve, writes an artifact, nor authorizes any subsequent
experiment.  Its only output is the two fail-closed records expected by the
Protocol-125 gate ledger.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from bhps.joint_parent_adjudication import (
    FLOOR,
    V_MESH_NAMES,
    score_acceleration_pair_on_v_meshes,
    score_q4_q5_derivative_images_on_v_meshes,
    score_state_pair_on_v_meshes,
)
from bhps.joint_parent_bulk_validation import compare_protocol125_common_v2
from bhps.joint_parent_construction import (
    validate_protocol125_successful_parent_provenance_record,
)
from bhps.joint_parent_native_evidence import _digest_tree as _native_evidence_digest_tree
from bhps.joint_parent_preacceleration import (
    NATIVE_EVIDENCE_PROTOCOL_IDENTIFIER,
    NATIVE_POSITION_TANGENT_LANES,
)
from bhps.joint_parent_refinement_diagnostics import (
    PHYSICAL_COMPONENT_ORDER,
    adjudicate_correction_refinement,
    frozen_validation_meshes,
)
from bhps.matched_staged_continuum import hash_arrays


PROTOCOL_IDENTIFIER = "Protocol-125-two-parent-composer-v1"
TWO_PARENT_RECORD_ORDER = (
    "N0_N1_representation",
    "correction_refinement",
)
REPRESENTATION_LANE_ORDER = (
    "state_spatial",
    "position_q4_q5_derivative_images",
    "acceleration",
    "acceleration_q4_q5_derivative_images",
    "bulk_common_V2_nonworsening",
    "native_completion_correction_refinement",
)
CORRECTION_REFINEMENT_LANE_ORDER = (
    "dense_wall_correction_refinement",
    "V2_hzz_zz_difference",
    "V2_a_hzz_difference",
    "V2_axis_acceleration_derivative_images",
)
INPUT_HASH_KEYS = (
    "N0_position_state_sha256",
    "N1_position_state_sha256",
    "N0_acceleration_state_sha256",
    "N1_acceleration_state_sha256",
    "N0_native_completion_evidence_sha256",
    "N1_native_completion_evidence_sha256",
    "N0_construction_provenance_sha256",
    "N1_construction_provenance_sha256",
    "v_meshes_sha256",
    "dense_wall_r_sha256",
    "N0_bulk_audit_sha256",
    "N1_bulk_audit_sha256",
    "N0_correction_profile_sha256",
    "N1_correction_profile_sha256",
    "N0_position_V2_sha256",
    "N1_position_V2_sha256",
    "N0_hzz_zz_V2_sha256",
    "N1_hzz_zz_V2_sha256",
    "N0_a_hzz_V2_sha256",
    "N1_a_hzz_V2_sha256",
    "N0_axis_image_profile_sha256",
    "N1_axis_image_profile_sha256",
)
_PARENT_LABELS = ("N0", "N1")
NUMERICAL_FLOOR = FLOOR
_NATIVE_COMPLETION_LANE = "native_position_completion"
_NATIVE_CORRECTION_RULES = MappingProxyType({
    "anisotropy_physical_correction": MappingProxyType({
        "value_key": "anisotropy_hrr_normalized_Linf",
        "ceiling": 1e-10,
        "refinement_predicate": "strict_decrease_f",
    }),
    "chi_physical_correction": MappingProxyType({
        "value_key": "chi_normalized_Linf",
        "ceiling": 1e-10,
        "refinement_predicate": "strict_decrease_f",
    }),
    "q4_axis_image_correction": MappingProxyType({
        "value_key": "axis_q4_second_derivative_image_normalized_Linf",
        "ceiling": 1e-10,
        "refinement_predicate": "strict_decrease_f",
    }),
    "lapse_owned_correction": MappingProxyType({
        "value_key": "lapse_h00_normalized_Linf",
        "ceiling": 0.05,
        "refinement_predicate": "nonworsen_f",
    }),
})
_CORRECTION_PROFILE_KEYS = (
    "physical_component_order",
    "signed_normalized_correction",
    "proper_radius",
    "proper_wall_weights",
    "full_physical_Linf",
    "full_physical_weighted_RMS",
    "hzz_C",
    "hzz_W",
    "localization",
    "small_full_Linf_gate",
    "small_full_RMS_gate",
    "order_one_failure",
)


@dataclass(frozen=True)
class Protocol125TwoParentInputs:
    """Explicit continuous and already-scored inputs for the two-parent gates."""

    n0_position_state: object
    n1_position_state: object
    n0_acceleration_state: object
    n1_acceleration_state: object
    n0_native_completion_evidence: Mapping
    n1_native_completion_evidence: Mapping
    n0_construction_provenance: Mapping
    n1_construction_provenance: Mapping
    v_meshes: Mapping
    dense_wall_r: np.ndarray
    n0_bulk_audit: Mapping
    n1_bulk_audit: Mapping
    n0_correction_profile: Mapping
    n1_correction_profile: Mapping
    n0_position_v2: np.ndarray
    n1_position_v2: np.ndarray
    n0_hzz_zz_v2: np.ndarray
    n1_hzz_zz_v2: np.ndarray
    n0_a_hzz_v2: np.ndarray
    n1_a_hzz_v2: np.ndarray
    n0_axis_image_profile: Mapping
    n1_axis_image_profile: Mapping


def _valid_sha256(value):
    value = str(value)
    return bool(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _parent_identities(parent_identities):
    if not isinstance(parent_identities, Mapping) or set(parent_identities) != set(
        _PARENT_LABELS
    ):
        raise ValueError("two-parent identities must be exactly N0 and N1")
    result = {
        label: str(parent_identities[label]) for label in _PARENT_LABELS
    }
    for label, identity in result.items():
        if not _valid_sha256(identity):
            raise ValueError(f"{label} parent identity must be a lowercase SHA-256 digest")
    if result["N0"] == result["N1"]:
        raise ValueError("N0 and N1 parent identities must be distinct")
    return MappingProxyType(result)


def strict_decrease_f(coarse, refined, *, floor=NUMERICAL_FLOOR):
    """Protocol-125 strict decrease, excusing ordering only below its floor."""
    coarse = float(coarse)
    refined = float(refined)
    floor = float(floor)
    if not all(np.isfinite(value) and value >= 0.0 for value in (
        coarse, refined, floor,
    )):
        raise ValueError("strict-decrease inputs must be finite and nonnegative")
    return bool(refined < coarse or max(coarse, refined) <= floor)


def nonworsen_f(coarse, refined, *, floor=NUMERICAL_FLOOR):
    """Protocol-125 nonworsening, excusing ordering only below its floor."""
    coarse = float(coarse)
    refined = float(refined)
    floor = float(floor)
    if not all(np.isfinite(value) and value >= 0.0 for value in (
        coarse, refined, floor,
    )):
        raise ValueError("nonworsening inputs must be finite and nonnegative")
    return bool(refined <= coarse or max(coarse, refined) <= floor)


def _digest_bytes(digest, tag, payload):
    tag = str(tag).encode("utf-8")
    payload = bytes(payload)
    digest.update(len(tag).to_bytes(8, "little"))
    digest.update(tag)
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def _update_tree_digest(digest, value, path):
    """Hash nested evidence without lossy JSON or object representations."""
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
        if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
            raise ValueError(f"hashed array {path} is nonfinite")
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


def _state_fingerprint(state, label):
    method = getattr(state, "fingerprint", None)
    if method is None or not callable(method):
        raise TypeError(f"{label} must expose fingerprint()")
    found = str(method())
    if not _valid_sha256(found):
        raise ValueError(f"{label} fingerprint must be a lowercase SHA-256 digest")
    return found


def _native_gate_fingerprint(name, lane):
    payload = {
        "lane": str(name),
        "complete": lane["complete"],
        "provenance_valid": lane["provenance_valid"],
        "passed": lane["passed"],
        "details": lane["details"],
    }
    return _native_evidence_digest_tree(
        payload, root=f"native/{name}",
    )


def _validate_native_completion_evidence(evidence, label, parent_identity):
    required = (
        "protocol_identifier",
        "parent_label",
        "parent_identity",
        "source_coordinate_sha256",
        "position_sha256",
        "input_fingerprint_before",
        "input_fingerprint_after",
        "lanes",
        "complete",
        "provenance_valid",
        "passed",
    )
    if not isinstance(evidence, Mapping) or any(name not in evidence for name in required):
        raise ValueError(f"{label} native-completion evidence is incomplete")
    if (
        str(evidence["protocol_identifier"]) != NATIVE_EVIDENCE_PROTOCOL_IDENTIFIER
        or str(evidence["parent_label"]) != label
        or str(evidence["parent_identity"]) != str(parent_identity)
        or not _valid_sha256(evidence["source_coordinate_sha256"])
        or not _valid_sha256(evidence["position_sha256"])
        or not _valid_sha256(evidence["input_fingerprint_before"])
        or str(evidence["input_fingerprint_after"])
        != str(evidence["input_fingerprint_before"])
    ):
        raise ValueError(f"{label} native-completion identity or provenance differs")
    for field in ("complete", "provenance_valid", "passed"):
        if type(evidence[field]) is not bool:
            raise TypeError(f"{label} native evidence {field} is not bool")
    if evidence["complete"] is not True or evidence["provenance_valid"] is not True:
        raise ValueError(f"{label} native-completion evidence is not validated")
    lanes = evidence["lanes"]
    if not isinstance(lanes, Mapping) or tuple(lanes) != NATIVE_POSITION_TANGENT_LANES:
        raise ValueError(f"{label} native evidence lane inventory differs")
    lane_summaries = {}
    for lane_name in NATIVE_POSITION_TANGENT_LANES:
        found_lane = lanes[lane_name]
        if not isinstance(found_lane, Mapping) or set(found_lane) != {
            "complete", "provenance_valid", "passed", "fingerprint", "details",
        }:
            raise ValueError(f"{label} native lane {lane_name} schema differs")
        for field in ("complete", "provenance_valid", "passed"):
            if type(found_lane[field]) is not bool:
                raise TypeError(f"{label} native lane {lane_name} {field} is not bool")
        if found_lane["complete"] is not True or found_lane[
            "provenance_valid"
        ] is not True:
            raise ValueError(f"{label} native lane {lane_name} is not validated")
        if (
            not _valid_sha256(found_lane["fingerprint"])
            or str(found_lane["fingerprint"])
            != _native_gate_fingerprint(lane_name, found_lane)
        ):
            raise ValueError(f"{label} native lane {lane_name} fingerprint differs")
        lane_summaries[lane_name] = found_lane
    if evidence["passed"] is not bool(
        all(found_lane["passed"] for found_lane in lane_summaries.values())
    ):
        raise ValueError(f"{label} native evidence pass flag does not reproduce")
    lane = lanes[_NATIVE_COMPLETION_LANE]
    details = lane["details"]
    detail_required = (
        "gates",
        "recorded_corrections",
        "recomputed_corrections",
        "recorded_correction_reproduction",
        "passed",
    )
    if not isinstance(details, Mapping) or any(
        name not in details for name in detail_required
    ):
        raise ValueError(f"{label} native-completion details are incomplete")
    gates = details["gates"]
    recorded = details["recorded_corrections"]
    recomputed = details["recomputed_corrections"]
    reproduction = details["recorded_correction_reproduction"]
    if not all(isinstance(value, Mapping) for value in (
        gates, recorded, recomputed, reproduction,
    )):
        raise ValueError(f"{label} native-completion correction evidence is invalid")
    if any(type(value) is not bool for value in gates.values()):
        raise TypeError(f"{label} native-completion gate value is not bool")
    correction_value_keys = {
        rule["value_key"] for rule in _NATIVE_CORRECTION_RULES.values()
    }
    if any(set(record) != correction_value_keys for record in (
        recorded, recomputed, reproduction,
    )):
        raise ValueError(f"{label} native-completion correction inventory differs")
    values = {}
    for gate_name, rule in _NATIVE_CORRECTION_RULES.items():
        value_key = rule["value_key"]
        if any(value_key not in record for record in (
            recorded, recomputed, reproduction,
        )) or gate_name not in gates:
            raise ValueError(f"{label} native correction {gate_name} is missing")
        found = float(recomputed[value_key])
        recorded_value = float(recorded[value_key])
        expected_gate = bool(
            np.isfinite(found) and found >= 0.0 and found <= float(rule["ceiling"])
        )
        if (
            not np.isfinite(recorded_value)
            or recorded_value != found
            or reproduction[value_key] is not True
            or type(gates[gate_name]) is not bool
            or gates[gate_name] is not expected_gate
        ):
            raise ValueError(f"{label} native correction {gate_name} does not reproduce")
        values[gate_name] = found
    if (
        type(details["passed"]) is not bool
        or details["passed"] is not bool(all(gates.values()))
        or lane["passed"] is not details["passed"]
    ):
        raise ValueError(f"{label} native-completion pass flag does not reproduce")
    return MappingProxyType({
        "lane_passed": bool(lane["passed"]),
        "values": MappingProxyType(values),
        "fingerprint": str(lane["fingerprint"]),
    })


def _validated_parent_evidence(inputs, parent_identities=None):
    identities = None if parent_identities is None else _parent_identities(
        parent_identities
    )
    found = {}
    for label, provenance, native in (
        (
            "N0",
            inputs.n0_construction_provenance,
            inputs.n0_native_completion_evidence,
        ),
        (
            "N1",
            inputs.n1_construction_provenance,
            inputs.n1_native_completion_evidence,
        ),
    ):
        expected_identity = None if identities is None else identities[label]
        validated_provenance = (
            validate_protocol125_successful_parent_provenance_record(
                provenance,
                expected_parent_label=label,
                expected_parent_identity=expected_identity,
            )
        )
        identity = str(validated_provenance["parent_identity"])
        found[label] = MappingProxyType({
            "provenance": validated_provenance,
            "native_completion": _validate_native_completion_evidence(
                native, label, identity,
            ),
        })
    if str(found["N0"]["provenance"]["parent_identity"]) == str(
        found["N1"]["provenance"]["parent_identity"]
    ):
        raise ValueError("N0 and N1 successful parent identities must be distinct")
    return MappingProxyType(found)


def _score_native_completion_correction_refinement(inputs, parent_identities):
    evidence = _validated_parent_evidence(inputs, parent_identities)
    parents = {}
    for label in _PARENT_LABELS:
        completion = evidence[label]["native_completion"]
        ceiling_gates = {
            gate_name: bool(
                completion["values"][gate_name] <= float(rule["ceiling"])
            )
            for gate_name, rule in _NATIVE_CORRECTION_RULES.items()
        }
        parents[label] = {
            "native_completion_lane_passed": bool(completion["lane_passed"]),
            "native_completion_fingerprint": completion["fingerprint"],
            "values": dict(completion["values"]),
            "ceiling_gates": ceiling_gates,
            "passed": bool(completion["lane_passed"] and all(ceiling_gates.values())),
        }
    refinement = {}
    for gate_name, rule in _NATIVE_CORRECTION_RULES.items():
        coarse = parents["N0"]["values"][gate_name]
        refined = parents["N1"]["values"][gate_name]
        predicate_name = str(rule["refinement_predicate"])
        predicate = strict_decrease_f if predicate_name == "strict_decrease_f" else (
            nonworsen_f
        )
        refinement[gate_name] = {
            "N0": coarse,
            "N1": refined,
            "floor": NUMERICAL_FLOOR,
            "predicate": predicate_name,
            "passed": predicate(coarse, refined),
        }
    return {
        "parents": parents,
        "refinement": refinement,
        "constituent_logical_AND": True,
        "passed": bool(
            all(record["passed"] for record in parents.values())
            and all(record["passed"] for record in refinement.values())
        ),
    }


def _validate_state(state, state_name, label):
    if str(getattr(state, "state_name", "")) != state_name:
        raise ValueError(f"{label} must be labeled {state_name}")
    for method_name in ("evaluate_coordinate_components", "evaluate_reduced"):
        if not callable(getattr(state, method_name, None)):
            raise TypeError(f"{label} lacks {method_name}()")
    _state_fingerprint(state, label)


def _validate_meshes(inputs):
    meshes = inputs.v_meshes
    if not isinstance(meshes, Mapping) or tuple(meshes) != V_MESH_NAMES:
        raise ValueError("two-parent comparison requires ordered V0/V1/V2 meshes")
    frozen = frozen_validation_meshes()
    for name in V_MESH_NAMES:
        record = meshes[name]
        if not isinstance(record, Mapping) or set(record) != {"z", "r", "sha256"}:
            raise ValueError(f"{name} mesh record must contain exactly z, r, sha256")
        z = np.asarray(record["z"], dtype=float)
        r = np.asarray(record["r"], dtype=float)
        found = hash_arrays(z, r)
        if found != frozen[name]["sha256"] or str(record["sha256"]) != found:
            raise ValueError(f"{name} mesh differs from the frozen Protocol-125 mesh")
    dense_r = np.asarray(inputs.dense_wall_r, dtype=float)
    if (
        dense_r.ndim != 1
        or not np.array_equal(dense_r, frozen["dense_wall"]["r"])
        or hash_arrays(dense_r) != frozen["dense_wall"]["sha256"]
    ):
        raise ValueError("dense-wall radius differs from the frozen Protocol-125 mesh")
    return frozen


def _validate_correction_profile(profile, dense_count, label):
    if not isinstance(profile, Mapping) or any(
        name not in profile for name in _CORRECTION_PROFILE_KEYS
    ):
        raise ValueError(f"{label} dense-wall correction profile is incomplete")
    if tuple(profile["physical_component_order"]) != PHYSICAL_COMPONENT_ORDER:
        raise ValueError(f"{label} correction physical-component order differs")
    correction = np.asarray(profile["signed_normalized_correction"], dtype=float)
    proper_radius = np.asarray(profile["proper_radius"], dtype=float)
    weights = np.asarray(profile["proper_wall_weights"], dtype=float)
    if correction.shape != (2, dense_count, len(PHYSICAL_COMPONENT_ORDER)):
        raise ValueError(f"{label} correction profile has the wrong dense-wall shape")
    if proper_radius.shape != (2, dense_count) or weights.shape != (2, dense_count):
        raise ValueError(f"{label} proper-wall evidence has the wrong shape")
    if not all(np.all(np.isfinite(value)) for value in (
        correction, proper_radius, weights,
    )):
        raise ValueError(f"{label} correction profile is nonfinite")
    if np.any(weights < 0.0):
        raise ValueError(f"{label} proper-wall weights are negative")


def _validate_inputs(inputs, *, parent_identities=None):
    if not isinstance(inputs, Protocol125TwoParentInputs):
        raise TypeError("two-parent inputs must use Protocol125TwoParentInputs")
    _validate_state(inputs.n0_position_state, "position", "N0 position state")
    _validate_state(inputs.n1_position_state, "position", "N1 position state")
    _validate_state(
        inputs.n0_acceleration_state, "acceleration", "N0 acceleration state",
    )
    _validate_state(
        inputs.n1_acceleration_state, "acceleration", "N1 acceleration state",
    )
    _validated_parent_evidence(inputs, parent_identities)
    frozen = _validate_meshes(inputs)
    for label, audit in (
        ("N0", inputs.n0_bulk_audit), ("N1", inputs.n1_bulk_audit),
    ):
        if not isinstance(audit, Mapping) or str(audit.get("parent_label")) != label:
            raise ValueError(f"{label} bulk audit is missing or mislabeled")
    dense_count = len(frozen["dense_wall"]["r"])
    _validate_correction_profile(inputs.n0_correction_profile, dense_count, "N0")
    _validate_correction_profile(inputs.n1_correction_profile, dense_count, "N1")
    v2_shape = (
        len(frozen["V2"]["z"]), len(frozen["V2"]["r"]),
    )
    for label, value in (
        ("N0 position V2", inputs.n0_position_v2),
        ("N1 position V2", inputs.n1_position_v2),
    ):
        array = np.asarray(value, dtype=float)
        if array.shape != v2_shape+(len(PHYSICAL_COMPONENT_ORDER),) or not np.all(
            np.isfinite(array)
        ):
            raise ValueError(f"{label} array has the wrong shape or is nonfinite")
    for label, value in (
        ("N0 hzz_zz V2", inputs.n0_hzz_zz_v2),
        ("N1 hzz_zz V2", inputs.n1_hzz_zz_v2),
        ("N0 a_hzz V2", inputs.n0_a_hzz_v2),
        ("N1 a_hzz V2", inputs.n1_a_hzz_v2),
    ):
        array = np.asarray(value, dtype=float)
        if array.shape != v2_shape or not np.all(np.isfinite(array)):
            raise ValueError(f"{label} array has the wrong shape or is nonfinite")
    for label, profile in (
        ("N0", inputs.n0_axis_image_profile),
        ("N1", inputs.n1_axis_image_profile),
    ):
        if not isinstance(profile, Mapping):
            raise ValueError(f"{label} axis-image profile is missing")
    return frozen


def protocol125_two_parent_input_hashes(inputs):
    """Hash the exact, structurally valid input inventory consumed by scoring."""
    _validate_inputs(inputs)
    hashes = {
        "N0_position_state_sha256": _state_fingerprint(
            inputs.n0_position_state, "N0 position state",
        ),
        "N1_position_state_sha256": _state_fingerprint(
            inputs.n1_position_state, "N1 position state",
        ),
        "N0_acceleration_state_sha256": _state_fingerprint(
            inputs.n0_acceleration_state, "N0 acceleration state",
        ),
        "N1_acceleration_state_sha256": _state_fingerprint(
            inputs.n1_acceleration_state, "N1 acceleration state",
        ),
        "N0_native_completion_evidence_sha256": _fingerprint_tree(
            inputs.n0_native_completion_evidence,
            root="N0_native_completion_evidence",
        ),
        "N1_native_completion_evidence_sha256": _fingerprint_tree(
            inputs.n1_native_completion_evidence,
            root="N1_native_completion_evidence",
        ),
        "N0_construction_provenance_sha256": _fingerprint_tree(
            inputs.n0_construction_provenance,
            root="N0_construction_provenance",
        ),
        "N1_construction_provenance_sha256": _fingerprint_tree(
            inputs.n1_construction_provenance,
            root="N1_construction_provenance",
        ),
        "v_meshes_sha256": _fingerprint_tree(inputs.v_meshes, root="v_meshes"),
        "dense_wall_r_sha256": _fingerprint_tree(
            inputs.dense_wall_r, root="dense_wall_r",
        ),
        "N0_bulk_audit_sha256": _fingerprint_tree(
            inputs.n0_bulk_audit, root="N0_bulk_audit",
        ),
        "N1_bulk_audit_sha256": _fingerprint_tree(
            inputs.n1_bulk_audit, root="N1_bulk_audit",
        ),
        "N0_correction_profile_sha256": _fingerprint_tree(
            inputs.n0_correction_profile, root="N0_correction_profile",
        ),
        "N1_correction_profile_sha256": _fingerprint_tree(
            inputs.n1_correction_profile, root="N1_correction_profile",
        ),
        "N0_position_V2_sha256": _fingerprint_tree(
            inputs.n0_position_v2, root="N0_position_V2",
        ),
        "N1_position_V2_sha256": _fingerprint_tree(
            inputs.n1_position_v2, root="N1_position_V2",
        ),
        "N0_hzz_zz_V2_sha256": _fingerprint_tree(
            inputs.n0_hzz_zz_v2, root="N0_hzz_zz_V2",
        ),
        "N1_hzz_zz_V2_sha256": _fingerprint_tree(
            inputs.n1_hzz_zz_v2, root="N1_hzz_zz_V2",
        ),
        "N0_a_hzz_V2_sha256": _fingerprint_tree(
            inputs.n0_a_hzz_v2, root="N0_a_hzz_V2",
        ),
        "N1_a_hzz_V2_sha256": _fingerprint_tree(
            inputs.n1_a_hzz_v2, root="N1_a_hzz_V2",
        ),
        "N0_axis_image_profile_sha256": _fingerprint_tree(
            inputs.n0_axis_image_profile, root="N0_axis_image_profile",
        ),
        "N1_axis_image_profile_sha256": _fingerprint_tree(
            inputs.n1_axis_image_profile, root="N1_axis_image_profile",
        ),
    }
    if tuple(hashes) != INPUT_HASH_KEYS or not all(
        _valid_sha256(value) for value in hashes.values()
    ):
        raise RuntimeError("two-parent input hash inventory is incomplete")
    return MappingProxyType(hashes)


def _immutable_array(value):
    array = np.asarray(value)
    if array.flags.c_contiguous and not array.flags.writeable:
        return array
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )


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


def _ledger_record(group_name, lanes, passed, identities, hashes_before, hashes_after):
    payload = {
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "group_name": str(group_name),
        "complete": True,
        "provenance_valid": True,
        "passed": bool(passed),
        "parent_identities": dict(identities),
        "required_lane_order": tuple(lanes),
        "lanes": lanes,
        "input_hashes_before": dict(hashes_before),
        "input_hashes_after": dict(hashes_after),
        "inputs_stable_while_scoring": True,
        "constituent_logical_AND": True,
        "scientific_execution_authorized": False,
        "artifact_written": False,
    }
    fingerprint = _fingerprint_tree(payload, root=group_name)
    return _freeze({**payload, "fingerprint": fingerprint})


def compose_protocol125_two_parent_records(inputs, *, parent_identities):
    """Return the two complete, ledger-ready Protocol-125 comparison records."""
    identities = _parent_identities(parent_identities)
    frozen = _validate_inputs(inputs, parent_identities=identities)
    hashes_before = protocol125_two_parent_input_hashes(inputs)

    representation_lanes = {}
    representation_lanes["state_spatial"] = score_state_pair_on_v_meshes(
        inputs.n0_position_state,
        inputs.n1_position_state,
        inputs.v_meshes,
        comparison_kind="N0_N1",
        groups=("position", "first_spatial", "second_spatial"),
    )
    representation_lanes["position_q4_q5_derivative_images"] = (
        score_q4_q5_derivative_images_on_v_meshes(
            inputs.n0_position_state,
            inputs.n1_position_state,
            inputs.v_meshes,
            comparison_kind="N0_N1",
            state_name="position",
        )
    )
    representation_lanes["acceleration"] = score_acceleration_pair_on_v_meshes(
        inputs.n0_acceleration_state,
        inputs.n1_acceleration_state,
        inputs.v_meshes,
        comparison_kind="N0_N1",
    )
    representation_lanes["acceleration_q4_q5_derivative_images"] = (
        score_q4_q5_derivative_images_on_v_meshes(
            inputs.n0_acceleration_state,
            inputs.n1_acceleration_state,
            inputs.v_meshes,
            comparison_kind="N0_N1",
            state_name="acceleration",
        )
    )
    common_v2 = compare_protocol125_common_v2(
        inputs.n0_bulk_audit, inputs.n1_bulk_audit,
    )
    representation_lanes["bulk_common_V2_nonworsening"] = MappingProxyType({
        **dict(common_v2),
        "passed": bool(common_v2["protocol_common_V2_pass"]),
    })
    representation_lanes["native_completion_correction_refinement"] = (
        _score_native_completion_correction_refinement(inputs, identities)
    )
    if tuple(representation_lanes) != REPRESENTATION_LANE_ORDER:
        raise RuntimeError("N0/N1 representation lane inventory is incomplete")

    correction = adjudicate_correction_refinement(
        inputs.n0_correction_profile,
        inputs.n1_correction_profile,
        inputs.n0_position_v2,
        inputs.n1_position_v2,
        frozen["V2"]["r"],
        hzz_zz_n0=inputs.n0_hzz_zz_v2,
        hzz_zz_n1=inputs.n1_hzz_zz_v2,
        a_hzz_n0=inputs.n0_a_hzz_v2,
        a_hzz_n1=inputs.n1_a_hzz_v2,
        axis_image_n0=inputs.n0_axis_image_profile,
        axis_image_n1=inputs.n1_axis_image_profile,
    )
    dense_gate_names = (
        "N0_full_physical_Linf",
        "N0_full_physical_RMS",
        "N0_not_order_one",
        "N1_full_physical_Linf",
        "N1_full_physical_RMS",
        "N1_not_order_one",
        "hzz_refinement",
    )
    correction_lanes = {}
    correction_lanes["dense_wall_correction_refinement"] = {
        "refinement": correction["refinement"],
        "gates": {
            name: bool(correction["gates"][name]) for name in dense_gate_names
        },
        "passed": bool(all(correction["gates"][name] for name in dense_gate_names)),
    }
    correction_lanes["V2_hzz_zz_difference"] = {
        "scaled_Linf_difference": correction["hzz_zz_scaled_Linf_difference"],
        "ceiling": 2e-3,
        "passed": bool(correction["gates"]["hzz_zz_difference"]),
    }
    correction_lanes["V2_a_hzz_difference"] = {
        "scaled_Linf_difference": correction["a_hzz_scaled_Linf_difference"],
        "ceiling": 2e-3,
        "passed": bool(correction["gates"]["a_hzz_difference"]),
    }
    correction_lanes["V2_axis_acceleration_derivative_images"] = {
        "comparison": correction["axis_acceleration_derivative_images"],
        "passed": bool(correction["gates"]["q4_q5_axis_acceleration_images"]),
    }
    if tuple(correction_lanes) != CORRECTION_REFINEMENT_LANE_ORDER:
        raise RuntimeError("correction-refinement lane inventory is incomplete")

    hashes_after = protocol125_two_parent_input_hashes(inputs)
    changed = tuple(
        name for name in INPUT_HASH_KEYS
        if str(hashes_before[name]) != str(hashes_after[name])
    )
    if changed:
        raise ValueError(f"two-parent inputs changed while scoring: {changed}")

    representation_pass = bool(all(
        representation_lanes[name]["passed"] for name in REPRESENTATION_LANE_ORDER
    ))
    correction_pass = bool(
        correction["pass"]
        and all(correction_lanes[name]["passed"] for name in CORRECTION_REFINEMENT_LANE_ORDER)
    )
    records = {
        "N0_N1_representation": _ledger_record(
            "N0_N1_representation",
            representation_lanes,
            representation_pass,
            identities,
            hashes_before,
            hashes_after,
        ),
        "correction_refinement": _ledger_record(
            "correction_refinement",
            correction_lanes,
            correction_pass,
            identities,
            hashes_before,
            hashes_after,
        ),
    }
    if tuple(records) != TWO_PARENT_RECORD_ORDER:
        raise RuntimeError("two-parent gate record inventory is incomplete")
    return MappingProxyType(records)
