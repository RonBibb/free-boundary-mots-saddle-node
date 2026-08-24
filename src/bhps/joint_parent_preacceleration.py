"""Pure Protocol-125 pre-acceleration gate composition.

This module consumes only already constructed, in-memory position data and
already produced evidence.  It does not construct or repair a parent, solve
an equation, form an acceleration, write an artifact, or authorize scientific
execution.  Every returned parent gate has the exact core required by
``Protocol125GateLedger`` and is bound to one parent label and identity.

The input bundle is sealed before scoring.  Missing or changed provenance,
an absent required evidence lane, or an exception in an independent scorer
produces an ``INVALID-audit`` result.  Unreached gates are nevertheless
emitted as explicit fail-closed records so a caller cannot accidentally omit
them from a ledger.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from bhps.joint_parent_adjudication import (
    V_MESH_NAMES,
    score_construction_reload_provenance,
    score_group_arrays,
    score_precomputed_groups_on_v_meshes,
    score_q4_q5_derivative_images_on_v_meshes,
    score_sampling_order,
    score_state_pair_on_v_meshes,
)
from bhps.joint_parent_bulk_reference import (
    FiniteWallReferenceHermitePair,
    REFERENCE_CHANNEL_ORDER,
)
from bhps.joint_parent_bulk_validation import (
    ALL_LANES as BULK_LANES,
    PROTOCOL_IDENTIFIER as BULK_PROTOCOL_IDENTIFIER,
)
from bhps.joint_parent_construction import (
    validate_protocol125_construction_failure_record,
    validate_protocol125_successful_parent_provenance_record,
)
from bhps.joint_parent_endpoint_audits import (
    score_state_endpoint_z_reproduction,
    score_state_outer_derivative_reproduction,
)
from bhps.joint_parent_position_audits import (
    bind_protocol125_position_audit_meshes,
    evaluate_protocol125_dense_outer_delta_robin_audit,
    evaluate_protocol125_dense_wall_audit,
    evaluate_protocol125_signature_union,
)
from bhps.joint_parent_position_pair import PositionOnlyConstrainedHermitePair
from bhps.joint_parent_representation import (
    COORDINATE_COMPONENT_ORDER,
    SEALED_ADVERSE_COMPARATOR_NAMES,
    validate_protocol125_representation_coefficient_failure,
)
from bhps.joint_parent_refinement_diagnostics import (
    frozen_validation_meshes,
    reduced_to_physical,
)
from bhps.matched_staged_continuum import hash_arrays
from bhps.regular_so3_gh_reduction import FIELD_ORDER as REDUCED_FIELD_ORDER


PROTOCOL_IDENTIFIER = "Protocol-125-pre-acceleration-composer-v1"
NATIVE_EVIDENCE_PROTOCOL_IDENTIFIER = (
    "Protocol-125-native-position-tangent-evidence-v1"
)
PRE_ACCELERATION_GROUPS = (
    "pre_acceleration_construction",
    "native_position_tangent",
    "position_representation",
    "dense_boundary_audit",
    "signature_union",
    "legacy_holdout",
    "sampling_order",
    "bulk_prerequisite",
)
POSITION_PREFIX_GROUPS = PRE_ACCELERATION_GROUPS[:5]
REPRESENTATION_PREFIX_GROUPS = PRE_ACCELERATION_GROUPS[:7]
NATIVE_POSITION_TANGENT_LANES = (
    "native_position_completion",
    "source_node_geometry_signature",
    "positive_zero_time_symmetric_tangent",
    "source_node_wall_rows",
    "source_node_ownership",
    "source_reload_and_hashes",
)
LEGACY_REQUIRED_GROUPS = (
    "position",
    "first_spatial",
    "second_spatial",
)
INPUT_HASH_KEYS = (
    "parent_mapping_sha256",
    "position_pair_sha256",
    "reference_pair_sha256",
    "construction_provenance_sha256",
    "native_position_tangent_evidence_sha256",
    "legacy_Q33_sha256",
    "legacy_Q55_sha256",
    "legacy_component_orders_sha256",
    "bulk_validation_audit_sha256",
    "frozen_validation_meshes_sha256",
)
CORE_INPUT_HASH_KEYS = INPUT_HASH_KEYS[:5] + (
    "frozen_validation_meshes_sha256",
)
REPRESENTATION_INPUT_HASH_KEYS = INPUT_HASH_KEYS[:8] + (
    "frozen_validation_meshes_sha256",
)
PROVENANCE_KEYS = (
    "protocol_identifier",
    "parent_label",
    "parent_identity",
    "required_group_order",
    "legacy_comparator_names",
    "input_hashes",
)
STAGED_PROVENANCE_KEYS = (
    "protocol_identifier",
    "stage_name",
    "parent_label",
    "parent_identity",
    "group_order",
    "input_hashes",
    "prior_stage_fingerprint",
    "fingerprint",
)
STAGED_RESULT_KEYS = (
    "protocol_identifier",
    "stage_name",
    "parent_label",
    "parent_identity",
    "group_order",
    "reached_group_order",
    "groups",
    "input_hashes_before",
    "input_hashes_after",
    "prior_stage_fingerprint",
    "complete",
    "provenance_valid",
    "passed",
    "first_failure",
    "fingerprint",
)
_GATE_CORE = ("complete", "provenance_valid", "passed", "fingerprint")


@dataclass(frozen=True)
class Protocol125PreAccelerationCoreInputs:
    """Inputs available before any legacy, sampling, or bulk job runs."""

    parent_mapping: Mapping
    position_pair: object
    reference_pair: object
    construction_provenance: Mapping
    native_position_tangent_evidence: Mapping


@dataclass(frozen=True)
class Protocol125PreAccelerationRepresentationInputs(
    Protocol125PreAccelerationCoreInputs,
):
    """Core inputs plus the prospectively sealed legacy holdout arrays."""

    legacy_Q33_by_mesh: Mapping
    legacy_Q55_by_mesh: Mapping
    legacy_component_orders: Mapping


@dataclass(frozen=True)
class Protocol125PreAccelerationInputs(
    Protocol125PreAccelerationRepresentationInputs,
):
    """Complete inputs after every representation prerequisite has passed."""

    bulk_validation_audit: Mapping


def _valid_sha256(value):
    value = str(value)
    return bool(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _bitwise_equal(left, right):
    left = np.ascontiguousarray(np.asarray(left))
    right = np.ascontiguousarray(np.asarray(right))
    return bool(
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes() == right.tobytes()
    )


def _hash_tree(value, *, root):
    """Hash nested array/scalar records without serializing or writing them."""
    digest = hashlib.sha256()

    def token(label):
        encoded = str(label).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)

    def visit(item, path):
        if isinstance(item, Mapping):
            token(f"mapping:{path}:{len(item)}")
            if any(not isinstance(name, str) or not name for name in item):
                raise ValueError(f"hashed mapping {path} has an invalid key")
            for name in sorted(item):
                token(f"key:{name}")
                visit(item[name], f"{path}/{name}")
            return
        if isinstance(item, (tuple, list)):
            token(f"sequence:{path}:{len(item)}")
            for index, child in enumerate(item):
                visit(child, f"{path}/{index}")
            return
        if item is None:
            token(f"none:{path}")
            return
        if isinstance(item, (str, bytes)):
            token(f"scalar:{path}:{type(item).__name__}")
            payload = item.encode("utf-8") if isinstance(item, str) else item
            digest.update(len(payload).to_bytes(8, "little"))
            digest.update(payload)
            return
        if isinstance(item, (bool, int, float, np.generic)) and not isinstance(
            item, np.ndarray,
        ):
            array = np.ascontiguousarray(np.asarray(item))
        else:
            array = np.ascontiguousarray(np.asarray(item))
        if array.dtype == object:
            raise ValueError(f"hashed input {path} has object dtype")
        if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
            raise ValueError(f"hashed input {path} is nonfinite")
        token(f"array:{path}")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())

    visit(value, str(root))
    return digest.hexdigest()


def _object_sha256(value, label):
    fingerprint = getattr(value, "fingerprint", None)
    if fingerprint is None or not callable(fingerprint):
        raise ValueError(f"{label} exposes no immutable fingerprint")
    found = str(fingerprint())
    if not _valid_sha256(found):
        raise ValueError(f"{label} fingerprint is missing or invalid")
    return found


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({str(name): _freeze(item) for name, item in value.items()})
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if not array.flags.writeable:
            return array
        return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _parent_context(inputs):
    if not isinstance(inputs, Protocol125PreAccelerationCoreInputs):
        raise TypeError(
            "pre-acceleration composition requires its data-only core bundle"
        )
    parent = inputs.parent_mapping
    if not isinstance(parent, Mapping):
        raise TypeError("pre-acceleration parent must be an explicit mapping")
    forbidden = (
        "acceleration",
        "bulk_acceleration",
        "compatible_acceleration",
        "source_second_time",
    )
    if any(name in parent for name in forbidden):
        raise ValueError("pre-acceleration parent contains a forbidden acceleration lane")
    required = (
        "label", "parent_identity", "z", "r", "position", "selector_q",
        "phi", "reference_q", "reference_phi",
    )
    if any(name not in parent for name in required):
        raise ValueError("pre-acceleration parent mapping is incomplete")
    label = str(parent["label"])
    identity = str(parent["parent_identity"])
    if label not in ("N0", "N1"):
        raise ValueError("pre-acceleration parent label must be N0 or N1")
    if not _valid_sha256(identity):
        raise ValueError("pre-acceleration parent identity is not a SHA-256 digest")
    z = np.asarray(parent["z"], dtype=float)
    r = np.asarray(parent["r"], dtype=float)
    position = np.asarray(parent["position"], dtype=float)
    shape = (len(z), len(r))
    scalar_arrays = {
        name: np.asarray(parent[name], dtype=float)
        for name in ("selector_q", "phi", "reference_q", "reference_phi")
    }
    if (
        z.ndim != 1
        or r.ndim != 1
        or len(z) < 2
        or len(r) < 2
        or np.any(np.diff(z) <= 0.0)
        or np.any(np.diff(r) <= 0.0)
        or r[0] != 0.0
        or position.shape != (*shape, 9)
        or any(value.shape != shape for value in scalar_arrays.values())
        or not all(np.all(np.isfinite(value)) for value in (
            z, r, position, *scalar_arrays.values(),
        ))
    ):
        raise ValueError("pre-acceleration parent arrays are invalid")
    reproduced = hash_arrays(
        np.asarray(label),
        z,
        r,
        position,
        scalar_arrays["selector_q"],
        scalar_arrays["phi"],
        scalar_arrays["reference_q"],
        scalar_arrays["reference_phi"],
    )
    if reproduced != identity:
        raise ValueError("pre-acceleration parent identity does not reproduce")
    return MappingProxyType({
        "label": label,
        "identity": identity,
        "z": z,
        "r": r,
        "position": position,
        **scalar_arrays,
        "source_coordinate_sha256": hash_arrays(z, r),
        "position_sha256": hash_arrays(position),
    })


def _v_meshes():
    frozen = frozen_validation_meshes()
    return MappingProxyType({name: frozen[name] for name in V_MESH_NAMES})


def _core_input_hashes(inputs):
    context = _parent_context(inputs)
    frozen = frozen_validation_meshes()
    hashes = {
        "parent_mapping_sha256": _hash_tree(
            inputs.parent_mapping, root="parent_mapping",
        ),
        "position_pair_sha256": _object_sha256(
            inputs.position_pair, "position-only pair",
        ),
        "reference_pair_sha256": _object_sha256(
            inputs.reference_pair, "finite-wall reference pair",
        ),
        "construction_provenance_sha256": _hash_tree(
            inputs.construction_provenance, root="construction_provenance",
        ),
        "native_position_tangent_evidence_sha256": _hash_tree(
            inputs.native_position_tangent_evidence,
            root="native_position_tangent_evidence",
        ),
        "frozen_validation_meshes_sha256": _hash_tree(
            frozen, root="frozen_validation_meshes",
        ),
    }
    if tuple(hashes) != CORE_INPUT_HASH_KEYS or not all(
        _valid_sha256(value) for value in hashes.values()
    ):
        raise RuntimeError("pre-acceleration core hash inventory is incomplete")
    return context, MappingProxyType(hashes)


def _representation_input_hashes(inputs):
    if not isinstance(inputs, Protocol125PreAccelerationRepresentationInputs):
        raise TypeError("legacy/sampling stage requires representation inputs")
    context, core = _core_input_hashes(inputs)
    hashes = {
        **{
            name: core[name]
            for name in CORE_INPUT_HASH_KEYS
            if name != "frozen_validation_meshes_sha256"
        },
        "legacy_Q33_sha256": _hash_tree(
            inputs.legacy_Q33_by_mesh, root=SEALED_ADVERSE_COMPARATOR_NAMES[0],
        ),
        "legacy_Q55_sha256": _hash_tree(
            inputs.legacy_Q55_by_mesh, root=SEALED_ADVERSE_COMPARATOR_NAMES[1],
        ),
        "legacy_component_orders_sha256": _hash_tree(
            inputs.legacy_component_orders, root="legacy_component_orders",
        ),
        "frozen_validation_meshes_sha256": core[
            "frozen_validation_meshes_sha256"
        ],
    }
    if tuple(hashes) != REPRESENTATION_INPUT_HASH_KEYS or not all(
        _valid_sha256(value) for value in hashes.values()
    ):
        raise RuntimeError(
            "pre-acceleration representation hash inventory is incomplete"
        )
    return context, MappingProxyType(hashes)


def _input_hashes(inputs):
    if not isinstance(inputs, Protocol125PreAccelerationInputs):
        raise TypeError("bulk stage requires the complete pre-acceleration inputs")
    context, representation = _representation_input_hashes(inputs)
    hashes = {
        **{
            name: representation[name]
            for name in REPRESENTATION_INPUT_HASH_KEYS
            if name != "frozen_validation_meshes_sha256"
        },
        "bulk_validation_audit_sha256": _hash_tree(
            inputs.bulk_validation_audit, root="bulk_validation_audit",
        ),
        "frozen_validation_meshes_sha256": representation[
            "frozen_validation_meshes_sha256"
        ],
    }
    if tuple(hashes) != INPUT_HASH_KEYS or not all(
        _valid_sha256(value) for value in hashes.values()
    ):
        raise RuntimeError("pre-acceleration input hash inventory is incomplete")
    return context, MappingProxyType(hashes)


def capture_protocol125_preacceleration_provenance(inputs):
    """Seal all explicit inputs before any prerequisite scorer is called."""
    context, hashes = _input_hashes(inputs)
    return MappingProxyType({
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "parent_label": context["label"],
        "parent_identity": context["identity"],
        "required_group_order": PRE_ACCELERATION_GROUPS,
        "legacy_comparator_names": tuple(SEALED_ADVERSE_COMPARATOR_NAMES),
        "input_hashes": hashes,
    })


def _validate_provenance(provenance, context, found_hashes):
    if not isinstance(provenance, Mapping) or set(provenance) != set(PROVENANCE_KEYS):
        raise ValueError("pre-acceleration provenance record is missing or incomplete")
    if str(provenance["protocol_identifier"]) != PROTOCOL_IDENTIFIER:
        raise ValueError("pre-acceleration protocol identifier differs")
    if str(provenance["parent_label"]) != context["label"]:
        raise ValueError("pre-acceleration parent label provenance differs")
    if str(provenance["parent_identity"]) != context["identity"]:
        raise ValueError("pre-acceleration parent identity provenance differs")
    if tuple(provenance["required_group_order"]) != PRE_ACCELERATION_GROUPS:
        raise ValueError("pre-acceleration required gate inventory differs")
    if tuple(provenance["legacy_comparator_names"]) != tuple(
        SEALED_ADVERSE_COMPARATOR_NAMES
    ):
        raise ValueError("sealed legacy comparator identities differ")
    expected = provenance["input_hashes"]
    if not isinstance(expected, Mapping) or tuple(expected) != INPUT_HASH_KEYS:
        raise ValueError("pre-acceleration provenance hash inventory differs")
    changed = tuple(
        name for name in INPUT_HASH_KEYS
        if not _valid_sha256(expected[name])
        or str(expected[name]) != str(found_hashes[name])
    )
    if changed:
        raise ValueError(f"pre-acceleration input hash mismatch: {changed}")


def _gate_record(
    group_name,
    context,
    *,
    complete,
    provenance_valid,
    passed,
    details,
    invalid_reasons=(),
    not_reached=False,
    blocked_by=None,
):
    complete = bool(complete)
    provenance_valid = bool(provenance_valid)
    passed = bool(passed and complete and provenance_valid)
    reasons = tuple(str(reason) for reason in invalid_reasons)
    if type(not_reached) is not bool:
        raise TypeError("gate not_reached marker must be a bool")
    if not_reached:
        if (
            not complete
            or not provenance_valid
            or passed
            or str(blocked_by) not in PRE_ACCELERATION_GROUPS
        ):
            raise ValueError("ordered not-reached gate metadata is inconsistent")
        blocked_by = str(blocked_by)
    elif blocked_by is not None:
        raise ValueError("a reached gate cannot carry a blocked_by marker")
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
    if not_reached:
        payload.update({
            "not_reached": True,
            "blocked_by": blocked_by,
        })
    fingerprint = _hash_tree(payload, root=f"gate/{group_name}")
    output = {
        "complete": complete,
        "provenance_valid": provenance_valid,
        "passed": passed,
        "fingerprint": fingerprint,
        "parent_label": context["label"],
        "parent_identity": context["identity"],
        "group_name": str(group_name),
        "invalid_reasons": reasons,
        "details": frozen_details,
    }
    if not_reached:
        output.update({
            "not_reached": True,
            "blocked_by": blocked_by,
        })
    return MappingProxyType(output)


def _invalid_gate(group_name, context, reason, *, prerequisite=None):
    details = {
        "scorer_executed": False,
        "failure_kind": "invalid-or-unreached-prerequisite",
    }
    if prerequisite is not None:
        details["blocked_by"] = str(prerequisite)
    return _gate_record(
        group_name,
        context,
        complete=False,
        provenance_valid=False,
        passed=False,
        details=details,
        invalid_reasons=(reason,),
    )


def _ordered_not_reached_gate(group_name, context, blocked_by):
    return _gate_record(
        group_name,
        context,
        complete=True,
        provenance_valid=True,
        passed=False,
        details={
            "scorer_executed": False,
            "failure_kind": "ordered-scientific-stop",
            "blocked_by": str(blocked_by),
        },
        not_reached=True,
        blocked_by=blocked_by,
    )


def _validate_position_bindings(inputs, context):
    pair = inputs.position_pair
    reference = inputs.reference_pair
    if not isinstance(pair, PositionOnlyConstrainedHermitePair):
        raise TypeError("pre-acceleration position input must be the position-only pair")
    if not isinstance(reference, FiniteWallReferenceHermitePair):
        raise TypeError("pre-acceleration reference input must be the finite-wall pair")
    for state in (pair.primary, pair.comparator):
        if not (
            _bitwise_equal(state.source_z, context["z"])
            and _bitwise_equal(state.source_r, context["r"])
        ):
            raise ValueError("position pair source coordinates differ from the parent")
    for member in (reference.primary, reference.comparator):
        if not (
            _bitwise_equal(member.source_z, context["z"])
            and _bitwise_equal(member.source_r, context["r"])
            and tuple(member.channel_order) == tuple(REFERENCE_CHANNEL_ORDER)
        ):
            raise ValueError("finite-wall reference source identity differs")
        expected = np.stack((
            context["reference_q"], context["reference_phi"],
        ), axis=-1)
        if not _bitwise_equal(member.source_values, expected):
            raise ValueError("finite-wall reference arrays differ from the parent")
    return pair, reference


def _score_construction_group(inputs, context):
    pair, reference = _validate_position_bindings(inputs, context)
    record = inputs.construction_provenance
    if not isinstance(record, Mapping) or str(record.get("parent_identity", "")) != context[
        "identity"
    ]:
        raise ValueError("construction provenance omits the bound parent identity")
    score = score_construction_reload_provenance(record, reference, pair)
    provenance_valid = bool(
        score["provenance_valid"]
        and score["gates"]["immutable_inputs"]
        and score["gates"]["physical_normalization_identifier"]
        and score["gates"]["branch_identifier"]
        and score["gates"]["parent_label"]
    )
    return _gate_record(
        "pre_acceleration_construction",
        context,
        complete=True,
        provenance_valid=provenance_valid,
        passed=bool(score["passed"]),
        details={
            "scorer": "score_construction_reload_provenance",
            "score": score,
            "parent_identity_reproduced": True,
            "position_and_reference_bound_to_parent": True,
        },
    )


def _score_native_group(inputs, context):
    evidence = inputs.native_position_tangent_evidence
    required = (
        "protocol_identifier", "parent_label", "parent_identity",
        "source_coordinate_sha256", "position_sha256", "lanes",
    )
    if not isinstance(evidence, Mapping) or any(name not in evidence for name in required):
        raise ValueError("native position/tangent evidence provenance is incomplete")
    if str(evidence["protocol_identifier"]) != NATIVE_EVIDENCE_PROTOCOL_IDENTIFIER:
        raise ValueError("native position/tangent evidence protocol differs")
    if not (
        str(evidence["parent_label"]) == context["label"]
        and str(evidence["parent_identity"]) == context["identity"]
        and str(evidence["source_coordinate_sha256"])
        == context["source_coordinate_sha256"]
        and str(evidence["position_sha256"]) == context["position_sha256"]
    ):
        raise ValueError("native position/tangent evidence identity differs")
    lanes = evidence["lanes"]
    if not isinstance(lanes, Mapping) or tuple(lanes) != NATIVE_POSITION_TANGENT_LANES:
        raise ValueError("native position/tangent evidence lane inventory differs")
    summaries = {}
    for name in NATIVE_POSITION_TANGENT_LANES:
        lane = lanes[name]
        if not isinstance(lane, Mapping) or any(field not in lane for field in _GATE_CORE):
            raise ValueError(f"native position/tangent lane {name} is incomplete")
        for field in ("complete", "provenance_valid", "passed"):
            if type(lane[field]) is not bool:
                raise TypeError(f"native position/tangent lane {name} {field} is not bool")
        if not _valid_sha256(lane["fingerprint"]):
            raise ValueError(f"native position/tangent lane {name} fingerprint is invalid")
        summaries[name] = {
            "complete": lane["complete"],
            "provenance_valid": lane["provenance_valid"],
            "passed": lane["passed"],
            "fingerprint": str(lane["fingerprint"]),
        }
    complete = all(lane["complete"] for lane in summaries.values())
    provenance_valid = all(
        lane["provenance_valid"] for lane in summaries.values()
    )
    passed = all(lane["passed"] for lane in summaries.values())
    return _gate_record(
        "native_position_tangent",
        context,
        complete=complete,
        provenance_valid=provenance_valid,
        passed=passed,
        details={
            "evidence_protocol_identifier": NATIVE_EVIDENCE_PROTOCOL_IDENTIFIER,
            "required_lane_order": NATIVE_POSITION_TANGENT_LANES,
            "lanes": summaries,
            "constituent_logical_AND": True,
        },
    )


def _last_axis_constituent_reports(error, component_order):
    error = np.asarray(error, dtype=float)
    component_order = tuple(component_order)
    if (
        error.ndim < 1
        or error.shape[-1] != len(component_order)
        or not np.all(np.isfinite(error))
    ):
        raise ValueError("source reproduction constituent array is invalid")
    records = {}
    for index, name in enumerate(component_order):
        lane = error[..., index]
        records[name] = {
            "E_inf": float(np.max(lane)),
            "E_RMS": float(np.sqrt(np.mean(lane**2))),
            "sample_count": int(lane.size),
        }
    return MappingProxyType({
        "component_order": component_order,
        "records": MappingProxyType(records),
    })


def _source_reproduction_score(state, context):
    reduced_values = np.asarray(
        state.evaluate_reduced(context["z"], context["r"]), dtype=float,
    )
    physical_values = np.asarray(
        state.evaluate_coordinate_components(context["z"], context["r"]),
        dtype=float,
    )
    physical_target = reduced_to_physical(context["position"], context["r"])
    physical_score = score_group_arrays(
        physical_values,
        physical_target,
        component_order=tuple(COORDINATE_COMPONENT_ORDER),
    )
    reduced_score = score_group_arrays(
        reduced_values,
        context["position"],
        component_order=tuple(REDUCED_FIELD_ORDER),
    )
    return MappingProxyType({
        **dict(physical_score),
        "comparison_group": "physical_coordinate_components",
        "constituents": _last_axis_constituent_reports(
            physical_score["elementwise_error"], COORDINATE_COMPONENT_ORDER,
        ),
        "reduced_coefficients_diagnostic": MappingProxyType({
            **dict(reduced_score),
            "constituents": _last_axis_constituent_reports(
                reduced_score["elementwise_error"], REDUCED_FIELD_ORDER,
            ),
            "gated": False,
        }),
        "ceiling": 1e-12,
        "passed": bool(physical_score["E_inf"] <= 1e-12),
    })


def _score_position_group(inputs, context):
    pair, _ = _validate_position_bindings(inputs, context)
    v_score = score_state_pair_on_v_meshes(
        pair.primary,
        pair.comparator,
        _v_meshes(),
        comparison_kind="Q53_Q33",
        groups=LEGACY_REQUIRED_GROUPS,
    )
    image_score = score_q4_q5_derivative_images_on_v_meshes(
        pair.primary,
        pair.comparator,
        _v_meshes(),
        comparison_kind="Q53_Q33",
        state_name="position",
    )
    source = {
        "Q53": _source_reproduction_score(pair.primary, context),
        "Q33": _source_reproduction_score(pair.comparator, context),
    }
    return _gate_record(
        "position_representation",
        context,
        complete=True,
        provenance_valid=True,
        passed=bool(
            v_score["passed"]
            and image_score["passed"]
            and all(item["passed"] for item in source.values())
        ),
        details={
            "Q53_Q33_V_meshes": v_score,
            "Q53_Q33_position_q4_q5_images": image_score,
            "source_node_reproduction": source,
            "q4_q5_physical_derivative_images_gated_pre_acceleration": True,
            "no_acceleration_lane_consumed": True,
        },
    )


def _score_dense_boundary_group(inputs, context):
    pair, reference = _validate_position_bindings(inputs, context)
    meshes = bind_protocol125_position_audit_meshes(pair)
    dense_wall = evaluate_protocol125_dense_wall_audit(pair, meshes)
    dense_outer = evaluate_protocol125_dense_outer_delta_robin_audit(
        pair, reference, meshes,
    )
    compact_endpoint = {
        "Q53": score_state_endpoint_z_reproduction(
            pair.primary, meshes.dense_wall_r,
        ),
        "Q33": score_state_endpoint_z_reproduction(
            pair.comparator, meshes.dense_wall_r,
        ),
    }
    outer_replay = {
        "Q53": score_state_outer_derivative_reproduction(
            pair.primary, meshes.dense_outer_z,
        ),
        "Q33": score_state_outer_derivative_reproduction(
            pair.comparator, meshes.dense_outer_z,
        ),
    }
    constituent = (
        dense_wall,
        dense_outer,
        *compact_endpoint.values(),
        *outer_replay.values(),
    )
    return _gate_record(
        "dense_boundary_audit",
        context,
        complete=True,
        provenance_valid=True,
        passed=all(record["passed"] for record in constituent),
        details={
            "independent_dense_wall": dense_wall,
            "independent_dense_outer_delta_Robin": dense_outer,
            "compact_endpoint_z_reproduction": compact_endpoint,
            "all_channel_outer_derivative_replay": outer_replay,
            "constituent_logical_AND": True,
        },
    )


def _score_signature_group(inputs, context):
    pair, _ = _validate_position_bindings(inputs, context)
    meshes = bind_protocol125_position_audit_meshes(pair)
    score = evaluate_protocol125_signature_union(pair, meshes)
    return _gate_record(
        "signature_union",
        context,
        complete=True,
        provenance_valid=True,
        passed=bool(score["passed"]),
        details={"signature_union": score},
    )


def _score_bulk_group(inputs, context):
    pair, reference = _validate_position_bindings(inputs, context)
    audit = inputs.bulk_validation_audit
    if not isinstance(audit, Mapping):
        raise TypeError("bulk-validation audit must be an explicit mapping")
    required = (
        "protocol", "parent_label", "identity", "lanes", "adjudication",
        "scientific_artifact_written", "acceleration_authorized",
    )
    if any(name not in audit for name in required):
        raise ValueError("bulk-validation audit is incomplete")
    identity = audit["identity"]
    lanes = audit["lanes"]
    adjudication = audit["adjudication"]
    if not isinstance(identity, Mapping) or not isinstance(adjudication, Mapping):
        raise ValueError("bulk-validation identity or adjudication is missing")
    identity_required = (
        "parent_label", "binding_sha256", "source_coordinate_sha256",
        "candidate_source_fingerprint", "candidate_endpoint_fingerprint",
        "candidate_pair_fingerprint", "reference_pair_fingerprint",
    )
    if any(name not in identity for name in identity_required):
        raise ValueError("bulk-validation binding provenance is incomplete")
    provenance_valid = bool(
        str(audit["protocol"]) == BULK_PROTOCOL_IDENTIFIER
        and str(audit["parent_label"]) == context["label"]
        and str(identity["parent_label"]) == context["label"]
        and _valid_sha256(identity["binding_sha256"])
        and str(identity["source_coordinate_sha256"])
        == context["source_coordinate_sha256"]
        and str(identity["candidate_source_fingerprint"])
        == str(pair.source_fingerprint)
        and str(identity["candidate_endpoint_fingerprint"])
        == str(pair.endpoint_fingerprint)
        and str(identity["candidate_pair_fingerprint"])
        == pair.fingerprint()
        and str(identity["reference_pair_fingerprint"])
        == reference.fingerprint()
        and audit["scientific_artifact_written"] is False
        and audit["acceleration_authorized"] is False
    )
    if not isinstance(lanes, Mapping) or tuple(lanes) != tuple(BULK_LANES):
        raise ValueError("bulk-validation lane inventory differs")
    if any(
        not isinstance(lanes[name], Mapping)
        or not isinstance(lanes[name].get("authoritative"), Mapping)
        for name in BULK_LANES
    ):
        raise ValueError("bulk-validation authoritative lane is missing")
    adjudication_required = (
        "lane_numerical_gates", "all_lane_numerical_gates_pass",
        "all_Q53_Q33_bulk_jet_gates_pass", "strip_layer_growth_gate",
        "parent_bulk_pass", "common_V2_two_parent_gate_required",
        "protocol_two_parent_bulk_pass", "fail_closed_pending_second_parent",
    )
    if any(name not in adjudication for name in adjudication_required):
        raise ValueError("bulk-validation adjudication is incomplete")
    lane_gates = adjudication["lane_numerical_gates"]
    strip = adjudication["strip_layer_growth_gate"]
    if (
        not isinstance(lane_gates, Mapping)
        or tuple(lane_gates) != tuple(BULK_LANES)
        or any(type(lane_gates[name]) is not bool for name in BULK_LANES)
        or not isinstance(strip, Mapping)
        or type(strip.get("pass")) is not bool
    ):
        raise ValueError("bulk-validation constituent gates are incomplete")
    for name in (
        "all_lane_numerical_gates_pass",
        "all_Q53_Q33_bulk_jet_gates_pass",
        "parent_bulk_pass",
        "common_V2_two_parent_gate_required",
        "protocol_two_parent_bulk_pass",
        "fail_closed_pending_second_parent",
    ):
        if type(adjudication[name]) is not bool:
            raise TypeError(f"bulk-validation adjudication {name} is not bool")
    lane_pass = all(lane_gates.values())
    expected = bool(
        lane_pass
        and adjudication["all_Q53_Q33_bulk_jet_gates_pass"]
        and strip["pass"]
    )
    provenance_valid = bool(
        provenance_valid
        and adjudication["all_lane_numerical_gates_pass"] == lane_pass
        and adjudication["parent_bulk_pass"] == expected
        and adjudication["common_V2_two_parent_gate_required"]
        and not adjudication["protocol_two_parent_bulk_pass"]
        and adjudication["fail_closed_pending_second_parent"]
    )
    return _gate_record(
        "bulk_prerequisite",
        context,
        complete=True,
        provenance_valid=provenance_valid,
        passed=bool(adjudication["parent_bulk_pass"]),
        details={
            "supplied_bulk_audit_sha256": _hash_tree(
                audit, root="supplied_bulk_validation_audit",
            ),
            "bulk_protocol_identifier": str(audit["protocol"]),
            "binding_sha256": str(identity["binding_sha256"]),
            "lane_numerical_gates": dict(lane_gates),
            "all_Q53_Q33_bulk_jet_gates_pass": bool(
                adjudication["all_Q53_Q33_bulk_jet_gates_pass"]
            ),
            "strip_layer_growth_pass": bool(strip["pass"]),
            "single_parent_bulk_pass": bool(adjudication["parent_bulk_pass"]),
            "common_V2_two_parent_gate_still_required": True,
            "audit_arrays_not_recomputed_or_mutated": True,
        },
    )


def _validate_legacy_arrays(inputs):
    meshes = _v_meshes()
    component_orders = inputs.legacy_component_orders
    if not isinstance(component_orders, Mapping) or tuple(component_orders) != LEGACY_REQUIRED_GROUPS:
        raise ValueError("legacy component orders are incomplete or reordered")
    for group in LEGACY_REQUIRED_GROUPS:
        order = tuple(component_orders[group])
        if not order or any(not isinstance(name, str) or not name for name in order):
            raise ValueError(f"legacy {group} component order is invalid")
    trailing_shapes = {}
    for label, collection in (
        (SEALED_ADVERSE_COMPARATOR_NAMES[0], inputs.legacy_Q33_by_mesh),
        (SEALED_ADVERSE_COMPARATOR_NAMES[1], inputs.legacy_Q55_by_mesh),
    ):
        if not isinstance(collection, Mapping) or tuple(collection) != V_MESH_NAMES:
            raise ValueError(f"{label} requires ordered V0/V1/V2 arrays")
        for mesh_name in V_MESH_NAMES:
            groups = collection[mesh_name]
            if not isinstance(groups, Mapping) or tuple(groups) != LEGACY_REQUIRED_GROUPS:
                raise ValueError(f"{label} {mesh_name} groups are incomplete")
            expected_leading = (
                len(meshes[mesh_name]["z"]), len(meshes[mesh_name]["r"]),
            )
            for group in LEGACY_REQUIRED_GROUPS:
                value = np.asarray(groups[group], dtype=float)
                if (
                    value.ndim < 3
                    or value.shape[:2] != expected_leading
                    or not np.all(np.isfinite(value))
                ):
                    raise ValueError(f"{label} {mesh_name} {group} array is invalid")
                trailing = value.shape[2:]
                if group in trailing_shapes and trailing_shapes[group] != trailing:
                    raise ValueError(f"sealed legacy {group} grouping changed")
                trailing_shapes[group] = trailing
    for group, trailing in trailing_shapes.items():
        if len(tuple(component_orders[group])) != int(np.prod(trailing)):
            raise ValueError(f"legacy {group} order does not name every lane")
    return meshes


def _score_legacy_group(inputs, context):
    meshes = _validate_legacy_arrays(inputs)
    score = score_precomputed_groups_on_v_meshes(
        inputs.legacy_Q33_by_mesh,
        inputs.legacy_Q55_by_mesh,
        meshes,
        comparison_kind="legacy_Q33_Q55",
        required_groups=LEGACY_REQUIRED_GROUPS,
        component_orders=inputs.legacy_component_orders,
    )
    return _gate_record(
        "legacy_holdout",
        context,
        complete=True,
        provenance_valid=True,
        passed=bool(score["passed"]),
        details={
            "sealed_comparator_names": tuple(SEALED_ADVERSE_COMPARATOR_NAMES),
            "score": score,
        },
    )


def _score_sampling_group(inputs, context):
    pair, _ = _validate_position_bindings(inputs, context)
    score = score_sampling_order(pair.primary, meshes=_v_meshes())
    return _gate_record(
        "sampling_order",
        context,
        complete=True,
        provenance_valid=True,
        passed=bool(score["passed"]),
        details={
            "unchanged_Q53_position_sampling": score,
            "acceleration_sampled": False,
        },
    )


_SCORERS = MappingProxyType({
    "pre_acceleration_construction": _score_construction_group,
    "native_position_tangent": _score_native_group,
    "position_representation": _score_position_group,
    "dense_boundary_audit": _score_dense_boundary_group,
    "signature_union": _score_signature_group,
    "legacy_holdout": _score_legacy_group,
    "sampling_order": _score_sampling_group,
    "bulk_prerequisite": _score_bulk_group,
})


def _staged_provenance(
    stage_name,
    context,
    group_order,
    input_hashes,
    *,
    prior_stage_fingerprint=None,
):
    prior = None if prior_stage_fingerprint is None else str(
        prior_stage_fingerprint
    )
    if prior is not None and not _valid_sha256(prior):
        raise ValueError("staged provenance prior fingerprint is invalid")
    payload = {
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "stage_name": str(stage_name),
        "parent_label": context["label"],
        "parent_identity": context["identity"],
        "group_order": tuple(group_order),
        "input_hashes": input_hashes,
        "prior_stage_fingerprint": prior,
    }
    return MappingProxyType({
        **payload,
        "fingerprint": _hash_tree(
            payload, root=f"staged-provenance/{stage_name}",
        ),
    })


def _validate_staged_provenance(
    provenance,
    *,
    stage_name,
    context,
    group_order,
    input_hashes,
    prior_stage_fingerprint=None,
):
    if not isinstance(provenance, Mapping) or tuple(provenance) != (
        STAGED_PROVENANCE_KEYS
    ):
        raise ValueError("staged provenance schema differs")
    expected = _staged_provenance(
        stage_name,
        context,
        group_order,
        input_hashes,
        prior_stage_fingerprint=prior_stage_fingerprint,
    )
    if _hash_tree(provenance, root="supplied-staged-provenance") != _hash_tree(
        expected, root="supplied-staged-provenance",
    ):
        raise ValueError("staged provenance does not bind the exact inputs")
    return expected


def _staged_result(
    stage_name,
    context,
    group_order,
    groups,
    hashes_before,
    hashes_after,
    *,
    prior_stage_fingerprint=None,
):
    group_order = tuple(group_order)
    groups = MappingProxyType(dict(groups))
    reached = tuple(groups)
    if reached != group_order[:len(reached)] or not reached:
        raise ValueError("staged result reached groups are not an ordered prefix")
    if dict(hashes_before) != dict(hashes_after):
        raise ValueError("staged inputs changed while scoring")
    for name, record in groups.items():
        if any(field not in record for field in _GATE_CORE):
            raise RuntimeError(f"staged gate {name} lacks the ledger core")
        if not record["complete"] or not record["provenance_valid"]:
            raise ValueError(f"staged gate {name} lacks valid reached evidence")
    failures = tuple(name for name, record in groups.items() if not record["passed"])
    if len(failures) > 1 or (failures and failures[0] != reached[-1]):
        raise ValueError("staged scoring continued after its first failure")
    passed = bool(not failures and reached == group_order)
    first_failure = failures[0] if failures else None
    prior = None if prior_stage_fingerprint is None else str(
        prior_stage_fingerprint
    )
    payload = {
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "stage_name": str(stage_name),
        "parent_label": context["label"],
        "parent_identity": context["identity"],
        "group_order": group_order,
        "reached_group_order": reached,
        "groups": groups,
        "input_hashes_before": hashes_before,
        "input_hashes_after": hashes_after,
        "prior_stage_fingerprint": prior,
        "complete": True,
        "provenance_valid": True,
        "passed": passed,
        "first_failure": first_failure,
    }
    return MappingProxyType({
        **payload,
        "fingerprint": _hash_tree(payload, root=f"staged-result/{stage_name}"),
    })


def _validate_staged_result(
    result,
    *,
    stage_name,
    group_order,
    require_passed,
):
    if not isinstance(result, Mapping) or tuple(result) != STAGED_RESULT_KEYS:
        raise ValueError("staged result schema differs")
    if (
        str(result["protocol_identifier"]) != PROTOCOL_IDENTIFIER
        or str(result["stage_name"]) != str(stage_name)
        or tuple(result["group_order"]) != tuple(group_order)
        or type(result["complete"]) is not bool
        or result["complete"] is not True
        or type(result["provenance_valid"]) is not bool
        or result["provenance_valid"] is not True
        or type(result["passed"]) is not bool
        or (require_passed and result["passed"] is not True)
    ):
        raise ValueError("staged result state differs")
    label = str(result["parent_label"])
    identity = str(result["parent_identity"])
    hashes_before = result["input_hashes_before"]
    hashes_after = result["input_hashes_after"]
    if (
        label not in ("N0", "N1")
        or not _valid_sha256(identity)
        or not isinstance(hashes_before, Mapping)
        or not isinstance(hashes_after, Mapping)
        or tuple(hashes_before) != tuple(hashes_after)
        or not all(
            _valid_sha256(hashes_before[name])
            and str(hashes_before[name]) == str(hashes_after[name])
            for name in hashes_before
        )
    ):
        raise ValueError("staged result identity or input hashes differ")
    payload = {name: result[name] for name in STAGED_RESULT_KEYS[:-1]}
    if (
        not _valid_sha256(result["fingerprint"])
        or str(result["fingerprint"])
        != _hash_tree(payload, root=f"staged-result/{stage_name}")
    ):
        raise ValueError("staged result fingerprint differs")
    rebuilt = _staged_result(
        stage_name,
        MappingProxyType({"label": label, "identity": identity}),
        group_order,
        result["groups"],
        hashes_before,
        hashes_after,
        prior_stage_fingerprint=result["prior_stage_fingerprint"],
    )
    if _hash_tree(rebuilt, root="validated-staged-result") != _hash_tree(
        result, root="validated-staged-result",
    ):
        raise ValueError("staged result semantics differ")
    return result


def _score_reached_prefix(inputs, context, groups, existing=()):
    records = dict(existing)
    for name in groups:
        record = _SCORERS[name](inputs, context)
        if any(field not in record for field in _GATE_CORE):
            raise RuntimeError(f"composed staged gate {name} lacks the ledger core")
        if not record["complete"] or not record["provenance_valid"]:
            raise ValueError(f"reached staged gate {name} is incomplete or invalid")
        records[name] = record
        if not record["passed"]:
            break
    return records


def capture_protocol125_position_prefix_provenance(inputs):
    """Seal only the inputs available before legacy/sampling or bulk work."""
    if not isinstance(inputs, Protocol125PreAccelerationCoreInputs):
        raise TypeError("position prefix requires core pre-acceleration inputs")
    context, hashes = _core_input_hashes(inputs)
    return _staged_provenance(
        "position-prefix", context, POSITION_PREFIX_GROUPS, hashes,
    )


def evaluate_protocol125_position_prefix(inputs, provenance):
    """Score construction through signature without calling later jobs."""
    context, hashes_before = _core_input_hashes(inputs)
    _validate_staged_provenance(
        provenance,
        stage_name="position-prefix",
        context=context,
        group_order=POSITION_PREFIX_GROUPS,
        input_hashes=hashes_before,
    )
    groups = _score_reached_prefix(
        inputs, context, POSITION_PREFIX_GROUPS,
    )
    _, hashes_after = _core_input_hashes(inputs)
    return _staged_result(
        "position-prefix",
        context,
        POSITION_PREFIX_GROUPS,
        groups,
        hashes_before,
        hashes_after,
    )


def capture_protocol125_legacy_sampling_provenance(inputs, position_prefix):
    """Seal legacy inputs only after the position prefix has passed."""
    prefix = _validate_staged_result(
        position_prefix,
        stage_name="position-prefix",
        group_order=POSITION_PREFIX_GROUPS,
        require_passed=True,
    )
    context, hashes = _representation_input_hashes(inputs)
    if (
        context["label"] != prefix["parent_label"]
        or context["identity"] != prefix["parent_identity"]
        or any(
            str(hashes[name]) != str(prefix["input_hashes_after"][name])
            for name in CORE_INPUT_HASH_KEYS
        )
    ):
        raise ValueError("legacy/sampling inputs differ from the passed prefix")
    return _staged_provenance(
        "legacy-sampling-prefix",
        context,
        REPRESENTATION_PREFIX_GROUPS,
        hashes,
        prior_stage_fingerprint=prefix["fingerprint"],
    )


def extend_protocol125_legacy_sampling(inputs, position_prefix, provenance):
    """Score legacy then sampling, stopping before bulk on either failure."""
    prefix = _validate_staged_result(
        position_prefix,
        stage_name="position-prefix",
        group_order=POSITION_PREFIX_GROUPS,
        require_passed=True,
    )
    context, hashes_before = _representation_input_hashes(inputs)
    capture_protocol125_legacy_sampling_provenance(inputs, prefix)
    _validate_staged_provenance(
        provenance,
        stage_name="legacy-sampling-prefix",
        context=context,
        group_order=REPRESENTATION_PREFIX_GROUPS,
        input_hashes=hashes_before,
        prior_stage_fingerprint=prefix["fingerprint"],
    )
    groups = _score_reached_prefix(
        inputs,
        context,
        PRE_ACCELERATION_GROUPS[5:7],
        existing=prefix["groups"].items(),
    )
    _, hashes_after = _representation_input_hashes(inputs)
    return _staged_result(
        "legacy-sampling-prefix",
        context,
        REPRESENTATION_PREFIX_GROUPS,
        groups,
        hashes_before,
        hashes_after,
        prior_stage_fingerprint=prefix["fingerprint"],
    )


def capture_protocol125_bulk_prerequisite_provenance(
    inputs, representation_prefix,
):
    """Seal the bulk audit only after every representation gate passed."""
    prefix = _validate_staged_result(
        representation_prefix,
        stage_name="legacy-sampling-prefix",
        group_order=REPRESENTATION_PREFIX_GROUPS,
        require_passed=True,
    )
    context, hashes = _input_hashes(inputs)
    if (
        context["label"] != prefix["parent_label"]
        or context["identity"] != prefix["parent_identity"]
        or any(
            str(hashes[name]) != str(prefix["input_hashes_after"][name])
            for name in REPRESENTATION_INPUT_HASH_KEYS
        )
    ):
        raise ValueError("bulk inputs differ from the passed representation prefix")
    return _staged_provenance(
        "bulk-prerequisite",
        context,
        PRE_ACCELERATION_GROUPS,
        hashes,
        prior_stage_fingerprint=prefix["fingerprint"],
    )


def finish_protocol125_bulk_prerequisite(
    inputs, representation_prefix, provenance,
):
    """Score the last pre-acceleration gate after all position gates pass."""
    prefix = _validate_staged_result(
        representation_prefix,
        stage_name="legacy-sampling-prefix",
        group_order=REPRESENTATION_PREFIX_GROUPS,
        require_passed=True,
    )
    context, hashes_before = _input_hashes(inputs)
    capture_protocol125_bulk_prerequisite_provenance(inputs, prefix)
    _validate_staged_provenance(
        provenance,
        stage_name="bulk-prerequisite",
        context=context,
        group_order=PRE_ACCELERATION_GROUPS,
        input_hashes=hashes_before,
        prior_stage_fingerprint=prefix["fingerprint"],
    )
    groups = _score_reached_prefix(
        inputs,
        context,
        ("bulk_prerequisite",),
        existing=prefix["groups"].items(),
    )
    _, hashes_after = _input_hashes(inputs)
    return _top_level_result(
        context,
        groups,
        hashes_before=hashes_before,
        hashes_after=hashes_after,
    )


def finalize_protocol125_preacceleration_stop(staged_result):
    """Expand one valid staged numerical failure into the full ordered stop."""
    stage_name = str(staged_result.get("stage_name", "")) if isinstance(
        staged_result, Mapping,
    ) else ""
    expected = {
        "position-prefix": POSITION_PREFIX_GROUPS,
        "legacy-sampling-prefix": REPRESENTATION_PREFIX_GROUPS,
    }
    if stage_name not in expected:
        raise ValueError("only a position or representation prefix may stop early")
    result = _validate_staged_result(
        staged_result,
        stage_name=stage_name,
        group_order=expected[stage_name],
        require_passed=False,
    )
    if result["passed"] or result["first_failure"] is None:
        raise ValueError("an early-stop result must contain a numerical failure")
    context = MappingProxyType({
        "label": str(result["parent_label"]),
        "identity": str(result["parent_identity"]),
    })
    groups = dict(result["groups"])
    failed = str(result["first_failure"])
    failed_index = PRE_ACCELERATION_GROUPS.index(failed)
    for name in PRE_ACCELERATION_GROUPS[failed_index+1:]:
        groups[name] = _ordered_not_reached_gate(name, context, failed)
    return _top_level_result(
        context,
        groups,
        hashes_before=result["input_hashes_before"],
        hashes_after=result["input_hashes_after"],
    )


def _top_level_result(
    context,
    groups,
    *,
    hashes_before,
    hashes_after,
    invalid_reasons=(),
):
    groups = MappingProxyType(dict(groups))
    invalid = tuple(str(reason) for reason in invalid_reasons)
    complete = bool(
        tuple(groups) == PRE_ACCELERATION_GROUPS
        and all(record["complete"] for record in groups.values())
    )
    provenance_valid = bool(
        not invalid
        and complete
        and all(record["provenance_valid"] for record in groups.values())
        and hashes_after is not None
        and dict(hashes_before) == dict(hashes_after)
    )
    passed = bool(
        complete
        and provenance_valid
        and all(record["passed"] for record in groups.values())
    )
    if not complete or not provenance_valid:
        classification = "INVALID-audit"
    elif passed:
        classification = "PASS-single-parent-pre-acceleration"
    else:
        classification = "FAIL-single-parent-pre-acceleration"
    return MappingProxyType({
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "classification": classification,
        "complete": complete,
        "provenance_valid": provenance_valid,
        "passed": passed,
        "parent_label": context["label"],
        "parent_identity": context["identity"],
        "required_group_order": PRE_ACCELERATION_GROUPS,
        "groups": groups,
        "invalid_reasons": invalid,
        "input_hashes_before": hashes_before,
        "input_hashes_after": hashes_after,
        "inputs_stable_while_scoring": bool(
            hashes_after is not None and dict(hashes_before) == dict(hashes_after)
        ),
        "single_parent_only": True,
        "second_parent_and_common_V2_still_required": True,
        "acceleration_evaluated": False,
        "acceleration_authorized": False,
        "scientific_execution_authorized": False,
        "artifact_written": False,
    })


def compose_protocol125_construction_failure_records(failure_record):
    """Bind a validated construction failure to an ordered pre-stage stop.

    The construction gate is a reached, complete scientific failure.  Every
    later prerequisite receives a complete record of non-execution, bound to
    that first failed gate.  Those records are neither placeholders nor
    numerical passes; an ordered adjudicator must treat them only as an
    authorized intra-stage stop.
    """
    failure = validate_protocol125_construction_failure_record(failure_record)
    context = MappingProxyType({
        "label": str(failure["parent_label"]),
        "identity": str(failure["parent_identity"]),
    })
    failure_classification = str(failure["classification"])
    failed_group = (
        "native_position_tangent"
        if failure_classification == "FAIL-parent-position"
        else "pre_acceleration_construction"
    )
    groups = {}
    if failed_group == "native_position_tangent":
        groups["pre_acceleration_construction"] = _gate_record(
            "pre_acceleration_construction",
            context,
            complete=True,
            provenance_valid=True,
            passed=True,
            details={
                "construction_solver_executed": True,
                "failure_record_validator_executed": True,
                "finite_wall_reference_passed": True,
                "joint_hybrid_residual_passed": True,
                "immutable_construction_inputs_passed": True,
                "native_position_completion_reached": True,
                "construction_failure_record_sha256": str(
                    failure["fingerprint"],
                ),
            },
        )
    measurement_finite = bool(failure["measurement_finite"])
    groups[failed_group] = _gate_record(
            failed_group,
            context,
            complete=True,
            provenance_valid=True,
            passed=False,
            details={
                "construction_solver_executed": True,
                "failure_record_validator_executed": True,
                "normal_construction_reload_scorer_executed": False,
                "failure_kind": "scientific-construction-threshold",
                "failure_classification": failure_classification,
                "failure_gate": str(failure["failure_gate"]),
                "measurement_finite": measurement_finite,
                "measured_value": (
                    float(failure["measured_value"])
                    if measurement_finite else None
                ),
                "measured_value_ieee754_hex": str(
                    failure["measured_value_ieee754_hex"],
                ),
                "strict_ceiling": float(failure["strict_ceiling"]),
                "strict_gate_failed": True,
                "construction_failure_record_sha256": str(
                    failure["fingerprint"],
                ),
                "construction_input_fingerprint": str(
                    failure["construction_input_fingerprint"],
                ),
                "source_coordinate_sha256": str(
                    failure["source_coordinate_sha256"],
                ),
                "reference_state_sha256": str(
                    failure["reference_state_sha256"],
                ),
                "acceleration_authorized": False,
                "retry_authorized": False,
                "candidate_or_phase_a_executed": False,
            },
        )
    failed_index = PRE_ACCELERATION_GROUPS.index(failed_group)
    for group_name in PRE_ACCELERATION_GROUPS[failed_index+1:]:
        groups[group_name] = _gate_record(
            group_name,
            context,
            complete=True,
            provenance_valid=True,
            passed=False,
            details={
                "scorer_executed": False,
                "failure_kind": "ordered-scientific-stop",
                "blocked_by": failed_group,
                "construction_failure_record_sha256": str(
                    failure["fingerprint"],
                ),
            },
            not_reached=True,
            blocked_by=failed_group,
        )
    groups = MappingProxyType(groups)
    failure_binding = MappingProxyType({
        "construction_failure_record_sha256": str(failure["fingerprint"]),
    })
    return MappingProxyType({
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "classification": "FAIL-single-parent-pre-acceleration",
        "complete": True,
        "provenance_valid": True,
        "passed": False,
        "parent_label": context["label"],
        "parent_identity": context["identity"],
        "required_group_order": PRE_ACCELERATION_GROUPS,
        "groups": groups,
        "invalid_reasons": (),
        "input_hashes_before": failure_binding,
        "input_hashes_after": failure_binding,
        "inputs_stable_while_scoring": True,
        "normal_pre_acceleration_scorers_executed": False,
        "construction_failure_record_sha256": str(failure["fingerprint"]),
        "single_parent_only": True,
        "second_parent_and_common_V2_still_required": True,
        "acceleration_evaluated": False,
        "acceleration_authorized": False,
        "scientific_execution_authorized": False,
        "artifact_written": False,
    })


def compose_protocol125_representation_coefficient_failure_records(
    parent_mapping,
    construction_provenance,
    failure_evidence,
):
    """Bind a fresh finite-input representation failure to construction.

    The parent itself has already passed its native construction solve.  The
    persisted representation is nevertheless a construction prerequisite, so
    a nonfinite coefficient is a complete ``FAIL-parent-bulk`` at the first
    pre-acceleration gate.  No normal representation, legacy, sampling, bulk,
    or acceleration scorer is permitted after this record is formed.
    """
    context = _parent_context(Protocol125PreAccelerationCoreInputs(
        parent_mapping=parent_mapping,
        position_pair=None,
        reference_pair=None,
        construction_provenance=construction_provenance,
        native_position_tangent_evidence=None,
    ))
    construction = validate_protocol125_successful_parent_provenance_record(
        construction_provenance,
        expected_parent_label=context["label"],
        expected_parent_identity=context["identity"],
    )
    stored_construction = parent_mapping.get("construction_provenance_record")
    if not isinstance(stored_construction, Mapping) or _hash_tree(
        stored_construction, root="stored-construction-provenance",
    ) != _hash_tree(
        construction, root="stored-construction-provenance",
    ):
        raise ValueError(
            "representation failure construction provenance differs from parent"
        )
    evidence = validate_protocol125_representation_coefficient_failure(
        failure_evidence,
    )
    if str(evidence["parent_identity"]) != context["identity"]:
        raise ValueError(
            "representation coefficient failure parent binding differs"
        )
    failed_group = "pre_acceleration_construction"
    groups = {
        failed_group: _gate_record(
            failed_group,
            context,
            complete=True,
            provenance_valid=True,
            passed=False,
            details={
                "construction_solver_executed": True,
                "successful_construction_record_validated": True,
                "normal_construction_reload_scorer_executed": False,
                "representation_builder_executed": True,
                "failure_kind": (
                    "scientific-representation-coefficient-nonfinite"
                ),
                "failure_classification": "FAIL-parent-bulk",
                "failure_gate": "persisted_representation_coefficients_finite",
                "recipe": str(evidence["recipe"]),
                "coefficient_shape": tuple(evidence["coefficient_shape"]),
                "coefficient_dtype": str(evidence["coefficient_dtype"]),
                "nonfinite_count": int(evidence["nonfinite_count"]),
                "representation_input_sha256": str(evidence["input_sha256"]),
                "representation_failure_fingerprint": str(
                    evidence["fingerprint"]
                ),
                "construction_provenance_fingerprint": str(
                    construction["fingerprint"]
                ),
                "acceleration_authorized": False,
                "retry_authorized": False,
                "candidate_or_phase_a_executed": False,
            },
        ),
    }
    for group_name in PRE_ACCELERATION_GROUPS[1:]:
        groups[group_name] = _ordered_not_reached_gate(
            group_name, context, failed_group,
        )
    binding = MappingProxyType({
        "parent_identity": context["identity"],
        "construction_provenance_sha256": str(construction["fingerprint"]),
        "representation_coefficient_failure_sha256": str(
            evidence["fingerprint"]
        ),
    })
    return _top_level_result(
        context,
        groups,
        hashes_before=binding,
        hashes_after=binding,
    )


def evaluate_protocol125_preacceleration(inputs, provenance):
    """Compose one parent's complete prerequisite gate set, fail closed."""
    try:
        context, hashes_before = _input_hashes(inputs)
    except (TypeError, ValueError, RuntimeError, KeyError, AttributeError) as error:
        return MappingProxyType({
            "protocol_identifier": PROTOCOL_IDENTIFIER,
            "classification": "INVALID-audit",
            "complete": False,
            "provenance_valid": False,
            "passed": False,
            "required_group_order": PRE_ACCELERATION_GROUPS,
            "groups": MappingProxyType({}),
            "invalid_reasons": (f"input structure: {error}",),
            "input_hashes_before": None,
            "input_hashes_after": None,
            "acceleration_evaluated": False,
            "acceleration_authorized": False,
            "scientific_execution_authorized": False,
            "artifact_written": False,
        })
    try:
        _validate_provenance(provenance, context, hashes_before)
    except (TypeError, ValueError, RuntimeError, KeyError) as error:
        reason = f"pre-score provenance: {error}"
        groups = {
            name: _invalid_gate(name, context, reason)
            for name in PRE_ACCELERATION_GROUPS
        }
        return _top_level_result(
            context,
            groups,
            hashes_before=hashes_before,
            hashes_after=None,
            invalid_reasons=(reason,),
        )

    groups = {}
    blocked_by = None
    scientific_stop = False
    invalid_reasons = []
    for name in PRE_ACCELERATION_GROUPS:
        if blocked_by is not None:
            if scientific_stop:
                groups[name] = _ordered_not_reached_gate(
                    name, context, blocked_by,
                )
            else:
                groups[name] = _invalid_gate(
                    name,
                    context,
                    f"unreached after fail-closed prerequisite {blocked_by}",
                    prerequisite=blocked_by,
                )
            continue
        try:
            record = _SCORERS[name](inputs, context)
            if any(field not in record for field in _GATE_CORE):
                raise RuntimeError("composed gate lacks the ledger core")
            groups[name] = record
            if record["complete"] and record["provenance_valid"]:
                if not record["passed"]:
                    blocked_by = name
                    scientific_stop = True
            else:
                blocked_by = name
                scientific_stop = False
        except (TypeError, ValueError, RuntimeError, KeyError, AttributeError) as error:
            reason = f"{name}: {error}"
            groups[name] = _invalid_gate(name, context, reason)
            invalid_reasons.append(reason)
            blocked_by = name
            scientific_stop = False

    try:
        _, hashes_after = _input_hashes(inputs)
    except (TypeError, ValueError, RuntimeError, KeyError, AttributeError) as error:
        hashes_after = None
        invalid_reasons.append(f"post-score input structure: {error}")
    if hashes_after is None or dict(hashes_before) != dict(hashes_after):
        reason = "pre-acceleration inputs changed while scoring"
        invalid_reasons.append(reason)
        groups = {
            name: _invalid_gate(name, context, reason)
            for name in PRE_ACCELERATION_GROUPS
        }
    return _top_level_result(
        context,
        groups,
        hashes_before=hashes_before,
        hashes_after=hashes_after,
        invalid_reasons=invalid_reasons,
    )
