"""Pure Protocol-125 final-representation matrix orchestration.

This module only composes already-defined, result-independent scorers.  It
does not build or repair a parent, execute an evolution equation, write an
artifact, or authorize a scientific calculation.  Every precomputed lane is
bound to an explicit pre-adjudication hash record, and the complete input
bundle is re-hashed after scoring so an in-place change fails closed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from bhps.joint_parent_adjudication import (
    V_MESH_NAMES,
    score_acceleration_pair_on_v_meshes,
    score_precomputed_groups_on_v_meshes,
    score_q4_q5_derivative_images_on_v_meshes,
    score_source_triplet_arrays,
    score_state_pair_on_v_meshes,
)
from bhps.joint_parent_endpoint_audits import (
    ACCELERATION_ENDPOINT_CONVERSION_LANES,
    score_acceleration_endpoint_conversion_pair,
    score_state_endpoint_z_reproduction,
    score_state_outer_derivative_reproduction,
    score_time_symmetric_velocity_endpoint_z,
)
from bhps.joint_parent_position_audits import (
    Protocol125PositionAuditMeshes,
    evaluate_protocol125_dense_outer_delta_robin_audit,
    evaluate_protocol125_dense_wall_audit,
)
from bhps.joint_parent_representation import (
    SEALED_ADVERSE_COMPARATOR_NAMES,
)
from bhps.joint_parent_refinement_diagnostics import (
    VALIDATION_MESH_SPECS,
)
from bhps.matched_staged_continuum import hash_arrays


PROTOCOL_IDENTIFIER = "Protocol-125-final-representation-matrix-v1"
SOURCE_TRIPLET_KEYS = ("source", "source_time", "source_second_time")
POSITION_SPATIAL_GROUPS = ("position", "first_spatial", "second_spatial")
LEGACY_REQUIRED_GROUPS = POSITION_SPATIAL_GROUPS+("acceleration",)
REQUIRED_FINAL_MATRIX_LANES = (
    "Q53_Q33_position_spatial",
    "Q53_Q33_position_q4_q5_images",
    "Q53_Q33_acceleration",
    "Q53_Q33_acceleration_q4_q5_images",
    "Q53_Q33_source_triplet",
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
    "independent_dense_wall_position",
    "independent_dense_outer_position",
    "sealed_legacy_Q33_Q55_position_spatial_acceleration",
    "sealed_legacy_Q33_Q55_source_triplet",
)
INPUT_HASH_KEYS = (
    "final_pair_sha256",
    "reference_pair_sha256",
    "v_meshes_sha256",
    "position_audit_meshes_sha256",
    "Q53_source_triplets_sha256",
    "Q33_source_triplets_sha256",
    "velocity_endpoint_z_sha256",
    "row_implied_acceleration_z_source_sha256",
    "row_implied_acceleration_z_dense_sha256",
    "direct_Dz7_acceleration_z_source_sha256",
    "shared_representation_sha256",
    "legacy_Q33_sha256",
    "legacy_Q55_sha256",
    "legacy_Q33_source_triplets_sha256",
    "legacy_Q55_source_triplets_sha256",
    "legacy_component_orders_sha256",
)
PROVENANCE_KEYS = (
    "protocol_identifier",
    "parent_label",
    "legacy_comparator_names",
    "input_hashes",
)


@dataclass(frozen=True)
class Protocol125FinalMatrixInputs:
    """All data consumed by the final representation matrix.

    The container is deliberately data-only.  In particular, it has no
    writer, runner, repair, resume, or candidate-construction callback.
    """

    final_pair: object
    reference_pair: object
    v_meshes: Mapping
    position_audit_meshes: Protocol125PositionAuditMeshes
    Q53_source_triplets_by_mesh: Mapping
    Q33_source_triplets_by_mesh: Mapping
    velocity_endpoint_z: np.ndarray
    row_implied_acceleration_z_source: Mapping
    row_implied_acceleration_z_dense: Mapping
    direct_Dz7_acceleration_z_source: Mapping
    shared_representation_fingerprint: str
    legacy_Q33_by_mesh: Mapping
    legacy_Q55_by_mesh: Mapping
    legacy_Q33_source_triplets_by_mesh: Mapping
    legacy_Q55_source_triplets_by_mesh: Mapping
    legacy_component_orders: Mapping


def _valid_sha256(value):
    value = str(value)
    return bool(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _digest_label(digest, label):
    encoded = str(label).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _digest_array(digest, label, value):
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype == object:
        raise ValueError(f"hashed input {label} has object dtype")
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise ValueError(f"hashed input {label} is nonfinite")
    _digest_label(digest, label)
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())


def _fingerprint_tree(value, *, root):
    """Hash a nested string-keyed mapping or one array without serialization."""
    digest = hashlib.sha256()

    def visit(item, path):
        if isinstance(item, Mapping):
            if not item or any(not isinstance(name, str) or not name for name in item):
                raise ValueError(f"hashed mapping {path} is empty or has invalid keys")
            _digest_label(digest, f"mapping:{path}")
            for name in sorted(item):
                visit(item[name], f"{path}/{name}")
            return
        _digest_array(digest, path, item)

    visit(value, str(root))
    return digest.hexdigest()


def _object_sha256(value, label):
    coefficient_arrays = getattr(value, "coefficient_arrays", None)
    if coefficient_arrays is not None and callable(coefficient_arrays):
        record = coefficient_arrays()
        if not isinstance(record, Mapping) or not record:
            raise ValueError(f"{label} coefficient record is missing")
        return _fingerprint_tree(record, root=label)
    fingerprint = getattr(value, "fingerprint", None)
    if fingerprint is None or not callable(fingerprint):
        raise ValueError(f"{label} exposes no coefficient or fingerprint record")
    found = str(fingerprint())
    if not _valid_sha256(found):
        raise ValueError(f"{label} fingerprint is missing or invalid")
    return found


def _extract_final_states(final_pair):
    try:
        primary = final_pair.primary
        comparator = final_pair.comparator
        states = {
            "Q53_position": primary.position,
            "Q33_position": comparator.position,
            "Q53_acceleration": primary.acceleration,
            "Q33_acceleration": comparator.acceleration,
        }
    except AttributeError as error:
        raise ValueError("final pair is missing a Q53/Q33 position/acceleration lane") from error
    expected = {
        "Q53_position": ("position", 5),
        "Q33_position": ("position", 3),
        "Q53_acceleration": ("acceleration", 5),
        "Q33_acceleration": ("acceleration", 3),
    }
    for name, state in states.items():
        state_name, degree = expected[name]
        if (
            str(getattr(state, "state_name", "")) != state_name
            or int(getattr(state, "z_degree", -1)) != degree
            or not callable(getattr(state, "evaluate_coordinate_components", None))
            or not callable(getattr(state, "evaluate_physical_channels", None))
        ):
            raise ValueError(f"final pair {name} state is incomplete or mislabeled")
    source_hashes = (
        str(getattr(primary, "source_fingerprint", "")),
        str(getattr(comparator, "source_fingerprint", "")),
    )
    endpoint_hashes = (
        str(getattr(primary, "endpoint_fingerprint", "")),
        str(getattr(comparator, "endpoint_fingerprint", "")),
    )
    if (
        source_hashes[0] != source_hashes[1]
        or endpoint_hashes[0] != endpoint_hashes[1]
        or not all(_valid_sha256(value) for value in source_hashes+endpoint_hashes)
    ):
        raise ValueError("final Q53/Q33 source or endpoint provenance differs")
    source_z = np.asarray(states["Q53_position"].source_z, dtype=float)
    source_r = np.asarray(states["Q53_position"].source_r, dtype=float)
    for state in states.values():
        if not (
            np.array_equal(np.asarray(state.source_z), source_z)
            and np.array_equal(np.asarray(state.source_r), source_r)
        ):
            raise ValueError("final representation states use different source grids")
    if source_r.ndim != 1 or len(source_r) < 2:
        raise ValueError("final representation source radius is invalid")
    return states, source_z, source_r, source_hashes[0], endpoint_hashes[0]


def _validate_v_meshes(v_meshes):
    if not isinstance(v_meshes, Mapping) or tuple(v_meshes) != V_MESH_NAMES:
        raise ValueError("final matrix requires ordered V0/V1/V2 meshes")
    shapes = {}
    for name in V_MESH_NAMES:
        mesh = v_meshes[name]
        if not isinstance(mesh, Mapping) or not {"z", "r", "sha256"} <= set(mesh):
            raise ValueError(f"final matrix {name} mesh record is incomplete")
        z = np.asarray(mesh["z"], dtype=float)
        r = np.asarray(mesh["r"], dtype=float)
        expected_nz, expected_nr, expected_hash = VALIDATION_MESH_SPECS[name]
        found = hash_arrays(z, r)
        if (
            z.shape != (expected_nz,)
            or r.shape != (expected_nr,)
            or found != expected_hash
            or str(mesh["sha256"]) != expected_hash
        ):
            raise ValueError(f"final matrix {name} mesh or hash differs from Protocol 125")
        shapes[name] = (expected_nz, expected_nr)
    return shapes


def _validate_source_triplets(by_mesh, shapes, label):
    if not isinstance(by_mesh, Mapping) or tuple(by_mesh) != V_MESH_NAMES:
        raise ValueError(f"{label} source triplets require ordered V0/V1/V2")
    for mesh_name in V_MESH_NAMES:
        record = by_mesh[mesh_name]
        if not isinstance(record, Mapping) or set(record) != set(SOURCE_TRIPLET_KEYS):
            raise ValueError(f"{label} {mesh_name} source triplet lanes are incomplete")
        expected = (*shapes[mesh_name], 3)
        for lane in SOURCE_TRIPLET_KEYS:
            value = np.asarray(record[lane], dtype=float)
            if value.shape != expected or not np.all(np.isfinite(value)):
                raise ValueError(f"{label} {mesh_name} {lane} lane is invalid")


def _validate_legacy(legacy, shapes, component_orders, label):
    if not isinstance(legacy, Mapping) or tuple(legacy) != V_MESH_NAMES:
        raise ValueError(f"{label} legacy input requires ordered V0/V1/V2")
    trailing_shapes = {}
    for mesh_name in V_MESH_NAMES:
        record = legacy[mesh_name]
        if not isinstance(record, Mapping) or set(record) != set(LEGACY_REQUIRED_GROUPS):
            raise ValueError(f"{label} {mesh_name} legacy groups are incomplete")
        for group in LEGACY_REQUIRED_GROUPS:
            value = np.asarray(record[group], dtype=float)
            if (
                value.ndim < 3
                or value.shape[:2] != shapes[mesh_name]
                or value.size == 0
                or not np.all(np.isfinite(value))
            ):
                raise ValueError(f"{label} {mesh_name} {group} lane is invalid")
            trailing = value.shape[2:]
            if group in trailing_shapes and trailing_shapes[group] != trailing:
                raise ValueError(f"{label} {group} grouping changed between meshes")
            trailing_shapes[group] = trailing
    for group, trailing in trailing_shapes.items():
        order = tuple(component_orders[group])
        if len(order) != int(np.prod(trailing)):
            raise ValueError(f"legacy {group} component order does not name every lane")


def _validate_inputs(inputs):
    if not isinstance(inputs, Protocol125FinalMatrixInputs):
        raise TypeError("final matrix requires Protocol125FinalMatrixInputs")
    states, source_z, source_r, source_hash, endpoint_hash = _extract_final_states(
        inputs.final_pair,
    )
    shapes = _validate_v_meshes(inputs.v_meshes)
    if not isinstance(inputs.position_audit_meshes, Protocol125PositionAuditMeshes):
        raise ValueError("final matrix requires a bound position-audit mesh union")
    _validate_source_triplets(
        inputs.Q53_source_triplets_by_mesh, shapes, "Q53",
    )
    _validate_source_triplets(
        inputs.Q33_source_triplets_by_mesh, shapes, "Q33",
    )
    _validate_source_triplets(
        inputs.legacy_Q33_source_triplets_by_mesh, shapes, "legacy Q33",
    )
    _validate_source_triplets(
        inputs.legacy_Q55_source_triplets_by_mesh, shapes, "legacy Q55",
    )
    if (
        not isinstance(inputs.legacy_component_orders, Mapping)
        or set(inputs.legacy_component_orders) != set(LEGACY_REQUIRED_GROUPS)
    ):
        raise ValueError("legacy component orders must name exactly every required group")
    for group in LEGACY_REQUIRED_GROUPS:
        order = tuple(inputs.legacy_component_orders[group])
        if not order or any(not isinstance(name, str) or not name for name in order):
            raise ValueError(f"legacy {group} component order is invalid")
    _validate_legacy(
        inputs.legacy_Q33_by_mesh,
        shapes,
        inputs.legacy_component_orders,
        SEALED_ADVERSE_COMPARATOR_NAMES[0],
    )
    _validate_legacy(
        inputs.legacy_Q55_by_mesh,
        shapes,
        inputs.legacy_component_orders,
        SEALED_ADVERSE_COMPARATOR_NAMES[1],
    )
    velocity = np.asarray(inputs.velocity_endpoint_z, dtype=float)
    if velocity.shape != (2, len(source_r), 8) or not np.all(np.isfinite(velocity)):
        raise ValueError("velocity endpoint-z lane is missing or has the wrong source shape")
    if any(not isinstance(record, Mapping) for record in (
        inputs.row_implied_acceleration_z_source,
        inputs.row_implied_acceleration_z_dense,
        inputs.direct_Dz7_acceleration_z_source,
    )) or not _valid_sha256(inputs.shared_representation_fingerprint):
        raise ValueError("acceleration endpoint conversion evidence is incomplete")
    return {
        "states": states,
        "source_z": source_z,
        "source_r": source_r,
        "source_fingerprint": source_hash,
        "endpoint_fingerprint": endpoint_hash,
        "v_shapes": shapes,
    }


def protocol125_final_matrix_input_hashes(inputs):
    """Return hashes for the complete, structurally valid matrix input bundle."""
    _validate_inputs(inputs)
    hashes = {
        "final_pair_sha256": _object_sha256(inputs.final_pair, "final_pair"),
        "reference_pair_sha256": _object_sha256(
            inputs.reference_pair, "reference_pair",
        ),
        "v_meshes_sha256": _fingerprint_tree(inputs.v_meshes, root="v_meshes"),
        "position_audit_meshes_sha256": _object_sha256(
            inputs.position_audit_meshes, "position_audit_meshes",
        ),
        "Q53_source_triplets_sha256": _fingerprint_tree(
            inputs.Q53_source_triplets_by_mesh, root="Q53_source_triplets",
        ),
        "Q33_source_triplets_sha256": _fingerprint_tree(
            inputs.Q33_source_triplets_by_mesh, root="Q33_source_triplets",
        ),
        "velocity_endpoint_z_sha256": _fingerprint_tree(
            inputs.velocity_endpoint_z, root="velocity_endpoint_z",
        ),
        "row_implied_acceleration_z_source_sha256": _fingerprint_tree(
            inputs.row_implied_acceleration_z_source,
            root="row_implied_acceleration_z_source",
        ),
        "row_implied_acceleration_z_dense_sha256": _fingerprint_tree(
            inputs.row_implied_acceleration_z_dense,
            root="row_implied_acceleration_z_dense",
        ),
        "direct_Dz7_acceleration_z_source_sha256": _fingerprint_tree(
            inputs.direct_Dz7_acceleration_z_source,
            root="direct_Dz7_acceleration_z_source",
        ),
        "shared_representation_sha256": _fingerprint_tree(
            inputs.shared_representation_fingerprint,
            root="shared_representation_fingerprint",
        ),
        "legacy_Q33_sha256": _fingerprint_tree(
            inputs.legacy_Q33_by_mesh, root=SEALED_ADVERSE_COMPARATOR_NAMES[0],
        ),
        "legacy_Q55_sha256": _fingerprint_tree(
            inputs.legacy_Q55_by_mesh, root=SEALED_ADVERSE_COMPARATOR_NAMES[1],
        ),
        "legacy_Q33_source_triplets_sha256": _fingerprint_tree(
            inputs.legacy_Q33_source_triplets_by_mesh,
            root="legacy_Q33_source_triplets",
        ),
        "legacy_Q55_source_triplets_sha256": _fingerprint_tree(
            inputs.legacy_Q55_source_triplets_by_mesh,
            root="legacy_Q55_source_triplets",
        ),
        "legacy_component_orders_sha256": _fingerprint_tree(
            inputs.legacy_component_orders, root="legacy_component_orders",
        ),
    }
    if tuple(hashes) != INPUT_HASH_KEYS or not all(
        _valid_sha256(value) for value in hashes.values()
    ):
        raise RuntimeError("final matrix input hash inventory is incomplete")
    return MappingProxyType(hashes)


def capture_protocol125_final_matrix_provenance(inputs, *, parent_label):
    """Capture the record that must be frozen before matrix scoring begins."""
    label = str(parent_label)
    if not label:
        raise ValueError("final matrix parent label is required")
    return MappingProxyType({
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "parent_label": label,
        "legacy_comparator_names": tuple(SEALED_ADVERSE_COMPARATOR_NAMES),
        "input_hashes": protocol125_final_matrix_input_hashes(inputs),
    })


def _validate_provenance(provenance, found_hashes):
    if not isinstance(provenance, Mapping) or set(provenance) != set(PROVENANCE_KEYS):
        raise ValueError("final matrix provenance record is missing or incomplete")
    if str(provenance["protocol_identifier"]) != PROTOCOL_IDENTIFIER:
        raise ValueError("final matrix protocol identifier differs")
    parent_label = str(provenance["parent_label"])
    if not parent_label:
        raise ValueError("final matrix provenance omits the parent label")
    if tuple(provenance["legacy_comparator_names"]) != tuple(
        SEALED_ADVERSE_COMPARATOR_NAMES
    ):
        raise ValueError("sealed legacy comparator identities differ")
    expected = provenance["input_hashes"]
    if not isinstance(expected, Mapping) or tuple(expected) != INPUT_HASH_KEYS:
        raise ValueError("final matrix provenance hash inventory is incomplete or reordered")
    for name in INPUT_HASH_KEYS:
        if not _valid_sha256(expected[name]):
            raise ValueError(f"final matrix provenance hash {name} is invalid")
        if str(expected[name]) != str(found_hashes[name]):
            raise ValueError(f"final matrix input hash mismatch: {name}")
    return parent_label


def _score_source_triplet_matrix(inputs):
    records = {}
    gates = {}
    for mesh_name in V_MESH_NAMES:
        score = score_source_triplet_arrays(
            inputs.Q53_source_triplets_by_mesh[mesh_name],
            inputs.Q33_source_triplets_by_mesh[mesh_name],
            comparison_kind="Q53_Q33",
        )
        coordinate_sha256 = hash_arrays(
            inputs.v_meshes[mesh_name]["z"],
            inputs.v_meshes[mesh_name]["r"],
        )
        records[mesh_name] = {
            "coordinate_sha256": coordinate_sha256,
            **dict(score),
        }
        gates[mesh_name] = bool(score["passed"])
    return MappingProxyType({
        "comparison_kind": "Q53_Q33",
        "mesh_order": V_MESH_NAMES,
        "records": records,
        "gates": gates,
        "passed": bool(all(gates.values())),
    })


def _score_legacy_source_triplet_matrix(inputs):
    records = {}
    gates = {}
    for mesh_name in V_MESH_NAMES:
        score = score_source_triplet_arrays(
            inputs.legacy_Q33_source_triplets_by_mesh[mesh_name],
            inputs.legacy_Q55_source_triplets_by_mesh[mesh_name],
            comparison_kind="legacy_Q33_Q55",
        )
        records[mesh_name] = {
            "coordinate_sha256": hash_arrays(
                inputs.v_meshes[mesh_name]["z"],
                inputs.v_meshes[mesh_name]["r"],
            ),
            **dict(score),
        }
        gates[mesh_name] = bool(score["passed"])
    return MappingProxyType({
        "comparison_kind": "legacy_Q33_Q55",
        "mesh_order": V_MESH_NAMES,
        "records": records,
        "gates": gates,
        "passed": bool(all(gates.values())),
    })


def _invalid_result(reason, *, found_hashes=None, lanes=None):
    return MappingProxyType({
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "classification": "INVALID-audit",
        "complete": False,
        "provenance_valid": False,
        "passed": False,
        "required_lane_order": REQUIRED_FINAL_MATRIX_LANES,
        "lanes": MappingProxyType({} if lanes is None else dict(lanes)),
        "failed_lanes": (),
        "invalid_reasons": (str(reason),),
        "input_hashes_before": found_hashes,
        "input_hashes_after": None,
        "phase_a_authorized": False,
        "scientific_execution_authorized": False,
        "artifact_written": False,
    })


def evaluate_protocol125_final_representation_matrix(inputs, provenance):
    """Score every implemented final-representation lane and fail closed.

    A passing return value means only that this matrix of implemented
    representation scorers passed.  It is not the Protocol-125 master
    decision and deliberately carries no scientific-execution authorization.
    """
    try:
        context = _validate_inputs(inputs)
        hashes_before = protocol125_final_matrix_input_hashes(inputs)
        parent_label = _validate_provenance(provenance, hashes_before)
    except (TypeError, ValueError, RuntimeError, KeyError, AttributeError) as error:
        return _invalid_result(f"pre-score provenance or structure: {error}")

    states = context["states"]
    lanes = {}
    try:
        lanes["Q53_Q33_position_spatial"] = score_state_pair_on_v_meshes(
            states["Q53_position"],
            states["Q33_position"],
            inputs.v_meshes,
            comparison_kind="Q53_Q33",
            groups=POSITION_SPATIAL_GROUPS,
        )
        lanes["Q53_Q33_position_q4_q5_images"] = (
            score_q4_q5_derivative_images_on_v_meshes(
                states["Q53_position"],
                states["Q33_position"],
                inputs.v_meshes,
                comparison_kind="Q53_Q33",
                state_name="position",
            )
        )
        lanes["Q53_Q33_acceleration"] = score_acceleration_pair_on_v_meshes(
            states["Q53_acceleration"],
            states["Q33_acceleration"],
            inputs.v_meshes,
            comparison_kind="Q53_Q33",
        )
        lanes["Q53_Q33_acceleration_q4_q5_images"] = (
            score_q4_q5_derivative_images_on_v_meshes(
                states["Q53_acceleration"],
                states["Q33_acceleration"],
                inputs.v_meshes,
                comparison_kind="Q53_Q33",
                state_name="acceleration",
            )
        )
        lanes["Q53_Q33_source_triplet"] = _score_source_triplet_matrix(inputs)
        dense_r = inputs.position_audit_meshes.dense_wall_r
        lanes["Q53_position_endpoint_z"] = score_state_endpoint_z_reproduction(
            states["Q53_position"], dense_r,
        )
        lanes["Q33_position_endpoint_z"] = score_state_endpoint_z_reproduction(
            states["Q33_position"], dense_r,
        )
        lanes["Q53_acceleration_endpoint_z"] = score_state_endpoint_z_reproduction(
            states["Q53_acceleration"], dense_r,
        )
        lanes["Q33_acceleration_endpoint_z"] = score_state_endpoint_z_reproduction(
            states["Q33_acceleration"], dense_r,
        )
        endpoint_conversion = score_acceleration_endpoint_conversion_pair(
            inputs.final_pair,
            inputs.row_implied_acceleration_z_source,
            inputs.row_implied_acceleration_z_dense,
            inputs.direct_Dz7_acceleration_z_source,
            dense_r,
        )
        for name in ACCELERATION_ENDPOINT_CONVERSION_LANES:
            lanes[name] = endpoint_conversion["lanes"][name]
        lanes["time_symmetric_velocity_endpoint_z"] = (
            score_time_symmetric_velocity_endpoint_z(inputs.velocity_endpoint_z)
        )
        dense_z = inputs.position_audit_meshes.dense_outer_z
        lanes["Q53_position_outer_derivative"] = (
            score_state_outer_derivative_reproduction(
                states["Q53_position"], dense_z,
            )
        )
        lanes["Q33_position_outer_derivative"] = (
            score_state_outer_derivative_reproduction(
                states["Q33_position"], dense_z,
            )
        )
        lanes["Q53_acceleration_outer_derivative"] = (
            score_state_outer_derivative_reproduction(
                states["Q53_acceleration"], dense_z,
            )
        )
        lanes["Q33_acceleration_outer_derivative"] = (
            score_state_outer_derivative_reproduction(
                states["Q33_acceleration"], dense_z,
            )
        )
        lanes["independent_dense_wall_position"] = (
            evaluate_protocol125_dense_wall_audit(
                inputs.final_pair, inputs.position_audit_meshes,
            )
        )
        lanes["independent_dense_outer_position"] = (
            evaluate_protocol125_dense_outer_delta_robin_audit(
                inputs.final_pair,
                inputs.reference_pair,
                inputs.position_audit_meshes,
            )
        )
        lanes["sealed_legacy_Q33_Q55_position_spatial_acceleration"] = (
            score_precomputed_groups_on_v_meshes(
                inputs.legacy_Q33_by_mesh,
                inputs.legacy_Q55_by_mesh,
                inputs.v_meshes,
                comparison_kind="legacy_Q33_Q55",
                required_groups=LEGACY_REQUIRED_GROUPS,
                component_orders=inputs.legacy_component_orders,
            )
        )
        lanes["sealed_legacy_Q33_Q55_source_triplet"] = (
            _score_legacy_source_triplet_matrix(inputs)
        )
        if tuple(lanes) != REQUIRED_FINAL_MATRIX_LANES:
            raise RuntimeError("final representation scorer lane inventory is incomplete")
        hashes_after = protocol125_final_matrix_input_hashes(inputs)
        changed = tuple(
            name for name in INPUT_HASH_KEYS
            if str(hashes_before[name]) != str(hashes_after[name])
        )
        if changed:
            raise ValueError(f"matrix inputs changed while scoring: {changed}")
    except (TypeError, ValueError, RuntimeError, KeyError, AttributeError) as error:
        return _invalid_result(
            f"scorer structure or stability: {error}",
            found_hashes=hashes_before,
            lanes=lanes,
        )

    failed = tuple(name for name in REQUIRED_FINAL_MATRIX_LANES if not lanes[name]["passed"])
    passed = not failed
    return MappingProxyType({
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "parent_label": parent_label,
        "classification": (
            "PASS-final-representation-matrix"
            if passed else "FAIL-final-representation-matrix"
        ),
        "complete": True,
        "provenance_valid": True,
        "passed": passed,
        "required_lane_order": REQUIRED_FINAL_MATRIX_LANES,
        "lanes": MappingProxyType(lanes),
        "failed_lanes": failed,
        "invalid_reasons": (),
        "source_fingerprint": context["source_fingerprint"],
        "endpoint_fingerprint": context["endpoint_fingerprint"],
        "input_hashes_before": hashes_before,
        "input_hashes_after": hashes_after,
        "inputs_stable_while_scoring": True,
        "constituent_logical_AND": True,
        "phase_a_authorized": False,
        "scientific_execution_authorized": False,
        "artifact_written": False,
    })
