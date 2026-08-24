"""Concrete append-only position lineage adapter for Protocol 125.

The generic :class:`PositionPayloadSnapshot` record deliberately does not
know how a constrained parent representation is laid out.  This module is
the production adapter between that generic record and the two concrete
Protocol-125 stages:

* the sealed, acceleration-free Q53/Q33 position pair; and
* the final shared position/acceleration Q53/Q33 pair.

Only position-owned subrecords are admitted to the invariant payload.  The
top-level shared compact and outer contracts are expected to acquire new
identities, while their embedded position records must remain byte-for-byte
unchanged.  Acceleration, source-second, and acceleration-outer records are
captured separately as appended children.

The adapter also evaluates both position members on the complete frozen mesh
union and requires exact binary64 identity between the two construction
stages.  It performs no solve, writes no artifact, and authorizes no parent.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.joint_parent_boundary_contracts import (
    NativeNormalizedCompactWallContract,
    Protocol125OuterOpenFaceDerivativeContract,
    Protocol125PositionOuterOpenFaceDerivativeContract,
)
from bhps.joint_parent_bulk_reference import (
    FiniteWallReferenceHermitePair,
    REFERENCE_CHANNEL_ORDER,
    SOURCE_STENCIL_WIDTH,
)
from bhps.joint_parent_position_audits import (
    bind_protocol125_position_audit_meshes,
)
from bhps.joint_parent_position_pair import PositionOnlyConstrainedHermitePair
from bhps.joint_parent_protocol125_sampling_lineage import (
    PositionPayloadSnapshot,
    Protocol125LineageError,
    validate_append_only_position_lineage,
    validate_protocol125_position_payload_group_order,
)
from bhps.joint_parent_representation import (
    NATIVE_CHANNEL_ORDER,
    RadialFirstConstrainedHermitePair,
)
from bhps.matched_staged_continuum import hash_arrays


LINEAGE_PROTOCOL = "Protocol-125-concrete-append-only-position-lineage-v1"
POSITION_MESH_DOMAIN_ORDER = (
    "source",
    "source_cell_midpoint",
    "V0",
    "V1",
    "V2",
    "dense_wall_lower",
    "dense_wall_upper",
    "dense_outer",
)
_COMPACT_POSITION_KEYS = (
    "background_values",
    "position_knots",
    "position_coefficients",
    "position_parent_r_max",
    "source_normal_knots",
    "source_normal_coefficients",
    "source_normal_parent_r_max",
    "position_ownership_mask",
)
_COMPACT_SHARED_CHILD_KEYS = (
    "source_second_normal_knots",
    "source_second_normal_coefficients",
    "source_second_normal_parent_r_max",
    "acceleration_ownership_mask",
)


def _immutable(value, dtype=None):
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _bitwise_equal(left, right):
    left = np.ascontiguousarray(np.asarray(left))
    right = np.ascontiguousarray(np.asarray(right))
    return bool(
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes() == right.tobytes()
    )


def _update_digest(digest, name, value):
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype == object:
        raise ValueError(f"lineage array {name} has forbidden object dtype")
    encoded = str(name).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())


def _fingerprint_arrays(arrays):
    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError("lineage fingerprint requires a nonempty array mapping")
    digest = hashlib.sha256()
    for name in sorted(arrays):
        _update_digest(digest, name, arrays[name])
    return digest.hexdigest()


def _ordered_digest_arrays(*values):
    """Reproduce the boundary contract's ordered source digest recipe."""
    digest = hashlib.sha256()
    for value in values:
        array = np.ascontiguousarray(np.asarray(value))
        if array.dtype == object:
            raise ValueError("ordered lineage digest forbids object arrays")
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({
            str(name): _freeze(item) for name, item in value.items()
        })
    if isinstance(value, np.ndarray):
        return _immutable(value)
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _position_states(position_only_pair, shared_pair):
    if not isinstance(position_only_pair, PositionOnlyConstrainedHermitePair):
        raise TypeError(
            "position lineage requires the sealed position-only Q53/Q33 pair"
        )
    if not isinstance(shared_pair, RadialFirstConstrainedHermitePair):
        raise TypeError(
            "position lineage requires the final shared Q53/Q33 pair"
        )
    position_only = {
        "Q53": position_only_pair.primary,
        "Q33": position_only_pair.comparator,
    }
    shared = {
        "Q53": shared_pair.primary.position,
        "Q33": shared_pair.comparator.position,
    }
    for stage, states in (("position-only", position_only), ("shared", shared)):
        for degree, state in states.items():
            expected_degree = 5 if degree == "Q53" else 3
            if (
                str(state.state_name) != "position"
                or int(state.z_degree) != expected_degree
            ):
                raise ValueError(
                    f"{stage} {degree} member is not the required position state"
                )
    return position_only, shared


def _contracts(states, stage):
    primary = states["Q53"]
    comparator = states["Q33"]
    if not (
        primary.compact_wall_contract is comparator.compact_wall_contract
        or (
            primary.compact_wall_contract_id
            == comparator.compact_wall_contract_id
            and primary.compact_wall_contract_fingerprint
            == comparator.compact_wall_contract_fingerprint
        )
    ):
        raise Protocol125LineageError(
            f"{stage} Q53/Q33 compact contract identities differ"
        )
    if not (
        primary.outer_open_face_contract is comparator.outer_open_face_contract
        or (
            primary.outer_open_face_contract_id
            == comparator.outer_open_face_contract_id
            and primary.outer_open_face_contract_fingerprint
            == comparator.outer_open_face_contract_fingerprint
        )
    ):
        raise Protocol125LineageError(
            f"{stage} Q53/Q33 outer contract identities differ"
        )
    compact = primary.compact_wall_contract
    outer = primary.outer_open_face_contract
    if not isinstance(compact, NativeNormalizedCompactWallContract):
        raise TypeError(
            f"{stage} compact contract is not the native normalized wall contract"
        )
    if stage == "position-only":
        if compact.source_second_normal_context is not None:
            raise Protocol125LineageError(
                "position-only compact contract already contains source-second data"
            )
        if not isinstance(outer, Protocol125PositionOuterOpenFaceDerivativeContract):
            raise TypeError(
                "position-only outer contract is not the Protocol-125 position contract"
            )
        position_outer = outer
    else:
        if compact.source_second_normal_context is None:
            raise Protocol125LineageError(
                "shared compact contract lacks appended source-second data"
            )
        if not isinstance(outer, Protocol125OuterOpenFaceDerivativeContract):
            raise TypeError(
                "shared outer contract is not the Protocol-125 two-state contract"
            )
        position_outer = outer.position_contract
    return compact, outer, position_outer


def _reference_binding(reference_pair, states, position_outer):
    if not isinstance(reference_pair, FiniteWallReferenceHermitePair):
        raise TypeError(
            "position lineage requires the sealed finite-wall Q53/Q33 reference pair"
        )
    primary = reference_pair.primary
    comparator = reference_pair.comparator
    state = states["Q53"]
    arrays = (
        primary.source_z,
        primary.source_r,
        primary.source_values,
        primary.endpoint_z_first,
        comparator.source_z,
        comparator.source_r,
        comparator.source_values,
        comparator.endpoint_z_first,
    )
    if any(np.asarray(value).flags.writeable for value in arrays):
        raise ValueError("finite-wall reference pair must be sealed before lineage")
    if not (
        _bitwise_equal(primary.source_z, state.source_z)
        and _bitwise_equal(primary.source_r, state.source_r)
        and _bitwise_equal(primary.source_z, comparator.source_z)
        and _bitwise_equal(primary.source_r, comparator.source_r)
        and _bitwise_equal(primary.source_values, comparator.source_values)
        and _bitwise_equal(
            primary.endpoint_z_first, comparator.endpoint_z_first,
        )
        and tuple(primary.channel_order) == REFERENCE_CHANNEL_ORDER
        and tuple(comparator.channel_order) == REFERENCE_CHANNEL_ORDER
        and int(primary.stencil_width) == SOURCE_STENCIL_WIDTH
        and int(comparator.stencil_width) == SOURCE_STENCIL_WIDTH
        and _bitwise_equal(position_outer.source_z, primary.source_z)
        and _bitwise_equal(position_outer.source_r, primary.source_r)
    ):
        raise Protocol125LineageError(
            "finite-wall reference/source identity differs from the position stage"
        )
    radial_operator = derivative_matrix(
        primary.source_r, 1, SOURCE_STENCIL_WIDTH,
    )
    q_slot = REFERENCE_CHANNEL_ORDER.index("q")
    phi_slot = REFERENCE_CHANNEL_ORDER.index("Phi")
    q = primary.source_values[:, :, q_slot]
    phi = primary.source_values[:, :, phi_slot]
    expected_outer = np.stack((
        q[:, -1],
        (radial_operator @ q.T).T[:, -1],
        phi[:, -1],
        (radial_operator @ phi.T).T[:, -1],
    ), axis=-1)
    if not _bitwise_equal(expected_outer, position_outer.reference_outer_values):
        raise Protocol125LineageError(
            "position outer contract is not bound to the supplied full reference"
        )
    expected_source_reference_fingerprint = _ordered_digest_arrays(
        np.asarray("fresh-reference-width-seven-outer-v1"),
        primary.source_z,
        primary.source_r,
        q,
        phi,
    )
    if (
        expected_source_reference_fingerprint
        != position_outer.source_reference_fingerprint
    ):
        raise Protocol125LineageError(
            "position outer source/reference fingerprint differs from the "
            "supplied full reference"
        )
    return primary


def _compact_position_record(compact, stage):
    record = compact.coefficient_arrays()
    expected = set(_COMPACT_POSITION_KEYS) | {"position_only"}
    if stage == "shared":
        expected |= set(_COMPACT_SHARED_CHILD_KEYS)
    if set(record) != expected:
        raise Protocol125LineageError(
            f"{stage} compact contract has an unclassified subrecord"
        )
    position_only_flag = np.asarray(record["position_only"])
    expected_flag = np.asarray(stage == "position-only")
    if not _bitwise_equal(position_only_flag, expected_flag):
        raise Protocol125LineageError(
            f"{stage} compact position-only marker is inconsistent"
        )
    return {
        name: np.asarray(record[name]).copy() for name in _COMPACT_POSITION_KEYS
    }


def _position_state_coefficients(state):
    arrays = {
        "source_z": state.source_z,
        "source_r": state.source_r,
        **state.radial_channels.coefficient_arrays("radial_channels"),
        **state.radial_anisotropy_numerator.coefficient_arrays(
            "radial_anisotropy_numerator"
        ),
        "stored_z_first_endpoints": state.stored_z_first_endpoints,
        "outer_ownership_mask": state.outer_ownership_mask,
        "z_degree": np.asarray(state.z_degree),
        "state_name": np.asarray(state.state_name),
    }
    arrays["position_state_subrecord_fingerprint"] = np.asarray(
        _fingerprint_arrays(arrays)
    )
    return arrays


def _stage_groups(states, compact, position_outer, reference):
    state = states["Q53"]
    source_channels = np.asarray(
        state.radial_channels.evaluate_s(state.source_r, 0), dtype=float,
    )
    expected = (
        len(state.source_z), len(state.source_r), len(NATIVE_CHANNEL_ORDER),
    )
    if source_channels.shape != expected or not np.all(np.isfinite(source_channels)):
        raise Protocol125LineageError(
            "stored position radial coefficients returned invalid source data"
        )
    comparator_channels = np.asarray(
        states["Q33"].radial_channels.evaluate_s(state.source_r, 0),
        dtype=float,
    )
    if not _bitwise_equal(source_channels, comparator_channels):
        raise Protocol125LineageError(
            "position Q53/Q33 stored native source data differ bitwise"
        )
    if not _bitwise_equal(source_channels[:, -1], position_outer.source_position):
        raise Protocol125LineageError(
            "position outer contract source values differ from the stored source"
        )
    velocity = np.zeros((len(state.source_z), len(state.source_r), 9))
    if np.any(velocity != 0.0) or np.any(np.signbit(velocity)):
        raise AssertionError("lineage positive-zero tangent construction failed")
    source_normal = compact.source_normal_context.jets(state.source_r, 1)[0]
    if source_normal.shape != (2, len(state.source_r), 1):
        raise Protocol125LineageError(
            "compact normal-source trace has the wrong shape"
        )
    source = {
        "source_z": state.source_z,
        "source_r": state.source_r,
        "native_channel_order": np.asarray(NATIVE_CHANNEL_ORDER),
        **{
            f"completed_native_{name}": source_channels[:, :, index]
            for index, name in enumerate(NATIVE_CHANNEL_ORDER)
        },
        "coordinate_time_velocity": velocity,
        "source_normal_wall_H_z": source_normal[:, :, 0],
        "reference_channel_order": np.asarray(REFERENCE_CHANNEL_ORDER),
        "finite_wall_reference_q": reference.source_values[
            :, :, REFERENCE_CHANNEL_ORDER.index("q")
        ],
        "finite_wall_reference_Phi": reference.source_values[
            :, :, REFERENCE_CHANNEL_ORDER.index("Phi")
        ],
        "finite_wall_reference_endpoint_z_first": reference.endpoint_z_first,
        "finite_wall_reference_source_record_fingerprint": np.asarray(
            _reference_source_record_fingerprint(reference)
        ),
        "source_reference_fingerprint": np.asarray(
            position_outer.source_reference_fingerprint
        ),
    }
    compact_record = _compact_position_record(compact, (
        "position-only"
        if compact.source_second_normal_context is None else "shared"
    ))
    compact_record["position_contract_identifier"] = np.asarray(
        "native-normalized-compact-wall-position-subrecord-v1"
    )
    compact_record["position_subrecord_fingerprint"] = np.asarray(
        _fingerprint_arrays(compact_record)
    )
    outer = {
        **position_outer.coefficient_arrays(),
        "position_contract_identifier": np.asarray(position_outer.identifier),
    }
    outer["position_subrecord_fingerprint"] = np.asarray(
        _fingerprint_arrays(outer)
    )
    coefficients = {}
    for degree in ("Q53", "Q33"):
        for name, value in _position_state_coefficients(states[degree]).items():
            coefficients[f"{degree}_{name}"] = value
    groups = {
        "source": source,
        "compact": compact_record,
        "outer": outer,
        "coefficients": coefficients,
    }
    validate_protocol125_position_payload_group_order(groups)
    return groups


def _reference_source_record_fingerprint(reference):
    """Hash the common source record from either sealed reference member.

    ``reference`` is one member after Q53/Q33 byte-identity validation.  The
    member-level source record is the invariant relevant to lineage; compact
    interpolation degree is intentionally not part of this fingerprint.
    """
    return _fingerprint_arrays({
        "source_z": reference.source_z,
        "source_r": reference.source_r,
        "source_values": reference.source_values,
        "endpoint_z_first": reference.endpoint_z_first,
        "stencil_width": np.asarray(reference.stencil_width),
        "channel_order": np.asarray(reference.channel_order),
    })


def _shared_children(shared_pair, compact, outer):
    compact_record = compact.coefficient_arrays()
    outer_record = outer.coefficient_arrays()
    expected_position_prefix = {
        f"position_{name}" for name in outer.position_contract.coefficient_arrays()
    }
    expected_outer = expected_position_prefix | {
        "position_contract_identifier",
        "source_acceleration_channel_order",
        "source_acceleration",
        "acceleration_r_first",
        "acceleration_source_fingerprint",
        "acceleration_ownership_mask",
        "acceleration_derivation_recipe",
        "corner_policy",
        "shared_state_recipe",
    }
    if set(outer_record) != expected_outer:
        raise Protocol125LineageError(
            "shared outer contract has an unclassified child subrecord"
        )
    children = {
        f"compact_{name}": compact_record[name]
        for name in _COMPACT_SHARED_CHILD_KEYS
    }
    source_second = compact.source_second_normal_context.jets(
        shared_pair.primary.position.source_r, 1,
    )[0]
    if source_second.shape != (
        2, len(shared_pair.primary.position.source_r), 1,
    ):
        raise Protocol125LineageError(
            "shared source-second compact trace has the wrong shape"
        )
    children["compact_source_second_normal_wall"] = source_second[:, :, 0]
    children["compact_source_second_context_present"] = np.asarray(True)
    for name in sorted(set(outer_record)-expected_position_prefix-{
        "position_contract_identifier",
    }):
        children[f"outer_{name}"] = outer_record[name]
    for degree, state in (
        ("Q53", shared_pair.primary.acceleration),
        ("Q33", shared_pair.comparator.acceleration),
    ):
        for name, value in _position_state_coefficients(state).items():
            # The helper is numerical-state generic; relabel its lone metadata
            # fingerprint so it is not mistaken for an invariant position hash.
            child_name = name.replace(
                "position_state_subrecord_fingerprint",
                "acceleration_state_subrecord_fingerprint",
            )
            children[f"{degree}_acceleration_{child_name}"] = value
    children.update({
        "shared_primary_source_fingerprint": np.asarray(
            shared_pair.primary.source_fingerprint
        ),
        "shared_primary_endpoint_fingerprint": np.asarray(
            shared_pair.primary.endpoint_fingerprint
        ),
        "shared_comparator_source_fingerprint": np.asarray(
            shared_pair.comparator.source_fingerprint
        ),
        "shared_comparator_endpoint_fingerprint": np.asarray(
            shared_pair.comparator.endpoint_fingerprint
        ),
    })
    return children


def _archive_fingerprint(pair, prefix):
    arrays = pair.coefficient_arrays(prefix)
    return _fingerprint_arrays(arrays)


def _mesh_domains(meshes):
    source_lower = float(meshes.source_z[0])
    source_upper = float(meshes.source_z[-1])
    radius = float(meshes.source_r[-1])
    return {
        "source": (meshes.source_z, meshes.source_r),
        "source_cell_midpoint": (meshes.midpoint_z, meshes.midpoint_r),
        "V0": (meshes.V0_z, meshes.V0_r),
        "V1": (meshes.V1_z, meshes.V1_r),
        "V2": (meshes.V2_z, meshes.V2_r),
        "dense_wall_lower": (
            np.asarray([source_lower]), meshes.dense_wall_r,
        ),
        "dense_wall_upper": (
            np.asarray([source_upper]), meshes.dense_wall_r,
        ),
        "dense_outer": (meshes.dense_outer_z, np.asarray([radius])),
    }


def validate_protocol125_position_mesh_identity(
    position_only_pair,
    shared_pair,
):
    """Require bitwise stage identity on the complete frozen mesh union."""
    position_only, shared = _position_states(position_only_pair, shared_pair)
    meshes = bind_protocol125_position_audit_meshes(position_only_pair)
    domains = _mesh_domains(meshes)
    if tuple(domains) != POSITION_MESH_DOMAIN_ORDER:
        raise AssertionError("Protocol-125 position lineage mesh union is incomplete")
    for degree in ("Q53", "Q33"):
        if not (
            _bitwise_equal(position_only[degree].source_z, shared[degree].source_z)
            and _bitwise_equal(
                position_only[degree].source_r, shared[degree].source_r,
            )
        ):
            raise Protocol125LineageError(
                f"{degree} position source coordinates changed at shared transition"
            )
    records = {}
    total_samples = 0
    for domain, (z, r) in domains.items():
        z = np.asarray(z, dtype=float)
        r = np.asarray(r, dtype=float)
        record = {
            "coordinate_sha256": hash_arrays(z, r),
            "shape": (len(z), len(r)),
            "sample_count": int(len(z)*len(r)),
        }
        total_samples += len(z)*len(r)
        for degree in ("Q53", "Q33"):
            early_reduced = np.asarray(
                position_only[degree].evaluate_reduced(z, r), dtype=float,
            )
            shared_reduced = np.asarray(
                shared[degree].evaluate_reduced(z, r), dtype=float,
            )
            early_coordinate = np.asarray(
                position_only[degree].evaluate_coordinate_components(z, r),
                dtype=float,
            )
            shared_coordinate = np.asarray(
                shared[degree].evaluate_coordinate_components(z, r),
                dtype=float,
            )
            if not _bitwise_equal(early_reduced, shared_reduced):
                raise Protocol125LineageError(
                    f"{degree} reduced position changed bitwise on {domain}"
                )
            if not _bitwise_equal(early_coordinate, shared_coordinate):
                raise Protocol125LineageError(
                    f"{degree} coordinate position changed bitwise on {domain}"
                )
            record[f"{degree}_reduced_sha256"] = hash_arrays(early_reduced)
            record[f"{degree}_coordinate_sha256"] = hash_arrays(
                early_coordinate
            )
            record[f"{degree}_passed"] = True
        record["passed"] = True
        records[domain] = record
    mesh_union_fingerprint = _fingerprint_arrays({
        f"{domain}_{axis}": value
        for domain, coordinates in domains.items()
        for axis, value in zip(("z", "r"), coordinates)
    })
    return _freeze({
        "protocol": "Protocol-125-position-stage-mesh-identity-v1",
        "domain_order": POSITION_MESH_DOMAIN_ORDER,
        "domains": records,
        "mesh_union_fingerprint": mesh_union_fingerprint,
        "mesh_binding_fingerprint": meshes.fingerprint(),
        "sample_count_per_degree": int(total_samples),
        "degrees": ("Q53", "Q33"),
        "reduced_and_coordinate_values_compared_bitwise": True,
        "tolerance_comparison_used": False,
        "passed": True,
    })


def build_protocol125_append_only_position_lineage(
    position_only_pair,
    shared_pair,
    finite_wall_reference_pair,
):
    """Build and validate the concrete Protocol-125 lineage record.

    The returned mapping contains both generic snapshots, the generic lineage
    gate record, and the independent full-mesh bitwise evaluation record.
    Any invariant drift raises :class:`Protocol125LineageError`; there is no
    tolerance-based fallback.
    """
    position_only, shared = _position_states(position_only_pair, shared_pair)
    early_compact, _, early_position_outer = _contracts(
        position_only, "position-only",
    )
    shared_compact, shared_outer, shared_position_outer = _contracts(
        shared, "shared",
    )
    if shared_position_outer.identifier != early_position_outer.identifier:
        raise Protocol125LineageError(
            "shared outer contract does not retain the position-only contract"
        )
    early_reference = _reference_binding(
        finite_wall_reference_pair, position_only, early_position_outer,
    )
    shared_reference = _reference_binding(
        finite_wall_reference_pair, shared, shared_position_outer,
    )
    if _reference_source_record_fingerprint(
        early_reference
    ) != _reference_source_record_fingerprint(shared_reference):
        raise AssertionError("finite-wall reference binding changed during lineage")

    early_groups = _stage_groups(
        position_only, early_compact, early_position_outer, early_reference,
    )
    shared_groups = _stage_groups(
        shared, shared_compact, shared_position_outer, shared_reference,
    )
    early_state = position_only["Q53"]
    shared_state = shared["Q53"]
    early_snapshot = PositionPayloadSnapshot.capture_position_only(
        early_groups,
        compact_identifier=early_state.compact_wall_contract_id,
        compact_fingerprint=early_state.compact_wall_contract_fingerprint,
        outer_identifier=early_state.outer_open_face_contract_id,
        outer_fingerprint=early_state.outer_open_face_contract_fingerprint,
        archive_fingerprint=_archive_fingerprint(
            position_only_pair, "protocol125_position_only_pair",
        ),
    )
    shared_snapshot = PositionPayloadSnapshot.capture_shared(
        shared_groups,
        parent=early_snapshot,
        appended_children=_shared_children(
            shared_pair, shared_compact, shared_outer,
        ),
        compact_identifier=shared_state.compact_wall_contract_id,
        compact_fingerprint=shared_state.compact_wall_contract_fingerprint,
        outer_identifier=shared_state.outer_open_face_contract_id,
        outer_fingerprint=shared_state.outer_open_face_contract_fingerprint,
        archive_fingerprint=_archive_fingerprint(
            shared_pair, "protocol125_shared_pair",
        ),
    )
    lineage = validate_append_only_position_lineage(
        early_snapshot, shared_snapshot,
    )
    mesh_identity = validate_protocol125_position_mesh_identity(
        position_only_pair, shared_pair,
    )
    return _freeze({
        "protocol": LINEAGE_PROTOCOL,
        "position_only_snapshot": early_snapshot,
        "shared_snapshot": shared_snapshot,
        "append_only_validation": lineage,
        "mesh_identity": mesh_identity,
        "reference_source_fingerprint": _reference_source_record_fingerprint(
            early_reference
        ),
        "top_level_composite_identities_evolved": True,
        "invariant_position_payload_bitwise": True,
        "acceleration_children_recorded": True,
        "passed": True,
    })
