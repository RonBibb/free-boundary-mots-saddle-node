"""Append-only construction of the final Protocol-125 shared representation.

This module starts from an already qualified position-only Q53/Q33 pair and
an already completed compatible acceleration.  It appends the acceleration
compact/outer contexts, derives the sole row-implied endpoint conversion, and
constructs the final shared Q53/Q33 pair in memory.  It performs no parent
solve, fixed-point iteration, scientific adjudication, artifact write, or
execution authorization.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.joint_parent_boundary_contracts import (
    NativeNormalizedCompactWallContract,
    Protocol125PositionOuterOpenFaceDerivativeContract,
    derive_protocol125_outer_derivative_bundle,
)
from bhps.joint_parent_endpoint_audits import (
    convert_native_fd_acceleration_z_comparator,
    convert_row_implied_acceleration_z,
)
from bhps.joint_parent_position_pair import PositionOnlyConstrainedHermitePair
from bhps.joint_parent_position_state import native_channel_mapping_from_reduced
from bhps.joint_parent_protocol125_sampling_lineage import (
    Protocol125BulkAccelerationSampler,
)
from bhps.joint_parent_refinement_diagnostics import frozen_validation_meshes
from bhps.joint_parent_representation import (
    NATIVE_CHANNEL_ORDER,
    RadialFirstConstrainedHermitePair,
)


PROTOCOL_IDENTIFIER = "Protocol-125-shared-representation-builder-v1"
SOURCE_TRIPLET_KEYS = ("source", "source_time", "source_second_time")
_INVARIANT_COMPACT_KEYS = (
    "background_values",
    "position_knots",
    "position_coefficients",
    "position_parent_r_max",
    "source_normal_knots",
    "source_normal_coefficients",
    "source_normal_parent_r_max",
)


class Protocol125SharedRepresentationError(RuntimeError):
    """Raised when the final contract transition is not append-only."""


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


def _scaled_linf(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or not (
        np.all(np.isfinite(left)) and np.all(np.isfinite(right))
    ):
        raise ValueError("scaled comparison arrays are invalid")
    scale = np.maximum.reduce((np.ones_like(left), np.abs(left), np.abs(right)))
    return float(np.max(np.abs(left-right)/scale))


def _update_digest(digest, name, value):
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype == object:
        raise ValueError(f"shared representation record {name} has object dtype")
    digest.update(str(name).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())


def _fingerprint_arrays(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        _update_digest(digest, name, value)
    return digest.hexdigest()


def _stack_native(mapping):
    if not isinstance(mapping, Mapping) or tuple(mapping) != NATIVE_CHANNEL_ORDER:
        raise ValueError("native channel mapping order differs")
    value = np.stack(tuple(np.asarray(mapping[name]) for name in NATIVE_CHANNEL_ORDER), axis=-1)
    if not np.all(np.isfinite(value)):
        raise ValueError("native channel stack is nonfinite")
    return value


def _endpoint_mapping(values):
    values = np.asarray(values, dtype=float)
    return {
        name: values[:, :, index].copy()
        for index, name in enumerate(NATIVE_CHANNEL_ORDER)
    }


def _state_invariant_arrays(state, prefix):
    arrays = {
        f"{prefix}_source_z": np.asarray(state.source_z),
        f"{prefix}_source_r": np.asarray(state.source_r),
        f"{prefix}_stored_z_first_endpoints": np.asarray(
            state.stored_z_first_endpoints
        ),
        f"{prefix}_outer_ownership_mask": np.asarray(
            state.outer_ownership_mask
        ),
        f"{prefix}_z_degree": np.asarray(state.z_degree),
    }
    arrays.update(state.radial_channels.coefficient_arrays(
        f"{prefix}_radial_channels"
    ))
    arrays.update(state.radial_anisotropy_numerator.coefficient_arrays(
        f"{prefix}_radial_anisotropy_numerator"
    ))
    return arrays


def _require_bitwise_mapping(left, right, label):
    if set(left) != set(right):
        raise Protocol125SharedRepresentationError(
            f"{label} invariant names changed"
        )
    changed = tuple(
        name for name in sorted(left)
        if not _bitwise_equal(left[name], right[name])
    )
    if changed:
        raise Protocol125SharedRepresentationError(
            f"{label} position invariants changed: {', '.join(changed)}"
        )


def _source_dz7_acceleration_endpoint(acceleration_native, z, r):
    """Form the independent native width-seven compact-endpoint comparator."""
    acceleration = np.asarray(acceleration_native, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    dz = derivative_matrix(z, 1, 7)
    du = derivative_matrix((r/float(r[-1]))**2, 1, 7)
    if hasattr(dz, "toarray"):
        dz = dz.toarray()
    if hasattr(du, "toarray"):
        du = du.toarray()
    dz = np.asarray(dz, dtype=float)
    du = np.asarray(du, dtype=float)
    endpoint = np.einsum("wi,irc->wrc", dz[[0, -1]], acceleration)

    index = {name: slot for slot, name in enumerate(NATIVE_CHANNEL_ORDER)}
    h_perp = index["h_perp"]
    h_rr = index["h_rr"]
    numerator = acceleration[:, :, h_rr]-acceleration[:, :, h_perp]
    numerator_z = np.einsum("wi,ir->wr", dz[[0, -1]], numerator)
    endpoint[:, :, h_rr] = endpoint[:, :, h_perp]+numerator_z
    # Exact regular-axis input is an algebraic identity, not a tolerance repair.
    if not (
        np.array_equal(numerator[:, 0], np.zeros(len(z)))
        and not np.any(np.signbit(numerator[:, 0]))
    ):
        raise ValueError("completed acceleration numerator is not positive zero")
    endpoint[:, 0, h_rr] = endpoint[:, 0, h_perp]

    endpoint_s_first = np.einsum("ij,wjc->wic", du, endpoint)
    endpoint_numerator = endpoint[:, :, h_rr]-endpoint[:, :, h_perp]
    endpoint_numerator_s = np.einsum("ij,wj->wi", du, endpoint_numerator)
    endpoint_s_first[:, :, h_rr] = (
        endpoint_s_first[:, :, h_perp]+endpoint_numerator_s
    )
    return endpoint, endpoint_s_first, {
        "compact_operator": _immutable(dz),
        "radial_u_operator": _immutable(du),
        "native_numerator": _immutable(numerator),
        "native_numerator_z": _immutable(numerator_z),
        "endpoint_numerator_s_first": _immutable(endpoint_numerator_s),
    }


@dataclass(frozen=True)
class Protocol125SharedRepresentationBuild:
    """Immutable in-memory result of the append-only shared-state build."""

    final_pair: object
    compact_contract: object
    outer_contract: object
    bulk_sampler: Protocol125BulkAccelerationSampler
    position_pair_fingerprint: str
    source_triplet_fingerprint: str
    position_endpoint_native: np.ndarray
    acceleration_endpoint_native: np.ndarray
    acceleration_endpoint_s_first_native: np.ndarray
    row_implied_source: Mapping
    row_implied_dense: Mapping
    direct_fd_source: Mapping
    direct_fd_inputs: Mapping
    acceleration_wall_s_jet_inputs: Mapping
    checks: Mapping

    def coefficient_arrays(self, prefix="protocol125_shared_build"):
        arrays = {
            f"{prefix}_protocol_identifier": np.asarray(PROTOCOL_IDENTIFIER),
            f"{prefix}_position_pair_fingerprint": np.asarray(
                self.position_pair_fingerprint
            ),
            f"{prefix}_source_triplet_fingerprint": np.asarray(
                self.source_triplet_fingerprint
            ),
            f"{prefix}_final_pair_fingerprint": np.asarray(
                _fingerprint_arrays(self.final_pair.coefficient_arrays())
            ),
            f"{prefix}_compact_identifier": np.asarray(
                self.compact_contract.identifier
            ),
            f"{prefix}_outer_identifier": np.asarray(
                self.outer_contract.identifier
            ),
            f"{prefix}_position_endpoint_native": np.asarray(
                self.position_endpoint_native
            ),
            f"{prefix}_acceleration_endpoint_native": np.asarray(
                self.acceleration_endpoint_native
            ),
            f"{prefix}_acceleration_endpoint_s_first_native": np.asarray(
                self.acceleration_endpoint_s_first_native
            ),
        }
        for group_name, group in (
            ("row_source", self.row_implied_source),
            ("row_dense", self.row_implied_dense),
            ("fd_source", self.direct_fd_source),
            ("fd_input", self.direct_fd_inputs),
            ("s_jet", self.acceleration_wall_s_jet_inputs),
        ):
            for name, value in group.items():
                if isinstance(value, (str, bool, int, float, np.generic, np.ndarray)):
                    arrays[f"{prefix}_{group_name}_{name}"] = np.asarray(value)
        for name, value in self.checks.items():
            arrays[f"{prefix}_check_{name}"] = np.asarray(value)
        arrays.update(self.bulk_sampler.coefficient_arrays(
            f"{prefix}_bulk_sampler"
        ))
        return arrays

    def fingerprint(self):
        return _fingerprint_arrays(self.coefficient_arrays())


def build_protocol125_shared_representation(
    parent,
    background,
    position_pair,
    position_state_record,
    compatible_acceleration,
    bulk_acceleration,
    source_triplet,
):
    """Append acceleration data and construct the final shared Q53/Q33 pair."""
    if not isinstance(parent, Mapping) or not isinstance(background, Mapping):
        raise TypeError("shared representation requires parent/background mappings")
    if not isinstance(position_pair, PositionOnlyConstrainedHermitePair):
        raise TypeError("shared representation requires the sealed position-only pair")
    if not isinstance(position_state_record, Mapping):
        raise TypeError("shared representation requires the position construction record")
    if not isinstance(source_triplet, Mapping) or set(source_triplet) != set(
        SOURCE_TRIPLET_KEYS
    ):
        raise ValueError("shared representation source triplet is incomplete")

    z = np.asarray(parent.get("z"), dtype=float)
    r = np.asarray(parent.get("r"), dtype=float)
    position = np.asarray(parent.get("position"), dtype=float)
    acceleration = np.asarray(compatible_acceleration, dtype=float)
    bulk = np.asarray(bulk_acceleration, dtype=float)
    expected = (len(z), len(r), 9)
    if (
        z.ndim != 1
        or r.ndim != 1
        or len(z) < 7
        or len(r) < 7
        or r[0] != 0.0
        or np.signbit(r[0])
        or np.any(np.diff(z) <= 0.0)
        or np.any(np.diff(r) <= 0.0)
        or position.shape != expected
        or acceleration.shape != expected
        or bulk.shape != expected
        or not all(np.all(np.isfinite(value)) for value in (
            z, r, position, acceleration, bulk,
        ))
    ):
        raise ValueError("shared representation source arrays are invalid")
    for state in (position_pair.primary, position_pair.comparator):
        if not (
            _bitwise_equal(state.source_z, z)
            and _bitwise_equal(state.source_r, r)
        ):
            raise ValueError("position-only pair source grid differs from parent")
    for name in SOURCE_TRIPLET_KEYS:
        value = np.asarray(source_triplet[name], dtype=float)
        if value.shape != (len(z), len(r), 3) or not np.all(np.isfinite(value)):
            raise ValueError(f"source triplet {name} is invalid")

    if not (
        np.all(acceleration[:, :, 0] == 0.0)
        and not np.any(np.signbit(acceleration[:, :, 0]))
    ):
        raise ValueError("compatible h_z0 acceleration must be IEEE positive zero")

    old_compact = position_pair.primary.compact_wall_contract
    old_outer = position_pair.primary.outer_open_face_contract
    if not isinstance(old_compact, NativeNormalizedCompactWallContract):
        raise TypeError("position-only compact contract has the wrong type")
    if old_compact.source_second_normal_context is not None:
        raise ValueError("position-only compact contract already has acceleration")
    if not isinstance(old_outer, Protocol125PositionOuterOpenFaceDerivativeContract):
        raise TypeError("position-only outer contract has the wrong type")
    if not (
        old_compact is position_pair.comparator.compact_wall_contract
        and old_outer is position_pair.comparator.outer_open_face_contract
    ):
        raise ValueError("Q53/Q33 position contracts are not the same objects")

    position_mapping = native_channel_mapping_from_reduced(position, r)
    acceleration_mapping = native_channel_mapping_from_reduced(acceleration, r)
    acceleration_stack = _stack_native(acceleration_mapping)
    position_stack = _stack_native(position_mapping)
    source_normal = np.asarray(
        position_state_record.get("source_normal_wall"), dtype=float,
    )
    recorded_endpoint = np.asarray(
        position_state_record.get("endpoint_z_first"), dtype=float,
    )
    if (
        source_normal.shape != (2, len(r))
        or recorded_endpoint.shape
        != (2, len(r), len(NATIVE_CHANNEL_ORDER))
        or not all(np.all(np.isfinite(value)) for value in (
            source_normal, recorded_endpoint,
        ))
    ):
        raise ValueError("position-only construction record is incomplete")
    final_source_normal = np.asarray(source_triplet["source"])[
        [0, -1], :, 1
    ]
    if not _bitwise_equal(source_normal, final_source_normal):
        raise Protocol125SharedRepresentationError(
            "final source normal trace differs from the sealed position trace"
        )
    if not (
        _bitwise_equal(recorded_endpoint, position_pair.primary.stored_z_first_endpoints)
        and _bitwise_equal(recorded_endpoint, position_pair.comparator.stored_z_first_endpoints)
    ):
        raise Protocol125SharedRepresentationError(
            "position endpoint trace differs from the sealed pair"
        )

    before_compact = old_compact.coefficient_arrays()
    compact = old_compact.append_acceleration(
        r,
        np.asarray(source_triplet["source_second_time"])[[0, -1], :, 1],
    )
    after_compact = compact.coefficient_arrays()
    if any(
        not _bitwise_equal(before_compact[name], after_compact[name])
        for name in _INVARIANT_COMPACT_KEYS
    ):
        raise Protocol125SharedRepresentationError(
            "append-only compact transition changed a position subrecord"
        )
    if compact.identifier == old_compact.identifier:
        raise Protocol125SharedRepresentationError(
            "shared compact identifier did not evolve"
        )

    position_endpoint = compact.z_first_s_jets(
        state_name="position",
        radius=r,
        wall_value_s_jets=(position_stack[[0, -1]],),
    )[0]
    if not _bitwise_equal(position_endpoint, recorded_endpoint):
        raise Protocol125SharedRepresentationError(
            "shared compact contract changed the position endpoint data"
        )
    acceleration_endpoint = compact.z_first_s_jets(
        state_name="acceleration",
        radius=r,
        wall_value_s_jets=(acceleration_stack[[0, -1]],),
    )[0]
    outer = derive_protocol125_outer_derivative_bundle(
        old_outer, acceleration_mapping,
    )
    if outer.position_contract is not old_outer or outer.identifier == old_outer.identifier:
        raise Protocol125SharedRepresentationError(
            "shared outer transition did not append to the sealed position contract"
        )

    final_pair = RadialFirstConstrainedHermitePair.build(
        z,
        r,
        position_mapping,
        acceleration_mapping,
        _endpoint_mapping(position_endpoint),
        _endpoint_mapping(acceleration_endpoint),
        compact_wall_contract=compact,
        outer_open_face_contract=outer,
        parent_r_max=float(r[-1]),
    )
    for label, old_state, new_state in (
        ("Q53", position_pair.primary, final_pair.primary.position),
        ("Q33", position_pair.comparator, final_pair.comparator.position),
    ):
        _require_bitwise_mapping(
            _state_invariant_arrays(old_state, "position"),
            _state_invariant_arrays(new_state, "position"),
            label,
        )

    source_position_q53_bitwise = _bitwise_equal(
        position_pair.primary.evaluate_reduced(z, r),
        final_pair.primary.position.evaluate_reduced(z, r),
    )
    source_position_q33_bitwise = _bitwise_equal(
        position_pair.comparator.evaluate_reduced(z, r),
        final_pair.comparator.position.evaluate_reduced(z, r),
    )
    if not (source_position_q53_bitwise and source_position_q33_bitwise):
        raise Protocol125SharedRepresentationError(
            "shared build changed a source-mesh position evaluation"
        )

    q53_acceleration_reproduction = _scaled_linf(
        final_pair.primary.acceleration.evaluate_reduced(z, r), acceleration,
    )
    q33_acceleration_reproduction = _scaled_linf(
        final_pair.comparator.acceleration.evaluate_reduced(z, r), acceleration,
    )

    acceleration_radial = final_pair.primary.acceleration.radial_channels
    source_value_input = acceleration_stack[[0, -1]]
    source_s_first_input = acceleration_radial.evaluate_s(r, 1)[[0, -1]]
    source_z_jets = compact.z_first_s_jets(
        state_name="acceleration",
        radius=r,
        wall_value_s_jets=(source_value_input, source_s_first_input),
    )
    if not _bitwise_equal(source_z_jets[0], acceleration_endpoint):
        raise Protocol125SharedRepresentationError(
            "acceleration endpoint changed when its s-jet was appended"
        )
    row_source = convert_row_implied_acceleration_z(
        source_value_input,
        source_z_jets[0],
        source_z_jets[1],
        r,
        parent_r_max=float(r[-1]),
    )

    dense_r = frozen_validation_meshes()["dense_wall"]["r"]
    dense_value_input = acceleration_radial.evaluate_s(dense_r, 0)[[0, -1]]
    dense_s_first_input = acceleration_radial.evaluate_s(dense_r, 1)[[0, -1]]
    dense_z_jets = compact.z_first_s_jets(
        state_name="acceleration",
        radius=dense_r,
        wall_value_s_jets=(dense_value_input, dense_s_first_input),
    )
    row_dense = convert_row_implied_acceleration_z(
        dense_value_input,
        dense_z_jets[0],
        dense_z_jets[1],
        dense_r,
        parent_r_max=float(r[-1]),
    )

    direct_endpoint, direct_s_first, direct_inputs = (
        _source_dz7_acceleration_endpoint(acceleration_stack, z, r)
    )
    direct_fd = convert_native_fd_acceleration_z_comparator(
        acceleration_stack[[0, -1]],
        direct_endpoint,
        direct_s_first,
        r,
        parent_r_max=float(r[-1]),
    )

    source_analytic_physical = {}
    source_analytic_reduced = {}
    dense_analytic_physical = {}
    dense_analytic_reduced = {}
    for label, state in (
        ("Q53", final_pair.primary.acceleration),
        ("Q33", final_pair.comparator.acceleration),
    ):
        source_analytic_physical[label] = _scaled_linf(
            state.evaluate_coordinate_components(z[[0, -1]], r, z_order=1),
            row_source["physical"],
        )
        source_analytic_reduced[label] = _scaled_linf(
            state.evaluate_reduced(z[[0, -1]], r, z_order=1),
            row_source["reduced"],
        )
        dense_analytic_physical[label] = _scaled_linf(
            state.evaluate_coordinate_components(
                z[[0, -1]], dense_r, z_order=1,
            ),
            row_dense["physical"],
        )
        dense_analytic_reduced[label] = _scaled_linf(
            state.evaluate_reduced(z[[0, -1]], dense_r, z_order=1),
            row_dense["reduced"],
        )

    bulk_sampler = Protocol125BulkAccelerationSampler.build(z, r, bulk)
    triplet_fingerprint = _fingerprint_arrays({
        name: np.asarray(source_triplet[name]) for name in SOURCE_TRIPLET_KEYS
    })
    checks = MappingProxyType({
        "position_Q53_source_bitwise": source_position_q53_bitwise,
        "position_Q33_source_bitwise": source_position_q33_bitwise,
        "acceleration_Q53_source_scaled_Linf": q53_acceleration_reproduction,
        "acceleration_Q33_source_scaled_Linf": q33_acceleration_reproduction,
        **{
            f"acceleration_{label}_source_endpoint_physical_scaled_Linf": value
            for label, value in source_analytic_physical.items()
        },
        **{
            f"acceleration_{label}_source_endpoint_reduced_scaled_Linf": value
            for label, value in source_analytic_reduced.items()
        },
        **{
            f"acceleration_{label}_dense_endpoint_physical_scaled_Linf": value
            for label, value in dense_analytic_physical.items()
        },
        **{
            f"acceleration_{label}_dense_endpoint_reduced_scaled_Linf": value
            for label, value in dense_analytic_reduced.items()
        },
        "append_only_compact_invariants_bitwise": True,
        "shared_outer_retains_position_contract_object": True,
        "acceleration_endpoint_s_jet_order": 1,
        "direct_comparator_stencil_width": 7,
        "candidate_or_artifact_written": False,
        "execution_authorized": False,
    })
    s_inputs = MappingProxyType({
        "source_r": _immutable(r),
        "source_value": _immutable(source_value_input),
        "source_s_first": _immutable(source_s_first_input),
        "dense_r": _immutable(dense_r),
        "dense_value": _immutable(dense_value_input),
        "dense_s_first": _immutable(dense_s_first_input),
        "dense_z_first": _immutable(dense_z_jets[0]),
        "dense_z_first_s_first": _immutable(dense_z_jets[1]),
    })
    return Protocol125SharedRepresentationBuild(
        final_pair,
        compact,
        outer,
        bulk_sampler,
        position_pair.fingerprint(),
        triplet_fingerprint,
        _immutable(position_endpoint),
        _immutable(acceleration_endpoint),
        _immutable(source_z_jets[1]),
        row_source,
        row_dense,
        direct_fd,
        MappingProxyType({name: _immutable(value) for name, value in direct_inputs.items()}),
        s_inputs,
        checks,
    )
