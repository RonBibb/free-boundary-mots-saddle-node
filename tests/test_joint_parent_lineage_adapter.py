from __future__ import annotations

import numpy as np
import pytest

from bhps.joint_parent_boundary_contracts import (
    NativeNormalizedCompactWallContract,
    derive_protocol125_outer_derivative_bundle,
)
from bhps.joint_parent_bulk_reference import FiniteWallReferenceHermitePair
from bhps.joint_parent_lineage_adapter import (
    POSITION_MESH_DOMAIN_ORDER,
    build_protocol125_append_only_position_lineage,
)
from bhps.joint_parent_position_pair import PositionOnlyConstrainedHermitePair
from bhps.joint_parent_position_state import (
    build_joint_parent_position_state,
    derive_joint_parent_position_outer_contract,
    native_channel_mapping_from_reduced,
)
from bhps.joint_parent_protocol125_sampling_lineage import Protocol125LineageError
from bhps.joint_parent_representation import (
    NATIVE_CHANNEL_ORDER,
    RadialFirstConstrainedHermitePair,
)


def _endpoint_mapping(values):
    return {
        name: np.asarray(values[:, :, index]).copy()
        for index, name in enumerate(NATIVE_CHANNEL_ORDER)
    }


def _flat_fixture(*, interior_position_change=0.0, reference_change=0.0):
    z = np.linspace(1.0, np.e, 9)
    r = np.linspace(0.0, 12.0, 17)
    shape = (len(z), len(r))
    position = np.zeros((*shape, 9))
    position[:, :, 2] = -1.0
    position[:, :, 3] = 1.0
    position[:, :, 6] = 1.0
    selector_q = np.broadcast_to((1.0-z)[:, None], shape).copy()
    zeros = np.zeros(shape)
    reference_q = selector_q.copy()
    reference_phi = zeros.copy()
    parent = {
        "z": z,
        "r": r,
        "position": position,
        "selector_q": selector_q,
        "psi_selector": np.ones(shape),
        "reference_q": reference_q,
        "reference_phi": reference_phi,
        "chi": zeros.copy(),
        "chi_r": zeros.copy(),
        **{
            f"shape_{name}": zeros.copy()
            for name in ("a", "b", "c", "a_r", "b_r", "c_r")
        },
    }
    background = {
        "wall_stiffness": 0.0,
        "v0": 0.0,
        "v1": 0.0,
        "beta_a": 0.0,
        "beta_b": 0.0,
        "wall_potential_a": 0.0,
        "wall_potential_b": 0.0,
    }
    position_outer = derive_joint_parent_position_outer_contract(parent)
    position_state, position_record = build_joint_parent_position_state(
        position,
        z,
        r,
        background,
        outer_open_face_contract=position_outer,
    )
    position_pair = PositionOnlyConstrainedHermitePair.from_primary(
        position_state
    )

    final_position = position.copy()
    if interior_position_change:
        final_position[len(z)//2, len(r)//2, 8] += float(
            interior_position_change
        )
    position_mapping = native_channel_mapping_from_reduced(final_position, r)
    position_stack = np.stack(tuple(position_mapping.values()), axis=-1)
    acceleration = np.zeros_like(position)
    acceleration_mapping = native_channel_mapping_from_reduced(acceleration, r)
    acceleration_stack = np.stack(tuple(acceleration_mapping.values()), axis=-1)
    source_normal = position_record["source_normal_wall"]
    compact = NativeNormalizedCompactWallContract.build(
        r,
        background,
        position_stack[[0, -1]],
        source_normal,
        np.zeros_like(source_normal),
    )
    position_endpoints = compact.z_first_s_jets(
        state_name="position",
        radius=r,
        wall_value_s_jets=(position_stack[[0, -1]],),
    )[0]
    acceleration_endpoints = compact.z_first_s_jets(
        state_name="acceleration",
        radius=r,
        wall_value_s_jets=(acceleration_stack[[0, -1]],),
    )[0]
    shared_outer = derive_protocol125_outer_derivative_bundle(
        position_outer, acceleration_mapping,
    )
    shared_pair = RadialFirstConstrainedHermitePair.build(
        z,
        r,
        position_mapping,
        acceleration_mapping,
        _endpoint_mapping(position_endpoints),
        _endpoint_mapping(acceleration_endpoints),
        compact_wall_contract=compact,
        outer_open_face_contract=shared_outer,
    )
    altered_reference_q = reference_q.copy()
    if reference_change:
        # Deliberately choose a node outside the outer width-seven stencil.
        # The full source/reference fingerprint, not only the outer row, must
        # still bind this array to the position contract.
        altered_reference_q[len(z)//2, len(r)//2] += float(reference_change)
    reference_pair = FiniteWallReferenceHermitePair.build(
        z, r, altered_reference_q, reference_phi,
    )
    return position_pair, shared_pair, reference_pair


def test_concrete_lineage_captures_invariant_payload_children_and_full_mesh():
    position_pair, shared_pair, reference_pair = _flat_fixture()
    record = build_protocol125_append_only_position_lineage(
        position_pair, shared_pair, reference_pair,
    )
    assert record["passed"]
    position = record["position_only_snapshot"]
    shared = record["shared_snapshot"]
    assert position.payload_hash == shared.payload_hash
    assert position.compact_identifier != shared.compact_identifier
    assert position.compact_fingerprint != shared.compact_fingerprint
    assert position.outer_identifier != shared.outer_identifier
    assert position.outer_fingerprint != shared.outer_fingerprint
    assert position.archive_fingerprint != shared.archive_fingerprint
    assert shared.parent_payload_hash == position.payload_hash
    child_names = tuple(name for name, _ in shared.child_entries)
    assert any("source_second_normal" in name for name in child_names)
    assert any("source_acceleration" in name for name in child_names)
    assert any("Q53_acceleration" in name for name in child_names)
    assert any("Q33_acceleration" in name for name in child_names)
    payload_names = tuple(name for name, _ in shared.payload_entries)
    assert "outer/position_contract_identifier" in payload_names
    assert not any(
        name.endswith("compact_wall_contract_id") for name in payload_names
    )

    mesh = record["mesh_identity"]
    assert mesh["passed"]
    assert mesh["domain_order"] == POSITION_MESH_DOMAIN_ORDER
    assert tuple(mesh["domains"]) == POSITION_MESH_DOMAIN_ORDER
    assert mesh["reduced_and_coordinate_values_compared_bitwise"]
    assert not mesh["tolerance_comparison_used"]
    assert all(
        domain["Q53_passed"] and domain["Q33_passed"]
        for domain in mesh["domains"].values()
    )


def test_concrete_lineage_rejects_one_interior_position_coefficient_change():
    position_pair, shared_pair, reference_pair = _flat_fixture(
        interior_position_change=1e-12,
    )
    with pytest.raises(Protocol125LineageError, match="payload"):
        build_protocol125_append_only_position_lineage(
            position_pair, shared_pair, reference_pair,
        )


def test_concrete_lineage_rejects_reference_not_bound_to_outer_contract():
    position_pair, shared_pair, reference_pair = _flat_fixture(
        reference_change=1e-8,
    )
    with pytest.raises(Protocol125LineageError, match="full reference"):
        build_protocol125_append_only_position_lineage(
            position_pair, shared_pair, reference_pair,
        )
