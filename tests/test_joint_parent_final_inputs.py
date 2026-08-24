from __future__ import annotations

import numpy as np
import pytest

from bhps.joint_parent_bulk_reference import FiniteWallReferenceHermitePair
from bhps.joint_parent_final_inputs import build_protocol125_final_matrix_inputs
from bhps.joint_parent_final_matrix import (
    evaluate_protocol125_final_representation_matrix,
)
from bhps.joint_parent_position_pair import PositionOnlyConstrainedHermitePair
from bhps.joint_parent_position_state import (
    build_joint_parent_position_state,
    derive_joint_parent_position_outer_contract,
)
from bhps.joint_parent_shared_representation import (
    build_protocol125_shared_representation,
)


def _flat_shared_pair():
    z = np.linspace(1.0, np.e, 9)
    r = np.linspace(0.0, 12.0, 17)
    shape = (len(z), len(r))
    zeros = np.zeros(shape)
    position = np.zeros((*shape, 9))
    position[:, :, 2] = -1.0
    position[:, :, 3] = 1.0
    position[:, :, 6] = 1.0
    selector_q = np.broadcast_to((1.0-z)[:, None], shape).copy()
    parent = {
        "z": z,
        "r": r,
        "position": position,
        "selector_q": selector_q,
        "psi_selector": np.ones(shape),
        "reference_q": selector_q.copy(),
        "reference_phi": zeros.copy(),
        "chi": zeros.copy(),
        "chi_r": zeros.copy(),
        **{
            f"shape_{name}": zeros.copy()
            for name in ("a", "b", "c", "a_r", "b_r", "c_r")
        },
    }
    background = {
        "mass_squared": 0.0,
        "wall_stiffness": 0.0,
        "v0": 0.0,
        "v1": 0.0,
        "beta_a": 0.0,
        "beta_b": 0.0,
        "wall_potential_a": 0.0,
        "wall_potential_b": 0.0,
    }
    outer = derive_joint_parent_position_outer_contract(parent)
    state, state_record = build_joint_parent_position_state(
        position,
        z,
        r,
        background,
        outer_open_face_contract=outer,
        parent_r_max=12.0,
    )
    position_pair = PositionOnlyConstrainedHermitePair.from_primary(state)
    acceleration = np.zeros_like(position)
    triplet = {
        name: np.zeros((*shape, 3))
        for name in ("source", "source_time", "source_second_time")
    }
    triplet["source"][[0, -1], :, 1] = state_record["source_normal_wall"]
    shared = build_protocol125_shared_representation(
        parent,
        background,
        position_pair,
        state_record,
        acceleration,
        acceleration,
        triplet,
    )
    reference = FiniteWallReferenceHermitePair.build(
        z, r, selector_q, zeros,
    )
    return shared, reference, background


def test_final_input_adapter_builds_complete_matrix_inputs_and_passes_flat_case():
    shared, reference, background = _flat_shared_pair()
    bundle = build_protocol125_final_matrix_inputs(
        shared,
        reference,
        background,
        parent_label="N0",
        parent_identity="a"*64,
    )
    assert bundle.parent_label == "N0"
    assert bundle.parent_identity == "a"*64
    assert bundle.adapter_record["velocity_endpoint_z_positive_zero"]
    assert bundle.adapter_record["frozen_V0_V1_V2_only"]
    assert bundle.adapter_record["shared_representation_fingerprint"] == (
        shared.fingerprint()
    )
    assert not bundle.adapter_record["artifact_written"]
    assert len(bundle.fingerprint()) == 64
    for by_mesh in (
        bundle.inputs.Q53_source_triplets_by_mesh,
        bundle.inputs.Q33_source_triplets_by_mesh,
    ):
        assert tuple(by_mesh) == ("V0", "V1", "V2")
        assert all(not value.flags.writeable for mesh in by_mesh.values() for value in mesh.values())
    result = evaluate_protocol125_final_representation_matrix(
        bundle.inputs, bundle.provenance,
    )
    assert result["complete"]
    assert result["provenance_valid"]
    assert result["passed"]


def test_final_input_adapter_rejects_identity_and_reference_grid_drift():
    shared, reference, background = _flat_shared_pair()
    with pytest.raises(ValueError, match="label or identity"):
        build_protocol125_final_matrix_inputs(
            shared,
            reference,
            background,
            parent_label="N2",
            parent_identity="a"*64,
        )
    other_z = np.linspace(1.0, np.e, 11)
    other_r = np.linspace(0.0, 12.0, 17)
    other = FiniteWallReferenceHermitePair.build(
        other_z,
        other_r,
        np.zeros((len(other_z), len(other_r))),
        np.zeros((len(other_z), len(other_r))),
    )
    with pytest.raises(ValueError, match="coordinates differ"):
        build_protocol125_final_matrix_inputs(
            shared,
            other,
            background,
            parent_label="N0",
            parent_identity="a"*64,
        )
