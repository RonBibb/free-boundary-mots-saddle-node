from __future__ import annotations

import numpy as np
import pytest

from bhps.joint_parent_endpoint_audits import (
    ACCELERATION_ENDPOINT_CONVERSION_LANES,
    score_acceleration_endpoint_conversion_pair,
)
from bhps.joint_parent_position_pair import PositionOnlyConstrainedHermitePair
from bhps.joint_parent_position_state import (
    build_joint_parent_position_state,
    derive_joint_parent_position_outer_contract,
)
from bhps.joint_parent_shared_representation import (
    Protocol125SharedRepresentationError,
    build_protocol125_shared_representation,
)


def _flat_inputs():
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
    pair = PositionOnlyConstrainedHermitePair.from_primary(state)
    acceleration = np.zeros_like(position)
    triplet = {
        name: np.zeros((*shape, 3))
        for name in ("source", "source_time", "source_second_time")
    }
    triplet["source"][[0, -1], :, 1] = state_record["source_normal_wall"]
    return parent, background, pair, state_record, acceleration, triplet


def test_shared_builder_is_append_only_and_records_both_endpoint_routes():
    parent, background, pair, record, acceleration, triplet = _flat_inputs()
    found = build_protocol125_shared_representation(
        parent,
        background,
        pair,
        record,
        acceleration,
        acceleration,
        triplet,
    )
    assert found.compact_contract.position_context is (
        pair.primary.compact_wall_contract.position_context
    )
    assert found.compact_contract.source_normal_context is (
        pair.primary.compact_wall_contract.source_normal_context
    )
    assert found.outer_contract.position_contract is (
        pair.primary.outer_open_face_contract
    )
    assert found.compact_contract.identifier != (
        pair.primary.compact_wall_contract.identifier
    )
    assert found.outer_contract.identifier != pair.primary.outer_open_face_contract.identifier
    assert found.checks["position_Q53_source_bitwise"]
    assert found.checks["position_Q33_source_bitwise"]
    assert found.row_implied_source["conversion_kind"] == (
        "row-implied-live-compact-contract"
    )
    assert found.direct_fd_source["conversion_kind"] == (
        "independent-source-Dz7-comparator"
    )
    np.testing.assert_array_equal(found.row_implied_source["physical"], 0.0)
    np.testing.assert_array_equal(found.row_implied_source["reduced"], 0.0)
    np.testing.assert_array_equal(found.direct_fd_source["physical"], 0.0)
    assert found.bulk_sampler.axis_reproduction_scaled_Linf == 0.0
    assert len(found.fingerprint()) == 64
    assert not found.position_endpoint_native.flags.writeable
    assert not found.acceleration_wall_s_jet_inputs["dense_s_first"].flags.writeable
    endpoint = score_acceleration_endpoint_conversion_pair(
        found.final_pair,
        found.row_implied_source,
        found.row_implied_dense,
        found.direct_fd_source,
        found.acceleration_wall_s_jet_inputs["dense_r"],
    )
    assert tuple(endpoint["lanes"]) == ACCELERATION_ENDPOINT_CONVERSION_LANES
    assert endpoint["passed"]
    assert all(lane["passed"] for lane in endpoint["lanes"].values())
    for name in ACCELERATION_ENDPOINT_CONVERSION_LANES[1:]:
        comparisons = endpoint["lanes"][name]["comparisons"]
        assert "source_analytic_vs_Dz7_axis_images" in comparisons
        assert "dense_analytic_vs_row_implied_axis_images" in comparisons


def test_shared_builder_rejects_changed_position_normal_source():
    parent, background, pair, record, acceleration, triplet = _flat_inputs()
    triplet["source"][0, 0, 1] = 1e-30
    with pytest.raises(
        Protocol125SharedRepresentationError,
        match="normal trace differs",
    ):
        build_protocol125_shared_representation(
            parent,
            background,
            pair,
            record,
            acceleration,
            acceleration,
            triplet,
        )


def test_shared_builder_rejects_nonzero_acceleration_gauge_lane():
    parent, background, pair, record, acceleration, triplet = _flat_inputs()
    acceleration[3, 4, 0] = 1e-30
    with pytest.raises(ValueError, match="IEEE positive zero"):
        build_protocol125_shared_representation(
            parent,
            background,
            pair,
            record,
            acceleration,
            np.zeros_like(acceleration),
            triplet,
        )
