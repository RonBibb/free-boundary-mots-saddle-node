from __future__ import annotations

import numpy as np
import pytest

from bhps.joint_parent_endpoint_audits import (
    DENSE_WALL_PROFILE_DERIVATIVE_RECIPE,
    SOURCE_WALL_PROFILE_DERIVATIVE_RECIPE,
    WALL_PROFILE_EVIDENCE_PROTOCOL_IDENTIFIER,
    _endpoint_conversion_arrays,
    build_protocol125_wall_profile_evidence,
    convert_native_fd_acceleration_z_comparator,
    convert_row_implied_acceleration_z,
    score_normalized_wall_profiles,
    score_time_symmetric_velocity_endpoint_z,
    wall_profile_evidence_fingerprint,
)
from bhps.joint_parent_position_pair import PositionOnlyConstrainedHermitePair
from bhps.joint_parent_position_state import (
    build_joint_parent_position_state,
    derive_joint_parent_position_outer_contract,
)
from bhps.joint_parent_representation import NATIVE_CHANNEL_ORDER
from bhps.joint_parent_shared_representation import (
    build_protocol125_shared_representation,
)
from bhps.joint_parent_refinement_diagnostics import DENSE_WALL_SHA256
from bhps.matched_staged_continuum import hash_arrays


def _native_index(name):
    return NATIVE_CHANNEL_ORDER.index(name)


def test_row_implied_acceleration_z_conversion_uses_analytic_axis_limits():
    r = np.linspace(0.0, 12.0, 9)
    native = np.zeros((2, len(r), 8))
    z_first = np.zeros_like(native)
    s_first = np.zeros_like(native)
    q4 = np.asarray((0.25, -0.4))[:, None]
    q5 = np.asarray((0.7, -0.2))[:, None]
    z_first[:, :, _native_index("h_perp")] = 0.1
    z_first[:, :, _native_index("h_rr")] = 0.1+q4*r[None, :]**2
    z_first[:, :, _native_index("v_0")] = q5
    s_first[:, 0, _native_index("h_rr")] = q4[:, 0]*12.0**2

    found = convert_row_implied_acceleration_z(native, z_first, s_first, r)
    np.testing.assert_allclose(
        found["reduced"][:, :, 4], np.broadcast_to(q4, (2, len(r))),
    )
    np.testing.assert_allclose(
        found["reduced"][:, :, 5], np.broadcast_to(q5, (2, len(r))),
    )
    np.testing.assert_allclose(found["physical"][:, :, 5], q5*r[None, :])
    np.testing.assert_allclose(found["axis_images"][:, 0], 2.0*q4[:, 0])
    np.testing.assert_allclose(found["axis_images"][:, 1], q5[:, 0])
    np.testing.assert_allclose(found["q4_axis_source"], q4[:, 0])
    np.testing.assert_allclose(found["q5_axis_source"], q5[:, 0])
    np.testing.assert_array_equal(found["ownership_mask"], True)
    np.testing.assert_array_equal(
        found["native_acceleration_z_s_first"], s_first,
    )
    assert found["physical_sha256"] == hash_arrays(found["physical"])
    assert found["reduced_sha256"] == hash_arrays(found["reduced"])
    assert len(found["fingerprint"]) == 64
    assert not found["physical"].flags.writeable


def test_endpoint_conversion_rejects_axis_source_not_reassembled_from_reduced():
    r = np.linspace(0.0, 12.0, 9)
    native = np.zeros((2, len(r), 8))
    z_first = np.zeros_like(native)
    s_first = np.zeros_like(native)
    found = convert_row_implied_acceleration_z(native, z_first, s_first, r)
    tampered = dict(found)
    tampered["q4_axis_source"] = np.asarray((1e-15, 0.0))
    with pytest.raises(ValueError, match="sources do not reassemble"):
        _endpoint_conversion_arrays(
            tampered,
            r,
            "tampered",
            "row-implied-live-compact-contract",
        )


def test_endpoint_conversion_rejects_ownership_s_jet_and_constituent_hash_tamper():
    r = np.linspace(0.0, 12.0, 9)
    native = np.zeros((2, len(r), 8))
    z_first = np.zeros_like(native)
    s_first = np.zeros_like(native)
    found = convert_row_implied_acceleration_z(native, z_first, s_first, r)

    ownership = dict(found)
    ownership["ownership_mask"] = np.asarray(
        (False,)+tuple(found["ownership_mask"])[1:], dtype=bool,
    )
    ownership["ownership_mask_sha256"] = hash_arrays(
        ownership["ownership_mask"],
    )
    with pytest.raises(ValueError, match="ownership or radius differs"):
        _endpoint_conversion_arrays(
            ownership, r, "ownership", "row-implied-live-compact-contract",
        )

    s_jet = dict(found)
    changed = np.asarray(found["native_acceleration_z_s_first"]).copy()
    changed[0, 0, _native_index("h_rr")] = 1e-12
    s_jet["native_acceleration_z_s_first"] = changed
    s_jet["s_jet_inputs_sha256"] = hash_arrays(changed)
    with pytest.raises(ValueError, match="sources do not reassemble"):
        _endpoint_conversion_arrays(
            s_jet, r, "s-jet", "row-implied-live-compact-contract",
        )

    constituent = dict(found)
    constituent["physical_sha256"] = "0"*64
    with pytest.raises(ValueError, match="constituent hash differs"):
        _endpoint_conversion_arrays(
            constituent, r, "hash", "row-implied-live-compact-contract",
        )


def test_row_implied_acceleration_z_rejects_hidden_nonregular_axis_trace():
    r = np.linspace(0.0, 12.0, 9)
    native = np.zeros((2, len(r), 8))
    z_first = np.zeros_like(native)
    s_first = np.zeros_like(native)
    z_first[0, 0, _native_index("h_rr")] = 1e-30
    with pytest.raises(ValueError, match="positive zero"):
        convert_row_implied_acceleration_z(native, z_first, s_first, r)


def test_fd_comparator_uses_same_conversion_without_claiming_wall_ownership():
    r = np.linspace(0.0, 12.0, 9)
    native = np.zeros((2, len(r), 8))
    z_first = np.zeros_like(native)
    s_first = np.zeros_like(native)
    z_first[:, 0, _native_index("v_z")] = (0.2, -0.3)
    found = convert_native_fd_acceleration_z_comparator(
        native, z_first, s_first, r,
    )
    np.testing.assert_array_equal(
        found["reduced"][:, 0, 1], np.asarray((0.2, -0.3)),
    )
    assert found["conversion_kind"] == "independent-source-Dz7-comparator"
    assert not found["normal_tangential_positive_zero_required"]
    with pytest.raises(ValueError, match="q1 axis trace"):
        convert_row_implied_acceleration_z(native, z_first, s_first, r)


def _closed_wall_profiles(nr=11):
    phi = np.stack((
        np.linspace(0.1, 0.2, nr),
        np.linspace(-0.2, -0.1, nr),
    ))
    gamma = 2.0
    targets = np.zeros((2, 1))
    delta = phi-targets
    sigma = np.asarray((-1.0, 1.0))[:, None]
    G = np.ones_like(phi)
    potential = 0.5*gamma*delta**2
    branch = np.asarray((1.0, -1.0))[:, None]
    beta = branch*potential/6.0
    position = {
        "Phi": phi,
        "Phi_z": -sigma*(gamma/2.0)*delta,
        "chi_z": np.zeros_like(phi),
        "G": G,
        "G_z": np.zeros_like(phi),
        "H_z": 4.0*beta,
    }
    acceleration = {
        name: np.zeros_like(phi)
        for name in ("a_Phi", "a_Phi_z", "a_chi_z", "a_G", "a_G_z", "H_ztt")
    }
    background = {
        "wall_stiffness": gamma,
        "v0": 0.0,
        "v1": 0.0,
        "beta_a": 0.0,
        "beta_b": 0.0,
        "wall_potential_a": 0.0,
        "wall_potential_b": 0.0,
    }
    return position, acceleration, background


def test_termwise_wall_profile_scales_close_each_named_row_and_wall():
    position, acceleration, background = _closed_wall_profiles()
    found = score_normalized_wall_profiles(position, acceleration, background)
    assert found["passed"]
    assert len(found["gates"]) == 12
    for stage in ("position", "acceleration"):
        for row in ("Phi", "chi", "normal_GH"):
            record = found["records"][stage][row]
            np.testing.assert_array_equal(record["signed_residual"], 0.0)
            assert np.all(record["positive_scale"] >= 1.0)
            assert record["wall_pass"] == (True, True)


def test_each_wall_profile_is_gated_independently_without_pooled_credit():
    position, acceleration, background = _closed_wall_profiles()
    acceleration["a_chi_z"][1, 3] = 2e-10
    found = score_normalized_wall_profiles(position, acceleration, background)
    assert not found["passed"]
    assert found["gates"]["acceleration_chi_lower"]
    assert not found["gates"]["acceleration_chi_upper"]
    assert found["records"]["acceleration"]["chi"]["wall_Linf"][1] == 2e-10


def test_phi_acceleration_scale_is_sum_of_term_magnitudes_not_one_plus_terms():
    position, acceleration, background = _closed_wall_profiles(nr=3)
    acceleration["a_Phi_z"][:] = 2.0
    acceleration["a_Phi"][:] = 3.0
    acceleration["a_G"][:] = 4.0
    found = score_normalized_wall_profiles(position, acceleration, background)
    terms = found["records"]["acceleration"]["Phi"]["signed_terms"]
    expected = np.maximum(1.0, np.sum(np.abs(terms), axis=0))
    np.testing.assert_allclose(
        found["records"]["acceleration"]["Phi"]["positive_scale"], expected,
    )


def test_time_symmetric_velocity_endpoint_z_requires_exact_positive_zero():
    velocity_z = np.zeros((2, 13, 8))
    found = score_time_symmetric_velocity_endpoint_z(velocity_z)
    assert found["passed"]
    assert found["bitwise_positive_zero"]
    assert len(found["fingerprint"]) == 64

    negative_zero = velocity_z.copy()
    negative_zero[1, 4, 3] = -0.0
    failed_sign = score_time_symmetric_velocity_endpoint_z(negative_zero)
    assert not failed_sign["passed"]
    assert not failed_sign["bitwise_positive_zero"]

    nonzero = velocity_z.copy()
    nonzero[0, 0, 0] = np.nextafter(0.0, 1.0)
    failed_value = score_time_symmetric_velocity_endpoint_z(nonzero)
    assert not failed_value["passed"]


def test_time_symmetric_velocity_endpoint_z_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        score_time_symmetric_velocity_endpoint_z(np.zeros((2, 13, 9)))


def _flat_wall_evidence_inputs(*, source_chi_acceleration=False):
    z = np.linspace(1.0, np.e, 9)
    r = np.linspace(0.0, 12.0, 17)
    shape = (len(z), len(r))
    zeros = np.zeros(shape)
    position = np.zeros((*shape, 9))
    position[:, :, 2] = -1.0
    position[:, :, 3] = 1.0
    position[:, :, 6] = 1.0
    selector_q = np.broadcast_to((1.0-z)[:, None], shape).copy()
    background = {
        "wall_stiffness": 0.0,
        "v0": 0.0,
        "v1": 0.0,
        "beta_a": 0.0,
        "beta_b": 0.0,
        "wall_potential_a": 0.0,
        "wall_potential_b": 0.0,
    }
    parent = {
        "label": "N0",
        "z": z,
        "r": r,
        "position": position,
        "selector_q": selector_q,
        "psi_selector": np.ones(shape),
        "phi": zeros.copy(),
        "reference_q": selector_q.copy(),
        "reference_phi": zeros.copy(),
        "chi": zeros.copy(),
        "chi_r": zeros.copy(),
        "background": background,
        **{
            f"shape_{name}": zeros.copy()
            for name in ("a", "b", "c", "a_r", "b_r", "c_r")
        },
    }
    parent["parent_identity"] = hash_arrays(
        np.asarray(parent["label"]), z, r, position, selector_q,
        parent["phi"], parent["reference_q"], parent["reference_phi"],
    )
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
    if source_chi_acceleration:
        acceleration[:, :, 8] = (
            (z-z[0])*(z[-1]-z)
        )[:, None]
    source_triplet = {
        name: np.zeros((*shape, 3))
        for name in ("source", "source_time", "source_second_time")
    }
    source_triplet["source"][[0, -1], :, 1] = state_record[
        "source_normal_wall"
    ]
    shared = build_protocol125_shared_representation(
        parent,
        background,
        position_pair,
        state_record,
        acceleration,
        np.zeros_like(acceleration),
        source_triplet,
    )
    return parent, acceleration, source_triplet, shared.final_pair


def test_two_mesh_wall_profile_evidence_binds_native_and_live_analytic_routes():
    parent, acceleration, source_triplet, final_pair = (
        _flat_wall_evidence_inputs()
    )
    velocity = np.zeros_like(parent["position"])
    found = build_protocol125_wall_profile_evidence(
        parent, velocity, acceleration, source_triplet, final_pair,
    )
    assert found["protocol_identifier"] == WALL_PROFILE_EVIDENCE_PROTOCOL_IDENTIFIER
    assert found["parent_label"] == "N0"
    assert found["parent_identity"] == parent["parent_identity"]
    assert found["derivative_recipes"]["source_recipe"] == (
        SOURCE_WALL_PROFILE_DERIVATIVE_RECIPE
    )
    assert found["derivative_recipes"]["dense_recipe"] == (
        DENSE_WALL_PROFILE_DERIVATIVE_RECIPE
    )
    assert found["coordinates"]["dense_r_sha256"] == DENSE_WALL_SHA256
    assert found["time_symmetry"]["source_bitwise_positive_zero"]
    assert found["time_symmetry"]["dense_bitwise_positive_zero"]
    assert tuple(found["meshes"]) == ("source", "dense")
    assert len(found["named_row_wall_gates"]) == 24
    assert all(found["named_row_wall_gates"].values())
    assert found["constituent_logical_AND"]
    assert found["passed"]
    assert found["fingerprint"] == wall_profile_evidence_fingerprint(found)
    assert not found["coordinates"]["dense_r"].flags.writeable


def test_source_native_Dz_failure_cannot_receive_credit_from_dense_contract():
    parent, acceleration, source_triplet, final_pair = (
        _flat_wall_evidence_inputs(source_chi_acceleration=True)
    )
    found = build_protocol125_wall_profile_evidence(
        parent,
        np.zeros_like(parent["position"]),
        acceleration,
        source_triplet,
        final_pair,
    )
    assert not found["meshes"]["source"]["passed"]
    assert found["meshes"]["dense"]["passed"]
    assert not found["named_row_wall_gates"]["source_acceleration_chi_lower"]
    assert not found["named_row_wall_gates"]["source_acceleration_chi_upper"]
    assert found["named_row_wall_gates"]["dense_acceleration_chi_lower"]
    assert found["named_row_wall_gates"]["dense_acceleration_chi_upper"]
    assert not found["named_row_wall_passed"]
    assert not found["passed"]


def test_wall_profile_evidence_records_negative_zero_time_symmetry_failure():
    parent, acceleration, source_triplet, final_pair = (
        _flat_wall_evidence_inputs()
    )
    velocity = np.zeros_like(parent["position"])
    velocity[2, 3, 4] = -0.0
    found = build_protocol125_wall_profile_evidence(
        parent, velocity, acceleration, source_triplet, final_pair,
    )
    assert found["meshes"]["source"]["passed"]
    assert found["meshes"]["dense"]["passed"]
    assert not found["time_symmetry"]["source_bitwise_positive_zero"]
    assert not found["time_symmetry"]["passed"]
    assert not found["passed"]


def test_wall_profile_evidence_rejects_unreproduced_parent_identity():
    parent, acceleration, source_triplet, final_pair = (
        _flat_wall_evidence_inputs()
    )
    parent["parent_identity"] = "f"*64
    with pytest.raises(ValueError, match="identity does not reproduce"):
        build_protocol125_wall_profile_evidence(
            parent,
            np.zeros_like(parent["position"]),
            acceleration,
            source_triplet,
            final_pair,
        )
