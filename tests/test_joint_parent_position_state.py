from __future__ import annotations

import numpy as np

from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.joint_parent_boundary_contracts import H00
from bhps.joint_parent_position_state import (
    build_joint_parent_position_state,
    derive_joint_parent_position_outer_contract,
    native_channel_mapping_from_reduced,
)


def _flat_case():
    z = np.linspace(1.0, 2.0, 9)
    r = np.linspace(0.0, 2.0, 17)
    q = np.zeros((len(z), len(r), 9))
    q[:, :, 2] = -1.0
    q[:, :, 3] = 1.0
    q[:, :, 6] = 1.0
    background = {
        "wall_stiffness": 0.0,
        "v0": 0.0,
        "v1": 0.0,
        "beta_a": 0.0,
        "beta_b": 0.0,
        "wall_potential_a": 0.0,
        "wall_potential_b": 0.0,
    }
    zeros = np.zeros((len(z), len(r)))
    selector_q = np.broadcast_to((1.0-z)[:, None], zeros.shape).copy()
    parent = {
        "z": z,
        "r": r,
        "position": q,
        "selector_q": selector_q,
        "psi_selector": np.ones_like(zeros),
        "reference_q": selector_q.copy(),
        "reference_phi": zeros.copy(),
        "chi": zeros.copy(),
        "chi_r": zeros.copy(),
        **{
            f"shape_{name}": zeros.copy()
            for name in ("a", "b", "c", "a_r", "b_r", "c_r")
        },
    }
    outer = derive_joint_parent_position_outer_contract(parent)
    return z, r, q, background, outer


def test_position_state_build_is_acceleration_free_and_reproduces_source():
    z, r, q, background, outer = _flat_case()
    state, record = build_joint_parent_position_state(
        q, z, r, background,
        outer_open_face_contract=outer,
        parent_r_max=2.0,
    )
    np.testing.assert_allclose(
        state.evaluate_reduced(z, r), q, rtol=0.0, atol=2e-13,
    )
    assert record["source_reproduction_scaled_Linf"] < 2e-13
    assert not record["acceleration_placeholder_used"]
    assert record["velocity_positive_zero"]
    assert "position-v1" in record["compact_contract_identifier"]


def test_native_mapping_derives_hrr_and_never_accepts_independent_q4():
    z = np.linspace(1.0, 2.0, 7)
    r = np.linspace(0.0, 3.0, 9)
    q = np.zeros((len(z), len(r), 9))
    q[:, :, 3] = 2.0
    q[:, :, 4] = 0.25
    mapping = native_channel_mapping_from_reduced(q, r)
    expected = np.broadcast_to(2.0+0.25*r[None, :]**2, (len(z), len(r)))
    np.testing.assert_allclose(mapping["h_rr"], expected)
    assert "q4" not in mapping


def test_parent_outer_adapter_uses_open_psi_r_and_native_width_seven_corners():
    z = np.linspace(1.0, 2.0, 9)
    r = np.linspace(0.0, 3.0, 17)
    radius = r[None, :]
    selector_q = 0.04+0.012*z[:, None]+0.006*radius
    psi = 1.0/(z[:, None]+selector_q)
    alpha = psi.copy()
    alpha[[0, -1]] *= 1.0+0.025*(radius/float(r[-1]))**2
    position = np.zeros((len(z), len(r), 9))
    position[:, :, 2] = -alpha**2
    position[:, :, 3] = psi**2
    position[:, :, 6] = psi**2
    zeros = np.zeros_like(selector_q)
    parent = {
        "z": z,
        "r": r,
        "position": position,
        "selector_q": selector_q,
        "psi_selector": psi,
        "reference_q": selector_q.copy(),
        "reference_phi": zeros.copy(),
        "chi": zeros.copy(),
        "chi_r": zeros.copy(),
        **{
            f"shape_{name}": zeros.copy()
            for name in ("a", "b", "c", "a_r", "b_r", "c_r")
        },
    }
    contract = derive_joint_parent_position_outer_contract(parent)
    record = contract.coefficient_arrays()
    primitive_names = [str(name) for name in record["primitive_keys"]]
    alpha_index = primitive_names.index("alpha")
    alpha_r_index = primitive_names.index("alpha_r")
    np.testing.assert_array_equal(
        record["primitive_values"][:, alpha_index], alpha[:, -1],
    )
    radial_operator = derivative_matrix(r, 1, 7)
    native_alpha_r = (radial_operator @ alpha.T).T[:, -1]
    reference_q_r = (radial_operator @ selector_q.T).T[:, -1]
    psi_r = -psi[:, -1]**2*reference_q_r
    stored_alpha_r = record["primitive_values"][:, alpha_r_index]
    np.testing.assert_array_equal(stored_alpha_r[1:-1], psi_r[1:-1])
    np.testing.assert_array_equal(
        stored_alpha_r[[0, -1]], native_alpha_r[[0, -1]],
    )
    np.testing.assert_array_equal(
        contract.position_r_first[1:-1, H00],
        -2.0*alpha[1:-1, -1]*psi_r[1:-1],
    )
    np.testing.assert_array_equal(
        contract.position_r_first[[0, -1], H00],
        -2.0*alpha[[0, -1], -1]*native_alpha_r[[0, -1]],
    )
