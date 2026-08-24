from __future__ import annotations

import numpy as np
import pytest

import bhps.joint_parent_boundary_contracts as boundary_contracts
from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.joint_parent_boundary_contracts import (
    CHI,
    H00,
    H_PERP,
    H_RR,
    H_ZZ,
    PHI,
    Protocol125OuterOpenFaceDerivativeContract,
    Protocol125PositionOuterOpenFaceDerivativeContract,
    _Protocol125PositionOuterOpenFaceDerivativeContract,
    V_0,
    V_Z,
    NativeNormalizedCompactWallContract,
    StoredOuterOpenFaceDerivativeContract,
    derive_protocol125_outer_derivative_bundle,
    _derive_protocol125_position_outer_contract_from_primitives,
)
from bhps.joint_parent_representation import (
    NATIVE_CHANNEL_ORDER,
    RadialFirstConstrainedHermitePair,
)


def _context():
    r = np.linspace(0.0, 12.0, 17)
    s = (r/12.0)**2
    wall = np.arange(2)[:, None]
    values = np.zeros((2, len(r), len(NATIVE_CHANNEL_ORDER)))
    values[:, :, H00] = -1.2-0.02*wall+0.03*s
    values[:, :, H_PERP] = 0.9+0.01*wall+0.02*s+0.01*s**2
    values[:, :, H_RR] = 0.95+0.015*wall+0.025*s+0.005*s**2
    values[:, :, H_ZZ] = 1.1+0.02*wall+0.04*s+0.01*s**2
    values[:, :, PHI] = 0.08-0.03*wall+0.02*s
    values[:, :, CHI] = 0.01+0.005*s
    values[:, :, V_Z] = 0.0
    values[:, :, V_0] = 0.015+0.003*s
    source = 0.07+0.01*wall+0.02*s
    source_second = -0.03+0.005*wall+0.004*s
    background = {
        "wall_stiffness": 3.0,
        "v0": -0.1,
        "v1": 0.12,
        "beta_a": 0.25,
        "beta_b": -0.2,
        "wall_potential_a": 0.01,
        "wall_potential_b": 0.015,
    }
    contract = NativeNormalizedCompactWallContract.build(
        r, background, values, source, source_second,
    )
    return r, values, source, source_second, background, contract


def test_native_position_contract_closes_every_owned_wall_row():
    r, _, _, _, background, contract = _context()
    query = np.linspace(0.0, 12.0, 31)
    values = contract.position_context.jets(query, 3)
    source = tuple(item[:, :, 0] for item in contract.source_normal_context.jets(
        query, 3,
    ))
    found = contract.z_first_s_jets(
        state_name="position",
        radius=query,
        wall_value_s_jets=values,
    )
    base = values[0]
    derivative = found[0]
    gamma = background["wall_stiffness"]
    targets = np.asarray((background["v0"], background["v1"]))[:, None]
    branch = np.asarray((1.0, -1.0))[:, None]
    wall_potential = np.asarray((
        background["wall_potential_a"], background["wall_potential_b"],
    ))[:, None]
    beta0 = np.asarray((background["beta_a"], background["beta_b"]))[:, None]
    delta = base[:, :, PHI]-targets
    beta = beta0+branch*(0.5*gamma*delta**2-wall_potential)/6.0
    A = np.sqrt(base[:, :, H_ZZ])
    for channel in (H00, H_PERP, H_RR, V_0):
        residual = derivative[:, :, channel]+2.0*beta*A*base[:, :, channel]
        assert np.max(np.abs(residual)) < 2e-14
    phi_residual = derivative[:, :, PHI] + (
        np.asarray((-1.0, 1.0))[:, None]
        * 0.5*gamma*delta*A
    )
    normal_residual = (
        derivative[:, :, H_ZZ]
        + 8.0*beta*base[:, :, H_ZZ]**1.5
        - 2.0*source[0]*base[:, :, H_ZZ]
    )
    assert np.max(np.abs(phi_residual)) < 2e-14
    assert np.max(np.abs(normal_residual)) < 2e-14
    assert np.array_equal(derivative[:, :, CHI], np.zeros_like(delta))
    assert np.array_equal(derivative[:, :, V_Z], np.zeros_like(delta))

    # The returned higher jets are analytic derivatives of the same nonlinear
    # row, not independently interpolated endpoint arrays.
    step = 2e-5
    s0 = 0.37
    radii = 12.0*np.sqrt(np.asarray([s0-step, s0, s0+step]))
    order0 = []
    for radius in radii:
        local_values = contract.position_context.jets([radius], 1)
        order0.append(contract.z_first_s_jets(
            state_name="position", radius=[radius],
            wall_value_s_jets=local_values,
        )[0][:, 0])
    numerical_first = (order0[2]-order0[0])/(2.0*step)
    center_values = contract.position_context.jets([radii[1]], 2)
    analytic_first = contract.z_first_s_jets(
        state_name="position", radius=[radii[1]],
        wall_value_s_jets=center_values,
    )[1][:, 0]
    assert np.max(np.abs(numerical_first-analytic_first)) < 2e-8


def test_position_only_compact_contract_has_no_acceleration_placeholder():
    r, values, source, _, background, full = _context()
    position_only = NativeNormalizedCompactWallContract.build_position(
        r, background, values, source,
    )
    expected = full.z_first_s_jets(
        state_name="position", radius=r,
        wall_value_s_jets=(values,),
    )[0]
    found = position_only.z_first_s_jets(
        state_name="position", radius=r,
        wall_value_s_jets=(values,),
    )[0]
    np.testing.assert_array_equal(found, expected)
    record = position_only.coefficient_arrays()
    assert bool(record["position_only"])
    np.testing.assert_array_equal(record["position_ownership_mask"], True)
    with pytest.raises(ValueError, match="no acceleration ownership"):
        _ = position_only.acceleration_ownership_mask
    assert not any(name.startswith("source_second") for name in record)
    with pytest.raises(ValueError, match="no acceleration data"):
        position_only.z_first_s_jets(
            state_name="acceleration", radius=r,
            wall_value_s_jets=(np.zeros_like(values),),
        )


def test_position_only_compact_contract_appends_only_acceleration_child():
    r, values, source, source_second, background, _ = _context()
    position_only = NativeNormalizedCompactWallContract.build_position(
        r, background, values, source,
    )
    before = position_only.coefficient_arrays()
    shared = position_only.append_acceleration(r, source_second)
    after = shared.coefficient_arrays()
    invariant = (
        "background_values",
        "position_knots",
        "position_coefficients",
        "position_parent_r_max",
        "source_normal_knots",
        "source_normal_coefficients",
        "source_normal_parent_r_max",
        "position_ownership_mask",
    )
    for name in invariant:
        np.testing.assert_array_equal(after[name], before[name])
        assert after[name].tobytes() == before[name].tobytes()
    assert bool(before["position_only"])
    assert not bool(after["position_only"])
    np.testing.assert_array_equal(after["position_ownership_mask"], True)
    np.testing.assert_array_equal(after["acceleration_ownership_mask"], True)
    assert shared.position_context is position_only.position_context
    assert shared.source_normal_context is position_only.source_normal_context
    assert shared.identifier != position_only.identifier
    with pytest.raises(ValueError, match="already contains acceleration"):
        shared.append_acceleration(r, source_second)


def test_native_acceleration_contract_is_normalized_and_single_owner_compatible():
    r, _, _, _, background, contract = _context()
    query = np.linspace(0.0, 12.0, 23)
    position = contract.position_context.jets(query, 1)[0]
    acceleration = np.zeros_like(position)
    acceleration[:, :, H00] = 0.04
    acceleration[:, :, H_PERP] = -0.02
    acceleration[:, :, H_RR] = 0.03
    acceleration[:, :, H_ZZ] = 0.07
    acceleration[:, :, PHI] = -0.05
    acceleration[:, :, CHI] = 0.01
    acceleration[:, :, V_0] = 0.006
    found = contract.z_first_s_jets(
        state_name="acceleration",
        radius=query,
        wall_value_s_jets=(acceleration,),
    )[0]
    position_z = contract.z_first_s_jets(
        state_name="position",
        radius=query,
        wall_value_s_jets=(position,),
    )[0]
    gamma = background["wall_stiffness"]
    targets = np.asarray((background["v0"], background["v1"]))[:, None]
    branch = np.asarray((1.0, -1.0))[:, None]
    beta0 = np.asarray((background["beta_a"], background["beta_b"]))[:, None]
    wall_potential = np.asarray((
        background["wall_potential_a"], background["wall_potential_b"],
    ))[:, None]
    delta = position[:, :, PHI]-targets
    beta = beta0+branch*(0.5*gamma*delta**2-wall_potential)/6.0
    beta_phi = branch*gamma*delta/6.0
    A = np.sqrt(position[:, :, H_ZZ])
    for channel in (H00, H_PERP, H_RR, V_0):
        normalized_row = (
            found[:, :, channel]/(2.0*A)
            - position_z[:, :, channel]*acceleration[:, :, H_ZZ]/(4.0*A**3)
            + beta_phi*acceleration[:, :, PHI]*position[:, :, channel]
            + beta*acceleration[:, :, channel]
        )
        assert np.max(np.abs(normalized_row)) < 2e-14
    assert np.array_equal(found[:, :, CHI], np.zeros(found.shape[:2]))
    assert np.array_equal(found[:, :, V_Z], np.zeros(found.shape[:2]))


def test_stored_outer_contract_reproduces_source_bundle_and_derivatives():
    z = np.linspace(1.0, 2.0, 11)
    lanes = np.arange(len(NATIVE_CHANNEL_ORDER))[None, :]
    position = (0.2+0.03*z[:, None]**4)*(1.0+0.01*lanes)
    acceleration = (-0.1+0.02*z[:, None]**3)*(1.0+0.02*lanes)
    mask = np.zeros(len(NATIVE_CHANNEL_ORDER), dtype=bool)
    mask[[H_PERP, H_RR, PHI]] = True
    contract = StoredOuterOpenFaceDerivativeContract.build(
        z,
        position,
        acceleration,
        position_ownership=mask,
        acceleration_ownership=np.ones_like(mask),
    )
    found = contract.r_first_z_jets(
        state_name="position",
        compact_coordinate=z,
        outer_value_z_jets=(np.zeros_like(position),),
    )
    np.testing.assert_allclose(found.r_first_z_jets[0], position, atol=2e-15)
    np.testing.assert_array_equal(found.ownership_mask, mask)
    dense = np.linspace(1.0, 2.0, 17)
    jets = contract.r_first_z_jets(
        state_name="acceleration",
        compact_coordinate=dense,
        outer_value_z_jets=(
            np.zeros((len(dense), len(NATIVE_CHANNEL_ORDER))),
            np.zeros((len(dense), len(NATIVE_CHANNEL_ORDER))),
        ),
    )
    expected_first = 0.06*dense[:, None]**2*(1.0+0.02*lanes)
    assert np.max(np.abs(jets.r_first_z_jets[1]-expected_first)) < 3e-13


def _protocol125_outer_context():
    z = np.linspace(1.0, 2.0, 11)
    radius = 12.0
    source_r = np.linspace(0.0, radius, 17)
    selector_q = 0.08+0.01*z**2
    q_r = -0.006*(1.0+0.2*z)
    reference_q_r = 0.0025*(1.0+0.1*z)
    reference_q_outer = selector_q+radius*(q_r-reference_q_r)
    phi = 0.04-0.015*z+0.003*z**3
    phi_r = 0.005*(1.0+0.1*z)
    reference_phi_r = -0.0017*(1.0+0.05*z)
    reference_phi_outer = phi+radius*(phi_r-reference_phi_r)
    reference_q = (
        reference_q_outer[:, None]
        + reference_q_r[:, None]*(source_r[None, :]-radius)
    )
    reference_phi = (
        reference_phi_outer[:, None]
        + reference_phi_r[:, None]*(source_r[None, :]-radius)
    )
    psi = 1.0/(z+selector_q)
    alpha = 1.03*psi
    alpha_r = -0.007*(1.0+0.05*z)
    profile = 0.004*z**2
    profile_r = 0.0015*(1.0+z)
    a = 3.0*profile
    b = -profile
    c = -profile
    a_r = 3.0*profile_r
    b_r = -profile_r
    c_r = -profile_r
    chi = 0.02*np.cos(0.4*z)
    chi_r = -0.003*(1.0+0.2*z)
    fields = {
        "h00": -alpha**2,
        "h_perp": psi**2*np.exp(2.0*c),
        "h_rr": psi**2*np.exp(2.0*b),
        "h_zz": psi**2*np.exp(2.0*a),
        "Phi": phi.copy(),
        "chi": chi.copy(),
        "v_z": np.zeros_like(z),
        "v_0": np.zeros_like(z),
    }
    # The compact wall owns these two rows, so completion may alter them
    # without making them outer-face inputs.
    fields["h00"][[0, -1]] *= 1.01
    fields["h_rr"][[0, -1]] *= 1.02
    fields["chi"][[0, -1]] += np.asarray([0.01, -0.01])
    primitives = {
        "selector_q": selector_q,
        "psi": psi,
        "alpha": alpha,
        "alpha_r": alpha_r,
    }
    shape = {
        "a": a,
        "b": b,
        "c": c,
        "a_r": a_r,
        "b_r": b_r,
        "c_r": c_r,
    }
    scalar = {"chi": chi, "chi_r": chi_r}
    expected = np.zeros((len(z), len(NATIVE_CHANNEL_ORDER)))
    psi_r = -psi**2*q_r
    expected[:, H00] = -2.0*alpha*alpha_r
    expected[:, H_PERP] = 2.0*fields["h_perp"]*(psi_r/psi+c_r)
    expected[:, H_RR] = 2.0*fields["h_rr"]*(psi_r/psi+b_r)
    expected[:, H_ZZ] = 2.0*fields["h_zz"]*(psi_r/psi+a_r)
    expected[:, PHI] = phi_r
    expected[:, CHI] = chi_r
    return (
        z,
        source_r,
        fields,
        primitives,
        reference_q,
        reference_phi,
        shape,
        scalar,
        expected,
    )


def test_protocol125_outer_position_bundle_is_derived_without_acceleration():
    (
        z,
        source_r,
        fields,
        primitives,
        reference_q,
        reference_phi,
        shape,
        scalar,
        expected,
    ) = (
        _protocol125_outer_context()
    )
    contract = _derive_protocol125_position_outer_contract_from_primitives(
        z,
        source_r,
        fields,
        completed_primitives=primitives,
        reference_q=reference_q,
        reference_phi=reference_phi,
        shape_map=shape,
        scalar_map=scalar,
    )
    assert isinstance(
        contract, _Protocol125PositionOuterOpenFaceDerivativeContract,
    )
    assert isinstance(
        contract, Protocol125PositionOuterOpenFaceDerivativeContract,
    )
    np.testing.assert_array_equal(
        contract.open_compact_mask,
        np.r_[False, np.ones(len(z)-2, dtype=bool), False],
    )
    np.testing.assert_array_equal(
        contract.ownership_mask,
        np.ones(len(NATIVE_CHANNEL_ORDER), dtype=bool),
    )
    np.testing.assert_allclose(
        contract.position_r_first[1:-1], expected[1:-1], atol=2e-15,
    )
    assert np.array_equal(
        contract.position_r_first[:, [V_Z, V_0]],
        np.zeros((len(z), 2)),
    )
    record = contract.coefficient_arrays()
    assert not any("acceleration" in name.lower() for name in record)
    assert record["derivation_recipe"] == (
        "delta-q/Phi-Robin+reference-r-first+"
        "completed-lapse+analytic-shape/scalar-v3"
    )
    assert record["reference_derivation_recipe"] == (
        "full-fresh-reference-width-seven-outer-row-v1"
    )
    assert np.array_equal(record["source_r"], source_r)
    assert len(str(record["source_reference_fingerprint"])) == 64
    primitive_names = [str(value) for value in record["primitive_keys"]]
    assert "reference_q_r" not in primitive_names
    assert "reference_phi_r" not in primitive_names
    reference_names = [
        str(value) for value in record["reference_outer_keys"]
    ]
    q_reference_r = reference_names.index("reference_q_r")
    phi_reference_r = reference_names.index("reference_phi_r")
    assert np.max(np.abs(
        record["reference_outer_values"][:, q_reference_r]
    )) > 0.0
    assert np.max(np.abs(
        record["reference_outer_values"][:, phi_reference_r]
    )) > 0.0
    radial_operator = derivative_matrix(source_r, 1, 7)
    np.testing.assert_allclose(
        record["reference_outer_values"][:, q_reference_r],
        (radial_operator @ reference_q.T).T[:, -1],
        atol=2e-15,
    )
    np.testing.assert_allclose(
        record["reference_outer_values"][:, phi_reference_r],
        (radial_operator @ reference_phi.T).T[:, -1],
        atol=2e-15,
    )
    found = contract.r_first_z_jets(
        state_name="position",
        compact_coordinate=z,
        outer_value_z_jets=(
            np.stack([fields[name] for name in NATIVE_CHANNEL_ORDER], axis=-1),
        ),
    )
    np.testing.assert_allclose(
        found.r_first_z_jets[0], contract.position_r_first, atol=2e-15,
    )
    with np.testing.assert_raises_regex(ValueError, "no acceleration bundle"):
        contract.r_first_z_jets(
            state_name="acceleration",
            compact_coordinate=z,
            outer_value_z_jets=(np.zeros_like(expected),),
        )

    repeated = _derive_protocol125_position_outer_contract_from_primitives(
        z,
        source_r,
        fields,
        completed_primitives=primitives,
        reference_q=reference_q,
        reference_phi=reference_phi,
        shape_map=shape,
        scalar_map=scalar,
    )
    assert repeated.identifier == contract.identifier
    assert not contract.position_r_first.flags.writeable
    changed_reference_q = reference_q.copy()
    changed_reference_q[:, 0] += 1e-9
    changed_reference = _derive_protocol125_position_outer_contract_from_primitives(
        z,
        source_r,
        fields,
        completed_primitives=primitives,
        reference_q=changed_reference_q,
        reference_phi=reference_phi,
        shape_map=shape,
        scalar_map=scalar,
    )
    assert changed_reference.identifier != contract.identifier
    assert np.array_equal(
        changed_reference.position_r_first, contract.position_r_first,
    )
    free_derivative = dict(primitives)
    free_derivative["reference_q_r"] = np.zeros_like(z)
    with np.testing.assert_raises_regex(ValueError, "exactly"):
        _derive_protocol125_position_outer_contract_from_primitives(
            z,
            source_r,
            fields,
            completed_primitives=free_derivative,
            reference_q=reference_q,
            reference_phi=reference_phi,
            shape_map=shape,
            scalar_map=scalar,
        )
    bad_fields = {name: value.copy() for name, value in fields.items()}
    bad_fields["v_z"][3] = 1e-16
    with np.testing.assert_raises_regex(ValueError, "IEEE positive zero"):
        _derive_protocol125_position_outer_contract_from_primitives(
            z,
            source_r,
            bad_fields,
            completed_primitives=primitives,
            reference_q=reference_q,
            reference_phi=reference_phi,
            shape_map=shape,
            scalar_map=scalar,
        )


def test_public_position_outer_type_cannot_accept_derivative_primitives():
    assert not hasattr(
        boundary_contracts,
        "derive_protocol125_position_outer_derivative_bundle",
    )
    assert not hasattr(
        Protocol125PositionOuterOpenFaceDerivativeContract,
        "derive",
    )
    with pytest.raises(TypeError, match="complete parent"):
        Protocol125PositionOuterOpenFaceDerivativeContract(
            alpha_r=np.zeros(3),
        )


def test_protocol125_two_state_outer_contract_derives_native_acceleration_row():
    (
        z,
        source_r,
        fields,
        primitives,
        reference_q,
        reference_phi,
        shape,
        scalar,
        _,
    ) = _protocol125_outer_context()
    position = _derive_protocol125_position_outer_contract_from_primitives(
        z,
        source_r,
        fields,
        completed_primitives=primitives,
        reference_q=reference_q,
        reference_phi=reference_phi,
        shape_map=shape,
        scalar_map=scalar,
    )
    lanes = np.arange(len(NATIVE_CHANNEL_ORDER), dtype=float)[None, None, :]
    source = (
        (0.3+0.02*z[:, None, None]**2)*(1.0+0.01*lanes)
        + (0.04+0.003*lanes)*source_r[None, :, None]**4
    )
    acceleration_fields = {
        name: source[:, :, index]
        for index, name in enumerate(NATIVE_CHANNEL_ORDER)
    }
    contract = derive_protocol125_outer_derivative_bundle(
        position, acceleration_fields,
    )
    assert isinstance(contract, Protocol125OuterOpenFaceDerivativeContract)
    radial_operator = derivative_matrix(source_r, 1, 7)
    expected = np.stack(tuple(
        (radial_operator @ acceleration_fields[name].T).T[:, -1]
        for name in NATIVE_CHANNEL_ORDER
    ), axis=-1)
    np.testing.assert_allclose(
        contract.acceleration_r_first, expected, rtol=0.0, atol=2e-12,
    )
    query = contract.r_first_z_jets(
        state_name="acceleration",
        compact_coordinate=z,
        outer_value_z_jets=(np.zeros_like(expected),),
    )
    np.testing.assert_allclose(
        query.r_first_z_jets[0], expected, rtol=0.0, atol=2e-12,
    )
    np.testing.assert_array_equal(
        query.ownership_mask,
        np.ones(len(NATIVE_CHANNEL_ORDER), dtype=bool),
    )
    position_query = contract.r_first_z_jets(
        state_name="position",
        compact_coordinate=z,
        outer_value_z_jets=(np.zeros_like(expected),),
    )
    np.testing.assert_allclose(
        position_query.r_first_z_jets[0], position.position_r_first,
        rtol=0.0, atol=2e-15,
    )
    record = contract.coefficient_arrays()
    np.testing.assert_array_equal(
        record["position_reference_outer_values"],
        position.coefficient_arrays()["reference_outer_values"],
    )
    assert str(record["position_contract_identifier"]) == position.identifier
    assert str(record["acceleration_derivation_recipe"]) == (
        "full-native-source-width-seven-outer-row-v1"
    )
    assert str(record["corner_policy"]) == (
        "deterministic-width-seven-record;compact-wall-owns-corners"
    )
    changed = {name: value.copy() for name, value in acceleration_fields.items()}
    changed["chi"][2, 3] += 1e-12
    assert derive_protocol125_outer_derivative_bundle(
        position, changed,
    ).identifier != contract.identifier
    with pytest.raises(ValueError, match="exactly"):
        derive_protocol125_outer_derivative_bundle(
            position, {**acceleration_fields, "q4": np.zeros_like(source[:, :, 0])},
        )


def test_protocol125_two_state_outer_contract_is_shared_by_final_q53_q33_pair():
    z = np.linspace(1.0, 2.0, 9)
    source_r = np.linspace(0.0, 2.0, 17)
    shape = (len(z), len(source_r))
    position_fields = {
        "h00": -np.ones(shape),
        "h_perp": np.ones(shape),
        "h_rr": np.ones(shape),
        "h_zz": np.ones(shape),
        "Phi": np.zeros(shape),
        "chi": np.zeros(shape),
        "v_z": np.zeros(shape),
        "v_0": np.zeros(shape),
    }
    acceleration_fields = {
        name: np.zeros(shape) for name in NATIVE_CHANNEL_ORDER
    }
    selector_q = np.broadcast_to((1.0-z)[:, None], shape).copy()
    zeros = np.zeros(shape)
    position_outer = _derive_protocol125_position_outer_contract_from_primitives(
        z,
        source_r,
        {name: value[:, -1] for name, value in position_fields.items()},
        completed_primitives={
            "selector_q": selector_q[:, -1],
            "psi": np.ones_like(z),
            "alpha": np.ones_like(z),
            "alpha_r": np.zeros_like(z),
        },
        reference_q=selector_q,
        reference_phi=zeros,
        shape_map={
            name: np.zeros_like(z)
            for name in ("a", "b", "c", "a_r", "b_r", "c_r")
        },
        scalar_map={"chi": np.zeros_like(z), "chi_r": np.zeros_like(z)},
    )
    outer = derive_protocol125_outer_derivative_bundle(
        position_outer, acceleration_fields,
    )
    background = {
        "wall_stiffness": 0.0,
        "v0": 0.0,
        "v1": 0.0,
        "beta_a": 0.0,
        "beta_b": 0.0,
        "wall_potential_a": 0.0,
        "wall_potential_b": 0.0,
    }
    position_stack = np.stack(tuple(position_fields.values()), axis=-1)
    acceleration_stack = np.stack(tuple(acceleration_fields.values()), axis=-1)
    compact = NativeNormalizedCompactWallContract.build(
        source_r,
        background,
        position_stack[[0, -1]],
        np.zeros((2, len(source_r))),
        np.zeros((2, len(source_r))),
    )
    position_endpoint = compact.z_first_s_jets(
        state_name="position",
        radius=source_r,
        wall_value_s_jets=(position_stack[[0, -1]],),
    )[0]
    acceleration_endpoint = compact.z_first_s_jets(
        state_name="acceleration",
        radius=source_r,
        wall_value_s_jets=(acceleration_stack[[0, -1]],),
    )[0]

    def endpoint_mapping(values):
        return {
            name: values[:, :, index]
            for index, name in enumerate(NATIVE_CHANNEL_ORDER)
        }

    pair = RadialFirstConstrainedHermitePair.build(
        z,
        source_r,
        position_fields,
        acceleration_fields,
        endpoint_mapping(position_endpoint),
        endpoint_mapping(acceleration_endpoint),
        compact_wall_contract=compact,
        outer_open_face_contract=outer,
        parent_r_max=2.0,
    )
    states = (
        pair.primary.position,
        pair.primary.acceleration,
        pair.comparator.position,
        pair.comparator.acceleration,
    )
    assert {state.outer_open_face_contract_id for state in states} == {
        outer.identifier
    }
    assert len({
        state.outer_open_face_contract_fingerprint for state in states
    }) == 1
