from __future__ import annotations

import numpy as np
import pytest

import bhps.joint_parent_acceleration as acceleration_module
from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.joint_parent_acceleration import (
    ACCELERATION_FAILURE_PROTOCOL_IDENTIFIER,
    Protocol125AccelerationScientificFailure,
    reconcile_joint_parent_native_axis_null_channels,
    represented_position_jet,
    solve_joint_parent_acceleration_fixed_point,
    validate_protocol125_acceleration_failure_record,
)
from bhps.joint_parent_boundary_contracts import (
    NativeNormalizedCompactWallContract,
    StoredOuterOpenFaceDerivativeContract,
)
from bhps.joint_parent_representation import (
    NATIVE_CHANNEL_ORDER,
    RadialFirstConstrainedHermiteRepresentation,
)
from bhps.joint_parent_selective_algebra import (
    SelectiveWallAlgebraicGateError,
)
from bhps.nonlinear_regular_so3_evolution import (
    CompactWallCoupledAlgebraicGateError,
)


PARENT_IDENTITY = "a"*64


def _flat_case():
    z = np.linspace(1.0, 2.0, 9)
    r = np.linspace(0.0, 2.0, 17)
    shape = (len(z), len(r))
    native = {
        name: np.zeros(shape) for name in NATIVE_CHANNEL_ORDER
    }
    native["h00"][:] = -1.0
    native["h_perp"][:] = 1.0
    native["h_rr"][:] = 1.0
    native["h_zz"][:] = 1.0
    acceleration = {
        name: np.zeros(shape) for name in NATIVE_CHANNEL_ORDER
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
    position_stack = np.stack(
        [native[name] for name in NATIVE_CHANNEL_ORDER], axis=-1,
    )
    zero_wall = np.zeros((2, len(r)))
    compact = NativeNormalizedCompactWallContract.build(
        r,
        background,
        position_stack[[0, -1]],
        zero_wall,
        zero_wall,
    )
    endpoints = compact.z_first_s_jets(
        state_name="position",
        radius=r,
        wall_value_s_jets=(position_stack[[0, -1]],),
    )[0]
    acceleration_stack = np.stack(
        [acceleration[name] for name in NATIVE_CHANNEL_ORDER], axis=-1,
    )
    acceleration_endpoints = compact.z_first_s_jets(
        state_name="acceleration",
        radius=r,
        wall_value_s_jets=(acceleration_stack[[0, -1]],),
    )[0]
    outer = StoredOuterOpenFaceDerivativeContract.build(
        z,
        np.zeros((len(z), len(NATIVE_CHANNEL_ORDER))),
        np.zeros((len(z), len(NATIVE_CHANNEL_ORDER))),
    )
    representation = RadialFirstConstrainedHermiteRepresentation.build(
        z,
        r,
        native,
        acceleration,
        {
            name: endpoints[:, :, index]
            for index, name in enumerate(NATIVE_CHANNEL_ORDER)
        },
        {
            name: acceleration_endpoints[:, :, index]
            for index, name in enumerate(NATIVE_CHANNEL_ORDER)
        },
        compact_wall_contract=compact,
        outer_open_face_contract=outer,
        parent_r_max=2.0,
    )
    q = representation.position.evaluate_reduced(z, r)
    return z, r, q, np.zeros_like(q), background, representation.position


def test_represented_position_jet_uses_analytic_spatial_lanes():
    z, r, q, acceleration, _, state = _flat_case()
    jet = represented_position_jet(state, z, r, acceleration)
    np.testing.assert_array_equal(jet.reduced_fields, q)
    assert np.array_equal(jet.reduced_first[0], np.zeros_like(q))
    assert np.max(np.abs(jet.reduced_first[1:])) < 2e-13
    assert np.max(np.abs(jet.reduced_second[1:, 1:])) < 2e-12


def _manufactured_native_parity_data():
    r = np.asarray((0.0, 0.11, 0.27, 0.53, 0.91, 1.42, 2.03, 2.71, 3.5))
    parent_radius = r[-1]
    s = (r/parent_radius)**2
    f0 = np.asarray((0.75, -1.25, 2.5))
    f1 = np.asarray((-0.2, 0.4, 0.7))
    f2 = np.asarray((0.03, -0.08, 0.11))
    g0 = np.asarray((-0.6, 1.1, 2.2))
    g1 = np.asarray((0.07, -0.13, 0.19))
    g2 = np.asarray((-0.004, 0.009, -0.015))
    acceleration = np.arange(3*len(r)*9, dtype=float).reshape(3, len(r), 9)/17
    acceleration[:, :, 4] = (
        f0[:, None] + f1[:, None]*s + f2[:, None]*s**2
    )
    acceleration[:, :, 5] = (
        g0[:, None] + g1[:, None]*r**2 + g2[:, None]*r**4
    )
    anisotropy = np.zeros((3, len(r)))
    time_radial = np.zeros((3, len(r)))
    anisotropy[:, 1:] = r[None, 1:]**2*acceleration[:, 1:, 4]
    time_radial[:, 1:] = r[None, 1:]*acceleration[:, 1:, 5]
    acceleration[:, 0, 4] = np.asarray((101.0, -202.0, 303.0))
    acceleration[:, 0, 5] = np.asarray((-404.0, 505.0, -606.0))
    return acceleration, r, anisotropy, time_radial, f0, g0


def test_native_axis_reconciliation_differentiates_physical_numerators():
    acceleration, r, anisotropy, time_radial, f0, g0 = (
        _manufactured_native_parity_data()
    )
    original = acceleration.copy()
    found, record = reconcile_joint_parent_native_axis_null_channels(
        acceleration,
        r,
        anisotropy_numerator_tt=anisotropy,
        time_radial_tt=time_radial,
    )

    s = (r/r[-1])**2
    ds = derivative_matrix(s, 1, 7).toarray()
    dr = derivative_matrix(r, 1, 7).toarray()
    expected_q4 = (ds @ anisotropy.T).T[:, 0]/r[-1]**2
    expected_q5 = (dr @ time_radial.T).T[:, 0]
    np.testing.assert_array_equal(found[:, 0, 4], expected_q4)
    np.testing.assert_array_equal(found[:, 0, 5], expected_q5)
    np.testing.assert_allclose(found[:, 0, 4], f0, rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(found[:, 0, 5], g0, rtol=0.0, atol=2e-12)

    original_bits = np.ascontiguousarray(original).view(np.uint64).reshape(
        original.shape
    )
    found_bits = np.ascontiguousarray(found).view(np.uint64).reshape(found.shape)
    protected = np.ones(original.shape, dtype=bool)
    protected[:, 0, 4:6] = False
    assert np.array_equal(found_bits[protected], original_bits[protected])
    np.testing.assert_array_equal(acceleration, original)
    assert record["stencil_width"] == 7
    assert record["only_q4_q5_axis_changed_bitwise"]
    assert record["positive_radius_reassembly_bitwise"]
    assert not record["polynomial_fit_applied"]


@pytest.mark.parametrize(
    ("name", "bad_value"),
    (
        ("anisotropy", -0.0),
        ("time_radial", -0.0),
        ("anisotropy", np.nextafter(0.0, 1.0)),
        ("time_radial", np.nextafter(0.0, 1.0)),
    ),
)
def test_native_axis_reconciliation_requires_exact_positive_zero_numerators(
    name, bad_value,
):
    acceleration, r, anisotropy, time_radial, _, _ = (
        _manufactured_native_parity_data()
    )
    target = anisotropy if name == "anisotropy" else time_radial
    target[1, 0] = bad_value
    with pytest.raises(ValueError, match="exact IEEE positive zero"):
        reconcile_joint_parent_native_axis_null_channels(
            acceleration,
            r,
            anisotropy_numerator_tt=anisotropy,
            time_radial_tt=time_radial,
        )


def test_native_axis_reconciliation_rejects_non_native_inputs():
    acceleration, r, anisotropy, time_radial, _, _ = (
        _manufactured_native_parity_data()
    )
    negative_zero_grid = r.copy()
    negative_zero_grid[0] = -0.0
    with pytest.raises(ValueError, match=r"exact \+0 axis"):
        reconcile_joint_parent_native_axis_null_channels(
            acceleration,
            negative_zero_grid,
            anisotropy_numerator_tt=anisotropy,
            time_radial_tt=time_radial,
        )

    mismatched = anisotropy.copy()
    mismatched[0, 3] = np.nextafter(mismatched[0, 3], np.inf)
    with pytest.raises(ValueError, match="disagree with the reduced"):
        reconcile_joint_parent_native_axis_null_channels(
            acceleration,
            r,
            anisotropy_numerator_tt=mismatched,
            time_radial_tt=time_radial,
        )

    nonfinite = time_radial.copy()
    nonfinite[0, 2] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        reconcile_joint_parent_native_axis_null_channels(
            acceleration,
            r,
            anisotropy_numerator_tt=anisotropy,
            time_radial_tt=nonfinite,
        )


def test_zero_fixed_point_converges_twice_without_hidden_owners():
    z, r, q, acceleration, background, state = _flat_case()
    found, record = solve_joint_parent_acceleration_fixed_point(
        state, q, acceleration, z, r, background,
        parent_label="N0",
        parent_identity=PARENT_IDENTITY,
    )
    assert np.max(np.abs(found)) < 2e-12
    assert record["maps_used"] == 2
    assert record["consecutive_converged_maps"] == 2
    assert record["normal_gauge"]["maximum"] < 1e-12
    assert record["axis_reconciliation"]["method"] == (
        "native-seven-point-physical-numerator-axis-parity"
    )
    assert not record["axis_reconciliation"]["polynomial_fit_applied"]
    assert not record["outer_overwrite_applied"]
    assert not record["generic_axis_fill_applied"]
    assert not record["endpoint_history_carried"]


def _source_triplet_following_acceleration(jet, *_args):
    value = jet.reduced_second[0, 0, :, :, 2]
    source = np.repeat(value[:, :, None], 3, axis=2)
    return {
        "source": source,
        "source_time": source.copy(),
        "source_second_time": source.copy(),
    }


def _identity_coupled(
    _q, _v, acceleration, *_args, **_kwargs,
):
    return np.asarray(acceleration, dtype=float).copy(), {
        "owner": "manufactured-coupled",
        "passed": True,
    }


def _identity_selective(
    _q, _v, acceleration, *_args, **_kwargs,
):
    return np.asarray(acceleration, dtype=float).copy(), {
        "owner": "manufactured-selective",
        "passed": True,
        "protected_0_1_6_7_bitwise": True,
        "q4_q5_axis_bitwise": True,
    }


def _solve_flat(state, q, acceleration, z, r, background):
    return solve_joint_parent_acceleration_fixed_point(
        state,
        q,
        acceleration,
        z,
        r,
        background,
        parent_label="N0",
        parent_identity=PARENT_IDENTITY,
    )


def test_fixed_point_nonconvergence_is_typed_sealed_scientific_failure(
    monkeypatch,
):
    z, r, q, acceleration, background, state = _flat_case()
    monkeypatch.setattr(
        acceleration_module,
        "initial_driver_source_triplet_from_acceleration",
        _source_triplet_following_acceleration,
    )

    def increment_from_source(
        _q, _v, work, source, *_args, **_kwargs,
    ):
        result = np.asarray(work, dtype=float).copy()
        result[:, :, 2] = np.asarray(source)[:, :, 0] + 1.0
        return result, {"owner": "manufactured-coupled", "passed": True}

    monkeypatch.setattr(
        acceleration_module,
        "solve_compact_wall_coupled_phi_normal_acceleration",
        increment_from_source,
    )
    monkeypatch.setattr(
        acceleration_module,
        "solve_compact_wall_tangential_chi_acceleration",
        _identity_selective,
    )
    with pytest.raises(Protocol125AccelerationScientificFailure) as caught:
        _solve_flat(state, q, acceleration, z, r, background)
    record = caught.value.record
    assert record["protocol_identifier"] == (
        ACCELERATION_FAILURE_PROTOCOL_IDENTIFIER
    )
    assert record["classification"] == "FAIL-acceleration"
    assert record["failure_group"] == "acceleration_closure"
    assert record["failure_reason"] == "fixed_point_nonconvergence"
    assert record["maps_completed"] == 8
    assert len(record["history"]) == 8
    np.testing.assert_array_equal(
        record["last_iterate_evidence"]["acceleration"][:, :, 2],
        np.full((len(z), len(r)), 8.0),
    )
    assert record["last_iterate_evidence"]["acceleration"].flags.writeable is False
    with pytest.raises(TypeError):
        record["history"][0]["map"] = 99
    assert (
        validate_protocol125_acceleration_failure_record(record)["fingerprint"]
        == record["fingerprint"]
    )
    tampered = dict(record)
    tampered["maps_completed"] = 7
    with pytest.raises(ValueError):
        validate_protocol125_acceleration_failure_record(tampered)


@pytest.mark.parametrize(
    ("owner", "error", "reason"),
    (
        (
            "coupled",
            CompactWallCoupledAlgebraicGateError(
                "manufactured coupled rejection",
                radial_index=3,
                gate="rank_condition_pivot",
                diagnostics={"rank": 3, "required_rank": 4},
            ),
            "coupled_wall_algebraic_gate_failure",
        ),
        (
            "selective",
            SelectiveWallAlgebraicGateError(
                "manufactured selective rejection",
                radial_index=4,
                field="h_00",
                gate="normalized_linear_residual",
                diagnostics={
                    "normalized_linear_residual": 2e-12,
                    "maximum_allowed_normalized_linear_residual": 1e-12,
                },
            ),
            "selective_wall_algebraic_gate_failure",
        ),
    ),
)
def test_structured_wall_algebra_rejections_are_scientific_not_technical(
    monkeypatch, owner, error, reason,
):
    z, r, q, acceleration, background, state = _flat_case()
    monkeypatch.setattr(
        acceleration_module,
        "initial_driver_source_triplet_from_acceleration",
        _source_triplet_following_acceleration,
    )
    if owner == "coupled":
        def reject(*_args, **_kwargs):
            raise error

        monkeypatch.setattr(
            acceleration_module,
            "solve_compact_wall_coupled_phi_normal_acceleration",
            reject,
        )
    else:
        monkeypatch.setattr(
            acceleration_module,
            "solve_compact_wall_coupled_phi_normal_acceleration",
            _identity_coupled,
        )

        def reject(*_args, **_kwargs):
            raise error

        monkeypatch.setattr(
            acceleration_module,
            "solve_compact_wall_tangential_chi_acceleration",
            reject,
        )
    with pytest.raises(Protocol125AccelerationScientificFailure) as caught:
        _solve_flat(state, q, acceleration, z, r, background)
    record = caught.value.record
    assert caught.value.__cause__ is error
    assert record["failure_group"] == "wall_algebra"
    assert record["failure_reason"] == reason
    assert record["maps_completed"] == 0
    event = record["failure_event"]
    assert event["owner"] == owner
    assert event["failed_map"] == 1
    assert event["radial_index"] == error.radial_index
    assert event["radius"] == r[error.radial_index]
    assert event["gate"] == error.gate


def test_finite_legacy_selective_residual_rejection_is_scientific(
    monkeypatch,
):
    z, r, q, acceleration, background, state = _flat_case()
    monkeypatch.setattr(
        acceleration_module,
        "initial_driver_source_triplet_from_acceleration",
        _source_triplet_following_acceleration,
    )
    monkeypatch.setattr(
        acceleration_module,
        "solve_compact_wall_coupled_phi_normal_acceleration",
        _identity_coupled,
    )

    def reject(*_args, **_kwargs):
        raise RuntimeError(
            "normalized tangential/chi wall residual gate failed: "
            "residual=2e-12, limit=1e-12"
        )

    monkeypatch.setattr(
        acceleration_module,
        "solve_compact_wall_tangential_chi_acceleration",
        reject,
    )
    with pytest.raises(Protocol125AccelerationScientificFailure) as caught:
        _solve_flat(state, q, acceleration, z, r, background)
    record = caught.value.record
    assert record["failure_reason"] == (
        "selective_wall_normalized_residual_gate_failure"
    )
    assert record["failure_event"]["diagnostics"][
        "radial_localization_unavailable_from_legacy_exception"
    ]


def test_final_normal_gauge_closure_failure_retains_final_measurement(
    monkeypatch,
):
    z, r, q, acceleration, background, state = _flat_case()
    normal = {
        "walls": [
            {
                "wall": "lower",
                "maximum_normalized": 2e-10,
                "maximum_absolute": 2e-10,
            },
            {
                "wall": "upper",
                "maximum_normalized": 3e-10,
                "maximum_absolute": 3e-10,
            },
        ],
        "maximum": 3e-10,
    }
    monkeypatch.setattr(
        acceleration_module,
        "compact_wall_normal_gauge_acceleration_residuals",
        lambda *_args, **_kwargs: normal,
    )
    with pytest.raises(Protocol125AccelerationScientificFailure) as caught:
        _solve_flat(state, q, acceleration, z, r, background)
    record = caught.value.record
    assert record["failure_group"] == "acceleration_closure"
    assert record["failure_reason"] == "final_normal_gauge_closure_failure"
    assert record["maps_completed"] == 2
    assert record["final_normal_gauge"]["maximum"] == 3e-10
    assert tuple(record["final_wall_second_tangent"]) == ("lower", "upper")


@pytest.mark.parametrize(
    "message",
    (
        "unexpected selective implementation failure",
        (
            "normalized tangential/chi wall residual gate failed: "
            "residual=nan, limit=1e-12"
        ),
    ),
)
def test_unstructured_or_nonfinite_runtime_failures_remain_technical(
    monkeypatch, message,
):
    z, r, q, acceleration, background, state = _flat_case()
    monkeypatch.setattr(
        acceleration_module,
        "initial_driver_source_triplet_from_acceleration",
        _source_triplet_following_acceleration,
    )
    monkeypatch.setattr(
        acceleration_module,
        "solve_compact_wall_coupled_phi_normal_acceleration",
        _identity_coupled,
    )

    def crash(*_args, **_kwargs):
        raise RuntimeError(message)

    monkeypatch.setattr(
        acceleration_module,
        "solve_compact_wall_tangential_chi_acceleration",
        crash,
    )
    with pytest.raises((RuntimeError, ValueError)) as caught:
        _solve_flat(state, q, acceleration, z, r, background)
    assert not isinstance(caught.value, Protocol125AccelerationScientificFailure)


def test_malformed_structured_gate_exception_remains_technical(monkeypatch):
    z, r, q, acceleration, background, state = _flat_case()
    monkeypatch.setattr(
        acceleration_module,
        "initial_driver_source_triplet_from_acceleration",
        _source_triplet_following_acceleration,
    )
    malformed = CompactWallCoupledAlgebraicGateError(
        "manufactured malformed gate",
        radial_index=2,
        gate="unregistered_gate",
        diagnostics={"rank": 3},
    )

    def reject(*_args, **_kwargs):
        raise malformed

    monkeypatch.setattr(
        acceleration_module,
        "solve_compact_wall_coupled_phi_normal_acceleration",
        reject,
    )
    with pytest.raises(ValueError, match="diagnostics are malformed") as caught:
        _solve_flat(state, q, acceleration, z, r, background)
    assert not isinstance(caught.value, Protocol125AccelerationScientificFailure)
