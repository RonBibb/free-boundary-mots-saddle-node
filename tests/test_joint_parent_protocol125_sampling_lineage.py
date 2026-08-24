from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.interpolate import BSpline

from bhps.joint_parent_protocol125_sampling_lineage import (
    AXIS_COEFFICIENT_ORDER,
    DENSE_WALL_POINT_COUNT,
    NATIVE_BULK_CHANNEL_ORDER,
    PHYSICAL_ACCELERATION_ORDER,
    V2_COMPACT_POINT_COUNT,
    PositionPayloadSnapshot,
    Protocol125BulkAccelerationSampler,
    Protocol125LineageError,
    Protocol125SamplingError,
    validate_append_only_position_lineage,
)


def _manufactured_reduced(z, r):
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    x = (z[:, None]-1.0)/(np.e-1.0)
    u = (r[None, :]/12.0)**2
    q = np.zeros((len(z), len(r), 9), dtype=float)
    q[:, :, 1] = 0.10+0.02*x+0.03*u+0.01*u**2
    q[:, :, 2] = -0.20+0.01*x+0.04*u-0.02*u**2
    q[:, :, 3] = 0.30+0.03*x+0.02*u+0.01*u**2
    q[:, :, 4] = 0.04+0.01*x+0.02*x**2+0.005*u+0.002*u**2
    q[:, :, 5] = -0.03+0.015*x-0.01*x**2+0.004*u**2
    q[:, :, 6] = 0.12-0.02*x+0.015*u+0.005*u**2
    q[:, :, 7] = 0.07+0.01*x+0.02*u-0.003*u**2
    q[:, :, 8] = -0.08+0.02*x**2-0.01*u+0.004*u**2
    return q


def _manufactured_source(nz=11, nr=13):
    z = np.linspace(1.0, np.e, nz)
    r = np.linspace(0.0, 12.0, nr)
    return z, r, _manufactured_reduced(z, r)


def _physical(reduced, r):
    radius = np.asarray(r, dtype=float)[None, :]
    return np.stack((
        reduced[:, :, 0],
        radius*reduced[:, :, 1],
        reduced[:, :, 2],
        reduced[:, :, 3],
        reduced[:, :, 3]+radius**2*reduced[:, :, 4],
        radius*reduced[:, :, 5],
        reduced[:, :, 6],
        reduced[:, :, 7],
        reduced[:, :, 8],
    ), axis=-1)


def test_bulk_sampler_reproduces_manufactured_dense_walls_and_v2_axis():
    z, r, acceleration = _manufactured_source()
    sampler = Protocol125BulkAccelerationSampler.build(z, r, acceleration)
    assert tuple(NATIVE_BULK_CHANNEL_ORDER) == (
        "h00", "h_perp", "h_rr", "h_zz", "Phi", "chi", "v_z", "v_0",
    )
    assert tuple(PHYSICAL_ACCELERATION_ORDER) == (
        "h_z0", "h_zr", "h_00", "h_perp", "h_rr", "h_0r", "h_zz",
        "Phi", "chi",
    )
    assert tuple(AXIS_COEFFICIENT_ORDER) == ("q4", "q5")
    assert sampler.axis_reproduction_scaled_Linf < 1e-13

    dense_r = np.linspace(0.0, 12.0, DENSE_WALL_POINT_COUNT)
    expected_wall = _physical(
        _manufactured_reduced(np.asarray((z[0], z[-1])), dense_r), dense_r,
    )
    np.testing.assert_allclose(
        sampler.dense_wall_physical(), expected_wall, rtol=0.0, atol=2e-13,
    )
    expected_reduced_wall = _manufactured_reduced(
        np.asarray((z[0], z[-1])), dense_r,
    )
    np.testing.assert_allclose(
        sampler.evaluate_wall_reduced(dense_r),
        expected_reduced_wall,
        rtol=0.0,
        atol=2e-13,
    )

    v2_z = np.linspace(1.0, np.e, V2_COMPACT_POINT_COUNT)
    expected_axis = _manufactured_reduced(v2_z, np.asarray((0.0,)))[:, 0]
    sampled_axis = sampler.v2_axis_reduced()
    np.testing.assert_allclose(
        sampled_axis[:, 4:6], expected_axis[:, 4:6], rtol=0.0, atol=2e-13,
    )
    np.testing.assert_array_equal(sampled_axis[:, :4], 0.0)
    np.testing.assert_array_equal(sampled_axis[:, 6:], 0.0)


def test_bulk_axis_quintic_uses_persisted_source_and_width_seven_endpoints():
    z, r, acceleration = _manufactured_source(13, 15)
    sampler = Protocol125BulkAccelerationSampler.build(z, r, acceleration)
    np.testing.assert_allclose(
        sampler.evaluate_axis_coefficients(z),
        sampler.axis_source,
        rtol=0.0,
        atol=2e-13,
    )
    spline = BSpline(
        sampler.axis_knots, sampler.axis_coefficients, 5, axis=0,
    )
    np.testing.assert_allclose(
        spline(z[[0, -1]], nu=1),
        sampler.axis_z_first,
        rtol=0.0,
        atol=2e-12,
    )


def test_bulk_sampler_roundtrip_is_bitwise_and_rejects_coefficient_tamper():
    z, r, acceleration = _manufactured_source()
    sampler = Protocol125BulkAccelerationSampler.build(z, r, acceleration)
    archive = sampler.coefficient_arrays()
    restored = Protocol125BulkAccelerationSampler.from_arrays(archive)
    assert restored.source_fingerprint == sampler.source_fingerprint
    assert restored.sampler_fingerprint == sampler.sampler_fingerprint
    np.testing.assert_array_equal(
        restored.dense_wall_physical(), sampler.dense_wall_physical(),
    )
    np.testing.assert_array_equal(
        restored.v2_axis_reduced(), sampler.v2_axis_reduced(),
    )

    tampered = {name: np.asarray(value).copy() for name, value in archive.items()}
    tampered["protocol125_bulk_sampler_wall_coefficients"][0, 0, 0] += 1e-8
    with pytest.raises(Protocol125SamplingError, match="wall_coefficients"):
        Protocol125BulkAccelerationSampler.from_arrays(tampered)


def test_bulk_sampler_rejects_poisoned_axis_and_nonpositive_zero_gauge_lane():
    z, r, acceleration = _manufactured_source()
    poisoned = acceleration.copy()
    poisoned[:, 0, 4] += 1e-4
    with pytest.raises(Protocol125SamplingError, match="q4/q5 axis"):
        Protocol125BulkAccelerationSampler.build(z, r, poisoned)

    negative_zero = acceleration.copy()
    negative_zero[:, :, 0] = -0.0
    with pytest.raises(ValueError, match="positive zero"):
        Protocol125BulkAccelerationSampler.build(z, r, negative_zero)


def _payload_groups():
    return {
        "source": {
            "z": np.linspace(1.0, np.e, 9),
            "r": np.linspace(0.0, 12.0, 11),
            "position": np.arange(18, dtype=np.float64).reshape(2, 9),
            "velocity": np.zeros((2, 9), dtype=np.float64),
            "normal_source": np.asarray((0.2, -0.1)),
            "reference_fingerprint": np.asarray("reference-source"),
        },
        "compact": {
            "position_knots": np.asarray((1.0, 1.0, np.e, np.e)),
            "position_coefficients": np.arange(12, dtype=float).reshape(3, 4),
            "ownership": np.ones(8, dtype=bool),
        },
        "outer": {
            "position_target": np.arange(16, dtype=float).reshape(2, 8),
            "corner_baseline": np.arange(16, dtype=float).reshape(2, 8)/3.0,
            "ownership": np.ones(8, dtype=bool),
        },
        "coefficients": {
            "Q53": np.arange(24, dtype=float).reshape(3, 8),
            "Q33": np.arange(24, dtype=float).reshape(3, 8)/2.0,
            "axis_q4": np.asarray((0.01, 0.02)),
        },
    }


def _position_only_snapshot(groups=None):
    return PositionPayloadSnapshot.capture_position_only(
        _payload_groups() if groups is None else groups,
        compact_identifier="compact-position-only",
        compact_fingerprint="compact-position-fingerprint",
        outer_identifier="outer-position-only",
        outer_fingerprint="outer-position-fingerprint",
        archive_fingerprint="position-pair-archive",
    )


def _shared_snapshot(parent, groups=None, **overrides):
    values = {
        "compact_identifier": "compact-shared",
        "compact_fingerprint": "compact-shared-fingerprint",
        "outer_identifier": "outer-shared",
        "outer_fingerprint": "outer-shared-fingerprint",
        "archive_fingerprint": "shared-pair-archive",
    }
    values.update(overrides)
    return PositionPayloadSnapshot.capture_shared(
        _payload_groups() if groups is None else groups,
        parent=parent,
        appended_children={
            "acceleration_coefficients": np.arange(20, dtype=float),
            "source_second": np.asarray((0.4, -0.2)),
        },
        **values,
    )


def test_append_only_lineage_preserves_payload_and_requires_identity_evolution():
    position = _position_only_snapshot()
    shared = _shared_snapshot(position)
    record = validate_append_only_position_lineage(position, shared)
    assert record["passed"]
    assert all(record["gates"].values())
    assert shared.payload_hash == position.payload_hash
    assert shared.child_hash
    for (_, left), (_, right) in zip(
        position.payload_entries, shared.payload_entries,
    ):
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert left.tobytes() == right.tobytes()
        assert not left.flags.writeable
        assert not right.flags.writeable


def test_lineage_roundtrip_and_payload_tamper_fail_closed():
    position = _position_only_snapshot()
    shared = _shared_snapshot(position)
    restored_position = PositionPayloadSnapshot.from_arrays(
        position.coefficient_arrays()
    )
    restored_shared = PositionPayloadSnapshot.from_arrays(
        shared.coefficient_arrays()
    )
    assert validate_append_only_position_lineage(
        restored_position, restored_shared,
    )["passed"]

    archive = shared.coefficient_arrays()
    tampered = {name: np.asarray(value).copy() for name, value in archive.items()}
    tampered["protocol125_position_lineage_payload_0000"].flat[0] += 1.0
    with pytest.raises(Protocol125LineageError, match="payload hash"):
        PositionPayloadSnapshot.from_arrays(tampered)

    child_tamper = {
        name: np.asarray(value).copy() for name, value in archive.items()
    }
    child_tamper["protocol125_position_lineage_child_0000"].flat[0] += 1.0
    with pytest.raises(Protocol125LineageError, match="child hash"):
        PositionPayloadSnapshot.from_arrays(child_tamper)


def test_lineage_rejects_changed_invariant_wrong_parent_and_unchanged_top_id():
    position = _position_only_snapshot()

    changed = _payload_groups()
    changed["coefficients"]["Q53"] = changed["coefficients"]["Q53"].copy()
    changed["coefficients"]["Q53"][0, 0] += 1e-9
    changed_shared = _shared_snapshot(position, changed)
    with pytest.raises(Protocol125LineageError, match="payload"):
        validate_append_only_position_lineage(position, changed_shared)

    wrong_parent = replace(
        _shared_snapshot(position), parent_payload_hash="wrong-parent-hash",
    )
    with pytest.raises(Protocol125LineageError, match="direct_parent_payload"):
        validate_append_only_position_lineage(position, wrong_parent)

    unchanged = _shared_snapshot(
        position, compact_identifier=position.compact_identifier,
    )
    with pytest.raises(
        Protocol125LineageError, match="compact_identifier_evolved",
    ):
        validate_append_only_position_lineage(position, unchanged)
