"""Acceleration-free position representation for Protocol 125."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.joint_parent_boundary_contracts import (
    NativeNormalizedCompactWallContract,
    _derive_protocol125_position_outer_contract_from_primitives,
)
from bhps.joint_parent_native_completion import (
    complete_normal_gauge_source_wall,
)
from bhps.joint_parent_representation import (
    NATIVE_CHANNEL_ORDER,
    RadialFirstConstrainedHermiteState,
)


def native_channel_mapping_from_reduced(reduced, r):
    """Return the eight represented native channels from one reduced field."""
    q = np.asarray(reduced, dtype=float)
    r = np.asarray(r, dtype=float)
    if q.ndim != 3 or q.shape[1:] != (len(r), 9):
        raise ValueError("native reduced field has the wrong shape")
    radius = r[None, :]
    mapping = {
        "h00": q[:, :, 2],
        "h_perp": q[:, :, 3],
        "h_rr": q[:, :, 3]+radius**2*q[:, :, 4],
        "h_zz": q[:, :, 6],
        "Phi": q[:, :, 7],
        "chi": q[:, :, 8],
        "v_z": q[:, :, 1],
        "v_0": q[:, :, 5],
    }
    if tuple(mapping) != NATIVE_CHANNEL_ORDER:
        raise AssertionError("native channel mapping order changed")
    if not all(np.all(np.isfinite(value)) for value in mapping.values()):
        raise RuntimeError("native channel map is nonfinite")
    return {name: np.asarray(value).copy() for name, value in mapping.items()}


def _stack_mapping(mapping):
    if tuple(mapping) != NATIVE_CHANNEL_ORDER:
        raise ValueError("native mapping has the wrong channel order")
    return np.stack(tuple(mapping.values()), axis=-1)


def _endpoint_mapping(values):
    return {
        name: np.asarray(values[:, :, index]).copy()
        for index, name in enumerate(NATIVE_CHANNEL_ORDER)
    }


def _required_parent_array(parent, name, shape):
    if name not in parent:
        raise ValueError(f"joint parent is missing {name}")
    value = np.asarray(parent[name], dtype=float)
    if value.shape != tuple(shape) or not np.all(np.isfinite(value)):
        raise ValueError(f"joint parent entry {name} is invalid")
    return value


def derive_joint_parent_position_outer_contract(
    parent,
    *,
    validation_tolerance=1e-12,
):
    """Derive the canonical Protocol-125 position outer contract.

    The lapse is always reconstructed from the completed native ``h00``.
    On open compact rows its radial derivative is the selector conformal
    derivative implied by the balanced delta-Robin row.  The compact-wall
    corners are not outer owners; nevertheless they receive a deterministic
    recorded value from the same native width-seven radial derivative of the
    completed lapse used elsewhere in the parent audit.
    """
    if not isinstance(parent, Mapping):
        raise TypeError("joint parent must be a mapping")
    if "z" not in parent or "r" not in parent:
        raise ValueError("joint parent is missing source coordinates")
    z = np.asarray(parent["z"], dtype=float)
    r = np.asarray(parent["r"], dtype=float)
    if (
        z.ndim != 1
        or r.ndim != 1
        or len(z) < 6
        or len(r) < 7
        or r[0] != 0.0
        or np.any(np.diff(z) <= 0.0)
        or np.any(np.diff(r) <= 0.0)
        or not np.all(np.isfinite(z))
        or not np.all(np.isfinite(r))
    ):
        raise ValueError("joint parent coordinates are invalid")
    source_shape = (len(z), len(r))
    position = _required_parent_array(
        parent, "position", source_shape+(9,),
    )
    mapping = native_channel_mapping_from_reduced(position, r)
    selector_q = _required_parent_array(
        parent, "selector_q", source_shape,
    )
    psi_selector = _required_parent_array(
        parent, "psi_selector", source_shape,
    )
    reference_q = _required_parent_array(
        parent, "reference_q", source_shape,
    )
    reference_phi = _required_parent_array(
        parent, "reference_phi", source_shape,
    )
    shape = {
        name: _required_parent_array(
            parent,
            f"shape_{name}",
            source_shape,
        )[:, -1]
        for name in ("a", "b", "c", "a_r", "b_r", "c_r")
    }
    chi = _required_parent_array(parent, "chi", source_shape)
    chi_r = _required_parent_array(parent, "chi_r", source_shape)

    h00 = np.asarray(mapping["h00"], dtype=float)
    if np.any(h00 >= 0.0):
        raise ValueError("completed joint-parent lapse is not real and positive")
    alpha = np.sqrt(-h00)
    radial_operator = derivative_matrix(r, 1, 7)
    native_alpha_r = (radial_operator @ alpha.T).T[:, -1]
    reference_q_r = (radial_operator @ reference_q.T).T[:, -1]
    selector_q_outer = selector_q[:, -1]
    reference_q_outer = reference_q[:, -1]
    q_r = reference_q_r-(selector_q_outer-reference_q_outer)/float(r[-1])
    psi_outer = psi_selector[:, -1]
    psi_r = -psi_outer**2*q_r
    alpha_r = native_alpha_r.copy()
    alpha_r[1:-1] = psi_r[1:-1]

    return _derive_protocol125_position_outer_contract_from_primitives(
        z,
        r,
        {name: value[:, -1] for name, value in mapping.items()},
        completed_primitives={
            "selector_q": selector_q_outer,
            "psi": psi_outer,
            "alpha": alpha[:, -1],
            "alpha_r": alpha_r,
        },
        reference_q=reference_q,
        reference_phi=reference_phi,
        shape_map=shape,
        scalar_map={
            "chi": chi[:, -1],
            "chi_r": chi_r[:, -1],
        },
        validation_tolerance=validation_tolerance,
    )


def build_joint_parent_position_state(
    position,
    z,
    r,
    background,
    *,
    outer_open_face_contract,
    parent_r_max=12.0,
):
    """Build the temporary position-only Q53 state and its audit record."""
    q = np.asarray(position, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    expected = (len(z), len(r), 9)
    if q.shape != expected:
        raise ValueError("completed position has the wrong shape")
    velocity = np.zeros_like(q)
    if np.any(velocity != 0.0) or np.any(np.signbit(velocity)):
        raise AssertionError("positive-zero velocity construction failed")
    raw_source = np.zeros((len(z), len(r), 3), dtype=float)
    source, source_record = complete_normal_gauge_source_wall(
        q, raw_source, z, r, background, stencil_width=7,
    )
    mapping = native_channel_mapping_from_reduced(q, r)
    stack = _stack_mapping(mapping)
    compact = NativeNormalizedCompactWallContract.build_position(
        r,
        background,
        stack[[0, -1]],
        source[[0, -1], :, 1],
    )
    endpoint_values = compact.z_first_s_jets(
        state_name="position",
        radius=r,
        wall_value_s_jets=(stack[[0, -1]],),
    )[0]
    state = RadialFirstConstrainedHermiteState.build_position(
        z,
        r,
        mapping,
        _endpoint_mapping(endpoint_values),
        compact_wall_contract=compact,
        outer_open_face_contract=outer_open_face_contract,
        parent_r_max=float(parent_r_max),
        z_degree=5,
    )
    represented = state.evaluate_reduced(z, r)
    scale = np.maximum.reduce((
        np.ones_like(q), np.abs(q), np.abs(represented),
    ))
    reproduction = float(np.max(np.abs(represented-q)/scale))
    if reproduction > 1e-12:
        raise RuntimeError(
            "position-only representation does not reproduce source data"
        )
    return state, {
        "method": "Protocol-125-position-only-radial-first-Q53",
        "source_normal_wall": source[[0, -1], :, 1].copy(),
        "source_completion": source_record,
        "endpoint_z_first": endpoint_values.copy(),
        "source_reproduction_scaled_Linf": reproduction,
        "state_fingerprint": state.fingerprint(),
        "compact_contract_identifier": compact.identifier,
        "outer_contract_identifier": outer_open_face_contract.identifier,
        "acceleration_placeholder_used": False,
        "velocity_positive_zero": True,
    }
