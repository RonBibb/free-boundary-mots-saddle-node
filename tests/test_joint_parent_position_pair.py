from __future__ import annotations

import numpy as np

from bhps.joint_parent_position_pair import PositionOnlyConstrainedHermitePair
from bhps.joint_parent_representation import (
    NATIVE_CHANNEL_ORDER,
    OuterOpenFaceDerivativeResult,
    RadialFirstConstrainedHermiteState,
)


class _ZeroCompactContract:
    identifier = "test-position-compact-zero-z-first"

    def z_first_s_jets(self, *, state_name, radius, wall_value_s_jets):
        assert state_name == "position"
        return tuple(np.zeros_like(value) for value in wall_value_s_jets)

    def coefficient_arrays(self):
        return {"zero_compact_recipe": np.asarray("exact-zero")}


class _ZeroOuterContract:
    identifier = "test-position-outer-zero-r-first"

    def r_first_z_jets(self, *, state_name, compact_coordinate, outer_value_z_jets):
        assert state_name == "position"
        return OuterOpenFaceDerivativeResult(
            tuple(np.zeros_like(value) for value in outer_value_z_jets),
            np.ones(len(NATIVE_CHANNEL_ORDER), dtype=bool),
        )

    def coefficient_arrays(self):
        return {
            "zero_outer_recipe": np.asarray("exact-zero"),
            "ownership": np.ones(len(NATIVE_CHANNEL_ORDER), dtype=bool),
        }


def _real_position_state():
    z = np.linspace(1.0, np.e, 17)
    r = np.linspace(0.0, 12.0, 19)
    rr = r[None, :]
    shape = (len(z), len(r))
    fields = {
        "h00": -1.1*np.ones(shape),
        "h_perp": np.broadcast_to(1.0+0.002*rr**2, shape).copy(),
        "h_rr": np.broadcast_to(1.0+0.003*rr**2, shape).copy(),
        "h_zz": 1.2*np.ones(shape),
        "Phi": 0.03*np.ones(shape),
        "chi": -0.02*np.ones(shape),
        "v_z": np.zeros(shape),
        "v_0": np.zeros(shape),
    }
    endpoints = {
        name: np.zeros((2, len(r))) for name in NATIVE_CHANNEL_ORDER
    }
    compact = _ZeroCompactContract()
    outer = _ZeroOuterContract()
    state = RadialFirstConstrainedHermiteState.build_position(
        z,
        r,
        fields,
        endpoints,
        compact_wall_contract=compact,
        outer_open_face_contract=outer,
        z_degree=5,
    )
    return state, compact, outer


def test_real_position_state_builds_identical_contract_q53_q33_pair():
    state, _, _ = _real_position_state()
    pair = PositionOnlyConstrainedHermitePair.from_primary(state)
    assert pair.primary is state
    assert pair.primary.z_degree == 5
    assert pair.comparator.z_degree == 3
    assert pair.primary.state_name == pair.comparator.state_name == "position"
    assert pair.source_fingerprint
    assert pair.endpoint_fingerprint
    assert pair.fingerprint()
    assert np.array_equal(
        pair.primary.radial_channels.coefficients,
        pair.comparator.radial_channels.coefficients,
    )
    assert np.array_equal(
        pair.primary.stored_z_first_endpoints,
        pair.comparator.stored_z_first_endpoints,
    )
    assert pair.primary.compact_wall_contract_fingerprint == (
        pair.comparator.compact_wall_contract_fingerprint
    )
    assert pair.primary.outer_open_face_contract_fingerprint == (
        pair.comparator.outer_open_face_contract_fingerprint
    )


def test_position_only_pair_persists_and_reloads_without_acceleration():
    state, compact, outer = _real_position_state()
    pair = PositionOnlyConstrainedHermitePair.from_primary(state)
    arrays = pair.coefficient_arrays()
    restored = PositionOnlyConstrainedHermitePair.from_arrays(
        arrays,
        compact_wall_contract=compact,
        outer_open_face_contract=outer,
    )
    assert restored.fingerprint() == pair.fingerprint()
    query_z = np.linspace(1.03, np.e-0.03, 15)
    query_r = np.linspace(0.0, 12.0, 17)
    for original, reloaded in (
        (pair.primary, restored.primary),
        (pair.comparator, restored.comparator),
    ):
        for z_order, r_order in ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2)):
            np.testing.assert_array_equal(
                original.evaluate_coordinate_components(
                    query_z, query_r, z_order=z_order, r_order=r_order,
                ),
                reloaded.evaluate_coordinate_components(
                    query_z, query_r, z_order=z_order, r_order=r_order,
                ),
            )
