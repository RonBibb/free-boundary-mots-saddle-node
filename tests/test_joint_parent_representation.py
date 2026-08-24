import numpy as np
import pytest
from types import SimpleNamespace
from scipy.interpolate import make_interp_spline

import bhps.joint_parent_representation as representation_module
from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.joint_parent_representation import (
    COORDINATE_COMPONENT_ORDER,
    NATIVE_CHANNEL_ORDER,
    Protocol125RepresentationCoefficientFailure,
    SEALED_ADVERSE_COMPARATOR_NAMES,
    SEALED_ADVERSE_COMPARATOR_RECIPES,
    NativeMetricHermitePair,
    NativeTensorHermiteSurface,
    OuterOpenFaceDerivativeResult,
    RadialFirstConstrainedHermitePair,
    RadialFirstConstrainedHermiteRepresentation,
    RadialFirstConstrainedHermiteState,
    sealed_reduced_q33_q55_adverse_projections,
    validate_protocol125_representation_coefficient_failure,
)


R_MAX = 12.0


class _ProductCompactWallContract:
    identifier = "synthetic-product-normalized-wall-v1"

    @staticmethod
    def _factor(state_name):
        return 1.0 if state_name == "position" else -0.7

    def z_first_s_jets(self, *, state_name, radius, wall_value_s_jets):
        del radius
        factor = self._factor(state_name)
        h_zz = NATIVE_CHANNEL_ORDER.index("h_zz")
        signs = np.array([-1.0, 1.0])[:, None, None]
        values = tuple(np.asarray(value) for value in wall_value_s_jets)
        metric = tuple(value[:, :, h_zz][:, :, None] for value in values)
        result = [signs*factor*metric[0]*values[0]]
        if len(values) >= 2:
            result.append(signs*factor*(
                metric[1]*values[0]+metric[0]*values[1]
            ))
        if len(values) >= 3:
            result.append(signs*factor*(
                metric[2]*values[0]
                + 2.0*metric[1]*values[1]
                + metric[0]*values[2]
            ))
        if len(values) > 3:
            raise ValueError("synthetic contract supports s orders through two")
        return tuple(result)

    def coefficient_arrays(self):
        return {
            "contract_kind": np.asarray("product-normalized-wall"),
            "position_factor": np.asarray(1.0),
            "acceleration_factor": np.asarray(-0.7),
        }


class _ProductOuterOpenFaceContract:
    identifier = "synthetic-product-outer-open-v1"

    @staticmethod
    def _factor(state_name):
        return 0.2 if state_name == "position" else 0.3

    def r_first_z_jets(
        self, *, state_name, compact_coordinate, outer_value_z_jets,
    ):
        del compact_coordinate
        factor = self._factor(state_name)
        h_zz = NATIVE_CHANNEL_ORDER.index("h_zz")
        values = tuple(np.asarray(value) for value in outer_value_z_jets)
        multiplier = tuple(
            factor+0.05*value[:, h_zz][:, None] for value in values
        )
        # Only the order-zero multiplier contains the constant part.
        if len(multiplier) >= 2:
            multiplier = (
                multiplier[0],
                0.05*values[1][:, h_zz][:, None],
                *multiplier[2:],
            )
        if len(multiplier) >= 3:
            multiplier = (
                multiplier[0],
                multiplier[1],
                0.05*values[2][:, h_zz][:, None],
            )
        result = [-multiplier[0]*values[0]]
        if len(values) >= 2:
            result.append(-(
                multiplier[1]*values[0]+multiplier[0]*values[1]
            ))
        if len(values) >= 3:
            result.append(-(
                multiplier[2]*values[0]
                + 2.0*multiplier[1]*values[1]
                + multiplier[0]*values[2]
            ))
        if len(values) > 3:
            raise ValueError("synthetic contract supports z orders through two")
        return OuterOpenFaceDerivativeResult(
            tuple(result), np.ones(len(NATIVE_CHANNEL_ORDER), dtype=bool),
        )

    def coefficient_arrays(self):
        return {
            "contract_kind": np.asarray("product-outer-open"),
            "position_factor": np.asarray(0.2),
            "acceleration_factor": np.asarray(0.3),
            "metric_coupling": np.asarray(0.05),
            "ownership_mask": np.ones(len(NATIVE_CHANNEL_ORDER), dtype=bool),
        }


class _PhiOnlyOuterOpenFaceContract(_ProductOuterOpenFaceContract):
    identifier = "synthetic-phi-only-outer-open-v1"

    def r_first_z_jets(self, **kwargs):
        full = super().r_first_z_jets(**kwargs)
        mask = np.zeros(len(NATIVE_CHANNEL_ORDER), dtype=bool)
        mask[NATIVE_CHANNEL_ORDER.index("Phi")] = True
        return OuterOpenFaceDerivativeResult(full.r_first_z_jets, mask)

    def coefficient_arrays(self):
        record = dict(super().coefficient_arrays())
        mask = np.zeros(len(NATIVE_CHANNEL_ORDER), dtype=bool)
        mask[NATIVE_CHANNEL_ORDER.index("Phi")] = True
        record["contract_kind"] = np.asarray("product-outer-open-phi-only")
        record["ownership_mask"] = mask
        return record


class _PositionOnlyOuterOpenFaceContract:
    identifier = "synthetic-position-only-outer-open-v1"

    def r_first_z_jets(
        self, *, state_name, compact_coordinate, outer_value_z_jets,
    ):
        del compact_coordinate
        if state_name != "position":
            raise ValueError("position-only contract has no acceleration data")
        values = tuple(np.asarray(value) for value in outer_value_z_jets)
        return OuterOpenFaceDerivativeResult(
            tuple(-0.125*value for value in values),
            np.ones(len(NATIVE_CHANNEL_ORDER), dtype=bool),
        )

    def coefficient_arrays(self):
        return {
            "contract_kind": np.asarray("position-only-outer-open"),
            "position_factor": np.asarray(-0.125),
            "ownership_mask": np.ones(len(NATIVE_CHANNEL_ORDER), dtype=bool),
        }


class _ChangedCompactRecord(_ProductCompactWallContract):
    # Deliberately retain the same identifier while changing the full record.
    def coefficient_arrays(self):
        record = dict(super().coefficient_arrays())
        record["position_factor"] = np.asarray(1.01)
        return record


class _AxisBrokenCompactContract(_ProductCompactWallContract):
    identifier = "synthetic-axis-broken-wall-v1"

    def z_first_s_jets(self, *, state_name, radius, wall_value_s_jets):
        result = [
            np.asarray(value).copy()
            for value in super().z_first_s_jets(
                state_name=state_name,
                radius=radius,
                wall_value_s_jets=wall_value_s_jets,
            )
        ]
        axis = np.flatnonzero(np.asarray(radius) == 0.0)
        if len(axis):
            result[0][:, axis, NATIVE_CHANNEL_ORDER.index("h_rr")] += 1e-8
        return tuple(result)

    def coefficient_arrays(self):
        record = dict(super().coefficient_arrays())
        record["contract_kind"] = np.asarray("axis-broken-wall")
        return record


class _MaskDriftOuterContract(_ProductOuterOpenFaceContract):
    def __init__(self):
        self.drift = False

    def r_first_z_jets(self, **kwargs):
        result = super().r_first_z_jets(**kwargs)
        if not self.drift:
            return result
        mask = np.zeros(len(NATIVE_CHANNEL_ORDER), dtype=bool)
        mask[NATIVE_CHANNEL_ORDER.index("Phi")] = True
        return OuterOpenFaceDerivativeResult(result.r_first_z_jets, mask)


def _meshes(z, r):
    return np.meshgrid(
        np.asarray(z, dtype=float),
        (np.asarray(r, dtype=float)/R_MAX)**2,
        indexing="ij",
    )


def _position_fields(z, r, z_order=0):
    zz, ss = _meshes(z, r)
    if z_order == 0:
        h_perp = 0.7+0.03*zz**5+0.04*zz**2*ss+0.02*ss**3
        q4 = 0.11+0.02*zz**4+0.03*zz*ss+0.015*ss**2
        fields = {
            "h00": -1.2+0.02*zz**5+0.01*zz*ss**2,
            "h_perp": h_perp,
            "h_rr": h_perp+R_MAX**2*ss*q4,
            "h_zz": 0.9+0.01*zz**4+0.02*zz**2*ss+0.01*ss**3,
            "Phi": 0.2+0.03*zz**5-0.02*zz*ss+0.01*ss**2,
            "chi": -0.1+0.04*zz**3+0.03*ss**3,
            "v_z": 0.05+0.02*zz**5+0.01*zz*ss**2,
            "v_0": -0.02+0.01*zz**4-0.005*zz**2*ss,
        }
    elif z_order == 1:
        h_perp = 0.15*zz**4+0.08*zz*ss
        q4 = 0.08*zz**3+0.03*ss
        fields = {
            "h00": 0.10*zz**4+0.01*ss**2,
            "h_perp": h_perp,
            "h_rr": h_perp+R_MAX**2*ss*q4,
            "h_zz": 0.04*zz**3+0.04*zz*ss,
            "Phi": 0.15*zz**4-0.02*ss,
            "chi": 0.12*zz**2,
            "v_z": 0.10*zz**4+0.01*ss**2,
            "v_0": 0.04*zz**3-0.01*zz*ss,
        }
    elif z_order == 2:
        h_perp = 0.60*zz**3+0.08*ss
        q4 = 0.24*zz**2
        fields = {
            "h00": 0.40*zz**3,
            "h_perp": h_perp,
            "h_rr": h_perp+R_MAX**2*ss*q4,
            "h_zz": 0.12*zz**2+0.04*ss,
            "Phi": 0.60*zz**3,
            "chi": 0.24*zz,
            "v_z": 0.40*zz**3,
            "v_0": 0.12*zz**2-0.01*ss,
        }
    else:
        raise ValueError("unsupported analytic derivative")
    return fields


def _position_q4(z, r, z_order=0):
    zz, ss = _meshes(z, r)
    if z_order == 0:
        return 0.11+0.02*zz**4+0.03*zz*ss+0.015*ss**2
    if z_order == 1:
        return 0.08*zz**3+0.03*ss
    if z_order == 2:
        return 0.24*zz**2
    raise ValueError("unsupported analytic derivative")


def _acceleration_fields(z, r, z_order=0):
    zz, ss = _meshes(z, r)
    if z_order == 0:
        h_perp = -0.2+0.015*zz**5-0.01*zz**2*ss+0.02*ss**3
        q4 = 0.07-0.01*zz**4+0.02*zz*ss+0.005*ss**2
        fields = {
            "h00": 0.08-0.02*zz**5+0.006*zz*ss**2,
            "h_perp": h_perp,
            "h_rr": h_perp+R_MAX**2*ss*q4,
            "h_zz": -0.03+0.012*zz**5+0.02*zz*ss+0.01*ss**3,
            "Phi": 0.04-0.01*zz**4+0.008*ss**2,
            "chi": 0.03+0.013*zz**3-0.004*ss**3,
            "v_z": -0.02+0.005*zz**4+0.003*zz*ss,
            "v_0": 0.01-0.004*zz**5+0.002*zz**2*ss,
        }
    elif z_order == 1:
        h_perp = 0.075*zz**4-0.02*zz*ss
        q4 = -0.04*zz**3+0.02*ss
        fields = {
            "h00": -0.10*zz**4+0.006*ss**2,
            "h_perp": h_perp,
            "h_rr": h_perp+R_MAX**2*ss*q4,
            "h_zz": 0.06*zz**4+0.02*ss,
            "Phi": -0.04*zz**3,
            "chi": 0.039*zz**2,
            "v_z": 0.02*zz**3+0.003*ss,
            "v_0": -0.02*zz**4+0.004*zz*ss,
        }
    else:
        raise ValueError("unsupported analytic derivative")
    return fields


def _acceleration_q4(z, r, z_order=0):
    zz, ss = _meshes(z, r)
    if z_order == 0:
        return 0.07-0.01*zz**4+0.02*zz*ss+0.005*ss**2
    if z_order == 1:
        return -0.04*zz**3+0.02*ss
    raise ValueError("unsupported analytic derivative")


def _endpoint_fields(function, z, r):
    endpoints = function(np.asarray(z)[[0, -1]], r, z_order=1)
    return {name: value.copy() for name, value in endpoints.items()}


def _contract_endpoint_fields(fields, contract, state_name, r):
    stacked = np.stack(
        [fields[name][[0, -1]] for name in NATIVE_CHANNEL_ORDER],
        axis=-1,
    )
    endpoint = contract.z_first_s_jets(
        state_name=state_name,
        radius=np.asarray(r),
        wall_value_s_jets=(stacked,),
    )[0]
    return {
        name: endpoint[:, :, index].copy()
        for index, name in enumerate(NATIVE_CHANNEL_ORDER)
    }


def _synthetic_pair():
    z = np.linspace(1.0, 2.0, 11)
    r = np.linspace(0.0, R_MAX, 13)
    position = _position_fields(z, r)
    acceleration = _acceleration_fields(z, r)
    pair = NativeMetricHermitePair.build(
        z,
        r,
        position,
        acceleration,
        _endpoint_fields(_position_fields, z, r),
        _endpoint_fields(_acceleration_fields, z, r),
    )
    return z, r, position, acceleration, pair


def _synthetic_constrained_representation(outer=None):
    z = np.linspace(1.0, 2.0, 11)
    r = np.linspace(0.0, R_MAX, 13)
    position = _position_fields(z, r)
    acceleration = _acceleration_fields(z, r)
    compact = _ProductCompactWallContract()
    outer = _ProductOuterOpenFaceContract() if outer is None else outer
    position_z = _contract_endpoint_fields(
        position, compact, "position", r,
    )
    acceleration_z = _contract_endpoint_fields(
        acceleration, compact, "acceleration", r,
    )
    constrained = RadialFirstConstrainedHermiteRepresentation.build(
        z,
        r,
        position,
        acceleration,
        position_z,
        acceleration_z,
        compact_wall_contract=compact,
        outer_open_face_contract=outer,
    )
    additive = NativeMetricHermitePair.build(
        z,
        r,
        position,
        acceleration,
        position_z,
        acceleration_z,
    )
    return z, r, position, acceleration, constrained, additive


def test_q53_reconstructs_native_metric_and_analytic_q4_axis_limits():
    _, _, _, _, pair = _synthetic_pair()
    target_z = np.linspace(1.0, 2.0, 17)
    target_r = np.linspace(0.0, 11.5, 20)
    primary = pair.primary
    found = primary.position.evaluate_physical_channels(target_z, target_r)
    expected = _position_fields(target_z, target_r)
    for index, name in enumerate(NATIVE_CHANNEL_ORDER):
        assert np.max(np.abs(found[:, :, index]-expected[name])) < 2e-10

    for z_order in (0, 1, 2):
        found_q4 = primary.position.evaluate_q4(
            target_z, target_r, z_order=z_order,
        )
        expected_q4 = _position_q4(target_z, target_r, z_order=z_order)
        assert np.max(np.abs(found_q4-expected_q4)) < 3e-10

    found_q4_r = primary.position.evaluate_q4(
        target_z, target_r, r_order=1,
    )
    zz, ss = _meshes(target_z, target_r)
    expected_q4_r = (0.03*zz+0.03*ss)*(2.0*target_r[None, :]/R_MAX**2)
    assert np.max(np.abs(found_q4_r-expected_q4_r)) < 3e-10
    found_q4_rr = primary.position.evaluate_q4(
        target_z, target_r, r_order=2,
    )
    expected_q4_rr = (
        0.03*(2.0*target_r[None, :]/R_MAX**2)**2
        +(0.03*zz+0.03*ss)*(2.0/R_MAX**2)
    )
    assert np.max(np.abs(found_q4_rr-expected_q4_rr)) < 2e-9
    assert np.max(np.abs(
        found_q4_rr[:, 0]-0.06*target_z/R_MAX**2
    )) < 2e-11

    found_acceleration_q4 = primary.acceleration.evaluate_q4(
        target_z, target_r,
    )
    assert np.max(np.abs(
        found_acceleration_q4-_acceleration_q4(target_z, target_r)
    )) < 3e-10
    assert primary.position.z_degree == 5
    assert primary.position.s_degree == 3
    assert "q4" not in primary.channel_order


def test_q53_and_q33_share_row_derived_position_and_acceleration_endpoints():
    z, _, _, _, pair = _synthetic_pair()
    wall_z = z[[0, -1]]
    dense_r = np.linspace(0.0, R_MAX, 31)
    for state_name, analytic in (
        ("position", _position_fields),
        ("acceleration", _acceleration_fields),
    ):
        primary_state = pair.primary.state(state_name)
        comparator_state = pair.comparator.state(state_name)
        primary_value = primary_state.evaluate_physical_channels(wall_z, dense_r)
        comparator_value = comparator_state.evaluate_physical_channels(
            wall_z, dense_r,
        )
        primary_z = primary_state.evaluate_physical_channels(
            wall_z, dense_r, z_order=1,
        )
        comparator_z = comparator_state.evaluate_physical_channels(
            wall_z, dense_r, z_order=1,
        )
        expected_z = analytic(wall_z, dense_r, z_order=1)
        assert np.max(np.abs(primary_value-comparator_value)) < 3e-12
        assert np.max(np.abs(primary_z-comparator_z)) < 3e-12
        for index, name in enumerate(NATIVE_CHANNEL_ORDER):
            assert np.max(np.abs(primary_z[:, :, index]-expected_z[name])) < 3e-11
        assert np.max(np.abs(
            primary_state.evaluate_q4(wall_z, dense_r, z_order=1)
            - comparator_state.evaluate_q4(wall_z, dense_r, z_order=1)
        )) < 3e-12

    interior_z = np.array([1.37])
    primary_h00 = pair.primary.position.evaluate_physical_channels(
        interior_z, dense_r,
    )[:, :, 0]
    comparator_h00 = pair.comparator.position.evaluate_physical_channels(
        interior_z, dense_r,
    )[:, :, 0]
    assert np.max(np.abs(primary_h00-comparator_h00)) > 1e-7
    assert pair.primary.position.channels.z_boundary == "clamped_row_derived_z_first"
    assert pair.primary.acceleration.channels.z_boundary == "clamped_row_derived_z_first"
    assert pair.comparator.position.channels.z_boundary == "clamped_row_derived_z_first"
    assert pair.primary.endpoint_fingerprint == pair.comparator.endpoint_fingerprint


def test_regular_vector_coefficients_supply_physical_radius_factors():
    _, _, _, _, pair = _synthetic_pair()
    target_z = np.array([1.15, 1.63])
    target_r = np.array([0.0, 0.7, 3.2])
    reduced = pair.primary.position.evaluate_reduced(target_z, target_r)
    coordinate = pair.primary.position.evaluate_coordinate_components(
        target_z, target_r,
    )
    fields = _position_fields(target_z, target_r)
    assert np.max(np.abs(reduced[:, :, 1]-fields["v_z"])) < 2e-11
    assert np.max(np.abs(reduced[:, :, 5]-fields["v_0"])) < 2e-11
    h_zr = COORDINATE_COMPONENT_ORDER.index("h_zr")
    h_0r = COORDINATE_COMPONENT_ORDER.index("h_0r")
    assert np.max(np.abs(
        coordinate[:, :, h_zr]-fields["v_z"]*target_r[None, :]
    )) < 2e-11
    assert np.max(np.abs(
        coordinate[:, :, h_0r]-fields["v_0"]*target_r[None, :]
    )) < 2e-11
    assert np.array_equal(coordinate[:, 0, h_zr], np.zeros(len(target_z)))
    assert np.array_equal(coordinate[:, 0, h_0r], np.zeros(len(target_z)))
    first_r = pair.primary.position.evaluate_coordinate_components(
        target_z, np.array([0.0]), r_order=1,
    )
    axis_fields = _position_fields(target_z, np.array([0.0]))
    assert np.max(np.abs(first_r[:, 0, h_zr]-axis_fields["v_z"][:, 0])) < 2e-11
    assert np.max(np.abs(first_r[:, 0, h_0r]-axis_fields["v_0"][:, 0])) < 2e-11
    assert np.array_equal(reduced[:, :, 0], np.zeros(reduced.shape[:2]))


def test_representation_rejects_q4_input_and_nonregular_axis_data():
    z = np.linspace(1.0, 2.0, 11)
    r = np.linspace(0.0, R_MAX, 13)
    position = _position_fields(z, r)
    acceleration = _acceleration_fields(z, r)
    position_z = _endpoint_fields(_position_fields, z, r)
    acceleration_z = _endpoint_fields(_acceleration_fields, z, r)

    with_q4 = {**position, "q4": np.zeros((len(z), len(r)))}
    with np.testing.assert_raises_regex(ValueError, "extra=.*q4"):
        NativeMetricHermitePair.build(
            z, r, with_q4, acceleration, position_z, acceleration_z,
        )

    bad_axis = {name: value.copy() for name, value in position.items()}
    bad_axis["h_rr"][4, 0] += 1e-8
    with np.testing.assert_raises_regex(ValueError, "agree exactly at the axis"):
        NativeMetricHermitePair.build(
            z, r, bad_axis, acceleration, position_z, acceleration_z,
        )

    bad_endpoint = {name: value.copy() for name, value in acceleration_z.items()}
    bad_endpoint["h_rr"][1, 0] += 1e-8
    with np.testing.assert_raises_regex(ValueError, "derivatives must agree exactly"):
        NativeMetricHermitePair.build(
            z, r, position, acceleration, position_z, bad_endpoint,
        )


def test_representation_is_deterministic_and_owns_immutable_copies():
    z, r, position, acceleration, first = _synthetic_pair()
    second = NativeMetricHermitePair.build(
        z,
        r,
        position,
        acceleration,
        _endpoint_fields(_position_fields, z, r),
        _endpoint_fields(_acceleration_fields, z, r),
    )
    assert first.primary.fingerprint() == second.primary.fingerprint()
    assert first.comparator.fingerprint() == second.comparator.fingerprint()
    original = first.primary.fingerprint()
    position["h00"][:] = 999.0
    acceleration["Phi"][:] = -999.0
    assert first.primary.fingerprint() == original
    assert not first.primary.position.channels.coefficients.flags.writeable
    assert not first.primary.position.z_first_endpoints.flags.writeable
    assert np.array_equal(
        first.primary.position.z_first_endpoints,
        first.comparator.position.z_first_endpoints,
    )
    restored = NativeMetricHermitePair.from_arrays(first.coefficient_arrays())
    assert restored.primary.fingerprint() == first.primary.fingerprint()
    assert restored.comparator.fingerprint() == first.comparator.fingerprint()
    assert np.array_equal(
        restored.primary.position.z_first_endpoints,
        first.primary.position.z_first_endpoints,
    )


def test_public_position_only_builder_has_no_acceleration_dependency():
    z = np.linspace(1.0, 2.0, 11)
    r = np.linspace(0.0, R_MAX, 13)
    position = _position_fields(z, r)
    compact = _ProductCompactWallContract()
    outer = _PositionOnlyOuterOpenFaceContract()
    endpoints = _contract_endpoint_fields(
        position, compact, "position", r,
    )
    state = RadialFirstConstrainedHermiteState.build_position(
        z,
        r,
        position,
        endpoints,
        compact_wall_contract=compact,
        outer_open_face_contract=outer,
    )
    expected = np.stack(
        [position[name] for name in NATIVE_CHANNEL_ORDER], axis=-1,
    )
    found = state.evaluate_physical_channels(z, r)
    assert np.max(np.abs(found-expected)) < 1e-12
    expected_endpoint = np.stack(
        [endpoints[name] for name in NATIVE_CHANNEL_ORDER], axis=-1,
    )
    found_endpoint = state.evaluate_physical_channels(
        z[[0, -1]], r, z_order=1,
    )
    assert np.max(np.abs(found_endpoint-expected_endpoint)) < 1e-12
    h_perp = NATIVE_CHANNEL_ORDER.index("h_perp")
    h_rr = NATIVE_CHANNEL_ORDER.index("h_rr")
    source_numerator = expected[:, :, h_rr]-expected[:, :, h_perp]
    source_s = (r/R_MAX)**2
    native_axis_q4 = (
        derivative_matrix(source_s, 1, 7) @ source_numerator.T
    )[0]/R_MAX**2
    represented_axis_q4 = state.evaluate_q4(z, np.asarray([0.0]))[:, 0]
    assert np.max(np.abs(native_axis_q4-represented_axis_q4)) < 1e-12
    assert np.max(np.abs(
        state.radial_anisotropy_numerator.inner_s_derivative[:, 0]
        / R_MAX**2-represented_axis_q4
    )) < 1e-12
    assert state.state_name == "position"
    assert len(state.compact_wall_contract_fingerprint) == 64
    assert len(state.outer_open_face_contract_fingerprint) == 64
    assert "acceleration" not in " ".join(
        name for name, _ in state.outer_open_face_contract_record
    ).lower()
    with np.testing.assert_raises_regex(ValueError, "no acceleration data"):
        outer.r_first_z_jets(
            state_name="acceleration",
            compact_coordinate=z,
            outer_value_z_jets=(expected[:, -1],),
        )

    probe_z = np.asarray([z[0], 1.37, z[-1]])
    probe_r = np.asarray([0.0, 1.7, 7.3, R_MAX])
    for r_order in (0, 1, 2):
        numerator = state.evaluate_anisotropy_numerator(
            probe_z, probe_r, r_order=r_order,
        )
        channels = state.evaluate_physical_channels(
            probe_z, probe_r, r_order=r_order,
        )
        assert np.max(np.abs(
            numerator-(channels[:, :, h_rr]-channels[:, :, h_perp])
        )) < 3e-12
    axis_numerator = state.evaluate_anisotropy_numerator(
        probe_z, np.asarray([0.0]),
    )
    assert np.array_equal(axis_numerator, np.zeros_like(axis_numerator))
    assert not np.any(np.signbit(axis_numerator))
    assert np.array_equal(
        state.evaluate_q4(probe_z, np.asarray([0.0]), r_order=1),
        np.zeros((len(probe_z), 1)),
    )

    archive = state.coefficient_arrays("position_only")
    restored = RadialFirstConstrainedHermiteState.from_arrays(
        archive,
        "position_only",
        compact_wall_contract=_ProductCompactWallContract(),
        outer_open_face_contract=_PositionOnlyOuterOpenFaceContract(),
    )
    assert restored.fingerprint() == state.fingerprint()
    assert (
        restored.compact_wall_contract_fingerprint
        == state.compact_wall_contract_fingerprint
    )
    assert (
        restored.outer_open_face_contract_fingerprint
        == state.outer_open_face_contract_fingerprint
    )
    for name in archive:
        assert np.array_equal(
            restored.coefficient_arrays("position_only")[name], archive[name],
        )


def test_radial_first_contract_preserves_nonlinear_wall_rows_on_dense_radii():
    z, r, position, _, constrained, additive = (
        _synthetic_constrained_representation()
    )
    source_found = constrained.position.evaluate_physical_channels(z, r)
    source_expected = np.stack(
        [position[name] for name in NATIVE_CHANNEL_ORDER], axis=-1,
    )
    assert np.max(np.abs(source_found-source_expected)) < 3e-11

    wall_z = z[[0, -1]]
    dense_r = np.linspace(0.0, R_MAX, 47)
    values = constrained.position.evaluate_physical_channels(wall_z, dense_r)
    z_first = constrained.position.evaluate_physical_channels(
        wall_z, dense_r, z_order=1,
    )
    h_zz = NATIVE_CHANNEL_ORDER.index("h_zz")
    signs = np.array([-1.0, 1.0])[:, None, None]
    residual = z_first-signs*values[:, :, h_zz][:, :, None]*values
    assert np.max(np.abs(residual)) < 2e-11

    additive_values = additive.primary.position.evaluate_physical_channels(
        wall_z, dense_r,
    )
    additive_z = additive.primary.position.evaluate_physical_channels(
        wall_z, dense_r, z_order=1,
    )
    additive_residual = (
        additive_z
        - signs*additive_values[:, :, h_zz][:, :, None]*additive_values
    )
    assert np.max(np.abs(additive_residual)) > 1e-7
    assert constrained.position.compact_wall_contract_id == (
        "synthetic-product-normalized-wall-v1"
    )
    rebuilt = _synthetic_constrained_representation()[4]
    assert constrained.fingerprint() == rebuilt.fingerprint()
    assert not constrained.position.radial_channels.coefficients.flags.writeable
    assert not constrained.position.radial_channels.outer_s_derivative.flags.writeable

    archive = constrained.coefficient_arrays()
    restored = RadialFirstConstrainedHermiteRepresentation.from_arrays(
        archive,
        compact_wall_contract=_ProductCompactWallContract(),
        outer_open_face_contract=_ProductOuterOpenFaceContract(),
    )
    assert restored.fingerprint() == constrained.fingerprint()
    restored_archive = restored.coefficient_arrays()
    assert set(restored_archive) == set(archive)
    for name in archive:
        assert np.array_equal(restored_archive[name], archive[name])
    reload_z = np.asarray([z[0], 1.37, z[-1]])
    reload_r = np.asarray([0.0, 2.3, 8.4, R_MAX])
    for state_name in ("position", "acceleration"):
        original_state = constrained.state(state_name)
        restored_state = restored.state(state_name)
        for z_order, r_order in ((0, 0), (1, 0), (0, 1), (0, 2)):
            assert np.array_equal(
                original_state.evaluate_reduced(
                    reload_z, reload_r, z_order=z_order, r_order=r_order,
                ),
                restored_state.evaluate_reduced(
                    reload_z, reload_r, z_order=z_order, r_order=r_order,
                ),
            )

    wrong_contract = _ProductCompactWallContract()
    wrong_contract.identifier = "wrong-compact-contract"
    with np.testing.assert_raises_regex(ValueError, "identifier differs"):
        RadialFirstConstrainedHermiteRepresentation.from_arrays(
            archive,
            compact_wall_contract=wrong_contract,
            outer_open_face_contract=_ProductOuterOpenFaceContract(),
        )

    with np.testing.assert_raises_regex(ValueError, "contract record differs"):
        RadialFirstConstrainedHermiteRepresentation.from_arrays(
            archive,
            compact_wall_contract=_ChangedCompactRecord(),
            outer_open_face_contract=_ProductOuterOpenFaceContract(),
        )

    tampered = {name: np.asarray(value).copy() for name, value in archive.items()}
    record_prefix = (
        "radial_first_constrained_position_compact_wall_contract_record"
    )
    record_names = [str(value) for value in tampered[f"{record_prefix}_names"]]
    factor_index = record_names.index("position_factor")
    tampered[f"{record_prefix}_value_{factor_index}"] = np.asarray(9.0)
    with np.testing.assert_raises_regex(ValueError, "fingerprint is invalid"):
        RadialFirstConstrainedHermiteRepresentation.from_arrays(
            tampered,
            compact_wall_contract=_ProductCompactWallContract(),
            outer_open_face_contract=_ProductOuterOpenFaceContract(),
        )

    probe_z = np.asarray([z[0], 1.41, z[-1]])
    probe_r = np.asarray([0.8, 4.3, 9.1])
    step = 1e-5
    plus = constrained.position.evaluate_physical_channels(
        probe_z, probe_r+step,
    )
    minus = constrained.position.evaluate_physical_channels(
        probe_z, probe_r-step,
    )
    finite_difference = (plus-minus)/(2.0*step)
    analytic = constrained.position.evaluate_physical_channels(
        probe_z, probe_r, r_order=1,
    )
    scale = np.maximum(1.0, np.maximum(np.abs(finite_difference), np.abs(analytic)))
    assert np.max(np.abs(finite_difference-analytic)/scale) < 2e-8


def test_outer_contract_owns_only_the_open_radial_face():
    z, source_r, position, _, constrained, _ = (
        _synthetic_constrained_representation()
    )
    query_z = np.concatenate(([z[0]], np.linspace(z[0], z[-1], 9)[1:-1], [z[-1]]))
    outer_r = np.asarray([R_MAX])
    state = constrained.position
    values = state.evaluate_physical_channels(query_z, outer_r)[:, 0]
    radial = state.evaluate_physical_channels(
        query_z, outer_r, r_order=1,
    )[:, 0]
    h_zz = NATIVE_CHANNEL_ORDER.index("h_zz")
    expected = -(0.2+0.05*values[:, h_zz][:, None])*values
    diagnostic = state.outer_open_face_residual(query_z)
    assert np.max(np.abs(
        diagnostic["residual"]-(radial-expected)
    )) < 2e-12
    assert diagnostic["maximum_normalized"] > 1e-7
    assert np.isfinite(diagnostic["maximum_normalized"])
    assert state.outer_open_face_contract_id == "synthetic-product-outer-open-v1"

    source_diagnostic = state.outer_open_face_residual(z)
    assert source_diagnostic["maximum_normalized"] < 3e-12
    source_values = state.evaluate_physical_channels(z, outer_r)[:, 0]
    source_target = -(
        0.2+0.05*source_values[:, h_zz][:, None]
    )*source_values
    outer_s_difference = (
        state.radial_channels.outer_s_derivative
        - 0.5*R_MAX*source_target
    )
    assert np.max(np.abs(outer_s_difference[1:-1])) < 3e-11
    assert np.max(np.abs(outer_s_difference[[0, -1]])) > 1e-5
    source_values = np.stack(
        [position[name] for name in NATIVE_CHANNEL_ORDER], axis=-1,
    )
    baseline = make_interp_spline(
        (source_r/R_MAX)**2, source_values, k=3, axis=1,
    )(1.0, nu=1)
    assert np.max(np.abs(
        state.radial_channels.outer_s_derivative[[0, -1]]
        - baseline[[0, -1]]
    )) < 3e-12
    assert np.array_equal(
        state.outer_ownership_mask,
        np.ones(len(NATIVE_CHANNEL_ORDER), dtype=bool),
    )

    approach_errors = []
    for distance in (1e-2, 5e-3, 2.5e-3):
        near = state.evaluate_physical_channels(
            query_z, np.asarray([R_MAX-distance]), r_order=1,
        )[:, 0]
        approach_errors.append(float(np.max(np.abs(near-radial))))
    assert approach_errors[2] < approach_errors[1] < approach_errors[0]
    assert np.all(state.radial_channels.s_knots[:4] == 0.0)
    assert np.all(state.radial_channels.s_knots[-4:] == 1.0)

    q4_r = state.evaluate_q4(query_z, outer_r, r_order=1)[:, 0]
    h_perp = NATIVE_CHANNEL_ORDER.index("h_perp")
    h_rr = NATIVE_CHANNEL_ORDER.index("h_rr")
    expected_n = values[:, h_rr]-values[:, h_perp]
    expected_nr = radial[:, h_rr]-radial[:, h_perp]
    expected_q4_r = expected_nr/R_MAX**2-2.0*expected_n/R_MAX**3
    assert np.max(np.abs(q4_r-expected_q4_r)) < 2e-12

    _, _, _, _, phi_only, _ = _synthetic_constrained_representation(
        _PhiOnlyOuterOpenFaceContract(),
    )
    partial_state = phi_only.position
    partial_values = partial_state.evaluate_physical_channels(
        query_z, outer_r,
    )[:, 0]
    partial_radial = partial_state.evaluate_physical_channels(
        query_z, outer_r, r_order=1,
    )[:, 0]
    partial_raw = partial_state._evaluate_channel_s(
        query_z, outer_r, z_order=0, s_order=1,
    )[:, 0]*(2.0/R_MAX)
    phi = NATIVE_CHANNEL_ORDER.index("Phi")
    h00 = NATIVE_CHANNEL_ORDER.index("h00")
    partial_target = -(
        0.2+0.05*partial_values[:, h_zz]
    )*partial_values[:, phi]
    partial_diagnostic = partial_state.outer_open_face_residual(query_z)
    assert np.array_equal(
        np.flatnonzero(partial_diagnostic["ownership_mask"]),
        np.asarray([phi]),
    )
    assert np.max(np.abs(
        partial_diagnostic["residual"][:, phi]
        -(partial_radial[:, phi]-partial_target)
    )) < 2e-12
    assert np.max(np.abs(
        partial_radial[:, h00]-partial_raw[:, h00]
    )) < 2e-12


def test_live_contract_masks_and_anisotropy_axis_fail_closed():
    z = np.linspace(1.0, 2.0, 11)
    r = np.linspace(0.0, R_MAX, 13)
    position = _position_fields(z, r)
    acceleration = _acceleration_fields(z, r)
    good_compact = _ProductCompactWallContract()
    position_z = _contract_endpoint_fields(
        position, good_compact, "position", r,
    )
    acceleration_z = _contract_endpoint_fields(
        acceleration, good_compact, "acceleration", r,
    )

    with np.testing.assert_raises_regex(ValueError, "nonregular at the axis"):
        RadialFirstConstrainedHermiteRepresentation.build(
            z,
            r,
            position,
            acceleration,
            position_z,
            acceleration_z,
            compact_wall_contract=_AxisBrokenCompactContract(),
            outer_open_face_contract=_ProductOuterOpenFaceContract(),
        )

    drifting_outer = _MaskDriftOuterContract()
    constrained = RadialFirstConstrainedHermiteRepresentation.build(
        z,
        r,
        position,
        acceleration,
        position_z,
        acceleration_z,
        compact_wall_contract=good_compact,
        outer_open_face_contract=drifting_outer,
    )
    drifting_outer.drift = True
    with np.testing.assert_raises_regex(ValueError, "ownership mask changed"):
        constrained.position.outer_open_face_residual(
            np.linspace(z[0], z[-1], 17),
        )

    clean = _synthetic_constrained_representation()[4]
    poisoned = {
        name: np.asarray(value).copy()
        for name, value in clean.coefficient_arrays().items()
    }
    numerator_key = (
        "radial_first_constrained_position_"
        "radial_anisotropy_numerator_coefficients"
    )
    poisoned[numerator_key][:, 0, 0] += 1e-8
    with np.testing.assert_raises_regex(
        ValueError, "exact h_rr-h_perp coefficient difference",
    ):
        RadialFirstConstrainedHermiteRepresentation.from_arrays(
            poisoned,
            compact_wall_contract=_ProductCompactWallContract(),
            outer_open_face_contract=_ProductOuterOpenFaceContract(),
        )
    poisoned_inner = {
        name: np.asarray(value).copy()
        for name, value in clean.coefficient_arrays().items()
    }
    numerator_inner_key = (
        "radial_first_constrained_position_"
        "radial_anisotropy_numerator_inner_s_derivative"
    )
    poisoned_inner[numerator_inner_key][3, 0] += 1e-8
    with np.testing.assert_raises_regex(
        ValueError, "exact h_rr-h_perp coefficient difference",
    ):
        RadialFirstConstrainedHermiteRepresentation.from_arrays(
            poisoned_inner,
            compact_wall_contract=_ProductCompactWallContract(),
            outer_open_face_contract=_ProductOuterOpenFaceContract(),
        )
    poisoned_outer = {
        name: np.asarray(value).copy()
        for name, value in clean.coefficient_arrays().items()
    }
    numerator_outer_key = (
        "radial_first_constrained_position_"
        "radial_anisotropy_numerator_outer_s_derivative"
    )
    poisoned_outer[numerator_outer_key][3, 0] += 1e-8
    with np.testing.assert_raises_regex(
        ValueError, "exact h_rr-h_perp coefficient difference",
    ):
        RadialFirstConstrainedHermiteRepresentation.from_arrays(
            poisoned_outer,
            compact_wall_contract=_ProductCompactWallContract(),
            outer_open_face_contract=_ProductOuterOpenFaceContract(),
        )


def test_sealed_reduced_q33_q55_pair_remains_separate_and_available():
    z, r, _, _, pair = _synthetic_pair()
    reduced = pair.primary.position.evaluate_reduced(z, r)
    first = np.zeros((3, *reduced.shape))
    second = np.zeros((3, 3, *reduced.shape))
    first[1] = pair.primary.position.evaluate_reduced(z, r, z_order=1)
    first[2] = pair.primary.position.evaluate_reduced(z, r, r_order=1)
    second[0, 0] = pair.primary.acceleration.evaluate_reduced(z, r)
    second[1, 1] = pair.primary.position.evaluate_reduced(z, r, z_order=2)
    second[1, 2] = second[2, 1] = pair.primary.position.evaluate_reduced(
        z, r, z_order=1, r_order=1,
    )
    second[2, 2] = pair.primary.position.evaluate_reduced(z, r, r_order=2)
    jet = SimpleNamespace(
        z=z,
        r=r,
        reduced_fields=reduced,
        reduced_first=first,
        reduced_second=second,
    )
    found = sealed_reduced_q33_q55_adverse_projections(
        jet,
        z,
        r,
        np.linspace(z[0], z[-1], 15),
        np.linspace(0.0, 10.0, 17),
        parent_identity="synthetic-sealed-comparator",
    )
    assert tuple(found) == SEALED_ADVERSE_COMPARATOR_NAMES
    assert found[SEALED_ADVERSE_COMPARATOR_NAMES[0]].reduced_fields.shape == (
        15, 17, 9,
    )
    assert found[SEALED_ADVERSE_COMPARATOR_NAMES[1]].reduced_fields.shape == (
        15, 17, 9,
    )
    q33 = found[SEALED_ADVERSE_COMPARATOR_NAMES[0]]
    q55 = found[SEALED_ADVERSE_COMPARATOR_NAMES[1]]
    assert np.max(np.abs(q33.reduced_fields-q55.reduced_fields)) > 1e-8
    assert np.max(np.abs(
        q33.reduced_second[1, 1]-q55.reduced_second[1, 1]
    )) > 1e-6
    assert (
        SEALED_ADVERSE_COMPARATOR_RECIPES[
            SEALED_ADVERSE_COMPARATOR_NAMES[0]
        ]["compact_degree"]
        == 3
    )
    assert (
        SEALED_ADVERSE_COMPARATOR_RECIPES[
            SEALED_ADVERSE_COMPARATOR_NAMES[1]
        ]["compact_degree"]
        == 5
    )
    assert (
        SEALED_ADVERSE_COMPARATOR_RECIPES[
            SEALED_ADVERSE_COMPARATOR_NAMES[0]
        ]["position_velocity_boundary"]
        == "stored-z-first-clamped"
    )
    assert (
        SEALED_ADVERSE_COMPARATOR_RECIPES[
            SEALED_ADVERSE_COMPARATOR_NAMES[1]
        ]["position_velocity_boundary"]
        == "not-a-knot"
    )
    assert pair.comparison_name == "native-metric-identical-endpoint-Q53-Q33"


def test_radial_first_q33_comparator_shares_contracts_and_outer_bundle():
    z = np.linspace(1.0, 2.0, 11)
    r = np.linspace(0.0, R_MAX, 13)
    position = _position_fields(z, r)
    acceleration = _acceleration_fields(z, r)
    compact = _ProductCompactWallContract()
    outer = _ProductOuterOpenFaceContract()
    pair = RadialFirstConstrainedHermitePair.build(
        z,
        r,
        position,
        acceleration,
        _contract_endpoint_fields(position, compact, "position", r),
        _contract_endpoint_fields(acceleration, compact, "acceleration", r),
        compact_wall_contract=compact,
        outer_open_face_contract=outer,
    )
    assert pair.primary.position.z_degree == 5
    assert pair.comparator.position.z_degree == 3
    assert pair.comparison_name == "radial-first-identical-contract-Q53-Q33"
    assert np.array_equal(
        pair.primary.position.radial_channels.coefficients,
        pair.comparator.position.radial_channels.coefficients,
    )
    assert np.array_equal(
        pair.primary.position.radial_channels.inner_s_derivative,
        pair.comparator.position.radial_channels.inner_s_derivative,
    )
    assert np.array_equal(
        pair.primary.position.radial_channels.outer_s_derivative,
        pair.comparator.position.radial_channels.outer_s_derivative,
    )
    assert np.array_equal(
        pair.primary.position.radial_anisotropy_numerator.coefficients,
        pair.comparator.position.radial_anisotropy_numerator.coefficients,
    )
    assert np.array_equal(
        pair.primary.position.radial_anisotropy_numerator.inner_s_derivative,
        pair.comparator.position.radial_anisotropy_numerator.inner_s_derivative,
    )
    assert np.array_equal(
        pair.primary.position.radial_anisotropy_numerator.outer_s_derivative,
        pair.comparator.position.radial_anisotropy_numerator.outer_s_derivative,
    )

    wall_z = z[[0, -1]]
    dense_r = np.linspace(0.0, R_MAX, 37)
    for state_name in ("position", "acceleration"):
        primary = pair.primary.state(state_name)
        comparator = pair.comparator.state(state_name)
        assert np.max(np.abs(
            primary.evaluate_physical_channels(wall_z, dense_r)
            - comparator.evaluate_physical_channels(wall_z, dense_r)
        )) < 3e-12
        assert np.max(np.abs(
            primary.evaluate_physical_channels(
                wall_z, dense_r, z_order=1,
            )
            - comparator.evaluate_physical_channels(
                wall_z, dense_r, z_order=1,
            )
        )) < 3e-12

    interior_z = np.asarray([1.37])
    primary = pair.primary.position.evaluate_physical_channels(
        interior_z, dense_r,
    )
    comparator = pair.comparator.position.evaluate_physical_channels(
        interior_z, dense_r,
    )
    assert np.max(np.abs(primary-comparator)) > 1e-7
    assert np.max(np.abs(
        pair.primary.position.evaluate_anisotropy_numerator(
            wall_z, dense_r, r_order=2,
        )
        - pair.comparator.position.evaluate_anisotropy_numerator(
            wall_z, dense_r, r_order=2,
        )
    )) < 3e-12

    archive = pair.coefficient_arrays()
    restored = RadialFirstConstrainedHermitePair.from_arrays(
        archive,
        compact_wall_contract=_ProductCompactWallContract(),
        outer_open_face_contract=_ProductOuterOpenFaceContract(),
    )
    assert np.array_equal(
        restored.comparator.position.radial_channels.coefficients,
        pair.comparator.position.radial_channels.coefficients,
    )
    assert restored.comparator.fingerprint() == pair.comparator.fingerprint()


def _poison_fresh_spline_coefficients(monkeypatch, module, call_number):
    original = module.make_interp_spline
    calls = {"count": 0}

    def poisoned(*args, **kwargs):
        spline = original(*args, **kwargs)
        calls["count"] += 1
        if calls["count"] == call_number:
            coefficients = np.asarray(spline.c)
            coefficients.reshape(-1)[0] = np.nan
        return spline

    monkeypatch.setattr(module, "make_interp_spline", poisoned)
    return calls


def _mutable_representation_failure_record(evidence):
    return {
        name: (
            np.asarray(value).copy()
            if isinstance(value, np.ndarray)
            else value
        )
        for name, value in evidence.items()
    }


def test_fresh_radial_nonfinite_coefficients_are_typed_and_evidence_is_sealed(
    monkeypatch,
):
    z = np.linspace(1.0, 2.0, 11)
    r = np.linspace(0.0, R_MAX, 13)
    fields = _position_fields(z, r)
    values = np.stack(
        [fields[name] for name in NATIVE_CHANNEL_ORDER], axis=-1,
    )
    inner = np.zeros((len(z), len(NATIVE_CHANNEL_ORDER)))
    outer = np.zeros_like(inner)
    calls = _poison_fresh_spline_coefficients(
        monkeypatch, representation_module, 1,
    )

    with pytest.raises(
        Protocol125RepresentationCoefficientFailure,
    ) as caught:
        representation_module._RadialCubicChannels.build(
            r,
            values,
            inner,
            outer,
            parent_r_max=R_MAX,
        )

    assert calls["count"] == 1
    evidence = caught.value.evidence
    assert evidence["recipe"] == "native-radial-cubic-s"
    assert evidence["nonfinite_count"] == 1
    once = validate_protocol125_representation_coefficient_failure(evidence)
    twice = validate_protocol125_representation_coefficient_failure(once)
    assert tuple(once) == tuple(twice)
    assert once["fingerprint"] == twice["fingerprint"]
    assert np.array_equal(
        once["coefficient_raw_bytes"],
        twice["coefficient_raw_bytes"],
    )
    assert not once["coefficient_raw_bytes"].flags.writeable
    with pytest.raises(TypeError):
        once["recipe"] = "native-tensor-Q53-compact"
    with pytest.raises(ValueError, match="read-only"):
        once["coefficient_raw_bytes"][0] = 0


def test_fresh_native_nonfinite_coefficients_raise_typed_failure(monkeypatch):
    z = np.linspace(1.0, 2.0, 11)
    r = np.linspace(0.0, R_MAX, 13)
    fields = _position_fields(z, r)
    endpoints = _endpoint_fields(_position_fields, z, r)
    values = np.stack(
        [fields[name] for name in NATIVE_CHANNEL_ORDER], axis=-1,
    )
    endpoint_values = np.stack(
        [endpoints[name] for name in NATIVE_CHANNEL_ORDER], axis=-1,
    )
    calls = _poison_fresh_spline_coefficients(
        monkeypatch, representation_module, 4,
    )

    with pytest.raises(
        Protocol125RepresentationCoefficientFailure,
    ) as caught:
        NativeTensorHermiteSurface.build(
            z,
            r,
            values,
            endpoint_values,
            z_degree=5,
            parent_r_max=R_MAX,
        )

    assert calls["count"] == 4
    evidence = caught.value.evidence
    assert evidence["recipe"] == "native-tensor-Q53-compact"
    assert evidence["coefficient_shape"][-1] == len(NATIVE_CHANNEL_ORDER)
    assert evidence["nonfinite_count"] == 1


def test_representation_failure_evidence_rejects_tampered_fractional_and_impossible_records(
    monkeypatch,
):
    z = np.linspace(1.0, 2.0, 11)
    r = np.linspace(0.0, R_MAX, 13)
    fields = _position_fields(z, r)
    values = np.stack(
        [fields[name] for name in NATIVE_CHANNEL_ORDER], axis=-1,
    )
    endpoint_fields = _endpoint_fields(_position_fields, z, r)
    endpoints = np.stack(
        [endpoint_fields[name] for name in NATIVE_CHANNEL_ORDER], axis=-1,
    )
    _poison_fresh_spline_coefficients(
        monkeypatch, representation_module, 4,
    )
    with pytest.raises(
        Protocol125RepresentationCoefficientFailure,
    ) as caught:
        NativeTensorHermiteSurface.build(
            z,
            r,
            values,
            endpoints,
            z_degree=5,
            parent_r_max=R_MAX,
        )
    evidence = caught.value.evidence

    tampered = _mutable_representation_failure_record(evidence)
    tampered["input_sha256"] = "0"*64
    with pytest.raises(ValueError, match="fingerprint differs"):
        validate_protocol125_representation_coefficient_failure(tampered)

    fractional = _mutable_representation_failure_record(evidence)
    fractional["nonfinite_count"] = 1.0
    with pytest.raises(ValueError, match="not nonfinite"):
        validate_protocol125_representation_coefficient_failure(fractional)

    impossible_shape = _mutable_representation_failure_record(evidence)
    shape = list(impossible_shape["coefficient_shape"])
    shape[0] += 1
    impossible_shape["coefficient_shape"] = tuple(shape)
    with pytest.raises(ValueError, match="payload is invalid"):
        validate_protocol125_representation_coefficient_failure(
            impossible_shape,
        )

    impossible_count = _mutable_representation_failure_record(evidence)
    impossible_count["nonfinite_count"] = int(
        np.prod(impossible_count["coefficient_shape"]),
    )+1
    with pytest.raises(ValueError, match="not nonfinite"):
        validate_protocol125_representation_coefficient_failure(
            impossible_count,
        )


def test_persisted_native_and_radial_nonfinite_coefficients_remain_invalid():
    native = _synthetic_pair()[-1]
    native_archive = {
        name: np.asarray(value).copy()
        for name, value in native.coefficient_arrays().items()
    }
    native_archive[
        "native_metric_hermite_primary_position_channels_coefficients"
    ][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="coefficients must be finite"):
        NativeMetricHermitePair.from_arrays(native_archive)

    constrained = _synthetic_constrained_representation()[4]
    constrained_archive = {
        name: np.asarray(value).copy()
        for name, value in constrained.coefficient_arrays().items()
    }
    constrained_archive[
        "radial_first_constrained_position_radial_channels_coefficients"
    ][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="invalid radial-first interpolation data"):
        RadialFirstConstrainedHermiteRepresentation.from_arrays(
            constrained_archive,
            compact_wall_contract=_ProductCompactWallContract(),
            outer_open_face_contract=_ProductOuterOpenFaceContract(),
        )
