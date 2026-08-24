"""Boundary-derivative contracts for the joint native parent representation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.joint_parent_representation import (
    NATIVE_CHANNEL_ORDER,
    OuterOpenFaceDerivativeResult,
)


H00 = NATIVE_CHANNEL_ORDER.index("h00")
H_PERP = NATIVE_CHANNEL_ORDER.index("h_perp")
H_RR = NATIVE_CHANNEL_ORDER.index("h_rr")
H_ZZ = NATIVE_CHANNEL_ORDER.index("h_zz")
PHI = NATIVE_CHANNEL_ORDER.index("Phi")
CHI = NATIVE_CHANNEL_ORDER.index("chi")
V_Z = NATIVE_CHANNEL_ORDER.index("v_z")
V_0 = NATIVE_CHANNEL_ORDER.index("v_0")


def _immutable(value, dtype=float):
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _digest_arrays(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _jet_add(left, right):
    return tuple(np.asarray(x)+np.asarray(y) for x, y in zip(left, right))


def _jet_scale(value, factor):
    return tuple(float(factor)*np.asarray(item) for item in value)


def _jet_multiply(left, right):
    if len(left) != len(right) or not 1 <= len(left) <= 3:
        raise ValueError("boundary jets must have depth one through three")
    result = [left[0]*right[0]]
    if len(left) >= 2:
        result.append(left[1]*right[0]+left[0]*right[1])
    if len(left) >= 3:
        result.append(
            left[2]*right[0]+2.0*left[1]*right[1]+left[0]*right[2]
        )
    return tuple(result)


def _jet_sqrt(value):
    root = np.sqrt(value[0])
    result = [root]
    if len(value) >= 2:
        result.append(value[1]/(2.0*root))
    if len(value) >= 3:
        result.append(
            value[2]/(2.0*root)-value[1]**2/(4.0*root**3)
        )
    return tuple(result)


def _jet_reciprocal(value):
    inverse = 1.0/value[0]
    result = [inverse]
    if len(value) >= 2:
        result.append(-value[1]/value[0]**2)
    if len(value) >= 3:
        result.append(
            2.0*value[1]**2/value[0]**3-value[2]/value[0]**2
        )
    return tuple(result)


def _jet_constant_like(reference, value):
    result = [np.full_like(reference[0], float(value))]
    result.extend(np.zeros_like(reference[0]) for _ in range(1, len(reference)))
    return tuple(result)


@dataclass(frozen=True)
class _WallRadialBundle:
    """Two-wall cubic radial context, optionally outer clamped."""

    knots: np.ndarray
    coefficients: np.ndarray
    parent_r_max: float

    @classmethod
    def build(cls, r, values, *, outer_s_derivative=None):
        r = np.asarray(r, dtype=float)
        values = np.asarray(values, dtype=float)
        if values.ndim != 3 or values.shape[:2] != (2, len(r)):
            raise ValueError("wall radial context must have shape (2,nr,nfield)")
        parent_r_max = float(r[-1])
        s = (r/parent_r_max)**2
        if outer_s_derivative is None:
            spline = make_interp_spline(s, values, k=3, axis=1)
        else:
            outer = np.asarray(outer_s_derivative, dtype=float)
            if outer.shape != (2, values.shape[-1]):
                raise ValueError("wall outer derivative bundle has the wrong shape")
            knots = np.concatenate((
                np.repeat(s[0], 4), s[2:-1], np.repeat(s[-1], 4),
            ))
            spline = make_interp_spline(
                s,
                values,
                k=3,
                axis=1,
                t=knots,
                bc_type=(None, [(1, outer)]),
            )
        coefficients = np.moveaxis(spline.c, 0, 1)
        return cls(
            _immutable(spline.t), _immutable(coefficients), parent_r_max,
        )

    def jets(self, radius, depth):
        radius = np.atleast_1d(np.asarray(radius, dtype=float))
        if not 1 <= int(depth) <= 3:
            raise ValueError("wall context derivative depth must be one through three")
        s = (radius/self.parent_r_max)**2
        spline = BSpline(self.knots, self.coefficients, 3, axis=1)
        return tuple(spline(s, nu=order) for order in range(int(depth)))


@dataclass(frozen=True)
class NativeNormalizedCompactWallContract:
    """Exact position and time-symmetric acceleration compact-wall rows.

    The position state is evaluated from the supplied query-time values.  The
    acceleration state additionally uses a radial representation of the
    already completed position plus the owned normal-source traces.
    Derivatives in squared radius are propagated analytically through every
    nonlinear product and normalization.
    """

    background_values: tuple
    position_context: _WallRadialBundle
    source_normal_context: _WallRadialBundle
    source_second_normal_context: _WallRadialBundle | None
    identifier: str

    @property
    def position_ownership_mask(self):
        """All native compact position rows are owned by this contract."""
        return _immutable(
            np.ones(len(NATIVE_CHANNEL_ORDER), dtype=bool), bool,
        )

    @property
    def acceleration_ownership_mask(self):
        """All native compact acceleration rows are owned once appended."""
        if self.source_second_normal_context is None:
            raise ValueError(
                "position-only compact contract has no acceleration ownership"
            )
        return _immutable(
            np.ones(len(NATIVE_CHANNEL_ORDER), dtype=bool), bool,
        )

    @classmethod
    def build(
        cls,
        r,
        background,
        position_wall_values,
        source_normal_wall,
        source_second_normal_wall,
        *,
        position_outer_s_derivative=None,
    ):
        names = (
            "wall_stiffness", "v0", "v1", "beta_a", "beta_b",
            "wall_potential_a", "wall_potential_b",
        )
        values = tuple(float(background[name]) for name in names)
        position = np.asarray(position_wall_values, dtype=float)
        source = np.asarray(source_normal_wall, dtype=float)
        source_second = np.asarray(source_second_normal_wall, dtype=float)
        if position.shape != (2, len(r), len(NATIVE_CHANNEL_ORDER)):
            raise ValueError("position wall context has the wrong shape")
        if source.shape != (2, len(r)) or source_second.shape != source.shape:
            raise ValueError("normal-source wall context has the wrong shape")
        digest = _digest_arrays(
            np.asarray(values), np.asarray(r), position, source, source_second,
            np.asarray([] if position_outer_s_derivative is None else position_outer_s_derivative),
        )
        return cls(
            values,
            _WallRadialBundle.build(
                r,
                position,
                outer_s_derivative=position_outer_s_derivative,
            ),
            _WallRadialBundle.build(r, source[:, :, None]),
            _WallRadialBundle.build(r, source_second[:, :, None]),
            f"native-normalized-compact-wall-v1:{digest}",
        )

    @classmethod
    def build_position(
        cls,
        r,
        background,
        position_wall_values,
        source_normal_wall,
        *,
        position_outer_s_derivative=None,
    ):
        """Build the position contract without placeholder acceleration data."""
        names = (
            "wall_stiffness", "v0", "v1", "beta_a", "beta_b",
            "wall_potential_a", "wall_potential_b",
        )
        values = tuple(float(background[name]) for name in names)
        position = np.asarray(position_wall_values, dtype=float)
        source = np.asarray(source_normal_wall, dtype=float)
        if position.shape != (2, len(r), len(NATIVE_CHANNEL_ORDER)):
            raise ValueError("position wall context has the wrong shape")
        if source.shape != (2, len(r)):
            raise ValueError("normal-source position context has the wrong shape")
        digest = _digest_arrays(
            np.asarray("position-only"),
            np.asarray(values),
            np.asarray(r),
            position,
            source,
            np.asarray(
                [] if position_outer_s_derivative is None
                else position_outer_s_derivative
            ),
        )
        return cls(
            values,
            _WallRadialBundle.build(
                r,
                position,
                outer_s_derivative=position_outer_s_derivative,
            ),
            _WallRadialBundle.build(r, source[:, :, None]),
            None,
            f"native-normalized-compact-wall-position-v1:{digest}",
        )

    def append_acceleration(self, r, source_second_normal_wall):
        """Append the acceleration context to one position-only contract.

        Protocol 125 treats the transition to the shared two-state contract as
        append-only.  In particular, the already sealed position and normal
        source radial bundles may not be reconstructed, even through the same
        spline factory.  This method therefore reuses those exact immutable
        bundle objects and constructs only the new ``H_z,tt`` child.
        """
        if self.source_second_normal_context is not None:
            raise ValueError("compact-wall contract already contains acceleration")
        r = np.asarray(r, dtype=float)
        source_second = np.asarray(source_second_normal_wall, dtype=float)
        if (
            r.ndim != 1
            or len(r) < 4
            or r[0] != 0.0
            or np.signbit(r[0])
            or np.any(np.diff(r) <= 0.0)
            or not np.all(np.isfinite(r))
            or r[-1] != self.position_context.parent_r_max
            or r[-1] != self.source_normal_context.parent_r_max
            or source_second.shape != (2, len(r))
            or not np.all(np.isfinite(source_second))
        ):
            raise ValueError("invalid append-only acceleration wall context")
        child = _WallRadialBundle.build(r, source_second[:, :, None])
        digest = _digest_arrays(
            np.asarray("append-acceleration-v1"),
            np.asarray(self.identifier),
            np.asarray(r),
            source_second,
            child.knots,
            child.coefficients,
            np.asarray(child.parent_r_max),
        )
        return NativeNormalizedCompactWallContract(
            self.background_values,
            self.position_context,
            self.source_normal_context,
            child,
            f"native-normalized-compact-wall-shared-v1:{digest}",
        )

    def _background(self):
        names = (
            "wall_stiffness", "v0", "v1", "beta_a", "beta_b",
            "wall_potential_a", "wall_potential_b",
        )
        return dict(zip(names, self.background_values))

    def coefficient_arrays(self):
        """Return the complete immutable record consumed by the representation."""
        result = {
            "background_values": np.asarray(self.background_values),
            "position_knots": self.position_context.knots.copy(),
            "position_coefficients": self.position_context.coefficients.copy(),
            "position_parent_r_max": np.asarray(
                self.position_context.parent_r_max
            ),
            "source_normal_knots": self.source_normal_context.knots.copy(),
            "source_normal_coefficients": (
                self.source_normal_context.coefficients.copy()
            ),
            "source_normal_parent_r_max": np.asarray(
                self.source_normal_context.parent_r_max
            ),
            "position_only": np.asarray(
                self.source_second_normal_context is None
            ),
            "position_ownership_mask": self.position_ownership_mask.copy(),
        }
        if self.source_second_normal_context is not None:
            result.update({
                "source_second_normal_knots": (
                    self.source_second_normal_context.knots.copy()
                ),
                "source_second_normal_coefficients": (
                    self.source_second_normal_context.coefficients.copy()
                ),
                "source_second_normal_parent_r_max": np.asarray(
                    self.source_second_normal_context.parent_r_max
                ),
                "acceleration_ownership_mask": (
                    self.acceleration_ownership_mask.copy()
                ),
            })
        return result

    def _beta_jets(self, phi):
        background = self._background()
        gamma = background["wall_stiffness"]
        target = np.asarray((background["v0"], background["v1"]))[:, None]
        base = np.asarray((background["beta_a"], background["beta_b"]))[:, None]
        wall_potential = np.asarray((
            background["wall_potential_a"],
            background["wall_potential_b"],
        ))[:, None]
        branch = np.asarray((1.0, -1.0))[:, None]
        delta = list(phi)
        delta[0] = delta[0]-target
        delta = tuple(delta)
        potential = _jet_scale(_jet_multiply(delta, delta), 0.5*gamma)
        beta = _jet_add(
            _jet_constant_like(delta, 0.0),
            _jet_scale(potential, 1.0/6.0),
        )
        beta = tuple(
            base + branch*(value-wall_potential)/6.0
            if order == 0 else branch*value/6.0
            for order, value in enumerate(_jet_scale(
                _jet_multiply(delta, delta), 0.5*gamma,
            ))
        )
        beta_phi = tuple(
            branch*gamma*value/6.0 for value in delta
        )
        return beta, beta_phi, delta

    def _position_derivatives(self, values, source_normal):
        depth = len(values)
        phi = tuple(item[:, :, PHI] for item in values)
        normal_metric = tuple(item[:, :, H_ZZ] for item in values)
        beta, _, delta = self._beta_jets(phi)
        A = _jet_sqrt(normal_metric)
        beta_A = _jet_multiply(beta, A)
        result = [np.zeros_like(item) for item in values]
        for channel in (H00, H_PERP, H_RR, V_0):
            field = tuple(item[:, :, channel] for item in values)
            derivative = _jet_scale(_jet_multiply(beta_A, field), -2.0)
            for order in range(depth):
                result[order][:, :, channel] = derivative[order]
        G_three_half = _jet_multiply(normal_metric, A)
        normal_derivative = _jet_add(
            _jet_scale(_jet_multiply(beta, G_three_half), -8.0),
            _jet_scale(_jet_multiply(source_normal, normal_metric), 2.0),
        )
        orientation = np.asarray((-1.0, 1.0))[:, None]
        phi_derivative = _jet_scale(
            _jet_multiply(delta, A), -0.5*self._background()["wall_stiffness"],
        )
        for order in range(depth):
            result[order][:, :, H_ZZ] = normal_derivative[order]
            result[order][:, :, PHI] = orientation*phi_derivative[order]
            result[order][:, :, CHI] = 0.0
            result[order][:, :, V_Z] = 0.0
        return tuple(result)

    def z_first_s_jets(self, *, state_name, radius, wall_value_s_jets):
        values = tuple(np.asarray(item, dtype=float) for item in wall_value_s_jets)
        if not 1 <= len(values) <= 3:
            raise ValueError("compact-wall contract supports s orders zero through two")
        expected = (2, len(np.atleast_1d(radius)), len(NATIVE_CHANNEL_ORDER))
        if any(item.shape != expected for item in values):
            raise ValueError("compact-wall query data have the wrong shape")
        source = tuple(
            item[:, :, 0]
            for item in self.source_normal_context.jets(radius, len(values))
        )
        if str(state_name) == "position":
            return self._position_derivatives(values, source)
        if str(state_name) != "acceleration":
            raise ValueError("state_name must be position or acceleration")
        if self.source_second_normal_context is None:
            raise ValueError(
                "position-only compact contract has no acceleration data"
            )

        position = self.position_context.jets(radius, len(values))
        source_second = tuple(
            item[:, :, 0]
            for item in self.source_second_normal_context.jets(radius, len(values))
        )
        position_z = self._position_derivatives(position, source)
        phi = tuple(item[:, :, PHI] for item in position)
        G = tuple(item[:, :, H_ZZ] for item in position)
        A = _jet_sqrt(G)
        inverse_G = _jet_reciprocal(G)
        beta, beta_phi, delta = self._beta_jets(phi)
        a_phi = tuple(item[:, :, PHI] for item in values)
        a_G = tuple(item[:, :, H_ZZ] for item in values)
        result = [np.zeros_like(item) for item in values]

        for channel in (H00, H_PERP, H_RR, V_0):
            f = tuple(item[:, :, channel] for item in position)
            f_z = tuple(item[:, :, channel] for item in position_z)
            a_f = tuple(item[:, :, channel] for item in values)
            normalization = _jet_scale(
                _jet_multiply(_jet_multiply(f_z, a_G), inverse_G), 0.5,
            )
            phi_coupling = _jet_scale(
                _jet_multiply(
                    _jet_multiply(_jet_multiply(A, beta_phi), a_phi), f,
                ),
                -2.0,
            )
            field_term = _jet_scale(
                _jet_multiply(_jet_multiply(A, beta), a_f), -2.0,
            )
            derivative = _jet_add(_jet_add(normalization, phi_coupling), field_term)
            for order in range(len(values)):
                result[order][:, :, channel] = derivative[order]

        coefficient = _jet_add(
            _jet_scale(_jet_multiply(beta, A), 12.0),
            _jet_scale(source, -2.0),
        )
        G_three_half = _jet_multiply(G, A)
        normal_derivative = _jet_add(
            _jet_add(
                _jet_scale(_jet_multiply(coefficient, a_G), -1.0),
                _jet_scale(
                    _jet_multiply(_jet_multiply(beta_phi, a_phi), G_three_half),
                    -8.0,
                ),
            ),
            _jet_scale(_jet_multiply(source_second, G), 2.0),
        )
        A_tt = _jet_scale(_jet_multiply(a_G, _jet_reciprocal(A)), 0.5)
        phi_derivative = _jet_scale(
            _jet_add(
                _jet_multiply(delta, A_tt),
                _jet_multiply(a_phi, A),
            ),
            -0.5*self._background()["wall_stiffness"],
        )
        orientation = np.asarray((-1.0, 1.0))[:, None]
        for order in range(len(values)):
            result[order][:, :, H_ZZ] = normal_derivative[order]
            result[order][:, :, PHI] = orientation*phi_derivative[order]
            result[order][:, :, CHI] = 0.0
            result[order][:, :, V_Z] = 0.0
        return tuple(result)


@dataclass(frozen=True)
class StoredOuterOpenFaceDerivativeContract:
    """Immutable full outer derivative bundle with explicit owned channels."""

    source_z: np.ndarray
    position_r_first: np.ndarray
    acceleration_r_first: np.ndarray
    position_ownership: np.ndarray
    acceleration_ownership: np.ndarray
    identifier: str

    @classmethod
    def build(
        cls,
        source_z,
        position_r_first,
        acceleration_r_first,
        *,
        position_ownership=None,
        acceleration_ownership=None,
    ):
        z = np.asarray(source_z, dtype=float)
        position = np.asarray(position_r_first, dtype=float)
        acceleration = np.asarray(acceleration_r_first, dtype=float)
        expected = (len(z), len(NATIVE_CHANNEL_ORDER))
        if position.shape != expected or acceleration.shape != expected:
            raise ValueError("stored outer derivative arrays have the wrong shape")
        if np.any(np.diff(z) <= 0.0) or not all(np.all(np.isfinite(item)) for item in (
            z, position, acceleration,
        )):
            raise ValueError("stored outer derivative contract is invalid")
        default = np.ones(len(NATIVE_CHANNEL_ORDER), dtype=bool)
        position_mask = default if position_ownership is None else np.asarray(
            position_ownership, dtype=bool,
        )
        acceleration_mask = default if acceleration_ownership is None else np.asarray(
            acceleration_ownership, dtype=bool,
        )
        if position_mask.shape != default.shape or acceleration_mask.shape != default.shape:
            raise ValueError("outer derivative ownership mask has the wrong shape")
        digest = _digest_arrays(z, position, acceleration, position_mask, acceleration_mask)
        return cls(
            _immutable(z),
            _immutable(position),
            _immutable(acceleration),
            _immutable(position_mask, bool),
            _immutable(acceleration_mask, bool),
            f"stored-outer-open-face-v1:{digest}",
        )

    def r_first_z_jets(self, *, state_name, compact_coordinate, outer_value_z_jets):
        z = np.atleast_1d(np.asarray(compact_coordinate, dtype=float))
        depth = len(tuple(outer_value_z_jets))
        if not 1 <= depth <= 3:
            raise ValueError("outer contract supports z orders zero through two")
        if str(state_name) == "position":
            values = self.position_r_first
            ownership = self.position_ownership
        elif str(state_name) == "acceleration":
            values = self.acceleration_r_first
            ownership = self.acceleration_ownership
        else:
            raise ValueError("state_name must be position or acceleration")
        degree = min(5, len(self.source_z)-1)
        spline = make_interp_spline(self.source_z, values, k=degree, axis=0)
        returned = tuple(spline(z, nu=order) for order in range(depth))
        return OuterOpenFaceDerivativeResult(returned, ownership)

    def coefficient_arrays(self):
        """Return the complete frozen bundle and ownership masks."""
        return {
            "source_z": self.source_z.copy(),
            "position_r_first": self.position_r_first.copy(),
            "acceleration_r_first": self.acceleration_r_first.copy(),
            "position_ownership": self.position_ownership.copy(),
            "acceleration_ownership": self.acceleration_ownership.copy(),
        }


_PROTOCOL125_PRIMITIVE_KEYS = (
    "selector_q",
    "psi",
    "alpha",
    "alpha_r",
)
_PROTOCOL125_REFERENCE_OUTER_KEYS = (
    "reference_q",
    "reference_q_r",
    "reference_phi",
    "reference_phi_r",
)
_PROTOCOL125_SHAPE_KEYS = ("a", "b", "c", "a_r", "b_r", "c_r")
_PROTOCOL125_SCALAR_KEYS = ("chi", "chi_r")


def _stack_exact_vector_mapping(mapping, keys, length, label):
    if not isinstance(mapping, Mapping) or set(mapping) != set(keys):
        raise ValueError(f"{label} must contain exactly {keys}")
    arrays = []
    for key in keys:
        value = np.asarray(mapping[key], dtype=float)
        if value.shape != (int(length),) or not np.all(np.isfinite(value)):
            raise ValueError(f"{label} entry {key} is invalid")
        arrays.append(value)
    return np.stack(arrays, axis=-1)


def _stack_outer_native_fields(fields, length):
    return _stack_exact_vector_mapping(
        fields, NATIVE_CHANNEL_ORDER, length, "completed outer native fields",
    )


def _scaled_linf(left, right, mask):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    if left.shape != right.shape or left.shape[0] != len(mask):
        raise ValueError("outer-map comparison shapes differ")
    selected_left = left[mask]
    selected_right = right[mask]
    if not selected_left.size:
        return 0.0
    scale = np.maximum.reduce((
        np.ones_like(selected_left),
        np.abs(selected_left),
        np.abs(selected_right),
    ))
    return float(np.max(np.abs(selected_left-selected_right)/scale))


def _derive_protocol125_position_outer_arrays(
    source_z,
    parent_r_max,
    source_position,
    primitive_values,
    reference_outer_values,
    shape_values,
    scalar_values,
    validation_tolerance,
):
    """Return the frozen native outer-position derivative bundle."""
    z = np.asarray(source_z, dtype=float)
    position = np.asarray(source_position, dtype=float)
    primitives = np.asarray(primitive_values, dtype=float)
    reference_outer = np.asarray(reference_outer_values, dtype=float)
    shape = np.asarray(shape_values, dtype=float)
    scalar = np.asarray(scalar_values, dtype=float)
    radius = float(parent_r_max)
    tolerance = float(validation_tolerance)
    expected_position = (len(z), len(NATIVE_CHANNEL_ORDER))
    if (
        z.ndim != 1
        or len(z) < 6
        or np.any(np.diff(z) <= 0.0)
        or not np.all(np.isfinite(z))
        or position.shape != expected_position
        or primitives.shape != (len(z), len(_PROTOCOL125_PRIMITIVE_KEYS))
        or reference_outer.shape
        != (len(z), len(_PROTOCOL125_REFERENCE_OUTER_KEYS))
        or shape.shape != (len(z), len(_PROTOCOL125_SHAPE_KEYS))
        or scalar.shape != (len(z), len(_PROTOCOL125_SCALAR_KEYS))
        or radius <= 0.0
        or not np.isfinite(radius)
        or tolerance <= 0.0
        or tolerance > 1e-12
        or not np.isfinite(tolerance)
        or not all(np.all(np.isfinite(item)) for item in (
            position, primitives, reference_outer, shape, scalar,
        ))
    ):
        raise ValueError("invalid Protocol-125 outer-position derivation data")

    open_compact = np.zeros(len(z), dtype=bool)
    open_compact[1:-1] = True
    pindex = {
        name: index for index, name in enumerate(_PROTOCOL125_PRIMITIVE_KEYS)
    }
    rindex = {
        name: index
        for index, name in enumerate(_PROTOCOL125_REFERENCE_OUTER_KEYS)
    }
    sindex = {name: index for index, name in enumerate(_PROTOCOL125_SHAPE_KEYS)}
    cindex = {name: index for index, name in enumerate(_PROTOCOL125_SCALAR_KEYS)}
    selector_q = primitives[:, pindex["selector_q"]]
    reference_q = reference_outer[:, rindex["reference_q"]]
    reference_q_r = reference_outer[:, rindex["reference_q_r"]]
    reference_phi = reference_outer[:, rindex["reference_phi"]]
    reference_phi_r = reference_outer[:, rindex["reference_phi_r"]]
    psi = primitives[:, pindex["psi"]]
    alpha = primitives[:, pindex["alpha"]]
    alpha_r = primitives[:, pindex["alpha_r"]]
    a = shape[:, sindex["a"]]
    b = shape[:, sindex["b"]]
    c = shape[:, sindex["c"]]
    a_r = shape[:, sindex["a_r"]]
    b_r = shape[:, sindex["b_r"]]
    c_r = shape[:, sindex["c_r"]]
    chi = scalar[:, cindex["chi"]]
    chi_r = scalar[:, cindex["chi_r"]]

    if (
        np.any(z+selector_q <= 0.0)
        or np.any(psi <= 0.0)
        or np.any(alpha <= 0.0)
    ):
        raise ValueError("outer primitive lapse and conformal factor must be positive")
    h00 = position[:, H00]
    h_perp = position[:, H_PERP]
    h_rr = position[:, H_RR]
    h_zz = position[:, H_ZZ]
    phi = position[:, PHI]
    if (
        np.any(h00 >= 0.0)
        or np.any(h_perp <= 0.0)
        or np.any(h_rr <= 0.0)
        or np.any(h_zz <= 0.0)
    ):
        raise ValueError("completed outer native metric has invalid signs")
    for channel in (V_Z, V_0):
        value = position[:, channel]
        if np.any(value != 0.0) or np.any(np.signbit(value)):
            raise ValueError(
                "Protocol-125 outer position vector coefficients must be "
                "IEEE positive zero"
            )

    reconstructed = {
        "selector conformal factor": 1.0/(z+selector_q),
        "completed lapse": np.sqrt(-h00),
        "h_perp shape map": psi**2*np.exp(2.0*c),
        "h_rr shape map": psi**2*np.exp(2.0*b),
        "h_zz shape map": psi**2*np.exp(2.0*a),
        "collapse scalar map": chi,
        "trace-free shape": np.zeros_like(a),
    }
    represented = {
        "selector conformal factor": psi,
        "completed lapse": alpha,
        "h_perp shape map": h_perp,
        "h_rr shape map": h_rr,
        "h_zz shape map": h_zz,
        "collapse scalar map": position[:, CHI],
        "trace-free shape": a+b+2.0*c,
    }
    for label in reconstructed:
        mismatch = _scaled_linf(
            reconstructed[label], represented[label], open_compact,
        )
        if mismatch > tolerance:
            raise ValueError(
                f"Protocol-125 outer {label} mismatch exceeds validation tolerance"
            )

    # The solved row is the balanced delta-Robin condition, not a Robin
    # condition on the candidate alone.
    q_r = reference_q_r-(selector_q-reference_q)/radius
    phi_r = reference_phi_r-(phi-reference_phi)/radius
    psi_r = -psi**2*q_r
    logarithmic_psi_r = psi_r/psi
    derivative = np.zeros_like(position)
    derivative[:, H00] = -2.0*alpha*alpha_r
    derivative[:, H_PERP] = 2.0*h_perp*(logarithmic_psi_r+c_r)
    derivative[:, H_RR] = 2.0*h_rr*(logarithmic_psi_r+b_r)
    derivative[:, H_ZZ] = 2.0*h_zz*(logarithmic_psi_r+a_r)
    derivative[:, PHI] = phi_r
    derivative[:, CHI] = chi_r
    # These are physical time-symmetric/parity zeros, validated above rather
    # than generic filler lanes.
    derivative[:, V_Z] = 0.0
    derivative[:, V_0] = 0.0
    return derivative, open_compact


class Protocol125PositionOuterOpenFaceDerivativeContract:
    """Public marker for canonical Protocol-125 position outer contracts.

    Direct construction is forbidden because the numerical record contains
    derivative primitives, including ``alpha_r``.  The sole production
    constructor is ``derive_joint_parent_position_outer_contract(parent)``.
    """

    def __new__(cls, *args, **kwargs):
        if cls is Protocol125PositionOuterOpenFaceDerivativeContract:
            raise TypeError(
                "Protocol-125 position outer contracts must be derived from "
                "the complete parent"
            )
        return super().__new__(cls)


@dataclass(frozen=True)
class _Protocol125PositionOuterOpenFaceDerivativeContract(
    Protocol125PositionOuterOpenFaceDerivativeContract,
):
    """Acceleration-free frozen outer bundle for the Protocol-125 position.

    The selector-q and Phi derivatives come from their native balanced
    delta-Robin rows, including the frozen seven-point derivatives of the
    fresh reference state.
    Lapse, spatial-metric, and collapse-scalar derivatives then follow from
    the supplied completed primitive, analytic shape, and analytic scalar
    maps.  Compact-wall-owned endpoint rows are recorded as excluded and are
    never outer-clamp owners.
    """

    source_z: np.ndarray
    source_r: np.ndarray
    parent_r_max: float
    source_position: np.ndarray
    primitive_values: np.ndarray
    reference_outer_values: np.ndarray
    source_reference_fingerprint: str
    shape_values: np.ndarray
    scalar_values: np.ndarray
    position_r_first: np.ndarray
    open_compact_mask: np.ndarray
    ownership_mask: np.ndarray
    validation_tolerance: float
    identifier: str

    def __post_init__(self):
        z = _immutable(self.source_z)
        r = _immutable(self.source_r)
        position = _immutable(self.source_position)
        primitives = _immutable(self.primitive_values)
        reference_outer = _immutable(self.reference_outer_values)
        reference_fingerprint = str(self.source_reference_fingerprint)
        shape = _immutable(self.shape_values)
        scalar = _immutable(self.scalar_values)
        derivative = _immutable(self.position_r_first)
        open_mask = _immutable(self.open_compact_mask, bool)
        ownership = _immutable(self.ownership_mask, bool)
        tolerance = float(self.validation_tolerance)
        radius = float(self.parent_r_max)
        if (
            r.ndim != 1
            or len(r) < 7
            or r[0] != 0.0
            or np.any(np.diff(r) <= 0.0)
            or not np.all(np.isfinite(r))
            or r[-1] != radius
        ):
            raise ValueError("Protocol-125 outer reference radial grid is invalid")
        if (
            len(reference_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in reference_fingerprint)
        ):
            raise ValueError("Protocol-125 source/reference fingerprint is invalid")
        expected_derivative, expected_open = (
            _derive_protocol125_position_outer_arrays(
                z,
                radius,
                position,
                primitives,
                reference_outer,
                shape,
                scalar,
                tolerance,
            )
        )
        if not np.array_equal(open_mask, expected_open):
            raise ValueError("Protocol-125 outer compact ownership rows differ")
        if not np.array_equal(
            ownership, np.ones(len(NATIVE_CHANNEL_ORDER), dtype=bool),
        ):
            raise ValueError("Protocol-125 outer position must derive every channel")
        if not (
            derivative.shape == expected_derivative.shape
            and derivative.tobytes()
            == np.ascontiguousarray(expected_derivative).tobytes()
        ):
            raise ValueError(
                "Protocol-125 outer derivative is not the exact derived bundle"
            )
        digest = _digest_arrays(
            np.asarray("protocol-125-position-outer-v3"),
            z,
            r,
            np.asarray(radius),
            position,
            primitives,
            reference_outer,
            np.asarray(reference_fingerprint),
            shape,
            scalar,
            derivative,
            open_mask,
            ownership,
            np.asarray(tolerance),
        )
        expected_identifier = f"protocol-125-position-outer-v3:{digest}"
        if str(self.identifier) != expected_identifier:
            raise ValueError("Protocol-125 outer derivative identifier is invalid")
        object.__setattr__(self, "source_z", z)
        object.__setattr__(self, "source_r", r)
        object.__setattr__(self, "parent_r_max", radius)
        object.__setattr__(self, "source_position", position)
        object.__setattr__(self, "primitive_values", primitives)
        object.__setattr__(self, "reference_outer_values", reference_outer)
        object.__setattr__(
            self, "source_reference_fingerprint", reference_fingerprint,
        )
        object.__setattr__(self, "shape_values", shape)
        object.__setattr__(self, "scalar_values", scalar)
        object.__setattr__(self, "position_r_first", derivative)
        object.__setattr__(self, "open_compact_mask", open_mask)
        object.__setattr__(self, "ownership_mask", ownership)
        object.__setattr__(self, "validation_tolerance", tolerance)
        object.__setattr__(self, "identifier", expected_identifier)

    @classmethod
    def _from_completed_primitives(
        cls,
        source_z,
        source_r,
        completed_position_fields,
        *,
        completed_primitives,
        reference_q,
        reference_phi,
        shape_map,
        scalar_map,
        validation_tolerance=1e-12,
    ):
        """Derive the bundle from full fresh reference arrays.

        The two reference radial derivatives are always computed here with
        the frozen width-seven outer row.  There is no public parameter by
        which a production caller can supply or zero those derivatives.
        """
        z = np.asarray(source_z, dtype=float)
        r = np.asarray(source_r, dtype=float)
        reference_q = np.asarray(reference_q, dtype=float)
        reference_phi = np.asarray(reference_phi, dtype=float)
        if (
            r.ndim != 1
            or len(r) < 7
            or r[0] != 0.0
            or np.any(np.diff(r) <= 0.0)
            or not np.all(np.isfinite(r))
            or reference_q.shape != (len(z), len(r))
            or reference_phi.shape != reference_q.shape
            or not np.all(np.isfinite(reference_q))
            or not np.all(np.isfinite(reference_phi))
        ):
            raise ValueError("invalid fresh outer-reference arrays")
        parent_r_max = float(r[-1])
        position = _stack_outer_native_fields(
            completed_position_fields, len(z),
        )
        primitives = _stack_exact_vector_mapping(
            completed_primitives,
            _PROTOCOL125_PRIMITIVE_KEYS,
            len(z),
            "completed outer primitives",
        )
        radial_operator = derivative_matrix(r, 1, 7)
        reference_q_r = (radial_operator @ reference_q.T).T[:, -1]
        reference_phi_r = (radial_operator @ reference_phi.T).T[:, -1]
        reference_outer = np.stack((
            reference_q[:, -1],
            reference_q_r,
            reference_phi[:, -1],
            reference_phi_r,
        ), axis=-1)
        source_reference_fingerprint = _digest_arrays(
            np.asarray("fresh-reference-width-seven-outer-v1"),
            z,
            r,
            reference_q,
            reference_phi,
        )
        shape = _stack_exact_vector_mapping(
            shape_map,
            _PROTOCOL125_SHAPE_KEYS,
            len(z),
            "analytic outer shape map",
        )
        scalar = _stack_exact_vector_mapping(
            scalar_map,
            _PROTOCOL125_SCALAR_KEYS,
            len(z),
            "analytic outer scalar map",
        )
        derivative, open_mask = _derive_protocol125_position_outer_arrays(
            z,
            parent_r_max,
            position,
            primitives,
            reference_outer,
            shape,
            scalar,
            validation_tolerance,
        )
        ownership = np.ones(len(NATIVE_CHANNEL_ORDER), dtype=bool)
        digest = _digest_arrays(
            np.asarray("protocol-125-position-outer-v3"),
            z,
            r,
            np.asarray(float(parent_r_max)),
            position,
            primitives,
            reference_outer,
            np.asarray(source_reference_fingerprint),
            shape,
            scalar,
            derivative,
            open_mask,
            ownership,
            np.asarray(float(validation_tolerance)),
        )
        return cls(
            z,
            r,
            float(parent_r_max),
            position,
            primitives,
            reference_outer,
            source_reference_fingerprint,
            shape,
            scalar,
            derivative,
            open_mask,
            ownership,
            float(validation_tolerance),
            f"protocol-125-position-outer-v3:{digest}",
        )

    def r_first_z_jets(
        self, *, state_name, compact_coordinate, outer_value_z_jets,
    ):
        if str(state_name) != "position":
            raise ValueError(
                "Protocol-125 position outer contract has no acceleration bundle"
            )
        z = np.atleast_1d(np.asarray(compact_coordinate, dtype=float))
        values = tuple(np.asarray(value, dtype=float) for value in outer_value_z_jets)
        if (
            z.ndim != 1
            or np.any(~np.isfinite(z))
            or np.any(z < self.source_z[0])
            or np.any(z > self.source_z[-1])
            or not 1 <= len(values) <= 3
            or any(
                value.shape != (len(z), len(NATIVE_CHANNEL_ORDER))
                or not np.all(np.isfinite(value))
                for value in values
            )
        ):
            raise ValueError("invalid Protocol-125 outer position query")
        spline = make_interp_spline(
            self.source_z,
            self.position_r_first,
            k=min(5, len(self.source_z)-1),
            axis=0,
        )
        returned = tuple(spline(z, nu=order) for order in range(len(values)))
        return OuterOpenFaceDerivativeResult(returned, self.ownership_mask)

    def coefficient_arrays(self):
        return {
            "source_z": self.source_z.copy(),
            "source_r": self.source_r.copy(),
            "parent_r_max": np.asarray(self.parent_r_max),
            "source_position": self.source_position.copy(),
            "primitive_keys": np.asarray(_PROTOCOL125_PRIMITIVE_KEYS),
            "primitive_values": self.primitive_values.copy(),
            "reference_outer_keys": np.asarray(
                _PROTOCOL125_REFERENCE_OUTER_KEYS
            ),
            "reference_outer_values": self.reference_outer_values.copy(),
            "source_reference_fingerprint": np.asarray(
                self.source_reference_fingerprint
            ),
            "shape_keys": np.asarray(_PROTOCOL125_SHAPE_KEYS),
            "shape_values": self.shape_values.copy(),
            "scalar_keys": np.asarray(_PROTOCOL125_SCALAR_KEYS),
            "scalar_values": self.scalar_values.copy(),
            "position_r_first": self.position_r_first.copy(),
            "open_compact_mask": self.open_compact_mask.copy(),
            "ownership_mask": self.ownership_mask.copy(),
            "validation_tolerance": np.asarray(self.validation_tolerance),
            "derivation_recipe": np.asarray(
                "delta-q/Phi-Robin+reference-r-first+"
                "completed-lapse+analytic-shape/scalar-v3"
            ),
            "reference_derivation_recipe": np.asarray(
                "full-fresh-reference-width-seven-outer-row-v1"
            ),
        }


def _derive_protocol125_position_outer_contract_from_primitives(
    source_z,
    source_r,
    completed_position_fields,
    *,
    completed_primitives,
    reference_q,
    reference_phi,
    shape_map,
    scalar_map,
    validation_tolerance=1e-12,
):
    """Internal primitive-level constructor used by the canonical adapter.

    This function is deliberately private because ``completed_primitives``
    contains ``alpha_r``.  Production construction must instead call
    ``derive_joint_parent_position_outer_contract(parent)``, which derives
    that lane from the completed parent and the frozen outer equations.
    """
    return _Protocol125PositionOuterOpenFaceDerivativeContract._from_completed_primitives(
        source_z,
        source_r,
        completed_position_fields,
        completed_primitives=completed_primitives,
        reference_q=reference_q,
        reference_phi=reference_phi,
        shape_map=shape_map,
        scalar_map=scalar_map,
        validation_tolerance=validation_tolerance,
    )


def _stack_source_native_fields(fields, source_shape, label):
    """Stack one full native source grid without accepting extra lanes."""
    if not isinstance(fields, Mapping) or set(fields) != set(NATIVE_CHANNEL_ORDER):
        raise ValueError(
            f"{label} must contain exactly {NATIVE_CHANNEL_ORDER}"
        )
    arrays = []
    for name in NATIVE_CHANNEL_ORDER:
        value = np.asarray(fields[name], dtype=float)
        if value.shape != tuple(source_shape) or not np.all(np.isfinite(value)):
            raise ValueError(f"{label} entry {name} is invalid")
        arrays.append(value)
    return np.stack(arrays, axis=-1)


def _protocol125_acceleration_outer_derivative(source_r, acceleration):
    """Apply the one frozen native radial operator to every source row."""
    radial_operator = derivative_matrix(source_r, 1, 7)
    return np.stack(tuple(
        (radial_operator @ acceleration[:, :, channel].T).T[:, -1]
        for channel in range(len(NATIVE_CHANNEL_ORDER))
    ), axis=-1)


@dataclass(frozen=True)
class Protocol125OuterOpenFaceDerivativeContract:
    """Shared Protocol-125 position/acceleration outer-face contract.

    The position half is the already validated delta-Robin contract, including
    its complete fresh-reference provenance.  The acceleration half accepts
    only full native nodal fields and derives all radial derivatives internally
    with the frozen seven-point operator.  Consequently the Q53 and Q33 final
    representations can consume one exact contract object for both states;
    callers cannot inject an independently prepared acceleration boundary row.
    """

    position_contract: _Protocol125PositionOuterOpenFaceDerivativeContract
    source_acceleration: np.ndarray
    acceleration_r_first: np.ndarray
    acceleration_source_fingerprint: str
    acceleration_ownership_mask: np.ndarray
    identifier: str

    def __post_init__(self):
        position = self.position_contract
        if not isinstance(
            position, _Protocol125PositionOuterOpenFaceDerivativeContract,
        ):
            raise TypeError(
                "Protocol-125 two-state outer contract requires its exact "
                "position contract"
            )
        acceleration = _immutable(self.source_acceleration)
        derivative = _immutable(self.acceleration_r_first)
        ownership = _immutable(self.acceleration_ownership_mask, bool)
        fingerprint = str(self.acceleration_source_fingerprint)
        expected_shape = (
            len(position.source_z),
            len(position.source_r),
            len(NATIVE_CHANNEL_ORDER),
        )
        if (
            acceleration.shape != expected_shape
            or not np.all(np.isfinite(acceleration))
            or derivative.shape
            != (len(position.source_z), len(NATIVE_CHANNEL_ORDER))
            or not np.all(np.isfinite(derivative))
        ):
            raise ValueError("Protocol-125 acceleration outer source is invalid")
        if not np.array_equal(
            ownership, np.ones(len(NATIVE_CHANNEL_ORDER), dtype=bool),
        ):
            raise ValueError(
                "Protocol-125 acceleration outer contract must derive every channel"
            )
        expected_derivative = _protocol125_acceleration_outer_derivative(
            position.source_r, acceleration,
        )
        if not (
            derivative.shape == expected_derivative.shape
            and derivative.tobytes()
            == np.ascontiguousarray(expected_derivative).tobytes()
        ):
            raise ValueError(
                "Protocol-125 acceleration radial derivative is not the exact "
                "width-seven bundle"
            )
        expected_fingerprint = _digest_arrays(
            np.asarray("protocol-125-native-acceleration-source-v1"),
            position.source_z,
            position.source_r,
            acceleration,
        )
        if fingerprint != expected_fingerprint:
            raise ValueError(
                "Protocol-125 acceleration source fingerprint is invalid"
            )
        digest = _digest_arrays(
            np.asarray("protocol-125-two-state-outer-v1"),
            np.asarray(position.identifier),
            np.asarray(expected_fingerprint),
            acceleration,
            derivative,
            ownership,
        )
        expected_identifier = f"protocol-125-two-state-outer-v1:{digest}"
        if str(self.identifier) != expected_identifier:
            raise ValueError("Protocol-125 two-state outer identifier is invalid")
        object.__setattr__(self, "source_acceleration", acceleration)
        object.__setattr__(self, "acceleration_r_first", derivative)
        object.__setattr__(
            self, "acceleration_source_fingerprint", expected_fingerprint,
        )
        object.__setattr__(self, "acceleration_ownership_mask", ownership)
        object.__setattr__(self, "identifier", expected_identifier)

    @classmethod
    def derive(cls, position_contract, completed_acceleration_fields):
        """Build from full native acceleration fields, never from derivatives."""
        if not isinstance(
            position_contract,
            _Protocol125PositionOuterOpenFaceDerivativeContract,
        ):
            raise TypeError(
                "completed acceleration requires the exact Protocol-125 "
                "position outer contract"
            )
        acceleration = _stack_source_native_fields(
            completed_acceleration_fields,
            (len(position_contract.source_z), len(position_contract.source_r)),
            "completed native acceleration fields",
        )
        derivative = _protocol125_acceleration_outer_derivative(
            position_contract.source_r, acceleration,
        )
        ownership = np.ones(len(NATIVE_CHANNEL_ORDER), dtype=bool)
        fingerprint = _digest_arrays(
            np.asarray("protocol-125-native-acceleration-source-v1"),
            position_contract.source_z,
            position_contract.source_r,
            acceleration,
        )
        digest = _digest_arrays(
            np.asarray("protocol-125-two-state-outer-v1"),
            np.asarray(position_contract.identifier),
            np.asarray(fingerprint),
            acceleration,
            derivative,
            ownership,
        )
        return cls(
            position_contract,
            acceleration,
            derivative,
            fingerprint,
            ownership,
            f"protocol-125-two-state-outer-v1:{digest}",
        )

    def r_first_z_jets(
        self, *, state_name, compact_coordinate, outer_value_z_jets,
    ):
        state_name = str(state_name)
        if state_name == "position":
            return self.position_contract.r_first_z_jets(
                state_name=state_name,
                compact_coordinate=compact_coordinate,
                outer_value_z_jets=outer_value_z_jets,
            )
        if state_name != "acceleration":
            raise ValueError("state_name must be position or acceleration")
        z = np.atleast_1d(np.asarray(compact_coordinate, dtype=float))
        values = tuple(
            np.asarray(value, dtype=float) for value in outer_value_z_jets
        )
        if (
            z.ndim != 1
            or np.any(~np.isfinite(z))
            or np.any(z < self.position_contract.source_z[0])
            or np.any(z > self.position_contract.source_z[-1])
            or not 1 <= len(values) <= 3
            or any(
                value.shape != (len(z), len(NATIVE_CHANNEL_ORDER))
                or not np.all(np.isfinite(value))
                for value in values
            )
        ):
            raise ValueError("invalid Protocol-125 outer acceleration query")
        spline = make_interp_spline(
            self.position_contract.source_z,
            self.acceleration_r_first,
            k=min(5, len(self.position_contract.source_z)-1),
            axis=0,
        )
        returned = tuple(spline(z, nu=order) for order in range(len(values)))
        return OuterOpenFaceDerivativeResult(
            returned, self.acceleration_ownership_mask,
        )

    def coefficient_arrays(self):
        """Return both states plus the complete position/reference record."""
        position_record = self.position_contract.coefficient_arrays()
        output = {
            f"position_{name}": np.asarray(value).copy()
            for name, value in position_record.items()
        }
        output.update({
            "position_contract_identifier": np.asarray(
                self.position_contract.identifier
            ),
            "source_acceleration_channel_order": np.asarray(
                NATIVE_CHANNEL_ORDER
            ),
            "source_acceleration": self.source_acceleration.copy(),
            "acceleration_r_first": self.acceleration_r_first.copy(),
            "acceleration_source_fingerprint": np.asarray(
                self.acceleration_source_fingerprint
            ),
            "acceleration_ownership_mask": (
                self.acceleration_ownership_mask.copy()
            ),
            "acceleration_derivation_recipe": np.asarray(
                "full-native-source-width-seven-outer-row-v1"
            ),
            "corner_policy": np.asarray(
                "deterministic-width-seven-record;compact-wall-owns-corners"
            ),
            "shared_state_recipe": np.asarray(
                "one-position-acceleration-outer-contract-v1"
            ),
        })
        return output


def derive_protocol125_outer_derivative_bundle(
    position_contract,
    completed_acceleration_fields,
):
    """Public factory for the final shared two-state outer contract."""
    return Protocol125OuterOpenFaceDerivativeContract.derive(
        position_contract, completed_acceleration_fields,
    )
