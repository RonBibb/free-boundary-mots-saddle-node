"""Analytic finite-wall reference surfaces for Protocol 125 bulk audits.

The reference defect is a numerical datum attached to each independently
constructed parent.  This module gives that datum one prospective off-source
meaning without solving or repairing it on a validation grid:

* source endpoint derivatives are formed once with the native width-seven
  polynomial differentiation matrix;
* the primary surface is Q53 in ``(z, s=(r/12)^2)``;
* the adverse degree-only comparator is Q33 with identical values and
  endpoint traces; and
* every coefficient, source array, trace, and recipe field is persistable and
  fingerprinted.

No function here constructs N0/N1, evaluates a bulk equation, writes an
artifact, or changes a finite-wall reference value.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.joint_parent_representation import (
    raise_if_nonfinite_protocol125_representation_coefficients,
)
from bhps.matched_staged_continuum import hash_arrays


PARENT_R_MAX = 12.0
REFERENCE_CHANNEL_ORDER = ("q", "Phi")
SOURCE_STENCIL_WIDTH = 7

SOURCE_CELL_MIDPOINT_SPECS = MappingProxyType({
    "N0": MappingProxyType({
        "source_shape": (145, 325),
        "source_coordinate_sha256": (
            "15ae4de252ce8c2ca7ff554aa96e12a53fb7c9150f19f912db98e768422b8b58"
        ),
        "midpoint_shape": (144, 324),
        "midpoint_coordinate_sha256": (
            "322493896dde3c69d29957b4420839e663dca9e60576d374385cf6dbbacfd615"
        ),
    }),
    "N1": MappingProxyType({
        "source_shape": (161, 361),
        "source_coordinate_sha256": (
            "6e877756a88dedfde524fc9860696fffb53cfb5396e202e0fbc8da9ab9f3fb4b"
        ),
        "midpoint_shape": (160, 360),
        "midpoint_coordinate_sha256": (
            "9713087c82c804d5d79e3a04d75bad442a5682102fa396ab9e50f8c49588fe8b"
        ),
    }),
})


def _immutable_array(value, dtype=float):
    """Return a C-contiguous array backed by immutable bytes."""
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _update_digest(digest, name, value):
    array = np.ascontiguousarray(value)
    digest.update(str(name).encode())
    digest.update(b"\0")
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes())


def _fingerprint_arrays(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        _update_digest(digest, name, value)
    return digest.hexdigest()


def _source_coordinates(z, r, parent_r_max):
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    parent_r_max = float(parent_r_max)
    if z.ndim != 1 or r.ndim != 1:
        raise ValueError("reference source coordinates must be one-dimensional")
    if len(z) < SOURCE_STENCIL_WIDTH or len(r) < 4:
        raise ValueError("reference source grid is too small for Q53/width-seven")
    if (
        not np.all(np.isfinite(z))
        or not np.all(np.isfinite(r))
        or np.any(np.diff(z) <= 0.0)
        or np.any(np.diff(r) <= 0.0)
    ):
        raise ValueError("reference source coordinates must be finite and increasing")
    if r[0] != 0.0 or not np.isclose(
        r[-1], parent_r_max, rtol=0.0, atol=1e-12,
    ):
        raise ValueError("reference source grid must cover the full radial domain")
    return z, r, parent_r_max


def source_endpoint_z_first(z, values, *, stencil_width=SOURCE_STENCIL_WIDTH):
    """Return the exact native source-stencil traces at both compact ends."""
    z = np.asarray(z, dtype=float)
    values = np.asarray(values, dtype=float)
    stencil_width = int(stencil_width)
    if stencil_width != SOURCE_STENCIL_WIDTH:
        raise ValueError("finite-wall reference endpoint stencil must be width seven")
    if z.ndim != 1 or values.ndim != 3 or values.shape[0] != len(z):
        raise ValueError("finite-wall reference values do not match the compact grid")
    if values.shape[-1] != len(REFERENCE_CHANNEL_ORDER):
        raise ValueError("finite-wall reference must contain exactly q and Phi")
    if not np.all(np.isfinite(values)):
        raise ValueError("finite-wall reference values are nonfinite")
    operator = derivative_matrix(z, 1, stencil_width)
    derivative = np.asarray(
        operator @ values.reshape(len(z), -1), dtype=float,
    ).reshape(values.shape)
    return _immutable_array(derivative[[0, -1]])


@dataclass(frozen=True)
class SourceCellMidpointCoordinates:
    """One frozen direct tensor of source-cell centers."""

    label: str
    z: np.ndarray
    r: np.ndarray
    coordinate_sha256: str

    def __post_init__(self):
        label = str(self.label)
        z = _immutable_array(self.z)
        r = _immutable_array(self.r)
        coordinate_sha256 = str(self.coordinate_sha256)
        if label not in SOURCE_CELL_MIDPOINT_SPECS:
            raise ValueError("source-cell midpoint label must be N0 or N1")
        expected = SOURCE_CELL_MIDPOINT_SPECS[label]
        if (len(z), len(r)) != expected["midpoint_shape"]:
            raise ValueError("source-cell midpoint shape differs from protocol")
        if hash_arrays(z, r) != expected["midpoint_coordinate_sha256"]:
            raise ValueError("source-cell midpoint coordinates differ from protocol")
        if coordinate_sha256 != expected["midpoint_coordinate_sha256"]:
            raise ValueError("source-cell midpoint digest differs from coordinates")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "r", r)
        object.__setattr__(self, "coordinate_sha256", coordinate_sha256)


def source_cell_midpoint_coordinates(label, source_z, source_r):
    """Return direct cell centers after verifying the frozen source grid."""
    label = str(label)
    if label not in SOURCE_CELL_MIDPOINT_SPECS:
        raise ValueError("source-cell midpoint label must be N0 or N1")
    specification = SOURCE_CELL_MIDPOINT_SPECS[label]
    source_z = np.asarray(source_z, dtype=float)
    source_r = np.asarray(source_r, dtype=float)
    if (len(source_z), len(source_r)) != specification["source_shape"]:
        raise ValueError("source grid shape differs from frozen parent")
    if hash_arrays(source_z, source_r) != specification["source_coordinate_sha256"]:
        raise ValueError("source coordinates differ from frozen parent")
    z = 0.5*(source_z[:-1]+source_z[1:])
    r = 0.5*(source_r[:-1]+source_r[1:])
    found = hash_arrays(z, r)
    if found != specification["midpoint_coordinate_sha256"]:
        raise RuntimeError("computed source-cell midpoint digest differs from protocol")
    return SourceCellMidpointCoordinates(label, z, r, found)


def frozen_source_cell_midpoint_coordinates(label):
    """Construct one frozen N0/N1 direct-cell-center mesh from canonical axes."""
    label = str(label)
    if label not in SOURCE_CELL_MIDPOINT_SPECS:
        raise ValueError("source-cell midpoint label must be N0 or N1")
    nz, nr = SOURCE_CELL_MIDPOINT_SPECS[label]["source_shape"]
    return source_cell_midpoint_coordinates(
        label,
        np.linspace(1.0, np.e, nz),
        np.linspace(0.0, PARENT_R_MAX, nr),
    )


@dataclass(frozen=True)
class _ReferenceTensorHermiteSurface:
    """One source-trace-clamped tensor surface in ``(z,s)``."""

    z_knots: np.ndarray
    s_knots: np.ndarray
    coefficients: np.ndarray
    z_degree: int
    s_degree: int
    parent_r_max: float
    z_boundary: str = "clamped_source_width7_z_first"

    def __post_init__(self):
        z_knots = _immutable_array(self.z_knots)
        s_knots = _immutable_array(self.s_knots)
        coefficients = _immutable_array(self.coefficients)
        z_degree = int(self.z_degree)
        s_degree = int(self.s_degree)
        parent_r_max = float(self.parent_r_max)
        if z_knots.ndim != 1 or s_knots.ndim != 1 or coefficients.ndim != 3:
            raise ValueError("invalid finite-wall reference surface arrays")
        if z_degree not in (3, 5) or s_degree != 3 or parent_r_max <= 0.0:
            raise ValueError("finite-wall reference surface must be Q53 or Q33")
        if coefficients.shape[0] != len(z_knots)-z_degree-1:
            raise ValueError("finite-wall reference compact basis mismatch")
        if coefficients.shape[1] != len(s_knots)-s_degree-1:
            raise ValueError("finite-wall reference radial basis mismatch")
        if coefficients.shape[-1] != len(REFERENCE_CHANNEL_ORDER):
            raise ValueError("finite-wall reference surface channel mismatch")
        if not all(np.all(np.isfinite(item)) for item in (
            z_knots, s_knots, coefficients,
        )):
            raise ValueError("finite-wall reference surface is nonfinite")
        if str(self.z_boundary) != "clamped_source_width7_z_first":
            raise ValueError("finite-wall reference surface boundary recipe changed")
        object.__setattr__(self, "z_knots", z_knots)
        object.__setattr__(self, "s_knots", s_knots)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "z_degree", z_degree)
        object.__setattr__(self, "s_degree", s_degree)
        object.__setattr__(self, "parent_r_max", parent_r_max)
        object.__setattr__(self, "z_boundary", str(self.z_boundary))

    @classmethod
    def build(
        cls, z, r, values, endpoint_z_first, *, z_degree, parent_r_max,
    ):
        z, r, parent_r_max = _source_coordinates(z, r, parent_r_max)
        values = np.asarray(values, dtype=float)
        endpoint_z_first = np.asarray(endpoint_z_first, dtype=float)
        if values.shape != (len(z), len(r), len(REFERENCE_CHANNEL_ORDER)):
            raise ValueError("finite-wall reference source values have wrong shape")
        if endpoint_z_first.shape != (2, len(r), len(REFERENCE_CHANNEL_ORDER)):
            raise ValueError("finite-wall reference endpoint traces have wrong shape")
        if not all(np.all(np.isfinite(item)) for item in (
            values, endpoint_z_first,
        )):
            raise ValueError("finite-wall reference construction data are nonfinite")
        z_degree = int(z_degree)
        if z_degree not in (3, 5):
            raise ValueError("finite-wall reference compact degree must be five or three")
        s = (r/parent_r_max)**2
        radial = make_interp_spline(s, values, k=3, axis=1)
        lower = make_interp_spline(s, endpoint_z_first[0], k=3, axis=0)
        upper = make_interp_spline(s, endpoint_z_first[1], k=3, axis=0)
        if not (
            np.array_equal(lower.t, radial.t)
            and np.array_equal(upper.t, radial.t)
        ):
            raise RuntimeError("finite-wall reference radial bases differ")
        radial_coefficients = np.moveaxis(radial.c, 0, 1)
        boundary = ([(1, lower.c)], [(1, upper.c)])
        if z_degree == 5:
            knots = np.concatenate((
                np.repeat(z[0], 6), z[2:-2], np.repeat(z[-1], 6),
            ))
            compact = make_interp_spline(
                z, radial_coefficients, k=5, axis=0, t=knots,
                bc_type=boundary,
            )
        else:
            compact = make_interp_spline(
                z, radial_coefficients, k=3, axis=0, bc_type=boundary,
            )
        compact_coefficients = np.asarray(compact.c).copy()
        raise_if_nonfinite_protocol125_representation_coefficients(
            compact_coefficients,
            recipe=(
                "finite-wall-reference-Q53-compact"
                if z_degree == 5
                else "finite-wall-reference-Q33-compact"
            ),
            input_arrays={
                "source_z": z,
                "source_r": r,
                "source_values": values,
                "endpoint_z_first": endpoint_z_first,
                "parent_r_max": np.asarray(parent_r_max),
            },
        )
        return cls(
            np.asarray(compact.t).copy(),
            np.asarray(radial.t).copy(),
            compact_coefficients,
            z_degree,
            3,
            parent_r_max,
        )

    def _coordinates(self, z, r):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        r = np.atleast_1d(np.asarray(r, dtype=float))
        z_min = self.z_knots[self.z_degree]
        z_max = self.z_knots[-self.z_degree-1]
        if z.ndim != 1 or r.ndim != 1:
            raise ValueError("finite-wall reference queries must be one-dimensional")
        if (
            np.any(~np.isfinite(z))
            or np.any(~np.isfinite(r))
            or np.any(z < z_min)
            or np.any(z > z_max)
            or np.any(r < 0.0)
            or np.any(r > self.parent_r_max)
        ):
            raise ValueError("finite-wall reference query lies outside its domain")
        return z, r

    def evaluate_s(self, z, r, *, z_order=0, s_order=0):
        z, r = self._coordinates(z, r)
        z_order = int(z_order)
        s_order = int(s_order)
        if not 0 <= z_order <= self.z_degree:
            raise ValueError("unsupported finite-wall reference compact derivative")
        if not 0 <= s_order <= self.s_degree:
            raise ValueError("unsupported finite-wall reference radial derivative")
        s = (r/self.parent_r_max)**2
        radial = BSpline(
            self.s_knots, self.coefficients, self.s_degree,
            axis=1, extrapolate=False,
        )(s, nu=s_order)
        return BSpline(
            self.z_knots, radial, self.z_degree,
            axis=0, extrapolate=False,
        )(z, nu=z_order)

    def evaluate(self, z, r, *, z_order=0, r_order=0):
        z, r = self._coordinates(z, r)
        r_order = int(r_order)
        if r_order == 0:
            return self.evaluate_s(z, r, z_order=z_order)
        ds_dr = 2.0*r/self.parent_r_max**2
        if r_order == 1:
            return self.evaluate_s(
                z, r, z_order=z_order, s_order=1,
            )*ds_dr[None, :, None]
        if r_order == 2:
            first = self.evaluate_s(
                z, r, z_order=z_order, s_order=1,
            )
            second = self.evaluate_s(
                z, r, z_order=z_order, s_order=2,
            )
            return (
                second*ds_dr[None, :, None]**2
                + first*(2.0/self.parent_r_max**2)
            )
        raise ValueError("finite-wall reference radial order must be zero through two")

    def coefficient_arrays(self, prefix):
        return {
            f"{prefix}_z_knots": self.z_knots.copy(),
            f"{prefix}_s_knots": self.s_knots.copy(),
            f"{prefix}_coefficients": self.coefficients.copy(),
            f"{prefix}_z_degree": np.asarray(self.z_degree),
            f"{prefix}_s_degree": np.asarray(self.s_degree),
            f"{prefix}_parent_r_max": np.asarray(self.parent_r_max),
            f"{prefix}_z_boundary": np.asarray(self.z_boundary),
        }

    @classmethod
    def from_arrays(cls, archive, prefix):
        return cls(
            np.asarray(archive[f"{prefix}_z_knots"]),
            np.asarray(archive[f"{prefix}_s_knots"]),
            np.asarray(archive[f"{prefix}_coefficients"]),
            int(archive[f"{prefix}_z_degree"]),
            int(archive[f"{prefix}_s_degree"]),
            float(archive[f"{prefix}_parent_r_max"]),
            str(archive[f"{prefix}_z_boundary"]),
        )


@dataclass(frozen=True)
class FiniteWallReferenceJet:
    """Reference q/Phi values and analytic spatial derivatives."""

    z: np.ndarray
    r: np.ndarray
    values: np.ndarray
    first: np.ndarray
    second: np.ndarray
    channel_order: tuple = REFERENCE_CHANNEL_ORDER

    def __post_init__(self):
        z = _immutable_array(self.z)
        r = _immutable_array(self.r)
        values = _immutable_array(self.values)
        first = _immutable_array(self.first)
        second = _immutable_array(self.second)
        channel_order = tuple(str(value) for value in self.channel_order)
        shape = (len(z), len(r), len(REFERENCE_CHANNEL_ORDER))
        if values.shape != shape:
            raise ValueError("finite-wall reference jet values have wrong shape")
        if first.shape != (2, *shape) or second.shape != (2, 2, *shape):
            raise ValueError("finite-wall reference derivative arrays have wrong shape")
        if channel_order != REFERENCE_CHANNEL_ORDER:
            raise ValueError("finite-wall reference jet channel order changed")
        if not all(np.all(np.isfinite(item)) for item in (
            z, r, values, first, second,
        )):
            raise ValueError("finite-wall reference jet is nonfinite")
        if not np.array_equal(second[0, 1], second[1, 0]):
            raise ValueError("finite-wall reference mixed derivatives are asymmetric")
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "r", r)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "first", first)
        object.__setattr__(self, "second", second)
        object.__setattr__(self, "channel_order", channel_order)

    def as_derivative_mapping(self):
        """Return immutable named arrays for the analytic bulk backend."""
        q_index = self.channel_order.index("q")
        phi_index = self.channel_order.index("Phi")
        return MappingProxyType({
            "q": self.values[:, :, q_index],
            "q_z": self.first[0, :, :, q_index],
            "q_r": self.first[1, :, :, q_index],
            "q_zz": self.second[0, 0, :, :, q_index],
            "q_zr": self.second[0, 1, :, :, q_index],
            "q_rr": self.second[1, 1, :, :, q_index],
            "phi": self.values[:, :, phi_index],
            "phi_z": self.first[0, :, :, phi_index],
            "phi_r": self.first[1, :, :, phi_index],
            "phi_zz": self.second[0, 0, :, :, phi_index],
            "phi_zr": self.second[0, 1, :, :, phi_index],
            "phi_rr": self.second[1, 1, :, :, phi_index],
        })


@dataclass(frozen=True)
class FiniteWallReferenceHermiteRepresentation:
    """One Q53 or Q33 representation of a fixed source reference."""

    surface: _ReferenceTensorHermiteSurface
    source_z: np.ndarray
    source_r: np.ndarray
    source_values: np.ndarray
    endpoint_z_first: np.ndarray
    recipe: str
    stencil_width: int = SOURCE_STENCIL_WIDTH
    channel_order: tuple = REFERENCE_CHANNEL_ORDER

    def __post_init__(self):
        source_z, source_r, _ = _source_coordinates(
            self.source_z, self.source_r, self.surface.parent_r_max,
        )
        source_z = _immutable_array(source_z)
        source_r = _immutable_array(source_r)
        source_values = _immutable_array(self.source_values)
        endpoints = _immutable_array(self.endpoint_z_first)
        recipe = str(self.recipe)
        stencil_width = int(self.stencil_width)
        channel_order = tuple(str(value) for value in self.channel_order)
        expected_recipe = {5: "primary-Q53", 3: "comparator-Q33"}
        if recipe != expected_recipe.get(self.surface.z_degree):
            raise ValueError("finite-wall reference recipe and compact degree differ")
        if stencil_width != SOURCE_STENCIL_WIDTH:
            raise ValueError("finite-wall reference source stencil must be width seven")
        if channel_order != REFERENCE_CHANNEL_ORDER:
            raise ValueError("finite-wall reference channel order changed")
        if source_values.shape != (
            len(source_z), len(source_r), len(REFERENCE_CHANNEL_ORDER),
        ):
            raise ValueError("finite-wall reference stored source values have wrong shape")
        if endpoints.shape != (2, len(source_r), len(REFERENCE_CHANNEL_ORDER)):
            raise ValueError("finite-wall reference stored endpoint traces have wrong shape")
        recomputed = source_endpoint_z_first(
            source_z, source_values, stencil_width=stencil_width,
        )
        if not np.array_equal(recomputed, endpoints):
            raise ValueError("stored finite-wall reference endpoint traces changed")
        found_values = self.surface.evaluate(source_z, source_r)
        scale = np.maximum.reduce((
            np.ones_like(found_values), np.abs(found_values), np.abs(source_values),
        ))
        if float(np.max(np.abs(found_values-source_values)/scale)) > 1e-12:
            raise ValueError("finite-wall reference surface does not reproduce source values")
        found_endpoints = self.surface.evaluate(
            source_z[[0, -1]], source_r, z_order=1,
        )
        endpoint_scale = np.maximum.reduce((
            np.ones_like(found_endpoints),
            np.abs(found_endpoints),
            np.abs(endpoints),
        ))
        if float(np.max(np.abs(found_endpoints-endpoints)/endpoint_scale)) > 1e-12:
            raise ValueError("finite-wall reference surface does not reproduce endpoint traces")
        object.__setattr__(self, "source_z", source_z)
        object.__setattr__(self, "source_r", source_r)
        object.__setattr__(self, "source_values", source_values)
        object.__setattr__(self, "endpoint_z_first", endpoints)
        object.__setattr__(self, "recipe", recipe)
        object.__setattr__(self, "stencil_width", stencil_width)
        object.__setattr__(self, "channel_order", channel_order)

    @classmethod
    def build(
        cls, z, r, source_values, endpoint_z_first, *, z_degree,
        parent_r_max=PARENT_R_MAX,
    ):
        surface = _ReferenceTensorHermiteSurface.build(
            z,
            r,
            source_values,
            endpoint_z_first,
            z_degree=z_degree,
            parent_r_max=parent_r_max,
        )
        return cls(
            surface,
            z,
            r,
            source_values,
            endpoint_z_first,
            "primary-Q53" if int(z_degree) == 5 else "comparator-Q33",
        )

    def evaluate(self, z, r):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        r = np.atleast_1d(np.asarray(r, dtype=float))
        values = self.surface.evaluate(z, r)
        first = np.stack((
            self.surface.evaluate(z, r, z_order=1),
            self.surface.evaluate(z, r, r_order=1),
        ))
        mixed = self.surface.evaluate(z, r, z_order=1, r_order=1)
        second = np.empty((2, 2, *values.shape), dtype=float)
        second[0, 0] = self.surface.evaluate(z, r, z_order=2)
        second[0, 1] = mixed
        second[1, 0] = mixed
        second[1, 1] = self.surface.evaluate(z, r, r_order=2)
        return FiniteWallReferenceJet(z, r, values, first, second)

    def coefficient_arrays(self, prefix="finite_wall_reference"):
        return {
            **self.surface.coefficient_arrays(f"{prefix}_surface"),
            f"{prefix}_source_z": self.source_z.copy(),
            f"{prefix}_source_r": self.source_r.copy(),
            f"{prefix}_source_values": self.source_values.copy(),
            f"{prefix}_endpoint_z_first": self.endpoint_z_first.copy(),
            f"{prefix}_recipe": np.asarray(self.recipe),
            f"{prefix}_stencil_width": np.asarray(self.stencil_width),
            f"{prefix}_channel_order": np.asarray(self.channel_order),
        }

    def fingerprint(self):
        return _fingerprint_arrays(self.coefficient_arrays())

    @classmethod
    def from_arrays(cls, archive, prefix="finite_wall_reference"):
        return cls(
            _ReferenceTensorHermiteSurface.from_arrays(
                archive, f"{prefix}_surface",
            ),
            np.asarray(archive[f"{prefix}_source_z"]),
            np.asarray(archive[f"{prefix}_source_r"]),
            np.asarray(archive[f"{prefix}_source_values"]),
            np.asarray(archive[f"{prefix}_endpoint_z_first"]),
            str(archive[f"{prefix}_recipe"]),
            int(archive[f"{prefix}_stencil_width"]),
            tuple(str(value) for value in archive[f"{prefix}_channel_order"]),
        )


@dataclass(frozen=True)
class FiniteWallReferenceHermitePair:
    """Q53 primary and identical-input Q33 finite-wall reference pair."""

    primary: FiniteWallReferenceHermiteRepresentation
    comparator: FiniteWallReferenceHermiteRepresentation

    def __post_init__(self):
        if self.primary.recipe != "primary-Q53":
            raise ValueError("finite-wall reference primary must be Q53")
        if self.comparator.recipe != "comparator-Q33":
            raise ValueError("finite-wall reference comparator must be Q33")
        if not all((
            np.array_equal(self.primary.source_z, self.comparator.source_z),
            np.array_equal(self.primary.source_r, self.comparator.source_r),
            np.array_equal(
                self.primary.source_values, self.comparator.source_values,
            ),
            np.array_equal(
                self.primary.endpoint_z_first,
                self.comparator.endpoint_z_first,
            ),
            self.primary.stencil_width == self.comparator.stencil_width,
            self.primary.channel_order == self.comparator.channel_order,
        )):
            raise ValueError("finite-wall reference pair does not share exact inputs")

    @classmethod
    def build(
        cls,
        z,
        r,
        q,
        phi,
        *,
        stencil_width=SOURCE_STENCIL_WIDTH,
        parent_r_max=PARENT_R_MAX,
    ):
        z, r, parent_r_max = _source_coordinates(z, r, parent_r_max)
        stencil_width = int(stencil_width)
        if stencil_width != SOURCE_STENCIL_WIDTH:
            raise ValueError("finite-wall reference source stencil must be width seven")
        q = np.asarray(q, dtype=float)
        phi = np.asarray(phi, dtype=float)
        expected = (len(z), len(r))
        if q.shape != expected or phi.shape != expected:
            raise ValueError("finite-wall reference q/Phi arrays have wrong shape")
        if not all(np.all(np.isfinite(item)) for item in (q, phi)):
            raise ValueError("finite-wall reference q/Phi arrays are nonfinite")
        values = np.stack((q, phi), axis=-1)
        endpoints = source_endpoint_z_first(
            z, values, stencil_width=stencil_width,
        )
        return cls(
            FiniteWallReferenceHermiteRepresentation.build(
                z, r, values, endpoints, z_degree=5,
                parent_r_max=parent_r_max,
            ),
            FiniteWallReferenceHermiteRepresentation.build(
                z, r, values, endpoints, z_degree=3,
                parent_r_max=parent_r_max,
            ),
        )

    def coefficient_arrays(self, prefix="finite_wall_reference_pair"):
        return {
            **self.primary.coefficient_arrays(f"{prefix}_primary"),
            **self.comparator.coefficient_arrays(f"{prefix}_comparator"),
        }

    def fingerprint(self):
        return _fingerprint_arrays(self.coefficient_arrays())

    @classmethod
    def from_arrays(cls, archive, prefix="finite_wall_reference_pair"):
        return cls(
            FiniteWallReferenceHermiteRepresentation.from_arrays(
                archive, f"{prefix}_primary",
            ),
            FiniteWallReferenceHermiteRepresentation.from_arrays(
                archive, f"{prefix}_comparator",
            ),
        )
