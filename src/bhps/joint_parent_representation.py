"""Native-metric Hermite surfaces for the Protocol-125 parent rebuild.

This module is intentionally isolated from the sealed matched-continuum
representation.  It does not build a scientific parent or apply a wall
repair.  It only turns already-completed native physical fields and their
row-derived compact endpoint derivatives into immutable tensor-product
surfaces.

The authoritative anisotropy object is the physical numerator
``N = h_rr - h_perp``.  The regular reduced coefficient
``q4 = N/r**2`` is evaluated from that surface, with analytic even limits at
the axis.  No API in this module accepts, fits, or splines an independent
``q4`` array.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.regular_so3_gh_reduction import FIELD_ORDER as REDUCED_FIELD_ORDER


PARENT_R_MAX = 12.0
PHYSICAL_SURFACE_CHANNELS = (
    "h00",
    "h_perp",
    "h_rr",
    "h_zz",
    "Phi",
    "chi",
)
REGULAR_VECTOR_CHANNELS = ("v_z", "v_0")
NATIVE_CHANNEL_ORDER = PHYSICAL_SURFACE_CHANNELS + REGULAR_VECTOR_CHANNELS
COORDINATE_COMPONENT_ORDER = (
    "h_z0",
    "h_zr",
    "h00",
    "h_perp",
    "h_rr",
    "h_0r",
    "h_zz",
    "Phi",
    "chi",
)
SEALED_ADVERSE_COMPARATOR_NAMES = (
    "sealed-reduced-Q33",
    "sealed-reduced-Q55-not-a-knot",
)
SEALED_ADVERSE_COMPARATOR_RECIPES = MappingProxyType({
    SEALED_ADVERSE_COMPARATOR_NAMES[0]: MappingProxyType({
        "basis": "tensor-product-BSpline",
        "compact_degree": 3,
        "squared_radius_degree": 3,
        "position_velocity_boundary": "stored-z-first-clamped",
        "acceleration_boundary": "not-a-knot",
    }),
    SEALED_ADVERSE_COMPARATOR_NAMES[1]: MappingProxyType({
        "basis": "RectBivariateSpline",
        "compact_degree": 5,
        "squared_radius_degree": 5,
        "position_velocity_boundary": "not-a-knot",
        "acceleration_boundary": "not-a-knot",
    }),
})
ADDITIVE_COMPARATOR_NAME = "native-metric-identical-endpoint-Q53-Q33"


class CompactWallDerivativeContract(Protocol):
    """Query-time nonlinear compact-wall derivative contract.

    ``wall_value_s_jets[m]`` has shape ``(2, nr, 8)`` and contains the
    ``m``-th squared-radius derivative of the radially interpolated native
    values at the lower and upper compact walls.  The returned tuple must
    have the same length and shapes and contains
    ``partial_s**m partial_z(field)`` derived from the exact native wall row.
    """

    identifier: str

    def z_first_s_jets(
        self, *, state_name, radius, wall_value_s_jets
    ): ...

    def coefficient_arrays(self): ...


class OuterOpenFaceDerivativeContract(Protocol):
    """Query-time radial derivative contract on the open outer face.

    ``outer_value_z_jets[m]`` has shape ``(nz, 8)``.  The returned
    :class:`OuterOpenFaceDerivativeResult` supplies the corresponding compact
    derivatives of the physical radial first derivative and an explicit
    channel-ownership mask.  Compact-wall corner values are deliberately
    ignored by the representation because the wall owns both corners.
    """

    identifier: str

    def r_first_z_jets(
        self, *, state_name, compact_coordinate, outer_value_z_jets
    ): ...

    def coefficient_arrays(self): ...


@dataclass(frozen=True)
class OuterOpenFaceDerivativeResult:
    """Contract result plus explicit native-channel ownership."""

    r_first_z_jets: tuple
    ownership_mask: np.ndarray

    def __post_init__(self):
        jets = tuple(_immutable_array(value) for value in self.r_first_z_jets)
        mask = _immutable_array(self.ownership_mask, bool)
        if mask.shape != (len(NATIVE_CHANNEL_ORDER),):
            raise ValueError("outer-open-face ownership mask has the wrong shape")
        if not np.any(mask):
            raise ValueError("outer-open-face contract must own at least one channel")
        object.__setattr__(self, "r_first_z_jets", jets)
        object.__setattr__(self, "ownership_mask", mask)


def _immutable_array(value, dtype=float):
    """Return a C-contiguous array backed by immutable bytes."""
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _arrays_bitwise_equal(left, right):
    """Return whether two arrays have identical shape, dtype, and bytes."""
    left = np.ascontiguousarray(left)
    right = np.ascontiguousarray(right)
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes() == right.tobytes()
    )


def _update_array_digest(digest, name, value):
    array = np.ascontiguousarray(value)
    digest.update(str(name).encode())
    digest.update(b"\0")
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes())


def _fingerprint_named_arrays(**arrays):
    digest = hashlib.sha256()
    for name in sorted(arrays):
        _update_array_digest(digest, name, arrays[name])
    return digest.hexdigest()


REPRESENTATION_COEFFICIENT_FAILURE_PROTOCOL_IDENTIFIER = (
    "Protocol-125-representation-coefficient-failure-v2"
)
REPRESENTATION_COEFFICIENT_RECIPES = (
    "native-tensor-Q53-compact",
    "native-tensor-Q33-compact",
    "native-radial-cubic-s",
    "finite-wall-reference-Q53-compact",
    "finite-wall-reference-Q33-compact",
)


class Protocol125RepresentationCoefficientFailure(RuntimeError):
    """Finite representation inputs produced nonfinite spline coefficients."""

    def __init__(self, evidence):
        self.evidence = validate_protocol125_representation_coefficient_failure(
            evidence,
        )
        super().__init__("native Hermite construction produced nonfinite coefficients")


def _representation_input_sha256(input_arrays):
    if not isinstance(input_arrays, Mapping) or not input_arrays:
        raise ValueError("representation failure inputs must be a nonempty mapping")
    normalized = {}
    for name in sorted(input_arrays):
        if not isinstance(name, str) or not name:
            raise ValueError("representation failure input name is invalid")
        value = np.ascontiguousarray(np.asarray(input_arrays[name]))
        if value.dtype == object:
            raise ValueError("representation failure input has object dtype")
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ValueError("representation failure inputs must be finite")
        normalized[name] = value
    return _fingerprint_named_arrays(**normalized)


def _representation_coefficient_failure_evidence(
    input_arrays,
    coefficients,
    *,
    recipe,
):
    recipe = str(recipe)
    coefficients = np.ascontiguousarray(np.asarray(coefficients))
    if (
        recipe not in REPRESENTATION_COEFFICIENT_RECIPES
        or coefficients.ndim != 3
        or coefficients.dtype != np.dtype(float)
    ):
        raise ValueError("representation failure output recipe is invalid")
    input_sha256 = _representation_input_sha256(input_arrays)
    raw = np.frombuffer(coefficients.tobytes(), dtype=np.uint8).copy()
    payload = {
        "protocol_identifier": (
            REPRESENTATION_COEFFICIENT_FAILURE_PROTOCOL_IDENTIFIER
        ),
        "recipe": recipe,
        "parent_identity": "",
        "input_sha256": input_sha256,
        "coefficient_shape": tuple(int(item) for item in coefficients.shape),
        "coefficient_dtype": coefficients.dtype.str,
        "coefficient_raw_bytes": _immutable_array(raw, None),
        "nonfinite_count": int(np.count_nonzero(~np.isfinite(coefficients))),
    }
    payload["fingerprint"] = _fingerprint_named_arrays(
        protocol_identifier=np.asarray(payload["protocol_identifier"]),
        recipe=np.asarray(recipe),
        parent_identity=np.asarray(payload["parent_identity"]),
        input_sha256=np.asarray(input_sha256),
        coefficient_shape=np.asarray(payload["coefficient_shape"], dtype=np.int64),
        coefficient_dtype=np.asarray(payload["coefficient_dtype"]),
        coefficient_raw_bytes=payload["coefficient_raw_bytes"],
        nonfinite_count=np.asarray(payload["nonfinite_count"]),
    )
    return validate_protocol125_representation_coefficient_failure(payload)


def validate_protocol125_representation_coefficient_failure(evidence):
    required = (
        "protocol_identifier", "recipe", "parent_identity", "input_sha256",
        "coefficient_shape", "coefficient_dtype", "coefficient_raw_bytes",
        "nonfinite_count", "fingerprint",
    )
    if not isinstance(evidence, Mapping) or tuple(evidence) != required:
        raise ValueError("representation coefficient failure schema differs")
    if str(evidence["protocol_identifier"]) != (
        REPRESENTATION_COEFFICIENT_FAILURE_PROTOCOL_IDENTIFIER
    ):
        raise ValueError("representation coefficient failure protocol differs")
    recipe = str(evidence["recipe"])
    if recipe not in REPRESENTATION_COEFFICIENT_RECIPES:
        raise ValueError("representation coefficient failure recipe differs")
    parent_identity = str(evidence["parent_identity"])
    if parent_identity and (
        len(parent_identity) != 64
        or any(character not in "0123456789abcdef" for character in parent_identity)
    ):
        raise ValueError("representation coefficient failure parent differs")
    input_sha256 = str(evidence["input_sha256"])
    fingerprint = str(evidence["fingerprint"])
    if any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in (input_sha256, fingerprint)
    ):
        raise ValueError("representation coefficient failure digest is invalid")
    raw_shape = evidence["coefficient_shape"]
    if (
        not isinstance(raw_shape, tuple)
        or len(raw_shape) != 3
        or any(type(item) is not int or item <= 0 for item in raw_shape)
    ):
        raise ValueError("representation coefficient shape is invalid")
    shape = raw_shape
    try:
        dtype = np.dtype(str(evidence["coefficient_dtype"]))
    except (TypeError, ValueError) as error:
        raise ValueError("representation coefficient dtype is invalid") from error
    raw = np.ascontiguousarray(np.asarray(evidence["coefficient_raw_bytes"]))
    if (
        dtype != np.dtype(float)
        or str(evidence["coefficient_dtype"]) != dtype.str
        or raw.dtype != np.uint8
        or raw.ndim != 1
        or raw.nbytes != int(np.prod(shape))*dtype.itemsize
    ):
        raise ValueError("representation coefficient failure payload is invalid")
    coefficients = np.frombuffer(raw.tobytes(), dtype=dtype).reshape(shape)
    nonfinite_count = int(np.count_nonzero(~np.isfinite(coefficients)))
    if (
        type(evidence["nonfinite_count"]) is not int
        or nonfinite_count <= 0
        or nonfinite_count != evidence["nonfinite_count"]
    ):
        raise ValueError("representation coefficient failure is not nonfinite")
    expected = _fingerprint_named_arrays(
        protocol_identifier=np.asarray(evidence["protocol_identifier"]),
        recipe=np.asarray(recipe),
        parent_identity=np.asarray(parent_identity),
        input_sha256=np.asarray(input_sha256),
        coefficient_shape=np.asarray(shape, dtype=np.int64),
        coefficient_dtype=np.asarray(dtype.str),
        coefficient_raw_bytes=raw,
        nonfinite_count=np.asarray(nonfinite_count),
    )
    if fingerprint != expected:
        raise ValueError("representation coefficient failure fingerprint differs")
    normalized = {
        "protocol_identifier": str(evidence["protocol_identifier"]),
        "recipe": recipe,
        "parent_identity": parent_identity,
        "input_sha256": input_sha256,
        "coefficient_shape": shape,
        "coefficient_dtype": dtype.str,
        "coefficient_raw_bytes": _immutable_array(raw, None),
        "nonfinite_count": nonfinite_count,
        "fingerprint": fingerprint,
    }
    if tuple(normalized) != required:
        raise RuntimeError("representation coefficient failure normalization differs")
    return MappingProxyType(normalized)


def bind_protocol125_representation_coefficient_failure(
    evidence,
    parent_identity,
):
    """Bind fresh failure evidence once to its production parent identity."""
    evidence = validate_protocol125_representation_coefficient_failure(evidence)
    parent_identity = str(parent_identity)
    if (
        len(parent_identity) != 64
        or any(character not in "0123456789abcdef" for character in parent_identity)
    ):
        raise ValueError("representation failure parent identity is invalid")
    if evidence["parent_identity"] not in ("", parent_identity):
        raise ValueError("representation failure is already bound to another parent")
    payload = {
        name: evidence[name]
        for name in evidence
        if name != "fingerprint"
    }
    payload["parent_identity"] = parent_identity
    payload["fingerprint"] = _fingerprint_named_arrays(
        protocol_identifier=np.asarray(payload["protocol_identifier"]),
        recipe=np.asarray(payload["recipe"]),
        parent_identity=np.asarray(parent_identity),
        input_sha256=np.asarray(payload["input_sha256"]),
        coefficient_shape=np.asarray(payload["coefficient_shape"], dtype=np.int64),
        coefficient_dtype=np.asarray(payload["coefficient_dtype"]),
        coefficient_raw_bytes=payload["coefficient_raw_bytes"],
        nonfinite_count=np.asarray(payload["nonfinite_count"]),
    )
    ordered = {
        name: payload[name]
        for name in (
            "protocol_identifier", "recipe", "parent_identity", "input_sha256",
            "coefficient_shape", "coefficient_dtype", "coefficient_raw_bytes",
            "nonfinite_count", "fingerprint",
        )
    }
    return validate_protocol125_representation_coefficient_failure(ordered)


def raise_if_nonfinite_protocol125_representation_coefficients(
    coefficients,
    *,
    recipe,
    input_arrays,
):
    """Raise typed evidence only for a fresh finite-input spline failure."""
    coefficients = np.ascontiguousarray(np.asarray(coefficients))
    if coefficients.dtype.kind in "fc" and not np.all(np.isfinite(coefficients)):
        raise Protocol125RepresentationCoefficientFailure(
            _representation_coefficient_failure_evidence(
                input_arrays,
                coefficients,
                recipe=recipe,
            )
        )


def _freeze_contract_record(contract, label):
    identifier = _contract_identifier(contract, label)
    method = getattr(contract, "coefficient_arrays", None)
    if method is None or not callable(method):
        raise ValueError(f"{label} contract requires coefficient_arrays()")
    raw = method()
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"{label} contract record must be a nonempty mapping")
    if "identifier" in raw:
        raise ValueError(f"{label} contract record reserves the identifier key")
    record = [("identifier", _immutable_array(np.asarray(identifier), None))]
    for name in sorted(raw):
        name = str(name)
        value = np.asarray(raw[name])
        if value.dtype == object:
            raise ValueError(f"{label} contract record {name} has object dtype")
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise ValueError(f"{label} contract record {name} is nonfinite")
        record.append((name, _immutable_array(value, None)))
    digest = hashlib.sha256()
    for name, value in record:
        _update_array_digest(digest, name, value)
    return tuple(record), digest.hexdigest()


def _contract_records_equal(left, right):
    return (
        len(left) == len(right)
        and all(
            left_name == right_name and np.array_equal(left_value, right_value)
            for (left_name, left_value), (right_name, right_value)
            in zip(left, right)
        )
    )


def _persist_contract_record(output, prefix, record, fingerprint):
    output[f"{prefix}_names"] = np.asarray([name for name, _ in record])
    for index, (_, value) in enumerate(record):
        output[f"{prefix}_value_{index}"] = np.asarray(value).copy()
    output[f"{prefix}_fingerprint"] = np.asarray(str(fingerprint))


def _load_contract_record(archive, prefix):
    names = tuple(str(value) for value in archive[f"{prefix}_names"])
    record = tuple(
        (name, _immutable_array(archive[f"{prefix}_value_{index}"], None))
        for index, name in enumerate(names)
    )
    digest = hashlib.sha256()
    for name, value in record:
        _update_array_digest(digest, name, value)
    found = digest.hexdigest()
    stored = str(archive[f"{prefix}_fingerprint"])
    if found != stored:
        raise ValueError("persisted contract record fingerprint is invalid")
    return record, found


def _stack_field_mapping(fields, shape, label):
    if not isinstance(fields, Mapping):
        raise TypeError(f"{label} fields must be a mapping")
    found = tuple(fields)
    if set(found) != set(NATIVE_CHANNEL_ORDER) or len(found) != len(
        NATIVE_CHANNEL_ORDER
    ):
        missing = sorted(set(NATIVE_CHANNEL_ORDER)-set(found))
        extra = sorted(set(found)-set(NATIVE_CHANNEL_ORDER))
        raise ValueError(
            f"{label} fields must contain exactly the native channels; "
            f"missing={missing}, extra={extra}"
        )
    arrays = []
    for name in NATIVE_CHANNEL_ORDER:
        array = np.asarray(fields[name], dtype=float)
        if array.shape != tuple(shape):
            raise ValueError(f"{label} field {name} has shape {array.shape}, expected {shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{label} field {name} is nonfinite")
        arrays.append(array)
    return np.stack(arrays, axis=-1)


def _validate_source_coordinates(z, r, parent_r_max):
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    parent_r_max = float(parent_r_max)
    if z.ndim != 1 or r.ndim != 1:
        raise ValueError("source coordinates must be one-dimensional")
    if len(z) < 6 or len(r) < 4:
        raise ValueError("Q53 construction requires at least six z and four r nodes")
    if (
        not np.all(np.isfinite(z))
        or not np.all(np.isfinite(r))
        or np.any(np.diff(z) <= 0.0)
        or np.any(np.diff(r) <= 0.0)
    ):
        raise ValueError("source coordinates must be finite and strictly increasing")
    if r[0] != 0.0:
        raise ValueError("the radial source grid must start at the axis")
    if parent_r_max <= 0.0 or not np.isclose(
        r[-1], parent_r_max, rtol=0.0, atol=1e-12,
    ):
        raise ValueError("source grid must cover the full parent radial domain")
    return z, r, parent_r_max


@dataclass(frozen=True)
class NativeTensorHermiteSurface:
    """One clamped tensor surface on ``(z, s=(r/R)^2)``."""

    z_knots: np.ndarray
    s_knots: np.ndarray
    coefficients: np.ndarray
    z_degree: int
    s_degree: int
    parent_r_max: float
    z_boundary: str = "clamped_row_derived_z_first"

    def __post_init__(self):
        z_knots = _immutable_array(self.z_knots)
        s_knots = _immutable_array(self.s_knots)
        coefficients = _immutable_array(self.coefficients)
        z_degree = int(self.z_degree)
        s_degree = int(self.s_degree)
        parent_r_max = float(self.parent_r_max)
        if z_knots.ndim != 1 or s_knots.ndim != 1 or coefficients.ndim != 3:
            raise ValueError("invalid native Hermite coefficient arrays")
        if z_degree not in (3, 5) or s_degree != 3 or parent_r_max <= 0.0:
            raise ValueError("native Hermite degrees must be Q53 or Q33")
        if coefficients.shape[0] != len(z_knots)-z_degree-1:
            raise ValueError("compact coefficient/knots mismatch")
        if coefficients.shape[1] != len(s_knots)-s_degree-1:
            raise ValueError("radial coefficient/knots mismatch")
        if not all(np.all(np.isfinite(item)) for item in (
            z_knots, s_knots, coefficients,
        )):
            raise ValueError("native Hermite coefficients must be finite")
        if str(self.z_boundary) != "clamped_row_derived_z_first":
            raise ValueError("native Hermite surfaces require row-derived clamping")
        object.__setattr__(self, "z_knots", z_knots)
        object.__setattr__(self, "s_knots", s_knots)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "z_degree", z_degree)
        object.__setattr__(self, "s_degree", s_degree)
        object.__setattr__(self, "parent_r_max", parent_r_max)
        object.__setattr__(self, "z_boundary", str(self.z_boundary))

    @classmethod
    def build(
        cls,
        z,
        r,
        values,
        z_first_endpoints,
        *,
        z_degree,
        parent_r_max=PARENT_R_MAX,
    ):
        z, r, parent_r_max = _validate_source_coordinates(
            z, r, parent_r_max
        )
        values = np.asarray(values, dtype=float)
        z_first_endpoints = np.asarray(z_first_endpoints, dtype=float)
        if values.ndim != 3 or values.shape[:2] != (len(z), len(r)):
            raise ValueError("surface values are not aligned with z,r")
        if z_first_endpoints.shape != (2, len(r), values.shape[-1]):
            raise ValueError("row-derived endpoint derivatives have the wrong shape")
        if not all(np.all(np.isfinite(item)) for item in (
            values, z_first_endpoints,
        )):
            raise ValueError("surface values and endpoint derivatives must be finite")
        z_degree = int(z_degree)
        if z_degree not in (3, 5):
            raise ValueError("compact degree must be five or three")

        s = (r/parent_r_max)**2
        radial = make_interp_spline(s, values, k=3, axis=1)
        lower = make_interp_spline(s, z_first_endpoints[0], k=3, axis=0)
        upper = make_interp_spline(s, z_first_endpoints[1], k=3, axis=0)
        if not (
            np.array_equal(lower.t, radial.t)
            and np.array_equal(upper.t, radial.t)
        ):
            raise RuntimeError("value and endpoint radial bases differ")
        radial_coefficients = np.moveaxis(radial.c, 0, 1)
        boundary = ([(1, lower.c)], [(1, upper.c)])
        if z_degree == 5:
            knots = np.concatenate((
                np.repeat(z[0], z_degree+1),
                z[2:-2],
                np.repeat(z[-1], z_degree+1),
            ))
            compact = make_interp_spline(
                z,
                radial_coefficients,
                k=z_degree,
                axis=0,
                t=knots,
                bc_type=boundary,
            )
        else:
            compact = make_interp_spline(
                z,
                radial_coefficients,
                k=z_degree,
                axis=0,
                bc_type=boundary,
            )
        compact_coefficients = np.asarray(compact.c).copy()
        raise_if_nonfinite_protocol125_representation_coefficients(
            compact_coefficients,
            recipe=(
                "native-tensor-Q53-compact"
                if z_degree == 5 else "native-tensor-Q33-compact"
            ),
            input_arrays={
                "source_z": z,
                "source_r": r,
                "source_values": values,
                "endpoint_z_first": z_first_endpoints,
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
            raise ValueError("evaluation coordinates must be scalar or one-dimensional")
        if (
            np.any(~np.isfinite(z))
            or np.any(~np.isfinite(r))
            or np.any(z < z_min)
            or np.any(z > z_max)
            or np.any(r < 0.0)
            or np.any(r > self.parent_r_max)
        ):
            raise ValueError("native Hermite evaluation lies outside its domain")
        return z, r

    def evaluate_s(self, z, r, *, z_order=0, s_order=0):
        """Evaluate derivatives in the native ``(z,s)`` coordinates."""
        z, r = self._coordinates(z, r)
        z_order = int(z_order)
        s_order = int(s_order)
        if not 0 <= z_order <= self.z_degree:
            raise ValueError("unsupported compact derivative order")
        if not 0 <= s_order <= self.s_degree:
            raise ValueError("unsupported squared-radius derivative order")
        s = (r/self.parent_r_max)**2
        radial = BSpline(
            self.s_knots,
            self.coefficients,
            self.s_degree,
            axis=1,
            extrapolate=False,
        )(s, nu=s_order)
        return BSpline(
            self.z_knots,
            radial,
            self.z_degree,
            axis=0,
            extrapolate=False,
        )(z, nu=z_order)

    def evaluate(self, z, r, *, z_order=0, r_order=0):
        """Evaluate derivatives in the physical ``(z,r)`` coordinates."""
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
        raise ValueError("physical radial derivative order must be zero, one, or two")

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
class NativeMetricHermiteState:
    """Position or acceleration surfaces in the native physical basis."""

    channels: NativeTensorHermiteSurface
    anisotropy_numerator: NativeTensorHermiteSurface
    z_first_endpoints: np.ndarray
    state_name: str

    def __post_init__(self):
        endpoints = _immutable_array(self.z_first_endpoints)
        state_name = str(self.state_name)
        if self.channels.coefficients.shape[-1] != len(NATIVE_CHANNEL_ORDER):
            raise ValueError("native channel surface has the wrong field count")
        if self.anisotropy_numerator.coefficients.shape[-1] != 1:
            raise ValueError("anisotropy numerator must have one field lane")
        if endpoints.ndim != 3 or endpoints.shape[0] != 2:
            raise ValueError("stored endpoint derivative array has the wrong shape")
        if endpoints.shape[-1] != len(NATIVE_CHANNEL_ORDER):
            raise ValueError("stored endpoint derivative field count mismatch")
        left = self.channels
        right = self.anisotropy_numerator
        if (
            left.z_degree != right.z_degree
            or left.s_degree != right.s_degree
            or left.parent_r_max != right.parent_r_max
            or not np.array_equal(left.z_knots, right.z_knots)
            or not np.array_equal(left.s_knots, right.s_knots)
        ):
            raise ValueError("native channels and anisotropy use different bases")
        if state_name not in ("position", "acceleration"):
            raise ValueError("native Hermite state must be position or acceleration")
        object.__setattr__(self, "z_first_endpoints", endpoints)
        object.__setattr__(self, "state_name", state_name)

    @property
    def z_degree(self):
        return self.channels.z_degree

    @property
    def s_degree(self):
        return self.channels.s_degree

    @property
    def parent_r_max(self):
        return self.channels.parent_r_max

    def evaluate_physical_channels(self, z, r, *, z_order=0, r_order=0):
        return self.channels.evaluate(
            z, r, z_order=z_order, r_order=r_order,
        )

    def evaluate_anisotropy_numerator(self, z, r, *, z_order=0, r_order=0):
        return self.anisotropy_numerator.evaluate(
            z, r, z_order=z_order, r_order=r_order,
        )[:, :, 0]

    def evaluate_q4(self, z, r, *, z_order=0, r_order=0):
        """Evaluate ``(h_rr-h_perp)/r^2`` with analytic axis limits."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        r = np.atleast_1d(np.asarray(r, dtype=float))
        self.channels._coordinates(z, r)
        z_order = int(z_order)
        r_order = int(r_order)
        if r_order not in (0, 1, 2):
            raise ValueError("q4 radial derivative order must be zero, one, or two")
        scale2 = self.parent_r_max**2
        s = (r/self.parent_r_max)**2
        positive = r > 0.0
        result = np.empty((len(z), len(r)), dtype=float)
        numerator = self.anisotropy_numerator.evaluate_s(
            z, r, z_order=z_order, s_order=0,
        )[:, :, 0]
        numerator_s = self.anisotropy_numerator.evaluate_s(
            z, r, z_order=z_order, s_order=1,
        )[:, :, 0]

        if r_order == 0:
            result[:, positive] = (
                numerator[:, positive]/(scale2*s[positive][None, :])
            )
            result[:, ~positive] = numerator_s[:, ~positive]/scale2
            return result

        if r_order == 1:
            if np.any(positive):
                q_s = (
                    s[positive][None, :]*numerator_s[:, positive]
                    - numerator[:, positive]
                )/(scale2*s[positive][None, :]**2)
                result[:, positive] = q_s*(2.0*r[positive][None, :]/scale2)
            result[:, ~positive] = 0.0
            return result

        numerator_ss = self.anisotropy_numerator.evaluate_s(
            z, r, z_order=z_order, s_order=2,
        )[:, :, 0]
        if np.any(positive):
            positive_s = s[positive][None, :]
            q_s = (
                positive_s*numerator_s[:, positive]-numerator[:, positive]
            )/(scale2*positive_s**2)
            q_ss = (
                positive_s**2*numerator_ss[:, positive]
                - 2.0*positive_s*numerator_s[:, positive]
                + 2.0*numerator[:, positive]
            )/(scale2*positive_s**3)
            ds_dr = 2.0*r[positive][None, :]/scale2
            result[:, positive] = q_ss*ds_dr**2+q_s*(2.0/scale2)
        result[:, ~positive] = numerator_ss[:, ~positive]/scale2**2
        return result

    def evaluate_reduced(self, z, r, *, z_order=0, r_order=0):
        """Return complete regular-SO(3) reduced lanes in production order."""
        channels = self.evaluate_physical_channels(
            z, r, z_order=z_order, r_order=r_order,
        )
        result = np.zeros((*channels.shape[:2], len(REDUCED_FIELD_ORDER)))
        indices = {name: index for index, name in enumerate(NATIVE_CHANNEL_ORDER)}
        result[:, :, 1] = channels[:, :, indices["v_z"]]
        result[:, :, 2] = channels[:, :, indices["h00"]]
        result[:, :, 3] = channels[:, :, indices["h_perp"]]
        result[:, :, 4] = self.evaluate_q4(
            z, r, z_order=z_order, r_order=r_order,
        )
        result[:, :, 5] = channels[:, :, indices["v_0"]]
        result[:, :, 6] = channels[:, :, indices["h_zz"]]
        result[:, :, 7] = channels[:, :, indices["Phi"]]
        result[:, :, 8] = channels[:, :, indices["chi"]]
        return result

    def evaluate_coordinate_components(self, z, r, *, z_order=0, r_order=0):
        """Return physical metric/scalar lanes used by correction norms."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        r = np.atleast_1d(np.asarray(r, dtype=float))
        channels = self.evaluate_physical_channels(
            z, r, z_order=z_order, r_order=r_order,
        )
        indices = {name: index for index, name in enumerate(NATIVE_CHANNEL_ORDER)}
        result = np.zeros((*channels.shape[:2], len(COORDINATE_COMPONENT_ORDER)))

        def radius_times(channel_name):
            if r_order == 0:
                coefficient = channels[:, :, indices[channel_name]]
                return coefficient*r[None, :]
            if r_order == 1:
                coefficient = self.channels.evaluate(
                    z, r, z_order=z_order, r_order=0,
                )[:, :, indices[channel_name]]
                coefficient_r = channels[:, :, indices[channel_name]]
                return coefficient+coefficient_r*r[None, :]
            if r_order == 2:
                coefficient_r = self.channels.evaluate(
                    z, r, z_order=z_order, r_order=1,
                )[:, :, indices[channel_name]]
                coefficient_rr = channels[:, :, indices[channel_name]]
                return 2.0*coefficient_r+coefficient_rr*r[None, :]
            raise ValueError("coordinate radial derivative order must be zero, one, or two")

        result[:, :, 1] = radius_times("v_z")
        result[:, :, 2] = channels[:, :, indices["h00"]]
        result[:, :, 3] = channels[:, :, indices["h_perp"]]
        result[:, :, 4] = channels[:, :, indices["h_rr"]]
        result[:, :, 5] = radius_times("v_0")
        result[:, :, 6] = channels[:, :, indices["h_zz"]]
        result[:, :, 7] = channels[:, :, indices["Phi"]]
        result[:, :, 8] = channels[:, :, indices["chi"]]
        return result

    def coefficient_arrays(self, prefix):
        return {
            **self.channels.coefficient_arrays(f"{prefix}_channels"),
            **self.anisotropy_numerator.coefficient_arrays(
                f"{prefix}_anisotropy_numerator"
            ),
            f"{prefix}_z_first_endpoints": self.z_first_endpoints.copy(),
            f"{prefix}_state_name": np.asarray(self.state_name),
        }

    @classmethod
    def from_arrays(cls, archive, prefix):
        return cls(
            NativeTensorHermiteSurface.from_arrays(
                archive, f"{prefix}_channels"
            ),
            NativeTensorHermiteSurface.from_arrays(
                archive, f"{prefix}_anisotropy_numerator"
            ),
            np.asarray(archive[f"{prefix}_z_first_endpoints"]),
            str(archive[f"{prefix}_state_name"]),
        )


@dataclass(frozen=True)
class _NativeMetricNodalBundle:
    z: np.ndarray
    r: np.ndarray
    position: np.ndarray
    acceleration: np.ndarray
    position_z_endpoints: np.ndarray
    acceleration_z_endpoints: np.ndarray
    parent_r_max: float
    source_fingerprint: str
    endpoint_fingerprint: str

    @classmethod
    def build(
        cls,
        z,
        r,
        position_fields,
        acceleration_fields,
        position_z_endpoints,
        acceleration_z_endpoints,
        *,
        parent_r_max,
    ):
        z, r, parent_r_max = _validate_source_coordinates(
            z, r, parent_r_max
        )
        position = _stack_field_mapping(
            position_fields, (len(z), len(r)), "position",
        )
        acceleration = _stack_field_mapping(
            acceleration_fields, (len(z), len(r)), "acceleration",
        )
        position_z = _stack_field_mapping(
            position_z_endpoints, (2, len(r)), "position endpoint",
        )
        acceleration_z = _stack_field_mapping(
            acceleration_z_endpoints, (2, len(r)), "acceleration endpoint",
        )
        h_perp = NATIVE_CHANNEL_ORDER.index("h_perp")
        h_rr = NATIVE_CHANNEL_ORDER.index("h_rr")
        for label, values in (
            ("position", position),
            ("acceleration", acceleration),
        ):
            if not np.array_equal(values[:, 0, h_perp], values[:, 0, h_rr]):
                raise ValueError(f"{label} h_rr and h_perp must agree exactly at the axis")
        for label, values in (
            ("position endpoint", position_z),
            ("acceleration endpoint", acceleration_z),
        ):
            if not np.array_equal(values[:, 0, h_perp], values[:, 0, h_rr]):
                raise ValueError(
                    f"{label} h_rr and h_perp derivatives must agree exactly at the axis"
                )
        source_fingerprint = _fingerprint_named_arrays(
            z=z,
            r=r,
            position=position,
            acceleration=acceleration,
        )
        endpoint_fingerprint = _fingerprint_named_arrays(
            position_z_endpoints=position_z,
            acceleration_z_endpoints=acceleration_z,
        )
        return cls(
            _immutable_array(z),
            _immutable_array(r),
            _immutable_array(position),
            _immutable_array(acceleration),
            _immutable_array(position_z),
            _immutable_array(acceleration_z),
            parent_r_max,
            source_fingerprint,
            endpoint_fingerprint,
        )


@dataclass(frozen=True)
class NativeMetricHermiteRepresentation:
    """One Q53 or Q33 native-metric position/acceleration representation."""

    position: NativeMetricHermiteState
    acceleration: NativeMetricHermiteState
    source_z: np.ndarray
    source_r: np.ndarray
    recipe: str
    source_fingerprint: str
    endpoint_fingerprint: str
    channel_order: tuple = NATIVE_CHANNEL_ORDER

    def __post_init__(self):
        source_z = _immutable_array(self.source_z)
        source_r = _immutable_array(self.source_r)
        recipe = str(self.recipe)
        channel_order = tuple(str(name) for name in self.channel_order)
        expected_recipe = {5: "primary-Q53", 3: "comparator-Q33"}
        if self.position.z_degree != self.acceleration.z_degree:
            raise ValueError("position and acceleration compact degrees differ")
        if recipe != expected_recipe.get(self.position.z_degree):
            raise ValueError("native Hermite recipe does not match compact degree")
        if self.position.s_degree != 3 or self.acceleration.s_degree != 3:
            raise ValueError("native Hermite radial degree must be three")
        if self.position.parent_r_max != self.acceleration.parent_r_max:
            raise ValueError("position and acceleration radial scales differ")
        if channel_order != NATIVE_CHANNEL_ORDER:
            raise ValueError("native Hermite channel order is not canonical")
        if source_z.ndim != 1 or source_r.ndim != 1:
            raise ValueError("stored source coordinates are invalid")
        expected_endpoint_shape = (
            2, len(source_r), len(NATIVE_CHANNEL_ORDER),
        )
        if (
            self.position.z_first_endpoints.shape != expected_endpoint_shape
            or self.acceleration.z_first_endpoints.shape
            != expected_endpoint_shape
        ):
            raise ValueError("stored endpoints do not match the source radial grid")
        object.__setattr__(self, "source_z", source_z)
        object.__setattr__(self, "source_r", source_r)
        object.__setattr__(self, "recipe", recipe)
        object.__setattr__(self, "source_fingerprint", str(self.source_fingerprint))
        object.__setattr__(self, "endpoint_fingerprint", str(self.endpoint_fingerprint))
        object.__setattr__(self, "channel_order", channel_order)

    @classmethod
    def _from_bundle(cls, bundle, z_degree):
        h_perp = NATIVE_CHANNEL_ORDER.index("h_perp")
        h_rr = NATIVE_CHANNEL_ORDER.index("h_rr")

        def build_state(values, endpoints, name):
            numerator = values[:, :, h_rr]-values[:, :, h_perp]
            numerator_z = endpoints[:, :, h_rr]-endpoints[:, :, h_perp]
            channels = NativeTensorHermiteSurface.build(
                bundle.z,
                bundle.r,
                values,
                endpoints,
                z_degree=z_degree,
                parent_r_max=bundle.parent_r_max,
            )
            anisotropy = NativeTensorHermiteSurface.build(
                bundle.z,
                bundle.r,
                numerator[:, :, None],
                numerator_z[:, :, None],
                z_degree=z_degree,
                parent_r_max=bundle.parent_r_max,
            )
            return NativeMetricHermiteState(
                channels, anisotropy, endpoints, name,
            )

        return cls(
            build_state(
                bundle.position, bundle.position_z_endpoints, "position",
            ),
            build_state(
                bundle.acceleration,
                bundle.acceleration_z_endpoints,
                "acceleration",
            ),
            bundle.z,
            bundle.r,
            "primary-Q53" if z_degree == 5 else "comparator-Q33",
            bundle.source_fingerprint,
            bundle.endpoint_fingerprint,
        )

    def state(self, name):
        name = str(name)
        if name == "position":
            return self.position
        if name == "acceleration":
            return self.acceleration
        raise ValueError("state name must be position or acceleration")

    def coefficient_arrays(self, prefix="representation"):
        output = {
            **self.position.coefficient_arrays(f"{prefix}_position"),
            **self.acceleration.coefficient_arrays(f"{prefix}_acceleration"),
            f"{prefix}_source_z": self.source_z.copy(),
            f"{prefix}_source_r": self.source_r.copy(),
            f"{prefix}_recipe": np.asarray(self.recipe),
            f"{prefix}_source_fingerprint": np.asarray(self.source_fingerprint),
            f"{prefix}_endpoint_fingerprint": np.asarray(self.endpoint_fingerprint),
            f"{prefix}_channel_order": np.asarray(self.channel_order),
        }
        return output

    def fingerprint(self):
        digest = hashlib.sha256()
        for name, value in sorted(self.coefficient_arrays().items()):
            _update_array_digest(digest, name, value)
        return digest.hexdigest()

    @classmethod
    def from_arrays(cls, archive, prefix="representation"):
        return cls(
            NativeMetricHermiteState.from_arrays(
                archive, f"{prefix}_position"
            ),
            NativeMetricHermiteState.from_arrays(
                archive, f"{prefix}_acceleration"
            ),
            np.asarray(archive[f"{prefix}_source_z"]),
            np.asarray(archive[f"{prefix}_source_r"]),
            str(archive[f"{prefix}_recipe"]),
            str(archive[f"{prefix}_source_fingerprint"]),
            str(archive[f"{prefix}_endpoint_fingerprint"]),
            tuple(str(value) for value in archive[f"{prefix}_channel_order"]),
        )


@dataclass(frozen=True)
class NativeMetricHermitePair:
    """Additive native-metric degree diagnostic with identical endpoints.

    This pair does not replace the sealed reduced Q33/Q55 adverse comparison.
    In particular, both members interpolate supplied derivative traces in
    radius and therefore neither member is the authoritative nonlinear
    dense-wall representation.
    """

    primary: NativeMetricHermiteRepresentation
    comparator: NativeMetricHermiteRepresentation

    def __post_init__(self):
        if self.primary.recipe != "primary-Q53":
            raise ValueError("primary representation must be Q53")
        if self.comparator.recipe != "comparator-Q33":
            raise ValueError("comparator representation must be Q33")
        if (
            self.primary.source_fingerprint != self.comparator.source_fingerprint
            or self.primary.endpoint_fingerprint
            != self.comparator.endpoint_fingerprint
            or not np.array_equal(self.primary.source_z, self.comparator.source_z)
            or not np.array_equal(self.primary.source_r, self.comparator.source_r)
            or not np.array_equal(
                self.primary.position.z_first_endpoints,
                self.comparator.position.z_first_endpoints,
            )
            or not np.array_equal(
                self.primary.acceleration.z_first_endpoints,
                self.comparator.acceleration.z_first_endpoints,
            )
        ):
            raise ValueError("primary and comparator do not share identical source data")

    @classmethod
    def build(
        cls,
        z,
        r,
        position_fields,
        acceleration_fields,
        position_z_endpoints,
        acceleration_z_endpoints,
        *,
        parent_r_max=PARENT_R_MAX,
    ):
        bundle = _NativeMetricNodalBundle.build(
            z,
            r,
            position_fields,
            acceleration_fields,
            position_z_endpoints,
            acceleration_z_endpoints,
            parent_r_max=parent_r_max,
        )
        return cls(
            NativeMetricHermiteRepresentation._from_bundle(bundle, 5),
            NativeMetricHermiteRepresentation._from_bundle(bundle, 3),
        )

    def coefficient_arrays(self, prefix="native_metric_hermite"):
        return {
            **self.primary.coefficient_arrays(f"{prefix}_primary"),
            **self.comparator.coefficient_arrays(f"{prefix}_comparator"),
        }

    @classmethod
    def from_arrays(cls, archive, prefix="native_metric_hermite"):
        return cls(
            NativeMetricHermiteRepresentation.from_arrays(
                archive, f"{prefix}_primary"
            ),
            NativeMetricHermiteRepresentation.from_arrays(
                archive, f"{prefix}_comparator"
            ),
        )

    @property
    def comparison_name(self):
        return ADDITIVE_COMPARATOR_NAME


def sealed_reduced_q33_q55_adverse_projections(
    jet_field, z_parent, r_parent, z_target, r_target, *, parent_identity,
):
    """Evaluate the unchanged sealed reduced Q33/Q55 adverse pair.

    Imports are local so the new native-metric implementation cannot alter
    the sealed representation module.  The returned Q33 is the historical
    clamped cubic reduced parent; Q55 is the historical two-directional
    not-a-knot ``RectBivariateSpline`` projection.
    """
    from bhps.matched_staged_continuum import (
        ContinuousReducedParent,
        quintic_adverse_projection,
    )

    q33 = ContinuousReducedParent.from_jet_field(
        jet_field,
        z_parent,
        r_parent,
        degree=3,
        parent_r_max=PARENT_R_MAX,
        parent_identity=str(parent_identity),
    ).project(z_target, r_target)
    q55 = quintic_adverse_projection(
        jet_field, z_parent, r_parent, z_target, r_target,
    )
    return {
        SEALED_ADVERSE_COMPARATOR_NAMES[0]: q33,
        SEALED_ADVERSE_COMPARATOR_NAMES[1]: q55,
    }


def _contract_identifier(contract, label):
    identifier = str(getattr(contract, "identifier", ""))
    if not identifier:
        raise ValueError(f"{label} contract requires a stable identifier")
    return identifier


@dataclass(frozen=True)
class _RadialCubicChannels:
    """Native-axis/outer-clamped cubic-in-s data before compact interpolation."""

    s_knots: np.ndarray
    coefficients: np.ndarray
    inner_s_derivative: np.ndarray
    outer_s_derivative: np.ndarray
    parent_r_max: float

    def __post_init__(self):
        s_knots = _immutable_array(self.s_knots)
        coefficients = _immutable_array(self.coefficients)
        inner_s_derivative = _immutable_array(self.inner_s_derivative)
        outer_s_derivative = _immutable_array(self.outer_s_derivative)
        parent_r_max = float(self.parent_r_max)
        if s_knots.ndim != 1 or coefficients.ndim != 3:
            raise ValueError("invalid radial-first coefficient arrays")
        if coefficients.shape[1] != len(s_knots)-4:
            raise ValueError("radial-first coefficient/knots mismatch")
        expected_derivative_shape = (
            coefficients.shape[0], coefficients.shape[-1],
        )
        if (
            inner_s_derivative.shape != expected_derivative_shape
            or outer_s_derivative.shape != expected_derivative_shape
        ):
            raise ValueError("radial derivative bundle has the wrong shape")
        if parent_r_max <= 0.0 or not all(np.all(np.isfinite(item)) for item in (
            s_knots, coefficients, inner_s_derivative, outer_s_derivative,
        )):
            raise ValueError("invalid radial-first interpolation data")
        object.__setattr__(self, "s_knots", s_knots)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "inner_s_derivative", inner_s_derivative)
        object.__setattr__(self, "outer_s_derivative", outer_s_derivative)
        object.__setattr__(self, "parent_r_max", parent_r_max)

    @classmethod
    def build(
        cls,
        r,
        values,
        inner_s_derivative,
        outer_s_derivative,
        *,
        parent_r_max,
    ):
        r = np.asarray(r, dtype=float)
        values = np.asarray(values, dtype=float)
        inner_s_derivative = np.asarray(inner_s_derivative, dtype=float)
        outer_s_derivative = np.asarray(outer_s_derivative, dtype=float)
        parent_r_max = float(parent_r_max)
        if values.ndim != 3 or values.shape[1] != len(r):
            raise ValueError("radial-first values do not match the radial grid")
        expected = (values.shape[0], values.shape[-1])
        if inner_s_derivative.shape != expected or outer_s_derivative.shape != expected:
            raise ValueError("endpoint s-derivatives do not match radial-first values")
        if not all(np.all(np.isfinite(item)) for item in (
            r, values, inner_s_derivative, outer_s_derivative,
        )):
            raise ValueError("radial-first construction inputs must be finite")
        if parent_r_max <= 0.0 or not np.isfinite(parent_r_max):
            raise ValueError("radial-first parent radius must be finite and positive")
        s = (r/parent_r_max)**2
        knots = np.concatenate((
            np.repeat(s[0], 4),
            s[1:-1],
            np.repeat(s[-1], 4),
        ))
        spline = make_interp_spline(
            s,
            values,
            k=3,
            axis=1,
            t=knots,
            bc_type=(
                [(1, inner_s_derivative)],
                [(1, outer_s_derivative)],
            ),
        )
        coefficients = np.moveaxis(spline.c, 0, 1)
        raise_if_nonfinite_protocol125_representation_coefficients(
            coefficients,
            recipe="native-radial-cubic-s",
            input_arrays={
                "source_r": r,
                "source_values": values,
                "inner_s_derivative": inner_s_derivative,
                "outer_s_derivative": outer_s_derivative,
                "parent_r_max": np.asarray(parent_r_max),
            },
        )
        return cls(
            np.asarray(spline.t).copy(),
            np.asarray(coefficients).copy(),
            np.asarray(inner_s_derivative).copy(),
            np.asarray(outer_s_derivative).copy(),
            parent_r_max,
        )

    def evaluate_s(self, r, s_order=0):
        r = np.atleast_1d(np.asarray(r, dtype=float))
        s_order = int(s_order)
        if r.ndim != 1 or np.any(~np.isfinite(r)):
            raise ValueError("radial-first query must be finite and one-dimensional")
        if np.any(r < 0.0) or np.any(r > self.parent_r_max):
            raise ValueError("radial-first query lies outside the parent")
        if not 0 <= s_order <= 3:
            raise ValueError("radial-first squared-radius order must be zero through three")
        s = (r/self.parent_r_max)**2
        result = BSpline(
            self.s_knots,
            self.coefficients,
            3,
            axis=1,
            extrapolate=False,
        )(s, nu=s_order)
        axis = np.flatnonzero(r == 0.0)
        if len(axis) and s_order in (0, 1):
            # Evaluate the exact endpoint constraints directly.  This is the
            # same stored spline, not a post-evaluation repair; it prevents a
            # BLAS cancellation from changing IEEE +0 parity into a tiny
            # signed anisotropy before the q4 quotient is taken.
            exact = (
                self.coefficients[:, 0]
                if s_order == 0 else self.inner_s_derivative
            )
            result[:, axis] = exact[:, None]
        return result

    def coefficient_arrays(self, prefix):
        return {
            f"{prefix}_s_knots": self.s_knots.copy(),
            f"{prefix}_coefficients": self.coefficients.copy(),
            f"{prefix}_inner_s_derivative": self.inner_s_derivative.copy(),
            f"{prefix}_outer_s_derivative": self.outer_s_derivative.copy(),
            f"{prefix}_parent_r_max": np.asarray(self.parent_r_max),
            f"{prefix}_boundary": np.asarray(
                "native-seven-point-inner-and-outer-clamped-cubic-s"
            ),
        }

    @classmethod
    def from_arrays(cls, archive, prefix):
        boundary = str(archive[f"{prefix}_boundary"])
        if boundary != "native-seven-point-inner-and-outer-clamped-cubic-s":
            raise ValueError("persisted radial-first boundary recipe is invalid")
        return cls(
            np.asarray(archive[f"{prefix}_s_knots"]),
            np.asarray(archive[f"{prefix}_coefficients"]),
            np.asarray(archive[f"{prefix}_inner_s_derivative"]),
            np.asarray(archive[f"{prefix}_outer_s_derivative"]),
            float(archive[f"{prefix}_parent_r_max"]),
        )


def _derived_radial_anisotropy_numerator(radial_channels):
    """Derive ``N=h_rr-h_perp`` from the stored native-channel spline.

    This is an exact coefficient-space projection, not a second spline fit.
    Keeping the numerator on the identical radial basis prevents an archive
    from independently tuning the quantity whose axis quotient defines q4.
    """
    if not isinstance(radial_channels, _RadialCubicChannels):
        raise TypeError("anisotropy derivation requires radial native channels")
    h_perp = NATIVE_CHANNEL_ORDER.index("h_perp")
    h_rr = NATIVE_CHANNEL_ORDER.index("h_rr")
    coefficients = (
        radial_channels.coefficients[:, :, h_rr]
        - radial_channels.coefficients[:, :, h_perp]
    )[:, :, None]
    inner_s_derivative = (
        radial_channels.inner_s_derivative[:, h_rr]
        - radial_channels.inner_s_derivative[:, h_perp]
    )[:, None]
    outer_s_derivative = (
        radial_channels.outer_s_derivative[:, h_rr]
        - radial_channels.outer_s_derivative[:, h_perp]
    )[:, None]
    return _RadialCubicChannels(
        radial_channels.s_knots,
        coefficients,
        inner_s_derivative,
        outer_s_derivative,
        radial_channels.parent_r_max,
    )


def _build_native_radial_channels(
    r,
    values,
    inner_s_derivative,
    outer_s_derivative,
    *,
    parent_r_max,
    native_numerator_inner_s=None,
):
    """Build physical channels through the regular ``(h_perp,N)`` basis.

    Solving the radial system with ``N=h_rr-h_perp`` as the temporary lane
    makes the stored axis numerator an exact IEEE +0 whenever the physical
    input is regular.  The returned public coefficients are converted back
    to ``h_rr`` and the persisted numerator remains a checked difference of
    those physical coefficients; no q4 lane is introduced.
    """
    values = np.asarray(values, dtype=float)
    inner = np.asarray(inner_s_derivative, dtype=float)
    outer = np.asarray(outer_s_derivative, dtype=float)
    h_perp = NATIVE_CHANNEL_ORDER.index("h_perp")
    h_rr = NATIVE_CHANNEL_ORDER.index("h_rr")
    transformed_values = values.copy()
    transformed_inner = inner.copy()
    transformed_outer = outer.copy()
    if native_numerator_inner_s is not None and np.asarray(
        native_numerator_inner_s,
    ).shape != (values.shape[0],):
        raise ValueError("native numerator axis derivative has the wrong shape")
    transformed_values[:, :, h_rr] = (
        values[:, :, h_rr]-values[:, :, h_perp]
    )
    transformed_inner[:, h_rr] = (
        inner[:, h_rr]-inner[:, h_perp]
        if native_numerator_inner_s is None
        else np.asarray(native_numerator_inner_s, dtype=float)
    )
    transformed_outer[:, h_rr] = outer[:, h_rr]-outer[:, h_perp]
    transformed = _RadialCubicChannels.build(
        r,
        transformed_values,
        transformed_inner,
        transformed_outer,
        parent_r_max=parent_r_max,
    )
    coefficients = transformed.coefficients.copy()
    restored_inner = transformed.inner_s_derivative.copy()
    restored_outer = transformed.outer_s_derivative.copy()
    coefficients[:, :, h_rr] = (
        coefficients[:, :, h_perp]+coefficients[:, :, h_rr]
    )
    restored_inner[:, h_rr] = (
        restored_inner[:, h_perp]+restored_inner[:, h_rr]
    )
    restored_outer[:, h_rr] = (
        restored_outer[:, h_perp]+restored_outer[:, h_rr]
    )
    return _RadialCubicChannels(
        transformed.s_knots,
        coefficients,
        restored_inner,
        restored_outer,
        transformed.parent_r_max,
    )


def _compact_hermite_evaluate(
    source_z,
    values,
    endpoint_z_first,
    query_z,
    *,
    z_degree,
    z_order,
):
    source_z = np.asarray(source_z, dtype=float)
    values = np.asarray(values, dtype=float)
    endpoint_z_first = np.asarray(endpoint_z_first, dtype=float)
    query_z = np.atleast_1d(np.asarray(query_z, dtype=float))
    if values.ndim != 3 or values.shape[0] != len(source_z):
        raise ValueError("radial-first compact data have the wrong shape")
    if endpoint_z_first.shape != (2, *values.shape[1:]):
        raise ValueError("query-time compact endpoint data have the wrong shape")
    if query_z.ndim != 1 or np.any(~np.isfinite(query_z)):
        raise ValueError("compact query must be finite and one-dimensional")
    if np.any(query_z < source_z[0]) or np.any(query_z > source_z[-1]):
        raise ValueError("compact query lies outside the parent")
    boundary = (
        [(1, endpoint_z_first[0])],
        [(1, endpoint_z_first[1])],
    )
    if int(z_degree) == 5:
        knots = np.concatenate((
            np.repeat(source_z[0], 6),
            source_z[2:-2],
            np.repeat(source_z[-1], 6),
        ))
        compact = make_interp_spline(
            source_z,
            values,
            k=5,
            axis=0,
            t=knots,
            bc_type=boundary,
        )
    elif int(z_degree) == 3:
        compact = make_interp_spline(
            source_z,
            values,
            k=3,
            axis=0,
            bc_type=boundary,
        )
    else:
        raise ValueError("radial-first compact degree must be five or three")
    return compact(query_z, nu=int(z_order))


@dataclass(frozen=True)
class RadialFirstConstrainedHermiteState:
    """One state evaluated by radial interpolation then native wall closure."""

    source_z: np.ndarray
    source_r: np.ndarray
    radial_channels: _RadialCubicChannels
    radial_anisotropy_numerator: _RadialCubicChannels
    stored_z_first_endpoints: np.ndarray
    compact_wall_contract: CompactWallDerivativeContract
    outer_open_face_contract: OuterOpenFaceDerivativeContract
    outer_ownership_mask: np.ndarray
    state_name: str
    z_degree: int = 5
    compact_wall_contract_record: tuple | None = None
    outer_open_face_contract_record: tuple | None = None

    def __post_init__(self):
        source_z = _immutable_array(self.source_z)
        source_r = _immutable_array(self.source_r)
        endpoints = _immutable_array(self.stored_z_first_endpoints)
        ownership = _immutable_array(self.outer_ownership_mask, bool)
        state_name = str(self.state_name)
        z_degree = int(self.z_degree)
        if source_z.ndim != 1 or source_r.ndim != 1:
            raise ValueError("radial-first source coordinates are invalid")
        if endpoints.shape != (2, len(source_r), len(NATIVE_CHANNEL_ORDER)):
            raise ValueError("radial-first stored endpoint array is invalid")
        if ownership.shape != (len(NATIVE_CHANNEL_ORDER),) or not np.any(ownership):
            raise ValueError("radial-first outer ownership mask is invalid")
        if self.radial_channels.coefficients.shape[0] != len(source_z):
            raise ValueError("radial-first channel/source compact counts differ")
        if self.radial_channels.coefficients.shape[-1] != len(NATIVE_CHANNEL_ORDER):
            raise ValueError("radial-first channel count mismatch")
        if self.radial_anisotropy_numerator.coefficients.shape != (
            self.radial_channels.coefficients.shape[0],
            self.radial_channels.coefficients.shape[1],
            1,
        ):
            raise ValueError("radial-first anisotropy basis mismatch")
        derived_anisotropy = _derived_radial_anisotropy_numerator(
            self.radial_channels,
        )
        if not (
            _arrays_bitwise_equal(
                self.radial_anisotropy_numerator.s_knots,
                derived_anisotropy.s_knots,
            )
            and _arrays_bitwise_equal(
                self.radial_anisotropy_numerator.coefficients,
                derived_anisotropy.coefficients,
            )
            and _arrays_bitwise_equal(
                self.radial_anisotropy_numerator.inner_s_derivative,
                derived_anisotropy.inner_s_derivative,
            )
            and _arrays_bitwise_equal(
                self.radial_anisotropy_numerator.outer_s_derivative,
                derived_anisotropy.outer_s_derivative,
            )
            and self.radial_anisotropy_numerator.parent_r_max
            == derived_anisotropy.parent_r_max
        ):
            raise ValueError(
                "radial anisotropy must be the exact h_rr-h_perp "
                "coefficient difference"
            )
        if state_name not in ("position", "acceleration"):
            raise ValueError("radial-first state must be position or acceleration")
        if z_degree not in (3, 5):
            raise ValueError("radial-first state must be Q53 or Q33")
        compact_id = _contract_identifier(
            self.compact_wall_contract, "compact-wall",
        )
        outer_id = _contract_identifier(
            self.outer_open_face_contract, "outer-open-face",
        )
        live_compact_record, live_compact_fingerprint = _freeze_contract_record(
            self.compact_wall_contract, "compact-wall",
        )
        live_outer_record, live_outer_fingerprint = _freeze_contract_record(
            self.outer_open_face_contract, "outer-open-face",
        )
        compact_record = (
            live_compact_record
            if self.compact_wall_contract_record is None
            else tuple(
                (str(name), _immutable_array(value, None))
                for name, value in self.compact_wall_contract_record
            )
        )
        outer_record = (
            live_outer_record
            if self.outer_open_face_contract_record is None
            else tuple(
                (str(name), _immutable_array(value, None))
                for name, value in self.outer_open_face_contract_record
            )
        )
        if not _contract_records_equal(compact_record, live_compact_record):
            raise ValueError("supplied compact-wall contract record differs")
        if not _contract_records_equal(outer_record, live_outer_record):
            raise ValueError("supplied outer-open-face contract record differs")
        object.__setattr__(self, "source_z", source_z)
        object.__setattr__(self, "source_r", source_r)
        object.__setattr__(self, "stored_z_first_endpoints", endpoints)
        object.__setattr__(self, "outer_ownership_mask", ownership)
        object.__setattr__(self, "state_name", state_name)
        object.__setattr__(self, "z_degree", z_degree)
        object.__setattr__(self, "compact_wall_contract_id", compact_id)
        object.__setattr__(self, "outer_open_face_contract_id", outer_id)
        object.__setattr__(self, "compact_wall_contract_record", compact_record)
        object.__setattr__(self, "outer_open_face_contract_record", outer_record)
        object.__setattr__(
            self, "compact_wall_contract_fingerprint", live_compact_fingerprint,
        )
        object.__setattr__(
            self, "outer_open_face_contract_fingerprint", live_outer_fingerprint,
        )

        recomputed = self._wall_z_first_s_jets(source_r, 0)[0]
        scale = max(
            1.0,
            float(np.max(np.abs(recomputed))),
            float(np.max(np.abs(endpoints))),
        )
        mismatch = float(np.max(np.abs(recomputed-endpoints))/scale)
        if mismatch > 1e-12:
            raise ValueError(
                "stored row-derived endpoint data disagree with the compact-wall contract"
            )

    @property
    def parent_r_max(self):
        return self.radial_channels.parent_r_max

    @classmethod
    def build_position(
        cls,
        z,
        r,
        position_fields,
        position_z_endpoints,
        *,
        compact_wall_contract,
        outer_open_face_contract,
        parent_r_max=PARENT_R_MAX,
        z_degree=5,
    ):
        """Build a constrained position state without acceleration data.

        This is the public Phase-A position gate.  It consumes only completed
        native position values, their independently recomputed compact-wall
        endpoint derivatives, and position-capable boundary contracts.  In
        particular, no acceleration array, acceleration endpoint trace, or
        dummy acceleration outer bundle is accepted by this API.
        """
        z, r, parent_r_max = _validate_source_coordinates(
            z, r, parent_r_max,
        )
        values = _stack_field_mapping(
            position_fields, (len(z), len(r)), "position",
        )
        endpoints = _stack_field_mapping(
            position_z_endpoints, (2, len(r)), "position endpoint",
        )
        h_perp = NATIVE_CHANNEL_ORDER.index("h_perp")
        h_rr = NATIVE_CHANNEL_ORDER.index("h_rr")
        if not np.array_equal(values[:, 0, h_perp], values[:, 0, h_rr]):
            raise ValueError(
                "position h_rr and h_perp must agree exactly at the axis"
            )
        if not np.array_equal(
            endpoints[:, 0, h_perp], endpoints[:, 0, h_rr],
        ):
            raise ValueError(
                "position endpoint h_rr and h_perp derivatives must agree "
                "exactly at the axis"
            )
        return _build_radial_first_constrained_state(
            z,
            r,
            values,
            endpoints,
            state_name="position",
            compact_wall_contract=compact_wall_contract,
            outer_open_face_contract=outer_open_face_contract,
            parent_r_max=parent_r_max,
            z_degree=z_degree,
        )

    def _assert_contract_records_unchanged(self):
        compact_record, compact_fingerprint = _freeze_contract_record(
            self.compact_wall_contract, "compact-wall",
        )
        outer_record, outer_fingerprint = _freeze_contract_record(
            self.outer_open_face_contract, "outer-open-face",
        )
        if (
            compact_fingerprint != self.compact_wall_contract_fingerprint
            or not _contract_records_equal(
                compact_record, self.compact_wall_contract_record,
            )
        ):
            raise ValueError("live compact-wall contract record changed")
        if (
            outer_fingerprint != self.outer_open_face_contract_fingerprint
            or not _contract_records_equal(
                outer_record, self.outer_open_face_contract_record,
            )
        ):
            raise ValueError("live outer-open-face contract record changed")

    def _wall_z_first_s_jets(self, r, maximum_s_order):
        self._assert_contract_records_unchanged()
        maximum_s_order = int(maximum_s_order)
        radius = np.atleast_1d(np.asarray(r, dtype=float))
        value_jets = tuple(
            self.radial_channels.evaluate_s(r, order)[[0, -1]]
            for order in range(maximum_s_order+1)
        )
        returned = self.compact_wall_contract.z_first_s_jets(
            state_name=self.state_name,
            radius=radius.copy(),
            wall_value_s_jets=value_jets,
        )
        try:
            returned = tuple(np.asarray(value, dtype=float) for value in returned)
        except TypeError as error:
            raise ValueError("compact-wall contract must return a derivative tuple") from error
        if len(returned) != len(value_jets):
            raise ValueError("compact-wall contract returned the wrong derivative depth")
        for order, (found, expected) in enumerate(zip(returned, value_jets)):
            if found.shape != expected.shape or not np.all(np.isfinite(found)):
                raise ValueError(
                    f"compact-wall contract returned invalid s-order {order} data"
                )
        axis_columns = np.flatnonzero(radius == 0.0)
        if len(axis_columns):
            h_perp = NATIVE_CHANNEL_ORDER.index("h_perp")
            h_rr = NATIVE_CHANNEL_ORDER.index("h_rr")
            if not np.array_equal(
                returned[0][:, axis_columns, h_perp],
                returned[0][:, axis_columns, h_rr],
            ):
                raise ValueError(
                    "compact-wall contract anisotropy z trace is nonregular at the axis"
                )
        return returned

    def _evaluate_channel_s(self, z, r, *, z_order, s_order):
        values = self.radial_channels.evaluate_s(r, s_order)
        endpoint = self._wall_z_first_s_jets(r, s_order)[s_order]
        return _compact_hermite_evaluate(
            self.source_z,
            values,
            endpoint,
            z,
            z_degree=self.z_degree,
            z_order=z_order,
        )

    def _evaluate_anisotropy_s(self, z, r, *, z_order, s_order):
        values = self.radial_anisotropy_numerator.evaluate_s(r, s_order)
        endpoint_channels = self._wall_z_first_s_jets(r, s_order)[s_order]
        h_perp = NATIVE_CHANNEL_ORDER.index("h_perp")
        h_rr = NATIVE_CHANNEL_ORDER.index("h_rr")
        endpoint = (
            endpoint_channels[:, :, h_rr]-endpoint_channels[:, :, h_perp]
        )[:, :, None]
        return _compact_hermite_evaluate(
            self.source_z,
            values,
            endpoint,
            z,
            z_degree=self.z_degree,
            z_order=z_order,
        )[:, :, 0]

    def _outer_open_mask(self, z):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        tolerance = 32.0*np.finfo(float).eps*max(
            1.0, abs(float(self.source_z[-1])),
        )
        return (
            (z > self.source_z[0]+tolerance)
            & (z < self.source_z[-1]-tolerance)
        )

    def _outer_r_first_z_jets(self, z, maximum_z_order):
        self._assert_contract_records_unchanged()
        z = np.atleast_1d(np.asarray(z, dtype=float))
        value_jets = tuple(
            self._evaluate_channel_s(
                z,
                np.asarray([self.parent_r_max]),
                z_order=order,
                s_order=0,
            )[:, 0]
            for order in range(int(maximum_z_order)+1)
        )
        contract_result = self.outer_open_face_contract.r_first_z_jets(
            state_name=self.state_name,
            compact_coordinate=z.copy(),
            outer_value_z_jets=value_jets,
        )
        if not isinstance(contract_result, OuterOpenFaceDerivativeResult):
            raise ValueError(
                "outer-open-face contract must return OuterOpenFaceDerivativeResult"
            )
        if not np.array_equal(
            contract_result.ownership_mask, self.outer_ownership_mask,
        ):
            raise ValueError("outer-open-face contract ownership mask changed")
        returned = contract_result.r_first_z_jets
        if len(returned) != len(value_jets):
            raise ValueError("outer-open-face contract returned the wrong derivative depth")
        for order, (found, expected) in enumerate(zip(returned, value_jets)):
            if found.shape != expected.shape or not np.all(np.isfinite(found)):
                raise ValueError(
                    f"outer-open-face contract returned invalid z-order {order} data"
                )
        return returned, contract_result.ownership_mask

    def evaluate_physical_channels(self, z, r, *, z_order=0, r_order=0):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        r = np.atleast_1d(np.asarray(r, dtype=float))
        z_order = int(z_order)
        r_order = int(r_order)
        if not 0 <= z_order <= 2:
            raise ValueError("constrained compact derivative order must be zero through two")
        if r_order == 0:
            result = self._evaluate_channel_s(
                z, r, z_order=z_order, s_order=0,
            )
        elif r_order == 1:
            result = self._evaluate_channel_s(
                z, r, z_order=z_order, s_order=1,
            )*(2.0*r[None, :, None]/self.parent_r_max**2)
        elif r_order == 2:
            first = self._evaluate_channel_s(
                z, r, z_order=z_order, s_order=1,
            )
            second = self._evaluate_channel_s(
                z, r, z_order=z_order, s_order=2,
            )
            ds_dr = 2.0*r/self.parent_r_max**2
            result = (
                second*ds_dr[None, :, None]**2
                + first*(2.0/self.parent_r_max**2)
            )
        else:
            raise ValueError("constrained radial derivative order must be zero through two")
        if not np.all(np.isfinite(result)):
            raise RuntimeError("constrained native-metric evaluation is nonfinite")
        return result

    def outer_open_face_residual(self, z):
        """Report, without repair, the dense nonlinear outer-row residual."""
        z = np.atleast_1d(np.asarray(z, dtype=float))
        outer_r = np.asarray([self.parent_r_max])
        found = self.evaluate_physical_channels(
            z, outer_r, r_order=1,
        )[:, 0]
        target_jets, ownership = self._outer_r_first_z_jets(z, 0)
        target = target_jets[0]
        open_mask = self._outer_open_mask(z)
        residual = found-target
        active = residual[np.ix_(open_mask, ownership)]
        active_found = found[np.ix_(open_mask, ownership)]
        active_target = target[np.ix_(open_mask, ownership)]
        if active.size:
            scale = np.maximum(1.0, np.abs(active_found)+np.abs(active_target))
            maximum_normalized = float(np.max(np.abs(active)/scale))
            maximum_absolute = float(np.max(np.abs(active)))
        else:
            maximum_normalized = 0.0
            maximum_absolute = 0.0
        return {
            "found_r_first": _immutable_array(found),
            "contract_target_r_first": _immutable_array(target),
            "residual": _immutable_array(residual),
            "ownership_mask": _immutable_array(ownership, bool),
            "open_compact_mask": _immutable_array(open_mask, bool),
            "maximum_normalized": maximum_normalized,
            "maximum_absolute": maximum_absolute,
        }

    def evaluate_anisotropy_numerator(self, z, r, *, z_order=0, r_order=0):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        r = np.atleast_1d(np.asarray(r, dtype=float))
        r_order = int(r_order)
        if r_order == 0:
            return self._evaluate_anisotropy_s(
                z, r, z_order=z_order, s_order=0,
            )
        if r_order == 1:
            return self._evaluate_anisotropy_s(
                z, r, z_order=z_order, s_order=1,
            )*(2.0*r[None, :]/self.parent_r_max**2)
        if r_order == 2:
            first = self._evaluate_anisotropy_s(
                z, r, z_order=z_order, s_order=1,
            )
            second = self._evaluate_anisotropy_s(
                z, r, z_order=z_order, s_order=2,
            )
            ds_dr = 2.0*r/self.parent_r_max**2
            return (
                second*ds_dr[None, :]**2
                + first*(2.0/self.parent_r_max**2)
            )
        raise ValueError("anisotropy radial derivative order must be zero through two")

    def evaluate_q4(self, z, r, *, z_order=0, r_order=0):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        r = np.atleast_1d(np.asarray(r, dtype=float))
        z_order = int(z_order)
        r_order = int(r_order)
        if r_order not in (0, 1, 2):
            raise ValueError("q4 radial derivative order must be zero through two")
        scale2 = self.parent_r_max**2
        s = (r/self.parent_r_max)**2
        positive = r > 0.0
        result = np.empty((len(z), len(r)))
        numerator = self._evaluate_anisotropy_s(
            z, r, z_order=z_order, s_order=0,
        )
        if np.any(~positive):
            axis_numerator = numerator[:, ~positive]
            if not (
                np.array_equal(axis_numerator, np.zeros_like(axis_numerator))
                and not np.any(np.signbit(axis_numerator))
            ):
                raise RuntimeError(
                    "anisotropy numerator must be exact positive zero at the axis"
                )
        numerator_s = self._evaluate_anisotropy_s(
            z, r, z_order=z_order, s_order=1,
        )
        if r_order == 0:
            result[:, positive] = numerator[:, positive]/(
                scale2*s[positive][None, :]
            )
            result[:, ~positive] = numerator_s[:, ~positive]/scale2
            return result
        if r_order == 1:
            if np.any(positive):
                q_s = (
                    s[positive][None, :]*numerator_s[:, positive]
                    - numerator[:, positive]
                )/(scale2*s[positive][None, :]**2)
                result[:, positive] = q_s*(2.0*r[positive][None, :]/scale2)
            result[:, ~positive] = 0.0
            return result
        numerator_ss = self._evaluate_anisotropy_s(
            z, r, z_order=z_order, s_order=2,
        )
        if np.any(positive):
            positive_s = s[positive][None, :]
            q_s = (
                positive_s*numerator_s[:, positive]-numerator[:, positive]
            )/(scale2*positive_s**2)
            q_ss = (
                positive_s**2*numerator_ss[:, positive]
                - 2.0*positive_s*numerator_s[:, positive]
                + 2.0*numerator[:, positive]
            )/(scale2*positive_s**3)
            ds_dr = 2.0*r[positive][None, :]/scale2
            result[:, positive] = q_ss*ds_dr**2+q_s*(2.0/scale2)
        result[:, ~positive] = numerator_ss[:, ~positive]/scale2**2
        return result

    def evaluate_reduced(self, z, r, *, z_order=0, r_order=0):
        channels = self.evaluate_physical_channels(
            z, r, z_order=z_order, r_order=r_order,
        )
        result = np.zeros((*channels.shape[:2], len(REDUCED_FIELD_ORDER)))
        indices = {name: index for index, name in enumerate(NATIVE_CHANNEL_ORDER)}
        result[:, :, 1] = channels[:, :, indices["v_z"]]
        result[:, :, 2] = channels[:, :, indices["h00"]]
        result[:, :, 3] = channels[:, :, indices["h_perp"]]
        result[:, :, 4] = self.evaluate_q4(
            z, r, z_order=z_order, r_order=r_order,
        )
        result[:, :, 5] = channels[:, :, indices["v_0"]]
        result[:, :, 6] = channels[:, :, indices["h_zz"]]
        result[:, :, 7] = channels[:, :, indices["Phi"]]
        result[:, :, 8] = channels[:, :, indices["chi"]]
        return result

    def evaluate_coordinate_components(self, z, r, *, z_order=0, r_order=0):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        r = np.atleast_1d(np.asarray(r, dtype=float))
        channels = self.evaluate_physical_channels(
            z, r, z_order=z_order, r_order=r_order,
        )
        indices = {name: index for index, name in enumerate(NATIVE_CHANNEL_ORDER)}
        result = np.zeros((*channels.shape[:2], len(COORDINATE_COMPONENT_ORDER)))

        def radius_times(channel_name):
            if int(r_order) == 0:
                return channels[:, :, indices[channel_name]]*r[None, :]
            if int(r_order) == 1:
                coefficient = self.evaluate_physical_channels(
                    z, r, z_order=z_order, r_order=0,
                )[:, :, indices[channel_name]]
                return coefficient+channels[:, :, indices[channel_name]]*r[None, :]
            if int(r_order) == 2:
                coefficient_r = self.evaluate_physical_channels(
                    z, r, z_order=z_order, r_order=1,
                )[:, :, indices[channel_name]]
                return 2.0*coefficient_r+channels[:, :, indices[channel_name]]*r[None, :]
            raise ValueError("coordinate radial derivative order must be zero through two")

        result[:, :, 1] = radius_times("v_z")
        result[:, :, 2] = channels[:, :, indices["h00"]]
        result[:, :, 3] = channels[:, :, indices["h_perp"]]
        result[:, :, 4] = channels[:, :, indices["h_rr"]]
        result[:, :, 5] = radius_times("v_0")
        result[:, :, 6] = channels[:, :, indices["h_zz"]]
        result[:, :, 7] = channels[:, :, indices["Phi"]]
        result[:, :, 8] = channels[:, :, indices["chi"]]
        return result

    def coefficient_arrays(self, prefix):
        output = {
            **self.radial_channels.coefficient_arrays(
                f"{prefix}_radial_channels"
            ),
            **self.radial_anisotropy_numerator.coefficient_arrays(
                f"{prefix}_radial_anisotropy_numerator"
            ),
            f"{prefix}_source_z": self.source_z.copy(),
            f"{prefix}_source_r": self.source_r.copy(),
            f"{prefix}_stored_z_first_endpoints": (
                self.stored_z_first_endpoints.copy()
            ),
            f"{prefix}_state_name": np.asarray(self.state_name),
            f"{prefix}_z_degree": np.asarray(self.z_degree),
            f"{prefix}_compact_wall_contract_id": np.asarray(
                self.compact_wall_contract_id
            ),
            f"{prefix}_outer_open_face_contract_id": np.asarray(
                self.outer_open_face_contract_id
            ),
            f"{prefix}_outer_ownership_mask": self.outer_ownership_mask.copy(),
        }
        _persist_contract_record(
            output,
            f"{prefix}_compact_wall_contract_record",
            self.compact_wall_contract_record,
            self.compact_wall_contract_fingerprint,
        )
        _persist_contract_record(
            output,
            f"{prefix}_outer_open_face_contract_record",
            self.outer_open_face_contract_record,
            self.outer_open_face_contract_fingerprint,
        )
        return output

    def fingerprint(self, prefix="radial_first_constrained_state"):
        digest = hashlib.sha256()
        for name, value in sorted(self.coefficient_arrays(prefix).items()):
            _update_array_digest(digest, name, value)
        return digest.hexdigest()

    @classmethod
    def from_arrays(
        cls,
        archive,
        prefix,
        *,
        compact_wall_contract,
        outer_open_face_contract,
    ):
        stored_compact = str(archive[f"{prefix}_compact_wall_contract_id"])
        stored_outer = str(archive[f"{prefix}_outer_open_face_contract_id"])
        supplied_compact = _contract_identifier(
            compact_wall_contract, "compact-wall",
        )
        supplied_outer = _contract_identifier(
            outer_open_face_contract, "outer-open-face",
        )
        if stored_compact != supplied_compact:
            raise ValueError("persisted compact-wall contract identifier differs")
        if stored_outer != supplied_outer:
            raise ValueError("persisted outer-open-face contract identifier differs")
        z_degree = int(archive[f"{prefix}_z_degree"])
        if z_degree not in (3, 5):
            raise ValueError("persisted constrained compact degree is invalid")
        compact_record, _ = _load_contract_record(
            archive, f"{prefix}_compact_wall_contract_record",
        )
        outer_record, _ = _load_contract_record(
            archive, f"{prefix}_outer_open_face_contract_record",
        )
        return cls(
            np.asarray(archive[f"{prefix}_source_z"]),
            np.asarray(archive[f"{prefix}_source_r"]),
            _RadialCubicChannels.from_arrays(
                archive, f"{prefix}_radial_channels"
            ),
            _RadialCubicChannels.from_arrays(
                archive, f"{prefix}_radial_anisotropy_numerator"
            ),
            np.asarray(archive[f"{prefix}_stored_z_first_endpoints"]),
            compact_wall_contract,
            outer_open_face_contract,
            np.asarray(archive[f"{prefix}_outer_ownership_mask"]),
            str(archive[f"{prefix}_state_name"]),
            z_degree,
            compact_record,
            outer_record,
        )


def _build_radial_first_constrained_state(
    z,
    r,
    values,
    endpoints,
    *,
    state_name,
    compact_wall_contract,
    outer_open_face_contract,
    parent_r_max,
    z_degree,
):
    """Construct one radial-first state from already validated nodal data."""
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    values = np.asarray(values, dtype=float)
    endpoints = np.asarray(endpoints, dtype=float)
    expected_values = (len(z), len(r), len(NATIVE_CHANNEL_ORDER))
    expected_endpoints = (2, len(r), len(NATIVE_CHANNEL_ORDER))
    if values.shape != expected_values or endpoints.shape != expected_endpoints:
        raise ValueError("radial-first state nodal data have the wrong shape")
    if not all(np.all(np.isfinite(item)) for item in (values, endpoints)):
        raise ValueError("radial-first state nodal data must be finite")
    if len(r) < 7:
        raise ValueError(
            "radial-first constrained state requires the frozen seven-point "
            "axis operator"
        )
    state_name = str(state_name)
    if state_name not in ("position", "acceleration"):
        raise ValueError("radial-first state must be position or acceleration")
    z_degree = int(z_degree)
    if z_degree not in (3, 5):
        raise ValueError("radial-first compact degree must be five or three")

    compact_record, compact_fingerprint = _freeze_contract_record(
        compact_wall_contract, "compact-wall",
    )
    outer_record, outer_fingerprint = _freeze_contract_record(
        outer_open_face_contract, "outer-open-face",
    )
    s = (r/float(parent_r_max))**2
    axis_operator = derivative_matrix(s, 1, 7)
    inner_s = np.stack(tuple(
        (axis_operator @ values[:, :, channel].T).T[:, 0]
        for channel in range(len(NATIVE_CHANNEL_ORDER))
    ), axis=-1)
    h_perp = NATIVE_CHANNEL_ORDER.index("h_perp")
    h_rr = NATIVE_CHANNEL_ORDER.index("h_rr")
    native_numerator_inner_s = (
        axis_operator
        @ (values[:, :, h_rr]-values[:, :, h_perp]).T
    ).T[:, 0]
    # Form the metric-channel endpoint data from the physical numerator
    # limit itself.  This avoids accepting q4 as an independent lane while
    # retaining the seven-point parity definition used by native completion.
    inner_s[:, h_rr] = inner_s[:, h_perp]+native_numerator_inner_s
    baseline = make_interp_spline(s, values, k=3, axis=1)
    outer_s = np.asarray(baseline(s[-1], nu=1), dtype=float).copy()
    contract_result = outer_open_face_contract.r_first_z_jets(
        state_name=state_name,
        compact_coordinate=z.copy(),
        outer_value_z_jets=(values[:, -1].copy(),),
    )
    if not isinstance(contract_result, OuterOpenFaceDerivativeResult):
        raise ValueError(
            "outer-open-face contract must return OuterOpenFaceDerivativeResult"
        )
    if len(contract_result.r_first_z_jets) != 1:
        raise ValueError(
            "outer-open-face build contract returned the wrong derivative depth"
        )
    target_r = np.asarray(contract_result.r_first_z_jets[0], dtype=float)
    if target_r.shape != outer_s.shape or not np.all(np.isfinite(target_r)):
        raise ValueError("outer-open-face build derivative has the wrong shape")
    ownership = np.asarray(contract_result.ownership_mask, dtype=bool)
    for channel in np.flatnonzero(ownership):
        # The compact wall owns both radial-outer corners.  Only source rows
        # on the open compact face receive the outer target.
        outer_s[1:-1, channel] = (
            0.5*float(parent_r_max)*target_r[1:-1, channel]
        )

    live_compact_record, live_compact_fingerprint = _freeze_contract_record(
        compact_wall_contract, "compact-wall",
    )
    live_outer_record, live_outer_fingerprint = _freeze_contract_record(
        outer_open_face_contract, "outer-open-face",
    )
    if (
        compact_fingerprint != live_compact_fingerprint
        or not _contract_records_equal(compact_record, live_compact_record)
    ):
        raise ValueError("compact-wall contract changed during state construction")
    if (
        outer_fingerprint != live_outer_fingerprint
        or not _contract_records_equal(outer_record, live_outer_record)
    ):
        raise ValueError("outer-open-face contract changed during state construction")

    radial_channels = _build_native_radial_channels(
        r,
        values,
        inner_s,
        outer_s,
        parent_r_max=parent_r_max,
        native_numerator_inner_s=native_numerator_inner_s,
    )
    return RadialFirstConstrainedHermiteState(
        z,
        r,
        radial_channels,
        _derived_radial_anisotropy_numerator(radial_channels),
        endpoints,
        compact_wall_contract,
        outer_open_face_contract,
        ownership,
        state_name,
        z_degree,
        compact_record,
        outer_record,
    )


@dataclass(frozen=True)
class RadialFirstConstrainedHermiteRepresentation:
    """Authoritative Q53 representation with query-time boundary contracts."""

    position: RadialFirstConstrainedHermiteState
    acceleration: RadialFirstConstrainedHermiteState
    source_fingerprint: str
    endpoint_fingerprint: str

    def __post_init__(self):
        if self.position.state_name != "position":
            raise ValueError("constrained position state is mislabeled")
        if self.acceleration.state_name != "acceleration":
            raise ValueError("constrained acceleration state is mislabeled")
        if self.position.z_degree != 5 or self.acceleration.z_degree != 5:
            raise ValueError("authoritative constrained representation must be Q53")
        if not (
            np.array_equal(self.position.source_z, self.acceleration.source_z)
            and np.array_equal(self.position.source_r, self.acceleration.source_r)
            and self.position.compact_wall_contract_id
            == self.acceleration.compact_wall_contract_id
            and self.position.outer_open_face_contract_id
            == self.acceleration.outer_open_face_contract_id
            and self.position.compact_wall_contract_fingerprint
            == self.acceleration.compact_wall_contract_fingerprint
            and self.position.outer_open_face_contract_fingerprint
            == self.acceleration.outer_open_face_contract_fingerprint
        ):
            raise ValueError("constrained position and acceleration recipes differ")
        object.__setattr__(self, "source_fingerprint", str(self.source_fingerprint))
        object.__setattr__(self, "endpoint_fingerprint", str(self.endpoint_fingerprint))

    @classmethod
    def build(
        cls,
        z,
        r,
        position_fields,
        acceleration_fields,
        position_z_endpoints,
        acceleration_z_endpoints,
        *,
        compact_wall_contract,
        outer_open_face_contract,
        parent_r_max=PARENT_R_MAX,
    ):
        bundle = _NativeMetricNodalBundle.build(
            z,
            r,
            position_fields,
            acceleration_fields,
            position_z_endpoints,
            acceleration_z_endpoints,
            parent_r_max=parent_r_max,
        )

        return cls(
            _build_radial_first_constrained_state(
                bundle.z,
                bundle.r,
                bundle.position,
                bundle.position_z_endpoints,
                state_name="position",
                compact_wall_contract=compact_wall_contract,
                outer_open_face_contract=outer_open_face_contract,
                parent_r_max=bundle.parent_r_max,
                z_degree=5,
            ),
            _build_radial_first_constrained_state(
                bundle.z,
                bundle.r,
                bundle.acceleration,
                bundle.acceleration_z_endpoints,
                state_name="acceleration",
                compact_wall_contract=compact_wall_contract,
                outer_open_face_contract=outer_open_face_contract,
                parent_r_max=bundle.parent_r_max,
                z_degree=5,
            ),
            bundle.source_fingerprint,
            bundle.endpoint_fingerprint,
        )

    def state(self, name):
        if str(name) == "position":
            return self.position
        if str(name) == "acceleration":
            return self.acceleration
        raise ValueError("state name must be position or acceleration")

    def coefficient_arrays(self, prefix="radial_first_constrained"):
        return {
            **self.position.coefficient_arrays(f"{prefix}_position"),
            **self.acceleration.coefficient_arrays(
                f"{prefix}_acceleration"
            ),
            f"{prefix}_source_fingerprint": np.asarray(
                self.source_fingerprint
            ),
            f"{prefix}_endpoint_fingerprint": np.asarray(
                self.endpoint_fingerprint
            ),
        }

    def fingerprint(self):
        digest = hashlib.sha256()
        for name, value in sorted(self.coefficient_arrays().items()):
            _update_array_digest(digest, name, value)
        return digest.hexdigest()

    @classmethod
    def from_arrays(
        cls,
        archive,
        prefix="radial_first_constrained",
        *,
        compact_wall_contract,
        outer_open_face_contract,
    ):
        return cls(
            RadialFirstConstrainedHermiteState.from_arrays(
                archive,
                f"{prefix}_position",
                compact_wall_contract=compact_wall_contract,
                outer_open_face_contract=outer_open_face_contract,
            ),
            RadialFirstConstrainedHermiteState.from_arrays(
                archive,
                f"{prefix}_acceleration",
                compact_wall_contract=compact_wall_contract,
                outer_open_face_contract=outer_open_face_contract,
            ),
            str(archive[f"{prefix}_source_fingerprint"]),
            str(archive[f"{prefix}_endpoint_fingerprint"]),
        )


@dataclass(frozen=True)
class RadialFirstConstrainedHermiteComparator:
    """Non-authoritative Q33 evaluator sharing the exact primary contracts."""

    position: RadialFirstConstrainedHermiteState
    acceleration: RadialFirstConstrainedHermiteState
    source_fingerprint: str
    endpoint_fingerprint: str

    def __post_init__(self):
        if self.position.state_name != "position":
            raise ValueError("constrained Q33 position state is mislabeled")
        if self.acceleration.state_name != "acceleration":
            raise ValueError("constrained Q33 acceleration state is mislabeled")
        if self.position.z_degree != 3 or self.acceleration.z_degree != 3:
            raise ValueError("constrained comparator must be Q33")
        if not (
            np.array_equal(self.position.source_z, self.acceleration.source_z)
            and np.array_equal(self.position.source_r, self.acceleration.source_r)
            and self.position.compact_wall_contract_id
            == self.acceleration.compact_wall_contract_id
            and self.position.outer_open_face_contract_id
            == self.acceleration.outer_open_face_contract_id
            and self.position.compact_wall_contract_fingerprint
            == self.acceleration.compact_wall_contract_fingerprint
            and self.position.outer_open_face_contract_fingerprint
            == self.acceleration.outer_open_face_contract_fingerprint
        ):
            raise ValueError("constrained Q33 position and acceleration differ")
        object.__setattr__(self, "source_fingerprint", str(self.source_fingerprint))
        object.__setattr__(self, "endpoint_fingerprint", str(self.endpoint_fingerprint))

    def state(self, name):
        if str(name) == "position":
            return self.position
        if str(name) == "acceleration":
            return self.acceleration
        raise ValueError("state name must be position or acceleration")

    def coefficient_arrays(self, prefix="radial_first_constrained_q33"):
        return {
            **self.position.coefficient_arrays(f"{prefix}_position"),
            **self.acceleration.coefficient_arrays(
                f"{prefix}_acceleration"
            ),
            f"{prefix}_source_fingerprint": np.asarray(
                self.source_fingerprint
            ),
            f"{prefix}_endpoint_fingerprint": np.asarray(
                self.endpoint_fingerprint
            ),
        }

    def fingerprint(self):
        digest = hashlib.sha256()
        for name, value in sorted(self.coefficient_arrays().items()):
            _update_array_digest(digest, name, value)
        return digest.hexdigest()

    @classmethod
    def from_arrays(
        cls,
        archive,
        prefix="radial_first_constrained_q33",
        *,
        compact_wall_contract,
        outer_open_face_contract,
    ):
        return cls(
            RadialFirstConstrainedHermiteState.from_arrays(
                archive,
                f"{prefix}_position",
                compact_wall_contract=compact_wall_contract,
                outer_open_face_contract=outer_open_face_contract,
            ),
            RadialFirstConstrainedHermiteState.from_arrays(
                archive,
                f"{prefix}_acceleration",
                compact_wall_contract=compact_wall_contract,
                outer_open_face_contract=outer_open_face_contract,
            ),
            str(archive[f"{prefix}_source_fingerprint"]),
            str(archive[f"{prefix}_endpoint_fingerprint"]),
        )


@dataclass(frozen=True)
class RadialFirstConstrainedHermitePair:
    """Authoritative Q53 plus additive identical-contract Q33 comparator."""

    primary: RadialFirstConstrainedHermiteRepresentation
    comparator: RadialFirstConstrainedHermiteComparator

    def __post_init__(self):
        if (
            self.primary.source_fingerprint != self.comparator.source_fingerprint
            or self.primary.endpoint_fingerprint
            != self.comparator.endpoint_fingerprint
        ):
            raise ValueError("constrained Q53/Q33 source fingerprints differ")
        for primary_state, comparator_state in (
            (self.primary.position, self.comparator.position),
            (self.primary.acceleration, self.comparator.acceleration),
        ):
            if not (
                np.array_equal(
                    primary_state.radial_channels.coefficients,
                    comparator_state.radial_channels.coefficients,
                )
                and np.array_equal(
                    primary_state.radial_channels.inner_s_derivative,
                    comparator_state.radial_channels.inner_s_derivative,
                )
                and np.array_equal(
                    primary_state.radial_channels.outer_s_derivative,
                    comparator_state.radial_channels.outer_s_derivative,
                )
                and _arrays_bitwise_equal(
                    primary_state.radial_anisotropy_numerator.coefficients,
                    comparator_state.radial_anisotropy_numerator.coefficients,
                )
                and _arrays_bitwise_equal(
                    primary_state.radial_anisotropy_numerator.inner_s_derivative,
                    comparator_state.radial_anisotropy_numerator.inner_s_derivative,
                )
                and _arrays_bitwise_equal(
                    primary_state.radial_anisotropy_numerator.outer_s_derivative,
                    comparator_state.radial_anisotropy_numerator.outer_s_derivative,
                )
                and np.array_equal(
                    primary_state.stored_z_first_endpoints,
                    comparator_state.stored_z_first_endpoints,
                )
                and primary_state.compact_wall_contract_id
                == comparator_state.compact_wall_contract_id
                and primary_state.outer_open_face_contract_id
                == comparator_state.outer_open_face_contract_id
                and np.array_equal(
                    primary_state.outer_ownership_mask,
                    comparator_state.outer_ownership_mask,
                )
                and primary_state.compact_wall_contract_fingerprint
                == comparator_state.compact_wall_contract_fingerprint
                and primary_state.outer_open_face_contract_fingerprint
                == comparator_state.outer_open_face_contract_fingerprint
            ):
                raise ValueError("constrained Q53/Q33 bundles are not identical")

    @classmethod
    def build(cls, *args, **kwargs):
        primary = RadialFirstConstrainedHermiteRepresentation.build(
            *args, **kwargs,
        )

        def cubic(state):
            return RadialFirstConstrainedHermiteState(
                state.source_z,
                state.source_r,
                state.radial_channels,
                state.radial_anisotropy_numerator,
                state.stored_z_first_endpoints,
                state.compact_wall_contract,
                state.outer_open_face_contract,
                state.outer_ownership_mask,
                state.state_name,
                3,
                state.compact_wall_contract_record,
                state.outer_open_face_contract_record,
            )

        comparator = RadialFirstConstrainedHermiteComparator(
            cubic(primary.position),
            cubic(primary.acceleration),
            primary.source_fingerprint,
            primary.endpoint_fingerprint,
        )
        return cls(primary, comparator)

    @property
    def comparison_name(self):
        return "radial-first-identical-contract-Q53-Q33"

    def coefficient_arrays(self, prefix="radial_first_constrained_pair"):
        return {
            **self.primary.coefficient_arrays(f"{prefix}_primary"),
            **self.comparator.coefficient_arrays(f"{prefix}_comparator"),
        }

    @classmethod
    def from_arrays(
        cls,
        archive,
        prefix="radial_first_constrained_pair",
        *,
        compact_wall_contract,
        outer_open_face_contract,
    ):
        return cls(
            RadialFirstConstrainedHermiteRepresentation.from_arrays(
                archive,
                f"{prefix}_primary",
                compact_wall_contract=compact_wall_contract,
                outer_open_face_contract=outer_open_face_contract,
            ),
            RadialFirstConstrainedHermiteComparator.from_arrays(
                archive,
                f"{prefix}_comparator",
                compact_wall_contract=compact_wall_contract,
                outer_open_face_contract=outer_open_face_contract,
            ),
        )
