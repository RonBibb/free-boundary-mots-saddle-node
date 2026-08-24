"""Frozen Protocol-125 bulk sampling and append-only contract lineage.

The compatible acceleration has a constrained two-dimensional Hermite
representation.  The unclosed bulk acceleration deliberately does not: it
does not satisfy the compact-wall acceleration rows.  This module therefore
provides only the two restricted continuous samplers authorized by Protocol
125:

* clamped cubics in squared radius on the two compact walls; and
* clamped compact quintics for the regular q4/q5 axis coefficients.

It also records and validates the append-only transition from a position-only
contract to a shared position/acceleration contract.  It constructs no parent,
performs no solve, writes no artifact, and contains no scientific adjudicator.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.matched_staged_continuum import hash_arrays


PARENT_R_MAX = 12.0
STENCIL_WIDTH = 7
DENSE_WALL_POINT_COUNT = 1025
V2_COMPACT_POINT_COUNT = 153
DENSE_WALL_SHA256 = (
    "8220cfeb81994fb820ed219f5fedb4c1f2ab8ae63798c86444649540323f90d6"
)
V2_COORDINATE_SHA256 = (
    "8e112ea841ccdf4e1cd3c932877f57d3e9046575d2c8cb0a978c735108186f8b"
)
NATIVE_BULK_CHANNEL_ORDER = (
    "h00", "h_perp", "h_rr", "h_zz", "Phi", "chi", "v_z", "v_0",
)
PHYSICAL_ACCELERATION_ORDER = (
    "h_z0", "h_zr", "h_00", "h_perp", "h_rr", "h_0r", "h_zz",
    "Phi", "chi",
)
AXIS_COEFFICIENT_ORDER = ("q4", "q5")
POSITION_PAYLOAD_GROUP_ORDER = (
    "source", "compact", "outer", "coefficients",
)

_POSITION_SOURCE_NAME_ORDER = (
    "source_z", "source_r", "native_channel_order",
    *(f"completed_native_{name}" for name in NATIVE_BULK_CHANNEL_ORDER),
    "coordinate_time_velocity", "source_normal_wall_H_z",
    "reference_channel_order", "finite_wall_reference_q",
    "finite_wall_reference_Phi", "finite_wall_reference_endpoint_z_first",
    "finite_wall_reference_source_record_fingerprint",
    "source_reference_fingerprint",
)
_POSITION_COMPACT_NAME_ORDER = (
    "background_values", "position_knots", "position_coefficients",
    "position_parent_r_max", "source_normal_knots",
    "source_normal_coefficients", "source_normal_parent_r_max",
    "position_ownership_mask", "position_contract_identifier",
    "position_subrecord_fingerprint",
)
_POSITION_OUTER_NAME_ORDER = (
    "source_z", "source_r", "parent_r_max", "source_position",
    "primitive_keys", "primitive_values", "reference_outer_keys",
    "reference_outer_values", "source_reference_fingerprint", "shape_keys",
    "shape_values", "scalar_keys", "scalar_values", "position_r_first",
    "open_compact_mask", "ownership_mask", "validation_tolerance",
    "derivation_recipe", "reference_derivation_recipe",
    "position_contract_identifier", "position_subrecord_fingerprint",
)
_POSITION_COEFFICIENT_SUFFIX_ORDER = (
    "source_z", "source_r",
    "radial_channels_s_knots", "radial_channels_coefficients",
    "radial_channels_inner_s_derivative",
    "radial_channels_outer_s_derivative", "radial_channels_parent_r_max",
    "radial_channels_boundary",
    "radial_anisotropy_numerator_s_knots",
    "radial_anisotropy_numerator_coefficients",
    "radial_anisotropy_numerator_inner_s_derivative",
    "radial_anisotropy_numerator_outer_s_derivative",
    "radial_anisotropy_numerator_parent_r_max",
    "radial_anisotropy_numerator_boundary", "stored_z_first_endpoints",
    "outer_ownership_mask", "z_degree", "state_name",
    "position_state_subrecord_fingerprint",
)
_POSITION_PAYLOAD_NAME_ORDER = {
    "source": _POSITION_SOURCE_NAME_ORDER,
    "compact": _POSITION_COMPACT_NAME_ORDER,
    "outer": _POSITION_OUTER_NAME_ORDER,
    "coefficients": tuple(
        f"{degree}_{suffix}"
        for degree in ("Q53", "Q33")
        for suffix in _POSITION_COEFFICIENT_SUFFIX_ORDER
    ),
}


class Protocol125SamplingError(RuntimeError):
    """Raised when the frozen bulk sampler cannot be reconstructed exactly."""


class Protocol125LineageError(RuntimeError):
    """Raised when a position-to-shared transition is not append-only."""


def _immutable(value, dtype=None):
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _bitwise_equal(left, right):
    left = np.ascontiguousarray(np.asarray(left))
    right = np.ascontiguousarray(np.asarray(right))
    return bool(
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes() == right.tobytes()
    )


def _update_named_digest(digest, name, value):
    array = np.ascontiguousarray(np.asarray(value))
    encoded = str(name).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())


def _fingerprint_entries(entries):
    digest = hashlib.sha256()
    for name, value in entries:
        _update_named_digest(digest, name, value)
    return digest.hexdigest()


def _dense_operator(coordinate):
    operator = derivative_matrix(
        np.asarray(coordinate, dtype=float), 1, STENCIL_WIDTH,
    )
    if hasattr(operator, "toarray"):
        operator = operator.toarray()
    return np.asarray(operator, dtype=float)


def _validate_source(z, r, acceleration):
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    acceleration = np.asarray(acceleration, dtype=float)
    if (
        z.ndim != 1
        or r.ndim != 1
        or len(z) < STENCIL_WIDTH
        or len(r) < STENCIL_WIDTH
        or z[0] != 1.0
        or z[-1] != np.e
        or r[0] != 0.0
        or np.signbit(r[0])
        or r[-1] != PARENT_R_MAX
        or np.any(np.diff(z) <= 0.0)
        or np.any(np.diff(r) <= 0.0)
        or not np.all(np.isfinite(z))
        or not np.all(np.isfinite(r))
        or acceleration.shape != (len(z), len(r), 9)
        or not np.all(np.isfinite(acceleration))
    ):
        raise ValueError("invalid Protocol-125 bulk-acceleration source")
    h_z0 = acceleration[:, :, 0]
    if np.any(h_z0 != 0.0) or np.any(np.signbit(h_z0)):
        raise ValueError("bulk h_z0 acceleration must be IEEE positive zero")
    return z, r, acceleration


def _native_bulk_channels(acceleration, r):
    radius = np.asarray(r, dtype=float)[None, :]
    channels = np.stack((
        acceleration[:, :, 2],
        acceleration[:, :, 3],
        acceleration[:, :, 3]+radius**2*acceleration[:, :, 4],
        acceleration[:, :, 6],
        acceleration[:, :, 7],
        acceleration[:, :, 8],
        acceleration[:, :, 1],
        acceleration[:, :, 5],
    ), axis=-1)
    if not np.all(np.isfinite(channels)):
        raise Protocol125SamplingError("native bulk acceleration is nonfinite")
    return channels


def _scaled_linf(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    scale = np.maximum.reduce((np.ones_like(left), np.abs(left), np.abs(right)))
    return float(np.max(np.abs(left-right)/scale))


def _derive_sampler_arrays(z, r, acceleration):
    z, r, acceleration = _validate_source(z, r, acceleration)
    native = _native_bulk_channels(acceleration, r)
    u = (r/PARENT_R_MAX)**2
    du = _dense_operator(u)
    dz = _dense_operator(z)

    wall_values = native[[0, -1]]
    wall_u_first = np.einsum("ij,wjc->wic", du[[0, -1]], wall_values)
    wall_knots = np.concatenate((
        np.repeat(u[0], 4), u[1:-1], np.repeat(u[-1], 4),
    ))
    wall_coefficients = []
    for wall in range(2):
        spline = make_interp_spline(
            u,
            wall_values[wall],
            k=3,
            axis=0,
            t=wall_knots,
            bc_type=(
                [(1, wall_u_first[wall, 0])],
                [(1, wall_u_first[wall, 1])],
            ),
        )
        if not _bitwise_equal(spline.t, wall_knots):
            raise Protocol125SamplingError("wall cubic knot construction changed")
        if (
            _scaled_linf(spline(u), wall_values[wall]) > 1e-12
            or _scaled_linf(
                spline(u[[0, -1]], nu=1), wall_u_first[wall],
            ) > 1e-12
        ):
            raise Protocol125SamplingError(
                "wall cubic does not reproduce its source/endpoint contract"
            )
        wall_coefficients.append(np.asarray(spline.c, dtype=float))
    wall_coefficients = np.stack(wall_coefficients)

    h_perp = native[:, :, NATIVE_BULK_CHANNEL_ORDER.index("h_perp")]
    h_rr = native[:, :, NATIVE_BULK_CHANNEL_ORDER.index("h_rr")]
    numerator = h_rr-h_perp
    b4 = np.einsum("j,ij->i", du[0], numerator)/PARENT_R_MAX**2
    b5 = native[:, 0, NATIVE_BULK_CHANNEL_ORDER.index("v_0")]
    axis_source = np.stack((b4, b5), axis=-1)
    stored_axis = acceleration[:, 0, 4:6]
    axis_reproduction = _scaled_linf(stored_axis, axis_source)
    if not np.isfinite(axis_reproduction) or axis_reproduction > 1e-12:
        raise Protocol125SamplingError(
            "stored bulk q4/q5 axis values do not reproduce native limits"
        )

    axis_z_first = dz[[0, -1]] @ axis_source
    axis_knots = np.concatenate((
        np.repeat(z[0], 6), z[2:-2], np.repeat(z[-1], 6),
    ))
    axis_spline = make_interp_spline(
        z,
        axis_source,
        k=5,
        axis=0,
        t=axis_knots,
        bc_type=(
            [(1, axis_z_first[0])],
            [(1, axis_z_first[1])],
        ),
    )
    if not _bitwise_equal(axis_spline.t, axis_knots):
        raise Protocol125SamplingError("axis quintic knot construction changed")
    if (
        _scaled_linf(axis_spline(z), axis_source) > 1e-12
        or _scaled_linf(
            axis_spline(z[[0, -1]], nu=1), axis_z_first,
        ) > 1e-12
    ):
        raise Protocol125SamplingError(
            "axis quintic does not reproduce its source/endpoint contract"
        )

    arrays = {
        "source_z": z,
        "source_r": r,
        "source_acceleration": acceleration,
        "wall_knots": wall_knots,
        "wall_coefficients": wall_coefficients,
        "wall_u_first": wall_u_first,
        "axis_source": axis_source,
        "axis_z_first": axis_z_first,
        "axis_knots": axis_knots,
        "axis_coefficients": np.asarray(axis_spline.c, dtype=float),
        "axis_reproduction_scaled_Linf": np.asarray(axis_reproduction),
    }
    source_fingerprint = _fingerprint_entries((
        ("recipe", np.asarray("protocol-125-unclosed-bulk-source-v1")),
        ("source_z", z),
        ("source_r", r),
        ("source_acceleration", acceleration),
    ))
    sampler_fingerprint = _fingerprint_entries((
        ("recipe", np.asarray("protocol-125-restricted-bulk-sampler-v1")),
        *((name, value) for name, value in arrays.items()),
    ))
    return arrays, source_fingerprint, sampler_fingerprint


@dataclass(frozen=True)
class Protocol125BulkAccelerationSampler:
    """Authoritative restricted continuous sampler for unclosed ``a_bulk``."""

    source_z: np.ndarray
    source_r: np.ndarray
    source_acceleration: np.ndarray
    wall_knots: np.ndarray
    wall_coefficients: np.ndarray
    wall_u_first: np.ndarray
    axis_source: np.ndarray
    axis_z_first: np.ndarray
    axis_knots: np.ndarray
    axis_coefficients: np.ndarray
    axis_reproduction_scaled_Linf: float
    source_fingerprint: str
    sampler_fingerprint: str

    def __post_init__(self):
        arrays, source_fingerprint, sampler_fingerprint = _derive_sampler_arrays(
            self.source_z, self.source_r, self.source_acceleration,
        )
        supplied = {
            "wall_knots": self.wall_knots,
            "wall_coefficients": self.wall_coefficients,
            "wall_u_first": self.wall_u_first,
            "axis_source": self.axis_source,
            "axis_z_first": self.axis_z_first,
            "axis_knots": self.axis_knots,
            "axis_coefficients": self.axis_coefficients,
            "axis_reproduction_scaled_Linf": np.asarray(
                self.axis_reproduction_scaled_Linf
            ),
        }
        for name, value in supplied.items():
            if not _bitwise_equal(value, arrays[name]):
                raise Protocol125SamplingError(
                    f"persisted bulk sampler {name} differs from its source"
                )
        if str(self.source_fingerprint) != source_fingerprint:
            raise Protocol125SamplingError("bulk sampler source fingerprint differs")
        if str(self.sampler_fingerprint) != sampler_fingerprint:
            raise Protocol125SamplingError("bulk sampler fingerprint differs")
        for name in (
            "source_z", "source_r", "source_acceleration", "wall_knots",
            "wall_coefficients", "wall_u_first", "axis_source",
            "axis_z_first", "axis_knots", "axis_coefficients",
        ):
            value = arrays[name]
            object.__setattr__(self, name, _immutable(value))
        object.__setattr__(
            self,
            "axis_reproduction_scaled_Linf",
            float(arrays["axis_reproduction_scaled_Linf"]),
        )
        object.__setattr__(self, "source_fingerprint", source_fingerprint)
        object.__setattr__(self, "sampler_fingerprint", sampler_fingerprint)

    @classmethod
    def build(cls, source_z, source_r, bulk_acceleration):
        arrays, source_fingerprint, sampler_fingerprint = _derive_sampler_arrays(
            source_z, source_r, bulk_acceleration,
        )
        return cls(
            arrays["source_z"],
            arrays["source_r"],
            arrays["source_acceleration"],
            arrays["wall_knots"],
            arrays["wall_coefficients"],
            arrays["wall_u_first"],
            arrays["axis_source"],
            arrays["axis_z_first"],
            arrays["axis_knots"],
            arrays["axis_coefficients"],
            float(arrays["axis_reproduction_scaled_Linf"]),
            source_fingerprint,
            sampler_fingerprint,
        )

    def evaluate_wall_native(self, radius):
        """Evaluate the two clamped native wall cubics at arbitrary radii."""
        radius = np.atleast_1d(np.asarray(radius, dtype=float))
        if (
            radius.ndim != 1
            or not np.all(np.isfinite(radius))
            or np.any(radius < 0.0)
            or np.any(radius > PARENT_R_MAX)
        ):
            raise ValueError("bulk wall query lies outside the parent radius")
        u = (radius/PARENT_R_MAX)**2
        values = [
            BSpline(
                self.wall_knots,
                self.wall_coefficients[wall],
                3,
                axis=0,
                extrapolate=False,
            )(u)
            for wall in range(2)
        ]
        result = np.stack(values)
        if not np.all(np.isfinite(result)):
            raise Protocol125SamplingError("bulk wall sampler returned nonfinite data")
        return result

    def evaluate_wall_physical(self, radius):
        """Evaluate the nine physical coordinate accelerations on both walls."""
        radius = np.atleast_1d(np.asarray(radius, dtype=float))
        native = self.evaluate_wall_native(radius)
        index = {name: slot for slot, name in enumerate(NATIVE_BULK_CHANNEL_ORDER)}
        zeros = np.zeros((2, len(radius)), dtype=float)
        result = np.stack((
            zeros,
            radius[None, :]*native[:, :, index["v_z"]],
            native[:, :, index["h00"]],
            native[:, :, index["h_perp"]],
            native[:, :, index["h_rr"]],
            radius[None, :]*native[:, :, index["v_0"]],
            native[:, :, index["h_zz"]],
            native[:, :, index["Phi"]],
            native[:, :, index["chi"]],
        ), axis=-1)
        if np.any(result[:, :, 0] != 0.0) or np.any(np.signbit(result[:, :, 0])):
            raise AssertionError("physical bulk h_z0 lane lost IEEE positive zero")
        return result

    def evaluate_wall_reduced(self, radius):
        """Evaluate the nine production reduced accelerations on both walls.

        The physical anisotropy numerator is divided only at positive radius.
        Its two axis values come from the separately frozen compact-Q5 axis
        sampler, so this helper never extrapolates a quotient through ``r=0``.
        """
        radius = np.atleast_1d(np.asarray(radius, dtype=float))
        native = self.evaluate_wall_native(radius)
        index = {name: slot for slot, name in enumerate(NATIVE_BULK_CHANNEL_ORDER)}
        result = np.zeros((2, len(radius), 9), dtype=float)
        result[:, :, 1] = native[:, :, index["v_z"]]
        result[:, :, 2] = native[:, :, index["h00"]]
        result[:, :, 3] = native[:, :, index["h_perp"]]
        result[:, :, 5] = native[:, :, index["v_0"]]
        result[:, :, 6] = native[:, :, index["h_zz"]]
        result[:, :, 7] = native[:, :, index["Phi"]]
        result[:, :, 8] = native[:, :, index["chi"]]
        positive = radius > 0.0
        numerator = (
            native[:, :, index["h_rr"]]-native[:, :, index["h_perp"]]
        )
        result[:, positive, 4] = (
            numerator[:, positive]/radius[positive][None, :]**2
        )
        axis = radius == 0.0
        if np.any(axis):
            if np.any(numerator[:, axis] != 0.0) or np.any(
                np.signbit(numerator[:, axis])
            ):
                raise Protocol125SamplingError(
                    "bulk wall anisotropy numerator is not IEEE positive zero at the axis"
                )
            endpoint_axis = self.evaluate_axis_coefficients(
                self.source_z[[0, -1]],
            )
            result[:, axis, 4] = endpoint_axis[:, 0][:, None]
        if np.any(result[:, :, 0] != 0.0) or np.any(np.signbit(result[:, :, 0])):
            raise AssertionError("reduced bulk h_z0 lane lost IEEE positive zero")
        if not np.all(np.isfinite(result)):
            raise Protocol125SamplingError("bulk reduced wall sampler is nonfinite")
        return result

    def dense_wall_physical(self):
        """Evaluate only the frozen 1025-radius Protocol-125 wall mesh."""
        radius = np.linspace(0.0, PARENT_R_MAX, DENSE_WALL_POINT_COUNT)
        if hash_arrays(radius) != DENSE_WALL_SHA256:
            raise Protocol125SamplingError("frozen dense-wall coordinate hash differs")
        return self.evaluate_wall_physical(radius)

    def evaluate_axis_coefficients(self, compact_coordinate):
        """Evaluate the clamped-Q5 bulk q4/q5 coefficients on the axis."""
        compact_coordinate = np.atleast_1d(
            np.asarray(compact_coordinate, dtype=float)
        )
        if (
            compact_coordinate.ndim != 1
            or not np.all(np.isfinite(compact_coordinate))
            or np.any(compact_coordinate < self.source_z[0])
            or np.any(compact_coordinate > self.source_z[-1])
        ):
            raise ValueError("bulk axis query lies outside the compact interval")
        result = BSpline(
            self.axis_knots,
            self.axis_coefficients,
            5,
            axis=0,
            extrapolate=False,
        )(compact_coordinate)
        if not np.all(np.isfinite(result)):
            raise Protocol125SamplingError("bulk axis sampler returned nonfinite data")
        return result

    def evaluate_axis_reduced(self, compact_coordinate):
        """Return a reduced nine-lane axis array with authoritative q4/q5."""
        coefficients = self.evaluate_axis_coefficients(compact_coordinate)
        result = np.zeros((len(coefficients), 9), dtype=float)
        result[:, 4:6] = coefficients
        return result

    def v2_axis_reduced(self):
        """Evaluate the sole frozen V2 compact-axis comparison mesh."""
        z = np.linspace(1.0, np.e, V2_COMPACT_POINT_COUNT)
        r = np.linspace(0.0, PARENT_R_MAX, 343)
        if hash_arrays(z, r) != V2_COORDINATE_SHA256:
            raise Protocol125SamplingError("frozen V2 coordinate hash differs")
        return self.evaluate_axis_reduced(z)

    def coefficient_arrays(self, prefix="protocol125_bulk_sampler"):
        return {
            f"{prefix}_source_z": self.source_z.copy(),
            f"{prefix}_source_r": self.source_r.copy(),
            f"{prefix}_source_acceleration": self.source_acceleration.copy(),
            f"{prefix}_wall_knots": self.wall_knots.copy(),
            f"{prefix}_wall_coefficients": self.wall_coefficients.copy(),
            f"{prefix}_wall_u_first": self.wall_u_first.copy(),
            f"{prefix}_axis_source": self.axis_source.copy(),
            f"{prefix}_axis_z_first": self.axis_z_first.copy(),
            f"{prefix}_axis_knots": self.axis_knots.copy(),
            f"{prefix}_axis_coefficients": self.axis_coefficients.copy(),
            f"{prefix}_axis_reproduction_scaled_Linf": np.asarray(
                self.axis_reproduction_scaled_Linf
            ),
            f"{prefix}_source_fingerprint": np.asarray(self.source_fingerprint),
            f"{prefix}_sampler_fingerprint": np.asarray(self.sampler_fingerprint),
            f"{prefix}_native_channel_order": np.asarray(
                NATIVE_BULK_CHANNEL_ORDER
            ),
            f"{prefix}_physical_component_order": np.asarray(
                PHYSICAL_ACCELERATION_ORDER
            ),
            f"{prefix}_axis_coefficient_order": np.asarray(
                AXIS_COEFFICIENT_ORDER
            ),
            f"{prefix}_recipe": np.asarray(
                "wall-Q3-clamped-Du7;axis-Q5-clamped-Dz7-v1"
            ),
        }

    @classmethod
    def from_arrays(cls, archive, prefix="protocol125_bulk_sampler"):
        expected_metadata = {
            f"{prefix}_native_channel_order": np.asarray(
                NATIVE_BULK_CHANNEL_ORDER
            ),
            f"{prefix}_physical_component_order": np.asarray(
                PHYSICAL_ACCELERATION_ORDER
            ),
            f"{prefix}_axis_coefficient_order": np.asarray(
                AXIS_COEFFICIENT_ORDER
            ),
            f"{prefix}_recipe": np.asarray(
                "wall-Q3-clamped-Du7;axis-Q5-clamped-Dz7-v1"
            ),
        }
        for name, expected in expected_metadata.items():
            if name not in archive or not _bitwise_equal(archive[name], expected):
                raise Protocol125SamplingError(f"bulk sampler metadata {name} differs")
        return cls(
            np.asarray(archive[f"{prefix}_source_z"]),
            np.asarray(archive[f"{prefix}_source_r"]),
            np.asarray(archive[f"{prefix}_source_acceleration"]),
            np.asarray(archive[f"{prefix}_wall_knots"]),
            np.asarray(archive[f"{prefix}_wall_coefficients"]),
            np.asarray(archive[f"{prefix}_wall_u_first"]),
            np.asarray(archive[f"{prefix}_axis_source"]),
            np.asarray(archive[f"{prefix}_axis_z_first"]),
            np.asarray(archive[f"{prefix}_axis_knots"]),
            np.asarray(archive[f"{prefix}_axis_coefficients"]),
            float(archive[f"{prefix}_axis_reproduction_scaled_Linf"]),
            str(archive[f"{prefix}_source_fingerprint"]),
            str(archive[f"{prefix}_sampler_fingerprint"]),
        )


def _freeze_payload_groups(groups):
    if not isinstance(groups, Mapping) or set(groups) != set(
        POSITION_PAYLOAD_GROUP_ORDER
    ):
        raise ValueError(
            f"position payload groups must be exactly {POSITION_PAYLOAD_GROUP_ORDER}"
        )
    entries = []
    for group in POSITION_PAYLOAD_GROUP_ORDER:
        values = groups[group]
        if not isinstance(values, Mapping) or not values:
            raise ValueError(f"position payload group {group} must be nonempty")
        if any(not isinstance(name, str) or not name for name in values):
            raise ValueError(f"position payload group {group} has invalid names")
        names = tuple(values)
        for name in names:
            value = np.asarray(values[name])
            if value.dtype == object:
                raise ValueError("position payload object arrays are forbidden")
            entries.append((f"{group}/{name}", _immutable(value)))
    return tuple(entries)


def validate_protocol125_position_payload_group_order(groups):
    """Validate the production payload's stated channel/Q53-to-Q33 order."""
    if not isinstance(groups, Mapping) or tuple(groups) != POSITION_PAYLOAD_GROUP_ORDER:
        raise ValueError("production position payload group order differs")
    for group in POSITION_PAYLOAD_GROUP_ORDER:
        values = groups[group]
        if not isinstance(values, Mapping) or tuple(values) != _POSITION_PAYLOAD_NAME_ORDER[group]:
            raise ValueError(
                f"production position payload group {group} order differs"
            )
    return True


def _freeze_child_entries(children):
    if not isinstance(children, Mapping) or not children:
        raise ValueError("shared contract must append nonempty child data")
    entries = []
    if any(not isinstance(name, str) or not name for name in children):
        raise ValueError("shared child entry is invalid")
    for name in sorted(children):
        value = np.asarray(children[name])
        if value.dtype == object:
            raise ValueError("shared child entry is invalid")
        entries.append((f"child/{name}", _immutable(value)))
    return tuple(entries)


def _validate_payload_entry_order(entries):
    names = tuple(name for name, _ in entries)
    found_by_group = {group: [] for group in POSITION_PAYLOAD_GROUP_ORDER}
    present_groups = set()
    for name, value in entries:
        pieces = name.split("/", 1)
        if (
            len(pieces) != 2
            or pieces[0] not in found_by_group
            or not pieces[1]
            or np.asarray(value).dtype == object
        ):
            raise ValueError("position payload entry is invalid")
        present_groups.add(pieces[0])
        found_by_group[pieces[0]].append(pieces[1])
    if present_groups != set(POSITION_PAYLOAD_GROUP_ORDER):
        raise ValueError("position payload omits a required invariant group")
    observed_groups = tuple(name.split("/", 1)[0] for name in names)
    compacted_groups = tuple(dict.fromkeys(observed_groups))
    if compacted_groups != POSITION_PAYLOAD_GROUP_ORDER:
        raise ValueError("position payload entries are not grouped canonically")
    return names


def _validate_child_entry_order(entries):
    names = tuple(name for name, _ in entries)
    if any(
        not name.startswith("child/")
        or not name[6:]
        or np.asarray(value).dtype == object
        for name, value in entries
    ) or names != tuple(sorted(names)):
        raise ValueError("shared child entries are not in canonical order")
    return names


@dataclass(frozen=True)
class PositionPayloadSnapshot:
    """Hash-fixed invariant position subrecords and top-level contract identity."""

    stage: str
    payload_entries: tuple
    payload_hash: str
    compact_identifier: str
    compact_fingerprint: str
    outer_identifier: str
    outer_fingerprint: str
    archive_fingerprint: str
    parent_payload_hash: str = ""
    parent_compact_identifier: str = ""
    parent_compact_fingerprint: str = ""
    parent_outer_identifier: str = ""
    parent_outer_fingerprint: str = ""
    parent_archive_fingerprint: str = ""
    child_entries: tuple = ()
    child_hash: str = ""

    def __post_init__(self):
        stage = str(self.stage)
        if stage not in ("position-only", "shared"):
            raise ValueError("position payload stage must be position-only or shared")
        entries = tuple(
            (str(name), _immutable(value)) for name, value in self.payload_entries
        )
        names = _validate_payload_entry_order(entries)
        if (
            len(entries) == 0
            or len(names) != len(set(names))
        ):
            raise ValueError("position payload entries must be unique")
        expected_hash = _fingerprint_entries(entries)
        if str(self.payload_hash) != expected_hash:
            raise Protocol125LineageError("position payload hash differs from entries")
        identities = (
            self.compact_identifier,
            self.compact_fingerprint,
            self.outer_identifier,
            self.outer_fingerprint,
            self.archive_fingerprint,
        )
        if any(not str(value) for value in identities):
            raise ValueError("position payload top-level identities must be nonempty")
        child_entries = tuple(
            (str(name), _immutable(value)) for name, value in self.child_entries
        )
        _validate_child_entry_order(child_entries)
        if stage == "position-only":
            if any((
                self.parent_payload_hash,
                self.parent_compact_identifier,
                self.parent_compact_fingerprint,
                self.parent_outer_identifier,
                self.parent_outer_fingerprint,
                self.parent_archive_fingerprint,
                self.child_hash,
            )) or child_entries:
                raise Protocol125LineageError(
                    "position-only snapshot cannot contain shared lineage"
                )
        else:
            lineage = (
                self.parent_payload_hash,
                self.parent_compact_identifier,
                self.parent_compact_fingerprint,
                self.parent_outer_identifier,
                self.parent_outer_fingerprint,
                self.parent_archive_fingerprint,
            )
            if any(not str(value) for value in lineage) or not child_entries:
                raise Protocol125LineageError(
                    "shared snapshot lacks direct parent or appended child data"
                )
            expected_child_hash = _fingerprint_entries(child_entries)
            if str(self.child_hash) != expected_child_hash:
                raise Protocol125LineageError("shared child hash differs from entries")
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "payload_entries", entries)
        object.__setattr__(self, "payload_hash", expected_hash)
        object.__setattr__(self, "compact_identifier", str(self.compact_identifier))
        object.__setattr__(self, "compact_fingerprint", str(self.compact_fingerprint))
        object.__setattr__(self, "outer_identifier", str(self.outer_identifier))
        object.__setattr__(self, "outer_fingerprint", str(self.outer_fingerprint))
        object.__setattr__(self, "archive_fingerprint", str(self.archive_fingerprint))
        object.__setattr__(self, "parent_payload_hash", str(self.parent_payload_hash))
        object.__setattr__(
            self, "parent_compact_identifier", str(self.parent_compact_identifier)
        )
        object.__setattr__(
            self, "parent_compact_fingerprint", str(self.parent_compact_fingerprint)
        )
        object.__setattr__(
            self, "parent_outer_identifier", str(self.parent_outer_identifier)
        )
        object.__setattr__(
            self, "parent_outer_fingerprint", str(self.parent_outer_fingerprint)
        )
        object.__setattr__(
            self, "parent_archive_fingerprint", str(self.parent_archive_fingerprint)
        )
        object.__setattr__(self, "child_entries", child_entries)
        object.__setattr__(self, "child_hash", str(self.child_hash))

    @classmethod
    def capture_position_only(
        cls,
        groups,
        *,
        compact_identifier,
        compact_fingerprint,
        outer_identifier,
        outer_fingerprint,
        archive_fingerprint,
    ):
        entries = _freeze_payload_groups(groups)
        return cls(
            "position-only",
            entries,
            _fingerprint_entries(entries),
            compact_identifier,
            compact_fingerprint,
            outer_identifier,
            outer_fingerprint,
            archive_fingerprint,
        )

    @classmethod
    def capture_shared(
        cls,
        groups,
        *,
        parent,
        appended_children,
        compact_identifier,
        compact_fingerprint,
        outer_identifier,
        outer_fingerprint,
        archive_fingerprint,
    ):
        if not isinstance(parent, cls) or parent.stage != "position-only":
            raise TypeError("shared snapshot requires its position-only parent")
        entries = _freeze_payload_groups(groups)
        children = _freeze_child_entries(appended_children)
        return cls(
            "shared",
            entries,
            _fingerprint_entries(entries),
            compact_identifier,
            compact_fingerprint,
            outer_identifier,
            outer_fingerprint,
            archive_fingerprint,
            parent.payload_hash,
            parent.compact_identifier,
            parent.compact_fingerprint,
            parent.outer_identifier,
            parent.outer_fingerprint,
            parent.archive_fingerprint,
            children,
            _fingerprint_entries(children),
        )

    def coefficient_arrays(self, prefix="protocol125_position_lineage"):
        result = {
            f"{prefix}_stage": np.asarray(self.stage),
            f"{prefix}_payload_names": np.asarray(
                tuple(name for name, _ in self.payload_entries)
            ),
            f"{prefix}_payload_hash": np.asarray(self.payload_hash),
            f"{prefix}_compact_identifier": np.asarray(self.compact_identifier),
            f"{prefix}_compact_fingerprint": np.asarray(self.compact_fingerprint),
            f"{prefix}_outer_identifier": np.asarray(self.outer_identifier),
            f"{prefix}_outer_fingerprint": np.asarray(self.outer_fingerprint),
            f"{prefix}_archive_fingerprint": np.asarray(self.archive_fingerprint),
            f"{prefix}_parent_payload_hash": np.asarray(self.parent_payload_hash),
            f"{prefix}_parent_compact_identifier": np.asarray(
                self.parent_compact_identifier
            ),
            f"{prefix}_parent_compact_fingerprint": np.asarray(
                self.parent_compact_fingerprint
            ),
            f"{prefix}_parent_outer_identifier": np.asarray(
                self.parent_outer_identifier
            ),
            f"{prefix}_parent_outer_fingerprint": np.asarray(
                self.parent_outer_fingerprint
            ),
            f"{prefix}_parent_archive_fingerprint": np.asarray(
                self.parent_archive_fingerprint
            ),
            f"{prefix}_child_names": np.asarray(
                tuple(name for name, _ in self.child_entries)
            ),
            f"{prefix}_child_hash": np.asarray(self.child_hash),
            f"{prefix}_recipe": np.asarray(
                "append-only-position-payload-lineage-v1"
            ),
        }
        for index, (_, value) in enumerate(self.payload_entries):
            result[f"{prefix}_payload_{index:04d}"] = np.asarray(value).copy()
        for index, (_, value) in enumerate(self.child_entries):
            result[f"{prefix}_child_{index:04d}"] = np.asarray(value).copy()
        return result

    @classmethod
    def from_arrays(cls, archive, prefix="protocol125_position_lineage"):
        recipe = np.asarray("append-only-position-payload-lineage-v1")
        if not _bitwise_equal(archive[f"{prefix}_recipe"], recipe):
            raise Protocol125LineageError("position lineage recipe differs")
        payload_names = tuple(str(value) for value in np.asarray(
            archive[f"{prefix}_payload_names"]
        ))
        child_names = tuple(str(value) for value in np.asarray(
            archive[f"{prefix}_child_names"]
        ))
        payload = tuple(
            (name, np.asarray(archive[f"{prefix}_payload_{index:04d}"]))
            for index, name in enumerate(payload_names)
        )
        children = tuple(
            (name, np.asarray(archive[f"{prefix}_child_{index:04d}"]))
            for index, name in enumerate(child_names)
        )
        return cls(
            str(archive[f"{prefix}_stage"]),
            payload,
            str(archive[f"{prefix}_payload_hash"]),
            str(archive[f"{prefix}_compact_identifier"]),
            str(archive[f"{prefix}_compact_fingerprint"]),
            str(archive[f"{prefix}_outer_identifier"]),
            str(archive[f"{prefix}_outer_fingerprint"]),
            str(archive[f"{prefix}_archive_fingerprint"]),
            str(archive[f"{prefix}_parent_payload_hash"]),
            str(archive[f"{prefix}_parent_compact_identifier"]),
            str(archive[f"{prefix}_parent_compact_fingerprint"]),
            str(archive[f"{prefix}_parent_outer_identifier"]),
            str(archive[f"{prefix}_parent_outer_fingerprint"]),
            str(archive[f"{prefix}_parent_archive_fingerprint"]),
            children,
            str(archive[f"{prefix}_child_hash"]),
        )


def validate_append_only_position_lineage(position_only, shared):
    """Fail closed unless ``shared`` appends to one bitwise-identical position."""
    if (
        not isinstance(position_only, PositionPayloadSnapshot)
        or not isinstance(shared, PositionPayloadSnapshot)
        or position_only.stage != "position-only"
        or shared.stage != "shared"
    ):
        raise TypeError("lineage validation requires position-only and shared snapshots")
    gates = {
        "payload_hash_invariant": shared.payload_hash == position_only.payload_hash,
        "payload_names_invariant": tuple(
            name for name, _ in shared.payload_entries
        ) == tuple(name for name, _ in position_only.payload_entries),
        "payload_arrays_bitwise": len(shared.payload_entries)
        == len(position_only.payload_entries) and all(
            left_name == right_name and _bitwise_equal(left, right)
            for (left_name, left), (right_name, right) in zip(
                position_only.payload_entries, shared.payload_entries
            )
        ),
        "direct_parent_payload": (
            shared.parent_payload_hash == position_only.payload_hash
        ),
        "direct_parent_compact": (
            shared.parent_compact_identifier == position_only.compact_identifier
            and shared.parent_compact_fingerprint
            == position_only.compact_fingerprint
        ),
        "direct_parent_outer": (
            shared.parent_outer_identifier == position_only.outer_identifier
            and shared.parent_outer_fingerprint == position_only.outer_fingerprint
        ),
        "direct_parent_archive": (
            shared.parent_archive_fingerprint == position_only.archive_fingerprint
        ),
        "compact_identifier_evolved": (
            shared.compact_identifier != position_only.compact_identifier
        ),
        "compact_fingerprint_evolved": (
            shared.compact_fingerprint != position_only.compact_fingerprint
        ),
        "outer_identifier_evolved": (
            shared.outer_identifier != position_only.outer_identifier
        ),
        "outer_fingerprint_evolved": (
            shared.outer_fingerprint != position_only.outer_fingerprint
        ),
        "archive_fingerprint_evolved": (
            shared.archive_fingerprint != position_only.archive_fingerprint
        ),
        "appended_child_present": bool(shared.child_entries and shared.child_hash),
    }
    if not all(gates.values()):
        failed = tuple(name for name, passed in gates.items() if not passed)
        raise Protocol125LineageError(
            "position/shared contract lineage failed: "+", ".join(failed)
        )
    return {
        "passed": True,
        "gates": gates,
        "position_payload_hash": position_only.payload_hash,
        "position_only": {
            "compact_identifier": position_only.compact_identifier,
            "compact_fingerprint": position_only.compact_fingerprint,
            "outer_identifier": position_only.outer_identifier,
            "outer_fingerprint": position_only.outer_fingerprint,
            "archive_fingerprint": position_only.archive_fingerprint,
        },
        "shared": {
            "compact_identifier": shared.compact_identifier,
            "compact_fingerprint": shared.compact_fingerprint,
            "outer_identifier": shared.outer_identifier,
            "outer_fingerprint": shared.outer_fingerprint,
            "archive_fingerprint": shared.archive_fingerprint,
            "appended_child_hash": shared.child_hash,
        },
    }
