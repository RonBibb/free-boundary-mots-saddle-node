"""Independent position-boundary scorers for draft Protocol 125.

The routines in this module are result independent: they bind one immutable
Q53/Q33 position pair to the already frozen validation meshes and then score
only represented fields.  They do not construct or repair a parent, call a
boundary contract's target-query method, write an artifact, or authorize an
N0/N1 calculation.

Three audits are supplied:

* the Lorentzian signature and eigenvalue margin on the complete prescribed
  union of source, source-cell midpoint, V0/V1/V2, dense-wall, and
  dense-outer samples;
* the nonlinear compact-wall rows, reconstructed from analytic represented
  values/derivatives and the persisted normal-source representation; and
* the balanced outer delta-Robin rows, reconstructed from analytic candidate
  derivatives and a fresh width-seven/quintic representation of the full
  finite-wall reference arrays.

All entry points fail closed when a mesh, source identity, contract identity,
or reference identity is absent or changes after mesh binding.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from scipy.interpolate import make_interp_spline

from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.joint_parent_boundary_contracts import (
    NativeNormalizedCompactWallContract,
    Protocol125OuterOpenFaceDerivativeContract,
    Protocol125PositionOuterOpenFaceDerivativeContract,
)
from bhps.joint_parent_bulk_reference import (
    FiniteWallReferenceHermitePair,
    REFERENCE_CHANNEL_ORDER,
    SOURCE_CELL_MIDPOINT_SPECS,
    SOURCE_STENCIL_WIDTH,
)
from bhps.joint_parent_refinement_diagnostics import (
    DENSE_OUTER_SHA256,
    DENSE_WALL_SHA256,
    VALIDATION_MESH_SPECS,
    frozen_validation_meshes,
)
from bhps.joint_parent_representation import (
    COORDINATE_COMPONENT_ORDER,
    NATIVE_CHANNEL_ORDER,
)
from bhps.matched_staged_continuum import hash_arrays


SIGNATURE_MARGIN_MINIMUM = 1e-8
DENSE_WALL_LINF_CEILING = 1e-10
DENSE_OUTER_LINF_CEILING = 1e-10
PARENT_Z_MINIMUM = 1.0
PARENT_Z_MAXIMUM = float(np.e)
PARENT_R_MAXIMUM = 12.0
WALL_BACKGROUND_ORDER = (
    "wall_stiffness",
    "v0",
    "v1",
    "beta_a",
    "beta_b",
    "wall_potential_a",
    "wall_potential_b",
)


def _immutable_array(value, dtype=float):
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({
            str(name): _freeze(item) for name, item in value.items()
        })
    if isinstance(value, np.ndarray):
        return _immutable_array(value, None)
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _update_digest(digest, name, value):
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype == object:
        raise ValueError(f"audit provenance {name} has object dtype")
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


def _valid_sha256(value):
    value = str(value)
    return (
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _bitwise_equal(left, right):
    left = np.ascontiguousarray(np.asarray(left))
    right = np.ascontiguousarray(np.asarray(right))
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes() == right.tobytes()
    )


def _object_fingerprint(value, label):
    method = getattr(value, "fingerprint", None)
    if method is not None and callable(method):
        fingerprint = str(method())
    else:
        coefficient_arrays = getattr(value, "coefficient_arrays", None)
        if coefficient_arrays is None or not callable(coefficient_arrays):
            raise ValueError(
                f"{label} must expose fingerprint() or coefficient_arrays()"
            )
        arrays = coefficient_arrays()
        if not isinstance(arrays, Mapping) or not arrays:
            raise ValueError(f"{label} coefficient record is missing")
        fingerprint = _fingerprint_arrays(arrays)
    if not _valid_sha256(fingerprint):
        raise ValueError(f"{label} fingerprint is missing or invalid")
    return fingerprint


def _contract_fingerprint(contract, label):
    """Reproduce the representation's identifier-plus-record fingerprint."""
    identifier = str(getattr(contract, "identifier", ""))
    coefficient_arrays = getattr(contract, "coefficient_arrays", None)
    if not identifier or coefficient_arrays is None or not callable(
        coefficient_arrays
    ):
        raise ValueError(f"{label} contract record is missing")
    arrays = coefficient_arrays()
    if not isinstance(arrays, Mapping) or not arrays or "identifier" in arrays:
        raise ValueError(f"{label} contract coefficient record is invalid")
    digest = hashlib.sha256()
    _update_digest(digest, "identifier", np.asarray(identifier))
    for name in sorted(arrays):
        _update_digest(digest, str(name), arrays[name])
    fingerprint = digest.hexdigest()
    if not _valid_sha256(fingerprint):
        raise ValueError(f"{label} contract fingerprint is invalid")
    return fingerprint


@dataclass(frozen=True)
class _PositionPairBinding:
    primary_state: object
    comparator_state: object
    source_fingerprint: str
    endpoint_fingerprint: str
    representation_fingerprint: str


def _position_pair_binding(candidate_pair):
    try:
        primary_member = candidate_pair.primary
        comparator_member = candidate_pair.comparator
    except AttributeError as error:
        raise TypeError("position audit requires a Q53/Q33 candidate pair") from error
    primary = getattr(primary_member, "position", primary_member)
    comparator = getattr(comparator_member, "position", comparator_member)
    if (
        str(getattr(primary, "state_name", "")) != "position"
        or str(getattr(comparator, "state_name", "")) != "position"
        or int(getattr(primary, "z_degree", -1)) != 5
        or int(getattr(comparator, "z_degree", -1)) != 3
    ):
        raise ValueError("position audit requires position Q53/Q33 states")

    source_fingerprint = getattr(candidate_pair, "source_fingerprint", None)
    endpoint_fingerprint = getattr(candidate_pair, "endpoint_fingerprint", None)
    if source_fingerprint is None:
        source_fingerprint = getattr(primary_member, "source_fingerprint", None)
        if source_fingerprint != getattr(
            comparator_member, "source_fingerprint", None,
        ):
            raise ValueError("candidate source fingerprints differ or are missing")
    if endpoint_fingerprint is None:
        endpoint_fingerprint = getattr(primary_member, "endpoint_fingerprint", None)
        if endpoint_fingerprint != getattr(
            comparator_member, "endpoint_fingerprint", None,
        ):
            raise ValueError("candidate endpoint fingerprints differ or are missing")
    source_fingerprint = str(source_fingerprint)
    endpoint_fingerprint = str(endpoint_fingerprint)
    if not _valid_sha256(source_fingerprint) or not _valid_sha256(
        endpoint_fingerprint
    ):
        raise ValueError("candidate source/endpoint provenance is invalid")

    for label, array in (
        ("primary source z", primary.source_z),
        ("primary source r", primary.source_r),
        ("comparator source z", comparator.source_z),
        ("comparator source r", comparator.source_r),
    ):
        if np.asarray(array).flags.writeable:
            raise ValueError(f"{label} must be immutable before position audit")
    if not (
        _bitwise_equal(primary.source_z, comparator.source_z)
        and _bitwise_equal(primary.source_r, comparator.source_r)
        and str(primary.compact_wall_contract_fingerprint)
        == str(comparator.compact_wall_contract_fingerprint)
        and str(primary.outer_open_face_contract_fingerprint)
        == str(comparator.outer_open_face_contract_fingerprint)
    ):
        raise ValueError("candidate Q53/Q33 position identities differ")
    for label, fingerprint in (
        ("compact-wall contract", primary.compact_wall_contract_fingerprint),
        ("outer-open contract", primary.outer_open_face_contract_fingerprint),
    ):
        if not _valid_sha256(fingerprint):
            raise ValueError(f"{label} fingerprint is missing")
    for label, evaluator in (
        ("physical-channel", getattr(primary, "evaluate_physical_channels", None)),
        ("coordinate-component", getattr(primary, "evaluate_coordinate_components", None)),
    ):
        if evaluator is None or not callable(evaluator):
            raise TypeError(f"candidate primary lacks {label} evaluation")
    return _PositionPairBinding(
        primary,
        comparator,
        source_fingerprint,
        endpoint_fingerprint,
        _object_fingerprint(candidate_pair, "candidate position pair"),
    )


def _validate_axis(name, value, lower, upper):
    value = np.asarray(value, dtype=float)
    if (
        value.ndim != 1
        or len(value) < 2
        or np.any(~np.isfinite(value))
        or np.any(np.diff(value) <= 0.0)
        or value[0] != float(lower)
        or value[-1] != float(upper)
    ):
        raise ValueError(f"{name} is not the complete frozen-domain axis")
    return value


@dataclass(frozen=True)
class Protocol125PositionAuditMeshes:
    """Complete mesh union bound to one candidate representation identity."""

    source_z: np.ndarray
    source_r: np.ndarray
    midpoint_z: np.ndarray
    midpoint_r: np.ndarray
    V0_z: np.ndarray
    V0_r: np.ndarray
    V1_z: np.ndarray
    V1_r: np.ndarray
    V2_z: np.ndarray
    V2_r: np.ndarray
    dense_wall_r: np.ndarray
    dense_outer_z: np.ndarray
    candidate_source_fingerprint: str
    candidate_endpoint_fingerprint: str
    candidate_representation_fingerprint: str
    canonical_parent_label: str

    def __post_init__(self):
        axes = {
            "source_z": _validate_axis(
                "source z", self.source_z, PARENT_Z_MINIMUM, PARENT_Z_MAXIMUM,
            ),
            "source_r": _validate_axis(
                "source r", self.source_r, 0.0, PARENT_R_MAXIMUM,
            ),
            "midpoint_z": np.asarray(self.midpoint_z, dtype=float),
            "midpoint_r": np.asarray(self.midpoint_r, dtype=float),
            "V0_z": _validate_axis(
                "V0 z", self.V0_z, PARENT_Z_MINIMUM, PARENT_Z_MAXIMUM,
            ),
            "V0_r": _validate_axis("V0 r", self.V0_r, 0.0, PARENT_R_MAXIMUM),
            "V1_z": _validate_axis(
                "V1 z", self.V1_z, PARENT_Z_MINIMUM, PARENT_Z_MAXIMUM,
            ),
            "V1_r": _validate_axis("V1 r", self.V1_r, 0.0, PARENT_R_MAXIMUM),
            "V2_z": _validate_axis(
                "V2 z", self.V2_z, PARENT_Z_MINIMUM, PARENT_Z_MAXIMUM,
            ),
            "V2_r": _validate_axis("V2 r", self.V2_r, 0.0, PARENT_R_MAXIMUM),
            "dense_wall_r": _validate_axis(
                "dense-wall r", self.dense_wall_r, 0.0, PARENT_R_MAXIMUM,
            ),
            "dense_outer_z": _validate_axis(
                "dense-outer z",
                self.dense_outer_z,
                PARENT_Z_MINIMUM,
                PARENT_Z_MAXIMUM,
            ),
        }
        midpoint_z = axes["midpoint_z"]
        midpoint_r = axes["midpoint_r"]
        expected_midpoint_z = 0.5*(axes["source_z"][:-1]+axes["source_z"][1:])
        expected_midpoint_r = 0.5*(axes["source_r"][:-1]+axes["source_r"][1:])
        if not (
            midpoint_z.ndim == midpoint_r.ndim == 1
            and np.all(np.isfinite(midpoint_z))
            and np.all(np.isfinite(midpoint_r))
            and _bitwise_equal(midpoint_z, expected_midpoint_z)
            and _bitwise_equal(midpoint_r, expected_midpoint_r)
        ):
            raise ValueError("source-cell midpoint axes are not direct cell centers")
        for name in ("V0", "V1", "V2"):
            expected_nz, expected_nr, expected_hash = VALIDATION_MESH_SPECS[name]
            if (
                (len(axes[f"{name}_z"]), len(axes[f"{name}_r"]))
                != (expected_nz, expected_nr)
                or hash_arrays(axes[f"{name}_z"], axes[f"{name}_r"])
                != expected_hash
            ):
                raise ValueError(f"{name} mesh differs from Protocol 125")
        if hash_arrays(axes["dense_wall_r"]) != DENSE_WALL_SHA256:
            raise ValueError("dense-wall mesh differs from Protocol 125")
        if hash_arrays(axes["dense_outer_z"]) != DENSE_OUTER_SHA256:
            raise ValueError("dense-outer mesh differs from Protocol 125")
        label = str(self.canonical_parent_label)
        source_hash = hash_arrays(axes["source_z"], axes["source_r"])
        matching_labels = [
            name for name, specification in SOURCE_CELL_MIDPOINT_SPECS.items()
            if tuple(specification["source_shape"])
            == (len(axes["source_z"]), len(axes["source_r"]))
        ]
        if matching_labels:
            expected_label = matching_labels[0]
            specification = SOURCE_CELL_MIDPOINT_SPECS[expected_label]
            if (
                label != expected_label
                or source_hash != specification["source_coordinate_sha256"]
                or hash_arrays(midpoint_z, midpoint_r)
                != specification["midpoint_coordinate_sha256"]
            ):
                raise ValueError("canonical parent source/midpoint provenance differs")
        elif label != "noncanonical-validation":
            raise ValueError("noncanonical validation mesh must be labeled explicitly")
        for label_name, fingerprint in (
            ("candidate source", self.candidate_source_fingerprint),
            ("candidate endpoint", self.candidate_endpoint_fingerprint),
            ("candidate representation", self.candidate_representation_fingerprint),
        ):
            if not _valid_sha256(fingerprint):
                raise ValueError(f"{label_name} fingerprint is missing")
        for name, value in axes.items():
            object.__setattr__(self, name, _immutable_array(value))
        object.__setattr__(
            self, "candidate_source_fingerprint", str(self.candidate_source_fingerprint),
        )
        object.__setattr__(
            self, "candidate_endpoint_fingerprint", str(self.candidate_endpoint_fingerprint),
        )
        object.__setattr__(
            self,
            "candidate_representation_fingerprint",
            str(self.candidate_representation_fingerprint),
        )
        object.__setattr__(self, "canonical_parent_label", label)

    def coefficient_arrays(self):
        return {
            "source_z": self.source_z,
            "source_r": self.source_r,
            "midpoint_z": self.midpoint_z,
            "midpoint_r": self.midpoint_r,
            "V0_z": self.V0_z,
            "V0_r": self.V0_r,
            "V1_z": self.V1_z,
            "V1_r": self.V1_r,
            "V2_z": self.V2_z,
            "V2_r": self.V2_r,
            "dense_wall_r": self.dense_wall_r,
            "dense_outer_z": self.dense_outer_z,
            "candidate_source_fingerprint": np.asarray(
                self.candidate_source_fingerprint
            ),
            "candidate_endpoint_fingerprint": np.asarray(
                self.candidate_endpoint_fingerprint
            ),
            "candidate_representation_fingerprint": np.asarray(
                self.candidate_representation_fingerprint
            ),
            "canonical_parent_label": np.asarray(self.canonical_parent_label),
        }

    def fingerprint(self):
        return _fingerprint_arrays(self.coefficient_arrays())


def bind_protocol125_position_audit_meshes(candidate_pair):
    """Bind the full, explicit frozen audit union to one position pair."""
    binding = _position_pair_binding(candidate_pair)
    source_z = np.asarray(binding.primary_state.source_z)
    source_r = np.asarray(binding.primary_state.source_r)
    _validate_axis("source z", source_z, PARENT_Z_MINIMUM, PARENT_Z_MAXIMUM)
    _validate_axis("source r", source_r, 0.0, PARENT_R_MAXIMUM)
    source_shape = (len(source_z), len(source_r))
    matching_labels = [
        name for name, specification in SOURCE_CELL_MIDPOINT_SPECS.items()
        if tuple(specification["source_shape"]) == source_shape
    ]
    label = matching_labels[0] if matching_labels else "noncanonical-validation"
    meshes = frozen_validation_meshes()
    return Protocol125PositionAuditMeshes(
        source_z,
        source_r,
        0.5*(source_z[:-1]+source_z[1:]),
        0.5*(source_r[:-1]+source_r[1:]),
        meshes["V0"]["z"],
        meshes["V0"]["r"],
        meshes["V1"]["z"],
        meshes["V1"]["r"],
        meshes["V2"]["z"],
        meshes["V2"]["r"],
        meshes["dense_wall"]["r"],
        meshes["dense_outer"]["z"],
        binding.source_fingerprint,
        binding.endpoint_fingerprint,
        binding.representation_fingerprint,
        label,
    )


def _require_bound_mesh(candidate_pair, meshes):
    if not isinstance(meshes, Protocol125PositionAuditMeshes):
        raise TypeError("a complete Protocol125PositionAuditMeshes binding is required")
    binding = _position_pair_binding(candidate_pair)
    if not (
        _bitwise_equal(binding.primary_state.source_z, meshes.source_z)
        and _bitwise_equal(binding.primary_state.source_r, meshes.source_r)
        and binding.source_fingerprint == meshes.candidate_source_fingerprint
        and binding.endpoint_fingerprint == meshes.candidate_endpoint_fingerprint
        and binding.representation_fingerprint
        == meshes.candidate_representation_fingerprint
    ):
        raise ValueError("candidate identity changed after audit-mesh binding")
    # Re-run dataclass validation indirectly through the frozen fingerprint;
    # all axes are immutable, and this additionally rejects a fabricated
    # object with missing coefficient provenance.
    if not _valid_sha256(meshes.fingerprint()):
        raise ValueError("audit-mesh provenance fingerprint is invalid")
    return binding


def _signature_domain(state, z, r, name, base_mesh_sha256=None):
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    components = np.asarray(
        state.evaluate_coordinate_components(z, r), dtype=float,
    )
    expected = (len(z), len(r), len(COORDINATE_COMPONENT_ORDER))
    if components.shape != expected or not np.all(np.isfinite(components)):
        raise ValueError(f"signature domain {name} returned invalid components")
    index = {
        channel: COORDINATE_COMPONENT_ORDER.index(channel)
        for channel in (
            "h_z0", "h_zr", "h00", "h_perp", "h_rr", "h_0r", "h_zz"
        )
    }
    shape = (len(z), len(r), 5, 5)
    metric = np.zeros(shape, dtype=float)
    metric[:, :, 0, 0] = components[:, :, index["h00"]]
    metric[:, :, 0, 1] = metric[:, :, 1, 0] = components[:, :, index["h_z0"]]
    metric[:, :, 0, 2] = metric[:, :, 2, 0] = components[:, :, index["h_0r"]]
    metric[:, :, 1, 1] = components[:, :, index["h_zz"]]
    metric[:, :, 1, 2] = metric[:, :, 2, 1] = components[:, :, index["h_zr"]]
    metric[:, :, 2, 2] = components[:, :, index["h_rr"]]
    metric[:, :, 3, 3] = components[:, :, index["h_perp"]]
    metric[:, :, 4, 4] = components[:, :, index["h_perp"]]
    eigenvalues = np.linalg.eigvalsh(metric)
    if eigenvalues.shape != (len(z), len(r), 5) or not np.all(
        np.isfinite(eigenvalues)
    ):
        raise RuntimeError(f"signature eigensolve failed on {name}")
    negative_count = np.count_nonzero(eigenvalues < 0.0, axis=-1)
    magnitude = np.abs(eigenvalues)
    margin = np.min(magnitude, axis=-1)/np.maximum(
        np.max(magnitude, axis=-1), 1e-300,
    )
    minimum_margin = float(np.min(margin))
    wrong_signature = int(np.count_nonzero(negative_count != 1))
    return {
        "name": str(name),
        "shape": (len(z), len(r)),
        "sample_count": int(len(z)*len(r)),
        "coordinate_sha256": hash_arrays(z, r),
        "base_mesh_sha256": (
            hash_arrays(z, r) if base_mesh_sha256 is None else str(base_mesh_sha256)
        ),
        "negative_eigenvalue_count_minimum": int(np.min(negative_count)),
        "negative_eigenvalue_count_maximum": int(np.max(negative_count)),
        "wrong_signature_sample_count": wrong_signature,
        "minimum_dimensionless_margin": minimum_margin,
        "maximum_dimensionless_margin": float(np.max(margin)),
        "signature_pass": wrong_signature == 0,
        "margin_pass": minimum_margin >= SIGNATURE_MARGIN_MINIMUM,
        "passed": (
            wrong_signature == 0 and minimum_margin >= SIGNATURE_MARGIN_MINIMUM
        ),
    }


def evaluate_protocol125_signature_union(candidate_pair, meshes):
    """Score the complete no-exclusion Lorentzian-signature sample union."""
    binding = _require_bound_mesh(candidate_pair, meshes)
    state = binding.primary_state
    domains = {
        "source": _signature_domain(
            state, meshes.source_z, meshes.source_r, "source",
        ),
        "source_cell_midpoint": _signature_domain(
            state, meshes.midpoint_z, meshes.midpoint_r, "source_cell_midpoint",
        ),
        "V0": _signature_domain(
            state,
            meshes.V0_z,
            meshes.V0_r,
            "V0",
            VALIDATION_MESH_SPECS["V0"][2],
        ),
        "V1": _signature_domain(
            state,
            meshes.V1_z,
            meshes.V1_r,
            "V1",
            VALIDATION_MESH_SPECS["V1"][2],
        ),
        "V2": _signature_domain(
            state,
            meshes.V2_z,
            meshes.V2_r,
            "V2",
            VALIDATION_MESH_SPECS["V2"][2],
        ),
        "dense_wall_lower": _signature_domain(
            state,
            np.asarray([meshes.source_z[0]]),
            meshes.dense_wall_r,
            "dense_wall_lower",
            DENSE_WALL_SHA256,
        ),
        "dense_wall_upper": _signature_domain(
            state,
            np.asarray([meshes.source_z[-1]]),
            meshes.dense_wall_r,
            "dense_wall_upper",
            DENSE_WALL_SHA256,
        ),
        "dense_outer": _signature_domain(
            state,
            meshes.dense_outer_z,
            np.asarray([meshes.source_r[-1]]),
            "dense_outer",
            DENSE_OUTER_SHA256,
        ),
    }
    minimum_margin = min(
        record["minimum_dimensionless_margin"] for record in domains.values()
    )
    wrong = sum(
        record["wrong_signature_sample_count"] for record in domains.values()
    )
    sample_count = sum(record["sample_count"] for record in domains.values())
    expected_names = (
        "source",
        "source_cell_midpoint",
        "V0",
        "V1",
        "V2",
        "dense_wall_lower",
        "dense_wall_upper",
        "dense_outer",
    )
    if tuple(domains) != expected_names:
        raise RuntimeError("signature domain union is incomplete")
    return _freeze({
        "protocol": "Protocol-125-position-signature-union-v1",
        "candidate_representation_fingerprint": binding.representation_fingerprint,
        "mesh_union_fingerprint": meshes.fingerprint(),
        "no_face_axis_collar_or_corner_exclusion": True,
        "domain_order": expected_names,
        "domains": domains,
        "union_sample_count": int(sample_count),
        "union_wrong_signature_sample_count": int(wrong),
        "union_minimum_dimensionless_margin": float(minimum_margin),
        "required_negative_eigenvalue_count": 1,
        "minimum_margin_threshold": SIGNATURE_MARGIN_MINIMUM,
        "passed": bool(wrong == 0 and minimum_margin >= SIGNATURE_MARGIN_MINIMUM),
    })


def _wall_beta(phi, background_values):
    background = dict(zip(WALL_BACKGROUND_ORDER, background_values))
    gamma = float(background["wall_stiffness"])
    if gamma < 0.0 or not all(np.isfinite(tuple(background.values()))):
        raise ValueError("compact-wall background is invalid")
    target = np.asarray((background["v0"], background["v1"]))[:, None]
    bare = np.asarray((background["beta_a"], background["beta_b"]))[:, None]
    potential = np.asarray((
        background["wall_potential_a"], background["wall_potential_b"],
    ))[:, None]
    branch = np.asarray((1.0, -1.0))[:, None]
    delta = phi-target
    beta = bare+branch*(0.5*gamma*delta**2-potential)/6.0
    orientation = np.asarray((-1.0, 1.0))[:, None]
    return gamma, target, beta, orientation


def _profile_linf(value):
    value = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(value)):
        raise RuntimeError("boundary audit produced a nonfinite residual")
    return float(np.max(np.abs(value)))


def evaluate_protocol125_dense_wall_audit(candidate_pair, meshes):
    """Independently evaluate every represented dense position wall row.

    The analytic candidate evaluator uses the live compact contract, as the
    radial-first representation requires.  This scorer independently
    reassembles every physical row from those represented values and
    derivatives; it never treats a stored residual or precomputed target
    array as the score.
    """
    binding = _require_bound_mesh(candidate_pair, meshes)
    state = binding.primary_state
    contract = state.compact_wall_contract
    if not isinstance(contract, NativeNormalizedCompactWallContract):
        raise TypeError("dense-wall audit requires the native normalized contract")
    if str(state.compact_wall_contract_fingerprint) != _contract_fingerprint(
        contract, "compact-wall contract",
    ):
        raise ValueError("persisted compact-wall contract fingerprint changed")
    if len(contract.background_values) != len(WALL_BACKGROUND_ORDER):
        raise ValueError("compact-wall background provenance is incomplete")
    r = meshes.dense_wall_r
    z_walls = meshes.source_z[[0, -1]]
    values = np.asarray(state.evaluate_coordinate_components(z_walls, r))
    z_first = np.asarray(
        state.evaluate_coordinate_components(z_walls, r, z_order=1),
    )
    expected = (2, len(r), len(COORDINATE_COMPONENT_ORDER))
    if (
        values.shape != expected
        or z_first.shape != expected
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(z_first))
    ):
        raise ValueError("dense-wall candidate representation is invalid")
    index = {
        name: COORDINATE_COMPONENT_ORDER.index(name)
        for name in COORDINATE_COMPONENT_ORDER
    }
    hzz = values[:, :, index["h_zz"]]
    if np.any(hzz <= 0.0):
        raise ValueError("dense-wall normal metric is not positive")
    A = np.sqrt(hzz)
    phi = values[:, :, index["Phi"]]
    phi_z = z_first[:, :, index["Phi"]]
    gamma, target, beta, orientation = _wall_beta(
        phi, contract.background_values,
    )
    source_normal = np.asarray(
        contract.source_normal_context.jets(r, 1)[0][:, :, 0], dtype=float,
    )
    if source_normal.shape != (2, len(r)) or not np.all(np.isfinite(source_normal)):
        raise ValueError("persisted normal-source wall representation is invalid")

    metric_records = {}
    for field in ("h00", "h_perp", "h_rr", "h_0r"):
        value = values[:, :, index[field]]
        derivative = z_first[:, :, index[field]]
        source_term = 2.0*beta*A*value
        numerator = derivative+source_term
        junction = orientation*numerator/(2.0*A)
        term_normalized = np.abs(numerator)/np.maximum(
            1.0, np.abs(derivative)+np.abs(source_term),
        )
        metric_records[field] = {
            wall: {
                "absolute_J_Linf": _profile_linf(junction[wall_index]),
                "term_normalized_Linf": _profile_linf(
                    term_normalized[wall_index]
                ),
                "passed": _profile_linf(junction[wall_index])
                < DENSE_WALL_LINF_CEILING,
            }
            for wall_index, wall in enumerate(("lower", "upper"))
        }

    phi_source = orientation*0.5*gamma*(phi-target)*A
    phi_residual = phi_z+phi_source
    phi_scale = np.maximum(1.0, np.abs(phi_z)+np.abs(phi_source))
    phi_profile = phi_residual/phi_scale
    chi_z = z_first[:, :, index["chi"]]
    chi_scale = np.maximum(1.0, np.abs(chi_z))
    chi_profile = chi_z/chi_scale
    hzz_z = z_first[:, :, index["h_zz"]]
    normal_terms = (
        hzz_z,
        8.0*beta*hzz**1.5,
        -2.0*source_normal*hzz,
    )
    normal_residual = sum(normal_terms)
    normal_scale = np.maximum(
        1.0, sum(np.abs(term) for term in normal_terms),
    )
    normal_profile = normal_residual/normal_scale
    other_records = {}
    for wall_index, wall in enumerate(("lower", "upper")):
        phi_profile_linf = _profile_linf(phi_profile[wall_index])
        chi_profile_linf = _profile_linf(chi_profile[wall_index])
        normal_profile_linf = _profile_linf(normal_profile[wall_index])
        other_records[wall] = {
            "Phi": {
                "signed_terms": {
                    "Phi_z": _immutable_array(phi_z[wall_index]),
                    "wall_source": _immutable_array(phi_source[wall_index]),
                },
                "signed_residual": _immutable_array(phi_residual[wall_index]),
                "positive_scale": _immutable_array(phi_scale[wall_index]),
                "signed_profile": _immutable_array(phi_profile[wall_index]),
                "absolute_profile": _immutable_array(
                    np.abs(phi_profile[wall_index])
                ),
                "profile_Linf": phi_profile_linf,
                "passed": phi_profile_linf < DENSE_WALL_LINF_CEILING,
            },
            "chi": {
                "signed_terms": {
                    "chi_z": _immutable_array(chi_z[wall_index]),
                },
                "signed_residual": _immutable_array(chi_z[wall_index]),
                "positive_scale": _immutable_array(chi_scale[wall_index]),
                "signed_profile": _immutable_array(chi_profile[wall_index]),
                "absolute_profile": _immutable_array(
                    np.abs(chi_profile[wall_index])
                ),
                "profile_Linf": chi_profile_linf,
                "passed": chi_profile_linf < DENSE_WALL_LINF_CEILING,
            },
            "normal_GH": {
                "signed_terms": {
                    "G_z": _immutable_array(normal_terms[0][wall_index]),
                    "beta_term": _immutable_array(normal_terms[1][wall_index]),
                    "normal_source_term": _immutable_array(
                        normal_terms[2][wall_index]
                    ),
                },
                "signed_residual": _immutable_array(
                    normal_residual[wall_index]
                ),
                "positive_scale": _immutable_array(normal_scale[wall_index]),
                "signed_profile": _immutable_array(
                    normal_profile[wall_index]
                ),
                "absolute_profile": _immutable_array(
                    np.abs(normal_profile[wall_index])
                ),
                "profile_Linf": normal_profile_linf,
                "passed": normal_profile_linf < DENSE_WALL_LINF_CEILING,
            },
        }
    metric_pass = all(
        wall_record["passed"]
        for field_record in metric_records.values()
        for wall_record in field_record.values()
    )
    other_pass = all(
        row["passed"]
        for wall in other_records.values()
        for row in wall.values()
    )
    combined_metric = max(
        wall_record["absolute_J_Linf"]
        for field_record in metric_records.values()
        for wall_record in field_record.values()
    )
    return _freeze({
        "protocol": "Protocol-125-independent-dense-wall-position-v1",
        "candidate_representation_fingerprint": binding.representation_fingerprint,
        "compact_wall_contract_fingerprint": str(
            state.compact_wall_contract_fingerprint
        ),
        "mesh_union_fingerprint": meshes.fingerprint(),
        "dense_wall_mesh_sha256": DENSE_WALL_SHA256,
        "sample_count_per_wall": len(r),
        "stored_residual_or_target_array_comparison_used": False,
        "live_compact_contract_used_by_candidate_evaluator": True,
        "metric_functional": "sigma*(f_z+2*beta*sqrt(h_zz)*f)/(2*sqrt(h_zz))",
        "metric": metric_records,
        "combined_metric_absolute_J_Linf": float(combined_metric),
        "other_rows": other_records,
        "strict_Linf_ceiling": DENSE_WALL_LINF_CEILING,
        "constituent_logical_AND": True,
        "passed": bool(metric_pass and other_pass),
    })


def _reference_pair_binding(reference_pair, candidate_binding):
    if not isinstance(reference_pair, FiniteWallReferenceHermitePair):
        raise TypeError("outer audit requires a finite-wall Q53/Q33 reference pair")
    primary = reference_pair.primary
    comparator = reference_pair.comparator
    for label, array in (
        ("reference primary z", primary.source_z),
        ("reference primary r", primary.source_r),
        ("reference primary values", primary.source_values),
        ("reference primary endpoints", primary.endpoint_z_first),
        ("reference comparator z", comparator.source_z),
        ("reference comparator r", comparator.source_r),
        ("reference comparator values", comparator.source_values),
        ("reference comparator endpoints", comparator.endpoint_z_first),
    ):
        if np.asarray(array).flags.writeable:
            raise ValueError(f"{label} must be immutable before outer audit")
    if not (
        _bitwise_equal(primary.source_z, candidate_binding.primary_state.source_z)
        and _bitwise_equal(primary.source_r, candidate_binding.primary_state.source_r)
        and _bitwise_equal(primary.source_z, comparator.source_z)
        and _bitwise_equal(primary.source_r, comparator.source_r)
        and _bitwise_equal(primary.source_values, comparator.source_values)
        and _bitwise_equal(primary.endpoint_z_first, comparator.endpoint_z_first)
        and int(primary.stencil_width) == SOURCE_STENCIL_WIDTH
        and int(comparator.stencil_width) == SOURCE_STENCIL_WIDTH
        and tuple(primary.channel_order) == REFERENCE_CHANNEL_ORDER
        and tuple(comparator.channel_order) == REFERENCE_CHANNEL_ORDER
    ):
        raise ValueError("candidate/reference source identity is incomplete or differs")
    return primary, _object_fingerprint(reference_pair, "finite-wall reference pair")


def _position_outer_contract(state):
    contract = state.outer_open_face_contract
    if isinstance(contract, Protocol125OuterOpenFaceDerivativeContract):
        contract = contract.position_contract
    if not isinstance(contract, Protocol125PositionOuterOpenFaceDerivativeContract):
        raise TypeError("outer audit requires the Protocol-125 position outer contract")
    if str(state.outer_open_face_contract_fingerprint) != _contract_fingerprint(
        state.outer_open_face_contract, "candidate outer-open contract",
    ):
        raise ValueError("persisted outer-open contract fingerprint changed")
    return contract


def evaluate_protocol125_dense_outer_delta_robin_audit(
    candidate_pair,
    reference_pair,
    meshes,
):
    """Independently score dense balanced q/Phi outer delta-Robin rows.

    Candidate q and q_r are reconstructed from the represented physical
    metric and its analytic radial derivative.  Reference values and radial
    derivatives are freshly formed from the full immutable source arrays with
    the native width-seven radial operator and independent all-row quintic
    splines.  No outer contract query or stored derivative target is used as
    a residual comparator.
    """
    binding = _require_bound_mesh(candidate_pair, meshes)
    state = binding.primary_state
    reference, reference_fingerprint = _reference_pair_binding(
        reference_pair, binding,
    )
    position_contract = _position_outer_contract(state)
    if not (
        _bitwise_equal(position_contract.source_z, reference.source_z)
        and _bitwise_equal(position_contract.source_r, reference.source_r)
    ):
        raise ValueError("outer contract source grid differs from reference")
    radial_operator = derivative_matrix(
        reference.source_r, 1, SOURCE_STENCIL_WIDTH,
    )
    source_values = np.asarray(reference.source_values)
    q_index = REFERENCE_CHANNEL_ORDER.index("q")
    phi_index = REFERENCE_CHANNEL_ORDER.index("Phi")
    source_q = source_values[:, :, q_index]
    source_phi = source_values[:, :, phi_index]
    source_q_r = (radial_operator @ source_q.T).T[:, -1]
    source_phi_r = (radial_operator @ source_phi.T).T[:, -1]
    reference_outer = np.stack((
        source_q[:, -1],
        source_q_r,
        source_phi[:, -1],
        source_phi_r,
    ), axis=-1)
    if not _bitwise_equal(
        reference_outer, position_contract.reference_outer_values,
    ):
        raise ValueError("outer contract is not bound to the supplied full reference")

    z = meshes.dense_outer_z
    radius = float(meshes.source_r[-1])
    candidate = np.asarray(
        state.evaluate_physical_channels(z, np.asarray([radius])), dtype=float,
    )[:, 0]
    candidate_r = np.asarray(
        state.evaluate_physical_channels(
            z, np.asarray([radius]), r_order=1,
        ),
        dtype=float,
    )[:, 0]
    expected = (len(z), len(NATIVE_CHANNEL_ORDER))
    if (
        candidate.shape != expected
        or candidate_r.shape != expected
        or not np.all(np.isfinite(candidate))
        or not np.all(np.isfinite(candidate_r))
    ):
        raise ValueError("dense-outer candidate representation is invalid")
    native_index = {name: NATIVE_CHANNEL_ORDER.index(name) for name in (
        "h_perp", "h_rr", "h_zz", "Phi",
    )}
    h_perp = candidate[:, native_index["h_perp"]]
    h_rr = candidate[:, native_index["h_rr"]]
    h_zz = candidate[:, native_index["h_zz"]]
    if np.any(h_perp <= 0.0) or np.any(h_rr <= 0.0) or np.any(h_zz <= 0.0):
        raise ValueError("dense-outer spatial metric is not positive")
    h_perp_r = candidate_r[:, native_index["h_perp"]]
    h_rr_r = candidate_r[:, native_index["h_rr"]]
    h_zz_r = candidate_r[:, native_index["h_zz"]]
    psi = (h_zz*h_rr*h_perp**2)**0.125
    logarithmic_psi_r = 0.125*(
        h_zz_r/h_zz+h_rr_r/h_rr+2.0*h_perp_r/h_perp
    )
    psi_r = psi*logarithmic_psi_r
    q = 1.0/psi-z
    q_r = -psi_r/psi**2
    phi = candidate[:, native_index["Phi"]]
    phi_r = candidate_r[:, native_index["Phi"]]

    # Fresh audit-only interpolation; this deliberately does not call either
    # finite-wall reference surface or outer target-query implementation.
    reference_spline = make_interp_spline(
        reference.source_z,
        reference_outer,
        k=5,
        axis=0,
    )
    dense_reference = np.asarray(reference_spline(z), dtype=float)
    q_reference = dense_reference[:, 0]
    q_reference_r = dense_reference[:, 1]
    phi_reference = dense_reference[:, 2]
    phi_reference_r = dense_reference[:, 3]
    q_terms = (
        q_r,
        -q_reference_r,
        (q-q_reference)/radius,
    )
    phi_terms = (
        phi_r,
        -phi_reference_r,
        (phi-phi_reference)/radius,
    )
    q_residual = sum(q_terms)
    phi_residual = sum(phi_terms)
    q_normalized = np.abs(q_residual)/np.maximum(
        1.0, sum(np.abs(term) for term in q_terms),
    )
    phi_normalized = np.abs(phi_residual)/np.maximum(
        1.0, sum(np.abs(term) for term in phi_terms),
    )
    open_mask = np.zeros(len(z), dtype=bool)
    open_mask[1:-1] = True
    if not np.array_equal(open_mask, position_contract.open_compact_mask):
        # The contract's mask is on source z, so only its ownership pattern,
        # not its different length, can be compared here.
        expected_source_mask = np.zeros(len(position_contract.source_z), dtype=bool)
        expected_source_mask[1:-1] = True
        if not np.array_equal(
            position_contract.open_compact_mask, expected_source_mask,
        ):
            raise ValueError("outer compact-wall ownership mask changed")
    q_linf = _profile_linf(q_normalized[open_mask])
    phi_linf = _profile_linf(phi_normalized[open_mask])
    endpoint_report = {
        "lower": {
            "q_normalized_absolute": float(q_normalized[0]),
            "Phi_normalized_absolute": float(phi_normalized[0]),
        },
        "upper": {
            "q_normalized_absolute": float(q_normalized[-1]),
            "Phi_normalized_absolute": float(phi_normalized[-1]),
        },
    }
    return _freeze({
        "protocol": "Protocol-125-independent-dense-outer-delta-Robin-v1",
        "candidate_representation_fingerprint": binding.representation_fingerprint,
        "reference_pair_fingerprint": reference_fingerprint,
        "outer_contract_fingerprint": str(
            state.outer_open_face_contract_fingerprint
        ),
        "source_reference_fingerprint": str(
            position_contract.source_reference_fingerprint
        ),
        "mesh_union_fingerprint": meshes.fingerprint(),
        "dense_outer_mesh_sha256": DENSE_OUTER_SHA256,
        "candidate_reconstruction": (
            "psi=(h_zz*h_rr*h_perp^2)^(1/8); q=1/psi-z; "
            "analytic represented radial derivatives"
        ),
        "reference_reconstruction": (
            "full source q/Phi; width-seven D_r outer row; fresh all-row "
            "degree-five default-not-a-knot z splines"
        ),
        "contract_target_query_called": False,
        "compact_endpoints_excluded_from_score": True,
        "open_sample_count": int(np.count_nonzero(open_mask)),
        "q": {
            "normalized_Linf": q_linf,
            "passed": q_linf < DENSE_OUTER_LINF_CEILING,
        },
        "Phi": {
            "normalized_Linf": phi_linf,
            "passed": phi_linf < DENSE_OUTER_LINF_CEILING,
        },
        "endpoint_report_not_scored": endpoint_report,
        "strict_Linf_ceiling": DENSE_OUTER_LINF_CEILING,
        "passed": bool(
            q_linf < DENSE_OUTER_LINF_CEILING
            and phi_linf < DENSE_OUTER_LINF_CEILING
        ),
    })
