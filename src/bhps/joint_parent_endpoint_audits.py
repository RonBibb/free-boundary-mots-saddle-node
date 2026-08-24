"""Protocol-125 endpoint conversion and normalized wall-profile scorers.

These helpers are pure audits.  They neither solve a boundary row nor modify
the position, acceleration, source, or representation supplied by a caller.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
from scipy.interpolate import make_interp_spline

from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.joint_parent_boundary_contracts import (
    NativeNormalizedCompactWallContract,
    Protocol125OuterOpenFaceDerivativeContract,
    Protocol125PositionOuterOpenFaceDerivativeContract,
)
from bhps.joint_parent_representation import (
    COORDINATE_COMPONENT_ORDER,
    NATIVE_CHANNEL_ORDER,
)
from bhps.joint_parent_refinement_diagnostics import (
    DENSE_OUTER_SHA256,
    DENSE_WALL_SHA256,
    frozen_validation_meshes,
)
from bhps.joint_parent_representation import RadialFirstConstrainedHermitePair
from bhps.matched_staged_continuum import hash_arrays
from bhps.regular_so3_gh_reduction import FIELD_ORDER as REDUCED_FIELD_ORDER


WALL_ORDER = ("lower", "upper")
WALL_PROFILE_ROWS = ("Phi", "chi", "normal_GH")
WALL_PROFILE_STAGES = ("position", "acceleration")
WALL_PROFILE_MESHES = ("source", "dense")
WALL_PROFILE_EVIDENCE_PROTOCOL_IDENTIFIER = (
    "Protocol-125-source-and-dense-wall-profile-evidence-v1"
)
SOURCE_WALL_PROFILE_DERIVATIVE_RECIPE = (
    "native-width-seven-Dz-on-completed-source-arrays"
)
DENSE_WALL_PROFILE_DERIVATIVE_RECIPE = (
    "final-Q53-analytic-z-order-one-with-live-compact-contract"
)
DENSE_WALL_SOURCE_RECIPE = (
    "live-compact-source-and-source-second-radial-contexts"
)
WALL_PROFILE_INPUT_HASH_KEYS = (
    "completed_position_sha256",
    "completed_velocity_sha256",
    "completed_acceleration_sha256",
    "source_sha256",
    "source_time_sha256",
    "source_second_time_sha256",
    "Q53_position_state_sha256",
    "Q53_acceleration_state_sha256",
)
ACCELERATION_ENDPOINT_CONVERSION_LANES = (
    "source_Dz7_vs_row_implied_acceleration_endpoint_z",
    "Q53_acceleration_endpoint_conversion",
    "Q33_acceleration_endpoint_conversion",
)
SOURCE_ACCELERATION_ENDPOINT_CEILING = 1e-12
DENSE_ACCELERATION_ENDPOINT_CEILING = 1e-10
COMPACT_NATIVE_OWNERSHIP_MASK = tuple(True for _ in NATIVE_CHANNEL_ORDER)


def _immutable(value, dtype=float):
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _positive_zero(value):
    value = np.asarray(value, dtype=float)
    return bool(np.all(value == 0.0) and not np.any(np.signbit(value)))


def _hash_named_arrays(values):
    digest = hashlib.sha256()
    for name, value in values:
        array = np.ascontiguousarray(np.asarray(value))
        digest.update(str(name).encode())
        digest.update(b"\0")
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _convert_native_acceleration_z(
    native_acceleration,
    native_acceleration_z,
    native_acceleration_z_s_first,
    r,
    *,
    parent_r_max=12.0,
    require_normal_tangential_positive_zero,
    conversion_kind,
):
    """Common Protocol-125 native-to-physical/reduced endpoint conversion.

    ``native_acceleration_z_s_first`` is the analytic first derivative with
    respect to ``s=(r/R)^2``.  It is mandatory because the q4 axis value may
    not be inferred from a positive-radius quotient or fit.
    """
    acceleration = np.asarray(native_acceleration, dtype=float)
    z_first = np.asarray(native_acceleration_z, dtype=float)
    s_first = np.asarray(native_acceleration_z_s_first, dtype=float)
    r = np.asarray(r, dtype=float)
    expected = (2, len(r), len(NATIVE_CHANNEL_ORDER))
    if (
        acceleration.shape != expected
        or z_first.shape != expected
        or s_first.shape != expected
        or r.ndim != 1
        or len(r) < 2
        or r[0] != 0.0
        or np.any(np.diff(r) <= 0.0)
        or not all(np.all(np.isfinite(value)) for value in (
            acceleration, z_first, s_first, r,
        ))
    ):
        raise ValueError("row-implied acceleration-z inputs are invalid")
    radius = float(parent_r_max)
    if not np.isfinite(radius) or radius <= 0.0 or r[-1] > radius:
        raise ValueError("row-implied acceleration-z parent radius is invalid")
    index = {name: NATIVE_CHANNEL_ORDER.index(name) for name in NATIVE_CHANNEL_ORDER}
    numerator = acceleration[:, 0, index["h_rr"]]-acceleration[:, 0, index["h_perp"]]
    numerator_z = z_first[:, 0, index["h_rr"]]-z_first[:, 0, index["h_perp"]]
    if not _positive_zero(numerator) or not _positive_zero(numerator_z):
        raise ValueError("q4 acceleration and z-numerator axis traces must be positive zero")
    if (
        bool(require_normal_tangential_positive_zero)
        and not _positive_zero(z_first[:, 0, index["v_z"]])
    ):
        raise ValueError("row-implied q1 axis trace must be positive zero")

    physical = np.zeros((2, len(r), len(COORDINATE_COMPONENT_ORDER)))
    physical[:, :, 0] = 0.0
    physical[:, :, 1] = r[None, :]*z_first[:, :, index["v_z"]]
    physical[:, :, 2] = z_first[:, :, index["h00"]]
    physical[:, :, 3] = z_first[:, :, index["h_perp"]]
    physical[:, :, 4] = z_first[:, :, index["h_rr"]]
    physical[:, :, 5] = r[None, :]*z_first[:, :, index["v_0"]]
    physical[:, :, 6] = z_first[:, :, index["h_zz"]]
    physical[:, :, 7] = z_first[:, :, index["Phi"]]
    physical[:, :, 8] = z_first[:, :, index["chi"]]

    reduced = np.zeros((2, len(r), len(REDUCED_FIELD_ORDER)))
    reduced[:, :, 0] = 0.0
    reduced[:, :, 1] = z_first[:, :, index["v_z"]]
    reduced[:, :, 2] = z_first[:, :, index["h00"]]
    reduced[:, :, 3] = z_first[:, :, index["h_perp"]]
    difference = z_first[:, :, index["h_rr"]]-z_first[:, :, index["h_perp"]]
    reduced[:, 1:, 4] = difference[:, 1:]/r[None, 1:]**2
    q4_axis_source = (
        s_first[:, 0, index["h_rr"]]-s_first[:, 0, index["h_perp"]]
    )/radius**2
    reduced[:, 0, 4] = q4_axis_source
    reduced[:, :, 5] = z_first[:, :, index["v_0"]]
    q5_axis_source = z_first[:, 0, index["v_0"]]
    reduced[:, :, 6] = z_first[:, :, index["h_zz"]]
    reduced[:, :, 7] = z_first[:, :, index["Phi"]]
    reduced[:, :, 8] = z_first[:, :, index["chi"]]
    if not np.all(np.isfinite(reduced)):
        raise RuntimeError("row-implied reduced acceleration-z conversion is nonfinite")
    axis_images = np.stack((2.0*q4_axis_source, q5_axis_source), axis=-1)
    ownership_mask = np.asarray(COMPACT_NATIVE_OWNERSHIP_MASK, dtype=bool)
    fingerprint = _hash_named_arrays((
        ("conversion_kind", np.asarray(str(conversion_kind))),
        ("r", r),
        ("native_acceleration", acceleration),
        ("native_acceleration_z", z_first),
        ("native_acceleration_z_s_first", s_first),
        ("physical", physical),
        ("reduced", reduced),
        ("native_acceleration_z_s_first", s_first),
        ("ownership_mask", ownership_mask),
        ("q4_axis_source", q4_axis_source),
        ("q5_axis_source", q5_axis_source),
        ("axis_images", axis_images),
    ))
    return MappingProxyType({
        "native_channel_order": NATIVE_CHANNEL_ORDER,
        "physical_component_order": COORDINATE_COMPONENT_ORDER,
        "reduced_field_order": tuple(REDUCED_FIELD_ORDER),
        "physical": _immutable(physical),
        "reduced": _immutable(reduced),
        "native_acceleration_z_s_first": _immutable(s_first),
        "ownership_mask": _immutable(ownership_mask, bool),
        "physical_sha256": hash_arrays(physical),
        "reduced_sha256": hash_arrays(reduced),
        "s_jet_inputs_sha256": hash_arrays(s_first),
        "ownership_mask_sha256": hash_arrays(ownership_mask),
        "axis_image_order": ("partial_r2_N_z", "partial_r_T_z"),
        "axis_images": _immutable(axis_images),
        "q4_axis_source": _immutable(q4_axis_source),
        "q5_axis_source": _immutable(q5_axis_source),
        "conversion_kind": str(conversion_kind),
        "normal_tangential_positive_zero_required": bool(
            require_normal_tangential_positive_zero
        ),
        "fingerprint": fingerprint,
    })


def convert_row_implied_acceleration_z(
    native_acceleration,
    native_acceleration_z,
    native_acceleration_z_s_first,
    r,
    *,
    parent_r_max=12.0,
):
    """Convert the acceleration derivatives owned by the live wall rows."""
    return _convert_native_acceleration_z(
        native_acceleration,
        native_acceleration_z,
        native_acceleration_z_s_first,
        r,
        parent_r_max=parent_r_max,
        require_normal_tangential_positive_zero=True,
        conversion_kind="row-implied-live-compact-contract",
    )


def convert_native_fd_acceleration_z_comparator(
    native_acceleration,
    native_acceleration_z,
    native_acceleration_z_s_first,
    r,
    *,
    parent_r_max=12.0,
):
    """Convert the independent source-width-seven endpoint comparator.

    This calls the same algebra as the row-owned conversion.  The comparator
    is not allowed to replace its finite-difference ``v_z,z`` value with the
    wall owner's exact zero, so only that ownership assertion is disabled.
    The anisotropy numerator and its compact derivative remain exact
    positive-zero prerequisites for the analytic q4 axis limit.
    """
    return _convert_native_acceleration_z(
        native_acceleration,
        native_acceleration_z,
        native_acceleration_z_s_first,
        r,
        parent_r_max=parent_r_max,
        require_normal_tangential_positive_zero=False,
        conversion_kind="independent-source-Dz7-comparator",
    )


def _scaled_score(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or not (
        np.all(np.isfinite(left)) and np.all(np.isfinite(right))
    ):
        raise ValueError("endpoint comparison arrays must be finite and shape matched")
    error = np.abs(left-right)/np.maximum.reduce((
        np.ones_like(left), np.abs(left), np.abs(right),
    ))
    return {
        "scaled_Linf": float(np.max(error)),
        "scaled_RMS": float(np.sqrt(np.mean(error**2))),
        "sample_count": int(error.size),
    }


def score_state_endpoint_z_reproduction(state, dense_r, *, require_frozen=True):
    """Score stored, live-contract, and analytic endpoint-z data independently."""
    state_name = str(getattr(state, "state_name", ""))
    if state_name not in ("position", "acceleration"):
        raise ValueError("endpoint-z scorer requires a position or acceleration state")
    source_z = np.asarray(getattr(state, "source_z", ()), dtype=float)
    source_r = np.asarray(getattr(state, "source_r", ()), dtype=float)
    dense_r = np.asarray(dense_r, dtype=float)
    evaluator = getattr(state, "evaluate_physical_channels", None)
    contract = getattr(state, "compact_wall_contract", None)
    stored = np.asarray(getattr(state, "stored_z_first_endpoints", ()), dtype=float)
    expected_source_shape = (2, len(source_r), len(NATIVE_CHANNEL_ORDER))
    if (
        source_z.ndim != 1
        or source_r.ndim != 1
        or dense_r.ndim != 1
        or len(source_z) < 2
        or len(source_r) < 2
        or np.any(np.diff(source_z) <= 0.0)
        or np.any(np.diff(source_r) <= 0.0)
        or np.any(np.diff(dense_r) <= 0.0)
        or source_r[0] != 0.0
        or dense_r[0] != 0.0
        or dense_r[-1] != source_r[-1]
        or stored.shape != expected_source_shape
        or not np.all(np.isfinite(stored))
        or evaluator is None
        or not callable(evaluator)
        or contract is None
    ):
        raise ValueError("endpoint-z state or coordinates are invalid")
    if require_frozen and hash_arrays(dense_r) != DENSE_WALL_SHA256:
        raise ValueError("endpoint-z dense comparison requires the frozen wall mesh")
    walls = source_z[[0, -1]]
    source_values = np.asarray(evaluator(walls, source_r), dtype=float)
    analytic_source = np.asarray(
        evaluator(walls, source_r, z_order=1), dtype=float,
    )
    live_source = np.asarray(contract.z_first_s_jets(
        state_name=state_name,
        radius=source_r,
        wall_value_s_jets=(source_values,),
    )[0], dtype=float)
    dense_values = np.asarray(evaluator(walls, dense_r), dtype=float)
    analytic_dense = np.asarray(
        evaluator(walls, dense_r, z_order=1), dtype=float,
    )
    live_dense = np.asarray(contract.z_first_s_jets(
        state_name=state_name,
        radius=dense_r,
        wall_value_s_jets=(dense_values,),
    )[0], dtype=float)
    source_stored_vs_live = _scaled_score(stored, live_source)
    source_analytic_vs_stored = _scaled_score(analytic_source, stored)
    dense_analytic_vs_live = _scaled_score(analytic_dense, live_dense)
    gates = {
        "source_stored_vs_live_contract": (
            source_stored_vs_live["scaled_Linf"] <= 1e-12
        ),
        "source_analytic_vs_stored": (
            source_analytic_vs_stored["scaled_Linf"] <= 1e-12
        ),
        "dense_analytic_vs_live_contract": (
            dense_analytic_vs_live["scaled_Linf"] <= 1e-10
        ),
    }
    return MappingProxyType({
        "state_name": state_name,
        "native_channel_order": NATIVE_CHANNEL_ORDER,
        "source_stored_vs_live_contract": source_stored_vs_live,
        "source_analytic_vs_stored": source_analytic_vs_stored,
        "dense_analytic_vs_live_contract": dense_analytic_vs_live,
        "dense_r_sha256": hash_arrays(dense_r),
        "gates": gates,
        "passed": bool(all(gates.values())),
    })


def score_time_symmetric_velocity_endpoint_z(velocity_z):
    """Require the persisted velocity endpoint-z lane to be exact IEEE +0."""
    velocity_z = np.asarray(velocity_z, dtype=float)
    if velocity_z.ndim != 3 or velocity_z.shape[0] != 2 or velocity_z.shape[-1] != 8:
        raise ValueError("velocity endpoint-z data must have shape (2,nr,8)")
    positive_zero = _positive_zero(velocity_z)
    return MappingProxyType({
        "shape": tuple(velocity_z.shape),
        "bitwise_positive_zero": positive_zero,
        "passed": positive_zero,
        "fingerprint": _hash_named_arrays((("velocity_z", velocity_z),)),
    })


def score_state_outer_derivative_reproduction(
    state, dense_z, *, require_frozen=True,
):
    """Independently replay one persisted all-channel outer derivative bundle.

    The target is rebuilt with a fresh degree-five spline from the persisted
    source derivative array.  The live contract query is deliberately not
    called, so agreement cannot be obtained by comparing a contract with
    itself.  Compact-wall-owned endpoints are retained in the report but are
    excluded from both source and dense open-face gates.
    """
    state_name = str(getattr(state, "state_name", ""))
    if state_name not in ("position", "acceleration"):
        raise ValueError("outer-derivative scorer requires position or acceleration")
    source_z = np.asarray(getattr(state, "source_z", ()), dtype=float)
    source_r = np.asarray(getattr(state, "source_r", ()), dtype=float)
    dense_z = np.asarray(dense_z, dtype=float)
    evaluator = getattr(state, "evaluate_physical_channels", None)
    contract = getattr(state, "outer_open_face_contract", None)
    if (
        source_z.ndim != 1
        or source_r.ndim != 1
        or dense_z.ndim != 1
        or len(source_z) < 6
        or len(source_r) < 7
        or len(dense_z) < 3
        or np.any(np.diff(source_z) <= 0.0)
        or np.any(np.diff(source_r) <= 0.0)
        or np.any(np.diff(dense_z) <= 0.0)
        or source_r[0] != 0.0
        or dense_z[0] != source_z[0]
        or dense_z[-1] != source_z[-1]
        or evaluator is None
        or not callable(evaluator)
    ):
        raise ValueError("outer-derivative state or coordinates are invalid")
    if require_frozen and hash_arrays(dense_z) != DENSE_OUTER_SHA256:
        raise ValueError("outer-derivative comparison requires the frozen outer mesh")

    if isinstance(contract, Protocol125OuterOpenFaceDerivativeContract):
        position_contract = contract.position_contract
        if state_name == "position":
            target = np.asarray(position_contract.position_r_first, dtype=float)
            ownership = np.asarray(position_contract.ownership_mask, dtype=bool)
        else:
            target = np.asarray(contract.acceleration_r_first, dtype=float)
            ownership = np.asarray(contract.acceleration_ownership_mask, dtype=bool)
    elif (
        state_name == "position"
        and isinstance(contract, Protocol125PositionOuterOpenFaceDerivativeContract)
    ):
        position_contract = contract
        target = np.asarray(position_contract.position_r_first, dtype=float)
        ownership = np.asarray(position_contract.ownership_mask, dtype=bool)
    else:
        raise TypeError("state does not carry the required Protocol-125 outer contract")
    expected_target = (len(source_z), len(NATIVE_CHANNEL_ORDER))
    if (
        not np.array_equal(source_z, np.asarray(position_contract.source_z))
        or not np.array_equal(source_r, np.asarray(position_contract.source_r))
        or target.shape != expected_target
        or not np.all(np.isfinite(target))
        or not np.array_equal(
            ownership, np.ones(len(NATIVE_CHANNEL_ORDER), dtype=bool),
        )
    ):
        raise ValueError("outer-derivative target provenance is incomplete")

    radius = np.asarray([source_r[-1]], dtype=float)
    source_found = np.asarray(
        evaluator(source_z, radius, r_order=1), dtype=float,
    )[:, 0]
    target_spline = make_interp_spline(source_z, target, k=5, axis=0)
    dense_target = np.asarray(target_spline(dense_z), dtype=float)
    dense_found = np.asarray(
        evaluator(dense_z, radius, r_order=1), dtype=float,
    )[:, 0]
    if not all(
        value.shape == expected and np.all(np.isfinite(value))
        for value, expected in (
            (source_found, expected_target),
            (dense_target, (len(dense_z), len(NATIVE_CHANNEL_ORDER))),
            (dense_found, (len(dense_z), len(NATIVE_CHANNEL_ORDER))),
        )
    ):
        raise ValueError("outer-derivative representation evaluation is invalid")

    source_open = np.zeros(len(source_z), dtype=bool)
    source_open[1:-1] = True
    if not np.array_equal(source_open, np.asarray(position_contract.open_compact_mask)):
        raise ValueError("outer compact-wall ownership mask changed")
    dense_open = np.zeros(len(dense_z), dtype=bool)
    dense_open[1:-1] = True
    source_records = {}
    dense_records = {}
    gates = {}
    for channel, name in enumerate(NATIVE_CHANNEL_ORDER):
        source_score = _scaled_score(
            source_found[source_open, channel], target[source_open, channel],
        )
        dense_score = _scaled_score(
            dense_found[dense_open, channel], dense_target[dense_open, channel],
        )
        source_pass = source_score["scaled_Linf"] <= 1e-12
        dense_pass = dense_score["scaled_Linf"] <= 1e-10
        source_records[name] = {**source_score, "ceiling": 1e-12, "passed": source_pass}
        dense_records[name] = {**dense_score, "ceiling": 1e-10, "passed": dense_pass}
        gates[f"source_{name}"] = source_pass
        gates[f"dense_{name}"] = dense_pass
    endpoint_scale = np.maximum.reduce((
        np.ones((2, len(NATIVE_CHANNEL_ORDER))),
        np.abs(dense_found[[0, -1]]),
        np.abs(dense_target[[0, -1]]),
    ))
    endpoint_error = np.abs(
        dense_found[[0, -1]]-dense_target[[0, -1]],
    )/endpoint_scale
    fingerprint = _hash_named_arrays((
        ("source_z", source_z),
        ("source_r", source_r),
        ("source_target", target),
        ("source_found", source_found),
        ("dense_z", dense_z),
        ("dense_target", dense_target),
        ("dense_found", dense_found),
        ("ownership", ownership),
    ))
    return MappingProxyType({
        "complete": True,
        "provenance_valid": True,
        "state_name": state_name,
        "native_channel_order": NATIVE_CHANNEL_ORDER,
        "source_records": source_records,
        "dense_records": dense_records,
        "endpoint_report_not_scored": MappingProxyType({
            "lower_scaled_absolute_by_channel": _immutable(endpoint_error[0]),
            "upper_scaled_absolute_by_channel": _immutable(endpoint_error[1]),
        }),
        "source_open_sample_count": int(np.count_nonzero(source_open)),
        "dense_open_sample_count": int(np.count_nonzero(dense_open)),
        "compact_endpoints_excluded_from_score": True,
        "contract_target_query_called": False,
        "fresh_degree_five_target_reconstruction": True,
        "dense_outer_sha256": hash_arrays(dense_z),
        "contract_identifier": str(position_contract.identifier),
        "gates": MappingProxyType(gates),
        "passed": bool(all(gates.values())),
        "fingerprint": fingerprint,
    })


def _required_wall_array(mapping, name, shape):
    if not isinstance(mapping, Mapping) or name not in mapping:
        raise ValueError(f"wall-profile input is missing {name}")
    value = np.asarray(mapping[name], dtype=float)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"wall-profile input {name} is invalid")
    return value


def _wall_coefficients(phi, background):
    required = (
        "wall_stiffness", "v0", "v1", "beta_a", "beta_b",
        "wall_potential_a", "wall_potential_b",
    )
    if not isinstance(background, Mapping) or any(name not in background for name in required):
        raise ValueError("wall-profile background is incomplete")
    gamma = float(background["wall_stiffness"])
    targets = np.asarray((background["v0"], background["v1"]), dtype=float)[:, None]
    bare = np.asarray((background["beta_a"], background["beta_b"]), dtype=float)[:, None]
    stored = np.asarray(
        (background["wall_potential_a"], background["wall_potential_b"]),
        dtype=float,
    )[:, None]
    if not np.isfinite(gamma) or not all(np.all(np.isfinite(value)) for value in (targets, bare, stored)):
        raise ValueError("wall-profile background is nonfinite")
    delta = phi-targets
    potential = 0.5*gamma*delta**2
    branch = np.asarray((1.0, -1.0))[:, None]
    beta = bare+branch*(potential-stored)/6.0
    beta_phi = branch*gamma*delta/6.0
    return gamma, delta, beta, beta_phi


def _row_record(terms, scale):
    terms = np.asarray(terms, dtype=float)
    scale = np.asarray(scale, dtype=float)
    residual = np.sum(terms, axis=0)
    if (
        terms.ndim != 3
        or terms.shape[1] != 2
        or scale.shape != terms.shape[1:]
        or np.any(scale < 1.0)
        or not (np.all(np.isfinite(terms)) and np.all(np.isfinite(scale)))
    ):
        raise ValueError("normalized wall row terms or scale are invalid")
    profile = residual/scale
    return {
        "signed_terms": _immutable(terms),
        "signed_residual": _immutable(residual),
        "positive_scale": _immutable(scale),
        "signed_profile": _immutable(profile),
        "absolute_profile": _immutable(np.abs(profile)),
        "wall_Linf": tuple(float(np.max(np.abs(profile[index]))) for index in range(2)),
        "wall_pass": tuple(bool(np.max(np.abs(profile[index])) < 1e-10) for index in range(2)),
    }


def score_normalized_wall_profiles(position, acceleration, background):
    """Score frozen Phi, chi, and normal-GH position/acceleration profiles."""
    if not isinstance(position, Mapping) or not isinstance(acceleration, Mapping):
        raise TypeError("wall-profile position and acceleration must be mappings")
    seed = np.asarray(position.get("Phi", ()), dtype=float)
    if seed.ndim != 2 or seed.shape[0] != 2 or seed.shape[1] == 0:
        raise ValueError("wall-profile Phi must have shape (2,nr)")
    shape = seed.shape
    phi = _required_wall_array(position, "Phi", shape)
    phi_z = _required_wall_array(position, "Phi_z", shape)
    chi_z = _required_wall_array(position, "chi_z", shape)
    G = _required_wall_array(position, "G", shape)
    G_z = _required_wall_array(position, "G_z", shape)
    H_z = _required_wall_array(position, "H_z", shape)
    if np.any(G <= 0.0):
        raise ValueError("normal-GH wall metric must be positive")
    a_phi = _required_wall_array(acceleration, "a_Phi", shape)
    a_phi_z = _required_wall_array(acceleration, "a_Phi_z", shape)
    a_chi_z = _required_wall_array(acceleration, "a_chi_z", shape)
    a_G = _required_wall_array(acceleration, "a_G", shape)
    a_G_z = _required_wall_array(acceleration, "a_G_z", shape)
    H_ztt = _required_wall_array(acceleration, "H_ztt", shape)
    gamma, delta, beta, beta_phi = _wall_coefficients(phi, background)
    sigma = np.asarray((-1.0, 1.0))[:, None]
    A = np.sqrt(G)

    phi_position_terms = np.stack((
        phi_z,
        sigma*(gamma/2.0)*delta*A,
    ))
    chi_position_terms = np.stack((chi_z,))
    gh_position_terms = np.stack((
        G_z,
        8.0*beta*G**1.5,
        -2.0*H_z*G,
    ))
    phi_acceleration_terms = np.stack((
        a_phi_z,
        sigma*(gamma/2.0)*A*a_phi,
        sigma*(gamma/2.0)*delta*a_G/(2.0*A),
    ))
    chi_acceleration_terms = np.stack((a_chi_z,))
    gh_acceleration_terms = np.stack((
        a_G_z,
        (12.0*beta*A-2.0*H_z)*a_G,
        8.0*beta_phi*a_phi*G**1.5,
        -2.0*H_ztt*G,
    ))

    def scale(terms):
        return np.maximum(1.0, np.sum(np.abs(terms), axis=0))

    records = {
        "position": {
            "Phi": _row_record(phi_position_terms, scale(phi_position_terms)),
            "chi": _row_record(chi_position_terms, scale(chi_position_terms)),
            "normal_GH": _row_record(gh_position_terms, scale(gh_position_terms)),
        },
        "acceleration": {
            "Phi": _row_record(phi_acceleration_terms, scale(phi_acceleration_terms)),
            "chi": _row_record(chi_acceleration_terms, scale(chi_acceleration_terms)),
            "normal_GH": _row_record(gh_acceleration_terms, scale(gh_acceleration_terms)),
        },
    }
    gates = {
        f"{stage}_{row}_{wall}": bool(record["wall_pass"][wall_index])
        for stage, stage_records in records.items()
        for row, record in stage_records.items()
        for wall_index, wall in enumerate(WALL_ORDER)
    }
    fingerprint_items = []
    for stage, stage_records in records.items():
        for row, record in stage_records.items():
            fingerprint_items.extend((
                (f"{stage}_{row}_terms", record["signed_terms"]),
                (f"{stage}_{row}_scale", record["positive_scale"]),
                (f"{stage}_{row}_profile", record["signed_profile"]),
            ))
    return MappingProxyType({
        "wall_order": WALL_ORDER,
        "row_order": WALL_PROFILE_ROWS,
        "records": records,
        "gates": gates,
        "passed": bool(all(gates.values())),
        "fingerprint": _hash_named_arrays(fingerprint_items),
        "gate": "each named row and wall has Linf < 1e-10",
    })


def _valid_sha256(value):
    value = str(value)
    return bool(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _freeze_tree(value):
    if isinstance(value, Mapping):
        return MappingProxyType({
            str(name): _freeze_tree(item) for name, item in value.items()
        })
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_tree(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _scaled_linf(left, right):
    return float(_scaled_score(left, right)["scaled_Linf"])


def _endpoint_conversion_arrays(record, radial_coordinate, label, expected_kind):
    radial_coordinate = np.asarray(radial_coordinate, dtype=float)
    radial_count = len(radial_coordinate)
    required = (
        "native_channel_order", "physical_component_order", "reduced_field_order",
        "physical", "reduced", "axis_image_order", "axis_images",
        "native_acceleration_z_s_first", "ownership_mask",
        "physical_sha256", "reduced_sha256", "s_jet_inputs_sha256",
        "ownership_mask_sha256",
        "q4_axis_source", "q5_axis_source", "conversion_kind",
        "normal_tangential_positive_zero_required", "fingerprint",
    )
    if not isinstance(record, Mapping) or set(record) != set(required):
        raise ValueError(f"{label} endpoint conversion record is incomplete")
    if (
        tuple(record["native_channel_order"]) != tuple(NATIVE_CHANNEL_ORDER)
        or tuple(record["physical_component_order"])
        != tuple(COORDINATE_COMPONENT_ORDER)
        or tuple(record["reduced_field_order"]) != tuple(REDUCED_FIELD_ORDER)
        or tuple(record["axis_image_order"])
        != ("partial_r2_N_z", "partial_r_T_z")
        or str(record["conversion_kind"]) != str(expected_kind)
        or not _valid_sha256(record["fingerprint"])
        or any(not _valid_sha256(record[name]) for name in (
            "physical_sha256", "reduced_sha256", "s_jet_inputs_sha256",
            "ownership_mask_sha256",
        ))
    ):
        raise ValueError(f"{label} endpoint conversion provenance differs")
    expected_owner = expected_kind == "row-implied-live-compact-contract"
    if type(record["normal_tangential_positive_zero_required"]) is not bool or (
        record["normal_tangential_positive_zero_required"] is not expected_owner
    ):
        raise ValueError(f"{label} endpoint conversion ownership flag differs")
    arrays = {
        "physical": np.asarray(record["physical"], dtype=float),
        "reduced": np.asarray(record["reduced"], dtype=float),
        "native_acceleration_z_s_first": np.asarray(
            record["native_acceleration_z_s_first"], dtype=float,
        ),
        "ownership_mask": np.asarray(record["ownership_mask"], dtype=bool),
        "q4_axis_source": np.asarray(record["q4_axis_source"], dtype=float),
        "q5_axis_source": np.asarray(record["q5_axis_source"], dtype=float),
        "axis_images": np.asarray(record["axis_images"], dtype=float),
    }
    expected_shapes = {
        "physical": (2, int(radial_count), len(COORDINATE_COMPONENT_ORDER)),
        "reduced": (2, int(radial_count), len(REDUCED_FIELD_ORDER)),
        "native_acceleration_z_s_first": (
            2, int(radial_count), len(NATIVE_CHANNEL_ORDER),
        ),
        "ownership_mask": (len(NATIVE_CHANNEL_ORDER),),
        "q4_axis_source": (2,),
        "q5_axis_source": (2,),
        "axis_images": (2, 2),
    }
    if any(
        arrays[name].shape != expected_shapes[name]
        or not np.all(np.isfinite(arrays[name]))
        for name in arrays
    ):
        raise ValueError(f"{label} endpoint conversion arrays are invalid")
    if (
        radial_coordinate.ndim != 1
        or radial_count < 2
        or radial_coordinate[0] != 0.0
        or np.signbit(radial_coordinate[0])
        or np.any(np.diff(radial_coordinate) <= 0.0)
        or not np.all(np.isfinite(radial_coordinate))
        or not np.array_equal(
            arrays["ownership_mask"],
            np.asarray(COMPACT_NATIVE_OWNERSHIP_MASK, dtype=bool),
        )
    ):
        raise ValueError(f"{label} endpoint conversion ownership or radius differs")
    expected_hashes = {
        "physical_sha256": hash_arrays(arrays["physical"]),
        "reduced_sha256": hash_arrays(arrays["reduced"]),
        "s_jet_inputs_sha256": hash_arrays(
            arrays["native_acceleration_z_s_first"],
        ),
        "ownership_mask_sha256": hash_arrays(arrays["ownership_mask"]),
    }
    if any(str(record[name]) != digest for name, digest in expected_hashes.items()):
        raise ValueError(f"{label} endpoint conversion constituent hash differs")
    native_index = {
        name: NATIVE_CHANNEL_ORDER.index(name) for name in NATIVE_CHANNEL_ORDER
    }
    s_first = arrays["native_acceleration_z_s_first"]
    q4_from_s_jet = (
        s_first[:, 0, native_index["h_rr"]]
        - s_first[:, 0, native_index["h_perp"]]
    )/float(radial_coordinate[-1])**2
    if (
        not np.array_equal(
            arrays["q4_axis_source"], arrays["reduced"][:, 0, 4],
        )
        or not np.array_equal(arrays["q4_axis_source"], q4_from_s_jet)
        or not np.array_equal(
            arrays["q5_axis_source"], arrays["reduced"][:, 0, 5],
        )
    ):
        raise ValueError(f"{label} q4/q5 endpoint sources do not reassemble")
    if not np.array_equal(
        arrays["axis_images"],
        np.stack((
            2.0*arrays["q4_axis_source"],
            arrays["q5_axis_source"],
        ), axis=-1),
    ):
        raise ValueError(f"{label} q4/q5 endpoint images do not reassemble")
    return arrays


def _analytic_acceleration_endpoint_arrays(state, radius):
    source_z = np.asarray(state.source_z, dtype=float)
    walls = source_z[[0, -1]]
    physical = np.asarray(
        state.evaluate_coordinate_components(walls, radius, z_order=1),
        dtype=float,
    )
    reduced = np.asarray(
        state.evaluate_reduced(walls, radius, z_order=1),
        dtype=float,
    )
    axis_images = np.stack((
        2.0*reduced[:, 0, 4], reduced[:, 0, 5],
    ), axis=-1)
    if not all(np.all(np.isfinite(value)) for value in (
        physical, reduced, axis_images,
    )):
        raise ValueError("analytic acceleration endpoint arrays are nonfinite")
    return {
        "physical": physical,
        "reduced": reduced,
        "axis_images": axis_images,
    }


def _endpoint_comparison(left, right, ceiling):
    score = _scaled_score(left, right)
    ceiling = float(ceiling)
    return MappingProxyType({
        **score,
        "ceiling": ceiling,
        "passed": bool(score["scaled_Linf"] <= ceiling),
    })


def _endpoint_lane(comparison_specs, *, source_fingerprints):
    comparisons = {
        name: _endpoint_comparison(left, right, ceiling)
        for name, left, right, ceiling in comparison_specs
    }
    return MappingProxyType({
        "comparison_order": tuple(comparisons),
        "comparisons": MappingProxyType(comparisons),
        "source_ceiling": SOURCE_ACCELERATION_ENDPOINT_CEILING,
        "dense_ceiling": DENSE_ACCELERATION_ENDPOINT_CEILING,
        "q4_q5_axis_images_scored_separately": True,
        "conversion_fingerprints": MappingProxyType(dict(source_fingerprints)),
        "passed": bool(all(record["passed"] for record in comparisons.values())),
    })


def score_acceleration_endpoint_conversion_pair(
    final_pair,
    row_implied_source,
    row_implied_dense,
    direct_fd_source,
    dense_r,
):
    """Score the row-owned and native-Dz7 acceleration endpoint routes.

    Source-node comparisons use the frozen ``1e-12`` ceiling; dense-wall
    analytic comparisons use ``1e-10``.  Physical, complete reduced, and
    explicit q4/q5 axis-image arrays are gated independently, so coordinate
    zeros cannot hide a regular-coefficient mismatch.
    """
    if not isinstance(final_pair, RadialFirstConstrainedHermitePair):
        raise TypeError("acceleration endpoint conversion requires the final Q53/Q33 pair")
    source_r = np.asarray(final_pair.primary.acceleration.source_r, dtype=float)
    dense_r = np.asarray(dense_r, dtype=float)
    frozen_dense = np.asarray(frozen_validation_meshes()["dense_wall"]["r"])
    if (
        source_r.ndim != 1
        or len(source_r) < 7
        or source_r[0] != 0.0
        or np.signbit(source_r[0])
        or np.any(np.diff(source_r) <= 0.0)
        or not np.array_equal(dense_r, frozen_dense)
        or hash_arrays(dense_r) != DENSE_WALL_SHA256
    ):
        raise ValueError("acceleration endpoint conversion meshes differ from Protocol 125")
    for state in (
        final_pair.primary.acceleration, final_pair.comparator.acceleration,
    ):
        if not np.array_equal(np.asarray(state.source_r), source_r):
            raise ValueError("Q53/Q33 acceleration source radii differ")

    row_source = _endpoint_conversion_arrays(
        row_implied_source, source_r, "source row-implied",
        "row-implied-live-compact-contract",
    )
    row_dense = _endpoint_conversion_arrays(
        row_implied_dense, dense_r, "dense row-implied",
        "row-implied-live-compact-contract",
    )
    direct_source = _endpoint_conversion_arrays(
        direct_fd_source, source_r, "source Dz7",
        "independent-source-Dz7-comparator",
    )
    conversion_fingerprints = {
        "row_implied_source": str(row_implied_source["fingerprint"]),
        "row_implied_dense": str(row_implied_dense["fingerprint"]),
        "direct_Dz7_source": str(direct_fd_source["fingerprint"]),
    }
    lanes = {
        ACCELERATION_ENDPOINT_CONVERSION_LANES[0]: _endpoint_lane(
            tuple(
                (
                    f"source_Dz7_vs_row_implied_{component}",
                    direct_source[component], row_source[component],
                    SOURCE_ACCELERATION_ENDPOINT_CEILING,
                )
                for component in ("physical", "reduced", "axis_images")
            ),
            source_fingerprints=conversion_fingerprints,
        ),
    }
    for representation, state in (
        ("Q53", final_pair.primary.acceleration),
        ("Q33", final_pair.comparator.acceleration),
    ):
        source_analytic = _analytic_acceleration_endpoint_arrays(state, source_r)
        dense_analytic = _analytic_acceleration_endpoint_arrays(state, dense_r)
        specifications = []
        for component in ("physical", "reduced", "axis_images"):
            specifications.extend((
                (
                    f"source_analytic_vs_row_implied_{component}",
                    source_analytic[component], row_source[component],
                    SOURCE_ACCELERATION_ENDPOINT_CEILING,
                ),
                (
                    f"source_analytic_vs_Dz7_{component}",
                    source_analytic[component], direct_source[component],
                    SOURCE_ACCELERATION_ENDPOINT_CEILING,
                ),
                (
                    f"dense_analytic_vs_row_implied_{component}",
                    dense_analytic[component], row_dense[component],
                    DENSE_ACCELERATION_ENDPOINT_CEILING,
                ),
            ))
        lanes[f"{representation}_acceleration_endpoint_conversion"] = _endpoint_lane(
            tuple(specifications),
            source_fingerprints=conversion_fingerprints,
        )
    if tuple(lanes) != ACCELERATION_ENDPOINT_CONVERSION_LANES:
        raise AssertionError("acceleration endpoint conversion lane order changed")
    return MappingProxyType({
        "lane_order": ACCELERATION_ENDPOINT_CONVERSION_LANES,
        "lanes": MappingProxyType(lanes),
        "source_r_sha256": hash_arrays(source_r),
        "dense_r_sha256": hash_arrays(dense_r),
        "passed": bool(all(record["passed"] for record in lanes.values())),
    })


def wall_profile_evidence_fingerprint(record):
    """Recompute the immutable summary digest of a two-mesh evidence record."""
    coordinates = record["coordinates"]
    recipes = record["derivative_recipes"]
    symmetry = record["time_symmetry"]
    context = record["live_compact_context"]
    inputs = record["input_hashes"]
    meshes = record["meshes"]
    gates = record["named_row_wall_gates"]
    context_fields = (
        "source_position_reproduction_scaled_Linf",
        "source_acceleration_reproduction_scaled_Linf",
        "source_normal_context_reproduction_scaled_Linf",
        "source_second_normal_context_reproduction_scaled_Linf",
        "position_and_acceleration_share_live_contract",
        "source_normal_context_present",
        "source_second_normal_context_present",
        "passed",
    )
    items = (
        ("protocol_identifier", np.asarray(record["protocol_identifier"])),
        ("parent_label", np.asarray(record["parent_label"])),
        ("parent_identity", np.asarray(record["parent_identity"])),
        ("source_fingerprint", np.asarray(record["source_fingerprint"])),
        ("endpoint_fingerprint", np.asarray(record["endpoint_fingerprint"])),
        ("source_z", np.asarray(coordinates["source_z"])),
        ("source_r", np.asarray(coordinates["source_r"])),
        ("dense_r", np.asarray(coordinates["dense_r"])),
        (
            "coordinate_hashes",
            np.asarray(tuple(
                str(coordinates[name]) for name in (
                    "source_z_sha256", "source_r_sha256",
                    "source_pair_sha256", "source_wall_coordinate_sha256",
                    "dense_r_sha256", "dense_wall_coordinate_sha256",
                )
            )),
        ),
        (
            "derivative_recipes",
            np.asarray((
                str(recipes["source_recipe"]),
                str(recipes["dense_recipe"]),
                str(recipes["dense_source_recipe"]),
            )),
        ),
        ("source_stencil_width", np.asarray(recipes["source_stencil_width"])),
        (
            "input_hashes",
            np.asarray(tuple(str(inputs[name]) for name in WALL_PROFILE_INPUT_HASH_KEYS)),
        ),
        (
            "time_symmetry",
            np.asarray((
                str(symmetry["source_velocity_shape"]),
                str(symmetry["source_velocity_sha256"]),
                str(symmetry["source_positive_zero_reference_sha256"]),
                str(symmetry["dense_velocity_shape"]),
                str(symmetry["dense_velocity_sha256"]),
                str(symmetry["dense_positive_zero_reference_sha256"]),
                str(symmetry["dense_velocity_recipe"]),
                str(bool(symmetry["source_bitwise_positive_zero"])),
                str(bool(symmetry["dense_bitwise_positive_zero"])),
                str(bool(symmetry["passed"])),
            )),
        ),
        ("compact_identifier", np.asarray(context["contract_identifier"])),
        ("compact_fingerprint", np.asarray(context["contract_fingerprint"])),
        (
            "compact_context_summary",
            np.asarray(tuple(context[name] for name in context_fields)),
        ),
        (
            "compact_context_hashes",
            np.asarray((
                str(context["dense_source_normal_sha256"]),
                str(context["dense_source_second_normal_sha256"]),
            )),
        ),
        ("source_score_fingerprint", np.asarray(meshes["source"]["fingerprint"])),
        ("dense_score_fingerprint", np.asarray(meshes["dense"]["fingerprint"])),
        ("named_gate_order", np.asarray(tuple(record["named_gate_order"]))),
        (
            "named_gate_values",
            np.asarray(tuple(bool(gates[name]) for name in record["named_gate_order"])),
        ),
        ("named_row_wall_passed", np.asarray(record["named_row_wall_passed"])),
        ("constituent_logical_AND", np.asarray(record["constituent_logical_AND"])),
        ("complete", np.asarray(record["complete"])),
        ("provenance_valid", np.asarray(record["provenance_valid"])),
        ("passed", np.asarray(record["passed"])),
    )
    return _hash_named_arrays(items)


def _protocol125_wall_profile_parent_context(parent):
    if not isinstance(parent, Mapping):
        raise TypeError("wall-profile evidence parent must be a mapping")
    required = (
        "label", "parent_identity", "z", "r", "position", "selector_q",
        "phi", "reference_q", "reference_phi", "background",
    )
    if any(name not in parent for name in required):
        raise ValueError("wall-profile evidence parent is incomplete")
    label = str(parent["label"])
    identity = str(parent["parent_identity"])
    if label not in ("N0", "N1") or not _valid_sha256(identity):
        raise ValueError("wall-profile evidence parent label or identity is invalid")
    z = np.asarray(parent["z"], dtype=float)
    r = np.asarray(parent["r"], dtype=float)
    position = np.asarray(parent["position"], dtype=float)
    shape = (len(z), len(r))
    if (
        z.ndim != 1
        or r.ndim != 1
        or len(z) < 7
        or len(r) < 7
        or np.any(np.diff(z) <= 0.0)
        or np.any(np.diff(r) <= 0.0)
        or r[0] != 0.0
        or np.signbit(r[0])
        or position.shape != shape+(9,)
        or not all(np.all(np.isfinite(value)) for value in (z, r, position))
    ):
        raise ValueError("wall-profile evidence source arrays are invalid")
    identity_arrays = {
        name: np.asarray(parent[name], dtype=float)
        for name in ("selector_q", "phi", "reference_q", "reference_phi")
    }
    if any(value.shape != shape for value in identity_arrays.values()):
        raise ValueError("wall-profile evidence identity arrays are invalid")
    reproduced = hash_arrays(
        np.asarray(label), z, r, position,
        identity_arrays["selector_q"], identity_arrays["phi"],
        identity_arrays["reference_q"], identity_arrays["reference_phi"],
    )
    if reproduced != identity:
        raise ValueError("wall-profile evidence parent identity does not reproduce")
    background = parent["background"]
    if not isinstance(background, Mapping):
        raise ValueError("wall-profile evidence background is missing")
    return label, identity, z, r, position, background


def _native_width_seven_wall_derivative(values, z):
    values = np.asarray(values, dtype=float)
    z = np.asarray(z, dtype=float)
    if values.ndim != 3 or values.shape[0] != len(z):
        raise ValueError("native wall derivative input has the wrong shape")
    operator = derivative_matrix(z, 1, 7)
    if hasattr(operator, "toarray"):
        operator = operator.toarray()
    operator = np.asarray(operator, dtype=float)
    result = np.einsum("wi,irc->wrc", operator[[0, -1]], values)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("native width-seven wall derivative is nonfinite")
    return result


def build_protocol125_wall_profile_evidence(
    parent,
    completed_velocity,
    compatible_acceleration,
    source_triplet,
    final_pair,
):
    """Build the source-node plus dense termwise Protocol-125 wall audit.

    Source-node derivatives are formed directly with the native width-seven
    compact operator.  Dense derivatives are analytic Q53 queries, while the
    two normal gauge-source traces are evaluated from the *live* compact
    source and source-second radial contexts.  Neither route is allowed to
    stand in for the other.
    """
    label, identity, z, r, position, background = (
        _protocol125_wall_profile_parent_context(parent)
    )
    velocity = np.asarray(completed_velocity, dtype=float)
    acceleration = np.asarray(compatible_acceleration, dtype=float)
    expected = position.shape
    if (
        velocity.shape != expected
        or acceleration.shape != expected
        or not all(np.all(np.isfinite(value)) for value in (velocity, acceleration))
    ):
        raise ValueError("wall-profile velocity or acceleration is invalid")
    if not isinstance(source_triplet, Mapping):
        raise TypeError("wall-profile source triplet must be a mapping")
    source_arrays = {}
    for name in ("source", "source_time", "source_second_time"):
        value = np.asarray(source_triplet.get(name), dtype=float)
        if value.shape != (len(z), len(r), 3) or not np.all(np.isfinite(value)):
            raise ValueError(f"wall-profile source triplet {name} is invalid")
        source_arrays[name] = value

    if not isinstance(final_pair, RadialFirstConstrainedHermitePair):
        raise TypeError("wall-profile evidence requires the final Q53/Q33 pair")
    q53 = final_pair.primary
    position_state = q53.position
    acceleration_state = q53.acceleration
    if position_state.z_degree != 5 or acceleration_state.z_degree != 5:
        raise ValueError("wall-profile dense route is not the final Q53 state")
    if not (
        np.array_equal(position_state.source_z, z)
        and np.array_equal(position_state.source_r, r)
        and np.array_equal(acceleration_state.source_z, z)
        and np.array_equal(acceleration_state.source_r, r)
    ):
        raise ValueError("wall-profile final-state source coordinates differ")
    contract = position_state.compact_wall_contract
    if not (
        contract is acceleration_state.compact_wall_contract
        and contract is final_pair.comparator.position.compact_wall_contract
        and contract is final_pair.comparator.acceleration.compact_wall_contract
        and isinstance(contract, NativeNormalizedCompactWallContract)
    ):
        raise ValueError("wall-profile states do not share the live compact contract")
    if (
        contract.source_normal_context is None
        or contract.source_second_normal_context is None
    ):
        raise ValueError("wall-profile live source contexts are incomplete")
    background_names = (
        "wall_stiffness", "v0", "v1", "beta_a", "beta_b",
        "wall_potential_a", "wall_potential_b",
    )
    background_values = tuple(float(background[name]) for name in background_names)
    if background_values != tuple(contract.background_values):
        raise ValueError("wall-profile background differs from the live compact contract")

    position_reproduction = _scaled_linf(
        position_state.evaluate_reduced(z, r), position,
    )
    acceleration_reproduction = _scaled_linf(
        acceleration_state.evaluate_reduced(z, r), acceleration,
    )
    source_normal = source_arrays["source"][[0, -1], :, 1]
    source_second_normal = source_arrays["source_second_time"][[0, -1], :, 1]
    live_source_normal = np.asarray(
        contract.source_normal_context.jets(r, 1)[0][:, :, 0], dtype=float,
    )
    live_source_second_normal = np.asarray(
        contract.source_second_normal_context.jets(r, 1)[0][:, :, 0], dtype=float,
    )
    source_normal_reproduction = _scaled_linf(live_source_normal, source_normal)
    source_second_reproduction = _scaled_linf(
        live_source_second_normal, source_second_normal,
    )
    context_pass = bool(
        position_reproduction <= 1e-12
        and acceleration_reproduction <= 1e-12
        and source_normal_reproduction <= 1e-12
        and source_second_reproduction <= 1e-12
    )

    position_z = _native_width_seven_wall_derivative(position, z)
    acceleration_z = _native_width_seven_wall_derivative(acceleration, z)
    source_position = {
        "Phi": position[[0, -1], :, 7],
        "Phi_z": position_z[:, :, 7],
        "chi_z": position_z[:, :, 8],
        "G": position[[0, -1], :, 6],
        "G_z": position_z[:, :, 6],
        "H_z": source_normal,
    }
    source_acceleration = {
        "a_Phi": acceleration[[0, -1], :, 7],
        "a_Phi_z": acceleration_z[:, :, 7],
        "a_chi_z": acceleration_z[:, :, 8],
        "a_G": acceleration[[0, -1], :, 6],
        "a_G_z": acceleration_z[:, :, 6],
        "H_ztt": source_second_normal,
    }
    source_score = score_normalized_wall_profiles(
        source_position, source_acceleration, background,
    )

    dense_r = frozen_validation_meshes()["dense_wall"]["r"]
    walls = z[[0, -1]]
    dense_position_values = np.asarray(
        position_state.evaluate_physical_channels(walls, dense_r), dtype=float,
    )
    dense_position_z = np.asarray(
        position_state.evaluate_physical_channels(
            walls, dense_r, z_order=1,
        ), dtype=float,
    )
    dense_acceleration_values = np.asarray(
        acceleration_state.evaluate_physical_channels(walls, dense_r), dtype=float,
    )
    dense_acceleration_z = np.asarray(
        acceleration_state.evaluate_physical_channels(
            walls, dense_r, z_order=1,
        ), dtype=float,
    )
    dense_source_normal = np.asarray(
        contract.source_normal_context.jets(dense_r, 1)[0][:, :, 0], dtype=float,
    )
    dense_source_second_normal = np.asarray(
        contract.source_second_normal_context.jets(dense_r, 1)[0][:, :, 0],
        dtype=float,
    )
    channel = {name: index for index, name in enumerate(NATIVE_CHANNEL_ORDER)}
    dense_position = {
        "Phi": dense_position_values[:, :, channel["Phi"]],
        "Phi_z": dense_position_z[:, :, channel["Phi"]],
        "chi_z": dense_position_z[:, :, channel["chi"]],
        "G": dense_position_values[:, :, channel["h_zz"]],
        "G_z": dense_position_z[:, :, channel["h_zz"]],
        "H_z": dense_source_normal,
    }
    dense_acceleration = {
        "a_Phi": dense_acceleration_values[:, :, channel["Phi"]],
        "a_Phi_z": dense_acceleration_z[:, :, channel["Phi"]],
        "a_chi_z": dense_acceleration_z[:, :, channel["chi"]],
        "a_G": dense_acceleration_values[:, :, channel["h_zz"]],
        "a_G_z": dense_acceleration_z[:, :, channel["h_zz"]],
        "H_ztt": dense_source_second_normal,
    }
    dense_score = score_normalized_wall_profiles(
        dense_position, dense_acceleration, background,
    )

    dense_velocity = np.zeros((2, len(dense_r), 9), dtype=float)
    source_positive_zero = _positive_zero(velocity)
    dense_positive_zero = _positive_zero(dense_velocity)
    time_symmetry = {
        "source_velocity_shape": tuple(velocity.shape),
        "source_velocity_sha256": hash_arrays(velocity),
        "source_positive_zero_reference_sha256": hash_arrays(
            np.zeros_like(velocity),
        ),
        "source_bitwise_positive_zero": source_positive_zero,
        "dense_velocity_shape": tuple(dense_velocity.shape),
        "dense_velocity_sha256": hash_arrays(dense_velocity),
        "dense_positive_zero_reference_sha256": hash_arrays(
            np.zeros_like(dense_velocity),
        ),
        "dense_bitwise_positive_zero": dense_positive_zero,
        "dense_velocity_recipe": "exact-positive-zero-time-symmetric-extension",
        "passed": bool(source_positive_zero and dense_positive_zero),
    }
    coordinates = {
        "source_z": z,
        "source_r": r,
        "dense_r": dense_r,
        "source_z_sha256": hash_arrays(z),
        "source_r_sha256": hash_arrays(r),
        "source_pair_sha256": hash_arrays(z, r),
        "source_wall_coordinate_sha256": hash_arrays(walls, r),
        "dense_r_sha256": hash_arrays(dense_r),
        "dense_wall_coordinate_sha256": hash_arrays(walls, dense_r),
    }
    recipes = {
        "source_recipe": SOURCE_WALL_PROFILE_DERIVATIVE_RECIPE,
        "source_stencil_width": 7,
        "dense_recipe": DENSE_WALL_PROFILE_DERIVATIVE_RECIPE,
        "dense_source_recipe": DENSE_WALL_SOURCE_RECIPE,
    }
    input_hashes = {
        "completed_position_sha256": hash_arrays(position),
        "completed_velocity_sha256": hash_arrays(velocity),
        "completed_acceleration_sha256": hash_arrays(acceleration),
        "source_sha256": hash_arrays(source_arrays["source"]),
        "source_time_sha256": hash_arrays(source_arrays["source_time"]),
        "source_second_time_sha256": hash_arrays(
            source_arrays["source_second_time"],
        ),
        "Q53_position_state_sha256": position_state.fingerprint(),
        "Q53_acceleration_state_sha256": acceleration_state.fingerprint(),
    }
    if tuple(input_hashes) != WALL_PROFILE_INPUT_HASH_KEYS or not all(
        _valid_sha256(value) for value in input_hashes.values()
    ):
        raise RuntimeError("wall-profile evidence input hash inventory is incomplete")
    live_context = {
        "contract_identifier": str(contract.identifier),
        "contract_fingerprint": str(position_state.compact_wall_contract_fingerprint),
        "position_and_acceleration_share_live_contract": True,
        "source_normal_context_present": True,
        "source_second_normal_context_present": True,
        "source_position_reproduction_scaled_Linf": position_reproduction,
        "source_acceleration_reproduction_scaled_Linf": acceleration_reproduction,
        "source_normal_context_reproduction_scaled_Linf": (
            source_normal_reproduction
        ),
        "source_second_normal_context_reproduction_scaled_Linf": (
            source_second_reproduction
        ),
        "dense_source_normal_sha256": hash_arrays(dense_source_normal),
        "dense_source_second_normal_sha256": hash_arrays(
            dense_source_second_normal,
        ),
        "passed": context_pass,
    }
    named_gate_order = tuple(
        f"{mesh}_{stage}_{row}_{wall}"
        for mesh in WALL_PROFILE_MESHES
        for stage in WALL_PROFILE_STAGES
        for row in WALL_PROFILE_ROWS
        for wall in WALL_ORDER
    )
    mesh_scores = {"source": source_score, "dense": dense_score}
    named_gates = {
        f"{mesh}_{name}": bool(mesh_scores[mesh]["gates"][name])
        for mesh in WALL_PROFILE_MESHES
        for name in mesh_scores[mesh]["gates"]
    }
    if tuple(named_gates) != named_gate_order:
        raise RuntimeError("wall-profile evidence named gate order differs")
    named_passed = bool(all(named_gates.values()))
    payload = {
        "protocol_identifier": WALL_PROFILE_EVIDENCE_PROTOCOL_IDENTIFIER,
        "parent_label": label,
        "parent_identity": identity,
        "source_fingerprint": str(q53.source_fingerprint),
        "endpoint_fingerprint": str(q53.endpoint_fingerprint),
        "wall_order": WALL_ORDER,
        "row_order": WALL_PROFILE_ROWS,
        "mesh_order": WALL_PROFILE_MESHES,
        "coordinates": coordinates,
        "derivative_recipes": recipes,
        "input_hashes": input_hashes,
        "time_symmetry": time_symmetry,
        "live_compact_context": live_context,
        "meshes": mesh_scores,
        "named_gate_order": named_gate_order,
        "named_row_wall_gates": named_gates,
        "named_row_wall_passed": named_passed,
        "constituent_logical_AND": True,
        "complete": True,
        "provenance_valid": True,
        "passed": bool(named_passed and time_symmetry["passed"] and context_pass),
        "scientific_execution_authorized": False,
        "artifact_written": False,
    }
    payload["fingerprint"] = wall_profile_evidence_fingerprint(payload)
    return _freeze_tree(payload)
