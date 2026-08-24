"""Pure adapters for the sealed Protocol-120 Q33/Q55 holdout.

The pre-acceleration entry point packages only position and spatial jets from
a position-only parent.  The final entry point additionally packages
acceleration and independently reconstructed source triplets.  Both preserve
the historical comparator recipes on the frozen Protocol-125 V meshes; they
never change a parent, solve a constraint, write an artifact, or authorize
execution.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
from scipy.interpolate import RectBivariateSpline

from bhps.joint_parent_acceleration import represented_position_jet
from bhps.joint_parent_position_pair import PositionOnlyConstrainedHermitePair
from bhps.joint_parent_representation import (
    PARENT_R_MAX,
    SEALED_ADVERSE_COMPARATOR_NAMES,
    SEALED_ADVERSE_COMPARATOR_RECIPES,
    sealed_reduced_q33_q55_adverse_projections,
)
from bhps.joint_parent_refinement_diagnostics import frozen_validation_meshes
from bhps.joint_parent_source_closure import (
    initial_driver_source_triplet_from_acceleration,
)
from bhps.matched_staged_continuum import (
    REDUCED_FIELD_ORDER,
    TensorSplineSurface,
    hash_arrays,
)


V_MESH_NAMES = ("V0", "V1", "V2")
SPATIAL_GROUP_ORDER = (
    "position", "first_spatial", "second_spatial", "acceleration",
)
PREACCELERATION_SPATIAL_GROUP_ORDER = (
    "position", "first_spatial", "second_spatial",
)
SOURCE_TRIPLET_ORDER = ("source", "source_time", "source_second_time")


def _immutable(value):
    array = np.ascontiguousarray(np.asarray(value))
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({str(name): _freeze(item) for name, item in value.items()})
    if isinstance(value, np.ndarray):
        return _immutable(value)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _valid_sha256(value):
    value = str(value)
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _digest_value(digest, name, value):
    encoded = str(name).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    if isinstance(value, Mapping):
        digest.update(b"mapping")
        for child in sorted(value):
            _digest_value(digest, child, value[child])
        return
    if isinstance(value, (tuple, list)):
        digest.update(b"sequence")
        for index, child in enumerate(value):
            _digest_value(digest, index, child)
        return
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype == object or (array.dtype.kind in "fc" and not np.all(np.isfinite(array))):
        raise ValueError(f"legacy holdout fingerprint input {name} is invalid")
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())


def _fingerprint(value):
    digest = hashlib.sha256()
    _digest_value(digest, "legacy_holdout", value)
    return digest.hexdigest()


def _pair_fingerprint(pair):
    coefficients = getattr(pair, "coefficient_arrays", None)
    if coefficients is None or not callable(coefficients):
        raise TypeError("legacy holdout requires a coefficient-backed final pair")
    arrays = coefficients()
    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError("legacy holdout final-pair coefficient record is empty")
    return _fingerprint(arrays)


def _component_orders():
    fields = tuple(str(name) for name in REDUCED_FIELD_ORDER)
    return MappingProxyType({
        "position": fields,
        "first_spatial": tuple(
            f"{field}:{direction}"
            for field in fields for direction in ("z", "r")
        ),
        "second_spatial": tuple(
            f"{field}:{direction}"
            for field in fields for direction in ("zz", "zr", "rr")
        ),
        "acceleration": fields,
    })


def _position_component_orders():
    fields = tuple(str(name) for name in REDUCED_FIELD_ORDER)
    return MappingProxyType({
        "position": fields,
        "first_spatial": tuple(
            f"{field}:{direction}"
            for field in fields for direction in ("z", "r")
        ),
        "second_spatial": tuple(
            f"{field}:{direction}"
            for field in fields for direction in ("zz", "zr", "rr")
        ),
    })


def _position_projection_groups(q, q_z, q_s, q_zz, q_zs, q_ss, radius):
    """Package position and physical spatial derivatives in sealed order."""
    radius = np.asarray(radius, dtype=float)
    scale2 = PARENT_R_MAX**2
    ds_dr = 2.0*radius/scale2
    d2s_dr2 = 2.0/scale2
    q_r = q_s*ds_dr[None, :, None]
    q_zr = q_zs*ds_dr[None, :, None]
    q_rr = q_ss*ds_dr[None, :, None]**2+q_s*d2s_dr2
    groups = {
        "position": np.asarray(q, dtype=float),
        "first_spatial": np.stack((q_z, q_r), axis=-1),
        "second_spatial": np.stack((q_zz, q_zr, q_rr), axis=-1),
    }
    position = groups["position"]
    if tuple(groups) != PREACCELERATION_SPATIAL_GROUP_ORDER or any(
        value.shape[:2] != position.shape[:2]
        or not np.all(np.isfinite(value))
        for value in groups.values()
    ):
        raise ValueError("sealed position-only legacy projection group is invalid")
    return groups


def _sealed_q33_position_projection(q, q_z, z_parent, r_parent, z_target, r_target):
    """Historical clamped-cubic reduced position projection only."""
    surface = TensorSplineSurface.build(
        z_parent,
        r_parent,
        q,
        z_first=q_z,
        degree=3,
        parent_r_max=PARENT_R_MAX,
    )
    return _position_projection_groups(
        surface.evaluate(z_target, r_target),
        surface.evaluate(z_target, r_target, z_order=1),
        surface.evaluate(z_target, r_target, s_order=1),
        surface.evaluate(z_target, r_target, z_order=2),
        surface.evaluate(z_target, r_target, z_order=1, s_order=1),
        surface.evaluate(z_target, r_target, s_order=2),
        r_target,
    )


def _sealed_q55_position_projection(q, z_parent, r_parent, z_target, r_target):
    """Historical zero-smoothing Q55 not-a-knot position projection only."""
    z_parent = np.asarray(z_parent, dtype=float)
    r_parent = np.asarray(r_parent, dtype=float)
    z_target = np.asarray(z_target, dtype=float)
    r_target = np.asarray(r_target, dtype=float)
    q = np.asarray(q, dtype=float)
    s_parent = (r_parent/PARENT_R_MAX)**2
    s_target = (r_target/PARENT_R_MAX)**2
    zz, ss = np.meshgrid(z_target, s_target, indexing="ij")
    splines = tuple(
        RectBivariateSpline(
            z_parent,
            s_parent,
            q[:, :, field],
            kx=5,
            ky=5,
            s=0,
        )
        for field in range(q.shape[-1])
    )

    def evaluate(dx=0, dy=0):
        result = np.empty((len(z_target), len(r_target), q.shape[-1]))
        for field, spline in enumerate(splines):
            result[:, :, field] = spline.ev(
                zz.ravel(), ss.ravel(), dx=int(dx), dy=int(dy),
            ).reshape(len(z_target), len(r_target))
        return result

    return _position_projection_groups(
        evaluate(),
        evaluate(dx=1),
        evaluate(dy=1),
        evaluate(dx=2),
        evaluate(dx=1, dy=1),
        evaluate(dy=2),
        r_target,
    )


def _projection_groups(projected):
    position = np.asarray(projected.reduced_fields, dtype=float)
    first = np.stack((
        np.asarray(projected.reduced_first[1], dtype=float),
        np.asarray(projected.reduced_first[2], dtype=float),
    ), axis=-1)
    second = np.stack((
        np.asarray(projected.reduced_second[1, 1], dtype=float),
        np.asarray(projected.reduced_second[1, 2], dtype=float),
        np.asarray(projected.reduced_second[2, 2], dtype=float),
    ), axis=-1)
    acceleration = np.asarray(projected.reduced_second[0, 0], dtype=float)
    groups = {
        "position": position,
        "first_spatial": first,
        "second_spatial": second,
        "acceleration": acceleration,
    }
    if tuple(groups) != SPATIAL_GROUP_ORDER or any(
        value.shape[:2] != position.shape[:2]
        or not np.all(np.isfinite(value))
        for value in groups.values()
    ):
        raise ValueError("sealed legacy projection group is invalid")
    return groups


def build_protocol125_preacceleration_legacy_position_inputs(
    position_pair,
    *,
    parent_identity,
    v_meshes=None,
):
    """Build the sealed position/spatial holdout without acceleration data.

    The Q33 lane is exactly the historical tensor-product cubic with stored
    compact endpoint ``q_z`` data.  The Q55 lane is exactly the historical
    zero-smoothing, two-directional not-a-knot ``RectBivariateSpline``.  This
    entry point deliberately accepts only a completed position-only Q53/Q33
    pair: no acceleration object, placeholder acceleration, or source-triplet
    bundle can enter the pre-acceleration audit through it.
    """
    if not isinstance(position_pair, PositionOnlyConstrainedHermitePair):
        raise TypeError(
            "pre-acceleration legacy holdout requires a position-only "
            "constrained Q53/Q33 pair"
        )
    identity = str(parent_identity)
    if not _valid_sha256(identity):
        raise ValueError(
            "pre-acceleration legacy holdout parent identity must be a "
            "SHA-256 digest"
        )
    state = position_pair.primary
    if state.state_name != "position" or state.z_degree != 5:
        raise ValueError("pre-acceleration legacy source must be Q53 position")
    if float(state.parent_r_max) != float(PARENT_R_MAX):
        raise ValueError("pre-acceleration legacy source radial scale changed")

    pair_hash_before = _pair_fingerprint(position_pair)
    z_parent = np.asarray(state.source_z, dtype=float)
    r_parent = np.asarray(state.source_r, dtype=float)
    q = state.evaluate_reduced(z_parent, r_parent)
    q_z = state.evaluate_reduced(z_parent, r_parent, z_order=1)
    expected = (len(z_parent), len(r_parent), len(REDUCED_FIELD_ORDER))
    if (
        q.shape != expected
        or q_z.shape != expected
        or not np.all(np.isfinite(q))
        or not np.all(np.isfinite(q_z))
    ):
        raise ValueError("pre-acceleration legacy source position jet is invalid")
    source_position_sha256 = hash_arrays(z_parent, r_parent, q, q_z)

    frozen = frozen_validation_meshes()
    if v_meshes is None:
        v_meshes = {name: frozen[name] for name in V_MESH_NAMES}
    if not isinstance(v_meshes, Mapping) or tuple(v_meshes) != V_MESH_NAMES:
        raise ValueError(
            "pre-acceleration legacy holdout requires ordered V0/V1/V2 meshes"
        )

    grouped = {name: {} for name in SEALED_ADVERSE_COMPARATOR_NAMES}
    coordinate_hashes = {}
    for mesh_name in V_MESH_NAMES:
        try:
            z_target = np.asarray(v_meshes[mesh_name]["z"], dtype=float)
            r_target = np.asarray(v_meshes[mesh_name]["r"], dtype=float)
            supplied_hash = str(v_meshes[mesh_name]["sha256"])
            expected_hash = str(frozen[mesh_name]["sha256"])
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"pre-acceleration legacy {mesh_name} mesh record is incomplete"
            ) from error
        if (
            hash_arrays(z_target, r_target) != expected_hash
            or supplied_hash != expected_hash
        ):
            raise ValueError(
                f"pre-acceleration legacy {mesh_name} mesh differs from Protocol 125"
            )
        coordinate_hashes[mesh_name] = expected_hash
        grouped[SEALED_ADVERSE_COMPARATOR_NAMES[0]][mesh_name] = (
            _sealed_q33_position_projection(
                q,
                q_z,
                z_parent,
                r_parent,
                z_target,
                r_target,
            )
        )
        grouped[SEALED_ADVERSE_COMPARATOR_NAMES[1]][mesh_name] = (
            _sealed_q55_position_projection(
                q,
                z_parent,
                r_parent,
                z_target,
                r_target,
            )
        )

    pair_hash_after = _pair_fingerprint(position_pair)
    if pair_hash_after != pair_hash_before:
        raise RuntimeError(
            "position-only pair changed during sealed legacy holdout evaluation"
        )
    component_orders = _position_component_orders()
    payload = {
        "grouped": grouped,
        "component_orders": component_orders,
        "coordinate_hashes": coordinate_hashes,
        "parent_identity": identity,
        "source_pair_sha256": pair_hash_before,
        "source_position_sha256": source_position_sha256,
        "comparator_names": SEALED_ADVERSE_COMPARATOR_NAMES,
        "comparator_recipes": SEALED_ADVERSE_COMPARATOR_RECIPES,
        "evaluated_groups": PREACCELERATION_SPATIAL_GROUP_ORDER,
    }
    return _freeze({
        "complete": True,
        "provenance_valid": True,
        "passed": True,
        "parent_identity": identity,
        "source_pair_sha256": pair_hash_before,
        "source_position_sha256": source_position_sha256,
        "coordinate_hashes": coordinate_hashes,
        "comparator_names": SEALED_ADVERSE_COMPARATOR_NAMES,
        "comparator_recipes": SEALED_ADVERSE_COMPARATOR_RECIPES,
        "evaluated_groups": PREACCELERATION_SPATIAL_GROUP_ORDER,
        "component_orders": component_orders,
        "legacy_Q33_by_mesh": grouped[SEALED_ADVERSE_COMPARATOR_NAMES[0]],
        "legacy_Q55_by_mesh": grouped[SEALED_ADVERSE_COMPARATOR_NAMES[1]],
        "inputs_stable_while_scoring": True,
        "fingerprint": _fingerprint(payload),
        "position_only": True,
        "acceleration_evaluated": False,
        "source_triplets_evaluated": False,
        "artifact_written": False,
        "phase_a_authorized": False,
        "scientific_execution_authorized": False,
    })


def build_protocol125_legacy_holdout_inputs(
    final_pair,
    background,
    *,
    parent_identity,
    v_meshes=None,
):
    """Evaluate both sealed representations on all frozen V meshes."""
    identity = str(parent_identity)
    if not _valid_sha256(identity):
        raise ValueError("legacy holdout parent identity must be a SHA-256 digest")
    if not isinstance(background, Mapping):
        raise TypeError("legacy holdout background must be an explicit mapping")
    try:
        position_state = final_pair.primary.position
        acceleration_state = final_pair.primary.acceleration
    except AttributeError as error:
        raise TypeError("legacy holdout requires a final Q53 position/acceleration pair") from error
    z_parent = np.asarray(position_state.source_z, dtype=float)
    r_parent = np.asarray(position_state.source_r, dtype=float)
    if not (
        str(getattr(position_state, "state_name", "")) == "position"
        and str(getattr(acceleration_state, "state_name", "")) == "acceleration"
        and np.array_equal(z_parent, np.asarray(acceleration_state.source_z))
        and np.array_equal(r_parent, np.asarray(acceleration_state.source_r))
    ):
        raise ValueError("legacy holdout final states differ in source identity")
    pair_hash_before = _pair_fingerprint(final_pair)
    acceleration = acceleration_state.evaluate_reduced(z_parent, r_parent)
    source_jet = represented_position_jet(
        position_state, z_parent, r_parent, acceleration,
    )
    frozen = frozen_validation_meshes()
    if v_meshes is None:
        v_meshes = {name: frozen[name] for name in V_MESH_NAMES}
    if not isinstance(v_meshes, Mapping) or tuple(v_meshes) != V_MESH_NAMES:
        raise ValueError("legacy holdout requires ordered V0/V1/V2 meshes")

    grouped = {name: {} for name in SEALED_ADVERSE_COMPARATOR_NAMES}
    triplets = {name: {} for name in SEALED_ADVERSE_COMPARATOR_NAMES}
    coordinate_hashes = {}
    for mesh_name in V_MESH_NAMES:
        z = np.asarray(v_meshes[mesh_name]["z"], dtype=float)
        r = np.asarray(v_meshes[mesh_name]["r"], dtype=float)
        expected = frozen[mesh_name]["sha256"]
        if hash_arrays(z, r) != expected or str(v_meshes[mesh_name]["sha256"]) != expected:
            raise ValueError(f"legacy holdout {mesh_name} mesh differs from Protocol 125")
        coordinate_hashes[mesh_name] = expected
        projected = sealed_reduced_q33_q55_adverse_projections(
            source_jet,
            z_parent,
            r_parent,
            z,
            r,
            parent_identity=identity,
        )
        if tuple(projected) != SEALED_ADVERSE_COMPARATOR_NAMES:
            raise RuntimeError("sealed legacy comparator inventory changed")
        for name in SEALED_ADVERSE_COMPARATOR_NAMES:
            member = projected[name]
            grouped[name][mesh_name] = _projection_groups(member)
            source = initial_driver_source_triplet_from_acceleration(
                member, z, r, background,
            )
            triplets[name][mesh_name] = {
                lane: np.asarray(source[lane], dtype=float)
                for lane in SOURCE_TRIPLET_ORDER
            }

    pair_hash_after = _pair_fingerprint(final_pair)
    if pair_hash_after != pair_hash_before:
        raise RuntimeError("final pair changed during sealed legacy holdout evaluation")
    component_orders = _component_orders()
    payload = {
        "grouped": grouped,
        "triplets": triplets,
        "component_orders": component_orders,
        "coordinate_hashes": coordinate_hashes,
        "parent_identity": identity,
        "source_pair_sha256": pair_hash_before,
        "comparator_names": SEALED_ADVERSE_COMPARATOR_NAMES,
        "comparator_recipes": SEALED_ADVERSE_COMPARATOR_RECIPES,
    }
    fingerprint = _fingerprint(payload)
    return _freeze({
        "complete": True,
        "provenance_valid": True,
        "passed": True,
        "parent_identity": identity,
        "source_pair_sha256": pair_hash_before,
        "coordinate_hashes": coordinate_hashes,
        "comparator_names": SEALED_ADVERSE_COMPARATOR_NAMES,
        "comparator_recipes": SEALED_ADVERSE_COMPARATOR_RECIPES,
        "component_orders": component_orders,
        "legacy_Q33_by_mesh": grouped[SEALED_ADVERSE_COMPARATOR_NAMES[0]],
        "legacy_Q55_by_mesh": grouped[SEALED_ADVERSE_COMPARATOR_NAMES[1]],
        "legacy_Q33_source_triplets_by_mesh": triplets[
            SEALED_ADVERSE_COMPARATOR_NAMES[0]
        ],
        "legacy_Q55_source_triplets_by_mesh": triplets[
            SEALED_ADVERSE_COMPARATOR_NAMES[1]
        ],
        "inputs_stable_while_scoring": True,
        "fingerprint": fingerprint,
        "artifact_written": False,
        "scientific_execution_authorized": False,
    })
