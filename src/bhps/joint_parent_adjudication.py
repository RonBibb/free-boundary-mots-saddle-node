"""Pure, result-independent representation and classification gates.

The helpers in this module do not construct a parent, repair a field, write an
artifact, or authorize Protocol 125.  They turn already constructed analytic
states and explicit gate records into the frozen representation scores,
sampling-order decision, and fail-closed classification required before a
scientific runner can exist.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
from scipy.optimize import brentq

from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.joint_parent_bulk_reference import FiniteWallReferenceHermitePair
from bhps.joint_parent_position_pair import PositionOnlyConstrainedHermitePair
from bhps.joint_parent_refinement_diagnostics import frozen_validation_meshes
from bhps.joint_parent_representation import COORDINATE_COMPONENT_ORDER
from bhps.matched_staged_continuum import hash_arrays


FLOOR = 1e-12
MASK_WIDTH = 7
SPATIAL_LANES = MappingProxyType({
    "position": ((0, 0),),
    "first_spatial": ((1, 0), (0, 1)),
    "second_spatial": ((2, 0), (1, 1), (0, 2)),
})
SPATIAL_DIRECTION_ORDER = MappingProxyType({
    "position": ("value",),
    "first_spatial": ("z", "r"),
    "second_spatial": ("zz", "zr", "rr"),
})
SOURCE_COMPONENT_ORDER = ("H_z", "H_0", "H_r/r")
COMPARISON_THRESHOLDS = MappingProxyType({
    "Q53_Q33": MappingProxyType({
        "position": 1e-8,
        "first_spatial": 1e-7,
        "second_spatial": 1e-5,
        "acceleration": 1e-5,
        "source": 1e-7,
    }),
    "legacy_Q33_Q55": MappingProxyType({
        "position": 1e-8,
        "first_spatial": 1e-7,
        "second_spatial": 1e-5,
        "acceleration": 1e-5,
        "source": 1e-7,
    }),
    "N0_N1": MappingProxyType({
        "position": 1e-4,
        "first_spatial": 5e-4,
        "second_spatial": 2e-3,
        "acceleration": 2e-3,
    }),
})
V_MESH_NAMES = ("V0", "V1", "V2")
COMPACT_INTERVAL_COUNTS = (104, 128, 152)

PARENT_GATE_GROUPS = MappingProxyType({
    "pre_acceleration_construction": "bulk",
    "native_position_tangent": "position",
    "position_representation": "bulk",
    "dense_boundary_audit": "position",
    "signature_union": "bulk",
    "legacy_holdout": "bulk",
    "sampling_order": "bulk",
    "bulk_prerequisite": "bulk",
    "acceleration_closure": "acceleration",
    "wall_algebra": "acceleration",
    "final_representation": "acceleration",
    "endpoint_derivatives": "acceleration",
    "correction_size": "acceleration",
})
TWO_PARENT_GATE_GROUPS = MappingProxyType({
    "N0_N1_representation": "bulk",
    "correction_refinement": "acceleration",
})


def _immutable(value, dtype=float):
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _bitwise_equal(left, right):
    left = np.ascontiguousarray(np.asarray(left))
    right = np.ascontiguousarray(np.asarray(right))
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes() == right.tobytes()
    )


def _archive_equal(left, right):
    return set(left) == set(right) and all(
        _bitwise_equal(left[name], right[name]) for name in left
    )


def score_construction_reload_provenance(record, reference_pair, position_pair):
    """Replay both immutable pairs and gate the independent construction record."""
    if not isinstance(record, Mapping):
        raise TypeError("construction provenance record must be a mapping")
    required = (
        "finite_wall_maximum_residual",
        "joint_hybrid_maximum_residual",
        "input_fingerprint_before",
        "input_fingerprint_after",
        "physical_normalization_identifier",
        "branch_identifier",
        "expected_parent_label",
        "actual_parent_label",
    )
    if any(name not in record for name in required):
        raise ValueError("construction provenance record is incomplete")
    if not isinstance(reference_pair, FiniteWallReferenceHermitePair):
        raise TypeError("construction provenance requires the finite-wall reference pair")
    if not isinstance(position_pair, PositionOnlyConstrainedHermitePair):
        raise TypeError("construction provenance requires the position-only pair")
    reference_archive = reference_pair.coefficient_arrays()
    position_archive = position_pair.coefficient_arrays()
    for label, archive in (
        ("reference", reference_archive), ("position", position_archive),
    ):
        for name, value in archive.items():
            array = np.asarray(value)
            if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
                raise ValueError(f"{label} archive entry {name} is nonfinite")
            if array.dtype == object:
                raise ValueError(f"{label} archive entry {name} has object dtype")
    reloaded_reference = FiniteWallReferenceHermitePair.from_arrays(reference_archive)
    primary = position_pair.primary
    reloaded_position = PositionOnlyConstrainedHermitePair.from_arrays(
        position_archive,
        compact_wall_contract=primary.compact_wall_contract,
        outer_open_face_contract=primary.outer_open_face_contract,
    )
    reference_reload_exact = (
        _archive_equal(reference_archive, reloaded_reference.coefficient_arrays())
        and reference_pair.fingerprint() == reloaded_reference.fingerprint()
    )
    position_reload_exact = (
        _archive_equal(position_archive, reloaded_position.coefficient_arrays())
        and position_pair.fingerprint() == reloaded_position.fingerprint()
    )
    finite_wall = float(record["finite_wall_maximum_residual"])
    joint = float(record["joint_hybrid_maximum_residual"])
    input_before = str(record["input_fingerprint_before"])
    input_after = str(record["input_fingerprint_after"])
    gates = {
        "finite_wall_residual": bool(np.isfinite(finite_wall) and finite_wall < 1e-9),
        "joint_hybrid_residual": bool(np.isfinite(joint) and joint < 1e-10),
        "immutable_inputs": bool(input_before and input_before == input_after),
        "physical_normalization_identifier": bool(
            str(record["physical_normalization_identifier"])
        ),
        "branch_identifier": bool(str(record["branch_identifier"])),
        "parent_label": bool(
            str(record["expected_parent_label"]) == str(record["actual_parent_label"])
        ),
        "reference_reload_bitwise": reference_reload_exact,
        "position_reload_bitwise": position_reload_exact,
    }
    return MappingProxyType({
        "complete": True,
        "provenance_valid": bool(reference_reload_exact and position_reload_exact),
        "passed": bool(all(gates.values())),
        "gates": gates,
        "finite_wall_maximum_residual": finite_wall,
        "joint_hybrid_maximum_residual": joint,
        "reference_fingerprint": reference_pair.fingerprint(),
        "position_fingerprint": position_pair.fingerprint(),
    })


def _scaled_error(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or not (
        np.all(np.isfinite(left)) and np.all(np.isfinite(right))
    ):
        raise ValueError("comparison arrays must be finite and shape matched")
    denominator = np.maximum.reduce((np.ones_like(left), np.abs(left), np.abs(right)))
    return np.abs(left-right)/denominator


def _score_error(error, component_order):
    error = np.asarray(error, dtype=float)
    if error.size == 0 or not np.all(np.isfinite(error)):
        raise ValueError("a scored comparison group must be finite and nonempty")
    return MappingProxyType({
        "E_inf": float(np.max(error)),
        "E_RMS": float(np.sqrt(np.mean(error**2))),
        "sample_count": int(error.size),
        "component_order": tuple(component_order),
        "elementwise_error": _immutable(error),
        "algebra": "abs(x-y)/max(1,abs(x),abs(y)); flatten then score",
    })


def _constituent_norms(error):
    error = np.asarray(error, dtype=float)
    if error.size == 0 or not np.all(np.isfinite(error)):
        raise ValueError("constituent error must be finite and nonempty")
    return MappingProxyType({
        "E_inf": float(np.max(error)),
        "E_RMS": float(np.sqrt(np.mean(error**2))),
        "sample_count": int(error.size),
    })


def _field_direction_reports(error, field_order, direction_order):
    """Retain every field/direction norm without changing the pooled gate."""
    error = np.asarray(error, dtype=float)
    field_order = tuple(field_order)
    direction_order = tuple(direction_order)
    if error.ndim < 2 or error.shape[:2] != (
        len(field_order), len(direction_order),
    ):
        raise ValueError("field/direction error axes differ from their frozen order")
    reports = {}
    order = []
    for field_index, field in enumerate(field_order):
        for direction_index, direction in enumerate(direction_order):
            name = f"{field}:{direction}"
            order.append(name)
            reports[name] = _constituent_norms(
                error[field_index, direction_index],
            )
    return MappingProxyType({
        "field_order": field_order,
        "direction_order": direction_order,
        "constituent_order": tuple(order),
        "records": MappingProxyType(reports),
    })


def score_group_arrays(left, right, *, component_order=COORDINATE_COMPONENT_ORDER):
    """Score one already ordered Protocol-125 comparison group."""
    error = _scaled_error(left, right)
    return _score_error(error, component_order)


def evaluate_coordinate_jet_groups(state, z, r):
    """Evaluate the physical position/first/second groups in frozen order."""
    evaluator = getattr(state, "evaluate_coordinate_components", None)
    if evaluator is None or not callable(evaluator):
        raise TypeError("representation state lacks coordinate-component evaluation")
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    if (
        z.ndim != 1 or r.ndim != 1 or len(z) < 2 or len(r) < 2
        or np.any(np.diff(z) <= 0.0) or np.any(np.diff(r) <= 0.0)
    ):
        raise ValueError("representation coordinates must be increasing vectors")
    expected = (len(z), len(r), len(COORDINATE_COMPONENT_ORDER))
    result = {}
    for group, orders in SPATIAL_LANES.items():
        lanes = []
        for z_order, r_order in orders:
            values = np.asarray(
                evaluator(z, r, z_order=z_order, r_order=r_order), dtype=float,
            )
            if values.shape != expected or not np.all(np.isfinite(values)):
                raise ValueError(f"representation {group} lane is invalid")
            # Component-major, then derivative direction, then point.
            lanes.append(np.moveaxis(values, -1, 0))
        result[group] = _immutable(np.stack(lanes, axis=1))
    return MappingProxyType(result)


def _validate_v_meshes(meshes, *, require_frozen):
    if not isinstance(meshes, Mapping) or tuple(meshes) != V_MESH_NAMES:
        raise ValueError("representation comparison requires ordered V0/V1/V2 meshes")
    expected = frozen_validation_meshes() if require_frozen else None
    result = {}
    for name in V_MESH_NAMES:
        try:
            z = np.asarray(meshes[name]["z"], dtype=float)
            r = np.asarray(meshes[name]["r"], dtype=float)
        except (KeyError, TypeError) as error:
            raise ValueError(f"{name} mesh is incomplete") from error
        if require_frozen and hash_arrays(z, r) != expected[name]["sha256"]:
            raise ValueError(f"{name} mesh differs from the frozen Protocol-125 mesh")
        result[name] = (z, r, hash_arrays(z, r))
    return result


def score_state_pair_on_v_meshes(
    left_state,
    right_state,
    meshes,
    *,
    comparison_kind,
    groups=("position", "first_spatial", "second_spatial"),
    require_frozen=True,
):
    """Score one Q53/Q33, legacy holdout, or N0/N1 state comparison."""
    if comparison_kind not in COMPARISON_THRESHOLDS:
        raise ValueError("unknown Protocol-125 representation comparison")
    groups = tuple(groups)
    if not groups or any(group not in SPATIAL_LANES for group in groups):
        raise ValueError("state comparison requested an invalid spatial group")
    coordinates = _validate_v_meshes(meshes, require_frozen=require_frozen)
    thresholds = COMPARISON_THRESHOLDS[comparison_kind]
    records = {}
    gates = {}
    for mesh_name, (z, r, digest) in coordinates.items():
        left = evaluate_coordinate_jet_groups(left_state, z, r)
        right = evaluate_coordinate_jet_groups(right_state, z, r)
        records[mesh_name] = {"coordinate_sha256": digest, "groups": {}}
        for group in groups:
            score = score_group_arrays(left[group], right[group])
            constituents = _field_direction_reports(
                score["elementwise_error"],
                COORDINATE_COMPONENT_ORDER,
                SPATIAL_DIRECTION_ORDER[group],
            )
            ceiling = float(thresholds[group])
            passed = bool(score["E_inf"] <= ceiling)
            records[mesh_name]["groups"][group] = {
                **dict(score),
                "constituents": constituents,
                "ceiling": ceiling,
                "passed": passed,
            }
            gates[f"{mesh_name}_{group}"] = passed
    return MappingProxyType({
        "comparison_kind": comparison_kind,
        "groups": groups,
        "records": records,
        "gates": gates,
        "passed": bool(all(gates.values())),
    })


def score_acceleration_pair_on_v_meshes(
    left_state, right_state, meshes, *, comparison_kind, require_frozen=True,
):
    """Score physical acceleration values without silently adding derivatives."""
    if comparison_kind not in COMPARISON_THRESHOLDS:
        raise ValueError("unknown Protocol-125 acceleration comparison")
    coordinates = _validate_v_meshes(meshes, require_frozen=require_frozen)
    ceiling = float(COMPARISON_THRESHOLDS[comparison_kind]["acceleration"])
    records = {}
    gates = {}
    for name, (z, r, digest) in coordinates.items():
        left = evaluate_coordinate_jet_groups(left_state, z, r)["position"]
        right = evaluate_coordinate_jet_groups(right_state, z, r)["position"]
        score = score_group_arrays(left, right)
        constituents = _field_direction_reports(
            score["elementwise_error"],
            COORDINATE_COMPONENT_ORDER,
            SPATIAL_DIRECTION_ORDER["position"],
        )
        passed = bool(score["E_inf"] <= ceiling)
        records[name] = {
            "coordinate_sha256": digest,
            **dict(score),
            "constituents": constituents,
            "ceiling": ceiling,
            "passed": passed,
        }
        gates[name] = passed
    return MappingProxyType({
        "comparison_kind": comparison_kind,
        "group": "acceleration",
        "records": records,
        "gates": gates,
        "passed": bool(all(gates.values())),
    })


def _evaluate_q4_q5_position_images(state, z, r):
    evaluator = getattr(state, "evaluate_reduced", None)
    if evaluator is None or not callable(evaluator):
        raise TypeError("q4/q5 image audit requires reduced-state evaluation")
    jets = {}
    for name, orders in {
        "value": (0, 0),
        "z": (1, 0),
        "r": (0, 1),
        "zz": (2, 0),
        "zr": (1, 1),
        "rr": (0, 2),
    }.items():
        value = np.asarray(
            evaluator(z, r, z_order=orders[0], r_order=orders[1]),
            dtype=float,
        )
        if value.shape != (len(z), len(r), 9) or not np.all(np.isfinite(value)):
            raise ValueError(f"q4/q5 reduced {name} jet is invalid")
        jets[name] = value
    radius = np.asarray(r, dtype=float)[None, :]
    q4 = {name: value[:, :, 4] for name, value in jets.items()}
    q5 = {name: value[:, :, 5] for name, value in jets.items()}
    position = np.stack((radius**2*q4["value"], radius*q5["value"]), axis=-1)
    first = np.stack((
        radius**2*q4["z"],
        2.0*radius*q4["value"]+radius**2*q4["r"],
        radius*q5["z"],
        q5["value"]+radius*q5["r"],
    ), axis=-1)
    second = np.stack((
        radius**2*q4["zz"],
        2.0*radius*q4["z"]+radius**2*q4["zr"],
        2.0*q4["value"]+4.0*radius*q4["r"]+radius**2*q4["rr"],
        radius*q5["zz"],
        q5["z"]+radius*q5["zr"],
        2.0*q5["r"]+radius*q5["rr"],
    ), axis=-1)
    raw = np.stack(tuple(q4.values())+tuple(q5.values()), axis=-1)
    return {
        "position": position,
        "first_spatial": first,
        "second_spatial": second,
        "raw_reduced_q4_q5_jets": raw,
    }


def _evaluate_q4_q5_acceleration_images(state, z, r):
    evaluator = getattr(state, "evaluate_reduced", None)
    if evaluator is None or not callable(evaluator):
        raise TypeError("q4/q5 acceleration-image audit requires reduced evaluation")
    value = np.asarray(evaluator(z, r), dtype=float)
    if value.shape != (len(z), len(r), 9) or not np.all(np.isfinite(value)):
        raise ValueError("q4/q5 reduced acceleration is invalid")
    radius = np.asarray(r, dtype=float)[None, :]
    full = np.stack((radius**2*value[:, :, 4], radius*value[:, :, 5]), axis=-1)
    axis = np.stack((2.0*value[:, 0, 4], value[:, 0, 5]), axis=-1)
    raw = value[:, :, 4:6]
    return {"full": full, "axis": axis, "raw_reduced_q4_q5": raw}


def score_q4_q5_derivative_images_on_v_meshes(
    left_state,
    right_state,
    meshes,
    *,
    comparison_kind,
    state_name,
    require_frozen=True,
):
    """Gate the regular q4/q5 physical derivative images on V0/V1/V2."""
    state_name = str(state_name)
    if state_name not in ("position", "acceleration"):
        raise ValueError("q4/q5 image state_name must be position or acceleration")
    if comparison_kind not in COMPARISON_THRESHOLDS:
        raise ValueError("unknown Protocol-125 q4/q5 image comparison")
    for label, state in (("left", left_state), ("right", right_state)):
        found_name = str(getattr(state, "state_name", state_name))
        if found_name != state_name:
            raise ValueError(f"{label} q4/q5 image state is mislabeled")
    coordinates = _validate_v_meshes(meshes, require_frozen=require_frozen)
    records = {}
    gates = {}
    if state_name == "position":
        groups = ("position", "first_spatial", "second_spatial")
        component_orders = {
            "position": ("N", "T"),
            "first_spatial": ("N_z", "N_r", "T_z", "T_r"),
            "second_spatial": (
                "N_zz", "N_zr", "N_rr", "T_zz", "T_zr", "T_rr",
            ),
        }
        for mesh_name, (z, r, digest) in coordinates.items():
            left = _evaluate_q4_q5_position_images(left_state, z, r)
            right = _evaluate_q4_q5_position_images(right_state, z, r)
            mesh_records = {}
            for group in groups:
                score = score_group_arrays(
                    left[group], right[group],
                    component_order=component_orders[group],
                )
                field_order = ("N", "T")
                direction_order = SPATIAL_DIRECTION_ORDER[group]
                constituent_error = np.moveaxis(
                    score["elementwise_error"], -1, 0,
                ).reshape(
                    len(field_order), len(direction_order),
                    *score["elementwise_error"].shape[:-1],
                )
                constituents = _field_direction_reports(
                    constituent_error, field_order, direction_order,
                )
                ceiling = float(COMPARISON_THRESHOLDS[comparison_kind][group])
                passed = bool(score["E_inf"] <= ceiling)
                mesh_records[group] = {
                    **dict(score),
                    "constituents": constituents,
                    "ceiling": ceiling,
                    "passed": passed,
                }
                gates[f"{mesh_name}_{group}"] = passed
            mesh_records["raw_reduced_q4_q5_jets_diagnostic"] = dict(
                score_group_arrays(
                    left["raw_reduced_q4_q5_jets"],
                    right["raw_reduced_q4_q5_jets"],
                    component_order=("q4/q5 raw jets",),
                )
            )
            records[mesh_name] = {
                "coordinate_sha256": digest,
                "groups": mesh_records,
            }
    else:
        groups = ("acceleration",)
        ceiling = float(COMPARISON_THRESHOLDS[comparison_kind]["acceleration"])
        for mesh_name, (z, r, digest) in coordinates.items():
            left = _evaluate_q4_q5_acceleration_images(left_state, z, r)
            right = _evaluate_q4_q5_acceleration_images(right_state, z, r)
            full_elementwise = _scaled_error(left["full"], right["full"])
            axis_elementwise = _scaled_error(left["axis"], right["axis"])
            full_error = full_elementwise.ravel()
            axis_error = axis_elementwise.ravel()
            score = _score_error(
                np.concatenate((full_error, axis_error)),
                ("N_tt", "T_tt", "axis_N_rr_tt", "axis_T_r_tt"),
            )
            passed = bool(score["E_inf"] <= ceiling)
            raw = score_group_arrays(
                left["raw_reduced_q4_q5"], right["raw_reduced_q4_q5"],
                component_order=("a4", "a5"),
            )
            constituent_records = MappingProxyType({
                "N_tt:full": _constituent_norms(
                    full_elementwise[..., 0],
                ),
                "T_tt:full": _constituent_norms(
                    full_elementwise[..., 1],
                ),
                "N_tt:axis_rr": _constituent_norms(
                    axis_elementwise[..., 0],
                ),
                "T_tt:axis_r": _constituent_norms(
                    axis_elementwise[..., 1],
                ),
            })
            records[mesh_name] = {
                "coordinate_sha256": digest,
                "groups": {
                    "acceleration": {
                        **dict(score),
                        "ceiling": ceiling,
                        "full_mesh_sample_count": int(full_error.size),
                        "axis_image_sample_count": int(axis_error.size),
                        "axis_images_concatenated_into_gate": True,
                        "constituent_order": tuple(constituent_records),
                        "constituents": constituent_records,
                        "passed": passed,
                    },
                    "raw_reduced_q4_q5_diagnostic": dict(raw),
                },
            }
            gates[f"{mesh_name}_acceleration"] = passed
    return MappingProxyType({
        "comparison_kind": comparison_kind,
        "state_name": state_name,
        "groups": groups,
        "records": records,
        "gates": gates,
        "product_rule_images_used": True,
        "axis_images_explicit": True,
        "passed": bool(all(gates.values())),
    })


def score_precomputed_groups_on_v_meshes(
    left_by_mesh,
    right_by_mesh,
    meshes,
    *,
    comparison_kind,
    required_groups,
    component_orders=None,
    require_frozen=True,
):
    """Score explicit arrays, including the sealed legacy Q33/Q55 grouping.

    This is the adapter boundary for representations that intentionally retain
    their Protocol-120 component grouping and therefore do not expose the new
    native-metric coordinate evaluator.
    """
    if comparison_kind not in COMPARISON_THRESHOLDS:
        raise ValueError("unknown Protocol-125 precomputed comparison")
    groups = tuple(required_groups)
    thresholds = COMPARISON_THRESHOLDS[comparison_kind]
    if not groups or any(group not in thresholds for group in groups):
        raise ValueError("precomputed comparison requested an unscored group")
    coordinates = _validate_v_meshes(meshes, require_frozen=require_frozen)
    if set(left_by_mesh) != set(V_MESH_NAMES) or set(right_by_mesh) != set(V_MESH_NAMES):
        raise ValueError("precomputed comparison requires all and only V0/V1/V2")
    component_orders = {} if component_orders is None else dict(component_orders)
    records = {}
    gates = {}
    for mesh_name, (_, _, digest) in coordinates.items():
        if not isinstance(left_by_mesh[mesh_name], Mapping) or not isinstance(
            right_by_mesh[mesh_name], Mapping,
        ):
            raise ValueError(f"{mesh_name} precomputed groups must be mappings")
        records[mesh_name] = {"coordinate_sha256": digest, "groups": {}}
        for group in groups:
            if group not in left_by_mesh[mesh_name] or group not in right_by_mesh[mesh_name]:
                raise ValueError(f"{mesh_name} is missing precomputed group {group}")
            order = component_orders.get(group, (f"{group}-sealed-order",))
            score = score_group_arrays(
                left_by_mesh[mesh_name][group],
                right_by_mesh[mesh_name][group],
                component_order=order,
            )
            ceiling = float(thresholds[group])
            passed = bool(score["E_inf"] <= ceiling)
            records[mesh_name]["groups"][group] = {
                **dict(score), "ceiling": ceiling, "passed": passed,
            }
            gates[f"{mesh_name}_{group}"] = passed
    return MappingProxyType({
        "comparison_kind": comparison_kind,
        "groups": groups,
        "records": records,
        "gates": gates,
        "passed": bool(all(gates.values())),
    })


def score_source_triplet_arrays(left, right, *, comparison_kind):
    """Score ordered (H,H_t,H_tt), each with three regular components."""
    if comparison_kind not in ("Q53_Q33", "legacy_Q33_Q55"):
        raise ValueError("source-triplet comparison has no N0/N1 ceiling")
    ordered = ("source", "source_time", "source_second_time")
    arrays = []
    for name in ordered:
        if name not in left or name not in right:
            raise ValueError(f"source comparison is missing {name}")
        left_value = np.asarray(left[name], dtype=float)
        right_value = np.asarray(right[name], dtype=float)
        if left_value.shape[-1:] != (3,) or right_value.shape != left_value.shape:
            raise ValueError("source triplets require matching three-component arrays")
        arrays.append((_scaled_error(left_value, right_value), name))
    # Triplet-major, then component, exactly as frozen by the protocol.
    combined = np.stack(
        [np.moveaxis(value, -1, 0) for value, _ in arrays], axis=0,
    )
    score = _score_error(combined, SOURCE_COMPONENT_ORDER)
    constituent_records = {}
    constituent_order = []
    for triplet_index, triplet_name in enumerate(ordered):
        for component_index, component_name in enumerate(SOURCE_COMPONENT_ORDER):
            name = f"{triplet_name}:{component_name}"
            constituent_order.append(name)
            constituent_records[name] = _constituent_norms(
                combined[triplet_index, component_index],
            )
    ceiling = float(COMPARISON_THRESHOLDS[comparison_kind]["source"])
    return MappingProxyType({
        **dict(score),
        "triplet_order": ordered,
        "source_component_order": SOURCE_COMPONENT_ORDER,
        "constituent_order": tuple(constituent_order),
        "constituents": MappingProxyType(constituent_records),
        "regular_component_count": 3,
        "ceiling": ceiling,
        "passed": bool(score["E_inf"] <= ceiling),
    })


def sampling_order_from_errors(errors, *, floor=FLOOR):
    """Apply the frozen nonuniform three-mesh order rule to one RMS sequence."""
    values = tuple(float(value) for value in errors)
    if len(values) != 3 or any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("sampling order requires three finite nonnegative errors")
    e0, e1, e2 = values
    monotone = bool(e0 >= e1 >= e2)
    d01 = e0-e1
    d12 = e1-e2
    floor_resolved = bool(abs(d01) <= floor and abs(d12) <= floor)
    record = {
        "errors": values,
        "differences": (d01, d12),
        "monotone": monotone,
        "floor": float(floor),
        "floor_resolved": floor_resolved,
        "search_interval": (1e-7, 20.0),
        "compact_interval_counts": COMPACT_INTERVAL_COUNTS,
    }
    if floor_resolved:
        return MappingProxyType({
            **record,
            "classification": "sampling-floor-resolved",
            "order": None,
            "root_exists": False,
            "passed": monotone,
        })
    if not monotone or d01 <= 0.0 or d12 <= 0.0:
        return MappingProxyType({
            **record,
            "classification": "sampling-order-failed",
            "order": None,
            "root_exists": False,
            "passed": False,
        })
    h0, h1, h2 = (1.0/count for count in COMPACT_INTERVAL_COUNTS)
    ratio = d01/d12

    def equation(order):
        numerator = h0**order-h1**order
        denominator = h1**order-h2**order
        return numerator/denominator-ratio

    lower, upper = record["search_interval"]
    f_lower = equation(lower)
    f_upper = equation(upper)
    root = None
    if np.isfinite(f_lower) and np.isfinite(f_upper) and f_lower*f_upper <= 0.0:
        try:
            root = float(brentq(equation, lower, upper, xtol=1e-13, rtol=1e-13))
        except ValueError:
            root = None
    passed = bool(root is not None and np.isfinite(root) and root >= 3.0)
    return MappingProxyType({
        **record,
        "classification": "sampling-order-pass" if passed else "sampling-order-failed",
        "order": root,
        "root_exists": root is not None,
        "passed": passed,
    })


def _apply_z(matrix, values):
    shape = values.shape
    return np.asarray(matrix @ values.reshape(shape[0], -1)).reshape(shape)


def _apply_r(matrix, values):
    return np.einsum("ij,zjc->zic", matrix.toarray(), values, optimize=True)


def score_sampling_order(state, *, meshes=None):
    """Compare one unchanged analytic Q53 position state with FD7 on V0/V1/V2."""
    if meshes is None:
        frozen = frozen_validation_meshes()
        meshes = {name: frozen[name] for name in V_MESH_NAMES}
    coordinates = _validate_v_meshes(meshes, require_frozen=True)
    records = {}
    first_sequence = []
    second_sequence = []
    mask_label = "[7:-7,7:-7]"
    for name, (z, r, digest) in coordinates.items():
        groups = evaluate_coordinate_jet_groups(state, z, r)
        values = np.moveaxis(groups["position"][:, 0], 0, -1)
        analytic_first = np.moveaxis(groups["first_spatial"], (0, 1), (-1, -2))
        analytic_second = np.moveaxis(groups["second_spatial"], (0, 1), (-1, -2))
        dz = derivative_matrix(z, 1, 7)
        dr = derivative_matrix(r, 1, 7)
        dzz = derivative_matrix(z, 2, 7)
        drr = derivative_matrix(r, 2, 7)
        fd_z = _apply_z(dz, values)
        fd_r = _apply_r(dr, values)
        fd_zz = _apply_z(dzz, values)
        fd_rr = _apply_r(drr, values)
        fd_zr = _apply_z(dz, fd_r)
        fd_first = np.stack((fd_z, fd_r), axis=-2)
        fd_second = np.stack((fd_zz, fd_zr, fd_rr), axis=-2)
        mask = (slice(MASK_WIDTH, -MASK_WIDTH), slice(MASK_WIDTH, -MASK_WIDTH))
        first_error = _scaled_error(analytic_first[mask], fd_first[mask])
        second_error = _scaled_error(analytic_second[mask], fd_second[mask])
        first_rms = float(np.sqrt(np.mean(first_error**2)))
        second_rms = float(np.sqrt(np.mean(second_error**2)))
        first_constituents = _field_direction_reports(
            np.moveaxis(first_error, (-1, -2), (0, 1)),
            COORDINATE_COMPONENT_ORDER,
            SPATIAL_DIRECTION_ORDER["first_spatial"],
        )
        second_constituents = _field_direction_reports(
            np.moveaxis(second_error, (-1, -2), (0, 1)),
            COORDINATE_COMPONENT_ORDER,
            SPATIAL_DIRECTION_ORDER["second_spatial"],
        )
        first_sequence.append(first_rms)
        second_sequence.append(second_rms)
        records[name] = {
            "coordinate_sha256": digest,
            "retained_mask": mask_label,
            "first_spatial_RMS": first_rms,
            "first_spatial_Linf": float(np.max(first_error)),
            "first_spatial_sample_count": int(first_error.size),
            "second_spatial_RMS": second_rms,
            "second_spatial_Linf": float(np.max(second_error)),
            "second_spatial_sample_count": int(second_error.size),
            "first_spatial_constituents": first_constituents,
            "second_spatial_constituents": second_constituents,
            "per_component_first_RMS": tuple(
                float(np.sqrt(np.mean(first_error[..., component]**2)))
                for component in range(len(COORDINATE_COMPONENT_ORDER))
            ),
            "per_component_second_RMS": tuple(
                float(np.sqrt(np.mean(second_error[..., component]**2)))
                for component in range(len(COORDINATE_COMPONENT_ORDER))
            ),
        }
    first_order = sampling_order_from_errors(first_sequence)
    second_order = sampling_order_from_errors(second_sequence)
    gates = {
        "first_spatial_monotonic_order": bool(first_order["passed"]),
        "second_spatial_monotonic_order": bool(second_order["passed"]),
        "V2_second_spatial_RMS": bool(second_sequence[-1] <= 1e-5),
    }
    return MappingProxyType({
        "records": records,
        "first_spatial_sequence": tuple(first_sequence),
        "second_spatial_sequence": tuple(second_sequence),
        "first_spatial_order": first_order,
        "second_spatial_order": second_order,
        "gates": gates,
        "passed": bool(all(gates.values())),
    })


def _gate_state(record, label):
    if not isinstance(record, Mapping):
        return None, f"{label}:missing-record"
    required = ("complete", "provenance_valid", "passed")
    if any(name not in record for name in required):
        return None, f"{label}:incomplete-record"
    if not bool(record["complete"]):
        return None, f"{label}:scorer-incomplete"
    if not bool(record["provenance_valid"]):
        return None, f"{label}:invalid-provenance"
    return bool(record["passed"]), None


def _validate_freeze_record(record):
    if not isinstance(record, Mapping):
        return False, "missing-protocol-freeze-record"
    flags = (
        "frozen_before_parent_data",
        "independent_review_passed",
        "scientific_candidates_absent_at_freeze",
    )
    if str(record.get("status", "")) != "FROZEN" or not all(
        bool(record.get(name, False)) for name in flags
    ):
        return False, "protocol-is-not-prospectively-frozen"
    for name in ("protocol_sha256", "adjudicator_sha256"):
        digest = str(record.get(name, ""))
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            return False, f"invalid-{name}"
    return True, None


def classify_protocol125_gate_records(
    parent_records, two_parent_record, protocol_freeze_record,
):
    """Apply one exhaustive, fail-closed Protocol-125 master classification."""
    freeze_valid, freeze_reason = _validate_freeze_record(protocol_freeze_record)
    if not freeze_valid:
        return MappingProxyType({
            "classification": "INVALID-audit",
            "invalid_reasons": (freeze_reason,),
            "failed_position_groups": (),
            "failed_bulk_groups": (),
            "failed_acceleration_groups": (),
            "phase_a_authorized": False,
            "rhs_rk_phase_b_full_matrix_authorized": False,
            "interface_physics_authorized": False,
        })
    if not isinstance(parent_records, Mapping) or set(parent_records) != {"N0", "N1"}:
        return MappingProxyType({
            "classification": "INVALID-audit",
            "invalid_reasons": ("parent-record-set-must-be-exactly-N0-N1",),
            "failed_position_groups": (),
            "failed_bulk_groups": (),
            "failed_acceleration_groups": (),
            "phase_a_authorized": False,
            "rhs_rk_phase_b_full_matrix_authorized": False,
            "interface_physics_authorized": False,
        })
    invalid = []
    failures = {"position": [], "bulk": [], "acceleration": []}
    for parent in ("N0", "N1"):
        records = parent_records[parent]
        if not isinstance(records, Mapping):
            invalid.append(f"{parent}:missing-parent-mapping")
            continue
        for name, failure_class in PARENT_GATE_GROUPS.items():
            state, reason = _gate_state(records.get(name), f"{parent}:{name}")
            if reason is not None:
                invalid.append(reason)
            elif not state:
                failures[failure_class].append(f"{parent}:{name}")
    if not isinstance(two_parent_record, Mapping):
        invalid.append("two-parent:missing-record-mapping")
    else:
        for name, failure_class in TWO_PARENT_GATE_GROUPS.items():
            state, reason = _gate_state(
                two_parent_record.get(name), f"two-parent:{name}",
            )
            if reason is not None:
                invalid.append(reason)
            elif not state:
                failures[failure_class].append(f"two-parent:{name}")
    if invalid:
        classification = "INVALID-audit"
    elif failures["position"]:
        classification = "FAIL-parent-position"
    elif failures["bulk"]:
        classification = "FAIL-parent-bulk"
    elif failures["acceleration"]:
        classification = "FAIL-acceleration"
    else:
        classification = "PASS-native-joint-parent"
    return MappingProxyType({
        "classification": classification,
        "invalid_reasons": tuple(invalid),
        "failed_position_groups": tuple(failures["position"]),
        "failed_bulk_groups": tuple(failures["bulk"]),
        "failed_acceleration_groups": tuple(failures["acceleration"]),
        "phase_a_authorized": classification == "PASS-native-joint-parent",
        "rhs_rk_phase_b_full_matrix_authorized": False,
        "interface_physics_authorized": False,
    })
