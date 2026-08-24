"""Single-owner acceleration/source fixed point for Protocol 125."""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.joint_parent_selective_algebra import (
    SelectiveWallAlgebraicGateError,
)
from bhps.joint_parent_source_closure import (
    initial_driver_source_triplet_from_acceleration,
)
from bhps.junction_second_preservation_diagnostic import (
    wall_junction_second_tangent,
)
from bhps.matched_staged_continuum import ProjectedJetField
from bhps.nonlinear_regular_so3_evolution import (
    CompactWallCoupledAlgebraicGateError,
    compact_wall_normal_gauge_acceleration_residuals,
    impose_compact_wall_normal_tangential_acceleration,
    solve_compact_wall_coupled_phi_normal_acceleration,
    solve_compact_wall_tangential_chi_acceleration,
)


ACCELERATION_FAILURE_PROTOCOL_IDENTIFIER = (
    "Protocol-125-scientific-acceleration-failure-v1"
)
ACCELERATION_FIXED_POINT_METHOD = (
    "Protocol-125-full-update-eight-map-fixed-point"
)
ACCELERATION_FAILURE_GROUPS = (
    "acceleration_closure",
    "wall_algebra",
)
ACCELERATION_FAILURE_REASONS = (
    "fixed_point_nonconvergence",
    "final_normal_gauge_closure_failure",
    "coupled_wall_algebraic_gate_failure",
    "selective_wall_algebraic_gate_failure",
    "selective_wall_normalized_residual_gate_failure",
)
_INPUT_FINGERPRINT_KEYS = (
    "position_state_sha256",
    "position_sha256",
    "bulk_acceleration_sha256",
    "z_sha256",
    "r_sha256",
    "coordinate_pair_sha256",
    "background_sha256",
)
_FAILURE_RECORD_KEYS = (
    "protocol_identifier",
    "method",
    "parent_label",
    "parent_identity",
    "attempt_fingerprint",
    "classification",
    "complete",
    "provenance_valid",
    "passed",
    "failure_group",
    "failure_reason",
    "fixed_settings",
    "input_provenance",
    "history",
    "maps_completed",
    "consecutive_converged_maps",
    "last_iterate_evidence",
    "failure_event",
    "final_normal_gauge",
    "final_wall_second_tangent",
    "outer_overwrite_applied",
    "generic_axis_fill_applied",
    "endpoint_history_carried",
    "acceleration_returned",
    "retry_authorized",
    "candidate_or_phase_a_executed",
    "fingerprint",
)
_LAST_ITERATE_KEYS = (
    "acceleration",
    "acceleration_sha256",
    "source_triplet",
    "source_triplet_sha256",
    "failed_stage_input",
    "failed_stage_input_sha256",
    "coupled",
    "selective",
    "axis_reconciliation",
)
_FAILURE_EVENT_KEYS = (
    "owner",
    "failed_map",
    "exception_type",
    "message",
    "gate",
    "radial_index",
    "radius",
    "field",
    "diagnostics",
)
_HISTORY_KEYS = (
    "map",
    "acceleration_scaled_Linf_change",
    "source_triplet_scaled_Linf_change",
    "consecutive_converged_maps",
    "coupled",
    "selective",
    "normal_tangential_correction_scaled_Linf",
    "axis_reconciliation_scaled_Linf",
    "axis_reconciliation",
)
_SELECTIVE_RESIDUAL_PATTERN = re.compile(
    r"^normalized tangential/chi wall residual gate failed: "
    r"residual=([^,]+), limit=(.+)$"
)


class Protocol125AccelerationScientificFailure(RuntimeError):
    """A measured acceleration failure with sealed scientific evidence."""

    def __init__(self, record):
        self.record = validate_protocol125_acceleration_failure_record(record)
        self.failure_record = self.record
        super().__init__(
            f"{self.record['parent_label']} Protocol-125 "
            f"{self.record['failure_group']} failed scientifically: "
            f"{self.record['failure_reason']}"
        )


def _valid_sha256(value):
    value = str(value)
    return bool(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _digest_bytes(digest, tag, payload):
    tag = str(tag).encode("utf-8")
    payload = bytes(payload)
    digest.update(len(tag).to_bytes(8, "little"))
    digest.update(tag)
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def _update_tree_digest(digest, value, path):
    if isinstance(value, Mapping):
        if any(not isinstance(name, str) or not name for name in value):
            raise ValueError(f"hashed mapping {path} contains an invalid key")
        _digest_bytes(digest, f"mapping:{path}", str(len(value)).encode("ascii"))
        for name in sorted(value):
            _digest_bytes(digest, f"key:{path}", name.encode("utf-8"))
            _update_tree_digest(digest, value[name], f"{path}/{name}")
        return
    if isinstance(value, (tuple, list)):
        _digest_bytes(digest, f"sequence:{path}", str(len(value)).encode("ascii"))
        for index, item in enumerate(value):
            _update_tree_digest(digest, item, f"{path}/{index}")
        return
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype == object:
            raise ValueError(f"hashed array {path} has object dtype")
        if array.dtype.kind in "fc" and np.any(np.isnan(array)):
            raise ValueError(f"hashed array {path} contains NaN")
        _digest_bytes(digest, f"array-shape:{path}", repr(array.shape).encode("ascii"))
        _digest_bytes(digest, f"array-dtype:{path}", array.dtype.str.encode("ascii"))
        _digest_bytes(digest, f"array-data:{path}", array.tobytes())
        return
    if value is None:
        _digest_bytes(digest, f"none:{path}", b"")
        return
    if isinstance(value, bool):
        _digest_bytes(digest, f"bool:{path}", b"1" if value else b"0")
        return
    if isinstance(value, int):
        _digest_bytes(digest, f"int:{path}", str(value).encode("ascii"))
        return
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError(f"hashed scalar {path} is NaN")
        _digest_bytes(digest, f"float:{path}", struct.pack("!d", value))
        return
    if isinstance(value, str):
        _digest_bytes(digest, f"str:{path}", value.encode("utf-8"))
        return
    if isinstance(value, bytes):
        _digest_bytes(digest, f"bytes:{path}", value)
        return
    raise TypeError(f"unsupported hashed evidence at {path}: {type(value).__name__}")


def _fingerprint_tree(value, *, root):
    digest = hashlib.sha256()
    _update_tree_digest(digest, value, str(root))
    return digest.hexdigest()


def _immutable_array(value):
    array = np.ascontiguousarray(np.asarray(value))
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _freeze_tree(value):
    if isinstance(value, Mapping):
        if any(not isinstance(name, str) or not name for name in value):
            raise ValueError("acceleration failure mappings require nonempty string keys")
        return MappingProxyType({name: _freeze_tree(item) for name, item in value.items()})
    if isinstance(value, np.ndarray):
        return _immutable_array(value)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_tree(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _exact_keys(record, expected, label):
    if not isinstance(record, Mapping) or tuple(record) != tuple(expected):
        raise ValueError(f"{label} schema differs")


def _strict_bool(value, label):
    if type(value) is not bool:
        raise TypeError(f"{label} must be a bool")
    return value


def _finite_number(value, label, *, nonnegative=False):
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{label} must be numeric")
    found = float(value)
    if not np.isfinite(found) or (nonnegative and found < 0.0):
        raise ValueError(f"{label} must be finite")
    return found


def _input_provenance(
    position_state,
    q,
    a_bulk,
    z,
    r,
    background,
    *,
    parent_label,
    parent_identity,
):
    label = str(parent_label)
    identity = str(parent_identity)
    if label not in ("N0", "N1"):
        raise ValueError("Protocol-125 acceleration parent label must be N0 or N1")
    if not _valid_sha256(identity):
        raise ValueError(
            "Protocol-125 acceleration parent identity must be a SHA-256 digest"
        )
    if not isinstance(background, Mapping):
        raise TypeError("Protocol-125 acceleration background must be a mapping")
    fingerprint = getattr(position_state, "fingerprint", None)
    if not callable(fingerprint):
        raise TypeError("position representation lacks a provenance fingerprint")
    position_state_sha256 = str(fingerprint())
    if not _valid_sha256(position_state_sha256):
        raise ValueError("position representation fingerprint is invalid")
    if not all(np.all(np.isfinite(value)) for value in (q, a_bulk, z, r)):
        raise ValueError("Protocol-125 acceleration inputs must be finite")
    background_snapshot = _freeze_tree(background)
    fingerprints = MappingProxyType({
        "position_state_sha256": position_state_sha256,
        "position_sha256": _fingerprint_tree(q, root="position"),
        "bulk_acceleration_sha256": _fingerprint_tree(
            a_bulk, root="bulk_acceleration",
        ),
        "z_sha256": _fingerprint_tree(z, root="z"),
        "r_sha256": _fingerprint_tree(r, root="r"),
        "coordinate_pair_sha256": _fingerprint_tree((z, r), root="coordinates"),
        "background_sha256": _fingerprint_tree(
            background_snapshot, root="background",
        ),
    })
    fixed = MappingProxyType({
        "maximum_maps": 8,
        "convergence_tolerance": 1e-12,
        "required_consecutive": 2,
        "normal_gauge_strict_ceiling": 1e-10,
        "stencil_width": 7,
        "full_replacement_without_relaxation": True,
    })
    provenance = MappingProxyType({
        "position_shape": tuple(int(value) for value in q.shape),
        "acceleration_shape": tuple(int(value) for value in a_bulk.shape),
        "z": _immutable_array(z),
        "r": _immutable_array(r),
        "background": background_snapshot,
        "input_fingerprints": fingerprints,
    })
    attempt = _fingerprint_tree({
        "parent_label": label,
        "parent_identity": identity,
        "fixed_settings": fixed,
        "input_fingerprints": fingerprints,
    }, root="acceleration_attempt")
    return label, identity, fixed, provenance, attempt


def _validate_normal_gauge_measurement(record):
    _exact_keys(record, ("walls", "maximum"), "final normal-GH audit")
    walls = record["walls"]
    if not isinstance(walls, (tuple, list)) or len(walls) != 2:
        raise ValueError("final normal-GH audit omits a wall")
    maxima = []
    for expected, wall in zip(("lower", "upper"), walls):
        if not isinstance(wall, Mapping) or str(wall.get("wall", "")) != expected:
            raise ValueError("final normal-GH wall evidence is malformed")
        maxima.append(_finite_number(
            wall.get("maximum_normalized"),
            f"{expected} normal-GH maximum",
            nonnegative=True,
        ))
        _finite_number(
            wall.get("maximum_absolute"),
            f"{expected} normal-GH absolute maximum",
            nonnegative=True,
        )
    maximum = _finite_number(
        record["maximum"], "final normal-GH maximum", nonnegative=True,
    )
    if maximum != max(maxima):
        raise ValueError("final normal-GH maximum differs from its walls")
    _fingerprint_tree(record, root="final_normal_gauge")
    return maximum


def _validate_wall_tangent_measurement(record):
    if not isinstance(record, Mapping) or tuple(record) != ("lower", "upper"):
        raise ValueError("final second-junction wall evidence is malformed")
    for expected in ("lower", "upper"):
        wall = record[expected]
        if (
            not isinstance(wall, Mapping)
            or str(wall.get("wall", "")) != expected
            or _strict_bool(
                wall.get("finite"), f"{expected} second-junction finite",
            ) is not True
        ):
            raise ValueError("final second-junction wall evidence is malformed")
    _fingerprint_tree(record, root="final_wall_second_tangent")


def _structured_wall_failure_event(error, *, owner, r, failed_map):
    if owner == "coupled":
        expected_type = CompactWallCoupledAlgebraicGateError
        field = None
        allowed_gates = {
            "finite_positive_full_row_norm",
            "rank_condition_pivot",
            "normalized_linear_residual",
        }
    elif owner == "selective":
        expected_type = SelectiveWallAlgebraicGateError
        field = str(getattr(error, "field", ""))
        if not field:
            raise ValueError("selective wall failure lacks a field identity")
        allowed_gates = {
            "finite_positive_full_row_norm",
            "rank_condition_pivot",
            "finite_solution",
            "normalized_linear_residual",
        }
    else:
        raise ValueError("unknown structured wall owner")
    if not isinstance(error, expected_type):
        raise TypeError("structured wall failure has the wrong exception type")
    radial_index = int(getattr(error, "radial_index", -1))
    if radial_index < 0 or radial_index >= len(r):
        raise ValueError("structured wall failure radial index is invalid")
    gate = str(getattr(error, "gate", ""))
    diagnostics = getattr(error, "diagnostics", None)
    if gate not in allowed_gates or not isinstance(diagnostics, Mapping):
        raise ValueError("structured wall failure diagnostics are malformed")
    _fingerprint_tree(diagnostics, root="wall_failure_diagnostics")
    return {
        "owner": owner,
        "failed_map": int(failed_map),
        "exception_type": type(error).__name__,
        "message": str(error),
        "gate": gate,
        "radial_index": radial_index,
        "radius": float(r[radial_index]),
        "field": field,
        "diagnostics": diagnostics,
    }


def _validate_source_triplet_core(record, shape):
    if not isinstance(record, Mapping):
        raise TypeError("source-triplet owner returned a nonmapping record")
    expected = (shape[0], shape[1], 3)
    for name in ("source", "source_time", "source_second_time"):
        if name not in record:
            raise ValueError(f"source-triplet owner omitted {name}")
        value = np.asarray(record[name], dtype=float)
        if value.shape != expected or not np.all(np.isfinite(value)):
            raise ValueError(f"source-triplet owner returned malformed {name}")
    _fingerprint_tree(record, root="source_triplet_owner_result")


def _validate_wall_owner_output(name, value, record, shape):
    value = np.asarray(value, dtype=float)
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} wall owner returned a malformed acceleration")
    if not isinstance(record, Mapping):
        raise TypeError(f"{name} wall owner returned a nonmapping record")
    if _strict_bool(record.get("passed"), f"{name} wall owner passed") is not True:
        raise ValueError(f"{name} wall owner returned a failed record")
    if name == "selective" and not (
        _strict_bool(
            record.get("protected_0_1_6_7_bitwise"),
            "selective protected owner",
        )
        and _strict_bool(
            record.get("q4_q5_axis_bitwise"),
            "selective q4/q5 axis owner",
        )
    ):
        raise ValueError("selective wall owner violated ownership")
    _fingerprint_tree(record, root=f"{name}_wall_owner_result")


def _selective_residual_failure_event(error, *, r, failed_map):
    matched = _SELECTIVE_RESIDUAL_PATTERN.fullmatch(str(error))
    if matched is None:
        return None
    residual = _finite_number(
        matched.group(1), "selective normalized wall residual", nonnegative=True,
    )
    limit = _finite_number(
        matched.group(2), "selective normalized wall residual limit",
        nonnegative=True,
    )
    if limit <= 0.0 or residual < limit:
        raise ValueError("selective normalized residual exception is inconsistent")
    maximum_index = None
    maximum_radius = None
    return {
        "owner": "selective",
        "failed_map": int(failed_map),
        "exception_type": type(error).__name__,
        "message": str(error),
        "gate": "normalized_wall_residual",
        "radial_index": maximum_index,
        "radius": maximum_radius,
        "field": "tangential_or_chi",
        "diagnostics": {
            "maximum_normalized_residual": residual,
            "strict_limit": limit,
            "strict_gate_failed": True,
            "radial_localization_unavailable_from_legacy_exception": True,
            "source_radial_count": int(len(r)),
        },
    }


def _acceleration_failure_record(
    *,
    parent_label,
    parent_identity,
    attempt_fingerprint,
    fixed_settings,
    input_provenance,
    failure_group,
    failure_reason,
    history,
    consecutive_converged_maps,
    current,
    source_triplet,
    failure_event,
    failed_stage_input=None,
    coupled=None,
    selective=None,
    axis_reconciliation=None,
    final_normal_gauge=None,
    final_wall_second_tangent=None,
):
    current = np.asarray(current, dtype=float)
    failed_input = (
        None if failed_stage_input is None
        else np.asarray(failed_stage_input, dtype=float)
    )
    last = {
        "acceleration": current,
        "acceleration_sha256": _fingerprint_tree(
            current, root="last_iterate_acceleration",
        ),
        "source_triplet": source_triplet,
        "source_triplet_sha256": _fingerprint_tree(
            source_triplet, root="last_iterate_source_triplet",
        ),
        "failed_stage_input": failed_input,
        "failed_stage_input_sha256": (
            None if failed_input is None
            else _fingerprint_tree(failed_input, root="failed_stage_input")
        ),
        "coupled": coupled,
        "selective": selective,
        "axis_reconciliation": axis_reconciliation,
    }
    record = {
        "protocol_identifier": ACCELERATION_FAILURE_PROTOCOL_IDENTIFIER,
        "method": ACCELERATION_FIXED_POINT_METHOD,
        "parent_label": str(parent_label),
        "parent_identity": str(parent_identity),
        "attempt_fingerprint": str(attempt_fingerprint),
        "classification": "FAIL-acceleration",
        "complete": True,
        "provenance_valid": True,
        "passed": False,
        "failure_group": str(failure_group),
        "failure_reason": str(failure_reason),
        "fixed_settings": fixed_settings,
        "input_provenance": input_provenance,
        "history": tuple(history),
        "maps_completed": len(history),
        "consecutive_converged_maps": int(consecutive_converged_maps),
        "last_iterate_evidence": last,
        "failure_event": failure_event,
        "final_normal_gauge": final_normal_gauge,
        "final_wall_second_tangent": final_wall_second_tangent,
        "outer_overwrite_applied": False,
        "generic_axis_fill_applied": False,
        "endpoint_history_carried": False,
        "acceleration_returned": False,
        "retry_authorized": False,
        "candidate_or_phase_a_executed": False,
    }
    record["fingerprint"] = _fingerprint_tree(
        record, root="scientific_acceleration_failure",
    )
    return record


def validate_protocol125_acceleration_failure_record(record):
    """Validate and defensively freeze a scientific acceleration failure."""
    _exact_keys(record, _FAILURE_RECORD_KEYS, "acceleration failure record")
    label = str(record["parent_label"])
    identity = str(record["parent_identity"])
    if (
        str(record["protocol_identifier"])
        != ACCELERATION_FAILURE_PROTOCOL_IDENTIFIER
        or str(record["method"]) != ACCELERATION_FIXED_POINT_METHOD
        or label not in ("N0", "N1")
        or not _valid_sha256(identity)
        or not _valid_sha256(record["attempt_fingerprint"])
        or str(record["classification"]) != "FAIL-acceleration"
    ):
        raise ValueError("acceleration failure identity or classification differs")
    required_flags = {
        "complete": True,
        "provenance_valid": True,
        "passed": False,
        "outer_overwrite_applied": False,
        "generic_axis_fill_applied": False,
        "endpoint_history_carried": False,
        "acceleration_returned": False,
        "retry_authorized": False,
        "candidate_or_phase_a_executed": False,
    }
    for name, expected in required_flags.items():
        if _strict_bool(record[name], f"acceleration failure {name}") is not expected:
            raise ValueError(f"acceleration failure flag {name} differs")

    fixed = record["fixed_settings"]
    fixed_keys = (
        "maximum_maps", "convergence_tolerance", "required_consecutive",
        "normal_gauge_strict_ceiling", "stencil_width",
        "full_replacement_without_relaxation",
    )
    _exact_keys(fixed, fixed_keys, "acceleration fixed settings")
    if (
        int(fixed["maximum_maps"]) != 8
        or _finite_number(
            fixed["convergence_tolerance"], "fixed convergence tolerance",
            nonnegative=True,
        ) != 1e-12
        or int(fixed["required_consecutive"]) != 2
        or _finite_number(
            fixed["normal_gauge_strict_ceiling"], "normal-GH ceiling",
            nonnegative=True,
        ) != 1e-10
        or int(fixed["stencil_width"]) != 7
        or _strict_bool(
            fixed["full_replacement_without_relaxation"],
            "full replacement flag",
        ) is not True
    ):
        raise ValueError("acceleration failure fixed settings differ")

    provenance = record["input_provenance"]
    _exact_keys(
        provenance,
        (
            "position_shape", "acceleration_shape", "z", "r",
            "background", "input_fingerprints",
        ),
        "acceleration input provenance",
    )
    position_shape = tuple(int(value) for value in provenance["position_shape"])
    acceleration_shape = tuple(
        int(value) for value in provenance["acceleration_shape"]
    )
    z = np.asarray(provenance["z"], dtype=float)
    r = np.asarray(provenance["r"], dtype=float)
    if (
        z.ndim != 1 or r.ndim != 1 or len(z) < 7 or len(r) < 7
        or position_shape != (len(z), len(r), 9)
        or acceleration_shape != position_shape
        or not all(np.all(np.isfinite(value)) for value in (z, r))
        or np.any(np.diff(z) <= 0.0) or np.any(np.diff(r) <= 0.0)
        or r[0] != 0.0 or np.signbit(r[0])
    ):
        raise ValueError("acceleration failure coordinate provenance differs")
    fingerprints = provenance["input_fingerprints"]
    _exact_keys(
        fingerprints, _INPUT_FINGERPRINT_KEYS,
        "acceleration input fingerprint inventory",
    )
    if any(not _valid_sha256(fingerprints[name]) for name in fingerprints):
        raise ValueError("acceleration input fingerprint is invalid")
    reproduced = {
        "z_sha256": _fingerprint_tree(z, root="z"),
        "r_sha256": _fingerprint_tree(r, root="r"),
        "coordinate_pair_sha256": _fingerprint_tree((z, r), root="coordinates"),
        "background_sha256": _fingerprint_tree(
            provenance["background"], root="background",
        ),
    }
    if any(str(fingerprints[name]) != value for name, value in reproduced.items()):
        raise ValueError("acceleration input provenance hash does not reproduce")
    expected_attempt = _fingerprint_tree({
        "parent_label": label,
        "parent_identity": identity,
        "fixed_settings": fixed,
        "input_fingerprints": fingerprints,
    }, root="acceleration_attempt")
    if str(record["attempt_fingerprint"]) != expected_attempt:
        raise ValueError("acceleration attempt fingerprint differs")

    history = record["history"]
    maps_completed = int(record["maps_completed"])
    if (
        not isinstance(history, (tuple, list))
        or not 0 <= len(history) <= 8
        or maps_completed != len(history)
    ):
        raise ValueError("acceleration failure history length differs")
    consecutive = 0
    tolerance = float(fixed["convergence_tolerance"])
    for map_index, item in enumerate(history, start=1):
        _exact_keys(item, _HISTORY_KEYS, f"acceleration map {map_index}")
        if int(item["map"]) != map_index:
            raise ValueError("acceleration failure map order differs")
        acceleration_change = _finite_number(
            item["acceleration_scaled_Linf_change"],
            "acceleration map change", nonnegative=True,
        )
        source_change = _finite_number(
            item["source_triplet_scaled_Linf_change"],
            "source-triplet map change", nonnegative=True,
        )
        _finite_number(
            item["normal_tangential_correction_scaled_Linf"],
            "normal-tangential map change", nonnegative=True,
        )
        _finite_number(
            item["axis_reconciliation_scaled_Linf"],
            "axis-reconciliation map change", nonnegative=True,
        )
        consecutive = consecutive + 1 if (
            acceleration_change < tolerance and source_change < tolerance
        ) else 0
        if int(item["consecutive_converged_maps"]) != consecutive:
            raise ValueError("acceleration failure convergence history differs")
        _fingerprint_tree(item, root=f"acceleration_map_{map_index}")
    if int(record["consecutive_converged_maps"]) != consecutive:
        raise ValueError("acceleration failure final convergence count differs")

    last = record["last_iterate_evidence"]
    _exact_keys(last, _LAST_ITERATE_KEYS, "last acceleration iterate evidence")
    acceleration = np.asarray(last["acceleration"], dtype=float)
    if (
        acceleration.shape != acceleration_shape
        or not np.all(np.isfinite(acceleration))
        or str(last["acceleration_sha256"]) != _fingerprint_tree(
            acceleration, root="last_iterate_acceleration",
        )
    ):
        raise ValueError("last acceleration iterate evidence differs")
    source_triplet = last["source_triplet"]
    if (
        not isinstance(source_triplet, Mapping)
        or str(last["source_triplet_sha256"]) != _fingerprint_tree(
            source_triplet, root="last_iterate_source_triplet",
        )
    ):
        raise ValueError("last source-triplet evidence differs")
    failed_stage_input = last["failed_stage_input"]
    if failed_stage_input is None:
        if last["failed_stage_input_sha256"] is not None:
            raise ValueError("absent failed-stage input retains a fingerprint")
    else:
        failed_stage_input = np.asarray(failed_stage_input, dtype=float)
        if (
            failed_stage_input.shape != acceleration_shape
            or not np.all(np.isfinite(failed_stage_input))
            or str(last["failed_stage_input_sha256"]) != _fingerprint_tree(
                failed_stage_input, root="failed_stage_input",
            )
        ):
            raise ValueError("failed-stage input evidence differs")

    event = record["failure_event"]
    _exact_keys(event, _FAILURE_EVENT_KEYS, "acceleration failure event")
    owner = str(event["owner"])
    gate = str(event["gate"])
    diagnostics = event["diagnostics"]
    if not gate or not isinstance(diagnostics, Mapping):
        raise ValueError("acceleration failure event diagnostics differ")
    _fingerprint_tree(event, root="acceleration_failure_event")
    reason = str(record["failure_reason"])
    group = str(record["failure_group"])
    if reason not in ACCELERATION_FAILURE_REASONS or group not in (
        ACCELERATION_FAILURE_GROUPS
    ):
        raise ValueError("acceleration failure reason or group differs")

    closure_reasons = {
        "fixed_point_nonconvergence",
        "final_normal_gauge_closure_failure",
    }
    wall_reasons = set(ACCELERATION_FAILURE_REASONS) - closure_reasons
    if (reason in closure_reasons) != (group == "acceleration_closure"):
        raise ValueError("acceleration failure group does not match its reason")
    if (reason in wall_reasons) != (group == "wall_algebra"):
        raise ValueError("wall failure group does not match its reason")

    final_normal = record["final_normal_gauge"]
    final_wall = record["final_wall_second_tangent"]
    if reason == "fixed_point_nonconvergence":
        if not (
            maps_completed == 8 and consecutive < 2
            and owner == "fixed_point" and event["failed_map"] is None
            and gate == "two_consecutive_map_convergence"
            and event["exception_type"] is None
            and failed_stage_input is None
            and final_normal is None and final_wall is None
            and all(isinstance(last[name], Mapping) for name in (
                "coupled", "selective", "axis_reconciliation",
            ))
        ):
            raise ValueError("fixed-point nonconvergence evidence is inconsistent")
        if not (
            int(diagnostics.get("maximum_maps", -1)) == 8
            and int(diagnostics.get("required_consecutive", -1)) == 2
            and int(diagnostics.get("observed_consecutive", -1)) == consecutive
            and _finite_number(
                diagnostics.get("convergence_tolerance"),
                "nonconvergence tolerance", nonnegative=True,
            ) == tolerance
            and _finite_number(
                diagnostics.get("final_acceleration_scaled_Linf_change"),
                "final acceleration change", nonnegative=True,
            ) == float(history[-1]["acceleration_scaled_Linf_change"])
            and _finite_number(
                diagnostics.get("final_source_triplet_scaled_Linf_change"),
                "final source-triplet change", nonnegative=True,
            ) == float(history[-1]["source_triplet_scaled_Linf_change"])
        ):
            raise ValueError("fixed-point nonconvergence diagnostics differ")
    elif reason == "final_normal_gauge_closure_failure":
        maximum = _validate_normal_gauge_measurement(final_normal)
        _validate_wall_tangent_measurement(final_wall)
        if not (
            maps_completed >= 2 and consecutive >= 2
            and owner == "final_normal_gauge" and event["failed_map"] is None
            and gate == "strict_normalized_residual"
            and event["exception_type"] is None
            and failed_stage_input is None
            and all(isinstance(last[name], Mapping) for name in (
                "coupled", "selective", "axis_reconciliation",
            ))
            and maximum >= float(fixed["normal_gauge_strict_ceiling"])
            and _finite_number(
                diagnostics.get("maximum_normalized_residual"),
                "normal-GH failure maximum", nonnegative=True,
            ) == maximum
            and _finite_number(
                diagnostics.get("strict_ceiling"),
                "normal-GH failure ceiling", nonnegative=True,
            ) == float(fixed["normal_gauge_strict_ceiling"])
            and _strict_bool(
                diagnostics.get("strict_gate_failed"),
                "normal-GH strict gate failed",
            ) is True
        ):
            raise ValueError("final normal-GH failure evidence is inconsistent")
    else:
        failed_map = event["failed_map"]
        radial_index = event["radial_index"]
        radius = event["radius"]
        if not (
            isinstance(failed_map, (int, np.integer))
            and int(failed_map) == maps_completed + 1
            and int(failed_map) <= 8
            and final_normal is None and final_wall is None
            and failed_stage_input is not None
            and last["selective"] is None
            and last["axis_reconciliation"] is None
        ):
            raise ValueError("wall-algebra failure stop location differs")
        if reason == "coupled_wall_algebraic_gate_failure":
            if not (
                owner == "coupled"
                and str(event["exception_type"])
                == "CompactWallCoupledAlgebraicGateError"
                and last["coupled"] is None
                and event["field"] is None
            ):
                raise ValueError("coupled wall failure evidence differs")
        else:
            if owner != "selective" or not isinstance(last["coupled"], Mapping):
                raise ValueError("selective wall failure evidence differs")
            if reason == "selective_wall_algebraic_gate_failure" and str(
                event["exception_type"]
            ) != "SelectiveWallAlgebraicGateError":
                raise ValueError("selective algebraic exception type differs")
            if reason == "selective_wall_normalized_residual_gate_failure":
                if not (
                    str(event["exception_type"]) == "RuntimeError"
                    and gate == "normalized_wall_residual"
                    and radial_index is None and radius is None
                    and _finite_number(
                        diagnostics.get("maximum_normalized_residual"),
                        "selective failure residual", nonnegative=True,
                    ) >= _finite_number(
                        diagnostics.get("strict_limit"),
                        "selective failure limit", nonnegative=True,
                    )
                ):
                    raise ValueError("selective residual failure evidence differs")
        if radial_index is not None:
            radial_index = int(radial_index)
            if (
                radial_index < 0 or radial_index >= len(r)
                or _finite_number(radius, "wall failure radius")
                != float(r[radial_index])
            ):
                raise ValueError("wall failure radial provenance differs")

    without_fingerprint = {
        name: record[name] for name in _FAILURE_RECORD_KEYS if name != "fingerprint"
    }
    if str(record["fingerprint"]) != _fingerprint_tree(
        without_fingerprint, root="scientific_acceleration_failure",
    ):
        raise ValueError("acceleration failure fingerprint differs")
    return _freeze_tree(record)


def _scaled_linf(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    denominator = np.maximum.reduce((
        np.ones_like(left), np.abs(left), np.abs(right),
    ))
    return float(np.max(np.abs(left-right)/denominator))


def represented_position_jet(position_state, z, r, acceleration):
    """Bind one acceleration to analytic spatial jets of a fixed position."""
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    acceleration = np.asarray(acceleration, dtype=float)
    q = position_state.evaluate_reduced(z, r)
    expected = (len(z), len(r), 9)
    if q.shape != expected or acceleration.shape != expected:
        raise ValueError("represented position/acceleration shapes differ")
    first = np.zeros((3, *expected))
    second = np.zeros((3, 3, *expected))
    first[1] = position_state.evaluate_reduced(z, r, z_order=1)
    first[2] = position_state.evaluate_reduced(z, r, r_order=1)
    second[0, 0] = acceleration
    second[1, 1] = position_state.evaluate_reduced(z, r, z_order=2)
    second[1, 2] = second[2, 1] = position_state.evaluate_reduced(
        z, r, z_order=1, r_order=1,
    )
    second[2, 2] = position_state.evaluate_reduced(z, r, r_order=2)
    if not all(np.all(np.isfinite(value)) for value in (q, first, second)):
        raise RuntimeError("represented parent jet is nonfinite")
    if np.any(first[0] != 0.0) or np.any(np.signbit(first[0])):
        raise AssertionError("represented parent velocity is not positive zero")
    return ProjectedJetField(z.copy(), r.copy(), q, first, second)


def _canonical_joint_parent_axis_numerators(acceleration, r):
    """Reassemble the two physical parity numerators with an exact +0 axis."""
    acceleration = np.asarray(acceleration, dtype=float)
    r = np.asarray(r, dtype=float)
    if r.ndim != 1:
        raise ValueError("joint parent radial coordinate must be one-dimensional")
    if acceleration.ndim != 3 or acceleration.shape[1:] != (len(r), 9):
        raise ValueError("joint parent acceleration must have shape (z,r,9)")
    anisotropy = np.zeros(acceleration.shape[:2], dtype=float)
    time_radial = np.zeros(acceleration.shape[:2], dtype=float)
    positive_radius = r[None, 1:]
    anisotropy[:, 1:] = (
        positive_radius**2 * acceleration[:, 1:, 4]
    )
    time_radial[:, 1:] = positive_radius * acceleration[:, 1:, 5]
    return anisotropy, time_radial


def reconcile_joint_parent_native_axis_null_channels(
    acceleration,
    r,
    *,
    anisotropy_numerator_tt,
    time_radial_tt,
):
    """Set only the joint parent's two null axis coefficients by parity.

    ``anisotropy_numerator_tt`` is the physical
    ``h_rr,tt-h_perp,tt`` field and ``time_radial_tt`` is ``h_0r,tt``.
    Their exact positive-zero axis traces are prerequisites.  The limits use
    the frozen native seven-point operators; no fit of the reduced fields is
    performed.
    """
    source = np.asarray(acceleration, dtype=float)
    r = np.asarray(r, dtype=float)
    anisotropy = np.asarray(anisotropy_numerator_tt, dtype=float)
    time_radial = np.asarray(time_radial_tt, dtype=float)
    if r.ndim != 1:
        raise ValueError("joint parent radial coordinate must be one-dimensional")
    if source.ndim != 3 or source.shape[1:] != (len(r), 9):
        raise ValueError("joint parent acceleration must have shape (z,r,9)")
    if (
        len(r) < 7
        or not np.all(np.isfinite(r))
        or np.any(np.diff(r) <= 0.0)
        or r[0] != 0.0
        or np.signbit(r[0])
        or r[-1] <= 0.0
    ):
        raise ValueError(
            "native parity requires an increasing radial grid with exact +0 axis"
        )
    if (
        anisotropy.shape != source.shape[:2]
        or time_radial.shape != source.shape[:2]
    ):
        raise ValueError("physical parity numerators must have shape (z,r)")
    if not all(np.all(np.isfinite(value)) for value in (
        source, anisotropy, time_radial,
    )):
        raise ValueError("native parity inputs must be finite")

    zero = np.zeros(source.shape[0], dtype=float)
    for name, numerator in (
        ("anisotropy", anisotropy),
        ("time-radial", time_radial),
    ):
        if (
            not np.array_equal(numerator[:, 0], zero)
            or np.any(np.signbit(numerator[:, 0]))
        ):
            raise ValueError(
                f"{name} numerator must be exact IEEE positive zero at the axis"
            )

    expected_anisotropy, expected_time_radial = (
        _canonical_joint_parent_axis_numerators(source, r)
    )
    positive_anisotropy_bits = np.ascontiguousarray(
        anisotropy[:, 1:],
    ).view(np.uint64)
    expected_anisotropy_bits = np.ascontiguousarray(
        expected_anisotropy[:, 1:],
    ).view(np.uint64)
    positive_time_radial_bits = np.ascontiguousarray(
        time_radial[:, 1:],
    ).view(np.uint64)
    expected_time_radial_bits = np.ascontiguousarray(
        expected_time_radial[:, 1:],
    ).view(np.uint64)
    if not (
        np.array_equal(positive_anisotropy_bits, expected_anisotropy_bits)
        and np.array_equal(positive_time_radial_bits, expected_time_radial_bits)
    ):
        raise ValueError(
            "physical parity numerators disagree with the reduced acceleration"
        )

    parent_radius = float(r[-1])
    s = (r / parent_radius)**2
    ds = derivative_matrix(s, 1, 7)
    dr = derivative_matrix(r, 1, 7)
    if hasattr(ds, "toarray"):
        ds = ds.toarray()
    if hasattr(dr, "toarray"):
        dr = dr.toarray()
    q4_axis = (ds @ anisotropy.T).T[:, 0] / parent_radius**2
    q5_axis = (dr @ time_radial.T).T[:, 0]
    if not np.all(np.isfinite(np.stack((q4_axis, q5_axis)))):
        raise RuntimeError("native parity reconciliation produced nonfinite limits")

    result = source.copy()
    source_bits = np.ascontiguousarray(source).view(np.uint64).reshape(source.shape)
    result[:, 0, 4] = q4_axis
    result[:, 0, 5] = q5_axis
    result_bits = np.ascontiguousarray(result).view(np.uint64).reshape(result.shape)
    protected = np.ones(source.shape, dtype=bool)
    protected[:, 0, 4:6] = False
    protected_bitwise = bool(np.array_equal(
        source_bits[protected], result_bits[protected],
    ))
    if not protected_bitwise:
        raise AssertionError("native parity reconciliation changed owned data")
    return result, {
        "method": "native-seven-point-physical-numerator-axis-parity",
        "stencil_width": 7,
        "parent_radius": parent_radius,
        "anisotropy_axis_positive_zero": True,
        "time_radial_axis_positive_zero": True,
        "positive_radius_reassembly_bitwise": True,
        "only_q4_q5_axis_changed_bitwise": protected_bitwise,
        "polynomial_fit_applied": False,
    }


def solve_joint_parent_acceleration_fixed_point(
    position_state,
    position,
    bulk_acceleration,
    z,
    r,
    background,
    *,
    parent_label,
    parent_identity,
    maximum_maps=8,
    convergence_tolerance=1e-12,
    required_consecutive=2,
    capture_profiles=True,
):
    """Apply the frozen full-update source/acceleration map.

    Every map restarts the owned wall solve from ``bulk_acceleration``.  No
    endpoint value or driver memory is carried through the solve itself.
    """
    q = np.asarray(position, dtype=float)
    a_bulk = np.asarray(bulk_acceleration, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    expected = (len(z), len(r), 9)
    if q.shape != expected or a_bulk.shape != expected:
        raise ValueError("joint acceleration fields have the wrong shape")
    maximum_maps = int(maximum_maps)
    required_consecutive = int(required_consecutive)
    tolerance = float(convergence_tolerance)
    if maximum_maps != 8 or required_consecutive != 2 or tolerance != 1e-12:
        raise ValueError("Protocol-125 fixed-point settings are frozen")
    (
        parent_label,
        parent_identity,
        fixed_settings,
        input_provenance,
        attempt_fingerprint,
    ) = _input_provenance(
        position_state,
        q,
        a_bulk,
        z,
        r,
        background,
        parent_label=parent_label,
        parent_identity=parent_identity,
    )
    represented_q = position_state.evaluate_reduced(z, r)
    if not np.allclose(represented_q, q, rtol=0.0, atol=1e-12):
        raise ValueError("position representation does not reproduce the parent")

    current = a_bulk.copy()
    consecutive = 0
    history = []
    final_records = None
    final_triplet = None
    final_axis_record = None
    for iteration in range(maximum_maps):
        jet = represented_position_jet(position_state, z, r, current)
        triplet = initial_driver_source_triplet_from_acceleration(
            jet, z, r, background,
        )
        _validate_source_triplet_core(triplet, expected)
        work = a_bulk.copy()
        try:
            coupled, coupled_record = (
                solve_compact_wall_coupled_phi_normal_acceleration(
                    q,
                    np.zeros_like(q),
                    work,
                    triplet["source"],
                    triplet["source_time"],
                    triplet["source_second_time"],
                    z,
                    r,
                    background,
                    stencil_width=7,
                    capture_profiles=bool(capture_profiles),
                )
            )
        except CompactWallCoupledAlgebraicGateError as error:
            event = _structured_wall_failure_event(
                error, owner="coupled", r=r, failed_map=iteration + 1,
            )
            failure = _acceleration_failure_record(
                parent_label=parent_label,
                parent_identity=parent_identity,
                attempt_fingerprint=attempt_fingerprint,
                fixed_settings=fixed_settings,
                input_provenance=input_provenance,
                failure_group="wall_algebra",
                failure_reason="coupled_wall_algebraic_gate_failure",
                history=history,
                consecutive_converged_maps=consecutive,
                current=current,
                source_triplet=triplet,
                failure_event=event,
                failed_stage_input=work,
            )
            raise Protocol125AccelerationScientificFailure(failure) from error
        _validate_wall_owner_output(
            "coupled", coupled, coupled_record, expected,
        )
        try:
            selective, selective_record = (
                solve_compact_wall_tangential_chi_acceleration(
                    q,
                    np.zeros_like(q),
                    coupled,
                    z,
                    r,
                    background,
                    stencil_width=7,
                    maximum_normalized_residual=1e-12,
                    capture_profiles=bool(capture_profiles),
                )
            )
        except SelectiveWallAlgebraicGateError as error:
            event = _structured_wall_failure_event(
                error, owner="selective", r=r, failed_map=iteration + 1,
            )
            failure = _acceleration_failure_record(
                parent_label=parent_label,
                parent_identity=parent_identity,
                attempt_fingerprint=attempt_fingerprint,
                fixed_settings=fixed_settings,
                input_provenance=input_provenance,
                failure_group="wall_algebra",
                failure_reason="selective_wall_algebraic_gate_failure",
                history=history,
                consecutive_converged_maps=consecutive,
                current=current,
                source_triplet=triplet,
                failure_event=event,
                failed_stage_input=coupled,
                coupled=coupled_record,
            )
            raise Protocol125AccelerationScientificFailure(failure) from error
        except RuntimeError as error:
            event = _selective_residual_failure_event(
                error, r=r, failed_map=iteration + 1,
            )
            if event is None:
                raise
            failure = _acceleration_failure_record(
                parent_label=parent_label,
                parent_identity=parent_identity,
                attempt_fingerprint=attempt_fingerprint,
                fixed_settings=fixed_settings,
                input_provenance=input_provenance,
                failure_group="wall_algebra",
                failure_reason=(
                    "selective_wall_normalized_residual_gate_failure"
                ),
                history=history,
                consecutive_converged_maps=consecutive,
                current=current,
                source_triplet=triplet,
                failure_event=event,
                failed_stage_input=coupled,
                coupled=coupled_record,
            )
            raise Protocol125AccelerationScientificFailure(failure) from error
        _validate_wall_owner_output(
            "selective", selective, selective_record, expected,
        )
        gauged = impose_compact_wall_normal_tangential_acceleration(selective)
        anisotropy_tt, time_radial_tt = (
            _canonical_joint_parent_axis_numerators(gauged, r)
        )
        completed, axis_record = (
            reconcile_joint_parent_native_axis_null_channels(
                gauged,
                r,
                anisotropy_numerator_tt=anisotropy_tt,
                time_radial_tt=time_radial_tt,
            )
        )
        next_jet = represented_position_jet(position_state, z, r, completed)
        next_triplet = initial_driver_source_triplet_from_acceleration(
            next_jet, z, r, background,
        )
        _validate_source_triplet_core(next_triplet, expected)
        acceleration_change = _scaled_linf(completed, current)
        source_change = max(
            _scaled_linf(triplet[name], next_triplet[name])
            for name in ("source", "source_time", "source_second_time")
        )
        if acceleration_change < tolerance and source_change < tolerance:
            consecutive += 1
        else:
            consecutive = 0
        history.append({
            "map": iteration+1,
            "acceleration_scaled_Linf_change": acceleration_change,
            "source_triplet_scaled_Linf_change": source_change,
            "consecutive_converged_maps": consecutive,
            "coupled": coupled_record,
            "selective": selective_record,
            "normal_tangential_correction_scaled_Linf": _scaled_linf(
                selective, gauged,
            ),
            "axis_reconciliation_scaled_Linf": _scaled_linf(
                gauged, completed,
            ),
            "axis_reconciliation": axis_record,
        })
        current = completed
        final_records = (coupled_record, selective_record)
        final_triplet = next_triplet
        final_axis_record = axis_record
        if consecutive >= required_consecutive:
            break
    if consecutive < required_consecutive:
        event = {
            "owner": "fixed_point",
            "failed_map": None,
            "exception_type": None,
            "message": (
                "joint parent acceleration/source fixed point did not converge"
            ),
            "gate": "two_consecutive_map_convergence",
            "radial_index": None,
            "radius": None,
            "field": None,
            "diagnostics": {
                "maximum_maps": maximum_maps,
                "required_consecutive": required_consecutive,
                "observed_consecutive": consecutive,
                "convergence_tolerance": tolerance,
                "final_acceleration_scaled_Linf_change": history[-1][
                    "acceleration_scaled_Linf_change"
                ],
                "final_source_triplet_scaled_Linf_change": history[-1][
                    "source_triplet_scaled_Linf_change"
                ],
            },
        }
        failure = _acceleration_failure_record(
            parent_label=parent_label,
            parent_identity=parent_identity,
            attempt_fingerprint=attempt_fingerprint,
            fixed_settings=fixed_settings,
            input_provenance=input_provenance,
            failure_group="acceleration_closure",
            failure_reason="fixed_point_nonconvergence",
            history=history,
            consecutive_converged_maps=consecutive,
            current=current,
            source_triplet=final_triplet,
            failure_event=event,
            coupled=final_records[0],
            selective=final_records[1],
            axis_reconciliation=final_axis_record,
        )
        raise Protocol125AccelerationScientificFailure(failure)

    velocity = np.zeros_like(q)
    normal = compact_wall_normal_gauge_acceleration_residuals(
        q,
        velocity,
        current,
        final_triplet["source"],
        final_triplet["source_time"],
        final_triplet["source_second_time"],
        z,
        r,
        background,
        7,
        radial_buffer=0,
        capture_profiles=bool(capture_profiles),
    )
    wall = {
        name: wall_junction_second_tangent(
            q, velocity, current, z, r, background, name, 7,
        )
        for name in ("lower", "upper")
    }
    normal_maximum = _validate_normal_gauge_measurement(normal)
    _validate_wall_tangent_measurement(wall)
    if normal_maximum >= 1e-10:
        event = {
            "owner": "final_normal_gauge",
            "failed_map": None,
            "exception_type": None,
            "message": "final normal-GH acceleration row does not close",
            "gate": "strict_normalized_residual",
            "radial_index": None,
            "radius": None,
            "field": "normal_GH",
            "diagnostics": {
                "maximum_normalized_residual": normal_maximum,
                "strict_ceiling": 1e-10,
                "strict_gate_failed": True,
            },
        }
        failure = _acceleration_failure_record(
            parent_label=parent_label,
            parent_identity=parent_identity,
            attempt_fingerprint=attempt_fingerprint,
            fixed_settings=fixed_settings,
            input_provenance=input_provenance,
            failure_group="acceleration_closure",
            failure_reason="final_normal_gauge_closure_failure",
            history=history,
            consecutive_converged_maps=consecutive,
            current=current,
            source_triplet=final_triplet,
            failure_event=event,
            coupled=final_records[0],
            selective=final_records[1],
            axis_reconciliation=final_axis_record,
            final_normal_gauge=normal,
            final_wall_second_tangent=wall,
        )
        raise Protocol125AccelerationScientificFailure(failure)
    return current, {
        "method": ACCELERATION_FIXED_POINT_METHOD,
        "history": history,
        "maps_used": len(history),
        "consecutive_converged_maps": consecutive,
        "source_triplet": final_triplet,
        "coupled": final_records[0],
        "selective": final_records[1],
        "axis_reconciliation": final_axis_record,
        "normal_gauge": normal,
        "wall_second_tangent": wall,
        "delta_acceleration": current-a_bulk,
        "outer_overwrite_applied": False,
        "generic_axis_fill_applied": False,
        "endpoint_history_carried": False,
    }
