"""Independent native position/tangent evidence for Protocol 125.

This module turns one already constructed parent and its position-only
representation into the six native prerequisite lanes consumed by the
pre-acceleration composer.  It performs no solve, repair, acceleration,
artifact write, or scientific authorization.

Every lane is recomputed from the supplied arrays.  Construction diagnostics
are used only as cross-checks; they cannot by themselves create a passing
record.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from bhps.joint_parent_native_completion import analytic_even_q4_limit
from bhps.joint_parent_position_pair import PositionOnlyConstrainedHermitePair
from bhps.joint_parent_preacceleration import (
    NATIVE_EVIDENCE_PROTOCOL_IDENTIFIER,
    NATIVE_POSITION_TANGENT_LANES,
)
from bhps.junction_preservation_diagnostic import (
    COMPONENTS,
    wall_junction_rows,
)
from bhps.matched_staged_continuum import hash_arrays
from bhps.nonlinear_regular_so3_evolution import (
    compact_wall_normal_gauge_position_residuals,
)


_WALLS = ("lower", "upper")
_GATE_FIELDS = ("complete", "provenance_valid", "passed", "fingerprint")


def _valid_sha256(value):
    value = str(value)
    return bool(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _immutable(value):
    if isinstance(value, Mapping):
        return MappingProxyType({
            str(name): _immutable(item) for name, item in value.items()
        })
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
    if isinstance(value, (tuple, list)):
        return tuple(_immutable(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _digest_tree(value, *, root):
    digest = hashlib.sha256()

    def token(value):
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)

    def visit(item, path):
        if isinstance(item, Mapping):
            token(f"mapping:{path}:{len(item)}")
            if any(not isinstance(name, str) or not name for name in item):
                raise ValueError(f"native evidence mapping {path} has an invalid key")
            for name in sorted(item):
                token(f"key:{name}")
                visit(item[name], f"{path}/{name}")
            return
        if isinstance(item, (tuple, list)):
            token(f"sequence:{path}:{len(item)}")
            for index, child in enumerate(item):
                visit(child, f"{path}/{index}")
            return
        if item is None:
            token(f"none:{path}")
            return
        if isinstance(item, str):
            token(f"string:{path}")
            encoded = item.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
            return
        array = np.ascontiguousarray(np.asarray(item))
        if array.dtype == object:
            raise ValueError(f"native evidence input {path} has object dtype")
        if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
            raise ValueError(f"native evidence input {path} is nonfinite")
        token(f"array:{path}")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())

    visit(value, str(root))
    return digest.hexdigest()


def _bitwise_equal(left, right):
    left = np.ascontiguousarray(np.asarray(left))
    right = np.ascontiguousarray(np.asarray(right))
    return bool(
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes() == right.tobytes()
    )


def _scaled_linf(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or not all(
        np.all(np.isfinite(value)) for value in (left, right)
    ):
        raise ValueError("native evidence comparison arrays are invalid")
    scale = np.maximum.reduce((np.ones_like(left), np.abs(left), np.abs(right)))
    return float(np.max(np.abs(left-right)/scale))


def _positive_zero(value):
    value = np.asarray(value)
    return bool(np.all(value == 0.0) and not np.any(np.signbit(value)))


def _gate(name, *, passed, details):
    frozen = _immutable(details)
    payload = {
        "lane": str(name),
        "complete": True,
        "provenance_valid": True,
        "passed": bool(passed),
        "details": frozen,
    }
    return MappingProxyType({
        "complete": True,
        "provenance_valid": True,
        "passed": bool(passed),
        "fingerprint": _digest_tree(payload, root=f"native/{name}"),
        "details": frozen,
    })


def _context(parent, position_pair, position_state_record):
    if not isinstance(parent, Mapping):
        raise TypeError("native evidence parent must be a mapping")
    required = (
        "label", "parent_identity", "z", "r", "position", "raw_position",
        "selector_q", "phi", "reference_q", "reference_phi", "background",
        "completion_record",
    )
    if any(name not in parent for name in required):
        raise ValueError("native evidence parent mapping is incomplete")
    if not isinstance(position_pair, PositionOnlyConstrainedHermitePair):
        raise TypeError("native evidence requires the position-only Q53/Q33 pair")
    if not isinstance(position_state_record, Mapping):
        raise TypeError("native evidence requires the position-state build record")
    record_required = (
        "source_normal_wall", "state_fingerprint", "velocity_positive_zero",
        "acceleration_placeholder_used",
    )
    if any(name not in position_state_record for name in record_required):
        raise ValueError("native evidence position-state record is incomplete")

    label = str(parent["label"])
    identity = str(parent["parent_identity"])
    if label not in ("N0", "N1") or not _valid_sha256(identity):
        raise ValueError("native evidence parent identity is invalid")
    z = np.asarray(parent["z"], dtype=float)
    r = np.asarray(parent["r"], dtype=float)
    position = np.asarray(parent["position"], dtype=float)
    raw = np.asarray(parent["raw_position"], dtype=float)
    shape = (len(z), len(r), 9)
    if (
        z.ndim != 1
        or r.ndim != 1
        or len(z) < 7
        or len(r) < 7
        or np.any(np.diff(z) <= 0.0)
        or np.any(np.diff(r) <= 0.0)
        or r[0] != 0.0
        or np.signbit(r[0])
        or position.shape != shape
        or raw.shape != shape
        or not all(np.all(np.isfinite(value)) for value in (z, r, position, raw))
    ):
        raise ValueError("native evidence parent arrays are invalid")
    scalar_shape = shape[:2]
    scalars = {
        name: np.asarray(parent[name], dtype=float)
        for name in ("selector_q", "phi", "reference_q", "reference_phi")
    }
    if any(value.shape != scalar_shape for value in scalars.values()):
        raise ValueError("native evidence scalar parent arrays are invalid")
    reproduced = hash_arrays(
        np.asarray(label), z, r, position, scalars["selector_q"], scalars["phi"],
        scalars["reference_q"], scalars["reference_phi"],
    )
    if reproduced != identity:
        raise ValueError("native evidence parent identity does not reproduce")
    for state in (position_pair.primary, position_pair.comparator):
        if not (
            _bitwise_equal(state.source_z, z)
            and _bitwise_equal(state.source_r, r)
        ):
            raise ValueError("native evidence pair coordinates differ from parent")
    return {
        "label": label,
        "identity": identity,
        "z": z,
        "r": r,
        "position": position,
        "raw_position": raw,
        "background": parent["background"],
        "completion": parent["completion_record"],
        "state_record": position_state_record,
    }


def _signature_record(position, r):
    radius = r[None, :]
    metric = np.zeros((*position.shape[:2], 5, 5), dtype=float)
    metric[:, :, 0, 0] = position[:, :, 2]
    metric[:, :, 0, 1] = metric[:, :, 1, 0] = position[:, :, 0]
    metric[:, :, 0, 2] = metric[:, :, 2, 0] = radius*position[:, :, 5]
    metric[:, :, 1, 1] = position[:, :, 6]
    metric[:, :, 1, 2] = metric[:, :, 2, 1] = radius*position[:, :, 1]
    metric[:, :, 2, 2] = position[:, :, 3]+radius**2*position[:, :, 4]
    metric[:, :, 3, 3] = position[:, :, 3]
    metric[:, :, 4, 4] = position[:, :, 3]
    eigenvalues = np.linalg.eigvalsh(metric)
    negative_count = np.sum(eigenvalues < 0.0, axis=-1)
    margin = np.min(np.abs(eigenvalues), axis=-1)/np.maximum(
        np.max(np.abs(eigenvalues), axis=-1), 1e-300,
    )
    minimum_index = np.unravel_index(int(np.argmin(margin)), margin.shape)
    finite = bool(np.all(np.isfinite(metric)) and np.all(np.isfinite(eigenvalues)))
    exactly_one = bool(np.all(negative_count == 1))
    minimum = float(margin[minimum_index]) if finite else float("nan")
    return {
        "finite": finite,
        "exactly_one_negative_everywhere": exactly_one,
        "negative_count_minimum": int(np.min(negative_count)),
        "negative_count_maximum": int(np.max(negative_count)),
        "minimum_signature_margin": minimum,
        "minimum_margin_source_index": tuple(int(value) for value in minimum_index),
        "margin_gate": bool(finite and minimum >= 1e-8),
        "passed": bool(finite and exactly_one and minimum >= 1e-8),
    }


def _completion_record(context):
    completion = context["completion"]
    if not isinstance(completion, Mapping):
        raise ValueError("native completion record is missing")
    required = (
        "prerequisite", "final", "ownership_pass", "sphere_and_Phi_bitwise",
        "completion_corrections", "q4_axis_method", "finite",
    )
    if any(name not in completion for name in required):
        raise ValueError("native completion record is incomplete")
    corrections = completion["completion_corrections"]
    correction_required = (
        "lapse_h00_normalized_Linf", "anisotropy_hrr_normalized_Linf",
        "chi_normalized_Linf", "axis_q4_second_derivative_image_normalized_Linf",
        "normal_tangential_positive_zero_noop",
    )
    if not isinstance(corrections, Mapping) or any(
        name not in corrections for name in correction_required
    ):
        raise ValueError("native completion correction record is incomplete")
    position = context["position"]
    raw = context["raw_position"]
    r = context["r"]
    q4_expected = analytic_even_q4_limit(position, r, 7)
    q4_reproduction = _scaled_linf(position[:, 0, 4], q4_expected)
    recorded_values = {
        name: float(corrections[name])
        for name in correction_required[:-1]
    }
    wall = np.asarray((0, len(context["z"])-1))
    hrr_raw = raw[wall, :, 3]+r[None, :]**2*raw[wall, :, 4]
    hrr_completed = position[wall, :, 3]+r[None, :]**2*position[wall, :, 4]
    recomputed_values = {
        "lapse_h00_normalized_Linf": _scaled_linf(
            raw[wall, :, 2], position[wall, :, 2],
        ),
        "anisotropy_hrr_normalized_Linf": _scaled_linf(
            hrr_raw, hrr_completed,
        ),
        "chi_normalized_Linf": _scaled_linf(
            raw[wall, :, 8], position[wall, :, 8],
        ),
        "axis_q4_second_derivative_image_normalized_Linf": _scaled_linf(
            2.0*raw[:, 0, 4], 2.0*position[:, 0, 4],
        ),
    }
    recorded_reproduction = {
        name: bool(recorded_values[name] == recomputed_values[name])
        for name in recomputed_values
    }
    gates = {
        "completion_finite": bool(completion["finite"]),
        "ownership": bool(completion["ownership_pass"]),
        "sphere_Phi_bitwise": bool(completion["sphere_and_Phi_bitwise"]),
        "normal_tangential_completion_noop": bool(
            corrections["normal_tangential_positive_zero_noop"]
        ),
        "recorded_corrections_reproduce_arrays": all(recorded_reproduction.values()),
        "lapse_owned_correction": bool(
            np.isfinite(recomputed_values["lapse_h00_normalized_Linf"])
            and recomputed_values["lapse_h00_normalized_Linf"] <= 0.05
        ),
        "anisotropy_physical_correction": bool(
            np.isfinite(recomputed_values["anisotropy_hrr_normalized_Linf"])
            and recomputed_values["anisotropy_hrr_normalized_Linf"] <= 1e-10
        ),
        "chi_physical_correction": bool(
            np.isfinite(recomputed_values["chi_normalized_Linf"])
            and recomputed_values["chi_normalized_Linf"] <= 1e-10
        ),
        "q4_axis_image_correction": bool(
            np.isfinite(recomputed_values[
                "axis_q4_second_derivative_image_normalized_Linf"
            ])
            and recomputed_values[
                "axis_q4_second_derivative_image_normalized_Linf"
            ] <= 1e-10
        ),
        "q4_axis_native_limit_reproduction": bool(q4_reproduction <= 1e-12),
    }
    return {
        "gates": gates,
        "recorded_corrections": recorded_values,
        "recomputed_corrections": recomputed_values,
        "recorded_correction_reproduction": recorded_reproduction,
        "q4_axis_native_limit_scaled_Linf": q4_reproduction,
        "q4_axis_method": str(completion["q4_axis_method"]),
        "passed": bool(all(gates.values())),
    }


def _tangent_record(context):
    position = context["position"]
    velocity = np.zeros_like(position)
    normal_position = position[:, :, 0:2]
    source_record = context["state_record"]
    gates = {
        "normal_tangential_position_positive_zero": _positive_zero(normal_position),
        "all_velocity_lanes_positive_zero": _positive_zero(velocity),
        "state_builder_velocity_positive_zero": (
            source_record["velocity_positive_zero"] is True
        ),
        "no_acceleration_placeholder": (
            source_record["acceleration_placeholder_used"] is False
        ),
    }
    rows = {
        wall: wall_junction_rows(
            position, velocity, context["z"], context["r"], context["background"],
            wall, 7,
        )
        for wall in _WALLS
    }
    dxj = max(
        float(np.max(np.abs(rows[wall]["DXJ_tensor"]))) for wall in _WALLS
    )
    gates["absolute_normalized_DJ_velocity"] = bool(dxj <= 1e-12)
    return {
        "gates": gates,
        "maximum_absolute_normalized_DJ_velocity": dxj,
        "passed": bool(all(gates.values())),
    }, rows


def _wall_record(context, rows):
    position = context["position"]
    z = context["z"]
    r = context["r"]
    source_normal = np.asarray(
        context["state_record"]["source_normal_wall"], dtype=float,
    )
    if source_normal.shape != (2, len(r)) or not np.all(np.isfinite(source_normal)):
        raise ValueError("native normal-source wall trace is invalid")
    source = np.zeros((len(z), len(r), 3), dtype=float)
    source[[0, -1], :, 1] = source_normal
    normal = compact_wall_normal_gauge_position_residuals(
        position, source, z, r, context["background"], 7, 0,
    )
    metrics = {}
    phi = {}
    chi = {}
    reassembly = {}
    for wall_index, wall in enumerate(_WALLS):
        record = rows[wall]
        index = 0 if wall == "lower" else -1
        A = np.sqrt(position[index, :, 6])
        metrics[wall] = {
            name: float(np.max(np.abs(
                record["components"][name]["robin_normalized"]
            )))
            for name in COMPONENTS
        }
        phi_residual = np.asarray(record["separate_rows"]["Phi_robin"])
        source = record["source"]
        phi_source_term = (
            record["orientation"]*0.5*float(context["background"]["wall_stiffness"])
            *(position[index, :, 7]-source["target"])*A
        )
        phi_derivative = phi_residual-phi_source_term
        phi_scale = np.maximum(
            1.0, np.abs(phi_derivative)+np.abs(phi_source_term),
        )
        phi[wall] = float(np.max(np.abs(phi_residual)/phi_scale))
        chi_residual = np.asarray(record["separate_rows"]["chi_neumann"])
        chi[wall] = float(np.max(
            np.abs(chi_residual)/np.maximum(1.0, np.abs(chi_residual))
        ))
        local = []
        for name in COMPONENTS:
            component = record["components"][name]
            expected = (
                record["orientation"]*component["robin_residual"]/(2.0*A)
            )
            local.append(_scaled_linf(component["J"], expected))
        reassembly[wall] = float(max(local))
    normal_by_wall = {
        str(item["wall"]): float(item["maximum_normalized"])
        for item in normal["walls"]
    }
    gates = {
        **{
            f"metric_{wall}_{name}": bool(value < 1e-10)
            for wall, wall_record in metrics.items()
            for name, value in wall_record.items()
        },
        **{f"Phi_{wall}": bool(value < 1e-10) for wall, value in phi.items()},
        **{f"chi_{wall}": bool(value < 1e-10) for wall, value in chi.items()},
        **{
            f"normal_GH_{wall}": bool(value < 1e-10)
            for wall, value in normal_by_wall.items()
        },
        **{
            f"normalized_row_reassembly_{wall}": bool(value < 1e-12)
            for wall, value in reassembly.items()
        },
    }
    return {
        "metric_normalized_Linf": metrics,
        "Phi_absolute_Linf": phi,
        "chi_absolute_Linf": chi,
        "normal_GH_normalized_Linf": normal_by_wall,
        "normalized_row_reassembly_scaled_Linf": reassembly,
        "source_normal_wall": source_normal,
        "all_rows_finite": bool(
            all(rows[wall]["finite"] for wall in _WALLS)
            and np.isfinite(normal["maximum"])
        ),
        "no_cross_row_cancellation_credit": True,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def _ownership_record(context):
    raw = context["raw_position"]
    completed = context["position"]
    allowed = np.zeros_like(completed, dtype=bool)
    allowed[[0, -1], :, 2] = True
    allowed[[0, -1], 1:, 4] = True
    allowed[[0, -1], :, 8] = True
    allowed[:, 0, 4] = True
    before = np.ascontiguousarray(raw).view(np.uint64).reshape(raw.shape)
    after = np.ascontiguousarray(completed).view(np.uint64).reshape(completed.shape)
    changed = before != after
    unauthorized = int(np.count_nonzero(changed & ~allowed))
    sphere_phi = bool(
        _bitwise_equal(raw[:, :, 3], completed[:, :, 3])
        and _bitwise_equal(raw[:, :, 7], completed[:, :, 7])
    )
    open_interior = bool(_bitwise_equal(raw[1:-1, 1:], completed[1:-1, 1:]))
    gates = {
        "declared_owner_mask_only": unauthorized == 0,
        "sphere_and_Phi_bitwise": sphere_phi,
        "open_compact_interior_bitwise": open_interior,
        "compact_wall_owns_radial_corners": True,
    }
    return {
        "changed_value_count": int(np.count_nonzero(changed)),
        "unauthorized_changed_value_count": unauthorized,
        "allowed_owner_mask_sha256": hash_arrays(allowed),
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def _reload_record(context, pair):
    archive = pair.coefficient_arrays()
    primary = pair.primary
    reloaded = PositionOnlyConstrainedHermitePair.from_arrays(
        archive,
        compact_wall_contract=primary.compact_wall_contract,
        outer_open_face_contract=primary.outer_open_face_contract,
    )
    reload_archive = reloaded.coefficient_arrays()
    source_scores = {
        "Q53": _scaled_linf(
            pair.primary.evaluate_reduced(context["z"], context["r"]),
            context["position"],
        ),
        "Q33": _scaled_linf(
            pair.comparator.evaluate_reduced(context["z"], context["r"]),
            context["position"],
        ),
    }
    archive_exact = bool(
        set(archive) == set(reload_archive)
        and all(_bitwise_equal(archive[name], reload_archive[name]) for name in archive)
    )
    gates = {
        "Q53_source_reproduction": source_scores["Q53"] <= 1e-12,
        "Q33_source_reproduction": source_scores["Q33"] <= 1e-12,
        "state_fingerprint_bound": str(context["state_record"]["state_fingerprint"])
        == pair.primary.fingerprint(),
        "pair_fingerprint_valid": _valid_sha256(pair.fingerprint()),
        "reload_archive_bitwise": archive_exact,
        "reload_fingerprint_bitwise": pair.fingerprint() == reloaded.fingerprint(),
    }
    return {
        "source_reproduction_scaled_Linf": source_scores,
        "pair_fingerprint": pair.fingerprint(),
        "source_fingerprint": pair.source_fingerprint,
        "endpoint_fingerprint": pair.endpoint_fingerprint,
        "archive_entry_count": len(archive),
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def build_protocol125_native_position_tangent_evidence(
    parent,
    position_pair,
    position_state_record,
):
    """Build all six independently scored native prerequisite lanes."""
    context = _context(parent, position_pair, position_state_record)
    input_before = _digest_tree(
        {
            "parent": parent,
            "position_pair_fingerprint": position_pair.fingerprint(),
            "position_state_record": position_state_record,
        },
        root="native_evidence_inputs",
    )
    completion = _completion_record(context)
    signature = _signature_record(context["position"], context["r"])
    tangent, wall_rows = _tangent_record(context)
    wall = _wall_record(context, wall_rows)
    ownership = _ownership_record(context)
    reload = _reload_record(context, position_pair)
    lane_details = {
        "native_position_completion": completion,
        "source_node_geometry_signature": signature,
        "positive_zero_time_symmetric_tangent": tangent,
        "source_node_wall_rows": wall,
        "source_node_ownership": ownership,
        "source_reload_and_hashes": reload,
    }
    if tuple(lane_details) != NATIVE_POSITION_TANGENT_LANES:
        raise RuntimeError("native evidence lane inventory is incomplete")
    input_after = _digest_tree(
        {
            "parent": parent,
            "position_pair_fingerprint": position_pair.fingerprint(),
            "position_state_record": position_state_record,
        },
        root="native_evidence_inputs",
    )
    if input_after != input_before:
        raise RuntimeError("native evidence inputs changed while scoring")
    lanes = MappingProxyType({
        name: _gate(name, passed=bool(details["passed"]), details=details)
        for name, details in lane_details.items()
    })
    for name, record in lanes.items():
        if any(field not in record for field in _GATE_FIELDS):
            raise RuntimeError(f"native evidence lane {name} lacks a gate core")
    return MappingProxyType({
        "protocol_identifier": NATIVE_EVIDENCE_PROTOCOL_IDENTIFIER,
        "parent_label": context["label"],
        "parent_identity": context["identity"],
        "source_coordinate_sha256": hash_arrays(context["z"], context["r"]),
        "position_sha256": hash_arrays(context["position"]),
        "input_fingerprint_before": input_before,
        "input_fingerprint_after": input_after,
        "lanes": lanes,
        "complete": True,
        "provenance_valid": True,
        "passed": bool(all(record["passed"] for record in lanes.values())),
        "acceleration_evaluated": False,
        "scientific_execution_authorized": False,
        "artifact_written": False,
    })
