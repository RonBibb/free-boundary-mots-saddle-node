"""Legacy, fail-closed Protocol-125 manufactured-data smoke scaffold.

This module is not the Protocol-125 production adjudicator and does not report
the protocol's current freeze status.  It preserves an early in-memory smoke
path for synthetic/manufactured regression data only.  It can never construct
N0/N1, write an artifact, run Phase A/B, or return scientific authorization.

Production ordering, recovery, and classification belong to
``joint_parent_scientific_runner`` and ``joint_parent_ordered_adjudicator``.
Keeping this older path explicitly non-authoritative prevents a passing smoke
test from being mistaken for a completed Protocol-125 qualification.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from bhps.joint_parent_acceleration import (
    solve_joint_parent_acceleration_fixed_point,
)
from bhps.joint_parent_boundary_contracts import (
    NativeNormalizedCompactWallContract,
    derive_protocol125_outer_derivative_bundle,
)
from bhps.joint_parent_bulk_reference import (
    FiniteWallReferenceHermitePair,
    REFERENCE_CHANNEL_ORDER,
    SOURCE_CELL_MIDPOINT_SPECS,
)
from bhps.joint_parent_bulk_validation import evaluate_protocol125_bulk_lane
from bhps.joint_parent_fields import bulk_acceleration_from_completed_position
from bhps.joint_parent_position_pair import PositionOnlyConstrainedHermitePair
from bhps.joint_parent_position_state import (
    build_joint_parent_position_state,
    derive_joint_parent_position_outer_contract,
    native_channel_mapping_from_reduced,
)
from bhps.joint_parent_representation import (
    NATIVE_CHANNEL_ORDER,
    RadialFirstConstrainedHermitePair,
)
from bhps.matched_staged_continuum import hash_arrays


PROTOCOL125_STATUS = "LEGACY-VALIDATE-ONLY-NON-AUTHORIZING"
VALIDATE_ONLY_SCOPES = ("synthetic", "manufactured")
FORBIDDEN_PARENT_LABELS = ("N0", "N1")
FORBIDDEN_PHASES = ("Phase A", "Phase B")

# These are permanent scope exclusions of this legacy smoke path, not claims
# that the named production capabilities are globally unimplemented.
LEGACY_VALIDATE_ONLY_EXCLUSIONS = (
    "scientific N0/N1 construction and adjudication",
    "durable checkpoint/reload and frozen-manifest validation",
    "complete pre-acceleration, post-acceleration, and two-parent gate composition",
    "Phase-A qualification and every Phase-B or interface-physics action",
)
# Backward-compatible name for archived callers.  Its semantics are now the
# legacy scaffold's exclusions, never a production readiness inventory.
UNIMPLEMENTED_PROTOCOL_SCORER_GAPS = LEGACY_VALIDATE_ONLY_EXCLUSIONS


class Protocol125AuthorizationError(RuntimeError):
    """Raised when invalid Protocol 125 is asked to execute a candidate."""


def _normal_token(value):
    return "".join(character.lower() for character in str(value) if character.isalnum())


def guard_protocol125_execution(
    execution_scope,
    *,
    parent_label=None,
    phase=None,
    source_z=None,
    source_r=None,
):
    """Reject every scientific candidate while Protocol 125 is invalid.

    The coordinate check prevents a canonical N0/N1 parent from being passed
    through the validate-only API under a cosmetic synthetic label.
    """
    scope = _normal_token(execution_scope)
    allowed = {_normal_token(value) for value in VALIDATE_ONLY_SCOPES}
    label = _normal_token("" if parent_label is None else parent_label)
    phase_token = _normal_token("" if phase is None else phase)
    forbidden_labels = {_normal_token(value) for value in FORBIDDEN_PARENT_LABELS}
    forbidden_phases = {_normal_token(value) for value in FORBIDDEN_PHASES}
    if scope not in allowed:
        raise Protocol125AuthorizationError(
            f"Protocol 125 is {PROTOCOL125_STATUS}; only synthetic/manufactured "
            "in-memory validation is allowed"
        )
    label_is_forbidden = any(
        label == token or label.startswith(token)
        for token in forbidden_labels
    )
    phase_is_forbidden = (
        phase_token in forbidden_phases
        or "phasea" in phase_token
        or "phaseb" in phase_token
    )
    if label_is_forbidden or phase_is_forbidden:
        raise Protocol125AuthorizationError(
            f"Protocol 125 is {PROTOCOL125_STATUS}; N0/N1 and Phase A/B "
            "candidate execution is forbidden"
        )
    if source_z is not None or source_r is not None:
        if source_z is None or source_r is None:
            raise ValueError("both source coordinates are required by the execution guard")
        z = np.asarray(source_z, dtype=float)
        r = np.asarray(source_r, dtype=float)
        coordinate_sha = hash_arrays(z, r)
        for candidate, specification in SOURCE_CELL_MIDPOINT_SPECS.items():
            if (
                (len(z), len(r)) == tuple(specification["source_shape"])
                or coordinate_sha == specification["source_coordinate_sha256"]
            ):
                raise Protocol125AuthorizationError(
                    f"Protocol 125 is {PROTOCOL125_STATUS}; canonical {candidate} "
                    "coordinates cannot enter validate-only execution"
                )
    return MappingProxyType({
        "protocol_status": PROTOCOL125_STATUS,
        "execution_scope": str(execution_scope),
        "parent_label": None if parent_label is None else str(parent_label),
        "phase": None if phase is None else str(phase),
        "validate_only": True,
        "candidate_execution_allowed": False,
    })


def _immutable_array(value, dtype=float):
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({str(name): _freeze(item) for name, item in value.items()})
    if isinstance(value, np.ndarray):
        return _immutable_array(value, None)
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _scaled_linf(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or not all(
        np.all(np.isfinite(value)) for value in (left, right)
    ):
        raise ValueError("scaled comparison arrays must be finite and shape matched")
    scale = np.maximum.reduce((np.ones_like(left), np.abs(left), np.abs(right)))
    return float(np.max(np.abs(left-right)/scale))


def _stack_native(mapping):
    if tuple(mapping) != tuple(NATIVE_CHANNEL_ORDER):
        raise ValueError("native mapping order changed")
    return np.stack(tuple(mapping.values()), axis=-1)


def _endpoint_mapping(values):
    return {
        name: np.asarray(values[:, :, index]).copy()
        for index, name in enumerate(NATIVE_CHANNEL_ORDER)
    }


def _digest_value(digest, name, value):
    digest.update(str(name).encode())
    digest.update(b"\0")
    if isinstance(value, Mapping):
        digest.update(b"mapping\0")
        for child in sorted(value, key=str):
            _digest_value(digest, child, value[child])
        return
    if isinstance(value, (tuple, list)):
        digest.update(b"sequence\0")
        for index, child in enumerate(value):
            _digest_value(digest, index, child)
        return
    if isinstance(value, (str, bytes)) or np.isscalar(value):
        digest.update(type(value).__name__.encode())
        digest.update(b"\0")
        digest.update(repr(value).encode())
        return
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype == object:
        raise ValueError(f"in-memory input {name} has unsupported object dtype")
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes())


def _input_fingerprint(parent, background, reference_pair):
    digest = hashlib.sha256()
    for name in sorted(parent):
        _digest_value(digest, name, parent[name])
    for name in sorted(background):
        _digest_value(digest, name, background[name])
    digest.update(reference_pair.fingerprint().encode())
    return digest.hexdigest()


def _final_pair_fingerprint(pair):
    digest = hashlib.sha256()
    digest.update(pair.primary.fingerprint().encode())
    digest.update(pair.comparator.fingerprint().encode())
    return digest.hexdigest()


def _reference_identity_gate(parent, reference_pair, z, r):
    if not isinstance(reference_pair, FiniteWallReferenceHermitePair):
        raise TypeError("validate-only execution requires the immutable reference pair")
    members = (reference_pair.primary, reference_pair.comparator)
    immutable_arrays = []
    for member in members:
        immutable_arrays.extend((
            member.source_z,
            member.source_r,
            member.source_values,
            member.endpoint_z_first,
            member.surface.z_knots,
            member.surface.s_knots,
            member.surface.coefficients,
        ))
    immutable = all(not np.asarray(value).flags.writeable for value in immutable_arrays)
    same_coordinates = all(
        np.array_equal(member.source_z, z) and np.array_equal(member.source_r, r)
        for member in members
    )
    same_pair_inputs = (
        np.array_equal(reference_pair.primary.source_values, reference_pair.comparator.source_values)
        and np.array_equal(
            reference_pair.primary.endpoint_z_first,
            reference_pair.comparator.endpoint_z_first,
        )
    )
    q_index = REFERENCE_CHANNEL_ORDER.index("q")
    phi_index = REFERENCE_CHANNEL_ORDER.index("Phi")
    parent_matches = (
        np.array_equal(
            np.asarray(parent["reference_q"]),
            reference_pair.primary.source_values[:, :, q_index],
        )
        and np.array_equal(
            np.asarray(parent["reference_phi"]),
            reference_pair.primary.source_values[:, :, phi_index],
        )
    )
    gates = {
        "reference_arrays_immutable": immutable,
        "shared_source_coordinates_bitwise": same_coordinates,
        "Q53_Q33_reference_inputs_bitwise": same_pair_inputs,
        "parent_reference_arrays_bitwise": parent_matches,
    }
    return {"gates": gates, "passed": bool(all(gates.values()))}


def _source_signature_gate(position, r):
    q = np.asarray(position, dtype=float)
    r = np.asarray(r, dtype=float)
    if q.ndim != 3 or q.shape[1:] != (len(r), 9):
        raise ValueError("source signature requires a reduced position field")
    radius = r[None, :]
    shape = q.shape[:2]
    metric = np.zeros((*shape, 5, 5), dtype=float)
    metric[:, :, 0, 0] = q[:, :, 2]
    metric[:, :, 0, 1] = metric[:, :, 1, 0] = q[:, :, 0]
    metric[:, :, 0, 2] = metric[:, :, 2, 0] = radius*q[:, :, 5]
    metric[:, :, 1, 1] = q[:, :, 6]
    metric[:, :, 1, 2] = metric[:, :, 2, 1] = radius*q[:, :, 1]
    metric[:, :, 2, 2] = q[:, :, 3]+radius**2*q[:, :, 4]
    metric[:, :, 3, 3] = q[:, :, 3]
    metric[:, :, 4, 4] = q[:, :, 3]
    eigenvalues = np.linalg.eigvalsh(metric)
    negative_count = np.sum(eigenvalues < 0.0, axis=-1)
    denominator = np.maximum(np.max(np.abs(eigenvalues), axis=-1), 1e-300)
    margin = np.min(np.abs(eigenvalues), axis=-1)/denominator
    finite = bool(np.all(np.isfinite(metric)) and np.all(np.isfinite(eigenvalues)))
    exactly_one_negative = bool(np.all(negative_count == 1))
    minimum = float(np.min(margin)) if finite else float("nan")
    gates = {
        "finite": finite,
        "exactly_one_negative_eigenvalue": exactly_one_negative,
        "minimum_margin_at_least_1e-8": bool(finite and minimum >= 1e-8),
    }
    return {
        "minimum_margin": minimum,
        "negative_count_minimum": int(np.min(negative_count)),
        "negative_count_maximum": int(np.max(negative_count)),
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def _stage(name, gates, **details):
    gates = {str(key): bool(value) for key, value in gates.items()}
    return {
        "name": str(name),
        "gates": gates,
        "passed": bool(all(gates.values())),
        **details,
    }


@dataclass(frozen=True)
class Protocol125ValidateOnlyResult:
    """In-memory integration result with no scientific authorization."""

    classification: str
    stage_order: tuple
    stage_records: Mapping
    position_pair: PositionOnlyConstrainedHermitePair | None
    final_pair: RadialFirstConstrainedHermitePair | None
    bulk_acceleration: np.ndarray | None
    compatible_acceleration: np.ndarray | None
    input_fingerprint: str
    acceleration_started: bool
    validate_only_smoke_pass: bool
    protocol_gate_complete: bool = False
    scientific_candidate_authorized: bool = False
    artifact_written: bool = False
    missing_protocol_scorers: tuple = UNIMPLEMENTED_PROTOCOL_SCORER_GAPS

    def __post_init__(self):
        if self.protocol_gate_complete or self.scientific_candidate_authorized:
            raise ValueError("invalid Protocol 125 cannot authorize a candidate")
        if self.artifact_written:
            raise ValueError("validate-only orchestration may not write an artifact")
        object.__setattr__(self, "stage_order", tuple(self.stage_order))
        object.__setattr__(self, "stage_records", _freeze(self.stage_records))
        object.__setattr__(self, "input_fingerprint", str(self.input_fingerprint))
        if self.bulk_acceleration is not None:
            object.__setattr__(
                self, "bulk_acceleration", _immutable_array(self.bulk_acceleration),
            )
        if self.compatible_acceleration is not None:
            object.__setattr__(
                self,
                "compatible_acceleration",
                _immutable_array(self.compatible_acceleration),
            )


def _blocked_result(
    classification,
    stages,
    fingerprint,
    *,
    position_pair=None,
    bulk_acceleration=None,
    acceleration_started=False,
):
    return Protocol125ValidateOnlyResult(
        classification,
        tuple(stages),
        stages,
        position_pair,
        None,
        bulk_acceleration,
        None,
        fingerprint,
        bool(acceleration_started),
        False,
    )


def run_protocol125_validate_only(
    parent,
    background,
    reference_pair,
    *,
    execution_scope,
    phase=None,
):
    """Run the maximal safe in-memory Protocol-125 integration path.

    This function intentionally has no path, writer, resume, candidate, or
    evolution argument.  A successful return means only that the merged
    implementation components interoperate on a synthetic/manufactured input.
    """
    if not isinstance(parent, Mapping) or not isinstance(background, Mapping):
        raise TypeError("parent and background must be explicit in-memory mappings")
    for name in ("z", "r", "position", "reference_q", "reference_phi"):
        if name not in parent:
            raise ValueError(f"validate-only parent is missing {name}")
    z = np.asarray(parent["z"], dtype=float)
    r = np.asarray(parent["r"], dtype=float)
    label = parent.get("label", None)
    guard = guard_protocol125_execution(
        execution_scope,
        parent_label=label,
        phase=phase,
        source_z=z,
        source_r=r,
    )
    fingerprint_before = _input_fingerprint(parent, background, reference_pair)
    stages = {}

    reference_gate = _reference_identity_gate(parent, reference_pair, z, r)
    signature = _source_signature_gate(parent["position"], r)
    stages["pre_acceleration_construction"] = _stage(
        "pre_acceleration_construction",
        {
            **reference_gate["gates"],
            "source_signature": signature["passed"],
            "protocol_guard_validate_only": not guard["candidate_execution_allowed"],
        },
        source_signature=signature,
        protocol_guard=dict(guard),
    )
    if not stages["pre_acceleration_construction"]["passed"]:
        return _blocked_result(
            "VALIDATE-ONLY-BLOCKED-CONSTRUCTION", stages, fingerprint_before,
        )

    position_outer = derive_joint_parent_position_outer_contract(parent)
    position_state, position_record = build_joint_parent_position_state(
        parent["position"],
        z,
        r,
        background,
        outer_open_face_contract=position_outer,
        parent_r_max=float(r[-1]),
    )
    position_pair = PositionOnlyConstrainedHermitePair.from_primary(position_state)
    primary_outer = position_pair.primary.outer_open_face_residual(z)
    comparator_outer = position_pair.comparator.outer_open_face_residual(z)
    stages["position_only_representation"] = _stage(
        "position_only_representation",
        {
            "source_reproduction": (
                position_record["source_reproduction_scaled_Linf"] <= 1e-12
            ),
            "Q53_outer_contract_reproduction": (
                primary_outer["maximum_normalized"] < 1e-10
            ),
            "Q33_outer_contract_reproduction": (
                comparator_outer["maximum_normalized"] < 1e-10
            ),
            "position_pair_fingerprinted": bool(position_pair.fingerprint()),
            "no_acceleration_placeholder": not position_record[
                "acceleration_placeholder_used"
            ],
        },
        source_reproduction_scaled_Linf=float(
            position_record["source_reproduction_scaled_Linf"]
        ),
        Q53_outer_maximum_normalized=float(primary_outer["maximum_normalized"]),
        Q33_outer_maximum_normalized=float(comparator_outer["maximum_normalized"]),
        position_pair_fingerprint=position_pair.fingerprint(),
    )
    if not stages["position_only_representation"]["passed"]:
        return _blocked_result(
            "VALIDATE-ONLY-BLOCKED-POSITION",
            stages,
            fingerprint_before,
            position_pair=position_pair,
        )

    source_fd = evaluate_protocol125_bulk_lane(
        position_pair.primary,
        reference_pair.primary,
        z,
        r,
        {"mass_squared": float(background["mass_squared"])},
        backend="source_fd7",
        physical_faces=True,
    )
    source_analytic = evaluate_protocol125_bulk_lane(
        position_pair.primary,
        reference_pair.primary,
        z,
        r,
        {"mass_squared": float(background["mass_squared"])},
        backend="analytic",
        physical_faces=True,
    )
    stages["bulk_prerequisite_smoke"] = _stage(
        "bulk_prerequisite_smoke",
        {
            "source_FD7_full_numerical": source_fd["scores"][
                "numerical_gate_pass"
            ],
            "source_analytic_full_numerical": source_analytic["scores"][
                "numerical_gate_pass"
            ],
            "source_FD7_balanced_RMS": source_fd["scores"]["gates"][
                "combined_balanced_RMS"
            ],
            "source_FD7_balanced_Linf": source_fd["scores"]["gates"][
                "combined_balanced_Linf"
            ],
            "source_analytic_balanced_RMS": source_analytic["scores"]["gates"][
                "combined_balanced_RMS"
            ],
            "source_analytic_balanced_Linf": source_analytic["scores"]["gates"][
                "combined_balanced_Linf"
            ],
            "source_FD7_reassembly": source_fd["scores"]["gates"][
                "reassembly_Linf"
            ],
            "source_analytic_reassembly": source_analytic["scores"]["gates"][
                "reassembly_Linf"
            ],
        },
        scope=(
            "source-lane integration smoke only: full midpoint/V0/V1/V2, "
            "common-V2, and two-parent bulk adjudication remain scientific "
            "gates and are not waived"
        ),
        source_FD7_full_numerical_gate=bool(
            source_fd["scores"]["numerical_gate_pass"]
        ),
        source_analytic_full_numerical_gate=bool(
            source_analytic["scores"]["numerical_gate_pass"]
        ),
        source_FD7_retained=source_fd["scores"]["retained"],
        source_analytic_retained=source_analytic["scores"]["retained"],
    )
    if not stages["bulk_prerequisite_smoke"]["passed"]:
        return _blocked_result(
            "VALIDATE-ONLY-BLOCKED-BULK",
            stages,
            fingerprint_before,
            position_pair=position_pair,
        )

    # The first acceleration-producing call occurs only after every preceding
    # integration gate is recorded and has passed.
    bulk_acceleration, bulk_record = bulk_acceleration_from_completed_position(
        parent["position"], z, r, background, stencil_width=7,
    )
    try:
        compatible, fixed = solve_joint_parent_acceleration_fixed_point(
            position_pair.primary,
            parent["position"],
            bulk_acceleration,
            z,
            r,
            background,
            # This legacy path is restricted above to non-canonical,
            # manufactured coordinates.  Give the now provenance-mandatory
            # production solver a deterministic internal binding without
            # promoting the smoke datum to a scientific candidate.
            parent_label="N0",
            parent_identity=fingerprint_before,
        )
    except RuntimeError as error:
        stages["post_prerequisite_acceleration"] = _stage(
            "post_prerequisite_acceleration",
            {"fixed_point_completed": False},
            error=str(error),
        )
        return _blocked_result(
            "VALIDATE-ONLY-BLOCKED-ACCELERATION",
            stages,
            fingerprint_before,
            position_pair=position_pair,
            bulk_acceleration=bulk_acceleration,
            acceleration_started=True,
        )

    source_triplet = fixed["source_triplet"]
    acceleration_change = _scaled_linf(bulk_acceleration, compatible)
    stages["post_prerequisite_acceleration"] = _stage(
        "post_prerequisite_acceleration",
        {
            "fixed_point_completed": True,
            "two_consecutive_maps": fixed["consecutive_converged_maps"] >= 2,
            "normal_gauge_closed": fixed["normal_gauge"]["maximum"] < 1e-10,
            "source_triplet_finite": all(
                np.all(np.isfinite(source_triplet[name]))
                for name in ("source", "source_time", "source_second_time")
            ),
        },
        maps_used=int(fixed["maps_used"]),
        normal_gauge_maximum=float(fixed["normal_gauge"]["maximum"]),
        source_nodal_acceleration_change_scaled_Linf=acceleration_change,
        bulk_lapse_wall_completion_applied=bool(
            bulk_record["lapse_seed"]["wall_completion_applied"]
        ),
    )
    if not stages["post_prerequisite_acceleration"]["passed"]:
        return _blocked_result(
            "VALIDATE-ONLY-BLOCKED-ACCELERATION",
            stages,
            fingerprint_before,
            position_pair=position_pair,
            bulk_acceleration=bulk_acceleration,
            acceleration_started=True,
        )

    position_mapping = native_channel_mapping_from_reduced(parent["position"], r)
    acceleration_mapping = native_channel_mapping_from_reduced(compatible, r)
    position_stack = _stack_native(position_mapping)
    acceleration_stack = _stack_native(acceleration_mapping)
    compact = NativeNormalizedCompactWallContract.build(
        r,
        background,
        position_stack[[0, -1]],
        source_triplet["source"][[0, -1], :, 1],
        source_triplet["source_second_time"][[0, -1], :, 1],
    )
    position_endpoints = compact.z_first_s_jets(
        state_name="position",
        radius=r,
        wall_value_s_jets=(position_stack[[0, -1]],),
    )[0]
    acceleration_endpoints = compact.z_first_s_jets(
        state_name="acceleration",
        radius=r,
        wall_value_s_jets=(acceleration_stack[[0, -1]],),
    )[0]
    final_outer = derive_protocol125_outer_derivative_bundle(
        position_outer, acceleration_mapping,
    )
    final_pair = RadialFirstConstrainedHermitePair.build(
        z,
        r,
        position_mapping,
        acceleration_mapping,
        _endpoint_mapping(position_endpoints),
        _endpoint_mapping(acceleration_endpoints),
        compact_wall_contract=compact,
        outer_open_face_contract=final_outer,
        parent_r_max=float(r[-1]),
    )

    final_position_primary = final_pair.primary.position.evaluate_reduced(z, r)
    final_position_comparator = final_pair.comparator.position.evaluate_reduced(z, r)
    final_acceleration_primary = final_pair.primary.acceleration.evaluate_reduced(z, r)
    final_acceleration_comparator = final_pair.comparator.acceleration.evaluate_reduced(z, r)
    initial_position_primary = position_pair.primary.evaluate_reduced(z, r)
    initial_position_comparator = position_pair.comparator.evaluate_reduced(z, r)
    position_primary_change = _scaled_linf(
        final_position_primary, initial_position_primary,
    )
    position_comparator_change = _scaled_linf(
        final_position_comparator, initial_position_comparator,
    )
    acceleration_primary_reproduction = _scaled_linf(
        final_acceleration_primary, compatible,
    )
    acceleration_comparator_reproduction = _scaled_linf(
        final_acceleration_comparator, compatible,
    )
    outer_records = {
        "position_Q53": final_pair.primary.position.outer_open_face_residual(z),
        "position_Q33": final_pair.comparator.position.outer_open_face_residual(z),
        "acceleration_Q53": final_pair.primary.acceleration.outer_open_face_residual(z),
        "acceleration_Q33": final_pair.comparator.acceleration.outer_open_face_residual(z),
    }
    stages["final_shared_representation"] = _stage(
        "final_shared_representation",
        {
            "Q53_position_unchanged": position_primary_change <= 1e-12,
            "Q33_position_unchanged": position_comparator_change <= 1e-12,
            "Q53_acceleration_source_reproduction": (
                acceleration_primary_reproduction <= 1e-12
            ),
            "Q33_acceleration_source_reproduction": (
                acceleration_comparator_reproduction <= 1e-12
            ),
            **{
                f"{name}_outer": record["maximum_normalized"] < 1e-10
                for name, record in outer_records.items()
            },
            "shared_source_fingerprint": (
                final_pair.primary.source_fingerprint
                == final_pair.comparator.source_fingerprint
            ),
            "shared_endpoint_fingerprint": (
                final_pair.primary.endpoint_fingerprint
                == final_pair.comparator.endpoint_fingerprint
            ),
        },
        Q53_position_change_scaled_Linf=position_primary_change,
        Q33_position_change_scaled_Linf=position_comparator_change,
        Q53_acceleration_reproduction_scaled_Linf=(
            acceleration_primary_reproduction
        ),
        Q33_acceleration_reproduction_scaled_Linf=(
            acceleration_comparator_reproduction
        ),
        outer_maximum_normalized={
            name: float(record["maximum_normalized"])
            for name, record in outer_records.items()
        },
        final_pair_fingerprint=_final_pair_fingerprint(final_pair),
    )

    fingerprint_after = _input_fingerprint(parent, background, reference_pair)
    stages["validate_only_adjudication"] = _stage(
        "validate_only_adjudication",
        {
            "all_integration_stages_pass": all(
                record["passed"] for record in stages.values()
            ),
            "inputs_unchanged": fingerprint_after == fingerprint_before,
            "protocol_remains_invalid": True,
            "artifact_not_written": True,
            "candidate_not_authorized": True,
        },
        protocol_status=PROTOCOL125_STATUS,
        protocol_gate_complete=False,
        scientific_candidate_authorized=False,
        missing_protocol_scorers=UNIMPLEMENTED_PROTOCOL_SCORER_GAPS,
    )
    smoke_pass = bool(all(record["passed"] for record in stages.values()))
    return Protocol125ValidateOnlyResult(
        (
            "VALIDATE-ONLY-SMOKE-PASS"
            if smoke_pass else "VALIDATE-ONLY-BLOCKED-FINAL-REPRESENTATION"
        ),
        tuple(stages),
        stages,
        position_pair,
        final_pair,
        bulk_acceleration,
        compatible,
        fingerprint_before,
        True,
        smoke_pass,
    )
