"""Prospectively frozen position construction for Protocol 125.

The functions here are pure construction helpers.  They do not write result
artifacts, project to target grids, call an evolution RHS, or retry a failed
parent with altered settings.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

import numpy as np

from bhps.corrected_A790_R12_builder import interpolate
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice
from bhps.joint_parent_builder import solve_joint_parent
from bhps.joint_parent_fields import native_position_from_primitives
from bhps.joint_parent_freeze_authority import (
    Protocol125FreezeAuthorityError,
    revalidate_protocol125_freeze_authority_snapshot,
    validate_protocol125_freeze_authority,
)
from bhps.joint_parent_native_completion import (
    Protocol125NativePositionPrerequisiteFailure,
    complete_native_parent_position,
)
from bhps.joint_parent_shape import (
    SHAPE_NORMALIZATION_SHA256,
    frozen_shape_fields_with_radial_derivative,
)
from bhps.matched_staged_continuum import hash_arrays
from bhps.recovery_indexer import sha256_file
from bhps.scalar_pulse import scalar_pulse


AMPLITUDE = 7.90
R_MAX = 12.0
STENCIL_WIDTH = 7
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEALED_PARENT = (
    _PROJECT_ROOT
    / "results/corrected_A790_matched_staged_continuum_recovery"
    / "phase_a_parent_projection.npz"
)
SEALED_PARENT_SHA256 = (
    "30c578ce142159a8e0842a22afab8b436bd85af55f1516c1bfe11fce96968dc7"
)
COMMON_SEED_SHA256 = (
    "86b5dc67b8c7b76cc10fa6cc01a060ee81da375274a85689a5d4b6b49e0fa2f6"
)
KNOT_STATE = _PROJECT_ROOT / "results/corrected_family_knot_A8_state.npz"
KNOT_STATE_SHA256 = (
    "e293184299baf5b791f6a49a91f6e6266ad656072f2e8fd2e39ce21d14c5416e"
)
COEFFICIENT_SHA256 = (
    "7143288a8a7bdfb3f3f6cb27d61bf1ff12b2e7d3b97253f1d4ea85b7e87a9e5b"
)
PARENT_SPECS = {
    "N0": {
        "nz": 145,
        "nr": 325,
        "reference_iterations": 450,
        "coordinate_sha256": (
            "15ae4de252ce8c2ca7ff554aa96e12a53fb7c9150f19f912db98e768422b8b58"
        ),
    },
    "N1": {
        "nz": 161,
        "nr": 361,
        "reference_iterations": 500,
        "coordinate_sha256": (
            "6e877756a88dedfde524fc9860696fffb53cfb5396e202e0fbc8da9ab9f3fb4b"
        ),
    },
}
CONSTRUCTION_FAILURE_PROTOCOL_IDENTIFIER = (
    "Protocol-125-scientific-construction-failure-v2"
)
SUCCESSFUL_CONSTRUCTION_PROTOCOL_IDENTIFIER = (
    "Protocol-125-successful-parent-construction-v2"
)
FINITE_WALL_REFERENCE_CEILING = 1e-9
JOINT_HYBRID_RESIDUAL_CEILING = 1e-10
NATIVE_POSITION_PREREQUISITE_CEILING = 1e-10
SUCCESSFUL_CONSTRUCTION_PROVENANCE_KEYS = (
    "protocol_identifier",
    "finite_wall_maximum_residual",
    "joint_hybrid_maximum_residual",
    "input_fingerprint_before",
    "input_fingerprint_after",
    "construction_input_fingerprint",
    "common_seed_sha256",
    "physical_normalization_identifier",
    "branch_identifier",
    "expected_parent_label",
    "actual_parent_label",
    "independent_of_other_parent",
    "parent_identity",
    "parent_identity_binding_sha256",
    "fingerprint",
)


class Protocol125ScientificConstructionFailure(RuntimeError):
    """A measured construction-gate failure with intact audit provenance."""

    def __init__(self, record):
        self.record = validate_protocol125_construction_failure_record(record)
        super().__init__(
            f"{self.record['parent_label']} construction gate "
            f"{self.record['failure_gate']} failed scientifically"
        )


def _immutable(value):
    array = np.ascontiguousarray(np.asarray(value))
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _freeze_tree(value):
    if isinstance(value, Mapping):
        return MappingProxyType({str(name): _freeze_tree(item) for name, item in value.items()})
    if isinstance(value, np.ndarray):
        return _immutable(value)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_tree(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _hash_tree(value, *, root="construction-failure"):
    digest = hashlib.sha256()

    def visit(item, path):
        label = str(path).encode("utf-8")
        digest.update(len(label).to_bytes(8, "little"))
        digest.update(label)
        if isinstance(item, Mapping):
            digest.update(b"mapping")
            for name in sorted(item):
                visit(item[name], f"{path}/{name}")
            return
        if isinstance(item, (tuple, list)):
            digest.update(b"sequence")
            for index, child in enumerate(item):
                visit(child, f"{path}/{index}")
            return
        array = np.ascontiguousarray(np.asarray(item))
        if array.dtype == object:
            raise ValueError("construction failure record contains object data")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())

    visit(value, str(root))
    return digest.hexdigest()


def _valid_sha256(value):
    value = str(value)
    return bool(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _float64_hex(value):
    return struct.pack("!d", float(value)).hex()


def _successful_parent_identity_binding_payload(
    *,
    parent_identity,
    parent_label,
    construction_input_fingerprint,
    common_seed_sha256,
    physical_normalization_identifier,
    branch_identifier,
    independent_of_other_parent,
):
    """Return the domain-separated identity material for one successful parent."""
    return {
        "record_kind": "Protocol-125-successful-parent-identity-binding-v1",
        "parent_identity": str(parent_identity),
        "parent_label": str(parent_label),
        "construction_input_fingerprint": str(construction_input_fingerprint),
        "common_seed_sha256": str(common_seed_sha256),
        "physical_normalization_identifier": str(
            physical_normalization_identifier
        ),
        "branch_identifier": str(branch_identifier),
        "independent_of_other_parent": independent_of_other_parent,
    }


def build_protocol125_successful_parent_provenance_record(
    label,
    parent_identity,
    construction_input_fingerprint,
    *,
    finite_wall_maximum_residual,
    joint_hybrid_maximum_residual,
):
    """Build and seal the exact provenance record for a successful parent.

    ``parent_identity`` remains the established scientific-state digest used by
    the representation pipeline.  ``parent_identity_binding_sha256`` binds that
    digest to the frozen physical inputs and the independence declaration, so
    neither a valid state digest nor a descriptive identifier can be replayed
    under different construction provenance.
    """
    label = str(label)
    identity = str(parent_identity)
    input_fingerprint = str(construction_input_fingerprint)
    identity_payload = _successful_parent_identity_binding_payload(
        parent_identity=identity,
        parent_label=label,
        construction_input_fingerprint=input_fingerprint,
        common_seed_sha256=COMMON_SEED_SHA256,
        physical_normalization_identifier=SHAPE_NORMALIZATION_SHA256,
        branch_identifier=COEFFICIENT_SHA256,
        independent_of_other_parent=True,
    )
    record = {
        "protocol_identifier": SUCCESSFUL_CONSTRUCTION_PROTOCOL_IDENTIFIER,
        "finite_wall_maximum_residual": float(finite_wall_maximum_residual),
        "joint_hybrid_maximum_residual": float(joint_hybrid_maximum_residual),
        "input_fingerprint_before": input_fingerprint,
        "input_fingerprint_after": input_fingerprint,
        "construction_input_fingerprint": input_fingerprint,
        "common_seed_sha256": COMMON_SEED_SHA256,
        "physical_normalization_identifier": SHAPE_NORMALIZATION_SHA256,
        "branch_identifier": COEFFICIENT_SHA256,
        "expected_parent_label": label,
        "actual_parent_label": label,
        "independent_of_other_parent": True,
        "parent_identity": identity,
        "parent_identity_binding_sha256": _hash_tree(
            identity_payload, root="successful-parent-identity-binding",
        ),
    }
    record["fingerprint"] = _hash_tree(
        record, root="successful-construction-provenance",
    )
    return validate_protocol125_successful_parent_provenance_record(
        record,
        expected_parent_label=label,
        expected_parent_identity=identity,
    )


def validate_protocol125_successful_parent_provenance_record(
    record,
    *,
    expected_parent_label=None,
    expected_parent_identity=None,
):
    """Validate exact frozen identifiers and all successful identity bindings."""
    if not isinstance(record, Mapping) or set(record) != set(
        SUCCESSFUL_CONSTRUCTION_PROVENANCE_KEYS
    ):
        raise ValueError("successful construction provenance schema differs")
    label = str(record["actual_parent_label"])
    identity = str(record["parent_identity"])
    input_before = str(record["input_fingerprint_before"])
    input_after = str(record["input_fingerprint_after"])
    construction_input = str(record["construction_input_fingerprint"])
    if (
        str(record["protocol_identifier"])
        != SUCCESSFUL_CONSTRUCTION_PROTOCOL_IDENTIFIER
        or label not in PARENT_SPECS
        or str(record["expected_parent_label"]) != label
        or (expected_parent_label is not None and label != str(expected_parent_label))
        or not _valid_sha256(identity)
        or (
            expected_parent_identity is not None
            and identity != str(expected_parent_identity)
        )
        or not _valid_sha256(input_before)
        or input_after != input_before
        or construction_input != input_before
    ):
        raise ValueError("successful construction identity or input binding differs")
    if str(record["common_seed_sha256"]) != COMMON_SEED_SHA256:
        raise ValueError("successful construction common-seed identifier differs")
    if (
        str(record["physical_normalization_identifier"])
        != SHAPE_NORMALIZATION_SHA256
    ):
        raise ValueError("successful construction physical normalization differs")
    if str(record["branch_identifier"]) != COEFFICIENT_SHA256:
        raise ValueError("successful construction branch identifier differs")
    if (
        type(record["independent_of_other_parent"]) is not bool
        or record["independent_of_other_parent"] is not True
    ):
        raise ValueError("successful construction independence declaration differs")
    finite_wall = float(record["finite_wall_maximum_residual"])
    joint = float(record["joint_hybrid_maximum_residual"])
    if not (
        np.isfinite(finite_wall)
        and 0.0 <= finite_wall < FINITE_WALL_REFERENCE_CEILING
        and np.isfinite(joint)
        and 0.0 <= joint < JOINT_HYBRID_RESIDUAL_CEILING
    ):
        raise ValueError("successful construction residual evidence does not pass")
    identity_payload = _successful_parent_identity_binding_payload(
        parent_identity=identity,
        parent_label=label,
        construction_input_fingerprint=construction_input,
        common_seed_sha256=record["common_seed_sha256"],
        physical_normalization_identifier=record[
            "physical_normalization_identifier"
        ],
        branch_identifier=record["branch_identifier"],
        independent_of_other_parent=record["independent_of_other_parent"],
    )
    if (
        not _valid_sha256(record["parent_identity_binding_sha256"])
        or str(record["parent_identity_binding_sha256"])
        != _hash_tree(
            identity_payload, root="successful-parent-identity-binding",
        )
    ):
        raise ValueError("successful parent identity binding differs")
    payload_without_fingerprint = {
        name: record[name] for name in record if name != "fingerprint"
    }
    if (
        not _valid_sha256(record["fingerprint"])
        or str(record["fingerprint"]) != _hash_tree(
            payload_without_fingerprint,
            root="successful-construction-provenance",
        )
    ):
        raise ValueError("successful construction provenance fingerprint differs")
    return _freeze_tree(record)


def _construction_failure_record(
    label,
    input_fingerprint,
    reference,
    *,
    failure_gate,
    measured_value,
    ceiling,
    selected=None,
    native_raw_position=None,
    native_prerequisite=None,
):
    z = np.asarray(reference["z"], dtype=float)
    r = np.asarray(reference["r"], dtype=float)
    reference_q = np.asarray(reference["q"], dtype=float)
    reference_phi = np.asarray(reference["phi"], dtype=float)
    payload_arrays = {
        "z": _immutable(z),
        "r": _immutable(r),
        "reference_q": _immutable(reference_q),
        "reference_phi": _immutable(reference_phi),
        "reference_history": _immutable(reference["history"]),
    }
    if selected is not None:
        payload_arrays.update({
            "selected_q": _immutable(selected["q"]),
            "selected_phi": _immutable(selected["phi"]),
            "selected_psi": _immutable(selected["psi"]),
            "selected_history": _immutable(selected["history"]),
            "selected_damping_history": _immutable(
                selected["damping_history"],
            ),
        })
    if native_raw_position is not None:
        payload_arrays["native_raw_position"] = _immutable(
            native_raw_position,
        )
    diagnostics = {
        "reference_converged": bool(reference["converged"]),
        "reference_max_abs_residual": float(reference["max_abs_residual"]),
        "reference_residual_l2": float(reference["residual_l2"]),
        "reference_iteration_cap": int(PARENT_SPECS[str(label)]["reference_iterations"]),
        "reference_stencil_width": int(STENCIL_WIDTH),
    }
    if selected is not None:
        diagnostics.update({
            "selected_converged": bool(selected["converged"]),
            "selected_maximum_residual": float(selected["maximum_residual"]),
            "selected_residual_l2": float(selected["residual_l2"]),
            "selected_iteration_cap": 80,
            "selected_stencil_width": int(STENCIL_WIDTH),
        })
    if native_prerequisite is not None:
        diagnostics["native_position_prerequisite"] = {
            str(name): float(value)
            for name, value in native_prerequisite.items()
        }
    attempt_payload = {
        "parent_label": str(label),
        "construction_input_fingerprint": str(input_fingerprint),
        "failure_gate": str(failure_gate),
        "scientific_payload": payload_arrays,
        "solver_diagnostics": diagnostics,
    }
    attempt_identity = _hash_tree(attempt_payload)
    measured_value = float(measured_value)
    measurement_finite = bool(np.isfinite(measured_value))
    classification = (
        "FAIL-parent-position"
        if str(failure_gate) == "native_position_prerequisite"
        else "FAIL-parent-bulk"
    )
    record = {
        "protocol_identifier": CONSTRUCTION_FAILURE_PROTOCOL_IDENTIFIER,
        "parent_label": str(label),
        "parent_identity": attempt_identity,
        "failure_gate": str(failure_gate),
        "classification": classification,
        "complete": True,
        "provenance_valid": True,
        "passed": False,
        "measured_value": measured_value,
        "measurement_finite": measurement_finite,
        "measured_value_ieee754_hex": _float64_hex(measured_value),
        "strict_ceiling": float(ceiling),
        "strict_gate_failed": bool(
            not measurement_finite or measured_value >= float(ceiling)
        ),
        "construction_input_fingerprint": str(input_fingerprint),
        "source_coordinate_sha256": hash_arrays(z, r),
        "reference_state_sha256": hash_arrays(reference_q, reference_phi),
        "scientific_payload": payload_arrays,
        "solver_diagnostics": diagnostics,
        "acceleration_authorized": False,
        "retry_authorized": False,
        "candidate_or_phase_a_executed": False,
    }
    return _freeze_tree({**record, "fingerprint": _hash_tree(record)})


def validate_protocol125_construction_failure_record(record):
    """Validate a measured construction failure without turning it invalid."""
    required = {
        "protocol_identifier", "parent_label", "parent_identity", "failure_gate",
        "classification", "complete", "provenance_valid", "passed",
        "measured_value", "measurement_finite",
        "measured_value_ieee754_hex", "strict_ceiling",
        "strict_gate_failed",
        "construction_input_fingerprint", "source_coordinate_sha256",
        "reference_state_sha256", "scientific_payload", "solver_diagnostics",
        "acceleration_authorized", "retry_authorized",
        "candidate_or_phase_a_executed", "fingerprint",
    }
    if not isinstance(record, Mapping) or set(record) != required:
        raise ValueError("construction failure record schema differs")
    label = str(record["parent_label"])
    failure_gate = str(record["failure_gate"])
    failure_specs = {
        "finite_wall_reference": (
            "FAIL-parent-bulk", FINITE_WALL_REFERENCE_CEILING, True,
        ),
        "finite_wall_reference_nonfinite": (
            "FAIL-parent-bulk", FINITE_WALL_REFERENCE_CEILING, False,
        ),
        "joint_hybrid_residual": (
            "FAIL-parent-bulk", JOINT_HYBRID_RESIDUAL_CEILING, True,
        ),
        "joint_hybrid_residual_nonfinite": (
            "FAIL-parent-bulk", JOINT_HYBRID_RESIDUAL_CEILING, False,
        ),
        "native_position_prerequisite": (
            "FAIL-parent-position", NATIVE_POSITION_PREREQUISITE_CEILING,
            True,
        ),
    }
    if (
        str(record["protocol_identifier"]) != CONSTRUCTION_FAILURE_PROTOCOL_IDENTIFIER
        or label not in PARENT_SPECS
        or failure_gate not in failure_specs
        or str(record["classification"]) != failure_specs[failure_gate][0]
        or not _valid_sha256(record["parent_identity"])
        or not _valid_sha256(record["construction_input_fingerprint"])
        or not _valid_sha256(record["source_coordinate_sha256"])
        or not _valid_sha256(record["reference_state_sha256"])
    ):
        raise ValueError("construction failure identity or classification differs")
    expected_flags = {
        "complete": True,
        "provenance_valid": True,
        "passed": False,
        "strict_gate_failed": True,
        "acceleration_authorized": False,
        "retry_authorized": False,
        "candidate_or_phase_a_executed": False,
    }
    if any(type(record[name]) is not bool or record[name] is not expected for name, expected in expected_flags.items()):
        raise ValueError("construction failure flags differ")
    measured = float(record["measured_value"])
    ceiling = float(record["strict_ceiling"])
    expected_ceiling = failure_specs[failure_gate][1]
    expected_finite = failure_specs[failure_gate][2]
    if not (
        type(record["measurement_finite"]) is bool
        and record["measurement_finite"] is expected_finite
        and bool(np.isfinite(measured)) is expected_finite
        and type(record["measured_value_ieee754_hex"]) is str
        and record["measured_value_ieee754_hex"] == _float64_hex(measured)
        and ceiling == expected_ceiling
        and (not expected_finite or measured >= ceiling)
    ):
        raise ValueError("construction failure measurement does not fail its gate")
    payload = record["scientific_payload"]
    required_payload = {
        "z", "r", "reference_q", "reference_phi", "reference_history",
    }
    if failure_gate.startswith("joint_hybrid_residual") or failure_gate == (
        "native_position_prerequisite"
    ):
        required_payload |= {
            "selected_q", "selected_phi", "selected_psi",
            "selected_history", "selected_damping_history",
        }
    if failure_gate == "native_position_prerequisite":
        required_payload.add("native_raw_position")
    if not isinstance(payload, Mapping) or set(payload) != required_payload:
        raise ValueError("construction failure scientific payload differs")
    z = np.asarray(payload["z"], dtype=float)
    r = np.asarray(payload["r"], dtype=float)
    q = np.asarray(payload["reference_q"], dtype=float)
    phi = np.asarray(payload["reference_phi"], dtype=float)
    reference_history = np.asarray(payload["reference_history"], dtype=float)
    specification = PARENT_SPECS[label]
    reference_nonfinite_failure = failure_gate == (
        "finite_wall_reference_nonfinite"
    )
    if (
        z.ndim != 1
        or r.ndim != 1
        or len(z) != int(specification["nz"])
        or len(r) != int(specification["nr"])
        or np.any(np.diff(z) <= 0.0)
        or np.any(np.diff(r) <= 0.0)
        or r[0] != 0.0
        or r[-1] != R_MAX
        or q.shape != (len(z), len(r))
        or phi.shape != q.shape
        or reference_history.ndim != 1
        or not 1 <= len(reference_history) <= int(specification["reference_iterations"])
        or not np.all(np.isfinite(z))
        or not np.all(np.isfinite(r))
        or (
            not reference_nonfinite_failure
            and not all(np.all(np.isfinite(value)) for value in (
                q, phi, reference_history,
            ))
        )
        or hash_arrays(z, r) != str(specification["coordinate_sha256"])
        or hash_arrays(z, r) != str(record["source_coordinate_sha256"])
        or hash_arrays(q, phi) != str(record["reference_state_sha256"])
    ):
        raise ValueError("construction failure payload hashes do not reproduce")
    diagnostics = record["solver_diagnostics"]
    required_diagnostics = {
        "reference_converged", "reference_max_abs_residual",
        "reference_residual_l2",
        "reference_iteration_cap", "reference_stencil_width",
    }
    if failure_gate.startswith("joint_hybrid_residual") or failure_gate == (
        "native_position_prerequisite"
    ):
        required_diagnostics |= {
            "selected_converged", "selected_maximum_residual",
            "selected_residual_l2",
            "selected_iteration_cap", "selected_stencil_width",
        }
    if failure_gate == "native_position_prerequisite":
        required_diagnostics.add("native_position_prerequisite")
    if not isinstance(diagnostics, Mapping) or set(diagnostics) != required_diagnostics:
        raise ValueError("construction failure solver diagnostics differ")
    reference_maximum = float(diagnostics["reference_max_abs_residual"])
    reference_l2 = float(diagnostics["reference_residual_l2"])
    if (
        type(diagnostics["reference_converged"]) is not bool
        or (
            reference_nonfinite_failure
            and np.isfinite(reference_maximum)
        )
        or (
            not reference_nonfinite_failure
            and (
                not np.isfinite(reference_maximum)
                or reference_maximum < 0.0
                or not np.isfinite(reference_l2)
                or reference_l2 < 0.0
            )
        )
        or int(diagnostics["reference_iteration_cap"])
        != int(specification["reference_iterations"])
        or int(diagnostics["reference_stencil_width"]) != STENCIL_WIDTH
    ):
        raise ValueError("finite-wall failure diagnostics are invalid")
    if failure_gate in (
        "finite_wall_reference", "finite_wall_reference_nonfinite",
    ):
        if _float64_hex(reference_maximum) != _float64_hex(measured):
            raise ValueError("finite-wall measurement differs from solver evidence")
    elif reference_maximum >= FINITE_WALL_REFERENCE_CEILING:
        raise ValueError("joint failure record did not pass the prior finite-wall gate")
    if failure_gate.startswith("joint_hybrid_residual") or failure_gate == (
        "native_position_prerequisite"
    ):
        selected_q = np.asarray(payload["selected_q"], dtype=float)
        selected_phi = np.asarray(payload["selected_phi"], dtype=float)
        selected_psi = np.asarray(payload["selected_psi"], dtype=float)
        selected_history = np.asarray(payload["selected_history"], dtype=float)
        damping_history = np.asarray(
            payload["selected_damping_history"], dtype=float,
        )
        selected_nonfinite_failure = failure_gate == (
            "joint_hybrid_residual_nonfinite"
        )
        selected_values = (
            selected_q, selected_phi, selected_psi,
            selected_history, damping_history,
        )
        if (
            any(value.shape != q.shape for value in (
                selected_q, selected_phi, selected_psi,
            ))
            or selected_history.ndim != 1
            or not 1 <= len(selected_history) <= 80
            or damping_history.ndim != 1
            or len(damping_history) > len(selected_history)
            or (
                not selected_nonfinite_failure
                and not all(np.all(np.isfinite(value)) for value in selected_values)
            )
            or (
                not selected_nonfinite_failure
                and np.any(z[:, None] + selected_q <= 0.0)
            )
            or (
                not selected_nonfinite_failure
                and not np.array_equal(
                    selected_psi, 1.0/(z[:, None] + selected_q),
                )
            )
            or type(diagnostics["selected_converged"]) is not bool
            or (
                selected_nonfinite_failure
                and np.isfinite(float(diagnostics["selected_maximum_residual"]))
            )
            or (
                not selected_nonfinite_failure
                and (
                    not np.isfinite(float(diagnostics["selected_maximum_residual"]))
                    or float(diagnostics["selected_maximum_residual"]) < 0.0
                    or not np.isfinite(float(diagnostics["selected_residual_l2"]))
                    or float(diagnostics["selected_residual_l2"]) < 0.0
                )
            )
            or int(diagnostics["selected_iteration_cap"]) != 80
            or int(diagnostics["selected_stencil_width"]) != STENCIL_WIDTH
            or (
                failure_gate == "native_position_prerequisite"
                and float(diagnostics["selected_maximum_residual"])
                >= JOINT_HYBRID_RESIDUAL_CEILING
            )
        ):
            raise ValueError("joint-hybrid failure payload is invalid")
        if _float64_hex(diagnostics["selected_maximum_residual"]) != (
            _float64_hex(measured)
        ) and failure_gate != "native_position_prerequisite":
            raise ValueError("joint measurement differs from solver evidence")
    if failure_gate == "native_position_prerequisite":
        raw_position = np.asarray(payload["native_raw_position"], dtype=float)
        native = diagnostics["native_position_prerequisite"]
        required_native = (
            "sphere_metric_normalized_Linf", "Phi_robin_Linf",
        )
        if (
            raw_position.shape != (len(z), len(r), 9)
            or not np.all(np.isfinite(raw_position))
            or not isinstance(native, Mapping)
            or tuple(native) != required_native
            or not all(
                np.isfinite(float(native[name])) and float(native[name]) >= 0.0
                for name in required_native
            )
            or max(float(native[name]) for name in required_native) != measured
        ):
            raise ValueError("native-position prerequisite evidence is invalid")
    attempt_payload = {
        "parent_label": label,
        "construction_input_fingerprint": str(
            record["construction_input_fingerprint"],
        ),
        "failure_gate": str(record["failure_gate"]),
        "scientific_payload": payload,
        "solver_diagnostics": diagnostics,
    }
    if str(record["parent_identity"]) != _hash_tree(attempt_payload):
        raise ValueError("construction failure parent attempt identity differs")
    payload_without_fingerprint = {
        name: record[name] for name in record if name != "fingerprint"
    }
    if str(record["fingerprint"]) != _hash_tree(payload_without_fingerprint):
        raise ValueError("construction failure fingerprint differs")
    return _freeze_tree(record)


def _construction_input_fingerprint(seed, label, specification):
    """Bind the immutable physical inputs and prospective solver settings."""
    return hash_arrays(
        np.asarray(seed["z"]),
        np.asarray(seed["r"]),
        np.asarray(seed["q"]),
        np.asarray(seed["phi"]),
        np.asarray(seed["coefficients"]),
        np.asarray(str(label)),
        np.asarray((
            AMPLITUDE,
            R_MAX,
            20.0,
            0.1,
            0.01,
            1e-10,
            float(specification["reference_iterations"]),
            float(STENCIL_WIDTH),
            1e-11,
            80.0,
            2.0**-20,
        )),
        np.asarray(str(specification["coordinate_sha256"])),
        np.asarray(str(seed["seed_sha256"])),
        np.asarray(str(seed["coefficient_sha256"])),
        np.asarray(str(SHAPE_NORMALIZATION_SHA256)),
    )


def load_frozen_common_seed():
    """Load exactly the sealed P10 q/Phi seed and immutable shape coefficients."""
    if sha256_file(SEALED_PARENT) != SEALED_PARENT_SHA256:
        raise RuntimeError("sealed Protocol-120 parent artifact hash mismatch")
    if sha256_file(KNOT_STATE) != KNOT_STATE_SHA256:
        raise RuntimeError("frozen family-knot artifact hash mismatch")
    with np.load(SEALED_PARENT, allow_pickle=False) as archive:
        z = np.asarray(archive["p10_z"], dtype=float).copy()
        r = np.asarray(archive["p10_r"], dtype=float).copy()
        q = 1.0/np.asarray(archive["p10_psi"], dtype=float)-z[:, None]
        phi = np.asarray(archive["p10_phi"], dtype=float).copy()
    if hash_arrays(z, r, q, phi) != COMMON_SEED_SHA256:
        raise RuntimeError("frozen P10 common-seed digest mismatch")
    with np.load(KNOT_STATE, allow_pickle=False) as archive:
        coefficients = np.asarray(archive["coefficients"], dtype=float).copy()
    if hash_arrays(coefficients) != COEFFICIENT_SHA256:
        raise RuntimeError("frozen shape-coefficient digest mismatch")
    return {
        "z": z,
        "r": r,
        "q": q,
        "phi": phi,
        "coefficients": coefficients,
        "seed_sha256": COMMON_SEED_SHA256,
        "coefficient_sha256": COEFFICIENT_SHA256,
    }


def construct_joint_parent_position(
    label,
    *,
    freeze_record=None,
    freeze_authority=None,
):
    """Construct one independent N0 or N1 position parent exactly once."""
    label = str(label)
    if label not in PARENT_SPECS:
        raise ValueError("joint parent label must be N0 or N1")
    if freeze_record is not None and freeze_authority is not None:
        raise Protocol125FreezeAuthorityError(
            "supply either a raw freeze record or its validated authority, not both"
        )
    if freeze_record is None and freeze_authority is None:
        raise Protocol125FreezeAuthorityError(
            "a verified Protocol-125 freeze record is required before parent construction"
        )
    if freeze_record is not None:
        freeze_authority = validate_protocol125_freeze_authority(freeze_record)
    else:
        freeze_authority = revalidate_protocol125_freeze_authority_snapshot(
            freeze_authority,
        )
    specification = PARENT_SPECS[label]
    seed = load_frozen_common_seed()
    if (
        str(seed.get("seed_sha256", "")) != COMMON_SEED_SHA256
        or str(seed.get("coefficient_sha256", "")) != COEFFICIENT_SHA256
    ):
        raise RuntimeError("joint-parent seed or branch identifier differs")
    input_fingerprint_before = _construction_input_fingerprint(
        seed, label, specification,
    )
    reference = solve_finite_wall_high_order_slice(
        AMPLITUDE,
        nz=specification["nz"],
        nr=specification["nr"],
        r_max=R_MAX,
        wall_stiffness=20.0,
        epsilon=0.1,
        backreaction=0.01,
        tolerance=1e-10,
        iterations=specification["reference_iterations"],
        stencil_width=STENCIL_WIDTH,
    )
    z = np.asarray(reference["z"], dtype=float)
    r = np.asarray(reference["r"], dtype=float)
    if hash_arrays(z, r) != specification["coordinate_sha256"]:
        raise RuntimeError("fresh parent coordinates differ from the frozen grid")
    reference_residual = float(reference["max_abs_residual"])
    if not np.isfinite(reference_residual):
        raise Protocol125ScientificConstructionFailure(
            _construction_failure_record(
                label,
                input_fingerprint_before,
                reference,
                failure_gate="finite_wall_reference_nonfinite",
                measured_value=reference_residual,
                ceiling=FINITE_WALL_REFERENCE_CEILING,
            ),
        )
    if reference_residual >= FINITE_WALL_REFERENCE_CEILING:
        raise Protocol125ScientificConstructionFailure(
            _construction_failure_record(
                label,
                input_fingerprint_before,
                reference,
                failure_gate="finite_wall_reference",
                measured_value=reference_residual,
                ceiling=FINITE_WALL_REFERENCE_CEILING,
            ),
        )
    a, b, c, a_r, b_r, c_r, shape_record = (
        frozen_shape_fields_with_radial_derivative(
            z, r, seed["coefficients"],
        )
    )
    if shape_record["sha256"] != SHAPE_NORMALIZATION_SHA256:
        raise RuntimeError("frozen shape normalization digest mismatch")
    chi, chi_r, chi_z = scalar_pulse(z, r, AMPLITUDE)
    initial_q = interpolate(seed["q"], seed["z"], seed["r"], z, r)
    initial_phi = interpolate(seed["phi"], seed["z"], seed["r"], z, r)
    selected = solve_joint_parent(
        z,
        r,
        reference["q"],
        reference["phi"],
        a,
        b,
        c,
        reference["background"],
        chi_r,
        chi_z,
        initial_q=initial_q,
        initial_phi=initial_phi,
        stencil_width=STENCIL_WIDTH,
        tolerance=1e-11,
        iterations=80,
    )
    joint_residual = float(selected["maximum_residual"])
    if not np.isfinite(joint_residual):
        raise Protocol125ScientificConstructionFailure(
            _construction_failure_record(
                label,
                input_fingerprint_before,
                reference,
                failure_gate="joint_hybrid_residual_nonfinite",
                measured_value=joint_residual,
                ceiling=JOINT_HYBRID_RESIDUAL_CEILING,
                selected=selected,
            ),
        )
    if joint_residual >= JOINT_HYBRID_RESIDUAL_CEILING:
        raise Protocol125ScientificConstructionFailure(
            _construction_failure_record(
                label,
                input_fingerprint_before,
                reference,
                failure_gate="joint_hybrid_residual",
                measured_value=joint_residual,
                ceiling=JOINT_HYBRID_RESIDUAL_CEILING,
                selected=selected,
            ),
        )
    psi = np.asarray(selected["psi"], dtype=float)
    raw_position = native_position_from_primitives(
        z,
        r,
        psi,
        psi,
        a,
        b,
        c,
        selected["phi"],
        chi,
    )
    try:
        completed_position, completion = complete_native_parent_position(
            raw_position,
            z,
            r,
            reference["background"],
            stencil_width=STENCIL_WIDTH,
            prerequisite_tolerance=NATIVE_POSITION_PREREQUISITE_CEILING,
        )
    except Protocol125NativePositionPrerequisiteFailure as error:
        if _construction_input_fingerprint(
            seed, label, specification,
        ) != input_fingerprint_before:
            raise RuntimeError(
                "immutable joint-parent construction inputs changed"
            ) from error
        raise Protocol125ScientificConstructionFailure(
            _construction_failure_record(
                label,
                input_fingerprint_before,
                reference,
                failure_gate="native_position_prerequisite",
                measured_value=max(error.measurements.values()),
                ceiling=NATIVE_POSITION_PREREQUISITE_CEILING,
                selected=selected,
                native_raw_position=raw_position,
                native_prerequisite=error.measurements,
            ),
        ) from error
    input_fingerprint_after = _construction_input_fingerprint(
        seed, label, specification,
    )
    if input_fingerprint_after != input_fingerprint_before:
        raise RuntimeError("immutable joint-parent construction inputs changed")
    parent_identity = hash_arrays(
        np.asarray(label),
        z,
        r,
        completed_position,
        np.asarray(selected["q"]),
        np.asarray(selected["phi"]),
        np.asarray(reference["q"]),
        np.asarray(reference["phi"]),
    )
    construction_provenance = build_protocol125_successful_parent_provenance_record(
        label,
        parent_identity,
        input_fingerprint_before,
        finite_wall_maximum_residual=reference_residual,
        joint_hybrid_maximum_residual=joint_residual,
    )
    return {
        "label": label,
        "z": z.copy(),
        "r": r.copy(),
        "position": completed_position,
        "raw_position": raw_position,
        "selector_q": np.asarray(selected["q"]).copy(),
        "psi_selector": psi.copy(),
        "shape_a": np.asarray(a).copy(),
        "shape_b": np.asarray(b).copy(),
        "shape_c": np.asarray(c).copy(),
        "shape_a_r": np.asarray(a_r).copy(),
        "shape_b_r": np.asarray(b_r).copy(),
        "shape_c_r": np.asarray(c_r).copy(),
        "phi": np.asarray(selected["phi"]).copy(),
        "chi": np.asarray(chi).copy(),
        "chi_r": np.asarray(chi_r).copy(),
        "chi_z": np.asarray(chi_z).copy(),
        "reference_q": np.asarray(reference["q"]).copy(),
        "reference_phi": np.asarray(reference["phi"]).copy(),
        "background": reference["background"],
        "reference_record": {
            "maximum_residual": float(reference["max_abs_residual"]),
            "iteration_cap": specification["reference_iterations"],
            "stencil_width": int(reference["stencil_width"]),
        },
        "joint_record": {
            key: selected[key]
            for key in (
                "converged", "maximum_residual", "residual_l2", "history",
                "damping_history",
            )
        },
        "completion_record": completion,
        "common_seed_sha256": seed["seed_sha256"],
        "coefficient_sha256": seed["coefficient_sha256"],
        "shape_normalization_sha256": shape_record["sha256"],
        "independent_of_other_parent": True,
        "freeze_authority": freeze_authority,
        "parent_identity": parent_identity,
        "parent_identity_binding_sha256": construction_provenance[
            "parent_identity_binding_sha256"
        ],
        "construction_provenance_record": construction_provenance,
    }
