from __future__ import annotations

import numpy as np
import pytest

import bhps.joint_parent_construction as construction
from bhps.joint_parent_construction import (
    COMMON_SEED_SHA256,
    COEFFICIENT_SHA256,
    FINITE_WALL_REFERENCE_CEILING,
    JOINT_HYBRID_RESIDUAL_CEILING,
    PARENT_SPECS,
    Protocol125ScientificConstructionFailure,
    SUCCESSFUL_CONSTRUCTION_PROTOCOL_IDENTIFIER,
    SUCCESSFUL_CONSTRUCTION_PROVENANCE_KEYS,
    build_protocol125_successful_parent_provenance_record,
    construct_joint_parent_position,
    load_frozen_common_seed,
    validate_protocol125_construction_failure_record,
    validate_protocol125_successful_parent_provenance_record,
)
from bhps.matched_staged_continuum import hash_arrays
from bhps.joint_parent_shape import SHAPE_NORMALIZATION_SHA256
from bhps.joint_parent_native_completion import (
    Protocol125NativePositionPrerequisiteFailure,
)


def test_frozen_common_seed_and_shape_inputs_reproduce_prospective_hashes():
    found = load_frozen_common_seed()
    assert found["seed_sha256"] == COMMON_SEED_SHA256
    assert found["coefficient_sha256"] == COEFFICIENT_SHA256
    assert hash_arrays(
        found["z"], found["r"], found["q"], found["phi"],
    ) == COMMON_SEED_SHA256
    assert hash_arrays(found["coefficients"]) == COEFFICIENT_SHA256
    assert found["q"].shape == found["phi"].shape
    assert found["coefficients"].shape == (80,)


def test_parent_specs_are_independent_and_frozen():
    assert PARENT_SPECS["N0"]["nz"] == 145
    assert PARENT_SPECS["N0"]["nr"] == 325
    assert PARENT_SPECS["N0"]["reference_iterations"] == 450
    assert PARENT_SPECS["N1"]["nz"] == 161
    assert PARENT_SPECS["N1"]["nr"] == 361
    assert PARENT_SPECS["N1"]["reference_iterations"] == 500
    assert (
        PARENT_SPECS["N0"]["coordinate_sha256"]
        != PARENT_SPECS["N1"]["coordinate_sha256"]
    )


def test_joint_parent_reference_call_is_bound_to_seven_point_operators():
    import ast
    from pathlib import Path

    tree = ast.parse(Path(
        "src/bhps/joint_parent_construction.py"
    ).read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "solve_finite_wall_high_order_slice"
    ]
    assert len(calls) == 1
    keywords = {item.arg: item.value for item in calls[0].keywords}
    assert isinstance(keywords["stencil_width"], ast.Name)
    assert keywords["stencil_width"].id == "STENCIL_WIDTH"


def test_unknown_parent_label_is_not_a_hidden_third_remediation():
    from bhps.joint_parent_construction import construct_joint_parent_position

    with pytest.raises(ValueError, match="N0 or N1"):
        construct_joint_parent_position("N2")


def test_current_draft_cannot_enter_n0_or_n1_construction():
    from bhps.joint_parent_construction import construct_joint_parent_position
    from bhps.joint_parent_freeze_authority import Protocol125FreezeAuthorityError

    for label in ("N0", "N1"):
        with pytest.raises(Protocol125FreezeAuthorityError, match="freeze record"):
            construct_joint_parent_position(label)


def test_joint_construction_uses_one_frozen_shape_normalization():
    import ast
    from pathlib import Path

    tree = ast.parse(Path(
        "src/bhps/joint_parent_construction.py"
    ).read_text())
    called = {
        getattr(node.func, "id", "")
        for node in ast.walk(tree) if isinstance(node, ast.Call)
    }
    assert "frozen_shape_fields_with_radial_derivative" in called
    assert "shape_fields" not in called
    assert SHAPE_NORMALIZATION_SHA256 != "TO_BE_FROZEN"


def _manufactured_construction_attempt(monkeypatch):
    z = np.linspace(1.0, np.e, 3)
    r = np.linspace(0.0, 12.0, 4)
    shape = (len(z), len(r))
    zeros = np.zeros(shape)
    specification = {
        "nz": len(z),
        "nr": len(r),
        "reference_iterations": 5,
        "coordinate_sha256": hash_arrays(z, r),
    }
    monkeypatch.setitem(PARENT_SPECS, "N0", specification)
    monkeypatch.setattr(
        construction,
        "revalidate_protocol125_freeze_authority_snapshot",
        lambda authority: authority,
    )
    monkeypatch.setattr(
        construction,
        "load_frozen_common_seed",
        lambda: {
            "z": z.copy(),
            "r": r.copy(),
            "q": zeros.copy(),
            "phi": zeros.copy(),
            "coefficients": np.zeros(1),
            "seed_sha256": COMMON_SEED_SHA256,
            "coefficient_sha256": COEFFICIENT_SHA256,
        },
    )
    monkeypatch.setattr(
        construction,
        "_construction_input_fingerprint",
        lambda *args: "a"*64,
    )
    reference = {
        "z": z.copy(),
        "r": r.copy(),
        "q": zeros.copy(),
        "phi": zeros.copy(),
        "background": {},
        "converged": True,
        "max_abs_residual": 1e-12,
        "residual_l2": 5e-13,
        "history": [1e-3, 1e-6, 1e-12],
        "stencil_width": 7,
    }
    return z, r, zeros, reference


def test_finite_wall_threshold_failure_is_typed_immutable_and_validated(
    monkeypatch,
):
    _, _, _, reference = _manufactured_construction_attempt(monkeypatch)
    reference["max_abs_residual"] = 2.0*FINITE_WALL_REFERENCE_CEILING
    reference["residual_l2"] = FINITE_WALL_REFERENCE_CEILING
    # The solver's own flag is diagnostic only; the frozen residual gate is
    # the scientific classifier.
    reference["converged"] = True
    monkeypatch.setattr(
        construction,
        "solve_finite_wall_high_order_slice",
        lambda *args, **kwargs: reference,
    )
    with pytest.raises(Protocol125ScientificConstructionFailure) as captured:
        construct_joint_parent_position(
            "N0", freeze_authority={"manufactured": True},
        )
    record = validate_protocol125_construction_failure_record(
        captured.value.record,
    )
    assert record["classification"] == "FAIL-parent-bulk"
    assert record["failure_gate"] == "finite_wall_reference"
    assert record["strict_ceiling"] == FINITE_WALL_REFERENCE_CEILING
    assert record["measured_value"] >= record["strict_ceiling"]
    assert record["complete"] is True
    assert record["provenance_valid"] is True
    assert record["passed"] is False
    assert record["acceleration_authorized"] is False
    assert record["retry_authorized"] is False
    assert record["candidate_or_phase_a_executed"] is False
    with pytest.raises((TypeError, ValueError)):
        record["scientific_payload"]["reference_q"][0, 0] = 1.0

    tampered = dict(record)
    tampered["measured_value"] = 0.5*FINITE_WALL_REFERENCE_CEILING
    with pytest.raises(ValueError, match="does not fail"):
        validate_protocol125_construction_failure_record(tampered)

    detached = dict(record)
    detached["solver_diagnostics"] = dict(detached["solver_diagnostics"])
    detached["solver_diagnostics"]["reference_max_abs_residual"] = (
        3.0*FINITE_WALL_REFERENCE_CEILING
    )
    with pytest.raises(ValueError, match="differs from solver evidence"):
        validate_protocol125_construction_failure_record(detached)


def test_joint_hybrid_threshold_failure_captures_selected_attempt(monkeypatch):
    z, r, zeros, reference = _manufactured_construction_attempt(monkeypatch)
    shape = zeros.shape
    monkeypatch.setattr(
        construction,
        "solve_finite_wall_high_order_slice",
        lambda *args, **kwargs: reference,
    )
    monkeypatch.setattr(
        construction,
        "frozen_shape_fields_with_radial_derivative",
        lambda *args, **kwargs: (
            *(zeros.copy() for _ in range(6)),
            {"sha256": SHAPE_NORMALIZATION_SHA256},
        ),
    )
    monkeypatch.setattr(
        construction,
        "scalar_pulse",
        lambda *args, **kwargs: (
            zeros.copy(), zeros.copy(), zeros.copy(),
        ),
    )
    monkeypatch.setattr(
        construction,
        "interpolate",
        lambda values, *args, **kwargs: np.asarray(values).copy(),
    )
    selected_q = zeros.copy()
    selected = {
        "q": selected_q,
        "phi": zeros.copy(),
        "psi": np.broadcast_to(1.0/z[:, None], shape).copy(),
        "converged": True,
        "maximum_residual": 2.0*JOINT_HYBRID_RESIDUAL_CEILING,
        "residual_l2": JOINT_HYBRID_RESIDUAL_CEILING,
        "history": [1e-4, 2.0*JOINT_HYBRID_RESIDUAL_CEILING],
        "damping_history": [1.0],
    }
    monkeypatch.setattr(
        construction,
        "solve_joint_parent",
        lambda *args, **kwargs: selected,
    )
    with pytest.raises(Protocol125ScientificConstructionFailure) as captured:
        construct_joint_parent_position(
            "N0", freeze_authority={"manufactured": True},
        )
    record = captured.value.record
    assert record["failure_gate"] == "joint_hybrid_residual"
    assert record["strict_ceiling"] == JOINT_HYBRID_RESIDUAL_CEILING
    assert set(record["scientific_payload"]) == {
        "z", "r", "reference_q", "reference_phi", "reference_history",
        "selected_q", "selected_phi", "selected_psi", "selected_history",
        "selected_damping_history",
    }
    assert validate_protocol125_construction_failure_record(record)[
        "fingerprint"
    ] == record["fingerprint"]


def test_nonfinite_construction_residual_is_typed_scientific_failure(monkeypatch):
    _, _, _, reference = _manufactured_construction_attempt(monkeypatch)
    reference["max_abs_residual"] = np.nan
    monkeypatch.setattr(
        construction,
        "solve_finite_wall_high_order_slice",
        lambda *args, **kwargs: reference,
    )
    with pytest.raises(Protocol125ScientificConstructionFailure) as captured:
        construct_joint_parent_position(
            "N0", freeze_authority={"manufactured": True},
        )
    record = captured.value.record
    assert record["classification"] == "FAIL-parent-bulk"
    assert record["failure_gate"] == "finite_wall_reference_nonfinite"
    assert record["measurement_finite"] is False
    assert record["measured_value_ieee754_hex"] == "7ff8000000000000"
    assert np.isnan(record["measured_value"])
    validate_protocol125_construction_failure_record(record)


def test_nonfinite_joint_residual_record_preserves_exact_nonfinite_evidence(
    monkeypatch,
):
    z, _, zeros, reference = _manufactured_construction_attempt(monkeypatch)
    selected = {
        "q": zeros.copy(),
        "phi": zeros.copy(),
        "psi": np.broadcast_to(1.0/z[:, None], zeros.shape).copy(),
        "converged": False,
        "maximum_residual": np.inf,
        "residual_l2": np.inf,
        "history": [1e-4, np.inf],
        "damping_history": [1.0],
    }
    record = construction._construction_failure_record(
        "N0",
        "a"*64,
        reference,
        failure_gate="joint_hybrid_residual_nonfinite",
        measured_value=np.inf,
        ceiling=JOINT_HYBRID_RESIDUAL_CEILING,
        selected=selected,
    )
    validated = validate_protocol125_construction_failure_record(record)
    assert validated["measurement_finite"] is False
    assert validated["measured_value_ieee754_hex"] == "7ff0000000000000"
    assert np.isposinf(validated["measured_value"])


def test_native_sphere_phi_prerequisite_is_typed_parent_position_failure(
    monkeypatch,
):
    z, r, zeros, reference = _manufactured_construction_attempt(monkeypatch)
    shape = zeros.shape
    monkeypatch.setattr(
        construction,
        "solve_finite_wall_high_order_slice",
        lambda *args, **kwargs: reference,
    )
    monkeypatch.setattr(
        construction,
        "frozen_shape_fields_with_radial_derivative",
        lambda *args, **kwargs: (
            *(zeros.copy() for _ in range(6)),
            {"sha256": SHAPE_NORMALIZATION_SHA256},
        ),
    )
    monkeypatch.setattr(
        construction,
        "scalar_pulse",
        lambda *args, **kwargs: (
            zeros.copy(), zeros.copy(), zeros.copy(),
        ),
    )
    monkeypatch.setattr(
        construction,
        "interpolate",
        lambda values, *args, **kwargs: np.asarray(values).copy(),
    )
    selected = {
        "q": zeros.copy(),
        "phi": zeros.copy(),
        "psi": np.broadcast_to(1.0/z[:, None], shape).copy(),
        "converged": True,
        "maximum_residual": 2e-12,
        "residual_l2": 1e-12,
        "history": [1e-4, 2e-12],
        "damping_history": [1.0],
    }
    monkeypatch.setattr(
        construction,
        "solve_joint_parent",
        lambda *args, **kwargs: selected,
    )
    raw = np.zeros((*shape, 9))
    monkeypatch.setattr(
        construction,
        "native_position_from_primitives",
        lambda *args, **kwargs: raw.copy(),
    )

    def fail_native(*args, **kwargs):
        raise Protocol125NativePositionPrerequisiteFailure({
            "sphere_metric_normalized_Linf": 3e-9,
            "Phi_robin_Linf": 2e-11,
        }, 1e-10)

    monkeypatch.setattr(
        construction, "complete_native_parent_position", fail_native,
    )
    with pytest.raises(Protocol125ScientificConstructionFailure) as captured:
        construct_joint_parent_position(
            "N0", freeze_authority={"manufactured": True},
        )
    record = captured.value.record
    assert record["classification"] == "FAIL-parent-position"
    assert record["failure_gate"] == "native_position_prerequisite"
    assert record["measured_value"] == 3e-9
    assert record["solver_diagnostics"][
        "native_position_prerequisite"
    ]["sphere_metric_normalized_Linf"] == 3e-9
    validate_protocol125_construction_failure_record(record)


def test_successful_provenance_binds_exact_frozen_identity_inputs():
    identity = "d"*64
    record = build_protocol125_successful_parent_provenance_record(
        "N0",
        identity,
        "a"*64,
        finite_wall_maximum_residual=1e-12,
        joint_hybrid_maximum_residual=2e-12,
    )
    assert tuple(record) == SUCCESSFUL_CONSTRUCTION_PROVENANCE_KEYS
    assert (
        record["protocol_identifier"]
        == SUCCESSFUL_CONSTRUCTION_PROTOCOL_IDENTIFIER
    )
    assert record["common_seed_sha256"] == COMMON_SEED_SHA256
    assert record["physical_normalization_identifier"] == (
        SHAPE_NORMALIZATION_SHA256
    )
    assert record["branch_identifier"] == COEFFICIENT_SHA256
    assert record["independent_of_other_parent"] is True
    assert validate_protocol125_successful_parent_provenance_record(
        record,
        expected_parent_label="N0",
        expected_parent_identity=identity,
    )["fingerprint"] == record["fingerprint"]

    adverse_values = {
        "common_seed_sha256": "1"*64,
        "physical_normalization_identifier": "2"*64,
        "branch_identifier": "3"*64,
        "independent_of_other_parent": False,
        "actual_parent_label": "N1",
    }
    for field, value in adverse_values.items():
        adverse = dict(record)
        adverse[field] = value
        with pytest.raises(ValueError):
            validate_protocol125_successful_parent_provenance_record(
                adverse,
                expected_parent_label="N0",
                expected_parent_identity=identity,
            )

    replayed_input = dict(record)
    for field in (
        "input_fingerprint_before",
        "input_fingerprint_after",
        "construction_input_fingerprint",
    ):
        replayed_input[field] = "e"*64
    with pytest.raises(ValueError, match="identity binding"):
        validate_protocol125_successful_parent_provenance_record(
            replayed_input,
            expected_parent_label="N0",
            expected_parent_identity=identity,
        )


def test_successful_construction_attaches_validated_identity_binding(monkeypatch):
    z, r, zeros, reference = _manufactured_construction_attempt(monkeypatch)
    shape = zeros.shape
    monkeypatch.setattr(
        construction,
        "solve_finite_wall_high_order_slice",
        lambda *args, **kwargs: reference,
    )
    monkeypatch.setattr(
        construction,
        "frozen_shape_fields_with_radial_derivative",
        lambda *args, **kwargs: (
            *(zeros.copy() for _ in range(6)),
            {"sha256": SHAPE_NORMALIZATION_SHA256},
        ),
    )
    monkeypatch.setattr(
        construction,
        "scalar_pulse",
        lambda *args, **kwargs: (
            zeros.copy(), zeros.copy(), zeros.copy(),
        ),
    )
    monkeypatch.setattr(
        construction,
        "interpolate",
        lambda values, *args, **kwargs: np.asarray(values).copy(),
    )
    selected = {
        "q": zeros.copy(),
        "phi": zeros.copy(),
        "psi": np.broadcast_to(1.0/z[:, None], shape).copy(),
        "converged": True,
        "maximum_residual": 2e-12,
        "residual_l2": 1e-12,
        "history": [1e-4, 2e-12],
        "damping_history": [1.0],
    }
    monkeypatch.setattr(
        construction,
        "solve_joint_parent",
        lambda *args, **kwargs: selected,
    )
    raw = np.zeros((*shape, 9))
    monkeypatch.setattr(
        construction,
        "native_position_from_primitives",
        lambda *args, **kwargs: raw.copy(),
    )
    monkeypatch.setattr(
        construction,
        "complete_native_parent_position",
        lambda *args, **kwargs: (raw.copy(), {"manufactured": True}),
    )
    parent = construct_joint_parent_position(
        "N0", freeze_authority={"manufactured": True},
    )
    expected_state_identity = hash_arrays(
        np.asarray("N0"),
        z,
        r,
        raw,
        selected["q"],
        selected["phi"],
        reference["q"],
        reference["phi"],
    )
    assert parent["parent_identity"] == expected_state_identity
    assert parent["parent_identity_binding_sha256"] == parent[
        "construction_provenance_record"
    ]["parent_identity_binding_sha256"]
    validate_protocol125_successful_parent_provenance_record(
        parent["construction_provenance_record"],
        expected_parent_label="N0",
        expected_parent_identity=expected_state_identity,
    )
