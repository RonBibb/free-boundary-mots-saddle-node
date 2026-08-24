from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import bhps.joint_parent_construction as construction
import bhps.joint_parent_preacceleration as preacceleration
import bhps.joint_parent_representation as representation
from bhps.joint_parent_bulk_reference import FiniteWallReferenceHermitePair
from bhps.joint_parent_bulk_validation import (
    ALL_LANES as BULK_LANES,
    PROTOCOL_IDENTIFIER as BULK_PROTOCOL_IDENTIFIER,
)
from bhps.joint_parent_gate_ledger import Protocol125GateLedger
from bhps.joint_parent_position_pair import PositionOnlyConstrainedHermitePair
from bhps.joint_parent_position_state import (
    build_joint_parent_position_state,
    derive_joint_parent_position_outer_contract,
)
from bhps.joint_parent_ordered_adjudicator import (
    adjudicate_protocol125_ordered,
)
from bhps.joint_parent_preacceleration import (
    NATIVE_EVIDENCE_PROTOCOL_IDENTIFIER,
    NATIVE_POSITION_TANGENT_LANES,
    PRE_ACCELERATION_GROUPS,
    Protocol125PreAccelerationCoreInputs,
    Protocol125PreAccelerationInputs,
    Protocol125PreAccelerationRepresentationInputs,
    capture_protocol125_bulk_prerequisite_provenance,
    capture_protocol125_legacy_sampling_provenance,
    capture_protocol125_position_prefix_provenance,
    capture_protocol125_preacceleration_provenance,
    compose_protocol125_construction_failure_records,
    compose_protocol125_representation_coefficient_failure_records,
    evaluate_protocol125_position_prefix,
    evaluate_protocol125_preacceleration,
    extend_protocol125_legacy_sampling,
    finalize_protocol125_preacceleration_stop,
    finish_protocol125_bulk_prerequisite,
)
from bhps.joint_parent_refinement_diagnostics import frozen_validation_meshes
from bhps.matched_staged_continuum import hash_arrays


def _manufactured_inputs():
    z = np.linspace(1.0, np.e, 145)
    r = np.linspace(0.0, 12.0, 17)
    shape = (len(z), len(r))
    zeros = np.zeros(shape)
    position = np.zeros((*shape, 9))
    position[:, :, 2] = -1.0
    position[:, :, 3] = 1.0
    position[:, :, 6] = 1.0
    compact_phase = 12.0*np.pi*(z-1.0)/(np.e-1.0)
    chi = np.broadcast_to(
        (1e-6*np.cos(compact_phase))[:, None], shape,
    ).copy()
    position[:, :, 8] = chi
    selector_q = np.broadcast_to((1.0-z)[:, None], shape).copy()
    parent_identity = hash_arrays(
        np.asarray("N0"),
        z,
        r,
        position,
        selector_q,
        zeros,
        selector_q,
        zeros,
    )
    parent = {
        "label": "N0",
        "parent_identity": parent_identity,
        "z": z,
        "r": r,
        "position": position,
        "selector_q": selector_q,
        "phi": zeros.copy(),
        "reference_q": selector_q.copy(),
        "reference_phi": zeros.copy(),
        "psi_selector": np.ones(shape),
        "chi": chi.copy(),
        "chi_r": zeros.copy(),
        **{
            f"shape_{name}": zeros.copy()
            for name in ("a", "b", "c", "a_r", "b_r", "c_r")
        },
    }
    background = {
        "wall_stiffness": 0.0,
        "v0": 0.0,
        "v1": 0.0,
        "beta_a": 0.0,
        "beta_b": 0.0,
        "wall_potential_a": 0.0,
        "wall_potential_b": 0.0,
    }
    outer = derive_joint_parent_position_outer_contract(parent)
    state, _ = build_joint_parent_position_state(
        position,
        z,
        r,
        background,
        outer_open_face_contract=outer,
        parent_r_max=12.0,
    )
    pair = PositionOnlyConstrainedHermitePair.from_primary(state)
    reference = FiniteWallReferenceHermitePair.build(
        z, r, selector_q, zeros,
    )
    construction = {
        "finite_wall_maximum_residual": 1e-12,
        "joint_hybrid_maximum_residual": 2e-12,
        "input_fingerprint_before": "immutable-manufactured-input",
        "input_fingerprint_after": "immutable-manufactured-input",
        "physical_normalization_identifier": "manufactured-flat-normalization",
        "branch_identifier": "manufactured-flat-branch",
        "expected_parent_label": "N0",
        "actual_parent_label": "N0",
        "parent_identity": parent_identity,
    }
    native_lanes = {
        name: {
            "complete": True,
            "provenance_valid": True,
            "passed": True,
            "fingerprint": f"{index:x}"*64,
        }
        for index, name in enumerate(NATIVE_POSITION_TANGENT_LANES, 1)
    }
    native = {
        "protocol_identifier": NATIVE_EVIDENCE_PROTOCOL_IDENTIFIER,
        "parent_label": "N0",
        "parent_identity": parent_identity,
        "source_coordinate_sha256": hash_arrays(z, r),
        "position_sha256": hash_arrays(position),
        "lanes": native_lanes,
    }
    lane_gates = {name: True for name in BULK_LANES}
    bulk = {
        "protocol": BULK_PROTOCOL_IDENTIFIER,
        "parent_label": "N0",
        "identity": {
            "parent_label": "N0",
            "binding_sha256": "a"*64,
            "source_coordinate_sha256": hash_arrays(z, r),
            "candidate_source_fingerprint": pair.source_fingerprint,
            "candidate_endpoint_fingerprint": pair.endpoint_fingerprint,
            "candidate_pair_fingerprint": pair.fingerprint(),
            "reference_pair_fingerprint": reference.fingerprint(),
        },
        "lanes": {
            name: {"authoritative": {"manufactured": True}}
            for name in BULK_LANES
        },
        "adjudication": {
            "lane_numerical_gates": lane_gates,
            "all_lane_numerical_gates_pass": True,
            "all_Q53_Q33_bulk_jet_gates_pass": True,
            "strip_layer_growth_gate": {"pass": True},
            "parent_bulk_pass": True,
            "common_V2_two_parent_gate_required": True,
            "protocol_two_parent_bulk_pass": False,
            "fail_closed_pending_second_parent": True,
        },
        "scientific_artifact_written": False,
        "acceleration_authorized": False,
    }
    frozen = frozen_validation_meshes()
    legacy_q33 = {}
    legacy_q55 = {}
    for name in ("V0", "V1", "V2"):
        leading = (len(frozen[name]["z"]), len(frozen[name]["r"]))
        legacy_q33[name] = {
            "position": np.zeros((*leading, 1)),
            "first_spatial": np.zeros((*leading, 1, 2)),
            "second_spatial": np.zeros((*leading, 1, 3)),
        }
        legacy_q55[name] = {
            group: values.copy() for group, values in legacy_q33[name].items()
        }
    component_orders = {
        "position": ("legacy_q0",),
        "first_spatial": ("legacy_q0:z", "legacy_q0:r"),
        "second_spatial": (
            "legacy_q0:zz", "legacy_q0:zr", "legacy_q0:rr",
        ),
    }
    return Protocol125PreAccelerationInputs(
        parent_mapping=parent,
        position_pair=pair,
        reference_pair=reference,
        construction_provenance=construction,
        native_position_tangent_evidence=native,
        bulk_validation_audit=bulk,
        legacy_Q33_by_mesh=legacy_q33,
        legacy_Q55_by_mesh=legacy_q55,
        legacy_component_orders=component_orders,
    )


def _freeze_record():
    return {
        "status": "FROZEN",
        "protocol_sha256": "c"*64,
        "adjudicator_sha256": "d"*64,
        "frozen_before_parent_data": True,
        "independent_review_passed": True,
        "scientific_candidates_absent_at_freeze": True,
    }


def _staged_input_views(inputs):
    core = Protocol125PreAccelerationCoreInputs(
        parent_mapping=inputs.parent_mapping,
        position_pair=inputs.position_pair,
        reference_pair=inputs.reference_pair,
        construction_provenance=inputs.construction_provenance,
        native_position_tangent_evidence=(
            inputs.native_position_tangent_evidence
        ),
    )
    representation = Protocol125PreAccelerationRepresentationInputs(
        parent_mapping=inputs.parent_mapping,
        position_pair=inputs.position_pair,
        reference_pair=inputs.reference_pair,
        construction_provenance=inputs.construction_provenance,
        native_position_tangent_evidence=(
            inputs.native_position_tangent_evidence
        ),
        legacy_Q33_by_mesh=inputs.legacy_Q33_by_mesh,
        legacy_Q55_by_mesh=inputs.legacy_Q55_by_mesh,
        legacy_component_orders=inputs.legacy_component_orders,
    )
    return core, representation


def test_staged_preacceleration_all_passes_before_bulk_in_frozen_order():
    inputs = _manufactured_inputs()
    core, representation = _staged_input_views(inputs)
    position_provenance = capture_protocol125_position_prefix_provenance(core)
    position = evaluate_protocol125_position_prefix(core, position_provenance)
    assert position["passed"] is True
    assert tuple(position["groups"]) == PRE_ACCELERATION_GROUPS[:5]

    representation_provenance = (
        capture_protocol125_legacy_sampling_provenance(
            representation, position,
        )
    )
    represented = extend_protocol125_legacy_sampling(
        representation, position, representation_provenance,
    )
    assert represented["passed"] is True
    assert tuple(represented["groups"]) == PRE_ACCELERATION_GROUPS[:7]

    bulk_provenance = capture_protocol125_bulk_prerequisite_provenance(
        inputs, represented,
    )
    result = finish_protocol125_bulk_prerequisite(
        inputs, represented, bulk_provenance,
    )
    assert result["passed"] is True
    assert tuple(result["groups"]) == PRE_ACCELERATION_GROUPS
    assert PRE_ACCELERATION_GROUPS[-1] == "bulk_prerequisite"


@pytest.mark.parametrize("failed_index", range(7))
def test_staged_preacceleration_stops_before_bulk_on_each_position_gate(
    monkeypatch, failed_index,
):
    inputs = _manufactured_inputs()
    core, representation = _staged_input_views(inputs)
    calls = []
    scorers = {}
    for name, scorer in preacceleration._SCORERS.items():
        def wrapped(bundle, context, *, _name=name, _scorer=scorer):
            calls.append(_name)
            if _name == PRE_ACCELERATION_GROUPS[failed_index]:
                return preacceleration._gate_record(
                    _name,
                    context,
                    complete=True,
                    provenance_valid=True,
                    passed=False,
                    details={"manufactured_failure": _name},
                )
            return _scorer(bundle, context)
        scorers[name] = wrapped
    monkeypatch.setattr(
        preacceleration, "_SCORERS", preacceleration.MappingProxyType(scorers),
    )

    position_provenance = capture_protocol125_position_prefix_provenance(core)
    position = evaluate_protocol125_position_prefix(core, position_provenance)
    if not position["passed"]:
        stopped = finalize_protocol125_preacceleration_stop(position)
    else:
        representation_provenance = (
            capture_protocol125_legacy_sampling_provenance(
                representation, position,
            )
        )
        represented = extend_protocol125_legacy_sampling(
            representation, position, representation_provenance,
        )
        assert represented["passed"] is False
        stopped = finalize_protocol125_preacceleration_stop(represented)
    failed = PRE_ACCELERATION_GROUPS[failed_index]
    assert calls == list(PRE_ACCELERATION_GROUPS[:failed_index+1])
    assert "bulk_prerequisite" not in calls
    assert stopped["groups"][failed]["passed"] is False
    for name in PRE_ACCELERATION_GROUPS[failed_index+1:]:
        assert stopped["groups"][name]["not_reached"] is True
        assert stopped["groups"][name]["blocked_by"] == failed


def test_staged_preacceleration_rejects_tampered_prefix_before_later_work():
    inputs = _manufactured_inputs()
    core, representation = _staged_input_views(inputs)
    provenance = capture_protocol125_position_prefix_provenance(core)
    position = evaluate_protocol125_position_prefix(core, provenance)
    tampered = dict(position)
    tampered["fingerprint"] = "0"*64
    with pytest.raises(ValueError, match="fingerprint"):
        capture_protocol125_legacy_sampling_provenance(
            representation, tampered,
        )


def test_nonfinite_representation_coefficients_are_bound_ordered_bulk_failure():
    inputs = _manufactured_inputs()
    parent = dict(inputs.parent_mapping)
    construction_record = (
        construction.build_protocol125_successful_parent_provenance_record(
            "N0",
            parent["parent_identity"],
            "a"*64,
            finite_wall_maximum_residual=1e-12,
            joint_hybrid_maximum_residual=2e-12,
        )
    )
    parent["construction_provenance_record"] = construction_record
    with pytest.raises(
        representation.Protocol125RepresentationCoefficientFailure,
    ) as captured:
        representation.raise_if_nonfinite_protocol125_representation_coefficients(
            np.asarray([[[np.nan]]], dtype=float),
            recipe="native-radial-cubic-s",
            input_arrays={"manufactured_finite_input": np.asarray([1.0])},
        )
    evidence = representation.bind_protocol125_representation_coefficient_failure(
        captured.value.evidence,
        parent["parent_identity"],
    )
    result = compose_protocol125_representation_coefficient_failure_records(
        parent,
        construction_record,
        evidence,
    )
    assert result["classification"] == "FAIL-single-parent-pre-acceleration"
    assert result["complete"] is True
    assert result["provenance_valid"] is True
    assert result["passed"] is False
    first = result["groups"]["pre_acceleration_construction"]
    assert first["passed"] is False
    assert first["details"]["failure_classification"] == "FAIL-parent-bulk"
    assert first["details"]["failure_gate"] == (
        "persisted_representation_coefficients_finite"
    )
    assert first["details"]["nonfinite_count"] == 1
    for name in PRE_ACCELERATION_GROUPS[1:]:
        stopped = result["groups"][name]
        assert stopped["not_reached"] is True
        assert stopped["blocked_by"] == "pre_acceleration_construction"

    n1_identity = "f"*64
    n1_context = {"label": "N1", "identity": n1_identity}
    n1_groups = {
        name: preacceleration._gate_record(
            name,
            n1_context,
            complete=True,
            provenance_valid=True,
            passed=True,
            details={"manufactured": name},
        )
        for name in PRE_ACCELERATION_GROUPS
    }
    adjudicated = adjudicate_protocol125_ordered(
        {"N0": result["groups"], "N1": n1_groups},
        parent_identities={
            "N0": parent["parent_identity"],
            "N1": n1_identity,
        },
        protocol_freeze_record=_freeze_record(),
    )
    assert adjudicated["classification"] == "FAIL-parent-bulk"
    assert adjudicated["failed_bulk_groups"] == (
        "N0:pre_acceleration_construction",
    )
    assert adjudicated["invalid_reasons"] == ()

    with pytest.raises(ValueError, match="parent binding"):
        compose_protocol125_representation_coefficient_failure_records(
            parent,
            construction_record,
            captured.value.evidence,
        )
    with pytest.raises(ValueError, match="another parent"):
        representation.bind_protocol125_representation_coefficient_failure(
            evidence,
            n1_identity,
        )


def test_preacceleration_composes_all_ledger_ready_groups_without_acceleration():
    inputs = _manufactured_inputs()
    provenance = capture_protocol125_preacceleration_provenance(inputs)
    result = evaluate_protocol125_preacceleration(inputs, provenance)
    assert result["classification"] == "PASS-single-parent-pre-acceleration"
    assert result["complete"]
    assert result["provenance_valid"]
    assert result["passed"]
    assert tuple(result["groups"]) == PRE_ACCELERATION_GROUPS
    for name, record in result["groups"].items():
        assert record["complete"] is True
        assert record["provenance_valid"] is True
        assert record["passed"] is True
        assert len(record["fingerprint"]) == 64
        assert record["parent_label"] == "N0"
        assert record["parent_identity"] == inputs.parent_mapping["parent_identity"]
        assert record["group_name"] == name
    assert not result["acceleration_evaluated"]
    assert not result["acceleration_authorized"]
    assert not result["scientific_execution_authorized"]
    assert not result["artifact_written"]
    representation = result["groups"]["position_representation"]["details"]
    assert representation[
        "q4_q5_physical_derivative_images_gated_pre_acceleration"
    ] is True
    assert representation["Q53_Q33_position_q4_q5_images"]["passed"] is True
    for degree in ("Q53", "Q33"):
        source = representation["source_node_reproduction"][degree]
        assert source["comparison_group"] == "physical_coordinate_components"
        assert source["reduced_coefficients_diagnostic"]["gated"] is False
        assert len(source["constituents"]["records"]) == 9

    identities = {
        "N0": inputs.parent_mapping["parent_identity"],
        "N1": "f"*64,
    }
    ledger = Protocol125GateLedger(_freeze_record(), identities)
    for name in PRE_ACCELERATION_GROUPS:
        ledger = ledger.append_parent_gate("N0", name, result["groups"][name])
    assert tuple(ledger.parent_records["N0"]) == PRE_ACCELERATION_GROUPS


def test_missing_native_lane_fails_closed_and_emits_every_unreached_group():
    inputs = _manufactured_inputs()
    evidence = dict(inputs.native_position_tangent_evidence)
    evidence["lanes"] = dict(evidence["lanes"])
    evidence["lanes"].pop(NATIVE_POSITION_TANGENT_LANES[-1])
    inputs = replace(inputs, native_position_tangent_evidence=evidence)
    provenance = capture_protocol125_preacceleration_provenance(inputs)
    result = evaluate_protocol125_preacceleration(inputs, provenance)
    assert result["classification"] == "INVALID-audit"
    assert not result["complete"]
    assert not result["provenance_valid"]
    assert tuple(result["groups"]) == PRE_ACCELERATION_GROUPS
    native = result["groups"]["native_position_tangent"]
    assert not native["complete"]
    assert not native["provenance_valid"]
    assert "lane inventory" in native["invalid_reasons"][0]
    for name in PRE_ACCELERATION_GROUPS[2:]:
        assert not result["groups"][name]["complete"]
        assert result["groups"][name]["details"]["scorer_executed"] is False
    assert not result["acceleration_evaluated"]


def test_missing_provenance_invalidates_all_groups_before_any_scorer():
    inputs = _manufactured_inputs()
    result = evaluate_protocol125_preacceleration(inputs, None)
    assert result["classification"] == "INVALID-audit"
    assert tuple(result["groups"]) == PRE_ACCELERATION_GROUPS
    assert all(not record["complete"] for record in result["groups"].values())
    assert all(
        record["details"]["scorer_executed"] is False
        for record in result["groups"].values()
    )
    assert result["input_hashes_after"] is None
    assert not result["acceleration_evaluated"]


def test_complete_native_scientific_failure_remains_a_failure_not_invalid():
    inputs = _manufactured_inputs()
    evidence = dict(inputs.native_position_tangent_evidence)
    evidence["lanes"] = {
        name: dict(record) for name, record in evidence["lanes"].items()
    }
    evidence["lanes"]["source_node_wall_rows"]["passed"] = False
    inputs = replace(inputs, native_position_tangent_evidence=evidence)
    scorers = dict(preacceleration._SCORERS)

    def forbidden_position_score(*args, **kwargs):
        raise AssertionError("position scorer ran after a native scientific failure")

    scorers["position_representation"] = forbidden_position_score
    provenance = capture_protocol125_preacceleration_provenance(inputs)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(preacceleration, "_SCORERS", scorers)
        result = evaluate_protocol125_preacceleration(inputs, provenance)
    assert result["classification"] == "FAIL-single-parent-pre-acceleration"
    assert result["complete"]
    assert result["provenance_valid"]
    assert not result["passed"]
    assert result["invalid_reasons"] == ()
    assert all(record["complete"] for record in result["groups"].values())
    assert all(
        record["provenance_valid"] for record in result["groups"].values()
    )
    assert not result["groups"]["native_position_tangent"]["passed"]
    assert result["groups"]["pre_acceleration_construction"]["passed"]
    for name in PRE_ACCELERATION_GROUPS[2:]:
        stopped = result["groups"][name]
        assert stopped["complete"] is True
        assert stopped["provenance_valid"] is True
        assert stopped["passed"] is False
        assert stopped["not_reached"] is True
        assert stopped["blocked_by"] == "native_position_tangent"
        assert stopped["details"]["scorer_executed"] is False
    assert not result["acceleration_evaluated"]


def test_source_reproduction_gates_physical_components_not_reduced_diagnostic():
    z = np.linspace(1.0, np.e, 5)
    r = np.linspace(0.0, 12.0, 7)
    reduced = np.zeros((len(z), len(r), 9))
    reduced[:, :, 2] = -1.0
    reduced[:, :, 3] = 1.0
    reduced[:, :, 6] = 1.0

    class PhysicalMismatchState:
        def evaluate_reduced(self, found_z, found_r):
            assert np.array_equal(found_z, z)
            assert np.array_equal(found_r, r)
            return reduced.copy()

        def evaluate_coordinate_components(self, found_z, found_r):
            physical = preacceleration.reduced_to_physical(reduced, found_r)
            physical[:, :, 4] += 2e-12
            return physical

    score = preacceleration._source_reproduction_score(
        PhysicalMismatchState(),
        {"z": z, "r": r, "position": reduced},
    )
    assert score["reduced_coefficients_diagnostic"]["E_inf"] == 0.0
    assert score["E_inf"] == pytest.approx(2e-12)
    assert score["constituents"]["records"]["h_rr"]["E_inf"] == pytest.approx(
        2e-12,
    )
    assert not score["passed"]

    class ReducedMismatchState:
        def evaluate_reduced(self, found_z, found_r):
            values = reduced.copy()
            values[:, :, 4] += 2e-6
            return values

        def evaluate_coordinate_components(self, found_z, found_r):
            return preacceleration.reduced_to_physical(reduced, found_r)

    diagnostic_only = preacceleration._source_reproduction_score(
        ReducedMismatchState(),
        {"z": z, "r": r, "position": reduced},
    )
    assert diagnostic_only["passed"]
    assert diagnostic_only["E_inf"] == 0.0
    assert diagnostic_only["reduced_coefficients_diagnostic"]["E_inf"] > 1e-12
    assert diagnostic_only["reduced_coefficients_diagnostic"]["gated"] is False


def test_pre_q4_q5_image_failure_stops_before_downstream_pure_scorers(monkeypatch):
    inputs = _manufactured_inputs()

    def failed_images(*args, **kwargs):
        return {
            "passed": False,
            "state_name": "position",
            "comparison_kind": "Q53_Q33",
            "manufactured_failure": True,
        }

    monkeypatch.setattr(
        preacceleration,
        "score_q4_q5_derivative_images_on_v_meshes",
        failed_images,
    )
    scorers = dict(preacceleration._SCORERS)

    def forbidden_dense_score(*args, **kwargs):
        raise AssertionError("dense scorer ran after the q4/q5 image failure")

    scorers["dense_boundary_audit"] = forbidden_dense_score
    monkeypatch.setattr(preacceleration, "_SCORERS", scorers)
    provenance = capture_protocol125_preacceleration_provenance(inputs)
    result = evaluate_protocol125_preacceleration(inputs, provenance)
    assert result["classification"] == "FAIL-single-parent-pre-acceleration"
    assert result["complete"] is True
    assert result["provenance_valid"] is True
    assert result["invalid_reasons"] == ()
    failed = result["groups"]["position_representation"]
    assert failed["complete"] is True
    assert failed["provenance_valid"] is True
    assert failed["passed"] is False
    assert failed["details"]["Q53_Q33_position_q4_q5_images"][
        "manufactured_failure"
    ] is True
    start = PRE_ACCELERATION_GROUPS.index("position_representation")+1
    for name in PRE_ACCELERATION_GROUPS[start:]:
        stopped = result["groups"][name]
        assert stopped["not_reached"] is True
        assert stopped["blocked_by"] == "position_representation"
        assert stopped["details"]["scorer_executed"] is False


def test_post_capture_legacy_tamper_is_rejected_before_scoring():
    inputs = _manufactured_inputs()
    provenance = capture_protocol125_preacceleration_provenance(inputs)
    inputs.legacy_Q55_by_mesh["V0"]["position"][0, 0, 0] = 1e-4
    result = evaluate_protocol125_preacceleration(inputs, provenance)
    assert result["classification"] == "INVALID-audit"
    assert "input hash mismatch" in result["invalid_reasons"][0]
    assert all(not record["provenance_valid"] for record in result["groups"].values())
    assert not result["acceleration_evaluated"]


def _manufactured_construction_failure(monkeypatch):
    z = np.linspace(1.0, np.e, 3)
    r = np.linspace(0.0, 12.0, 4)
    zeros = np.zeros((len(z), len(r)))
    monkeypatch.setitem(
        construction.PARENT_SPECS,
        "N0",
        {
            "nz": len(z),
            "nr": len(r),
            "reference_iterations": 5,
            "coordinate_sha256": hash_arrays(z, r),
        },
    )
    reference = {
        "z": z,
        "r": r,
        "q": zeros,
        "phi": zeros,
        "history": [1e-3, 2e-9],
        "converged": False,
        "max_abs_residual": 2e-9,
        "residual_l2": 1e-9,
    }
    return construction._construction_failure_record(
        "N0",
        "a"*64,
        reference,
        failure_gate="finite_wall_reference",
        measured_value=2e-9,
        ceiling=construction.FINITE_WALL_REFERENCE_CEILING,
    )


def test_construction_failure_composer_emits_one_failure_then_ordered_stops(
    monkeypatch,
):
    failure = _manufactured_construction_failure(monkeypatch)
    result = compose_protocol125_construction_failure_records(failure)
    assert result["classification"] == "FAIL-single-parent-pre-acceleration"
    assert result["complete"] is True
    assert result["provenance_valid"] is True
    assert result["passed"] is False
    assert result["parent_label"] == "N0"
    assert result["parent_identity"] == failure["parent_identity"]
    assert tuple(result["groups"]) == PRE_ACCELERATION_GROUPS
    first = result["groups"]["pre_acceleration_construction"]
    assert first["complete"] is True
    assert first["provenance_valid"] is True
    assert first["passed"] is False
    assert first.get("not_reached", False) is False
    assert first["details"]["construction_solver_executed"] is True
    assert first["details"]["failure_record_validator_executed"] is True
    assert first["details"]["normal_construction_reload_scorer_executed"] is False
    assert first["details"]["failure_gate"] == "finite_wall_reference"
    assert first["details"]["construction_failure_record_sha256"] == failure[
        "fingerprint"
    ]
    for name in PRE_ACCELERATION_GROUPS[1:]:
        stopped = result["groups"][name]
        assert stopped["complete"] is True
        assert stopped["provenance_valid"] is True
        assert stopped["passed"] is False
        assert stopped["not_reached"] is True
        assert stopped["blocked_by"] == "pre_acceleration_construction"
        assert stopped["details"]["scorer_executed"] is False
        assert stopped["details"]["failure_kind"] == "ordered-scientific-stop"
    assert len({
        record["fingerprint"] for record in result["groups"].values()
    }) == len(PRE_ACCELERATION_GROUPS)
    assert result["acceleration_evaluated"] is False
    assert result["acceleration_authorized"] is False
    assert result["scientific_execution_authorized"] is False
    assert result["artifact_written"] is False
    with pytest.raises(TypeError):
        result["groups"]["native_position_tangent"] = {}


def test_construction_failure_composer_rejects_tampered_record(monkeypatch):
    failure = _manufactured_construction_failure(monkeypatch)
    tampered = dict(failure)
    payload = dict(tampered["scientific_payload"])
    payload["reference_q"] = np.asarray(payload["reference_q"]).copy()
    payload["reference_q"][0, 0] = 1.0
    tampered["scientific_payload"] = payload
    with pytest.raises(ValueError, match="payload hashes"):
        compose_protocol125_construction_failure_records(tampered)


def test_native_prerequisite_failure_passes_construction_then_stops_position(
    monkeypatch,
):
    z = np.linspace(1.0, np.e, 3)
    r = np.linspace(0.0, 12.0, 4)
    zeros = np.zeros((len(z), len(r)))
    monkeypatch.setitem(
        construction.PARENT_SPECS,
        "N0",
        {
            "nz": len(z),
            "nr": len(r),
            "reference_iterations": 5,
            "coordinate_sha256": hash_arrays(z, r),
        },
    )
    reference = {
        "z": z,
        "r": r,
        "q": zeros,
        "phi": zeros,
        "history": [1e-3, 1e-12],
        "converged": True,
        "max_abs_residual": 1e-12,
        "residual_l2": 5e-13,
    }
    selected = {
        "q": zeros,
        "phi": zeros,
        "psi": np.broadcast_to(1.0/z[:, None], zeros.shape).copy(),
        "history": [1e-4, 1e-12],
        "damping_history": [1.0],
        "converged": True,
        "maximum_residual": 1e-12,
        "residual_l2": 5e-13,
    }
    failure = construction._construction_failure_record(
        "N0",
        "a"*64,
        reference,
        failure_gate="native_position_prerequisite",
        measured_value=3e-9,
        ceiling=construction.NATIVE_POSITION_PREREQUISITE_CEILING,
        selected=selected,
        native_raw_position=np.zeros((*zeros.shape, 9)),
        native_prerequisite={
            "sphere_metric_normalized_Linf": 3e-9,
            "Phi_robin_Linf": 2e-11,
        },
    )
    result = compose_protocol125_construction_failure_records(failure)
    construction_gate = result["groups"]["pre_acceleration_construction"]
    native_gate = result["groups"]["native_position_tangent"]
    assert construction_gate["passed"] is True
    assert native_gate["passed"] is False
    assert native_gate["details"]["failure_classification"] == (
        "FAIL-parent-position"
    )
    for name in PRE_ACCELERATION_GROUPS[2:]:
        assert result["groups"][name]["not_reached"] is True
        assert result["groups"][name]["blocked_by"] == (
            "native_position_tangent"
        )
