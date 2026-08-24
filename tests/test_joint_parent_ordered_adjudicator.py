from __future__ import annotations

import hashlib

from bhps.joint_parent_ordered_adjudicator import (
    POST_ACCELERATION_GROUPS,
    PRE_ACCELERATION_GROUPS,
    TWO_PARENT_GROUPS,
    adjudicate_protocol125_ordered,
)


def _digest(label):
    return hashlib.sha256(label.encode()).hexdigest()


def _freeze():
    return {
        "status": "FROZEN",
        "frozen_before_parent_data": True,
        "independent_review_passed": True,
        "scientific_candidates_absent_at_freeze": True,
        "protocol_sha256": _digest("protocol"),
        "adjudicator_sha256": _digest("adjudicator"),
    }


def _identities():
    return {"N0": _digest("N0"), "N1": _digest("N1")}


def _parent_records(groups, identities, *, failed=None, invalid=None):
    failed = failed or ()
    invalid = invalid or ()
    return {
        parent: {
            group: {
                "complete": True,
                "provenance_valid": (parent, group) not in invalid,
                "passed": (parent, group) not in failed,
                "fingerprint": _digest(f"{parent}:{group}"),
                "parent_label": parent,
                "parent_identity": identities[parent],
            }
            for group in groups
        }
        for parent in ("N0", "N1")
    }


def _two_parent_records(identities, *, failed=()):
    return {
        group: {
            "complete": True,
            "provenance_valid": True,
            "passed": group not in failed,
            "fingerprint": _digest(group),
            "parent_identities": identities,
        }
        for group in TWO_PARENT_GROUPS
    }


def _mark_ordered_stop(records, parent, groups, failed_group):
    """Turn one parent's complete group ledger into a valid early-stop ledger."""
    failure_index = groups.index(failed_group)
    records[parent][failed_group]["passed"] = False
    for group in groups[failure_index + 1 :]:
        records[parent][group]["passed"] = False
        records[parent][group]["not_reached"] = True
        records[parent][group]["blocked_by"] = failed_group
    return records


def test_ordered_adjudicator_accepts_legitimate_preacceleration_stop():
    identities = _identities()
    pre = _parent_records(
        PRE_ACCELERATION_GROUPS,
        identities,
        failed=(("N1", "dense_boundary_audit"),),
    )
    result = adjudicate_protocol125_ordered(
        pre,
        parent_identities=identities,
        protocol_freeze_record=_freeze(),
    )
    assert result["classification"] == "FAIL-parent-position"
    assert result["downstream_absence_is_ordered_stop"]
    assert not result["phase_a_authorized"]
    assert result["invalid_reasons"] == ()


def test_ordered_adjudicator_rejects_acceleration_after_prerequisite_failure():
    identities = _identities()
    pre = _parent_records(
        PRE_ACCELERATION_GROUPS,
        identities,
        failed=(("N0", "bulk_prerequisite"),),
    )
    post = _parent_records(POST_ACCELERATION_GROUPS, identities)
    result = adjudicate_protocol125_ordered(
        pre,
        parent_identities=identities,
        protocol_freeze_record=_freeze(),
        post_acceleration_records=post,
    )
    assert result["classification"] == "INVALID-audit"
    assert "downstream-records" in result["invalid_reasons"][0]


def test_ordered_adjudicator_accepts_legitimate_postacceleration_stop():
    identities = _identities()
    pre = _parent_records(PRE_ACCELERATION_GROUPS, identities)
    post = _parent_records(
        POST_ACCELERATION_GROUPS,
        identities,
        failed=(("N0", "endpoint_derivatives"),),
    )
    result = adjudicate_protocol125_ordered(
        pre,
        parent_identities=identities,
        protocol_freeze_record=_freeze(),
        post_acceleration_records=post,
    )
    assert result["classification"] == "FAIL-acceleration"
    assert result["reached_stage"] == "post-acceleration"


def test_ordered_adjudicator_full_pass_is_only_phase_a_authority():
    identities = _identities()
    result = adjudicate_protocol125_ordered(
        _parent_records(PRE_ACCELERATION_GROUPS, identities),
        parent_identities=identities,
        protocol_freeze_record=_freeze(),
        post_acceleration_records=_parent_records(
            POST_ACCELERATION_GROUPS, identities,
        ),
        two_parent_records=_two_parent_records(identities),
    )
    assert result["classification"] == "PASS-native-joint-parent"
    assert result["phase_a_authorized"]
    assert not result["rhs_rk_phase_b_full_matrix_authorized"]
    assert not result["interface_physics_authorized"]
    assert len(result["fingerprint"]) == 64


def test_ordered_adjudicator_missing_reached_stage_is_invalid_not_failure():
    identities = _identities()
    result = adjudicate_protocol125_ordered(
        _parent_records(PRE_ACCELERATION_GROUPS, identities),
        parent_identities=identities,
        protocol_freeze_record=_freeze(),
    )
    assert result["classification"] == "INVALID-audit"
    assert result["reached_stage"] == "post-acceleration"


def test_ordered_adjudicator_invalid_provenance_dominates_scientific_failure():
    identities = _identities()
    pre = _parent_records(
        PRE_ACCELERATION_GROUPS,
        identities,
        failed=(("N0", "native_position_tangent"),),
        invalid=(("N1", "signature_union"),),
    )
    result = adjudicate_protocol125_ordered(
        pre,
        parent_identities=identities,
        protocol_freeze_record=_freeze(),
    )
    assert result["classification"] == "INVALID-audit"
    assert result["invalid_reasons"]


def test_ordered_adjudicator_accepts_explicit_construction_failure_stop():
    identities = _identities()
    pre = _mark_ordered_stop(
        _parent_records(PRE_ACCELERATION_GROUPS, identities),
        "N0",
        PRE_ACCELERATION_GROUPS,
        "pre_acceleration_construction",
    )
    result = adjudicate_protocol125_ordered(
        pre,
        parent_identities=identities,
        protocol_freeze_record=_freeze(),
    )
    assert result["classification"] == "FAIL-parent-bulk"
    assert result["failed_bulk_groups"] == (
        "N0:pre_acceleration_construction",
    )
    assert result["invalid_reasons"] == ()


def test_ordered_adjudicator_accepts_explicit_acceleration_failure_stop():
    identities = _identities()
    post = _mark_ordered_stop(
        _parent_records(POST_ACCELERATION_GROUPS, identities),
        "N1",
        POST_ACCELERATION_GROUPS,
        "acceleration_closure",
    )
    result = adjudicate_protocol125_ordered(
        _parent_records(PRE_ACCELERATION_GROUPS, identities),
        parent_identities=identities,
        protocol_freeze_record=_freeze(),
        post_acceleration_records=post,
    )
    assert result["classification"] == "FAIL-acceleration"
    assert result["failed_acceleration_groups"] == (
        "N1:acceleration_closure",
    )
    assert result["invalid_reasons"] == ()


def test_ordered_adjudicator_rejects_not_reached_before_failure():
    identities = _identities()
    pre = _parent_records(PRE_ACCELERATION_GROUPS, identities)
    group = PRE_ACCELERATION_GROUPS[1]
    pre["N0"][group].update(
        passed=False,
        not_reached=True,
        blocked_by=PRE_ACCELERATION_GROUPS[0],
    )
    result = adjudicate_protocol125_ordered(
        pre,
        parent_identities=identities,
        protocol_freeze_record=_freeze(),
    )
    assert result["classification"] == "INVALID-audit"
    assert any("not-reached-before-failure" in reason for reason in result["invalid_reasons"])


def test_ordered_adjudicator_rejects_wrong_blocker_and_reached_after_stop():
    identities = _identities()
    pre = _mark_ordered_stop(
        _parent_records(PRE_ACCELERATION_GROUPS, identities),
        "N0",
        PRE_ACCELERATION_GROUPS,
        "native_position_tangent",
    )
    first_not_reached = PRE_ACCELERATION_GROUPS[2]
    pre["N0"][first_not_reached]["blocked_by"] = "bulk_prerequisite"
    later = PRE_ACCELERATION_GROUPS[3]
    pre["N0"][later].pop("not_reached")
    pre["N0"][later].pop("blocked_by")
    pre["N0"][later]["passed"] = True
    result = adjudicate_protocol125_ordered(
        pre,
        parent_identities=identities,
        protocol_freeze_record=_freeze(),
    )
    assert result["classification"] == "INVALID-audit"
    assert any("blocked-by-mismatch" in reason for reason in result["invalid_reasons"])
    assert any("reached-after-ordered-stop" in reason for reason in result["invalid_reasons"])
