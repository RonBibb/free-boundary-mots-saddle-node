from __future__ import annotations

import pytest

from bhps.joint_parent_adjudication import (
    PARENT_GATE_GROUPS,
    TWO_PARENT_GATE_GROUPS,
)
from bhps.joint_parent_gate_ledger import Protocol125GateLedger


N0_IDENTITY = "0"*64
N1_IDENTITY = "1"*64
PARENT_IDENTITIES = {"N0": N0_IDENTITY, "N1": N1_IDENTITY}


def _freeze_record(status="FROZEN"):
    return {
        "status": status,
        "protocol_sha256": "a"*64,
        "adjudicator_sha256": "b"*64,
        "frozen_before_parent_data": True,
        "independent_review_passed": True,
        "scientific_candidates_absent_at_freeze": True,
    }


def _parent_gate(parent, index=0, passed=True):
    return {
        "complete": True,
        "provenance_valid": True,
        "passed": passed,
        "fingerprint": f"{index % 16:x}"*64,
        "parent_label": parent,
        "parent_identity": PARENT_IDENTITIES[parent],
    }


def _two_parent_gate(index=0, passed=True):
    return {
        "complete": True,
        "provenance_valid": True,
        "passed": passed,
        "fingerprint": f"{index % 16:x}"*64,
        "parent_identities": dict(PARENT_IDENTITIES),
    }


def _complete_ledger(*, freeze_record=None, failed=None):
    ledger = Protocol125GateLedger(
        _freeze_record() if freeze_record is None else freeze_record,
        PARENT_IDENTITIES,
    )
    index = 2
    for parent in ("N0", "N1"):
        for group in PARENT_GATE_GROUPS:
            ledger = ledger.append_parent_gate(
                parent,
                group,
                _parent_gate(parent, index, passed=(parent, group) != failed),
            )
            index += 1
    for group in TWO_PARENT_GATE_GROUPS:
        ledger = ledger.append_two_parent_gate(
            group,
            _two_parent_gate(index, passed=("two-parent", group) != failed),
        )
        index += 1
    return ledger


def test_missing_gate_finalizes_as_invalid_and_never_authorizes():
    ledger = Protocol125GateLedger(_freeze_record(), PARENT_IDENTITIES)
    ledger = ledger.append_parent_gate(
        "N0", "pre_acceleration_construction", _parent_gate("N0"),
    )
    result = ledger.finalize()
    assert result["classification"] == "INVALID-audit"
    assert result["invalid_reasons"]
    assert not result["phase_a_authorized"]


def test_append_is_persistent_and_duplicate_gate_is_rejected():
    empty = Protocol125GateLedger(_freeze_record(), PARENT_IDENTITIES)
    record = _parent_gate("N0")
    once = empty.append_parent_gate("N0", "signature_union", record)
    assert not empty.parent_records["N0"]
    assert "signature_union" in once.parent_records["N0"]
    with pytest.raises(ValueError, match="duplicate parent gate"):
        once.append_parent_gate("N0", "signature_union", record)


def test_parent_and_two_parent_identity_mismatches_are_rejected():
    ledger = Protocol125GateLedger(_freeze_record(), PARENT_IDENTITIES)
    wrong_parent = _parent_gate("N0")
    wrong_parent["parent_identity"] = N1_IDENTITY
    with pytest.raises(ValueError, match="parent identity is mismatched"):
        ledger.append_parent_gate("N0", "signature_union", wrong_parent)

    wrong_pair = _two_parent_gate()
    wrong_pair["parent_identities"]["N1"] = "2"*64
    with pytest.raises(ValueError, match="parent identities are mismatched"):
        ledger.append_two_parent_gate("N0_N1_representation", wrong_pair)


def test_source_record_and_freeze_record_tampering_cannot_change_snapshot():
    freeze = _freeze_record()
    record = _parent_gate("N0")
    ledger = Protocol125GateLedger(freeze, PARENT_IDENTITIES)
    ledger = ledger.append_parent_gate("N0", "signature_union", record)
    freeze["status"] = "DRAFT — INVALID-specification"
    record["passed"] = False
    assert ledger.protocol_freeze_record["status"] == "FROZEN"
    assert ledger.parent_records["N0"]["signature_union"]["passed"] is True
    with pytest.raises(TypeError):
        ledger.parent_records["N0"]["signature_union"]["passed"] = False
    with pytest.raises(AttributeError, match="snapshots are immutable"):
        ledger._protocol_freeze_record = _freeze_record("DRAFT")


def test_invalid_freeze_record_dominates_an_all_pass_ledger():
    ledger = _complete_ledger(
        freeze_record=_freeze_record("DRAFT — INVALID-specification"),
    )
    result = ledger.finalize()
    assert result["classification"] == "INVALID-audit"
    assert result["invalid_reasons"] == ("protocol-is-not-prospectively-frozen",)
    assert not result["phase_a_authorized"]


def test_complete_all_pass_simulated_frozen_ledger_classifies_pass():
    result = _complete_ledger().finalize()
    assert result["classification"] == "PASS-native-joint-parent"
    assert result["phase_a_authorized"]
    assert not result["rhs_rk_phase_b_full_matrix_authorized"]
    assert not result["interface_physics_authorized"]


def test_failed_gate_is_preserved_as_scientific_failure_not_invalid_audit():
    result = _complete_ledger(failed=("N1", "wall_algebra")).finalize()
    assert result["classification"] == "FAIL-acceleration"
    assert result["failed_acceleration_groups"] == ("N1:wall_algebra",)
    assert not result["phase_a_authorized"]


def test_gate_schema_is_fail_closed():
    ledger = Protocol125GateLedger(_freeze_record(), PARENT_IDENTITIES)
    incomplete = _parent_gate("N0")
    del incomplete["fingerprint"]
    with pytest.raises(ValueError, match="missing: fingerprint"):
        ledger.append_parent_gate("N0", "signature_union", incomplete)
    non_boolean = _parent_gate("N0")
    non_boolean["complete"] = 1
    with pytest.raises(TypeError, match="complete must be a bool"):
        ledger.append_parent_gate("N0", "signature_union", non_boolean)
