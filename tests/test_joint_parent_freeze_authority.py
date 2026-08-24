from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bhps.joint_parent_freeze_authority import (
    AUTHORIZATION_SCOPE,
    Protocol125FreezeAuthorityError,
    _manifest_fingerprint,
    revalidate_protocol125_freeze_authority_snapshot,
    validate_protocol125_freeze_authority,
)
from bhps.joint_parent_gate_ledger import Protocol125GateLedger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_PROTOCOL = PROJECT_ROOT / "notes/125_A790_joint_parent_builder_protocol.md"
REAL_ADJUDICATOR = PROJECT_ROOT / "src/bhps/joint_parent_adjudication.py"
REAL_LEDGER = PROJECT_ROOT / "src/bhps/joint_parent_gate_ledger.py"
REAL_FREEZE_AUTHORITY = PROJECT_ROOT / "src/bhps/joint_parent_freeze_authority.py"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_entry(path):
    return {"path": str(path), "sha256": _sha256(path)}


def _manufactured_record(tmp_path, *, candidate_state="absent"):
    protocol = tmp_path / "protocol.md"
    protocol.write_text(
        "# Manufactured Protocol 125\n\nDate: 2026-08-15\n"
        "Status: **FROZEN**\n\n## Scope\nNo candidate exists.\n",
        encoding="utf-8",
    )
    adjudicator = tmp_path / "adjudicator.py"
    adjudicator.write_text("DECISION = 'fail-closed'\n", encoding="utf-8")
    source = tmp_path / "scorer.py"
    source.write_text("def score():\n    return False\n", encoding="utf-8")
    manifest = {"scorer": _file_entry(source)}
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps({
            "verdict": "PASS",
            "protocol_sha256": _sha256(protocol),
            "adjudicator_sha256": _sha256(adjudicator),
            "source_manifest_sha256": _manifest_fingerprint(manifest),
            "review": "independent",
        }) + "\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate-output"
    if candidate_state == "empty":
        candidate.mkdir()
    return {
        "status": "FROZEN",
        "protocol": _file_entry(protocol),
        "adjudicator": _file_entry(adjudicator),
        "source_manifest": manifest,
        "independent_review": {
            **_file_entry(review),
            "verdict": "PASS",
        },
        "frozen_before_parent_data": True,
        "scientific_candidates_absent_at_freeze": True,
        "candidate_output_directory": str(candidate),
        "candidate_output_state_at_freeze": candidate_state,
    }


@pytest.mark.parametrize("candidate_state", ["absent", "empty"])
def test_valid_manufactured_record_returns_deeply_immutable_read_only_authority(
    tmp_path, candidate_state,
):
    record = _manufactured_record(tmp_path, candidate_state=candidate_state)
    authority = validate_protocol125_freeze_authority(record)
    assert authority["status"] == "FROZEN"
    assert authority["authorization_scope"] == AUTHORIZATION_SCOPE
    assert authority["independent_review_passed"] is True
    assert authority["candidate_output_state_at_freeze"] == candidate_state
    assert authority["scientific_execution_authorized"] is False
    with pytest.raises(TypeError):
        authority["status"] = "DRAFT"
    with pytest.raises(TypeError):
        authority["source_manifest"]["scorer"]["sha256"] = "0"*64
    record["source_manifest"]["scorer"]["sha256"] = "0"*64
    assert authority["source_manifest"]["scorer"]["sha256"] != "0"*64


def test_authority_is_ledger_compatible_but_does_not_bypass_missing_gates(tmp_path):
    authority = validate_protocol125_freeze_authority(
        _manufactured_record(tmp_path),
    )
    ledger = Protocol125GateLedger(authority, {"N0": "0"*64, "N1": "1"*64})
    decision = ledger.finalize()
    assert ledger.protocol_freeze_record["protocol_sha256"] == authority[
        "protocol_sha256"
    ]
    assert decision["classification"] == "INVALID-audit"
    assert decision["phase_a_authorized"] is False
    assert decision["rhs_rk_phase_b_full_matrix_authorized"] is False


def test_current_real_protocol_cannot_use_an_unbound_review_as_freeze_authority(
    tmp_path,
):
    # Use a fresh absent candidate path so this review-binding test remains
    # valid after a historical prospective run has populated its own sealed
    # output directory.
    missing_candidate = tmp_path / "unbound-review-candidate"
    assert not missing_candidate.exists()
    record = {
        "status": "FROZEN",
        "protocol": _file_entry(REAL_PROTOCOL),
        "adjudicator": _file_entry(REAL_ADJUDICATOR),
        "source_manifest": {
            "freeze-authority": _file_entry(REAL_FREEZE_AUTHORITY),
        },
        # This deliberately generic file is never an authoritative review.
        # Before final freeze the protocol header rejects the record; after
        # final freeze the missing exact review bindings reject it instead.
        "independent_review": {
            **_file_entry(REAL_LEDGER),
            "verdict": "PASS",
        },
        "frozen_before_parent_data": True,
        "scientific_candidates_absent_at_freeze": True,
        "candidate_output_directory": str(missing_candidate),
        "candidate_output_state_at_freeze": "absent",
    }
    with pytest.raises(
        Protocol125FreezeAuthorityError,
        match=(
            "protocol header status is not exactly FROZEN"
            "|independent review"
        ),
    ):
        validate_protocol125_freeze_authority(record)


def test_freeze_record_and_protocol_status_must_both_be_exact(tmp_path):
    record = _manufactured_record(tmp_path)
    record["status"] = "frozen"
    with pytest.raises(Protocol125FreezeAuthorityError, match="status must be exactly"):
        validate_protocol125_freeze_authority(record)

    record = _manufactured_record(tmp_path)
    protocol = Path(record["protocol"]["path"])
    protocol.write_text(
        "# Manufactured Protocol 125\n\nStatus: **DRAFT**\n"
        "\n## Scope\nStatus: **FROZEN**\n",
        encoding="utf-8",
    )
    record["protocol"] = _file_entry(protocol)
    with pytest.raises(Protocol125FreezeAuthorityError, match="protocol header status"):
        validate_protocol125_freeze_authority(record)


@pytest.mark.parametrize("entry_name", ["protocol", "adjudicator"])
def test_primary_file_hashes_are_lowercase_and_byte_exact(tmp_path, entry_name):
    record = _manufactured_record(tmp_path)
    record[entry_name]["sha256"] = record[entry_name]["sha256"].upper()
    with pytest.raises(Protocol125FreezeAuthorityError, match="lowercase SHA-256"):
        validate_protocol125_freeze_authority(record)

    record = _manufactured_record(tmp_path)
    record[entry_name]["sha256"] = "0"*64
    with pytest.raises(Protocol125FreezeAuthorityError, match="does not match file bytes"):
        validate_protocol125_freeze_authority(record)


def test_source_manifest_hash_and_post_record_tampering_fail_closed(tmp_path):
    record = _manufactured_record(tmp_path)
    record["source_manifest"]["scorer"]["sha256"] = (
        record["source_manifest"]["scorer"]["sha256"].upper()
    )
    with pytest.raises(Protocol125FreezeAuthorityError, match="lowercase SHA-256"):
        validate_protocol125_freeze_authority(record)

    record = _manufactured_record(tmp_path)
    source = Path(record["source_manifest"]["scorer"]["path"])
    source.write_text("TAMPERED = True\n", encoding="utf-8")
    with pytest.raises(Protocol125FreezeAuthorityError, match="does not match file bytes"):
        validate_protocol125_freeze_authority(record)


def test_independent_review_must_be_explicit_pass_in_record_and_file(tmp_path):
    record = _manufactured_record(tmp_path)
    record["independent_review"]["verdict"] = "FAIL"
    with pytest.raises(Protocol125FreezeAuthorityError, match="recorded.*PASS"):
        validate_protocol125_freeze_authority(record)

    record = _manufactured_record(tmp_path)
    review = Path(record["independent_review"]["path"])
    review.write_text(json.dumps({"verdict": "FAIL"}) + "\n", encoding="utf-8")
    record["independent_review"].update(_file_entry(review))
    with pytest.raises(
        Protocol125FreezeAuthorityError,
        match="exact review bindings",
    ):
        validate_protocol125_freeze_authority(record)

    record = _manufactured_record(tmp_path)
    record["independent_review"]["sha256"] = "0"*64
    with pytest.raises(Protocol125FreezeAuthorityError, match="does not match file bytes"):
        validate_protocol125_freeze_authority(record)


def test_markdown_independent_review_pass_is_supported(tmp_path):
    record = _manufactured_record(tmp_path)
    review = Path(record["independent_review"]["path"])
    protocol_sha256 = record["protocol"]["sha256"]
    adjudicator_sha256 = record["adjudicator"]["sha256"]
    manifest_sha256 = _manifest_fingerprint(record["source_manifest"])
    review.write_text(
        "# Independent freeze review\n\n"
        "Verdict: **PASS**\n"
        f"Protocol-SHA256: {protocol_sha256}\n"
        f"Adjudicator-SHA256: {adjudicator_sha256}\n"
        f"Source-Manifest-SHA256: {manifest_sha256}\n\n"
        "## Findings\nClosed.\n",
        encoding="utf-8",
    )
    record["independent_review"] = {**_file_entry(review), "verdict": "PASS"}
    authority = validate_protocol125_freeze_authority(record)
    assert authority["independent_review"]["verdict"] == "PASS"


def test_generic_pass_review_without_exact_bindings_is_rejected(tmp_path):
    record = _manufactured_record(tmp_path)
    review = Path(record["independent_review"]["path"])
    review.write_text(json.dumps({"verdict": "PASS"}) + "\n", encoding="utf-8")
    record["independent_review"].update(_file_entry(review))
    with pytest.raises(
        Protocol125FreezeAuthorityError, match="exact review bindings",
    ):
        validate_protocol125_freeze_authority(record)


@pytest.mark.parametrize(
    "flag",
    ["frozen_before_parent_data", "scientific_candidates_absent_at_freeze"],
)
@pytest.mark.parametrize("invalid", [False, 1, "true"])
def test_prospective_freeze_flags_must_be_exact_boolean_true(tmp_path, flag, invalid):
    record = _manufactured_record(tmp_path)
    record[flag] = invalid
    with pytest.raises(Protocol125FreezeAuthorityError, match=f"{flag} must be exactly true"):
        validate_protocol125_freeze_authority(record)


def test_candidate_directory_record_must_match_absent_or_empty_state(tmp_path):
    absent_case = tmp_path / "absent-case"
    absent_case.mkdir()
    record = _manufactured_record(absent_case)
    candidate = Path(record["candidate_output_directory"])
    candidate.mkdir()
    with pytest.raises(Protocol125FreezeAuthorityError, match="does not match"):
        validate_protocol125_freeze_authority(record)

    empty_case = tmp_path / "empty-case"
    empty_case.mkdir()
    record = _manufactured_record(empty_case, candidate_state="empty")
    candidate = Path(record["candidate_output_directory"])
    (candidate / "candidate.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(Protocol125FreezeAuthorityError, match="does not match"):
        validate_protocol125_freeze_authority(record)

    invalid_state_case = tmp_path / "invalid-state-case"
    invalid_state_case.mkdir()
    record = _manufactured_record(invalid_state_case)
    record["candidate_output_state_at_freeze"] = "nonempty"
    with pytest.raises(Protocol125FreezeAuthorityError, match="absent or empty"):
        validate_protocol125_freeze_authority(record)


def test_validated_authority_snapshot_rehashes_inputs_but_preserves_freeze_time_state(
    tmp_path,
):
    record = _manufactured_record(tmp_path)
    authority = validate_protocol125_freeze_authority(record)
    candidate = Path(record["candidate_output_directory"])
    candidate.mkdir()
    (candidate/"recovery_index.json").write_text("{}\n", encoding="utf-8")

    recovered = revalidate_protocol125_freeze_authority_snapshot(authority)
    assert recovered["candidate_output_state_at_freeze"] == "absent"
    assert recovered["candidate_output_directory"] == str(candidate)

    source = Path(record["source_manifest"]["scorer"]["path"])
    source.write_text("TAMPERED = True\n", encoding="utf-8")
    with pytest.raises(Protocol125FreezeAuthorityError, match="does not match"):
        revalidate_protocol125_freeze_authority_snapshot(authority)


def test_validated_authority_snapshot_rejects_recorded_review_binding_tamper(
    tmp_path,
):
    authority = validate_protocol125_freeze_authority(
        _manufactured_record(tmp_path),
    )
    tampered = dict(authority)
    tampered["independent_review"] = dict(authority["independent_review"])
    tampered["independent_review"]["source_manifest_sha256"] = "0"*64
    with pytest.raises(
        Protocol125FreezeAuthorityError,
        match="review is duplicated or no longer passes",
    ):
        revalidate_protocol125_freeze_authority_snapshot(tampered)


def test_schema_is_strict_and_manifest_must_be_nonempty(tmp_path):
    record = _manufactured_record(tmp_path)
    record["unexpected"] = True
    with pytest.raises(Protocol125FreezeAuthorityError, match="unexpected"):
        validate_protocol125_freeze_authority(record)

    record = _manufactured_record(tmp_path)
    record["source_manifest"] = {}
    with pytest.raises(Protocol125FreezeAuthorityError, match="nonempty mapping"):
        validate_protocol125_freeze_authority(record)
