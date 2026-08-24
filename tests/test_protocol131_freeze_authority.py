from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from bhps.protocol131_freeze_authority import (
    AUTHORIZATION_SCOPE,
    Protocol131FreezeAuthorityError,
    manifest_fingerprint,
    revalidate_protocol131_freeze_authority_snapshot,
    validate_protocol131_freeze_authority,
)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(path):
    return {"path": str(path), "sha256": _sha256(path)}


def _thaw(value):
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _record(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    candidate = tmp_path / "candidate-output"
    protocol = tmp_path / "protocol.md"
    protocol.write_text(
        "# Protocol 131 manufactured fixture\n\n"
        "Status: **FROZEN**\n"
        f"Candidate-Output-Directory: {candidate}\n\n"
        "## Scope\nArchive-only.\n",
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime.json"
    runtime.write_text('{"numpy":"frozen"}\n', encoding="utf-8")
    runner = tmp_path / "runner.py"
    runner.write_text("ARCHIVE_ONLY = True\n", encoding="utf-8")
    n0 = tmp_path / "parent_N0.npz"
    n0.write_bytes(b"sealed-N0")
    n1 = tmp_path / "parent_N1.npz"
    n1.write_bytes(b"sealed-N1")
    source_manifest = {"postmortem-runner": _entry(runner)}
    input_manifest = {"N0": _entry(n0), "N1": _entry(n1)}
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps({
            "verdict": "PASS",
            "protocol_sha256": _sha256(protocol),
            "runtime_contract_sha256": _sha256(runtime),
            "source_manifest_sha256": manifest_fingerprint(source_manifest),
            "input_manifest_sha256": manifest_fingerprint(input_manifest),
        }, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "status": "FROZEN",
        "protocol": _entry(protocol),
        "runtime_contract": _entry(runtime),
        "source_manifest": source_manifest,
        "input_manifest": input_manifest,
        "independent_review": {**_entry(review), "verdict": "PASS"},
        "frozen_before_execution": True,
        "candidate_output_directory": str(candidate),
        "candidate_output_state_at_freeze": "absent",
    }


def test_valid_record_returns_deeply_immutable_archive_only_authority(tmp_path):
    record = _record(tmp_path)
    authority = validate_protocol131_freeze_authority(record)
    assert authority["authorization_scope"] == AUTHORIZATION_SCOPE
    assert authority["archive_only_postmortem_authorized"] is True
    assert authority["new_parent_construction_authorized"] is False
    assert authority["evolution_authorized"] is False
    assert authority["candidate_output_state_at_freeze"] == "absent"
    with pytest.raises(TypeError):
        authority["status"] = "DRAFT"
    with pytest.raises(TypeError):
        authority["input_manifest"]["N0"]["sha256"] = "0" * 64
    record["input_manifest"]["N0"]["sha256"] = "0" * 64
    assert authority["input_manifest"]["N0"]["sha256"] != "0" * 64


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda record: record.update({"extra": True}), "unexpected"),
        (lambda record: record["protocol"].update({"extra": True}), "unexpected"),
        (lambda record: record["independent_review"].pop("verdict"), "missing"),
        (lambda record: record.update({"source_manifest": {}}), "nonempty"),
        (lambda record: record.update({"input_manifest": {}}), "nonempty"),
    ],
)
def test_every_freeze_record_schema_is_exact(tmp_path, change, message):
    record = _record(tmp_path)
    change(record)
    with pytest.raises(Protocol131FreezeAuthorityError, match=message):
        validate_protocol131_freeze_authority(record)


@pytest.mark.parametrize(
    "component",
    ["protocol", "runtime_contract", "source", "input", "review"],
)
def test_every_frozen_component_is_rehashed_from_file_bytes(tmp_path, component):
    record = _record(tmp_path)
    if component in {"protocol", "runtime_contract"}:
        path = Path(record[component]["path"])
    elif component == "review":
        path = Path(record["independent_review"]["path"])
    elif component == "source":
        path = Path(record["source_manifest"]["postmortem-runner"]["path"])
    else:
        path = Path(record["input_manifest"]["N0"]["path"])
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(Protocol131FreezeAuthorityError, match="does not match file bytes"):
        validate_protocol131_freeze_authority(record)


@pytest.mark.parametrize(
    "binding",
    [
        "protocol_sha256",
        "runtime_contract_sha256",
        "source_manifest_sha256",
        "input_manifest_sha256",
    ],
)
def test_review_must_bind_every_exact_freeze_input(tmp_path, binding):
    record = _record(tmp_path)
    review = Path(record["independent_review"]["path"])
    payload = json.loads(review.read_text(encoding="utf-8"))
    payload[binding] = "0" * 64
    review.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    record["independent_review"].update(_entry(review))
    with pytest.raises(Protocol131FreezeAuthorityError, match="does not bind"):
        validate_protocol131_freeze_authority(record)


def test_generic_pass_or_extra_review_key_is_not_authority(tmp_path):
    record = _record(tmp_path)
    review = Path(record["independent_review"]["path"])
    review.write_text('{"verdict":"PASS"}\n', encoding="utf-8")
    record["independent_review"].update(_entry(review))
    with pytest.raises(Protocol131FreezeAuthorityError, match="invalid keys"):
        validate_protocol131_freeze_authority(record)

    record = _record(tmp_path / "extra-key")
    review = Path(record["independent_review"]["path"])
    payload = json.loads(review.read_text(encoding="utf-8"))
    payload["comment"] = "generic assurance"
    review.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    record["independent_review"].update(_entry(review))
    with pytest.raises(Protocol131FreezeAuthorityError, match="unexpected"):
        validate_protocol131_freeze_authority(record)


def test_markdown_review_with_all_exact_bindings_is_supported(tmp_path):
    record = _record(tmp_path)
    review = Path(record["independent_review"]["path"])
    review.write_text(
        "# Protocol 131 independent review\n\n"
        "Verdict: **PASS**\n"
        f"Protocol-SHA256: {record['protocol']['sha256']}\n"
        f"Runtime-Contract-SHA256: {record['runtime_contract']['sha256']}\n"
        "Source-Manifest-SHA256: "
        f"{manifest_fingerprint(record['source_manifest'])}\n"
        "Input-Manifest-SHA256: "
        f"{manifest_fingerprint(record['input_manifest'])}\n\n"
        "## Findings\nThe prospective package is internally bound.\n",
        encoding="utf-8",
    )
    record["independent_review"] = {**_entry(review), "verdict": "PASS"}
    authority = validate_protocol131_freeze_authority(record)
    assert authority["independent_review"]["verdict"] == "PASS"


@pytest.mark.parametrize("existing_kind", ["empty-directory", "file"])
def test_first_validation_requires_output_path_to_be_truly_absent(
    tmp_path, existing_kind,
):
    record = _record(tmp_path)
    candidate = Path(record["candidate_output_directory"])
    if existing_kind == "empty-directory":
        candidate.mkdir()
    else:
        candidate.write_text("already exists\n", encoding="utf-8")
    with pytest.raises(Protocol131FreezeAuthorityError, match="must be absent"):
        validate_protocol131_freeze_authority(record)

    record["candidate_output_state_at_freeze"] = "empty"
    with pytest.raises(Protocol131FreezeAuthorityError, match="exactly absent"):
        validate_protocol131_freeze_authority(record)


def test_protocol_and_flags_must_be_semantically_frozen(tmp_path):
    record = _record(tmp_path)
    protocol = Path(record["protocol"]["path"])
    protocol.write_text(
        "# Protocol 131 manufactured fixture\n\nStatus: **DRAFT**\n"
        f"Candidate-Output-Directory: {record['candidate_output_directory']}\n",
        encoding="utf-8",
    )
    record["protocol"] = _entry(protocol)
    with pytest.raises(Protocol131FreezeAuthorityError, match="header status"):
        validate_protocol131_freeze_authority(record)

    record = _record(tmp_path / "bad-flag")
    record["frozen_before_execution"] = 1
    with pytest.raises(Protocol131FreezeAuthorityError, match="exactly true"):
        validate_protocol131_freeze_authority(record)


def test_candidate_redirection_differs_from_reviewed_protocol(tmp_path):
    record = _record(tmp_path)
    record["candidate_output_directory"] = str(tmp_path / "redirected")
    with pytest.raises(Protocol131FreezeAuthorityError, match="reviewed protocol"):
        validate_protocol131_freeze_authority(record)


def test_duplicate_files_across_manifests_fail_closed(tmp_path):
    record = _record(tmp_path)
    record["input_manifest"]["N0"] = dict(
        record["source_manifest"]["postmortem-runner"]
    )
    review = Path(record["independent_review"]["path"])
    payload = json.loads(review.read_text(encoding="utf-8"))
    payload["input_manifest_sha256"] = manifest_fingerprint(
        record["input_manifest"]
    )
    review.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    record["independent_review"].update(_entry(review))
    with pytest.raises(Protocol131FreezeAuthorityError, match="duplicates"):
        validate_protocol131_freeze_authority(record)


def test_validated_snapshot_can_be_revalidated_after_output_creation(tmp_path):
    record = _record(tmp_path)
    authority = validate_protocol131_freeze_authority(record)
    Path(record["candidate_output_directory"]).mkdir()
    restored = revalidate_protocol131_freeze_authority_snapshot(
        _thaw(authority)
    )
    assert restored["authorization_scope"] == AUTHORIZATION_SCOPE
    assert restored["candidate_output_state_at_freeze"] == "absent"
    assert restored["archive_only_postmortem_authorized"] is True
    with pytest.raises(TypeError):
        restored["source_manifest"]["postmortem-runner"]["sha256"] = "0" * 64


@pytest.mark.parametrize("component", ["source", "input", "review"])
def test_snapshot_revalidation_rehashes_every_frozen_byte(tmp_path, component):
    record = _record(tmp_path)
    authority = _thaw(validate_protocol131_freeze_authority(record))
    Path(record["candidate_output_directory"]).mkdir()
    if component == "source":
        path = Path(record["source_manifest"]["postmortem-runner"]["path"])
    elif component == "input":
        path = Path(record["input_manifest"]["N1"]["path"])
    else:
        path = Path(record["independent_review"]["path"])
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(Protocol131FreezeAuthorityError):
        revalidate_protocol131_freeze_authority_snapshot(authority)


def test_raw_record_is_not_accepted_as_a_validated_snapshot(tmp_path):
    record = _record(tmp_path)
    Path(record["candidate_output_directory"]).mkdir()
    with pytest.raises(Protocol131FreezeAuthorityError, match="invalid keys"):
        revalidate_protocol131_freeze_authority_snapshot(record)
