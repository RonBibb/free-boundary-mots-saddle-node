"""Fail-closed prospective freeze authority for Protocol 131.

This module performs provenance checks only.  It does not create, modify, or
read scientific arrays; evaluate a residual or Jacobian; create the output
directory; or classify a result.  The first validation succeeds only while
the declared Protocol-131 output path is absent.

The raw freeze record has the exact schema::

    {
        "status": "FROZEN",
        "protocol": {"path": ABSOLUTE_PATH, "sha256": LOWERCASE_SHA256},
        "runtime_contract": {
            "path": ABSOLUTE_PATH,
            "sha256": LOWERCASE_SHA256,
        },
        "source_manifest": {
            LOGICAL_NAME: {
                "path": ABSOLUTE_PATH,
                "sha256": LOWERCASE_SHA256,
            },
            ...
        },
        "input_manifest": {
            LOGICAL_NAME: {
                "path": ABSOLUTE_PATH,
                "sha256": LOWERCASE_SHA256,
            },
            ...
        },
        "independent_review": {
            "path": ABSOLUTE_PATH,
            "sha256": LOWERCASE_SHA256,
            "verdict": "PASS",
        },
        "frozen_before_execution": True,
        "candidate_output_directory": ABSOLUTE_PATH,
        "candidate_output_state_at_freeze": "absent",
    }

The independent review can be JSON or Markdown.  It must bind the exact
protocol and runtime-contract digests and the canonical fingerprints of the
complete source and input manifests.  A generic PASS note is insufficient.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType


AUTHORIZATION_KIND = "Protocol-131-read-only-freeze-authority-v1"
AUTHORIZATION_SCOPE = "Protocol-131-archive-only-postmortem"

_TOP_LEVEL_KEYS = frozenset({
    "status",
    "protocol",
    "runtime_contract",
    "source_manifest",
    "input_manifest",
    "independent_review",
    "frozen_before_execution",
    "candidate_output_directory",
    "candidate_output_state_at_freeze",
})
_FILE_ENTRY_KEYS = frozenset({"path", "sha256"})
_REVIEW_ENTRY_KEYS = frozenset({"path", "sha256", "verdict"})
_REVIEW_BINDING_KEYS = frozenset({
    "verdict",
    "protocol_sha256",
    "runtime_contract_sha256",
    "source_manifest_sha256",
    "input_manifest_sha256",
})
_VALIDATED_AUTHORITY_KEYS = frozenset({
    "authorization_kind",
    "authorization_scope",
    "status",
    "protocol_path",
    "protocol_sha256",
    "runtime_contract_path",
    "runtime_contract_sha256",
    "source_manifest",
    "source_manifest_sha256",
    "input_manifest",
    "input_manifest_sha256",
    "independent_review",
    "frozen_before_execution",
    "candidate_output_directory",
    "candidate_output_state_at_freeze",
    "archive_only_postmortem_authorized",
    "new_parent_construction_authorized",
    "evolution_authorized",
})
_VALIDATED_REVIEW_KEYS = frozenset({
    "path",
    "sha256",
    "verdict",
    "protocol_sha256",
    "runtime_contract_sha256",
    "source_manifest_sha256",
    "input_manifest_sha256",
})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class Protocol131FreezeAuthorityError(ValueError):
    """Raised when prospective Protocol-131 authority cannot be established."""


def _require_exact_keys(record, expected, label):
    if not isinstance(record, Mapping):
        raise Protocol131FreezeAuthorityError(f"{label} must be a mapping")
    found = frozenset(record)
    if found != expected:
        missing = tuple(sorted(expected - found))
        unexpected = tuple(sorted(found - expected, key=str))
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unexpected:
            details.append(
                "unexpected " + ", ".join(str(name) for name in unexpected)
            )
        raise Protocol131FreezeAuthorityError(
            f"{label} has invalid keys: {'; '.join(details)}"
        )


def _require_lowercase_sha256(value, label):
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise Protocol131FreezeAuthorityError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _require_absolute_path(value, label):
    if type(value) is not str or not value:
        raise Protocol131FreezeAuthorityError(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise Protocol131FreezeAuthorityError(f"{label} must be an absolute path")
    return path


def _read_verified_file(entry, label):
    _require_exact_keys(entry, _FILE_ENTRY_KEYS, label)
    path = _require_absolute_path(entry["path"], f"{label} path")
    expected = _require_lowercase_sha256(entry["sha256"], f"{label} SHA-256")
    try:
        if path.is_symlink() or not path.is_file():
            raise Protocol131FreezeAuthorityError(
                f"{label} path is not a regular non-symlink file"
            )
        resolved = path.resolve(strict=True)
        payload = path.read_bytes()
    except Protocol131FreezeAuthorityError:
        raise
    except OSError as error:
        raise Protocol131FreezeAuthorityError(
            f"{label} file cannot be read"
        ) from error
    if hashlib.sha256(payload).hexdigest() != expected:
        raise Protocol131FreezeAuthorityError(
            f"{label} SHA-256 does not match file bytes"
        )
    return resolved, expected, payload


def _semantic_markdown_value(raw_value):
    value = raw_value.strip()
    if value.startswith("**") and value.endswith("**") and len(value) >= 4:
        value = value[2:-2]
    return value


def _markdown_preamble_field(payload, field, label):
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise Protocol131FreezeAuthorityError(
            f"{label} is not UTF-8 Markdown"
        ) from error
    first_content = next((line for line in lines if line.strip()), "")
    if not first_content.startswith("# "):
        raise Protocol131FreezeAuthorityError(
            f"{label} lacks a level-one Markdown header"
        )
    prefix = f"{field}:"
    values = []
    for line in lines:
        if line.startswith("## ") or line == "##":
            break
        if line.startswith(prefix):
            values.append(_semantic_markdown_value(line[len(prefix):]))
    if len(values) != 1:
        raise Protocol131FreezeAuthorityError(
            f"{label} header must contain exactly one {field} field"
        )
    return values[0]


def manifest_fingerprint(manifest):
    """Return the canonical structural fingerprint of one exact manifest.

    This helper validates entry schema, absolute paths, and digest syntax but
    intentionally does not read files.  Full validation re-hashes every entry
    before accepting a review binding.
    """
    if not isinstance(manifest, Mapping) or not manifest:
        raise Protocol131FreezeAuthorityError("manifest must be a nonempty mapping")
    if any(type(name) is not str or not name for name in manifest):
        raise Protocol131FreezeAuthorityError(
            "manifest names must be nonempty strings"
        )
    digest = hashlib.sha256()
    digest.update(b"Protocol-131-manifest-fingerprint-v1\0")
    for name in sorted(manifest):
        entry = manifest[name]
        _require_exact_keys(entry, _FILE_ENTRY_KEYS, f"manifest entry {name}")
        path = _require_absolute_path(entry["path"], f"manifest entry {name} path")
        sha256 = _require_lowercase_sha256(
            entry["sha256"], f"manifest entry {name} SHA-256"
        )
        for value in (name, str(path), sha256):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def _verified_manifest(manifest, label, occupied_paths):
    if not isinstance(manifest, Mapping) or not manifest:
        raise Protocol131FreezeAuthorityError(
            f"{label} must be a nonempty mapping"
        )
    if any(type(name) is not str or not name for name in manifest):
        raise Protocol131FreezeAuthorityError(
            f"{label} names must be nonempty strings"
        )
    verified = {}
    for name in sorted(manifest):
        path, sha256, _ = _read_verified_file(
            manifest[name], f"{label} entry {name}"
        )
        if path in occupied_paths:
            raise Protocol131FreezeAuthorityError(
                f"{label} entry {name} duplicates another recorded file"
            )
        occupied_paths.add(path)
        verified[name] = MappingProxyType({
            "path": str(path),
            "sha256": sha256,
        })
    return MappingProxyType(verified)


def _reject_duplicate_json_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise Protocol131FreezeAuthorityError(
                "independent review JSON contains a duplicate key"
            )
        result[key] = value
    return result


def _review_binding_from_bytes(payload):
    if payload.lstrip().startswith(b"{"):
        try:
            record = json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_json_keys,
            )
        except Protocol131FreezeAuthorityError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise Protocol131FreezeAuthorityError(
                "independent review is not valid UTF-8 JSON"
            ) from error
        _require_exact_keys(record, _REVIEW_BINDING_KEYS, "independent review JSON")
        if any(type(record[name]) is not str for name in _REVIEW_BINDING_KEYS):
            raise Protocol131FreezeAuthorityError(
                "independent review JSON bindings must be strings"
            )
        return dict(record)
    fields = {
        "verdict": "Verdict",
        "protocol_sha256": "Protocol-SHA256",
        "runtime_contract_sha256": "Runtime-Contract-SHA256",
        "source_manifest_sha256": "Source-Manifest-SHA256",
        "input_manifest_sha256": "Input-Manifest-SHA256",
    }
    return {
        name: _markdown_preamble_field(
            payload, field, "independent review"
        )
        for name, field in fields.items()
    }


def validate_protocol131_freeze_authority(freeze_record):
    """Validate a first-run Protocol-131 freeze and return immutable authority.

    The candidate output path must be absent at the instant of this call.  Any
    discrepancy raises :class:`Protocol131FreezeAuthorityError`; partial
    authority is never returned.
    """
    _require_exact_keys(freeze_record, _TOP_LEVEL_KEYS, "freeze record")
    if freeze_record["status"] != "FROZEN" or type(freeze_record["status"]) is not str:
        raise Protocol131FreezeAuthorityError(
            "freeze-record status must be exactly FROZEN"
        )
    if freeze_record["frozen_before_execution"] is not True or type(
        freeze_record["frozen_before_execution"]
    ) is not bool:
        raise Protocol131FreezeAuthorityError(
            "frozen_before_execution must be exactly true"
        )
    if freeze_record["candidate_output_state_at_freeze"] != "absent" or type(
        freeze_record["candidate_output_state_at_freeze"]
    ) is not str:
        raise Protocol131FreezeAuthorityError(
            "candidate output state at freeze must be exactly absent"
        )

    protocol_path, protocol_sha256, protocol_bytes = _read_verified_file(
        freeze_record["protocol"], "protocol"
    )
    if _markdown_preamble_field(protocol_bytes, "Status", "protocol") != "FROZEN":
        raise Protocol131FreezeAuthorityError(
            "protocol header status is not exactly FROZEN"
        )
    runtime_path, runtime_sha256, _ = _read_verified_file(
        freeze_record["runtime_contract"], "runtime contract"
    )
    occupied_paths = {protocol_path, runtime_path}
    if len(occupied_paths) != 2:
        raise Protocol131FreezeAuthorityError(
            "protocol and runtime contract must be distinct files"
        )

    source_manifest = _verified_manifest(
        freeze_record["source_manifest"], "source manifest", occupied_paths
    )
    input_manifest = _verified_manifest(
        freeze_record["input_manifest"], "input manifest", occupied_paths
    )
    source_manifest_sha256 = manifest_fingerprint(source_manifest)
    input_manifest_sha256 = manifest_fingerprint(input_manifest)

    review_entry = freeze_record["independent_review"]
    _require_exact_keys(review_entry, _REVIEW_ENTRY_KEYS, "independent review")
    if review_entry["verdict"] != "PASS" or type(review_entry["verdict"]) is not str:
        raise Protocol131FreezeAuthorityError(
            "recorded independent-review verdict must be exactly PASS"
        )
    review_path, review_sha256, review_bytes = _read_verified_file(
        {"path": review_entry["path"], "sha256": review_entry["sha256"]},
        "independent review",
    )
    if review_path in occupied_paths:
        raise Protocol131FreezeAuthorityError(
            "independent review duplicates another recorded file"
        )
    expected_review_binding = {
        "verdict": "PASS",
        "protocol_sha256": protocol_sha256,
        "runtime_contract_sha256": runtime_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "input_manifest_sha256": input_manifest_sha256,
    }
    if _review_binding_from_bytes(review_bytes) != expected_review_binding:
        raise Protocol131FreezeAuthorityError(
            "independent-review file does not bind the exact reviewed freeze inputs"
        )

    candidate_path = _require_absolute_path(
        freeze_record["candidate_output_directory"],
        "candidate output directory",
    )
    declared_candidate = _markdown_preamble_field(
        protocol_bytes, "Candidate-Output-Directory", "protocol",
    )
    if declared_candidate != str(candidate_path):
        raise Protocol131FreezeAuthorityError(
            "candidate output directory differs from the reviewed protocol"
        )
    try:
        if candidate_path.is_symlink() or candidate_path.exists():
            raise Protocol131FreezeAuthorityError(
                "candidate output directory must be absent on first validation"
            )
    except Protocol131FreezeAuthorityError:
        raise
    except OSError as error:
        raise Protocol131FreezeAuthorityError(
            "candidate output directory state cannot be inspected"
        ) from error

    return MappingProxyType({
        "authorization_kind": AUTHORIZATION_KIND,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "status": "FROZEN",
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "runtime_contract_path": str(runtime_path),
        "runtime_contract_sha256": runtime_sha256,
        "source_manifest": source_manifest,
        "source_manifest_sha256": source_manifest_sha256,
        "input_manifest": input_manifest,
        "input_manifest_sha256": input_manifest_sha256,
        "independent_review": MappingProxyType({
            "path": str(review_path),
            "sha256": review_sha256,
            **expected_review_binding,
        }),
        "frozen_before_execution": True,
        "candidate_output_directory": str(candidate_path),
        "candidate_output_state_at_freeze": "absent",
        "archive_only_postmortem_authorized": True,
        "new_parent_construction_authorized": False,
        "evolution_authorized": False,
    })


def revalidate_protocol131_freeze_authority_snapshot(authority):
    """Revalidate a first-run authority after its output directory exists.

    A raw freeze record can issue authority only while the candidate path is
    absent.  A restart cannot reproduce that historical observation, so it
    must use the exact validated snapshot written by the first run.  This
    function re-hashes every frozen source and input and rechecks every
    semantic binding, while preserving (rather than re-observing) the recorded
    ``absent`` state.  It cannot issue authority from a raw freeze record.
    """
    _require_exact_keys(
        authority, _VALIDATED_AUTHORITY_KEYS, "validated freeze authority",
    )
    if not (
        authority["authorization_kind"] == AUTHORIZATION_KIND
        and type(authority["authorization_kind"]) is str
        and authority["authorization_scope"] == AUTHORIZATION_SCOPE
        and type(authority["authorization_scope"]) is str
        and authority["status"] == "FROZEN"
        and type(authority["status"]) is str
        and authority["frozen_before_execution"] is True
        and type(authority["frozen_before_execution"]) is bool
        and authority["candidate_output_state_at_freeze"] == "absent"
        and type(authority["candidate_output_state_at_freeze"]) is str
        and authority["archive_only_postmortem_authorized"] is True
        and type(authority["archive_only_postmortem_authorized"]) is bool
        and authority["new_parent_construction_authorized"] is False
        and type(authority["new_parent_construction_authorized"]) is bool
        and authority["evolution_authorized"] is False
        and type(authority["evolution_authorized"]) is bool
    ):
        raise Protocol131FreezeAuthorityError(
            "validated freeze-authority scope or flags differ"
        )

    protocol_path, protocol_sha256, protocol_bytes = _read_verified_file(
        {
            "path": authority["protocol_path"],
            "sha256": authority["protocol_sha256"],
        },
        "validated-authority protocol",
    )
    if _markdown_preamble_field(
        protocol_bytes, "Status", "validated-authority protocol",
    ) != "FROZEN":
        raise Protocol131FreezeAuthorityError(
            "validated-authority protocol is no longer frozen"
        )
    runtime_path, runtime_sha256, _ = _read_verified_file(
        {
            "path": authority["runtime_contract_path"],
            "sha256": authority["runtime_contract_sha256"],
        },
        "validated-authority runtime contract",
    )
    occupied_paths = {protocol_path, runtime_path}
    if len(occupied_paths) != 2:
        raise Protocol131FreezeAuthorityError(
            "validated authority reuses protocol/runtime paths"
        )
    source_manifest = _verified_manifest(
        authority["source_manifest"], "validated source manifest", occupied_paths,
    )
    input_manifest = _verified_manifest(
        authority["input_manifest"], "validated input manifest", occupied_paths,
    )
    source_manifest_sha256 = manifest_fingerprint(source_manifest)
    input_manifest_sha256 = manifest_fingerprint(input_manifest)
    if (
        authority["source_manifest_sha256"] != source_manifest_sha256
        or type(authority["source_manifest_sha256"]) is not str
        or authority["input_manifest_sha256"] != input_manifest_sha256
        or type(authority["input_manifest_sha256"]) is not str
    ):
        raise Protocol131FreezeAuthorityError(
            "validated-authority manifest fingerprint differs"
        )

    review = authority["independent_review"]
    _require_exact_keys(
        review, _VALIDATED_REVIEW_KEYS, "validated-authority review",
    )
    if review["verdict"] != "PASS" or type(review["verdict"]) is not str:
        raise Protocol131FreezeAuthorityError(
            "validated-authority review verdict differs"
        )
    review_path, review_sha256, review_bytes = _read_verified_file(
        {"path": review["path"], "sha256": review["sha256"]},
        "validated-authority independent review",
    )
    expected_review_binding = {
        "verdict": "PASS",
        "protocol_sha256": protocol_sha256,
        "runtime_contract_sha256": runtime_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "input_manifest_sha256": input_manifest_sha256,
    }
    if (
        review_path in occupied_paths
        or _review_binding_from_bytes(review_bytes) != expected_review_binding
        or any(review[name] != value for name, value in expected_review_binding.items())
    ):
        raise Protocol131FreezeAuthorityError(
            "validated-authority independent-review binding differs"
        )

    candidate_path = _require_absolute_path(
        authority["candidate_output_directory"],
        "validated-authority candidate output directory",
    )
    declared_candidate = _markdown_preamble_field(
        protocol_bytes,
        "Candidate-Output-Directory",
        "validated-authority protocol",
    )
    if declared_candidate != str(candidate_path):
        raise Protocol131FreezeAuthorityError(
            "validated-authority candidate path differs from protocol"
        )
    return MappingProxyType({
        "authorization_kind": AUTHORIZATION_KIND,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "status": "FROZEN",
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "runtime_contract_path": str(runtime_path),
        "runtime_contract_sha256": runtime_sha256,
        "source_manifest": source_manifest,
        "source_manifest_sha256": source_manifest_sha256,
        "input_manifest": input_manifest,
        "input_manifest_sha256": input_manifest_sha256,
        "independent_review": MappingProxyType({
            "path": str(review_path),
            "sha256": review_sha256,
            **expected_review_binding,
        }),
        "frozen_before_execution": True,
        "candidate_output_directory": str(candidate_path),
        "candidate_output_state_at_freeze": "absent",
        "archive_only_postmortem_authorized": True,
        "new_parent_construction_authorized": False,
        "evolution_authorized": False,
    })
