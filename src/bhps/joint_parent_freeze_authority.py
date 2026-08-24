"""Read-only, fail-closed freeze authority for Protocol 125.

The validator in this module verifies one explicit freeze-record mapping
against the bytes that are present on disk.  It cannot edit or freeze a
protocol, create a candidate directory, construct a parent, run a scorer, or
authorize scientific execution.  A successful call returns only a deeply
immutable record that may be supplied to the Protocol-125 gate ledger.

The input mapping has this exact shape::

    {
        "status": "FROZEN",
        "protocol": {"path": ABSOLUTE_PATH, "sha256": LOWERCASE_SHA256},
        "adjudicator": {"path": ABSOLUTE_PATH, "sha256": LOWERCASE_SHA256},
        "source_manifest": {
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
        "frozen_before_parent_data": True,
        "scientific_candidates_absent_at_freeze": True,
        "candidate_output_directory": ABSOLUTE_PATH,
        "candidate_output_state_at_freeze": "absent" | "empty",
    }

The protocol is a Markdown document whose preamble contains exactly one
``Status:`` field, with the semantic value ``FROZEN``.  The independent
review may be JSON or Markdown, but in both cases it must record ``PASS`` and
bind the exact protocol, adjudicator, and complete source-manifest digests it
reviewed.  A generic PASS note is never freeze authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType


AUTHORIZATION_KIND = "Protocol-125-read-only-freeze-authority-v1"
AUTHORIZATION_SCOPE = "Protocol-125-gate-adjudication-only"

_AUTHORITY_KEYS = frozenset({
    "authorization_kind",
    "authorization_scope",
    "status",
    "protocol_path",
    "protocol_sha256",
    "adjudicator_path",
    "adjudicator_sha256",
    "source_manifest",
    "independent_review",
    "frozen_before_parent_data",
    "independent_review_passed",
    "scientific_candidates_absent_at_freeze",
    "candidate_output_directory",
    "candidate_output_state_at_freeze",
    "scientific_execution_authorized",
})

_TOP_LEVEL_KEYS = frozenset({
    "status",
    "protocol",
    "adjudicator",
    "source_manifest",
    "independent_review",
    "frozen_before_parent_data",
    "scientific_candidates_absent_at_freeze",
    "candidate_output_directory",
    "candidate_output_state_at_freeze",
})
_FILE_ENTRY_KEYS = frozenset({"path", "sha256"})
_REVIEW_ENTRY_KEYS = frozenset({"path", "sha256", "verdict"})
_VALIDATED_REVIEW_KEYS = frozenset({
    "path", "sha256", "verdict", "protocol_sha256",
    "adjudicator_sha256", "source_manifest_sha256",
})
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class Protocol125FreezeAuthorityError(ValueError):
    """Raised when no Protocol-125 freeze authority can be established."""


def _require_exact_keys(record, expected, label):
    if not isinstance(record, Mapping):
        raise Protocol125FreezeAuthorityError(f"{label} must be a mapping")
    found = frozenset(record)
    if found != expected:
        missing = tuple(sorted(expected-found))
        unexpected = tuple(sorted(found-expected, key=str))
        detail = []
        if missing:
            detail.append(f"missing {', '.join(missing)}")
        if unexpected:
            detail.append(
                "unexpected " + ", ".join(str(name) for name in unexpected)
            )
        raise Protocol125FreezeAuthorityError(
            f"{label} has invalid keys: {'; '.join(detail)}"
        )


def _require_lowercase_sha256(value, label):
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise Protocol125FreezeAuthorityError(
            f"{label} must be a lowercase SHA-256 digest"
        )
    return value


def _require_absolute_path(value, label):
    if type(value) is not str or not value:
        raise Protocol125FreezeAuthorityError(f"{label} must be an absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise Protocol125FreezeAuthorityError(f"{label} must be an absolute path")
    return path


def _read_verified_file(entry, label):
    _require_exact_keys(entry, _FILE_ENTRY_KEYS, label)
    path = _require_absolute_path(entry["path"], f"{label} path")
    expected = _require_lowercase_sha256(entry["sha256"], f"{label} SHA-256")
    try:
        if path.is_symlink() or not path.is_file():
            raise Protocol125FreezeAuthorityError(
                f"{label} path is not a regular non-symlink file"
            )
        resolved = path.resolve(strict=True)
        payload = path.read_bytes()
    except Protocol125FreezeAuthorityError:
        raise
    except OSError as error:
        raise Protocol125FreezeAuthorityError(f"{label} file cannot be read") from error
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise Protocol125FreezeAuthorityError(f"{label} SHA-256 does not match file bytes")
    return resolved, expected, payload


def _semantic_markdown_value(raw_value):
    value = raw_value.strip()
    if value.startswith("**") and value.endswith("**") and len(value) >= 4:
        value = value[2:-2]
    return value


def _markdown_preamble_field(payload, field, label):
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Protocol125FreezeAuthorityError(f"{label} is not UTF-8 Markdown") from error
    lines = text.splitlines()
    first_content = next((line for line in lines if line.strip()), "")
    if not first_content.startswith("# "):
        raise Protocol125FreezeAuthorityError(
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
        raise Protocol125FreezeAuthorityError(
            f"{label} header must contain exactly one {field} field"
        )
    return values[0]


def _review_binding_from_bytes(payload, label):
    stripped = payload.lstrip()
    if stripped.startswith(b"{"):
        def reject_duplicate_keys(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise Protocol125FreezeAuthorityError(
                        f"{label} JSON contains a duplicate key"
                    )
                result[key] = value
            return result

        try:
            record = json.loads(
                payload.decode("utf-8"), object_pairs_hook=reject_duplicate_keys,
            )
        except Protocol125FreezeAuthorityError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise Protocol125FreezeAuthorityError(
                f"{label} is not valid UTF-8 JSON"
            ) from error
        required = (
            "verdict", "protocol_sha256", "adjudicator_sha256",
            "source_manifest_sha256",
        )
        if not isinstance(record, Mapping) or any(
            type(record.get(name)) is not str for name in required
        ):
            raise Protocol125FreezeAuthorityError(
                f"{label} JSON lacks the exact review bindings"
            )
        return {name: record[name] for name in required}
    return {
        "verdict": _markdown_preamble_field(payload, "Verdict", label),
        "protocol_sha256": _markdown_preamble_field(
            payload, "Protocol-SHA256", label,
        ),
        "adjudicator_sha256": _markdown_preamble_field(
            payload, "Adjudicator-SHA256", label,
        ),
        "source_manifest_sha256": _markdown_preamble_field(
            payload, "Source-Manifest-SHA256", label,
        ),
    }


def _manifest_fingerprint(source_manifest):
    digest = hashlib.sha256()
    for name in sorted(source_manifest):
        entry = source_manifest[name]
        for value in (name, entry["path"], entry["sha256"]):
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def _verified_manifest(source_manifest, occupied_paths):
    if not isinstance(source_manifest, Mapping) or not source_manifest:
        raise Protocol125FreezeAuthorityError(
            "source manifest must be a nonempty mapping"
        )
    if any(type(name) is not str or not name for name in source_manifest):
        raise Protocol125FreezeAuthorityError(
            "source manifest names must be nonempty strings"
        )
    verified = {}
    for name in sorted(source_manifest):
        path, digest, _ = _read_verified_file(
            source_manifest[name], f"source manifest entry {name}",
        )
        if path in occupied_paths:
            raise Protocol125FreezeAuthorityError(
                f"source manifest entry {name} duplicates another recorded file"
            )
        occupied_paths.add(path)
        verified[name] = MappingProxyType({
            "path": str(path),
            "sha256": digest,
        })
    return MappingProxyType(verified)


def _observed_candidate_directory_state(path):
    if path.is_symlink():
        raise Protocol125FreezeAuthorityError(
            "candidate output directory may not be a symlink"
        )
    try:
        if not path.exists():
            return "absent"
        if not path.is_dir():
            raise Protocol125FreezeAuthorityError(
                "candidate output path exists but is not a directory"
            )
        return "empty" if next(path.iterdir(), None) is None else "nonempty"
    except Protocol125FreezeAuthorityError:
        raise
    except OSError as error:
        raise Protocol125FreezeAuthorityError(
            "candidate output directory state cannot be inspected"
        ) from error


def validate_protocol125_freeze_authority(freeze_record):
    """Verify one prospective freeze record and return an immutable authority.

    No partial authority is returned.  Any schema, status, digest, review,
    flag, path, or candidate-directory discrepancy raises
    :class:`Protocol125FreezeAuthorityError`.
    """
    _require_exact_keys(freeze_record, _TOP_LEVEL_KEYS, "freeze record")
    if type(freeze_record["status"]) is not str or freeze_record["status"] != "FROZEN":
        raise Protocol125FreezeAuthorityError(
            "freeze-record status must be exactly FROZEN"
        )
    for flag in (
        "frozen_before_parent_data",
        "scientific_candidates_absent_at_freeze",
    ):
        if type(freeze_record[flag]) is not bool or freeze_record[flag] is not True:
            raise Protocol125FreezeAuthorityError(f"{flag} must be exactly true")

    protocol_path, protocol_sha256, protocol_bytes = _read_verified_file(
        freeze_record["protocol"], "protocol",
    )
    protocol_status = _markdown_preamble_field(
        protocol_bytes, "Status", "protocol",
    )
    if protocol_status != "FROZEN":
        raise Protocol125FreezeAuthorityError(
            "protocol header status is not exactly FROZEN"
        )

    adjudicator_path, adjudicator_sha256, _ = _read_verified_file(
        freeze_record["adjudicator"], "adjudicator",
    )
    occupied_paths = {protocol_path, adjudicator_path}
    if len(occupied_paths) != 2:
        raise Protocol125FreezeAuthorityError(
            "protocol and adjudicator must be distinct files"
        )
    source_manifest = _verified_manifest(
        freeze_record["source_manifest"], occupied_paths,
    )

    review_entry = freeze_record["independent_review"]
    _require_exact_keys(review_entry, _REVIEW_ENTRY_KEYS, "independent review")
    if type(review_entry["verdict"]) is not str or review_entry["verdict"] != "PASS":
        raise Protocol125FreezeAuthorityError(
            "recorded independent-review verdict must be exactly PASS"
        )
    review_path, review_sha256, review_bytes = _read_verified_file(
        {"path": review_entry["path"], "sha256": review_entry["sha256"]},
        "independent review",
    )
    if review_path in occupied_paths:
        raise Protocol125FreezeAuthorityError(
            "independent review duplicates another recorded file"
        )
    manifest_sha256 = _manifest_fingerprint(source_manifest)
    review_binding = _review_binding_from_bytes(
        review_bytes, "independent review",
    )
    if review_binding != {
        "verdict": "PASS",
        "protocol_sha256": protocol_sha256,
        "adjudicator_sha256": adjudicator_sha256,
        "source_manifest_sha256": manifest_sha256,
    }:
        raise Protocol125FreezeAuthorityError(
            "independent-review file does not bind the exact reviewed freeze inputs"
        )

    candidate_path = _require_absolute_path(
        freeze_record["candidate_output_directory"],
        "candidate output directory",
    )
    recorded_candidate_state = freeze_record["candidate_output_state_at_freeze"]
    if type(recorded_candidate_state) is not str or recorded_candidate_state not in {
        "absent", "empty",
    }:
        raise Protocol125FreezeAuthorityError(
            "candidate output state at freeze must be exactly absent or empty"
        )
    observed_candidate_state = _observed_candidate_directory_state(candidate_path)
    if observed_candidate_state != recorded_candidate_state:
        raise Protocol125FreezeAuthorityError(
            "recorded candidate output state does not match the directory"
        )

    return MappingProxyType({
        "authorization_kind": AUTHORIZATION_KIND,
        "authorization_scope": AUTHORIZATION_SCOPE,
        "status": "FROZEN",
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "adjudicator_path": str(adjudicator_path),
        "adjudicator_sha256": adjudicator_sha256,
        "source_manifest": source_manifest,
        "independent_review": MappingProxyType({
            "path": str(review_path),
            "sha256": review_sha256,
            "verdict": "PASS",
            "protocol_sha256": protocol_sha256,
            "adjudicator_sha256": adjudicator_sha256,
            "source_manifest_sha256": manifest_sha256,
        }),
        "frozen_before_parent_data": True,
        "independent_review_passed": True,
        "scientific_candidates_absent_at_freeze": True,
        "candidate_output_directory": str(candidate_path),
        "candidate_output_state_at_freeze": recorded_candidate_state,
        "scientific_execution_authorized": False,
    })


def revalidate_protocol125_freeze_authority_snapshot(authority):
    """Revalidate a previously issued authority after checkpoint creation.

    The prospective validator above must be called before the first candidate
    byte is created.  Once a recovery directory contains checkpoints, its
    historical ``absent``/``empty`` observation can no longer be reproduced.
    This function therefore re-hashes every frozen file and rechecks every
    semantic flag in the validator's immutable output, but intentionally does
    not reinterpret the *current* candidate-directory contents as the state at
    freeze time.  It cannot issue an authority from a raw freeze record.
    """
    _require_exact_keys(authority, _AUTHORITY_KEYS, "validated freeze authority")
    if not (
        authority["authorization_kind"] == AUTHORIZATION_KIND
        and authority["authorization_scope"] == AUTHORIZATION_SCOPE
        and authority["status"] == "FROZEN"
        and authority["frozen_before_parent_data"] is True
        and authority["independent_review_passed"] is True
        and authority["scientific_candidates_absent_at_freeze"] is True
        and authority["scientific_execution_authorized"] is False
    ):
        raise Protocol125FreezeAuthorityError(
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
        raise Protocol125FreezeAuthorityError(
            "validated-authority protocol is no longer frozen"
        )
    adjudicator_path, adjudicator_sha256, _ = _read_verified_file(
        {
            "path": authority["adjudicator_path"],
            "sha256": authority["adjudicator_sha256"],
        },
        "validated-authority adjudicator",
    )
    occupied_paths = {protocol_path, adjudicator_path}
    if len(occupied_paths) != 2:
        raise Protocol125FreezeAuthorityError(
            "validated authority reuses protocol/adjudicator paths"
        )
    source_manifest = _verified_manifest(
        authority["source_manifest"], occupied_paths,
    )
    review = authority["independent_review"]
    _require_exact_keys(
        review, _VALIDATED_REVIEW_KEYS, "validated-authority review",
    )
    if review["verdict"] != "PASS":
        raise Protocol125FreezeAuthorityError(
            "validated-authority review verdict differs"
        )
    review_path, review_sha256, review_bytes = _read_verified_file(
        {"path": review["path"], "sha256": review["sha256"]},
        "validated-authority review",
    )
    manifest_sha256 = _manifest_fingerprint(source_manifest)
    expected_review_binding = {
        "verdict": "PASS",
        "protocol_sha256": protocol_sha256,
        "adjudicator_sha256": adjudicator_sha256,
        "source_manifest_sha256": manifest_sha256,
    }
    if (
        review_path in occupied_paths
        or _review_binding_from_bytes(
            review_bytes, "validated-authority review",
        ) != expected_review_binding
        or any(
            str(review[name]) != expected
            for name, expected in expected_review_binding.items()
        )
    ):
        raise Protocol125FreezeAuthorityError(
            "validated-authority review is duplicated or no longer passes"
        )
    candidate_path = _require_absolute_path(
        authority["candidate_output_directory"],
        "validated-authority candidate output directory",
    )
    candidate_state = authority["candidate_output_state_at_freeze"]
    if candidate_state not in {"absent", "empty"}:
        raise Protocol125FreezeAuthorityError(
            "validated-authority historical candidate state differs"
        )
    return MappingProxyType({
        **dict(authority),
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "adjudicator_path": str(adjudicator_path),
        "adjudicator_sha256": adjudicator_sha256,
        "source_manifest": source_manifest,
        "independent_review": MappingProxyType({
            "path": str(review_path),
            "sha256": review_sha256,
            **expected_review_binding,
        }),
        "candidate_output_directory": str(candidate_path),
    })
