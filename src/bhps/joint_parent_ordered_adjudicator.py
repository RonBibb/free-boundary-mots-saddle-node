"""Ordered, fail-closed master classification for Protocol 125.

The exhaustive ledger is appropriate only after every gate has executed.  A
scientific runner, however, must stop before acceleration when either parent
fails a prerequisite and before the two-parent comparison when either final
parent fails.  This module gives those intentionally absent downstream groups
their only valid interpretation: *not reached because an earlier scientific
gate failed*.  Missing evidence at a reached stage remains ``INVALID-audit``.

No scorer or solver is called here and no artifact or authorization is
created.  The result authorizes Phase A only when all ordered groups exist and
pass under a prospectively frozen protocol record.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from bhps.joint_parent_adjudication import (
    PARENT_GATE_GROUPS,
    TWO_PARENT_GATE_GROUPS,
)


PROTOCOL_IDENTIFIER = "Protocol-125-ordered-master-adjudicator-v1"
PARENT_LABELS = ("N0", "N1")
PRE_ACCELERATION_GROUPS = (
    "pre_acceleration_construction",
    "native_position_tangent",
    "position_representation",
    "dense_boundary_audit",
    "signature_union",
    "legacy_holdout",
    "sampling_order",
    "bulk_prerequisite",
)
POST_ACCELERATION_GROUPS = (
    "acceleration_closure",
    "wall_algebra",
    "final_representation",
    "endpoint_derivatives",
    "correction_size",
)
TWO_PARENT_GROUPS = (
    "N0_N1_representation",
    "correction_refinement",
)
_CORE_KEYS = ("complete", "provenance_valid", "passed", "fingerprint")


def _valid_sha256(value):
    value = str(value)
    return bool(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({str(name): _freeze(item) for name, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _hash_tree(value):
    digest = hashlib.sha256()

    def visit(item, path):
        encoded = str(path).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
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
            raise ValueError("ordered adjudicator cannot hash object data")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())

    visit(value, "ordered-adjudication")
    return digest.hexdigest()


def _freeze_valid(record):
    if not isinstance(record, Mapping):
        return False, "missing-protocol-freeze-record"
    if str(record.get("status", "")) != "FROZEN":
        return False, "protocol-is-not-prospectively-frozen"
    for flag in (
        "frozen_before_parent_data",
        "independent_review_passed",
        "scientific_candidates_absent_at_freeze",
    ):
        if type(record.get(flag)) is not bool or not record[flag]:
            return False, f"invalid-freeze-flag:{flag}"
    for name in ("protocol_sha256", "adjudicator_sha256"):
        if not _valid_sha256(record.get(name, "")):
            return False, f"invalid-{name}"
    return True, ""


def _identities_valid(parent_identities):
    if not isinstance(parent_identities, Mapping) or tuple(parent_identities) != PARENT_LABELS:
        return False
    values = tuple(str(parent_identities[label]) for label in PARENT_LABELS)
    return bool(
        all(_valid_sha256(value) for value in values)
        and values[0] != values[1]
    )


def _record_state(record, *, label, parent_label=None, parent_identities=None):
    if not isinstance(record, Mapping):
        return None, f"{label}:missing-record"
    missing = tuple(name for name in _CORE_KEYS if name not in record)
    if missing:
        return None, f"{label}:missing-core:{','.join(missing)}"
    if any(type(record[name]) is not bool for name in _CORE_KEYS[:3]):
        return None, f"{label}:nonboolean-core"
    if not _valid_sha256(record["fingerprint"]):
        return None, f"{label}:invalid-fingerprint"
    if not record["complete"]:
        return None, f"{label}:incomplete"
    if not record["provenance_valid"]:
        return None, f"{label}:invalid-provenance"
    if parent_label is not None:
        if (
            str(record.get("parent_label", "")) != parent_label
            or str(record.get("parent_identity", ""))
            != str(parent_identities[parent_label])
        ):
            return None, f"{label}:parent-binding-mismatch"
    else:
        supplied = record.get("parent_identities")
        if not isinstance(supplied, Mapping) or any(
            str(supplied.get(parent, "")) != str(parent_identities[parent])
            for parent in PARENT_LABELS
        ):
            return None, f"{label}:parent-binding-mismatch"
    not_reached = record.get("not_reached", False)
    if type(not_reached) is not bool:
        return None, f"{label}:nonboolean-not-reached"
    if not_reached:
        if record["passed"]:
            return None, f"{label}:not-reached-record-passed"
        blocked_by = record.get("blocked_by")
        if not isinstance(blocked_by, str) or not blocked_by:
            return None, f"{label}:not-reached-without-blocker"
        return ("not-reached", blocked_by), ""
    if "blocked_by" in record:
        return None, f"{label}:blocker-on-reached-record"
    return ("pass" if record["passed"] else "fail", None), ""


def _validate_stage_records(
    records,
    groups,
    *,
    stage,
    parent_identities,
    two_parent=False,
):
    invalid = []
    failures = {"position": [], "bulk": [], "acceleration": []}
    if two_parent:
        if not isinstance(records, Mapping) or tuple(records) != groups:
            return failures, (f"{stage}:record-set-differs",)
        for group in groups:
            state, reason = _record_state(
                records[group],
                label=f"two-parent:{group}",
                parent_identities=parent_identities,
            )
            if reason:
                invalid.append(reason)
            elif state[0] == "not-reached":
                invalid.append(f"two-parent:{group}:not-reached-is-not-permitted")
            elif state[0] == "fail":
                failures[TWO_PARENT_GATE_GROUPS[group]].append(
                    f"two-parent:{group}"
                )
        return failures, tuple(invalid)

    if not isinstance(records, Mapping) or tuple(records) != PARENT_LABELS:
        return failures, (f"{stage}:parent-record-set-differs",)
    for parent in PARENT_LABELS:
        parent_records = records[parent]
        if not isinstance(parent_records, Mapping) or tuple(parent_records) != groups:
            invalid.append(f"{stage}:{parent}:group-set-differs")
            continue
        first_failure = None
        ordered_stop_started = False
        for group in groups:
            state, reason = _record_state(
                parent_records[group],
                label=f"{parent}:{group}",
                parent_label=parent,
                parent_identities=parent_identities,
            )
            if reason:
                invalid.append(reason)
                continue
            state_name, blocked_by = state
            if state_name == "not-reached":
                if first_failure is None:
                    invalid.append(f"{parent}:{group}:not-reached-before-failure")
                elif blocked_by != first_failure:
                    invalid.append(
                        f"{parent}:{group}:blocked-by-mismatch:"
                        f"expected-{first_failure}:received-{blocked_by}"
                    )
                ordered_stop_started = True
                continue
            if ordered_stop_started:
                invalid.append(f"{parent}:{group}:reached-after-ordered-stop")
                continue
            if state_name == "fail":
                if first_failure is None:
                    first_failure = group
                failures[PARENT_GATE_GROUPS[group]].append(f"{parent}:{group}")
    return failures, tuple(invalid)


def _result(
    classification,
    *,
    reached_stage,
    parent_identities,
    invalid_reasons=(),
    failures=None,
    freeze_record=None,
):
    failures = failures or {"position": [], "bulk": [], "acceleration": []}
    payload = {
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "classification": str(classification),
        "reached_stage": str(reached_stage),
        "parent_identities": dict(parent_identities),
        "invalid_reasons": tuple(invalid_reasons),
        "failed_position_groups": tuple(failures["position"]),
        "failed_bulk_groups": tuple(failures["bulk"]),
        "failed_acceleration_groups": tuple(failures["acceleration"]),
        "phase_a_authorized": classification == "PASS-native-joint-parent",
        "rhs_rk_phase_b_full_matrix_authorized": False,
        "interface_physics_authorized": False,
        "downstream_absence_is_ordered_stop": classification in (
            "FAIL-parent-position", "FAIL-parent-bulk", "FAIL-acceleration"
        ),
        "freeze_protocol_sha256": str(
            freeze_record.get("protocol_sha256", "") if isinstance(freeze_record, Mapping) else ""
        ),
        "freeze_adjudicator_sha256": str(
            freeze_record.get("adjudicator_sha256", "") if isinstance(freeze_record, Mapping) else ""
        ),
    }
    return _freeze({**payload, "fingerprint": _hash_tree(payload)})


def adjudicate_protocol125_ordered(
    pre_acceleration_records,
    *,
    parent_identities,
    protocol_freeze_record,
    post_acceleration_records=None,
    two_parent_records=None,
):
    """Classify the furthest legitimately reached Protocol-125 stage.

    ``None`` is accepted for downstream records only when an earlier complete
    scientific gate has failed.  Supplying a later stage after such a failure
    is an operation-order violation and therefore ``INVALID-audit``.
    """
    freeze_ok, freeze_reason = _freeze_valid(protocol_freeze_record)
    if not freeze_ok:
        return _result(
            "INVALID-audit",
            reached_stage="freeze",
            parent_identities=parent_identities if isinstance(parent_identities, Mapping) else {},
            invalid_reasons=(freeze_reason,),
            freeze_record=protocol_freeze_record,
        )
    if not _identities_valid(parent_identities):
        return _result(
            "INVALID-audit",
            reached_stage="identity",
            parent_identities=parent_identities if isinstance(parent_identities, Mapping) else {},
            invalid_reasons=("parent-identities-invalid",),
            freeze_record=protocol_freeze_record,
        )

    pre_failures, invalid = _validate_stage_records(
        pre_acceleration_records,
        PRE_ACCELERATION_GROUPS,
        stage="pre-acceleration",
        parent_identities=parent_identities,
    )
    if invalid:
        return _result(
            "INVALID-audit",
            reached_stage="pre-acceleration",
            parent_identities=parent_identities,
            invalid_reasons=invalid,
            failures=pre_failures,
            freeze_record=protocol_freeze_record,
        )
    if any(pre_failures.values()):
        if post_acceleration_records is not None or two_parent_records is not None:
            return _result(
                "INVALID-audit",
                reached_stage="operation-order",
                parent_identities=parent_identities,
                invalid_reasons=("downstream-records-exist-after-prerequisite-failure",),
                failures=pre_failures,
                freeze_record=protocol_freeze_record,
            )
        classification = (
            "FAIL-parent-position"
            if pre_failures["position"] else "FAIL-parent-bulk"
        )
        return _result(
            classification,
            reached_stage="pre-acceleration",
            parent_identities=parent_identities,
            failures=pre_failures,
            freeze_record=protocol_freeze_record,
        )

    post_failures, invalid = _validate_stage_records(
        post_acceleration_records,
        POST_ACCELERATION_GROUPS,
        stage="post-acceleration",
        parent_identities=parent_identities,
    )
    if invalid:
        return _result(
            "INVALID-audit",
            reached_stage="post-acceleration",
            parent_identities=parent_identities,
            invalid_reasons=invalid,
            failures=post_failures,
            freeze_record=protocol_freeze_record,
        )
    if any(post_failures.values()):
        if two_parent_records is not None:
            return _result(
                "INVALID-audit",
                reached_stage="operation-order",
                parent_identities=parent_identities,
                invalid_reasons=("two-parent-records-exist-after-acceleration-failure",),
                failures=post_failures,
                freeze_record=protocol_freeze_record,
            )
        return _result(
            "FAIL-acceleration",
            reached_stage="post-acceleration",
            parent_identities=parent_identities,
            failures=post_failures,
            freeze_record=protocol_freeze_record,
        )

    two_failures, invalid = _validate_stage_records(
        two_parent_records,
        TWO_PARENT_GROUPS,
        stage="two-parent",
        parent_identities=parent_identities,
        two_parent=True,
    )
    if invalid:
        return _result(
            "INVALID-audit",
            reached_stage="two-parent",
            parent_identities=parent_identities,
            invalid_reasons=invalid,
            failures=two_failures,
            freeze_record=protocol_freeze_record,
        )
    if two_failures["bulk"]:
        classification = "FAIL-parent-bulk"
    elif two_failures["acceleration"]:
        classification = "FAIL-acceleration"
    else:
        classification = "PASS-native-joint-parent"
    return _result(
        classification,
        reached_stage="two-parent",
        parent_identities=parent_identities,
        failures=two_failures,
        freeze_record=protocol_freeze_record,
    )
