"""Immutable, fail-closed gate ledger for Protocol 125.

This module only binds already produced gate decisions to one prospective
protocol-freeze record and to the two parent identities.  It cannot execute a
scorer, construct or repair a parent, write an artifact, or advance an
experiment.  Appending a gate returns a new ledger, so a recorded gate can
never be overwritten in place.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from bhps.joint_parent_adjudication import (
    PARENT_GATE_GROUPS,
    TWO_PARENT_GATE_GROUPS,
    classify_protocol125_gate_records,
)


_PARENT_LABELS = ("N0", "N1")
_GATE_FIELDS = ("complete", "provenance_valid", "passed", "fingerprint")


def _is_sha256(value):
    value = str(value)
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _freeze_value(value):
    """Take an immutable defensive snapshot of simple record data."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    raise TypeError("ledger records may contain only immutable scalar/container data")


def _snapshot_mapping(record, label):
    if not isinstance(record, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return _freeze_value(record)


def _validate_parent_identities(parent_identities):
    if not isinstance(parent_identities, Mapping) or set(parent_identities) != set(
        _PARENT_LABELS
    ):
        raise ValueError("parent identities must be exactly N0 and N1")
    identities = {label: str(parent_identities[label]) for label in _PARENT_LABELS}
    for label, fingerprint in identities.items():
        if not _is_sha256(fingerprint):
            raise ValueError(f"{label} parent identity must be a lowercase SHA-256 digest")
    if identities["N0"] == identities["N1"]:
        raise ValueError("N0 and N1 parent identities must be distinct")
    return MappingProxyType(identities)


@dataclass(frozen=True)
class _GateEntry:
    complete: bool
    provenance_valid: bool
    passed: bool
    fingerprint: str
    parent_identities: tuple[tuple[str, str], ...]

    def classifier_record(self):
        return MappingProxyType({
            "complete": self.complete,
            "provenance_valid": self.provenance_valid,
            "passed": self.passed,
            "fingerprint": self.fingerprint,
        })

    def public_record(self):
        bindings = dict(self.parent_identities)
        if len(bindings) == 1:
            parent_label, parent_identity = next(iter(bindings.items()))
            identity_record = {
                "parent_label": parent_label,
                "parent_identity": parent_identity,
            }
        else:
            identity_record = {
                "parent_identities": MappingProxyType(bindings),
            }
        return MappingProxyType({
            **dict(self.classifier_record()),
            **identity_record,
        })


def _gate_core(record, label):
    if not isinstance(record, Mapping):
        raise TypeError(f"{label} gate record must be a mapping")
    missing = tuple(name for name in _GATE_FIELDS if name not in record)
    if missing:
        raise ValueError(f"{label} gate record is missing: {', '.join(missing)}")
    for name in ("complete", "provenance_valid", "passed"):
        if type(record[name]) is not bool:
            raise TypeError(f"{label} {name} must be a bool")
    fingerprint = str(record["fingerprint"])
    if not _is_sha256(fingerprint):
        raise ValueError(f"{label} fingerprint must be a lowercase SHA-256 digest")
    return (
        record["complete"],
        record["provenance_valid"],
        record["passed"],
        fingerprint,
    )


class Protocol125GateLedger:
    """Persistent append-once ledger for the exhaustive Protocol-125 gates."""

    __slots__ = (
        "_protocol_freeze_record",
        "_parent_identities",
        "_parent_records",
        "_two_parent_records",
    )

    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError("Protocol125GateLedger snapshots are immutable")
        object.__setattr__(self, name, value)

    def __init__(self, protocol_freeze_record, parent_identities):
        self._protocol_freeze_record = _snapshot_mapping(
            protocol_freeze_record, "protocol freeze record",
        )
        self._parent_identities = _validate_parent_identities(parent_identities)
        self._parent_records = MappingProxyType({
            label: MappingProxyType({}) for label in _PARENT_LABELS
        })
        self._two_parent_records = MappingProxyType({})

    @classmethod
    def _from_parts(
        cls,
        protocol_freeze_record,
        parent_identities,
        parent_records,
        two_parent_records,
    ):
        ledger = object.__new__(cls)
        ledger._protocol_freeze_record = protocol_freeze_record
        ledger._parent_identities = parent_identities
        ledger._parent_records = MappingProxyType({
            label: MappingProxyType(dict(parent_records[label]))
            for label in _PARENT_LABELS
        })
        ledger._two_parent_records = MappingProxyType(dict(two_parent_records))
        return ledger

    @property
    def protocol_freeze_record(self):
        return self._protocol_freeze_record

    @property
    def parent_identities(self):
        return self._parent_identities

    @property
    def parent_records(self):
        return MappingProxyType({
            label: MappingProxyType({
                name: entry.public_record()
                for name, entry in self._parent_records[label].items()
            })
            for label in _PARENT_LABELS
        })

    @property
    def two_parent_records(self):
        return MappingProxyType({
            name: entry.public_record()
            for name, entry in self._two_parent_records.items()
        })

    def append_parent_gate(self, parent_label, group_name, record):
        """Return a new ledger containing one parent-local gate decision."""
        parent_label = str(parent_label)
        group_name = str(group_name)
        if parent_label not in _PARENT_LABELS:
            raise ValueError("parent gate label must be N0 or N1")
        if group_name not in PARENT_GATE_GROUPS:
            raise ValueError("unknown Protocol-125 parent gate group")
        if group_name in self._parent_records[parent_label]:
            raise ValueError(f"duplicate parent gate: {parent_label}:{group_name}")
        core = _gate_core(record, f"{parent_label}:{group_name}")
        if str(record.get("parent_label", "")) != parent_label:
            raise ValueError(f"{parent_label}:{group_name} parent label is mismatched")
        identity = str(record.get("parent_identity", ""))
        if identity != self._parent_identities[parent_label]:
            raise ValueError(f"{parent_label}:{group_name} parent identity is mismatched")
        entry = _GateEntry(
            *core,
            parent_identities=((parent_label, identity),),
        )
        parent_records = {
            label: dict(self._parent_records[label]) for label in _PARENT_LABELS
        }
        parent_records[parent_label][group_name] = entry
        return self._from_parts(
            self._protocol_freeze_record,
            self._parent_identities,
            parent_records,
            self._two_parent_records,
        )

    def append_two_parent_gate(self, group_name, record):
        """Return a new ledger containing one N0/N1 comparison gate decision."""
        group_name = str(group_name)
        if group_name not in TWO_PARENT_GATE_GROUPS:
            raise ValueError("unknown Protocol-125 two-parent gate group")
        if group_name in self._two_parent_records:
            raise ValueError(f"duplicate two-parent gate: {group_name}")
        core = _gate_core(record, f"two-parent:{group_name}")
        supplied = record.get("parent_identities")
        if not isinstance(supplied, Mapping) or set(supplied) != set(_PARENT_LABELS):
            raise ValueError(
                f"two-parent:{group_name} identities must be exactly N0 and N1"
            )
        identities = tuple(
            (label, str(supplied[label])) for label in _PARENT_LABELS
        )
        if dict(identities) != dict(self._parent_identities):
            raise ValueError(f"two-parent:{group_name} parent identities are mismatched")
        entry = _GateEntry(*core, parent_identities=identities)
        two_parent_records = dict(self._two_parent_records)
        two_parent_records[group_name] = entry
        return self._from_parts(
            self._protocol_freeze_record,
            self._parent_identities,
            self._parent_records,
            two_parent_records,
        )

    def finalize(self):
        """Classify the immutable snapshot; absent or invalid inputs fail closed."""
        parent_records = {
            label: {
                name: entry.classifier_record()
                for name, entry in self._parent_records[label].items()
            }
            for label in _PARENT_LABELS
        }
        two_parent_records = {
            name: entry.classifier_record()
            for name, entry in self._two_parent_records.items()
        }
        return classify_protocol125_gate_records(
            parent_records,
            two_parent_records,
            self._protocol_freeze_record,
        )
