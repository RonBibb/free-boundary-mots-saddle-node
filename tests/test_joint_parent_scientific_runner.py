from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

import bhps.joint_parent_scientific_runner as runner
from bhps.joint_parent_freeze_authority import (
    Protocol125FreezeAuthorityError,
    _manifest_fingerprint,
    validate_protocol125_freeze_authority,
)
from bhps.recovery_indexer import RecoveryIndex, sha256_file


def _digest(label):
    return hashlib.sha256(str(label).encode()).hexdigest()


def _write(path, value):
    path.write_text(value)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _freeze_record(tmp_path, *, status="FROZEN"):
    protocol = tmp_path/"protocol.md"
    adjudicator = tmp_path/"adjudicator.py"
    review = tmp_path/"review.md"
    adapter_source = Path(__file__).resolve()
    runner_source = Path(runner.__file__).resolve()
    protocol_entry = _write(
        protocol,
        f"# Manufactured Protocol 125\nStatus: **{status}**\n\n## Body\n",
    )
    adjudicator_entry = _write(adjudicator, "# frozen manufactured adjudicator\n")
    candidate = tmp_path/"candidate"
    source_manifest = {
        runner.RUNNER_MANIFEST_LOGICAL_NAME: {
            "path": str(runner_source),
            "sha256": sha256_file(runner_source),
        },
        "manufactured_test_adapter": {
            "path": str(adapter_source),
            "sha256": sha256_file(adapter_source),
        },
    }
    review_entry = _write(
        review,
        "# Independent manufactured review\n"
        "Verdict: **PASS**\n"
        f"Protocol-SHA256: {protocol_entry['sha256']}\n"
        f"Adjudicator-SHA256: {adjudicator_entry['sha256']}\n"
        f"Source-Manifest-SHA256: {_manifest_fingerprint(source_manifest)}\n\n"
        "## Body\n",
    )
    record = {
        "status": "FROZEN",
        "protocol": protocol_entry,
        "adjudicator": adjudicator_entry,
        "source_manifest": source_manifest,
        "independent_review": {**review_entry, "verdict": "PASS"},
        "frozen_before_parent_data": True,
        "scientific_candidates_absent_at_freeze": True,
        "candidate_output_directory": str(candidate.resolve()),
        "candidate_output_state_at_freeze": "absent",
    }
    return record, candidate


class _ManufacturedAdapter:
    def __init__(
        self,
        *,
        pre_failure=None,
        post_failure=None,
        two_failure=None,
        gate_overrides=None,
        ordered_stops=False,
        technical_abort=None,
    ):
        self.pre_failure = pre_failure
        self.post_failure = post_failure
        self.two_failure = two_failure
        self.gate_overrides = dict(gate_overrides or {})
        self.ordered_stops = bool(ordered_stops)
        self.technical_abort = technical_abort
        self.calls = []
        self.identities = {label: _digest(f"parent:{label}") for label in runner.PARENT_LABELS}

    def _records(self, kind, groups, bindings, failure):
        records = {}
        ordered_stop_started = False
        for group in groups:
            not_reached = bool(
                self.ordered_stops and ordered_stop_started
            )
            record = {
                "complete": True,
                "provenance_valid": True,
                "passed": group != failure and not not_reached,
                "fingerprint": _digest(f"{tuple(bindings.items())}:{group}"),
            }
            if not_reached:
                record.update({
                    "not_reached": True,
                    "blocked_by": failure,
                })
            if len(bindings) == 1:
                label = next(iter(bindings))
                record.update({
                    "parent_label": label,
                    "parent_identity": bindings[label],
                })
            else:
                record["parent_identities"] = dict(bindings)
                label = "two-parent"
            record.update(
                self.gate_overrides.get((kind, label, group), {})
            )
            records[group] = record
            if group == failure:
                ordered_stop_started = True
        return records

    @staticmethod
    def _checkpoint(kind, bindings, gate_records):
        if gate_records is None:
            record_sha256 = runner._tree_sha256(
                {"bindings": bindings, "stage_kind": kind},
                root="parent-stage-record",
            )
            passed = True
        else:
            record_sha256 = runner._tree_sha256(gate_records, root="gate-records")
            passed = all(record["passed"] for record in gate_records.values())
        arrays = {
            "binding_labels": np.asarray(tuple(bindings)),
            "binding_identities": np.asarray(tuple(bindings.values())),
            "all_scientific_gates_passed": np.asarray(passed),
        }
        return runner.Protocol125CheckpointPayload(
            arrays=arrays,
            metadata={
                "stage_kind": kind,
                "complete_state": True,
                "restartable_without_unrecorded_state": True,
                "record_sha256": record_sha256,
                # Exercise JSON-array metadata through write, immutable reload,
                # adapter reconstruction, and immediate restore validation.
                "binding_order": list(bindings),
            },
        )

    def _stage(self, kind, bindings, groups=None, failure=None):
        records = (
            None if groups is None
            else self._records(kind, groups, bindings, failure)
        )
        return runner.Protocol125RunnerStage(
            runtime={"kind": kind, "bindings": dict(bindings)},
            checkpoint=self._checkpoint(kind, bindings, records),
            bindings=dict(bindings),
            gate_records=records,
        )

    def construct_parent(self, label, *, freeze_authority):
        self.calls.append(f"parent:{label}")
        assert freeze_authority["status"] == "FROZEN"
        return self._stage("parent", {label: self.identities[label]})

    def compose_pre_acceleration(self, label, parent_stage, *, freeze_authority):
        self.calls.append(f"pre:{label}")
        assert parent_stage.runtime["kind"] == "parent"
        failure = self.pre_failure[1] if self.pre_failure and self.pre_failure[0] == label else None
        return self._stage(
            "pre-acceleration",
            {label: self.identities[label]},
            runner.PRE_ACCELERATION_GROUPS,
            failure,
        )

    def compose_post_acceleration(
        self, label, parent_stage, pre_acceleration_stage, *, freeze_authority,
    ):
        self.calls.append(f"post:{label}")
        if self.technical_abort == f"post:{label}":
            raise RuntimeError(f"manufactured technical abort at post:{label}")
        assert parent_stage.runtime["kind"] == "parent"
        assert pre_acceleration_stage.runtime["kind"] == "pre-acceleration"
        failure = self.post_failure[1] if self.post_failure and self.post_failure[0] == label else None
        return self._stage(
            "post-acceleration",
            {label: self.identities[label]},
            runner.POST_ACCELERATION_GROUPS,
            failure,
        )

    def compose_two_parent(
        self,
        parent_stages,
        pre_acceleration_stages,
        post_acceleration_stages,
        *,
        parent_identities,
        freeze_authority,
    ):
        self.calls.append("two-parent")
        assert tuple(parent_stages) == runner.PARENT_LABELS
        assert tuple(pre_acceleration_stages) == runner.PARENT_LABELS
        assert tuple(post_acceleration_stages) == runner.PARENT_LABELS
        return self._stage(
            "two-parent",
            dict(parent_identities),
            runner.TWO_PARENT_GROUPS,
            self.two_failure,
        )

    def restore_checkpoint(self, stage_id, archived, *, context):
        self.calls.append(f"restore:{stage_id}")
        labels = tuple(str(value) for value in archived["arrays"]["binding_labels"])
        identities = tuple(str(value) for value in archived["arrays"]["binding_identities"])
        bindings = dict(zip(labels, identities, strict=True))
        kind = str(archived["envelope"]["stage_kind"])
        if kind == "parent":
            return self._stage(kind, bindings)
        if kind == "pre-acceleration":
            label = next(iter(bindings))
            failure = self.pre_failure[1] if self.pre_failure and self.pre_failure[0] == label else None
            return self._stage(kind, bindings, runner.PRE_ACCELERATION_GROUPS, failure)
        if kind == "post-acceleration":
            label = next(iter(bindings))
            failure = self.post_failure[1] if self.post_failure and self.post_failure[0] == label else None
            return self._stage(kind, bindings, runner.POST_ACCELERATION_GROUPS, failure)
        if kind == "two-parent":
            return self._stage(kind, bindings, runner.TWO_PARENT_GROUPS, self.two_failure)
        raise AssertionError(kind)


def _adapters(adapter):
    implementation = Path(__file__).resolve()
    return runner.Protocol125ScientificAdapters(
        identifier="manufactured-complete-runner-adapter-v1",
        implementation_path=str(implementation),
        implementation_sha256=sha256_file(implementation),
        source_manifest_name="manufactured_test_adapter",
        capabilities=runner.REQUIRED_ADAPTER_CAPABILITIES,
        construct_parent=adapter.construct_parent,
        compose_pre_acceleration=adapter.compose_pre_acceleration,
        compose_post_acceleration=adapter.compose_post_acceleration,
        compose_two_parent=adapter.compose_two_parent,
        restore_checkpoint=adapter.restore_checkpoint,
    )


def test_no_frozen_protocol_means_no_parent_callback_and_no_output(tmp_path):
    freeze, candidate = _freeze_record(tmp_path, status="DRAFT")
    adapter = _ManufacturedAdapter()
    with pytest.raises(Protocol125FreezeAuthorityError, match="not exactly FROZEN"):
        runner.run_protocol125_scientific(
            freeze_record=freeze,
            adapters=_adapters(adapter),
        )
    assert adapter.calls == []
    assert not candidate.exists()


def test_unregistered_production_adapter_fails_before_output(tmp_path):
    freeze, candidate = _freeze_record(tmp_path)
    with pytest.raises(runner.Protocol125AdapterBlocker, match="no Protocol-125"):
        runner.run_protocol125_scientific(freeze_record=freeze)
    readiness = runner.production_adapter_readiness()
    assert readiness["ready"]
    assert readiness["adapter_implementation_ready"]
    assert not readiness["blockers"]
    assert readiness["prospective_freeze_blockers"]
    assert not readiness["scientific_run_ready"]
    assert not candidate.exists()


def test_pre_acceleration_failure_stops_before_every_acceleration_callback(tmp_path):
    freeze, candidate = _freeze_record(tmp_path)
    adapter = _ManufacturedAdapter(
        pre_failure=("N1", "dense_boundary_audit"),
    )
    result = runner.run_protocol125_scientific(
        freeze_record=freeze,
        adapters=_adapters(adapter),
    )
    assert result["ordered_adjudication"]["classification"] == "FAIL-parent-position"
    assert not result["phase_a_authorized"]
    assert not any(call.startswith("post:") for call in adapter.calls)
    assert "two-parent" not in adapter.calls
    stages = __import__("json").loads(
        (candidate/"recovery_index.json").read_text()
    )["stages"]
    assert not any(name.startswith("post-acceleration/") for name in stages)
    assert "two-parent/comparison" not in stages


def test_recovery_hash_tamper_is_a_hard_failure(tmp_path):
    freeze, candidate = _freeze_record(tmp_path)
    authority = validate_protocol125_freeze_authority(freeze)
    adapter = _ManufacturedAdapter(
        pre_failure=("N0", "bulk_prerequisite"),
    )
    runner.run_protocol125_scientific(
        freeze_record=freeze,
        adapters=_adapters(adapter),
    )
    index = RecoveryIndex(
        candidate/"recovery_index.json",
        authority["protocol_path"],
        runner._authority_inputs(authority),
        maximum_stage_seconds=43200.0,
    )
    checkpoint = candidate/"parent_N0.npz"
    checkpoint.write_bytes(checkpoint.read_bytes()+b"tamper")
    with pytest.raises(runner.Protocol125RecoveryError, match="hash/byte"):
        runner.reload_protocol125_recovery_checkpoint(index, "parent/N0")


def test_only_complete_ordered_pass_authorizes_separate_phase_a(tmp_path):
    freeze, candidate = _freeze_record(tmp_path)
    adapter = _ManufacturedAdapter()
    result = runner.run_protocol125_scientific(
        freeze_record=freeze,
        adapters=_adapters(adapter),
    )
    ordered = result["ordered_adjudication"]
    assert ordered["classification"] == "PASS-native-joint-parent"
    assert result["phase_a_authorized"]
    assert not result["phase_a_executed"]
    assert not result["rhs_rk_phase_b_full_matrix_authorized"]
    assert not result["rhs_rk_phase_b_full_matrix_executed"]
    assert not result["interface_physics_authorized"]
    assert adapter.calls.count("two-parent") == 1
    assert (candidate/"adjudication_final.json").is_file()


def test_public_recovery_revalidates_prior_authority_and_restores_without_producers(
    tmp_path,
):
    freeze, candidate = _freeze_record(tmp_path)
    authority = validate_protocol125_freeze_authority(freeze)
    first_adapter = _ManufacturedAdapter()
    first = runner.run_protocol125_scientific(
        freeze_record=freeze,
        adapters=_adapters(first_adapter),
    )

    # The raw prospective record cannot be replayed after candidate bytes
    # exist.  Recovery is a separate, explicit authority path.
    with pytest.raises(Protocol125FreezeAuthorityError, match="does not match"):
        runner.run_protocol125_scientific(
            freeze_record=freeze,
            adapters=_adapters(_ManufacturedAdapter()),
        )

    recovery_adapter = _ManufacturedAdapter()
    second = runner.run_protocol125_scientific(
        freeze_authority=authority,
        adapters=_adapters(recovery_adapter),
    )
    assert second == first
    assert recovery_adapter.calls
    assert all(call.startswith("restore:") for call in recovery_adapter.calls)
    assert (candidate/"recovery_index.json").is_file()


def test_post_failure_cannot_reach_two_parent_or_authorize_phase_a(tmp_path):
    freeze, _ = _freeze_record(tmp_path)
    adapter = _ManufacturedAdapter(
        post_failure=("N0", "endpoint_derivatives"),
    )
    result = runner.run_protocol125_scientific(
        freeze_record=freeze,
        adapters=_adapters(adapter),
    )
    assert result["ordered_adjudication"]["classification"] == "FAIL-acceleration"
    assert not result["phase_a_authorized"]
    assert "two-parent" not in adapter.calls


def test_incomplete_reached_gate_aborts_before_checkpoint_or_final(tmp_path):
    freeze, candidate = _freeze_record(tmp_path)
    group = runner.PRE_ACCELERATION_GROUPS[0]
    adapter = _ManufacturedAdapter(gate_overrides={
        ("pre-acceleration", "N0", group): {"complete": False},
    })

    with pytest.raises(
        runner.Protocol125RunnerError,
        match="reached gate is incomplete",
    ):
        runner.run_protocol125_scientific(
            freeze_record=freeze,
            adapters=_adapters(adapter),
        )

    assert not (candidate/"pre-acceleration_N0.npz").exists()
    assert not (candidate/"adjudication_final.json").exists()
    stages = __import__("json").loads(
        (candidate/"recovery_index.json").read_text()
    )["stages"]
    assert stages["pre-acceleration/N0"]["status"] == "failed"
    assert "adjudication/final" not in stages


def test_invalid_provenance_on_restore_aborts_without_final(tmp_path):
    freeze, candidate = _freeze_record(tmp_path)
    authority = validate_protocol125_freeze_authority(freeze)
    interrupted = _ManufacturedAdapter(technical_abort="post:N0")
    with pytest.raises(RuntimeError, match="manufactured technical abort"):
        runner.run_protocol125_scientific(
            freeze_record=freeze,
            adapters=_adapters(interrupted),
        )
    assert (candidate/"pre-acceleration_N0.npz").is_file()
    assert not (candidate/"adjudication_final.json").exists()

    group = runner.PRE_ACCELERATION_GROUPS[1]
    invalid_restore = _ManufacturedAdapter(gate_overrides={
        ("pre-acceleration", "N0", group): {
            "provenance_valid": False,
        },
    })
    with pytest.raises(
        runner.Protocol125RunnerError,
        match="reached gate has invalid provenance",
    ):
        runner.run_protocol125_scientific(
            freeze_authority=authority,
            adapters=_adapters(invalid_restore),
        )
    assert not (candidate/"adjudication_final.json").exists()
    assert "restore:pre-acceleration/N0" in invalid_restore.calls


def test_complete_scientific_failure_with_not_reached_records_restores_and_classifies(
    tmp_path,
):
    freeze, candidate = _freeze_record(tmp_path)
    authority = validate_protocol125_freeze_authority(freeze)
    failed_group = runner.PRE_ACCELERATION_GROUPS[0]
    first_adapter = _ManufacturedAdapter(
        pre_failure=("N0", failed_group),
        ordered_stops=True,
    )
    first = runner.run_protocol125_scientific(
        freeze_record=freeze,
        adapters=_adapters(first_adapter),
    )
    assert first["ordered_adjudication"]["classification"] == "FAIL-parent-bulk"
    assert (candidate/"pre-acceleration_N0.npz").is_file()
    assert (candidate/"adjudication_final.json").is_file()

    recovery_adapter = _ManufacturedAdapter(
        pre_failure=("N0", failed_group),
        ordered_stops=True,
    )
    second = runner.run_protocol125_scientific(
        freeze_authority=authority,
        adapters=_adapters(recovery_adapter),
    )
    assert second == first
    assert recovery_adapter.calls
    assert all(call.startswith("restore:") for call in recovery_adapter.calls)
