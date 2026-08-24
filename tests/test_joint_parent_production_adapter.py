from __future__ import annotations

import json
import struct
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import bhps.joint_parent_production_adapter as production
import bhps.joint_parent_scientific_runner as runner
import bhps.joint_parent_construction as construction
import bhps.joint_parent_acceleration as acceleration
import bhps.joint_parent_representation as representation
from bhps.joint_parent_freeze_authority import (
    _manifest_fingerprint,
    validate_protocol125_freeze_authority,
)
from bhps.joint_parent_postacceleration import POST_ACCELERATION_GROUPS
from bhps.joint_parent_preacceleration import (
    INPUT_HASH_KEYS as PRE_ACCELERATION_INPUT_HASH_KEYS,
    PRE_ACCELERATION_GROUPS,
    PROTOCOL_IDENTIFIER as PRE_ACCELERATION_PROTOCOL_IDENTIFIER,
    _gate_record as _preacceleration_gate_record,
)
from bhps.matched_staged_continuum import hash_arrays
from bhps.recovery_indexer import sha256_file


def _entry(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _freeze_record(tmp_path, inventory, *, omit=None):
    protocol = tmp_path/"protocol.md"
    adjudicator = tmp_path/"adjudicator.py"
    review = tmp_path/"review.md"
    protocol.write_text(
        "# Manufactured Protocol 125\nStatus: **FROZEN**\n\n## Body\n",
        encoding="utf-8",
    )
    adjudicator.write_text("# manufactured adjudicator\n", encoding="utf-8")
    manifest = {
        name: _entry(path) for name, path in inventory.items() if name != omit
    }
    review.write_text(
        "# Independent manufactured review\n"
        "Verdict: **PASS**\n"
        f"Protocol-SHA256: {_entry(protocol)['sha256']}\n"
        f"Adjudicator-SHA256: {_entry(adjudicator)['sha256']}\n"
        f"Source-Manifest-SHA256: {_manifest_fingerprint(manifest)}\n\n"
        "## Body\n",
        encoding="utf-8",
    )
    return {
        "status": "FROZEN",
        "protocol": _entry(protocol),
        "adjudicator": _entry(adjudicator),
        "source_manifest": manifest,
        "independent_review": {**_entry(review), "verdict": "PASS"},
        "frozen_before_parent_data": True,
        "scientific_candidates_absent_at_freeze": True,
        "candidate_output_directory": str((tmp_path/"candidate").resolve()),
        "candidate_output_state_at_freeze": "absent",
    }


def _assert_array_bits_equal(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    assert left.dtype == right.dtype
    assert left.shape == right.shape
    assert left.tobytes() == right.tobytes()


def _manufactured_passed_pre_acceleration_result(parent_identity):
    context = {"label": "N0", "identity": parent_identity}
    groups = {
        name: _preacceleration_gate_record(
            name,
            context,
            complete=True,
            provenance_valid=True,
            passed=True,
            details={"manufactured": name},
        )
        for name in PRE_ACCELERATION_GROUPS
    }
    hashes = {
        name: f"{index + 1:064x}"
        for index, name in enumerate(PRE_ACCELERATION_INPUT_HASH_KEYS)
    }
    return {
        "protocol_identifier": PRE_ACCELERATION_PROTOCOL_IDENTIFIER,
        "classification": "PASS-single-parent-pre-acceleration",
        "complete": True,
        "provenance_valid": True,
        "passed": True,
        "parent_label": "N0",
        "parent_identity": parent_identity,
        "required_group_order": PRE_ACCELERATION_GROUPS,
        "groups": groups,
        "invalid_reasons": (),
        "input_hashes_before": hashes,
        "input_hashes_after": dict(hashes),
        "inputs_stable_while_scoring": True,
        "single_parent_only": True,
        "second_parent_and_common_V2_still_required": True,
        "acceleration_evaluated": False,
        "acceleration_authorized": False,
        "scientific_execution_authorized": False,
        "artifact_written": False,
    }


def _manufactured_ordered_pre_failure(parent_identity, failed_group):
    context = {"label": "N0", "identity": parent_identity}
    groups = {}
    failed_index = PRE_ACCELERATION_GROUPS.index(failed_group)
    for index, name in enumerate(PRE_ACCELERATION_GROUPS):
        groups[name] = _preacceleration_gate_record(
            name,
            context,
            complete=True,
            provenance_valid=True,
            passed=index < failed_index,
            details={"manufactured": name},
            not_reached=index > failed_index,
            blocked_by=failed_group if index > failed_index else None,
        )
    return {
        "classification": "FAIL-single-parent-pre-acceleration",
        "complete": True,
        "provenance_valid": True,
        "passed": False,
        "parent_label": "N0",
        "parent_identity": parent_identity,
        "groups": groups,
    }


class _ManufacturedPositionState:
    def fingerprint(self):
        return "9"*64


def _manufactured_fixed_point_failure(
    parent, position_state, bulk_acceleration,
):
    z = np.asarray(parent["z"])
    r = np.asarray(parent["r"])
    current = np.asarray(bulk_acceleration)
    label, identity, fixed, input_provenance, attempt = (
        acceleration._input_provenance(
            position_state,
            np.asarray(parent["position"]),
            current,
            z,
            r,
            parent["background"],
            parent_label="N0",
            parent_identity=parent["parent_identity"],
        )
    )
    history = tuple({
        "map": index,
        "acceleration_scaled_Linf_change": 0.1,
        "source_triplet_scaled_Linf_change": 0.2,
        "consecutive_converged_maps": 0,
        "coupled": {"map": index, "passed": True},
        "selective": {"map": index, "passed": True},
        "normal_tangential_correction_scaled_Linf": 0.0,
        "axis_reconciliation_scaled_Linf": 0.0,
        "axis_reconciliation": {"map": index, "passed": True},
    } for index in range(1, 9))
    source_triplet = {
        name: np.zeros((len(z), len(r), 3))
        for name in ("source", "source_time", "source_second_time")
    }
    event = {
        "owner": "fixed_point",
        "failed_map": None,
        "exception_type": None,
        "message": "manufactured fixed-point nonconvergence",
        "gate": "two_consecutive_map_convergence",
        "radial_index": None,
        "radius": None,
        "field": None,
        "diagnostics": {
            "maximum_maps": 8,
            "required_consecutive": 2,
            "observed_consecutive": 0,
            "convergence_tolerance": 1e-12,
            "final_acceleration_scaled_Linf_change": 0.1,
            "final_source_triplet_scaled_Linf_change": 0.2,
        },
    }
    return acceleration._acceleration_failure_record(
        parent_label=label,
        parent_identity=identity,
        attempt_fingerprint=attempt,
        fixed_settings=fixed,
        input_provenance=input_provenance,
        failure_group="acceleration_closure",
        failure_reason="fixed_point_nonconvergence",
        history=history,
        consecutive_converged_maps=0,
        current=current,
        source_triplet=source_triplet,
        failure_event=event,
        coupled=history[-1]["coupled"],
        selective=history[-1]["selective"],
        axis_reconciliation=history[-1]["axis_reconciliation"],
    )


def test_lossless_codec_roundtrips_types_and_ieee_payloads_without_pickle(tmp_path):
    nan_payload = np.asarray([0x7FF8000000000042], dtype=np.uint64).view(np.float64)
    roots = {
        "mixed": {
            "float_array": np.asarray([-0.0, np.inf, -np.inf]),
            "nan_payload": nan_payload,
            "big_endian": np.asarray([1.25, -2.5], dtype=">f8"),
            "unicode": np.asarray(["alpha", "β"], dtype="U5"),
            "numpy_scalar": np.uint64(2**63+7),
            "python_float": -0.0,
            "python_complex": complex(-0.0, np.inf),
            "bytes": b"\x00\xffProtocol-125",
            "tuple": (1, True, None),
            "list": ["x", 3],
        }
    }
    arrays, metadata = production._pack_roots(roots)
    assert arrays
    assert all(value.dtype == np.uint8 and value.ndim == 1 for value in arrays.values())
    assert metadata == json.loads(json.dumps(metadata))
    assert metadata["codec_root_order"] == ["mixed"]

    path = tmp_path/"codec.npz"
    np.savez(path, **arrays)
    with np.load(path, allow_pickle=False) as archive:
        reloaded_arrays = {name: archive[name] for name in archive.files}
    found = production._unpack_roots(reloaded_arrays, metadata)

    assert production._tree_bitwise_digest(found) == production._tree_bitwise_digest(roots)
    mixed = found["mixed"]
    _assert_array_bits_equal(mixed["float_array"], roots["mixed"]["float_array"])
    _assert_array_bits_equal(mixed["nan_payload"], nan_payload)
    _assert_array_bits_equal(mixed["big_endian"], roots["mixed"]["big_endian"])
    _assert_array_bits_equal(mixed["unicode"], roots["mixed"]["unicode"])
    assert isinstance(mixed["numpy_scalar"], np.uint64)
    assert struct.pack("!d", mixed["python_float"]) == struct.pack("!d", -0.0)
    assert struct.pack("!dd", mixed["python_complex"].real, mixed["python_complex"].imag) == struct.pack(
        "!dd", -0.0, np.inf,
    )
    assert type(mixed["tuple"]) is tuple
    assert type(mixed["list"]) is list


def test_lossless_codec_rejects_byte_tamper_and_nonraw_payload():
    arrays, metadata = production._pack_roots({"value": np.arange(9, dtype=np.float64)})
    tampered = {name: value.copy() for name, value in arrays.items()}
    raw_name = next(name for name in tampered if name != "codec_manifest_utf8")
    tampered[raw_name][0] ^= np.uint8(1)
    with pytest.raises(production.Protocol125ProductionAdapterError, match="payload hash"):
        production._unpack_roots(tampered, metadata)

    nonraw = dict(arrays)
    nonraw[raw_name] = np.asarray([1.0])
    with pytest.raises(production.Protocol125ProductionAdapterError, match="non-raw"):
        production._unpack_roots(nonraw, metadata)


def test_production_checkpoint_rejects_incomplete_reached_scorer_evidence():
    identity = "a"*64
    result = _manufactured_passed_pre_acceleration_result(identity)
    groups = {
        name: dict(record) for name, record in result["groups"].items()
    }
    groups[PRE_ACCELERATION_GROUPS[0]]["complete"] = False
    with pytest.raises(
        runner.Protocol125RunnerError,
        match="reached gate is incomplete",
    ):
        production._checkpoint(
            "pre-acceleration",
            {"N0": identity},
            groups,
            {"pre_result": result},
        )


def test_parent_checkpoint_roundtrip_and_tamper_rejection_are_exact():
    z = np.asarray([1.0, np.e])
    r = np.asarray([0.0, 12.0])
    position = np.zeros((2, 2, 9))
    parent = {
        "label": "N0",
        "z": z,
        "r": r,
        "position": position,
        "selector_q": np.zeros((2, 2)),
        "phi": np.zeros((2, 2)),
        "reference_q": np.zeros((2, 2)),
        "reference_phi": np.zeros((2, 2)),
        "background": {"method": "manufactured", "signed_zero": -0.0},
    }
    identity = production._parent_identity(parent)
    parent["parent_identity"] = identity
    bindings = {"N0": identity}
    payload = production._checkpoint("parent", bindings, None, {"parent": parent})
    archived = {
        "metadata": payload.metadata,
        "arrays": payload.arrays,
        "envelope": {"stage_kind": "parent", "bindings": bindings},
    }
    adapter = production.Protocol125ProductionAdapter()
    restored = adapter.restore_checkpoint(
        "parent/N0", archived, context={"freeze_authority": {"validated": True}},
    )
    assert restored.bindings == bindings
    production._require_tree_equal(restored.runtime.parent, parent, "parent roundtrip")

    tampered = dict(archived)
    tampered_arrays = {name: value.copy() for name, value in payload.arrays.items()}
    raw_name = next(name for name in tampered_arrays if name != "codec_manifest_utf8")
    tampered_arrays[raw_name][0] ^= np.uint8(1)
    tampered["arrays"] = tampered_arrays
    with pytest.raises(production.Protocol125ProductionAdapterError, match="payload hash"):
        adapter.restore_checkpoint(
            "parent/N0", tampered, context={"freeze_authority": {"validated": True}},
        )


def test_scientific_construction_failure_is_checkpointed_composed_and_restored(
    monkeypatch,
):
    z = np.linspace(1.0, np.e, 3)
    r = np.linspace(0.0, 12.0, 4)
    zeros = np.zeros((len(z), len(r)))
    monkeypatch.setitem(
        construction.PARENT_SPECS,
        "N0",
        {
            "nz": len(z), "nr": len(r), "reference_iterations": 5,
            "coordinate_sha256": hash_arrays(z, r),
        },
    )
    reference = {
        "z": z, "r": r, "q": zeros, "phi": zeros,
        "history": [1e-3, 2e-9], "converged": False,
        "max_abs_residual": 2e-9, "residual_l2": 1e-9,
    }
    failure = construction._construction_failure_record(
        "N0", "a"*64, reference,
        failure_gate="finite_wall_reference",
        measured_value=2e-9,
        ceiling=construction.FINITE_WALL_REFERENCE_CEILING,
    )

    def fail_parent(*args, **kwargs):
        raise construction.Protocol125ScientificConstructionFailure(failure)

    monkeypatch.setattr(production, "construct_joint_parent_position", fail_parent)
    adapter = production.Protocol125ProductionAdapter()
    parent = adapter.construct_parent("N0", freeze_authority={"manufactured": True})
    assert parent.runtime.parent is None
    assert parent.runtime.construction_failure["fingerprint"] == failure["fingerprint"]
    assert parent.bindings == {"N0": failure["parent_identity"]}

    pre = adapter.compose_pre_acceleration(
        "N0", parent, freeze_authority={"manufactured": True},
    )
    assert not pre.runtime.pre_result["passed"]
    first = pre.gate_records[runner.PRE_ACCELERATION_GROUPS[0]]
    assert first["passed"] is False
    assert first.get("not_reached", False) is False
    assert all(
        pre.gate_records[name]["not_reached"] is True
        for name in runner.PRE_ACCELERATION_GROUPS[1:]
    )

    def archive(stage, kind):
        return {
            "metadata": stage.checkpoint.metadata,
            "arrays": stage.checkpoint.arrays,
            "envelope": {"stage_kind": kind, "bindings": dict(stage.bindings)},
        }

    restored_parent = adapter.restore_checkpoint(
        "parent/N0", archive(parent, "parent"),
        context={"freeze_authority": {"manufactured": True}},
    )
    restored_pre = adapter.restore_checkpoint(
        "pre-acceleration/N0", archive(pre, "pre-acceleration"),
        context={
            "freeze_authority": {"manufactured": True},
            "parent_stage": restored_parent,
        },
    )
    production._require_tree_equal(
        restored_pre.runtime.pre_result, pre.runtime.pre_result,
        "construction-failure pre restart",
    )


@pytest.mark.parametrize("failure_site", ("position", "reference"))
def test_representation_coefficient_failure_stops_and_restores_without_retry(
    monkeypatch,
    failure_site,
):
    z = np.linspace(1.0, np.e, 7)
    r = np.linspace(0.0, 12.0, 7)
    scalar = np.zeros((len(z), len(r)))
    position = np.zeros((len(z), len(r), 9))
    parent = {
        "label": "N0",
        "z": z,
        "r": r,
        "position": position,
        "selector_q": scalar.copy(),
        "phi": scalar.copy(),
        "reference_q": scalar.copy(),
        "reference_phi": scalar.copy(),
        "background": {},
    }
    identity = production._parent_identity(parent)
    parent["parent_identity"] = identity
    parent["construction_provenance_record"] = (
        construction.build_protocol125_successful_parent_provenance_record(
            "N0",
            identity,
            "a"*64,
            finite_wall_maximum_residual=1e-12,
            joint_hybrid_maximum_residual=2e-12,
        )
    )
    parent_stage = runner.Protocol125RunnerStage(
        production.Protocol125ProductionParentState(parent, None),
        checkpoint=None,
        bindings={"N0": identity},
        gate_records=None,
    )
    with pytest.raises(
        representation.Protocol125RepresentationCoefficientFailure,
    ) as captured:
        representation.raise_if_nonfinite_protocol125_representation_coefficients(
            np.asarray([[[np.inf]]]),
            recipe=(
                "native-radial-cubic-s"
                if failure_site == "position"
                else "finite-wall-reference-Q53-compact"
            ),
            input_arrays={"manufactured_finite_input": np.asarray([1.0])},
        )
    failure = captured.value
    calls = []

    monkeypatch.setattr(
        production,
        "derive_joint_parent_position_outer_contract",
        lambda parent: object(),
    )

    def build_position(*args, **kwargs):
        calls.append("position")
        if failure_site == "position":
            raise failure
        return object(), {"manufactured": "position"}

    monkeypatch.setattr(
        production, "build_joint_parent_position_state", build_position,
    )
    pair = SimpleNamespace(
        coefficient_arrays=lambda prefix: {
            f"{prefix}_values": np.asarray([1.0]),
        },
    )
    monkeypatch.setattr(
        production,
        "PositionOnlyConstrainedHermitePair",
        SimpleNamespace(from_primary=lambda state: pair),
    )

    class ReferenceFactory:
        @staticmethod
        def build(*args, **kwargs):
            calls.append("reference")
            raise failure

    monkeypatch.setattr(
        production, "FiniteWallReferenceHermitePair", ReferenceFactory,
    )

    def forbidden_later(*args, **kwargs):
        raise AssertionError("later work ran after representation failure")

    monkeypatch.setattr(
        production,
        "build_protocol125_native_position_tangent_evidence",
        forbidden_later,
    )
    monkeypatch.setattr(
        production,
        "build_protocol125_preacceleration_legacy_position_inputs",
        forbidden_later,
    )
    monkeypatch.setattr(
        production, "run_protocol125_bulk_validation", forbidden_later,
    )

    adapter = production.Protocol125ProductionAdapter()
    pre = adapter.compose_pre_acceleration(
        "N0", parent_stage, freeze_authority={"manufactured": True},
    )
    assert isinstance(
        pre.runtime,
        production.Protocol125ProductionRepresentationFailurePreState,
    )
    assert pre.runtime.representation_coefficient_failure[
        "parent_identity"
    ] == identity
    assert pre.gate_records["pre_acceleration_construction"]["passed"] is False
    assert all(
        pre.gate_records[name]["not_reached"] is True
        for name in PRE_ACCELERATION_GROUPS[1:]
    )
    roots = production._unpack_roots(
        pre.checkpoint.arrays,
        pre.checkpoint.metadata,
    )
    assert tuple(roots) == (
        "representation_coefficient_failure", "pre_result",
    )
    calls_before_restore = tuple(calls)
    archived = {
        "metadata": pre.checkpoint.metadata,
        "arrays": pre.checkpoint.arrays,
        "envelope": {
            "stage_kind": "pre-acceleration",
            "bindings": dict(pre.bindings),
        },
    }
    restored = adapter.restore_checkpoint(
        "pre-acceleration/N0",
        archived,
        context={
            "freeze_authority": {"manufactured": True},
            "parent_stage": parent_stage,
        },
    )
    assert tuple(calls) == calls_before_restore
    production._require_tree_equal(
        restored.runtime.pre_result,
        pre.runtime.pre_result,
        "representation-failure pre restart",
    )

    wrong_parent = dict(parent)
    wrong_parent["position"] = position.copy()
    wrong_parent["position"][0, 0, 0] = 1.0
    wrong_identity = production._parent_identity(wrong_parent)
    wrong_parent["parent_identity"] = wrong_identity
    wrong_stage = runner.Protocol125RunnerStage(
        production.Protocol125ProductionParentState(wrong_parent, None),
        checkpoint=None,
        bindings={"N0": wrong_identity},
        gate_records=None,
    )
    with pytest.raises(
        production.Protocol125ProductionAdapterError,
        match="parent binding differs",
    ):
        adapter.restore_checkpoint(
            "pre-acceleration/N0",
            archived,
            context={
                "freeze_authority": {"manufactured": True},
                "parent_stage": wrong_stage,
            },
        )


@pytest.mark.parametrize(
    ("stop_kind", "failed_group"),
    (
        ("position", "signature_union"),
        ("representation", "legacy_holdout"),
    ),
)
def test_production_pre_staging_never_calls_bulk_after_earlier_failure_and_restarts(
    monkeypatch, stop_kind, failed_group,
):
    z = np.asarray([1.0, np.e])
    r = np.asarray([0.0, 12.0])
    position = np.zeros((2, 2, 9))
    scalar = np.zeros((2, 2))
    parent = {
        "label": "N0",
        "z": z,
        "r": r,
        "position": position,
        "selector_q": scalar,
        "phi": scalar,
        "reference_q": scalar,
        "reference_phi": scalar,
        "background": {},
        "construction_provenance_record": {"manufactured": True},
    }
    identity = production._parent_identity(parent)
    parent["parent_identity"] = identity
    parent_stage = runner.Protocol125RunnerStage(
        production.Protocol125ProductionParentState(parent, None),
        checkpoint=None,
        bindings={"N0": identity},
        gate_records=None,
    )

    class Pair:
        def coefficient_arrays(self, prefix):
            return {f"{prefix}_values": np.asarray([1.0])}

    class Reference:
        def coefficient_arrays(self, prefix):
            return {f"{prefix}_values": np.asarray([2.0])}

    pair = Pair()
    reference = Reference()
    calls = []
    monkeypatch.setattr(
        production, "derive_joint_parent_position_outer_contract",
        lambda parent: object(),
    )
    monkeypatch.setattr(
        production, "build_joint_parent_position_state",
        lambda *args, **kwargs: (object(), {"manufactured": "position"}),
    )
    monkeypatch.setattr(
        production, "PositionOnlyConstrainedHermitePair",
        SimpleNamespace(from_primary=lambda state: pair),
    )
    monkeypatch.setattr(
        production, "FiniteWallReferenceHermitePair",
        SimpleNamespace(build=lambda *args: reference),
    )
    monkeypatch.setattr(
        production, "build_protocol125_native_position_tangent_evidence",
        lambda *args: {"manufactured": "native"},
    )
    monkeypatch.setattr(
        production, "capture_protocol125_position_prefix_provenance",
        lambda inputs: {"manufactured": "position-provenance"},
    )
    position_prefix = {
        "passed": stop_kind != "position",
        "manufactured": "position-prefix",
    }
    monkeypatch.setattr(
        production, "evaluate_protocol125_position_prefix",
        lambda inputs, provenance: position_prefix,
    )
    legacy = {
        "legacy_Q33_by_mesh": {},
        "legacy_Q55_by_mesh": {},
        "component_orders": {},
    }

    def build_legacy(*args, **kwargs):
        calls.append("legacy")
        return legacy

    monkeypatch.setattr(
        production,
        "build_protocol125_preacceleration_legacy_position_inputs",
        build_legacy,
    )
    monkeypatch.setattr(
        production, "capture_protocol125_legacy_sampling_provenance",
        lambda inputs, prefix: {"manufactured": "representation-provenance"},
    )
    representation_prefix = {
        "passed": False,
        "manufactured": "representation-prefix",
    }
    monkeypatch.setattr(
        production, "extend_protocol125_legacy_sampling",
        lambda inputs, prefix, provenance: representation_prefix,
    )

    def forbidden_bulk(*args, **kwargs):
        calls.append("bulk")
        raise AssertionError("bulk ran after an earlier failure")

    monkeypatch.setattr(
        production, "run_protocol125_bulk_validation", forbidden_bulk,
    )
    stopped_result = _manufactured_ordered_pre_failure(
        identity, failed_group,
    )
    monkeypatch.setattr(
        production, "finalize_protocol125_preacceleration_stop",
        lambda staged: stopped_result,
    )

    adapter = production.Protocol125ProductionAdapter()
    pre = adapter.compose_pre_acceleration(
        "N0", parent_stage, freeze_authority={"manufactured": True},
    )
    assert "bulk" not in calls
    assert calls == ([] if stop_kind == "position" else ["legacy"])
    assert pre.gate_records[failed_group]["passed"] is False

    archived = {
        "metadata": pre.checkpoint.metadata,
        "arrays": pre.checkpoint.arrays,
        "envelope": {
            "stage_kind": "pre-acceleration",
            "bindings": dict(pre.bindings),
        },
    }
    restored = adapter.restore_checkpoint(
        "pre-acceleration/N0",
        archived,
        context={
            "freeze_authority": {"manufactured": True},
            "parent_stage": parent_stage,
        },
    )
    production._require_tree_equal(
        restored.runtime.pre_result, pre.runtime.pre_result,
        "staged-stop pre restart",
    )
    assert "bulk" not in calls


def test_production_inventory_is_explicit_and_every_entry_is_freeze_required(tmp_path):
    adapter = production.Protocol125ProductionAdapter().runner_adapters()
    inventory = adapter.source_manifest_inventory
    assert inventory == production.protocol125_production_source_inventory()
    assert len(inventory) == 62
    assert "environment:uv-lock" in inventory
    assert "environment:runtime-contract" in inventory
    assert "input:sealed-protocol120-parent" in inventory


def test_omitted_transitive_manifest_entry_fails_before_output(tmp_path):
    adapter = production.Protocol125ProductionAdapter().runner_adapters()
    inventory = adapter.source_manifest_inventory
    complete_root = tmp_path/"complete"
    complete_root.mkdir()
    complete = _freeze_record(complete_root, inventory)
    authority = validate_protocol125_freeze_authority(complete)
    assert runner._validated_adapter(adapter, authority) is adapter

    omitted_root = tmp_path/"omitted"
    omitted_root.mkdir()
    omitted_name = "source:bhps.adm_corner"
    omitted = _freeze_record(omitted_root, inventory, omit=omitted_name)
    omitted_authority = validate_protocol125_freeze_authority(omitted)
    with pytest.raises(runner.Protocol125AdapterBlocker, match=omitted_name):
        runner._validated_adapter(adapter, omitted_authority)
    assert not Path(omitted["candidate_output_directory"]).exists()


def test_runner_rechecks_frozen_runtime_environment_before_output(tmp_path):
    adapter = production.Protocol125ProductionAdapter().runner_adapters()
    freeze = _freeze_record(tmp_path, adapter.source_manifest_inventory)
    authority = validate_protocol125_freeze_authority(freeze)
    calls = []

    def reject_runtime(path):
        calls.append(Path(path))
        raise RuntimeError("manufactured runtime drift")

    drifted = replace(
        adapter,
        runtime_environment_verifier=reject_runtime,
    )
    with pytest.raises(
        runner.Protocol125AdapterBlocker,
        match="active runtime differs",
    ):
        runner._validated_adapter(drifted, authority)
    assert calls == [
        Path(adapter.source_manifest_inventory["environment:runtime-contract"])
    ]
    assert not Path(freeze["candidate_output_directory"]).exists()


def test_two_parent_inputs_bind_each_restored_pre_bulk_audit():
    def post(label):
        primary = SimpleNamespace(position=object(), acceleration=object())
        return SimpleNamespace(
            shared_build=SimpleNamespace(
                final_pair=SimpleNamespace(primary=primary),
            ),
            correction_profile={"label": label},
            position_v2=np.zeros((153, 343, 9)),
            hzz_zz_v2=np.zeros((153, 343)),
            a_hzz_v2=np.zeros((153, 343)),
            axis_image_profile={"label": label},
        )

    posts = {label: post(label) for label in production.PARENT_LABELS}
    native = {
        label: {"parent_label": label, "kind": "native-completion"}
        for label in production.PARENT_LABELS
    }
    construction_provenance = {
        label: {"parent_label": label, "kind": "construction-provenance"}
        for label in production.PARENT_LABELS
    }
    pres = {
        label: SimpleNamespace(
            bulk_audit={"parent_label": label},
            native_evidence=native[label],
            pre_inputs=SimpleNamespace(
                construction_provenance=construction_provenance[label],
            ),
        )
        for label in production.PARENT_LABELS
    }
    inputs = production.Protocol125ProductionAdapter._two_parent_inputs(posts, pres)
    assert inputs.n0_bulk_audit is pres["N0"].bulk_audit
    assert inputs.n1_bulk_audit is pres["N1"].bulk_audit
    assert inputs.n0_native_completion_evidence is native["N0"]
    assert inputs.n1_native_completion_evidence is native["N1"]
    assert inputs.n0_construction_provenance is construction_provenance["N0"]
    assert inputs.n1_construction_provenance is construction_provenance["N1"]


def test_post_products_use_canonical_lineage_schema_and_persist_full_wrapper(monkeypatch):
    v2_z = np.linspace(1.0, np.e, 3)
    v2_r = np.linspace(0.0, 12.0, 4)
    dense_r = np.linspace(0.0, 12.0, 5)
    reduced_v2 = np.full((3, 4, 9), 7.0)

    class Position:
        def evaluate_reduced(self, z, r):
            if len(z) == len(v2_z) and len(r) == len(v2_r):
                return reduced_v2.copy()
            return np.ones((len(z), len(r), 9))

        def evaluate_coordinate_components(self, z, r, **orders):
            return np.full((len(z), len(r), 9), 99.0)

    class Acceleration:
        def evaluate_reduced(self, z, r):
            return np.ones((len(z), len(r), 9))

        def evaluate_coordinate_components(self, z, r, **orders):
            return np.ones((len(z), len(r), 9))

    class BulkSampler:
        def evaluate_wall_reduced(self, r):
            return np.ones((2, len(r), 9))

        def v2_axis_reduced(self):
            return np.ones((len(v2_z), 9))

    position = Position()
    acceleration = Acceleration()
    shared = SimpleNamespace(
        final_pair=SimpleNamespace(
            primary=SimpleNamespace(position=position, acceleration=acceleration),
        ),
        bulk_sampler=BulkSampler(),
    )
    lineage_validation = {
        "passed": True,
        "gates": {},
        "position_payload_hash": "1"*64,
        "position_only": {},
        "shared": {},
    }
    lineage_wrapper = {
        "protocol": "Protocol-125-concrete-append-only-position-lineage-v1",
        "position_only_snapshot": object(),
        "shared_snapshot": object(),
        "append_only_validation": lineage_validation,
        "mesh_identity": {"passed": True},
        "reference_source_fingerprint": "2"*64,
        "top_level_composite_identities_evolved": True,
        "invariant_position_payload_bitwise": True,
        "acceleration_children_recorded": True,
        "passed": True,
    }
    captured = {}

    def capture_post_inputs(inputs):
        captured["post_inputs"] = inputs
        return {}

    monkeypatch.setattr(production, "build_protocol125_shared_representation", lambda *a: shared)
    monkeypatch.setattr(production, "build_protocol125_wall_profile_evidence", lambda *a: {})
    monkeypatch.setattr(
        production, "build_protocol125_final_matrix_inputs",
        lambda *a, **k: SimpleNamespace(inputs={}, provenance={}, adapter_record={}),
    )
    monkeypatch.setattr(production, "evaluate_protocol125_final_representation_matrix", lambda *a: {})
    monkeypatch.setattr(
        production,
        "build_protocol125_append_only_position_lineage",
        lambda *a: lineage_wrapper,
    )
    monkeypatch.setattr(
        production, "frozen_validation_meshes",
        lambda: {"dense_wall": {"r": dense_r}, "V2": {"z": v2_z, "r": v2_r}},
    )
    monkeypatch.setattr(production, "correction_profile", lambda *a: {})
    monkeypatch.setattr(production, "axis_acceleration_derivative_image_profile", lambda *a: {})
    monkeypatch.setattr(production, "capture_protocol125_bulk_sampler_provenance", lambda *a, **k: {})
    monkeypatch.setattr(
        production,
        "capture_protocol125_postacceleration_provenance",
        capture_post_inputs,
    )
    monkeypatch.setattr(
        production, "compose_protocol125_postacceleration_records",
        lambda *a: {"parent_identity": "0"*64},
    )
    parent = {
        "label": "N0", "parent_identity": "0"*64,
        "background": {}, "z": np.asarray([1.0, np.e]),
        "r": np.asarray([0.0, 12.0]), "position": np.zeros((2, 2, 9)),
    }
    pre = SimpleNamespace(
        position_pair=object(), position_state_record={}, reference_pair=object(),
        pre_result={},
    )
    products = production.Protocol125ProductionAdapter._post_products(
        parent, pre, np.zeros((2, 2, 9)), np.zeros((2, 2, 9)),
        {"source_triplet": {}},
    )
    assert np.array_equal(products["position_v2"], reduced_v2)
    assert products["lineage"] is lineage_wrapper
    assert captured["post_inputs"].append_only_lineage is lineage_validation
    assert captured["post_inputs"].append_only_lineage is not lineage_wrapper


def test_scientific_acceleration_failure_is_checkpointed_and_restored_without_retry(
    monkeypatch,
):
    z = np.linspace(1.0, np.e, 7)
    r = np.linspace(0.0, 12.0, 7)
    position = np.zeros((len(z), len(r), 9))
    parent = {
        "label": "N0",
        "z": z,
        "r": r,
        "position": position,
        "selector_q": np.zeros((len(z), len(r))),
        "phi": np.zeros((len(z), len(r))),
        "reference_q": np.zeros((len(z), len(r))),
        "reference_phi": np.zeros((len(z), len(r))),
        "background": {},
    }
    parent["parent_identity"] = production._parent_identity(parent)
    identity = parent["parent_identity"]
    bindings = {"N0": identity}
    position_state = _ManufacturedPositionState()
    pre_result = _manufactured_passed_pre_acceleration_result(identity)
    parent_stage = runner.Protocol125RunnerStage(
        production.Protocol125ProductionParentState(parent),
        production._checkpoint("parent", bindings, None, {"parent": parent}),
        bindings,
    )
    pre_runtime = production.Protocol125ProductionPreState(
        SimpleNamespace(primary=position_state),
        {},
        None,
        {},
        {},
        {},
        None,
        {},
        pre_result,
    )
    pre_stage = runner.Protocol125RunnerStage(
        pre_runtime,
        production._checkpoint(
            "pre-acceleration", bindings, pre_result["groups"],
            {"pre_result": pre_result},
        ),
        bindings,
        pre_result["groups"],
    )
    bulk = np.zeros_like(position)
    bulk_record = {
        "method": "manufactured deterministic bulk acceleration",
        "bulk_sha256": hash_arrays(bulk),
    }
    bulk_calls = []

    def manufactured_bulk(*args, **kwargs):
        bulk_calls.append((args, kwargs))
        return bulk.copy(), dict(bulk_record)

    failure = _manufactured_fixed_point_failure(
        parent, position_state, bulk,
    )
    solve_calls = []

    def scientific_stop(*args, **kwargs):
        solve_calls.append((args, kwargs))
        raise acceleration.Protocol125AccelerationScientificFailure(failure)

    monkeypatch.setattr(
        production, "bulk_acceleration_from_completed_position",
        manufactured_bulk,
    )
    monkeypatch.setattr(
        production, "solve_joint_parent_acceleration_fixed_point",
        scientific_stop,
    )
    adapter = production.Protocol125ProductionAdapter()
    post = adapter.compose_post_acceleration(
        "N0", parent_stage, pre_stage,
        freeze_authority={"manufactured": True},
    )

    assert isinstance(
        post.runtime,
        production.Protocol125ProductionAccelerationFailurePostState,
    )
    assert post.runtime.post_result["classification"] == "FAIL-acceleration"
    assert post.runtime.post_result["complete"] is True
    assert post.runtime.post_result["provenance_valid"] is True
    assert post.runtime.post_result["failed_groups"] == (
        "acceleration_closure",
    )
    assert post.runtime.post_result["not_reached_groups"] == tuple(
        POST_ACCELERATION_GROUPS[1:]
    )
    assert len(solve_calls) == 1

    archived = {
        "metadata": post.checkpoint.metadata,
        "arrays": post.checkpoint.arrays,
        "envelope": {
            "stage_kind": "post-acceleration",
            "bindings": bindings,
        },
    }
    restored = adapter.restore_checkpoint(
        "post-acceleration/N0",
        archived,
        context={
            "freeze_authority": {"manufactured": True},
            "parent_stage": parent_stage,
            "pre_acceleration_stage": pre_stage,
        },
    )
    assert len(solve_calls) == 1
    assert len(bulk_calls) == 2
    production._require_tree_equal(
        restored.runtime.acceleration_failure,
        post.runtime.acceleration_failure,
        "acceleration failure restart",
    )
    production._require_tree_equal(
        restored.runtime.post_result,
        post.runtime.post_result,
        "acceleration failure post restart",
    )

    def technical_fault(*args, **kwargs):
        raise RuntimeError("manufactured technical defect")

    monkeypatch.setattr(
        production, "solve_joint_parent_acceleration_fixed_point",
        technical_fault,
    )
    with pytest.raises(RuntimeError, match="manufactured technical defect"):
        adapter.compose_post_acceleration(
            "N0", parent_stage, pre_stage,
            freeze_authority={"manufactured": True},
        )
