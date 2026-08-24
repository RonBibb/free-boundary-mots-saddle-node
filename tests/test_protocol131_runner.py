from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import run_A790_protocol131_postmortem as runner
from bhps.protocol131_freeze_authority import manifest_fingerprint


@pytest.fixture(autouse=True)
def _manufactured_runtime_contract(monkeypatch):
    monkeypatch.setattr(
        runner,
        "validate_protocol131_environment_contract",
        lambda path: {"manufactured": str(path)},
    )
    monkeypatch.setattr(
        runner,
        "validate_protocol131_source_manifest",
        lambda manifest: manifest,
    )
    monkeypatch.setattr(
        runner,
        "validate_protocol131_input_manifest",
        lambda manifest: manifest,
    )


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(path):
    return {"path": str(path), "sha256": _sha256(path)}


def _freeze_record(tmp_path):
    candidate = tmp_path / "candidate"
    protocol = tmp_path / "protocol.md"
    protocol.write_text(
        "# Protocol 131 runner fixture\n\n"
        "Status: **FROZEN**\n"
        f"Candidate-Output-Directory: {candidate}\n\n"
        "## Scope\nArchive-only.\n",
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime.json"
    runtime.write_text('{"runtime":"manufactured"}\n', encoding="utf-8")
    source = tmp_path / "postmortem_source.py"
    source.write_text("ARCHIVE_ONLY = True\n", encoding="utf-8")
    n0 = tmp_path / "parent_N0.npz"
    n0.write_bytes(b"immutable-N0")
    n1 = tmp_path / "parent_N1.npz"
    n1.write_bytes(b"immutable-N1")
    source_manifest = {"postmortem-source": _entry(source)}
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


def _callbacks(calls):
    def analyze(label):
        calls.append(f"analyze:{label}")
        return {
            "summary": {
                "parent_label": label,
                "replay": {"maximum": 1.0e-10},
                "archive_only": True,
            },
            "arrays": {
                "residual": np.asarray([0.0, 1.0 if label == "N0" else 2.0]),
                "profile": np.arange(6, dtype=float).reshape(2, 3),
            },
        }

    def classify(summaries, arrays):
        calls.append("classify")
        assert list(summaries) == ["N0", "N1"]
        assert list(arrays) == ["N0", "N1"]
        assert float(arrays["N0"]["residual"][-1]) == 1.0
        assert float(arrays["N1"]["residual"][-1]) == 2.0
        return {
            "classification": "INCONCLUSIVE-MIXED",
            "complete": True,
            "provenance_valid": True,
            "reason": "manufactured runner test",
        }

    return analyze, classify


def _install_callbacks(monkeypatch, calls):
    analyze, classify = _callbacks(calls)
    monkeypatch.setattr(runner, "_default_parent_analyzer", analyze)
    monkeypatch.setattr(runner, "_default_finalizer", classify)


def _forbid_callbacks(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("completed atomic evidence must not be recomputed")

    monkeypatch.setattr(runner, "_default_parent_analyzer", forbidden)
    monkeypatch.setattr(runner, "_default_finalizer", forbidden)


def _write_external_snapshot(record):
    candidate = Path(record["candidate_output_directory"])
    authority = runner.validate_protocol131_freeze_authority(record)
    snapshot_path = runner._snapshot_path(candidate)
    runner.atomic_write_json(snapshot_path, runner._jsonable(authority, "authority"))
    return snapshot_path, json.loads(snapshot_path.read_text(encoding="utf-8"))


def _load_snapshot(candidate):
    path = runner._snapshot_path(candidate)
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_json_loader_rejects_nonfinite_constants(tmp_path, constant):
    path = tmp_path / "nonfinite.json"
    path.write_text(f'{{"value":{constant}}}\n', encoding="utf-8")
    with pytest.raises(runner.Protocol131RunnerError, match="nonfinite JSON"):
        runner._load_json_file(path, "manufactured nonfinite JSON")


def test_first_run_is_ordered_single_artifact_and_archive_only(tmp_path, monkeypatch):
    record = _freeze_record(tmp_path)
    candidate = Path(record["candidate_output_directory"])
    calls = []
    _install_callbacks(monkeypatch, calls)
    result = runner.run_protocol131_postmortem(freeze_record=record)
    assert calls == ["analyze:N0", "analyze:N1", "classify"]
    assert result["classification"] == "INCONCLUSIVE-MIXED"
    assert result["parent_construction_authorized"] is False
    assert result["parent_update_authorized"] is False
    assert result["phase_a_authorized"] is False
    assert result["evolution_authorized"] is False
    assert result["interface_physics_authorized"] is False
    assert runner._snapshot_path(candidate).is_file()
    assert runner._snapshot_path(candidate).parent == candidate.parent
    assert not (candidate / "freeze_authority_snapshot.json").exists()
    ledger = json.loads((candidate / runner.LEDGER_NAME).read_text())
    assert ledger["stage_order"] == list(runner.STAGE_ORDER)
    assert all(ledger["stages"][stage]["status"] == "complete" for stage in runner.STAGE_ORDER)
    for label in runner.PARENT_LABELS:
        path = runner._parent_path(candidate, label)
        assert path.is_file()
        assert not (candidate / f"diagnostic_{label}_summary.json").exists()
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive[runner._RESERVED_ARRAY_KEY]))
            assert metadata["scientific_summary"]["parent_label"] == label
            assert "residual" in archive.files


def test_completed_resume_reloads_without_reanalysis(tmp_path, monkeypatch):
    record = _freeze_record(tmp_path)
    _install_callbacks(monkeypatch, [])
    first = runner.run_protocol131_postmortem(freeze_record=record)
    candidate = Path(record["candidate_output_directory"])
    snapshot = _load_snapshot(candidate)
    _forbid_callbacks(monkeypatch)
    restored = runner.run_protocol131_postmortem(freeze_authority=snapshot)
    assert restored == first


def test_first_run_validation_failure_creates_no_candidate_or_snapshot(tmp_path, monkeypatch):
    record = _freeze_record(tmp_path)
    candidate = Path(record["candidate_output_directory"])
    record["input_manifest"]["N0"]["sha256"] = "0" * 64
    _install_callbacks(monkeypatch, [])
    with pytest.raises(Exception, match="does not match file bytes"):
        runner.run_protocol131_postmortem(freeze_record=record)
    assert not candidate.exists()
    assert not runner._snapshot_path(candidate).exists()


def test_default_api_preflight_fails_before_namespace_mutation(tmp_path, monkeypatch):
    record = _freeze_record(tmp_path)
    candidate = Path(record["candidate_output_directory"])
    monkeypatch.setattr(runner, "extended_precision_residual", None)
    with pytest.raises(runner.Protocol131RunnerError, match="precision"):
        runner.run_protocol131_postmortem(freeze_record=record)
    assert not candidate.exists()
    assert not runner._snapshot_path(candidate).exists()


def test_failed_jacobian_stops_before_linear_and_precision(monkeypatch):
    parent = {
        "record": {"parent_identity": "0" * 64},
        "generated_input_sha256": "1" * 64,
    }
    residual = np.asarray([2.0e-10, 0.0])
    replay = {"maximum": 2.0e-10, "rms": np.sqrt(2.0)*1.0e-10}
    localization = {
        "atoms": {},
        "blocks": {},
        "dominant_atom_by_Linf": "metric/lower/axis",
        "binned_energy_16x16": np.zeros((2, 16, 16)),
        "wall_even": np.zeros((2, 1)),
        "wall_odd": np.zeros((2, 1)),
    }
    shaped = residual.reshape(2, 1, 1)
    cancellation = {
        "raw": shaped,
        "reference_defect": np.zeros_like(shaped),
        "balanced_before_wall_override": shaped,
        "final_subtraction_roundoff_bound": np.zeros_like(shaped),
        "final_row_roundoff_bound": np.zeros_like(shaped),
        "wall_terms": {},
    }
    monkeypatch.setattr(runner.postmortem, "load_terminal_parent", lambda label: parent)
    monkeypatch.setattr(
        runner.postmortem,
        "replay_residual_and_jacobian",
        lambda value: (residual, np.eye(2), replay),
    )
    monkeypatch.setattr(
        runner.postmortem, "residual_localization",
        lambda value, result: localization,
    )
    monkeypatch.setattr(
        runner.postmortem, "residual_cancellation_terms",
        lambda value, result: cancellation,
    )
    monkeypatch.setattr(
        runner.postmortem, "audit_analytic_jacobian",
        lambda *args: {"passed": False, "directions": {}},
    )
    monkeypatch.setattr(
        runner.postmortem,
        "linear_range_analysis",
        lambda *args: pytest.fail("linear analysis must be ordered-not-reached"),
    )
    monkeypatch.setattr(
        runner,
        "extended_precision_residual",
        lambda *args: pytest.fail("precision must be ordered-not-reached"),
    )
    result = runner._default_parent_analyzer("N0")
    summary, arrays = runner._normalize_parent_result(result, "N0")
    assert summary["jacobian_audit"]["passed"] is False
    assert summary["linear"]["not_reached"] is True
    assert summary["precision"]["not_reached"] is True
    assert arrays["residual"].shape == (2,)


def test_sigma_max_failure_reaches_inconclusive_artifact_path(monkeypatch):
    parent = {
        "label": "N0",
        "record": {"parent_identity": "0" * 64},
        "generated_input_sha256": "1" * 64,
        "z": np.asarray([1.0]),
        "r": np.asarray([0.0]),
        "q": np.asarray([[0.1]]),
        "phi": np.asarray([[0.2]]),
        "background": {"v0": 0.1, "v1": 0.01},
    }
    residual = np.asarray([2.0e-10, 0.0])
    replay = {"maximum": 2.0e-10, "rms": np.sqrt(2.0)*1.0e-10}
    localization = {
        "atoms": {},
        "blocks": {},
        "dominant_atom_by_Linf": "metric/lower/axis",
        "binned_energy_16x16": np.zeros((2, 16, 16)),
        "wall_even": np.zeros((2, 1)),
        "wall_odd": np.zeros((2, 1)),
    }
    shaped = residual.reshape(2, 1, 1)
    cancellation = {
        "raw": shaped,
        "reference_defect": np.zeros_like(shaped),
        "balanced_before_wall_override": shaped,
        "final_subtraction_roundoff_bound": np.zeros_like(shaped),
        "final_row_roundoff_bound": np.zeros_like(shaped),
        "wall_terms": {},
    }
    linear = {
        "analysis_complete": False,
        "failure_stage": "largest-singular-value",
        "column_scale": np.ones(2),
        "column_exponents": np.zeros(2, dtype=np.int64),
        "sigma_max": None,
        "tau_rank": 2*np.finfo(float).eps,
        "lu": {"succeeded": False, "not_reached": True},
        "lsmr": {"not_reached": True},
        "lsqr": {"not_reached": True},
        "projection_accepted": False,
        "projection_numerically_zero_floor": False,
        "mode_method": "not-reached",
        "spectrum_certified": False,
        "spectrum_attempts": [],
        "spectrum_errors": ["sigma-max: manufactured"],
        "modes": [],
        "mode_left_vectors": np.empty((2, 0)),
        "mode_right_vectors": np.empty((2, 0)),
        "mode_extension_used": False,
        "high_nullity_unresolved": False,
        "ruiz": {
            "not_reached": True,
            "obstruction_certificate_complete": False,
        },
    }
    monkeypatch.setattr(runner.postmortem, "load_terminal_parent", lambda label: parent)
    monkeypatch.setattr(
        runner.postmortem, "replay_residual_and_jacobian",
        lambda value: (residual, np.eye(2), replay),
    )
    monkeypatch.setattr(
        runner.postmortem, "residual_localization",
        lambda value, result: localization,
    )
    monkeypatch.setattr(
        runner.postmortem, "residual_cancellation_terms",
        lambda value, result: cancellation,
    )
    monkeypatch.setattr(
        runner.postmortem, "audit_analytic_jacobian",
        lambda *args: {"passed": True, "directions": {}},
    )
    monkeypatch.setattr(
        runner.postmortem, "linear_range_analysis", lambda *args: linear,
    )
    monkeypatch.setattr(
        runner,
        "extended_precision_residual",
        lambda *args: ({
            "complete": True,
            "mp_certified": True,
            "dual_certified": True,
            "eta_F": 0.0,
            "arithmetic_max_below_target": False,
            "longdouble_maximum": 2.0e-10,
        }, {}),
    )
    result = runner._default_parent_analyzer("N0")
    summary, arrays = runner._normalize_parent_result(result, "N0")
    assert summary["linear"]["spectrum_certified"] is False
    assert summary["trust_radius"]["not_reached"] is True
    classification = runner.postmortem.classify_protocol131(
        {"N0": summary, "N1": summary}, {"N0": arrays, "N1": arrays},
    )
    assert classification["classification"] == "INCONCLUSIVE-MIXED"
    assert "spectrum" in classification["reason"]


@pytest.mark.parametrize("tamper", ["source", "input"])
def test_resume_rehashes_source_and_input_before_restoring(
    tmp_path, monkeypatch, tamper,
):
    record = _freeze_record(tmp_path)
    _install_callbacks(monkeypatch, [])
    runner.run_protocol131_postmortem(freeze_record=record)
    candidate = Path(record["candidate_output_directory"])
    snapshot = _load_snapshot(candidate)
    if tamper == "source":
        path = Path(record["source_manifest"]["postmortem-source"]["path"])
    else:
        path = Path(record["input_manifest"]["N1"]["path"])
    path.write_bytes(path.read_bytes() + b"tamper")
    _forbid_callbacks(monkeypatch)
    with pytest.raises(Exception):
        runner.run_protocol131_postmortem(freeze_authority=snapshot)


def test_snapshot_before_candidate_mkdir_is_recoverable(tmp_path, monkeypatch):
    record = _freeze_record(tmp_path)
    candidate = Path(record["candidate_output_directory"])
    snapshot_path, snapshot = _write_external_snapshot(record)
    assert snapshot_path.is_file()
    assert not candidate.exists()
    calls = []
    _install_callbacks(monkeypatch, calls)
    result = runner.run_protocol131_postmortem(freeze_authority=snapshot)
    assert candidate.is_dir()
    assert calls == ["analyze:N0", "analyze:N1", "classify"]
    assert result["classification"] == "INCONCLUSIVE-MIXED"


def test_candidate_mkdir_before_ledger_is_recoverable(tmp_path, monkeypatch):
    record = _freeze_record(tmp_path)
    candidate = Path(record["candidate_output_directory"])
    _, snapshot = _write_external_snapshot(record)
    candidate.mkdir()
    assert not runner._ledger_path(candidate).exists()
    calls = []
    _install_callbacks(monkeypatch, calls)
    runner.run_protocol131_postmortem(freeze_authority=snapshot)
    assert runner._ledger_path(candidate).is_file()
    assert calls == ["analyze:N0", "analyze:N1", "classify"]


def test_parent_artifact_before_ledger_commit_is_adopted(tmp_path, monkeypatch):
    record = _freeze_record(tmp_path)
    candidate = Path(record["candidate_output_directory"])
    first_calls = []
    _install_callbacks(monkeypatch, first_calls)
    original = runner._record_parent_completion

    def crash_after_n0_artifact(ledger_path, ledger, label, **kwargs):
        if label == "N0":
            raise RuntimeError("manufactured crash after atomic N0 rename")
        return original(ledger_path, ledger, label, **kwargs)

    monkeypatch.setattr(runner, "_record_parent_completion", crash_after_n0_artifact)
    with pytest.raises(RuntimeError, match="atomic N0"):
        runner.run_protocol131_postmortem(freeze_record=record)
    assert first_calls == ["analyze:N0"]
    assert runner._parent_path(candidate, "N0").is_file()
    snapshot = _load_snapshot(candidate)

    resume_calls = []
    _install_callbacks(monkeypatch, resume_calls)
    monkeypatch.setattr(runner, "_record_parent_completion", original)
    runner.run_protocol131_postmortem(freeze_authority=snapshot)
    assert resume_calls == ["analyze:N1", "classify"]
    ledger = json.loads(runner._ledger_path(candidate).read_text())
    assert ledger["stages"]["diagnostic/N0"]["adopted_after_interruption"] is True


def test_final_artifact_before_ledger_commit_is_adopted(tmp_path, monkeypatch):
    record = _freeze_record(tmp_path)
    candidate = Path(record["candidate_output_directory"])
    calls = []
    _install_callbacks(monkeypatch, calls)
    original = runner._record_final_completion

    def crash_after_final_artifact(*args, **kwargs):
        raise RuntimeError("manufactured crash after atomic final rename")

    monkeypatch.setattr(runner, "_record_final_completion", crash_after_final_artifact)
    with pytest.raises(RuntimeError, match="atomic final"):
        runner.run_protocol131_postmortem(freeze_record=record)
    assert calls == ["analyze:N0", "analyze:N1", "classify"]
    assert (candidate / "classification_final.json").is_file()
    snapshot = _load_snapshot(candidate)
    _forbid_callbacks(monkeypatch)
    monkeypatch.setattr(runner, "_record_final_completion", original)
    result = runner.run_protocol131_postmortem(freeze_authority=snapshot)
    assert result["classification"] == "INCONCLUSIVE-MIXED"
    ledger = json.loads(runner._ledger_path(candidate).read_text())
    assert ledger["stages"]["classification/final"]["adopted_after_interruption"] is True


def test_missing_ledger_adopts_all_complete_artifacts(tmp_path, monkeypatch):
    record = _freeze_record(tmp_path)
    _install_callbacks(monkeypatch, [])
    first = runner.run_protocol131_postmortem(freeze_record=record)
    candidate = Path(record["candidate_output_directory"])
    snapshot = _load_snapshot(candidate)
    runner._ledger_path(candidate).unlink()
    _forbid_callbacks(monkeypatch)
    restored = runner.run_protocol131_postmortem(freeze_authority=snapshot)
    assert restored == first
    ledger = json.loads(runner._ledger_path(candidate).read_text())
    assert all(
        ledger["stages"][stage]["adopted_after_interruption"] is True
        for stage in runner.STAGE_ORDER
    )


def test_tampered_unindexed_parent_artifact_hard_stops(tmp_path, monkeypatch):
    record = _freeze_record(tmp_path)
    candidate = Path(record["candidate_output_directory"])
    _, snapshot = _write_external_snapshot(record)
    candidate.mkdir()
    runner._parent_path(candidate, "N0").write_bytes(b"not-an-npz")
    _forbid_callbacks(monkeypatch)
    with pytest.raises(Exception):
        runner.run_protocol131_postmortem(freeze_authority=snapshot)


def test_changed_completed_parent_artifact_is_not_recomputed(tmp_path, monkeypatch):
    record = _freeze_record(tmp_path)
    _install_callbacks(monkeypatch, [])
    runner.run_protocol131_postmortem(freeze_record=record)
    candidate = Path(record["candidate_output_directory"])
    snapshot = _load_snapshot(candidate)
    path = runner._parent_path(candidate, "N0")
    path.write_bytes(path.read_bytes() + b"tamper")
    _forbid_callbacks(monkeypatch)
    with pytest.raises(runner.Protocol131RunnerError, match="artifact changed"):
        runner.run_protocol131_postmortem(freeze_authority=snapshot)


def test_placeholder_classification_is_never_published(tmp_path, monkeypatch):
    record = _freeze_record(tmp_path)
    analyze, _ = _callbacks([])
    monkeypatch.setattr(runner, "_default_parent_analyzer", analyze)

    def placeholder(summaries, arrays):
        return {
            "classification": "INCONCLUSIVE-MIXED",
            "complete": False,
            "provenance_valid": False,
            "placeholder": True,
        }

    monkeypatch.setattr(runner, "_default_finalizer", placeholder)
    with pytest.raises(runner.Protocol131RunnerError, match="placeholder"):
        runner.run_protocol131_postmortem(freeze_record=record)
    candidate = Path(record["candidate_output_directory"])
    assert not (candidate / "classification_final.json").exists()
    ledger = json.loads(runner._ledger_path(candidate).read_text())
    assert ledger["stages"]["diagnostic/N0"]["status"] == "complete"
    assert ledger["stages"]["diagnostic/N1"]["status"] == "complete"
    assert ledger["stages"]["classification/final"]["status"] == "failed"
