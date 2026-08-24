import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("protocol230_tested", ROOT / "protocol230.py")
P230 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P230)
ACADEMIC_ROOT = ROOT.parents[2]


def real_records():
    P230.validate_p229_authority(ACADEMIC_ROOT)
    return {
        label: P230.validate_grid_artifact(ACADEMIC_ROOT, label)[0]
        for label in P230.GRIDS
    }


def test_protocol229_authority_and_all_grid_artifacts_authenticate():
    authority = P230.validate_p229_authority(ACADEMIC_ROOT)
    assert authority["fingerprint"] == "66df1b22ae6c4a66b7efe1fdc6965eca6f62bd47be893c76059561a525b919e1"
    records = real_records()
    assert set(records) == set(P230.GRIDS)
    assert all(record["passed"] is True for record in records.values())


def test_adjacent_grid_metrics_pass_and_are_json_native():
    transfers, passed = P230.cross_grid_checks(real_records())
    assert passed is True and type(passed) is bool
    assert set(transfers) == {"G9-G10", "G10-G11"}
    for item in transfers.values():
        assert item["passed"] is True and type(item["passed"]) is bool
        assert item["transversality_sign_agrees"] is True
        assert type(item["transversality_sign_agrees"]) is bool
        assert item["quadratic_sign_agrees"] is True
        assert type(item["quadratic_sign_agrees"]) is bool
        P230.canonical(item)


def test_adjacent_grid_metrics_are_well_inside_frozen_limits():
    transfers, passed = P230.cross_grid_checks(real_records())
    assert passed
    for item in transfers.values():
        assert item["critical_time_absolute_difference"] <= P230.DT / 4
        assert item["critical_area_relative_difference"] < 0.01
        assert item["transversality_relative_difference"] < 0.20
        assert item["quadratic_relative_difference"] < 0.20


def test_nonzero_sign_gate_rejects_zero_and_sign_change():
    assert not P230.same_nonzero_sign(0.0, 1.0)
    assert not P230.same_nonzero_sign(-1.0, 1.0)
    assert P230.same_nonzero_sign(-1.0, -2.0)


def test_protocol229_numpy_boolean_failure_is_not_reintroduced():
    transfers, _ = P230.cross_grid_checks(real_records())
    payload = {"adjacent_grid_transfers": transfers}
    decoded = json.loads(P230.canonical(payload))
    assert decoded == payload


def test_grid_fingerprint_rejects_changed_result():
    path = P230.p229_root(ACADEMIC_ROOT) / "candidate-output/protocol229_G9.json"
    value = P230.read_json(path)
    fingerprint = value.pop("fingerprint")
    assert P230.p229_grid_fingerprint(value) == fingerprint
    value["result"]["critical_time_estimate"] += 1e-12
    assert P230.p229_grid_fingerprint(value) != fingerprint


def test_freeze_and_finalization_lifecycle(tmp_path):
    root = tmp_path / "capsule"
    (root / "authority").mkdir(parents=True)
    (root / "tests").mkdir()
    for relative in ("PROTOCOL.md", "protocol230.py", "tests/test_protocol230.py"):
        source = ROOT / relative
        target = root / relative
        target.write_bytes(source.read_bytes())
    authority = P230.freeze(root, ACADEMIC_ROOT)
    assert authority["candidate_output_absent_at_freeze"] is True
    assert not (root / "candidate-output").exists()
    result = P230.finalize(root, ACADEMIC_ROOT)
    assert result["status"] == "PASS"
    assert result["classification"] == "FREE-BOUNDARY-MOTS-SADDLE-NODE-CLOSURE-PASS"
    assert result["archive_only_finalization"] is True
    assert result["scientific_reexecution_performed"] is False
    assert result["resolved_free_boundary_mots_saddle_node_claim_authorized"] is True
    assert P230.finalize(root, ACADEMIC_ROOT) == result


def test_coherently_refingerprinted_final_tamper_is_rejected(tmp_path):
    root = tmp_path / "capsule"
    (root / "authority").mkdir(parents=True)
    (root / "tests").mkdir()
    for relative in ("PROTOCOL.md", "protocol230.py", "tests/test_protocol230.py"):
        source = ROOT / relative
        target = root / relative
        target.write_bytes(source.read_bytes())
    P230.freeze(root, ACADEMIC_ROOT)
    P230.finalize(root, ACADEMIC_ROOT)
    final_path = root / "candidate-output/protocol230_result.json"
    final_path.chmod(0o644)
    value = P230.read_json(final_path)
    value["grid_summaries"]["G9"]["critical_time_estimate"] += 1e-12
    value.pop("fingerprint")
    value["fingerprint"] = P230.result_fingerprint(value)
    final_path.write_bytes(P230.canonical(value))
    final_path.chmod(0o444)
    try:
        P230.finalize(root, ACADEMIC_ROOT)
    except P230.Protocol230Error as error:
        assert "differs from recomputation" in str(error)
    else:
        raise AssertionError("coherent final-result tamper was accepted")
