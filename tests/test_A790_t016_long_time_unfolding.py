import json

import numpy as np
import pytest

import run_corrected_A790_t016_long_time_unfolding as protocol


def test_fixed_time_and_stage_lattice():
    assert protocol.DT == 0.000125
    assert protocol.FINAL_STEP == 128
    assert protocol.FINAL_STEP * protocol.DT == 0.016
    assert protocol.ANCHOR_STEP * protocol.DT == 0.008
    assert protocol.R8_SURFACE_STEPS == tuple(range(32, 129, 8))
    assert protocol.R10_SURFACE_STEPS == (64, 80, 96, 112, 128)
    assert protocol.STABILITY_STEPS == (64, 96, 128)


def test_direct_scientific_entry_is_rejected():
    with pytest.raises(RuntimeError, match="frozen bootstrap"):
        protocol.validate_bootstrap_entry()


def test_strict_json_rejects_duplicate_and_nonfinite(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"a":1,"a":2}\n')
    with pytest.raises(ValueError, match="duplicate"):
        protocol.strict_json(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a":NaN}\n')
    with pytest.raises(ValueError, match="non-finite"):
        protocol.strict_json(nonfinite)


def test_relative_l2_exact_and_perturbed():
    value = np.asarray([1.0, 2.0, 3.0])
    assert protocol.relative_l2(value, value.copy()) == 0.0
    changed = value.copy()
    changed[-1] += 1e-8
    assert 0.0 < protocol.relative_l2(value, changed) < 1e-7


def _diagnostic(**updates):
    result = {
        "finite": True,
        "Lorentzian": True,
        "global_GH_constraint": 1e-5,
        "wall_position_residual": 1e-5,
        "normal_wall_position_residual": 1e-5,
        "outer_position_residual": 1e-14,
        "outer_source_residual": 1e-14,
    }
    result.update(updates)
    return result


def test_diagnostic_gate_is_strict():
    assert protocol.diagnostic_passes(_diagnostic())
    assert not protocol.diagnostic_passes(_diagnostic(global_GH_constraint=0.005))
    assert not protocol.diagnostic_passes(_diagnostic(Lorentzian=False))


def test_midpoint_stage_gate_is_strict():
    record = {
        "all_stages_finite": True,
        "all_stage_signatures_lorentzian": True,
        "maximum_normal_wall_acceleration_residual": 1e-14,
        "maximum_outer_acceleration_residual": 1e-14,
        "maximum_outer_source_residual": 1e-14,
        "maximum_outer_metric_correction": 0.01,
        "maximum_outer_scalar_correction": 0.02,
        "maximum_outer_source_correction": 0.01,
    }
    assert protocol.stage_diagnostic_passes(record)
    record["maximum_outer_scalar_correction"] = 0.20
    assert not protocol.stage_diagnostic_passes(record)
    record["maximum_outer_scalar_correction"] = 0.02
    record["all_stage_signatures_lorentzian"] = False
    assert not protocol.stage_diagnostic_passes(record)


@pytest.mark.parametrize(
    "gates,expected",
    [
        ((True, True, True, True), ("pass-domain", "LONG-TIME-PAIR-DOMAIN-TRANSFERRED")),
        ((True, True, True, False), ("pass", "LONG-TIME-PAIR-WITH-RESOLVED-STABILITY")),
        ((True, False, True, False), ("review", "BRANCH-STRUCTURE-CHANGED-OR-UNRESOLVED")),
        ((False, True, True, False), ("review", "LONG-TIME-NUMERICAL-CONTROL-INCOMPLETE")),
    ],
)
def test_result_taxonomy(gates, expected):
    assert protocol.classify_result(*gates) == expected


def _geometry(axis, brane, area=40.0):
    return {
        "admitted": True,
        "geometry": {
            "finite": True,
            "one_sided_cap_area": area,
            "equivalent_area_radius": area ** (1.0 / 3.0),
            "rho_axis": axis,
            "rho_brane": brane,
        },
    }


def _surface_record(step, inner, outer):
    return {"step": step, "branches": [inner, outer]}


def test_branch_identity_prefers_continuation():
    records = {
        "32": _surface_record(32, _geometry(1.0, 1.2), _geometry(1.4, 1.6)),
        "40": _surface_record(40, _geometry(1.01, 1.19), _geometry(1.41, 1.61)),
        "48": _surface_record(48, _geometry(1.02, 1.18), _geometry(1.42, 1.62)),
    }
    result = protocol.branch_identity_record(records)
    assert result["passed"]
    assert all(item["same_assignment_preferred"] for item in result["comparisons"])


def test_branch_identity_rejects_swapped_order():
    records = {
        "32": _surface_record(32, _geometry(1.0, 1.2), _geometry(1.4, 1.6)),
        "40": _surface_record(40, _geometry(1.4, 1.6), _geometry(1.0, 1.2)),
    }
    assert not protocol.branch_identity_record(records)["passed"]


def _trend_history(changes):
    values = [10.0]
    for change in changes:
        values.append(values[-1] + change)
    return {
        str(step): _surface_record(
            step,
            _geometry(1.0, 1.2, values[index]),
            _geometry(1.4, 1.6, values[index] + 2.0),
        )
        for index, step in enumerate(range(32, 32 + 8 * len(values), 8))
    }


def test_trend_classification_slowing_and_continuing():
    slowing = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.20, 0.15, 0.10, 0.08, 0.06, 0.04]
    surfaces = {
        "R8G7": _trend_history(slowing),
        "R8G8": _trend_history([value * 1.001 for value in slowing]),
    }
    result = protocol.trend_summary(surfaces)
    assert result["classifications"]["inner"]["area"]["classification"] == "SLOWING"

    continuing = [0.4] * 12
    surfaces = {
        "R8G7": _trend_history(continuing),
        "R8G8": _trend_history([0.40001] * 12),
    }
    result = protocol.trend_summary(surfaces)
    assert result["classifications"]["inner"]["area"]["classification"] == "CONTINUING"


def test_freeze_loader_fails_closed_when_absent(monkeypatch, tmp_path):
    missing = tmp_path / "missing.json"
    monkeypatch.setattr(protocol, "FREEZE", missing)
    with pytest.raises(FileNotFoundError, match="freeze is absent"):
        protocol.load_and_verify_freeze()


def test_freeze_loader_verifies_protocol_and_file(monkeypatch, tmp_path):
    protocol_file = tmp_path / "protocol.md"
    protocol_file.write_text("prospective\n")
    source = tmp_path / "source.py"
    source.write_text("x = 1\n")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({
        "schema": "A790-t016-long-time-unfolding-freeze-v1",
        "protocol_sha256": protocol.sha256_file(protocol_file),
        "runtime": protocol.runtime_record(),
        "files": {str(source): protocol.sha256_file(source)},
    }, sort_keys=True) + "\n")
    monkeypatch.setattr(protocol, "PROTOCOL", protocol_file)
    monkeypatch.setattr(protocol, "FREEZE", freeze)
    assert protocol.load_and_verify_freeze() == {
        str(source): protocol.sha256_file(source)
    }
    source.write_text("x = 2\n")
    with pytest.raises(RuntimeError, match="hash differs"):
        protocol.load_and_verify_freeze()
