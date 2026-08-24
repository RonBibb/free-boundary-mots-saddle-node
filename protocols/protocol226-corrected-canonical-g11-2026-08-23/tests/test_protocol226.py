import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("protocol226_tested", ROOT / "protocol226.py")
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)


def test_field_decision_strict_decrease():
    prior = {"maximum_absolute": 4.0, "RMS": 2.0, "increment_scale": 1.0}
    current = {"maximum_absolute": 3.0, "RMS": 1.0, "increment_scale": 1.0,
               "common_lattice_shape": [17, 31], "coordinate_maximum_absolute_mismatch": 0.0}
    assert p.field_decision(prior, current)["passed"]


def test_field_decision_detects_rms_plateau():
    prior = {"maximum_absolute": 4.0, "RMS": 2.0, "increment_scale": 1.0}
    current = {"maximum_absolute": 3.0, "RMS": 2.1, "increment_scale": 1.0,
               "common_lattice_shape": [17, 31], "coordinate_maximum_absolute_mismatch": 0.0}
    result = p.field_decision(prior, current)
    assert not result["passed"] and result["norms"]["maximum_absolute"]["passed"]


def test_roundoff_rule():
    floor = 64.0 * 2.220446049250313e-16
    prior = {"maximum_absolute": floor / 2, "RMS": floor / 3, "increment_scale": 1.0}
    current = {"maximum_absolute": floor * 0.75, "RMS": floor * 0.5, "increment_scale": 1.0,
               "common_lattice_shape": [17, 31], "coordinate_maximum_absolute_mismatch": 0.0}
    assert p.field_decision(prior, current)["passed"]


def test_result_fingerprint_changes():
    assert p.result_fingerprint({"a": 1}) != p.result_fingerprint({"a": 2})


def test_corrected_zero_step_policy_distinguishes_modes():
    canonical = np.array([1.0, 2.0])
    archived_second_mode = np.array([1.0, 3.0])
    checks = p.zero_step_policy_checks(
        canonical, canonical.copy(), canonical.copy(), archived_second_mode.copy(),
        archived_second_mode, True, True,
    )
    assert all(checks.values())


def test_corrected_zero_step_policy_rejects_cross_mode_replay():
    canonical = np.array([1.0, 2.0])
    archived_second_mode = np.array([1.0, 3.0])
    checks = p.zero_step_policy_checks(
        canonical, canonical.copy(), archived_second_mode, archived_second_mode,
        archived_second_mode, True, True,
    )
    assert not checks["first_stage_matches_canonical_zero_step"]
