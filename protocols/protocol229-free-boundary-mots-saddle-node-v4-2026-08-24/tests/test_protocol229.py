import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("protocol229_tested", ROOT / "protocol229.py")
P229 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P229)


def test_frozen_continuation_constants_are_exact():
    assert P229.DT == 0.00003125
    assert P229.START_TIME == 0.001
    assert P229.END_TIME == 0.0015
    assert P229.INITIAL_ARC_STEP == 1 / 64
    assert P229.MINIMUM_ARC_BRACKET == 1 / 4096
    assert P229.MINIMUM_CORRECTOR_STEP == 1 / 4096
    assert P229.MINIMUM_BRACKET_CORRECTOR_STEP == 1 / 65536
    assert P229.CORRECTOR_ARCLENGTH_LIMIT == 2e-5
    assert P229.MAXIMUM_CONTINUATION_STEPS == 64
    assert P229.PAIR_OFFSET_MULTIPLIERS == (1, 2, 4, 8, 16)


def test_time_tau_roundtrip():
    for value in (0.001, 0.00125, 0.0015):
        assert abs(P229.tau_to_time(P229.time_to_tau(value)) - value) < 1e-18


def test_dense_RK2_extension_reproduces_both_endpoints():
    shape = (2, 3)
    state = (np.ones(shape), 2 * np.ones(shape))
    k1 = (3 * np.ones(shape), 4 * np.ones(shape))
    k2 = (5 * np.ones(shape), 6 * np.ones(shape))
    trajectory = P229.DenseTrajectory([{"time": P229.START_TIME, "state": state, "k1": k1, "k2": k2}])
    assert all(np.array_equal(a, b) for a, b in zip(trajectory.state(P229.START_TIME), state))
    expected = tuple(value + P229.DT * slope for value, slope in zip(state, k2))
    end = P229.START_TIME + P229.DT
    old_end = P229.END_TIME
    P229.END_TIME = end
    try:
        assert all(np.array_equal(a, b) for a, b in zip(trajectory.state(end), expected))
    finally:
        P229.END_TIME = old_end


def test_bracket_continuation_uses_custom_tolerance_and_dyadic_backoff(monkeypatch):
    calls = []

    def fake_corrector(second_derivative, previous, current, step_size, **options):
        del second_derivative, previous, current
        calls.append((step_size, options["tolerance"]))
        success = step_size <= 1 / 256
        return {
            "success": success,
            "message": "accepted" if success else "manufactured mesh exhaustion",
            "iterations": 2,
            "mesh_nodes_used": 100,
            "point": {"tau": 0.3},
            "predicted_tau": 0.3,
            "corrected_tau": 0.3,
            "secant_norm": 0.01,
            "arclength_residual": 1e-12,
            "boundary_slope_error": 1e-14,
            "ode_second_defect": 1e-8,
        }

    monkeypatch.setattr(P229.CORE, "pseudo_arclength_step", fake_corrector)
    monkeypatch.setattr(
        P229, "refine_point",
        lambda point, trajectory, z, r, p227: (point, None, None),
    )
    monkeypatch.setattr(
        P229, "stability_value",
        lambda point, trajectory, z, r, p227: (-0.1, None, None),
    )
    result = P229.continuation_advance(
        None, {}, {}, None, None, None, None,
        initial_step=1 / 128, minimum_step=1 / 1024, tolerance=1e-7,
    )
    assert result["success"]
    assert result["accepted_step_size"] == 1 / 256
    assert calls == [(1 / 128, 1e-7), (1 / 256, 1e-7)]
    assert result["payload"]["eigenvalue"] == -0.1


def test_cross_grid_transfer_passes_identical_records():
    record = {
        "critical_time_estimate": 0.00125,
        "critical_geometry": {"one_sided_cap_area": 40.0},
        "critical_coefficients": {
            "transversality_values": [2.0, 2.0],
            "quadratic_values": [-3.0, -3.0],
        },
    }
    transfers, passed = P229.cross_grid_checks({label: record for label in ("G9", "G10", "G11")})
    assert passed
    assert all(item["passed"] for item in transfers.values())


def test_cross_grid_transfer_rejects_sign_change():
    records = {}
    for label, sign in (("G9", 1), ("G10", -1), ("G11", -1)):
        records[label] = {
            "critical_time_estimate": 0.00125,
            "critical_geometry": {"one_sided_cap_area": 40.0},
            "critical_coefficients": {
                "transversality_values": [2.0 * sign, 2.0 * sign],
                "quadratic_values": [-3.0, -3.0],
            },
        }
    _, passed = P229.cross_grid_checks(records)
    assert not passed


def test_required_input_inventory_binds_protocol228_science():
    paths = P229.input_paths(P229.ROOT.parents[2])
    assert {
        "p228/authority", "p228/result", "p228/profiles",
        "p228/G9-checkpoint", "p228/G10-checkpoint", "p228/G11-checkpoint",
        "G9/start", "G10/start", "G11/start",
    } <= set(paths)


def test_candidate_output_absent_before_science():
    assert not (P229.ROOT / "candidate-output").exists()
