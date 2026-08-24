import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("continuation_core_tested", ROOT / "continuation_core.py")
CORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORE)


def constant_point(rho, tau):
    theta = np.linspace(1e-4, np.pi / 2, 101)
    return {"theta": theta, "rho": np.full_like(theta, rho), "slope": np.zeros_like(theta), "tau": tau}


def test_product_norm_and_secant_are_normalized():
    theta = np.linspace(1e-4, np.pi / 2, 101)
    first = constant_point(1.0, 0.0)
    second = constant_point(1.1, 0.2)
    rho, tau, norm = CORE.normalized_secant(first, second, theta)
    assert norm > 0
    assert abs(CORE.product_norm(theta, rho, tau) - 1.0) < 1e-12


def test_physical_mode_is_scaled_on_interior_before_neumann_extension():
    extension = np.array([
        [2.0, -1.0, 0.5],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [-0.5, 1.5, 2.0],
    ])
    normal_factor = np.array([9.0, 2.0, 4.0, 8.0, 7.0])
    physical_mode = np.array([4.0, 8.0, 16.0])
    expected = extension @ np.array([2.0, 2.0, 2.0])
    observed = CORE.physical_mode_to_radial(
        extension, normal_factor, physical_mode,
    )
    assert observed.shape == (5,)
    assert np.array_equal(observed, expected)


def test_physical_mode_mapping_rejects_interior_length_mismatch():
    with np.testing.assert_raises(ValueError):
        CORE.physical_mode_to_radial(
            np.ones((5, 3)), np.ones(5), np.ones(5),
        )


def test_dyadic_backoff_accepts_first_success_and_archives_all_attempts():
    seen = []

    def attempt(step):
        seen.append(step)
        passed = step <= 1 / 256
        return {
            "success": passed,
            "reason": "accepted" if passed else "manufactured-corrector-failure",
            "metrics": {"manufactured_step": step},
            "payload": {"value": 7} if passed else None,
        }

    result = CORE.dyadic_backoff(attempt, 1 / 64, 1 / 4096)
    assert result["success"]
    assert result["accepted_step_size"] == 1 / 256
    assert seen == [1 / 64, 1 / 128, 1 / 256]
    assert result["payload"] == {"value": 7}
    assert [item["success"] for item in result["attempts"]] == [False, False, True]


def test_dyadic_backoff_fails_closed_at_frozen_floor():
    result = CORE.dyadic_backoff(
        lambda step: {
            "success": False, "reason": "always-fails",
            "metrics": {"step": step}, "payload": None,
        },
        1 / 64,
        1 / 256,
    )
    assert not result["success"]
    assert result["accepted_step_size"] is None
    assert [item["step_size"] for item in result["attempts"]] == [1 / 64, 1 / 128, 1 / 256]


def test_manufactured_fold_is_crossed_by_pseudo_arclength():
    # Constant Neumann solutions of rho''=rho^2-tau are rho=+/-sqrt(tau).
    second = lambda tau, theta, rho, slope: rho**2 - tau
    previous = constant_point(np.sqrt(0.25), 0.25)
    current = constant_point(np.sqrt(0.20), 0.20)
    points = [previous, current]
    for _ in range(40):
        result = CORE.pseudo_arclength_step(
            second, points[-2], points[-1], 0.025,
            nodes=51, dense_nodes=101, tolerance=1e-7,
        )
        assert result["success"]
        assert result["arclength_residual"] < 1e-6
        points.append(result["point"])
        if np.mean(points[-1]["rho"]) < -0.05:
            break
    assert min(point["tau"] for point in points) < 2e-3
    assert np.mean(points[-1]["rho"]) < 0


def test_principal_left_right_normalization_for_nonsymmetric_matrix():
    modes = CORE.principal_modes(np.array([[0.0, 2.0], [1.0, 3.0]]))
    assert abs(modes["overlap"] - 1.0) < 1e-12
    assert abs(modes["eigenvalue"].imag) < 1e-12
    assert modes["eigenvalue"].real < modes["next_eigenvalue"].real


def test_square_root_fit_recovers_exact_critical_time_and_exponent():
    critical = 0.123
    times = critical + np.array([1, 2, 4, 8, 16]) * 1e-4
    separation = 3.5 * np.sqrt(times - critical)
    fit = CORE.linear_square_root_fit(times, separation)
    assert abs(fit["critical_time"] - critical) < 1e-12
    assert fit["R_squared"] > 1 - 1e-12
    assert abs(fit["log_exponent"] - 0.5) < 1e-12
