import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test2d_surfaces", ROOT / "run_corrected_A790_test2d_surfaces.py",
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_surface_task_set_and_seeds_are_fixed():
    tasks = [(label, branch) for label in (*RUNNER.SPATIAL, *RUNNER.TEMPORAL) for branch in RUNNER.BRANCHES]
    assert len(tasks) == 12 and len(set(tasks)) == 12
    assert RUNNER.SEEDS == {"inner": 1.30, "outer": 1.55}


def test_profile_norms_are_exact_for_constant():
    record = RUNNER._profile_norm(np.ones(501))
    assert abs(record["L2"] - 1.0) < 1e-12
    assert record["q95"] == 1.0


def test_termination_margin_exempts_axis_and_brane_boundaries():
    inverse = SimpleNamespace(normalized_depth=np.array([0.0, 0.25, 0.9]))
    chart = np.empty((101, 7))
    assert np.allclose(RUNNER._termination_margin(inverse, chart), [100.0, 75.0, 10.0])


def test_stability_collar_matches_fixed_operator_queries_on_flat_control():
    z, r = np.linspace(0.0, 4.0, 25), np.linspace(0.0, 3.0, 33)
    position = np.zeros((len(z), len(r), 9))
    position[..., 2], position[..., 3], position[..., 6] = -1.0, 1.0, 1.0
    velocity = np.zeros_like(position)
    velocity[..., 3] = velocity[..., 6] = -1.6
    prepared = RUNNER.prepare_capped_expansion_slice(position, velocity, z, r)
    theta = np.linspace(0.0, np.pi / 2.0, 501)
    points = RUNNER._stability_collar_points(
        prepared, {"theta": theta, "rho": np.full_like(theta, 1.1)},
    )
    assert points.shape == (709, 2)
    assert np.all(np.isfinite(points))
    assert np.min(points[:, 0]) > z[0] and np.max(points[:, 0]) <= z[-1]
    assert np.min(points[:, 1]) > r[0] and np.max(points[:, 1]) < r[-1]


def test_runner_refuses_unsealed_chart_parent():
    assert RUNNER.CHART_MANIFEST_SHA256 is None
