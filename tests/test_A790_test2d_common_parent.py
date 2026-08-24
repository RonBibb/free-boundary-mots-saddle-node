import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test2d_common_parent", ROOT / "run_corrected_A790_test2d_common_parent.py",
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_common_parent_frame_helpers():
    mapped = {"covariant": np.broadcast_to(np.eye(2), (2, 3, 2, 2)).copy()}
    assert np.allclose(RUNNER._metric4(mapped), np.eye(4))
    extrinsic = {
        "K_DD": np.ones((2, 3)), "K_DS": np.ones((2, 3)),
        "K_SS": np.ones((2, 3)), "K_Omega": np.ones((2, 3)),
    }
    assert RUNNER._extrinsic4(extrinsic).shape == (2, 3, 4, 4)


def test_common_parent_thresholds_and_seeds_are_fixed():
    assert RUNNER.SEEDS == {"inner": 1.30, "outer": 1.55}
    assert RUNNER.CHART_MANIFEST_SHA256 is None


def test_common_parent_termination_margin_is_terminal_only():
    inverse = SimpleNamespace(normalized_depth=np.array([0.0, 0.5, 0.95]))
    chart = np.empty((81, 9))
    assert np.allclose(RUNNER._termination_margin(inverse, chart), [80.0, 40.0, 4.0])


def test_ragged_common_parent_norm_uses_local_depth_jacobian():
    U, S = np.linspace(0.0, 1.0, 9), np.linspace(0.0, 1.0, 11)
    shape = (len(U), len(S), 1)
    dmax = 1.0 + S
    left = {"value": np.ones(shape), "weight": np.ones(shape[:2]), "dmax": dmax}
    right = {"value": np.zeros(shape), "weight": np.ones(shape[:2]), "dmax": dmax}
    record = RUNNER.paired_summary(left, right, U, S)
    assert abs(record["absolute_L2"] - np.sqrt(1.5)) < 1e-12
    assert abs(record["relative_L2"] - 1.0) < 1e-12
    assert record["weighted_q95"] == 1.0
