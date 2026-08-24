import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test2d_fields", ROOT / "run_corrected_A790_test2d_fields.py",
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_metric_and_extrinsic_four_frames():
    metric = {"covariant": np.broadcast_to(np.eye(2), (3, 4, 2, 2)).copy()}
    metric4 = RUNNER._metric4(metric)
    assert metric4.shape == (3, 4, 4, 4)
    assert np.allclose(metric4, np.eye(4))
    extrinsic = {
        "K_DD": np.ones((3, 4)), "K_DS": 2.0 * np.ones((3, 4)),
        "K_SS": 3.0 * np.ones((3, 4)), "K_Omega": 4.0 * np.ones((3, 4)),
    }
    K = RUNNER._extrinsic4(extrinsic)
    assert np.all(K[..., 0, 1] == 2.0)
    assert np.all(K[..., 1, 0] == 2.0)
    assert np.all(K[..., 3, 3] == 4.0)


def test_field_observables_and_sequences_are_fixed():
    assert RUNNER.OBSERVABLES == (
        "final_metric", "metric_increment", "ADM_K", "areal_radius",
    )
    assert RUNNER.SPATIAL == ("G9", "G10", "G11")
    assert RUNNER.TEMPORAL == ("G10_coarse", "G10_standard", "G10_half")


def test_runner_refuses_unsealed_chart_parent():
    assert RUNNER.CHART_MANIFEST_SHA256 is None
