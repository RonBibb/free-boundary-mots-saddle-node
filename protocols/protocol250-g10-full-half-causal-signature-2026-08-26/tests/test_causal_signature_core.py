import numpy as np

from causal_signature_core import (
    causal_resolution, classify, compare_norm_records, embedded_curve,
    projected_tube_norm,
)


def test_projected_norm_removes_leaf_relabeling():
    count = 5
    lapse = np.ones(count)
    shift = np.zeros((count, 2))
    metric = np.broadcast_to(np.eye(2), (count, 2, 2)).copy()
    tangent = np.broadcast_to(np.array([1.0, 0.0]), (count, 2)).copy()
    velocity = np.broadcast_to(np.array([3.0, 2.0]), (count, 2)).copy()
    value = projected_tube_norm(lapse, shift, shift, metric, velocity, tangent)
    assert np.array_equal(value, np.full(count, 3.0))


def test_spacelike_resolution_and_unresolved_margin():
    summary, resolved, spread = causal_resolution(
        np.array([4.8, 5.8]), np.array([5.0, 6.0]), np.array([5.1, 6.1]),
    )
    assert summary["label"] == "UNIFORMLY-SPACELIKE"
    assert np.all(resolved) and np.all(spread > 0)
    summary, resolved, _ = causal_resolution(
        np.array([-0.1, 5.8]), np.array([5.0, 6.0]), np.array([10.1, 6.1]),
    )
    assert summary["label"] == "MIXED-OR-UNRESOLVED"
    assert not resolved[0]


def test_temporal_comparison_is_strict_and_path_complete():
    full = {name: np.array([10.0, 20.0]) for name in ("backward", "centered", "forward")}
    half = {name: value * 1.005 for name, value in full.items()}
    assert compare_norm_records(full, half, 0.01)["passed"]
    half["forward"] = full["forward"] * 1.02
    assert not compare_norm_records(full, half, 0.01)["passed"]


def test_embedded_curve_and_ordered_classification():
    theta = np.linspace(0.0, np.pi / 2.0, 5)
    coordinates, tangent = embedded_curve(theta, np.ones(5), np.zeros(5))
    assert coordinates.shape == tangent.shape == (5, 2)
    assert classify(True, True, True, True, True).endswith("PASS")
    assert classify(True, False, True, True, True) == "HALF-DT-CAUSAL-SIGNATURE-FAIL"
