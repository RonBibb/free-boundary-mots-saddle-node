import numpy as np

import run_corrected_A790_test2b_invariant_convergence as run


def synthetic_history(steps=8):
    time = np.linspace(0.0, 1.0, steps + 1)
    position = np.zeros((steps + 1, 3, 4, 9))
    velocity = np.zeros_like(position)
    position[..., 2] = -4.0
    position[..., 3] = 1.0
    position[..., 6] = 1.0
    position[..., 0] = time[:, None, None]
    velocity[..., 0] = 1.0
    return {"time": time, "position": position, "velocity": velocity}


def test_proper_time_and_alignment_use_brane_axis_clock():
    history = synthetic_history()
    tau = run.proper_time(history)
    assert np.max(np.abs(tau - 2.0 * history["time"])) < 1e-14
    position, velocity = run.align_history(history, 0.75)
    assert np.max(np.abs(position[..., 0] - 0.375)) < 1e-14
    assert np.max(np.abs(velocity[..., 0] - 1.0)) < 1e-14


def test_formation_bracket_uses_preceding_zero_slice():
    tau = np.linspace(0.0, 0.8, 9)
    assert run._bracket([0, 0, 0, 0, 2, 2, 2, 2], tau) == [0.4, 0.5]


def test_scalar_tail_recovers_second_order_and_rejects_growth():
    counts = np.asarray((112.0, 128.0, 144.0))
    passed = run.scalar_tail(counts**-2)
    assert passed["passed"]
    assert passed["order_interval"][0] > 1.99
    failed = run.scalar_tail([1.0, 0.9, 1.2])
    assert not failed["passed"]


def test_pair_summary_zero_and_nonzero_controls():
    distance = np.linspace(0.0, 1.0, 17)
    radius = np.linspace(0.1, 2.0, 25)
    shape = (len(distance), len(radius), 2)
    weight = np.ones(shape[:2])
    left = {"value": np.ones(shape), "weight": weight}
    same = run.paired_summary(left, left, distance, radius)
    assert same["absolute_L2"] == 0.0
    right = {"value": np.zeros(shape), "weight": weight}
    changed = run.paired_summary(left, right, distance, radius)
    assert changed["absolute_L2"] > 0.0
    assert changed["relative_L2"] == 1.0
