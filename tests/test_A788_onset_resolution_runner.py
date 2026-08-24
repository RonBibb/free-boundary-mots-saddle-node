import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_corrected_A788_onset_resolution as runner


def fake_driver_stage(case, time, position, velocity, source, memory):
    slopes = (
        np.full_like(position, 1.0),
        np.full_like(velocity, 2.0),
        np.full_like(source, 3.0),
        np.full_like(memory, 4.0),
    )
    diagnostic = {
        "finite": True,
        "normal_wall_gauge": None,
        "outer_sommerfeld": None,
        "outer_source_sommerfeld": None,
    }
    return slopes, diagnostic


class A788OnsetRunnerTests(unittest.TestCase):
    def test_segmented_resume_matches_uninterrupted_index_sequence(self):
        position = np.zeros((2, 3, 9))
        state = (
            position.copy(), position.copy(),
            np.zeros((2, 3, 6)), np.zeros((2, 3, 6)),
        )
        case = {"label": "toy", "initial": position.copy()}
        with patch.object(runner.live, "driver_stage", fake_driver_stage):
            full, full_snapshots, _ = runner.integrate_segment(case, state, 0, 8)
            first, first_snapshots, _ = runner.integrate_segment(case, state, 0, 4)
            resumed, second_snapshots, _ = runner.integrate_segment(case, first, 4, 8)
        for left, right in zip(full, resumed):
            np.testing.assert_allclose(left, right, rtol=0.0, atol=0.0)
        self.assertEqual(
            set(full_snapshots), set(first_snapshots) | set(second_snapshots)
        )

    def test_segment_validator_accepts_three_component_driver_state(self):
        import tempfile

        from bhps.recovery_indexer import atomic_write_npz

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segment.npz"
            metric_shape = (2, 3, 9)
            source_shape = (2, 3, 3)
            arrays = {
                "start_step": np.asarray(0), "end_step": np.asarray(1),
                "end_position": np.zeros(metric_shape),
                "end_velocity": np.zeros(metric_shape),
                "end_source": np.zeros(source_shape),
                "end_memory": np.zeros(source_shape),
                "step_001_increment": np.zeros(metric_shape),
                "step_001_velocity": np.zeros(metric_shape),
            }
            atomic_write_npz(path, **arrays)
            runner.validate_segment(path, metric_shape, source_shape, 0, 1)


if __name__ == "__main__":
    unittest.main()
