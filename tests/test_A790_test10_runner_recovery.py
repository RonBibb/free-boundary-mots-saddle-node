import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import run_corrected_A790_test10_joint_convergence as runner
from bhps.recovery_indexer import RecoveryIndex, sha256_file


class Test10RunnerRecoveryTests(unittest.TestCase):
    @staticmethod
    def case():
        shape = (2, 3, 9)
        source_shape = (2, 3, 4)
        return {
            "label": "mock-test10",
            "z": np.linspace(1.0, 2.0, shape[0]),
            "r": np.linspace(0.0, 1.0, shape[1]),
            "initial": np.zeros(shape),
            "source0": np.zeros(source_shape),
            "memory0": np.zeros(source_shape),
        }

    @staticmethod
    def driver_stage(case, time_value, position, velocity, source, memory):
        del case, time_value
        slopes = (
            np.full_like(position, 0.1),
            np.full_like(velocity, 0.2),
            np.full_like(source, 0.3),
            np.full_like(memory, 0.4),
        )
        diagnostic = {
            "finite": True,
            "normal_wall_gauge": None,
            "outer_sommerfeld": None,
            "outer_source_sommerfeld": None,
        }
        return slopes, diagnostic

    def index(self, root):
        protocol = root / "protocol.md"
        protocol.write_text("sealed test protocol\n")
        return RecoveryIndex(
            root / "index.json", protocol,
            {str(protocol): sha256_file(protocol)}, maximum_stage_seconds=2400.0,
        )

    def test_segmented_run_resumes_and_rebuilds_only_corrupt_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = self.index(root)
            spec = {"mock": {"grid": "mock", "steps": 4, "dt": 0.01, "segment": 2}}
            signature = {"all_points_one_negative_direction": True}
            with (
                patch.object(runner, "RECOVERY_ROOT", root),
                patch.object(runner, "EVOLUTION_SPECS", spec),
                patch.object(runner.live, "driver_stage", side_effect=self.driver_stage) as driver,
                patch.object(runner.live, "signature_summary", return_value=signature),
            ):
                state, diagnostics, paths = runner.run_evolution(index, "mock", self.case())
                self.assertEqual(driver.call_count, 8)
                expected = tuple(value.copy() for value in state)
                self.assertEqual(len(diagnostics), 2)

                driver.reset_mock(side_effect=True)
                driver.side_effect = AssertionError("valid segments must be reused")
                resumed, _, _ = runner.run_evolution(index, "mock", self.case())
                for left, right in zip(expected, resumed):
                    np.testing.assert_array_equal(left, right)
                self.assertEqual(driver.call_count, 0)

                paths[0].write_bytes(paths[0].read_bytes() + b"corruption")
                driver.side_effect = self.driver_stage
                rebuilt, _, _ = runner.run_evolution(index, "mock", self.case())
                self.assertEqual(driver.call_count, 4)
                for left, right in zip(expected, rebuilt):
                    np.testing.assert_array_equal(left, right)

    def test_segment_validation_rejects_wrong_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            np.savez(
                path, start_step=np.asarray(0), end_step=np.asarray(1),
                end_position=np.zeros((1,)), end_velocity=np.zeros((1,)),
                end_source=np.zeros((1,)), end_memory=np.zeros((1,)),
                step_001_increment=np.zeros((1,)), step_001_velocity=np.zeros((1,)),
            )
            with self.assertRaisesRegex(ValueError, "shape"):
                runner.validate_segment(path, self.case(), 0, 1)


if __name__ == "__main__":
    unittest.main()
