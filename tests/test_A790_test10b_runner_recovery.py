import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import run_corrected_A790_test10b_domain_normalized as runner
from bhps.recovery_indexer import RecoveryIndex, sha256_file


class Test10BRunnerRecoveryTests(unittest.TestCase):
    @staticmethod
    def case():
        shape = (2, 3, 9)
        source_shape = (2, 3, 4)
        initial = np.zeros(shape)
        jet = SimpleNamespace(
            reduced_first=np.zeros((3, *shape)),
            reduced_second=np.zeros((3, 3, *shape)),
        )
        return {
            "label": "mock-test10b",
            "z": np.linspace(1.0, 2.0, shape[0]),
            "r": np.linspace(0.0, 1.0, shape[1]),
            "initial": initial,
            "source0": np.zeros(source_shape),
            "memory0": np.zeros(source_shape),
            "geometry": {"jet_field": jet, "background": None},
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
        protocol.write_text("sealed Test-10B recovery protocol\n")
        return RecoveryIndex(
            root / "index.json", protocol,
            {str(protocol): sha256_file(protocol)}, maximum_stage_seconds=2400.0,
        )

    def test_segmented_run_resumes_and_rebuilds_only_corrupt_segment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = self.index(root)
            signature = {"all_points_one_negative_direction": True}
            with (
                patch.object(runner, "RECOVERY_ROOT", root),
                patch.object(runner, "STEPS", 4),
                patch.object(runner, "SEGMENT", 2),
                patch.object(runner, "DT", 0.01),
                patch.object(
                    runner.live, "driver_stage", side_effect=self.driver_stage,
                ) as driver,
                patch.object(
                    runner.live, "signature_summary", return_value=signature,
                ),
                patch.object(
                    runner.live, "compact_wall_position_residuals",
                    return_value={"maximum": 0.0},
                ),
                patch.object(
                    runner.live, "compact_wall_normal_gauge_position_residuals",
                    return_value={"maximum": 0.0},
                ),
            ):
                state, diagnostics, paths = runner.run_evolution(
                    index, "mock", self.case(),
                )
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
                step_001_increment=np.zeros((1,)),
                step_001_velocity=np.zeros((1,)),
                step_001_source_increment=np.zeros((1,)),
            )
            with self.assertRaisesRegex(ValueError, "shape"):
                runner.validate_segment(path, self.case(), 0, 1)

    def test_changed_upstream_segment_breaks_parent_hash_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = self.index(root)
            signature = {"all_points_one_negative_direction": True}
            with (
                patch.object(runner, "RECOVERY_ROOT", root),
                patch.object(runner, "STEPS", 4),
                patch.object(runner, "SEGMENT", 2),
                patch.object(runner, "DT", 0.01),
                patch.object(
                    runner.live, "driver_stage", side_effect=self.driver_stage,
                ),
                patch.object(
                    runner.live, "signature_summary", return_value=signature,
                ),
                patch.object(
                    runner.live, "compact_wall_position_residuals",
                    return_value={"maximum": 0.0},
                ),
                patch.object(
                    runner.live, "compact_wall_normal_gauge_position_residuals",
                    return_value={"maximum": 0.0},
                ),
            ):
                _, _, paths = runner.run_evolution(index, "mock", self.case())
                with np.load(paths[0]) as archive:
                    arrays = {key: np.asarray(archive[key]) for key in archive.files}
                arrays["end_position"] = arrays["end_position"].copy()
                arrays["end_position"][0, 0, 0] += 1e-12
                runner.atomic_write_npz(paths[0], **arrays)
                index.mark_complete(
                    "evolution/mock/steps_001_002", paths[0], 0.0,
                )
                with self.assertRaisesRegex(RuntimeError, "metadata mismatch"):
                    runner.run_evolution(index, "mock", self.case())


if __name__ == "__main__":
    unittest.main()
