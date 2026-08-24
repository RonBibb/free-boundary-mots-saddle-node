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

import run_corrected_A790_test10c_outer_scalar_closure as runner
from bhps.recovery_indexer import RecoveryIndex, sha256_file


class Test10CRunnerRecoveryTests(unittest.TestCase):
    @staticmethod
    def case():
        shape = (5, 6, 9)
        source_shape = (5, 6, 4)
        initial = np.zeros(shape)
        jet = SimpleNamespace(
            reduced_first=np.zeros((3, *shape)),
            reduced_second=np.zeros((3, 3, *shape)),
        )
        return {
            "initial": initial, "source0": np.zeros(source_shape),
            "memory0": np.zeros(source_shape),
            "z": np.linspace(1.0, 2.0, shape[0]),
            "r": np.linspace(0.0, 1.0, shape[1]),
            "geometry": {"jet_field": jet},
            "_test10c_label": "mock",
        }

    @staticmethod
    def fake_metrics():
        metrics = {key: 0.0 for key in runner.METRIC_KEYS}
        metrics.update({
            "proper": {
                "simpson": {"ratio": 0.0, "term_balance_ratio": 0.0},
                "trapezoid": {"ratio": 0.0, "term_balance_ratio": 0.0},
            },
            "pointwise": {"maximum": 0.0, "index": [0, 0]},
            "component_legacy_ratios": [0.0, 0.0],
            "component_proper_ratios": [0.0, 0.0],
            "component_proper_trapezoid_ratios": [0.0, 0.0],
            "finite": True,
        })
        return metrics

    @classmethod
    def instrument(cls, case, time_value, state):
        del time_value
        position, velocity, source, memory = state
        slopes = (
            np.full_like(position, 0.1), np.full_like(velocity, 0.2),
            np.full_like(source, 0.3), np.full_like(memory, 0.4),
        )
        raw = {
            key: np.zeros((len(case["z"]) - 2, 2))
            for key in ("before", "after", "target", "term_A", "term_V", "term_C")
        }
        return slopes, cls.fake_metrics(), raw

    @staticmethod
    def source_archive(path, case, dt=0.01):
        arrays = {}
        for step in (1, 2):
            arrays[f"step_{step:03d}_increment"] = np.full_like(
                case["initial"], step * dt * 0.1,
            )
            arrays[f"step_{step:03d}_velocity"] = np.full_like(
                case["initial"], step * dt * 0.2,
            )
            arrays[f"step_{step:03d}_source_increment"] = np.full_like(
                case["source0"], step * dt * 0.3,
            )
        arrays.update({
            "end_position": arrays["step_002_increment"],
            "end_velocity": arrays["step_002_velocity"],
            "end_source": arrays["step_002_source_increment"],
            "end_memory": np.full_like(case["memory0"], 2 * dt * 0.4),
        })
        np.savez(path, **arrays)

    @staticmethod
    def index(root):
        protocol = root / "protocol.md"
        protocol.write_text("sealed Test10C recovery\n")
        return RecoveryIndex(
            root / "index.json", protocol,
            {str(protocol): sha256_file(protocol)}, maximum_stage_seconds=2400.0,
        )

    def test_replay_resumes_and_rebuilds_corrupt_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = self.case()
            source = root / "source.npz"
            self.source_archive(source, case)
            index = self.index(root)
            with (
                patch.object(runner, "RECOVERY_ROOT", root),
                patch.object(runner.test10b, "DT", 0.01),
                patch.object(runner.test10b, "segment_path", return_value=source),
                patch.object(runner, "instrument_stage", side_effect=self.instrument) as replay,
            ):
                path = runner.replay_segment(
                    index, 0, "mock", case, 0, 2, "parent",
                )
                self.assertEqual(replay.call_count, 4)
                replay.reset_mock(side_effect=True)
                replay.side_effect = AssertionError("valid checkpoint must be reused")
                resumed = runner.replay_segment(
                    index, 0, "mock", case, 0, 2, "parent",
                )
                self.assertEqual(path, resumed)
                self.assertEqual(replay.call_count, 0)
                path.write_bytes(path.read_bytes() + b"corruption")
                replay.side_effect = self.instrument
                runner.replay_segment(index, 0, "mock", case, 0, 2, "parent")
                self.assertEqual(replay.call_count, 4)

    def test_wrong_enumeration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            count = 2
            arrays = {
                "enumeration": np.asarray([2, 3]),
                "step": np.asarray([1, 1]), "rk_stage": np.asarray([1, 2]),
                "time": np.zeros(count),
                "before": np.zeros((count, 3, 2)),
                "after": np.zeros((count, 3, 2)),
                "target": np.zeros((count, 3, 2)),
                "term_A": np.zeros((count, 3, 2)),
                "term_V": np.zeros((count, 3, 2)),
                "term_C": np.zeros((count, 3, 2)),
                "component_legacy": np.zeros((count, 2)),
                "component_proper": np.zeros((count, 2)),
                "component_proper_trapezoid": np.zeros((count, 2)),
                "step_replay_error": np.zeros((1, 4)),
                **{key: np.zeros(count) for key in runner.FLAT_METRIC_KEYS},
            }
            np.savez(path, **arrays)
            with self.assertRaisesRegex(RuntimeError, "enumeration"):
                runner.validate_checkpoint(path, 3, count, 1, 2)


if __name__ == "__main__":
    unittest.main()
