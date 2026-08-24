import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import run_corrected_A790_test10e_genuine_high_z_boundary_resolution as runner
from bhps.recovery_indexer import RecoveryIndex, atomic_write_npz, sha256_file


def fake_case():
    return {
        "initial": np.zeros((5, 7, 9)),
        "source0": np.zeros((5, 7, 5)),
        "z": np.linspace(0.0, 1.0, 5),
    }


def checkpoint_arrays(case, start=0, end=4, label="G9_R8"):
    count = 2 * (end - start)
    open_count = len(case["z"]) - 2
    shape = case["initial"].shape
    source_shape = case["source0"].shape
    arrays = {
        "start_step": np.asarray(start), "end_step": np.asarray(end),
        "end_position": np.zeros(shape), "end_velocity": np.zeros(shape),
        "end_source": np.zeros(source_shape), "end_memory": np.zeros(source_shape),
        "enumeration": np.arange(
            runner.enumeration(label, start + 1, 1), runner.enumeration(label, end, 2) + 1,
        ),
        "step": np.repeat(np.arange(start + 1, end + 1), 2),
        "rk_stage": np.tile((1, 2), end - start), "time": np.arange(count, dtype=float),
    }
    for key in ("before", "after", "target", "term_A", "term_V", "term_C"):
        arrays[key] = np.zeros((count, open_count, 2))
    arrays["q_perp"] = np.ones((count, open_count))
    arrays["q_zz"] = np.ones((count, open_count))
    for key in (
        "legacy_ratio", "pointwise_ratio", "maximum_absolute_correction",
        "collar_rms", "production_residual", "source_residual", "normal_wall_residual",
        "maximum_any_correction", "finite", "closure_absolute", "closure_scale",
        "closure_pass", "target_absolute", "target_scale", "target_pass",
    ):
        arrays[key] = np.zeros(count)
    for scheme in runner.SCHEMES:
        for level in runner.LEVELS:
            for metric in runner.METRIC_KEYS:
                arrays[runner.refined_key(scheme, level, metric)] = np.zeros(count)
        for metric in runner.METRIC_KEYS:
            arrays[f"{scheme}_converged_{metric}"] = np.ones(count)
    for metric in runner.METRIC_KEYS:
        arrays[f"cross_scheme_{metric}"] = np.ones(count)
        arrays[f"ensemble_{metric}"] = np.zeros(count)
    for step in range(start + 1, end + 1):
        arrays[f"step_{step:03d}_increment"] = np.zeros(shape)
        arrays[f"step_{step:03d}_velocity"] = np.zeros(shape)
        arrays[f"step_{step:03d}_source_increment"] = np.zeros(source_shape)
    return arrays


class Test10ERunnerRecovery(unittest.TestCase):
    def test_all_enumeration_ranges(self):
        self.assertEqual((runner.enumeration("G9_R8", 1, 1), runner.enumeration("G9_R8", 16, 2)), (1, 32))
        self.assertEqual((runner.enumeration("G10_R12", 1, 1), runner.enumeration("G10_R12", 16, 2)), (161, 192))
        self.assertEqual((runner.enumeration("Z9_R10", 1, 1), runner.enumeration("Z10_R10", 16, 2)), (193, 256))
        self.assertEqual((runner.enumeration("G10H_R10", 1, 1), runner.enumeration("G10H_R10", 32, 2)), (257, 320))

    def test_checkpoint_enumeration_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            case = fake_case()
            arrays = checkpoint_arrays(case)
            atomic_write_npz(path, **arrays)
            runner.validate_checkpoint(path, case, "G9_R8", 0, 4)
            arrays["enumeration"] += 1
            atomic_write_npz(path, **arrays)
            with self.assertRaisesRegex(RuntimeError, "enumeration"):
                runner.validate_checkpoint(path, case, "G9_R8", 0, 4)

    def test_restart_corruption_interruption_and_wrong_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.md"
            protocol.write_text("sealed Test10E recovery\n")
            inputs = {str(protocol): sha256_file(protocol)}
            manifest = root / "index.json"
            output = root / "checkpoint.npz"
            first = RecoveryIndex(manifest, protocol, inputs, maximum_stage_seconds=3600.0)
            first.register("physical/test", "physical-evolution-face", 60.0, {"parent_hash": "right"})
            first.mark_running("physical/test")
            atomic_write_npz(output, values=np.arange(4.0))
            first.mark_complete("physical/test", output, 1.0)
            resumed = RecoveryIndex(manifest, protocol, inputs, maximum_stage_seconds=3600.0)
            self.assertEqual(resumed.validated_path("physical/test"), output)
            with self.assertRaisesRegex(RuntimeError, "metadata"):
                resumed.register("physical/test", "physical-evolution-face", 60.0, {"parent_hash": "wrong"})
            output.write_bytes(output.read_bytes() + b"corruption")
            self.assertIsNone(resumed.validated_path("physical/test"))
            resumed.register("physical/interrupted", "physical-evolution-face", 60.0, {})
            resumed.mark_running("physical/interrupted")
            restarted = RecoveryIndex(manifest, protocol, inputs, maximum_stage_seconds=3600.0)
            self.assertEqual(restarted.data["stages"]["physical/interrupted"]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
