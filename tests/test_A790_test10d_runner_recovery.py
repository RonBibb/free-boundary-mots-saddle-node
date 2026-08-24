import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import run_corrected_A790_test10d_boundary_resolution as runner
from bhps.recovery_indexer import (
    RecoveryIndex,
    atomic_write_npz,
    sha256_file,
)


def checkpoint_arrays(count=2, enumeration_start=1):
    arrays = {
        key: np.zeros(count) for key in (
            "enumeration", "step", "rk_stage", "time", "legacy_ratio",
            "pointwise_ratio", "maximum_absolute_correction", "collar_rms",
            "closure_absolute", "closure_scale", "closure_pass",
            "target_absolute", "target_scale", "target_pass",
            "production_residual", "production_pass", "stage_position_error",
        )
    }
    arrays["enumeration"] = np.arange(enumeration_start, enumeration_start + count)
    for scheme in runner.SCHEMES:
        for level in runner.LEVELS:
            for metric in runner.METRIC_KEYS:
                arrays[runner.refined_key(scheme, level, metric)] = np.zeros(count)
        for metric in runner.METRIC_KEYS:
            arrays[f"{scheme}_converged_{metric}"] = np.ones(count)
    for metric in runner.METRIC_KEYS:
        arrays[f"cross_scheme_{metric}"] = np.ones(count)
        arrays[f"ensemble_{metric}"] = np.zeros(count)
    return arrays


class Test10DRunnerRecovery(unittest.TestCase):
    def test_checkpoint_enumeration_and_corruption_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.npz"
            atomic_write_npz(path, **checkpoint_arrays())
            runner.validate_checkpoint(path, 2, 1, 2)
            arrays = checkpoint_arrays()
            arrays["enumeration"] = np.asarray([2, 3])
            atomic_write_npz(path, **arrays)
            with self.assertRaisesRegex(RuntimeError, "enumeration"):
                runner.validate_checkpoint(path, 2, 1, 2)

    def test_recovery_restart_and_hash_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.md"
            protocol.write_text("sealed Test10D recovery\n")
            inputs = {str(protocol): sha256_file(protocol)}
            manifest = root / "index.json"
            output = root / "checkpoint.npz"
            first = RecoveryIndex(manifest, protocol, inputs, maximum_stage_seconds=1200.0)
            first.register("analysis/test", "refined-boundary-analysis", 60.0, {"e": [1, 2]})
            first.mark_running("analysis/test")
            atomic_write_npz(output, **checkpoint_arrays())
            first.mark_complete("analysis/test", output, 1.0)
            original_hash = sha256_file(output)
            resumed = RecoveryIndex(manifest, protocol, inputs, maximum_stage_seconds=1200.0)
            self.assertEqual(resumed.validated_path("analysis/test"), output)
            self.assertEqual(sha256_file(resumed.validated_path("analysis/test")), original_hash)
            output.write_bytes(output.read_bytes() + b"corruption")
            self.assertIsNone(resumed.validated_path("analysis/test"))

    def test_running_stage_returns_to_pending_after_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.md"
            protocol.write_text("sealed Test10D interruption\n")
            inputs = {str(protocol): sha256_file(protocol)}
            manifest = root / "index.json"
            first = RecoveryIndex(manifest, protocol, inputs, maximum_stage_seconds=1200.0)
            first.register("analysis/test", "refined-boundary-analysis", 60.0, {})
            first.mark_running("analysis/test")
            resumed = RecoveryIndex(manifest, protocol, inputs, maximum_stage_seconds=1200.0)
            self.assertEqual(resumed.data["stages"]["analysis/test"]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
