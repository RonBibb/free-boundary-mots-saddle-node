import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.recovery_indexer import (
    RecoveryIndex,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
    validate_npz,
)


class RecoveryIndexerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.protocol = self.root / "protocol.md"
        self.source = self.root / "source.txt"
        self.protocol.write_text("sealed\n")
        self.source.write_text("input\n")
        self.expected = {str(self.source): sha256_file(self.source)}

    def tearDown(self):
        self.temp.cleanup()

    def index(self):
        return RecoveryIndex(
            self.root / "index.json", self.protocol, self.expected,
        )

    def test_atomic_npz_and_validated_resume(self):
        index = self.index()
        index.register("grid/G7/steps/1-4", "evolution", 120.0, {"end": 4})
        index.mark_running("grid/G7/steps/1-4")
        output = self.root / "stage.npz"
        atomic_write_npz(output, state=np.ones((2, 3)), step=np.asarray(4))
        validate_npz(output, {"state": (2, 3), "step": ()})
        index.mark_complete("grid/G7/steps/1-4", output, 1.0)
        resumed = self.index()
        self.assertEqual(resumed.validated_path("grid/G7/steps/1-4"), output)

    def test_npz_validation_accepts_unicode_metadata_and_still_rejects_nonfinite(self):
        output = self.root / "metadata.npz"
        atomic_write_npz(
            output,
            state=np.ones((2, 3)),
            schema=np.asarray("protocol-125-v1"),
            configuration_json=np.asarray('{"mode":"owner-last"}'),
        )
        record = validate_npz(
            output,
            {"state": (2, 3), "schema": (), "configuration_json": ()},
        )
        self.assertIn("schema", record["keys"])

        bad = self.root / "nonfinite.npz"
        atomic_write_npz(bad, state=np.asarray((1.0, np.nan)), schema=np.asarray("v1"))
        with self.assertRaisesRegex(ValueError, "nonfinite NPZ arrays"):
            validate_npz(bad)

    def test_corrupted_stage_is_not_resumed(self):
        index = self.index()
        index.register("s", "detector", 60.0)
        output = self.root / "stage.json"
        atomic_write_json(output, {"ok": True})
        index.mark_complete("s", output, 0.1)
        output.write_text(json.dumps({"ok": False}))
        self.assertIsNone(index.validated_path("s"))

    def test_running_stage_reverts_to_pending(self):
        index = self.index()
        index.register("s", "detector", 60.0)
        index.mark_running("s")
        resumed = self.index()
        self.assertEqual(resumed.data["stages"]["s"]["status"], "pending")

    def test_duration_policy_and_provenance_are_enforced(self):
        index = self.index()
        with self.assertRaises(ValueError):
            index.register("too-long", "evolution", 3600.1)
        self.source.write_text("changed\n")
        with self.assertRaises(RuntimeError):
            self.index()


if __name__ == "__main__":
    unittest.main()
