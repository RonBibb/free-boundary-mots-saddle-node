import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test2d_qualification", ROOT / "run_corrected_A790_test2d_qualification.py",
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_protocol_hashes_are_sealed():
    assert RUNNER.sha256_file(ROOT / RUNNER.PROTOCOL) == RUNNER.PROTOCOL_SHA256
    assert RUNNER.sha256_file(ROOT / RUNNER.MACHINE_PROTOCOL) == RUNNER.MACHINE_PROTOCOL_SHA256


def test_parent_hashes_are_fixed():
    for path, expected in RUNNER.PARENTS.items():
        assert RUNNER.sha256_file(ROOT / path) == expected
