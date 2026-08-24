import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "test2d_chart_build", ROOT / "run_corrected_A790_test2d_chart_build.py",
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_exactly_ninety_unique_prospective_chart_tasks():
    tasks = RUNNER.chart_tasks()
    identifiers = [RUNNER.task_id(task) for task in tasks]
    filenames = [RUNNER.task_filename(task) for task in tasks]
    assert len(tasks) == 90
    assert len(set(identifiers)) == 90
    assert len(set(filenames)) == 90
    assert sum(task["family"] == "aligned" for task in tasks) == 54
    assert sum(task["family"] == "common_parent" for task in tasks) == 36


def test_resolution_counts_are_frozen():
    assert RUNNER.RESOLUTIONS == {
        "coarse": {"ray_count": 193, "distance_samples": 129, "coarse": True},
        "primary": {"ray_count": 257, "distance_samples": 193, "coarse": False},
        "fine": {"ray_count": 385, "distance_samples": 257, "coarse": False},
    }


def test_sealed_parent_hashes():
    assert RUNNER.sha256_file(ROOT / RUNNER.PROTOCOL) == RUNNER.PROTOCOL_SHA256
    assert RUNNER.sha256_file(ROOT / RUNNER.MACHINE_PROTOCOL) == RUNNER.MACHINE_PROTOCOL_SHA256
    assert RUNNER.sha256_file(ROOT / RUNNER.QUALIFICATION) == RUNNER.QUALIFICATION_SHA256
    assert RUNNER.sha256_file(ROOT / RUNNER.ALIGNED) == RUNNER.ALIGNED_SHA256
    assert RUNNER.sha256_file(ROOT / RUNNER.GEOMETRIES) == RUNNER.GEOMETRIES_SHA256
    assert RUNNER.sha256_file(ROOT / RUNNER.COMMON_PARENT) == RUNNER.COMMON_PARENT_SHA256
