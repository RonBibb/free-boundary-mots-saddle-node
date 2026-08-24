import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("protocol228_tested", ROOT / "protocol228.py")
P228 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P228)


def test_time_plan_is_exact_and_prospective():
    assert P228.DT == 0.00003125
    assert P228.START_STEP == 32
    assert P228.SEGMENT_STEPS == 16
    assert P228.SAMPLE_STEPS == tuple(range(48, 257, 16))
    assert [round(step * P228.DT, 7) for step in P228.SAMPLE_STEPS] == [
        0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045,
        0.005, 0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008,
    ]


def test_checkpoint_names_are_canonical():
    assert P228.checkpoint_name("G10", 48) == "G10_step0048"
    assert P228.checkpoint_name("G11", 256) == "G11_step0256"


def test_endpoint_field_shapes_distinguish_geometry_and_driver_channels():
    assert P228.expected_state_shapes("G9") == (
        (113, 211, 9), (113, 211, 9), (113, 211, 3), (113, 211, 3),
    )
    assert P228.expected_state_shapes("G10") == (
        (129, 241, 9), (129, 241, 9), (129, 241, 3), (129, 241, 3),
    )
    assert P228.expected_state_shapes("G11") == (
        (145, 271, 9), (145, 271, 9), (145, 271, 3), (145, 271, 3),
    )


def test_acceptance_fails_closed_without_branches():
    records = {label: {"distinct_cluster_count": 0, "branches": {}} for label in P228.GRIDS}
    transfers = {
        "G9-G10": {branch: {"available": False} for branch in ("inner", "outer")},
        "G10-G11": {branch: {"available": False} for branch in ("inner", "outer")},
    }
    assert not any(P228.acceptance(records, transfers).values())


def test_result_fingerprint_is_domain_separated_and_stable():
    value = {"schema": P228.SCHEMA, "classification": "TEST"}
    assert P228.result_fingerprint(value) == P228.result_fingerprint(value)
    assert P228.result_fingerprint(value) != P228.checkpoint_fingerprint(value)


def test_input_inventory_names_all_three_endpoints():
    paths = P228.input_paths(P228.ROOT.parents[2])
    assert {"G9/endpoint", "G10/endpoint", "G11/endpoint"} <= set(paths)


def test_candidate_output_absent_before_freeze_and_science():
    assert not (P228.ROOT / "candidate-output").exists()
