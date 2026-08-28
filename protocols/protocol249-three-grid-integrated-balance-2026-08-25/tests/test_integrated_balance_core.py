import pytest

from integrated_balance_core import compare_segments, classify, segment_record, trapezoid


TERMS = {
    "coupled_seam_global_radius": -20.0,
    "coupled_seam_joint_work": 300.0,
    "matter": 220.0,
}


def local(scale=1.0):
    steps = {}
    for step in (44, 45, 46, 47):
        ledger = {
            "target_rate": 500.0 * scale,
            "total_flux": 500.0 * scale,
            "balance_norm": 1500.0 * scale,
            "terms": {name: value * scale for name, value in TERMS.items()},
        }
        record = {"ledger": ledger}
        steps[str(step)] = {"records": {str(width): {"centered": record} for width in (5, 7, 9)}}
    return {"steps": steps}


def charges(scale=1.0):
    dt = 3.125e-5
    return {step: scale * (10.0 + 500.0 * dt * (step - 44)) for step in (44, 45, 46, 47)}


def test_trapezoid_constant():
    assert trapezoid([2.0, 2.0, 2.0, 2.0]) == pytest.approx(1.875e-4)


def test_exact_segment_passes_and_records_brane_flux():
    result = segment_record(local(), charges(), 7)
    assert result["passed"]
    assert result["integrated_brane_endpoint_flux"] == pytest.approx(0.02625)


def test_physical_flux_failure_is_detected():
    datum = local()
    for step in datum["steps"].values():
        for width in step["records"].values():
            width["centered"]["ledger"]["total_flux"] = 700.0
    assert not segment_record(datum, charges(), 7)["passed"]


def test_segment_comparison_and_ordered_taxonomy():
    left = segment_record(local(), charges(), 7)
    right = segment_record(local(1.0001), charges(1.0001), 7)
    assert compare_segments(left, right)["passed"]
    gates = {name: True for name in (
        "prerequisite_admission", "segment_orientation", "charge_quadrature_closure",
        "integrated_flux_closure", "brane_ledger_completeness", "stencil_robustness",
        "spatial_transfer", "temporal_control_admission",
    )}
    assert classify(gates) == "G9-G10-G11-FINITE-SEGMENT-INTEGRATED-BALANCE-PASS"
    for name in tuple(gates):
        case = dict(gates); case[name] = False
        assert classify(case).endswith("FAIL")
