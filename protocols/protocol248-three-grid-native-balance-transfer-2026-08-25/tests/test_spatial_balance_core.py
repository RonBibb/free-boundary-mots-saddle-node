import pytest

from spatial_balance_core import compare_centered, classify, symmetric_relative


def step(scale=1.0, sign=1.0):
    record = {
        "area_transport": {"finite_difference_rate": 50.0 * scale, "marginal_integral_rate": 50.01 * scale},
        "seam": {"c_geometric": -1.0 * scale, "c_wall": -1.0 * scale},
        "native_wall_rate": {
            "epsilon_records": {"5.0e-07": {"directional_rate": -0.006 * scale * sign}},
            "history_rate": -0.006 * scale * sign, "wall_rate": -0.006 * scale * sign,
        },
        "ledger": {
            "target_rate": 600.0 * scale, "total_flux": 600.1 * scale,
            "balance_norm": 1800.0 * scale, "normalized_absolute_residual": 0.001,
            "terms": {"matter": 900.0 * scale, "geometry": -299.9 * scale},
        },
    }
    return {"step": 44, "area_values": {"44": 40.0 * scale}, "records": {str(width): {"centered": record} for width in (5, 7, 9)}}


def test_symmetric_relative():
    assert symmetric_relative(10.0, 9.0) == pytest.approx(0.1)


def test_matched_balance_passes():
    result = compare_centered(step(), step(1.0001), 7)
    assert result["passed"]
    assert result["sign_pass"] and result["geometry_pass"] and result["balance_pass"] and result["native_operator_pass"]


def test_sign_or_large_transfer_fails():
    assert not compare_centered(step(), step(1.0, sign=-1.0), 7)["passed"]
    assert not compare_centered(step(), step(1.2), 7)["passed"]


def test_ordered_classification():
    good = {name: True for name in (
        "parent_admission", "local_balance_admission", "time_alignment", "sign_consistency",
        "geometry_consistency", "balance_consistency", "native_operator_consistency",
    )}
    assert classify(good) == "G9-G10-G11-NATIVE-LOCAL-BALANCE-TRANSFER-PASS"
    for key in tuple(good):
        case = dict(good); case[key] = False
        assert classify(case).endswith("FAIL")
