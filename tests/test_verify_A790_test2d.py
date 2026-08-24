import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_test2d", ROOT / "verify_corrected_A790_test2d.py",
)
VERIFY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFY)


def test_independent_generalized_order_recovers_second_order():
    n1, n2, n3 = 112.0, 128.0, 144.0
    d12 = n1**-2 - n2**-2
    d23 = n2**-2 - n3**-2
    assert abs(VERIFY.generalized_order(d12, d23) - 2.0) < 1e-10


def test_independent_interval_rejects_zero_lower_bound():
    assert VERIFY.order_interval(1.0, 1.0, 0.5, 0.1) is None


def test_independent_surface_coverage_recomputes_profile_and_collar_gates():
    coverage = {
        "profile_points": 501,
        "stability_collar_points": 801,
        "fine_unique_points": 501,
        "primary_unique_points": 501,
        "fine_unique_stability_collar_points": 801,
        "primary_unique_stability_collar_points": 801,
        "fine_minimum_termination_margin": 4.1,
        "primary_minimum_termination_margin": 4.2,
        "fine_collar_minimum_termination_margin": 4.3,
        "primary_collar_minimum_termination_margin": 4.4,
        "inverse_residual_limit": 1e-8,
        "fine_inverse_residual_maximum": 1e-10,
        "primary_inverse_residual_maximum": 2e-10,
        "fine_collar_inverse_residual_maximum": 3e-10,
        "primary_collar_inverse_residual_maximum": 4e-10,
        "passed": True,
    }
    assert VERIFY.verify_surface_coverage(coverage)
    coverage["fine_unique_stability_collar_points"] -= 1
    assert not VERIFY.verify_surface_coverage(coverage)


def test_independent_common_parent_coverage_recomputes_collar_gate():
    chart = {
        "profile_points": 501,
        "unique_points": 501,
        "stability_collar_points": 801,
        "unique_stability_collar_points": 801,
        "minimum_termination_margin": 5.0,
        "collar_minimum_termination_margin": 4.5,
        "maximum_inverse_residual": 1e-10,
        "maximum_collar_inverse_residual": 2e-10,
        "inverse_residual_limit": 1e-8,
        "passed": True,
    }
    record = {
        "native_surface_valid": True,
        "charts": {"primary": dict(chart), "fine": dict(chart)},
        "passed": True,
    }
    assert VERIFY.verify_common_coverage(record)
    record["charts"]["fine"]["collar_minimum_termination_margin"] = 3.9
    assert not VERIFY.verify_common_coverage(record)
