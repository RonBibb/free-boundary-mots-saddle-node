import importlib.util
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("protocol231_tested", ROOT / "protocol231.py")
P = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(P)


def test_fixed_half_exact_data():
    critical = 1.0
    times = [2.0, 3.0, 5.0, 9.0, 17.0]
    values = [3.0 * math.sqrt(value - critical) for value in times]
    result = P.fixed_time_fit(times, values, critical)
    assert abs(result["free_log_exponent"] - 0.5) < 1e-14
    assert result["fixed_half_maximum_relative_residual"] < 1e-14


def test_sequence_rejects_false_order_for_reversal():
    result = P.sequence_diagnostic([1.0, 1.1, 1.05])
    assert result["monotone"] is False
    assert result["real_richardson_order_defined"] is False


def test_sequence_reports_ordered_data():
    result = P.sequence_diagnostic([1.0, 1.05, 1.075])
    assert result["monotone"] is True
    assert result["real_richardson_order_defined"] is True
