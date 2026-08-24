import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("protocol232_tested", ROOT / "protocol232.py")
P = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(P)


def record(time=1.0, area=2.0, a=-3.0, b=4.0, exponent=0.5, passed=True):
    return {
        "passed": passed,
        "critical_time_estimate": time,
        "critical_geometry": {"one_sided_cap_area": area},
        "critical_coefficients": {"transversality_values": [a, a], "quadratic_values": [b, b]},
        "square_root_fit": {"log_exponent": exponent},
    }


def test_exact_temporal_transfer_passes():
    result = P.compare_results(record(), record())
    assert result["transfer_pass"] is True


def test_time_drift_fails():
    result = P.compare_results(record(), record(time=1.0 + P.COARSE_DT))
    assert result["transfer_pass"] is False


def test_coefficient_sign_flip_fails():
    result = P.compare_results(record(), record(a=3.0))
    assert result["transfer_pass"] is False


def test_half_timestep_exact():
    assert P.HALF_DT * 2 == P.COARSE_DT
    assert P.START_STEP * P.HALF_DT == P.START_TIME
    assert P.END_STEP * P.HALF_DT == P.END_TIME
