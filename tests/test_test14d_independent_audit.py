import json
from pathlib import Path


def test_independent_audit_reproduces_review_grade():
    path = Path("results/corrected_A790_test14d_independent_audit.json")
    if not path.is_file():
        return
    result = json.loads(path.read_text())
    assert result["passed"]
    assert result["recomputed_grade"] == "REVIEW"
    assert result["maximum_alternative_summary_error"] < 1e-4
    assert result["independent_israel_rate_pass_fraction"] < 1.0
