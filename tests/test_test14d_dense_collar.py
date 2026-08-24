import json
from pathlib import Path


def test_dense_result_has_separate_numerical_and_physical_grades():
    path = Path("results/corrected_A790_test14d_dense_collar.json")
    if not path.is_file():
        return
    result = json.loads(path.read_text())
    assessment = result["assessment"]
    assert assessment["gates"]["complete_matrix"]
    assert assessment["gates"]["finite"]
    assert assessment["numerical_balance_subgrade"] in ("PASS", "FAIL")
    assert assessment["physical_israel_rate_subgrade"] in ("PASS", "REVIEW")
    assert assessment["overall_grade"] in ("PASS", "REVIEW", "FAIL")
