import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "finalize_test2d", ROOT / "finalize_corrected_A790_test2d.py",
)
FINALIZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINALIZE)


def test_final_classification_requires_independent_verifier():
    assert FINALIZE.classify({"verifier_passed": False}) == (
        "REVIEW", "invalid_test2d_numerical_audit",
    )


def test_final_classification_matches_sealed_names():
    for status, classification in (
        ("PASS", "high_order_ragged_chart_above_first_order_continuum_evidence"),
        ("REVIEW", "high_order_ragged_chart_convergence_mixed"),
        ("FAIL", "high_order_ragged_chart_nonconvergence_or_branch_failure"),
    ):
        assert FINALIZE.classify({
            "verifier_passed": True, "status": status, "classification": classification,
        }) == (status, classification)


def test_classification_mismatch_is_invalid_review():
    assert FINALIZE.classify({
        "verifier_passed": True, "status": "PASS", "classification": "wrong",
    }) == ("REVIEW", "invalid_test2d_numerical_audit")
