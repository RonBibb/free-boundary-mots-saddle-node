#!/usr/bin/env python3
"""Independent consistency verifier for the Test-4C REVIEW artifact."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.recovery_indexer import atomic_write_json, sha256_file
from bhps.test4c_aggregation import grade_test4c


RESULT = Path("results/test4c_correlated_validated_bounded_certificate.json")
OUTPUT = Path(
    "results/test4c_correlated_validated_bounded_independent_verification.json"
)
TEST_MODULES = (
    "tests.test_correlated_validated_shooting",
    "tests.test_validated_capped_surface_shooting",
    "tests.test_test4b_validated_pipeline",
    "tests.test_test4c_correlated_pipeline",
)


def main():
    result = json.loads(RESULT.read_text())
    manifest_path = Path(result["provenance"]["manifest"])
    manifest = json.loads(manifest_path.read_text())
    leaf_record = result["decisive_leaf"]
    leaf_path = Path(leaf_record["path"])
    leaf = json.loads(leaf_path.read_text())
    qualification = result["restart_qualification"]
    rerun = subprocess.run(
        [sys.executable, "-m", "unittest", *TEST_MODULES],
        capture_output=True, text=True, timeout=300,
    )

    checks = {
        "result_is_review_not_pass_or_fail": (
            result["status"] == "REVIEW"
            and result["certificate_pass"] is False
            and result["validated_a790_root_found"] is False
        ),
        "protocol_hash_matches": (
            sha256_file(result["protocol"]) == result["protocol_sha256"]
        ),
        "implementation_note_hashes_match": all(
            sha256_file(item["path"]) == item["sha256"]
            for item in result["implementation_notes"]
        ),
        "manifest_hash_matches": (
            sha256_file(manifest_path) == result["provenance"]["manifest_sha256"]
        ),
        "all_manifest_inputs_still_match": all(
            Path(path).is_file() and sha256_file(path) == expected
            for path, expected in manifest["expected_inputs"].items()
        ),
        "leaf_hash_and_size_match": (
            leaf_path.stat().st_size == leaf_record["byte_count"]
            and sha256_file(leaf_path) == leaf_record["sha256"]
        ),
        "leaf_is_required_a790_unresolved": (
            leaf_record["grid"] == "G9"
            and leaf["classification"] == "unresolved_correlated_step"
            and leaf["theta"] < 3.141592653589793 / 2
            and "terminal_residual" not in leaf
        ),
        "minimum_step_was_exhausted": (
            leaf["step"] < leaf["numerical_policy"]["minimum_validation_step"]
            and "contraction" in leaf["reason"].lower()
        ),
        "matrix_tail_never_exceeded_acceptance_threshold": (
            leaf["audit_summary"]["maximum_matrix_tail"]
            <= leaf["numerical_policy"]["matrix_tail_limit"]
        ),
        "restart_qualification_passed": (
            qualification["status"] == "PASS"
            and qualification["running_reset_to_pending"]
            and qualification["corrupted_completed_leaf_rejected"]
            and qualification["bound_and_precision_provenance_change_rejected"]
        ),
        "focused_controls_rerun_passed": rerun.returncode == 0,
        "independent_grade_is_review": grade_test4c(
            True, [leaf], [], 0, 1, False,
        ) == "REVIEW",
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "status": status,
        "result": str(RESULT),
        "result_sha256": sha256_file(RESULT),
        "checks": checks,
        "focused_control_rerun": {
            "return_code": rerun.returncode,
            "stdout": rerun.stdout,
            "stderr": rerun.stderr,
        },
        "verified_conclusion": (
            "Test 4C is REVIEW because a required A=7.90 launch leaf remained "
            "unresolved before the brane. It is neither a no-root certificate "
            "nor evidence of a root."
        ),
    }
    atomic_write_json(OUTPUT, payload)
    print(json.dumps({"status": status, "checks": checks}, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
