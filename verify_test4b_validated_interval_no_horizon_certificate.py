#!/usr/bin/env python3
"""Independent structural verifier for the archived Test-4B REVIEW result."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.recovery_indexer import atomic_write_json, sha256_file


RESULT = Path("results/test4b_validated_interval_no_horizon_certificate.json")
OUTPUT = Path("results/test4b_validated_interval_no_horizon_independent_verification.json")


def main():
    result = json.loads(RESULT.read_text())
    manifest_path = Path(result["provenance"]["manifest"])
    manifest = json.loads(manifest_path.read_text())
    checks = {}
    checks["result_is_review_not_certificate"] = bool(
        result["status"] == "REVIEW" and not result["certificate_pass"]
    )
    checks["input_hashes_match"] = bool(all(
        Path(path).is_file() and sha256_file(path) == expected
        for path, expected in result["provenance"]["input_sha256"].items()
    ))
    checks["manifest_matches_result_provenance"] = bool(
        manifest["expected_inputs"] == result["provenance"]["input_sha256"]
        and manifest["protocol_sha256"] == result["protocol_sha256"]
    )
    complete_stages = [
        stage for stage in manifest["stages"].values()
        if stage.get("status") == "complete"
    ]
    checks["completed_stage_hashes_match"] = bool(
        complete_stages and all(
            Path(stage["output_path"]).is_file()
            and Path(stage["output_path"]).stat().st_size == stage["byte_count"]
            and sha256_file(stage["output_path"]) == stage["sha256"]
            for stage in complete_stages
        )
    )
    controls = result["backend_and_representation_controls"]
    checks["backend_and_representation_controls_pass"] = bool(
        controls["all_controls_passed"]
        and controls["libmp_source_uses_directed_rounding"]
        and all(item["contains"] for item in controls["spline_checks"])
        and all(item["contains"] for item in controls["rhs_checks"])
    )
    navigation = result["corrected_axis_navigation"]
    checks["corrected_A790_targeted_minima_positive"] = bool(all(
        navigation["A790"][label]["brane_residual"] > 0.0
        and all(item["brane_residual"] > 0.0
                for item in navigation["A790"][label]["theta_cut_variants"])
        for label in ("G9", "G10")
    ))
    roots = navigation["A794_G7"]["corrected_roots"]
    checks["A794_adverse_brackets_have_opposite_signs"] = bool(
        len(roots) == 2 and all(
            item["bracket_residuals"][0] * item["bracket_residuals"][1] < 0.0
            for item in roots
        )
    )
    interval_attempt = result["validated_interval_attempt"]
    leaves = [
        item for values in interval_attempt["probes"].values() for item in values
    ]
    checks["all_reported_interval_probes_are_unresolved"] = bool(
        len(leaves) == interval_attempt["unresolved_probe_count"] == 8
        and all(item["classification"].startswith("unresolved") for item in leaves)
    )
    checks["point_launch_failure_is_explicit"] = bool(any(
        "point" in item["probe_name"] and item["classification"] == "unresolved_step"
        for item in leaves
    ))
    checks["scope_does_not_expand_topology_or_continuum"] = bool(
        "No conclusion about arbitrary topologies" in result["claim_boundary"]
        and "continuum spacetime" in result["claim_boundary"]
    )
    passed = bool(all(checks.values()))
    payload = {
        "status": "PASS" if passed else "FAIL",
        "classification": "independent_Test4B_archive_and_claim_verification",
        "checks": checks,
        "verified_result": str(RESULT),
        "verified_result_sha256": sha256_file(RESULT),
        "verified_manifest": str(manifest_path),
        "verified_manifest_sha256": sha256_file(manifest_path),
    }
    atomic_write_json(OUTPUT, payload)
    print(json.dumps(payload, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
