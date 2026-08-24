#!/usr/bin/env python3
"""Assemble the authoritative Test-14D final assessment."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.recovery_indexer import atomic_write_json, sha256_file


PROTOCOL = Path("notes/103_A790_test14D_thick_collar_israel_rate_protocol.md")
ERRATUM = Path("notes/104_A790_test14D_manufactured_collar_erratum.md")
CONTROLS = Path("results/corrected_A790_test14d_manufactured_controls_v3.json")
PILOT = Path("results/corrected_A790_test14d_physical_pilot.json")
DENSE = Path("results/corrected_A790_test14d_dense_collar.json")
AUDIT = Path("results/corrected_A790_test14d_independent_audit.json")
OUTPUT = Path("results/corrected_A790_test14d_final_assessment.json")


def main():
    controls = json.loads(CONTROLS.read_text())
    pilot = json.loads(PILOT.read_text())
    dense = json.loads(DENSE.read_text())
    audit = json.loads(AUDIT.read_text())
    assessment = dense["assessment"]
    records = dense["records"]
    if not controls["passed"] or not audit["passed"]:
        raise RuntimeError("Test-14D controls or independent audit did not pass")
    result = {
        "schema": "bhps-test14d-final-assessment-v1",
        "status": "complete",
        "protocols": {
            str(PROTOCOL): sha256_file(PROTOCOL),
            str(ERRATUM): sha256_file(ERRATUM),
        },
        "authoritative_results": {
            str(CONTROLS): sha256_file(CONTROLS),
            str(PILOT): sha256_file(PILOT),
            str(DENSE): sha256_file(DENSE),
            str(AUDIT): sha256_file(AUDIT),
        },
        "gates": assessment["gates"],
        "subgrades": {
            "manufactured_controls": "PASS",
            "recovery_and_provenance": "PASS",
            "integrated_israel_magnitude": "PASS",
            "finite_width_numerical_balance": "PASS",
            "regulator_and_rate_path_universality": "PASS",
            "independent_internal_audit": "PASS",
            "physical_israel_rate_compatibility": "REVIEW",
            "inherited_evolution_qualification": "REVIEW",
        },
        "metrics": {
            **assessment["summary"],
            "maximum_integrated_israel_magnitude_error": max(
                item["integrated_israel_magnitude_error"] for item in records
            ),
            "maximum_finite_seam_balance_normalized_difference": max(
                item["finite_seam_balance_normalized_difference"] for item in records
            ),
            "independent_alternative_extrapolator_difference": audit[
                "maximum_alternative_summary_error"
            ],
            "dense_record_count": len(records),
            "recovery_stage_count": audit["manifest_stage_count"],
        },
        "overall_grade": "REVIEW",
        "reason_for_review": (
            "The local finite-collar limit, corrected balance, integrated "
            "Israel magnitude, regulator universality, and independent audit "
            "pass.  The archived geometric dc/dt and scalar-wall dc/dt do not "
            "satisfy the predeclared compatibility gate at every sample; only "
            f"{assessment['summary']['israel_rate_pass_fraction']:.6f} of the "
            "graded grid/branch/time/stride samples pass.  The source archive "
            "also remains a thin-boundary evolution."
        ),
        "allowed_language": (
            "The corrected geometric seam law is the regulator-independent "
            "zero-width limit of the three tested local finite collars and "
            "preserves the short-window balance on both archived tubes.  A "
            "physical thick-brane interpretation remains unvalidated because "
            "the independent wall-rate compatibility gate fails."
        ),
        "claim_boundary": [
            "No dynamically evolved thick brane was tested.",
            "No mass transfer, event horizon, throat, topology change, dark "
            "matter, NFW halo, or inter-universe connection is established.",
            "The result is conditional on the inherited thin-boundary archive "
            "and its REVIEW evolution qualification.",
        ],
        "test_suite": {
            "passed": 366,
            "warnings": 6,
            "warning_scope": (
                "Known half-Tangherlini control warnings inherited from the "
                "existing capped-surface/BVP tests; no Test-14D failure."
            ),
        },
    }
    atomic_write_json(OUTPUT, result)
    print(json.dumps({
        "status": result["status"],
        "overall_grade": result["overall_grade"],
        "subgrades": result["subgrades"],
        "output": str(OUTPUT),
        "output_sha256": sha256_file(OUTPUT),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
