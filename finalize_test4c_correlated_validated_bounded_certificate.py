#!/usr/bin/env python3
"""Finalize the sealed Test-4C attempt after its decisive unresolved leaf."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.recovery_indexer import atomic_write_json, sha256_file
from bhps.test4c_aggregation import grade_test4c


PROTOCOL = Path("notes/102_test4c_correlated_validated_bounded_certificate_protocol.md")
IMPLEMENTATION_NOTES = tuple(
    Path(f"notes/102{suffix}_test4c_{name}.md")
    for suffix, name in (
        ("a", "step_growth_implementation_correction"),
        ("b", "checkpoint_duration_correction"),
        ("c", "correlated_axis_cone_refinement"),
        ("d", "exact_axis_weighted_ratio_refinement"),
        ("e", "divergence_coordinate_refinement"),
        ("f", "matrix_tail_growth_guard"),
    )
)
MANIFEST = Path("results/test4c_correlated_validated_bounded_recovery_v6.json")
QUALIFICATION = Path(
    "results/test4c_correlated_validated_bounded_stages_v6/"
    "restart_qualification.json"
)
OUTPUT = Path("results/test4c_correlated_validated_bounded_certificate.json")
TEST_MODULES = (
    "tests.test_correlated_validated_shooting",
    "tests.test_validated_capped_surface_shooting",
    "tests.test_test4b_validated_pipeline",
    "tests.test_test4c_correlated_pipeline",
)


def validated_stage(manifest, stage_id):
    stage = manifest["stages"][stage_id]
    path = Path(stage["output_path"])
    if stage["status"] != "complete":
        raise RuntimeError(f"stage is not complete: {stage_id}")
    if path.stat().st_size != stage["byte_count"]:
        raise RuntimeError(f"stage byte count mismatch: {stage_id}")
    if sha256_file(path) != stage["sha256"]:
        raise RuntimeError(f"stage hash mismatch: {stage_id}")
    return path


def run_controls():
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", *TEST_MODULES],
        capture_output=True, text=True, timeout=300,
    )
    return {
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "return_code": completed.returncode,
        "module_count": len(TEST_MODULES),
        "test_count": 25,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main():
    manifest = json.loads(MANIFEST.read_text())
    qualification = json.loads(QUALIFICATION.read_text())
    leaf_stage = "leaf/G9_b217_pbase"
    leaf_path = validated_stage(manifest, leaf_stage)
    leaf = json.loads(leaf_path.read_text())
    controls = run_controls()
    controls_pass = bool(
        controls["status"] == "PASS" and qualification["status"] == "PASS"
    )
    status = grade_test4c(
        controls_pass,
        [leaf], [], 0, 1,
        independently_confirmed_a790_root=False,
    )
    if status != "REVIEW":
        raise RuntimeError(f"unexpected Test-4C grade: {status}")
    payload = {
        "status": status,
        "classification": "correlated_validation_unresolved_before_brane",
        "certificate_pass": False,
        "validated_a790_root_found": False,
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "implementation_notes": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in IMPLEMENTATION_NOTES
        ],
        "controls": controls,
        "restart_qualification": qualification,
        "decisive_leaf": {
            "stage_id": leaf_stage,
            "path": str(leaf_path),
            "sha256": sha256_file(leaf_path),
            "byte_count": leaf_path.stat().st_size,
            "grid": "G9",
            "base_index": 217,
            "launch_interval": leaf["axis_radius"],
            "classification": leaf["classification"],
            "reason": leaf.get("reason"),
            "theta_reached": leaf["theta"],
            "accepted_steps": leaf["accepted_steps"],
            "step_rejections": leaf["step_rejections"],
            "final_attempted_step": leaf["step"],
            "state": leaf.get("state"),
            "audit_summary": leaf["audit_summary"],
        },
        "coverage": {
            "required_A790_base_leaves": {"G9": 256, "G10": 256},
            "required_A794_base_leaves": 256,
            "terminal_v6_leaves_evaluated": 1,
            "unresolved_required_A790_leaves": 1,
            "remaining_cover_not_run": True,
            "stop_reason": (
                "One unresolved required A=7.90 leaf is sufficient to make "
                "PASS impossible under the prospective grading rule."
            ),
        },
        "diagnosis": (
            "The exact spline, axis, matrix-tail, defect, affine-correlation, "
            "and recovery controls pass. The IVP Lohner tube nevertheless "
            "accumulates a symmetric nonlinear remainder transverse to the "
            "regular-axis solution manifold; at theta=0.03793359375 its "
            "contraction cannot close before the sealed minimum step."
        ),
        "why_not_fail": (
            "No validated A=7.90 residual zero was isolated, and no independent "
            "BVP/expansion confirmation of such a zero exists."
        ),
        "next_mathematical_requirement": (
            "A validated collocation or multiple-shooting radii-polynomial BVP "
            "certificate that enforces regularity globally rather than "
            "propagating a finite-axis IVP over-enclosure."
        ),
        "claim_boundary": (
            "REVIEW of the bounded Test-4C certificate attempt. This result "
            "does not negate the corrected positive floating A=7.90 residuals "
            "and is not evidence for a horizon."
        ),
        "provenance": {
            "manifest": str(MANIFEST),
            "manifest_sha256": sha256_file(MANIFEST),
            "bound_sources": manifest["expected_inputs"],
            "finalizer_sha256": sha256_file(Path(__file__)),
            "aggregation_sha256": sha256_file("src/bhps/test4c_aggregation.py"),
            "pipeline_test_sha256": sha256_file(
                "tests/test_test4c_correlated_pipeline.py"
            ),
            "excluded_development_manifests": [
                f"results/test4c_correlated_validated_bounded_recovery_v{version}.json"
                for version in range(1, 6)
            ],
        },
    }
    atomic_write_json(OUTPUT, payload)
    print(json.dumps({
        "status": status,
        "classification": payload["classification"],
        "controls": controls["status"],
        "restart_qualification": qualification["status"],
        "theta_reached": leaf["theta"],
        "accepted_steps": leaf["accepted_steps"],
        "reason": leaf.get("reason"),
    }, indent=2))


if __name__ == "__main__":
    main()
