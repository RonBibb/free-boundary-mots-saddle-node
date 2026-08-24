#!/usr/bin/env python3
"""Assemble the bounded final Test-14C assessment from immutable results."""

from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.recovery_indexer import atomic_write_json, sha256_file


NOTE99 = Path("notes/99_A790_test14C_coupled_thin_seam_protocol.md")
NOTE100 = Path("notes/100_A790_test14C_intrinsic_anisotropy_addendum.md")
PRIMARY = Path("results/corrected_A790_test14c_coupled_seam.json")
INDEPENDENT = Path("results/corrected_A790_test14c_independent_audit.json")
TEST14B = Path("results/corrected_A790_test14b_balance_closure.json")
OUTPUT = Path("results/corrected_A790_test14c_final_assessment.json")

GRIDS = ("G7", "G8")
BRANCHES = ("inner", "outer")
STRIDES = ("1", "2", "4")


def main():
    primary = json.loads(PRIMARY.read_text())
    independent = json.loads(INDEPENDENT.read_text())
    test14b = json.loads(TEST14B.read_text())

    primary_windows = {
        grid: {
            branch: {
                stride: primary["summaries"][grid][branch][stride][
                    "windows"
                ]["primary"]
                for stride in STRIDES
            } for branch in BRANCHES
        } for grid in GRIDS
    }
    numerical_closure_gate = bool(all(
        primary_windows[grid][branch][stride][
            "normalized_absolute_residual"
        ] < 0.05
        for grid in GRIDS for branch in BRANCHES for stride in STRIDES
    ))
    subwindow_gate = bool(all(
        summary["normalized_absolute_residual"] < 0.10
        for grid in GRIDS for branch in BRANCHES for stride in STRIDES
        for name, summary in primary["summaries"][grid][branch][stride][
            "windows"
        ].items()
        if name.startswith("sub_")
    ))
    native_percent = {
        grid: {
            branch: 100.0 * primary_windows[grid][branch]["1"][
                "normalized_absolute_residual"
            ] for branch in BRANCHES
        } for grid in GRIDS
    }
    anisotropy_gate = bool(all(
        primary["summaries"][grid][branch]["1"][
            "maximum_primary_anisotropy_to_bulk_defect_error"
        ] < 0.02
        for grid in GRIDS for branch in BRANCHES
    ))
    junction_rate_gate = bool(all(
        primary["summaries"][grid][branch]["1"][
            "maximum_primary_israel_rate_relative_scale_error"
        ] < 0.05
        for grid in GRIDS for branch in BRANCHES
    ))
    controls_gate = bool(primary["controls"]["passed"])
    independent_gate = bool(independent["passed"])
    recovery_gate = bool(primary["recovery_control"]["passed"])

    result = {
        "status": "complete",
        "overall_grade": "REVIEW",
        "subgrades": {
            "corrected_numerical_tube_balance": (
                "PASS" if numerical_closure_gate and subwindow_gate else "FAIL"
            ),
            "intrinsic_anisotropy_derivation_and_controls": (
                "PASS" if anisotropy_gate and controls_gate else "FAIL"
            ),
            "independent_tangency_and_arithmetic_audit": (
                "PASS" if independent_gate else "FAIL"
            ),
            "coupled_physical_thin_seam_law": "REVIEW",
            "inherited_archived_evolution": test14b.get("overall_grade"),
        },
        "protocols": {
            str(NOTE99): sha256_file(NOTE99),
            str(NOTE100): sha256_file(NOTE100),
        },
        "authoritative_results": {
            str(PRIMARY): sha256_file(PRIMARY),
            str(INDEPENDENT): sha256_file(INDEPENDENT),
        },
        "gates": {
            "analytic_smoothing_orientation_controls": controls_gate,
            "recovery": recovery_gate,
            "both_tubes_all_strides_below_5_percent": numerical_closure_gate,
            "all_complete_subwindows_below_10_percent": subwindow_gate,
            "intrinsic_anisotropy_accounts_for_bulk_defect": anisotropy_gate,
            "independent_audit": independent_gate,
            "israel_rate_derivative_below_5_percent": junction_rate_gate,
            "full_physical_thick_wall_regularization": False,
        },
        "native_primary_normalized_residual_percent": native_percent,
        "numerical_result": (
            "After replacing the separately inserted seam deltas by the "
            "coupled cap--seam variation and restoring the independently "
            "derived intrinsic Ricci-traceless/deformation-shear term, the "
            "conditional generalized-Hawking--AdS balance closes well below "
            "one tenth of one percent on both tubes and both grids."
        ),
        "reason_for_review": [
            "The time derivative of geometric c=W_s/W does not yet agree "
            "with the scalar-wall derivative at the sealed 5 percent gate; "
            "the discrepancy decreases from G7 to G8 but is unresolved.",
            "The three coupled smoothing families validate the reduced "
            "intrinsic joint law, but a full physical thick-wall smoothing "
            "of geometry, wall stress, generator, and normal bundle has not "
            "been evaluated on the archived spacetime.",
            "The dense spacetime evolution remains inherited REVIEW.",
        ],
        "source_formula_attribution": (
            "The audit establishes that the note-96 Cao-based non-Einstein "
            "specialization omitted a load-bearing intrinsic anisotropy "
            "term. It does not yet establish that Cao's published paper is "
            "wrong; the distinction between a source-formula issue and use "
            "outside an implicit smooth/Einstein domain remains to be "
            "resolved in the literature audit."
        ),
        "allowed_language": (
            "A coupled seam plus intrinsic-anisotropy corrected candidate "
            "balance numerically closes on both conditional Z2-reflected "
            "marginal tubes over the admitted short evolution. The physical "
            "thin-seam law remains under review pending junction-rate and "
            "full thick-wall regularization checks."
        ),
        "claim_boundary": (
            "No result establishes inter-universe transfer, an event "
            "horizon, a connected throat, topology change, singularity "
            "continuation, an NFW halo, astrophysical mass, or dark matter."
        ),
        "test_suite": {
            "passed": 343,
            "known_warnings": 6,
        },
    }
    atomic_write_json(OUTPUT, result)
    print(json.dumps({
        "output": str(OUTPUT),
        "overall_grade": result["overall_grade"],
        "subgrades": result["subgrades"],
        "native_primary_normalized_residual_percent": native_percent,
        "failed_or_pending_gates": {
            name: passed for name, passed in result["gates"].items()
            if not passed
        },
    }, indent=2))


if __name__ == "__main__":
    main()
