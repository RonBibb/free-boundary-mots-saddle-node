#!/usr/bin/env python3
"""Independent arithmetic/provenance audit of the Test-14D dense result."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.recovery_indexer import atomic_write_json, sha256_file


DENSE = Path("results/corrected_A790_test14d_dense_collar.json")
MANIFEST = Path("results/corrected_A790_test14d_dense_recovery_v1.json")
SOURCE = Path("src/bhps/test14d_thick_collar.py")
OUTPUT = Path("results/corrected_A790_test14d_independent_audit.json")


def relative_scale_error(value, expected, scale=1.0):
    return float(
        abs(float(value) - float(expected))
        / max(abs(float(value)), abs(float(expected)), abs(float(scale)), 1e-300)
    )


def lagrange_at_zero(widths, values):
    """Evaluate the unique cubic interpolant at zero without polyfit."""
    widths = np.asarray(widths, dtype=float)
    values = np.asarray(values, dtype=float)
    total = 0.0
    for index, width in enumerate(widths):
        weight = 1.0
        for other, other_width in enumerate(widths):
            if other != index:
                weight *= -other_width / (width - other_width)
        total += weight * values[index]
    return float(total)


def independent_window(records, left=0.000625, right=0.004):
    selected = sorted(
        [item for item in records if left - 1e-12 <= item["time"] <= right + 1e-12],
        key=lambda item: item["time"],
    )
    times = np.asarray([item["time"] for item in selected], dtype=float)
    charges = np.asarray([item["charge"] for item in selected], dtype=float)
    names = tuple(selected[0]["finite_corrected_rates"])
    arrays = {
        name: np.asarray([
            item["finite_corrected_rates"][name] for item in selected
        ], dtype=float) for name in names
    }
    total = np.sum(np.stack(list(arrays.values())), axis=0)
    integrated = {
        name: float(np.trapezoid(values, times))
        for name, values in arrays.items()
    }
    delta = float(charges[-1] - charges[0])
    integrated_total = float(np.trapezoid(total, times))
    norm = max(
        abs(delta),
        float(np.trapezoid(
            np.sum(np.abs(np.stack(list(arrays.values()))), axis=0), times,
        )),
        1e-12,
    )
    residual = float(delta - integrated_total)
    return {
        "integrated_named_rates": integrated,
        "integrated_total_rate": integrated_total,
        "closure_residual": residual,
        "balance_norm": norm,
        "normalized_absolute_residual": abs(residual) / norm,
    }


def main():
    dense = json.loads(DENSE.read_text())
    records = dense["records"]
    assessment = dense["assessment"]
    manifest = json.loads(MANIFEST.read_text())

    provenance_checks = {}
    for path, expected in dense["inputs"].items():
        candidate = Path(path)
        provenance_checks[path] = bool(
            candidate.is_file() and sha256_file(candidate) == expected
        )

    manifest_checks = []
    for stage_id, stage in manifest["stages"].items():
        path = Path(stage.get("output_path", ""))
        valid = bool(
            stage.get("status") == "complete"
            and path.is_file()
            and path.stat().st_size == stage.get("byte_count")
            and sha256_file(path) == stage.get("sha256")
        )
        manifest_checks.append(valid)

    axes = {
        "grids": sorted(set(item["grid"] for item in records)),
        "branches": sorted(set(item["branch"] for item in records)),
        "strides": sorted(set(item["stride"] for item in records)),
        "profiles": sorted(set(item["profile"] for item in records)),
        "widths": sorted(set(item["epsilon_over_RA"] for item in records)),
        "rate_paths": sorted(set(item["rate_path"] for item in records)),
        "times": sorted(set(item["time"] for item in records)),
    }
    expected_count = (
        len(axes["grids"]) * len(axes["branches"]) * len(axes["strides"])
        * len(axes["profiles"]) * len(axes["widths"])
        * len(axes["rate_paths"]) * len(axes["times"])
    )

    maximum_sum_error = 0.0
    maximum_thin_formula_error = 0.0
    maximum_junction_identity_error = 0.0
    for item in records:
        total = sum(item["finite_corrected_rates"].values())
        maximum_sum_error = max(maximum_sum_error, relative_scale_error(
            total, item["finite_corrected_total_rate"],
            max(abs(total), 1.0),
        ))
        sphere_area = item["sphere_area"]
        coefficient = item["israel_coefficient"]
        area_radius = item["area_radius"]
        h_sphere = item["collar"]["parameters"]["H_Omega"]
        area_rate = item["collar"]["parameters"]["area_fractional_rate"]
        seam_integral = 8.0 * coefficient * sphere_area
        independent_thin = (
            area_radius * seam_integral * area_rate / 12.0
            + 2.0 * area_radius * sphere_area * coefficient * h_sphere
        )
        maximum_thin_formula_error = max(
            maximum_thin_formula_error,
            relative_scale_error(
                independent_thin, item["thin_seam_rate"],
                max(abs(independent_thin), 1.0),
            ),
        )
        maximum_junction_identity_error = max(
            maximum_junction_identity_error,
            relative_scale_error(
                item["integrated_israel_magnitude"], 6.0 * coefficient,
                max(abs(6.0 * coefficient), 1.0),
            ),
        )

    summary_comparisons = []
    alternative_extrapolated_residuals = []
    for grid in ("G7", "G8"):
        for branch in ("inner", "outer"):
            for stride in (1, 2, 4):
                for profile in ("tanh", "erf", "compact_c2"):
                    for rate_path in ("geometry", "wall"):
                        selected = [
                            item for item in records
                            if item["grid"] == grid and item["branch"] == branch
                            and item["stride"] == stride
                            and item["profile"] == profile
                            and item["rate_path"] == rate_path
                        ]
                        by_time = {}
                        for item in selected:
                            by_time.setdefault(item["time"], []).append(item)
                        extrapolated_history = []
                        for current_time in sorted(by_time):
                            time_records = sorted(
                                by_time[current_time],
                                key=lambda item: item["epsilon_over_RA"],
                            )
                            widths = [item["epsilon_over_RA"] for item in time_records]
                            values = [item["finite_seam_rate"] for item in time_records]
                            extrapolated = lagrange_at_zero(widths, values)
                            template = dict(time_records[0])
                            rates = dict(template["finite_corrected_rates"])
                            rates["finite_collar_seam"] = extrapolated
                            template["finite_corrected_rates"] = rates
                            template["finite_corrected_total_rate"] = sum(rates.values())
                            extrapolated_history.append(template)
                        independent = independent_window(extrapolated_history)
                        stored = assessment["summaries"][grid][branch][str(stride)][
                            profile
                        ][rate_path]["extrapolated"]["primary"]
                        errors = {
                            key: relative_scale_error(
                                independent[key], stored[key],
                                max(abs(stored.get("balance_norm", 1.0)), 1.0),
                            )
                            for key in (
                                "integrated_total_rate", "closure_residual",
                                "balance_norm", "normalized_absolute_residual",
                            )
                        }
                        summary_comparisons.append({
                            "grid": grid,
                            "branch": branch,
                            "stride": stride,
                            "profile": profile,
                            "rate_path": rate_path,
                            "errors": errors,
                        })
                        alternative_extrapolated_residuals.append(
                            independent["normalized_absolute_residual"]
                        )

    independent_israel = []
    for item in records:
        if (
            item["profile"] == "compact_c2"
            and item["epsilon_over_RA"] == max(axes["widths"])
            and item["rate_path"] == "geometry"
        ):
            absolute = abs(item["geometry_israel_rate"] - item["wall_israel_rate"])
            relative = absolute / max(
                abs(item["geometry_israel_rate"]),
                abs(item["wall_israel_rate"]), 0.005,
            )
            independent_israel.append(bool(relative < 0.05 or absolute < 0.005))
    independent_rate_pass_fraction = float(np.mean(independent_israel))

    source_text = SOURCE.read_text()
    collar_start = source_text.index("def collar_record(")
    collar_end = source_text.index("\ndef zero_width_fit", collar_start)
    collar_source = source_text[collar_start:collar_end]
    circular_tokens = [
        token for token in (
            "charge_rate_target", "corrected_total", "closure_residual",
            "test14c", "test14b",
        ) if token in collar_source
    ]

    maximum_summary_error = max(
        value
        for item in summary_comparisons
        for value in item["errors"].values()
    )
    recomputed_grade = (
        "REVIEW"
        if assessment["numerical_balance_subgrade"] == "PASS"
        and independent_rate_pass_fraction < 1.0
        else assessment["overall_grade"]
    )
    gates = {
        "input_provenance": all(provenance_checks.values()),
        "recovery_manifest_complete_and_hashed": (
            len(manifest_checks) == 8065 and all(manifest_checks)
        ),
        "matrix_cartesian_complete": (
            len(records) == expected_count == 8064
        ),
        "record_rate_arithmetic": maximum_sum_error < 2e-12,
        "thin_formula_independent_reconstruction": (
            maximum_thin_formula_error < 2e-10
        ),
        "junction_identity_independent_reconstruction": (
            maximum_junction_identity_error < 2e-6
        ),
        "alternative_zero_width_and_cumulative_reconstruction": (
            # This deliberately compares a cubic four-point interpolant with
            # the primary quadratic least-squares extrapolant.  Agreement is
            # required well below the 1% physical gate, not at roundoff.
            maximum_summary_error < 1e-4
        ),
        "israel_rate_failure_reproduced": (
            abs(
                independent_rate_pass_fraction
                - assessment["summary"]["israel_rate_pass_fraction"]
            ) < 2e-12
            and independent_rate_pass_fraction < 1.0
        ),
        "no_charge_closure_input_to_collar": not circular_tokens,
        "grade_reproduced": recomputed_grade == assessment["overall_grade"],
    }
    result = {
        "schema": "bhps-test14d-independent-audit-v1",
        "status": "complete",
        "dense_result": str(DENSE),
        "dense_result_sha256": sha256_file(DENSE),
        "manifest": str(MANIFEST),
        "manifest_sha256": sha256_file(MANIFEST),
        "input_provenance": provenance_checks,
        "axes": axes,
        "expected_record_count": expected_count,
        "manifest_stage_count": len(manifest_checks),
        "maximum_rate_sum_error": maximum_sum_error,
        "maximum_thin_formula_error": maximum_thin_formula_error,
        "maximum_junction_identity_error": maximum_junction_identity_error,
        "maximum_alternative_summary_error": maximum_summary_error,
        "alternative_extrapolator_agreement_gate": 1e-4,
        "maximum_alternative_extrapolated_primary_residual": max(
            alternative_extrapolated_residuals
        ),
        "independent_israel_rate_pass_fraction": independent_rate_pass_fraction,
        "circular_tokens_in_primary_collar_evaluator": circular_tokens,
        "recomputed_grade": recomputed_grade,
        "gates": gates,
        "passed": bool(all(gates.values())),
        "interpretation": (
            "The numerical finite-collar limit is independently reproduced. "
            "The overall REVIEW grade is also reproduced because the archived "
            "geometry and scalar-wall Israel rates do not agree at every sample."
        ),
    }
    atomic_write_json(OUTPUT, result)
    if not result["passed"]:
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"Test-14D independent audit failed: {failed}")
    print(json.dumps({
        "status": result["status"],
        "passed": result["passed"],
        "recomputed_grade": result["recomputed_grade"],
        "maximum_alternative_summary_error": maximum_summary_error,
        "independent_israel_rate_pass_fraction": independent_rate_pass_fraction,
        "output": str(OUTPUT),
        "output_sha256": sha256_file(OUTPUT),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
