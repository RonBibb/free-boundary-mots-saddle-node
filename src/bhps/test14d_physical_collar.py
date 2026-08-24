"""Archived-tube finite-collar pilot for Test 14D.

This consumes the already audited Test-14B/14C records.  It does not read or
modify the spacetime evolution archive and does not claim a thick-brane
evolution.
"""

from __future__ import annotations

import math

import numpy as np

from bhps.test14d_thick_collar import (
    FAMILIES,
    RESOLUTIONS,
    WIDTH_RATIOS,
    collar_record,
    relative_scale_error,
    zero_width_fit,
)


RATE_PATHS = ("geometry", "wall")


def physical_collar_record(
    test14c_record, test14b_record, family, width_ratio,
    nodes_per_epsilon, rate_path,
):
    """Evaluate one regularized collar from an archived thin-tube record."""
    if rate_path not in RATE_PATHS:
        raise ValueError(f"unknown Test-14D rate path {rate_path!r}")
    charge = test14b_record["charge"]
    seam = test14b_record["seam"]
    area_radius = float(charge["equivalent_area_radius"])
    sphere_area = float(seam["sphere_area_A2"])
    seam_warp = math.sqrt(sphere_area / (4.0 * math.pi))
    coefficient = float(seam["geometric_Ws_over_W"])
    geometry_rate = float(test14c_record["geometric_israel_rate"])
    wall_rate = float(test14c_record["wall_israel_rate"])
    coefficient_rate = geometry_rate if rate_path == "geometry" else wall_rate
    epsilon = float(width_ratio) * area_radius
    collar = collar_record(
        family, epsilon, nodes_per_epsilon,
        c=coefficient,
        c_rate=coefficient_rate,
        seam_warp=seam_warp,
        h_sphere=float(test14c_record["H_Omega"]),
        h_meridional=float(test14c_record["H_meridional_normal"]),
        area_radius=area_radius,
        area_fractional_rate=float(test14c_record["area_fractional_rate"]),
    )

    thin_seam = float(test14c_record["coupled_seam"]["total"])
    rates = {key: float(value) for key, value in test14c_record[
        "corrected_rates"
    ].items()}
    balance_norm_rate = max(
        abs(float(test14c_record["charge_rate_target"])),
        float(sum(abs(value) for value in rates.values())),
        1e-12,
    )
    finite_total = float(
        test14c_record["corrected_total_rate"]
        - thin_seam + collar["finite_seam_rate"]
    )
    target_rate = float(test14c_record["charge_rate_target"])
    residual = float(target_rate - finite_total)
    wall_energy = float(seam["wall_energy_bare_plus_potential"])
    wall_magnitude_error = relative_scale_error(
        collar["junction_integral"], wall_energy,
        max(abs(wall_energy), 1.0),
    )
    stored_formula_error = relative_scale_error(
        collar["thin_seam_rate_target"], thin_seam,
        max(abs(thin_seam), 0.01 * balance_norm_rate, 1e-12),
    )
    return {
        "grid": str(test14c_record["grid"]),
        "branch": str(test14c_record["branch"]),
        "time": float(test14c_record["time"]),
        "stride": int(test14c_record["stride"]),
        "profile": str(family),
        "epsilon_over_RA": float(width_ratio),
        "epsilon": epsilon,
        "nodes_per_epsilon": int(nodes_per_epsilon),
        "rate_path": str(rate_path),
        "area_radius": area_radius,
        "sphere_area": sphere_area,
        "israel_coefficient": coefficient,
        "geometry_israel_rate": geometry_rate,
        "wall_israel_rate": wall_rate,
        "israel_rate_absolute_difference": abs(geometry_rate - wall_rate),
        "israel_rate_relative_scale_error": relative_scale_error(
            geometry_rate, wall_rate, 0.005,
        ),
        "wall_energy": wall_energy,
        "integrated_israel_magnitude": float(collar["junction_integral"]),
        "integrated_israel_magnitude_error": wall_magnitude_error,
        "thin_seam_rate": thin_seam,
        "recomputed_thin_seam_rate": float(collar["thin_seam_rate_target"]),
        "stored_thin_formula_error": stored_formula_error,
        "finite_seam_rate": float(collar["finite_seam_rate"]),
        "finite_seam_absolute_difference": abs(
            float(collar["finite_seam_rate"]) - thin_seam
        ),
        "finite_seam_relative_scale_error": relative_scale_error(
            collar["finite_seam_rate"], thin_seam,
            max(abs(thin_seam), 0.01 * balance_norm_rate, 1e-12),
        ),
        "finite_seam_balance_normalized_difference": float(
            abs(float(collar["finite_seam_rate"]) - thin_seam)
            / balance_norm_rate
        ),
        "charge_rate_target": target_rate,
        "thin_corrected_total_rate": float(
            test14c_record["corrected_total_rate"]
        ),
        "finite_corrected_total_rate": finite_total,
        "finite_pointwise_residual": residual,
        "pointwise_balance_norm_rate": balance_norm_rate,
        "finite_pointwise_normalized_residual": abs(residual) / balance_norm_rate,
        "collar": collar,
        "finite": bool(
            test14c_record["finite"] and test14b_record["finite"]
            and collar["finite"]
            and np.all(np.isfinite([
                finite_total, target_rate, residual, wall_magnitude_error,
            ]))
        ),
    }


def _admitted_difference(value, target, norm, relative_gate, norm_gate):
    difference = abs(float(value) - float(target))
    relative = difference / max(abs(float(value)), abs(float(target)), 1e-300)
    normalized = difference / max(float(norm), 1e-300)
    return {
        "absolute_difference": difference,
        "relative_error": relative,
        "balance_normalized_difference": normalized,
        "passed": bool(relative < relative_gate or normalized < norm_gate),
    }


def physical_pilot_assessment(records):
    """Grade the sealed four-tube physical pilot without cumulative claims."""
    records = list(records)
    expected_count = (
        2 * 2 * len(FAMILIES) * len(WIDTH_RATIOS)
        * len(RESOLUTIONS) * len(RATE_PATHS)
    )
    if len(records) != expected_count:
        raise ValueError(
            f"pilot has {len(records)} records, expected {expected_count}"
        )
    fits = {}
    tube_summaries = {}
    for grid in ("G7", "G8"):
        fits[grid] = {}
        tube_summaries[grid] = {}
        for branch in ("inner", "outer"):
            fits[grid][branch] = {}
            tube_records = [
                item for item in records
                if item["grid"] == grid and item["branch"] == branch
            ]
            reference = tube_records[0]
            norm = float(reference["pointwise_balance_norm_rate"])
            tube_summaries[grid][branch] = {
                "geometry_israel_rate": reference["geometry_israel_rate"],
                "wall_israel_rate": reference["wall_israel_rate"],
                "israel_rate_absolute_difference": reference[
                    "israel_rate_absolute_difference"
                ],
                "israel_rate_relative_scale_error": reference[
                    "israel_rate_relative_scale_error"
                ],
                "israel_rate_gate_passed": bool(
                    reference["israel_rate_relative_scale_error"] < 0.05
                    or reference["israel_rate_absolute_difference"] < 0.005
                ),
                "balance_norm_rate": norm,
                "thin_pointwise_normalized_residual": abs(
                    reference["charge_rate_target"]
                    - reference["thin_corrected_total_rate"]
                ) / norm,
            }
            for family in FAMILIES:
                fits[grid][branch][family] = {}
                for path in RATE_PATHS:
                    selected = [
                        item for item in tube_records
                        if item["profile"] == family
                        and item["rate_path"] == path
                        and item["nodes_per_epsilon"] == 128
                    ]
                    fit = zero_width_fit(
                        selected, "finite_seam_rate",
                        reference["thin_seam_rate"],
                        max(abs(reference["thin_seam_rate"]), 0.01 * norm),
                    )
                    finest = min(selected, key=lambda item: item[
                        "epsilon_over_RA"
                    ])
                    fit["finest_admission"] = _admitted_difference(
                        finest["finite_seam_rate"],
                        reference["thin_seam_rate"], norm, 0.02, 0.01,
                    )
                    fit["extrapolated_admission"] = _admitted_difference(
                        fit["extrapolated"], reference["thin_seam_rate"],
                        norm, 0.01, 0.01,
                    )
                    fit["finest_pointwise_normalized_residual"] = abs(
                        reference["charge_rate_target"]
                        - (
                            reference["thin_corrected_total_rate"]
                            - reference["thin_seam_rate"]
                            + finest["finite_seam_rate"]
                        )
                    ) / norm
                    fit["extrapolated_pointwise_normalized_residual"] = abs(
                        reference["charge_rate_target"]
                        - (
                            reference["thin_corrected_total_rate"]
                            - reference["thin_seam_rate"]
                            + fit["extrapolated"]
                        )
                    ) / norm
                    fits[grid][branch][family][path] = fit

            profile_checks = []
            for path in RATE_PATHS:
                values = [
                    fits[grid][branch][family][path]["extrapolated"]
                    for family in FAMILIES
                ]
                spread = max(values) - min(values)
                profile_checks.append({
                    "rate_path": path,
                    "absolute_spread": float(spread),
                    "balance_normalized_spread": float(spread / norm),
                    "relative_spread": float(
                        spread / max(max(abs(value) for value in values), 1e-300)
                    ),
                    "passed": bool(
                        spread / max(max(abs(value) for value in values), 1e-300)
                        < 0.02 or spread / norm < 0.01
                    ),
                })
            path_checks = []
            for family in FAMILIES:
                geometry = fits[grid][branch][family]["geometry"]["extrapolated"]
                wall = fits[grid][branch][family]["wall"]["extrapolated"]
                path_checks.append({
                    "profile": family,
                    **_admitted_difference(geometry, wall, norm, 0.02, 0.01),
                })
            tube_summaries[grid][branch]["profile_checks"] = profile_checks
            tube_summaries[grid][branch]["rate_path_checks"] = path_checks

    maximum_israel_magnitude_error = max(
        item["integrated_israel_magnitude_error"] for item in records
        if item["nodes_per_epsilon"] == 128
    )
    maximum_stored_formula_error = max(
        item["stored_thin_formula_error"] for item in records
    )
    all_finest = all(
        fits[grid][branch][family][path]["finest_admission"]["passed"]
        for grid in ("G7", "G8") for branch in ("inner", "outer")
        for family in FAMILIES for path in RATE_PATHS
    )
    all_extrapolated = all(
        fits[grid][branch][family][path]["extrapolated_admission"]["passed"]
        for grid in ("G7", "G8") for branch in ("inner", "outer")
        for family in FAMILIES for path in RATE_PATHS
    )
    all_profiles = all(
        item["passed"]
        for grid in ("G7", "G8") for branch in ("inner", "outer")
        for item in tube_summaries[grid][branch]["profile_checks"]
    )
    all_paths = all(
        item["passed"]
        for grid in ("G7", "G8") for branch in ("inner", "outer")
        for item in tube_summaries[grid][branch]["rate_path_checks"]
    )
    all_israel_rates = all(
        tube_summaries[grid][branch]["israel_rate_gate_passed"]
        for grid in ("G7", "G8") for branch in ("inner", "outer")
    )
    gates = {
        "complete_matrix": len(records) == expected_count,
        "finite": all(item["finite"] for item in records),
        "stored_thin_formula": maximum_stored_formula_error < 2e-10,
        "integrated_israel_magnitude": maximum_israel_magnitude_error < 0.01,
        "israel_rate_compatibility": all_israel_rates,
        "finest_width_to_thin": all_finest,
        "extrapolated_to_thin": all_extrapolated,
        "profile_universality": all_profiles,
        "rate_path_universality": all_paths,
    }
    numerical_gates = {
        key: value for key, value in gates.items()
        if key != "israel_rate_compatibility"
    }
    return {
        "schema": "bhps-test14d-physical-pilot-assessment-v1",
        "record_count": len(records),
        "expected_record_count": expected_count,
        "fits": fits,
        "tube_summaries": tube_summaries,
        "summary": {
            "maximum_israel_magnitude_error": maximum_israel_magnitude_error,
            "maximum_stored_thin_formula_error": maximum_stored_formula_error,
            "maximum_finest_pointwise_normalized_residual": max(
                fits[grid][branch][family][path][
                    "finest_pointwise_normalized_residual"
                ]
                for grid in ("G7", "G8") for branch in ("inner", "outer")
                for family in FAMILIES for path in RATE_PATHS
            ),
            "maximum_extrapolated_pointwise_normalized_residual": max(
                fits[grid][branch][family][path][
                    "extrapolated_pointwise_normalized_residual"
                ]
                for grid in ("G7", "G8") for branch in ("inner", "outer")
                for family in FAMILIES for path in RATE_PATHS
            ),
        },
        "gates": gates,
        "numerical_collar_subgrade": (
            "PASS" if all(numerical_gates.values()) else "FAIL"
        ),
        "physical_israel_rate_subgrade": (
            "PASS" if gates["israel_rate_compatibility"] else "REVIEW"
        ),
        "pilot_grade": (
            "PASS" if all(gates.values())
            else "REVIEW" if all(numerical_gates.values())
            else "FAIL"
        ),
        "finite": bool(all(item["finite"] for item in records)),
    }
