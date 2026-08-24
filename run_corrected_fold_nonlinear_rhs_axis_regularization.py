#!/usr/bin/env python3
"""Regularize nonlinear SO(3) acceleration quotients on their native grids."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.linearized_gh_einstein_scalar import (
    metric_geometry_from_jets,
    solve_reduced_einstein_two_scalar_acceleration,
)
from run_corrected_fold_boundary_constraint_pulse import build_geometry


FIELD_ORDER = (
    "g_z0_tt", "v_z_tt", "g_00_tt", "g_perp_tt", "d_tt",
    "v_0_tt", "g_zz_tt", "Phi_tt", "chi_tt",
)
QUOTIENT_POWERS = np.array((0, 1, 0, 0, 2, 1, 0, 0, 0))
Z_FRACTIONS = (0.05, 0.25, 0.5, 0.75, 0.95)
MATCHED_R = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0)
AXIS_FIT_COUNTS = (6, 8)
OUTPUT = Path("results/corrected_fold_nonlinear_rhs_axis_regularization.json")


def relative_norm(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(left), np.linalg.norm(right), 1e-300))


def solve_raw(geometry, z_value, r_value):
    background = geometry["jet_field"].at(z_value, r_value)
    full_geometry = metric_geometry_from_jets(
        background["metric"], background["metric_first"], background["metric_second"],
    )
    solved = solve_reduced_einstein_two_scalar_acceleration(
        background["metric"], background["metric_first"], background["metric_second"],
        background["phi"], background["phi_first"], background["phi_second"],
        background["chi"], background["chi_first"], background["chi_second"],
        full_geometry["contracted_christoffel_covector"],
        full_geometry["contracted_christoffel_covector_first"],
        mass_squared=geometry["mass_squared"], potential_offset=-6.0,
        kappa5_squared=1.0,
    )
    acceleration = np.asarray(solved["metric_acceleration"])
    transverse = 0.5 * (acceleration[3, 3] + acceleration[4, 4])
    return np.array((
        acceleration[1, 0], acceleration[1, 2], acceleration[0, 0],
        transverse, acceleration[2, 2] - transverse, acceleration[0, 2],
        acceleration[1, 1], solved["phi_acceleration"], solved["chi_acceleration"],
    ))


def axis_limit(positive_r, regular_positive, fit_count):
    count = int(fit_count)
    squared = positive_r[:count] ** 2
    result = np.empty(regular_positive.shape[1])
    for field in range(regular_positive.shape[1]):
        result[field] = np.polynomial.polynomial.polyfit(
            squared, regular_positive[:count, field], 3,
        )[0]
    return result


def assemble_radial_line(geometry, z_value):
    positive_r = np.asarray(geometry["r"])[1:]
    positive_r = positive_r[positive_r <= 4.5]
    raw = np.asarray([solve_raw(geometry, z_value, r_value) for r_value in positive_r])
    regular_positive = raw / positive_r[:, None] ** QUOTIENT_POWERS[None, :]
    limits = {
        str(count): axis_limit(positive_r, regular_positive, count)
        for count in AXIS_FIT_COUNTS
    }
    selected_limit = limits[str(AXIS_FIT_COUNTS[-1])]
    radial_grid = np.r_[0.0, positive_r]
    regular_grid = np.vstack((selected_limit, regular_positive))
    matched = np.empty((len(MATCHED_R), len(FIELD_ORDER)))
    for field in range(len(FIELD_ORDER)):
        matched[:, field] = CubicSpline(radial_grid, regular_grid[:, field])(
            np.asarray(MATCHED_R)
        )
    return {
        "z": float(z_value),
        "native_positive_radial_points": int(len(positive_r)),
        "axis_limits": limits,
        "axis_fit_relative_difference": relative_norm(
            limits[str(AXIS_FIT_COUNTS[0])], limits[str(AXIS_FIT_COUNTS[1])],
        ),
        "matched_regular_acceleration": matched,
        "finite": bool(np.all(np.isfinite(regular_grid)) and np.all(np.isfinite(matched))),
    }


def build_case(resolution):
    geometry = build_geometry(resolution)
    z0 = float(geometry["z"][0])
    z1 = float(geometry["z"][-1])
    lines = [
        assemble_radial_line(geometry, z0 + fraction * (z1 - z0))
        for fraction in Z_FRACTIONS
    ]
    return {
        "resolution": resolution,
        "source_grid": geometry["source_grid"],
        "selector_maximum": geometry["selector_maximum"],
        "lines": lines,
    }


def scaled_field_differences(left, right):
    left = np.asarray(left)
    right = np.asarray(right)
    total_scale = max(np.linalg.norm(left), np.linalg.norm(right), 1e-300)
    result = {}
    for field, name in enumerate(FIELD_ORDER):
        numerator = np.linalg.norm(left[..., field] - right[..., field])
        denominator = max(
            np.linalg.norm(left[..., field]), np.linalg.norm(right[..., field]),
            1e-8 * total_scale,
        )
        result[name] = float(numerator / denominator)
    return result


def main():
    cases = [build_case(resolution) for resolution in ("G5", "G6")]
    coarse, fine = cases
    coarse_matched = np.asarray([line["matched_regular_acceleration"] for line in coarse["lines"]])
    fine_matched = np.asarray([line["matched_regular_acceleration"] for line in fine["lines"]])
    coarse_axis = np.asarray([
        line["axis_limits"][str(AXIS_FIT_COUNTS[-1])] for line in coarse["lines"]
    ])
    fine_axis = np.asarray([
        line["axis_limits"][str(AXIS_FIT_COUNTS[-1])] for line in fine["lines"]
    ])
    matched_transfer = relative_norm(coarse_matched, fine_matched)
    axis_transfer = relative_norm(coarse_axis, fine_axis)
    matched_fields = scaled_field_differences(coarse_matched, fine_matched)
    axis_fields = scaled_field_differences(coarse_axis, fine_axis)
    maximum_axis_fit = max(
        line["axis_fit_relative_difference"] for case in cases for line in case["lines"]
    )
    acceptance = {
        "all_values_finite": bool(all(line["finite"] for case in cases for line in case["lines"])),
        "selectors_below_1e_8": bool(all(case["selector_maximum"] < 1e-8 for case in cases)),
        "native_regularized_matched_transfer_below_5_percent": bool(matched_transfer < 0.05),
        "native_axis_fit_robustness_below_5_percent": bool(maximum_axis_fit < 0.05),
        "native_axis_G5_G6_transfer_below_5_percent": bool(axis_transfer < 0.05),
    }
    for case in cases:
        for line in case["lines"]:
            line["axis_limits"] = {
                key: np.asarray(value).tolist() for key, value in line["axis_limits"].items()
            }
            line["matched_regular_acceleration"] = np.asarray(
                line["matched_regular_acceleration"]
            ).tolist()
    summary = {
        "matched_regular_G5_G6_relative_difference": matched_transfer,
        "axis_G5_G6_relative_difference": axis_transfer,
        "maximum_axis_fit_relative_difference": maximum_axis_fit,
        "matched_field_scaled_relative_differences": matched_fields,
        "axis_field_scaled_relative_differences": axis_fields,
    }
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "scope": "native-grid quotient and removable-axis regularization of the nonlinear regular SO(3) bulk acceleration candidate",
        "field_order": list(FIELD_ORDER),
        "quotient_powers": QUOTIENT_POWERS.tolist(),
        "z_fractions": list(Z_FRACTIONS),
        "matched_r": list(MATCHED_R),
        "axis_fit_counts": list(AXIS_FIT_COUNTS),
        "cases": cases,
        "summary": summary,
        "acceptance": acceptance,
        "limitations": [
            "radial-line assembly at five open-bulk compact locations",
            "cubic axis extrapolation in squared radius",
            "fixed-stage initial gauge-source jets",
            "nonlinear compact-wall rows and time integration remain open",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "summary": summary, "acceptance": acceptance}, indent=2), flush=True)


if __name__ == "__main__":
    main()
