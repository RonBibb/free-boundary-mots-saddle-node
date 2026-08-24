#!/usr/bin/env python3
"""Use grid-independent physical windows for nonlinear SO(3) axis limits."""

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from run_corrected_fold_nonlinear_rhs_axis_regularization import (
    FIELD_ORDER,
    MATCHED_R,
    QUOTIENT_POWERS,
    Z_FRACTIONS,
    build_geometry,
    relative_norm,
    scaled_field_differences,
    solve_raw,
)


AXIS_WINDOWS = (0.5, 0.75)
AXIS_DEGREE = 2
OUTPUT = Path("results/corrected_fold_nonlinear_rhs_fixed_window_axis.json")


def fixed_window_axis_limit(positive_r, regular_positive, window):
    keep = positive_r <= float(window) + 1e-12
    squared = positive_r[keep] ** 2
    if np.count_nonzero(keep) < AXIS_DEGREE + 1:
        raise ValueError("axis window has too few native points")
    result = np.empty(regular_positive.shape[1])
    for field in range(regular_positive.shape[1]):
        result[field] = np.polynomial.polynomial.polyfit(
            squared, regular_positive[keep, field], AXIS_DEGREE,
        )[0]
    return result, int(np.count_nonzero(keep))


def assemble_line(geometry, z_value):
    positive_r = np.asarray(geometry["r"])[1:]
    positive_r = positive_r[positive_r <= 4.5]
    raw = np.asarray([solve_raw(geometry, z_value, r_value) for r_value in positive_r])
    regular_positive = raw / positive_r[:, None] ** QUOTIENT_POWERS[None, :]
    limits = {}
    counts = {}
    for window in AXIS_WINDOWS:
        value, count = fixed_window_axis_limit(positive_r, regular_positive, window)
        limits[str(window)] = value
        counts[str(window)] = count
    selected = limits[str(AXIS_WINDOWS[0])]
    radial_grid = np.r_[0.0, positive_r]
    regular_grid = np.vstack((selected, regular_positive))
    matched = np.empty((len(MATCHED_R), len(FIELD_ORDER)))
    for field in range(len(FIELD_ORDER)):
        matched[:, field] = CubicSpline(radial_grid, regular_grid[:, field])(
            np.asarray(MATCHED_R)
        )
    return {
        "z": float(z_value),
        "axis_limits": limits,
        "axis_window_native_counts": counts,
        "axis_window_relative_difference": relative_norm(
            limits[str(AXIS_WINDOWS[0])], limits[str(AXIS_WINDOWS[1])],
        ),
        "matched_regular_acceleration": matched,
        "finite": bool(np.all(np.isfinite(regular_grid)) and np.all(np.isfinite(matched))),
    }


def build_case(resolution):
    geometry = build_geometry(resolution)
    z0 = float(geometry["z"][0])
    z1 = float(geometry["z"][-1])
    return {
        "resolution": resolution,
        "source_grid": geometry["source_grid"],
        "selector_maximum": geometry["selector_maximum"],
        "lines": [
            assemble_line(geometry, z0 + fraction * (z1 - z0))
            for fraction in Z_FRACTIONS
        ],
    }


def main():
    cases = [build_case(resolution) for resolution in ("G5", "G6")]
    coarse, fine = cases
    selected = str(AXIS_WINDOWS[0])
    coarse_matched = np.asarray([line["matched_regular_acceleration"] for line in coarse["lines"]])
    fine_matched = np.asarray([line["matched_regular_acceleration"] for line in fine["lines"]])
    coarse_axis = np.asarray([line["axis_limits"][selected] for line in coarse["lines"]])
    fine_axis = np.asarray([line["axis_limits"][selected] for line in fine["lines"]])
    matched_transfer = relative_norm(coarse_matched, fine_matched)
    axis_transfer = relative_norm(coarse_axis, fine_axis)
    maximum_window_difference = max(
        line["axis_window_relative_difference"] for case in cases for line in case["lines"]
    )
    acceptance = {
        "all_values_finite": bool(all(line["finite"] for case in cases for line in case["lines"])),
        "selectors_below_1e_8": bool(all(case["selector_maximum"] < 1e-8 for case in cases)),
        "fixed_window_matched_transfer_below_5_percent": bool(matched_transfer < 0.05),
        "fixed_window_axis_robustness_below_5_percent": bool(maximum_window_difference < 0.05),
        "fixed_window_axis_G5_G6_transfer_below_5_percent": bool(axis_transfer < 0.05),
    }
    summary = {
        "matched_regular_G5_G6_relative_difference": matched_transfer,
        "axis_G5_G6_relative_difference": axis_transfer,
        "maximum_axis_window_relative_difference": maximum_window_difference,
        "matched_field_scaled_relative_differences": scaled_field_differences(coarse_matched, fine_matched),
        "axis_field_scaled_relative_differences": scaled_field_differences(coarse_axis, fine_axis),
    }
    for case in cases:
        for line in case["lines"]:
            line["axis_limits"] = {
                key: np.asarray(value).tolist() for key, value in line["axis_limits"].items()
            }
            line["matched_regular_acceleration"] = np.asarray(
                line["matched_regular_acceleration"]
            ).tolist()
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "scope": "grid-independent physical-window axis regularization of the nonlinear regular SO(3) bulk acceleration candidate",
        "field_order": list(FIELD_ORDER),
        "z_fractions": list(Z_FRACTIONS),
        "matched_r": list(MATCHED_R),
        "axis_windows": list(AXIS_WINDOWS),
        "axis_polynomial_degree": AXIS_DEGREE,
        "cases": cases,
        "summary": summary,
        "acceptance": acceptance,
        "limitations": [
            "quadratic even fits on fixed r<=0.5 and r<=0.75 physical windows",
            "five open-bulk compact locations",
            "fixed-stage initial gauge-source jets",
            "nonlinear compact-wall rows and finite-time evolution remain open",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "summary": summary, "acceptance": acceptance}, indent=2), flush=True)


if __name__ == "__main__":
    main()
