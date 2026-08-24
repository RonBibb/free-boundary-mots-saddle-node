"""Dense primary-window finite-collar assessment for Test 14D."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from bhps.test14d_physical_collar import (
    FAMILIES,
    RATE_PATHS,
    WIDTH_RATIOS,
    physical_collar_record,
)


GRIDS = ("G7", "G8")
BRANCHES = ("inner", "outer")
STRIDES = (1, 2, 4)
PRIMARY_LEFT = 0.000625
PRIMARY_RIGHT = 0.004
WINDOWS = {
    "primary": (PRIMARY_LEFT, PRIMARY_RIGHT),
    "sub_001_002": (0.001, 0.002),
    "sub_002_003": (0.002, 0.003),
    "sub_003_004": (0.003, 0.004),
}


def dense_physical_collar_record(
    test14c_record, test14b_record, family, width_ratio,
    nodes_per_epsilon, rate_path,
):
    """Add cumulative-balance bookkeeping to one physical collar record."""
    record = physical_collar_record(
        test14c_record, test14b_record, family, width_ratio,
        nodes_per_epsilon, rate_path,
    )
    thin_rates = {
        key: float(value)
        for key, value in test14c_record["corrected_rates"].items()
    }
    finite_rates = {
        key: value for key, value in thin_rates.items()
        if key not in ("coupled_seam_global_radius", "coupled_seam_joint_work")
    }
    finite_rates["finite_collar_seam"] = float(record["finite_seam_rate"])
    finite_sum = float(sum(finite_rates.values()))
    if not np.isclose(
        finite_sum, record["finite_corrected_total_rate"],
        rtol=2e-13, atol=2e-10,
    ):
        raise RuntimeError("finite collar rate replacement does not sum")
    return {
        **record,
        "charge": float(test14c_record["charge"]),
        "thin_corrected_rates": thin_rates,
        "finite_corrected_rates": finite_rates,
        "finite_corrected_total_rate": finite_sum,
    }


def window_summary(records, left, right):
    """Integrate one fixed grid/branch/stride/profile/width/path history."""
    selected = sorted(
        [item for item in records if left - 1e-12 <= item["time"] <= right + 1e-12],
        key=lambda item: item["time"],
    )
    if len(selected) < 2:
        raise ValueError("Test-14D dense window has fewer than two samples")
    times = np.asarray([item["time"] for item in selected], dtype=float)
    charges = np.asarray([item["charge"] for item in selected], dtype=float)
    names = tuple(selected[0]["finite_corrected_rates"])
    arrays = {
        name: np.asarray([
            item["finite_corrected_rates"][name] for item in selected
        ], dtype=float)
        for name in names
    }
    total = np.sum(np.stack(list(arrays.values())), axis=0)
    integrated = {
        name: float(np.trapezoid(values, times))
        for name, values in arrays.items()
    }
    integrated_total = float(np.trapezoid(total, times))
    delta = float(charges[-1] - charges[0])
    flux_norm = float(np.trapezoid(
        np.sum(np.abs(np.stack(list(arrays.values()))), axis=0), times,
    ))
    norm = max(abs(delta), flux_norm, 1e-12)
    residual = float(delta - integrated_total)
    return {
        "left_time": float(times[0]),
        "right_time": float(times[-1]),
        "sample_count": len(selected),
        "delta_charge": delta,
        "integrated_named_rates": integrated,
        "integrated_total_rate": integrated_total,
        "integrated_finite_collar_seam": integrated["finite_collar_seam"],
        "closure_residual": residual,
        "balance_norm": norm,
        "normalized_absolute_residual": abs(residual) / norm,
    }


def extrapolated_history(records):
    """Apply the sealed quadratic width fit independently at every time."""
    times = sorted(set(float(item["time"]) for item in records))
    output = []
    for current_time in times:
        selected = sorted(
            [item for item in records if item["time"] == current_time],
            key=lambda item: item["epsilon_over_RA"], reverse=True,
        )
        if len(selected) != 4:
            raise ValueError("dense extrapolation requires four widths per time")
        widths = np.asarray([
            item["epsilon_over_RA"] for item in selected
        ], dtype=float)
        values = np.asarray([
            item["finite_seam_rate"] for item in selected
        ], dtype=float)
        extrapolated = float(np.polyfit(widths, values, 2)[-1])
        item = deepcopy(selected[-1])
        item["epsilon_over_RA"] = 0.0
        item["epsilon"] = 0.0
        item["width_extrapolated"] = True
        item["finite_seam_rate"] = extrapolated
        item["finite_corrected_rates"]["finite_collar_seam"] = extrapolated
        item["finite_corrected_total_rate"] = float(sum(
            item["finite_corrected_rates"].values()
        ))
        item["finite_pointwise_residual"] = float(
            item["charge_rate_target"] - item["finite_corrected_total_rate"]
        )
        item["finite_pointwise_normalized_residual"] = float(
            abs(item["finite_pointwise_residual"])
            / item["pointwise_balance_norm_rate"]
        )
        output.append(item)
    return output


def _admit_transfer(first, second, norm, relative_gate, normalized_gate):
    difference = abs(float(first) - float(second))
    relative = difference / max(abs(float(first)), abs(float(second)), 1e-300)
    normalized = difference / max(float(norm), 1e-300)
    return {
        "first": float(first),
        "second": float(second),
        "absolute_difference": difference,
        "relative_difference": relative,
        "normalized_difference": normalized,
        "passed": bool(relative < relative_gate or normalized < normalized_gate),
    }


def dense_assessment(records):
    """Apply the predeclared Test-14D primary-window and subwindow gates."""
    records = list(records)
    time_count = len(set(float(item["time"]) for item in records))
    expected_count = (
        len(GRIDS) * len(BRANCHES) * time_count * len(STRIDES)
        * len(FAMILIES) * len(WIDTH_RATIOS) * len(RATE_PATHS)
    )
    if len(records) != expected_count:
        raise ValueError(f"dense matrix has {len(records)}, expected {expected_count}")

    summaries = {}
    for grid in GRIDS:
        summaries[grid] = {}
        for branch in BRANCHES:
            summaries[grid][branch] = {}
            for stride in STRIDES:
                summaries[grid][branch][str(stride)] = {}
                for family in FAMILIES:
                    summaries[grid][branch][str(stride)][family] = {}
                    for path in RATE_PATHS:
                        selected = [
                            item for item in records
                            if item["grid"] == grid and item["branch"] == branch
                            and item["stride"] == stride
                            and item["profile"] == family
                            and item["rate_path"] == path
                        ]
                        width_summaries = {}
                        for ratio in WIDTH_RATIOS:
                            history = [
                                item for item in selected
                                if item["epsilon_over_RA"] == ratio
                            ]
                            width_summaries[str(ratio)] = {
                                name: window_summary(history, *limits)
                                for name, limits in WINDOWS.items()
                            }
                        extrapolated = extrapolated_history(selected)
                        extrapolated_windows = {
                            name: window_summary(extrapolated, *limits)
                            for name, limits in WINDOWS.items()
                        }
                        summaries[grid][branch][str(stride)][family][path] = {
                            "widths": width_summaries,
                            "extrapolated": extrapolated_windows,
                        }

    finest_key = str(min(WIDTH_RATIOS))
    primary_checks = []
    subwindow_checks = []
    for grid in GRIDS:
        for branch in BRANCHES:
            for stride in STRIDES:
                for family in FAMILIES:
                    for path in RATE_PATHS:
                        item = summaries[grid][branch][str(stride)][family][path]
                        finest = item["widths"][finest_key]["primary"]
                        extrapolated = item["extrapolated"]["primary"]
                        primary_checks.append({
                            "grid": grid, "branch": branch, "stride": stride,
                            "profile": family, "rate_path": path,
                            "finest_residual": finest["normalized_absolute_residual"],
                            "extrapolated_residual": extrapolated[
                                "normalized_absolute_residual"
                            ],
                            "passed": bool(
                                finest["normalized_absolute_residual"] < 0.05
                                and extrapolated["normalized_absolute_residual"] < 0.02
                            ),
                        })
                        for window_name in WINDOWS:
                            if window_name == "primary":
                                continue
                            finest_window = item["widths"][finest_key][window_name]
                            extrapolated_window = item["extrapolated"][window_name]
                            subwindow_checks.append({
                                "grid": grid, "branch": branch, "stride": stride,
                                "profile": family, "rate_path": path,
                                "window": window_name,
                                "finest_residual": finest_window[
                                    "normalized_absolute_residual"
                                ],
                                "extrapolated_residual": extrapolated_window[
                                    "normalized_absolute_residual"
                                ],
                                "passed": bool(
                                    finest_window["normalized_absolute_residual"] < 0.10
                                    and extrapolated_window[
                                        "normalized_absolute_residual"
                                    ] < 0.10
                                ),
                            })

    profile_checks = []
    path_checks = []
    for grid in GRIDS:
        for branch in BRANCHES:
            for stride in STRIDES:
                for path in RATE_PATHS:
                    windows = [
                        summaries[grid][branch][str(stride)][family][path][
                            "extrapolated"
                        ]["primary"] for family in FAMILIES
                    ]
                    seams = [item["integrated_finite_collar_seam"] for item in windows]
                    residuals = [item["closure_residual"] for item in windows]
                    norm = max(item["balance_norm"] for item in windows)
                    seam_spread = max(seams) - min(seams)
                    residual_spread = max(residuals) - min(residuals)
                    relative = seam_spread / max(max(abs(v) for v in seams), 1e-300)
                    profile_checks.append({
                        "grid": grid, "branch": branch, "stride": stride,
                        "rate_path": path,
                        "seam_relative_spread": relative,
                        "seam_normalized_spread": seam_spread / norm,
                        "residual_normalized_spread": residual_spread / norm,
                        "passed": bool(
                            (relative < 0.02 or seam_spread / norm < 0.01)
                            and residual_spread / norm < 0.01
                        ),
                    })
                for family in FAMILIES:
                    geometry = summaries[grid][branch][str(stride)][family][
                        "geometry"
                    ]["extrapolated"]["primary"]
                    wall = summaries[grid][branch][str(stride)][family][
                        "wall"
                    ]["extrapolated"]["primary"]
                    norm = max(geometry["balance_norm"], wall["balance_norm"])
                    seam_check = _admit_transfer(
                        geometry["integrated_finite_collar_seam"],
                        wall["integrated_finite_collar_seam"], norm, 0.02, 0.01,
                    )
                    residual_check = _admit_transfer(
                        geometry["closure_residual"], wall["closure_residual"],
                        norm, 1.0, 0.01,
                    )
                    path_checks.append({
                        "grid": grid, "branch": branch, "stride": stride,
                        "profile": family,
                        "seam": seam_check,
                        "residual": residual_check,
                        "passed": bool(seam_check["passed"] and residual_check["passed"]),
                    })

    grid_checks = []
    for branch in BRANCHES:
        for stride in STRIDES:
            for family in FAMILIES:
                for path in RATE_PATHS:
                    g7 = summaries["G7"][branch][str(stride)][family][path][
                        "extrapolated"
                    ]["primary"]
                    g8 = summaries["G8"][branch][str(stride)][family][path][
                        "extrapolated"
                    ]["primary"]
                    norm = max(g7["balance_norm"], g8["balance_norm"])
                    seam_check = _admit_transfer(
                        g7["integrated_finite_collar_seam"],
                        g8["integrated_finite_collar_seam"], norm, 0.15, 0.02,
                    )
                    residual_plateau = bool(
                        abs(g8["closure_residual"]) <= abs(g7["closure_residual"])
                        or g8["normalized_absolute_residual"] < 0.02
                    )
                    grid_checks.append({
                        "branch": branch, "stride": stride,
                        "profile": family, "rate_path": path,
                        "seam_transfer": seam_check,
                        "residual_refines_or_plateaus": residual_plateau,
                        "passed": bool(seam_check["passed"] and residual_plateau),
                    })

    stride_checks = []
    for grid in GRIDS:
        for branch in BRANCHES:
            for family in FAMILIES:
                for path in RATE_PATHS:
                    native = summaries[grid][branch]["1"][family][path][
                        "extrapolated"
                    ]["primary"]
                    stride2 = summaries[grid][branch]["2"][family][path][
                        "extrapolated"
                    ]["primary"]
                    norm = max(native["balance_norm"], stride2["balance_norm"])
                    check = _admit_transfer(
                        native["integrated_total_rate"],
                        stride2["integrated_total_rate"], norm, 1.0, 0.05,
                    )
                    stride_checks.append({
                        "grid": grid, "branch": branch,
                        "profile": family, "rate_path": path,
                        **check,
                    })

    unique_rate_records = [
        item for item in records
        if item["profile"] == FAMILIES[0]
        and item["epsilon_over_RA"] == WIDTH_RATIOS[0]
        and item["rate_path"] == "geometry"
    ]
    israel_rate_checks = [{
        "grid": item["grid"], "branch": item["branch"],
        "time": item["time"], "stride": item["stride"],
        "absolute_difference": item["israel_rate_absolute_difference"],
        "relative_scale_error": item["israel_rate_relative_scale_error"],
        "passed": bool(
            item["israel_rate_relative_scale_error"] < 0.05
            or item["israel_rate_absolute_difference"] < 0.005
        ),
    } for item in unique_rate_records]

    gates = {
        "complete_matrix": len(records) == expected_count,
        "finite": all(item["finite"] for item in records),
        "stored_thin_formula": max(
            item["stored_thin_formula_error"] for item in records
        ) < 2e-10,
        "integrated_israel_magnitude": max(
            item["integrated_israel_magnitude_error"] for item in records
        ) < 0.01,
        "israel_rate_compatibility": all(
            item["passed"] for item in israel_rate_checks
        ),
        "primary_balance": all(item["passed"] for item in primary_checks),
        "subwindow_balance": all(item["passed"] for item in subwindow_checks),
        "profile_universality": all(item["passed"] for item in profile_checks),
        "rate_path_universality": all(item["passed"] for item in path_checks),
        "grid_transfer": all(item["passed"] for item in grid_checks),
        "time_stride": all(item["passed"] for item in stride_checks),
    }
    numerical_gates = {
        key: value for key, value in gates.items()
        if key != "israel_rate_compatibility"
    }
    return {
        "schema": "bhps-test14d-dense-assessment-v1",
        "record_count": len(records),
        "expected_record_count": expected_count,
        "time_count": time_count,
        "windows": WINDOWS,
        "summaries": summaries,
        "primary_checks": primary_checks,
        "subwindow_checks": subwindow_checks,
        "profile_checks": profile_checks,
        "rate_path_checks": path_checks,
        "grid_checks": grid_checks,
        "stride_checks": stride_checks,
        "israel_rate_checks": israel_rate_checks,
        "summary": {
            "maximum_finest_primary_residual": max(
                item["finest_residual"] for item in primary_checks
            ),
            "maximum_extrapolated_primary_residual": max(
                item["extrapolated_residual"] for item in primary_checks
            ),
            "maximum_finest_subwindow_residual": max(
                item["finest_residual"] for item in subwindow_checks
            ),
            "maximum_extrapolated_subwindow_residual": max(
                item["extrapolated_residual"] for item in subwindow_checks
            ),
            "maximum_israel_rate_relative_scale_error": max(
                item["relative_scale_error"] for item in israel_rate_checks
            ),
            "maximum_israel_rate_absolute_difference": max(
                item["absolute_difference"] for item in israel_rate_checks
            ),
            "israel_rate_pass_fraction": float(np.mean([
                item["passed"] for item in israel_rate_checks
            ])),
        },
        "gates": gates,
        "numerical_balance_subgrade": (
            "PASS" if all(numerical_gates.values()) else "FAIL"
        ),
        "physical_israel_rate_subgrade": (
            "PASS" if gates["israel_rate_compatibility"] else "REVIEW"
        ),
        "overall_grade": (
            "PASS" if all(gates.values())
            else "REVIEW" if all(numerical_gates.values())
            else "FAIL"
        ),
    }
