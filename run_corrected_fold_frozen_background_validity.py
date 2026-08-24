#!/usr/bin/env python3
"""Estimate the time window in which the corrected initial slice may be frozen."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from run_corrected_fold_boundary_constraint_pulse import build_geometry


OUTPUT = Path("results/corrected_fold_frozen_background_validity.json")


def metric_from_regular(values, radius):
    h_z0, v_z, h00, h_perp, d, v_0, h_zz = np.asarray(values)[:7]
    result = np.zeros((5, 5))
    result[0, 0] = h00
    result[1, 1] = h_zz
    result[2, 2] = h_perp + radius**2 * d
    result[3, 3] = h_perp
    result[4, 4] = h_perp
    result[0, 1] = result[1, 0] = h_z0
    result[1, 2] = result[2, 1] = radius * v_z
    result[0, 2] = result[2, 0] = radius * v_0
    return result


def local_record(z, r, values, accelerations):
    metric = metric_from_regular(values, r)
    metric_acceleration = metric_from_regular(accelerations, r)
    diagonal = np.diag(metric)
    frame = 1 / np.sqrt(np.maximum(np.abs(diagonal), 1e-300))
    orthonormal = metric_acceleration * frame[:, None] * frame[None, :]
    maximum = float(np.max(np.abs(orthonormal)))
    frobenius = float(np.linalg.norm(orthonormal))
    maximum_index = np.unravel_index(np.argmax(np.abs(orthonormal)), orthonormal.shape)
    return {
        "z": float(z),
        "r": float(r),
        "orthonormal_metric_acceleration_maximum": maximum,
        "orthonormal_metric_acceleration_frobenius": frobenius,
        "maximum_component": [int(value) for value in maximum_index],
        "time_to_1_percent_quadratic_metric_change": float(np.sqrt(0.02 / max(maximum, 1e-300))),
        "time_to_10_percent_quadratic_metric_change": float(np.sqrt(0.2 / max(maximum, 1e-300))),
        "time_to_order_one_quadratic_metric_change": float(np.sqrt(2.0 / max(maximum, 1e-300))),
        "estimated_fractional_metric_change_at_t0p1": float(0.5 * maximum * 0.1**2),
        "estimated_fractional_metric_change_at_t0p8": float(0.5 * maximum * 0.8**2),
        "estimated_fractional_metric_change_at_t3": float(0.5 * maximum * 3.0**2),
    }


def main():
    geometry = build_geometry("G6")
    field = geometry["jet_field"]
    values = np.asarray(field.reduced_fields)
    first_time = np.asarray(field.reduced_first[0])
    accelerations = np.asarray(field.reduced_second[0, 0])
    records = []
    for i, z_value in enumerate(field.z):
        for j, r_value in enumerate(field.r):
            records.append(local_record(
                z_value, r_value, values[i, j], accelerations[i, j],
            ))
    summaries = []
    for cutoff in (1.5, 2.0, 3.0, 4.0, 8.0):
        selected = [item for item in records if item["r"] <= cutoff + 1e-12]
        worst = max(
            selected, key=lambda item: item["orthonormal_metric_acceleration_maximum"]
        )
        summaries.append({
            "radial_cutoff": cutoff,
            "maximum_orthonormal_metric_acceleration": worst["orthonormal_metric_acceleration_maximum"],
            "worst_location": [worst["z"], worst["r"]],
            "time_to_1_percent_quadratic_metric_change": worst["time_to_1_percent_quadratic_metric_change"],
            "time_to_10_percent_quadratic_metric_change": worst["time_to_10_percent_quadratic_metric_change"],
            "time_to_order_one_quadratic_metric_change": worst["time_to_order_one_quadratic_metric_change"],
            "estimated_fractional_metric_change_at_t0p1": worst["estimated_fractional_metric_change_at_t0p1"],
            "estimated_fractional_metric_change_at_t0p8": worst["estimated_fractional_metric_change_at_t0p8"],
            "estimated_fractional_metric_change_at_t3": worst["estimated_fractional_metric_change_at_t3"],
            "maximum_component": worst["maximum_component"],
        })
    target_points = []
    for z_target, r_target in ((2.6, 1.5), (2.6, 2.0), (2.6, 3.0), (2.45, 3.4)):
        nearest = min(
            records, key=lambda item: abs(item["z"] - z_target) + abs(item["r"] - r_target)
        )
        target_points.append({
            "target": [z_target, r_target],
            "nearest_grid_record": nearest,
        })
    r4 = next(item for item in summaries if item["radial_cutoff"] == 4.0)
    acceptance = {
        "background_time_first_derivatives_zero": bool(np.max(np.abs(first_time)) < 1e-13),
        "t0p1_r4_quadratic_change_below_10_percent": bool(
            r4["estimated_fractional_metric_change_at_t0p1"] < 0.1
        ),
        "t0p8_r4_quadratic_change_below_10_percent": bool(
            r4["estimated_fractional_metric_change_at_t0p8"] < 0.1
        ),
        "t3_r4_quadratic_change_below_10_percent": bool(
            r4["estimated_fractional_metric_change_at_t3"] < 0.1
        ),
    }
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "scope": "quadratic-in-time validity estimate for freezing the corrected-G6 time-symmetric initial slice",
        "source_grid": [len(field.z), len(field.r)],
        "summaries": summaries,
        "target_points": target_points,
        "acceptance": acceptance,
        "interpretation_rule": (
            "For zero first-time derivatives, half the orthonormal second-time derivative times t^2 estimates the earliest fractional metric drift; a failed threshold means the frozen-background evolution cannot be interpreted as the perturbation dynamics of the evolving slice over that duration."
        ),
        "limitations": [
            "local Taylor estimate rather than an evolved nonlinear background",
            "the estimate uses the largest orthonormal metric component and ignores cancellations",
            "passing a short-time threshold is necessary but not sufficient for a controlled frozen-background approximation",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "summaries": summaries,
        "target_points": target_points,
        "acceptance": acceptance,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
