#!/usr/bin/env python3
"""Post-hoc diagnosis of the sealed note-79 raw-field monotonicity review."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.regular_so3_gh_reduction import FIELD_ORDER
from run_corrected_A790_third_grid_formation_reproduction import generalized_order
from run_corrected_fold_short_nonlinear_evolution import interpolate_fields


BASELINE = Path("results/corrected_A790_formation_time_refinement_state.npz")
G9_STATE = Path("results/corrected_A790_third_grid_formation_reproduction_state.npz")
SEALED_RESULT = Path("results/corrected_A790_third_grid_formation_reproduction.json")
OUTPUT = Path("results/corrected_A790_third_grid_review_diagnosis.json")
RADIAL_CUT = 6.0


def difference_summary(left, middle, right):
    error_78 = np.asarray(left) - np.asarray(middle)
    error_89 = np.asarray(middle) - np.asarray(right)
    d78 = float(np.linalg.norm(error_78))
    d89 = float(np.linalg.norm(error_89))
    correlation = float(
        np.vdot(error_78.ravel(), error_89.ravel()).real
        / max(d78 * d89, 1e-300)
    )
    return {
        "G7_G8_absolute_difference": d78,
        "G8_G9_absolute_difference": d89,
        "difference_ratio_G7G8_over_G8G9": d78 / max(d89, 1e-300),
        "generalized_empirical_order": generalized_order(d78, d89),
        "error_correlation": correlation,
        "difference_decreases": bool(d89 < d78),
    }


def common_fields(baseline, g9, key):
    z7 = baseline["G7_z"]
    r7 = baseline["G7_r"]
    mask = r7 <= RADIAL_CUT + 1e-12
    target_r = r7[mask]
    left = np.asarray(baseline[f"G7_8step_{key}"])[:, mask]
    middle = interpolate_fields(
        baseline[f"G8_8step_{key}"], baseline["G8_z"], baseline["G8_r"],
        z7, target_r,
    )
    right = interpolate_fields(
        g9[f"G9_8step_{key}"], g9["G9_z"], g9["G9_r"], z7, target_r,
    )
    return target_r, left, middle, right


def diagnose_metric_field(baseline, g9, key):
    r, left, middle, right = common_fields(baseline, g9, key)
    components = {
        FIELD_ORDER[index]: difference_summary(
            left[:, :, index], middle[:, :, index], right[:, :, index],
        )
        for index in range(len(FIELD_ORDER))
    }
    raw = difference_summary(left, middle, right)
    raw_78_squared = raw["G7_G8_absolute_difference"] ** 2
    raw_89_squared = raw["G8_G9_absolute_difference"] ** 2
    anisotropy = components[FIELD_ORDER[4]]

    outside_axis = difference_summary(
        left[:, 2:], middle[:, 2:], right[:, 2:],
    )
    physical = [array.copy() for array in (left, middle, right)]
    radius = r[None, :]
    for array in physical:
        array[:, :, 1] *= radius
        array[:, :, 5] *= radius
        array[:, :, 4] *= radius**2
    geometric_proxy = difference_summary(*physical)
    geometric_proxy["G7_G8_relative_difference"] = (
        geometric_proxy["G7_G8_absolute_difference"]
        / max(float(np.linalg.norm(physical[0])), float(np.linalg.norm(physical[1])), 1e-300)
    )
    geometric_proxy["G8_G9_relative_difference"] = (
        geometric_proxy["G8_G9_absolute_difference"]
        / max(float(np.linalg.norm(physical[1])), float(np.linalg.norm(physical[2])), 1e-300)
    )
    return {
        "raw_reduced_field": raw,
        "by_component": components,
        "nondecreasing_components": [
            name for name, item in components.items()
            if not item["difference_decreases"]
        ],
        "anisotropy_d_fraction_of_squared_difference": {
            "G7_G8": anisotropy["G7_G8_absolute_difference"] ** 2
            / max(raw_78_squared, 1e-300),
            "G8_G9": anisotropy["G8_G9_absolute_difference"] ** 2
            / max(raw_89_squared, 1e-300),
        },
        "excluding_first_two_radial_nodes": outside_axis,
        "coordinate_component_proxy": geometric_proxy,
        "coordinate_component_proxy_definition": (
            "multiply v_z and v_0 by r and d by r^2; this reconstructs their "
            "coordinate-component weights but is not an invariant tensor norm"
        ),
    }


def main():
    sealed = json.loads(SEALED_RESULT.read_text())
    if sealed["status"] != "review":
        raise RuntimeError("diagnosis expects the unchanged sealed REVIEW result")
    baseline = np.load(BASELINE)
    g9 = np.load(G9_STATE)
    position = diagnose_metric_field(baseline, g9, "increment")
    velocity = diagnose_metric_field(baseline, g9, "velocity")
    source_r, source7, source8, source9 = common_fields(
        baseline, g9, "source_increment",
    )
    source = difference_summary(source7, source8, source9)
    payload = {
        "status": "diagnosis_complete_sealed_review_unchanged",
        "scope": "post-hoc localization of the one failed note-79 monotonicity rule",
        "sealed_result": str(SEALED_RESULT),
        "field_order": list(FIELD_ORDER),
        "radial_cut": RADIAL_CUT,
        "first_two_common_grid_radii": source_r[:2].tolist(),
        "position_increment": position,
        "velocity": velocity,
        "source_increment": source,
        "interpretation": [
            "the only nondecreasing reduced component is d=(h_rr-h_perp)/r^2",
            "the raw failure is localized to the first two common-grid radial nodes",
            "all raw fields jointly decrease after excluding those two nodes",
            "a coordinate-component-weighted proxy decreases with positive generalized order",
            "this diagnostic does not retroactively alter the sealed REVIEW classification",
        ],
        "limitations": [
            "post-hoc diagnostic rather than a prospectively accepted test",
            "coordinate-component proxy is not an invariant spacetime norm",
            "a dedicated axis-refinement audit is required to resolve the raw d coefficient",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "position_nondecreasing_components": position["nondecreasing_components"],
        "position_axis_excluded": position["excluding_first_two_radial_nodes"],
        "position_coordinate_proxy": position["coordinate_component_proxy"],
        "velocity_coordinate_proxy": velocity["coordinate_component_proxy"],
        "anisotropy_fraction": position["anisotropy_d_fraction_of_squared_difference"],
        "source": source,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
