#!/usr/bin/env python3
"""Prospectively sealed near-axis refinement audit for the A=7.90 d field."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.regular_so3_gh_reduction import FIELD_ORDER
from run_corrected_A790_third_grid_formation_reproduction import generalized_order
from run_corrected_fold_short_nonlinear_evolution import interpolate_fields


BASELINE_STATE = Path("results/corrected_A790_formation_time_refinement_state.npz")
G9_STATE = Path("results/corrected_A790_third_grid_formation_reproduction_state.npz")
NOTE79_RESULT = Path("results/corrected_A790_third_grid_formation_reproduction.json")
OUTPUT = Path("results/corrected_A790_near_axis_d_refinement_audit.json")
PROTOCOL = "notes/83_A790_near_axis_d_refinement_audit_protocol.md"
FIELD_INDEX = 4
RADIAL_CUT = 6.0
FIT_VARIANTS = (
    ("degree3_r0.5_primary", 3, 0.5),
    ("degree2_r0.5", 2, 0.5),
    ("degree2_r0.666667", 2, 2.0 / 3.0),
    ("degree3_r0.666667", 3, 2.0 / 3.0),
)
PRIMARY = FIT_VARIANTS[0][0]
EXPECTED_SIZES = {"G7": (81, 121), "G8": (97, 145), "G9": (113, 169)}


def relative_norm(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return float(
        np.linalg.norm(left - right)
        / max(np.linalg.norm(left), np.linalg.norm(right), 1e-300)
    )


def even_polynomial_fit(values, r, degree, window):
    """Fit every z row in positive-radius samples as a polynomial in r^2."""
    values = np.asarray(values, dtype=float)
    r = np.asarray(r, dtype=float)
    keep = (r > 0.0) & (r <= float(window) + 1e-12)
    x = r[keep] ** 2
    design = np.polynomial.polynomial.polyvander(x, int(degree))
    coefficients, _, _, _ = np.linalg.lstsq(design, values[:, keep].T, rcond=None)
    coefficients = coefficients.T
    fitted = coefficients @ design.T
    sample = values[:, keep]
    residual = float(
        np.linalg.norm(fitted - sample) / max(np.linalg.norm(sample), 1e-300)
    )
    return {
        "coefficients": coefficients,
        "axis": coefficients[:, 0],
        "fit_residual": residual,
        "positive_node_count": int(np.count_nonzero(keep)),
        "native_radii": r[keep],
    }


def evaluate_fit(coefficients, radii):
    coefficients = np.asarray(coefficients, dtype=float)
    x = np.asarray(radii, dtype=float) ** 2
    design = np.polynomial.polynomial.polyvander(x, coefficients.shape[1] - 1)
    return coefficients @ design.T


def transfer_axis(values, source_z, target_z):
    values = np.asarray(values, dtype=float)
    if np.array_equal(np.asarray(source_z), np.asarray(target_z)):
        return values.copy()
    return CubicSpline(source_z, values, axis=0)(target_z)


def difference_metrics(g7, g8, g9):
    d78 = float(np.linalg.norm(np.asarray(g7) - np.asarray(g8)))
    d89 = float(np.linalg.norm(np.asarray(g8) - np.asarray(g9)))
    return {
        "G7_G8_absolute_difference": d78,
        "G8_G9_absolute_difference": d89,
        "difference_ratio_G7G8_over_G8G9": d78 / max(d89, 1e-300),
        "generalized_empirical_order": generalized_order(d78, d89),
        "difference_decreases": bool(d89 < d78),
    }


def radial_volume_difference(left, right, z, r):
    difference = np.asarray(left) - np.asarray(right)
    integrand = difference**2 * np.asarray(r)[None, :] ** 2
    radial = np.trapezoid(integrand, x=r, axis=1)
    return float(np.sqrt(max(float(np.trapezoid(radial, x=z)), 0.0)))


def weighted_metrics(g7, g8, g9, z, r):
    volume78 = radial_volume_difference(g7, g8, z, r)
    volume89 = radial_volume_difference(g8, g9, z, r)
    euclidean = difference_metrics(g7, g8, g9)
    return {
        "radial_volume": {
            "G7_G8_absolute_difference": volume78,
            "G8_G9_absolute_difference": volume89,
            "difference_ratio_G7G8_over_G8G9": volume78 / max(volume89, 1e-300),
            "generalized_empirical_order": generalized_order(volume78, volume89),
            "difference_decreases": bool(volume89 < volume78),
        },
        "euclidean": euclidean,
    }


def load_fields():
    baseline = np.load(BASELINE_STATE)
    g9_state = np.load(G9_STATE)
    fields = {
        "position_increment": {
            "G7": np.asarray(baseline["G7_8step_increment"])[:, :, FIELD_INDEX],
            "G8": np.asarray(baseline["G8_8step_increment"])[:, :, FIELD_INDEX],
            "G9": np.asarray(g9_state["G9_8step_increment"])[:, :, FIELD_INDEX],
        },
        "velocity": {
            "G7": np.asarray(baseline["G7_8step_velocity"])[:, :, FIELD_INDEX],
            "G8": np.asarray(baseline["G8_8step_velocity"])[:, :, FIELD_INDEX],
            "G9": np.asarray(g9_state["G9_8step_velocity"])[:, :, FIELD_INDEX],
        },
    }
    grids = {
        "G7": (np.asarray(baseline["G7_z"]), np.asarray(baseline["G7_r"])),
        "G8": (np.asarray(baseline["G8_z"]), np.asarray(baseline["G8_r"])),
        "G9": (np.asarray(g9_state["G9_z"]), np.asarray(g9_state["G9_r"])),
    }
    return fields, grids


def analyze_variable(native, grids):
    z7, r7_full = grids["G7"]
    radial_mask = r7_full <= RADIAL_CUT + 1e-12
    r7 = r7_full[radial_mask]
    fits = {label: {} for label in grids}
    for label, (z, r) in grids.items():
        for name, degree, window in FIT_VARIANTS:
            fits[label][name] = even_polynomial_fit(
                native[label], r, degree, window,
            )

    axis_variants = {}
    for name, degree, window in FIT_VARIANTS:
        transferred = {
            label: transfer_axis(fits[label][name]["axis"], grids[label][0], z7)
            for label in grids
        }
        axis_variants[name] = {
            "degree_in_r_squared": degree,
            "window": window,
            "positive_node_counts": {
                label: fits[label][name]["positive_node_count"] for label in grids
            },
            "fit_residuals": {
                label: fits[label][name]["fit_residual"] for label in grids
            },
            "three_grid_difference": difference_metrics(
                transferred["G7"], transferred["G8"], transferred["G9"],
            ),
        }

    primary_axes = {
        label: transfer_axis(
            fits[label][PRIMARY]["axis"], grids[label][0], z7,
        ) for label in grids
    }
    fit_robustness = {label: {} for label in grids}
    for label in grids:
        for name, _, _ in FIT_VARIANTS:
            candidate = transfer_axis(
                fits[label][name]["axis"], grids[label][0], z7,
            )
            fit_robustness[label][name] = relative_norm(
                primary_axes[label], candidate,
            )

    stored_axes = {
        label: transfer_axis(native[label][:, 0], grids[label][0], z7)
        for label in grids
    }
    stored_axis_metrics = difference_metrics(
        stored_axes["G7"], stored_axes["G8"], stored_axes["G9"],
    )
    stored_vs_reconstructed = {
        label: relative_norm(stored_axes[label], primary_axes[label])
        for label in grids
    }

    common = {
        "G7": native["G7"][:, radial_mask].copy(),
        "G8": interpolate_fields(
            native["G8"][:, :, None], grids["G8"][0], grids["G8"][1],
            z7, r7,
        )[:, :, 0],
        "G9": interpolate_fields(
            native["G9"][:, :, None], grids["G9"][0], grids["G9"][1],
            z7, r7,
        )[:, :, 0],
    }
    raw_profile_metrics = difference_metrics(
        common["G7"], common["G8"], common["G9"],
    )

    smoothed = {label: values.copy() for label, values in common.items()}
    replacement_radii = r7[:2]
    for label in grids:
        predicted_native = evaluate_fit(
            fits[label][PRIMARY]["coefficients"], replacement_radii,
        )
        predicted_common_z = transfer_axis(
            predicted_native, grids[label][0], z7,
        )
        smoothed[label][:, :2] = predicted_common_z
    smoothed_metrics = difference_metrics(
        smoothed["G7"], smoothed["G8"], smoothed["G9"],
    )

    q = {
        label: values * r7[None, :] ** 2 for label, values in common.items()
    }
    physical_metrics = weighted_metrics(
        q["G7"], q["G8"], q["G9"], z7, r7,
    )
    return {
        "fit_variants": axis_variants,
        "fit_robustness_relative_to_primary": fit_robustness,
        "stored_axis_coefficient": stored_axis_metrics,
        "stored_vs_primary_reconstructed_axis_relative_difference": (
            stored_vs_reconstructed
        ),
        "raw_unweighted_d_profile": raw_profile_metrics,
        "smoothed_two_node_d_profile": smoothed_metrics,
        "replacement_common_radii": replacement_radii.tolist(),
        "physical_qd_equals_r_squared_d": physical_metrics,
    }


def variable_rule_summary(record):
    fit_residual_pass = all(
        residual < 0.05
        for variant in record["fit_variants"].values()
        for residual in variant["fit_residuals"].values()
    )
    fit_robustness_pass = all(
        difference < 0.05
        for grid in record["fit_robustness_relative_to_primary"].values()
        for difference in grid.values()
    )
    reconstructed_axis_pass = all(
        variant["three_grid_difference"]["difference_decreases"]
        and variant["three_grid_difference"]["generalized_empirical_order"] is not None
        and variant["three_grid_difference"]["generalized_empirical_order"] > 0.0
        for variant in record["fit_variants"].values()
    )
    smoothed = record["smoothed_two_node_d_profile"]
    smoothed_pass = bool(
        smoothed["difference_decreases"]
        and smoothed["generalized_empirical_order"] is not None
        and smoothed["generalized_empirical_order"] > 1.0
    )
    physical = record["physical_qd_equals_r_squared_d"]
    volume = physical["radial_volume"]
    physical_pass = bool(
        volume["difference_decreases"]
        and volume["generalized_empirical_order"] is not None
        and volume["generalized_empirical_order"] > 1.0
        and physical["euclidean"]["difference_decreases"]
    )
    return {
        "fit_residuals_below_5_percent": fit_residual_pass,
        "fit_robustness_below_5_percent": fit_robustness_pass,
        "all_reconstructed_axis_variants_have_positive_order": reconstructed_axis_pass,
        "smoothed_profile_order_above_1": smoothed_pass,
        "physical_qd_volume_order_above_1_and_euclidean_decreases": physical_pass,
    }


def main():
    if not all(path.exists() for path in (BASELINE_STATE, G9_STATE, NOTE79_RESULT)):
        raise FileNotFoundError("notes 77 and 79 state/result archives are required")
    note79 = json.loads(NOTE79_RESULT.read_text())
    if note79.get("status") != "review":
        raise RuntimeError("note-79 sealed REVIEW status must remain unchanged")
    fields, grids = load_fields()
    input_checks = {
        "note79_status_is_review": note79.get("status") == "review",
        "grid_sizes_match_protocol": all(
            (len(grids[label][0]), len(grids[label][1])) == EXPECTED_SIZES[label]
            for label in grids
        ),
        "all_grids_and_fields_finite": bool(
            all(np.all(np.isfinite(item)) for pair in grids.values() for item in pair)
            and all(
                np.all(np.isfinite(values))
                for variable in fields.values() for values in variable.values()
            )
        ),
    }
    analyses = {
        name: analyze_variable(values, grids) for name, values in fields.items()
    }
    variable_rules = {
        name: variable_rule_summary(record) for name, record in analyses.items()
    }
    acceptance = {
        "inputs_match_and_are_finite": all(input_checks.values()),
        "all_fits_and_fit_variants_are_robust": all(
            rules["fit_residuals_below_5_percent"]
            and rules["fit_robustness_below_5_percent"]
            for rules in variable_rules.values()
        ),
        "all_reconstructed_axes_converge": all(
            rules["all_reconstructed_axis_variants_have_positive_order"]
            for rules in variable_rules.values()
        ),
        "both_smoothed_profiles_converge_above_order_1": all(
            rules["smoothed_profile_order_above_1"]
            for rules in variable_rules.values()
        ),
        "both_physical_qd_comparisons_converge_above_order_1": all(
            rules["physical_qd_volume_order_above_1_and_euclidean_decreases"]
            for rules in variable_rules.values()
        ),
    }
    if all(acceptance.values()):
        status = "pass"
        classification = "near_axis_reconstruction_artifact_supported"
    elif any(
        not rules["all_reconstructed_axis_variants_have_positive_order"]
        and not rules["physical_qd_volume_order_above_1_and_euclidean_decreases"]
        for rules in variable_rules.values()
    ):
        status = "fail"
        classification = "genuine_d_nonconvergence_candidate"
    else:
        status = "review"
        classification = "mixed_near_axis_d_refinement_evidence"

    payload = {
        "status": status,
        "classification": classification,
        "scope": "sealed near-axis d-coefficient reconstruction and physical-contribution audit",
        "protocol": PROTOCOL,
        "note79_status_preserved": note79["status"],
        "field_index": FIELD_INDEX,
        "field_name": FIELD_ORDER[FIELD_INDEX],
        "fit_variants": [
            {"name": name, "degree_in_r_squared": degree, "window": window}
            for name, degree, window in FIT_VARIANTS
        ],
        "input_checks": input_checks,
        "analysis": analyses,
        "variable_rule_summary": variable_rules,
        "acceptance": acceptance,
        "limitations": [
            "three existing grids and one short-time slice only",
            "positive-radius polynomial reconstruction assumes even smoothness",
            "r^2 d norms are coordinate-component proxies rather than full invariant tensor norms",
            "does not alter the sealed note-79 REVIEW classification",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": status,
        "classification": classification,
        "acceptance": acceptance,
        "variable_rules": variable_rules,
        "key_metrics": {
            name: {
                "stored_axis": record["stored_axis_coefficient"],
                "reconstructed_axis_orders": {
                    variant: data["three_grid_difference"]["generalized_empirical_order"]
                    for variant, data in record["fit_variants"].items()
                },
                "smoothed_profile": record["smoothed_two_node_d_profile"],
                "physical_qd": record["physical_qd_equals_r_squared_d"],
            } for name, record in analyses.items()
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
