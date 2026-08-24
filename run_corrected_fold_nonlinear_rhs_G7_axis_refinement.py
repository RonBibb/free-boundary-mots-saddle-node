#!/usr/bin/env python3
"""Refine the regular nonlinear acceleration axis limit from G6 to G7."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.anisotropic_geometry import anisotropic_metric_acceleration, anisotropic_scalar_acceleration
from bhps.anisotropic_initial_data import solve_anisotropic_initial_data
from bhps.axisymmetric_reduced_wave_evolution import axisymmetric_principal_coefficients
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice
from bhps.generalized_harmonic_jets import spatial_metric_acceleration_trace
from bhps.lapse_acceleration_corner import construct_localized_target_lapse_acceleration_completion
from bhps.physical_corner_corrector import combine_shape_modes, tracefree_shape_basis
from bhps.regular_so3_gh_reduction import RegularSO3BackgroundJetField
from bhps.scalar_pulse import scalar_pulse
from run_corrected_fold_nonlinear_rhs_axis_regularization import (
    FIELD_ORDER,
    MATCHED_R,
    QUOTIENT_POWERS,
    Z_FRACTIONS,
    relative_norm,
    scaled_field_differences,
    solve_raw,
)
from run_corrected_fold_regular_so3_runtime import build_geometry


AXIS_WINDOWS = (0.5, 2.0 / 3.0)
AXIS_DEGREE = 3
OUTPUT = Path("results/corrected_fold_nonlinear_rhs_G7_axis_refinement.json")


def interpolate(field, source_z, source_r, target_z, target_r):
    zz, rr = np.meshgrid(target_z, target_r, indexing="ij")
    return RectBivariateSpline(source_z, source_r, np.asarray(field), kx=3, ky=3, s=0).ev(
        zz.ravel(), rr.ravel()
    ).reshape(len(target_z), len(target_r))


def build_refined(coarse,nz,nr,name,selector_iterations=40,slice_iterations=260):
    """Construct one refined corrected-fold slice from a coarser initial guess."""
    nz=int(nz);nr=int(nr)
    amplitude = float(coarse["fold_amplitude"])
    reference = solve_finite_wall_high_order_slice(
        amplitude, nz=nz, nr=nr, r_max=8.0, wall_stiffness=20.0,
        epsilon=0.1, backreaction=0.01, tolerance=1e-10, iterations=int(slice_iterations),
    )
    chi, chi_r, chi_z = scalar_pulse(reference["z"], reference["r"], amplitude)
    archive = np.load("results/corrected_family_knot_A8_state.npz")
    coefficients = archive["coefficients"]
    modes = tracefree_shape_basis(
        reference["z"], reference["r"], 6, (0.5, 1.0), 8.0,
        ((7.5, 1.5), (7.5, 3.0)),
    )["modes"]
    a, b, c = combine_shape_modes(coefficients, modes)
    coarse_q = 1 / np.asarray(coarse["psi"]) - np.asarray(coarse["z"])[:, None]
    initial_q = interpolate(
        coarse_q, coarse["z"], coarse["r"], reference["z"], reference["r"],
    )
    initial_phi = interpolate(
        coarse["phi"], coarse["z"], coarse["r"], reference["z"], reference["r"],
    )
    selected = solve_anisotropic_initial_data(
        reference["z"], reference["r"], reference["q"], reference["phi"],
        a, b, c, reference["background"], chi_r, chi_z,
        initial_q=initial_q, initial_phi=initial_phi, stencil_width=7,
        tolerance=1e-9, iterations=int(selector_iterations),
    )
    z = reference["z"]
    r = reference["r"]
    psi = 1 / (z[:, None] + selected["q"])
    phi = selected["phi"]
    mass = float(reference["background"]["mass_squared"])
    acceleration = anisotropic_metric_acceleration(
        z, r, psi, a, b, c, phi, chi_r, chi_z, mass, chi=chi,
        stencil_width=7, lapse=psi,
    )
    phi_tt = anisotropic_scalar_acceleration(
        z, r, psi, a, b, c, phi, mass, lapse=psi, stencil_width=7,
    )
    chi_tt = anisotropic_scalar_acceleration(
        z, r, psi, a, b, c, chi, 0.0, lapse=psi, stencil_width=7,
    )
    trace = spatial_metric_acceleration_trace(acceleration, psi, a, b, c)
    completion = construct_localized_target_lapse_acceleration_completion(
        z, acceleration, psi, psi, a, phi, reference["background"],
        phi_tt, 0.5 * trace, 0.15,
    )
    return {
        "name": str(name),
        "source_grid": [nz, nr],
        "fold_amplitude": amplitude,
        "selector_maximum": float(selected["maximum_residual"]),
        "z": z,
        "r": r,
        "psi": psi,
        "a": a,
        "b": b,
        "c": c,
        "phi": phi,
        "background": reference["background"],
        "mass_squared": mass,
        "principal": axisymmetric_principal_coefficients(psi, a, b, c),
        "jet_field": RegularSO3BackgroundJetField(
            z, r, psi, psi, a, b, c, phi, chi, acceleration,
            completion["lapse_acceleration"], phi_tt, chi_tt, 7,
        ),
    }


def build_g7(coarse):
    return build_refined(coarse,81,121,"G7R8")


def build_g8(coarse):
    return build_refined(
        coarse,97,145,"G8R8",selector_iterations=45,slice_iterations=280,
    )


def axis_limit(positive_r, regular_positive, window):
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
        value, count = axis_limit(positive_r, regular_positive, window)
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


def assemble_case(geometry, label):
    z0 = float(geometry["z"][0])
    z1 = float(geometry["z"][-1])
    return {
        "resolution": label,
        "source_grid": geometry["source_grid"],
        "selector_maximum": geometry["selector_maximum"],
        "lines": [
            assemble_line(geometry, z0 + fraction * (z1 - z0))
            for fraction in Z_FRACTIONS
        ],
    }


def main():
    g6 = build_geometry("G6")
    g7 = build_g7(g6)
    cases = [assemble_case(g6, "G6"), assemble_case(g7, "G7")]
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
        "G6_G7_matched_transfer_below_5_percent": bool(matched_transfer < 0.05),
        "G6_G7_axis_transfer_below_5_percent": bool(axis_transfer < 0.05),
        "axis_window_robustness_below_5_percent": bool(maximum_window_difference < 0.05),
    }
    summary = {
        "matched_regular_G6_G7_relative_difference": matched_transfer,
        "axis_G6_G7_relative_difference": axis_transfer,
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
        "scope": "G6-to-G7 refinement of the fixed-physical-window regular SO(3) nonlinear acceleration axis limit",
        "field_order": list(FIELD_ORDER),
        "z_fractions": list(Z_FRACTIONS),
        "matched_r": list(MATCHED_R),
        "axis_windows": list(AXIS_WINDOWS),
        "axis_polynomial_degree": AXIS_DEGREE,
        "cases": cases,
        "summary": summary,
        "acceptance": acceptance,
        "limitations": [
            "G7 uses the G6 fold amplitude and an interpolated G6 initial guess",
            "axis coefficient is inferred from a radial limit rather than a differentiated axis PDE",
            "fixed-stage initial gauge-source jets",
            "nonlinear compact-wall rows and finite-time evolution remain open",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "summary": summary, "acceptance": acceptance}, indent=2), flush=True)


if __name__ == "__main__":
    main()
