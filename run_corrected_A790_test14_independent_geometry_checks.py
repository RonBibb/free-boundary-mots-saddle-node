#!/usr/bin/env python3
"""Independent physical checks for the sealed Test-14 charge history."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.interpolate import CubicSpline, RectBivariateSpline

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.dynamical_capped_horizon import (
    capped_outgoing_expansion,
    prepare_capped_expansion_slice,
)
from bhps.recovery_indexer import atomic_write_json, sha256_file
from bhps.test14_quasilocal_charge import reflected_cap_charge
from run_corrected_A790_blind_horizon_detector_engineering import solve_seed
from run_corrected_A790_surface_geometry_history import (
    archived_slice,
    selections_for,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry


PROTOCOL = Path("notes/95_A790_quasilocal_mass_flux_bridge_protocol.md")
PRIMARY = Path("results/corrected_A790_test14_quasilocal_charge_history.json")
OUTPUT = Path("results/corrected_A790_test14_independent_geometry_checks.json")
FINE_RESULT = Path("results/corrected_A790_formation_time_refinement.json")
FINE_STATE = Path("results/corrected_A790_formation_time_refinement_state.npz")
LONG_RESULT = Path("results/corrected_A790_two_grid_formation_search.json")
LONG_STATE = Path("results/corrected_A790_two_grid_formation_search_state.npz")
FINAL_RESULT = Path("results/corrected_A790_t004_discrepancy_formation_confirmation.json")
TIMES = (0.000625, 0.001, 0.002, 0.003, 0.004)


def relative_difference(left, right):
    return float(abs(left - right) / max(abs(left), abs(right), 1e-300))


def independent_intrinsic_integral(position, z, r, profile, prepared):
    theta = np.asarray(profile["theta"], dtype=float)
    rho = np.asarray(profile["rho"], dtype=float)
    slope = np.asarray(profile["slope"], dtype=float)
    zcoord = float(z[-1]) - rho * np.cos(theta)
    radius = rho * np.sin(theta)
    tangent = np.stack((
        rho * np.sin(theta) - slope * np.cos(theta),
        rho * np.cos(theta) + slope * np.sin(theta),
    ), axis=1)
    metric = np.empty((len(theta), 2, 2))
    for left in range(2):
        for right in range(2):
            metric[:, left, right] = prepared.sample(
                ("base_metric", left, right), zcoord, radius,
            )
    speed = np.sqrt(np.einsum(
        "...a,...ab,...b->...", tangent, metric, tangent,
    ))
    transverse_spline = RectBivariateSpline(
        z, r, position[:, :, 3], kx=min(3, len(z) - 1),
        ky=min(3, len(r) - 1), s=0,
    )
    transverse = transverse_spline.ev(zcoord, radius)
    transverse_theta = (
        transverse_spline.ev(zcoord, radius, dx=1, dy=0) * tangent[:, 0]
        + transverse_spline.ev(zcoord, radius, dx=0, dy=1) * tangent[:, 1]
    )
    root = np.sqrt(transverse)
    sphere_radius = radius * root
    sphere_radius_theta = (
        tangent[:, 1] * root + 0.5 * radius * transverse_theta / root
    )
    sphere_radius_slope = sphere_radius_theta / speed

    sampled_s = cumulative_trapezoid(speed, theta, initial=0.0)
    missing_axis_length = float(speed[0] * theta[0])
    proper_s = np.concatenate(([0.0], missing_axis_length + sampled_s))
    proper_w = np.concatenate(([0.0], sphere_radius))
    curve = CubicSpline(
        proper_s, proper_w,
        bc_type=((1, 1.0), (1, float(sphere_radius_slope[-1]))),
    )
    dense_s = np.linspace(0.0, proper_s[-1], 8001)
    dense_w = curve(dense_s)
    dense_first = curve(dense_s, 1)
    dense_second = curve(dense_s, 2)
    one_sided_bulk = float(simpson(
        4.0 * math.pi * (
            -4.0 * dense_w * dense_second
            + 2.0 * (1.0 - dense_first**2)
        ), x=dense_s,
    ))
    seam = float(
        32.0 * math.pi * sphere_radius[-1] * sphere_radius_slope[-1]
    )
    direct_completed = 2.0 * one_sided_bulk + seam
    first_derivative_identity = float(simpson(
        16.0 * math.pi * (1.0 + sphere_radius_slope**2) * speed,
        x=theta,
    ))
    return {
        "direct_doubled_bulk_integral": 2.0 * one_sided_bulk,
        "direct_seam_integral": seam,
        "direct_completed_integral": direct_completed,
        "first_derivative_identity_integral": first_derivative_identity,
        "relative_difference": relative_difference(
            direct_completed, first_derivative_identity,
        ),
    }


def main():
    primary = json.loads(PRIMARY.read_text())
    if primary["protocol_sha256"] != sha256_file(PROTOCOL):
        raise RuntimeError("primary result and protocol hash disagree")
    fine_result = json.loads(FINE_RESULT.read_text())
    long_result = json.loads(LONG_RESULT.read_text())
    final_result = json.loads(FINAL_RESULT.read_text())
    fine_state = np.load(FINE_STATE)
    long_state = np.load(LONG_STATE)

    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": 7.90}
    geometries = {
        "G7": build_refined(
            seed, 81, 121, "G7A790-test14-independent",
            selector_iterations=40, slice_iterations=270,
        ),
    }
    geometries["G8"] = build_refined(
        geometries["G7"], 97, 145, "G8A790-test14-independent",
        selector_iterations=45, slice_iterations=280,
    )
    records = []
    for current_time in TIMES:
        for label, geometry in geometries.items():
            position, velocity = archived_slice(
                label, current_time, geometry, fine_state, long_state,
            )
            prepared = prepare_capped_expansion_slice(
                position, velocity, geometry["z"], geometry["r"],
            )
            selections = selections_for(
                label, current_time, fine_result, long_result, final_result,
            )
            for branch_index, branch in enumerate(("inner", "outer")):
                selection = selections[branch_index]
                modes = selection["confirmation_modes"]
                print(
                    f"{label} t={current_time:.6f} {branch}: {modes} modes",
                    flush=True,
                )
                surface = solve_seed(
                    position, velocity, geometry, selection["seed"], modes,
                    prepared,
                )
                if "error" in surface or not surface.get("converged"):
                    raise RuntimeError(surface)
                expansion = capped_outgoing_expansion(
                    position, velocity, geometry["z"], geometry["r"],
                    surface, prepared=prepared,
                )
                ingoing_raw = (
                    -expansion["mean_curvature"]
                    + expansion["extrinsic_curvature_correction"]
                )
                interior = expansion["two_cell_interior_mask"]
                intrinsic = independent_intrinsic_integral(
                    position, geometry["z"], geometry["r"], surface,
                    prepared,
                )
                charge = reflected_cap_charge(
                    position, velocity, geometry["z"], geometry["r"],
                    surface, prepared=prepared,
                )
                records.append({
                    "grid": label, "time": current_time, "branch": branch,
                    "cosine_modes": int(modes),
                    "maximum_raw_ingoing_expansion_interior": float(
                        np.max(ingoing_raw[interior])
                    ),
                    "minimum_raw_ingoing_expansion_interior": float(
                        np.min(ingoing_raw[interior])
                    ),
                    "ingoing_expansion_negative_everywhere_interior": bool(
                        np.max(ingoing_raw[interior]) < 0.0
                    ),
                    "intrinsic_curvature_crosscheck": intrinsic,
                    "primary_intrinsic_scalar_integral": charge[
                        "intrinsic_scalar_curvature_integral"
                    ],
                    "crosscheck_to_primary_relative_difference": (
                        relative_difference(
                            intrinsic["direct_completed_integral"],
                            charge["intrinsic_scalar_curvature_integral"],
                        )
                    ),
                })

    payload = {
        "status": "PASS" if all(
            record["ingoing_expansion_negative_everywhere_interior"]
            for record in records
        ) else "REVIEW",
        "classification": "independent_test14_geometry_and_future_MOTS_checks",
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "primary_result": str(PRIMARY),
        "primary_result_sha256": sha256_file(PRIMARY),
        "records": records,
        "summary": {
            "all_ingoing_expansions_negative": bool(all(
                record["ingoing_expansion_negative_everywhere_interior"]
                for record in records
            )),
            "maximum_intrinsic_crosscheck_relative_difference": max(
                record["crosscheck_to_primary_relative_difference"]
                for record in records
            ),
            "largest_raw_ingoing_expansion": max(
                record["maximum_raw_ingoing_expansion_interior"]
                for record in records
            ),
        },
        "claim_boundary": (
            "Post-seal independent consistency audit; it does not supply the "
            "missing horizon-tube evolution field or Israel brane flux."
        ),
    }
    atomic_write_json(OUTPUT, payload)
    print(json.dumps({
        "status": payload["status"], "summary": payload["summary"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

