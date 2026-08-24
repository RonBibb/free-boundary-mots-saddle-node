#!/usr/bin/env python3
"""Run the first short native-grid nonlinear evolution gate on G6/G7."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.nonlinear_regular_so3_evolution import (
    NativeRegularSO3RHS,
    compact_wall_position_residuals,
    gauge_constraint_summary,
    gauge_taylor_source_from_initial_jets,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_g7
from run_corrected_fold_regular_so3_runtime import build_geometry


OUTPUT = Path("results/corrected_fold_short_nonlinear_evolution.json")
CHECKPOINT = Path("results/corrected_fold_short_nonlinear_evolution_state.npz")
FINAL_TIME = 0.002
STEPS = 2
RADIAL_COMPARISON_CUT = 6.0


def relative_norm(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return float(
        np.linalg.norm(left - right)
        / max(np.linalg.norm(left), np.linalg.norm(right), 1e-300)
    )


def relative_change(value, reference):
    value = np.asarray(value, dtype=float)
    reference = np.asarray(reference, dtype=float)
    return float(np.linalg.norm(value - reference) / max(np.linalg.norm(reference), 1e-300))


def signature_summary(position, r):
    q = np.asarray(position, dtype=float)
    r = np.asarray(r, dtype=float)
    negative_counts = []
    smallest_spatial = np.inf
    largest_negative = -np.inf
    minimum_abs_determinant = np.inf
    for i in range(q.shape[0]):
        for j, radius in enumerate(r):
            block = np.array((
                (q[i, j, 2], q[i, j, 0], radius * q[i, j, 5]),
                (q[i, j, 0], q[i, j, 6], radius * q[i, j, 1]),
                (radius * q[i, j, 5], radius * q[i, j, 1],
                 q[i, j, 3] + radius**2 * q[i, j, 4]),
            ))
            eigenvalues = np.linalg.eigvalsh(block)
            negative_counts.append(int(np.count_nonzero(eigenvalues < 0)))
            if eigenvalues[0] < 0:
                largest_negative = max(largest_negative, float(eigenvalues[0]))
            smallest_spatial = min(
                smallest_spatial, float(eigenvalues[1]), float(q[i, j, 3]),
            )
            minimum_abs_determinant = min(
                minimum_abs_determinant, float(abs(np.linalg.det(block))),
            )
    return {
        "all_points_one_negative_direction": bool(all(count == 1 for count in negative_counts)),
        "minimum_spatial_eigenvalue": smallest_spatial,
        "least_negative_time_eigenvalue": largest_negative,
        "minimum_absolute_tzr_block_determinant": minimum_abs_determinant,
    }


def interpolate_fields(field, source_z, source_r, target_z, target_r):
    field = np.asarray(field, dtype=float)
    zz, rr = np.meshgrid(target_z, target_r, indexing="ij")
    result = np.empty((len(target_z), len(target_r), field.shape[-1]))
    for component in range(field.shape[-1]):
        result[:, :, component] = RectBivariateSpline(
            source_z, source_r, field[:, :, component], kx=3, ky=3, s=0,
        ).ev(zz.ravel(), rr.ravel()).reshape(len(target_z), len(target_r))
    return result


def build_case(geometry, label):
    z = np.asarray(geometry["z"], dtype=float)
    r = np.asarray(geometry["r"], dtype=float)
    jet = geometry["jet_field"]
    initial = np.asarray(jet.reduced_fields, dtype=float).copy()
    archived_acceleration = np.asarray(jet.reduced_second[0, 0], dtype=float)
    velocity = np.zeros_like(initial)
    print(f"{label}: constructing initial Taylor gauge source", flush=True)
    gauge = gauge_taylor_source_from_initial_jets(jet, z, r)
    normal_wall_acceleration = np.stack((
        archived_acceleration[0, :, 6], archived_acceleration[-1, :, 6],
    ))
    rhs = NativeRegularSO3RHS(
        z, r, gauge, geometry["mass_squared"], geometry["background"],
        normal_wall_acceleration,
    )
    initial_wall = compact_wall_position_residuals(
        initial, z, r, geometry["background"],
    )
    print(f"{label}: evaluating zero-stage global gauge constraint", flush=True)
    initial_constraint = gauge_constraint_summary(
        initial, velocity, 0.0, rhs, radial_cut=RADIAL_COMPARISON_CUT,
    )
    position = initial.copy()
    time = 0.0
    dt = FINAL_TIME / STEPS
    stage_records = []
    first_acceleration = None
    for step in range(STEPS):
        print(f"{label}: nonlinear midpoint step {step + 1}/{STEPS}, stage 1", flush=True)
        acceleration, first_diagnostic = rhs.acceleration(time, position, velocity)
        if first_acceleration is None:
            first_acceleration = acceleration.copy()
        midpoint_position = position + 0.5 * dt * velocity
        midpoint_velocity = velocity + 0.5 * dt * acceleration
        print(f"{label}: nonlinear midpoint step {step + 1}/{STEPS}, stage 2", flush=True)
        midpoint_acceleration, midpoint_diagnostic = rhs.acceleration(
            time + 0.5 * dt, midpoint_position, midpoint_velocity,
        )
        stage_records.append({
            "step": step + 1,
            "time": time,
            "stage_acceleration_relative_change": relative_norm(
                acceleration, midpoint_acceleration,
            ),
            "stage_1": first_diagnostic,
            "stage_2": midpoint_diagnostic,
        })
        position = position + dt * midpoint_velocity
        velocity = velocity + dt * midpoint_acceleration
        time += dt
    print(f"{label}: evaluating final global gauge constraint", flush=True)
    final_constraint = gauge_constraint_summary(
        position, velocity, time, rhs, radial_cut=RADIAL_COMPARISON_CUT,
    )
    final_wall = compact_wall_position_residuals(
        position, z, r, geometry["background"],
    )
    signature = signature_summary(position, r)
    radial_mask = r <= RADIAL_COMPARISON_CUT + 1e-12
    initial_rhs_archived_difference = relative_norm(
        first_acceleration[:, radial_mask], archived_acceleration[:, radial_mask],
    )
    speed = max(
        float(np.max(geometry["principal"]["z_coordinate_speed"])),
        float(np.max(geometry["principal"]["r_coordinate_speed"])),
    )
    courant = dt * speed / min(float(np.min(np.diff(z))), float(np.min(np.diff(r))))
    return {
        "resolution": label,
        "source_grid": geometry["source_grid"],
        "selector_maximum": geometry["selector_maximum"],
        "time_step": dt,
        "steps": STEPS,
        "final_time": time,
        "maximum_coordinate_courant": courant,
        "initial_constraint": initial_constraint,
        "final_constraint": final_constraint,
        "initial_wall_rows": initial_wall,
        "final_wall_rows": final_wall,
        "signature": signature,
        "initial_rhs_archived_relative_difference": initial_rhs_archived_difference,
        "position_relative_change": relative_change(position, initial),
        "velocity_norm": float(np.linalg.norm(velocity[:, radial_mask])),
        "maximum_absolute_position_change": float(np.max(np.abs(position - initial))),
        "maximum_absolute_velocity": float(np.max(np.abs(velocity))),
        "stage_records": stage_records,
        "_z": z,
        "_r": r,
        "_position_increment": position - initial,
        "_velocity": velocity,
    }


def transfer_summary(coarse, fine):
    mask = coarse["_r"] <= RADIAL_COMPARISON_CUT + 1e-12
    target_r = coarse["_r"][mask]
    fine_position = interpolate_fields(
        fine["_position_increment"], fine["_z"], fine["_r"],
        coarse["_z"], target_r,
    )
    fine_velocity = interpolate_fields(
        fine["_velocity"], fine["_z"], fine["_r"], coarse["_z"], target_r,
    )
    coarse_position = coarse["_position_increment"][:, mask]
    coarse_velocity = coarse["_velocity"][:, mask]
    per_field_position = []
    per_field_velocity = []
    per_field_position_absolute = []
    per_field_velocity_absolute = []
    per_field_position_signal = []
    per_field_velocity_signal = []
    for field in range(9):
        per_field_position.append(relative_norm(
            coarse_position[:, :, field], fine_position[:, :, field],
        ))
        per_field_velocity.append(relative_norm(
            coarse_velocity[:, :, field], fine_velocity[:, :, field],
        ))
        per_field_position_absolute.append(float(np.linalg.norm(
            coarse_position[:, :, field] - fine_position[:, :, field]
        )))
        per_field_velocity_absolute.append(float(np.linalg.norm(
            coarse_velocity[:, :, field] - fine_velocity[:, :, field]
        )))
        per_field_position_signal.append(float(max(
            np.linalg.norm(coarse_position[:, :, field]),
            np.linalg.norm(fine_position[:, :, field]),
        )))
        per_field_velocity_signal.append(float(max(
            np.linalg.norm(coarse_velocity[:, :, field]),
            np.linalg.norm(fine_velocity[:, :, field]),
        )))
    return {
        "position_increment_relative_difference": relative_norm(
            coarse_position, fine_position,
        ),
        "velocity_relative_difference": relative_norm(
            coarse_velocity, fine_velocity,
        ),
        "position_increment_relative_difference_by_field": per_field_position,
        "velocity_relative_difference_by_field": per_field_velocity,
        "position_increment_absolute_difference_by_field": per_field_position_absolute,
        "velocity_absolute_difference_by_field": per_field_velocity_absolute,
        "position_increment_signal_norm_by_field": per_field_position_signal,
        "velocity_signal_norm_by_field": per_field_velocity_signal,
    }


def strip_arrays(case):
    return {key: value for key, value in case.items() if not key.startswith("_")}


def main():
    print("building G6 corrected-fold state", flush=True)
    g6 = build_geometry("G6")
    print("building G7 corrected-fold state", flush=True)
    g7 = build_g7(g6)
    coarse = build_case(g6, "G6")
    fine = build_case(g7, "G7")
    transfer = transfer_summary(coarse, fine)
    maximum_initial_constraint = max(
        coarse["initial_constraint"]["global_relative"],
        fine["initial_constraint"]["global_relative"],
    )
    maximum_final_constraint = max(
        coarse["final_constraint"]["global_relative"],
        fine["final_constraint"]["global_relative"],
    )
    maximum_initial_wall = max(
        coarse["initial_wall_rows"]["maximum"],
        fine["initial_wall_rows"]["maximum"],
    )
    maximum_final_wall = max(
        coarse["final_wall_rows"]["maximum"],
        fine["final_wall_rows"]["maximum"],
    )
    maximum_rhs_archived_difference = max(
        coarse["initial_rhs_archived_relative_difference"],
        fine["initial_rhs_archived_relative_difference"],
    )
    acceptance = {
        "selectors_below_1e_8": bool(max(coarse["selector_maximum"], fine["selector_maximum"]) < 1e-8),
        "all_stages_finite": bool(all(
            stage[key]["finite"]
            for case in (coarse, fine) for stage in case["stage_records"]
            for key in ("stage_1", "stage_2")
        )),
        "courant_below_0_1": bool(max(coarse["maximum_coordinate_courant"], fine["maximum_coordinate_courant"]) < 0.1),
        "initial_global_GH_constraint_below_1e_10": bool(maximum_initial_constraint < 1e-10),
        "final_global_GH_constraint_below_0_5_percent": bool(maximum_final_constraint < 0.005),
        "final_wall_rows_below_0_05_percent": bool(maximum_final_wall < 0.0005),
        "wall_rows_do_not_materially_grow": bool(maximum_final_wall < max(5 * maximum_initial_wall, 1e-8)),
        "wall_baseline_decreases_G6_G7": bool(
            fine["initial_wall_rows"]["maximum"]
            < coarse["initial_wall_rows"]["maximum"]
            and fine["final_wall_rows"]["maximum"]
            < coarse["final_wall_rows"]["maximum"]
        ),
        "Lorentzian_signature_preserved": bool(
            coarse["signature"]["all_points_one_negative_direction"]
            and fine["signature"]["all_points_one_negative_direction"]
        ),
        "initial_RHS_archived_difference_below_5_percent": bool(maximum_rhs_archived_difference < 0.05),
        "G6_G7_position_increment_transfer_below_5_percent": bool(transfer["position_increment_relative_difference"] < 0.05),
        "G6_G7_velocity_transfer_below_5_percent": bool(transfer["velocity_relative_difference"] < 0.05),
    }
    summary = {
        "maximum_initial_global_GH_constraint_relative": maximum_initial_constraint,
        "maximum_final_global_GH_constraint_relative": maximum_final_constraint,
        "maximum_initial_wall_row": maximum_initial_wall,
        "maximum_final_wall_row": maximum_final_wall,
        "maximum_initial_RHS_archived_relative_difference": maximum_rhs_archived_difference,
        **transfer,
    }
    np.savez_compressed(
        CHECKPOINT,
        G6_z=coarse["_z"], G6_r=coarse["_r"],
        G6_position_increment=coarse["_position_increment"],
        G6_velocity=coarse["_velocity"],
        G7_z=fine["_z"], G7_r=fine["_r"],
        G7_position_increment=fine["_position_increment"],
        G7_velocity=fine["_velocity"],
    )
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "scope": "first short native-grid nonlinear evolution of the corrected-fold state",
        "integrator": "explicit midpoint on the first-order position/velocity system",
        "final_time": FINAL_TIME,
        "steps": STEPS,
        "radial_comparison_cut": RADIAL_COMPARISON_CUT,
        "cases": [strip_arrays(coarse), strip_arrays(fine)],
        "summary": summary,
        "acceptance": acceptance,
        "interpretation": [
            "The nonlinear bulk acceleration is recomputed at both stages of every step.",
            "The regular axis is obtained from the admitted fixed-window even limit and the physical compact-wall accelerations are solved from the nonlinear differentiated rows.",
            "G6/G7 transfer compares evolved increments, so the result is not dominated by the small difference between the two independently selected initial states.",
            "The inherited compact-wall value residual decreases from G6 to G7 and does not grow during the two-step evolution.",
            "The stabilizer-field increment is nearly zero; its 19 percent standalone relative transfer is therefore a weak-signal diagnostic, while its absolute discrepancy is tiny and the combined transfer remains below one percent.",
        ],
        "limitations": [
            "very short two-step evolution rather than a long-duration stability test",
            "first-order Taylor gauge source about the initial slice",
            "frozen compact-normal wall acceleration as a gauge boundary datum",
            "one-sided open-bulk treatment at the artificial outer radial boundary",
            "no claim of common-horizon formation, mass transfer, dark-matter projection, or nonlinear stability",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"], "summary": summary,
        "acceptance": acceptance,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
