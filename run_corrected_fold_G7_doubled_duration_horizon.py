#!/usr/bin/env python3
"""Sealed doubled-duration G7 evolution with live spectral-cap tracking."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.anisotropic_capped_surface import solve_anisotropic_capped_profile
from bhps.dynamical_capped_horizon import solve_spectral_dynamical_capped_surface
from bhps.nonlinear_regular_so3_evolution import regular_so3_outward_radial_speed
import run_corrected_fold_live_nonlinear_gauge_source as live
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import relative_norm


OUTPUT = Path("results/corrected_fold_G7_doubled_duration_horizon.json")
CHECKPOINT = Path("results/corrected_fold_G7_doubled_duration_horizon_state.npz")
BASELINE = Path("results/corrected_fold_G7_horizon_time_refinement_state.npz")
FINAL_TIME = 0.004
STEPS = 8
HALFWAY_STEP = 4
MODES = (48, 56)
COLLOCATION = 257


def relative_scalar(left, right):
    return float(abs(left - right) / max(abs(left), abs(right), 1e-300))


def static_cap(geometry):
    result = solve_anisotropic_capped_profile(
        geometry["z"], geometry["r"], geometry["psi"], geometry["a"],
        geometry["b"], geometry["c"], 1.53, tolerance=1e-8, nodes=220,
        max_nodes=12000,
    )
    if not result["converged"]:
        raise RuntimeError("initial outer cap failed")
    return result


def spectral_pair(position, velocity, case, initial):
    return [
        solve_spectral_dynamical_capped_surface(
            position, velocity, case["z"], case["r"], initial,
            tolerance=5e-4, collocation_nodes=COLLOCATION,
            cosine_modes=modes, maximum_evaluations=200,
        )
        for modes in MODES
    ]


def angular_transfer(pair, initial_on):
    coarse, fine = pair
    return {
        "profile_relative_difference": relative_norm(coarse["rho"], fine["rho"]),
        "axis_relative_difference": relative_scalar(coarse["rho_axis"], fine["rho_axis"]),
        "brane_relative_difference": relative_scalar(coarse["rho_brane"], fine["rho_brane"]),
        "displacement_relative_difference": relative_norm(
            coarse["rho"] - initial_on, fine["rho"] - initial_on,
        ),
    }


def main():
    if not BASELINE.exists():
        raise FileNotFoundError("note-65 time-refinement checkpoint is required")
    baseline = np.load(BASELINE)
    print("building corrected G7 A=7.94 state", flush=True)
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": 7.94}
    geometry = build_refined(
        seed, 81, 121, "G7A794-duration", selector_iterations=40,
        slice_iterations=270,
    )
    initial = static_cap(geometry)
    case = live.setup_case(
        geometry, "G7-duration", live_normal_wall_gauge=True,
        live_outer_sommerfeld=True,
    )
    live.FINAL_TIME = FINAL_TIME
    live.STEPS = STEPS
    print("evolving to t=0.004 with an actual t=0.002 checkpoint", flush=True)
    run = live.integrate(case, checkpoint_steps=(HALFWAY_STEP,))
    halfway = run["_checkpoints"][HALFWAY_STEP]

    print("solving halfway 48/56-mode caps", flush=True)
    halfway_surfaces = spectral_pair(
        halfway["_position"], halfway["_velocity"], case, initial,
    )
    print("solving final 48/56-mode caps", flush=True)
    final_surfaces = spectral_pair(run["_position"], run["_velocity"], case, initial)
    initial_on = np.interp(
        final_surfaces[-1]["theta"], initial["theta"], initial["rho"],
    )

    baseline_reproduction = {
        "position_increment": relative_norm(
            halfway["_increment"], baseline["steps_4_increment"],
        ),
        "velocity": relative_norm(
            halfway["_velocity"], baseline["steps_4_velocity"],
        ),
        "source_increment": relative_norm(
            halfway["_source_increment"], baseline["steps_4_source_increment"],
        ),
        "horizon_48_mode": relative_norm(
            halfway_surfaces[0]["rho"], baseline["steps_4_horizon"],
        ),
    }
    angular = {
        "t_0_002": angular_transfer(halfway_surfaces, initial_on),
        "t_0_004": angular_transfer(final_surfaces, initial_on),
    }
    selected_halfway = halfway_surfaces[-1]
    selected_final = final_surfaces[-1]
    halfway_displacement = selected_halfway["rho"] - initial_on
    final_displacement = selected_final["rho"] - initial_on
    displacement_ratio = float(
        np.linalg.norm(final_displacement)
        / max(np.linalg.norm(halfway_displacement), 1e-300)
    )
    selected_changes = {
        "t_0_002": {
            name: float((selected_halfway[name] - initial[name]) / initial[name])
            for name in ("rho_axis", "rho_brane")
        },
        "t_0_004": {
            name: float((selected_final[name] - initial[name]) / initial[name])
            for name in ("rho_axis", "rho_brane")
        },
    }
    initial_speed = regular_so3_outward_radial_speed(case["initial"], case["r"])
    final_speed = regular_so3_outward_radial_speed(run["_position"], case["r"])
    corrections = {
        "metric": run["maximum_outer_metric_correction"],
        "scalar": run["maximum_outer_scalar_correction"],
        "source": run["maximum_outer_source_correction"],
    }
    row_residuals = {
        "normal_wall_acceleration": run["maximum_normal_wall_acceleration_residual"],
        "outer_acceleration": run["maximum_outer_acceleration_residual"],
        "outer_source_acceleration": run["maximum_outer_source_residual"],
        "final_outer_position": run["final_outer_sommerfeld_position_residual"]["maximum_normalized"],
        "final_outer_source": run["final_outer_source_sommerfeld_residual"]["maximum_normalized"],
    }

    acceptance = {
        "halfway_reproduces_note_65_below_1e_8": bool(
            max(baseline_reproduction.values()) < 1e-8
        ),
        "finite_signature_constraint_and_walls_pass": bool(
            run["all_stages_finite"]
            and run["signature"]["all_points_one_negative_direction"]
            and run["final_constraint"]["global_relative"] < .005
            and max(
                run["final_wall"]["maximum"],
                run["final_normal_wall_position_residual"]["maximum"],
            ) < .0005
        ),
        "outer_reference_and_rows_pass": bool(
            np.all(np.isfinite(initial_speed)) and np.all(initial_speed > 0)
            and np.all(np.isfinite(final_speed)) and np.all(final_speed > 0)
            and max(row_residuals.values()) < 1e-10
            and max(corrections.values()) > 1e-8
            and max(corrections.values()) < .05
        ),
        "all_spectral_surfaces_converge_below_5e_4": bool(all(
            surface["converged"] and surface["in_domain"]
            and surface["interior_expansion_maximum"] < 5e-4
            for pair in (halfway_surfaces, final_surfaces) for surface in pair
        )),
        "both_angular_transfers_pass": bool(
            max(
                max(
                    record["profile_relative_difference"],
                    record["axis_relative_difference"],
                    record["brane_relative_difference"],
                )
                for record in angular.values()
            ) < .0005
            and max(
                record["displacement_relative_difference"]
                for record in angular.values()
            ) < .01
        ),
        "selected_cap_continues_outward": bool(
            selected_final["rho_axis"] > selected_halfway["rho_axis"] > initial["rho_axis"]
            and selected_final["rho_brane"] > selected_halfway["rho_brane"] > initial["rho_brane"]
            and max(selected_changes["t_0_004"].values()) > .001
        ),
        "total_displacement_ratio_between_2_and_6": bool(
            2.0 < displacement_ratio < 6.0
        ),
    }
    summary = {
        "halfway_note_65_reproduction": baseline_reproduction,
        "angular_transfer": angular,
        "selected_fractional_radius_changes": selected_changes,
        "selected_total_displacement_ratio": displacement_ratio,
        "expansion_residuals": {
            label: {
                str(modes): surface["interior_expansion_maximum"]
                for modes, surface in zip(MODES, pair)
            }
            for label, pair in (
                ("t_0_002", halfway_surfaces), ("t_0_004", final_surfaces),
            )
        },
        "initial_outward_speed_range": [float(np.min(initial_speed)), float(np.max(initial_speed))],
        "final_outward_speed_range": [float(np.min(final_speed)), float(np.max(final_speed))],
        "maximum_outer_corrections": corrections,
        "maximum_boundary_row_residuals": row_residuals,
        "final_global_GH_constraint": run["final_constraint"]["global_relative"],
        "final_wall_position_residual": run["final_wall"]["maximum"],
        "final_normal_wall_position_residual": run["final_normal_wall_position_residual"]["maximum"],
    }
    np.savez_compressed(
        CHECKPOINT, z=case["z"], r=case["r"],
        halfway_increment=halfway["_increment"],
        halfway_velocity=halfway["_velocity"],
        final_increment=run["_increment"], final_velocity=run["_velocity"],
        horizon_theta=selected_final["theta"], initial_horizon=initial_on,
        halfway_horizon_48=halfway_surfaces[0]["rho"],
        halfway_horizon_56=selected_halfway["rho"],
        final_horizon_48=final_surfaces[0]["rho"],
        final_horizon_56=selected_final["rho"],
    )
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "scope": "sealed doubled-duration G7 live evolution with halfway and final smooth marginal-cap tracking",
        "protocol": "notes/66_G7_doubled_duration_horizon_protocol.md",
        "final_time": FINAL_TIME,
        "steps": STEPS,
        "time_step": FINAL_TIME / STEPS,
        "cosine_modes": list(MODES),
        "collocation_nodes": COLLOCATION,
        "summary": summary,
        "acceptance": acceptance,
        "limitations": [
            "persistence and motion of a pre-existing cap",
            "single G7 spacetime grid",
            "quadratic outer reference admitted only through t=0.004",
            "not formation, event-horizon location, topology change, long-time stability, branch selection, an open basin, or mass transfer",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
