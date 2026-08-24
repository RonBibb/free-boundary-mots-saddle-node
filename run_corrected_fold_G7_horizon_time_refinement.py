#!/usr/bin/env python3
"""Sealed G7 time refinement of short evolved spectral-cap motion."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.anisotropic_capped_surface import solve_anisotropic_capped_profile
from bhps.dynamical_capped_horizon import solve_spectral_dynamical_capped_surface
import run_corrected_fold_live_nonlinear_gauge_source as live
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import FINAL_TIME, relative_norm


OUTPUT = Path("results/corrected_fold_G7_horizon_time_refinement.json")
CHECKPOINT = Path("results/corrected_fold_G7_horizon_time_refinement_state.npz")
TIME_STEPS = (2, 4, 8)
MODES = 48
COLLOCATION = 257


def relative_scalar(left, right):
    return float(abs(left - right) / max(abs(left), abs(right), 1e-300))


def observed_rate(first, second):
    return float(np.log2(max(first, 1e-300) / max(second, 1e-300)))


def successive_differences(records, key):
    return [
        float(np.linalg.norm(records[index][key] - records[index + 1][key]))
        for index in range(2)
    ]


def static_cap(geometry):
    result = solve_anisotropic_capped_profile(
        geometry["z"], geometry["r"], geometry["psi"], geometry["a"],
        geometry["b"], geometry["c"], 1.53, tolerance=1e-8, nodes=220,
        max_nodes=12000,
    )
    if not result["converged"]:
        raise RuntimeError("initial outer cap failed")
    return result


def main():
    print("building corrected G7 A=7.94 state", flush=True)
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": 7.94}
    geometry = build_refined(
        seed, 81, 121, "G7A794-time", selector_iterations=40,
        slice_iterations=270,
    )
    initial = static_cap(geometry)
    case = live.setup_case(
        geometry, "G7-time", live_normal_wall_gauge=True,
        live_outer_sommerfeld=True,
    )

    runs = []
    surfaces = []
    for steps in TIME_STEPS:
        print(f"evolving G7 with {steps} steps", flush=True)
        live.STEPS = steps
        run = live.integrate(case)
        print(f"solving {MODES}-mode final cap for {steps} steps", flush=True)
        surface = solve_spectral_dynamical_capped_surface(
            run["_position"], run["_velocity"], case["z"], case["r"],
            initial, tolerance=5e-4, collocation_nodes=COLLOCATION,
            cosine_modes=MODES, maximum_evaluations=200,
        )
        runs.append(run)
        surfaces.append(surface)

    field_differences = {}
    field_rates = {}
    for name, key in (
        ("position_increment", "_increment"),
        ("velocity", "_velocity"),
        ("source_increment", "_source_increment"),
    ):
        differences = successive_differences(runs, key)
        field_differences[name] = differences
        field_rates[name] = observed_rate(*differences)

    profile_differences = [
        float(np.linalg.norm(surfaces[index]["rho"] - surfaces[index + 1]["rho"]))
        for index in range(2)
    ]
    profile_rate = observed_rate(*profile_differences)
    initial_on = np.interp(surfaces[-1]["theta"], initial["theta"], initial["rho"])
    fine_profile_transfer = relative_norm(surfaces[1]["rho"], surfaces[2]["rho"])
    fine_radius_transfer = {
        name: relative_scalar(surfaces[1][name], surfaces[2][name])
        for name in ("rho_axis", "rho_brane")
    }
    fine_displacement_transfer = relative_norm(
        surfaces[1]["rho"] - initial_on, surfaces[2]["rho"] - initial_on,
    )
    changes = {
        str(steps): {
            name: float((surface[name] - initial[name]) / initial[name])
            for name in ("rho_axis", "rho_brane")
        }
        for steps, surface in zip(TIME_STEPS, surfaces)
    }

    acceptance = {
        "all_stages_finite_and_Lorentzian": bool(all(
            run["all_stages_finite"]
            and run["signature"]["all_points_one_negative_direction"]
            for run in runs
        )),
        "constraint_wall_and_boundary_rows_pass": bool(
            max(run["final_constraint"]["global_relative"] for run in runs) < .005
            and max(max(
                run["final_wall"]["maximum"],
                run["final_normal_wall_position_residual"]["maximum"],
            ) for run in runs) < .0005
            and max(max(
                run["maximum_normal_wall_acceleration_residual"],
                run["maximum_outer_acceleration_residual"],
                run["maximum_outer_source_residual"],
                run["final_outer_sommerfeld_position_residual"]["maximum_normalized"],
                run["final_outer_source_sommerfeld_residual"]["maximum_normalized"],
            ) for run in runs) < 1e-10
        ),
        "spacetime_and_source_time_rates_at_least_1_5": bool(
            min(field_rates.values()) >= 1.5
        ),
        "all_spectral_surfaces_converge_below_5e_4": bool(all(
            surface["converged"] and surface["in_domain"]
            and surface["interior_expansion_maximum"] < 5e-4
            for surface in surfaces
        )),
        "horizon_profile_time_rate_at_least_1_5": bool(profile_rate >= 1.5),
        "fine_horizon_profile_radii_and_displacement_transfer": bool(
            max(fine_profile_transfer, *fine_radius_transfer.values()) < .0005
            and fine_displacement_transfer < .01
        ),
        "all_motion_directions_positive_and_nonzero": bool(
            all(value > 0 for record in changes.values() for value in record.values())
            and max(value for record in changes.values() for value in record.values()) > .001
        ),
    }

    summary = {
        "field_successive_differences": field_differences,
        "field_time_refinement_rates": field_rates,
        "horizon_profile_successive_differences": profile_differences,
        "horizon_profile_time_refinement_rate": profile_rate,
        "fine_4_8_profile_transfer": fine_profile_transfer,
        "fine_4_8_radius_transfer": fine_radius_transfer,
        "fine_4_8_displacement_transfer": fine_displacement_transfer,
        "fractional_radius_changes": changes,
        "expansion_residuals": {
            str(steps): surface["interior_expansion_maximum"]
            for steps, surface in zip(TIME_STEPS, surfaces)
        },
        "final_global_GH_constraints": {
            str(steps): run["final_constraint"]["global_relative"]
            for steps, run in zip(TIME_STEPS, runs)
        },
    }
    np.savez_compressed(
        CHECKPOINT, z=case["z"], r=case["r"], horizon_theta=surfaces[-1]["theta"],
        initial_horizon=initial_on,
        **{
            f"steps_{steps}_{name}": value
            for steps, run, surface in zip(TIME_STEPS, runs, surfaces)
            for name, value in (
                ("increment", run["_increment"]),
                ("velocity", run["_velocity"]),
                ("source_increment", run["_source_increment"]),
                ("horizon", surface["rho"]),
            )
        },
    )
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "scope": "sealed G7 time-step refinement of short evolved spectral-cap motion",
        "protocol": "notes/65_G7_horizon_time_refinement_protocol.md",
        "final_time": FINAL_TIME,
        "time_steps": list(TIME_STEPS),
        "cosine_modes": MODES,
        "collocation_nodes": COLLOCATION,
        "driver_parameters": {"mu": live.DRIVER_MU, "eta": live.DRIVER_ETA},
        "target_parameters": {
            "mu_lapse": live.TARGET_MU_LAPSE,
            "mu_shift": live.TARGET_MU_SHIFT,
            "determinant_power": live.TARGET_POWER,
        },
        "summary": summary,
        "acceptance": acceptance,
        "limitations": [
            "t=0.002 persistence of a pre-existing cap",
            "single G7 spacetime grid and fixed 48-mode horizon representation",
            "not formation, event-horizon location, topology change, long-time stability, branch selection, an open basin, or mass transfer",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
