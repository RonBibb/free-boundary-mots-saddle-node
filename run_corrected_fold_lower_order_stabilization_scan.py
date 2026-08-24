#!/usr/bin/env python3
"""Scan consistent mesh-scaled velocity diffusion against lower-order growth."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.regular_so3_gh_reduction import FIELD_ORDER
from run_corrected_fold_boundary_constraint_pulse import (
    AMPLITUDE,
    DAMPING_RATE,
    build_geometry,
    outgoing_initial_pulse,
    sample_coefficients,
)
from run_corrected_fold_regular_so3_runtime import sampled_system


FINAL_TIME = 0.8
GRIDS = ((25, 37), (33, 49))
SCREEN_PARAMETERS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
OUTPUT = Path("results/corrected_fold_lower_order_stabilization_scan.json")


def field_l2(wave, values):
    flat = np.asarray(values).reshape(wave.nodes, wave.field_count)
    return np.sqrt(np.maximum(0.0, np.sum((wave.mass @ flat) * flat, axis=0)))


def run_case(wave, position, velocity, maximum_speed, active_lower, parameter, label):
    saved_left = wave.left_robin.copy()
    saved_right = wave.right_robin.copy()
    saved_lower = dict(wave.coupled_lower_order)
    saved_diffusion = wave.velocity_diffusion
    zero_lower = {key: matrix * 0.0 for key, matrix in saved_lower.items()}
    spacing = min(np.min(np.diff(wave.z)), np.min(np.diff(wave.r)))
    wave.left_robin[...] = 0.0
    wave.right_robin[...] = 0.0
    wave.coupled_lower_order = {
        key: (saved_lower[key] if key in active_lower else zero_lower[key])
        for key in saved_lower
    }
    wave.velocity_diffusion = float(parameter) * spacing
    history = []

    def diagnostic(time, q, v):
        q_l2 = field_l2(wave, q)
        v_l2 = field_l2(wave, v)
        history.append((float(time), float(np.linalg.norm(np.hypot(q_l2, v_l2)))))
        return None

    try:
        result = wave.integrate(
            position,
            velocity,
            FINAL_TIME,
            courant=0.026 / maximum_speed,
            diagnostic=diagnostic,
        )
    finally:
        wave.left_robin[...] = saved_left
        wave.right_robin[...] = saved_right
        wave.coupled_lower_order = saved_lower
        wave.velocity_diffusion = saved_diffusion
    times = np.array([item[0] for item in history])
    states = np.array([item[1] for item in history])
    mask = (times >= 0.55) & (states > 0)
    growth = float(np.polyfit(times[mask], np.log(states[mask]), 1)[0])
    final_fields = np.hypot(
        field_l2(wave, result["position"]), field_l2(wave, result["velocity"])
    )
    dominant = int(np.argmax(final_fields))
    return {
        "record": {
            "label": label,
            "mesh_diffusion_parameter": float(parameter),
            "velocity_diffusion_length": float(parameter * spacing),
            "steps": result["steps"],
            "time_step": result["time_step"],
            "initial_state_l2": float(states[0]),
            "final_state_l2": float(states[-1]),
            "maximum_state_l2": float(np.max(states)),
            "amplification": float(states[-1] / max(states[0], 1e-300)),
            "maximum_amplification": float(np.max(states) / max(states[0], 1e-300)),
            "late_logarithmic_growth_rate": growth,
            "dominant_final_field": FIELD_ORDER[dominant],
            "dominant_final_field_fraction": float(
                final_fields[dominant] / max(np.linalg.norm(final_fields), 1e-300)
            ),
            "finite": bool(np.all(np.isfinite(states))),
        },
        "position": result["position"],
        "velocity": result["velocity"],
    }


def initial_data(geometry, coefficients, grid):
    wave, principal, _, _, _, _ = sampled_system(
        geometry, coefficients, *grid, r_max=4.0, outer_dirichlet=False,
    )
    seed = np.array((0.31, -0.22, 0.17, -0.29, 0.13, 0.23, -0.19))
    seed /= np.linalg.norm(seed)
    basis = np.broadcast_to(seed, (wave.nz, 7)).copy()
    position, velocity = outgoing_initial_pulse(
        wave.z,
        wave.r,
        principal["r_speed"],
        basis,
        r_center=3.45,
        r_half_width=0.20,
    )
    maximum_speed = float(max(
        np.max(principal["r_speed"]), np.max(principal["z_speed"])
    ))
    return wave, position, velocity, maximum_speed


def state_difference(wave, left, right):
    difference = np.hypot(
        wave.l2_norm(left["position"] - right["position"]),
        wave.l2_norm(left["velocity"] - right["velocity"]),
    )
    scale = np.hypot(
        wave.l2_norm(right["position"]), wave.l2_norm(right["velocity"]),
    )
    return float(difference / max(scale, 1e-300))


def main():
    geometry = build_geometry("G6")
    print(f"sampling kappa={DAMPING_RATE:g} wave coefficients", flush=True)
    coefficients = sample_coefficients(geometry, constraint_damping=DAMPING_RATE)
    full_lower = ("reaction", "time_first", "z_first", "r_first")

    coarse_wave, coarse_q, coarse_v, coarse_speed = initial_data(
        geometry, coefficients, GRIDS[0]
    )
    coarse_scan = []
    selected = None
    for parameter in SCREEN_PARAMETERS:
        print(f"coarse full bulk epsilon={parameter:g}", flush=True)
        result = run_case(
            coarse_wave, coarse_q, coarse_v, coarse_speed,
            full_lower, parameter, "full_bulk_zero_robin",
        )
        coarse_scan.append(result["record"])
        if selected is None and parameter > 0 and result["record"]["maximum_amplification"] < 10.0:
            selected = float(parameter)
    if selected is None:
        selected = float(SCREEN_PARAMETERS[-1])

    fine_wave, fine_q, fine_v, fine_speed = initial_data(
        geometry, coefficients, GRIDS[1]
    )
    fine_parameters = tuple(dict.fromkeys((0.0, selected, min(2 * selected, SCREEN_PARAMETERS[-1]))))
    fine_scan = []
    for parameter in fine_parameters:
        print(f"fine full bulk epsilon={parameter:g}", flush=True)
        fine_scan.append(run_case(
            fine_wave, fine_q, fine_v, fine_speed,
            full_lower, parameter, "full_bulk_zero_robin",
        )["record"])
    stable_candidates = [
        item["mesh_diffusion_parameter"] for item in fine_scan
        if item["mesh_diffusion_parameter"] > 0 and item["maximum_amplification"] < 10.0
    ]
    selected_fine = min(stable_candidates) if stable_candidates else max(fine_parameters)

    fidelity = []
    for wave, q, v, speed, grid in (
        (coarse_wave, coarse_q, coarse_v, coarse_speed, GRIDS[0]),
        (fine_wave, fine_q, fine_v, fine_speed, GRIDS[1]),
    ):
        print(f"principal fidelity grid {grid[0]} x {grid[1]}", flush=True)
        reference = run_case(
            wave, q, v, speed, (), 0.0, "principal_reference",
        )
        stabilized = run_case(
            wave, q, v, speed, (), selected_fine, "principal_stabilized",
        )
        fidelity.append({
            "grid_size": list(grid),
            "selected_parameter": selected_fine,
            "relative_final_state_difference": state_difference(
                wave, stabilized, reference,
            ),
            "reference": reference["record"],
            "stabilized": stabilized["record"],
        })

    selected_records = [
        next(item for item in scan if item["mesh_diffusion_parameter"] == selected_fine)
        for scan in (coarse_scan, fine_scan)
    ]
    acceptance = {
        "selected_parameter_found_on_fine_grid": bool(stable_candidates),
        "selected_full_runs_finite": bool(all(item["finite"] for item in selected_records)),
        "selected_full_maximum_amplification_below_10x": bool(all(
            item["maximum_amplification"] < 10.0 for item in selected_records
        )),
        "selected_full_late_growth_nonpositive": bool(all(
            item["late_logarithmic_growth_rate"] <= 0.0 for item in selected_records
        )),
        "principal_final_state_change_below_10_percent": bool(max(
            item["relative_final_state_difference"] for item in fidelity
        ) < 0.10),
    }
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "scope": "mesh-consistent velocity-diffusion stabilization of corrected-G6 lower-order growth",
        "field_order": list(FIELD_ORDER),
        "pulse_amplitude": AMPLITUDE,
        "final_time": FINAL_TIME,
        "grids": [list(grid) for grid in GRIDS],
        "screen_parameters": list(SCREEN_PARAMETERS),
        "selected_parameter": selected_fine,
        "coarse_full_scan": coarse_scan,
        "fine_full_scan": fine_scan,
        "principal_fidelity": fidelity,
        "acceptance": acceptance,
        "interpretation_rule": (
            "The stabilization is admissible only if one mesh-scaled parameter bounds both grids while its effect on the resolved principal pulse decreases and remains below the sealed fidelity threshold."
        ),
        "limitations": [
            "linear fixed-background pulse and two screening grids",
            "velocity diffusion is a numerical regulator, not a physical term",
            "a successful screen would still require refined convergence and constraint-preserving tests",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "selected_parameter": selected_fine,
        "coarse_full_scan": coarse_scan,
        "fine_full_scan": fine_scan,
        "principal_fidelity": [{
            "grid_size": item["grid_size"],
            "relative_final_state_difference": item["relative_final_state_difference"],
            "reference_amplification": item["reference"]["amplification"],
            "stabilized_amplification": item["stabilized"]["amplification"],
        } for item in fidelity],
        "acceptance": acceptance,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
