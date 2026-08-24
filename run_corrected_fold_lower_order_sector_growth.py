#!/usr/bin/env python3
"""Classify the lower-order mesh growth by initial characteristic sector."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.constraint_ibvp import (
    evaluate_regular_so3_constraint_field,
    regular_so3_metric_characteristic_projector_matrices,
)
from bhps.regular_so3_gh_reduction import FIELD_ORDER
from run_corrected_fold_boundary_constraint_pulse import (
    AMPLITUDE,
    DAMPING_RATE,
    build_geometry,
    outgoing_initial_pulse,
    sample_coefficients,
)
from run_corrected_fold_free_constraint_pulse import (
    interpolate_constraint_coefficients,
    sample_constraint_coefficients,
)
from run_corrected_fold_regular_so3_runtime import sampled_system


GRID = (25, 37)
FINAL_TIME = 0.8
OUTPUT = Path("results/corrected_fold_lower_order_sector_growth.json")


def field_l2(wave, values):
    flat = np.asarray(values).reshape(wave.nodes, wave.field_count)
    return np.sqrt(np.maximum(0.0, np.sum((wave.mass @ flat) * flat, axis=0)))


def constraint_l2(wave, constraint):
    flat = np.asarray(constraint).reshape(wave.nodes, -1)
    return float(np.sqrt(max(0.0, np.sum(wave.lumped_mass[:, None] * flat**2))))


def derivative_state_norm(wave, q, v):
    flat_q = np.asarray(q).reshape(wave.nodes, wave.field_count)
    flat_v = np.asarray(v).reshape(wave.nodes, wave.field_count)
    value = np.sum((wave.mass @ flat_v) * flat_v)
    value += np.sum((wave.stiffness @ flat_q) * flat_q)
    return float(np.sqrt(max(0.0, value)))


def sector_basis(geometry, wave, sector, seed):
    metrics = np.asarray([
        geometry["jet_field"].at(z_value, wave.r[-1])["metric"]
        for z_value in wave.z
    ])
    basis = np.empty((wave.nz, 7))
    for index, metric in enumerate(metrics):
        projectors = regular_so3_metric_characteristic_projector_matrices(
            metric, wave.r[-1], "radial", 1.0,
        )
        basis[index] = seed if sector == "mixed" else projectors[sector] @ seed
    basis /= max(np.max(np.linalg.norm(basis, axis=1)), 1e-300)
    return basis


def run_sector(wave, principal, constraint_zero, constraint_first, position, velocity, label):
    saved_left = wave.left_robin.copy()
    saved_right = wave.right_robin.copy()
    wave.left_robin[...] = 0.0
    wave.right_robin[...] = 0.0
    spacing = min(np.min(np.diff(wave.z)), np.min(np.diff(wave.r)))
    maximum_speed = float(max(
        np.max(principal["r_speed"]), np.max(principal["z_speed"])
    ))
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
    q = result["position"]
    v = result["velocity"]
    source = np.zeros((wave.nz, wave.nr, 3))
    initial_constraint = evaluate_regular_so3_constraint_field(
        wave.z,
        wave.r,
        position,
        velocity,
        source,
        constraint_zero,
        constraint_first,
        5,
        radial_first_is_scaled=True,
    )["constraint"]
    final_constraint = evaluate_regular_so3_constraint_field(
        wave.z,
        wave.r,
        q,
        v,
        source,
        constraint_zero,
        constraint_first,
        5,
        radial_first_is_scaled=True,
    )["constraint"]
    times = np.array([item[0] for item in history])
    states = np.array([item[1] for item in history])
    mask = (times >= 0.55) & (states > 0)
    growth = float(np.polyfit(times[mask], np.log(states[mask]), 1)[0])
    final_fields = np.hypot(field_l2(wave, q), field_l2(wave, v))
    dominant = int(np.argmax(final_fields))
    initial_constraint_norm = constraint_l2(wave, initial_constraint)
    final_constraint_norm = constraint_l2(wave, final_constraint)
    initial_derivative_norm = derivative_state_norm(wave, position, velocity)
    final_derivative_norm = derivative_state_norm(wave, q, v)
    return {
        "initial_sector": label,
        "steps": result["steps"],
        "time_step": result["time_step"],
        "initial_state_l2": float(states[0]),
        "final_state_l2": float(states[-1]),
        "amplification": float(states[-1] / max(states[0], 1e-300)),
        "late_logarithmic_growth_rate": growth,
        "dominant_final_field_index": dominant,
        "dominant_final_field": FIELD_ORDER[dominant],
        "dominant_final_field_fraction": float(
            final_fields[dominant] / max(np.linalg.norm(final_fields), 1e-300)
        ),
        "initial_constraint_l2": initial_constraint_norm,
        "final_constraint_l2": final_constraint_norm,
        "constraint_amplification": float(
            final_constraint_norm / max(initial_constraint_norm, 1e-300)
        ),
        "initial_constraint_to_derivative_state_ratio": float(
            initial_constraint_norm / max(initial_derivative_norm, 1e-300)
        ),
        "final_constraint_to_derivative_state_ratio": float(
            final_constraint_norm / max(final_derivative_norm, 1e-300)
        ),
    }


def main():
    geometry = build_geometry("G6")
    print(f"sampling kappa={DAMPING_RATE:g} wave coefficients", flush=True)
    coefficients = sample_coefficients(geometry, constraint_damping=DAMPING_RATE)
    print("sampling constraint diagnostic coefficients", flush=True)
    constraint_coefficients = sample_constraint_coefficients(geometry)
    wave, principal, _, _, _, _ = sampled_system(
        geometry, coefficients, *GRID, r_max=4.0, outer_dirichlet=False,
    )
    constraint_zero, constraint_first = interpolate_constraint_coefficients(
        constraint_coefficients, wave.z, wave.r,
    )
    seed = np.array((0.31, -0.22, 0.17, -0.29, 0.13, 0.23, -0.19))
    records = []
    for sector in ("gauge", "constraint", "physical", "mixed"):
        print(f"initial {sector} sector", flush=True)
        basis = sector_basis(geometry, wave, sector, seed)
        position, velocity = outgoing_initial_pulse(
            wave.z,
            wave.r,
            principal["r_speed"],
            basis,
            r_center=3.45,
            r_half_width=0.20,
        )
        records.append(run_sector(
            wave,
            principal,
            constraint_zero,
            constraint_first,
            position,
            velocity,
            sector,
        ))
    physical = next(item for item in records if item["initial_sector"] == "physical")
    acceptance = {
        "all_runs_finite": bool(all(np.isfinite(item["final_state_l2"]) for item in records)),
        "physical_initial_pulse_remains_below_10x": bool(physical["amplification"] < 10.0),
        "physical_final_constraint_ratio_below_0p1": bool(
            physical["final_constraint_to_derivative_state_ratio"] < 0.1
        ),
    }
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "scope": "characteristic-sector excitation of the corrected-G6 lower-order mesh mode",
        "grid_size": list(GRID),
        "final_time": FINAL_TIME,
        "field_order": list(FIELD_ORDER),
        "records": records,
        "acceptance": acceptance,
        "interpretation_rule": (
            "Growth from every seed with a large final constraint ratio indicates sector leakage into a constraint-violating mode; bounded physical data would support a constraint/gauge-only diagnosis."
        ),
        "limitations": [
            "sector projectors are defined at the artificial radial face and seed an interior radial pulse",
            "single screening grid and linear fixed background",
            "the compact-wall Robin load is disabled to isolate the bulk operator",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "records": records,
        "acceptance": acceptance,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
