#!/usr/bin/env python3
"""Isolate the source of the corrected-fold long-duration instability.

The mixed-pulse audit grows before the pulse can reach the artificial radial
boundary.  This runner independently switches the compact-wall Robin load and
the sampled covariant lower-order bulk terms on and off.  It therefore tests
whether the unstable mode belongs to the principal IBVP, the wall conditions,
or the retained lower-order operator.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.axisymmetric_reduced_wave_evolution import AxisymmetricVariableReducedWaveIBVP
from bhps.regular_so3_gh_reduction import FIELD_ORDER
from run_corrected_fold_boundary_constraint_pulse import (
    DAMPING_RATE,
    build_case,
    build_geometry,
    sample_coefficients,
)
from run_corrected_fold_free_constraint_pulse import sample_constraint_coefficients
from run_corrected_fold_gh_driver_runtime import sample_source_coefficients


FINAL_TIME = 0.8
GRID = (25, 37)
OUTPUT = Path("results/corrected_fold_long_duration_instability_isolation.json")


def principal_clone(wave, keep_robin):
    left = wave.left_robin if keep_robin else np.zeros_like(wave.left_robin)
    right = wave.right_robin if keep_robin else np.zeros_like(wave.right_robin)
    return AxisymmetricVariableReducedWaveIBVP(
        wave.z,
        wave.r,
        wave.mass_weight,
        wave.z_gradient_weight,
        wave.r_gradient_weight,
        left,
        right,
        wave.left_boundary_weight,
        wave.right_boundary_weight,
        dirichlet_fields=wave.dirichlet_fields,
        outer_dirichlet=False,
    )


def field_l2(wave, values):
    flat = np.asarray(values).reshape(wave.nodes, wave.field_count)
    return np.sqrt(np.maximum(0.0, np.sum((wave.mass @ flat) * flat, axis=0)))


def run_variant(wave, position, velocity, maximum_speed, label):
    spacing = min(np.min(np.diff(wave.z)), np.min(np.diff(wave.r)))
    courant = 0.026 / maximum_speed
    initial_field = field_l2(wave, position)

    def diagnostic(time, q, v):
        q_l2 = field_l2(wave, q)
        v_l2 = field_l2(wave, v)
        state = float(np.linalg.norm(np.hypot(q_l2, v_l2)))
        field_state = np.hypot(q_l2, v_l2)
        maximum_index = np.unravel_index(np.argmax(np.abs(q)), q.shape)
        return {
            "time": float(time),
            "state_l2": state,
            "field_state_l2": field_state.tolist(),
            "position_maximum": float(np.max(np.abs(q))),
            "position_maximum_index": [int(value) for value in maximum_index],
        }

    result = wave.integrate(
        position,
        velocity,
        FINAL_TIME,
        courant=courant,
        diagnostic=diagnostic,
    )
    records = result["diagnostics"]
    times = np.array([item["time"] for item in records])
    states = np.array([item["state_l2"] for item in records])
    fit_mask = (times >= 0.55) & (states > 0)
    growth_rate = float(np.polyfit(times[fit_mask], np.log(states[fit_mask]), 1)[0])
    final_field = np.array(records[-1]["field_state_l2"])
    dominant_field = int(np.argmax(final_field))
    initial_state = float(records[0]["state_l2"])
    thresholds = {}
    for factor in (10.0, 1e3, 1e6):
        indices = np.flatnonzero(states >= factor * initial_state)
        thresholds[f"time_to_{factor:g}_initial"] = (
            float(times[indices[0]]) if len(indices) else None
        )
    return {
        "label": label,
        "steps": result["steps"],
        "time_step": result["time_step"],
        "initial_state_l2": initial_state,
        "final_state_l2": float(states[-1]),
        "amplification": float(states[-1] / max(initial_state, 1e-300)),
        "late_logarithmic_growth_rate": growth_rate,
        "initial_position_l2_by_field": initial_field.tolist(),
        "final_state_l2_by_field": final_field.tolist(),
        "dominant_final_field_index": dominant_field,
        "dominant_final_field_fraction": float(
            final_field[dominant_field] / max(np.linalg.norm(final_field), 1e-300)
        ),
        "final_position_maximum_index": records[-1]["position_maximum_index"],
        "finite": bool(np.all(np.isfinite(states))),
        **thresholds,
        "diagnostics": records,
    }


def run_bulk_component_variants(wave, position, velocity, maximum_speed):
    """Run selected assembled lower-order pieces with the Robin load disabled."""
    saved_left = wave.left_robin.copy()
    saved_right = wave.right_robin.copy()
    saved_lower = dict(wave.coupled_lower_order)
    zero_lower = {key: matrix * 0.0 for key, matrix in saved_lower.items()}
    selections = (
        ("bulk_reaction_only", ("reaction",)),
        ("bulk_time_first_only", ("time_first",)),
        ("bulk_z_first_only", ("z_first",)),
        ("bulk_r_first_only", ("r_first",)),
        ("bulk_all_first_derivatives", ("time_first", "z_first", "r_first")),
        ("bulk_reaction_plus_spatial_first", ("reaction", "z_first", "r_first")),
    )
    records = []
    wave.left_robin[...] = 0.0
    wave.right_robin[...] = 0.0
    try:
        for label, active in selections:
            print(label.replace("_", " "), flush=True)
            wave.coupled_lower_order = {
                key: (saved_lower[key] if key in active else zero_lower[key])
                for key in saved_lower
            }
            records.append(run_variant(
                wave, position, velocity, maximum_speed, label
            ))
    finally:
        wave.left_robin[...] = saved_left
        wave.right_robin[...] = saved_right
        wave.coupled_lower_order = saved_lower
    return records


def main():
    geometry = build_geometry("G6")
    print(f"sampling kappa={DAMPING_RATE:g} coefficients", flush=True)
    wave_coefficients = sample_coefficients(geometry, constraint_damping=DAMPING_RATE)
    source_coefficients = sample_source_coefficients(geometry, DAMPING_RATE)
    constraint_coefficients = sample_constraint_coefficients(geometry)
    setup = build_case(
        geometry,
        wave_coefficients,
        source_coefficients,
        constraint_coefficients,
        *GRID,
        pulse_sector="mixed",
        r_center=3.45,
        r_half_width=0.20,
    )
    full = setup["wave"]
    maximum_speed = max(setup["maximum_radial_speed"], setup["maximum_compact_speed"])
    position = setup["position"]
    velocity = setup["velocity"]
    records = []

    print("full bulk + Robin walls", flush=True)
    records.append(run_variant(
        full, position, velocity, maximum_speed, "full_bulk_plus_robin"
    ))

    saved_left = full.left_robin.copy()
    saved_right = full.right_robin.copy()
    full.left_robin[...] = 0.0
    full.right_robin[...] = 0.0
    try:
        print("full bulk + zero Robin walls", flush=True)
        records.append(run_variant(
            full, position, velocity, maximum_speed, "full_bulk_zero_robin"
        ))
    finally:
        full.left_robin[...] = saved_left
        full.right_robin[...] = saved_right

    print("principal only + Robin walls", flush=True)
    principal_robin = principal_clone(full, True)
    records.append(run_variant(
        principal_robin,
        position,
        velocity,
        maximum_speed,
        "principal_only_plus_robin",
    ))

    print("principal only + zero Robin walls", flush=True)
    principal_zero = principal_clone(full, False)
    records.append(run_variant(
        principal_zero,
        position,
        velocity,
        maximum_speed,
        "principal_only_zero_robin",
    ))

    records.extend(run_bulk_component_variants(
        full, position, velocity, maximum_speed,
    ))

    by_label = {item["label"]: item for item in records}
    reference = by_label["principal_only_zero_robin"]["amplification"]
    acceptance = {
        "all_runs_finite": bool(all(item["finite"] for item in records)),
        "principal_zero_robin_bounded_below_10x": bool(reference < 10.0),
    }
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "scope": "operator isolation for the corrected-G6 mixed-pulse long-duration instability",
        "grid_size": list(GRID),
        "final_time": FINAL_TIME,
        "field_order": list(FIELD_ORDER),
        "radial_arrival_lower_bound": setup["radial_arrival_lower_bound"],
        "compact_arrival_lower_bound": setup["compact_arrival_lower_bound"],
        "records": records,
        "acceptance": acceptance,
        "limitations": [
            "single coarse grid used for rapid mechanism isolation",
            "linear fixed-background operator and one mixed initial pulse",
            "the artificial radial boundary is left uncontrolled in every variant",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "radial_arrival_lower_bound": payload["radial_arrival_lower_bound"],
        "compact_arrival_lower_bound": payload["compact_arrival_lower_bound"],
        "records": [{
            key: item[key] for key in (
                "label",
                "amplification",
                "late_logarithmic_growth_rate",
                "time_to_10_initial",
                "time_to_1000_initial",
                "time_to_1e+06_initial",
                "dominant_final_field_index",
                "dominant_final_field_fraction",
                "final_position_maximum_index",
            )
        } for item in records],
        "acceptance": acceptance,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
