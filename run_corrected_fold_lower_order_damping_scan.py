#!/usr/bin/env python3
"""Test whether GH constraint damping drives the long-duration mesh mode."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from run_corrected_fold_boundary_constraint_pulse import (
    AMPLITUDE,
    build_geometry,
    outgoing_initial_pulse,
    sample_coefficients,
)
from run_corrected_fold_lower_order_growth_refinement import (
    coefficient_summary,
    run_variant,
)
from run_corrected_fold_regular_so3_runtime import sampled_system


GRID = (25, 37)
DAMPING_RATES = (0.0, 1.0, 2.0, 4.0)
OUTPUT = Path("results/corrected_fold_lower_order_damping_scan.json")


def affine_coefficients(undamped, unit, rate):
    rate = float(rate)
    result = dict(undamped)
    result["reaction"] = undamped["reaction"] + rate * (
        unit["reaction"] - undamped["reaction"]
    )
    result["first"] = undamped["first"] + rate * (
        unit["first"] - undamped["first"]
    )
    result["constraint_damping_rate"] = rate
    return result


def main():
    geometry = build_geometry("G6")
    print("sampling undamped coefficients", flush=True)
    undamped = sample_coefficients(geometry, constraint_damping=0.0)
    print("sampling unit-damped coefficients", flush=True)
    unit = sample_coefficients(geometry, constraint_damping=1.0)
    seed = np.array((0.31, -0.22, 0.17, -0.29, 0.13, 0.23, -0.19))
    seed /= np.linalg.norm(seed)
    records = []
    for rate in DAMPING_RATES:
        print(f"damping rate {rate:g}", flush=True)
        coefficients = affine_coefficients(undamped, unit, rate)
        wave, principal, _, first, _, _ = sampled_system(
            geometry, coefficients, *GRID, r_max=4.0, outer_dirichlet=False,
        )
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
        variants = []
        for label, active in (
            ("compact_first_only_zero_robin", ("z_first",)),
            ("full_bulk_zero_robin", ("reaction", "time_first", "z_first", "r_first")),
        ):
            print(f"  {label}", flush=True)
            item = run_variant(
                wave, position, velocity, maximum_speed, active, label,
            )
            item.pop("normalized_dominant_position_profile")
            variants.append(item)
        records.append({
            "constraint_damping_rate": rate,
            "coefficient_summary": coefficient_summary(first),
            "variants": variants,
        })

    def selected(label):
        return [
            next(item for item in record["variants"] if item["label"] == label)
            for record in records
        ]

    compact = selected("compact_first_only_zero_robin")
    full = selected("full_bulk_zero_robin")
    compact_growth = np.array([item["late_logarithmic_growth_rate"] for item in compact])
    full_growth = np.array([item["late_logarithmic_growth_rate"] for item in full])
    acceptance = {
        "undamped_compact_growth_below_kappa4": bool(compact_growth[0] < compact_growth[-1]),
        "undamped_full_growth_below_kappa4": bool(full_growth[0] < full_growth[-1]),
        "all_results_finite": bool(all(
            np.isfinite(item["final_state_l2"])
            for record in records for item in record["variants"]
        )),
    }
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "scope": "constraint-damping dependence of the corrected-G6 lower-order mesh-scale growth",
        "grid_size": list(GRID),
        "pulse_amplitude": AMPLITUDE,
        "damping_rates": list(DAMPING_RATES),
        "records": records,
        "compact_first_growth_rates": compact_growth.tolist(),
        "full_bulk_growth_rates": full_growth.tolist(),
        "acceptance": acceptance,
        "interpretation_rule": (
            "A large monotone growth-rate increase with kappa identifies the damping addition as a driver; persistence at kappa=0 identifies the undamped background lower-order operator as an independent source."
        ),
        "limitations": [
            "single screening grid",
            "linear fixed-background evolution and a mixed pulse",
            "mesh-scale growth is a numerical/formulation warning rather than a resolved physical mode",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "compact_first_growth_rates": payload["compact_first_growth_rates"],
        "full_bulk_growth_rates": payload["full_bulk_growth_rates"],
        "summaries": [{
            "kappa": record["constraint_damping_rate"],
            "compact_first_coefficient": record["coefficient_summary"]["compact"],
            "variants": [{
                key: item[key] for key in (
                    "label", "amplification", "late_logarithmic_growth_rate",
                    "dominant_field", "effective_gradient_length_scale",
                    "effective_gradient_length_scale_in_minimum_grid_spacings",
                )
            } for item in record["variants"]],
        } for record in records],
        "acceptance": acceptance,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
