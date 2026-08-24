#!/usr/bin/env python3
"""Locate the radial onset of corrected-fold lower-order mesh growth."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from run_corrected_fold_boundary_constraint_pulse import (
    AMPLITUDE,
    DAMPING_RATE,
    build_geometry,
    outgoing_initial_pulse,
    sample_coefficients,
)
from run_corrected_fold_lower_order_growth_refinement import (
    coefficient_summary,
    run_variant,
)
from run_corrected_fold_regular_so3_runtime import sampled_system


NZ = 25
DOMAINS = ((1.5, 15), (2.0, 19), (3.0, 28), (4.0, 37))
OUTPUT = Path("results/corrected_fold_lower_order_domain_scan.json")


def main():
    geometry = build_geometry("G6")
    print(f"sampling kappa={DAMPING_RATE:g} wave coefficients", flush=True)
    coefficients = sample_coefficients(geometry, constraint_damping=DAMPING_RATE)
    seed = np.array((0.31, -0.22, 0.17, -0.29, 0.13, 0.23, -0.19))
    seed /= np.linalg.norm(seed)
    records = []
    for r_max, nr in DOMAINS:
        print(f"domain r_max={r_max:g}, grid {NZ} x {nr}", flush=True)
        wave, principal, _, first, _, _ = sampled_system(
            geometry, coefficients, NZ, nr, r_max=r_max, outer_dirichlet=False,
        )
        basis = np.broadcast_to(seed, (wave.nz, 7)).copy()
        position, velocity = outgoing_initial_pulse(
            wave.z,
            wave.r,
            principal["r_speed"],
            basis,
            r_center=r_max - 0.55,
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
            "radial_domain_maximum": r_max,
            "grid_size": [NZ, nr],
            "radial_spacing": float(np.max(np.diff(wave.r))),
            "pulse_center": r_max - 0.55,
            "coefficient_summary": coefficient_summary(first),
            "variants": variants,
        })
    compact_growth = [
        next(item for item in record["variants"] if item["label"] == "compact_first_only_zero_robin")["late_logarithmic_growth_rate"]
        for record in records
    ]
    full_growth = [
        next(item for item in record["variants"] if item["label"] == "full_bulk_zero_robin")["late_logarithmic_growth_rate"]
        for record in records
    ]
    interior_full = [
        next(item for item in record["variants"] if item["label"] == "full_bulk_zero_robin")
        for record in records if record["radial_domain_maximum"] <= 2.0
    ]
    acceptance = {
        "matched_radial_spacings_within_5_percent": bool(
            max(item["radial_spacing"] for item in records)
            / min(item["radial_spacing"] for item in records) < 1.05
        ),
        "interior_full_runs_bounded_below_10x": bool(all(
            item["amplification"] < 10.0
            for item in interior_full
        )),
        "compact_growth_increases_with_domain": bool(all(
            compact_growth[index + 1] > compact_growth[index]
            for index in range(len(compact_growth) - 1)
        )),
        "full_growth_increases_with_domain": bool(all(
            full_growth[index + 1] > full_growth[index]
            for index in range(len(full_growth) - 1)
        )),
    }
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "scope": "matched-spacing radial-domain localization of corrected-G6 lower-order mesh growth",
        "damping_rate": DAMPING_RATE,
        "pulse_amplitude": AMPLITUDE,
        "records": records,
        "compact_first_growth_rates": compact_growth,
        "full_bulk_growth_rates": full_growth,
        "acceptance": acceptance,
        "interpretation_rule": (
            "Bounded matched-spacing interior runs with increasing exterior growth identify a large-radius coefficient/variable-conditioning problem rather than a global principal instability."
        ),
        "limitations": [
            "one compact resolution and matched but not identical radial spacings",
            "the pulse is translated with the outer face, so this is a local exterior stress test",
            "linear fixed background and disabled compact-wall Robin load",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "compact_first_growth_rates": compact_growth,
        "full_bulk_growth_rates": full_growth,
        "records": [{
            "radial_domain_maximum": item["radial_domain_maximum"],
            "grid_size": item["grid_size"],
            "radial_spacing": item["radial_spacing"],
            "compact_coefficient": item["coefficient_summary"]["compact"],
            "variants": [{
                key: variant[key] for key in (
                    "label", "amplification", "late_logarithmic_growth_rate",
                    "dominant_field", "position_maximum_coordinates",
                    "effective_gradient_length_scale_in_minimum_grid_spacings",
                )
            } for variant in item["variants"]],
        } for item in records],
        "acceptance": acceptance,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
