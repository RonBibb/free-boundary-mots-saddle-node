#!/usr/bin/env python3
"""Frozen local mode audit of the corrected-fold exterior lower-order block."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.regular_so3_gh_reduction import (
    FIELD_ORDER,
    regular_so3_gh_coefficient_matrices,
)
from run_corrected_fold_boundary_constraint_pulse import build_geometry


Z_VALUES = (1.0, np.sqrt(np.e), 2.2, 2.45, 2.6, np.e)
R_VALUES = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
DAMPING_RATES = (0.0, 4.0)
WAVENUMBERS = np.geomspace(0.1, 1e4, 61)
OUTPUT = Path("results/corrected_fold_lower_order_frozen_mode_audit.json")


def leading_components(vector, count=4):
    amplitude = np.abs(np.asarray(vector))
    amplitude /= max(float(np.linalg.norm(amplitude)), 1e-300)
    indices = np.argsort(amplitude)[::-1][:count]
    return [
        {"field": FIELD_ORDER[index], "fraction": float(amplitude[index])}
        for index in indices
    ]


def local_record(geometry, z_value, r_value, damping_rate):
    background = geometry["jet_field"].at(z_value, r_value)
    coefficients = regular_so3_gh_coefficient_matrices(
        background,
        r_value,
        mass_squared=geometry["mass_squared"],
        potential_offset=-6.0,
        constraint_damping=damping_rate,
    )
    reaction = coefficients["evolution_reaction_matrix"]
    first = coefficients["evolution_first_matrices"]
    time_first = first[0]
    compact_first = first[1]
    metric = np.asarray(background["metric"])
    compact_speed = float(np.sqrt(-metric[0, 0] / metric[1, 1]))

    eigenvalues, eigenvectors = np.linalg.eig(compact_first)
    high_index = int(np.argmax(np.abs(eigenvalues.real)))
    high_eigenvalue = eigenvalues[high_index]
    asymptotic_rate = float(abs(high_eigenvalue.real) / (2 * compact_speed))
    maximum_entry_index = np.unravel_index(
        np.argmax(np.abs(compact_first)), compact_first.shape,
    )
    _, singular_values, right_singular = np.linalg.svd(compact_first)

    identity = np.eye(len(FIELD_ORDER))
    best = None
    for wavenumber in WAVENUMBERS:
        lower_left = (
            -(compact_speed * wavenumber)**2 * identity
            - reaction
            + 1j * wavenumber * compact_first
        )
        generator = np.block([
            [np.zeros_like(identity), identity],
            [lower_left, time_first],
        ])
        values, vectors = np.linalg.eig(generator)
        index = int(np.argmax(values.real))
        candidate = {
            "growth_rate": float(values[index].real),
            "frequency": float(values[index].imag),
            "wavenumber": float(wavenumber),
            "position_components": leading_components(vectors[:9, index]),
            "velocity_components": leading_components(vectors[9:, index]),
        }
        if best is None or candidate["growth_rate"] > best["growth_rate"]:
            best = candidate

    physical_scale = np.diag((1.0, r_value, 1.0, 1.0, r_value**2, r_value, 1.0, 1.0, 1.0))
    scaled_compact = physical_scale @ compact_first @ np.linalg.inv(physical_scale)
    return {
        "z": float(z_value),
        "r": float(r_value),
        "damping_rate": float(damping_rate),
        "compact_coordinate_speed": compact_speed,
        "compact_first_frobenius_norm": float(np.linalg.norm(compact_first)),
        "compact_first_spectral_norm": float(singular_values[0]),
        "physically_scaled_compact_first_spectral_norm": float(np.linalg.norm(scaled_compact, 2)),
        "compact_first_spectral_abscissa": float(np.max(eigenvalues.real)),
        "compact_first_minimum_real_eigenvalue": float(np.min(eigenvalues.real)),
        "compact_first_high_frequency_growth_bound": asymptotic_rate,
        "dominant_high_frequency_eigenvalue": {
            "real": float(high_eigenvalue.real),
            "imaginary": float(high_eigenvalue.imag),
            "components": leading_components(eigenvectors[:, high_index]),
        },
        "maximum_absolute_entry": {
            "row": FIELD_ORDER[maximum_entry_index[0]],
            "column": FIELD_ORDER[maximum_entry_index[1]],
            "value": float(compact_first[maximum_entry_index]),
        },
        "leading_input_singular_vector_components": leading_components(
            right_singular[0].conj(),
        ),
        "frozen_full_z_scan": best,
    }


def main():
    geometry = build_geometry("G6")
    records = []
    for damping_rate in DAMPING_RATES:
        print(f"frozen modes kappa={damping_rate:g}", flush=True)
        for z_value in Z_VALUES:
            for r_value in R_VALUES:
                records.append(local_record(
                    geometry, z_value, r_value, damping_rate,
                ))
    summaries = []
    for damping_rate in DAMPING_RATES:
        selected = [item for item in records if item["damping_rate"] == damping_rate]
        by_cutoff = []
        for cutoff in (1.5, 2.0, 3.0, 4.0):
            interior = [item for item in selected if item["r"] <= cutoff]
            high = max(interior, key=lambda item: item["compact_first_high_frequency_growth_bound"])
            full = max(interior, key=lambda item: item["frozen_full_z_scan"]["growth_rate"])
            by_cutoff.append({
                "radial_cutoff": cutoff,
                "maximum_compact_high_frequency_growth_bound": high["compact_first_high_frequency_growth_bound"],
                "compact_high_frequency_growth_location": [high["z"], high["r"]],
                "maximum_frozen_full_z_growth": full["frozen_full_z_scan"]["growth_rate"],
                "frozen_full_z_growth_location": [full["z"], full["r"]],
                "frozen_full_z_growth_wavenumber": full["frozen_full_z_scan"]["wavenumber"],
            })
        maximum_high = max(
            selected, key=lambda item: item["compact_first_high_frequency_growth_bound"]
        )
        maximum_full = max(
            selected, key=lambda item: item["frozen_full_z_scan"]["growth_rate"]
        )
        summaries.append({
            "damping_rate": damping_rate,
            "by_radial_cutoff": by_cutoff,
            "maximum_compact_high_frequency_record": maximum_high,
            "maximum_frozen_full_z_record": maximum_full,
        })
    damped = next(item for item in summaries if item["damping_rate"] == 4.0)
    damped_bounds = [
        item["maximum_compact_high_frequency_growth_bound"]
        for item in damped["by_radial_cutoff"]
    ]
    acceptance = {
        "all_local_quantities_finite": bool(all(
            np.isfinite(item["compact_first_high_frequency_growth_bound"])
            and np.isfinite(item["frozen_full_z_scan"]["growth_rate"])
            for item in records
        )),
        "damped_high_frequency_bound_increases_from_r1p5_to_r4": bool(
            damped_bounds[-1] > 2 * damped_bounds[0]
        ),
        "damped_full_frozen_scan_has_positive_growth": bool(
            damped["maximum_frozen_full_z_record"]["frozen_full_z_scan"]["growth_rate"] > 0
        ),
    }
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "scope": "local frozen compact-direction lower-order dispersion audit on corrected G6",
        "field_order": list(FIELD_ORDER),
        "z_values": [float(value) for value in Z_VALUES],
        "r_values": [float(value) for value in R_VALUES],
        "wavenumbers": [float(value) for value in WAVENUMBERS],
        "records": records,
        "summaries": summaries,
        "acceptance": acceptance,
        "interpretation_rule": (
            "A finite high-wavenumber growth limit identifies strong but not unbounded lower-order continuum amplification; agreement with the time-domain radial trend distinguishes it from a time-integrator instability."
        ),
        "limitations": [
            "frozen local coefficients and compact-direction Fourier modes",
            "radial derivatives, walls, variable-coefficient transport, and constraints are omitted from the dispersion scan",
            "positive local growth does not by itself establish a global eigenmode",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "summaries": [{
            "damping_rate": item["damping_rate"],
            "by_radial_cutoff": item["by_radial_cutoff"],
            "maximum_compact": {
                key: item["maximum_compact_high_frequency_record"][key]
                for key in (
                    "z", "r", "compact_first_high_frequency_growth_bound",
                    "dominant_high_frequency_eigenvalue", "maximum_absolute_entry",
                )
            },
            "maximum_full": {
                "z": item["maximum_frozen_full_z_record"]["z"],
                "r": item["maximum_frozen_full_z_record"]["r"],
                **item["maximum_frozen_full_z_record"]["frozen_full_z_scan"],
            },
        } for item in summaries],
        "acceptance": acceptance,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
