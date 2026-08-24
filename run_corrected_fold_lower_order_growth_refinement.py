#!/usr/bin/env python3
"""Refinement and profile audit of the corrected-fold lower-order growth."""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

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
GRIDS = ((25, 37), (33, 49), (41, 61))
OUTPUT = Path("results/corrected_fold_lower_order_growth_refinement.json")


def field_l2(wave, values):
    flat = np.asarray(values).reshape(wave.nodes, wave.field_count)
    return np.sqrt(np.maximum(0.0, np.sum((wave.mass @ flat) * flat, axis=0)))


def coefficient_summary(first):
    result = {}
    for direction, name in enumerate(("time", "compact", "radial_scaled")):
        matrices = first[direction]
        norms = np.linalg.norm(matrices, axis=(-2, -1))
        maximum_index = np.unravel_index(np.argmax(norms), norms.shape)
        local = matrices[maximum_index]
        eigenvalues = np.linalg.eigvals(local)
        numerical = np.linalg.eigvalsh(0.5 * (local + local.T))
        result[name] = {
            "maximum_frobenius_norm": float(norms[maximum_index]),
            "maximum_location_index": [int(value) for value in maximum_index],
            "maximum_location_matrix_spectral_norm": float(np.linalg.norm(local, 2)),
            "maximum_location_spectral_abscissa": float(np.max(eigenvalues.real)),
            "maximum_location_largest_eigenvalue_imaginary_part": float(
                np.max(np.abs(eigenvalues.imag))
            ),
            "maximum_location_numerical_abscissa": float(np.max(numerical)),
        }
    return result


def run_variant(wave, position, velocity, maximum_speed, active_lower, label):
    saved_left = wave.left_robin.copy()
    saved_right = wave.right_robin.copy()
    saved_lower = dict(wave.coupled_lower_order)
    zero_lower = {key: matrix * 0.0 for key, matrix in saved_lower.items()}
    wave.left_robin[...] = 0.0
    wave.right_robin[...] = 0.0
    wave.coupled_lower_order = {
        key: (saved_lower[key] if key in active_lower else zero_lower[key])
        for key in saved_lower
    }
    spacing = min(np.min(np.diff(wave.z)), np.min(np.diff(wave.r)))
    records = []

    def diagnostic(time, q, v):
        q_l2 = field_l2(wave, q)
        v_l2 = field_l2(wave, v)
        records.append((float(time), float(np.linalg.norm(np.hypot(q_l2, v_l2)))))
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
    times = np.array([item[0] for item in records])
    states = np.array([item[1] for item in records])
    mask = (times >= 0.55) & (states > 0)
    growth = float(np.polyfit(times[mask], np.log(states[mask]), 1)[0])
    final_q = result["position"]
    final_v = result["velocity"]
    final_fields = np.hypot(field_l2(wave, final_q), field_l2(wave, final_v))
    dominant = int(np.argmax(final_fields))
    q_profile = final_q[:, :, dominant]
    q_norm = float(np.sqrt(max(0.0, q_profile.ravel() @ (wave.mass @ q_profile.ravel()))))
    gradient_norm = float(np.sqrt(max(
        0.0, q_profile.ravel() @ (wave.stiffness @ q_profile.ravel())
    )))
    normalized_profile = q_profile / max(q_norm, 1e-300)
    maximum_index = np.unravel_index(np.argmax(np.abs(q_profile)), q_profile.shape)
    return {
        "label": label,
        "steps": result["steps"],
        "time_step": result["time_step"],
        "initial_state_l2": float(states[0]),
        "final_state_l2": float(states[-1]),
        "amplification": float(states[-1] / max(states[0], 1e-300)),
        "late_logarithmic_growth_rate": growth,
        "dominant_field_index": dominant,
        "dominant_field": FIELD_ORDER[dominant],
        "dominant_field_fraction": float(
            final_fields[dominant] / max(np.linalg.norm(final_fields), 1e-300)
        ),
        "position_maximum_index": [int(value) for value in maximum_index],
        "position_maximum_coordinates": [
            float(wave.z[maximum_index[0]]), float(wave.r[maximum_index[1]])
        ],
        "effective_gradient_wavenumber": float(gradient_norm / max(q_norm, 1e-300)),
        "effective_gradient_length_scale": float(q_norm / max(gradient_norm, 1e-300)),
        "effective_gradient_length_scale_in_minimum_grid_spacings": float(
            q_norm / max(gradient_norm * spacing, 1e-300)
        ),
        "normalized_dominant_position_profile": normalized_profile.tolist(),
    }


def profile_correlation(coarse_wave, coarse, fine_wave, fine, absolute=False):
    if coarse["dominant_field_index"] != fine["dominant_field_index"]:
        return 0.0
    coarse_profile = np.asarray(coarse["normalized_dominant_position_profile"])
    fine_profile = np.asarray(fine["normalized_dominant_position_profile"])
    if absolute:
        coarse_profile = np.abs(coarse_profile)
        fine_profile = np.abs(fine_profile)
    interpolated = RectBivariateSpline(
        fine_wave.z, fine_wave.r, fine_profile, kx=3, ky=3, s=0,
    )(coarse_wave.z, coarse_wave.r, grid=True)
    left = coarse_profile.ravel()
    right = interpolated.ravel()
    mass = coarse_wave.mass
    numerator = float(left @ (mass @ right))
    denominator = np.sqrt(
        max(float(left @ (mass @ left)), 0.0)
        * max(float(right @ (mass @ right)), 0.0)
    )
    return float(abs(numerator) / max(denominator, 1e-300))


def main():
    geometry = build_geometry("G6")
    print(f"sampling kappa={DAMPING_RATE:g} wave coefficients", flush=True)
    coefficients = sample_coefficients(geometry, constraint_damping=DAMPING_RATE)
    seed = np.array((0.31, -0.22, 0.17, -0.29, 0.13, 0.23, -0.19))
    seed /= np.linalg.norm(seed)
    records = []
    waves = []
    for nz, nr in GRIDS:
        print(f"growth refinement grid {nz} x {nr}", flush=True)
        wave, principal, _, first, _, _ = sampled_system(
            geometry, coefficients, nz, nr, r_max=4.0, outer_dirichlet=False,
        )
        waves.append(wave)
        basis = np.broadcast_to(seed, (nz, 7)).copy()
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
            ("principal_only_zero_robin", ()),
            ("compact_first_only_zero_robin", ("z_first",)),
            ("full_bulk_zero_robin", ("reaction", "time_first", "z_first", "r_first")),
        ):
            print(f"  {label}", flush=True)
            variants.append(run_variant(
                wave, position, velocity, maximum_speed, active, label,
            ))
        records.append({
            "grid_size": [nz, nr],
            "coefficient_summary": coefficient_summary(first),
            "variants": variants,
        })

    comparisons = {}
    for label in (
        "principal_only_zero_robin",
        "compact_first_only_zero_robin",
        "full_bulk_zero_robin",
    ):
        selected = [
            next(item for item in record["variants"] if item["label"] == label)
            for record in records
        ]
        growth = np.array([item["late_logarithmic_growth_rate"] for item in selected])
        comparisons[label] = {
            "growth_rates": growth.tolist(),
            "growth_rate_relative_difference": float(
                abs(growth[-1] - growth[-2])
                / max(abs(growth[-1]), abs(growth[-2]), 1e-300)
            ),
            "dominant_profile_absolute_correlation": profile_correlation(
                waves[-2], selected[-2], waves[-1], selected[-1],
            ),
            "dominant_amplitude_profile_correlation": profile_correlation(
                waves[-2], selected[-2], waves[-1], selected[-1], absolute=True,
            ),
        }
    compact = comparisons["compact_first_only_zero_robin"]
    full = comparisons["full_bulk_zero_robin"]
    principal_amplification = max(
        next(item for item in record["variants"] if item["label"] == "principal_only_zero_robin")["amplification"]
        for record in records
    )
    acceptance = {
        "principal_runs_bounded_below_10x": bool(principal_amplification < 10.0),
        "compact_first_growth_reproduced_above_rate_10": bool(min(compact["growth_rates"]) > 10.0),
        "full_bulk_growth_reproduced_above_rate_10": bool(min(full["growth_rates"]) > 10.0),
        "compact_first_growth_rates_agree_within_20_percent": bool(compact["growth_rate_relative_difference"] < 0.20),
        "full_bulk_growth_rates_agree_within_20_percent": bool(full["growth_rate_relative_difference"] < 0.20),
        "compact_first_dominant_profiles_correlate_above_0p9": bool(compact["dominant_profile_absolute_correlation"] > 0.90),
        "full_bulk_dominant_profiles_correlate_above_0p9": bool(full["dominant_profile_absolute_correlation"] > 0.90),
        "compact_first_amplitude_profiles_correlate_above_0p9": bool(compact["dominant_amplitude_profile_correlation"] > 0.90),
        "full_bulk_amplitude_profiles_correlate_above_0p9": bool(full["dominant_amplitude_profile_correlation"] > 0.90),
    }
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "scope": "two-grid growth-rate and mode-shape test of the corrected-G6 lower-order instability",
        "field_order": list(FIELD_ORDER),
        "pulse_amplitude": AMPLITUDE,
        "final_time": FINAL_TIME,
        "grids": [list(grid) for grid in GRIDS],
        "records": records,
        "comparisons": comparisons,
        "acceptance": acceptance,
        "interpretation_rule": (
            "Grid-stable positive growth and a convergent normalized mode shape support a finite continuum lower-order instability; growth increasing without profile convergence would indicate a mesh-scale defect."
        ),
        "limitations": [
            "two grids and one mixed pulse",
            "linear fixed-background operator with the artificial radial face left uncontrolled",
            "a convergent mode does not distinguish gauge, constraint, and physical instability",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "comparisons": comparisons,
        "acceptance": acceptance,
        "summaries": [{
            "grid": record["grid_size"],
            "compact_coefficient": record["coefficient_summary"]["compact"],
            "variants": [{
                key: item[key] for key in (
                    "label", "amplification", "late_logarithmic_growth_rate",
                    "dominant_field", "dominant_field_fraction",
                    "position_maximum_coordinates", "effective_gradient_length_scale",
                    "effective_gradient_length_scale_in_minimum_grid_spacings",
                )
            } for item in record["variants"]],
        } for record in records],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
