#!/usr/bin/env python3
"""Unscored mode-spectrum diagnosis for the archived G7 t=0.004 cap."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.anisotropic_capped_surface import solve_anisotropic_capped_profile
from bhps.dynamical_capped_horizon import solve_spectral_dynamical_capped_surface
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import relative_norm


OUTPUT = Path("results/corrected_fold_G7_t004_mode_diagnosis_engineering.json")
CHECKPOINT = Path("results/corrected_fold_G7_doubled_duration_horizon_state.npz")
MODES = (12, 16, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72)


def main():
    archived = np.load(CHECKPOINT)
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": 7.94}
    geometry = build_refined(
        seed, 81, 121, "G7A794-t004-diagnosis", selector_iterations=40,
        slice_iterations=270,
    )
    initial = solve_anisotropic_capped_profile(
        geometry["z"], geometry["r"], geometry["psi"], geometry["a"],
        geometry["b"], geometry["c"], 1.53, tolerance=1e-8, nodes=220,
        max_nodes=12000,
    )
    position = geometry["jet_field"].reduced_fields + archived["final_increment"]
    velocity = archived["final_velocity"]
    records = []
    previous = None
    for modes in MODES:
        print(f"engineering solve: {modes} modes", flush=True)
        surface = solve_spectral_dynamical_capped_surface(
            position, velocity, geometry["z"], geometry["r"], initial,
            tolerance=1.0, collocation_nodes=257, cosine_modes=modes,
            maximum_evaluations=240,
        )
        coefficients = np.asarray(surface["cosine_coefficients"])
        indices = np.arange(modes, dtype=float)
        tail_start = max(1, modes // 2)
        record = {
            "modes": modes,
            "expansion_maximum": surface["interior_expansion_maximum"],
            "expansion_rms": surface["interior_expansion_l2"]
            / np.sqrt(surface["interior_equation_count"]),
            "rho_axis": surface["rho_axis"],
            "rho_brane": surface["rho_brane"],
            "axis_change": float((surface["rho_axis"] - initial["rho_axis"]) / initial["rho_axis"]),
            "brane_change": float((surface["rho_brane"] - initial["rho_brane"]) / initial["rho_brane"]),
            "condition_number": surface["jacobian_condition_number"],
            "coefficient_norm": float(np.linalg.norm(coefficients)),
            "upper_half_coefficient_fraction": float(
                np.linalg.norm(coefficients[tail_start:])
                / max(np.linalg.norm(coefficients), 1e-300)
            ),
            "spectral_curvature_norm": float(np.linalg.norm((2 * indices) ** 2 * coefficients)),
        }
        if previous is not None:
            record["previous_profile_relative_difference"] = relative_norm(
                previous["rho"], surface["rho"],
            )
            record["previous_axis_relative_difference"] = float(
                abs(previous["rho_axis"] - surface["rho_axis"])
                / max(abs(previous["rho_axis"]), abs(surface["rho_axis"]), 1e-300)
            )
        records.append(record)
        previous = surface
    payload = {
        "status": "engineering_only",
        "scope": "unscored mode-spectrum and conditioning diagnosis of the archived G7 t=0.004 cap",
        "records": records,
        "limitations": [
            "no prospective acceptance decision",
            "archived note-66 spacetime",
            "diagnostic input to a resolution-aware tracker repair",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
