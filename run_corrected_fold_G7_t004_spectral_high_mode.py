#!/usr/bin/env python3
"""Fresh high-mode audit of the archived final G7 t=0.004 marginal cap."""

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


OUTPUT = Path("results/corrected_fold_G7_t004_spectral_high_mode.json")
CHECKPOINT = Path("results/corrected_fold_G7_doubled_duration_horizon_state.npz")
MODES = (56, 64, 72)
COLLOCATION = 257


def relative_scalar(left, right):
    return float(abs(left - right) / max(abs(left), abs(right), 1e-300))


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError("note-66 doubled-duration checkpoint is required")
    archived = np.load(CHECKPOINT)
    print("reconstructing corrected G7 A=7.94 initial geometry", flush=True)
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": 7.94}
    geometry = build_refined(
        seed, 81, 121, "G7A794-t004-high", selector_iterations=40,
        slice_iterations=270,
    )
    initial = solve_anisotropic_capped_profile(
        geometry["z"], geometry["r"], geometry["psi"], geometry["a"],
        geometry["b"], geometry["c"], 1.53, tolerance=1e-8, nodes=220,
        max_nodes=12000,
    )
    if not initial["converged"]:
        raise RuntimeError("initial outer cap failed")
    position = geometry["jet_field"].reduced_fields + archived["final_increment"]
    velocity = archived["final_velocity"]
    surfaces = []
    for modes in MODES:
        print(f"solving final G7 cap with {modes} modes", flush=True)
        surfaces.append(solve_spectral_dynamical_capped_surface(
            position, velocity, geometry["z"], geometry["r"], initial,
            tolerance=5e-4, collocation_nodes=COLLOCATION,
            cosine_modes=modes, maximum_evaluations=240,
        ))
    initial_on = np.interp(surfaces[-1]["theta"], initial["theta"], initial["rho"])
    records = [
        {
            "modes": modes,
            "converged": surface["converged"],
            "in_domain": surface["in_domain"],
            "expansion_maximum": surface["interior_expansion_maximum"],
            "rho_axis": surface["rho_axis"],
            "rho_brane": surface["rho_brane"],
            "axis_change": float((surface["rho_axis"] - initial["rho_axis"]) / initial["rho_axis"]),
            "brane_change": float((surface["rho_brane"] - initial["rho_brane"]) / initial["rho_brane"]),
            "jacobian_condition_number": surface["jacobian_condition_number"],
        }
        for modes, surface in zip(MODES, surfaces)
    ]
    comparisons = []
    for left_modes, right_modes, left, right in zip(
        MODES[:-1], MODES[1:], surfaces[:-1], surfaces[1:],
    ):
        comparisons.append({
            "modes": [left_modes, right_modes],
            "profile_relative_difference": relative_norm(left["rho"], right["rho"]),
            "axis_relative_difference": relative_scalar(left["rho_axis"], right["rho_axis"]),
            "brane_relative_difference": relative_scalar(left["rho_brane"], right["rho_brane"]),
            "displacement_relative_difference": relative_norm(
                left["rho"] - initial_on, right["rho"] - initial_on,
            ),
        })
    acceptance = {
        "all_surfaces_converge_below_5e_4": bool(all(
            record["converged"] and record["in_domain"]
            and record["expansion_maximum"] < 5e-4 for record in records
        )),
        "adjacent_profiles_and_radii_transfer_below_0_05_percent": bool(max(
            max(
                item["profile_relative_difference"],
                item["axis_relative_difference"],
                item["brane_relative_difference"],
            ) for item in comparisons
        ) < .0005),
        "adjacent_displacements_transfer_below_1_percent": bool(max(
            item["displacement_relative_difference"] for item in comparisons
        ) < .01),
        "all_motion_directions_positive": bool(all(
            record["axis_change"] > 0 and record["brane_change"] > 0
            for record in records
        )),
    }
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "scope": "fresh sealed 56/64/72-mode refinement of the archived final G7 t=0.004 smooth marginal cap",
        "protocol": "notes/67_G7_t004_spectral_high_mode_protocol.md",
        "records": records,
        "comparisons": comparisons,
        "acceptance": acceptance,
        "limitations": [
            "reuses archived note-66 G7 spacetime",
            "does not rescore note 66 or alter its displacement-ratio failure",
            "pre-existing cap rather than formation or long-time stability",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
