#!/usr/bin/env python3
"""Sealed cross-slice discrepancy-principle spectral horizon tracker."""

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


OUTPUT = Path("results/corrected_fold_discrepancy_horizon_tracker.json")
SHORT = Path("results/corrected_fold_G7_G8_spectral_horizon_refinement_state.npz")
LONG = Path("results/corrected_fold_G7_doubled_duration_horizon_state.npz")
CANDIDATES = (24, 28, 32, 36, 40, 48, 56, 64)
TARGET = 5e-4
COLLOCATION = 257


def relative_scalar(left, right):
    return float(abs(left - right) / max(abs(left), abs(right), 1e-300))


def static_cap(geometry):
    result = solve_anisotropic_capped_profile(
        geometry["z"], geometry["r"], geometry["psi"], geometry["a"],
        geometry["b"], geometry["c"], 1.53, tolerance=1e-8, nodes=220,
        max_nodes=12000,
    )
    if not result["converged"]:
        raise RuntimeError("initial outer cap failed")
    return result


def select_surface(label, position, velocity, geometry, initial):
    scanned = []
    first_index = None
    for index, modes in enumerate(CANDIDATES):
        print(f"{label}: solving {modes} modes", flush=True)
        surface = solve_spectral_dynamical_capped_surface(
            position, velocity, geometry["z"], geometry["r"], initial,
            tolerance=TARGET, collocation_nodes=COLLOCATION,
            cosine_modes=modes, maximum_evaluations=240,
        )
        scanned.append(surface)
        if first_index is None and surface["converged"]:
            first_index = index
        if first_index is not None and index == first_index + 1:
            break
    if first_index is None or first_index + 1 >= len(scanned):
        raise RuntimeError(f"{label}: no discrepancy crossing plus confirmation")
    selected = scanned[first_index]
    confirmation = scanned[first_index + 1]
    initial_on = np.interp(selected["theta"], initial["theta"], initial["rho"])
    comparison = {
        "profile_relative_difference": relative_norm(selected["rho"], confirmation["rho"]),
        "axis_relative_difference": relative_scalar(selected["rho_axis"], confirmation["rho_axis"]),
        "brane_relative_difference": relative_scalar(selected["rho_brane"], confirmation["rho_brane"]),
        "displacement_relative_difference": relative_norm(
            selected["rho"] - initial_on, confirmation["rho"] - initial_on,
        ),
    }
    return {
        "label": label,
        "scanned": scanned,
        "first_index": first_index,
        "selected": selected,
        "confirmation": confirmation,
        "initial_on": initial_on,
        "comparison": comparison,
    }


def public_record(record):
    first = record["first_index"]
    scanned = record["scanned"]
    return {
        "selected_modes": record["selected"]["cosine_modes"],
        "confirmation_modes": record["confirmation"]["cosine_modes"],
        "previous_modes": scanned[first - 1]["cosine_modes"] if first else None,
        "previous_expansion_maximum": scanned[first - 1]["interior_expansion_maximum"] if first else None,
        "selected_expansion_maximum": record["selected"]["interior_expansion_maximum"],
        "confirmation_expansion_maximum": record["confirmation"]["interior_expansion_maximum"],
        "selected_condition_number": record["selected"]["jacobian_condition_number"],
        "confirmation_condition_number": record["confirmation"]["jacobian_condition_number"],
        "selected_axis_change": float(
            (record["selected"]["rho_axis"] - record["initial_on"][0])
            / record["initial_on"][0]
        ),
        "selected_brane_change": float(
            (record["selected"]["rho_brane"] - record["initial_on"][-1])
            / record["initial_on"][-1]
        ),
        "confirmation_axis_change": float(
            (record["confirmation"]["rho_axis"] - record["initial_on"][0])
            / record["initial_on"][0]
        ),
        "confirmation_brane_change": float(
            (record["confirmation"]["rho_brane"] - record["initial_on"][-1])
            / record["initial_on"][-1]
        ),
        "selected_confirmation_transfer": record["comparison"],
    }


def main():
    short = np.load(SHORT)
    long = np.load(LONG)
    print("reconstructing corrected G7/G8 A=7.94 geometries", flush=True)
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": 7.94}
    g7 = build_refined(
        seed, 81, 121, "G7A794-discrepancy", selector_iterations=40,
        slice_iterations=270,
    )
    g8 = build_refined(
        g7, 97, 145, "G8A794-discrepancy", selector_iterations=45,
        slice_iterations=280,
    )
    initial7 = static_cap(g7)
    initial8 = static_cap(g8)
    position7 = g7["jet_field"].reduced_fields
    position8 = g8["jet_field"].reduced_fields
    records = [
        select_surface(
            "G7_t0.002", position7 + short["G7_increment"],
            short["G7_velocity"], g7, initial7,
        ),
        select_surface(
            "G8_t0.002", position8 + short["G8_increment"],
            short["G8_velocity"], g8, initial8,
        ),
        select_surface(
            "G7_t0.004", position7 + long["final_increment"],
            long["final_velocity"], g7, initial7,
        ),
    ]
    g7_short, g8_short = records[:2]
    cross_grid = {
        "profile_relative_difference": relative_norm(
            g7_short["selected"]["rho"], g8_short["selected"]["rho"],
        ),
        "axis_relative_difference": relative_scalar(
            g7_short["selected"]["rho_axis"], g8_short["selected"]["rho_axis"],
        ),
        "brane_relative_difference": relative_scalar(
            g7_short["selected"]["rho_brane"], g8_short["selected"]["rho_brane"],
        ),
        "displacement_relative_difference": relative_norm(
            g7_short["selected"]["rho"] - g7_short["initial_on"],
            g8_short["selected"]["rho"] - g8_short["initial_on"],
        ),
    }
    acceptance = {
        "unique_first_crossings_with_previous_misses": bool(all(
            record["first_index"] > 0
            and record["scanned"][record["first_index"] - 1]["interior_expansion_maximum"] >= TARGET
            for record in records
        )),
        "selected_and_confirmation_surfaces_pass_discrepancy": bool(all(
            surface["converged"] and surface["in_domain"]
            and surface["interior_expansion_maximum"] < TARGET
            for record in records
            for surface in (record["selected"], record["confirmation"])
        )),
        "selected_and_confirmation_condition_below_1e5": bool(all(
            surface["jacobian_condition_number"] < 1e5
            for record in records
            for surface in (record["selected"], record["confirmation"])
        )),
        "within_slice_profile_radii_and_displacement_transfer": bool(
            max(
                max(
                    record["comparison"]["profile_relative_difference"],
                    record["comparison"]["axis_relative_difference"],
                    record["comparison"]["brane_relative_difference"],
                ) for record in records
            ) < .0005
            and max(
                record["comparison"]["displacement_relative_difference"]
                for record in records
            ) < .01
        ),
        "selected_and_confirmation_motion_positive": bool(all(
            surface[name] > record["initial_on"][endpoint]
            for record in records
            for surface in (record["selected"], record["confirmation"])
            for name, endpoint in (("rho_axis", 0), ("rho_brane", -1))
        )),
        "selected_G7_G8_t0_002_transfer": bool(
            max(
                cross_grid["profile_relative_difference"],
                cross_grid["axis_relative_difference"],
                cross_grid["brane_relative_difference"],
            ) < .002
            and cross_grid["displacement_relative_difference"] < .10
        ),
    }
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "scope": "sealed discrepancy-principle spectral tracker across G7/G8 t=0.002 and G7 t=0.004 archived slices",
        "protocol": "notes/68_discrepancy_principle_horizon_tracker_protocol.md",
        "candidate_modes": list(CANDIDATES),
        "expansion_discrepancy_target": TARGET,
        "collocation_nodes": COLLOCATION,
        "records": {record["label"]: public_record(record) for record in records},
        "selected_G7_G8_t0_002_transfer": cross_grid,
        "acceptance": acceptance,
        "limitations": [
            "selection rule follows an unscored engineering diagnosis",
            "does not rescore notes 63, 66, or 67",
            "three archived short-time slices and pre-existing caps",
            "not formation, event-horizon location, topology change, long-time stability, branch selection, an open basin, or mass transfer",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
