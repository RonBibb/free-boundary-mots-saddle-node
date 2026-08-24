#!/usr/bin/env python3
"""Unscored blind-control audit for a horizonless-start detector.

The same scalar seed bank is applied to a below-fold A=7.90 slice and a
nearby A=7.94 positive control.  Dynamic spectral candidates must survive an
adjacent-mode comparison; the static multi-seed BVP supplies an independent
reference rather than a profile seed for the dynamic solver.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.anisotropic_capped_surface import find_anisotropic_donor_capped_surfaces
from bhps.dynamical_capped_horizon import (
    capped_outgoing_expansion,
    prepare_capped_expansion_slice,
    solve_spectral_dynamical_capped_surface,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import relative_norm


OUTPUT = Path("results/corrected_A790_blind_horizon_detector_engineering.json")
AMPLITUDES = (7.90, 7.94)
SEEDS = tuple(np.linspace(1.15, 1.70, 12))
MODES = (40, 48)
TARGET = 5e-4
COLLOCATION = 257


def scalar_relative(left, right):
    return float(abs(left - right) / max(abs(left), abs(right), 1e-300))


def independent_expansion(position, velocity, geometry, surface, prepared=None):
    theta = np.linspace(1e-4, np.pi / 2, 513)
    indices = np.arange(surface["cosine_modes"])
    coefficients = np.asarray(surface["cosine_coefficients"])
    profile = {
        "theta": theta,
        "rho": np.cos(2 * theta[:, None] * indices[None, :]) @ coefficients,
        "slope": (
            -2 * indices[None, :] * np.sin(2 * theta[:, None] * indices[None, :])
        ) @ coefficients,
    }
    evaluated = capped_outgoing_expansion(
        position, velocity, geometry["z"], geometry["r"], profile,
        prepared=prepared,
    )
    return {
        "nodes": len(theta),
        "two_cell_interior_maximum": evaluated["two_cell_interior_maximum_absolute"],
        "two_cell_interior_normalized_maximum": evaluated[
            "two_cell_interior_maximum_normalized"
        ],
    }


def public_surface(surface):
    if "error" in surface:
        return surface
    return {
        "converged": surface["converged"],
        "optimizer_success": surface["optimizer_success"],
        "in_domain": surface["in_domain"],
        "function_evaluations": surface["function_evaluations"],
        "cosine_modes": surface["cosine_modes"],
        "rho_axis": surface["rho_axis"],
        "rho_brane": surface["rho_brane"],
        "interior_expansion_maximum": surface["interior_expansion_maximum"],
        "jacobian_condition_number": surface["jacobian_condition_number"],
        "independent_expansion": surface.get("independent_expansion"),
    }


def solve_seed(position, velocity, geometry, seed, modes, prepared=None):
    try:
        surface = solve_spectral_dynamical_capped_surface(
            position, velocity, geometry["z"], geometry["r"], seed,
            tolerance=TARGET, collocation_nodes=COLLOCATION,
            cosine_modes=modes, maximum_evaluations=240, prepared=prepared,
        )
        if surface["converged"]:
            surface["independent_expansion"] = independent_expansion(
                position, velocity, geometry, surface, prepared,
            )
        return surface
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        return {"error": f"{type(error).__name__}: {error}", "cosine_modes": modes}


def stable_pair(left, right):
    if "error" in left or "error" in right:
        return None
    profile_difference = relative_norm(left["rho"], right["rho"])
    axis_difference = scalar_relative(left["rho_axis"], right["rho_axis"])
    brane_difference = scalar_relative(left["rho_brane"], right["rho_brane"])
    left_validation = left.get("independent_expansion", {}).get(
        "two_cell_interior_maximum", np.inf,
    )
    right_validation = right.get("independent_expansion", {}).get(
        "two_cell_interior_maximum", np.inf,
    )
    admitted = bool(
        left["converged"] and right["converged"]
        and max(left["jacobian_condition_number"], right["jacobian_condition_number"]) < 1e5
        and max(profile_difference, axis_difference, brane_difference) < .002
        and max(left_validation, right_validation) < .002
    )
    return {
        "admitted": admitted,
        "profile_relative_difference": profile_difference,
        "axis_relative_difference": axis_difference,
        "brane_relative_difference": brane_difference,
        "representative_rho_axis": right["rho_axis"],
        "representative_rho_brane": right["rho_brane"],
    }


def deduplicate(pairs):
    accepted = []
    for pair in pairs:
        if pair is None or not pair["admitted"]:
            continue
        signature = np.array((
            pair["representative_rho_axis"], pair["representative_rho_brane"],
        ))
        if not any(
            np.linalg.norm(signature - np.array((
                old["representative_rho_axis"], old["representative_rho_brane"],
            ))) < 5e-3
            for old in accepted
        ):
            accepted.append(pair)
    return accepted


def run_case(fold, amplitude):
    print(f"building corrected G7 A={amplitude:.2f} blind-control slice", flush=True)
    geometry = build_refined(
        {**fold, "fold_amplitude": amplitude}, 81, 121,
        f"G7A{int(round(100 * amplitude))}-blind-control",
        selector_iterations=40, slice_iterations=270,
    )
    static = find_anisotropic_donor_capped_surfaces(
        geometry["z"], geometry["r"], geometry["psi"], geometry["a"],
        geometry["b"], geometry["c"], guesses=SEEDS, tolerance=2e-5,
    )
    position = np.asarray(geometry["jet_field"].reduced_fields)
    velocity = np.zeros_like(position)
    prepared = prepare_capped_expansion_slice(
        position, velocity, geometry["z"], geometry["r"],
    )
    trials = []
    pairs = []
    for seed in SEEDS:
        surfaces = []
        for modes in MODES:
            print(
                f"A={amplitude:.2f}, seed={seed:.2f}: solving {modes} modes",
                flush=True,
            )
            surfaces.append(solve_seed(
                position, velocity, geometry, seed, modes, prepared,
            ))
        pair = stable_pair(*surfaces)
        pairs.append(pair)
        trials.append({
            "seed": float(seed),
            "surfaces": [public_surface(surface) for surface in surfaces],
            "adjacent_mode_pair": pair,
        })
    blind = deduplicate(pairs)
    static_signatures = [
        [item["rho_axis"], item["rho_brane"]] for item in static["accepted"]
    ]
    blind_signatures = [
        [item["representative_rho_axis"], item["representative_rho_brane"]]
        for item in blind
    ]
    nearest = []
    for signature in blind_signatures:
        nearest.append(min(
            (
                float(np.linalg.norm(np.asarray(signature) - np.asarray(reference))),
                reference,
            )
            for reference in static_signatures
        ) if static_signatures else None)
    return {
        "amplitude": amplitude,
        "grid_size": [len(geometry["z"]), len(geometry["r"])],
        "static_multiseed": {
            "trial_count": static["trial_count"],
            "accepted_count": len(static["accepted"]),
            "accepted_signatures": static_signatures,
        },
        "dynamic_blind": {
            "trial_count": len(trials),
            "admitted_distinct_count": len(blind),
            "admitted_signatures": blind_signatures,
            "nearest_static_signature": nearest,
            "trials": trials,
        },
    }


def main():
    fold = build_geometry("G6")
    records = [run_case(fold, amplitude) for amplitude in AMPLITUDES]
    negative, positive = records
    checks = {
        "A7_90_static_and_dynamic_find_no_cap": bool(
            negative["static_multiseed"]["accepted_count"] == 0
            and negative["dynamic_blind"]["admitted_distinct_count"] == 0
        ),
        "A7_94_static_and_dynamic_recover_two_caps": bool(
            positive["static_multiseed"]["accepted_count"] == 2
            and positive["dynamic_blind"]["admitted_distinct_count"] == 2
        ),
        "A7_94_dynamic_caps_match_static_endpoints": bool(
            len(positive["dynamic_blind"]["nearest_static_signature"]) == 2
            and all(
                match is not None and match[0] < .02
                for match in positive["dynamic_blind"]["nearest_static_signature"]
            )
        ),
    }
    payload = {
        "status": "engineering_control_pass" if all(checks.values()) else "engineering_review",
        "scope": "unscored blind multiseed t=0 control for a horizonless-start dynamic cap detector",
        "amplitudes": list(AMPLITUDES),
        "seeds": list(SEEDS),
        "cosine_modes": list(MODES),
        "expansion_target": TARGET,
        "checks": checks,
        "records": records,
        "limitations": [
            "engineering control with no prospectively sealed acceptance rules",
            "one G7 grid and time-symmetric initial slices",
            "finite seed bank cannot prove global nonexistence of a cap",
            "not formation, branch selection, event-horizon location, topology change, or mass transfer",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"], "checks": checks,
        "counts": {
            str(record["amplitude"]): [
                record["static_multiseed"]["accepted_count"],
                record["dynamic_blind"]["admitted_distinct_count"],
            ] for record in records
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
