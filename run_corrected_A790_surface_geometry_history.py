#!/usr/bin/env python3
"""Sealed proper-area and shape history of the A=7.90 formed cap pair."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.dynamical_capped_geometry import capped_surface_geometry
from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice
from run_corrected_A790_blind_horizon_detector_engineering import (
    public_surface, solve_seed, stable_pair,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry


OUTPUT = Path("results/corrected_A790_surface_geometry_history.json")
PROTOCOL = "notes/81_A790_surface_geometry_protocol.md"
FINE_RESULT = Path("results/corrected_A790_formation_time_refinement.json")
FINE_STATE = Path("results/corrected_A790_formation_time_refinement_state.npz")
LONG_RESULT = Path("results/corrected_A790_two_grid_formation_search.json")
LONG_STATE = Path("results/corrected_A790_two_grid_formation_search_state.npz")
FINAL_RESULT = Path("results/corrected_A790_t004_discrepancy_formation_confirmation.json")
AMPLITUDE = 7.90
TIMES = (0.000625, 0.001, 0.002, 0.003, 0.004)


def relative_difference(left, right):
    return float(abs(left - right) / max(abs(left), abs(right), 1e-300))


def cluster_selections(clusters):
    selections = []
    for cluster in sorted(clusters, key=lambda item: item["signature"][1]):
        middle = len(cluster["seeds"]) // 2
        selections.append({
            "seed": cluster["seeds"][middle],
            "selected_modes": cluster["selected_confirmation_modes"][middle][0],
            "confirmation_modes": cluster["selected_confirmation_modes"][middle][1],
            "archived_signature": cluster["signature"],
        })
    return selections


def fixed_pair_selections(record):
    selections = []
    for signature in sorted(record["admitted_signatures"], key=lambda item: item[1]):
        admitted = [
            trial for trial in record["trials"]
            if trial["adjacent_mode_pair"] is not None
            and trial["adjacent_mode_pair"]["admitted"]
        ]
        trial = min(admitted, key=lambda item: np.linalg.norm(
            np.asarray(signature) - np.asarray((
                item["adjacent_mode_pair"]["representative_rho_axis"],
                item["adjacent_mode_pair"]["representative_rho_brane"],
            ))
        ))
        modes = [item["cosine_modes"] for item in trial["surfaces"]]
        selections.append({
            "seed": trial["seed"], "selected_modes": modes[0],
            "confirmation_modes": modes[1], "archived_signature": signature,
        })
    return selections


def selections_for(label, time, fine_result, long_result, final_result):
    if time <= 0.001:
        index = int(round(time / 0.000125)) - 1
        return cluster_selections(
            fine_result["trajectory"][label][index]["discrepancy"]["clusters"]
        )
    if time < 0.004:
        index = {0.002: 1, 0.003: 2}[time]
        return fixed_pair_selections(long_result["dynamic_search"][label][index])
    return cluster_selections(final_result["dynamic_search"][label]["clusters"])


def archived_slice(label, time, geometry, fine_state, long_state):
    initial = np.asarray(geometry["jet_field"].reduced_fields)
    if time < 0.001:
        index = int(round(time / 0.000125)) - 1
        prefix = f"{label}_fine_time_{index}"
        archive = fine_state
    elif time == 0.001:
        prefix = f"{label}_8step"
        archive = fine_state
    else:
        index = {0.002: 1, 0.003: 2, 0.004: 3}[time]
        prefix = f"{label}_time_{index}"
        archive = long_state
    return initial + archive[f"{prefix}_increment"], archive[f"{prefix}_velocity"]


def solve_branch(position, velocity, geometry, selection, prepared):
    surfaces = [
        solve_seed(
            position, velocity, geometry, selection["seed"], modes, prepared,
        )
        for modes in (
            selection["selected_modes"], selection["confirmation_modes"],
        )
    ]
    pair = stable_pair(*surfaces)
    geometric = [
        capped_surface_geometry(
            position, velocity, geometry["z"], geometry["r"], surface,
            prepared=prepared,
        )
        if "error" not in surface and surface.get("converged") else None
        for surface in surfaces
    ]
    area_difference = (
        relative_difference(
            geometric[0]["one_sided_cap_area"],
            geometric[1]["one_sided_cap_area"],
        ) if all(item is not None for item in geometric) else None
    )
    return {
        "selection": selection,
        "discrepancy_pair": pair,
        "pair_admitted": bool(pair is not None and pair["admitted"]),
        "surfaces": [public_surface(surface) for surface in surfaces],
        "geometry": geometric,
        "selected_confirmation_area_relative_difference": area_difference,
    }


def main():
    required = (FINE_RESULT, FINE_STATE, LONG_RESULT, LONG_STATE, FINAL_RESULT)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing archived inputs: {missing}")
    fine_result = json.loads(FINE_RESULT.read_text())
    long_result = json.loads(LONG_RESULT.read_text())
    final_result = json.loads(FINAL_RESULT.read_text())
    fine_state = np.load(FINE_STATE)
    long_state = np.load(LONG_STATE)

    print("reconstructing corrected G7/G8 A=7.90 geometries", flush=True)
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": AMPLITUDE}
    geometries = {
        "G7": build_refined(
            seed, 81, 121, "G7A790-area", selector_iterations=40,
            slice_iterations=270,
        ),
    }
    geometries["G8"] = build_refined(
        geometries["G7"], 97, 145, "G8A790-area",
        selector_iterations=45, slice_iterations=280,
    )

    records = {label: [] for label in geometries}
    for time in TIMES:
        for label, geometry in geometries.items():
            position, velocity = archived_slice(
                label, time, geometry, fine_state, long_state,
            )
            prepared = prepare_capped_expansion_slice(
                position, velocity, geometry["z"], geometry["r"],
            )
            selections = selections_for(
                label, time, fine_result, long_result, final_result,
            )
            print(
                f"{label} t={time:.6f}: solving {len(selections)} area branches",
                flush=True,
            )
            records[label].append({
                "time": time,
                "branches": [
                    solve_branch(
                        position, velocity, geometry, selection, prepared,
                    )
                    for selection in selections
                ],
            })

    cross_grid = []
    for time_index, time in enumerate(TIMES):
        branches = []
        for branch_index, name in enumerate(("inner", "outer")):
            left = records["G7"][time_index]["branches"][branch_index]
            right = records["G8"][time_index]["branches"][branch_index]
            left_area = left["geometry"][1]["one_sided_cap_area"]
            right_area = right["geometry"][1]["one_sided_cap_area"]
            branches.append({
                "branch": name,
                "area_relative_difference": relative_difference(
                    left_area, right_area,
                ),
                "equivalent_radius_relative_difference": relative_difference(
                    left["geometry"][1]["equivalent_area_radius"],
                    right["geometry"][1]["equivalent_area_radius"],
                ),
            })
        cross_grid.append({"time": time, "branches": branches})

    all_branches = [
        branch for grid in records.values() for item in grid
        for branch in item["branches"]
    ]
    all_geometry = [
        value for branch in all_branches for value in branch["geometry"]
        if value is not None
    ]
    area_ordering = all(
        item["branches"][1]["geometry"][1]["one_sided_cap_area"]
        > item["branches"][0]["geometry"][1]["one_sided_cap_area"]
        for grid in records.values() for item in grid
    )
    acceptance = {
        "both_branches_recovered_and_discrepancy_admitted_everywhere": bool(
            len(all_branches) == 2 * 2 * len(TIMES)
            and all(branch["pair_admitted"] for branch in all_branches)
        ),
        "all_areas_and_lengths_finite_and_positive": bool(
            len(all_geometry) == 2 * len(all_branches)
            and all(
                item["finite"] and item["one_sided_cap_area"] > 0
                and item["proper_meridional_length"] > 0
                for item in all_geometry
            )
        ),
        "all_mode_pair_area_differences_below_0_2_percent": bool(all(
            branch["selected_confirmation_area_relative_difference"] is not None
            and branch["selected_confirmation_area_relative_difference"] < 0.002
            for branch in all_branches
        )),
        "all_cross_grid_area_differences_below_1_percent": bool(all(
            branch["area_relative_difference"] < 0.01
            for item in cross_grid for branch in item["branches"]
        )),
        "outer_area_exceeds_inner_area_everywhere": bool(area_ordering),
    }
    status = "pass" if all(acceptance.values()) else "review"
    payload = {
        "status": status,
        "classification": (
            "resolved_proper_geometry_history_of_formed_pair"
            if status == "pass" else "unresolved_formed_pair_geometry_history"
        ),
        "scope": "sealed proper-area and basic shape history of the corrected A=7.90 marginal-surface pair",
        "protocol": PROTOCOL,
        "amplitude": AMPLITUDE,
        "times": list(TIMES),
        "records": records,
        "cross_grid": cross_grid,
        "acceptance": acceptance,
        "limitations": [
            "proper geometry of already admitted star-shaped donor-capped marginal surfaces",
            "one-sided cap area; doubled area assumes reflection symmetry",
            "not a MOTS stability spectrum, quasi-local mass, flux law, event horizon, topology change, connected bulk geometry, or dark-matter halo",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": status, "classification": payload["classification"],
        "acceptance": acceptance, "cross_grid": cross_grid,
        "confirmation_histories": {
            label: [{
                "time": item["time"],
                "areas": [
                    branch["geometry"][1]["one_sided_cap_area"]
                    for branch in item["branches"]
                ],
                "area_radii": [
                    branch["geometry"][1]["equivalent_area_radius"]
                    for branch in item["branches"]
                ],
            } for item in grid]
            for label, grid in records.items()
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
