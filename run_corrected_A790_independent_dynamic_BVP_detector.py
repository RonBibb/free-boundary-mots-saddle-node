#!/usr/bin/env python3
"""Sealed independent local-BVP confirmation of the A=7.90 cap pair."""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.dynamical_capped_horizon_bvp import (
    local_outgoing_expansion,
    solve_dynamical_capped_surface_bvp,
)
from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry


OUTPUT = Path("results/corrected_A790_independent_dynamic_BVP_detector.json")
TIME_STATE = Path("results/corrected_A790_formation_time_refinement_state.npz")
LONG_STATE = Path("results/corrected_A790_two_grid_formation_search_state.npz")
TIME_RESULT = Path("results/corrected_A790_formation_time_refinement.json")
LONG_RESULT = Path("results/corrected_A790_t004_discrepancy_formation_confirmation.json")
PROTOCOL = "notes/80_A790_independent_dynamic_BVP_detector_protocol.md"
AMPLITUDE = 7.90
SEEDS = tuple(np.linspace(1.15, 1.70, 12))
TIMES = (0.0, 0.000625, 0.001, 0.004)
LOCAL_LIMIT = 2e-4
CROSSCHECK_LIMIT = 0.002
SIGNATURE_LIMIT = 0.005


def scalar_relative(left, right):
    return float(abs(left - right) / max(abs(left), abs(right), 1e-300))


def public_surface(surface):
    if "error" in surface:
        return surface
    return {
        key: surface[key] for key in (
            "converged", "solver_success", "message", "in_domain",
            "iterations", "mesh_nodes_used", "rho_axis", "rho_brane",
            "rho_min", "rho_max", "boundary_slope_error",
            "local_expansion_interior_maximum", "local_expansion_full_maximum",
            "ode_defect_maximum", "primary_evaluator_crosscheck",
            "interior_point_count", "runtime_seconds",
        )
    }


def admitted(surface):
    crosscheck = surface.get("primary_evaluator_crosscheck", {})
    return bool(
        "error" not in surface
        and surface["converged"]
        and surface["local_expansion_interior_maximum"] < LOCAL_LIMIT
        and surface["boundary_slope_error"] < LOCAL_LIMIT
        and "error" not in crosscheck
        and crosscheck.get("two_cell_interior_maximum", np.inf) < CROSSCHECK_LIMIT
    )


def cluster_trials(trials):
    clusters = []
    for trial in trials:
        if not trial["admitted"]:
            continue
        signature = np.asarray((
            trial["surface"]["rho_axis"], trial["surface"]["rho_brane"],
        ))
        destination = None
        for cluster in clusters:
            if np.linalg.norm(signature - np.asarray(cluster["signature"])) < SIGNATURE_LIMIT:
                destination = cluster
                break
        if destination is None:
            destination = {"signature": signature.tolist(), "members": []}
            clusters.append(destination)
        destination["members"].append({
            "seed": trial["seed"], "signature": signature.tolist(),
        })
        member_signatures = np.asarray([
            member["signature"] for member in destination["members"]
        ])
        destination["signature"] = np.median(member_signatures, axis=0).tolist()
    return sorted(clusters, key=lambda item: item["signature"][1])


def search_slice(label, position, velocity, geometry):
    trials = []
    for seed in SEEDS:
        print(f"{label}, seed={seed:.2f}: local BVP", flush=True)
        started = time.perf_counter()
        try:
            surface = solve_dynamical_capped_surface_bvp(
                position, velocity, geometry["z"], geometry["r"], seed,
                tolerance=2e-5, nodes=121, maximum_nodes=6000,
                dense_nodes=501,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            surface = {"error": f"{type(error).__name__}: {error}"}
        surface["runtime_seconds"] = float(time.perf_counter() - started)
        trials.append({
            "seed": float(seed), "admitted": admitted(surface),
            "surface": public_surface(surface),
        })
    clusters = cluster_trials(trials)
    return {
        "label": label,
        "trial_count": len(trials),
        "admitted_trial_count": int(sum(trial["admitted"] for trial in trials)),
        "admitted_distinct_count": len(clusters),
        "clusters": clusters,
        "admitted_signatures": [cluster["signature"] for cluster in clusters],
        "runtime_seconds": float(sum(
            trial["surface"]["runtime_seconds"] for trial in trials
        )),
        "trials": trials,
    }


def reference_signatures():
    refined = json.loads(TIME_RESULT.read_text())
    long = json.loads(LONG_RESULT.read_text())
    references = {label: {} for label in ("G7", "G8")}
    for label in references:
        references[label]["0.000625"] = refined["trajectory"][label][4][
            "admitted_signatures"
        ]
        references[label]["0.001"] = refined["fine_median_signatures"][label]
        references[label]["0.004"] = long["dynamic_search"][label][
            "admitted_signatures"
        ]
    return references


def branch_comparison(found, reference):
    if len(found) != 2 or len(reference) != 2:
        return None
    records = []
    for local, primary in zip(
        sorted(found, key=lambda value: value[1]),
        sorted(reference, key=lambda value: value[1]),
    ):
        records.append({
            "bvp_signature": local,
            "spectral_signature": primary,
            "axis_relative_difference": scalar_relative(local[0], primary[0]),
            "brane_relative_difference": scalar_relative(local[1], primary[1]),
        })
    return records


def transfer(left, right):
    comparison = branch_comparison(left, right)
    if comparison is None:
        return None
    maximum = max(
        value for item in comparison for value in (
            item["axis_relative_difference"], item["brane_relative_difference"],
        )
    )
    return {"branches": comparison, "maximum": float(maximum)}


def analytic_controls():
    z = np.linspace(0.0, 4.0, 49)
    r = np.linspace(0.0, 3.0, 65)
    position = np.zeros((len(z), len(r), 9))
    position[:, :, 2] = -1.0
    position[:, :, 3] = 1.0
    position[:, :, 6] = 1.0
    zero_velocity = np.zeros_like(position)
    theta = np.linspace(1e-3, np.pi / 2.0, 301)
    radius = 1.2
    prepared = prepare_capped_expansion_slice(
        position, zero_velocity, z, r,
    )
    flat_expansion = local_outgoing_expansion(
        prepared, theta, np.full_like(theta, radius),
        np.zeros_like(theta), np.zeros_like(theta),
    )
    flat_error = float(np.max(np.abs(flat_expansion - 3.0 / radius)))
    k = 0.8
    velocity = np.zeros_like(position)
    velocity[:, :, 3] = -2.0 * k
    velocity[:, :, 6] = -2.0 * k
    solved = solve_dynamical_capped_surface_bvp(
        position, velocity, z, r, 1.1, tolerance=1e-7, nodes=81,
        dense_nodes=301,
    )
    target = 1.0 / k
    endpoint_error = max(
        scalar_relative(solved["rho_axis"], target),
        scalar_relative(solved["rho_brane"], target),
    )
    passed = bool(
        flat_error < 2e-8 and solved["converged"]
        and endpoint_error < 1e-4
        and solved["primary_evaluator_crosscheck"][
            "two_cell_interior_maximum"
        ] < 2e-5
    )
    return {
        "passed": passed,
        "flat_theta_plus_maximum_absolute_error": flat_error,
        "constant_curvature_target_radius": target,
        "constant_curvature_rho_axis": solved["rho_axis"],
        "constant_curvature_rho_brane": solved["rho_brane"],
        "constant_curvature_endpoint_relative_error": endpoint_error,
        "constant_curvature_local_expansion": solved[
            "local_expansion_interior_maximum"
        ],
        "constant_curvature_primary_crosscheck": solved[
            "primary_evaluator_crosscheck"
        ],
    }


def main():
    required = (TIME_STATE, LONG_STATE, TIME_RESULT, LONG_RESULT)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required archived inputs missing: {missing}")
    overall_started = time.perf_counter()
    controls = analytic_controls()
    print("reconstructing corrected G7/G8 A=7.90 geometries", flush=True)
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": AMPLITUDE}
    geometries = {
        "G7": build_refined(
            seed, 81, 121, "G7A790-independent-BVP",
            selector_iterations=40, slice_iterations=270,
        ),
    }
    geometries["G8"] = build_refined(
        geometries["G7"], 97, 145, "G8A790-independent-BVP",
        selector_iterations=45, slice_iterations=280,
    )
    time_archive = np.load(TIME_STATE)
    long_archive = np.load(LONG_STATE)
    references = reference_signatures()
    searches = {label: {} for label in geometries}
    for label, geometry in geometries.items():
        initial = np.asarray(geometry["jet_field"].reduced_fields)
        states = {
            "0": (initial, np.zeros_like(initial)),
            "0.000625": (
                initial + time_archive[f"{label}_fine_time_4_increment"],
                time_archive[f"{label}_fine_time_4_velocity"],
            ),
            "0.001": (
                initial + time_archive[f"{label}_8step_increment"],
                time_archive[f"{label}_8step_velocity"],
            ),
            "0.004": (
                initial + long_archive[f"{label}_time_3_increment"],
                long_archive[f"{label}_time_3_velocity"],
            ),
        }
        for time_label, (position, velocity) in states.items():
            searches[label][time_label] = search_slice(
                f"{label}-A790-t{time_label}", position, velocity, geometry,
            )

    comparisons = {label: {} for label in geometries}
    for label in geometries:
        for time_label in ("0.000625", "0.001", "0.004"):
            comparisons[label][time_label] = branch_comparison(
                searches[label][time_label]["admitted_signatures"],
                references[label][time_label],
            )
    transfers = {
        time_label: transfer(
            searches["G7"][time_label]["admitted_signatures"],
            searches["G8"][time_label]["admitted_signatures"],
        ) for time_label in ("0.000625", "0.001", "0.004")
    }
    admitted_surfaces = [
        trial["surface"]
        for label in searches.values() for record in label.values()
        for trial in record["trials"] if trial["admitted"]
    ]
    all_comparisons = [
        item for label in comparisons.values() for records in label.values()
        for item in (records or [])
    ]
    acceptance = {
        "analytic_controls_pass": controls["passed"],
        "both_initial_slices_admit_zero_candidates": bool(all(
            searches[label]["0"]["admitted_distinct_count"] == 0
            for label in searches
        )),
        "both_grids_admit_two_multiseed_branches_at_all_positive_times": bool(all(
            searches[label][time_label]["admitted_distinct_count"] == 2
            and all(len(cluster["members"]) >= 2 for cluster in searches[label][time_label]["clusters"])
            for label in searches for time_label in ("0.000625", "0.001", "0.004")
        )),
        "all_admitted_residuals_and_spectral_endpoint_comparisons_pass": bool(
            admitted_surfaces and all_comparisons
            and all(
                surface["local_expansion_interior_maximum"] < LOCAL_LIMIT
                and surface["boundary_slope_error"] < LOCAL_LIMIT
                and surface["primary_evaluator_crosscheck"]["two_cell_interior_maximum"] < CROSSCHECK_LIMIT
                for surface in admitted_surfaces
            )
            and max(
                value for item in all_comparisons for value in (
                    item["axis_relative_difference"], item["brane_relative_difference"],
                )
            ) < 0.005
        ),
        "two_grid_BVP_endpoints_transfer_below_1_percent": bool(all(
            item is not None and item["maximum"] < 0.01
            for item in transfers.values()
        )),
    }
    payload = {
        "status": "pass" if all(acceptance.values()) else "review",
        "classification": (
            "independent_local_BVP_confirmation_of_paired_formation_candidate"
            if all(acceptance.values()) else "independent_detector_review"
        ),
        "scope": "sealed independent local-BVP audit of the corrected A=7.90 marginal-surface pair",
        "protocol": PROTOCOL,
        "amplitude": AMPLITUDE,
        "times": list(TIMES),
        "seeds": list(SEEDS),
        "thresholds": {
            "local_expansion": LOCAL_LIMIT,
            "primary_evaluator_crosscheck": CROSSCHECK_LIMIT,
            "deduplication_signature_distance": SIGNATURE_LIMIT,
            "spectral_endpoint_relative_difference": 0.005,
            "two_grid_endpoint_relative_difference": 0.01,
        },
        "analytic_controls": controls,
        "searches": searches,
        "spectral_references": references,
        "spectral_endpoint_comparisons": comparisons,
        "two_grid_BVP_endpoint_transfer": transfers,
        "runtime": {
            "total_seconds": float(time.perf_counter() - overall_started),
            "surface_solve_seconds": float(sum(
                record["runtime_seconds"]
                for label in searches.values() for record in label.values()
            )),
            "surface_solve_count": int(sum(
                record["trial_count"]
                for label in searches.values() for record in label.values()
            )),
        },
        "acceptance": acceptance,
        "limitations": [
            "shares the audited ADM slice reduction and interpolation with the primary detector",
            "finite twelve-seed star-shaped donor-capped search is not a global nonexistence proof",
            "apparent-horizon position remains foliation dependent",
            "not event-horizon location, topology change, open amplitude basin, long-time stability, connected bulk geometry, or mass transfer",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "classification": payload["classification"],
        "counts": {
            label: {
                time_label: record["admitted_distinct_count"]
                for time_label, record in records.items()
            } for label, records in searches.items()
        },
        "comparisons": comparisons,
        "transfers": transfers,
        "runtime": payload["runtime"],
        "acceptance": acceptance,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
