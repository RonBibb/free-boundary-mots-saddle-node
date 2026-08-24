#!/usr/bin/env python3
"""Sealed bounded multi-topology pilot on archived corrected A=7.90 slices."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.corrected_A790_R12_builder import build_A790_R12_pair
from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice
from bhps.dynamical_capped_horizon_bvp import solve_dynamical_capped_surface_bvp
from bhps.dynamical_multitopology import (
    closed_local_expansion,
    solve_dynamical_closed_surface_bvp,
    solve_dynamical_closed_surface_fd,
    solve_dynamical_spanning_surface_bvp,
    solve_dynamical_spanning_surface_fd,
    spanning_local_expansion,
)
from bhps.recovery_indexer import (
    RecoveryIndex,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
    validate_npz,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry


PROTOCOL = Path("notes/97a_test12a_runner_builder_interface_erratum.md")
OUTPUT = Path("results/test12a_bounded_multitopology_pilot.json")
RECOVERY = Path("results/test12a_recovery_v2")
MANIFEST = RECOVERY / "manifest.json"
INITIAL_STATE = RECOVERY / "initial_geometries.npz"
R8_FINE = Path("results/corrected_A790_formation_time_refinement_state.npz")
R8_LONG = Path("results/corrected_A790_two_grid_formation_search_state.npz")
R12_LONG = Path("results/corrected_A790_R12_domain_sequence_state.npz")
AMPLITUDE = 7.90
LOCAL_LIMIT = 2e-4
INDEPENDENT_LIMIT = 0.002
SIGNATURE_DISTANCE = 0.005
CAP_SEEDS = tuple(float(value) for value in np.linspace(1.15, 1.70, 12))
CLOSED_CENTER_FRACTIONS = (0.20, 0.35, 0.50, 0.65, 0.80)
CLOSED_SEEDS = (0.16, 0.24, 0.34, 0.46, 0.60, 0.75)
SPANNING_SEED_COUNT = 18
CASES = (
    ("R8", "G7", "pre", 0.0005),
    ("R8", "G8", "pre", 0.0005),
    ("R8", "G7", "near", 0.000625),
    ("R8", "G8", "near", 0.000625),
    ("R8", "G7", "persistent", 0.004),
    ("R8", "G8", "persistent", 0.004),
    ("R12", "G7", "persistent", 0.004),
    ("R12", "G8", "persistent", 0.004),
)


def builtin(value):
    if isinstance(value, dict):
        return {str(key): builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def scalar_relative(left, right):
    return float(abs(left - right) / max(abs(left), abs(right), 1e-300))


def expected_inputs():
    paths = (
        R8_FINE,
        R8_LONG,
        R12_LONG,
        Path("src/bhps/dynamical_multitopology.py"),
        Path("src/bhps/dynamical_capped_horizon.py"),
        Path("src/bhps/dynamical_capped_horizon_bvp.py"),
        Path("src/bhps/recovery_indexer.py"),
        Path("src/bhps/corrected_A790_R12_builder.py"),
        Path("run_corrected_fold_nonlinear_rhs_G7_axis_refinement.py"),
        Path("run_corrected_fold_regular_so3_runtime.py"),
        Path(__file__),
    )
    return {str(path): sha256_file(path) for path in paths}


def run_json_stage(index, stage_id, expected_seconds, metadata, function):
    path = RECOVERY / f"{stage_id.replace('/', '__')}.json"
    index.register(stage_id, "json", expected_seconds, metadata)
    cached = index.validated_path(stage_id)
    if cached is not None:
        return json.loads(cached.read_text())
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        payload = builtin(function())
        atomic_write_json(path, payload)
        index.mark_complete(
            stage_id, path, time.perf_counter() - started,
            {"record_count": int(payload.get("trial_count", 1))},
        )
        return payload
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def build_initial_geometries(index):
    stage_id = "geometry/initial"
    metadata = {
        "amplitude": AMPLITUDE,
        "domains": ["R8", "R12"],
        "grids": ["G7", "G8"],
    }
    index.register(stage_id, "npz", 2400.0, metadata)
    cached = index.validated_path(stage_id)
    if cached is not None:
        validate_npz(cached)
        return cached
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        fold = build_geometry("G6")
        seed = {**fold, "fold_amplitude": AMPLITUDE}
        r8g7 = build_refined(
            seed, 81, 121, "G7A790-test12a-R8",
            selector_iterations=40, slice_iterations=270,
        )
        r8g8 = build_refined(
            r8g7, 97, 145, "G8A790-test12a-R8",
            selector_iterations=45, slice_iterations=280,
        )
        r12g7, r12g8 = build_A790_R12_pair()
        geometries = {
            ("R8", "G7"): r8g7,
            ("R8", "G8"): r8g8,
            ("R12", "G7"): r12g7,
            ("R12", "G8"): r12g8,
        }
        arrays = {}
        with np.load(R8_FINE, allow_pickle=False) as r8_archive, np.load(
            R12_LONG, allow_pickle=False,
        ) as r12_archive:
            for (domain, grid), geometry in geometries.items():
                source = r8_archive if domain == "R8" else r12_archive
                if not np.array_equal(geometry["z"], source[f"{grid}_z"]):
                    raise RuntimeError(f"{domain}/{grid} compact grid mismatch")
                if not np.array_equal(geometry["r"], source[f"{grid}_r"]):
                    raise RuntimeError(f"{domain}/{grid} radial grid mismatch")
                prefix = f"{domain}_{grid}"
                arrays[f"{prefix}_z"] = np.asarray(geometry["z"])
                arrays[f"{prefix}_r"] = np.asarray(geometry["r"])
                arrays[f"{prefix}_position"] = np.asarray(
                    geometry["jet_field"].reduced_fields,
                )
        atomic_write_npz(INITIAL_STATE, **arrays)
        validate_npz(INITIAL_STATE)
        index.mark_complete(
            stage_id, INITIAL_STATE, time.perf_counter() - started,
            {"array_count": len(arrays)},
        )
        return INITIAL_STATE
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def flat_state(z, r):
    position = np.zeros((len(z), len(r), 9))
    position[:, :, 2] = -1.0
    position[:, :, 3] = 1.0
    position[:, :, 6] = 1.0
    return position


def controls():
    z = np.linspace(0.0, 4.0, 65)
    r = np.linspace(0.0, 4.0, 81)
    position = flat_state(z, r)
    zero = np.zeros_like(position)
    prepared = prepare_capped_expansion_slice(position, zero, z, r)
    theta = np.linspace(1e-3, np.pi - 1e-3, 401)
    sphere_radius = 0.8
    closed_flat = closed_local_expansion(
        prepared, 2.0, theta, np.full_like(theta, sphere_radius),
        np.zeros_like(theta), np.zeros_like(theta),
    )
    compact = np.linspace(z[0], z[-1], 401)
    cylinder_radius = 1.2
    spanning_flat = spanning_local_expansion(
        prepared, compact, np.full_like(compact, cylinder_radius),
        np.zeros_like(compact), np.zeros_like(compact),
    )
    k_closed = 0.8
    velocity_closed = np.zeros_like(position)
    velocity_closed[:, :, 3] = -2.0 * k_closed
    velocity_closed[:, :, 6] = -2.0 * k_closed
    closed_bvp = solve_dynamical_closed_surface_bvp(
        position, velocity_closed, z, r, 2.0, 1.1,
        tolerance=1e-7, nodes=81, dense_nodes=401,
    )
    closed_fd = solve_dynamical_closed_surface_fd(
        position, velocity_closed, z, r, 2.0, closed_bvp,
        nodes=81, tolerance=1e-9,
    )
    k_spanning = 0.4
    velocity_spanning = np.zeros_like(position)
    velocity_spanning[:, :, 3] = -2.0 * k_spanning
    velocity_spanning[:, :, 6] = -2.0 * k_spanning
    spanning_bvp = solve_dynamical_spanning_surface_bvp(
        position, velocity_spanning, z, r, 1.4,
        tolerance=1e-7, nodes=81, dense_nodes=401,
    )
    spanning_fd = solve_dynamical_spanning_surface_fd(
        position, velocity_spanning, z, r, spanning_bvp,
        nodes=81, tolerance=1e-9,
    )
    flat_closed_trials = [
        solve_dynamical_closed_surface_bvp(position, zero, z, r, 2.0, seed)
        for seed in (0.4, 0.8, 1.2)
    ]
    flat_spanning_trials = [
        solve_dynamical_spanning_surface_bvp(position, zero, z, r, seed)
        for seed in (0.5, 1.0, 2.0, 3.0)
    ]
    target_closed = 1.0 / k_closed
    target_spanning = 2.0 / (3.0 * k_spanning)
    analytic = {
        "flat_closed_sphere_error": float(
            np.max(np.abs(closed_flat - 3.0 / sphere_radius)),
        ),
        "flat_spanning_cylinder_error": float(
            np.max(np.abs(spanning_flat - 2.0 / cylinder_radius)),
        ),
        "closed_BVP_target_relative_error": scalar_relative(
            closed_bvp["radius_max"], target_closed,
        ),
        "closed_FD_target_relative_error": scalar_relative(
            closed_fd["radius_max"], target_closed,
        ),
        "spanning_BVP_target_relative_error": scalar_relative(
            spanning_bvp["radius_A"], target_spanning,
        ),
        "spanning_FD_target_relative_error": scalar_relative(
            spanning_fd["radius_A"], target_spanning,
        ),
        "closed_BVP_independent_expansion": closed_bvp[
            "independent_expansion_interior_maximum"
        ],
        "spanning_BVP_independent_expansion": spanning_bvp[
            "independent_expansion_interior_maximum"
        ],
    }
    recovery = recovery_adverse_control()
    checks = {
        "flat_local_analytic_formulas": bool(
            analytic["flat_closed_sphere_error"] < 2e-8
            and analytic["flat_spanning_cylinder_error"] < 2e-8
        ),
        "closed_BVP_and_FD_positive_control": bool(
            closed_bvp["converged"] and closed_fd["converged"]
            and analytic["closed_BVP_target_relative_error"] < 1e-4
            and analytic["closed_FD_target_relative_error"] < 1e-3
            and analytic["closed_BVP_independent_expansion"] < 2e-4
        ),
        "spanning_BVP_and_FD_positive_control": bool(
            spanning_bvp["converged"] and spanning_fd["converged"]
            and analytic["spanning_BVP_target_relative_error"] < 1e-4
            and analytic["spanning_FD_target_relative_error"] < 1e-3
            and analytic["spanning_BVP_independent_expansion"] < 2e-4
        ),
        "flat_negative_searches": bool(
            not any(item["converged"] for item in flat_closed_trials)
            and not any(item["converged"] for item in flat_spanning_trials)
        ),
        "recovery_restart_and_corruption_rejected": bool(all(recovery.values())),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "analytic": analytic,
        "flat_closed_solver_success_count": int(sum(
            item["solver_success"] for item in flat_closed_trials
        )),
        "flat_closed_admitted_count": int(sum(
            item["converged"] for item in flat_closed_trials
        )),
        "flat_spanning_solver_success_count": int(sum(
            item["solver_success"] for item in flat_spanning_trials
        )),
        "flat_spanning_admitted_count": int(sum(
            item["converged"] for item in flat_spanning_trials
        )),
        "recovery": recovery,
    }


def recovery_adverse_control():
    with tempfile.TemporaryDirectory(prefix="test12a-recovery-") as directory:
        root = Path(directory)
        protocol = root / "protocol.md"
        protocol.write_text("sealed\n")
        source = root / "source.dat"
        source.write_text("input\n")
        manifest = root / "manifest.json"
        index = RecoveryIndex(
            manifest, protocol, {str(source): sha256_file(source)},
            maximum_stage_seconds=10.0,
        )
        index.register("stage", "json", 2.0, {"index": 0})
        index.mark_running("stage")
        restarted = RecoveryIndex(
            manifest, protocol, {str(source): sha256_file(source)},
            maximum_stage_seconds=10.0,
        )
        running_reset = restarted.data["stages"]["stage"]["status"] == "pending"
        output = root / "stage.json"
        atomic_write_json(output, {"finite": True})
        restarted.mark_complete("stage", output, 0.1)
        output.write_text("corrupt\n")
        corruption_rejected = restarted.validated_path("stage") is None
        metadata_rejected = False
        try:
            restarted.register("stage", "json", 2.0, {"index": 1})
        except RuntimeError:
            metadata_rejected = True
        return {
            "running_stage_reset_to_pending": bool(running_reset),
            "corrupted_completed_stage_rejected": bool(corruption_rejected),
            "changed_stage_metadata_rejected": bool(metadata_rejected),
        }


def load_case(initial_archive, domain, grid, phase):
    prefix = f"{domain}_{grid}"
    z = np.asarray(initial_archive[f"{prefix}_z"])
    r = np.asarray(initial_archive[f"{prefix}_r"])
    initial = np.asarray(initial_archive[f"{prefix}_position"])
    if domain == "R8":
        if phase == "pre":
            archive_path = R8_FINE
            state_prefix = f"{grid}_fine_time_3"
        elif phase == "near":
            archive_path = R8_FINE
            state_prefix = f"{grid}_fine_time_4"
        else:
            archive_path = R8_LONG
            state_prefix = f"{grid}_time_3"
    else:
        archive_path = R12_LONG
        state_prefix = f"{grid}_long_2"
    with np.load(archive_path, allow_pickle=False) as archive:
        position = initial + np.asarray(archive[f"{state_prefix}_increment"])
        velocity = np.asarray(archive[f"{state_prefix}_velocity"])
    return position, velocity, z, r


def admitted(surface):
    return bool(
        surface.get("converged", False)
        and surface.get("local_expansion_interior_maximum", np.inf) < LOCAL_LIMIT
        and surface.get("independent_expansion_interior_maximum", np.inf)
        < INDEPENDENT_LIMIT
        and surface.get("boundary_slope_error", np.inf) < LOCAL_LIMIT
    )


def capped_admitted(surface):
    crosscheck = surface.get("primary_evaluator_crosscheck", {})
    return bool(
        surface.get("converged", False)
        and surface.get("local_expansion_interior_maximum", np.inf) < LOCAL_LIMIT
        and surface.get("boundary_slope_error", np.inf) < LOCAL_LIMIT
        and "error" not in crosscheck
        and crosscheck.get("two_cell_interior_maximum", np.inf) < INDEPENDENT_LIMIT
    )


def public_surface(topology, surface, keep_profile):
    if "error" in surface:
        return {"error": surface["error"]}
    common = {
        key: surface[key] for key in (
            "converged", "solver_success", "message", "in_domain",
            "iterations", "mesh_nodes_used", "boundary_slope_error",
            "local_expansion_interior_maximum", "interior_point_count",
        ) if key in surface
    }
    if topology == "donor_capped":
        common.update({
            "rho_axis": surface["rho_axis"],
            "rho_brane": surface["rho_brane"],
            "primary_evaluator_crosscheck": surface[
                "primary_evaluator_crosscheck"
            ],
        })
        if keep_profile:
            common["profile"] = {
                key: surface[key] for key in ("theta", "rho", "slope")
            }
    elif topology == "closed_bulk":
        common.update({
            key: surface[key] for key in (
                "z_center", "z_lower_tip", "z_upper_tip", "radius_max",
                "rho_min", "rho_max", "independent_expansion_interior_maximum",
            )
        })
        if keep_profile:
            common["profile"] = {
                key: surface[key] for key in ("theta", "rho", "slope")
            }
    else:
        common.update({
            key: surface[key] for key in (
                "radius_A", "radius_B", "radius_min", "radius_max",
                "independent_expansion_interior_maximum",
            )
        })
        if keep_profile:
            common["profile"] = {
                key: surface[key] for key in ("z", "radius", "slope")
            }
    return common


def run_trial_batch(topology, seeds, center, state):
    position, velocity, z, r = state
    prepared = prepare_capped_expansion_slice(position, velocity, z, r)
    trials = []
    for seed in seeds:
        started = time.perf_counter()
        try:
            if topology == "donor_capped":
                surface = solve_dynamical_capped_surface_bvp(
                    position, velocity, z, r, seed, tolerance=2e-5,
                    nodes=121, maximum_nodes=6000, dense_nodes=501,
                    prepared=prepared,
                )
                accepted = capped_admitted(surface)
            elif topology == "closed_bulk":
                surface = solve_dynamical_closed_surface_bvp(
                    position, velocity, z, r, center, seed,
                    tolerance=2e-5, nodes=121, maximum_nodes=6000,
                    dense_nodes=601, prepared=prepared,
                )
                accepted = admitted(surface)
            else:
                surface = solve_dynamical_spanning_surface_bvp(
                    position, velocity, z, r, seed, tolerance=2e-5,
                    nodes=121, maximum_nodes=6000, dense_nodes=501,
                    prepared=prepared,
                )
                accepted = admitted(surface)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            surface = {"error": f"{type(error).__name__}: {error}"}
            accepted = False
        runtime = float(time.perf_counter() - started)
        trials.append({
            "seed": float(seed),
            "center": None if center is None else float(center),
            "admitted": bool(accepted),
            "runtime_seconds": runtime,
            "surface": public_surface(topology, surface, accepted),
        })
    return {
        "topology": topology,
        "center": center,
        "seeds": list(seeds),
        "trial_count": len(trials),
        "admitted_trial_count": int(sum(item["admitted"] for item in trials)),
        "runtime_seconds": float(sum(item["runtime_seconds"] for item in trials)),
        "trials": trials,
    }


def signature(topology, trial):
    surface = trial["surface"]
    if topology == "donor_capped":
        return np.asarray((surface["rho_axis"], surface["rho_brane"]))
    if topology == "closed_bulk":
        return np.asarray((
            surface["z_lower_tip"], surface["z_upper_tip"],
            surface["radius_max"],
        ))
    return np.asarray((
        surface["radius_A"], surface["radius_B"],
        surface["radius_min"], surface["radius_max"],
    ))


def cluster_trials(topology, trials):
    clusters = []
    ordered = sorted(
        (item for item in trials if item["admitted"]),
        key=lambda item: tuple(signature(topology, item)),
    )
    for trial in ordered:
        local = signature(topology, trial)
        destination = None
        for cluster in clusters:
            if np.linalg.norm(local - np.asarray(cluster["signature"])) < SIGNATURE_DISTANCE:
                destination = cluster
                break
        if destination is None:
            destination = {"signature": local.tolist(), "members": []}
            clusters.append(destination)
        destination["members"].append(trial)
        values = np.asarray([
            signature(topology, member) for member in destination["members"]
        ])
        destination["signature"] = np.median(values, axis=0).tolist()
    for cluster in clusters:
        center = np.asarray(cluster["signature"])
        cluster["representative"] = min(
            cluster["members"],
            key=lambda item: float(np.linalg.norm(signature(topology, item) - center)),
        )
        cluster["seed_count"] = len(cluster["members"])
    return sorted(clusters, key=lambda item: tuple(item["signature"]))


def confirmation_stage(index, case_id, topology, candidate_index, cluster, state):
    position, velocity, z, r = state
    stage_id = f"confirmation/{case_id}/{topology}/candidate_{candidate_index:02d}"
    metadata = {
        "case": case_id,
        "topology": topology,
        "candidate_index": candidate_index,
        "BVP_signature": cluster["signature"],
        "FD_nodes": 101 if topology == "closed_bulk" else 81,
    }

    def calculate():
        representative = cluster["representative"]["surface"]
        profile = representative["profile"]
        prepared = prepare_capped_expansion_slice(position, velocity, z, r)
        if topology == "closed_bulk":
            surface = solve_dynamical_closed_surface_fd(
                position, velocity, z, r, representative["z_center"], profile,
                nodes=101, tolerance=1e-9, prepared=prepared,
            )
        else:
            surface = solve_dynamical_spanning_surface_fd(
                position, velocity, z, r, profile, nodes=81,
                tolerance=1e-9, prepared=prepared,
            )
        trial = {
            "admitted": admitted(surface),
            "surface": public_surface(topology, surface, False),
        }
        fd_signature = signature(topology, trial) if trial["admitted"] else None
        component_difference = None
        if fd_signature is not None:
            component_difference = max(
                scalar_relative(left, right)
                for left, right in zip(cluster["signature"], fd_signature)
            )
        return {
            "BVP_signature": cluster["signature"],
            "FD_signature": None if fd_signature is None else fd_signature,
            "FD_admitted": trial["admitted"],
            "maximum_component_relative_difference": component_difference,
            "confirmed": bool(
                trial["admitted"] and component_difference is not None
                and component_difference < 0.005
            ),
            "FD_surface": trial["surface"],
        }

    return run_json_stage(index, stage_id, 900.0, metadata, calculate)


def matched_clusters(left, right, limit=0.02):
    if len(left) != len(right):
        return False
    unmatched = list(range(len(right)))
    for item in left:
        candidates = []
        for index in unmatched:
            comparison = right[index]
            difference = max(
                scalar_relative(a, b)
                for a, b in zip(item["signature"], comparison["signature"])
            )
            candidates.append((difference, index))
        if not candidates:
            return False
        difference, selected = min(candidates)
        if difference >= limit:
            return False
        unmatched.remove(selected)
    return not unmatched


def main():
    if not PROTOCOL.exists():
        raise FileNotFoundError(
            f"sealed protocol is required before physical search: {PROTOCOL}"
        )
    overall_started = time.perf_counter()
    index = RecoveryIndex(
        MANIFEST, PROTOCOL, expected_inputs(), maximum_stage_seconds=3600.0,
    )
    control_result = run_json_stage(
        index, "controls/all", 300.0, {"control_version": 1}, controls,
    )
    if not control_result["passed"]:
        raise RuntimeError("Test12A controls failed before physical search")
    build_initial_geometries(index)
    initial_archive = np.load(INITIAL_STATE, allow_pickle=False)
    searches = {}
    try:
        for domain, grid, phase, time_value in CASES:
            case_id = f"{domain}_{grid}_{phase}_t{time_value:.6f}"
            print(f"Test12A {case_id}", flush=True)
            state = load_case(initial_archive, domain, grid, phase)
            position, velocity, z, r = state
            batches = []
            for batch_index in range(2):
                seeds = CAP_SEEDS[6 * batch_index:6 * (batch_index + 1)]
                stage_id = f"search/{case_id}/donor_capped/batch_{batch_index:02d}"
                batches.append(run_json_stage(
                    index, stage_id, 900.0,
                    {"topology": "donor_capped", "seeds": list(seeds)},
                    lambda seeds=seeds, state=state: run_trial_batch(
                        "donor_capped", seeds, None, state,
                    ),
                ))
            cap_trials = [trial for batch in batches for trial in batch["trials"]]
            cap_clusters = cluster_trials("donor_capped", cap_trials)

            closed_batches = []
            centers = [
                float(z[0] + fraction * (z[-1] - z[0]))
                for fraction in CLOSED_CENTER_FRACTIONS
            ]
            for center_index, center in enumerate(centers):
                stage_id = f"search/{case_id}/closed_bulk/center_{center_index:02d}/batch_00"
                closed_batches.append(run_json_stage(
                    index, stage_id, 900.0,
                    {
                        "topology": "closed_bulk", "center": center,
                        "center_fraction": CLOSED_CENTER_FRACTIONS[center_index],
                        "seeds": list(CLOSED_SEEDS),
                    },
                    lambda center=center, state=state: run_trial_batch(
                        "closed_bulk", CLOSED_SEEDS, center, state,
                    ),
                ))
            closed_trials = [
                trial for batch in closed_batches for trial in batch["trials"]
            ]
            closed_clusters = cluster_trials("closed_bulk", closed_trials)

            spanning_seeds = tuple(float(value) for value in np.geomspace(
                max(3.0 * r[1], 0.18), 0.75 * r[-1], SPANNING_SEED_COUNT,
            ))
            spanning_batches = []
            for batch_index in range(3):
                seeds = spanning_seeds[6 * batch_index:6 * (batch_index + 1)]
                stage_id = f"search/{case_id}/brane_spanning/batch_{batch_index:02d}"
                spanning_batches.append(run_json_stage(
                    index, stage_id, 900.0,
                    {"topology": "brane_spanning", "seeds": list(seeds)},
                    lambda seeds=seeds, state=state: run_trial_batch(
                        "brane_spanning", seeds, None, state,
                    ),
                ))
            spanning_trials = [
                trial for batch in spanning_batches for trial in batch["trials"]
            ]
            spanning_clusters = cluster_trials("brane_spanning", spanning_trials)

            confirmations = {"closed_bulk": [], "brane_spanning": []}
            for topology, clusters in (
                ("closed_bulk", closed_clusters),
                ("brane_spanning", spanning_clusters),
            ):
                for candidate_index, cluster in enumerate(clusters):
                    confirmations[topology].append(confirmation_stage(
                        index, case_id, topology, candidate_index, cluster, state,
                    ))
            searches[case_id] = {
                "domain": domain,
                "grid": grid,
                "phase": phase,
                "time": time_value,
                "grid_shape": [len(z), len(r)],
                "donor_capped": {
                    "trial_count": len(cap_trials),
                    "admitted_trial_count": int(sum(
                        trial["admitted"] for trial in cap_trials
                    )),
                    "distinct_count": len(cap_clusters),
                    "clusters": cap_clusters,
                },
                "closed_bulk": {
                    "centers": centers,
                    "seeds": list(CLOSED_SEEDS),
                    "trial_count": len(closed_trials),
                    "admitted_trial_count": int(sum(
                        trial["admitted"] for trial in closed_trials
                    )),
                    "distinct_count": len(closed_clusters),
                    "clusters": closed_clusters,
                    "confirmations": confirmations["closed_bulk"],
                },
                "brane_spanning": {
                    "seeds": list(spanning_seeds),
                    "trial_count": len(spanning_trials),
                    "admitted_trial_count": int(sum(
                        trial["admitted"] for trial in spanning_trials
                    )),
                    "distinct_count": len(spanning_clusters),
                    "clusters": spanning_clusters,
                    "confirmations": confirmations["brane_spanning"],
                },
            }
    finally:
        initial_archive.close()

    def record(domain, grid, phase):
        return next(
            item for item in searches.values()
            if item["domain"] == domain and item["grid"] == grid
            and item["phase"] == phase
        )

    expected_caps = True
    for item in searches.values():
        target = 0 if item["phase"] == "pre" else 2
        expected_caps = expected_caps and item["donor_capped"]["distinct_count"] == target
    new_positive = any(
        item[topology]["distinct_count"] > 0
        for item in searches.values()
        for topology in ("closed_bulk", "brane_spanning")
    )
    all_confirmed = all(
        confirmation["confirmed"]
        for item in searches.values()
        for topology in ("closed_bulk", "brane_spanning")
        for confirmation in item[topology]["confirmations"]
    )
    cross_grid = {}
    for domain, phase in (("R8", "pre"), ("R8", "near"), ("R8", "persistent"), ("R12", "persistent")):
        key = f"{domain}_{phase}"
        cross_grid[key] = {}
        for topology in ("closed_bulk", "brane_spanning"):
            cross_grid[key][topology] = matched_clusters(
                record(domain, "G7", phase)[topology]["clusters"],
                record(domain, "G8", phase)[topology]["clusters"],
            )
    cross_domain = {}
    for grid in ("G7", "G8"):
        cross_domain[grid] = {}
        for topology in ("closed_bulk", "brane_spanning"):
            cross_domain[grid][topology] = matched_clusters(
                record("R8", grid, "persistent")[topology]["clusters"],
                record("R12", grid, "persistent")[topology]["clusters"],
            )
    all_transfer = bool(
        all(value for item in cross_grid.values() for value in item.values())
        and all(value for item in cross_domain.values() for value in item.values())
    )
    observed_topologies = {
        topology for topology in ("closed_bulk", "brane_spanning")
        if any(item[topology]["distinct_count"] > 0 for item in searches.values())
    }
    positive_domain_support = bool(all(
        all(
            record(domain, grid, "persistent")[topology]["distinct_count"] > 0
            for domain in ("R8", "R12") for grid in ("G7", "G8")
        )
        and cross_grid["R8_persistent"][topology]
        and cross_grid["R12_persistent"][topology]
        and cross_domain["G7"][topology]
        and cross_domain["G8"][topology]
        for topology in observed_topologies
    ))
    no_new = not new_positive
    if expected_caps and no_new:
        status = "pass"
        classification = "bounded_multitopology_pilot_no_competing_topology_detected"
    elif (
        expected_caps and new_positive and all_confirmed and all_transfer
        and positive_domain_support
    ):
        status = "pass"
        classification = "bounded_multitopology_pilot_confirmed_additional_topology"
    else:
        status = "review"
        classification = "bounded_multitopology_pilot_mixed_or_unconfirmed"
    acceptance = {
        "all_controls_pass": control_result["passed"],
        "donor_capped_reference_history_recovered": bool(expected_caps),
        "all_new_candidates_independently_confirmed": bool(all_confirmed),
        "new_topology_cross_grid_and_persistent_domain_transfer": bool(
            all_transfer and positive_domain_support if new_positive else True
        ),
        "all_recovery_stages_content_valid": bool(all(
            index.validated_path(stage_id) is not None
            for stage_id, stage in index.data["stages"].items()
            if stage["status"] == "complete"
        )),
    }
    payload = {
        "status": status,
        "classification": classification,
        "scope": (
            "bounded dynamic search of donor-capped, closed star-shaped S3, "
            "and brane-spanning S2xI surfaces on archived corrected A=7.90 slices"
        ),
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "amplitude": AMPLITUDE,
        "cases": [list(item) for item in CASES],
        "coverage": {
            "donor_capped_seeds": list(CAP_SEEDS),
            "closed_center_fractions": list(CLOSED_CENTER_FRACTIONS),
            "closed_radius_seeds": list(CLOSED_SEEDS),
            "spanning_seed_rule": (
                "18 geometric seeds from max(3*dr,0.18) through 0.75*r_max"
            ),
        },
        "thresholds": {
            "local_expansion": LOCAL_LIMIT,
            "independent_expansion": INDEPENDENT_LIMIT,
            "deduplication_signature_distance": SIGNATURE_DISTANCE,
            "FD_BVP_component_relative_difference": 0.005,
            "cross_grid_and_domain_component_relative_difference": 0.02,
        },
        "controls": control_result,
        "searches": searches,
        "new_topology_detected": bool(new_positive),
        "observed_new_topology_classes": sorted(observed_topologies),
        "positive_persistent_domain_support": positive_domain_support,
        "cross_grid": cross_grid,
        "persistent_cross_domain": cross_domain,
        "acceptance": acceptance,
        "null_scope": (
            "A zero count means no candidate was detected within the stated "
            "parameterizations, grids, domains, slices, and seed coverage; it "
            "is not a topology nonexistence theorem."
        ),
        "claim_boundary": (
            "A positive additional topology is physics-admissible only after "
            "independent FD confirmation and the declared cross-grid/domain "
            "transfer. This pilot does not establish global outermostness, "
            "topology change, an event horizon, a throat, or mass transfer."
        ),
        "manifest": str(MANIFEST),
        "runtime_seconds": float(time.perf_counter() - overall_started),
    }
    atomic_write_json(OUTPUT, builtin(payload))
    print(json.dumps({
        "status": status,
        "classification": classification,
        "new_topology_detected": new_positive,
        "runtime_seconds": payload["runtime_seconds"],
        "output": str(OUTPUT),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
