#!/usr/bin/env python3
"""Complete-branch, expansion, stability, and formation lane for Test 2D."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.interpolate import CubicSpline, PchipInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import run_corrected_A790_test2b_invariant_convergence as old
from bhps.dynamical_capped_geometry import capped_surface_geometry
from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice
from bhps.dynamical_capped_horizon_bvp import (
    _sample_geometry,
    dynamical_rho_second,
    local_outgoing_expansion,
    solve_dynamical_capped_surface_bvp,
)
from bhps.high_order_invariant_interpolation import (
    INTERPOLATION_ORDER_GATE,
    SAFETY_FACTOR,
    endpoint_preserving_indices,
    interpolate,
)
from bhps.invariant_physical_chart import conservative_order_interval, sign_coherence
from bhps.dynamical_mots_stability import (
    finite_difference_matrix,
    neumann_extension,
    physical_normal_factor,
)
from bhps.ragged_normal_arclength_chart import inverse_ragged_chart, load_ragged_chart
from bhps.recovery_indexer import RecoveryIndex, atomic_write_json, atomic_write_npz, sha256_file, validate_npz
from run_corrected_A790_dynamic_MOTS_stability import surface_passes


PROTOCOL = Path("notes/110_A790_test2D_high_order_ragged_chart_protocol.md")
PROTOCOL_SHA256 = "f11989b23cff2d5b87bf6e730ff91c77b1e095eb2a7a49d1a277a9d3bf2666e5"
MACHINE_PROTOCOL = Path("results/corrected_A790_test2d_high_order_ragged_chart_protocol.json")
MACHINE_PROTOCOL_SHA256 = "d3a1f59c61499d5e8f67fcd29a4592a7c1afd65b74e7bc90d78d0a35121734e4"
CHART_MANIFEST = Path("results/corrected_A790_test2d_high_order_ragged_chart_recovery/chart_index_v2.json")
CHART_MANIFEST_SHA256 = None
ALIGNED = Path("results/corrected_A790_test2b_invariant_convergence_recovery/aligned_states.npz")
ALIGNED_SHA256 = "42ec6dcb65038a6aa7ab1fc724e848bbba7be781b887cd18f3512cc030256265"
GEOMETRIES = Path("results/corrected_A790_test2b_invariant_convergence_recovery/geometries.npz")
GEOMETRIES_SHA256 = "fd5cf03f1fb37b9e9f7d1f14f8112c2452ccb4f0f6d8438d251d1432bb2dd618"
RAGGED_ROOT = Path("results/corrected_A790_test2d_high_order_ragged_chart_recovery")
ROOT = Path("results/corrected_A790_test2d_high_order_ragged_chart_surface_recovery")
MANIFEST = ROOT / "index.json"
OUTPUT = Path("results/corrected_A790_test2d_high_order_ragged_chart_surfaces.json")
SPATIAL = ("G9", "G10", "G11")
TEMPORAL = ("G10_coarse", "G10_standard", "G10_half")
BRANCHES = ("inner", "outer")
SEEDS = {"inner": 1.30, "outer": 1.55}


def recovery_inputs():
    if not isinstance(CHART_MANIFEST_SHA256, str):
        raise RuntimeError("surface runner is not sealed to a completed chart manifest")
    fixed = {
        str(MACHINE_PROTOCOL): MACHINE_PROTOCOL_SHA256,
        str(CHART_MANIFEST): CHART_MANIFEST_SHA256,
        str(ALIGNED): ALIGNED_SHA256, str(GEOMETRIES): GEOMETRIES_SHA256,
    }
    dynamic = (
        Path(__file__), Path("src/bhps/high_order_invariant_interpolation.py"),
        Path("src/bhps/ragged_normal_arclength_chart.py"),
        Path("src/bhps/recovery_indexer.py"),
    )
    return {**fixed, **{str(path): sha256_file(path) for path in dynamic}}


def stage_json(index, stage_id, filename, kind, metadata, producer, expected=7200.0):
    path = ROOT / filename
    index.register(stage_id, kind, expected, metadata)
    cached = index.validated_path(stage_id)
    if cached is not None:
        return json.loads(cached.read_text())
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        payload = {"stage_id": stage_id, "protocol_sha256": index.protocol_sha256, **producer()}
        atomic_write_json(path, payload)
        index.mark_complete(stage_id, path, time.perf_counter() - started)
        return payload
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def load_native(label):
    grid = label if label in SPATIAL else "G10"
    with np.load(GEOMETRIES) as geometry, np.load(ALIGNED) as aligned:
        return (
            np.asarray(geometry[f"{grid}_z"]), np.asarray(geometry[f"{grid}_r"]),
            np.asarray(aligned[f"{label}_position"]),
            np.asarray(aligned[f"{label}_velocity"]),
        )


def chart(label, resolution):
    return load_ragged_chart(
        RAGGED_ROOT / f"chart_aligned_{label}_pchip_{resolution}.npz"
    )


def _profile_norm(values):
    values = np.asarray(values, dtype=float)
    return {
        "L2": float(np.sqrt(simpson(values**2, x=np.linspace(0.0, 1.0, len(values))))),
        "q95": float(np.quantile(np.abs(values), 0.95)),
    }


def _termination_margin(inverse, chart):
    """Distance, in chart rows, from the ray-specific terminal boundary.

    The protocol exempts the brane and axis boundaries.  The inverse mapper's
    generic ``boundary_margin`` also includes those boundaries, so it cannot be
    used for this gate on a capped surface whose endpoints lie on them.
    """
    return (1.0 - np.asarray(inverse.normalized_depth)) * (chart.shape[0] - 1)


def _stability_collar_points(prepared, surface):
    """Return every distinct off-surface point queried by the fixed stability audit."""
    source_theta = np.asarray(surface["theta"], dtype=float)
    source_rho = np.asarray(surface["rho"], dtype=float)
    curve = CubicSpline(
        source_theta, source_rho, bc_type=((1, 0.0), (1, 0.0)),
    )
    points = []
    for nodes, relative_step in ((49, 1e-5), (65, 1e-5), (81, 1e-5), (81, 2e-5)):
        theta = np.linspace(1e-4, np.pi / 2.0, nodes)
        rho = curve(theta)
        first_matrix = finite_difference_matrix(theta, 1, 7)
        extension = neumann_extension(first_matrix)
        normal_factor = physical_normal_factor(
            prepared, theta, rho, first_matrix @ rho,
        )
        deformation = extension / normal_factor[1:-1][None, :]
        physical_step = float(relative_step) * max(1.0, float(np.mean(rho)))
        for sign in (-1.0, 1.0):
            displaced = rho[:, None] + sign * physical_step * deformation
            native_z = prepared.z[-1] - displaced * np.cos(theta)[:, None]
            native_r = displaced * np.sin(theta)[:, None]
            points.append(np.column_stack((native_z.ravel(), native_r.ravel())))
    # The dense column construction repeats unchanged collocation points.  An
    # exact row-wise unique keeps the query set complete while avoiding tens
    # of thousands of redundant global triangle searches.
    return np.unique(np.concatenate(points, axis=0), axis=0)


def profile_and_coverage(position, velocity, z, r, surface, fine_chart, primary_chart):
    prepared = prepare_capped_expansion_slice(position, velocity, z, r)
    theta, rho, slope = (np.asarray(surface[key]) for key in ("theta", "rho", "slope"))
    second = dynamical_rho_second(prepared, theta, rho, slope)
    sampled = _sample_geometry(prepared, theta, rho, slope)
    outgoing = local_outgoing_expansion(prepared, theta, rho, slope, second)
    tangent_K = np.einsum(
        "...a,...ab,...b->...", sampled["tangent"], sampled["extrinsic"], sampled["tangent"],
    )
    ingoing = -outgoing + 2.0 * (-tangent_K - 2.0 * sampled["sphere_extrinsic"])
    native_z = z[-1] - rho * np.cos(theta)
    native_r = rho * np.sin(theta)
    fine = inverse_ragged_chart(fine_chart, native_z, native_r, candidate_count=256)
    primary = inverse_ragged_chart(primary_chart, native_z, native_r, candidate_count=256)
    collar_points = _stability_collar_points(prepared, surface)
    fine_collar = inverse_ragged_chart(
        fine_chart, collar_points[:, 0], collar_points[:, 1], candidate_count=256,
    )
    primary_collar = inverse_ragged_chart(
        primary_chart, collar_points[:, 0], collar_points[:, 1], candidate_count=256,
    )
    domain_diagonal = math.hypot(float(z[-1] - z[0]), float(r[-1] - r[0]))
    coverage = {
        "profile_points": len(theta),
        "inverse_residual_limit": 1e-8 * domain_diagonal,
        "fine_unique_points": int(np.sum(fine.root_count == 1)),
        "primary_unique_points": int(np.sum(primary.root_count == 1)),
        "stability_collar_points": int(len(collar_points)),
        "fine_unique_stability_collar_points": int(np.sum(fine_collar.root_count == 1)),
        "primary_unique_stability_collar_points": int(np.sum(primary_collar.root_count == 1)),
        "fine_minimum_termination_margin": float(np.min(_termination_margin(fine, fine_chart))),
        "primary_minimum_termination_margin": float(np.min(_termination_margin(primary, primary_chart))),
        "fine_collar_minimum_termination_margin": float(np.min(
            _termination_margin(fine_collar, fine_chart)
        )),
        "primary_collar_minimum_termination_margin": float(np.min(
            _termination_margin(primary_collar, primary_chart)
        )),
        "fine_inverse_residual_maximum": float(np.max(fine.residual)),
        "primary_inverse_residual_maximum": float(np.max(primary.residual)),
        "fine_collar_inverse_residual_maximum": float(np.max(fine_collar.residual)),
        "primary_collar_inverse_residual_maximum": float(np.max(primary_collar.residual)),
    }
    coverage["passed"] = bool(
        coverage["fine_unique_points"] == len(theta)
        and coverage["primary_unique_points"] == len(theta)
        and coverage["fine_unique_stability_collar_points"] == len(collar_points)
        and coverage["primary_unique_stability_collar_points"] == len(collar_points)
        and coverage["fine_minimum_termination_margin"] >= 4.0
        and coverage["primary_minimum_termination_margin"] >= 4.0
        and coverage["fine_collar_minimum_termination_margin"] >= 4.0
        and coverage["primary_collar_minimum_termination_margin"] >= 4.0
        and coverage["fine_inverse_residual_maximum"] < 1e-8 * domain_diagonal
        and coverage["primary_inverse_residual_maximum"] < 1e-8 * domain_diagonal
        and coverage["fine_collar_inverse_residual_maximum"] < 1e-8 * domain_diagonal
        and coverage["primary_collar_inverse_residual_maximum"] < 1e-8 * domain_diagonal
    )

    sphere_primary = interpolate(position[..., 3], z, r, native_z, native_r, "quintic")
    sphere_independent = interpolate(position[..., 3], z, r, native_z, native_r, "independent5")
    sphere_cubic = interpolate(position[..., 3], z, r, native_z, native_r, "cubic")
    R = native_r * np.sqrt(sphere_primary)
    R55 = native_r * np.sqrt(sphere_independent)
    R53 = native_r * np.sqrt(sphere_cubic)
    speed = np.sqrt(sampled["speed_squared"])
    length = np.concatenate(([0.0], cumulative_trapezoid(speed, x=theta)))
    normalized = length / length[-1]
    target = np.linspace(0.0, 1.0, 501)
    pchip = lambda value: PchipInterpolator(normalized, value)(target)
    profile = {
        "s": target, "D": pchip(fine.distance), "S": pchip(fine.arclength),
        "D_primary": pchip(primary.distance), "S_primary": pchip(primary.arclength),
        "R": pchip(R), "R_independent5": pchip(R55), "R_cubic": pchip(R53),
        "theta_minus": pchip(ingoing), "theta_plus": pchip(outgoing),
        "native_z": pchip(native_z), "native_r": pchip(native_r),
    }
    R_envelope = {"2": {"L2": 0.0, "q95": 0.0}, "4": {"L2": 0.0, "q95": 0.0}}
    R_records = {"2": [], "4": []}
    for stride in (2, 4):
        for offset_z in range(stride):
            iz = endpoint_preserving_indices(len(z), stride, offset_z)
            for offset_r in range(stride):
                ir = endpoint_preserving_indices(len(r), stride, offset_r)
                source = position[..., 3][np.ix_(iz, ir)]
                sphere_variant = interpolate(source, z[iz], r[ir], native_z, native_r, "quintic")
                R_variant = pchip(native_r * np.sqrt(sphere_variant))
                norms = _profile_norm(profile["R"] - R_variant)
                R_records[str(stride)].append({
                    "offset_z": offset_z, "offset_r": offset_r, **norms,
                })
                for norm in ("L2", "q95"):
                    R_envelope[str(stride)][norm] = max(
                        R_envelope[str(stride)][norm], norms[norm],
                    )
    R_allowance = {"admissible": True, "norms": {}, "records": R_records}
    for norm in ("L2", "q95"):
        e2, e4 = R_envelope["2"][norm], R_envelope["4"][norm]
        order = math.log2(e4 / e2) if e2 > 0.0 and e4 > e2 else None
        admitted = order is not None and order > INTERPOLATION_ORDER_GATE
        richardson = e2 / (2.0**order - 1.0) if admitted else math.inf
        delta55 = _profile_norm(profile["R"] - profile["R_independent5"])[norm]
        delta53 = _profile_norm(profile["R"] - profile["R_cubic"])[norm]
        allowance = SAFETY_FACTOR * max(richardson, delta55, delta53)
        R_allowance["admissible"] = bool(R_allowance["admissible"] and admitted and np.isfinite(allowance))
        R_allowance["norms"][norm] = {
            "e2": e2, "e4": e4, "order": order, "richardson": richardson,
            "production_vs_independent_degree5": delta55,
            "quintic_vs_cubic": delta53, "allowance": allowance,
        }
    return profile, coverage, R_allowance


def surface_stage(index, label, branch):
    def compute():
        z, r, position, velocity = load_native(label)
        surface = solve_dynamical_capped_surface_bvp(
            position, velocity, z, r, SEEDS[branch],
            tolerance=2e-5, nodes=121, maximum_nodes=6000, dense_nodes=501,
        )
        if not surface_passes(surface):
            return {
                "native_branch_found": False,
                "surface": {key: surface.get(key) for key in (
                    "converged", "in_domain", "rho_axis", "rho_brane",
                    "boundary_slope_error", "local_expansion_interior_maximum",
                    "primary_evaluator_crosscheck",
                )},
                "coverage": {"passed": False},
            }
        profile, coverage, R_allowance = profile_and_coverage(
            position, velocity, z, r, surface, chart(label, "fine"), chart(label, "primary"),
        )
        profile_path = ROOT / f"profile_{label}_{branch}.npz"
        atomic_write_npz(profile_path, **profile)
        validate_npz(profile_path)
        stability = old._stability_record(position, velocity, z, r, surface)
        return {
            "native_branch_found": True,
            "surface": {key: surface[key] for key in (
                "converged", "in_domain", "rho_axis", "rho_brane",
                "boundary_slope_error", "local_expansion_interior_maximum",
                "primary_evaluator_crosscheck",
            )},
            "coverage": coverage, "R_interpolation": R_allowance,
            "geometry": capped_surface_geometry(position, velocity, z, r, surface),
            "stability": stability, "profile_archive": str(profile_path),
            "profile_sha256": sha256_file(profile_path),
        }
    return stage_json(
        index, f"surface/{label}/{branch}", f"surface_{label}_{branch}.json",
        "test2d-complete-branch-expansion-stability",
        {"label": label, "branch": branch, "seed": SEEDS[branch]},
        lambda: {"record": compute()},
    )["record"]


def load_surface(label, branch):
    return json.loads((ROOT / f"surface_{label}_{branch}.json").read_text())["record"]


def load_profile(record):
    with np.load(record["profile_archive"]) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def profile_score(records, key):
    profiles = {label: load_profile(records[label]) for label in SPATIAL}
    differences = {"L2": [], "q95": []}
    uncertainties = {"L2": [], "q95": []}
    for left, right in zip(SPATIAL[:-1], SPATIAL[1:]):
        norms = _profile_norm(profiles[left][key] - profiles[right][key])
        state_uncertainty = {}
        for label in (left, right):
            if key in ("D", "S"):
                state_uncertainty[label] = _profile_norm(
                    profiles[label][key] - profiles[label][f"{key}_primary"]
                )
            elif key == "R":
                state_uncertainty[label] = {
                    norm: records[label]["R_interpolation"]["norms"][norm]["allowance"]
                    for norm in ("L2", "q95")
                }
            else:
                state_uncertainty[label] = {"L2": 0.0, "q95": 0.0}
        for norm in ("L2", "q95"):
            differences[norm].append(norms[norm])
            uncertainties[norm].append(
                state_uncertainty[left][norm] + state_uncertainty[right][norm]
            )
    orders, norm_pass = {}, {}
    for norm in ("L2", "q95"):
        order = conservative_order_interval(
            differences[norm][0], uncertainties[norm][0],
            differences[norm][1], uncertainties[norm][1],
        )
        orders[norm] = order
        norm_pass[norm] = bool(
            differences[norm][1] + uncertainties[norm][1]
            < max(differences[norm][0] - uncertainties[norm][0], 0.0)
            and order is not None and order[0] > 1.0
            and uncertainties[norm][1] / max(differences[norm][1], 1e-300) < 0.25
        )
    coherence = sign_coherence(
        profiles["G9"][key] - profiles["G10"][key],
        profiles["G10"][key] - profiles["G11"][key],
        np.ones_like(profiles["G9"][key]),
    )
    interpolation_admissible = bool(
        key != "R" or all(records[label]["R_interpolation"]["admissible"] for label in SPATIAL)
    )
    passed = bool(
        interpolation_admissible and all(norm_pass.values())
        and coherence is not None and coherence >= 0.70
    )
    return {
        "differences": differences, "uncertainties": uncertainties,
        "order_intervals": orders, "norm_passed": norm_pass,
        "sign_coherence": coherence,
        "interpolation_admissible": interpolation_admissible, "passed": passed,
    }


def branch_score(branch):
    records = {label: load_surface(label, branch) for label in SPATIAL}
    lost = [label for label in SPATIAL if not records[label].get("native_branch_found", False)]
    if lost:
        return {
            "records": records, "branch_loss": True,
            "lost_on_grids": lost, "passed": False,
        }
    geometry = {}
    for key in (
        "one_sided_cap_area", "equivalent_area_radius",
        "proper_meridional_length", "maximum_sphere_radius",
    ):
        geometry[key] = old.scalar_tail([records[label]["geometry"][key] for label in SPATIAL])
    eigenvalues = [records[label]["stability"]["eigenvalue"] for label in SPATIAL]
    eigen_errors = [
        max(records[label]["stability"]["angular_error"], records[label]["stability"]["step_error"])
        for label in SPATIAL
    ]
    stability = old.scalar_tail(eigenvalues, eigen_errors)
    gap = old.scalar_tail([records[label]["stability"]["spectral_gap"] for label in SPATIAL])
    expected_class = "outward_unstable" if branch == "inner" else "outward_stable"
    classifications = [records[label]["stability"]["classification"] for label in SPATIAL]
    nodal = [
        records[label]["stability"]["spectra"]["81"]["principal_eigenfunction_sign_changes"]
        for label in SPATIAL
    ]
    stability_gate = bool(
        stability["passed"] and gap["passed"]
        and all(value == expected_class for value in classifications)
        and all(value == 0 for value in nodal)
    )
    profiles = {
        key: profile_score(records, key) for key in ("D", "S", "R", "theta_minus")
    }
    coverage = bool(all(records[label]["coverage"]["passed"] for label in SPATIAL))
    passed = bool(
        coverage and all(item["passed"] for item in geometry.values())
        and stability_gate and all(item["passed"] for item in profiles.values())
    )
    return {
        "records": records, "branch_loss": False, "coverage_passed": coverage,
        "geometry": geometry, "stability": stability, "spectral_gap": gap,
        "stability_classifications": classifications,
        "principal_mode_sign_changes": nodal,
        "stability_gate": stability_gate, "profiles": profiles, "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-surfaces", type=int, default=None)
    args = parser.parse_args()
    if sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("sealed Test-2D protocol hash mismatch")
    chart_manifest = json.loads(CHART_MANIFEST.read_text())
    chart_records = [value for key, value in chart_manifest["stages"].items() if key.startswith("chart/")]
    if len(chart_records) != 90 or not all(item.get("status") == "complete" for item in chart_records):
        raise RuntimeError("all 90 Test-2D charts must complete before native surface detection")
    if not all(item.get("completion_metadata", {}).get("validity", {}).get("valid") for item in chart_records):
        raise RuntimeError("a required Test-2D chart is invalid")
    ROOT.mkdir(parents=True, exist_ok=True)
    index = RecoveryIndex(MANIFEST, PROTOCOL, recovery_inputs(), maximum_stage_seconds=7200.0)
    tasks = [(label, branch) for label in (*SPATIAL, *TEMPORAL) for branch in BRANCHES]
    pending = [task for task in tasks if index.validated_path(f"surface/{task[0]}/{task[1]}") is None]
    if args.max_new_surfaces is not None:
        pending = pending[:max(args.max_new_surfaces, 0)]
    for label, branch in pending:
        record = surface_stage(index, label, branch)
        print(json.dumps({
            "label": label, "branch": branch,
            "coverage": record["coverage"]["passed"], "complete": True,
        }), flush=True)
    if any(index.validated_path(f"surface/{label}/{branch}") is None for label, branch in tasks):
        print(json.dumps({
            "phase": "surfaces", "complete": len(tasks) - len([
                task for task in tasks if index.validated_path(f"surface/{task[0]}/{task[1]}") is None
            ]), "total": len(tasks), "physical_verdict": None,
        }, indent=2))
        return
    branches = {branch: branch_score(branch) for branch in BRANCHES}
    temporal_coverage = {
        label: {
            branch: load_surface(label, branch)["coverage"] for branch in BRANCHES
        } for label in TEMPORAL
    }
    formation = old.formation_analysis(ALIGNED)
    coverage_pass = bool(all(
        item.get("passed", False) for branches_record in temporal_coverage.values()
        for item in branches_record.values()
    ))
    passed = bool(all(item["passed"] for item in branches.values()) and coverage_pass and formation["passed"])
    branch_loss = bool(any(item.get("branch_loss", False) for item in branches.values()))
    stability_reversal = bool(any(
        item.get("branch_loss") is False and any(
            value == ("outward_stable" if branch == "inner" else "outward_unstable")
            for value in item["stability_classifications"]
        ) for branch, item in branches.items()
    ))
    status = "PASS" if passed else "FAIL" if branch_loss or stability_reversal else "REVIEW"
    result = {
        "protocol": str(PROTOCOL), "protocol_sha256": PROTOCOL_SHA256,
        "status": status,
        "classification": (
            "test2d_complete_branch_continuum_evidence"
            if passed else "test2d_branch_lane_resolved_loss_or_reversal"
            if status == "FAIL" else "test2d_branch_lane_convergence_mixed"
        ),
        "branch_loss": branch_loss, "resolved_stability_reversal": stability_reversal,
        "branches": branches, "temporal_coverage": temporal_coverage,
        "formation": formation, "passed": passed,
    }
    atomic_write_json(OUTPUT, result)
    stage_json(
        index, "surface/result", "result.json", "test2d-surface-result", {},
        lambda: {"status": result["status"], "classification": result["classification"],
                 "output": str(OUTPUT), "output_sha256": sha256_file(OUTPUT)},
    )
    print(json.dumps({"status": result["status"], "classification": result["classification"]}, indent=2))


if __name__ == "__main__":
    main()
