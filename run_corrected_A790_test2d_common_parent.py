#!/usr/bin/env python3
"""Ragged-chart common-parent restriction control for sealed Test 2D."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import run_corrected_A790_test2b_invariant_convergence as old
from bhps.corrected_A790_physical_tensor_convergence import adm_extrinsic_curvature_tensor
from bhps.dynamical_capped_horizon_bvp import solve_dynamical_capped_surface_bvp
from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice
from bhps.dynamical_mots_stability import (
    finite_difference_matrix,
    neumann_extension,
    physical_normal_factor,
)
from bhps.high_order_invariant_interpolation import interpolate, mapped_extrinsic_fields
from bhps.ragged_normal_arclength_chart import (
    inverse_ragged_chart,
    load_ragged_chart,
    ragged_chart_to_native,
)
from bhps.recovery_indexer import RecoveryIndex, atomic_write_json, sha256_file
from run_corrected_A790_dynamic_MOTS_stability import surface_passes


PROTOCOL = Path("notes/110_A790_test2D_high_order_ragged_chart_protocol.md")
PROTOCOL_SHA256 = "f11989b23cff2d5b87bf6e730ff91c77b1e095eb2a7a49d1a277a9d3bf2666e5"
MACHINE_PROTOCOL = Path("results/corrected_A790_test2d_high_order_ragged_chart_protocol.json")
MACHINE_PROTOCOL_SHA256 = "d3a1f59c61499d5e8f67fcd29a4592a7c1afd65b74e7bc90d78d0a35121734e4"
CHART_MANIFEST = Path("results/corrected_A790_test2d_high_order_ragged_chart_recovery/chart_index_v2.json")
CHART_MANIFEST_SHA256 = None
SOURCE = Path("results/corrected_A790_test10b_domain_normalized_state.npz")
SOURCE_SHA256 = "b19e7e8901fdade6abb485c65a3512bf327a86f13974fb9a8c0876340a503928"
RAGGED_ROOT = Path("results/corrected_A790_test2d_high_order_ragged_chart_recovery")
ROOT = Path("results/corrected_A790_test2d_high_order_ragged_chart_common_parent_recovery")
MANIFEST = ROOT / "index.json"
OUTPUT = Path("results/corrected_A790_test2d_high_order_ragged_chart_common_parent.json")
SEEDS = {"inner": 1.30, "outer": 1.55}


def recovery_inputs():
    if not isinstance(CHART_MANIFEST_SHA256, str):
        raise RuntimeError("common-parent runner is not sealed to a completed chart manifest")
    fixed = {
        str(MACHINE_PROTOCOL): MACHINE_PROTOCOL_SHA256,
        str(CHART_MANIFEST): CHART_MANIFEST_SHA256, str(SOURCE): SOURCE_SHA256,
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


def source_record(grid, domain):
    with np.load(SOURCE) as archive:
        prefix = f"{grid}_R{domain}"
        return {
            "z": np.asarray(archive[f"{grid}_R12_z"]),
            "r": np.asarray(archive[f"{prefix}_r"]),
            "initial": np.asarray(archive[f"{prefix}_initial"]),
            "position": np.asarray(archive[f"{prefix}_step_016_position"]),
            "velocity": np.asarray(archive[f"{prefix}_step_016_velocity"]),
        }


def chart(grid, domain, slice_name, resolution):
    return load_ragged_chart(
        RAGGED_ROOT / f"chart_common_parent_{grid}_R{domain}_{slice_name}_{resolution}.npz"
    )


def _metric4(mapped):
    value = np.zeros((*mapped["covariant"].shape[:2], 4, 4))
    value[..., :2, :2] = mapped["covariant"]
    value[..., 2, 2] = value[..., 3, 3] = 1.0
    return value


def _extrinsic4(mapped):
    value = np.zeros((*mapped["K_DD"].shape, 4, 4))
    value[..., 0, 0] = mapped["K_DD"]
    value[..., 0, 1] = value[..., 1, 0] = mapped["K_DS"]
    value[..., 1, 1] = mapped["K_SS"]
    value[..., 2, 2] = value[..., 3, 3] = mapped["K_Omega"]
    return value


def domain_coordinates(grid, domain, D, S, resolution):
    SS = np.broadcast_to(S[None, :], D.shape)
    initial = chart(grid, domain, "initial", resolution)
    final = chart(grid, domain, "final", resolution)
    return (*ragged_chart_to_native(initial, D, SS), *ragged_chart_to_native(final, D, SS))


def _mapped_metric_ragged(metric, sphere, z, r, U, S, D, native_z, native_r, method):
    native_metric = interpolate(metric, z, r, native_z, native_r, method)
    native_sphere = interpolate(sphere, z, r, native_z, native_r, method)
    dmax = D[-1] / U[-1]
    dmax_S = np.gradient(dmax, S, edge_order=2)
    x_U = np.stack((
        np.gradient(native_z, U, axis=0, edge_order=2),
        np.gradient(native_r, U, axis=0, edge_order=2),
    ), axis=-1)
    x_S_at_U = np.stack((
        np.gradient(native_z, S, axis=1, edge_order=2),
        np.gradient(native_r, S, axis=1, edge_order=2),
    ), axis=-1)
    x_D = x_U / dmax[None, :, None]
    x_S = x_S_at_U - x_U * (
        U[:, None] * dmax_S[None, :] / dmax[None, :]
    )[..., None]
    covariant = np.empty((*D.shape, 2, 2))
    covariant[..., 0, 0] = np.einsum("...a,...ab,...b->...", x_D, native_metric, x_D)
    covariant[..., 0, 1] = covariant[..., 1, 0] = np.einsum(
        "...a,...ab,...b->...", x_D, native_metric, x_S,
    )
    covariant[..., 1, 1] = np.einsum("...a,...ab,...b->...", x_S, native_metric, x_S)
    determinant = np.linalg.det(covariant)
    if np.min(native_sphere) <= 0.0 or np.min(determinant) <= 0.0:
        raise ValueError("common-parent ragged metric is nonpositive")
    native_areal_radius = native_r * np.sqrt(native_sphere)
    return {
        "covariant": covariant,
        "native_metric": native_metric,
        "basis_D": x_D,
        "basis_S": x_S,
        "native_areal_radius": native_areal_radius,
        "volume_density": 4.0 * math.pi * native_areal_radius**2 * np.sqrt(determinant),
    }


def map_record(record, U, S, D, resolution="fine", method="quintic"):
    z, r = record["z"], record["r"]
    zi, ri, zf, rf = domain_coordinates(record["grid"], record["domain"], D, S, resolution)
    initial_metric, initial_sphere = old.reduced_metric(record["initial"], r)
    final_metric, final_sphere = old.reduced_metric(record["position"], r)
    initial = _mapped_metric_ragged(
        initial_metric, initial_sphere, z, r, U, S, D, zi, ri, method,
    )
    final = _mapped_metric_ragged(
        final_metric, final_sphere, z, r, U, S, D, zf, rf, method,
    )
    native_K = adm_extrinsic_curvature_tensor(record["position"], record["velocity"], z, r)
    extrinsic = mapped_extrinsic_fields(
        native_K, final_sphere, z, r, zf, rf, final, method,
    )
    initial4, final4 = _metric4(initial), _metric4(final)
    return {
        "initial_metric": initial4, "final_metric": final4,
        "metric_increment": final4 - initial4, "ADM_K": _extrinsic4(extrinsic),
        "weight": final["volume_density"], "dmax": D[-1] / U[-1],
    }


def comparison_domain(grid):
    charts = [
        chart(grid, domain, slice_name, "fine")
        for domain in (8, 10, 12) for slice_name in ("initial", "final")
    ]
    spacing = max(float(np.max(np.diff(item.arclength))) for item in charts)
    S = np.linspace(2.0 * spacing, min(float(item.arclength[-1]) for item in charts), 193)
    dmax = np.min(np.stack([
        np.interp(S, item.arclength, item.maximum_distance) for item in charts
    ]), axis=0)
    U = np.linspace(0.0, 0.95, 129)
    D = U[:, None] * dmax[None, :]
    return U, S, D


def paired_summary(left, right, U, S):
    difference = np.asarray(left["value"] - right["value"])
    dmax = np.minimum(np.asarray(left["dmax"]), np.asarray(right["dmax"]))
    weight = 0.5 * (np.asarray(left["weight"]) + np.asarray(right["weight"]))
    component_axes = tuple(range(2, difference.ndim))
    squared = np.sum(difference**2, axis=component_axes)
    def integral(values):
        return float(simpson(simpson(values * dmax[None, :], x=S, axis=1), x=U))
    absolute = math.sqrt(max(integral(squared * weight), 0.0))
    left_squared = np.sum(np.asarray(left["value"])**2, axis=component_axes)
    right_squared = np.sum(np.asarray(right["value"])**2, axis=component_axes)
    left_norm = math.sqrt(max(integral(left_squared * weight), 0.0))
    right_norm = math.sqrt(max(integral(right_squared * weight), 0.0))
    pointwise = np.sqrt(squared)
    point_left, point_right = np.sqrt(left_squared), np.sqrt(right_squared)
    quantile_weight = weight * dmax[None, :]
    q95 = old.weighted_quantile(pointwise, quantile_weight)
    scale95 = max(
        old.weighted_quantile(point_left, quantile_weight),
        old.weighted_quantile(point_right, quantile_weight), 1e-300,
    )
    return {
        "absolute_L2": absolute,
        "relative_L2": absolute / max(left_norm, right_norm, 1e-300),
        "weighted_q95": q95,
        "relative_weighted_q95": q95 / scale95,
    }


def _termination_margin(inverse, active_chart):
    """Distance in chart rows to the ray-specific terminal boundary only."""
    return (1.0 - np.asarray(inverse.normalized_depth)) * (active_chart.shape[0] - 1)


def _stability_collar_points(prepared, surface):
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
        factor = physical_normal_factor(prepared, theta, rho, first_matrix @ rho)
        deformation = extension / factor[1:-1][None, :]
        step = float(relative_step) * max(1.0, float(np.mean(rho)))
        for sign in (-1.0, 1.0):
            displaced = rho[:, None] + sign * step * deformation
            points.append(np.column_stack((
                (prepared.z[-1] - displaced * np.cos(theta)[:, None]).ravel(),
                (displaced * np.sin(theta)[:, None]).ravel(),
            )))
    return np.unique(np.concatenate(points, axis=0), axis=0)


def branch_coverage(record, branch):
    z, r = record["z"], record["r"]
    surface = solve_dynamical_capped_surface_bvp(
        record["position"], record["velocity"], z, r, SEEDS[branch],
        tolerance=2e-5, nodes=121, maximum_nodes=6000, dense_nodes=501,
    )
    if not surface_passes(surface):
        return {"native_surface_valid": False, "passed": False}
    theta, rho = np.asarray(surface["theta"]), np.asarray(surface["rho"])
    native_z, native_r = z[-1] - rho * np.cos(theta), rho * np.sin(theta)
    collar_points = _stability_collar_points(
        prepare_capped_expansion_slice(record["position"], record["velocity"], z, r),
        surface,
    )
    charts = {
        resolution: chart(record["grid"], record["domain"], "final", resolution)
        for resolution in ("primary", "fine")
    }
    inverses = {
        resolution: inverse_ragged_chart(active_chart, native_z, native_r, candidate_count=256)
        for resolution, active_chart in charts.items()
    }
    collar_inverses = {
        resolution: inverse_ragged_chart(
            active_chart, collar_points[:, 0], collar_points[:, 1], candidate_count=256,
        )
        for resolution, active_chart in charts.items()
    }
    diagnostics = {}
    passed = True
    for resolution, inverse in inverses.items():
        active_chart = charts[resolution]
        collar_inverse = collar_inverses[resolution]
        termination_margin = _termination_margin(inverse, active_chart)
        collar_margin = _termination_margin(collar_inverse, active_chart)
        residual_limit = 1e-8 * math.hypot(z[-1] - z[0], r[-1] - r[0])
        item = {
            "profile_points": int(len(theta)),
            "unique_points": int(np.sum(inverse.root_count == 1)),
            "stability_collar_points": int(len(collar_points)),
            "unique_stability_collar_points": int(np.sum(collar_inverse.root_count == 1)),
            "minimum_termination_margin": float(np.min(termination_margin)),
            "collar_minimum_termination_margin": float(np.min(collar_margin)),
            "maximum_inverse_residual": float(np.max(inverse.residual)),
            "maximum_collar_inverse_residual": float(np.max(collar_inverse.residual)),
            "inverse_residual_limit": residual_limit,
        }
        item["passed"] = bool(
            item["unique_points"] == len(theta)
            and item["unique_stability_collar_points"] == len(collar_points)
            and item["minimum_termination_margin"] >= 4.0
            and item["collar_minimum_termination_margin"] >= 4.0
            and item["maximum_inverse_residual"] < residual_limit
            and item["maximum_collar_inverse_residual"] < residual_limit
        )
        diagnostics[resolution] = item
        passed = passed and item["passed"]
    return {"native_surface_valid": True, "charts": diagnostics, "passed": bool(passed)}


def grid_analysis(grid):
    U, S, D = comparison_domain(grid)
    records = {}
    variants = {}
    coverage = {}
    for domain in (8, 10, 12):
        record = source_record(grid, domain)
        record.update({"grid": grid, "domain": domain})
        records[domain] = map_record(record, U, S, D)
        variants[domain] = {
            "primary_map": map_record(record, U, S, D, resolution="primary"),
            "independent5": map_record(record, U, S, D, method="independent5"),
            "cubic": map_record(record, U, S, D, method="cubic"),
        }
        coverage[domain] = {
            branch: branch_coverage(record, branch) for branch in SEEDS
        }
    pairs = {}
    passed = True
    for left, right in ((8, 10), (10, 12)):
        pair = {}
        for observable in ("initial_metric", "metric_increment", "final_metric", "ADM_K"):
            primary = paired_summary(
                {"value": records[left][observable], "weight": records[left]["weight"], "dmax": records[left]["dmax"]},
                {"value": records[right][observable], "weight": records[right]["weight"], "dmax": records[right]["dmax"]}, U, S,
            )
            changes = []
            for variant_name in ("primary_map", "independent5", "cubic"):
                alternate = paired_summary(
                    {"value": variants[left][variant_name][observable], "weight": records[left]["weight"], "dmax": records[left]["dmax"]},
                    {"value": variants[right][variant_name][observable], "weight": records[right]["weight"], "dmax": records[right]["dmax"]}, U, S,
                )
                changes.append(abs(float(alternate["relative_L2"]) - float(primary["relative_L2"])))
            uncertainty = max(*changes, 1e-12)
            threshold = 1e-8 if observable == "initial_metric" else 1e-3
            item_pass = bool(float(primary["relative_L2"]) + uncertainty < threshold)
            pair[observable] = {
                **old.public_pair(primary), "relative_uncertainty": uncertainty,
                "threshold": threshold, "passed": item_pass,
            }
            if observable in ("initial_metric", "metric_increment", "ADM_K"):
                passed = passed and item_pass
        pairs[f"R{left}_R{right}"] = pair
    coverage_pass = bool(all(
        item["passed"] for domain in coverage.values() for item in domain.values()
    ))
    passed = bool(passed and coverage_pass)
    return {
        "grid": grid, "comparison_domain": {
            "U": [float(U[0]), float(U[-1])], "S": [float(S[0]), float(S[-1])],
            "Dmax_range": [float(np.min(D[-1])), float(np.max(D[-1]))],
            "ragged_common_depth": True,
        }, "pairs": pairs, "branch_coverage": coverage,
        "coverage_passed": coverage_pass, "passed": passed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", choices=("G7", "G8", "all"), default="all")
    args = parser.parse_args()
    if sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("sealed Test-2D protocol hash mismatch")
    chart_manifest = json.loads(CHART_MANIFEST.read_text())
    chart_records = [value for key, value in chart_manifest["stages"].items() if key.startswith("chart/")]
    if len(chart_records) != 90 or not all(item.get("status") == "complete" for item in chart_records):
        raise RuntimeError("all 90 Test-2D charts must complete before common-parent scoring")
    ROOT.mkdir(parents=True, exist_ok=True)
    index = RecoveryIndex(MANIFEST, PROTOCOL, recovery_inputs(), maximum_stage_seconds=7200.0)
    grids = ("G7", "G8") if args.grid == "all" else (args.grid,)
    for grid in grids:
        result = stage_json(
            index, f"common_parent/{grid}", f"common_parent_{grid}.json",
            "test2d-common-parent-grid", {"grid": grid},
            lambda grid=grid: {"analysis": grid_analysis(grid)},
        )
        print(json.dumps({"grid": grid, "passed": result["analysis"]["passed"]}), flush=True)
    if any(index.validated_path(f"common_parent/{grid}") is None for grid in ("G7", "G8")):
        print(json.dumps({"phase": "common_parent", "physical_verdict": None}, indent=2))
        return
    analyses = {
        grid: json.loads((ROOT / f"common_parent_{grid}.json").read_text())["analysis"]
        for grid in ("G7", "G8")
    }
    passed = bool(all(item["passed"] for item in analyses.values()))
    result = {
        "protocol": str(PROTOCOL), "protocol_sha256": PROTOCOL_SHA256,
        "status": "PASS" if passed else "REVIEW",
        "classification": (
            "test2d_common_parent_restriction_pass"
            if passed else "test2d_common_parent_restriction_mixed"
        ), "grids": analyses, "passed": passed,
    }
    atomic_write_json(OUTPUT, result)
    stage_json(
        index, "common_parent/result", "result.json", "test2d-common-parent-result", {},
        lambda: {"status": result["status"], "classification": result["classification"],
                 "output": str(OUTPUT), "output_sha256": sha256_file(OUTPUT)},
    )
    print(json.dumps({"status": result["status"], "classification": result["classification"]}, indent=2))


if __name__ == "__main__":
    main()
