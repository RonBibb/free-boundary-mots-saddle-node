#!/usr/bin/env python3
"""Tier-2 exhaustive floating shooting audit under sealed note 91."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import brentq, minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.anisotropic_capped_surface import _splines
from bhps.capped_surface_shooting_audit import (
    adjacent_status_cells,
    shoot_axis_radius,
    shooting_scan,
    summarize_shooting_scan,
)
from bhps.recovery_indexer import RecoveryIndex, atomic_write_json, sha256_file


PROTOCOL = Path("notes/91_A790_initial_no_horizon_certificate_protocol.md")
TIER1_RESULT = Path("results/corrected_A790_initial_no_horizon_certificate.json")
GEOMETRY = {
    "G9": Path("results/corrected_A790_initial_no_horizon_certificate_stages/A790_G9_metric.npz"),
    "G10": Path("results/corrected_A790_initial_no_horizon_certificate_stages/A790_G10_metric.npz"),
    "A794_G7": Path("results/corrected_A790_initial_no_horizon_certificate_stages/A794_G7_metric.npz"),
}
OUTPUT = Path("results/corrected_A790_initial_no_horizon_shooting_audit.json")
MANIFEST = Path("results/corrected_A790_initial_no_horizon_shooting_recovery.json")
RECOVERY = Path("results/corrected_A790_initial_no_horizon_shooting_stages")
RHO_BOUNDS = (0.10, 1.67)
SAMPLE_COUNT = 2049
BATCH_SIZE = 64
BASE_OPTIONS = {
    "rho_bounds": list(RHO_BOUNDS),
    "theta_cut": 1e-3,
    "relative_tolerance": 2e-9,
    "absolute_tolerance": 2e-11,
    "maximum_step": 0.01,
    "graph_slope_guard": 100.0,
}


def load_geometry(path):
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def splines_for(geometry):
    return _splines(
        geometry["z"], geometry["r"], geometry["psi"],
        geometry["a"], geometry["b"], geometry["c"],
    )


def json_stage(index, stage_id, kind, path, expected_seconds, compute, metadata):
    index.register(stage_id, kind, expected_seconds, metadata)
    validated = index.validated_path(stage_id)
    if validated is not None:
        return json.loads(validated.read_text())
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        payload = compute()
        atomic_write_json(path, payload)
        json.loads(path.read_text())
        index.mark_complete(stage_id, path, time.perf_counter() - started)
        return payload
    except Exception as error:
        index.mark_failed(stage_id, repr(error))
        raise


def checkpointed_scan(index, label, geometry, axis_values):
    splines = splines_for(geometry)
    z_brane = float(geometry["z"][-1])
    records = []
    for start in range(0, len(axis_values), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(axis_values))
        path = RECOVERY / f"{label}_shoot_{start:04d}_{stop - 1:04d}.json"
        stage_id = f"shoot/{label}/samples-{start:04d}-{stop - 1:04d}"

        def compute(start=start, stop=stop):
            return {
                "start": start,
                "stop": stop,
                "records": shooting_scan(
                    axis_values[start:stop], z_brane, splines, **BASE_OPTIONS,
                ),
            }

        payload = json_stage(
            index, stage_id, "shooting-samples", path, 3600.0, compute,
            {
                "label": label, "start": start, "stop": stop,
                "sample_count": len(axis_values), **BASE_OPTIONS,
            },
        )
        records.extend(payload["records"])
        print(f"{label}: {stop}/{len(axis_values)} shooting samples", flush=True)
    return records


def residual_function(axis_radius, geometry, options=None):
    options = BASE_OPTIONS if options is None else options
    result = shoot_axis_radius(
        axis_radius, float(geometry["z"][-1]), splines_for(geometry), **options,
    )
    if result["status"] != "reached_brane":
        return None, result
    return result["brane_residual"], result


def refine_physical_features(label, geometry, records):
    summary = summarize_shooting_scan(records)
    reached = [record for record in records if record["status"] == "reached_brane"]
    roots = []
    splines = splines_for(geometry)
    z_brane = float(geometry["z"][-1])

    def objective(value):
        record = shoot_axis_radius(value, z_brane, splines, **BASE_OPTIONS)
        if record["status"] != "reached_brane":
            return 1e4 + abs(record.get("end_slope", 0.0))
        return record["brane_residual"]

    for bracket in summary["sign_change_brackets"]:
        left, right = bracket[:2]
        root = brentq(objective, left, right, xtol=2e-11, rtol=2e-11)
        residual, record = residual_function(root, geometry)
        roots.append({
            "axis_radius": float(root), "brane_residual": float(residual),
            "brane_radius": record["brane_radius"],
        })

    if reached:
        sampled_minimum = min(reached, key=lambda item: item["brane_residual"])
        spacing = (RHO_BOUNDS[1] - RHO_BOUNDS[0]) / (SAMPLE_COUNT - 1)
        lower = max(RHO_BOUNDS[0], sampled_minimum["axis_radius"] - 3 * spacing)
        upper = min(RHO_BOUNDS[1], sampled_minimum["axis_radius"] + 3 * spacing)
        minimized = minimize_scalar(
            objective, bounds=(lower, upper), method="bounded",
            options={"xatol": 2e-10, "maxiter": 80},
        )
        minimum_residual, minimum_record = residual_function(minimized.x, geometry)
        minimum = {
            "axis_radius": float(minimized.x),
            "brane_residual": (
                None if minimum_residual is None else float(minimum_residual)
            ),
            "brane_radius": minimum_record.get("brane_radius"),
            "optimizer_success": bool(minimized.success),
            "function_evaluations": int(minimized.nfev),
        }
    else:
        minimum = None

    variants = []
    if minimum is not None:
        for theta_cut in (2e-3, 1e-3, 5e-4, 2.5e-4):
            for relative_tolerance, maximum_step in (
                (1e-8, 0.02), (2e-9, 0.01), (5e-10, 0.005),
            ):
                options = {
                    **BASE_OPTIONS,
                    "theta_cut": theta_cut,
                    "relative_tolerance": relative_tolerance,
                    "absolute_tolerance": relative_tolerance * 0.01,
                    "maximum_step": maximum_step,
                }
                residual, record = residual_function(
                    minimum["axis_radius"], geometry, options,
                )
                variants.append({
                    "theta_cut": theta_cut,
                    "relative_tolerance": relative_tolerance,
                    "maximum_step": maximum_step,
                    "status": record["status"],
                    "brane_residual": residual,
                    "brane_radius": record.get("brane_radius"),
                })
    finite_variant_residuals = [
        item["brane_residual"] for item in variants
        if item["brane_residual"] is not None
    ]
    return {
        "label": label,
        "scan_summary": summary,
        "cell_coverage": adjacent_status_cells(records),
        "refined_sign_change_roots": roots,
        "refined_minimum": minimum,
        "minimum_variants": variants,
        "variant_residual_range": (
            [min(finite_variant_residuals), max(finite_variant_residuals)]
            if finite_variant_residuals else None
        ),
    }


def main():
    RECOVERY.mkdir(parents=True, exist_ok=True)
    expected_paths = (
        PROTOCOL, TIER1_RESULT, *GEOMETRY.values(),
        Path("src/bhps/anisotropic_capped_surface.py"),
        Path("src/bhps/capped_surface_barrier_certificate.py"),
        Path("src/bhps/capped_surface_shooting_audit.py"),
    )
    expected = {str(path): sha256_file(path) for path in expected_paths}
    index = RecoveryIndex(
        MANIFEST, PROTOCOL, expected, maximum_stage_seconds=3600.0,
    )
    geometries = {label: load_geometry(path) for label, path in GEOMETRY.items()}
    axis_values = np.linspace(RHO_BOUNDS[0], RHO_BOUNDS[1], SAMPLE_COUNT)
    records = {}
    features = {}
    for label in ("G9", "G10", "A794_G7"):
        records[label] = checkpointed_scan(index, label, geometries[label], axis_values)
        path = RECOVERY / f"{label}_features.json"
        features[label] = json_stage(
            index, f"features/{label}", "shooting-feature-refinement",
            path, 3600.0,
            lambda label=label: refine_physical_features(
                label, geometries[label], records[label],
            ),
            {"label": label, "sample_count": SAMPLE_COUNT},
        )

    no_sampled_root = bool(all(
        features[label]["scan_summary"]["sign_change_count"] == 0
        and features[label]["scan_summary"]["all_reached_residuals_positive"]
        and features[label]["refined_minimum"] is not None
        and features[label]["refined_minimum"]["brane_residual"] > 0.0
        and features[label]["variant_residual_range"][0] > 0.0
        for label in ("G9", "G10")
    ))
    adverse_pass = bool(
        features["A794_G7"]["scan_summary"]["sign_change_count"] == 2
        and len(features["A794_G7"]["refined_sign_change_roots"]) == 2
    )
    grid_minima = [
        features[label]["refined_minimum"]["brane_residual"]
        for label in ("G9", "G10")
    ]
    grid_minimum_transfer = float(
        abs(grid_minima[0] - grid_minima[1])
        / max(abs(grid_minima[0]), abs(grid_minima[1]), 1e-300)
    )
    numerical_audit_pass = bool(
        no_sampled_root and adverse_pass and grid_minimum_transfer < 0.05
    )
    result = {
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "status": "REVIEW",
        "classification": (
            "exhaustive_floating_shooting_audit_no_sampled_root"
            if numerical_audit_pass else "shooting_audit_unresolved"
        ),
        "numerical_audit_pass": numerical_audit_pass,
        "sample_count_per_grid": SAMPLE_COUNT,
        "axis_radius_bounds": list(RHO_BOUNDS),
        "base_options": BASE_OPTIONS,
        "features": features,
        "G9_G10_refined_minimum_relative_difference": grid_minimum_transfer,
        "why_not_certificate_PASS": (
            "The complete axis-radius band is densely and deterministically "
            "sampled, but state and first-variation intervals were not "
            "propagated through the nonlinear ODE. Same-exit endpoint cells "
            "therefore remain numerical evidence rather than a mathematical "
            "interval exclusion of a hidden brane-reaching island or tangent root."
        ),
        "claim_boundary": (
            "No sampled root in the sealed C_cap class on the bicubic G9/G10 "
            "discrete metrics; not a proof for arbitrary surfaces or the "
            "continuum spacetime."
        ),
        "provenance": {
            "recovery_manifest": str(MANIFEST),
            "input_sha256": expected,
        },
    }
    atomic_write_json(OUTPUT, result)
    print(json.dumps({
        "status": result["status"],
        "classification": result["classification"],
        "numerical_audit_pass": numerical_audit_pass,
        "A790_minima": dict(zip(("G9", "G10"), grid_minima)),
        "minimum_relative_difference": grid_minimum_transfer,
        "A794_roots": features["A794_G7"]["refined_sign_change_roots"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
