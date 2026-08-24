#!/usr/bin/env python3
"""Refine the four sampled status transitions left by note-91 shooting."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.anisotropic_capped_surface import _splines
from bhps.capped_surface_shooting_audit import shoot_axis_radius
from bhps.recovery_indexer import RecoveryIndex, atomic_write_json, sha256_file


PROTOCOL = Path("notes/91_A790_initial_no_horizon_certificate_protocol.md")
SHOOTING_RESULT = Path("results/corrected_A790_initial_no_horizon_shooting_audit.json")
GEOMETRY = {
    "G9": Path("results/corrected_A790_initial_no_horizon_certificate_stages/A790_G9_metric.npz"),
    "G10": Path("results/corrected_A790_initial_no_horizon_certificate_stages/A790_G10_metric.npz"),
}
OUTPUT = Path("results/corrected_A790_no_horizon_transition_refinement.json")
MANIFEST = Path("results/corrected_A790_no_horizon_transition_refinement_recovery.json")
RECOVERY = Path("results/corrected_A790_no_horizon_transition_refinement_stages")
OPTIONS = {
    "rho_bounds": (0.10, 1.67), "theta_cut": 1e-3,
    "relative_tolerance": 2e-9, "absolute_tolerance": 2e-11,
    "maximum_step": 0.01, "graph_slope_guard": 100.0,
}
DEPTH = 12


def load_geometry(path):
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def solve(value, geometry, splines):
    return shoot_axis_radius(
        value, float(geometry["z"][-1]), splines, **OPTIONS,
    )


def refine_cell(cell, geometry, splines):
    left_value, right_value, expected_left, expected_right = cell
    left = solve(left_value, geometry, splines)
    right = solve(right_value, geometry, splines)
    history = []
    for depth in range(DEPTH):
        middle_value = 0.5 * (left_value + right_value)
        middle = solve(middle_value, geometry, splines)
        history.append({
            "depth": depth + 1, "axis_radius": middle_value,
            "status": middle["status"],
            "brane_residual": middle.get("brane_residual"),
        })
        if middle["status"] == left["status"]:
            left_value, left = middle_value, middle
        elif middle["status"] == right["status"]:
            right_value, right = middle_value, middle
        else:
            # Preserve the newly exposed status and the narrower adjacent side.
            if (middle_value - left_value) <= (right_value - middle_value):
                right_value, right = middle_value, middle
            else:
                left_value, left = middle_value, middle
    residuals = [
        item["brane_residual"] for item in (left, right)
        if item.get("brane_residual") is not None
    ]
    return {
        "initial_cell": cell,
        "expected_endpoint_statuses": [expected_left, expected_right],
        "actual_initial_endpoint_statuses": [
            solve(cell[0], geometry, splines)["status"],
            solve(cell[1], geometry, splines)["status"],
        ],
        "refined_left_axis_radius": left_value,
        "refined_right_axis_radius": right_value,
        "refined_width": right_value - left_value,
        "refined_endpoint_statuses": [left["status"], right["status"]],
        "brane_endpoint_residuals": residuals,
        "all_observed_brane_residuals_positive": bool(
            all(value > 0.0 for value in residuals)
        ),
        "history": history,
    }


def json_stage(index, stage_id, path, compute, metadata):
    index.register(stage_id, "transition-refinement", 3600.0, metadata)
    validated = index.validated_path(stage_id)
    if validated is not None:
        return json.loads(validated.read_text())
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        payload = compute()
        atomic_write_json(path, payload)
        index.mark_complete(stage_id, path, time.perf_counter() - started)
        return payload
    except Exception as error:
        index.mark_failed(stage_id, repr(error))
        raise


def main():
    RECOVERY.mkdir(parents=True, exist_ok=True)
    expected_paths = (
        PROTOCOL, SHOOTING_RESULT, *GEOMETRY.values(),
        Path("src/bhps/anisotropic_capped_surface.py"),
        Path("src/bhps/capped_surface_shooting_audit.py"),
    )
    expected = {str(path): sha256_file(path) for path in expected_paths}
    index = RecoveryIndex(MANIFEST, PROTOCOL, expected, 3600.0)
    shooting = json.loads(SHOOTING_RESULT.read_text())
    results = {}
    for label, path in GEOMETRY.items():
        geometry = load_geometry(path)
        splines = _splines(
            geometry["z"], geometry["r"], geometry["psi"],
            geometry["a"], geometry["b"], geometry["c"],
        )
        cells = shooting["features"][label]["cell_coverage"]["mixed_cells"]
        results[label] = []
        for index_value, cell in enumerate(cells):
            output = RECOVERY / f"{label}_transition_{index_value}.json"
            result = json_stage(
                index, f"transition/{label}/{index_value}", output,
                lambda cell=cell: refine_cell(cell, geometry, splines),
                {"label": label, "cell_index": index_value, "depth": DEPTH},
            )
            results[label].append(result)
            print(
                f"{label} transition {index_value + 1}/{len(cells)}: "
                f"width={result['refined_width']:.3e}", flush=True,
            )
    passed = bool(all(
        result["refined_width"]
        <= (1.67 - 0.10) / (2048 * 2**DEPTH) * (1.0 + 1e-8)
        and result["all_observed_brane_residuals_positive"]
        for values in results.values() for result in values
    ))
    payload = {
        "protocol": str(PROTOCOL), "protocol_sha256": sha256_file(PROTOCOL),
        "status": "PASS" if passed else "REVIEW",
        "classification": "sampled_status_transition_refinement",
        "depth": DEPTH, "results": results,
        "all_transition_refinement_checks_passed": passed,
        "claim_boundary": (
            "Refines observed status changes only; it does not interval-exclude "
            "a hidden brane-reaching island inside a same-status sampled cell."
        ),
        "provenance": {"manifest": str(MANIFEST), "input_sha256": expected},
    }
    atomic_write_json(OUTPUT, payload)
    print(json.dumps({
        "status": payload["status"],
        "all_transition_refinement_checks_passed": passed,
        "maximum_refined_width": max(
            item["refined_width"] for values in results.values() for item in values
        ),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
