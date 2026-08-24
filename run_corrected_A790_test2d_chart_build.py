#!/usr/bin/env python3
"""Build all prospectively sealed Test-2D ragged charts with recovery."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import run_corrected_A790_test2b_invariant_convergence as old
from bhps.ragged_normal_arclength_chart import (
    build_ragged_normal_chart,
    ragged_chart_arrays,
    ragged_chart_validity,
)
from bhps.recovery_indexer import RecoveryIndex, atomic_write_json, atomic_write_npz, sha256_file, validate_npz


PROTOCOL = Path("notes/110_A790_test2D_high_order_ragged_chart_protocol.md")
PROTOCOL_SHA256 = "f11989b23cff2d5b87bf6e730ff91c77b1e095eb2a7a49d1a277a9d3bf2666e5"
MACHINE_PROTOCOL = Path("results/corrected_A790_test2d_high_order_ragged_chart_protocol.json")
MACHINE_PROTOCOL_SHA256 = "d3a1f59c61499d5e8f67fcd29a4592a7c1afd65b74e7bc90d78d0a35121734e4"
QUALIFICATION = Path("results/corrected_A790_test2d_high_order_ragged_chart_qualification_v3.json")
QUALIFICATION_SHA256 = "784e29ca9199b14e9d764e60b4bce7fa1d1610b6176c3ae02fed686d81dbf207"
ALIGNED = Path("results/corrected_A790_test2b_invariant_convergence_recovery/aligned_states.npz")
ALIGNED_SHA256 = "42ec6dcb65038a6aa7ab1fc724e848bbba7be781b887cd18f3512cc030256265"
GEOMETRIES = Path("results/corrected_A790_test2b_invariant_convergence_recovery/geometries.npz")
GEOMETRIES_SHA256 = "fd5cf03f1fb37b9e9f7d1f14f8112c2452ccb4f0f6d8438d251d1432bb2dd618"
COMMON_PARENT = Path("results/corrected_A790_test10b_domain_normalized_state.npz")
COMMON_PARENT_SHA256 = "b19e7e8901fdade6abb485c65a3512bf327a86f13974fb9a8c0876340a503928"

ROOT = Path("results/corrected_A790_test2d_high_order_ragged_chart_recovery")
MANIFEST = ROOT / "chart_index_v2.json"
SUMMARY = ROOT / "chart_summary_v2.json"
RESOLUTIONS = {
    "coarse": {"ray_count": 193, "distance_samples": 129, "coarse": True},
    "primary": {"ray_count": 257, "distance_samples": 193, "coarse": False},
    "fine": {"ray_count": 385, "distance_samples": 257, "coarse": False},
}
SPATIAL = ("G9", "G10", "G11")
TEMPORAL = ("G10_coarse", "G10_standard", "G10_half")
SLICES = ("initial", "pchip", "linear")


def recovery_inputs():
    fixed = {
        str(MACHINE_PROTOCOL): MACHINE_PROTOCOL_SHA256,
        str(QUALIFICATION): QUALIFICATION_SHA256,
        str(ALIGNED): ALIGNED_SHA256,
        str(GEOMETRIES): GEOMETRIES_SHA256,
        str(COMMON_PARENT): COMMON_PARENT_SHA256,
    }
    dynamic = (
        Path(__file__), Path("src/bhps/ragged_normal_arclength_chart.py"),
        Path("src/bhps/recovery_indexer.py"),
    )
    return {**fixed, **{str(path): sha256_file(path) for path in dynamic}}


def chart_tasks():
    tasks = []
    for label in (*SPATIAL, *TEMPORAL):
        for slice_name in SLICES:
            for resolution, config in RESOLUTIONS.items():
                tasks.append({
                    "family": "aligned", "label": label,
                    "slice": slice_name, "resolution": resolution, **config,
                })
    for grid in ("G7", "G8"):
        for domain in (8, 10, 12):
            for slice_name in ("initial", "final"):
                for resolution, config in RESOLUTIONS.items():
                    tasks.append({
                        "family": "common_parent", "label": grid,
                        "domain": domain, "slice": slice_name,
                        "resolution": resolution, **config,
                    })
    return tasks


def task_id(task):
    if task["family"] == "aligned":
        return "/".join(("chart", "aligned", task["label"], task["slice"], task["resolution"]))
    return "/".join((
        "chart", "common_parent", task["label"], f"R{task['domain']}",
        task["slice"], task["resolution"],
    ))


def task_filename(task):
    return task_id(task).replace("/", "_") + ".npz"


def _aligned_native(task):
    grid = task["label"] if task["label"] in SPATIAL else "G10"
    with np.load(GEOMETRIES) as geometry, np.load(ALIGNED) as aligned:
        z, r = np.asarray(geometry[f"{grid}_z"]), np.asarray(geometry[f"{grid}_r"])
        if task["slice"] == "initial":
            key = f"{grid}_initial"
        elif task["slice"] == "pchip":
            key = f"{task['label']}_position"
        else:
            key = f"{task['label']}_linear_position"
        position = np.asarray(aligned[key])
    return z, r, position


def _common_parent_native(task):
    with np.load(COMMON_PARENT) as archive:
        prefix = f"{task['label']}_R{task['domain']}"
        z = np.asarray(archive[f"{task['label']}_R12_z"])
        r = np.asarray(archive[f"{prefix}_r"])
        key = f"{prefix}_initial" if task["slice"] == "initial" else f"{prefix}_step_016_position"
        position = np.asarray(archive[key])
    return z, r, position


def _build_one(task, destination):
    started = time.perf_counter()
    if task["family"] == "aligned":
        z, r, position = _aligned_native(task)
    else:
        z, r, position = _common_parent_native(task)
    metric, sphere = old.reduced_metric(position, r)
    chart = build_ragged_normal_chart(
        z, r, metric, sphere, launch_radius_max=6.0,
        ray_count=task["ray_count"], distance_samples=task["distance_samples"],
    )
    validity = ragged_chart_validity(chart, coarse=task["coarse"])
    atomic_write_npz(destination, **ragged_chart_arrays(chart))
    validate_npz(destination)
    return {
        "path": str(destination), "elapsed_seconds": time.perf_counter() - started,
        "validity": validity,
    }


def progress(index):
    records = [record for key, record in index.data["stages"].items() if key.startswith("chart/")]
    counts = {}
    for record in records:
        status = record.get("status", "pending")
        counts[status] = counts.get(status, 0) + 1
    complete = counts.get("complete", 0)
    valid = sum(
        bool(record.get("completion_metadata", {}).get("validity", {}).get("valid"))
        for record in records if record.get("status") == "complete"
    )
    return {
        "total": len(records), "complete": complete, "valid": valid,
        "invalid_complete": complete - valid, "counts": counts,
        "percent": 100.0 * complete / max(len(records), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-charts", type=int, default=None)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("sealed Test-2D protocol hash mismatch")
    if sha256_file(MACHINE_PROTOCOL) != MACHINE_PROTOCOL_SHA256:
        raise RuntimeError("sealed Test-2D machine protocol hash mismatch")
    qualification = json.loads(QUALIFICATION.read_text())
    if qualification.get("status") != "QUALIFIED" or qualification.get("physical_verdict") is not None:
        raise RuntimeError("Test-2D numerical qualification is not a clean pre-physical parent")
    ROOT.mkdir(parents=True, exist_ok=True)
    index = RecoveryIndex(MANIFEST, PROTOCOL, recovery_inputs(), maximum_stage_seconds=43200.0)
    pending = []
    for task in chart_tasks():
        identifier = task_id(task)
        metadata = {key: value for key, value in task.items()}
        index.register(identifier, "test2d-ragged-normal-chart", 43200.0, metadata)
        if index.validated_path(identifier) is None:
            pending.append(task)
    if args.max_new_charts is not None:
        pending = pending[:max(args.max_new_charts, 0)]

    futures = {}
    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
        for task in pending:
            identifier = task_id(task)
            destination = ROOT / task_filename(task)
            index.mark_running(identifier)
            futures[executor.submit(_build_one, task, destination)] = (identifier, destination)
        for future in as_completed(futures):
            identifier, destination = futures[future]
            try:
                result = future.result()
                index.mark_complete(
                    identifier, destination, result["elapsed_seconds"],
                    metadata={"validity": result["validity"]},
                )
                print(json.dumps({
                    "stage": identifier, "valid": result["validity"]["valid"],
                    "elapsed_seconds": result["elapsed_seconds"],
                }), flush=True)
            except Exception as error:
                index.mark_failed(identifier, f"{type(error).__name__}: {error}")
                print(json.dumps({"stage": identifier, "failed": str(error)}), flush=True)

    summary = {
        "protocol": str(PROTOCOL), "protocol_sha256": PROTOCOL_SHA256,
        "progress": progress(index), "manifest": str(MANIFEST),
        "all_complete": progress(index)["complete"] == len(chart_tasks()),
        "all_valid": progress(index)["valid"] == len(chart_tasks()),
        "physical_verdict": None,
    }
    atomic_write_json(SUMMARY, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
