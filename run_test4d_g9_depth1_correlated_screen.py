#!/usr/bin/env python3
"""Correlated interval screens for both G9 depth-1 Test-4D children."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.correlated_affine_spline import correlated_divergence_jacobian_hull
from bhps.recovery_indexer import (
    RecoveryIndex,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
    validate_npz,
)
from bhps.validated_capped_surface_shooting import interval_hull
from run_test4b_validated_interval_no_horizon_certificate import (
    SPLINE_ARCHIVE,
    load_validated_metric,
)
from run_test4d_bulk_interval_jacobian_screen import (
    build_bulk_interval_jacobian,
    component_scale_vector,
)
from run_test4d_subdivided_correlated_bulk_screen import (
    assemble_subdivided_jacobian,
    task_data,
)


PROTOCOL = Path("notes/104_test4d_validated_global_bvp_certificate_protocol.md")
FLOATING_SUMMARY = Path(
    "results/test4d_g9_depth1_floating_screen_stages/"
    "G9_b217_depth1_D12_M70_P160_floating_summary.json"
)
CHILD_ARCHIVES = {
    path: Path(
        "results/test4d_g9_depth1_floating_screen_stages/"
        f"G9_b217_p{path}_D12_M70_P160_floating_operator.npz"
    )
    for path in ("0", "1")
}
RECOVERY = Path("results/test4d_g9_depth1_correlated_screen_stages")
MANIFEST = Path("results/test4d_g9_depth1_correlated_screen_recovery.json")
SUMMARY = RECOVERY / "G9_b217_depth1_D12_M70_P160_correlated_summary.json"
LABEL = "G9"
PARAMETER_SUBDIVISIONS = 16
WORKERS = 8
_WORKER_METRIC = None


def _initialize_worker(spline_archive):
    global _WORKER_METRIC
    _WORKER_METRIC = load_validated_metric(Path(spline_archive))


def _subdivided_node_hull(task):
    theta, rho_center, rho_parameter, w_center, w_parameter = task
    entries = [[[] for _ in range(2)] for _ in range(2)]
    maximum_segments = 0
    for index in range(PARAMETER_SUBDIVISIONS):
        lower = -1.0 + 2.0 * index / PARAMETER_SUBDIVISIONS
        upper = -1.0 + 2.0 * (index + 1) / PARAMETER_SUBDIVISIONS
        jacobian, audit = correlated_divergence_jacobian_hull(
            theta,
            rho_center,
            rho_parameter,
            w_center,
            w_parameter,
            _WORKER_METRIC,
            xi_lower=lower,
            xi_upper=upper,
        )
        maximum_segments = max(
            maximum_segments, audit["parameter_segment_count"],
        )
        for row in range(2):
            for column in range(2):
                entries[row][column].append(jacobian[row, column])
    hull = np.asarray([
        [interval_hull(entries[row][column]) for column in range(2)]
        for row in range(2)
    ], dtype=object)
    return (
        np.asarray([
            [[hull[row, column].lower, hull[row, column].upper]
             for column in range(2)]
            for row in range(2)
        ]),
        maximum_segments,
    )


def provenance_inputs():
    paths = (
        PROTOCOL,
        FLOATING_SUMMARY,
        *CHILD_ARCHIVES.values(),
        SPLINE_ARCHIVE[LABEL],
        Path("src/bhps/correlated_affine_spline.py"),
        Path("run_test4d_bulk_interval_jacobian_screen.py"),
        Path("run_test4d_subdivided_correlated_bulk_screen.py"),
        Path(__file__),
    )
    return {str(path): sha256_file(path) for path in paths}


def recovery_index():
    if json.loads(FLOATING_SUMMARY.read_text()).get("certificate_claimed") is not False:
        raise RuntimeError("depth-1 floating precursor has invalid claim label")
    return RecoveryIndex(
        MANIFEST, PROTOCOL, provenance_inputs(), maximum_stage_seconds=1800.0,
    )


def output_path(path):
    return RECOVERY / f"G9_b217_p{path}_D12_M70_P160_correlated_xi16.npz"


def ensure_child(index, path):
    stage_id = f"physical/G9/base217/path{path}/D12-M70-P160/correlated-xi16"
    metadata = {
        "classification": "partial_child_bulk_screen_not_a_certificate",
        "subdivision_path": path,
        "parameter_subdivisions": PARAMETER_SUBDIVISIONS,
        "worker_count": WORKERS,
        "axis_rows_included": False,
    }
    index.register(stage_id, "child-correlated-bulk-screen", 1800.0, metadata)
    reusable = index.validated_path(stage_id)
    if reusable is not None:
        validate_npz(reusable, require_finite=True)
        return index.data["stages"][stage_id]["completion_metadata"]["diagnostics"], True
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        with np.load(CHILD_ARCHIVES[path]) as archive:
            data = {key: np.asarray(archive[key]) for key in archive.files}
        center = data["center_vector"]
        parameter = data["parameter_vector"]
        mesh = data["mesh"]
        shape = tuple(data["jacobian_shape"].astype(int))
        finite_difference = csc_matrix((
            data["center_jacobian_data"],
            data["center_jacobian_indices"],
            data["center_jacobian_indptr"],
        ), shape=shape).tocsr()
        metric = load_validated_metric(SPLINE_ARCHIVE[LABEL])
        print(f"child {path}: building analytic point center", flush=True)
        analytic_center, _, _ = build_bulk_interval_jacobian(
            center, np.zeros_like(center), mesh, metric,
        )
        axis_rows = 34
        analytic_center = analytic_center.tolil()
        analytic_center[:axis_rows, :] = finite_difference[:axis_rows, :]
        analytic_center = analytic_center.tocsc()
        tasks, _ = task_data(center, parameter, mesh)
        print(
            f"child {path}: dispatching {len(tasks)} nodes x "
            f"{PARAMETER_SUBDIVISIONS} pieces",
            flush=True,
        )
        context = multiprocessing.get_context("fork")
        results = []
        with ProcessPoolExecutor(
            max_workers=WORKERS,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(str(SPLINE_ARCHIVE[LABEL]),),
        ) as executor:
            for count, result in enumerate(
                executor.map(_subdivided_node_hull, tasks, chunksize=1), start=1,
            ):
                results.append(result)
                if count % 120 == 0:
                    print(f"child {path}: nodes {count}/{len(tasks)}", flush=True)
        interval_midpoint, interval_radius, max_width, max_segments = (
            assemble_subdivided_jacobian(center, parameter, mesh, results)
        )
        difference = interval_midpoint - analytic_center
        difference.data = np.abs(difference.data)
        difference = (difference + interval_radius).tolil()
        difference[:axis_rows, :] = 0.0
        difference = difference.tocsr()
        difference.eliminate_zeros()
        factor = splu(analytic_center)
        inverse = factor.solve(np.eye(analytic_center.shape[0]))
        scales = component_scale_vector(data["component_scales"])
        inverse_scaled = np.abs(inverse / scales[:, None])
        image = inverse_scaled @ (difference @ diags(scales))
        row_sums = np.asarray(image.sum(axis=1)).reshape(-1)
        partial_z1 = float(np.max(row_sums))
        arrays = {
            "difference_magnitude_data": difference.data,
            "difference_magnitude_indices": difference.indices,
            "difference_magnitude_indptr": difference.indptr,
            "shape": np.asarray(difference.shape, dtype=np.int64),
            "row_sum_bulk_only": row_sums,
        }
        output = output_path(path)
        atomic_write_npz(output, **arrays)
        validate_npz(output, require_finite=True)
        diagnostics = {
            "subdivision_path": path,
            "status": "partial_child_bulk_screen_complete_not_a_certificate",
            "certificate_claimed": False,
            "parameter_subdivisions": PARAMETER_SUBDIVISIONS,
            "partial_correlated_bulk_Z1_like_upper": partial_z1,
            "maximum_correlated_derivative_width": max_width,
            "maximum_knot_aligned_segments_per_piece": max_segments,
            "all_finite": bool(
                np.isfinite(partial_z1)
                and np.all(np.isfinite(difference.data))
            ),
            "archive": {"path": str(output), "sha256": sha256_file(output)},
        }
        index.mark_complete(
            stage_id,
            output,
            time.perf_counter() - started,
            {"diagnostics": diagnostics},
        )
        return diagnostics, False
    except Exception as error:
        index.mark_failed(stage_id, repr(error))
        raise


def ensure_summary(index, records):
    stage_id = "physical/G9/base217/depth1/D12-M70-P160/correlated-summary"
    metadata = {
        "paths": [record["subdivision_path"] for record in records],
        "archive_hashes": [record["archive"]["sha256"] for record in records],
    }
    index.register(stage_id, "child-correlated-summary", 120.0, metadata)
    reusable = index.validated_path(stage_id)
    if reusable is not None:
        return json.loads(reusable.read_text()), True
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        floating = json.loads(FLOATING_SUMMARY.read_text())
        maximum_z1 = max(
            record["partial_correlated_bulk_Z1_like_upper"] for record in records
        )
        payload = {
            "schema": "test4d-g9-depth1-correlated-screen-v1",
            "status": "partial_depth1_correlated_screen_complete_not_a_certificate",
            "certificate_claimed": False,
            "records": records,
            "maximum_partial_bulk_Z1_upper": maximum_z1,
            "maximum_point_Y_proxy": floating["maximum_point_Y_proxy"],
            "zero_Z2_radius_polynomial_proxy_at_1e-4": (
                floating["maximum_point_Y_proxy"]
                + (maximum_z1 - 1.0) * 1e-4
            ),
            "omissions": [
                "regular-axis integral rows",
                "directed high-precision inverse product",
                "Bernstein off-node and coefficient tails",
                "candidate-radius Z2 variation",
            ],
        }
        atomic_write_json(SUMMARY, payload)
        index.mark_complete(stage_id, SUMMARY, time.perf_counter() - started)
        return payload, False
    except Exception as error:
        index.mark_failed(stage_id, repr(error))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--both-children", action="store_true")
    arguments = parser.parse_args()
    if not arguments.both_children:
        raise SystemExit("select --both-children")
    index = recovery_index()
    records = []
    reuse = []
    for path in ("0", "1"):
        record, reused = ensure_child(index, path)
        records.append(record)
        reuse.append(reused)
    summary, summary_reused = ensure_summary(index, records)
    print(json.dumps({
        "child_reused": reuse,
        "summary_reused": summary_reused,
        "summary": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
