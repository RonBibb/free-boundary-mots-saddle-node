#!/usr/bin/env python3
"""Bulk tube-Jacobian feasibility screen at the sealed largest radius.

The launch parameter is split into sixteen boxes per depth-1 child.  Each box
is enlarged by the point-evaluation image of a normalized coefficient ball of
radius 1e-4.  Exact interval bicubic splines enclose the resulting local
Jacobian.  The result is still not a certificate because the inverse product,
axis rows, and coefficient/off-node tails are not all directed here.
"""

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

from bhps.recovery_indexer import (
    RecoveryIndex,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
    validate_npz,
)
from bhps.validated_capped_surface_shooting import (
    VInterval,
    interval_hull,
    regularized_divergence_rhs_jet,
)
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
CORRELATED_SUMMARY = Path(
    "results/test4d_g9_depth1_correlated_screen_stages/"
    "G9_b217_depth1_D12_M70_P160_correlated_summary.json"
)
AXIS_SUMMARY = Path(
    "results/test4d_g9_depth1_axis_screen_stages/"
    "G9_b217_depth1_D12_M70_P160_axis_summary.json"
)
FLOATING_ARCHIVES = {
    path: Path(
        "results/test4d_g9_depth1_floating_screen_stages/"
        f"G9_b217_p{path}_D12_M70_P160_floating_operator.npz"
    )
    for path in ("0", "1")
}
CORRELATED_ARCHIVES = {
    path: Path(
        "results/test4d_g9_depth1_correlated_screen_stages/"
        f"G9_b217_p{path}_D12_M70_P160_correlated_xi16.npz"
    )
    for path in ("0", "1")
}
RECOVERY = Path("results/test4d_g9_depth1_z2_tube_screen_stages")
MANIFEST = Path("results/test4d_g9_depth1_z2_tube_screen_recovery.json")
SUMMARY = RECOVERY / "G9_b217_depth1_D12_M70_P160_z2_tube_summary.json"
LABEL = "G9"
PARAMETER_SUBDIVISIONS = 16
CANDIDATE_RADIUS = 1e-4
WORKERS = 8
_WORKER_METRIC = None
_WORKER_RHO_BALL = None
_WORKER_W_BALL = None


def _initialize_worker(spline_archive, rho_ball, w_ball):
    global _WORKER_METRIC, _WORKER_RHO_BALL, _WORKER_W_BALL
    _WORKER_METRIC = load_validated_metric(Path(spline_archive))
    _WORKER_RHO_BALL = float(rho_ball)
    _WORKER_W_BALL = float(w_ball)


def _tube_node_hull(task):
    theta, rho_center, rho_parameter, w_center, w_parameter = task
    entries = [[[] for _ in range(2)] for _ in range(2)]
    for index in range(PARAMETER_SUBDIVISIONS):
        lower = -1.0 + 2.0 * index / PARAMETER_SUBDIVISIONS
        upper = -1.0 + 2.0 * (index + 1) / PARAMETER_SUBDIVISIONS
        midpoint = 0.5 * (lower + upper)
        halfwidth = 0.5 * (upper - lower)
        rho_midpoint = rho_center + rho_parameter * midpoint
        w_midpoint = w_center + w_parameter * midpoint
        rho_radius = abs(rho_parameter) * halfwidth + _WORKER_RHO_BALL
        w_radius = abs(w_parameter) * halfwidth + _WORKER_W_BALL
        jet = regularized_divergence_rhs_jet(
            VInterval.point(theta),
            VInterval(rho_midpoint - rho_radius, rho_midpoint + rho_radius),
            VInterval(w_midpoint - w_radius, w_midpoint + w_radius),
            _WORKER_METRIC,
        )
        local = (
            (jet[0].derivative[1], jet[0].derivative[2]),
            (jet[1].derivative[1], jet[1].derivative[2]),
        )
        for row in range(2):
            for column in range(2):
                entries[row][column].append(local[row][column])
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
        1,
    )


def provenance_inputs():
    paths = (
        PROTOCOL,
        CORRELATED_SUMMARY,
        AXIS_SUMMARY,
        *FLOATING_ARCHIVES.values(),
        *CORRELATED_ARCHIVES.values(),
        SPLINE_ARCHIVE[LABEL],
        Path("src/bhps/validated_capped_surface_shooting.py"),
        Path("run_test4d_bulk_interval_jacobian_screen.py"),
        Path("run_test4d_subdivided_correlated_bulk_screen.py"),
        Path(__file__),
    )
    return {str(path): sha256_file(path) for path in paths}


def recovery_index():
    for path in (CORRELATED_SUMMARY, AXIS_SUMMARY):
        if json.loads(path.read_text()).get("certificate_claimed") is not False:
            raise RuntimeError(f"precursor has invalid claim label: {path}")
    return RecoveryIndex(
        MANIFEST, PROTOCOL, provenance_inputs(), maximum_stage_seconds=1800.0,
    )


def ensure_child(index, path):
    stage_id = f"physical/G9/base217/path{path}/D12-M70-P160/z2-tube-r1e-4"
    metadata = {
        "classification": "bulk_tube_feasibility_not_a_certificate",
        "candidate_radius": CANDIDATE_RADIUS,
        "parameter_subdivisions": PARAMETER_SUBDIVISIONS,
        "worker_count": WORKERS,
        "axis_rows_included": False,
        "directed_inverse_product": False,
    }
    index.register(stage_id, "bulk-z2-tube-screen", 1800.0, metadata)
    reusable = index.validated_path(stage_id)
    if reusable is not None:
        validate_npz(reusable, require_finite=True)
        return index.data["stages"][stage_id]["completion_metadata"]["diagnostics"], True
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        with np.load(FLOATING_ARCHIVES[path]) as archive:
            center = np.asarray(archive["center_vector"])
            parameter = np.asarray(archive["parameter_vector"])
            mesh = np.asarray(archive["mesh"])
            component_scales = np.asarray(archive["component_scales"])
            shape = tuple(np.asarray(archive["jacobian_shape"]).astype(int))
            finite_difference = csc_matrix((
                np.asarray(archive["center_jacobian_data"]),
                np.asarray(archive["center_jacobian_indices"]),
                np.asarray(archive["center_jacobian_indptr"]),
            ), shape=shape).tocsr()
        metric = load_validated_metric(SPLINE_ARCHIVE[LABEL])
        print(f"child {path}: building analytic center", flush=True)
        analytic_center, _, _ = build_bulk_interval_jacobian(
            center, np.zeros_like(center), mesh, metric,
        )
        axis_rows = 34
        analytic_center = analytic_center.tolil()
        analytic_center[:axis_rows, :] = finite_difference[:axis_rows, :]
        analytic_center = analytic_center.tocsc()
        tasks, _ = task_data(center, parameter, mesh)
        rho_ball = float(component_scales[0]) * CANDIDATE_RADIUS
        w_ball = float(component_scales[2]) * CANDIDATE_RADIUS
        print(
            f"child {path}: dispatching {len(tasks)} nodes x "
            f"{PARAMETER_SUBDIVISIONS} tube boxes",
            flush=True,
        )
        context = multiprocessing.get_context("fork")
        results = []
        with ProcessPoolExecutor(
            max_workers=WORKERS,
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(str(SPLINE_ARCHIVE[LABEL]), rho_ball, w_ball),
        ) as executor:
            for count, result in enumerate(
                executor.map(_tube_node_hull, tasks, chunksize=1), start=1,
            ):
                results.append(result)
                if count % 120 == 0:
                    print(f"child {path}: nodes {count}/{len(tasks)}", flush=True)
        interval_midpoint, interval_radius, max_width, _ = (
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
        scales = component_scale_vector(component_scales)
        inverse_scaled = np.abs(inverse / scales[:, None])
        image = inverse_scaled @ (difference @ diags(scales))
        row_sums = np.asarray(image.sum(axis=1)).reshape(-1)
        tube_z = float(np.max(row_sums))
        with np.load(CORRELATED_ARCHIVES[path]) as archive:
            base_z = float(np.max(np.asarray(archive["row_sum_bulk_only"])))
        increment = max(0.0, tube_z - base_z)
        output = RECOVERY / f"G9_b217_p{path}_D12_M70_P160_z2_tube_r1e-4.npz"
        atomic_write_npz(
            output,
            difference_magnitude_data=difference.data,
            difference_magnitude_indices=difference.indices,
            difference_magnitude_indptr=difference.indptr,
            shape=np.asarray(difference.shape, dtype=np.int64),
            row_sum_bulk_tube=row_sums,
        )
        validate_npz(output, require_finite=True)
        diagnostics = {
            "subdivision_path": path,
            "status": "bulk_tube_feasibility_complete_not_a_certificate",
            "certificate_claimed": False,
            "candidate_radius": CANDIDATE_RADIUS,
            "parameter_subdivisions": PARAMETER_SUBDIVISIONS,
            "rho_point_ball": rho_ball,
            "w_point_ball": w_ball,
            "base_correlated_bulk_Z1_upper": base_z,
            "bulk_tube_Z1_plus_Z2r_like_upper": tube_z,
            "increment_over_base_norm_proxy": increment,
            "implied_Z2_like_proxy": increment / CANDIDATE_RADIUS,
            "maximum_tube_derivative_width": max_width,
            "contraction_below_one": tube_z < 1.0,
            "all_finite": bool(
                np.isfinite(tube_z) and np.all(np.isfinite(difference.data))
            ),
            "omissions": [
                "regular-axis tube rows",
                "directed high-precision inverse product",
                "coefficient/off-node tails",
                "separate directed Z1 and affine Z2 decomposition",
            ],
            "archive": {"path": str(output), "sha256": sha256_file(output)},
        }
        index.mark_complete(
            stage_id, output, time.perf_counter() - started,
            {"diagnostics": diagnostics},
        )
        return diagnostics, False
    except Exception as error:
        index.mark_failed(stage_id, repr(error))
        raise


def ensure_summary(index, records):
    stage_id = "physical/G9/base217/depth1/D12-M70-P160/z2-tube-summary"
    index.register(
        stage_id, "bulk-z2-tube-summary", 120.0,
        {"archive_hashes": [record["archive"]["sha256"] for record in records]},
    )
    reusable = index.validated_path(stage_id)
    if reusable is not None:
        return json.loads(reusable.read_text()), True
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        maximum_tube = max(
            record["bulk_tube_Z1_plus_Z2r_like_upper"] for record in records
        )
        payload = {
            "schema": "test4d-g9-depth1-z2-tube-screen-v1",
            "status": "bulk_tube_feasibility_complete_not_a_certificate",
            "certificate_claimed": False,
            "candidate_radius": CANDIDATE_RADIUS,
            "records": records,
            "maximum_bulk_tube_Z1_plus_Z2r_like_upper": maximum_tube,
            "bulk_tube_contraction_below_one": maximum_tube < 1.0,
            "interpretation": (
                "A value below one supports local tube feasibility at the "
                "sealed largest radius, but cannot replace the complete "
                "directed radii-polynomial decomposition."
            ),
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
