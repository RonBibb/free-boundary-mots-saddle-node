#!/usr/bin/env python3
"""32-piece correlated bulk interval screen for sealed Test 4D."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import csc_matrix, diags, lil_matrix
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.correlated_affine_spline import correlated_divergence_jacobian_hull
from bhps.global_bvp_collocation import (
    increasing_lobatto_nodes,
    lobatto_differentiation_matrix,
    shared_nodal_layout,
    unpack_shared_vector,
)
from bhps.recovery_indexer import (
    RecoveryIndex,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
    validate_npz,
)
from bhps.validated_capped_surface_shooting import VInterval, interval_hull
from bhps.validated_global_bvp import CONFIGURATIONS
from run_test4b_validated_interval_no_horizon_certificate import (
    SPLINE_ARCHIVE,
    load_validated_metric,
)
from run_test4d_bulk_interval_jacobian_screen import (
    build_bulk_interval_jacobian,
    component_scale_vector,
    variable_column,
)


PROTOCOL = Path("notes/104_test4d_validated_global_bvp_certificate_protocol.md")
PREDICTOR = Path(
    "results/test4d_validated_global_bvp_feasibility_stages/"
    "G9_b217_D12_M70_P160_affine_predictor.npz"
)
AFFINE = Path(
    "results/test4d_affine_operator_feasibility_stages/"
    "G9_b217_D12_M70_P160_affine_operator.npz"
)
ANALYTIC_V1 = Path(
    "results/test4d_analytic_correlated_bulk_screen_stages/"
    "G9_b217_D12_M70_P160_analytic_correlated_bulk_xi1_summary.json"
)
RECOVERY = Path("results/test4d_subdivided_correlated_bulk_screen_stages")
MANIFEST = Path("results/test4d_subdivided_correlated_bulk_screen_recovery.json")
OUTPUT = RECOVERY / "G9_b217_D12_M70_P160_correlated_bulk_xi32.npz"
SUMMARY = RECOVERY / "G9_b217_D12_M70_P160_correlated_bulk_xi32_summary.json"
CONFIGURATION = CONFIGURATIONS[0]
LABEL = "G9"
PARAMETER_SUBDIVISIONS = 32
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
        PREDICTOR,
        AFFINE,
        ANALYTIC_V1,
        SPLINE_ARCHIVE[LABEL],
        Path("src/bhps/correlated_affine_spline.py"),
        Path("run_test4d_bulk_interval_jacobian_screen.py"),
        Path(__file__),
    )
    return {str(path): sha256_file(path) for path in paths}


def recovery_index():
    precursor = json.loads(ANALYTIC_V1.read_text())
    if precursor.get("certificate_claimed") is not False:
        raise RuntimeError("analytic precursor has invalid claim label")
    return RecoveryIndex(
        MANIFEST, PROTOCOL, provenance_inputs(), maximum_stage_seconds=1800.0,
    )


def task_data(center, parameter, mesh):
    layout = shared_nodal_layout(CONFIGURATION)
    degree = layout["degree"]
    center_state = unpack_shared_vector(center, CONFIGURATION)
    parameter_state = unpack_shared_vector(parameter, CONFIGURATION)
    nodes = increasing_lobatto_nodes(degree)
    tasks = []
    locations = []
    for block, (left, right) in enumerate(zip(mesh[:-1], mesh[1:])):
        theta_nodes = 0.5 * (left + right) + 0.5 * (right - left) * nodes
        for selected in range(1, degree + 1):
            tasks.append((
                float(theta_nodes[selected]),
                float(center_state["rho_blocks"][block, selected]),
                float(parameter_state["rho_blocks"][block, selected]),
                float(center_state["w_blocks"][block, selected]),
                float(parameter_state["w_blocks"][block, selected]),
            ))
            locations.append((block, selected))
    return tasks, locations


def assemble_subdivided_jacobian(center, parameter, mesh, results):
    layout = shared_nodal_layout(CONFIGURATION)
    size = layout["size"]
    degree = layout["degree"]
    domains = layout["domains"]
    derivative = lobatto_differentiation_matrix(degree)
    midpoint = lil_matrix((size, size), dtype=float)
    radius = lil_matrix((size, size), dtype=float)
    rho_row_start = 2 * layout["axis_count"]
    w_row_start = rho_row_start + domains * degree
    maximum_width = 0.0
    maximum_segments = 0
    ordinal = 0
    for block, (left, right) in enumerate(zip(mesh[:-1], mesh[1:])):
        differentiation = 2.0 / (right - left) * derivative
        for selected in range(1, degree + 1):
            bounds, segment_count = results[ordinal]
            ordinal += 1
            maximum_segments = max(maximum_segments, segment_count)
            offset = block * degree + selected - 1
            rho_row = rho_row_start + offset
            w_row = w_row_start + offset
            for local_node in range(degree + 1):
                global_node = block * degree + local_node
                rho_column = variable_column(global_node, "rho", layout)
                w_column = variable_column(global_node, "w", layout)
                midpoint[rho_row, rho_column] += differentiation[
                    selected, local_node
                ]
                w_factor = (
                    math.sin(float(mesh[0]))**3 if global_node == 0 else 1.0
                )
                midpoint[w_row, w_column] += (
                    differentiation[selected, local_node] * w_factor
                )
            jacobian = np.asarray([
                [VInterval(bounds[row, column, 0], bounds[row, column, 1])
                 for column in range(2)]
                for row in range(2)
            ], dtype=object)
            rho_column = variable_column(block * degree + selected, "rho", layout)
            w_column = variable_column(block * degree + selected, "w", layout)
            entries = {
                (rho_row, w_column): -jacobian[0, 1],
                (w_row, rho_column): -jacobian[1, 0],
                (w_row, w_column): -jacobian[1, 1],
            }
            for (row, column), interval in entries.items():
                midpoint[row, column] += interval.midpoint
                halfwidth = max(
                    interval.upper - interval.midpoint,
                    interval.midpoint - interval.lower,
                )
                radius[row, column] += halfwidth
                maximum_width = max(maximum_width, interval.width)
    return midpoint.tocsr(), radius.tocsr(), maximum_width, maximum_segments


def ensure_screen(index):
    stage_id = "physical/G9/base217/D12-M70-P160/correlated-bulk-xi32"
    metadata = {
        "classification": "partial_subdivided_bulk_screen_not_a_certificate",
        "parameter_subdivisions": PARAMETER_SUBDIVISIONS,
        "worker_count": WORKERS,
        "axis_rows_included": False,
    }
    index.register(stage_id, "subdivided-correlated-bulk-screen", 1800.0, metadata)
    reusable = index.validated_path(stage_id)
    if reusable is not None:
        validate_npz(reusable, require_finite=True)
        completion = index.data["stages"][stage_id]["completion_metadata"]
        if (
            completion.get("summary_path") != str(SUMMARY)
            or not SUMMARY.is_file()
            or completion.get("summary_sha256") != sha256_file(SUMMARY)
        ):
            raise RuntimeError("subdivided summary failed recovery validation")
        return json.loads(SUMMARY.read_text()), True
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        with np.load(PREDICTOR) as archive:
            predictor = {key: np.asarray(archive[key]) for key in archive.files}
        with np.load(AFFINE) as archive:
            center = np.asarray(archive["center_vector"])
            parameter = np.asarray(archive["parameter_vector"])
            shape = tuple(np.asarray(archive["jacobian_shape"]).astype(int))
            finite_difference = csc_matrix((
                np.asarray(archive["center_jacobian_data"]),
                np.asarray(archive["center_jacobian_indices"]),
                np.asarray(archive["center_jacobian_indptr"]),
            ), shape=shape).tocsr()
        metric = load_validated_metric(SPLINE_ARCHIVE[LABEL])
        print("building analytic point center", flush=True)
        analytic_center, _, _ = build_bulk_interval_jacobian(
            center, np.zeros_like(center), predictor["mesh"], metric,
        )
        axis_rows = 2 * shared_nodal_layout(CONFIGURATION)["axis_count"]
        analytic_center = analytic_center.tolil()
        analytic_center[:axis_rows, :] = finite_difference[:axis_rows, :]
        analytic_center = analytic_center.tocsc()
        tasks, _ = task_data(center, parameter, predictor["mesh"])
        print(
            f"dispatching {len(tasks)} nodes x {PARAMETER_SUBDIVISIONS} pieces",
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
                if count % 60 == 0:
                    print(f"completed subdivided nodes {count}/{len(tasks)}", flush=True)
        interval_midpoint, interval_radius, max_width, max_segments = (
            assemble_subdivided_jacobian(
                center, parameter, predictor["mesh"], results,
            )
        )
        difference = interval_midpoint - analytic_center
        difference.data = np.abs(difference.data)
        difference = (difference + interval_radius).tolil()
        difference[:axis_rows, :] = 0.0
        difference = difference.tocsr()
        difference.eliminate_zeros()
        factor = splu(analytic_center)
        inverse = factor.solve(np.eye(analytic_center.shape[0]))
        scales = component_scale_vector(predictor["component_scales"])
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
        atomic_write_npz(OUTPUT, **arrays)
        validate_npz(OUTPUT, require_finite=True)
        payload = {
            "schema": "test4d-subdivided-correlated-bulk-screen-v1",
            "status": "partial_subdivided_bulk_screen_complete_not_a_certificate",
            "certificate_claimed": False,
            "parameter_subdivisions": PARAMETER_SUBDIVISIONS,
            "partial_correlated_bulk_Z1_like_upper": partial_z1,
            "maximum_correlated_derivative_width": max_width,
            "maximum_knot_aligned_segments_per_piece": max_segments,
            "all_finite": bool(
                np.isfinite(partial_z1)
                and np.all(np.isfinite(difference.data))
            ),
            "omissions": [
                "regular-axis integral rows",
                "directed high-precision inverse product",
                "Bernstein off-node and coefficient tails",
                "candidate-radius Z2 variation",
            ],
            "archive": {"path": str(OUTPUT), "sha256": sha256_file(OUTPUT)},
        }
        atomic_write_json(SUMMARY, payload)
        index.mark_complete(
            stage_id,
            OUTPUT,
            time.perf_counter() - started,
            {
                "summary_path": str(SUMMARY),
                "summary_sha256": sha256_file(SUMMARY),
            },
        )
        return payload, False
    except Exception as error:
        index.mark_failed(stage_id, repr(error))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--g9-xi32", action="store_true")
    arguments = parser.parse_args()
    if not arguments.g9_xi32:
        raise SystemExit("select --g9-xi32")
    index = recovery_index()
    payload, reused = ensure_screen(index)
    print(json.dumps({"reused": reused, "summary": payload}, indent=2))


if __name__ == "__main__":
    main()
