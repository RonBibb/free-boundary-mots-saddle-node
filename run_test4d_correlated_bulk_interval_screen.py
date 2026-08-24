#!/usr/bin/env python3
"""Correlated affine-parameter bulk Jacobian screen for sealed Test 4D."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import csc_matrix, diags, lil_matrix
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.correlated_affine_spline import (
    correlated_divergence_jacobian_hull,
)
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
from bhps.validated_global_bvp import CONFIGURATIONS
from run_test4b_validated_interval_no_horizon_certificate import (
    SPLINE_ARCHIVE,
    load_validated_metric,
)
from run_test4d_bulk_interval_jacobian_screen import (
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
AFFINE_SUMMARY = Path(
    "results/test4d_affine_operator_feasibility_stages/"
    "G9_b217_D12_M70_P160_affine_operator_summary.json"
)
RECOVERY = Path("results/test4d_correlated_bulk_interval_screen_stages")
MANIFEST = Path("results/test4d_correlated_bulk_interval_screen_recovery.json")
OUTPUT = RECOVERY / "G9_b217_D12_M70_P160_correlated_bulk_xi1.npz"
SUMMARY = RECOVERY / "G9_b217_D12_M70_P160_correlated_bulk_xi1_summary.json"
CONFIGURATION = CONFIGURATIONS[0]
LABEL = "G9"


def provenance_inputs():
    paths = (
        PROTOCOL,
        PREDICTOR,
        AFFINE,
        AFFINE_SUMMARY,
        SPLINE_ARCHIVE[LABEL],
        Path("src/bhps/correlated_affine_spline.py"),
        Path("src/bhps/global_bvp_collocation.py"),
        Path("run_test4d_bulk_interval_jacobian_screen.py"),
        Path("tests/test_correlated_affine_spline.py"),
        Path(__file__),
    )
    return {str(path): sha256_file(path) for path in paths}


def recovery_index():
    if json.loads(AFFINE_SUMMARY.read_text()).get("certificate_claimed") is not False:
        raise RuntimeError("affine precursor has invalid claim label")
    return RecoveryIndex(
        MANIFEST,
        PROTOCOL,
        provenance_inputs(),
        maximum_stage_seconds=1800.0,
    )


def build_correlated_bulk_jacobian(center, parameter, mesh, metric):
    layout = shared_nodal_layout(CONFIGURATION)
    size = layout["size"]
    degree = layout["degree"]
    domains = layout["domains"]
    center_state = unpack_shared_vector(center, CONFIGURATION)
    parameter_state = unpack_shared_vector(parameter, CONFIGURATION)
    nodes = increasing_lobatto_nodes(degree)
    derivative = lobatto_differentiation_matrix(degree)
    midpoint = lil_matrix((size, size), dtype=float)
    radius = lil_matrix((size, size), dtype=float)
    rho_row_start = 2 * layout["axis_count"]
    w_row_start = rho_row_start + domains * degree
    maximum_width = 0.0
    maximum_segments = 0
    for block, (left, right) in enumerate(zip(mesh[:-1], mesh[1:])):
        theta_nodes = 0.5 * (left + right) + 0.5 * (right - left) * nodes
        differentiation = 2.0 / (right - left) * derivative
        for selected in range(1, degree + 1):
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
            theta = float(theta_nodes[selected])
            jacobian, audit = correlated_divergence_jacobian_hull(
                theta,
                center_state["rho_blocks"][block, selected],
                parameter_state["rho_blocks"][block, selected],
                center_state["w_blocks"][block, selected],
                parameter_state["w_blocks"][block, selected],
                metric,
            )
            maximum_segments = max(
                maximum_segments, audit["parameter_segment_count"],
            )
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
        if block % 5 == 4:
            print(f"completed correlated bulk block {block + 1}/{domains}", flush=True)
    return midpoint.tocsr(), radius.tocsr(), maximum_width, maximum_segments


def ensure_screen(index):
    stage_id = "physical/G9/base217/D12-M70-P160/correlated-bulk-xi1"
    metadata = {
        "classification": "partial_correlated_bulk_screen_not_a_certificate",
        "xi_cover": [-1.0, 1.0],
        "knot_aligned": True,
        "axis_rows_included": False,
        "tails_included": False,
    }
    index.register(stage_id, "correlated-bulk-interval-jacobian", 1800.0, metadata)
    reusable = index.validated_path(stage_id)
    if reusable is not None:
        validate_npz(reusable, require_finite=True)
        completion = index.data["stages"][stage_id]["completion_metadata"]
        if (
            completion.get("summary_path") != str(SUMMARY)
            or not SUMMARY.is_file()
            or completion.get("summary_sha256") != sha256_file(SUMMARY)
        ):
            raise RuntimeError("correlated screen summary failed validation")
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
            center_jacobian = csc_matrix((
                np.asarray(archive["center_jacobian_data"]),
                np.asarray(archive["center_jacobian_indices"]),
                np.asarray(archive["center_jacobian_indptr"]),
            ), shape=shape).tocsr()
        metric = load_validated_metric(SPLINE_ARCHIVE[LABEL])
        interval_midpoint, interval_radius, max_width, max_segments = (
            build_correlated_bulk_jacobian(
                center, parameter, predictor["mesh"], metric,
            )
        )
        difference = interval_midpoint - center_jacobian
        difference.data = np.abs(difference.data)
        difference = (difference + interval_radius).tolil()
        axis_rows = 2 * shared_nodal_layout(CONFIGURATION)["axis_count"]
        difference[:axis_rows, :] = 0.0
        difference = difference.tocsr()
        difference.eliminate_zeros()
        factor = splu(csc_matrix(center_jacobian))
        inverse = factor.solve(np.eye(center_jacobian.shape[0]))
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
            "schema": "test4d-correlated-bulk-interval-screen-v1",
            "status": "partial_correlated_bulk_screen_complete_not_a_certificate",
            "certificate_claimed": False,
            "xi_cover": [-1.0, 1.0],
            "partial_correlated_bulk_Z1_like_upper": partial_z1,
            "maximum_correlated_derivative_width": max_width,
            "maximum_knot_aligned_parameter_segments": max_segments,
            "all_finite": bool(
                np.isfinite(partial_z1)
                and np.all(np.isfinite(difference.data))
            ),
            "omissions": [
                "regular-axis integral rows",
                "directed high-precision inverse product",
                "Bernstein off-node and coefficient-tail terms",
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
    parser.add_argument("--g9-full-xi", action="store_true")
    arguments = parser.parse_args()
    if not arguments.g9_full_xi:
        raise SystemExit("select --g9-full-xi")
    index = recovery_index()
    payload, reused = ensure_screen(index)
    print(json.dumps({"reused": reused, "summary": payload}, indent=2))


if __name__ == "__main__":
    main()
