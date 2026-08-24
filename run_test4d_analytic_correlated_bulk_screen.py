#!/usr/bin/env python3
"""Analytic-center correlated bulk Z1 screen for sealed Test 4D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.global_bvp_collocation import shared_nodal_layout
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
    build_bulk_interval_jacobian,
    component_scale_vector,
)
from run_test4d_correlated_bulk_interval_screen import (
    build_correlated_bulk_jacobian,
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
CORRELATED_V1 = Path(
    "results/test4d_correlated_bulk_interval_screen_stages/"
    "G9_b217_D12_M70_P160_correlated_bulk_xi1_summary.json"
)
RECOVERY = Path("results/test4d_analytic_correlated_bulk_screen_stages")
MANIFEST = Path("results/test4d_analytic_correlated_bulk_screen_recovery.json")
OUTPUT = RECOVERY / "G9_b217_D12_M70_P160_analytic_correlated_bulk_xi1.npz"
SUMMARY = RECOVERY / "G9_b217_D12_M70_P160_analytic_correlated_bulk_xi1_summary.json"
CONFIGURATION = CONFIGURATIONS[0]
LABEL = "G9"


def provenance_inputs():
    paths = (
        PROTOCOL,
        PREDICTOR,
        AFFINE,
        CORRELATED_V1,
        SPLINE_ARCHIVE[LABEL],
        Path("src/bhps/correlated_affine_spline.py"),
        Path("run_test4d_bulk_interval_jacobian_screen.py"),
        Path("run_test4d_correlated_bulk_interval_screen.py"),
        Path(__file__),
    )
    return {str(path): sha256_file(path) for path in paths}


def recovery_index():
    if json.loads(CORRELATED_V1.read_text()).get("certificate_claimed") is not False:
        raise RuntimeError("correlated v1 precursor has invalid claim label")
    return RecoveryIndex(
        MANIFEST, PROTOCOL, provenance_inputs(), maximum_stage_seconds=1800.0,
    )


def ensure_screen(index):
    stage_id = "physical/G9/base217/D12-M70-P160/analytic-correlated-bulk-xi1"
    metadata = {
        "classification": "partial_analytic_correlated_screen_not_a_certificate",
        "xi_cover": [-1.0, 1.0],
        "analytic_bulk_center": True,
        "axis_rows_included": False,
    }
    index.register(stage_id, "analytic-correlated-bulk-screen", 1800.0, metadata)
    reusable = index.validated_path(stage_id)
    if reusable is not None:
        validate_npz(reusable, require_finite=True)
        completion = index.data["stages"][stage_id]["completion_metadata"]
        if (
            completion.get("summary_path") != str(SUMMARY)
            or not SUMMARY.is_file()
            or completion.get("summary_sha256") != sha256_file(SUMMARY)
        ):
            raise RuntimeError("analytic correlated summary failed validation")
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
        analytic_center, point_radius, point_width = build_bulk_interval_jacobian(
            center, np.zeros_like(center), predictor["mesh"], metric,
        )
        axis_rows = 2 * shared_nodal_layout(CONFIGURATION)["axis_count"]
        analytic_center = analytic_center.tolil()
        analytic_center[:axis_rows, :] = finite_difference[:axis_rows, :]
        analytic_center = analytic_center.tocsc()
        print("building correlated full-cell enclosure", flush=True)
        interval_midpoint, interval_radius, max_width, max_segments = (
            build_correlated_bulk_jacobian(
                center, parameter, predictor["mesh"], metric,
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
        point_difference = point_radius.tolil()
        point_difference[:axis_rows, :] = 0.0
        point_difference = point_difference.tocsr()
        point_image = inverse_scaled @ (point_difference @ diags(scales))
        point_roundoff_z = float(np.max(
            np.asarray(point_image.sum(axis=1)).reshape(-1)
        ))
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
            "schema": "test4d-analytic-correlated-bulk-screen-v1",
            "status": "partial_analytic_correlated_screen_complete_not_a_certificate",
            "certificate_claimed": False,
            "xi_cover": [-1.0, 1.0],
            "partial_correlated_bulk_Z1_like_upper": partial_z1,
            "point_interval_roundoff_Z_like_upper": point_roundoff_z,
            "maximum_point_derivative_width": point_width,
            "maximum_correlated_derivative_width": max_width,
            "maximum_knot_aligned_parameter_segments": max_segments,
            "minimum_absolute_u_diagonal": float(
                np.min(np.abs(factor.U.diagonal()))
            ),
            "all_finite": bool(
                np.isfinite(partial_z1)
                and np.isfinite(point_roundoff_z)
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
    parser.add_argument("--g9-full-xi", action="store_true")
    arguments = parser.parse_args()
    if not arguments.g9_full_xi:
        raise SystemExit("select --g9-full-xi")
    index = recovery_index()
    payload, reused = ensure_screen(index)
    print(json.dumps({"reused": reused, "summary": payload}, indent=2))


if __name__ == "__main__":
    main()
