#!/usr/bin/env python3
"""Floating depth-1 child screens for G9 base cell 217 in Test 4D."""

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

from bhps.global_bvp_collocation import (
    floating_global_collocation_residual,
    global_collocation_sparsity,
    pack_predictor_center_shared,
    unpack_shared_vector,
)
from bhps.global_bvp_newton import (
    sparse_colored_central_jacobian,
    sparse_newton_step,
)
from bhps.recovery_indexer import (
    RecoveryIndex,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
    validate_npz,
)
from bhps.validated_global_bvp import CONFIGURATIONS, affine_floating_predictor
from run_test4b_validated_interval_no_horizon_certificate import (
    GEOMETRY,
    SPLINE_ARCHIVE,
    load_geometry,
    load_validated_metric,
    scipy_splines,
)
from run_test4d_affine_operator_feasibility import (
    component_scale_vector,
    normalized_inverse_difference_norm,
    normalized_vector_norm,
    parameter_vector,
)


PROTOCOL = Path("notes/104_test4d_validated_global_bvp_certificate_protocol.md")
PARENT_SCREEN = Path(
    "results/test4d_subdivided_correlated_bulk_screen_stages/"
    "G9_b217_D12_M70_P160_correlated_bulk_xi32_summary.json"
)
RECOVERY = Path("results/test4d_g9_depth1_floating_screen_stages")
MANIFEST = Path("results/test4d_g9_depth1_floating_screen_recovery.json")
SUMMARY = RECOVERY / "G9_b217_depth1_D12_M70_P160_floating_summary.json"
CONFIGURATION = CONFIGURATIONS[0]
LABEL = "G9"
PARENT_LOWER = 1.20458203125
PARENT_UPPER = 1.2046953125
PARENT_MIDPOINT = 0.5 * (PARENT_LOWER + PARENT_UPPER)
CHILDREN = (
    ("0", PARENT_LOWER, PARENT_MIDPOINT),
    ("1", PARENT_MIDPOINT, PARENT_UPPER),
)


def provenance_inputs():
    paths = (
        PROTOCOL,
        PARENT_SCREEN,
        GEOMETRY[LABEL],
        SPLINE_ARCHIVE[LABEL],
        Path("src/bhps/validated_global_bvp.py"),
        Path("src/bhps/global_bvp_collocation.py"),
        Path("src/bhps/global_bvp_newton.py"),
        Path("run_test4d_affine_operator_feasibility.py"),
        Path(__file__),
    )
    return {str(path): sha256_file(path) for path in paths}


def recovery_index():
    parent = json.loads(PARENT_SCREEN.read_text())
    if parent.get("certificate_claimed") is not False:
        raise RuntimeError("parent interval screen has invalid claim label")
    return RecoveryIndex(
        MANIFEST, PROTOCOL, provenance_inputs(), maximum_stage_seconds=1800.0,
    )


def output_path(path):
    return RECOVERY / f"G9_b217_p{path}_D12_M70_P160_floating_operator.npz"


def ensure_child(index, path, lower, upper):
    stage_id = f"physical/G9/base217/path{path}/D12-M70-P160/floating-operator"
    metadata = {
        "classification": "floating_child_operator_not_a_certificate",
        "subdivision_path": path,
        "launch_interval": [lower, upper],
        "relative_step": 1e-7,
    }
    index.register(stage_id, "floating-depth1-child-operator", 900.0, metadata)
    reusable = index.validated_path(stage_id)
    if reusable is not None:
        validate_npz(reusable, require_finite=True)
        return index.data["stages"][stage_id]["completion_metadata"]["diagnostics"], True
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        geometry = load_geometry(GEOMETRY[LABEL])
        metric = load_validated_metric(SPLINE_ARCHIVE[LABEL])
        splines = scipy_splines(geometry)
        predictor = affine_floating_predictor(
            lower, upper, metric, splines, CONFIGURATION,
        )
        center = pack_predictor_center_shared(predictor, CONFIGURATION)
        launch_midpoint = 0.5 * (lower + upper)

        def residual_at_launch(launch):
            return lambda vector: floating_global_collocation_residual(
                vector,
                launch,
                predictor["mesh"],
                CONFIGURATION,
                float(geometry["z"][-1]),
                splines,
                quadrature_order=32,
            )

        pattern = global_collocation_sparsity(CONFIGURATION)
        center_function = residual_at_launch(launch_midpoint)
        first = sparse_newton_step(
            center_function,
            center,
            pattern,
            CONFIGURATION,
            predictor["component_scales"],
            relative_step=1e-7,
        )
        refined_center = first["updated_vector"]
        affine_parameter = parameter_vector(predictor)
        lower_vector = refined_center - affine_parameter
        upper_vector = refined_center + affine_parameter
        lower_function = residual_at_launch(lower)
        upper_function = residual_at_launch(upper)
        center_jacobian, _ = sparse_colored_central_jacobian(
            center_function, refined_center, pattern, relative_step=1e-7,
        )
        lower_jacobian, _ = sparse_colored_central_jacobian(
            lower_function, lower_vector, pattern, relative_step=1e-7,
        )
        upper_jacobian, _ = sparse_colored_central_jacobian(
            upper_function, upper_vector, pattern, relative_step=1e-7,
        )
        factor = splu(csc_matrix(center_jacobian))
        residual_center = center_function(refined_center)
        residual_lower = lower_function(lower_vector)
        residual_upper = upper_function(upper_vector)
        correction_center = factor.solve(-residual_center)
        correction_lower = factor.solve(-residual_lower)
        correction_upper = factor.solve(-residual_upper)
        scales = component_scale_vector(predictor["component_scales"])
        z1_lower = normalized_inverse_difference_norm(
            factor, lower_jacobian - center_jacobian, scales,
        )
        z1_upper = normalized_inverse_difference_norm(
            factor, upper_jacobian - center_jacobian, scales,
        )
        lower_state = unpack_shared_vector(
            lower_vector + correction_lower, CONFIGURATION,
        )
        upper_state = unpack_shared_vector(
            upper_vector + correction_upper, CONFIGURATION,
        )
        jacobian = center_jacobian.tocsr()
        arrays = {
            "center_vector": refined_center,
            "parameter_vector": affine_parameter,
            "center_residual": residual_center,
            "lower_residual": residual_lower,
            "upper_residual": residual_upper,
            "center_correction": correction_center,
            "lower_correction": correction_lower,
            "upper_correction": correction_upper,
            "component_scales": predictor["component_scales"],
            "mesh": predictor["mesh"],
            "center_jacobian_data": jacobian.data,
            "center_jacobian_indices": jacobian.indices,
            "center_jacobian_indptr": jacobian.indptr,
            "jacobian_shape": np.asarray(jacobian.shape, dtype=np.int64),
        }
        output = output_path(path)
        atomic_write_npz(output, **arrays)
        validate_npz(output, require_finite=True)
        y_values = {
            "center": normalized_vector_norm(correction_center, scales),
            "lower": normalized_vector_norm(correction_lower, scales),
            "upper": normalized_vector_norm(correction_upper, scales),
        }
        diagnostics = {
            "subdivision_path": path,
            "launch_interval": [lower, upper],
            "status": "floating_child_operator_complete_not_a_certificate",
            "certificate_claimed": False,
            "point_Y_proxies": {**y_values, "maximum": max(y_values.values())},
            "endpoint_Z1_proxies": {
                "lower": z1_lower,
                "upper": z1_upper,
                "maximum": max(z1_lower, z1_upper),
            },
            "corrected_endpoint_terminal_w": {
                "lower": float(lower_state["w_shared"][-1]),
                "upper": float(upper_state["w_shared"][-1]),
            },
            "center_newton": first["diagnostics"],
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
    stage_id = "physical/G9/base217/depth1/D12-M70-P160/floating-summary"
    metadata = {
        "paths": [record["subdivision_path"] for record in records],
        "archive_hashes": [record["archive"]["sha256"] for record in records],
    }
    index.register(stage_id, "floating-depth1-summary", 120.0, metadata)
    reusable = index.validated_path(stage_id)
    if reusable is not None:
        return json.loads(reusable.read_text()), True
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        payload = {
            "schema": "test4d-g9-depth1-floating-screen-v1",
            "status": "floating_depth1_screen_complete_not_a_certificate",
            "certificate_claimed": False,
            "records": records,
            "maximum_point_Y_proxy": max(
                record["point_Y_proxies"]["maximum"] for record in records
            ),
            "maximum_endpoint_Z1_proxy": max(
                record["endpoint_Z1_proxies"]["maximum"] for record in records
            ),
            "interpretation": (
                "Point diagnostics only; child interval, axis, tail, and Z2 "
                "bounds remain required."
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
    for path, lower, upper in CHILDREN:
        record, reused = ensure_child(index, path, lower, upper)
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
