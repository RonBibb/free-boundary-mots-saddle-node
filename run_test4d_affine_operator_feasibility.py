#!/usr/bin/env python3
"""Floating affine-cell inverse diagnostics for the sealed Test-4D gate."""

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
    shared_nodal_layout,
    unpack_shared_vector,
)
from bhps.global_bvp_newton import sparse_colored_central_jacobian
from bhps.recovery_indexer import (
    RecoveryIndex,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
    validate_npz,
)
from bhps.validated_global_bvp import CONFIGURATIONS
from run_test4b_validated_interval_no_horizon_certificate import (
    GEOMETRY,
    load_geometry,
    scipy_splines,
)


PROTOCOL = Path("notes/104_test4d_validated_global_bvp_certificate_protocol.md")
PREDICTOR = Path(
    "results/test4d_validated_global_bvp_feasibility_stages/"
    "G9_b217_D12_M70_P160_affine_predictor.npz"
)
REFINEMENT = Path(
    "results/test4d_global_newton_refinement_stages/"
    "G9_b217_D12_M70_P160_newton_refinement.npz"
)
REFINEMENT_SUMMARY = Path(
    "results/test4d_global_newton_refinement_stages/"
    "G9_b217_D12_M70_P160_newton_refinement_summary.json"
)
RECOVERY = Path("results/test4d_affine_operator_feasibility_stages")
MANIFEST = Path("results/test4d_affine_operator_feasibility_recovery.json")
OUTPUT = RECOVERY / "G9_b217_D12_M70_P160_affine_operator.npz"
SUMMARY = RECOVERY / "G9_b217_D12_M70_P160_affine_operator_summary.json"
CONFIGURATION = CONFIGURATIONS[0]
LABEL = "G9"
BASE_INDEX = 217
LAUNCH_LOWER = 1.20458203125
LAUNCH_UPPER = 1.2046953125
LAUNCH_MIDPOINT = 1.204638671875


def provenance_inputs():
    paths = (
        PROTOCOL,
        PREDICTOR,
        REFINEMENT,
        REFINEMENT_SUMMARY,
        GEOMETRY[LABEL],
        Path("src/bhps/global_bvp_collocation.py"),
        Path("src/bhps/global_bvp_newton.py"),
        Path(__file__),
    )
    return {str(path): sha256_file(path) for path in paths}


def recovery_index():
    summary = json.loads(REFINEMENT_SUMMARY.read_text())
    if summary.get("certificate_claimed") is not False:
        raise RuntimeError("Newton refinement has invalid claim label")
    if summary["archive"]["sha256"] != sha256_file(REFINEMENT):
        raise RuntimeError("Newton refinement archive hash mismatch")
    return RecoveryIndex(
        MANIFEST,
        PROTOCOL,
        provenance_inputs(),
        maximum_stage_seconds=1800.0,
    )


def component_scale_vector(component_scales):
    layout = shared_nodal_layout(CONFIGURATION)
    srho, su, sw = np.asarray(component_scales, dtype=float)
    return np.concatenate((
        np.full(layout["axis_count"], srho),
        np.full(layout["axis_count"], su),
        np.full(layout["bulk_free_count"], srho),
        np.full(layout["bulk_free_count"], sw),
    ))


def normalized_vector_norm(vector, scale_vector):
    return float(np.max(np.abs(np.asarray(vector)) / scale_vector))


def normalized_inverse_difference_norm(factor, difference, scale_vector):
    right_scaled = difference @ diags(scale_vector)
    image = factor.solve(right_scaled.toarray())
    image /= scale_vector[:, None]
    return float(np.max(np.sum(np.abs(image), axis=1)))


def parameter_vector(predictor):
    synthetic = {
        key.replace("_parameter", "_center"): value
        for key, value in predictor.items() if key.endswith("_parameter")
    }
    return pack_predictor_center_shared(synthetic, CONFIGURATION)


def ensure_affine_operator(index):
    stage_id = "physical/G9/base217/D12-M70-P160/affine-operator"
    metadata = {
        "classification": "floating_affine_operator_not_a_certificate",
        "launch_interval": [LAUNCH_LOWER, LAUNCH_UPPER],
        "relative_step": 1e-7,
    }
    index.register(stage_id, "floating-affine-operator", 900.0, metadata)
    reusable = index.validated_path(stage_id)
    layout = shared_nodal_layout(CONFIGURATION)
    required = {
        "center_vector": (layout["size"],),
        "parameter_vector": (layout["size"],),
        "lower_residual": (layout["size"],),
        "upper_residual": (layout["size"],),
        "lower_correction": (layout["size"],),
        "upper_correction": (layout["size"],),
    }
    if reusable is not None:
        validate_npz(reusable, required, require_finite=True)
        completion = index.data["stages"][stage_id].get(
            "completion_metadata", {}
        )
        if (
            completion.get("summary_path") != str(SUMMARY)
            or not SUMMARY.is_file()
            or completion.get("summary_sha256") != sha256_file(SUMMARY)
        ):
            raise RuntimeError("affine summary failed recovery validation")
        return json.loads(SUMMARY.read_text()), True

    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        with np.load(PREDICTOR) as archive:
            predictor = {key: np.asarray(archive[key]) for key in archive.files}
        with np.load(REFINEMENT) as archive:
            center = np.asarray(archive["updated_vector"])
        geometry = load_geometry(GEOMETRY[LABEL])
        splines = scipy_splines(geometry)

        def at_launch(launch):
            return lambda vector: floating_global_collocation_residual(
                vector,
                launch,
                predictor["mesh"],
                CONFIGURATION,
                float(geometry["z"][-1]),
                splines,
                quadrature_order=32,
            )

        affine_parameter = parameter_vector(predictor)
        lower_vector = center - affine_parameter
        upper_vector = center + affine_parameter
        center_function = at_launch(LAUNCH_MIDPOINT)
        lower_function = at_launch(LAUNCH_LOWER)
        upper_function = at_launch(LAUNCH_UPPER)
        pattern = global_collocation_sparsity(CONFIGURATION)
        center_jacobian, center_fd = sparse_colored_central_jacobian(
            center_function, center, pattern, relative_step=1e-7,
        )
        lower_jacobian, lower_fd = sparse_colored_central_jacobian(
            lower_function, lower_vector, pattern, relative_step=1e-7,
        )
        upper_jacobian, upper_fd = sparse_colored_central_jacobian(
            upper_function, upper_vector, pattern, relative_step=1e-7,
        )
        center_residual = center_function(center)
        lower_residual = lower_function(lower_vector)
        upper_residual = upper_function(upper_vector)
        factor = splu(csc_matrix(center_jacobian))
        center_correction = factor.solve(-center_residual)
        lower_correction = factor.solve(-lower_residual)
        upper_correction = factor.solve(-upper_residual)
        scales = component_scale_vector(predictor["component_scales"])
        lower_z1 = normalized_inverse_difference_norm(
            factor, lower_jacobian - center_jacobian, scales,
        )
        upper_z1 = normalized_inverse_difference_norm(
            factor, upper_jacobian - center_jacobian, scales,
        )
        lower_state = unpack_shared_vector(
            lower_vector + lower_correction, CONFIGURATION,
        )
        upper_state = unpack_shared_vector(
            upper_vector + upper_correction, CONFIGURATION,
        )
        arrays = {
            "center_vector": center,
            "parameter_vector": affine_parameter,
            "center_residual": center_residual,
            "lower_residual": lower_residual,
            "upper_residual": upper_residual,
            "center_correction": center_correction,
            "lower_correction": lower_correction,
            "upper_correction": upper_correction,
            "center_jacobian_data": center_jacobian.data,
            "center_jacobian_indices": center_jacobian.indices,
            "center_jacobian_indptr": center_jacobian.indptr,
            "lower_jacobian_data": lower_jacobian.data,
            "lower_jacobian_indices": lower_jacobian.indices,
            "lower_jacobian_indptr": lower_jacobian.indptr,
            "upper_jacobian_data": upper_jacobian.data,
            "upper_jacobian_indices": upper_jacobian.indices,
            "upper_jacobian_indptr": upper_jacobian.indptr,
            "jacobian_shape": np.asarray(center_jacobian.shape, dtype=np.int64),
        }
        atomic_write_npz(OUTPUT, **arrays)
        validate_npz(OUTPUT, required, require_finite=True)
        payload = {
            "schema": "test4d-floating-affine-operator-v1",
            "status": "floating_affine_operator_complete_not_a_certificate",
            "certificate_claimed": False,
            "geometry": LABEL,
            "base_index": BASE_INDEX,
            "launch_interval": [LAUNCH_LOWER, LAUNCH_UPPER],
            "finite_difference_diagnostics": {
                "center": center_fd,
                "lower": lower_fd,
                "upper": upper_fd,
            },
            "point_Y_proxies": {
                "center": normalized_vector_norm(center_correction, scales),
                "lower": normalized_vector_norm(lower_correction, scales),
                "upper": normalized_vector_norm(upper_correction, scales),
                "maximum": max(
                    normalized_vector_norm(center_correction, scales),
                    normalized_vector_norm(lower_correction, scales),
                    normalized_vector_norm(upper_correction, scales),
                ),
            },
            "endpoint_Z1_proxies": {
                "lower": lower_z1,
                "upper": upper_z1,
                "maximum": max(lower_z1, upper_z1),
            },
            "corrected_endpoint_terminal_w": {
                "lower": float(lower_state["w_shared"][-1]),
                "upper": float(upper_state["w_shared"][-1]),
            },
            "all_finite": bool(
                all(np.all(np.isfinite(value)) for value in arrays.values())
                and np.isfinite(lower_z1)
                and np.isfinite(upper_z1)
            ),
            "archive": {"path": str(OUTPUT), "sha256": sha256_file(OUTPUT)},
            "interpretation": (
                "Point/endpoint diagnostics only: interpolation between xi "
                "and all interval/tail contributions remain unproved."
            ),
            "next_required_step": (
                "interval subdivision in xi plus directed spline-composition "
                "and coefficient-tail bounds"
            ),
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
    parser.add_argument("--g9-cell", action="store_true")
    arguments = parser.parse_args()
    if not arguments.g9_cell:
        raise SystemExit("select --g9-cell")
    index = recovery_index()
    payload, reused = ensure_affine_operator(index)
    print(json.dumps({"reused": reused, "summary": payload}, indent=2))


if __name__ == "__main__":
    main()
