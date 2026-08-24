#!/usr/bin/env python3
"""Floating D16-M70-P160 escalation for the two G9 depth-1 children."""

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


PROTOCOL = Path("notes/104_test4d_validated_global_bvp_certificate_protocol.md")
D12_Z2_SUMMARY = Path(
    "results/test4d_g9_depth1_z2_tube_screen_stages/"
    "G9_b217_depth1_D12_M70_P160_z2_tube_summary.json"
)
RECOVERY = Path("results/test4d_g9_d16_floating_feasibility_stages")
MANIFEST = Path("results/test4d_g9_d16_floating_feasibility_recovery.json")
SUMMARY = RECOVERY / "G9_b217_depth1_D16_M70_P160_floating_summary.json"
CONFIGURATION = CONFIGURATIONS[1]
LABEL = "G9"
CHILDREN = (
    ("0", 1.20458203125, 1.204638671875),
    ("1", 1.204638671875, 1.2046953125),
)


def scale_vector(component_scales):
    layout = shared_nodal_layout(CONFIGURATION)
    srho, su, sw = np.asarray(component_scales, dtype=float)
    return np.concatenate((
        np.full(layout["axis_count"], srho),
        np.full(layout["axis_count"], su),
        np.full(layout["bulk_free_count"], srho),
        np.full(layout["bulk_free_count"], sw),
    ))


def normalized_norm(vector, scales):
    return float(np.max(np.abs(np.asarray(vector)) / scales))


def inverse_difference_norm(factor, difference, scales):
    image = factor.solve((difference @ diags(scales)).toarray())
    image /= scales[:, None]
    return float(np.max(np.sum(np.abs(image), axis=1)))


def predictor_parameter_vector(predictor):
    synthetic = {
        key.replace("_parameter", "_center"): value
        for key, value in predictor.items() if key.endswith("_parameter")
    }
    return pack_predictor_center_shared(synthetic, CONFIGURATION)


def provenance_inputs():
    paths = (
        PROTOCOL,
        D12_Z2_SUMMARY,
        GEOMETRY[LABEL],
        SPLINE_ARCHIVE[LABEL],
        Path("src/bhps/validated_global_bvp.py"),
        Path("src/bhps/global_bvp_collocation.py"),
        Path("src/bhps/global_bvp_newton.py"),
        Path(__file__),
    )
    return {str(path): sha256_file(path) for path in paths}


def recovery_index():
    if json.loads(D12_Z2_SUMMARY.read_text()).get("certificate_claimed") is not False:
        raise RuntimeError("D12 precursor has invalid claim label")
    return RecoveryIndex(
        MANIFEST, PROTOCOL, provenance_inputs(), maximum_stage_seconds=1800.0,
    )


def output_path(path):
    return RECOVERY / f"G9_b217_p{path}_D16_M70_P160_floating_operator.npz"


def ensure_child(index, path, lower, upper):
    stage_id = f"physical/G9/base217/path{path}/D16-M70-P160/floating-operator"
    metadata = {
        "classification": "floating_d16_escalation_not_a_certificate",
        "launch_interval": [lower, upper],
        "relative_step": 1e-7,
    }
    index.register(stage_id, "floating-d16-child-operator", 1800.0, metadata)
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
        midpoint = 0.5 * (lower + upper)

        def at_launch(launch):
            return lambda vector: floating_global_collocation_residual(
                vector, launch, predictor["mesh"], CONFIGURATION,
                float(geometry["z"][-1]), splines, quadrature_order=32,
            )

        pattern = global_collocation_sparsity(CONFIGURATION)
        center_function = at_launch(midpoint)
        first = sparse_newton_step(
            center_function, center, pattern, CONFIGURATION,
            predictor["component_scales"], relative_step=1e-7,
        )
        refined = first["updated_vector"]
        parameter = predictor_parameter_vector(predictor)
        lower_vector = refined - parameter
        upper_vector = refined + parameter
        lower_function = at_launch(lower)
        upper_function = at_launch(upper)
        center_jacobian, _ = sparse_colored_central_jacobian(
            center_function, refined, pattern, relative_step=1e-7,
        )
        lower_jacobian, _ = sparse_colored_central_jacobian(
            lower_function, lower_vector, pattern, relative_step=1e-7,
        )
        upper_jacobian, _ = sparse_colored_central_jacobian(
            upper_function, upper_vector, pattern, relative_step=1e-7,
        )
        factor = splu(csc_matrix(center_jacobian))
        residual_center = center_function(refined)
        residual_lower = lower_function(lower_vector)
        residual_upper = upper_function(upper_vector)
        correction_center = factor.solve(-residual_center)
        correction_lower = factor.solve(-residual_lower)
        correction_upper = factor.solve(-residual_upper)
        scales = scale_vector(predictor["component_scales"])
        y_values = [
            normalized_norm(value, scales)
            for value in (correction_center, correction_lower, correction_upper)
        ]
        z_values = [
            inverse_difference_norm(
                factor, jacobian - center_jacobian, scales,
            )
            for jacobian in (lower_jacobian, upper_jacobian)
        ]
        lower_state = unpack_shared_vector(
            lower_vector + correction_lower, CONFIGURATION,
        )
        upper_state = unpack_shared_vector(
            upper_vector + correction_upper, CONFIGURATION,
        )
        jacobian = center_jacobian.tocsr()
        output = output_path(path)
        atomic_write_npz(
            output,
            center_vector=refined,
            parameter_vector=parameter,
            center_residual=residual_center,
            lower_residual=residual_lower,
            upper_residual=residual_upper,
            center_correction=correction_center,
            lower_correction=correction_lower,
            upper_correction=correction_upper,
            component_scales=predictor["component_scales"],
            mesh=predictor["mesh"],
            center_jacobian_data=jacobian.data,
            center_jacobian_indices=jacobian.indices,
            center_jacobian_indptr=jacobian.indptr,
            jacobian_shape=np.asarray(jacobian.shape, dtype=np.int64),
        )
        validate_npz(output, require_finite=True)
        diagnostics = {
            "subdivision_path": path,
            "launch_interval": [lower, upper],
            "status": "floating_d16_escalation_complete_not_a_certificate",
            "certificate_claimed": False,
            "component_scales": predictor["component_scales"].tolist(),
            "point_Y_proxies": {
                "center": y_values[0], "lower": y_values[1],
                "upper": y_values[2], "maximum": max(y_values),
            },
            "endpoint_Z1_proxies": {
                "lower": z_values[0], "upper": z_values[1],
                "maximum": max(z_values),
            },
            "corrected_endpoint_terminal_w": {
                "lower": float(lower_state["w_shared"][-1]),
                "upper": float(upper_state["w_shared"][-1]),
            },
            "first_newton": first["diagnostics"],
            "all_finite": bool(
                all(np.all(np.isfinite(value)) for value in (
                    residual_center, residual_lower, residual_upper,
                    correction_center, correction_lower, correction_upper,
                ))
            ),
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
    stage_id = "physical/G9/base217/depth1/D16-M70-P160/floating-summary"
    index.register(
        stage_id, "floating-d16-summary", 120.0,
        {"archive_hashes": [record["archive"]["sha256"] for record in records]},
    )
    reusable = index.validated_path(stage_id)
    if reusable is not None:
        return json.loads(reusable.read_text()), True
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        payload = {
            "schema": "test4d-g9-depth1-d16-floating-feasibility-v1",
            "status": "floating_d16_escalation_complete_not_a_certificate",
            "certificate_claimed": False,
            "records": records,
            "maximum_point_Y_proxy": max(
                record["point_Y_proxies"]["maximum"] for record in records
            ),
            "maximum_endpoint_Z1_proxy": max(
                record["endpoint_Z1_proxies"]["maximum"] for record in records
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
