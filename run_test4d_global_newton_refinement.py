#!/usr/bin/env python3
"""Second floating Newton/scaled-conditioning checkpoint for Test 4D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, diags
from scipy.sparse.linalg import LinearOperator, onenormest, splu

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.global_bvp_collocation import (
    collocation_residual_summary,
    floating_global_collocation_residual,
    global_collocation_sparsity,
    shared_nodal_layout,
    unpack_shared_vector,
)
from bhps.global_bvp_newton import sparse_newton_step
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
FIRST_NEWTON = Path(
    "results/test4d_global_newton_feasibility_stages/"
    "G9_b217_D12_M70_P160_newton_step.npz"
)
FIRST_SUMMARY = Path(
    "results/test4d_global_newton_feasibility_stages/"
    "G9_b217_D12_M70_P160_newton_step_summary.json"
)
PREDICTOR = Path(
    "results/test4d_validated_global_bvp_feasibility_stages/"
    "G9_b217_D12_M70_P160_affine_predictor.npz"
)
RECOVERY = Path("results/test4d_global_newton_refinement_stages")
MANIFEST = Path("results/test4d_global_newton_refinement_recovery.json")
OUTPUT = RECOVERY / "G9_b217_D12_M70_P160_newton_refinement.npz"
SUMMARY = RECOVERY / "G9_b217_D12_M70_P160_newton_refinement_summary.json"
CONFIGURATION = CONFIGURATIONS[0]
LABEL = "G9"
LAUNCH_MIDPOINT = 1.204638671875


def provenance_inputs():
    paths = (
        PROTOCOL,
        FIRST_NEWTON,
        FIRST_SUMMARY,
        PREDICTOR,
        GEOMETRY[LABEL],
        Path("src/bhps/global_bvp_collocation.py"),
        Path("src/bhps/global_bvp_newton.py"),
        Path(__file__),
    )
    return {str(path): sha256_file(path) for path in paths}


def recovery_index():
    summary = json.loads(FIRST_SUMMARY.read_text())
    if summary.get("certificate_claimed") is not False:
        raise RuntimeError("first Newton checkpoint has invalid claim label")
    if summary["newton_archive"]["sha256"] != sha256_file(FIRST_NEWTON):
        raise RuntimeError("first Newton archive hash mismatch")
    return RecoveryIndex(
        MANIFEST,
        PROTOCOL,
        provenance_inputs(),
        maximum_stage_seconds=1800.0,
    )


def scaled_condition_diagnostic(jacobian, mesh, component_scales):
    layout = shared_nodal_layout(CONFIGURATION)
    axis_count = layout["axis_count"]
    domains = layout["domains"]
    degree = layout["degree"]
    srho, su, sw = np.asarray(component_scales, dtype=float)
    column = np.concatenate((
        np.full(axis_count, srho),
        np.full(axis_count, su),
        np.full(domains * degree, srho),
        np.full(domains * degree, sw),
    ))
    halfwidths = 0.5 * np.diff(np.asarray(mesh, dtype=float))
    row = np.concatenate((
        np.full(axis_count, 1.0 / srho),
        np.full(axis_count, 1.0 / su),
        np.repeat(halfwidths / srho, degree),
        np.repeat(halfwidths / sw, degree),
    ))
    scaled = diags(row) @ csr_matrix(jacobian) @ diags(column)
    factor = splu(csc_matrix(scaled))
    matrix_norm = float(onenormest(scaled))
    inverse = LinearOperator(
        scaled.shape,
        matvec=lambda value: factor.solve(value),
        rmatvec=lambda value: factor.solve(value, trans="T"),
        dtype=float,
    )
    inverse_norm = float(onenormest(inverse))
    return {
        "coordinate_scaling": (
            "sealed component scales and domain halfwidth-integrated rows"
        ),
        "matrix_one_norm_estimate": matrix_norm,
        "inverse_one_norm_estimate": inverse_norm,
        "condition_one_norm_estimate": matrix_norm * inverse_norm,
        "minimum_absolute_u_diagonal": float(np.min(np.abs(factor.U.diagonal()))),
        "all_finite": bool(
            np.all(np.isfinite(scaled.data))
            and np.isfinite(matrix_norm)
            and np.isfinite(inverse_norm)
        ),
    }


def run_refinement(index):
    stage_id = "physical/G9/base217/D12-M70-P160/newton-refinement"
    metadata = {
        "classification": "floating_newton_only_not_a_certificate",
        "relative_step": 1e-7,
        "row_scaling": "domain-halfwidth-integrated",
    }
    index.register(stage_id, "floating-newton-refinement", 900.0, metadata)
    reusable = index.validated_path(stage_id)
    layout = shared_nodal_layout(CONFIGURATION)
    required = {
        "updated_vector": (layout["size"],),
        "correction": (layout["size"],),
        "residual_before": (layout["size"],),
        "residual_after": (layout["size"],),
        "jacobian_shape": (2,),
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
            raise RuntimeError("Newton refinement summary failed recovery validation")
        with np.load(reusable) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
        return reusable, arrays, True

    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        with np.load(FIRST_NEWTON) as archive:
            center = np.asarray(archive["updated_vector"])
        with np.load(PREDICTOR) as archive:
            predictor = {key: np.asarray(archive[key]) for key in archive.files}
        geometry = load_geometry(GEOMETRY[LABEL])
        splines = scipy_splines(geometry)

        def residual_function(vector):
            return floating_global_collocation_residual(
                vector,
                LAUNCH_MIDPOINT,
                predictor["mesh"],
                CONFIGURATION,
                float(geometry["z"][-1]),
                splines,
                quadrature_order=32,
            )

        step = sparse_newton_step(
            residual_function,
            center,
            global_collocation_sparsity(CONFIGURATION),
            CONFIGURATION,
            predictor["component_scales"],
            relative_step=1e-7,
        )
        jacobian = step["jacobian"].tocsr()
        arrays = {
            "updated_vector": step["updated_vector"],
            "correction": step["correction"],
            "residual_before": step["residual_before"],
            "residual_after": step["residual_after"],
            "jacobian_data": jacobian.data,
            "jacobian_indices": jacobian.indices,
            "jacobian_indptr": jacobian.indptr,
            "jacobian_shape": np.asarray(jacobian.shape, dtype=np.int64),
        }
        atomic_write_npz(OUTPUT, **arrays)
        validate_npz(OUTPUT, required, require_finite=True)
        state_before = unpack_shared_vector(center, CONFIGURATION)
        state_after = unpack_shared_vector(step["updated_vector"], CONFIGURATION)
        payload = {
            "schema": "test4d-floating-global-newton-refinement-v1",
            "status": "floating_newton_refinement_complete_not_a_certificate",
            "certificate_claimed": False,
            "newton_diagnostics": step["diagnostics"],
            "scaled_condition_diagnostic": scaled_condition_diagnostic(
                jacobian, predictor["mesh"], predictor["component_scales"],
            ),
            "residual_before": collocation_residual_summary(
                step["residual_before"], CONFIGURATION,
            ),
            "residual_after": collocation_residual_summary(
                step["residual_after"], CONFIGURATION,
            ),
            "terminal_w_before": float(state_before["w_shared"][-1]),
            "terminal_w_after": float(state_after["w_shared"][-1]),
            "archive": {"path": str(OUTPUT), "sha256": sha256_file(OUTPUT)},
            "next_required_step": (
                "directed interval enclosure of the affine parameter residual "
                "and Jacobian/tail bounds"
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
        return OUTPUT, arrays, False
    except Exception as error:
        index.mark_failed(stage_id, repr(error))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refine", action="store_true")
    arguments = parser.parse_args()
    if not arguments.refine:
        raise SystemExit("select --refine")
    index = recovery_index()
    _, _, reused = run_refinement(index)
    print(json.dumps({
        "reused": reused,
        "summary": json.loads(SUMMARY.read_text()),
    }, indent=2))


if __name__ == "__main__":
    main()
