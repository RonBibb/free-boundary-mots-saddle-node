#!/usr/bin/env python3
"""Sealed A=7.94 point-root floating operator controls for Test 4D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.global_bvp_collocation import (
    collocation_residual_summary,
    floating_global_collocation_residual,
    global_collocation_sparsity,
    pack_predictor_center_shared,
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
from bhps.validated_global_bvp import (
    CONFIGURATIONS,
    floating_chebyshev_predictor,
)
from run_test4b_validated_interval_no_horizon_certificate import (
    GEOMETRY,
    SPLINE_ARCHIVE,
    load_geometry,
    load_validated_metric,
    scipy_splines,
)


PROTOCOL = Path("notes/104_test4d_validated_global_bvp_certificate_protocol.md")
G9_AFFINE_SUMMARY = Path(
    "results/test4d_affine_operator_feasibility_stages/"
    "G9_b217_D12_M70_P160_affine_operator_summary.json"
)
RECOVERY = Path("results/test4d_a794_point_operator_control_stages")
MANIFEST = Path("results/test4d_a794_point_operator_control_recovery.json")
SUMMARY = RECOVERY / "A794_G7_D12_M70_P160_point_operator_summary.json"
CONFIGURATION = CONFIGURATIONS[0]
LABEL = "A794_G7"
ROOTS = (1.2044718608, 1.2080960720)


def provenance_inputs():
    paths = (
        PROTOCOL,
        G9_AFFINE_SUMMARY,
        GEOMETRY[LABEL],
        SPLINE_ARCHIVE[LABEL],
        Path("src/bhps/validated_global_bvp.py"),
        Path("src/bhps/global_bvp_collocation.py"),
        Path("src/bhps/global_bvp_newton.py"),
        Path(__file__),
    )
    return {str(path): sha256_file(path) for path in paths}


def recovery_index():
    g9 = json.loads(G9_AFFINE_SUMMARY.read_text())
    if g9.get("certificate_claimed") is not False or not g9.get("all_finite"):
        raise RuntimeError("G9 affine checkpoint is not a valid finite precursor")
    return RecoveryIndex(
        MANIFEST,
        PROTOCOL,
        provenance_inputs(),
        maximum_stage_seconds=1800.0,
    )


def output_path(ordinal):
    return RECOVERY / f"A794_G7_root{ordinal}_D12_M70_P160_operator.npz"


def point_center_mapping(predictor):
    return {
        key + "_center": predictor[key]
        for key in ("axis_rho", "axis_u", "rho_blocks", "w_blocks")
    }


def ensure_root(index, ordinal, launch_radius):
    stage_id = f"physical/A794_G7/root{ordinal}/D12-M70-P160/point-operator"
    metadata = {
        "classification": "floating_adverse_root_control_not_a_certificate",
        "launch_radius": float(launch_radius),
        "configuration": CONFIGURATION,
        "relative_step": 1e-7,
    }
    index.register(stage_id, "floating-point-root-operator", 900.0, metadata)
    reusable = index.validated_path(stage_id)
    layout = shared_nodal_layout(CONFIGURATION)
    required = {
        "center_vector": (layout["size"],),
        "updated_vector": (layout["size"],),
        "correction": (layout["size"],),
        "residual_before": (layout["size"],),
        "residual_after": (layout["size"],),
        "component_scales": (3,),
        "mesh": (71,),
    }
    if reusable is not None:
        validate_npz(reusable, required, require_finite=True)
        completion = index.data["stages"][stage_id]["completion_metadata"]
        return completion["diagnostics"], True

    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        geometry = load_geometry(GEOMETRY[LABEL])
        metric = load_validated_metric(SPLINE_ARCHIVE[LABEL])
        splines = scipy_splines(geometry)
        predictor = floating_chebyshev_predictor(
            launch_radius, metric, splines, CONFIGURATION,
        )
        center = pack_predictor_center_shared(
            point_center_mapping(predictor), CONFIGURATION,
        )

        def residual_function(vector):
            return floating_global_collocation_residual(
                vector,
                launch_radius,
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
        before_state = unpack_shared_vector(center, CONFIGURATION)
        after_state = unpack_shared_vector(
            step["updated_vector"], CONFIGURATION,
        )
        jacobian = step["jacobian"].tocsr()
        arrays = {
            "center_vector": center,
            "updated_vector": step["updated_vector"],
            "correction": step["correction"],
            "residual_before": step["residual_before"],
            "residual_after": step["residual_after"],
            "component_scales": predictor["component_scales"],
            "mesh": predictor["mesh"],
            "axis_rho": predictor["axis_rho"],
            "axis_u": predictor["axis_u"],
            "rho_blocks": predictor["rho_blocks"],
            "w_blocks": predictor["w_blocks"],
            "jacobian_data": jacobian.data,
            "jacobian_indices": jacobian.indices,
            "jacobian_indptr": jacobian.indptr,
            "jacobian_shape": np.asarray(jacobian.shape, dtype=np.int64),
        }
        output = output_path(ordinal)
        atomic_write_npz(output, **arrays)
        validate_npz(output, required, require_finite=True)
        diagnostics = {
            "ordinal": int(ordinal),
            "launch_radius": float(launch_radius),
            "status": "floating_point_operator_complete_not_a_certificate",
            "certificate_claimed": False,
            "reference_dense_steps": int(predictor["reference_dense_steps"]),
            "newton": step["diagnostics"],
            "residual_before": collocation_residual_summary(
                step["residual_before"], CONFIGURATION,
            ),
            "residual_after": collocation_residual_summary(
                step["residual_after"], CONFIGURATION,
            ),
            "terminal_w_before": float(before_state["w_shared"][-1]),
            "terminal_w_after": float(after_state["w_shared"][-1]),
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
    stage_id = "physical/A794_G7/D12-M70-P160/point-root-control-summary"
    metadata = {
        "root_locations": list(ROOTS),
        "archive_hashes": [record["archive"]["sha256"] for record in records],
    }
    index.register(stage_id, "floating-point-root-control-summary", 120.0, metadata)
    reusable = index.validated_path(stage_id)
    if reusable is not None:
        return json.loads(reusable.read_text()), True
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        passed = all(
            record["newton"]["all_finite"]
            and record["residual_after"]["maximum_absolute"] < 1e-8
            and abs(record["terminal_w_after"]) < 1e-6
            for record in records
        )
        payload = {
            "schema": "test4d-a794-point-operator-control-v1",
            "status": "PASS" if passed else "REVIEW",
            "certificate_claimed": False,
            "scope": "floating point-launch operator control only",
            "records": records,
            "all_two_point_controls_numerically_resolved": passed,
            "interpretation": (
                "This checks operator behavior at the frozen root locations; "
                "it does not isolate either root or validate uniqueness."
            ),
        }
        atomic_write_json(SUMMARY, payload)
        index.mark_complete(
            stage_id, SUMMARY, time.perf_counter() - started,
        )
        return payload, False
    except Exception as error:
        index.mark_failed(stage_id, repr(error))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--both-roots", action="store_true")
    arguments = parser.parse_args()
    if not arguments.both_roots:
        raise SystemExit("select --both-roots")
    index = recovery_index()
    records = []
    reuse = []
    for ordinal, launch_radius in enumerate(ROOTS, start=1):
        record, reused = ensure_root(index, ordinal, launch_radius)
        records.append(record)
        reuse.append(reused)
    summary, summary_reused = ensure_summary(index, records)
    print(json.dumps({
        "root_reused": reuse,
        "summary_reused": summary_reused,
        "summary": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
