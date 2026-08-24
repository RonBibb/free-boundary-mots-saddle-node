#!/usr/bin/env python3
"""Restartable sparse floating Newton checkpoint for sealed Test 4D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.global_bvp_collocation import (
    collocation_residual_summary,
    floating_global_collocation_residual,
    global_collocation_sparsity,
    shared_nodal_layout,
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
SEAL = Path("results/test4d_validated_global_bvp_protocol_seal.json")
QUALIFICATION = Path(
    "results/test4d_validated_global_bvp_stages/"
    "restart_and_control_qualification.json"
)
PREDICTOR = Path(
    "results/test4d_validated_global_bvp_feasibility_stages/"
    "G9_b217_D12_M70_P160_affine_predictor.npz"
)
RESIDUAL = Path(
    "results/test4d_global_collocation_feasibility_stages/"
    "G9_b217_D12_M70_P160_center_residual.npz"
)
RESIDUAL_SUMMARY = Path(
    "results/test4d_global_collocation_feasibility_stages/"
    "G9_b217_D12_M70_P160_center_residual_summary.json"
)
RECOVERY = Path("results/test4d_global_newton_feasibility_stages")
MANIFEST = Path("results/test4d_global_newton_feasibility_recovery.json")
OUTPUT = RECOVERY / "G9_b217_D12_M70_P160_newton_step.npz"
SUMMARY = RECOVERY / "G9_b217_D12_M70_P160_newton_step_summary.json"
CONFIGURATION = CONFIGURATIONS[0]
LABEL = "G9"
BASE_INDEX = 217
LAUNCH_MIDPOINT = 1.204638671875
TEST_MODULES = (
    "tests.test_test4d_global_bvp_collocation",
    "tests.test_test4d_global_bvp_newton",
)


def provenance_inputs():
    paths = (
        PROTOCOL,
        SEAL,
        QUALIFICATION,
        PREDICTOR,
        RESIDUAL,
        RESIDUAL_SUMMARY,
        GEOMETRY[LABEL],
        Path("src/bhps/global_bvp_collocation.py"),
        Path("src/bhps/global_bvp_newton.py"),
        Path("tests/test_test4d_global_bvp_collocation.py"),
        Path("tests/test_test4d_global_bvp_newton.py"),
        Path(__file__),
    )
    return {str(path): sha256_file(path) for path in paths}


def recovery_index():
    if json.loads(QUALIFICATION.read_text()).get("status") != "PASS":
        raise RuntimeError("Test-4D qualification is not PASS")
    residual_summary = json.loads(RESIDUAL_SUMMARY.read_text())
    if residual_summary.get("certificate_claimed") is not False:
        raise RuntimeError("residual checkpoint has invalid claim label")
    if residual_summary["residual_archive"]["sha256"] != sha256_file(RESIDUAL):
        raise RuntimeError("residual hash disagrees with summary")
    return RecoveryIndex(
        MANIFEST,
        PROTOCOL,
        provenance_inputs(),
        maximum_stage_seconds=1800.0,
    )


def load_archives():
    with np.load(PREDICTOR) as archive:
        predictor = {key: np.asarray(archive[key]) for key in archive.files}
    with np.load(RESIDUAL) as archive:
        center = np.asarray(archive["center_vector"])
        archived_residual = np.asarray(archive["residual"])
    return predictor, center, archived_residual


def ensure_newton(index):
    stage_id = "physical/G9/base217/D12-M70-P160/floating-newton-step"
    metadata = {
        "classification": "floating_newton_only_not_a_certificate",
        "configuration": CONFIGURATION,
        "launch_midpoint": LAUNCH_MIDPOINT,
        "finite_difference": "colored-central",
        "relative_step": 6e-6,
    }
    index.register(stage_id, "floating-sparse-newton", 900.0, metadata)
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
        with np.load(reusable) as archive:
            return reusable, {
                key: np.asarray(archive[key]) for key in archive.files
            }, True

    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        predictor, center, archived_residual = load_archives()
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

        recomputed = residual_function(center)
        if not np.array_equal(recomputed, archived_residual):
            if not np.allclose(recomputed, archived_residual, rtol=0.0, atol=1e-15):
                raise RuntimeError("archived center residual is not reproducible")
        step = sparse_newton_step(
            residual_function,
            center,
            global_collocation_sparsity(CONFIGURATION),
            CONFIGURATION,
            predictor["component_scales"],
            relative_step=6e-6,
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
        validation = validate_npz(OUTPUT, required, require_finite=True)
        index.mark_complete(
            stage_id,
            OUTPUT,
            time.perf_counter() - started,
            {
                "validation": validation,
                "newton_diagnostics": step["diagnostics"],
            },
        )
        return OUTPUT, arrays, False
    except Exception as error:
        index.mark_failed(stage_id, repr(error))
        raise


def ensure_summary(index, newton_file, arrays):
    stage_id = "physical/G9/base217/D12-M70-P160/floating-newton-summary"
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", *TEST_MODULES],
        capture_output=True,
        text=True,
        timeout=120,
    )
    metadata = {
        "newton_sha256": sha256_file(newton_file),
        "test_modules": list(TEST_MODULES),
    }
    index.register(stage_id, "floating-sparse-newton-summary", 120.0, metadata)
    reusable = index.validated_path(stage_id)
    if reusable is not None:
        return json.loads(reusable.read_text()), True

    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        before = collocation_residual_summary(
            arrays["residual_before"], CONFIGURATION,
        )
        after = collocation_residual_summary(
            arrays["residual_after"], CONFIGURATION,
        )
        payload = {
            "schema": "test4d-floating-global-newton-v1",
            "status": "floating_newton_complete_not_a_certificate",
            "certificate_claimed": False,
            "geometry": LABEL,
            "base_index": BASE_INDEX,
            "configuration": CONFIGURATION,
            "launch_midpoint": LAUNCH_MIDPOINT,
            "newton_archive": {
                "path": str(newton_file),
                "sha256": sha256_file(newton_file),
            },
            "residual_before": before,
            "residual_after": after,
            "maximum_correction": float(np.max(np.abs(arrays["correction"]))),
            "jacobian": {
                "shape": arrays["jacobian_shape"].astype(int).tolist(),
                "stored_nonzero_count": int(len(arrays["jacobian_data"])),
            },
            "focused_tests": {
                "modules": list(TEST_MODULES),
                "return_code": tests.returncode,
                "stdout": tests.stdout,
                "stderr": tests.stderr,
            },
            "next_required_step": (
                "repeat Newton to convergence, then compute directed "
                "finite/infinite-dimensional Y, Z0, Z1, and Z2 bounds"
            ),
        }
        if tests.returncode != 0:
            payload["status"] = "focused_tests_failed"
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
    parser.add_argument("--one-step", action="store_true")
    arguments = parser.parse_args()
    if not arguments.one_step:
        raise SystemExit("select --one-step")
    index = recovery_index()
    newton_file, arrays, newton_reused = ensure_newton(index)
    summary, summary_reused = ensure_summary(index, newton_file, arrays)
    print(json.dumps({
        "newton_reused": newton_reused,
        "summary_reused": summary_reused,
        "summary": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
