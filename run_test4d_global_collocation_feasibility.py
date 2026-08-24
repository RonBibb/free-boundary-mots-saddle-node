#!/usr/bin/env python3
"""Restartable floating global-collocation feasibility checks for Test 4D."""

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
    pack_predictor_center_shared,
    shared_nodal_layout,
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
PREDICTOR_DIAGNOSTICS = Path(
    "results/test4d_validated_global_bvp_feasibility_stages/"
    "G9_b217_D12_M70_P160_floating_diagnostics.json"
)
RECOVERY = Path("results/test4d_global_collocation_feasibility_stages")
MANIFEST = Path("results/test4d_global_collocation_feasibility_recovery.json")
OUTPUT = RECOVERY / "G9_b217_D12_M70_P160_center_residual.npz"
SUMMARY = RECOVERY / "G9_b217_D12_M70_P160_center_residual_summary.json"
CONFIGURATION = CONFIGURATIONS[0]
LABEL = "G9"
BASE_INDEX = 217
LAUNCH_MIDPOINT = 0.5 * (
    1.180 + BASE_INDEX * (1.209 - 1.180) / 256
    + 1.180 + (BASE_INDEX + 1) * (1.209 - 1.180) / 256
)
TEST_MODULE = "tests.test_test4d_global_bvp_collocation"


def provenance_inputs():
    paths = (
        PROTOCOL,
        SEAL,
        QUALIFICATION,
        PREDICTOR,
        PREDICTOR_DIAGNOSTICS,
        GEOMETRY[LABEL],
        Path("src/bhps/global_bvp_collocation.py"),
        Path("src/bhps/validated_global_bvp.py"),
        Path("tests/test_test4d_global_bvp_collocation.py"),
        Path(__file__),
    )
    return {str(path): sha256_file(path) for path in paths}


def recovery_index():
    qualification = json.loads(QUALIFICATION.read_text())
    diagnostics = json.loads(PREDICTOR_DIAGNOSTICS.read_text())
    if qualification.get("status") != "PASS":
        raise RuntimeError("Test-4D qualification is not PASS")
    if diagnostics.get("certificate_claimed") is not False:
        raise RuntimeError("predictor checkpoint has invalid claim label")
    if diagnostics["predictor"]["sha256"] != sha256_file(PREDICTOR):
        raise RuntimeError("predictor hash disagrees with its diagnostics")
    return RecoveryIndex(
        MANIFEST,
        PROTOCOL,
        provenance_inputs(),
        maximum_stage_seconds=1800.0,
    )


def load_predictor():
    with np.load(PREDICTOR) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def ensure_residual(index):
    stage_id = "physical/G9/base217/D12-M70-P160/center-global-residual"
    metadata = {
        "classification": "floating_global_residual_not_a_certificate",
        "configuration": CONFIGURATION,
        "launch_midpoint": LAUNCH_MIDPOINT,
        "shared_endpoint_representation": True,
        "quadrature_order": 32,
    }
    index.register(stage_id, "floating-global-residual", 900.0, metadata)
    reusable = index.validated_path(stage_id)
    layout = shared_nodal_layout(CONFIGURATION)
    required = {"center_vector": (layout["size"],), "residual": (layout["size"],)}
    if reusable is not None:
        validate_npz(reusable, required, require_finite=True)
        with np.load(reusable) as archive:
            return (
                reusable,
                np.asarray(archive["center_vector"]),
                np.asarray(archive["residual"]),
                True,
            )

    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        predictor = load_predictor()
        geometry = load_geometry(GEOMETRY[LABEL])
        center = pack_predictor_center_shared(predictor, CONFIGURATION)
        residual = floating_global_collocation_residual(
            center,
            LAUNCH_MIDPOINT,
            predictor["mesh"],
            CONFIGURATION,
            float(geometry["z"][-1]),
            scipy_splines(geometry),
            quadrature_order=32,
        )
        atomic_write_npz(OUTPUT, center_vector=center, residual=residual)
        validation = validate_npz(OUTPUT, required, require_finite=True)
        index.mark_complete(
            stage_id,
            OUTPUT,
            time.perf_counter() - started,
            {"validation": validation},
        )
        return OUTPUT, center, residual, False
    except Exception as error:
        index.mark_failed(stage_id, repr(error))
        raise


def ensure_summary(index, residual_file, residual):
    stage_id = "physical/G9/base217/D12-M70-P160/center-global-residual-summary"
    tests = subprocess.run(
        [sys.executable, "-m", "unittest", TEST_MODULE],
        capture_output=True,
        text=True,
        timeout=120,
    )
    metadata = {
        "residual_sha256": sha256_file(residual_file),
        "test_module": TEST_MODULE,
    }
    index.register(stage_id, "floating-global-residual-summary", 120.0, metadata)
    reusable = index.validated_path(stage_id)
    if reusable is not None:
        return json.loads(reusable.read_text()), True

    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        payload = {
            "schema": "test4d-floating-global-residual-v1",
            "status": "floating_global_residual_complete_not_a_certificate",
            "certificate_claimed": False,
            "geometry": LABEL,
            "base_index": BASE_INDEX,
            "configuration": CONFIGURATION,
            "launch_midpoint": LAUNCH_MIDPOINT,
            "shared_endpoint_representation": True,
            "residual_archive": {
                "path": str(residual_file),
                "sha256": sha256_file(residual_file),
            },
            "residual": collocation_residual_summary(
                residual, CONFIGURATION,
            ),
            "focused_tests": {
                "module": TEST_MODULE,
                "return_code": tests.returncode,
                "stdout": tests.stdout,
                "stderr": tests.stderr,
            },
            "next_required_step": (
                "form the sparse Jacobian and compute a high-precision "
                "approximate inverse/Newton correction"
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
    parser.add_argument("--center-residual", action="store_true")
    arguments = parser.parse_args()
    if not arguments.center_residual:
        raise SystemExit("select --center-residual")
    index = recovery_index()
    residual_file, _, residual, residual_reused = ensure_residual(index)
    summary, summary_reused = ensure_summary(index, residual_file, residual)
    print(json.dumps({
        "residual_reused": residual_reused,
        "summary_reused": summary_reused,
        "summary": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
