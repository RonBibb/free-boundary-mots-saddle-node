#!/usr/bin/env python3
"""Restartable sealed Test-4D floating-feasibility checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.recovery_indexer import (
    RecoveryIndex,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
    validate_npz,
)
from bhps.validated_global_bvp import (
    CONFIGURATIONS,
    affine_floating_predictor,
    floating_offnode_residual_diagnostics,
    predictor_diagnostics,
)
from run_test4b_validated_interval_no_horizon_certificate import (
    GEOMETRY,
    SPLINE_ARCHIVE,
    load_geometry,
    load_validated_metric,
    scipy_splines,
)


PROTOCOL = Path("notes/104_test4d_validated_global_bvp_certificate_protocol.md")
SEAL = Path("results/test4d_validated_global_bvp_protocol_seal.json")
QUALIFICATION = Path(
    "results/test4d_validated_global_bvp_stages/"
    "restart_and_control_qualification.json"
)
RECOVERY = Path("results/test4d_validated_global_bvp_feasibility_stages")
MANIFEST = Path("results/test4d_validated_global_bvp_feasibility_recovery.json")
CONFIGURATION = CONFIGURATIONS[0]
LABEL = "G9"
BASE_INDEX = 217
LAUNCH_LOWER = 1.180 + BASE_INDEX * (1.209 - 1.180) / 256
LAUNCH_UPPER = 1.180 + (BASE_INDEX + 1) * (1.209 - 1.180) / 256


def provenance_inputs():
    paths = (
        PROTOCOL,
        SEAL,
        QUALIFICATION,
        GEOMETRY[LABEL],
        SPLINE_ARCHIVE[LABEL],
        Path("src/bhps/validated_global_bvp.py"),
        Path("src/bhps/correlated_validated_shooting.py"),
        Path("src/bhps/validated_capped_surface_shooting.py"),
        Path("src/bhps/recovery_indexer.py"),
        Path(__file__),
    )
    return {str(path): sha256_file(path) for path in paths}


def recovery_index():
    qualification = json.loads(QUALIFICATION.read_text())
    if qualification.get("status") != "PASS":
        raise RuntimeError("Test-4D qualification must PASS before physics")
    if qualification.get("physical_outcomes_evaluated") is not False:
        raise RuntimeError("qualification has invalid physical-outcome flag")
    return RecoveryIndex(
        MANIFEST,
        PROTOCOL,
        provenance_inputs(),
        maximum_stage_seconds=1800.0,
    )


def predictor_path():
    return RECOVERY / "G9_b217_D12_M70_P160_affine_predictor.npz"


def diagnostics_path():
    return RECOVERY / "G9_b217_D12_M70_P160_floating_diagnostics.json"


def expected_predictor_shapes():
    domains = 70
    axis_size = CONFIGURATION["axis_degree"] + 1
    block_size = CONFIGURATION["bulk_degree"] + 1
    shapes = {
        "launch_interval": (2,),
        "launch_midpoint": (),
        "launch_halfwidth": (),
        "mesh": (domains + 1,),
        "component_scales": (3,),
        "reference_dense_steps": (3,),
    }
    for name in ("axis_rho", "axis_u"):
        for suffix in ("center", "parameter", "affine_endpoint_defect"):
            shapes[f"{name}_{suffix}"] = (axis_size,)
    for name in ("rho_blocks", "w_blocks"):
        for suffix in ("center", "parameter", "affine_endpoint_defect"):
            shapes[f"{name}_{suffix}"] = (domains, block_size)
    return shapes


def load_predictor(path):
    validate_npz(path, expected_predictor_shapes(), require_finite=True)
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def ensure_predictor(index):
    stage_id = "physical/G9/base217/D12-M70-P160/predictor"
    metadata = {
        "classification": "floating_predictor_only_not_a_certificate",
        "geometry": LABEL,
        "base_index": BASE_INDEX,
        "launch_interval": [LAUNCH_LOWER, LAUNCH_UPPER],
        "configuration": CONFIGURATION,
    }
    index.register(stage_id, "affine-floating-predictor", 900.0, metadata)
    reusable = index.validated_path(stage_id)
    if reusable is not None:
        return reusable, load_predictor(reusable), True

    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        geometry = load_geometry(GEOMETRY[LABEL])
        metric = load_validated_metric(SPLINE_ARCHIVE[LABEL])
        predictor = affine_floating_predictor(
            LAUNCH_LOWER,
            LAUNCH_UPPER,
            metric,
            scipy_splines(geometry),
            CONFIGURATION,
        )
        output = predictor_path()
        atomic_write_npz(output, **predictor)
        validation = validate_npz(
            output, expected_predictor_shapes(), require_finite=True,
        )
        index.mark_complete(
            stage_id,
            output,
            time.perf_counter() - started,
            {"validation": validation},
        )
        return output, load_predictor(output), False
    except Exception as error:
        index.mark_failed(stage_id, repr(error))
        raise


def ensure_diagnostics(index, predictor_file, predictor):
    stage_id = "physical/G9/base217/D12-M70-P160/floating-diagnostics"
    metadata = {
        "classification": "floating_diagnostic_only_not_a_certificate",
        "predictor_path": str(predictor_file),
        "predictor_sha256": sha256_file(predictor_file),
        "offnode_samples_per_domain": 33,
    }
    index.register(stage_id, "floating-offnode-diagnostic", 300.0, metadata)
    reusable = index.validated_path(stage_id)
    if reusable is not None:
        return json.loads(reusable.read_text()), True

    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        geometry = load_geometry(GEOMETRY[LABEL])
        floating = floating_offnode_residual_diagnostics(
            predictor,
            float(geometry["z"][-1]),
            scipy_splines(geometry),
            samples_per_domain=33,
        )
        ranges = floating["floating_affine_sampled_ranges"]
        spline_domain = {
            "z": [float(geometry["z"][0]), float(geometry["z"][-1])],
            "r": [float(geometry["r"][0]), float(geometry["r"][-1])],
            "rho": [0.10, 1.67],
        }
        guards = {
            key: (
                ranges[key][0] >= spline_domain[key][0]
                and ranges[key][1] <= spline_domain[key][1]
            )
            for key in ("z", "r", "rho")
        }
        payload = {
            "schema": "test4d-floating-feasibility-checkpoint-v1",
            "status": "floating_diagnostic_complete_not_a_certificate",
            "physical_outcome_evaluated": True,
            "certificate_claimed": False,
            "geometry": LABEL,
            "base_index": BASE_INDEX,
            "launch_interval": [LAUNCH_LOWER, LAUNCH_UPPER],
            "configuration": CONFIGURATION,
            "predictor": {
                "path": str(predictor_file),
                "sha256": sha256_file(predictor_file),
                "reference_dense_steps": predictor[
                    "reference_dense_steps"
                ].astype(int).tolist(),
                "component_scales": predictor[
                    "component_scales"
                ].tolist(),
            },
            "coefficient_diagnostics": predictor_diagnostics(predictor),
            "offnode_diagnostics": floating,
            "sampled_domain": spline_domain,
            "floating_affine_sampled_domain_guards": guards,
            "all_sampled_domain_guards_pass": all(guards.values()),
            "next_required_step": (
                "construct the complete global collocation residual and "
                "directed interval Y/Z bounds"
            ),
        }
        output = diagnostics_path()
        atomic_write_json(output, _jsonable(payload))
        checked = json.loads(output.read_text())
        if checked.get("certificate_claimed") is not False:
            raise RuntimeError("floating checkpoint mislabeled as certificate")
        index.mark_complete(
            stage_id, output, time.perf_counter() - started,
            {"sampled_domain_guards": guards},
        )
        return checked, False
    except Exception as error:
        index.mark_failed(stage_id, repr(error))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--g9-base217-predictor", action="store_true")
    arguments = parser.parse_args()
    if not arguments.g9_base217_predictor:
        raise SystemExit("select --g9-base217-predictor")
    index = recovery_index()
    predictor_file, predictor, predictor_reused = ensure_predictor(index)
    diagnostics, diagnostics_reused = ensure_diagnostics(
        index, predictor_file, predictor,
    )
    print(json.dumps({
        "predictor_reused": predictor_reused,
        "diagnostics_reused": diagnostics_reused,
        "diagnostics": diagnostics,
    }, indent=2))


if __name__ == "__main__":
    main()
