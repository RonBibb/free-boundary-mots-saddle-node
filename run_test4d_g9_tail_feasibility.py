#!/usr/bin/env python3
"""Floating coefficient-tail and dense off-node screens for Test-4D G9."""

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
    sha256_file,
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
D12_Z2_SUMMARY = Path(
    "results/test4d_g9_depth1_z2_tube_screen_stages/"
    "G9_b217_depth1_D12_M70_P160_z2_tube_summary.json"
)
D16_SUMMARY = Path(
    "results/test4d_g9_d16_floating_feasibility_stages/"
    "G9_b217_depth1_D16_M70_P160_floating_summary.json"
)
RECOVERY = Path("results/test4d_g9_tail_feasibility_stages")
MANIFEST = Path("results/test4d_g9_tail_feasibility_recovery.json")
SUMMARY = RECOVERY / "G9_b217_depth1_tail_offnode_summary.json"
LABEL = "G9"
CHILDREN = (
    ("0", 1.20458203125, 1.204638671875),
    ("1", 1.204638671875, 1.2046953125),
)
CONFIGS = (CONFIGURATIONS[0], CONFIGURATIONS[1])


def provenance_inputs():
    paths = (
        PROTOCOL,
        D12_Z2_SUMMARY,
        D16_SUMMARY,
        GEOMETRY[LABEL],
        SPLINE_ARCHIVE[LABEL],
        Path("src/bhps/validated_global_bvp.py"),
        Path(__file__),
    )
    return {str(path): sha256_file(path) for path in paths}


def recovery_index():
    for path in (D12_Z2_SUMMARY, D16_SUMMARY):
        if json.loads(path.read_text()).get("certificate_claimed") is not False:
            raise RuntimeError(f"precursor has invalid claim label: {path}")
    return RecoveryIndex(
        MANIFEST, PROTOCOL, provenance_inputs(), maximum_stage_seconds=900.0,
    )


def ensure_record(index, configuration, path, lower, upper):
    name = configuration["name"]
    stage_id = f"physical/G9/base217/path{path}/{name}/tail-offnode-floating"
    metadata = {
        "classification": "floating_tail_offnode_not_a_certificate",
        "configuration": name,
        "subdivision_path": path,
        "samples_per_domain": max(33, 2 * configuration["bulk_degree"] + 1),
    }
    index.register(stage_id, "floating-tail-offnode-screen", 900.0, metadata)
    reusable = index.validated_path(stage_id)
    if reusable is not None:
        return json.loads(reusable.read_text()), True
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        geometry = load_geometry(GEOMETRY[LABEL])
        metric = load_validated_metric(SPLINE_ARCHIVE[LABEL])
        splines = scipy_splines(geometry)
        predictor = affine_floating_predictor(
            lower, upper, metric, splines, configuration,
        )
        coefficients = predictor_diagnostics(predictor)
        offnode = floating_offnode_residual_diagnostics(
            predictor,
            float(geometry["z"][-1]),
            splines,
            samples_per_domain=max(33, 2 * configuration["bulk_degree"] + 1),
        )
        payload = {
            "schema": "test4d-g9-tail-offnode-feasibility-record-v1",
            "status": "floating_tail_offnode_complete_not_a_certificate",
            "certificate_claimed": False,
            "configuration": name,
            "subdivision_path": path,
            "launch_interval": [lower, upper],
            "coefficient_diagnostics": coefficients,
            "offnode_diagnostics": offnode,
            "all_finite": bool(
                np.isfinite(max(
                    coefficients[component]["maximum_last_three_weighted_norm"]
                    for component in ("axis_rho", "axis_u", "rho_blocks", "w_blocks")
                ))
            ),
            "omissions": [
                "Bernstein directed conversion",
                "interpolation/aliasing tail proof",
                "spline-composition tail proof",
                "inverse-tail contribution",
            ],
        }
        output = RECOVERY / f"G9_b217_p{path}_{name}_tail_offnode.json"
        atomic_write_json(output, payload)
        index.mark_complete(stage_id, output, time.perf_counter() - started)
        return payload, False
    except Exception as error:
        index.mark_failed(stage_id, repr(error))
        raise


def ensure_summary(index, records):
    stage_id = "physical/G9/base217/depth1/tail-offnode-summary"
    index.register(
        stage_id, "floating-tail-offnode-summary", 120.0,
        {"records": [
            [record["configuration"], record["subdivision_path"]]
            for record in records
        ]},
    )
    reusable = index.validated_path(stage_id)
    if reusable is not None:
        return json.loads(reusable.read_text()), True
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        max_tail_ratio = max(
            record["coefficient_diagnostics"][component]["maximum_tail_ratio"]
            for record in records
            for component in ("axis_rho", "axis_u", "rho_blocks", "w_blocks")
        )
        max_rho_residual = max(
            record["offnode_diagnostics"]["maximum_absolute_ode_residual"]
            ["rho_equation"] for record in records
        )
        max_w_residual = max(
            record["offnode_diagnostics"]["maximum_absolute_ode_residual"]
            ["w_equation"] for record in records
        )
        payload = {
            "schema": "test4d-g9-tail-offnode-feasibility-summary-v1",
            "status": "floating_tail_offnode_complete_not_a_certificate",
            "certificate_claimed": False,
            "records": records,
            "maximum_last_three_coefficient_tail_ratio": max_tail_ratio,
            "maximum_dense_offnode_rho_residual": max_rho_residual,
            "maximum_dense_offnode_w_residual": max_w_residual,
            "interpretation": (
                "Small floating values support tail feasibility, but the "
                "sealed Bernstein and directed tail bounds remain unresolved."
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
    parser.add_argument("--all", action="store_true")
    arguments = parser.parse_args()
    if not arguments.all:
        raise SystemExit("select --all")
    index = recovery_index()
    records = []
    reused = []
    for configuration in CONFIGS:
        for path, lower, upper in CHILDREN:
            record, was_reused = ensure_record(
                index, configuration, path, lower, upper,
            )
            records.append(record)
            reused.append(was_reused)
    summary, summary_reused = ensure_summary(index, records)
    print(json.dumps({
        "records_reused": reused,
        "summary_reused": summary_reused,
        "summary": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
