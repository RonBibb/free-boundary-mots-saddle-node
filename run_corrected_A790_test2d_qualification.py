#!/usr/bin/env python3
"""Prospective numerical qualification checkpoint for A790 Test 2D."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.high_order_invariant_interpolation import manufactured_interpolation_controls
from bhps.ragged_normal_arclength_chart import (
    build_ragged_normal_chart,
    inverse_ragged_chart,
    ragged_chart_arrays,
    ragged_chart_validity,
)
from bhps.recovery_indexer import RecoveryIndex, atomic_write_json, atomic_write_npz, sha256_file, validate_npz


PROTOCOL = Path("notes/110_A790_test2D_high_order_ragged_chart_protocol.md")
PROTOCOL_SHA256 = "f11989b23cff2d5b87bf6e730ff91c77b1e095eb2a7a49d1a277a9d3bf2666e5"
MACHINE_PROTOCOL = Path("results/corrected_A790_test2d_high_order_ragged_chart_protocol.json")
MACHINE_PROTOCOL_SHA256 = "d3a1f59c61499d5e8f67fcd29a4592a7c1afd65b74e7bc90d78d0a35121734e4"
ROOT = Path("results/corrected_A790_test2d_high_order_ragged_chart_qualification_v3_recovery")
MANIFEST = ROOT / "index.json"
OUTPUT = Path("results/corrected_A790_test2d_high_order_ragged_chart_qualification_v3.json")

PARENTS = {
    "notes/108_A790_test2C_proper_brane_arclength_convergence_protocol.md": "3ea036749550a11362cdce4bc4dfdf38eb305e69aa671e7c3a150edeaead6cfe",
    "notes/108_A790_test2C_proper_brane_arclength_convergence_result.md": "ab203c58ffdd7d1958d99b807f965a7e10295f9d0d21951906fe477c73f81436",
    "results/corrected_A790_test2c_proper_arclength_convergence.json": "5f1e74688c1184be2e8982a254313d1394317e8c945bc92483d4289e77ae5dd7",
    "results/corrected_A790_test2c_proper_arclength_convergence_recovery/index.json": "eebf285a37a3c173457b7ef150a13587d1fbf087624e24cd09c12ac981e77c75",
    str(MACHINE_PROTOCOL): MACHINE_PROTOCOL_SHA256,
}


def inputs():
    dynamic = (
        Path(__file__),
        Path("src/bhps/high_order_invariant_interpolation.py"),
        Path("src/bhps/ragged_normal_arclength_chart.py"),
        Path("src/bhps/recovery_indexer.py"),
    )
    return {**PARENTS, **{str(path): sha256_file(path) for path in dynamic}}


def stage_json(index, stage_id, filename, kind, producer):
    path = ROOT / filename
    index.register(stage_id, kind, 1800.0, {})
    cached = index.validated_path(stage_id)
    if cached is not None:
        return json.loads(cached.read_text())
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        payload = {"stage_id": stage_id, "protocol_sha256": index.protocol_sha256, **producer()}
        atomic_write_json(path, payload)
        index.mark_complete(stage_id, path, time.perf_counter() - started)
        return payload
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def stage_npz(index, stage_id, filename, kind, producer):
    path = ROOT / filename
    index.register(stage_id, kind, 1800.0, {})
    cached = index.validated_path(stage_id)
    if cached is not None:
        return cached
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        atomic_write_npz(path, **producer())
        validate_npz(path)
        index.mark_complete(stage_id, path, time.perf_counter() - started)
        return path
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def flat_chart_arrays():
    z = np.linspace(0.0, 1.0, 25)
    r = np.linspace(0.0, 2.0, 33)
    metric = np.broadcast_to(np.eye(2), (len(z), len(r), 2, 2)).copy()
    sphere = np.ones((len(z), len(r)))
    chart = build_ragged_normal_chart(
        z, r, metric, sphere, launch_radius_max=1.5,
        ray_count=17, distance_samples=21,
    )
    return ragged_chart_arrays(chart)


def ragged_controls(chart):
    validity = ragged_chart_validity(chart)
    native_z = np.asarray([0.25, 0.40, 0.72])
    native_r = np.asarray([0.31, 0.80, 1.21])
    inverse = inverse_ragged_chart(chart, native_z, native_r)
    distance_error = float(np.max(np.abs(inverse.distance - (1.0 - native_z))))
    arclength_error = float(np.max(np.abs(inverse.arclength - native_r)))
    crossed_r = chart.r.copy()
    crossed_r[:, [7, 8]] = crossed_r[:, [8, 7]]
    crossing_rejected = not ragged_chart_validity(replace(chart, r=crossed_r))["valid"]
    nonpositive_rejected = False
    z = np.linspace(0.0, 1.0, 9)
    r = np.linspace(0.0, 2.0, 11)
    bad_metric = np.broadcast_to(np.eye(2), (len(z), len(r), 2, 2)).copy()
    bad_metric[..., 0, 0] = -1.0
    try:
        build_ragged_normal_chart(z, r, bad_metric, np.ones((len(z), len(r))))
    except ValueError:
        nonpositive_rejected = True
    passed = bool(
        validity["valid"] and np.all(inverse.root_count == 1)
        and distance_error < 1e-11 and arclength_error < 1e-11
        and crossing_rejected and nonpositive_rejected
    )
    return {
        "passed": passed, "validity": validity,
        "unique_inverse_roots": inverse.root_count.tolist(),
        "distance_roundtrip_error": distance_error,
        "arclength_roundtrip_error": arclength_error,
        "crossing_rejected": crossing_rejected,
        "nonpositive_metric_rejected": nonpositive_rejected,
    }


def load_chart(path):
    from bhps.ragged_normal_arclength_chart import load_ragged_chart
    return load_ragged_chart(path)


def main():
    if sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("sealed Test-2D protocol hash mismatch")
    if sha256_file(MACHINE_PROTOCOL) != MACHINE_PROTOCOL_SHA256:
        raise RuntimeError("sealed Test-2D machine protocol hash mismatch")
    ROOT.mkdir(parents=True, exist_ok=True)
    index = RecoveryIndex(MANIFEST, PROTOCOL, inputs(), maximum_stage_seconds=1800.0)
    interpolation = stage_json(
        index, "controls/interpolation", "interpolation.json",
        "high-order-manufactured-controls",
        lambda: {"interpolation": manufactured_interpolation_controls()},
    )["interpolation"]
    chart_path = stage_npz(
        index, "controls/ragged_chart", "flat_ragged_chart.npz",
        "ragged-chart-manufactured-archive", flat_chart_arrays,
    )
    ragged = stage_json(
        index, "controls/ragged_verifier", "ragged_verifier.json",
        "independent-ragged-chart-controls",
        lambda: {"ragged": ragged_controls(load_chart(chart_path))},
    )["ragged"]
    passed = bool(interpolation["passed"] and ragged["passed"])
    result = {
        "protocol": str(PROTOCOL), "protocol_sha256": PROTOCOL_SHA256,
        "status": "QUALIFIED" if passed else "REVIEW",
        "classification": (
            "test2d_numerical_implementation_qualified"
            if passed else "invalid_test2d_numerical_audit"
        ),
        "physical_verdict": None,
        "interpolation": interpolation, "ragged_chart": ragged,
        "recovery_manifest": str(MANIFEST),
    }
    atomic_write_json(OUTPUT, result)
    stage_json(
        index, "qualification/result", "result.json", "qualification-result",
        lambda: {"status": result["status"], "classification": result["classification"],
                 "output": str(OUTPUT), "output_sha256": sha256_file(OUTPUT)},
    )
    print(json.dumps({
        "status": result["status"], "classification": result["classification"],
        "physical_verdict": None, "manifest": str(MANIFEST),
    }, indent=2))


if __name__ == "__main__":
    main()
