#!/usr/bin/env python3
"""Sampled regular-axis feasibility screen for the sealed Test-4D children.

This stage deliberately remains a floating diagnostic.  It measures whether
the regular-axis rows are likely to consume the contraction margin left by the
directed correlated bulk screen.  The certificate still requires the sealed
parity-coefficient interval implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, diags
from scipy.sparse.linalg import splu

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.global_bvp_collocation import _axis_integral_residual
from bhps.recovery_indexer import (
    RecoveryIndex,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
    validate_npz,
)
from run_test4b_validated_interval_no_horizon_certificate import (
    GEOMETRY,
    load_geometry,
    scipy_splines,
)
from run_test4d_bulk_interval_jacobian_screen import component_scale_vector


PROTOCOL = Path("notes/104_test4d_validated_global_bvp_certificate_protocol.md")
FLOATING_SUMMARY = Path(
    "results/test4d_g9_depth1_floating_screen_stages/"
    "G9_b217_depth1_D12_M70_P160_floating_summary.json"
)
CORRELATED_SUMMARY = Path(
    "results/test4d_g9_depth1_correlated_screen_stages/"
    "G9_b217_depth1_D12_M70_P160_correlated_summary.json"
)
FLOATING_ARCHIVES = {
    path: Path(
        "results/test4d_g9_depth1_floating_screen_stages/"
        f"G9_b217_p{path}_D12_M70_P160_floating_operator.npz"
    )
    for path in ("0", "1")
}
CORRELATED_ARCHIVES = {
    path: Path(
        "results/test4d_g9_depth1_correlated_screen_stages/"
        f"G9_b217_p{path}_D12_M70_P160_correlated_xi16.npz"
    )
    for path in ("0", "1")
}
RECOVERY = Path("results/test4d_g9_depth1_axis_screen_stages")
MANIFEST = Path("results/test4d_g9_depth1_axis_screen_recovery.json")
SUMMARY = RECOVERY / "G9_b217_depth1_D12_M70_P160_axis_summary.json"
CHILDREN = {
    "0": (1.20458203125, 1.204638671875),
    "1": (1.204638671875, 1.2046953125),
}
AXIS_COUNT = 17
XI_SAMPLES = 65
RELATIVE_STEP = 1e-7


def provenance_inputs():
    paths = (
        PROTOCOL,
        FLOATING_SUMMARY,
        CORRELATED_SUMMARY,
        *FLOATING_ARCHIVES.values(),
        *CORRELATED_ARCHIVES.values(),
        GEOMETRY["G9"],
        Path("src/bhps/global_bvp_collocation.py"),
        Path("run_test4d_bulk_interval_jacobian_screen.py"),
        Path(__file__),
    )
    return {str(path): sha256_file(path) for path in paths}


def recovery_index():
    for path in (FLOATING_SUMMARY, CORRELATED_SUMMARY):
        if json.loads(path.read_text()).get("certificate_claimed") is not False:
            raise RuntimeError(f"precursor has invalid claim label: {path}")
    return RecoveryIndex(
        MANIFEST, PROTOCOL, provenance_inputs(), maximum_stage_seconds=900.0,
    )


def axis_jacobian(axis_vector, launch, z_brane, splines):
    axis_vector = np.asarray(axis_vector, dtype=float)

    def function(vector):
        rho, u = _axis_integral_residual(
            vector[:AXIS_COUNT],
            vector[AXIS_COUNT:],
            launch,
            z_brane,
            splines,
            quadrature_order=32,
        )
        return np.concatenate((rho, u))

    steps = RELATIVE_STEP * np.maximum(1.0, np.abs(axis_vector))
    jacobian = np.empty((2 * AXIS_COUNT, 2 * AXIS_COUNT))
    for column, step in enumerate(steps):
        delta = np.zeros_like(axis_vector)
        delta[column] = step
        jacobian[:, column] = (
            function(axis_vector + delta) - function(axis_vector - delta)
        ) / (2.0 * step)
    return jacobian


def ensure_child(index, path):
    stage_id = f"physical/G9/base217/path{path}/D12-M70-P160/axis-sampled"
    metadata = {
        "classification": "sampled_axis_feasibility_not_a_certificate",
        "xi_sample_count": XI_SAMPLES,
        "relative_step": RELATIVE_STEP,
        "parity_directed_interval": False,
    }
    index.register(stage_id, "sampled-axis-feasibility", 900.0, metadata)
    reusable = index.validated_path(stage_id)
    if reusable is not None:
        validate_npz(reusable, require_finite=True)
        return index.data["stages"][stage_id]["completion_metadata"]["diagnostics"], True
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        lower, upper = CHILDREN[path]
        midpoint = 0.5 * (lower + upper)
        halfwidth = 0.5 * (upper - lower)
        geometry = load_geometry(GEOMETRY["G9"])
        splines = scipy_splines(geometry)
        z_brane = float(geometry["z"][-1])
        with np.load(FLOATING_ARCHIVES[path]) as archive:
            center = np.asarray(archive["center_vector"])
            parameter = np.asarray(archive["parameter_vector"])
            scales = component_scale_vector(archive["component_scales"])
            shape = tuple(np.asarray(archive["jacobian_shape"]).astype(int))
            center_jacobian = csr_matrix((
                np.asarray(archive["center_jacobian_data"]),
                np.asarray(archive["center_jacobian_indices"]),
                np.asarray(archive["center_jacobian_indptr"]),
            ), shape=shape)
        with np.load(CORRELATED_ARCHIVES[path]) as archive:
            bulk_difference = csr_matrix((
                np.asarray(archive["difference_magnitude_data"]),
                np.asarray(archive["difference_magnitude_indices"]),
                np.asarray(archive["difference_magnitude_indptr"]),
            ), shape=tuple(np.asarray(archive["shape"]).astype(int)))

        xis = np.linspace(-1.0, 1.0, XI_SAMPLES)
        axis_reference = center_jacobian[:2 * AXIS_COUNT, :2 * AXIS_COUNT].toarray()
        envelope = np.zeros_like(axis_reference)
        sample_z1 = []
        factor = splu(csc_matrix(center_jacobian))
        inverse = factor.solve(np.eye(shape[0]))
        inverse_scaled = np.abs(inverse / scales[:, None])
        for xi in xis:
            state = center + xi * parameter
            launch = midpoint + xi * halfwidth
            jacobian = axis_jacobian(
                state[:2 * AXIS_COUNT], launch, z_brane, splines,
            )
            difference = np.abs(jacobian - axis_reference)
            envelope = np.maximum(envelope, difference)
            embedded = np.zeros(shape)
            embedded[:2 * AXIS_COUNT, :2 * AXIS_COUNT] = difference
            image = inverse_scaled @ (embedded @ np.diag(scales))
            sample_z1.append(float(np.max(np.sum(image, axis=1))))

        combined = bulk_difference.tolil()
        combined[:2 * AXIS_COUNT, :2 * AXIS_COUNT] = (
            combined[:2 * AXIS_COUNT, :2 * AXIS_COUNT]
            + csr_matrix(envelope)
        )
        combined = combined.tocsr()
        row_sums = np.asarray(
            inverse_scaled @ (combined @ diags(scales))
        ).sum(axis=1)
        row_sums = np.asarray(row_sums).reshape(-1)
        output = RECOVERY / f"G9_b217_p{path}_D12_M70_P160_axis_sampled.npz"
        atomic_write_npz(
            output,
            xi_samples=xis,
            axis_difference_envelope=envelope,
            sampled_axis_z1=np.asarray(sample_z1),
            combined_bulk_axis_row_sums=row_sums,
        )
        validate_npz(output, require_finite=True)
        diagnostics = {
            "subdivision_path": path,
            "status": "sampled_axis_feasibility_complete_not_a_certificate",
            "certificate_claimed": False,
            "xi_sample_count": XI_SAMPLES,
            "maximum_sampled_axis_Z1_like_contribution": max(sample_z1),
            "maximum_combined_bulk_axis_Z1_like_proxy": float(np.max(row_sums)),
            "maximum_axis_jacobian_entry_deviation": float(np.max(envelope)),
            "all_finite": bool(
                np.all(np.isfinite(envelope)) and np.all(np.isfinite(row_sums))
            ),
            "omission": (
                "sampled floating Gauss axis operator; sealed coefficientwise "
                "parity interval enclosure remains required"
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
    stage_id = "physical/G9/base217/depth1/D12-M70-P160/axis-summary"
    index.register(
        stage_id, "sampled-axis-summary", 120.0,
        {"archive_hashes": [record["archive"]["sha256"] for record in records]},
    )
    reusable = index.validated_path(stage_id)
    if reusable is not None:
        return json.loads(reusable.read_text()), True
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        payload = {
            "schema": "test4d-g9-depth1-axis-sampled-screen-v1",
            "status": "sampled_axis_feasibility_complete_not_a_certificate",
            "certificate_claimed": False,
            "records": records,
            "maximum_sampled_axis_Z1_like_contribution": max(
                record["maximum_sampled_axis_Z1_like_contribution"]
                for record in records
            ),
            "maximum_combined_bulk_axis_Z1_like_proxy": max(
                record["maximum_combined_bulk_axis_Z1_like_proxy"]
                for record in records
            ),
            "interpretation": (
                "The regular-axis rows appear too small to consume the bulk "
                "contraction margin, but this sampled result is not directed."
            ),
            "remaining_obligations": [
                "coefficientwise parity-directed axis enclosure",
                "directed high-precision inverse products",
                "Bernstein off-node and coefficient tails",
                "candidate-radius Z2 variation",
            ],
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
    for path in ("0", "1"):
        record, reused = ensure_child(index, path)
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
