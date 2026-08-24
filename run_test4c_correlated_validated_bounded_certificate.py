#!/usr/bin/env python3
"""Run restartable work units for the sealed Test-4C certificate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
import time

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.correlated_validated_shooting import (
    ArchivedDop853Reference,
    advance_correlated_propagation,
    initialize_correlated_propagation,
)
from bhps.recovery_indexer import (
    RecoveryIndex,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
)
from bhps.validated_capped_surface_shooting import VInterval
from run_test4b_validated_interval_no_horizon_certificate import (
    GEOMETRY,
    NOTE91,
    SPLINE_ARCHIVE,
    load_geometry,
    load_validated_metric,
    scipy_splines,
)


PROTOCOL = Path("notes/102_test4c_correlated_validated_bounded_certificate_protocol.md")
TEST4B_PROTOCOL = Path("notes/101_test4b_validated_interval_no_horizon_certificate_protocol.md")
TEST4B_ERRATA = (
    Path("notes/101a_test4b_regular_axis_factor_erratum.md"),
    Path("notes/101b_test4b_axis_cone_divergence_form_erratum.md"),
    Path("notes/101c_test4b_validated_interval_no_horizon_result.md"),
)
TEST4C_IMPLEMENTATION_NOTE = Path(
    "notes/102a_test4c_step_growth_implementation_correction.md"
)
TEST4C_RECOVERY_NOTE = Path(
    "notes/102b_test4c_checkpoint_duration_correction.md"
)
TEST4C_AXIS_NOTE = Path(
    "notes/102c_test4c_correlated_axis_cone_refinement.md"
)
TEST4C_RATIO_NOTE = Path(
    "notes/102d_test4c_exact_axis_weighted_ratio_refinement.md"
)
TEST4C_DIVERGENCE_NOTE = Path(
    "notes/102e_test4c_divergence_coordinate_refinement.md"
)
TEST4C_TAIL_NOTE = Path(
    "notes/102f_test4c_matrix_tail_growth_guard.md"
)
TEST4B_RESULT = Path("results/test4b_validated_interval_no_horizon_certificate.json")
RECOVERY = Path("results/test4c_correlated_validated_bounded_stages_v6")
MANIFEST = Path("results/test4c_correlated_validated_bounded_recovery_v6.json")
QUALIFICATION = RECOVERY / "restart_qualification.json"

LABEL_POLICY = {
    "G9": {"lower": 1.180, "upper": 1.209, "base_cells": 256},
    "G10": {"lower": 1.180, "upper": 1.209, "base_cells": 256},
    "A794_G7": {"lower": 1.2038, "upper": 1.2087, "base_cells": 256},
}
NUMERICAL_POLICY = {
    "theta_axis": 1e-3,
    "coordinate_system": "rho_and_sin_squared_theta_times_rho_prime",
    "axis_subdivisions": 128,
    "axis_launch_subdivisions": 8,
    "reference_method": "DOP853",
    "reference_rtol": 2e-12,
    "reference_atol": 2e-14,
    "reference_maximum_step": 2.5e-4,
    "initial_validation_step": 1e-3,
    "minimum_validation_step": 1e-6,
    "defect_subdivision_schedule": [32, 64, 128, 256],
    "matrix_exponential_order": 28,
    "matrix_tail_limit": 1e-18,
    "interval_precision_bits": 96,
    "rho_bounds": [0.10, 1.67],
    "maximum_launch_refinement": 8,
    "accepted_steps_per_chunk": 4,
}


def provenance_inputs():
    paths = (
        PROTOCOL, TEST4C_IMPLEMENTATION_NOTE, TEST4C_RECOVERY_NOTE,
        TEST4C_AXIS_NOTE, TEST4C_RATIO_NOTE, TEST4C_DIVERGENCE_NOTE,
        TEST4C_TAIL_NOTE,
        TEST4B_PROTOCOL, *TEST4B_ERRATA, NOTE91, TEST4B_RESULT,
        *GEOMETRY.values(), *SPLINE_ARCHIVE.values(),
        Path("src/bhps/correlated_validated_shooting.py"),
        Path("src/bhps/validated_capped_surface_shooting.py"),
        Path("src/bhps/recovery_indexer.py"), Path(__file__),
    )
    return {str(path): sha256_file(path) for path in paths}


def recovery_index():
    return RecoveryIndex(
        MANIFEST, PROTOCOL, provenance_inputs(), maximum_stage_seconds=900.0,
    )


def launch_cell(label, base_index, subdivision_path=""):
    policy = LABEL_POLICY[label]
    base_index = int(base_index)
    if not 0 <= base_index < policy["base_cells"]:
        raise ValueError("base index outside the sealed 256-cell cover")
    if len(subdivision_path) > NUMERICAL_POLICY["maximum_launch_refinement"]:
        raise ValueError("subdivision path exceeds the sealed depth")
    if set(subdivision_path) - {"0", "1"}:
        raise ValueError("subdivision path may contain only 0 and 1")
    spacing = (policy["upper"] - policy["lower"]) / policy["base_cells"]
    lower = policy["lower"] + base_index * spacing
    upper = policy["lower"] + (base_index + 1) * spacing
    for branch in subdivision_path:
        midpoint = 0.5 * lower + 0.5 * upper
        if branch == "0":
            upper = midpoint
        else:
            lower = midpoint
    return VInterval(lower, upper)


def work_unit_slug(label, base_index, subdivision_path=""):
    path = subdivision_path if subdivision_path else "base"
    return f"{label}_b{int(base_index):03d}_p{path}"


def _reference_path(slug):
    return RECOVERY / f"{slug}_reference.npz"


def _checkpoint_path(slug, accepted_steps):
    return RECOVERY / f"{slug}_step_{int(accepted_steps):07d}.json"


def _validated_latest_state(index, slug):
    prefix = f"propagation/{slug}/"
    candidates = []
    initial_path = index.validated_path(f"initial/{slug}")
    if initial_path is not None:
        candidates.append((0, initial_path))
    for stage_id in index.data["stages"]:
        if stage_id.startswith(prefix):
            path = index.validated_path(stage_id)
            if path is not None:
                accepted = int(stage_id.rsplit("/", 1)[-1])
                candidates.append((accepted, path))
    if not candidates:
        return None
    _, path = max(candidates)
    return json.loads(path.read_text())


def ensure_reference_and_initial_state(index, label, base_index, subdivision_path):
    slug = work_unit_slug(label, base_index, subdivision_path)
    reference_stage = f"reference/{slug}"
    launch = launch_cell(label, base_index, subdivision_path)
    metadata = {
        "label": label,
        "base_index": int(base_index),
        "subdivision_path": subdivision_path,
        "launch_interval": [launch.lower, launch.upper],
                "reference_policy": {
            key: NUMERICAL_POLICY[key] for key in (
                "theta_axis", "reference_method", "reference_rtol",
                "reference_atol", "reference_maximum_step",
            )
        },
    }
    index.register(reference_stage, "dop853-reference", 120.0, metadata)
    reference_file = index.validated_path(reference_stage)
    geometry = load_geometry(GEOMETRY[label])
    scipy_fields = scipy_splines(geometry)
    metric = load_validated_metric(SPLINE_ARCHIVE[label])
    recovered_state = _validated_latest_state(index, slug)
    initial_state = recovered_state
    if reference_file is None:
        index.mark_running(reference_stage)
        started = time.perf_counter()
        try:
            reference, initial_state = initialize_correlated_propagation(
                launch, metric, scipy_fields,
                theta_axis=NUMERICAL_POLICY["theta_axis"],
            )
            payload = reference.archive_payload()
            payload.update({
                "axis_radius": np.asarray([launch.lower, launch.upper]),
                "z_brane": np.asarray(metric.z_brane),
            })
            reference_file = _reference_path(slug)
            atomic_write_npz(reference_file, **payload)
            with np.load(reference_file) as archive:
                ArchivedDop853Reference.from_archive(archive)
            index.mark_complete(
                reference_stage, reference_file, time.perf_counter() - started,
                {"dense_step_count": len(reference.boundaries) - 1},
            )
        except Exception as error:
            index.mark_failed(reference_stage, repr(error))
            raise
    with np.load(reference_file) as archive:
        reference = ArchivedDop853Reference.from_archive(archive)
    if recovered_state is None:
        if initial_state is None:
            # The reference may have survived while its initial state did not;
            # reconstruct the inexpensive exact axis set without rebuilding it.
            from bhps.correlated_validated_shooting import (
                regular_axis_affine_error,
            )
            from bhps.validated_capped_surface_shooting import regular_axis_cone

            axis_state, axis_audit = regular_axis_cone(
                launch, metric, theta_axis=NUMERICAL_POLICY["theta_axis"],
                theta_subdivisions=NUMERICAL_POLICY["axis_subdivisions"],
                launch_subdivisions=NUMERICAL_POLICY["axis_launch_subdivisions"],
            )
            slope_error = regular_axis_affine_error(
                launch, axis_state, axis_audit,
                reference.value(NUMERICAL_POLICY["theta_axis"]),
                NUMERICAL_POLICY["theta_axis"],
            )
            sine = math.sin(NUMERICAL_POLICY["theta_axis"])
            error = slope_error.linear_map(np.diag([1.0, sine**2]))
            initial_state = {
                "schema": "test4c-correlated-propagation-v2",
                "coordinate_system": "divergence",
                "classification": "running",
                "axis_radius": [launch.lower, launch.upper],
                "theta_axis": NUMERICAL_POLICY["theta_axis"],
                "theta": NUMERICAL_POLICY["theta_axis"],
                "step": NUMERICAL_POLICY["initial_validation_step"],
                "error": error.to_dict(),
                "accepted_steps": 0,
                "step_rejections": 0,
                "accepted_since_rejection": 0,
                "axis": {
                    "cone": [axis_audit["cone"].lower, axis_audit["cone"].upper],
                    "invariant_cone": [
                        axis_audit["invariant_cone"].lower,
                        axis_audit["invariant_cone"].upper,
                    ],
                    "image": [axis_audit["image"].lower, axis_audit["image"].upper],
                    "source": [axis_audit["source"].lower, axis_audit["source"].upper],
                    "iterations": axis_audit["iterations"],
                    "theta_subdivisions": axis_audit["theta_subdivisions"],
                    "launch_subdivisions": axis_audit["launch_subdivisions"],
                    "weighted_ratio": [
                        axis_audit["weighted_ratio"].lower,
                        axis_audit["weighted_ratio"].upper,
                    ],
                    "coarse_weighted_ratio": [
                        axis_audit["coarse_weighted_ratio"].lower,
                        axis_audit["coarse_weighted_ratio"].upper,
                    ],
                    "radial_launch_cells": [
                        [value.lower, value.upper]
                        for value in axis_audit["radial_launch_cells"]
                    ],
                    "radial_images": [
                        [value.lower, value.upper]
                        for value in axis_audit["radial_images"]
                    ],
                },
                "audit_summary": {
                    "maximum_defect": 0.0,
                    "maximum_matrix_tail": 0.0,
                    "maximum_tube_radius": float(np.max(error.radius)),
                    "maximum_contraction_product": 0.0,
                    "maximum_subdivisions": 0,
                },
            }
        initial_state.update({
            "work_unit": metadata,
            "reference_path": str(reference_file),
            "reference_sha256": sha256_file(reference_file),
            "numerical_policy": NUMERICAL_POLICY,
        })
        state_stage = f"initial/{slug}"
        state_file = _checkpoint_path(slug, 0)
        index.register(
            state_stage, "correlated-propagation-checkpoint", 30.0,
            {**metadata, "accepted_steps_before": 0},
        )
        index.mark_running(state_stage)
        atomic_write_json(state_file, initial_state)
        index.mark_complete(state_stage, state_file, 0.0)
    return reference, initial_state, metric, scipy_fields


def advance_work_unit(label, base_index, subdivision_path="", chunks=1):
    index = recovery_index()
    reference, state, metric, scipy_fields = ensure_reference_and_initial_state(
        index, label, base_index, subdivision_path,
    )
    slug = work_unit_slug(label, base_index, subdivision_path)
    for _ in range(int(chunks)):
        if state["classification"] != "running":
            break
        before = int(state["accepted_steps"])
        stage_id = f"propagation/{slug}/{before:07d}"
        metadata = {
            "label": label, "base_index": int(base_index),
            "subdivision_path": subdivision_path,
            "accepted_steps_before": before,
            "accepted_step_budget": NUMERICAL_POLICY["accepted_steps_per_chunk"],
        }
        index.register(
            stage_id, "correlated-propagation-chunk", 300.0, metadata,
        )
        completed = index.validated_path(stage_id)
        if completed is not None and before != 0:
            state = json.loads(completed.read_text())
            continue
        index.mark_running(stage_id)
        started = time.perf_counter()
        try:
            state = advance_correlated_propagation(
                state, reference, metric, scipy_fields,
                rho_bounds=tuple(NUMERICAL_POLICY["rho_bounds"]),
                accepted_step_budget=NUMERICAL_POLICY["accepted_steps_per_chunk"],
                initial_step=NUMERICAL_POLICY["initial_validation_step"],
                minimum_step=NUMERICAL_POLICY["minimum_validation_step"],
                defect_subdivision_schedule=tuple(
                    NUMERICAL_POLICY["defect_subdivision_schedule"]
                ),
            )
            after = int(state["accepted_steps"])
            state["reference_sha256"] = sha256_file(_reference_path(slug))
            output = _checkpoint_path(slug, after)
            atomic_write_json(output, state)
            index.mark_complete(
                stage_id, output, time.perf_counter() - started,
                {"accepted_steps_after": after, "theta_after": state["theta"]},
            )
        except Exception as error:
            index.mark_failed(stage_id, repr(error))
            raise
    if state["classification"] != "running":
        leaf_stage = f"leaf/{slug}"
        leaf = RECOVERY / f"{slug}_leaf.json"
        index.register(
            leaf_stage, "correlated-validated-leaf", 30.0,
            {"label": label, "base_index": int(base_index),
             "subdivision_path": subdivision_path},
        )
        if index.validated_path(leaf_stage) is None:
            index.mark_running(leaf_stage)
            atomic_write_json(leaf, state)
            index.mark_complete(leaf_stage, leaf, 0.0)
    return state


def restart_qualification():
    """Exercise interruption, corruption, and policy provenance rejection."""
    with tempfile.TemporaryDirectory(prefix="test4c-restart-") as directory:
        root = Path(directory)
        policy = root / "policy.json"
        atomic_write_json(policy, {
            "rho_bounds": NUMERICAL_POLICY["rho_bounds"],
            "precision": NUMERICAL_POLICY["interval_precision_bits"],
        })
        expected = {str(policy): sha256_file(policy)}
        manifest = root / "manifest.json"
        index = RecoveryIndex(manifest, PROTOCOL, expected, 60.0)
        index.register("leaf/0", "qualification-leaf", 10.0, {"ordinal": 0})
        index.mark_running("leaf/0")
        restarted = RecoveryIndex(manifest, PROTOCOL, expected, 60.0)
        interrupted_reset = restarted.data["stages"]["leaf/0"]["status"] == "pending"

        leaf = root / "leaf.json"
        atomic_write_json(leaf, {"classification": "root_free_positive"})
        restarted.mark_running("leaf/0")
        restarted.mark_complete("leaf/0", leaf, 0.0)
        valid_before_corruption = restarted.validated_path("leaf/0") == leaf
        atomic_write_json(leaf, {"classification": "corrupted"})
        corruption_rejected = restarted.validated_path("leaf/0") is None

        atomic_write_json(policy, {
            "rho_bounds": [0.11, 1.67],
            "precision": NUMERICAL_POLICY["interval_precision_bits"] + 1,
        })
        provenance_rejected = False
        try:
            RecoveryIndex(manifest, PROTOCOL, expected, 60.0)
        except RuntimeError:
            provenance_rejected = True
    payload = {
        "status": "PASS" if all((
            interrupted_reset, valid_before_corruption,
            corruption_rejected, provenance_rejected,
        )) else "FAIL",
        "running_reset_to_pending": interrupted_reset,
        "completed_leaf_valid_before_corruption": valid_before_corruption,
        "corrupted_completed_leaf_rejected": corruption_rejected,
        "bound_and_precision_provenance_change_rejected": provenance_rejected,
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "numerical_policy": NUMERICAL_POLICY,
        "source_sha256": {
            "runner": sha256_file(Path(__file__)),
            "correlated_backend": sha256_file(
                "src/bhps/correlated_validated_shooting.py"
            ),
            "recovery_backend": sha256_file("src/bhps/recovery_indexer.py"),
        },
    }
    RECOVERY.mkdir(parents=True, exist_ok=True)
    atomic_write_json(QUALIFICATION, payload)
    return payload


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualify-restart", action="store_true")
    parser.add_argument("--label", choices=tuple(LABEL_POLICY))
    parser.add_argument("--base-index", type=int)
    parser.add_argument("--subdivision-path", default="")
    parser.add_argument("--chunks", type=int, default=1)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    if arguments.qualify_restart:
        print(json.dumps(restart_qualification(), indent=2))
        return
    if arguments.label is None or arguments.base_index is None:
        raise SystemExit("--label and --base-index are required")
    result = advance_work_unit(
        arguments.label, arguments.base_index,
        arguments.subdivision_path, arguments.chunks,
    )
    print(json.dumps({
        "classification": result["classification"],
        "theta": result["theta"],
        "accepted_steps": result["accepted_steps"],
        "step_rejections": result["step_rejections"],
        "audit_summary": result["audit_summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
