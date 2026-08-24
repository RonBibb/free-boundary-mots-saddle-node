#!/usr/bin/env python3
"""Current-code staged D_X^2 J replay on archived Test-10B G8/R10 states.

This runner rebuilds the deterministic Test-10B G8 R12 parent, restricts it
bitwise to R10, verifies the sealed case fingerprint, and evaluates the
current legacy and experimental owner-last RHS implementations at the
archived accepted endpoints t=0.001 and t=0.002.

It is deliberately not labelled an exact historical RHS replay.  Test 10B
did not archive the endpoint source derivatives or initial outer-reference
acceleration and did not hash all transitive RHS modules.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import run_corrected_fold_live_nonlinear_gauge_source as live  # noqa: E402
from bhps.corrected_A790_R12_builder import build_A790_R12_pair  # noqa: E402
from bhps.corrected_A790_test10b_domain_normalized import (  # noqa: E402
    restrict_geometry,
)
from bhps.junction_preservation_diagnostic import (  # noqa: E402
    _orthonormal_frames,
)
from bhps.junction_second_preservation_diagnostic import (  # noqa: E402
    wall_junction_second_tangent,
)
from bhps.nonlinear_regular_so3_evolution import (  # noqa: E402
    NativeRegularSO3RHS,
    StageRegularGaugeSource,
)
from bhps.recovery_indexer import atomic_write_json  # noqa: E402


OUTPUT = ROOT / "results/corrected_A790_G8_R10_current_code_staged_replay.json"
PROTOCOL = ROOT / "notes/117_A790_G8_R10_current_code_staged_replay_protocol.md"
TEST10B_STATE = ROOT / "results/corrected_A790_test10b_domain_normalized_state.npz"
RECOVERY = ROOT / "results/corrected_A790_test10b_domain_normalized_recovery"
MANIFEST = RECOVERY / "index.json"
DT = 0.000125
EXPECTED_CASE_FINGERPRINT = (
    "00ed754f9b15e8d392038dd62c0d4a9b7d82ab4dfd8606a85e17d946a0530be3"
)
HISTORICAL_TEST10B_RUNNER_SHA256 = (
    "b886bf79d57f98b372d8f756d22016f56192d1816b893536e4f6fd5ac242c203"
)
PRE_REFACTOR_CORE_SHA256_FROM_DIRECT_AUDIT = (
    "b40c2dc89a6e5958d7365876a8d0e95d691e66a2dee100bccc1ef234d381c161"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def case_fingerprint(case):
    digest = hashlib.sha256()
    for value in (
        case["z"],
        case["r"],
        case["initial"],
        case["source0"],
        case["memory0"],
        case["geometry"]["jet_field"].reduced_first,
        case["geometry"]["jet_field"].reduced_second,
    ):
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def endpoint_path(step):
    start = ((int(step) - 1) // 4) * 4
    end = min(start + 4, 16)
    return RECOVERY / f"evolution_G8_R10_steps_{start + 1:03d}_{end:03d}.npz"


def load_endpoint(step):
    path = endpoint_path(step)
    with np.load(path) as archive:
        values = tuple(np.asarray(archive[key]).copy() for key in (
            "end_position", "end_velocity", "end_source", "end_memory",
        ))
        archived_start = int(archive["start_step"])
        archived_end = int(archive["end_step"])
    if archived_end != int(step) or archived_start != int(step) - 4:
        raise RuntimeError("endpoint chunk does not end at the requested step")
    return path, values


def verified_manifest_sha256(path, manifest):
    relative = str(Path(path).relative_to(ROOT))
    matches = [
        record for record in manifest["stages"].values()
        if record.get("output_path") == relative
    ]
    if len(matches) != 1:
        raise RuntimeError(f"recovery manifest has no unique entry for {relative}")
    expected = str(matches[0]["sha256"])
    found = sha256_file(path)
    if found != expected:
        raise RuntimeError(f"recovery chunk hash mismatch for {relative}")
    return {"computed": found, "manifest": expected, "match": True}


def reconstruct_live_gauge(case, time_value, position, velocity, source, memory):
    source_z, source_r = live.regular_source_spatial_derivatives(
        source, case["z"], case["r"],
    )
    target = live.regular_so3_nonlinear_anchored_damped_wave_target(
        position, case["initial"], case["source0"], case["r"],
        live.TARGET_MU_LAPSE, live.TARGET_MU_SHIFT, live.TARGET_POWER,
    )
    advection = live.regular_so3_live_source_shift_advection(
        position, case["r"], source, source_z, source_r,
    )
    source_time, memory_time = live.source_driver_rhs(
        source, memory, target, live.DRIVER_MU, live.DRIVER_ETA, advection,
    )
    source_time, outer_source = live.apply_outer_source_sommerfeld(
        source, source_time, case["source0"], case["source_time0"],
        case["_initial_source_second_time"], position, time_value,
        case["r"], case["rhs"].stencil_width,
    )
    gauge = StageRegularGaugeSource(
        source, source_time, case["z"], case["r"],
    )
    source_second_time = live.live_regular_source_second_time(
        position, velocity, case["initial"], case["source0"], source,
        source_time, memory_time, case["z"], case["r"], live.DRIVER_MU,
        live.TARGET_MU_LAPSE, live.TARGET_MU_SHIFT, live.TARGET_POWER,
    )
    return gauge, source_second_time, outer_source


def make_owner_last_rhs(case):
    rhs = NativeRegularSO3RHS(
        case["z"], case["r"], case["taylor"],
        case["geometry"]["mass_squared"], case["geometry"]["background"],
        case["rhs"].normal_wall_acceleration,
        stencil_width=case["rhs"].stencil_width,
        live_normal_wall_gauge=True,
        live_outer_sommerfeld=True,
        boundary_closure_mode="wall_owner_last_experimental",
    )
    rhs.set_outer_sommerfeld_reference(
        case["rhs"].outer_reference_position,
        case["rhs"].outer_reference_acceleration,
    )
    return rhs


def metric_summary(record):
    metric = np.asarray(record["metric_tensor"])
    frames, frame_defect = _orthonormal_frames(metric)
    tangent = np.asarray(record["DX2J_tensor"])
    tangent_hat = np.einsum("nai,nab,nbj->nij", frames, tangent, frames)
    tangent_norm = np.linalg.norm(tangent_hat, axis=(1, 2))
    wall_row = np.maximum.reduce([
        np.abs(component["DX2_robin_residual"])
        for component in record["components"].values()
    ])
    junction_norm = np.linalg.norm(np.einsum(
        "nai,nab,nbj->nij", frames, record["J_tensor"], frames,
    ), axis=(1, 2))
    first_norm = np.linalg.norm(np.einsum(
        "nai,nab,nbj->nij", frames, record["DXJ_tensor"], frames,
    ), axis=(1, 2))
    return {
        "DX2J_orthonormal_frobenius": {
            "axis": float(tangent_norm[0]),
            "outer": float(tangent_norm[-1]),
            "interior_maximum": float(np.max(tangent_norm[7:-7])),
            "global_maximum": float(np.max(tangent_norm)),
        },
        "DX2_coordinate_robin_row_maximum_absolute": {
            "axis": float(wall_row[0]),
            "outer": float(wall_row[-1]),
            "global_maximum": float(np.max(wall_row)),
        },
        "state_baseline": {
            "J_orthonormal_frobenius_axis": float(junction_norm[0]),
            "J_orthonormal_frobenius_outer": float(junction_norm[-1]),
            "DXJ_orthonormal_frobenius_axis": float(first_norm[0]),
            "DXJ_orthonormal_frobenius_outer": float(first_norm[-1]),
        },
        "frame_defect_maximum": float(np.max(frame_defect)),
        "decomposition_maximum_absolute_defect": float(
            record["decomposition_maximum_absolute_defect"]
        ),
    }


def stage_summary(position, velocity, stage, case):
    result = {"name": str(stage["name"])}
    if "iteration" in stage:
        result["iteration"] = int(stage["iteration"])
    walls = {}
    for wall in ("lower", "upper"):
        record = wall_junction_second_tangent(
            position, velocity, stage["acceleration"], case["z"], case["r"],
            case["geometry"]["background"], wall, case["rhs"].stencil_width,
        )
        separate = record["separate_rows"]
        walls[wall] = {
            "metric": metric_summary(record),
            "Phi_DX2_maximum_absolute": {
                "axis": float(abs(separate["DX2_Phi_robin"][0])),
                "outer": float(abs(separate["DX2_Phi_robin"][-1])),
            },
            "chi_DX2_maximum_absolute": {
                "axis": float(abs(separate["DX2_chi_neumann"][0])),
                "outer": float(abs(separate["DX2_chi_neumann"][-1])),
            },
        }
    result["walls"] = walls
    return result


def select_stages(mode, stages):
    legacy = {
        "initial_axis_fill", "final_compact_wall_endpoint_solve",
        "final_compact_post_wall_axis_fill", "pre_outer", "post_outer",
    }
    owner = {
        "initial_axis_fill", "outer_open_face_before_wall",
        "coupled_Phi_gzz_wall_solve", "final_compact_wall_endpoint_solve",
        "post_wall_owner_reconciliation",
    }
    selected_names = legacy if mode == "legacy" else owner
    return [
        (index, stage) for index, stage in enumerate(stages)
        if stage["name"] in selected_names
    ]


def replay_mode(mode, rhs, time_value, position, velocity, gauge, source_second, case):
    started = time.perf_counter()
    acceleration, diagnostic = rhs.acceleration(
        time_value, position, velocity, gauge, source_second,
        capture_boundary_stages=True,
    )
    elapsed = time.perf_counter() - started
    selected = []
    for index, stage in select_stages(mode, diagnostic["boundary_stages"]):
        record = stage_summary(position, velocity, stage, case)
        record["index"] = int(index)
        selected.append(record)
    returned = stage_summary(position, velocity, {
        "name": "returned_final", "acceleration": acceleration,
    }, case)
    return {
        "_runtime_seconds_descriptive": float(elapsed),
        "stage_count": len(diagnostic["boundary_stages"]),
        "stage_names": [stage["name"] for stage in diagnostic["boundary_stages"]],
        "selected_stages": selected,
        "returned_final": returned,
        "normal_method": diagnostic["normal_wall_gauge"]["method"],
        "normal_residual_maximum": float(
            diagnostic["normal_wall_gauge"]["final_residual"]["maximum"]
        ),
        "outer_residual_maximum": float(
            diagnostic["outer_sommerfeld"][
                "maximum_normalized_acceleration_residual"
            ]
        ),
        "_acceleration": acceleration,
        "_stages": diagnostic["boundary_stages"],
    }


def public_mode(record):
    return {key: value for key, value in record.items() if not key.startswith("_")}


def selected_stage(mode_record, name):
    matches = [
        stage for stage in mode_record["selected_stages"]
        if stage["name"] == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"no unique selected stage named {name}")
    return matches[0]


def coordinate_row_value(stage, wall, endpoint):
    return float(stage["walls"][wall]["metric"][
        "DX2_coordinate_robin_row_maximum_absolute"
    ][endpoint])


def orthonormal_value(stage, wall, endpoint):
    return float(stage["walls"][wall]["metric"][
        "DX2J_orthonormal_frobenius"
    ][endpoint])


def main():
    manifest = json.loads(MANIFEST.read_text())
    started = time.perf_counter()
    _, g8_parent = build_A790_R12_pair()
    build_seconds = time.perf_counter() - started
    geometry = restrict_geometry(
        g8_parent, 10.0, "G8-A790-test10b-R10-current-code-replay",
    )
    with np.load(TEST10B_STATE) as archive:
        archived_initial = np.asarray(archive["G8_R10_initial"])
    initial = np.asarray(geometry["jet_field"].reduced_fields)
    initial_equal = bool(np.array_equal(initial, archived_initial))
    if not initial_equal:
        raise RuntimeError("rebuilt G8/R10 initial state is not the archived state")

    started = time.perf_counter()
    case = live.setup_case(
        geometry, "G8_R10-A790-test10b-current-code-replay",
        live_normal_wall_gauge=True, live_outer_sommerfeld=True,
    )
    setup_seconds = time.perf_counter() - started
    fingerprint = case_fingerprint(case)
    if fingerprint != EXPECTED_CASE_FINGERPRINT:
        raise RuntimeError("rebuilt case fingerprint does not match Test 10B")
    owner_rhs = make_owner_last_rhs(case)

    endpoint_records = {}
    endpoint_hashes = {}
    endpoint_gate_records = {}
    for step in (8, 16):
        time_value = step * DT
        path, state = load_endpoint(step)
        endpoint_hashes[str(path.relative_to(ROOT))] = verified_manifest_sha256(
            path, manifest,
        )
        position, velocity, source, memory = state
        gauge, source_second, outer_source = reconstruct_live_gauge(
            case, time_value, position, velocity, source, memory,
        )
        if not all(np.all(np.isfinite(value)) for value in (
            position, velocity, source, memory, gauge.source,
            gauge.source_time, source_second,
        )):
            raise RuntimeError("nonfinite reconstructed endpoint data")
        legacy = replay_mode(
            "legacy", case["rhs"], time_value, position, velocity, gauge,
            source_second, case,
        )
        owner = replay_mode(
            "owner", owner_rhs, time_value, position, velocity, gauge,
            source_second, case,
        )
        owner_outer_stage = next(
            stage for stage in owner["_stages"]
            if stage["name"] == "outer_open_face_before_wall"
        )
        owner_final = owner["_acceleration"]
        legacy_final = legacy["_acceleration"]
        owner_open_unchanged = bool(np.array_equal(
            owner_outer_stage["acceleration"][1:-1, -1],
            owner_final[1:-1, -1],
        ))
        legacy_owner_open_equal = bool(np.array_equal(
            legacy_final[1:-1, -1], owner_final[1:-1, -1],
        ))
        legacy_wall = selected_stage(
            legacy, "final_compact_wall_endpoint_solve",
        )
        legacy_axis = selected_stage(
            legacy, "final_compact_post_wall_axis_fill",
        )
        legacy_pre_outer = selected_stage(legacy, "pre_outer")
        legacy_post_outer = selected_stage(legacy, "post_outer")
        owner_wall = selected_stage(owner, "final_compact_wall_endpoint_solve")
        owner_final_stage = selected_stage(
            owner, "post_wall_owner_reconciliation",
        )
        wall_gate_records = {}
        for wall in ("lower", "upper"):
            axis_before = coordinate_row_value(legacy_wall, wall, "axis")
            axis_after = coordinate_row_value(legacy_axis, wall, "axis")
            outer_before = coordinate_row_value(
                legacy_pre_outer, wall, "outer",
            )
            outer_after = coordinate_row_value(
                legacy_post_outer, wall, "outer",
            )
            owner_outer = orthonormal_value(owner_final_stage, wall, "outer")
            pre_outer_scale = orthonormal_value(
                legacy_pre_outer, wall, "outer",
            )
            wall_gate_records[wall] = {
                "legacy_post_axis_coordinate_row_amplification_factor": float(
                    axis_after / max(axis_before, np.finfo(float).tiny)
                ),
                "legacy_post_axis_amplification_detected": bool(
                    axis_after > 1e6 * max(axis_before, np.finfo(float).tiny)
                ),
                "legacy_post_outer_coordinate_row_amplification_factor": float(
                    outer_after / max(outer_before, np.finfo(float).tiny)
                ),
                "legacy_post_outer_amplification_detected": bool(
                    outer_after > 1e6 * max(outer_before, np.finfo(float).tiny)
                ),
                "owner_to_legacy_pre_outer_orthonormal_scale_ratio": float(
                    owner_outer / max(pre_outer_scale, np.finfo(float).tiny)
                ),
                "owner_preserves_pre_overwrite_outer_scale_within_factor_2": bool(
                    owner_outer <= 2.0 * max(
                        pre_outer_scale, np.finfo(float).tiny,
                    )
                ),
            }
        owner_post_wall_bitwise_equal = bool(np.array_equal(
            owner_wall["walls"]["lower"]["metric"][
                "DX2J_orthonormal_frobenius"
            ]["global_maximum"],
            owner_final_stage["walls"]["lower"]["metric"][
                "DX2J_orthonormal_frobenius"
            ]["global_maximum"],
        ) and np.array_equal(
            owner["_stages"][-1]["acceleration"], owner_final,
        ))
        all_finite = bool(all(np.all(np.isfinite(value)) for value in (
            position, velocity, source, memory, gauge.source,
            gauge.source_time, source_second, legacy_final, owner_final,
        )) and all(
            np.all(np.isfinite(stage["acceleration"]))
            for mode_record in (legacy, owner)
            for stage in mode_record["_stages"]
        ))
        endpoint_gate_records[f"t{time_value:.3f}"] = {
            "walls": wall_gate_records,
            "owner_post_wall_stage_equals_returned_acceleration_bitwise": (
                owner_post_wall_bitwise_equal
            ),
            "owner_open_outer_face_bitwise_unchanged": owner_open_unchanged,
            "legacy_owner_open_outer_targets_bitwise_equal": (
                legacy_owner_open_equal
            ),
            "all_values_and_stages_finite": all_finite,
        }
        endpoint_records[f"t{time_value:.3f}"] = {
            "step": int(step),
            "time": float(time_value),
            "archive": str(path.relative_to(ROOT)),
            "source_derivative_status": (
                "reconstructed_by_current_sealed_runner_equations; not archived"
            ),
            "outer_source_residual_maximum": float(
                outer_source["maximum_normalized"]
            ),
            "legacy": public_mode(legacy),
            "owner_last": public_mode(owner),
            "ownership_checks": {
                "owner_open_outer_face_bitwise_unchanged_after_wall_solve": (
                    owner_open_unchanged
                ),
                "legacy_and_owner_returned_open_outer_targets_bitwise_equal": (
                    legacy_owner_open_equal
                ),
            },
        }

    input_paths = (
        Path(__file__),
        PROTOCOL,
        TEST10B_STATE,
        MANIFEST,
        ROOT / "results/corrected_family_knot_A8_state.npz",
        ROOT / "run_corrected_fold_live_nonlinear_gauge_source.py",
        ROOT / "src/bhps/corrected_A790_R12_builder.py",
        ROOT / "src/bhps/corrected_A790_test10b_domain_normalized.py",
        ROOT / "src/bhps/gh_source_driver.py",
        ROOT / "src/bhps/nonlinear_regular_so3_evolution.py",
        ROOT / "src/bhps/junction_preservation_diagnostic.py",
        ROOT / "src/bhps/junction_second_preservation_diagnostic.py",
    )
    current_core_hash = sha256_file(
        ROOT / "src/bhps/nonlinear_regular_so3_evolution.py"
    )
    manifest_gate = bool(
        len(endpoint_hashes) == 2
        and all(record["match"] for record in endpoint_hashes.values())
    )
    ownership_gate = bool(all(
        record["owner_open_outer_face_bitwise_unchanged"]
        and record["legacy_owner_open_outer_targets_bitwise_equal"]
        and record["owner_post_wall_stage_equals_returned_acceleration_bitwise"]
        for record in endpoint_gate_records.values()
    ))
    post_axis_gate = bool(all(
        wall["legacy_post_axis_amplification_detected"]
        for record in endpoint_gate_records.values()
        for wall in record["walls"].values()
    ))
    post_outer_gate = bool(all(
        wall["legacy_post_outer_amplification_detected"]
        for record in endpoint_gate_records.values()
        for wall in record["walls"].values()
    ))
    owner_scale_gate = bool(all(
        wall["owner_preserves_pre_overwrite_outer_scale_within_factor_2"]
        for record in endpoint_gate_records.values()
        for wall in record["walls"].values()
    ))
    finite_gate = bool(all(
        record["all_values_and_stages_finite"]
        for record in endpoint_gate_records.values()
    ))
    gates = {
        "both_endpoint_chunks_match_recovery_manifest_sha256": manifest_gate,
        "rebuilt_initial_and_case_fingerprint_exact": bool(
            initial_equal and fingerprint == EXPECTED_CASE_FINGERPRINT
        ),
        "both_endpoint_ownership_checks_bitwise": ownership_gate,
        "legacy_post_axis_amplification_detected_at_both_walls_and_times": (
            post_axis_gate
        ),
        "legacy_post_outer_amplification_detected_at_both_walls_and_times": (
            post_outer_gate
        ),
        "owner_last_preserves_pre_overwrite_outer_scale": owner_scale_gate,
        "all_reconstructed_values_and_captured_stages_finite": finite_gate,
    }
    all_gates = bool(all(gates.values()))
    result = {
        "schema": "A790-G8-R10-current-code-staged-replay-v1",
        "classification": "current_code_deterministic_replay_not_historical_RHS_proof",
        "scope": (
            "fixed-grid coordinate-time semi-discrete D_X^2J on exact archived "
            "Test-10B G8/R10 accepted states; no moving-cap, physical, or "
            "continuum interpretation"
        ),
        "provenance": {
            "rebuilt_parent": "deterministic G8 R12 parent restricted bitwise to R10",
            "initial_array_equal": initial_equal,
            "initial_maximum_absolute_difference": float(
                np.max(np.abs(initial - archived_initial))
            ),
            "case_fingerprint": fingerprint,
            "expected_case_fingerprint": EXPECTED_CASE_FINGERPRINT,
            "case_fingerprint_match": fingerprint == EXPECTED_CASE_FINGERPRINT,
            "fingerprint_covers": [
                "z", "r", "initial q", "initial source", "initial memory",
                "archived reduced first jets", "archived reduced second jets",
            ],
            "historical_Test10B_runner_sha256": HISTORICAL_TEST10B_RUNNER_SHA256,
            "current_Test10B_dependency_runner_sha256": sha256_file(
                ROOT / "run_corrected_fold_live_nonlinear_gauge_source.py"
            ),
            "dependency_runner_hash_match": bool(
                sha256_file(ROOT / "run_corrected_fold_live_nonlinear_gauge_source.py")
                == HISTORICAL_TEST10B_RUNNER_SHA256
            ),
            "pre_refactor_core_sha256_recorded_by_direct_audit": (
                PRE_REFACTOR_CORE_SHA256_FROM_DIRECT_AUDIT
            ),
            "current_core_sha256": current_core_hash,
            "current_core_differs_from_pre_refactor_record": bool(
                current_core_hash != PRE_REFACTOR_CORE_SHA256_FROM_DIRECT_AUDIT
            ),
            "historical_identity_limit": (
                "Test 10B did not archive endpoint source-time/source-second-time "
                "arrays or the initial outer-reference acceleration and did not "
                "hash all transitive RHS modules. Those quantities are recomputed "
                "deterministically by current code, so bitwise historical RHS "
                "identity is not established."
            ),
            "input_sha256": {
                str(path.relative_to(ROOT)): sha256_file(path)
                for path in input_paths
            },
            "endpoint_archive_sha256": endpoint_hashes,
        },
        "configuration": {
            "grid": "G8", "parent_domain": "R12", "replay_domain": "R10",
            "shape": list(case["initial"].shape), "dt": DT,
            "stencil_width": int(case["rhs"].stencil_width),
            "legacy_mode": "legacy_wall_axis_outer",
            "experimental_mode": "wall_owner_last_experimental",
        },
        "endpoints": endpoint_records,
        "stage_attribution_gate_records": endpoint_gate_records,
        "gates": {
            **gates,
            "all_stage_attribution_gates_pass": all_gates,
            "classification": (
                "pass_current_code_stage_attribution"
                if all_gates else "review_current_code_stage_attribution"
            ),
            "gate_scope": (
                "comparative fixed-state stage attribution only; no arbitrary "
                "physical residual smallness gate"
            ),
        },
        "interpretation": {
            "permitted": (
                "current-code operation-order attribution on exact archived states"
            ),
            "not_permitted": [
                "bitwise historical native-RHS claim",
                "physical interface residual claim",
                "continuum convergence claim",
                "moving-cap or covariant second-derivative claim",
            ],
            "absolute_DX2J_caveat": (
                "The archived q and v have nonzero semi-discrete J and D_XJ. "
                "The wall solve closes its differentiated coordinate Robin row, "
                "but the normalized J second tangent can retain terms proportional "
                "to those state-level defects. Stage jumps at fixed q,v are the "
                "operation-order discriminator."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
    }
    atomic_write_json(OUTPUT, result)
    print(json.dumps({
        "output": str(OUTPUT.relative_to(ROOT)),
        "output_sha256": sha256_file(OUTPUT),
        "case_fingerprint": fingerprint,
        "build_seconds": build_seconds,
        "setup_seconds": setup_seconds,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
