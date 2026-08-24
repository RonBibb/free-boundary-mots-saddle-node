#!/usr/bin/env python3
"""Run the staged acceleration boundary-repair manufactured controls."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.junction_preservation_diagnostic import (  # noqa: E402
    WALLS,
    _orthonormal_frames,
    manufactured_state,
    wall_junction_rows,
)
from bhps.junction_second_preservation_diagnostic import (  # noqa: E402
    wall_junction_second_tangent,
)
from bhps.nonlinear_regular_so3_evolution import (  # noqa: E402
    GaugeTaylorSource,
    NativeRegularSO3RHS,
    StageRegularGaugeSource,
    apply_compact_wall_acceleration,
    apply_outer_sommerfeld_acceleration,
    compact_wall_normal_gauge_acceleration_residuals,
    fill_regular_axis,
    reconcile_wall_owner_axis_null_channels,
    solve_compact_wall_coupled_phi_normal_acceleration,
)
from bhps.recovery_indexer import atomic_write_json, sha256_file  # noqa: E402
from bhps.staged_boundary_preservation import (  # noqa: E402
    evaluate_boundary_stage_sequence,
)


PROTOCOL = Path("notes/115_A790_staged_acceleration_boundary_repair_protocol.md")
OUTPUT = Path("results/corrected_A790_staged_boundary_repair_controls.json")
INPUTS = (
    PROTOCOL,
    Path(__file__).name,
    Path("src/bhps/nonlinear_regular_so3_evolution.py"),
    Path("src/bhps/junction_preservation_diagnostic.py"),
    Path("src/bhps/junction_second_preservation_diagnostic.py"),
    Path("src/bhps/staged_boundary_preservation.py"),
    Path("src/bhps/simultaneous_acceleration_boundary_closure.py"),
    Path("src/bhps/adm_corner.py"),
    Path("src/bhps/gw_slice_high_order_solver.py"),
    Path("src/bhps/gh_source_driver.py"),
    Path("src/bhps/linearized_gh_einstein_scalar.py"),
    Path("src/bhps/regular_so3_gh_reduction.py"),
    Path("src/bhps/recovery_indexer.py"),
    Path("tests/test_nonlinear_regular_so3_evolution.py"),
    Path("tests/test_junction_second_preservation_diagnostic.py"),
    Path("tests/test_simultaneous_acceleration_boundary_closure.py"),
)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _second_endpoint_record(q, v, acceleration, z, r, background):
    walls = {}
    for wall in WALLS:
        record = wall_junction_second_tangent(
            q, v, acceleration, z, r, background, wall,
        )
        frames, frame_defect = _orthonormal_frames(record["metric_tensor"])
        transformed = np.einsum(
            "nai,nab,nbj->nij", frames, record["DX2J_tensor"], frames,
        )
        profile = np.linalg.norm(transformed, axis=(1, 2))
        walls[wall] = {
            "axis_DX2J_orthonormal_frobenius": float(profile[0]),
            "outer_DX2J_orthonormal_frobenius": float(profile[-1]),
            "full_DX2J_orthonormal_Linf": float(np.max(profile)),
            "Phi_DX2_axis": float(abs(
                record["separate_rows"]["DX2_Phi_robin"][0]
            )),
            "Phi_DX2_outer": float(abs(
                record["separate_rows"]["DX2_Phi_robin"][-1]
            )),
            "Phi_DX2_full_Linf": float(np.max(np.abs(
                record["separate_rows"]["DX2_Phi_robin"]
            ))),
            "chi_DX2_axis": float(abs(
                record["separate_rows"]["DX2_chi_neumann"][0]
            )),
            "chi_DX2_outer": float(abs(
                record["separate_rows"]["DX2_chi_neumann"][-1]
            )),
            "chi_DX2_full_Linf": float(np.max(np.abs(
                record["separate_rows"]["DX2_chi_neumann"]
            ))),
            "frame_defect_maximum": float(np.max(frame_defect)),
            "raw_vs_cancellation_exposed_defect": record[
                "raw_vs_cancellation_exposed_maximum_absolute_defect"
            ],
        }
    return walls


def _maximum_lane(records, key):
    return max(float(record[key]) for record in records.values())


def manufactured_order_control():
    data = manufactured_state(nz=17, nr=25)
    q = data["position"]
    v = data["velocity"]
    z = data["z"]
    r = data["r"]
    background = data["background"]
    zz, rr = np.meshgrid(z, r, indexing="ij")
    acceleration = np.zeros_like(q)
    for field in range(9):
        acceleration[:, :, field] = (
            (0.03 + 0.004 * field) * (1.0 + 0.2 * zz)
            * np.exp((0.4 + 0.03 * field) * rr**2)
        )
    acceleration = fill_regular_axis(acceleration, r)
    normal = np.stack((acceleration[0, :, 6], acceleration[-1, :, 6]))

    prefill, _ = apply_compact_wall_acceleration(
        q, v, acceleration, z, r, background, normal, fill_axis_after=False,
    )
    wall_then_axis, _ = apply_compact_wall_acceleration(
        q, v, acceleration, z, r, background, normal,
    )
    legacy_final, legacy_outer = apply_outer_sommerfeld_acceleration(
        q, v, wall_then_axis, q, np.zeros_like(q), 0.0, r,
    )
    outer_first, owner_outer = apply_outer_sommerfeld_acceleration(
        q, v, acceleration, q, np.zeros_like(q), 0.0, r,
    )
    owner_wall, _ = apply_compact_wall_acceleration(
        q, v, outer_first, z, r, background, normal, fill_axis_after=False,
    )
    owner_prefit = fill_regular_axis(owner_wall, r)
    owner_axis_null_defect_before = float(np.max(np.abs(
        owner_wall[:, 0, 4:6] - owner_prefit[:, 0, 4:6]
    )))
    owner_final = reconcile_wall_owner_axis_null_channels(owner_wall, r)
    owner_other_channel_change = float(np.max(np.abs(
        owner_final[:, :, (0, 1, 2, 3, 6, 7, 8)]
        - owner_wall[:, :, (0, 1, 2, 3, 6, 7, 8)]
    )))
    owner_null_off_axis_change = float(np.max(np.abs(
        owner_final[:, 1:, 4:6] - owner_wall[:, 1:, 4:6]
    )))
    owner_postfit = fill_regular_axis(owner_final, r)
    owner_axis_null_defect_after = float(np.max(np.abs(
        owner_final[:, 0, 4:6] - owner_postfit[:, 0, 4:6]
    )))
    null_reconciliation_tensor_change = 0.0
    for wall in WALLS:
        before = wall_junction_second_tangent(
            q, v, owner_wall, z, r, background, wall,
        )["DX2J_tensor"]
        after = wall_junction_second_tangent(
            q, v, owner_final, z, r, background, wall,
        )["DX2J_tensor"]
        null_reconciliation_tensor_change = max(
            null_reconciliation_tensor_change,
            float(np.max(np.abs(after - before))),
        )

    exact = {
        wall: wall_junction_rows(q, v, z, r, background, wall)
        for wall in WALLS
    }
    exact_j = max(np.max(np.abs(item["J_tensor"])) for item in exact.values())
    exact_dxj = max(np.max(np.abs(item["DXJ_tensor"])) for item in exact.values())
    endpoints = {
        "wall_endpoint_solve_before_axis": _second_endpoint_record(
            q, v, prefill, z, r, background,
        ),
        "legacy_after_axis_fill": _second_endpoint_record(
            q, v, wall_then_axis, z, r, background,
        ),
        "legacy_after_outer_overwrite": _second_endpoint_record(
            q, v, legacy_final, z, r, background,
        ),
        "owner_last_final": _second_endpoint_record(
            q, v, owner_final, z, r, background,
        ),
    }
    legacy_audit = evaluate_boundary_stage_sequence(
        q, v, [
            {"name": "initial_axis_fill", "acceleration": acceleration},
            {"name": "wall_endpoint_solve", "acceleration": prefill},
            {"name": "post_wall_axis_fill", "acceleration": wall_then_axis},
            {"name": "post_wall_outer_overwrite", "acceleration": legacy_final},
        ], z, r, background,
    )
    owner_audit = evaluate_boundary_stage_sequence(
        q, v, [
            {"name": "initial_axis_fill", "acceleration": acceleration},
            {"name": "outer_open_face", "acceleration": outer_first},
            {"name": "wall_owner_last_pre_axis_null_reconciliation", "acceleration": owner_wall},
            {"name": "wall_owner_last", "acceleration": owner_final},
        ], z, r, background,
    )
    causal_defect = max(
        jump["walls"][wall]["causal_identity_maximum_absolute_defect"]
        for audit in (legacy_audit, owner_audit)
        for jump in audit["jumps"] for wall in WALLS
    )
    hessian_change = max(
        jump["walls"][wall]["velocity_hessian_change_maximum_absolute"]
        for audit in (legacy_audit, owner_audit)
        for jump in audit["jumps"] for wall in WALLS
    )
    prefill_axis = _maximum_lane(
        endpoints["wall_endpoint_solve_before_axis"],
        "axis_DX2J_orthonormal_frobenius",
    )
    prefill_outer = _maximum_lane(
        endpoints["wall_endpoint_solve_before_axis"],
        "outer_DX2J_orthonormal_frobenius",
    )
    prefill_full = _maximum_lane(
        endpoints["wall_endpoint_solve_before_axis"],
        "full_DX2J_orthonormal_Linf",
    )
    prefill_scalar_full = max(
        record[key]
        for record in endpoints["wall_endpoint_solve_before_axis"].values()
        for key in ("Phi_DX2_full_Linf", "chi_DX2_full_Linf")
    )
    legacy_axis = _maximum_lane(
        endpoints["legacy_after_axis_fill"],
        "axis_DX2J_orthonormal_frobenius",
    )
    legacy_outer_value = _maximum_lane(
        endpoints["legacy_after_outer_overwrite"],
        "outer_DX2J_orthonormal_frobenius",
    )
    owner_axis = _maximum_lane(
        endpoints["owner_last_final"],
        "axis_DX2J_orthonormal_frobenius",
    )
    owner_outer_value = _maximum_lane(
        endpoints["owner_last_final"],
        "outer_DX2J_orthonormal_frobenius",
    )
    owner_full = _maximum_lane(
        endpoints["owner_last_final"],
        "full_DX2J_orthonormal_Linf",
    )
    owner_scalar = max(
        record[key]
        for record in endpoints["owner_last_final"].values()
        for key in ("Phi_DX2_full_Linf", "chi_DX2_full_Linf")
    )
    gates = {
        "compatible_J_below_1e_11": bool(exact_j < 1e-11),
        "compatible_DXJ_below_1e_11": bool(exact_dxj < 1e-11),
        "prefill_axis_below_1e_11": bool(prefill_axis < 1e-11),
        "prefill_outer_below_1e_11": bool(prefill_outer < 1e-11),
        "prefill_full_metric_below_1e_11": bool(prefill_full < 1e-11),
        "prefill_full_scalar_rows_below_1e_11": bool(
            prefill_scalar_full < 1e-11
        ),
        "post_wall_axis_defect_detected": bool(legacy_axis > 1e-12),
        "post_wall_outer_defect_detected": bool(legacy_outer_value > 1e-3),
        "owner_axis_below_1e_11": bool(owner_axis < 1e-11),
        "owner_outer_below_1e_11": bool(owner_outer_value < 1e-11),
        "owner_full_metric_below_1e_11": bool(owner_full < 1e-11),
        "owner_scalar_rows_below_1e_11": bool(owner_scalar < 1e-11),
        "owner_axis_null_defect_reproduced_before_reconciliation": bool(
            owner_axis_null_defect_before > 1e-5
        ),
        "owner_axis_null_channels_reconciled_below_1e_12": bool(
            owner_axis_null_defect_after < 1e-12
        ),
        "axis_null_reconciliation_leaves_wall_tensor_exactly_unchanged": bool(
            null_reconciliation_tensor_change == 0.0
        ),
        "axis_null_reconciliation_changes_only_q4_q5_axis_values": bool(
            owner_other_channel_change == 0.0
            and owner_null_off_axis_change == 0.0
        ),
        "owner_preserves_all_compact_open_values_bitwise": bool(
            np.array_equal(owner_final[1:-1], outer_first[1:-1])
        ),
        "owner_outer_target_residual_below_1e_12": bool(
            owner_outer["maximum_normalized_acceleration_residual"] < 1e-12
        ),
        "causal_identity_below_1e_12": bool(causal_defect < 1e-12),
        "velocity_hessian_stage_change_zero": bool(hessian_change == 0.0),
        "all_stage_audits_finite": bool(legacy_audit["finite"] and owner_audit["finite"]),
    }
    return {
        "exact_J_maximum_absolute": float(exact_j),
        "exact_DXJ_maximum_absolute": float(exact_dxj),
        "endpoint_records": endpoints,
        "legacy_outer_diagnostic": legacy_outer,
        "owner_outer_diagnostic": owner_outer,
        "owner_open_values_maximum_absolute_change": float(
            np.max(np.abs(owner_final[1:-1] - outer_first[1:-1]))
        ),
        "owner_axis_null_defect_before_reconciliation": (
            owner_axis_null_defect_before
        ),
        "owner_axis_null_defect_after_reconciliation": (
            owner_axis_null_defect_after
        ),
        "axis_null_reconciliation_DX2J_tensor_change_maximum_absolute": (
            null_reconciliation_tensor_change
        ),
        "axis_null_reconciliation_other_channel_change_maximum_absolute": (
            owner_other_channel_change
        ),
        "axis_null_reconciliation_q4_q5_off_axis_change_maximum_absolute": (
            owner_null_off_axis_change
        ),
        "causal_identity_maximum_absolute_defect": float(causal_defect),
        "velocity_hessian_stage_change_maximum_absolute": float(hessian_change),
        "legacy_stage_audit": legacy_audit,
        "owner_stage_audit": owner_audit,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def coupled_block_control():
    z = np.linspace(1.0, 2.0, 17)
    r = np.linspace(0.0, 1.0, 25)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    q = np.zeros((len(z), len(r), 9))
    v = np.zeros_like(q)
    acceleration = np.zeros_like(q)
    q[:, :, 2] = -1.0 + 0.01 * zz
    q[:, :, 3] = 1.0 + 0.02 * zz
    q[:, :, 4] = 0.01
    q[:, :, 5] = 0.005
    q[:, :, 6] = 1.1 + 0.02 * zz + 0.01 * rr**2
    q[:, :, 7] = 0.03 * zz + 0.01 * rr**2
    q[:, :, 8] = 0.02 * zz
    v[:, :, 2] = 0.01 * (1.0 + zz)
    v[:, :, 3] = -0.02
    v[:, :, 6] = 0.015
    v[:, :, 7] = 0.01
    v[:, :, 8] = 0.02
    for field in range(9):
        acceleration[:, :, field] = (
            (0.03 + 0.002 * field) * (1.0 + 0.1 * zz + 0.02 * rr**2)
        )
    source = np.zeros((len(z), len(r), 3))
    source_time = np.zeros_like(source)
    source_second = np.zeros_like(source)
    background = {
        "wall_stiffness": 2.0, "v0": 0.02, "v1": 0.06,
        "beta_a": 0.1, "beta_b": 0.12,
        "wall_potential_a": 0.0, "wall_potential_b": 0.0,
    }
    coupled, algebra = solve_compact_wall_coupled_phi_normal_acceleration(
        q, v, acceleration, source, source_time, source_second,
        z, r, background,
    )
    normal = np.stack((coupled[0, :, 6], coupled[-1, :, 6]))
    solved, _ = apply_compact_wall_acceleration(
        q, v, coupled, z, r, background, normal, fill_axis_after=False,
    )
    normal_residual = compact_wall_normal_gauge_acceleration_residuals(
        q, v, solved, source, source_time, source_second,
        z, r, background, radial_buffer=0,
    )
    scalar = {}
    for wall in WALLS:
        record = wall_junction_second_tangent(
            q, v, solved, z, r, background, wall,
        )
        scalar[wall] = {
            "Phi_DX2_maximum_absolute": float(np.max(np.abs(
                record["separate_rows"]["DX2_Phi_robin"]
            ))),
            "chi_DX2_maximum_absolute": float(np.max(np.abs(
                record["separate_rows"]["DX2_chi_neumann"]
            ))),
        }
    scalar_maximum = max(value for item in scalar.values() for value in item.values())
    gates = {
        "coupled_algebra_passes": bool(algebra["passed"]),
        "condition_below_1e12": bool(algebra["maximum_condition"] <= 1e12),
        "pivot_strength_above_1e_10": bool(
            algebra["minimum_pivot_strength"] >= 1e-10
        ),
        "linear_residual_below_1e_12": bool(
            algebra["maximum_normalized_linear_residual"] < 1e-12
        ),
        "full_unbuffered_normal_residual_below_1e_10": bool(
            normal_residual["maximum"] < 1e-10
        ),
        "Phi_chi_second_rows_below_1e_11": bool(scalar_maximum < 1e-11),
    }
    return {
        "algebra": algebra,
        "normal_residual": normal_residual,
        "separate_scalar_rows": scalar,
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def native_rhs_control():
    z = np.linspace(1.0, 2.0, 9)
    r = np.linspace(0.0, 1.0, 13)
    q = np.zeros((len(z), len(r), 9))
    v = np.zeros_like(q)
    q[:, :, 2] = -1.0
    q[:, :, 3] = 1.0
    q[:, :, 6] = 1.0
    source = GaugeTaylorSource(
        np.zeros((len(z), len(r), 5)),
        np.zeros((len(z), len(r), 5, 5)), z, r,
    )
    background = {
        "wall_stiffness": 0.0, "v0": 0.0, "v1": 0.0,
        "beta_a": 0.0, "beta_b": 0.0,
        "wall_potential_a": 0.0, "wall_potential_b": 0.0,
    }
    common = {
        "z": z, "r": r, "gauge_source": source, "mass_squared": 0.0,
        "background": background,
        "normal_wall_acceleration": np.zeros((2, len(r))),
        "live_outer_sommerfeld": True,
    }
    legacy = NativeRegularSO3RHS(
        **common, boundary_closure_mode="legacy_wall_axis_outer",
    )
    owner = NativeRegularSO3RHS(
        **common, boundary_closure_mode="wall_owner_last_experimental",
    )
    for rhs in (legacy, owner):
        rhs.outer_reference_position = q.copy()
        rhs.outer_reference_acceleration = np.zeros_like(q)

    legacy_plain, _ = legacy.acceleration(0.0, q, v)
    legacy_traced, legacy_diagnostic = legacy.acceleration(
        0.0, q, v, capture_boundary_stages=True,
    )
    owner_plain, _ = owner.acceleration(0.0, q, v)
    owner_traced, owner_diagnostic = owner.acceleration(
        0.0, q, v, capture_boundary_stages=True,
    )
    live_source_values = np.zeros((len(z), len(r), 3))
    live_source = StageRegularGaugeSource(
        live_source_values, np.zeros_like(live_source_values), z, r,
    )
    live_normal_owner = NativeRegularSO3RHS(
        z, r, live_source, 0.0, background, np.zeros((2, len(r))),
        live_normal_wall_gauge=True,
        boundary_closure_mode="wall_owner_last_experimental",
    )
    live_normal_acceleration, live_normal_diagnostic = (
        live_normal_owner.acceleration(
            0.0, q, v,
            gauge_source_second_time=np.zeros_like(live_source_values),
            capture_boundary_stages=True,
        )
    )
    live_normal_endpoints = _second_endpoint_record(
        q, v, live_normal_acceleration, z, r, background,
    )
    live_normal_metric = _maximum_lane(
        live_normal_endpoints, "full_DX2J_orthonormal_Linf",
    )
    live_normal_scalar = max(
        record[key]
        for record in live_normal_endpoints.values()
        for key in ("Phi_DX2_full_Linf", "chi_DX2_full_Linf")
    )
    legacy_endpoints = _second_endpoint_record(
        q, v, legacy_plain, z, r, background,
    )
    owner_endpoints = _second_endpoint_record(
        q, v, owner_plain, z, r, background,
    )
    legacy_outer = _maximum_lane(
        legacy_endpoints, "outer_DX2J_orthonormal_frobenius",
    )
    owner_outer = _maximum_lane(
        owner_endpoints, "outer_DX2J_orthonormal_frobenius",
    )
    legacy_audit = evaluate_boundary_stage_sequence(
        q, v, legacy_diagnostic["boundary_stages"], z, r, background,
        buffer_points=3,
    )
    owner_audit = evaluate_boundary_stage_sequence(
        q, v, owner_diagnostic["boundary_stages"], z, r, background,
        buffer_points=3,
    )
    owner_outer_stage = next(
        stage["acceleration"] for stage in owner_diagnostic["boundary_stages"]
        if stage["name"] == "outer_open_face_before_wall"
    )
    gates = {
        "legacy_trace_bitwise_observational": bool(
            np.array_equal(legacy_plain, legacy_traced)
        ),
        "owner_trace_bitwise_observational": bool(
            np.array_equal(owner_plain, owner_traced)
        ),
        "legacy_outer_defect_above_1": bool(legacy_outer > 1.0),
        "owner_outer_defect_below_1e_10": bool(owner_outer < 1e-10),
        "owner_open_values_bitwise_preserved": bool(
            np.array_equal(owner_plain[1:-1], owner_outer_stage[1:-1])
        ),
        "legacy_audit_finite": bool(legacy_audit["finite"]),
        "owner_audit_finite": bool(owner_audit["finite"]),
        "integrated_live_normal_owner_finite": bool(
            live_normal_diagnostic["finite"]
        ),
        "integrated_live_normal_coupled_block_passes": bool(
            live_normal_diagnostic["normal_wall_gauge"]["coupled_block"]["passed"]
        ),
        "integrated_live_normal_full_residual_below_1e_10": bool(
            live_normal_diagnostic["normal_wall_gauge"]["final_residual"]["maximum"]
            < 1e-10
        ),
        "integrated_live_normal_metric_below_1e_10": bool(
            live_normal_metric < 1e-10
        ),
        "integrated_live_normal_scalar_rows_below_1e_11": bool(
            live_normal_scalar < 1e-11
        ),
        "integrated_live_normal_method_is_direct_4x4": bool(
            live_normal_diagnostic["boundary_parameters"]["normal_wall_method"]
            == "direct_coupled_4x4_both_walls"
        ),
    }
    return {
        "legacy_endpoint_records": legacy_endpoints,
        "owner_endpoint_records": owner_endpoints,
        "legacy_diagnostic": {
            key: value for key, value in legacy_diagnostic.items()
            if key != "boundary_stages"
        },
        "owner_diagnostic": {
            key: value for key, value in owner_diagnostic.items()
            if key != "boundary_stages"
        },
        "legacy_stage_audit": legacy_audit,
        "owner_stage_audit": owner_audit,
        "integrated_live_normal_endpoint_records": live_normal_endpoints,
        "integrated_live_normal_diagnostic": {
            key: value for key, value in live_normal_diagnostic.items()
            if key != "boundary_stages"
        },
        "gates": gates,
        "passed": bool(all(gates.values())),
    }


def main():
    missing = [str(path) for path in INPUTS if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"missing staged-repair inputs: {missing}")
    manufactured = manufactured_order_control()
    coupled = coupled_block_control()
    native = native_rhs_control()
    gates = {
        "manufactured_operation_order_control_passes": manufactured["passed"],
        "coupled_Phi_gzz_control_passes": coupled["passed"],
        "native_RHS_smoke_control_passes": native["passed"],
    }
    payload = {
        "schema": "bhps-staged-boundary-repair-controls-v1",
        "date": "2026-08-15",
        "classification": (
            "implementation_ordering_mechanism_reproduced_and_"
            "experimental_repair_qualified_for_physical_audit"
            if all(gates.values()) else "control_failure"
        ),
        "scope": (
            "manufactured and small native-RHS implementation controls only; "
            "not a physical A=7.90 continuum result"
        ),
        "physics_changes": {
            "action_changed": False,
            "Israel_wall_law_changed": False,
            "scalar_wall_laws_changed": False,
            "new_interface_term_added": False,
            "experimental_change": (
                "boundary operation ownership/order plus a directly coupled "
                "Phi/g_zz acceleration solve"
            ),
        },
        "configuration": {
            "stencil_width": 7,
            "axis_fit_window": 0.5,
            "axis_fit_degree": 3,
            "legacy_boundary_mode": "legacy_wall_axis_outer",
            "experimental_boundary_mode": "wall_owner_last_experimental",
            "coupled_maximum_condition": 1e12,
            "coupled_minimum_pivot_strength": 1e-10,
            "coupled_maximum_normalized_linear_residual": 1e-12,
            "full_unbuffered_normal_residual_limit": 1e-10,
            "owner_axis_null_reconciliation_channels": [4, 5],
            "manufactured_grid": [17, 25, 9],
            "native_smoke_grid": [9, 13, 9],
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "input_sha256": {str(path): sha256_file(path) for path in INPUTS},
        "manufactured_operation_order": manufactured,
        "coupled_Phi_gzz_block": coupled,
        "native_RHS_smoke": native,
        "gates": gates,
        "passed": bool(all(gates.values())),
        "remaining_decision_gate": (
            "fresh common-parent G8/G9/G10 physical sequence at dt, dt/2, "
            "and dt/4 with full stage/source snapshots; old Test10E recovery "
            "artifacts are not cache-valid for this refactored boundary code"
        ),
    }
    atomic_write_json(OUTPUT, _jsonable(payload))
    print(json.dumps({
        "output": str(OUTPUT),
        "sha256": sha256_file(OUTPUT),
        "classification": payload["classification"],
        "passed": payload["passed"],
        "gates": gates,
    }, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
