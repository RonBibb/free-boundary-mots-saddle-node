#!/usr/bin/env python3
"""Restartable dense Test-14B quasi-local balance-closure audit."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.dynamical_capped_horizon import (
    prepare_capped_expansion_slice,
    solve_spectral_dynamical_capped_surface,
)
from bhps.dynamical_capped_horizon_bvp import solve_dynamical_capped_surface_bvp
from bhps.recovery_indexer import RecoveryIndex, atomic_write_json, sha256_file
from bhps.test14_quasilocal_charge import reflected_cap_charge
from bhps.test14b_balance_closure import (
    analytic_controls,
    evaluate_balance_leaf,
    five_point_history_derivative,
    relative_scale_error,
)
from run_corrected_A790_independent_dynamic_BVP_detector import admitted
from run_corrected_fold_regular_so3_runtime import build_geometry


PROTOCOL = Path("notes/96_A790_test14B_balance_closure_protocol.md")
STATE = Path("results/corrected_A790_t008_long_evolution_state.npz")
NOTE85 = Path("results/corrected_A790_t008_long_evolution.json")
NOTE95 = Path("results/corrected_A790_test14_quasilocal_charge_history.json")
OUTPUT = Path("results/corrected_A790_test14b_balance_closure.json")
MANIFEST = Path("results/corrected_A790_test14b_recovery_v4.json")
STAGES = Path("results/corrected_A790_test14b_stages")
TIME_START_INDEX = 5
PRIMARY_END = 0.004
STRIDES = (1, 2, 4)
BRANCHES = ("inner", "outer")
GRIDS = ("G7", "G8")
RATE_NAMES = (
    "curvature_smooth", "curvature_seam",
    "matter_scalar_ll_smooth", "matter_scalar_ln_smooth",
    "shear_smooth", "normal_connection_smooth",
    "brane_matter_seam", "normal_connection_seam",
    "vacuum", "ads_background",
)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def stage_json(index, stage_id, path, compute, metadata, maximum_seconds=3600.0):
    index.register(stage_id, "test14b-balance", maximum_seconds, metadata)
    validated = index.validated_path(stage_id)
    if validated is not None:
        return json.loads(validated.read_text())
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        payload = _jsonable(compute())
        atomic_write_json(path, payload)
        index.mark_complete(
            stage_id, path, time.perf_counter() - started,
            {"finite": bool(payload.get("finite", True))},
        )
        return payload
    except Exception as error:
        index.mark_failed(stage_id, repr(error))
        raise


def recovery_smoke_test():
    with tempfile.TemporaryDirectory(prefix="bhps-test14b-recovery-") as directory:
        root = Path(directory)
        source = root / "input.txt"
        source.write_text("fixed Test-14B input\n")
        expected = {str(source): sha256_file(source)}
        manifest = root / "manifest.json"
        stage = root / "stage.json"
        first = RecoveryIndex(manifest, PROTOCOL, expected, 60.0)
        first.register("smoke", "test14b-smoke", 10.0, {"stage": 1})
        first.mark_running("smoke")
        atomic_write_json(stage, {"finite": True, "value": 1})
        first.mark_complete("smoke", stage, 0.01)
        second = RecoveryIndex(manifest, PROTOCOL, expected, 60.0)
        resumed = second.validated_path("smoke") == stage
        atomic_write_json(stage, {"finite": True, "value": 2})
        rejected = second.validated_path("smoke") is None
        return {
            "valid_partial_restart": bool(resumed),
            "corruption_rejected": bool(rejected),
            "passed": bool(resumed and rejected),
        }


def surface_payload(surface, grid, current_time, branch):
    return {
        "grid": grid,
        "time": float(current_time),
        "branch": branch,
        "surface": {
            "theta": np.asarray(surface["theta"]),
            "rho": np.asarray(surface["rho"]),
            "slope": np.asarray(surface["slope"]),
            "rho_axis": float(surface["rho_axis"]),
            "rho_brane": float(surface["rho_brane"]),
            "converged": bool(surface["converged"]),
            "solver_success": bool(surface.get("solver_success", True)),
            "in_domain": bool(surface["in_domain"]),
            "local_expansion_interior_maximum": float(surface[
                "local_expansion_interior_maximum"
            ]),
            "boundary_slope_error": float(surface["boundary_slope_error"]),
            "mesh_nodes_used": int(surface["mesh_nodes_used"]),
        },
        "finite": bool(
            surface["converged"]
            and np.all(np.isfinite(surface["rho"]))
            and np.all(np.isfinite(surface["slope"]))
        ),
    }


def unpack_surface(record):
    return {
        "theta": np.asarray(record["surface"]["theta"], dtype=float),
        "rho": np.asarray(record["surface"]["rho"], dtype=float),
        "slope": np.asarray(record["surface"]["slope"], dtype=float),
        **{
            key: record["surface"][key]
            for key in (
                "rho_axis", "rho_brane", "converged", "solver_success",
                "in_domain", "local_expansion_interior_maximum",
                "boundary_slope_error", "mesh_nodes_used",
            )
        },
    }


def solve_surface(position, velocity, z, r, initial, grid, current_time, branch):
    surface = solve_dynamical_capped_surface_bvp(
        position, velocity, z, r, initial, tolerance=2e-5, nodes=121,
        maximum_nodes=6000, dense_nodes=501,
    )
    if not admitted(surface):
        raise RuntimeError(
            f"unadmitted {grid} t={current_time:.6f} {branch} surface: {surface}"
        )
    return surface_payload(surface, grid, current_time, branch)


def confirmation_payload(
    position, velocity, z, r, initial, grid, current_time, branch,
):
    modes = 28 if branch == "inner" else 36
    surface = solve_spectral_dynamical_capped_surface(
        position, velocity, z, r, initial, tolerance=5e-5,
        collocation_nodes=max(101, 2 * modes + 21), cosine_modes=modes,
        maximum_evaluations=300,
    )
    confirmation_admitted = bool(
        surface["optimizer_success"] and surface["in_domain"]
        and surface["interior_expansion_maximum"] < 5e-3
        and surface["boundary_slope_error"] < 1e-8
    )
    if not confirmation_admitted:
        raise RuntimeError(
            f"unadmitted confirmation {grid} t={current_time:.6f} {branch}: "
            f"{surface}"
        )
    charge = reflected_cap_charge(position, velocity, z, r, surface)
    selected_charge = reflected_cap_charge(position, velocity, z, r, initial)
    theta = np.asarray(initial["theta"], dtype=float)
    confirmation_rho = np.interp(
        theta, np.asarray(surface["theta"]), np.asarray(surface["rho"]),
    )
    profile_scale = max(float(np.max(np.abs(initial["rho"]))), 1e-300)
    return {
        "grid": grid, "time": float(current_time), "branch": branch,
        "cosine_modes": modes,
        "maximum_profile_relative_scale_difference": float(
            np.max(np.abs(confirmation_rho - initial["rho"])) / profile_scale
        ),
        "area_relative_scale_difference": relative_scale_error(
            charge["reflection_doubled_area"],
            selected_charge["reflection_doubled_area"],
            selected_charge["reflection_doubled_area"],
        ),
        "charge_relative_scale_difference": relative_scale_error(
            charge["generalized_hawking_ads_charge_kappa5_squared_E"],
            selected_charge["generalized_hawking_ads_charge_kappa5_squared_E"],
            selected_charge["generalized_hawking_ads_charge_kappa5_squared_E"],
        ),
        "confirmation_expansion_maximum": float(
            surface["interior_expansion_maximum"]
        ),
        "confirmation_admitted_under_note81_threshold": confirmation_admitted,
        "finite": bool(charge["finite"]),
    }


def window_summary(records, left, right):
    selected = [
        item for item in records
        if float(left) - 1e-12 <= item["time"] <= float(right) + 1e-12
    ]
    if len(selected) < 2:
        raise ValueError("balance window has fewer than two records")
    times = np.asarray([item["time"] for item in selected])
    charge = np.asarray([item["charge_rate_target"]["charge"] for item in selected])
    total = np.asarray([item["total_balance_rate"] for item in selected])
    named = {
        name: np.asarray([item["rates"][name] for item in selected])
        for name in RATE_NAMES
    }
    delta = float(charge[-1] - charge[0])
    integrated = {name: float(np.trapezoid(value, times)) for name, value in named.items()}
    integrated_total = float(np.trapezoid(total, times))
    absolute_flux_norm = float(np.trapezoid(
        np.sum(np.abs(np.stack(list(named.values()))), axis=0), times,
    ))
    norm = max(abs(delta), absolute_flux_norm, 1e-12)
    residual = delta - integrated_total
    return {
        "left_time": float(times[0]), "right_time": float(times[-1]),
        "sample_count": len(selected), "delta_charge": delta,
        "integrated_named_rates": integrated,
        "integrated_total_rate": integrated_total,
        "closure_residual": float(residual),
        "balance_norm": float(norm),
        "normalized_absolute_residual": float(abs(residual) / norm),
    }


def term_window_stage(index, grid, branch, stride, name, window_name, summary):
    stage_id = f"term/{grid}/{branch}/stride{stride}/{name}/{window_name}"
    path = STAGES / (
        f"term_{grid}_{branch}_stride{stride}_{name}_{window_name}.json"
    )
    return stage_json(
        index, stage_id, path,
        lambda: {
            "grid": grid, "branch": branch, "stride": stride,
            "term": name, "window": window_name,
            "integrated_rate": summary["integrated_named_rates"][name],
            "left_time": summary["left_time"],
            "right_time": summary["right_time"], "finite": True,
        },
        {"grid": grid, "branch": branch, "stride": stride,
         "term": name, "window": window_name},
        maximum_seconds=60.0,
    )


def field_velocity_consistency(position, velocity, times, stride):
    derivative = five_point_history_derivative(position, times, stride=stride)
    difference = derivative - velocity
    relative = float(
        np.linalg.norm(difference)
        / max(np.linalg.norm(derivative), np.linalg.norm(velocity), 1e-300)
    )
    return {
        "relative_L2_difference": relative,
        "below_2_percent": bool(relative < 0.02),
    }


def main():
    STAGES.mkdir(parents=True, exist_ok=True)
    required = (
        PROTOCOL, STATE, NOTE85, NOTE95,
        Path("src/bhps/test14b_balance_closure.py"),
        Path("src/bhps/test14_quasilocal_charge.py"),
        Path("src/bhps/dynamical_capped_horizon.py"),
        Path("src/bhps/dynamical_capped_horizon_bvp.py"),
        Path("src/bhps/recovery_indexer.py"), Path(__file__),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing Test-14B inputs: {missing}")
    expected = {str(path): sha256_file(path) for path in required}
    recovery = recovery_smoke_test()
    index = RecoveryIndex(MANIFEST, PROTOCOL, expected, 3600.0)
    controls = stage_json(
        index, "controls/all", STAGES / "analytic_controls.json",
        analytic_controls, {"sealed_controls": [1, 2, 4, 5]}, 60.0,
    )
    if not controls["passed"] or not recovery["passed"]:
        raise RuntimeError("Test-14B analytic or recovery controls failed")

    print("loading dense A=7.90 state and fixed background", flush=True)
    archive = np.load(STATE)
    all_times = np.asarray(archive["times"], dtype=float)
    selected_indices = np.arange(TIME_START_INDEX, len(all_times))
    times = all_times[selected_indices]
    background_all = build_geometry("G6")["background"]
    background = {
        key: background_all[key] for key in (
            "mass_squared", "wall_stiffness", "v1", "wall_potential_b",
            "retuned_bare_tension_b", "beta_b",
        )
    }

    surface_records = {grid: {branch: [] for branch in BRANCHES} for grid in GRIDS}
    for grid in GRIDS:
        z = np.asarray(archive[f"{grid}_z"])
        r = np.asarray(archive[f"{grid}_r"])
        positions = np.asarray(archive[f"{grid}_position_history"])[selected_indices]
        velocities = np.asarray(archive[f"{grid}_velocity_history"])[selected_indices]
        for branch, initial_seed in zip(BRANCHES, (1.30, 1.60)):
            seed = initial_seed
            for local_index, current_time in enumerate(times):
                label = f"{current_time:.6f}".replace(".", "p")
                stage_id = f"surface/{grid}/{branch}/{label}"
                path = STAGES / f"surface_{grid}_{branch}_{label}.json"
                print(
                    f"surface {grid} {branch} {local_index + 1}/{len(times)} "
                    f"t={current_time:.6f}", flush=True,
                )
                record = stage_json(
                    index, stage_id, path,
                    lambda li=local_index, s=seed, ct=current_time: solve_surface(
                        positions[li], velocities[li], z, r, s, grid, ct,
                        branch,
                    ),
                    {"grid": grid, "branch": branch, "time": float(current_time),
                     "representation": "local_BVP_501"},
                )
                surface_records[grid][branch].append(record)
                seed = unpack_surface(record)

    confirmations = []
    absolute_confirmation_indices = sorted(set(
        [TIME_START_INDEX, 32, 64]
        + list(range(8, len(all_times), 8))
    ))
    for grid in GRIDS:
        z = np.asarray(archive[f"{grid}_z"])
        r = np.asarray(archive[f"{grid}_r"])
        positions = np.asarray(archive[f"{grid}_position_history"])
        velocities = np.asarray(archive[f"{grid}_velocity_history"])
        for absolute_index in absolute_confirmation_indices:
            if absolute_index < TIME_START_INDEX:
                continue
            local_index = absolute_index - TIME_START_INDEX
            current_time = all_times[absolute_index]
            for branch in BRANCHES:
                selected = unpack_surface(surface_records[grid][branch][local_index])
                label = f"{current_time:.6f}".replace(".", "p")
                stage_id = f"confirmation/{grid}/{branch}/{label}"
                path = STAGES / f"confirmation_{grid}_{branch}_{label}.json"
                print(f"confirmation {grid} {branch} t={current_time:.6f}", flush=True)
                confirmations.append(stage_json(
                    index, stage_id, path,
                    lambda ai=absolute_index, s=selected, ct=current_time:
                    confirmation_payload(
                        positions[ai], velocities[ai], z, r, s, grid, ct,
                        branch,
                    ),
                    {"grid": grid, "branch": branch, "time": float(current_time),
                     "representation": "cosine_spectral"},
                ))

    balance_records = {
        grid: {branch: {stride: [] for stride in STRIDES} for branch in BRANCHES}
        for grid in GRIDS
    }
    velocity_checks = {}
    for grid in GRIDS:
        z = np.asarray(archive[f"{grid}_z"])
        r = np.asarray(archive[f"{grid}_r"])
        positions = np.asarray(archive[f"{grid}_position_history"])[selected_indices]
        velocities = np.asarray(archive[f"{grid}_velocity_history"])[selected_indices]
        velocity_checks[grid] = field_velocity_consistency(
            positions, velocities, times, stride=1,
        )
        derivatives = {}
        for branch in BRANCHES:
            profiles = [unpack_surface(item) for item in surface_records[grid][branch]]
            rho_history = np.stack([item["rho"] for item in profiles])
            derivatives[branch] = {
                stride: five_point_history_derivative(
                    rho_history, times, stride=stride,
                ) for stride in STRIDES
            }
        for local_index, current_time in enumerate(times):
            prepared = prepare_capped_expansion_slice(
                positions[local_index], velocities[local_index], z, r,
            )
            for branch in BRANCHES:
                profile = unpack_surface(surface_records[grid][branch][local_index])
                for stride in STRIDES:
                    label = f"{current_time:.6f}".replace(".", "p")
                    stage_id = f"balance/{grid}/{branch}/stride{stride}/{label}"
                    path = STAGES / (
                        f"balance_{grid}_{branch}_stride{stride}_{label}.json"
                    )
                    print(
                        f"balance {grid} {branch} stride={stride} "
                        f"t={current_time:.6f}", flush=True,
                    )
                    record = stage_json(
                        index, stage_id, path,
                        lambda li=local_index, p=profile, st=stride, ct=current_time:
                        {
                            **evaluate_balance_leaf(
                                positions[li], velocities[li], z, r, p,
                                derivatives[branch][st][li], background,
                                prepared=prepared,
                            ),
                            "grid": grid, "branch": branch, "stride": st,
                            "time": float(ct),
                        },
                        {"grid": grid, "branch": branch, "stride": stride,
                         "time": float(current_time), "term": "all"},
                    )
                    balance_records[grid][branch][stride].append(record)

    summaries = {grid: {branch: {} for branch in BRANCHES} for grid in GRIDS}
    primary_windows = {
        "primary": (float(times[0]), PRIMARY_END),
        "sub_001_002": (0.001, 0.002),
        "sub_002_003": (0.002, 0.003),
        "sub_003_004": (0.003, 0.004),
        "extended": (float(times[0]), float(times[-1])),
    }
    for grid in GRIDS:
        for branch in BRANCHES:
            for stride in STRIDES:
                records = balance_records[grid][branch][stride]
                charge_history = np.asarray([
                    item["charge"]["generalized_hawking_ads_charge_kappa5_squared_E"]
                    for item in records
                ])
                area_history = np.asarray([
                    item["charge"]["reflection_doubled_area"] for item in records
                ])
                charge_rate = five_point_history_derivative(
                    charge_history, times, stride=stride,
                )
                area_rate = five_point_history_derivative(
                    area_history, times, stride=stride,
                )
                for item, q, qdot, adot in zip(
                    records, charge_history, charge_rate, area_rate,
                ):
                    item["charge_rate_target"] = {
                        "charge": float(q), "finite_difference_rate": float(qdot),
                    }
                    item["area_transport_check"] = {
                        "finite_difference_rate": float(adot),
                        "generator_rate": float(item["area_rate_transport"]),
                        "relative_scale_error": relative_scale_error(
                            adot, item["area_rate_transport"],
                            max(abs(adot), abs(item["area_rate_transport"]), 1.0),
                        ),
                    }
                    item["pointwise_balance_residual"] = float(
                        qdot - item["total_balance_rate"]
                    )
                windows = {
                    name: window_summary(records, *bounds)
                    for name, bounds in primary_windows.items()
                }
                summaries[grid][branch][str(stride)] = {
                    "windows": windows,
                    "maximum_area_transport_relative_scale_error": max(
                        item["area_transport_check"]["relative_scale_error"]
                        for item in records if item["time"] <= PRIMARY_END + 1e-12
                    ),
                    "maximum_absolute_theta_l": max(
                        item["geometry"]["maximum_absolute_theta_l"]
                        for item in records if item["time"] <= PRIMARY_END + 1e-12
                    ),
                    "all_theta_n_negative": bool(all(
                        item["geometry"]["all_theta_n_negative"]
                        for item in records if item["time"] <= PRIMARY_END + 1e-12
                    )),
                    "maximum_israel_seam_relative_scale_error": max(
                        item["seam"]["israel_intrinsic_relative_scale_error"]
                        for item in records if item["time"] <= PRIMARY_END + 1e-12
                    ),
                    "maximum_background_cancellation_error": max(
                        item["background"]["cancellation_relative_scale_error"]
                        for item in records if item["time"] <= PRIMARY_END + 1e-12
                    ),
                }
                for window_name, summary in windows.items():
                    for rate_name in RATE_NAMES:
                        term_window_stage(
                            index, grid, branch, stride, rate_name,
                            window_name, summary,
                        )

    confirmation_gate = bool(all(
        item["maximum_profile_relative_scale_difference"] < 0.002
        and item["area_relative_scale_difference"] < 0.002
        and item["charge_relative_scale_difference"] < 0.002
        for item in confirmations
    ))
    leaf_gate = bool(all(
        record["finite"] for grid in GRIDS for branch in BRANCHES
        for record in surface_records[grid][branch]
        if record["time"] <= PRIMARY_END + 1e-12
    ))
    separation_gate = bool(all(
        surface_records[grid]["inner"][index_value]["surface"]["rho_brane"]
        < surface_records[grid]["outer"][index_value]["surface"]["rho_brane"]
        and surface_records[grid]["inner"][index_value]["surface"]["rho_axis"]
        < surface_records[grid]["outer"][index_value]["surface"]["rho_axis"]
        for grid in GRIDS for index_value, current_time in enumerate(times)
        if current_time <= PRIMARY_END + 1e-12
    ))
    primary_closure_gate = bool(all(
        summaries[grid][branch]["1"]["windows"]["primary"][
            "normalized_absolute_residual"
        ] < 0.05 for grid in GRIDS for branch in BRANCHES
    ))
    subwindow_gate = bool(all(
        summaries[grid][branch]["1"]["windows"][name][
            "normalized_absolute_residual"
        ] < 0.10
        for grid in GRIDS for branch in BRANCHES
        for name in ("sub_001_002", "sub_002_003", "sub_003_004")
    ))
    area_gate = bool(all(
        summaries[grid][branch]["1"][
            "maximum_area_transport_relative_scale_error"
        ] < 0.02 for grid in GRIDS for branch in BRANCHES
    ))
    seam_gate = bool(all(
        summaries[grid][branch]["1"][
            "maximum_israel_seam_relative_scale_error"
        ] < 0.01 for grid in GRIDS for branch in BRANCHES
    ))
    temporal_gate = bool(all(
        abs(
            summaries[grid][branch]["1"]["windows"]["primary"][
                "closure_residual"
            ]
            - summaries[grid][branch]["2"]["windows"]["primary"][
                "closure_residual"
            ]
        ) / summaries[grid][branch]["1"]["windows"]["primary"][
            "balance_norm"
        ] < 0.05
        and summaries[grid][branch]["1"]["windows"]["primary"][
            "normalized_absolute_residual"
        ] <= summaries[grid][branch]["4"]["windows"]["primary"][
            "normalized_absolute_residual"
        ] + 0.02
        for grid in GRIDS for branch in BRANCHES
    ))
    grid_residual_gate = bool(all(
        summaries["G8"][branch]["1"]["windows"]["primary"][
            "normalized_absolute_residual"
        ] <= summaries["G7"][branch]["1"]["windows"]["primary"][
            "normalized_absolute_residual"
        ]
        or max(
            summaries[grid][branch]["1"]["windows"]["primary"][
                "normalized_absolute_residual"
            ] for grid in GRIDS
        ) < 0.02
        for branch in BRANCHES
    ))
    cross_grid_charge_gate = bool(all(
        relative_scale_error(
            balance_records["G7"][branch][1][index_value]["charge"][
                "generalized_hawking_ads_charge_kappa5_squared_E"
            ],
            balance_records["G8"][branch][1][index_value]["charge"][
                "generalized_hawking_ads_charge_kappa5_squared_E"
            ],
            balance_records["G8"][branch][1][index_value]["charge"][
                "generalized_hawking_ads_charge_kappa5_squared_E"
            ],
        ) < 0.02
        for branch in BRANCHES for index_value, current_time in enumerate(times)
        if current_time <= PRIMARY_END + 1e-12
    ))
    term_grid_transfer = []
    for branch in BRANCHES:
        left = summaries["G7"][branch]["1"]["windows"]["primary"]
        right = summaries["G8"][branch]["1"]["windows"]["primary"]
        norm = max(left["balance_norm"], right["balance_norm"], 1e-300)
        for name in RATE_NAMES:
            left_value = left["integrated_named_rates"][name]
            right_value = right["integrated_named_rates"][name]
            relative = abs(left_value - right_value) / max(
                abs(left_value), abs(right_value), 1e-300,
            )
            absolute_normalized = abs(left_value - right_value) / norm
            term_grid_transfer.append({
                "branch": branch, "term": name, "G7": left_value,
                "G8": right_value, "relative_difference": relative,
                "absolute_difference_normalized_by_balance": absolute_normalized,
                "passed": bool(relative < 0.15 or absolute_normalized < 0.02),
            })
    term_grid_gate = bool(all(item["passed"] for item in term_grid_transfer))
    term_completeness_gate = bool(all(
        item["finite"] and item["geometry"]["all_theta_n_negative"]
        and item["background"]["cancellation_relative_scale_error"] < 1e-10
        for grid in GRIDS for branch in BRANCHES
        for item in balance_records[grid][branch][1]
        if item["time"] <= PRIMARY_END + 1e-12
    ))
    gates = {
        "controls": bool(controls["passed"] and recovery["passed"]),
        "primary_leaves": leaf_gate,
        "branch_separation": separation_gate,
        "confirmation_below_0_2_percent": confirmation_gate,
        "cross_grid_charge_below_2_percent": cross_grid_charge_gate,
        "archived_field_velocity_below_2_percent": bool(all(
            item["below_2_percent"] for item in velocity_checks.values()
        )),
        "area_transport_below_2_percent": area_gate,
        "israel_seam_below_1_percent": seam_gate,
        "all_terms_finite_future_and_background_cancelled": term_completeness_gate,
        "primary_cumulative_closure_below_5_percent": primary_closure_gate,
        "subwindow_closure_below_10_percent": subwindow_gate,
        "temporal_convergence": temporal_gate,
        "grid_residual_convergence_or_plateau": grid_residual_gate,
        "named_term_cross_grid_transfer": term_grid_gate,
    }
    numerical_pass = bool(all(gates.values()))
    residuals = [
        summaries[grid][branch]["1"]["windows"]["primary"][
            "normalized_absolute_residual"
        ] for grid in GRIDS for branch in BRANCHES
    ]
    if numerical_pass:
        subgrade = "PASS"
        overall = "REVIEW"
        classification = (
            "numerically_closed_balance_conditional_on_reviewed_evolution"
        )
    elif max(residuals) > 0.25 and not grid_residual_gate:
        subgrade = "FAIL"
        overall = "FAIL"
        classification = "unclosed_quasilocal_balance"
    else:
        subgrade = "REVIEW"
        overall = "REVIEW"
        classification = "complete_balance_terms_without_sealed_closure_pass"

    payload = {
        "status": overall, "overall_grade": overall,
        "tube_balance_subgrade": subgrade, "classification": classification,
        "protocol": str(PROTOCOL), "protocol_sha256": sha256_file(PROTOCOL),
        "inputs": expected, "controls": controls,
        "recovery_control": recovery, "background": background,
        "times": times.tolist(), "primary_end": PRIMARY_END,
        "surface_records": surface_records,
        "confirmation_records": confirmations,
        "balance_records": balance_records,
        "summaries": summaries, "velocity_checks": velocity_checks,
        "term_grid_transfer": term_grid_transfer,
        "superseded_non_authoritative_manifests": [
            "results/corrected_A790_test14b_recovery.json",
            "results/corrected_A790_test14b_recovery_v2.json",
            "results/corrected_A790_test14b_recovery_v3.json",
        ],
        "gates": gates,
        "claim_boundary": (
            "A passing result is a conditional generalized-Hawking-AdS "
            "balance on nested Z2-reflected marginal tubes. It is not a "
            "mass-transfer law, event-horizon result, topology change, throat, "
            "NFW halo, dark-matter, singularity, or inter-universe claim."
        ),
    }
    atomic_write_json(OUTPUT, _jsonable(payload))
    print(json.dumps({
        "status": overall, "subgrade": subgrade, "classification": classification,
        "gates": gates,
        "primary_normalized_residuals": {
            grid: {
                branch: summaries[grid][branch]["1"]["windows"]["primary"][
                    "normalized_absolute_residual"
                ] for branch in BRANCHES
            } for grid in GRIDS
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
