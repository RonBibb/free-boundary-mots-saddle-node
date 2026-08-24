#!/usr/bin/env python3
"""Read-only endpoint replay for the direct Israel-preservation diagnostic.

This runner performs no constraint solve, evolution, or surface solve.  It
uses the three archived G8/R10 endpoints, accepted G9/G10/G10H Test-10E
states, and the four saved G9/G10 cap endpoint radii.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.gw_background import solve_gw_background
from bhps.junction_preservation_diagnostic import (
    WALLS,
    cap_spline_sphere_residual,
    compare_wall_records,
    directional_derivative_ladder,
    high_precision_local_directional_ladder,
    interpolate_native_sphere,
    manufactured_controls,
    summarize_wall,
    wall_junction_rows,
)
from bhps.recovery_indexer import atomic_write_json, sha256_file
from bhps.test14b_balance_closure import five_point_history_derivative
from bhps.test14c_coupled_seam import seam_endpoint_transport


PROTOCOL = Path("notes/113_A790_direct_junction_preservation_protocol.md")
CORE = Path("src/bhps/junction_preservation_diagnostic.py")
TEST10B_STATE = Path("results/corrected_A790_test10b_domain_normalized_state.npz")
LONG_STATE = Path("results/corrected_A790_t008_long_evolution_state.npz")
TEST14B = Path("results/corrected_A790_test14b_balance_closure.json")
TEST14C = Path("results/corrected_A790_test14c_coupled_seam.json")
TEST10E_BUILDER = Path(
    "run_corrected_A790_test10e_genuine_high_z_boundary_resolution.py"
)
TEST10E_ROOT = Path(
    "results/corrected_A790_test10e_genuine_high_z_boundary_resolution_recovery"
)
OUTPUT = Path("results/corrected_A790_direct_junction_endpoint_archive.json")
PRIMARY_DT = 0.000125
HALF_DT = 0.0000625
CAP_STEPS = (10, 12, 14, 16)


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


def _background(z):
    result = solve_gw_background(
        z, epsilon=0.1, backreaction=0.01, wall_stiffness=20.0,
    )
    if not result["converged"]:
        raise RuntimeError(f"fixed background reconstruction failed: {result['message']}")
    return result


def _test10e_chunk(label, start, end):
    return TEST10E_ROOT / f"physical_{label}_steps_{start + 1:03d}_{end:03d}.npz"


def load_test10e_history(label, steps, dt):
    """Recover absolute accepted states without rebuilding the initial slice."""
    positions = []
    velocities = []
    initials = []
    paths = []
    for start in range(0, int(steps), 4):
        end = min(start + 4, int(steps))
        path = _test10e_chunk(label, start, end)
        if not path.is_file():
            raise FileNotFoundError(path)
        paths.append(path)
        with np.load(path) as archive:
            initial = (
                np.asarray(archive["end_position"])
                - np.asarray(archive[f"step_{end:03d}_increment"])
            )
            initials.append(initial)
            for step in range(start + 1, end + 1):
                positions.append(
                    initial + np.asarray(archive[f"step_{step:03d}_increment"])
                )
                velocities.append(np.asarray(archive[f"step_{step:03d}_velocity"]))
    consistency = max(
        float(np.max(np.abs(item - initials[0]))) for item in initials
    )
    if consistency > 1e-15:
        raise RuntimeError(f"{label} reconstructed initial states disagree: {consistency}")
    positions.insert(0, initials[0])
    velocities.insert(0, np.zeros_like(initials[0]))
    shape = positions[0].shape
    z = np.linspace(1.0, math.e, shape[0])
    r = np.linspace(0.0, 10.0, shape[1])
    return {
        "label": label,
        "times": np.arange(int(steps) + 1, dtype=float) * float(dt),
        "position": positions,
        "velocity": velocities,
        "z": z,
        "r": r,
        "initial_reconstruction_maximum_difference": consistency,
        "paths": paths,
        "grid_provenance": (
            "canonical uniform [1,e]x[0,10] reconstructed from the sealed "
            "Test-10E builder because physical chunks omit coordinate arrays"
        ),
    }


def load_g8_endpoints():
    if not TEST10B_STATE.is_file():
        raise FileNotFoundError(TEST10B_STATE)
    with np.load(TEST10B_STATE) as archive:
        prefix = "G8_R10_"
        positions = [
            np.asarray(archive[prefix + "initial"]),
            np.asarray(archive[prefix + "step_008_position"]),
            np.asarray(archive[prefix + "step_016_position"]),
        ]
        velocities = [
            np.zeros_like(positions[0]),
            np.asarray(archive[prefix + "step_008_velocity"]),
            np.asarray(archive[prefix + "step_016_velocity"]),
        ]
        return {
            "label": "G8_R10",
            "times": np.asarray((0.0, 0.001, 0.002)),
            "position": positions,
            "velocity": velocities,
            "z": np.asarray(archive[prefix + "z"]),
            "r": np.asarray(archive[prefix + "r"]),
            "paths": [TEST10B_STATE],
            "grid_provenance": "coordinate arrays stored directly in Test-10B state",
        }


def load_g8_dense_test14_history():
    if not LONG_STATE.is_file():
        raise FileNotFoundError(LONG_STATE)
    with np.load(LONG_STATE) as archive:
        return {
            "label": "G8_R8_dense_Test14",
            "times": np.asarray(archive["times"]),
            "position": np.asarray(archive["G8_position_history"]),
            "velocity": np.asarray(archive["G8_velocity_history"]),
            "z": np.asarray(archive["G8_z"]),
            "r": np.asarray(archive["G8_r"]),
            "paths": [LONG_STATE],
            "grid_provenance": "coordinate arrays stored in the dense Test-14 archive",
        }


def state_records(history, selected_indices):
    background = _background(history["z"])
    records = {}
    for index in selected_indices:
        time = float(history["times"][index])
        walls = {}
        for wall in WALLS:
            record = wall_junction_rows(
                history["position"][index], history["velocity"][index],
                history["z"], history["r"], background, wall,
            )
            walls[wall] = {
                "raw": record,
                "summary": summarize_wall(record, history["r"]),
            }
        records[f"{time:.7f}"] = {"time": time, "walls": walls}
    return records, background


def public_state_records(records):
    return {
        time: {
            "time": item["time"],
            "walls": {
                wall: item["walls"][wall]["summary"] for wall in WALLS
            },
        } for time, item in records.items()
    }


def _history_derivative(values, times):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    count = len(times)
    if count < 3 or values.shape[0] != count:
        raise ValueError("history derivative requires at least three aligned samples")
    result = np.empty_like(values)
    for index in range(count):
        shifted = times - times[index]
        matrix = np.vstack([shifted**power for power in range(count)])
        right = np.zeros(count)
        right[1] = 1.0
        weights = np.linalg.solve(matrix, right)
        result[index] = np.tensordot(weights, values, axes=(0, 0))
    return result


def cap_route(history, background, grid):
    records = {branch: [] for branch in ("inner", "outer")}
    for step in CAP_STEPS:
        geometry_path = TEST10E_ROOT / f"geometry_{grid}_R10_step_{step:03d}.json"
        if not geometry_path.is_file():
            raise FileNotFoundError(geometry_path)
        geometry = json.loads(geometry_path.read_text())
        native = wall_junction_rows(
            history["position"][step], history["velocity"][step],
            history["z"], history["r"], background, "upper",
        )
        for branch in ("inner", "outer"):
            branch_record = next(
                item for item in geometry["branches"] if item["branch"] == branch
            )
            radius = float(branch_record["surface"]["rho_brane"])
            cap = cap_spline_sphere_residual(
                history["position"][step], history["z"], history["r"],
                radius, background, "upper",
            )
            records[branch].append({
                "step": step,
                "time": float(history["times"][step]),
                "radius": radius,
                "cap": cap,
                "native_record": native,
            })
    public = {}
    for branch, items in records.items():
        times = np.asarray([item["time"] for item in items])
        radii = np.asarray([item["radius"] for item in items])
        radius_rate = _history_derivative(radii, times)
        cap_residual = np.asarray([item["cap"]["residual"] for item in items])
        cap_rate = _history_derivative(cap_residual, times)
        native = [
            interpolate_native_sphere(
                item["native_record"], history["r"], item["radius"], radius_rate[index],
            ) for index, item in enumerate(items)
        ]
        native_residual = np.asarray([item["residual"] for item in native])
        native_material_rate = np.asarray([item["material_DX_residual"] for item in native])
        gap = cap_residual - native_residual
        rate_gap = cap_rate - native_material_rate
        public[branch] = {
            "derivative_cadence": "four samples; one global cubic differentiator",
            "records": [{
                "step": item["step"], "time": item["time"], "radius": item["radius"],
                "radius_rate": float(radius_rate[index]),
                "cap_residual": float(cap_residual[index]),
                "native_residual": float(native_residual[index]),
                "cap_minus_native": float(gap[index]),
                "cap_time_difference_rate": float(cap_rate[index]),
                "native_fixed_grid_DX_rate": native[index][
                    "fixed_grid_DX_residual"
                ],
                "native_radial_derivative": native[index]["radial_derivative"],
                "native_radial_motion_term": float(
                    radius_rate[index] * native[index]["radial_derivative"]
                ),
                "native_material_DX_rate": float(native_material_rate[index]),
                "rate_gap": float(rate_gap[index]),
                "native_pchip_minus_cubic": native[index]["pchip_minus_cubic"],
            } for index, item in enumerate(items)],
            "maximum_absolute_cap_residual": float(np.max(np.abs(cap_residual))),
            "maximum_absolute_native_residual": float(np.max(np.abs(native_residual))),
            "maximum_absolute_route_gap": float(np.max(np.abs(gap))),
            "maximum_absolute_rate_gap": float(np.max(np.abs(rate_gap))),
        }
    return public


def dense_g8_test14_route(history, background):
    """Replay the exact dense Test-14 cap-rate path and compare it to native J."""
    test14b = json.loads(TEST14B.read_text())
    test14c = json.loads(TEST14C.read_text())
    times = np.asarray(test14b["times"], dtype=float)
    archive_times = np.asarray(history["times"], dtype=float)
    if len(archive_times) < len(times) or not np.array_equal(
        archive_times[-len(times):], times,
    ):
        raise RuntimeError("dense G8 state history does not align exactly with Test-14B")
    positions = history["position"][-len(times):]
    velocities = history["velocity"][-len(times):]
    z = history["z"]
    r = history["r"]
    native_wall = [
        wall_junction_rows(q, v, z, r, background, "upper")
        for q, v in zip(positions, velocities)
    ]
    output = {}
    reproduction = {
        "archive_time_maximum_absolute_difference": 0.0,
        "cap_geometric_coefficient_maximum_absolute_difference": 0.0,
        "cap_residual_maximum_absolute_difference": 0.0,
        "stored_geometric_rate_maximum_absolute_difference": 0.0,
        "stored_wall_rate_maximum_absolute_difference": 0.0,
    }
    primary = times <= 0.004 + 1e-15
    for branch in ("inner", "outer"):
        surfaces = test14b["surface_records"]["G8"][branch]
        base = test14b["balance_records"]["G8"][branch]["1"]
        stored_rate = test14c["physical_records"]["G8"][branch]["1"]
        rho_history = np.stack([
            np.asarray(item["surface"]["rho"], dtype=float) for item in surfaces
        ])
        rho_rate = five_point_history_derivative(rho_history, times, stride=1)
        geometric = np.asarray([
            item["seam"]["geometric_Ws_over_W"] for item in base
        ], dtype=float)
        geometric_rate = five_point_history_derivative(
            geometric, times, stride=1,
        )
        records = []
        for index, time in enumerate(times):
            profile = surfaces[index]["surface"]
            radius = float(profile["rho_brane"])
            radius_rate = float(rho_rate[index, -1])
            cap = cap_spline_sphere_residual(
                positions[index], z, r, radius, background, "upper",
            )
            endpoint = seam_endpoint_transport(
                positions[index], velocities[index], z, r, profile,
                rho_rate[index], test14b["background"]["wall_stiffness"],
                test14b["background"]["v1"],
            )
            wall_rate = float(endpoint["wall_israel_coefficient_rate"])
            residual_rate = float(geometric_rate[index] - wall_rate)
            native = interpolate_native_sphere(
                native_wall[index], r, radius, radius_rate,
            )
            motion = float(radius_rate * native["radial_derivative"])
            stored_wall = float(
                base[index]["seam"]["wall_energy_bare_plus_potential"] / 6.0
            )
            stored_residual = float(geometric[index] - stored_wall)
            reproduction["cap_geometric_coefficient_maximum_absolute_difference"] = max(
                reproduction["cap_geometric_coefficient_maximum_absolute_difference"],
                abs(cap["geometric_sphere_coefficient"] - geometric[index]),
            )
            reproduction["cap_residual_maximum_absolute_difference"] = max(
                reproduction["cap_residual_maximum_absolute_difference"],
                abs(cap["residual"] - stored_residual),
            )
            reproduction["stored_geometric_rate_maximum_absolute_difference"] = max(
                reproduction["stored_geometric_rate_maximum_absolute_difference"],
                abs(geometric_rate[index] - stored_rate[index]["geometric_israel_rate"]),
            )
            reproduction["stored_wall_rate_maximum_absolute_difference"] = max(
                reproduction["stored_wall_rate_maximum_absolute_difference"],
                abs(wall_rate - stored_rate[index]["wall_israel_rate"]),
            )
            records.append({
                "time": float(time),
                "radius": radius,
                "radius_rate": radius_rate,
                "cap_residual": float(cap["residual"]),
                "stored_cap_residual": stored_residual,
                "cap_residual_rate_geometry_minus_wall": residual_rate,
                "native_residual": native["residual"],
                "native_fixed_grid_DX_residual": native["fixed_grid_DX_residual"],
                "native_radial_derivative": native["radial_derivative"],
                "native_radial_motion_term": motion,
                "native_material_DX_residual": native["material_DX_residual"],
                "cap_minus_native_residual": float(
                    cap["residual"] - native["residual"]
                ),
                "cap_rate_minus_native_material_rate": float(
                    residual_rate - native["material_DX_residual"]
                ),
            })

        def primary_maximum(key):
            values = np.asarray([item[key] for item in records], dtype=float)
            return float(np.max(np.abs(values[primary])))

        output[branch] = {
            "derivative_cadence": (
                "exact Test-14 five-point stride-1 history derivative; 60 samples"
            ),
            "records": records,
            "primary_t_le_0p004": {
                "maximum_absolute_cap_residual": primary_maximum("cap_residual"),
                "maximum_absolute_native_residual": primary_maximum("native_residual"),
                "maximum_absolute_cap_residual_rate": primary_maximum(
                    "cap_residual_rate_geometry_minus_wall"
                ),
                "maximum_absolute_native_fixed_grid_DX_residual": primary_maximum(
                    "native_fixed_grid_DX_residual"
                ),
                "maximum_absolute_native_radial_motion_term": primary_maximum(
                    "native_radial_motion_term"
                ),
                "maximum_absolute_native_material_DX_residual": primary_maximum(
                    "native_material_DX_residual"
                ),
                "maximum_absolute_cap_rate_minus_native_material_rate": primary_maximum(
                    "cap_rate_minus_native_material_rate"
                ),
                "maximum_absolute_radius_rate": primary_maximum("radius_rate"),
            },
        }
    reproduction["passed_below_1e_11"] = bool(
        max(reproduction.values()) < 1e-11
    )
    return {
        "scope": (
            "exact dense G8/R8 replay of the failing Test-14 cap coefficient and "
            "rate path; native material comparator is a scalar transport, not a "
            "full covariant horizon-generator derivative"
        ),
        "reproduction": reproduction,
        "branches": output,
    }


def descriptive_grid_behavior(source_records):
    intervals = np.asarray((96.0, 112.0, 128.0))
    output = {}
    for time in ("0.0000000", "0.0010000", "0.0020000"):
        output[time] = {}
        for wall in WALLS:
            values = []
            tangent = []
            for grid in ("G8_R10", "G9_R10", "G10_R10"):
                zone = source_records[grid][time]["walls"][wall]["summary"]["zones"]["interior"]
                values.append(zone["proper_statistics"]["J"]["proper_RMS"])
                tangent.append(zone["proper_statistics"]["DXJ"]["proper_RMS"])
            def slope(array):
                array = np.asarray(array)
                if np.any(array <= 1e-250):
                    return None
                return float(-np.polyfit(np.log(intervals), np.log(array), 1)[0])
            output[time][wall] = {
                "J_interior_proper_RMS": dict(zip(("G8", "G9", "G10"), values)),
                "DXJ_interior_proper_RMS": dict(zip(("G8", "G9", "G10"), tangent)),
                "descriptive_J_decay_exponent": slope(values),
                "descriptive_DXJ_decay_exponent": slope(tangent),
                "warning": "distinct constraint parents; exponent is not a Richardson order",
            }
    return output


def timestep_comparison(primary, half):
    zones = ("axis", "interior", "outer_corner")
    maximum = {
        wall: {
            zone: {
                "J_orthonormal_Linf": 0.0,
                "DXJ_orthonormal_Linf": 0.0,
                "J_proper_RMS": 0.0,
                "DXJ_proper_RMS": 0.0,
            } for zone in zones
        } for wall in WALLS
    }
    coordinate_secondary = {wall: {"J": 0.0, "DXJ": 0.0} for wall in WALLS}
    by_time = []
    background = _background(primary["z"])
    for step in range(17):
        item = {"time": float(primary["times"][step]), "walls": {}}
        for wall in WALLS:
            left = wall_junction_rows(
                primary["position"][step], primary["velocity"][step],
                primary["z"], primary["r"], background, wall,
            )
            right = wall_junction_rows(
                half["position"][2 * step], half["velocity"][2 * step],
                half["z"], half["r"], background, wall,
            )
            comparison = compare_wall_records(left, right, primary["r"])
            j = float(np.max(np.abs(left["J_tensor"] - right["J_tensor"])))
            tangent = float(np.max(np.abs(left["DXJ_tensor"] - right["DXJ_tensor"])))
            coordinate_secondary[wall]["J"] = max(
                coordinate_secondary[wall]["J"], j,
            )
            coordinate_secondary[wall]["DXJ"] = max(
                coordinate_secondary[wall]["DXJ"], tangent,
            )
            for zone in zones:
                local = comparison["zones"][zone]
                maximum[wall][zone]["J_orthonormal_Linf"] = max(
                    maximum[wall][zone]["J_orthonormal_Linf"],
                    local["J"]["Linf"],
                )
                maximum[wall][zone]["DXJ_orthonormal_Linf"] = max(
                    maximum[wall][zone]["DXJ_orthonormal_Linf"],
                    local["DXJ"]["Linf"],
                )
                maximum[wall][zone]["J_proper_RMS"] = max(
                    maximum[wall][zone]["J_proper_RMS"],
                    local["J"]["proper_RMS"],
                )
                maximum[wall][zone]["DXJ_proper_RMS"] = max(
                    maximum[wall][zone]["DXJ_proper_RMS"],
                    local["DXJ"]["proper_RMS"],
                )
            item["walls"][wall] = comparison
        by_time.append(item)
    return {
        "comparison": "G10 dt versus G10H dt/2 on the identical spatial grid",
        "maximum_zoned_primary_orthonormal_frame": maximum,
        "secondary_coordinate_component_maximum": coordinate_secondary,
        "by_time": by_time,
        "interpretation": (
            "zoned tensor differences use the primary-dt orthonormal frame; "
            "two-step consistency only, because dt/4 is required for temporal order"
        ),
    }


def main():
    required = (
        PROTOCOL, CORE, TEST10B_STATE, LONG_STATE, TEST14B, TEST14C,
        TEST10E_BUILDER,
        Path("src/bhps/gw_background.py"),
        Path("src/bhps/gw_slice_high_order_solver.py"),
        Path("src/bhps/nonlinear_regular_so3_evolution.py"),
        Path("src/bhps/test14b_balance_closure.py"),
        Path("src/bhps/test14c_coupled_seam.py"),
        Path(__file__),
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing junction diagnostic inputs: {missing}")
    controls = manufactured_controls()
    if not controls["passed"]:
        raise RuntimeError("junction diagnostic controls failed")

    histories = {
        "G8_R10": load_g8_endpoints(),
        "G9_R10": load_test10e_history("G9_R10", 16, PRIMARY_DT),
        "G10_R10": load_test10e_history("G10_R10", 16, PRIMARY_DT),
        "G10H_R10": load_test10e_history("G10H_R10", 32, HALF_DT),
    }
    dense_g8 = load_g8_dense_test14_history()
    selected = {
        "G8_R10": (0, 1, 2),
        "G9_R10": (0, 8, 16),
        "G10_R10": (0, 8, 16),
        "G10H_R10": (0, 16, 32),
    }
    raw_records = {}
    backgrounds = {}
    public_records = {}
    for label, history in histories.items():
        raw_records[label], backgrounds[label] = state_records(history, selected[label])
        public_records[label] = public_state_records(raw_records[label])

    float64_derivative_history = {}
    high_precision_derivative_controls = {}
    for label in ("G8_R10", "G9_R10", "G10_R10"):
        index = selected[label][1]
        float64_derivative_history[label] = directional_derivative_ladder(
            histories[label]["position"][index], histories[label]["velocity"][index],
            histories[label]["z"], histories[label]["r"], backgrounds[label],
        )
        high_precision_derivative_controls[label] = (
            high_precision_local_directional_ladder(
                histories[label]["position"][index],
                histories[label]["velocity"][index],
                histories[label]["z"], histories[label]["r"],
                backgrounds[label], decimal_digits=80,
            )
        )

    cap = {
        grid: cap_route(histories[f"{grid}_R10"], backgrounds[f"{grid}_R10"], grid)
        for grid in ("G9", "G10")
    }
    dense_cap = dense_g8_test14_route(dense_g8, _background(dense_g8["z"]))
    timestep = timestep_comparison(histories["G10_R10"], histories["G10H_R10"])
    source_grid = descriptive_grid_behavior(raw_records)
    gauge_valid = bool(all(
        item["walls"][wall]["summary"]["full_J_tensor_interpretation_valid"]
        for grid in raw_records.values() for item in grid.values() for wall in WALLS
    ))
    derivative_valid = bool(all(
        item["finite"] and item["adjacent_accurate_pair"]
        and item["second_order_pair_before_roundoff"]
        and item["best_maximum_relative_scale_error"] < 1e-7
        and item["core_vs_high_precision_analytic_maximum_relative"] < 1e-9
        for item in high_precision_derivative_controls.values()
    ))
    dense_cap_valid = bool(dense_cap["reproduction"]["passed_below_1e_11"])
    inputs = {str(path): sha256_file(path) for path in required}
    for history in histories.values():
        for path in history["paths"]:
            inputs.setdefault(str(path), sha256_file(path))
    for path in dense_g8["paths"]:
        inputs.setdefault(str(path), sha256_file(path))
    for grid in ("G9", "G10"):
        for step in CAP_STEPS:
            path = TEST10E_ROOT / f"geometry_{grid}_R10_step_{step:03d}.json"
            inputs[str(path)] = sha256_file(path)

    payload = {
        "schema": "bhps-direct-junction-endpoint-archive-v1",
        "status": "complete",
        "classification": "preliminary_archive_diagnostic",
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "inputs": inputs,
        "controls": controls,
        "physical_high_precision_directional_controls": (
            high_precision_derivative_controls
        ),
        "diagnostic_history": {
            "ordinary_float64_physical_directional_ladder": {
                "outcome": "cancellation_limited_audit_method_not_a_physics_failure",
                "reason": (
                    "the physical velocity changes a nonzero baseline J below the "
                    "float64 centered-subtraction floor; the ladders retain clean "
                    "second-order truncation before an approximately 1e-10 absolute floor"
                ),
                "validity_role": (
                    "historical secondary evidence only; manufactured full-state and "
                    "high-precision factored lanes validate the analytic derivative"
                ),
                "records": float64_derivative_history,
            },
            "abandoned_rescaling_attempt": {
                "outcome": "not_used",
                "reason": (
                    "multiplying the direction only shifts the effective epsilon and "
                    "cannot improve the truncation-roundoff optimum after rescaling back"
                ),
            },
        },
        "archive_reconstruction": {
            label: {
                key: value for key, value in history.items()
                if key in ("grid_provenance", "initial_reconstruction_maximum_difference")
            } for label, history in histories.items()
        },
        "endpoint_records": public_records,
        "descriptive_source_grid_behavior": source_grid,
        "G10_timestep_consistency": timestep,
        "cap_route_low_cadence": cap,
        "cap_route_dense_G8_exact_Test14": dense_cap,
        "implementation_order_hypotheses": {
            "outer_corner": (
                "compact-wall acceleration rows are solved before the open outer "
                "face is overwritten; the wall-corner Dz stencil samples changed "
                "neighboring face nodes"
            ),
            "upper_axis": (
                "the generic regular-axis fill follows compact-wall endpoint solves "
                "and restores only the separately owned g_zz axis datum"
            ),
            "claim_status": (
                "source-level mechanisms consistent with localized DXJ; direct "
                "before/after D_X^2J instrumentation is still required"
            ),
        },
        "validity": {
            "manufactured_controls_pass": controls["passed"],
            "wall_adapted_gauge_pass_at_reported_endpoints": gauge_valid,
            "physical_high_precision_directional_controls_pass": derivative_valid,
            "dense_G8_Test14_route_reproduction_pass": dense_cap_valid,
            "ordinary_float64_physical_ladder": "cancellation_limited_not_a_gate",
        },
        "claim_boundary": (
            "read-only discrete mechanism evidence; no matched G8/G9/G10 x "
            "dt/dt2/dt4 sequence and no continuum junction claim"
        ),
        "required_new_runs": [
            "matched G8/G9/G10 on one domain and constraint family at dt, dt/2, and dt/4",
            "new G8H and G9H partners plus dt/4 runs for a temporal order",
            "save q, v, gauge source/memory, and acceleration before boundaries, after wall endpoints, after axis fill, and after the outer row",
            "save full cap profiles at identical cadence for native, cap, and time-difference replay",
            "boundary-stencil variant and large-known-DXc manufactured dynamic wall mode",
        ],
    }
    atomic_write_json(OUTPUT, _jsonable(payload))
    print(json.dumps({
        "status": payload["status"],
        "classification": payload["classification"],
        "output": str(OUTPUT),
        "output_sha256": sha256_file(OUTPUT),
        "validity": payload["validity"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
