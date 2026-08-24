#!/usr/bin/env python3
"""Finite-archive backward null-generator reconstruction pilot."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.dynamical_capped_horizon_bvp import solve_dynamical_capped_surface_bvp
from bhps.finite_archive_null_proxy import (
    ArchivedSpacetime,
    detect_synthetic_caustic,
    finite_terminal_classification,
    generator_strip_diagnostics,
    initialize_terminal_generators,
    integrate_coordinate_generators,
    integrate_hamiltonian_generators,
    polar_coordinates,
    terminal_profile_from_surface,
)
from bhps.recovery_indexer import (
    RecoveryIndex,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
    validate_npz,
)


PROTOCOL = Path("notes/100a_test15a_endpoint_core_checker_erratum.md")
OUTPUT = Path("results/test15a_finite_archive_null_proxy.json")
RECOVERY = Path("results/test15a_recovery_v2")
MANIFEST = RECOVERY / "manifest.json"
R8_ARCHIVE = Path("results/corrected_A790_t008_long_evolution_state.npz")
R8_RESULT = Path("results/corrected_A790_t008_long_evolution.json")
TEST10_STATE = Path("results/corrected_A790_test10_joint_convergence_state.npz")
TEST10_RESULT = Path("results/corrected_A790_test10_joint_convergence.json")
TEST10_RECOVERY = Path("results/corrected_A790_test10_joint_convergence_recovery")
TIMES = np.arange(65, dtype=float) * 0.000125
THETA = np.linspace(1e-4, np.pi / 2.0, 65)
CASES = (("R8", "G7"), ("R8", "G8"), ("R10", "G7"), ("R10", "G8"))
TERMINAL_SPECS = (
    ("mots_T006", 0.006, 1.0),
    ("mots_T007", 0.007, 1.0),
    ("mots_T008", 0.008, 1.0),
    ("offset_minus2_T008", 0.008, 0.98),
    ("offset_plus2_T008", 0.008, 1.02),
)
REFERENCE_TIME = 0.004
SEGMENT_STEPS = 16
PRIMARY_RTOL = 2e-9
PRIMARY_ATOL = 2e-11
STRICT_RTOL = 2e-11
STRICT_ATOL = 2e-13
NULL_LIMIT = 1e-7
STRICT_CELL_LIMIT = 0.05
HAMILTONIAN_CELL_LIMIT = 0.10
ROUNDTRIP_CELL_LIMIT = 0.10
GRID_LIMIT = 0.01
DOMAIN_LIMIT = 0.03
TERMINAL_TIME_LIMIT = 0.03
OFFSET_ENVELOPE_LIMIT = 0.06
OFFSET_AMPLIFICATION_LIMIT = 1.5
BRACKET_FRACTION = 0.90


def builtin(value):
    if isinstance(value, dict):
        return {str(key): builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def expected_inputs():
    paths = (
        R8_ARCHIVE, R8_RESULT, TEST10_STATE, TEST10_RESULT,
        Path("notes/85_A790_t008_long_evolution_protocol.md"),
        Path("notes/86_A790_gauge_slicing_variation_protocol.md"),
        Path("notes/88_A790_R12_domain_sequence_protocol.md"),
        Path("notes/93_A790_publication_joint_convergence_protocol.md"),
        Path("notes/93_A790_publication_joint_convergence_result.md"),
        Path("src/bhps/finite_archive_null_proxy.py"),
        Path("src/bhps/dynamical_capped_horizon.py"),
        Path("src/bhps/dynamical_capped_horizon_bvp.py"),
        Path("src/bhps/recovery_indexer.py"), Path(__file__),
    )
    paths += tuple(sorted(TEST10_RECOVERY.glob("evolution_R10G*_long_steps_*.npz")))
    return {str(path): sha256_file(path) for path in paths}


def run_json_stage(index, stage_id, seconds, metadata, function):
    path = RECOVERY / f"{stage_id.replace('/', '__')}.json"
    index.register(stage_id, "json", seconds, metadata)
    cached = index.validated_path(stage_id)
    if cached is not None:
        return json.loads(cached.read_text())
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        payload = builtin(function())
        atomic_write_json(path, payload)
        index.mark_complete(stage_id, path, time.perf_counter() - started)
        return payload
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def run_npz_stage(index, stage_id, seconds, metadata, function):
    path = RECOVERY / f"{stage_id.replace('/', '__')}.npz"
    index.register(stage_id, "npz", seconds, metadata)
    cached = index.validated_path(stage_id)
    if cached is not None:
        validate_npz(cached)
        return cached
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        arrays = function()
        atomic_write_npz(path, **arrays)
        validate_npz(path)
        index.mark_complete(
            stage_id, path, time.perf_counter() - started,
            {"array_count": len(arrays)},
        )
        return path
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def load_r8(grid):
    with np.load(R8_ARCHIVE, allow_pickle=False) as archive:
        times = np.asarray(archive["times"])
        if not np.array_equal(times, TIMES):
            raise RuntimeError("R8 archive cadence mismatch")
        return {
            "times": times,
            "z": np.asarray(archive[f"{grid}_z"]),
            "r": np.asarray(archive[f"{grid}_r"]),
            "position": np.asarray(archive[f"{grid}_position_history"]),
            "velocity": np.asarray(archive[f"{grid}_velocity_history"]),
        }


def load_r10(grid):
    with np.load(TEST10_STATE, allow_pickle=False) as state:
        z = np.asarray(state[f"R10{grid}_z"])
        r = np.asarray(state[f"R10{grid}_r"])
        final_reference = np.asarray(state[f"R10{grid}_long_step_064_position"])
    files = sorted(TEST10_RECOVERY.glob(f"evolution_R10{grid}_long_steps_*.npz"))
    if len(files) != 8:
        raise RuntimeError(f"expected eight R10{grid} long segments")
    with np.load(files[0], allow_pickle=False) as first:
        initial = np.asarray(first["end_position"]) - np.asarray(first["step_008_increment"])
    position = np.empty((65, len(z), len(r), 9))
    velocity = np.empty_like(position)
    position[0] = initial
    velocity[0] = 0.0
    seen = set()
    for path in files:
        with np.load(path, allow_pickle=False) as archive:
            start = int(archive["start_step"])
            end = int(archive["end_step"])
            for step in range(start + 1, end + 1):
                position[step] = initial + np.asarray(archive[f"step_{step:03d}_increment"])
                velocity[step] = np.asarray(archive[f"step_{step:03d}_velocity"])
                seen.add(step)
            if not np.allclose(position[end], archive["end_position"], rtol=0.0, atol=2e-13):
                raise RuntimeError(f"R10{grid} segment endpoint mismatch at {end}")
    if seen != set(range(1, 65)):
        raise RuntimeError(f"R10{grid} dense step coverage incomplete")
    if not np.allclose(position[-1], final_reference, rtol=0.0, atol=2e-13):
        raise RuntimeError(f"R10{grid} final-state crosscheck failed")
    return {"times": TIMES, "z": z, "r": r, "position": position, "velocity": velocity}


def load_history(domain, grid):
    return load_r8(grid) if domain == "R8" else load_r10(grid)


def history_check(history, domain, grid):
    determinants = []
    for level in (0, 32, 48, 56, 64):
        q = history["position"][level]
        radius = history["r"][None, :]
        spatial = q[:, :, 6] * (q[:, :, 3] + radius**2 * q[:, :, 4]) - (radius * q[:, :, 1])**2
        determinants.append(float(np.min(spatial)))
    return {
        "domain": domain,
        "grid": grid,
        "shape": list(history["position"].shape),
        "time_cadence_exact": bool(np.array_equal(history["times"], TIMES)),
        "finite": bool(
            np.all(np.isfinite(history["position"]))
            and np.all(np.isfinite(history["velocity"]))
        ),
        "sampled_minimum_spatial_base_determinants": determinants,
        "sampled_positive_spatial_base": bool(min(determinants) > 0.0),
    }


def recovery_adverse_control():
    with tempfile.TemporaryDirectory(prefix="test15a-recovery-") as directory:
        root = Path(directory)
        protocol = root / "protocol.md"
        source = root / "source.dat"
        protocol.write_text("sealed\n")
        source.write_text("input\n")
        manifest = root / "manifest.json"
        expected = {str(source): sha256_file(source)}
        index = RecoveryIndex(manifest, protocol, expected, maximum_stage_seconds=10.0)
        index.register("ray/R8/G7/candidate/segment", "json", 2.0, {"segment": 0})
        index.mark_running("ray/R8/G7/candidate/segment")
        restarted = RecoveryIndex(manifest, protocol, expected, maximum_stage_seconds=10.0)
        reset = restarted.data["stages"]["ray/R8/G7/candidate/segment"]["status"] == "pending"
        output = root / "stage.json"
        atomic_write_json(output, {"finite": True})
        restarted.mark_complete("ray/R8/G7/candidate/segment", output, 0.1)
        output.write_text("corrupt\n")
        corrupted = restarted.validated_path("ray/R8/G7/candidate/segment") is None
        changed = False
        try:
            restarted.register(
                "ray/R8/G7/candidate/segment", "json", 2.0, {"segment": 1},
            )
        except RuntimeError:
            changed = True
        return {
            "running_resets_pending": bool(reset),
            "corrupt_content_rejected": bool(corrupted),
            "changed_index_metadata_rejected": bool(changed),
        }


def flat_history(times, z, r, shift=0.0):
    position = np.zeros((len(times), len(z), len(r), 9))
    position[:, :, :, 0] = shift
    position[:, :, :, 2] = -1.0 + shift**2
    position[:, :, :, 3] = 1.0
    position[:, :, :, 6] = 1.0
    return position, np.zeros_like(position)


def analytic_controls():
    times = np.linspace(0.0, 0.1, 11)
    z = np.linspace(0.0, 3.0, 49)
    r = np.linspace(0.0, 3.0, 49)
    theta = np.linspace(0.08, np.pi / 2.0 - 0.08, 25)
    profile = {"theta": theta, "rho": np.ones_like(theta), "slope": np.zeros_like(theta)}
    records = {}
    for label, shift in (("flat", 0.0), ("constant_shift", 0.2)):
        position, velocity = flat_history(times, z, r, shift)
        spacetime = ArchivedSpacetime(times, z, r, position, velocity)
        terminal, terminal_velocity = initialize_terminal_generators(
            spacetime, 0.1, profile,
        )
        primary = integrate_coordinate_generators(
            spacetime, 0.1, 0.0, terminal, terminal_velocity,
            output_times=np.asarray((0.1, 0.05, 0.0)),
            rtol=1e-11, atol=1e-13,
        )
        independent = integrate_hamiltonian_generators(
            spacetime, 0.1, 0.0, terminal, terminal_velocity,
            output_times=np.asarray((0.1, 0.05, 0.0)),
            rtol=1e-11, atol=1e-13,
        )
        analytic_final = terminal - 0.1 * terminal_velocity
        records[label] = {
            "analytic_coordinate_error": float(np.max(np.linalg.norm(
                primary["positions"][-1] - analytic_final, axis=1,
            ))),
            "Hamiltonian_coordinate_error": float(np.max(np.linalg.norm(
                independent["positions"] - primary["positions"], axis=2,
            ))),
            "maximum_null_residual": primary["maximum_normalized_null_residual"],
        }
    position, velocity = flat_history(times, z, r)
    position[:, :, :, 2] = 1.0
    rejected = False
    try:
        ArchivedSpacetime(times, z, r, position, velocity).metric_and_derivatives(
            0.05, np.asarray((1.0,)), np.asarray((1.0,)),
        )
    except RuntimeError:
        rejected = True
    recovery = recovery_adverse_control()
    checks = {
        "flat_analytic_ray": records["flat"]["analytic_coordinate_error"] < 1e-8,
        "constant_shift_analytic_ray": records["constant_shift"]["analytic_coordinate_error"] < 1e-8,
        "independent_Hamiltonian_formulation": max(
            item["Hamiltonian_coordinate_error"] for item in records.values()
        ) < 1e-8,
        "null_constraint": max(
            item["maximum_null_residual"] for item in records.values()
        ) < 1e-10,
        "non_Lorentzian_rejected": rejected,
        "synthetic_caustic_detected": detect_synthetic_caustic(
            np.asarray((0.0, 0.4, 0.3, 1.0)),
        ),
        "recovery_adverse_controls": all(recovery.values()),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "analytic_records": records,
        "recovery": recovery,
    }


def surface_public(surface):
    return {
        key: builtin(surface[key]) for key in (
            "converged", "solver_success", "message", "in_domain",
            "iterations", "mesh_nodes_used", "rho_axis", "rho_brane",
            "rho_min", "rho_max", "boundary_slope_error",
            "local_expansion_interior_maximum", "local_expansion_full_maximum",
            "ode_defect_maximum", "primary_evaluator_crosscheck",
            "interior_point_count", "theta", "rho", "slope",
        )
    }


def solve_outer_profile(history, time_value):
    index = int(round(time_value / 0.000125))
    surface = solve_dynamical_capped_surface_bvp(
        history["position"][index], history["velocity"][index],
        history["z"], history["r"], 1.6, tolerance=2e-5,
        nodes=121, maximum_nodes=6000, dense_nodes=501,
    )
    cross = surface.get("primary_evaluator_crosscheck", {})
    admitted = bool(
        surface["converged"]
        and surface["local_expansion_interior_maximum"] < 2e-4
        and surface["boundary_slope_error"] < 2e-4
        and cross.get("two_cell_interior_maximum", np.inf) < 0.002
    )
    if not admitted:
        raise RuntimeError(f"outer MOTS terminal profile failed at t={time_value}")
    return {"time": time_value, "admitted": admitted, "surface": surface_public(surface)}


def terminal_candidate(index, case_id, history, name, time_value, scale, profiles):
    stage_id = f"terminal/{case_id}/{name}"
    metadata = {"name": name, "time": time_value, "scale": scale, "labels": len(THETA)}

    def calculate():
        base_name = f"outer_T{int(round(time_value * 1000)):03d}"
        base = profiles[base_name]["surface"]
        profile = terminal_profile_from_surface(base, THETA, scale=scale)
        spacetime = ArchivedSpacetime(
            history["times"], history["z"], history["r"],
            history["position"], history["velocity"],
        )
        position, velocity = initialize_terminal_generators(
            spacetime, time_value, profile,
        )
        return {
            "name": name, "time": time_value, "scale": scale,
            "profile": profile,
            "position": position,
            "velocity": velocity,
            "minimum_radius_coordinate": float(np.min(position[:, 1])),
            "maximum_radius_coordinate": float(np.max(position[:, 1])),
        }

    return run_json_stage(index, stage_id, 300.0, metadata, calculate)


def segment_ranges(terminal_time):
    end_step = int(round(terminal_time / 0.000125))
    values = []
    high = end_step
    while high > 0:
        low = max(0, high - SEGMENT_STEPS)
        values.append((high, low))
        high = low
    return values


def trace_candidate(index, case_id, spacetime, candidate):
    position = np.asarray(candidate["position"], dtype=float)
    velocity = np.asarray(candidate["velocity"], dtype=float)
    paths = []
    parent = f"terminal/{case_id}/{candidate['name']}"
    for high, low in segment_ranges(candidate["time"]):
        stage_id = f"trace/{case_id}/{candidate['name']}/steps_{high:03d}_{low:03d}"
        output_times = TIMES[low:high + 1][::-1]
        metadata = {
            "candidate": candidate["name"], "high_step": high,
            "low_step": low, "parent_stage": parent,
            "rtol": PRIMARY_RTOL, "atol": PRIMARY_ATOL,
        }
        initial_position = position.copy()
        initial_velocity = velocity.copy()

        def calculate(
            initial_position=initial_position, initial_velocity=initial_velocity,
            output_times=output_times,
        ):
            result = integrate_coordinate_generators(
                spacetime, output_times[0], output_times[-1],
                initial_position, initial_velocity, output_times=output_times,
                rtol=PRIMARY_RTOL, atol=PRIMARY_ATOL,
            )
            return {
                "times": result["times"],
                "positions": result["positions"],
                "velocities": result["velocities"],
                "normalized_null_residual": result["normalized_null_residual"],
                "function_evaluations": np.asarray(result["function_evaluations"]),
            }

        path = run_npz_stage(index, stage_id, 1800.0, metadata, calculate)
        with np.load(path, allow_pickle=False) as archive:
            position = np.asarray(archive["positions"][-1])
            velocity = np.asarray(archive["velocities"][-1])
        paths.append(path)
        parent = stage_id
    times = []
    positions = []
    velocities = []
    residuals = []
    for path_index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as archive:
            start = 0 if path_index == 0 else 1
            times.append(np.asarray(archive["times"])[start:])
            positions.append(np.asarray(archive["positions"])[start:])
            velocities.append(np.asarray(archive["velocities"])[start:])
            residuals.append(np.asarray(archive["normalized_null_residual"])[start:])
    return {
        "times": np.concatenate(times),
        "positions": np.concatenate(positions),
        "velocities": np.concatenate(velocities),
        "normalized_null_residual": np.concatenate(residuals),
        "segment_paths": [str(path) for path in paths],
    }


def time_index(trace, value):
    index = int(np.argmin(np.abs(trace["times"] - float(value))))
    if abs(trace["times"][index] - float(value)) > 1e-10:
        raise RuntimeError(f"trace lacks time {value}")
    return index


def numerical_checks(index, case_id, spacetime, candidate, trace):
    output_times = np.asarray((candidate["time"], REFERENCE_TIME, 0.0))
    stage_id = f"checks/{case_id}/{candidate['name']}"
    metadata = {
        "candidate": candidate["name"], "strict_rtol": STRICT_RTOL,
        "Hamiltonian_rtol": PRIMARY_RTOL, "roundtrip": True,
    }

    def calculate():
        core = slice(2, len(candidate["position"]) - 2)
        terminal_position = np.asarray(candidate["position"])[core]
        terminal_velocity = np.asarray(candidate["velocity"])[core]
        strict = integrate_coordinate_generators(
            spacetime, candidate["time"], 0.0, terminal_position, terminal_velocity,
            output_times=output_times, rtol=STRICT_RTOL, atol=STRICT_ATOL,
        )
        hamiltonian = integrate_hamiltonian_generators(
            spacetime, candidate["time"], 0.0, terminal_position, terminal_velocity,
            output_times=output_times, rtol=PRIMARY_RTOL, atol=PRIMARY_ATOL,
        )
        primary_positions = np.stack([
            trace["positions"][time_index(trace, value)][core]
            for value in output_times
        ])
        cell = min(np.min(np.diff(spacetime.z)), np.min(np.diff(spacetime.r)))
        strict_cells = float(np.max(np.linalg.norm(
            strict["positions"] - primary_positions, axis=2,
        )) / cell)
        hamiltonian_cells = float(np.max(np.linalg.norm(
            hamiltonian["positions"] - primary_positions, axis=2,
        )) / cell)
        zero_position = trace["positions"][-1][core]
        zero_velocity = trace["velocities"][-1][core]
        forward = integrate_coordinate_generators(
            spacetime, 0.0, candidate["time"], zero_position, zero_velocity,
            output_times=np.asarray((0.0, candidate["time"])),
            rtol=PRIMARY_RTOL, atol=PRIMARY_ATOL,
        )
        roundtrip_cells = float(np.max(np.linalg.norm(
            forward["positions"][-1] - terminal_position, axis=1,
        )) / cell)
        return {
            "minimum_native_cell": cell,
            "strict_primary_maximum_cell_difference": strict_cells,
            "Hamiltonian_primary_maximum_cell_difference": hamiltonian_cells,
            "roundtrip_maximum_cell_difference": roundtrip_cells,
            "strict_maximum_null_residual": strict["maximum_normalized_null_residual"],
            "Hamiltonian_maximum_null_residual": hamiltonian["maximum_normalized_null_residual"],
        }

    return run_json_stage(index, stage_id, 1800.0, metadata, calculate)


def finite_bracket_check(index, case_id, spacetime, candidate, trace):
    stage_id = f"bracket/{case_id}/{candidate['name']}"
    metadata = {
        "reference_time": REFERENCE_TIME, "perturbation": 0.005,
        "buffer_fraction": 0.0005, "core_margin": 2,
    }

    def calculate():
        reference_index = time_index(trace, REFERENCE_TIME)
        reference_position = trace["positions"][reference_index]
        angle, rho = polar_coordinates(spacetime.z[-1], reference_position)
        if np.any(np.diff(angle) <= 0.0):
            raise RuntimeError("cannot construct bracket after ordering loss")
        base = CubicSpline(angle, rho)
        terminal_profile = candidate["profile"]
        records = {}
        core = slice(2, len(angle) - 2)
        for label, scale, expected in (
            ("inward", 0.995, "inside"), ("outward", 1.005, "outside"),
        ):
            local_rho = scale * base(angle)
            local_slope = scale * base(angle, 1)
            profile = {"theta": angle, "rho": local_rho, "slope": local_slope}
            position, velocity = initialize_terminal_generators(
                spacetime, REFERENCE_TIME, profile,
            )
            forward = integrate_coordinate_generators(
                spacetime, REFERENCE_TIME, candidate["time"], position, velocity,
                output_times=np.asarray((REFERENCE_TIME, candidate["time"])),
                rtol=PRIMARY_RTOL, atol=PRIMARY_ATOL,
            )
            buffer = 0.0005 * float(np.median(terminal_profile["rho"]))
            classified = finite_terminal_classification(
                spacetime.z[-1], forward["positions"][-1], terminal_profile, buffer,
            )
            differences = np.asarray(classified.pop("signed_rho_difference"))[core]
            valid = np.isfinite(differences)
            count = max(int(np.count_nonzero(valid)), 1)
            records[label] = {
                **classified,
                "core_valid_count": int(np.count_nonzero(valid)),
                "core_outside_fraction": float(np.count_nonzero(differences[valid] > buffer) / count),
                "core_inside_fraction": float(np.count_nonzero(differences[valid] < -buffer) / count),
                "expected_class": expected,
                "buffer": buffer,
                "maximum_forward_null_residual": forward["maximum_normalized_null_residual"],
            }
        return records

    return run_json_stage(index, stage_id, 1800.0, metadata, calculate)


def profile_gap_at_reference(spacetime, trace, reference_profile):
    index = time_index(trace, REFERENCE_TIME)
    angle, rho = polar_coordinates(spacetime.z[-1], trace["positions"][index])
    spline = CubicSpline(
        np.asarray(reference_profile["theta"]),
        np.asarray(reference_profile["rho"]), extrapolate=False,
    )
    reference = spline(angle)
    core = slice(2, len(angle) - 2)
    difference = rho[core] - reference[core]
    relative = difference / np.maximum(reference[core], 1e-300)
    return {
        "minimum_signed_rho_gap": float(np.min(difference)),
        "maximum_signed_rho_gap": float(np.max(difference)),
        "median_signed_rho_gap": float(np.median(difference)),
        "maximum_absolute_relative_rho_gap": float(np.max(np.abs(relative))),
        "outside_fraction": float(np.mean(difference >= 0.0)),
    }


def trace_difference(left, right, time_values):
    records = []
    for value in time_values:
        lp = left["positions"][time_index(left, value)]
        rp = right["positions"][time_index(right, value)]
        scale = np.median(np.sqrt((lp[:, 0] - 2.718281828459045)**2 + lp[:, 1]**2))
        records.append({
            "time": float(value),
            "maximum_relative_coordinate_separation": float(
                np.max(np.linalg.norm(lp - rp, axis=1)) / max(scale, 1e-300)
            ),
        })
    return records


def main():
    if not PROTOCOL.exists():
        raise FileNotFoundError(f"sealed protocol required: {PROTOCOL}")
    overall_started = time.perf_counter()
    index = RecoveryIndex(
        MANIFEST, PROTOCOL, expected_inputs(), maximum_stage_seconds=3600.0,
    )
    controls = run_json_stage(
        index, "controls/all", 300.0, {"control_version": 1}, analytic_controls,
    )
    if not controls["passed"]:
        raise RuntimeError("Test15A analytic/adverse controls failed")
    case_results = {}
    trace_store = {}
    for domain, grid in CASES:
        case_id = f"{domain}_{grid}"
        print(f"Test15A {case_id}: loading dense archive", flush=True)
        history = load_history(domain, grid)
        checked = run_json_stage(
            index, f"history/{case_id}", 120.0,
            {"domain": domain, "grid": grid, "levels": 65},
            lambda history=history, domain=domain, grid=grid: history_check(
                history, domain, grid,
            ),
        )
        if not checked["finite"] or not checked["sampled_positive_spatial_base"]:
            raise RuntimeError(f"invalid archived history for {case_id}")
        spacetime = ArchivedSpacetime(
            history["times"], history["z"], history["r"],
            history["position"], history["velocity"],
        )
        profiles = {}
        for time_value in (0.004, 0.006, 0.007, 0.008):
            label = f"outer_T{int(round(time_value * 1000)):03d}"
            profiles[label] = run_json_stage(
                index, f"profile/{case_id}/{label}", 300.0,
                {"time": time_value, "seed": 1.6, "branch": "outer"},
                lambda history=history, time_value=time_value: solve_outer_profile(
                    history, time_value,
                ),
            )
        candidates = {}
        traces = {}
        diagnostics = {}
        for name, time_value, scale in TERMINAL_SPECS:
            print(f"Test15A {case_id}: {name}", flush=True)
            candidate = terminal_candidate(
                index, case_id, history, name, time_value, scale, profiles,
            )
            candidates[name] = candidate
            trace = trace_candidate(index, case_id, spacetime, candidate)
            traces[name] = trace
            strip = generator_strip_diagnostics(
                spacetime, trace["times"], trace["positions"], core_margin=2,
            )
            diagnostics[name] = {
                "maximum_normalized_null_residual": float(np.max(
                    trace["normalized_null_residual"]
                )),
                "strip": strip,
                "segments": trace["segment_paths"],
            }
        central = candidates["mots_T008"]
        central_trace = traces["mots_T008"]
        checks = numerical_checks(
            index, case_id, spacetime, central, central_trace,
        )
        bracket = finite_bracket_check(
            index, case_id, spacetime, central, central_trace,
        )
        reference_profile = terminal_profile_from_surface(
            profiles["outer_T004"]["surface"], THETA, scale=1.0,
        )
        mots_gap = profile_gap_at_reference(
            spacetime, central_trace, reference_profile,
        )
        case_results[case_id] = {
            "domain": domain,
            "grid": grid,
            "history": checked,
            "terminal_profiles": {
                key: {
                    "time": value["time"],
                    "rho_axis": value["surface"]["rho_axis"],
                    "rho_brane": value["surface"]["rho_brane"],
                    "local_expansion": value["surface"]["local_expansion_interior_maximum"],
                    "crosscheck": value["surface"]["primary_evaluator_crosscheck"]["two_cell_interior_maximum"],
                } for key, value in profiles.items()
            },
            "candidate_diagnostics": diagnostics,
            "numerical_checks": checks,
            "finite_bracket": bracket,
            "proxy_outer_MOTS_gap_at_t004": mots_gap,
        }
        trace_store[case_id] = traces
        del spacetime, history

    common_times = TIMES
    grid_comparisons = {}
    for domain in ("R8", "R10"):
        grid_comparisons[domain] = trace_difference(
            trace_store[f"{domain}_G7"]["mots_T008"],
            trace_store[f"{domain}_G8"]["mots_T008"], common_times,
        )
    domain_comparisons = {}
    for grid in ("G7", "G8"):
        domain_comparisons[grid] = trace_difference(
            trace_store[f"R8_{grid}"]["mots_T008"],
            trace_store[f"R10_{grid}"]["mots_T008"], common_times,
        )
    terminal_time_sensitivity = {}
    offset_sensitivity = {}
    for case_id, traces in trace_store.items():
        pair_records = []
        for left, right in (("mots_T006", "mots_T007"), ("mots_T006", "mots_T008"), ("mots_T007", "mots_T008")):
            maximum = min(
                traces[left]["times"][0], traces[right]["times"][0],
            )
            values = TIMES[TIMES <= maximum + 1e-12]
            pair_records.append({
                "pair": [left, right],
                "records": trace_difference(traces[left], traces[right], values),
            })
        terminal_time_sensitivity[case_id] = pair_records
        offset_records = trace_difference(
            traces["offset_minus2_T008"], traces["offset_plus2_T008"], TIMES,
        )
        terminal_width = offset_records[-1]["maximum_relative_coordinate_separation"]
        maximum_width = max(
            item["maximum_relative_coordinate_separation"] for item in offset_records
        )
        offset_sensitivity[case_id] = {
            "records": offset_records,
            "terminal_width": terminal_width,
            "maximum_width": maximum_width,
            "maximum_amplification": float(maximum_width / max(terminal_width, 1e-300)),
        }

    max_null = max(
        value["maximum_normalized_null_residual"]
        for item in case_results.values()
        for value in item["candidate_diagnostics"].values()
    )
    max_strict = max(
        item["numerical_checks"]["strict_primary_maximum_cell_difference"]
        for item in case_results.values()
    )
    max_hamiltonian = max(
        item["numerical_checks"]["Hamiltonian_primary_maximum_cell_difference"]
        for item in case_results.values()
    )
    max_roundtrip = max(
        item["numerical_checks"]["roundtrip_maximum_cell_difference"]
        for item in case_results.values()
    )
    caustic_free = all(
        not value["strip"]["caustic_or_ordering_loss_detected"]
        and value["strip"]["all_core_generators_in_domain"]
        for item in case_results.values()
        for value in item["candidate_diagnostics"].values()
    )
    brackets_pass = all(
        item["finite_bracket"]["outward"]["core_outside_fraction"] >= BRACKET_FRACTION
        and item["finite_bracket"]["inward"]["core_inside_fraction"] >= BRACKET_FRACTION
        for item in case_results.values()
    )
    max_grid = max(
        value["maximum_relative_coordinate_separation"]
        for records in grid_comparisons.values() for value in records
    )
    max_domain = max(
        value["maximum_relative_coordinate_separation"]
        for records in domain_comparisons.values() for value in records
    )
    max_terminal_time = max(
        value["maximum_relative_coordinate_separation"]
        for records in terminal_time_sensitivity.values()
        for pair in records for value in pair["records"]
    )
    max_offset = max(item["maximum_width"] for item in offset_sensitivity.values())
    max_amplification = max(
        item["maximum_amplification"] for item in offset_sensitivity.values()
    )
    acceptance = {
        "controls_pass": controls["passed"],
        "primary_null_residual_below_limit": max_null < NULL_LIMIT,
        "strict_integrator_convergence": max_strict < STRICT_CELL_LIMIT,
        "independent_Hamiltonian_agreement": max_hamiltonian < HAMILTONIAN_CELL_LIMIT,
        "roundtrip_reversibility": max_roundtrip < ROUNDTRIP_CELL_LIMIT,
        "no_core_caustic_or_domain_departure": caustic_free,
        "finite_inward_outward_bracket": brackets_pass,
        "G7_G8_transfer_below_1_percent": max_grid < GRID_LIMIT,
        "R8_R10_transfer_below_3_percent": max_domain < DOMAIN_LIMIT,
        "terminal_time_sensitivity_below_3_percent": max_terminal_time < TERMINAL_TIME_LIMIT,
        "offset_envelope_below_6_percent": max_offset < OFFSET_ENVELOPE_LIMIT,
        "offset_envelope_amplification_below_1p5": max_amplification < OFFSET_AMPLIFICATION_LIMIT,
        "all_recovery_stages_valid": bool(all(
            value["status"] == "complete" and index.validated_path(stage_id) is not None
            for stage_id, value in index.data["stages"].items()
        )),
    }
    numerical_core = all(
        acceptance[key] for key in (
            "controls_pass", "primary_null_residual_below_limit",
            "strict_integrator_convergence", "independent_Hamiltonian_agreement",
            "roundtrip_reversibility", "all_recovery_stages_valid",
        )
    )
    if all(acceptance.values()):
        status = "pass"
        classification = "resolved_finite_archive_causal_proxy_with_bounded_terminal_sensitivity"
    elif numerical_core:
        status = "review"
        classification = "finite_archive_causal_proxy_numerically_valid_but_sensitive_or_mixed"
    else:
        status = "fail"
        classification = "invalid_or_unresolved_finite_archive_null_reconstruction"
    payload = {
        "status": status,
        "classification": classification,
        "scope": (
            "finite-archive backward outgoing-null-generator reconstruction; "
            "a causal-separatrix/event-horizon proxy, not a true event horizon"
        ),
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "preserved_prior_statuses": {
            "note85_R8_long": "REVIEW",
            "note88_R12_domain_sequence": "REVIEW",
            "note93_joint_convergence": "REVIEW",
        },
        "archive_times": TIMES,
        "generator_labels": THETA,
        "terminal_specs": TERMINAL_SPECS,
        "controls": controls,
        "case_results": case_results,
        "G7_G8_comparisons": grid_comparisons,
        "R8_R10_comparisons": domain_comparisons,
        "terminal_time_sensitivity": terminal_time_sensitivity,
        "offset_sensitivity": offset_sensitivity,
        "global_metrics": {
            "maximum_primary_normalized_null_residual": max_null,
            "maximum_strict_primary_cell_difference": max_strict,
            "maximum_Hamiltonian_primary_cell_difference": max_hamiltonian,
            "maximum_roundtrip_cell_difference": max_roundtrip,
            "maximum_G7_G8_relative_separation": max_grid,
            "maximum_R8_R10_relative_separation": max_domain,
            "maximum_terminal_time_relative_separation": max_terminal_time,
            "maximum_offset_envelope_relative_width": max_offset,
            "maximum_offset_envelope_amplification": max_amplification,
        },
        "acceptance": acceptance,
        "escape_capture_semantics": (
            "inside/outside classifications refer only to signed terminal-profile "
            "separation over the finite archive; they do not mean capture by a "
            "global black hole or escape to null infinity"
        ),
        "claim_boundary": (
            "No stationary future endpoint or asymptotic exterior is available. "
            "The reconstructed object is terminal-conditioned and cannot be "
            "called the event horizon. Apparent-horizon comparisons are descriptive."
        ),
        "manifest": str(MANIFEST),
        "runtime_seconds": float(time.perf_counter() - overall_started),
    }
    atomic_write_json(OUTPUT, builtin(payload))
    print(json.dumps({
        "status": status, "classification": classification,
        "global_metrics": payload["global_metrics"],
        "runtime_seconds": payload["runtime_seconds"], "output": str(OUTPUT),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
