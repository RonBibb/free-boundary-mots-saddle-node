#!/usr/bin/env python3
"""Three-grid free-boundary MOTS saddle-node continuation."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import continuation_core as CORE


SCHEMA = "protocol229-free-boundary-mots-saddle-node-v4"
GRID_SCHEMA = "protocol229-grid-continuation-v4"
DT = 0.00003125
START_TIME = 0.001
END_TIME = 0.0015
TIME_SCALE = 0.0005
START_STEP = 32
END_STEP = 48
INITIAL_ARC_STEP = 1.0 / 64.0
MINIMUM_ARC_BRACKET = 1.0 / 4096.0
MINIMUM_CORRECTOR_STEP = 1.0 / 4096.0
MINIMUM_BRACKET_CORRECTOR_STEP = 1.0 / 65536.0
CORRECTOR_ARCLENGTH_LIMIT = 2.0e-5
MAXIMUM_CONTINUATION_STEPS = 64
PAIR_OFFSET_MULTIPLIERS = (1, 2, 4, 8, 16)
GRIDS = ("G10", "G9", "G11")
FIELDS = ("q", "v", "source", "memory")
THREAD_VARS = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)


class Protocol229Error(RuntimeError):
    pass


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path, root=None):
    path = Path(path).absolute()
    return {
        "path": path.relative_to(Path(root).absolute()).as_posix() if root else str(path),
        "byte_count": path.stat().st_size,
        "sha256": sha256(path),
    }


def same(path, item):
    path = Path(path)
    return bool(
        set(item) == {"path", "byte_count", "sha256"}
        and path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
        and type(item["byte_count"]) is int and path.stat().st_size == item["byte_count"]
        and type(item["sha256"]) is str and sha256(path) == item["sha256"]
    )


def protocol228_root(academic_root):
    return Path(academic_root).absolute() / "Discussion/protocols/protocol228-repaired-parent-formation-time-2026-08-23"


def input_paths(academic_root):
    p228 = protocol228_root(academic_root)
    protocols = p228.parent
    return {
        "p228/protocol": p228 / "PROTOCOL.md",
        "p228/source": p228 / "protocol228.py",
        "p228/bootstrap": p228 / "bootstrap.py",
        "p228/authority": p228 / "authority/freeze_record.json",
        "p228/result": p228 / "candidate-output/protocol228_result.json",
        "p228/profiles": p228 / "candidate-output/protocol228_profiles.npz",
        "p228/G9-checkpoint": p228 / "candidate-output/G9_step0048.npz",
        "p228/G10-checkpoint": p228 / "candidate-output/G10_step0048.npz",
        "p228/G11-checkpoint": p228 / "candidate-output/G11_step0048.npz",
        "p228/G9-receipt": p228 / "candidate-output/G9_step0048.json",
        "p228/G10-receipt": p228 / "candidate-output/G10_step0048.json",
        "p228/G11-receipt": p228 / "candidate-output/G11_step0048.json",
        "G9/start": protocols / "protocol220-recoverable-canonical-spatial-2026-08-22/candidate-output/G9_repeat1.npz",
        "G10/start": protocols / "protocol216-doubled-horizon-canonical-2026-08-22/linux-output/g10_doubled_horizon_canonical.npz",
        "G11/start": protocols / "protocol226-corrected-canonical-g11-2026-08-23/candidate-output/G11_repeat1.npz",
    }


def freeze(root, academic_root):
    root = Path(root).absolute(); inputs = input_paths(academic_root)
    if any((root / "authority").iterdir()) or (root / "candidate-output").exists():
        raise Protocol229Error("prospective namespace differs")
    for name, path in inputs.items():
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
            raise Protocol229Error("unsafe input: " + name)
    result = read_json(inputs["p228/result"])
    if result.get("classification") != "REPAIRED-PARENT-FORMATION-CLOSURE-PASS" or not all(result.get("acceptance", {}).values()):
        raise Protocol229Error("Protocol228 prerequisite differs")
    if result.get("candidate_time_over_ell") != END_TIME or result.get("formation_sampling_bracket") != [START_TIME, END_TIME]:
        raise Protocol229Error("Protocol228 bracket differs")
    sources = (
        "PROTOCOL.md", "continuation_core.py", "protocol229.py", "bootstrap.py",
        "tests/test_continuation_core.py", "tests/test_protocol229.py",
    )
    authority = {
        "schema": SCHEMA, "status": "FROZEN", "dt": DT,
        "time_interval": [START_TIME, END_TIME], "time_scale": TIME_SCALE,
        "grids": list(GRIDS), "initial_arc_step": INITIAL_ARC_STEP,
        "minimum_arc_bracket": MINIMUM_ARC_BRACKET,
        "minimum_corrector_step": MINIMUM_CORRECTOR_STEP,
        "minimum_bracket_corrector_step": MINIMUM_BRACKET_CORRECTOR_STEP,
        "corrector_arclength_limit": CORRECTOR_ARCLENGTH_LIMIT,
        "corrector_backoff": "restart-each-accepted-step-at-1/64-then-dyadic-to-1/4096",
        "maximum_continuation_steps": MAXIMUM_CONTINUATION_STEPS,
        "pair_offset_multipliers": list(PAIR_OFFSET_MULTIPLIERS),
        "dense_RK2_formula": "y_n + dt*((theta-theta^2)*k1 + theta^2*k2)",
        "sources": {name: record(root / name, root) for name in sources},
        "inputs": {name: record(path) for name, path in inputs.items()},
        "candidate_output_absent_at_freeze": True,
        "parent_solve_or_repair_authorized": False,
        "continuum_theorem_claim_authorized": False,
        "event_horizon_claim_authorized": False,
        "phase_selection_claim_authorized": False,
        "source_ownership_claim_authorized": False,
    }
    authority["fingerprint"] = hashlib.sha256(b"protocol229-freeze\0" + canonical(authority)).hexdigest()
    atomic_json(root / "authority/freeze_record.json", authority)
    return authority


def verify(root, academic_root):
    root = Path(root).absolute(); authority = read_json(root / "authority/freeze_record.json")
    fingerprint = authority.pop("fingerprint", None)
    expected = hashlib.sha256(b"protocol229-freeze\0" + canonical(authority)).hexdigest()
    authority["fingerprint"] = fingerprint
    if authority.get("schema") != SCHEMA or fingerprint != expected:
        raise Protocol229Error("authority differs")
    for name, item in authority["sources"].items():
        if not same(root / item["path"], item):
            raise Protocol229Error("source differs: " + name)
    inputs = input_paths(academic_root)
    if set(inputs) != set(authority["inputs"]):
        raise Protocol229Error("input inventory differs")
    for name, item in authority["inputs"].items():
        if not same(inputs[name], item):
            raise Protocol229Error("input differs: " + name)
    return authority, inputs


def fsync_dir(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value):
    path = Path(path); temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    if path.exists() or path.is_symlink():
        raise Protocol229Error("output exists: " + path.name)
    with temporary.open("xb") as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    os.replace(temporary, path); fsync_dir(path.parent)


def atomic_npz(path, arrays):
    path = Path(path); temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    if path.exists() or path.is_symlink():
        raise Protocol229Error("output exists: " + path.name)
    with temporary.open("xb") as stream:
        np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    os.replace(temporary, path); fsync_dir(path.parent)


def runtime_preflight(root):
    if not sys.dont_write_bytecode or any(os.environ.get(name) != "1" for name in THREAD_VARS):
        raise Protocol229Error("runtime controls differ")
    expected = (Path(root) / "bootstrap.py").resolve()
    observed = Path(getattr(sys.modules.get("__main__"), "__file__", "")).resolve()
    if observed != expected:
        raise Protocol229Error("authorized bootstrap was bypassed")


class DenseTrajectory:
    def __init__(self, intervals):
        self.intervals = intervals

    def state(self, time_value):
        time_value = float(time_value)
        if time_value < START_TIME - 1e-15 or time_value > END_TIME + 1e-15:
            raise Protocol229Error("dense trajectory queried outside frozen interval")
        if time_value >= END_TIME:
            interval = self.intervals[-1]; fraction = 1.0
        else:
            index = min(int(math.floor((time_value - START_TIME) / DT)), len(self.intervals) - 1)
            interval = self.intervals[index]
            fraction = (time_value - interval["time"]) / DT
        if fraction < -1e-12 or fraction > 1 + 1e-12:
            raise Protocol229Error("dense trajectory fraction differs")
        fraction = min(max(fraction, 0.0), 1.0)
        one = fraction - fraction * fraction
        two = fraction * fraction
        return tuple(np.ascontiguousarray(
            interval["state"][index] + DT * (one * interval["k1"][index] + two * interval["k2"][index])
        ) for index in range(2))


def load_state(label, path):
    with np.load(path, allow_pickle=False) as archive:
        prefix = "endpoint_" if label == "G10" else ""
        return tuple(np.ascontiguousarray(archive[prefix + name], dtype=np.float64) for name in FIELDS)


def load_endpoint(label, path):
    with np.load(path, allow_pickle=False) as archive:
        return tuple(np.ascontiguousarray(archive[name], dtype=np.float64) for name in FIELDS)


def replay_dense_trajectory(context, label, start_state, expected_endpoint):
    runner = context["runner"]; bundle = context["g11_bundle"] if label == "G11" else context["bundles"][label]
    state = tuple(np.ascontiguousarray(value) for value in start_state); intervals = []
    for step in range(START_STEP + 1, END_STEP + 1):
        time_value = (step - 1) * DT
        first = runner.reconstruct_driver_stage(bundle, context["p228"].MODE, time_value, state, capture=True)
        first_audit = runner._technical_stage_audit(bundle, context["p228"].MODE, first)
        midpoint = tuple(value + np.float64(0.5 * DT) * slope for value, slope in zip(state, first["slopes"]))
        second = runner.reconstruct_driver_stage(bundle, context["p228"].MODE, time_value + 0.5 * DT, midpoint, capture=True)
        second_audit = runner._technical_stage_audit(bundle, context["p228"].MODE, second)
        if not (first["finite"] and second["finite"] and first_audit["gates"]["all_technical_gates_pass"] and second_audit["gates"]["all_technical_gates_pass"]):
            raise Protocol229Error(label + " dense trajectory stage failed")
        intervals.append({
            "time": time_value,
            "state": tuple(np.ascontiguousarray(value) for value in state[:2]),
            "k1": tuple(np.ascontiguousarray(value) for value in first["slopes"][:2]),
            "k2": tuple(np.ascontiguousarray(value) for value in second["slopes"][:2]),
        })
        state = tuple(value + np.float64(DT) * slope for value, slope in zip(state, second["slopes"]))
    exact = all(np.array_equal(left, right) for left, right in zip(state, expected_endpoint))
    if not exact:
        raise Protocol229Error(label + " Protocol228 endpoint replay differs")
    trajectory = DenseTrajectory(intervals)
    dense_endpoint = trajectory.state(END_TIME)
    if not all(np.array_equal(dense_endpoint[index], state[index]) for index in range(2)):
        raise Protocol229Error(label + " dense endpoint differs")
    return trajectory


def coordinates(label):
    shape = {"G9": (113, 211), "G10": (129, 241), "G11": (145, 271)}[label]
    return np.linspace(1.0, math.e, shape[0]), np.linspace(0.0, 10.0, shape[1])


def tau_to_time(tau):
    return START_TIME + float(tau) * TIME_SCALE


def time_to_tau(time_value):
    return (float(time_value) - START_TIME) / TIME_SCALE


def load_anchor(inputs, label, branch):
    with np.load(inputs["p228/profiles"], allow_pickle=False) as archive:
        prefix = label + "_" + branch + "_"
        return {
            "theta": np.ascontiguousarray(archive[prefix + "theta"]),
            "rho": np.ascontiguousarray(archive[prefix + "rho"]),
            "slope": np.ascontiguousarray(archive[prefix + "slope"]),
            "tau": time_to_tau(END_TIME),
        }


def prepared_at(trajectory, time_value, z, r, p227):
    position, velocity = trajectory.state(time_value)
    return position, velocity, p227.prepare_capped_expansion_slice(position, velocity, z, r)


def refine_point(point, trajectory, z, r, p227):
    time_value = tau_to_time(point["tau"])
    position, velocity, prepared = prepared_at(trajectory, time_value, z, r, p227)
    surface = p227.solve_dynamical_capped_surface_bvp(
        position, velocity, z, r, point, tolerance=2e-6,
        nodes=121, maximum_nodes=6000, dense_nodes=501, prepared=prepared,
    )
    if not p227.admitted(surface):
        raise Protocol229Error("continuation point fails inherited admission")
    return {
        "theta": np.ascontiguousarray(surface["theta"]),
        "rho": np.ascontiguousarray(surface["rho"]),
        "slope": np.ascontiguousarray(surface["slope"]),
        "tau": float(point["tau"]),
    }, surface, (position, velocity, prepared)


def stability_value(point, trajectory, z, r, p227, nodes=65):
    time_value = tau_to_time(point["tau"])
    position, velocity, prepared = prepared_at(trajectory, time_value, z, r, p227)
    value = p227.mots_stability_matrix(
        position, velocity, z, r, point, nodes=nodes,
        relative_step=1e-5, prepared=prepared,
    )
    return float(value["principal_eigenvalue_real"]), value, (position, velocity, prepared)


def point_public(point, eigenvalue):
    return {
        "time_over_ell": tau_to_time(point["tau"]), "tau": float(point["tau"]),
        "rho_axis": float(point["rho"][0]), "rho_brane": float(point["rho"][-1]),
        "principal_eigenvalue_65": float(eigenvalue),
    }


def continuation_advance(
    second_derivative, previous, current, trajectory, z, r, p227, *,
    initial_step=INITIAL_ARC_STEP, minimum_step=MINIMUM_CORRECTOR_STEP,
    tolerance=2e-6,
):
    def attempt(step_size):
        corrected = CORE.pseudo_arclength_step(
            second_derivative, previous, current, step_size,
            nodes=121, tolerance=tolerance, maximum_nodes=6000, dense_nodes=501,
        )
        metrics = {
            key: corrected[key] for key in (
                "success", "message", "iterations", "mesh_nodes_used",
                "predicted_tau", "corrected_tau", "secant_norm",
                "arclength_residual", "boundary_slope_error", "ode_second_defect",
            )
        }
        if not corrected["success"]:
            return {
                "success": False, "reason": "augmented-corrector-failed",
                "metrics": metrics, "payload": None,
            }
        if corrected["arclength_residual"] > CORRECTOR_ARCLENGTH_LIMIT:
            return {
                "success": False, "reason": "arclength-residual-exceeded",
                "metrics": metrics, "payload": None,
            }
        try:
            point, _, _ = refine_point(corrected["point"], trajectory, z, r, p227)
            eigenvalue, _, _ = stability_value(point, trajectory, z, r, p227)
        except (ValueError, RuntimeError, np.linalg.LinAlgError, Protocol229Error) as error:
            metrics["postprocess_error"] = type(error).__name__ + ": " + str(error)
            return {
                "success": False, "reason": "inherited-fixed-time-admission-failed",
                "metrics": metrics, "payload": None,
            }
        return {
            "success": True, "reason": "accepted",
            "metrics": metrics, "payload": {"point": point, "eigenvalue": eigenvalue},
        }

    return CORE.dyadic_backoff(attempt, initial_step, minimum_step)


def critical_coefficients(point, trajectory, z, r, p227):
    time_value = tau_to_time(point["tau"])
    position, velocity, prepared = prepared_at(trajectory, time_value, z, r, p227)
    stability_module = importlib.import_module(p227.mots_stability_matrix.__module__)
    bvp_module = importlib.import_module(p227.solve_dynamical_capped_surface_bvp.__module__)
    low = p227.mots_stability_matrix(position, velocity, z, r, point, nodes=49, relative_step=1e-5, prepared=prepared)
    middle = p227.mots_stability_matrix(position, velocity, z, r, point, nodes=65, relative_step=1e-5, prepared=prepared)
    primary = p227.mots_stability_matrix(position, velocity, z, r, point, nodes=81, relative_step=1e-5, prepared=prepared)
    check = p227.mots_stability_matrix(position, velocity, z, r, point, nodes=81, relative_step=2e-5, prepared=prepared)
    modes = CORE.principal_modes(primary["matrix"])
    if abs(modes["eigenvalue"].imag) > 1e-8 or abs(modes["next_eigenvalue"].imag) > 1e-8:
        raise Protocol229Error("critical eigenmode is complex")
    theta = primary["theta"]; rho = primary["rho"]
    first_matrix = stability_module.finite_difference_matrix(theta, 1, 7)
    second_matrix = stability_module.finite_difference_matrix(theta, 2, 7)
    extension = stability_module.neumann_extension(first_matrix)
    right = np.asarray(modes["right"].real, dtype=float)
    left = np.asarray(modes["left"].real, dtype=float)
    eta = CORE.physical_mode_to_radial(
        extension, primary["normal_factor"], right,
    )
    first = first_matrix @ rho; second = second_matrix @ rho

    def expansion_at(prepared_value, radial):
        return bvp_module.local_outgoing_expansion(
            prepared_value, theta, radial, first_matrix @ radial, second_matrix @ radial,
        )[1:-1]

    base = expansion_at(prepared, rho)
    temporal = []
    for half_width in (DT / 64.0, DT / 128.0):
        _, _, plus_prepared = prepared_at(trajectory, time_value + half_width, z, r, p227)
        _, _, minus_prepared = prepared_at(trajectory, time_value - half_width, z, r, p227)
        derivative = (expansion_at(plus_prepared, rho) - expansion_at(minus_prepared, rho)) / (2 * half_width)
        temporal.append(float(np.dot(left, derivative)))
    quadratic = []
    for relative_step in (1e-5, 5e-6):
        physical_step = relative_step * max(1.0, float(np.mean(rho)))
        plus = expansion_at(prepared, rho + physical_step * eta)
        minus = expansion_at(prepared, rho - physical_step * eta)
        second_direction = (plus - 2 * base + minus) / (physical_step * physical_step)
        quadratic.append(float(0.5 * np.dot(left, second_direction)))
    a_error = abs(temporal[0] - temporal[1]); b_error = abs(quadratic[0] - quadratic[1])
    return {
        "principal_eigenvalues_by_nodes": {
            "49": float(low["principal_eigenvalue_real"]),
            "65": float(middle["principal_eigenvalue_real"]),
            "81": float(primary["principal_eigenvalue_real"]),
        },
        "principal_65_81_difference": abs(middle["principal_eigenvalue_real"] - primary["principal_eigenvalue_real"]),
        "principal_eigenvalue_81": float(modes["eigenvalue"].real),
        "next_eigenvalue_81": float(modes["next_eigenvalue"].real),
        "principal_step_difference": abs(primary["principal_eigenvalue_real"] - check["principal_eigenvalue_real"]),
        "left_right_overlap_real": float(modes["overlap"].real),
        "left_right_overlap_imaginary": float(modes["overlap"].imag),
        "transversality_values": temporal, "transversality_step_difference": a_error,
        "quadratic_values": quadratic, "quadratic_step_difference": b_error,
        "transversality_resolved_nonzero": bool(np.sign(temporal[0]) == np.sign(temporal[1]) != 0 and abs(temporal[1]) > 5 * max(a_error, 1e-14)),
        "quadratic_resolved_nonzero": bool(np.sign(quadratic[0]) == np.sign(quadratic[1]) != 0 and abs(quadratic[1]) > 5 * max(b_error, 1e-14)),
        "simple_mode_gap_pass": bool(abs(modes["next_eigenvalue"].real - modes["eigenvalue"].real) > 1.0),
        "critical_operator_resolution_pass": bool(
            max(
                abs(low["principal_eigenvalue_real"] - middle["principal_eigenvalue_real"]),
                abs(middle["principal_eigenvalue_real"] - primary["principal_eigenvalue_real"]),
                abs(primary["principal_eigenvalue_real"] - check["principal_eigenvalue_real"]),
            ) < 0.02
        ),
        "principal_mode_sign_changes": int(primary["principal_eigenfunction_sign_changes"]),
        "normal_factor_positive": bool(primary["minimum_normal_factor"] > 0),
    }


def square_root_test(critical_time, positive_point, negative_point, trajectory, z, r, p227):
    records = []; positive_seed = positive_point; negative_seed = negative_point
    for multiplier in PAIR_OFFSET_MULTIPLIERS:
        target = critical_time + multiplier * DT / 16.0
        if target >= END_TIME:
            continue
        try:
            positive_seed, positive_surface, positive_context = refine_point(
                {**positive_seed, "tau": time_to_tau(target)}, trajectory, z, r, p227,
            )
            negative_seed, negative_surface, negative_context = refine_point(
                {**negative_seed, "tau": time_to_tau(target)}, trajectory, z, r, p227,
            )
            positive_lambda, _, _ = stability_value(positive_seed, trajectory, z, r, p227)
            negative_lambda, _, _ = stability_value(negative_seed, trajectory, z, r, p227)
            if not (positive_lambda > 0 and negative_lambda < 0):
                continue
            positive_geometry = p227.capped_surface_geometry(
                positive_context[0], positive_context[1], z, r, positive_surface, prepared=positive_context[2],
            )
            negative_geometry = p227.capped_surface_geometry(
                negative_context[0], negative_context[1], z, r, negative_surface, prepared=negative_context[2],
            )
            separation = abs(positive_geometry["one_sided_cap_area"] - negative_geometry["one_sided_cap_area"])
            if separation <= 0:
                continue
            records.append({
                "multiplier": multiplier, "time_over_ell": target,
                "positive_eigenvalue": positive_lambda, "negative_eigenvalue": negative_lambda,
                "positive_area": positive_geometry["one_sided_cap_area"],
                "negative_area": negative_geometry["one_sided_cap_area"],
                "area_separation": separation,
            })
        except (ValueError, RuntimeError, np.linalg.LinAlgError, Protocol229Error):
            continue
    fit = CORE.linear_square_root_fit(
        [item["time_over_ell"] for item in records],
        [item["area_separation"] for item in records],
    ) if len(records) >= 3 else None
    return records, fit


def execute_grid(context, inputs, label):
    p227 = context["p227"]; z, r = coordinates(label)
    start = load_state(label, inputs[label + "/start"])
    endpoint = load_endpoint(label, inputs["p228/" + label + "-checkpoint"])
    trajectory = replay_dense_trajectory(context, label, start, endpoint)
    outer = load_anchor(inputs, label, "outer")
    inner = load_anchor(inputs, label, "inner")
    outer_lambda, _, _ = stability_value(outer, trajectory, z, r, p227)
    inner_lambda, _, _ = stability_value(inner, trajectory, z, r, p227)
    if not (outer_lambda > 0 and inner_lambda < 0):
        raise Protocol229Error(label + " Protocol228 stability anchors differ")
    second_anchor, _, _ = refine_point(
        {**outer, "tau": time_to_tau(END_TIME - DT)}, trajectory, z, r, p227,
    )
    second_lambda, _, _ = stability_value(second_anchor, trajectory, z, r, p227)
    if second_lambda <= 0:
        raise Protocol229Error(label + " second outer anchor crosses fold")
    previous, current = outer, second_anchor
    previous_lambda, current_lambda = outer_lambda, second_lambda
    trace = [point_public(previous, previous_lambda), point_public(current, current_lambda)]
    bracket = None; pre_left = None
    cache = {"tau": None, "prepared": None}
    bvp_module = importlib.import_module(p227.solve_dynamical_capped_surface_bvp.__module__)

    def second_derivative(tau, theta, rho, slope):
        if cache["tau"] != tau:
            time_value = tau_to_time(tau)
            if time_value < START_TIME or time_value > END_TIME:
                return np.full_like(rho, np.nan)
            cache["tau"] = tau
            cache["prepared"] = prepared_at(trajectory, time_value, z, r, p227)[2]
        return bvp_module.dynamical_rho_second(cache["prepared"], theta, rho, slope)

    for index in range(MAXIMUM_CONTINUATION_STEPS):
        advanced = continuation_advance(
            second_derivative, previous, current, trajectory, z, r, p227,
        )
        print(json.dumps({
            "grid": label, "continuation_step_index": index,
            "accepted_step_size": advanced["accepted_step_size"],
            "attempts": advanced["attempts"],
        }, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)
        if not advanced["success"]:
            raise Protocol229Error(
                label + " pseudo-arclength corrector exhausted at accepted-step index " + str(index)
            )
        point = advanced["payload"]["point"]
        eigenvalue = advanced["payload"]["eigenvalue"]
        public = point_public(point, eigenvalue)
        public["accepted_arclength_step"] = advanced["accepted_step_size"]
        public["continuation_attempts"] = advanced["attempts"]
        trace.append(public)
        if current_lambda > 0 and eigenvalue < 0:
            pre_left, bracket = previous, {"positive": current, "positive_lambda": current_lambda, "negative": point, "negative_lambda": eigenvalue}
            break
        previous, current = current, point
        previous_lambda, current_lambda = current_lambda, eigenvalue
    if bracket is None:
        raise Protocol229Error(label + " principal sign crossing not found")

    step = INITIAL_ARC_STEP * 0.5
    bracket_refinement_index = 0
    while step > MINIMUM_ARC_BRACKET or abs(tau_to_time(bracket["positive"]["tau"]) - tau_to_time(bracket["negative"]["tau"])) > DT / 64.0:
        advanced = continuation_advance(
            second_derivative, pre_left, bracket["positive"], trajectory, z, r, p227,
            initial_step=step, minimum_step=MINIMUM_BRACKET_CORRECTOR_STEP,
            tolerance=1e-7,
        )
        print(json.dumps({
            "grid": label, "bracket_refinement_index": bracket_refinement_index,
            "requested_step_size": step,
            "accepted_step_size": advanced["accepted_step_size"],
            "attempts": advanced["attempts"],
        }, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)
        if not advanced["success"]:
            raise Protocol229Error(
                label + " zero bracket refinement exhausted at index "
                + str(bracket_refinement_index)
            )
        proposal = advanced["payload"]["point"]
        eigenvalue = advanced["payload"]["eigenvalue"]
        public = point_public(proposal, eigenvalue)
        public["bracket_refinement_index"] = bracket_refinement_index
        public["requested_arclength_step"] = step
        public["accepted_arclength_step"] = advanced["accepted_step_size"]
        public["continuation_attempts"] = advanced["attempts"]
        trace.append(public)
        if eigenvalue > 0:
            pre_left = bracket["positive"]
            bracket["positive"] = proposal; bracket["positive_lambda"] = eigenvalue
        elif eigenvalue < 0:
            bracket["negative"] = proposal; bracket["negative_lambda"] = eigenvalue
        else:
            bracket["negative"] = proposal; bracket["negative_lambda"] = eigenvalue
            break
        step = advanced["accepted_step_size"] * 0.5
        bracket_refinement_index += 1
        if step < MINIMUM_BRACKET_CORRECTOR_STEP:
            remaining_width = abs(
                tau_to_time(bracket["positive"]["tau"])
                - tau_to_time(bracket["negative"]["tau"])
            )
            if remaining_width > DT / 64.0:
                raise Protocol229Error(label + " zero bracket width unresolved at frozen step floor")
            break
    critical_time = 0.5 * (tau_to_time(bracket["positive"]["tau"]) + tau_to_time(bracket["negative"]["tau"]))
    critical = min((bracket["positive"], bracket["negative"]), key=lambda point: abs(stability_value(point, trajectory, z, r, p227)[0]))
    critical_lambda, _, critical_context = stability_value(critical, trajectory, z, r, p227)
    coefficients = critical_coefficients(critical, trajectory, z, r, p227)
    critical_surface = {"theta": critical["theta"], "rho": critical["rho"], "slope": critical["slope"]}
    geometry = p227.capped_surface_geometry(
        critical_context[0], critical_context[1], z, r, critical_surface, prepared=critical_context[2],
    )
    pairs, fit = square_root_test(
        critical_time, bracket["positive"], bracket["negative"], trajectory, z, r, p227,
    )
    time_width = abs(tau_to_time(bracket["positive"]["tau"]) - tau_to_time(bracket["negative"]["tau"]))
    scaling_pass = bool(
        len(pairs) >= 4 and fit is not None and fit["slope"] > 0 and fit["R_squared"] >= 0.98
        and 0.40 <= fit["log_exponent"] <= 0.60
        and abs(fit["critical_time"] - critical_time) <= max(time_width, DT / 64.0)
    )
    checks = {
        "Protocol228_endpoint_replay_bitwise_exact": True,
        "outer_and_inner_anchor_signs_exact": True,
        "opposite_principal_sign_bracket_found": True,
        "zero_time_bracket_below_dt_over_64": time_width <= DT / 64.0,
        "critical_principal_magnitude_below_0_02": abs(critical_lambda) < 0.02,
        "simple_real_sign_definite_principal_mode": bool(
            coefficients["simple_mode_gap_pass"] and coefficients["principal_mode_sign_changes"] == 0
            and coefficients["normal_factor_positive"] and coefficients["critical_operator_resolution_pass"]
        ),
        "adjoint_transversality_resolved_nonzero": coefficients["transversality_resolved_nonzero"],
        "quadratic_nondegeneracy_resolved_nonzero": coefficients["quadratic_resolved_nonzero"],
        "invariant_area_square_root_scaling_pass": scaling_pass,
    }
    passed = bool(all(checks.values()))
    arrays = {
        label + "_critical_theta": np.asarray(critical["theta"]),
        label + "_critical_rho": np.asarray(critical["rho"]),
        label + "_critical_slope": np.asarray(critical["slope"]),
    }
    result = {
        "schema": GRID_SCHEMA, "grid": label, "passed": passed,
        "checks": checks, "continuation_trace": trace,
        "zero_bracket": {
            "positive": point_public(bracket["positive"], bracket["positive_lambda"]),
            "negative": point_public(bracket["negative"], bracket["negative_lambda"]),
            "time_width": time_width,
        },
        "critical_time_estimate": critical_time,
        "critical_principal_eigenvalue": critical_lambda,
        "critical_geometry": geometry,
        "critical_coefficients": coefficients,
        "square_root_pairs": pairs, "square_root_fit": fit,
    }
    return result, arrays


def relative_difference(left, right):
    return float(abs(left - right) / max(abs(left), abs(right), 1e-300))


def cross_grid_checks(records):
    pairs = (("G9", "G10"), ("G10", "G11"))
    transfers = {}
    passed = True
    for left, right in pairs:
        a, b = records[left], records[right]
        item = {
            "critical_time_absolute_difference": abs(a["critical_time_estimate"] - b["critical_time_estimate"]),
            "critical_area_relative_difference": relative_difference(a["critical_geometry"]["one_sided_cap_area"], b["critical_geometry"]["one_sided_cap_area"]),
            "transversality_relative_difference": relative_difference(a["critical_coefficients"]["transversality_values"][1], b["critical_coefficients"]["transversality_values"][1]),
            "quadratic_relative_difference": relative_difference(a["critical_coefficients"]["quadratic_values"][1], b["critical_coefficients"]["quadratic_values"][1]),
            "transversality_sign_agrees": np.sign(a["critical_coefficients"]["transversality_values"][1]) == np.sign(b["critical_coefficients"]["transversality_values"][1]),
            "quadratic_sign_agrees": np.sign(a["critical_coefficients"]["quadratic_values"][1]) == np.sign(b["critical_coefficients"]["quadratic_values"][1]),
        }
        item["passed"] = bool(
            item["critical_time_absolute_difference"] <= DT / 4.0
            and item["critical_area_relative_difference"] < 0.01
            and item["transversality_sign_agrees"] and item["quadratic_sign_agrees"]
            and item["transversality_relative_difference"] < 0.20
            and item["quadratic_relative_difference"] < 0.20
        )
        transfers[left + "-" + right] = item; passed = passed and item["passed"]
    return transfers, bool(passed)


def grid_fingerprint(value):
    return hashlib.sha256(b"protocol229-grid\0" + canonical(value)).hexdigest()


def result_fingerprint(value):
    return hashlib.sha256(b"protocol229-result\0" + canonical(value)).hexdigest()


def publish_grid(output, context, inputs, authority_sha, label):
    json_path = output / ("protocol229_" + label + ".json")
    npz_path = output / ("protocol229_" + label + ".npz")
    if json_path.exists() or npz_path.exists():
        if not (json_path.is_file() and npz_path.is_file()):
            raise Protocol229Error("partial grid artifact: " + label)
        value = read_json(json_path); fingerprint = value.pop("fingerprint", None)
        expected = grid_fingerprint(value); value["fingerprint"] = fingerprint
        if fingerprint != expected or value.get("authority_sha256") != authority_sha or value.get("grid") != label:
            raise Protocol229Error("grid artifact differs: " + label)
        return value["result"]
    print(label + ": reconstructing dense trajectory and continuing branch", flush=True)
    result, arrays = execute_grid(context, inputs, label)
    atomic_npz(npz_path, arrays)
    value = {
        "schema": GRID_SCHEMA, "authority_sha256": authority_sha, "grid": label,
        "archive": record(npz_path, ROOT), "result": context["p220"].P190.jsonable(result),
    }
    value["fingerprint"] = grid_fingerprint(value); atomic_json(json_path, value)
    return value["result"]


def validate_final(path, authority_sha):
    value = read_json(path); fingerprint = value.pop("fingerprint", None)
    expected = result_fingerprint(value); value["fingerprint"] = fingerprint
    if fingerprint != expected or value.get("schema") != SCHEMA or value.get("authority_sha256") != authority_sha:
        raise Protocol229Error("final result differs")
    return value


def run(root, academic_root, visual_root, project_root):
    root = Path(root).absolute(); runtime_preflight(root)
    _, inputs = verify(root, academic_root); authority_sha = sha256(root / "authority/freeze_record.json")
    p228 = load(inputs["p228/source"], "protocol228_bound229")
    p228_authority, p228_inputs = p228.verify(protocol228_root(academic_root), academic_root)
    del p228_authority
    p220_path = protocol228_root(academic_root).parent / "protocol220-recoverable-canonical-spatial-2026-08-22/protocol220.py"
    p220 = load(p220_path, "protocol220_lock_bound229")
    with p220.exclusive_lock(root):
        output = root / "candidate-output"
        if not output.exists():
            output.mkdir(); fsync_dir(root)
        final_path = output / "protocol229_result.json"
        if final_path.exists():
            return validate_final(final_path, authority_sha)
        context = p228.load_modules_and_context(root, academic_root, visual_root, project_root, p228_inputs)
        context.update({"p228": p228, "p220": context["p220"]})
        old = context["runner"].axis_even_crossfit_audit
        context["runner"].axis_even_crossfit_audit = context["p220"].P190.node_crossfit
        records = {}
        try:
            records["G10"] = publish_grid(output, context, inputs, authority_sha, "G10")
            if records["G10"]["passed"]:
                records["G9"] = publish_grid(output, context, inputs, authority_sha, "G9")
                records["G11"] = publish_grid(output, context, inputs, authority_sha, "G11")
        finally:
            context["runner"].axis_even_crossfit_audit = old
        if set(records) == set(GRIDS):
            transfers, transfer_pass = cross_grid_checks(records)
            passed = bool(all(item["passed"] for item in records.values()) and transfer_pass)
            classification = "FREE-BOUNDARY-MOTS-SADDLE-NODE-CLOSURE-PASS" if passed else "SADDLE-NODE-CONDITIONS-NOT-SATISFIED"
        else:
            transfers, transfer_pass, passed = {}, False, False
            classification = "BIFURCATION-INCONCLUSIVE"
        value = {
            "schema": SCHEMA, "authority_sha256": authority_sha,
            "status": "PASS" if passed else "REVIEW", "classification": classification,
            "grids": records, "adjacent_grid_transfers": transfers,
            "all_grid_gates_pass": bool(set(records) == set(GRIDS) and all(item["passed"] for item in records.values())),
            "all_cross_grid_gates_pass": transfer_pass,
            "continuum_theorem_claim_authorized": False, "event_horizon_claim_authorized": False,
            "phase_selection_claim_authorized": False, "source_ownership_claim_authorized": False,
        }
        value["fingerprint"] = result_fingerprint(value); atomic_json(final_path, value)
        verify(root, academic_root)
        return validate_final(final_path, authority_sha)


def status(root):
    output = Path(root) / "candidate-output"
    grids = [label for label in GRIDS if (output / ("protocol229_" + label + ".json")).is_file()] if output.exists() else []
    final = output / "protocol229_result.json"; value = read_json(final) if final.is_file() else None
    return {
        "schema": "protocol229-status-v1", "completed_grids": grids,
        "active_or_next_grid": None if value else GRIDS[len(grids)] if len(grids) < len(GRIDS) else "finalization",
        "final_result_present": value is not None,
        "classification": None if value is None else value.get("classification"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(); commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "verify"):
        command = commands.add_parser(name); command.add_argument("--root", required=True); command.add_argument("--academic-root", required=True)
    command = commands.add_parser("run"); command.add_argument("--root", required=True); command.add_argument("--academic-root", required=True)
    command.add_argument("--visual-root", required=True); command.add_argument("--project-root", required=True)
    command = commands.add_parser("status"); command.add_argument("--root", required=True)
    values = vars(parser.parse_args(argv)); selected = values.pop("command")
    if selected == "freeze": result = freeze(**values)
    elif selected == "verify": authority, _ = verify(**values); result = {"verified": True, "fingerprint": authority["fingerprint"]}
    elif selected == "run": result = run(**values)
    else: result = status(**values)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
