#!/usr/bin/env python3
"""Protocol 245 full-timestep native-operator dense balance replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.interpolate import CubicSpline

# NumPy introduced ``trapezoid`` after the sealed Linux numerical runtime.
# The inherited manufactured control uses the new spelling; ``trapz`` is the
# same composite trapezoidal rule and keeps that control portable.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice
from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.test14_quasilocal_charge import reflected_cap_charge
from bhps.test14b_balance_closure import (
    _one_cap_integral,
    _sample,
    analytic_controls as balance_analytic_controls,
    evaluate_balance_leaf,
)
from bhps.test14c_coupled_seam import (
    analytic_controls as coupled_analytic_controls,
    apply_intrinsic_anisotropy,
    evaluate_geometric_bulk_leaf,
    leaf_marginal_transport_fields,
    marginal_tangency_rate,
    physical_coupled_record,
    seam_endpoint_transport,
)

from authority import file_record, sha256, verify_freeze
from dense_balance_core import (
    centered_directional,
    classify,
    inherited_rate_pass,
    normalized_ledger_residual,
    null_decomposition,
    projected_norm,
    relative_scale_error,
    three_point_rates,
)


SCHEMA = "protocol245-full-dt-native-operator-dense-balance-result-v1"
P244 = ROOT / "sealed-inputs/protocol244"
P244_OUTPUT = P244 / "candidate-output"
P243 = ROOT / "sealed-inputs/protocol243"
BACKGROUND_PATH = ROOT / "sealed-inputs/historical-balance/corrected_A790_test14b_balance_closure.json"
OUTPUT = ROOT / "candidate-output"
FULL_DT = 3.125e-5
LEAF_STEPS = tuple(range(38, 49))
INTERIOR_STEPS = tuple(range(39, 48))
WIDTHS = (5, 7, 9)
PATHS = ("backward", "centered", "forward")
EPSILONS = (1e-6, 5e-7, 2.5e-7)
FIELDS = ("q", "v", "source", "memory")
GRID_LABEL = "G10"
P244_AUTHORITY_SHA = "aea0e391b06b72a367b0d0914826acf028bf89e7d281a368942bacdaed2c6de1"
P244_RESULT_SHA = "60ca87bcb4bd45937d2babace30e50b1ea059839fa554cd21c9ff3803879958f"
P244_ARRAYS_SHA = "5e02f5edbd45f81a7da0c667ed9c55af6c6184ef3eb808471b4952d9bf0ae8f6"
P243_AUTHORITY_SHA = "021b0511249882d89a1e3a3f08d7f937bfedcd12be8a0ddd95caf0b4d88e2a36"
P243_RESULT_SHA = "b340e4953ed5463ba0d7401535f85567602c216e724b2ac96064ea1b607c06bc"


class Protocol245Error(RuntimeError):
    pass


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def raw_array_record(value):
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "byte_count": int(array.nbytes),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def fsync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path, payload):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise Protocol245Error(f"output namespace is not fresh: {path.name}")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o444)
    os.replace(temporary, path)
    fsync_directory(path.parent)


def atomic_json(path, value):
    atomic_bytes(path, canonical(value))
    if read_json(path) != value:
        raise Protocol245Error("JSON immediate replay failed")


def atomic_npz(path, arrays):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise Protocol245Error(f"output namespace is not fresh: {path.name}")
    with temporary.open("xb") as stream:
        np.savez(stream, **{name: np.ascontiguousarray(value) for name, value in arrays.items()})
        stream.flush()
        os.fsync(stream.fileno())
        os.fchmod(stream.fileno(), 0o444)
    os.replace(temporary, path)
    fsync_directory(path.parent)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(arrays) or any(
            not np.array_equal(archive[name], arrays[name]) for name in arrays
        ):
            raise Protocol245Error("NPZ immediate replay failed")


def current_peak_rss_bytes():
    usage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return usage if sys.platform == "darwin" else usage * 1024


def check_fingerprint(record, prefix):
    bare = dict(record)
    fingerprint = bare.pop("fingerprint", None)
    return fingerprint == hashlib.sha256(prefix + canonical(bare)).hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def validate_p243():
    if sha256(P243 / "freeze_record.json") != P243_AUTHORITY_SHA:
        raise Protocol245Error("Protocol 243 authority differs")
    path = P243 / "protocol243_result.json"
    if sha256(path) != P243_RESULT_SHA:
        raise Protocol245Error("Protocol 243 result differs")
    result = read_json(path)
    if not (
        check_fingerprint(result, b"protocol243-result-v1\0")
        and result.get("schema") == "protocol243-native-operator-dense-balance-admission-result-v1"
        and result.get("classification") == "DENSE-NATIVE-OPERATOR-LOCAL-BALANCE-PASS"
        and result.get("complete_repeat_exact") is True
        and result.get("passed") is True
        and all(result.get("ordered_gates", {}).get(name) is True for name in (
            "orientation", "area", "seam", "native_wall_rate", "flux_ledger",
        ))
        and result.get("temporal_or_cross_grid_native_balance_study_authorized") is True
        and result.get("temporal_or_cross_grid_balance_closure_established") is False
        and result.get("spacetime_evolution_executed") is False
        and result.get("surface_solve_executed") is False
    ):
        raise Protocol245Error("Protocol 243 semantics differ")
    return result


def validate_p244():
    if sha256(P244 / "freeze_record.json") != P244_AUTHORITY_SHA:
        raise Protocol245Error("Protocol 244 authority differs")
    result_path = P244_OUTPUT / "protocol244_result.json"
    archive_path = P244_OUTPUT / "protocol244_dense_tube_arrays.npz"
    if sha256(result_path) != P244_RESULT_SHA or sha256(archive_path) != P244_ARRAYS_SHA:
        raise Protocol245Error("Protocol 244 final bytes differ")
    result = read_json(result_path)
    scientific = result.get("scientific", {})
    if not (
        check_fingerprint(result, b"protocol244-result-v1\0")
        and result.get("schema") == "protocol244-full-dt-g10-dense-tube-result-v1"
        and result.get("authority_sha256") == P244_AUTHORITY_SHA
        and scientific.get("classification") == "FULL-DT-DENSE-G10-OUTER-MARGINAL-TUBE-PASS"
        and scientific.get("leaf_steps") == list(LEAF_STEPS)
        and scientific.get("checkpoint_steps") == list(range(33, 49))
        and all(scientific.get("gates", {}).values())
        and scientific.get("all_parent_checkpoint_controls_bitwise_exact") is True
        and result.get("full_dt_native_balance_replay_authorized") is True
        and result.get("parent_or_published_artifact_modified") is False
        and result.get("submitted_paper_edited") is False
    ):
        raise Protocol245Error("Protocol 244 semantics differ")
    final_arrays = load_npz(archive_path)
    if set(final_arrays) != set(result.get("array_records", {})) or any(
        raw_array_record(value) != result["array_records"][name]
        for name, value in final_arrays.items()
    ):
        raise Protocol245Error("Protocol 244 final array records differ")

    states = {}
    profiles = {}
    leaves = {}
    for step in range(33, 49):
        stem = f"G10_full_step{step:04d}"
        receipt_path = P244_OUTPUT / f"{stem}.json"
        state_path = P244_OUTPUT / f"{stem}.npz"
        receipt = read_json(receipt_path)
        if not (
            check_fingerprint(receipt, b"protocol244-checkpoint-v1\0")
            and receipt.get("schema") == "protocol244-full-dt-g10-checkpoint-v1"
            and receipt.get("authority_sha256") == P244_AUTHORITY_SHA
            and receipt.get("end_step") == step
            and receipt.get("end_time") == step * FULL_DT
            and receipt.get("dt") == FULL_DT
            and receipt.get("passed") is True
            and receipt.get("endpoint_repeat_exact") is True
            and receipt.get("archive", {}).get("byte_count") == state_path.stat().st_size
            and receipt.get("archive", {}).get("sha256") == sha256(state_path)
        ):
            raise Protocol245Error(f"Protocol 244 checkpoint differs: {step}")
        arrays = load_npz(state_path)
        if set(arrays) != set(FIELDS) or any(
            value.dtype != np.float64
            or value.shape != (129, 241, 9 if name in {"q", "v"} else 3)
            or not np.all(np.isfinite(value))
            or raw_array_record(value) != receipt.get("arrays", {}).get(name)
            for name, value in arrays.items()
        ):
            raise Protocol245Error(f"Protocol 244 checkpoint arrays differ: {step}")
        states[step] = arrays

    for step in LEAF_STEPS:
        stem = f"G10_outer_step{step:04d}"
        receipt_path = P244_OUTPUT / f"{stem}.json"
        profile_path = P244_OUTPUT / f"{stem}.npz"
        receipt = read_json(receipt_path)
        if not (
            check_fingerprint(receipt, b"protocol244-outer-leaf-v1\0")
            and receipt.get("schema") == "protocol244-full-dt-g10-outer-leaf-v1"
            and receipt.get("authority_sha256") == P244_AUTHORITY_SHA
            and receipt.get("step") == step
            and receipt.get("time_over_ell") == step * FULL_DT
            and receipt.get("passed") is True
            and receipt.get("solve_repeat_exact") is True
            and receipt.get("archive", {}).get("byte_count") == profile_path.stat().st_size
            and receipt.get("archive", {}).get("sha256") == sha256(profile_path)
            and receipt.get("evaluation") == scientific["evaluation"]["leaves"][str(step)]
        ):
            raise Protocol245Error(f"Protocol 244 leaf differs: {step}")
        arrays = load_npz(profile_path)
        if set(arrays) != {"theta", "rho", "slope"} or any(
            value.dtype != np.float64 or value.shape != (501,) or not np.all(np.isfinite(value))
            or raw_array_record(value) != receipt.get("profile_arrays", {}).get(name)
            for name, value in arrays.items()
        ):
            raise Protocol245Error(f"Protocol 244 profile arrays differ: {step}")
        profiles[step] = arrays
        leaves[step] = receipt["evaluation"]
    if not all(np.array_equal(profiles[LEAF_STEPS[0]]["theta"], profiles[step]["theta"]) for step in LEAF_STEPS[1:]):
        raise Protocol245Error("Protocol 244 theta grids differ")
    return result, states, profiles, leaves


def load_background():
    record = read_json(BACKGROUND_PATH)
    background = record.get("background")
    keys = {
        "beta_b", "mass_squared", "retuned_bare_tension_b", "v1",
        "wall_potential_b", "wall_stiffness",
    }
    if not isinstance(background, dict) or set(background) != keys:
        raise Protocol245Error("historical background schema differs")
    background = {name: float(value) for name, value in background.items()}
    if not all(math.isfinite(value) for value in background.values()) or not (
        background["mass_squared"] == 0.41 and background["wall_stiffness"] == 20.0
        and record.get("classification") == "complete_balance_terms_without_sealed_closure_pass"
        and record.get("status") == "REVIEW"
    ):
        raise Protocol245Error("historical background semantics differ")
    return background


def manufactured_controls():
    balance = balance_analytic_controls()
    coupled = coupled_analytic_controls()
    stationary = normalized_ledger_residual(0.0, {"motion": 0.0, "seam": 0.0})
    return {
        "balance": balance,
        "coupled": coupled,
        "stationary_duplicate": stationary,
        "passed": bool(
            balance.get("passed") is True
            and coupled.get("passed") is True
            and stationary["normalized_absolute_residual"] < 1e-10
        ),
    }


def generator_fields(profile, rho_rate, prepared):
    theta = profile["theta"]
    rho = profile["rho"]
    slope = profile["slope"]
    sine = np.sin(theta)
    cosine = np.cos(theta)
    zcoord = prepared.z[-1] - rho * cosine
    radius = rho * sine
    tangent_coordinate = np.stack((rho * sine - slope * cosine, rho * cosine + slope * sine), axis=1)
    embedding_velocity = np.stack((-rho_rate * cosine, rho_rate * sine), axis=1)
    metric = _sample(prepared, "base_metric", zcoord, radius)
    lapse = _sample(prepared, "lapse", zcoord, radius)
    shift = _sample(prepared, "shift", zcoord, radius)
    shift_covector = _sample(prepared, "shift_covector", zcoord, radius)
    decomposition = null_decomposition(lapse, shift, metric, embedding_velocity, tangent_coordinate)
    projector = projected_norm(lapse, shift_covector, shift, metric, embedding_velocity, tangent_coordinate)
    tangent_speed = np.sqrt(np.einsum("...a,...ab,...b->...", tangent_coordinate, metric, tangent_coordinate))
    tangent = tangent_coordinate / tangent_speed[:, None]
    tangent_covector = np.einsum("...ab,...b->...a", metric, tangent)
    tangential_speed = np.einsum("...a,...a->...", tangent_covector, shift + embedding_velocity)
    return {
        **decomposition,
        "projector": projector,
        "projector_relative_error": relative_scale_error(decomposition["norm"], projector),
        "lapse": lapse,
        "tangential_speed": tangential_speed,
        "endpoint_coordinate_z_rate": float(embedding_velocity[-1, 0]),
    }


def endpoint_geometry(q, profile, r):
    radius = float(profile["rho"][-1])
    slope = float(profile["slope"][-1])
    if not (0.0 < radius < r[-1] and math.isfinite(slope)):
        raise Protocol245Error("invalid endpoint geometry")
    hzz = float(CubicSpline(r, q[-1, :, 6])(radius))
    hzr = float(CubicSpline(r, r * q[-1, :, 1])(radius))
    hrr = float(CubicSpline(r, q[-1, :, 3] + r**2 * q[-1, :, 4])(radius))
    tangent = np.asarray([radius, slope], dtype=float)
    metric = np.asarray([[hzz, hzr], [hzr, hrr]], dtype=float)
    speed = float(math.sqrt(tangent @ metric @ tangent))
    return radius, tangent, speed


def native_coefficient(q, profile, z, r, operators):
    radius, tangent, speed = endpoint_geometry(q, profile, r)
    transverse = np.asarray(q[:, :, 3], dtype=float)
    derivative_z_wall = np.asarray(operators["Dz"].getrow(-1) @ transverse).reshape(-1)
    derivative_r_wall = np.asarray(operators["Dr"] @ transverse[-1, :]).reshape(-1)
    value = float(CubicSpline(r, transverse[-1, :])(radius))
    derivative_z = float(CubicSpline(r, derivative_z_wall)(radius))
    derivative_r = float(CubicSpline(r, derivative_r_wall)(radius))
    root = math.sqrt(value)
    sphere = radius * root
    gradient = np.asarray([
        radius * 0.5 * derivative_z / root,
        root + radius * 0.5 * derivative_r / root,
    ])
    result = float((tangent @ gradient) / speed / sphere)
    if not math.isfinite(result):
        raise Protocol245Error("nonfinite native seam coefficient")
    return result


def direct_wall_rate(
    state, z, r, profile, rho_rate, slope_rate, width, operators,
    rejected_formula, wall, history,
):
    base = native_coefficient(state["q"], profile, z, r, operators)
    epsilon_records = {}
    estimates = []
    for epsilon in EPSILONS:
        plus_profile = {
            "theta": profile["theta"],
            "rho": profile["rho"] + epsilon * rho_rate,
            "slope": profile["slope"] + epsilon * slope_rate,
        }
        minus_profile = {
            "theta": profile["theta"],
            "rho": profile["rho"] - epsilon * rho_rate,
            "slope": profile["slope"] - epsilon * slope_rate,
        }
        plus_state = {"q": state["q"] + epsilon * state["v"], "v": state["v"]}
        minus_state = {"q": state["q"] - epsilon * state["v"], "v": state["v"]}
        plus = native_coefficient(plus_state["q"], plus_profile, z, r, operators)
        minus = native_coefficient(minus_state["q"], minus_profile, z, r, operators)
        estimate = centered_directional(plus, minus, epsilon)
        epsilon_records[f"{epsilon:.1e}"] = {
            "c_plus": plus,
            "c_minus": minus,
            "directional_rate": estimate,
            "wall_pass": inherited_rate_pass(estimate, wall),
            "history_pass": inherited_rate_pass(estimate, history),
        }
        estimates.append(estimate)
    stable = all(
        inherited_rate_pass(estimates[left], estimates[right])
        for left, right in ((0, 1), (1, 2), (0, 2))
    )
    passed = bool(
        stable
        and all(item["wall_pass"] and item["history_pass"] for item in epsilon_records.values())
    )
    return {
        "unperturbed_c_geometric": base,
        "surface_variation_rate_rejected_non_authoritative": float(rejected_formula),
        "wall_rate": float(wall),
        "history_rate": float(history),
        "epsilon_records": epsilon_records,
        "epsilon_stability_pass": stable,
        "passed": passed,
    }, np.asarray(estimates, dtype=np.float64)


def evaluate(states, profiles, leaves, background):
    sample = states[LEAF_STEPS[0]]["q"]
    if sample.ndim != 3 or sample.shape[2] != 9:
        raise Protocol245Error("state shape differs")
    z = np.linspace(1.0, math.e, sample.shape[0])
    r = np.linspace(0.0, 10.0, sample.shape[1])
    spacing = FULL_DT
    operators = {
        width: {"Dz": derivative_matrix(z, 1, width), "Dr": derivative_matrix(r, 1, width)}
        for width in WIDTHS
    }
    prepared = {
        (step, width): prepare_capped_expansion_slice(
            states[step]["q"], states[step]["v"], z, r, stencil_width=width,
        )
        for step in LEAF_STEPS for width in WIDTHS
    }
    charges = {width: {} for width in WIDTHS}
    c_history = {width: {} for width in WIDTHS}
    theta_l = {width: {} for width in WIDTHS}
    zero_rate = np.zeros(501, dtype=np.float64)
    for width in WIDTHS:
        for step in LEAF_STEPS:
            charge = reflected_cap_charge(
                states[step]["q"], states[step]["v"], z, r, profiles[step],
                stencil_width=width, prepared=prepared[(step, width)],
            )
            charges[width][step] = charge
            c_history[width][step] = native_coefficient(
                states[step]["q"], profiles[step], z, r, operators[width],
            )
            fields = leaf_marginal_transport_fields(
                states[step]["q"], states[step]["v"], z, r,
                profiles[step], zero_rate, prepared=prepared[(step, width)],
            )
            theta_l[width][step] = np.asarray(fields["theta_l"], dtype=np.float64)

    area_values = {step: float(leaves[step]["geometry"]["one_sided_cap_area"]) for step in LEAF_STEPS}
    step_records = {}
    arrays = {}
    all_orientation = True
    all_area = True
    all_seam = True
    all_wall = True
    all_ledger = True
    centered_residuals = []
    centered_area_errors = []
    wall_rates = []

    for step in INTERIOR_STEPS:
        early = step - 1
        late = step + 1
        profile = profiles[step]
        rho_rates = three_point_rates(
            profiles[early]["rho"], profile["rho"], profiles[late]["rho"], spacing,
        )
        slope_rates = three_point_rates(
            profiles[early]["slope"], profile["slope"], profiles[late]["slope"], spacing,
        )
        area_rates = {
            name: float(value) for name, value in three_point_rates(
                np.asarray(area_values[early]), np.asarray(area_values[step]),
                np.asarray(area_values[late]), spacing,
            ).items()
        }
        charge_rates = {
            width: {
                name: float(value) for name, value in three_point_rates(
                    np.asarray(charges[width][early]["generalized_hawking_ads_charge_kappa5_squared_E"]),
                    np.asarray(charges[width][step]["generalized_hawking_ads_charge_kappa5_squared_E"]),
                    np.asarray(charges[width][late]["generalized_hawking_ads_charge_kappa5_squared_E"]),
                    spacing,
                ).items()
            }
            for width in WIDTHS
        }
        c_rates = {
            width: {
                name: float(value) for name, value in three_point_rates(
                    np.asarray(c_history[width][early]), np.asarray(c_history[width][step]),
                    np.asarray(c_history[width][late]), spacing,
                ).items()
            }
            for width in WIDTHS
        }
        theta_l_rates = {
            width: (theta_l[width][late] - theta_l[width][early]) / (2.0 * spacing)
            for width in WIDTHS
        }
        path_orientation = {}
        for name in PATHS:
            fields = generator_fields(profile, rho_rates[name], prepared[(step, 7)])
            orientation = {
                "minimum_lapse": float(np.min(fields["lapse"])),
                "minimum_normal_speed": float(np.min(fields["normal_speed"])),
                "minimum_A": float(np.min(fields["A"])),
                "minimum_B": float(np.min(fields["B"])),
                "minimum_norm": float(np.min(fields["norm"])),
                "projector_relative_scale_error": float(fields["projector_relative_error"]),
                "endpoint_coordinate_z_rate": fields["endpoint_coordinate_z_rate"],
                "endpoint_V_dot_nu": float(fields["tangential_speed"][-1]),
            }
            orientation["passed"] = bool(
                orientation["minimum_lapse"] > 0.0
                and orientation["minimum_normal_speed"] > orientation["minimum_lapse"]
                and orientation["minimum_A"] > 0.0
                and orientation["minimum_B"] > 0.0
                and orientation["minimum_norm"] > 0.0
                and orientation["projector_relative_scale_error"] < 1e-12
                and abs(orientation["endpoint_coordinate_z_rate"]) < 1e-10
                and abs(orientation["endpoint_V_dot_nu"]) < 1e-10
            )
            path_orientation[name] = orientation
            for key in ("A", "B", "norm", "projector", "tangential_speed"):
                arrays[f"step{step:04d}_{name}_{key}"] = np.asarray(fields[key])

        width_records = {}
        for width in WIDTHS:
            width_records[str(width)] = {}
            for name in PATHS:
                middle_prepared = prepared[(step, width)]
                generator = generator_fields(profile, rho_rates[name], middle_prepared)
                transport = leaf_marginal_transport_fields(
                    states[step]["q"], states[step]["v"], z, r,
                    profile, rho_rates[name], prepared=middle_prepared,
                )
                marginal_rate = float(_one_cap_integral(
                    -generator["B"] * transport["theta_n"],
                    transport["sphere_radius"], transport["speed"], transport["theta"],
                ))
                finite_rate = float(_one_cap_integral(
                    generator["A"] * transport["theta_l"] - generator["B"] * transport["theta_n"],
                    transport["sphere_radius"], transport["speed"], transport["theta"],
                ))
                boundary_rate = float(
                    4.0 * math.pi * transport["sphere_radius"][-1] ** 2 * generator["tangential_speed"][-1]
                )
                area_error = relative_scale_error(finite_rate + boundary_rate, area_rates[name], 1e-12)

                balance = evaluate_balance_leaf(
                    states[step]["q"], states[step]["v"], z, r, profile,
                    rho_rates[name], background, stencil_width=width, prepared=middle_prepared,
                )
                balance.update({
                    "grid": GRID_LABEL, "branch": "outer", "stride": PATHS.index(name),
                    "time": step * FULL_DT,
                    "charge_rate_target": {
                        "charge": float(charges[width][step]["generalized_hawking_ads_charge_kappa5_squared_E"]),
                        "finite_difference_rate": charge_rates[width][name],
                    },
                })
                endpoint = seam_endpoint_transport(
                    states[step]["q"], states[step]["v"], z, r, profile,
                    rho_rates[name], background["wall_stiffness"], background["v1"],
                )
                bulk = evaluate_geometric_bulk_leaf(
                    states[step]["q"], states[step]["v"], z, r, profile,
                    rho_rates[name], balance, prepared=middle_prepared,
                )
                rejected_formula = float(
                    bulk["H_Omega_s_seam"]
                    - balance["seam"]["geometric_Ws_over_W"] * bulk["H_meridional_seam"]
                )
                physical = physical_coupled_record(balance, endpoint, rejected_formula)
                physical = apply_intrinsic_anisotropy(physical, bulk)
                tangency = marginal_tangency_rate(
                    theta_l_rates[width], transport, charges[width][step]["equivalent_area_radius"],
                )
                ledger_terms = dict(physical["corrected_rates"])
                ledger_terms["marginal_tangency_defect"] = float(tangency["hawking_product_derivative_term"])
                ledger = normalized_ledger_residual(charge_rates[width][name], ledger_terms)

                c_geom = native_coefficient(
                    states[step]["q"], profile, z, r, operators[width],
                )
                c_wall = float(balance["seam"]["israel_coefficient_c"])
                seam_magnitude_error = relative_scale_error(c_geom, c_wall, 1e-12)
                intrinsic_error = float(balance["seam"]["israel_intrinsic_relative_scale_error"])
                coupled_error = float(physical["uncombined"]["compatibility_error"])
                wall_record, directional_rates = direct_wall_rate(
                    states[step], z, r, profile, rho_rates[name], slope_rates[name], width,
                    operators[width], rejected_formula,
                    float(endpoint["wall_israel_coefficient_rate"]), c_rates[width][name],
                )
                arrays[f"step{step:04d}_w{width}_{name}_directional_rates"] = directional_rates
                record = {
                    "velocity_path": name,
                    "stencil_width": width,
                    "orientation": path_orientation[name],
                    "area_transport": {
                        "finite_difference_rate": area_rates[name],
                        "marginal_integral_rate": marginal_rate,
                        "finite_theta_l_integral_rate": finite_rate,
                        "boundary_V_dot_nu_rate": boundary_rate,
                        "relative_scale_error": area_error,
                    },
                    "seam": {
                        "c_geometric": c_geom,
                        "c_wall": c_wall,
                        "magnitude_relative_scale_error": seam_magnitude_error,
                        "intrinsic_integral_relative_scale_error": intrinsic_error,
                        "coupled_identity_relative_scale_error": coupled_error,
                        "magnitude_pass": bool(seam_magnitude_error < 0.01 and intrinsic_error < 0.01),
                        "coupled_identity_pass": bool(coupled_error < 2e-4),
                    },
                    "native_wall_rate": wall_record,
                    "tangency": tangency,
                    "ledger": ledger,
                    "finite": bool(
                        balance["finite"] and endpoint["finite"] and bulk["finite"]
                        and physical["finite"] and tangency["finite"]
                    ),
                }
                if not record["finite"]:
                    raise Protocol245Error(f"nonfinite physical record: {step}/{width}/{name}")
                width_records[str(width)][name] = record
                wall_rates.extend(float(item["directional_rate"]) for item in wall_record["epsilon_records"].values())

        orientation_pass = bool(all(item["passed"] for item in path_orientation.values()))
        area_pass = bool(
            all(width_records[str(width)]["centered"]["area_transport"]["relative_scale_error"] < 0.02 for width in WIDTHS)
            and all(
                width_records[str(width)][name]["area_transport"]["finite_difference_rate"] > 0.0
                and width_records[str(width)][name]["area_transport"]["finite_theta_l_integral_rate"] > 0.0
                for width in WIDTHS for name in ("backward", "forward")
            )
        )
        seam_pass = bool(all(
            width_records[str(width)][name]["seam"]["magnitude_pass"]
            and width_records[str(width)][name]["seam"]["coupled_identity_pass"]
            for width in WIDTHS for name in PATHS
        ))
        wall_pass = bool(all(
            width_records[str(width)][name]["native_wall_rate"]["passed"]
            for width in WIDTHS for name in PATHS
        ))
        ledger_pass = bool(all(
            width_records[str(width)]["centered"]["ledger"]["normalized_absolute_residual"] < 0.01
            for width in WIDTHS
        ) and all(
            width_records[str(width)][name]["ledger"]["target_rate"] != 0.0
            and np.sign(width_records[str(width)][name]["ledger"]["target_rate"])
            == np.sign(width_records[str(width)][name]["ledger"]["total_flux"])
            for width in WIDTHS for name in PATHS
        ))
        gates = {
            "orientation": orientation_pass,
            "area": area_pass,
            "seam": seam_pass,
            "native_wall_rate": wall_pass,
            "flux_ledger": ledger_pass,
        }
        step_records[str(step)] = {
            "step": step,
            "time_over_ell": step * FULL_DT,
            "neighbor_spacing_over_ell": spacing,
            "gates": gates,
            "classification": classify(gates),
            "area_values": {str(value): area_values[value] for value in (early, step, late)},
            "area_rates": area_rates,
            "records": width_records,
        }
        all_orientation = all_orientation and orientation_pass
        all_area = all_area and area_pass
        all_seam = all_seam and seam_pass
        all_wall = all_wall and wall_pass
        all_ledger = all_ledger and ledger_pass
        centered_residuals.extend(
            width_records[str(width)]["centered"]["ledger"]["normalized_absolute_residual"]
            for width in WIDTHS
        )
        centered_area_errors.extend(
            width_records[str(width)]["centered"]["area_transport"]["relative_scale_error"]
            for width in WIDTHS
        )

    gates = {
        "orientation": bool(all_orientation),
        "area": bool(all_area),
        "seam": bool(all_seam),
        "native_wall_rate": bool(all_wall),
        "flux_ledger": bool(all_ledger),
    }
    return {
        "classification": classify(gates),
        "gates": gates,
        "leaf_steps": list(LEAF_STEPS),
        "interior_steps": list(INTERIOR_STEPS),
        "interior_leaf_count": len(INTERIOR_STEPS),
        "maximum_centered_area_relative_error": float(max(centered_area_errors)),
        "maximum_centered_flux_ledger_residual": float(max(centered_residuals)),
        "minimum_native_directional_wall_rate": float(min(wall_rates)),
        "maximum_native_directional_wall_rate": float(max(wall_rates)),
        "steps": step_records,
    }, arrays


def execute():
    started = time.monotonic()
    authority = verify_freeze(ROOT)
    validate_p243()
    _, states, profiles, leaves = validate_p244()
    background = load_background()
    controls = manufactured_controls()
    if controls.get("passed") is not True:
        raise Protocol245Error("manufactured controls failed")
    first, first_arrays = evaluate(states, profiles, leaves, background)
    second, second_arrays = evaluate(states, profiles, leaves, background)
    if canonical(first) != canonical(second) or set(first_arrays) != set(second_arrays) or any(
        not np.array_equal(first_arrays[name], second_arrays[name]) for name in first_arrays
    ):
        raise Protocol245Error("complete physical repeat differs")
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise Protocol245Error("candidate-output must be absent")
    OUTPUT.mkdir()
    fsync_directory(ROOT)
    archive_path = OUTPUT / "protocol245_dense_balance_arrays.npz"
    atomic_npz(archive_path, first_arrays)
    result = {
        "schema": SCHEMA,
        "classification": first["classification"],
        "authority_sha256": sha256(ROOT / "authority/freeze_record.json"),
        "protocol244_result_sha256": P244_RESULT_SHA,
        "protocol243_result_sha256": P243_RESULT_SHA,
        "manufactured_controls": controls,
        "complete_physical_repeat_exact": True,
        "scientific": first,
        "archive": file_record(archive_path, ROOT),
        "array_records": {name: raw_array_record(value) for name, value in sorted(first_arrays.items())},
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
            "elapsed_wall_seconds": float(time.monotonic() - started),
            "peak_rss_bytes": current_peak_rss_bytes(),
        },
        "spacetime_evolution_executed": False,
        "surface_solve_executed": False,
        "parent_or_checkpoint_modified": False,
        "submitted_paper_edited": False,
        "continuum_dynamical_horizon_claim_authorized": False,
        "integrated_or_global_balance_claim_authorized": False,
        "event_horizon_claim_authorized": False,
        "source_ownership_claim_authorized": False,
        "full_half_native_balance_comparison_authorized": bool(
            first["classification"] == "FULL-DT-DENSE-NATIVE-OPERATOR-LOCAL-BALANCE-PASS"
        ),
        "temporal_balance_closure_established": False,
    }
    result["fingerprint"] = hashlib.sha256(b"protocol245-result-v1\0" + canonical(result)).hexdigest()
    atomic_json(OUTPUT / "protocol245_result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--run", action="store_true")
    arguments = parser.parse_args()
    if arguments.verify == arguments.run:
        parser.error("select exactly one mode")
    if arguments.verify:
        verify_freeze(ROOT)
        validate_p243()
        validate_p244()
        load_background()
        print(json.dumps({
            "status": "VERIFIED",
            "authority_sha256": sha256(ROOT / "authority/freeze_record.json"),
        }, sort_keys=True))
    else:
        print(json.dumps(execute(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
