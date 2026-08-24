#!/usr/bin/env python3
"""Sealed, restartable Test-2B invariant-coordinate convergence audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.interpolate import PchipInterpolator, RectBivariateSpline

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import run_corrected_fold_live_nonlinear_gauge_source as live
from bhps.corrected_A790_physical_tensor_convergence import (
    adm_extrinsic_curvature_tensor,
    spatial_metric_tensor,
)
from bhps.dynamical_capped_geometry import capped_surface_geometry
from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice
from bhps.dynamical_capped_horizon_bvp import (
    _sample_geometry,
    dynamical_rho_second,
    local_outgoing_expansion,
    solve_dynamical_capped_surface_bvp,
)
from bhps.dynamical_mots_stability import mots_stability_matrix, public_stability
from bhps.invariant_physical_chart import (
    NormalGeodesicChart,
    build_normal_geodesic_chart,
    chart_validity,
    common_areal_interval,
    conservative_order_interval,
    generalized_order,
    interpolate_regular_field,
    inverse_chart_at,
    mapped_extrinsic_fields,
    mapped_metric_fields,
    native_to_proper_distance,
    sign_coherence,
    weighted_l2,
    weighted_quantile,
)
from bhps.recovery_indexer import (
    RecoveryIndex,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
    validate_npz,
)
from run_corrected_A790_dynamic_MOTS_stability import (
    analytic_control as stability_analytic_control,
    surface_passes,
)
from run_corrected_A790_test10_joint_convergence import build_all_geometries


PROTOCOL = Path("notes/107_A790_test2B_invariant_physical_coordinate_convergence_protocol.md")
PROTOCOL_SHA256 = "e1a13f613fc530a811c88fcf5dbe4fb061f88bbfa66f2c1db05c4fcf7638485c"
OUTPUT = Path("results/corrected_A790_test2b_invariant_convergence.json")
STATE_OUTPUT = Path("results/corrected_A790_test2b_invariant_convergence_state.npz")
RECOVERY_ROOT = Path("results/corrected_A790_test2b_invariant_convergence_recovery")
MANIFEST = RECOVERY_ROOT / "index.json"
G78_STATE = Path("results/corrected_A790_formation_time_refinement_state.npz")
G9_STATE = Path("results/corrected_A790_third_grid_formation_reproduction_state.npz")
G10_STATE = Path("results/corrected_A790_fourth_grid_physical_tensor_convergence_state.npz")
G10_TEST_STATE = Path("results/corrected_A790_test10_joint_convergence_state.npz")
LONG_STATE = Path("results/corrected_A790_t008_long_evolution_state.npz")
COMMON_PARENT_STATE = Path("results/corrected_A790_test10b_domain_normalized_state.npz")
TEST10_RECOVERY = Path("results/corrected_A790_test10_joint_convergence_recovery")
STANDARD_GRIDS = ("G7", "G8", "G9", "G10", "G11")
PRIMARY_GRIDS = ("G9", "G10", "G11")
BRANCH_SEEDS = {"inner": 1.30, "outer": 1.55}
MAP_RESOLUTIONS = {
    "coarse": (129, 193), "primary": (193, 257), "fine": (257, 385),
}

FIXED_INPUT_HASHES = {
    "notes/84_A790_dynamic_MOTS_stability_protocol.md": "34f719852df2ef4b007c2ef0d6ba8d57a4e79a9dbff9b605e03c0408dde4446b",
    "notes/89_A790_fourth_grid_physical_tensor_convergence_protocol.md": "f3edbca20287805f95810ad25b6aa59a814136eea0491ca59721180d49c9f0d3",
    "notes/93_A790_publication_joint_convergence_protocol.md": "c27ec08d2b8d7332e6d09601f5e73450268f5374d68832fb6c0d012532af9b01",
    "notes/93_A790_publication_joint_convergence_result.md": "ffd3e6182044109c075e8c8fbaa1f16f48fc42c7eb1c3807f9f047b12e4686bb",
    "notes/98_A790_domain_normalized_family_protocol.md": "7c0298c67d03654b2e24c7c8602a77a8c70bd75beba14645fdd76adfef11cc23",
    "notes/98_A790_domain_normalized_family_result.md": "618b72fb125ddc67c8ea523789687a4836216fd52bda26e59223c8163a4e42d0",
    str(G78_STATE): "845ead50eb5e336b0e9a5ad2357016147cdfc199df5c05d460d54d06ddd1c038",
    str(G9_STATE): "9867e8636847dbf351d9f18d0e6516d2e234b6e2b86109a8f14bab52db85380e",
    str(LONG_STATE): "ad7dba798550f6499dbb966833371e6f1be59477d04f53dd683a96d7fc5c24c8",
    str(G10_STATE): "fa9a9c2833d02132f096b3c7f43d7457edd6ddf375ffb534b7a44aa83b0a095b",
    str(G10_TEST_STATE): "cd3e6d9f6d4bf543390c7272523bcced7c6ef0c8f9a57dace35b2f1389e947d4",
    str(COMMON_PARENT_STATE): "b19e7e8901fdade6abb485c65a3512bf327a86f13974fb9a8c0876340a503928",
    "results/corrected_A790_formation_time_refinement.json": "57cd14b9712907c23e09b8fc2cc79de738c0d963353a27b9d2ea53d37a86d010",
    "results/corrected_A790_third_grid_formation_reproduction.json": "fe3607d0fad311b81dd91d3592a3fc3eaed2ed20a3e3fccde5022f7fdb06682b",
    "results/corrected_A790_fourth_grid_physical_tensor_convergence.json": "03d9f7ff4496ad430ecf368452dfe8ce911fda94a5d594269abec57ebd177b98",
    "results/corrected_A790_test10_joint_convergence.json": "da8c375216b4ca7e42529c711a403b0a2210f9a415df6200629098191ea758bf",
    "results/corrected_A790_test10b_domain_normalized.json": "08c91892930c05fef6afe37d0a4f5ed1c49f545a84b6df4c81e08c8f2cbd1bd3",
}


def recovery_inputs():
    dynamic = (
        Path(__file__), Path("src/bhps/invariant_physical_chart.py"),
        Path("src/bhps/corrected_A790_physical_tensor_convergence.py"),
        Path("src/bhps/dynamical_mots_stability.py"),
        Path("src/bhps/recovery_indexer.py"),
    )
    return {**FIXED_INPUT_HASHES, **{str(path): sha256_file(path) for path in dynamic}}


def stage_json(index, stage_id, filename, kind, metadata, producer, expected=900.0):
    path = RECOVERY_ROOT / filename
    index.register(stage_id, kind, expected, metadata)
    cached = index.validated_path(stage_id)
    if cached is not None:
        payload = json.loads(cached.read_text())
        if payload.get("protocol_sha256") != index.protocol_sha256:
            raise RuntimeError(f"protocol mismatch in {stage_id}")
        return payload, True
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        payload = {"stage_id": stage_id, "protocol_sha256": index.protocol_sha256, **producer()}
        atomic_write_json(path, payload)
        checked = json.loads(path.read_text())
        if checked.get("stage_id") != stage_id:
            raise RuntimeError(f"stage validation failed for {stage_id}")
        index.mark_complete(stage_id, path, time.perf_counter() - started)
        return payload, False
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def stage_npz(index, stage_id, filename, kind, metadata, producer, expected=900.0):
    path = RECOVERY_ROOT / filename
    index.register(stage_id, kind, expected, metadata)
    cached = index.validated_path(stage_id)
    if cached is not None:
        return cached, True
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        arrays = producer()
        atomic_write_npz(path, **arrays)
        validate_npz(path)
        index.mark_complete(stage_id, path, time.perf_counter() - started)
        return path, False
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def recovery_summary(index):
    stages = index.data.get("stages", {})
    counts = {}
    for record in stages.values():
        status = record.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return {
        "manifest": str(index.path), "protocol_sha256": index.protocol_sha256,
        "stage_count": len(stages), "status_counts": counts,
        "completed_bytes": int(sum(
            record.get("byte_count", 0) for record in stages.values()
            if record.get("status") == "complete"
        )),
    }


def flat_metric(z, r):
    metric = np.zeros((len(z), len(r), 2, 2))
    metric[..., 0, 0] = 1.0
    metric[..., 1, 1] = 1.0
    return metric, np.ones((len(z), len(r)))


def manufactured_controls():
    z = np.linspace(0.0, 1.0, 49)
    r = np.linspace(0.0, 2.0, 65)
    distance = np.linspace(0.0, 0.75, 97)
    metric, sphere = flat_metric(z, r)
    flat = build_normal_geodesic_chart(z, r, metric, sphere, distance, ray_count=129)
    flat_validity = chart_validity(flat)
    target_r = np.linspace(0.08, 1.8, 101)
    native_z, native_r = inverse_chart_at(flat, distance, target_r)
    flat_fields = mapped_metric_fields(
        metric, sphere, z, r, distance, target_r, native_z, native_r,
    )
    flat_error = float(max(
        np.max(np.abs(flat_fields["q_DD"] - 1.0)),
        np.max(np.abs(flat_fields["q_DR"])),
        np.max(np.abs(flat_fields["q_RR"] - 1.0)),
    ))

    # Smooth native-coordinate deformation of the same Euclidean geometry.
    dy_dz = 1.0 + 0.3 * z
    dR_dr = 1.0 + 0.2 * r
    physical_r = r + 0.1 * r**2
    deformed_metric = np.zeros_like(metric)
    deformed_metric[..., 0, 0] = dy_dz[:, None]**2
    deformed_metric[..., 1, 1] = dR_dr[None, :]**2
    sphere_line = np.ones_like(r)
    sphere_line[1:] = (physical_r[1:] / r[1:])**2
    deformed_sphere = np.broadcast_to(sphere_line, sphere.shape).copy()
    deformed = build_normal_geodesic_chart(
        z, r, deformed_metric, deformed_sphere, distance, ray_count=129,
    )
    deformed_validity = chart_validity(deformed)
    high = min(float(np.min(deformed.areal_radius[:, -1])), 1.8)
    deformed_target = np.linspace(0.08, high, 101)
    dzmap, drmap = inverse_chart_at(deformed, distance, deformed_target)
    deformed_fields = mapped_metric_fields(
        deformed_metric, deformed_sphere, z, r, distance, deformed_target,
        dzmap, drmap,
    )
    deformation_error = float(max(
        np.max(np.abs(deformed_fields["q_DD"][2:-2, 2:-2] - 1.0)),
        np.max(np.abs(deformed_fields["q_DR"][2:-2, 2:-2])),
        np.max(np.abs(deformed_fields["q_RR"][2:-2, 2:-2] - 1.0)),
    ))

    counts = np.asarray((112.0, 128.0, 144.0))
    first = counts[:-1]**-1 - counts[1:]**-1
    second = counts[:-1]**-2 - counts[1:]**-2
    order_one = generalized_order(*first)
    order_two = generalized_order(*second)
    interval = conservative_order_interval(
        second[0], second[0] * 1e-5, second[1], second[1] * 1e-5,
    )
    temporal_second = math.log2((0.25**2 - 0.125**2) / (0.125**2 - 0.0625**2))
    stability = stability_analytic_control()

    recovery_ok = False
    with tempfile.TemporaryDirectory() as folder:
        good = Path(folder) / "good.npz"
        atomic_write_npz(good, sample=np.arange(8.0))
        validate_npz(good, {"sample": (8,)})
        bad = Path(folder) / "bad.npz"
        bad.write_bytes(good.read_bytes()[: max(1, good.stat().st_size // 2)])
        try:
            validate_npz(bad, {"sample": (8,)})
        except Exception:
            recovery_ok = True

    passed = bool(
        flat_validity["valid"] and deformed_validity["valid"]
        and flat_error < 1e-10 and deformation_error < 5e-4
        and abs(order_one - 1.0) < 1e-6 and abs(order_two - 2.0) < 1e-6
        and interval is not None and interval[0] > 1.99
        and abs(temporal_second - 2.0) < 1e-12
        and stability["passed"] and recovery_ok
    )
    return {
        "passed": passed,
        "flat_chart": flat_validity,
        "flat_mapped_metric_maximum_error": flat_error,
        "deformed_chart": deformed_validity,
        "coordinate_deformation_maximum_error": deformation_error,
        "manufactured_orders": {"first": order_one, "second": order_two},
        "second_order_uncertainty_interval": interval,
        "temporal_second_order": temporal_second,
        "stability_control": stability,
        "corruption_rejected": recovery_ok,
    }


def geometry_archive(geometries):
    arrays = {}
    for label in STANDARD_GRIDS:
        geometry = geometries[label]
        arrays[f"{label}_z"] = np.asarray(geometry["z"])
        arrays[f"{label}_r"] = np.asarray(geometry["r"])
        arrays[f"{label}_initial"] = np.asarray(geometry["jet_field"].reduced_fields)
    return arrays


def _checkpoint_history(run_label, steps, initial):
    positions = [np.asarray(initial)]
    velocities = [np.zeros_like(initial)]
    for step in range(1, steps + 1):
        segment = 2 if run_label == "G10_coarse" else 4
        start = ((step - 1) // segment) * segment + 1
        end = min(start + segment - 1, steps)
        path = TEST10_RECOVERY / f"evolution_{run_label}_steps_{start:03d}_{end:03d}.npz"
        with np.load(path) as archive:
            positions.append(initial + archive[f"step_{step:03d}_increment"])
            velocities.append(archive[f"step_{step:03d}_velocity"])
    return np.asarray(positions), np.asarray(velocities)


def state_histories(initials):
    histories = {}
    with np.load(LONG_STATE) as archive:
        for label in ("G7", "G8"):
            histories[label] = {
                "time": np.asarray(archive["times"][:9]),
                "position": np.asarray(archive[f"{label}_position_history"][:9]),
                "velocity": np.asarray(archive[f"{label}_velocity_history"][:9]),
            }
    for label, path, time_key in (
        ("G9", G9_STATE, "times"), ("G10", G10_STATE, "times"),
    ):
        with np.load(path) as archive:
            times = np.concatenate(([0.0], np.asarray(archive[time_key])))
            position = [initials[label]]
            velocity = [np.zeros_like(initials[label])]
            for index in range(7):
                position.append(initials[label] + archive[f"{label}_time_{index}_increment"])
                velocity.append(archive[f"{label}_time_{index}_velocity"])
            position.append(initials[label] + archive[f"{label}_8step_increment"])
            velocity.append(archive[f"{label}_8step_velocity"])
            histories[label] = {
                "time": times, "position": np.asarray(position),
                "velocity": np.asarray(velocity),
            }
    g11_position, g11_velocity = _checkpoint_history("G11_standard", 8, initials["G11"])
    histories["G11"] = {
        "time": np.linspace(0.0, 0.001, 9),
        "position": g11_position, "velocity": g11_velocity,
    }
    for label, steps in (("G10_coarse", 4), ("G10_half", 16)):
        position, velocity = _checkpoint_history(label, steps, initials["G10"])
        histories[label] = {
            "time": np.linspace(0.0, 0.001, steps + 1),
            "position": position, "velocity": velocity,
        }
    histories["G10_standard"] = histories["G10"]
    return histories


def proper_time(history):
    integrand_squared = -np.asarray(history["position"][:, -1, 0, 2])
    if np.min(integrand_squared) <= 0.0:
        raise RuntimeError("brane-axis coordinate worldline is not timelike")
    integrand = np.sqrt(integrand_squared)
    tau = np.concatenate((
        [0.0], cumulative_trapezoid(integrand, x=history["time"]),
    ))
    if np.any(np.diff(tau) <= 0.0):
        raise RuntimeError("brane-axis proper time is not monotone")
    return tau


def align_history(history, target_tau, method="pchip"):
    tau = proper_time(history)
    if target_tau < tau[0] or target_tau > tau[-1]:
        raise ValueError("proper-time target leaves history")
    if method == "pchip":
        position = PchipInterpolator(tau, history["position"], axis=0)(target_tau)
        velocity = PchipInterpolator(tau, history["velocity"], axis=0)(target_tau)
    elif method == "linear":
        right = int(np.searchsorted(tau, target_tau, side="right"))
        right = min(max(right, 1), len(tau) - 1)
        left = right - 1
        fraction = (target_tau - tau[left]) / (tau[right] - tau[left])
        position = (1.0 - fraction) * history["position"][left] + fraction * history["position"][right]
        velocity = (1.0 - fraction) * history["velocity"][left] + fraction * history["velocity"][right]
    else:
        raise ValueError("unknown time interpolation")
    return np.asarray(position), np.asarray(velocity)


def aligned_state_archive(geometries):
    initials = {
        label: np.asarray(geometries[label]["jet_field"].reduced_fields)
        for label in STANDARD_GRIDS
    }
    histories = state_histories(initials)
    spatial_tau = min(proper_time(histories[label])[-1] for label in STANDARD_GRIDS)
    temporal_tau = min(
        proper_time(histories[label])[-1]
        for label in ("G10_coarse", "G10_standard", "G10_half")
    )
    arrays = {"spatial_common_tau": np.asarray(spatial_tau), "temporal_common_tau": np.asarray(temporal_tau)}
    for label in STANDARD_GRIDS:
        position, velocity = align_history(histories[label], spatial_tau)
        linear_position, linear_velocity = align_history(histories[label], spatial_tau, method="linear")
        arrays[f"{label}_initial"] = initials[label]
        arrays[f"{label}_position"] = position
        arrays[f"{label}_velocity"] = velocity
        arrays[f"{label}_linear_position"] = linear_position
        arrays[f"{label}_linear_velocity"] = linear_velocity
        arrays[f"{label}_tau_history"] = proper_time(histories[label])
    for label in ("G10_coarse", "G10_standard", "G10_half"):
        position, velocity = align_history(histories[label], temporal_tau)
        linear_position, linear_velocity = align_history(histories[label], temporal_tau, method="linear")
        arrays[f"{label}_position"] = position
        arrays[f"{label}_velocity"] = velocity
        arrays[f"{label}_linear_position"] = linear_position
        arrays[f"{label}_linear_velocity"] = linear_velocity
        arrays[f"{label}_tau_history"] = proper_time(histories[label])
    return arrays


def reduced_metric(position, radius):
    full = spatial_metric_tensor(position, radius)
    return full[..., :2, :2], np.asarray(position[..., 3])


def common_distance_limit(states, grids):
    values = []
    for label, position in states:
        z, r = grids[label]
        metric, _ = reduced_metric(position, r)
        selected = r <= 6.0
        lengths = simpson(np.sqrt(metric[:, selected, 0, 0]), x=z, axis=0)
        values.append(float(np.min(lengths)))
    return 0.90 * min(values)


def chart_arrays(chart):
    return {
        "distance": chart.distance, "ray_label": chart.ray_label,
        "z": chart.z, "r": chart.r, "velocity": chart.velocity,
        "areal_radius": chart.areal_radius, "speed_squared": chart.speed_squared,
        "jacobian_DR_zr": chart.jacobian_DR_zr,
        "eikonal_qDD": chart.eikonal_qDD,
    }


def load_chart(path):
    with np.load(path) as archive:
        return NormalGeodesicChart(**{key: np.asarray(archive[key]) for key in (
            "distance", "ray_label", "z", "r", "velocity", "areal_radius",
            "speed_squared", "jacobian_DR_zr", "eikonal_qDD",
        )})


def build_chart_stage(index, label, slice_name, resolution_name, position, z, r, dmax):
    nD, nRays = MAP_RESOLUTIONS[resolution_name]
    stage_id = f"chart/{label}/{slice_name}/{resolution_name}"
    filename = f"chart_{label}_{slice_name}_{resolution_name}.npz"
    metric, sphere = reduced_metric(position, r)
    path, _ = stage_npz(
        index, stage_id, filename, "normal-geodesic-chart",
        {"grid": label, "slice": slice_name, "nD": nD, "rays": nRays, "dmax": dmax},
        lambda: chart_arrays(build_normal_geodesic_chart(
            z, r, metric, sphere, np.linspace(0.0, dmax, nD), ray_count=nRays,
        )), expected=1200.0,
    )
    chart = load_chart(path)
    validity = chart_validity(chart)
    return chart, validity


def metric4(mapped):
    shape = mapped["covariant"].shape[:2]
    result = np.zeros((*shape, 4, 4))
    result[..., :2, :2] = mapped["covariant"]
    result[..., 2, 2] = 1.0
    result[..., 3, 3] = 1.0
    return result


def extrinsic4(mapped):
    shape = mapped["K_DD"].shape
    result = np.zeros((*shape, 4, 4))
    result[..., 0, 0] = mapped["K_DD"]
    result[..., 0, 1] = result[..., 1, 0] = mapped["K_DR"]
    result[..., 1, 1] = mapped["K_RR"]
    result[..., 2, 2] = mapped["K_Omega"]
    result[..., 3, 3] = mapped["K_Omega"]
    return result


def mapped_state(position, velocity, initial, z, r, chart_initial, chart_final, distance, radius, method="cubic"):
    zi, ri = inverse_chart_at(chart_initial, distance, radius)
    zf, rf = inverse_chart_at(chart_final, distance, radius)
    initial_metric, initial_sphere = reduced_metric(initial, r)
    final_metric, final_sphere = reduced_metric(position, r)
    mapped_initial = mapped_metric_fields(
        initial_metric, initial_sphere, z, r, distance, radius, zi, ri, method=method,
    )
    mapped_final = mapped_metric_fields(
        final_metric, final_sphere, z, r, distance, radius, zf, rf, method=method,
    )
    native_K = adm_extrinsic_curvature_tensor(position, velocity, z, r)
    mapped_K = mapped_extrinsic_fields(
        native_K, final_sphere, z, r, zf, rf, mapped_final, method=method,
    )
    initial4, final4 = metric4(mapped_initial), metric4(mapped_final)
    return {
        "initial_metric": initial4,
        "final_metric": final4,
        "metric_increment": final4 - initial4,
        "ADM_K": extrinsic4(mapped_K),
        "trace_K": mapped_K["trace_K"], "KijKij": mapped_K["KijKij"],
        "weight": mapped_final["volume_density"],
        "native_z": zf, "native_r": rf,
    }


def paired_summary(left, right, distance, radius):
    difference = np.asarray(left["value"] - right["value"])
    weight = 0.5 * (np.asarray(left["weight"]) + np.asarray(right["weight"]))
    absolute = weighted_l2(difference, weight, distance, radius)
    left_norm = weighted_l2(left["value"], weight, distance, radius)
    right_norm = weighted_l2(right["value"], weight, distance, radius)
    pointwise = np.sqrt(np.sum(difference**2, axis=tuple(range(2, difference.ndim))))
    point_left = np.sqrt(np.sum(left["value"]**2, axis=tuple(range(2, difference.ndim))))
    point_right = np.sqrt(np.sum(right["value"]**2, axis=tuple(range(2, difference.ndim))))
    q95 = weighted_quantile(pointwise, weight)
    scale95 = max(weighted_quantile(point_left, weight), weighted_quantile(point_right, weight), 1e-300)
    return {
        "absolute_L2": absolute,
        "relative_L2": absolute / max(left_norm, right_norm, 1e-300),
        "weighted_q95": q95,
        "relative_weighted_q95": q95 / scale95,
        "difference": difference,
        "weight": weight,
    }


def public_pair(record):
    return {key: float(record[key]) for key in (
        "absolute_L2", "relative_L2", "weighted_q95", "relative_weighted_q95",
    )}


def sequence_score(primary_states, alternate_states, observable, intervals=(112, 128, 144)):
    pairs = (("G9", "G10"), ("G10", "G11"))
    primary = []
    uncertainty = []
    public = {}
    differences = []
    for left, right in pairs:
        record = paired_summary(
            {"value": primary_states[left][observable], "weight": primary_states[left]["weight"]},
            {"value": primary_states[right][observable], "weight": primary_states[right]["weight"]},
            primary_states["distance"], primary_states["radius"],
        )
        variants = []
        for variant in alternate_states:
            alternate = paired_summary(
                {"value": variant[left][observable], "weight": variant[left]["weight"]},
                {"value": variant[right][observable], "weight": variant[right]["weight"]},
                primary_states["distance"], primary_states["radius"],
            )
            variants.append(abs(alternate["absolute_L2"] - record["absolute_L2"]))
        u = max(variants, default=0.0)
        primary.append(record["absolute_L2"])
        uncertainty.append(u)
        public[f"{left}_{right}"] = {**public_pair(record), "uncertainty": u}
        differences.append(record["difference"])
    order = conservative_order_interval(primary[0], uncertainty[0], primary[1], uncertainty[1], intervals)
    coherence = sign_coherence(
        differences[0], differences[1],
        np.broadcast_to(primary[1] * 0.0 + primary_states["G10"]["weight"][..., None, None], differences[0].shape),
    )
    lower_coarse = max(primary[0] - uncertainty[0], 0.0)
    upper_fine = primary[1] + uncertainty[1]
    passed = bool(
        upper_fine < lower_coarse and order is not None and order[0] > 1.0
        and uncertainty[1] / max(primary[1], 1e-300) < 0.25
        and coherence is not None and coherence >= 0.70
    )
    return {
        "pairs": public, "order_interval": order, "sign_coherence": coherence,
        "strict_adverse_monotonicity": bool(upper_fine < lower_coarse),
        "map_limited": bool(
            primary[1] <= uncertainty[1]
            or uncertainty[1] / max(primary[1], 1e-300) >= 0.25
        ),
        "passed": passed,
    }


def state_variant(
    positions, grids, charts, distance, radius, chart_resolution="fine",
    field_method="cubic", time_variant="pchip",
):
    output = {"distance": distance, "radius": radius}
    for label, record in positions.items():
        z, r = grids[label if label in grids else "G10"]
        final_key = "position" if time_variant == "pchip" else "linear_position"
        velocity_key = "velocity" if time_variant == "pchip" else "linear_velocity"
        output[label] = mapped_state(
            record[final_key], record[velocity_key], record["initial"], z, r,
            charts[(label, "initial", chart_resolution)],
            charts[(label, time_variant, chart_resolution)],
            distance, radius, method=field_method,
        )
    return output


def field_analysis(index, geometries, aligned_path):
    with np.load(aligned_path) as archive:
        aligned = {key: np.asarray(archive[key]) for key in archive.files}
    grids = {label: (np.asarray(geometries[label]["z"]), np.asarray(geometries[label]["r"])) for label in STANDARD_GRIDS}
    spatial_records = {
        label: {
            "initial": aligned[f"{label}_initial"],
            "position": aligned[f"{label}_position"], "velocity": aligned[f"{label}_velocity"],
            "linear_position": aligned[f"{label}_linear_position"],
            "linear_velocity": aligned[f"{label}_linear_velocity"],
        } for label in PRIMARY_GRIDS
    }
    temporal_records = {
        label: {
            "initial": aligned["G10_initial"],
            "position": aligned[f"{label}_position"], "velocity": aligned[f"{label}_velocity"],
            "linear_position": aligned[f"{label}_linear_position"],
            "linear_velocity": aligned[f"{label}_linear_velocity"],
        } for label in ("G10_coarse", "G10_standard", "G10_half")
    }
    all_for_limit = []
    for label, record in spatial_records.items():
        all_for_limit.extend(((label, record["initial"]), (label, record["position"])))
    for label, record in temporal_records.items():
        all_for_limit.append(("G10", record["position"]))
    dmax = common_distance_limit(all_for_limit, grids)

    charts = {}
    validity = {}
    for label, record in {**spatial_records, **temporal_records}.items():
        grid_label = label if label in grids else "G10"
        z, r = grids[grid_label]
        slices = {
            "initial": record["initial"], "pchip": record["position"],
            "linear": record["linear_position"],
        }
        for slice_name, position in slices.items():
            for resolution in MAP_RESOLUTIONS:
                key = (label, slice_name, resolution)
                if key in charts:
                    continue
                chart, checked = build_chart_stage(
                    index, label, slice_name, resolution, position, z, r, dmax,
                )
                charts[key] = chart
                validity["/".join(key)] = checked
    if not all(record["valid"] for record in validity.values()):
        raise RuntimeError("one or more physical charts failed prospective validity")

    candidate_charts = [
        charts[(label, slice_name, "fine")]
        for label in (*PRIMARY_GRIDS, "G10_coarse", "G10_standard", "G10_half")
        for slice_name in ("initial", "pchip")
    ]
    rlower, rupper = common_areal_interval(candidate_charts, outer_limit=6.0)
    native_spacing = max(float(np.max(np.diff(grids[label][1]))) for label in PRIMARY_GRIDS)
    rlower = max(rlower, 2.0 * native_spacing)
    distance = np.linspace(0.0, 0.95 * dmax, MAP_RESOLUTIONS["primary"][0])
    radius = np.linspace(rlower, rupper, MAP_RESOLUTIONS["primary"][1])

    spatial_primary = state_variant(spatial_records, grids, charts, distance, radius)
    spatial_map_alt = state_variant(
        spatial_records, grids, charts, distance, radius,
        chart_resolution="primary", field_method="cubic",
    )
    spatial_interp_alt = state_variant(
        spatial_records, grids, charts, distance, radius,
        chart_resolution="fine", field_method="linear",
    )
    spatial_time_alt = state_variant(
        spatial_records, grids, charts, distance, radius,
        chart_resolution="fine", field_method="cubic", time_variant="linear",
    )
    spatial_scores = {
        observable: sequence_score(
            spatial_primary, (spatial_map_alt, spatial_interp_alt, spatial_time_alt), observable,
        ) for observable in ("metric_increment", "final_metric", "ADM_K")
    }

    # Temporal labels use uniform resolution counts 4/8/16.  Reuse the pair
    # machinery with explicit names and log2 order.
    temporal_primary = state_variant(temporal_records, grids, charts, distance, radius)
    temporal_map_alt = state_variant(
        temporal_records, grids, charts, distance, radius, chart_resolution="primary",
    )
    temporal_interp_alt = state_variant(
        temporal_records, grids, charts, distance, radius, field_method="linear",
    )
    temporal_time_alt = state_variant(
        temporal_records, grids, charts, distance, radius, time_variant="linear",
    )
    temporal_scores = {}
    for observable in ("metric_increment", "ADM_K"):
        pair_names = (("G10_coarse", "G10_standard"), ("G10_standard", "G10_half"))
        errors, uncertainties, published = [], [], {}
        for left, right in pair_names:
            record = paired_summary(
                {"value": temporal_primary[left][observable], "weight": temporal_primary[left]["weight"]},
                {"value": temporal_primary[right][observable], "weight": temporal_primary[right]["weight"]},
                distance, radius,
            )
            changes = []
            for variant in (temporal_map_alt, temporal_interp_alt, temporal_time_alt):
                alternate = paired_summary(
                    {"value": variant[left][observable], "weight": variant[left]["weight"]},
                    {"value": variant[right][observable], "weight": variant[right]["weight"]},
                    distance, radius,
                )
                changes.append(abs(alternate["absolute_L2"] - record["absolute_L2"]))
            uncertainty = max(changes)
            errors.append(record["absolute_L2"])
            uncertainties.append(uncertainty)
            published[f"{left}_{right}"] = {**public_pair(record), "uncertainty": uncertainty}
        if min(errors[0] - uncertainties[0], errors[1] - uncertainties[1]) > 0.0:
            ratio_low = (errors[0] - uncertainties[0]) / (errors[1] + uncertainties[1])
            ratio_high = (errors[0] + uncertainties[0]) / (errors[1] - uncertainties[1])
            order_interval = (math.log2(ratio_low), math.log2(ratio_high))
        else:
            order_interval = None
        spatial_fine = spatial_scores[observable]["pairs"]["G10_G11"]["absolute_L2"]
        spatial_u = spatial_scores[observable]["pairs"]["G10_G11"]["uncertainty"]
        passed = bool(
            errors[1] + uncertainties[1] < errors[0] - uncertainties[0]
            and order_interval is not None and order_interval[0] > 1.5
            and uncertainties[1] / max(errors[1], 1e-300) < 0.25
            and errors[1] + uncertainties[1] < 0.5 * max(spatial_fine - spatial_u, 0.0)
        )
        temporal_scores[observable] = {
            "pairs": published, "order_interval": order_interval,
            "strict_adverse_monotonicity": bool(errors[1] + uncertainties[1] < errors[0] - uncertainties[0]),
            "fine_temporal_below_half_spatial": bool(errors[1] + uncertainties[1] < 0.5 * max(spatial_fine - spatial_u, 0.0)),
            "passed": passed,
        }

    arrays = {"distance": distance, "radius": radius, "dmax": np.asarray(dmax)}
    for label in PRIMARY_GRIDS:
        for observable in ("initial_metric", "final_metric", "metric_increment", "ADM_K", "trace_K", "KijKij", "weight", "native_z", "native_r"):
            arrays[f"{label}_{observable}"] = spatial_primary[label][observable]
    stage_path, _ = stage_npz(
        index, "fields/primary", "fields_primary.npz", "mapped-physical-fields",
        {"distance_nodes": len(distance), "radius_nodes": len(radius)},
        lambda: arrays, expected=1200.0,
    )
    payload = {
        "dmax": dmax, "comparison_distance_max": float(distance[-1]),
        "areal_radius_interval": [float(radius[0]), float(radius[-1])],
        "chart_validity": validity, "spatial": spatial_scores,
        "temporal": temporal_scores, "field_archive": str(stage_path),
    }
    return payload, spatial_primary, charts, distance, radius


def _surface_profile(position, velocity, z, r, surface, chart):
    prepared = prepare_capped_expansion_slice(position, velocity, z, r)
    theta = np.asarray(surface["theta"])
    rho = np.asarray(surface["rho"])
    slope = np.asarray(surface["slope"])
    second = dynamical_rho_second(prepared, theta, rho, slope)
    sampled = _sample_geometry(prepared, theta, rho, slope)
    outgoing = local_outgoing_expansion(prepared, theta, rho, slope, second)
    tangent_K = np.einsum(
        "...a,...ab,...b->...", sampled["tangent"], sampled["extrinsic"], sampled["tangent"],
    )
    correction = -tangent_K - 2.0 * sampled["sphere_extrinsic"]
    ingoing = -outgoing + 2.0 * correction
    native_z = z[-1] - rho * np.cos(theta)
    native_r = rho * np.sin(theta)
    D_linear = native_to_proper_distance(chart, native_z, native_r, method="linear")
    D_cubic = native_to_proper_distance(chart, native_z, native_r, method="cubic")
    sphere = interpolate_regular_field(position[..., 3], z, r, native_z, native_r)
    R = native_r * np.sqrt(sphere)
    speed = np.sqrt(sampled["speed_squared"])
    arclength = np.concatenate(([0.0], cumulative_trapezoid(speed, x=theta)))
    normalized = arclength / arclength[-1]
    target = np.linspace(0.0, 1.0, 501)
    return {
        "s": target,
        "D": PchipInterpolator(normalized, D_cubic)(target),
        "D_linear": PchipInterpolator(normalized, D_linear)(target),
        "R": PchipInterpolator(normalized, R)(target),
        "theta_minus": PchipInterpolator(normalized, ingoing)(target),
        "theta_plus": PchipInterpolator(normalized, outgoing)(target),
        "proper_length": arclength[-1],
    }


def _stability_record(position, velocity, z, r, surface):
    prepared = prepare_capped_expansion_slice(position, velocity, z, r)
    spectra = {
        nodes: mots_stability_matrix(
            position, velocity, z, r, surface, nodes=nodes,
            relative_step=1e-5, prepared=prepared,
        ) for nodes in (49, 65, 81)
    }
    check = mots_stability_matrix(
        position, velocity, z, r, surface, nodes=81,
        relative_step=2e-5, prepared=prepared,
    )
    eigenvalue = float(spectra[81]["principal_eigenvalue_real"])
    leading = spectra[81]["leading_eigenvalues"]
    spectral_gap = float(leading[1]["real"] - leading[0]["real"])
    angular_error = abs(eigenvalue - float(spectra[65]["principal_eigenvalue_real"]))
    step_error = abs(eigenvalue - float(check["principal_eigenvalue_real"]))
    signs = [np.sign(spectra[n]["principal_eigenvalue_real"]) for n in (49, 65, 81)]
    resolved = bool(
        signs[0] != 0 and signs[0] == signs[1] == signs[2]
        and abs(eigenvalue) > 5.0 * max(angular_error, step_error, 1e-14)
    )
    return {
        "eigenvalue": eigenvalue, "spectral_gap": spectral_gap,
        "angular_error": angular_error,
        "step_error": step_error, "resolved": resolved,
        "classification": (
            "outward_stable" if resolved and eigenvalue > 0.0
            else "outward_unstable" if resolved and eigenvalue < 0.0
            else "unresolved"
        ),
        "spectra": {str(key): public_stability(value) for key, value in spectra.items()},
        "step_check": public_stability(check),
    }


def surface_stage(index, label, branch, position, velocity, geometry, chart):
    stage_id = f"surfaces/{label}/{branch}"
    json_name = f"surface_{label}_{branch}.json"
    npz_name = f"surface_{label}_{branch}.npz"
    z, r = np.asarray(geometry["z"]), np.asarray(geometry["r"])

    def compute():
        surface = solve_dynamical_capped_surface_bvp(
            position, velocity, z, r, BRANCH_SEEDS[branch],
            tolerance=2e-5, nodes=121, maximum_nodes=6000, dense_nodes=501,
        )
        if not surface_passes(surface):
            raise RuntimeError(f"{label} {branch} surface failed admission")
        profile = _surface_profile(position, velocity, z, r, surface, chart)
        geometry_record = capped_surface_geometry(position, velocity, z, r, surface)
        stability = _stability_record(position, velocity, z, r, surface)
        atomic_write_npz(RECOVERY_ROOT / npz_name, **profile)
        return {
            "surface": {key: surface[key] for key in (
                "converged", "in_domain", "rho_axis", "rho_brane",
                "boundary_slope_error", "local_expansion_interior_maximum",
                "primary_evaluator_crosscheck",
            )},
            "geometry": geometry_record, "stability": stability,
            "profile_archive": str(RECOVERY_ROOT / npz_name),
            "profile_sha256": sha256_file(RECOVERY_ROOT / npz_name),
        }
    payload, _ = stage_json(
        index, stage_id, json_name, "surface-expansion-stability",
        {"grid": label, "branch": branch, "seed": BRANCH_SEEDS[branch]},
        compute, expected=1800.0,
    )
    return payload


def scalar_tail(values, uncertainties=None):
    uncertainties = [0.0, 0.0, 0.0] if uncertainties is None else uncertainties
    e12, e23 = abs(values[0] - values[1]), abs(values[1] - values[2])
    u12 = math.hypot(uncertainties[0], uncertainties[1])
    u23 = math.hypot(uncertainties[1], uncertainties[2])
    interval = conservative_order_interval(e12, u12, e23, u23)
    passed = bool(
        e23 + u23 < max(e12 - u12, 0.0) and interval is not None
        and interval[0] > 1.0 and u23 / max(e23, 1e-300) < 0.25
    )
    return {
        "values": [float(x) for x in values], "adjacent_differences": [e12, e23],
        "uncertainties": [u12, u23], "order_interval": interval,
        "passed": passed,
    }


def surface_analysis(index, geometries, aligned_path, charts):
    with np.load(aligned_path) as archive:
        aligned = {key: np.asarray(archive[key]) for key in archive.files}
    records = {branch: {} for branch in BRANCH_SEEDS}
    for label in PRIMARY_GRIDS:
        for branch in BRANCH_SEEDS:
            records[branch][label] = surface_stage(
                index, label, branch, aligned[f"{label}_position"],
                aligned[f"{label}_velocity"], geometries[label],
                charts[(label, "pchip", "fine")],
            )
    scores = {}
    for branch, grids in records.items():
        branch_score = {}
        for key in (
            "one_sided_cap_area", "equivalent_area_radius", "proper_meridional_length",
            "maximum_sphere_radius",
        ):
            branch_score[key] = scalar_tail([grids[label]["geometry"][key] for label in PRIMARY_GRIDS])
        eigenvalues = [grids[label]["stability"]["eigenvalue"] for label in PRIMARY_GRIDS]
        eigen_errors = [
            max(grids[label]["stability"]["angular_error"], grids[label]["stability"]["step_error"])
            for label in PRIMARY_GRIDS
        ]
        branch_score["stability"] = scalar_tail(eigenvalues, eigen_errors)
        branch_score["stability"]["classifications"] = [
            grids[label]["stability"]["classification"] for label in PRIMARY_GRIDS
        ]
        branch_score["spectral_gap"] = scalar_tail([
            grids[label]["stability"]["spectral_gap"] for label in PRIMARY_GRIDS
        ])
        profiles = {}
        for label in PRIMARY_GRIDS:
            with np.load(grids[label]["profile_archive"]) as archive:
                profiles[label] = {key: np.asarray(archive[key]) for key in archive.files}
        for key in ("D", "R", "theta_minus"):
            differences = [
                float(np.sqrt(simpson((profiles[left][key] - profiles[right][key])**2, x=profiles[left]["s"])))
                for left, right in (("G9", "G10"), ("G10", "G11"))
            ]
            if key == "D":
                uncertainties = [
                    float(np.sqrt(simpson((profiles[label]["D"] - profiles[label]["D_linear"])**2, x=profiles[label]["s"])))
                    for label in PRIMARY_GRIDS
                ]
                u = [math.hypot(uncertainties[0], uncertainties[1]), math.hypot(uncertainties[1], uncertainties[2])]
            else:
                u = [0.0, 0.0]
            interval = conservative_order_interval(differences[0], u[0], differences[1], u[1])
            branch_score[key] = {
                "adjacent_L2_differences": differences, "uncertainties": u,
                "order_interval": interval,
                "sign_coherence": (
                    sign_coherence(
                        profiles["G9"][key] - profiles["G10"][key],
                        profiles["G10"][key] - profiles["G11"][key],
                        np.ones_like(profiles["G9"][key]),
                    ) if key == "theta_minus" else None
                ),
                "passed": bool(
                    differences[1] + u[1] < max(differences[0] - u[0], 0.0)
                    and interval is not None and interval[0] > 1.0
                    and u[1] / max(differences[1], 1e-300) < 0.25
                    and (
                        key != "theta_minus"
                        or sign_coherence(
                            profiles["G9"][key] - profiles["G10"][key],
                            profiles["G10"][key] - profiles["G11"][key],
                            np.ones_like(profiles["G9"][key]),
                        ) is not None
                        and sign_coherence(
                            profiles["G9"][key] - profiles["G10"][key],
                            profiles["G10"][key] - profiles["G11"][key],
                            np.ones_like(profiles["G9"][key]),
                        ) >= 0.70
                    )
                ),
            }
        for endpoint in (0, -1):
            for key in ("D", "R"):
                branch_score[f"{key}_endpoint_{endpoint}"] = scalar_tail([
                    profiles[label][key][endpoint] for label in PRIMARY_GRIDS
                ])
        branch_score["outgoing_residuals"] = [
            grids[label]["surface"]["local_expansion_interior_maximum"] for label in PRIMARY_GRIDS
        ]
        scores[branch] = branch_score
    inner_classes = scores["inner"]["stability"]["classifications"]
    outer_classes = scores["outer"]["stability"]["classifications"]
    overall = bool(
        all(inner == "outward_unstable" for inner in inner_classes)
        and all(outer == "outward_stable" for outer in outer_classes)
        and all(item["passed"] for branch in scores.values() for key, item in branch.items()
                if key not in ("outgoing_residuals",))
    )
    return {"records": records, "scores": scores, "passed": overall}


def _bracket(counts, tau):
    first = next((index for index, value in enumerate(counts) if value == 2), None)
    if first is None:
        return None
    return [float(tau[first]), float(tau[first + 1])]


def formation_analysis(aligned_path):
    with np.load(aligned_path) as archive:
        tau = {key.removesuffix("_tau_history"): np.asarray(archive[key]) for key in archive.files if key.endswith("_tau_history")}
    with Path("results/corrected_A790_formation_time_refinement.json").open() as stream:
        base = json.load(stream)
    with Path("results/corrected_A790_third_grid_formation_reproduction.json").open() as stream:
        g9 = json.load(stream)
    with Path("results/corrected_A790_fourth_grid_physical_tensor_convergence.json").open() as stream:
        g10 = json.load(stream)
    with Path("results/corrected_A790_test10_joint_convergence.json").open() as stream:
        test10 = json.load(stream)
    counts = {
        "G7": base["count_histories"]["G7"], "G8": base["count_histories"]["G8"],
        "G9": g9["count_history"], "G10": g10["independent_BVP_count_histories"]["G10"],
        "G11": test10["short_spatial_time_analysis"]["histories"]["G11_standard"],
        "G10_coarse": test10["short_spatial_time_analysis"]["histories"]["G10_coarse"],
        "G10_standard": g10["independent_BVP_count_histories"]["G10"],
        "G10_half": test10["short_spatial_time_analysis"]["histories"]["G10_half"],
    }
    brackets = {label: _bracket(values, tau[label]) for label, values in counts.items()}
    spatial = [brackets[label] for label in STANDARD_GRIDS]
    overlap = max(item[0] for item in spatial) <= min(item[1] for item in spatial)
    temporal = [brackets[label] for label in ("G10_coarse", "G10_standard", "G10_half")]
    widths = [right - left for left, right in temporal]
    temporal_overlap = all(
        max(temporal[index][0], temporal[index + 1][0])
        <= min(temporal[index][1], temporal[index + 1][1])
        for index in range(2)
    )
    histories_valid = all(
        all(value in (0, 2) for value in values)
        and all(left <= right for left, right in zip(values, values[1:]))
        and 2 in values for values in counts.values()
    )
    passed = bool(histories_valid and overlap and temporal_overlap and widths[2] < widths[1] < widths[0])
    return {
        "count_histories": counts, "proper_time_brackets": brackets,
        "spatial_interval_overlap": overlap, "temporal_adjacent_overlap": temporal_overlap,
        "temporal_widths": widths, "passed": passed,
    }


def common_parent_analysis():
    with np.load(COMMON_PARENT_STATE) as archive:
        result = {}
        initial_native_maximum = 0.0
        evolved_native_maximum = 0.0
        mapped_initial_relative_maximum = 0.0
        mapped_evolved_relative_maximum = 0.0
        maps_valid = True
        for grid in ("G7", "G8"):
            z = np.asarray(archive[f"{grid}_R12_z"])
            r8 = np.asarray(archive[f"{grid}_R8_r"])
            r10 = np.asarray(archive[f"{grid}_R10_r"])
            r12 = np.asarray(archive[f"{grid}_R12_r"])
            pairs = {}
            for time_key in ("initial", "step_008_position", "step_016_position"):
                arrays = {
                    domain: np.asarray(archive[f"{grid}_R{domain}_{time_key}"])
                    for domain in (8, 10, 12)
                }
                local = []
                for left, right in ((8, 10), (10, 12)):
                    count = len(r8)
                    difference = float(np.max(np.abs(arrays[left][:, :count] - arrays[right][:, :count])))
                    local.append(difference)
                    if time_key == "initial":
                        initial_native_maximum = max(initial_native_maximum, difference)
                    else:
                        evolved_native_maximum = max(evolved_native_maximum, difference)
                pairs[time_key] = local
            # The normal charts use the native radial nodes as ray labels, so
            # the exact common-parent prefixes receive identical samples.
            positions = {
                domain: {
                    "initial": np.asarray(archive[f"{grid}_R{domain}_initial"]),
                    "position": np.asarray(archive[f"{grid}_R{domain}_step_016_position"]),
                    "velocity": np.asarray(archive[f"{grid}_R{domain}_step_016_velocity"]),
                    "r": np.asarray(archive[f"{grid}_R{domain}_r"]),
                } for domain in (8, 10, 12)
            }
            dmax = common_distance_limit(
                [(f"R{domain}", record[slice_name])
                 for domain, record in positions.items()
                 for slice_name in ("initial", "position")],
                {f"R{domain}": (z, record["r"]) for domain, record in positions.items()},
            )
            distance = np.linspace(0.0, 0.95 * dmax, 129)
            domain_charts = {}
            for domain, record in positions.items():
                native_r = record["r"]
                initial_metric, initial_sphere = reduced_metric(record["initial"], native_r)
                final_metric, final_sphere = reduced_metric(record["position"], native_r)
                initial_chart = build_normal_geodesic_chart(
                    z, native_r, initial_metric, initial_sphere,
                    np.linspace(0.0, dmax, 193), ray_count=len(native_r),
                )
                final_chart = build_normal_geodesic_chart(
                    z, native_r, final_metric, final_sphere,
                    np.linspace(0.0, dmax, 193), ray_count=len(native_r),
                )
                maps_valid = bool(
                    maps_valid and chart_validity(initial_chart)["valid"]
                    and chart_validity(final_chart)["valid"]
                )
                domain_charts[domain] = (initial_chart, final_chart)
            common_radius = common_areal_interval([
                chart for pair in domain_charts.values() for chart in pair
            ], outer_limit=6.0)
            radius = np.linspace(
                max(common_radius[0], 2.0 * max(np.diff(r8))),
                common_radius[1], 193,
            )
            mapped = {}
            for domain, record in positions.items():
                native_r = record["r"]
                initial_chart, final_chart = domain_charts[domain]
                mapped[domain] = mapped_state(
                    record["position"], record["velocity"], record["initial"],
                    z, native_r, initial_chart, final_chart, distance, radius,
                )
            mapped_pairs = {}
            for left, right in ((8, 10), (10, 12)):
                mapped_pairs[f"R{left}_R{right}"] = {}
                for observable in ("initial_metric", "metric_increment", "final_metric", "ADM_K"):
                    score = paired_summary(
                        {"value": mapped[left][observable], "weight": mapped[left]["weight"]},
                        {"value": mapped[right][observable], "weight": mapped[right]["weight"]},
                        distance, radius,
                    )
                    mapped_pairs[f"R{left}_R{right}"][observable] = public_pair(score)
                    if observable == "initial_metric":
                        mapped_initial_relative_maximum = max(
                            mapped_initial_relative_maximum, score["relative_L2"],
                        )
                    else:
                        mapped_evolved_relative_maximum = max(
                            mapped_evolved_relative_maximum, score["relative_L2"],
                        )
            pairs["mapped"] = mapped_pairs
            result[grid] = pairs
    return {
        "native_exact_initial_common_interior_maximum": initial_native_maximum,
        "native_evolved_common_interior_maximum": evolved_native_maximum,
        "native_pairs": result,
        "mapped_initial_relative_maximum": mapped_initial_relative_maximum,
        "mapped_evolved_relative_maximum": mapped_evolved_relative_maximum,
        "mapped_null_bound": 1e-8, "maps_valid": maps_valid,
        "passed": bool(
            initial_native_maximum < 1e-12 and maps_valid
            and mapped_initial_relative_maximum < 1e-8
            and mapped_evolved_relative_maximum < 1e-3
        ),
        "classification": "exact common-parent native and mapped physical-chart control",
    }


def independent_audit(fields):
    required = ("spatial", "temporal", "chart_validity")
    finite = all(key in fields for key in required)
    order_checks = {}
    spatial_consistent = True
    for observable in ("metric_increment", "final_metric", "ADM_K"):
        record = fields["spatial"][observable]
        pairs = record["pairs"]
        spatial_consistent = spatial_consistent and set(pairs) == {"G9_G10", "G10_G11"}
        recomputed = conservative_order_interval(
            pairs["G9_G10"]["absolute_L2"], pairs["G9_G10"]["uncertainty"],
            pairs["G10_G11"]["absolute_L2"], pairs["G10_G11"]["uncertainty"],
        )
        stored = record["order_interval"]
        agrees = (
            recomputed is None and stored is None
            or recomputed is not None and stored is not None
            and np.allclose(recomputed, stored, rtol=1e-12, atol=1e-14)
        )
        order_checks[observable] = {"recomputed": recomputed, "stored": stored, "agrees": bool(agrees)}
        spatial_consistent = spatial_consistent and bool(agrees)
    return {
        "passed": bool(finite and spatial_consistent),
        "analysis_structure_recomputed": True, "order_checks": order_checks,
    }


def final_grade(controls, fields, surfaces, formation, common_parent, audit):
    map_pass = bool(all(item["valid"] for item in fields["chart_validity"].values()))
    metric_pass = bool(fields["spatial"]["metric_increment"]["passed"])
    full_metric_pass = bool(fields["spatial"]["final_metric"]["passed"])
    adm_pass = bool(fields["spatial"]["ADM_K"]["passed"])
    temporal_pass = bool(all(item["passed"] for item in fields["temporal"].values()))
    all_pass = bool(
        controls["passed"] and map_pass and metric_pass and full_metric_pass
        and adm_pass and temporal_pass and surfaces["passed"]
        and formation["passed"] and common_parent["passed"] and audit["passed"]
    )
    simultaneous_growth = bool(
        not fields["spatial"]["metric_increment"]["strict_adverse_monotonicity"]
        and not fields["spatial"]["ADM_K"]["strict_adverse_monotonicity"]
        and not fields["spatial"]["metric_increment"]["map_limited"]
        and not fields["spatial"]["ADM_K"]["map_limited"]
    )
    if all_pass:
        status = "PASS"
        classification = "invariant_chart_above_first_order_continuum_evidence"
    elif controls["passed"] and map_pass and simultaneous_growth:
        status = "FAIL"
        classification = "invariant_chart_nonconvergence_or_branch_failure"
    else:
        status = "REVIEW"
        classification = "invariant_chart_convergence_mixed"
    return {
        "status": status, "classification": classification,
        "subverdicts": {
            "controls": controls["passed"], "map": map_pass,
            "metric_increment": metric_pass, "full_metric": full_metric_pass,
            "ADM_K": adm_pass, "temporal_separation": temporal_pass,
            "surface_expansion_stability": surfaces["passed"],
            "formation_interval": formation["passed"],
            "common_parent_control": common_parent["passed"],
            "independent_audit": audit["passed"],
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("controls", "fields", "surfaces", "all"), default="all")
    args = parser.parse_args()
    if sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("sealed Test-2B protocol hash mismatch")
    RECOVERY_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    index = RecoveryIndex(MANIFEST, PROTOCOL, recovery_inputs(), maximum_stage_seconds=2400.0)
    cached_final = index.validated_path("final/result")
    if args.phase == "all" and cached_final is not None:
        final_record = json.loads(cached_final.read_text())
        if (
            OUTPUT.is_file() and STATE_OUTPUT.is_file()
            and sha256_file(OUTPUT) == final_record["output_sha256"]
            and sha256_file(STATE_OUTPUT) == final_record["state_sha256"]
        ):
            result = json.loads(OUTPUT.read_text())
            print(json.dumps({
                "status": result["status"],
                "classification": result["classification"],
                "recovered_complete_result": True,
                "recovery": recovery_summary(index),
            }, indent=2))
            return
    controls, _ = stage_json(
        index, "controls/manufactured", "controls.json", "manufactured-controls", {},
        lambda: {"controls": manufactured_controls()}, expected=1800.0,
    )
    controls = controls["controls"]
    if not controls["passed"]:
        raise RuntimeError("manufactured controls failed")
    if args.phase == "controls":
        print(json.dumps({"phase": "controls", "passed": True, "recovery": recovery_summary(index)}, indent=2))
        return

    print("reconstructing G7--G11 geometries", flush=True)
    geometries = build_all_geometries()
    geometry_path, _ = stage_npz(
        index, "qualification/geometries", "geometries.npz", "qualified-initial-geometries", {},
        lambda: geometry_archive(geometries), expected=1200.0,
    )
    aligned_path, _ = stage_npz(
        index, "qualification/aligned_states", "aligned_states.npz", "proper-time-aligned-states", {},
        lambda: aligned_state_archive(geometries), expected=600.0,
    )
    fields_payload, _, charts, _, _ = field_analysis(index, geometries, aligned_path)
    fields, _ = stage_json(
        index, "analysis/fields", "analysis_fields.json", "continuum-field-analysis", {},
        lambda: {"fields": fields_payload}, expected=1200.0,
    )
    fields = fields["fields"]
    if args.phase == "fields":
        print(json.dumps({"phase": "fields", "spatial": fields["spatial"], "temporal": fields["temporal"], "recovery": recovery_summary(index)}, indent=2))
        return

    surfaces_payload = surface_analysis(index, geometries, aligned_path, charts)
    surfaces, _ = stage_json(
        index, "analysis/surfaces", "analysis_surfaces.json", "surface-continuum-analysis", {},
        lambda: {"surfaces": surfaces_payload}, expected=1200.0,
    )
    surfaces = surfaces["surfaces"]
    if args.phase == "surfaces":
        print(json.dumps({"phase": "surfaces", "passed": surfaces["passed"], "recovery": recovery_summary(index)}, indent=2))
        return

    formation, _ = stage_json(
        index, "formation/history", "formation.json", "proper-time-formation", {},
        lambda: {"formation": formation_analysis(aligned_path)}, expected=120.0,
    )
    formation = formation["formation"]
    common_parent, _ = stage_json(
        index, "common_parent/control", "common_parent.json", "common-parent-control", {},
        lambda: {"common_parent": common_parent_analysis()}, expected=120.0,
    )
    common_parent = common_parent["common_parent"]
    audit, _ = stage_json(
        index, "audit/independent", "independent_audit.json", "independent-internal-audit", {},
        lambda: {"audit": independent_audit(fields)}, expected=120.0,
    )
    audit = audit["audit"]
    grade = final_grade(controls, fields, surfaces, formation, common_parent, audit)
    result = {
        "protocol": str(PROTOCOL), "protocol_sha256": PROTOCOL_SHA256,
        **grade, "controls": controls, "fields": fields, "surfaces": surfaces,
        "formation": formation, "common_parent": common_parent,
        "independent_audit": audit, "recovery": recovery_summary(index),
        "runtime_seconds_this_invocation": time.perf_counter() - started,
        "claim_boundary": (
            "tested invariant-chart continuum behavior only; not an event horizon, "
            "topology, throat, halo, mass-transfer, or cosmological claim"
        ),
    }
    atomic_write_json(OUTPUT, result)
    atomic_write_npz(
        STATE_OUTPUT,
        protocol_sha256=np.asarray(PROTOCOL_SHA256),
        status=np.asarray(grade["status"]),
        classification=np.asarray(grade["classification"]),
    )
    final, _ = stage_json(
        index, "final/result", "final_result.json", "final-result", {},
        lambda: {"output": str(OUTPUT), "output_sha256": sha256_file(OUTPUT),
                 "state_output": str(STATE_OUTPUT), "state_sha256": sha256_file(STATE_OUTPUT),
                 "status": grade["status"], "classification": grade["classification"]},
        expected=120.0,
    )
    print(json.dumps({**grade, "final": final, "recovery": recovery_summary(index)}, indent=2))


if __name__ == "__main__":
    main()
