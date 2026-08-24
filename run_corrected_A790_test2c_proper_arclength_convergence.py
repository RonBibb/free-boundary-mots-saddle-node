#!/usr/bin/env python3
"""Sealed Test-2C convergence audit in proper normal/arclength coordinates."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid, simpson
from scipy.interpolate import PchipInterpolator

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import run_corrected_A790_test2b_invariant_convergence as old
from bhps.corrected_A790_physical_tensor_convergence import adm_extrinsic_curvature_tensor
from bhps.dynamical_capped_geometry import capped_surface_geometry
from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice
from bhps.dynamical_capped_horizon_bvp import (
    _sample_geometry,
    dynamical_rho_second,
    local_outgoing_expansion,
    solve_dynamical_capped_surface_bvp,
)
from bhps.invariant_physical_chart import (
    NormalGeodesicChart,
    conservative_order_interval,
    interpolate_regular_field,
    mapped_extrinsic_fields,
    mapped_metric_fields,
    sign_coherence,
)
from bhps.invariant_proper_arclength_chart import (
    ProperArclengthChart,
    arclength_at_native_radius,
    chart_validity,
    inverse_chart_at,
    native_to_coordinates,
    relabel_normal_chart,
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


PROTOCOL = Path("notes/108_A790_test2C_proper_brane_arclength_convergence_protocol.md")
PROTOCOL_SHA256 = "3ea036749550a11362cdce4bc4dfdf38eb305e69aa671e7c3a150edeaead6cfe"
OUTPUT = Path("results/corrected_A790_test2c_proper_arclength_convergence.json")
STATE_OUTPUT = Path("results/corrected_A790_test2c_proper_arclength_convergence_state.npz")
RECOVERY_ROOT = Path("results/corrected_A790_test2c_proper_arclength_convergence_recovery")
MANIFEST = RECOVERY_ROOT / "index.json"
OLD_ROOT = Path("results/corrected_A790_test2b_invariant_convergence_recovery")
OLD_MANIFEST = OLD_ROOT / "index.json"
ALIGNED_STATE = OLD_ROOT / "aligned_states.npz"
GEOMETRY_STATE = OLD_ROOT / "geometries.npz"
COMMON_PARENT_STATE = Path("results/corrected_A790_test10b_domain_normalized_state.npz")
PRIMARY_GRIDS = ("G9", "G10", "G11")
TEMPORAL_RUNS = ("G10_coarse", "G10_standard", "G10_half")
STANDARD_GRIDS = ("G7", "G8", "G9", "G10", "G11")
BRANCH_SEEDS = {"inner": 1.30, "outer": 1.55}
RESOLUTIONS = ("coarse", "primary", "fine")

FIXED_INPUT_HASHES = {
    "notes/107_A790_test2B_invariant_physical_coordinate_convergence_protocol.md": "e1a13f613fc530a811c88fcf5dbe4fb061f88bbfa66f2c1db05c4fcf7638485c",
    "notes/107_A790_test2B_invariant_physical_coordinate_convergence_result.md": "87524f31321d2f75eb4df0e5fc3e5978e205f1afaf31abe7508606a7c10489d2",
    "results/corrected_A790_test2b_invariant_convergence.json": "8615863b87f51caf72e48a0135e81a8e7a00d8b44954ba6db07e85d76c8cdd57",
    str(OLD_MANIFEST): "c7fad4658a73750b4b2a81e41f020fb62a4b054ce5bfbb621545b69240de392f",
    str(GEOMETRY_STATE): "fd5cf03f1fb37b9e9f7d1f14f8112c2452ccb4f0f6d8438d251d1432bb2dd618",
    str(ALIGNED_STATE): "42ec6dcb65038a6aa7ab1fc724e848bbba7be781b887cd18f3512cc030256265",
    "run_corrected_A790_test2b_invariant_convergence.py": "7e4e8e02e770eb58435231c40d579a64fc309b74cd035d8c5a0784c7c76a4122",
    "src/bhps/invariant_physical_chart.py": "74a0d1102750ada7845d2f3b7028a4a16db9c720b288537e3c64a07b78d68f8b",
    str(COMMON_PARENT_STATE): "b19e7e8901fdade6abb485c65a3512bf327a86f13974fb9a8c0876340a503928",
    "results/corrected_A790_formation_time_refinement.json": "57cd14b9712907c23e09b8fc2cc79de738c0d963353a27b9d2ea53d37a86d010",
    "results/corrected_A790_third_grid_formation_reproduction.json": "fe3607d0fad311b81dd91d3592a3fc3eaed2ed20a3e3fccde5022f7fdb06682b",
    "results/corrected_A790_fourth_grid_physical_tensor_convergence.json": "03d9f7ff4496ad430ecf368452dfe8ce911fda94a5d594269abec57ebd177b98",
    "results/corrected_A790_test10_joint_convergence.json": "da8c375216b4ca7e42529c711a403b0a2210f9a415df6200629098191ea758bf",
}


def recovery_inputs():
    dynamic = (
        Path(__file__), Path("src/bhps/invariant_proper_arclength_chart.py"),
        Path("src/bhps/recovery_indexer.py"),
    )
    return {**FIXED_INPUT_HASHES, **{str(path): sha256_file(path) for path in dynamic}}


def stage_json(index, stage_id, filename, kind, metadata, producer, expected=900.0):
    path = RECOVERY_ROOT / filename
    index.register(stage_id, kind, expected, metadata)
    cached = index.validated_path(stage_id)
    if cached is not None:
        return json.loads(cached.read_text()), True
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        payload = {"stage_id": stage_id, "protocol_sha256": index.protocol_sha256, **producer()}
        atomic_write_json(path, payload)
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
        atomic_write_npz(path, **producer())
        validate_npz(path)
        index.mark_complete(stage_id, path, time.perf_counter() - started)
        return path, False
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def recovery_summary(index):
    statuses = {}
    for record in index.data["stages"].values():
        status = record.get("status", "unknown")
        statuses[status] = statuses.get(status, 0) + 1
    return {"manifest": str(index.path), "stage_count": len(index.data["stages"]), "statuses": statuses}


def validate_old_recovery():
    manifest = json.loads(OLD_MANIFEST.read_text())
    checked = 0
    for stage_id, record in manifest["stages"].items():
        if not stage_id.startswith("chart/"):
            continue
        path = Path(record["output_path"])
        if (
            record.get("status") != "complete" or not path.is_file()
            or path.stat().st_size != record["byte_count"]
            or sha256_file(path) != record["sha256"]
        ):
            raise RuntimeError(f"invalid reused Test-2B stage {stage_id}")
        checked += 1
    if checked != 54:
        raise RuntimeError(f"expected 54 frozen charts, found {checked}")
    return {"validated_chart_stages": checked, "old_manifest_sha256": sha256_file(OLD_MANIFEST)}


def manufactured_controls():
    base = old.manufactured_controls()
    z = np.linspace(0.0, 1.0, 49)
    r = np.linspace(0.0, 2.0, 65)
    metric = np.zeros((len(z), len(r), 2, 2))
    metric[..., 0, 0] = (1.0 + 0.3 * z[:, None])**2
    metric[..., 1, 1] = (1.0 + 0.2 * r[None, :])**2
    physical_r = r * (1.0 - 0.3 * np.exp(-((r - 1.0) / 0.25)**2))
    sphere_line = np.ones_like(r)
    sphere_line[1:] = (physical_r[1:] / r[1:])**2
    sphere = np.broadcast_to(sphere_line, (len(z), len(r))).copy()
    normal = old.build_normal_geodesic_chart(
        z, r, metric, sphere, np.linspace(0.0, 0.7, 97), ray_count=129,
    )
    chart = relabel_normal_chart(normal, z, r, metric)
    expected = chart.native_brane_radius + 0.1 * chart.native_brane_radius**2
    arclength_error = float(np.max(np.abs(chart.arclength - expected)))
    nonmonotone_R = bool(np.any(np.diff(chart.areal_radius[0]) < 0.0))
    valid = chart_validity(chart)
    target_D = np.linspace(0.05, 0.65, 31)
    target_S = np.linspace(0.05, min(2.2, chart.arclength[-1] - 0.05), 47)
    native_z, native_r = inverse_chart_at(chart, target_D, target_S)
    recovered_D, recovered_S = native_to_coordinates(chart, native_z, native_r)
    inverse_error = float(max(
        np.max(np.abs(recovered_D - target_D[:, None])),
        np.max(np.abs(recovered_S - target_S[None, :])),
    ))
    passed = bool(
        base["passed"] and valid["valid"] and arclength_error < 1e-10
        and nonmonotone_R and inverse_error < 5e-5
    )
    return {
        "passed": passed, "test2b_controls": base,
        "proper_arclength_error": arclength_error,
        "nonmonotone_areal_radius_reproduced": nonmonotone_R,
        "proper_coordinate_roundtrip_error": inverse_error,
        "chart": valid,
    }


def load_normal(path):
    with np.load(path) as archive:
        return NormalGeodesicChart(**{key: np.asarray(archive[key]) for key in (
            "distance", "ray_label", "z", "r", "velocity", "areal_radius",
            "speed_squared", "jacobian_DR_zr", "eikonal_qDD",
        )})


def chart_arrays(chart):
    return {
        "distance": chart.distance, "arclength": chart.arclength,
        "native_brane_radius": chart.native_brane_radius,
        "z": chart.z, "r": chart.r, "velocity": chart.velocity,
        "areal_radius": chart.areal_radius, "speed_squared": chart.speed_squared,
        "jacobian_DS_zr": chart.jacobian_DS_zr, "eikonal_qDD": chart.eikonal_qDD,
    }


def load_arclength_chart(path):
    with np.load(path) as archive:
        return ProperArclengthChart(**{key: np.asarray(archive[key]) for key in (
            "distance", "arclength", "native_brane_radius", "z", "r", "velocity",
            "areal_radius", "speed_squared", "jacobian_DS_zr", "eikonal_qDD",
        )})


def state_records(aligned):
    spatial = {
        label: {
            "initial": aligned[f"{label}_initial"],
            "position": aligned[f"{label}_position"], "velocity": aligned[f"{label}_velocity"],
            "linear_position": aligned[f"{label}_linear_position"],
            "linear_velocity": aligned[f"{label}_linear_velocity"],
        } for label in PRIMARY_GRIDS
    }
    temporal = {
        label: {
            "initial": aligned["G10_initial"],
            "position": aligned[f"{label}_position"], "velocity": aligned[f"{label}_velocity"],
            "linear_position": aligned[f"{label}_linear_position"],
            "linear_velocity": aligned[f"{label}_linear_velocity"],
        } for label in TEMPORAL_RUNS
    }
    return spatial, temporal


def relabeled_charts(index, geometries, spatial, temporal):
    charts, diagnostics = {}, {}
    for label, record in {**spatial, **temporal}.items():
        grid = label if label in geometries else "G10"
        z, r = np.asarray(geometries[grid]["z"]), np.asarray(geometries[grid]["r"])
        for slice_name, position in (
            ("initial", record["initial"]), ("pchip", record["position"]),
            ("linear", record["linear_position"]),
        ):
            metric, _ = old.reduced_metric(position, r)
            for resolution in RESOLUTIONS:
                source = OLD_ROOT / f"chart_{label}_{slice_name}_{resolution}.npz"
                normal = load_normal(source)
                stage_id = f"chart/{label}/{slice_name}/{resolution}"
                filename = f"chart_{label}_{slice_name}_{resolution}.npz"
                path, _ = stage_npz(
                    index, stage_id, filename, "proper-arclength-chart",
                    {"source": str(source), "source_sha256": sha256_file(source)},
                    lambda normal=normal, metric=metric, z=z, r=r: chart_arrays(
                        relabel_normal_chart(normal, z, r, metric)
                    ), expected=300.0,
                )
                chart = load_arclength_chart(path)
                checked = chart_validity(chart, coarse=resolution == "coarse")
                charts[(label, slice_name, resolution)] = chart
                diagnostics["/".join((label, slice_name, resolution))] = checked
    return charts, diagnostics


def mapped_state(position, velocity, initial, z, r, initial_chart, final_chart, D, S, method="cubic"):
    zi, ri = inverse_chart_at(initial_chart, D, S)
    zf, rf = inverse_chart_at(final_chart, D, S)
    initial_metric, initial_sphere = old.reduced_metric(initial, r)
    final_metric, final_sphere = old.reduced_metric(position, r)
    mapped_initial = mapped_metric_fields(
        initial_metric, initial_sphere, z, r, D, S, zi, ri, method=method,
    )
    mapped_final = mapped_metric_fields(
        final_metric, final_sphere, z, r, D, S, zf, rf, method=method,
    )
    native_K = adm_extrinsic_curvature_tensor(position, velocity, z, r)
    mapped_K = mapped_extrinsic_fields(
        native_K, final_sphere, z, r, zf, rf, mapped_final, method=method,
    )
    initial4, final4 = old.metric4(mapped_initial), old.metric4(mapped_final)
    return {
        "initial_metric": initial4, "final_metric": final4,
        "metric_increment": final4 - initial4,
        "ADM_K": old.extrinsic4(mapped_K), "trace_K": mapped_K["trace_K"],
        "KijKij": mapped_K["KijKij"],
        "areal_radius": mapped_final["native_areal_radius"],
        "initial_areal_radius": mapped_initial["native_areal_radius"],
        "weight": mapped_final["volume_density"], "native_z": zf, "native_r": rf,
    }


def mapped_variant(records, geometries, charts, D, S, resolution="fine", method="cubic", time="pchip"):
    output = {"distance": D, "arclength": S}
    for label, record in records.items():
        grid = label if label in geometries else "G10"
        z, r = np.asarray(geometries[grid]["z"]), np.asarray(geometries[grid]["r"])
        output[label] = mapped_state(
            record["position" if time == "pchip" else "linear_position"],
            record["velocity" if time == "pchip" else "linear_velocity"],
            record["initial"], z, r,
            charts[(label, "initial", resolution)], charts[(label, time, resolution)],
            D, S, method=method,
        )
    return output


def sequence_score(primary, alternatives, observable):
    pair_names = (("G9", "G10"), ("G10", "G11"))
    errors, uncertainties, q95_errors, q95_uncertainties = [], [], [], []
    raw_differences, published = [], {}
    for left, right in pair_names:
        record = old.paired_summary(
            {"value": primary[left][observable], "weight": primary[left]["weight"]},
            {"value": primary[right][observable], "weight": primary[right]["weight"]},
            primary["distance"], primary["arclength"],
        )
        changes, q95_changes = [], []
        for alternative in alternatives:
            other = old.paired_summary(
                {"value": alternative[left][observable], "weight": alternative[left]["weight"]},
                {"value": alternative[right][observable], "weight": alternative[right]["weight"]},
                primary["distance"], primary["arclength"],
            )
            changes.append(abs(other["absolute_L2"] - record["absolute_L2"]))
            q95_changes.append(abs(other["weighted_q95"] - record["weighted_q95"]))
        scale = record["absolute_L2"] / max(record["relative_L2"], 1e-300)
        q95_scale = record["weighted_q95"] / max(record["relative_weighted_q95"], 1e-300)
        uncertainty = max(*changes, 1e-8 * scale)
        q95_uncertainty = max(*q95_changes, 1e-8 * q95_scale)
        errors.append(record["absolute_L2"])
        uncertainties.append(uncertainty)
        q95_errors.append(record["weighted_q95"])
        q95_uncertainties.append(q95_uncertainty)
        raw_differences.append(record["difference"])
        published[f"{left}_{right}"] = {
            **old.public_pair(record), "uncertainty": uncertainty,
            "weighted_q95_uncertainty": q95_uncertainty,
        }
    order = conservative_order_interval(errors[0], uncertainties[0], errors[1], uncertainties[1])
    q95_order = conservative_order_interval(
        q95_errors[0], q95_uncertainties[0], q95_errors[1], q95_uncertainties[1],
    )
    middle_weight = primary["G10"]["weight"]
    component_shape = (1,) * (raw_differences[0].ndim - 2)
    coherence = sign_coherence(
        raw_differences[0], raw_differences[1],
        middle_weight.reshape((*middle_weight.shape, *component_shape)),
    )
    l2_pass = bool(
        errors[1] + uncertainties[1] < max(errors[0] - uncertainties[0], 0.0)
        and order is not None and order[0] > 1.0
        and uncertainties[1] / max(errors[1], 1e-300) < 0.25
    )
    q95_pass = bool(
        q95_errors[1] + q95_uncertainties[1]
        < max(q95_errors[0] - q95_uncertainties[0], 0.0)
        and q95_order is not None and q95_order[0] > 1.0
        and q95_uncertainties[1] / max(q95_errors[1], 1e-300) < 0.25
    )
    component_growth = []
    if raw_differences[0].ndim > 2:
        for component in np.ndindex(raw_differences[0].shape[2:]):
            first = old.weighted_l2(
                raw_differences[0][(slice(None), slice(None), *component)],
                primary["G10"]["weight"], primary["distance"], primary["arclength"],
            )
            second = old.weighted_l2(
                raw_differences[1][(slice(None), slice(None), *component)],
                primary["G10"]["weight"], primary["distance"], primary["arclength"],
            )
            component_growth.append(bool(second <= 1.05 * first))
    else:
        component_growth.append(bool(errors[1] <= 1.05 * errors[0]))
    component_guard = bool(all(component_growth))
    passed = bool(
        l2_pass and q95_pass and coherence is not None and coherence >= 0.70
        and component_guard
    )
    return {
        "pairs": published, "order_interval": order,
        "weighted_q95_order_interval": q95_order,
        "sign_coherence": coherence, "component_growth_guard": component_guard,
        "L2_passed": l2_pass, "weighted_q95_passed": q95_pass,
        "strict_adverse_monotonicity": bool(
            errors[1] + uncertainties[1] < max(errors[0] - uncertainties[0], 0.0)
        ),
        "map_limited": bool(
            errors[1] <= uncertainties[1]
            or uncertainties[1] / max(errors[1], 1e-300) >= 0.25
        ),
        "passed": passed,
    }


def temporal_score(primary, alternatives, observable, spatial_score):
    pairs = (("G10_coarse", "G10_standard"), ("G10_standard", "G10_half"))
    errors, uncertainties, published = [], [], {}
    for left, right in pairs:
        record = old.paired_summary(
            {"value": primary[left][observable], "weight": primary[left]["weight"]},
            {"value": primary[right][observable], "weight": primary[right]["weight"]},
            primary["distance"], primary["arclength"],
        )
        changes = []
        for alternative in alternatives:
            other = old.paired_summary(
                {"value": alternative[left][observable], "weight": alternative[left]["weight"]},
                {"value": alternative[right][observable], "weight": alternative[right]["weight"]},
                primary["distance"], primary["arclength"],
            )
            changes.append(abs(other["absolute_L2"] - record["absolute_L2"]))
        scale = record["absolute_L2"] / max(record["relative_L2"], 1e-300)
        uncertainty = max(*changes, 1e-8 * scale)
        errors.append(record["absolute_L2"])
        uncertainties.append(uncertainty)
        published[f"{left}_{right}"] = {**old.public_pair(record), "uncertainty": uncertainty}
    if min(errors[0] - uncertainties[0], errors[1] - uncertainties[1]) > 0.0:
        order = (
            math.log2((errors[0] - uncertainties[0]) / (errors[1] + uncertainties[1])),
            math.log2((errors[0] + uncertainties[0]) / (errors[1] - uncertainties[1])),
        )
    else:
        order = None
    spatial_error = spatial_score["pairs"]["G10_G11"]["absolute_L2"]
    spatial_uncertainty = spatial_score["pairs"]["G10_G11"]["uncertainty"]
    separation = errors[1] + uncertainties[1] < 0.5 * max(spatial_error - spatial_uncertainty, 0.0)
    passed = bool(
        errors[1] + uncertainties[1] < max(errors[0] - uncertainties[0], 0.0)
        and order is not None and order[0] > 1.5
        and uncertainties[1] / max(errors[1], 1e-300) < 0.25 and separation
    )
    return {"pairs": published, "order_interval": order, "fine_temporal_below_half_spatial": separation, "passed": passed}


def field_analysis(index, geometries, aligned):
    spatial, temporal = state_records(aligned)
    charts, validity = relabeled_charts(index, geometries, spatial, temporal)
    if not all(item["valid"] for item in validity.values()):
        return {"map_valid": False, "chart_validity": validity}, None, charts
    fine_charts = [
        charts[(label, slice_name, "fine")]
        for label in (*PRIMARY_GRIDS, *TEMPORAL_RUNS)
        for slice_name in ("initial", "pchip")
    ]
    dmax = min(chart.distance[-1] for chart in fine_charts)
    D = np.linspace(0.0, 0.95 * dmax, 193)
    coarsest_spacing = max(
        np.max(np.diff(charts[(label, "initial", "fine")].arclength))
        for label in PRIMARY_GRIDS
    )
    Smin = 2.0 * coarsest_spacing
    Smax = min(arclength_at_native_radius(chart, 6.0) for chart in fine_charts)
    S = np.linspace(Smin, Smax, 257)

    spatial_primary = mapped_variant(spatial, geometries, charts, D, S)
    spatial_alternatives = (
        mapped_variant(spatial, geometries, charts, D, S, resolution="primary"),
        mapped_variant(spatial, geometries, charts, D, S, method="linear"),
        mapped_variant(spatial, geometries, charts, D, S, time="linear"),
    )
    spatial_scores = {
        observable: sequence_score(spatial_primary, spatial_alternatives, observable)
        for observable in ("metric_increment", "final_metric", "ADM_K", "areal_radius")
    }
    temporal_primary = mapped_variant(temporal, geometries, charts, D, S)
    temporal_alternatives = (
        mapped_variant(temporal, geometries, charts, D, S, resolution="primary"),
        mapped_variant(temporal, geometries, charts, D, S, method="linear"),
        mapped_variant(temporal, geometries, charts, D, S, time="linear"),
    )
    temporal_scores = {
        observable: temporal_score(
            temporal_primary, temporal_alternatives, observable, spatial_scores[observable],
        ) for observable in ("metric_increment", "ADM_K")
    }

    extrema = {}
    for label in PRIMARY_GRIDS:
        chart = charts[(label, "pchip", "fine")]
        R = chart.areal_radius[0]
        change = np.diff(R)
        indices = np.where(np.sign(change[:-1]) != np.sign(change[1:]))[0] + 1
        extrema[label] = [
            {"kind": "maximum" if change[index - 1] > 0.0 else "minimum",
             "S": float(chart.arclength[index]), "R": float(R[index])}
            for index in indices
        ]

    arrays = {"distance": D, "arclength": S}
    for label in PRIMARY_GRIDS:
        for observable in (
            "initial_metric", "final_metric", "metric_increment", "ADM_K",
            "trace_K", "KijKij", "areal_radius", "initial_areal_radius",
            "weight", "native_z", "native_r",
        ):
            arrays[f"{label}_{observable}"] = spatial_primary[label][observable]
    archive, _ = stage_npz(
        index, "fields/primary", "fields_primary.npz", "proper-arclength-fields",
        {"D_nodes": len(D), "S_nodes": len(S)}, lambda: arrays, expected=1200.0,
    )
    return {
        "map_valid": True, "chart_validity": validity,
        "comparison_domain": {"D": [float(D[0]), float(D[-1])], "S": [float(S[0]), float(S[-1])]},
        "spatial": spatial_scores, "temporal": temporal_scores,
        "brane_areal_radius_extrema": extrema, "field_archive": str(archive),
    }, spatial_primary, charts


def surface_profile(position, velocity, z, r, surface, chart):
    prepared = prepare_capped_expansion_slice(position, velocity, z, r)
    theta, rho, slope = (np.asarray(surface[key]) for key in ("theta", "rho", "slope"))
    second = dynamical_rho_second(prepared, theta, rho, slope)
    sampled = _sample_geometry(prepared, theta, rho, slope)
    outgoing = local_outgoing_expansion(prepared, theta, rho, slope, second)
    tangent_K = np.einsum("...a,...ab,...b->...", sampled["tangent"], sampled["extrinsic"], sampled["tangent"])
    correction = -tangent_K - 2.0 * sampled["sphere_extrinsic"]
    ingoing = -outgoing + 2.0 * correction
    native_z = z[-1] - rho * np.cos(theta)
    native_r = rho * np.sin(theta)
    D, S = native_to_coordinates(chart, native_z, native_r, method="cubic")
    D_linear, S_linear = native_to_coordinates(chart, native_z, native_r, method="linear")
    sphere = interpolate_regular_field(position[..., 3], z, r, native_z, native_r)
    R = native_r * np.sqrt(sphere)
    speed = np.sqrt(sampled["speed_squared"])
    length = np.concatenate(([0.0], cumulative_trapezoid(speed, x=theta)))
    normalized = length / length[-1]
    target = np.linspace(0.0, 1.0, 501)
    return {
        "s": target, "D": PchipInterpolator(normalized, D)(target),
        "S": PchipInterpolator(normalized, S)(target),
        "D_linear": PchipInterpolator(normalized, D_linear)(target),
        "S_linear": PchipInterpolator(normalized, S_linear)(target),
        "R": PchipInterpolator(normalized, R)(target),
        "theta_minus": PchipInterpolator(normalized, ingoing)(target),
        "theta_plus": PchipInterpolator(normalized, outgoing)(target),
    }


def surface_stage(index, label, branch, position, velocity, geometry, chart):
    stage_id = f"surface/{label}/{branch}"
    json_name, npz_name = f"surface_{label}_{branch}.json", f"surface_{label}_{branch}.npz"
    z, r = np.asarray(geometry["z"]), np.asarray(geometry["r"])
    def compute():
        surface = solve_dynamical_capped_surface_bvp(
            position, velocity, z, r, BRANCH_SEEDS[branch],
            tolerance=2e-5, nodes=121, maximum_nodes=6000, dense_nodes=501,
        )
        if not surface_passes(surface):
            raise RuntimeError(f"{label} {branch} surface failed admission")
        profile = surface_profile(position, velocity, z, r, surface, chart)
        profile_path = RECOVERY_ROOT / npz_name
        atomic_write_npz(profile_path, **profile)
        stability = old._stability_record(position, velocity, z, r, surface)
        return {
            "surface": {key: surface[key] for key in (
                "converged", "in_domain", "rho_axis", "rho_brane",
                "boundary_slope_error", "local_expansion_interior_maximum",
                "primary_evaluator_crosscheck",
            )},
            "geometry": capped_surface_geometry(position, velocity, z, r, surface),
            "stability": stability, "profile_archive": str(profile_path),
            "profile_sha256": sha256_file(profile_path),
        }
    payload, _ = stage_json(
        index, stage_id, json_name, "surface-expansion-stability",
        {"grid": label, "branch": branch}, compute, expected=1800.0,
    )
    return payload


def profile_score(profiles, key):
    differences = [
        float(np.sqrt(simpson((profiles[left][key] - profiles[right][key])**2, x=profiles[left]["s"])))
        for left, right in (("G9", "G10"), ("G10", "G11"))
    ]
    if key in ("D", "S"):
        uncertainties_grid = [
            float(np.sqrt(simpson((profiles[label][key] - profiles[label][f"{key}_linear"])**2, x=profiles[label]["s"])))
            for label in PRIMARY_GRIDS
        ]
        uncertainty = [
            math.hypot(uncertainties_grid[0], uncertainties_grid[1]),
            math.hypot(uncertainties_grid[1], uncertainties_grid[2]),
        ]
    else:
        uncertainty = [0.0, 0.0]
    order = conservative_order_interval(differences[0], uncertainty[0], differences[1], uncertainty[1])
    coherence = None
    if key == "theta_minus":
        coherence = sign_coherence(
            profiles["G9"][key] - profiles["G10"][key],
            profiles["G10"][key] - profiles["G11"][key],
            np.ones_like(profiles["G9"][key]),
        )
    passed = bool(
        differences[1] + uncertainty[1] < max(differences[0] - uncertainty[0], 0.0)
        and order is not None and order[0] > 1.0
        and uncertainty[1] / max(differences[1], 1e-300) < 0.25
        and (key != "theta_minus" or coherence is not None and coherence >= 0.70)
    )
    return {"adjacent_L2_differences": differences, "uncertainty": uncertainty, "order_interval": order, "sign_coherence": coherence, "passed": passed}


def surface_analysis(index, geometries, aligned, charts):
    records = {branch: {} for branch in BRANCH_SEEDS}
    for branch in BRANCH_SEEDS:
        for label in PRIMARY_GRIDS:
            records[branch][label] = surface_stage(
                index, label, branch, aligned[f"{label}_position"],
                aligned[f"{label}_velocity"], geometries[label],
                charts[(label, "pchip", "fine")],
            )
    scores = {}
    for branch, grids in records.items():
        score = {}
        for key in ("one_sided_cap_area", "equivalent_area_radius", "proper_meridional_length", "maximum_sphere_radius"):
            score[key] = old.scalar_tail([grids[label]["geometry"][key] for label in PRIMARY_GRIDS])
        eigenvalues = [grids[label]["stability"]["eigenvalue"] for label in PRIMARY_GRIDS]
        eigen_errors = [max(grids[label]["stability"]["angular_error"], grids[label]["stability"]["step_error"]) for label in PRIMARY_GRIDS]
        score["stability"] = old.scalar_tail(eigenvalues, eigen_errors)
        score["stability"]["classifications"] = [grids[label]["stability"]["classification"] for label in PRIMARY_GRIDS]
        score["spectral_gap"] = old.scalar_tail([grids[label]["stability"]["spectral_gap"] for label in PRIMARY_GRIDS])
        profiles = {}
        for label in PRIMARY_GRIDS:
            with np.load(grids[label]["profile_archive"]) as archive:
                profiles[label] = {key: np.asarray(archive[key]) for key in archive.files}
        for key in ("D", "S", "R", "theta_minus"):
            score[key] = profile_score(profiles, key)
        for endpoint in (0, -1):
            for key in ("D", "S", "R"):
                score[f"{key}_endpoint_{endpoint}"] = old.scalar_tail([profiles[label][key][endpoint] for label in PRIMARY_GRIDS])
        score["outgoing_residuals"] = [grids[label]["surface"]["local_expansion_interior_maximum"] for label in PRIMARY_GRIDS]
        scores[branch] = score
    inner = scores["inner"]["stability"]["classifications"]
    outer = scores["outer"]["stability"]["classifications"]
    passed = bool(
        all(value == "outward_unstable" for value in inner)
        and all(value == "outward_stable" for value in outer)
        and all(item["passed"] for score in scores.values() for key, item in score.items() if key != "outgoing_residuals")
    )
    return {"records": records, "scores": scores, "passed": passed}


def common_parent_analysis():
    results, initial_max, evolved_max, maps_valid = {}, 0.0, 0.0, True
    with np.load(COMMON_PARENT_STATE) as archive:
        for grid in ("G7", "G8"):
            z = np.asarray(archive[f"{grid}_R12_z"])
            records, charts = {}, {}
            for domain in (8, 10, 12):
                r = np.asarray(archive[f"{grid}_R{domain}_r"])
                initial = np.asarray(archive[f"{grid}_R{domain}_initial"])
                position = np.asarray(archive[f"{grid}_R{domain}_step_016_position"])
                velocity = np.asarray(archive[f"{grid}_R{domain}_step_016_velocity"])
                records[domain] = {"r": r, "initial": initial, "position": position, "velocity": velocity}
            dmax = old.common_distance_limit(
                [(f"R{domain}", record[slice_name]) for domain, record in records.items() for slice_name in ("initial", "position")],
                {f"R{domain}": (z, record["r"]) for domain, record in records.items()},
            )
            for domain, record in records.items():
                initial_metric, initial_sphere = old.reduced_metric(record["initial"], record["r"])
                final_metric, final_sphere = old.reduced_metric(record["position"], record["r"])
                normal_initial = old.build_normal_geodesic_chart(z, record["r"], initial_metric, initial_sphere, np.linspace(0.0, dmax, 193), ray_count=len(record["r"]))
                normal_final = old.build_normal_geodesic_chart(z, record["r"], final_metric, final_sphere, np.linspace(0.0, dmax, 193), ray_count=len(record["r"]))
                charts[domain] = (
                    relabel_normal_chart(normal_initial, z, record["r"], initial_metric),
                    relabel_normal_chart(normal_final, z, record["r"], final_metric),
                )
                maps_valid = bool(maps_valid and all(chart_validity(chart)["valid"] for chart in charts[domain]))
            D = np.linspace(0.0, 0.95 * dmax, 129)
            Smin = 2.0 * max(np.diff(charts[8][0].arclength))
            Smax = min(arclength_at_native_radius(chart, 6.0) for pair in charts.values() for chart in pair)
            S = np.linspace(Smin, Smax, 193)
            mapped = {}
            for domain, record in records.items():
                mapped[domain] = mapped_state(
                    record["position"], record["velocity"], record["initial"],
                    z, record["r"], charts[domain][0], charts[domain][1], D, S,
                )
            grid_result = {}
            for left, right in ((8, 10), (10, 12)):
                pair = {}
                for observable in ("initial_metric", "metric_increment", "final_metric", "ADM_K"):
                    score = old.paired_summary(
                        {"value": mapped[left][observable], "weight": mapped[left]["weight"]},
                        {"value": mapped[right][observable], "weight": mapped[right]["weight"]}, D, S,
                    )
                    pair[observable] = old.public_pair(score)
                    if observable == "initial_metric":
                        initial_max = max(initial_max, score["relative_L2"])
                    else:
                        evolved_max = max(evolved_max, score["relative_L2"])
                grid_result[f"R{left}_R{right}"] = pair
            results[grid] = grid_result
    return {
        "maps_valid": maps_valid, "mapped_initial_relative_maximum": initial_max,
        "mapped_evolved_relative_maximum": evolved_max, "pairs": results,
        "passed": bool(maps_valid and initial_max < 1e-8 and evolved_max < 1e-3),
    }


def independent_audit(fields):
    checks = {}
    passed = True
    for observable, record in fields["spatial"].items():
        p12, p23 = record["pairs"]["G9_G10"], record["pairs"]["G10_G11"]
        recomputed = conservative_order_interval(
            p12["absolute_L2"], p12["uncertainty"], p23["absolute_L2"], p23["uncertainty"],
        )
        stored = record["order_interval"]
        l2_agrees = bool(
            recomputed is None and stored is None
            or recomputed is not None and stored is not None
            and np.allclose(recomputed, stored, rtol=1e-12, atol=1e-14)
        )
        recomputed_q95 = conservative_order_interval(
            p12["weighted_q95"], p12["weighted_q95_uncertainty"],
            p23["weighted_q95"], p23["weighted_q95_uncertainty"],
        )
        stored_q95 = record["weighted_q95_order_interval"]
        q95_agrees = bool(
            recomputed_q95 is None and stored_q95 is None
            or recomputed_q95 is not None and stored_q95 is not None
            and np.allclose(recomputed_q95, stored_q95, rtol=1e-12, atol=1e-14)
        )
        agrees = bool(l2_agrees and q95_agrees)
        checks[observable] = {
            "stored_L2": stored, "recomputed_L2": recomputed,
            "stored_q95": stored_q95, "recomputed_q95": recomputed_q95,
            "agrees": agrees,
        }
        passed = passed and agrees
    return {"passed": bool(passed), "order_checks": checks}


def grade(controls, fields, surfaces, formation, common_parent, audit):
    map_pass = bool(fields.get("map_valid"))
    if not controls["passed"] or not map_pass:
        return {"status": "REVIEW", "classification": "invalid_proper_arclength_chart_audit"}
    field_pass = bool(all(record["passed"] for record in fields["spatial"].values()))
    temporal_pass = bool(all(record["passed"] for record in fields["temporal"].values()))
    if field_pass and temporal_pass and surfaces["passed"] and formation["passed"] and common_parent["passed"] and audit["passed"]:
        return {"status": "PASS", "classification": "proper_arclength_chart_above_first_order_continuum_evidence"}
    metric = fields["spatial"]["metric_increment"]
    adm = fields["spatial"]["ADM_K"]
    if (
        not metric["strict_adverse_monotonicity"] and not adm["strict_adverse_monotonicity"]
        and not metric["map_limited"] and not adm["map_limited"]
    ):
        return {"status": "FAIL", "classification": "proper_arclength_chart_nonconvergence_or_branch_failure"}
    return {"status": "REVIEW", "classification": "proper_arclength_chart_convergence_mixed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("controls", "fields", "surfaces", "all"), default="all")
    args = parser.parse_args()
    if sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("sealed Test-2C protocol hash mismatch")
    RECOVERY_ROOT.mkdir(parents=True, exist_ok=True)
    index = RecoveryIndex(MANIFEST, PROTOCOL, recovery_inputs(), maximum_stage_seconds=2400.0)
    reuse, _ = stage_json(index, "qualification/reuse", "reuse.json", "old-recovery-qualification", {}, lambda: {"reuse": validate_old_recovery()}, expected=120.0)
    controls, _ = stage_json(index, "controls/manufactured", "controls.json", "manufactured-controls", {}, lambda: {"controls": manufactured_controls()}, expected=1800.0)
    controls = controls["controls"]
    if not controls["passed"]:
        raise RuntimeError("Test-2C controls failed")
    if args.phase == "controls":
        print(json.dumps({"phase": "controls", "passed": True, "recovery": recovery_summary(index)}, indent=2))
        return
    print("reconstructing G7--G11 geometry metadata", flush=True)
    geometries = build_all_geometries()
    with np.load(ALIGNED_STATE) as archive:
        aligned = {key: np.asarray(archive[key]) for key in archive.files}
    fields_payload, _, charts = field_analysis(index, geometries, aligned)
    fields, _ = stage_json(index, "analysis/fields", "fields.json", "proper-arclength-field-analysis", {}, lambda: {"fields": fields_payload}, expected=1200.0)
    fields = fields["fields"]
    if not fields.get("map_valid") or args.phase == "fields":
        print(json.dumps({"phase": "fields", "map_valid": fields.get("map_valid"), "spatial": fields.get("spatial"), "temporal": fields.get("temporal"), "recovery": recovery_summary(index)}, indent=2))
        return
    surfaces_payload = surface_analysis(index, geometries, aligned, charts)
    surfaces, _ = stage_json(index, "analysis/surfaces", "surfaces.json", "surface-analysis", {}, lambda: {"surfaces": surfaces_payload}, expected=1200.0)
    surfaces = surfaces["surfaces"]
    if args.phase == "surfaces":
        print(json.dumps({"phase": "surfaces", "passed": surfaces["passed"], "recovery": recovery_summary(index)}, indent=2))
        return
    formation, _ = stage_json(index, "formation/history", "formation.json", "proper-time-formation", {}, lambda: {"formation": old.formation_analysis(ALIGNED_STATE)}, expected=120.0)
    formation = formation["formation"]
    common_parent, _ = stage_json(index, "common_parent/control", "common_parent.json", "common-parent-control", {}, lambda: {"common_parent": common_parent_analysis()}, expected=1800.0)
    common_parent = common_parent["common_parent"]
    audit, _ = stage_json(index, "audit/independent", "independent_audit.json", "independent-audit", {}, lambda: {"audit": independent_audit(fields)}, expected=120.0)
    audit = audit["audit"]
    verdict = grade(controls, fields, surfaces, formation, common_parent, audit)
    result = {
        "protocol": str(PROTOCOL), "protocol_sha256": PROTOCOL_SHA256,
        **verdict, "controls": controls, "reuse": reuse["reuse"],
        "fields": fields, "surfaces": surfaces, "formation": formation,
        "common_parent": common_parent, "independent_audit": audit,
        "claim_boundary": "tested invariant numerical continuum behavior only",
    }
    atomic_write_json(OUTPUT, result)
    atomic_write_npz(STATE_OUTPUT, protocol_sha256=np.asarray(PROTOCOL_SHA256), status=np.asarray(verdict["status"]), classification=np.asarray(verdict["classification"]))
    final, _ = stage_json(index, "final/result", "final.json", "final-result", {}, lambda: {"status": verdict["status"], "classification": verdict["classification"], "output": str(OUTPUT), "output_sha256": sha256_file(OUTPUT), "state": str(STATE_OUTPUT), "state_sha256": sha256_file(STATE_OUTPUT)}, expected=120.0)
    print(json.dumps({**verdict, "final": final, "recovery": recovery_summary(index)}, indent=2))


if __name__ == "__main__":
    main()
