#!/usr/bin/env python3
"""High-order invariant field and temporal lanes for sealed A790 Test 2D."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import run_corrected_A790_test2b_invariant_convergence as old
from bhps.corrected_A790_physical_tensor_convergence import adm_extrinsic_curvature_tensor
from bhps.high_order_invariant_interpolation import (
    INTERPOLATION_ORDER_GATE,
    SAFETY_FACTOR,
    endpoint_preserving_indices,
    leave_level_out,
    mapped_extrinsic_fields,
    mapped_metric_fields,
)
from bhps.invariant_physical_chart import conservative_order_interval, sign_coherence
from bhps.invariant_proper_arclength_chart import (
    ProperArclengthChart,
    arclength_at_native_radius,
    inverse_chart_at,
)
from bhps.ragged_normal_arclength_chart import load_ragged_chart, ragged_chart_to_native
from bhps.recovery_indexer import RecoveryIndex, atomic_write_json, atomic_write_npz, sha256_file, validate_npz


PROTOCOL = Path("notes/110_A790_test2D_high_order_ragged_chart_protocol.md")
PROTOCOL_SHA256 = "f11989b23cff2d5b87bf6e730ff91c77b1e095eb2a7a49d1a277a9d3bf2666e5"
MACHINE_PROTOCOL = Path("results/corrected_A790_test2d_high_order_ragged_chart_protocol.json")
MACHINE_PROTOCOL_SHA256 = "d3a1f59c61499d5e8f67fcd29a4592a7c1afd65b74e7bc90d78d0a35121734e4"
CHART_MANIFEST = Path("results/corrected_A790_test2d_high_order_ragged_chart_recovery/chart_index_v2.json")
# Filled only after all 90 prospective charts are complete and immutable.
CHART_MANIFEST_SHA256 = None
ALIGNED = Path("results/corrected_A790_test2b_invariant_convergence_recovery/aligned_states.npz")
ALIGNED_SHA256 = "42ec6dcb65038a6aa7ab1fc724e848bbba7be781b887cd18f3512cc030256265"
GEOMETRIES = Path("results/corrected_A790_test2b_invariant_convergence_recovery/geometries.npz")
GEOMETRIES_SHA256 = "fd5cf03f1fb37b9e9f7d1f14f8112c2452ccb4f0f6d8438d251d1432bb2dd618"
RECT_ROOT = Path("results/corrected_A790_test2c_proper_arclength_convergence_recovery")
RECT_MANIFEST = RECT_ROOT / "index.json"
RECT_MANIFEST_SHA256 = "eebf285a37a3c173457b7ef150a13587d1fbf087624e24cd09c12ac981e77c75"
RAGGED_ROOT = Path("results/corrected_A790_test2d_high_order_ragged_chart_recovery")
ROOT = Path("results/corrected_A790_test2d_high_order_ragged_chart_field_recovery")
MANIFEST = ROOT / "index.json"
OUTPUT = Path("results/corrected_A790_test2d_high_order_ragged_chart_fields.json")
STATE_OUTPUT = Path("results/corrected_A790_test2d_high_order_ragged_chart_fields_state.npz")
SPATIAL = ("G9", "G10", "G11")
TEMPORAL = ("G10_coarse", "G10_standard", "G10_half")
OBSERVABLES = ("final_metric", "metric_increment", "ADM_K", "areal_radius")
NATIVE_FAMILIES = ("initial_metric", "final_metric", "ADM_K", "sphere_factor")


def recovery_inputs():
    if not isinstance(CHART_MANIFEST_SHA256, str):
        raise RuntimeError("field runner is not sealed to a completed chart manifest")
    fixed = {
        str(MACHINE_PROTOCOL): MACHINE_PROTOCOL_SHA256,
        str(CHART_MANIFEST): CHART_MANIFEST_SHA256,
        str(ALIGNED): ALIGNED_SHA256, str(GEOMETRIES): GEOMETRIES_SHA256,
        str(RECT_MANIFEST): RECT_MANIFEST_SHA256,
    }
    dynamic = (
        Path(__file__), Path("src/bhps/high_order_invariant_interpolation.py"),
        Path("src/bhps/ragged_normal_arclength_chart.py"),
        Path("src/bhps/recovery_indexer.py"),
    )
    return {**fixed, **{str(path): sha256_file(path) for path in dynamic}}


def stage_json(index, stage_id, filename, kind, metadata, producer, expected=21600.0):
    path = ROOT / filename
    index.register(stage_id, kind, expected, metadata)
    cached = index.validated_path(stage_id)
    if cached is not None:
        return json.loads(cached.read_text())
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        payload = {"stage_id": stage_id, "protocol_sha256": index.protocol_sha256, **producer()}
        atomic_write_json(path, payload)
        index.mark_complete(stage_id, path, time.perf_counter() - started)
        return payload
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def stage_npz(index, stage_id, filename, kind, metadata, producer, expected=21600.0):
    path = ROOT / filename
    index.register(stage_id, kind, expected, metadata)
    cached = index.validated_path(stage_id)
    if cached is not None:
        return cached
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        atomic_write_npz(path, **producer())
        validate_npz(path)
        index.mark_complete(stage_id, path, time.perf_counter() - started)
        return path
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def load_geometry(grid):
    with np.load(GEOMETRIES) as archive:
        return np.asarray(archive[f"{grid}_z"]), np.asarray(archive[f"{grid}_r"])


def load_record(label):
    grid = label if label in SPATIAL else "G10"
    with np.load(ALIGNED) as archive:
        return {
            "initial": np.asarray(archive[f"{grid}_initial"]),
            "position": np.asarray(archive[f"{label}_position"]),
            "velocity": np.asarray(archive[f"{label}_velocity"]),
            "linear_position": np.asarray(archive[f"{label}_linear_position"]),
            "linear_velocity": np.asarray(archive[f"{label}_linear_velocity"]),
        }


def rect_chart(label, slice_name, resolution):
    path = RECT_ROOT / f"chart_{label}_{slice_name}_{resolution}.npz"
    with np.load(path) as archive:
        return ProperArclengthChart(**{
            key: np.asarray(archive[key]) for key in (
                "distance", "arclength", "native_brane_radius", "z", "r",
                "velocity", "areal_radius", "speed_squared",
                "jacobian_DS_zr", "eikonal_qDD",
            )
        })


def ragged_chart(label, slice_name, resolution="fine"):
    path = RAGGED_ROOT / f"chart_aligned_{label}_{slice_name}_{resolution}.npz"
    return load_ragged_chart(path)


def comparison_domain():
    charts = [
        rect_chart(label, slice_name, "fine")
        for label in (*SPATIAL, *TEMPORAL) for slice_name in ("initial", "pchip")
    ]
    dmax = min(float(chart.distance[-1]) for chart in charts)
    D = np.linspace(0.0, 0.95 * dmax, 193)
    spacing = max(
        float(np.max(np.diff(rect_chart(label, "initial", "fine").arclength)))
        for label in SPATIAL
    )
    Smin = 2.0 * spacing
    Smax = min(arclength_at_native_radius(chart, 6.0) for chart in charts)
    return D, np.linspace(Smin, Smax, 257)


def native_bundle(position, velocity, initial, z, r):
    initial_metric, initial_sphere = old.reduced_metric(initial, r)
    final_metric, final_sphere = old.reduced_metric(position, r)
    native_K = adm_extrinsic_curvature_tensor(position, velocity, z, r)
    determinant = np.linalg.det(final_metric)
    radius = np.broadcast_to(r[None, :], determinant.shape)
    native_weight = 4.0 * math.pi * (radius * np.sqrt(final_sphere))**2 * np.sqrt(determinant)
    return {
        "initial_metric": initial_metric, "initial_sphere": initial_sphere,
        "final_metric": final_metric, "sphere_factor": final_sphere,
        "ADM_K": native_K, "native_weight": native_weight,
    }


def _metric4(mapped):
    shape = mapped["covariant"].shape[:2]
    value = np.zeros((*shape, 4, 4))
    value[..., :2, :2] = mapped["covariant"]
    value[..., 2, 2] = value[..., 3, 3] = 1.0
    return value


def _extrinsic4(mapped):
    shape = mapped["K_DD"].shape
    value = np.zeros((*shape, 4, 4))
    value[..., 0, 0] = mapped["K_DD"]
    value[..., 0, 1] = value[..., 1, 0] = mapped["K_DS"]
    value[..., 1, 1] = mapped["K_SS"]
    value[..., 2, 2] = value[..., 3, 3] = mapped["K_Omega"]
    return value


def map_bundle(bundle, z, r, D, S, zi, ri, zf, rf, method="quintic", iz=None, ir=None):
    if iz is None:
        iz = np.arange(len(z))
    if ir is None:
        ir = np.arange(len(r))
    z_source, r_source = z[iz], r[ir]
    select = np.ix_(iz, ir)
    initial = mapped_metric_fields(
        bundle["initial_metric"][select], bundle["initial_sphere"][select],
        z_source, r_source, D, S, zi, ri, method,
    )
    final = mapped_metric_fields(
        bundle["final_metric"][select], bundle["sphere_factor"][select],
        z_source, r_source, D, S, zf, rf, method,
    )
    extrinsic = mapped_extrinsic_fields(
        bundle["ADM_K"][select], bundle["sphere_factor"][select],
        z_source, r_source, zf, rf, final, method,
    )
    initial4, final4 = _metric4(initial), _metric4(final)
    return {
        "initial_metric": initial4, "final_metric": final4,
        "metric_increment": final4 - initial4, "ADM_K": _extrinsic4(extrinsic),
        "areal_radius": final["native_areal_radius"],
        "trace_K": extrinsic["trace_K"], "KijKij": extrinsic["KijKij"],
        "weight": final["volume_density"], "native_z": zf, "native_r": rf,
    }


def coordinates(label, D, S, resolution="fine", time="pchip", use_ragged=False):
    if use_ragged:
        DD, SS = np.broadcast_arrays(D[:, None], S[None, :])
        zi, ri = ragged_chart_to_native(ragged_chart(label, "initial", resolution), DD, SS)
        zf, rf = ragged_chart_to_native(ragged_chart(label, time, resolution), DD, SS)
        return zi, ri, zf, rf
    zi, ri = inverse_chart_at(rect_chart(label, "initial", resolution), D, S)
    zf, rf = inverse_chart_at(rect_chart(label, time, resolution), D, S)
    return zi, ri, zf, rf


def difference_norm(reference, alternative, weight, D, S):
    record = old.paired_summary(
        {"value": reference, "weight": weight},
        {"value": alternative, "weight": weight}, D, S,
    )
    return {"L2": float(record["absolute_L2"]), "q95": float(record["weighted_q95"])}


def compact_leave(record):
    return {
        "envelopes": record["envelopes"], "orders": record["orders"],
        "roundoff_floor": record["roundoff_floor"], "admissible": record["admissible"],
    }


def state_analysis(label, D, S):
    grid = label if label in SPATIAL else "G10"
    z, r = load_geometry(grid)
    record = load_record(label)
    bundle = native_bundle(record["position"], record["velocity"], record["initial"], z, r)
    zi, ri, zf, rf = coordinates(label, D, S)
    primary = map_bundle(bundle, z, r, D, S, zi, ri, zf, rf)
    independent = map_bundle(bundle, z, r, D, S, zi, ri, zf, rf, "independent5")
    cubic = map_bundle(bundle, z, r, D, S, zi, ri, zf, rf, "cubic")
    linear = map_bundle(bundle, z, r, D, S, zi, ri, zf, rf, "linear")
    primary_coordinates = coordinates(label, D, S, resolution="primary")
    map_primary = map_bundle(bundle, z, r, D, S, *primary_coordinates)
    ragged_coordinates = coordinates(label, D, S, use_ragged=True)
    map_independent = map_bundle(bundle, z, r, D, S, *ragged_coordinates)
    linear_bundle = native_bundle(
        record["linear_position"], record["linear_velocity"], record["initial"], z, r,
    )
    time_coordinates = coordinates(label, D, S, time="linear")
    time_linear = map_bundle(linear_bundle, z, r, D, S, *time_coordinates)

    native_leave = {
        "initial_metric": compact_leave(leave_level_out(
            bundle["initial_metric"], z, r, bundle["native_weight"],
        )),
        "final_metric": compact_leave(leave_level_out(
            bundle["final_metric"], z, r, bundle["native_weight"],
        )),
        "ADM_K": compact_leave(leave_level_out(
            bundle["ADM_K"], z, r, bundle["native_weight"],
        )),
        "sphere_factor": compact_leave(leave_level_out(
            bundle["sphere_factor"], z, r, bundle["native_weight"],
        )),
    }
    mapped_records = {
        observable: {"2": [], "4": []} for observable in OBSERVABLES
    }
    mapped_envelopes = {
        observable: {"2": {"L2": 0.0, "q95": 0.0}, "4": {"L2": 0.0, "q95": 0.0}}
        for observable in OBSERVABLES
    }
    for stride in (2, 4):
        for offset_z in range(stride):
            iz = endpoint_preserving_indices(len(z), stride, offset_z)
            for offset_r in range(stride):
                ir = endpoint_preserving_indices(len(r), stride, offset_r)
                variant = map_bundle(bundle, z, r, D, S, zi, ri, zf, rf, iz=iz, ir=ir)
                for observable in OBSERVABLES:
                    norms = difference_norm(
                        primary[observable], variant[observable], primary["weight"], D, S,
                    )
                    mapped_records[observable][str(stride)].append({
                        "offset_z": offset_z, "offset_r": offset_r, **norms,
                    })
                    for norm in ("L2", "q95"):
                        mapped_envelopes[observable][str(stride)][norm] = max(
                            mapped_envelopes[observable][str(stride)][norm], norms[norm],
                        )

    allowances = {}
    for observable in OBSERVABLES:
        delta55 = difference_norm(
            primary[observable], independent[observable], primary["weight"], D, S,
        )
        delta53 = difference_norm(
            primary[observable], cubic[observable], primary["weight"], D, S,
        )
        adverse = difference_norm(
            primary[observable], linear[observable], primary["weight"], D, S,
        )
        map_changes = (
            difference_norm(primary[observable], map_primary[observable], primary["weight"], D, S),
            difference_norm(primary[observable], map_independent[observable], primary["weight"], D, S),
        )
        time_change = difference_norm(
            primary[observable], time_linear[observable], primary["weight"], D, S,
        )
        norms = {}
        mapped_admissible = True
        for norm in ("L2", "q95"):
            e2 = mapped_envelopes[observable]["2"][norm]
            e4 = mapped_envelopes[observable]["4"][norm]
            order = math.log2(e4 / e2) if e2 > 0.0 and e4 > e2 else None
            mapped_admissible = bool(
                mapped_admissible and order is not None and order > INTERPOLATION_ORDER_GATE
            )
            richardson = (
                e2 / (2.0**order - 1.0)
                if order is not None and order > INTERPOLATION_ORDER_GATE else math.inf
            )
            interpolation = SAFETY_FACTOR * max(richardson, delta55[norm], delta53[norm])
            map_error = max(item[norm] for item in map_changes)
            scale = old.weighted_l2(primary[observable], primary["weight"], D, S)
            roundoff = 1e-8 * max(scale, 1e-300)
            total = map_error + interpolation + time_change[norm] + roundoff
            norms[norm] = {
                "leave_out_e2": e2, "leave_out_e4": e4,
                "leave_out_order": order, "richardson": richardson,
                "production_vs_independent_degree5": delta55[norm],
                "quintic_vs_cubic": delta53[norm], "linear_adverse": adverse[norm],
                "map": map_error, "interpolation": interpolation,
                "time": time_change[norm], "parent": 0.0, "roundoff": roundoff,
                "total": total,
            }
        native_admissible = bool(all(item["admissible"] for item in native_leave.values()))
        allowances[observable] = {
            "admissible": bool(mapped_admissible and native_admissible),
            "mapped_leave_out_records": mapped_records[observable],
            "norms": norms,
        }
    arrays = {"distance": D, "arclength": S}
    for observable in (
        "initial_metric", "final_metric", "metric_increment", "ADM_K",
        "areal_radius", "trace_K", "KijKij", "weight", "native_z", "native_r",
    ):
        arrays[observable] = primary[observable]
    return arrays, {
        "label": label, "grid": grid, "native_leave_out": native_leave,
        "allowances": allowances,
    }


def load_state(label):
    path = ROOT / f"state_{label}.npz"
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def load_allowance(label):
    return json.loads((ROOT / f"allowance_{label}.json").read_text())["analysis"]


def sequence_score(labels, observable, temporal=False, spatial_score=None):
    states = {label: load_state(label) for label in labels}
    allowances = {label: load_allowance(label)["allowances"][observable] for label in labels}
    pairs = list(zip(labels[:-1], labels[1:]))
    errors = {"L2": [], "q95": []}
    uncertainties = {"L2": [], "q95": []}
    differences, published = [], {}
    for left, right in pairs:
        summary = old.paired_summary(
            {"value": states[left][observable], "weight": states[left]["weight"]},
            {"value": states[right][observable], "weight": states[right]["weight"]},
            states[left]["distance"], states[left]["arclength"],
        )
        differences.append(summary["difference"])
        pair_record = old.public_pair(summary)
        for norm, key in (("L2", "absolute_L2"), ("q95", "weighted_q95")):
            error = float(summary[key])
            uncertainty = float(
                allowances[left]["norms"][norm]["total"]
                + allowances[right]["norms"][norm]["total"]
            )
            errors[norm].append(error)
            uncertainties[norm].append(uncertainty)
            pair_record[f"{norm}_uncertainty"] = uncertainty
        published[f"{left}_{right}"] = pair_record
    orders, norm_pass = {}, {}
    for norm in ("L2", "q95"):
        if temporal:
            if min(errors[norm][0] - uncertainties[norm][0], errors[norm][1] - uncertainties[norm][1]) > 0.0:
                order = (
                    math.log2((errors[norm][0] - uncertainties[norm][0]) / (errors[norm][1] + uncertainties[norm][1])),
                    math.log2((errors[norm][0] + uncertainties[norm][0]) / (errors[norm][1] - uncertainties[norm][1])),
                )
            else:
                order = None
            minimum_order = 1.5
        else:
            order = conservative_order_interval(
                errors[norm][0], uncertainties[norm][0],
                errors[norm][1], uncertainties[norm][1],
            )
            minimum_order = 1.0
        orders[norm] = order
        norm_pass[norm] = bool(
            errors[norm][1] + uncertainties[norm][1]
            < max(errors[norm][0] - uncertainties[norm][0], 0.0)
            and order is not None and order[0] > minimum_order
            and uncertainties[norm][1] / max(errors[norm][1], 1e-300) < 0.25
        )
    weight = states[labels[1]]["weight"]
    component_shape = (1,) * (differences[0].ndim - 2)
    coherence = sign_coherence(
        differences[0], differences[1],
        weight.reshape((*weight.shape, *component_shape)),
    )
    component_guard = True
    if differences[0].ndim > 2:
        for component in np.ndindex(differences[0].shape[2:]):
            first = old.weighted_l2(
                differences[0][(..., *component)], weight,
                states[labels[1]]["distance"], states[labels[1]]["arclength"],
            )
            second = old.weighted_l2(
                differences[1][(..., *component)], weight,
                states[labels[1]]["distance"], states[labels[1]]["arclength"],
            )
            if second > 1.05 * first + uncertainties["L2"][1]:
                component_guard = False
    else:
        component_guard = errors["L2"][1] <= 1.05 * errors["L2"][0] + uncertainties["L2"][1]
    all_admissible = bool(all(allowances[label]["admissible"] for label in labels))
    separation = None
    if temporal:
        spatial_pair = spatial_score["pairs"]["G10_G11"]
        separation = bool(
            errors["L2"][1] + uncertainties["L2"][1]
            < 0.5 * max(spatial_pair["absolute_L2"] - spatial_pair["L2_uncertainty"], 0.0)
        )
    passed = bool(
        all_admissible and all(norm_pass.values())
        and coherence is not None and coherence >= 0.70 and component_guard
        and (not temporal or separation)
    )
    return {
        "pairs": published, "order_intervals": orders,
        "norm_passed": norm_pass, "sign_coherence": coherence,
        "component_growth_guard": bool(component_guard),
        "all_state_estimators_admissible": all_admissible,
        "fine_temporal_below_half_spatial": separation,
        "strict_adverse_monotonicity": bool(
            errors["L2"][1] + uncertainties["L2"][1]
            < max(errors["L2"][0] - uncertainties["L2"][0], 0.0)
        ),
        "map_limited": bool(any(
            allowances[label]["norms"]["L2"]["map"] > 0.20 * max(errors["L2"][1], 1e-300)
            for label in labels[-2:]
        )),
        "interpolation_limited": bool(any(
            allowances[label]["norms"]["L2"]["interpolation"] > 0.20 * max(errors["L2"][1], 1e-300)
            for label in labels[-2:]
        )),
        "passed": passed,
    }


def grade(spatial, temporal):
    spatial_pass = all(record["passed"] for record in spatial.values())
    temporal_pass = all(record["passed"] for record in temporal.values())
    if spatial_pass and temporal_pass:
        return {"status": "PASS", "classification": "test2d_field_lane_above_first_order"}
    metric, adm = spatial["metric_increment"], spatial["ADM_K"]
    if (
        not metric["strict_adverse_monotonicity"]
        and not adm["strict_adverse_monotonicity"]
        and not metric["map_limited"] and not adm["map_limited"]
        and not metric["interpolation_limited"] and not adm["interpolation_limited"]
    ):
        return {"status": "FAIL", "classification": "test2d_field_lane_resolved_nonconvergence"}
    return {"status": "REVIEW", "classification": "test2d_field_lane_convergence_mixed"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-states", type=int, default=None)
    args = parser.parse_args()
    if sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("sealed Test-2D protocol hash mismatch")
    chart_manifest = json.loads(CHART_MANIFEST.read_text())
    chart_records = [value for key, value in chart_manifest["stages"].items() if key.startswith("chart/")]
    if len(chart_records) != 90 or not all(item.get("status") == "complete" for item in chart_records):
        raise RuntimeError("all 90 prospective Test-2D charts must complete before fields")
    if not all(item.get("completion_metadata", {}).get("validity", {}).get("valid") for item in chart_records):
        raise RuntimeError("a prospectively required Test-2D chart is invalid")
    ROOT.mkdir(parents=True, exist_ok=True)
    index = RecoveryIndex(MANIFEST, PROTOCOL, recovery_inputs(), maximum_stage_seconds=21600.0)
    D, S = comparison_domain()
    labels = list((*SPATIAL, *TEMPORAL))
    pending = [
        label for label in labels
        if index.validated_path(f"fields/{label}/state") is None
        or index.validated_path(f"fields/{label}/allowance") is None
    ]
    if args.max_new_states is not None:
        pending = pending[:max(args.max_new_states, 0)]
    for label in pending:
        arrays, analysis = state_analysis(label, D, S)
        state_path = stage_npz(
            index, f"fields/{label}/state", f"state_{label}.npz",
            "test2d-primary-high-order-state", {"label": label},
            lambda arrays=arrays: arrays,
        )
        stage_json(
            index, f"fields/{label}/allowance", f"allowance_{label}.json",
            "test2d-separated-state-allowance", {"label": label},
            lambda analysis=analysis: {"analysis": analysis},
        )
        print(json.dumps({"state": label, "archive": str(state_path), "complete": True}), flush=True)
    if any(
        index.validated_path(f"fields/{label}/state") is None
        or index.validated_path(f"fields/{label}/allowance") is None
        for label in labels
    ):
        print(json.dumps({"phase": "fields", "complete_states": sum(
            index.validated_path(f"fields/{label}/state") is not None for label in labels
        ), "total_states": len(labels), "physical_verdict": None}, indent=2))
        return
    spatial = {
        observable: sequence_score(SPATIAL, observable) for observable in OBSERVABLES
    }
    temporal = {
        observable: sequence_score(TEMPORAL, observable, temporal=True, spatial_score=spatial[observable])
        for observable in ("metric_increment", "ADM_K")
    }
    verdict = grade(spatial, temporal)
    result = {
        "protocol": str(PROTOCOL), "protocol_sha256": PROTOCOL_SHA256,
        **verdict, "spatial": spatial, "temporal": temporal,
        "comparison_domain": {"D": [float(D[0]), float(D[-1])], "S": [float(S[0]), float(S[-1])]},
        "state_allowances": {label: load_allowance(label) for label in labels},
        "claim_boundary": "Test-2D invariant numerical field lane only",
    }
    atomic_write_json(OUTPUT, result)
    atomic_write_npz(
        STATE_OUTPUT, protocol_sha256=np.asarray(PROTOCOL_SHA256),
        status=np.asarray(verdict["status"]), classification=np.asarray(verdict["classification"]),
    )
    stage_json(
        index, "fields/result", "result.json", "test2d-field-result", {},
        lambda: {"status": verdict["status"], "classification": verdict["classification"],
                 "output": str(OUTPUT), "output_sha256": sha256_file(OUTPUT)},
    )
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
