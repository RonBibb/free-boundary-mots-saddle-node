#!/usr/bin/env python3
"""Sealed G10 and physical tensor-norm convergence audit for A=7.90."""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import run_corrected_fold_live_nonlinear_gauge_source as live
from bhps.corrected_A790_physical_tensor_convergence import (
    adm_extrinsic_curvature_tensor,
    four_grid_orders,
    interpolate_tensor,
    physical_tensor_difference,
    physical_tensor_l2,
    spatial_metric_tensor,
)
from bhps.regular_so3_gh_reduction import FIELD_ORDER
from run_corrected_A790_independent_dynamic_BVP_detector import (
    analytic_controls as bvp_analytic_controls,
    search_slice,
)
from run_corrected_A790_two_grid_formation_search import (
    endpoint_transfer,
    evolution_pass,
    static_search,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import interpolate_fields


OUTPUT = Path("results/corrected_A790_fourth_grid_physical_tensor_convergence.json")
STATE = Path("results/corrected_A790_fourth_grid_physical_tensor_convergence_state.npz")
PROTOCOL = "notes/89_A790_fourth_grid_physical_tensor_convergence_protocol.md"
BASELINE_RESULT = Path("results/corrected_A790_formation_time_refinement.json")
BASELINE_STATE = Path("results/corrected_A790_formation_time_refinement_state.npz")
G9_RESULT = Path("results/corrected_A790_third_grid_formation_reproduction.json")
G9_STATE = Path("results/corrected_A790_third_grid_formation_reproduction_state.npz")
NOTE83_RESULT = Path("results/corrected_A790_near_axis_d_refinement_audit.json")
BVP_RESULT = Path("results/corrected_A790_independent_dynamic_BVP_detector.json")
AMPLITUDE = 7.90
FINAL_TIME = 0.001
STEPS = 8
TIMES = tuple((index + 1) * FINAL_TIME / STEPS for index in range(STEPS))
INTERVALS = (80, 96, 112, 128)
EXPECTED_SIZES = {
    "G7": (81, 121), "G8": (97, 145),
    "G9": (113, 169), "G10": (129, 193),
}
RADIAL_CUT = 6.0


def relative_norm(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return float(
        np.linalg.norm(left - right)
        / max(np.linalg.norm(left), np.linalg.norm(right), 1e-300)
    )


def analytic_tensor_controls():
    z = np.linspace(0.0, 2.0, 33)
    r = np.linspace(0.0, 1.5, 49)
    q = np.zeros((len(z), len(r), 9))
    q[..., 2] = -1.0
    q[..., 3] = 1.0
    q[..., 6] = 1.0
    d_state = q.copy()
    d_state[..., 4] = 2.5
    d_metric = spatial_metric_tensor(d_state, r)
    d_error = float(max(
        np.max(np.abs(d_metric[..., 1, 1] - (1.0 + 2.5 * r[None, :] ** 2))),
        np.max(np.abs(d_metric[:, 0, 1, 1] - 1.0)),
        np.max(np.abs(d_metric[..., 2, 2] - 1.0)),
    ))

    rate = 0.7
    velocity = np.zeros_like(q)
    velocity[..., 3] = -2.0 * rate
    velocity[..., 6] = -2.0 * rate
    extrinsic = adm_extrinsic_curvature_tensor(q, velocity, z, r)
    k_error = float(np.max(np.abs(
        extrinsic - np.broadcast_to(rate * np.eye(4), extrinsic.shape)
    )))

    flat_metric = spatial_metric_tensor(q, r)
    identity = np.broadcast_to(np.eye(4), flat_metric.shape)
    norm = physical_tensor_l2(identity, flat_metric, z, r)
    analytic_volume = 2.0 * 4.0 * math.pi * 1.5**3 / 3.0
    volume_relative_error = float(abs(norm - np.sqrt(4.0 * analytic_volume)) / norm)

    shape_metric = np.zeros_like(flat_metric)
    shape_metric[..., 0, 0] = 0.3
    shape_metric[..., 1, 1] = r[None, :] ** 2
    shape_metric[..., 2, 2] = shape_metric[..., 3, 3] = 0.2
    shape_k = np.zeros_like(flat_metric)
    shape_k[..., 0, 0] = 0.4
    shape_k[..., 0, 1] = shape_k[..., 1, 0] = 0.1 * r[None, :]
    shape_k[..., 1, 1] = 0.5
    shape_k[..., 2, 2] = shape_k[..., 3, 3] = 0.25

    def manufactured_orders(shape):
        fields = [count**-3 * shape for count in INTERVALS]
        differences = [
            physical_tensor_difference(
                left, right, flat_metric, flat_metric, z, r,
            )["absolute_difference"]
            for left, right in zip(fields[:-1], fields[1:])
        ]
        return {"differences": differences, **four_grid_orders(differences, INTERVALS)}

    metric_orders = manufactured_orders(shape_metric)
    k_orders = manufactured_orders(shape_k)
    passed = bool(
        d_error < 1e-13 and k_error < 2e-12
        and volume_relative_error < 1e-12
        and max(
            abs(metric_orders["coarse_triplet_order"] - 3.0),
            abs(metric_orders["fine_triplet_order"] - 3.0),
            abs(k_orders["coarse_triplet_order"] - 3.0),
            abs(k_orders["fine_triplet_order"] - 3.0),
        ) < 1e-7
    )
    return {
        "passed": passed,
        "r_squared_d_reconstruction_maximum_error": d_error,
        "isotropic_ADM_K_maximum_error": k_error,
        "constant_tensor_volume_norm_relative_error": volume_relative_error,
        "manufactured_spatial_metric_orders": metric_orders,
        "manufactured_extrinsic_curvature_orders": k_orders,
    }


def archive_final_fields(baseline, g9_archive, g10_run):
    return {
        "position_increment": {
            "G7": np.asarray(baseline["G7_8step_increment"]),
            "G8": np.asarray(baseline["G8_8step_increment"]),
            "G9": np.asarray(g9_archive["G9_8step_increment"]),
            "G10": np.asarray(g10_run["_increment"]),
        },
        "velocity": {
            "G7": np.asarray(baseline["G7_8step_velocity"]),
            "G8": np.asarray(baseline["G8_8step_velocity"]),
            "G9": np.asarray(g9_archive["G9_8step_velocity"]),
            "G10": np.asarray(g10_run["_velocity"]),
        },
        "source_increment": {
            "G7": np.asarray(baseline["G7_8step_source_increment"]),
            "G8": np.asarray(baseline["G8_8step_source_increment"]),
            "G9": np.asarray(g9_archive["G9_8step_source_increment"]),
            "G10": np.asarray(g10_run["_source_increment"]),
        },
    }


def common_fields(native, geometries, target_z, target_r):
    result = {}
    for label, values in native.items():
        result[label] = interpolate_fields(
            values, geometries[label]["z"], geometries[label]["r"],
            target_z, target_r,
        )
    return result


def scalar_sequence_diagnostics(vectors):
    labels = ("G7_G8", "G8_G9", "G9_G10")
    grids = ("G7", "G8", "G9", "G10")
    records = {}
    differences = []
    for pair, left, right in zip(labels, grids[:-1], grids[1:]):
        lv = np.asarray(vectors[left])
        rv = np.asarray(vectors[right])
        absolute = float(np.linalg.norm(lv - rv))
        differences.append(absolute)
        records[pair] = {
            "absolute_difference": absolute,
            "relative_difference": float(
                absolute / max(np.linalg.norm(lv), np.linalg.norm(rv), 1e-300)
            ),
        }
    orders = four_grid_orders(differences, INTERVALS)
    return {
        "pairs": records,
        "adjacent_absolute_differences": differences,
        **orders,
        "strictly_decreasing": bool(
            differences[0] > differences[1] > differences[2]
        ),
    }


def radial_volume_sequence(vectors, z, r):
    records = {}
    differences = []
    for pair, left, right in zip(
        ("G7_G8", "G8_G9", "G9_G10"),
        ("G7", "G8", "G9"), ("G8", "G9", "G10"),
    ):
        delta = np.asarray(vectors[left]) - np.asarray(vectors[right])
        radial = np.trapezoid(4.0 * math.pi * r[None, :] ** 2 * delta**2, x=r, axis=1)
        absolute = float(np.sqrt(max(float(np.trapezoid(radial, x=z)), 0.0)))
        differences.append(absolute)
        records[pair] = {"absolute_difference": absolute}
    return {
        "pairs": records, "adjacent_absolute_differences": differences,
        **four_grid_orders(differences, INTERVALS),
        "strictly_decreasing": bool(differences[0] > differences[1] > differences[2]),
        "measure": "4*pi*r^2 dz dr",
    }


def reduced_field_diagnostics(final_fields, geometries, target_z, target_r):
    result = {}
    common_by_family = {
        family: common_fields(values, geometries, target_z, target_r)
        for family, values in final_fields.items()
    }
    for family in ("position_increment", "velocity"):
        common = common_by_family[family]
        result[family] = {
            FIELD_ORDER[index]: scalar_sequence_diagnostics({
                label: values[..., index] for label, values in common.items()
            }) for index in range(9)
        }
        qd = {
            label: values[..., 4] * target_r[None, :] ** 2
            for label, values in common.items()
        }
        result[family]["physical_qd_equals_r_squared_d_radial_volume"] = (
            radial_volume_sequence(qd, target_z, target_r)
        )
    source = common_by_family["source_increment"]
    result["source_increment"] = scalar_sequence_diagnostics(source)
    return result, common_by_family


def physical_tensor_diagnostics(final_fields, geometries, target_z, target_r):
    final_metric = {}
    metric_increment = {}
    extrinsic = {}
    for label, geometry in geometries.items():
        initial = np.asarray(geometry["jet_field"].reduced_fields)
        position = initial + final_fields["position_increment"][label]
        velocity = final_fields["velocity"][label]
        initial_metric = spatial_metric_tensor(initial, geometry["r"])
        current_metric = spatial_metric_tensor(position, geometry["r"])
        current_k = adm_extrinsic_curvature_tensor(
            position, velocity, geometry["z"], geometry["r"],
        )
        final_metric[label] = interpolate_tensor(
            current_metric, geometry["z"], geometry["r"], target_z, target_r,
        )
        metric_increment[label] = interpolate_tensor(
            current_metric - initial_metric, geometry["z"], geometry["r"],
            target_z, target_r,
        )
        extrinsic[label] = interpolate_tensor(
            current_k, geometry["z"], geometry["r"], target_z, target_r,
        )

    def sequence(fields):
        records = {}
        differences = []
        for pair, left, right in zip(
            ("G7_G8", "G8_G9", "G9_G10"),
            ("G7", "G8", "G9"), ("G8", "G9", "G10"),
        ):
            record = physical_tensor_difference(
                fields[left], fields[right], final_metric[left],
                final_metric[right], target_z, target_r,
            )
            records[pair] = record
            differences.append(record["absolute_difference"])
        return {
            "pairs": records,
            "adjacent_absolute_differences": differences,
            **four_grid_orders(differences, INTERVALS),
            "strictly_decreasing": bool(
                differences[0] > differences[1] > differences[2]
            ),
        }

    return {
        "spatial_metric_increment": sequence(metric_increment),
        "ADM_extrinsic_curvature": sequence(extrinsic),
        "full_final_spatial_metric_secondary": sequence(final_metric),
        "coordinate_identification": (
            "componentwise cubic transfer to fixed G7 (z,r) nodes over r<=6; "
            "equal coordinate labels are assumed to identify physical points"
        ),
    }


def archive_states(initial, archive, prefix):
    records = []
    for index in range(STEPS):
        if index < STEPS - 1:
            increment = archive[f"{prefix}_time_{index}_increment"]
            velocity = archive[f"{prefix}_time_{index}_velocity"]
        else:
            increment = archive[f"{prefix}_8step_increment"]
            velocity = archive[f"{prefix}_8step_velocity"]
        records.append({
            "_position": initial + increment,
            "_velocity": np.asarray(velocity),
        })
    return records


def run_states(run):
    return [
        run["_checkpoints"][index + 1] if index < STEPS - 1 else run
        for index in range(STEPS)
    ]


def detector_history(label, geometry, states):
    records = []
    for time_value, state in zip(TIMES, states):
        search = search_slice(
            f"{label}-t{time_value:.6f}", state["_position"],
            state["_velocity"], geometry,
        )
        records.append({
            "time": float(time_value),
            "admitted_distinct_count": search["admitted_distinct_count"],
            "admitted_signatures": search["admitted_signatures"],
            "search": search,
        })
    return records


def detector_history_pass(records):
    counts = [record["admitted_distinct_count"] for record in records]
    if counts != [0, 0, 0, 0, 2, 2, 2, 2]:
        return False
    return bool(all(
        all(len(cluster["members"]) >= 2 for cluster in record["search"]["clusters"])
        for record in records[4:]
    ))


def endpoint_sequence(signatures):
    vectors = {
        label: np.asarray(values, dtype=float).ravel()
        for label, values in signatures.items()
    }
    return scalar_sequence_diagnostics(vectors)


def save_state(g10, case, run):
    values = {
        "G10_z": g10["z"], "G10_r": g10["r"],
        "times": np.asarray(TIMES),
        "G10_8step_increment": run["_increment"],
        "G10_8step_velocity": run["_velocity"],
        "G10_8step_source_increment": run["_source_increment"],
    }
    for index in range(STEPS - 1):
        state = run["_checkpoints"][index + 1]
        values[f"G10_time_{index}_increment"] = state["_position"] - case["initial"]
        values[f"G10_time_{index}_velocity"] = state["_velocity"]
    np.savez_compressed(STATE, **values)


def main():
    required = (
        BASELINE_RESULT, BASELINE_STATE, G9_RESULT, G9_STATE,
        NOTE83_RESULT, BVP_RESULT, Path(PROTOCOL),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing sealed inputs: {missing}")
    overall_started = time.perf_counter()
    controls = analytic_tensor_controls()
    bvp_controls = bvp_analytic_controls()
    baseline_result = json.loads(BASELINE_RESULT.read_text())
    g9_result = json.loads(G9_RESULT.read_text())
    note83_result = json.loads(NOTE83_RESULT.read_text())
    bvp_result = json.loads(BVP_RESULT.read_text())
    baseline = np.load(BASELINE_STATE)
    g9_archive = np.load(G9_STATE)

    print("constructing fresh corrected G7/G8/G9/G10 A=7.90 sequence", flush=True)
    build_started = time.perf_counter()
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": AMPLITUDE}
    geometries = {
        "G7": build_refined(
            seed, 81, 121, "G7A790-G10-parent",
            selector_iterations=40, slice_iterations=270,
        ),
    }
    geometries["G8"] = build_refined(
        geometries["G7"], 97, 145, "G8A790-G10-parent",
        selector_iterations=45, slice_iterations=280,
    )
    geometries["G9"] = build_refined(
        geometries["G8"], 113, 169, "G9A790-G10-parent",
        selector_iterations=50, slice_iterations=300,
    )
    geometries["G10"] = build_refined(
        geometries["G9"], 129, 193, "G10A790-fourth-grid",
        selector_iterations=55, slice_iterations=320,
    )
    build_seconds = time.perf_counter() - build_started

    print("G10 initial static and dynamic BVP searches", flush=True)
    initial_static = static_search(geometries["G10"])
    g10_initial = np.asarray(geometries["G10"]["jet_field"].reduced_fields)
    initial_dynamic = search_slice(
        "G10-A790-t0", g10_initial, np.zeros_like(g10_initial),
        geometries["G10"],
    )

    print("G10 live evolution to t=0.001", flush=True)
    case = live.setup_case(
        geometries["G10"], "G10-A790-fourth-grid",
        live_normal_wall_gauge=True, live_outer_sommerfeld=True,
    )
    live.FINAL_TIME = FINAL_TIME
    live.STEPS = STEPS
    evolution_started = time.perf_counter()
    run = live.integrate(case, checkpoint_steps=range(1, STEPS))
    evolution_seconds = time.perf_counter() - evolution_started
    save_state(geometries["G10"], case, run)

    print("independent BVP histories on archived G9 and fresh G10", flush=True)
    detector_started = time.perf_counter()
    g9_initial = np.asarray(geometries["G9"]["jet_field"].reduced_fields)
    histories = {
        "G9": detector_history(
            "G9-A790-four-grid-control", geometries["G9"],
            archive_states(g9_initial, g9_archive, "G9"),
        ),
        "G10": detector_history(
            "G10-A790-fourth-grid", geometries["G10"], run_states(run),
        ),
    }
    detector_seconds = time.perf_counter() - detector_started

    final_fields = archive_final_fields(baseline, g9_archive, run)
    target_z = np.asarray(geometries["G7"]["z"])
    target_r = np.asarray(geometries["G7"]["r"])
    target_r = target_r[target_r <= RADIAL_CUT + 1e-12]
    tensor_diagnostics = physical_tensor_diagnostics(
        final_fields, geometries, target_z, target_r,
    )
    reduced_diagnostics, common = reduced_field_diagnostics(
        final_fields, geometries, target_z, target_r,
    )

    signatures = {
        "G7": bvp_result["searches"]["G7"]["0.001"]["admitted_signatures"],
        "G8": bvp_result["searches"]["G8"]["0.001"]["admitted_signatures"],
        "G9": histories["G9"][-1]["admitted_signatures"],
        "G10": histories["G10"][-1]["admitted_signatures"],
    }
    endpoints = endpoint_sequence(signatures)
    g9_g10_endpoint = endpoint_transfer(signatures["G9"], signatures["G10"])

    spacing = {
        label: {
            "size": list(EXPECTED_SIZES[label]),
            "z_spacing": float(np.diff(geometry["z"])[0]),
            "r_spacing": float(np.diff(geometry["r"])[0]),
            "selector_residual": float(geometry["selector_maximum"]),
        } for label, geometry in geometries.items()
    }
    construction_pass = bool(
        all(
            (len(geometries[label]["z"]), len(geometries[label]["r"]))
            == EXPECTED_SIZES[label] for label in geometries
        )
        and np.isclose(spacing["G10"]["r_spacing"], 8.0 / 192.0)
        and geometries["G10"]["selector_maximum"] < 1e-9
        and np.all(np.isfinite(g10_initial))
    )
    spatial = tensor_diagnostics["spatial_metric_increment"]
    extrinsic = tensor_diagnostics["ADM_extrinsic_curvature"]

    def order_pass(record):
        return bool(
            record["strictly_decreasing"]
            and record["coarse_triplet_order"] is not None
            and record["fine_triplet_order"] is not None
            and record["coarse_triplet_order"] > 1.0
            and record["fine_triplet_order"] > 1.0
        )

    gate1 = bool(controls["passed"] and bvp_controls["passed"] and construction_pass)
    gate2 = bool(
        initial_static["accepted_count"] == 0
        and initial_dynamic["admitted_distinct_count"] == 0
        and evolution_pass(run)
        and spatial["pairs"]["G9_G10"]["relative_difference"] < 0.05
        and extrinsic["pairs"]["G9_G10"]["relative_difference"] < 0.05
        and reduced_diagnostics["source_increment"]["pairs"]["G9_G10"][
            "relative_difference"
        ] < 0.05
    )
    gate3 = bool(
        all(detector_history_pass(records) for records in histories.values())
        and g9_g10_endpoint is not None
        and g9_g10_endpoint["maximum"] < 0.01
        and baseline_result["trajectory"]["G7"]
        and baseline_result["trajectory"]["G8"]
        and g9_result["count_history"] == [0, 0, 0, 0, 2, 2, 2, 2]
    )
    gate4 = order_pass(spatial)
    gate5 = order_pass(extrinsic)
    acceptance = {
        "manufactured_controls_and_exact_G10_construction": gate1,
        "initial_zero_evolution_and_G9_G10_transfer": gate2,
        "G9_G10_independent_BVP_histories_and_endpoint_transfer": gate3,
        "physical_spatial_metric_increment_four_grid_order_above_1": gate4,
        "physical_ADM_extrinsic_curvature_four_grid_order_above_1": gate5,
    }
    if all(acceptance.values()):
        status = "pass"
        classification = "four_grid_physical_spatial_convergence_resolved"
    elif gate1 and gate2 and gate3 and not gate4 and not gate5 and (
        spatial["adjacent_absolute_differences"][2]
        >= spatial["adjacent_absolute_differences"][1]
        and extrinsic["adjacent_absolute_differences"][2]
        >= extrinsic["adjacent_absolute_differences"][1]
    ):
        status = "fail"
        classification = "fourth_grid_formation_or_physical_convergence_failure"
    elif gate1 and gate2 and not gate3 and evolution_pass(run):
        status = "fail"
        classification = "fourth_grid_formation_or_physical_convergence_failure"
    else:
        status = "review"
        classification = "four_grid_physical_spatial_convergence_mixed"

    payload = {
        "status": status,
        "classification": classification,
        "scope": "sealed A=7.90 fourth-grid and physical spatial tensor-norm convergence audit",
        "protocol": PROTOCOL,
        "amplitude": AMPLITUDE,
        "preserved_prior_statuses": {
            "note79": g9_result["status"], "note83": note83_result["status"],
        },
        "grids": spacing,
        "common_comparison_grid": {
            "z_nodes": len(target_z), "r_nodes": len(target_r),
            "r_max": float(target_r[-1]),
        },
        "time_step": FINAL_TIME / STEPS,
        "times": list(TIMES),
        "analytic_tensor_controls": controls,
        "independent_BVP_analytic_controls": bvp_controls,
        "G10_initial_static_search": initial_static,
        "G10_initial_dynamic_BVP_search": initial_dynamic,
        "G10_evolution_diagnostics": {
            "passes_established_gate": evolution_pass(run),
            "finite": run["all_stages_finite"],
            "Lorentzian": run["signature"]["all_points_one_negative_direction"],
            "global_GH_constraint": run["final_constraint"]["global_relative"],
            "wall_position_residual": run["final_wall"]["maximum"],
            "normal_wall_position_residual": run[
                "final_normal_wall_position_residual"
            ]["maximum"],
            "maximum_live_boundary_residual": max(
                run["maximum_normal_wall_acceleration_residual"],
                run["maximum_outer_acceleration_residual"],
                run["maximum_outer_source_residual"],
                run["final_outer_sommerfeld_position_residual"]["maximum_normalized"],
                run["final_outer_source_sommerfeld_residual"]["maximum_normalized"],
            ),
            "maximum_outer_correction": max(
                run["maximum_outer_metric_correction"],
                run["maximum_outer_scalar_correction"],
                run["maximum_outer_source_correction"],
            ),
        },
        "prior_count_histories": {
            "G7": [item["admitted_distinct_count"] for item in baseline_result["trajectory"]["G7"]],
            "G8": [item["admitted_distinct_count"] for item in baseline_result["trajectory"]["G8"]],
            "G9_note79": g9_result["count_history"],
        },
        "independent_BVP_histories": histories,
        "independent_BVP_count_histories": {
            label: [record["admitted_distinct_count"] for record in records]
            for label, records in histories.items()
        },
        "final_independent_BVP_signatures": signatures,
        "endpoint_sequence": endpoints,
        "G9_G10_endpoint_transfer": g9_g10_endpoint,
        "physical_tensor_diagnostics": tensor_diagnostics,
        "raw_reduced_field_diagnostics": reduced_diagnostics,
        "raw_d_disclosure": {
            "position_increment": reduced_diagnostics["position_increment"][FIELD_ORDER[4]],
            "velocity": reduced_diagnostics["velocity"][FIELD_ORDER[4]],
            "physical_r_squared_d_position": reduced_diagnostics["position_increment"][
                "physical_qd_equals_r_squared_d_radial_volume"
            ],
            "physical_r_squared_d_velocity": reduced_diagnostics["velocity"][
                "physical_qd_equals_r_squared_d_radial_volume"
            ],
            "raw_d_is_not_an_acceptance_gate": True,
        },
        "source_increment_diagnostics": reduced_diagnostics["source_increment"],
        "acceptance": acceptance,
        "state_archive": str(STATE),
        "runtime": {
            "geometry_build_seconds": build_seconds,
            "evolution_seconds": evolution_seconds,
            "detector_seconds": detector_seconds,
            "total_seconds": float(time.perf_counter() - overall_started),
        },
        "coordinate_identification_limits": [
            "equal z,r labels on the shared domain are assumed to identify physical points",
            "componentwise cubic transfer is not a solved diffeomorphism or best-match map",
            "midpoint-metric contraction is tensorial only after that identification is fixed",
            "the baseline generalized-harmonic foliation is held fixed",
        ],
        "limitations": [
            "short-time t=0.001 result at one fixed time step",
            "finite twelve-seed star-shaped donor-capped BVP search",
            "raw regularized d remains separately reported and is not silently discarded",
            "not continuum existence, event horizon, topology change, amplitude/gauge basin, long-time stability, connected bulk geometry, dark-matter halo, or mass transfer",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": status, "classification": classification,
        "independent_BVP_count_histories": payload["independent_BVP_count_histories"],
        "G9_G10_endpoint_transfer": g9_g10_endpoint,
        "endpoint_orders": {
            key: endpoints[key] for key in (
                "coarse_triplet_order", "fine_triplet_order",
            )
        },
        "physical_spatial_metric_increment": spatial,
        "physical_ADM_extrinsic_curvature": extrinsic,
        "raw_d_disclosure": payload["raw_d_disclosure"],
        "acceptance": acceptance, "runtime": payload["runtime"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
