#!/usr/bin/env python3
"""Sealed, resumable Test-10 joint grid/time/domain convergence audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import run_corrected_fold_live_nonlinear_gauge_source as live
from bhps.corrected_A790_R10_builder import build_A790_R10_pair
from bhps.corrected_A790_test10_convergence import (
    domain_initial_dominance,
    proper_endpoint_distances,
    relative_difference,
    temporal_tensor_sequence,
    tensor_fields_on_grid,
    three_grid_sequence,
)
from bhps.dynamical_capped_geometry import capped_surface_geometry
from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice
from bhps.dynamical_capped_horizon_bvp import solve_dynamical_capped_surface_bvp
from bhps.recovery_indexer import (
    RecoveryIndex,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
    validate_npz,
)
from run_corrected_A790_fourth_grid_physical_tensor_convergence import (
    analytic_tensor_controls,
)
from run_corrected_A790_independent_dynamic_BVP_detector import (
    admitted as bvp_admitted,
    analytic_controls as bvp_analytic_controls,
    public_surface as public_bvp_surface,
    search_slice,
)
from run_corrected_A790_R10_domain_robustness import public_diagnostics
from run_corrected_A790_two_grid_formation_search import evolution_pass, static_search
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import (
    interpolate_fields,
    relative_norm,
)


PROTOCOL = Path("notes/93_A790_publication_joint_convergence_protocol.md")
OUTPUT = Path("results/corrected_A790_test10_joint_convergence.json")
STATE_OUTPUT = Path("results/corrected_A790_test10_joint_convergence_state.npz")
RECOVERY_ROOT = Path("results/corrected_A790_test10_joint_convergence_recovery")
MANIFEST = RECOVERY_ROOT / "index.json"
AMPLITUDE = 7.90
R8_FINE_STATE = Path("results/corrected_A790_formation_time_refinement_state.npz")
G9_STATE = Path("results/corrected_A790_third_grid_formation_reproduction_state.npz")
R8_LONG_RESULT = Path("results/corrected_A790_t008_long_evolution.json")
R8_LONG_STATE = Path("results/corrected_A790_t008_long_evolution_state.npz")
R10_SHORT_STATE = Path("results/corrected_A790_R10_domain_robustness_state.npz")
R12_RESULT = Path("results/corrected_A790_R12_domain_sequence.json")
R12_STATE = Path("results/corrected_A790_R12_domain_sequence_state.npz")
G10_RESULT = Path("results/corrected_A790_fourth_grid_physical_tensor_convergence.json")
G10_STATE = Path("results/corrected_A790_fourth_grid_physical_tensor_convergence_state.npz")
SHORT_FINAL = 0.001
LONG_FINAL = 0.008
R_CUT = 6.0
LONG_SURFACE_STEPS = (32, 40, 48, 56, 64)
LONG_SURFACE_TIMES = tuple(step * 0.000125 for step in LONG_SURFACE_STEPS)

FIXED_INPUT_HASHES = {
    "notes/77_A790_formation_time_refinement_protocol.md": "109a0e2022793f21133f67f7515dcc3cc580d54749af2e29d23b03dcf24338f3",
    "notes/79_A790_third_grid_formation_reproduction_protocol.md": "19b48008a5ed26e253ad139ea803e0c28884f42ba78ce7936d9ab8c874c31487",
    "notes/83_A790_near_axis_d_refinement_audit_protocol.md": "091b8bfd4cf45ec0c66c7eedde45d8ad9d63e8e1ea02ff700a39745ec60835f5",
    "notes/85_A790_t008_long_evolution_protocol.md": "157c90a80c4d446a9ff94e78c39b07d6f7287b84e88f5bf407db4943eb7dacae",
    "notes/88_A790_R12_domain_sequence_protocol.md": "6a30701fb76b8c9b801d8e4a9cd9ed91cab4038092f73547d2843934f5d84356",
    "notes/89_A790_fourth_grid_physical_tensor_convergence_protocol.md": "f3edbca20287805f95810ad25b6aa59a814136eea0491ca59721180d49c9f0d3",
    str(R8_FINE_STATE): "845ead50eb5e336b0e9a5ad2357016147cdfc199df5c05d460d54d06ddd1c038",
    str(G9_STATE): "9867e8636847dbf351d9f18d0e6516d2e234b6e2b86109a8f14bab52db85380e",
    str(R8_LONG_STATE): "ad7dba798550f6499dbb966833371e6f1be59477d04f53dd683a96d7fc5c24c8",
    str(R10_SHORT_STATE): "bcd2e4cfaeb584a856da733c6f176a126ba2287b69443b3fa7a68e69503d69c7",
    str(R12_STATE): "994c0d0e061c4da461ba8ba56a96716393656351071033fd087ae830ffcb947a",
    str(G10_STATE): "fa9a9c2833d02132f096b3c7f43d7457edd6ddf375ffb534b7a44aa83b0a095b",
    str(R8_LONG_RESULT): "4a754a65b1b59695ba3967a99a342c1cb3a1da97f906342e86ff3a5a001f499a",
    str(R12_RESULT): "8dd9119315d5b62113dfc5c541884d1bc696baba73bf9ee7bbbaa6ae1e4389a1",
    str(G10_RESULT): "03d9f7ff4496ad430ecf368452dfe8ce911fda94a5d594269abec57ebd177b98",
    "src/bhps/corrected_A790_physical_tensor_convergence.py": "d7dd8f88bcfc73f9b7efb8bbdbc0caa4722e2a6583c6ef52733d251edb49dcec",
    "src/bhps/corrected_A790_R10_builder.py": "2cd02b1da4b76c0b072a0b1a73a76bf2b87a9d8e87608978cd8edec43c7e8076",
    "src/bhps/recovery_indexer.py": "1460478fba42433bd340a2ef9e09c0946882a35d3eb63c2c95ea9b055bb549fa",
}

EVOLUTION_SPECS = {
    "G10_coarse": {"grid": "G10", "steps": 4, "dt": 0.00025, "segment": 2},
    "G10_half": {"grid": "G10", "steps": 16, "dt": 0.0000625, "segment": 4},
    "G11_standard": {"grid": "G11", "steps": 8, "dt": 0.000125, "segment": 4},
    "R10G7_long": {"grid": "R10G7", "steps": 64, "dt": 0.000125, "segment": 8},
    "R10G8_long": {"grid": "R10G8", "steps": 64, "dt": 0.000125, "segment": 8},
}


def recovery_inputs():
    dynamic = (
        Path(__file__), Path("src/bhps/corrected_A790_test10_convergence.py"),
        Path("run_corrected_fold_live_nonlinear_gauge_source.py"),
        Path("run_corrected_A790_independent_dynamic_BVP_detector.py"),
    )
    return {**FIXED_INPUT_HASHES, **{str(path): sha256_file(path) for path in dynamic}}


def geometry_fingerprint(case):
    digest = hashlib.sha256()
    for value in (
        case["z"], case["r"], case["initial"], case["source0"], case["memory0"],
    ):
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def stage_json(index, stage_id, path, kind, metadata, producer, expected=900.0):
    index.register(stage_id, kind, expected, metadata)
    cached = index.validated_path(stage_id)
    if cached is not None:
        payload = json.loads(cached.read_text())
        if payload.get("protocol_sha256") != index.protocol_sha256:
            raise RuntimeError(f"cached protocol mismatch in {stage_id}")
        return payload, True
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        payload = {
            "stage_id": stage_id, "protocol_sha256": index.protocol_sha256,
            **producer(),
        }
        atomic_write_json(path, payload)
        checked = json.loads(path.read_text())
        if checked.get("stage_id") != stage_id:
            raise RuntimeError(f"stage validation failed for {stage_id}")
        index.mark_complete(stage_id, path, time.perf_counter() - started)
        return payload, False
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def build_all_geometries():
    print("constructing Test-10 R8 G7--G11 and matched R10 G7/G8 geometries", flush=True)
    fold = build_geometry("G6")
    parent = {**fold, "fold_amplitude": AMPLITUDE}
    r8 = {}
    for label, nz, nr, selector, slices in (
        ("G7", 81, 121, 40, 270), ("G8", 97, 145, 45, 280),
        ("G9", 113, 169, 50, 300), ("G10", 129, 193, 55, 320),
        ("G11", 145, 217, 60, 340),
    ):
        r8[label] = build_refined(
            parent, nz, nr, f"{label}A790-test10", selector_iterations=selector,
            slice_iterations=slices,
        )
        parent = r8[label]
    g7r10, g8r10 = build_A790_R10_pair()
    return {
        **r8, "R10G7": g7r10, "R10G8": g8r10,
    }


def make_cases(geometries):
    return {
        label: live.setup_case(
            geometry, f"{label}-A790-test10", live_normal_wall_gauge=True,
            live_outer_sommerfeld=True,
        ) for label, geometry in geometries.items()
        if label in {spec["grid"] for spec in EVOLUTION_SPECS.values()}
    }


def empty_segment_diagnostics():
    return {
        "all_stages_finite": True,
        "maximum_stage_acceleration_relative_change": 0.0,
        "maximum_normal_wall_acceleration_residual": 0.0,
        "maximum_outer_acceleration_residual": 0.0,
        "maximum_outer_source_residual": 0.0,
        "maximum_outer_metric_correction": 0.0,
        "maximum_outer_scalar_correction": 0.0,
        "maximum_outer_source_correction": 0.0,
    }


def update_segment_diagnostics(summary, k1, k2, diagnostics):
    summary["maximum_stage_acceleration_relative_change"] = max(
        summary["maximum_stage_acceleration_relative_change"], relative_norm(k1[1], k2[1]),
    )
    for diagnostic in diagnostics:
        summary["all_stages_finite"] = bool(
            summary["all_stages_finite"] and diagnostic["finite"]
        )
        normal = diagnostic["normal_wall_gauge"]
        if normal is not None:
            summary["maximum_normal_wall_acceleration_residual"] = max(
                summary["maximum_normal_wall_acceleration_residual"],
                normal["final_residual"]["maximum"],
            )
        outer = diagnostic["outer_sommerfeld"]
        if outer is not None:
            summary["maximum_outer_acceleration_residual"] = max(
                summary["maximum_outer_acceleration_residual"],
                outer["maximum_normalized_acceleration_residual"],
            )
            summary["maximum_outer_metric_correction"] = max(
                summary["maximum_outer_metric_correction"], outer["metric_relative_correction"],
            )
            summary["maximum_outer_scalar_correction"] = max(
                summary["maximum_outer_scalar_correction"], outer["scalar_relative_correction"],
            )
        outer_source = diagnostic["outer_source_sommerfeld"]
        if outer_source is not None:
            summary["maximum_outer_source_residual"] = max(
                summary["maximum_outer_source_residual"], outer_source["maximum_normalized"],
            )
            summary["maximum_outer_source_correction"] = max(
                summary["maximum_outer_source_correction"], outer_source["relative_correction"],
            )


def integrate_segment(case, state, start_step, end_step, total_steps, dt):
    current = tuple(np.asarray(value).copy() for value in state)
    diagnostic = empty_segment_diagnostics()
    snapshots = {}
    for step in range(start_step + 1, end_step + 1):
        current_time = (step - 1) * dt
        print(f"{case['label']}: restartable step {step}/{total_steps}, stage 1", flush=True)
        k1, d1 = live.driver_stage(case, current_time, *current)
        midpoint = tuple(value + 0.5 * dt * slope for value, slope in zip(current, k1))
        print(f"{case['label']}: restartable step {step}/{total_steps}, stage 2", flush=True)
        k2, d2 = live.driver_stage(case, current_time + 0.5 * dt, *midpoint)
        update_segment_diagnostics(diagnostic, k1, k2, (d1, d2))
        current = tuple(value + dt * slope for value, slope in zip(current, k2))
        snapshots[f"step_{step:03d}_increment"] = current[0] - case["initial"]
        snapshots[f"step_{step:03d}_velocity"] = current[1].copy()
    return current, snapshots, diagnostic


def diagnostic_arrays(record):
    return {f"diag_{key}": np.asarray(value) for key, value in record.items()}


def diagnostic_from_archive(archive):
    return {
        key.removeprefix("diag_"): archive[key].item()
        for key in archive.files if key.startswith("diag_")
    }


def segment_path(run_label, start, end):
    return RECOVERY_ROOT / f"evolution_{run_label}_steps_{start + 1:03d}_{end:03d}.npz"


def validate_segment(path, case, start, end):
    shape = tuple(case["initial"].shape)
    source_shape = tuple(case["source0"].shape)
    required = {
        "start_step": (), "end_step": (), "end_position": shape,
        "end_velocity": shape, "end_source": source_shape,
        "end_memory": source_shape,
    }
    for step in range(start + 1, end + 1):
        required[f"step_{step:03d}_increment"] = shape
        required[f"step_{step:03d}_velocity"] = shape
    validate_npz(path, required)
    with np.load(path) as archive:
        if int(archive["start_step"]) != start or int(archive["end_step"]) != end:
            raise RuntimeError("segment index mismatch")


def run_evolution(index, run_label, case):
    spec = EVOLUTION_SPECS[run_label]
    steps, dt, segment_size = spec["steps"], spec["dt"], spec["segment"]
    state = (
        case["initial"].copy(), np.zeros_like(case["initial"]),
        case["source0"].copy(), case["memory0"].copy(),
    )
    parent_sha = geometry_fingerprint(case)
    paths = []
    diagnostics = []
    for start in range(0, steps, segment_size):
        end = min(start + segment_size, steps)
        stage_id = f"evolution/{run_label}/steps_{start + 1:03d}_{end:03d}"
        path = segment_path(run_label, start, end)
        metadata = {
            "run": run_label, "grid": spec["grid"], "start_step": start,
            "end_step": end, "dt": dt, "parent_sha256": parent_sha,
            "geometry_fingerprint": geometry_fingerprint(case),
        }
        index.register(stage_id, "evolution-segment", 2400.0, metadata)
        cached = index.validated_path(stage_id)
        if cached is None:
            index.mark_running(stage_id)
            started = time.perf_counter()
            try:
                state, snapshots, diagnostic = integrate_segment(
                    case, state, start, end, steps, dt,
                )
                if not diagnostic["all_stages_finite"]:
                    raise RuntimeError("nonfinite evolution stage")
                if max(
                    diagnostic["maximum_outer_metric_correction"],
                    diagnostic["maximum_outer_scalar_correction"],
                    diagnostic["maximum_outer_source_correction"],
                ) > 0.20:
                    raise RuntimeError("sealed 20-percent adverse outer-correction stop")
                if not live.signature_summary(state[0], case["r"])[
                    "all_points_one_negative_direction"
                ]:
                    raise RuntimeError("lost Lorentzian signature at segment end")
                atomic_write_npz(
                    path, start_step=np.asarray(start), end_step=np.asarray(end),
                    end_position=state[0], end_velocity=state[1],
                    end_source=state[2], end_memory=state[3], **snapshots,
                    **diagnostic_arrays(diagnostic),
                )
                validate_segment(path, case, start, end)
                index.mark_complete(stage_id, path, time.perf_counter() - started)
                cached = path
            except Exception as error:
                index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
                raise
        else:
            validate_segment(cached, case, start, end)
        with np.load(cached) as archive:
            state = tuple(np.asarray(archive[key]) for key in (
                "end_position", "end_velocity", "end_source", "end_memory",
            ))
            diagnostics.append(diagnostic_from_archive(archive))
        paths.append(cached)
        parent_sha = sha256_file(cached)
    return state, diagnostics, paths


def combine_segment_diagnostics(records):
    combined = empty_segment_diagnostics()
    for record in records:
        combined["all_stages_finite"] = bool(
            combined["all_stages_finite"] and record["all_stages_finite"]
        )
        for key in combined:
            if key != "all_stages_finite":
                combined[key] = max(float(combined[key]), float(record[key]))
    return combined


def finalize_run(case, state, diagnostics, steps, dt):
    position, velocity, source, memory = state
    final_time = steps * dt
    source_z, source_r = live.regular_source_spatial_derivatives(
        source, case["z"], case["r"],
    )
    target = live.regular_so3_nonlinear_anchored_damped_wave_target(
        position, case["initial"], case["source0"], case["r"],
        live.TARGET_MU_LAPSE, live.TARGET_MU_SHIFT, live.TARGET_POWER,
    )
    advection = live.regular_so3_live_source_shift_advection(
        position, case["r"], source, source_z, source_r,
    )
    source_dot, _ = live.source_driver_rhs(
        source, memory, target, live.DRIVER_MU, live.DRIVER_ETA, advection,
    )
    final_outer_source = None
    if case["rhs"].live_outer_sommerfeld:
        source_dot, final_outer_source = live.apply_outer_source_sommerfeld(
            source, source_dot, case["source0"], case["source_time0"],
            case["_initial_source_second_time"], position, final_time,
            case["r"], case["rhs"].stencil_width,
        )
    gauge = live.StageRegularGaugeSource(source, source_dot, case["z"], case["r"])
    print(f"{case['label']}: exact resumed-run final diagnostics", flush=True)
    return {
        "final_time": final_time, "steps": steps, "time_step": dt,
        **combine_segment_diagnostics(diagnostics),
        "final_constraint": live.gauge_constraint_summary(
            position, velocity, final_time, case["rhs"], R_CUT, gauge,
        ),
        "final_wall": live.compact_wall_position_residuals(
            position, case["z"], case["r"], case["geometry"]["background"],
        ),
        "signature": live.signature_summary(position, case["r"]),
        "final_normal_wall_position_residual":
            live.compact_wall_normal_gauge_position_residuals(
                position, source, case["z"], case["r"],
                case["geometry"]["background"],
            ),
        "final_outer_sommerfeld_position_residual":
            live.outer_sommerfeld_position_residuals(
                position, velocity, case["rhs"].outer_reference_position,
                case["rhs"].outer_reference_acceleration, final_time,
                case["r"], case["rhs"].stencil_width,
            ),
        "final_outer_source_sommerfeld_residual": final_outer_source,
        "_position": position, "_increment": position - case["initial"],
        "_velocity": velocity, "_source": source,
        "_source_increment": source - case["source0"], "_memory": memory,
    }


def load_step(run_label, step, case):
    spec = EVOLUTION_SPECS[run_label]
    size = spec["segment"]
    start = ((int(step) - 1) // size) * size
    end = min(start + size, spec["steps"])
    with np.load(segment_path(run_label, start, end)) as archive:
        return {
            "position": case["initial"] + archive[f"step_{step:03d}_increment"],
            "velocity": np.asarray(archive[f"step_{step:03d}_velocity"]),
        }


def load_segment_end(run_label, step, case):
    spec = EVOLUTION_SPECS[run_label]
    size = spec["segment"]
    start = int(step) - size
    path = segment_path(run_label, start, int(step))
    with np.load(path) as archive:
        return {
            "position": np.asarray(archive["end_position"]),
            "velocity": np.asarray(archive["end_velocity"]),
            "source": np.asarray(archive["end_source"]),
            "memory": np.asarray(archive["end_memory"]),
        }


def initial_stage(index, label, geometry):
    stage_id = f"initial/{label}"
    path = RECOVERY_ROOT / f"initial_{label}.json"
    position = np.asarray(geometry["jet_field"].reduced_fields)
    payload, _ = stage_json(
        index, stage_id, path, "initial-search", {"grid": label, "time": 0.0},
        lambda: {
            "static": static_search(geometry),
            "BVP": search_slice(
                f"{label}-A790-test10-t0", position, np.zeros_like(position), geometry,
            ),
            "selector_residual": float(geometry["selector_maximum"]),
            "grid_size": [len(geometry["z"]), len(geometry["r"])],
        },
    )
    if (
        payload["static"]["accepted_count"] != 0
        or payload["BVP"]["admitted_distinct_count"] != 0
        or payload["selector_residual"] >= 1e-9
    ):
        raise RuntimeError(f"sealed initial stop condition on {label}")
    return payload


def bvp_stage(index, run_label, step, state, geometry):
    stage_id = f"detector/{run_label}/step_{step:03d}"
    path = RECOVERY_ROOT / f"bvp_{run_label}_step_{step:03d}.json"
    spec = EVOLUTION_SPECS[run_label]
    payload, _ = stage_json(
        index, stage_id, path, "independent-BVP",
        {"run": run_label, "step": int(step), "time": step * spec["dt"]},
        lambda: {"search": search_slice(
            f"{run_label}-t{step * spec['dt']:.6f}", state["position"],
            state["velocity"], geometry,
        )},
        expected=900.0,
    )
    return payload["search"]


def representative_geometry(search, state, geometry):
    prepared = prepare_capped_expansion_slice(
        state["position"], state["velocity"], geometry["z"], geometry["r"],
    )
    records = []
    for name, cluster in zip(
        ("inner", "outer"), sorted(search["clusters"], key=lambda item: item["signature"][1]),
    ):
        members = sorted(cluster["members"], key=lambda item: item["seed"])
        seed = float(members[len(members) // 2]["seed"])
        started = time.perf_counter()
        surface = solve_dynamical_capped_surface_bvp(
            state["position"], state["velocity"], geometry["z"], geometry["r"],
            seed, tolerance=2e-5, nodes=121, maximum_nodes=6000, dense_nodes=501,
        )
        surface["runtime_seconds"] = float(time.perf_counter() - started)
        geometric = capped_surface_geometry(
            state["position"], state["velocity"], geometry["z"], geometry["r"],
            surface, prepared=prepared,
        )
        records.append({
            "branch": name, "seed": seed, "admitted": bvp_admitted(surface),
            "surface": public_bvp_surface(surface), "geometry": geometric,
            "proper_endpoints": proper_endpoint_distances(
                state["position"], geometry["z"], geometry["r"],
                surface["rho_axis"], surface["rho_brane"],
            ),
        })
    return records


def geometry_stage(index, run_label, step, search, state, geometry):
    stage_id = f"geometry/{run_label}/step_{step:03d}"
    path = RECOVERY_ROOT / f"geometry_{run_label}_step_{step:03d}.json"
    payload, _ = stage_json(
        index, stage_id, path, "proper-branch-geometry",
        {"run": run_label, "step": int(step), "time": step * EVOLUTION_SPECS[run_label]["dt"]},
        lambda: {"branches": representative_geometry(search, state, geometry)},
        expected=600.0,
    )
    return payload["branches"]


def valid_transition(counts):
    return bool(
        all(value in (0, 2) for value in counts)
        and 2 in counts
        and all(left <= right for left, right in zip(counts, counts[1:]))
    )


def detector_records(index, run_label, geometry, case, steps):
    result = {}
    for step in steps:
        state = load_step(run_label, step, case)
        result[step] = bvp_stage(index, run_label, step, state, geometry)
    return result


def surface_transfer(left, right):
    values = []
    records = []
    for a, b in zip(left, right):
        record = {"branch": a["branch"]}
        for key, av, bv in (
            ("axis", a["geometry"]["rho_axis"], b["geometry"]["rho_axis"]),
            ("brane", a["geometry"]["rho_brane"], b["geometry"]["rho_brane"]),
            ("area", a["geometry"]["one_sided_cap_area"], b["geometry"]["one_sided_cap_area"]),
            ("equivalent_radius", a["geometry"]["equivalent_area_radius"], b["geometry"]["equivalent_area_radius"]),
            ("proper_axis", a["proper_endpoints"]["compact_axis_endpoint_to_brane"], b["proper_endpoints"]["compact_axis_endpoint_to_brane"]),
            ("proper_brane", a["proper_endpoints"]["radial_axis_to_brane_endpoint"], b["proper_endpoints"]["radial_axis_to_brane_endpoint"]),
        ):
            record[key] = relative_difference(av, bv)
            values.append(record[key])
        records.append(record)
    return {"branches": records, "maximum": max(values) if values else None}


def short_standard_state(label, geometry, archive):
    return {
        "position": np.asarray(geometry["jet_field"].reduced_fields) + archive[f"{label}_8step_increment"],
        "velocity": np.asarray(archive[f"{label}_8step_velocity"]),
        "source_increment": np.asarray(archive[f"{label}_8step_source_increment"]),
    }


def spatial_time_analysis(geometries, cases, runs, short_searches):
    g9_archive = np.load(G9_STATE)
    g10_archive = np.load(G10_STATE)
    g10_result = json.loads(G10_RESULT.read_text())
    states = {
        "G9": short_standard_state("G9", geometries["G9"], g9_archive),
        "G10": short_standard_state("G10", geometries["G10"], g10_archive),
        "G11": {
            "position": runs["G11_standard"]["_position"],
            "velocity": runs["G11_standard"]["_velocity"],
            "source_increment": runs["G11_standard"]["_source_increment"],
        },
    }
    target_z = geometries["G9"]["z"]
    target_r = geometries["G9"]["r"]
    target_r = target_r[target_r <= R_CUT + 1e-12]
    bundles = {
        label: tensor_fields_on_grid(
            state["position"], state["velocity"], geometries[label], target_z, target_r,
        ) for label, state in states.items()
    }
    spatial = {
        family: three_grid_sequence(
            {label: bundles[label][family] for label in states},
            {label: bundles[label]["final_metric"] for label in states},
            target_z, target_r, (112, 128, 144),
        ) for family in ("metric_increment", "ADM_K")
    }
    common_source = {
        label: interpolate_fields(
            states[label]["source_increment"], geometries[label]["z"], geometries[label]["r"],
            target_z, target_r,
        ) for label in states
    }
    source_transfer = relative_norm(common_source["G10"], common_source["G11"])

    temporal_states = {
        "coarse": {
            "position": runs["G10_coarse"]["_position"],
            "velocity": runs["G10_coarse"]["_velocity"],
        },
        "medium": states["G10"],
        "fine": {
            "position": runs["G10_half"]["_position"],
            "velocity": runs["G10_half"]["_velocity"],
        },
    }
    temporal_bundles = {
        label: tensor_fields_on_grid(
            state["position"], state["velocity"], geometries["G10"], target_z, target_r,
        ) for label, state in temporal_states.items()
    }
    temporal = {}
    for family in ("metric_increment", "ADM_K"):
        temporal[family] = temporal_tensor_sequence(
            temporal_bundles["coarse"][family], temporal_bundles["medium"][family],
            temporal_bundles["fine"][family],
            {label: temporal_bundles[label]["final_metric"] for label in temporal_bundles},
            target_z, target_r,
        )
        temporal[family]["temporal_to_G10_G11_spatial_ratio"] = float(
            temporal[family]["medium_fine"]["absolute_difference"]
            / spatial[family]["pairs"]["G10_G11"]["absolute_difference"]
        )

    standard_search = g10_result["independent_BVP_histories"]["G10"][-1]["search"]
    final_searches = {
        "coarse": short_searches["G10_coarse"][4],
        "medium": standard_search,
        "fine": short_searches["G10_half"][16],
        "G11": short_searches["G11_standard"][8],
    }
    final_states = {
        "coarse": temporal_states["coarse"], "medium": temporal_states["medium"],
        "fine": temporal_states["fine"], "G11": states["G11"],
    }
    final_geometry = {
        label: representative_geometry(
            final_searches[label], final_states[label],
            geometries["G11"] if label == "G11" else geometries["G10"],
        ) for label in final_searches
    }
    surface_transfer_records = {
        "G10_standard_G11_standard": surface_transfer(final_geometry["medium"], final_geometry["G11"]),
        "G10_standard_G10_half": surface_transfer(final_geometry["medium"], final_geometry["fine"]),
    }
    histories = {
        label: [records[step]["admitted_distinct_count"] for step in sorted(records)]
        for label, records in short_searches.items()
    }
    half_counts = histories["G10_half"]
    first_half = half_counts.index(2)
    half_times = np.arange(1, 17) * 0.0000625
    half_bracket = [float(half_times[first_half - 1] if first_half else 0.0), float(half_times[first_half])]
    overlap = bool(max(half_bracket[0], 0.0005) < min(half_bracket[1], 0.000625) + 1e-15)
    gates = {
        "G11_formation": bool(
            histories["G11_standard"] == [0, 0, 0, 0, 2, 2, 2, 2]
        ),
        "G10_time_detector": bool(
            histories["G10_coarse"][-1] == 2 and valid_transition(half_counts) and overlap
        ),
        "surface_geometry": bool(
            surface_transfer_records["G10_standard_G11_standard"]["maximum"] < 0.01
            and surface_transfer_records["G10_standard_G10_half"]["maximum"] < 0.002
        ),
        "spatial_tensors": bool(all(
            spatial[family]["strictly_decreasing"]
            and spatial[family]["order"] is not None and spatial[family]["order"] > 1.0
            for family in spatial
        ) and source_transfer < 0.05),
        "temporal_tensors": bool(all(
            temporal[family]["strictly_decreasing"]
            and temporal[family]["order"] is not None and temporal[family]["order"] > 1.5
            and temporal[family]["medium_fine"]["relative_difference"] < 0.002
            and temporal[family]["temporal_to_G10_G11_spatial_ratio"] < 0.5
            for family in temporal
        )),
    }
    return {
        "histories": histories, "G10_half_formation_bracket": half_bracket,
        "G10_half_overlaps_standard_bracket": overlap,
        "spatial_physical_tensors": spatial, "temporal_physical_tensors": temporal,
        "G10_G11_source_transfer": source_transfer,
        "representative_geometry": final_geometry,
        "surface_transfers": surface_transfer_records, "gates": gates,
    }


def r8_long_state(archive, label, step):
    return {
        "position": np.asarray(archive[f"{label}_position_history"])[step],
        "velocity": np.asarray(archive[f"{label}_velocity_history"])[step],
        "source_increment": np.asarray(archive[f"{label}_source_diagnostic_history"])[step // 8 - 1],
    }


def existing_geometry_record(record, state, geometry):
    result = []
    for branch_name, representative in zip(("inner", "outer"), record["representatives"]):
        geometric = representative["geometry"]
        result.append({
            "branch": branch_name, "seed": representative["seed"],
            "admitted": representative["admitted"], "surface": representative["surface"],
            "geometry": geometric,
            "proper_endpoints": proper_endpoint_distances(
                state["position"], geometry["z"], geometry["r"],
                geometric["rho_axis"], geometric["rho_brane"],
            ),
        })
    return result


def tensor_pair(left_state, right_state, left_geometry, right_geometry, target_z, target_r):
    left = tensor_fields_on_grid(
        left_state["position"], left_state["velocity"], left_geometry, target_z, target_r,
    )
    right = tensor_fields_on_grid(
        right_state["position"], right_state["velocity"], right_geometry, target_z, target_r,
    )
    result = {}
    for family in ("initial_metric", "metric_increment", "final_metric", "ADM_K"):
        result[family] = live_tensor_difference(
            left[family], right[family], left["final_metric"], right["final_metric"],
            target_z, target_r,
        )
    result["initial_dominance"] = domain_initial_dominance(
        result["initial_metric"]["absolute_difference"],
        result["metric_increment"]["absolute_difference"],
        result["final_metric"]["absolute_difference"],
    )
    return result


def live_tensor_difference(left, right, metric_left, metric_right, z, r):
    from bhps.corrected_A790_physical_tensor_convergence import physical_tensor_difference
    return physical_tensor_difference(left, right, metric_left, metric_right, z, r)


def long_domain_analysis(geometries, cases, runs, long_searches, long_geometry):
    r8_result = json.loads(R8_LONG_RESULT.read_text())
    r12_result = json.loads(R12_RESULT.read_text())
    r8_archive = np.load(R8_LONG_STATE)
    r12_archive = np.load(R12_STATE)
    transfers = {label: {} for label in ("G7", "G8")}
    r8_geometry_records = {label: {} for label in ("G7", "G8")}
    for label in ("G7", "G8"):
        for index, (step, time_value) in enumerate(zip(LONG_SURFACE_STEPS, LONG_SURFACE_TIMES)):
            key = f"{time_value:.3f}"
            state8 = r8_long_state(r8_archive, label, step)
            record8 = existing_geometry_record(
                r8_result["surface_history"][label][index], state8, geometries[label],
            )
            r8_geometry_records[label][key] = record8
            transfers[label][key] = surface_transfer(
                record8, long_geometry[f"R10{label}_long"][step],
            )

    long_tensors = {label: {} for label in ("G7", "G8")}
    for label in ("G7", "G8"):
        target_z = geometries[label]["z"]
        target_r = np.asarray(geometries[label]["r"])
        target_r = target_r[target_r <= R_CUT + 1e-12]
        state8 = r8_long_state(r8_archive, label, 64)
        state10 = load_segment_end(f"R10{label}_long", 64, cases[f"R10{label}"])
        long_tensors[label] = tensor_pair(
            state8, state10, geometries[label], geometries[f"R10{label}"],
            target_z, target_r,
        )

    domain_states = {}
    domain_slice_states = {}
    for label in ("G7", "G8"):
        target_z = geometries[label]["z"]
        target_r = np.asarray(geometries[label]["r"])
        target_r = target_r[target_r <= R_CUT + 1e-12]
        state8 = r8_long_state(r8_archive, label, 32)
        state10 = load_segment_end(f"R10{label}_long", 32, cases[f"R10{label}"])
        r12_geometry = rebuild_r12_geometry(label)
        initial12 = np.asarray(r12_geometry["jet_field"].reduced_fields)
        state12 = {
            "position": initial12 + r12_archive[f"{label}_long_2_increment"],
            "velocity": np.asarray(r12_archive[f"{label}_long_2_velocity"]),
        }
        domain_slice_states[label] = {
            "R8": (state8, geometries[label]),
            "R10": (state10, geometries[f"R10{label}"]),
            "R12": (state12, r12_geometry),
        }
        domain_states[label] = {
            "R8_R10": tensor_pair(
                state8, state10, geometries[label], geometries[f"R10{label}"],
                target_z, target_r,
            ),
            "R10_R12": tensor_pair(
                state10, state12, geometries[f"R10{label}"], r12_geometry,
                target_z, target_r,
            ),
        }
        domain_states[label]["R12_geometry_size"] = r12_geometry["source_grid"]

    domain_surface_ratios = []
    domain_surface_records = []
    for label in ("G7", "G8"):
        values = r12_result["all_domain_representative_surface_geometry"]
        entries = {domain: values[domain][label]["0.004"] for domain in ("R8", "R10", "R12")}
        for branch_index, branch in enumerate(("inner", "outer")):
            branch_values = {}
            for domain in ("R8", "R10", "R12"):
                geometry_record = entries[domain][branch_index]["geometry"]
                state, geometry = domain_slice_states[label][domain]
                proper = proper_endpoint_distances(
                    state["position"], geometry["z"], geometry["r"],
                    geometry_record["rho_axis"], geometry_record["rho_brane"],
                )
                branch_values[domain] = {
                    "area": geometry_record["one_sided_cap_area"],
                    "equivalent_radius": geometry_record["equivalent_area_radius"],
                    "proper_axis": proper["compact_axis_endpoint_to_brane"],
                    "proper_brane": proper["radial_axis_to_brane_endpoint"],
                }
            for observable in (
                "area", "equivalent_radius", "proper_axis", "proper_brane",
            ):
                x8 = branch_values["R8"][observable]
                x10 = branch_values["R10"][observable]
                x12 = branch_values["R12"][observable]
                ratio = abs(x12 - x10) / max(abs(x10 - x8), 1e-300)
                domain_surface_ratios.append(float(ratio))
                domain_surface_records.append({
                    "grid": label, "branch": branch, "observable": observable,
                    "R8_R10": abs(x10 - x8), "R10_R12": abs(x12 - x10),
                    "shrink_ratio": float(ratio),
                })

    causal = {}
    for label in ("G7", "G8"):
        geometry = geometries[f"R10{label}"]
        speed = float(np.max(geometry["principal"]["r_coordinate_speed"]))
        lower = float((10.0 - R_CUT) / speed)
        causal[label] = {
            "maximum_initial_radial_coordinate_speed": speed,
            "one_way_boundary_to_r6_lower_bound": lower,
            "final_time_fraction": LONG_FINAL / lower,
            "passes": bool(LONG_FINAL < lower),
        }

    max_surface = max(
        record["maximum"] for grid in transfers.values() for record in grid.values()
    )
    long_tensor_pass = bool(all(
        max(
            long_tensors[label][family]["relative_difference"]
            for family in ("metric_increment", "ADM_K", "final_metric")
        ) < 0.02 for label in ("G7", "G8")
    ))
    domain_tensor_pass = bool(all(
        domain_states[label]["R10_R12"][family]["absolute_difference"]
        < domain_states[label]["R8_R10"][family]["absolute_difference"]
        and domain_states[label]["R10_R12"][family]["relative_difference"] < 0.02
        for label in ("G7", "G8") for family in ("metric_increment", "ADM_K")
    ))
    dominance_pass = bool(all(
        domain_states[label][pair]["initial_dominance"]["initial_data_dominated"]
        for label in ("G7", "G8") for pair in ("R8_R10", "R10_R12")
    ))
    long_count_pass = bool(all(
        [long_searches[f"R10{label}_long"][step]["admitted_distinct_count"] for step in LONG_SURFACE_STEPS]
        == [2, 2, 2, 2, 2]
        for label in ("G7", "G8")
    ))
    long_geometry_pass = bool(
        max_surface < 0.01
        and all(
            branch["admitted"] and branch["geometry"]["finite"]
            for records in long_geometry.values() for branches in records.values()
            for branch in branches
        )
        and all(
            branches[1]["geometry"]["one_sided_cap_area"]
            > branches[0]["geometry"]["one_sided_cap_area"]
            for records in long_geometry.values() for branches in records.values()
        )
    )
    outer_pass = bool(all(
        max(
            runs[f"R10{label}_long"]["maximum_outer_metric_correction"],
            runs[f"R10{label}_long"]["maximum_outer_scalar_correction"],
            runs[f"R10{label}_long"]["maximum_outer_source_correction"],
        ) < 0.05 for label in ("G7", "G8")
    ))
    return {
        "R8_geometry": r8_geometry_records, "R10_geometry": long_geometry,
        "R8_R10_surface_transfers": transfers, "maximum_surface_transfer": max_surface,
        "R8_R10_t008_physical_tensors": long_tensors,
        "R8_R10_R12_t004_tensor_decomposition": domain_states,
        "R8_R10_R12_intrinsic_surface_shrink": {
            "records": domain_surface_records,
            "median_shrink_ratio": float(np.median(domain_surface_ratios)),
        },
        "causal_timing": causal,
        "gates": {
            "long_counts": long_count_pass, "long_geometry": long_geometry_pass,
            "long_tensors": long_tensor_pass,
            "three_domain_tensors": domain_tensor_pass,
            "initial_data_dominance": dominance_pass,
            "intrinsic_surface_shrink": bool(np.median(domain_surface_ratios) < 1.0),
            "outer_corrections": outer_pass,
            "causal": bool(all(item["passes"] for item in causal.values())),
        },
    }


_R12_CACHE = None


def rebuild_r12_geometry(label):
    global _R12_CACHE
    if _R12_CACHE is None:
        from bhps.corrected_A790_R12_builder import build_A790_R12_pair
        g7, g8 = build_A790_R12_pair()
        _R12_CACHE = {"G7": g7, "G8": g8}
    return _R12_CACHE[label]


def controls_stage(index):
    def producer():
        tensor = analytic_tensor_controls()
        bvp = bvp_analytic_controls()
        z = np.linspace(1.0, 2.0, 33)
        r = np.linspace(0.0, 2.0, 49)
        q = np.zeros((len(z), len(r), 9))
        q[..., 2] = -1.0
        q[..., 3] = q[..., 6] = 1.0
        proper = proper_endpoint_distances(q, z, r, 0.4, 1.2)
        proper_error = max(
            abs(proper["compact_axis_endpoint_to_brane"] - 0.4),
            abs(proper["radial_axis_to_brane_endpoint"] - 1.2),
        )
        adverse_initial = domain_initial_dominance(10.0, 1.0, 10.5)
        adverse_evolved = domain_initial_dominance(10.0, 4.0, 14.0)
        passed = bool(
            tensor["passed"] and bvp["passed"] and proper_error < 1e-10
            and adverse_initial["initial_data_dominated"]
            and not adverse_evolved["initial_data_dominated"]
        )
        return {
            "passed": passed, "tensor": tensor, "BVP": bvp,
            "flat_proper_distance_error": proper_error,
            "adverse_initial_control": adverse_initial,
            "adverse_evolution_control": adverse_evolved,
        }
    payload, _ = stage_json(
        index, "controls/all", RECOVERY_ROOT / "controls.json", "manufactured-controls",
        {}, producer, expected=600.0,
    )
    if not payload["passed"]:
        raise RuntimeError("sealed Test-10 controls failed")
    return payload


def all_manifest_stages_validate(index):
    return bool(all(
        stage.get("status") == "complete"
        and index.validated_path(stage_id) is not None
        for stage_id, stage in index.data["stages"].items()
    ))


def analysis_stage(index, stage_id, filename, producer, expected=2400.0):
    """Persist a JSON-safe analysis so combination itself is restartable."""
    payload, _ = stage_json(
        index, stage_id, RECOVERY_ROOT / filename, "analysis", {},
        lambda: {"analysis": producer()}, expected=expected,
    )
    return payload["analysis"]


def assemble_state(index, geometries, cases, runs):
    values = {}
    for run_label in ("G10_coarse", "G10_half", "G11_standard"):
        run = runs[run_label]
        for name, key in (
            ("position", "_position"), ("velocity", "_velocity"),
            ("source_increment", "_source_increment"),
        ):
            values[f"{run_label}_{name}"] = run[key]
    for label in ("G7", "G8"):
        run_label = f"R10{label}_long"
        values[f"R10{label}_z"] = geometries[f"R10{label}"]["z"]
        values[f"R10{label}_r"] = geometries[f"R10{label}"]["r"]
        for step in LONG_SURFACE_STEPS:
            state = load_segment_end(run_label, step, cases[f"R10{label}"])
            values[f"{run_label}_step_{step:03d}_position"] = state["position"]
            values[f"{run_label}_step_{step:03d}_velocity"] = state["velocity"]
            values[f"{run_label}_step_{step:03d}_source"] = state["source"]
    stage_id = "final/state"
    index.register(stage_id, "compact-state-archive", 1200.0, {"arrays": len(values)})
    cached = index.validated_path(stage_id)
    if cached is None:
        index.mark_running(stage_id)
        started = time.perf_counter()
        atomic_write_npz(STATE_OUTPUT, **values)
        validate_npz(STATE_OUTPUT)
        index.mark_complete(stage_id, STATE_OUTPUT, time.perf_counter() - started)
    else:
        validate_npz(cached)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("all", "short", "long", "combine"), default="all",
    )
    args = parser.parse_args()
    overall_started = time.perf_counter()
    index = RecoveryIndex(MANIFEST, PROTOCOL, recovery_inputs(), maximum_stage_seconds=2400.0)
    controls = controls_stage(index)
    geometries = build_all_geometries()
    cases = make_cases(geometries)

    initial = {}
    for label in ("G10", "G11", "R10G7", "R10G8"):
        initial[label] = initial_stage(index, label, geometries[label])

    run_labels = []
    if args.phase in ("all", "short"):
        run_labels.extend(("G10_coarse", "G10_half", "G11_standard"))
    if args.phase in ("all", "long"):
        run_labels.extend(("R10G7_long", "R10G8_long"))
    if args.phase == "combine":
        run_labels = list(EVOLUTION_SPECS)

    runs = {}
    for run_label in run_labels:
        spec = EVOLUTION_SPECS[run_label]
        case = cases[spec["grid"]]
        print(f"running/resuming {run_label}", flush=True)
        state, diagnostics, _ = run_evolution(index, run_label, case)
        runs[run_label] = finalize_run(case, state, diagnostics, spec["steps"], spec["dt"])

    short_searches = {}
    long_searches = {}
    long_geometry = {}
    if args.phase in ("all", "short", "combine"):
        for run_label, steps in (
            ("G10_coarse", range(1, 5)),
            ("G10_half", range(1, 17)),
            ("G11_standard", range(1, 9)),
        ):
            spec = EVOLUTION_SPECS[run_label]
            short_searches[run_label] = detector_records(
                index, run_label, geometries[spec["grid"]], cases[spec["grid"]], steps,
            )
    if args.phase in ("all", "long", "combine"):
        for label in ("G7", "G8"):
            run_label = f"R10{label}_long"
            long_searches[run_label] = detector_records(
                index, run_label, geometries[f"R10{label}"], cases[f"R10{label}"],
                LONG_SURFACE_STEPS,
            )
            long_geometry[run_label] = {}
            for step in LONG_SURFACE_STEPS:
                state = load_step(run_label, step, cases[f"R10{label}"])
                long_geometry[run_label][step] = geometry_stage(
                    index, run_label, step, long_searches[run_label][step], state,
                    geometries[f"R10{label}"],
                )

    if args.phase in ("short", "long"):
        print(json.dumps({
            "phase": args.phase, "status": "stage_complete",
            "new_stage_seconds": time.perf_counter() - overall_started,
        }, indent=2), flush=True)
        return

    if set(runs) != set(EVOLUTION_SPECS):
        raise RuntimeError("all evolution stages are required for combination")
    short = analysis_stage(
        index, "analysis/spatial_time", "analysis_spatial_time.json",
        lambda: spatial_time_analysis(
            geometries, cases, runs, short_searches,
        ),
    )
    long = analysis_stage(
        index, "analysis/long_domain", "analysis_long_domain.json",
        lambda: long_domain_analysis(
            geometries, cases, runs, long_searches, long_geometry,
        ),
    )
    analysis_stage(
        index, "analysis/elliptic_decomposition",
        "analysis_elliptic_decomposition.json",
        lambda: {
            "R8_R10_R12_t004_tensor_decomposition": long[
                "R8_R10_R12_t004_tensor_decomposition"
            ],
            "R8_R10_R12_intrinsic_surface_shrink": long[
                "R8_R10_R12_intrinsic_surface_shrink"
            ],
        },
    )

    all_new_evolutions_pass = bool(all(evolution_pass(run) for run in runs.values()))
    rule1 = bool(
        initial["G11"]["grid_size"] == [145, 217]
        and np.isclose(geometries["G11"]["r"][1] - geometries["G11"]["r"][0], 8.0 / 216.0)
        and short["gates"]["G11_formation"]
        and evolution_pass(runs["G11_standard"])
    )
    rule2 = bool(
        short["gates"]["G10_time_detector"]
        and evolution_pass(runs["G10_coarse"])
        and evolution_pass(runs["G10_half"])
    )
    rule3 = bool(
        long["gates"]["long_counts"]
        and evolution_pass(runs["R10G7_long"])
        and evolution_pass(runs["R10G8_long"])
    )
    rule4 = short["gates"]["surface_geometry"]
    rule5 = long["gates"]["long_geometry"]
    rule6 = short["gates"]["spatial_tensors"]
    rule7 = short["gates"]["temporal_tensors"]
    rule8 = bool(
        long["gates"]["three_domain_tensors"]
        and long["gates"]["intrinsic_surface_shrink"]
        and long["gates"]["long_tensors"]
    )
    rule9 = long["gates"]["initial_data_dominance"]
    rule10 = bool(long["gates"]["outer_corrections"] and long["gates"]["causal"])
    assemble_state(index, geometries, cases, runs)
    prefinal_recovery_pass = all_manifest_stages_validate(index)
    rule11 = bool(controls["passed"] and prefinal_recovery_pass)
    acceptance = {
        "1_G11_exact_formation_and_evolution": rule1,
        "2_G10_three_timestep_detector_consistency": rule2,
        "3_R10_long_evolution_and_pair_persistence": rule3,
        "4_short_grid_time_branch_geometry": rule4,
        "5_long_R8_R10_branch_geometry": rule5,
        "6_fifth_grid_physical_tensor_order": rule6,
        "7_G10_temporal_order_and_error_separation": rule7,
        "8_domain_tensor_and_intrinsic_geometry_shrink": rule8,
        "9_elliptic_initial_data_dominance": rule9,
        "10_outer_boundary_and_causal_control": rule10,
        "11_controls_provenance_and_recovery": rule11,
    }
    subclaims = {
        "pair_existence_persistence": "pass" if all((rule1, rule2, rule3, rule10, rule11)) else "review",
        "branch_geometry": "pass" if all((rule4, rule5, rule11)) else "review",
        "physical_fields_tensors": "pass" if all((rule6, rule7, rule8, rule9, rule10, rule11)) else "review",
        "formation_time": "domain_indexed_not_converged_for_current_family",
    }
    valid_pair_loss = bool(
        all_new_evolutions_pass and (
            not short["gates"]["G11_formation"] or not long["gates"]["long_counts"]
        )
    )
    simultaneous_tensor_growth = bool(all(
        not short["spatial_physical_tensors"][family]["strictly_decreasing"]
        and short["spatial_physical_tensors"][family]["pairs"]["G10_G11"]["relative_difference"] >= 0.05
        for family in ("metric_increment", "ADM_K")
    ))
    if valid_pair_loss or simultaneous_tensor_growth:
        status = "fail"
        classification = "pair_or_physical_tensor_convergence_failure"
    elif all(acceptance.values()):
        status = "pass"
        classification = "publication_scoped_pair_geometry_tensor_convergence"
    else:
        status = "review"
        classification = "publication_convergence_matrix_mixed"

    payload = {
        "status": status, "classification": classification,
        "scope": "sealed Test-10 joint grid/time/domain convergence audit",
        "protocol": str(PROTOCOL), "protocol_sha256": index.protocol_sha256,
        "preserved_prior_statuses": {
            "note79": "review", "note83": "review", "note85": "review",
            "note88": "review", "note89": "review",
        },
        "new_grids": {
            "G11_R8": [145, 217], "R10G7": [81, 151], "R10G8": [97, 181],
        },
        "evolution_specs": EVOLUTION_SPECS,
        "provenance": {
            "fixed_inputs": recovery_inputs(), "manifest": str(MANIFEST),
            "manifest_sha256_before_final": sha256_file(MANIFEST),
        },
        "controls": controls, "initial_searches": initial,
        "evolution_diagnostics": {
            label: public_diagnostics(run) for label, run in runs.items()
        },
        "short_spatial_time_analysis": short,
        "long_domain_analysis": long,
        "acceptance": acceptance, "subclaims": subclaims,
        "recovery": {
            "validated_stage_count_before_final": len(index.data["stages"]),
            "all_prefinal_stages_validated": prefinal_recovery_pass,
            "state_archive": str(STATE_OUTPUT),
        },
        "runtime": {
            "wall_seconds_this_invocation": time.perf_counter() - overall_started,
            "cumulative_stage_compute_seconds": float(sum(
                stage.get("elapsed_seconds", 0.0) for stage in index.data["stages"].values()
            )),
        },
        "limitations": [
            "formation time remains domain-indexed and is explicitly excluded from the PASS claim",
            "fixed baseline foliation and finite twelve-seed donor-capped BVP class",
            "equal-coordinate tensor identification plus separately integrated proper endpoint distances",
            "R12 t=0.004 comparison retains its sealed coarse long-run time step",
            "not global initial horizonlessness, event-horizon formation, topology change, nonlinear basin, connected geometry, dark matter, or mass transfer",
        ],
    }
    final_stage = "final/result"
    index.register(final_stage, "combined-result", 300.0, {"status": status})
    cached_final = index.validated_path(final_stage)
    if cached_final is None:
        index.mark_running(final_stage)
        atomic_write_json(OUTPUT, payload)
        index.mark_complete(final_stage, OUTPUT, 0.0)
    else:
        payload = json.loads(cached_final.read_text())
    print(json.dumps({
        "status": status, "classification": classification,
        "subclaims": subclaims, "acceptance": acceptance,
        "spatial_orders": {
            family: short["spatial_physical_tensors"][family]["order"]
            for family in ("metric_increment", "ADM_K")
        },
        "temporal_orders": {
            family: short["temporal_physical_tensors"][family]["order"]
            for family in ("metric_increment", "ADM_K")
        },
        "runtime": payload["runtime"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
