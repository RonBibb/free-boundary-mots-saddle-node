#!/usr/bin/env python3
"""Sealed, resumable Test-10B common-parent domain-normalization audit."""

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
from bhps.corrected_A790_R12_builder import build_A790_R12_pair
from bhps.corrected_A790_test10_convergence import proper_endpoint_distances
from bhps.corrected_A790_test10b_domain_normalized import (
    array_relative_difference,
    brackets_overlap,
    classify_test10b,
    common_radius_invariants,
    first_detection_bracket,
    invariant_transfer,
    relative_difference,
    restrict_geometry,
    restriction_identity,
    tensor_domain_transfer,
    valid_persistent_pair_history,
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
from run_corrected_fold_short_nonlinear_evolution import relative_norm


PROTOCOL = Path("notes/98_A790_domain_normalized_family_protocol.md")
OUTPUT = Path("results/corrected_A790_test10b_domain_normalized.json")
STATE_OUTPUT = Path("results/corrected_A790_test10b_domain_normalized_state.npz")
RECOVERY_ROOT = Path("results/corrected_A790_test10b_domain_normalized_recovery")
MANIFEST = RECOVERY_ROOT / "index.json"
ORIGINAL_TEST10 = Path("results/corrected_A790_test10_joint_convergence.json")
DT = 0.000125
STEPS = 16
FINAL_TIME = 0.002
SEGMENT = 4
R_CUT = 6.0
DOMAINS = ("R8", "R10", "R12")
GRIDS = ("G7", "G8")
GEOMETRY_STEPS = (10, 12, 14, 16)

FIXED_INPUT_HASHES = {
    "notes/82_A790_larger_radial_domain_protocol.md": "d58dab4b3bd9b2ac08b602c5164d16b6cbc0ba2d5f2a79674530566c715fcc3b",
    "notes/88_A790_R12_domain_sequence_protocol.md": "6a30701fb76b8c9b801d8e4a9cd9ed91cab4038092f73547d2843934f5d84356",
    "notes/89_A790_fourth_grid_physical_tensor_convergence_protocol.md": "f3edbca20287805f95810ad25b6aa59a814136eea0491ca59721180d49c9f0d3",
    "notes/93_A790_publication_joint_convergence_protocol.md": "c27ec08d2b8d7332e6d09601f5e73450268f5374d68832fb6c0d012532af9b01",
    "notes/93_A790_publication_joint_convergence_result.md": "ffd3e6182044109c075e8c8fbaa1f16f48fc42c7eb1c3807f9f047b12e4686bb",
    "results/corrected_family_knot_A8_state.npz": "e293184299baf5b791f6a49a91f6e6266ad656072f2e8fd2e39ce21d14c5416e",
    "results/corrected_A790_R10_domain_robustness.json": "7b89042ddfbec2a4353289bfc05ddb6e5d23f87e3f8e39a3f34caa4ef681af2d",
    "results/corrected_A790_R12_domain_sequence.json": "8dd9119315d5b62113dfc5c541884d1bc696baba73bf9ee7bbbaa6ae1e4389a1",
    "results/corrected_A790_R12_domain_sequence_state.npz": "994c0d0e061c4da461ba8ba56a96716393656351071033fd087ae830ffcb947a",
    "results/corrected_A790_test10_joint_convergence.json": "da8c375216b4ca7e42529c711a403b0a2210f9a415df6200629098191ea758bf",
    "src/bhps/corrected_A790_R12_builder.py": "a0d0a5e7c12fef5bdacb2c97710787266c5ed3beba5a435dd7653edaf88322cc",
    "src/bhps/recovery_indexer.py": "1460478fba42433bd340a2ef9e09c0946882a35d3eb63c2c95ea9b055bb549fa",
    "src/bhps/corrected_A790_test10_convergence.py": "10f8117bfdac2ae9b12e1c8978f325d479dd3e30657d80dbc06ec44c136fd135",
    "src/bhps/corrected_A790_physical_tensor_convergence.py": "d7dd8f88bcfc73f9b7efb8bbdbc0caa4722e2a6583c6ef52733d251edb49dcec",
    "run_corrected_fold_live_nonlinear_gauge_source.py": "b886bf79d57f98b372d8f756d22016f56192d1816b893536e4f6fd5ac242c203",
    "run_corrected_A790_independent_dynamic_BVP_detector.py": "4c71239822a5674cb9166483b647effdccc5b2c629d7e5b936ac7933cbc2a1f4",
}


def run_label(grid, domain):
    return f"{grid}_{domain}"


def recovery_inputs():
    dynamic = (
        Path(__file__),
        Path("src/bhps/corrected_A790_test10b_domain_normalized.py"),
        Path("tests/test_A790_test10b_domain_normalized.py"),
        Path("tests/test_A790_test10b_runner_recovery.py"),
    )
    return {**FIXED_INPUT_HASHES, **{
        str(path): sha256_file(path) for path in dynamic
    }}


def case_fingerprint(case):
    digest = hashlib.sha256()
    for value in (
        case["z"], case["r"], case["initial"], case["source0"], case["memory0"],
        case["geometry"]["jet_field"].reduced_first,
        case["geometry"]["jet_field"].reduced_second,
    ):
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode())
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def stage_json(index, stage_id, path, kind, metadata, producer, expected=1200.0):
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
            "stage_id": stage_id,
            "protocol_sha256": index.protocol_sha256,
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


def build_geometries():
    print("constructing Test-10B R12 common parents", flush=True)
    g7, g8 = build_A790_R12_pair()
    parents = {"G7": g7, "G8": g8}
    geometries = {}
    for grid, parent in parents.items():
        for domain, endpoint in (("R8", 8.0), ("R10", 10.0), ("R12", 12.0)):
            geometries[run_label(grid, domain)] = restrict_geometry(
                parent, endpoint, f"{grid}-A790-test10b-{domain}",
            )
    return parents, geometries


def make_cases(geometries):
    return {
        label: live.setup_case(
            geometry, f"{label}-A790-test10b", live_normal_wall_gauge=True,
            live_outer_sommerfeld=True,
        ) for label, geometry in geometries.items()
    }


def parent_stage(index, grid, parent, geometries):
    stage_id = f"parent/{grid}"
    path = RECOVERY_ROOT / f"parent_{grid}.json"
    payload, _ = stage_json(
        index, stage_id, path, "common-parent-restriction",
        {"grid": grid, "parent": parent["name"]},
        lambda: {
            "parent_grid_size": [len(parent["z"]), len(parent["r"])],
            "parent_reference_residual": float(parent["reference_maximum_residual"]),
            "parent_selector_residual": float(parent["selector_maximum"]),
            "restrictions": {
                domain: {
                    "grid_size": list(geometries[run_label(grid, domain)]["source_grid"]),
                    "r_max": float(geometries[run_label(grid, domain)]["r"][-1]),
                    "identity": restriction_identity(
                        parent, geometries[run_label(grid, domain)],
                    ),
                } for domain in DOMAINS
            },
        },
    )
    if (
        payload["parent_reference_residual"] >= 1e-9
        or payload["parent_selector_residual"] >= 1e-9
        or not all(item["identity"]["passed"] for item in payload["restrictions"].values())
    ):
        raise RuntimeError(f"sealed common-parent stop on {grid}")
    return payload


def initial_stage(index, label, geometry):
    stage_id = f"initial/{label}"
    path = RECOVERY_ROOT / f"initial_{label}.json"
    position = np.asarray(geometry["jet_field"].reduced_fields)
    payload, _ = stage_json(
        index, stage_id, path, "initial-zero-cap-search",
        {"run": label, "grid_size": list(geometry["source_grid"])},
        lambda: {
            "static": static_search(geometry),
            "BVP": search_slice(
                f"{label}-A790-test10b-t0", position, np.zeros_like(position),
                geometry,
            ),
        },
    )
    if (
        payload["static"]["accepted_count"] != 0
        or payload["BVP"]["admitted_distinct_count"] != 0
    ):
        raise RuntimeError(f"sealed initial-cap stop on {label}")
    return payload


def empty_segment_diagnostics():
    return {
        "all_stages_finite": True,
        "maximum_stage_acceleration_relative_change": 0.0,
        "maximum_wall_position_residual": 0.0,
        "maximum_normal_wall_position_residual": 0.0,
        "maximum_normal_wall_acceleration_residual": 0.0,
        "maximum_outer_acceleration_residual": 0.0,
        "maximum_outer_source_residual": 0.0,
        "maximum_outer_metric_correction": 0.0,
        "maximum_outer_scalar_correction": 0.0,
        "maximum_outer_source_correction": 0.0,
    }


def update_segment_diagnostics(summary, k1, k2, diagnostics):
    summary["maximum_stage_acceleration_relative_change"] = max(
        summary["maximum_stage_acceleration_relative_change"],
        relative_norm(k1[1], k2[1]),
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


def integrate_segment(case, state, start_step, end_step):
    current = tuple(np.asarray(value).copy() for value in state)
    diagnostic = empty_segment_diagnostics()
    snapshots = {}
    for step in range(start_step + 1, end_step + 1):
        current_time = (step - 1) * DT
        print(f"{case['label']}: restartable step {step}/{STEPS}, stage 1", flush=True)
        k1, d1 = live.driver_stage(case, current_time, *current)
        midpoint = tuple(value + 0.5 * DT * slope for value, slope in zip(current, k1))
        print(f"{case['label']}: restartable step {step}/{STEPS}, stage 2", flush=True)
        k2, d2 = live.driver_stage(case, current_time + 0.5 * DT, *midpoint)
        update_segment_diagnostics(diagnostic, k1, k2, (d1, d2))
        current = tuple(value + DT * slope for value, slope in zip(current, k2))
        wall = live.compact_wall_position_residuals(
            current[0], case["z"], case["r"], case["geometry"]["background"],
        )
        normal_wall = live.compact_wall_normal_gauge_position_residuals(
            current[0], current[2], case["z"], case["r"],
            case["geometry"]["background"],
        )
        diagnostic["maximum_wall_position_residual"] = max(
            diagnostic["maximum_wall_position_residual"], wall["maximum"],
        )
        diagnostic["maximum_normal_wall_position_residual"] = max(
            diagnostic["maximum_normal_wall_position_residual"],
            normal_wall["maximum"],
        )
        snapshots[f"step_{step:03d}_increment"] = current[0] - case["initial"]
        snapshots[f"step_{step:03d}_velocity"] = current[1].copy()
        snapshots[f"step_{step:03d}_source_increment"] = current[2] - case["source0"]
    return current, snapshots, diagnostic


def diagnostic_arrays(record):
    return {f"diag_{key}": np.asarray(value) for key, value in record.items()}


def diagnostic_from_archive(archive):
    return {
        key.removeprefix("diag_"): archive[key].item()
        for key in archive.files if key.startswith("diag_")
    }


def segment_path(label, start, end):
    return RECOVERY_ROOT / f"evolution_{label}_steps_{start + 1:03d}_{end:03d}.npz"


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
        required[f"step_{step:03d}_source_increment"] = source_shape
    validate_npz(path, required)
    with np.load(path) as archive:
        if int(archive["start_step"]) != start or int(archive["end_step"]) != end:
            raise RuntimeError("segment index mismatch")


def run_evolution(index, label, case):
    state = (
        case["initial"].copy(), np.zeros_like(case["initial"]),
        case["source0"].copy(), case["memory0"].copy(),
    )
    parent_sha = case_fingerprint(case)
    diagnostics = []
    paths = []
    for start in range(0, STEPS, SEGMENT):
        end = min(start + SEGMENT, STEPS)
        stage_id = f"evolution/{label}/steps_{start + 1:03d}_{end:03d}"
        path = segment_path(label, start, end)
        metadata = {
            "run": label, "start_step": start, "end_step": end, "dt": DT,
            "parent_sha256": parent_sha, "case_fingerprint": case_fingerprint(case),
        }
        index.register(stage_id, "evolution-segment", 2400.0, metadata)
        cached = index.validated_path(stage_id)
        if cached is None:
            index.mark_running(stage_id)
            started = time.perf_counter()
            try:
                state, snapshots, diagnostic = integrate_segment(case, state, start, end)
                if not diagnostic["all_stages_finite"]:
                    raise RuntimeError("nonfinite evolution stage")
                if max(
                    diagnostic["maximum_outer_metric_correction"],
                    diagnostic["maximum_outer_scalar_correction"],
                    diagnostic["maximum_outer_source_correction"],
                ) > 0.20:
                    raise RuntimeError("sealed 20-percent outer-correction stop")
                if not live.signature_summary(state[0], case["r"])[
                    "all_points_one_negative_direction"
                ]:
                    raise RuntimeError("lost Lorentzian signature")
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


def finalize_run(case, state, diagnostics):
    position, velocity, source, memory = state
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
    source_dot, final_outer_source = live.apply_outer_source_sommerfeld(
        source, source_dot, case["source0"], case["source_time0"],
        case["_initial_source_second_time"], position, FINAL_TIME,
        case["r"], case["rhs"].stencil_width,
    )
    gauge = live.StageRegularGaugeSource(source, source_dot, case["z"], case["r"])
    print(f"{case['label']}: exact final diagnostics", flush=True)
    return {
        "final_time": FINAL_TIME, "steps": STEPS, "time_step": DT,
        **combine_segment_diagnostics(diagnostics),
        "final_constraint": live.gauge_constraint_summary(
            position, velocity, FINAL_TIME, case["rhs"], R_CUT, gauge,
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
                case["rhs"].outer_reference_acceleration, FINAL_TIME,
                case["r"], case["rhs"].stencil_width,
            ),
        "final_outer_source_sommerfeld_residual": final_outer_source,
        "_position": position, "_velocity": velocity, "_source": source,
        "_memory": memory, "_increment": position - case["initial"],
        "_source_increment": source - case["source0"],
    }


def test10b_public_diagnostics(run):
    return {
        **public_diagnostics(run),
        "maximum_wall_position_residual": run[
            "maximum_wall_position_residual"
        ],
        "maximum_normal_wall_position_residual": run[
            "maximum_normal_wall_position_residual"
        ],
    }


def load_step(label, step, case):
    start = ((int(step) - 1) // SEGMENT) * SEGMENT
    end = min(start + SEGMENT, STEPS)
    with np.load(segment_path(label, start, end)) as archive:
        return {
            "position": case["initial"] + archive[f"step_{step:03d}_increment"],
            "velocity": np.asarray(archive[f"step_{step:03d}_velocity"]),
            "source_increment": np.asarray(
                archive[f"step_{step:03d}_source_increment"]
            ),
        }


def detector_stage(index, label, step, state, geometry):
    stage_id = f"detector/{label}/step_{step:03d}"
    path = RECOVERY_ROOT / f"detector_{label}_step_{step:03d}.json"
    payload, _ = stage_json(
        index, stage_id, path, "independent-BVP",
        {"run": label, "step": int(step), "time": step * DT},
        lambda: {"search": search_slice(
            f"{label}-test10b-t{step * DT:.6f}", state["position"],
            state["velocity"], geometry,
        )},
        expected=1200.0,
    )
    return payload["search"]


def representative_geometry(search, state, geometry):
    if search["admitted_distinct_count"] != 2:
        return []
    prepared = prepare_capped_expansion_slice(
        state["position"], state["velocity"], geometry["z"], geometry["r"],
    )
    records = []
    for branch_name, cluster in zip(
        ("inner", "outer"),
        sorted(search["clusters"], key=lambda item: item["signature"][1]),
    ):
        members = sorted(cluster["members"], key=lambda item: item["seed"])
        seed = float(members[len(members) // 2]["seed"])
        started = time.perf_counter()
        surface = solve_dynamical_capped_surface_bvp(
            state["position"], state["velocity"], geometry["z"], geometry["r"],
            seed, tolerance=2e-5, nodes=121, maximum_nodes=6000,
            dense_nodes=501,
        )
        surface["runtime_seconds"] = float(time.perf_counter() - started)
        geometric = capped_surface_geometry(
            state["position"], state["velocity"], geometry["z"], geometry["r"],
            surface, prepared=prepared,
        )
        records.append({
            "branch": branch_name, "seed": seed,
            "admitted": bvp_admitted(surface),
            "surface": public_bvp_surface(surface), "geometry": geometric,
            "proper_endpoints": proper_endpoint_distances(
                state["position"], geometry["z"], geometry["r"],
                surface["rho_axis"], surface["rho_brane"],
            ),
        })
    return records


def geometry_stage(index, label, step, search, state, geometry):
    stage_id = f"geometry/{label}/step_{step:03d}"
    path = RECOVERY_ROOT / f"geometry_{label}_step_{step:03d}.json"
    payload, _ = stage_json(
        index, stage_id, path, "proper-branch-geometry",
        {"run": label, "step": int(step), "time": step * DT},
        lambda: {"branches": representative_geometry(search, state, geometry)},
        expected=1200.0,
    )
    return payload["branches"]


def surface_transfer(left, right):
    if len(left) != 2 or len(right) != 2:
        return {"usable": False, "branches": [], "maximum": None}
    records = []
    values = []
    for a, b in zip(left, right):
        record = {"branch": a["branch"]}
        for name, av, bv in (
            ("axis", a["geometry"]["rho_axis"], b["geometry"]["rho_axis"]),
            ("brane", a["geometry"]["rho_brane"], b["geometry"]["rho_brane"]),
            ("area", a["geometry"]["one_sided_cap_area"], b["geometry"]["one_sided_cap_area"]),
            ("equivalent_radius", a["geometry"]["equivalent_area_radius"], b["geometry"]["equivalent_area_radius"]),
            ("meridional_length", a["geometry"]["proper_meridional_length"], b["geometry"]["proper_meridional_length"]),
            ("proper_axis", a["proper_endpoints"]["compact_axis_endpoint_to_brane"], b["proper_endpoints"]["compact_axis_endpoint_to_brane"]),
            ("proper_brane", a["proper_endpoints"]["radial_axis_to_brane_endpoint"], b["proper_endpoints"]["radial_axis_to_brane_endpoint"]),
        ):
            record[name] = relative_difference(av, bv)
            values.append(record[name])
        records.append(record)
    return {"usable": True, "branches": records, "maximum": max(values)}


def common_prefix(array, geometry, radius_cut=R_CUT):
    keep = np.asarray(geometry["r"]) <= float(radius_cut) + 1e-12
    return np.asarray(array)[:, keep]


def normalization_analysis(parents, geometries, cases, parent_records, initial):
    result = {"grids": {}}
    maxima = {
        "invariant": 0.0, "source": 0.0, "memory": 0.0,
        "acceleration": 0.0,
    }
    all_exact = True
    parent_residuals = []
    initial_controls = []
    for grid in GRIDS:
        parent = parents[grid]
        parent_residuals.extend((
            float(parent["reference_maximum_residual"]),
            float(parent["selector_maximum"]),
        ))
        grid_result = {
            "restrictions": parent_records[grid]["restrictions"],
            "pairs": {},
        }
        invariants = {}
        for domain in DOMAINS:
            label = run_label(grid, domain)
            geometry = geometries[label]
            case = cases[label]
            invariants[domain] = common_radius_invariants(
                case["initial"], case["z"], case["r"], R_CUT,
            )
            initial_controls.append(bool(
                case["initial_constraint"]["global_relative"] < 0.005
                and case["initial_wall"]["maximum"] < 5e-4
                and case["initial_live_finite"] and case["initial_Taylor_finite"]
                and initial[label]["static"]["accepted_count"] == 0
                and initial[label]["BVP"]["admitted_distinct_count"] == 0
            ))
            all_exact = bool(
                all_exact and parent_records[grid]["restrictions"][domain][
                    "identity"
                ]["passed"]
            )
        for left, right in (("R8", "R10"), ("R10", "R12")):
            left_label = run_label(grid, left)
            right_label = run_label(grid, right)
            left_geometry = geometries[left_label]
            right_geometry = geometries[right_label]
            left_case = cases[left_label]
            right_case = cases[right_label]
            invariant = invariant_transfer(invariants[left], invariants[right])
            source = array_relative_difference(
                common_prefix(left_case["source0"], left_geometry),
                common_prefix(right_case["source0"], right_geometry),
            )
            memory = array_relative_difference(
                common_prefix(left_case["memory0"], left_geometry),
                common_prefix(right_case["memory0"], right_geometry),
            )
            acceleration = array_relative_difference(
                common_prefix(left_case["_initial_live_acceleration"], left_geometry),
                common_prefix(right_case["_initial_live_acceleration"], right_geometry),
            )
            count = len(left_geometry["r"])
            exact_initial = bool(np.array_equal(
                left_case["initial"], right_case["initial"][:, :count],
            ))
            grid_result["pairs"][f"{left}_{right}"] = {
                "initial_array_equal": exact_initial,
                "invariant_transfer": invariant,
                "source_relative_difference": source,
                "memory_relative_difference": memory,
                "live_acceleration_relative_difference": acceleration,
            }
            all_exact = bool(all_exact and exact_initial)
            maxima["invariant"] = max(maxima["invariant"], invariant["maximum"])
            maxima["source"] = max(maxima["source"], source)
            maxima["memory"] = max(maxima["memory"], memory)
            maxima["acceleration"] = max(maxima["acceleration"], acceleration)
        result["grids"][grid] = grid_result
    result["maxima"] = maxima
    result["gate"] = bool(
        all_exact and max(parent_residuals) < 1e-9 and all(initial_controls)
        and maxima["invariant"] < 1e-12
        and max(maxima["source"], maxima["memory"], maxima["acceleration"]) < 1e-10
    )
    return result


def search_quality(records):
    for search in records.values():
        for cluster in search["clusters"]:
            if len(cluster["members"]) < 2:
                return False
        for trial in search["trials"]:
            if trial["admitted"] and not bvp_admitted(trial["surface"]):
                return False
    return True


def formation_analysis(searches):
    histories = {
        label: [searches[label][step]["admitted_distinct_count"] for step in range(1, STEPS + 1)]
        for label in searches
    }
    brackets = {
        label: first_detection_bracket(counts, DT) for label, counts in histories.items()
    }
    domain_identical = {
        grid: bool(
            histories[run_label(grid, "R8")]
            == histories[run_label(grid, "R10")]
            == histories[run_label(grid, "R12")]
        ) for grid in GRIDS
    }
    grid_brackets = {grid: brackets[run_label(grid, "R12")] for grid in GRIDS}
    all_zero = bool(all(2 not in counts for counts in histories.values()))
    pair_gate = bool(
        all(valid_persistent_pair_history(counts) and counts[-1] == 2 for counts in histories.values())
        and all(search_quality(searches[label]) for label in searches)
    )
    formation_gate = bool(
        not all_zero and all(domain_identical.values())
        and brackets_overlap(grid_brackets["G7"], grid_brackets["G8"])
    )
    return {
        "histories": histories, "brackets": brackets,
        "same_grid_domain_histories_identical": domain_identical,
        "G7_G8_brackets_overlap": brackets_overlap(
            grid_brackets["G7"], grid_brackets["G8"],
        ),
        "all_histories_right_censored": all_zero,
        "pair_gate": pair_gate, "formation_gate": formation_gate,
    }


def geometry_analysis(geometry_records):
    domain_transfers = {}
    grid_transfers = {}
    maxima = {"domain": 0.0, "grid": 0.0}
    usable = True
    ordering = True
    for grid in GRIDS:
        domain_transfers[grid] = {}
        for step in GEOMETRY_STEPS:
            domain_transfers[grid][str(step)] = {}
            for left, right in (("R8", "R10"), ("R10", "R12")):
                transfer = surface_transfer(
                    geometry_records[run_label(grid, left)][step],
                    geometry_records[run_label(grid, right)][step],
                )
                domain_transfers[grid][str(step)][f"{left}_{right}"] = transfer
                usable = bool(usable and transfer["usable"])
                if transfer["maximum"] is not None:
                    maxima["domain"] = max(maxima["domain"], transfer["maximum"])
    for domain in DOMAINS:
        grid_transfers[domain] = {}
        for step in GEOMETRY_STEPS:
            transfer = surface_transfer(
                geometry_records[run_label("G7", domain)][step],
                geometry_records[run_label("G8", domain)][step],
            )
            grid_transfers[domain][str(step)] = transfer
            usable = bool(usable and transfer["usable"])
            if transfer["maximum"] is not None:
                maxima["grid"] = max(maxima["grid"], transfer["maximum"])
    for records in geometry_records.values():
        for branches in records.values():
            ordering = bool(
                ordering and len(branches) == 2
                and all(branch["admitted"] and branch["geometry"]["finite"] for branch in branches)
                and branches[1]["geometry"]["one_sided_cap_area"]
                > branches[0]["geometry"]["one_sided_cap_area"]
            )
    return {
        "domain_transfers": domain_transfers, "cross_grid_transfers": grid_transfers,
        "maxima": maxima, "usable": usable, "ordering": ordering,
        "gate": bool(usable and ordering and maxima["domain"] < 0.0025 and maxima["grid"] < 0.01),
    }


def tensor_analysis(geometries, cases):
    original = json.loads(ORIGINAL_TEST10.read_text())
    original_domain = original["long_domain_analysis"][
        "R8_R10_R12_t004_tensor_decomposition"
    ]
    records = {grid: {} for grid in GRIDS}
    common_interior_history = {grid: {} for grid in GRIDS}
    maximum_relative = 0.0
    maximum_source = 0.0
    initial_exact = True
    improves_original = True
    for grid in GRIDS:
        for left, right in (("R8", "R10"), ("R10", "R12")):
            pair = f"{left}_{right}"
            left_label = run_label(grid, left)
            right_label = run_label(grid, right)
            left_geometry = geometries[left_label]
            common_interior_history[grid][pair] = []
            for step in range(1, STEPS + 1):
                left_state = load_step(left_label, step, cases[left_label])
                right_state = load_step(right_label, step, cases[right_label])
                right_common = {
                    name: np.asarray(right_state[name])[:, :len(left_geometry["r"])]
                    for name in ("position", "velocity", "source_increment")
                }
                common_interior_history[grid][pair].append({
                    "step": step,
                    "time": step * DT,
                    "reduced_position_relative_difference": array_relative_difference(
                        common_prefix(left_state["position"], left_geometry),
                        common_prefix(right_common["position"], left_geometry),
                    ),
                    "reduced_velocity_relative_difference": array_relative_difference(
                        common_prefix(left_state["velocity"], left_geometry),
                        common_prefix(right_common["velocity"], left_geometry),
                    ),
                    "source_increment_relative_difference": array_relative_difference(
                        common_prefix(left_state["source_increment"], left_geometry),
                        common_prefix(right_common["source_increment"], left_geometry),
                    ),
                })
        for step in (0, 8, 16):
            key = f"{step * DT:.3f}"
            records[grid][key] = {}
            if step == 0:
                states = {
                    domain: {
                        "position": cases[run_label(grid, domain)]["initial"],
                        "velocity": np.zeros_like(
                            cases[run_label(grid, domain)]["initial"]
                        ),
                        "source_increment": np.zeros_like(
                            cases[run_label(grid, domain)]["source0"]
                        ),
                    } for domain in DOMAINS
                }
            else:
                states = {
                    domain: load_step(
                        run_label(grid, domain), step,
                        cases[run_label(grid, domain)],
                    ) for domain in DOMAINS
                }
            for left, right in (("R8", "R10"), ("R10", "R12")):
                left_label = run_label(grid, left)
                right_label = run_label(grid, right)
                left_geometry = geometries[left_label]
                right_state = {
                    name: np.asarray(states[right][name])[:, :len(left_geometry["r"])]
                    for name in ("position", "velocity", "source_increment")
                }
                transfer = tensor_domain_transfer(
                    states[left], right_state, left_geometry["z"], left_geometry["r"],
                    cases[left_label]["initial"], R_CUT,
                )
                source = array_relative_difference(
                    common_prefix(states[left]["source_increment"], left_geometry),
                    common_prefix(right_state["source_increment"], left_geometry),
                )
                transfer["source_increment_relative_difference"] = source
                pair = f"{left}_{right}"
                records[grid][key][pair] = transfer
                if step > 0:
                    maximum_source = max(maximum_source, source)
                    maximum_relative = max(maximum_relative, *(
                        transfer[family]["relative_difference"]
                        for family in ("full_metric", "metric_increment", "ADM_K")
                    ))
                    original_pair = "R8_R10" if pair == "R8_R10" else "R10_R12"
                    improves_original = bool(
                        improves_original
                        and transfer["full_metric"]["relative_difference"]
                        < original_domain[grid][original_pair]["final_metric"]["relative_difference"]
                    )
        r8 = cases[run_label(grid, "R8")]["initial"]
        r10 = cases[run_label(grid, "R10")]["initial"][:, :r8.shape[1]]
        r12 = cases[run_label(grid, "R12")]["initial"][:, :r8.shape[1]]
        initial_exact = bool(
            initial_exact and np.array_equal(r8, r10) and np.array_equal(r8, r12)
        )
    return {
        "records": records,
        "common_interior_evolution_history": common_interior_history,
        "initial_metric_array_exact": initial_exact,
        "maximum_primary_relative_difference": maximum_relative,
        "maximum_source_increment_relative_difference": maximum_source,
        "strictly_improves_original_full_metric": improves_original,
        "gate": bool(
            initial_exact and maximum_relative < 0.001
            and maximum_source < 0.005 and improves_original
        ),
    }


def causal_analysis(geometries):
    records = {}
    for label, geometry in geometries.items():
        speed = float(np.max(geometry["principal"]["r_coordinate_speed"]))
        lower = float((geometry["r"][-1] - R_CUT) / speed)
        records[label] = {
            "maximum_initial_radial_coordinate_speed": speed,
            "one_way_boundary_to_r6_lower_bound": lower,
            "final_time_fraction": FINAL_TIME / lower,
            "passes": bool(FINAL_TIME < lower),
        }
    return {"records": records, "gate": bool(all(item["passes"] for item in records.values()))}


def controls_stage(index):
    def producer():
        tensor = analytic_tensor_controls()
        bvp = bvp_analytic_controls()
        z = np.linspace(1.0, 2.0, 33)
        r = np.linspace(0.0, 2.0, 49)
        q = np.zeros((len(z), len(r), 9))
        q[..., 2] = -1.0
        q[..., 3] = q[..., 6] = 1.0
        proper = proper_endpoint_distances(q, z, r, 0.5, 2.0)
        proper_error = max(
            abs(proper["compact_axis_endpoint_to_brane"] - 0.5),
            abs(proper["radial_axis_to_brane_endpoint"] - 2.0),
        )
        return {
            "passed": bool(tensor["passed"] and bvp["passed"] and proper_error < 1e-10),
            "tensor": tensor, "BVP": bvp,
            "flat_proper_distance_error": proper_error,
        }
    payload, _ = stage_json(
        index, "controls/all", RECOVERY_ROOT / "controls.json",
        "manufactured-controls", {}, producer,
    )
    if not payload["passed"]:
        raise RuntimeError("sealed Test-10B controls failed")
    return payload


def analysis_stage(index, name, producer):
    payload, _ = stage_json(
        index, f"analysis/{name}", RECOVERY_ROOT / f"analysis_{name}.json",
        "analysis", {}, lambda: {"analysis": producer()}, expected=2400.0,
    )
    return payload["analysis"]


def all_stages_validate(index):
    return bool(all(
        stage.get("status") == "complete" and index.validated_path(stage_id) is not None
        for stage_id, stage in index.data["stages"].items()
    ))


def assemble_state(index, geometries, cases):
    values = {"times": np.arange(1, STEPS + 1) * DT}
    for label in sorted(geometries):
        values[f"{label}_z"] = geometries[label]["z"]
        values[f"{label}_r"] = geometries[label]["r"]
        values[f"{label}_initial"] = cases[label]["initial"]
        for step in (8, 16):
            state = load_step(label, step, cases[label])
            for key in ("position", "velocity", "source_increment"):
                values[f"{label}_step_{step:03d}_{key}"] = state[key]
    stage_id = "final/state"
    index.register(stage_id, "state-archive", 1200.0, {"arrays": len(values)})
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
    parser.add_argument("--phase", choices=("all", "G7", "G8", "combine"), default="all")
    args = parser.parse_args()
    started = time.perf_counter()
    index = RecoveryIndex(MANIFEST, PROTOCOL, recovery_inputs(), maximum_stage_seconds=2400.0)
    controls = controls_stage(index)
    parents, geometries = build_geometries()
    cases = make_cases(geometries)
    parent_records = {
        grid: parent_stage(index, grid, parents[grid], geometries) for grid in GRIDS
    }
    initial = {
        label: initial_stage(index, label, geometry)
        for label, geometry in geometries.items()
    }

    active_grids = GRIDS if args.phase in ("all", "combine") else (args.phase,)
    runs = {}
    searches = {}
    geometry_records = {}
    for grid in active_grids:
        for domain in DOMAINS:
            label = run_label(grid, domain)
            print(f"running/resuming {label}", flush=True)
            state, diagnostics, _ = run_evolution(index, label, cases[label])
            runs[label] = finalize_run(cases[label], state, diagnostics)
            searches[label] = {}
            for step in range(1, STEPS + 1):
                step_state = load_step(label, step, cases[label])
                searches[label][step] = detector_stage(
                    index, label, step, step_state, geometries[label],
                )
            geometry_records[label] = {}
            for step in GEOMETRY_STEPS:
                step_state = load_step(label, step, cases[label])
                geometry_records[label][step] = geometry_stage(
                    index, label, step, searches[label][step], step_state,
                    geometries[label],
                )

    if args.phase in ("G7", "G8"):
        print(json.dumps({
            "phase": args.phase, "status": "stage_complete",
            "wall_seconds": time.perf_counter() - started,
        }, indent=2), flush=True)
        return

    if set(runs) != set(geometries):
        raise RuntimeError("all six runs are required for combination")

    normalization = analysis_stage(
        index, "normalization",
        lambda: normalization_analysis(
            parents, geometries, cases, parent_records, initial,
        ),
    )
    formation = analysis_stage(index, "formation", lambda: formation_analysis(searches))
    geometry = analysis_stage(index, "geometry", lambda: geometry_analysis(geometry_records))
    tensors = analysis_stage(index, "tensors", lambda: tensor_analysis(geometries, cases))
    causal = analysis_stage(index, "causality", lambda: causal_analysis(geometries))

    evolutions_pass = bool(all(evolution_pass(run) for run in runs.values()))
    cut_gate = bool(all(
        max(
            run["maximum_outer_acceleration_residual"],
            run["maximum_outer_source_residual"],
            run["final_outer_sommerfeld_position_residual"]["maximum_normalized"],
            run["final_outer_source_sommerfeld_residual"]["maximum_normalized"],
        ) < 1e-10
        and max(
            run["maximum_wall_position_residual"],
            run["maximum_normal_wall_position_residual"],
        ) < 5e-4
        and max(
            run["maximum_outer_metric_correction"],
            run["maximum_outer_scalar_correction"],
            run["maximum_outer_source_correction"],
        ) < 0.05 for run in runs.values()
    ))
    construction_gate = bool(normalization["gate"] and controls["passed"])
    pair_gate = bool(formation["pair_gate"] and evolutions_pass and cut_gate and causal["gate"])
    formation_gate = formation["formation_gate"]
    geometry_gate = geometry["gate"]
    tensor_gate = tensors["gate"]

    assemble_state(index, geometries, cases)
    provenance_gate = all_stages_validate(index)
    valid = bool(controls["passed"] and evolutions_pass and cut_gate and causal["gate"] and provenance_gate)
    histories = formation["histories"]
    cross_domain_mismatch = bool(any(
        not formation["same_grid_domain_histories_identical"][grid] for grid in GRIDS
    ))
    branch_loss = bool(any(
        2 in counts and any(
            left > right for left, right in zip(counts, counts[1:])
        ) for counts in histories.values()
    ))
    physical_fail = bool(
        valid and construction_gate and (
            cross_domain_mismatch or branch_loss
            or any(len({histories[run_label(grid, domain)][-1] for domain in DOMAINS}) > 1 for grid in GRIDS)
            or tensors["maximum_primary_relative_difference"] > 0.01
        )
    )
    status, classification = classify_test10b(
        valid, construction_gate, pair_gate, formation_gate,
        geometry_gate, tensor_gate, physical_fail,
    )
    subverdicts = {
        "common_parent_construction": "pass" if construction_gate else "review",
        "pair_existence_persistence": "pass" if pair_gate else "review",
        "formation_time": (
            "pass" if formation_gate else (
                "right_censored_domain_normalized_formation"
                if formation["all_histories_right_censored"] else "review"
            )
        ),
        "branch_geometry": "pass" if geometry_gate else "review",
        "physical_fields_tensors": "pass" if tensor_gate else "review",
    }
    acceptance = {
        "1_exact_common_parent_construction": construction_gate,
        "2_valid_evolution_and_cut_closure": bool(evolutions_pass and cut_gate),
        "3_pair_existence_and_persistence": pair_gate,
        "4_domain_normalized_formation": formation_gate,
        "5_branch_geometry": geometry_gate,
        "6_physical_fields_and_tensors": tensor_gate,
        "7_causality": causal["gate"],
        "8_controls_provenance_and_recovery": bool(controls["passed"] and provenance_gate),
    }
    payload = {
        "status": status, "classification": classification,
        "scope": "sealed Test-10B common-parent domain-normalization audit",
        "protocol": str(PROTOCOL), "protocol_sha256": index.protocol_sha256,
        "preserved_prior_statuses": {
            "note82": "review", "note88": "review", "note89": "review",
            "note93": "review",
        },
        "definition": {
            "parent": "independently solved R12 G7/G8 A=7.90 slices",
            "map": "exact radial-prefix restriction on nested nodes",
            "domains": list(DOMAINS), "grids": list(GRIDS),
            "steps": STEPS, "dt": DT, "final_time": FINAL_TIME,
            "common_radius": R_CUT,
        },
        "controls": controls, "parents": parent_records,
        "initial_searches": initial,
        "evolution_diagnostics": {
            label: test10b_public_diagnostics(run) for label, run in runs.items()
        },
        "normalization_analysis": normalization,
        "formation_analysis": formation,
        "geometry_analysis": geometry,
        "tensor_analysis": tensors,
        "causal_analysis": causal,
        "acceptance": acceptance, "subverdicts": subverdicts,
        "recovery": {
            "manifest": str(MANIFEST), "state_archive": str(STATE_OUTPUT),
            "validated_stage_count_before_final": len(index.data["stages"]),
            "all_prefinal_stages_validated": provenance_gate,
        },
        "runtime": {
            "wall_seconds_this_invocation": time.perf_counter() - started,
            "cumulative_stage_compute_seconds": float(sum(
                stage.get("elapsed_seconds", 0.0) for stage in index.data["stages"].values()
            )),
        },
        "limitations": [
            "diagnostic common-parent restriction family, not a replacement elliptic boundary-value problem",
            "original separately solved R8/R10/R12 REVIEW results remain unchanged",
            "fixed foliation and finite twelve-seed donor-capped BVP class",
            "short causal window through t=0.002",
            "not an event horizon, topology change, connected bulk throat, dark matter, mass transfer, or nonlinear stability result",
        ],
    }
    final_stage = "final/result"
    index.register(final_stage, "combined-result", 300.0, {"status": status})
    cached = index.validated_path(final_stage)
    if cached is None:
        index.mark_running(final_stage)
        atomic_write_json(OUTPUT, payload)
        index.mark_complete(final_stage, OUTPUT, 0.0)
    else:
        payload = json.loads(cached.read_text())
    print(json.dumps({
        "status": status, "classification": classification,
        "subverdicts": subverdicts, "acceptance": acceptance,
        "histories": formation["histories"],
        "maximum_domain_geometry_transfer": geometry["maxima"]["domain"],
        "maximum_tensor_relative_difference": tensors["maximum_primary_relative_difference"],
        "runtime": payload["runtime"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
