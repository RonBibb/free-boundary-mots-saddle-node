#!/usr/bin/env python3
"""Sealed fresh two-grid A=7.90 evolution and cap tracking through t=0.008."""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import run_corrected_fold_live_nonlinear_gauge_source as live
from bhps.dynamical_capped_geometry import capped_surface_geometry
from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice
from bhps.dynamical_capped_horizon_bvp import solve_dynamical_capped_surface_bvp
from bhps.gh_source_driver import (
    regular_so3_live_source_shift_advection,
    regular_so3_nonlinear_anchored_damped_wave_target,
    source_driver_rhs,
)
from bhps.nonlinear_regular_so3_evolution import (
    StageRegularGaugeSource,
    apply_outer_source_sommerfeld,
    compact_wall_normal_gauge_position_residuals,
    compact_wall_position_residuals,
    gauge_constraint_summary,
    outer_sommerfeld_position_residuals,
    regular_source_spatial_derivatives,
)
from run_corrected_A790_independent_dynamic_BVP_detector import (
    admitted as bvp_admitted,
    public_surface as public_bvp_surface,
    search_slice,
)
from run_corrected_A790_two_grid_formation_search import (
    endpoint_transfer,
    evolution_pass,
    field_transfer,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import (
    RADIAL_COMPARISON_CUT,
    relative_norm,
    signature_summary,
)


OUTPUT = Path("results/corrected_A790_t008_long_evolution.json")
STATE = Path("results/corrected_A790_t008_long_evolution_state.npz")
PROTOCOL = "notes/85_A790_t008_long_evolution_protocol.md"
NOTE74_STATE = Path("results/corrected_A790_two_grid_formation_search_state.npz")
NOTE80_RESULT = Path("results/corrected_A790_independent_dynamic_BVP_detector.json")
NOTE81_RESULT = Path("results/corrected_A790_surface_geometry_history.json")
AMPLITUDE = 7.90
FINAL_TIME = 0.008
STEPS = 64
DT = FINAL_TIME / STEPS
DIAGNOSTIC_STEPS = tuple(range(8, STEPS + 1, 8))
SURFACE_STEPS = (32, 40, 48, 56, 64)
SURFACE_TIMES = tuple(step * DT for step in SURFACE_STEPS)


def relative_difference(left, right):
    return float(abs(left - right) / max(abs(left), abs(right), 1e-300))


def classify_long_run(all_acceptance, surfaces_persist):
    if all(all_acceptance.values()):
        return "pass", "two_grid_R8_paired_surface_persistence_through_t0.008"
    if surfaces_persist:
        return "review", "two_grid_R8_persistence_candidate_with_long_evolution_review"
    return "review", "unresolved_or_lost_branch"


def motion_summary(records):
    summary = {}
    for label, history in records.items():
        branches = {}
        for branch_index, branch_name in enumerate(("inner", "outer")):
            endpoints = np.asarray([
                item["search"]["clusters"][branch_index]["signature"]
                for item in history
            ])
            areas = np.asarray([
                item["representatives"][branch_index]["geometry"][
                    "one_sided_cap_area"
                ] for item in history
            ])
            branches[branch_name] = {
                "initial_axis": float(endpoints[0, 0]),
                "final_axis": float(endpoints[-1, 0]),
                "axis_relative_change": (
                    float((endpoints[-1, 0] - endpoints[0, 0]) / endpoints[0, 0])
                ),
                "initial_brane": float(endpoints[0, 1]),
                "final_brane": float(endpoints[-1, 1]),
                "brane_relative_change": (
                    float((endpoints[-1, 1] - endpoints[0, 1]) / endpoints[0, 1])
                ),
                "initial_area": float(areas[0]),
                "final_area": float(areas[-1]),
                "area_relative_change": float((areas[-1] - areas[0]) / areas[0]),
                "axis_monotone": bool(
                    np.all(np.diff(endpoints[:, 0]) >= 0)
                    or np.all(np.diff(endpoints[:, 0]) <= 0)
                ),
                "brane_monotone": bool(
                    np.all(np.diff(endpoints[:, 1]) >= 0)
                    or np.all(np.diff(endpoints[:, 1]) <= 0)
                ),
                "area_monotone": bool(
                    np.all(np.diff(areas) >= 0) or np.all(np.diff(areas) <= 0)
                ),
            }
        summary[label] = branches
    return summary


def diagnose_state(case, current_time, state):
    position, velocity, source, memory = state
    source_z, source_r = regular_source_spatial_derivatives(
        source, case["z"], case["r"],
    )
    target = regular_so3_nonlinear_anchored_damped_wave_target(
        position, case["initial"], case["source0"], case["r"],
        live.TARGET_MU_LAPSE, live.TARGET_MU_SHIFT, live.TARGET_POWER,
    )
    advection = regular_so3_live_source_shift_advection(
        position, case["r"], source, source_z, source_r,
    )
    source_dot, memory_dot = source_driver_rhs(
        source, memory, target, live.DRIVER_MU, live.DRIVER_ETA, advection,
    )
    outer_source = None
    if case["rhs"].live_outer_sommerfeld:
        source_dot, outer_source = apply_outer_source_sommerfeld(
            source, source_dot, case["source0"], case["source_time0"],
            case["_initial_source_second_time"], position, current_time,
            case["r"], case["rhs"].stencil_width,
        )
    gauge = StageRegularGaugeSource(source, source_dot, case["z"], case["r"])
    constraint = gauge_constraint_summary(
        position, velocity, current_time, case["rhs"],
        RADIAL_COMPARISON_CUT, gauge,
    )
    wall = compact_wall_position_residuals(
        position, case["z"], case["r"], case["geometry"]["background"],
    )
    normal_wall = compact_wall_normal_gauge_position_residuals(
        position, source, case["z"], case["r"],
        case["geometry"]["background"],
    )
    outer_position = outer_sommerfeld_position_residuals(
        position, velocity, case["rhs"].outer_reference_position,
        case["rhs"].outer_reference_acceleration, current_time, case["r"],
        case["rhs"].stencil_width,
    )
    return {
        "time": float(current_time),
        "finite": bool(
            all(np.all(np.isfinite(value)) for value in state)
            and np.all(np.isfinite(source_dot))
            and np.all(np.isfinite(memory_dot))
        ),
        "Lorentzian": signature_summary(position, case["r"])[
            "all_points_one_negative_direction"
        ],
        "global_GH_constraint": constraint["global_relative"],
        "wall_position_residual": wall["maximum"],
        "normal_wall_position_residual": normal_wall["maximum"],
        "outer_position_residual": outer_position["maximum_normalized"],
        "outer_source_residual": outer_source["maximum_normalized"],
        "_constraint": constraint,
        "_wall": wall,
        "_normal_wall": normal_wall,
        "_outer_position": outer_position,
        "_outer_source": outer_source,
    }


def integrate_dense(case):
    state = (
        case["initial"].copy(), np.zeros_like(case["initial"]),
        case["source0"].copy(), case["memory0"].copy(),
    )
    current_time = 0.0
    all_finite = True
    maxima = {
        "maximum_normal_wall_acceleration_residual": 0.0,
        "maximum_outer_acceleration_residual": 0.0,
        "maximum_outer_source_residual": 0.0,
        "maximum_outer_metric_correction": 0.0,
        "maximum_outer_scalar_correction": 0.0,
        "maximum_outer_source_correction": 0.0,
    }
    position_history = [state[0].copy()]
    velocity_history = [state[1].copy()]
    checkpoint_states = {}
    diagnostics = []
    source_diagnostic_history = []
    for step in range(1, STEPS + 1):
        print(f"{case['label']}: long step {step}/{STEPS}, stage 1", flush=True)
        k1, d1 = live.driver_stage(case, current_time, *state)
        midpoint = tuple(
            value + 0.5 * DT * slope for value, slope in zip(state, k1)
        )
        print(f"{case['label']}: long step {step}/{STEPS}, stage 2", flush=True)
        k2, d2 = live.driver_stage(case, current_time + 0.5 * DT, *midpoint)
        all_finite = bool(all_finite and d1["finite"] and d2["finite"])
        for diagnostic in (d1, d2):
            normal = diagnostic["normal_wall_gauge"]
            if normal is not None:
                maxima["maximum_normal_wall_acceleration_residual"] = max(
                    maxima["maximum_normal_wall_acceleration_residual"],
                    normal["final_residual"]["maximum"],
                )
            outer = diagnostic["outer_sommerfeld"]
            if outer is not None:
                maxima["maximum_outer_acceleration_residual"] = max(
                    maxima["maximum_outer_acceleration_residual"],
                    outer["maximum_normalized_acceleration_residual"],
                )
                maxima["maximum_outer_metric_correction"] = max(
                    maxima["maximum_outer_metric_correction"],
                    outer["metric_relative_correction"],
                )
                maxima["maximum_outer_scalar_correction"] = max(
                    maxima["maximum_outer_scalar_correction"],
                    outer["scalar_relative_correction"],
                )
            outer_source = diagnostic["outer_source_sommerfeld"]
            if outer_source is not None:
                maxima["maximum_outer_source_residual"] = max(
                    maxima["maximum_outer_source_residual"],
                    outer_source["maximum_normalized"],
                )
                maxima["maximum_outer_source_correction"] = max(
                    maxima["maximum_outer_source_correction"],
                    outer_source["relative_correction"],
                )
        state = tuple(value + DT * slope for value, slope in zip(state, k2))
        current_time += DT
        position_history.append(state[0].copy())
        velocity_history.append(state[1].copy())
        if step in SURFACE_STEPS:
            checkpoint_states[step] = {
                "_position": state[0].copy(),
                "_velocity": state[1].copy(),
                "_increment": state[0] - case["initial"],
                "_source_increment": state[2] - case["source0"],
            }
        if step in DIAGNOSTIC_STEPS:
            print(
                f"{case['label']}: exact diagnostics at t={current_time:.6f}",
                flush=True,
            )
            diagnostics.append(diagnose_state(case, current_time, state))
            source_diagnostic_history.append((state[2] - case["source0"]).copy())

    final = diagnostics[-1]
    return {
        "final_time": current_time,
        "steps": STEPS,
        "time_step": DT,
        "all_stages_finite": all_finite,
        "signature": {"all_points_one_negative_direction": final["Lorentzian"]},
        "final_constraint": final["_constraint"],
        "final_wall": final["_wall"],
        "final_normal_wall_position_residual": final["_normal_wall"],
        "final_outer_sommerfeld_position_residual": final["_outer_position"],
        "final_outer_source_sommerfeld_residual": final["_outer_source"],
        **maxima,
        "checkpoint_diagnostics": [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in diagnostics
        ],
        "_position": state[0],
        "_increment": state[0] - case["initial"],
        "_velocity": state[1],
        "_source_increment": state[2] - case["source0"],
        "_checkpoints": checkpoint_states,
        "_position_history": np.stack(position_history),
        "_velocity_history": np.stack(velocity_history),
        "_source_diagnostic_history": np.stack(source_diagnostic_history),
    }


def representative_geometry(position, velocity, geometry, cluster):
    member = cluster["members"][len(cluster["members"]) // 2]
    seed = member["seed"]
    prepared = prepare_capped_expansion_slice(
        position, velocity, geometry["z"], geometry["r"],
    )
    started = time.perf_counter()
    try:
        surface = solve_dynamical_capped_surface_bvp(
            position, velocity, geometry["z"], geometry["r"], seed,
            tolerance=2e-5, nodes=121, maximum_nodes=6000,
            dense_nodes=501, prepared=prepared,
        )
        surface["runtime_seconds"] = time.perf_counter() - started
        geometry_record = (
            capped_surface_geometry(
                position, velocity, geometry["z"], geometry["r"], surface,
                prepared=prepared,
            ) if bvp_admitted(surface) else None
        )
        return {
            "seed": seed,
            "admitted": bvp_admitted(surface),
            "surface": public_bvp_surface(surface),
            "geometry": geometry_record,
        }
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        return {
            "seed": seed,
            "admitted": False,
            "error": f"{type(error).__name__}: {error}",
            "geometry": None,
        }


def surface_history(label, geometry, run):
    records = []
    for step, current_time in zip(SURFACE_STEPS, SURFACE_TIMES):
        state = run["_checkpoints"][step]
        print(f"{label}: independent BVP search at t={current_time:.6f}", flush=True)
        search = search_slice(
            f"{label}-A790-long-t{current_time:.6f}",
            state["_position"], state["_velocity"], geometry,
        )
        representatives = [
            representative_geometry(
                state["_position"], state["_velocity"], geometry, cluster,
            ) for cluster in search["clusters"]
        ]
        records.append({
            "time": current_time,
            "admitted_distinct_count": search["admitted_distinct_count"],
            "admitted_signatures": search["admitted_signatures"],
            "search": search,
            "representatives": representatives,
        })
    return records


def note81_area_reference(note81, label):
    record = next(
        item for item in note81["records"][label]
        if np.isclose(item["time"], 0.004)
    )
    return [
        branch["geometry"][1]["one_sided_cap_area"]
        for branch in record["branches"]
    ]


def public_evolution(run):
    return {
        "final_time": run["final_time"],
        "steps": run["steps"],
        "time_step": run["time_step"],
        "all_stages_finite": run["all_stages_finite"],
        "maximum_normal_wall_acceleration_residual": run[
            "maximum_normal_wall_acceleration_residual"
        ],
        "maximum_outer_acceleration_residual": run[
            "maximum_outer_acceleration_residual"
        ],
        "maximum_outer_source_residual": run["maximum_outer_source_residual"],
        "maximum_outer_metric_correction": run["maximum_outer_metric_correction"],
        "maximum_outer_scalar_correction": run["maximum_outer_scalar_correction"],
        "maximum_outer_source_correction": run["maximum_outer_source_correction"],
        "checkpoint_diagnostics": run["checkpoint_diagnostics"],
    }


def main():
    required = (NOTE74_STATE, NOTE80_RESULT, NOTE81_RESULT)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required sealed inputs missing: {missing}")
    overall_started = time.perf_counter()
    note74 = np.load(NOTE74_STATE)
    note80 = json.loads(NOTE80_RESULT.read_text())
    note81 = json.loads(NOTE81_RESULT.read_text())

    print("building fresh corrected G7/G8 A=7.90 Rmax=8 geometries", flush=True)
    build_started = time.perf_counter()
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": AMPLITUDE}
    geometries = {
        "G7": build_refined(
            seed, 81, 121, "G7A790-t008-long",
            selector_iterations=40, slice_iterations=270,
        ),
    }
    geometries["G8"] = build_refined(
        geometries["G7"], 97, 145, "G8A790-t008-long",
        selector_iterations=45, slice_iterations=280,
    )
    build_seconds = time.perf_counter() - build_started

    cases = {
        label: live.setup_case(
            geometry, f"{label}-A790-t008-long",
            live_normal_wall_gauge=True, live_outer_sommerfeld=True,
        ) for label, geometry in geometries.items()
    }
    runs = {}
    evolution_seconds = {}
    for label in ("G7", "G8"):
        started = time.perf_counter()
        runs[label] = integrate_dense(cases[label])
        evolution_seconds[label] = time.perf_counter() - started

    surface_started = time.perf_counter()
    surfaces = {
        label: surface_history(label, geometries[label], runs[label])
        for label in ("G7", "G8")
    }
    surface_seconds = time.perf_counter() - surface_started

    t004_anchor = {label: {} for label in ("G7", "G8")}
    for label in ("G7", "G8"):
        fresh = runs[label]["_checkpoints"][32]
        t004_anchor[label]["note74_field_relative_difference"] = {
            "position_increment": relative_norm(
                fresh["_increment"], note74[f"{label}_time_3_increment"],
            ),
            "velocity": relative_norm(
                fresh["_velocity"], note74[f"{label}_time_3_velocity"],
            ),
        }
        t004_anchor[label]["note80_endpoint_transfer"] = endpoint_transfer(
            surfaces[label][0]["admitted_signatures"],
            note80["searches"][label]["0.004"]["admitted_signatures"],
        )
        reference_areas = note81_area_reference(note81, label)
        current_areas = [
            item["geometry"]["one_sided_cap_area"]
            for item in surfaces[label][0]["representatives"]
            if item["geometry"] is not None
        ]
        t004_anchor[label]["note81_area_relative_differences"] = (
            [
                relative_difference(current, reference)
                for current, reference in zip(current_areas, reference_areas)
            ] if len(current_areas) == len(reference_areas) == 2 else None
        )

    cross_grid_surfaces = []
    for index, current_time in enumerate(SURFACE_TIMES):
        endpoint = endpoint_transfer(
            surfaces["G7"][index]["admitted_signatures"],
            surfaces["G8"][index]["admitted_signatures"],
        )
        area = []
        for branch_index, name in enumerate(("inner", "outer")):
            try:
                left = surfaces["G7"][index]["representatives"][branch_index][
                    "geometry"
                ]["one_sided_cap_area"]
                right = surfaces["G8"][index]["representatives"][branch_index][
                    "geometry"
                ]["one_sided_cap_area"]
                area.append({
                    "branch": name,
                    "relative_difference": relative_difference(left, right),
                })
            except (IndexError, TypeError):
                area.append({"branch": name, "relative_difference": None})
        cross_grid_surfaces.append({
            "time": current_time,
            "endpoint_transfer": endpoint,
            "area_transfer": area,
        })

    final_field_transfer = {
        name: field_transfer(
            cases["G7"], runs["G7"], cases["G8"], runs["G8"], key,
        ) for name, key in (
            ("position_increment", "_increment"),
            ("velocity", "_velocity"),
            ("source_increment", "_source_increment"),
        )
    }

    diagnostics_pass = bool(all(
        record["finite"] and record["Lorentzian"]
        and record["global_GH_constraint"] < 0.005
        and record["wall_position_residual"] < 0.0005
        and record["normal_wall_position_residual"] < 0.0005
        and record["outer_position_residual"] < 1e-10
        and record["outer_source_residual"] < 1e-10
        for run in runs.values() for record in run["checkpoint_diagnostics"]
    ))
    all_searches = [item for grid in surfaces.values() for item in grid]
    surfaces_persist = bool(all(
        item["admitted_distinct_count"] == 2
        and len(item["search"]["clusters"]) == 2
        and all(len(cluster["members"]) >= 2 for cluster in item["search"]["clusters"])
        for item in all_searches
    ))
    admitted_surface_rules = bool(
        surfaces_persist and all(
            trial["surface"]["local_expansion_interior_maximum"] < 2e-4
            and trial["surface"]["boundary_slope_error"] < 2e-4
            and trial["surface"]["primary_evaluator_crosscheck"][
                "two_cell_interior_maximum"
            ] < 0.002
            for item in all_searches for trial in item["search"]["trials"]
            if trial["admitted"]
        )
        and all(
            representative["admitted"]
            for item in all_searches for representative in item["representatives"]
        )
    )
    all_representatives = [
        representative for item in all_searches
        for representative in item["representatives"]
    ]
    geometry_pass = bool(
        len(all_representatives) == 2 * len(all_searches)
        and all(
            item["geometry"] is not None and item["geometry"]["finite"]
            and item["geometry"]["one_sided_cap_area"] > 0.0
            for item in all_representatives
        )
        and all(
            item["representatives"][1]["geometry"]["one_sided_cap_area"]
            > item["representatives"][0]["geometry"]["one_sided_cap_area"]
            for item in all_searches
        )
    )
    archive_times = np.arange(STEPS + 1, dtype=float) * DT
    archive_pass = bool(
        len(archive_times) == 65
        and np.allclose(np.diff(archive_times), DT, rtol=0.0, atol=1e-15)
        and all(
            run["_position_history"].shape[0] == 65
            and run["_velocity_history"].shape[0] == 65
            and run["_source_diagnostic_history"].shape[0] == 8
            and np.all(np.isfinite(run["_position_history"]))
            and np.all(np.isfinite(run["_velocity_history"]))
            and np.all(np.isfinite(run["_source_diagnostic_history"]))
            for run in runs.values()
        )
    )
    acceptance = {
        "all_evolution_constraint_boundary_and_signature_rules_pass": bool(
            all(geometry["selector_maximum"] < 1e-8 for geometry in geometries.values())
            and diagnostics_pass
            and all(evolution_pass(run) for run in runs.values())
        ),
        "fine_t004_run_anchors_to_notes74_80_81": bool(all(
            max(item["note74_field_relative_difference"].values()) < 0.05
            and item["note80_endpoint_transfer"] is not None
            and item["note80_endpoint_transfer"]["maximum"] < 0.01
            and item["note81_area_relative_differences"] is not None
            and max(item["note81_area_relative_differences"]) < 0.01
            for item in t004_anchor.values()
        )),
        "two_multiseed_BVP_branches_persist_and_pass_all_detector_rules": (
            admitted_surface_rules
        ),
        "representative_areas_are_positive_and_outer_exceeds_inner": geometry_pass,
        "all_cross_grid_surface_and_final_field_transfers_pass": bool(
            all(
                item["endpoint_transfer"] is not None
                and item["endpoint_transfer"]["maximum"] < 0.01
                and all(
                    branch["relative_difference"] is not None
                    and branch["relative_difference"] < 0.01
                    for branch in item["area_transfer"]
                ) for item in cross_grid_surfaces
            )
            and max(final_field_transfer.values()) < 0.05
        ),
        "dense_state_archive_is_complete_and_finite": archive_pass,
    }
    status, classification = classify_long_run(acceptance, surfaces_persist)
    motion = motion_summary(surfaces) if geometry_pass else None
    runtime = {
        "geometry_build_seconds": build_seconds,
        "evolution_seconds": evolution_seconds,
        "surface_tracking_seconds": surface_seconds,
        "total_seconds": time.perf_counter() - overall_started,
    }
    payload = {
        "status": status,
        "classification": classification,
        "scope": "sealed fresh two-grid Rmax=8 A=7.90 paired-surface persistence evolution through t=0.008",
        "protocol": PROTOCOL,
        "amplitude": AMPLITUDE,
        "domain": {"r_max": 8.0, "note82_timing_sensitivity_preserved": True},
        "time_step": DT,
        "steps": STEPS,
        "diagnostic_times": [step * DT for step in DIAGNOSTIC_STEPS],
        "surface_times": list(SURFACE_TIMES),
        "evolution": {label: public_evolution(run) for label, run in runs.items()},
        "surface_history": surfaces,
        "t004_anchor": t004_anchor,
        "cross_grid_surface_transfer": cross_grid_surfaces,
        "final_field_transfer": final_field_transfer,
        "motion_t004_to_t008": motion,
        "acceptance": acceptance,
        "runtime": runtime,
        "archive": {
            "path": str(STATE),
            "time_levels": 65,
            "metric_velocity_cadence": DT,
            "source_diagnostic_levels": 8,
        },
        "limitations": [
            "Rmax=8 first-stage long evolution; note82 formation-time domain sensitivity remains unresolved",
            "one fine time step and two spatial grids; no long-interval temporal-convergence sequence",
            "finite twelve-seed star-shaped donor-capped BVP search",
            "apparent horizons are foliation dependent",
            "not stability, event-horizon reconstruction, domain-robust long-time behavior, continuum topology change, connected bulk geometry, quasi-local mass transfer, or dark-matter halo evidence",
        ],
    }
    state_values = {"times": archive_times}
    for label in ("G7", "G8"):
        state_values[f"{label}_z"] = geometries[label]["z"]
        state_values[f"{label}_r"] = geometries[label]["r"]
        state_values[f"{label}_position_history"] = runs[label]["_position_history"]
        state_values[f"{label}_velocity_history"] = runs[label]["_velocity_history"]
        state_values[f"{label}_source_diagnostic_history"] = runs[label][
            "_source_diagnostic_history"
        ]
    np.savez_compressed(STATE, **state_values)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": status,
        "classification": classification,
        "acceptance": acceptance,
        "final_field_transfer": final_field_transfer,
        "cross_grid_surface_transfer": cross_grid_surfaces,
        "motion_t004_to_t008": motion,
        "runtime": runtime,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
