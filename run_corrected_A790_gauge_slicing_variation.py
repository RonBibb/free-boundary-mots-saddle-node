#!/usr/bin/env python3
"""Sealed two-gauge, two-grid A=7.90 paired-MOTS robustness audit."""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import run_corrected_fold_live_nonlinear_gauge_source as live
from bhps.corrected_A790_gauge_variations import (
    BASELINE,
    VARIATIONS,
    brackets_baseline,
    configure_live_module,
)
from bhps.dynamical_capped_geometry import capped_surface_geometry
from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice
from bhps.dynamical_capped_horizon_bvp import solve_dynamical_capped_surface_bvp
from run_corrected_A790_independent_dynamic_BVP_detector import (
    admitted,
    analytic_controls,
    public_surface,
    search_slice,
)
from run_corrected_A790_two_grid_formation_search import (
    endpoint_transfer,
    evolution_pass,
    field_transfer,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import relative_norm


OUTPUT = Path("results/corrected_A790_gauge_slicing_variation.json")
STATE = Path("results/corrected_A790_gauge_slicing_variation_state.npz")
PROTOCOL = "notes/86_A790_gauge_slicing_variation_protocol.md"
BASELINE_BVP = Path("results/corrected_A790_independent_dynamic_BVP_detector.json")
BASELINE_GEOMETRY = Path("results/corrected_A790_surface_geometry_history.json")
AMPLITUDE = 7.90
FINE_FINAL = 0.001
FINE_STEPS = 8
FINE_TIMES = tuple((index + 1) * FINE_FINAL / FINE_STEPS for index in range(FINE_STEPS))
LONG_FINAL = 0.004
LONG_STEPS = 8
LONG_CHECKPOINT_STEPS = (4, 6)
LONG_TIMES = (0.002, 0.003, 0.004)
BASELINE_TIMES = (0.000625, 0.001, 0.004)


def working_paths(name):
    stem = Path(f"results/corrected_A790_gauge_slicing_variation_{name}_working")
    return stem.with_suffix(".json"), stem.with_suffix(".npz")


def scalar_relative(left, right):
    return float(abs(left - right) / max(abs(left), abs(right), 1e-300))


def public_diagnostics(run):
    return {
        "final_time": run["final_time"],
        "steps": run["steps"],
        "time_step": run["time_step"],
        "all_stages_finite": run["all_stages_finite"],
        "Lorentzian": run["signature"]["all_points_one_negative_direction"],
        "global_GH_constraint": run["final_constraint"]["global_relative"],
        "wall_position_residual": run["final_wall"]["maximum"],
        "normal_wall_position_residual": run[
            "final_normal_wall_position_residual"
        ]["maximum"],
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
        "final_outer_position_residual": run[
            "final_outer_sommerfeld_position_residual"
        ]["maximum_normalized"],
        "final_outer_source_residual": run[
            "final_outer_source_sommerfeld_residual"
        ]["maximum_normalized"],
        "final_source_increment_norm": run["final_source_increment_norm"],
        "final_source_target_relative_difference": run[
            "final_source_target_relative_difference"
        ],
    }


def public_initial(case):
    return {
        "initial_target_anchoring_relative": case[
            "initial_target_anchoring_relative"
        ],
        "initial_source_time_relative_error": case[
            "initial_source_time_relative_error"
        ],
        "initial_live_Taylor_acceleration_relative_difference": case[
            "initial_live_Taylor_acceleration_relative_difference"
        ],
        "initial_constraint_global_relative": case[
            "initial_constraint"
        ]["global_relative"],
        "initial_live_normal_wall_gauge_enabled": bool(
            case["initial_live_normal_wall_gauge"] is not None
        ),
        "initial_live_outer_sommerfeld_enabled": bool(
            case["initial_live_outer_sommerfeld"] is not None
        ),
    }


def integrate_pair(geometry, label):
    case = live.setup_case(
        geometry, label, live_normal_wall_gauge=True,
        live_outer_sommerfeld=True,
    )
    live.FINAL_TIME = FINE_FINAL
    live.STEPS = FINE_STEPS
    fine = live.integrate(case, checkpoint_steps=range(1, FINE_STEPS))
    live.FINAL_TIME = LONG_FINAL
    live.STEPS = LONG_STEPS
    long = live.integrate(case, checkpoint_steps=LONG_CHECKPOINT_STEPS)
    return case, fine, long


def fine_states(run):
    return [
        run["_checkpoints"][index + 1] if index + 1 < FINE_STEPS else run
        for index in range(FINE_STEPS)
    ]


def long_states(run):
    return [
        run["_checkpoints"][4], run["_checkpoints"][6], run,
    ]


def selected_geometry(position, velocity, geometry, search):
    prepared = prepare_capped_expansion_slice(
        position, velocity, geometry["z"], geometry["r"],
    )
    records = []
    for branch, cluster in zip(
        ("inner", "outer"),
        sorted(search["clusters"], key=lambda item: item["signature"][1]),
    ):
        members = sorted(cluster["members"], key=lambda item: item["seed"])
        seed = float(members[len(members) // 2]["seed"])
        started = time.perf_counter()
        solved = solve_dynamical_capped_surface_bvp(
            position, velocity, geometry["z"], geometry["r"], seed,
            tolerance=2e-5, nodes=121, maximum_nodes=6000,
            dense_nodes=501,
        )
        solved["runtime_seconds"] = float(time.perf_counter() - started)
        records.append({
            "branch": branch,
            "selected_seed": seed,
            "admitted": admitted(solved),
            "surface": public_surface(solved),
            "geometry": capped_surface_geometry(
                position, velocity, geometry["z"], geometry["r"], solved,
                prepared=prepared,
            ),
        })
    return records


def save_working(name, variation, cases, runs):
    """Checkpoint costly evolutions before any detector/reporting work."""
    meta_path, state_path = working_paths(name)
    arrays = {}
    public_runs = {}
    for label in cases:
        public_runs[label] = {}
        for family, run in runs[label].items():
            public_runs[label][family] = live.public_run(run)
            arrays[f"{label}_{family}_increment"] = run["_increment"]
            arrays[f"{label}_{family}_velocity"] = run["_velocity"]
            arrays[f"{label}_{family}_source_increment"] = run["_source_increment"]
            for step, checkpoint in run["_checkpoints"].items():
                arrays[f"{label}_{family}_step{step}_increment"] = checkpoint["_increment"]
                arrays[f"{label}_{family}_step{step}_velocity"] = checkpoint["_velocity"]
                arrays[f"{label}_{family}_step{step}_source_increment"] = checkpoint[
                    "_source_increment"
                ]
    np.savez_compressed(state_path, **arrays)
    meta_path.write_text(json.dumps({
        "variation": variation.public(), "runs": public_runs,
    }, indent=2, sort_keys=True) + "\n")


def load_working(name, variation, cases):
    """Restore a completed variation's evolution arrays and diagnostics."""
    meta_path, state_path = working_paths(name)
    if not (meta_path.exists() and state_path.exists()):
        return None
    meta = json.loads(meta_path.read_text())
    if meta.get("variation") != variation.public():
        raise RuntimeError(f"working checkpoint parameter mismatch for {name}")
    archive = np.load(state_path)
    runs = {}
    checkpoint_map = {"fine": range(1, FINE_STEPS), "long": LONG_CHECKPOINT_STEPS}
    for label, case in cases.items():
        runs[label] = {}
        for family in ("fine", "long"):
            run = dict(meta["runs"][label][family])
            run["_increment"] = archive[f"{label}_{family}_increment"]
            run["_position"] = case["initial"] + run["_increment"]
            run["_velocity"] = archive[f"{label}_{family}_velocity"]
            run["_source_increment"] = archive[
                f"{label}_{family}_source_increment"
            ]
            checkpoints = {}
            for step in checkpoint_map[family]:
                increment = archive[f"{label}_{family}_step{step}_increment"]
                checkpoints[int(step)] = {
                    "time": float(step * run["time_step"]),
                    "_increment": increment,
                    "_position": case["initial"] + increment,
                    "_velocity": archive[f"{label}_{family}_step{step}_velocity"],
                    "_source_increment": archive[
                        f"{label}_{family}_step{step}_source_increment"
                    ],
                }
            run["_checkpoints"] = checkpoints
            runs[label][family] = run
    return runs


def search_history(prefix, geometry, times, states):
    records = []
    for time_value, state in zip(times, states):
        label = f"{prefix}-t{time_value:.6f}"
        result = search_slice(
            label, state["_position"], state["_velocity"], geometry,
        )
        geometries = (
            selected_geometry(
                state["_position"], state["_velocity"], geometry, result,
            ) if result["admitted_distinct_count"] == 2 else []
        )
        records.append({
            "time": float(time_value),
            "admitted_distinct_count": result["admitted_distinct_count"],
            "admitted_signatures": result["admitted_signatures"],
            "branches": geometries,
            "search": result,
        })
    return records


def paired_history(record):
    counts = [item["admitted_distinct_count"] for item in record]
    nonzero = [index for index, count in enumerate(counts) if count]
    if not nonzero:
        return False, None
    first = nonzero[0]
    passed = bool(
        first > 0
        and all(count == 0 for count in counts[:first])
        and all(count == 2 for count in counts[first:])
    )
    return passed, first


def all_searches_resolved(records):
    for record in records:
        if record["admitted_distinct_count"] == 0:
            continue
        if record["admitted_distinct_count"] != 2:
            return False
        if any(len(cluster["members"]) < 2 for cluster in record["search"]["clusters"]):
            return False
        if len(record["branches"]) != 2 or not all(
            branch["admitted"] for branch in record["branches"]
        ):
            return False
    return True


def cross_grid_surface_transfer(left, right):
    endpoint = endpoint_transfer(
        left["admitted_signatures"], right["admitted_signatures"],
    )
    if len(left["branches"]) != 2 or len(right["branches"]) != 2:
        return {"endpoint": endpoint, "branches": []}
    branches = []
    for coarse, fine in zip(left["branches"], right["branches"]):
        branches.append({
            "branch": coarse["branch"],
            "proper_area_relative_difference": scalar_relative(
                coarse["geometry"]["one_sided_cap_area"],
                fine["geometry"]["one_sided_cap_area"],
            ),
            "equivalent_radius_relative_difference": scalar_relative(
                coarse["geometry"]["equivalent_area_radius"],
                fine["geometry"]["equivalent_area_radius"],
            ),
            "proper_meridional_length_relative_difference": scalar_relative(
                coarse["geometry"]["proper_meridional_length"],
                fine["geometry"]["proper_meridional_length"],
            ),
        })
    return {"endpoint": endpoint, "branches": branches}


def baseline_references():
    bvp = json.loads(BASELINE_BVP.read_text())
    geometry = json.loads(BASELINE_GEOMETRY.read_text())
    result = {label: {} for label in ("G7", "G8")}
    for label in result:
        geometry_by_time = {
            round(item["time"], 9): item for item in geometry["records"][label]
        }
        for time_value in BASELINE_TIMES:
            key = f"{time_value:g}"
            area_record = geometry_by_time[round(time_value, 9)]
            result[label][key] = {
                "signatures": bvp["searches"][label][key]["admitted_signatures"],
                "branches": [
                    branch["geometry"][1] for branch in area_record["branches"]
                ],
            }
    return result


def baseline_comparison(record, reference):
    endpoint = endpoint_transfer(
        record["admitted_signatures"], reference["signatures"],
    )
    branches = []
    if len(record["branches"]) == len(reference["branches"]) == 2:
        for found, baseline in zip(record["branches"], reference["branches"]):
            branches.append({
                "branch": found["branch"],
                "proper_area_relative_difference": scalar_relative(
                    found["geometry"]["one_sided_cap_area"],
                    baseline["one_sided_cap_area"],
                ),
                "equivalent_radius_relative_difference": scalar_relative(
                    found["geometry"]["equivalent_area_radius"],
                    baseline["equivalent_area_radius"],
                ),
            })
    return {
        "time": record["time"], "coordinate_endpoint_difference": endpoint,
        "proper_geometry_difference": branches,
        "interpretation": (
            "descriptive only: equal coordinate-time labels need not be the same physical slice"
        ),
    }


def record_at(records, time_value):
    return next(item for item in records if np.isclose(item["time"], time_value))


def state_arrays(data, geometries):
    values = {
        "fine_times": np.asarray(FINE_TIMES),
        "long_times": np.asarray(LONG_TIMES),
    }
    for label, geometry in geometries.items():
        values[f"{label}_z"] = geometry["z"]
        values[f"{label}_r"] = geometry["r"]
    for variation, variation_data in data.items():
        for label, runs in variation_data["runs"].items():
            for family, times, states in (
                ("fine", FINE_TIMES, fine_states(runs["fine"])),
                ("long", LONG_TIMES, long_states(runs["long"])),
            ):
                for index, (time_value, state) in enumerate(zip(times, states)):
                    del time_value
                    for name, key in (
                        ("increment", "_increment"),
                        ("velocity", "_velocity"),
                        ("source_increment", "_source_increment"),
                    ):
                        values[f"{variation}_{label}_{family}_{index}_{name}"] = state[key]
    return values


def main():
    required = (BASELINE_BVP, BASELINE_GEOMETRY, Path(PROTOCOL))
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing sealed inputs: {missing}")
    started = time.perf_counter()
    controls = analytic_controls()
    print("reconstructing fixed corrected G7/G8 A=7.90 data", flush=True)
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": AMPLITUDE}
    geometries = {
        "G7": build_refined(
            seed, 81, 121, "G7A790-gauge-audit",
            selector_iterations=40, slice_iterations=270,
        ),
    }
    geometries["G8"] = build_refined(
        geometries["G7"], 97, 145, "G8A790-gauge-audit",
        selector_iterations=45, slice_iterations=280,
    )

    initial_search = {}
    for label, geometry in geometries.items():
        initial = np.asarray(geometry["jet_field"].reduced_fields)
        initial_search[label] = search_slice(
            f"{label}-A790-gauge-audit-t0", initial,
            np.zeros_like(initial), geometry,
        )

    data = {}
    for variation in VARIATIONS:
        configure_live_module(live, variation)
        print(f"starting sealed gauge variation: {variation.name}", flush=True)
        cases = {}
        for label, geometry in geometries.items():
            cases[label] = live.setup_case(
                geometry, f"{variation.name}-{label}-A790",
                live_normal_wall_gauge=True, live_outer_sommerfeld=True,
            )
        runs = load_working(variation.name, variation, cases)
        if runs is None:
            runs = {}
            for label, case in cases.items():
                print(f"{variation.name} {label}: fine and long evolutions", flush=True)
                live.FINAL_TIME = FINE_FINAL
                live.STEPS = FINE_STEPS
                fine = live.integrate(
                    case, checkpoint_steps=range(1, FINE_STEPS),
                )
                live.FINAL_TIME = LONG_FINAL
                live.STEPS = LONG_STEPS
                long = live.integrate(
                    case, checkpoint_steps=LONG_CHECKPOINT_STEPS,
                )
                runs[label] = {"fine": fine, "long": long}
            save_working(variation.name, variation, cases, runs)
        else:
            print(f"restored {variation.name} completed evolutions", flush=True)
        searches = {label: {} for label in geometries}
        for label, geometry in geometries.items():
            searches[label]["fine"] = search_history(
                f"{variation.name}-{label}-fine", geometry, FINE_TIMES,
                fine_states(runs[label]["fine"]),
            )
            searches[label]["long"] = search_history(
                f"{variation.name}-{label}-long", geometry, LONG_TIMES,
                long_states(runs[label]["long"]),
            )
        data[variation.name] = {
            "parameters": variation.public(), "cases": cases,
            "runs": runs, "searches": searches,
        }

    initial_identity = {}
    slow_cases = data[VARIATIONS[0].name]["cases"]
    fast_cases = data[VARIATIONS[1].name]["cases"]
    for label in geometries:
        initial_identity[label] = {
            "metric_state_relative_difference": relative_norm(
                slow_cases[label]["initial"], fast_cases[label]["initial"],
            ),
            "zero_velocity_relative_difference": 0.0,
            "source_relative_difference": relative_norm(
                slow_cases[label]["source0"], fast_cases[label]["source0"],
            ),
            "source_time_relative_difference": relative_norm(
                slow_cases[label]["source_time0"], fast_cases[label]["source_time0"],
            ),
            "memory_relative_difference": relative_norm(
                slow_cases[label]["memory0"], fast_cases[label]["memory0"],
            ),
        }

    cross_grid = {}
    field_transfers = {}
    formation = {}
    for variation in VARIATIONS:
        name = variation.name
        searches = data[name]["searches"]
        cross_grid[name] = {"fine": [], "long": []}
        for family, times in (("fine", FINE_TIMES), ("long", LONG_TIMES)):
            for index, time_value in enumerate(times):
                left = searches["G7"][family][index]
                right = searches["G8"][family][index]
                transfer = (
                    cross_grid_surface_transfer(left, right)
                    if left["admitted_distinct_count"] == right["admitted_distinct_count"] == 2
                    else None
                )
                cross_grid[name][family].append({
                    "time": float(time_value), "transfer": transfer,
                })
        runs = data[name]["runs"]
        cases = data[name]["cases"]
        field_transfers[name] = {"t0.001": {}, "t0.004": {}}
        for field, key in (
            ("position_increment", "_increment"),
            ("velocity", "_velocity"),
            ("source_increment", "_source_increment"),
        ):
            field_transfers[name]["t0.001"][field] = field_transfer(
                cases["G7"], runs["G7"]["fine"], cases["G8"],
                runs["G8"]["fine"], key,
            )
            field_transfers[name]["t0.004"][field] = field_transfer(
                cases["G7"], runs["G7"]["long"], cases["G8"],
                runs["G8"]["long"], key,
            )
        histories = {}
        first_indices = {}
        for label in geometries:
            fine = searches[label]["fine"]
            passed, first = paired_history(fine)
            histories[label] = {
                "counts": [item["admitted_distinct_count"] for item in fine],
                "paired_transition_pass": passed,
                "first_detection_index": first,
                "first_detection_time": FINE_TIMES[first] if first is not None else None,
                "last_zero_time": FINE_TIMES[first - 1] if first else None,
            }
            first_indices[label] = first
        indices_valid = all(value is not None for value in first_indices.values())
        bracket = None
        if indices_valid:
            lower = min(FINE_TIMES[value - 1] for value in first_indices.values())
            upper = max(FINE_TIMES[value] for value in first_indices.values())
            bracket = {"lower": lower, "upper": upper, "width": upper - lower}
        formation[name] = {
            "by_grid": histories,
            "first_detection_indices_differ_by_at_most_one": bool(
                indices_valid
                and abs(first_indices["G7"] - first_indices["G8"]) <= 1
            ),
            "two_grid_envelope_bracket": bracket,
        }

    references = baseline_references()
    baseline_comparisons = {variation.name: {} for variation in VARIATIONS}
    for variation in VARIATIONS:
        name = variation.name
        for label in geometries:
            records = (
                data[name]["searches"][label]["fine"]
                + data[name]["searches"][label]["long"]
            )
            baseline_comparisons[name][label] = []
            for time_value in BASELINE_TIMES:
                record = record_at(records, time_value)
                baseline_comparisons[name][label].append(
                    baseline_comparison(
                        record, references[label][f"{time_value:g}"],
                    )
                )

    all_cases = [
        case for variation in data.values() for case in variation["cases"].values()
    ]
    all_runs = [
        run for variation in data.values() for grid in variation["runs"].values()
        for run in grid.values()
    ]
    all_search_records = [
        record for variation in data.values()
        for grid in variation["searches"].values()
        for family in grid.values() for record in family
    ]
    all_branches = [
        branch for record in all_search_records for branch in record["branches"]
    ]
    all_grid_transfers = [
        item["transfer"] for variation in cross_grid.values()
        for family in variation.values() for item in family
        if item["transfer"] is not None
    ]
    identity_max = max(
        value for record in initial_identity.values()
        for key, value in record.items() if key != "memory_relative_difference"
    )
    initial_gate = bool(
        controls["passed"] and identity_max < 1e-13
        and max(
            case["initial_target_anchoring_relative"] for case in all_cases
        ) < 1e-10
        and max(
            case["initial_source_time_relative_error"] for case in all_cases
        ) < 1e-10
        and all(
            record["admitted_distinct_count"] == 0
            for record in initial_search.values()
        )
    )
    evolution_gate = bool(
        all(evolution_pass(run) for run in all_runs)
        and max(
            value for variation in field_transfers.values()
            for time_record in variation.values() for value in time_record.values()
        ) < 0.05
    )
    formation_gate = True
    for variation in VARIATIONS:
        name = variation.name
        formation_gate = bool(
            formation_gate
            and all(
                item["paired_transition_pass"]
                for item in formation[name]["by_grid"].values()
            )
            and formation[name]["first_detection_indices_differ_by_at_most_one"]
            and formation[name]["two_grid_envelope_bracket"] is not None
            and formation[name]["two_grid_envelope_bracket"]["width"] <= 0.00025
            and all(
                record["admitted_distinct_count"] == 2
                for label in geometries
                for record in data[name]["searches"][label]["long"]
            )
        )
    detector_gate = bool(
        all(all_searches_resolved(
            data[variation.name]["searches"][label][family]
        ) for variation in VARIATIONS for label in geometries
          for family in ("fine", "long"))
        and all(
            transfer["endpoint"] is not None
            and transfer["endpoint"]["maximum"] < 0.01
            for transfer in all_grid_transfers
        )
    )
    geometry_gate = bool(
        all(
            branch["admitted"] and branch["geometry"]["finite"]
            and branch["geometry"]["one_sided_cap_area"] > 0
            and branch["geometry"]["equivalent_area_radius"] > 0
            for branch in all_branches
        )
        and all(
            record["branches"][1]["geometry"]["one_sided_cap_area"]
            > record["branches"][0]["geometry"]["one_sided_cap_area"]
            for record in all_search_records if len(record["branches"]) == 2
        )
        and all(
            branch["proper_area_relative_difference"] < 0.01
            and branch["equivalent_radius_relative_difference"] < 0.01
            for transfer in all_grid_transfers for branch in transfer["branches"]
        )
    )
    acceptance = {
        "fixed_initial_data_and_initial_zero_BVP_gate": initial_gate,
        "evolution_boundary_constraint_signature_and_field_transfer_gate": evolution_gate,
        "single_pair_transition_and_persistence_gate": formation_gate,
        "independent_multiseed_BVP_and_endpoint_transfer_gate": detector_gate,
        "proper_geometry_ordering_and_transfer_gate": geometry_gate,
    }
    status = "pass" if all(acceptance.values()) else "review"

    np.savez_compressed(STATE, **state_arrays(data, geometries))
    payload = {
        "status": status,
        "classification": (
            "paired_marginal_surface_formation_robust_to_two_live_gauge_variations"
            if status == "pass" else "gauge_slicing_variation_review"
        ),
        "scope": "prospectively sealed live gauge/slicing audit of corrected A=7.90 paired marginal-surface formation",
        "protocol": PROTOCOL,
        "amplitude": AMPLITUDE,
        "baseline_parameters": BASELINE.public(),
        "variations": [item.public() for item in VARIATIONS],
        "variations_bracket_baseline": brackets_baseline(),
        "fixed_closure": {
            "live_normal_wall_gauge": True,
            "live_outer_sommerfeld": True,
            "physical_initial_geometry_shared": True,
            "source_and_first_time_jet_shared": True,
            "driver_memory_is_gauge_dependent": True,
        },
        "times": {"fine": list(FINE_TIMES), "long": list(LONG_TIMES)},
        "analytic_controls": controls,
        "initial_dynamic_BVP_search": initial_search,
        "initial_identity": initial_identity,
        "initial_gauge_diagnostics": {
            name: {label: public_initial(case) for label, case in item["cases"].items()}
            for name, item in data.items()
        },
        "evolution_diagnostics": {
            name: {
                label: {family: public_diagnostics(run) for family, run in runs.items()}
                for label, runs in item["runs"].items()
            } for name, item in data.items()
        },
        "fine_count_histories": {
            name: {
                label: [record["admitted_distinct_count"] for record in item["searches"][label]["fine"]]
                for label in geometries
            } for name, item in data.items()
        },
        "long_count_histories": {
            name: {
                label: [record["admitted_distinct_count"] for record in item["searches"][label]["long"]]
                for label in geometries
            } for name, item in data.items()
        },
        "formation_localization": formation,
        "searches": {
            name: item["searches"] for name, item in data.items()
        },
        "cross_grid_surface_transfer": cross_grid,
        "cross_grid_field_transfer": field_transfers,
        "baseline_descriptive_comparisons": baseline_comparisons,
        "acceptance": acceptance,
        "state_archive": str(STATE),
        "runtime_seconds": float(time.perf_counter() - started),
        "interpretation": {
            "foliation_dependent": [
                "coordinate formation time and bracket",
                "coordinate endpoint locations",
                "comparison of surface area at equal coordinate-time labels across gauges",
            ],
            "robustness_evidence_if_pass": [
                "zero initial caps in the fixed searched class",
                "paired appearance and persistence in both tested foliations",
                "outer-greater-than-inner intrinsic area ordering",
                "two-grid resolution of intrinsic areas and equivalent radii within each foliation",
            ],
        },
        "limitations": [
            "two nearby admitted live-driver variations are not an exhaustive gauge family",
            "finite twelve-seed star-shaped donor-capped search is not a global surface search",
            "same coordinate time across gauges need not identify the same physical hypersurface",
            "long branch uses dt=0.0005 and is not a new long-time step-refinement study",
            "not an event horizon, continuum topology change, preferred slicing, connected bulk geometry, dark-matter halo, or mass-transfer result",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": status,
        "classification": payload["classification"],
        "fine_count_histories": payload["fine_count_histories"],
        "long_count_histories": payload["long_count_histories"],
        "formation_localization": formation,
        "cross_grid_field_transfer": field_transfers,
        "acceptance": acceptance,
        "runtime_seconds": payload["runtime_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
