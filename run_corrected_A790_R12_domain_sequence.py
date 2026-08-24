#!/usr/bin/env python3
"""Prospectively sealed matched-spacing R8/R10/R12 domain-sequence audit."""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import run_corrected_fold_live_nonlinear_gauge_source as live
from bhps.corrected_A790_R10_builder import build_A790_R10_pair
from bhps.corrected_A790_R12_builder import build_A790_R12_pair
from bhps.dynamical_capped_geometry import capped_surface_geometry
from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice
from bhps.dynamical_capped_horizon_bvp import solve_dynamical_capped_surface_bvp
from run_corrected_A790_independent_dynamic_BVP_detector import (
    admitted as bvp_admitted,
    public_surface as public_bvp_surface,
    search_slice,
)
from run_corrected_A790_R10_domain_robustness import (
    geometry_field_transfer,
    public_diagnostics,
    reduced_initial_transfer,
    same_resolution_transfer,
)
from run_corrected_A790_two_grid_formation_search import (
    evolution_pass,
    field_transfer,
    static_search,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry


OUTPUT = Path("results/corrected_A790_R12_domain_sequence.json")
STATE = Path("results/corrected_A790_R12_domain_sequence_state.npz")
PROTOCOL = "notes/88_A790_R12_domain_sequence_protocol.md"
R10_RESULT = Path("results/corrected_A790_R10_domain_robustness.json")
R10_STATE = Path("results/corrected_A790_R10_domain_robustness_state.npz")
R8_BVP_RESULT = Path("results/corrected_A790_independent_dynamic_BVP_detector.json")
R8_FINE_STATE = Path("results/corrected_A790_formation_time_refinement_state.npz")
R8_LONG_STATE = Path("results/corrected_A790_two_grid_formation_search_state.npz")
AMPLITUDE = 7.90
FINE_FINAL = 0.001
FINE_STEPS = 8
FINE_TIMES = np.arange(1, FINE_STEPS + 1, dtype=float) * (FINE_FINAL / FINE_STEPS)
LONG_FINAL = 0.004
LONG_STEPS = 8
LONG_CHECKPOINT_STEPS = (4, 6)
SURFACE_TIMES = (0.001, 0.002, 0.003, 0.004)
COMMON_RADIUS = 6.0


def relative_difference(left, right):
    return float(abs(left - right) / max(abs(left), abs(right), 1e-300))


def integrate(geometry, label, final_time, steps, checkpoints):
    case = live.setup_case(
        geometry, label, live_normal_wall_gauge=True,
        live_outer_sommerfeld=True,
    )
    live.FINAL_TIME = float(final_time)
    live.STEPS = int(steps)
    return case, live.integrate(case, checkpoint_steps=checkpoints)


def search_trajectory(label, geometry, states):
    records = []
    for time_value, state in states:
        record = search_slice(
            f"{label}-t{time_value:.6f}", state["position"], state["velocity"],
            geometry,
        )
        records.append({
            "time": float(time_value),
            "admitted_distinct_count": record["admitted_distinct_count"],
            "admitted_signatures": record["admitted_signatures"],
            "search": record,
        })
    return records


def fine_states(run):
    return [
        {
            "position": (
                run["_checkpoints"][index + 1]["_position"]
                if index < FINE_STEPS - 1 else run["_position"]
            ),
            "velocity": (
                run["_checkpoints"][index + 1]["_velocity"]
                if index < FINE_STEPS - 1 else run["_velocity"]
            ),
        }
        for index in range(FINE_STEPS)
    ]


def long_states(run):
    return [
        {
            "position": run["_checkpoints"][4]["_position"],
            "velocity": run["_checkpoints"][4]["_velocity"],
        },
        {
            "position": run["_checkpoints"][6]["_position"],
            "velocity": run["_checkpoints"][6]["_velocity"],
        },
        {"position": run["_position"], "velocity": run["_velocity"]},
    ]


def archived_state(domain, label, time_value, geometry, archives):
    initial = np.asarray(geometry["jet_field"].reduced_fields)
    if domain == "R12":
        fine_run, long_run = archives
        if np.isclose(time_value, 0.001):
            return {
                "position": fine_run["_position"],
                "velocity": fine_run["_velocity"],
                "source_increment": fine_run["_source_increment"],
            }
        if np.isclose(time_value, 0.002):
            state = long_run["_checkpoints"][4]
        elif np.isclose(time_value, 0.003):
            state = long_run["_checkpoints"][6]
        else:
            state = long_run
        return {
            "position": state["_position"], "velocity": state["_velocity"],
            "source_increment": state["_source_increment"],
        }
    archive = archives
    if domain == "R10":
        if np.isclose(time_value, 0.001):
            prefix = f"{label}_fine_7"
        else:
            index = {0.002: 0, 0.003: 1, 0.004: 2}[time_value]
            prefix = f"{label}_long_{index}"
        return {
            "position": initial + archive[f"{prefix}_increment"],
            "velocity": archive[f"{prefix}_velocity"],
            "source_increment": archive[f"{prefix}_source_increment"],
        }
    if np.isclose(time_value, 0.001):
        prefix = f"{label}_8step"
        archive = archives[0]
        source = archive[f"{prefix}_source_increment"]
    else:
        index = {0.002: 1, 0.003: 2, 0.004: 3}[time_value]
        prefix = f"{label}_time_{index}"
        archive = archives[1]
        source = None
    return {
        "position": initial + archive[f"{prefix}_increment"],
        "velocity": archive[f"{prefix}_velocity"],
        "source_increment": source,
    }


def r10_search_at(result, label, time_value):
    if np.isclose(time_value, 0.001):
        return result["fine_BVP_searches"][label][-1]["search"]
    index = {0.002: 0, 0.003: 1, 0.004: 2}[time_value]
    return result["long_BVP_searches"][label][index]["search"]


def representative_branch_geometry(search, state, geometry):
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
        solve_started = time.perf_counter()
        surface = solve_dynamical_capped_surface_bvp(
            state["position"], state["velocity"], geometry["z"], geometry["r"],
            seed, tolerance=2e-5, nodes=121, maximum_nodes=6000,
            dense_nodes=501,
        )
        surface["runtime_seconds"] = float(time.perf_counter() - solve_started)
        geometric = capped_surface_geometry(
            state["position"], state["velocity"], geometry["z"], geometry["r"],
            surface, prepared=prepared,
        )
        records.append({
            "branch": branch_name,
            "seed": seed,
            "admitted": bvp_admitted(surface),
            "surface": public_bvp_surface(surface),
            "axis": float(surface["rho_axis"]),
            "brane": float(surface["rho_brane"]),
            "area": geometric["one_sided_cap_area"],
            "equivalent_radius": geometric["equivalent_area_radius"],
            "geometry": geometric,
        })
    return records


def write_R12_state_archive(r12, fine_runs, long_runs):
    """Write evolution states before detector/geometry postprocessing."""
    state_values = {
        "fine_times": FINE_TIMES,
        "long_times": np.asarray((0.002, 0.003, 0.004)),
    }
    for label in ("G7", "G8"):
        state_values[f"{label}_z"] = r12[label]["z"]
        state_values[f"{label}_r"] = r12[label]["r"]
        for index in range(FINE_STEPS):
            state = (
                fine_runs[label]["_checkpoints"][index + 1]
                if index < FINE_STEPS - 1 else fine_runs[label]
            )
            for name, key in (
                ("increment", "_increment"), ("velocity", "_velocity"),
                ("source_increment", "_source_increment"),
            ):
                state_values[f"{label}_fine_{index}_{name}"] = state[key]
        for index, state in enumerate((
            long_runs[label]["_checkpoints"][4],
            long_runs[label]["_checkpoints"][6], long_runs[label],
        )):
            for name, key in (
                ("increment", "_increment"), ("velocity", "_velocity"),
                ("source_increment", "_source_increment"),
            ):
                state_values[f"{label}_long_{index}_{name}"] = state[key]
    archive_finite = bool(all(
        np.all(np.isfinite(value)) for value in state_values.values()
    ))
    np.savez_compressed(STATE, **state_values)
    return archive_finite


def surface_transfer(left, right):
    records = []
    for left_branch, right_branch in zip(left, right):
        records.append({
            "branch": left_branch["branch"],
            **{
                name: relative_difference(left_branch[name], right_branch[name])
                for name in ("axis", "brane", "area", "equivalent_radius")
            },
        })
    return records


def surface_sequence_statistics(surface_values):
    ratios = []
    signed_continuations = []
    records = []
    for label in ("G7", "G8"):
        for time_value in SURFACE_TIMES:
            key = f"{time_value:.3f}"
            values = [surface_values[domain][label][key] for domain in ("R8", "R10", "R12")]
            if not all(len(item) == 2 for item in values):
                continue
            for branch_index, branch_name in enumerate(("inner", "outer")):
                for observable in ("axis", "brane", "area", "equivalent_radius"):
                    x8, x10, x12 = (
                        values[index][branch_index][observable] for index in range(3)
                    )
                    d810 = x10 - x8
                    d1012 = x12 - x10
                    scale = max(abs(x8), abs(x10), abs(x12), 1.0)
                    ratio = None if abs(d810) <= 1e-12 * scale else abs(d1012) / abs(d810)
                    same_direction = None
                    if abs(d810) > 1e-12 * scale and abs(d1012) > 1e-12 * scale:
                        same_direction = bool(np.sign(d810) == np.sign(d1012))
                        signed_continuations.append(same_direction)
                    if ratio is not None:
                        ratios.append(float(ratio))
                    records.append({
                        "grid": label, "time": time_value, "branch": branch_name,
                        "observable": observable, "R8_to_R10_signed": float(d810),
                        "R10_to_R12_signed": float(d1012),
                        "absolute_change_ratio": None if ratio is None else float(ratio),
                        "same_signed_direction": same_direction,
                    })
    return {
        "records": records,
        "defined_ratio_count": len(ratios),
        "median_absolute_change_ratio": (
            float(np.median(ratios)) if ratios else None
        ),
        "fraction_absolute_change_ratios_below_one": (
            float(np.mean(np.asarray(ratios) < 1.0)) if ratios else None
        ),
        "defined_signed_direction_count": len(signed_continuations),
        "fraction_signed_changes_continuing": (
            float(np.mean(signed_continuations)) if signed_continuations else None
        ),
    }


def valid_single_transition(counts):
    return bool(
        counts == [0] * len(counts)
        or (
            all(value in (0, 2) for value in counts)
            and all(left <= right for left, right in zip(counts, counts[1:]))
            and counts.count(2) > 0
        )
    )


def classify_domain_sequence(
    numerical_rules, fine_counts, sequence_statistics, persistence_usable,
):
    construction_evolution_valid = bool(
        numerical_rules["clean_R12_construction_and_initial_transfer"]
        and numerical_rules["R12_evolutions_and_cross_grid_fields_pass"]
    )
    if not construction_evolution_valid:
        return "invalid_R12_domain_audit"
    exact_r10_history = all(
        fine_counts[label] == [0, 0, 0, 0, 0, 0, 0, 2]
        for label in ("G7", "G8")
    )
    median_ratio = sequence_statistics["median_absolute_change_ratio"]
    fraction_shrinking = sequence_statistics[
        "fraction_absolute_change_ratios_below_one"
    ]
    category_a = bool(
        all(numerical_rules.values()) and exact_r10_history
        and median_ratio is not None and median_ratio < 0.75
        and fraction_shrinking is not None and fraction_shrinking >= 0.75
    )
    if category_a:
        return "R10_R12_agreement_R8_boundary_sensitive_outlier_supported"
    r10_first_index = 7
    r12_later_or_absent = any(
        (2 not in fine_counts[label])
        or fine_counts[label].index(2) > r10_first_index
        for label in ("G7", "G8")
    )
    continuing = sequence_statistics["fraction_signed_changes_continuing"]
    ratio_continues = bool(
        median_ratio is not None and median_ratio >= 0.75
        and continuing is not None and continuing >= 0.75
    )
    if persistence_usable and (r12_later_or_absent or ratio_continues):
        return "continuing_domain_drift"
    return "unresolved_nonmonotonic_domain_behavior"


def initial_causal_record(geometry):
    maximum_speed = float(np.max(np.asarray(
        geometry["principal"]["r_coordinate_speed"]
    )))
    lower_bound = float((geometry["r"][-1] - COMMON_RADIUS) / maximum_speed)
    return {
        "maximum_initial_radial_coordinate_speed": maximum_speed,
        "one_way_boundary_to_r6_lower_bound": lower_bound,
        "final_time_fraction_of_lower_bound": float(LONG_FINAL / lower_bound),
        "final_time_below_lower_bound": bool(LONG_FINAL < lower_bound),
    }


def main():
    required = (R10_RESULT, R10_STATE, R8_BVP_RESULT, R8_FINE_STATE, R8_LONG_STATE)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required sealed inputs missing: {missing}")
    started = time.perf_counter()

    print("building genuine matched-spacing Rmax=12 G7/G8 data", flush=True)
    g7r12, g8r12 = build_A790_R12_pair()
    r12 = {"G7": g7r12, "G8": g8r12}
    print("rebuilding independent Rmax=10 and Rmax=8 references", flush=True)
    g7r10, g8r10 = build_A790_R10_pair()
    r10 = {"G7": g7r10, "G8": g8r10}
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": AMPLITUDE}
    g7r8 = build_refined(
        seed, 81, 121, "G7A790R8-R12-sequence", selector_iterations=40,
        slice_iterations=270,
    )
    g8r8 = build_refined(
        g7r8, 97, 145, "G8A790R8-R12-sequence", selector_iterations=45,
        slice_iterations=280,
    )
    geometries = {
        "R8": {"G7": g7r8, "G8": g8r8},
        "R10": r10,
        "R12": r12,
    }

    initial_transfers = {"R8_R10": {}, "R10_R12": {}}
    for label in ("G7", "G8"):
        for pair_name, left, right in (
            ("R8_R10", geometries["R8"][label], r10[label]),
            ("R10_R12", r10[label], r12[label]),
        ):
            initial_transfers[pair_name][label] = {
                "reduced_fields": reduced_initial_transfer(left, right),
                **{
                    field: geometry_field_transfer(left, right, field)
                    for field in ("psi", "phi", "a", "b", "c")
                },
            }
    shape_identity = {
        label: max(
            float(np.max(np.abs(r10[label][field] - r12[label][field][:, :len(r10[label]["r"])])))
            for field in ("a", "b", "c")
        ) for label in ("G7", "G8")
    }

    initial_static = {}
    initial_bvp = {}
    for label in ("G7", "G8"):
        print(f"{label} R12: initial static and independent BVP searches", flush=True)
        initial_static[label] = static_search(r12[label])
        initial = np.asarray(r12[label]["jet_field"].reduced_fields)
        initial_bvp[label] = search_slice(
            f"{label}-A790-R12-t0", initial, np.zeros_like(initial), r12[label],
        )

    fine_cases = {}
    fine_runs = {}
    long_cases = {}
    long_runs = {}
    for label in ("G7", "G8"):
        print(f"evolving {label} R12 fine formation history", flush=True)
        fine_cases[label], fine_runs[label] = integrate(
            r12[label], f"{label}-A790-R12-fine", FINE_FINAL, FINE_STEPS,
            range(1, FINE_STEPS),
        )
        print(f"evolving {label} R12 persistence history", flush=True)
        long_cases[label], long_runs[label] = integrate(
            r12[label], f"{label}-A790-R12-long", LONG_FINAL, LONG_STEPS,
            LONG_CHECKPOINT_STEPS,
        )

    archive_finite = write_R12_state_archive(r12, fine_runs, long_runs)
    print(f"wrote recoverable R12 evolution archive to {STATE}", flush=True)

    fine_searches = {}
    long_searches = {}
    for label in ("G7", "G8"):
        fine_searches[label] = search_trajectory(
            f"{label}-A790-R12-fine", r12[label],
            list(zip(FINE_TIMES, fine_states(fine_runs[label]))),
        )
        long_searches[label] = search_trajectory(
            f"{label}-A790-R12-long", r12[label],
            list(zip((0.002, 0.003, 0.004), long_states(long_runs[label]))),
        )

    r10_result = json.loads(R10_RESULT.read_text())
    r8_bvp_result = json.loads(R8_BVP_RESULT.read_text())
    r10_archive = np.load(R10_STATE)
    r8_archives = (np.load(R8_FINE_STATE), np.load(R8_LONG_STATE))
    r12_archives = {
        label: (fine_runs[label], long_runs[label]) for label in ("G7", "G8")
    }

    searches = {domain: {label: {} for label in ("G7", "G8")} for domain in geometries}
    surface_values = {domain: {label: {} for label in ("G7", "G8")} for domain in geometries}
    for label in ("G7", "G8"):
        for time_index, time_value in enumerate(SURFACE_TIMES):
            time_key = f"{time_value:.3f}"
            r12_search = (
                fine_searches[label][-1]["search"] if np.isclose(time_value, 0.001)
                else long_searches[label][time_index - 1]["search"]
            )
            searches["R12"][label][time_key] = r12_search
            searches["R10"][label][time_key] = r10_search_at(
                r10_result, label, time_value,
            )
            if time_key in ("0.001", "0.004"):
                searches["R8"][label][time_key] = r8_bvp_result["searches"][label][time_key]
            else:
                state = archived_state(
                    "R8", label, time_value, geometries["R8"][label], r8_archives,
                )
                print(f"{label} R8 t={time_key}: filling independent BVP reference", flush=True)
                searches["R8"][label][time_key] = search_slice(
                    f"{label}-A790-R8-t{time_key}-R12-sequence",
                    state["position"], state["velocity"], geometries["R8"][label],
                )
            for domain, archive in (
                ("R8", r8_archives), ("R10", r10_archive),
                ("R12", r12_archives[label]),
            ):
                state = archived_state(
                    domain, label, time_value, geometries[domain][label], archive,
                )
                search = searches[domain][label][time_key]
                print(
                    f"{label} {domain} t={time_key}: proper branch geometry",
                    flush=True,
                )
                surface_values[domain][label][time_key] = (
                    representative_branch_geometry(search, state, geometries[domain][label])
                    if search["admitted_distinct_count"] == 2 else []
                )

    surface_transfers = {
        "R8_R10": {label: {} for label in ("G7", "G8")},
        "R10_R12": {label: {} for label in ("G7", "G8")},
        "R12_G7_G8": {},
    }
    for label in ("G7", "G8"):
        for time_value in SURFACE_TIMES:
            key = f"{time_value:.3f}"
            for name, left, right in (
                ("R8_R10", "R8", "R10"), ("R10_R12", "R10", "R12"),
            ):
                if len(surface_values[left][label][key]) == len(surface_values[right][label][key]) == 2:
                    surface_transfers[name][label][key] = surface_transfer(
                        surface_values[left][label][key], surface_values[right][label][key],
                    )
                else:
                    surface_transfers[name][label][key] = None
            if len(surface_values["R12"]["G7"][key]) == len(surface_values["R12"]["G8"][key]) == 2:
                surface_transfers["R12_G7_G8"][key] = surface_transfer(
                    surface_values["R12"]["G7"][key], surface_values["R12"]["G8"][key],
                )
            else:
                surface_transfers["R12_G7_G8"][key] = None

    field_transfers = {
        "R8_R10": {label: {} for label in ("G7", "G8")},
        "R10_R12": {label: {} for label in ("G7", "G8")},
        "R12_G7_G8": {"t0.001": {}, "t0.004": {}},
    }
    for label in ("G7", "G8"):
        for time_value in (0.001, 0.004):
            time_key = f"t{time_value:.3f}"
            states = {
                "R8": archived_state("R8", label, time_value, geometries["R8"][label], r8_archives),
                "R10": archived_state("R10", label, time_value, r10[label], r10_archive),
                "R12": archived_state("R12", label, time_value, r12[label], r12_archives[label]),
            }
            for name, left_domain, right_domain in (
                ("R8_R10", "R8", "R10"), ("R10_R12", "R10", "R12"),
            ):
                fields = {
                    "position_increment": same_resolution_transfer(
                        states[left_domain]["position"] - np.asarray(geometries[left_domain][label]["jet_field"].reduced_fields),
                        geometries[left_domain][label]["z"], geometries[left_domain][label]["r"],
                        states[right_domain]["position"] - np.asarray(geometries[right_domain][label]["jet_field"].reduced_fields),
                        geometries[right_domain][label],
                    ),
                    "velocity": same_resolution_transfer(
                        states[left_domain]["velocity"], geometries[left_domain][label]["z"],
                        geometries[left_domain][label]["r"], states[right_domain]["velocity"],
                        geometries[right_domain][label],
                    ),
                }
                if states[left_domain]["source_increment"] is not None and states[right_domain]["source_increment"] is not None:
                    fields["source_increment"] = same_resolution_transfer(
                        states[left_domain]["source_increment"], geometries[left_domain][label]["z"],
                        geometries[left_domain][label]["r"], states[right_domain]["source_increment"],
                        geometries[right_domain][label],
                    )
                field_transfers[name][label][time_key] = fields
    for name, key in (
        ("position_increment", "_increment"), ("velocity", "_velocity"),
        ("source_increment", "_source_increment"),
    ):
        field_transfers["R12_G7_G8"]["t0.001"][name] = field_transfer(
            fine_cases["G7"], fine_runs["G7"], fine_cases["G8"], fine_runs["G8"], key,
        )
        field_transfers["R12_G7_G8"]["t0.004"][name] = field_transfer(
            long_cases["G7"], long_runs["G7"], long_cases["G8"], long_runs["G8"], key,
        )

    fine_counts = {
        label: [item["admitted_distinct_count"] for item in fine_searches[label]]
        for label in ("G7", "G8")
    }
    long_counts = {
        label: [item["admitted_distinct_count"] for item in long_searches[label]]
        for label in ("G7", "G8")
    }
    sequence_statistics = surface_sequence_statistics(surface_values)
    causal = {
        domain: {label: initial_causal_record(geometries[domain][label]) for label in ("G7", "G8")}
        for domain in geometries
    }

    construction_pass = bool(all(
        geometry["reference_maximum_residual"] < 1e-9
        and geometry["selector_maximum"] < 1e-9
        for geometry in r12.values()
    ))
    spacing_pass = bool(
        np.isclose(r12["G7"]["r"][1] - r12["G7"]["r"][0], 8.0 / 120.0)
        and np.isclose(r12["G8"]["r"][1] - r12["G8"]["r"][0], 8.0 / 144.0)
    )
    initial_transfer_pass = bool(max(
        initial_transfers["R10_R12"][label][field]
        for label in ("G7", "G8") for field in ("reduced_fields", "psi", "phi")
    ) < 0.02)
    initial_zero_pass = bool(all(
        initial_static[label]["accepted_count"] == 0
        and initial_bvp[label]["admitted_distinct_count"] == 0
        for label in ("G7", "G8")
    ))
    evolution_gate = bool(
        all(evolution_pass(run) for run in fine_runs.values())
        and all(evolution_pass(run) for run in long_runs.values())
        and max(
            value for time_values in field_transfers["R12_G7_G8"].values()
            for value in time_values.values()
        ) < 0.05
    )
    multi_seed_persistence = bool(all(
        record["admitted_distinct_count"] == 2
        and all(len(cluster["members"]) >= 2 for cluster in record["search"]["clusters"])
        for label in ("G7", "G8") for record in long_searches[label]
    ))
    count_pass = bool(
        all(valid_single_transition(fine_counts[label]) for label in ("G7", "G8"))
        and long_counts == {"G7": [2, 2, 2], "G8": [2, 2, 2]}
        and multi_seed_persistence
    )
    compared_r10_r12 = [
        value for label in ("G7", "G8")
        for record in surface_transfers["R10_R12"][label].values()
        if record is not None for branch in record
        for value in (branch["axis"], branch["brane"], branch["area"], branch["equivalent_radius"])
    ]
    compared_r12_grid = [
        value for record in surface_transfers["R12_G7_G8"].values()
        if record is not None for branch in record
        for value in (branch["axis"], branch["brane"], branch["area"], branch["equivalent_radius"])
    ]
    r10_r12_fields = [
        value for label in ("G7", "G8")
        for record in field_transfers["R10_R12"][label].values()
        for value in record.values()
    ]
    comparison_pass = bool(
        compared_r10_r12 and compared_r12_grid
        and max(compared_r10_r12 + compared_r12_grid) < 0.01
        and max(r10_r12_fields) < 0.02
    )
    causal_pass = bool(all(
        causal["R12"][label]["final_time_below_lower_bound"]
        for label in ("G7", "G8")
    ))
    numerical_rules = {
        "clean_R12_construction_and_initial_transfer": bool(
            construction_pass and spacing_pass and max(shape_identity.values()) < 2e-14
            and initial_transfer_pass
        ),
        "both_initial_searches_admit_zero_caps": initial_zero_pass,
        "R12_evolutions_and_cross_grid_fields_pass": evolution_gate,
        "fine_transition_and_long_pair_persistence_pass": count_pass,
        "R10_R12_and_R12_cross_grid_surface_and_field_transfers_pass": comparison_pass,
        "causal_boundary_timing_pass": causal_pass,
        "state_archive_complete_and_finite": archive_finite,
    }
    persistence_usable = bool(
        construction_pass and evolution_gate and multi_seed_persistence
    )
    classification = classify_domain_sequence(
        numerical_rules, fine_counts, sequence_statistics, persistence_usable,
    )
    pair_support = bool(
        multi_seed_persistence and comparison_pass and evolution_gate
        and archive_finite and causal_pass
    )
    formation_support = bool(
        classification == "R10_R12_agreement_R8_boundary_sensitive_outlier_supported"
        and pair_support
    )
    if classification == "invalid_R12_domain_audit":
        status = "fail"
    elif all(numerical_rules.values()) and pair_support and formation_support:
        status = "pass"
    else:
        status = "review"

    payload = {
        "status": status,
        "classification": classification,
        "scope": "sealed matched-spacing R8/R10/R12 domain-sequence audit of corrected A=7.90 formation",
        "protocol": PROTOCOL,
        "amplitude": AMPLITUDE,
        "domain_construction": {
            label: {
                "grid_size": geometry["source_grid"],
                "r_max": geometry["radial_domain"][-1],
                "radial_spacing": float(geometry["r"][1] - geometry["r"][0]),
                "reference_residual": geometry["reference_maximum_residual"],
                "selector_residual": geometry["selector_maximum"],
                "R10_common_node_shape_max_abs_difference": shape_identity[label],
            } for label, geometry in r12.items()
        },
        "initial_field_transfers": initial_transfers,
        "initial_static_searches_R12": initial_static,
        "initial_dynamic_BVP_searches_R12": initial_bvp,
        "fine_times": FINE_TIMES.tolist(),
        "fine_count_histories_R12": fine_counts,
        "long_times": [0.002, 0.003, 0.004],
        "long_count_histories_R12": long_counts,
        "fine_BVP_searches_R12": fine_searches,
        "long_BVP_searches_R12": long_searches,
        "all_domain_surface_searches": searches,
        "all_domain_representative_surface_geometry": surface_values,
        "surface_transfers": surface_transfers,
        "surface_domain_sequence_statistics": sequence_statistics,
        "evolved_field_transfers": field_transfers,
        "causal_boundary_timing": causal,
        "evolution_diagnostics_R12": {
            label: {
                "fine": public_diagnostics(fine_runs[label]),
                "long": public_diagnostics(long_runs[label]),
            } for label in ("G7", "G8")
        },
        "numerical_acceptance": numerical_rules,
        "claim_support": {
            "pair_existence_and_persistence_domain_asymptotic_support": pair_support,
            "formation_time_domain_asymptotic_support": formation_support,
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "state_archive": str(STATE),
        "limitations": [
            "finite twelve-seed star-shaped donor-capped search",
            "R12 uses the audited shared-domain A=8 physical shape family rather than refitting its coefficients at A=7.90",
            "globally elliptic initial data can depend immediately on radial-domain placement",
            "short-time foliation- and detector-dependent apparent-horizon result",
            "not an event horizon, topology change, open amplitude basin, long-time stability, connected bulk geometry, dark-matter halo, or mass transfer",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": status,
        "classification": classification,
        "fine_counts": fine_counts,
        "long_counts": long_counts,
        "sequence_statistics": sequence_statistics,
        "claim_support": payload["claim_support"],
        "numerical_acceptance": numerical_rules,
        "causal_R12": causal["R12"],
        "runtime_seconds": payload["runtime_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
