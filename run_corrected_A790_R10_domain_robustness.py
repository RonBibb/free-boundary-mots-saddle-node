#!/usr/bin/env python3
"""Sealed matched-spacing Rmax=10 robustness test of A=7.90 formation."""

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import run_corrected_fold_live_nonlinear_gauge_source as live
from bhps.corrected_A790_R10_builder import build_A790_R10_pair
from run_corrected_A790_independent_dynamic_BVP_detector import search_slice
from run_corrected_A790_two_grid_formation_search import (
    endpoint_transfer,
    evolution_pass,
    field_transfer,
    static_search,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry
from run_corrected_fold_short_nonlinear_evolution import (
    RADIAL_COMPARISON_CUT,
    interpolate_fields,
    relative_norm,
)


OUTPUT = Path("results/corrected_A790_R10_domain_robustness.json")
STATE = Path("results/corrected_A790_R10_domain_robustness_state.npz")
PROTOCOL = "notes/82_A790_larger_radial_domain_protocol.md"
R8_FINE_STATE = Path("results/corrected_A790_formation_time_refinement_state.npz")
R8_LONG_STATE = Path("results/corrected_A790_two_grid_formation_search_state.npz")
R8_FINE_RESULT = Path("results/corrected_A790_formation_time_refinement.json")
R8_LONG_RESULT = Path("results/corrected_A790_two_grid_formation_search.json")
R8_FINAL_BVP_RESULT = Path(
    "results/corrected_A790_independent_dynamic_BVP_detector.json"
)
AMPLITUDE = 7.90
FINE_FINAL = 0.001
FINE_STEPS = 8
FINE_TIMES = np.arange(1, 9, dtype=float) * (FINE_FINAL / FINE_STEPS)
LONG_FINAL = 0.004
LONG_STEPS = 8
LONG_CHECKPOINT_STEPS = (4, 6)
LONG_TIMES = (0.002, 0.003, 0.004)


def public_diagnostics(run):
    return {
        "final_time": run["final_time"],
        "steps": run["steps"],
        "finite": run["all_stages_finite"],
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
    }


def integrate(geometry, label, final_time, steps, checkpoints):
    case = live.setup_case(
        geometry, label, live_normal_wall_gauge=True,
        live_outer_sommerfeld=True,
    )
    live.FINAL_TIME = float(final_time)
    live.STEPS = int(steps)
    run = live.integrate(case, checkpoint_steps=checkpoints)
    return case, run


def same_resolution_transfer(reference, reference_z, reference_r, candidate, geometry):
    mask = np.asarray(reference_r) <= RADIAL_COMPARISON_CUT + 1e-12
    candidate_on_reference = interpolate_fields(
        candidate, geometry["z"], geometry["r"],
        np.asarray(reference_z), np.asarray(reference_r)[mask],
    )
    return relative_norm(np.asarray(reference)[:, mask], candidate_on_reference)


def geometry_field_transfer(r8, r10, field):
    left = np.asarray(r8[field])
    right = np.asarray(r10[field])
    mask = np.asarray(r8["r"]) <= RADIAL_COMPARISON_CUT + 1e-12
    zz, rr = np.meshgrid(r8["z"], np.asarray(r8["r"])[mask], indexing="ij")
    right_on_left = RectBivariateSpline(
        r10["z"], r10["r"], right, kx=3, ky=3, s=0,
    ).ev(zz.ravel(), rr.ravel()).reshape(len(r8["z"]), np.count_nonzero(mask))
    return relative_norm(left[:, mask], right_on_left)


def reduced_initial_transfer(r8, r10):
    left = np.asarray(r8["jet_field"].reduced_fields)
    right = np.asarray(r10["jet_field"].reduced_fields)
    mask = np.asarray(r8["r"]) <= RADIAL_COMPARISON_CUT + 1e-12
    right_on_left = interpolate_fields(
        right, r10["z"], r10["r"], r8["z"], np.asarray(r8["r"])[mask],
    )
    return relative_norm(left[:, mask], right_on_left)


def median_signatures(search):
    return [cluster["signature"] for cluster in search["clusters"]]


def reference_signatures():
    fine = json.loads(R8_FINE_RESULT.read_text())
    long = json.loads(R8_LONG_RESULT.read_text())
    bvp = json.loads(R8_FINAL_BVP_RESULT.read_text())
    result = {label: {} for label in ("G7", "G8")}
    for label in result:
        for index, time_value in enumerate(FINE_TIMES):
            time_label = f"{time_value:.6f}"
            if np.isclose(time_value, 0.000625):
                values = bvp["searches"][label]["0.000625"][
                    "admitted_signatures"
                ]
                detector = "local_BVP_note80"
            elif np.isclose(time_value, 0.001):
                values = bvp["searches"][label]["0.001"][
                    "admitted_signatures"
                ]
                detector = "local_BVP_note80"
            else:
                values = fine["trajectory"][label][index]["admitted_signatures"]
                detector = "spectral_note77"
            result[label][time_label] = {
                "signatures": values, "detector": detector,
            }
        for index, time_value in enumerate((0.001, 0.002, 0.003, 0.004)):
            time_label = f"{time_value:.3f}"
            if np.isclose(time_value, 0.004):
                values = bvp["searches"][label]["0.004"][
                    "admitted_signatures"
                ]
                detector = "local_BVP_note80"
            else:
                values = long["dynamic_search"][label][index][
                    "admitted_signatures"
                ]
                detector = "spectral_note74"
            result[label][time_label] = {
                "signatures": values, "detector": detector,
            }
    return result


def search_trajectory(label, geometry, states):
    records = []
    for time_value, state in states:
        time_label = f"{time_value:.6f}"
        record = search_slice(
            f"{label}-R10-t{time_label}", state["_position"],
            state["_velocity"], geometry,
        )
        records.append({
            "time": float(time_value),
            "admitted_distinct_count": record["admitted_distinct_count"],
            "admitted_signatures": median_signatures(record),
            "search": record,
        })
    return records


def reference_comparisons(records, references, long_labels=False):
    comparisons = []
    for record in records:
        key = (
            f"{record['time']:.3f}" if long_labels
            else f"{record['time']:.6f}"
        )
        reference = references[key]
        transfer = (
            endpoint_transfer(
                record["admitted_signatures"], reference["signatures"],
            ) if record["admitted_signatures"] or reference["signatures"]
            else None
        )
        comparisons.append({
            "time": record["time"],
            "reference_detector": reference["detector"],
            "transfer": transfer,
        })
    return comparisons


def all_positive_searches_pass(records):
    return bool(all(
        record["admitted_distinct_count"] == 2
        and all(len(cluster["members"]) >= 2 for cluster in record["search"]["clusters"])
        for record in records
    ))


def main():
    required = (
        R8_FINE_STATE, R8_LONG_STATE, R8_FINE_RESULT, R8_LONG_RESULT,
        R8_FINAL_BVP_RESULT,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required sealed inputs missing: {missing}")
    started = time.perf_counter()
    print("building matched-spacing Rmax=10 G7/G8 initial data", flush=True)
    g7r10, g8r10 = build_A790_R10_pair()
    r10 = {"G7": g7r10, "G8": g8r10}

    print("rebuilding same-resolution Rmax=8 initial references", flush=True)
    fold = build_geometry("G6")
    amplitude_seed = {**fold, "fold_amplitude": AMPLITUDE}
    g7r8 = build_refined(
        amplitude_seed, 81, 121, "G7A790R8-domain-reference",
        selector_iterations=40, slice_iterations=270,
    )
    g8r8 = build_refined(
        g7r8, 97, 145, "G8A790R8-domain-reference",
        selector_iterations=45, slice_iterations=280,
    )
    r8 = {"G7": g7r8, "G8": g8r8}

    initial_transfer = {}
    initial_static = {}
    initial_dynamic = {}
    for label in ("G7", "G8"):
        initial_transfer[label] = {
            "reduced_fields": reduced_initial_transfer(r8[label], r10[label]),
            **{
                field: geometry_field_transfer(r8[label], r10[label], field)
                for field in ("psi", "phi", "a", "b", "c")
            },
        }
        initial_static[label] = static_search(r10[label])
        initial = np.asarray(r10[label]["jet_field"].reduced_fields)
        initial_dynamic[label] = search_slice(
            f"{label}-A790-R10-t0", initial, np.zeros_like(initial), r10[label],
        )

    fine_cases = {}
    fine_runs = {}
    long_cases = {}
    long_runs = {}
    for label in ("G7", "G8"):
        print(f"evolving {label} Rmax=10 fine formation history", flush=True)
        fine_cases[label], fine_runs[label] = integrate(
            r10[label], f"{label}-A790-R10-fine", FINE_FINAL, FINE_STEPS,
            range(1, FINE_STEPS),
        )
        print(f"evolving {label} Rmax=10 persistence history", flush=True)
        long_cases[label], long_runs[label] = integrate(
            r10[label], f"{label}-A790-R10-long", LONG_FINAL, LONG_STEPS,
            LONG_CHECKPOINT_STEPS,
        )

    fine_searches = {}
    long_searches = {}
    for label in ("G7", "G8"):
        fine_states = [
            (
                time_value,
                fine_runs[label]["_checkpoints"][index + 1]
                if index < 7 else fine_runs[label],
            ) for index, time_value in enumerate(FINE_TIMES)
        ]
        long_states = [
            (0.002, long_runs[label]["_checkpoints"][4]),
            (0.003, long_runs[label]["_checkpoints"][6]),
            (0.004, long_runs[label]),
        ]
        fine_searches[label] = search_trajectory(
            f"{label}-A790-R10-fine", r10[label], fine_states,
        )
        long_searches[label] = search_trajectory(
            f"{label}-A790-R10-long", r10[label], long_states,
        )

    references = reference_signatures()
    r8_endpoint_comparisons = {
        label: {
            "fine": reference_comparisons(
                fine_searches[label], references[label], long_labels=False,
            ),
            "long": reference_comparisons(
                long_searches[label], references[label], long_labels=True,
            ),
        } for label in ("G7", "G8")
    }
    cross_grid_endpoint_transfer = {
        "fine": [
            {
                "time": float(time_value),
                "transfer": endpoint_transfer(
                    fine_searches["G7"][index]["admitted_signatures"],
                    fine_searches["G8"][index]["admitted_signatures"],
                ),
            } for index, time_value in enumerate(FINE_TIMES)
        ],
        "long": [
            {
                "time": float(time_value),
                "transfer": endpoint_transfer(
                    long_searches["G7"][index]["admitted_signatures"],
                    long_searches["G8"][index]["admitted_signatures"],
                ),
            } for index, time_value in enumerate(LONG_TIMES)
        ],
    }

    r8_fine = np.load(R8_FINE_STATE)
    r8_long = np.load(R8_LONG_STATE)
    domain_field_transfer = {label: {"t0.001": {}, "t0.004": {}} for label in r10}
    for label in r10:
        for name, key, archive_name in (
            ("position_increment", "_increment", "increment"),
            ("velocity", "_velocity", "velocity"),
            ("source_increment", "_source_increment", "source_increment"),
        ):
            domain_field_transfer[label]["t0.001"][name] = same_resolution_transfer(
                r8_fine[f"{label}_8step_{archive_name}"],
                r8_fine[f"{label}_z"], r8_fine[f"{label}_r"],
                fine_runs[label][key], r10[label],
            )
        for name, key, archive_name in (
            ("position_increment", "_increment", "increment"),
            ("velocity", "_velocity", "velocity"),
        ):
            domain_field_transfer[label]["t0.004"][name] = same_resolution_transfer(
                r8_long[f"{label}_time_3_{archive_name}"],
                r8_long[f"{label}_z"], r8_long[f"{label}_r"],
                long_runs[label][key], r10[label],
            )

    r10_grid_transfer = {"t0.001": {}, "t0.004": {}}
    for name, key in (
        ("position_increment", "_increment"),
        ("velocity", "_velocity"),
        ("source_increment", "_source_increment"),
    ):
        r10_grid_transfer["t0.001"][name] = field_transfer(
            fine_cases["G7"], fine_runs["G7"], fine_cases["G8"],
            fine_runs["G8"], key,
        )
        r10_grid_transfer["t0.004"][name] = field_transfer(
            long_cases["G7"], long_runs["G7"], long_cases["G8"],
            long_runs["G8"], key,
        )

    fine_counts = {
        label: [item["admitted_distinct_count"] for item in records]
        for label, records in fine_searches.items()
    }
    long_counts = {
        label: [item["admitted_distinct_count"] for item in records]
        for label, records in long_searches.items()
    }
    all_r8_transfers = [
        item["transfer"]
        for label in r8_endpoint_comparisons.values()
        for records in label.values() for item in records
        if item["transfer"] is not None
    ]
    all_cross_grid_transfers = [
        item["transfer"]
        for records in cross_grid_endpoint_transfer.values() for item in records
        if item["transfer"] is not None
    ]
    construction_pass = all(
        geometry["reference_maximum_residual"] < 1e-9
        and geometry["selector_maximum"] < 1e-9
        for geometry in r10.values()
    )
    spacing_pass = bool(
        np.isclose(r10["G7"]["r"][1] - r10["G7"]["r"][0], 8.0 / 120.0)
        and np.isclose(r10["G8"]["r"][1] - r10["G8"]["r"][0], 8.0 / 144.0)
    )
    initial_field_pass = max(
        value for record in initial_transfer.values() for value in record.values()
    ) < 0.05
    initial_horizon_pass = all(
        initial_static[label]["accepted_count"] == 0
        and initial_dynamic[label]["admitted_distinct_count"] == 0
        for label in r10
    )
    positive_search_pass = bool(
        fine_counts == {
            "G7": [0, 0, 0, 0, 2, 2, 2, 2],
            "G8": [0, 0, 0, 0, 2, 2, 2, 2],
        }
        and long_counts == {"G7": [2, 2, 2], "G8": [2, 2, 2]}
        and all(
            all_positive_searches_pass(records[4:])
            for records in fine_searches.values()
        )
        and all(
            all_positive_searches_pass(records)
            for records in long_searches.values()
        )
    )
    evolution_gate_pass = bool(
        all(evolution_pass(run) for run in fine_runs.values())
        and all(evolution_pass(run) for run in long_runs.values())
        and max(
            value for record in r10_grid_transfer.values() for value in record.values()
        ) < 0.05
    )
    domain_field_pass = max(
        value for label in domain_field_transfer.values()
        for record in label.values() for value in record.values()
    ) < 0.05
    endpoint_pass = bool(
        len(all_r8_transfers) == 14
        and len(all_cross_grid_transfers) == 7
        and max(
            item["maximum"] for item in all_r8_transfers + all_cross_grid_transfers
        ) < 0.01
    )
    acceptance = {
        "clean_matched_spacing_construction_and_initial_field_transfer": bool(
            construction_pass and spacing_pass and initial_field_pass
        ),
        "both_initial_static_and_dynamic_searches_admit_zero_caps": initial_horizon_pass,
        "fine_formation_and_long_persistence_histories_pass": positive_search_pass,
        "all_evolutions_and_R10_grid_transfers_pass": evolution_gate_pass,
        "R10_R8_common_interior_field_transfer_below_5_percent": domain_field_pass,
        "all_grid_and_domain_endpoint_transfers_below_1_percent": endpoint_pass,
    }
    result_status = "pass" if all(acceptance.values()) else "review"

    state_values = {
        "fine_times": FINE_TIMES,
        "long_times": np.asarray(LONG_TIMES),
    }
    for label in r10:
        state_values[f"{label}_z"] = r10[label]["z"]
        state_values[f"{label}_r"] = r10[label]["r"]
        for index, time_value in enumerate(FINE_TIMES):
            state = (
                fine_runs[label]["_checkpoints"][index + 1]
                if index < 7 else fine_runs[label]
            )
            for name, key in (
                ("increment", "_increment"),
                ("velocity", "_velocity"),
                ("source_increment", "_source_increment"),
            ):
                state_values[f"{label}_fine_{index}_{name}"] = state[key]
        for index, state in enumerate((
            long_runs[label]["_checkpoints"][4],
            long_runs[label]["_checkpoints"][6],
            long_runs[label],
        )):
            for name, key in (
                ("increment", "_increment"),
                ("velocity", "_velocity"),
                ("source_increment", "_source_increment"),
            ):
                state_values[f"{label}_long_{index}_{name}"] = state[key]
    np.savez_compressed(STATE, **state_values)

    payload = {
        "status": result_status,
        "classification": (
            "paired_formation_candidate_robust_to_matched_Rmax10_extension"
            if result_status == "pass" else "larger_domain_robustness_review"
        ),
        "scope": "sealed matched-spacing Rmax=10 robustness test of corrected A=7.90 formation",
        "protocol": PROTOCOL,
        "amplitude": AMPLITUDE,
        "domain_construction": {
            label: {
                "grid_size": geometry["source_grid"],
                "r_max": geometry["radial_domain"][-1],
                "radial_spacing": float(geometry["r"][1] - geometry["r"][0]),
                "reference_residual": geometry["reference_maximum_residual"],
                "selector_residual": geometry["selector_maximum"],
            } for label, geometry in r10.items()
        },
        "initial_field_R10_R8_transfer": initial_transfer,
        "initial_static_search": initial_static,
        "initial_dynamic_BVP_search": initial_dynamic,
        "fine_times": FINE_TIMES.tolist(),
        "long_times": list(LONG_TIMES),
        "fine_count_histories": fine_counts,
        "long_count_histories": long_counts,
        "fine_BVP_searches": fine_searches,
        "long_BVP_searches": long_searches,
        "R8_endpoint_references": references,
        "R10_R8_endpoint_comparisons": r8_endpoint_comparisons,
        "R10_cross_grid_endpoint_transfer": cross_grid_endpoint_transfer,
        "R10_R8_common_interior_field_transfer": domain_field_transfer,
        "R10_cross_grid_field_transfer": r10_grid_transfer,
        "evolution_diagnostics": {
            label: {
                "fine": public_diagnostics(fine_runs[label]),
                "long": public_diagnostics(long_runs[label]),
            } for label in r10
        },
        "acceptance": acceptance,
        "runtime_seconds": float(time.perf_counter() - started),
        "state_archive": str(STATE),
        "limitations": [
            "finite twelve-seed star-shaped donor-capped search",
            "Rmax=10 uses the previously audited shared-domain physical shape family rather than refitting its coefficients at A=7.90",
            "t=0 no-detection is not a proof of global nonexistence",
            "short-time foliation- and detector-dependent apparent-horizon result",
            "not event-horizon location, topology change, open amplitude basin, long-time stability, connected bulk geometry, or mass transfer",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "classification": payload["classification"],
        "fine_counts": fine_counts,
        "long_counts": long_counts,
        "initial_transfer": initial_transfer,
        "domain_field_transfer": domain_field_transfer,
        "cross_grid_field_transfer": r10_grid_transfer,
        "maximum_R8_endpoint_transfer": max(
            (item["maximum"] for item in all_r8_transfers), default=None,
        ),
        "maximum_R10_cross_grid_endpoint_transfer": max(
            (item["maximum"] for item in all_cross_grid_transfers), default=None,
        ),
        "evolution_diagnostics": payload["evolution_diagnostics"],
        "acceptance": acceptance,
        "runtime_seconds": payload["runtime_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
