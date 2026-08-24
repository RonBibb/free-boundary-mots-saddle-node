#!/usr/bin/env python3
"""Run the archive-only Test 10E boundary/corner localization addendum."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bhps.corrected_A790_test10e_boundary_corner_localization import (
    COMPONENTS,
    NODE_COLLAR_COUNTS,
    PROPER_COLLAR_WIDTHS,
    empirical_power_law,
    localize_face_enumeration,
    manufactured_controls,
)
from bhps.recovery_indexer import atomic_write_json, atomic_write_npz, sha256_file


PROTOCOL = Path(
    "notes/111_A790_test10E_archive_boundary_corner_localization_protocol.md"
)
OUTPUT = Path(
    "results/corrected_A790_test10e_archive_boundary_corner_localization.json"
)
STATE_OUTPUT = Path(
    "results/corrected_A790_test10e_archive_boundary_corner_localization_state.npz"
)
TEST10E_ROOT = Path(
    "results/corrected_A790_test10e_genuine_high_z_boundary_resolution_recovery"
)
TEST10C_ROOT = Path(
    "results/corrected_A790_test10c_outer_scalar_closure_recovery"
)
TEST10B_ROOT = Path("results/corrected_A790_test10b_domain_normalized_recovery")
TEST10B_STATE = Path("results/corrected_A790_test10b_domain_normalized_state.npz")
TEST10E_RESULT = Path(
    "results/corrected_A790_test10e_genuine_high_z_boundary_resolution.json"
)
TEST10C_RESULT = Path("results/corrected_A790_test10c_outer_scalar_closure.json")

DT = 0.000125
PRIMARY_GRIDS = ("G7", "G8", "G9", "G10")
DOMAINS = ("R8", "R10", "R12")
INTERVALS = (80, 96, 112, 128)
CONTROL_LABELS = ("Z9_R10", "Z10_R10", "G10H_R10")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_chunks(paths, keys):
    paths = tuple(sorted(Path(path) for path in paths))
    if not paths:
        raise FileNotFoundError("no archive chunks matched")
    values = {key: [] for key in keys}
    for path in paths:
        with np.load(path) as archive:
            missing = [key for key in keys if key not in archive.files]
            if missing:
                raise KeyError(f"archive chunk {path} is missing {missing}")
            for key in keys:
                values[key].append(np.asarray(archive[key]))
    return {key: np.concatenate(items, axis=0) for key, items in values.items()}, paths


def _test10e_run(label):
    keys = (
        "enumeration", "step", "rk_stage", "time", "before", "after",
        "target", "term_A", "term_V", "term_C", "q_perp", "q_zz",
    )
    arrays, paths = _load_chunks(
        TEST10E_ROOT.glob(f"physical_{label}_steps_*.npz"), keys,
    )
    count = len(arrays["time"])
    expected = 64 if label == "G10H_R10" else 32
    if count != expected:
        raise RuntimeError(f"{label} has {count} RK evaluations, expected {expected}")
    if not np.array_equal(arrays["rk_stage"], np.tile((1, 2), count // 2)):
        raise RuntimeError(f"{label} RK enumeration is not alternating 1,2")
    if np.any(np.diff(arrays["enumeration"]) != 1):
        raise RuntimeError(f"{label} archive enumeration is not contiguous")
    open_count = arrays["before"].shape[1]
    z = np.linspace(1.0, math.e, open_count + 2)
    radius = float(label.split("_R")[-1])
    return {
        "label": label,
        "source": "Test10E physical RK-face chunks",
        "enumeration": arrays["enumeration"],
        "step": arrays["step"],
        "rk_stage": arrays["rk_stage"],
        "time": arrays["time"],
        "before": arrays["before"],
        "after": arrays["after"],
        "q_perp": arrays["q_perp"],
        "q_zz": arrays["q_zz"],
        "z": z,
        "radius": radius,
        "input_paths": paths,
        "stage_position_reconstruction_maximum": 0.0,
    }


def _accepted_test10b_states(label, initial):
    increments = {}
    velocities = {}
    paths = tuple(sorted(TEST10B_ROOT.glob(f"evolution_{label}_steps_*.npz")))
    if len(paths) != 4:
        raise RuntimeError(f"{label} does not have four Test10B evolution chunks")
    for path in paths:
        with np.load(path) as archive:
            start = int(archive["start_step"])
            end = int(archive["end_step"])
            for step in range(start + 1, end + 1):
                increments[step] = np.asarray(archive[f"step_{step:03d}_increment"])
                velocities[step] = np.asarray(archive[f"step_{step:03d}_velocity"])
    if tuple(sorted(increments)) != tuple(range(1, 17)):
        raise RuntimeError(f"{label} accepted-state archive is incomplete")
    positions = {0: np.asarray(initial, dtype=float)}
    accepted_velocities = {0: np.zeros_like(initial, dtype=float)}
    for step in range(1, 17):
        positions[step] = np.asarray(initial, dtype=float) + increments[step]
        accepted_velocities[step] = velocities[step]
    return positions, accepted_velocities, paths


def _historical_run(label):
    keys = (
        "enumeration", "step", "rk_stage", "time", "before", "after",
        "target", "term_A", "term_V", "term_C", "step_replay_error",
    )
    arrays, replay_paths = _load_chunks(
        TEST10C_ROOT.glob(f"replay_{label}_steps_*.npz"), keys,
    )
    if len(arrays["time"]) != 32:
        raise RuntimeError(f"{label} historical replay does not have 32 RK evaluations")
    with np.load(TEST10B_STATE) as state:
        z = np.asarray(state[f"{label}_z"], dtype=float)
        r = np.asarray(state[f"{label}_r"], dtype=float)
        initial = np.asarray(state[f"{label}_initial"], dtype=float)
    canonical = np.linspace(1.0, math.e, len(z))
    if not np.array_equal(z, canonical):
        raise RuntimeError(f"{label} compact grid is not the canonical archived grid")
    positions, velocities, evolution_paths = _accepted_test10b_states(label, initial)
    q_perp = []
    q_zz = []
    for step, rk_stage, when in zip(
        arrays["step"], arrays["rk_stage"], arrays["time"], strict=True,
    ):
        step = int(step)
        rk_stage = int(rk_stage)
        expected_time = (step - 1) * DT + (0.5 * DT if rk_stage == 2 else 0.0)
        if not np.isclose(when, expected_time, rtol=0.0, atol=1e-15):
            raise RuntimeError(f"{label} replay time is not aligned")
        stage_position = positions[step - 1].copy()
        if rk_stage == 2:
            stage_position += 0.5 * DT * velocities[step - 1]
        elif rk_stage != 1:
            raise RuntimeError(f"{label} has an unknown RK stage")
        q_perp.append(stage_position[1:-1, -1, 3])
        q_zz.append(stage_position[1:-1, -1, 6])
    replay_error = float(np.max(np.abs(arrays["step_replay_error"])))
    if not np.isfinite(replay_error) or replay_error > 1e-13:
        raise RuntimeError(f"{label} replay reconstruction error is {replay_error}")
    return {
        "label": label,
        "source": "Test10C replay plus Test10B archived stage reconstruction",
        "enumeration": arrays["enumeration"],
        "step": arrays["step"],
        "rk_stage": arrays["rk_stage"],
        "time": arrays["time"],
        "before": arrays["before"],
        "after": arrays["after"],
        "q_perp": np.asarray(q_perp),
        "q_zz": np.asarray(q_zz),
        "z": z,
        "radius": float(r[-1]),
        "input_paths": (*replay_paths, *evolution_paths, TEST10B_STATE),
        "stage_position_reconstruction_maximum": replay_error,
    }


def _empty_analysis(run):
    count = len(run["time"])
    proper_count = len(PROPER_COLLAR_WIDTHS)
    node_count = len(NODE_COLLAR_COUNTS)
    return {
        "enumeration": np.asarray(run["enumeration"], dtype=int),
        "step": np.asarray(run["step"], dtype=int),
        "rk_stage": np.asarray(run["rk_stage"], dtype=int),
        "time": np.asarray(run["time"], dtype=float),
        "proper_length": np.zeros(count),
        "total_l1": np.zeros((count, 2)),
        "total_squared_l2": np.zeros((count, 2)),
        "total_l2": np.zeros((count, 2)),
        "combined_total_l1": np.zeros(count),
        "combined_total_squared_l2": np.zeros(count),
        "combined_total_l2": np.zeros(count),
        "phi_squared_l2_fraction": np.zeros(count),
        "pointwise_maximum": np.zeros((count, 2)),
        "pointwise_index": np.zeros((count, 2), dtype=int),
        "pointwise_z": np.zeros((count, 2)),
        "pointwise_proper": np.zeros((count, 2)),
        "pointwise_nearest_edge": np.zeros((count, 2)),
        "combined_pointwise_maximum": np.zeros(count),
        "combined_pointwise_index": np.zeros(count, dtype=int),
        "combined_pointwise_z": np.zeros(count),
        "combined_pointwise_proper": np.zeros(count),
        "combined_pointwise_nearest_edge": np.zeros(count),
        "proper_lower_squared_fraction": np.zeros((count, proper_count, 3)),
        "proper_upper_squared_fraction": np.zeros((count, proper_count, 3)),
        "proper_union_l1_fraction": np.zeros((count, proper_count, 3)),
        "proper_union_squared_fraction": np.zeros((count, proper_count, 3)),
        "proper_central_squared_fraction": np.zeros((count, proper_count, 3)),
        "proper_alternate_union_squared_fraction": np.zeros((count, proper_count)),
        "proper_method_absolute_difference": np.zeros((count, proper_count)),
        "node_lower_squared_fraction": np.zeros((count, node_count, 3)),
        "node_upper_squared_fraction": np.zeros((count, node_count, 3)),
        "node_union_l1_fraction": np.zeros((count, node_count, 3)),
        "node_union_squared_fraction": np.zeros((count, node_count, 3)),
        "node_central_squared_fraction": np.zeros((count, node_count, 3)),
        "node_alternate_union_squared_fraction": np.zeros((count, node_count)),
        "node_method_absolute_difference": np.zeros((count, node_count)),
        "node_lower_span": np.zeros((count, node_count)),
        "node_upper_span": np.zeros((count, node_count)),
        "corner_absolute": np.zeros((count, 2, 2)),
        "corner_normalized": np.zeros((count, 2, 2)),
        "corner_delta_z_scaled": np.zeros((count, 2, 2)),
    }


def _component_and_combined(component_values, combined):
    return np.r_[np.asarray(component_values, dtype=float), float(combined)]


def analyze_run(run):
    analysis = _empty_analysis(run)
    for index in range(len(run["time"])):
        record = localize_face_enumeration(
            run["z"], run["radius"], run["q_perp"][index], run["q_zz"][index],
            run["before"][index], run["after"][index],
        )
        analysis["proper_length"][index] = record["proper_length_open_nodes"]
        for key in ("total_l1", "total_squared_l2", "total_l2"):
            analysis[key][index] = record[key]
        for key in (
            "combined_total_l1", "combined_total_squared_l2", "combined_total_l2",
            "phi_squared_l2_fraction",
        ):
            analysis[key][index] = record[key]
        analysis["pointwise_maximum"][index] = record["pointwise_maximum"]
        analysis["pointwise_index"][index] = record["pointwise_index"]
        analysis["pointwise_z"][index] = record["pointwise_z"]
        analysis["pointwise_proper"][index] = record[
            "pointwise_proper_from_first_open"
        ]
        analysis["pointwise_nearest_edge"][index] = record[
            "pointwise_proper_to_nearest_open_edge"
        ]
        analysis["combined_pointwise_maximum"][index] = record[
            "combined_pointwise_maximum"
        ]
        analysis["combined_pointwise_index"][index] = record[
            "combined_pointwise_index"
        ]
        analysis["combined_pointwise_z"][index] = record[
            "combined_pointwise_z"
        ]
        analysis["combined_pointwise_proper"][index] = record[
            "combined_pointwise_proper_from_first_open"
        ]
        analysis["combined_pointwise_nearest_edge"][index] = record[
            "combined_pointwise_proper_to_nearest_open_edge"
        ]
        for collar_index, collar in enumerate(record["proper_collars"]):
            analysis["proper_lower_squared_fraction"][index, collar_index] = (
                _component_and_combined(
                    collar["lower_squared_l2_fraction"],
                    collar["combined_lower_squared_l2_fraction"],
                )
            )
            analysis["proper_upper_squared_fraction"][index, collar_index] = (
                _component_and_combined(
                    collar["upper_squared_l2_fraction"],
                    collar["combined_upper_squared_l2_fraction"],
                )
            )
            analysis["proper_union_l1_fraction"][index, collar_index] = (
                _component_and_combined(
                    collar["union_l1_fraction"],
                    collar["combined_union_l1_fraction"],
                )
            )
            analysis["proper_union_squared_fraction"][index, collar_index] = (
                _component_and_combined(
                    collar["union_squared_l2_fraction"],
                    collar["combined_union_squared_l2_fraction"],
                )
            )
            analysis["proper_central_squared_fraction"][index, collar_index] = (
                _component_and_combined(
                    collar["central_squared_l2_fraction"],
                    collar["combined_central_squared_l2_fraction"],
                )
            )
            analysis["proper_alternate_union_squared_fraction"][index, collar_index] = (
                collar["alternate_combined_union_squared_l2_fraction"]
            )
            analysis["proper_method_absolute_difference"][index, collar_index] = (
                collar["primary_alternate_union_squared_fraction_absolute_difference"]
            )
        for collar_index, collar in enumerate(record["node_collars"]):
            analysis["node_lower_squared_fraction"][index, collar_index] = (
                _component_and_combined(
                    collar["lower_squared_l2_fraction"],
                    collar["combined_lower_squared_l2_fraction"],
                )
            )
            analysis["node_upper_squared_fraction"][index, collar_index] = (
                _component_and_combined(
                    collar["upper_squared_l2_fraction"],
                    collar["combined_upper_squared_l2_fraction"],
                )
            )
            analysis["node_union_l1_fraction"][index, collar_index] = (
                _component_and_combined(
                    collar["union_l1_fraction"],
                    collar["combined_union_l1_fraction"],
                )
            )
            analysis["node_union_squared_fraction"][index, collar_index] = (
                _component_and_combined(
                    collar["union_squared_l2_fraction"],
                    collar["combined_union_squared_l2_fraction"],
                )
            )
            analysis["node_central_squared_fraction"][index, collar_index] = (
                _component_and_combined(
                    collar["central_squared_l2_fraction"],
                    collar["combined_central_squared_l2_fraction"],
                )
            )
            analysis["node_alternate_union_squared_fraction"][index, collar_index] = (
                collar["alternate_combined_union_squared_l2_fraction"]
            )
            analysis["node_method_absolute_difference"][index, collar_index] = (
                collar["primary_alternate_union_squared_fraction_absolute_difference"]
            )
            analysis["node_lower_span"][index, collar_index] = collar[
                "lower_proper_span"
            ]
            analysis["node_upper_span"][index, collar_index] = collar[
                "upper_proper_span"
            ]
        compatibility = record["corner_compatibility"]
        analysis["corner_absolute"][index] = compatibility["absolute"]
        analysis["corner_normalized"][index] = compatibility[
            "derivative_cancellation_normalized"
        ]
        analysis["corner_delta_z_scaled"][index] = compatibility["delta_z_scaled"]
    if any(np.any(~np.isfinite(value)) for value in analysis.values()):
        raise RuntimeError(f"{run['label']} localization contains nonfinite values")
    return analysis


def _stage_summary(analysis, index):
    result = {
        "array_index": int(index),
        "enumeration": int(analysis["enumeration"][index]),
        "accepted_step": int(analysis["step"][index]),
        "rk_stage": int(analysis["rk_stage"][index]),
        "time": float(analysis["time"][index]),
        "proper_length_first_to_last_open_node": float(analysis["proper_length"][index]),
        "component_l1": dict(zip(COMPONENTS, analysis["total_l1"][index], strict=True)),
        "component_l2": dict(zip(COMPONENTS, analysis["total_l2"][index], strict=True)),
        "combined_l1": float(analysis["combined_total_l1"][index]),
        "combined_l2": float(analysis["combined_total_l2"][index]),
        "phi_squared_l2_fraction": float(analysis["phi_squared_l2_fraction"][index]),
        "pointwise": {},
        "fixed_proper_collars": {},
        "fixed_node_collars": {},
        "compact_wall_overwrite_compatibility_defect": {},
    }
    for component, component_index in zip(COMPONENTS, range(2), strict=True):
        result["pointwise"][component] = {
            "maximum_absolute_correction": float(
                analysis["pointwise_maximum"][index, component_index]
            ),
            "open_node_index_zero_based": int(
                analysis["pointwise_index"][index, component_index]
            ),
            "z": float(analysis["pointwise_z"][index, component_index]),
            "proper_from_first_open_node": float(
                analysis["pointwise_proper"][index, component_index]
            ),
            "proper_to_nearest_open_edge": float(
                analysis["pointwise_nearest_edge"][index, component_index]
            ),
        }
    result["pointwise"]["combined_vector_magnitude"] = {
        "maximum_absolute_correction": float(
            analysis["combined_pointwise_maximum"][index]
        ),
        "open_node_index_zero_based": int(
            analysis["combined_pointwise_index"][index]
        ),
        "z": float(analysis["combined_pointwise_z"][index]),
        "proper_from_first_open_node": float(
            analysis["combined_pointwise_proper"][index]
        ),
        "proper_to_nearest_open_edge": float(
            analysis["combined_pointwise_nearest_edge"][index]
        ),
    }
    labels = (*COMPONENTS, "combined")
    for collar_index, width in enumerate(PROPER_COLLAR_WIDTHS):
        result["fixed_proper_collars"][str(width)] = {
            "alternate_combined_union_squared_l2_fraction": float(
                analysis["proper_alternate_union_squared_fraction"][index, collar_index]
            ),
            "primary_alternate_absolute_fraction_difference": float(
                analysis["proper_method_absolute_difference"][index, collar_index]
            ),
            **{label: {
                "lower_squared_l2_fraction": float(
                    analysis["proper_lower_squared_fraction"][index, collar_index, item]
                ),
                "upper_squared_l2_fraction": float(
                    analysis["proper_upper_squared_fraction"][index, collar_index, item]
                ),
                "union_l1_fraction": float(
                    analysis["proper_union_l1_fraction"][index, collar_index, item]
                ),
                "union_squared_l2_fraction": float(
                    analysis["proper_union_squared_fraction"][index, collar_index, item]
                ),
                "central_squared_l2_fraction": float(
                    analysis["proper_central_squared_fraction"][index, collar_index, item]
                ),
            } for item, label in enumerate(labels)}
        }
    for collar_index, count in enumerate(NODE_COLLAR_COUNTS):
        result["fixed_node_collars"][str(count)] = {
            "lower_proper_span": float(analysis["node_lower_span"][index, collar_index]),
            "upper_proper_span": float(analysis["node_upper_span"][index, collar_index]),
            "alternate_combined_union_squared_l2_fraction": float(
                analysis["node_alternate_union_squared_fraction"][index, collar_index]
            ),
            "primary_alternate_absolute_fraction_difference": float(
                analysis["node_method_absolute_difference"][index, collar_index]
            ),
            "fractions": {
                label: {
                    "lower_squared_l2_fraction": float(
                        analysis["node_lower_squared_fraction"][index, collar_index, item]
                    ),
                    "upper_squared_l2_fraction": float(
                        analysis["node_upper_squared_fraction"][index, collar_index, item]
                    ),
                    "union_l1_fraction": float(
                        analysis["node_union_l1_fraction"][index, collar_index, item]
                    ),
                    "union_squared_l2_fraction": float(
                        analysis["node_union_squared_fraction"][index, collar_index, item]
                    ),
                    "central_squared_l2_fraction": float(
                        analysis["node_central_squared_fraction"][index, collar_index, item]
                    ),
                } for item, label in enumerate(labels)
            },
        }
    for wall, wall_index in (("lower", 0), ("upper", 1)):
        result["compact_wall_overwrite_compatibility_defect"][wall] = {
            component: {
                "absolute": float(analysis["corner_absolute"][index, wall_index, item]),
                "derivative_cancellation_normalized": float(
                    analysis["corner_normalized"][index, wall_index, item]
                ),
                "delta_z_over_max_correction_scaled": float(
                    analysis["corner_delta_z_scaled"][index, wall_index, item]
                ),
            } for item, component in enumerate(COMPONENTS)
        }
    return result


def summarize_run(run, analysis):
    peak = int(np.argmax(analysis["combined_total_l2"]))
    maxima = {
        "combined_l1": float(np.max(analysis["combined_total_l1"])),
        "combined_l2": float(np.max(analysis["combined_total_l2"])),
        "Phi_l1": float(np.max(analysis["total_l1"][:, 0])),
        "Phi_l2": float(np.max(analysis["total_l2"][:, 0])),
        "chi_l1": float(np.max(analysis["total_l1"][:, 1])),
        "chi_l2": float(np.max(analysis["total_l2"][:, 1])),
        "Phi_pointwise": float(np.max(analysis["pointwise_maximum"][:, 0])),
        "chi_pointwise": float(np.max(analysis["pointwise_maximum"][:, 1])),
        "corner_absolute": float(np.max(analysis["corner_absolute"])),
        "corner_Phi_absolute": float(np.max(analysis["corner_absolute"][:, :, 0])),
        "corner_chi_absolute": float(np.max(analysis["corner_absolute"][:, :, 1])),
        "maximum_proper_regional_method_absolute_fraction_difference": float(
            np.max(analysis["proper_method_absolute_difference"])
        ),
        "maximum_node_regional_method_absolute_fraction_difference": float(
            np.max(analysis["node_method_absolute_difference"])
        ),
    }
    return {
        "source": run["source"],
        "compact_grid_nodes": int(len(run["z"])),
        "compact_grid_intervals": int(len(run["z"]) - 1),
        "radial_cut": float(run["radius"]),
        "rk_boundary_enumeration_count": int(len(run["time"])),
        "accepted_step_count": int(np.max(run["step"])),
        "stage_position_reconstruction_maximum": float(
            run["stage_position_reconstruction_maximum"]
        ),
        "maxima_over_all_rk_boundary_enumerations": maxima,
        "maximum_combined_l2_enumeration": _stage_summary(analysis, peak),
    }


def _nondecreasing(values, tolerance=1e-12):
    values = np.asarray(values, dtype=float)
    return bool(np.all(np.diff(values) >= -float(tolerance)))


def _strictly_decreasing(values):
    return bool(np.all(np.diff(np.asarray(values, dtype=float)) < 0.0))


def source_grid_summary(analyses, summaries):
    result = {}
    for domain in DOMAINS:
        labels = [f"{grid}_{domain}" for grid in PRIMARY_GRIDS]
        peaks = [int(np.argmax(analyses[label]["combined_total_l2"])) for label in labels]
        maxima = {
            metric: [
                summaries[label]["maxima_over_all_rk_boundary_enumerations"][metric]
                for label in labels
            ] for metric in (
                "combined_l1", "combined_l2", "Phi_l1", "Phi_l2", "chi_l1",
                "chi_l2", "Phi_pointwise", "chi_pointwise", "corner_Phi_absolute",
            )
        }
        slopes = {
            metric: empirical_power_law(INTERVALS, values)
            for metric, values in maxima.items()
        }
        fixed_proper = {}
        for collar_index, width in enumerate(PROPER_COLLAR_WIDTHS):
            fixed_proper[str(width)] = {
                "combined_union_squared_l2_fraction_at_run_peak": [
                    float(analyses[label]["proper_union_squared_fraction"][peak, collar_index, 2])
                    for label, peak in zip(labels, peaks, strict=True)
                ],
                "combined_central_squared_l2_fraction_at_run_peak": [
                    float(analyses[label]["proper_central_squared_fraction"][peak, collar_index, 2])
                    for label, peak in zip(labels, peaks, strict=True)
                ],
                "primary_alternate_absolute_fraction_difference_at_run_peak": [
                    float(analyses[label]["proper_method_absolute_difference"][peak, collar_index])
                    for label, peak in zip(labels, peaks, strict=True)
                ],
            }
        fixed_node = {}
        for collar_index, count in enumerate(NODE_COLLAR_COUNTS):
            fractions = [
                float(analyses[label]["node_union_squared_fraction"][peak, collar_index, 2])
                for label, peak in zip(labels, peaks, strict=True)
            ]
            lower_spans = [
                float(analyses[label]["node_lower_span"][peak, collar_index])
                for label, peak in zip(labels, peaks, strict=True)
            ]
            upper_spans = [
                float(analyses[label]["node_upper_span"][peak, collar_index])
                for label, peak in zip(labels, peaks, strict=True)
            ]
            fixed_node[str(count)] = {
                "combined_union_squared_l2_fraction_at_run_peak": fractions,
                "lower_proper_span_at_run_peak": lower_spans,
                "upper_proper_span_at_run_peak": upper_spans,
                "fraction_non_decreasing": _nondecreasing(fractions),
                "lower_span_strictly_decreasing": _strictly_decreasing(lower_spans),
                "upper_span_strictly_decreasing": _strictly_decreasing(upper_spans),
                "primary_alternate_absolute_fraction_difference_at_run_peak": [
                    float(analyses[label]["node_method_absolute_difference"][peak, collar_index])
                    for label, peak in zip(labels, peaks, strict=True)
                ],
            }
        phi_fractions = [
            float(analyses[label]["phi_squared_l2_fraction"][peak])
            for label, peak in zip(labels, peaks, strict=True)
        ]
        three = fixed_node["3"]
        ten = fixed_node["10"]
        central_g10 = fixed_proper["0.2"][
            "combined_central_squared_l2_fraction_at_run_peak"
        ][-1]
        corner_g9 = maxima["corner_Phi_absolute"][-2]
        corner_g10 = maxima["corner_Phi_absolute"][-1]
        shrinking_indicators = {
            "Phi_fraction_all_at_least_0p99": bool(min(phi_fractions) >= 0.99),
            "three_node_fraction_non_decreasing": three["fraction_non_decreasing"],
            "three_node_spans_shrink": bool(
                three["lower_span_strictly_decreasing"]
                and three["upper_span_strictly_decreasing"]
            ),
            "ten_node_fraction_non_decreasing": ten["fraction_non_decreasing"],
            "ten_node_spans_shrink": bool(
                ten["lower_span_strictly_decreasing"]
                and ten["upper_span_strictly_decreasing"]
            ),
            "G10_fixed_0p20_central_fraction_below_0p25": bool(central_g10 < 0.25),
            "G10_corner_Phi_defect_not_below_G9": bool(corner_g10 >= corner_g9),
            "regional_method_difference_at_peak_below_0p05": bool(max(
                fixed_proper["0.2"][
                    "primary_alternate_absolute_fraction_difference_at_run_peak"
                ][-2:]
                + three[
                    "primary_alternate_absolute_fraction_difference_at_run_peak"
                ][-2:]
                + ten[
                    "primary_alternate_absolute_fraction_difference_at_run_peak"
                ][-2:]
            ) < 0.05),
        }
        fixed_physical_indicators = {
            "G9_G10_fixed_0p10_fraction_close_0p05": bool(
                abs(
                    fixed_proper["0.1"][
                        "combined_union_squared_l2_fraction_at_run_peak"
                    ][-1]
                    - fixed_proper["0.1"][
                        "combined_union_squared_l2_fraction_at_run_peak"
                    ][-2]
                ) < 0.05
            ),
            "G9_G10_three_node_decline_exceeds_method_difference": bool(
                three["combined_union_squared_l2_fraction_at_run_peak"][-2]
                - three["combined_union_squared_l2_fraction_at_run_peak"][-1]
                > max(
                    three[
                        "primary_alternate_absolute_fraction_difference_at_run_peak"
                    ][-2:]
                )
            ),
        }
        if all(shrinking_indicators.values()):
            category = "corner_layer_supported_not_proven"
        elif all(fixed_physical_indicators.values()):
            category = "fixed_physical_wall_layer_supported"
        elif central_g10 >= 0.25:
            category = "face_wide_operator_defect_supported"
        else:
            category = "mixed_localization"
        result[domain] = {
            "grid_order": list(PRIMARY_GRIDS),
            "compact_intervals": list(INTERVALS),
            "run_peak_array_indices": peaks,
            "maxima": maxima,
            "descriptive_log_log_slopes_not_convergence_orders": slopes,
            "phi_squared_l2_fraction_at_run_peak": phi_fractions,
            "fixed_proper_collars": fixed_proper,
            "fixed_node_collars": fixed_node,
            "shrinking_corner_indicators": shrinking_indicators,
            "fixed_physical_width_indicators": fixed_physical_indicators,
            "category": category,
        }
    categories = [result[domain]["category"] for domain in DOMAINS]
    if categories.count("corner_layer_supported_not_proven") >= 2:
        overall = "corner_layer_supported_not_proven"
    elif len(set(categories)) == 1:
        overall = categories[0]
    else:
        overall = "mixed_localization"
    return {"domains": result, "overall_category": overall}


def _relative_difference(left, right):
    return float(abs(float(left) - float(right)) / max(abs(float(left)), abs(float(right)), 1e-300))


def control_transfer_summary(analyses, summaries):
    pairs = (
        ("G9_R10", "Z9_R10", "fixed_radial_G9_Z9"),
        ("G10_R10", "Z10_R10", "fixed_radial_G10_Z10"),
        ("G10_R10", "G10H_R10", "half_timestep"),
    )
    records = {}
    for primary, control, name in pairs:
        left_peak = int(np.argmax(analyses[primary]["combined_total_l2"]))
        right_peak = int(np.argmax(analyses[control]["combined_total_l2"]))
        left_max = summaries[primary]["maxima_over_all_rk_boundary_enumerations"]
        right_max = summaries[control]["maxima_over_all_rk_boundary_enumerations"]
        metrics = {}
        for metric in (
            "combined_l2", "Phi_l2", "chi_l2", "Phi_pointwise",
            "corner_Phi_absolute",
        ):
            metrics[metric] = {
                "primary": left_max[metric],
                "control": right_max[metric],
                "symmetric_relative_difference": _relative_difference(
                    left_max[metric], right_max[metric],
                ),
            }
        metrics["three_node_union_fraction_at_run_peak"] = {
            "primary": float(analyses[primary]["node_union_squared_fraction"][left_peak, 1, 2]),
            "control": float(analyses[control]["node_union_squared_fraction"][right_peak, 1, 2]),
        }
        metrics["fixed_0p10_union_fraction_at_run_peak"] = {
            "primary": float(analyses[primary]["proper_union_squared_fraction"][left_peak, 1, 2]),
            "control": float(analyses[control]["proper_union_squared_fraction"][right_peak, 1, 2]),
        }
        aligned = {}
        series = {
            "combined_l2": (
                analyses[primary]["combined_total_l2"],
                analyses[control]["combined_total_l2"],
            ),
            "Phi_l2": (
                analyses[primary]["total_l2"][:, 0],
                analyses[control]["total_l2"][:, 0],
            ),
            "chi_l2": (
                analyses[primary]["total_l2"][:, 1],
                analyses[control]["total_l2"][:, 1],
            ),
            "Phi_pointwise": (
                analyses[primary]["pointwise_maximum"][:, 0],
                analyses[control]["pointwise_maximum"][:, 0],
            ),
            "corner_Phi_absolute": (
                np.max(analyses[primary]["corner_absolute"][:, :, 0], axis=1),
                np.max(analyses[control]["corner_absolute"][:, :, 0], axis=1),
            ),
            "three_node_union_fraction": (
                analyses[primary]["node_union_squared_fraction"][:, 1, 2],
                analyses[control]["node_union_squared_fraction"][:, 1, 2],
            ),
            "fixed_0p10_union_fraction": (
                analyses[primary]["proper_union_squared_fraction"][:, 1, 2],
                analyses[control]["proper_union_squared_fraction"][:, 1, 2],
            ),
        }
        primary_times = np.asarray(analyses[primary]["time"])
        control_times = np.asarray(analyses[control]["time"])
        matches = []
        for item, when in enumerate(primary_times):
            found = np.flatnonzero(np.isclose(control_times, when, rtol=0.0, atol=1e-15))
            if len(found) != 1:
                raise RuntimeError(f"{name} does not have unique coincident RK times")
            matches.append((item, int(found[0])))
        primary_combined = np.asarray([
            analyses[primary]["combined_total_l2"][i] for i, _ in matches
        ])
        control_combined = np.asarray([
            analyses[control]["combined_total_l2"][j] for _, j in matches
        ])
        activity_scale = float(max(np.max(primary_combined), np.max(control_combined)))
        active = np.maximum(primary_combined, control_combined) >= 1e-3 * activity_scale
        for metric, (left_values, right_values) in series.items():
            left = np.asarray([left_values[i] for i, _ in matches], dtype=float)
            right = np.asarray([right_values[j] for _, j in matches], dtype=float)
            scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1e-300)
            aligned[metric] = {
                "coincident_count": len(matches),
                "active_count_above_0p1_percent_combined_l2_max": int(np.sum(active)),
                "maximum_absolute_difference": float(np.max(np.abs(left - right))),
                "maximum_symmetric_relative_difference": float(
                    np.max(np.abs(left - right) / scale)
                ),
                "active_maximum_symmetric_relative_difference": float(
                    np.max(np.abs(left[active] - right[active]) / scale[active])
                ),
                "run_maximum_absolute_difference": float(
                    abs(float(np.max(left)) - float(np.max(right)))
                ),
            }
        records[name] = {
            "primary": primary,
            "control": control,
            "independent_maxima_and_own_peak_summary": True,
            "metrics": metrics,
            "coincident_RK_boundary_evaluation_summary": aligned,
        }
    return records


def state_arrays(analyses):
    arrays = {
        "proper_collar_widths": np.asarray(PROPER_COLLAR_WIDTHS),
        "node_collar_counts": np.asarray(NODE_COLLAR_COUNTS),
        "component_order": np.asarray((*COMPONENTS, "combined")),
        "wall_order": np.asarray(("lower", "upper")),
    }
    for label, analysis in analyses.items():
        for key, value in analysis.items():
            arrays[f"{label}_{key}"] = np.asarray(value)
    return arrays


def input_hashes(runs):
    paths = {PROTOCOL, TEST10E_RESULT, TEST10C_RESULT, Path(__file__), Path(
        "src/bhps/corrected_A790_test10e_boundary_corner_localization.py"
    )}
    for run in runs.values():
        paths.update(run["input_paths"])
    return {str(path): sha256_file(path) for path in sorted(paths, key=str)}


def run(output=OUTPUT, state_output=STATE_OUTPUT):
    controls = manufactured_controls()
    if not controls["passed"]:
        raise RuntimeError(f"localization controls failed: {controls['gates']}")

    runs = {}
    for grid in ("G7", "G8"):
        for domain in DOMAINS:
            label = f"{grid}_{domain}"
            runs[label] = _historical_run(label)
    for grid in ("G9", "G10"):
        for domain in DOMAINS:
            label = f"{grid}_{domain}"
            runs[label] = _test10e_run(label)
    for label in CONTROL_LABELS:
        runs[label] = _test10e_run(label)

    analyses = {label: analyze_run(item) for label, item in runs.items()}
    summaries = {
        label: summarize_run(runs[label], analyses[label]) for label in runs
    }
    source = source_grid_summary(analyses, summaries)
    transfer = control_transfer_summary(analyses, summaries)
    hashes = input_hashes(runs)

    payload = {
        "status": "complete_archive_localization",
        "classification": source["overall_category"],
        "sealed_test10e_regraded": False,
        "protocol": str(PROTOCOL),
        "protocol_sha256": hashes[str(PROTOCOL)],
        "scope": {
            "archive_only": True,
            "new_elliptic_solves": 0,
            "new_evolution_steps": 0,
            "new_surface_searches": 0,
            "boundary_quantity_interpretation": (
                "numerical scalar acceleration-row overwrite; not physical flux or transport"
            ),
            "enumeration_note": (
                "first array axis is RK boundary evaluations, two per accepted step"
            ),
            "proper_coordinate_note": (
                "proper distance is first-open-node to last-open-node; wall endpoint intervals are not archived"
            ),
            "compatibility_note": (
                "discrete compact-wall residual change induced by the subsequent open-face overwrite; not a continuum corner equation test"
            ),
        },
        "controls": controls,
        "source_grid_localization": source,
        "fixed_radial_and_temporal_controls": transfer,
        "run_summaries": summaries,
        "limitations": [
            "The G7/G8 face metrics are reconstructed from archived accepted states and velocities; no RHS is rerun.",
            "The open-face archives omit compact-wall endpoints, so wall-to-collar proper distance is unavailable.",
            "The compact-wall diagnostic is an overwrite-induced discrete residual defect, not a continuum commutator.",
            "G9/G10 is a two-grid transfer comparison, not an observed convergence order.",
            "Distinct source-grid parents make fitted slopes descriptive rather than convergence orders.",
            "Localization cannot validate the current boundary model or revive transport claims.",
        ],
        "input_sha256": hashes,
        "state_output": str(state_output),
    }
    atomic_write_npz(state_output, **state_arrays(analyses))
    atomic_write_json(output, _jsonable(payload))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--state-output", type=Path, default=STATE_OUTPUT)
    args = parser.parse_args()
    payload = run(args.output, args.state_output)
    print(json.dumps({
        "status": payload["status"],
        "classification": payload["classification"],
        "output": str(args.output),
        "state_output": str(args.state_output),
    }, indent=2))


if __name__ == "__main__":
    main()
