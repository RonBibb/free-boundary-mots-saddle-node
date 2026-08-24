#!/usr/bin/env python3
"""Run the sealed bounded initial no-horizon certificate (note 91)."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.anisotropic_capped_surface import (
    _splines,
    anisotropic_rho_second,
    find_anisotropic_donor_capped_surfaces,
)
from bhps.capped_surface_barrier_certificate import (
    BilinearMetricEnclosure,
    Interval,
    ParameterBox,
    cover_summary,
    initial_parameter_boxes,
    point_barrier_from_splines,
    process_cover_chunk,
)
from bhps.recovery_indexer import (
    RecoveryIndex,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
    validate_npz,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry


PROTOCOL = Path("notes/91_A790_initial_no_horizon_certificate_protocol.md")
OUTPUT = Path("results/corrected_A790_initial_no_horizon_certificate.json")
MANIFEST = Path("results/corrected_A790_initial_no_horizon_certificate_recovery.json")
RECOVERY = Path("results/corrected_A790_initial_no_horizon_certificate_stages")
GEOMETRY_PATHS = {
    "G9": RECOVERY / "A790_G9_metric.npz",
    "G10": RECOVERY / "A790_G10_metric.npz",
    "A794_G7": RECOVERY / "A794_G7_metric.npz",
}
RHO_BOUNDS = (0.10, 1.67)
THETA_COUNT = 64
RHO_COUNT = 64
CHUNK_EVALUATIONS = 512
MAXIMUM_DEPTH = 20
MAXIMUM_TERMINAL_BOXES = 2_000_000
THRESHOLD = 1e-10
BICUBIC_COUNT = 2049
SEEDS = tuple(np.arange(1.15, 1.7001, 0.05))
EXPECTED_INPUT_PATHS = (
    Path("results/corrected_family_knot_A8_state.npz"),
    Path("results/corrected_anisotropic_arclength_G6.json"),
    Path("run_corrected_fold_regular_so3_runtime.py"),
    Path("run_corrected_fold_nonlinear_rhs_G7_axis_refinement.py"),
    Path("src/bhps/anisotropic_capped_surface.py"),
    Path("src/bhps/capped_surface_barrier_certificate.py"),
)


def _json_stage(index, stage_id, kind, output, seconds, compute, metadata=None):
    index.register(stage_id, kind, seconds, metadata or {})
    validated = index.validated_path(stage_id)
    if validated is not None:
        return json.loads(validated.read_text())
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        payload = compute()
        atomic_write_json(output, payload)
        json.loads(output.read_text())
        elapsed = time.perf_counter() - started
        index.mark_complete(stage_id, output, elapsed)
        return payload
    except Exception as error:
        index.mark_failed(stage_id, repr(error))
        raise


def _npz_stage(index, stage_id, kind, output, seconds, compute, metadata=None):
    index.register(stage_id, kind, seconds, metadata or {})
    validated = index.validated_path(stage_id)
    if validated is not None:
        validate_npz(validated)
        return validated
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        arrays = compute()
        atomic_write_npz(output, **arrays)
        validate_npz(output)
        elapsed = time.perf_counter() - started
        index.mark_complete(stage_id, output, elapsed)
        return output
    except Exception as error:
        index.mark_failed(stage_id, repr(error))
        raise


def _archive_geometry(geometry):
    return {
        "z": np.asarray(geometry["z"]),
        "r": np.asarray(geometry["r"]),
        "psi": np.asarray(geometry["psi"]),
        "a": np.asarray(geometry["a"]),
        "b": np.asarray(geometry["b"]),
        "c": np.asarray(geometry["c"]),
        "phi": np.asarray(geometry["phi"]),
        "A": np.asarray(geometry["psi"]) * np.exp(np.asarray(geometry["a"])),
        "B": np.asarray(geometry["psi"]) * np.exp(np.asarray(geometry["b"])),
        "C": np.asarray(geometry["psi"]) * np.exp(np.asarray(geometry["c"])),
        "fold_amplitude": np.asarray(float(geometry["fold_amplitude"])),
        "selector_maximum": np.asarray(float(geometry["selector_maximum"])),
    }


def _load_geometry(path):
    with np.load(path) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _build_A790_G9():
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": 7.90}
    g7 = build_refined(
        seed, 81, 121, "G7A790-note91-parent",
        selector_iterations=40, slice_iterations=270,
    )
    g8 = build_refined(
        g7, 97, 145, "G8A790-note91-parent",
        selector_iterations=45, slice_iterations=280,
    )
    g9 = build_refined(
        g8, 113, 169, "G9A790-note91",
        selector_iterations=50, slice_iterations=300,
    )
    return _archive_geometry(g9)


def _build_A790_G10():
    g9 = _load_geometry(GEOMETRY_PATHS["G9"])
    g10 = build_refined(
        g9, 129, 193, "G10A790-note91",
        selector_iterations=55, slice_iterations=320,
    )
    return _archive_geometry(g10)


def _build_A794_G7():
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": 7.94}
    g7 = build_refined(
        seed, 81, 121, "G7A794-note91-control",
        selector_iterations=40, slice_iterations=270,
    )
    return _archive_geometry(g7)


def _metric(geometry):
    return BilinearMetricEnclosure(
        geometry["z"], geometry["r"],
        geometry["A"], geometry["B"], geometry["C"],
    )


def analytic_controls():
    flat = BilinearMetricEnclosure.flat(
        np.linspace(0.8, 3.0, 9), np.linspace(0.0, 2.0, 11),
    )
    queue = initial_parameter_boxes(16, 16, RHO_BOUNDS)
    certified = []
    nonpositive = []
    unresolved = []
    evaluations = 0
    while queue:
        state = process_cover_chunk(
            flat, queue, certified, nonpositive, unresolved,
            maximum_evaluations=CHUNK_EVALUATIONS,
            threshold=THRESHOLD, maximum_depth=MAXIMUM_DEPTH,
            maximum_terminal_boxes=MAXIMUM_TERMINAL_BOXES,
        )
        queue = state["queue"]
        certified = state["certified"]
        nonpositive = state["nonpositive"]
        unresolved = state["unresolved"]
        evaluations += state["evaluations"]
    flat_summary = cover_summary(
        queue, certified, nonpositive, unresolved,
        total_area=(math.pi / 2) * (RHO_BOUNDS[1] - RHO_BOUNDS[0]),
    )

    x = Interval(0.49, 0.51)
    tangent = (x - 0.5) ** 2
    positive = tangent + 0.01
    crossing = x - 0.5
    manufactured = {
        "tangent_interval": [tangent.lower, tangent.upper],
        "strictly_positive_interval": [positive.lower, positive.upper],
        "sign_changing_interval": [crossing.lower, crossing.upper],
    }
    passed = bool(
        flat_summary["complete"]
        and flat_summary["certified_area_fraction"] > 1.0 - 1e-14
        and flat_summary["nonpositive_box_count"] == 0
        and flat_summary["unresolved_box_count"] == 0
        and flat_summary["minimum_certified_lower_bound"] > 0.0
        and tangent.lower <= 0.0 <= tangent.upper
        and positive.lower > 0.0
        and crossing.lower <= 0.0 <= crossing.upper
    )
    return {
        "passed": passed,
        "flat_cover": flat_summary,
        "flat_cover_evaluations": evaluations,
        "manufactured_interval_controls": manufactured,
    }


def formula_equivalence(geometry):
    splines = _splines(
        geometry["z"], geometry["r"], geometry["psi"],
        geometry["a"], geometry["b"], geometry["c"],
    )
    theta_values = np.linspace(0.025, math.pi / 2, 31)
    rho_values = np.linspace(RHO_BOUNDS[0], RHO_BOUNDS[1], 37)
    theta, rho = np.meshgrid(theta_values, rho_values, indexing="ij")
    reduced = point_barrier_from_splines(
        theta, rho, geometry["z"][-1], splines,
    )
    direct = anisotropic_rho_second(
        theta, rho, np.zeros_like(rho), geometry["z"][-1], splines,
    ) / rho
    error = np.abs(reduced - direct)
    return {
        "maximum_absolute_error": float(np.max(error)),
        "rms_absolute_error": float(np.sqrt(np.mean(error**2))),
        "passed": bool(np.max(error) < 2e-10),
    }


def _empty_cover_state(boxes):
    return {
        "queue": [box.to_list() for box in boxes],
        "certified": [],
        "nonpositive": [],
        "unresolved": [],
        "evaluations_total": 0,
    }


def _refinement_boxes(records):
    boxes = []
    for record in records:
        box = ParameterBox.from_list(record[:5])
        boxes.extend(box.bisect(math.pi / 2, RHO_BOUNDS[1] - RHO_BOUNDS[0]))
    return boxes


def checkpointed_cover(index, label, metric, boxes, stop_on_nonpositive=False):
    state = _empty_cover_state(boxes)
    chunk = 0
    while state["queue"]:
        stage_id = f"cover/{label}/chunk-{chunk:05d}"
        path = RECOVERY / f"{label}_chunk_{chunk:05d}.json"

        def compute(state=state):
            advanced = process_cover_chunk(
                metric,
                [ParameterBox.from_list(item) for item in state["queue"]],
                state["certified"], state["nonpositive"], state["unresolved"],
                maximum_evaluations=CHUNK_EVALUATIONS,
                threshold=THRESHOLD, maximum_depth=MAXIMUM_DEPTH,
                maximum_terminal_boxes=MAXIMUM_TERMINAL_BOXES,
            )
            return {
                "queue": [box.to_list() for box in advanced["queue"]],
                "certified": advanced["certified"],
                "nonpositive": advanced["nonpositive"],
                "unresolved": advanced["unresolved"],
                "evaluations_total": (
                    state["evaluations_total"] + advanced["evaluations"]
                ),
            }

        state = _json_stage(
            index, stage_id, "interval-cover", path, 600.0, compute,
            {"label": label, "chunk": chunk, "maximum_evaluations": CHUNK_EVALUATIONS},
        )
        if chunk % 8 == 0:
            print(
                f"{label}: {state['evaluations_total']} boxes evaluated; "
                f"{len(state['queue'])} queued",
                flush=True,
            )
        if stop_on_nonpositive and state["nonpositive"]:
            break
        chunk += 1
    summary = cover_summary(
        [ParameterBox.from_list(item) for item in state["queue"]],
        state["certified"], state["nonpositive"], state["unresolved"],
        total_area=(math.pi / 2) * (RHO_BOUNDS[1] - RHO_BOUNDS[0]),
    )
    summary["evaluations"] = int(state["evaluations_total"])
    summary["chunk_count"] = int(chunk + 1)
    return state, summary


def bicubic_scan(index, label, geometry):
    splines = _splines(
        geometry["z"], geometry["r"], geometry["psi"],
        geometry["a"], geometry["b"], geometry["c"],
    )
    theta_values = np.linspace(0.0, math.pi / 2, BICUBIC_COUNT)
    rho_values = np.linspace(RHO_BOUNDS[0], RHO_BOUNDS[1], BICUBIC_COUNT)
    records = []
    row_chunk = 64
    for start in range(0, BICUBIC_COUNT, row_chunk):
        stop = min(start + row_chunk, BICUBIC_COUNT)
        stage_id = f"bicubic/{label}/rows-{start:04d}-{stop - 1:04d}"
        path = RECOVERY / f"{label}_bicubic_{start:04d}_{stop - 1:04d}.json"

        def compute(start=start, stop=stop):
            theta, rho = np.meshgrid(
                theta_values[start:stop], rho_values, indexing="ij",
            )
            values = point_barrier_from_splines(
                theta, rho, geometry["z"][-1], splines,
            )
            location = np.unravel_index(int(np.argmin(values)), values.shape)
            return {
                "row_start": start,
                "row_stop": stop,
                "minimum": float(values[location]),
                "maximum": float(np.max(values)),
                "minimum_theta": float(theta[location]),
                "minimum_rho": float(rho[location]),
                "finite": bool(np.all(np.isfinite(values))),
            }

        records.append(_json_stage(
            index, stage_id, "bicubic-representation-audit", path, 600.0,
            compute, {"label": label, "row_start": start, "row_stop": stop},
        ))
    minimum_record = min(records, key=lambda item: item["minimum"])
    return {
        "sample_shape": [BICUBIC_COUNT, BICUBIC_COUNT],
        "minimum": minimum_record["minimum"],
        "minimum_theta": minimum_record["minimum_theta"],
        "minimum_rho": minimum_record["minimum_rho"],
        "maximum": max(record["maximum"] for record in records),
        "all_finite": bool(all(record["finite"] for record in records)),
        "all_positive": bool(minimum_record["minimum"] > 0.0),
    }


def adverse_control(geometry, metric):
    queue = initial_parameter_boxes(64, 64, RHO_BOUNDS)
    certified = []
    nonpositive = []
    unresolved = []
    evaluations = 0
    while queue and not nonpositive:
        state = process_cover_chunk(
            metric, queue, certified, nonpositive, unresolved,
            maximum_evaluations=CHUNK_EVALUATIONS,
            threshold=THRESHOLD, maximum_depth=MAXIMUM_DEPTH,
            maximum_terminal_boxes=MAXIMUM_TERMINAL_BOXES,
        )
        queue = state["queue"]
        certified = state["certified"]
        nonpositive = state["nonpositive"]
        unresolved = state["unresolved"]
        evaluations += state["evaluations"]
    barrier = cover_summary(
        queue, certified, nonpositive, unresolved,
        total_area=(math.pi / 2) * (RHO_BOUNDS[1] - RHO_BOUNDS[0]),
    )
    barrier["evaluations"] = evaluations
    bvp = find_anisotropic_donor_capped_surfaces(
        geometry["z"], geometry["r"], geometry["psi"], geometry["a"],
        geometry["b"], geometry["c"], guesses=SEEDS, tolerance=2e-5,
    )
    signatures = sorted(
        ([item["rho_axis"], item["rho_brane"]] for item in bvp["accepted"]),
        key=lambda item: item[1],
    )
    return {
        "barrier": barrier,
        "barrier_did_not_certify": bool(
            nonpositive or unresolved or queue
        ),
        "bvp_accepted_count": len(signatures),
        "bvp_signatures": signatures,
        "passed": bool(
            (nonpositive or unresolved or queue) and len(signatures) == 2
        ),
    }


def main():
    RECOVERY.mkdir(parents=True, exist_ok=True)
    expected_inputs = {
        str(path): sha256_file(path) for path in EXPECTED_INPUT_PATHS
    }
    index = RecoveryIndex(
        MANIFEST, PROTOCOL, expected_inputs, maximum_stage_seconds=3600.0,
    )

    print("note91 controls", flush=True)
    controls = _json_stage(
        index, "controls/analytic", "analytic-controls",
        RECOVERY / "analytic_controls.json", 300.0, analytic_controls,
    )

    print("constructing or validating A=7.90 G9", flush=True)
    _npz_stage(
        index, "geometry/A790/G9", "initial-geometry",
        GEOMETRY_PATHS["G9"], 3600.0, _build_A790_G9,
        {"amplitude": 7.90, "shape": [113, 169]},
    )
    print("constructing or validating A=7.90 G10", flush=True)
    _npz_stage(
        index, "geometry/A790/G10", "initial-geometry",
        GEOMETRY_PATHS["G10"], 3600.0, _build_A790_G10,
        {"amplitude": 7.90, "shape": [129, 193]},
    )
    geometries = {
        label: _load_geometry(GEOMETRY_PATHS[label]) for label in ("G9", "G10")
    }

    formula = {}
    primary_states = {}
    primary = {}
    refined = {}
    bicubic = {}
    for label in ("G9", "G10"):
        formula[label] = _json_stage(
            index, f"controls/formula/{label}", "formula-equivalence",
            RECOVERY / f"{label}_formula_equivalence.json", 300.0,
            lambda label=label: formula_equivalence(geometries[label]),
            {"grid": label},
        )
        metric = _metric(geometries[label])
        primary_states[label], primary[label] = checkpointed_cover(
            index, f"{label}_primary", metric,
            initial_parameter_boxes(THETA_COUNT, RHO_COUNT, RHO_BOUNDS),
            stop_on_nonpositive=True,
        )
        if (
            primary[label]["complete"]
            and primary[label]["nonpositive_box_count"] == 0
            and primary[label]["unresolved_box_count"] == 0
        ):
            _, refined[label] = checkpointed_cover(
                index, f"{label}_refined", metric,
                _refinement_boxes(primary_states[label]["certified"]),
            )
        else:
            refined[label] = {
                "complete": False, "skipped": True,
                "reason": "primary cover was not completely positive",
            }
        bicubic[label] = bicubic_scan(index, label, geometries[label])

    print("constructing or validating A=7.94 adverse control", flush=True)
    _npz_stage(
        index, "geometry/A794/G7", "control-geometry",
        GEOMETRY_PATHS["A794_G7"], 3600.0, _build_A794_G7,
        {"amplitude": 7.94, "shape": [81, 121]},
    )
    adverse_geometry = _load_geometry(GEOMETRY_PATHS["A794_G7"])
    adverse = _json_stage(
        index, "controls/A794", "adverse-physical-control",
        RECOVERY / "A794_adverse_control.json", 3600.0,
        lambda: adverse_control(adverse_geometry, _metric(adverse_geometry)),
    )

    construction = {
        label: {
            "shape": [len(geometry["z"]), len(geometry["r"])],
            "z_spacing": float(np.diff(geometry["z"])[0]),
            "r_spacing": float(np.diff(geometry["r"])[0]),
            "selector_maximum": float(geometry["selector_maximum"]),
            "finite": bool(all(np.all(np.isfinite(geometry[key])) for key in (
                "psi", "a", "b", "c", "A", "B", "C",
            ))),
        } for label, geometry in geometries.items()
    }
    construction_pass = bool(
        construction["G9"]["shape"] == [113, 169]
        and construction["G10"]["shape"] == [129, 193]
        and all(item["finite"] and item["selector_maximum"] < 1e-9
                for item in construction.values())
    )
    cover_pass = bool(all(
        primary[label].get("complete", False)
        and primary[label].get("certified_area_fraction", 0.0) > 1.0 - 1e-14
        and primary[label].get("nonpositive_box_count", 1) == 0
        and primary[label].get("unresolved_box_count", 1) == 0
        and primary[label].get("minimum_certified_lower_bound", -1.0) > THRESHOLD
        and refined[label].get("complete", False)
        and refined[label].get("certified_area_fraction", 0.0) > 1.0 - 1e-14
        and refined[label].get("nonpositive_box_count", 1) == 0
        and refined[label].get("unresolved_box_count", 1) == 0
        and refined[label].get("minimum_certified_lower_bound", -1.0) > THRESHOLD
        and bicubic[label]["all_finite"]
        and bicubic[label]["all_positive"]
        for label in ("G9", "G10")
    ))
    controls_pass = bool(
        controls["passed"] and adverse["passed"]
        and all(formula[label]["passed"] for label in ("G9", "G10"))
    )
    if controls_pass and construction_pass and cover_pass:
        status = "PASS"
        classification = "bounded_discrete_initial_no_horizon_certificate"
    else:
        status = "REVIEW"
        classification = "bounded_initial_no_horizon_certificate_unresolved"

    result = {
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "status": status,
        "classification": classification,
        "surface_class": {
            "topology": "axisymmetric star-shaped donor-capped half-S3",
            "theta_bounds": [0.0, math.pi / 2],
            "rho_bounds": list(RHO_BOUNDS),
            "regularity": "C2 with rho'(0)=rho'(pi/2)=0",
        },
        "method": "maximum-principle positive constant-cap barrier",
        "discrete_geometry": "piecewise-bilinear G9 and G10 nodal metrics",
        "analytic_controls": controls,
        "formula_equivalence": formula,
        "construction": construction,
        "primary_interval_cover": primary,
        "mandatory_refined_interval_cover": refined,
        "bicubic_representation_audit": bicubic,
        "A794_adverse_control": adverse,
        "gates": {
            "controls": controls_pass,
            "construction": construction_pass,
            "two_grid_interval_and_representation_cover": cover_pass,
        },
        "claim_boundary": (
            "Excludes only C2 axisymmetric star-shaped donor-capped minimal "
            "graphs with 0.10<=rho<=1.67 in the discrete G9/G10 initial "
            "metrics. It is not absence of arbitrary apparent horizons or a "
            "continuum-spacetime theorem."
        ),
        "provenance": {
            "manifest": str(MANIFEST),
            "manifest_sha256_before_final_write": sha256_file(MANIFEST),
            "input_sha256": expected_inputs,
        },
    }
    atomic_write_json(OUTPUT, result)
    print(json.dumps({
        "status": status,
        "classification": classification,
        "gates": result["gates"],
        "primary_minima": {
            label: primary[label]["minimum_certified_lower_bound"]
            for label in ("G9", "G10")
        },
        "bicubic_minima": {
            label: bicubic[label]["minimum"] for label in ("G9", "G10")
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
