#!/usr/bin/env python3
"""Sealed replay audit for the Test-10B outer scalar correction."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import run_corrected_A790_test10b_domain_normalized as test10b
import run_corrected_fold_live_nonlinear_gauge_source as live
from bhps.corrected_A790_test10c_outer_scalar import (
    classify_test10c,
    correction_metrics,
    endpoint_derivative,
    independent_characteristic_terms,
    normalized_radial_difference,
)
from bhps.recovery_indexer import (
    RecoveryIndex,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
    validate_npz,
)
from bhps.nonlinear_regular_so3_evolution import (
    apply_outer_sommerfeld_acceleration,
)


PROTOCOL = Path("notes/103_A790_test10C_outer_scalar_closure_audit_protocol.md")
OUTPUT = Path("results/corrected_A790_test10c_outer_scalar_closure.json")
STATE_OUTPUT = Path("results/corrected_A790_test10c_outer_scalar_closure_state.npz")
RECOVERY_ROOT = Path("results/corrected_A790_test10c_outer_scalar_closure_recovery")
MANIFEST = RECOVERY_ROOT / "index.json"
RUNS = tuple(
    test10b.run_label(grid, domain)
    for grid in test10b.GRIDS for domain in test10b.DOMAINS
)

FIXED_INPUT_HASHES = {
    "notes/98_A790_domain_normalized_family_protocol.md": "7c0298c67d03654b2e24c7c8602a77a8c70bd75beba14645fdd76adfef11cc23",
    "notes/98_A790_domain_normalized_family_result.md": "618b72fb125ddc67c8ea523789687a4836216fd52bda26e59223c8163a4e42d0",
    "results/corrected_A790_test10b_domain_normalized.json": "08c91892930c05fef6afe37d0a4f5ed1c49f545a84b6df4c81e08c8f2cbd1bd3",
    "results/corrected_A790_test10b_domain_normalized_state.npz": "b19e7e8901fdade6abb485c65a3512bf327a86f13974fb9a8c0876340a503928",
    "results/corrected_A790_test10b_domain_normalized_recovery/index.json": "7d064cfc491a0d2772c051cc1d10985ea936d320f26651a9423fa5ddb9947a0b",
    "src/bhps/recovery_indexer.py": "1460478fba42433bd340a2ef9e09c0946882a35d3eb63c2c95ea9b055bb549fa",
    "src/bhps/corrected_A790_R12_builder.py": "a0d0a5e7c12fef5bdacb2c97710787266c5ed3beba5a435dd7653edaf88322cc",
    "src/bhps/corrected_A790_test10b_domain_normalized.py": "67a44f7b4ae50d505f57f61d7fa4c33c485738d938d924291eb0296ffc94a62e",
    "run_corrected_A790_test10b_domain_normalized.py": "37ce44ba92af732a0c18202fd756286e3e5b935a75e0864d04fd9f140e32664b",
    "run_corrected_fold_live_nonlinear_gauge_source.py": "b886bf79d57f98b372d8f756d22016f56192d1816b893536e4f6fd5ac242c203",
    "src/bhps/nonlinear_regular_so3_evolution.py": "b40c2dc89a6e5958d7365876a8d0e95d691e66a2dee100bccc1ef234d381c161",
}


def recovery_inputs():
    dynamic = (
        Path(__file__),
        Path("src/bhps/corrected_A790_test10c_outer_scalar.py"),
        Path("tests/test_A790_test10c_outer_scalar.py"),
        Path("tests/test_A790_test10c_runner_recovery.py"),
    )
    return {
        **FIXED_INPUT_HASHES,
        **{str(path): sha256_file(path) for path in dynamic},
    }


def relative_array_error(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return float(np.linalg.norm(left - right) / max(
        np.linalg.norm(left), np.linalg.norm(right), 1e-300,
    ))


def validate_test10b_archive():
    index = RecoveryIndex(
        test10b.MANIFEST, test10b.PROTOCOL, test10b.recovery_inputs(),
        maximum_stage_seconds=2400.0,
    )
    invalid = [
        stage_id for stage_id, stage in index.data["stages"].items()
        if stage.get("status") != "complete" or index.validated_path(stage_id) is None
    ]
    if len(index.data["stages"]) != 160 or invalid:
        raise RuntimeError(f"invalid Test-10B source archive: {invalid}")
    return {"stage_count": 160, "invalid": invalid}


def initial_segment_state(case, start):
    if start == 0:
        return (
            case["initial"].copy(), np.zeros_like(case["initial"]),
            case["source0"].copy(), case["memory0"].copy(),
        )
    previous = test10b.segment_path(case["_test10c_label"], start - 4, start)
    with np.load(previous) as archive:
        return tuple(np.asarray(archive[key]) for key in (
            "end_position", "end_velocity", "end_source", "end_memory",
        ))


def instrument_stage(case, time_value, state):
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
    source_dot, memory_dot = live.source_driver_rhs(
        source, memory, target, live.DRIVER_MU, live.DRIVER_ETA, advection,
    )
    source_dot, _ = live.apply_outer_source_sommerfeld(
        source, source_dot, case["source0"], case["source_time0"],
        case["_initial_source_second_time"], position, time_value,
        case["r"], case["rhs"].stencil_width,
    )
    gauge = live.StageRegularGaugeSource(
        source, source_dot, case["z"], case["r"],
    )
    source_second = live.live_regular_source_second_time(
        position, velocity, case["initial"], case["source0"], source,
        source_dot, memory_dot, case["z"], case["r"], live.DRIVER_MU,
        live.TARGET_MU_LAPSE, live.TARGET_MU_SHIFT, live.TARGET_POWER,
    )
    outer_enabled = case["rhs"].live_outer_sommerfeld
    case["rhs"].live_outer_sommerfeld = False
    try:
        before, diagnostic = case["rhs"].acceleration(
            time_value, position, velocity, gauge, source_second,
        )
    finally:
        case["rhs"].live_outer_sommerfeld = outer_enabled
    after, production = apply_outer_sommerfeld_acceleration(
        position, velocity, before, case["rhs"].outer_reference_position,
        case["rhs"].outer_reference_acceleration, time_value, case["r"],
        case["rhs"].stencil_width,
    )
    terms = independent_characteristic_terms(
        position, velocity, before, case["rhs"].outer_reference_position,
        case["rhs"].outer_reference_acceleration, time_value, case["r"],
        case["rhs"].stencil_width,
    )
    metrics = correction_metrics(
        position, before, after, case["rhs"].outer_reference_acceleration,
        terms, case["z"], case["r"], case["rhs"].stencil_width,
    )
    metrics["production_legacy_ratio"] = float(
        production["scalar_relative_correction"]
    )
    metrics["production_residual"] = float(
        production["maximum_normalized_acceleration_residual"]
    )
    metrics["finite"] = bool(
        diagnostic["finite"] and np.all(np.isfinite(after))
        and np.all(np.isfinite(source_dot)) and np.all(np.isfinite(memory_dot))
    )
    raw = {
        "before": before[1:-1, -1, 7:9],
        "after": after[1:-1, -1, 7:9],
        "target": terms["target"][1:-1, 7:9],
        "term_A": terms["term_A"][1:-1, 7:9],
        "term_V": terms["term_V"][1:-1, 7:9],
        "term_C": terms["term_C"][1:-1, 7:9],
    }
    return (velocity, after, source_dot, memory_dot), metrics, raw


METRIC_KEYS = (
    "legacy_ratio", "legacy_numerator", "legacy_before_norm",
    "legacy_after_norm", "quadrature_relative_difference",
    "maximum_absolute_correction", "face_rms_correction", "face_rms_before",
    "face_rms_after", "face_rms_reference", "collar_rms_before",
    "collar_ratio", "independent_target_relative_difference",
    "independent_target_maximum_absolute_difference",
    "characteristic_closure_relative", "post_correction_relative_residual",
    "weight_minimum", "weight_maximum", "production_legacy_ratio",
    "production_residual",
)


def flatten_metrics(metrics):
    result = {key: float(metrics[key]) for key in METRIC_KEYS}
    result.update({
        "proper_simpson_ratio": float(metrics["proper"]["simpson"]["ratio"]),
        "proper_trapezoid_ratio": float(metrics["proper"]["trapezoid"]["ratio"]),
        "term_balance_simpson": float(
            metrics["proper"]["simpson"]["term_balance_ratio"]
        ),
        "term_balance_trapezoid": float(
            metrics["proper"]["trapezoid"]["term_balance_ratio"]
        ),
        "pointwise_ratio": float(metrics["pointwise"]["maximum"]),
        "pointwise_z_index": int(metrics["pointwise"]["index"][0]),
        "pointwise_scalar_index": int(metrics["pointwise"]["index"][1]),
        "finite": int(metrics["finite"]),
    })
    return result


FLAT_METRIC_KEYS = tuple(flatten_metrics({
    **{key: 0.0 for key in METRIC_KEYS},
    "proper": {
        "simpson": {"ratio": 0.0, "term_balance_ratio": 0.0},
        "trapezoid": {"ratio": 0.0, "term_balance_ratio": 0.0},
    },
    "pointwise": {"maximum": 0.0, "index": [0, 0]},
    "finite": True,
}).keys())


def checkpoint_path(label, start, end):
    return RECOVERY_ROOT / f"replay_{label}_steps_{start + 1:03d}_{end:03d}.npz"


def enumeration(run_index, step, rk_stage):
    return 1 + 32 * int(run_index) + 2 * (int(step) - 1) + (int(rk_stage) - 1)


def validate_checkpoint(path, open_count, count, enumeration_start, enumeration_end):
    required = {
        "enumeration": (count,), "step": (count,), "rk_stage": (count,),
        "time": (count,), "before": (count, open_count, 2),
        "after": (count, open_count, 2), "target": (count, open_count, 2),
        "term_A": (count, open_count, 2), "term_V": (count, open_count, 2),
        "term_C": (count, open_count, 2), "component_legacy": (count, 2),
        "component_proper": (count, 2), "step_replay_error": (count // 2, 4),
        "component_proper_trapezoid": (count, 2),
        **{key: (count,) for key in FLAT_METRIC_KEYS},
    }
    validate_npz(path, required)
    with np.load(path) as archive:
        values = np.asarray(archive["enumeration"], dtype=int)
        if values[0] != enumeration_start or values[-1] != enumeration_end:
            raise RuntimeError("checkpoint enumeration bounds mismatch")
        if not np.array_equal(values, np.arange(enumeration_start, enumeration_end + 1)):
            raise RuntimeError("checkpoint enumeration is not contiguous")


def replay_segment(index, run_index, label, case, start, end, parent_sha):
    source_path = test10b.segment_path(label, start, end)
    source_sha = sha256_file(source_path)
    enum_start = enumeration(run_index, start + 1, 1)
    enum_end = enumeration(run_index, end, 2)
    stage_id = f"replay/{label}/enumeration_{enum_start:03d}_{enum_end:03d}"
    path = checkpoint_path(label, start, end)
    metadata = {
        "run": label, "start": start, "end": end,
        "enumeration_start": enum_start, "enumeration_end": enum_end,
        "source_sha256": source_sha, "parent_sha256": parent_sha,
        "case_fingerprint": test10b.case_fingerprint(case),
    }
    index.register(stage_id, "outer-scalar-replay", 2400.0, metadata)
    cached = index.validated_path(stage_id)
    if cached is not None:
        validate_checkpoint(cached, len(case["z"]) - 2, 2 * (end - start), enum_start, enum_end)
        return cached
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        state = initial_segment_state(case, start)
        records = []
        replay_errors = []
        with np.load(source_path) as source:
            for step in range(start + 1, end + 1):
                current_time = (step - 1) * test10b.DT
                print(f"{label}: Test10C step {step}/16 stage 1", flush=True)
                k1, m1, raw1 = instrument_stage(case, current_time, state)
                midpoint = tuple(
                    value + 0.5 * test10b.DT * slope
                    for value, slope in zip(state, k1)
                )
                print(f"{label}: Test10C step {step}/16 stage 2", flush=True)
                k2, m2, raw2 = instrument_stage(
                    case, current_time + 0.5 * test10b.DT, midpoint,
                )
                state = tuple(
                    value + test10b.DT * slope
                    for value, slope in zip(state, k2)
                )
                expected = (
                    case["initial"] + source[f"step_{step:03d}_increment"],
                    source[f"step_{step:03d}_velocity"],
                    case["source0"] + source[f"step_{step:03d}_source_increment"],
                )
                errors = [
                    relative_array_error(state[i], expected[i]) for i in range(3)
                ]
                errors.append(0.0)
                replay_errors.append(errors)
                for rk_stage, stage_time, metrics, raw in (
                    (1, current_time, m1, raw1),
                    (2, current_time + 0.5 * test10b.DT, m2, raw2),
                ):
                    records.append({
                        "enumeration": enumeration(run_index, step, rk_stage),
                        "step": step, "rk_stage": rk_stage,
                        "time": stage_time, "metrics": metrics, "raw": raw,
                    })
            replay_errors[-1][3] = relative_array_error(state[3], source["end_memory"])
            endpoint_errors = [
                relative_array_error(state[i], source[key]) for i, key in enumerate((
                    "end_position", "end_velocity", "end_source", "end_memory",
                ))
            ]
        if max(max(row) for row in replay_errors) >= 1e-13 or max(endpoint_errors) >= 1e-13:
            raise RuntimeError("archived replay mismatch")
        arrays = {
            "enumeration": np.asarray([r["enumeration"] for r in records]),
            "step": np.asarray([r["step"] for r in records]),
            "rk_stage": np.asarray([r["rk_stage"] for r in records]),
            "time": np.asarray([r["time"] for r in records]),
            "component_legacy": np.asarray([
                r["metrics"]["component_legacy_ratios"] for r in records
            ]),
            "component_proper": np.asarray([
                r["metrics"]["component_proper_ratios"] for r in records
            ]),
            "component_proper_trapezoid": np.asarray([
                r["metrics"]["component_proper_trapezoid_ratios"] for r in records
            ]),
            "step_replay_error": np.asarray(replay_errors),
        }
        for raw_key in ("before", "after", "target", "term_A", "term_V", "term_C"):
            arrays[raw_key] = np.asarray([r["raw"][raw_key] for r in records])
        flattened = [flatten_metrics(r["metrics"]) for r in records]
        for key in FLAT_METRIC_KEYS:
            arrays[key] = np.asarray([record[key] for record in flattened])
        atomic_write_npz(path, **arrays)
        validate_checkpoint(path, len(case["z"]) - 2, len(records), enum_start, enum_end)
        index.mark_complete(
            stage_id, path, time.perf_counter() - started,
            {"endpoint_replay_errors": endpoint_errors},
        )
        return path
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def manufactured_controls():
    from bhps.corrected_A790_test10c_outer_scalar import correction_metrics

    z = np.linspace(1.0, 2.0, 65)
    r = np.linspace(0.0, 3.0, 49)
    q = np.zeros((len(z), len(r), 9))
    q[..., 2] = -1.0
    q[..., 3] = q[..., 6] = 1.0
    v = np.zeros_like(q)
    rr = r[None, :] / r[-1]
    q0 = q.copy()
    q[..., 7] = 0.02 * rr**3
    q[..., 8] = -0.01 * rr**2
    v[..., 7] = 0.03 * rr**4
    v[..., 8] = 0.015 * rr**3
    before = np.zeros_like(q)
    reference_acceleration = np.zeros_like(q)
    corrected, _ = apply_outer_sommerfeld_acceleration(
        q, v, before, q0, reference_acceleration, 0.2, r, 7,
    )
    terms = independent_characteristic_terms(
        q, v, before, q0, reference_acceleration, 0.2, r, 7,
    )
    independent_error = relative_array_error(
        corrected[1:-1, -1, 7:9], terms["target"][1:-1, 7:9],
    )
    target = terms["target"].copy()
    exact_before = before.copy()
    exact_before[:, -1] = target
    exact_after, _ = apply_outer_sommerfeld_acceleration(
        q, v, exact_before, q0, reference_acceleration, 0.2, r, 7,
    )
    exact_outgoing_error = relative_array_error(
        exact_after[1:-1, -1, 7:9], exact_before[1:-1, -1, 7:9],
    )
    polynomial = np.zeros((2, len(r), 1))
    polynomial[..., 0] = r[None, :] ** 5
    polynomial_error = float(np.max(np.abs(
        endpoint_derivative(polynomial, r, 7)[:, 0] - 5.0 * r[-1] ** 4
    )))
    base_before = np.zeros_like(q)
    base_after = np.zeros_like(q)
    base_before[..., 7:9] = 2.0
    base_after[..., 7:9] = 1.0
    base_terms = {
        "target": base_after[:, -1].copy(),
        "term_A": base_before[:, -1].copy(),
        "term_V": np.zeros_like(base_before[:, -1]),
        "term_C": -base_after[:, -1].copy(),
    }
    base = correction_metrics(
        q, base_before, base_after, base_before, base_terms, z, r,
    )
    scale_errors = []
    for scale in (1e-6, 1e6):
        found = correction_metrics(
            q, base_before * scale, base_after * scale, base_before * scale,
            {key: value * scale for key, value in base_terms.items()}, z, r,
        )
        scale_errors.extend((
            abs(found["legacy_ratio"] - base["legacy_ratio"]),
            abs(found["proper"]["simpson"]["ratio"] - base["proper"]["simpson"]["ratio"]),
            abs(found["collar_ratio"] - base["collar_ratio"]),
        ))
    hidden_before = base_before.copy()
    hidden_after = base_before.copy()
    middle = len(z) // 2
    hidden_before[middle, -1, 7] = 0.1
    hidden_after[middle, -1, 7] = 1.1
    hidden_terms = {
        "target": hidden_after[:, -1].copy(),
        "term_A": hidden_before[:, -1] - hidden_after[:, -1],
        "term_V": np.zeros_like(hidden_before[:, -1]),
        "term_C": np.zeros_like(hidden_before[:, -1]),
    }
    hidden = correction_metrics(
        q, hidden_before, hidden_after, hidden_before, hidden_terms, z, r,
    )
    rejected_nonpositive = False
    broken = q.copy()
    broken[5, -1, 3] = -1.0
    try:
        correction_metrics(
            broken, base_before, base_after, base_before, base_terms, z, r,
        )
    except ValueError:
        rejected_nonpositive = True
    gates = {
        "independent_target": independent_error < 1e-10,
        "exact_outgoing": exact_outgoing_error < 1e-11,
        "polynomial_weights": polynomial_error < 1e-9,
        "scale_invariance": bool(max(scale_errors) < 1e-10),
        "global_hiding_exposed": bool(
            hidden["legacy_ratio"] < 0.05 and hidden["pointwise"]["maximum"] > 0.8
        ),
        "positive_measure": rejected_nonpositive,
        "quadrature": base["quadrature_relative_difference"] < 0.02,
        "uniform_ratio": bool(
            abs(base["legacy_ratio"] - 0.5) < 1e-12
            and abs(base["proper"]["simpson"]["ratio"] - 0.5) < 1e-12
        ),
    }
    return {
        "passed": bool(all(gates.values())), "gates": gates,
        "independent_target_error": independent_error,
        "exact_outgoing_error": exact_outgoing_error,
        "polynomial_error": polynomial_error,
        "maximum_scale_error": float(max(scale_errors)),
        "hidden_global_ratio": hidden["legacy_ratio"],
        "hidden_pointwise_ratio": hidden["pointwise"]["maximum"],
        "quadrature_relative_difference": base["quadrature_relative_difference"],
    }


def json_stage(index, stage_id, path, producer):
    index.register(stage_id, "analysis", 2400.0, {})
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


def replay_analysis(paths):
    immutable = json.loads(test10b.OUTPUT.read_text())
    runs = {}
    independent_maxima = {
        "target_relative": 0.0, "target_absolute": 0.0,
        "characteristic_closure": 0.0, "post_residual": 0.0,
        "production_residual": 0.0, "quadrature": 0.0,
    }
    all_replay = True
    all_finite = True
    normalization_all = True
    robust_grids = set()
    for label in RUNS:
        stage_records = []
        for path in paths[label]:
            with np.load(path) as archive:
                all_replay = bool(all_replay and np.max(archive["step_replay_error"]) < 1e-13)
                for index in range(len(archive["enumeration"])):
                    record = {
                        "enumeration": int(archive["enumeration"][index]),
                        "step": int(archive["step"][index]),
                        "rk_stage": int(archive["rk_stage"][index]),
                        "time": float(archive["time"][index]),
                        **{
                            key: float(archive[key][index])
                            for key in FLAT_METRIC_KEYS
                        },
                        "component_legacy": archive["component_legacy"][index].tolist(),
                        "component_proper": archive["component_proper"][index].tolist(),
                        "component_proper_trapezoid": archive[
                            "component_proper_trapezoid"
                        ][index].tolist(),
                    }
                    stage_records.append(record)
        legacy_max = max(record["legacy_ratio"] for record in stage_records)
        expected = immutable["evolution_diagnostics"][label][
            "maximum_outer_scalar_correction"
        ]
        reproduction_error = abs(legacy_max - expected)
        excess = [record for record in stage_records if record["legacy_ratio"] >= 0.05]
        normalization = bool(excess and all(
            record["proper_simpson_ratio"] < 0.05
            and record["proper_trapezoid_ratio"] < 0.05
            and record["term_balance_simpson"] < 0.05
            and record["term_balance_trapezoid"] < 0.05
            and record["collar_ratio"] < 0.05
            and record["pointwise_ratio"] < 0.20
            and max(record["component_proper"]) < 0.05
            and max(record["component_proper_trapezoid"]) < 0.05
            for record in excess
        ))
        robust = bool(any(
            record["proper_simpson_ratio"] >= 0.05
            or record["proper_trapezoid_ratio"] >= 0.05
            or record["collar_ratio"] >= 0.05
            or record["pointwise_ratio"] >= 0.20
            for record in excess
        ))
        if robust:
            robust_grids.add(label.split("_")[0])
        if excess:
            normalization_all = bool(normalization_all and normalization)
        maximum = max(stage_records, key=lambda record: record["legacy_ratio"])
        runs[label] = {
            "legacy_maximum": legacy_max,
            "immutable_test10b_maximum": expected,
            "legacy_reproduction_absolute_error": reproduction_error,
            "legacy_excess_stage_count": len(excess),
            "normalization_rule_at_all_excesses": normalization,
            "robust_excess_present": robust,
            "maximum_stage": maximum,
            "maximum_proper_face_ratio": max(r["proper_simpson_ratio"] for r in stage_records),
            "maximum_term_balance_ratio": max(r["term_balance_simpson"] for r in stage_records),
            "maximum_pointwise_ratio": max(r["pointwise_ratio"] for r in stage_records),
            "maximum_collar_ratio": max(r["collar_ratio"] for r in stage_records),
        }
        all_replay = bool(all_replay and reproduction_error < 1e-12)
        all_finite = bool(all_finite and all(record["finite"] == 1.0 for record in stage_records))
        for record in stage_records:
            independent_maxima["target_relative"] = max(
                independent_maxima["target_relative"],
                record["independent_target_relative_difference"],
            )
            independent_maxima["target_absolute"] = max(
                independent_maxima["target_absolute"],
                record["independent_target_maximum_absolute_difference"],
            )
            independent_maxima["characteristic_closure"] = max(
                independent_maxima["characteristic_closure"],
                record["characteristic_closure_relative"],
            )
            independent_maxima["post_residual"] = max(
                independent_maxima["post_residual"],
                record["post_correction_relative_residual"],
            )
            independent_maxima["production_residual"] = max(
                independent_maxima["production_residual"], record["production_residual"],
            )
            independent_maxima["quadrature"] = max(
                independent_maxima["quadrature"], record["quadrature_relative_difference"],
            )
    independent_gate = bool(
        independent_maxima["target_relative"] < 1e-10
        and independent_maxima["target_absolute"] < 1e-11
        and independent_maxima["characteristic_closure"] < 1e-10
        and independent_maxima["post_residual"] < 1e-10
        and independent_maxima["production_residual"] < 1e-10
        and independent_maxima["quadrature"] < 0.02
    )
    return {
        "runs": runs, "all_replays_and_legacy_maxima_reproduced": all_replay,
        "all_finite": all_finite, "independent_maxima": independent_maxima,
        "independent_gate": independent_gate,
        "normalization_rule_all_legacy_excesses": normalization_all,
        "robust_excess_on_both_grids": robust_grids == set(test10b.GRIDS),
        "robust_grids": sorted(robust_grids),
    }


def contamination_analysis(geometries, cases):
    records = {}
    common_maximum = 0.0
    signals = {}
    state_arrays = {}
    for grid in test10b.GRIDS:
        records[grid] = {}
        for left, right in (("R8", "R10"), ("R10", "R12")):
            pair = f"{left}_{right}"
            left_label = test10b.run_label(grid, left)
            right_label = test10b.run_label(grid, right)
            geometry = geometries[left_label]
            count = len(geometry["r"])
            steps = []
            profiles = []
            pair_maximum = 0.0
            for step in range(1, test10b.STEPS + 1):
                left_state = test10b.load_step(left_label, step, cases[left_label])
                right_state = test10b.load_step(right_label, step, cases[right_label])
                named_profiles = {}
                for key in ("position", "velocity", "source_increment"):
                    named_profiles[key] = normalized_radial_difference(
                        left_state[key], np.asarray(right_state[key])[:, :count],
                    )
                combined = np.maximum.reduce(list(named_profiles.values()))
                inside = geometry["r"] <= test10b.R_CUT + 1e-12
                interior_maximum = float(np.max(combined[inside]))
                common_maximum = max(common_maximum, interior_maximum)
                pair_maximum = max(pair_maximum, interior_maximum)
                affected = np.flatnonzero(combined > 1e-13)
                innermost = float(geometry["r"][affected[0]]) if len(affected) else None
                steps.append({
                    "step": step, "time": step * test10b.DT,
                    "common_r6_maximum": interior_maximum,
                    "innermost_radius_above_1e_13": innermost,
                    "global_maximum": float(np.max(combined)),
                })
                profiles.append(combined)
            records[grid][pair] = {"steps": steps, "common_r6_maximum": pair_maximum}
            signals[f"{grid}_{pair}"] = pair_maximum > 1e-10
            state_arrays[f"{grid}_{pair}_radial_maximum"] = np.max(np.asarray(profiles), axis=0)
            state_arrays[f"{grid}_{pair}_r"] = geometry["r"]
    resolved = bool(
        any(signals[f"{grid}_R8_R10"] and signals[f"{grid}_R10_R12"] for grid in test10b.GRIDS)
        or any(signals[f"G7_{pair}"] and signals[f"G8_{pair}"] for pair in ("R8_R10", "R10_R12"))
    )
    indeterminate = bool(not resolved and common_maximum >= 1e-12)
    return {
        "records": records, "common_r6_maximum": common_maximum,
        "resolved_contamination": resolved, "indeterminate_band": indeterminate,
        "no_common_interior_contamination": common_maximum < 1e-12,
        "_state_arrays": state_arrays,
    }


def assemble_state(index, contamination, replay):
    arrays = dict(contamination["_state_arrays"])
    for label, record in replay["runs"].items():
        arrays[f"{label}_summary"] = np.asarray([
            record["legacy_maximum"], record["maximum_proper_face_ratio"],
            record["maximum_term_balance_ratio"], record["maximum_pointwise_ratio"],
            record["maximum_collar_ratio"],
        ])
    stage_id = "final/state"
    index.register(stage_id, "state-archive", 1200.0, {"arrays": len(arrays)})
    cached = index.validated_path(stage_id)
    if cached is None:
        index.mark_running(stage_id)
        started = time.perf_counter()
        atomic_write_npz(STATE_OUTPUT, **arrays)
        validate_npz(STATE_OUTPUT)
        index.mark_complete(stage_id, STATE_OUTPUT, time.perf_counter() - started)


def main():
    started = time.perf_counter()
    index = RecoveryIndex(MANIFEST, PROTOCOL, recovery_inputs(), maximum_stage_seconds=2400.0)
    source_qualification = json_stage(
        index, "qualification/test10b", RECOVERY_ROOT / "qualification_test10b.json",
        lambda: {"qualification": validate_test10b_archive()},
    )
    controls = json_stage(
        index, "controls/manufactured", RECOVERY_ROOT / "controls.json",
        lambda: {"controls": manufactured_controls()},
    )["controls"]
    if not controls["passed"]:
        raise RuntimeError("Test10C manufactured controls failed")
    print("constructing Test10C exact common-parent cases", flush=True)
    _, geometries = test10b.build_geometries()
    cases = test10b.make_cases(geometries)
    paths = {label: [] for label in RUNS}
    for run_index, label in enumerate(RUNS):
        cases[label]["_test10c_label"] = label
        parent_sha = test10b.case_fingerprint(cases[label])
        for start in range(0, test10b.STEPS, test10b.SEGMENT):
            end = min(start + test10b.SEGMENT, test10b.STEPS)
            path = replay_segment(
                index, run_index, label, cases[label], start, end, parent_sha,
            )
            paths[label].append(path)
            parent_sha = sha256_file(path)
    replay = json_stage(
        index, "analysis/replay", RECOVERY_ROOT / "analysis_replay.json",
        lambda: {"analysis": replay_analysis(paths)},
    )["analysis"]
    contamination = json_stage(
        index, "analysis/contamination", RECOVERY_ROOT / "analysis_contamination.json",
        lambda: {"analysis": {
            key: value for key, value in contamination_analysis(geometries, cases).items()
            if key != "_state_arrays"
        }},
    )["analysis"]
    contamination_with_state = contamination_analysis(geometries, cases)
    independent = json_stage(
        index, "analysis/independent", RECOVERY_ROOT / "analysis_independent.json",
        lambda: {"analysis": {
            "gate": replay["independent_gate"],
            "maxima": replay["independent_maxima"],
            "algorithm": "polynomial-moment endpoint weights and direct characteristic target",
        }},
    )["analysis"]
    valid = bool(
        source_qualification["qualification"]["stage_count"] == 160
        and controls["passed"] and replay["all_replays_and_legacy_maxima_reproduced"]
        and replay["all_finite"] and independent["gate"]
    )
    normalization = bool(
        replay["normalization_rule_all_legacy_excesses"]
        and contamination["no_common_interior_contamination"]
    )
    model_response = bool(
        replay["robust_excess_on_both_grids"]
        and contamination["no_common_interior_contamination"]
    )
    status, classification = classify_test10c(
        valid, contamination["resolved_contamination"],
        contamination["indeterminate_band"], normalization, model_response,
    )
    assemble_state(index, contamination_with_state, replay)
    prefinal_valid = bool(all(
        stage.get("status") == "complete" and index.validated_path(stage_id) is not None
        for stage_id, stage in index.data["stages"].items()
    ))
    payload = {
        "status": status, "classification": classification,
        "scope": "sealed Test-10C outer-scalar closure and normalization audit",
        "preserved_test10b": "review: invalid_common_parent_audit",
        "protocol": str(PROTOCOL), "protocol_sha256": index.protocol_sha256,
        "source_qualification": source_qualification["qualification"],
        "controls": controls, "replay_analysis": replay,
        "independent_audit": independent, "contamination_analysis": contamination,
        "diagnoses": {
            "normalization_inconsistency_rule": normalization,
            "genuine_boundary_model_response_rule": model_response,
            "resolved_contamination_rule": contamination["resolved_contamination"],
        },
        "acceptance": {
            "source_and_recovery": source_qualification["qualification"]["stage_count"] == 160,
            "manufactured_and_adverse_controls": controls["passed"],
            "exact_replay_and_legacy_reproduction": replay["all_replays_and_legacy_maxima_reproduced"],
            "independent_reconstruction": independent["gate"],
            "positive_finite_measures": replay["all_finite"],
            "prefinal_provenance": prefinal_valid,
        },
        "runtime": {
            "wall_seconds_this_invocation": time.perf_counter() - started,
            "cumulative_stage_compute_seconds": float(sum(
                item.get("elapsed_seconds", 0.0) for item in index.data["stages"].values()
            )),
        },
        "limitations": [
            "Test10B remains immutable REVIEW and its 5-percent gate is unchanged",
            "proper-face weighting does not make coordinate-time acceleration a spacetime scalar",
            "short t<=0.002 common-parent restriction audit only",
            "not an event horizon, topology, halo, mass-transfer, or nonlinear-stability result",
        ],
    }
    stage_id = "final/result"
    index.register(stage_id, "combined-result", 300.0, {"status": status})
    cached = index.validated_path(stage_id)
    if cached is None:
        index.mark_running(stage_id)
        atomic_write_json(OUTPUT, payload)
        index.mark_complete(stage_id, OUTPUT, 0.0)
    else:
        payload = json.loads(cached.read_text())
    print(json.dumps({
        "status": payload["status"], "classification": payload["classification"],
        "diagnoses": payload["diagnoses"],
        "common_r6_maximum": payload["contamination_analysis"]["common_r6_maximum"],
        "run_maxima": {
            label: {
                key: value for key, value in record.items()
                if key in ("legacy_maximum", "maximum_proper_face_ratio", "maximum_term_balance_ratio", "maximum_pointwise_ratio", "maximum_collar_ratio")
            } for label, record in payload["replay_analysis"]["runs"].items()
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
