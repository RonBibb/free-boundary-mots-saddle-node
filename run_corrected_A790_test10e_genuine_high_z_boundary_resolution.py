#!/usr/bin/env python3
"""Run the prospectively sealed Test-10E genuine high-z audit."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import run_corrected_A790_test10b_domain_normalized as test10b
import run_corrected_A790_test10c_outer_scalar_closure as test10c
import run_corrected_A790_test10d_boundary_resolution as test10d
import run_corrected_A790_dynamic_MOTS_stability as mots_stability
from bhps.corrected_A790_R12_builder import (
    build_A790_R12_pair,
    build_A790_R12_refined,
)
from bhps.corrected_A790_test10b_domain_normalized import (
    first_detection_bracket,
    relative_difference,
    restrict_geometry,
    restriction_identity,
)
from bhps.corrected_A790_test10c_outer_scalar import (
    correction_metrics,
    independent_characteristic_terms,
    normalized_radial_difference,
)
from bhps.corrected_A790_test10d_boundary_resolution import (
    LEVELS,
    METRIC_KEYS,
    SCHEMES,
    closure_gate,
    ensemble_metrics,
    evaluate_all_levels,
    refinement_flags,
)
from bhps.corrected_A790_test10e_high_z import (
    SOURCE_METRICS,
    classify_test10e,
    consistency_close,
    manufactured_controls as test10e_controls,
    scheme_spread,
    separation_record,
    source_sequence,
    spread_gate,
    stage_history,
)
from bhps.nonlinear_regular_so3_evolution import apply_outer_sommerfeld_acceleration
from bhps.recovery_indexer import (
    RecoveryIndex,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
    validate_npz,
)


PROTOCOL = Path("notes/109_A790_test10E_genuine_high_z_boundary_resolution_protocol.md")
PROTOCOL_SHA256 = "6769b09019bc443cb5a2ace7632065ea0b76e14d9e3a4debd288164defb088a8"
OUTPUT = Path("results/corrected_A790_test10e_genuine_high_z_boundary_resolution.json")
STATE_OUTPUT = Path("results/corrected_A790_test10e_genuine_high_z_boundary_resolution_state.npz")
RECOVERY_ROOT = Path("results/corrected_A790_test10e_genuine_high_z_boundary_resolution_recovery")
MANIFEST = RECOVERY_ROOT / "index.json"
SEGMENT = 4
FINAL_TIME = 0.002
PRIMARY_DT = 0.000125
HALF_DT = 0.0000625
DOMAINS = ("R8", "R10", "R12")
PRIMARY_GRIDS = ("G9", "G10")
GEOMETRY_STEPS = (10, 12, 14, 16)

FIXED_INPUT_HASHES = {
    "notes/89_A790_fourth_grid_physical_tensor_convergence_protocol.md": "f3edbca20287805f95810ad25b6aa59a814136eea0491ca59721180d49c9f0d3",
    "notes/98_A790_domain_normalized_family_protocol.md": "7c0298c67d03654b2e24c7c8602a77a8c70bd75beba14645fdd76adfef11cc23",
    "notes/98_A790_domain_normalized_family_result.md": "618b72fb125ddc67c8ea523789687a4836216fd52bda26e59223c8163a4e42d0",
    "notes/103_A790_test10C_outer_scalar_closure_audit_protocol.md": "73a4ebaec15c53477dbf2a270b55c8e7c688a987513ce53fadba432547cffe50",
    "notes/103_A790_test10C_outer_scalar_closure_audit_result.md": "f04352c2c895339cea9d1860e168c97564deffcf3bc762dc292922765ee14f1f",
    "notes/106_A790_test10D_boundary_resolution_audit_protocol.md": "143a683c30ee3a1f3b5423eeeadc0c6baf61ead5c9f271cee184a46224fde2ab",
    "notes/106_A790_test10D_boundary_resolution_audit_result.md": "3eef75ed7e198fe37184f38c351b908bd5bd7a74f90eb5d57c7ca84a85a648b7",
    "results/corrected_A790_test10b_domain_normalized.json": "08c91892930c05fef6afe37d0a4f5ed1c49f545a84b6df4c81e08c8f2cbd1bd3",
    "results/corrected_A790_test10b_domain_normalized_state.npz": "b19e7e8901fdade6abb485c65a3512bf327a86f13974fb9a8c0876340a503928",
    "results/corrected_A790_test10b_domain_normalized_recovery/index.json": "7d064cfc491a0d2772c051cc1d10985ea936d320f26651a9423fa5ddb9947a0b",
    "results/corrected_A790_test10d_boundary_resolution.json": "795bade55cbaad8233984aa0edefbece4d94efb8c71a62f7094a9f7acac12816",
    "results/corrected_A790_test10d_boundary_resolution_state.npz": "c8e88504a48c6ca98bd1d971a8fee78bae21cd42f3872213da7626b73c763cbc",
    "results/corrected_A790_test10d_boundary_resolution_recovery/index.json": "f52d5b79d586c2b1ca96f11b24055839b33ccc02b9692f05f5e09b4d0a6af35d",
    "src/bhps/corrected_A790_R12_builder.py": "a0d0a5e7c12fef5bdacb2c97710787266c5ed3beba5a435dd7653edaf88322cc",
    "src/bhps/recovery_indexer.py": "1460478fba42433bd340a2ef9e09c0946882a35d3eb63c2c95ea9b055bb549fa",
    "run_corrected_A790_test10b_domain_normalized.py": "37ce44ba92af732a0c18202fd756286e3e5b935a75e0864d04fd9f140e32664b",
    "run_corrected_A790_test10d_boundary_resolution.py": "32c84addc2fc97dc3db50e5366e969ff75d16cfecb77608d98e0634daebe6194",
    "src/bhps/corrected_A790_test10d_boundary_resolution.py": "d5585b8fd4fcf3be221f527f4c6928f98255c1c27425e74aaff04305f243e52a",
    "run_corrected_fold_live_nonlinear_gauge_source.py": "b886bf79d57f98b372d8f756d22016f56192d1816b893536e4f6fd5ac242c203",
}


def recovery_inputs():
    dynamic = (
        Path(__file__),
        Path("src/bhps/corrected_A790_test10e_high_z.py"),
        Path("tests/test_A790_test10e_high_z.py"),
        Path("tests/test_A790_test10e_runner_recovery.py"),
        Path("run_corrected_A790_dynamic_MOTS_stability.py"),
        Path("src/bhps/dynamical_mots_stability.py"),
    )
    return {**FIXED_INPUT_HASHES, **{str(path): sha256_file(path) for path in dynamic}}


def json_stage(index, stage_id, path, producer, kind="analysis", metadata=None, expected=2400.0):
    index.register(stage_id, kind, expected, metadata or {})
    cached = index.validated_path(stage_id)
    if cached is not None:
        payload = json.loads(cached.read_text())
        if payload.get("protocol_sha256") != index.protocol_sha256:
            raise RuntimeError(f"cached protocol mismatch: {stage_id}")
        return payload
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


def validate_sources():
    if sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("Test10E protocol identity changed")
    for path, expected in FIXED_INPUT_HASHES.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"fixed input identity changed: {path}")
    return {"passed": True, "fixed_input_count": len(FIXED_INPUT_HASHES)}


def run_spec(label):
    if label.startswith("G10H_"):
        return {"steps": 32, "dt": HALF_DT, "base": 256}
    if label == "Z9_R10":
        return {"steps": 16, "dt": PRIMARY_DT, "base": 192}
    if label == "Z10_R10":
        return {"steps": 16, "dt": PRIMARY_DT, "base": 224}
    grid, domain = label.split("_")
    grid_index = PRIMARY_GRIDS.index(grid)
    domain_index = DOMAINS.index(domain)
    return {"steps": 16, "dt": PRIMARY_DT, "base": 96 * grid_index + 32 * domain_index}


def enumeration(label, step, rk_stage):
    return 1 + run_spec(label)["base"] + 2 * (int(step) - 1) + (int(rk_stage) - 1)


def checkpoint_path(label, start, end):
    return RECOVERY_ROOT / f"physical_{label}_steps_{start + 1:03d}_{end:03d}.npz"


def refined_key(scheme, level, metric):
    return f"{scheme}_L{int(level):02d}_{metric}"


def build_families():
    print("Test10E: constructing genuine G7/G8/G9/G10 and z-only parents", flush=True)
    g7, g8 = build_A790_R12_pair()
    g9 = build_A790_R12_refined(
        g8, 113, 253, "G9A790R12-test10e", selector_iterations=55,
        slice_iterations=360,
    )
    g10 = build_A790_R12_refined(
        g9, 129, 289, "G10A790R12-test10e", selector_iterations=60,
        slice_iterations=380,
    )
    z9 = build_A790_R12_refined(
        g8, 113, 217, "Z9A790R12-test10e", selector_iterations=55,
        slice_iterations=360,
    )
    z10 = build_A790_R12_refined(
        z9, 129, 217, "Z10A790R12-test10e", selector_iterations=60,
        slice_iterations=380,
    )
    parents = {"G7": g7, "G8": g8, "G9": g9, "G10": g10, "Z9": z9, "Z10": z10}
    geometries = {}
    for grid in PRIMARY_GRIDS:
        for domain, endpoint in zip(DOMAINS, (8.0, 10.0, 12.0)):
            label = f"{grid}_{domain}"
            geometries[label] = restrict_geometry(parents[grid], endpoint, f"{label}-test10e")
    for grid in ("Z9", "Z10"):
        geometries[f"{grid}_R10"] = restrict_geometry(
            parents[grid], 10.0, f"{grid}_R10-test10e",
        )
    geometries["G10H_R10"] = geometries["G10_R10"]
    return parents, geometries


def make_cases(geometries):
    cases = {}
    for label, geometry in geometries.items():
        cases[label] = test10c.live.setup_case(
            geometry, f"{label}-A790-test10e", live_normal_wall_gauge=True,
            live_outer_sommerfeld=True,
        )
    return cases


def parent_summary(parent, geometries, grid):
    domains = DOMAINS if grid in PRIMARY_GRIDS else ("R10",)
    return {
        "name": parent["name"],
        "grid_size": list(parent["source_grid"]),
        "reference_residual": float(parent["reference_maximum_residual"]),
        "selector_residual": float(parent["selector_maximum"]),
        "restrictions": {
            domain: {
                "grid_size": list(geometries[f"{grid}_{domain}"]["source_grid"]),
                "identity": restriction_identity(parent, geometries[f"{grid}_{domain}"]),
            } for domain in domains
        },
    }


def instrument_stage(case, time_value, state):
    position, velocity, source, memory = state
    live = test10c.live
    source_z, source_r = live.regular_source_spatial_derivatives(source, case["z"], case["r"])
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
    source_dot, outer_source = live.apply_outer_source_sommerfeld(
        source, source_dot, case["source0"], case["source_time0"],
        case["_initial_source_second_time"], position, time_value, case["r"],
        case["rhs"].stencil_width,
    )
    gauge = live.StageRegularGaugeSource(source, source_dot, case["z"], case["r"])
    source_second = live.live_regular_source_second_time(
        position, velocity, case["initial"], case["source0"], source,
        source_dot, memory_dot, case["z"], case["r"], live.DRIVER_MU,
        live.TARGET_MU_LAPSE, live.TARGET_MU_SHIFT, live.TARGET_POWER,
    )
    enabled = case["rhs"].live_outer_sommerfeld
    case["rhs"].live_outer_sommerfeld = False
    try:
        before, diagnostic = case["rhs"].acceleration(
            time_value, position, velocity, gauge, source_second,
        )
    finally:
        case["rhs"].live_outer_sommerfeld = enabled
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
    open_slice = (slice(1, -1), -1, slice(7, 9))
    raw = {
        "before": before[open_slice], "after": after[open_slice],
        "target": terms["target"][1:-1, 7:9],
        "term_A": terms["term_A"][1:-1, 7:9],
        "term_V": terms["term_V"][1:-1, 7:9],
        "term_C": terms["term_C"][1:-1, 7:9],
        "q_perp": position[1:-1, -1, 3],
        "q_zz": position[1:-1, -1, 6],
    }
    delta = raw["after"] - raw["before"]
    fields = {
        "delta": delta, "before": raw["before"], "after": raw["after"],
        "reference": case["rhs"].outer_reference_acceleration[1:-1, -1, 7:9],
        "term_a": raw["term_A"], "term_v": raw["term_V"],
        "term_c": raw["term_C"],
        "term_sum": raw["term_A"] + raw["term_V"] + raw["term_C"],
    }
    refined = evaluate_all_levels(
        case["z"][1:-1], float(case["r"][-1]), raw["q_perp"], raw["q_zz"],
        fields, metrics["collar_rms_before"],
    )
    flags = refinement_flags(refined)
    ensemble = ensemble_metrics(refined)
    closure = raw["term_A"] + raw["term_V"] + raw["term_C"] + delta
    closure_absolute = float(np.max(np.abs(closure)))
    closure_scale = float(max(np.max(np.abs(x)) for x in (
        raw["term_A"], raw["term_V"], raw["term_C"], delta,
    )))
    target_absolute = float(np.max(np.abs(raw["after"] - raw["target"])))
    target_scale = float(max(np.max(np.abs(raw["after"])), np.max(np.abs(raw["target"]))))
    normal = diagnostic["normal_wall_gauge"]
    record = {
        "legacy_ratio": float(metrics["legacy_ratio"]),
        "pointwise_ratio": float(metrics["pointwise"]["maximum"]),
        "maximum_absolute_correction": float(metrics["maximum_absolute_correction"]),
        "collar_rms": float(metrics["collar_rms_before"]),
        "production_residual": float(production["maximum_normalized_acceleration_residual"]),
        "source_residual": float(outer_source["maximum_normalized"]),
        "normal_wall_residual": float(normal["final_residual"]["maximum"]),
        "maximum_any_correction": float(max(
            production["metric_relative_correction"],
            production["scalar_relative_correction"], outer_source["relative_correction"],
        )),
        "finite": bool(
            diagnostic["finite"] and np.all(np.isfinite(after))
            and np.all(np.isfinite(source_dot)) and np.all(np.isfinite(memory_dot))
        ),
        "closure_absolute": closure_absolute, "closure_scale": closure_scale,
        "closure_pass": closure_gate(closure_absolute, closure_scale),
        "target_absolute": target_absolute, "target_scale": target_scale,
        "target_pass": closure_gate(target_absolute, target_scale),
        "refined": refined, "flags": flags, "ensemble": ensemble,
    }
    return (velocity, after, source_dot, memory_dot), record, raw


def _record_arrays(records):
    scalar = (
        "enumeration", "step", "rk_stage", "time", "legacy_ratio",
        "pointwise_ratio", "maximum_absolute_correction", "collar_rms",
        "production_residual", "source_residual", "normal_wall_residual",
        "maximum_any_correction", "finite", "closure_absolute", "closure_scale",
        "closure_pass", "target_absolute", "target_scale", "target_pass",
    )
    arrays = {key: np.asarray([record[key] for record in records]) for key in scalar}
    for key in ("before", "after", "target", "term_A", "term_V", "term_C", "q_perp", "q_zz"):
        arrays[key] = np.asarray([record["raw"][key] for record in records])
    for scheme in SCHEMES:
        for level in LEVELS:
            for metric in METRIC_KEYS:
                arrays[refined_key(scheme, level, metric)] = np.asarray([
                    record["refined"][scheme][level][metric] for record in records
                ])
        for metric in METRIC_KEYS:
            arrays[f"{scheme}_converged_{metric}"] = np.asarray([
                record["flags"][scheme][metric] for record in records
            ])
    for metric in METRIC_KEYS:
        arrays[f"cross_scheme_{metric}"] = np.asarray([
            record["flags"]["cross_scheme"][metric] for record in records
        ])
        arrays[f"ensemble_{metric}"] = np.asarray([
            record["ensemble"][metric] for record in records
        ])
    return arrays


def validate_checkpoint(path, case, label, start, end):
    count = 2 * (end - start)
    open_count = len(case["z"]) - 2
    shape = tuple(case["initial"].shape)
    source_shape = tuple(case["source0"].shape)
    required = {
        "start_step": (), "end_step": (), "end_position": shape,
        "end_velocity": shape, "end_source": source_shape, "end_memory": source_shape,
        "enumeration": (count,), "step": (count,), "rk_stage": (count,),
        "time": (count,), "before": (count, open_count, 2),
        "after": (count, open_count, 2), "target": (count, open_count, 2),
        "term_A": (count, open_count, 2), "term_V": (count, open_count, 2),
        "term_C": (count, open_count, 2), "q_perp": (count, open_count),
        "q_zz": (count, open_count),
    }
    for key in (
        "legacy_ratio", "pointwise_ratio", "maximum_absolute_correction",
        "collar_rms", "production_residual", "source_residual",
        "normal_wall_residual", "maximum_any_correction", "finite",
        "closure_absolute", "closure_scale", "closure_pass",
        "target_absolute", "target_scale", "target_pass",
    ):
        required[key] = (count,)
    for scheme in SCHEMES:
        for level in LEVELS:
            for metric in METRIC_KEYS:
                required[refined_key(scheme, level, metric)] = (count,)
        for metric in METRIC_KEYS:
            required[f"{scheme}_converged_{metric}"] = (count,)
    for metric in METRIC_KEYS:
        required[f"cross_scheme_{metric}"] = (count,)
        required[f"ensemble_{metric}"] = (count,)
    for step in range(start + 1, end + 1):
        required[f"step_{step:03d}_increment"] = shape
        required[f"step_{step:03d}_velocity"] = shape
        required[f"step_{step:03d}_source_increment"] = source_shape
    validate_npz(path, required)
    expected = np.arange(enumeration(label, start + 1, 1), enumeration(label, end, 2) + 1)
    with np.load(path) as archive:
        if not np.array_equal(archive["enumeration"], expected):
            raise RuntimeError("Test10E enumeration mismatch")
        if int(archive["start_step"]) != start or int(archive["end_step"]) != end:
            raise RuntimeError("Test10E segment bounds mismatch")


def run_physical(index, label, case):
    spec = run_spec(label)
    state = (
        case["initial"].copy(), np.zeros_like(case["initial"]),
        case["source0"].copy(), case["memory0"].copy(),
    )
    paths = []
    parent_hash = test10b.case_fingerprint(case)
    for start in range(0, spec["steps"], SEGMENT):
        end = min(start + SEGMENT, spec["steps"])
        stage_id = f"physical/{label}/steps_{start + 1:03d}_{end:03d}"
        path = checkpoint_path(label, start, end)
        metadata = {
            "run": label, "start": start, "end": end, "dt": spec["dt"],
            "enumeration": [enumeration(label, start + 1, 1), enumeration(label, end, 2)],
            "parent_hash": parent_hash, "case_fingerprint": test10b.case_fingerprint(case),
        }
        index.register(stage_id, "physical-evolution-face", 3600.0, metadata)
        cached = index.validated_path(stage_id)
        if cached is None:
            index.mark_running(stage_id)
            started = time.perf_counter()
            try:
                records = []
                snapshots = {}
                for step in range(start + 1, end + 1):
                    t0 = (step - 1) * spec["dt"]
                    print(f"{label}: Test10E step {step}/{spec['steps']} stage 1", flush=True)
                    k1, m1, raw1 = instrument_stage(case, t0, state)
                    midpoint = tuple(value + 0.5 * spec["dt"] * slope for value, slope in zip(state, k1))
                    print(f"{label}: Test10E step {step}/{spec['steps']} stage 2", flush=True)
                    k2, m2, raw2 = instrument_stage(case, t0 + 0.5 * spec["dt"], midpoint)
                    for rk, when, metric, raw in ((1, t0, m1, raw1), (2, t0 + 0.5 * spec["dt"], m2, raw2)):
                        records.append({
                            "enumeration": enumeration(label, step, rk), "step": step,
                            "rk_stage": rk, "time": when, **metric, "raw": raw,
                        })
                    state = tuple(value + spec["dt"] * slope for value, slope in zip(state, k2))
                    snapshots[f"step_{step:03d}_increment"] = state[0] - case["initial"]
                    snapshots[f"step_{step:03d}_velocity"] = state[1].copy()
                    snapshots[f"step_{step:03d}_source_increment"] = state[2] - case["source0"]
                if not all(record["finite"] for record in records):
                    raise RuntimeError("nonfinite Test10E stage")
                if max(record["production_residual"] for record in records) >= 1e-10:
                    raise RuntimeError("outer acceleration row residual exceeds 1e-10")
                if max(record["source_residual"] for record in records) >= 1e-10:
                    raise RuntimeError("outer source row residual exceeds 1e-10")
                if max(record["normal_wall_residual"] for record in records) >= 1e-10:
                    raise RuntimeError("normal wall row residual exceeds 1e-10")
                if max(record["maximum_any_correction"] for record in records) > 0.50:
                    raise RuntimeError("outer correction exceeds technical 50-percent stop")
                if not test10c.live.signature_summary(state[0], case["r"])["all_points_one_negative_direction"]:
                    raise RuntimeError("lost Lorentzian signature")
                arrays = _record_arrays(records)
                atomic_write_npz(
                    path, start_step=np.asarray(start), end_step=np.asarray(end),
                    end_position=state[0], end_velocity=state[1], end_source=state[2],
                    end_memory=state[3], **snapshots, **arrays,
                )
                validate_checkpoint(path, case, label, start, end)
                index.mark_complete(stage_id, path, time.perf_counter() - started)
                cached = path
            except Exception as error:
                index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
                raise
        else:
            validate_checkpoint(cached, case, label, start, end)
        with np.load(cached) as archive:
            state = tuple(np.asarray(archive[key]) for key in (
                "end_position", "end_velocity", "end_source", "end_memory",
            ))
        paths.append(cached)
        parent_hash = sha256_file(cached)
    return paths


def load_endpoint(label, step, case):
    start = ((int(step) - 1) // SEGMENT) * SEGMENT
    end = min(start + SEGMENT, run_spec(label)["steps"])
    with np.load(checkpoint_path(label, start, end)) as archive:
        return {
            "position": case["initial"] + archive[f"step_{step:03d}_increment"],
            "velocity": np.asarray(archive[f"step_{step:03d}_velocity"]),
            "source_increment": np.asarray(archive[f"step_{step:03d}_source_increment"]),
        }


def read_history(paths):
    keys = (
        "legacy_ratio", "pointwise_ratio", "maximum_absolute_correction",
        *[f"ensemble_{metric}" for metric in METRIC_KEYS],
        *[f"pchip_gauss_L16_{metric}" for metric in METRIC_KEYS],
        *[f"natural_cubic_romberg_L16_{metric}" for metric in METRIC_KEYS],
    )
    records = {key: [] for key in keys}
    flags = {"internal": True, "cross": True, "closure": True, "target": True, "finite": True}
    residuals = {"production": 0.0, "source": 0.0, "normal_wall": 0.0, "correction": 0.0}
    times = []
    for path in paths:
        with np.load(path) as archive:
            times.extend(np.asarray(archive["time"], dtype=float).tolist())
            for key in keys:
                records[key].extend(np.asarray(archive[key], dtype=float).tolist())
            for scheme in SCHEMES:
                for metric in METRIC_KEYS:
                    flags["internal"] &= bool(np.all(archive[f"{scheme}_converged_{metric}"]))
            for metric in METRIC_KEYS:
                flags["cross"] &= bool(np.all(archive[f"cross_scheme_{metric}"]))
            flags["closure"] &= bool(np.all(archive["closure_pass"]))
            flags["target"] &= bool(np.all(archive["target_pass"]))
            flags["finite"] &= bool(np.all(archive["finite"]))
            residuals["production"] = max(residuals["production"], float(np.max(archive["production_residual"])))
            residuals["source"] = max(residuals["source"], float(np.max(archive["source_residual"])))
            residuals["normal_wall"] = max(residuals["normal_wall"], float(np.max(archive["normal_wall_residual"])))
            residuals["correction"] = max(residuals["correction"], float(np.max(archive["maximum_any_correction"])))
    arrays = {key: np.asarray(value) for key, value in records.items()}
    maxima = {
        "legacy_ratio": float(np.max(arrays["legacy_ratio"])),
        "proper_ratio": float(np.max(arrays["ensemble_proper_ratio"])),
        "collar_ratio": float(np.max(arrays["ensemble_collar_ratio"])),
        "component_phi_ratio": float(np.max(arrays["ensemble_component_phi_ratio"])),
        "pointwise_ratio": float(np.max(arrays["pointwise_ratio"])),
        "maximum_absolute_correction": float(np.max(arrays["maximum_absolute_correction"])),
        "term_balance_ratio": float(np.max(arrays["ensemble_term_balance_ratio"])),
        "component_chi_ratio": float(np.max(arrays["ensemble_component_chi_ratio"])),
        "face_rms_correction": float(np.max(arrays["ensemble_face_rms_delta"])),
    }
    spread_metrics = {
        "proper_ratio": "proper_ratio", "collar_ratio": "collar_ratio",
        "component_phi_ratio": "component_phi_ratio",
        "maximum_absolute_correction": "norm_delta",
        "face_rms_correction": "face_rms_delta",
    }
    spreads = {
        name: scheme_spread(arrays[f"pchip_gauss_L16_{metric}"], arrays[f"natural_cubic_romberg_L16_{metric}"])
        for name, metric in spread_metrics.items()
    }
    histories = {
        "proper_ratio": arrays["ensemble_proper_ratio"],
        "collar_ratio": arrays["ensemble_collar_ratio"],
        "component_phi_ratio": arrays["ensemble_component_phi_ratio"],
        "pointwise_ratio": arrays["pointwise_ratio"],
        "maximum_absolute_correction": arrays["maximum_absolute_correction"],
    }
    return {"times": np.asarray(times), "histories": histories, "maxima": maxima, "spreads": spreads, "flags": flags, "residuals": residuals}


def historical_history(grid, domain):
    label = f"{grid}_{domain}"
    paths = [test10d.checkpoint_path(label, start, start + 4) for start in range(0, 16, 4)]
    records = {key: [] for key in (
        "legacy_ratio", "pointwise_ratio", "maximum_absolute_correction",
        *[f"ensemble_{metric}" for metric in METRIC_KEYS],
        *[f"pchip_gauss_L16_{metric}" for metric in METRIC_KEYS],
        *[f"natural_cubic_romberg_L16_{metric}" for metric in METRIC_KEYS],
    )}
    for path in paths:
        with np.load(path) as archive:
            for key in records:
                records[key].extend(np.asarray(archive[key], dtype=float).tolist())
    arrays = {key: np.asarray(value) for key, value in records.items()}
    maxima = {
        "legacy_ratio": float(np.max(arrays["legacy_ratio"])),
        "proper_ratio": float(np.max(arrays["ensemble_proper_ratio"])),
        "collar_ratio": float(np.max(arrays["ensemble_collar_ratio"])),
        "component_phi_ratio": float(np.max(arrays["ensemble_component_phi_ratio"])),
        "pointwise_ratio": float(np.max(arrays["pointwise_ratio"])),
        "maximum_absolute_correction": float(np.max(arrays["maximum_absolute_correction"])),
    }
    return {
        "histories": {
            "proper_ratio": arrays["ensemble_proper_ratio"],
            "collar_ratio": arrays["ensemble_collar_ratio"],
            "component_phi_ratio": arrays["ensemble_component_phi_ratio"],
            "pointwise_ratio": arrays["pointwise_ratio"],
            "maximum_absolute_correction": arrays["maximum_absolute_correction"],
        },
        "maxima": maxima,
    }


def source_grid_analysis(new_histories):
    histories = {
        f"{grid}_{domain}": (
            historical_history(grid, domain) if grid in ("G7", "G8")
            else new_histories[f"{grid}_{domain}"]
        ) for grid in ("G7", "G8", "G9", "G10") for domain in DOMAINS
    }
    records = {}
    all_sequences = True
    all_stage_histories = True
    all_spreads = True
    for domain in DOMAINS:
        domain_record = {"sequences": {}, "stage_histories": {}, "scheme_spreads": {}}
        for metric in SOURCE_METRICS:
            values = [histories[f"{grid}_{domain}"]["maxima"][metric] for grid in ("G7", "G8", "G9", "G10")]
            sequence = source_sequence(metric, values)
            aligned = np.vstack([
                histories[f"{grid}_{domain}"]["histories"][metric]
                for grid in ("G7", "G8", "G9", "G10")
            ])
            stage = stage_history(metric, aligned)
            domain_record["sequences"][metric] = sequence
            domain_record["stage_histories"][metric] = stage
            all_sequences &= sequence["passes"]
            all_stage_histories &= stage["passes"]
        for metric in (
            "proper_ratio", "collar_ratio", "component_phi_ratio",
            "maximum_absolute_correction", "face_rms_correction",
        ):
            spread = spread_gate(
                histories[f"G9_{domain}"]["spreads"][metric],
                histories[f"G10_{domain}"]["spreads"][metric],
            )
            domain_record["scheme_spreads"][metric] = spread
            all_spreads &= spread["passes"]
        records[domain] = domain_record
    return {
        "records": records,
        "run_maxima": {label: record["maxima"] for label, record in histories.items()},
        "run_flags": {
            label: record.get("flags") for label, record in histories.items() if label.startswith(("G9_", "G10_"))
        },
        "sequence_gate": bool(all_sequences),
        "stage_history_gate": bool(all_stage_histories),
        "scheme_spread_gate": bool(all_spreads),
        "gate": bool(all_sequences and all_stage_histories and all_spreads),
    }


def fixed_radial_analysis(histories):
    records = {}
    gate = True
    for full, control in (("G9_R10", "Z9_R10"), ("G10_R10", "Z10_R10")):
        pair = {}
        for metric in SOURCE_METRICS:
            item = separation_record(
                metric, histories[full]["maxima"][metric], histories[control]["maxima"][metric],
            )
            pair[metric] = item
            gate &= item["passes"]
        records[f"{full}_{control}"] = pair
    z_consistency = {}
    for metric in SOURCE_METRICS:
        left = histories["Z9_R10"]["maxima"][metric]
        right = histories["Z10_R10"]["maxima"][metric]
        passed = consistency_close(metric, left, right)
        z_consistency[metric] = {"Z9": left, "Z10": right, "passes": passed}
        gate &= passed
    return {"records": records, "Z9_Z10_consistency": z_consistency, "gate": bool(gate)}


def temporal_analysis(histories):
    full = histories["G10_R10"]
    half = histories["G10H_R10"]
    records = {}
    gate = True
    for metric in SOURCE_METRICS:
        full_times = np.asarray(full["times"])
        half_times = np.asarray(half["times"])
        full_values = np.asarray(full["histories"][metric])
        half_values = np.asarray(half["histories"][metric])
        comparisons = []
        for when, value in zip(full_times, full_values):
            matches = np.flatnonzero(np.isclose(half_times, when, rtol=0.0, atol=1e-15))
            if len(matches) != 1:
                raise RuntimeError("half-step RK times are not uniquely aligned")
            control = half_values[matches[0]]
            comparisons.append(separation_record(metric, value, control, temporal=True))
        maximum = separation_record(
            metric, full["maxima"][metric], half["maxima"][metric], temporal=True,
        )
        passed = bool(all(item["passes"] for item in comparisons) and maximum["passes"])
        records[metric] = {"coincident": comparisons, "maxima": maximum, "passes": passed}
        gate &= passed
    return {"records": records, "gate": bool(gate)}


def locality_analysis(cases):
    records = {}
    maximum_r6 = 0.0
    innermost_gate = True
    signals = {}
    state_arrays = {}
    for grid in PRIMARY_GRIDS:
        records[grid] = {}
        for left, right in (("R8", "R10"), ("R10", "R12")):
            left_label, right_label = f"{grid}_{left}", f"{grid}_{right}"
            geometry = cases[left_label]["geometry"]
            count = len(geometry["r"])
            profiles = []
            steps = []
            for step in range(1, 17):
                left_state = load_endpoint(left_label, step, cases[left_label])
                right_state = load_endpoint(right_label, step, cases[right_label])
                component_profiles = [
                    normalized_radial_difference(
                        left_state[key], np.asarray(right_state[key])[:, :count],
                    ) for key in ("position", "velocity", "source_increment")
                ]
                combined = np.maximum.reduce(component_profiles)
                profiles.append(combined)
                inside = geometry["r"] <= 6.0 + 1e-12
                local = float(np.max(combined[inside]))
                maximum_r6 = max(maximum_r6, local)
                affected = np.flatnonzero(combined > 1e-13)
                inner_index = int(affected[0]) if len(affected) else None
                within_seven = inner_index is None or inner_index >= len(geometry["r"]) - 7
                innermost_gate &= within_seven
                steps.append({
                    "step": step, "time": step * PRIMARY_DT,
                    "common_r6_maximum": local,
                    "innermost_radius_above_1e_13": None if inner_index is None else float(geometry["r"][inner_index]),
                    "within_outer_seven_rows": bool(within_seven),
                    "global_maximum": float(np.max(combined)),
                })
            pair_maximum = max(item["common_r6_maximum"] for item in steps)
            records[grid][f"{left}_{right}"] = {"steps": steps, "common_r6_maximum": pair_maximum}
            signals[f"{grid}_{left}_{right}"] = pair_maximum > 1e-10
            state_arrays[f"{grid}_{left}_{right}_profile"] = np.max(np.asarray(profiles), axis=0)
            state_arrays[f"{grid}_{left}_{right}_r"] = geometry["r"]
    resolved = bool(
        any(signals[f"{grid}_R8_R10"] and signals[f"{grid}_R10_R12"] for grid in PRIMARY_GRIDS)
        or any(signals[f"G9_{pair}"] and signals[f"G10_{pair}"] for pair in ("R8_R10", "R10_R12"))
    )
    indeterminate = bool(not resolved and maximum_r6 >= 1e-12)
    return {
        "records": records, "common_r6_maximum": maximum_r6,
        "resolved_contamination": resolved, "indeterminate_band": indeterminate,
        "clean_r6": maximum_r6 < 1e-12, "outer_seven_row_gate": bool(innermost_gate),
        "gate": bool(maximum_r6 < 1e-12 and innermost_gate),
        "_state_arrays": state_arrays,
    }


def causal_analysis(geometries):
    records = {}
    for label in (f"{grid}_{domain}" for grid in PRIMARY_GRIDS for domain in DOMAINS):
        geometry = geometries[label]
        speed = float(np.max(geometry["principal"]["r_coordinate_speed"]))
        lower = float((geometry["r"][-1] - 6.0) / speed)
        fraction = FINAL_TIME / lower
        records[label] = {
            "maximum_initial_radial_coordinate_speed": speed,
            "one_way_boundary_to_r6_lower_bound": lower,
            "final_time_fraction": fraction, "passes": bool(fraction < 0.01),
        }
    return {"records": records, "gate": bool(all(item["passes"] for item in records.values()))}


def boundary_model_locality(histories):
    records = {}
    gate = True
    for grid in PRIMARY_GRIDS:
        item = {}
        for metric in ("proper_ratio", "collar_ratio"):
            r10 = histories[f"{grid}_R10"]["maxima"][metric]
            r12 = histories[f"{grid}_R12"]["maxima"][metric]
            passed = bool(r10 < 0.005 or (r12 < 0.005 and r12 < 0.25 * r10))
            item[metric] = {"R10": r10, "R12": r12, "passes": passed}
            gate &= passed
        records[grid] = item
    return {"records": records, "gate": bool(gate)}


def detector_stage(index, label, step, state, geometry, dt):
    stage_id = f"detector/{label}/step_{step:03d}"
    path = RECOVERY_ROOT / f"detector_{label}_step_{step:03d}.json"
    return json_stage(
        index, stage_id, path,
        lambda: {"search": test10b.search_slice(
            f"{label}-test10e-t{step * dt:.7f}", state["position"], state["velocity"], geometry,
        )},
        kind="independent-BVP", metadata={"run": label, "step": step, "time": step * dt},
    )["search"]


def initial_stage(index, label, case):
    path = RECOVERY_ROOT / f"initial_{label}.json"
    geometry = case["geometry"]
    return json_stage(
        index, f"initial/{label}", path,
        lambda: {
            "static": test10b.static_search(geometry),
            "BVP": test10b.search_slice(
                f"{label}-test10e-t0", case["initial"], np.zeros_like(case["initial"]), geometry,
            ),
        }, kind="initial-zero-cap-search", metadata={"run": label},
    )


def geometry_stage(index, label, step, search, state, geometry):
    stage_id = f"geometry/{label}/step_{step:03d}"
    path = RECOVERY_ROOT / f"geometry_{label}_step_{step:03d}.json"
    def produce():
        branches = []
        if search["admitted_distinct_count"] == 2:
            for branch_name, cluster in zip(
                ("inner", "outer"), sorted(search["clusters"], key=lambda item: item["signature"][1]),
            ):
                members = sorted(cluster["members"], key=lambda item: item["seed"])
                seed = float(members[len(members) // 2]["seed"])
                surface = mots_stability.recover_surface(state["position"], state["velocity"], geometry, seed)
                geometric = test10b.capped_surface_geometry(
                    state["position"], state["velocity"], geometry["z"], geometry["r"], surface,
                )
                stability = mots_stability.stability_series(
                    state["position"], state["velocity"], geometry, surface,
                )
                branches.append({
                    "branch": branch_name, "seed": seed,
                    "admitted": test10b.bvp_admitted(surface),
                    "surface": test10b.public_bvp_surface(surface),
                    "geometry": geometric,
                    "proper_endpoints": test10b.proper_endpoint_distances(
                        state["position"], geometry["z"], geometry["r"],
                        surface["rho_axis"], surface["rho_brane"],
                    ),
                    "stability": stability,
                })
        return {"branches": branches}
    return json_stage(
        index, stage_id, path, produce, kind="surface-invariant-stability",
        metadata={"run": label, "step": step, "time": step * PRIMARY_DT},
        expected=3600.0,
    )["branches"]


def formation_analysis(initial, searches, half_searches):
    expected = [0] * 9 + [2] * 7
    histories = {
        label: [searches[label][step]["admitted_distinct_count"] for step in range(1, 17)]
        for label in searches
    }
    initial_zero = bool(all(
        item["static"]["accepted_count"] == 0 and item["BVP"]["admitted_distinct_count"] == 0
        for item in initial.values()
    ))
    history_gate = bool(all(value == expected for value in histories.values()))
    brackets = {label: first_detection_bracket(value, PRIMARY_DT) for label, value in histories.items()}
    bracket_gate = bool(all(value == [0.001125, 0.00125] or tuple(value) == (0.001125, 0.00125) for value in brackets.values()))
    half_counts = [half_searches[step]["admitted_distinct_count"] for step in range(2, 33, 2)]
    temporal_gate = half_counts == histories["G10_R10"]
    return {
        "initial_zero": initial_zero, "histories": histories, "brackets": brackets,
        "expected_history": expected, "primary_history_gate": history_gate,
        "half_step_coincident_history": half_counts,
        "temporal_formation_gate": temporal_gate,
        "gate": bool(initial_zero and history_gate and bracket_gate and temporal_gate),
    }


SURFACE_SCALARS = (
    "rho_axis", "rho_brane", "one_sided_cap_area", "equivalent_area_radius",
    "proper_meridional_length",
)


def surface_analysis(records):
    archived = json.loads(test10b.OUTPUT.read_text())["geometry_analysis"]["maxima"]["grid"]
    comparisons = {}
    maximum = 0.0
    structural = True
    stability_gate = True
    for domain in DOMAINS:
        comparisons[domain] = {}
        for step in GEOMETRY_STEPS:
            left = records[f"G9_{domain}"][step]
            right = records[f"G10_{domain}"][step]
            step_records = []
            structural &= len(left) == len(right) == 2
            if len(left) == len(right) == 2:
                for branch_index, branch_name in enumerate(("inner", "outer")):
                    a, b = left[branch_index], right[branch_index]
                    transfers = {
                        metric: relative_difference(a["geometry"][metric], b["geometry"][metric])
                        for metric in SURFACE_SCALARS
                    }
                    transfers.update({
                        "proper_axis": relative_difference(
                            a["proper_endpoints"]["compact_axis_endpoint_to_brane"],
                            b["proper_endpoints"]["compact_axis_endpoint_to_brane"],
                        ),
                        "proper_brane": relative_difference(
                            a["proper_endpoints"]["radial_axis_to_brane_endpoint"],
                            b["proper_endpoints"]["radial_axis_to_brane_endpoint"],
                        ),
                    })
                    maximum = max(maximum, *transfers.values())
                    desired = "outward_unstable" if branch_name == "inner" else "outward_stable"
                    for item in (a, b):
                        structural &= bool(
                            item["branch"] == branch_name and item["admitted"]
                            and item["geometry"]["finite"]
                            and item["geometry"]["one_sided_cap_area"] > 0.0
                            and item["geometry"]["rho_brane"] > item["geometry"]["rho_axis"]
                        )
                        stability_gate &= item["stability"]["classification"] == desired
                    step_records.append({"branch": branch_name, "transfers": transfers})
                structural &= bool(
                    left[1]["geometry"]["one_sided_cap_area"] > left[0]["geometry"]["one_sided_cap_area"]
                    and right[1]["geometry"]["one_sided_cap_area"] > right[0]["geometry"]["one_sided_cap_area"]
                )
            comparisons[domain][str(step)] = step_records
    transfer_gate = bool(maximum < 0.01 and maximum <= 1.2 * archived)
    return {
        "comparisons": comparisons, "maximum_G9_G10_transfer": maximum,
        "archived_G7_G8_maximum": archived, "structural_gate": bool(structural),
        "stability_label_gate": bool(stability_gate), "transfer_gate": transfer_gate,
        "gate": bool(structural and stability_gate and transfer_gate),
    }


def normalization_response(histories):
    legacy_excess_count = 0
    normalization = True
    response_grids = set()
    uncontrolled = False
    records = {}
    for grid in PRIMARY_GRIDS:
        records[grid] = {}
        for domain in DOMAINS:
            label = f"{grid}_{domain}"
            paths = [checkpoint_path(label, start, start + 4) for start in range(0, 16, 4)]
            stage_count = 0
            local_normalization = True
            for path in paths:
                with np.load(path) as archive:
                    legacy = np.asarray(archive["legacy_ratio"], dtype=float)
                    uncontrolled |= bool(np.any(legacy >= 0.20))
                    excess = legacy >= 0.05
                    stage_count += int(np.sum(excess))
                    if np.any(excess):
                        for scheme in SCHEMES:
                            for metric in (
                                "proper_ratio", "collar_ratio", "component_phi_ratio",
                                "term_balance_ratio",
                            ):
                                local_normalization &= bool(np.all(
                                    np.asarray(archive[f"{scheme}_L16_{metric}"])[excess] < 0.05
                                ))
                        local_normalization &= bool(np.all(
                            np.asarray(archive["pointwise_ratio"])[excess] < 0.20
                        ))
            legacy_excess_count += stage_count
            normalization &= local_normalization
            maxima = histories[label]["maxima"]
            response = bool(
                maxima["proper_ratio"] >= 0.05 or maxima["collar_ratio"] >= 0.05
                or maxima["component_phi_ratio"] >= 0.05
                or maxima["pointwise_ratio"] >= 0.20
                or maxima["legacy_ratio"] >= 0.05
            )
            if response:
                response_grids.add(grid)
            records[grid][domain] = {
                "legacy_excess_stage_count": stage_count,
                "normalization_at_all_excesses": bool(local_normalization),
                "response_threshold": response,
            }
    return {
        "records": records, "legacy_excess_stage_count": legacy_excess_count,
        "normalization_rule": bool(legacy_excess_count > 0 and normalization),
        "response_grids": sorted(response_grids),
        "response_on_both_high_grids": response_grids == set(PRIMARY_GRIDS),
        "uncontrolled_legacy_response": bool(uncontrolled),
    }


def all_new_stage_quality(histories):
    return {
        "finite": bool(all(record["flags"]["finite"] for record in histories.values())),
        "internal_refinement": bool(all(record["flags"]["internal"] for record in histories.values())),
        "cross_scheme": bool(all(record["flags"]["cross"] for record in histories.values())),
        "dual_closure": bool(all(record["flags"]["closure"] for record in histories.values())),
        "independent_target": bool(all(record["flags"]["target"] for record in histories.values())),
        "row_residuals": bool(all(
            max(record["residuals"]["production"], record["residuals"]["source"], record["residuals"]["normal_wall"]) < 1e-10
            for record in histories.values()
        )),
        "below_technical_correction_stop": bool(all(
            record["residuals"]["correction"] <= 0.50 for record in histories.values()
        )),
    }


def independent_recompute(histories, cases, searches, half_searches, geometry_records, primary):
    local = locality_analysis(cases)
    local = {key: value for key, value in local.items() if key != "_state_arrays"}
    repeated = {
        "source": source_grid_analysis(histories),
        "fixed_radial": fixed_radial_analysis(histories),
        "temporal": temporal_analysis(histories),
        "locality": local,
        "boundary_model_locality": boundary_model_locality(histories),
        "formation": formation_analysis(primary["initial"], searches, half_searches),
        "surface": surface_analysis(geometry_records),
        "normalization_response": normalization_response(histories),
        "quality": all_new_stage_quality(histories),
    }
    comparisons = {
        key: json.dumps(repeated[key], sort_keys=True, default=float)
        == json.dumps(primary[key], sort_keys=True, default=float)
        for key in repeated
    }
    return {"gate": bool(all(comparisons.values())), "comparisons": comparisons}


def assemble_state(index, histories, locality):
    arrays = {}
    for label, record in histories.items():
        arrays[f"{label}_times"] = record["times"]
        for metric in SOURCE_METRICS:
            arrays[f"{label}_{metric}"] = record["histories"][metric]
    arrays.update(locality["_state_arrays"])
    index.register("final/state", "state-archive", 1200.0, {"arrays": len(arrays)})
    cached = index.validated_path("final/state")
    if cached is None:
        index.mark_running("final/state")
        started = time.perf_counter()
        atomic_write_npz(STATE_OUTPUT, **arrays)
        validate_npz(STATE_OUTPUT)
        index.mark_complete("final/state", STATE_OUTPUT, time.perf_counter() - started)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qualification-only", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    index = RecoveryIndex(MANIFEST, PROTOCOL, recovery_inputs(), maximum_stage_seconds=3600.0)
    qualification = json_stage(
        index, "qualification/sources", RECOVERY_ROOT / "qualification_sources.json",
        lambda: {"qualification": validate_sources()}, kind="qualification",
    )["qualification"]
    controls = json_stage(
        index, "controls/manufactured", RECOVERY_ROOT / "controls.json",
        lambda: {"controls": {
            "test10d": test10d.manufactured_controls(),
            "test10e": test10e_controls(),
        }}, kind="manufactured-controls",
    )["controls"]
    controls["passed"] = bool(controls["test10d"]["passed"] and controls["test10e"]["passed"])
    if not controls["passed"]:
        raise RuntimeError("Test10E manufactured controls failed")
    if args.qualification_only:
        print(json.dumps({"qualification": qualification, "controls": controls}, indent=2))
        return

    parents, geometries = build_families()
    parent_records = {}
    for grid in ("G9", "G10", "Z9", "Z10"):
        summary = json_stage(
            index, f"parent/{grid}", RECOVERY_ROOT / f"parent_{grid}.json",
            lambda grid=grid: {"parent": parent_summary(parents[grid], geometries, grid)},
            kind="genuine-parent", metadata={"grid": grid},
        )["parent"]
        if (
            summary["reference_residual"] >= 1e-9 or summary["selector_residual"] >= 1e-9
            or not all(record["identity"]["passed"] for record in summary["restrictions"].values())
        ):
            raise RuntimeError(f"invalid genuine Test10E parent: {grid}")
        parent_records[grid] = summary
    expected_sizes = {"G9": [113, 253], "G10": [129, 289], "Z9": [113, 217], "Z10": [129, 217]}
    size_gate = all(parent_records[grid]["grid_size"] == size for grid, size in expected_sizes.items())
    distinct_gate = bool(
        not np.array_equal(parents["G9"]["psi"], parents["Z9"]["psi"])
        and not np.array_equal(parents["G10"]["psi"], parents["Z10"]["psi"])
    )
    if not size_gate or not distinct_gate:
        raise RuntimeError("Test10E parent size or physical identity control failed")

    cases = make_cases(geometries)
    primary_labels = [f"{grid}_{domain}" for grid in PRIMARY_GRIDS for domain in DOMAINS]
    initial = {
        label: initial_stage(index, label, cases[label]) for label in primary_labels
    }
    if not all(
        record["static"]["accepted_count"] == 0 and record["BVP"]["admitted_distinct_count"] == 0
        for record in initial.values()
    ):
        raise RuntimeError("Test10E initial cap control failed")

    run_labels = primary_labels + ["Z9_R10", "Z10_R10", "G10H_R10"]
    paths = {label: run_physical(index, label, cases[label]) for label in run_labels}
    histories = {label: read_history(paths[label]) for label in run_labels}

    source = json_stage(
        index, "analysis/source_grid", RECOVERY_ROOT / "analysis_source_grid.json",
        lambda: {"analysis": source_grid_analysis(histories)},
    )["analysis"]
    fixed = json_stage(
        index, "analysis/fixed_radial", RECOVERY_ROOT / "analysis_fixed_radial.json",
        lambda: {"analysis": fixed_radial_analysis(histories)},
    )["analysis"]
    temporal = json_stage(
        index, "analysis/temporal", RECOVERY_ROOT / "analysis_temporal.json",
        lambda: {"analysis": temporal_analysis(histories)},
    )["analysis"]
    locality = json_stage(
        index, "analysis/locality", RECOVERY_ROOT / "analysis_locality.json",
        lambda: {"analysis": {key: value for key, value in locality_analysis(cases).items() if key != "_state_arrays"}},
    )["analysis"]
    locality_full = locality_analysis(cases)
    boundary = json_stage(
        index, "analysis/boundary_model_locality", RECOVERY_ROOT / "analysis_boundary_model_locality.json",
        lambda: {"analysis": boundary_model_locality(histories)},
    )["analysis"]
    causal = json_stage(
        index, "analysis/causality", RECOVERY_ROOT / "analysis_causality.json",
        lambda: {"analysis": causal_analysis(geometries)},
    )["analysis"]
    norm_response = json_stage(
        index, "analysis/normalization_response", RECOVERY_ROOT / "analysis_normalization_response.json",
        lambda: {"analysis": normalization_response(histories)},
    )["analysis"]
    quality = json_stage(
        index, "analysis/technical_quality", RECOVERY_ROOT / "analysis_technical_quality.json",
        lambda: {"analysis": all_new_stage_quality(histories)},
    )["analysis"]

    searches = {label: {} for label in primary_labels}
    for label in primary_labels:
        for step in range(1, 17):
            searches[label][step] = detector_stage(
                index, label, step, load_endpoint(label, step, cases[label]),
                geometries[label], PRIMARY_DT,
            )
    half_searches = {}
    for step in range(2, 33, 2):
        half_searches[step] = detector_stage(
            index, "G10H_R10", step, load_endpoint("G10H_R10", step, cases["G10H_R10"]),
            geometries["G10H_R10"], HALF_DT,
        )
    formation = json_stage(
        index, "analysis/formation", RECOVERY_ROOT / "analysis_formation.json",
        lambda: {"analysis": formation_analysis(initial, searches, half_searches)},
    )["analysis"]

    geometry_records = {label: {} for label in primary_labels}
    for label in primary_labels:
        for step in GEOMETRY_STEPS:
            geometry_records[label][step] = geometry_stage(
                index, label, step, searches[label][step],
                load_endpoint(label, step, cases[label]), geometries[label],
            )
    surfaces = json_stage(
        index, "analysis/surfaces", RECOVERY_ROOT / "analysis_surfaces.json",
        lambda: {"analysis": surface_analysis(geometry_records)},
    )["analysis"]

    primary = {
        "source": source, "fixed_radial": fixed, "temporal": temporal,
        "locality": locality, "boundary_model_locality": boundary,
        "formation": formation, "surface": surfaces,
        "normalization_response": norm_response, "quality": quality,
        "initial": initial,
    }
    independent = json_stage(
        index, "analysis/independent", RECOVERY_ROOT / "analysis_independent.json",
        lambda: {"analysis": independent_recompute(
            histories, cases, searches, half_searches, geometry_records, primary,
        )},
    )["analysis"]
    assemble_state(index, histories, locality_full)
    prefinal_valid = bool(all(
        record.get("status") == "complete" and index.validated_path(stage_id) is not None
        for stage_id, record in index.data["stages"].items()
    ))
    construction = bool(size_gate and distinct_gate and all(
        record["reference_residual"] < 1e-9 and record["selector_residual"] < 1e-9
        and all(item["identity"]["passed"] for item in record["restrictions"].values())
        for record in parent_records.values()
    ))
    technical = bool(all(quality.values()))
    valid = bool(
        qualification["passed"] and controls["passed"] and construction and technical
        and formation["gate"] and surfaces["gate"] and independent["gate"]
        and prefinal_valid
    )
    convergence = bool(source["gate"] and boundary["gate"] and locality["gate"] and causal["gate"])
    status, classification = classify_test10e(
        valid, norm_response["uncontrolled_legacy_response"],
        locality["resolved_contamination"], convergence,
        fixed["gate"], temporal["gate"],
        norm_response["response_on_both_high_grids"], norm_response["normalization_rule"],
    )
    if valid and locality["indeterminate_band"]:
        status, classification = "review", "mixed_high_z_boundary_diagnosis"
    payload = {
        "status": status, "classification": classification,
        "scope": "sealed Test-10E genuine high-z source-grid boundary-resolution audit",
        "protocol": str(PROTOCOL), "protocol_sha256": index.protocol_sha256,
        "preserved_prior_grades": {
            "test10b": "review: invalid_common_parent_audit",
            "test10c": "review: invalid_outer_scalar_audit",
            "test10d": "review: invalid_refined_boundary_audit",
            "legacy_threshold": 0.05,
        },
        "qualification": qualification, "controls": controls,
        "parents": parent_records, "construction_gate": construction,
        "source_grid_analysis": source, "fixed_radial_analysis": fixed,
        "temporal_analysis": temporal, "locality_analysis": locality,
        "boundary_model_locality": boundary, "causal_analysis": causal,
        "formation_analysis": formation, "surface_analysis": surfaces,
        "normalization_response": norm_response, "technical_quality": quality,
        "independent_audit": independent,
        "acceptance": {
            "valid_provenance_and_controls": qualification["passed"] and controls["passed"],
            "genuine_parent_construction": construction,
            "all_new_stage_quality": technical,
            "source_grid_convergence": source["gate"],
            "fixed_radial_separation": fixed["gate"],
            "temporal_separation": temporal["gate"],
            "common_interior_locality": locality["gate"],
            "boundary_model_locality": boundary["gate"],
            "causal_reach": causal["gate"],
            "formation": formation["gate"], "surface_and_stability": surfaces["gate"],
            "independent_reproduction": independent["gate"],
            "prefinal_recovery": prefinal_valid,
        },
        "runtime": {
            "wall_seconds_this_invocation": time.perf_counter() - started,
            "cumulative_stage_compute_seconds": float(sum(
                record.get("elapsed_seconds", 0.0) for record in index.data["stages"].values()
            )),
        },
        "limitations": [
            "prior Test10B/C/D REVIEW grades remain immutable",
            "an artificial-boundary response is not bulk physics",
            "clean r<=6 evidence is limited to t<=.002",
            "not an event horizon, topology, halo, mass-transfer, throat, or nonlinear-stability result",
        ],
    }
    index.register("final/result", "combined-result", 300.0, {"status": status})
    cached = index.validated_path("final/result")
    if cached is None:
        index.mark_running("final/result")
        atomic_write_json(OUTPUT, payload)
        index.mark_complete("final/result", OUTPUT, 0.0)
    else:
        payload = json.loads(cached.read_text())
    print(json.dumps({
        "status": payload["status"], "classification": payload["classification"],
        "acceptance": payload["acceptance"],
        "high_grid_maxima": {
            key: value for key, value in payload["source_grid_analysis"]["run_maxima"].items()
            if key.startswith(("G9_", "G10_"))
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
