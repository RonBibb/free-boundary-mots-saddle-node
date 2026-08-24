#!/usr/bin/env python3
"""Run the prospectively sealed Test-10D refined boundary audit."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import run_corrected_A790_test10b_domain_normalized as test10b
import run_corrected_A790_test10c_outer_scalar_closure as test10c
from bhps.corrected_A790_test10d_boundary_resolution import (
    LEVELS,
    METRIC_KEYS,
    SCHEMES,
    classify_test10d,
    closure_gate,
    ensemble_metrics,
    evaluate_all_levels,
    pointwise_ratio,
    refinement_flags,
    source_grid_close,
)
from bhps.recovery_indexer import (
    RecoveryIndex,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
    validate_npz,
)


PROTOCOL = Path("notes/106_A790_test10D_boundary_resolution_audit_protocol.md")
PROTOCOL_SHA256 = "143a683c30ee3a1f3b5423eeeadc0c6baf61ead5c9f271cee184a46224fde2ab"
OUTPUT = Path("results/corrected_A790_test10d_boundary_resolution.json")
STATE_OUTPUT = Path("results/corrected_A790_test10d_boundary_resolution_state.npz")
RECOVERY_ROOT = Path("results/corrected_A790_test10d_boundary_resolution_recovery")
MANIFEST = RECOVERY_ROOT / "index.json"
RUNS = tuple(
    test10b.run_label(grid, domain)
    for grid in test10b.GRIDS for domain in test10b.DOMAINS
)

FIXED_INPUT_HASHES = {
    "notes/98_A790_domain_normalized_family_protocol.md": "7c0298c67d03654b2e24c7c8602a77a8c70bd75beba14645fdd76adfef11cc23",
    "notes/98_A790_domain_normalized_family_result.md": "618b72fb125ddc67c8ea523789687a4836216fd52bda26e59223c8163a4e42d0",
    "notes/103_A790_test10C_outer_scalar_closure_audit_protocol.md": "73a4ebaec15c53477dbf2a270b55c8e7c688a987513ce53fadba432547cffe50",
    "notes/103_A790_test10C_outer_scalar_closure_audit_result.md": "f04352c2c895339cea9d1860e168c97564deffcf3bc762dc292922765ee14f1f",
    "results/corrected_A790_test10b_domain_normalized.json": "08c91892930c05fef6afe37d0a4f5ed1c49f545a84b6df4c81e08c8f2cbd1bd3",
    "results/corrected_A790_test10b_domain_normalized_state.npz": "b19e7e8901fdade6abb485c65a3512bf327a86f13974fb9a8c0876340a503928",
    "results/corrected_A790_test10b_domain_normalized_recovery/index.json": "7d064cfc491a0d2772c051cc1d10985ea936d320f26651a9423fa5ddb9947a0b",
    "results/corrected_A790_test10c_outer_scalar_closure.json": "e929986f879dd94818cbdf0a3e067c34a09201df70928c4166402d54684cebbb",
    "results/corrected_A790_test10c_outer_scalar_closure_state.npz": "bc1f99abb3a67273d7cc8d2f05ee807f9ea342d08df6721421d0894dc45adec6",
    "results/corrected_A790_test10c_outer_scalar_closure_recovery/index.json": "0e83c686545fd052ce5f9dadd6e231ec9ccaaf979fe1e44c4fb25b4dc8d1dece",
    "src/bhps/recovery_indexer.py": "1460478fba42433bd340a2ef9e09c0946882a35d3eb63c2c95ea9b055bb549fa",
    "run_corrected_A790_test10c_outer_scalar_closure.py": "06d4eb18aac3de51830bc7556588118e7838275bbdd2996b3cd78b6d3500f456",
    "src/bhps/corrected_A790_test10c_outer_scalar.py": "5fefceb14b4b5671bfabbcc0485e076a5272be8d80097139b17f57447d6d63e7",
}


def recovery_inputs():
    dynamic = (
        Path(__file__),
        Path("src/bhps/corrected_A790_test10d_boundary_resolution.py"),
        Path("tests/test_A790_test10d_boundary_resolution.py"),
        Path("tests/test_A790_test10d_runner_recovery.py"),
    )
    return {
        **FIXED_INPUT_HASHES,
        **{str(path): sha256_file(path) for path in dynamic},
    }


def relative_error(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return float(np.linalg.norm(left - right) / max(
        np.linalg.norm(left), np.linalg.norm(right), 1e-300,
    ))


def validate_sources():
    if sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("Test10D protocol identity changed")
    for path, expected in FIXED_INPUT_HASHES.items():
        if sha256_file(Path(path)) != expected:
            raise RuntimeError(f"fixed input identity changed: {path}")
    index_b = RecoveryIndex(
        test10b.MANIFEST, test10b.PROTOCOL, test10b.recovery_inputs(),
        maximum_stage_seconds=2400.0,
    )
    invalid_b = [
        key for key, record in index_b.data["stages"].items()
        if record.get("status") != "complete" or index_b.validated_path(key) is None
    ]
    index_c = RecoveryIndex(
        test10c.MANIFEST, test10c.PROTOCOL, test10c.recovery_inputs(),
        maximum_stage_seconds=2400.0,
    )
    invalid_c = [
        key for key, record in index_c.data["stages"].items()
        if record.get("status") != "complete" or index_c.validated_path(key) is None
    ]
    if len(index_b.data["stages"]) != 160 or invalid_b:
        raise RuntimeError(f"invalid Test10B archive: {invalid_b}")
    if len(index_c.data["stages"]) != 31 or invalid_c:
        raise RuntimeError(f"invalid Test10C archive: {invalid_c}")
    return {
        "test10b_stage_count": 160,
        "test10c_stage_count": 31,
        "invalid_test10b": invalid_b,
        "invalid_test10c": invalid_c,
    }


def json_stage(index, stage_id, path, builder, kind="analysis"):
    index.register(stage_id, kind, 1200.0, {})
    cached = index.validated_path(stage_id)
    if cached is not None:
        return json.loads(cached.read_text())
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        payload = builder()
        atomic_write_json(path, payload)
        index.mark_complete(stage_id, path, time.perf_counter() - started)
        return payload
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def manufactured_controls():
    z = np.linspace(1.0, 2.0, 65)
    q_perp = np.ones_like(z)
    q_zz = np.ones_like(z)
    before = np.column_stack((1.0 + 2.0 * z, 0.5 + z))
    delta = np.column_stack((0.1 * (1.0 + 2.0 * z), -0.05 * (0.5 + z)))
    after = before + delta
    zero = np.zeros_like(before)
    base_fields = {
        "delta": delta,
        "before": before,
        "after": after,
        "reference": before.copy(),
        "term_a": -delta,
        "term_v": zero.copy(),
        "term_c": zero.copy(),
        "term_sum": -delta,
    }
    radius = 3.0
    records = evaluate_all_levels(z, radius, q_perp, q_zz, base_fields, 1.0)
    flags = refinement_flags(records)
    first_integral = lambda x: x + 2.0 * x**2 + (4.0 / 3.0) * x**3
    second_integral = lambda x: x**3 / 3.0 + 0.5 * x**2 + 0.25 * x
    exact_squared = 4.0 * np.pi * radius**2 * (
        first_integral(2.0) - first_integral(1.0)
        + second_integral(2.0) - second_integral(1.0)
    )
    exact = float(np.sqrt(exact_squared))
    analytic_errors = {
        scheme: abs(records[scheme][16]["norm_before"] - exact) / exact
        for scheme in SCHEMES
    }
    scale_values = []
    for scale in (1e-6, 1.0, 1e6):
        scaled = {key: value * scale for key, value in base_fields.items()}
        result = ensemble_metrics(evaluate_all_levels(
            z, radius, q_perp, q_zz, scaled, scale,
        ))
        scale_values.append(np.asarray([
            result["proper_ratio"], result["term_balance_ratio"],
            result["component_phi_ratio"], result["component_chi_ratio"],
            pointwise_ratio(scaled["delta"], scaled["before"], scaled["after"]),
        ]))
    scale_error = max(relative_error(value, scale_values[1]) for value in scale_values)
    spike_fields = {key: value.copy() for key, value in base_fields.items()}
    spike = np.zeros_like(delta)
    spike[0] = (1.0, -0.5)
    spike_fields["delta"] = spike
    spike_fields["after"] = before + spike
    spike_fields["term_a"] = -spike
    spike_fields["term_sum"] = -spike
    spike_records = evaluate_all_levels(z, radius, q_perp, q_zz, spike_fields, 1.0)
    spike_exposed = not refinement_flags(spike_records)["cross_scheme"]["norm_delta"]
    positivity_rejected = False
    try:
        bad = q_perp.copy()
        bad[10] = 0.0
        evaluate_all_levels(z, radius, bad, q_zz, base_fields, 1.0)
    except ValueError:
        positivity_rejected = True
    gates = {
        "analytic": max(analytic_errors.values()) < 1e-10,
        "internal_refinement": all(
            flags[scheme][key] for scheme in SCHEMES for key in METRIC_KEYS
        ),
        "cross_scheme_analytic": all(flags["cross_scheme"].values()),
        "scale_invariance": scale_error < 1e-10,
        "exact_closure": closure_gate(0.0, 5e-17),
        "adverse_closure": not closure_gate(2e-18, 0.0),
        "exact_target": closure_gate(0.0, 5e-17),
        "adverse_target": not closure_gate(2e-18, 0.0),
        "single_node_spike_exposed": spike_exposed,
        "positivity_rejected": positivity_rejected,
    }
    return {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "analytic_relative_errors": analytic_errors,
        "scale_invariance_error": scale_error,
        "spike_finest_values": {
            scheme: spike_records[scheme][16]["norm_delta"] for scheme in SCHEMES
        },
    }


def checkpoint_path(label, start, end):
    return RECOVERY_ROOT / f"analysis_{label}_steps_{start + 1:03d}_{end:03d}.npz"


def enumeration(run_index, step, rk_stage):
    return 1 + 32 * int(run_index) + 2 * (int(step) - 1) + (int(rk_stage) - 1)


def refined_key(scheme, level, metric):
    return f"{scheme}_L{int(level):02d}_{metric}"


def validate_checkpoint(path, count, enumeration_start, enumeration_end):
    scalar_keys = (
        "enumeration", "step", "rk_stage", "time", "legacy_ratio",
        "pointwise_ratio", "maximum_absolute_correction", "collar_rms",
        "closure_absolute", "closure_scale", "closure_pass",
        "target_absolute", "target_scale", "target_pass",
        "production_residual", "production_pass", "stage_position_error",
    )
    required = {key: (count,) for key in scalar_keys}
    for scheme in SCHEMES:
        for level in LEVELS:
            for metric in METRIC_KEYS:
                required[refined_key(scheme, level, metric)] = (count,)
        for metric in METRIC_KEYS:
            required[f"{scheme}_converged_{metric}"] = (count,)
    for metric in METRIC_KEYS:
        required[f"cross_scheme_{metric}"] = (count,)
        required[f"ensemble_{metric}"] = (count,)
    validate_npz(path, required)
    with np.load(path) as archive:
        values = np.asarray(archive["enumeration"], dtype=int)
        if not np.array_equal(values, np.arange(enumeration_start, enumeration_end + 1)):
            raise RuntimeError("Test10D enumeration mismatch")


def source_stage_positions(case, source, start, end):
    if start == 0:
        position = case["initial"].copy()
        velocity = np.zeros_like(position)
        expected_start_position = case["initial"]
        expected_start_velocity = np.zeros_like(position)
    else:
        previous = test10b.segment_path(case["_test10d_label"], start - 4, start)
        with np.load(previous) as archive:
            position = np.asarray(archive["end_position"])
            velocity = np.asarray(archive["end_velocity"])
            expected_start_position = np.asarray(archive["end_position"])
            expected_start_velocity = np.asarray(archive["end_velocity"])
    records = []
    errors = []
    for step in range(start + 1, end + 1):
        if step == start + 1:
            expected_position_before = expected_start_position
            expected_velocity_before = expected_start_velocity
        else:
            expected_position_before = case["initial"] + source[f"step_{step - 1:03d}_increment"]
            expected_velocity_before = source[f"step_{step - 1:03d}_velocity"]
        reconstruction_error = max(
            relative_error(position, expected_position_before),
            relative_error(velocity, expected_velocity_before),
        )
        records.extend((position.copy(), position + 0.5 * test10b.DT * velocity))
        expected_position = case["initial"] + source[f"step_{step:03d}_increment"]
        expected_velocity = source[f"step_{step:03d}_velocity"]
        errors.extend((reconstruction_error, reconstruction_error))
        position = expected_position
        velocity = expected_velocity
    endpoint_error = max(
        relative_error(position, source["end_position"]),
        relative_error(velocity, source["end_velocity"]),
    )
    if endpoint_error >= 1e-13:
        raise RuntimeError("Test10D source-stage reconstruction mismatch")
    return records, errors


def analyze_segment(index, run_index, label, case, start, end):
    source_b_path = test10b.segment_path(label, start, end)
    source_c_path = test10c.checkpoint_path(label, start, end)
    source_hashes = {"test10b": sha256_file(source_b_path), "test10c": sha256_file(source_c_path)}
    enum_start = enumeration(run_index, start + 1, 1)
    enum_end = enumeration(run_index, end, 2)
    stage_id = f"analysis/{label}/enumeration_{enum_start:03d}_{enum_end:03d}"
    path = checkpoint_path(label, start, end)
    metadata = {
        "run": label, "start": start, "end": end,
        "enumeration_start": enum_start, "enumeration_end": enum_end,
        "source_hashes": source_hashes,
        "case_fingerprint": test10b.case_fingerprint(case),
    }
    index.register(stage_id, "refined-boundary-analysis", 1200.0, metadata)
    cached = index.validated_path(stage_id)
    if cached is not None:
        validate_checkpoint(cached, 2 * (end - start), enum_start, enum_end)
        return cached
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        with np.load(source_b_path) as source_b, np.load(source_c_path) as source_c:
            positions, position_errors = source_stage_positions(case, source_b, start, end)
            count = 2 * (end - start)
            expected_enumeration = np.arange(enum_start, enum_end + 1)
            if not np.array_equal(source_c["enumeration"], expected_enumeration):
                raise RuntimeError("Test10C source enumeration changed")
            records = []
            source_times = np.asarray(source_c["time"], dtype=float).copy()
            reference = np.asarray(
                case["rhs"].outer_reference_acceleration[1:-1, -1, 7:9],
                dtype=float,
            )
            z_open = np.asarray(case["z"][1:-1], dtype=float)
            radius = float(case["r"][-1])
            for local in range(count):
                print(
                    f"{label}: Test10D enumeration {expected_enumeration[local]}/192",
                    flush=True,
                )
                before = np.asarray(source_c["before"][local], dtype=float)
                after = np.asarray(source_c["after"][local], dtype=float)
                delta = after - before
                term_a = np.asarray(source_c["term_A"][local], dtype=float)
                term_v = np.asarray(source_c["term_V"][local], dtype=float)
                term_c = np.asarray(source_c["term_C"][local], dtype=float)
                position = positions[local]
                fields = {
                    "delta": delta, "before": before, "after": after,
                    "reference": reference, "term_a": term_a,
                    "term_v": term_v, "term_c": term_c,
                    "term_sum": term_a + term_v + term_c,
                }
                refined = evaluate_all_levels(
                    z_open, radius, position[1:-1, -1, 3],
                    position[1:-1, -1, 6], fields,
                    float(source_c["collar_rms_before"][local]),
                )
                flags = refinement_flags(refined)
                ensemble = ensemble_metrics(refined)
                closure = term_a + term_v + term_c + delta
                closure_absolute = float(np.max(np.abs(closure)))
                closure_scale = float(max(
                    np.max(np.abs(term_a)), np.max(np.abs(term_v)),
                    np.max(np.abs(term_c)), np.max(np.abs(delta)),
                ))
                target = np.asarray(source_c["target"][local], dtype=float)
                target_absolute = float(np.max(np.abs(after - target)))
                target_scale = float(max(np.max(np.abs(after)), np.max(np.abs(target))))
                records.append({
                    "refined": refined, "flags": flags, "ensemble": ensemble,
                    "legacy_ratio": float(source_c["legacy_ratio"][local]),
                    "pointwise_ratio": pointwise_ratio(delta, before, after),
                    "maximum_absolute_correction": float(np.max(np.abs(delta))),
                    "collar_rms": float(source_c["collar_rms_before"][local]),
                    "closure_absolute": closure_absolute,
                    "closure_scale": closure_scale,
                    "closure_pass": closure_gate(closure_absolute, closure_scale),
                    "target_absolute": target_absolute,
                    "target_scale": target_scale,
                    "target_pass": closure_gate(target_absolute, target_scale),
                    "production_residual": float(source_c["production_residual"][local]),
                    "production_pass": float(source_c["production_residual"][local]) < 1e-10,
                })
            if max(position_errors, default=0.0) >= 1e-13:
                raise RuntimeError("Test10D stage-position mismatch")
            if not np.array_equal(source_c["step"], np.repeat(np.arange(start + 1, end + 1), 2)):
                raise RuntimeError("Test10C step alignment changed")
        arrays = {
            "enumeration": expected_enumeration,
            "step": np.repeat(np.arange(start + 1, end + 1), 2),
            "rk_stage": np.tile((1, 2), end - start),
            "time": source_times,
            "stage_position_error": np.asarray(position_errors),
        }
        scalar_keys = (
            "legacy_ratio", "pointwise_ratio", "maximum_absolute_correction",
            "collar_rms", "closure_absolute", "closure_scale", "closure_pass",
            "target_absolute", "target_scale", "target_pass",
            "production_residual", "production_pass",
        )
        for key in scalar_keys:
            arrays[key] = np.asarray([record[key] for record in records])
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
        atomic_write_npz(path, **arrays)
        validate_checkpoint(path, len(records), enum_start, enum_end)
        index.mark_complete(
            stage_id, path, time.perf_counter() - started,
            {"maximum_stage_position_error": max(position_errors, default=0.0)},
        )
        return path
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def combined_analysis(paths):
    result_c = json.loads(test10c.OUTPUT.read_text())
    runs = {}
    all_refinement = True
    all_cross = True
    all_closure = True
    all_target = True
    all_production = True
    all_finite = True
    stage_count = 0
    legacy_errors = []
    for label in RUNS:
        records = {key: [] for key in (
            "legacy_ratio", "pointwise_ratio", "maximum_absolute_correction",
            *[f"ensemble_{metric}" for metric in METRIC_KEYS],
        )}
        response_stages = 0
        normalization_at_excess = True
        for path in paths[label]:
            with np.load(path) as archive:
                count = len(archive["enumeration"])
                stage_count += count
                for key in records:
                    records[key].extend(np.asarray(archive[key], dtype=float).tolist())
                for scheme in SCHEMES:
                    for metric in METRIC_KEYS:
                        all_refinement &= bool(np.all(archive[f"{scheme}_converged_{metric}"]))
                for metric in METRIC_KEYS:
                    all_cross &= bool(np.all(archive[f"cross_scheme_{metric}"]))
                all_closure &= bool(np.all(archive["closure_pass"]))
                all_target &= bool(np.all(archive["target_pass"]))
                all_production &= bool(np.all(archive["production_pass"]))
                all_finite &= all(np.all(np.isfinite(archive[key])) for key in archive.files)
        values = {key: np.asarray(value) for key, value in records.items()}
        response_mask = (
            (values["ensemble_proper_ratio"] >= 0.05)
            | (values["ensemble_collar_ratio"] >= 0.05)
            | (values["ensemble_component_phi_ratio"] >= 0.05)
            | (values["ensemble_component_chi_ratio"] >= 0.05)
            | (values["ensemble_term_balance_ratio"] >= 0.05)
            | (values["pointwise_ratio"] >= 0.20)
        )
        response_stages = int(np.sum(response_mask))
        excess = values["legacy_ratio"] >= 0.05
        if np.any(excess):
            normalization_mask = (
                (values["ensemble_proper_ratio"] < 0.05)
                & (values["ensemble_collar_ratio"] < 0.05)
                & (values["ensemble_component_phi_ratio"] < 0.05)
                & (values["ensemble_component_chi_ratio"] < 0.05)
                & (values["ensemble_term_balance_ratio"] < 0.05)
                & (values["pointwise_ratio"] < 0.20)
            )
            normalization_at_excess = bool(np.all(normalization_mask[excess]))
        immutable = result_c["replay_analysis"]["runs"][label]["legacy_maximum"]
        legacy_maximum = float(np.max(values["legacy_ratio"]))
        legacy_errors.append(abs(legacy_maximum - immutable))
        runs[label] = {
            "legacy_maximum": legacy_maximum,
            "legacy_reproduction_absolute_error": abs(legacy_maximum - immutable),
            "legacy_excess_stage_count": int(np.sum(excess)),
            "response_stage_count": response_stages,
            "response_present": response_stages > 0,
            "normalization_at_all_legacy_excesses": normalization_at_excess,
            "maxima": {
                key.removeprefix("ensemble_"): float(np.max(value))
                for key, value in values.items() if key.startswith("ensemble_")
            } | {
                "pointwise_ratio": float(np.max(values["pointwise_ratio"])),
                "maximum_absolute_correction": float(np.max(values["maximum_absolute_correction"])),
            },
        }
    response_by_grid = {
        grid: {domain: runs[f"{grid}_{domain}"]["response_present"] for domain in test10b.DOMAINS}
        for grid in test10b.GRIDS
    }
    response = any(
        response_by_grid["G7"][domain] and response_by_grid["G8"][domain]
        for domain in test10b.DOMAINS
    )
    normalization = bool(all(
        record["normalization_at_all_legacy_excesses"] for record in runs.values()
    ))
    return {
        "stage_count": stage_count,
        "all_finite": all_finite,
        "all_internal_refinement": all_refinement,
        "all_cross_scheme_agreement": all_cross,
        "all_dual_closure": all_closure,
        "all_independent_targets": all_target,
        "all_production_residuals": all_production,
        "all_legacy_maxima_reproduced": max(legacy_errors, default=0.0) == 0.0,
        "normalization_rule": normalization,
        "response_rule": response,
        "response_by_grid": response_by_grid,
        "runs": runs,
    }


GRID_METRICS = (
    "proper_ratio", "term_balance_ratio", "component_phi_ratio",
    "component_chi_ratio", "collar_ratio",
)


def source_grid_analysis(combined):
    records = {}
    all_pass = True
    for domain in test10b.DOMAINS:
        left = combined["runs"][f"G7_{domain}"]["maxima"]
        right = combined["runs"][f"G8_{domain}"]["maxima"]
        domain_records = {}
        for metric in GRID_METRICS:
            passed = source_grid_close(left[metric], right[metric])
            domain_records[metric] = {
                "G7": left[metric], "G8": right[metric],
                "absolute_difference": abs(left[metric] - right[metric]),
                "passes": passed,
            }
            all_pass &= passed
        passed = source_grid_close(
            left["pointwise_ratio"], right["pointwise_ratio"], pointwise=True,
        )
        domain_records["pointwise_ratio"] = {
            "G7": left["pointwise_ratio"], "G8": right["pointwise_ratio"],
            "absolute_difference": abs(left["pointwise_ratio"] - right["pointwise_ratio"]),
            "passes": passed,
        }
        all_pass &= passed
        records[domain] = domain_records
    return {"gate": bool(all_pass), "records": records}


def locality_analysis():
    result_b = json.loads(test10b.OUTPUT.read_text())
    result_c = json.loads(test10c.OUTPUT.read_text())
    contamination = result_c["contamination_analysis"]
    causal = result_b["causal_analysis"]
    return {
        "common_r6_maximum": contamination["common_r6_maximum"],
        "no_common_interior_contamination": contamination["no_common_interior_contamination"],
        "resolved_contamination": contamination["resolved_contamination"],
        "indeterminate_band": contamination["indeterminate_band"],
        "causal_gate": causal["gate"],
        "causal_records": causal["records"],
        "gate": bool(
            contamination["no_common_interior_contamination"]
            and not contamination["indeterminate_band"] and causal["gate"]
        ),
    }


def assemble_state(index, combined, grid, locality):
    arrays = {}
    for label, record in combined["runs"].items():
        arrays[f"{label}_maxima"] = np.asarray([
            record["legacy_maximum"], record["maxima"]["proper_ratio"],
            record["maxima"]["term_balance_ratio"],
            record["maxima"]["pointwise_ratio"],
            record["maxima"]["collar_ratio"],
        ])
    arrays["source_grid_pass"] = np.asarray([grid["gate"]], dtype=int)
    arrays["common_r6_maximum"] = np.asarray([locality["common_r6_maximum"]])
    stage_id = "final/state"
    index.register(stage_id, "state-archive", 300.0, {"arrays": len(arrays)})
    cached = index.validated_path(stage_id)
    if cached is None:
        index.mark_running(stage_id)
        started = time.perf_counter()
        atomic_write_npz(STATE_OUTPUT, **arrays)
        validate_npz(STATE_OUTPUT)
        index.mark_complete(stage_id, STATE_OUTPUT, time.perf_counter() - started)


def main():
    started = time.perf_counter()
    index = RecoveryIndex(MANIFEST, PROTOCOL, recovery_inputs(), maximum_stage_seconds=1200.0)
    qualification = json_stage(
        index, "qualification/sources", RECOVERY_ROOT / "qualification_sources.json",
        lambda: {"qualification": validate_sources()}, "qualification",
    )["qualification"]
    controls = json_stage(
        index, "controls/manufactured", RECOVERY_ROOT / "controls.json",
        lambda: {"controls": manufactured_controls()}, "controls",
    )["controls"]
    if not controls["passed"]:
        raise RuntimeError("Test10D manufactured/adverse controls failed")
    print("constructing frozen Test10D common-parent references", flush=True)
    _, geometries = test10b.build_geometries()
    cases = test10b.make_cases(geometries)
    with np.load(test10b.STATE_OUTPUT) as state_b:
        for label in RUNS:
            case = cases[label]
            case["_test10d_label"] = label
            if relative_error(case["initial"], state_b[f"{label}_initial"]) >= 1e-13:
                raise RuntimeError(f"Test10D initial archive mismatch: {label}")
            if relative_error(case["z"], state_b[f"{label}_z"]) >= 1e-13:
                raise RuntimeError(f"Test10D z archive mismatch: {label}")
            if relative_error(case["r"], state_b[f"{label}_r"]) >= 1e-13:
                raise RuntimeError(f"Test10D r archive mismatch: {label}")
    paths = {label: [] for label in RUNS}
    for run_index, label in enumerate(RUNS):
        for start in range(0, test10b.STEPS, test10b.SEGMENT):
            end = min(start + test10b.SEGMENT, test10b.STEPS)
            paths[label].append(analyze_segment(
                index, run_index, label, cases[label], start, end,
            ))
    combined = json_stage(
        index, "analysis/refinement", RECOVERY_ROOT / "analysis_refinement.json",
        lambda: {"analysis": combined_analysis(paths)},
    )["analysis"]
    grid = json_stage(
        index, "analysis/source_grid", RECOVERY_ROOT / "analysis_source_grid.json",
        lambda: {"analysis": source_grid_analysis(combined)},
    )["analysis"]
    locality = json_stage(
        index, "analysis/locality", RECOVERY_ROOT / "analysis_locality.json",
        lambda: {"analysis": locality_analysis()},
    )["analysis"]
    independent = json_stage(
        index, "analysis/independent", RECOVERY_ROOT / "analysis_independent.json",
        lambda: {"analysis": {
            "gate": bool(
                combined["all_cross_scheme_agreement"]
                and combined["all_dual_closure"]
                and combined["all_independent_targets"]
            ),
            "evaluator": "natural cubic primitive interpolation plus cellwise Romberg",
            "all_cross_scheme_agreement": combined["all_cross_scheme_agreement"],
            "all_dual_closure": combined["all_dual_closure"],
            "all_independent_targets": combined["all_independent_targets"],
        }},
    )["analysis"]
    base_valid = bool(
        qualification["test10b_stage_count"] == 160
        and qualification["test10c_stage_count"] == 31
        and controls["passed"] and combined["stage_count"] == 192
        and combined["all_finite"] and combined["all_internal_refinement"]
        and combined["all_cross_scheme_agreement"]
        and combined["all_dual_closure"] and combined["all_independent_targets"]
        and combined["all_production_residuals"]
        and combined["all_legacy_maxima_reproduced"] and independent["gate"]
    )
    assemble_state(index, combined, grid, locality)
    prefinal_valid = bool(all(
        record.get("status") == "complete" and index.validated_path(stage_id) is not None
        for stage_id, record in index.data["stages"].items()
    ))
    valid = bool(base_valid and prefinal_valid)
    normalization = bool(combined["normalization_rule"] and locality["gate"])
    response = bool(combined["response_rule"] and locality["gate"])
    status, classification = classify_test10d(
        valid, locality["resolved_contamination"], locality["indeterminate_band"],
        normalization, response, grid["gate"],
    )
    payload = {
        "status": status,
        "classification": classification,
        "scope": "sealed Test-10D refined artificial-boundary audit",
        "protocol": str(PROTOCOL),
        "protocol_sha256": index.protocol_sha256,
        "preserved_prior_grades": {
            "test10b": "review: invalid_common_parent_audit",
            "test10c": "review: invalid_outer_scalar_audit",
            "legacy_threshold": 0.05,
        },
        "qualification": qualification,
        "controls": controls,
        "refinement_analysis": combined,
        "source_grid_analysis": grid,
        "locality_analysis": locality,
        "independent_audit": independent,
        "diagnoses": {
            "normalization_rule": normalization,
            "boundary_response_rule": response,
            "source_grid_consistent": grid["gate"],
            "resolved_contamination": locality["resolved_contamination"],
            "no_resolved_common_interior_effect": locality["gate"],
        },
        "acceptance": {
            "source_provenance": qualification["test10b_stage_count"] == 160 and qualification["test10c_stage_count"] == 31,
            "controls": controls["passed"],
            "record_alignment": combined["stage_count"] == 192,
            "internal_refinement": combined["all_internal_refinement"],
            "cross_scheme_agreement": combined["all_cross_scheme_agreement"],
            "dual_closure": combined["all_dual_closure"],
            "independent_target": combined["all_independent_targets"],
            "production_residual": combined["all_production_residuals"],
            "source_grid_consistency": grid["gate"],
            "locality_and_causality": locality["gate"],
            "prefinal_recovery": prefinal_valid,
        },
        "runtime": {
            "wall_seconds_this_invocation": time.perf_counter() - started,
            "cumulative_stage_compute_seconds": float(sum(
                record.get("elapsed_seconds", 0.0) for record in index.data["stages"].values()
            )),
        },
        "limitations": [
            "Test10B and Test10C grades remain immutable REVIEW",
            "artificial-boundary response is not bulk physics",
            "two-grid consistency is not a convergence order",
            "no resolved interior effect is limited to common r<=6 and t<=.002",
            "not an event horizon, topology, halo, mass-transfer, throat, or nonlinear-stability result",
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
        "status": payload["status"],
        "classification": payload["classification"],
        "diagnoses": payload["diagnoses"],
        "acceptance": payload["acceptance"],
        "run_maxima": {
            label: record["maxima"]
            for label, record in payload["refinement_analysis"]["runs"].items()
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
