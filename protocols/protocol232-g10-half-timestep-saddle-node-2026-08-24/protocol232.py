#!/usr/bin/env python3
"""Recoverable G10 half-timestep temporal saddle-node discriminator."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SCHEMA = "protocol232-g10-half-timestep-saddle-node-v1"
GRID_SCHEMA = "protocol232-g10-half-timestep-grid-v1"
CHECKPOINT_SCHEMA = "protocol232-half-timestep-checkpoint-v1"
ANCHOR_SCHEMA = "protocol232-half-timestep-anchors-v1"
COARSE_DT = 0.00003125
HALF_DT = 0.000015625
START_TIME = 0.001
END_TIME = 0.0015
START_STEP = 64
END_STEP = 96
CHECKPOINT_STEPS = (72, 80, 88, 96)
FIELDS = ("q", "v", "source", "memory")
THREAD_VARS = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)


class Protocol232Error(RuntimeError):
    pass


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None); raise
    return module


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path, root=None):
    path = Path(path).absolute()
    return {"path": path.relative_to(Path(root).absolute()).as_posix() if root else str(path), "byte_count": path.stat().st_size, "sha256": sha256(path)}


def same(path, item):
    path = Path(path)
    return bool(
        set(item) == {"path", "byte_count", "sha256"}
        and path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
        and type(item["byte_count"]) is int and path.stat().st_size == item["byte_count"]
        and type(item["sha256"]) is str and sha256(path) == item["sha256"]
    )


def protocol_roots(academic_root):
    base = Path(academic_root).absolute() / "Discussion/protocols"
    return {
        "p228": base / "protocol228-repaired-parent-formation-time-2026-08-23",
        "p229": base / "protocol229-free-boundary-mots-saddle-node-v4-2026-08-24",
        "p230": base / "protocol230-protocol229-archive-finalization-2026-08-24",
        "p231": base / "protocol231-saddle-node-existing-data-audit-2026-08-24",
        "p220": base / "protocol220-recoverable-canonical-spatial-2026-08-22",
        "p216": base / "protocol216-doubled-horizon-canonical-2026-08-22",
    }


def input_paths(academic_root):
    roots = protocol_roots(academic_root)
    return {
        "p229/protocol": roots["p229"] / "PROTOCOL.md",
        "p229/source": roots["p229"] / "protocol229.py",
        "p229/core": roots["p229"] / "continuation_core.py",
        "p229/authority": roots["p229"] / "authority/freeze_record.json",
        "p229/G10-json": roots["p229"] / "candidate-output/protocol229_G10.json",
        "p229/G10-npz": roots["p229"] / "candidate-output/protocol229_G10.npz",
        "p230/result": roots["p230"] / "candidate-output/protocol230_result.json",
        "p231/source": roots["p231"] / "protocol231.py",
        "p231/authority": roots["p231"] / "authority/freeze_record.json",
        "p231/result": roots["p231"] / "candidate-output/protocol231_result.json",
        "p228/source": roots["p228"] / "protocol228.py",
        "p228/authority": roots["p228"] / "authority/freeze_record.json",
        "p228/result": roots["p228"] / "candidate-output/protocol228_result.json",
        "p228/profiles": roots["p228"] / "candidate-output/protocol228_profiles.npz",
        "G10/start": roots["p216"] / "linux-output/g10_doubled_horizon_canonical.npz",
        "p220/source": roots["p220"] / "protocol220.py",
    }


def fsync_dir(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def atomic_json(path, value):
    path = Path(path); temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    if path.exists() or path.is_symlink(): raise Protocol232Error("output exists: " + path.name)
    with temporary.open("xb") as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    os.replace(temporary, path); fsync_dir(path.parent)


def atomic_npz(path, arrays):
    path = Path(path); temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    if path.exists() or path.is_symlink(): raise Protocol232Error("output exists: " + path.name)
    with temporary.open("xb") as stream:
        np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    os.replace(temporary, path); fsync_dir(path.parent)


def freeze(root, academic_root):
    root = Path(root).absolute(); inputs = input_paths(academic_root)
    if any((root / "authority").iterdir()) or (root / "candidate-output").exists():
        raise Protocol232Error("prospective namespace differs")
    for name, path in inputs.items():
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
            raise Protocol232Error("unsafe input: " + name)
    coarse = read_json(inputs["p230/result"])
    audit = read_json(inputs["p231/result"])
    if coarse.get("classification") != "FREE-BOUNDARY-MOTS-SADDLE-NODE-CLOSURE-PASS":
        raise Protocol232Error("Protocol230 prerequisite differs")
    if audit.get("classification") != "EXISTING-DATA-PRECISION-AUDIT-COMPLETE":
        raise Protocol232Error("Protocol231 prerequisite differs")
    sources = ("PROTOCOL.md", "bootstrap.py", "protocol232.py", "tests/test_protocol232.py")
    authority = {
        "schema": SCHEMA, "status": "FROZEN", "grid": "G10",
        "coarse_dt": COARSE_DT, "half_dt": HALF_DT,
        "time_interval": [START_TIME, END_TIME], "half_step_interval": [START_STEP, END_STEP],
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "critical_time_transfer_limit": COARSE_DT / 64.0,
        "critical_area_relative_limit": 0.01,
        "coefficient_relative_limit": 0.20,
        "exponent_interval": [0.40, 0.60],
        "sources": {name: record(root / name, root) for name in sources},
        "inputs": {name: record(path) for name, path in inputs.items()},
        "candidate_output_absent_at_freeze": True,
        "equation_or_parent_change_authorized": False,
        "dt_quarter_authorized": False,
        "continuum_theorem_claim_authorized": False,
        "full_nonsymmetric_spectral_claim_authorized": False,
        "event_horizon_claim_authorized": False,
        "phase_selection_claim_authorized": False,
        "source_ownership_claim_authorized": False,
    }
    authority["fingerprint"] = hashlib.sha256(b"protocol232-freeze\0" + canonical(authority)).hexdigest()
    atomic_json(root / "authority/freeze_record.json", authority)
    return authority


def verify(root, academic_root):
    root = Path(root).absolute(); authority = read_json(root / "authority/freeze_record.json")
    fingerprint = authority.pop("fingerprint", None)
    expected = hashlib.sha256(b"protocol232-freeze\0" + canonical(authority)).hexdigest(); authority["fingerprint"] = fingerprint
    if authority.get("schema") != SCHEMA or fingerprint != expected: raise Protocol232Error("authority differs")
    for name, item in authority["sources"].items():
        if not same(root / item["path"], item): raise Protocol232Error("source differs: " + name)
    inputs = input_paths(academic_root)
    if set(inputs) != set(authority["inputs"]): raise Protocol232Error("input inventory differs")
    for name, item in authority["inputs"].items():
        if not same(inputs[name], item): raise Protocol232Error("input differs: " + name)
    return authority, inputs


def runtime_preflight(root):
    if not sys.dont_write_bytecode or any(os.environ.get(name) != "1" for name in THREAD_VARS):
        raise Protocol232Error("runtime controls differ")
    expected = (Path(root) / "bootstrap.py").resolve()
    observed = Path(getattr(sys.modules.get("__main__"), "__file__", "")).resolve()
    if observed != expected: raise Protocol232Error("authorized bootstrap was bypassed")


def checkpoint_fingerprint(value):
    return hashlib.sha256(b"protocol232-checkpoint\0" + canonical(value)).hexdigest()


def arrays_from_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(FIELDS): raise Protocol232Error("checkpoint inventory differs")
        return {name: np.ascontiguousarray(archive[name]) for name in FIELDS}


def endpoint_gate(context, state, time_value):
    bundle = context["bundles"]["G10"]
    stage = context["runner"].reconstruct_driver_stage(bundle, context["p228"].MODE, time_value, state, capture=True)
    audit = context["runner"]._technical_stage_audit(bundle, context["p228"].MODE, stage)
    return bool(stage["finite"] and audit["gates"]["all_technical_gates_pass"]), context["p220"].P190.jsonable(audit["gates"])


def validate_checkpoint(output, context, authority_sha, start_step, end_step, previous_sha):
    name = f"G10_half_step{end_step:04d}"; archive_path = output / (name + ".npz"); receipt_path = output / (name + ".json")
    if not (archive_path.is_file() and receipt_path.is_file()) or archive_path.is_symlink() or receipt_path.is_symlink():
        raise Protocol232Error("checkpoint pair differs: " + name)
    arrays = arrays_from_npz(archive_path); state = tuple(arrays[name] for name in FIELDS)
    passed, gates = endpoint_gate(context, state, end_step * HALF_DT)
    receipt = read_json(receipt_path); fingerprint = receipt.pop("fingerprint", None)
    expected = checkpoint_fingerprint(receipt); receipt["fingerprint"] = fingerprint
    manifest = {name: context["p220"].P211.array_record(arrays[name]) for name in sorted(arrays)}
    if not (
        passed and fingerprint == expected and receipt.get("schema") == CHECKPOINT_SCHEMA
        and receipt.get("authority_sha256") == authority_sha and receipt.get("start_step") == start_step
        and receipt.get("end_step") == end_step and receipt.get("dt") == HALF_DT
        and receipt.get("previous_checkpoint_sha256") == previous_sha
        and receipt.get("archive") == record(archive_path, ROOT) and receipt.get("arrays") == manifest
        and receipt.get("endpoint_gates") == gates and receipt.get("passed") is True
    ): raise Protocol232Error("checkpoint semantics differ: " + name)
    return state, sha256(receipt_path)


def publish_checkpoint(output, context, authority_sha, start_step, end_step, state, previous_sha):
    name = f"G10_half_step{end_step:04d}"; archive_path = output / (name + ".npz"); receipt_path = output / (name + ".json")
    if archive_path.exists() or receipt_path.exists(): return validate_checkpoint(output, context, authority_sha, start_step, end_step, previous_sha)
    bundle = context["bundles"]["G10"]
    validator = lambda stage: context["runner"]._technical_stage_audit(bundle, context["p228"].MODE, stage)
    print(f"G10 half-step evolution {start_step * HALF_DT:.7f} -> {end_step * HALF_DT:.7f}", flush=True)
    segment = context["p220"].P216.run_half(
        bundle, context["runner"], context["p228"].MODE, HALF_DT, end_step - start_step,
        state, start_step + 1, validator,
    )
    if not (segment["completed"] and len(segment["records"]) == 2 * (end_step - start_step) and all(item["finite"] for item in segment["records"])):
        raise Protocol232Error("half-step evolution segment failed")
    arrays = {name: np.ascontiguousarray(segment["end_state"][index]) for index, name in enumerate(FIELDS)}
    passed, gates = endpoint_gate(context, tuple(arrays[name] for name in FIELDS), end_step * HALF_DT)
    if not passed: raise Protocol232Error("half-step endpoint gate failed")
    atomic_npz(archive_path, arrays)
    manifest = {name: context["p220"].P211.array_record(arrays[name]) for name in sorted(arrays)}
    receipt = {
        "schema": CHECKPOINT_SCHEMA, "authority_sha256": authority_sha,
        "grid": "G10", "start_step": start_step, "end_step": end_step,
        "start_time": start_step * HALF_DT, "end_time": end_step * HALF_DT,
        "dt": HALF_DT, "mode": context["p228"].MODE,
        "previous_checkpoint_sha256": previous_sha, "archive": record(archive_path, ROOT),
        "arrays": manifest, "endpoint_gates": gates, "passed": True,
    }
    receipt["fingerprint"] = checkpoint_fingerprint(receipt); atomic_json(receipt_path, receipt)
    return validate_checkpoint(output, context, authority_sha, start_step, end_step, previous_sha)


def publish_anchors(output, context, inputs, authority_sha, endpoint_state):
    archive_path = output / "G10_half_anchors.npz"; receipt_path = output / "G10_half_anchors.json"
    p227 = context["p227"]; position, velocity = endpoint_state[:2]
    z, r = context["p229"].coordinates("G10"); prepared = p227.prepare_capped_expansion_slice(position, velocity, z, r)
    with np.load(inputs["p228/profiles"], allow_pickle=False) as source:
        source_profiles = {branch: {key: np.ascontiguousarray(source[f"G10_{branch}_{key}"]) for key in ("theta", "rho", "slope")} for branch in ("inner", "outer")}
    arrays = {}; public = {}
    for branch in ("inner", "outer"):
        refined = p227.solve_dynamical_capped_surface_bvp(
            position, velocity, z, r, source_profiles[branch], tolerance=2e-6,
            nodes=121, maximum_nodes=6000, dense_nodes=501, prepared=prepared,
        )
        if not p227.admitted(refined): raise Protocol232Error("half-step anchor admission failed: " + branch)
        stability = p227.mots_stability_matrix(position, velocity, z, r, refined, nodes=65, relative_step=1e-5, prepared=prepared)
        eigenvalue = float(stability["principal_eigenvalue_real"])
        if (branch == "inner" and eigenvalue >= 0) or (branch == "outer" and eigenvalue <= 0):
            raise Protocol232Error("half-step anchor sign differs: " + branch)
        for key in ("theta", "rho", "slope"): arrays[f"G10_{branch}_{key}"] = np.ascontiguousarray(refined[key])
        public[branch] = {"surface": p227.public_surface(refined), "principal_eigenvalue_65": eigenvalue}
    if archive_path.exists() or receipt_path.exists():
        if not (archive_path.is_file() and receipt_path.is_file()): raise Protocol232Error("partial anchor artifact")
        with np.load(archive_path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays) or any(not np.array_equal(existing[name], arrays[name]) for name in arrays):
                raise Protocol232Error("recovered anchors differ")
        receipt = read_json(receipt_path); fingerprint = receipt.pop("fingerprint", None)
        expected = hashlib.sha256(b"protocol232-anchors\0" + canonical(receipt)).hexdigest(); receipt["fingerprint"] = fingerprint
        if not (
            fingerprint == expected and receipt.get("schema") == ANCHOR_SCHEMA
            and receipt.get("authority_sha256") == authority_sha and receipt.get("passed") is True
            and receipt.get("archive") == record(archive_path, ROOT) and receipt.get("branches") == public
        ): raise Protocol232Error("anchor receipt differs")
        return archive_path, public
    atomic_npz(archive_path, arrays)
    receipt = {"schema": ANCHOR_SCHEMA, "authority_sha256": authority_sha, "archive": record(archive_path, ROOT), "branches": public, "passed": True}
    receipt["fingerprint"] = hashlib.sha256(b"protocol232-anchors\0" + canonical(receipt)).hexdigest(); atomic_json(receipt_path, receipt)
    return archive_path, public


def relative_difference(left, right):
    return float(abs(float(left) - float(right)) / max(abs(float(left)), abs(float(right)), 1e-300))


def compare_results(coarse, half):
    coarse_a = coarse["critical_coefficients"]["transversality_values"][1]
    half_a = half["critical_coefficients"]["transversality_values"][1]
    coarse_b = coarse["critical_coefficients"]["quadratic_values"][1]
    half_b = half["critical_coefficients"]["quadratic_values"][1]
    comparison = {
        "critical_time_absolute_difference": abs(half["critical_time_estimate"] - coarse["critical_time_estimate"]),
        "critical_time_limit": COARSE_DT / 64.0,
        "critical_area_relative_difference": relative_difference(half["critical_geometry"]["one_sided_cap_area"], coarse["critical_geometry"]["one_sided_cap_area"]),
        "transversality_relative_difference": relative_difference(half_a, coarse_a),
        "quadratic_relative_difference": relative_difference(half_b, coarse_b),
        "transversality_sign_agrees": bool(np.sign(half_a) == np.sign(coarse_a) != 0),
        "quadratic_sign_agrees": bool(np.sign(half_b) == np.sign(coarse_b) != 0),
        "coarse_log_exponent": coarse["square_root_fit"]["log_exponent"],
        "half_log_exponent": half["square_root_fit"]["log_exponent"],
    }
    comparison["transfer_pass"] = bool(
        comparison["critical_time_absolute_difference"] <= comparison["critical_time_limit"]
        and comparison["critical_area_relative_difference"] < 0.01
        and comparison["transversality_sign_agrees"] and comparison["quadratic_sign_agrees"]
        and comparison["transversality_relative_difference"] < 0.20
        and comparison["quadratic_relative_difference"] < 0.20
        and 0.40 <= comparison["coarse_log_exponent"] <= 0.60
        and 0.40 <= comparison["half_log_exponent"] <= 0.60
    )
    return comparison


def result_fingerprint(value):
    return hashlib.sha256(b"protocol232-result\0" + canonical(value)).hexdigest()


def run(root, academic_root, visual_root, project_root):
    root = Path(root).absolute(); runtime_preflight(root); _, inputs = verify(root, academic_root)
    authority_sha = sha256(root / "authority/freeze_record.json")
    p229 = load(inputs["p229/source"], "protocol229_bound232"); p229.verify(protocol_roots(academic_root)["p229"], academic_root)
    p228 = load(inputs["p228/source"], "protocol228_bound232")
    _, p228_inputs = p228.verify(protocol_roots(academic_root)["p228"], academic_root)
    p220 = load(inputs["p220/source"], "protocol220_bound232")
    with p220.exclusive_lock(root):
        output = root / "candidate-output"
        if not output.exists(): output.mkdir(); fsync_dir(root)
        final_path = output / "protocol232_result.json"
        if final_path.exists():
            value = read_json(final_path); fingerprint = value.pop("fingerprint", None); expected = result_fingerprint(value); value["fingerprint"] = fingerprint
            if fingerprint != expected or value.get("authority_sha256") != authority_sha: raise Protocol232Error("final result differs")
            return value
        context = p228.load_modules_and_context(root, academic_root, visual_root, project_root, p228_inputs)
        context.update({"p228": p228, "p220": context["p220"], "p229": p229})
        old_crossfit = context["runner"].axis_even_crossfit_audit
        context["runner"].axis_even_crossfit_audit = context["p220"].P190.node_crossfit
        try:
            state = p229.load_state("G10", inputs["G10/start"]); previous_sha = sha256(inputs["G10/start"]); current_step = START_STEP
            for end_step in CHECKPOINT_STEPS:
                state, previous_sha = publish_checkpoint(output, context, authority_sha, current_step, end_step, state, previous_sha)
                current_step = end_step
            endpoint_path = output / f"G10_half_step{END_STEP:04d}.npz"
            anchors_path, anchors = publish_anchors(output, context, inputs, authority_sha, state)
            inherited_inputs = dict(p229.input_paths(academic_root))
            inherited_inputs["p228/G10-checkpoint"] = endpoint_path
            inherited_inputs["p228/profiles"] = anchors_path
            saved = (p229.DT, p229.START_STEP, p229.END_STEP)
            p229.DT, p229.START_STEP, p229.END_STEP = HALF_DT, START_STEP, END_STEP
            try:
                half_result, arrays = p229.execute_grid(context, inherited_inputs, "G10")
            finally:
                p229.DT, p229.START_STEP, p229.END_STEP = saved
        finally:
            context["runner"].axis_even_crossfit_audit = old_crossfit
        half_result["schema"] = GRID_SCHEMA
        half_result["checks"]["half_timestep_endpoint_replay_bitwise_exact"] = half_result["checks"].pop("Protocol228_endpoint_replay_bitwise_exact")
        archive_path = output / "protocol232_G10_half.npz"; atomic_npz(archive_path, arrays)
        coarse = read_json(inputs["p229/G10-json"])["result"]; comparison = compare_results(coarse, half_result)
        if half_result["passed"] and comparison["transfer_pass"]:
            classification, status = "G10-HALF-TIMESTEP-SADDLE-NODE-CLOSURE-PASS", "PASS"
        elif half_result["passed"]:
            classification, status = "MATERIAL-TEMPORAL-DRIFT-REQUIRES-DT-QUARTER", "REVIEW"
        elif half_result["checks"].get("opposite_principal_sign_bracket_found"):
            classification, status = "HALF-TIMESTEP-SADDLE-NODE-CONDITIONS-NOT-SATISFIED", "REVIEW"
        else:
            classification, status = "HALF-TIMESTEP-BIFURCATION-INCONCLUSIVE", "REVIEW"
        value = {
            "schema": SCHEMA, "authority_sha256": authority_sha, "status": status, "classification": classification,
            "coarse_G10": {"artifact": record(inputs["p229/G10-json"]), "result": coarse},
            "half_G10": {"artifact": record(archive_path, ROOT), "anchors": anchors, "result": half_result},
            "temporal_comparison": comparison,
            "dt_quarter_authorized": False, "continuum_theorem_claim_authorized": False,
            "full_nonsymmetric_spectral_claim_authorized": False, "event_horizon_claim_authorized": False,
            "phase_selection_claim_authorized": False, "source_ownership_claim_authorized": False,
        }
        value["fingerprint"] = result_fingerprint(value); atomic_json(final_path, value); verify(root, academic_root)
        return value


def status(root):
    output = Path(root) / "candidate-output"
    completed = [step for step in CHECKPOINT_STEPS if (output / f"G10_half_step{step:04d}.json").is_file()] if output.exists() else []
    final = output / "protocol232_result.json"
    return {"schema": "protocol232-status-v1", "completed_half_steps": completed, "next_step": None if final.exists() else (CHECKPOINT_STEPS[len(completed)] if len(completed) < len(CHECKPOINT_STEPS) else "continuation"), "final_result_present": final.exists(), "classification": read_json(final).get("classification") if final.exists() else None}


def main(argv=None):
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "verify"):
        command = sub.add_parser(name); command.add_argument("--root", required=True); command.add_argument("--academic-root", required=True)
    command = sub.add_parser("run"); command.add_argument("--root", required=True); command.add_argument("--academic-root", required=True); command.add_argument("--visual-root", required=True); command.add_argument("--project-root", required=True)
    command = sub.add_parser("status"); command.add_argument("--root", required=True)
    args = vars(parser.parse_args(argv)); command = args.pop("command")
    if command == "freeze": value = freeze(**args)
    elif command == "verify": value = {"verified": True, "fingerprint": verify(**args)[0]["fingerprint"]}
    elif command == "run": value = run(**args)
    else: value = status(**args)
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
