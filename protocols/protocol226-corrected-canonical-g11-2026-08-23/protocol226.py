"""Corrected recoverable canonical G11 same-time spatial closure."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SCHEMA = "protocol226-corrected-canonical-g11-v1"
STAGE_SCHEMA = "protocol226-g11-trajectory-receipt-v1"
MODE = "legacy_wall_axis_outer"
DT = 0.00003125
STEPS = 32
END_TIME = 0.001
FIELDS = ("q", "v", "source", "memory")
STAGE_PLAN = ("G11_repeat1", "G11_repeat2")
RELATIVE_TENSOR_CEILING = 0.002


class Protocol226Error(RuntimeError):
    pass


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def roots(academic_root):
    protocols = Path(academic_root).absolute() / "Discussion/protocols"
    return {
        "p216": protocols / "protocol216-doubled-horizon-canonical-2026-08-22",
        "p220": protocols / "protocol220-recoverable-canonical-spatial-2026-08-22",
        "p224": protocols / "protocol224-g11-complete-parent-qualification-2026-08-23",
        "p225": protocols / "protocol225-recoverable-canonical-g11-2026-08-23",
    }


def paths(academic_root):
    found = roots(academic_root)
    return {
        "p216/result": found["p216"] / "linux-output/protocol216_result.json",
        "p216/archive": found["p216"] / "linux-output/g10_doubled_horizon_canonical.npz",
        "p220/source": found["p220"] / "protocol220.py",
        "p220/authority": found["p220"] / "authority/freeze_record.json",
        "p220/result": found["p220"] / "candidate-output/protocol220_result.json",
        "p224/authority": found["p224"] / "authority/freeze_record.json",
        "p224/result": found["p224"] / "linux-output/protocol224_result.json",
        "p224/archive": found["p224"] / "linux-output/g11_complete_parent_and_zero_step.npz",
        "p225/protocol": found["p225"] / "PROTOCOL.md",
        "p225/source": found["p225"] / "protocol225.py",
        "p225/authority": found["p225"] / "authority/freeze_record.json",
    }


def freeze(root, academic_root):
    root = Path(root).absolute()
    inputs = paths(academic_root)
    if any((root / "authority").iterdir()) or (root / "candidate-output").exists():
        raise Protocol226Error("prospective namespace differs")
    for name, path in inputs.items():
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
            raise Protocol226Error("unsafe input: " + name)
    p216 = read_json(inputs["p216/result"])
    p220 = read_json(inputs["p220/result"])
    p224 = read_json(inputs["p224/result"])
    if (p216.get("classification") != "DOUBLED-HORIZON-CANONICAL-PASS"
            or p216.get("dt") != DT or p216.get("steps") != STEPS or p216.get("endpoint_time") != END_TIME):
        raise Protocol226Error("Protocol216 prerequisite differs")
    if p220.get("classification") != "CANONICAL-SAME-TIME-SPATIAL-PASS" or not all(p220.get("checks", {}).values()):
        raise Protocol226Error("Protocol220 prerequisite differs")
    if (p224.get("classification") != "G11-COMPLETE-PARENT-QUALIFICATION-PASS"
            or p224.get("g11_canonical_trajectory_authorized") is not True or not all(p224.get("checks", {}).values())):
        raise Protocol226Error("Protocol224 prerequisite differs")
    authority = {
        "schema": SCHEMA, "status": "FROZEN", "mode": MODE,
        "grids": ["G9", "G10", "G11"], "new_grid_shape": [145, 271],
        "stage_plan": list(STAGE_PLAN), "dt": DT, "steps": STEPS, "endpoint_time": END_TIME,
        "field_gate": "G10-G11 below sealed G9-G10 in max and RMS, or both below 64-epsilon floor",
        "physical_tensor_gate": "G10-G11 absolute below G9-G10 and relative below 0.002",
        "auxiliary_plateau_is_distinct_from_physical_failure": True,
        "correction": "Protocol224 archived its zero-step acceleration under the second-mode wall-owner policy; Protocol226 computes and repeats the canonical zero-step value directly",
        "sources": {name: record(root / name, root) for name in ("PROTOCOL.md", "protocol226.py", "tests/test_protocol226.py")},
        "inputs": {name: record(path) for name, path in inputs.items()},
        "candidate_output_absent_at_freeze": True,
        "continuum_order_claim_authorized": False, "phase_a_authorized": False,
        "production_evolution_authorized": False, "evolved_physics_claim_authorized": False,
    }
    authority["fingerprint"] = hashlib.sha256(b"protocol226-freeze\0" + canonical(authority)).hexdigest()
    atomic_json(root / "authority/freeze_record.json", authority)
    return authority


def verify(root, academic_root):
    root = Path(root).absolute()
    authority = read_json(root / "authority/freeze_record.json")
    fingerprint = authority.pop("fingerprint", None)
    expected = hashlib.sha256(b"protocol226-freeze\0" + canonical(authority)).hexdigest()
    authority["fingerprint"] = fingerprint
    if authority.get("schema") != SCHEMA or fingerprint != expected:
        raise Protocol226Error("authority differs")
    for name, item in authority["sources"].items():
        if not same(root / item["path"], item):
            raise Protocol226Error("source differs: " + name)
    inputs = paths(academic_root)
    if set(inputs) != set(authority["inputs"]):
        raise Protocol226Error("input inventory differs")
    for name, item in authority["inputs"].items():
        if not same(inputs[name], item):
            raise Protocol226Error("input differs: " + name)
    return authority, inputs


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path, root=None):
    path = Path(path).absolute()
    return {"path": path.relative_to(Path(root).absolute()).as_posix() if root else str(path),
            "byte_count": path.stat().st_size, "sha256": sha256(path)}


def same(path, item):
    path = Path(path)
    return bool(set(item) == {"path", "byte_count", "sha256"} and path.is_file() and not path.is_symlink()
                and path.stat().st_nlink == 1 and type(item["byte_count"]) is int
                and path.stat().st_size == item["byte_count"] and type(item["sha256"]) is str
                and sha256(path) == item["sha256"])


def fsync_dir(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def atomic_json(path, value):
    path = Path(path); temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    payload = canonical(value)
    if path.exists() or path.is_symlink(): raise Protocol226Error("output exists")
    with temporary.open("xb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    os.replace(temporary, path); fsync_dir(path.parent)


def atomic_npz(path, arrays):
    path = Path(path); temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    if path.exists() or path.is_symlink(): raise Protocol226Error("archive exists")
    with temporary.open("xb") as stream:
        np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    os.replace(temporary, path); fsync_dir(path.parent)


def array_manifest(p220, arrays):
    return {name: p220.P211.array_record(value) for name, value in sorted(arrays.items())}


def receipt_fingerprint(value):
    return hashlib.sha256(b"protocol226-stage\0" + canonical(value)).hexdigest()


def result_fingerprint(value):
    return hashlib.sha256(b"protocol226-result\0" + canonical(value)).hexdigest()


def load_context(root, academic_root, visual_root, project_root, inputs):
    p220 = load(inputs["p220/source"], "protocol220_bound226")
    _, p220_inputs = p220.verify(roots(academic_root)["p220"], academic_root, project_root)
    context = p220._load_context(root, academic_root, visual_root, project_root, p220_inputs)
    with np.load(inputs["p224/archive"], allow_pickle=False) as archive:
        g11 = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    prefix = "p219-current/p218-current/p217-current/p216-current/p215-current/p214-current/p213-current/p212-current/p211-current/p210-current/p209-current/p208-current/p190-input/inherited/"
    inherited = {name[len(prefix):]: path for name, path in p220_inputs.items() if name.startswith(prefix)}
    matched = load(inherited["project/matched"], "matched_bound226")
    background = context["bundles"]["G10"].geometry["background"]
    bundle = p220.P212.make_bundle("G11", g11, background, context["runner"], matched)
    context.update({"p220": p220, "g11": g11, "g11_bundle": bundle})
    return context


def zero_step_policy_checks(canonical_one, canonical_two, first_stage, second_mode, archived,
                            canonical_finite, canonical_native_gate):
    return {
        "canonical_zero_step_repeat_exact": bool(np.array_equal(canonical_one, canonical_two)),
        "canonical_zero_step_native_gates_pass": bool(canonical_finite and canonical_native_gate),
        "first_stage_matches_canonical_zero_step": bool(np.array_equal(first_stage, canonical_one)),
        "protocol224_zero_step_matches_second_mode": bool(np.array_equal(second_mode, archived)),
        "protocol224_zero_step_differs_from_canonical": bool(not np.array_equal(canonical_one, archived)),
    }


def zero_step_policy_evidence(context, initial, first_stage=None):
    p220 = context["p220"]; runner = context["runner"]; bundle = context["g11_bundle"]
    canonical_one = runner.reconstruct_driver_stage(bundle, MODE, 0.0, initial, capture=True)
    canonical_two = runner.reconstruct_driver_stage(bundle, MODE, 0.0, initial, capture=True)
    canonical_audit = runner._technical_stage_audit(bundle, MODE, canonical_one)
    second_mode = runner.reconstruct_driver_stage(
        bundle, "wall_owner_last_experimental", 0.0, initial, capture=True
    )
    archived = context["g11"]["G11_zero_step_acceleration"]
    first_acceleration = canonical_one["acceleration"] if first_stage is None else first_stage["acceleration"]
    checks = zero_step_policy_checks(
        canonical_one["acceleration"], canonical_two["acceleration"], first_acceleration,
        second_mode["acceleration"], archived, canonical_one["finite"],
        canonical_audit["gates"]["all_technical_gates_pass"],
    )
    delta = np.asarray(canonical_one["acceleration"] - archived, dtype=np.float64)
    evidence = {
        "archived_protocol224_mode": "wall_owner_last_experimental",
        "required_protocol226_mode": MODE,
        "differing_value_count": int(np.count_nonzero(delta)),
        "maximum_absolute_difference": float(np.max(np.abs(delta))),
        "RMS_difference": float(math.sqrt(math.fsum(float(value) * float(value) for value in delta.ravel(order="C")) / delta.size)),
    }
    return checks, evidence


def execute_trajectory(context, repeat, authority_sha):
    p220 = context["p220"]; runner = context["runner"]; bundle = context["g11_bundle"]
    initial = bundle.initial_state()
    validator = lambda stage: runner._technical_stage_audit(bundle, MODE, stage)
    old = runner.axis_even_crossfit_audit; runner.axis_even_crossfit_audit = p220.P190.node_crossfit
    try:
        segment = p220.P216.run_half(bundle, runner, MODE, DT, STEPS, initial, 1, validator)
        zero_checks, zero_evidence = zero_step_policy_evidence(context, initial, segment["records"][0])
        endpoint = runner.reconstruct_driver_stage(bundle, MODE, END_TIME, segment["end_state"], capture=True)
        endpoint_audit = runner._technical_stage_audit(bundle, MODE, endpoint)
    finally:
        runner.axis_even_crossfit_audit = old
    arrays = {name: np.ascontiguousarray(segment["end_state"][index]) for index, name in enumerate(FIELDS)}
    arrays["acceleration"] = np.ascontiguousarray(endpoint["acceleration"])
    checks = {
        "trajectory_complete": bool(segment["completed"]), "stage_count_exact": len(segment["records"]) == 2 * STEPS,
        "all_stages_finite": bool(all(item["finite"] for item in segment["records"])),
        "all_native_stage_gates_pass": bool(all(item["technical_audit"]["gates"]["all_technical_gates_pass"] for item in segment["records"])),
        **zero_checks,
        "endpoint_finite": bool(endpoint["finite"]), "endpoint_native_gates_pass": bool(endpoint_audit["gates"]["all_technical_gates_pass"]),
    }
    if not all(checks.values()): raise Protocol226Error("G11 trajectory gate failed: " + repr(checks))
    summaries = [p220.P211.stage_summary(stage, initial) for stage in segment["records"]]
    receipt = {
        "schema": STAGE_SCHEMA, "authority_sha256": authority_sha, "stage_name": "G11_repeat" + str(repeat),
        "repeat": repeat, "mode": MODE, "dt": DT, "steps": STEPS, "endpoint_time": END_TIME,
        "checks": checks, "stage_trace_sha256": hashlib.sha256(b"protocol226-trace\0" + canonical(summaries)).hexdigest(),
        "zero_step_correction_evidence": zero_evidence,
        "arrays": array_manifest(p220, arrays), "endpoint_gates": p220.P190.jsonable(endpoint_audit["gates"]), "passed": True,
    }
    return arrays, receipt


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {*FIELDS, "acceleration"}: raise Protocol226Error("checkpoint inventory differs")
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def validate_stage(output, name, context, authority_sha):
    receipt = read_json(output / (name + ".json")); fingerprint = receipt.pop("fingerprint", None)
    expected = receipt_fingerprint(receipt); receipt["fingerprint"] = fingerprint
    if fingerprint != expected or receipt.get("schema") != STAGE_SCHEMA or receipt.get("authority_sha256") != authority_sha:
        raise Protocol226Error("checkpoint receipt differs")
    archive_path = output / (name + ".npz")
    if not same(archive_path, receipt.get("archive", {})): raise Protocol226Error("checkpoint archive differs")
    arrays = load_npz(archive_path)
    if array_manifest(context["p220"], arrays) != receipt.get("arrays"): raise Protocol226Error("checkpoint arrays differ")
    bundle = context["g11_bundle"]; runner = context["runner"]
    old = runner.axis_even_crossfit_audit; runner.axis_even_crossfit_audit = context["p220"].P190.node_crossfit
    try:
        endpoint = runner.reconstruct_driver_stage(bundle, MODE, END_TIME, tuple(arrays[field] for field in FIELDS), capture=True)
        audit = runner._technical_stage_audit(bundle, MODE, endpoint)
    finally: runner.axis_even_crossfit_audit = old
    if (not endpoint["finite"] or not audit["gates"]["all_technical_gates_pass"]
            or not np.array_equal(endpoint["acceleration"], arrays["acceleration"])):
        raise Protocol226Error("checkpoint endpoint replay differs")
    return arrays, receipt


def publish_stage(output, name, context, authority_sha):
    archive_path = output / (name + ".npz"); receipt_path = output / (name + ".json")
    if receipt_path.exists(): return validate_stage(output, name, context, authority_sha)
    repeat = int(name[-1]); arrays, receipt = execute_trajectory(context, repeat, authority_sha)
    if archive_path.exists():
        partial = load_npz(archive_path)
        if any(not np.array_equal(partial[key], arrays[key]) for key in arrays): raise Protocol226Error("partial archive replay differs")
    else: atomic_npz(archive_path, arrays)
    receipt["archive"] = record(archive_path, ROOT); receipt["fingerprint"] = receipt_fingerprint(receipt)
    atomic_json(receipt_path, receipt)
    return validate_stage(output, name, context, authority_sha)


def common_difference(p220, left, right, left_initial, right_initial, left_z, right_z, left_r, right_r):
    return p220.P212.common_increment_difference(left, right, left_initial, right_initial, left_z, right_z, left_r, right_r)


def field_decision(prior, current):
    floor = 64.0 * np.finfo(np.float64).eps * max(prior["increment_scale"], current["increment_scale"])
    norms = {}
    for name in ("maximum_absolute", "RMS"):
        at_floor = bool(prior[name] <= floor and current[name] <= floor)
        norms[name] = {"G9_G10": prior[name], "G10_G11": current[name], "strictly_decreases": current[name] < prior[name],
                       "both_at_roundoff_floor": at_floor, "roundoff_floor": float(floor),
                       "passed": bool(current[name] < prior[name] or at_floor)}
    return {"norms": norms, "common_lattice_exact_within_2e_15": bool(current["common_lattice_shape"] == [17, 31]
            and current["coordinate_maximum_absolute_mismatch"] <= 2e-15), "pair_detail": current,
            "passed": bool(all(item["passed"] for item in norms.values()))}


def finalize(root, context, checkpoints, authority_sha, inputs):
    p220 = context["p220"]; prior = read_json(inputs["p220/result"])
    endpoint10 = {field: context["g10"]["endpoint_" + field] for field in FIELDS}
    endpoint11 = checkpoints["G11_repeat1"][0]
    initial10 = context["bundles"]["G10"].initial_state(); initial11 = context["g11_bundle"].initial_state()
    fields = {}; field_pass = True
    for index, field in enumerate(FIELDS):
        current = common_difference(p220, endpoint10[field], endpoint11[field], initial10[index], initial11[index],
                                    context["arrays"]["G10_z"], context["g11"]["G11_z"],
                                    context["arrays"]["G10_r"], context["g11"]["G11_r"])
        decision = field_decision(prior["field_increment_sequences"][field]["pair_details"]["G9_G10"], current)
        fields[field] = decision; field_pass = field_pass and decision["passed"] and decision["common_lattice_exact_within_2e_15"]
    target_z = context["arrays"]["G10_z"][::8]; target_r = context["arrays"]["G10_r"][::8]
    tensors10 = context["convergence"].tensor_fields_on_grid(endpoint10["q"], endpoint10["v"], context["bundles"]["G10"].geometry, target_z, target_r)
    tensors11 = context["convergence"].tensor_fields_on_grid(endpoint11["q"], endpoint11["v"], context["g11_bundle"].geometry, target_z, target_r)
    physical = {}
    for family in ("metric_increment", "ADM_K"):
        current = context["convergence"].physical_tensor_difference(tensors10[family], tensors11[family], tensors10["final_metric"], tensors11["final_metric"], target_z, target_r)
        previous = prior["physical_tensor_sequences"][family]["G9_G10"]
        passed = bool(current["absolute_difference"] < previous["absolute_difference"] and current["relative_difference"] < RELATIVE_TENSOR_CEILING)
        physical[family] = {"G9_G10": previous, "G10_G11": p220.P190.jsonable(current),
                            "absolute_strictly_decreases": current["absolute_difference"] < previous["absolute_difference"],
                            "fine_relative_below_0_002": current["relative_difference"] < RELATIVE_TENSOR_CEILING, "passed": passed}
    physical_pass = bool(all(item["passed"] for item in physical.values()))
    repeat_exact = bool(checkpoints["G11_repeat1"][1]["stage_trace_sha256"] == checkpoints["G11_repeat2"][1]["stage_trace_sha256"]
                        and all(np.array_equal(checkpoints["G11_repeat1"][0][key], checkpoints["G11_repeat2"][0][key]) for key in checkpoints["G11_repeat1"][0]))
    if physical_pass and field_pass: classification = "CANONICAL-G11-SPATIAL-CLOSURE-PASS"
    elif physical_pass: classification = "CANONICAL-G11-PHYSICAL-PASS-AUXILIARY-RMS-PLATEAU"
    else: classification = "CANONICAL-G11-PHYSICAL-SPATIAL-FAIL"
    checks = {"protocol224_parent_bound": True, "both_G11_trajectories_pass": True, "G11_repeats_bitwise_exact": repeat_exact,
              "all_field_sequences_pass": bool(field_pass), "all_physical_tensor_sequences_pass": physical_pass}
    result = {"schema": SCHEMA, "classification": classification, "authority_sha256": authority_sha,
              "mode": MODE, "dt": DT, "steps": STEPS, "endpoint_time": END_TIME, "checks": checks,
              "field_increment_sequences": p220.P190.jsonable(fields), "physical_tensor_sequences": p220.P190.jsonable(physical),
              "trajectory_artifacts": {name: record(root / "candidate-output" / (name + ".json"), root) for name in STAGE_PLAN},
              "protocol225_technical_failure_corrected": True,
              "paper_facing_spatial_closure_supported": physical_pass,
              "auxiliary_RMS_plateau": bool(physical_pass and not field_pass), "further_spatial_test_authorized": False,
              "continuum_order_claim_authorized": False, "phase_a_authorized": False,
              "production_evolution_authorized": False, "evolved_physics_claim_authorized": False}
    result["fingerprint"] = result_fingerprint(result); atomic_json(root / "candidate-output/protocol226_result.json", result)
    return result


def validate_inventory(output):
    if not output.exists(): return
    allowed = {name + suffix for name in STAGE_PLAN for suffix in (".npz", ".json")} | {"protocol226_result.json"}
    names = {path.name for path in output.iterdir()}
    if not names <= allowed: raise Protocol226Error("unexpected output")
    gap = False
    for name in STAGE_PLAN:
        archive = (output / (name + ".npz")).exists(); receipt = (output / (name + ".json")).exists()
        if receipt and not archive: raise Protocol226Error("receipt without archive")
        if gap and (archive or receipt): raise Protocol226Error("non-prefix output")
        if not (archive and receipt): gap = True
    if (output / "protocol226_result.json").exists() and gap: raise Protocol226Error("final precedes checkpoints")


def run(root, academic_root, visual_root, project_root):
    root = Path(root).absolute(); authority, inputs = verify(root, academic_root); authority_sha = sha256(root / "authority/freeze_record.json")
    p220 = load(inputs["p220/source"], "protocol220_lock_bound226")
    with p220.exclusive_lock(root):
        output = root / "candidate-output"
        if not output.exists(): output.mkdir(); fsync_dir(root)
        validate_inventory(output); context = load_context(root, academic_root, visual_root, project_root, inputs)
        checkpoints = {name: publish_stage(output, name, context, authority_sha) for name in STAGE_PLAN}
        final = output / "protocol226_result.json"
        result = read_json(final) if final.exists() else finalize(root, context, checkpoints, authority_sha, inputs)
        verify(root, academic_root); validate_inventory(output); return result


def diagnose_zero_step(root, academic_root, visual_root, project_root):
    root = Path(root).absolute(); authority, inputs = verify(root, academic_root)
    context = load_context(root, academic_root, visual_root, project_root, inputs)
    runner = context["runner"]; initial = context["g11_bundle"].initial_state()
    old = runner.axis_even_crossfit_audit; runner.axis_even_crossfit_audit = context["p220"].P190.node_crossfit
    try:
        checks, evidence = zero_step_policy_evidence(context, initial)
    finally:
        runner.axis_even_crossfit_audit = old
    return {"schema": "protocol226-zero-step-diagnostic-v1", "authority_fingerprint": authority["fingerprint"],
            "checks": checks, "evidence": evidence, "passed": bool(all(checks.values()))}


def status(root):
    output = Path(root) / "candidate-output"; completed = [name for name in STAGE_PLAN if (output / (name + ".json")).is_file()]
    final = output / "protocol226_result.json"; result = read_json(final) if final.is_file() else None
    return {"schema": "protocol226-status-v1", "completed_stage_count": len(completed), "total_stage_count": 2,
            "completed_stages": completed, "active_or_next_stage": None if result else STAGE_PLAN[len(completed)] if len(completed) < 2 else "finalization",
            "final_result_present": result is not None, "classification": None if result is None else result.get("classification")}


def main(argv=None):
    parser = argparse.ArgumentParser(); commands = parser.add_subparsers(dest="command", required=True)
    for name in ("freeze", "verify"):
        command = commands.add_parser(name); command.add_argument("--root", required=True); command.add_argument("--academic-root", required=True)
    command = commands.add_parser("run"); command.add_argument("--root", required=True); command.add_argument("--academic-root", required=True)
    command.add_argument("--visual-root", required=True); command.add_argument("--project-root", required=True)
    command = commands.add_parser("diagnose-zero-step"); command.add_argument("--root", required=True); command.add_argument("--academic-root", required=True)
    command.add_argument("--visual-root", required=True); command.add_argument("--project-root", required=True)
    command = commands.add_parser("status"); command.add_argument("--root", required=True)
    values = vars(parser.parse_args(argv)); selected = values.pop("command")
    if selected == "freeze": result = freeze(**values)
    elif selected == "verify": authority, _ = verify(**values); result = {"verified": True, "fingerprint": authority["fingerprint"]}
    elif selected == "run": result = run(**values)
    elif selected == "diagnose-zero-step": result = diagnose_zero_step(**values)
    else: result = status(**values)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__": main()
