#!/usr/bin/env python3
"""Archive-only Protocol 250 full/half causal-signature comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.interpolate import RectBivariateSpline


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bhps.dynamical_capped_horizon import regular_so3_adm_slice

from authority import file_record, sha256, verify_freeze
from causal_signature_core import (
    causal_resolution, classify, compare_norm_records, embedded_curve,
    projected_tube_norm,
)


SCHEMA = "protocol250-g10-full-half-causal-signature-result-v1"
OUTPUT = ROOT / "candidate-output"
P240 = ROOT / "sealed-inputs/protocol240"
P244 = ROOT / "sealed-inputs/protocol244"
P246 = ROOT / "sealed-inputs/protocol246"
P247 = ROOT / "sealed-inputs/protocol247"
P249 = ROOT / "sealed-inputs/protocol249"
FULL_DT = 3.125e-5
HALF_DT = 1.5625e-5
FULL_CENTERS = tuple(range(39, 48))
FULL_LEAVES = tuple(range(38, 49))
HALF_CENTERS = tuple(range(78, 96, 2))
HALF_LEAVES = tuple(range(76, 98, 2))
PATHS = ("backward", "centered", "forward")
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
EXPECTED_RUNTIME = {"python": "3.8.10", "numpy": "1.24.4", "scipy": "1.10.1", "system": "Linux", "machine": "aarch64"}


class Protocol250Error(RuntimeError):
    pass


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def read_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )


def check_fingerprint(record, prefix):
    bare = dict(record); observed = bare.pop("fingerprint", None)
    return observed == hashlib.sha256(prefix + canonical(bare)).hexdigest()


def array_record(value):
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "byte_count": int(array.nbytes),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def regular(path, immutable=False):
    path = Path(path)
    return bool(
        path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
        and (not immutable or (path.stat().st_mode & 0o222) == 0)
    )


def fsync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path, payload):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise Protocol250Error(f"output path is not fresh: {path.name}")
    with temporary.open("xb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    os.replace(temporary, path)
    fsync_directory(path.parent)


def atomic_json(path, value):
    atomic_bytes(path, canonical(value))


def atomic_npz(path, arrays):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or path.is_symlink() or temporary.exists() or temporary.is_symlink():
        raise Protocol250Error(f"output path is not fresh: {path.name}")
    with temporary.open("xb") as stream:
        np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(), 0o444)
    os.replace(temporary, path)
    fsync_directory(path.parent)


def load_npz(path):
    if not regular(path, immutable=True):
        raise Protocol250Error(f"archive is missing or unsafe: {path.name}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def runtime_preflight():
    observed = {
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "system": platform.system(), "machine": platform.machine(),
    }
    if observed != EXPECTED_RUNTIME:
        raise Protocol250Error(f"runtime differs: {observed}")
    if any(os.environ.get(name) != "1" for name in THREAD_VARS):
        raise Protocol250Error("thread controls differ")


def current_peak_rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if platform.system() == "Linux" else value


def parent_result(path, prefix, authority_path):
    result = read_json(path)
    if not check_fingerprint(result, prefix) or result.get("authority_sha256") != sha256(authority_path):
        raise Protocol250Error(f"parent result identity differs: {path}")
    return result


def validate_prerequisites():
    p240 = parent_result(P240 / "candidate-output/protocol240_result.json", b"protocol240-result-v3\0", P240 / "freeze_record.json")
    p244 = parent_result(P244 / "candidate-output/protocol244_result.json", b"protocol244-result-v1\0", P244 / "freeze_record.json")
    p246 = parent_result(P246 / "candidate-output/protocol246_result.json", b"protocol246-result-v1\0", P246 / "freeze_record.json")
    p247 = parent_result(P247 / "candidate-output/protocol247_result.json", b"protocol247-result-v1\0", P247 / "freeze_record.json")
    p249 = parent_result(P249 / "candidate-output/protocol249_result.json", b"protocol249-result-v1\0", P249 / "freeze_record.json")
    if not (
        p240.get("scientific", {}).get("classification") == "DENSE-G10-OUTER-MARGINAL-TUBE-PASS"
        and p240.get("scientific", {}).get("evaluation", {}).get("all_interior_tube_signatures_resolved_spacelike") is True
        and p244.get("scientific", {}).get("classification") == "FULL-DT-DENSE-G10-OUTER-MARGINAL-TUBE-PASS"
        and p244.get("scientific", {}).get("evaluation", {}).get("all_interior_tube_signatures_resolved_spacelike") is True
        and p246.get("classification") == "FULL-HALF-NATIVE-BALANCE-CONSISTENCY-PASS"
        and p246.get("complete_comparison_repeat_exact") is True
        and all(p246.get("scientific", {}).get("gates", {}).values())
        and p247.get("scientific", {}).get("classification") == "G9-G10-G11-BOUNDED-OUTER-TUBE-SPATIAL-TRANSFER-PASS"
        and all(p247.get("scientific", {}).get("gates", {}).values())
        and p249.get("classification") == "G9-G10-G11-FINITE-SEGMENT-INTEGRATED-BALANCE-PASS"
        and p249.get("finite_segment_integrated_balance_established") is True
        and all(p249.get("scientific", {}).get("gates", {}).values())
    ):
        raise Protocol250Error("passing prerequisite semantics differ")
    return {"p240": p240, "p244": p244, "p246": p246, "p247": p247, "p249": p249}


LANES = {
    "full": {
        "root": P244, "dt": FULL_DT, "centers": FULL_CENTERS, "leaves": FULL_LEAVES,
        "state_stem": "G10_full_step{step:04d}", "leaf_stem": "G10_outer_step{step:04d}",
        "state_schema": "protocol244-full-dt-g10-checkpoint-v1",
        "leaf_schema": "protocol244-full-dt-g10-outer-leaf-v1",
        "state_prefix": b"protocol244-checkpoint-v1\0", "leaf_prefix": b"protocol244-outer-leaf-v1\0",
    },
    "half": {
        "root": P240, "dt": HALF_DT, "centers": HALF_CENTERS, "leaves": HALF_LEAVES,
        "state_stem": "G10_dense_step{step:04d}", "leaf_stem": "G10_outer_step{step:04d}",
        "state_schema": "protocol240-dense-g10-checkpoint-v3",
        "leaf_schema": "protocol240-dense-g10-outer-leaf-v3",
        "state_prefix": b"protocol240-checkpoint-v3\0", "leaf_prefix": b"protocol240-outer-leaf-v3\0",
    },
}


def validate_archive_receipt(path, receipt):
    archive = receipt.get("archive", {})
    if not regular(path, immutable=True):
        raise Protocol250Error(f"parent archive is missing or unsafe: {path.name}")
    if archive.get("byte_count") != path.stat().st_size or archive.get("sha256") != sha256(path):
        raise Protocol250Error(f"parent archive record differs: {path.name}")


def load_state(lane, step):
    config = LANES[lane]; stem = config["state_stem"].format(step=step)
    root = config["root"] / "candidate-output"; receipt = read_json(root / f"{stem}.json")
    if not (
        check_fingerprint(receipt, config["state_prefix"]) and receipt.get("schema") == config["state_schema"]
        and receipt.get("authority_sha256") == sha256(config["root"] / "freeze_record.json")
        and receipt.get("passed") is True and receipt.get("endpoint_repeat_exact") is True
        and receipt.get("end_step") == step and receipt.get("dt") == config["dt"]
        and abs(float(receipt.get("end_time")) - step * config["dt"]) < 1e-18
        and all(receipt.get("endpoint_gates", {}).values())
    ):
        raise Protocol250Error(f"state receipt semantics differ: {lane}/{step}")
    path = root / f"{stem}.npz"; validate_archive_receipt(path, receipt); arrays = load_npz(path)
    if set(arrays) != {"q", "v", "source", "memory"}:
        raise Protocol250Error(f"state inventory differs: {lane}/{step}")
    for name, array in arrays.items():
        expected_shape = (129, 241, 9 if name in {"q", "v"} else 3)
        if array.dtype != np.float64 or array.shape != expected_shape or not np.all(np.isfinite(array)):
            raise Protocol250Error(f"state array semantics differ: {lane}/{step}/{name}")
        if array_record(array) != receipt.get("arrays", {}).get(name):
            raise Protocol250Error(f"state array record differs: {lane}/{step}/{name}")
    return {"q": arrays["q"], "v": arrays["v"]}


def load_leaf(lane, step):
    config = LANES[lane]; stem = config["leaf_stem"].format(step=step)
    root = config["root"] / "candidate-output"; receipt = read_json(root / f"{stem}.json")
    evaluation = receipt.get("evaluation", {}); surface = evaluation.get("surface", {})
    if not (
        check_fingerprint(receipt, config["leaf_prefix"]) and receipt.get("schema") == config["leaf_schema"]
        and receipt.get("authority_sha256") == sha256(config["root"] / "freeze_record.json")
        and receipt.get("passed") is True and receipt.get("solve_repeat_exact") is True
        and receipt.get("step") == step
        and abs(float(receipt.get("time_over_ell")) - step * config["dt"]) < 1e-18
        and evaluation.get("passed") is True
        and evaluation.get("negative_inward_resolution", {}).get("passed") is True
        and evaluation.get("stability", {}).get("classification") == "outward-stable"
        and evaluation.get("stability", {}).get("resolved") is True
        and evaluation.get("stability", {}).get("controls_pass") is True
        and surface.get("converged") is True and surface.get("in_domain") is True
        and float(surface.get("boundary_slope_error", math.inf)) < 2e-4
        and float(surface.get("local_expansion_interior_maximum", math.inf)) < 2e-4
        and float(surface.get("primary_evaluator_crosscheck", {}).get("two_cell_interior_maximum", math.inf)) < 0.002
    ):
        raise Protocol250Error(f"leaf receipt semantics differ: {lane}/{step}")
    path = root / f"{stem}.npz"; validate_archive_receipt(path, receipt); arrays = load_npz(path)
    if set(arrays) != {"theta", "rho", "slope"}:
        raise Protocol250Error(f"leaf inventory differs: {lane}/{step}")
    for name, array in arrays.items():
        if array.dtype != np.float64 or array.shape != (501,) or not np.all(np.isfinite(array)):
            raise Protocol250Error(f"leaf array semantics differ: {lane}/{step}/{name}")
        if array_record(array) != receipt.get("profile_arrays", {}).get(name):
            raise Protocol250Error(f"leaf array record differs: {lane}/{step}/{name}")
    if not np.all(np.diff(arrays["theta"]) > 0):
        raise Protocol250Error(f"leaf theta ordering differs: {lane}/{step}")
    return arrays, {
        "boundary_slope_error": float(surface["boundary_slope_error"]),
        "local_expansion_interior_maximum": float(surface["local_expansion_interior_maximum"]),
        "maximum_theta_minus": float(evaluation["negative_inward_resolution"]["maximum_theta_minus_any_stencil"]),
        "one_sided_cap_area": float(evaluation["geometry"]["one_sided_cap_area"]),
        "principal_eigenvalue": float(evaluation["stability"]["fine_principal_eigenvalue"]),
    }


def load_all_inputs():
    lanes = {}
    for lane, config in LANES.items():
        states = {step: load_state(lane, step) for step in config["centers"]}
        leaves = {}; admission = {}
        for step in config["leaves"]:
            leaves[step], admission[step] = load_leaf(lane, step)
        lanes[lane] = {"states": states, "leaves": leaves, "admission": admission}
    return lanes


def sample_mid_metric(state, theta, rho, slope):
    z = np.linspace(1.0, math.e, state["q"].shape[0])
    r = np.linspace(0.0, 10.0, state["q"].shape[1])
    adm = regular_so3_adm_slice(state["q"], state["v"], z, r)
    coordinates, tangent = embedded_curve(theta, rho, slope)
    zcoord = z[-1] + coordinates[:, 0]; radius = coordinates[:, 1]

    def sample(field):
        field = np.asarray(field)
        if field.ndim == 2:
            return RectBivariateSpline(z, r, field, kx=3, ky=3, s=0).ev(zcoord, radius)
        result = np.empty((theta.size, *field.shape[2:]), dtype=np.float64)
        for index in np.ndindex(field.shape[2:]):
            spline = RectBivariateSpline(z, r, field[(slice(None), slice(None), *index)], kx=3, ky=3, s=0)
            result[(slice(None), *index)] = spline.ev(zcoord, radius)
        return result

    return {
        "tangent": tangent, "lapse": sample(adm["lapse"]),
        "shift_covector": sample(adm["shift_covector"]), "shift": sample(adm["shift"]),
        "metric": sample(adm["base_metric"]),
    }


def evaluate_lane(lane, inputs):
    config = LANES[lane]; metadata = {}; output_arrays = {}; norm_map = {}; theta_map = {}
    centers = config["centers"]; leaf_stride = 1 if lane == "full" else 2
    for step in centers:
        early = step - leaf_stride; late = step + leaf_stride
        profiles = inputs["leaves"]; theta = profiles[step]["theta"]
        if not np.array_equal(theta, profiles[early]["theta"]) or not np.array_equal(theta, profiles[late]["theta"]):
            raise Protocol250Error(f"theta grid differs around {lane}/{step}")
        early_coordinates, _ = embedded_curve(theta, profiles[early]["rho"], profiles[early]["slope"])
        middle_coordinates, _ = embedded_curve(theta, profiles[step]["rho"], profiles[step]["slope"])
        late_coordinates, _ = embedded_curve(theta, profiles[late]["rho"], profiles[late]["slope"])
        spacing = leaf_stride * config["dt"]
        velocities = {
            "backward": (middle_coordinates - early_coordinates) / spacing,
            "centered": (late_coordinates - early_coordinates) / (2.0 * spacing),
            "forward": (late_coordinates - middle_coordinates) / spacing,
        }
        metric = sample_mid_metric(inputs["states"][step], theta, profiles[step]["rho"], profiles[step]["slope"])
        norms = {
            name: projected_tube_norm(
                metric["lapse"], metric["shift_covector"], metric["shift"], metric["metric"],
                velocity, metric["tangent"],
            )
            for name, velocity in velocities.items()
        }
        summary, resolved, spread = causal_resolution(norms["backward"], norms["centered"], norms["forward"])
        summary.update({
            "step": step, "time_over_ell": step * config["dt"], "physical_leaf_spacing": spacing,
            "minimum_lapse": float(np.min(metric["lapse"])),
        })
        metadata[str(step)] = summary; norm_map[step] = norms; theta_map[step] = theta
        for name, value in norms.items():
            output_arrays[f"{lane}_step{step:04d}_norm_{name}"] = value
        output_arrays[f"{lane}_step{step:04d}_resolved"] = resolved.astype(np.uint8)
        output_arrays[f"{lane}_step{step:04d}_one_sided_spread"] = spread
    lane_pass = bool(all(item["label"] == "UNIFORMLY-SPACELIKE" and item["resolved_fraction"] == 1.0 for item in metadata.values()))
    admission = {
        "maximum_boundary_slope_error": float(max(item["boundary_slope_error"] for item in inputs["admission"].values())),
        "maximum_local_expansion_interior": float(max(item["local_expansion_interior_maximum"] for item in inputs["admission"].values())),
        "maximum_theta_minus": float(max(item["maximum_theta_minus"] for item in inputs["admission"].values())),
        "area_strictly_increasing": bool(np.all(np.diff([inputs["admission"][step]["one_sided_cap_area"] for step in config["leaves"]]) > 0)),
        "principal_eigenvalue_strictly_positive": bool(all(item["principal_eigenvalue"] > 0 for item in inputs["admission"].values())),
    }
    admission["passed"] = bool(
        admission["maximum_boundary_slope_error"] < 2e-4
        and admission["maximum_local_expansion_interior"] < 2e-4
        and admission["maximum_theta_minus"] < 0
        and admission["area_strictly_increasing"] and admission["principal_eigenvalue_strictly_positive"]
    )
    return {"records": metadata, "admission": admission, "passed": lane_pass}, output_arrays, norm_map, theta_map


def build_science(inputs):
    full_meta, full_arrays, full_norms, full_theta = evaluate_lane("full", inputs["full"])
    half_meta, half_arrays, half_norms, half_theta = evaluate_lane("half", inputs["half"])
    comparisons = {}; temporal_pass = True
    for full_step, half_step in zip(FULL_CENTERS, HALF_CENTERS):
        if not np.array_equal(full_theta[full_step], half_theta[half_step]):
            raise Protocol250Error(f"full/half theta grid differs at {full_step}")
        comparison = compare_norm_records(full_norms[full_step], half_norms[half_step], 0.01)
        comparison.update({
            "full_step": full_step, "half_step": half_step,
            "time_over_ell": full_step * FULL_DT,
            "time_alignment_exact": bool(full_step * FULL_DT == half_step * HALF_DT),
        })
        temporal_pass = bool(temporal_pass and comparison["passed"] and comparison["time_alignment_exact"])
        comparisons[str(full_step)] = comparison
    inherited_spatial = True
    inherited_balance = True
    classification = classify(full_meta["passed"], half_meta["passed"], temporal_pass, inherited_spatial, inherited_balance)
    gates = {
        "parent_admission": True,
        "matched_time_and_parameter_grid_alignment": bool(all(item["time_alignment_exact"] for item in comparisons.values())),
        "free_boundary_marginal_leaf_admission": bool(full_meta["admission"]["passed"] and half_meta["admission"]["passed"]),
        "full_timestep_resolved_spacelike": full_meta["passed"],
        "half_timestep_resolved_spacelike": half_meta["passed"],
        "full_half_pointwise_consistency": temporal_pass,
        "three_grid_spatial_transfer_admission": inherited_spatial,
        "finite_segment_integrated_balance_admission": inherited_balance,
    }
    return {
        "classification": classification,
        "full_steps": list(FULL_CENTERS), "half_steps": list(HALF_CENTERS),
        "physical_times": [step * FULL_DT for step in FULL_CENTERS],
        "projection": "g(V,V)-g(V,S)^2/g(S,S)", "signature_convention": "(-,+,+,+,+)",
        "relative_limit": 0.01, "relative_floor": 1e-12,
        "lanes": {"full": full_meta, "half": half_meta},
        "comparisons": comparisons,
        "maximum_full_half_relative_difference": float(max(item["maximum_relative_difference"] for item in comparisons.values())),
        "gates": gates,
    }, {**full_arrays, **half_arrays}


def exact_repeat(first_science, first_arrays, second_science, second_arrays):
    return bool(
        canonical(first_science) == canonical(second_science)
        and set(first_arrays) == set(second_arrays)
        and all(np.array_equal(first_arrays[name], second_arrays[name]) for name in first_arrays)
    )


def validate_output_archive(path, expected):
    observed = load_npz(path)
    if set(observed) != set(expected) or any(not np.array_equal(observed[name], expected[name]) for name in expected):
        raise Protocol250Error("recovered causal-signature archive differs")


def execute():
    started = time.monotonic(); runtime_preflight(); verify_freeze(ROOT); validate_prerequisites(); inputs = load_all_inputs()
    first_science, first_arrays = build_science(inputs)
    second_science, second_arrays = build_science(inputs)
    if not exact_repeat(first_science, first_arrays, second_science, second_arrays):
        raise Protocol250Error("complete causal-signature repeat differs")
    allowed = {"protocol250_causal_signature_arrays.npz", "protocol250_result.json"}
    if OUTPUT.exists():
        if OUTPUT.is_symlink() or not OUTPUT.is_dir() or not {path.name for path in OUTPUT.iterdir()} <= allowed:
            raise Protocol250Error("candidate-output inventory differs")
    else:
        OUTPUT.mkdir(mode=0o755); fsync_directory(ROOT)
    archive_path = OUTPUT / "protocol250_causal_signature_arrays.npz"
    if archive_path.exists():
        validate_output_archive(archive_path, first_arrays)
    else:
        atomic_npz(archive_path, first_arrays)
    authority_sha = sha256(ROOT / "authority/freeze_record.json")
    pass_class = first_science["classification"] == "G10-FULL-HALF-CAUSAL-SIGNATURE-CONSISTENCY-PASS"
    bare = {
        "schema": SCHEMA,
        "authority_sha256": authority_sha,
        "classification": first_science["classification"],
        "scientific": first_science,
        "complete_metric_signature_and_comparison_repeat_exact": True,
        "archive": file_record(archive_path, ROOT),
        "array_records": {name: array_record(value) for name, value in sorted(first_arrays.items())},
        "runtime": {
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "elapsed_wall_seconds": float(time.monotonic() - started),
            "peak_rss_bytes": current_peak_rss_bytes(),
        },
        "finite_resolution_free_boundary_dynamical_horizon_evidence_established": pass_class,
        "spacetime_evolution_executed": False, "surface_solve_executed": False,
        "parent_or_published_artifact_modified": False, "submitted_paper_edited": False,
        "continuum_dynamical_horizon_claim_authorized": False, "event_horizon_claim_authorized": False,
        "connected_topology_claim_authorized": False, "global_intersector_charge_claim_authorized": False,
        "source_ownership_claim_authorized": False,
    }
    result = dict(bare)
    result["fingerprint"] = hashlib.sha256(b"protocol250-result-v1\0" + canonical(bare)).hexdigest()
    result_path = OUTPUT / "protocol250_result.json"
    if result_path.exists():
        recovered = read_json(result_path)
        if not (
            check_fingerprint(recovered, b"protocol250-result-v1\0")
            and recovered.get("schema") == SCHEMA and recovered.get("authority_sha256") == authority_sha
            and recovered.get("classification") == first_science["classification"]
            and recovered.get("scientific") == first_science
            and recovered.get("complete_metric_signature_and_comparison_repeat_exact") is True
            and recovered.get("archive") == file_record(archive_path, ROOT)
            and recovered.get("array_records") == {name: array_record(value) for name, value in sorted(first_arrays.items())}
            and recovered.get("finite_resolution_free_boundary_dynamical_horizon_evidence_established") == pass_class
        ):
            raise Protocol250Error("recovered result differs")
        result = recovered
    else:
        atomic_json(result_path, result)
    verify_freeze(ROOT)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "verify", "status"))
    args = parser.parse_args(argv)
    if args.command == "run":
        result = execute()
    elif args.command == "verify":
        runtime_preflight(); verify_freeze(ROOT); validate_prerequisites(); load_all_inputs()
        result = {"status": "VERIFIED", "authority_sha256": sha256(ROOT / "authority/freeze_record.json")}
    else:
        path = OUTPUT / "protocol250_result.json"
        value = read_json(path) if path.is_file() else None
        result = {"status": "COMPLETE" if value else "NOT-STARTED", "classification": None if value is None else value["classification"]}
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
