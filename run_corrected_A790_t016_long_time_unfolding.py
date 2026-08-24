#!/usr/bin/env python3
"""Prospective, restartable A=7.90 branch unfolding through t/ell=0.016."""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import run_corrected_A790_dynamic_MOTS_stability as stability84
import run_corrected_A790_independent_dynamic_BVP_detector as detector80
import run_corrected_A790_t008_long_evolution as long85
import run_corrected_A790_test10_joint_convergence as test10
import run_corrected_fold_live_nonlinear_gauge_source as live
from bhps.corrected_A790_R10_builder import build_A790_R10_pair
from bhps.recovery_indexer import (
    RecoveryIndex,
    atomic_write_json,
    atomic_write_npz,
    sha256_file,
    validate_npz,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry


PROTOCOL = Path("notes/134_A790_t016_long_time_unfolding_protocol.md")
FREEZE = Path("notes/134_A790_t016_long_time_unfolding_freeze.json")
OUTPUT = Path("results/corrected_A790_t016_long_time_unfolding.json")
RECOVERY = Path("results/corrected_A790_t016_long_time_unfolding_recovery")
MANIFEST = RECOVERY / "index.json"

R8_ARCHIVE = Path("results/corrected_A790_t008_long_evolution_state.npz")
R8_RESULT = Path("results/corrected_A790_t008_long_evolution.json")
R10_STATE = Path("results/corrected_A790_test10_joint_convergence_state.npz")
R10_RECOVERY = Path("results/corrected_A790_test10_joint_convergence_recovery")

AMPLITUDE = 7.90
DT = 0.000125
FINAL_STEP = 128
ANCHOR_STEP = 64
SEGMENT_STEPS = 8
R8_SURFACE_STEPS = tuple(range(32, FINAL_STEP + 1, 8))
R10_SURFACE_STEPS = (64, 80, 96, 112, 128)
STABILITY_STEPS = (64, 96, 128)
REPLAY_LIMIT = 5e-10
OUTER_CORRECTION_LIMIT = 0.20
R8_LABELS = ("R8G7", "R8G8")
R10_LABELS = ("R10G7", "R10G8")
BOOTSTRAP = Path("run_corrected_A790_t016_long_time_unfolding_bootstrap.py")


class ReplayFailure(RuntimeError):
    """The first 64 steps did not reproduce the sealed trajectory."""


class EvolutionControlFailure(RuntimeError):
    """A technical evolution-validity gate failed."""


def validate_bootstrap_entry():
    main_module = sys.modules.get("__main__")
    observed = Path(getattr(main_module, "__file__", "")).resolve()
    expected = (ROOT / BOOTSTRAP).resolve()
    if observed != expected:
        raise RuntimeError("scientific runner was not entered through the frozen bootstrap")


def builtin(value):
    if isinstance(value, dict):
        return {str(key): builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def strict_json(path: Path) -> dict:
    def reject_constant(value):
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def load_and_verify_freeze() -> dict[str, str]:
    if not FREEZE.is_file():
        raise FileNotFoundError(
            "prospective freeze is absent; manufactured review must finish before science"
        )
    record = strict_json(FREEZE)
    if set(record) != {"schema", "protocol_sha256", "runtime", "files"}:
        raise RuntimeError("freeze record schema keys differ")
    if record["schema"] != "A790-t016-long-time-unfolding-freeze-v1":
        raise RuntimeError("freeze schema differs")
    if sha256_file(PROTOCOL) != record["protocol_sha256"]:
        raise RuntimeError("protocol bytes differ from freeze")
    if record["runtime"] != runtime_record():
        raise RuntimeError("live numerical runtime differs from freeze")
    files = record["files"]
    if not isinstance(files, dict) or not files:
        raise RuntimeError("freeze source inventory is empty")
    for name, expected in files.items():
        if not isinstance(name, str) or not isinstance(expected, str):
            raise RuntimeError("freeze file record type differs")
        path = Path(name)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"frozen file is absent or nonregular: {name}")
        if sha256_file(path) != expected:
            raise RuntimeError(f"frozen file hash differs: {name}")
    return dict(files)


def runtime_record():
    dependencies = np.__config__.CONFIG.get("Build Dependencies", {})
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "executable": str(Path(sys.executable).resolve()),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "numpy_version": np.__version__,
        "numpy_file": str(Path(np.__file__).resolve()),
        "scipy_version": scipy.__version__,
        "scipy_file": str(Path(scipy.__file__).resolve()),
        "longdouble_itemsize": np.dtype(np.longdouble).itemsize,
        "blas": dependencies.get("blas", {}).get("name"),
        "lapack": dependencies.get("lapack", {}).get("name"),
        "thread_controls": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def relative_l2(left, right) -> float:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    return float(
        np.linalg.norm((a - b).ravel())
        / max(np.linalg.norm(a.ravel()), np.linalg.norm(b.ravel()), 1e-300)
    )


def build_r8_geometries():
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": AMPLITUDE}
    g7 = build_refined(
        seed,
        81,
        121,
        "G7A790-t016-unfolding",
        selector_iterations=40,
        slice_iterations=270,
    )
    g8 = build_refined(
        g7,
        97,
        145,
        "G8A790-t016-unfolding",
        selector_iterations=45,
        slice_iterations=280,
    )
    return {"R8G7": g7, "R8G8": g8}


def build_r10_geometries():
    g7, g8 = build_A790_R10_pair()
    return {"R10G7": g7, "R10G8": g8}


def make_cases(geometries):
    return {
        label: live.setup_case(
            geometry,
            f"{label}-A790-t016-unfolding",
            live_normal_wall_gauge=True,
            live_outer_sommerfeld=True,
        )
        for label, geometry in geometries.items()
    }


def stage_json(index, stage_id, path, kind, metadata, producer, expected=900.0):
    index.register(stage_id, kind, expected, metadata)
    cached = index.validated_path(stage_id)
    if cached is not None:
        return strict_json(cached)
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        payload = builtin(producer())
        atomic_write_json(path, payload)
        if strict_json(path) != payload:
            raise RuntimeError(f"JSON reload differs for {stage_id}")
        index.mark_complete(stage_id, path, time.perf_counter() - started)
        return payload
    except Exception as error:
        index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
        raise


def segment_path(label, start, end):
    return RECOVERY / f"evolution_{label}_steps_{start + 1:03d}_{end:03d}.npz"


def validate_segment(path, case, start, end):
    shape = tuple(case["initial"].shape)
    source_shape = tuple(case["source0"].shape)
    required = {
        "start_step": (),
        "end_step": (),
        "end_position": shape,
        "end_velocity": shape,
        "end_source": source_shape,
        "end_memory": source_shape,
    }
    for step in range(start + 1, end + 1):
        required[f"step_{step:03d}_increment"] = shape
        required[f"step_{step:03d}_velocity"] = shape
    validate_npz(path, required)
    with np.load(path, allow_pickle=False) as archive:
        if int(archive["start_step"]) != start or int(archive["end_step"]) != end:
            raise RuntimeError("segment index differs")
        if str(archive["schema"].item()) != "A790-t016-evolution-segment-v1":
            raise RuntimeError("segment schema differs")


def exact_diagnostic(case, current_time, state):
    record = long85.diagnose_state(case, current_time, state)
    return {key: value for key, value in record.items() if not key.startswith("_")}


def diagnostic_passes(record):
    return bool(
        record["finite"]
        and record["Lorentzian"]
        and record["global_GH_constraint"] < 0.005
        and record["wall_position_residual"] < 0.0005
        and record["normal_wall_position_residual"] < 0.0005
        and record["outer_position_residual"] < 1e-10
        and record["outer_source_residual"] < 1e-10
    )


def stage_diagnostic_passes(record):
    return bool(
        record["all_stages_finite"]
        and record["all_stage_signatures_lorentzian"]
        and record["maximum_normal_wall_acceleration_residual"] < 1e-10
        and record["maximum_outer_acceleration_residual"] < 1e-10
        and record["maximum_outer_source_residual"] < 1e-10
        and max(
            record["maximum_outer_metric_correction"],
            record["maximum_outer_scalar_correction"],
            record["maximum_outer_source_correction"],
        )
        < OUTER_CORRECTION_LIMIT
    )


def integrate_segment_checked(case, state, start_step, end_step):
    current = tuple(np.asarray(value).copy() for value in state)
    diagnostic = test10.empty_segment_diagnostics()
    diagnostic["all_stage_signatures_lorentzian"] = True
    snapshots = {}
    for step in range(start_step + 1, end_step + 1):
        current_time = (step - 1) * DT
        print(f"{case['label']}: restartable step {step}/{FINAL_STEP}, stage 1", flush=True)
        diagnostic["all_stage_signatures_lorentzian"] = bool(
            diagnostic["all_stage_signatures_lorentzian"]
            and long85.signature_summary(current[0], case["r"])[
                "all_points_one_negative_direction"
            ]
        )
        k1, d1 = live.driver_stage(case, current_time, *current)
        midpoint = tuple(
            value + 0.5 * DT * slope for value, slope in zip(current, k1)
        )
        diagnostic["all_stage_signatures_lorentzian"] = bool(
            diagnostic["all_stage_signatures_lorentzian"]
            and long85.signature_summary(midpoint[0], case["r"])[
                "all_points_one_negative_direction"
            ]
        )
        print(f"{case['label']}: restartable step {step}/{FINAL_STEP}, stage 2", flush=True)
        k2, d2 = live.driver_stage(case, current_time + 0.5 * DT, *midpoint)
        test10.update_segment_diagnostics(diagnostic, k1, k2, (d1, d2))
        current = tuple(value + DT * slope for value, slope in zip(current, k2))
        diagnostic["all_stage_signatures_lorentzian"] = bool(
            diagnostic["all_stage_signatures_lorentzian"]
            and long85.signature_summary(current[0], case["r"])[
                "all_points_one_negative_direction"
            ]
        )
        snapshots[f"step_{step:03d}_increment"] = current[0] - case["initial"]
        snapshots[f"step_{step:03d}_velocity"] = current[1].copy()
    return current, snapshots, diagnostic


def reference_anchor(label, case):
    short = label.removeprefix("R8").removeprefix("R10")
    if label.startswith("R8"):
        with np.load(R8_ARCHIVE, allow_pickle=False) as archive:
            if not np.array_equal(case["z"], archive[f"{short}_z"]):
                raise ReplayFailure(f"{label} compact grid differs")
            if not np.array_equal(case["r"], archive[f"{short}_r"]):
                raise ReplayFailure(f"{label} radial grid differs")
            return {
                "position": np.asarray(archive[f"{short}_position_history"])[64],
                "velocity": np.asarray(archive[f"{short}_velocity_history"])[64],
                "source_increment": np.asarray(
                    archive[f"{short}_source_diagnostic_history"]
                )[-1],
                "memory": None,
            }
    with np.load(R10_STATE, allow_pickle=False) as state_archive:
        if not np.array_equal(case["z"], state_archive[f"{label}_z"]):
            raise ReplayFailure(f"{label} compact grid differs")
        if not np.array_equal(case["r"], state_archive[f"{label}_r"]):
            raise ReplayFailure(f"{label} radial grid differs")
    path = R10_RECOVERY / f"evolution_{label}_long_steps_057_064.npz"
    with np.load(path, allow_pickle=False) as archive:
        return {
            "position": np.asarray(archive["end_position"]),
            "velocity": np.asarray(archive["end_velocity"]),
            "source_increment": np.asarray(archive["end_source"]) - case["source0"],
            "memory": np.asarray(archive["end_memory"]),
        }


def replay_record(label, case, state):
    reference = reference_anchor(label, case)
    values = {
        "position_relative_l2": relative_l2(state[0], reference["position"]),
        "velocity_relative_l2": relative_l2(state[1], reference["velocity"]),
        "source_increment_relative_l2": relative_l2(
            state[2] - case["source0"], reference["source_increment"]
        ),
    }
    if reference["memory"] is not None:
        values["memory_relative_l2"] = relative_l2(state[3], reference["memory"])
    return {
        "label": label,
        "step": ANCHOR_STEP,
        "time": ANCHOR_STEP * DT,
        "limit": REPLAY_LIMIT,
        "differences": values,
        "passed": bool(max(values.values()) < REPLAY_LIMIT),
    }


def read_end_state(path):
    with np.load(path, allow_pickle=False) as archive:
        return tuple(
            np.asarray(archive[key])
            for key in ("end_position", "end_velocity", "end_source", "end_memory")
        )


def run_evolution_case(index, label, case):
    state = (
        case["initial"].copy(),
        np.zeros_like(case["initial"]),
        case["source0"].copy(),
        case["memory0"].copy(),
    )
    parent_fingerprint = test10.geometry_fingerprint(case)
    previous_sha = parent_fingerprint
    diagnostics = []
    replay = None
    paths = []
    for start in range(0, FINAL_STEP, SEGMENT_STEPS):
        if start == ANCHOR_STEP:
            replay_path = RECOVERY / f"replay_{label}_step_064.json"
            replay = stage_json(
                index,
                f"replay/{label}/step_064",
                replay_path,
                "sealed-trajectory-replay",
                {"label": label, "step": ANCHOR_STEP, "limit": REPLAY_LIMIT},
                lambda: replay_record(label, case, state),
                expected=60.0,
            )
            if not replay["passed"]:
                raise ReplayFailure(f"{label} did not reproduce sealed t=0.008 state")
        end = min(start + SEGMENT_STEPS, FINAL_STEP)
        stage_id = f"evolution/{label}/steps_{start + 1:03d}_{end:03d}"
        path = segment_path(label, start, end)
        metadata = {
            "label": label,
            "start_step": start,
            "end_step": end,
            "dt": DT,
            "parent_fingerprint": parent_fingerprint,
            "previous_sha256": previous_sha,
        }
        index.register(stage_id, "evolution-segment", 3600.0, metadata)
        cached = index.validated_path(stage_id)
        if cached is None:
            index.mark_running(stage_id)
            started = time.perf_counter()
            try:
                state, snapshots, diagnostic = integrate_segment_checked(
                    case, state, start, end
                )
                exact = exact_diagnostic(case, end * DT, state)
                if not diagnostic["all_stages_finite"] or not exact["finite"]:
                    raise EvolutionControlFailure("nonfinite evolution stage")
                if not exact["Lorentzian"]:
                    raise EvolutionControlFailure("Lorentzian signature lost")
                corrections = (
                    diagnostic["maximum_outer_metric_correction"],
                    diagnostic["maximum_outer_scalar_correction"],
                    diagnostic["maximum_outer_source_correction"],
                )
                if max(corrections) >= OUTER_CORRECTION_LIMIT:
                    raise EvolutionControlFailure("20-percent outer correction stop")
                if not stage_diagnostic_passes(diagnostic):
                    raise EvolutionControlFailure("midpoint-stage diagnostic failed")
                if not diagnostic_passes(exact):
                    raise EvolutionControlFailure("exact segment diagnostic failed")
                arrays = {
                    "schema": np.asarray("A790-t016-evolution-segment-v1"),
                    "start_step": np.asarray(start),
                    "end_step": np.asarray(end),
                    "end_position": state[0],
                    "end_velocity": state[1],
                    "end_source": state[2],
                    "end_memory": state[3],
                    **snapshots,
                    **test10.diagnostic_arrays(diagnostic),
                    **{f"exact_{key}": np.asarray(value) for key, value in exact.items()},
                }
                atomic_write_npz(path, **arrays)
                validate_segment(path, case, start, end)
                index.mark_complete(stage_id, path, time.perf_counter() - started)
                cached = path
            except Exception as error:
                index.mark_failed(stage_id, f"{type(error).__name__}: {error}")
                raise
        validate_segment(cached, case, start, end)
        state = read_end_state(cached)
        with np.load(cached, allow_pickle=False) as archive:
            exact = {
                key.removeprefix("exact_"): archive[key].item()
                for key in archive.files
                if key.startswith("exact_")
            }
            diagnostic = test10.diagnostic_from_archive(archive)
        if not diagnostic_passes(exact):
            raise EvolutionControlFailure(f"cached diagnostic failed for {stage_id}")
        if not stage_diagnostic_passes(diagnostic):
            raise EvolutionControlFailure(f"cached stage diagnostic failed for {stage_id}")
        diagnostics.append({"step": end, "exact": exact, "stage": diagnostic})
        paths.append(cached)
        previous_sha = sha256_file(cached)
    if replay is None:
        replay_path = RECOVERY / f"replay_{label}_step_064.json"
        replay = strict_json(replay_path)
    return {
        "label": label,
        "state": state,
        "replay": replay,
        "diagnostics": diagnostics,
        "paths": paths,
        "final": {
            "position": state[0],
            "velocity": state[1],
            "source": state[2],
            "memory": state[3],
            "increment": state[0] - case["initial"],
            "source_increment": state[2] - case["source0"],
        },
    }


def load_step(label, case, step):
    start = ((int(step) - 1) // SEGMENT_STEPS) * SEGMENT_STEPS
    end = min(start + SEGMENT_STEPS, FINAL_STEP)
    with np.load(segment_path(label, start, end), allow_pickle=False) as archive:
        return {
            "position": case["initial"] + np.asarray(
                archive[f"step_{step:03d}_increment"]
            ),
            "velocity": np.asarray(archive[f"step_{step:03d}_velocity"]),
        }


def surface_stage(index, label, step, state, geometry):
    stage_id = f"surface/{label}/step_{step:03d}"
    path = RECOVERY / f"surface_{label}_step_{step:03d}.json"
    return stage_json(
        index,
        stage_id,
        path,
        "donor-capped-BVP-and-geometry",
        {"label": label, "step": int(step), "time": step * DT},
        lambda: _calculate_surface_record(label, step, state, geometry),
        expected=1200.0,
    )


def _calculate_surface_record(label, step, state, geometry):
    search = detector80.search_slice(
        f"{label}-A790-t016-t{step * DT:.6f}",
        state["position"],
        state["velocity"],
        geometry,
    )
    branches = []
    if search["admitted_distinct_count"] == 2 and len(search["clusters"]) == 2:
        branches = test10.representative_geometry(search, state, geometry)
    return {
        "label": label,
        "step": int(step),
        "time": step * DT,
        "search": search,
        "branches": branches,
    }


def surfaces_for_case(index, label, case, geometry, steps):
    records = {}
    for step in steps:
        state = load_step(label, case, step)
        records[str(step)] = surface_stage(index, label, step, state, geometry)
    return records


def median_cluster_seed(cluster):
    seeds = sorted(float(member["seed"]) for member in cluster["members"])
    return seeds[len(seeds) // 2]


def stability_stage(index, label, step, state, geometry, surface_record):
    stage_id = f"stability/{label}/step_{step:03d}"
    path = RECOVERY / f"stability_{label}_step_{step:03d}.json"
    return stage_json(
        index,
        stage_id,
        path,
        "principal-MOTS-stability",
        {"label": label, "step": int(step), "time": step * DT},
        lambda: _calculate_stability(state, geometry, surface_record),
        expected=1800.0,
    )


def _calculate_stability(state, geometry, surface_record):
    search = surface_record["search"]
    if search["admitted_distinct_count"] != 2 or len(search["clusters"]) != 2:
        return {"available": False, "reason": "surface count is not two"}
    records = {}
    for branch, cluster in zip(("inner", "outer"), search["clusters"]):
        seed = median_cluster_seed(cluster)
        surface = stability84.recover_surface(
            state["position"], state["velocity"], geometry, seed
        )
        records[branch] = {
            "seed": seed,
            "surface_passes": stability84.surface_passes(surface),
            "stability": stability84.stability_series(
                state["position"], state["velocity"], geometry, surface
            ),
        }
    return {"available": True, "branches": records}


def stability_for_r8(index, label, case, geometry, surfaces):
    records = {}
    for step in STABILITY_STEPS:
        state = load_step(label, case, step)
        records[str(step)] = stability_stage(
            index, label, step, state, geometry, surfaces[str(step)]
        )
    return records


def detector_rule_passes(search):
    return bool(
        search["admitted_distinct_count"] == 2
        and len(search["clusters"]) == 2
        and all(len(cluster["members"]) >= 2 for cluster in search["clusters"])
        and all(
            trial["surface"]["local_expansion_interior_maximum"] < 2e-4
            and trial["surface"]["boundary_slope_error"] < 2e-4
            and trial["surface"]["primary_evaluator_crosscheck"][
                "two_cell_interior_maximum"
            ]
            < 0.002
            for trial in search["trials"]
            if trial["admitted"]
        )
    )


def surface_history_passes(records):
    return bool(
        all(detector_rule_passes(record["search"]) for record in records.values())
        and all(
            len(record["branches"]) == 2
            and all(
                branch["admitted"]
                and branch["geometry"]["finite"]
                and branch["geometry"]["one_sided_cap_area"] > 0.0
                for branch in record["branches"]
            )
            for record in records.values()
        )
        and branch_identity_record(records)["passed"]
    )


def branch_identity_record(records):
    ordered = [records[str(step)] for step in sorted(int(value) for value in records)]
    comparisons = []
    passed = True
    for left, right in zip(ordered, ordered[1:]):
        if len(left["branches"]) != 2 or len(right["branches"]) != 2:
            comparisons.append({
                "left_step": left["step"],
                "right_step": right["step"],
                "available": False,
            })
            passed = False
            continue
        a = np.asarray([
            [item["geometry"]["rho_axis"], item["geometry"]["rho_brane"]]
            for item in left["branches"]
        ])
        b = np.asarray([
            [item["geometry"]["rho_axis"], item["geometry"]["rho_brane"]]
            for item in right["branches"]
        ])
        same = float(np.linalg.norm(a[0] - b[0]) + np.linalg.norm(a[1] - b[1]))
        swapped = float(np.linalg.norm(a[0] - b[1]) + np.linalg.norm(a[1] - b[0]))
        local_pass = bool(same < swapped)
        comparisons.append({
            "left_step": left["step"],
            "right_step": right["step"],
            "available": True,
            "same_assignment_distance": same,
            "swapped_assignment_distance": swapped,
            "same_assignment_preferred": local_pass,
        })
        passed = bool(passed and local_pass)
    return {"passed": passed, "comparisons": comparisons}


def stability_branch_passes(branch):
    record = branch["stability"]
    fine = record["fine_principal_eigenvalue"]
    convergence_limit = max(0.02, 0.05 * abs(fine))
    matrix_pass = all(
        spectrum["left_neumann_defect"] < 1e-10
        and spectrum["right_neumann_defect"] < 1e-10
        and spectrum["minimum_normal_factor"] > 0.0
        and np.isfinite(spectrum["operator_frobenius_norm"])
        and abs(spectrum["principal_eigenvalue_imaginary"])
        < 1e-6 * max(1.0, abs(spectrum["principal_eigenvalue_real"]))
        and spectrum["principal_eigenfunction_sign_changes"] == 0
        for spectrum in record["spectra"].values()
    )
    return bool(
        branch["surface_passes"]
        and record["resolved"]
        and record["angular_difference_65_81"] < convergence_limit
        and record["Frechet_step_difference_81"] < convergence_limit
        and matrix_pass
    )


def stability_history_passes(records):
    expected = {"inner": "outward_unstable", "outer": "outward_stable"}
    return bool(
        all(record.get("available") for record in records.values())
        and all(
            stability_branch_passes(branch)
            and branch["stability"]["classification"] == expected[name]
            for record in records.values()
            for name, branch in record["branches"].items()
        )
    )


def cross_grid_stability_passes(left, right):
    records = {}
    passed = True
    for step in STABILITY_STEPS:
        step_key = str(step)
        records[step_key] = {}
        for branch in ("inner", "outer"):
            lv = left[step_key]["branches"][branch]["stability"][
                "fine_principal_eigenvalue"
            ]
            rv = right[step_key]["branches"][branch]["stability"][
                "fine_principal_eigenvalue"
            ]
            transfer = stability84.transfer(lv, rv)
            records[step_key][branch] = transfer
            passed = bool(
                passed
                and (transfer["relative"] < 0.10 or transfer["absolute"] < 0.02)
            )
    return passed, records


def branch_series(records, branch_index, key):
    values = []
    for step in sorted(int(item) for item in records):
        branch = records[str(step)]["branches"][branch_index]
        if key == "area":
            value = branch["geometry"]["one_sided_cap_area"]
        elif key == "equivalent_radius":
            value = branch["geometry"]["equivalent_area_radius"]
        elif key == "axis":
            value = branch["geometry"]["rho_axis"]
        elif key == "brane":
            value = branch["geometry"]["rho_brane"]
        else:
            raise KeyError(key)
        values.append((step, float(value)))
    return values


def trend_record(records, branch_index, key):
    series = branch_series(records, branch_index, key)
    changes = [
        {"left_step": a[0], "right_step": b[0], "change": b[1] - a[1]}
        for a, b in zip(series, series[1:])
    ]
    early = [abs(item["change"]) for item in changes if 32 <= item["left_step"] < 64]
    late = [abs(item["change"]) for item in changes if 96 <= item["left_step"] < 128]
    signs = [np.sign(item["change"]) for item in changes]
    return {
        "series": [{"step": step, "time": step * DT, "value": value} for step, value in series],
        "changes": changes,
        "early_median_absolute_change": float(np.median(early)) if early else None,
        "late_median_absolute_change": float(np.median(late)) if late else None,
        "sign_coherent": bool(signs and all(sign == signs[0] and sign != 0 for sign in signs)),
    }


def trend_summary(surfaces):
    raw = {
        label: {
            branch: {
                key: trend_record(records, branch_index, key)
                for key in ("area", "equivalent_radius", "axis", "brane")
            }
            for branch_index, branch in enumerate(("inner", "outer"))
        }
        for label, records in surfaces.items()
    }
    if set(raw) != {"R8G7", "R8G8"}:
        return {"raw": raw, "classifications": {}}
    classifications = {}
    for branch in ("inner", "outer"):
        classifications[branch] = {}
        for key in ("area", "equivalent_radius", "axis", "brane"):
            left = raw["R8G7"][branch][key]
            right = raw["R8G8"][branch][key]
            left_changes = [item["change"] for item in left["changes"]]
            right_changes = [item["change"] for item in right["changes"]]
            if len(left_changes) != len(right_changes) or not left_changes:
                classification = "UNRESOLVED"
                floor = None
            else:
                floor = float(np.max(np.abs(np.asarray(left_changes) - right_changes)))
                coherent = bool(left["sign_coherent"] and right["sign_coherent"])
                early = max(
                    left["early_median_absolute_change"],
                    right["early_median_absolute_change"],
                )
                late = max(
                    left["late_median_absolute_change"],
                    right["late_median_absolute_change"],
                )
                if coherent and early - late > floor:
                    classification = "SLOWING"
                elif coherent:
                    classification = "CONTINUING"
                else:
                    classification = "NONMONOTONE"
            classifications[branch][key] = {
                "classification": classification,
                "maximum_cross_grid_change_disagreement": floor,
            }
    return {"raw": raw, "classifications": classifications}


def stability_trends(records):
    result = {}
    for label, history in records.items():
        result[label] = {}
        for branch in ("inner", "outer"):
            series = [
                {
                    "step": int(step),
                    "time": int(step) * DT,
                    "principal_eigenvalue": history[step]["branches"][branch][
                        "stability"
                    ]["fine_principal_eigenvalue"],
                }
                for step in sorted(history, key=int)
                if history[step].get("available")
            ]
            result[label][branch] = {
                "series": series,
                "signed_changes": [
                    right["principal_eigenvalue"] - left["principal_eigenvalue"]
                    for left, right in zip(series, series[1:])
                ],
            }
    return result


def cross_grid_surface_transfer(left, right, limit=0.01):
    records = {}
    passed = True
    for step in sorted(set(left) & set(right), key=int):
        if len(left[step]["branches"]) != 2 or len(right[step]["branches"]) != 2:
            records[step] = None
            passed = False
            continue
        transfer = test10.surface_transfer(
            left[step]["branches"], right[step]["branches"]
        )
        records[step] = transfer
        passed = bool(passed and transfer["maximum"] < limit)
    return passed, records


def classify_result(r8_evolution, r8_surfaces, r8_stability, domain_pass):
    r8_pass = bool(r8_evolution and r8_surfaces and r8_stability)
    if r8_pass and domain_pass:
        return "pass-domain", "LONG-TIME-PAIR-DOMAIN-TRANSFERRED"
    if r8_pass:
        return "pass", "LONG-TIME-PAIR-WITH-RESOLVED-STABILITY"
    if r8_evolution:
        return "review", "BRANCH-STRUCTURE-CHANGED-OR-UNRESOLVED"
    return "review", "LONG-TIME-NUMERICAL-CONTROL-INCOMPLETE"


def final_field_domain_transfer(r8_cases, r8_runs, r10_cases, r10_runs):
    records = {}
    for grid in ("G7", "G8"):
        left_label = f"R8{grid}"
        right_label = f"R10{grid}"
        left = r8_runs[left_label]["final"]
        right = r10_runs[right_label]["final"]
        left_public = {
            "_increment": left["increment"],
            "_velocity": left["velocity"],
            "_source_increment": left["source_increment"],
        }
        right_public = {
            "_increment": right["increment"],
            "_velocity": right["velocity"],
            "_source_increment": right["source_increment"],
        }
        records[grid] = {
            name: long85.field_transfer(
                r8_cases[left_label], left_public,
                r10_cases[right_label], right_public, field,
            )
            for name, field in (
                ("position_increment", "_increment"),
                ("velocity", "_velocity"),
                ("source_increment", "_source_increment"),
            )
        }
    return records


def surface_domain_transfer(r8_surfaces, r10_surfaces):
    records = {}
    passed = True
    for grid in ("G7", "G8"):
        left_label = f"R8{grid}"
        right_label = f"R10{grid}"
        records[grid] = {}
        for step in R10_SURFACE_STEPS:
            transfer = test10.surface_transfer(
                r8_surfaces[left_label][str(step)]["branches"],
                r10_surfaces[right_label][str(step)]["branches"],
            )
            records[grid][str(step)] = transfer
            passed = bool(passed and transfer["maximum"] < 0.03)
    return passed, records


def public_run(run):
    return {
        "label": run["label"],
        "replay": run["replay"],
        "diagnostics": run["diagnostics"],
        "segment_files": [
            {"path": str(path), "sha256": sha256_file(path), "byte_count": path.stat().st_size}
            for path in run["paths"]
        ],
    }


def main():
    validate_bootstrap_entry()
    started = time.perf_counter()
    frozen_files = load_and_verify_freeze()
    RECOVERY.mkdir(parents=True, exist_ok=True)
    index = RecoveryIndex(MANIFEST, PROTOCOL, frozen_files, maximum_stage_seconds=3600.0)

    controls = stage_json(
        index,
        "controls/preflight",
        RECOVERY / "controls_preflight.json",
        "manufactured-controls",
        {"before_new_physical_slice": True},
        lambda: {
            "BVP": detector80.analytic_controls(),
            "stability": stability84.analytic_control(),
        },
        expected=900.0,
    )
    if not controls["BVP"]["passed"] or not controls["stability"]["passed"]:
        raise RuntimeError("manufactured controls failed")

    print("building fresh R8 G7/G8 geometries", flush=True)
    r8_geometries = build_r8_geometries()
    r8_cases = make_cases(r8_geometries)
    r8_runs = {}
    r8_surfaces = {}
    r8_stability = {}
    for label in R8_LABELS:
        print(f"starting restartable {label} evolution through t=0.016", flush=True)
        r8_runs[label] = run_evolution_case(index, label, r8_cases[label])
        r8_surfaces[label] = surfaces_for_case(
            index, label, r8_cases[label], r8_geometries[label], R8_SURFACE_STEPS
        )
        r8_stability[label] = stability_for_r8(
            index, label, r8_cases[label], r8_geometries[label], r8_surfaces[label]
        )

    stability_transfer_pass, stability_transfer = cross_grid_stability_passes(
        r8_stability["R8G7"], r8_stability["R8G8"]
    )
    r8_surface_transfer_pass, r8_surface_transfer = cross_grid_surface_transfer(
        r8_surfaces["R8G7"], r8_surfaces["R8G8"], limit=0.01
    )
    r8_evolution_pass = all(
        run["replay"]["passed"]
        and all(diagnostic_passes(item["exact"]) for item in run["diagnostics"])
        for run in r8_runs.values()
    )
    r8_surface_pass = bool(
        all(surface_history_passes(item) for item in r8_surfaces.values())
        and r8_surface_transfer_pass
    )
    r8_stability_pass = bool(
        all(stability_history_passes(item) for item in r8_stability.values())
        and stability_transfer_pass
    )
    r8_pass = bool(r8_evolution_pass and r8_surface_pass and r8_stability_pass)

    r10_executed = False
    r10_geometries = {}
    r10_cases = {}
    r10_runs = {}
    r10_surfaces = {}
    domain_surface_transfer = None
    domain_field_transfer = None
    domain_pass = False
    if r8_pass:
        r10_executed = True
        print("R8 barriers pass; building conditional R10 G7/G8 geometries", flush=True)
        r10_geometries = build_r10_geometries()
        r10_cases = make_cases(r10_geometries)
        for label in R10_LABELS:
            print(f"starting conditional restartable {label} evolution", flush=True)
            r10_runs[label] = run_evolution_case(index, label, r10_cases[label])
            r10_surfaces[label] = surfaces_for_case(
                index,
                label,
                r10_cases[label],
                r10_geometries[label],
                R10_SURFACE_STEPS,
            )
        domain_surface_pass, domain_surface_transfer = surface_domain_transfer(
            r8_surfaces, r10_surfaces
        )
        domain_field_transfer = final_field_domain_transfer(
            r8_cases, r8_runs, r10_cases, r10_runs
        )
        domain_pass = bool(
            all(
                run["replay"]["passed"]
                and all(diagnostic_passes(item["exact"]) for item in run["diagnostics"])
                for run in r10_runs.values()
            )
            and all(surface_history_passes(item) for item in r10_surfaces.values())
            and domain_surface_pass
            and max(
                value
                for grid in domain_field_transfer.values()
                for value in grid.values()
            )
            < 0.05
        )

    status, classification = classify_result(
        r8_evolution_pass, r8_surface_pass, r8_stability_pass, domain_pass
    )

    payload = {
        "schema": "A790-t016-long-time-unfolding-result-v1",
        "status": status,
        "classification": classification,
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "freeze_sha256": sha256_file(FREEZE),
        "amplitude": AMPLITUDE,
        "dt": DT,
        "final_step": FINAL_STEP,
        "final_time": FINAL_STEP * DT,
        "controls": controls,
        "R8": {
            "runs": {label: public_run(run) for label, run in r8_runs.items()},
            "surfaces": r8_surfaces,
            "branch_identity": {
                label: branch_identity_record(records)
                for label, records in r8_surfaces.items()
            },
            "surface_transfer": r8_surface_transfer,
            "stability": r8_stability,
            "stability_transfer": stability_transfer,
            "trend_observables": trend_summary(r8_surfaces),
            "stability_trends": stability_trends(r8_stability),
            "gates": {
                "evolution_and_replay": r8_evolution_pass,
                "surface_history": r8_surface_pass,
                "stability": r8_stability_pass,
            },
        },
        "R10": {
            "executed": r10_executed,
            "runs": {label: public_run(run) for label, run in r10_runs.items()},
            "surfaces": r10_surfaces,
            "surface_domain_transfer": domain_surface_transfer,
            "final_field_domain_transfer": domain_field_transfer,
            "domain_pass": domain_pass,
        },
        "next_authorized_calculation": (
            "frozen T=.012,.014,.016 finite-time null-proxy extension"
            if classification == "LONG-TIME-PAIR-DOMAIN-TRANSFERRED"
            else "bounded topology/continuation confirmation"
            if classification == "BRANCH-STRUCTURE-CHANGED-OR-UNRESOLVED"
            else "diagnose numerical control before physical interpretation"
        ),
        "firewall": {
            "event_horizon_claimed": False,
            "connected_throat_claimed": False,
            "topology_change_claimed": False,
            "phase_diagram_claimed": False,
            "source_ownership_claimed": False,
            "manuscript_updated": False,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    stage_json(
        index,
        "result/final",
        OUTPUT,
        "final-result",
        {"schema": payload["schema"], "final_time": payload["final_time"]},
        lambda: payload,
        expected=300.0,
    )
    print(
        json.dumps(
            {
                "status": status,
                "classification": classification,
                "R8_gates": payload["R8"]["gates"],
                "R10_executed": r10_executed,
                "R10_domain_pass": domain_pass,
                "runtime_seconds": payload["runtime_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
