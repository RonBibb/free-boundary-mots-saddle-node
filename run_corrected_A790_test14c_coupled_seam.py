#!/usr/bin/env python3
"""Restartable Test-14C coupled thin-seam control and physical evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.recovery_indexer import RecoveryIndex, atomic_write_json, sha256_file
from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice
from bhps.test14b_balance_closure import five_point_history_derivative
from bhps.test14c_coupled_seam import (
    apply_intrinsic_anisotropy,
    analytic_controls,
    evaluate_geometric_bulk_leaf,
    physical_coupled_record,
    seam_endpoint_transport,
)


PROTOCOL = Path("notes/99_A790_test14C_coupled_thin_seam_protocol.md")
NOTE95 = Path("notes/95_A790_quasilocal_mass_flux_bridge_protocol.md")
NOTE96 = Path("notes/96_A790_test14B_balance_closure_protocol.md")
NOTE100 = Path("notes/100_A790_test14C_intrinsic_anisotropy_addendum.md")
STATE = Path("results/corrected_A790_t008_long_evolution_state.npz")
TEST14B = Path("results/corrected_A790_test14b_balance_closure.json")
SOURCE = Path("src/bhps/test14c_coupled_seam.py")
OUTPUT = Path("results/corrected_A790_test14c_coupled_seam.json")
CONTROL_OUTPUT = Path("results/corrected_A790_test14c_analytic_controls.json")
MANIFEST = Path("results/corrected_A790_test14c_recovery_v4.json")
STAGES = Path("results/corrected_A790_test14c_stages")

GRIDS = ("G7", "G8")
BRANCHES = ("inner", "outer")
STRIDES = (1, 2, 4)
PRIMARY_END = 0.004


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def stage_json(index, stage_id, path, compute, metadata, maximum_seconds=60.0):
    index.register(stage_id, "test14c-coupled-seam", maximum_seconds, metadata)
    validated = index.validated_path(stage_id)
    if validated is not None:
        return json.loads(validated.read_text())
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        payload = _jsonable(compute())
        atomic_write_json(path, payload)
        index.mark_complete(
            stage_id, path, time.perf_counter() - started,
            {"finite": bool(payload.get("finite", True))},
        )
        return payload
    except Exception as error:
        index.mark_failed(stage_id, repr(error))
        raise


def recovery_smoke_test():
    with tempfile.TemporaryDirectory(prefix="bhps-test14c-recovery-") as directory:
        root = Path(directory)
        source = root / "input.txt"
        source.write_text("fixed Test-14C input\n")
        expected = {str(source): sha256_file(source)}
        manifest = root / "manifest.json"
        stage = root / "stage.json"
        first = RecoveryIndex(manifest, PROTOCOL, expected, 60.0)
        first.register("smoke", "test14c-smoke", 10.0, {"stage": 1})
        first.mark_running("smoke")
        atomic_write_json(stage, {"finite": True, "value": 1})
        first.mark_complete("smoke", stage, 0.01)
        second = RecoveryIndex(manifest, PROTOCOL, expected, 60.0)
        resumed = second.validated_path("smoke") == stage
        atomic_write_json(stage, {"finite": True, "value": 2})
        rejected = second.validated_path("smoke") is None
        return {
            "valid_partial_restart": bool(resumed),
            "corruption_rejected": bool(rejected),
            "passed": bool(resumed and rejected),
        }


def window_summary(records, left, right):
    selected = [
        record for record in records
        if float(left) - 1e-12 <= record["time"] <= float(right) + 1e-12
    ]
    if len(selected) < 2:
        raise ValueError("Test-14C window has fewer than two records")
    times = np.asarray([record["time"] for record in selected])
    charges = np.asarray([record["charge"] for record in selected])
    total = np.asarray([record["corrected_total_rate"] for record in selected])
    names = tuple(selected[0]["corrected_rates"])
    named = {
        name: np.asarray([
            record["corrected_rates"][name] for record in selected
        ]) for name in names
    }
    delta = float(charges[-1] - charges[0])
    integrated = {
        name: float(np.trapezoid(values, times))
        for name, values in named.items()
    }
    integrated_total = float(np.trapezoid(total, times))
    flux_norm = float(np.trapezoid(
        np.sum(np.abs(np.stack(list(named.values()))), axis=0), times,
    ))
    norm = max(abs(delta), flux_norm, 1e-12)
    residual = delta - integrated_total
    return {
        "left_time": float(times[0]),
        "right_time": float(times[-1]),
        "sample_count": len(selected),
        "delta_charge": delta,
        "integrated_named_rates": integrated,
        "integrated_total_rate": integrated_total,
        "closure_residual": float(residual),
        "balance_norm": float(norm),
        "normalized_absolute_residual": float(abs(residual) / norm),
    }


def main():
    STAGES.mkdir(parents=True, exist_ok=True)
    required = (
        PROTOCOL, NOTE95, NOTE96, NOTE100, STATE, TEST14B, SOURCE,
        Path(__file__),
        Path("src/bhps/test14b_balance_closure.py"),
        Path("src/bhps/recovery_indexer.py"),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing Test-14C inputs: {missing}")
    expected = {str(path): sha256_file(path) for path in required}
    recovery = recovery_smoke_test()
    index = RecoveryIndex(MANIFEST, PROTOCOL, expected, 600.0)

    controls = stage_json(
        index, "controls/coupled", STAGES / "analytic_controls.json",
        analytic_controls,
        {"regularizers": ["softabs", "tanh", "polynomial"]}, 120.0,
    )
    atomic_write_json(CONTROL_OUTPUT, {
        "status": "complete",
        "protocol_sha256": sha256_file(PROTOCOL),
        "inputs": expected,
        "controls": controls,
        "recovery_control": recovery,
        "passed": bool(controls["passed"] and recovery["passed"]),
    })
    if not controls["passed"] or not recovery["passed"]:
        raise RuntimeError("Test-14C analytic, smoothing, or recovery control failed")

    test14b = json.loads(TEST14B.read_text())
    archive = np.load(STATE)
    times = np.asarray(test14b["times"], dtype=float)
    background = test14b["background"]
    physical = {
        grid: {branch: {str(stride): [] for stride in STRIDES}
               for branch in BRANCHES}
        for grid in GRIDS
    }

    for grid in GRIDS:
        z = np.asarray(archive[f"{grid}_z"], dtype=float)
        r = np.asarray(archive[f"{grid}_r"], dtype=float)
        positions = np.asarray(archive[f"{grid}_position_history"])[5:]
        velocities = np.asarray(archive[f"{grid}_velocity_history"])[5:]
        for branch in BRANCHES:
            surfaces = test14b["surface_records"][grid][branch]
            rho_history = np.stack([
                np.asarray(item["surface"]["rho"], dtype=float)
                for item in surfaces
            ])
            for stride in STRIDES:
                rho_rate = five_point_history_derivative(
                    rho_history, times, stride=stride,
                )
                base_records = test14b["balance_records"][grid][branch][
                    str(stride)
                ]
                c_history = np.asarray([
                    item["seam"]["geometric_Ws_over_W"]
                    for item in base_records
                ])
                c_rate = five_point_history_derivative(
                    c_history, times, stride=stride,
                )
                for local_index, current_time in enumerate(times):
                    label = f"{current_time:.6f}".replace(".", "p")
                    stage_id = f"physical/{grid}/{branch}/stride{stride}/{label}"
                    path = STAGES / (
                        f"physical_{grid}_{branch}_stride{stride}_{label}.json"
                    )

                    def compute(li=local_index, st=stride):
                        profile = surfaces[li]["surface"]
                        endpoint = seam_endpoint_transport(
                            positions[li], velocities[li], z, r, profile,
                            rho_rate[li], background["wall_stiffness"],
                            background["v1"],
                        )
                        return physical_coupled_record(
                            base_records[li], endpoint, c_rate[li],
                        )

                    physical[grid][branch][str(stride)].append(stage_json(
                        index, stage_id, path, compute,
                        {"grid": grid, "branch": branch, "stride": stride,
                         "time": float(current_time)},
                    ))

    bulk_records = {
        grid: {branch: {str(stride): [] for stride in STRIDES}
               for branch in BRANCHES}
        for grid in GRIDS
    }
    corrected_physical = {
        grid: {branch: {str(stride): [] for stride in STRIDES}
               for branch in BRANCHES}
        for grid in GRIDS
    }
    for grid in GRIDS:
        z = np.asarray(archive[f"{grid}_z"], dtype=float)
        r = np.asarray(archive[f"{grid}_r"], dtype=float)
        positions = np.asarray(archive[f"{grid}_position_history"])[5:]
        velocities = np.asarray(archive[f"{grid}_velocity_history"])[5:]
        surfaces_by_branch = {
            branch: test14b["surface_records"][grid][branch]
            for branch in BRANCHES
        }
        rho_rates = {}
        for branch in BRANCHES:
            rho_history = np.stack([
                np.asarray(item["surface"]["rho"], dtype=float)
                for item in surfaces_by_branch[branch]
            ])
            rho_rates[branch] = {
                stride: five_point_history_derivative(
                    rho_history, times, stride=stride,
                ) for stride in STRIDES
            }
        for local_index, current_time in enumerate(times):
            prepared = prepare_capped_expansion_slice(
                positions[local_index], velocities[local_index], z, r,
            )
            for branch in BRANCHES:
                profile = surfaces_by_branch[branch][local_index]["surface"]
                for stride in STRIDES:
                    base_record = test14b["balance_records"][grid][branch][
                        str(stride)
                    ][local_index]
                    label = f"{current_time:.6f}".replace(".", "p")
                    stage_id = f"bulk/{grid}/{branch}/stride{stride}/{label}"
                    path = STAGES / (
                        f"bulk_{grid}_{branch}_stride{stride}_{label}.json"
                    )
                    diagnostic = stage_json(
                        index, stage_id, path,
                        lambda br=base_record, p=profile, b=branch, st=stride,
                               li=local_index: evaluate_geometric_bulk_leaf(
                            positions[li], velocities[li], z, r, p,
                            rho_rates[b][st][li], br, prepared=prepared,
                        ),
                        {"grid": grid, "branch": branch, "stride": stride,
                         "time": float(current_time),
                         "term": "intrinsic_anisotropy"},
                    )
                    bulk_records[grid][branch][str(stride)].append(diagnostic)
                    corrected_physical[grid][branch][str(stride)].append(
                        apply_intrinsic_anisotropy(
                            physical[grid][branch][str(stride)][local_index],
                            diagnostic,
                        )
                    )
    physical = corrected_physical

    windows = {
        "primary": (float(times[0]), PRIMARY_END),
        "sub_001_002": (0.001, 0.002),
        "sub_002_003": (0.002, 0.003),
        "sub_003_004": (0.003, 0.004),
        "extended": (float(times[0]), float(times[-1])),
    }
    summaries = {
        grid: {branch: {} for branch in BRANCHES} for grid in GRIDS
    }
    for grid in GRIDS:
        for branch in BRANCHES:
            for stride in STRIDES:
                records = physical[grid][branch][str(stride)]
                summaries[grid][branch][str(stride)] = {
                    "windows": {
                        name: window_summary(records, *bounds)
                        for name, bounds in windows.items()
                    },
                    "maximum_primary_israel_rate_relative_scale_error": max(
                        record["israel_rate_relative_scale_error"]
                        for record in records
                        if record["time"] <= PRIMARY_END + 1e-12
                    ),
                    "maximum_primary_uncombined_compatibility_error": max(
                        record["uncombined"]["compatibility_error"]
                        for record in records
                        if record["time"] <= PRIMARY_END + 1e-12
                    ),
                    "maximum_primary_anisotropy_to_bulk_defect_error": max(
                        record["anisotropy_to_bulk_defect_error"]
                        for record in records
                        if record["time"] <= PRIMARY_END + 1e-12
                    ),
                }

    pilot_index = int(np.argmin(np.abs(times - 0.001)))
    pilot = {
        grid: {
            branch: physical[grid][branch]["1"][pilot_index]
            for branch in BRANCHES
        } for grid in GRIDS
    }
    preliminary_closure = bool(all(
        summaries[grid][branch]["1"]["windows"]["primary"][
            "normalized_absolute_residual"
        ] < 0.05
        for grid in GRIDS for branch in BRANCHES
    ))
    result = {
        "status": "coupled seam and intrinsic-anisotropy dense candidate "
                  "complete; independent/tangency audits pending",
        "overall_grade": "IN_PROGRESS",
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "inputs": expected,
        "controls": controls,
        "recovery_control": recovery,
        "times": times.tolist(),
        "pilot_time": float(times[pilot_index]),
        "pilot": pilot,
        "physical_records": physical,
        "bulk_records": bulk_records,
        "summaries": summaries,
        "preliminary_coupled_closure_below_5_percent": preliminary_closure,
        "pending": [
            "physical coupled regularization at selected leaves",
            "normal-versus-material generator transport audit",
            "independent reconstruction and final grade",
        ],
        "claim_boundary": "No physical closure claim while bulk gate is pending.",
    }
    atomic_write_json(OUTPUT, result)
    print(json.dumps({
        "output": str(OUTPUT),
        "protocol_sha256": result["protocol_sha256"],
        "controls_passed": controls["passed"],
        "preliminary_coupled_closure_below_5_percent": preliminary_closure,
        "pilot": {
            grid: {
                branch: {
                    "coupled_seam_rate": pilot[grid][branch][
                        "coupled_seam"
                    ]["total"],
                    "corrected_pointwise_residual": pilot[grid][branch][
                        "corrected_pointwise_residual"
                    ],
                } for branch in BRANCHES
            } for grid in GRIDS
        },
    }, indent=2))


if __name__ == "__main__":
    main()
