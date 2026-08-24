#!/usr/bin/env python3
"""Independent tangency and arithmetic audit for dense Test 14C."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice
from bhps.recovery_indexer import RecoveryIndex, atomic_write_json, sha256_file
from bhps.test14b_balance_closure import five_point_history_derivative
from bhps.test14c_coupled_seam import (
    leaf_marginal_transport_fields,
    marginal_tangency_rate,
    relative_scale_error,
)


PROTOCOL = Path("notes/99_A790_test14C_coupled_thin_seam_protocol.md")
ADDENDUM = Path("notes/100_A790_test14C_intrinsic_anisotropy_addendum.md")
STATE = Path("results/corrected_A790_t008_long_evolution_state.npz")
TEST14B = Path("results/corrected_A790_test14b_balance_closure.json")
TEST14C = Path("results/corrected_A790_test14c_coupled_seam.json")
EINSTEIN14B = Path(
    "results/corrected_A790_test14b_independent_einstein_checks.json"
)
SOURCE = Path("src/bhps/test14c_coupled_seam.py")
OUTPUT = Path("results/corrected_A790_test14c_independent_audit.json")
MANIFEST = Path("results/corrected_A790_test14c_independent_recovery_v2.json")
STAGES = Path("results/corrected_A790_test14c_independent_stages")

GRIDS = ("G7", "G8")
BRANCHES = ("inner", "outer")
STRIDES = (1, 2)
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


def stage_json(index, stage_id, path, compute, metadata, maximum_seconds=120.0):
    index.register(stage_id, "test14c-independent", maximum_seconds, metadata)
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


def unpack_fields(record):
    return {
        name: np.asarray(value, dtype=float)
        for name, value in record.items()
        if name not in ("finite", "grid", "branch", "stride", "time")
    }


def main():
    STAGES.mkdir(parents=True, exist_ok=True)
    required = (
        PROTOCOL, ADDENDUM, STATE, TEST14B, TEST14C, EINSTEIN14B,
        SOURCE, Path(__file__), Path("src/bhps/recovery_indexer.py"),
    )
    expected = {str(path): sha256_file(path) for path in required}
    index = RecoveryIndex(MANIFEST, PROTOCOL, expected, 600.0)
    test14b = json.loads(TEST14B.read_text())
    test14c = json.loads(TEST14C.read_text())
    einstein = json.loads(EINSTEIN14B.read_text())
    archive = np.load(STATE)
    times = np.asarray(test14c["times"], dtype=float)

    leaf_fields = {
        grid: {branch: {str(stride): [] for stride in STRIDES}
               for branch in BRANCHES}
        for grid in GRIDS
    }
    for grid in GRIDS:
        z = np.asarray(archive[f"{grid}_z"], dtype=float)
        r = np.asarray(archive[f"{grid}_r"], dtype=float)
        positions = np.asarray(archive[f"{grid}_position_history"])[5:]
        velocities = np.asarray(archive[f"{grid}_velocity_history"])[5:]
        surfaces = {
            branch: test14b["surface_records"][grid][branch]
            for branch in BRANCHES
        }
        rho_rates = {}
        for branch in BRANCHES:
            rho = np.stack([
                np.asarray(item["surface"]["rho"], dtype=float)
                for item in surfaces[branch]
            ])
            rho_rates[branch] = {
                stride: five_point_history_derivative(
                    rho, times, stride=stride,
                ) for stride in STRIDES
            }
        for local_index, current_time in enumerate(times):
            prepared = prepare_capped_expansion_slice(
                positions[local_index], velocities[local_index], z, r,
            )
            for branch in BRANCHES:
                profile = surfaces[branch][local_index]["surface"]
                for stride in STRIDES:
                    label = f"{current_time:.6f}".replace(".", "p")
                    stage_id = f"leaf/{grid}/{branch}/stride{stride}/{label}"
                    path = STAGES / (
                        f"leaf_{grid}_{branch}_stride{stride}_{label}.json"
                    )

                    def compute(li=local_index, b=branch, st=stride, p=profile):
                        fields = leaf_marginal_transport_fields(
                            positions[li], velocities[li], z, r, p,
                            rho_rates[b][st][li], prepared=prepared,
                        )
                        return {
                            **fields, "grid": grid, "branch": b,
                            "stride": st, "time": float(times[li]),
                        }

                    leaf_fields[grid][branch][str(stride)].append(stage_json(
                        index, stage_id, path, compute,
                        {"grid": grid, "branch": branch, "stride": stride,
                         "time": float(current_time)},
                    ))

    tangency = {
        grid: {branch: {} for branch in BRANCHES} for grid in GRIDS
    }
    for grid in GRIDS:
        for branch in BRANCHES:
            for stride in STRIDES:
                fields_records = leaf_fields[grid][branch][str(stride)]
                theta_l_history = np.stack([
                    np.asarray(item["theta_l"], dtype=float)
                    for item in fields_records
                ])
                theta_l_rate = five_point_history_derivative(
                    theta_l_history, times, stride=stride,
                )
                records = []
                for local_index, current_time in enumerate(times):
                    fields = unpack_fields(fields_records[local_index])
                    area_radius = test14b["balance_records"][grid][branch][
                        str(stride)
                    ][local_index]["charge"]["equivalent_area_radius"]
                    record = marginal_tangency_rate(
                        theta_l_rate[local_index], fields, area_radius,
                    )
                    records.append({
                        **record, "grid": grid, "branch": branch,
                        "stride": stride, "time": float(current_time),
                    })
                selected = [
                    record for record in records
                    if record["time"] <= PRIMARY_END + 1e-12
                ]
                selected_times = np.asarray([x["time"] for x in selected])
                terms = np.asarray([
                    x["hawking_product_derivative_term"] for x in selected
                ])
                integrated = float(np.trapezoid(terms, selected_times))
                norm = test14c["summaries"][grid][branch][str(stride)][
                    "windows"
                ]["primary"]["balance_norm"]
                tangency[grid][branch][str(stride)] = {
                    "records": records,
                    "primary_integrated_hawking_product_term": integrated,
                    "primary_relative_to_balance_norm": float(
                        abs(integrated) / max(abs(norm), 1e-300)
                    ),
                    "passed_below_2_percent": bool(
                        abs(integrated) / max(abs(norm), 1e-300) < 0.02
                    ),
                }

    arithmetic = {
        grid: {branch: {} for branch in BRANCHES} for grid in GRIDS
    }
    for grid in GRIDS:
        for branch in BRANCHES:
            for stride in STRIDES:
                physical = test14c["physical_records"][grid][branch][str(stride)]
                bulk = test14c["bulk_records"][grid][branch][str(stride)]
                base = test14b["balance_records"][grid][branch][str(stride)]
                seam_charge = np.asarray([
                    item["charge"]["equivalent_area_radius"]
                    * item["seam"]["geometric_intrinsic_integral"] / 4.0
                    for item in base
                ])
                seam_charge_rate = five_point_history_derivative(
                    seam_charge, times, stride=stride,
                )
                records = []
                for local_index, current_time in enumerate(times):
                    anisotropy_alternative = (
                        bulk[local_index]["geometric_bulk_rate"]
                        - 2.0 * bulk[local_index]["global_radius_part"]
                        + base[local_index]["rates"]["curvature_smooth"]
                    )
                    direct_seam = (
                        seam_charge_rate[local_index]
                        + bulk[local_index]["intrinsic_boundary_rate"]
                    )
                    records.append({
                        "time": float(current_time),
                        "anisotropy_primary": physical[local_index][
                            "intrinsic_anisotropy_rate"
                        ],
                        "anisotropy_alternative": float(
                            anisotropy_alternative
                        ),
                        "anisotropy_error": relative_scale_error(
                            physical[local_index]["intrinsic_anisotropy_rate"],
                            anisotropy_alternative,
                            max(abs(anisotropy_alternative), 1.0),
                        ),
                        "coupled_seam_primary": physical[local_index][
                            "coupled_seam"
                        ]["total"],
                        "coupled_seam_direct_history_plus_boundary": float(
                            direct_seam
                        ),
                        "coupled_seam_error": relative_scale_error(
                            physical[local_index]["coupled_seam"]["total"],
                            direct_seam, max(abs(direct_seam), 1.0),
                        ),
                    })
                graded = records[2: min(len(records) - 2, 28)]
                primary = [
                    item for item in records
                    if item["time"] <= PRIMARY_END + 1e-12
                ]
                primary_times = np.asarray([item["time"] for item in primary])
                primary_seam = float(np.trapezoid(
                    np.asarray([
                        item["coupled_seam_primary"] for item in primary
                    ]), primary_times,
                ))
                direct_seam = float(np.trapezoid(
                    np.asarray([
                        item["coupled_seam_direct_history_plus_boundary"]
                        for item in primary
                    ]), primary_times,
                ))
                arithmetic[grid][branch][str(stride)] = {
                    "records": records,
                    "maximum_graded_anisotropy_error": max(
                        item["anisotropy_error"] for item in graded
                    ),
                    "maximum_graded_coupled_seam_error": max(
                        item["coupled_seam_error"] for item in graded
                    ),
                    "primary_integrated_coupled_seam": primary_seam,
                    "primary_integrated_direct_seam": direct_seam,
                    "primary_integrated_coupled_seam_error": (
                        relative_scale_error(
                            primary_seam, direct_seam,
                            max(abs(primary_seam), abs(direct_seam), 1e-12),
                        )
                    ),
                }

    tangency_gate = bool(all(
        tangency[grid][branch][str(stride)]["passed_below_2_percent"]
        for grid in GRIDS for branch in BRANCHES for stride in STRIDES
    ))
    anisotropy_gate = bool(all(
        arithmetic[grid][branch][str(stride)][
            "maximum_graded_anisotropy_error"
        ] < 0.02
        for grid in GRIDS for branch in BRANCHES for stride in STRIDES
    ))
    seam_gate = bool(all(
        arithmetic[grid][branch][str(stride)][
            "primary_integrated_coupled_seam_error"
        ] < 0.02
        for grid in GRIDS for branch in BRANCHES for stride in STRIDES
    ))
    result = {
        "status": "complete",
        "protocol_sha256": sha256_file(PROTOCOL),
        "addendum_sha256": sha256_file(ADDENDUM),
        "inputs": expected,
        "tangency": tangency,
        "arithmetic": arithmetic,
        "existing_independent_einstein_audit": {
            "path": str(EINSTEIN14B),
            "sha256": sha256_file(EINSTEIN14B),
            "overall_grade": einstein.get("overall_grade"),
            "maximum_null_norm_error": einstein.get(
                "maximum_null_norm_error"
            ),
        },
        "gates": {
            "marginal_tangency_below_2_percent": tangency_gate,
            "alternative_anisotropy_below_2_percent": anisotropy_gate,
            "direct_coupled_seam_below_2_percent": seam_gate,
        },
        "passed": bool(tangency_gate and anisotropy_gate and seam_gate),
    }
    atomic_write_json(OUTPUT, result)
    print(json.dumps({
        "output": str(OUTPUT),
        "passed": result["passed"],
        "gates": result["gates"],
        "tangency_primary_relative": {
            grid: {
                branch: tangency[grid][branch]["1"][
                    "primary_relative_to_balance_norm"
                ] for branch in BRANCHES
            } for grid in GRIDS
        },
    }, indent=2))


if __name__ == "__main__":
    main()
