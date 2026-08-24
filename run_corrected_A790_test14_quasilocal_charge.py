#!/usr/bin/env python3
"""Indexed Test-14 generalized-Hawking-AdS charge-history audit."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice
from bhps.dynamical_capped_horizon_bvp import solve_dynamical_capped_surface_bvp
from bhps.recovery_indexer import (
    RecoveryIndex,
    atomic_write_json,
    sha256_file,
)
from bhps.test14_quasilocal_charge import (
    analytic_controls,
    reflected_cap_charge,
    relative_difference,
)
from run_corrected_A790_blind_horizon_detector_engineering import (
    public_surface,
    solve_seed,
)
from run_corrected_A790_independent_dynamic_BVP_detector import (
    admitted as bvp_admitted,
    public_surface as public_bvp_surface,
)
from run_corrected_A790_surface_geometry_history import (
    archived_slice,
    selections_for,
)
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry


PROTOCOL = Path("notes/95_A790_quasilocal_mass_flux_bridge_protocol.md")
OUTPUT = Path("results/corrected_A790_test14_quasilocal_charge_history.json")
MANIFEST = Path("results/corrected_A790_test14_recovery_v2.json")
STAGES = Path("results/corrected_A790_test14_stages")
FINE_RESULT = Path("results/corrected_A790_formation_time_refinement.json")
FINE_STATE = Path("results/corrected_A790_formation_time_refinement_state.npz")
LONG_RESULT = Path("results/corrected_A790_two_grid_formation_search.json")
LONG_STATE = Path("results/corrected_A790_two_grid_formation_search_state.npz")
FINAL_RESULT = Path("results/corrected_A790_t004_discrepancy_formation_confirmation.json")
NOTE85_RESULT = Path("results/corrected_A790_t008_long_evolution.json")
NOTE85_STATE = Path("results/corrected_A790_t008_long_evolution_state.npz")
PRIMARY_TIMES = (0.000625, 0.001, 0.002, 0.003, 0.004)
SECONDARY_TIMES = (0.005, 0.006, 0.007, 0.008)
AMPLITUDE = 7.90
CHARGE_KEY = "generalized_hawking_ads_charge_kappa5_squared_E"


def stage_json(index, stage_id, path, compute, metadata, maximum_seconds=3600.0):
    index.register(
        stage_id, "test14-quasilocal-charge", maximum_seconds, metadata,
    )
    validated = index.validated_path(stage_id)
    if validated is not None:
        return json.loads(validated.read_text())
    index.mark_running(stage_id)
    started = time.perf_counter()
    try:
        payload = compute()
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
    with tempfile.TemporaryDirectory(prefix="bhps-test14-recovery-") as directory:
        root = Path(directory)
        source = root / "source.txt"
        source.write_text("fixed input\n")
        expected = {str(source): sha256_file(source)}
        manifest = root / "recovery.json"
        stage = root / "stage.json"
        index = RecoveryIndex(manifest, PROTOCOL, expected, 60.0)
        index.register("smoke", "test14-smoke", 10.0, {"index": 0})
        index.mark_running("smoke")
        atomic_write_json(stage, {"finite": True, "value": 1})
        index.mark_complete("smoke", stage, 0.01)
        resumed = RecoveryIndex(manifest, PROTOCOL, expected, 60.0)
        valid_restart = resumed.validated_path("smoke") == stage
        atomic_write_json(stage, {"finite": True, "value": 2})
        corruption_rejected = resumed.validated_path("smoke") is None
        return {
            "valid_completed_stage_resumed": bool(valid_restart),
            "corrupted_stage_rejected": bool(corruption_rejected),
            "passed": bool(valid_restart and corruption_rejected),
        }


def build_geometries():
    fold = build_geometry("G6")
    seed = {**fold, "fold_amplitude": AMPLITUDE}
    geometries = {
        "G7": build_refined(
            seed, 81, 121, "G7A790-test14", selector_iterations=40,
            slice_iterations=270,
        ),
    }
    geometries["G8"] = build_refined(
        geometries["G7"], 97, 145, "G8A790-test14",
        selector_iterations=45, slice_iterations=280,
    )
    return geometries


def primary_stage(
    label, current_time, branch_name, representation, geometry, position,
    velocity, selection, prepared,
):
    modes = selection[
        "selected_modes" if representation == "selected" else "confirmation_modes"
    ]
    surface = solve_seed(
        position, velocity, geometry, selection["seed"], modes, prepared,
    )
    if "error" in surface or not surface.get("converged"):
        raise RuntimeError(f"surface solve failed: {surface}")
    charge = reflected_cap_charge(
        position, velocity, geometry["z"], geometry["r"], surface,
        prepared=prepared,
    )
    return {
        "grid": label,
        "time": float(current_time),
        "branch": branch_name,
        "representation": representation,
        "cosine_modes": int(modes),
        "seed": float(selection["seed"]),
        "archived_signature": selection["archived_signature"],
        "surface": public_surface(surface),
        "charge": charge,
        "finite": bool(charge["finite"]),
    }


def secondary_stage(
    label, current_time, branch_name, geometry, position, velocity, seed,
):
    prepared = prepare_capped_expansion_slice(
        position, velocity, geometry["z"], geometry["r"],
    )
    started = time.perf_counter()
    surface = solve_dynamical_capped_surface_bvp(
        position, velocity, geometry["z"], geometry["r"], seed,
        tolerance=2e-5, nodes=121, maximum_nodes=6000,
        dense_nodes=501, prepared=prepared,
    )
    surface["runtime_seconds"] = float(time.perf_counter() - started)
    if not bvp_admitted(surface):
        raise RuntimeError(f"secondary BVP surface not admitted: {surface}")
    charge = reflected_cap_charge(
        position, velocity, geometry["z"], geometry["r"], surface,
        prepared=prepared,
    )
    return {
        "grid": label,
        "time": float(current_time),
        "branch": branch_name,
        "representation": "note85_BVP_reconstruction",
        "seed": float(seed),
        "surface": public_bvp_surface(surface),
        "charge": charge,
        "finite": bool(charge["finite"]),
        "secondary_note85_review_scope": True,
    }


def charge(record):
    return float(record["charge"][CHARGE_KEY])


def confirmation_record(records, label, current_time, branch):
    return next(
        record for record in records
        if record["grid"] == label
        and math.isclose(record["time"], current_time, abs_tol=1e-12)
        and record["branch"] == branch
        and record["representation"] == "confirmation"
    )


def summarize_primary(records):
    leaf_gate = bool(all(
        record["charge"]["finite"]
        and record["charge"]["reflection_doubled_area"] > 0.0
        and record["charge"][CHARGE_KEY] > 0.0
        and record["charge"]["minimum_curve_speed"] > 0.0
        and record["charge"]["minimum_transverse_metric"] > 0.0
        and record["charge"]["axis_regularity_defect"] < 0.05
        and np.isfinite(record["charge"][
            "distributional_seam_curvature_integral"
        ])
        for record in records
    ))
    representation_transfer = []
    cross_grid_transfer = []
    for current_time in PRIMARY_TIMES:
        for label in ("G7", "G8"):
            for branch in ("inner", "outer"):
                selected = next(
                    record for record in records
                    if record["grid"] == label
                    and math.isclose(record["time"], current_time, abs_tol=1e-12)
                    and record["branch"] == branch
                    and record["representation"] == "selected"
                )
                confirmation = confirmation_record(
                    records, label, current_time, branch,
                )
                representation_transfer.append({
                    "grid": label,
                    "time": current_time,
                    "branch": branch,
                    "relative_difference": relative_difference(
                        charge(selected), charge(confirmation),
                    ),
                })
        for branch in ("inner", "outer"):
            left = confirmation_record(records, "G7", current_time, branch)
            right = confirmation_record(records, "G8", current_time, branch)
            cross_grid_transfer.append({
                "time": current_time,
                "branch": branch,
                "relative_difference": relative_difference(
                    charge(left), charge(right),
                ),
            })

    histories = {}
    interval_rates = {}
    for label in ("G7", "G8"):
        histories[label] = {}
        interval_rates[label] = {}
        for branch in ("inner", "outer"):
            values = np.asarray([
                charge(confirmation_record(records, label, value, branch))
                for value in PRIMARY_TIMES
            ])
            times = np.asarray(PRIMARY_TIMES)
            rates = np.diff(values) / np.diff(times)
            histories[label][branch] = [
                {"time": float(t), "charge": float(q)}
                for t, q in zip(times, values)
            ]
            interval_rates[label][branch] = [
                {
                    "left_time": float(times[index_value]),
                    "right_time": float(times[index_value + 1]),
                    "rate": float(rates[index_value]),
                }
                for index_value in range(len(rates))
            ]

    rate_transfer = []
    signs = []
    for branch in ("inner", "outer"):
        for index_value in range(len(PRIMARY_TIMES) - 1):
            left = interval_rates["G7"][branch][index_value]["rate"]
            right = interval_rates["G8"][branch][index_value]["rate"]
            mean_charge = max(
                abs(histories["G7"][branch][index_value]["charge"]),
                abs(histories["G8"][branch][index_value]["charge"]),
                1e-300,
            )
            relative = relative_difference(left, right)
            scale_normalized_absolute = abs(left - right) / mean_charge
            same_nonzero_sign = bool(left * right > 0.0)
            signs.extend((np.sign(left), np.sign(right)))
            rate_transfer.append({
                "branch": branch,
                "left_time": PRIMARY_TIMES[index_value],
                "right_time": PRIMARY_TIMES[index_value + 1],
                "G7_rate": left,
                "G8_rate": right,
                "same_nonzero_sign": same_nonzero_sign,
                "relative_difference": relative,
                "absolute_difference_normalized_by_charge": (
                    scale_normalized_absolute
                ),
                "transfer_gate": bool(
                    same_nonzero_sign
                    and (relative < 0.15 or scale_normalized_absolute < 0.02)
                ),
            })
    branch_sign_consistency = bool(all(
        all(
            np.sign(item["rate"]) == np.sign(values[0]["rate"])
            and item["rate"] != 0.0
            for item in values
        )
        for grid in interval_rates.values() for values in grid.values()
    ))
    gates = {
        "all_leaf_geometry_and_finiteness_rules_pass": leaf_gate,
        "all_selected_confirmation_charge_differences_below_0_5_percent": bool(
            all(item["relative_difference"] < 0.005
                for item in representation_transfer)
        ),
        "all_cross_grid_charge_differences_below_2_percent": bool(
            all(item["relative_difference"] < 0.02
                for item in cross_grid_transfer)
        ),
        "resolved_monotonic_rate_sign_gate": bool(
            branch_sign_consistency
            and all(item["transfer_gate"] for item in rate_transfer)
        ),
    }
    return {
        "gates": gates,
        "representation_transfer": representation_transfer,
        "cross_grid_transfer": cross_grid_transfer,
        "histories": histories,
        "interval_charge_rates": interval_rates,
        "rate_transfer": rate_transfer,
    }


def main():
    STAGES.mkdir(parents=True, exist_ok=True)
    required = (
        PROTOCOL, FINE_RESULT, FINE_STATE, LONG_RESULT, LONG_STATE,
        FINAL_RESULT, NOTE85_RESULT, NOTE85_STATE,
        Path("src/bhps/test14_quasilocal_charge.py"),
        Path("src/bhps/dynamical_capped_horizon.py"),
        Path("src/bhps/dynamical_capped_horizon_bvp.py"),
        Path("src/bhps/recovery_indexer.py"), Path(__file__),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing Test-14 inputs: {missing}")
    expected = {str(path): sha256_file(path) for path in required}
    recovery_control = recovery_smoke_test()
    index = RecoveryIndex(MANIFEST, PROTOCOL, expected, 3600.0)
    controls = stage_json(
        index, "controls/analytic", STAGES / "analytic_controls.json",
        analytic_controls, {"sealed_gates": [1, 2, 3]}, 60.0,
    )
    if not controls["passed"] or not recovery_control["passed"]:
        raise RuntimeError("Test-14 analytic or recovery control failed")

    print("building corrected G7/G8 A=7.90 Test-14 geometries", flush=True)
    geometries = build_geometries()
    fine_result = json.loads(FINE_RESULT.read_text())
    long_result = json.loads(LONG_RESULT.read_text())
    final_result = json.loads(FINAL_RESULT.read_text())
    fine_state = np.load(FINE_STATE)
    long_state = np.load(LONG_STATE)
    records = []
    for current_time in PRIMARY_TIMES:
        for label, geometry in geometries.items():
            position, velocity = archived_slice(
                label, current_time, geometry, fine_state, long_state,
            )
            prepared = prepare_capped_expansion_slice(
                position, velocity, geometry["z"], geometry["r"],
            )
            selections = selections_for(
                label, current_time, fine_result, long_result, final_result,
            )
            for branch_index, branch_name in enumerate(("inner", "outer")):
                selection = selections[branch_index]
                for representation in ("selected", "confirmation"):
                    modes = selection[
                        "selected_modes" if representation == "selected"
                        else "confirmation_modes"
                    ]
                    token = f"{current_time:.6f}".replace(".", "p")
                    path = STAGES / (
                        f"primary_{label}_{token}_{branch_name}_{representation}.json"
                    )
                    stage_id = (
                        f"primary/{label}/{token}/{branch_name}/{representation}"
                    )
                    print(
                        f"{stage_id}: {modes} spectral modes", flush=True,
                    )
                    records.append(stage_json(
                        index, stage_id, path,
                        lambda label=label, current_time=current_time,
                        branch_name=branch_name,
                        representation=representation, geometry=geometry,
                        position=position, velocity=velocity,
                        selection=selection, prepared=prepared: primary_stage(
                            label, current_time, branch_name, representation,
                            geometry, position, velocity, selection, prepared,
                        ),
                        {
                            "grid": label, "time": current_time,
                            "branch": branch_name,
                            "representation": representation,
                            "cosine_modes": int(modes),
                        },
                    ))

    primary = summarize_primary(records)

    note85 = json.loads(NOTE85_RESULT.read_text())
    with np.load(NOTE85_STATE) as note85_state:
        secondary = []
        for current_time in SECONDARY_TIMES:
            state_index = int(round(current_time / 0.000125))
            history_index = list(note85["surface_times"]).index(current_time)
            for label, geometry in geometries.items():
                position = np.asarray(
                    note85_state[f"{label}_position_history"][state_index]
                )
                velocity = np.asarray(
                    note85_state[f"{label}_velocity_history"][state_index]
                )
                history = note85["surface_history"][label][history_index]
                for branch_index, branch_name in enumerate(("inner", "outer")):
                    seed = history["representatives"][branch_index]["seed"]
                    token = f"{current_time:.6f}".replace(".", "p")
                    path = STAGES / (
                        f"secondary_{label}_{token}_{branch_name}.json"
                    )
                    stage_id = f"secondary/{label}/{token}/{branch_name}"
                    print(f"{stage_id}: BVP seed {seed:.6g}", flush=True)
                    secondary.append(stage_json(
                        index, stage_id, path,
                        lambda label=label, current_time=current_time,
                        branch_name=branch_name, geometry=geometry,
                        position=position, velocity=velocity, seed=seed:
                        secondary_stage(
                            label, current_time, branch_name, geometry,
                            position, velocity, seed,
                        ),
                        {
                            "grid": label, "time": current_time,
                            "branch": branch_name, "seed": float(seed),
                            "scope": "secondary_note85_review",
                        },
                    ))

    charge_gates_pass = bool(
        controls["passed"]
        and recovery_control["passed"]
        and primary["gates"][
            "all_leaf_geometry_and_finiteness_rules_pass"
        ]
        and primary["gates"][
            "all_selected_confirmation_charge_differences_below_0_5_percent"
        ]
        and primary["gates"][
            "all_cross_grid_charge_differences_below_2_percent"
        ]
    )
    rate_sign_resolved = bool(
        primary["gates"]["resolved_monotonic_rate_sign_gate"]
    )
    flux_closure = {
        "status": "REVIEW",
        "closed": False,
        "reason": (
            "Archived sparse leaves do not independently supply the full "
            "non-Einstein curvature-redistribution, horizon-evolution-field, "
            "and Israel brane/seam flux terms. Missing terms are not zeroed."
        ),
        "balance_claim": "mass-transfer balance not established",
    }
    payload = {
        "status": "REVIEW",
        "classification": (
            "resolved_conditional_generalized_Hawking_AdS_charge_history_"
            "with_mass_transfer_balance_not_established"
            if charge_gates_pass else
            "unresolved_generalized_Hawking_AdS_charge_history"
        ),
        "protocol": str(PROTOCOL),
        "protocol_sha256": sha256_file(PROTOCOL),
        "definition": {
            "primary": (
                "Q=kappa5^2 E_H,AdS=R_A[I_R+6A]/4 on a MOTS"
            ),
            "units": "ell=kappa5=1; Q has dimensions length^2",
            "closure": (
                "Z2 reflection double with distributional brane-seam "
                "intrinsic-curvature completion"
            ),
            "not_area_only": True,
        },
        "analytic_controls": controls,
        "recovery_control": recovery_control,
        "primary_records": records,
        "primary_summary": primary,
        "secondary_note85_review_records": secondary,
        "charge_history_subgrade": "PASS" if charge_gates_pass else "REVIEW",
        "charge_rate_sign_subgrade": "PASS" if rate_sign_resolved else "REVIEW",
        "flux_closure": flux_closure,
        "overall_grade": "REVIEW",
        "claim_boundary": [
            "A passing charge subgrade is conditional on the Z2 distributional seam completion.",
            "Charge rates on nested branches are not exchange between independent reservoirs.",
            "No physical mass-transfer claim is made without the full balance-law closure.",
            "The t=0.005--0.008 extension inherits note 85's evolution-review scope.",
            "No event-horizon, topology, throat, halo, or astrophysical-mass claim is made.",
        ],
        "provenance": {
            "manifest": str(MANIFEST),
            "input_sha256": expected,
        },
    }
    atomic_write_json(OUTPUT, payload)
    print(json.dumps({
        "status": payload["status"],
        "classification": payload["classification"],
        "charge_history_subgrade": payload["charge_history_subgrade"],
        "charge_rate_sign_subgrade": payload["charge_rate_sign_subgrade"],
        "flux_closure": flux_closure,
        "primary_gates": primary["gates"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
