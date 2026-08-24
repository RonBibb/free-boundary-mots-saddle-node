#!/usr/bin/env python3
"""Independent artifact-only verifier for prospectively sealed Test 2D."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from bhps.recovery_indexer import atomic_write_json, sha256_file


PROTOCOL = Path("notes/110_A790_test2D_high_order_ragged_chart_protocol.md")
PROTOCOL_SHA256 = "f11989b23cff2d5b87bf6e730ff91c77b1e095eb2a7a49d1a277a9d3bf2666e5"
QUALIFICATION = Path("results/corrected_A790_test2d_high_order_ragged_chart_qualification_v3.json")
CHART_MANIFEST = Path("results/corrected_A790_test2d_high_order_ragged_chart_recovery/chart_index_v2.json")
FIELDS = Path("results/corrected_A790_test2d_high_order_ragged_chart_fields.json")
SURFACES = Path("results/corrected_A790_test2d_high_order_ragged_chart_surfaces.json")
COMMON_PARENT = Path("results/corrected_A790_test2d_high_order_ragged_chart_common_parent.json")
OUTPUT = Path("results/corrected_A790_test2d_high_order_ragged_chart_independent_audit.json")


def generalized_order(difference_12, difference_23, intervals=(112, 128, 144)):
    d12, d23 = float(difference_12), float(difference_23)
    n1, n2, n3 = map(float, intervals)
    if d12 <= 0.0 or d23 <= 0.0:
        return None
    ratio = d12 / d23
    def residual(order):
        return (n1**-order - n2**-order) / (n2**-order - n3**-order) - ratio
    if residual(1e-8) * residual(30.0) > 0.0:
        return None
    return float(brentq(residual, 1e-8, 30.0))


def order_interval(e12, u12, e23, u23, intervals=(112, 128, 144)):
    lower12, upper12 = max(float(e12) - float(u12), 0.0), float(e12) + float(u12)
    lower23, upper23 = max(float(e23) - float(u23), 0.0), float(e23) + float(u23)
    if lower12 <= 0.0 or lower23 <= 0.0:
        return None
    lower = generalized_order(lower12, upper23, intervals)
    upper = generalized_order(upper12, lower23, intervals)
    if lower is None or upper is None:
        return None
    return min(lower, upper), max(lower, upper)


def same_optional(first, second):
    return bool(
        first is None and second is None
        or first is not None and second is not None
        and np.allclose(first, second, rtol=1e-12, atol=1e-14)
    )


def verify_charts(manifest):
    records = {key: value for key, value in manifest["stages"].items() if key.startswith("chart/")}
    checks = {}
    for key, record in records.items():
        path = Path(record.get("output_path", ""))
        checks[key] = bool(
            record.get("status") == "complete" and path.is_file()
            and path.stat().st_size == record.get("byte_count")
            and sha256_file(path) == record.get("sha256")
            and record.get("completion_metadata", {}).get("validity", {}).get("valid")
        )
    return {"count": len(records), "checks": checks, "passed": len(records) == 90 and all(checks.values())}


def verify_field_sequence(record, temporal=False):
    keys = (
        ("G10_coarse_G10_standard", "G10_standard_G10_half")
        if temporal else ("G9_G10", "G10_G11")
    )
    pairs = [record["pairs"][key] for key in keys]
    agreements, passes = {}, []
    for norm, error_key, uncertainty_key, minimum_order in (
        ("L2", "absolute_L2", "L2_uncertainty", 1.5 if temporal else 1.0),
        ("q95", "weighted_q95", "q95_uncertainty", 1.5 if temporal else 1.0),
    ):
        errors = [float(item[error_key]) for item in pairs]
        uncertainties = [float(item[uncertainty_key]) for item in pairs]
        if temporal:
            recomputed = (
                math.log2((errors[0] - uncertainties[0]) / (errors[1] + uncertainties[1])),
                math.log2((errors[0] + uncertainties[0]) / (errors[1] - uncertainties[1])),
            ) if min(errors[0] - uncertainties[0], errors[1] - uncertainties[1]) > 0.0 else None
        else:
            recomputed = order_interval(errors[0], uncertainties[0], errors[1], uncertainties[1])
        stored = record["order_intervals"][norm]
        norm_pass = bool(
            errors[1] + uncertainties[1] < max(errors[0] - uncertainties[0], 0.0)
            and recomputed is not None and recomputed[0] > minimum_order
            and uncertainties[1] / max(errors[1], 1e-300) < 0.25
        )
        agreements[norm] = {
            "stored_order": stored, "recomputed_order": recomputed,
            "order_agrees": same_optional(stored, recomputed),
            "stored_pass": record["norm_passed"][norm], "recomputed_pass": norm_pass,
        }
        passes.append(norm_pass)
    recomputed_pass = bool(
        all(passes) and record["all_state_estimators_admissible"]
        and record["sign_coherence"] is not None and record["sign_coherence"] >= 0.70
        and record["component_growth_guard"]
        and (not temporal or record["fine_temporal_below_half_spatial"])
    )
    agrees = bool(
        all(item["order_agrees"] and item["stored_pass"] == item["recomputed_pass"] for item in agreements.values())
        and record["passed"] == recomputed_pass
    )
    return {"norms": agreements, "stored_pass": record["passed"],
            "recomputed_pass": recomputed_pass, "agrees": agrees}


def verify_fields(fields):
    spatial = {
        key: verify_field_sequence(value) for key, value in fields["spatial"].items()
    }
    temporal = {
        key: verify_field_sequence(value, temporal=True) for key, value in fields["temporal"].items()
    }
    spatial_pass = all(item["recomputed_pass"] for item in spatial.values())
    temporal_pass = all(item["recomputed_pass"] for item in temporal.values())
    if spatial_pass and temporal_pass:
        status = "PASS"
    else:
        metric, adm = fields["spatial"]["metric_increment"], fields["spatial"]["ADM_K"]
        status = "FAIL" if (
            not metric["strict_adverse_monotonicity"] and not adm["strict_adverse_monotonicity"]
            and not metric["map_limited"] and not adm["map_limited"]
            and not metric["interpolation_limited"] and not adm["interpolation_limited"]
        ) else "REVIEW"
    return {
        "spatial": spatial, "temporal": temporal,
        "stored_status": fields["status"], "recomputed_status": status,
        "passed": bool(
            all(item["agrees"] for item in (*spatial.values(), *temporal.values()))
            and fields["status"] == status
        ),
    }


def verify_tail(record):
    values, differences, uncertainties = record["values"], record["adjacent_differences"], record["uncertainties"]
    recomputed_differences = [abs(values[0] - values[1]), abs(values[1] - values[2])]
    recomputed_order = order_interval(
        recomputed_differences[0], uncertainties[0],
        recomputed_differences[1], uncertainties[1],
    )
    recomputed_pass = bool(
        recomputed_differences[1] + uncertainties[1]
        < max(recomputed_differences[0] - uncertainties[0], 0.0)
        and recomputed_order is not None and recomputed_order[0] > 1.0
        and uncertainties[1] / max(recomputed_differences[1], 1e-300) < 0.25
    )
    return bool(
        np.allclose(differences, recomputed_differences, rtol=1e-12, atol=1e-14)
        and same_optional(record["order_interval"], recomputed_order)
        and record["passed"] == recomputed_pass
    )


def verify_profile(record):
    agreements = []
    for norm in ("L2", "q95"):
        differences, uncertainties = record["differences"][norm], record["uncertainties"][norm]
        recomputed = order_interval(differences[0], uncertainties[0], differences[1], uncertainties[1])
        recomputed_pass = bool(
            differences[1] + uncertainties[1] < max(differences[0] - uncertainties[0], 0.0)
            and recomputed is not None and recomputed[0] > 1.0
            and uncertainties[1] / max(differences[1], 1e-300) < 0.25
        )
        agreements.append(bool(
            same_optional(record["order_intervals"][norm], recomputed)
            and record["norm_passed"][norm] == recomputed_pass
        ))
    recomputed_overall = bool(
        all(record["norm_passed"].values()) and record["interpolation_admissible"]
        and record["sign_coherence"] is not None and record["sign_coherence"] >= 0.70
    )
    return bool(all(agreements) and record["passed"] == recomputed_overall)


def verify_surface_coverage(record):
    points = int(record["profile_points"])
    collar = int(record["stability_collar_points"])
    limit = float(record["inverse_residual_limit"])
    recomputed = bool(
        record["fine_unique_points"] == points
        and record["primary_unique_points"] == points
        and record["fine_unique_stability_collar_points"] == collar
        and record["primary_unique_stability_collar_points"] == collar
        and record["fine_minimum_termination_margin"] >= 4.0
        and record["primary_minimum_termination_margin"] >= 4.0
        and record["fine_collar_minimum_termination_margin"] >= 4.0
        and record["primary_collar_minimum_termination_margin"] >= 4.0
        and record["fine_inverse_residual_maximum"] < limit
        and record["primary_inverse_residual_maximum"] < limit
        and record["fine_collar_inverse_residual_maximum"] < limit
        and record["primary_collar_inverse_residual_maximum"] < limit
    )
    return bool(record["passed"] == recomputed)


def verify_formation(record):
    histories = record["count_histories"]
    histories_valid = all(
        all(value in (0, 2) for value in values)
        and all(left <= right for left, right in zip(values, values[1:]))
        and 2 in values
        for values in histories.values()
    )
    brackets = record["proper_time_brackets"]
    spatial = [brackets[label] for label in ("G7", "G8", "G9", "G10", "G11")]
    spatial_overlap = max(item[0] for item in spatial) <= min(item[1] for item in spatial)
    temporal = [brackets[label] for label in ("G10_coarse", "G10_standard", "G10_half")]
    temporal_overlap = all(
        max(temporal[index][0], temporal[index + 1][0])
        <= min(temporal[index][1], temporal[index + 1][1])
        for index in range(2)
    )
    widths = [right - left for left, right in temporal]
    recomputed = bool(
        histories_valid and spatial_overlap and temporal_overlap
        and widths[2] < widths[1] < widths[0]
    )
    return bool(
        record["spatial_interval_overlap"] == spatial_overlap
        and record["temporal_adjacent_overlap"] == temporal_overlap
        and np.allclose(record["temporal_widths"], widths, rtol=0.0, atol=1e-15)
        and record["passed"] == recomputed
    )


def verify_surfaces(surfaces):
    checks = {}
    for branch, record in surfaces["branches"].items():
        if record.get("branch_loss"):
            checks[branch] = bool(not record["passed"])
            continue
        tail_checks = [verify_tail(item) for item in record["geometry"].values()]
        tail_checks.extend((verify_tail(record["stability"]), verify_tail(record["spectral_gap"])))
        profile_checks = [verify_profile(item) for item in record["profiles"].values()]
        coverage_checks = [
            verify_surface_coverage(item["coverage"])
            for item in record["records"].values()
        ]
        recomputed_coverage = all(item["coverage"]["passed"] for item in record["records"].values())
        recomputed_pass = bool(
            recomputed_coverage and all(item["passed"] for item in record["geometry"].values())
            and record["stability_gate"] and all(item["passed"] for item in record["profiles"].values())
        )
        checks[branch] = bool(
            all(tail_checks) and all(profile_checks) and all(coverage_checks)
            and record["coverage_passed"] == recomputed_coverage
            and record["passed"] == recomputed_pass
        )
    branch_loss = any(item.get("branch_loss", False) for item in surfaces["branches"].values())
    temporal_checks = [
        verify_surface_coverage(item)
        for state in surfaces["temporal_coverage"].values() for item in state.values()
    ]
    temporal_pass = all(
        item["passed"] for state in surfaces["temporal_coverage"].values() for item in state.values()
    )
    formation_check = verify_formation(surfaces["formation"])
    recomputed_overall = bool(
        all(item["passed"] for item in surfaces["branches"].values())
        and temporal_pass and surfaces["formation"]["passed"]
    )
    reversal = any(
        item.get("branch_loss") is False and any(
            value == ("outward_stable" if branch == "inner" else "outward_unstable")
            for value in item["stability_classifications"]
        )
        for branch, item in surfaces["branches"].items()
    )
    status = "PASS" if recomputed_overall else "FAIL" if branch_loss or reversal else "REVIEW"
    return {"branch_checks": checks, "stored_status": surfaces["status"],
            "recomputed_status": status,
            "passed": bool(
                all(checks.values()) and all(temporal_checks) and formation_check
                and surfaces["resolved_stability_reversal"] == reversal
                and surfaces["passed"] == recomputed_overall
                and status == surfaces["status"]
            )}


def verify_common_coverage(record):
    if not record.get("native_surface_valid", False):
        return bool(record.get("passed") is False)
    checks = []
    for item in record["charts"].values():
        recomputed = bool(
            item["unique_points"] == item["profile_points"]
            and item["unique_stability_collar_points"] == item["stability_collar_points"]
            and item["minimum_termination_margin"] >= 4.0
            and item["collar_minimum_termination_margin"] >= 4.0
            and item["maximum_inverse_residual"] < item["inverse_residual_limit"]
            and item["maximum_collar_inverse_residual"] < item["inverse_residual_limit"]
        )
        checks.append(item["passed"] == recomputed)
    recomputed_overall = bool(all(item["passed"] for item in record["charts"].values()))
    return bool(all(checks) and record["passed"] == recomputed_overall)


def verify_common_parent(common):
    checks = {}
    for grid, record in common["grids"].items():
        pair_checks = []
        for pair in record["pairs"].values():
            for observable in ("initial_metric", "metric_increment", "ADM_K"):
                item = pair[observable]
                recomputed = bool(item["relative_L2"] + item["relative_uncertainty"] < item["threshold"])
                pair_checks.append(recomputed == item["passed"])
        coverage_checks = [
            verify_common_coverage(item)
            for domain in record["branch_coverage"].values() for item in domain.values()
        ]
        recomputed_coverage = all(
            item["passed"]
            for domain in record["branch_coverage"].values() for item in domain.values()
        )
        recomputed_grid = bool(
            recomputed_coverage
            and all(pair[observable]["passed"] for pair in record["pairs"].values()
                    for observable in ("initial_metric", "metric_increment", "ADM_K"))
        )
        checks[grid] = bool(
            all(pair_checks) and all(coverage_checks)
            and record["coverage_passed"] == recomputed_coverage
            and recomputed_grid == record["passed"]
        )
    recomputed_status = "PASS" if all(item["passed"] for item in common["grids"].values()) else "REVIEW"
    return {"grid_checks": checks, "stored_status": common["status"],
            "recomputed_status": recomputed_status,
            "passed": all(checks.values()) and common["status"] == recomputed_status}


def main():
    if sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("sealed Test-2D protocol hash mismatch")
    qualification = json.loads(QUALIFICATION.read_text())
    charts = verify_charts(json.loads(CHART_MANIFEST.read_text()))
    fields = verify_fields(json.loads(FIELDS.read_text()))
    surfaces = verify_surfaces(json.loads(SURFACES.read_text()))
    common = verify_common_parent(json.loads(COMMON_PARENT.read_text()))
    verifier_pass = bool(
        qualification["status"] == "QUALIFIED" and charts["passed"]
        and fields["passed"] and surfaces["passed"] and common["passed"]
    )
    if not verifier_pass:
        status, classification = "REVIEW", "invalid_test2d_numerical_audit"
    elif fields["recomputed_status"] == "PASS" and surfaces["recomputed_status"] == "PASS" and common["recomputed_status"] == "PASS":
        status, classification = "PASS", "high_order_ragged_chart_above_first_order_continuum_evidence"
    elif fields["recomputed_status"] == "FAIL" or surfaces["recomputed_status"] == "FAIL":
        status, classification = "FAIL", "high_order_ragged_chart_nonconvergence_or_branch_failure"
    else:
        status, classification = "REVIEW", "high_order_ragged_chart_convergence_mixed"
    result = {
        "protocol": str(PROTOCOL), "protocol_sha256": PROTOCOL_SHA256,
        "status": status, "classification": classification,
        "verifier_passed": verifier_pass,
        "qualification": qualification["status"], "charts": charts,
        "fields": fields, "surfaces": surfaces, "common_parent": common,
        "source_hashes": {
            str(QUALIFICATION): sha256_file(QUALIFICATION),
            str(CHART_MANIFEST): sha256_file(CHART_MANIFEST),
            str(FIELDS): sha256_file(FIELDS), str(SURFACES): sha256_file(SURFACES),
            str(COMMON_PARENT): sha256_file(COMMON_PARENT),
        },
    }
    atomic_write_json(OUTPUT, result)
    print(json.dumps({"status": status, "classification": classification,
                      "verifier_passed": verifier_pass}, indent=2))


if __name__ == "__main__":
    main()
