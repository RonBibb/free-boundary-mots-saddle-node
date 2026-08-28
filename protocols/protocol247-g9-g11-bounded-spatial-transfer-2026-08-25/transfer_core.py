"""Pure comparison rules for Protocol 247."""

from __future__ import annotations

import math


GEOMETRY_FIELDS = (
    "one_sided_cap_area",
    "equivalent_area_radius",
    "proper_meridional_length",
    "rho_axis",
    "rho_brane",
)


def symmetric_relative(left, right):
    left = float(left)
    right = float(right)
    if not (math.isfinite(left) and math.isfinite(right)):
        raise ValueError("comparison values must be finite")
    return abs(left - right) / max(abs(left), abs(right), 1e-300)


def adjacent_leaf_transfer(left, right):
    geometry = {
        name: symmetric_relative(left["geometry"][name], right["geometry"][name])
        for name in GEOMETRY_FIELDS
    }
    left_eigenvalue = float(left["stability"]["fine_principal_eigenvalue"])
    right_eigenvalue = float(right["stability"]["fine_principal_eigenvalue"])
    eigenvalue_absolute = abs(left_eigenvalue - right_eigenvalue)
    eigenvalue_relative = symmetric_relative(left_eigenvalue, right_eigenvalue)
    geometry_pass = max(geometry.values()) < 0.01
    stability_pass = bool(
        left["stability"]["classification"] == right["stability"]["classification"]
        and left["stability"]["classification"] == "outward-stable"
        and (eigenvalue_relative < 0.10 or eigenvalue_absolute < 0.02)
    )
    return {
        "geometry_relative_differences": geometry,
        "maximum_geometry_relative_difference": max(geometry.values()),
        "geometry_pass_below_1_percent": geometry_pass,
        "principal_eigenvalue_absolute_difference": eigenvalue_absolute,
        "principal_eigenvalue_relative_difference": eigenvalue_relative,
        "stability_classification_agrees": (
            left["stability"]["classification"] == right["stability"]["classification"]
        ),
        "stability_pass_below_10_percent_or_0p02_absolute": stability_pass,
        "passed": bool(geometry_pass and stability_pass),
    }


def classify(local_pass, transfer_pass, terminal_pass, causal_pass):
    gates = {
        "all_local_grid_gates_pass": bool(local_pass),
        "all_terminal_controls_pass": bool(terminal_pass),
        "all_adjacent_geometry_and_stability_transfers_pass": bool(transfer_pass),
        "all_three_grids_have_resolved_spacelike_interior_signatures": bool(causal_pass),
    }
    if not gates["all_local_grid_gates_pass"]:
        label = "BOUNDED-SPATIAL-TRANSFER-LOCAL-GRID-FAIL"
    elif not gates["all_terminal_controls_pass"]:
        label = "BOUNDED-SPATIAL-TRANSFER-CONTROL-FAIL"
    elif not gates["all_adjacent_geometry_and_stability_transfers_pass"]:
        label = "BOUNDED-SPATIAL-TRANSFER-GEOMETRY-OR-STABILITY-FAIL"
    elif not gates["all_three_grids_have_resolved_spacelike_interior_signatures"]:
        label = "BOUNDED-SPATIAL-TRANSFER-CAUSAL-SIGNATURE-FAIL"
    else:
        label = "G9-G10-G11-BOUNDED-OUTER-TUBE-SPATIAL-TRANSFER-PASS"
    return label, gates
