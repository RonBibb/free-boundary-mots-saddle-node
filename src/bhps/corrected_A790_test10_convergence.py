"""Analysis primitives for the sealed A=7.90 Test-10 convergence audit."""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import PchipInterpolator

from bhps.corrected_A790_physical_tensor_convergence import (
    adm_extrinsic_curvature_tensor,
    generalized_order_nonuniform,
    interpolate_tensor,
    physical_tensor_difference,
    spatial_metric_tensor,
)


def uniform_time_order(coarse_medium, medium_fine):
    """Return the order of a 2:1:1/2 uniform time-step sequence."""
    left = float(coarse_medium)
    right = float(medium_fine)
    if left <= 0.0 or right <= 0.0:
        return None
    return float(math.log(left / right, 2.0))


def three_grid_sequence(fields, metrics, z, r, counts):
    """Physical differences and nonuniform order for three tensor fields."""
    names = tuple(fields)
    if len(names) != 3 or tuple(metrics) != names:
        raise ValueError("three aligned tensor fields are required")
    pair_names = (f"{names[0]}_{names[1]}", f"{names[1]}_{names[2]}")
    records = {}
    differences = []
    for pair_name, left, right in zip(pair_names, names[:-1], names[1:]):
        record = physical_tensor_difference(
            fields[left], fields[right], metrics[left], metrics[right], z, r,
        )
        records[pair_name] = record
        differences.append(record["absolute_difference"])
    return {
        "pairs": records,
        "absolute_differences": differences,
        "strictly_decreasing": bool(differences[1] < differences[0]),
        "order": generalized_order_nonuniform(
            differences[0], differences[1], counts,
        ),
    }


def temporal_tensor_sequence(coarse, medium, fine, metrics, z, r):
    """Physical 4/8/16-step difference sequence on a common grid."""
    first = physical_tensor_difference(
        coarse, medium, metrics["coarse"], metrics["medium"], z, r,
    )
    second = physical_tensor_difference(
        medium, fine, metrics["medium"], metrics["fine"], z, r,
    )
    return {
        "coarse_medium": first,
        "medium_fine": second,
        "strictly_decreasing": bool(
            second["absolute_difference"] < first["absolute_difference"]
        ),
        "order": uniform_time_order(
            first["absolute_difference"], second["absolute_difference"],
        ),
    }


def tensor_fields_on_grid(position, velocity, geometry, target_z, target_r):
    """Return final metric, metric increment, and ADM K on a target grid."""
    initial = np.asarray(geometry["jet_field"].reduced_fields)
    final_metric_native = spatial_metric_tensor(position, geometry["r"])
    initial_metric_native = spatial_metric_tensor(initial, geometry["r"])
    k_native = adm_extrinsic_curvature_tensor(
        position, velocity, geometry["z"], geometry["r"],
    )
    return {
        "initial_metric": interpolate_tensor(
            initial_metric_native, geometry["z"], geometry["r"],
            target_z, target_r,
        ),
        "final_metric": interpolate_tensor(
            final_metric_native, geometry["z"], geometry["r"],
            target_z, target_r,
        ),
        "metric_increment": interpolate_tensor(
            final_metric_native - initial_metric_native,
            geometry["z"], geometry["r"], target_z, target_r,
        ),
        "ADM_K": interpolate_tensor(
            k_native, geometry["z"], geometry["r"], target_z, target_r,
        ),
    }


def _proper_line_distance(coordinate, metric, endpoint):
    coordinate = np.asarray(coordinate, dtype=float)
    metric = np.asarray(metric, dtype=float)
    endpoint = float(endpoint)
    if coordinate.ndim != 1 or metric.shape != coordinate.shape:
        raise ValueError("line coordinate and metric must be aligned vectors")
    if np.any(np.diff(coordinate) <= 0.0):
        raise ValueError("line coordinate must be strictly increasing")
    if np.any(metric <= 0.0) or not np.all(np.isfinite(metric)):
        raise ValueError("line metric must be finite and positive")
    if endpoint < coordinate[0] or endpoint > coordinate[-1]:
        raise ValueError("proper-distance endpoint leaves line domain")
    inside = coordinate < endpoint
    nodes = np.concatenate((coordinate[inside], np.asarray([endpoint])))
    values = PchipInterpolator(coordinate, np.sqrt(metric))(nodes)
    return float(simpson(values, x=nodes)) if len(nodes) >= 3 else float(
        np.trapezoid(values, x=nodes)
    )


def proper_endpoint_distances(position, z, r, rho_axis, rho_brane):
    """Intrinsic line distances corresponding to the two cap endpoints."""
    position = np.asarray(position, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    if position.shape != (len(z), len(r), 9):
        raise ValueError("invalid reduced metric state")
    compact_endpoint = float(z[-1] - float(rho_axis))
    compact_from_axis_endpoint_to_brane = _proper_line_distance(
        z, position[:, 0, 6], z[-1],
    ) - _proper_line_distance(z, position[:, 0, 6], compact_endpoint)
    radial_metric = position[-1, :, 3] + r**2 * position[-1, :, 4]
    radial_from_axis_to_brane_endpoint = _proper_line_distance(
        r, radial_metric, float(rho_brane),
    )
    return {
        "compact_axis_endpoint_to_brane": compact_from_axis_endpoint_to_brane,
        "radial_axis_to_brane_endpoint": radial_from_axis_to_brane_endpoint,
    }


def domain_initial_dominance(initial_difference, increment_difference, final_difference):
    """Apply the sealed 25% initial-data-dominance rule."""
    initial = float(initial_difference)
    increment = float(increment_difference)
    final = float(final_difference)
    if initial <= 0.0:
        raise ValueError("initial difference must be positive")
    increment_ratio = increment / initial
    final_change_ratio = abs(final - initial) / initial
    return {
        "increment_to_initial_ratio": float(increment_ratio),
        "final_change_from_initial_ratio": float(final_change_ratio),
        "initial_data_dominated": bool(
            increment_ratio < 0.25 and final_change_ratio < 0.25
        ),
    }


def relative_difference(left, right):
    return float(abs(left - right) / max(abs(left), abs(right), 1e-300))

