"""Common-parent restriction and analysis primitives for sealed Test 10B."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson

from bhps.corrected_A790_physical_tensor_convergence import (
    adm_extrinsic_curvature_tensor,
    physical_tensor_difference,
    spatial_metric_tensor,
)
from bhps.corrected_A790_test10_convergence import proper_endpoint_distances


@dataclass(frozen=True)
class RestrictedJetField:
    """Exact radial prefix of one fully constructed parent jet field."""

    z: np.ndarray
    r: np.ndarray
    reduced_fields: np.ndarray
    reduced_first: np.ndarray
    reduced_second: np.ndarray

    @classmethod
    def from_parent(cls, parent, radial_count):
        count = int(radial_count)
        if count < 8 or count > len(parent.r):
            raise ValueError("invalid radial restriction count")
        return cls(
            z=np.asarray(parent.z).copy(),
            r=np.asarray(parent.r)[:count].copy(),
            reduced_fields=np.asarray(parent.reduced_fields)[:, :count].copy(),
            reduced_first=np.asarray(parent.reduced_first)[:, :, :count].copy(),
            reduced_second=np.asarray(parent.reduced_second)[:, :, :, :count].copy(),
        )


def exact_radial_index(radius, endpoint, tolerance=1e-12):
    radius = np.asarray(radius, dtype=float)
    matches = np.flatnonzero(np.isclose(
        radius, float(endpoint), rtol=0.0, atol=float(tolerance),
    ))
    if len(matches) != 1:
        raise ValueError("restriction endpoint is not one exact parent node")
    return int(matches[0])


def restrict_geometry(parent, endpoint, name):
    """Return a bitwise field/jet restriction of an R12 parent geometry."""
    index = exact_radial_index(parent["r"], endpoint)
    count = index + 1
    jet = RestrictedJetField.from_parent(parent["jet_field"], count)
    result = {
        "name": str(name),
        "source_grid": [len(parent["z"]), count],
        "fold_amplitude": float(parent["fold_amplitude"]),
        "radial_domain": [float(parent["r"][0]), float(endpoint)],
        "selector_maximum": float(parent["selector_maximum"]),
        "reference_maximum_residual": float(parent["reference_maximum_residual"]),
        "z": np.asarray(parent["z"]).copy(),
        "r": np.asarray(parent["r"])[:count].copy(),
        "background": parent["background"],
        "mass_squared": float(parent["mass_squared"]),
        "jet_field": jet,
        "common_parent_name": str(parent["name"]),
        "restriction_parent_radial_count": int(len(parent["r"])),
        "restriction_endpoint_index": index,
    }
    for key in ("psi", "a", "b", "c", "phi"):
        result[key] = np.asarray(parent[key])[:, :count].copy()
    result["principal"] = {
        key: np.asarray(value)[:, :count].copy()
        for key, value in parent["principal"].items()
    }
    return result


def restriction_identity(parent, restricted):
    """Audit exact values and archived jets, including the cut node."""
    count = len(restricted["r"])
    records = {}
    for key in ("z", "r"):
        expected = np.asarray(parent[key]) if key == "z" else np.asarray(parent[key])[:count]
        found = np.asarray(restricted[key])
        records[key] = {
            "array_equal": bool(np.array_equal(expected, found)),
            "maximum_absolute_difference": float(np.max(np.abs(expected - found))),
        }
    for key in ("psi", "a", "b", "c", "phi"):
        expected = np.asarray(parent[key])[:, :count]
        found = np.asarray(restricted[key])
        records[key] = {
            "array_equal": bool(np.array_equal(expected, found)),
            "maximum_absolute_difference": float(np.max(np.abs(expected - found))),
        }
    parent_jet = parent["jet_field"]
    restricted_jet = restricted["jet_field"]
    for key, expected, found in (
        ("reduced_fields", parent_jet.reduced_fields[:, :count], restricted_jet.reduced_fields),
        ("reduced_first", parent_jet.reduced_first[:, :, :count], restricted_jet.reduced_first),
        (
            "reduced_second", parent_jet.reduced_second[:, :, :, :count],
            restricted_jet.reduced_second,
        ),
    ):
        expected = np.asarray(expected)
        found = np.asarray(found)
        records[key] = {
            "array_equal": bool(np.array_equal(expected, found)),
            "maximum_absolute_difference": float(np.max(np.abs(expected - found))),
        }
    cut = count - 1
    records["cut_first_jet_maximum_absolute_difference"] = float(np.max(np.abs(
        parent_jet.reduced_first[:, :, cut] - restricted_jet.reduced_first[:, :, cut]
    )))
    records["cut_second_jet_maximum_absolute_difference"] = float(np.max(np.abs(
        parent_jet.reduced_second[:, :, :, cut]
        - restricted_jet.reduced_second[:, :, :, cut]
    )))
    records["passed"] = bool(all(
        record["array_equal"]
        for record in records.values() if isinstance(record, dict) and "array_equal" in record
    ) and records["cut_first_jet_maximum_absolute_difference"] == 0.0
      and records["cut_second_jet_maximum_absolute_difference"] == 0.0)
    return records


def _proper_volume_weight(position, radius):
    metric = spatial_metric_tensor(position, radius)
    base_determinant = (
        metric[..., 0, 0] * metric[..., 1, 1] - metric[..., 0, 1] ** 2
    )
    transverse = 0.5 * (metric[..., 2, 2] + metric[..., 3, 3])
    if np.any(base_determinant <= 0.0) or np.any(transverse <= 0.0):
        raise ValueError("common-radius spatial metric is not positive")
    return (
        4.0 * math.pi * np.asarray(radius)[None, :] ** 2
        * transverse * np.sqrt(base_determinant)
    )


def _integral(field, weight, z, r):
    radial = simpson(np.asarray(field) * weight, x=r, axis=1)
    return float(simpson(radial, x=z))


def common_radius_invariants(position, z, r, radius_cut=6.0):
    """Invariant common-cylinder integrals and fixed proper line distances."""
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    keep = r <= float(radius_cut) + 1e-12
    local_r = r[keep]
    local = np.asarray(position)[:, keep]
    weight = _proper_volume_weight(local, local_r)
    phi = local[..., 7]
    chi = local[..., 8]
    volume = _integral(np.ones_like(phi), weight, z, local_r)
    distances = proper_endpoint_distances(
        local, z, local_r, rho_axis=0.5, rho_brane=2.0,
    )
    return {
        "proper_four_volume": volume,
        "Phi_moment": _integral(phi, weight, z, local_r),
        "chi_moment": _integral(chi, weight, z, local_r),
        "Phi_squared_moment": _integral(phi**2, weight, z, local_r),
        "chi_squared_moment": _integral(chi**2, weight, z, local_r),
        **distances,
    }


def relative_difference(left, right):
    return float(abs(float(left) - float(right)) / max(
        abs(float(left)), abs(float(right)), 1e-300,
    ))


def invariant_transfer(left, right):
    if set(left) != set(right):
        raise ValueError("invariant records are not aligned")
    records = {
        key: relative_difference(left[key], right[key]) for key in left
    }
    return {"records": records, "maximum": max(records.values(), default=0.0)}


def array_relative_difference(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError("array comparison requires equal shapes")
    return float(np.linalg.norm(left - right) / max(
        np.linalg.norm(left), np.linalg.norm(right), 1e-300,
    ))


def tensor_domain_transfer(left_state, right_state, z, r, initial, radius_cut=6.0):
    """Compare full metric, evolved increment, and ADM K on exact common nodes."""
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    keep = r <= float(radius_cut) + 1e-12
    local_r = r[keep]
    left_position = np.asarray(left_state["position"])[:, keep]
    right_position = np.asarray(right_state["position"])[:, keep]
    left_velocity = np.asarray(left_state["velocity"])[:, keep]
    right_velocity = np.asarray(right_state["velocity"])[:, keep]
    initial_position = np.asarray(initial)[:, keep]
    left_metric = spatial_metric_tensor(left_position, local_r)
    right_metric = spatial_metric_tensor(right_position, local_r)
    initial_metric = spatial_metric_tensor(initial_position, local_r)
    left_k = adm_extrinsic_curvature_tensor(left_position, left_velocity, z, local_r)
    right_k = adm_extrinsic_curvature_tensor(right_position, right_velocity, z, local_r)
    return {
        "full_metric": physical_tensor_difference(
            left_metric, right_metric, left_metric, right_metric, z, local_r,
        ),
        "metric_increment": physical_tensor_difference(
            left_metric - initial_metric, right_metric - initial_metric,
            left_metric, right_metric, z, local_r,
        ),
        "ADM_K": physical_tensor_difference(
            left_k, right_k, left_metric, right_metric, z, local_r,
        ),
    }


def valid_persistent_pair_history(counts):
    values = [int(value) for value in counts]
    return bool(
        values and all(value in (0, 2) for value in values)
        and 0 in values and 2 in values
        and all(left <= right for left, right in zip(values, values[1:]))
    )


def first_detection_bracket(counts, dt):
    values = [int(value) for value in counts]
    if 2 not in values:
        return None
    index = values.index(2)
    return [float(index * dt), float((index + 1) * dt)]


def brackets_overlap(left, right):
    if left is None or right is None:
        return False
    return bool(max(left[0], right[0]) < min(left[1], right[1]) + 1e-15)


def classify_test10b(valid, construction, pair, formation, geometry, tensors, physical_fail):
    """Apply the sealed top-level Test-10B classification."""
    if not valid or not construction:
        return "review", "invalid_common_parent_audit"
    if physical_fail:
        return "fail", "residual_domain_dependence_under_valid_common_parent"
    if all((pair, formation, geometry, tensors)):
        return (
            "pass",
            "original_domain_drift_attributed_to_separate_elliptic_initial_data_within_common_parent_test",
        )
    return "review", "common_parent_test_mixed"
