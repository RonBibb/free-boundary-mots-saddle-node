"""Physical tensor norms for the four-grid A=7.90 convergence audit."""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import simpson
from scipy.optimize import brentq

from bhps.dynamical_capped_horizon import regular_so3_adm_slice


def spatial_metric_tensor(position, radius):
    """Expand a regular SO(3) state into the spatial Cartesian metric.

    The representative Cartesian radial direction is the first of the three
    SO(3) directions.  The final two entries are degenerate tangential
    directions; contractions include their multiplicity explicitly.
    """
    q = np.asarray(position, dtype=float)
    r = np.asarray(radius, dtype=float)
    if q.ndim != 3 or q.shape[-1] != 9 or r.ndim != 1 or q.shape[1] != len(r):
        raise ValueError("invalid regular SO(3) metric state")
    radius_field = r[None, :]
    result = np.zeros((*q.shape[:2], 4, 4))
    result[..., 0, 0] = q[..., 6]
    result[..., 0, 1] = result[..., 1, 0] = radius_field * q[..., 1]
    result[..., 1, 1] = q[..., 3] + radius_field**2 * q[..., 4]
    result[..., 2, 2] = q[..., 3]
    result[..., 3, 3] = q[..., 3]
    if not np.all(np.isfinite(result)):
        raise ValueError("spatial metric is nonfinite")
    return result


def adm_extrinsic_curvature_tensor(position, velocity, z, radius, stencil_width=7):
    """Return full covariant ADM ``K_ij`` from the evolved metric and shift."""
    q = np.asarray(position, dtype=float)
    adm = regular_so3_adm_slice(q, velocity, z, radius, stencil_width)
    result = np.zeros((*q.shape[:2], 4, 4))
    result[..., :2, :2] = adm["extrinsic_base"]
    tangential = q[..., 3] * adm["extrinsic_sphere_eigenvalue"]
    result[..., 2, 2] = tangential
    result[..., 3, 3] = tangential
    if not np.all(np.isfinite(result)):
        raise ValueError("ADM extrinsic curvature is nonfinite")
    return result


def interpolate_tensor(field, source_z, source_r, target_z, target_r):
    """Cubic componentwise transfer of a tensor field to a common grid."""
    values = np.asarray(field, dtype=float)
    source_z = np.asarray(source_z, dtype=float)
    source_r = np.asarray(source_r, dtype=float)
    target_z = np.asarray(target_z, dtype=float)
    target_r = np.asarray(target_r, dtype=float)
    if values.shape[:2] != (len(source_z), len(source_r)):
        raise ValueError("tensor and source grid do not match")
    zz, rr = np.meshgrid(target_z, target_r, indexing="ij")
    result = np.empty((len(target_z), len(target_r), *values.shape[2:]))
    for index in np.ndindex(values.shape[2:]):
        spline = RectBivariateSpline(
            source_z, source_r,
            values[(slice(None), slice(None), *index)],
            kx=min(3, len(source_z) - 1), ky=min(3, len(source_r) - 1), s=0,
        )
        result[(slice(None), slice(None), *index)] = spline.ev(
            zz.ravel(), rr.ravel(),
        ).reshape(len(target_z), len(target_r))
    return result


def physical_tensor_l2(tensor, reference_metric, z, radius):
    """SO(3)-integrated proper-volume L2 norm of a covariant two-tensor."""
    value = np.asarray(tensor, dtype=float)
    metric = np.asarray(reference_metric, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(radius, dtype=float)
    expected = (len(z), len(r), 4, 4)
    if value.shape != expected or metric.shape != expected:
        raise ValueError("tensor norm fields and grid do not match")
    eigenvalues = np.linalg.eigvalsh(metric)
    if np.min(eigenvalues) <= 0:
        raise ValueError("reference spatial metric is not positive definite")
    inverse = np.linalg.inv(metric)
    pointwise = np.einsum(
        "...ik,...jl,...ij,...kl->...", inverse, inverse, value, value,
        optimize=True,
    )
    base_determinant = (
        metric[..., 0, 0] * metric[..., 1, 1] - metric[..., 0, 1] ** 2
    )
    transverse = 0.5 * (metric[..., 2, 2] + metric[..., 3, 3])
    volume = 4.0 * math.pi * r[None, :] ** 2 * transverse * np.sqrt(base_determinant)
    radial = simpson(np.maximum(pointwise, 0.0) * volume, x=r, axis=1)
    squared = float(simpson(radial, x=z))
    return float(np.sqrt(max(squared, 0.0)))


def physical_tensor_difference(left, right, metric_left, metric_right, z, radius):
    """Absolute and relative proper-volume difference using midpoint metric."""
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    midpoint = 0.5 * (
        np.asarray(metric_left, dtype=float) + np.asarray(metric_right, dtype=float)
    )
    absolute = physical_tensor_l2(left - right, midpoint, z, radius)
    left_norm = physical_tensor_l2(left, midpoint, z, radius)
    right_norm = physical_tensor_l2(right, midpoint, z, radius)
    return {
        "absolute_difference": absolute,
        "relative_difference": float(
            absolute / max(left_norm, right_norm, 1e-300)
        ),
        "left_norm": left_norm,
        "right_norm": right_norm,
        "reference_metric": "arithmetic midpoint of the two final spatial metrics",
        "volume_measure": "4*pi*r^2*gamma_perp*sqrt(det(gamma_zr))*dz*dr",
    }


def generalized_order_nonuniform(difference_12, difference_23, intervals):
    """Infer ``p`` for three arbitrary successively refined interval counts."""
    d12 = float(difference_12)
    d23 = float(difference_23)
    counts = tuple(float(value) for value in intervals)
    if (
        d12 <= 0 or d23 <= 0 or len(counts) != 3
        or not counts[0] < counts[1] < counts[2]
    ):
        return None
    h1, h2, h3 = (1.0 / value for value in counts)
    ratio = d12 / d23

    def residual(order):
        return (h1**order - h2**order) / (h2**order - h3**order) - ratio

    lower, upper = 1e-7, 20.0
    if residual(lower) * residual(upper) > 0:
        return None
    return float(brentq(residual, lower, upper))


def four_grid_orders(differences, intervals=(80, 96, 112, 128)):
    """Return local generalized orders on grids 1/2/3 and 2/3/4."""
    values = tuple(float(value) for value in differences)
    counts = tuple(float(value) for value in intervals)
    if len(values) != 3 or len(counts) != 4:
        raise ValueError("four-grid order requires three differences and four spacings")
    return {
        "coarse_triplet_order": generalized_order_nonuniform(
            values[0], values[1], counts[:3],
        ),
        "fine_triplet_order": generalized_order_nonuniform(
            values[1], values[2], counts[1:],
        ),
    }
