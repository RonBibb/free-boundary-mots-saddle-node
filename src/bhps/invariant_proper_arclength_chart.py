"""Proper-brane-arclength identification chart for Test 2C."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    PchipInterpolator,
    RectBivariateSpline,
)

from bhps.invariant_physical_chart import NormalGeodesicChart


@dataclass(frozen=True)
class ProperArclengthChart:
    distance: np.ndarray
    arclength: np.ndarray
    native_brane_radius: np.ndarray
    z: np.ndarray
    r: np.ndarray
    velocity: np.ndarray
    areal_radius: np.ndarray
    speed_squared: np.ndarray
    jacobian_DS_zr: np.ndarray
    eikonal_qDD: np.ndarray

    @property
    def shape(self):
        return self.z.shape


def brane_arclength(z, r, metric, native_brane_radius):
    """Integrate the induced radial line element on brane B."""
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    metric = np.asarray(metric, dtype=float)
    native = np.asarray(native_brane_radius, dtype=float)
    if metric.shape != (len(z), len(r), 2, 2):
        raise ValueError("base metric and native grid do not match")
    if native.ndim != 1 or native[0] != 0.0 or np.any(np.diff(native) <= 0.0):
        raise ValueError("native brane radii must start at zero and increase")
    line_metric = RectBivariateSpline(
        z, r, metric[..., 1, 1], s=0,
        kx=min(3, len(z) - 1), ky=min(3, len(r) - 1),
    ).ev(np.full_like(native, z[-1]), native)
    if np.min(line_metric) <= 0.0 or not np.all(np.isfinite(line_metric)):
        raise ValueError("brane radial line metric is not positive")
    return np.concatenate((
        [0.0], cumulative_trapezoid(np.sqrt(line_metric), x=native),
    ))


def relabel_normal_chart(normal, z, r, metric):
    """Relabel fixed normal rays by physical arclength of their launch point."""
    if not isinstance(normal, NormalGeodesicChart):
        raise TypeError("a frozen normal-geodesic chart is required")
    S = brane_arclength(z, r, metric, normal.ray_label)
    position = np.stack((normal.z, normal.r), axis=-1)
    x_D = np.gradient(position, normal.distance, axis=0, edge_order=2)
    x_S = np.gradient(position, S, axis=1, edge_order=2)
    forward = x_D[..., 0] * x_S[..., 1] - x_D[..., 1] * x_S[..., 0]
    jacobian = 1.0 / forward

    splines = {
        (a, b): RectBivariateSpline(
            z, r, metric[..., a, b], s=0,
            kx=min(3, len(z) - 1), ky=min(3, len(r) - 1),
        ) for a in range(2) for b in range(2)
    }
    native_metric = np.empty((*normal.z.shape, 2, 2))
    for a in range(2):
        for b in range(2):
            native_metric[..., a, b] = splines[a, b].ev(
                normal.z.ravel(), normal.r.ravel(),
            ).reshape(normal.z.shape)
    chart_metric = np.empty((*normal.z.shape, 2, 2))
    chart_metric[..., 0, 0] = np.einsum(
        "...a,...ab,...b->...", x_D, native_metric, x_D,
    )
    chart_metric[..., 0, 1] = chart_metric[..., 1, 0] = np.einsum(
        "...a,...ab,...b->...", x_D, native_metric, x_S,
    )
    chart_metric[..., 1, 1] = np.einsum(
        "...a,...ab,...b->...", x_S, native_metric, x_S,
    )
    eikonal = np.linalg.inv(chart_metric)[..., 0, 0]
    return ProperArclengthChart(
        distance=np.asarray(normal.distance), arclength=S,
        native_brane_radius=np.asarray(normal.ray_label),
        z=np.asarray(normal.z), r=np.asarray(normal.r),
        velocity=np.asarray(normal.velocity),
        areal_radius=np.asarray(normal.areal_radius),
        speed_squared=np.asarray(normal.speed_squared),
        jacobian_DS_zr=jacobian, eikonal_qDD=eikonal,
    )


def chart_validity(chart, coarse=False, jacobian_fraction=1e-6):
    arrays = (
        chart.distance, chart.arclength, chart.z, chart.r,
        chart.jacobian_DS_zr, chart.eikonal_qDD,
    )
    finite = bool(all(np.all(np.isfinite(value)) for value in arrays))
    jacobian = np.asarray(chart.jacobian_DS_zr)
    sign = float(np.sign(np.median(jacobian))) if finite else 0.0
    median = float(np.median(np.abs(jacobian))) if finite else 0.0
    floor = float(jacobian_fraction * median)
    jacobian_valid = bool(
        finite and sign != 0.0 and np.all(sign * jacobian > floor)
    )
    arclength_monotone = bool(
        finite and chart.arclength[0] == 0.0
        and np.all(np.diff(chart.arclength) > 0.0)
    )
    error = np.abs(chart.eikonal_qDD - 1.0)[1:-1, 1:-1]
    l2 = float(np.sqrt(np.mean(error**2)))
    linf = float(np.max(error))
    l2_limit = 1e-4 if coarse else 5e-5
    linf_limit = 1e-3 if coarse else 5e-4
    return {
        "finite": finite, "jacobian_sign": sign,
        "minimum_oriented_jacobian": float(np.min(sign * jacobian)) if finite else None,
        "jacobian_floor": floor, "jacobian_valid": jacobian_valid,
        "arclength_strictly_monotone": arclength_monotone,
        "eikonal_L2": l2, "eikonal_Linf": linf,
        "eikonal_limits": {"L2": l2_limit, "Linf": linf_limit},
        "valid": bool(
            jacobian_valid and arclength_monotone
            and l2 < l2_limit and linf < linf_limit
        ),
    }


def inverse_chart_at(chart, target_distance, target_arclength):
    target_distance = np.asarray(target_distance, dtype=float)
    target_arclength = np.asarray(target_arclength, dtype=float)
    if (
        target_distance.ndim != 1 or target_arclength.ndim != 1
        or np.any(np.diff(target_distance) <= 0.0)
        or np.any(np.diff(target_arclength) <= 0.0)
    ):
        raise ValueError("target physical coordinates must increase")
    if (
        target_distance[0] < chart.distance[0]
        or target_distance[-1] > chart.distance[-1]
        or target_arclength[0] < chart.arclength[0]
        or target_arclength[-1] > chart.arclength[-1]
    ):
        raise ValueError("proper-arclength inverse would extrapolate")
    ray_z = PchipInterpolator(chart.distance, chart.z, axis=0)(target_distance)
    ray_r = PchipInterpolator(chart.distance, chart.r, axis=0)(target_distance)
    native_z = PchipInterpolator(chart.arclength, ray_z, axis=1)(target_arclength)
    native_r = PchipInterpolator(chart.arclength, ray_r, axis=1)(target_arclength)
    return native_z, native_r


def native_to_coordinates(chart, native_z, native_r, method="linear"):
    native_z, native_r = np.broadcast_arrays(
        np.asarray(native_z, dtype=float), np.asarray(native_r, dtype=float),
    )
    points = np.column_stack((chart.z.ravel(), chart.r.ravel()))
    values_D = np.broadcast_to(chart.distance[:, None], chart.z.shape).ravel()
    values_S = np.broadcast_to(chart.arclength[None, :], chart.z.shape).ravel()
    interpolator = (
        LinearNDInterpolator if method == "linear"
        else CloughTocher2DInterpolator if method == "cubic"
        else None
    )
    if interpolator is None:
        raise ValueError("method must be 'linear' or 'cubic'")
    D = interpolator(points, values_D, fill_value=np.nan)(
        native_z.ravel(), native_r.ravel(),
    ).reshape(native_z.shape)
    S = interpolator(points, values_S, fill_value=np.nan)(
        native_z.ravel(), native_r.ravel(),
    ).reshape(native_z.shape)
    if not (np.all(np.isfinite(D)) and np.all(np.isfinite(S))):
        raise ValueError("native point lies outside the proper-arclength chart")
    return D, S


def arclength_at_native_radius(chart, native_radius):
    native_radius = float(native_radius)
    if native_radius < chart.native_brane_radius[0] or native_radius > chart.native_brane_radius[-1]:
        raise ValueError("native radius leaves chart brane interval")
    return float(PchipInterpolator(
        chart.native_brane_radius, chart.arclength,
    )(native_radius))
