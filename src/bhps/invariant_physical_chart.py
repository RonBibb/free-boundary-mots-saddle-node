"""Geometric grid identification for the Test-2B convergence audit.

The native ``(z,r)`` labels are replaced by inward proper distance from the
brane and areal radius.  Proper-distance characteristics are the unit-speed
geodesics launched orthogonally from the brane.  The construction is admitted
only before the normal congruence develops a caustic or leaves the grid.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import simpson
from scipy.interpolate import (
    CloughTocher2DInterpolator,
    LinearNDInterpolator,
    PchipInterpolator,
    RectBivariateSpline,
)
from scipy.optimize import brentq


def _as_base_metric(metric: np.ndarray, z: np.ndarray, r: np.ndarray) -> np.ndarray:
    value = np.asarray(metric, dtype=float)
    expected = (len(z), len(r), 2, 2)
    if value.shape != expected:
        raise ValueError(f"base metric has shape {value.shape}, expected {expected}")
    if not np.all(np.isfinite(value)):
        raise ValueError("base metric is nonfinite")
    if np.min(np.linalg.eigvalsh(value)) <= 0.0:
        raise ValueError("base metric is not positive definite")
    return value


class MetricSplines:
    """Cubic interpolants of a regular-grid two-metric and sphere factor."""

    def __init__(self, z, r, metric, sphere_factor):
        self.z = np.asarray(z, dtype=float)
        self.r = np.asarray(r, dtype=float)
        self.metric = _as_base_metric(metric, self.z, self.r)
        self.sphere_factor = np.asarray(sphere_factor, dtype=float)
        if self.sphere_factor.shape != (len(self.z), len(self.r)):
            raise ValueError("sphere factor and grid do not match")
        if np.min(self.sphere_factor) <= 0.0:
            raise ValueError("sphere factor must be positive")
        self._metric_splines = {
            (a, b): RectBivariateSpline(
                self.z, self.r, self.metric[..., a, b], s=0,
                kx=min(3, len(self.z) - 1), ky=min(3, len(self.r) - 1),
            )
            for a in range(2) for b in range(2)
        }
        self._sphere_spline = RectBivariateSpline(
            self.z, self.r, self.sphere_factor, s=0,
            kx=min(3, len(self.z) - 1), ky=min(3, len(self.r) - 1),
        )

    @staticmethod
    def _points(z, r):
        zz, rr = np.broadcast_arrays(np.asarray(z, dtype=float), np.asarray(r, dtype=float))
        return zz, rr, zz.shape

    def evaluate_metric(self, z, r):
        zz, rr, shape = self._points(z, r)
        result = np.empty((*shape, 2, 2))
        for a in range(2):
            for b in range(2):
                result[..., a, b] = self._metric_splines[a, b].ev(
                    zz.ravel(), rr.ravel(),
                ).reshape(shape)
        return result

    def evaluate_sphere_factor(self, z, r):
        zz, rr, shape = self._points(z, r)
        return self._sphere_spline.ev(zz.ravel(), rr.ravel()).reshape(shape)

    def metric_and_christoffel(self, z, r):
        zz, rr, shape = self._points(z, r)
        metric = np.empty((*shape, 2, 2))
        derivative = np.empty((*shape, 2, 2, 2))
        for a in range(2):
            for b in range(2):
                spline = self._metric_splines[a, b]
                metric[..., a, b] = spline.ev(zz.ravel(), rr.ravel()).reshape(shape)
                derivative[..., 0, a, b] = spline.ev(
                    zz.ravel(), rr.ravel(), dx=1, dy=0,
                ).reshape(shape)
                derivative[..., 1, a, b] = spline.ev(
                    zz.ravel(), rr.ravel(), dx=0, dy=1,
                ).reshape(shape)
        inverse = np.linalg.inv(metric)
        gamma = np.empty((*shape, 2, 2, 2))
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    total = np.zeros(shape)
                    for d in range(2):
                        total += inverse[..., a, d] * (
                            derivative[..., b, d, c]
                            + derivative[..., c, d, b]
                            - derivative[..., d, b, c]
                        )
                    gamma[..., a, b, c] = 0.5 * total
        return metric, gamma


@dataclass(frozen=True)
class NormalGeodesicChart:
    distance: np.ndarray
    ray_label: np.ndarray
    z: np.ndarray
    r: np.ndarray
    velocity: np.ndarray
    areal_radius: np.ndarray
    speed_squared: np.ndarray
    jacobian_DR_zr: np.ndarray
    eikonal_qDD: np.ndarray

    @property
    def shape(self):
        return self.z.shape


def _geodesic_rhs(interpolator: MetricSplines, state: np.ndarray) -> np.ndarray:
    position = state[..., :2]
    velocity = state[..., 2:]
    _, christoffel = interpolator.metric_and_christoffel(
        position[..., 0], position[..., 1],
    )
    acceleration = -np.einsum(
        "...abc,...b,...c->...a", christoffel, velocity, velocity,
        optimize=True,
    )
    return np.concatenate((velocity, acceleration), axis=-1)


def build_normal_geodesic_chart(
    z, r, metric, sphere_factor, distance, ray_count=None,
) -> NormalGeodesicChart:
    """Integrate the inward unit-normal congruence from the brane.

    ``distance`` is the common physical-distance grid chosen before comparing
    resolutions.  The routine rejects any characteristic that exits the
    native domain, so downstream code cannot extrapolate it back in.
    """
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    distance = np.asarray(distance, dtype=float)
    if z.ndim != 1 or r.ndim != 1 or distance.ndim != 1:
        raise ValueError("chart coordinates must be one-dimensional")
    if len(distance) < 3 or distance[0] != 0.0 or np.any(np.diff(distance) <= 0.0):
        raise ValueError("distance grid must start at zero and increase")
    if not np.allclose(np.diff(distance), np.diff(distance)[0], rtol=1e-12, atol=1e-14):
        raise ValueError("RK4 chart construction requires uniform distance spacing")
    ray_count = int(ray_count or len(r))
    if ray_count < 5:
        raise ValueError("at least five normal rays are required")
    ray_label = np.linspace(r[0], r[-1], ray_count)
    interpolator = MetricSplines(z, r, metric, sphere_factor)

    boundary_metric = interpolator.evaluate_metric(np.full(ray_count, z[-1]), ray_label)
    boundary_inverse = np.linalg.inv(boundary_metric)
    initial_velocity = -boundary_inverse[..., :, 0] / np.sqrt(
        boundary_inverse[..., 0, 0],
    )[..., None]
    state = np.concatenate((
        np.column_stack((np.full(ray_count, z[-1]), ray_label)),
        initial_velocity,
    ), axis=1)
    states = np.empty((len(distance), ray_count, 4))
    states[0] = state
    step = float(distance[1] - distance[0])
    for index in range(1, len(distance)):
        k1 = _geodesic_rhs(interpolator, state)
        k2 = _geodesic_rhs(interpolator, state + 0.5 * step * k1)
        k3 = _geodesic_rhs(interpolator, state + 0.5 * step * k2)
        k4 = _geodesic_rhs(interpolator, state + step * k3)
        state = state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        if (
            np.min(state[:, 0]) < z[0] or np.max(state[:, 0]) > z[-1] + 1e-10
            or np.min(state[:, 1]) < r[0] - 1e-10
            or np.max(state[:, 1]) > r[-1] + 1e-10
        ):
            raise ValueError("normal congruence exits the native grid")
        states[index] = state

    position = states[..., :2]
    velocity = states[..., 2:]
    native_metric = interpolator.evaluate_metric(position[..., 0], position[..., 1])
    speed_squared = np.einsum(
        "...a,...ab,...b->...", velocity, native_metric, velocity,
        optimize=True,
    )
    sphere = interpolator.evaluate_sphere_factor(position[..., 0], position[..., 1])
    areal_radius = position[..., 1] * np.sqrt(sphere)

    x_D = np.gradient(position, distance, axis=0, edge_order=2)
    x_u = np.gradient(position, ray_label, axis=1, edge_order=2)
    forward_jacobian = x_D[..., 0] * x_u[..., 1] - x_D[..., 1] * x_u[..., 0]
    R_u = np.gradient(areal_radius, ray_label, axis=1, edge_order=2)
    jacobian = R_u / forward_jacobian
    chart_metric = np.empty((*position.shape[:2], 2, 2))
    chart_metric[..., 0, 0] = np.einsum(
        "...a,...ab,...b->...", x_D, native_metric, x_D,
    )
    chart_metric[..., 0, 1] = chart_metric[..., 1, 0] = np.einsum(
        "...a,...ab,...b->...", x_D, native_metric, x_u,
    )
    chart_metric[..., 1, 1] = np.einsum(
        "...a,...ab,...b->...", x_u, native_metric, x_u,
    )
    eikonal_qDD = np.linalg.inv(chart_metric)[..., 0, 0]
    return NormalGeodesicChart(
        distance=distance, ray_label=ray_label,
        z=position[..., 0], r=position[..., 1], velocity=velocity,
        areal_radius=areal_radius, speed_squared=speed_squared,
        jacobian_DR_zr=jacobian, eikonal_qDD=eikonal_qDD,
    )


def chart_validity(chart: NormalGeodesicChart, jacobian_fraction=1e-6):
    """Return the fixed map diagnostics and prospective validity decision."""
    jacobian = np.asarray(chart.jacobian_DR_zr)
    finite = bool(
        np.all(np.isfinite(chart.z)) and np.all(np.isfinite(chart.r))
        and np.all(np.isfinite(jacobian)) and np.all(np.isfinite(chart.eikonal_qDD))
    )
    sign = np.sign(np.median(jacobian)) if finite else 0.0
    median = float(np.median(np.abs(jacobian))) if finite else 0.0
    jacobian_floor = float(jacobian_fraction * median)
    jacobian_valid = bool(
        finite and sign != 0.0 and np.all(sign * jacobian > jacobian_floor)
    )
    radius_monotone = bool(
        finite and np.all(np.diff(chart.areal_radius, axis=1) > 0.0)
    )
    error = np.abs(chart.eikonal_qDD - 1.0)
    interior = error[1:-1, 1:-1]
    l2 = float(np.sqrt(np.mean(interior**2)))
    linf = float(np.max(interior))
    speed_error = float(np.max(np.abs(chart.speed_squared - 1.0)))
    return {
        "finite": finite,
        "jacobian_sign": float(sign),
        "minimum_oriented_jacobian": float(np.min(sign * jacobian)) if finite else None,
        "jacobian_floor": jacobian_floor,
        "jacobian_valid": jacobian_valid,
        "areal_radius_strictly_monotone": radius_monotone,
        "eikonal_l2": l2,
        "eikonal_linf": linf,
        "geodesic_speed_linf_error": speed_error,
        "valid": bool(jacobian_valid and radius_monotone and l2 < 5e-5 and linf < 5e-4),
    }


def common_areal_interval(charts, outer_limit=6.0):
    """Intersection of the strictly monotone areal-radius images."""
    if not charts:
        raise ValueError("at least one chart is required")
    lower = max(float(np.max(chart.areal_radius[:, 0])) for chart in charts)
    upper = min(float(np.min(chart.areal_radius[:, -1])) for chart in charts)
    upper = min(upper, float(outer_limit))
    if not lower < upper:
        raise ValueError("charts have no common areal-radius interval")
    return lower, upper


def inverse_chart_at(chart: NormalGeodesicChart, target_distance, target_radius):
    """Invert the normal chart without native-coordinate extrapolation."""
    target_distance = np.asarray(target_distance, dtype=float)
    target_radius = np.asarray(target_radius, dtype=float)
    if (
        target_distance.ndim != 1 or target_radius.ndim != 1
        or np.any(np.diff(target_distance) <= 0.0)
        or np.any(np.diff(target_radius) <= 0.0)
    ):
        raise ValueError("target physical coordinates must be increasing vectors")
    if (
        target_distance[0] < chart.distance[0]
        or target_distance[-1] > chart.distance[-1]
    ):
        raise ValueError("inverse chart would extrapolate in proper distance")
    if np.array_equal(target_distance, chart.distance):
        ray_z, ray_r, ray_R = chart.z, chart.r, chart.areal_radius
    else:
        ray_z = PchipInterpolator(chart.distance, chart.z, axis=0)(target_distance)
        ray_r = PchipInterpolator(chart.distance, chart.r, axis=0)(target_distance)
        ray_R = PchipInterpolator(
            chart.distance, chart.areal_radius, axis=0,
        )(target_distance)
    shape = (len(target_distance), len(target_radius))
    native_z = np.empty(shape)
    native_r = np.empty(shape)
    for index in range(len(target_distance)):
        radius = ray_R[index]
        if target_radius[0] < radius[0] or target_radius[-1] > radius[-1]:
            raise ValueError("inverse chart would extrapolate")
        native_z[index] = PchipInterpolator(radius, ray_z[index])(target_radius)
        native_r[index] = PchipInterpolator(radius, ray_r[index])(target_radius)
    return native_z, native_r


def inverse_chart(chart: NormalGeodesicChart, target_radius):
    return inverse_chart_at(chart, chart.distance, target_radius)


def native_to_proper_distance(chart, native_z, native_r, method="linear"):
    """Evaluate the chart's distance scalar at paired native points."""
    native_z, native_r = np.broadcast_arrays(
        np.asarray(native_z, dtype=float), np.asarray(native_r, dtype=float),
    )
    points = np.column_stack((chart.z.ravel(), chart.r.ravel()))
    values = np.broadcast_to(
        chart.distance[:, None], chart.z.shape,
    ).ravel()
    if method == "linear":
        interpolator = LinearNDInterpolator(points, values, fill_value=np.nan)
    elif method == "cubic":
        interpolator = CloughTocher2DInterpolator(points, values, fill_value=np.nan)
    else:
        raise ValueError("method must be 'linear' or 'cubic'")
    result = interpolator(native_z.ravel(), native_r.ravel()).reshape(native_z.shape)
    if not np.all(np.isfinite(result)):
        raise ValueError("native point lies outside the physical chart")
    return result


def interpolate_regular_field(field, z, r, target_z, target_r, method="cubic"):
    """Evaluate a scalar- or tensor-valued regular-grid field at paired points."""
    values = np.asarray(field, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    target_z, target_r = np.broadcast_arrays(
        np.asarray(target_z, dtype=float), np.asarray(target_r, dtype=float),
    )
    if values.shape[:2] != (len(z), len(r)):
        raise ValueError("field and source grid do not match")
    degree = 3 if method == "cubic" else 1 if method == "linear" else None
    if degree is None:
        raise ValueError("method must be 'cubic' or 'linear'")
    result = np.empty((*target_z.shape, *values.shape[2:]))
    for component in np.ndindex(values.shape[2:]):
        spline = RectBivariateSpline(
            z, r, values[(slice(None), slice(None), *component)], s=0,
            kx=min(degree, len(z) - 1), ky=min(degree, len(r) - 1),
        )
        result[(slice(None), slice(None), *component)] = spline.ev(
            target_z.ravel(), target_r.ravel(),
        ).reshape(target_z.shape)
    return result


def mapped_metric_fields(
    metric, sphere_factor, z, r, distance, target_radius,
    native_z, native_r, method="cubic",
):
    """Metric coefficients and proper-volume weight in the ``(D,R)`` chart."""
    native_metric = interpolate_regular_field(
        metric, z, r, native_z, native_r, method=method,
    )
    x_D = np.stack((
        np.gradient(native_z, distance, axis=0, edge_order=2),
        np.gradient(native_r, distance, axis=0, edge_order=2),
    ), axis=-1)
    x_R = np.stack((
        np.gradient(native_z, target_radius, axis=1, edge_order=2),
        np.gradient(native_r, target_radius, axis=1, edge_order=2),
    ), axis=-1)
    covariant = np.empty((*native_z.shape, 2, 2))
    covariant[..., 0, 0] = np.einsum(
        "...a,...ab,...b->...", x_D, native_metric, x_D,
    )
    covariant[..., 0, 1] = covariant[..., 1, 0] = np.einsum(
        "...a,...ab,...b->...", x_D, native_metric, x_R,
    )
    covariant[..., 1, 1] = np.einsum(
        "...a,...ab,...b->...", x_R, native_metric, x_R,
    )
    inverse = np.linalg.inv(covariant)
    determinant = np.linalg.det(covariant)
    sphere = interpolate_regular_field(
        sphere_factor, z, r, native_z, native_r, method=method,
    )
    native_areal = native_r * np.sqrt(sphere)
    volume = 4.0 * math.pi * native_areal**2 * np.sqrt(determinant)
    return {
        "covariant": covariant,
        "q_DD": inverse[..., 0, 0],
        "q_DR": inverse[..., 0, 1],
        "q_RR": inverse[..., 1, 1],
        "volume_density": volume,
        "native_areal_radius": native_areal,
        "basis_D": x_D,
        "basis_R": x_R,
        "native_metric": native_metric,
    }


def mapped_extrinsic_fields(
    extrinsic, sphere_factor, z, r, native_z, native_r, mapped_metric,
    method="cubic",
):
    """ADM extrinsic-curvature components in the fixed physical frame."""
    K = interpolate_regular_field(extrinsic, z, r, native_z, native_r, method=method)
    base_K = K[..., :2, :2]
    metric = mapped_metric["native_metric"]
    x_D = mapped_metric["basis_D"]
    x_R = mapped_metric["basis_R"]
    # grad(D) from the inverse metric in the mapped coordinate basis.
    chart_inverse = np.linalg.inv(mapped_metric["covariant"])
    grad_D = (
        chart_inverse[..., 0, 0, None] * x_D
        + chart_inverse[..., 1, 0, None] * x_R
    )
    norm_D = np.sqrt(np.einsum("...a,...ab,...b->...", grad_D, metric, grad_D))
    e_D = grad_D / norm_D[..., None]
    projection = np.einsum("...a,...ab,...b->...", x_R, metric, e_D)
    tangent = x_R - projection[..., None] * e_D
    norm_R = np.sqrt(np.einsum("...a,...ab,...b->...", tangent, metric, tangent))
    e_R = tangent / norm_R[..., None]
    K_DD = np.einsum("...a,...ab,...b->...", e_D, base_K, e_D)
    K_DR = np.einsum("...a,...ab,...b->...", e_D, base_K, e_R)
    K_RR = np.einsum("...a,...ab,...b->...", e_R, base_K, e_R)
    K_tangent_covariant = 0.5 * (K[..., 2, 2] + K[..., 3, 3])
    mapped_sphere_factor = interpolate_regular_field(
        sphere_factor, z, r, native_z, native_r, method=method,
    )
    K_Omega = K_tangent_covariant / mapped_sphere_factor
    base_inverse = np.linalg.inv(metric)
    trace_base = np.einsum("...ab,...ab->...", base_inverse, base_K)
    K_base_squared = np.einsum(
        "...ac,...bd,...ab,...cd->...", base_inverse, base_inverse,
        base_K, base_K, optimize=True,
    )
    return {
        "K_DD": K_DD, "K_DR": K_DR, "K_RR": K_RR,
        "K_Omega": K_Omega,
        "trace_K": trace_base + 2.0 * K_Omega,
        "KijKij": K_base_squared + 2.0 * K_Omega**2,
    }


def weighted_l2(field, weight, distance, radius):
    values = np.asarray(field, dtype=float)
    weight = np.asarray(weight, dtype=float)
    if values.shape[:2] != weight.shape:
        raise ValueError("field and weight do not match")
    squared = np.sum(values**2, axis=tuple(range(2, values.ndim)))
    radial = simpson(squared * weight, x=radius, axis=1)
    return float(np.sqrt(max(float(simpson(radial, x=distance)), 0.0)))


def weighted_quantile(values, weight, quantile=0.95):
    values = np.asarray(values, dtype=float).ravel()
    weight = np.asarray(weight, dtype=float).ravel()
    if values.shape != weight.shape or np.any(weight < 0.0):
        raise ValueError("weighted quantile inputs are invalid")
    order = np.argsort(values)
    values, weight = values[order], weight[order]
    cumulative = np.cumsum(weight)
    if cumulative[-1] <= 0.0:
        raise ValueError("weighted quantile requires positive total weight")
    return float(values[np.searchsorted(cumulative, quantile * cumulative[-1])])


def generalized_order(difference_12, difference_23, intervals=(112, 128, 144)):
    d12, d23 = float(difference_12), float(difference_23)
    n1, n2, n3 = map(float, intervals)
    if d12 <= 0.0 or d23 <= 0.0 or not n1 < n2 < n3:
        return None
    ratio = d12 / d23
    def residual(order):
        return (
            (n1**-order - n2**-order) / (n2**-order - n3**-order)
            - ratio
        )
    lower, upper = 1e-8, 30.0
    if residual(lower) * residual(upper) > 0.0:
        return None
    return float(brentq(residual, lower, upper))


def conservative_order_interval(e12, u12, e23, u23, intervals=(112, 128, 144)):
    lower12, upper12 = max(float(e12) - float(u12), 0.0), float(e12) + float(u12)
    lower23, upper23 = max(float(e23) - float(u23), 0.0), float(e23) + float(u23)
    if lower12 <= 0.0 or lower23 <= 0.0:
        return None
    lower_order = generalized_order(lower12, upper23, intervals)
    upper_order = generalized_order(upper12, lower23, intervals)
    if lower_order is None or upper_order is None:
        return None
    return min(lower_order, upper_order), max(lower_order, upper_order)


def sign_coherence(coarse_difference, fine_difference, weight, uncertainty=0.0):
    coarse = np.asarray(coarse_difference, dtype=float)
    fine = np.asarray(fine_difference, dtype=float)
    weight = np.broadcast_to(np.asarray(weight, dtype=float), coarse.shape)
    if coarse.shape != fine.shape:
        raise ValueError("difference fields do not match")
    admitted = (np.abs(coarse) > uncertainty) & (np.abs(fine) > uncertainty)
    denominator = float(np.sum(weight[admitted]))
    if denominator == 0.0:
        return None
    coherent = admitted & (np.sign(coarse) == np.sign(fine))
    return float(np.sum(weight[coherent]) / denominator)
