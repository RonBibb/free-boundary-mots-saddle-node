"""Finite-archive backward null-generator reconstruction utilities.

These routines reconstruct a finite-time null hypersurface proxy from a
terminal spacelike cross-section.  They do not identify a global event
horizon: the metric history has finite future extent and the terminal surface
is an explicit input.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, RectBivariateSpline


METRIC_COMPONENTS = (0, 1, 2, 3, 4, 5, 6)


class ArchivedSpacetime:
    """Bicubic-space, cubic-Hermite-time interpolator for regular SO(3) data."""

    def __init__(self, times, z, r, position, velocity, cache_size=160):
        self.times = np.asarray(times, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.r = np.asarray(r, dtype=float)
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)
        expected = (len(self.times), len(self.z), len(self.r), 9)
        if (
            self.position.shape != expected or self.velocity.shape != expected
            or len(self.times) < 2 or np.any(np.diff(self.times) <= 0)
            or np.any(np.diff(self.z) <= 0) or np.any(np.diff(self.r) <= 0)
            or self.r[0] != 0.0
        ):
            raise ValueError("invalid archived regular-SO(3) spacetime")
        if not (
            np.all(np.isfinite(self.position))
            and np.all(np.isfinite(self.velocity))
        ):
            raise ValueError("nonfinite archived spacetime")
        self.cache_size = int(cache_size)
        self._cache = OrderedDict()

    def _spline(self, kind, level, component):
        key = (str(kind), int(level), int(component))
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        source = self.position if kind == "position" else self.velocity
        value = RectBivariateSpline(
            self.z, self.r, source[level, :, :, component],
            kx=min(3, len(self.z) - 1), ky=min(3, len(self.r) - 1), s=0,
        )
        self._cache[key] = value
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value

    def _interval(self, time):
        time = float(time)
        tolerance = 1e-12 * max(1.0, abs(self.times[-1]))
        if time < self.times[0] - tolerance or time > self.times[-1] + tolerance:
            raise ValueError("time leaves archived interval")
        clipped = min(max(time, self.times[0]), self.times[-1])
        index = int(np.searchsorted(self.times, clipped, side="right") - 1)
        index = min(max(index, 0), len(self.times) - 2)
        left = self.times[index]
        right = self.times[index + 1]
        return index, clipped, right - left, (clipped - left) / (right - left)

    @staticmethod
    def _hermite_weights(fraction):
        s = float(fraction)
        values = np.asarray((
            2.0 * s**3 - 3.0 * s**2 + 1.0,
            s**3 - 2.0 * s**2 + s,
            -2.0 * s**3 + 3.0 * s**2,
            s**3 - s**2,
        ))
        derivatives = np.asarray((
            6.0 * s**2 - 6.0 * s,
            3.0 * s**2 - 4.0 * s + 1.0,
            -6.0 * s**2 + 6.0 * s,
            3.0 * s**2 - 2.0 * s,
        ))
        return values, derivatives

    def sample_reduced(self, time, zcoord, radius):
        """Return q and its t/z/r derivatives for components 0--6."""
        zcoord = np.atleast_1d(np.asarray(zcoord, dtype=float))
        radius = np.atleast_1d(np.asarray(radius, dtype=float))
        if zcoord.shape != radius.shape:
            raise ValueError("coordinate arrays differ")
        margin = 1e-10
        if (
            np.min(zcoord) < self.z[0] - margin
            or np.max(zcoord) > self.z[-1] + margin
            or np.min(radius) < self.r[0] - margin
            or np.max(radius) > self.r[-1] + margin
        ):
            raise ValueError("generator leaves interpolation domain")
        sample_z = np.clip(zcoord, self.z[0], self.z[-1])
        sample_r = np.clip(radius, self.r[0], self.r[-1])
        index, _, step, fraction = self._interval(time)
        weights, weight_derivatives = self._hermite_weights(fraction)
        count = len(sample_z)
        q = np.empty((count, 7))
        derivatives = np.empty((count, 3, 7))
        for component in METRIC_COMPONENTS:
            p0 = self._spline("position", index, component)
            v0 = self._spline("velocity", index, component)
            p1 = self._spline("position", index + 1, component)
            v1 = self._spline("velocity", index + 1, component)
            for derivative, dx, dy in ((None, 0, 0), (1, 1, 0), (2, 0, 1)):
                values = np.stack((
                    p0.ev(sample_z, sample_r, dx=dx, dy=dy),
                    step * v0.ev(sample_z, sample_r, dx=dx, dy=dy),
                    p1.ev(sample_z, sample_r, dx=dx, dy=dy),
                    step * v1.ev(sample_z, sample_r, dx=dx, dy=dy),
                ), axis=1)
                interpolated = values @ weights
                if derivative is None:
                    q[:, component] = interpolated
                else:
                    derivatives[:, derivative, component] = interpolated
            endpoint_values = np.stack((
                p0.ev(sample_z, sample_r),
                step * v0.ev(sample_z, sample_r),
                p1.ev(sample_z, sample_r),
                step * v1.ev(sample_z, sample_r),
            ), axis=1)
            derivatives[:, 0, component] = endpoint_values @ weight_derivatives / step
        return q, derivatives

    def metric_and_derivatives(self, time, zcoord, radius):
        """Return 2+1 metric, inverse, and coordinate derivatives."""
        radius = np.atleast_1d(np.asarray(radius, dtype=float))
        q, dq = self.sample_reduced(time, zcoord, radius)
        count = len(radius)
        metric = np.zeros((count, 3, 3))
        metric[:, 0, 0] = q[:, 2]
        metric[:, 0, 1] = metric[:, 1, 0] = q[:, 0]
        metric[:, 0, 2] = metric[:, 2, 0] = radius * q[:, 5]
        metric[:, 1, 1] = q[:, 6]
        metric[:, 1, 2] = metric[:, 2, 1] = radius * q[:, 1]
        metric[:, 2, 2] = q[:, 3] + radius**2 * q[:, 4]
        derivatives = np.zeros((count, 3, 3, 3))
        for direction in range(3):
            derivatives[:, direction, 0, 0] = dq[:, direction, 2]
            derivatives[:, direction, 0, 1] = derivatives[:, direction, 1, 0] = dq[:, direction, 0]
            derivatives[:, direction, 0, 2] = derivatives[:, direction, 2, 0] = radius * dq[:, direction, 5]
            derivatives[:, direction, 1, 1] = dq[:, direction, 6]
            derivatives[:, direction, 1, 2] = derivatives[:, direction, 2, 1] = radius * dq[:, direction, 1]
            derivatives[:, direction, 2, 2] = dq[:, direction, 3] + radius**2 * dq[:, direction, 4]
        derivatives[:, 2, 0, 2] += q[:, 5]
        derivatives[:, 2, 2, 0] += q[:, 5]
        derivatives[:, 2, 1, 2] += q[:, 1]
        derivatives[:, 2, 2, 1] += q[:, 1]
        derivatives[:, 2, 2, 2] += 2.0 * radius * q[:, 4]
        determinants = np.linalg.det(metric)
        if np.any(determinants >= 0.0):
            raise RuntimeError("interpolated metric is not Lorentzian")
        inverse = np.linalg.inv(metric)
        return metric, inverse, derivatives


def christoffel_symbols(metric_inverse, metric_derivatives):
    count = len(metric_inverse)
    connection = np.zeros((count, 3, 3, 3))
    for upper in range(3):
        for left in range(3):
            for right in range(3):
                for contracted in range(3):
                    connection[:, upper, left, right] += 0.5 * metric_inverse[:, upper, contracted] * (
                        metric_derivatives[:, left, contracted, right]
                        + metric_derivatives[:, right, contracted, left]
                        - metric_derivatives[:, contracted, left, right]
                    )
    return connection


def outgoing_surface_velocity(spacetime, time, zcoord, radius, dz_du, dr_du):
    """Coordinate velocity of the future outgoing null normal."""
    metric, _, _ = spacetime.metric_and_derivatives(time, zcoord, radius)
    spatial = metric[:, 1:, 1:]
    spatial_inverse = np.linalg.inv(spatial)
    shift_covector = metric[:, 0, 1:]
    shift = np.einsum("...ab,...b->...a", spatial_inverse, shift_covector)
    lapse_squared = -metric[:, 0, 0] + np.einsum(
        "...a,...a->...", shift_covector, shift,
    )
    if np.any(lapse_squared <= 0.0):
        raise RuntimeError("coordinate-time normal is not timelike")
    tangent = np.stack((dz_du, dr_du), axis=1)
    normal_covector = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
    norm = np.sqrt(np.einsum(
        "...a,...ab,...b->...", normal_covector, spatial_inverse, normal_covector,
    ))
    normal_covector /= norm[:, None]
    normal = np.einsum("...ab,...b->...a", spatial_inverse, normal_covector)
    return -shift + np.sqrt(lapse_squared)[:, None] * normal


def profile_coordinates(z_brane, theta, rho, slope):
    theta = np.asarray(theta, dtype=float)
    rho = np.asarray(rho, dtype=float)
    slope = np.asarray(slope, dtype=float)
    sine = np.sin(theta)
    cosine = np.cos(theta)
    zcoord = float(z_brane) - rho * cosine
    radius = rho * sine
    dz = rho * sine - slope * cosine
    dr = rho * cosine + slope * sine
    return zcoord, radius, dz, dr


def terminal_profile_from_surface(surface, theta, scale=1.0):
    source = np.asarray(surface["theta"], dtype=float)
    rho_spline = CubicSpline(source, np.asarray(surface["rho"], dtype=float))
    theta = np.asarray(theta, dtype=float)
    rho = float(scale) * rho_spline(theta)
    slope = float(scale) * rho_spline(theta, 1)
    return {"theta": theta, "rho": rho, "slope": slope, "scale": float(scale)}


def initialize_terminal_generators(spacetime, time, profile):
    zcoord, radius, dz, dr = profile_coordinates(
        spacetime.z[-1], profile["theta"], profile["rho"], profile["slope"],
    )
    velocity = outgoing_surface_velocity(
        spacetime, time, zcoord, radius, dz, dr,
    )
    return np.stack((zcoord, radius), axis=1), velocity


def _coordinate_geodesic_rhs(spacetime, count):
    def rhs(time, flattened):
        values = np.asarray(flattened).reshape(count, 4)
        position = values[:, :2]
        velocity = values[:, 2:]
        _, inverse, derivatives = spacetime.metric_and_derivatives(
            time, position[:, 0], position[:, 1],
        )
        connection = christoffel_symbols(inverse, derivatives)
        tangent = np.concatenate((np.ones((count, 1)), velocity), axis=1)
        acceleration = np.einsum(
            "...mab,...a,...b->...m", connection, tangent, tangent,
        )
        coordinate_acceleration = -acceleration[:, 1:] + velocity * acceleration[:, :1]
        return np.concatenate((velocity, coordinate_acceleration), axis=1).ravel()
    return rhs


def normalized_null_residual(spacetime, times, positions, velocities):
    values = np.empty((len(times), positions.shape[1]))
    for index, time in enumerate(times):
        metric, _, _ = spacetime.metric_and_derivatives(
            time, positions[index, :, 0], positions[index, :, 1],
        )
        tangent = np.concatenate((
            np.ones((positions.shape[1], 1)), velocities[index],
        ), axis=1)
        numerator = np.abs(np.einsum(
            "...a,...ab,...b->...", tangent, metric, tangent,
        ))
        scale = np.einsum(
            "...a,...ab,...b->...", np.abs(tangent), np.abs(metric), np.abs(tangent),
        )
        values[index] = numerator / np.maximum(scale, 1e-300)
    return values


def integrate_coordinate_generators(
    spacetime, start_time, end_time, position, velocity, output_times=None,
    rtol=2e-9, atol=2e-11, method="DOP853",
):
    """Integrate null generators using coordinate time as the parameter."""
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    if position.shape != velocity.shape or position.ndim != 2 or position.shape[1] != 2:
        raise ValueError("invalid generator state")
    count = len(position)
    if output_times is None:
        output_times = np.asarray((start_time, end_time), dtype=float)
    output_times = np.asarray(output_times, dtype=float)
    direction = np.sign(float(end_time) - float(start_time))
    if direction == 0 or np.any(direction * np.diff(output_times) < 0):
        raise ValueError("output times do not follow integration direction")
    initial = np.concatenate((position, velocity), axis=1).ravel()
    solved = solve_ivp(
        _coordinate_geodesic_rhs(spacetime, count),
        (float(start_time), float(end_time)), initial, t_eval=output_times,
        rtol=float(rtol), atol=float(atol), method=str(method),
    )
    if not solved.success or solved.y.shape[1] != len(output_times):
        raise RuntimeError(f"coordinate geodesic integration failed: {solved.message}")
    values = solved.y.T.reshape(len(output_times), count, 4)
    positions = values[:, :, :2]
    velocities = values[:, :, 2:]
    residual = normalized_null_residual(
        spacetime, output_times, positions, velocities,
    )
    return {
        "success": True,
        "message": solved.message,
        "function_evaluations": int(solved.nfev),
        "times": output_times,
        "positions": positions,
        "velocities": velocities,
        "normalized_null_residual": residual,
        "maximum_normalized_null_residual": float(np.max(residual)),
    }


def _future_pt(inverse, spatial_momentum):
    a = inverse[:, 0, 0]
    b = 2.0 * np.einsum("...i,...i->...", inverse[:, 0, 1:], spatial_momentum)
    c = np.einsum(
        "...ij,...i,...j->...", inverse[:, 1:, 1:], spatial_momentum, spatial_momentum,
    )
    discriminant = b**2 - 4.0 * a * c
    if np.any(discriminant <= 0.0):
        raise RuntimeError("null momentum quadratic has no real future root")
    root = np.sqrt(discriminant)
    candidates = np.stack(((-b + root) / (2.0 * a), (-b - root) / (2.0 * a)), axis=1)
    kt = a[:, None] * candidates + np.einsum(
        "...i,...i->...", inverse[:, 0, 1:], spatial_momentum,
    )[:, None]
    choose = np.argmax(kt, axis=1)
    selected = candidates[np.arange(len(candidates)), choose]
    selected_kt = kt[np.arange(len(kt)), choose]
    if np.any(selected_kt <= 0.0):
        raise RuntimeError("failed to select future null momentum")
    return selected, selected_kt


def spatial_momentum_from_velocity(spacetime, time, position, velocity):
    metric, _, _ = spacetime.metric_and_derivatives(
        time, position[:, 0], position[:, 1],
    )
    tangent = np.concatenate((np.ones((len(position), 1)), velocity), axis=1)
    covector = np.einsum("...ab,...b->...a", metric, tangent)
    return covector[:, 1:]


def _hamiltonian_rhs(spacetime, count):
    def rhs(time, flattened):
        values = np.asarray(flattened).reshape(count, 4)
        position = values[:, :2]
        spatial_momentum = values[:, 2:]
        _, inverse, derivatives = spacetime.metric_and_derivatives(
            time, position[:, 0], position[:, 1],
        )
        pt, kt = _future_pt(inverse, spatial_momentum)
        momentum = np.concatenate((pt[:, None], spatial_momentum), axis=1)
        tangent = np.einsum("...ab,...b->...a", inverse, momentum)
        coordinate_velocity = tangent[:, 1:] / kt[:, None]
        inverse_derivatives = np.empty_like(derivatives)
        for direction in range(3):
            inverse_derivatives[:, direction] = -np.einsum(
                "...ac,...cd,...db->...ab",
                inverse, derivatives[:, direction], inverse,
            )
        momentum_derivative = np.empty_like(spatial_momentum)
        for index in range(2):
            numerator = -0.5 * np.einsum(
                "...a,...ab,...b->...",
                momentum, inverse_derivatives[:, index + 1], momentum,
            )
            momentum_derivative[:, index] = numerator / kt
        return np.concatenate((coordinate_velocity, momentum_derivative), axis=1).ravel()
    return rhs


def integrate_hamiltonian_generators(
    spacetime, start_time, end_time, position, velocity, output_times=None,
    rtol=2e-9, atol=2e-11,
):
    """Independent reduced-Hamiltonian null-geodesic integration."""
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    count = len(position)
    momentum = spatial_momentum_from_velocity(
        spacetime, start_time, position, velocity,
    )
    initial = np.concatenate((position, momentum), axis=1).ravel()
    if output_times is None:
        output_times = np.asarray((start_time, end_time), dtype=float)
    output_times = np.asarray(output_times, dtype=float)
    solved = solve_ivp(
        _hamiltonian_rhs(spacetime, count),
        (float(start_time), float(end_time)), initial, t_eval=output_times,
        rtol=float(rtol), atol=float(atol), method="DOP853",
    )
    if not solved.success or solved.y.shape[1] != len(output_times):
        raise RuntimeError(f"Hamiltonian geodesic integration failed: {solved.message}")
    values = solved.y.T.reshape(len(output_times), count, 4)
    positions = values[:, :, :2]
    velocities = np.empty_like(positions)
    for level, time in enumerate(output_times):
        _, inverse, _ = spacetime.metric_and_derivatives(
            time, positions[level, :, 0], positions[level, :, 1],
        )
        pt, kt = _future_pt(inverse, values[level, :, 2:])
        momentum = np.concatenate((pt[:, None], values[level, :, 2:]), axis=1)
        tangent = np.einsum("...ab,...b->...a", inverse, momentum)
        velocities[level] = tangent[:, 1:] / kt[:, None]
    residual = normalized_null_residual(
        spacetime, output_times, positions, velocities,
    )
    return {
        "success": True,
        "message": solved.message,
        "function_evaluations": int(solved.nfev),
        "times": output_times,
        "positions": positions,
        "velocities": velocities,
        "normalized_null_residual": residual,
        "maximum_normalized_null_residual": float(np.max(residual)),
    }


def polar_coordinates(z_brane, positions):
    positions = np.asarray(positions, dtype=float)
    compact_distance = float(z_brane) - positions[..., 0]
    angle = np.arctan2(positions[..., 1], compact_distance)
    rho = np.sqrt(compact_distance**2 + positions[..., 1]**2)
    return angle, rho


def generator_strip_diagnostics(spacetime, times, positions, core_margin=2):
    """Detect ordering loss/caustics and domain departure in an ordered strip."""
    angle, rho = polar_coordinates(spacetime.z[-1], positions)
    angle_spacing = np.diff(angle, axis=1)
    coordinate_separation = np.linalg.norm(np.diff(positions, axis=1), axis=2)
    core = slice(int(core_margin), positions.shape[1] - int(core_margin))
    core_points = positions[:, core]
    in_domain = (
        (core_points[:, :, 0] >= spacetime.z[0])
        & (core_points[:, :, 0] <= spacetime.z[-1])
        & (core_points[:, :, 1] >= spacetime.r[0])
        & (core_points[:, :, 1] <= spacetime.r[-1])
    )
    minimum_angle_spacing = float(np.min(angle_spacing[:, core_margin:-core_margin]))
    minimum_separation = float(np.min(
        coordinate_separation[:, core_margin:-core_margin]
    ))
    separation_floor = 1e-6 * min(
        np.min(np.diff(spacetime.z)), np.min(np.diff(spacetime.r)),
    )
    caustic = bool(minimum_angle_spacing <= 0.0 or minimum_separation <= separation_floor)
    first = None
    for index, time in enumerate(times):
        local_angles = angle_spacing[index, core_margin:-core_margin]
        local_separation = coordinate_separation[index, core_margin:-core_margin]
        if np.any(local_angles <= 0.0) or np.any(local_separation <= separation_floor):
            first = float(time)
            break
    return {
        "caustic_or_ordering_loss_detected": caustic,
        "first_detected_time_in_integration_order": first,
        "minimum_polar_angle_spacing": minimum_angle_spacing,
        "minimum_neighbor_coordinate_separation": minimum_separation,
        "all_core_generators_in_domain": bool(np.all(in_domain)),
        "minimum_rho": float(np.min(rho[:, core])),
        "maximum_rho": float(np.max(rho[:, core])),
    }


def finite_terminal_classification(z_brane, final_positions, terminal_profile, buffer):
    """Classify endpoints relative to a terminal polar graph.

    ``outside`` and ``inside`` are finite-time relative classes, not escape to
    null infinity or capture by a global black hole.
    """
    angle, rho = polar_coordinates(z_brane, final_positions)
    spline = CubicSpline(
        np.asarray(terminal_profile["theta"]),
        np.asarray(terminal_profile["rho"]), extrapolate=False,
    )
    reference = spline(angle)
    difference = rho - reference
    valid = np.isfinite(reference)
    outside = valid & (difference > float(buffer))
    inside = valid & (difference < -float(buffer))
    unresolved = valid & ~(outside | inside)
    return {
        "outside_count": int(np.count_nonzero(outside)),
        "inside_count": int(np.count_nonzero(inside)),
        "unresolved_count": int(np.count_nonzero(unresolved)),
        "invalid_angle_count": int(np.count_nonzero(~valid)),
        "valid_count": int(np.count_nonzero(valid)),
        "minimum_signed_rho_difference": float(np.nanmin(difference)),
        "maximum_signed_rho_difference": float(np.nanmax(difference)),
        "signed_rho_difference": difference,
    }


def detect_synthetic_caustic(theta):
    theta = np.asarray(theta, dtype=float)
    return bool(np.any(np.diff(theta) <= 0.0))
