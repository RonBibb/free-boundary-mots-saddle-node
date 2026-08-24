"""Maximal ragged normal/arclength charts for A790 Test 2D.

Each brane-normal geodesic is integrated to its own first native-domain
boundary.  The resulting chart is ragged in proper distance and therefore
does not inherit the shallowest-ray cutoff of a rectangular congruence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import simpson, solve_ivp
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, PchipInterpolator
from scipy.optimize import brentq
from scipy.sparse import kron as sparse_kron
from scipy.sparse import eye as sparse_eye
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

from bhps.invariant_physical_chart import MetricSplines
from bhps.invariant_proper_arclength_chart import brane_arclength


@dataclass(frozen=True)
class RaggedNormalChart:
    normalized_depth: np.ndarray
    arclength: np.ndarray
    native_brane_radius: np.ndarray
    maximum_distance: np.ndarray
    distance: np.ndarray
    z: np.ndarray
    r: np.ndarray
    velocity: np.ndarray
    areal_radius: np.ndarray
    speed_squared: np.ndarray
    jacobian_DS_zr: np.ndarray
    eikonal_qDD: np.ndarray
    endpoint_error_dop853: np.ndarray
    endpoint_error_radau: np.ndarray
    termination_code: np.ndarray

    @property
    def shape(self):
        return self.z.shape


@dataclass(frozen=True)
class RaggedInverseResult:
    distance: np.ndarray
    arclength: np.ndarray
    normalized_depth: np.ndarray
    ray_fraction: np.ndarray
    root_count: np.ndarray
    boundary_margin: np.ndarray
    residual: np.ndarray


def _geodesic_rhs(interpolator, _distance, state):
    state = np.asarray(state)
    position = state[..., :2]
    velocity = state[..., 2:]
    _, christoffel = interpolator.metric_and_christoffel(
        position[..., 0], position[..., 1],
    )
    acceleration = -np.einsum(
        "...abc,...b,...c->...a", christoffel, velocity, velocity, optimize=True,
    )
    return np.concatenate((velocity, acceleration), axis=-1)


def _normal_initial_state(interpolator, brane_z, launch_radius):
    metric = interpolator.evaluate_metric(brane_z, launch_radius)
    inverse = np.linalg.inv(metric)
    velocity = -inverse[:, 0] / math.sqrt(float(inverse[0, 0]))
    return np.asarray([brane_z, launch_radius, velocity[0], velocity[1]], dtype=float)


def _domain_events(z_lower, r_lower, r_upper, include_axis):
    def lower_z(_distance, state):
        return state[0] - z_lower
    lower_z.terminal = True
    lower_z.direction = -1.0

    def lower_r(_distance, state):
        return state[1] - r_lower
    lower_r.terminal = True
    lower_r.direction = -1.0

    def upper_r(_distance, state):
        return r_upper - state[1]
    upper_r.terminal = True
    upper_r.direction = -1.0
    return (lower_z, upper_r) if include_axis else (lower_z, lower_r, upper_r)


def _proper_depth_upper_bound(interpolator, z, launch_radius):
    zz = np.asarray(z, dtype=float)
    metric = interpolator.evaluate_metric(zz, np.full_like(zz, launch_radius))
    line_depth = float(simpson(np.sqrt(metric[..., 0, 0]), x=zz))
    return max(4.0 * line_depth, 2.0 * float(zz[-1] - zz[0]), 1e-3)


def _integrate_ray(
    interpolator, z, r, launch_radius, distance_samples,
    rtol=1e-10, atol=1e-12,
):
    initial = _normal_initial_state(interpolator, z[-1], launch_radius)
    upper = _proper_depth_upper_bound(interpolator, z, launch_radius)
    max_step = upper / max(int(distance_samples) - 1, 1)
    events = _domain_events(z[0], r[0], r[-1], include_axis=launch_radius == r[0])
    rhs = lambda distance, state: _geodesic_rhs(interpolator, distance, state)
    primary = solve_ivp(
        rhs, (0.0, upper), initial, method="DOP853", rtol=rtol, atol=atol,
        max_step=max_step, events=events, dense_output=True,
    )
    if not primary.success or primary.status != 1:
        raise RuntimeError("normal ray did not reach a native boundary")
    event_times = [item[0] if len(item) else math.inf for item in primary.t_events]
    event_index = int(np.argmin(event_times))
    maximum_distance = float(event_times[event_index])
    if not np.isfinite(maximum_distance) or maximum_distance <= 0.0:
        raise RuntimeError("normal ray has invalid boundary distance")
    termination_code = event_index + 1
    sample_distance = np.linspace(0.0, maximum_distance, int(distance_samples))
    sampled = primary.sol(sample_distance).T

    endpoint_references = []
    for method, step_factor in (("DOP853", 0.5), ("Radau", 0.5)):
        solved = solve_ivp(
            rhs, (0.0, maximum_distance), initial, method=method,
            rtol=rtol, atol=atol, max_step=step_factor * max_step,
        )
        if not solved.success:
            raise RuntimeError(f"independent {method} normal ray failed")
        endpoint_references.append(np.asarray(solved.y[:, -1]))
    domain_diagonal = math.hypot(float(z[-1] - z[0]), float(r[-1] - r[0]))
    endpoint_errors = [
        float(np.linalg.norm(reference[:2] - sampled[-1, :2]) / domain_diagonal)
        for reference in endpoint_references
    ]
    return {
        "maximum_distance": maximum_distance,
        "distance": sample_distance,
        "state": sampled,
        "endpoint_error_dop853": endpoint_errors[0],
        "endpoint_error_radau": endpoint_errors[1],
        "termination_code": termination_code,
    }


def _cut_locus_limits(rays, arclength, distance_samples, relative_floor=1e-6):
    """Find the first neighboring-ray orientation loss at common distance."""
    limits = np.asarray([item["maximum_distance"] for item in rays], dtype=float)
    cut = limits.copy()
    guard = 4
    sample_count = max(2 * int(distance_samples) - 1, 17)
    for ray_index in range(len(rays) - 1):
        left, right = rays[ray_index], rays[ray_index + 1]
        maximum = min(limits[ray_index], limits[ray_index + 1])
        common = np.linspace(0.0, maximum, sample_count)
        left_state = PchipInterpolator(left["distance"], left["state"], axis=0)(common)
        right_state = PchipInterpolator(right["distance"], right["state"], axis=0)(common)
        separation = (
            right_state[:, :2] - left_state[:, :2]
        ) / float(arclength[ray_index + 1] - arclength[ray_index])
        velocity = 0.5 * (left_state[:, 2:] + right_state[:, 2:])
        forward = (
            velocity[:, 0] * separation[:, 1]
            - velocity[:, 1] * separation[:, 0]
        )
        sign = float(np.sign(np.median(forward[:min(5, len(forward))])))
        floor = relative_floor * max(float(np.median(np.abs(forward[:min(5, len(forward))]))), 1e-15)
        invalid = np.where(sign * forward <= floor)[0]
        if len(invalid):
            admitted_index = max(int(invalid[0]) - guard, 1)
            pair_limit = float(common[admitted_index])
            cut[ray_index] = min(cut[ray_index], pair_limit)
            cut[ray_index + 1] = min(cut[ray_index + 1], pair_limit)
    return cut


def _resample_rays_to_limits(rays, limits, distance_samples):
    output = []
    normalized = np.linspace(0.0, 1.0, int(distance_samples))
    for item, limit in zip(rays, limits):
        new = dict(item)
        old_limit = float(item["maximum_distance"])
        new["maximum_distance"] = float(limit)
        new["distance"] = normalized * float(limit)
        new["state"] = PchipInterpolator(
            item["distance"], item["state"], axis=0,
        )(new["distance"])
        if limit < old_limit * (1.0 - 1e-12):
            new["termination_code"] = 4
        output.append(new)
    return output


def _rk4_step(interpolator, distance, state, step):
    k1 = _geodesic_rhs(interpolator, distance, state)
    k2 = _geodesic_rhs(interpolator, distance + 0.5 * step, state + 0.5 * step * k1)
    k3 = _geodesic_rhs(interpolator, distance + 0.5 * step, state + 0.5 * step * k2)
    k4 = _geodesic_rhs(interpolator, distance + step, state + step * k3)
    return state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _preliminary_boundaries(interpolator, z, r, launch, distance_samples):
    """Bracket each first boundary with a shared high-resolution RK4 sweep."""
    initial = np.stack([
        _normal_initial_state(interpolator, z[-1], value) for value in launch
    ])
    metric_line = interpolator.evaluate_metric(
        np.broadcast_to(z[:, None], (len(z), len(launch))),
        np.broadcast_to(launch[None, :], (len(z), len(launch))),
    )
    line_depth = simpson(np.sqrt(metric_line[..., 0, 0]), x=z, axis=0)
    upper = max(4.0 * float(np.max(line_depth)), 2.0 * float(z[-1] - z[0]), 1e-3)
    steps = max(8 * (int(distance_samples) - 1), 256)
    step = upper / steps
    state = initial.copy()
    active = np.ones(len(launch), dtype=bool)
    lower = np.full(len(launch), np.nan)
    upper_bracket = np.full(len(launch), np.nan)
    code = np.zeros(len(launch), dtype=int)
    for index in range(steps):
        if not np.any(active):
            break
        prior = state.copy()
        candidate = _rk4_step(interpolator, index * step, state, step)
        crossed = np.zeros(len(launch), dtype=bool)
        fractions = np.full((len(launch), 3), np.inf)
        z_cross = active & (candidate[:, 0] <= z[0])
        fractions[z_cross, 0] = (
            (prior[z_cross, 0] - z[0])
            / np.maximum(prior[z_cross, 0] - candidate[z_cross, 0], 1e-300)
        )
        r_low = active & (launch > r[0]) & (candidate[:, 1] <= r[0])
        fractions[r_low, 1] = (
            (prior[r_low, 1] - r[0])
            / np.maximum(prior[r_low, 1] - candidate[r_low, 1], 1e-300)
        )
        r_high = active & (candidate[:, 1] >= r[-1])
        fractions[r_high, 2] = (
            (r[-1] - prior[r_high, 1])
            / np.maximum(candidate[r_high, 1] - prior[r_high, 1], 1e-300)
        )
        event = np.argmin(fractions, axis=1)
        fraction = fractions[np.arange(len(launch)), event]
        crossed = active & np.isfinite(fraction)
        lower[crossed] = index * step
        upper_bracket[crossed] = (index + 1) * step
        code[crossed] = event[crossed] + 1
        state[active] = candidate[active]
        active[crossed] = False
    if np.any(active):
        raise RuntimeError("preliminary ragged rays did not reach a native boundary")
    return initial, lower, upper_bracket, code, step


def _boundary_value(state, code, z, r):
    if code == 1:
        return float(state[0] - z[0])
    if code == 2:
        return float(state[1] - r[0])
    if code == 3:
        return float(r[-1] - state[1])
    raise ValueError("unknown native-boundary code")


def _batched_rays(
    interpolator, z, r, launch, distance_samples, rtol, atol, batch_size=32,
):
    """Integrate uncoupled ray blocks with three independent adaptive solves."""
    initial, lower, upper, codes, bracket_step = _preliminary_boundaries(
        interpolator, z, r, launch, distance_samples,
    )
    order = np.argsort(upper)
    rays = [None] * len(launch)
    domain_diagonal = math.hypot(float(z[-1] - z[0]), float(r[-1] - r[0]))
    for block_start in range(0, len(order), int(batch_size)):
        members = order[block_start:block_start + int(batch_size)]
        block_initial = initial[members]
        end = float(np.max(upper[members]) + 2.0 * bracket_step)

        def rhs(distance, flat_state):
            state = np.asarray(flat_state).reshape(len(members), 4)
            return _geodesic_rhs(interpolator, distance, state).ravel()

        solves = []
        for method, factor in (("DOP853", 1.0), ("DOP853", 0.5), ("Radau", 0.5)):
            options = {}
            if method == "Radau":
                options["jac_sparsity"] = sparse_kron(
                    sparse_eye(len(members), format="csr"),
                    csr_matrix(np.ones((4, 4))), format="csr",
                )
            solved = solve_ivp(
                rhs, (0.0, end), block_initial.ravel(), method=method,
                rtol=rtol, atol=atol,
                max_step=factor * bracket_step, dense_output=True, **options,
            )
            if not solved.success:
                raise RuntimeError(f"batched independent {method} normal rays failed")
            solves.append(solved)

        for local, global_index in enumerate(members):
            def boundary_at(distance):
                state = solves[0].sol(distance).reshape(len(members), 4)[local]
                return _boundary_value(state, int(codes[global_index]), z, r)

            lo = max(0.0, float(lower[global_index] - 2.0 * bracket_step))
            hi = min(end, float(upper[global_index] + 2.0 * bracket_step))
            if boundary_at(lo) * boundary_at(hi) > 0.0:
                scan = np.linspace(0.0, end, 257)
                values = np.asarray([boundary_at(value) for value in scan])
                changes = np.where(values[:-1] * values[1:] <= 0.0)[0]
                if not len(changes):
                    raise RuntimeError("adaptive ray did not bracket its native boundary")
                lo, hi = float(scan[changes[0]]), float(scan[changes[0] + 1])
            maximum_distance = float(brentq(boundary_at, lo, hi, xtol=1e-13, rtol=1e-13))
            sample_distance = np.linspace(0.0, maximum_distance, int(distance_samples))
            sampled = np.stack([
                solves[0].sol(value).reshape(len(members), 4)[local]
                for value in sample_distance
            ])
            endpoint = sampled[-1, :2]
            reference = solves[1].sol(maximum_distance).reshape(len(members), 4)[local, :2]
            dop853_error = float(np.linalg.norm(reference - endpoint) / domain_diagonal)
            radau_reference = solves[2].sol(maximum_distance).reshape(len(members), 4)[local, :2]
            radau_error = float(np.linalg.norm(radau_reference - endpoint) / domain_diagonal)
            rays[global_index] = {
                "maximum_distance": maximum_distance,
                "distance": sample_distance, "state": sampled,
                "endpoint_error_dop853": dop853_error,
                "endpoint_error_radau": radau_error,
                "termination_code": int(codes[global_index]),
            }
    return rays


def _chart_differentials(distance, arclength, position, velocity, metric):
    maximum_distance = distance[-1]
    unit_depth = distance[:, 0] / maximum_distance[0]
    normalized = np.broadcast_to(unit_depth[:, None], distance.shape)
    dmax_dS = _local_polynomial_derivative(maximum_distance, arclength, axis=0)
    x_S_at_u = _local_polynomial_derivative(position, arclength, axis=1)
    # D=u*Dmax(S); therefore du/dS at fixed D is -u Dmax'/Dmax.
    transformed = x_S_at_u - velocity * (
        normalized * dmax_dS[None, :]
    )[..., None]
    # Near a kink in D_max(S), differentiating at fixed normalized depth
    # subtracts two individually nonsmooth terms.  Independently evaluate a
    # local six-ray panel at common proper distance and use it wherever six
    # rays actually cover that distance.
    x_S_at_D = _constant_distance_transverse(
        unit_depth, maximum_distance, arclength, position, transformed,
    )
    x_D = velocity
    forward = x_D[..., 0] * x_S_at_D[..., 1] - x_D[..., 1] * x_S_at_D[..., 0]
    jacobian = 1.0 / forward
    chart_metric = np.empty((*position.shape[:2], 2, 2))
    chart_metric[..., 0, 0] = np.einsum(
        "...a,...ab,...b->...", x_D, metric, x_D,
    )
    chart_metric[..., 0, 1] = chart_metric[..., 1, 0] = np.einsum(
        "...a,...ab,...b->...", x_D, metric, x_S_at_D,
    )
    chart_metric[..., 1, 1] = np.einsum(
        "...a,...ab,...b->...", x_S_at_D, metric, x_S_at_D,
    )
    eikonal = np.linalg.inv(chart_metric)[..., 0, 0]
    return jacobian, eikonal


def _local_polynomial_derivative(values, coordinate, axis=0, width=6):
    """Fifth-degree local derivative on a strictly increasing nonuniform axis."""
    values = np.asarray(values, dtype=float)
    coordinate = np.asarray(coordinate, dtype=float)
    if coordinate.ndim != 1 or len(coordinate) < width or np.any(np.diff(coordinate) <= 0.0):
        raise ValueError("invalid coordinate for local polynomial derivative")
    moved = np.moveaxis(values, axis, 0)
    if moved.shape[0] != len(coordinate):
        raise ValueError("derivative coordinate and value axis do not match")
    derivative = np.empty_like(moved)
    for index in range(len(coordinate)):
        start = min(max(index - width // 2, 0), len(coordinate) - width)
        indices = np.arange(start, start + width)
        offsets = coordinate[indices] - coordinate[index]
        vandermonde = np.stack([offsets**power for power in range(width)], axis=0)
        target = np.zeros(width)
        target[1] = 1.0
        weights = np.linalg.solve(vandermonde, target)
        derivative[index] = np.tensordot(weights, moved[indices], axes=(0, 0))
    return np.moveaxis(derivative, 0, axis)


def _derivative_weights(nodes, target):
    offsets = np.asarray(nodes, dtype=float) - float(target)
    width = len(offsets)
    vandermonde = np.stack([offsets**power for power in range(width)], axis=0)
    right = np.zeros(width)
    right[1] = 1.0
    return np.linalg.solve(vandermonde, right)


def _constant_distance_transverse(
    normalized, maximum_distance, arclength, position, fallback, width=6,
):
    result = np.asarray(fallback).copy()
    interpolants = [
        PchipInterpolator(normalized, position[:, ray], axis=0)
        for ray in range(position.shape[1])
    ]
    for depth_index, unit_depth in enumerate(normalized):
        targets = unit_depth * maximum_distance
        for ray_index, target_distance in enumerate(targets):
            eligible = np.where(
                maximum_distance >= target_distance * (1.0 - 2e-13)
            )[0]
            if len(eligible) < width or ray_index not in eligible:
                continue
            location = int(np.searchsorted(eligible, ray_index))
            start = min(max(location - width // 2, 0), len(eligible) - width)
            panel = eligible[start:start + width]
            weights = _derivative_weights(arclength[panel], arclength[ray_index])
            sampled = np.stack([
                interpolants[source](target_distance / maximum_distance[source])
                for source in panel
            ])
            result[depth_index, ray_index] = weights @ sampled
    return result


def build_ragged_normal_chart(
    z, r, metric, sphere_factor, launch_radius_max=6.0,
    ray_count=385, distance_samples=257, rtol=1e-10, atol=1e-12,
):
    """Build a per-ray maximal normal chart without surface information."""
    z, r = np.asarray(z, dtype=float), np.asarray(r, dtype=float)
    metric, sphere_factor = np.asarray(metric, dtype=float), np.asarray(sphere_factor, dtype=float)
    if z.ndim != 1 or r.ndim != 1 or np.any(np.diff(z) <= 0.0) or np.any(np.diff(r) <= 0.0):
        raise ValueError("native chart coordinates must strictly increase")
    if not r[0] <= launch_radius_max < r[-1]:
        raise ValueError("launch-radius cut must lie inside the native domain")
    if int(ray_count) < 6 or int(distance_samples) < 6:
        raise ValueError("ragged chart requires at least six rays and distance samples")
    interpolator = MetricSplines(z, r, metric, sphere_factor)
    launch = np.linspace(r[0], float(launch_radius_max), int(ray_count))
    arclength = brane_arclength(z, r, metric, launch)
    if int(ray_count) > 64:
        rays = _batched_rays(
            interpolator, z, r, launch, distance_samples, rtol, atol,
        )
    else:
        rays = [
            _integrate_ray(
                interpolator, z, r, value, distance_samples, rtol=rtol, atol=atol,
            ) for value in launch
        ]
    cut_limits = _cut_locus_limits(rays, arclength, distance_samples)
    rays = _resample_rays_to_limits(rays, cut_limits, distance_samples)
    normalized = np.linspace(0.0, 1.0, int(distance_samples))
    maximum_distance = np.asarray([item["maximum_distance"] for item in rays])
    distance = normalized[:, None] * maximum_distance[None, :]
    states = np.stack([item["state"] for item in rays], axis=1)
    position, velocity = states[..., :2], states[..., 2:]
    native_metric = interpolator.evaluate_metric(position[..., 0], position[..., 1])
    speed_squared = np.einsum(
        "...a,...ab,...b->...", velocity, native_metric, velocity, optimize=True,
    )
    sphere = interpolator.evaluate_sphere_factor(position[..., 0], position[..., 1])
    if np.min(sphere) <= 0.0:
        raise ValueError("ragged chart sphere factor is nonpositive")
    areal_radius = position[..., 1] * np.sqrt(sphere)
    jacobian, eikonal = _chart_differentials(
        distance, arclength, position, velocity, native_metric,
    )
    return RaggedNormalChart(
        normalized_depth=normalized, arclength=arclength,
        native_brane_radius=launch, maximum_distance=maximum_distance,
        distance=distance, z=position[..., 0], r=position[..., 1],
        velocity=velocity, areal_radius=areal_radius,
        speed_squared=speed_squared, jacobian_DS_zr=jacobian,
        eikonal_qDD=eikonal,
        endpoint_error_dop853=np.asarray([item["endpoint_error_dop853"] for item in rays]),
        endpoint_error_radau=np.asarray([item["endpoint_error_radau"] for item in rays]),
        termination_code=np.asarray([item["termination_code"] for item in rays], dtype=int),
    )


def _signed_triangle_areas(chart):
    points = np.stack((chart.z, chart.r), axis=-1)
    p00, p10 = points[:-1, :-1], points[1:, :-1]
    p01, p11 = points[:-1, 1:], points[1:, 1:]
    cross = lambda a, b: a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    first = cross(p10 - p00, p01 - p00)
    second = cross(p01 - p11, p10 - p11)
    return np.stack((first, second), axis=-1)


def _nonlocal_collision_count(chart, tolerance):
    points = np.column_stack((chart.z.ravel(), chart.r.ravel()))
    pairs = cKDTree(points).query_pairs(float(tolerance), output_type="ndarray")
    if len(pairs) == 0:
        return 0
    shape = chart.z.shape
    left = np.column_stack(np.unravel_index(pairs[:, 0], shape))
    right = np.column_stack(np.unravel_index(pairs[:, 1], shape))
    local = np.all(np.abs(left - right) <= 1, axis=1)
    return int(np.sum(~local))


def _segments_intersect(a, b, c, d, tolerance=1e-14):
    cross = lambda x, y: x[0] * y[1] - x[1] * y[0]
    o1 = cross(b - a, c - a)
    o2 = cross(b - a, d - a)
    o3 = cross(d - c, a - c)
    o4 = cross(d - c, b - c)
    return bool(o1 * o2 < -tolerance and o3 * o4 < -tolerance)


def _nonlocal_segment_intersections(chart):
    points = np.stack((chart.z, chart.r), axis=-1)
    count = 0
    ray_count = points.shape[1]
    bounds = np.stack((np.min(points, axis=0), np.max(points, axis=0)), axis=1)

    # Self-intersections: only nonadjacent segments of the same ray matter.
    for ray in range(ray_count):
        starts, stops = points[:-1, ray], points[1:, ray]
        lengths = np.linalg.norm(stops - starts, axis=1)
        midpoints = 0.5 * (starts + stops)
        candidates = cKDTree(midpoints).query_pairs(
            float(np.max(lengths)), output_type="ndarray",
        )
        for left, right in candidates:
            if abs(int(left) - int(right)) <= 1:
                continue
            if _segments_intersect(starts[left], stops[left], starts[right], stops[right]):
                count += 1

    # Distinct non-neighboring rays are pruned by their complete polyline
    # bounding boxes before segment-level searches.
    for left_ray in range(ray_count - 2):
        left_start, left_stop = points[:-1, left_ray], points[1:, left_ray]
        left_mid = 0.5 * (left_start + left_stop)
        left_length = np.linalg.norm(left_stop - left_start, axis=1)
        for right_ray in range(left_ray + 2, ray_count):
            if (
                bounds[left_ray, 1, 0] < bounds[right_ray, 0, 0]
                or bounds[right_ray, 1, 0] < bounds[left_ray, 0, 0]
                or bounds[left_ray, 1, 1] < bounds[right_ray, 0, 1]
                or bounds[right_ray, 1, 1] < bounds[left_ray, 0, 1]
            ):
                continue
            right_start, right_stop = points[:-1, right_ray], points[1:, right_ray]
            right_mid = 0.5 * (right_start + right_stop)
            right_length = np.linalg.norm(right_stop - right_start, axis=1)
            tree = cKDTree(right_mid)
            radius = float(np.max(left_length) + np.max(right_length))
            for left_index, choices in enumerate(tree.query_ball_point(left_mid, radius)):
                for right_index in choices:
                    if _segments_intersect(
                        left_start[left_index], left_stop[left_index],
                        right_start[right_index], right_stop[right_index],
                    ):
                        count += 1
    return count


def ragged_chart_validity(chart, coarse=False, jacobian_fraction=1e-6):
    arrays = (
        chart.normalized_depth, chart.arclength, chart.maximum_distance,
        chart.distance, chart.z, chart.r, chart.velocity, chart.speed_squared,
        chart.jacobian_DS_zr, chart.eikonal_qDD,
        chart.endpoint_error_dop853, chart.endpoint_error_radau,
    )
    finite = bool(all(np.all(np.isfinite(value)) for value in arrays))
    jacobian = np.asarray(chart.jacobian_DS_zr)
    # Four terminal rows form the prospectively fixed cut-locus/boundary
    # guard collar and are not part of the admitted physical chart.
    interior_jacobian = jacobian[1:-4, 1:-1]
    sign = float(np.sign(np.median(interior_jacobian))) if finite else 0.0
    median = float(np.median(np.abs(interior_jacobian))) if finite else 0.0
    jacobian_floor = jacobian_fraction * median
    jacobian_valid = bool(
        finite and sign != 0.0
        and np.all(sign * interior_jacobian > jacobian_floor)
    )
    areas = _signed_triangle_areas(chart)[:-4]
    area_sign = float(np.sign(np.median(areas))) if finite else 0.0
    median_area = float(np.median(np.abs(areas))) if finite else 0.0
    area_floor = jacobian_fraction * median_area
    orientation_valid = bool(
        finite and area_sign != 0.0 and np.all(area_sign * areas > area_floor)
    )
    error = np.abs(chart.eikonal_qDD - 1.0)[1:-4, 1:-1]
    eikonal_l2 = float(np.sqrt(np.mean(error**2)))
    eikonal_linf = float(np.max(error))
    l2_limit, linf_limit = ((1e-4, 1e-3) if coarse else (5e-5, 5e-4))
    domain_diagonal = math.hypot(
        float(np.max(chart.z) - np.min(chart.z)),
        float(np.max(chart.r) - np.min(chart.r)),
    )
    collision_tolerance = 1e-7 * domain_diagonal
    collision_count = _nonlocal_collision_count(chart, collision_tolerance)
    segment_intersections = _nonlocal_segment_intersections(chart)
    endpoint_maximum = float(max(
        np.max(chart.endpoint_error_dop853), np.max(chart.endpoint_error_radau),
    ))
    speed_error = float(np.max(np.abs(chart.speed_squared - 1.0)))
    monotone = bool(
        chart.normalized_depth[0] == 0.0
        and np.all(np.diff(chart.normalized_depth) > 0.0)
        and chart.arclength[0] == 0.0 and np.all(np.diff(chart.arclength) > 0.0)
        and np.all(chart.maximum_distance > 0.0)
        and np.all(np.diff(chart.distance, axis=0) > 0.0)
    )
    valid = bool(
        finite and monotone and jacobian_valid and orientation_valid
        and eikonal_l2 < l2_limit and eikonal_linf < linf_limit
        and endpoint_maximum < 1e-8 and collision_count == 0
        and segment_intersections == 0
    )
    return {
        "finite": finite, "monotone_coordinates": monotone,
        "jacobian_sign": sign, "minimum_oriented_jacobian": (
            float(np.min(sign * interior_jacobian)) if finite else None
        ), "jacobian_floor": jacobian_floor, "jacobian_valid": jacobian_valid,
        "triangle_orientation_sign": area_sign,
        "minimum_oriented_triangle_area": (
            float(np.min(area_sign * areas)) if finite else None
        ), "triangle_area_floor": area_floor,
        "orientation_valid": orientation_valid,
        "eikonal_L2": eikonal_l2, "eikonal_Linf": eikonal_linf,
        "eikonal_limits": {"L2": l2_limit, "Linf": linf_limit},
        "geodesic_speed_Linf_error": speed_error,
        "independent_endpoint_relative_maximum": endpoint_maximum,
        "nonlocal_collision_count": collision_count,
        "nonlocal_segment_intersection_count": segment_intersections,
        "valid": valid,
    }


def _structured_triangles(chart):
    nu, ns = chart.shape
    node = np.arange(nu * ns).reshape(nu, ns)
    first = np.stack((node[:-1, :-1], node[1:, :-1], node[:-1, 1:]), axis=-1)
    second = np.stack((node[1:, 1:], node[:-1, 1:], node[1:, :-1]), axis=-1)
    return np.concatenate((first.reshape(-1, 3), second.reshape(-1, 3)), axis=0)


def _barycentric_coordinates(point, triangle):
    matrix = np.column_stack((triangle[1] - triangle[0], triangle[2] - triangle[0]))
    determinant = float(np.linalg.det(matrix))
    if determinant == 0.0:
        return None
    uv = np.linalg.solve(matrix, point - triangle[0])
    return np.asarray([1.0 - uv[0] - uv[1], uv[0], uv[1]])


def inverse_ragged_chart(chart, native_z, native_r, candidate_count=64):
    """Globally search structured triangles and return unique inverse roots."""
    qz, qr = np.broadcast_arrays(np.asarray(native_z, dtype=float), np.asarray(native_r, dtype=float))
    queries = np.column_stack((qz.ravel(), qr.ravel()))
    points = np.column_stack((chart.z.ravel(), chart.r.ravel()))
    triangles = _structured_triangles(chart)
    vertices = points[triangles]
    centroids = np.mean(vertices, axis=1)
    tree = cKDTree(centroids)
    count = min(int(candidate_count), len(triangles))
    _, candidates = tree.query(queries, k=count)
    if count == 1:
        candidates = candidates[:, None]
    Dnodes = chart.distance.ravel()
    Snodes = np.broadcast_to(chart.arclength[None, :], chart.shape).ravel()
    Unodes = np.broadcast_to(chart.normalized_depth[:, None], chart.shape).ravel()
    ray_nodes = np.broadcast_to(
        np.arange(chart.shape[1])[None, :], chart.shape,
    ).ravel().astype(float)
    outputs = []
    tolerance = 5e-11
    for query, choices in zip(queries, candidates):
        roots = []
        for triangle_index in np.unique(choices):
            vertex_indices = triangles[int(triangle_index)]
            weights = _barycentric_coordinates(query, points[vertex_indices])
            if weights is None or np.min(weights) < -tolerance or np.max(weights) > 1.0 + tolerance:
                continue
            D = float(weights @ Dnodes[vertex_indices])
            S = float(weights @ Snodes[vertex_indices])
            U = float(weights @ Unodes[vertex_indices])
            ray_fraction = float(weights @ ray_nodes[vertex_indices])
            reconstructed = weights @ points[vertex_indices]
            residual = float(np.linalg.norm(reconstructed - query))
            roots.append((D, S, U, ray_fraction, residual))
        unique = []
        for root in roots:
            if not any(abs(root[0] - prior[0]) < 1e-9 and abs(root[1] - prior[1]) < 1e-9 for prior in unique):
                unique.append(root)
        if len(unique) != 1:
            outputs.append((math.nan, math.nan, math.nan, math.nan, len(unique), -1.0, math.inf))
            continue
        D, S, U, ray_fraction, residual = unique[0]
        iu = U * (chart.shape[0] - 1)
        margin = min(iu, chart.shape[0] - 1 - iu, ray_fraction, chart.shape[1] - 1 - ray_fraction)
        outputs.append((D, S, U, ray_fraction, 1, margin, residual))
    output = np.asarray(outputs, dtype=float)
    shape = qz.shape
    return RaggedInverseResult(
        distance=output[:, 0].reshape(shape), arclength=output[:, 1].reshape(shape),
        normalized_depth=output[:, 2].reshape(shape), ray_fraction=output[:, 3].reshape(shape),
        root_count=output[:, 4].astype(int).reshape(shape),
        boundary_margin=output[:, 5].reshape(shape), residual=output[:, 6].reshape(shape),
    )


def ragged_chart_to_native(chart, target_distance, target_arclength, method="linear"):
    """Evaluate the forward ragged map at paired or gridded `(D,S)` targets."""
    D, S = np.broadcast_arrays(
        np.asarray(target_distance, dtype=float), np.asarray(target_arclength, dtype=float),
    )
    if not (np.all(np.isfinite(D)) and np.all(np.isfinite(S))):
        raise ValueError("ragged-chart targets are nonfinite")
    points = np.column_stack((chart.distance.ravel(), np.broadcast_to(
        chart.arclength[None, :], chart.shape,
    ).ravel()))
    constructor = (
        LinearNDInterpolator if method == "linear"
        else CloughTocher2DInterpolator if method == "cubic" else None
    )
    if constructor is None:
        raise ValueError("ragged forward method must be linear or cubic")
    values = []
    for field in (chart.z, chart.r):
        interpolator = constructor(points, field.ravel(), fill_value=np.nan)
        values.append(interpolator(D.ravel(), S.ravel()).reshape(D.shape))
    if not all(np.all(np.isfinite(value)) for value in values):
        raise ValueError("target leaves the admitted ragged chart")
    return tuple(values)


def ragged_chart_arrays(chart):
    return {
        "normalized_depth": chart.normalized_depth,
        "arclength": chart.arclength,
        "native_brane_radius": chart.native_brane_radius,
        "maximum_distance": chart.maximum_distance,
        "distance": chart.distance,
        "z": chart.z, "r": chart.r, "velocity": chart.velocity,
        "areal_radius": chart.areal_radius, "speed_squared": chart.speed_squared,
        "jacobian_DS_zr": chart.jacobian_DS_zr,
        "eikonal_qDD": chart.eikonal_qDD,
        "endpoint_error_dop853": chart.endpoint_error_dop853,
        "endpoint_error_radau": chart.endpoint_error_radau,
        "termination_code": chart.termination_code,
    }


def load_ragged_chart(path):
    with np.load(path) as archive:
        return RaggedNormalChart(**{
            key: np.asarray(archive[key]) for key in (
                "normalized_depth", "arclength", "native_brane_radius",
                "maximum_distance", "distance", "z", "r", "velocity",
                "areal_radius", "speed_squared", "jacobian_DS_zr",
                "eikonal_qDD", "endpoint_error_dop853",
                "endpoint_error_radau", "termination_code",
            )
        })
