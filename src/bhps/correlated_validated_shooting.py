"""Lohner-style correlated validation for bounded capped-surface shooting."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from mpmath import libmp
from scipy.integrate import solve_ivp

from bhps.anisotropic_capped_surface import anisotropic_rho_second
from bhps.capped_surface_barrier_certificate import point_barrier_from_splines
from bhps.validated_capped_surface_shooting import (
    VInterval,
    as_interval,
    interval_hull,
    interval_to_list,
    regular_axis_cone,
    regularized_divergence_rhs_interval,
    regularized_divergence_rhs_jet,
    regularized_rhs_interval,
    regularized_rhs_jet,
)


EXP_PRECISION = 96


class MatrixTailFailure(RuntimeError):
    """A step cannot satisfy the sealed matrix-tail threshold."""


def _exp_upper(value):
    raw = libmp.from_float(float(value))
    result = libmp.mpf_exp(raw, EXP_PRECISION, libmp.round_ceiling)
    return float(np.nextafter(
        libmp.to_float(result, strict=True, rnd=libmp.round_ceiling),
        float("inf"),
    ))


def interval_matrix_identity(size=2):
    return np.asarray([
        [VInterval.point(1.0 if i == j else 0.0) for j in range(size)]
        for i in range(size)
    ], dtype=object)


def interval_matrix_point(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return np.asarray([
        [VInterval.point(matrix[i, j]) for j in range(matrix.shape[1])]
        for i in range(matrix.shape[0])
    ], dtype=object)


def interval_matrix_add(left, right):
    left = np.asarray(left, dtype=object)
    right = np.asarray(right, dtype=object)
    return np.asarray([
        [left[i, j] + right[i, j] for j in range(left.shape[1])]
        for i in range(left.shape[0])
    ], dtype=object)


def interval_matrix_multiply(left, right):
    left = np.asarray(left, dtype=object)
    right = np.asarray(right, dtype=object)
    output = []
    for i in range(left.shape[0]):
        row = []
        for j in range(right.shape[1]):
            total = VInterval.point(0.0)
            for k in range(left.shape[1]):
                total = total + left[i, k] * right[k, j]
            row.append(total)
        output.append(row)
    return np.asarray(output, dtype=object)


def interval_matrix_vector(matrix, vector):
    matrix = np.asarray(matrix, dtype=object)
    vector = [as_interval(value) for value in vector]
    return [
        sum((matrix[i, j] * vector[j] for j in range(len(vector))),
            VInterval.point(0.0))
        for i in range(matrix.shape[0])
    ]


def matrix_infinity_norm_upper(matrix):
    matrix = np.asarray(matrix, dtype=object)
    return max(
        sum(as_interval(matrix[i, j]).magnitude for j in range(matrix.shape[1]))
        for i in range(matrix.shape[0])
    )


def validated_matrix_exponential(matrix, step, order=28):
    """Outward interval Taylor enclosure of exp(step*matrix)."""
    matrix = np.asarray(matrix, dtype=float)
    step = float(step)
    scaled = interval_matrix_point(matrix * step)
    total = interval_matrix_identity(matrix.shape[0])
    term = interval_matrix_identity(matrix.shape[0])
    for degree in range(1, int(order) + 1):
        term = interval_matrix_multiply(term, scaled)
        term = np.asarray([
            [entry / degree for entry in row] for row in term
        ], dtype=object)
        total = interval_matrix_add(total, term)
    norm = float(np.linalg.norm(matrix * step, ord=np.inf))
    tail = (
        _exp_upper(norm) * norm**(int(order) + 1)
        / math.factorial(int(order) + 1)
    )
    tail = float(np.nextafter(tail, float("inf")))
    if not math.isfinite(tail):
        raise FloatingPointError("nonfinite matrix exponential tail")
    widened = np.asarray([
        [entry + VInterval(-tail, tail) for entry in row]
        for row in total
    ], dtype=object)
    return widened, tail


def _split_bezier(control, value):
    levels = [np.asarray(control, dtype=float)]
    while len(levels[-1]) > 1:
        levels.append(
            (1.0 - float(value)) * levels[-1][:-1]
            + float(value) * levels[-1][1:]
        )
    left = np.asarray([level[0] for level in levels])
    right = np.asarray([level[-1] for level in levels[::-1]])
    return left, right


def _bezier_subcurve(control, lower, upper):
    control = np.asarray(control, dtype=float)
    if upper < 1.0:
        control, _ = _split_bezier(control, upper)
    if lower > 0.0:
        _, control = _split_bezier(control, lower / upper)
    return control


def _bezier_value(control, value):
    left, _ = _split_bezier(control, value)
    return left[-1]


def _outward_hull(values):
    values = np.asarray(values, dtype=float)
    return VInterval(
        float(np.nextafter(np.min(values), -np.inf)),
        float(np.nextafter(np.max(values), np.inf)),
    )


@dataclass(frozen=True)
class HermiteReferenceStep:
    theta: float
    step: float
    control: np.ndarray

    @classmethod
    def from_endpoints(cls, theta, step, left, right, left_rhs, right_rhs):
        theta = float(theta)
        step = float(step)
        left = np.asarray(left, dtype=float)
        right = np.asarray(right, dtype=float)
        left_rhs = np.asarray(left_rhs, dtype=float)
        right_rhs = np.asarray(right_rhs, dtype=float)
        control = np.asarray([
            left,
            left + step * left_rhs / 3.0,
            right - step * right_rhs / 3.0,
            right,
        ])
        return cls(theta, step, control)

    def subpiece(self, lower, upper):
        control = _bezier_subcurve(self.control, lower, upper)
        local_step = self.step * (upper - lower)
        first = 3.0 * np.diff(control, axis=0) / local_step
        second = 2.0 * np.diff(first, axis=0) / local_step
        return {
            "theta": VInterval(
                self.theta + lower * self.step,
                self.theta + upper * self.step,
            ),
            "state": [_outward_hull(control[:, index]) for index in range(2)],
            "first": [_outward_hull(first[:, index]) for index in range(2)],
            "second": [_outward_hull(second[:, index]) for index in range(2)],
        }

    def value(self, fraction):
        return _bezier_value(self.control, float(fraction))

    def derivative(self, fraction):
        derivative_control = 3.0 * np.diff(self.control, axis=0) / self.step
        return _bezier_value(derivative_control, float(fraction))

    def state_hull(self):
        return [_outward_hull(self.control[:, index]) for index in range(2)]


@dataclass(frozen=True)
class ArchivedDop853Reference:
    """Portable numeric representation of SciPy's DOP853 dense output."""

    boundaries: np.ndarray
    initial_states: np.ndarray
    coefficients: np.ndarray

    def __post_init__(self):
        boundaries = np.asarray(self.boundaries, dtype=float)
        initial_states = np.asarray(self.initial_states, dtype=float)
        coefficients = np.asarray(self.coefficients, dtype=float)
        count = len(boundaries) - 1
        if boundaries.ndim != 1 or count < 1:
            raise ValueError("reference boundaries must describe at least one step")
        if initial_states.shape != (count, 2):
            raise ValueError("reference initial states have the wrong shape")
        if coefficients.shape != (count, 7, 2):
            raise ValueError("DOP853 coefficient archive has the wrong shape")
        if not np.all(np.diff(boundaries) > 0.0):
            raise ValueError("reference boundaries must be strictly increasing")
        if not all(np.all(np.isfinite(item)) for item in (
            boundaries, initial_states, coefficients,
        )):
            raise ValueError("reference archive contains nonfinite values")
        object.__setattr__(self, "boundaries", boundaries)
        object.__setattr__(self, "initial_states", initial_states)
        object.__setattr__(self, "coefficients", coefficients)

    @classmethod
    def from_solve(cls, solved):
        interpolants = solved.sol.interpolants
        return cls(
            np.asarray(solved.sol.ts, dtype=float),
            np.asarray([item.y_old for item in interpolants], dtype=float),
            np.asarray([item.F for item in interpolants], dtype=float),
        )

    @classmethod
    def from_archive(cls, archive):
        return cls(
            np.asarray(archive["boundaries"]),
            np.asarray(archive["initial_states"]),
            np.asarray(archive["coefficients"]),
        )

    def archive_payload(self):
        return {
            "boundaries": self.boundaries,
            "initial_states": self.initial_states,
            "coefficients": self.coefficients,
        }

    def value(self, theta):
        theta = float(theta)
        if theta < self.boundaries[0] or theta > self.boundaries[-1]:
            raise ValueError("reference query lies outside the archive")
        index = int(np.searchsorted(self.boundaries, theta, side="right") - 1)
        index = min(max(index, 0), len(self.initial_states) - 1)
        step = self.boundaries[index + 1] - self.boundaries[index]
        fraction = (theta - self.boundaries[index]) / step
        value = np.zeros(2)
        for coefficient_index, coefficient in enumerate(
            reversed(self.coefficients[index])
        ):
            value += coefficient
            if coefficient_index % 2 == 0:
                value *= fraction
            else:
                value *= 1.0 - fraction
        return value + self.initial_states[index]


@dataclass(frozen=True)
class DivergenceReference:
    """View a rho/rho' DOP853 reference in rho/w coordinates."""

    slope_reference: ArchivedDop853Reference

    def value(self, theta):
        state = self.slope_reference.value(theta)
        sine = math.sin(float(theta))
        return np.asarray([state[0], sine**2 * state[1]])


@dataclass(frozen=True)
class AffineErrorSet:
    """A two-dimensional affine error set plus a symmetric box remainder.

    The represented set is ``center + generators @ [-1,1]^m + remainder``.
    Keeping the generator columns across steps is the Lohner wrapping control;
    converting them to a coordinate box after each step would discard the
    regular-axis correlation that Test 4C is intended to retain.
    """

    center: np.ndarray
    generators: np.ndarray
    remainder: np.ndarray

    def __post_init__(self):
        center = np.asarray(self.center, dtype=float)
        generators = np.asarray(self.generators, dtype=float)
        remainder = np.asarray(self.remainder, dtype=float)
        if center.shape != (2,) or remainder.shape != (2,):
            raise ValueError("affine error center and remainder must be length two")
        if generators.ndim != 2 or generators.shape[0] != 2:
            raise ValueError("affine error generators must have shape (2,m)")
        if np.any(remainder < 0.0) or not np.all(np.isfinite(remainder)):
            raise ValueError("affine error remainder must be finite and nonnegative")
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(generators)):
            raise ValueError("affine error data must be finite")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "generators", generators)
        object.__setattr__(self, "remainder", remainder)

    @classmethod
    def from_interval_box(cls, values):
        values = [as_interval(value) for value in values]
        center = np.asarray([value.midpoint for value in values])
        remainder = np.asarray([
            max(value.upper - value.midpoint, value.midpoint - value.lower)
            for value in values
        ])
        return cls(center, np.zeros((2, 0)), remainder)

    @property
    def radius(self):
        return (
            np.abs(self.center)
            + np.sum(np.abs(self.generators), axis=1)
            + self.remainder
        )

    def intervals(self):
        radius = np.sum(np.abs(self.generators), axis=1) + self.remainder
        return [
            VInterval(
                float(np.nextafter(self.center[index] - radius[index], -np.inf)),
                float(np.nextafter(self.center[index] + radius[index], np.inf)),
            )
            for index in range(2)
        ]

    def linear_map(self, matrix):
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (2, 2):
            raise ValueError("affine error map must have shape (2,2)")
        return AffineErrorSet(
            matrix @ self.center,
            matrix @ self.generators,
            np.abs(matrix) @ self.remainder,
        )

    def to_dict(self):
        return {
            "center": self.center.tolist(),
            "generators": self.generators.tolist(),
            "remainder": self.remainder.tolist(),
        }

    @classmethod
    def from_dict(cls, payload):
        generators = np.asarray(payload["generators"], dtype=float)
        if generators.size == 0:
            generators = np.zeros((2, 0))
        return cls(
            np.asarray(payload["center"], dtype=float), generators,
            np.asarray(payload["remainder"], dtype=float),
        )


def regular_axis_affine_error(
    axis_radius, axis_state, axis_audit, reference_initial, theta_axis,
):
    """Retain the exact two-parameter affine structure of the axis cone."""
    axis_radius = as_interval(axis_radius)
    cone = as_interval(axis_audit["cone"])
    reference_initial = np.asarray(reference_initial, dtype=float)
    alpha = 1.0 - math.cos(float(theta_axis))
    sine = math.sin(float(theta_axis))
    axis_midpoint = axis_radius.midpoint
    radial_launch_cells = [
        as_interval(value) for value in axis_audit.get(
            "radial_launch_cells", [axis_radius],
        )
    ]
    radial_images = [
        as_interval(value) for value in axis_audit.get("radial_images", [cone])
    ]
    if len(radial_launch_cells) != len(radial_images):
        raise ValueError("axis radial-cell/image audit length mismatch")
    if len(radial_images) == 1 or axis_radius.lower == axis_radius.upper:
        correlation_slope = 0.0
        intercept = radial_images[0].midpoint
    else:
        left_center = radial_launch_cells[0].midpoint
        right_center = radial_launch_cells[-1].midpoint
        correlation_slope = (
            radial_images[-1].midpoint - radial_images[0].midpoint
        ) / (right_center - left_center)
        intercept = (
            radial_images[0].midpoint
            + correlation_slope * (axis_radius.midpoint - left_center)
        )
    residuals = [
        image - (
            intercept
            + correlation_slope * (cell - axis_radius.midpoint)
        )
        for cell, image in zip(radial_launch_cells, radial_images)
    ]
    residual = interval_hull(residuals)
    residual_midpoint = residual.midpoint
    residual_halfwidth = max(
        residual.upper - residual_midpoint,
        residual_midpoint - residual.lower,
    )
    cone_midpoint = intercept + residual_midpoint
    axis_radius_halfwidth = max(
        axis_radius.upper - axis_midpoint,
        axis_midpoint - axis_radius.lower,
    )
    center = np.asarray([
        axis_midpoint + alpha * cone_midpoint,
        sine * cone_midpoint,
    ]) - reference_initial
    generators = np.asarray([
        [
            (1.0 + alpha * correlation_slope) * axis_radius_halfwidth,
            alpha * residual_halfwidth,
        ],
        [
            sine * correlation_slope * axis_radius_halfwidth,
            sine * residual_halfwidth,
        ],
    ])
    candidate = AffineErrorSet(center, generators, np.zeros(2))
    candidate_intervals = candidate.intervals()
    required = [
        axis_state["rho"] - reference_initial[0],
        axis_state["slope"] - reference_initial[1],
    ]
    padding = np.asarray([
        max(
            0.0,
            candidate_intervals[index].lower - required[index].lower,
            required[index].upper - candidate_intervals[index].upper,
        )
        for index in range(2)
    ])
    padding = np.nextafter(padding, np.inf)
    return AffineErrorSet(center, generators, padding)


def hermite_defect_bound(
    reference, metric, subdivisions=32, coordinate_system="slope",
):
    component_bounds = [0.0, 0.0]
    maximum_derivative_magnitude = 0.0
    for index in range(int(subdivisions)):
        lower = index / int(subdivisions)
        upper = (index + 1) / int(subdivisions)
        piece = reference.subpiece(lower, upper)
        middle = 0.5 * (lower + upper)
        theta_mid = reference.theta + middle * reference.step
        state_mid = reference.value(middle)
        derivative_mid = reference.derivative(middle)
        if coordinate_system == "slope":
            rhs_mid = [
                VInterval.point(state_mid[1]),
                regularized_rhs_interval(
                    VInterval.point(theta_mid),
                    VInterval.point(state_mid[0]),
                    VInterval.point(state_mid[1]), metric,
                ),
            ]
            second_jet = regularized_rhs_jet(
                piece["theta"], piece["state"][0], piece["state"][1], metric,
            )
            rhs_jets = [
                type(second_jet)(piece["state"][1], (
                    VInterval.point(0.0), VInterval.point(0.0),
                    VInterval.point(1.0),
                )),
                second_jet,
            ]
        elif coordinate_system == "divergence":
            rhs_mid = list(regularized_divergence_rhs_interval(
                VInterval.point(theta_mid), VInterval.point(state_mid[0]),
                VInterval.point(state_mid[1]), metric,
            ))
            rhs_jets = list(regularized_divergence_rhs_jet(
                piece["theta"], piece["state"][0], piece["state"][1], metric,
            ))
        else:
            raise ValueError(f"unknown coordinate system: {coordinate_system}")
        midpoint_defect = [
            VInterval.point(derivative_mid[component]) - rhs_mid[component]
            for component in range(2)
        ]
        total_rhs_derivative = [
            jet.derivative[0]
            + jet.derivative[1] * piece["first"][0]
            + jet.derivative[2] * piece["first"][1]
            for jet in rhs_jets
        ]
        defect_derivative = [
            piece["second"][component] - total_rhs_derivative[component]
            for component in range(2)
        ]
        half_width = 0.5 * reference.step / int(subdivisions)
        for component in range(2):
            enclosure = (
                midpoint_defect[component]
                + VInterval(-half_width, half_width)
                * defect_derivative[component]
            )
            component_bounds[component] = max(
                component_bounds[component], enclosure.magnitude,
            )
            maximum_derivative_magnitude = max(
                maximum_derivative_magnitude,
                defect_derivative[component].magnitude,
            )
    return component_bounds, {
        "subdivisions": int(subdivisions),
        "maximum_defect_derivative_magnitude": maximum_derivative_magnitude,
    }


def _jacobian_matrix(theta, first, second, metric, coordinate_system="slope"):
    if coordinate_system == "slope":
        jet = regularized_rhs_jet(theta, first, second, metric)
        zero = VInterval.point(0.0)
        one = VInterval.point(1.0)
        return np.asarray([
            [zero, one],
            [jet.derivative[1], jet.derivative[2]],
        ], dtype=object), (None, jet)
    if coordinate_system == "divergence":
        jets = regularized_divergence_rhs_jet(
            theta, first, second, metric,
        )
        return np.asarray([
            [jets[row].derivative[column] for column in (1, 2)]
            for row in range(2)
        ], dtype=object), jets
    raise ValueError(f"unknown coordinate system: {coordinate_system}")


def _matrix_difference_norm(interval_matrix, point_matrix):
    point_matrix = np.asarray(point_matrix, dtype=float)
    return max(
        sum((interval_matrix[i, j] - point_matrix[i, j]).magnitude
            for j in range(point_matrix.shape[1]))
        for i in range(point_matrix.shape[0])
    )


def _matrix_difference_magnitudes(interval_matrix, point_matrix):
    point_matrix = np.asarray(point_matrix, dtype=float)
    return np.asarray([
        [(interval_matrix[i, j] - point_matrix[i, j]).magnitude
         for j in range(point_matrix.shape[1])]
        for i in range(point_matrix.shape[0])
    ])


def correlated_lohner_step(
    reference, error0, metric, defect_subdivisions=32,
    matrix_order=28, maximum_tube_iterations=12,
    coordinate_system="slope",
):
    """Validate one reference step and propagate its correlated error box."""
    if not isinstance(error0, AffineErrorSet):
        error0 = AffineErrorSet.from_interval_box(error0)
    middle_state = reference.value(0.5)
    middle_theta = reference.theta + 0.5 * reference.step
    middle_jacobian, _ = _jacobian_matrix(
        VInterval.point(middle_theta), VInterval.point(middle_state[0]),
        VInterval.point(middle_state[1]), metric, coordinate_system,
    )
    point_matrix = np.asarray([
        [middle_jacobian[i, j].midpoint for j in range(2)]
        for i in range(2)
    ])
    exponential, exponential_tail = validated_matrix_exponential(
        point_matrix, reference.step, matrix_order,
    )
    if exponential_tail > 1e-18:
        raise MatrixTailFailure(
            "matrix exponential tail exceeds sealed threshold"
        )
    majorant_interval, majorant_tail = validated_matrix_exponential(
        np.abs(point_matrix), reference.step, matrix_order,
    )
    if majorant_tail > 1e-18:
        raise MatrixTailFailure("matrix majorant tail exceeds sealed threshold")
    _, doubled_exponential_tail = validated_matrix_exponential(
        point_matrix, 2.0 * reference.step, matrix_order,
    )
    _, doubled_majorant_tail = validated_matrix_exponential(
        np.abs(point_matrix), 2.0 * reference.step, matrix_order,
    )
    defect, defect_audit = hermite_defect_bound(
        reference, metric, defect_subdivisions, coordinate_system,
    )
    majorant = np.asarray([
        [majorant_interval[i, j].upper for j in range(2)]
        for i in range(2)
    ])
    error0_radius = error0.radius
    defect_radius = np.asarray(defect)
    radius = majorant @ error0_radius + reference.step * majorant @ defect_radius
    radius = np.maximum(1.25 * radius, 1e-15)
    path_hull = reference.state_hull()
    accepted = False
    delta = None
    delta_matrix = None
    computed_radius = None
    for iteration in range(int(maximum_tube_iterations)):
        rho_tube = path_hull[0] + VInterval(-radius[0], radius[0])
        slope_tube = path_hull[1] + VInterval(-radius[1], radius[1])
        jacobian, _ = _jacobian_matrix(
            VInterval(reference.theta, reference.theta + reference.step),
            rho_tube, slope_tube, metric, coordinate_system,
        )
        delta_matrix = _matrix_difference_magnitudes(jacobian, point_matrix)
        fixed_point_matrix = reference.step * majorant @ delta_matrix
        contraction = float(np.linalg.norm(fixed_point_matrix, ord=np.inf))
        if contraction >= 1.0:
            raise RuntimeError("Lohner tube contraction denominator is nonpositive")
        computed_radius = (
            majorant @ error0_radius
            + reference.step * majorant @ (
                defect_radius + delta_matrix @ radius
            )
        )
        if np.all(radius > computed_radius):
            accepted = True
            break
        radius = np.maximum(1.25 * computed_radius, 1.10 * radius)
    if not accepted:
        raise RuntimeError("Lohner tube radius did not close")
    forcing = reference.step * majorant @ (
        defect_radius + delta_matrix @ radius
    )
    exponential_midpoint = np.asarray([
        [exponential[i, j].midpoint for j in range(2)]
        for i in range(2)
    ])
    exponential_radius = np.asarray([
        [max(
            exponential[i, j].upper - exponential[i, j].midpoint,
            exponential[i, j].midpoint - exponential[i, j].lower,
        ) for j in range(2)]
        for i in range(2)
    ])
    center1 = exponential_midpoint @ error0.center
    generators1 = exponential_midpoint @ error0.generators
    linear_remainder = (
        np.abs(exponential_midpoint) @ error0.remainder
        + exponential_radius @ error0_radius
    )
    error1 = AffineErrorSet(
        center1, generators1, linear_remainder + forcing,
    )
    endpoint = reference.value(1.0)
    endpoint_error = error1.intervals()
    second_name = "slope" if coordinate_system == "slope" else "momentum"
    state1 = {
        "rho": VInterval.point(endpoint[0]) + endpoint_error[0],
        second_name: VInterval.point(endpoint[1]) + endpoint_error[1],
    }
    return state1, error1, {
        "defect_component_bounds": defect,
        "defect": defect_audit,
        "point_matrix": point_matrix.tolist(),
        "matrix_exponential": [
            [interval_to_list(exponential[i, j]) for j in range(2)]
            for i in range(2)
        ],
        "matrix_exponential_tail": exponential_tail,
        "majorant_matrix_tail": majorant_tail,
        "doubled_matrix_exponential_tail": doubled_exponential_tail,
        "doubled_majorant_matrix_tail": doubled_majorant_tail,
        "majorant_matrix": majorant.tolist(),
        "jacobian_delta_matrix": delta_matrix.tolist(),
        "jacobian_delta_norm": float(np.linalg.norm(delta_matrix, ord=np.inf)),
        "contraction_product": contraction,
        "tube_radius": radius.tolist(),
        "computed_radius": computed_radius.tolist(),
        "forcing_radius": forcing.tolist(),
        "affine_center": error1.center.tolist(),
        "affine_generators": error1.generators.tolist(),
        "affine_remainder": error1.remainder.tolist(),
        "tube_iterations": iteration + 1,
    }


def build_reference_solution(axis_radius, z_brane, scipy_splines, theta_axis=1e-3):
    axis_radius = float(axis_radius)
    barrier = float(np.asarray(point_barrier_from_splines(
        0.0, axis_radius, z_brane, scipy_splines,
    )).reshape(-1)[0])
    second = axis_radius * barrier / 3.0
    initial = np.asarray([
        axis_radius + 0.5 * second * theta_axis**2,
        second * theta_axis,
    ])

    def rhs(theta, state):
        second_value = float(anisotropic_rho_second(
            np.asarray([theta]), np.asarray([state[0]]),
            np.asarray([state[1]]), z_brane, scipy_splines,
        )[0])
        return np.asarray([state[1], second_value])

    solved = solve_ivp(
        rhs, (float(theta_axis), math.pi / 2), initial,
        method="DOP853", rtol=2e-12, atol=2e-14,
        max_step=2.5e-4, dense_output=True,
    )
    if not solved.success:
        raise RuntimeError(f"reference solve failed: {solved.message}")
    return solved, rhs


def point_reference_rhs(theta, state, z_brane, scipy_splines):
    state = np.asarray(state, dtype=float)
    second_value = float(anisotropic_rho_second(
        np.asarray([theta]), np.asarray([state[0]]),
        np.asarray([state[1]]), z_brane, scipy_splines,
    )[0])
    return np.asarray([state[1], second_value])


def point_divergence_reference_rhs(theta, state, z_brane, scipy_splines):
    theta = float(theta)
    state = np.asarray(state, dtype=float)
    sine = math.sin(theta)
    cosine = math.cos(theta)
    slope = state[1] / sine**2
    second_value = float(anisotropic_rho_second(
        np.asarray([theta]), np.asarray([state[0]]),
        np.asarray([slope]), z_brane, scipy_splines,
    )[0])
    source = second_value + 2.0 * cosine * slope / sine
    return np.asarray([slope, sine**2 * source])


def initialize_correlated_propagation(
    axis_radius, metric, scipy_splines, theta_axis=1e-3,
):
    """Build the reusable reference and JSON-safe initial propagation state."""
    axis_radius = as_interval(axis_radius)
    axis_state, axis_audit = regular_axis_cone(
        axis_radius, metric, theta_axis=theta_axis, theta_subdivisions=128,
    )
    solved, _ = build_reference_solution(
        axis_radius.midpoint, metric.z_brane, scipy_splines, theta_axis,
    )
    slope_reference = ArchivedDop853Reference.from_solve(solved)
    reference_initial = slope_reference.value(theta_axis)
    slope_error = regular_axis_affine_error(
        axis_radius, axis_state, axis_audit, reference_initial, theta_axis,
    )
    sine = math.sin(float(theta_axis))
    error = slope_error.linear_map(np.diag([1.0, sine**2]))
    state = {
        "schema": "test4c-correlated-propagation-v2",
        "coordinate_system": "divergence",
        "classification": "running",
        "axis_radius": interval_to_list(axis_radius),
        "theta_axis": float(theta_axis),
        "theta": float(theta_axis),
        "step": 1e-3,
        "error": error.to_dict(),
        "accepted_steps": 0,
        "step_rejections": 0,
        "accepted_since_rejection": 0,
        "axis": {
            "cone": interval_to_list(axis_audit["cone"]),
            "invariant_cone": interval_to_list(axis_audit["invariant_cone"]),
            "image": interval_to_list(axis_audit["image"]),
            "source": interval_to_list(axis_audit["source"]),
            "iterations": axis_audit["iterations"],
            "theta_subdivisions": axis_audit["theta_subdivisions"],
            "launch_subdivisions": axis_audit["launch_subdivisions"],
            "weighted_ratio": interval_to_list(axis_audit["weighted_ratio"]),
            "coarse_weighted_ratio": interval_to_list(
                axis_audit["coarse_weighted_ratio"]
            ),
            "radial_launch_cells": [
                interval_to_list(value)
                for value in axis_audit["radial_launch_cells"]
            ],
            "radial_images": [
                interval_to_list(value) for value in axis_audit["radial_images"]
            ],
        },
        "audit_summary": {
            "maximum_defect": 0.0,
            "maximum_matrix_tail": 0.0,
            "maximum_tube_radius": float(np.max(error.radius)),
            "maximum_contraction_product": 0.0,
            "maximum_subdivisions": 0,
        },
    }
    return slope_reference, state


def advance_correlated_propagation(
    state, reference, metric, scipy_splines, rho_bounds=(0.10, 1.67),
    accepted_step_budget=16, initial_step=1e-3, minimum_step=1e-6,
    defect_subdivision_schedule=(32, 64, 128, 256),
):
    """Advance a restartable propagation state by a bounded accepted-step chunk."""
    state = {
        **state,
        "axis": dict(state["axis"]),
        "audit_summary": dict(state["audit_summary"]),
    }
    if state["classification"] != "running":
        return state
    coordinate_system = state.get("coordinate_system", "slope")
    propagation_reference = (
        DivergenceReference(reference)
        if coordinate_system == "divergence" else reference
    )
    error = AffineErrorSet.from_dict(state["error"])
    theta = float(state["theta"])
    step = float(state["step"])
    rejections = int(state["step_rejections"])
    accepted_since_rejection = int(state["accepted_since_rejection"])
    accepted_steps = int(state["accepted_steps"])
    accepted_in_chunk = 0
    lower_bound, upper_bound = map(float, rho_bounds)
    last_state = None
    while theta < math.pi / 2 and accepted_in_chunk < int(accepted_step_budget):
        local_step = min(step, math.pi / 2 - theta)
        left = propagation_reference.value(theta)
        right = propagation_reference.value(theta + local_step)
        rhs_function = (
            point_divergence_reference_rhs
            if coordinate_system == "divergence" else point_reference_rhs
        )
        left_rhs = rhs_function(theta, left, metric.z_brane, scipy_splines)
        right_rhs = rhs_function(
            theta + local_step, right, metric.z_brane, scipy_splines,
        )
        reference_step = HermiteReferenceStep.from_endpoints(
            theta, local_step, left, right, left_rhs, right_rhs,
        )
        accepted = None
        failure = None
        for subdivisions in defect_subdivision_schedule:
            try:
                accepted = correlated_lohner_step(
                    reference_step, error, metric,
                    defect_subdivisions=subdivisions, matrix_order=28,
                    coordinate_system=coordinate_system,
                )
                break
            except MatrixTailFailure as error_value:
                failure = error_value
                break
            except (ValueError, ZeroDivisionError, FloatingPointError, RuntimeError) as error_value:
                failure = error_value
        if accepted is None:
            step *= 0.5
            rejections += 1
            accepted_since_rejection = 0
            if step < minimum_step:
                state.update({
                    "classification": "unresolved_correlated_step",
                    "reason": repr(failure),
                })
                break
            continue
        last_state, error, audit = accepted
        theta += local_step
        accepted_steps += 1
        accepted_in_chunk += 1
        if (
            audit["contraction_product"] < 0.25
            and audit["doubled_matrix_exponential_tail"] <= 1e-18
            and audit["doubled_majorant_matrix_tail"] <= 1e-18
        ):
            accepted_since_rejection += 1
        else:
            accepted_since_rejection = 0
        summary = state["audit_summary"]
        summary["maximum_defect"] = max(
            summary["maximum_defect"], *audit["defect_component_bounds"],
        )
        summary["maximum_matrix_tail"] = max(
            summary["maximum_matrix_tail"],
            audit["matrix_exponential_tail"], audit["majorant_matrix_tail"],
        )
        summary["maximum_tube_radius"] = max(
            summary["maximum_tube_radius"], *audit["tube_radius"],
        )
        summary["maximum_contraction_product"] = max(
            summary["maximum_contraction_product"], audit["contraction_product"],
        )
        summary["maximum_subdivisions"] = max(
            summary["maximum_subdivisions"], subdivisions,
        )
        summary["last_contraction_product"] = audit["contraction_product"]
        summary["last_defect_component_bounds"] = audit[
            "defect_component_bounds"
        ]
        summary["last_tube_radius"] = audit["tube_radius"]
        summary["last_doubled_matrix_tail"] = max(
            audit["doubled_matrix_exponential_tail"],
            audit["doubled_majorant_matrix_tail"],
        )
        if (last_state["rho"].lower < lower_bound
                or last_state["rho"].upper > upper_bound):
            state["classification"] = "unresolved_radial_band"
            break
        if accepted_since_rejection >= 4 and step < initial_step:
            step = min(initial_step, 2.0 * step)
            accepted_since_rejection = 0
    if theta >= math.pi / 2:
        if last_state is None:
            endpoint = propagation_reference.value(theta)
            error_interval = error.intervals()
            second_name = (
                "momentum" if coordinate_system == "divergence" else "slope"
            )
            last_state = {
                "rho": VInterval.point(endpoint[0]) + error_interval[0],
                second_name: VInterval.point(endpoint[1]) + error_interval[1],
            }
        residual_name = (
            "momentum" if coordinate_system == "divergence" else "slope"
        )
        if last_state[residual_name].lower > 0.0:
            state["classification"] = "root_free_positive"
        elif last_state[residual_name].upper < 0.0:
            state["classification"] = "root_free_negative"
        else:
            state["classification"] = "zero_containing_residual"
    state.update({
        "theta": theta,
        "step": step,
        "error": error.to_dict(),
        "accepted_steps": accepted_steps,
        "step_rejections": rejections,
        "accepted_since_rejection": accepted_since_rejection,
        "last_chunk_accepted_steps": accepted_in_chunk,
    })
    if last_state is not None:
        state["state"] = {
            key: interval_to_list(value) for key, value in last_state.items()
        }
        if theta >= math.pi / 2:
            state["terminal_residual"] = state["state"][residual_name]
    return state


def correlated_propagate_launch_cell(
    axis_radius, metric, scipy_splines, rho_bounds=(0.10, 1.67),
    theta_axis=1e-3, initial_step=1e-3, minimum_step=1e-6,
    defect_subdivision_schedule=(32, 64, 128, 256), maximum_steps=200000,
):
    axis_radius = as_interval(axis_radius)
    try:
        axis_state, axis_audit = regular_axis_cone(
            axis_radius, metric, theta_axis=theta_axis,
            theta_subdivisions=128,
        )
    except Exception as error:
        return {
            "classification": "unresolved_axis_cone", "reason": repr(error),
            "axis_radius": interval_to_list(axis_radius),
        }
    center = axis_radius.midpoint
    solved, reference_rhs = build_reference_solution(
        center, metric.z_brane, scipy_splines, theta_axis,
    )
    reference_initial = solved.sol(theta_axis)
    error = regular_axis_affine_error(
        axis_radius, axis_state, axis_audit, reference_initial, theta_axis,
    )
    theta = float(theta_axis)
    step = float(initial_step)
    rejections = 0
    audit_summary = {
        "maximum_defect": 0.0, "maximum_matrix_tail": 0.0,
        "maximum_tube_radius": float(np.max(error.radius)),
        "maximum_contraction_product": 0.0,
        "maximum_subdivisions": 0,
    }
    accepted_since_rejection = 0
    lower_bound, upper_bound = map(float, rho_bounds)
    for step_index in range(int(maximum_steps)):
        if theta >= math.pi / 2:
            break
        local_step = min(step, math.pi / 2 - theta)
        left = solved.sol(theta)
        right = solved.sol(theta + local_step)
        left_rhs = reference_rhs(theta, left)
        right_rhs = reference_rhs(theta + local_step, right)
        reference = HermiteReferenceStep.from_endpoints(
            theta, local_step, left, right, left_rhs, right_rhs,
        )
        accepted = None
        failure = None
        for subdivisions in defect_subdivision_schedule:
            try:
                accepted = correlated_lohner_step(
                    reference, error, metric,
                    defect_subdivisions=subdivisions, matrix_order=28,
                )
                break
            except MatrixTailFailure as error_value:
                failure = error_value
                break
            except (ValueError, ZeroDivisionError, FloatingPointError, RuntimeError) as error_value:
                failure = error_value
        if accepted is None:
            step *= 0.5
            rejections += 1
            accepted_since_rejection = 0
            if step < minimum_step:
                return {
                    "classification": "unresolved_correlated_step",
                    "reason": repr(failure), "theta": theta,
                    "axis_radius": interval_to_list(axis_radius),
                    "state": {
                        "rho": interval_to_list(
                            VInterval.point(left[0]) + error.intervals()[0]
                        ),
                        "slope": interval_to_list(
                            VInterval.point(left[1]) + error.intervals()[1]
                        ),
                    },
                    "axis": {
                        "cone": interval_to_list(axis_audit["cone"]),
                        "iterations": axis_audit["iterations"],
                    },
                    "step_rejections": rejections,
                    "audit_summary": audit_summary,
                }
            continue
        state, error, audit = accepted
        theta += local_step
        if (
            audit["contraction_product"] < 0.25
            and audit["doubled_matrix_exponential_tail"] <= 1e-18
            and audit["doubled_majorant_matrix_tail"] <= 1e-18
        ):
            accepted_since_rejection += 1
        else:
            accepted_since_rejection = 0
        audit_summary["maximum_defect"] = max(
            audit_summary["maximum_defect"], *audit["defect_component_bounds"],
        )
        audit_summary["maximum_matrix_tail"] = max(
            audit_summary["maximum_matrix_tail"],
            audit["matrix_exponential_tail"], audit["majorant_matrix_tail"],
        )
        audit_summary["maximum_tube_radius"] = max(
            audit_summary["maximum_tube_radius"], *audit["tube_radius"],
        )
        audit_summary["maximum_contraction_product"] = max(
            audit_summary["maximum_contraction_product"], audit["contraction_product"],
        )
        audit_summary["maximum_subdivisions"] = max(
            audit_summary["maximum_subdivisions"], subdivisions,
        )
        if state["rho"].lower < lower_bound or state["rho"].upper > upper_bound:
            return {
                "classification": "unresolved_radial_band",
                "theta": theta, "axis_radius": interval_to_list(axis_radius),
                "state": {key: interval_to_list(value) for key, value in state.items()},
                "axis": {"cone": interval_to_list(axis_audit["cone"])},
                "step_rejections": rejections, "audit_summary": audit_summary,
            }
        if accepted_since_rejection >= 4 and step < initial_step:
            step = min(initial_step, 2.0 * step)
            accepted_since_rejection = 0
    else:
        return {
            "classification": "unresolved_step_limit", "theta": theta,
            "axis_radius": interval_to_list(axis_radius),
            "step_rejections": rejections, "audit_summary": audit_summary,
        }
    if state["slope"].lower > 0.0:
        classification = "root_free_positive"
    elif state["slope"].upper < 0.0:
        classification = "root_free_negative"
    else:
        classification = "zero_containing_residual"
    return {
        "classification": classification, "theta": theta,
        "axis_radius": interval_to_list(axis_radius),
        "state": {key: interval_to_list(value) for key, value in state.items()},
        "terminal_residual": interval_to_list(state["slope"]),
        "axis": {
            "cone": interval_to_list(axis_audit["cone"]),
            "iterations": axis_audit["iterations"],
        },
        "step_count": step_index, "step_rejections": rejections,
        "audit_summary": audit_summary,
    }
