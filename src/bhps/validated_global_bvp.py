"""Core prospective machinery for the Test-4D global BVP certificate."""

from __future__ import annotations

import math

import numpy as np

from bhps.correlated_validated_shooting import (
    build_reference_solution,
    point_divergence_reference_rhs,
)
from bhps.validated_capped_surface_shooting import (
    VInterval,
    as_interval,
    axis_second_interval,
)


CONFIGURATIONS = (
    {
        "name": "D12-M70-P160", "axis_degree": 16,
        "bulk_degree": 12, "bisect_mesh": False, "precision_bits": 160,
    },
    {
        "name": "D16-M70-P160", "axis_degree": 20,
        "bulk_degree": 16, "bisect_mesh": False, "precision_bits": 160,
    },
    {
        "name": "D16-M140-P192", "axis_degree": 20,
        "bulk_degree": 16, "bisect_mesh": True, "precision_bits": 192,
    },
    {
        "name": "D20-M140-P256", "axis_degree": 24,
        "bulk_degree": 20, "bisect_mesh": True, "precision_bits": 256,
    },
)
CHEBYSHEV_WEIGHT = 1.05


def base_nonaxis_mesh():
    geometric = np.asarray([0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064])
    bulk = np.linspace(0.064, math.pi / 2, 65)
    return np.concatenate((geometric, bulk[1:]))


def configuration_mesh(configuration):
    mesh = base_nonaxis_mesh()
    if not configuration["bisect_mesh"]:
        return mesh
    output = [mesh[0]]
    for left, right in zip(mesh[:-1], mesh[1:]):
        output.extend((0.5 * left + 0.5 * right, right))
    return np.asarray(output)


def radius_candidates():
    output = []
    for exponent in range(-14, -3):
        for multiplier in (1.0, 2.0, 5.0):
            value = multiplier * 10.0**exponent
            if value <= 1e-4:
                output.append(value)
    if output[-1] != 1e-4:
        output.append(1e-4)
    return tuple(output)


def chebyshev_l1_nu(coefficients, nu=CHEBYSHEV_WEIGHT):
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 1 or len(coefficients) == 0:
        raise ValueError("Chebyshev coefficients must be a nonempty vector")
    weights = np.ones(len(coefficients))
    if len(coefficients) > 1:
        weights[1:] = 2.0 * float(nu) ** np.arange(1, len(coefficients))
    value = float(np.sum(np.abs(coefficients) * weights))
    return float(np.nextafter(value, np.inf))


def deterministic_component_scales(rho_blocks, u_axis, w_blocks):
    rho_norm = max(chebyshev_l1_nu(block) for block in rho_blocks)
    u_norm = chebyshev_l1_nu(u_axis)
    w_norm = max(chebyshev_l1_nu(block) for block in w_blocks)
    return np.asarray([
        max(1.0, rho_norm), max(1.0, u_norm), max(1e-3, w_norm),
    ])


def chebyshev_lobatto_nodes(degree):
    degree = int(degree)
    if degree < 1:
        raise ValueError("Chebyshev degree must be positive")
    return np.cos(np.pi * np.arange(degree + 1) / degree)


def chebyshev_coefficients_from_lobatto(values):
    values = np.asarray(values, dtype=float)
    degree = len(values) - 1
    nodes = chebyshev_lobatto_nodes(degree)
    return np.polynomial.chebyshev.chebfit(nodes, values, degree)


def evaluate_chebyshev_blocks(coefficients, local_coordinate):
    coefficients = np.asarray(coefficients, dtype=float)
    return np.asarray([
        np.polynomial.chebyshev.chebval(local_coordinate, block)
        for block in coefficients
    ])


def floating_chebyshev_predictor(
    axis_radius, metric, scipy_splines, configuration,
):
    """Deterministic floating center; never itself a certificate."""
    axis_radius = float(axis_radius)
    axis_degree = int(configuration["axis_degree"])
    degree = int(configuration["bulk_degree"])
    mesh = configuration_mesh(configuration)
    solved, _ = build_reference_solution(
        axis_radius, metric.z_brane, scipy_splines, theta_axis=1e-3,
    )

    axis_second = float(
        axis_second_interval(VInterval.point(axis_radius), metric).midpoint
    )
    axis_nodes = chebyshev_lobatto_nodes(axis_degree)
    axis_x = 0.5 * (axis_nodes + 1.0)
    axis_rho_values = axis_radius + 0.5 * axis_second * 1e-6 * axis_x
    axis_u_values = np.full(axis_degree + 1, axis_second)
    axis_rho = chebyshev_coefficients_from_lobatto(axis_rho_values)
    axis_u = chebyshev_coefficients_from_lobatto(axis_u_values)

    lobatto = chebyshev_lobatto_nodes(degree)
    rho_blocks = []
    w_blocks = []
    for left, right in zip(mesh[:-1], mesh[1:]):
        theta = 0.5 * (left + right) + 0.5 * (right - left) * lobatto
        state = solved.sol(theta)
        rho_blocks.append(chebyshev_coefficients_from_lobatto(state[0]))
        w_blocks.append(chebyshev_coefficients_from_lobatto(
            np.sin(theta)**2 * state[1]
        ))
    rho_blocks = np.asarray(rho_blocks)
    w_blocks = np.asarray(w_blocks)
    scales = deterministic_component_scales(rho_blocks, axis_u, w_blocks)
    return {
        "axis_radius": axis_radius,
        "axis_second": axis_second,
        "axis_rho": axis_rho,
        "axis_u": axis_u,
        "mesh": mesh,
        "rho_blocks": rho_blocks,
        "w_blocks": w_blocks,
        "component_scales": scales,
        "reference_dense_steps": len(solved.sol.ts) - 1,
    }


def affine_floating_predictor(
    launch_lower, launch_upper, metric, scipy_splines, configuration,
):
    launch_lower = float(launch_lower)
    launch_upper = float(launch_upper)
    launch_midpoint = 0.5 * launch_lower + 0.5 * launch_upper
    predictors = [
        floating_chebyshev_predictor(
            value, metric, scipy_splines, configuration,
        )
        for value in (launch_lower, launch_midpoint, launch_upper)
    ]
    lower, center, upper = predictors
    output = {
        "launch_interval": np.asarray([launch_lower, launch_upper]),
        "launch_midpoint": np.asarray(launch_midpoint),
        "launch_halfwidth": np.asarray(0.5 * (launch_upper - launch_lower)),
        "mesh": center["mesh"],
        "component_scales": center["component_scales"],
        "reference_dense_steps": np.asarray([
            item["reference_dense_steps"] for item in predictors
        ]),
    }
    for key in ("axis_rho", "axis_u", "rho_blocks", "w_blocks"):
        output[key + "_center"] = center[key]
        output[key + "_parameter"] = 0.5 * (upper[key] - lower[key])
        output[key + "_affine_endpoint_defect"] = np.maximum(
            np.abs((center[key] - output[key + "_parameter"]) - lower[key]),
            np.abs((center[key] + output[key + "_parameter"]) - upper[key]),
        )
    return output


def predictor_diagnostics(predictor):
    diagnostics = {}
    for component in ("axis_rho", "axis_u", "rho_blocks", "w_blocks"):
        coefficients = np.asarray(predictor[component + "_center"])
        blocks = coefficients[None, :] if coefficients.ndim == 1 else coefficients
        weighted_norms = [chebyshev_l1_nu(block) for block in blocks]
        tail_norms = [
            chebyshev_l1_nu(np.concatenate((
                np.zeros(max(0, len(block) - 3)), block[-3:],
            ))) for block in blocks
        ]
        diagnostics[component] = {
            "maximum_weighted_norm": max(weighted_norms),
            "maximum_last_three_weighted_norm": max(tail_norms),
            "maximum_tail_ratio": max(
                tail / total if total > 0.0 else 0.0
                for tail, total in zip(tail_norms, weighted_norms)
            ),
            "maximum_affine_endpoint_defect": float(np.max(
                predictor[component + "_affine_endpoint_defect"]
            )),
        }
    mesh = predictor["mesh"]
    rho = predictor["rho_blocks_center"]
    w = predictor["w_blocks_center"]
    rho_jumps = []
    w_jumps = []
    for index in range(len(mesh) - 2):
        rho_jumps.append(abs(
            np.polynomial.chebyshev.chebval(-1.0, rho[index + 1])
            - np.polynomial.chebyshev.chebval(1.0, rho[index])
        ))
        w_jumps.append(abs(
            np.polynomial.chebyshev.chebval(-1.0, w[index + 1])
            - np.polynomial.chebyshev.chebval(1.0, w[index])
        ))
    diagnostics["maximum_internal_continuity_jump"] = {
        "rho": max(rho_jumps, default=0.0),
        "w": max(w_jumps, default=0.0),
    }
    theta_axis = float(mesh[0])
    axis_rho_right = np.polynomial.chebyshev.chebval(
        1.0, predictor["axis_rho_center"],
    )
    axis_u_right = np.polynomial.chebyshev.chebval(
        1.0, predictor["axis_u_center"],
    )
    diagnostics["axis_to_divergence_continuity_jump"] = {
        "rho": abs(
            np.polynomial.chebyshev.chebval(-1.0, rho[0])
            - axis_rho_right
        ),
        "w": abs(
            np.polynomial.chebyshev.chebval(-1.0, w[0])
            - math.sin(theta_axis)**3 * axis_u_right
        ),
    }
    return diagnostics


def floating_offnode_residual_diagnostics(
    predictor, z_brane, scipy_splines, samples_per_domain=None,
):
    """Dense floating defect screen; never a rigorous interval bound."""
    mesh = np.asarray(predictor["mesh"], dtype=float)
    rho_blocks = np.asarray(predictor["rho_blocks_center"], dtype=float)
    w_blocks = np.asarray(predictor["w_blocks_center"], dtype=float)
    degree = rho_blocks.shape[1] - 1
    sample_count = int(samples_per_domain or max(33, 2 * degree + 1))
    if sample_count < degree + 2:
        raise ValueError("off-node sample count must exceed polynomial degree")
    local = np.cos(
        np.pi * (np.arange(sample_count, dtype=float) + 0.5) / sample_count
    )
    maximum = np.zeros(2)
    radial_minimum = math.inf
    radial_maximum = -math.inf
    affine_radial_minimum = math.inf
    affine_radial_maximum = -math.inf
    z_minimum = math.inf
    z_maximum = -math.inf
    affine_z_minimum = math.inf
    affine_z_maximum = -math.inf
    r_minimum = math.inf
    r_maximum = -math.inf
    affine_r_minimum = math.inf
    affine_r_maximum = -math.inf
    for index, (left, right) in enumerate(zip(mesh[:-1], mesh[1:])):
        center = 0.5 * (left + right)
        halfwidth = 0.5 * (right - left)
        theta = center + halfwidth * local
        rho = np.polynomial.chebyshev.chebval(local, rho_blocks[index])
        w = np.polynomial.chebyshev.chebval(local, w_blocks[index])
        rho_parameter = np.polynomial.chebyshev.chebval(
            local, predictor["rho_blocks_parameter"][index],
        )
        rho_defect = float(np.sum(np.abs(
            predictor["rho_blocks_affine_endpoint_defect"][index]
        )))
        rho_radius = np.abs(rho_parameter) + rho_defect
        rho_lower = rho - rho_radius
        rho_upper = rho + rho_radius
        rho_prime = np.polynomial.chebyshev.chebval(
            local, np.polynomial.chebyshev.chebder(rho_blocks[index]),
        ) / halfwidth
        w_prime = np.polynomial.chebyshev.chebval(
            local, np.polynomial.chebyshev.chebder(w_blocks[index]),
        ) / halfwidth
        for point_index in range(sample_count):
            rhs = point_divergence_reference_rhs(
                theta[point_index],
                np.asarray([rho[point_index], w[point_index]]),
                z_brane,
                scipy_splines,
            )
            maximum = np.maximum(
                maximum,
                np.abs(
                    np.asarray([rho_prime[point_index], w_prime[point_index]])
                    - rhs
                ),
            )
        z_coordinate = float(z_brane) - rho * np.cos(theta)
        r_coordinate = rho * np.sin(theta)
        affine_z_lower = float(z_brane) - rho_upper * np.cos(theta)
        affine_z_upper = float(z_brane) - rho_lower * np.cos(theta)
        affine_r_lower = rho_lower * np.sin(theta)
        affine_r_upper = rho_upper * np.sin(theta)
        radial_minimum = min(radial_minimum, float(np.min(rho)))
        radial_maximum = max(radial_maximum, float(np.max(rho)))
        affine_radial_minimum = min(
            affine_radial_minimum, float(np.min(rho_lower)),
        )
        affine_radial_maximum = max(
            affine_radial_maximum, float(np.max(rho_upper)),
        )
        z_minimum = min(z_minimum, float(np.min(z_coordinate)))
        z_maximum = max(z_maximum, float(np.max(z_coordinate)))
        affine_z_minimum = min(
            affine_z_minimum, float(np.min(affine_z_lower)),
        )
        affine_z_maximum = max(
            affine_z_maximum, float(np.max(affine_z_upper)),
        )
        r_minimum = min(r_minimum, float(np.min(r_coordinate)))
        r_maximum = max(r_maximum, float(np.max(r_coordinate)))
        affine_r_minimum = min(
            affine_r_minimum, float(np.min(affine_r_lower)),
        )
        affine_r_maximum = max(
            affine_r_maximum, float(np.max(affine_r_upper)),
        )
    terminal_center = float(np.polynomial.chebyshev.chebval(
        1.0, predictor["w_blocks_center"][-1],
    ))
    terminal_parameter = float(np.polynomial.chebyshev.chebval(
        1.0, predictor["w_blocks_parameter"][-1],
    ))
    terminal_defect = float(np.sum(np.abs(
        predictor["w_blocks_affine_endpoint_defect"][-1]
    )))
    return {
        "status": "floating_diagnostic_only_not_a_certificate",
        "samples_per_domain": sample_count,
        "maximum_absolute_ode_residual": {
            "rho_equation": float(maximum[0]),
            "w_equation": float(maximum[1]),
        },
        "sampled_ranges": {
            "rho": [radial_minimum, radial_maximum],
            "z": [z_minimum, z_maximum],
            "r": [r_minimum, r_maximum],
        },
        "floating_affine_sampled_ranges": {
            "rho": [affine_radial_minimum, affine_radial_maximum],
            "z": [affine_z_minimum, affine_z_maximum],
            "r": [affine_r_minimum, affine_r_maximum],
        },
        "floating_terminal_affine_envelope": [
            terminal_center - abs(terminal_parameter) - terminal_defect,
            terminal_center + abs(terminal_parameter) + terminal_defect,
        ],
        "terminal_center": terminal_center,
        "terminal_parameter": terminal_parameter,
        "terminal_affine_endpoint_defect_bound": terminal_defect,
    }


def _interval_matrix_point(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return np.asarray([
        [VInterval.point(matrix[i, j]) for j in range(matrix.shape[1])]
        for i in range(matrix.shape[0])
    ], dtype=object)


def _interval_matrix_multiply(left, right):
    left = np.asarray(left, dtype=object)
    right = np.asarray(right, dtype=object)
    output = []
    for row in range(left.shape[0]):
        values = []
        for column in range(right.shape[1]):
            total = VInterval.point(0.0)
            for inner in range(left.shape[1]):
                total = total + left[row, inner] * right[inner, column]
            values.append(total)
        output.append(values)
    return np.asarray(output, dtype=object)


def _interval_matrix_vector(matrix, vector):
    matrix = np.asarray(matrix, dtype=object)
    vector = [as_interval(value) for value in vector]
    return [
        sum(
            (matrix[row, column] * vector[column]
             for column in range(len(vector))),
            VInterval.point(0.0),
        )
        for row in range(matrix.shape[0])
    ]


def scaled_interval_matrix_norm(matrix, scales):
    matrix = np.asarray(matrix, dtype=object)
    scales = np.asarray(scales, dtype=float)
    return max(
        sum(
            as_interval(matrix[row, column]).magnitude
            * scales[column] / scales[row]
            for column in range(matrix.shape[1])
        )
        for row in range(matrix.shape[0])
    )


def finite_radii_bounds(
    approximate_inverse, approximate_jacobian, interval_jacobian,
    residual, scales, z20=0.0, z21=0.0,
):
    """Directed finite-dimensional Y/Z bounds for manufactured controls."""
    approximate_inverse = np.asarray(approximate_inverse, dtype=float)
    approximate_jacobian = np.asarray(approximate_jacobian, dtype=float)
    interval_jacobian = np.asarray(interval_jacobian, dtype=object)
    scales = np.asarray(scales, dtype=float)
    if approximate_inverse.shape != approximate_jacobian.shape:
        raise ValueError("inverse and Jacobian shapes differ")
    size = approximate_inverse.shape[0]
    if interval_jacobian.shape != (size, size) or scales.shape != (size,):
        raise ValueError("interval Jacobian or scale shape mismatch")
    inverse_interval = _interval_matrix_point(approximate_inverse)
    correction = _interval_matrix_vector(inverse_interval, residual)
    Y = max(correction[index].magnitude / scales[index] for index in range(size))

    identity_error = np.eye(size) - approximate_inverse @ approximate_jacobian
    z0_rows = np.sum(
        np.abs(identity_error) * scales[np.newaxis, :] / scales[:, np.newaxis],
        axis=1,
    )
    Z0 = float(np.nextafter(np.max(z0_rows), np.inf))

    difference = np.asarray([
        [VInterval.point(approximate_jacobian[i, j]) - interval_jacobian[i, j]
         for j in range(size)]
        for i in range(size)
    ], dtype=object)
    inverse_difference = _interval_matrix_multiply(inverse_interval, difference)
    Z1 = float(np.nextafter(
        scaled_interval_matrix_norm(inverse_difference, scales), np.inf,
    ))
    return {
        "Y": float(np.nextafter(Y, np.inf)),
        "Z0": Z0,
        "Z1": Z1,
        "z20": float(np.nextafter(float(z20), np.inf)),
        "z21": float(np.nextafter(float(z21), np.inf)),
    }


def radius_polynomial(bounds, radius):
    radius = float(radius)
    return (
        float(bounds["Y"])
        + (float(bounds["Z0"]) + float(bounds["Z1"]) - 1.0) * radius
        + float(bounds["z20"]) * radius**2
        + float(bounds["z21"]) * radius**3
    )


def contraction_bound(bounds, radius):
    radius = float(radius)
    return (
        float(bounds["Z0"]) + float(bounds["Z1"])
        + (float(bounds["z20"]) + float(bounds["z21"]) * radius) * radius
    )


def first_certified_radius(bounds):
    for radius in radius_candidates():
        polynomial = float(np.nextafter(radius_polynomial(bounds, radius), np.inf))
        contraction = float(np.nextafter(contraction_bound(bounds, radius), np.inf))
        if polynomial < 0.0 and contraction < 1.0:
            return {
                "radius": radius,
                "radius_polynomial_upper": polynomial,
                "contraction_upper": contraction,
            }
    return None


def grade_test4d(
    controls_pass, g9_leaves, g10_leaves, unique_control_roots,
    unresolved_control_leaves, independently_confirmed_a790_root=False,
):
    a790_leaves = list(g9_leaves) + list(g10_leaves)
    if (
        independently_confirmed_a790_root
        and any(item.get("classification") == "validated_root"
                for item in a790_leaves)
    ):
        return "FAIL"
    if (
        controls_pass and a790_leaves
        and all(item.get("classification") == "root_free_positive"
                for item in a790_leaves)
        and int(unique_control_roots) == 2
        and int(unresolved_control_leaves) == 0
    ):
        return "PASS"
    return "REVIEW"
