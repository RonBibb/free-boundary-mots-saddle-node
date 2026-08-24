"""Floating prototype of the sealed Test-4D shared-node global operator.

Every function in this module is diagnostic unless its output is subsequently
recomputed with directed interval arithmetic and the sealed tail bounds.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.sparse import csr_matrix

from bhps.anisotropic_capped_surface import anisotropic_rho_second
from bhps.capped_surface_barrier_certificate import point_barrier_from_splines
from bhps.validated_global_bvp import chebyshev_lobatto_nodes


def increasing_lobatto_nodes(degree):
    return chebyshev_lobatto_nodes(degree)[::-1]


def lobatto_differentiation_matrix(degree):
    """Derivative with respect to local coordinate, in increasing-node order."""
    nodes = increasing_lobatto_nodes(degree)
    matrix = np.empty((degree + 1, degree + 1))
    for column in range(degree + 1):
        values = np.zeros(degree + 1)
        values[column] = 1.0
        coefficients = np.polynomial.chebyshev.chebfit(nodes, values, degree)
        matrix[:, column] = np.polynomial.chebyshev.chebval(
            nodes, np.polynomial.chebyshev.chebder(coefficients),
        )
    return matrix


def shared_nodal_layout(configuration):
    axis_count = int(configuration["axis_degree"]) + 1
    domains = 140 if configuration["bisect_mesh"] else 70
    degree = int(configuration["bulk_degree"])
    bulk_free_count = domains * degree
    slices = {
        "axis_rho": slice(0, axis_count),
        "axis_u": slice(axis_count, 2 * axis_count),
        "bulk_rho_free": slice(
            2 * axis_count, 2 * axis_count + bulk_free_count,
        ),
        "bulk_w_free": slice(
            2 * axis_count + bulk_free_count,
            2 * axis_count + 2 * bulk_free_count,
        ),
    }
    return {
        "axis_count": axis_count,
        "domains": domains,
        "degree": degree,
        "bulk_free_count": bulk_free_count,
        "size": 2 * axis_count + 2 * bulk_free_count,
        "slices": slices,
    }


def _coefficients_to_increasing_values(coefficients):
    coefficients = np.asarray(coefficients, dtype=float)
    nodes = increasing_lobatto_nodes(coefficients.shape[-1] - 1)
    return np.asarray([
        np.polynomial.chebyshev.chebval(nodes, block)
        for block in np.atleast_2d(coefficients)
    ])


def _shared_values_from_blocks(block_values):
    block_values = np.asarray(block_values, dtype=float)
    output = [block_values[0]]
    output.extend(block[1:] for block in block_values[1:])
    return np.concatenate(output)


def pack_predictor_center_shared(predictor, configuration):
    layout = shared_nodal_layout(configuration)
    axis_rho = _coefficients_to_increasing_values(
        predictor["axis_rho_center"],
    )[0]
    axis_u = _coefficients_to_increasing_values(
        predictor["axis_u_center"],
    )[0]
    rho_blocks = _coefficients_to_increasing_values(
        predictor["rho_blocks_center"],
    )
    w_blocks = _coefficients_to_increasing_values(
        predictor["w_blocks_center"],
    )
    rho_shared = _shared_values_from_blocks(rho_blocks)
    w_shared = _shared_values_from_blocks(w_blocks)
    vector = np.concatenate((
        axis_rho,
        axis_u,
        rho_shared[1:],
        w_shared[1:],
    ))
    if vector.shape != (layout["size"],):
        raise RuntimeError("shared predictor vector has unexpected size")
    return vector


def unpack_shared_vector(vector, configuration, theta_axis=1e-3):
    vector = np.asarray(vector, dtype=float)
    layout = shared_nodal_layout(configuration)
    if vector.shape != (layout["size"],):
        raise ValueError("global vector has unexpected size")
    slices = layout["slices"]
    axis_rho = vector[slices["axis_rho"]]
    axis_u = vector[slices["axis_u"]]
    rho_first = axis_rho[-1]
    w_first = math.sin(float(theta_axis))**3 * axis_u[-1]
    rho_shared = np.concatenate((
        np.asarray([rho_first]), vector[slices["bulk_rho_free"]],
    ))
    w_shared = np.concatenate((
        np.asarray([w_first]), vector[slices["bulk_w_free"]],
    ))
    degree = layout["degree"]
    rho_blocks = np.asarray([
        rho_shared[index * degree:(index + 1) * degree + 1]
        for index in range(layout["domains"])
    ])
    w_blocks = np.asarray([
        w_shared[index * degree:(index + 1) * degree + 1]
        for index in range(layout["domains"])
    ])
    return {
        "axis_rho": axis_rho,
        "axis_u": axis_u,
        "rho_shared": rho_shared,
        "w_shared": w_shared,
        "rho_blocks": rho_blocks,
        "w_blocks": w_blocks,
    }


def _axis_integral_residual(
    axis_rho, axis_u, launch_radius, z_brane, scipy_splines,
    theta_axis=1e-3, quadrature_order=32,
):
    degree = len(axis_rho) - 1
    local_nodes = increasing_lobatto_nodes(degree)
    x_nodes = 0.5 * (local_nodes + 1.0)
    rho_coefficients = np.polynomial.chebyshev.chebfit(
        local_nodes, axis_rho, degree,
    )
    u_coefficients = np.polynomial.chebyshev.chebfit(
        local_nodes, axis_u, degree,
    )
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(
        int(quadrature_order),
    )
    rho_residual = np.empty(degree + 1)
    u_residual = np.empty(degree + 1)
    axis_second = float(launch_radius) * float(np.asarray(
        point_barrier_from_splines(
            0.0, float(launch_radius), z_brane, scipy_splines,
        )
    ).reshape(-1)[0]) / 3.0
    for index, x_value in enumerate(x_nodes):
        theta_end = float(theta_axis) * math.sqrt(max(0.0, x_value))
        if theta_end == 0.0:
            rho_residual[index] = axis_rho[index] - float(launch_radius)
            u_residual[index] = axis_u[index] - axis_second
            continue
        theta = 0.5 * theta_end * (gauss_nodes + 1.0)
        local = 2.0 * (theta / float(theta_axis))**2 - 1.0
        rho = np.polynomial.chebyshev.chebval(local, rho_coefficients)
        u = np.polynomial.chebyshev.chebval(local, u_coefficients)
        sine = np.sin(theta)
        cosine = np.cos(theta)
        slope = sine * u
        second = anisotropic_rho_second(
            theta, rho, slope, z_brane, scipy_splines,
        )
        source = second + 2.0 * cosine * u
        rho_integral = 0.5 * theta_end * np.sum(
            gauss_weights * sine * u
        )
        source_integral = 0.5 * theta_end * np.sum(
            gauss_weights * sine**2 * source
        )
        rho_residual[index] = (
            axis_rho[index] - float(launch_radius) - rho_integral
        )
        u_residual[index] = (
            axis_u[index] - source_integral / math.sin(theta_end)**3
        )
    return rho_residual, u_residual


def floating_global_collocation_residual(
    vector, launch_radius, mesh, configuration, z_brane, scipy_splines,
    theta_axis=1e-3, quadrature_order=32,
):
    """Square shared-node residual at one floating launch parameter."""
    mesh = np.asarray(mesh, dtype=float)
    layout = shared_nodal_layout(configuration)
    if len(mesh) != layout["domains"] + 1:
        raise ValueError("mesh does not match configuration")
    state = unpack_shared_vector(vector, configuration, theta_axis)
    axis_rho_residual, axis_u_residual = _axis_integral_residual(
        state["axis_rho"], state["axis_u"], launch_radius,
        z_brane, scipy_splines, theta_axis, quadrature_order,
    )
    degree = layout["degree"]
    nodes = increasing_lobatto_nodes(degree)
    differentiation = lobatto_differentiation_matrix(degree)
    rho_residual = np.empty((layout["domains"], degree))
    w_residual = np.empty((layout["domains"], degree))
    for index, (left, right) in enumerate(zip(mesh[:-1], mesh[1:])):
        theta = 0.5 * (left + right) + 0.5 * (right - left) * nodes
        scale = 2.0 / (right - left)
        rho = state["rho_blocks"][index]
        w = state["w_blocks"][index]
        derivative_rho = scale * differentiation @ rho
        derivative_w = scale * differentiation @ w
        # Exclude the left endpoint.  Its state is represented once and the
        # right/collocation equations give exactly a square global system.
        selected = slice(1, None)
        local_theta = theta[selected]
        local_rho = rho[selected]
        local_w = w[selected]
        sine = np.sin(local_theta)
        cosine = np.cos(local_theta)
        slope = local_w / sine**2
        second = anisotropic_rho_second(
            local_theta, local_rho, slope, z_brane, scipy_splines,
        )
        rhs_w = sine**2 * (second + 2.0 * cosine * slope / sine)
        rho_residual[index] = derivative_rho[selected] - slope
        w_residual[index] = derivative_w[selected] - rhs_w
    residual = np.concatenate((
        axis_rho_residual,
        axis_u_residual,
        rho_residual.reshape(-1),
        w_residual.reshape(-1),
    ))
    if residual.shape != (layout["size"],):
        raise RuntimeError("global residual is not square")
    return residual


def global_collocation_sparsity(configuration):
    """Conservative exact dependency pattern for colored finite differences."""
    layout = shared_nodal_layout(configuration)
    size = layout["size"]
    axis_count = layout["axis_count"]
    degree = layout["degree"]
    domains = layout["domains"]
    rows = []
    columns = []

    axis_columns = np.arange(2 * axis_count)
    for row in range(2 * axis_count):
        rows.extend([row] * len(axis_columns))
        columns.extend(axis_columns)

    rho_free_start = layout["slices"]["bulk_rho_free"].start
    w_free_start = layout["slices"]["bulk_w_free"].start
    rho_row_start = 2 * axis_count
    w_row_start = rho_row_start + domains * degree
    for block in range(domains):
        local_columns = [axis_count - 1, 2 * axis_count - 1]
        for global_node in range(block * degree, (block + 1) * degree + 1):
            if global_node > 0:
                local_columns.append(rho_free_start + global_node - 1)
                local_columns.append(w_free_start + global_node - 1)
        local_columns = sorted(set(local_columns))
        for offset in range(degree):
            for row in (
                rho_row_start + block * degree + offset,
                w_row_start + block * degree + offset,
            ):
                rows.extend([row] * len(local_columns))
                columns.extend(local_columns)
    return csr_matrix(
        (np.ones(len(rows), dtype=bool), (rows, columns)),
        shape=(size, size),
    )


def collocation_residual_summary(residual, configuration):
    residual = np.asarray(residual, dtype=float)
    layout = shared_nodal_layout(configuration)
    if residual.shape != (layout["size"],):
        raise ValueError("residual has unexpected size")
    axis_count = layout["axis_count"]
    block_count = layout["domains"] * layout["degree"]
    pieces = {
        "axis_rho_integral": residual[:axis_count],
        "axis_u_integral": residual[axis_count:2 * axis_count],
        "bulk_rho_ode": residual[2 * axis_count:2 * axis_count + block_count],
        "bulk_w_ode": residual[2 * axis_count + block_count:],
    }
    return {
        "size": layout["size"],
        "all_finite": bool(np.all(np.isfinite(residual))),
        "maximum_absolute": float(np.max(np.abs(residual))),
        "euclidean_norm": float(np.linalg.norm(residual)),
        "component_maximum_absolute": {
            name: float(np.max(np.abs(values))) for name, values in pieces.items()
        },
    }
