"""Finite-difference Frechet linearization of evolved capped MOTSs.

The operator constructed here is

    L f = d/d epsilon theta_+[S moved by epsilon f s] at epsilon=0,

where ``s`` is the outward unit spatial normal.  This is the generally
non-self-adjoint MOTS stability operator on a fixed evolved ADM slice; it is
not the time-symmetric area Hessian.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import eig

from bhps.dynamical_capped_horizon import prepare_capped_expansion_slice
from bhps.dynamical_capped_horizon_bvp import local_outgoing_expansion


def _finite_difference_weights(x0, stencil, derivative):
    offsets = np.asarray(stencil, dtype=float) - float(x0)
    count = len(offsets)
    matrix = np.vstack([offsets**power for power in range(count)])
    target = np.zeros(count)
    target[int(derivative)] = math.factorial(int(derivative))
    return np.linalg.solve(matrix, target)


def finite_difference_matrix(grid, derivative, stencil_width=7):
    """Polynomial finite-difference matrix on a one-dimensional grid."""
    grid = np.asarray(grid, dtype=float)
    width = min(int(stencil_width), len(grid))
    if width <= int(derivative) or width < 3:
        raise ValueError("finite-difference stencil is too short")
    if width % 2 == 0:
        width -= 1
    result = np.zeros((len(grid), len(grid)))
    half = width // 2
    for index, value in enumerate(grid):
        start = min(max(index - half, 0), len(grid) - width)
        selected = np.arange(start, start + width)
        result[index, selected] = _finite_difference_weights(
            value, grid[selected], derivative,
        )
    return result


def neumann_extension(first_derivative):
    """Map interior polar-radius values to a full vector with eta'=0 ends."""
    derivative = np.asarray(first_derivative, dtype=float)
    count = derivative.shape[0]
    if derivative.shape != (count, count) or count < 5:
        raise ValueError("invalid first-derivative matrix")
    extension = np.zeros((count, count - 2))
    extension[1:-1] = np.eye(count - 2)
    if abs(derivative[0, 0]) < 1e-14 or abs(derivative[-1, -1]) < 1e-14:
        raise ValueError("endpoint derivative closure is singular")
    extension[0] = -derivative[0, 1:-1] / derivative[0, 0]
    extension[-1] = -derivative[-1, 1:-1] / derivative[-1, -1]
    return extension


def physical_normal_factor(prepared, theta, rho, slope):
    """Return f/delta-rho for a radial-graph displacement."""
    theta = np.asarray(theta, dtype=float)
    rho = np.asarray(rho, dtype=float)
    slope = np.asarray(slope, dtype=float)
    sine = np.sin(theta)
    cosine = np.cos(theta)
    radius = rho * sine
    zcoord = prepared.z[-1] - rho * cosine
    metric_inverse = np.empty((len(theta), 2, 2))
    for left in range(2):
        for right in range(2):
            metric_inverse[:, left, right] = prepared.sample(
                ("base_inverse", left, right), zcoord, radius,
            )
    tangent_coordinate = np.stack((
        rho * sine - slope * cosine,
        rho * cosine + slope * sine,
    ), axis=1)
    normal_covector = np.stack(
        (-tangent_coordinate[:, 1], tangent_coordinate[:, 0]), axis=1,
    )
    norm = np.sqrt(np.einsum(
        "...a,...ab,...b->...",
        normal_covector, metric_inverse, normal_covector,
    ))
    normal_covector /= norm[:, None]
    graph_radial_direction = np.stack((-cosine, sine), axis=1)
    return np.einsum(
        "...a,...a->...", normal_covector, graph_radial_direction,
    )


def _profile_expansion(prepared, theta, rho, first, second):
    return local_outgoing_expansion(prepared, theta, rho, first, second)


def mots_stability_matrix(
    position, velocity, z, r, profile, nodes=65, relative_step=1e-5,
    surface_stencil_width=7, spacetime_stencil_width=7, prepared=None,
):
    """Return a physical-normal finite-difference MOTS stability matrix.

    The unknowns are physical normal-displacement values at interior angular
    nodes.  A high-order endpoint elimination enforces ``(f/w)'=0``, where
    ``w=f/delta-rho`` is the coordinate-to-unit-normal conversion.  The output
    expansion is collocated at the same interior nodes.
    """
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    count = int(nodes)
    if count < 17:
        raise ValueError("stability matrix needs at least 17 angular nodes")
    theta = np.linspace(1e-4, np.pi / 2.0, count)
    source_theta = np.asarray(profile["theta"], dtype=float)
    source_rho = np.asarray(profile["rho"], dtype=float)
    curve = CubicSpline(
        source_theta, source_rho, bc_type=((1, 0.0), (1, 0.0)),
    )
    rho = curve(theta)
    first_matrix = finite_difference_matrix(theta, 1, surface_stencil_width)
    second_matrix = finite_difference_matrix(theta, 2, surface_stencil_width)
    extension = neumann_extension(first_matrix)
    first = first_matrix @ rho
    second = second_matrix @ rho
    prepared = (
        prepare_capped_expansion_slice(
            position, velocity, z, r, spacetime_stencil_width,
        ) if prepared is None else prepared
    )
    normal_factor = physical_normal_factor(prepared, theta, rho, first)
    if np.any(~np.isfinite(normal_factor)) or np.min(normal_factor) <= 0:
        raise RuntimeError("radial graph is not positively oriented to its unit normal")
    # eta=delta rho, while f=w eta.  Columns below correspond to unit nodal f.
    deformation = extension / normal_factor[1:-1][None, :]
    physical_step = float(relative_step) * max(1.0, float(np.mean(rho)))
    matrix = np.empty((count - 2, count - 2))
    for column in range(count - 2):
        eta = deformation[:, column]
        plus_rho = rho + physical_step * eta
        minus_rho = rho - physical_step * eta
        plus = _profile_expansion(
            prepared, theta, plus_rho,
            first_matrix @ plus_rho, second_matrix @ plus_rho,
        )
        minus = _profile_expansion(
            prepared, theta, minus_rho,
            first_matrix @ minus_rho, second_matrix @ minus_rho,
        )
        matrix[:, column] = (
            plus[1:-1] - minus[1:-1]
        ) / (2.0 * physical_step)
    base_expansion = _profile_expansion(
        prepared, theta, rho, first, second,
    )
    values, vectors = eig(matrix)
    order = np.argsort(values.real)
    values = values[order]
    vectors = vectors[:, order]
    principal = values[0]
    vector = vectors[:, 0].astype(complex, copy=True)
    phase_index = int(np.argmax(np.abs(vector)))
    vector *= np.exp(-1j * np.angle(vector[phase_index]))
    real_vector = vector.real
    threshold = 1e-6 * max(float(np.max(np.abs(real_vector))), 1e-300)
    significant = real_vector[np.abs(real_vector) > threshold]
    sign_changes = int(np.count_nonzero(significant[:-1] * significant[1:] < 0))
    return {
        "matrix": matrix,
        "theta": theta,
        "rho": rho,
        "normal_factor": normal_factor,
        "physical_step": physical_step,
        "relative_step": float(relative_step),
        "nodes": count,
        "interior_nodes": count - 2,
        "base_expansion_interior_maximum": float(
            np.max(np.abs(base_expansion[1:-1]))
        ),
        "left_neumann_defect": float(np.max(np.abs(first_matrix[0] @ extension))),
        "right_neumann_defect": float(np.max(np.abs(first_matrix[-1] @ extension))),
        "minimum_normal_factor": float(np.min(normal_factor)),
        "maximum_normal_factor": float(np.max(normal_factor)),
        "principal_eigenvalue_real": float(principal.real),
        "principal_eigenvalue_imaginary": float(principal.imag),
        "principal_eigenfunction_sign_changes": sign_changes,
        "leading_eigenvalues": [
            {"real": float(value.real), "imaginary": float(value.imag)}
            for value in values[:min(6, len(values))]
        ],
        "operator_frobenius_norm": float(np.linalg.norm(matrix)),
        "transpose_asymmetry_ratio": float(
            np.linalg.norm(matrix - matrix.T) / max(np.linalg.norm(matrix), 1e-300)
        ),
        "convention": "L f = delta_(f s) theta_plus; stable iff principal eigenvalue >= 0",
        "boundary_condition": "d(delta rho)/dtheta=0 at axis and compact wall, equivalently d(f/w)/dtheta=0",
    }


def public_stability(record):
    return {
        key: value for key, value in record.items()
        if key not in ("matrix", "theta", "rho", "normal_factor")
    }
