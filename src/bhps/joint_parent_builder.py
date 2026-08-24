"""Hybrid bulk--wall selector for Protocol 125 joint parent data.

The open compact-direction nodes use the established well-balanced
Hamiltonian and stationary-scalar residual.  Both complete compact walls,
including their radial-axis and radial-outer corners, instead use the
absolute native sphere-metric Israel row and the absolute Phi Robin row.

This module is deliberately isolated from the production parent builders.
It constructs only the nonlinear selector functional and its exact analytic
Jacobian; it does not complete a native state or write result artifacts.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import spsolve

from bhps.anisotropic_initial_data import (
    anisotropic_initial_data_jacobian,
    anisotropic_initial_data_residual,
)
from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.junction_preservation_diagnostic import wall_source_coefficients


def _validated_fields(
    q,
    phi,
    z,
    r,
    a,
    b,
    c,
    chi_r,
    chi_z,
    reference_q,
    reference_phi,
    stencil_width,
):
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    if z.ndim != 1 or r.ndim != 1:
        raise ValueError("z and r must be one-dimensional grids")
    width = int(stencil_width)
    if width != stencil_width or width < 3:
        raise ValueError("stencil_width must be an integer of at least three")
    if len(z) < width or len(r) < width:
        raise ValueError("both grids must support the requested stencil width")
    if np.any(np.diff(z) <= 0.0) or np.any(np.diff(r) <= 0.0):
        raise ValueError("z and r must be strictly increasing")
    shape = (len(z), len(r))
    fields = tuple(
        np.asarray(field, dtype=float)
        for field in (
            q,
            phi,
            a,
            b,
            c,
            chi_r,
            chi_z,
            reference_q,
            reference_phi,
        )
    )
    if any(field.shape != shape for field in fields):
        raise ValueError("all fields must share the z-r grid shape")
    if not all(np.all(np.isfinite(field)) for field in fields):
        raise ValueError("all fields must be finite")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(r)):
        raise ValueError("grid coordinates must be finite")
    q = fields[0]
    if np.any(z[:, None] + q <= 0.0):
        raise ValueError("z + q must remain positive")
    return (q, *fields[1:], z, r, width)


def _wall_primitives(q, phi, z, a, c, background, stencil_width):
    """Return fields shared by the native wall residual and Jacobian."""
    psi = 1.0 / (z[:, None] + q)
    sphere_metric = psi**2 * np.exp(2.0 * c)
    normal_scale = psi * np.exp(a)
    dz = derivative_matrix(z, 1, stencil_width).tocsr()
    dz_sphere = dz @ sphere_metric
    dz_phi = dz @ phi
    walls = []
    for name, index in (("lower", 0), ("upper", len(z) - 1)):
        source = wall_source_coefficients(phi[index], background, name)
        orientation = source["orientation"]
        beta = source["beta"]
        local_scale = normal_scale[index]
        robin = dz_sphere[index] + 2.0 * beta * local_scale * sphere_metric[index]
        junction = orientation * robin / (2.0 * local_scale)
        delta = phi[index] - source["target"]
        phi_robin = (
            dz_phi[index]
            + orientation
            * 0.5
            * float(background["wall_stiffness"])
            * delta
            * local_scale
        )
        walls.append(
            {
                "name": name,
                "index": index,
                "source": source,
                "robin": robin,
                "junction": junction,
                "phi_robin": phi_robin,
            }
        )
    return {
        "psi": psi,
        "sphere_metric": sphere_metric,
        "normal_scale": normal_scale,
        "Dz": dz,
        "walls": walls,
    }


def joint_parent_residual(
    q,
    phi,
    z,
    r,
    a,
    b,
    c,
    background,
    chi_r,
    chi_z,
    reference_q,
    reference_phi,
    stencil_width=7,
):
    """Evaluate the Protocol-125 hybrid selector residual.

    Flattening follows :mod:`bhps.anisotropic_initial_data`: the first block
    is the Hamiltonian/native sphere-metric row and the second is the Phi
    row.  At open compact-direction nodes this is exactly the established
    raw-reference-defect residual.  Each compact wall owns every radial node,
    so the radial outer condition survives only at open-z nodes.
    """
    (
        q,
        phi,
        a,
        b,
        c,
        chi_r,
        chi_z,
        reference_q,
        reference_phi,
        z,
        r,
        stencil_width,
    ) = _validated_fields(
        q,
        phi,
        z,
        r,
        a,
        b,
        c,
        chi_r,
        chi_z,
        reference_q,
        reference_phi,
        stencil_width,
    )
    residual = anisotropic_initial_data_residual(
        q,
        phi,
        z,
        r,
        a,
        b,
        c,
        background,
        chi_r,
        chi_z,
        reference_q,
        reference_phi,
        stencil_width,
    ).copy()
    n = q.size
    metric_rows = residual[:n].reshape(q.shape)
    phi_rows = residual[n:].reshape(q.shape)
    wall_data = _wall_primitives(
        q, phi, z, a, c, background, stencil_width,
    )
    for wall in wall_data["walls"]:
        index = wall["index"]
        metric_rows[index, :] = wall["junction"]
        phi_rows[index, :] = wall["phi_robin"]
    return residual


def _replace_sparse_row(matrix, row, entries):
    """Replace one LIL row from an index-to-value mapping."""
    filtered = sorted(
        (int(index), float(value))
        for index, value in entries.items()
        if value != 0.0
    )
    matrix.rows[row] = [index for index, _ in filtered]
    matrix.data[row] = [value for _, value in filtered]


def joint_parent_jacobian(
    q,
    phi,
    z,
    r,
    a,
    b,
    c,
    background,
    chi_r,
    chi_z,
    reference_q,
    reference_phi,
    stencil_width=7,
):
    """Return the exact analytic Jacobian of :func:`joint_parent_residual`.

    The normalized metric row is differentiated off manifold.  In
    particular, the derivative of ``1/(2*sqrt(g_zz))`` multiplies the full
    (generally nonzero) Robin numerator; it is not dropped or replaced by a
    continuum-transformed numerator.
    """
    (
        q,
        phi,
        a,
        b,
        c,
        chi_r,
        chi_z,
        reference_q,
        reference_phi,
        z,
        r,
        stencil_width,
    ) = _validated_fields(
        q,
        phi,
        z,
        r,
        a,
        b,
        c,
        chi_r,
        chi_z,
        reference_q,
        reference_phi,
        stencil_width,
    )
    jacobian = anisotropic_initial_data_jacobian(
        q,
        phi,
        z,
        r,
        a,
        b,
        c,
        background,
        chi_r,
        chi_z,
        reference_q,
        reference_phi,
        stencil_width,
    ).tolil()
    wall_data = _wall_primitives(
        q, phi, z, a, c, background, stencil_width,
    )
    psi = wall_data["psi"]
    sphere_metric = wall_data["sphere_metric"]
    normal_scale = wall_data["normal_scale"]
    dz = wall_data["Dz"]
    gamma = float(background["wall_stiffness"])
    nz, nr = q.shape
    n = q.size

    for wall in wall_data["walls"]:
        wall_index = wall["index"]
        source = wall["source"]
        orientation = float(source["orientation"])
        beta = np.asarray(source["beta"], dtype=float)
        beta_phi = np.asarray(source["beta_phi"], dtype=float)
        target = float(source["target"])
        dz_row = dz.getrow(wall_index)
        compact_indices = dz_row.indices
        compact_weights = dz_row.data
        for radial_index in range(nr):
            row = wall_index * nr + radial_index
            local_scale = normal_scale[wall_index, radial_index]
            local_metric = sphere_metric[wall_index, radial_index]
            local_psi = psi[wall_index, radial_index]
            robin = wall["robin"][radial_index]
            metric_entries = {}

            for compact_index, weight in zip(compact_indices, compact_weights):
                column = compact_index * nr + radial_index
                dmetric_dq = (
                    -2.0
                    * psi[compact_index, radial_index]
                    * sphere_metric[compact_index, radial_index]
                )
                drobin_dq = weight * dmetric_dq
                if compact_index == wall_index:
                    dscale_dq = -local_psi * local_scale
                    drobin_dq += 2.0 * beta[radial_index] * (
                        dscale_dq * local_metric
                        + local_scale * dmetric_dq
                    )
                derivative = orientation * drobin_dq / (2.0 * local_scale)
                if compact_index == wall_index:
                    dscale_dq = -local_psi * local_scale
                    # Off-manifold normalization derivative.  This remains
                    # active whenever the native Robin numerator is nonzero.
                    derivative -= (
                        orientation
                        * robin
                        * dscale_dq
                        / (2.0 * local_scale**2)
                    )
                metric_entries[column] = derivative

            phi_column = n + row
            metric_entries[phi_column] = (
                orientation * beta_phi[radial_index] * local_metric
            )
            _replace_sparse_row(jacobian, row, metric_entries)

            phi_entries = {}
            for compact_index, weight in zip(compact_indices, compact_weights):
                column = n + compact_index * nr + radial_index
                phi_entries[column] = float(weight)
            phi_entries[n + row] = phi_entries.get(n + row, 0.0) + (
                orientation * 0.5 * gamma * local_scale
            )
            delta = phi[wall_index, radial_index] - target
            dscale_dq = -local_psi * local_scale
            phi_entries[row] = (
                orientation * 0.5 * gamma * delta * dscale_dq
            )
            _replace_sparse_row(jacobian, n + row, phi_entries)

    return jacobian.tocsr()


def joint_parent_residual_and_jacobian(
    q,
    phi,
    z,
    r,
    a,
    b,
    c,
    background,
    chi_r,
    chi_z,
    reference_q,
    reference_phi,
    stencil_width=7,
):
    """Evaluate the hybrid residual and its analytic Jacobian."""
    residual = joint_parent_residual(
        q,
        phi,
        z,
        r,
        a,
        b,
        c,
        background,
        chi_r,
        chi_z,
        reference_q,
        reference_phi,
        stencil_width,
    )
    jacobian = joint_parent_jacobian(
        q,
        phi,
        z,
        r,
        a,
        b,
        c,
        background,
        chi_r,
        chi_z,
        reference_q,
        reference_phi,
        stencil_width,
    )
    return residual, jacobian


def solve_joint_parent(
    z,
    r,
    reference_q,
    reference_phi,
    a,
    b,
    c,
    background,
    chi_r,
    chi_z,
    initial_q=None,
    initial_phi=None,
    stencil_width=7,
    tolerance=1e-11,
    iterations=20,
):
    """Damped Newton helper for the isolated hybrid selector.

    This helper returns only the solved primitives and convergence metadata.
    Native-state completion and result-file generation intentionally remain
    outside this module.
    """
    q = (
        np.asarray(reference_q, dtype=float).copy()
        if initial_q is None
        else np.asarray(initial_q, dtype=float).copy()
    )
    phi = (
        np.asarray(reference_phi, dtype=float).copy()
        if initial_phi is None
        else np.asarray(initial_phi, dtype=float).copy()
    )
    z_array = np.asarray(z, dtype=float)
    history = []
    damping_history = []
    for _ in range(int(iterations)):
        residual = joint_parent_residual(
            q,
            phi,
            z,
            r,
            a,
            b,
            c,
            background,
            chi_r,
            chi_z,
            reference_q,
            reference_phi,
            stencil_width,
        )
        norm = float(np.max(np.abs(residual)))
        history.append(norm)
        if norm < tolerance:
            break
        jacobian = joint_parent_jacobian(
            q,
            phi,
            z,
            r,
            a,
            b,
            c,
            background,
            chi_r,
            chi_z,
            reference_q,
            reference_phi,
            stencil_width,
        )
        step = np.asarray(spsolve(jacobian, -residual), dtype=float)
        if not np.all(np.isfinite(step)):
            break
        size = q.size
        dq = step[:size].reshape(q.shape)
        dphi = step[size:].reshape(phi.shape)
        damping = 1.0
        accepted = False
        while damping >= 2.0**-20:
            candidate_q = q + damping * dq
            candidate_phi = phi + damping * dphi
            if np.min(z_array[:, None] + candidate_q) > 0.0:
                trial = joint_parent_residual(
                    candidate_q,
                    candidate_phi,
                    z,
                    r,
                    a,
                    b,
                    c,
                    background,
                    chi_r,
                    chi_z,
                    reference_q,
                    reference_phi,
                    stencil_width,
                )
                if np.max(np.abs(trial)) < norm:
                    q = candidate_q
                    phi = candidate_phi
                    accepted = True
                    damping_history.append(damping)
                    break
            damping *= 0.5
        if not accepted:
            break
    final = joint_parent_residual(
        q,
        phi,
        z,
        r,
        a,
        b,
        c,
        background,
        chi_r,
        chi_z,
        reference_q,
        reference_phi,
        stencil_width,
    )
    maximum = float(np.max(np.abs(final)))
    return {
        "q": q,
        "phi": phi,
        "psi": 1.0 / (z_array[:, None] + q),
        "converged": bool(maximum < tolerance),
        "maximum_residual": maximum,
        "residual_l2": float(np.sqrt(np.mean(final**2))),
        "history": history,
        "damping_history": damping_history,
    }
