"""Frozen reduced-wave corner compatibility diagnostics."""

from __future__ import annotations

import numpy as np


def full_boundary_residual(
    mixed_dirichlet_values,robin_values,robin_outward_normal_derivatives,
    robin_matrix,
):
    """Return the 17 boundary residuals for one time derivative order."""
    mixed=np.asarray(mixed_dirichlet_values,dtype=float)
    values=np.asarray(robin_values,dtype=float)
    derivatives=np.asarray(robin_outward_normal_derivatives,dtype=float)
    matrix=np.asarray(robin_matrix,dtype=float)
    if mixed.shape!=(4,) or values.shape!=(13,) or derivatives.shape!=(13,) or matrix.shape!=(13,13):
        raise ValueError("expected four Dirichlet and thirteen Robin fields")
    return np.concatenate((mixed,derivatives-matrix@values))


def frozen_principal_corner_hierarchy(
    robin_matrix,robin_seed,tangential_wavenumber=0.,maximum_time_order=10,
):
    """Generate compatible jets for a frozen principal vector-wave system.

    Near the wall use ``u(0,x,y)=exp(-xR)v exp(i k.y)`` and zero initial
    velocity.  Since ``M=R^2-|k|^2 I`` commutes with ``R``, every even time
    derivative is ``M^m v`` and satisfies the same Robin row; odd derivatives
    vanish.  The four Dirichlet gauge fields vanish at every order.
    """
    matrix=np.asarray(robin_matrix,dtype=float)
    seed=np.asarray(robin_seed,dtype=float)
    wave=float(tangential_wavenumber)
    order=int(maximum_time_order)
    if matrix.shape!=(13,13) or seed.shape!=(13,) or wave<0 or order<0:
        raise ValueError("invalid frozen hierarchy inputs")
    propagator=matrix@matrix-wave*wave*np.eye(13)
    commutator=matrix@propagator-propagator@matrix
    records=[]
    for derivative_order in range(order+1):
        if derivative_order%2:
            value=np.zeros(13);normal=np.zeros(13)
        else:
            value=np.linalg.matrix_power(propagator,derivative_order//2)@seed
            # Independently ordered products expose any failure of R and M to
            # commute in the generated hierarchy.
            normal=np.linalg.matrix_power(propagator,derivative_order//2)@(matrix@seed)
        residual=full_boundary_residual(np.zeros(4),value,normal,matrix)
        required=matrix@value
        scale=max(1.,float(np.linalg.norm(normal)),float(np.linalg.norm(required)))
        records.append({
            "time_derivative_order":derivative_order,
            "value_norm":float(np.linalg.norm(value)),
            "boundary_residual_norm":float(np.linalg.norm(residual)),
            "normalized_boundary_residual":float(np.linalg.norm(residual)/scale),
        })
    return {
        "records":records,
        "maximum_boundary_residual_norm":float(max(x["boundary_residual_norm"] for x in records)),
        "maximum_normalized_boundary_residual":float(max(x["normalized_boundary_residual"] for x in records)),
        "propagator_commutator_norm":float(np.linalg.norm(commutator,ord=2)),
        "construction":"u0=exp(-x R)v exp(i k.y), u1=0",
    }


def stabilizer_acceleration_corner_residual(
    robin_matrix,wall_value,wall_outward_normal_derivative,
):
    """Second-corner residual for a pure stabilizer acceleration wall jet."""
    values=np.zeros(13);derivatives=np.zeros(13)
    values[11]=float(wall_value);derivatives[11]=float(wall_outward_normal_derivative)
    residual=full_boundary_residual(np.zeros(4),values,derivatives,robin_matrix)
    return {
        "residual":residual,
        "maximum_absolute_residual":float(np.max(np.abs(residual))),
        "passes":bool(np.max(np.abs(residual))<1e-12),
    }
