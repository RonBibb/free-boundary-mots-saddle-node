"""Sparse floating Newton diagnostics for the sealed Test-4D operator."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, onenormest, splu

from bhps.global_bvp_collocation import shared_nodal_layout


def greedy_disjoint_row_column_groups(sparsity):
    """Color columns so columns in one group have disjoint supported rows."""
    pattern = csc_matrix(sparsity, dtype=bool)
    supported_rows = [
        set(pattern.indices[pattern.indptr[column]:pattern.indptr[column + 1]])
        for column in range(pattern.shape[1])
    ]
    order = sorted(
        range(pattern.shape[1]), key=lambda column: -len(supported_rows[column]),
    )
    groups = []
    occupied = []
    for column in order:
        for group, rows in zip(groups, occupied):
            if rows.isdisjoint(supported_rows[column]):
                group.append(column)
                rows.update(supported_rows[column])
                break
        else:
            groups.append([column])
            occupied.append(set(supported_rows[column]))
    return tuple(tuple(sorted(group)) for group in groups)


def sparse_colored_central_jacobian(
    function, vector, sparsity, relative_step=6e-6,
):
    """Central finite-difference Jacobian using structurally safe coloring."""
    vector = np.asarray(vector, dtype=float)
    pattern = csc_matrix(sparsity, dtype=bool)
    if pattern.shape != (len(vector), len(vector)):
        raise ValueError("Jacobian sparsity must be square and match the vector")
    groups = greedy_disjoint_row_column_groups(pattern)
    steps = float(relative_step) * np.maximum(1.0, np.abs(vector))
    rows = []
    columns = []
    data = []
    evaluation_count = 0
    for group in groups:
        perturbation = np.zeros_like(vector)
        perturbation[list(group)] = steps[list(group)]
        plus = np.asarray(function(vector + perturbation), dtype=float)
        minus = np.asarray(function(vector - perturbation), dtype=float)
        evaluation_count += 2
        difference = plus - minus
        for column in group:
            start = pattern.indptr[column]
            stop = pattern.indptr[column + 1]
            supported = pattern.indices[start:stop]
            rows.extend(supported.tolist())
            columns.extend([column] * len(supported))
            data.extend((difference[supported] / (2.0 * steps[column])).tolist())
    jacobian = csr_matrix(
        (np.asarray(data), (np.asarray(rows), np.asarray(columns))),
        shape=pattern.shape,
    )
    jacobian.eliminate_zeros()
    return jacobian, {
        "color_count": len(groups),
        "residual_evaluation_count": evaluation_count,
        "relative_step": float(relative_step),
        "structural_nonzero_count": int(pattern.nnz),
        "computed_nonzero_count": int(jacobian.nnz),
    }


def normalized_correction_norm(correction, configuration, component_scales):
    correction = np.asarray(correction, dtype=float)
    scales = np.asarray(component_scales, dtype=float)
    layout = shared_nodal_layout(configuration)
    slices = layout["slices"]
    pieces = {
        "axis_rho": correction[slices["axis_rho"]] / scales[0],
        "axis_u": correction[slices["axis_u"]] / scales[1],
        "bulk_rho": correction[slices["bulk_rho_free"]] / scales[0],
        "bulk_w": correction[slices["bulk_w_free"]] / scales[2],
    }
    return {
        "maximum": float(max(np.max(np.abs(value)) for value in pieces.values())),
        "components": {
            name: float(np.max(np.abs(value))) for name, value in pieces.items()
        },
    }


def sparse_newton_step(
    function, vector, sparsity, configuration, component_scales,
    relative_step=6e-6,
):
    vector = np.asarray(vector, dtype=float)
    residual_before = np.asarray(function(vector), dtype=float)
    jacobian, differentiation = sparse_colored_central_jacobian(
        function, vector, sparsity, relative_step,
    )
    factor = splu(csc_matrix(jacobian))
    correction = factor.solve(-residual_before)
    updated = vector + correction
    residual_after = np.asarray(function(updated), dtype=float)
    jacobian_norm = float(onenormest(jacobian))
    inverse_operator = LinearOperator(
        jacobian.shape,
        matvec=lambda value: factor.solve(value),
        rmatvec=lambda value: factor.solve(value, trans="T"),
        dtype=float,
    )
    inverse_norm = float(onenormest(inverse_operator))
    return {
        "updated_vector": updated,
        "correction": correction,
        "residual_before": residual_before,
        "residual_after": residual_after,
        "jacobian": jacobian,
        "diagnostics": {
            "differentiation": differentiation,
            "jacobian_one_norm_estimate": jacobian_norm,
            "inverse_one_norm_estimate": inverse_norm,
            "condition_one_norm_estimate": jacobian_norm * inverse_norm,
            "normalized_correction": normalized_correction_norm(
                correction, configuration, component_scales,
            ),
            "maximum_residual_before": float(np.max(np.abs(residual_before))),
            "maximum_residual_after": float(np.max(np.abs(residual_after))),
            "residual_reduction_factor": float(
                np.max(np.abs(residual_before))
                / max(np.max(np.abs(residual_after)), np.finfo(float).tiny)
            ),
            "all_finite": bool(
                np.all(np.isfinite(correction))
                and np.all(np.isfinite(residual_after))
            ),
        },
    }
