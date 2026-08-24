"""Per-field selective compact-wall algebraic gates for Protocol 125.

This helper is deliberately independent of the wall physics.  It consumes a
two-endpoint matrix, the corresponding two complete compact-stencil rows, and
one or more field right-hand sides.  Row equilibration always uses the
Euclidean norms of the *full* rows, never the truncated endpoint matrix.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


SELECTIVE_REQUIRED_RANK = 2
SELECTIVE_MAXIMUM_CONDITION = 1e12
SELECTIVE_MINIMUM_PIVOT = 1e-10
SELECTIVE_MAXIMUM_NORMALIZED_RESIDUAL = 1e-12


class SelectiveWallAlgebraicGateError(RuntimeError):
    """Structured rejection of one selective two-wall endpoint block."""

    def __init__(self, message, *, radial_index, field, gate, diagnostics):
        super().__init__(message)
        self.radial_index = int(radial_index)
        self.field = str(field)
        self.gate = str(gate)
        self.diagnostics = dict(diagnostics)


def solve_selective_wall_endpoint_block(
    endpoint_matrix,
    full_stencil_rows,
    right_hand_sides,
    field_names: Sequence[str],
    *,
    radial_index,
):
    """Solve and gate one two-wall block with separate field evidence.

    ``right_hand_sides`` has shape ``(2,nfield)``.  The structural matrix is
    allowed to be shared, but every field receives its own complete record and
    independently gated direct normalized residual.
    """
    matrix = np.asarray(endpoint_matrix, dtype=float)
    full = np.asarray(full_stencil_rows, dtype=float)
    right = np.asarray(right_hand_sides, dtype=float)
    names = tuple(str(name) for name in field_names)
    radial_index = int(radial_index)
    if matrix.shape != (2, 2):
        raise ValueError("selective endpoint matrix must be 2x2")
    if full.ndim != 2 or full.shape[0] != 2 or full.shape[1] < 2:
        raise ValueError("selective full-stencil rows must have shape (2,nz)")
    if right.shape != (2, len(names)) or not names or len(set(names)) != len(names):
        raise ValueError("selective right-hand sides and field names differ")
    if radial_index < 0:
        raise ValueError("selective radial index must be nonnegative")
    if not all(np.all(np.isfinite(value)) for value in (matrix, full, right)):
        raise ValueError("selective algebraic inputs must be finite")

    row_norms = np.linalg.norm(full, axis=1)
    if np.any(row_norms <= 0.0) or not np.all(np.isfinite(row_norms)):
        raise SelectiveWallAlgebraicGateError(
            "degenerate full compact-stencil row in selective solve",
            radial_index=radial_index,
            field=names[0],
            gate="finite_positive_full_row_norm",
            diagnostics={"full_row_norms": row_norms.tolist()},
        )
    equilibrated = matrix/row_norms[:, None]
    singular = np.linalg.svd(equilibrated, compute_uv=False)
    rank = int(np.linalg.matrix_rank(equilibrated))
    pivot = float(singular[-1]) if len(singular) else 0.0
    condition = (
        float(singular[0]/singular[-1]) if pivot > 0.0 else float("inf")
    )
    raw_condition = float(np.linalg.cond(matrix))
    structural = {
        "rank": rank,
        "required_rank": SELECTIVE_REQUIRED_RANK,
        "equilibrated_condition": condition,
        "maximum_allowed_condition": SELECTIVE_MAXIMUM_CONDITION,
        "normalized_pivot": pivot,
        "minimum_allowed_normalized_pivot": SELECTIVE_MINIMUM_PIVOT,
        "raw_condition": raw_condition,
        "full_row_norms": row_norms.copy(),
        "equilibration": "endpoint_matrix/full_compact_stencil_row_L2_norm",
    }
    structural_gate = (
        rank == SELECTIVE_REQUIRED_RANK
        and np.isfinite(condition)
        and condition <= SELECTIVE_MAXIMUM_CONDITION
        and pivot >= SELECTIVE_MINIMUM_PIVOT
    )
    if not structural_gate:
        raise SelectiveWallAlgebraicGateError(
            "selective two-wall endpoint block failed rank/condition/pivot gate",
            radial_index=radial_index,
            field=names[0],
            gate="rank_condition_pivot",
            diagnostics=structural,
        )

    solved = np.linalg.solve(matrix, right)
    if not np.all(np.isfinite(solved)):
        raise SelectiveWallAlgebraicGateError(
            "selective two-wall endpoint solution is nonfinite",
            radial_index=radial_index,
            field=names[0],
            gate="finite_solution",
            diagnostics=structural,
        )
    fields = {}
    for field_index, name in enumerate(names):
        solution = solved[:, field_index]
        rhs = right[:, field_index]
        residual = matrix@solution-rhs
        normalized = float(
            np.max(np.abs(residual))
            /max(
                1.0,
                np.linalg.norm(matrix, ord=np.inf)*np.max(np.abs(solution)),
                np.max(np.abs(rhs)),
            )
        )
        record = {
            **structural,
            "field": name,
            "radial_index": radial_index,
            "normalized_linear_residual": normalized,
            "maximum_allowed_normalized_linear_residual": (
                SELECTIVE_MAXIMUM_NORMALIZED_RESIDUAL
            ),
            "direct_residual_strict_inequality": True,
            "passed": bool(
                structural_gate
                and np.isfinite(normalized)
                and normalized < SELECTIVE_MAXIMUM_NORMALIZED_RESIDUAL
            ),
        }
        if not record["passed"]:
            raise SelectiveWallAlgebraicGateError(
                "selective field failed its direct normalized residual gate",
                radial_index=radial_index,
                field=name,
                gate="normalized_linear_residual",
                diagnostics=record,
            )
        fields[name] = record
    return solved, fields


def summarize_selective_field_evidence(records, field_order):
    """Summarize mandatory per-radius records without pooling fields."""
    names = tuple(str(name) for name in field_order)
    if set(records) != set(names) or len(records) != len(names):
        raise ValueError("selective evidence field set changed")
    output = {}
    for name in names:
        local = tuple(records[name])
        if not local:
            raise ValueError(f"selective field {name} has no radial evidence")
        radial_indices = np.asarray(
            [int(record["radial_index"]) for record in local], dtype=int,
        )
        if not np.array_equal(radial_indices, np.arange(len(local))):
            raise ValueError(f"selective field {name} radial evidence is incomplete")
        for record in local:
            if record["field"] != name or not bool(record["passed"]):
                raise ValueError(f"selective field {name} contains failed evidence")
        profiles = {
            "rank": np.asarray([record["rank"] for record in local], dtype=int),
            "equilibrated_condition": np.asarray([
                record["equilibrated_condition"] for record in local
            ]),
            "raw_condition": np.asarray([
                record["raw_condition"] for record in local
            ]),
            "normalized_pivot": np.asarray([
                record["normalized_pivot"] for record in local
            ]),
            "normalized_linear_residual": np.asarray([
                record["normalized_linear_residual"] for record in local
            ]),
        }
        output[name] = {
            "required_rank": SELECTIVE_REQUIRED_RANK,
            "maximum_allowed_condition": SELECTIVE_MAXIMUM_CONDITION,
            "minimum_allowed_normalized_pivot": SELECTIVE_MINIMUM_PIVOT,
            "maximum_allowed_normalized_linear_residual": (
                SELECTIVE_MAXIMUM_NORMALIZED_RESIDUAL
            ),
            "minimum_rank": int(np.min(profiles["rank"])),
            "maximum_equilibrated_condition": float(np.max(
                profiles["equilibrated_condition"]
            )),
            "maximum_raw_condition": float(np.max(profiles["raw_condition"])),
            "minimum_normalized_pivot": float(np.min(
                profiles["normalized_pivot"]
            )),
            "maximum_normalized_linear_residual": float(np.max(
                profiles["normalized_linear_residual"]
            )),
            "profiles": profiles,
            "passed": True,
        }
    return {
        "field_order": names,
        "fields": output,
        "each_field_gated_separately": True,
        "chi_credited_only_with_chi_block": True,
        "passed": bool(all(record["passed"] for record in output.values())),
    }
