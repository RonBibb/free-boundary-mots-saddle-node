"""Opt-in simultaneous acceleration boundary closures.

This module is deliberately isolated from the production evolution RHS.  It
provides the linear algebra needed to replace two sequential corner operations
by one constrained profile solve *after* the nonlinear state ``q,v`` has been
frozen.  At that point the differentiated axis, wall, and outgoing rows are
linear in the unknown acceleration values.

For a preferred one-dimensional acceleration profile ``p`` the core problem is

``minimize ||diag(w) (x-p)||_2  subject to C x = d``.

Positive ``w`` are correction penalties (larger means harder to move).  The
solver row-equilibrates ``C diag(w)^-1``, uses an SVD minimum-norm projection,
accepts consistent redundant constraints, rejects inconsistent constraints,
and reports rank, conditioning, consistency, KKT, and boundary residuals.

The two convenience closures operate on one acceleration field at a time.  By
default they implement **owner-last** semantics: every non-corner preferred
value is fixed exactly and the two compact-wall endpoint values are the only
adjustable unknowns.  This reproduces “axis/outgoing first, compact walls last”
without sequentially solving potentially coupled endpoints:

* ``close_regular_axis_wall_profile`` fixes the open regular-axis extrapolation
  and lets the compact walls own the two axis/wall corners;
* ``close_outer_face_wall_profile`` fixes the outgoing targets on open nodes and
  lets the compact walls own the two outer-face/wall corners.

Both wrappers accept ``adjustment_scope="all_nodes"`` to run the generic
minimum-correction projector as a diagnostic comparator.  That comparator may
move open axis/outgoing values; it is not the minimal owner-last repair.

The regular-axis wrapper is itself only a diagnostic candidate until parity,
regular Cartesian jets, and the positive-radius fit defect are shown to
converge after endpoint correction.  Pointwise wall rows do not independently
determine every regularized axis channel.

For genuinely coupled acceleration rows, flatten all involved profiles and use
``weighted_minimum_correction`` or its indexed variant with the assembled
coupled matrix directly.  In particular, an unknown ``h_zz,tt``/``Phi_tt``
coupling must not be hidden in the numeric forcing of an independent wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_CONDITION_LIMIT = 1.0e12


@dataclass(frozen=True)
class ConstrainedClosureResult:
    """One constrained profile and its audit diagnostics."""

    profile: np.ndarray
    diagnostics: dict


class InconsistentBoundaryConstraints(ValueError):
    """Raised when the requested hard boundary rows have no common solution."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


class DegenerateBoundaryOwnership(np.linalg.LinAlgError):
    """Raised when selected owner unknowns cannot safely determine all rows."""

    def __init__(self, message, diagnostics):
        super().__init__(message)
        self.diagnostics = diagnostics


def _effective_condition(singular_values, rank):
    values = np.asarray(singular_values, dtype=float)
    rank = int(rank)
    if rank == 0:
        return 1.0
    return float(values[0] / values[rank - 1])


def _svd_summary(matrix, rcond=None):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("SVD diagnostic requires a matrix")
    if matrix.size == 0:
        return {
            "u": np.empty((matrix.shape[0], 0)),
            "singular_values": np.empty(0),
            "vh": np.empty((0, matrix.shape[1])),
            "rank": 0,
            "rank_tolerance": 0.0,
            "effective_condition_number": 1.0,
        }
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    largest = float(singular[0]) if len(singular) else 0.0
    if rcond is None:
        tolerance = max(matrix.shape) * np.finfo(float).eps * largest
    else:
        rcond = float(rcond)
        if not np.isfinite(rcond) or rcond < 0.0:
            raise ValueError("rcond must be finite and nonnegative")
        tolerance = rcond * largest
    rank = int(np.sum(singular > tolerance))
    return {
        "u": u,
        "singular_values": singular,
        "vh": vh,
        "rank": rank,
        "rank_tolerance": float(tolerance),
        "effective_condition_number": _effective_condition(singular, rank),
    }


def _full_row_smallest_singular_value(summary, row_count):
    """Return sigma_min including structural zero singular values."""
    row_count = int(row_count)
    if row_count == 0:
        return float("inf")
    singular = np.asarray(summary["singular_values"], dtype=float)
    if len(singular) < row_count:
        return 0.0
    return float(singular[row_count - 1])


def _constraint_residual_diagnostics(matrix, profile, rhs):
    matrix = np.asarray(matrix, dtype=float)
    profile = np.asarray(profile, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    residual = matrix @ profile - rhs
    row_scale = np.abs(matrix) @ np.abs(profile) + np.abs(rhs)
    normalized = np.zeros_like(residual)
    positive = row_scale > 0.0
    normalized[positive] = np.abs(residual[positive]) / row_scale[positive]
    normalized[~positive] = np.abs(residual[~positive])
    return {
        "residual": residual,
        "maximum_absolute": float(np.max(np.abs(residual), initial=0.0)),
        "l2": float(np.linalg.norm(residual)),
        "maximum_row_normalized": float(np.max(normalized, initial=0.0)),
        "row_absolute_scales": row_scale,
    }


def weighted_minimum_correction(
    preferred,
    constraint_matrix,
    constraint_rhs,
    correction_weights=None,
    *,
    rcond=None,
    consistency_atol=1.0e-12,
    consistency_rtol=1.0e-10,
    condition_limit=DEFAULT_CONDITION_LIMIT,
    require_full_row_rank=False,
    require_well_conditioned=False,
):
    """Solve a weighted minimum-correction problem with exact linear rows.

    Parameters
    ----------
    preferred:
        One-dimensional preferred acceleration vector ``p``.
    constraint_matrix, constraint_rhs:
        Numeric hard rows ``C x = d``.  Coefficients may already contain any
        state-dependent factors obtained by freezing ``q,v``.
    correction_weights:
        Positive square-root objective weights ``w``.  The minimized norm is
        ``||w * (x-p)||_2``.  The default is uniform weight one.
    rcond:
        Relative SVD rank cutoff.  ``None`` uses the standard dimension-scaled
        machine-epsilon threshold.
    consistency_atol, consistency_rtol:
        Tolerance for the component of the equilibrated RHS outside the row
        range.  Inconsistent redundant/overdetermined rows raise
        :class:`InconsistentBoundaryConstraints`.
    condition_limit:
        Diagnostic threshold applied to the row-equilibrated effective
        condition number.  It does not reject unless
        ``require_well_conditioned`` is true.

    Notes
    -----
    The objective is strictly convex for positive weights, so a consistent
    constraint set has one unique minimum-correction profile even when its rows
    are redundant.  This routine does not linearize nonlinear boundary data;
    callers must assemble numeric rows after freezing the stage state.
    """
    preferred = np.asarray(preferred, dtype=float)
    matrix = np.asarray(constraint_matrix, dtype=float)
    rhs = np.asarray(constraint_rhs, dtype=float)
    if preferred.ndim != 1 or len(preferred) == 0:
        raise ValueError("preferred acceleration profile must be nonempty and one-dimensional")
    if matrix.ndim != 2 or matrix.shape[1] != len(preferred):
        raise ValueError("constraint matrix does not match the acceleration profile")
    if rhs.shape != (matrix.shape[0],):
        raise ValueError("constraint RHS does not match the constraint rows")
    if np.any(~np.isfinite(preferred)) or np.any(~np.isfinite(matrix)) or np.any(~np.isfinite(rhs)):
        raise ValueError("minimum-correction inputs must be finite")
    if correction_weights is None:
        weights = np.ones_like(preferred)
    else:
        weights = np.asarray(correction_weights, dtype=float)
    if weights.shape != preferred.shape or np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("correction weights must be positive, finite, and profile-aligned")
    consistency_atol = float(consistency_atol)
    consistency_rtol = float(consistency_rtol)
    condition_limit = float(condition_limit)
    if (
        not np.isfinite(consistency_atol) or consistency_atol < 0.0
        or not np.isfinite(consistency_rtol) or consistency_rtol < 0.0
        or not np.isfinite(condition_limit) or condition_limit < 1.0
    ):
        raise ValueError("invalid consistency or conditioning threshold")

    baseline = _constraint_residual_diagnostics(matrix, preferred, rhs)
    transformed = matrix / weights[None, :]
    transformed_rhs = rhs - matrix @ preferred
    row_norms = np.linalg.norm(transformed, axis=1)
    row_scales = np.where(row_norms > 0.0, row_norms, 1.0)
    equilibrated = transformed / row_scales[:, None]
    equilibrated_rhs = transformed_rhs / row_scales

    raw_svd = _svd_summary(matrix, rcond)
    weighted_svd = _svd_summary(transformed, rcond)
    svd = _svd_summary(equilibrated, rcond)
    rank = svd["rank"]
    u = svd["u"]
    singular = svd["singular_values"]
    vh = svd["vh"]
    if rank:
        range_component = u[:, :rank] @ (u[:, :rank].T @ equilibrated_rhs)
    else:
        range_component = np.zeros_like(equilibrated_rhs)
    inconsistency = equilibrated_rhs - range_component
    consistency_scale = max(
        float(np.linalg.norm(equilibrated_rhs)),
        float(np.linalg.norm(range_component)),
    )
    consistency_tolerance = consistency_atol + consistency_rtol * consistency_scale

    diagnostics = {
        "unknown_count": int(len(preferred)),
        "constraint_count": int(matrix.shape[0]),
        "numerical_rank": int(rank),
        "degrees_of_freedom_after_constraints": int(len(preferred) - rank),
        "redundant_constraint_count": int(matrix.shape[0] - rank),
        "full_row_rank": bool(rank == matrix.shape[0]),
        "rank_tolerance": svd["rank_tolerance"],
        "raw_numerical_rank": int(raw_svd["rank"]),
        "raw_full_row_rank": bool(raw_svd["rank"] == matrix.shape[0]),
        "raw_rank_tolerance": raw_svd["rank_tolerance"],
        "equilibrated_singular_values": singular.tolist(),
        "raw_singular_values": raw_svd["singular_values"].tolist(),
        "raw_full_row_smallest_singular_value": _full_row_smallest_singular_value(
            raw_svd, matrix.shape[0],
        ),
        "equilibrated_effective_condition_number": svd["effective_condition_number"],
        "raw_effective_condition_number": raw_svd["effective_condition_number"],
        "raw_full_row_condition_number": (
            raw_svd["effective_condition_number"]
            if raw_svd["rank"] == matrix.shape[0] else float("inf")
        ),
        "weighted_unequilibrated_numerical_rank": int(weighted_svd["rank"]),
        "weighted_unequilibrated_full_row_rank": bool(
            weighted_svd["rank"] == matrix.shape[0]
        ),
        "weighted_unequilibrated_rank_tolerance": weighted_svd["rank_tolerance"],
        "weighted_unequilibrated_singular_values": weighted_svd[
            "singular_values"
        ].tolist(),
        "weighted_unequilibrated_full_row_condition_number": (
            weighted_svd["effective_condition_number"]
            if weighted_svd["rank"] == matrix.shape[0] else float("inf")
        ),
        "full_row_condition_number": (
            svd["effective_condition_number"] if rank == matrix.shape[0] else float("inf")
        ),
        "condition_limit": condition_limit,
        "well_conditioned": bool(svd["effective_condition_number"] <= condition_limit),
        "minimum_constraint_row_norm": float(np.min(row_norms)) if len(row_norms) else 0.0,
        "maximum_constraint_row_norm": float(np.max(row_norms)) if len(row_norms) else 0.0,
        "weight_minimum": float(np.min(weights)),
        "weight_maximum": float(np.max(weights)),
        "weight_dynamic_range": float(np.max(weights) / np.min(weights)),
        "baseline_constraint_maximum_absolute": baseline["maximum_absolute"],
        "baseline_constraint_l2": baseline["l2"],
        "equilibrated_consistency_residual_maximum_absolute": float(
            np.max(np.abs(inconsistency), initial=0.0)
        ),
        "equilibrated_consistency_residual_l2": float(np.linalg.norm(inconsistency)),
        "consistency_tolerance": float(consistency_tolerance),
        "constraints_consistent": bool(np.linalg.norm(inconsistency) <= consistency_tolerance),
    }
    if require_full_row_rank and rank != matrix.shape[0]:
        raise InconsistentBoundaryConstraints(
            "boundary constraint matrix is not full row rank", diagnostics,
        )
    if np.linalg.norm(inconsistency) > consistency_tolerance:
        raise InconsistentBoundaryConstraints(
            "boundary constraint rows are mutually inconsistent", diagnostics,
        )
    if require_well_conditioned and not diagnostics["well_conditioned"]:
        raise np.linalg.LinAlgError(
            "row-equilibrated boundary constraint matrix exceeds condition limit"
        )

    if rank:
        projected_coefficients = u[:, :rank].T @ equilibrated_rhs
        transformed_correction = vh[:rank].T @ (
            projected_coefficients / singular[:rank]
        )
        dual = u[:, :rank] @ (projected_coefficients / singular[:rank] ** 2)
        stationarity = transformed_correction - equilibrated.T @ dual
    else:
        transformed_correction = np.zeros_like(preferred)
        stationarity = np.zeros_like(preferred)
    correction = transformed_correction / weights
    profile = preferred + correction
    closure = _constraint_residual_diagnostics(matrix, profile, rhs)
    residual_tolerance = consistency_atol + consistency_rtol * max(
        float(np.linalg.norm(rhs)), float(np.linalg.norm(matrix @ profile)),
    )
    diagnostics.update({
        "constraint_residuals": closure["residual"].tolist(),
        "constraint_maximum_absolute_residual": closure["maximum_absolute"],
        "constraint_l2_residual": closure["l2"],
        "constraint_maximum_row_normalized_residual": closure[
            "maximum_row_normalized"
        ],
        "constraint_residual_tolerance": float(residual_tolerance),
        "constraint_residual_passes": bool(closure["l2"] <= residual_tolerance),
        "weighted_correction_l2": float(np.linalg.norm(weights * correction)),
        "unweighted_correction_l2": float(np.linalg.norm(correction)),
        "maximum_absolute_correction": float(np.max(np.abs(correction), initial=0.0)),
        "transformed_kkt_stationarity_l2": float(np.linalg.norm(stationarity)),
        "transformed_kkt_stationarity_maximum_absolute": float(
            np.max(np.abs(stationarity), initial=0.0)
        ),
    })
    if not diagnostics["constraint_residual_passes"]:
        raise np.linalg.LinAlgError("minimum-correction solve did not close its constraints")
    return ConstrainedClosureResult(profile=profile, diagnostics=diagnostics)


def weighted_minimum_correction_on_indices(
    preferred,
    constraint_matrix,
    constraint_rhs,
    adjustable_indices,
    correction_weights=None,
    *,
    owner_pivot_atol=1.0e-12,
    owner_pivot_rtol=1.0e-12,
    require_resolved_owner=True,
    **solver_options,
):
    """Solve while fixing every non-adjustable value exactly to its preference.

    This is the algebraic form of owner-last boundary ordering.  If ``J`` is the
    adjustable index set and ``F`` its complement, the reduced hard rows are

    ``C[:,J] x[J] = d - C[:,F] p[F]``.

    By default the selected-owner solve fails closed unless the original owner
    block is full row rank, has adequate absolute and full-row-relative pivot
    strength, and is well conditioned.  Set ``require_resolved_owner=False``
    only for a diagnostic probe.  There is no automatic fallback to the
    all-node projector.
    """
    preferred = np.asarray(preferred, dtype=float)
    matrix = np.asarray(constraint_matrix, dtype=float)
    rhs = np.asarray(constraint_rhs, dtype=float)
    indices = np.asarray(adjustable_indices, dtype=int)
    if preferred.ndim != 1 or len(preferred) == 0:
        raise ValueError("preferred acceleration profile must be nonempty and one-dimensional")
    if matrix.ndim != 2 or matrix.shape[1] != len(preferred) or rhs.shape != (matrix.shape[0],):
        raise ValueError("indexed minimum-correction constraints are not profile-aligned")
    if (
        indices.ndim != 1 or len(indices) == 0
        or np.any(indices < 0) or np.any(indices >= len(preferred))
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("adjustable indices must be unique valid profile indices")
    if correction_weights is None:
        weights = np.ones_like(preferred)
    else:
        weights = np.asarray(correction_weights, dtype=float)
    if weights.shape != preferred.shape or np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("correction weights must be positive, finite, and profile-aligned")
    owner_pivot_atol = float(owner_pivot_atol)
    owner_pivot_rtol = float(owner_pivot_rtol)
    if (
        not np.isfinite(owner_pivot_atol) or owner_pivot_atol < 0.0
        or not np.isfinite(owner_pivot_rtol) or owner_pivot_rtol < 0.0
    ):
        raise ValueError("owner pivot tolerances must be finite and nonnegative")
    fixed_mask = np.ones(len(preferred), dtype=bool)
    fixed_mask[indices] = False
    fixed = np.flatnonzero(fixed_mask)
    reduced_rhs = rhs - matrix[:, fixed] @ preferred[fixed]
    reduced = weighted_minimum_correction(
        preferred[indices], matrix[:, indices], reduced_rhs, weights[indices],
        **solver_options,
    )
    profile = preferred.copy()
    profile[indices] = reduced.profile
    closure = _constraint_residual_diagnostics(matrix, profile, rhs)
    consistency_atol = float(solver_options.get("consistency_atol", 1.0e-12))
    consistency_rtol = float(solver_options.get("consistency_rtol", 1.0e-10))
    residual_tolerance = consistency_atol + consistency_rtol * max(
        float(np.linalg.norm(rhs)), float(np.linalg.norm(matrix @ profile)),
    )

    # Row equilibration is useful for solving, but it can turn a physically
    # tiny endpoint coefficient into a unit pivot.  Audit the unscaled owner
    # block as well as the block normalized only by each *full* wall-row norm.
    # The latter detects cancellation of endpoint ownership against an O(1)
    # stencil row; the absolute test mirrors the production endpoint-denominator
    # safeguard and catches a uniformly tiny owner block with O(1) forcing.
    rcond = solver_options.get("rcond")
    condition_limit = float(solver_options.get("condition_limit", DEFAULT_CONDITION_LIMIT))
    owner_matrix = matrix[:, indices]
    full_row_norms = np.linalg.norm(matrix, axis=1)
    full_row_scales = np.where(full_row_norms > 0.0, full_row_norms, 1.0)
    owner_row_norms = np.linalg.norm(owner_matrix, axis=1)
    owner_row_strengths = np.zeros_like(owner_row_norms)
    positive_full_rows = full_row_norms > 0.0
    owner_row_strengths[positive_full_rows] = (
        owner_row_norms[positive_full_rows] / full_row_norms[positive_full_rows]
    )
    relative_owner_matrix = owner_matrix / full_row_scales[:, None]
    owner_raw_svd = _svd_summary(owner_matrix, rcond)
    owner_relative_svd = _svd_summary(relative_owner_matrix, rcond)
    row_count = matrix.shape[0]
    owner_raw_sigma_min = _full_row_smallest_singular_value(owner_raw_svd, row_count)
    owner_relative_sigma_min = _full_row_smallest_singular_value(
        owner_relative_svd, row_count,
    )
    owner_raw_full_rank = owner_raw_svd["rank"] == row_count
    owner_relative_full_rank = owner_relative_svd["rank"] == row_count
    owner_absolute_pivot_passes = owner_raw_sigma_min > owner_pivot_atol
    owner_relative_pivot_passes = owner_relative_sigma_min > owner_pivot_rtol
    owner_condition_number = (
        owner_relative_svd["effective_condition_number"]
        if owner_relative_full_rank else float("inf")
    )
    owner_raw_condition_number = (
        owner_raw_svd["effective_condition_number"]
        if owner_raw_full_rank else float("inf")
    )
    owner_condition_passes = (
        owner_raw_condition_number <= condition_limit
        and owner_condition_number <= condition_limit
        and reduced.diagnostics[
            "weighted_unequilibrated_full_row_condition_number"
        ] <= condition_limit
        and reduced.diagnostics["equilibrated_effective_condition_number"] <= condition_limit
    )
    owner_resolution_passes = bool(
        owner_raw_full_rank
        and owner_relative_full_rank
        and owner_absolute_pivot_passes
        and owner_relative_pivot_passes
        and owner_condition_passes
    )
    diagnostics = dict(reduced.diagnostics)
    diagnostics.update({
        "reduced_unknown_count": diagnostics["unknown_count"],
        "unknown_count": int(len(preferred)),
        "adjustable_unknown_count": int(len(indices)),
        "fixed_preference_count": int(len(fixed)),
        "adjustable_indices": indices.tolist(),
        "fixed_preference_indices": fixed.tolist(),
        "adjustment_scope": "selected_indices",
        "owner_raw_numerical_rank": int(owner_raw_svd["rank"]),
        "owner_raw_full_row_rank": bool(owner_raw_full_rank),
        "owner_raw_rank_tolerance": owner_raw_svd["rank_tolerance"],
        "owner_raw_singular_values": owner_raw_svd["singular_values"].tolist(),
        "owner_raw_smallest_singular_value": owner_raw_sigma_min,
        "owner_raw_full_row_condition_number": owner_raw_condition_number,
        "owner_full_row_normalized_numerical_rank": int(owner_relative_svd["rank"]),
        "owner_full_row_normalized_full_row_rank": bool(owner_relative_full_rank),
        "owner_full_row_normalized_singular_values": owner_relative_svd[
            "singular_values"
        ].tolist(),
        "owner_full_row_normalized_smallest_singular_value": owner_relative_sigma_min,
        "owner_full_row_normalized_condition_number": owner_condition_number,
        "owner_row_norms": owner_row_norms.tolist(),
        "full_constraint_row_norms": full_row_norms.tolist(),
        "owner_to_full_row_norm_ratios": owner_row_strengths.tolist(),
        "minimum_owner_to_full_row_norm_ratio": float(
            np.min(owner_row_strengths) if len(owner_row_strengths) else 1.0
        ),
        "owner_pivot_atol": owner_pivot_atol,
        "owner_pivot_rtol": owner_pivot_rtol,
        "owner_absolute_pivot_passes": bool(owner_absolute_pivot_passes),
        "owner_relative_pivot_passes": bool(owner_relative_pivot_passes),
        "owner_condition_passes": bool(owner_condition_passes),
        "owner_resolution_passes": owner_resolution_passes,
        "require_resolved_owner": bool(require_resolved_owner),
        "maximum_fixed_preference_deviation": float(
            np.max(np.abs(profile[fixed] - preferred[fixed]), initial=0.0)
        ),
        "constraint_residuals": closure["residual"].tolist(),
        "constraint_maximum_absolute_residual": closure["maximum_absolute"],
        "constraint_l2_residual": closure["l2"],
        "constraint_maximum_row_normalized_residual": closure[
            "maximum_row_normalized"
        ],
        "constraint_residual_tolerance": float(residual_tolerance),
        "constraint_residual_passes": bool(closure["l2"] <= residual_tolerance),
        "weighted_correction_l2": float(np.linalg.norm(weights * (profile - preferred))),
        "unweighted_correction_l2": float(np.linalg.norm(profile - preferred)),
        "maximum_absolute_correction": float(
            np.max(np.abs(profile - preferred), initial=0.0)
        ),
    })
    if not diagnostics["constraint_residual_passes"]:
        raise np.linalg.LinAlgError("indexed minimum-correction solve did not close its constraints")
    if require_resolved_owner and not diagnostics["owner_resolution_passes"]:
        raise DegenerateBoundaryOwnership(
            "selected boundary owners do not provide a safely resolved constraint block",
            diagnostics,
        )
    return ConstrainedClosureResult(profile, diagnostics)


def compact_wall_rows(
    compact_derivative,
    *,
    lower_robin=0.0,
    upper_robin=0.0,
    lower_forcing=0.0,
    upper_forcing=0.0,
):
    """Assemble two frozen compact-wall acceleration rows.

    The convention matches the production endpoint equations:

    ``(D_z a)_wall + robin_wall a_wall + forcing_wall = 0``.

    Once ``q,v`` are fixed, nonlinear wall coefficients and forcing terms are
    ordinary numeric inputs to this function.
    """
    derivative = np.asarray(
        compact_derivative.toarray()
        if hasattr(compact_derivative, "toarray") else compact_derivative,
        dtype=float,
    )
    if (
        derivative.ndim != 2 or derivative.shape[0] != derivative.shape[1]
        or derivative.shape[0] < 3 or np.any(~np.isfinite(derivative))
    ):
        raise ValueError("compact derivative must be a finite square matrix")
    coefficients = np.asarray((lower_robin, upper_robin), dtype=float)
    forcing = np.asarray((lower_forcing, upper_forcing), dtype=float)
    if np.any(~np.isfinite(coefficients)) or np.any(~np.isfinite(forcing)):
        raise ValueError("compact-wall coefficients and forcing must be finite")
    rows = np.stack((derivative[0], derivative[-1])).copy()
    rows[0, 0] += coefficients[0]
    rows[1, -1] += coefficients[1]
    return rows, -forcing


def _wall_augmented_diagnostics(result, rows, rhs, preferred, kind):
    diagnostics = dict(result.diagnostics)
    before = _constraint_residual_diagnostics(rows, preferred, rhs)
    after = _constraint_residual_diagnostics(rows, result.profile, rhs)
    diagnostics.update({
        "closure_kind": kind,
        "preferred_wall_residuals": before["residual"].tolist(),
        "preferred_wall_maximum_absolute_residual": before["maximum_absolute"],
        "closed_wall_residuals": after["residual"].tolist(),
        "closed_wall_maximum_absolute_residual": after["maximum_absolute"],
    })
    return ConstrainedClosureResult(result.profile, diagnostics)


def close_regular_axis_wall_profile(
    regular_axis_preferred,
    compact_derivative,
    *,
    lower_robin=0.0,
    upper_robin=0.0,
    lower_forcing=0.0,
    upper_forcing=0.0,
    correction_weights=None,
    adjustment_scope="owner_endpoints",
    owner_pivot_atol=1.0e-12,
    owner_pivot_rtol=1.0e-12,
    require_resolved_owner=True,
    **solver_options,
):
    """Close one regular-axis acceleration profile with both wall rows.

    ``regular_axis_preferred`` is the full compact profile obtained from the
    positive-radius regular-axis extrapolation.  In the default
    ``owner_endpoints`` scope its open values are exact and only its two compact
    endpoints may move.  The alternative ``all_nodes`` scope is a weighted
    minimum-correction comparator and may alter open axis values.  This wrapper
    is an algebraic diagnostic, not evidence that the corrected endpoints retain
    regular Cartesian axis jets.  Owner mode fails closed by default on raw
    rank, absolute/relative pivot strength, or conditioning defects.
    """
    preferred = np.asarray(regular_axis_preferred, dtype=float)
    rows, rhs = compact_wall_rows(
        compact_derivative,
        lower_robin=lower_robin,
        upper_robin=upper_robin,
        lower_forcing=lower_forcing,
        upper_forcing=upper_forcing,
    )
    if adjustment_scope == "owner_endpoints":
        result = weighted_minimum_correction_on_indices(
            preferred, rows, rhs, (0, len(preferred) - 1), correction_weights,
            owner_pivot_atol=owner_pivot_atol,
            owner_pivot_rtol=owner_pivot_rtol,
            require_resolved_owner=require_resolved_owner,
            **solver_options,
        )
    elif adjustment_scope == "all_nodes":
        result = weighted_minimum_correction(
            preferred, rows, rhs, correction_weights, **solver_options,
        )
        result = ConstrainedClosureResult(
            result.profile, {**result.diagnostics, "adjustment_scope": "all_nodes"},
        )
    else:
        raise ValueError("unknown regular-axis adjustment scope")
    result = _wall_augmented_diagnostics(
        result, rows, rhs, preferred, "regular_axis_plus_compact_walls",
    )
    diagnostics = dict(result.diagnostics)
    diagnostics.update({
        "regular_axis_preferred_maximum_absolute_deviation": float(
            np.max(np.abs(result.profile - preferred), initial=0.0)
        ),
        "regular_axis_preferred_l2_deviation": float(
            np.linalg.norm(result.profile - preferred)
        ),
        "open_regular_axis_preferred_maximum_absolute_deviation": float(
            np.max(np.abs(result.profile[1:-1] - preferred[1:-1]), initial=0.0)
        ),
        "exact_full_preference_plus_walls_consistent": bool(
            np.max(np.abs(rows @ preferred - rhs), initial=0.0) <= 1e-12
        ),
    })
    return ConstrainedClosureResult(result.profile, diagnostics)


def close_outer_face_wall_profile(
    current_outer_profile,
    outgoing_targets,
    compact_derivative,
    *,
    outgoing_indices=None,
    lower_robin=0.0,
    upper_robin=0.0,
    lower_forcing=0.0,
    upper_forcing=0.0,
    correction_weights=None,
    adjustment_scope="owner_endpoints",
    owner_pivot_atol=1.0e-12,
    owner_pivot_rtol=1.0e-12,
    require_resolved_owner=True,
    **solver_options,
):
    """Close one outer-face outgoing profile with both compact-wall rows.

    By default ``outgoing_targets`` has length ``nz-2`` and applies to open
    nodes ``1:-1``.  Current corner values remain in the preferred profile.
    In the default ``owner_endpoints`` scope all non-corner preferences,
    including outgoing targets, remain exact; the two wall-owned corners are
    solved simultaneously.  ``all_nodes`` instead runs the generic weighted
    projector as a comparator and may move outgoing values.  Diagnostics retain
    the outgoing-target mismatch and the residual that a full exact preference
    would have produced.  Owner mode fails closed by default on raw rank,
    absolute/relative pivot strength, or conditioning defects.
    """
    current = np.asarray(current_outer_profile, dtype=float)
    targets = np.asarray(outgoing_targets, dtype=float)
    if current.ndim != 1 or len(current) < 3 or np.any(~np.isfinite(current)):
        raise ValueError("current outer acceleration profile is invalid")
    if outgoing_indices is None:
        indices = np.arange(1, len(current) - 1)
    else:
        indices = np.asarray(outgoing_indices, dtype=int)
    if (
        indices.ndim != 1 or len(indices) == 0 or targets.shape != (len(indices),)
        or np.any(indices <= 0) or np.any(indices >= len(current) - 1)
        or len(np.unique(indices)) != len(indices) or np.any(~np.isfinite(targets))
    ):
        raise ValueError("outgoing targets and open-face indices are invalid")
    preferred = current.copy()
    preferred[indices] = targets
    rows, rhs = compact_wall_rows(
        compact_derivative,
        lower_robin=lower_robin,
        upper_robin=upper_robin,
        lower_forcing=lower_forcing,
        upper_forcing=upper_forcing,
    )
    if adjustment_scope == "owner_endpoints":
        result = weighted_minimum_correction_on_indices(
            preferred, rows, rhs, (0, len(preferred) - 1), correction_weights,
            owner_pivot_atol=owner_pivot_atol,
            owner_pivot_rtol=owner_pivot_rtol,
            require_resolved_owner=require_resolved_owner,
            **solver_options,
        )
    elif adjustment_scope == "all_nodes":
        result = weighted_minimum_correction(
            preferred, rows, rhs, correction_weights, **solver_options,
        )
        result = ConstrainedClosureResult(
            result.profile, {**result.diagnostics, "adjustment_scope": "all_nodes"},
        )
    else:
        raise ValueError("unknown outer-face adjustment scope")
    result = _wall_augmented_diagnostics(
        result, rows, rhs, preferred, "outer_outgoing_plus_compact_walls",
    )
    outgoing_residual = result.profile[indices] - targets
    weights = (
        np.ones_like(current) if correction_weights is None
        else np.asarray(correction_weights, dtype=float)
    )
    diagnostics = dict(result.diagnostics)
    diagnostics.update({
        "outgoing_indices": indices.tolist(),
        "outgoing_target_residuals": outgoing_residual.tolist(),
        "outgoing_target_maximum_absolute_residual": float(
            np.max(np.abs(outgoing_residual), initial=0.0)
        ),
        "outgoing_target_l2_residual": float(np.linalg.norm(outgoing_residual)),
        "outgoing_target_weighted_l2_residual": float(
            np.linalg.norm(weights[indices] * outgoing_residual)
        ),
        "corner_changes_from_current": (
            result.profile[[0, -1]] - current[[0, -1]]
        ).tolist(),
        "maximum_absolute_change_from_current": float(
            np.max(np.abs(result.profile - current), initial=0.0)
        ),
        "exact_full_preference_plus_walls_consistent": bool(
            np.max(np.abs(rows @ preferred - rhs), initial=0.0) <= 1e-12
        ),
    })
    return ConstrainedClosureResult(result.profile, diagnostics)
