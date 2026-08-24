"""Prospective representation diagnostics for the A790 Phase-A2 audit.

These helpers are deliberately separate from the frozen Phase-A production
recipe.  They construct comparison representations only; they do not alter a
parent, repair a target, solve a constraint, or apply a boundary condition.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import make_interp_spline

from bhps.matched_staged_continuum import (
    PARENT_R_MAX,
    ContinuousReducedParent,
    TensorSplineSurface,
)


def _mixed_degree_surface(
    z, r, values, *, z_first=None, z_degree=5, s_degree=3,
    parent_r_max=PARENT_R_MAX,
):
    """Build a z/s tensor spline with an optional first-z wall trace.

    When ``z_first`` is supplied, the quintic compact spline uses the two
    supplied first-derivative conditions and omits the first and last interior
    knots.  This supplies the remaining two not-a-knot-like closures without
    inventing second-derivative endpoint data.  Radial interpolation remains
    cubic and is shared exactly by values and endpoint derivative curves.
    """
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    values = np.asarray(values, dtype=float)
    z_degree = int(z_degree)
    s_degree = int(s_degree)
    parent_r_max = float(parent_r_max)
    if values.shape[:2] != (len(z), len(r)):
        raise ValueError("surface values are not aligned with z,r")
    if z_degree != 5 or s_degree != 3:
        raise ValueError("Phase-A2 endpoint-matched surface must be Q53")
    if len(z) <= z_degree or len(r) <= s_degree:
        raise ValueError("insufficient nodes for Q53 comparison surface")
    if (
        np.any(np.diff(z) <= 0.0) or r[0] != 0.0
        or np.any(np.diff(r) <= 0.0)
        or r[-1] > parent_r_max+1e-12
    ):
        raise ValueError("invalid Q53 source coordinates")
    if not all(np.all(np.isfinite(value)) for value in (z, r, values)):
        raise ValueError("Q53 source data must be finite")
    s = (r/parent_r_max)**2
    radial = make_interp_spline(s, values, k=s_degree, axis=1)
    radial_coefficients = np.moveaxis(radial.c, 0, 1)
    boundary = "not-a-knot"
    if z_first is None:
        compact = make_interp_spline(
            z, radial_coefficients, k=z_degree, axis=0,
        )
    else:
        z_first = np.asarray(z_first, dtype=float)
        if z_first.shape != values.shape or not np.all(np.isfinite(z_first)):
            raise ValueError(
                "Q53 endpoint first derivative has wrong shape or is nonfinite"
            )
        lower = make_interp_spline(s, z_first[0], k=s_degree, axis=0)
        upper = make_interp_spline(s, z_first[-1], k=s_degree, axis=0)
        if not (
            np.array_equal(lower.t, radial.t)
            and np.array_equal(upper.t, radial.t)
        ):
            raise RuntimeError("Q53 radial spline bases differ")
        knots = np.concatenate((
            np.repeat(z[0], z_degree+1),
            z[2:-2],
            np.repeat(z[-1], z_degree+1),
        ))
        compact = make_interp_spline(
            z, radial_coefficients, k=z_degree, axis=0, t=knots,
            bc_type=([(1, lower.c)], [(1, upper.c)]),
        )
        boundary = "first-z-plus-omitted-edge-knots"
    return TensorSplineSurface(
        np.asarray(compact.t).copy(), np.asarray(radial.t).copy(),
        np.asarray(compact.c).copy(), z_degree, s_degree,
        parent_r_max, boundary,
    )


def endpoint_data_matched_q53_parent(
    jet_field, z, r, *, parent_identity="P11-Q53-endpoint-matched",
    parent_r_max=PARENT_R_MAX,
):
    """Return the Phase-A2 Q53 comparison bound to one parent jet.

    Position and velocity use the same archived first-z wall curves as the
    production cubic.  Acceleration remains not-a-knot because ``q_ttz`` is
    unavailable.  All three surfaces keep the production cubic radial basis.
    """
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    if not (
        np.array_equal(z, np.asarray(jet_field.z, dtype=float))
        and np.array_equal(r, np.asarray(jet_field.r, dtype=float))
    ):
        raise ValueError("Q53 parent coordinates differ from source jet")
    q = np.asarray(jet_field.reduced_fields, dtype=float)
    first = np.asarray(jet_field.reduced_first, dtype=float)
    second = np.asarray(jet_field.reduced_second, dtype=float)
    expected = (len(z), len(r), q.shape[-1])
    if (
        q.shape != expected or first.shape != (3, *expected)
        or second.shape != (3, 3, *expected)
    ):
        raise ValueError("invalid Q53 parent jet shapes")
    return ContinuousReducedParent(
        _mixed_degree_surface(
            z, r, q, z_first=first[1], parent_r_max=parent_r_max,
        ),
        _mixed_degree_surface(
            z, r, first[0], z_first=second[0, 1],
            parent_r_max=parent_r_max,
        ),
        _mixed_degree_surface(
            z, r, second[0, 0], parent_r_max=parent_r_max,
        ),
        parent_identity=str(parent_identity),
        parent_nodal_fingerprint="diagnostic-Q53-bound-to-supplied-jet",
    )


def endpoint_trace_comparison(primary, q53, z, r):
    """Score the explicitly shared wall traces and free q_zz outcome."""
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    primary_jet = primary.project(z, r)
    q53_jet = q53.project(z, r)

    def linf(left, right):
        return float(np.max(np.abs(np.asarray(left)-np.asarray(right))))

    endpoints = [0, -1]
    return {
        "position": linf(
            primary_jet.reduced_fields[endpoints],
            q53_jet.reduced_fields[endpoints],
        ),
        "position_r": linf(
            primary_jet.reduced_first[2, endpoints],
            q53_jet.reduced_first[2, endpoints],
        ),
        "position_rr": linf(
            primary_jet.reduced_second[2, 2, endpoints],
            q53_jet.reduced_second[2, 2, endpoints],
        ),
        "position_z": linf(
            primary_jet.reduced_first[1, endpoints],
            q53_jet.reduced_first[1, endpoints],
        ),
        "position_zr": linf(
            primary_jet.reduced_second[1, 2, endpoints],
            q53_jet.reduced_second[1, 2, endpoints],
        ),
        "velocity": linf(
            primary_jet.reduced_first[0, endpoints],
            q53_jet.reduced_first[0, endpoints],
        ),
        "velocity_z": linf(
            primary_jet.reduced_second[0, 1, endpoints],
            q53_jet.reduced_second[0, 1, endpoints],
        ),
        "acceleration": linf(
            primary_jet.reduced_second[0, 0, endpoints],
            q53_jet.reduced_second[0, 0, endpoints],
        ),
        "position_zz_free_outcome": linf(
            primary_jet.reduced_second[1, 1, endpoints],
            q53_jet.reduced_second[1, 1, endpoints],
        ),
    }
