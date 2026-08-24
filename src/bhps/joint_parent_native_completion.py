"""Native position and gauge completion for the Protocol-125 parent.

The routines in this module are construction primitives only.  They operate
once on a parent grid, never on a target grid, and do not evaluate an
evolution RHS.  The compact wall owns both radial corners.
"""

from __future__ import annotations

import numpy as np

from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.junction_preservation_diagnostic import (
    wall_junction_rows,
    wall_source_coefficients,
)
from bhps.nonlinear_regular_so3_evolution import (
    compact_wall_normal_gauge_position_residuals,
)


FIELD_COUNT = 9


class Protocol125NativePositionPrerequisiteFailure(RuntimeError):
    """A finite, measured sphere/Phi prerequisite failure.

    This is scientific failure evidence, not a malformed-input exception.  It
    is raised only after the native rows have been evaluated on a valid finite
    parent state, so the construction layer can archive an ordered
    ``FAIL-parent-position`` stop without authorizing a repair.
    """

    def __init__(self, measurements, ceiling):
        measurements = {
            str(name): float(value) for name, value in measurements.items()
        }
        ceiling = float(ceiling)
        if (
            tuple(measurements) != (
                "sphere_metric_normalized_Linf", "Phi_robin_Linf",
            )
            or not all(np.isfinite(value) and value >= 0.0 for value in measurements.values())
            or not np.isfinite(ceiling)
            or ceiling <= 0.0
            or max(measurements.values()) < ceiling
        ):
            raise ValueError("native prerequisite failure evidence is invalid")
        self.measurements = measurements
        self.ceiling = ceiling
        super().__init__(
            "joint parent sphere/Phi rows do not satisfy the prerequisite"
        )


def _dense_derivative(points, order=1, stencil_width=7):
    matrix = derivative_matrix(
        np.asarray(points, dtype=float), int(order), int(stencil_width),
    )
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def _validate_position(position, z, r):
    q = np.asarray(position, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    if q.shape != (len(z), len(r), FIELD_COUNT):
        raise ValueError("position must have shape (z,r,9)")
    if (
        len(z) < 7 or len(r) < 7 or r[0] != 0.0
        or np.any(np.diff(z) <= 0.0) or np.any(np.diff(r) <= 0.0)
        or not all(np.all(np.isfinite(value)) for value in (q, z, r))
    ):
        raise ValueError("invalid native parent position or coordinates")
    if np.any(q[:, :, 6] <= 0.0):
        raise ValueError("native parent g_zz must be positive")
    return q, z, r


def _solve_two_endpoint_robin(values, dz, coefficients, forcing=None):
    """Solve two endpoint rows while preserving every open value.

    Each row is ``D values + coefficient*value + forcing = 0``.  The two
    endpoint values are solved simultaneously so the helper remains correct
    even when a compact stencil spans the full compact interval.
    """
    values = np.asarray(values, dtype=float)
    dz = np.asarray(dz, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if values.ndim != 1 or dz.shape != (len(values), len(values)):
        raise ValueError("invalid endpoint Robin data")
    if coefficients.shape != (2,):
        raise ValueError("two endpoint coefficients are required")
    forcing = np.zeros(2) if forcing is None else np.asarray(forcing, dtype=float)
    if forcing.shape != (2,):
        raise ValueError("two endpoint forcing values are required")
    endpoints = np.asarray((0, len(values)-1), dtype=int)
    interior = np.arange(1, len(values)-1)
    matrix = dz[endpoints][:, endpoints].copy()
    matrix[np.arange(2), np.arange(2)] += coefficients
    rhs = -(dz[endpoints][:, interior] @ values[interior] + forcing)
    condition = float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition > 1e12:
        raise RuntimeError("native endpoint Robin block is ill conditioned")
    solved = np.linalg.solve(matrix, rhs)
    result = values.copy()
    result[endpoints] = solved
    residual = dz[endpoints] @ result + coefficients*result[endpoints] + forcing
    return result, {
        "condition": condition,
        "maximum_absolute_residual": float(np.max(np.abs(residual))),
        "endpoint_correction_Linf": float(np.max(np.abs(
            solved-values[endpoints]
        ))),
    }


def _wall_absolute_rows(position, z, r, background, stencil_width=7):
    zero = np.zeros_like(position)
    records = {
        wall: wall_junction_rows(
            position, zero, z, r, background, wall, stencil_width,
        ) for wall in ("lower", "upper")
    }
    return records


def _maximum_row(records, section, row=None):
    maxima = []
    for wall in ("lower", "upper"):
        record = records[wall]
        if section == "metric":
            maxima.append(np.max(np.abs(
                record["components"][row]["robin_normalized"]
            )))
        else:
            maxima.append(np.max(np.abs(record["separate_rows"][section])))
    return float(max(maxima))


def analytic_even_q4_limit(position, r, stencil_width=7):
    """Return the squared-radius derivative limit of ``(h_rr-h_perp)/r^2``.

    The numerator, not q4, is differentiated.  This avoids a free axis fit and
    is the discrete counterpart of ``lim N/r^2 = partial_s N/R^2`` with
    ``s=(r/R)^2``.
    """
    q = np.asarray(position, dtype=float)
    r = np.asarray(r, dtype=float)
    if q.ndim != 3 or q.shape[1] != len(r) or q.shape[2] != FIELD_COUNT:
        raise ValueError("invalid q4 limit data")
    if r[0] != 0.0 or r[-1] <= 0.0:
        raise ValueError("q4 limit requires a positive radial domain")
    radius2 = r**2
    numerator = radius2[None, :]*q[:, :, 4]
    s = (r/r[-1])**2
    ds = _dense_derivative(s, 1, stencil_width)
    return (ds @ numerator.T).T[:, 0]/r[-1]**2


def complete_native_parent_position(
    position, z, r, background, *, stencil_width=7,
    prerequisite_tolerance=1e-10,
):
    """Complete only the Protocol-125 parent-owned position channels.

    The supplied sphere-metric and Phi rows are prerequisites and remain
    bitwise fixed.  The completion owns wall ``h_00``, radial anisotropy,
    chi, the two positive-zero normal-tangential fields, and the analytic q4
    axis limit.  It never changes an open compact-interior value.
    """
    q0, z, r = _validate_position(position, z, r)
    normal_tangential = q0[:, :, 0:2]
    if (
        np.any(normal_tangential != 0.0)
        or np.any(np.signbit(normal_tangential))
    ):
        raise RuntimeError(
            "normal-tangential position must already be IEEE positive zero"
        )
    q = q0.copy()
    dz = _dense_derivative(z, 1, stencil_width)
    before = _wall_absolute_rows(q, z, r, background, stencil_width)
    prerequisite = {
        "sphere_metric_normalized_Linf": _maximum_row(
            before, "metric", "sphere",
        ),
        "Phi_robin_Linf": _maximum_row(before, "Phi_robin"),
    }
    if max(prerequisite.values()) >= float(prerequisite_tolerance):
        raise Protocol125NativePositionPrerequisiteFailure(
            prerequisite,
            prerequisite_tolerance,
        )

    wall_indices = (0, len(z)-1)
    sources = [
        wall_source_coefficients(
            q[index, :, 7], background, wall,
        ) for wall, index in zip(("lower", "upper"), wall_indices)
    ]
    A = np.stack((
        np.sqrt(q[wall_indices[0], :, 6]),
        np.sqrt(q[wall_indices[1], :, 6]),
    ))
    beta = np.stack((sources[0]["beta"], sources[1]["beta"]))
    robin = 2.0*beta*A

    diagnostics = {
        "prerequisite": prerequisite,
        "h00": [], "radial_metric": [], "chi": [],
    }
    for radial in range(len(r)):
        q[:, radial, 2], record = _solve_two_endpoint_robin(
            q[:, radial, 2], dz, robin[:, radial],
        )
        diagnostics["h00"].append(record)

    radial_metric = q[:, :, 3] + r[None, :]**2*q[:, :, 4]
    for radial in range(1, len(r)):
        radial_metric[:, radial], record = _solve_two_endpoint_robin(
            radial_metric[:, radial], dz, robin[:, radial],
        )
        q[0, radial, 4] = (
            radial_metric[0, radial]-q[0, radial, 3]
        )/r[radial]**2
        q[-1, radial, 4] = (
            radial_metric[-1, radial]-q[-1, radial, 3]
        )/r[radial]**2
        diagnostics["radial_metric"].append(record)

    for radial in range(len(r)):
        q[:, radial, 8], record = _solve_two_endpoint_robin(
            q[:, radial, 8], dz, np.zeros(2),
        )
        diagnostics["chi"].append(record)

    q[:, 0, 4] = analytic_even_q4_limit(q, r, stencil_width)
    after = _wall_absolute_rows(q, z, r, background, stencil_width)
    final = {
        "metric_normalized_Linf": float(max(
            _maximum_row(after, "metric", row)
            for row in ("tt", "rr", "sphere", "tr")
        )),
        "Phi_robin_Linf": _maximum_row(after, "Phi_robin"),
        "chi_neumann_Linf": _maximum_row(after, "chi_neumann"),
    }
    allowed = np.zeros_like(q, dtype=bool)
    allowed[[0, -1], :, 2] = True
    allowed[[0, -1], 1:, 4] = True
    allowed[[0, -1], :, 8] = True
    allowed[:, 0, 4] = True
    changed = np.ascontiguousarray(q).view(np.uint64).reshape(q.shape) != (
        np.ascontiguousarray(q0).view(np.uint64).reshape(q0.shape)
    )
    ownership_pass = bool(not np.any(changed & ~allowed))
    sphere_phi_bitwise = bool(
        np.array_equal(q[:, :, 3], q0[:, :, 3])
        and np.array_equal(q[:, :, 7], q0[:, :, 7])
    )

    def scaled_linf(before_values, after_values):
        denominator = np.maximum.reduce((
            np.ones_like(before_values),
            np.abs(before_values),
            np.abs(after_values),
        ))
        return float(np.max(np.abs(after_values-before_values)/denominator))

    wall = np.asarray([0, len(z)-1])
    hrr_before = q0[wall, :, 3]+r[None, :]**2*q0[wall, :, 4]
    hrr_after = q[wall, :, 3]+r[None, :]**2*q[wall, :, 4]
    axis_q4_before = 2.0*q0[:, 0, 4]
    axis_q4_after = 2.0*q[:, 0, 4]
    completion_corrections = {
        "lapse_h00_normalized_Linf": scaled_linf(
            q0[wall, :, 2], q[wall, :, 2],
        ),
        "anisotropy_hrr_normalized_Linf": scaled_linf(
            hrr_before, hrr_after,
        ),
        "chi_normalized_Linf": scaled_linf(
            q0[wall, :, 8], q[wall, :, 8],
        ),
        "q4_reduced_absolute_Linf": float(np.max(np.abs(
            q[:, :, 4]-q0[:, :, 4]
        ))),
        "axis_q4_second_derivative_image_normalized_Linf": scaled_linf(
            axis_q4_before, axis_q4_after,
        ),
        "normal_tangential_positive_zero_noop": bool(
            np.array_equal(q[:, :, 0:2], q0[:, :, 0:2])
            and not np.any(np.signbit(q[:, :, 0:2]))
        ),
    }
    diagnostics.update({
        "final": final,
        "ownership_pass": ownership_pass,
        "sphere_and_Phi_bitwise": sphere_phi_bitwise,
        "changed_value_count": int(np.count_nonzero(changed)),
        "completion_corrections": completion_corrections,
        "q4_axis_method": "d_s(h_rr-h_perp)/Rmax^2 at s=0",
        "finite": bool(np.all(np.isfinite(q))),
    })
    if not ownership_pass or not sphere_phi_bitwise:
        raise RuntimeError("native position completion violated ownership")
    return q, diagnostics


def complete_normal_gauge_source_wall(
    position, source, z, r, background, *, stencil_width=7,
):
    """Set only the normal source trace required by the native GH wall row."""
    q, z, r = _validate_position(position, z, r)
    h = np.asarray(source, dtype=float)
    if h.shape != (len(z), len(r), 3) or not np.all(np.isfinite(h)):
        raise ValueError("invalid reduced gauge source")
    result = h.copy()
    dz = _dense_derivative(z, 1, stencil_width)
    G = q[:, :, 6]
    for wall, index in (("lower", 0), ("upper", -1)):
        beta = wall_source_coefficients(
            q[index, :, 7], background, wall,
        )["beta"]
        result[index, :, 1] = (
            (dz @ G)[index] + 8.0*beta*G[index]**1.5
        )/(2.0*G[index])
    audit = compact_wall_normal_gauge_position_residuals(
        q, result, z, r, background, stencil_width, 0,
    )
    allowed = np.zeros_like(result, dtype=bool)
    allowed[[0, -1], :, 1] = True
    changed = np.ascontiguousarray(result).view(np.uint64).reshape(result.shape) != (
        np.ascontiguousarray(h).view(np.uint64).reshape(h.shape)
    )
    if np.any(changed & ~allowed):
        raise RuntimeError("normal gauge completion changed an unowned source")
    return result, {
        "normal_gauge": audit,
        "ownership_pass": True,
        "changed_value_count": int(np.count_nonzero(changed)),
        "finite": bool(np.all(np.isfinite(result))),
    }
