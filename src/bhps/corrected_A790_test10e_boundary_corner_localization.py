"""Archive-only boundary/corner localization addendum for A=7.90 Test 10E.

The diagnostic operates on the scalar acceleration row already archived by
Tests 10C and 10E.  It does not reinterpret the row correction as a physical
flux.  All integrations are performed on the open radial face; the compact
wall endpoints are excluded because the production Sommerfeld row excludes
them.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import PchipInterpolator

from bhps.gw_slice_high_order_solver import derivative_matrix


COMPONENTS = ("Phi", "chi")
PROPER_COLLAR_WIDTHS = (0.05, 0.10, 0.20)
NODE_COLLAR_COUNTS = (1, 3, 5, 10)
STENCIL_WIDTH = 7
ALTERNATE_GAUSS_LEVEL = 16


def cumulative_open_face_proper_coordinate(z_open, q_zz):
    """Return proper distance along the sampled open face.

    Distance zero is the first open node, not the compact-wall endpoint.  This
    convention is forced by the archived face primitives and is reported in
    every result.
    """
    z = np.asarray(z_open, dtype=float)
    compact = np.asarray(q_zz, dtype=float)
    if z.ndim != 1 or compact.shape != z.shape or len(z) < 3:
        raise ValueError("invalid open-face proper-coordinate primitives")
    if np.any(~np.isfinite(z)) or np.any(~np.isfinite(compact)):
        raise ValueError("nonfinite open-face proper-coordinate primitives")
    if np.any(np.diff(z) <= 0.0) or np.any(compact <= 0.0):
        raise ValueError("open-face coordinate or compact metric is not positive")
    increments = 0.5 * (
        np.sqrt(compact[:-1]) + np.sqrt(compact[1:])
    ) * np.diff(z)
    return np.r_[0.0, np.cumsum(increments)]


def clipped_trapezoid(x, values, left, right):
    """Integrate a piecewise-linear nodal field on one clipped interval."""
    coordinate = np.asarray(x, dtype=float)
    field = np.asarray(values, dtype=float)
    left = float(left)
    right = float(right)
    if coordinate.ndim != 1 or field.shape != coordinate.shape:
        raise ValueError("clipped integral inputs are not aligned")
    if np.any(np.diff(coordinate) <= 0.0):
        raise ValueError("clipped integral coordinate is not increasing")
    if left < coordinate[0] - 1e-14 or right > coordinate[-1] + 1e-14 or right < left:
        raise ValueError("clipped integral interval is outside the face")
    left = max(left, float(coordinate[0]))
    right = min(right, float(coordinate[-1]))
    if right <= left:
        return 0.0
    interior = coordinate[(coordinate > left) & (coordinate < right)]
    sample = np.r_[left, interior, right]
    sampled_field = np.interp(sample, coordinate, field)
    return float(np.trapezoid(sampled_field, x=sample))


def _interval_component_integrals(proper, area_density, delta, left, right):
    l1 = []
    squared = []
    for component in range(2):
        value = delta[:, component]
        l1.append(clipped_trapezoid(
            proper, area_density * np.abs(value), left, right,
        ))
        squared.append(clipped_trapezoid(
            proper, area_density * value**2, left, right,
        ))
    return np.asarray(l1), np.asarray(squared)


class _PchipGaussIntegrator:
    """Independent regional evaluator in the same derived proper coordinate."""

    def __init__(self, proper, area_density, delta, level=ALTERNATE_GAUSS_LEVEL):
        self.proper = np.asarray(proper, dtype=float)
        self.area = PchipInterpolator(
            self.proper, np.asarray(area_density, dtype=float), extrapolate=False,
        )
        self.delta = PchipInterpolator(
            self.proper, np.asarray(delta, dtype=float), axis=0, extrapolate=False,
        )
        self.nodes, self.weights = leggauss(int(level))

    def integrate(self, left, right):
        left = float(left)
        right = float(right)
        l1 = np.zeros(2)
        squared = np.zeros(2)
        for cell_left, cell_right in zip(self.proper[:-1], self.proper[1:]):
            a = max(left, float(cell_left))
            b = min(right, float(cell_right))
            if b <= a:
                continue
            midpoint = 0.5 * (a + b)
            halfwidth = 0.5 * (b - a)
            sample = midpoint + halfwidth * self.nodes
            area = np.asarray(self.area(sample), dtype=float)
            value = np.asarray(self.delta(sample), dtype=float)
            if np.any(area <= 0.0) or np.any(~np.isfinite(area)) or np.any(~np.isfinite(value)):
                raise ValueError("alternate regional evaluator is not positive and finite")
            l1 += halfwidth * np.tensordot(
                self.weights, area[:, None] * np.abs(value), axes=(0, 0),
            )
            squared += halfwidth * np.tensordot(
                self.weights, area[:, None] * value**2, axes=(0, 0),
            )
        return l1, squared


def _fractions(region, total):
    region = np.asarray(region, dtype=float)
    total = np.asarray(total, dtype=float)
    out = np.zeros_like(region)
    positive = total > 0.0
    out[positive] = region[positive] / total[positive]
    return out


def compact_wall_derivative_defect(z_full, delta_open, stencil_width=STENCIL_WIDTH):
    """Compute the compact-wall residual change caused by the outer overwrite.

    The production sequence first solves the compact-wall acceleration rows and
    then changes only open outer-face nodes.  The correction is therefore zero
    at both compact-wall/radial-face corners.  For both scalar wall rows the
    change in their endpoint residual is exactly ``D_z correction``: the Robin
    coefficient multiplies a zero endpoint correction and the corner forcing is
    unchanged.  This is a compatibility *defect*, not a continuum commutator or
    physical flux.
    """
    z = np.asarray(z_full, dtype=float)
    correction = np.asarray(delta_open, dtype=float)
    if correction.shape != (len(z) - 2, 2):
        raise ValueError("corner-defect correction is not aligned with z")
    full = np.zeros((len(z), 2), dtype=float)
    full[1:-1] = correction
    operator = derivative_matrix(z, 1, int(stencil_width))
    if hasattr(operator, "toarray"):
        operator = operator.toarray()
    rows = np.asarray(operator, dtype=float)[[0, -1]]
    residual = rows @ full
    scale = np.abs(rows) @ np.abs(full)
    normalized = np.zeros_like(residual)
    positive = scale > 0.0
    normalized[positive] = np.abs(residual[positive]) / scale[positive]
    maximum = np.max(np.abs(full), axis=0)
    dz = np.asarray((z[1] - z[0], z[-1] - z[-2]))[:, None]
    scaled = np.zeros_like(residual)
    nonzero = maximum > 0.0
    scaled[:, nonzero] = (
        np.abs(residual[:, nonzero]) * dz / maximum[None, nonzero]
    )
    return {
        "absolute": np.abs(residual),
        "row_absolute_scale": scale,
        "derivative_cancellation_normalized": normalized,
        "delta_z_scaled": scaled,
    }


def _union_integrals(
    proper, area_density, delta, lower_right, upper_left, alternate=None,
):
    total = float(proper[-1])
    lower_l1, lower_sq = _interval_component_integrals(
        proper, area_density, delta, 0.0, lower_right,
    )
    upper_l1, upper_sq = _interval_component_integrals(
        proper, area_density, delta, upper_left, total,
    )
    if upper_left < lower_right:
        overlap_l1, overlap_sq = _interval_component_integrals(
            proper, area_density, delta, upper_left, lower_right,
        )
    else:
        overlap_l1 = np.zeros(2)
        overlap_sq = np.zeros(2)
    union_l1 = lower_l1 + upper_l1 - overlap_l1
    union_sq = lower_sq + upper_sq - overlap_sq
    central_left = min(lower_right, upper_left)
    central_right = max(lower_right, upper_left)
    if upper_left >= lower_right:
        central_l1, central_sq = _interval_component_integrals(
            proper, area_density, delta, lower_right, upper_left,
        )
    else:
        central_l1 = np.zeros(2)
        central_sq = np.zeros(2)
    result = {
        "lower_l1": lower_l1,
        "lower_squared_l2": lower_sq,
        "upper_l1": upper_l1,
        "upper_squared_l2": upper_sq,
        "union_l1": union_l1,
        "union_squared_l2": union_sq,
        "central_l1": central_l1,
        "central_squared_l2": central_sq,
        "central_interval": np.asarray((central_left, central_right)),
    }
    if alternate is not None:
        alt_total_l1, alt_total_sq = alternate.integrate(0.0, total)
        alt_lower_l1, alt_lower_sq = alternate.integrate(0.0, lower_right)
        alt_upper_l1, alt_upper_sq = alternate.integrate(upper_left, total)
        if upper_left < lower_right:
            alt_overlap_l1, alt_overlap_sq = alternate.integrate(upper_left, lower_right)
        else:
            alt_overlap_l1 = np.zeros(2)
            alt_overlap_sq = np.zeros(2)
        alt_union_l1 = alt_lower_l1 + alt_upper_l1 - alt_overlap_l1
        alt_union_sq = alt_lower_sq + alt_upper_sq - alt_overlap_sq
        if upper_left >= lower_right:
            alt_central_l1, alt_central_sq = alternate.integrate(lower_right, upper_left)
        else:
            alt_central_l1 = np.zeros(2)
            alt_central_sq = np.zeros(2)
        result.update({
            "alternate_total_l1": alt_total_l1,
            "alternate_total_squared_l2": alt_total_sq,
            "alternate_union_l1": alt_union_l1,
            "alternate_union_squared_l2": alt_union_sq,
            "alternate_central_l1": alt_central_l1,
            "alternate_central_squared_l2": alt_central_sq,
        })
    return result


def localize_face_enumeration(
    z_full,
    radius,
    q_perp_open,
    q_zz_open,
    before_open,
    after_open,
    proper_widths=PROPER_COLLAR_WIDTHS,
    node_counts=NODE_COLLAR_COUNTS,
):
    """Localize one archived RK boundary evaluation on the open radial face."""
    z = np.asarray(z_full, dtype=float)
    radius = float(radius)
    z_open = z[1:-1]
    perpendicular = np.asarray(q_perp_open, dtype=float)
    compact = np.asarray(q_zz_open, dtype=float)
    before = np.asarray(before_open, dtype=float)
    after = np.asarray(after_open, dtype=float)
    if perpendicular.shape != z_open.shape or compact.shape != z_open.shape:
        raise ValueError("face metric primitives are not aligned")
    if before.shape != (len(z_open), 2) or after.shape != before.shape:
        raise ValueError("face scalar primitives are not aligned")
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError("outer-face radius is not positive and finite")
    if np.any(~np.isfinite(before)) or np.any(~np.isfinite(after)):
        raise ValueError("face scalar primitives are not finite")
    if np.any(perpendicular <= 0.0) or np.any(~np.isfinite(perpendicular)):
        raise ValueError("open-face transverse metric is not positive and finite")
    delta = after - before
    proper = cumulative_open_face_proper_coordinate(z_open, compact)
    area_density = 4.0 * math.pi * radius**2 * perpendicular
    alternate = _PchipGaussIntegrator(proper, area_density, delta)
    total_l1, total_sq = _interval_component_integrals(
        proper, area_density, delta, 0.0, float(proper[-1]),
    )
    total_sq_combined = float(np.sum(total_sq))
    total_l1_combined = float(np.sum(total_l1))

    pointwise = np.max(np.abs(delta), axis=0)
    point_indices = np.argmax(np.abs(delta), axis=0)
    point_z = z_open[point_indices]
    point_proper = proper[point_indices]
    point_nearest_edge = np.minimum(point_proper, proper[-1] - point_proper)
    combined_pointwise_field = np.linalg.norm(delta, axis=1)
    combined_pointwise_index = int(np.argmax(combined_pointwise_field))

    proper_records = []
    for width in proper_widths:
        width = float(width)
        if width <= 0.0 or 2.0 * width >= float(proper[-1]):
            raise ValueError("fixed proper collar does not fit open face")
        record = _union_integrals(
            proper, area_density, delta, width, float(proper[-1]) - width,
            alternate=alternate,
        )
        alternate_total_sq_combined = float(np.sum(record["alternate_total_squared_l2"]))
        alternate_union_sq_combined = float(np.sum(record["alternate_union_squared_l2"]))
        alternate_union_fraction = (
            0.0 if alternate_total_sq_combined == 0.0
            else alternate_union_sq_combined / alternate_total_sq_combined
        )
        proper_records.append({
            "width": width,
            **record,
            "lower_l1_fraction": _fractions(record["lower_l1"], total_l1),
            "lower_squared_l2_fraction": _fractions(
                record["lower_squared_l2"], total_sq,
            ),
            "upper_l1_fraction": _fractions(record["upper_l1"], total_l1),
            "upper_squared_l2_fraction": _fractions(
                record["upper_squared_l2"], total_sq,
            ),
            "union_l1_fraction": _fractions(record["union_l1"], total_l1),
            "union_squared_l2_fraction": _fractions(
                record["union_squared_l2"], total_sq,
            ),
            "central_squared_l2_fraction": _fractions(
                record["central_squared_l2"], total_sq,
            ),
            "combined_union_l1_fraction": (
                0.0 if total_l1_combined == 0.0
                else float(np.sum(record["union_l1"]) / total_l1_combined)
            ),
            "combined_lower_squared_l2_fraction": (
                0.0 if total_sq_combined == 0.0
                else float(np.sum(record["lower_squared_l2"]) / total_sq_combined)
            ),
            "combined_upper_squared_l2_fraction": (
                0.0 if total_sq_combined == 0.0
                else float(np.sum(record["upper_squared_l2"]) / total_sq_combined)
            ),
            "combined_union_squared_l2_fraction": (
                0.0 if total_sq_combined == 0.0
                else float(np.sum(record["union_squared_l2"]) / total_sq_combined)
            ),
            "combined_central_squared_l2_fraction": (
                0.0 if total_sq_combined == 0.0
                else float(np.sum(record["central_squared_l2"]) / total_sq_combined)
            ),
            "alternate_combined_union_squared_l2_fraction": float(
                alternate_union_fraction
            ),
            "primary_alternate_union_squared_fraction_absolute_difference": float(
                abs(
                    alternate_union_fraction
                    - (
                        0.0 if total_sq_combined == 0.0
                        else np.sum(record["union_squared_l2"]) / total_sq_combined
                    )
                )
            ),
        })

    node_records = []
    for count in node_counts:
        count = int(count)
        if count <= 0 or 2 * count >= len(proper):
            raise ValueError("fixed-node collar count does not fit open face")
        lower_right = 0.5 * (proper[count - 1] + proper[count])
        upper_left = 0.5 * (proper[-count] + proper[-count - 1])
        record = _union_integrals(
            proper, area_density, delta, lower_right, upper_left,
            alternate=alternate,
        )
        alternate_total_sq_combined = float(np.sum(record["alternate_total_squared_l2"]))
        alternate_union_sq_combined = float(np.sum(record["alternate_union_squared_l2"]))
        alternate_union_fraction = (
            0.0 if alternate_total_sq_combined == 0.0
            else alternate_union_sq_combined / alternate_total_sq_combined
        )
        node_records.append({
            "count": count,
            "lower_proper_span": float(lower_right),
            "upper_proper_span": float(proper[-1] - upper_left),
            **record,
            "lower_l1_fraction": _fractions(record["lower_l1"], total_l1),
            "lower_squared_l2_fraction": _fractions(
                record["lower_squared_l2"], total_sq,
            ),
            "upper_l1_fraction": _fractions(record["upper_l1"], total_l1),
            "upper_squared_l2_fraction": _fractions(
                record["upper_squared_l2"], total_sq,
            ),
            "union_l1_fraction": _fractions(record["union_l1"], total_l1),
            "union_squared_l2_fraction": _fractions(
                record["union_squared_l2"], total_sq,
            ),
            "combined_union_l1_fraction": (
                0.0 if total_l1_combined == 0.0
                else float(np.sum(record["union_l1"]) / total_l1_combined)
            ),
            "combined_lower_squared_l2_fraction": (
                0.0 if total_sq_combined == 0.0
                else float(np.sum(record["lower_squared_l2"]) / total_sq_combined)
            ),
            "combined_upper_squared_l2_fraction": (
                0.0 if total_sq_combined == 0.0
                else float(np.sum(record["upper_squared_l2"]) / total_sq_combined)
            ),
            "combined_union_squared_l2_fraction": (
                0.0 if total_sq_combined == 0.0
                else float(np.sum(record["union_squared_l2"]) / total_sq_combined)
            ),
            "central_l1_fraction": _fractions(record["central_l1"], total_l1),
            "central_squared_l2_fraction": _fractions(
                record["central_squared_l2"], total_sq,
            ),
            "combined_central_squared_l2_fraction": (
                0.0 if total_sq_combined == 0.0
                else float(np.sum(record["central_squared_l2"]) / total_sq_combined)
            ),
            "alternate_combined_union_squared_l2_fraction": float(
                alternate_union_fraction
            ),
            "primary_alternate_union_squared_fraction_absolute_difference": float(
                abs(
                    alternate_union_fraction
                    - (
                        0.0 if total_sq_combined == 0.0
                        else np.sum(record["union_squared_l2"]) / total_sq_combined
                    )
                )
            ),
        })

    compatibility = compact_wall_derivative_defect(z, delta)
    return {
        "proper_length_open_nodes": float(proper[-1]),
        "total_l1": total_l1,
        "total_squared_l2": total_sq,
        "total_l2": np.sqrt(np.maximum(total_sq, 0.0)),
        "combined_total_l1": total_l1_combined,
        "combined_total_squared_l2": total_sq_combined,
        "combined_total_l2": math.sqrt(max(total_sq_combined, 0.0)),
        "phi_squared_l2_fraction": (
            0.0 if total_sq_combined == 0.0 else float(total_sq[0] / total_sq_combined)
        ),
        "pointwise_maximum": pointwise,
        "pointwise_index": point_indices,
        "pointwise_z": point_z,
        "pointwise_proper_from_first_open": point_proper,
        "pointwise_proper_to_nearest_open_edge": point_nearest_edge,
        "combined_pointwise_maximum": float(combined_pointwise_field[combined_pointwise_index]),
        "combined_pointwise_index": combined_pointwise_index,
        "combined_pointwise_z": float(z_open[combined_pointwise_index]),
        "combined_pointwise_proper_from_first_open": float(
            proper[combined_pointwise_index]
        ),
        "combined_pointwise_proper_to_nearest_open_edge": float(
            min(proper[combined_pointwise_index], proper[-1] - proper[combined_pointwise_index])
        ),
        "proper_collars": proper_records,
        "node_collars": node_records,
        "corner_compatibility": compatibility,
    }


def empirical_power_law(intervals, values):
    """Return a descriptive log-log slope; this is not a convergence order."""
    counts = np.asarray(intervals, dtype=float)
    data = np.asarray(values, dtype=float)
    if counts.ndim != 1 or data.shape != counts.shape or len(counts) < 2:
        raise ValueError("power-law inputs are not aligned")
    if np.any(counts <= 0.0) or np.any(data <= 0.0) or np.any(~np.isfinite(data)):
        return None
    return float(np.polyfit(np.log(counts), np.log(data), 1)[0])


def manufactured_controls():
    """Self/adverse controls for face-wide, fixed-proper and fixed-node layers."""
    grids = (113, 129)
    records = {}
    for size in grids:
        z = np.linspace(1.0, 2.0, size)
        z_open = z[1:-1]
        q = np.ones_like(z_open)
        x = z_open - z[0]
        smooth = x**2 * (1.0 - x) ** 2
        fixed_proper = x**2 * np.exp(-x / 0.08) + (1.0 - x) ** 2 * np.exp(
            -(1.0 - x) / 0.08
        )
        node_index = np.minimum(
            np.arange(1, size - 1), np.arange(size - 2, 0, -1),
        )
        fixed_node = np.exp(-(node_index - 1.0))
        profiles = {
            "zero": np.zeros_like(x),
            "smooth_face": smooth,
            "fixed_proper_layer": fixed_proper,
            "fixed_node_layer": fixed_node,
        }
        records[size] = {}
        for name, profile in profiles.items():
            before = np.ones((len(x), 2))
            after = before.copy()
            after[:, 0] += profile
            records[size][name] = localize_face_enumeration(
                z, 1.0, q, q, before, after,
            )

    coarse, fine = grids
    smooth_defect = max(
        np.max(records[size]["smooth_face"]["corner_compatibility"]["absolute"])
        for size in grids
    )
    zero_norm = max(records[size]["zero"]["combined_total_l2"] for size in grids)
    fixed_proper_fraction = [
        records[size]["fixed_proper_layer"]["proper_collars"][1][
            "combined_union_squared_l2_fraction"
        ] for size in grids
    ]
    fixed_proper_node_fraction = [
        records[size]["fixed_proper_layer"]["node_collars"][1][
            "combined_union_squared_l2_fraction"
        ] for size in grids
    ]
    fixed_node_fraction = [
        records[size]["fixed_node_layer"]["node_collars"][1][
            "combined_union_squared_l2_fraction"
        ] for size in grids
    ]
    fixed_node_span = [
        records[size]["fixed_node_layer"]["node_collars"][1]["lower_proper_span"]
        for size in grids
    ]
    fixed_node_defect = [
        float(np.max(records[size]["fixed_node_layer"]["corner_compatibility"]["absolute"]))
        for size in grids
    ]
    alternate_differences = [
        records[size][profile][family][index][
            "primary_alternate_union_squared_fraction_absolute_difference"
        ]
        for size in grids
        for profile in ("smooth_face", "fixed_proper_layer", "fixed_node_layer")
        for family, index in (("proper_collars", 1), ("node_collars", 1))
    ]
    gates = {
        "zero_is_zero": bool(zero_norm == 0.0),
        "smooth_compatible": bool(smooth_defect < 1e-9),
        "fixed_proper_collar_stable": bool(
            abs(fixed_proper_fraction[1] - fixed_proper_fraction[0]) < 0.03
        ),
        "fixed_proper_node_fraction_decreases": bool(
            fixed_proper_node_fraction[1] < fixed_proper_node_fraction[0]
        ),
        "fixed_node_fraction_non_decreasing": bool(
            fixed_node_fraction[1] >= fixed_node_fraction[0] - 1e-12
        ),
        "fixed_node_span_shrinks": bool(fixed_node_span[1] < fixed_node_span[0]),
        "fixed_node_defect_grows": bool(fixed_node_defect[1] > fixed_node_defect[0]),
        "alternate_regional_evaluator_agrees": bool(max(alternate_differences) < 0.03),
    }
    return {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "smooth_maximum_corner_defect": float(smooth_defect),
        "fixed_proper_width_0p10_fraction": fixed_proper_fraction,
        "fixed_proper_three_node_fraction": fixed_proper_node_fraction,
        "fixed_node_three_node_fraction": fixed_node_fraction,
        "fixed_node_three_node_lower_span": fixed_node_span,
        "fixed_node_maximum_corner_defect": fixed_node_defect,
        "maximum_primary_alternate_fraction_difference": float(max(alternate_differences)),
    }
