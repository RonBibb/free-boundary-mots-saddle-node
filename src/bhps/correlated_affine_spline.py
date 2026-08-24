"""One-parameter affine arithmetic for correlated Test-4D spline screens."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from bhps.validated_capped_surface_shooting import (
    VInterval,
    as_interval,
    interval_hull,
    interval_sqrt,
)


EPSILON = VInterval(-1.0, 1.0)


def _rounding_error(value, enclosure):
    return as_interval(enclosure) - VInterval.point(value)


@dataclass(frozen=True)
class AffineForm:
    """c + a*epsilon + remainder, with one shared epsilon in [-1,1]."""

    center: float
    linear: float
    remainder: VInterval

    @classmethod
    def constant(cls, value):
        return cls(float(value), 0.0, VInterval.point(0.0))

    @classmethod
    def parameter(cls, center, linear):
        return cls(float(center), float(linear), VInterval.point(0.0))

    @property
    def range(self):
        return (
            VInterval.point(self.center)
            + VInterval(-abs(self.linear), abs(self.linear))
            + self.remainder
        )

    def __neg__(self):
        return AffineForm(-self.center, -self.linear, -self.remainder)

    def __add__(self, other):
        other = as_affine(other)
        center = self.center + other.center
        linear = self.linear + other.linear
        center_error = _rounding_error(
            center,
            VInterval.point(self.center) + VInterval.point(other.center),
        )
        linear_error = _rounding_error(
            linear,
            VInterval.point(self.linear) + VInterval.point(other.linear),
        )
        return AffineForm(
            center,
            linear,
            self.remainder + other.remainder
            + center_error + linear_error * EPSILON,
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-as_affine(other))

    def __rsub__(self, other):
        return as_affine(other) - self

    def __mul__(self, other):
        other = as_affine(other)
        center = self.center * other.center
        linear = self.center * other.linear + self.linear * other.center
        center_error = _rounding_error(
            center,
            VInterval.point(self.center) * VInterval.point(other.center),
        )
        linear_exact = (
            VInterval.point(self.center) * VInterval.point(other.linear)
            + VInterval.point(self.linear) * VInterval.point(other.center)
        )
        linear_error = _rounding_error(linear, linear_exact)
        self_linear = VInterval(-abs(self.linear), abs(self.linear))
        other_linear = VInterval(-abs(other.linear), abs(other.linear))
        self_affine_range = VInterval.point(self.center) + self_linear
        other_affine_range = VInterval.point(other.center) + other_linear
        nonlinear = (
            VInterval.point(self.linear * other.linear) * VInterval(0.0, 1.0)
            + self_affine_range * other.remainder
            + other_affine_range * self.remainder
            + self.remainder * other.remainder
        )
        return AffineForm(
            center,
            linear,
            nonlinear + center_error + linear_error * EPSILON,
        )

    __rmul__ = __mul__

    def reciprocal(self):
        if self.range.lower <= 0.0 <= self.range.upper:
            raise ZeroDivisionError("affine denominator contains zero")
        center = 1.0 / self.center
        linear = -self.linear / self.center**2
        center_error = _rounding_error(
            center, VInterval.point(self.center).reciprocal(),
        )
        linear_exact = (
            -VInterval.point(self.linear) / VInterval.point(self.center)**2
        )
        linear_error = _rounding_error(linear, linear_exact)
        delta = (
            VInterval(-abs(self.linear), abs(self.linear)) + self.remainder
        )
        remainder = (
            -self.remainder / VInterval.point(self.center)**2
            + delta**2 / (
                VInterval.point(self.center)**2 * self.range
            )
        )
        return AffineForm(
            center,
            linear,
            remainder + center_error + linear_error * EPSILON,
        )

    def __truediv__(self, other):
        return self * as_affine(other).reciprocal()

    def __rtruediv__(self, other):
        return as_affine(other) / self

    def __pow__(self, exponent):
        exponent = int(exponent)
        if exponent == 0:
            return AffineForm.constant(1.0)
        if exponent < 0:
            return (self ** (-exponent)).reciprocal()
        output = AffineForm.constant(1.0)
        base = self
        while exponent:
            if exponent & 1:
                output = output * base
            exponent //= 2
            if exponent:
                base = base * base
        return output


def as_affine(value):
    return value if isinstance(value, AffineForm) else AffineForm.constant(value)


def affine_sqrt(value):
    value = as_affine(value)
    if value.range.lower <= 0.0:
        raise ValueError("affine sqrt is not strictly positive")
    center = math.sqrt(value.center)
    linear = value.linear / (2.0 * center)
    center_error = _rounding_error(
        center, interval_sqrt(VInterval.point(value.center)),
    )
    linear_exact = (
        VInterval.point(value.linear) / (2.0 * VInterval.point(center))
    )
    linear_error = _rounding_error(linear, linear_exact)
    delta = VInterval(-abs(value.linear), abs(value.linear)) + value.remainder
    root_range = interval_sqrt(value.range)
    nonlinear = -delta**2 / (
        2.0 * VInterval.point(center)
        * (root_range + VInterval.point(center))**2
    )
    remainder = (
        value.remainder / (2.0 * VInterval.point(center)) + nonlinear
    )
    return AffineForm(
        center,
        linear,
        remainder + center_error + linear_error * EPSILON,
    )


@dataclass(frozen=True)
class AffineJet:
    value: AffineForm
    derivative: tuple[AffineForm, AffineForm]

    @classmethod
    def constant(cls, value):
        zero = AffineForm.constant(0.0)
        return cls(as_affine(value), (zero, zero))

    @classmethod
    def variable(cls, value, index):
        derivatives = [AffineForm.constant(0.0), AffineForm.constant(0.0)]
        derivatives[int(index)] = AffineForm.constant(1.0)
        return cls(as_affine(value), tuple(derivatives))

    def __neg__(self):
        return AffineJet(-self.value, tuple(-item for item in self.derivative))

    def __add__(self, other):
        other = as_affine_jet(other)
        return AffineJet(
            self.value + other.value,
            tuple(a + b for a, b in zip(self.derivative, other.derivative)),
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-as_affine_jet(other))

    def __rsub__(self, other):
        return as_affine_jet(other) - self

    def __mul__(self, other):
        other = as_affine_jet(other)
        return AffineJet(
            self.value * other.value,
            tuple(
                left * other.value + self.value * right
                for left, right in zip(self.derivative, other.derivative)
            ),
        )

    __rmul__ = __mul__

    def reciprocal(self):
        reciprocal = self.value.reciprocal()
        return AffineJet(
            reciprocal,
            tuple(-item * reciprocal**2 for item in self.derivative),
        )

    def __truediv__(self, other):
        return self * as_affine_jet(other).reciprocal()

    def __rtruediv__(self, other):
        return as_affine_jet(other) / self

    def __pow__(self, exponent):
        exponent = int(exponent)
        if exponent == 0:
            return AffineJet.constant(1.0)
        if exponent < 0:
            return (self ** (-exponent)).reciprocal()
        value = self.value**exponent
        factor = exponent * self.value ** (exponent - 1)
        return AffineJet(
            value, tuple(factor * item for item in self.derivative),
        )


def as_affine_jet(value):
    return value if isinstance(value, AffineJet) else AffineJet.constant(value)


def affine_jet_sqrt(value):
    value = as_affine_jet(value)
    root = affine_sqrt(value.value)
    return AffineJet(
        root, tuple(item / (2.0 * root) for item in value.derivative),
    )


def _basis_levels(knots, degree, coordinate, span):
    zero = AffineJet.constant(0.0)
    one = AffineJet.constant(1.0)
    values = [{int(span): one}]
    first = [{int(span): zero}]
    second = [{int(span): zero}]
    for local_degree in range(1, int(degree) + 1):
        current = {}
        current_first = {}
        current_second = {}
        previous = values[-1]
        previous_first = first[-1]
        for index in range(int(span) - local_degree, int(span) + 1):
            value = zero
            derivative = zero
            derivative_second = zero
            left_denominator = knots[index + local_degree] - knots[index]
            if left_denominator != 0.0:
                scale = 1.0 / left_denominator
                value = value + (
                    (coordinate - knots[index]) * scale
                    * previous.get(index, zero)
                )
                derivative = (
                    derivative + local_degree * scale * previous.get(index, zero)
                )
                derivative_second = (
                    derivative_second
                    + local_degree * scale * previous_first.get(index, zero)
                )
            right_denominator = knots[index + local_degree + 1] - knots[index + 1]
            if right_denominator != 0.0:
                scale = 1.0 / right_denominator
                value = value + (
                    (knots[index + local_degree + 1] - coordinate) * scale
                    * previous.get(index + 1, zero)
                )
                derivative = (
                    derivative - local_degree * scale
                    * previous.get(index + 1, zero)
                )
                derivative_second = (
                    derivative_second - local_degree * scale
                    * previous_first.get(index + 1, zero)
                )
            current[index] = value
            current_first[index] = derivative
            current_second[index] = derivative_second
        values.append(current)
        first.append(current_first)
        second.append(current_second)
    return values[-1], first[-1], second[-1]


def _span(knots, degree, coordinate):
    span = int(np.searchsorted(knots, coordinate, side="right") - 1)
    return min(max(span, degree), len(knots) - degree - 2)


def _field_jets(spline, z, r):
    zspan = _span(spline.knots_z, spline.degree_z, z.value.center)
    rspan = _span(spline.knots_r, spline.degree_r, r.value.center)
    zbasis = _basis_levels(spline.knots_z, spline.degree_z, z, zspan)
    rbasis = _basis_levels(spline.knots_r, spline.degree_r, r, rspan)
    output = []
    for dz, dr in ((0, 0), (1, 0), (0, 1)):
        total = AffineJet.constant(0.0)
        for i, bz in zbasis[dz].items():
            if not 0 <= i < spline.coefficients.shape[0]:
                continue
            for j, br in rbasis[dr].items():
                if not 0 <= j < spline.coefficients.shape[1]:
                    continue
                total = total + spline.coefficients[i, j] * bz * br
        output.append(total)
    return output


def _divergence_rhs(theta, rho, momentum, metric):
    sine = math.sin(float(theta))
    cosine = math.cos(float(theta))
    slope = momentum / sine**2
    z = metric.z_brane - rho * cosine
    r = rho * sine
    fields = {}
    for name in ("A", "B", "C"):
        fields[name], fields[name + "z"], fields[name + "r"] = (
            _field_jets(metric.fields[name], z, r)
        )
    A, B, C = fields["A"], fields["B"], fields["C"]
    Az, Ar = fields["Az"], fields["Ar"]
    Bz, Br = fields["Bz"], fields["Br"]
    Cz, Cr = fields["Cz"], fields["Cr"]
    aa = A**2
    bb = B**2
    aaz = 2 * A * Az
    aar = 2 * A * Ar
    bbz = 2 * B * Bz
    bbr = 2 * B * Br
    aaq = -cosine * aaz + sine * aar
    bbq = -cosine * bbz + sine * bbr
    aat = rho * (sine * aaz + cosine * aar)
    bbt = rho * (sine * bbz + cosine * bbr)
    h = aa * cosine**2 + bb * sine**2
    j = rho * sine * cosine * (bb - aa)
    k = rho**2 * (aa * sine**2 + bb * cosine**2)
    hq = aaq * cosine**2 + bbq * sine**2
    jq = sine * cosine * (bb - aa) + rho * sine * cosine * (bbq - aaq)
    kq = 2 * rho * (aa * sine**2 + bb * cosine**2) + rho**2 * (
        aaq * sine**2 + bbq * cosine**2
    )
    ht = aat * cosine**2 + bbt * sine**2 + 2 * sine * cosine * (bb - aa)
    jt = rho * (
        (cosine**2 - sine**2) * (bb - aa)
        + sine * cosine * (bbt - aat)
    )
    kt = rho**2 * (
        aat * sine**2 + bbt * cosine**2
        + 2 * sine * cosine * (aa - bb)
    )
    z_tangent = -slope * cosine + rho * sine
    r_tangent = slope * sine + rho * cosine
    energy = aa * z_tangent**2 + bb * r_tangent**2
    speed = affine_jet_sqrt(energy)
    moment = h * slope + j
    energy_q = hq * slope**2 + 2 * jq * slope + kq
    energy_t = ht * slope**2 + 2 * jt * slope + kt
    moment_q = hq * slope + jq
    moment_t = ht * slope + jt
    Cq = -cosine * Cz + sine * Cr
    Ct = rho * (sine * Cz + cosine * Cr)
    weight_q = 2 * Cq / C + 2 / rho
    quotient = momentum / sine**3
    reduced_moment = h * quotient + rho * cosine * (bb - aa)
    weight_t_moment = 2 * Ct * moment / C + 2 * cosine * reduced_moment
    bracket = (
        weight_q * speed + energy_q / (2 * speed)
        - weight_t_moment / speed - moment_t / speed
        + moment * energy_t / (2 * speed**3)
        - slope * (
            weight_q * moment / speed + moment_q / speed
            - moment * energy_q / (2 * speed**3)
        )
    )
    second = speed**3 * bracket / (aa * bb * rho**2)
    source = second + 2 * cosine * slope / sine
    return slope, sine**2 * source


def _coordinate_cuts(center, linear, knots, lower, upper):
    if linear == 0.0:
        return []
    lo_value = center + linear * lower
    hi_value = center + linear * upper
    value_lower = min(lo_value, hi_value)
    value_upper = max(lo_value, hi_value)
    return [
        (float(knot) - center) / linear
        for knot in knots
        if value_lower < float(knot) < value_upper
    ]


def _pad_ulps(interval, count=64):
    lower = float(interval.lower)
    upper = float(interval.upper)
    for _ in range(int(count)):
        lower = float(np.nextafter(lower, -np.inf))
        upper = float(np.nextafter(upper, np.inf))
    return VInterval(lower, upper)


def correlated_divergence_jacobian_hull(
    theta, rho_center, rho_parameter, w_center, w_parameter, metric,
    xi_lower=-1.0, xi_upper=1.0,
):
    """Enclose D_(rho,w) RHS while retaining one shared launch parameter."""
    theta = float(theta)
    rho_center = float(rho_center)
    rho_parameter = float(rho_parameter)
    w_center = float(w_center)
    w_parameter = float(w_parameter)
    cosine = math.cos(theta)
    sine = math.sin(theta)
    z_center = metric.z_brane - cosine * rho_center
    z_parameter = -cosine * rho_parameter
    r_center = sine * rho_center
    r_parameter = sine * rho_parameter
    cuts = [float(xi_lower), float(xi_upper)]
    for spline in metric.fields.values():
        cuts.extend(_coordinate_cuts(
            z_center, z_parameter, spline.knots_z,
            xi_lower, xi_upper,
        ))
        cuts.extend(_coordinate_cuts(
            r_center, r_parameter, spline.knots_r,
            xi_lower, xi_upper,
        ))
    cuts = sorted(set(value for value in cuts if xi_lower <= value <= xi_upper))
    enclosures = [[[] for _ in range(2)] for _ in range(2)]
    for left, right in zip(cuts[:-1], cuts[1:]):
        midpoint = 0.5 * left + 0.5 * right
        halfwidth = 0.5 * (right - left)
        rho_form = AffineForm.parameter(
            rho_center + rho_parameter * midpoint,
            rho_parameter * halfwidth,
        )
        w_form = AffineForm.parameter(
            w_center + w_parameter * midpoint,
            w_parameter * halfwidth,
        )
        rho = AffineJet.variable(rho_form, 0)
        momentum = AffineJet.variable(w_form, 1)
        rhs = _divergence_rhs(theta, rho, momentum, metric)
        for row in range(2):
            for column in range(2):
                enclosures[row][column].append(
                    rhs[row].derivative[column].range
                )
    result = np.asarray([
        [_pad_ulps(interval_hull(enclosures[row][column])) for column in range(2)]
        for row in range(2)
    ], dtype=object)
    result[0, 0] = VInterval.point(0.0)
    result[0, 1] = (
        VInterval.point(1.0) / VInterval.point(math.sin(theta))**2
    )
    return result, {
        "parameter_segment_count": len(cuts) - 1,
        "parameter_cuts": cuts,
    }
