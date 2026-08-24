"""Validated interval shooting for the sealed donor-capped graph class.

The discrete metric is the exact tensor-product bicubic spline represented by
archived binary64 knots and coefficients.  Interval propagation never calls
SciPy's floating spline evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from mpmath import libmp


NEGATIVE_INFINITY = -float("inf")
POSITIVE_INFINITY = float("inf")
MP_INTERVAL_PRECISION = 96


def _down(value):
    return float(np.nextafter(float(value), NEGATIVE_INFINITY))


def _up(value):
    return float(np.nextafter(float(value), POSITIVE_INFINITY))


@dataclass(frozen=True)
class VInterval:
    """Closed outward-rounded binary64 interval."""

    lower: float
    upper: float

    def __post_init__(self):
        if math.isnan(self.lower) or math.isnan(self.upper):
            raise ValueError("NaN interval endpoint")
        if self.lower > self.upper:
            raise ValueError("reversed interval")

    @classmethod
    def point(cls, value):
        value = float(value)
        return cls(value, value)

    @property
    def width(self):
        return _up(self.upper - self.lower)

    @property
    def midpoint(self):
        return 0.5 * self.lower + 0.5 * self.upper

    @property
    def magnitude(self):
        return max(abs(self.lower), abs(self.upper))

    def contains(self, value):
        if isinstance(value, VInterval):
            return self.lower <= value.lower and value.upper <= self.upper
        return self.lower <= float(value) <= self.upper

    def strict_contains(self, other):
        other = as_interval(other)
        return self.lower < other.lower and other.upper < self.upper

    def hull(self, other):
        other = as_interval(other)
        return VInterval(min(self.lower, other.lower), max(self.upper, other.upper))

    def inflate(self, relative=0.05, absolute=1e-14):
        radius = max(0.5 * self.width * float(relative), float(absolute))
        return VInterval(_down(self.lower - radius), _up(self.upper + radius))

    def __neg__(self):
        return VInterval(_down(-self.upper), _up(-self.lower))

    def __add__(self, other):
        other = as_interval(other)
        return VInterval(
            _down(self.lower + other.lower), _up(self.upper + other.upper),
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-as_interval(other))

    def __rsub__(self, other):
        return as_interval(other) - self

    def __mul__(self, other):
        other = as_interval(other)
        products = (
            self.lower * other.lower, self.lower * other.upper,
            self.upper * other.lower, self.upper * other.upper,
        )
        return VInterval(_down(min(products)), _up(max(products)))

    __rmul__ = __mul__

    def reciprocal(self):
        if self.lower <= 0.0 <= self.upper:
            raise ZeroDivisionError("interval denominator contains zero")
        values = (1.0 / self.lower, 1.0 / self.upper)
        return VInterval(_down(min(values)), _up(max(values)))

    def __truediv__(self, other):
        return self * as_interval(other).reciprocal()

    def __rtruediv__(self, other):
        return as_interval(other) / self

    def __pow__(self, exponent):
        exponent = int(exponent)
        if exponent < 0:
            return (self ** (-exponent)).reciprocal()
        if exponent == 0:
            return VInterval.point(1.0)
        if exponent == 1:
            return self
        if exponent % 2 == 0 and self.lower <= 0.0 <= self.upper:
            return VInterval(0.0, _up(self.magnitude ** exponent))
        values = (self.lower ** exponent, self.upper ** exponent)
        return VInterval(_down(min(values)), _up(max(values)))


def as_interval(value):
    return value if isinstance(value, VInterval) else VInterval.point(value)


def interval_hull(values):
    values = list(values)
    if not values:
        raise ValueError("empty interval hull")
    return VInterval(
        min(value.lower for value in values),
        max(value.upper for value in values),
    )


def _libmp_unary(function, value):
    value = as_interval(value)
    raw = (libmp.from_float(value.lower), libmp.from_float(value.upper))
    result = function(raw, MP_INTERVAL_PRECISION)
    lower = libmp.to_float(result[0], strict=True, rnd=libmp.round_floor)
    upper = libmp.to_float(result[1], strict=True, rnd=libmp.round_ceiling)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise FloatingPointError("nonfinite libmp interval result")
    return VInterval(_down(lower), _up(upper))


def interval_sin(value):
    value = as_interval(value)
    result = _libmp_unary(libmp.mpi_sin, value)
    if value.lower >= 0.0 and value.upper <= math.pi / 2:
        return VInterval(max(0.0, result.lower), min(1.0, result.upper))
    return result


def interval_cos(value):
    value = as_interval(value)
    result = _libmp_unary(libmp.mpi_cos, value)
    if value.lower >= 0.0 and value.upper <= math.pi / 2:
        return VInterval(max(0.0, result.lower), min(1.0, result.upper))
    return result


def interval_sqrt(value):
    value = as_interval(value)
    if value.lower <= 0.0:
        raise ValueError("sqrt interval is not strictly positive")
    return _libmp_unary(libmp.mpi_sqrt, value)


def exact_axis_weighted_ratio(theta_axis):
    """Directed enclosure of integral(sin(t)^2,0,x)/sin(x)^3."""
    theta_axis = float(theta_axis)
    if not 0.0 < theta_axis <= 1e-3:
        raise ValueError("axis weighted ratio requires 0 < theta_axis <= 1e-3")
    raw = libmp.from_float(theta_axis)
    point = (raw, raw)
    two = (libmp.from_int(2), libmp.from_int(2))
    four = (libmp.from_int(4), libmp.from_int(4))
    half_theta = libmp.mpi_div(point, two, MP_INTERVAL_PRECISION)
    twice_theta = libmp.mpi_mul(point, two, MP_INTERVAL_PRECISION)
    sine_twice = libmp.mpi_sin(twice_theta, MP_INTERVAL_PRECISION)
    numerator = libmp.mpi_sub(
        half_theta,
        libmp.mpi_div(sine_twice, four, MP_INTERVAL_PRECISION),
        MP_INTERVAL_PRECISION,
    )
    sine = libmp.mpi_sin(point, MP_INTERVAL_PRECISION)
    denominator = libmp.mpi_mul(
        libmp.mpi_mul(sine, sine, MP_INTERVAL_PRECISION),
        sine, MP_INTERVAL_PRECISION,
    )
    ratio = libmp.mpi_div(numerator, denominator, MP_INTERVAL_PRECISION)
    lower = libmp.to_float(
        ratio[0], strict=True, rnd=libmp.round_floor,
    )
    upper = libmp.to_float(
        ratio[1], strict=True, rnd=libmp.round_ceiling,
    )
    return VInterval(_down(lower), _up(upper))


def _basis_levels(knots, degree, x, span):
    """B-spline basis values and first two derivatives on one knot span."""
    x = as_interval(x)
    zero = VInterval.point(0.0)
    one = VInterval.point(1.0)
    values = [{int(span): one}]
    first = [{int(span): zero}]
    second = [{int(span): zero}]
    for local_degree in range(1, int(degree) + 1):
        current = {}
        current_first = {}
        current_second = {}
        previous = values[-1]
        previous_first = first[-1]
        previous_second = second[-1]
        for index in range(int(span) - local_degree, int(span) + 1):
            value = zero
            derivative = zero
            derivative_second = zero
            left_denominator = knots[index + local_degree] - knots[index]
            if left_denominator != 0.0:
                scale = VInterval.point(1.0 / left_denominator)
                value = value + (x - knots[index]) * scale * previous.get(index, zero)
                derivative = derivative + local_degree * scale * previous.get(index, zero)
                derivative_second = (
                    derivative_second
                    + local_degree * scale * previous_first.get(index, zero)
                )
            right_denominator = (
                knots[index + local_degree + 1] - knots[index + 1]
            )
            if right_denominator != 0.0:
                scale = VInterval.point(1.0 / right_denominator)
                value = (
                    value + (knots[index + local_degree + 1] - x)
                    * scale * previous.get(index + 1, zero)
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


def _knot_aligned_segments(knots, degree, box):
    box = as_interval(box)
    lower_domain = float(knots[degree])
    upper_domain = float(knots[-degree - 1])
    tolerance = 16.0 * np.finfo(float).eps * max(
        abs(lower_domain), abs(upper_domain), 1.0,
    )
    if box.lower < lower_domain - tolerance or box.upper > upper_domain + tolerance:
        raise ValueError("spline coordinate leaves archived domain")
    lower = max(box.lower, lower_domain)
    upper = min(box.upper, upper_domain)
    cuts = [lower]
    cuts.extend(float(value) for value in knots if lower < value < upper)
    cuts.append(upper)
    segments = []
    for left, right in zip(cuts[:-1], cuts[1:]):
        probe = 0.5 * left + 0.5 * right
        span = int(np.searchsorted(knots, probe, side="right") - 1)
        span = min(max(span, degree), len(knots) - degree - 2)
        segments.append((VInterval(left, right), span))
    if not segments:
        span = int(np.searchsorted(knots, lower, side="right") - 1)
        span = min(max(span, degree), len(knots) - degree - 2)
        segments.append((VInterval.point(lower), span))
    return segments


class TensorBicubicIntervalSpline:
    """Exact-binary tensor B-spline with interval evaluation."""

    degree_z = 3
    degree_r = 3

    def __init__(self, knots_z, knots_r, coefficients):
        self.knots_z = np.asarray(knots_z, dtype=float)
        self.knots_r = np.asarray(knots_r, dtype=float)
        count_z = len(self.knots_z) - self.degree_z - 1
        count_r = len(self.knots_r) - self.degree_r - 1
        self.coefficients = np.asarray(coefficients, dtype=float).reshape(
            count_z, count_r,
        )
        if not np.all(np.isfinite(self.coefficients)):
            raise ValueError("nonfinite spline coefficient")

    @classmethod
    def from_scipy(cls, spline):
        knots_z, knots_r, coefficients = spline.tck
        if tuple(spline.degrees) != (3, 3):
            raise ValueError("Test 4B requires a bicubic spline")
        return cls(knots_z, knots_r, coefficients)

    def archive_payload(self, prefix):
        return {
            f"{prefix}_knots_z": self.knots_z,
            f"{prefix}_knots_r": self.knots_r,
            f"{prefix}_coefficients": self.coefficients,
        }

    def evaluate(self, z, r, maximum_derivative=2):
        maximum_derivative = int(maximum_derivative)
        if maximum_derivative not in (0, 1, 2):
            raise ValueError("supported spline derivative order is 0, 1, or 2")
        pieces = {(dz, dr): [] for dz in range(maximum_derivative + 1)
                  for dr in range(maximum_derivative + 1 - dz)}
        for zbox, zspan in _knot_aligned_segments(
            self.knots_z, self.degree_z, z,
        ):
            zbasis = _basis_levels(
                self.knots_z, self.degree_z, zbox, zspan,
            )
            for rbox, rspan in _knot_aligned_segments(
                self.knots_r, self.degree_r, r,
            ):
                rbasis = _basis_levels(
                    self.knots_r, self.degree_r, rbox, rspan,
                )
                for derivative, values in pieces.items():
                    dz, dr = derivative
                    total = VInterval.point(0.0)
                    for i, bz in zbasis[dz].items():
                        if i < 0 or i >= self.coefficients.shape[0]:
                            continue
                        for j, br in rbasis[dr].items():
                            if j < 0 or j >= self.coefficients.shape[1]:
                                continue
                            total = total + (
                                VInterval.point(self.coefficients[i, j]) * bz * br
                            )
                    values.append(total)
        return {derivative: interval_hull(values)
                for derivative, values in pieces.items()}


@dataclass(frozen=True)
class IntervalJet:
    value: VInterval
    derivative: tuple[VInterval, VInterval, VInterval]

    @classmethod
    def constant(cls, value):
        zero = VInterval.point(0.0)
        return cls(as_interval(value), (zero, zero, zero))

    @classmethod
    def variable(cls, value, index):
        zero = VInterval.point(0.0)
        one = VInterval.point(1.0)
        derivatives = [zero, zero, zero]
        derivatives[int(index)] = one
        return cls(as_interval(value), tuple(derivatives))

    def __neg__(self):
        return IntervalJet(-self.value, tuple(-item for item in self.derivative))

    def __add__(self, other):
        other = as_jet(other)
        return IntervalJet(
            self.value + other.value,
            tuple(a + b for a, b in zip(self.derivative, other.derivative)),
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-as_jet(other))

    def __rsub__(self, other):
        return as_jet(other) - self

    def __mul__(self, other):
        other = as_jet(other)
        return IntervalJet(
            self.value * other.value,
            tuple(
                a * other.value + self.value * b
                for a, b in zip(self.derivative, other.derivative)
            ),
        )

    __rmul__ = __mul__

    def reciprocal(self):
        reciprocal = self.value.reciprocal()
        return IntervalJet(
            reciprocal,
            tuple(-item * reciprocal**2 for item in self.derivative),
        )

    def __truediv__(self, other):
        return self * as_jet(other).reciprocal()

    def __rtruediv__(self, other):
        return as_jet(other) / self

    def __pow__(self, exponent):
        exponent = int(exponent)
        if exponent == 0:
            return IntervalJet.constant(1.0)
        if exponent < 0:
            return (self ** (-exponent)).reciprocal()
        value = self.value ** exponent
        factor = exponent * self.value ** (exponent - 1)
        return IntervalJet(value, tuple(factor * item for item in self.derivative))


def as_jet(value):
    return value if isinstance(value, IntervalJet) else IntervalJet.constant(value)


def jet_sin(value):
    value = as_jet(value)
    sine = interval_sin(value.value)
    cosine = interval_cos(value.value)
    return IntervalJet(sine, tuple(cosine * item for item in value.derivative))


def jet_cos(value):
    value = as_jet(value)
    cosine = interval_cos(value.value)
    sine = interval_sin(value.value)
    return IntervalJet(cosine, tuple(-sine * item for item in value.derivative))


def jet_sqrt(value):
    value = as_jet(value)
    root = interval_sqrt(value.value)
    return IntervalJet(
        root, tuple(item / (2 * root) for item in value.derivative),
    )


class ValidatedBicubicMetric:
    def __init__(self, z_brane, fields):
        self.z_brane = float(z_brane)
        self.fields = dict(fields)
        if set(self.fields) != {"A", "B", "C"}:
            raise ValueError("metric requires A, B, and C splines")

    def interval_fields(self, z, r, maximum_derivative=1):
        return {
            name: spline.evaluate(z, r, maximum_derivative)
            for name, spline in self.fields.items()
        }

    def jet_fields(self, z, r):
        output = {}
        for name, spline in self.fields.items():
            values = spline.evaluate(z.value, r.value, 2)
            value = values[(0, 0)]
            dz = values[(1, 0)]
            dr = values[(0, 1)]
            dzz = values[(2, 0)]
            dzr = values[(1, 1)]
            drr = values[(0, 2)]
            output[name] = IntervalJet(
                value,
                tuple(dz * zd + dr * rd
                      for zd, rd in zip(z.derivative, r.derivative)),
            )
            output[name + "z"] = IntervalJet(
                dz,
                tuple(dzz * zd + dzr * rd
                      for zd, rd in zip(z.derivative, r.derivative)),
            )
            output[name + "r"] = IntervalJet(
                dr,
                tuple(dzr * zd + drr * rd
                      for zd, rd in zip(z.derivative, r.derivative)),
            )
        return output


def _regularized_formula(theta, rho, slope, fields, axis_u=None):
    """Algebraically regular Euler--Lagrange solve for ``rho''``."""
    sine = jet_sin(theta) if isinstance(theta, IntervalJet) else interval_sin(theta)
    cosine = jet_cos(theta) if isinstance(theta, IntervalJet) else interval_cos(theta)
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
    kq = (
        2 * rho * (aa * sine**2 + bb * cosine**2)
        + rho**2 * (aaq * sine**2 + bbq * cosine**2)
    )
    ht = aat * cosine**2 + bbt * sine**2 + 2 * sine * cosine * (bb - aa)
    jt = rho * (
        (cosine**2 - sine**2) * (bb - aa)
        + sine * cosine * (bbt - aat)
    )
    kt = rho**2 * (
        aat * sine**2 + bbt * cosine**2 + 2 * sine * cosine * (aa - bb)
    )
    # The equivalent geometric sum of squares is materially tighter than
    # ``h*p**2 + 2*j*p + k`` under interval dependency.
    z_tangent = -slope * cosine + rho * sine
    r_tangent = slope * sine + rho * cosine
    energy = aa * z_tangent**2 + bb * r_tangent**2
    speed = jet_sqrt(energy) if isinstance(energy, IntervalJet) else interval_sqrt(energy)
    moment = h * slope + j
    energy_q = hq * slope**2 + 2 * jq * slope + kq
    energy_t = ht * slope**2 + 2 * jt * slope + kt
    moment_q = hq * slope + jq
    moment_t = ht * slope + jt
    Cq = -cosine * Cz + sine * Cr
    Ct = rho * (sine * Cz + cosine * Cr)
    weight_q = 2 * Cq / C + 2 / rho
    if axis_u is None:
        quotient = slope / sine
    else:
        quotient = axis_u
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
    denominator = aa * bb * rho**2
    return speed**3 * bracket / denominator


def regularized_rhs_interval(theta, rho, slope, metric, axis_u=None):
    theta = as_interval(theta)
    rho = as_interval(rho)
    slope = as_interval(slope)
    sine = interval_sin(theta)
    cosine = interval_cos(theta)
    z = VInterval.point(metric.z_brane) - rho * cosine
    r = rho * sine
    values = metric.interval_fields(z, r, 1)
    fields = {}
    for name in ("A", "B", "C"):
        fields[name] = values[name][(0, 0)]
        fields[name + "z"] = values[name][(1, 0)]
        fields[name + "r"] = values[name][(0, 1)]
    return _regularized_formula(theta, rho, slope, fields, axis_u=axis_u)


def regularized_rhs_jet(theta, rho, slope, metric):
    theta = IntervalJet.variable(theta, 0)
    rho = IntervalJet.variable(rho, 1)
    slope = IntervalJet.variable(slope, 2)
    sine = jet_sin(theta)
    cosine = jet_cos(theta)
    z = metric.z_brane - rho * cosine
    r = rho * sine
    fields = metric.jet_fields(z, r)
    return _regularized_formula(theta, rho, slope, fields)


def axis_second_interval(axis_radius, metric):
    axis_radius = as_interval(axis_radius)
    z = VInterval.point(metric.z_brane) - axis_radius
    r = VInterval.point(0.0)
    values = metric.interval_fields(z, r, 1)
    A = values["A"][(0, 0)]
    B = values["B"][(0, 0)]
    C = values["C"][(0, 0)]
    Bz = values["B"][(1, 0)]
    Cz = values["C"][(1, 0)]
    barrier = 3 - 2 * B**2 * Cz * axis_radius / (A**2 * C)
    barrier = barrier - B * Bz * axis_radius / A**2
    # ``rho * barrier`` is the zero-slope point value.  The regular-axis
    # limit also contains ``-2*u`` with u=rho''(0), so 3*u=rho*barrier.
    return axis_radius * barrier / 3


def _axis_divergence_source(theta, rho, axis_u, fields):
    """Regular ``G=f+2*cot(theta)*p`` after substituting ``p=u*sin``.

    The expanded form removes the exact axis cancellation before interval
    evaluation.  Coefficients of u, u**2, and u**3 vanish at least linearly
    in sin(theta), so the launch fixed-point map is contractive.
    """
    sine = jet_sin(theta) if isinstance(theta, IntervalJet) else interval_sin(theta)
    cosine = jet_cos(theta) if isinstance(theta, IntervalJet) else interval_cos(theta)
    A, B, C = fields["A"], fields["B"], fields["C"]
    Az, Ar = fields["Az"], fields["Ar"]
    Bz, Br = fields["Bz"], fields["Br"]
    Cz, Cr = fields["Cz"], fields["Cr"]
    R = rho
    s = sine
    c = cosine
    coefficient_0 = R**3 * (
        2*A**4*C*s**2 + 2*A**4*Cr*R*s**3 + A**3*Ar*C*R*s**3
        - 2*A**2*B**2*C*s**2 + 3*A**2*B**2*C
        - 2*A**2*B**2*Cr*R*s**3 + 2*A**2*B**2*Cr*R*s
        - 2*A**2*B**2*Cz*R*c*s**2
        + A**2*B*Br*C*R*s**3 - A**2*B*Br*C*R*s
        - 2*A**2*B*Bz*C*R*c*s**2
        - 2*A*Ar*B**2*C*R*s**3 + 2*A*Ar*B**2*C*R*s
        + A*Az*B**2*C*R*c*s**2
        + 2*B**4*Cz*R*c*s**2 - 2*B**4*Cz*R*c
        + B**3*Bz*C*R*c*s**2 - B**3*Bz*C*R*c
    )
    coefficient_1 = -R**2*s * (
        6*A**4*C*c*s + 6*A**4*Cr*R*c*s**2
        + 3*A**3*Ar*C*R*c*s**2 - 6*A**2*B**2*C*c*s
        - 6*A**2*B**2*Cr*R*c*s**2 + 2*A**2*B**2*Cr*R*c
        + 6*A**2*B**2*Cz*R*s**3 - 4*A**2*B**2*Cz*R*s
        + 3*A**2*B*Br*C*R*c*s**2 - A**2*B*Br*C*R*c
        + 6*A**2*B*Bz*C*R*s**3 - 4*A**2*B*Bz*C*R*s
        - 6*A*Ar*B**2*C*R*c*s**2 + 2*A*Ar*B**2*C*R*c
        - 3*A*Az*B**2*C*R*s**3 + 2*A*Az*B**2*C*R*s
        - 6*B**4*Cz*R*s**3 + 6*B**4*Cz*R*s
        - 3*B**3*Bz*C*R*s**3 + 3*B**3*Bz*C*R*s
    )
    coefficient_2 = -R*s**2 * (
        6*A**4*C*s**2 - 6*A**4*C
        + 6*A**4*Cr*R*s**3 - 6*A**4*Cr*R*s
        + 3*A**3*Ar*C*R*s**3 - 3*A**3*Ar*C*R*s
        - 6*A**2*B**2*C*s**2 + 2*A**2*B**2*C
        - 6*A**2*B**2*Cr*R*s**3 + 4*A**2*B**2*Cr*R*s
        - 6*A**2*B**2*Cz*R*c*s**2 + 2*A**2*B**2*Cz*R*c
        + 3*A**2*B*Br*C*R*s**3 - 2*A**2*B*Br*C*R*s
        - 6*A**2*B*Bz*C*R*c*s**2 + 2*A**2*B*Bz*C*R*c
        - 6*A*Ar*B**2*C*R*s**3 + 4*A*Ar*B**2*C*R*s
        + 3*A*Az*B**2*C*R*c*s**2 - A*Az*B**2*C*R*c
        + 6*B**4*Cz*R*c*s**2 + 3*B**3*Bz*C*R*c*s**2
    )
    coefficient_3 = s**2 * (
        2*A**4*C*c*s**2 - 2*A**4*C*c
        + 2*A**4*Cr*R*c*s**3 - 2*A**4*Cr*R*c*s
        + A**3*Ar*C*R*c*s**3 - A**3*Ar*C*R*c*s
        - 2*A**2*B**2*C*c*s**2
        - 2*A**2*B**2*Cr*R*c*s**3
        + 2*A**2*B**2*Cz*R*s**4 - 2*A**2*B**2*Cz*R*s**2
        + A**2*B*Br*C*R*c*s**3
        + 2*A**2*B*Bz*C*R*s**4 - 2*A**2*B*Bz*C*R*s**2
        - 2*A*Ar*B**2*C*R*c*s**3
        - A*Az*B**2*C*R*s**4 + A*Az*B**2*C*R*s**2
        - 2*B**4*Cz*R*s**4 - B**3*Bz*C*R*s**4
    )
    numerator = (
        coefficient_0 + coefficient_1*axis_u
        + coefficient_2*axis_u**2 + coefficient_3*axis_u**3
    )
    return numerator / (A**2 * B**2 * C * R**2)


def axis_divergence_source_interval(theta, rho, axis_u, metric):
    theta = as_interval(theta)
    rho = as_interval(rho)
    axis_u = as_interval(axis_u)
    sine = interval_sin(theta)
    cosine = interval_cos(theta)
    z = VInterval.point(metric.z_brane) - rho * cosine
    r = rho * sine
    values = metric.interval_fields(z, r, 1)
    fields = {}
    for name in ("A", "B", "C"):
        fields[name] = values[name][(0, 0)]
        fields[name + "z"] = values[name][(1, 0)]
        fields[name + "r"] = values[name][(0, 1)]
    return _axis_divergence_source(theta, rho, axis_u, fields)


def axis_divergence_source_jet(theta, rho, axis_u, metric):
    """Divergence source and its theta/rho/u derivatives on an interval box."""
    theta = IntervalJet.variable(theta, 0)
    rho = IntervalJet.variable(rho, 1)
    axis_u = IntervalJet.variable(axis_u, 2)
    sine = jet_sin(theta)
    cosine = jet_cos(theta)
    z = metric.z_brane - rho * cosine
    r = rho * sine
    fields = metric.jet_fields(z, r)
    return _axis_divergence_source(theta, rho, axis_u, fields)


def axis_divergence_source_mean_value(theta, rho, axis_u, metric):
    """Intersect direct and centered mean-value source enclosures."""
    theta = as_interval(theta)
    rho = as_interval(rho)
    axis_u = as_interval(axis_u)
    center = (theta.midpoint, rho.midpoint, axis_u.midpoint)
    point = axis_divergence_source_interval(
        VInterval.point(center[0]), VInterval.point(center[1]),
        VInterval.point(center[2]), metric,
    )
    jet = axis_divergence_source_jet(theta, rho, axis_u, metric)
    mean_value = point
    for variable, midpoint, derivative in zip(
        (theta, rho, axis_u), center, jet.derivative,
    ):
        mean_value = mean_value + derivative * (variable - midpoint)
    direct = axis_divergence_source_interval(theta, rho, axis_u, metric)
    lower = max(direct.lower, mean_value.lower)
    upper = min(direct.upper, mean_value.upper)
    if lower > upper:
        raise RuntimeError("independent axis-source enclosures are disjoint")
    return VInterval(lower, upper)


def axis_divergence_source_correlated_mean_value(
    theta, axis_radius, axis_u, metric,
):
    """Centered enclosure retaining rho=a+(1-cos(theta))*u correlation."""
    theta = as_interval(theta)
    axis_radius = as_interval(axis_radius)
    axis_u = as_interval(axis_u)
    theta_jet = IntervalJet.variable(theta, 0)
    radius_jet = IntervalJet.variable(axis_radius, 1)
    u_jet = IntervalJet.variable(axis_u, 2)
    sine_jet = jet_sin(theta_jet)
    cosine_jet = jet_cos(theta_jet)
    rho_jet = radius_jet + (1 - cosine_jet) * u_jet
    z_jet = metric.z_brane - rho_jet * cosine_jet
    r_jet = rho_jet * sine_jet
    fields = metric.jet_fields(z_jet, r_jet)
    jet = _axis_divergence_source(theta_jet, rho_jet, u_jet, fields)

    center = (theta.midpoint, axis_radius.midpoint, axis_u.midpoint)
    theta_midpoint, radius_midpoint, u_midpoint = center
    rho_midpoint = (
        radius_midpoint
        + (1.0 - math.cos(theta_midpoint)) * u_midpoint
    )
    point = axis_divergence_source_interval(
        VInterval.point(theta_midpoint), VInterval.point(rho_midpoint),
        VInterval.point(u_midpoint), metric,
    )
    mean_value = point
    for variable, midpoint, derivative in zip(
        (theta, axis_radius, axis_u), center, jet.derivative,
    ):
        mean_value = mean_value + derivative * (variable - midpoint)
    rho_box = (
        axis_radius
        + (VInterval.point(1.0) - interval_cos(theta)) * axis_u
    )
    direct = axis_divergence_source_interval(theta, rho_box, axis_u, metric)
    lower = max(direct.lower, mean_value.lower)
    upper = min(direct.upper, mean_value.upper)
    if lower > upper:
        raise RuntimeError("correlated and direct axis-source enclosures are disjoint")
    return VInterval(lower, upper)


def regularized_divergence_rhs_interval(theta, rho, momentum, metric):
    """Exact first-order system in w=sin(theta)^2*rho'."""
    theta = as_interval(theta)
    rho = as_interval(rho)
    momentum = as_interval(momentum)
    sine = interval_sin(theta)
    slope = momentum / sine**2
    axis_u = momentum / sine**3
    z = VInterval.point(metric.z_brane) - rho * interval_cos(theta)
    r = rho * sine
    values = metric.interval_fields(z, r, 1)
    fields = {}
    for name in ("A", "B", "C"):
        fields[name] = values[name][(0, 0)]
        fields[name + "z"] = values[name][(1, 0)]
        fields[name + "r"] = values[name][(0, 1)]
    source = _axis_divergence_source(theta, rho, axis_u, fields)
    return slope, sine**2 * source


def regularized_divergence_rhs_jet(theta, rho, momentum, metric):
    """Divergence-coordinate RHS and theta/rho/w interval derivatives."""
    theta = IntervalJet.variable(theta, 0)
    rho = IntervalJet.variable(rho, 1)
    momentum = IntervalJet.variable(momentum, 2)
    sine = jet_sin(theta)
    cosine = jet_cos(theta)
    slope = momentum / sine**2
    axis_u = momentum / sine**3
    z = metric.z_brane - rho * cosine
    r = rho * sine
    fields = metric.jet_fields(z, r)
    source = _axis_divergence_source(theta, rho, axis_u, fields)
    return slope, sine**2 * source


def regular_axis_cone(axis_radius, metric, theta_axis=1e-3,
                      maximum_iterations=24, theta_subdivisions=128,
                      launch_subdivisions=8):
    axis_radius = as_interval(axis_radius)
    theta_axis = float(theta_axis)
    if theta_axis <= 0.0:
        raise ValueError("theta_axis must be positive")
    cone = axis_second_interval(axis_radius, metric).inflate(0.10, 1e-10)
    theta = VInterval(0.0, theta_axis)
    # J(theta)=int_0^theta sin(t)^2 dt / sin(theta)^3.  On the sealed
    # theta_axis<=1e-3 range, [1/3, 0.334] is a conservative enclosure.
    if theta_axis > 1e-3:
        raise ValueError("regular-axis J enclosure requires theta_axis <= 1e-3")
    coarse_weighted_ratio = VInterval(1.0 / 3.0, 0.334)
    exact_weighted_ratio_enclosure = exact_axis_weighted_ratio(theta_axis)
    weighted_ratio = VInterval(
        max(coarse_weighted_ratio.lower, exact_weighted_ratio_enclosure.lower),
        min(coarse_weighted_ratio.upper, exact_weighted_ratio_enclosure.upper),
    )
    radial_count = (
        1 if axis_radius.lower == axis_radius.upper else int(launch_subdivisions)
    )
    radial_launch_cells = []
    for radial_index in range(radial_count):
        radial_launch_cells.append(VInterval(
            axis_radius.lower + radial_index * axis_radius.width / radial_count,
            axis_radius.lower + (radial_index + 1) * axis_radius.width / radial_count,
        ))
    for iteration in range(int(maximum_iterations)):
        source_pieces = []
        radial_source_pieces = [[] for _ in range(radial_count)]
        for subindex in range(int(theta_subdivisions)):
            left = theta_axis * subindex / int(theta_subdivisions)
            right = theta_axis * (subindex + 1) / int(theta_subdivisions)
            theta_piece = VInterval(left, right)
            for radial_index, radial_cell in enumerate(radial_launch_cells):
                piece = axis_divergence_source_correlated_mean_value(
                    theta_piece, radial_cell, cone, metric,
                )
                source_pieces.append(piece)
                radial_source_pieces[radial_index].append(piece)
        source = interval_hull(source_pieces)
        image = weighted_ratio * source
        radial_images = [
            weighted_ratio * interval_hull(pieces)
            for pieces in radial_source_pieces
        ]
        if cone.strict_contains(image):
            # Every fixed point in the invariant cone lies in T(cone)=image.
            # Retaining the larger existence cone here would discard this
            # rigorous one-step contraction before propagation even begins.
            solution_cone = image
            one_minus_cosine = (
                VInterval.point(1.0)
                - interval_cos(VInterval.point(theta_axis))
            )
            endpoint = {
                "rho": axis_radius + one_minus_cosine * solution_cone,
                "slope": (
                    interval_sin(VInterval.point(theta_axis)) * solution_cone
                ),
            }
            return endpoint, {
                "iterations": iteration + 1,
                "axis_second": axis_second_interval(axis_radius, metric),
                "cone": solution_cone,
                "invariant_cone": cone,
                "image": image,
                "source": source,
                "theta_axis": theta_axis,
                "theta_subdivisions": int(theta_subdivisions),
                "launch_subdivisions": int(launch_subdivisions),
                "weighted_ratio": weighted_ratio,
                "coarse_weighted_ratio": coarse_weighted_ratio,
                "radial_launch_cells": radial_launch_cells,
                "radial_images": radial_images,
            }
        cone = cone.hull(image).inflate(0.15, 1e-10)
        if not math.isfinite(cone.lower) or not math.isfinite(cone.upper):
            break
    raise RuntimeError("regular-axis invariant cone did not close")


def validated_taylor_step(theta, state, step, metric,
                          maximum_picard_iterations=16):
    """One validated second/third-order endpoint step."""
    theta = float(theta)
    step = float(step)
    rho0 = as_interval(state["rho"])
    slope0 = as_interval(state["slope"])
    time_box = VInterval(theta, theta + step)
    f0 = regularized_rhs_interval(
        VInterval.point(theta), rho0, slope0, metric,
    )
    rho_guess = rho0.hull(rho0 + step * slope0 + 0.5 * step**2 * f0)
    slope_guess = slope0.hull(slope0 + step * f0)
    rho_tube = rho_guess.inflate(0.10, 1e-13)
    slope_tube = slope_guess.inflate(0.10, 1e-13)
    elapsed = VInterval(0.0, step)
    picard_closed = False
    for iteration in range(int(maximum_picard_iterations)):
        f_tube = regularized_rhs_interval(
            time_box, rho_tube, slope_tube, metric,
        )
        rho_image = rho0 + elapsed * slope_tube
        slope_image = slope0 + elapsed * f_tube
        if (rho_tube.strict_contains(rho_image)
                and slope_tube.strict_contains(slope_image)):
            picard_closed = True
            break
        rho_tube = rho_tube.hull(rho_image).inflate(0.10, 1e-13)
        slope_tube = slope_tube.hull(slope_image).inflate(0.10, 1e-13)
    if not picard_closed:
        raise RuntimeError("Picard box did not close")
    f_jet = regularized_rhs_jet(time_box, rho_tube, slope_tube, metric)
    total_derivative = (
        f_jet.derivative[0]
        + f_jet.derivative[1] * slope_tube
        + f_jet.derivative[2] * f_tube
    )
    slope1 = slope0 + step * f0 + 0.5 * step**2 * total_derivative
    rho1 = (
        rho0 + step * slope0 + 0.5 * step**2 * f0
        + (step**3 / 6.0) * total_derivative
    )
    return {"rho": rho1, "slope": slope1}, {
        "picard_iterations": iteration + 1,
        "rho_tube": rho_tube,
        "slope_tube": slope_tube,
        "f_tube": f_tube,
        "total_derivative": total_derivative,
    }


def propagate_launch_cell(axis_radius, metric, rho_bounds=(0.10, 1.67),
                          theta_axis=1e-3, initial_step=0.004,
                          minimum_step=1.0e-5, maximum_steps=200000):
    """Propagate one launch interval and return a certificate classification."""
    axis_radius = as_interval(axis_radius)
    try:
        state, axis = regular_axis_cone(axis_radius, metric, theta_axis)
    except Exception as error:
        return {
            "classification": "unresolved_axis_cone",
            "reason": repr(error), "axis_radius": axis_radius,
        }
    theta = float(theta_axis)
    step = float(initial_step)
    step_rejections = 0
    maximum_width = max(state["rho"].width, state["slope"].width)
    for step_index in range(int(maximum_steps)):
        if theta >= math.pi / 2:
            break
        local_step = min(step, math.pi / 2 - theta)
        try:
            next_state, audit = validated_taylor_step(
                theta, state, local_step, metric,
            )
        except (ValueError, ZeroDivisionError, FloatingPointError, RuntimeError):
            local_step *= 0.5
            step = local_step
            step_rejections += 1
            if step < minimum_step:
                return {
                    "classification": "unresolved_step",
                    "reason": "minimum validated step exhausted",
                    "axis_radius": axis_radius, "theta": theta,
                    "state": state, "axis": axis,
                    "step_rejections": step_rejections,
                }
            continue
        theta += local_step
        state = next_state
        maximum_width = max(
            maximum_width, state["rho"].width, state["slope"].width,
        )
        lower, upper = map(float, rho_bounds)
        if state["rho"].lower > upper and state["slope"].lower > 0.0:
            return {
                "classification": "oriented_upper_band_exit",
                "axis_radius": axis_radius, "theta": theta,
                "state": state, "axis": axis, "step_count": step_index + 1,
                "step_rejections": step_rejections,
                "maximum_interval_width": maximum_width,
            }
        if state["rho"].upper < lower and state["slope"].upper < 0.0:
            return {
                "classification": "oriented_lower_band_exit",
                "axis_radius": axis_radius, "theta": theta,
                "state": state, "axis": axis, "step_count": step_index + 1,
                "step_rejections": step_rejections,
                "maximum_interval_width": maximum_width,
            }
        if audit["picard_iterations"] <= 5 and step < initial_step:
            step = min(initial_step, 2.0 * step)
    else:
        return {
            "classification": "unresolved_step_limit",
            "axis_radius": axis_radius, "theta": theta,
            "state": state, "axis": axis,
            "step_rejections": step_rejections,
        }
    if state["slope"].lower > 0.0:
        classification = "brane_root_free_positive"
    elif state["slope"].upper < 0.0:
        classification = "brane_root_free_negative"
    else:
        classification = "unresolved_zero_residual"
    return {
        "classification": classification,
        "axis_radius": axis_radius, "theta": theta,
        "state": state, "axis": axis, "step_count": step_index,
        "step_rejections": step_rejections,
        "maximum_interval_width": maximum_width,
    }


def interval_to_list(value):
    value = as_interval(value)
    return [value.lower, value.upper]


def jsonable_interval_tree(value):
    if isinstance(value, VInterval):
        return interval_to_list(value)
    if isinstance(value, dict):
        return {key: jsonable_interval_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_interval_tree(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value
