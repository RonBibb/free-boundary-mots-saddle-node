"""Maximum-principle and interval barriers for donor-capped minimal graphs."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


NEGATIVE_INFINITY = -float("inf")
POSITIVE_INFINITY = float("inf")


def _down(value: float) -> float:
    return float(np.nextafter(float(value), NEGATIVE_INFINITY))


def _up(value: float) -> float:
    return float(np.nextafter(float(value), POSITIVE_INFINITY))


@dataclass(frozen=True)
class Interval:
    """Small outward-rounded binary64 interval used by the discrete audit."""

    lower: float
    upper: float

    def __post_init__(self):
        if math.isnan(self.lower) or math.isnan(self.upper):
            raise ValueError("interval endpoint is NaN")
        if self.lower > self.upper:
            raise ValueError("interval endpoints are reversed")

    @classmethod
    def point(cls, value: float) -> "Interval":
        return cls(float(value), float(value))

    @property
    def width(self) -> float:
        return float(self.upper - self.lower)

    def __neg__(self) -> "Interval":
        return Interval(_down(-self.upper), _up(-self.lower))

    def __add__(self, other) -> "Interval":
        other = as_interval(other)
        return Interval(
            _down(self.lower + other.lower), _up(self.upper + other.upper),
        )

    __radd__ = __add__

    def __sub__(self, other) -> "Interval":
        return self + (-as_interval(other))

    def __rsub__(self, other) -> "Interval":
        return as_interval(other) - self

    def __mul__(self, other) -> "Interval":
        other = as_interval(other)
        products = (
            self.lower * other.lower, self.lower * other.upper,
            self.upper * other.lower, self.upper * other.upper,
        )
        return Interval(_down(min(products)), _up(max(products)))

    __rmul__ = __mul__

    def reciprocal(self) -> "Interval":
        if self.lower <= 0.0 <= self.upper:
            raise ZeroDivisionError("interval contains zero")
        values = (1.0 / self.lower, 1.0 / self.upper)
        return Interval(_down(min(values)), _up(max(values)))

    def __truediv__(self, other) -> "Interval":
        return self * as_interval(other).reciprocal()

    def __rtruediv__(self, other) -> "Interval":
        return as_interval(other) / self

    def __pow__(self, exponent: int) -> "Interval":
        exponent = int(exponent)
        if exponent < 0:
            return (self ** (-exponent)).reciprocal()
        if exponent == 0:
            return Interval.point(1.0)
        if exponent == 1:
            return self
        if exponent % 2 == 0 and self.lower <= 0.0 <= self.upper:
            high = max(abs(self.lower), abs(self.upper)) ** exponent
            return Interval(0.0, _up(high))
        values = (self.lower ** exponent, self.upper ** exponent)
        return Interval(_down(min(values)), _up(max(values)))


def as_interval(value) -> Interval:
    return value if isinstance(value, Interval) else Interval.point(float(value))


def interval_sin_nonnegative(value: Interval) -> Interval:
    """Sine enclosure for a subinterval of ``[0, pi/2]``."""
    if value.lower < 0.0 or value.upper > math.pi / 2 + 1e-15:
        raise ValueError("sine helper requires [0, pi/2]")
    lower = math.sin(value.lower)
    upper = math.sin(value.upper)
    for _ in range(4):
        lower = np.nextafter(lower, NEGATIVE_INFINITY)
        upper = np.nextafter(upper, POSITIVE_INFINITY)
    return Interval(max(0.0, float(lower)), min(1.0, float(upper)))


def interval_cos_nonnegative(value: Interval) -> Interval:
    """Cosine enclosure for a subinterval of ``[0, pi/2]``."""
    if value.lower < 0.0 or value.upper > math.pi / 2 + 1e-15:
        raise ValueError("cosine helper requires [0, pi/2]")
    lower = 0.0 if value.upper >= math.pi / 2 else math.cos(value.upper)
    upper = math.cos(value.lower)
    for _ in range(4):
        lower = np.nextafter(lower, NEGATIVE_INFINITY)
        upper = np.nextafter(upper, POSITIVE_INFINITY)
    return Interval(max(0.0, float(lower)), min(1.0, float(upper)))


def barrier_from_values(rho, sine, cosine, values):
    """Evaluate the reduced dimensionless constant-cap barrier.

    ``values`` supplies ``A, B, C`` and their ``z``/``r`` derivatives. Inputs
    can be scalars, NumPy arrays, or :class:`Interval` objects.
    """
    A = values["A"]
    B = values["B"]
    C = values["C"]
    Az = values["Az"]
    Ar = values["Ar"]
    Bz = values["Bz"]
    Br = values["Br"]
    Cz = values["Cz"]
    Cr = values["Cr"]
    s = sine
    c = cosine
    R = rho
    numerator = (
        2 * A**4 * C * s**2
        + 2 * A**4 * Cr * R * s**3
        + A**3 * Ar * C * R * s**3
        + 3 * A**2 * B**2 * C * c**2
        + A**2 * B**2 * C * s**2
        + 2 * A**2 * B**2 * Cr * R * c**2 * s
        - 2 * A**2 * B**2 * Cz * R * c * s**2
        - A**2 * B * Br * C * R * c**2 * s
        - 2 * A**2 * B * Bz * C * R * c * s**2
        + 2 * A * Ar * B**2 * C * R * c**2 * s
        + A * Az * B**2 * C * R * c * s**2
        - 2 * B**4 * Cz * R * c**3
        - B**3 * Bz * C * R * c**3
    )
    return numerator / (A**2 * B**2 * C)


class BilinearMetricEnclosure:
    """Interval hulls for a piecewise-bilinear diagonal spatial metric."""

    def __init__(self, z, r, A, B, C):
        self.z = np.asarray(z, dtype=float)
        self.r = np.asarray(r, dtype=float)
        self.fields = {
            "A": np.asarray(A, dtype=float),
            "B": np.asarray(B, dtype=float),
            "C": np.asarray(C, dtype=float),
        }
        shape = (len(self.z), len(self.r))
        if len(self.z) < 2 or len(self.r) < 2:
            raise ValueError("metric grid is too small")
        if np.any(np.diff(self.z) <= 0.0) or np.any(np.diff(self.r) <= 0.0):
            raise ValueError("metric coordinates must increase")
        for name, field in self.fields.items():
            if field.shape != shape or not np.all(np.isfinite(field)):
                raise ValueError(f"invalid {name} field")
            if np.min(field) <= 0.0:
                raise ValueError(f"nonpositive {name} scale factor")

    @classmethod
    def flat(cls, z=None, r=None):
        z = np.linspace(0.8, 3.0, 9) if z is None else np.asarray(z)
        r = np.linspace(0.0, 2.0, 11) if r is None else np.asarray(r)
        ones = np.ones((len(z), len(r)))
        return cls(z, r, ones, ones, ones)

    @property
    def z_brane(self) -> float:
        return float(self.z[-1])

    @staticmethod
    def _cell_span(coordinate, lower, upper):
        first = int(np.searchsorted(coordinate, lower, side="right") - 1)
        last = int(np.searchsorted(coordinate, upper, side="left"))
        first = min(max(first, 0), len(coordinate) - 2)
        last = min(max(last, 0), len(coordinate) - 2)
        return first, last

    def _field_enclosure(self, name, zbox: Interval, rbox: Interval):
        i0, i1 = self._cell_span(self.z, zbox.lower, zbox.upper)
        j0, j1 = self._cell_span(self.r, rbox.lower, rbox.upper)
        field = self.fields[name]
        nodes = field[i0:i1 + 2, j0:j1 + 2]
        value = Interval(_down(np.min(nodes)), _up(np.max(nodes)))

        dz = np.diff(self.z)[i0:i1 + 1, None]
        z_difference = (
            field[i0 + 1:i1 + 2, j0:j1 + 2]
            - field[i0:i1 + 1, j0:j1 + 2]
        )
        z_lower = np.nextafter(z_difference, NEGATIVE_INFINITY)
        z_upper = np.nextafter(z_difference, POSITIVE_INFINITY)
        z_lower = np.nextafter(z_lower / dz, NEGATIVE_INFINITY)
        z_upper = np.nextafter(z_upper / dz, POSITIVE_INFINITY)
        derivative_z = Interval(float(np.min(z_lower)), float(np.max(z_upper)))

        dr = np.diff(self.r)[None, j0:j1 + 1]
        r_difference = (
            field[i0:i1 + 2, j0 + 1:j1 + 2]
            - field[i0:i1 + 2, j0:j1 + 1]
        )
        r_lower = np.nextafter(r_difference, NEGATIVE_INFINITY)
        r_upper = np.nextafter(r_difference, POSITIVE_INFINITY)
        r_lower = np.nextafter(r_lower / dr, NEGATIVE_INFINITY)
        r_upper = np.nextafter(r_upper / dr, POSITIVE_INFINITY)
        derivative_r = Interval(float(np.min(r_lower)), float(np.max(r_upper)))
        return value, derivative_z, derivative_r

    def barrier_interval(
        self, theta_lower, theta_upper, rho_lower, rho_upper,
    ) -> Interval:
        theta = Interval(float(theta_lower), float(theta_upper))
        rho = Interval(float(rho_lower), float(rho_upper))
        sine = interval_sin_nonnegative(theta)
        cosine = interval_cos_nonnegative(theta)
        radius = rho * sine
        zcoord = Interval.point(self.z_brane) - rho * cosine
        tolerance = 16.0 * np.finfo(float).eps * max(
            abs(self.z[0]), abs(self.z[-1]), abs(self.r[-1]), 1.0,
        )
        if zcoord.lower >= self.z[0] - tolerance and zcoord.upper <= self.z[-1] + tolerance:
            zcoord = Interval(max(zcoord.lower, self.z[0]), min(zcoord.upper, self.z[-1]))
        if radius.lower >= self.r[0] - tolerance and radius.upper <= self.r[-1] + tolerance:
            radius = Interval(max(radius.lower, self.r[0]), min(radius.upper, self.r[-1]))
        if (
            zcoord.lower < self.z[0] or zcoord.upper > self.z[-1]
            or radius.lower < self.r[0] or radius.upper > self.r[-1]
        ):
            raise ValueError("barrier parameter box leaves metric domain")
        values = {}
        for name in ("A", "B", "C"):
            value, derivative_z, derivative_r = self._field_enclosure(
                name, zcoord, radius,
            )
            values[name] = value
            values[name + "z"] = derivative_z
            values[name + "r"] = derivative_r
        result = barrier_from_values(rho, sine, cosine, values)
        if not isinstance(result, Interval):
            raise TypeError("interval evaluation lost its enclosure")
        return result


def point_barrier_from_splines(theta, rho, z_brane, splines):
    """Floating-point barrier on the established bicubic representation."""
    theta = np.asarray(theta, dtype=float)
    rho = np.asarray(rho, dtype=float)
    sine = np.sin(theta)
    cosine = np.cos(theta)
    radius = rho * sine
    zcoord = float(z_brane) - rho * cosine
    values = {}
    for name, spline in splines.items():
        values[name] = spline.ev(zcoord, radius)
        values[name + "z"] = spline.ev(zcoord, radius, dx=1, dy=0)
        values[name + "r"] = spline.ev(zcoord, radius, dx=0, dy=1)
    return barrier_from_values(rho, sine, cosine, values)


@dataclass(frozen=True)
class ParameterBox:
    theta_lower: float
    theta_upper: float
    rho_lower: float
    rho_upper: float
    depth: int = 0

    @property
    def area(self) -> float:
        return ((self.theta_upper - self.theta_lower)
                * (self.rho_upper - self.rho_lower))

    def bisect(self, theta_span: float, rho_span: float):
        theta_width = (self.theta_upper - self.theta_lower) / theta_span
        rho_width = (self.rho_upper - self.rho_lower) / rho_span
        if theta_width >= rho_width:
            middle = 0.5 * (self.theta_lower + self.theta_upper)
            return (
                ParameterBox(
                    self.theta_lower, middle, self.rho_lower, self.rho_upper,
                    self.depth + 1,
                ),
                ParameterBox(
                    middle, self.theta_upper, self.rho_lower, self.rho_upper,
                    self.depth + 1,
                ),
            )
        middle = 0.5 * (self.rho_lower + self.rho_upper)
        return (
            ParameterBox(
                self.theta_lower, self.theta_upper, self.rho_lower, middle,
                self.depth + 1,
            ),
            ParameterBox(
                self.theta_lower, self.theta_upper, middle, self.rho_upper,
                self.depth + 1,
            ),
        )

    def to_list(self):
        return [
            self.theta_lower, self.theta_upper,
            self.rho_lower, self.rho_upper, self.depth,
        ]

    @classmethod
    def from_list(cls, values):
        return cls(
            float(values[0]), float(values[1]), float(values[2]),
            float(values[3]), int(values[4]),
        )


def initial_parameter_boxes(
    theta_count=64, rho_count=64, rho_bounds=(0.10, 1.67),
):
    theta_edges = np.linspace(0.0, math.pi / 2, int(theta_count) + 1)
    rho_edges = np.linspace(float(rho_bounds[0]), float(rho_bounds[1]), int(rho_count) + 1)
    return [
        ParameterBox(theta_edges[i], theta_edges[i + 1], rho_edges[j], rho_edges[j + 1])
        for i in range(int(theta_count)) for j in range(int(rho_count))
    ]


def process_cover_chunk(
    metric, queue, certified, nonpositive, unresolved, maximum_evaluations=512,
    threshold=1e-10, maximum_depth=20, maximum_terminal_boxes=2_000_000,
    theta_span=math.pi / 2, rho_span=1.57,
):
    """Advance an adaptive interval cover by a deterministic bounded chunk."""
    queue = list(queue)
    certified = list(certified)
    nonpositive = list(nonpositive)
    unresolved = list(unresolved)
    evaluations = 0
    minimum_lower = POSITIVE_INFINITY
    while queue and evaluations < int(maximum_evaluations):
        box = queue.pop(0)
        enclosure = metric.barrier_interval(
            box.theta_lower, box.theta_upper, box.rho_lower, box.rho_upper,
        )
        evaluations += 1
        minimum_lower = min(minimum_lower, enclosure.lower)
        record = box.to_list() + [enclosure.lower, enclosure.upper]
        if enclosure.lower > threshold:
            certified.append(record)
        elif enclosure.upper < 0.0:
            nonpositive.append(record)
        elif box.depth >= int(maximum_depth):
            unresolved.append(record)
        elif len(certified) + len(nonpositive) + len(unresolved) + len(queue) >= int(maximum_terminal_boxes):
            unresolved.append(record)
        else:
            queue.extend(box.bisect(theta_span, rho_span))
    return {
        "queue": queue,
        "certified": certified,
        "nonpositive": nonpositive,
        "unresolved": unresolved,
        "evaluations": evaluations,
        "minimum_lower_in_chunk": minimum_lower,
    }


def cover_summary(queue, certified, nonpositive, unresolved, total_area):
    def area(records):
        return float(sum(
            (item[1] - item[0]) * (item[3] - item[2]) for item in records
        ))

    certified_area = area(certified)
    nonpositive_area = area(nonpositive)
    unresolved_area = area(unresolved) + sum(box.area for box in queue)
    return {
        "complete": not queue,
        "certified_box_count": len(certified),
        "nonpositive_box_count": len(nonpositive),
        "unresolved_box_count": len(unresolved) + len(queue),
        "certified_area_fraction": certified_area / total_area,
        "nonpositive_area_fraction": nonpositive_area / total_area,
        "unresolved_area_fraction": unresolved_area / total_area,
        "minimum_certified_lower_bound": float(min(
            (item[5] for item in certified), default=POSITIVE_INFINITY,
        )),
    }
