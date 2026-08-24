"""Grid-independent evaluation of the frozen Protocol-125 shape branch.

The legacy basis normalized several smooth profiles by a maximum sampled on
the caller's grid.  That is harmless for a single construction but makes two
nominally identical refinement parents carry slightly different physical
input.  Protocol 125 instead freezes those normalizations on the sealed
G5R12 knot grid and evaluates the same analytic functions on every later
grid.  The convention reproduces the sealed G5R12 shape arrays to IEEE
roundoff while removing parent-grid normalization drift.
"""

from __future__ import annotations

import math

import numpy as np

from bhps.matched_staged_continuum import hash_arrays
from bhps.physical_corner_corrector import combine_shape_modes


BASIS_RADIUS = 8.0
AXIS_WIDTHS = (0.5, 1.0)
ANNULAR_PROFILES = ((7.5, 1.5), (7.5, 3.0))
RADIAL_MODES = 6
CANONICAL_Z = np.linspace(1.0, np.e, 49)
CANONICAL_R = np.linspace(0.0, 12.0, 109)
# Filled from ``shape_normalization_record()['sha256']`` and checked on every
# call.  It binds the input convention, not a constructed parent result.
SHAPE_NORMALIZATION_SHA256 = (
    "30c9570dc3909d43f2f2c731985ebf1c780b16b4f55a7463f6ff21989e728dc7"
)


def _compact_polynomials(z, normalization_z=CANONICAL_Z):
    z = np.asarray(z, dtype=float)
    normalization_z = np.asarray(normalization_z, dtype=float)
    matrix = np.zeros((8, 8))
    conditions = []
    row = 0
    for endpoint in (0.0, 1.0):
        for order in range(4):
            for power in range(order, 8):
                matrix[row, power] = (
                    math.factorial(power)
                    / math.factorial(power-order)
                    * endpoint**(power-order)
                )
            conditions.append((endpoint, order))
            row += 1
    x = (z-1.0)/(np.e-1.0)
    canonical_x = (normalization_z-1.0)/(np.e-1.0)
    profiles = []
    normalizations = []
    for endpoint, order in ((0.0, 2), (0.0, 3), (1.0, 2), (1.0, 3)):
        target = np.zeros(8)
        target[conditions.index((endpoint, order))] = 1.0
        coefficients = np.linalg.solve(matrix, target)
        canonical = np.polynomial.polynomial.polyval(
            canonical_x, coefficients,
        )
        normalization = float(np.max(np.abs(canonical)))
        profiles.append(
            np.polynomial.polynomial.polyval(x, coefficients)/normalization
        )
        normalizations.append(normalization)
    return np.asarray(profiles), np.asarray(normalizations)


def _taper(r):
    r = np.asarray(r, dtype=float)
    x = r/BASIS_RADIUS
    inside = x < 1.0
    value = np.zeros_like(r)
    derivative = np.zeros_like(r)
    y = 1.0-x[inside]**2
    value[inside] = y**4
    derivative[inside] = (
        -8.0*r[inside]/BASIS_RADIUS**2*y**3
    )
    return value, derivative


def _common_radial_raw(r, center, spherical):
    r = np.asarray(r, dtype=float)
    width = max(1.25*BASIS_RADIUS/max(RADIAL_MODES-1, 1), 0.35)
    minus = np.exp(-((r-center)/width)**2)
    plus = np.exp(-((r+center)/width)**2)
    gaussian = minus+plus
    gaussian_r = (
        -2.0*(r-center)/width**2*minus
        -2.0*(r+center)/width**2*plus
    )
    taper, taper_r = _taper(r)
    value = gaussian*taper
    derivative = gaussian_r*taper+gaussian*taper_r
    if spherical:
        x2 = (r/BASIS_RADIUS)**2
        derivative = (
            derivative*x2
            + value*(2.0*r/BASIS_RADIUS**2)
        )
        value = value*x2
    return value, derivative


def _axis_radial_raw(r, width, spherical):
    r = np.asarray(r, dtype=float)
    gaussian = np.exp(-(r/float(width))**2)
    gaussian_r = -2.0*r/float(width)**2*gaussian
    taper, taper_r = _taper(r)
    value = gaussian*taper
    derivative = gaussian_r*taper+gaussian*taper_r
    if spherical:
        x2 = (r/BASIS_RADIUS)**2
        derivative = (
            derivative*x2
            + value*(2.0*r/BASIS_RADIUS**2)
        )
        value = value*x2
    return value, derivative


def _annular_radial(r, center, width, spherical):
    r = np.asarray(r, dtype=float)
    center = float(center)
    width = float(width)
    minus = np.exp(-((r-center)/width)**2)
    plus = np.exp(-((r+center)/width)**2)
    normalization = 1.0+np.exp(-(2.0*center/width)**2)
    value = (minus+plus)/normalization
    derivative = (
        -2.0*(r-center)/width**2*minus
        -2.0*(r+center)/width**2*plus
    )/normalization
    if spherical:
        x2 = (r/center)**2
        derivative = derivative*x2+value*(2.0*r/center**2)
        value = value*x2
    return value, derivative


def _normalization_arrays():
    _, compact = _compact_polynomials(CANONICAL_Z)
    centers = np.linspace(0.0, 0.82*BASIS_RADIUS, RADIAL_MODES)
    common = []
    spherical = []
    for center in centers:
        common.append(np.max(np.abs(
            _common_radial_raw(CANONICAL_R, center, False)[0]
        )))
        spherical.append(np.max(np.abs(
            _common_radial_raw(CANONICAL_R, center, True)[0]
        )))
    axis = []
    spherical_axis = []
    for width in AXIS_WIDTHS:
        axis.append(np.max(np.abs(
            _axis_radial_raw(CANONICAL_R, width, False)[0]
        )))
        spherical_axis.append(np.max(np.abs(
            _axis_radial_raw(CANONICAL_R, width, True)[0]
        )))
    return tuple(np.asarray(value, dtype=float) for value in (
        compact, common, spherical, axis, spherical_axis,
    ))


def shape_normalization_record():
    compact, common, spherical, axis, spherical_axis = _normalization_arrays()
    digest = hash_arrays(
        CANONICAL_Z,
        CANONICAL_R,
        compact,
        common,
        spherical,
        axis,
        spherical_axis,
        np.asarray(ANNULAR_PROFILES),
    )
    return {
        "canonical_z": CANONICAL_Z.copy(),
        "canonical_r": CANONICAL_R.copy(),
        "compact": compact,
        "common": common,
        "spherical": spherical,
        "axis": axis,
        "spherical_axis": spherical_axis,
        "annular_profiles": np.asarray(ANNULAR_PROFILES),
        "sha256": digest,
    }


def frozen_shape_fields_with_radial_derivative(z, r, coefficients):
    """Return common physical ``(a,b,c)`` and analytic radial derivatives."""
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    if (
        z.ndim != 1 or r.ndim != 1
        or np.any(np.diff(z) <= 0.0) or np.any(np.diff(r) <= 0.0)
        or len(coefficients) != 80
        or not all(np.all(np.isfinite(item)) for item in (z, r, coefficients))
    ):
        raise ValueError("invalid frozen shape input")
    record = shape_normalization_record()
    if (
        SHAPE_NORMALIZATION_SHA256 != "TO_BE_FROZEN"
        and record["sha256"] != SHAPE_NORMALIZATION_SHA256
    ):
        raise RuntimeError("frozen shape normalization digest mismatch")
    compact, _ = _compact_polynomials(z)
    centers = np.linspace(0.0, 0.82*BASIS_RADIUS, RADIAL_MODES)
    modes = []
    radial_modes = []

    def append_compact(shape, shape_r):
        modes.append((3.0*shape, -shape, -shape))
        radial_modes.append((3.0*shape_r, -shape_r, -shape_r))

    def append_spherical(shape, shape_r):
        zero = np.zeros_like(shape)
        zero_r = np.zeros_like(shape_r)
        modes.append((zero, 2.0*shape, -shape))
        radial_modes.append((zero_r, 2.0*shape_r, -shape_r))

    # Preserve the sealed coefficient ordering exactly: all ordinary radial
    # modes for all four compact profiles, then all axis-localized modes,
    # then all annular modes.
    for compact_profile in compact:
        for index, center in enumerate(centers):
            radial, radial_r = _common_radial_raw(r, center, False)
            radial /= record["common"][index]
            radial_r /= record["common"][index]
            shape = compact_profile[:, None]*radial[None, :]
            shape_r = compact_profile[:, None]*radial_r[None, :]
            append_compact(shape, shape_r)
        for index, center in enumerate(centers):
            radial, radial_r = _common_radial_raw(r, center, True)
            radial /= record["spherical"][index]
            radial_r /= record["spherical"][index]
            shape = compact_profile[:, None]*radial[None, :]
            shape_r = compact_profile[:, None]*radial_r[None, :]
            append_spherical(shape, shape_r)
    for compact_profile in compact:
        for index, width in enumerate(AXIS_WIDTHS):
            radial, radial_r = _axis_radial_raw(r, width, False)
            radial /= record["axis"][index]
            radial_r /= record["axis"][index]
            shape = compact_profile[:, None]*radial[None, :]
            shape_r = compact_profile[:, None]*radial_r[None, :]
            append_compact(shape, shape_r)

            radial, radial_r = _axis_radial_raw(r, width, True)
            radial /= record["spherical_axis"][index]
            radial_r /= record["spherical_axis"][index]
            shape = compact_profile[:, None]*radial[None, :]
            shape_r = compact_profile[:, None]*radial_r[None, :]
            append_spherical(shape, shape_r)
    for compact_profile in compact:
        for center, width in ANNULAR_PROFILES:
            radial, radial_r = _annular_radial(
                r, center, width, False,
            )
            shape = compact_profile[:, None]*radial[None, :]
            shape_r = compact_profile[:, None]*radial_r[None, :]
            append_compact(shape, shape_r)

            radial, radial_r = _annular_radial(
                r, center, width, True,
            )
            shape = compact_profile[:, None]*radial[None, :]
            shape_r = compact_profile[:, None]*radial_r[None, :]
            append_spherical(shape, shape_r)
    fields = combine_shape_modes(coefficients, modes)
    derivatives = combine_shape_modes(coefficients, radial_modes)
    if not all(np.all(np.isfinite(item)) for item in (*fields, *derivatives)):
        raise RuntimeError("frozen shape map is nonfinite")
    return (*fields, *derivatives, record)
