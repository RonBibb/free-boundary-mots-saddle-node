from __future__ import annotations

import numpy as np

from bhps.joint_parent_shape import (
    CANONICAL_R,
    CANONICAL_Z,
    SHAPE_NORMALIZATION_SHA256,
    frozen_shape_fields_with_radial_derivative,
    shape_normalization_record,
)


def _coefficients():
    with np.load(
        "results/corrected_family_knot_A8_state.npz", allow_pickle=False,
    ) as archive:
        return np.asarray(archive["coefficients"]).copy()


def test_frozen_shape_normalization_digest_is_stable():
    assert shape_normalization_record()["sha256"] == SHAPE_NORMALIZATION_SHA256


def test_common_shape_convention_reproduces_sealed_g5r12_arrays_to_roundoff():
    with np.load(
        "results/corrected_family_knot_A8_state.npz", allow_pickle=False,
    ) as archive:
        coefficients = np.asarray(archive["coefficients"])
        expected = tuple(
            np.asarray(archive[f"{name}_G5R12"])
            for name in ("a", "b", "c")
        )
    found = frozen_shape_fields_with_radial_derivative(
        CANONICAL_Z, CANONICAL_R, coefficients,
    )[:3]
    for actual, target in zip(found, expected):
        np.testing.assert_allclose(actual, target, rtol=0.0, atol=5e-17)


def test_shape_and_radial_derivative_are_trace_free():
    z = np.linspace(1.0, np.e, 17)
    r = np.linspace(0.0, 12.0, 31)
    a, b, c, a_r, b_r, c_r, _ = (
        frozen_shape_fields_with_radial_derivative(z, r, _coefficients())
    )
    assert np.max(np.abs(a+b+2.0*c)) < 2e-14
    assert np.max(np.abs(a_r+b_r+2.0*c_r)) < 2e-14


def test_analytic_radial_derivative_matches_centered_difference():
    z = np.linspace(1.0, np.e, 11)
    r0 = np.asarray((1.3, 4.7, 9.1, 11.4))
    step = 1e-5
    coefficients = _coefficients()
    fields = frozen_shape_fields_with_radial_derivative(
        z, r0, coefficients,
    )
    plus = frozen_shape_fields_with_radial_derivative(
        z, r0+step, coefficients,
    )[:3]
    minus = frozen_shape_fields_with_radial_derivative(
        z, r0-step, coefficients,
    )[:3]
    for derivative, high, low in zip(fields[3:6], plus, minus):
        np.testing.assert_allclose(
            derivative, (high-low)/(2.0*step), rtol=2e-8, atol=2e-9,
        )
