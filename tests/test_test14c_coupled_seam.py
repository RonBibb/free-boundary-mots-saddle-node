import math

import numpy as np

from bhps.test14c_coupled_seam import (
    analytic_controls,
    anisotropy_controls,
    cap_boundary_variation,
    combined_intrinsic_joint_variation,
    compatible_israel_rate,
    coupled_from_uncombined,
    coupled_seam_rate,
    seam_intrinsic_derivative,
    seam_intrinsic_integral,
    smoothing_layer_record,
)


def test_intrinsic_seam_and_charge_rate_factors():
    area_radius = 1.6
    leaf_rate = 0.25
    sphere_area = 15.0
    coefficient = -1.0
    h_sphere = -2.0
    intrinsic = seam_intrinsic_integral(sphere_area, coefficient)
    assert intrinsic == -120.0
    result = coupled_seam_rate(
        area_radius, leaf_rate, sphere_area, coefficient, h_sphere,
    )
    expected = area_radius * intrinsic * leaf_rate / 12.0
    expected += 2.0 * area_radius * sphere_area * coefficient * h_sphere
    assert result["total"] == expected


def test_uncombined_identity_reduces_to_coupled_limit():
    area = 12.0
    coefficient = -0.8
    h_meridional = 0.17
    h_sphere = -0.31
    h_sphere_s = 0.29
    coefficient_rate = compatible_israel_rate(
        h_sphere_s, coefficient, h_meridional,
    )
    boundary = cap_boundary_variation(
        area, coefficient, h_meridional, h_sphere, h_sphere_s,
    )
    seam_rate = seam_intrinsic_derivative(
        area, coefficient, coefficient_rate, h_sphere,
    )
    expected = combined_intrinsic_joint_variation(
        area, coefficient, h_sphere,
    )
    assert math.isclose(boundary + seam_rate, expected, rel_tol=1e-14)
    result = coupled_from_uncombined(
        area, coefficient, coefficient_rate, h_meridional, h_sphere,
        h_sphere_s,
    )
    assert result["compatibility_error"] < 1e-14


def test_orientation_requires_both_signed_factors_to_reverse():
    original = combined_intrinsic_joint_variation(10.0, -0.7, 0.4)
    correct = combined_intrinsic_joint_variation(10.0, 0.7, -0.4)
    incorrect = combined_intrinsic_joint_variation(10.0, 0.7, 0.4)
    assert correct == original
    assert incorrect == -original


def test_polynomial_smoothing_recovers_signed_seam_limit():
    result = smoothing_layer_record("polynomial", 0.005, 128)
    assert result["finite"]
    assert result["rate_error"] < 2e-4
    assert np.sign(result["layer_rate"]) == np.sign(
        result["expected_coupled_rate"]
    )


def test_all_analytic_and_smoothing_controls_pass():
    result = analytic_controls()
    assert result["passed"]
    assert result["smoothing"]["maximum_finest_rate_error"] < 0.01
    assert result["smoothing"]["family_spread_relative_scale_error"] < 0.01


def test_intrinsic_anisotropy_controls_pass_without_fitted_coefficient():
    result = anisotropy_controls()
    assert result["passed"]
    assert result["einstein_leaf_zero"] < 1e-12
    assert result["isotropic_deformation_zero"] < 1e-12
    assert max(
        item["direct_first_variation_error"]
        for item in result["manufactured"]
    ) < 2e-4
    assert min(
        item["omission_difference"] for item in result["manufactured"]
    ) > 0.01
