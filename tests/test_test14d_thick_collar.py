import math

from bhps.test14d_thick_collar import (
    FAMILIES,
    collar_grid,
    collar_record,
    manufactured_controls,
    profile_diagnostics,
)


def test_profiles_preserve_parity_moments_and_endpoints():
    epsilon = 0.02
    for family in FAMILIES:
        x = collar_grid(epsilon, 128)
        diagnostics = profile_diagnostics(x, epsilon, family)
        assert diagnostics["finite"]
        assert diagnostics["q_odd_error"] < 1e-12
        assert diagnostics["delta_even_error"] < 1e-12
        assert diagnostics["normalization_error"] < 2e-8
        assert diagnostics["first_moment_scaled_error"] < 2e-8
        assert diagnostics["rms_width_relative_error"] < 1e-3
        assert diagnostics["p_endpoint_error"] < 1e-12


def test_junction_coefficient_and_rate_are_preserved():
    record = collar_record("compact_c2", 0.01, 128)
    assert record["finite"]
    assert record["junction_error"] < 2e-6
    assert record["junction_rate_error"] < 2e-4
    assert math.isclose(
        record["junction_target"], 6.0 * record["parameters"]["c"],
        rel_tol=1e-14,
    )


def test_static_and_flat_collars_have_no_completed_seam_work():
    static = collar_record(
        "erf", 0.01, 128, c_rate=0.0, h_sphere=0.0,
        h_meridional=0.0, area_fractional_rate=0.0,
    )
    flat = collar_record(
        "tanh", 0.01, 128, c=0.0, c_rate=0.0,
    )
    assert abs(static["finite_seam_rate"]) < 1e-12
    assert abs(flat["curvature_excess"]) < 1e-12
    assert abs(flat["finite_seam_rate"]) < 1e-12


def test_full_manufactured_control_matrix_passes():
    result = manufactured_controls()
    assert result["finite"]
    assert result["passed"], [
        name for name, passed in result["gates"].items() if not passed
    ]
