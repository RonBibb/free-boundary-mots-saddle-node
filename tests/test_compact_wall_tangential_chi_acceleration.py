import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.junction_second_preservation_diagnostic import (
    wall_junction_second_tangent,
)
from bhps.nonlinear_regular_so3_evolution import (
    compact_wall_normal_gauge_acceleration_residuals,
    solve_compact_wall_coupled_phi_normal_acceleration,
    solve_compact_wall_tangential_chi_acceleration,
)


def _bits(values):
    return np.ascontiguousarray(values).view(np.uint64)


def _off_manifold_time_symmetric_case():
    z = np.linspace(1.0, 2.0, 17)
    r = np.linspace(0.0, 2.0, 13)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    q = np.zeros((len(z), len(r), 9))
    q[:, :, 2] = -1.25 + 0.035 * zz + 0.004 * rr**2
    q[:, :, 3] = 1.05 + 0.028 * zz + 0.007 * rr**2
    q[:, :, 4] = 0.018 + 0.003 * zz + 0.001 * rr**2
    q[:, :, 5] = 0.012 - 0.002 * zz + 0.0005 * rr**2
    q[:, :, 6] = 1.15 + 0.045 * zz + 0.006 * rr**2
    q[:, :, 7] = 0.025 * zz - 0.008 * rr**2
    q[:, :, 8] = 0.04 * np.cos(0.7 * zz) * np.exp(-0.1 * rr**2)
    velocity = np.zeros_like(q)
    acceleration = np.empty_like(q)
    for field in range(9):
        acceleration[:, :, field] = (
            0.025
            * (field + 1)
            * (1.0 + 0.08 * zz + 0.015 * rr**2)
        )
    # Distinct protected values make accidental reassignment easy to detect.
    acceleration[:, :, 0] *= -1.0
    acceleration[:, :, 1] += 0.017
    acceleration[:, :, 6] += 0.21 + 0.01 * zz
    acceleration[:, :, 7] -= 0.13 - 0.004 * rr**2
    background = {
        "wall_stiffness": 3.5,
        "v0": -0.08,
        "v1": 0.12,
        "beta_a": 0.31,
        "beta_b": -0.27,
        "wall_potential_a": 0.014,
        "wall_potential_b": 0.009,
    }
    return z, r, q, velocity, acceleration, background


def test_selective_owner_closes_normalized_rows_and_preserves_protected_bits():
    z, r, q, velocity, acceleration, background = (
        _off_manifold_time_symmetric_case()
    )
    original = acceleration.copy()
    solved, diagnostic = solve_compact_wall_tangential_chi_acceleration(
        q,
        velocity,
        acceleration,
        z,
        r,
        background,
        capture_profiles=True,
    )

    np.testing.assert_array_equal(_bits(acceleration), _bits(original))
    np.testing.assert_array_equal(
        _bits(solved[:, :, (0, 1, 6, 7)]),
        _bits(original[:, :, (0, 1, 6, 7)]),
    )
    np.testing.assert_array_equal(
        _bits(solved[:, 0, 4:6]), _bits(original[:, 0, 4:6]),
    )
    np.testing.assert_array_equal(_bits(solved[1:-1]), _bits(original[1:-1]))
    assert diagnostic["protected_0_1_6_7_bitwise"]
    assert diagnostic["q4_q5_axis_bitwise"]
    assert diagnostic["passed"]
    algebraic = diagnostic["per_field_algebraic_evidence"]
    assert algebraic["field_order"] == (
        "h_00", "h_perp", "h_rr", "h_0r", "chi",
    )
    assert algebraic["each_field_gated_separately"]
    assert algebraic["chi_credited_only_with_chi_block"]
    assert algebraic["passed"]
    for field in algebraic["field_order"]:
        evidence = algebraic["fields"][field]
        assert evidence["minimum_rank"] == 2
        assert evidence["maximum_equilibrated_condition"] <= 1e12
        assert evidence["minimum_normalized_pivot"] >= 1e-10
        assert evidence["maximum_normalized_linear_residual"] < 1e-12
        assert all(
            np.asarray(profile).shape == (len(r),)
            for profile in evidence["profiles"].values()
        )

    for wall in ("lower", "upper"):
        independent = wall_junction_second_tangent(
            q, velocity, solved, z, r, background, wall, 7,
        )
        for component in ("tt", "sphere", "rr", "tr"):
            assert np.max(np.abs(
                independent["components"][component]["DX2J"]
            )) < 2e-12
            np.testing.assert_array_equal(
                independent["components"][component][
                    "D2J_velocity_velocity"
                ],
                np.zeros(len(r)),
            )
        assert np.max(np.abs(
            independent["separate_rows"]["DX2_chi_neumann"]
        )) < 2e-12


def test_normalized_owner_is_not_the_raw_robin_tt_owner_and_profiles_corners():
    z, r, q, velocity, acceleration, background = (
        _off_manifold_time_symmetric_case()
    )
    solved, diagnostic = solve_compact_wall_tangential_chi_acceleration(
        q,
        velocity,
        acceleration,
        z,
        r,
        background,
        capture_profiles=True,
    )
    raw_maximum = 0.0
    for wall_number, wall in enumerate(("lower", "upper")):
        independent = wall_junction_second_tangent(
            q, velocity, solved, z, r, background, wall, 7,
        )
        profile = diagnostic["walls"][wall_number]
        np.testing.assert_array_equal(
            profile["radial_indices"], np.arange(len(r)),
        )
        assert profile["radial_indices"][0] == 0
        assert profile["radial_indices"][-1] == len(r) - 1
        for component in ("tt", "sphere", "rr", "tr"):
            component_profile = profile["components"][component]
            assert component_profile["terms"].shape == (4, len(r))
            assert component_profile["residual"].shape == (len(r),)
            assert component_profile["normalized"].shape == (len(r),)
            raw_maximum = max(
                raw_maximum,
                float(np.max(np.abs(
                    independent["components"][component][
                        "DX2_robin_residual"
                    ]
                ))),
            )
        assert profile["chi"]["residual"].shape == (len(r),)
        assert profile["chi"]["contributions"].shape == (len(z), len(r))

    # Off the J=0 manifold, normalized D_X^2J=0 contains the denominator
    # derivative and therefore intentionally does not solve raw Robin_tt=0.
    assert raw_maximum > 1e-3
    assert diagnostic["maximum_metric_normalized_residual"] < 1e-12
    assert diagnostic["maximum_chi_normalized_residual"] < 1e-12


def test_row_implied_owned_endpoint_derivatives_and_time_symmetry_guard():
    z, r, q, velocity, acceleration, background = (
        _off_manifold_time_symmetric_case()
    )
    solved, diagnostic = solve_compact_wall_tangential_chi_acceleration(
        q, velocity, acceleration, z, r, background, capture_profiles=True,
    )
    del solved
    expected_mask = np.asarray(
        (False, False, True, True, True, True, False, False, True),
    )
    np.testing.assert_array_equal(diagnostic["row_defined_mask"], expected_mask)
    assert diagnostic["direct_physical_a_z"].shape == (2, len(r), 9)
    assert diagnostic["row_implied_physical_a_z"].shape == (2, len(r), 9)
    np.testing.assert_allclose(
        diagnostic["direct_physical_a_z"][:, :, expected_mask],
        diagnostic["row_implied_physical_a_z"][:, :, expected_mask],
        rtol=2e-13,
        atol=2e-12,
    )
    assert diagnostic["maximum_row_implied_scaled_defect"] < 1e-12

    non_time_symmetric = velocity.copy()
    non_time_symmetric[4, 5, 3] = 1e-30
    with pytest.raises(ValueError, match="exact time symmetry"):
        solve_compact_wall_tangential_chi_acceleration(
            q, non_time_symmetric, acceleration, z, r, background,
        )


def test_selective_owner_preserves_a_completed_coupled_phi_normal_block():
    z, r, q, velocity, acceleration, background = (
        _off_manifold_time_symmetric_case()
    )
    source = np.zeros((len(z), len(r), 3))
    coupled, coupled_record = solve_compact_wall_coupled_phi_normal_acceleration(
        q,
        velocity,
        acceleration,
        source,
        source,
        source,
        z,
        r,
        background,
        capture_profiles=True,
    )
    assert coupled_record["passed"]
    coupled_bits = _bits(coupled[:, :, (6, 7)]).copy()
    solved, _ = solve_compact_wall_tangential_chi_acceleration(
        q, velocity, coupled, z, r, background, capture_profiles=True,
    )
    np.testing.assert_array_equal(_bits(solved[:, :, (6, 7)]), coupled_bits)

    normal = compact_wall_normal_gauge_acceleration_residuals(
        q,
        velocity,
        solved,
        source,
        source,
        source,
        z,
        r,
        background,
        radial_buffer=0,
        capture_profiles=True,
    )
    assert normal["maximum"] < 2e-12
    for wall in ("lower", "upper"):
        independent = wall_junction_second_tangent(
            q, velocity, solved, z, r, background, wall, 7,
        )
        assert np.max(np.abs(
            independent["separate_rows"]["DX2_Phi_robin"]
        )) < 2e-12
