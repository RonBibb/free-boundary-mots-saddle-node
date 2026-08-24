import numpy as np

from bhps.junction_preservation_diagnostic import (
    COMPONENTS,
    WALLS,
    cap_spline_sphere_residual,
    compare_wall_records,
    directional_derivative_ladder,
    high_precision_local_directional_ladder,
    manufactured_controls,
    manufactured_state,
    radial_zones,
    wall_junction_rows,
)
from bhps.nonlinear_regular_so3_evolution import compact_wall_position_residuals


def test_manufactured_and_adverse_controls_pass():
    result = manufactured_controls()
    assert result["passed"]
    assert result["exact_J_maximum_absolute"] < 1e-11
    assert result["exact_DXJ_maximum_absolute"] < 1e-11
    assert result["localized_upper_adverse_J_maximum"] > 1e-3
    assert result["localized_adverse_lower_leakage"] < 1e-11
    assert min(result["wrong_orientation_J_maximum"].values()) > 0.1
    assert result["off_gauge_adverse_maximum"] > 0.02


def test_directional_ladder_checks_every_component_on_both_walls():
    data = manufactured_state()
    q = data["position"]
    z = data["z"]
    r = data["r"]
    zz, rr = np.meshgrid(z, r, indexing="ij")
    direction = np.zeros_like(q)
    direction[:, :, 2] = 0.03 * (1 + zz + rr**2)
    direction[:, :, 3] = -0.02 * (1 + zz**2)
    direction[:, :, 4] = 0.004 * (1 + zz)
    direction[:, :, 5] = -0.003 * (1 + rr**2)
    direction[:, :, 6] = 0.01 * (1 + 0.1 * zz)
    direction[:, :, 7] = 0.02 * (1 + 0.1 * rr**2)
    result = directional_derivative_ladder(
        q, direction, z, r, data["background"],
    )
    assert result["finite"]
    assert result["best_maximum_relative_scale_error"] < 1e-7
    for item in result["records"]:
        assert set(item["walls"]) == set(WALLS)
        for wall in WALLS:
            assert set(item["walls"][wall]["components"]) == set(COMPONENTS)


def test_oriented_robin_conversion_and_separate_rows():
    data = manufactured_state()
    q = data["position"].copy()
    z = data["z"]
    r = data["r"]
    fraction = (z - z[0]) / (z[-1] - z[0])
    q[:, :, 3] += 0.02 * (fraction**2 * (fraction - 1))[:, None]
    zero = np.zeros_like(q)
    for wall in WALLS:
        record = wall_junction_rows(
            q, zero, z, r, data["background"], wall,
        )
        A = np.sqrt(q[-1 if wall == "upper" else 0, :, 6])
        for name in COMPONENTS:
            expected = (
                record["orientation"]
                * record["components"][name]["robin_residual"] / (2 * A)
            )
            np.testing.assert_allclose(record["components"][name]["J"], expected)
        assert "Phi_robin" in record["separate_rows"]
        assert "chi_neumann" in record["separate_rows"]
        assert record["tensor_norms"]["mixed_eigenvalue_maximum_imaginary"] < 1e-12
        assert np.max(record["tensor_norms"]["frame_defect"]) < 1e-12


def test_radial_zones_are_disjoint_and_complete():
    r = np.linspace(0, 2, 25)
    zones = radial_zones(r, 7)
    joined = np.concatenate(list(zones.values()))
    assert len(np.unique(joined)) == len(r)
    np.testing.assert_array_equal(np.sort(joined), np.arange(len(r)))
    assert zones["axis"][-1] < zones["interior"][0]
    assert zones["interior"][-1] < zones["outer_corner"][0]


def test_cap_cubic_and_native_wall_routes_agree_on_cubic_control():
    data = manufactured_state()
    radius = 0.83
    cap = cap_spline_sphere_residual(
        data["position"], data["z"], data["r"], radius,
        data["background"], "upper",
    )
    native = wall_junction_rows(
        data["position"], data["velocity"], data["z"], data["r"],
        data["background"], "upper",
    )
    sampled = np.interp(
        radius, data["r"], native["components"]["sphere"]["mixed_coefficient"],
    )
    assert abs(cap["residual"]) < 1e-11
    assert abs(sampled) < 1e-11
    assert abs(cap["residual"] - sampled) < 1e-11


def test_off_gauge_state_is_flagged_separately():
    data = manufactured_state()
    q = data["position"].copy()
    q[0, :, 0] = 0.04
    record = wall_junction_rows(
        q, np.zeros_like(q), data["z"], data["r"], data["background"],
        "lower",
    )
    assert record["wall_adapted_gauge_maximum"] >= 0.04
    assert np.max(np.abs(record["components"]["sphere"]["J"])) < 1e-11


def test_raw_rows_match_independent_production_helper_on_adverse_state():
    data = manufactured_state()
    q = data["position"].copy()
    z = data["z"]
    r = data["r"]
    fraction = (z - z[0]) / (z[-1] - z[0])
    shape = (fraction**2 * (fraction - 1.0))[:, None]
    q[:, :, 2] += 0.013 * shape
    q[:, :, 3] -= 0.017 * shape
    q[:, :, 4] += 0.006 * shape
    q[:, :, 5] -= 0.004 * shape
    production = compact_wall_position_residuals(
        q, z, r, data["background"], radial_buffer=7,
    )
    mapping = {"tt": "g00", "rr": "radial", "sphere": "perp", "tr": "g0r"}
    retained = slice(None, -7)
    for wall_item in production["walls"]:
        wall = wall_item["wall"]
        direct = wall_junction_rows(
            q, np.zeros_like(q), z, r, data["background"], wall,
        )
        for component, production_name in mapping.items():
            expected = np.max(
                direct["components"][component]["robin_normalized"][retained]
            )
            assert np.isclose(
                expected, wall_item["rows"][production_name], rtol=0.0, atol=2e-14,
            )


def test_high_precision_physical_scale_ladder_is_accurate_and_nonmutating():
    data = manufactured_state()
    q = data["position"].copy()
    z = data["z"]
    r = data["r"]
    fraction = (z - z[0]) / (z[-1] - z[0])
    q[:, :, 3] += 0.02 * (fraction**2 * (fraction - 1.0))[:, None]
    zz, rr = np.meshgrid(z, r, indexing="ij")
    direction = np.zeros_like(q)
    direction[:, :, 2] = 3e-6 * (1 + zz + rr**2)
    direction[:, :, 3] = -2e-6 * (1 + zz**2)
    direction[:, :, 4] = 4e-7 * (1 + zz)
    direction[:, :, 5] = -3e-7 * (1 + rr**2)
    direction[:, :, 6] = 1e-6 * (1 + 0.1 * zz)
    direction[:, :, 7] = 2e-6 * (1 + 0.1 * rr**2)
    before = q.copy()
    result = high_precision_local_directional_ladder(
        q, direction, z, r, data["background"], decimal_digits=60,
    )
    np.testing.assert_array_equal(q, before)
    assert result["finite"]
    assert result["adjacent_accurate_pair"]
    assert result["second_order_pair_before_roundoff"]
    assert result["best_maximum_relative_scale_error"] < 1e-7
    assert result["core_vs_high_precision_analytic_maximum_relative"] < 1e-9


def test_wall_record_comparison_is_zoned_and_orthonormal():
    data = manufactured_state()
    left = wall_junction_rows(
        data["position"], data["velocity"], data["z"], data["r"],
        data["background"], "upper",
    )
    same = compare_wall_records(left, left, data["r"])
    assert set(same["zones"]) == {"axis", "interior", "outer_corner"}
    assert max(zone["DXJ"]["Linf"] for zone in same["zones"].values()) == 0.0
    perturbed = data["position"].copy()
    perturbed[-1, -1, 3] += 1e-6
    right = wall_junction_rows(
        perturbed, data["velocity"], data["z"], data["r"],
        data["background"], "upper",
    )
    difference = compare_wall_records(left, right, data["r"])
    assert difference["zones"]["outer_corner"]["J"]["Linf"] > 0.0
    assert difference["finite"]
