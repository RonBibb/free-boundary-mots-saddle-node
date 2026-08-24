import math

import numpy as np

from bhps.junction_preservation_diagnostic import (
    COMPONENTS,
    WALLS,
    manufactured_state,
    wall_junction_rows,
    wall_source_coefficients,
)
from bhps.junction_second_preservation_diagnostic import (
    summarize_wall_second_tangent,
    wall_junction_second_tangent,
)


def _generic_dynamic_state():
    data = manufactured_state(nz=17, nr=25)
    q = data["position"].copy()
    z = data["z"]
    r = data["r"]
    zz, rr = np.meshgrid(z, r, indexing="ij")
    fraction = (z - z[0]) / (z[-1] - z[0])
    shape = (fraction**2 * (fraction - 1.0))[:, None]

    # Move off the compatible manifold while preserving a Lorentzian wall
    # metric.  Nonzero normal and scalar dynamics exercise every nonlinear
    # beta(Phi) and sqrt(g_zz) second-derivative term.
    q[:, :, 2] += 0.007 * shape * (1.0 + 0.03 * rr**2)
    q[:, :, 3] -= 0.006 * shape * (1.0 + 0.02 * rr**2)
    q[:, :, 4] += 0.002 * shape
    q[:, :, 5] -= 0.001 * shape
    q[:, :, 6] += 0.010 * shape * (1.0 + 0.01 * rr**2)
    q[:, :, 7] += 0.004 * shape * (1.0 + 0.02 * rr**2)
    q[:, :, 8] += 0.003 * shape * (1.0 + 0.01 * rr**2)

    v = np.zeros_like(q)
    v[:, :, 2] = 0.030 * (1.0 + zz + 0.02 * rr**2)
    v[:, :, 3] = -0.020 * (1.0 + 0.3 * zz**2 + 0.01 * rr**2)
    v[:, :, 4] = 0.004 * (1.0 + zz)
    v[:, :, 5] = -0.003 * (1.0 + 0.2 * zz)
    v[:, :, 6] = 0.012 * (1.0 + 0.1 * zz + 0.01 * rr**2)
    v[:, :, 7] = 0.015 * (1.0 + 0.1 * rr**2)
    v[:, :, 8] = 0.006 * (1.0 + 0.1 * zz + 0.02 * rr**2)

    a = np.zeros_like(q)
    a[:, :, 2] = -0.017 * (1.0 + 0.2 * zz + 0.01 * rr**2)
    a[:, :, 3] = 0.013 * (1.0 + 0.2 * zz**2)
    a[:, :, 4] = -0.003 * (1.0 + 0.1 * zz)
    a[:, :, 5] = 0.002 * (1.0 + 0.1 * rr**2)
    a[:, :, 6] = -0.008 * (1.0 + 0.05 * zz)
    a[:, :, 7] = 0.011 * (1.0 + 0.02 * rr**2)
    a[:, :, 8] = -0.004 * (1.0 + 0.1 * zz + 0.01 * rr**2)
    return data, q, v, a


def test_generic_second_tangent_matches_full_state_centered_ladder():
    data, q, v, a = _generic_dynamic_state()
    z = data["z"]
    r = data["r"]
    zero = np.zeros_like(q)
    epsilons = (3e-2, 1e-2, 3e-3)

    for wall in WALLS:
        record = wall_junction_second_tangent(
            q, v, a, z, r, data["background"], wall,
        )
        errors = []
        relative = []
        baseline = wall_junction_rows(
            q, zero, z, r, data["background"], wall,
        )["J_tensor"]
        for epsilon in epsilons:
            plus = q + epsilon * v + 0.5 * epsilon**2 * a
            minus = q - epsilon * v + 0.5 * epsilon**2 * a
            plus_j = wall_junction_rows(
                plus, zero, z, r, data["background"], wall,
            )["J_tensor"]
            minus_j = wall_junction_rows(
                minus, zero, z, r, data["background"], wall,
            )["J_tensor"]
            numerical = (plus_j - 2.0 * baseline + minus_j) / epsilon**2
            error = float(np.max(np.abs(
                numerical - record["DX2J_tensor"]
            )))
            scale = max(
                float(np.max(np.abs(numerical))),
                float(np.max(np.abs(record["DX2J_tensor"]))),
                1e-14,
            )
            errors.append(error)
            relative.append(error / scale)

        slope = math.log(errors[0] / errors[1]) / math.log(
            epsilons[0] / epsilons[1]
        )
        assert 1.8 < slope < 2.2
        assert min(relative) < 1e-6
        assert record["finite"]
        assert record["decomposition_maximum_absolute_defect"] < 1e-15
        assert record[
            "raw_vs_cancellation_exposed_maximum_absolute_defect"
        ] < 1e-14
        assert np.max(np.abs(record["D2J_velocity_velocity_tensor"])) > 1e-5


def test_acceleration_only_reduces_to_first_directional_derivative():
    data, q, _, a = _generic_dynamic_state()
    zero = np.zeros_like(q)
    for wall in WALLS:
        second = wall_junction_second_tangent(
            q, zero, a, data["z"], data["r"], data["background"], wall,
        )
        first_along_a = wall_junction_rows(
            q, a, data["z"], data["r"], data["background"], wall,
        )
        np.testing.assert_allclose(
            second["DX2J_tensor"], first_along_a["DXJ_tensor"],
            rtol=0.0, atol=2e-15,
        )
        np.testing.assert_array_equal(
            second["D2J_velocity_velocity_tensor"],
            np.zeros_like(second["D2J_velocity_velocity_tensor"]),
        )


def test_velocity_hessian_is_independent_of_supplied_acceleration():
    data, q, v, a = _generic_dynamic_state()
    for wall in WALLS:
        with_acceleration = wall_junction_second_tangent(
            q, v, a, data["z"], data["r"], data["background"], wall,
        )
        without_acceleration = wall_junction_second_tangent(
            q, v, np.zeros_like(a), data["z"], data["r"],
            data["background"], wall,
        )
        np.testing.assert_allclose(
            with_acceleration["D2J_velocity_velocity_tensor"],
            without_acceleration["D2J_velocity_velocity_tensor"],
            rtol=0.0, atol=5e-18,
        )


def test_scalar_second_rows_match_full_state_centered_difference():
    data, q, v, a = _generic_dynamic_state()
    zero = np.zeros_like(q)
    epsilon = 1e-2
    plus = q + epsilon * v + 0.5 * epsilon**2 * a
    minus = q - epsilon * v + 0.5 * epsilon**2 * a
    for wall in WALLS:
        second = wall_junction_second_tangent(
            q, v, a, data["z"], data["r"], data["background"], wall,
        )
        baseline = wall_junction_rows(
            q, zero, data["z"], data["r"], data["background"], wall,
        )["separate_rows"]
        plus_rows = wall_junction_rows(
            plus, zero, data["z"], data["r"], data["background"], wall,
        )["separate_rows"]
        minus_rows = wall_junction_rows(
            minus, zero, data["z"], data["r"], data["background"], wall,
        )["separate_rows"]
        for source_name, target_name in (
            ("Phi_robin", "DX2_Phi_robin"),
            ("chi_neumann", "DX2_chi_neumann"),
        ):
            numerical = (
                plus_rows[source_name]
                - 2.0 * baseline[source_name]
                + minus_rows[source_name]
            ) / epsilon**2
            np.testing.assert_allclose(
                second["separate_rows"][target_name], numerical,
                rtol=2e-5, atol=2e-9,
            )


def test_zone_summary_is_complete_orthonormal_and_finite():
    data, q, v, a = _generic_dynamic_state()
    for wall in WALLS:
        record = wall_junction_second_tangent(
            q, v, a, data["z"], data["r"], data["background"], wall,
        )
        summary = summarize_wall_second_tangent(record, data["r"])
        assert summary["finite"]
        assert summary["frame_defect_maximum"] < 1e-12
        assert set(summary["zones"]) == {"axis", "interior", "outer_corner"}
        for zone in summary["zones"].values():
            assert zone["proper_statistics"]["proper_RMS"] >= 0.0
            assert zone["proper_statistics"]["Linf"] >= 0.0
            assert (
                zone["DX2J_orthonormal_frobenius"]["maximum_absolute"]
                >= zone["DX2J_orthonormal_max_component"]["maximum_absolute"]
            )


def test_sqrt_gzz_terms_cancel_when_normal_derivative_is_zero():
    data = manufactured_state(nz=17, nr=25)
    z = data["z"]
    r = data["r"]
    q = np.zeros((len(z), len(r), 9))
    q[:, :, 2] = -1.10
    q[:, :, 3] = 1.00
    q[:, :, 4] = 0.01
    q[:, :, 5] = 0.02
    q[:, :, 6] = 1.20
    q[:, :, 7] = 0.11
    v = np.zeros_like(q)
    a = np.zeros_like(q)
    for field, value_t, value_tt in (
        (2, 0.04, -0.02), (3, -0.03, 0.01),
        (4, 0.005, -0.003), (5, -0.002, 0.001),
    ):
        v[:, :, field] = value_t
        a[:, :, field] = value_tt
    v[:, :, 6] = 0.20
    a[:, :, 6] = -0.10
    v[:, :, 7] = 0.03
    a[:, :, 7] = -0.02

    radius = r[None, :]
    fields = {
        "tt": q[:, :, 2],
        "rr": q[:, :, 3] + radius**2 * q[:, :, 4],
        "sphere": q[:, :, 3],
        "tr": radius * q[:, :, 5],
    }
    rates = {
        "tt": v[:, :, 2],
        "rr": v[:, :, 3] + radius**2 * v[:, :, 4],
        "sphere": v[:, :, 3],
        "tr": radius * v[:, :, 5],
    }
    accelerations = {
        "tt": a[:, :, 2],
        "rr": a[:, :, 3] + radius**2 * a[:, :, 4],
        "sphere": a[:, :, 3],
        "tr": radius * a[:, :, 5],
    }
    for wall in WALLS:
        index = -1 if wall == "upper" else 0
        record = wall_junction_second_tangent(
            q, v, a, z, r, data["background"], wall,
        )
        source = wall_source_coefficients(q[index, :, 7], data["background"], wall)
        beta_t = source["beta_phi"] * v[index, :, 7]
        beta_phiphi = (
            -data["background"]["wall_stiffness"] / 6.0
            if wall == "upper" else
            data["background"]["wall_stiffness"] / 6.0
        )
        beta_tt = (
            source["beta_phi"] * a[index, :, 7]
            + beta_phiphi * v[index, :, 7] ** 2
        )
        for name in COMPONENTS:
            expected = source["orientation"] * (
                beta_tt * fields[name][index]
                + 2.0 * beta_t * rates[name][index]
                + source["beta"] * accelerations[name][index]
            )
            np.testing.assert_allclose(
                record["components"][name]["DX2J"], expected,
                rtol=0.0, atol=2e-13,
            )


def test_compatible_metric_acceleration_keeps_second_tangent_zero():
    data = manufactured_state(nz=17, nr=25)
    q = data["position"]
    v = data["velocity"]
    multiplier = 0.04 * (1.0 + 0.02 * data["r"][None, :] ** 2)
    a = np.zeros_like(q)
    for field in (2, 3, 4, 5):
        a[:, :, field] = multiplier * q[:, :, field]
    for wall in WALLS:
        record = wall_junction_second_tangent(
            q, v, a, data["z"], data["r"], data["background"], wall,
        )
        assert np.max(np.abs(record["J_tensor"])) < 1e-11
        assert np.max(np.abs(record["DXJ_tensor"])) < 1e-11
        assert np.max(np.abs(record["DX2J_tensor"])) < 1e-11


def test_localized_acceleration_failure_is_detected_at_only_one_wall():
    data = manufactured_state(nz=17, nr=25)
    q = data["position"]
    v = data["velocity"]
    z = data["z"]
    fraction = (z - z[0]) / (z[-1] - z[0])
    upper_shape = fraction**2 * (fraction - 1.0)
    a = np.zeros_like(q)
    a[:, :, 3] = 0.035 * upper_shape[:, None] * (
        1.0 + 0.02 * data["r"][None, :] ** 2
    )
    lower = wall_junction_second_tangent(
        q, v, a, z, data["r"], data["background"], "lower",
    )
    upper = wall_junction_second_tangent(
        q, v, a, z, data["r"], data["background"], "upper",
    )
    assert np.max(np.abs(lower["DX2J_tensor"])) < 1e-11
    assert np.max(np.abs(upper["DX2J_tensor"])) > 1e-3
