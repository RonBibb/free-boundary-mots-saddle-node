import numpy as np

from bhps.invariant_physical_chart import (
    build_normal_geodesic_chart,
    chart_validity,
    common_areal_interval,
    conservative_order_interval,
    generalized_order,
    interpolate_regular_field,
    inverse_chart,
    inverse_chart_at,
    mapped_extrinsic_fields,
    mapped_metric_fields,
    native_to_proper_distance,
    sign_coherence,
    weighted_l2,
    weighted_quantile,
)


def flat_data(nz=33, nr=49):
    z = np.linspace(0.0, 1.0, nz)
    r = np.linspace(0.0, 2.0, nr)
    metric = np.zeros((nz, nr, 2, 2))
    metric[..., 0, 0] = 1.0
    metric[..., 1, 1] = 1.0
    sphere = np.ones((nz, nr))
    return z, r, metric, sphere


def test_flat_normal_chart_and_inverse_are_exact():
    z, r, metric, sphere = flat_data()
    distance = np.linspace(0.0, 0.8, 41)
    chart = build_normal_geodesic_chart(
        z, r, metric, sphere, distance, ray_count=65,
    )
    assert np.max(np.abs(chart.z - (1.0 - distance[:, None]))) < 2e-13
    assert np.max(np.abs(chart.r - chart.ray_label[None, :])) < 2e-13
    assert np.max(np.abs(chart.areal_radius - chart.r)) < 2e-13
    validity = chart_validity(chart)
    assert validity["valid"]
    target_radius = np.linspace(0.05, 1.8, 57)
    mapped_z, mapped_r = inverse_chart(chart, target_radius)
    assert np.max(np.abs(mapped_z - (1.0 - distance[:, None]))) < 2e-13
    assert np.max(np.abs(mapped_r - target_radius[None, :])) < 2e-13
    subset_distance = np.linspace(0.1, 0.7, 17)
    mapped_z, mapped_r = inverse_chart_at(chart, subset_distance, target_radius)
    assert np.max(np.abs(mapped_z - (1.0 - subset_distance[:, None]))) < 2e-13
    recovered = native_to_proper_distance(chart, mapped_z, mapped_r)
    assert np.max(np.abs(recovered - subset_distance[:, None])) < 2e-13


def test_deformed_native_coordinates_recover_physical_chart():
    z = np.linspace(0.0, 1.0, 49)
    r = np.linspace(0.0, 2.0, 65)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    dy_dz = 1.0 + 0.3 * z
    dR_dr = 1.0 + 0.2 * r
    physical_radius = r + 0.1 * r**2
    metric = np.zeros((len(z), len(r), 2, 2))
    metric[..., 0, 0] = dy_dz[:, None] ** 2
    metric[..., 1, 1] = dR_dr[None, :] ** 2
    sphere_line = np.ones_like(r)
    sphere_line[1:] = (physical_radius[1:] / r[1:]) ** 2
    sphere_line[0] = 1.0
    sphere = np.broadcast_to(sphere_line, zz.shape).copy()
    distance = np.linspace(0.0, 0.7, 71)
    chart = build_normal_geodesic_chart(
        z, r, metric, sphere, distance, ray_count=97,
    )
    assert chart_validity(chart)["valid"]
    low, high = common_areal_interval([chart], outer_limit=2.0)
    target_radius = np.linspace(max(low, 0.04), min(high, 1.9), 81)
    mapped_z, mapped_r = inverse_chart(chart, target_radius)
    fields = mapped_metric_fields(
        metric, sphere, z, r, distance, target_radius, mapped_z, mapped_r,
    )
    assert np.max(np.abs(fields["q_DD"][2:-2, 2:-2] - 1.0)) < 2e-4
    assert np.max(np.abs(fields["q_DR"][2:-2, 2:-2])) < 2e-5
    assert np.max(np.abs(fields["q_RR"][2:-2, 2:-2] - 1.0)) < 2e-4
    assert np.max(
        np.abs(fields["native_areal_radius"] - target_radius[None, :])
    ) < 2e-6


def test_regular_field_interpolation_respects_paired_points():
    z, r, _, _ = flat_data()
    zz, rr = np.meshgrid(z, r, indexing="ij")
    field = np.stack((zz + 2.0 * rr, zz * rr), axis=-1)
    target_z = np.linspace(0.1, 0.9, 21)[:, None] * np.ones((1, 25))
    target_r = np.ones((21, 1)) * np.linspace(0.1, 1.9, 25)[None, :]
    mapped = interpolate_regular_field(field, z, r, target_z, target_r)
    assert np.max(np.abs(mapped[..., 0] - (target_z + 2.0 * target_r))) < 1e-11
    assert np.max(np.abs(mapped[..., 1] - target_z * target_r)) < 1e-11


def test_flat_extrinsic_tensor_recovers_orthonormal_components():
    z, r, metric, sphere = flat_data()
    distance = np.linspace(0.0, 0.8, 41)
    radius = np.linspace(0.05, 1.8, 57)
    chart = build_normal_geodesic_chart(z, r, metric, sphere, distance, ray_count=65)
    mapped_z, mapped_r = inverse_chart(chart, radius)
    mapped_metric = mapped_metric_fields(
        metric, sphere, z, r, distance, radius, mapped_z, mapped_r,
    )
    extrinsic = np.zeros((*metric.shape[:2], 4, 4))
    for axis in range(4):
        extrinsic[..., axis, axis] = 0.3
    mapped = mapped_extrinsic_fields(
        extrinsic, sphere, z, r, mapped_z, mapped_r, mapped_metric,
    )
    for key in ("K_DD", "K_RR", "K_Omega"):
        assert np.max(np.abs(mapped[key] - 0.3)) < 2e-12
    assert np.max(np.abs(mapped["K_DR"])) < 2e-12
    assert np.max(np.abs(mapped["trace_K"] - 1.2)) < 2e-12
    assert np.max(np.abs(mapped["KijKij"] - 0.36)) < 2e-12


def test_orders_uncertainty_and_coherence_controls():
    counts = np.array([112.0, 128.0, 144.0])
    differences = counts[:-1] ** -2 - counts[1:] ** -2
    assert abs(generalized_order(*differences) - 2.0) < 1e-9
    interval = conservative_order_interval(
        differences[0], 1e-4 * differences[0],
        differences[1], 1e-4 * differences[1],
    )
    assert interval[0] > 1.99 and interval[1] < 2.01
    weight = np.ones((4, 5))
    coarse = np.ones((4, 5))
    fine = np.ones((4, 5))
    fine[0] = -1.0
    assert abs(sign_coherence(coarse, fine, weight) - 0.75) < 1e-14


def test_weighted_statistics_controls():
    distance = np.linspace(0.0, 1.0, 33)
    radius = np.linspace(0.0, 2.0, 49)
    field = np.ones((len(distance), len(radius), 3))
    weight = np.ones(field.shape[:2])
    assert abs(weighted_l2(field, weight, distance, radius) - np.sqrt(6.0)) < 1e-12
    values = np.arange(10.0)
    assert weighted_quantile(values, np.ones_like(values), 0.5) == 4.0


def test_nonpositive_metric_and_extrapolation_are_rejected():
    z, r, metric, sphere = flat_data()
    bad = metric.copy()
    bad[..., 0, 0] = -1.0
    try:
        build_normal_geodesic_chart(z, r, bad, sphere, np.linspace(0.0, 0.5, 9))
    except ValueError as error:
        assert "positive definite" in str(error)
    else:
        raise AssertionError("nonpositive metric was admitted")
    chart = build_normal_geodesic_chart(
        z, r, metric, sphere, np.linspace(0.0, 0.5, 9), ray_count=33,
    )
    try:
        inverse_chart(chart, np.linspace(0.1, 2.1, 9))
    except ValueError as error:
        assert "extrapolate" in str(error)
    else:
        raise AssertionError("inverse-map extrapolation was admitted")
