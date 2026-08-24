from dataclasses import replace

import numpy as np

from bhps.ragged_normal_arclength_chart import (
    build_ragged_normal_chart,
    inverse_ragged_chart,
    ragged_chart_to_native,
    ragged_chart_validity,
)


def flat_chart(ray_count=17, distance_samples=21):
    z = np.linspace(0.0, 1.0, 25)
    r = np.linspace(0.0, 2.0, 33)
    metric = np.broadcast_to(np.eye(2), (len(z), len(r), 2, 2)).copy()
    sphere = np.ones((len(z), len(r)))
    return build_ragged_normal_chart(
        z, r, metric, sphere, launch_radius_max=1.5,
        ray_count=ray_count, distance_samples=distance_samples,
    )


def test_flat_ragged_chart_reaches_deep_boundary_and_is_valid():
    chart = flat_chart()
    assert chart.shape == (21, 17)
    assert np.max(np.abs(chart.maximum_distance - 1.0)) < 1e-11
    assert np.max(np.abs(chart.z - (1.0 - chart.distance))) < 1e-11
    assert np.max(np.abs(chart.r - chart.native_brane_radius[None, :])) < 1e-11
    validity = ragged_chart_validity(chart)
    assert validity["valid"]
    assert validity["nonlocal_collision_count"] == 0
    assert np.all(chart.termination_code == 1)


def test_ragged_inverse_roundtrip_and_unique_root():
    chart = flat_chart(ray_count=25, distance_samples=25)
    native_z = np.asarray([0.25, 0.40, 0.72])
    native_r = np.asarray([0.31, 0.80, 1.21])
    result = inverse_ragged_chart(chart, native_z, native_r)
    assert np.all(result.root_count == 1)
    assert np.max(np.abs(result.distance - (1.0 - native_z))) < 1e-11
    assert np.max(np.abs(result.arclength - native_r)) < 1e-11
    assert np.max(result.residual) < 1e-12
    D = np.asarray([0.75, 0.60, 0.28])
    S = native_r
    z2, r2 = ragged_chart_to_native(chart, D, S)
    assert np.max(np.abs(z2 - native_z)) < 1e-11
    assert np.max(np.abs(r2 - native_r)) < 1e-11


def test_crossing_orientation_is_rejected():
    chart = flat_chart()
    crossed = chart.r.copy()
    crossed[:, [7, 8]] = crossed[:, [8, 7]]
    bad = replace(chart, r=crossed)
    assert not ragged_chart_validity(bad)["orientation_valid"]


def test_nonpositive_metric_is_rejected():
    z = np.linspace(0.0, 1.0, 9)
    r = np.linspace(0.0, 2.0, 11)
    metric = np.broadcast_to(np.eye(2), (len(z), len(r), 2, 2)).copy()
    metric[..., 0, 0] = -1.0
    sphere = np.ones((len(z), len(r)))
    try:
        build_ragged_normal_chart(z, r, metric, sphere, launch_radius_max=1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("nonpositive metric was admitted")


def test_batched_flat_chart_matches_scalar_construction():
    scalar = flat_chart(ray_count=9, distance_samples=9)
    batched = flat_chart(ray_count=65, distance_samples=17)
    assert ragged_chart_validity(batched)["valid"]
    assert np.max(np.abs(batched.maximum_distance - 1.0)) < 1e-10
    assert np.max(np.abs(batched.z - (1.0 - batched.distance))) < 1e-10
    assert np.max(batched.endpoint_error_dop853) < 1e-12
    assert np.max(batched.endpoint_error_radau) < 1e-12
    assert np.max(np.abs(scalar.maximum_distance - 1.0)) < 1e-10
