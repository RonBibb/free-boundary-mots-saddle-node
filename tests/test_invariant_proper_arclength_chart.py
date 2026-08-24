import numpy as np

from bhps.invariant_physical_chart import build_normal_geodesic_chart
from bhps.invariant_proper_arclength_chart import (
    arclength_at_native_radius,
    brane_arclength,
    chart_validity,
    inverse_chart_at,
    native_to_coordinates,
    relabel_normal_chart,
)


def deformed_data(nonmonotone_areal=False):
    z = np.linspace(0.0, 1.0, 49)
    r = np.linspace(0.0, 2.0, 65)
    metric = np.zeros((len(z), len(r), 2, 2))
    metric[..., 0, 0] = (1.0 + 0.3 * z[:, None])**2
    metric[..., 1, 1] = (1.0 + 0.2 * r[None, :])**2
    physical_r = r + 0.1 * r**2
    if nonmonotone_areal:
        physical_r = r * (1.0 - 0.3 * np.exp(-((r - 1.0) / 0.25)**2))
    sphere_line = np.ones_like(r)
    sphere_line[1:] = (physical_r[1:] / r[1:])**2
    sphere = np.broadcast_to(sphere_line, (len(z), len(r))).copy()
    return z, r, metric, sphere


def test_analytic_brane_arclength_and_inverse():
    z, r, metric, sphere = deformed_data()
    distance = np.linspace(0.0, 0.7, 97)
    normal = build_normal_geodesic_chart(
        z, r, metric, sphere, distance, ray_count=129,
    )
    chart = relabel_normal_chart(normal, z, r, metric)
    expected = chart.native_brane_radius + 0.1 * chart.native_brane_radius**2
    assert np.max(np.abs(chart.arclength - expected)) < 2e-15
    assert chart_validity(chart)["valid"]
    target_D = np.linspace(0.05, 0.65, 31)
    target_S = np.linspace(0.05, 2.2, 47)
    native_z, native_r = inverse_chart_at(chart, target_D, target_S)
    D, S = native_to_coordinates(chart, native_z, native_r)
    assert np.max(np.abs(D - target_D[:, None])) < 2e-5
    assert np.max(np.abs(S - target_S[None, :])) < 2e-5
    assert abs(arclength_at_native_radius(chart, 1.5) - 1.725) < 2e-6


def test_nonmonotone_areal_radius_does_not_break_DS_chart():
    z, r, metric, sphere = deformed_data(nonmonotone_areal=True)
    normal = build_normal_geodesic_chart(
        z, r, metric, sphere, np.linspace(0.0, 0.7, 97), ray_count=129,
    )
    assert np.any(np.diff(normal.areal_radius[0]) < 0.0)
    chart = relabel_normal_chart(normal, z, r, metric)
    assert chart_validity(chart)["valid"]


def test_nonpositive_brane_metric_is_rejected():
    z, r, metric, _ = deformed_data()
    metric[-1, :, 1, 1] = -1.0
    try:
        brane_arclength(z, r, metric, np.linspace(0.0, 2.0, 33))
    except ValueError as error:
        assert "not positive" in str(error)
    else:
        raise AssertionError("nonpositive brane metric was admitted")
