import numpy as np
import pytest

from bhps.high_order_invariant_interpolation import (
    barycentric5_interpolate,
    endpoint_preserving_indices,
    leave_level_out,
    manufactured_interpolation_controls,
    mapped_metric_fields,
    spline_interpolate,
)


def test_quintic_and_independent_reproduce_degree_five_polynomial():
    z = np.linspace(-0.3, 1.0, 17)
    r = np.linspace(0.0, 1.8, 21)
    field = z[:, None]**5 - 0.2 * z[:, None]**2 * r[None, :]**3 + r[None, :]**5
    Z, R = np.meshgrid(
        np.linspace(z[0], z[-1], 31), np.linspace(r[0], r[-1], 37), indexing="ij",
    )
    truth = Z**5 - 0.2 * Z**2 * R**3 + R**5
    assert np.max(np.abs(spline_interpolate(field, z, r, Z, R, 5) - truth)) < 1e-11
    assert np.max(np.abs(barycentric5_interpolate(field, z, r, Z, R) - truth)) < 1e-11


def test_tensor_components_and_boundary_panels():
    z = np.linspace(0.0, 1.0, 13)
    r = np.linspace(-1.0, 1.0, 15)
    scalar = z[:, None]**2 + r[None, :]**3
    field = np.stack((scalar, 2.0 * scalar), axis=-1)
    Z, R = np.meshgrid([z[0], 0.27, z[-1]], [r[0], 0.31, r[-1]], indexing="ij")
    value = barycentric5_interpolate(field, z, r, Z, R)
    truth = Z**2 + R**3
    assert value.shape == (3, 3, 2)
    assert np.max(np.abs(value[..., 0] - truth)) < 1e-12
    assert np.max(np.abs(value[..., 1] - 2.0 * truth)) < 2e-12


def test_flat_metric_maps_to_flat_physical_chart():
    z = np.linspace(0.0, 1.0, 13)
    r = np.linspace(0.0, 2.0, 17)
    metric = np.broadcast_to(np.eye(2), (len(z), len(r), 2, 2)).copy()
    sphere = np.ones((len(z), len(r)))
    D = np.linspace(0.1, 0.8, 11)
    S = np.linspace(0.2, 1.7, 15)
    native_z = 1.0 - np.broadcast_to(D[:, None], (len(D), len(S)))
    native_r = np.broadcast_to(S[None, :], (len(D), len(S)))
    mapped = mapped_metric_fields(metric, sphere, z, r, D, S, native_z, native_r)
    identity = np.broadcast_to(np.eye(2), mapped["covariant"].shape)
    assert np.max(np.abs(mapped["covariant"] - identity)) < 1e-11
    assert np.max(np.abs(mapped["native_areal_radius"] - native_r)) < 1e-11


def test_endpoint_preserving_leave_out_and_smooth_order():
    z = np.linspace(0.0, 1.0, 33)
    r = np.linspace(0.0, 1.5, 49)
    for stride in (2, 4):
        for offset in range(stride):
            indices = endpoint_preserving_indices(len(z), stride, offset)
            assert indices[0] == 0 and indices[-1] == len(z) - 1
            assert len(indices) >= 6
    field = np.exp(0.2 * z[:, None]) * np.cos(0.4 * r[None, :])
    result = leave_level_out(field, z, r)
    assert result["admissible"]
    assert result["orders"]["L2"] > 2.5
    assert result["orders"]["q95"] > 2.5


def test_manufactured_and_adverse_controls_pass():
    result = manufactured_interpolation_controls()
    assert result["passed"]
    assert result["smooth_relabeling_tensor_error"] < 1e-11
    assert set(result["adverse"]) == {
        "cusp", "near_nyquist", "one_cell_layer", "injected_checkerboard",
    }
    assert all(item["contained_or_rejected"] for item in result["adverse"].values())


@pytest.mark.parametrize("bad", ["short", "duplicate", "nonfinite", "outside"])
def test_invalid_inputs_rejected(bad):
    z = np.linspace(0.0, 1.0, 9)
    r = np.linspace(0.0, 1.0, 11)
    field = z[:, None] + r[None, :]
    target_z, target_r = np.asarray([[0.2]]), np.asarray([[0.3]])
    if bad == "short":
        z, field = z[:5], field[:5]
    elif bad == "duplicate":
        z = z.copy()
        z[3] = z[2]
    elif bad == "nonfinite":
        field = field.copy()
        field[2, 3] = np.nan
    else:
        target_z = np.asarray([[1.2]])
    with pytest.raises(ValueError):
        spline_interpolate(field, z, r, target_z, target_r, degree=5)
