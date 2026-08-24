from __future__ import annotations

import numpy as np

from bhps.joint_parent_fields import (
    bulk_acceleration_from_completed_position,
    native_position_from_primitives,
    reconstruct_native_spatial_ansatz,
)
from bhps.gw_slice_high_order_solver import derivative_matrix


def _fields(nz=9, nr=17):
    z = np.linspace(1.0, 2.0, nz)
    r = np.linspace(0.0, 2.0, nr)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    psi = 0.8+0.01*zz+0.005*rr**2
    a = 0.01*np.sin(np.pi*(zz-1.0))*np.exp(-rr**2)
    radial_anisotropy = (
        0.003*rr**2*np.sin(np.pi*(zz-1.0))*np.exp(-rr**2)
    )
    b = (-a+2.0*radial_anisotropy)/3.0
    c = (-a-radial_anisotropy)/3.0
    alpha = psi*(1.0+0.02*(zz-1.0)**2)
    phi = 0.03*np.cos(np.pi*(zz-1.0))*np.exp(-0.2*rr**2)
    chi = 0.02*np.cos(np.pi*(zz-1.0))*np.exp(-0.5*rr**2)
    return z, r, alpha, psi, a, b, c, phi, chi


def test_native_reconstruction_keeps_lapse_independent_and_spatial_tracefree():
    z, r, alpha, psi, a, b, c, phi, chi = _fields()
    q = native_position_from_primitives(
        z, r, alpha, psi, a, b, c, phi, chi,
    )
    found = reconstruct_native_spatial_ansatz(q, r)
    np.testing.assert_allclose(found["alpha"], alpha, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(found["psi"], psi, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(found["a"], a, rtol=0.0, atol=3e-15)
    np.testing.assert_allclose(found["b"], b, rtol=0.0, atol=3e-15)
    np.testing.assert_allclose(found["c"], c, rtol=0.0, atol=3e-15)
    assert found["tracefree_maximum_absolute"] < 3e-16
    assert found["lapse_conformal_scaled_Linf"] > 1e-3


def _background(**updates):
    result = {
        "mass_squared": 0.5,
        "wall_stiffness": 0.0,
        "v0": 0.0,
        "v1": 0.0,
        "beta_a": 0.0,
        "beta_b": 0.0,
        "wall_potential_a": 0.0,
        "wall_potential_b": 0.0,
    }
    result.update(updates)
    return result


def test_bulk_acceleration_is_unclosed_gauge_seed_and_finite():
    z, r, alpha, psi, a, b, c, phi, chi = _fields()
    q = native_position_from_primitives(
        z, r, alpha, psi, a, b, c, phi, chi,
    )
    acceleration, record = bulk_acceleration_from_completed_position(
        q, z, r, _background(),
    )
    assert acceleration.shape == q.shape
    assert np.all(np.isfinite(acceleration))
    assert record["lapse_is_independent_of_spatial_conformal_factor"]
    assert record["spatial_reconstruction"]["lapse_conformal_scaled_Linf"] > 1e-3
    assert not record["compact_wall_completion_applied"]
    assert not record["lapse_seed"]["wall_completion_applied"]
    assert "lapse_completion" not in record
    trace = record["lapse_seed"]["spatial_metric_acceleration_trace"]
    np.testing.assert_allclose(
        record["lapse_seed"]["target_relative_lapse_acceleration"],
        0.5*trace,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        record["lapse_seed"]["lapse_acceleration"],
        0.5*alpha*trace,
        rtol=2e-15,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        acceleration[:, :, 2],
        -alpha**2*trace,
        rtol=2e-15,
        atol=2e-15,
    )
    gamma_0_time_derivative = (
        record["lapse_seed"]["lapse_acceleration"]/alpha-0.5*trace
    )
    assert np.max(np.abs(gamma_0_time_derivative)) < 2e-14


def test_bulk_acceleration_derives_native_chi_gradients_with_axis_parity():
    z, r, alpha, psi, a, b, c, phi, chi = _fields()
    q = native_position_from_primitives(
        z, r, alpha, psi, a, b, c, phi, chi,
    )
    _, record = bulk_acceleration_from_completed_position(
        q, z, r, _background(), stencil_width=7,
    )
    dz = derivative_matrix(z, 1, 7)
    dr = derivative_matrix(r, 1, 7)
    if hasattr(dz, "toarray"):
        dz = dz.toarray()
    if hasattr(dr, "toarray"):
        dr = dr.toarray()
    expected_z = dz @ chi
    expected_r = (dr @ chi.T).T
    expected_r[:, 0] = 0.0
    np.testing.assert_array_equal(
        record["native_chi_gradients"]["z"], expected_z,
    )
    np.testing.assert_array_equal(
        record["native_chi_gradients"]["r"], expected_r,
    )
    assert record["native_chi_gradients"]["axis_radial_positive_zero"]
    assert not np.any(np.signbit(record["native_chi_gradients"]["r"][:, 0]))


def test_bulk_seed_is_independent_of_compact_wall_parameters():
    z, r, alpha, psi, a, b, c, phi, chi = _fields()
    q = native_position_from_primitives(
        z, r, alpha, psi, a, b, c, phi, chi,
    )
    baseline, _ = bulk_acceleration_from_completed_position(
        q, z, r, _background(),
    )
    adverse_wall, _ = bulk_acceleration_from_completed_position(
        q,
        z,
        r,
        _background(
            wall_stiffness=13.0,
            v0=-7.0,
            v1=11.0,
            beta_a=5.0,
            beta_b=-3.0,
            wall_potential_a=17.0,
            wall_potential_b=-19.0,
        ),
    )
    np.testing.assert_array_equal(adverse_wall, baseline)


def test_bulk_seed_preserves_native_zero_channels_and_axis_quotients():
    z, r, alpha, psi, a, b, c, phi, chi = _fields()
    q = native_position_from_primitives(
        z, r, alpha, psi, a, b, c, phi, chi,
    )
    acceleration, record = bulk_acceleration_from_completed_position(
        q, z, r, _background(),
    )
    for channel in (0, 5):
        assert np.all(acceleration[:, :, channel] == 0.0)
        assert not np.any(np.signbit(acceleration[:, :, channel]))

    metric = record["spatial_metric_acceleration"]
    np.testing.assert_allclose(
        acceleration[:, 1:, 1]*r[None, 1:],
        metric["zr"][:, 1:],
        rtol=2e-15,
        atol=2e-15,
    )
    radial_difference = metric["radial"]-metric["transverse"]
    np.testing.assert_allclose(
        acceleration[:, 1:, 4]*r[None, 1:]**2,
        radial_difference[:, 1:],
        rtol=2e-15,
        atol=2e-15,
    )
    dr = derivative_matrix(r, 1, 7)
    if hasattr(dr, "toarray"):
        dr = dr.toarray()
    expected_q1_axis = (dr @ metric["zr"].T).T[:, 0]
    s = (r/r[-1])**2
    ds = derivative_matrix(s, 1, 7)
    if hasattr(ds, "toarray"):
        ds = ds.toarray()
    expected_q4_axis = (ds @ radial_difference.T).T[:, 0]/r[-1]**2
    np.testing.assert_array_equal(acceleration[:, 0, 1], expected_q1_axis)
    np.testing.assert_array_equal(acceleration[:, 0, 4], expected_q4_axis)
