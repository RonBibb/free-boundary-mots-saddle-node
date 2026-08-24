import numpy as np
import pytest

import bhps.joint_parent_bulk_reference as bulk_reference_module
from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.joint_parent_bulk_reference import (
    REFERENCE_CHANNEL_ORDER,
    SOURCE_CELL_MIDPOINT_SPECS,
    FiniteWallReferenceHermitePair,
    frozen_source_cell_midpoint_coordinates,
    source_cell_midpoint_coordinates,
)
from bhps.joint_parent_representation import (
    Protocol125RepresentationCoefficientFailure,
)


R_MAX = 12.0


def _reference_source(nz=13, nr=17):
    z = np.linspace(1.0, np.e, nz)
    r = np.linspace(0.0, R_MAX, nr)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    s = (rr/R_MAX)**2
    q = 0.1+0.03*zz**4+0.02*s**3+0.01*zz**2*s
    phi = 0.2-0.01*zz**3+0.015*s**2+0.005*zz*s**3
    return z, r, q, phi


def _analytic_mapping(z, r):
    zz, rr = np.meshgrid(z, r, indexing="ij")
    s = (rr/R_MAX)**2
    s_r = 2.0*rr/R_MAX**2
    s_rr = 2.0/R_MAX**2

    q_s = 0.06*s**2+0.01*zz**2
    q_ss = 0.12*s
    phi_s = 0.03*s+0.015*zz*s**2
    phi_ss = 0.03+0.03*zz*s
    return {
        "q": 0.1+0.03*zz**4+0.02*s**3+0.01*zz**2*s,
        "q_z": 0.12*zz**3+0.02*zz*s,
        "q_r": q_s*s_r,
        "q_zz": 0.36*zz**2+0.02*s,
        "q_zr": 0.02*zz*s_r,
        "q_rr": q_ss*s_r**2+q_s*s_rr,
        "phi": 0.2-0.01*zz**3+0.015*s**2+0.005*zz*s**3,
        "phi_z": -0.03*zz**2+0.005*s**3,
        "phi_r": phi_s*s_r,
        "phi_zz": -0.06*zz,
        "phi_zr": 0.015*s**2*s_r,
        "phi_rr": phi_ss*s_r**2+phi_s*s_rr,
    }


def test_q53_reference_jet_matches_manufactured_values_and_derivatives():
    z, r, q, phi = _reference_source()
    pair = FiniteWallReferenceHermitePair.build(z, r, q, phi)
    query_z = np.linspace(z[0], z[-1], 9)
    query_r = np.linspace(r[0], r[-1], 11)
    jet = pair.primary.evaluate(query_z, query_r)
    found = jet.as_derivative_mapping()
    expected = _analytic_mapping(query_z, query_r)
    assert tuple(jet.channel_order) == REFERENCE_CHANNEL_ORDER
    assert set(found) == set(expected)
    for name in expected:
        assert np.max(np.abs(found[name]-expected[name])) < 3e-11, name
        assert not found[name].flags.writeable
    assert np.array_equal(jet.second[0, 1], jet.second[1, 0])
    assert np.array_equal(
        found["q_r"][:, 0], np.zeros(len(query_z)),
    )
    assert np.array_equal(
        found["phi_r"][:, 0], np.zeros(len(query_z)),
    )


def test_pair_binds_exact_width_seven_source_endpoint_traces():
    z, r, q, phi = _reference_source()
    pair = FiniteWallReferenceHermitePair.build(z, r, q, phi)
    values = np.stack((q, phi), axis=-1)
    operator = derivative_matrix(z, 1, 7)
    expected = np.asarray(
        operator @ values.reshape(len(z), -1), dtype=float,
    ).reshape(values.shape)[[0, -1]]
    assert np.array_equal(pair.primary.endpoint_z_first, expected)
    assert np.array_equal(
        pair.primary.endpoint_z_first, pair.comparator.endpoint_z_first,
    )
    assert np.max(np.abs(
        pair.primary.evaluate(z, r).values-values
    )) < 2e-15
    found_endpoint = pair.primary.surface.evaluate(
        z[[0, -1]], r, z_order=1,
    )
    assert np.max(np.abs(found_endpoint-expected)) < 2e-13
    assert not pair.primary.source_values.flags.writeable
    assert not pair.primary.endpoint_z_first.flags.writeable


def test_q53_q33_are_identical_input_but_numerically_distinct():
    z = np.linspace(1.0, np.e, 15)
    r = np.linspace(0.0, R_MAX, 19)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    s = (rr/R_MAX)**2
    q = 0.08+0.01*zz**6+0.004*np.sin(1.7*zz)*s
    phi = 0.12+0.02*np.cos(1.3*zz)*(1.0+0.2*s**2)
    pair = FiniteWallReferenceHermitePair.build(z, r, q, phi)
    assert np.array_equal(
        pair.primary.source_values, pair.comparator.source_values,
    )
    assert np.array_equal(
        pair.primary.endpoint_z_first, pair.comparator.endpoint_z_first,
    )
    query_z = 0.5*(z[:-1]+z[1:])
    query_r = 0.5*(r[:-1]+r[1:])
    primary = pair.primary.evaluate(query_z, query_r)
    comparator = pair.comparator.evaluate(query_z, query_r)
    assert np.max(np.abs(primary.values-comparator.values)) > 1e-12
    assert np.max(np.abs(primary.second-comparator.second)) > 1e-10


def test_pair_persistence_fingerprint_and_reload_are_bitwise_stable():
    z, r, q, phi = _reference_source()
    pair = FiniteWallReferenceHermitePair.build(z, r, q, phi)
    fingerprint = pair.fingerprint()
    arrays = pair.coefficient_arrays()
    restored = FiniteWallReferenceHermitePair.from_arrays(arrays)
    assert restored.fingerprint() == fingerprint
    query_z = np.linspace(z[0], z[-1], 8)
    query_r = np.linspace(r[0], r[-1], 10)
    original_jet = pair.primary.evaluate(query_z, query_r)
    restored_jet = restored.primary.evaluate(query_z, query_r)
    assert np.array_equal(original_jet.values, restored_jet.values)
    assert np.array_equal(original_jet.first, restored_jet.first)
    assert np.array_equal(original_jet.second, restored_jet.second)
    q[:] = 999.0
    phi[:] = -999.0
    assert pair.fingerprint() == fingerprint


def test_reload_rejects_changed_trace_or_stencil_recipe():
    z, r, q, phi = _reference_source()
    clean = FiniteWallReferenceHermitePair.build(z, r, q, phi)
    changed_trace = {
        name: np.asarray(value).copy()
        for name, value in clean.coefficient_arrays().items()
    }
    changed_trace[
        "finite_wall_reference_pair_primary_endpoint_z_first"
    ][0, 2, 0] += 1e-8
    with np.testing.assert_raises_regex(ValueError, "endpoint traces changed"):
        FiniteWallReferenceHermitePair.from_arrays(changed_trace)

    changed_width = {
        name: np.asarray(value).copy()
        for name, value in clean.coefficient_arrays().items()
    }
    changed_width[
        "finite_wall_reference_pair_primary_stencil_width"
    ] = np.asarray(5)
    with np.testing.assert_raises_regex(ValueError, "width seven"):
        FiniteWallReferenceHermitePair.from_arrays(changed_width)


def test_builder_rejects_non_width_seven_or_wrong_field_shapes():
    z, r, q, phi = _reference_source()
    with np.testing.assert_raises_regex(ValueError, "width seven"):
        FiniteWallReferenceHermitePair.build(
            z, r, q, phi, stencil_width=5,
        )
    with np.testing.assert_raises_regex(ValueError, "wrong shape"):
        FiniteWallReferenceHermitePair.build(z, r, q[:, :-1], phi)


def test_frozen_direct_cell_center_coordinates_and_hashes():
    for label in ("N0", "N1"):
        coordinates = frozen_source_cell_midpoint_coordinates(label)
        specification = SOURCE_CELL_MIDPOINT_SPECS[label]
        assert (len(coordinates.z), len(coordinates.r)) == (
            specification["midpoint_shape"]
        )
        assert coordinates.coordinate_sha256 == (
            specification["midpoint_coordinate_sha256"]
        )
        nz, nr = specification["source_shape"]
        source_z = np.linspace(1.0, np.e, nz)
        source_r = np.linspace(0.0, R_MAX, nr)
        assert np.array_equal(
            coordinates.z, 0.5*(source_z[:-1]+source_z[1:]),
        )
        assert np.array_equal(
            coordinates.r, 0.5*(source_r[:-1]+source_r[1:]),
        )
        assert coordinates.z[0] > source_z[0]
        assert coordinates.z[-1] < source_z[-1]
        assert coordinates.r[0] > source_r[0]
        assert coordinates.r[-1] < source_r[-1]
        assert not coordinates.z.flags.writeable
        assert not coordinates.r.flags.writeable


def test_cell_center_builder_rejects_noncanonical_source_coordinates():
    z = np.linspace(1.0, np.e, 145)
    r = np.linspace(0.0, R_MAX, 325)
    z[71] += 1e-12
    with np.testing.assert_raises_regex(ValueError, "differ from frozen"):
        source_cell_midpoint_coordinates("N0", z, r)


def test_fresh_reference_nonfinite_coefficients_raise_typed_failure(monkeypatch):
    z, r, q, phi = _reference_source()
    original = bulk_reference_module.make_interp_spline
    calls = {"count": 0}

    def poisoned(*args, **kwargs):
        spline = original(*args, **kwargs)
        calls["count"] += 1
        if calls["count"] == 4:
            np.asarray(spline.c).reshape(-1)[0] = np.nan
        return spline

    monkeypatch.setattr(
        bulk_reference_module, "make_interp_spline", poisoned,
    )
    with pytest.raises(
        Protocol125RepresentationCoefficientFailure,
    ) as caught:
        FiniteWallReferenceHermitePair.build(z, r, q, phi)

    assert calls["count"] == 4
    evidence = caught.value.evidence
    assert evidence["recipe"] == "finite-wall-reference-Q53-compact"
    assert evidence["coefficient_shape"][-1] == len(REFERENCE_CHANNEL_ORDER)
    assert evidence["nonfinite_count"] == 1


def test_persisted_reference_nonfinite_coefficients_remain_invalid():
    z, r, q, phi = _reference_source()
    pair = FiniteWallReferenceHermitePair.build(z, r, q, phi)
    archive = {
        name: np.asarray(value).copy()
        for name, value in pair.coefficient_arrays().items()
    }
    archive[
        "finite_wall_reference_pair_primary_surface_coefficients"
    ][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="reference surface is nonfinite"):
        FiniteWallReferenceHermitePair.from_arrays(archive)
