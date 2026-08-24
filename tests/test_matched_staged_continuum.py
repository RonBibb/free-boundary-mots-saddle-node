import numpy as np

from bhps.matched_staged_continuum import (
    BOUNDARY_MODES,
    TARGET_GRIDS,
    ContinuousPrimitiveParent,
    ContinuousReducedParent,
    DriverConfiguration,
    ProjectedJetField,
    TensorSplineSurface,
    build_mode_neutral_case,
    projected_geometry,
    reconstruct_anisotropic_fields,
    select_landmarks,
    validate_bundle_integrity,
)


class SyntheticJet:
    pass


def polynomial(z, s, factors):
    zz, ss = np.meshgrid(z, s, indexing="ij")
    base = 1.2 + 0.3*zz + 0.2*zz**2 + 0.4*ss + 0.1*zz*ss + 0.2*ss**2
    return base[:, :, None] * factors[None, None, :]


def test_clamped_tensor_cubic_reproduces_even_polynomial_and_derivatives():
    z = np.linspace(1.0, 2.0, 9)
    r = np.linspace(0.0, 12.0, 11)
    s = (r/12.0)**2
    factors = np.arange(1.0, 10.0)
    values = polynomial(z, s, factors)
    zz, ss = np.meshgrid(z, s, indexing="ij")
    z_first = (
        0.3 + 0.4*zz + 0.1*ss
    )[:, :, None] * factors[None, None, :]
    surface = TensorSplineSurface.build(
        z, r, values, z_first=z_first, degree=3,
    )
    target_z = np.linspace(1.0, 2.0, 13)
    target_r = np.linspace(0.0, 10.0, 15)
    target_s = (target_r/12.0)**2
    tz, ts = np.meshgrid(target_z, target_s, indexing="ij")
    expected = polynomial(target_z, target_s, factors)
    expected_z = (
        0.3 + 0.4*tz + 0.1*ts
    )[:, :, None] * factors[None, None, :]
    expected_s = (
        0.4 + 0.1*tz + 0.4*ts
    )[:, :, None] * factors[None, None, :]
    assert np.max(np.abs(surface.evaluate(target_z, target_r)-expected)) < 3e-13
    assert np.max(np.abs(
        surface.evaluate(target_z, target_r, z_order=1)-expected_z
    )) < 3e-12
    assert np.max(np.abs(
        surface.evaluate(target_z, target_r, s_order=1)-expected_s
    )) < 6e-12


def test_clamped_tensor_cubic_honors_non_natural_endpoint_data():
    z = np.linspace(1.0, 2.0, 9)
    r = np.linspace(0.0, 12.0, 10)
    zz, ss = np.meshgrid(z, (r/12.0)**2, indexing="ij")
    values = (np.sin(zz)+0.2*ss+0.1*zz*ss)[:, :, None]
    endpoint = np.zeros_like(values)
    endpoint[0, :, 0] = 0.37+0.11*ss[0]
    endpoint[-1, :, 0] = -0.29+0.07*ss[-1]
    surface = TensorSplineSurface.build(
        z, r, values, z_first=endpoint, degree=3,
    )
    assert np.max(np.abs(surface.evaluate(z, r)-values)) < 2e-15
    lower = surface.evaluate([z[0]], r, z_order=1)[0, :, 0]
    upper = surface.evaluate([z[-1]], r, z_order=1)[0, :, 0]
    assert np.max(np.abs(lower-endpoint[0, :, 0])) < 2e-13
    assert np.max(np.abs(upper-endpoint[-1, :, 0])) < 2e-13


def test_projected_spatial_jets_use_analytic_squared_radius_chain_rule():
    z = np.linspace(1.0, 2.0, 9)
    r = np.linspace(0.0, 12.0, 11)
    s = (r/12.0)**2
    factors = np.arange(1.0, 10.0)
    q = polynomial(z, s, factors)
    velocity = 0.05*q
    acceleration = -0.07*q
    zz, ss = np.meshgrid(z, s, indexing="ij")
    qz = (0.3+0.4*zz+0.1*ss)[:, :, None]*factors
    qs = (0.4+0.1*zz+0.4*ss)[:, :, None]*factors
    qzz = np.broadcast_to(0.4*factors, q.shape)
    qzs = np.broadcast_to(0.1*factors, q.shape)
    qss = np.broadcast_to(0.4*factors, q.shape)
    ds = 2.0*r/12.0**2
    d2s = 2.0/12.0**2
    jet = SyntheticJet()
    jet.z = z
    jet.r = r
    jet.reduced_fields = q
    jet.reduced_first = np.zeros((3, *q.shape))
    jet.reduced_second = np.zeros((3, 3, *q.shape))
    jet.reduced_first[0] = velocity
    jet.reduced_first[1] = qz
    jet.reduced_first[2] = qs*ds[None, :, None]
    jet.reduced_second[0, 0] = acceleration
    jet.reduced_second[0, 1] = jet.reduced_second[1, 0] = 0.05*qz
    jet.reduced_second[0, 2] = jet.reduced_second[2, 0] = (
        0.05*qs*ds[None, :, None]
    )
    jet.reduced_second[1, 1] = qzz
    jet.reduced_second[1, 2] = jet.reduced_second[2, 1] = qzs*ds[None, :, None]
    jet.reduced_second[2, 2] = qss*ds[None, :, None]**2 + qs*d2s
    parent = ContinuousReducedParent.from_jet_field(jet, z, r)
    target_z = np.linspace(1.0, 2.0, 12)
    target_r = np.linspace(0.0, 10.0, 14)
    found = parent.project(target_z, target_r)
    target_s = (target_r/12.0)**2
    tq = polynomial(target_z, target_s, factors)
    tz, ts = np.meshgrid(target_z, target_s, indexing="ij")
    tqs = (0.4+0.1*tz+0.4*ts)[:, :, None]*factors
    tds = 2.0*target_r/12.0**2
    expected_r = tqs*tds[None, :, None]
    expected_rr = (
        (0.4*factors)[None, None, :]*tds[None, :, None]**2
        + tqs*d2s
    )
    assert np.max(np.abs(found.reduced_fields-tq)) < 3e-13
    assert np.max(np.abs(found.reduced_first[2]-expected_r)) < 2e-13
    assert np.max(np.abs(found.reduced_second[2, 2]-expected_rr)) < 2e-13
    assert np.max(np.abs(found.reduced_first[2, :, 0])) == 0.0
    with np.testing.assert_raises(ValueError):
        found.reduced_fields.setflags(write=True)
    coefficient = parent.position.coefficients
    with np.testing.assert_raises(ValueError):
        coefficient.setflags(write=True)


def test_continuous_parent_binds_coordinates_and_serialized_coefficients_copy():
    z = np.linspace(1.0, 2.0, 8)
    r = np.linspace(0.0, 12.0, 9)
    shape = (len(z), len(r), 9)
    jet = SyntheticJet()
    jet.z = z.copy()
    jet.r = r.copy()
    jet.reduced_fields = np.ones(shape)
    jet.reduced_first = np.zeros((3, *shape))
    jet.reduced_second = np.zeros((3, 3, *shape))
    parent = ContinuousReducedParent.from_jet_field(
        jet, z, r, parent_identity="synthetic",
        expected_shape=(8, 9), require_full_radial_domain=True,
    )
    original = parent.position.coefficients.copy()
    archive = parent.coefficient_arrays()
    archive["position_coefficients"].fill(-123.0)
    assert np.array_equal(parent.position.coefficients, original)
    bad_z = z.copy()
    bad_z[-1] += 1e-3
    with np.testing.assert_raises(ValueError):
        ContinuousReducedParent.from_jet_field(jet, bad_z, r)


def test_primitive_parent_metadata_round_trip_and_binding_rejection():
    z = np.linspace(1.0, np.e, 145)
    r = np.linspace(0.0, 12.0, 325)
    shape = (len(z), len(r), 9)
    q = np.zeros(shape)
    q[:, :, 2] = -1.0
    q[:, :, 3] = 1.0
    q[:, :, 6] = 1.0
    first = np.zeros((3, *shape))
    second = np.zeros((3, 3, *shape))
    jet = ProjectedJetField(z, r, q, first, second)
    scalar_shape = shape[:2]
    geometry = {
        "name": "synthetic-canonical-P11",
        "z": z,
        "r": r,
        "jet_field": jet,
        "fold_amplitude": 7.90,
        "mass_squared": 0.0,
        "background": {},
        "psi": np.ones(scalar_shape),
        "a": np.zeros(scalar_shape),
        "b": np.zeros(scalar_shape),
        "c": np.zeros(scalar_shape),
        "phi": np.zeros(scalar_shape),
    }
    primary = ContinuousReducedParent.from_jet_field(
        jet, z, r, parent_identity=geometry["name"],
        expected_shape=(145, 325), require_full_radial_domain=True,
    )
    primitive = ContinuousPrimitiveParent.from_geometry(
        geometry, expected_shape=(145, 325),
        require_full_radial_domain=True,
    )
    archive = primitive.coefficient_arrays()
    restored = ContinuousPrimitiveParent.from_arrays(archive)
    assert restored.parent_identity == primitive.parent_identity
    assert (
        restored.source_coordinate_fingerprint
        == primitive.source_coordinate_fingerprint
    )
    assert restored.primitive_nodal_fingerprint == primitive.primitive_nodal_fingerprint
    assert restored.fingerprint() == primitive.fingerprint()
    for name, expected in primitive.project(z[::24], r[::41]).items():
        assert np.array_equal(restored.project(z[::24], r[::41])[name], expected)

    projected_geometry(
        geometry, primary, TARGET_GRIDS["G8"], restored,
    )
    mismatched_archive = {
        name: np.array(value, copy=True) for name, value in archive.items()
    }
    mismatched_archive["primitive_parent_identity"] = np.asarray(
        "different-parent"
    )
    mismatched = ContinuousPrimitiveParent.from_arrays(mismatched_archive)
    with np.testing.assert_raises_regex(ValueError, "not bound to this P11"):
        projected_geometry(
            geometry, primary, TARGET_GRIDS["G8"], mismatched,
        )


def _synthetic_mode_neutral_bundle():
    z = np.linspace(1.0, np.e, 9)
    r = np.linspace(0.0, 1.0, 31)
    shape = (len(z), len(r), 9)
    q = np.zeros(shape)
    q[:, :, 2] = -1.0
    q[:, :, 3] = 1.0
    q[:, :, 6] = 1.0
    jet = ProjectedJetField(
        z, r, q, np.zeros((3, *shape)), np.zeros((3, 3, *shape)),
    )
    background = {
        "wall_stiffness": 0.0,
        "v0": 0.0,
        "v1": 0.0,
        "beta_a": 0.0,
        "beta_b": 0.0,
        "wall_potential_a": 0.0,
        "wall_potential_b": 0.0,
    }
    return build_mode_neutral_case({
        "name": "synthetic-mode-neutral-case",
        "z": z,
        "r": r,
        "jet_field": jet,
        "mass_squared": 0.0,
        "background": background,
    }, "synthetic-mode-neutral-case")


def test_bundle_integrity_detects_each_live_rhs_outer_reference_mutation():
    bundle = _synthetic_mode_neutral_bundle()
    validate_bundle_integrity(bundle)
    references = {
        "outer_reference_position": bundle.outer_reference_position,
        "outer_reference_acceleration": bundle.outer_reference_acceleration,
    }
    for mode in BOUNDARY_MODES:
        rhs = bundle.rhs_by_mode[mode]
        for attribute, expected in references.items():
            live = getattr(rhs, attribute)
            live[0, 0, 0] += 1.0
            prefix = "outer_reference_"
            message = attribute[len(prefix):] if attribute.startswith(prefix) else attribute
            with np.testing.assert_raises_regex(RuntimeError, message):
                validate_bundle_integrity(bundle, mode)
            live[...] = expected
            validate_bundle_integrity(bundle, mode)


def test_driver_configuration_rejects_alternate_stencil_width():
    with np.testing.assert_raises_regex(ValueError, "configuration is frozen"):
        DriverConfiguration(stencil_width=5)


def test_reconstruct_anisotropic_fields_uses_lapse_as_psi_exactly():
    z = np.linspace(1.0, 2.0, 8)
    r = np.linspace(0.0, 3.0, 10)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    psi = 0.7 + 0.03*zz + 0.01*rr**2
    a = 0.02*zz + 0.01*rr**2
    c = -0.01*zz + 0.005*rr**2
    b = c + 0.04*rr**2
    compact = psi**2*np.exp(2*a)
    radial = psi**2*np.exp(2*b)
    transverse = psi**2*np.exp(2*c)
    quotient = np.empty_like(radial)
    quotient[:, 1:] = (radial[:, 1:]-transverse[:, 1:])/r[None, 1:]**2
    quotient[:, 0] = 0.08*transverse[:, 0]
    q = np.zeros((len(z), len(r), 9))
    q[:, :, 2] = -psi**2
    q[:, :, 3] = transverse
    q[:, :, 4] = quotient
    q[:, :, 6] = compact
    first = np.zeros((3, *q.shape))
    second = np.zeros((3, 3, *q.shape))
    projected = ProjectedJetField(z, r, q, first, second)
    found = reconstruct_anisotropic_fields(projected)
    assert np.max(np.abs(found["psi"]-psi)) < 2e-16
    assert np.max(np.abs(found["a"]-a)) < 3e-16
    assert np.max(np.abs(found["b"]-b)) < 4e-16
    assert np.max(np.abs(found["c"]-c)) < 3e-16


def test_landmark_selector_requires_unique_order_and_returns_copies():
    shape = (8, 9, 9)
    names = (
        "bulk_positive_radius", "initial_axis_fill",
        "normal_iteration_wall_endpoint_solve",
        "final_compact_wall_endpoint_solve",
        "final_compact_post_wall_axis_fill", "pre_outer", "post_outer",
        "post_axis_operator_repair",
    )
    stages = [
        {"name": name, "acceleration": np.full(shape, index, dtype=float)}
        for index, name in enumerate(names)
    ]
    selected = select_landmarks("legacy_wall_axis_outer", stages)
    assert tuple(selected) == (
        "bulk_positive_radius", "initial_axis_fill",
        "final_compact_wall_endpoint_solve",
        "final_compact_post_wall_axis_fill", "pre_outer", "post_outer",
        "post_axis_operator_repair",
    )
    stages[-2]["acceleration"].fill(-1.0)
    assert np.all(selected["post_outer"] == len(names)-2)
