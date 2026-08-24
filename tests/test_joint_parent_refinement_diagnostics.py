from __future__ import annotations

import numpy as np

from bhps.joint_parent_refinement_diagnostics import (
    AXIS_ACCELERATION_IMAGE_ORDER,
    DENSE_OUTER_SHA256,
    DENSE_WALL_SHA256,
    PHYSICAL_COMPONENT_ORDER,
    VALIDATION_MESH_SPECS,
    adjudicate_correction_refinement,
    axis_acceleration_derivative_image_profile,
    compare_axis_acceleration_image_refinement,
    compare_correction_refinement,
    correction_profile,
    frozen_validation_meshes,
    maximum_v2_proper_spacing,
    reduced_to_physical,
)
from bhps.matched_staged_continuum import hash_arrays


def _flat_position(nz, r):
    q = np.zeros((nz, len(r), 9))
    q[:, :, 2] = -1.0
    q[:, :, 3] = 1.0
    q[:, :, 6] = 1.0
    return q


def test_every_validation_coordinate_digest_is_frozen():
    meshes = frozen_validation_meshes()
    for name, (_, _, expected) in VALIDATION_MESH_SPECS.items():
        assert meshes[name]["sha256"] == expected
        assert hash_arrays(meshes[name]["z"], meshes[name]["r"]) == expected
    assert hash_arrays(meshes["dense_wall"]["r"]) == DENSE_WALL_SHA256
    assert hash_arrays(meshes["dense_outer"]["z"]) == DENSE_OUTER_SHA256


def test_reduced_to_physical_uses_radius_factors_and_anisotropy_numerator():
    r = np.asarray((0.0, 0.5, 1.0))
    q = np.zeros((2, len(r), 9))
    q[:, :, 1] = 2.0
    q[:, :, 3] = 3.0
    q[:, :, 4] = 4.0
    q[:, :, 5] = 5.0
    physical = reduced_to_physical(q, r)
    expected = np.broadcast_to(r, (2, len(r)))
    np.testing.assert_allclose(physical[:, :, 1], 2.0*expected)
    np.testing.assert_allclose(physical[:, :, 4], 3.0+4.0*expected**2)
    np.testing.assert_allclose(physical[:, :, 5], 5.0*expected)


def test_correction_profile_reports_signed_hzz_and_full_physical_gates():
    r = np.linspace(0.0, 12.0, 65)
    position = _flat_position(5, r)
    bulk = np.zeros_like(position)
    compatible = bulk.copy()
    compatible[[0, -1], :, 6] = 0.004
    profile = correction_profile(
        position, bulk, compatible, r, require_frozen_mesh=False,
    )
    assert profile["physical_component_order"] == PHYSICAL_COMPONENT_ORDER
    np.testing.assert_allclose(
        profile["signed_normalized_correction"][:, :, 6], 0.004,
    )
    assert np.isclose(profile["hzz_C"], 0.004)
    assert np.isclose(profile["hzz_W"], 0.004)
    assert profile["small_full_Linf_gate"]
    assert profile["small_full_RMS_gate"]
    for wall in ("lower", "upper"):
        assert np.isclose(
            profile["localization"][wall]["area_energy_fraction_sum"], 1.0,
        )


def test_refinement_comparison_requires_strict_decrease_above_floor():
    meshes = frozen_validation_meshes()
    r = meshes["dense_wall"]["r"]
    r_v2 = meshes["V2"]["r"]
    position = _flat_position(5, r)
    position_v2 = _flat_position(5, r_v2)
    bulk = np.zeros_like(position)
    coarse = bulk.copy()
    refined = bulk.copy()
    coarse[[0, -1], :, 6] = 0.004
    refined[[0, -1], :, 6] = 0.003
    n0 = correction_profile(position, bulk, coarse, r)
    n1 = correction_profile(position, bulk, refined, r)
    comparison = compare_correction_refinement(
        n0, n1, position_v2, position_v2, r_v2,
    )
    assert comparison["gates"]["C_decreases"]
    assert comparison["gates"]["W_decreases"]
    assert np.isclose(comparison["D_Linf"], 0.001)
    assert comparison["hzz_refinement_pass"]

    equal = compare_correction_refinement(
        n0, n0, position_v2, position_v2, r_v2,
    )
    assert not equal["gates"]["C_decreases"]
    assert not equal["gates"]["W_decreases"]


def test_v2_spacing_uses_mean_proper_geometry_of_both_parents():
    r = np.linspace(0.0, 12.0, 13)
    n0 = _flat_position(3, r)
    n1 = _flat_position(3, r)
    n1[:, :, 4] = 3.0/r[-1]**2
    spacing = maximum_v2_proper_spacing(
        n0, n1, r, require_frozen_mesh=False,
    )
    assert spacing.shape == (2,)
    assert np.all(spacing > np.diff(r).max())


def test_production_correction_and_spacing_apis_reject_substituted_meshes():
    r = np.linspace(0.0, 12.0, 65)
    position = _flat_position(3, r)
    acceleration = np.zeros_like(position)
    with np.testing.assert_raises_regex(ValueError, "1025-point"):
        correction_profile(position, acceleration, acceleration, r)
    with np.testing.assert_raises_regex(ValueError, "frozen V2"):
        maximum_v2_proper_spacing(position, position, r)


def _axis_profile(q4, q5):
    z = frozen_validation_meshes()["V2"]["z"]
    bulk = np.zeros((len(z), 9))
    compatible = np.zeros_like(bulk)
    compatible[:, 4] = q4
    compatible[:, 5] = q5
    return axis_acceleration_derivative_image_profile(
        bulk, compatible, z,
    )


def test_axis_acceleration_images_expose_q4_q5_corrections_hidden_by_area_norm():
    record = _axis_profile(0.002, -0.003)
    assert tuple(record["image_order"]) == AXIS_ACCELERATION_IMAGE_ORDER
    q4 = record["images"]["drr_hrr_minus_hperp_from_q4"]
    q5 = record["images"]["dr_h0r_from_q5"]
    np.testing.assert_array_equal(q4["compatible_image"], 0.004)
    np.testing.assert_array_equal(q5["compatible_image"], -0.003)
    np.testing.assert_array_equal(q4["signed_normalized_correction"], 0.004)
    np.testing.assert_array_equal(q5["signed_normalized_correction"], -0.003)
    assert q4["K"] == 0.004
    assert q5["K"] == 0.003
    assert record["all_small"]
    assert not record["any_order_one"]
    assert np.all(record["raw_q4_change"] == 0.002)
    assert np.all(record["raw_q5_change"] == -0.003)


def test_axis_image_size_order_one_and_refinement_gates_fail_independently():
    too_large = _axis_profile(0.03, 0.6)
    assert not too_large["all_small"]
    assert too_large["images"][
        "drr_hrr_minus_hperp_from_q4"
    ]["K"] == 0.06
    assert too_large["images"]["dr_h0r_from_q5"]["order_one_failure"]

    n0 = _axis_profile(0.002, 0.004)
    n1 = _axis_profile(0.0015, 0.003)
    good = compare_axis_acceleration_image_refinement(n0, n1)
    assert good["pass"]
    for record in good["images"].values():
        assert record["gates"]["strict_decrease"]
        assert record["gates"]["profile_difference"]
        assert record["gates"]["conservative_envelope"]

    equal = compare_axis_acceleration_image_refinement(n0, n0)
    assert not equal["pass"]
    assert all(
        not record["gates"]["strict_decrease"]
        for record in equal["images"].values()
    )

    z = frozen_validation_meshes()["V2"]["z"]
    with np.testing.assert_raises_regex(ValueError, "frozen V2 compact"):
        axis_acceleration_derivative_image_profile(
            np.zeros((len(z)-1, 9)),
            np.zeros((len(z)-1, 9)),
            z[:-1],
        )


def test_master_adjudicator_cannot_hide_non_hzz_or_jet_failure():
    meshes = frozen_validation_meshes()
    r = meshes["dense_wall"]["r"]
    r_v2 = meshes["V2"]["r"]
    position = _flat_position(3, r)
    position_v2 = _flat_position(3, r_v2)
    bulk = np.zeros_like(position)
    coarse = bulk.copy()
    refined = bulk.copy()
    coarse[[0, -1], :, 6] = 0.004
    refined[[0, -1], :, 6] = 0.003
    n0 = correction_profile(position, bulk, coarse, r)
    n1 = correction_profile(position, bulk, refined, r)
    axis_n0 = _axis_profile(0.002, 0.004)
    axis_n1 = _axis_profile(0.0015, 0.003)
    good = adjudicate_correction_refinement(
        n0, n1, position_v2, position_v2, r_v2,
        hzz_zz_n0=np.zeros(3), hzz_zz_n1=np.zeros(3),
        a_hzz_n0=np.zeros(3), a_hzz_n1=np.zeros(3),
        axis_image_n0=axis_n0, axis_image_n1=axis_n1,
    )
    assert good["pass"]

    n1_bad = dict(n1)
    n1_bad["small_full_Linf_gate"] = False
    hidden = adjudicate_correction_refinement(
        n0, n1_bad, position_v2, position_v2, r_v2,
        hzz_zz_n0=np.zeros(3), hzz_zz_n1=np.zeros(3),
        a_hzz_n0=np.zeros(3), a_hzz_n1=np.full(3, 0.003),
        axis_image_n0=axis_n0, axis_image_n1=axis_n0,
    )
    assert not hidden["pass"]
    assert not hidden["gates"]["N1_full_physical_Linf"]
    assert not hidden["gates"]["a_hzz_difference"]
    assert not hidden["gates"]["q4_q5_axis_acceleration_images"]
