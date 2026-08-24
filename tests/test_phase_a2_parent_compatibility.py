"""Focused prospective controls for the Protocol-123 P11 diagnostic.

These tests exercise construction helpers only.  They do not authorize or
run the scientific correction stage, target projection, an evolution RHS/RK
step, or the matched continuum matrix.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

import run_A790_phase_a2_parent_compatibility as runner
from bhps.nonlinear_regular_so3_evolution import (
    CompactWallCoupledAlgebraicGateError,
    apply_compact_wall_acceleration,
    compact_wall_normal_gauge_acceleration_residuals,
    impose_compact_wall_normal_tangential_acceleration,
    solve_compact_wall_coupled_phi_normal_acceleration,
)


def _owned_stage_sequence(nz=9, nr=6):
    """Return independent arrays satisfying the frozen substep ownership."""
    shape = (nz, nr, 9)
    bulk = (0.25 + np.arange(np.prod(shape), dtype=float)).reshape(shape)

    coupled = bulk.copy()
    coupled[[0, -1], :, 6] += 0.125
    coupled[[0, -1], :, 7] -= 0.25

    physical = coupled.copy()
    physical[[0, -1], :, 2] += 0.375
    physical[[0, -1], :, 3] -= 0.5
    # The physical rows do not own q4/q5 at the radial axis.  Those two
    # tensor-null values are left for the final selective reconciliation.
    physical[[0, -1], 1:, 4] += 0.625
    physical[[0, -1], 1:, 5] -= 0.75
    physical[[0, -1], :, 7] += 0.875
    physical[[0, -1], :, 8] -= 1.0

    gauge = physical.copy()
    gauge[[0, -1], :, 0:2] = 0.0

    final = gauge.copy()
    final[:, 0, 4] += 1.125
    final[:, 0, 5] -= 1.25
    return bulk, coupled, physical, gauge, final


def _flat_coupled_case(nz=9, nr=5):
    z = np.linspace(1.0, 2.0, nz)
    r = np.linspace(0.0, 1.0, nr)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    q = np.zeros((nz, nr, 9))
    v = np.zeros_like(q)
    a = np.zeros_like(q)
    q[:, :, 2] = -1.0
    q[:, :, 3] = 1.0
    q[:, :, 6] = 1.0
    a[:, :, 6] = (0.2 + 0.03 * zz + 0.01 * zz**2) * (1.0 + 0.1 * rr**2)
    a[:, :, 7] = (0.1 - 0.02 * zz + 0.015 * zz**3) * (1.0 + 0.2 * rr**2)
    source = np.zeros((nz, nr, 3))
    background = {
        "wall_stiffness": 0.0,
        "v0": 0.0,
        "v1": 0.0,
        "beta_a": 0.0,
        "beta_b": 0.0,
        "wall_potential_a": 0.0,
        "wall_potential_b": 0.0,
    }
    return z, r, q, v, a, source, background


def _physical_norm_case(delta):
    r = np.linspace(0.0, 1.0, 5)
    position = np.zeros((2, len(r), 9))
    position[:, :, 2] = -1.0
    position[:, :, 3] = 1.0
    position[:, :, 6] = 1.0
    bulk = np.zeros_like(position)
    compatible = bulk.copy()
    compatible[:, :, 2] = float(delta)
    return runner.correction_norms(position, bulk, compatible, r)


def _patch_sealed_authorization_inputs(monkeypatch, tmp_path):
    names = (
        ("PROTOCOL", "PROTOCOL_SHA256"),
        ("PARENT", "PARENT_SHA256"),
        ("PHASE_A", "PHASE_A_SHA256"),
        ("DECOMPOSITION", "DECOMPOSITION_SHA256"),
    )
    for index, (path_name, hash_name) in enumerate(names):
        path = tmp_path / f"sealed-{index}.dat"
        path.write_bytes(f"sealed-input-{index}".encode())
        monkeypatch.setattr(runner, path_name, path)
        monkeypatch.setattr(runner, hash_name, runner.sha256_file(path))
    validation = tmp_path / "validation.json"
    monkeypatch.setattr(runner, "VALIDATION", validation)
    return validation


def _correction_stage_parent(nz=9, nr=7):
    z, r, q, v, a, _, background = _flat_coupled_case(nz=nz, nr=nr)
    first = np.zeros((3, nz, nr, 9))
    first[0] = v
    second = np.zeros((3, 3, nz, nr, 9))
    second[0, 0] = a
    return {
        "z": z,
        "r": r,
        "q": q,
        "first": first,
        "second": second,
        "background": background,
    }


class _NoCacheIndex:
    def __init__(self, cached=None):
        self.cached = cached
        self.running = []
        self.completed = []
        self.failed = []

    def register(self, *args):
        return None

    def validated_path(self, stage_id):
        return self.cached

    def mark_running(self, stage_id):
        self.running.append(stage_id)

    def mark_complete(self, stage_id, path, elapsed, metadata):
        self.completed.append((stage_id, path, elapsed, metadata))

    def mark_failed(self, stage_id, failure):
        self.failed.append((stage_id, failure))


def test_support_audit_and_masks_accept_exact_frozen_stage_ownership():
    stages = _owned_stage_sequence()
    support = runner._support_audit(*stages)
    required = (
        "coupled_changes_only_wall_Phi_gzz",
        "physical_stage_changes_only_owned_wall_fields",
        "gauge_stage_changes_only_normal_tangential_wall_fields",
        "physical_stage_preserves_coupled_gzz_bitwise",
        "reconciliation_changes_only_q4_q5_axis",
        "final_changes_only_wall_or_q4_q5_axis",
        "stage_arrays_do_not_share_memory",
    )
    assert all(support[name] for name in required)

    masks = runner._support_masks(*stages)
    for actual, allowed in (
        ("mask_actual_coupled", "mask_allowed_coupled"),
        ("mask_actual_physical", "mask_allowed_physical"),
        ("mask_actual_gauge", "mask_allowed_gauge"),
        ("mask_actual_reconciliation", "mask_allowed_reconciliation"),
        ("mask_actual_final", "mask_allowed_final"),
    ):
        assert not np.any(masks[actual].astype(bool) & ~masks[allowed].astype(bool))
    np.testing.assert_array_equal(
        masks["mask_allowed_physical"][[0, -1], 0, 4:6], 0,
    )
    np.testing.assert_array_equal(
        masks["mask_allowed_physical"][[0, -1], 1:, 4:6], 1,
    )


def test_support_audit_rejects_wall_axis_and_open_parent_intrusions():
    bulk, coupled, physical, gauge, final = _owned_stage_sequence()

    wall_axis = physical.copy()
    wall_axis[0, 0, 4] += 1.0
    assert not runner._support_audit(
        bulk, coupled, wall_axis, gauge, final,
    )["physical_stage_changes_only_owned_wall_fields"]

    open_parent = physical.copy()
    open_parent[1, -1, 2] += 1.0
    assert not runner._support_audit(
        bulk, coupled, open_parent, gauge, final,
    )["physical_stage_changes_only_owned_wall_fields"]

    bad_reconciliation = final.copy()
    bad_reconciliation[1, 1, 4] += 1.0
    assert not runner._support_audit(
        bulk, coupled, physical, gauge, bad_reconciliation,
    )["reconciliation_changes_only_q4_q5_axis"]


def test_support_audit_treats_signed_zero_as_a_bitwise_gzz_change():
    bulk, coupled, physical, gauge, final = _owned_stage_sequence()
    coupled[0, 0, 6] = 0.0
    physical[0, 0, 6] = -0.0
    gauge[0, 0, 6] = -0.0
    final[0, 0, 6] = -0.0
    support = runner._support_audit(bulk, coupled, physical, gauge, final)
    assert not support["physical_stage_preserves_coupled_gzz_bitwise"]
    assert not support["physical_stage_changes_only_owned_wall_fields"]


def test_substep_boundary_copies_are_eight_distinct_byte_exact_snapshots():
    stages = list(_owned_stage_sequence())
    for index, stage in enumerate(stages):
        stage[2, index, index] = -0.0
    bulk, coupled, physical, gauge, compatible = stages
    copies = runner._substep_boundary_copies(
        bulk, coupled, physical, gauge, compatible,
    )
    expected = {
        "a_pre_coupled": bulk,
        "a_post_coupled": coupled,
        "a_pre_physical": coupled,
        "a_post_physical": physical,
        "a_pre_gauge": physical,
        "a_post_gauge": gauge,
        "a_pre_reconciliation": gauge,
        "a_post_reconciliation": compatible,
    }
    assert list(copies) == list(expected)
    assert len(copies) == 8

    def bits(value):
        contiguous = np.ascontiguousarray(value)
        return contiguous.view(np.uint64).reshape(contiguous.shape)

    for name, source in expected.items():
        np.testing.assert_array_equal(bits(copies[name]), bits(source))
        assert not np.shares_memory(copies[name], source)
        assert all(not np.shares_memory(copies[name], stage) for stage in stages)
    values = list(copies.values())
    assert all(
        not np.shares_memory(left, right)
        for index, left in enumerate(values)
        for right in values[index + 1:]
    )

    untouched = bits(copies["a_post_coupled"]).copy()
    copies["a_pre_coupled"][0, 0, 0] += 1.0
    np.testing.assert_array_equal(bits(copies["a_post_coupled"]), untouched)


def test_gauge_substep_writes_positive_zero_and_is_byte_isolated():
    rng = np.random.default_rng(731)
    before = rng.normal(size=(9, 6, 9))
    after = impose_compact_wall_normal_tangential_acceleration(before)

    assert not np.shares_memory(before, after)
    assert runner._positive_zero_wall_gauge(after)
    wall_bits = np.ascontiguousarray(after[[0, -1], :, 0:2]).view(np.uint64)
    np.testing.assert_array_equal(wall_bits, 0)

    allowed = np.zeros(before.shape, dtype=bool)
    allowed[[0, -1], :, 0:2] = True
    before_bits = np.ascontiguousarray(before).view(np.uint64).reshape(before.shape)
    after_bits = np.ascontiguousarray(after).view(np.uint64).reshape(after.shape)
    np.testing.assert_array_equal(after_bits[~allowed], before_bits[~allowed])
    assert np.all(after_bits[allowed] == 0)


def test_reduced_to_physical_acceleration_mapping_includes_radius_factors():
    r = np.asarray((0.0, 0.25, 1.0))
    reduced = np.empty((2, len(r), 9))
    for field in range(9):
        reduced[:, :, field] = float(field + 1)
    physical = runner._physical_acceleration(reduced, r)

    radius = r[None, :]
    expected = np.stack((
        reduced[:, :, 0],
        radius * reduced[:, :, 1],
        reduced[:, :, 2],
        reduced[:, :, 3],
        reduced[:, :, 3] + radius**2 * reduced[:, :, 4],
        radius * reduced[:, :, 5],
        reduced[:, :, 6],
        reduced[:, :, 7],
        reduced[:, :, 8],
    ), axis=-1)
    np.testing.assert_array_equal(physical, expected)
    np.testing.assert_array_equal(physical[:, 0, 4], reduced[:, 0, 3])
    np.testing.assert_array_equal(physical[:, 0, 5], 0.0)


def test_correction_norms_use_frozen_physical_and_proper_wall_gates():
    small = _physical_norm_case(0.009)
    assert small["global_normalized_Linf"] == pytest.approx(0.009)
    assert small["combined_proper_wall_weighted_RMS"] == pytest.approx(0.003)
    assert small["small_Linf_gate"]
    assert small["small_weighted_RMS_gate"]
    assert not small["order_one_failure"]
    assert small["proper_wall_measure"] == (
        "4*pi*r^2*g_sphere*sqrt(g_rr)*nodal_dr"
    )
    assert small["proper_wall_gate_equation"] == (
        "sqrt(sum_wall,r,component(w*e^2)/(9*sum_wall,r(w)))"
    )

    assert _physical_norm_case(0.05)["small_Linf_gate"]
    assert not _physical_norm_case(np.nextafter(0.05, np.inf))["small_Linf_gate"]
    assert not _physical_norm_case(0.5)["order_one_failure"]
    assert _physical_norm_case(np.nextafter(0.5, np.inf))["order_one_failure"]


def test_correction_norms_retain_axis_null_channel_derivative_images():
    r = np.linspace(0.0, 1.0, 5)
    position = np.zeros((2, len(r), 9))
    position[:, :, 2] = -1.0
    position[:, :, 3] = 1.0
    position[:, :, 6] = 1.0
    bulk = np.zeros_like(position)
    compatible = bulk.copy()
    compatible[:, 0, 4] = (0.2, -0.1)
    compatible[:, 0, 5] = (-0.3, 0.15)

    record = runner.correction_norms(position, bulk, compatible, r)
    assert record["global_normalized_Linf"] == 0.0
    assert record["raw_reduced_q4"]["absolute_Linf"] == pytest.approx(0.2)
    assert record["raw_reduced_q5"]["absolute_Linf"] == pytest.approx(0.3)
    images = record["axis_derivative_images"]
    np.testing.assert_allclose(
        images["2_delta_q4_for_drr_hrr_minus_hperp"]["values"],
        (0.4, -0.2),
    )
    np.testing.assert_allclose(
        images["delta_q5_for_dr_h0r"]["values"], (-0.3, 0.15),
    )


def test_correction_norms_use_full_parent_rms_and_unweighted_e2_r90():
    nz = 4
    r = np.linspace(0.0, 1.0, 5)
    position = np.zeros((nz, len(r), 9))
    position[:, :, 2] = -1.0
    position[:, :, 3] = 1.0
    position[:, :, 6] = 1.0
    bulk = np.zeros_like(position)
    compatible = bulk.copy()
    e_h00 = np.asarray((
        (0.05, 0.0, 0.0, 0.0, 0.10),
        (0.0, 0.0, 0.20, 0.0, 0.40),
        (0.0, 0.30, 0.0, 0.10, 0.0),
        (0.15, 0.0, 0.25, 0.0, 0.0),
    ))
    compatible[:, :, 2] = e_h00
    record = runner.correction_norms(position, bulk, compatible, r)

    def r90(density):
        cumulative = np.cumsum(np.asarray(density, dtype=float))
        if cumulative[-1] <= 0.0:
            return float(r[0])
        return float(r[np.searchsorted(cumulative, 0.9 * cumulative[-1])])

    true_parent_component_rms = float(np.sqrt(np.mean(e_h00**2)))
    wall_only_component_rms = float(np.sqrt(np.mean(e_h00[[0, -1]]**2)))
    assert true_parent_component_rms != pytest.approx(wall_only_component_rms)
    h00 = record["global_components"]["h_00"]
    assert h00["normalized_RMS"] == pytest.approx(true_parent_component_rms)
    assert record["global_normalized_RMS"] == pytest.approx(
        true_parent_component_rms / 3.0
    )

    for wall, index in (("lower", 0), ("upper", -1)):
        local_component_rms = float(np.sqrt(np.mean(e_h00[index]**2)))
        local = record["walls"][wall]
        assert local["combined_normalized_RMS"] == pytest.approx(
            local_component_rms / 3.0
        )
        expected_r90 = r90(e_h00[index]**2)
        assert local["unweighted_e_squared_r90"] == expected_r90
        assert (
            local["components"]["h_00"]["unweighted_e_squared_r90"]
            == expected_r90
        )

    global_density = np.sum(e_h00**2, axis=0)
    expected_global_r90 = r90(global_density)
    assert record["global_unweighted_e_squared_r90"] == expected_global_r90
    assert h00["unweighted_e_squared_r90"] == expected_global_r90


@pytest.mark.parametrize(
    ("structural", "linf", "rms", "expected"),
    (
        (True, 0.05, 0.01, "PASS-small-parent-acceleration"),
        (True, np.nextafter(0.05, np.inf), 0.01, "REVIEW-large-correction"),
        (True, 0.05, np.nextafter(0.01, np.inf), "REVIEW-large-correction"),
        (True, 0.5, 0.5, "REVIEW-large-correction"),
        (True, np.nextafter(0.5, np.inf), 0.0, "FAIL-parent-acceleration"),
        (False, 0.0, 0.0, "FAIL-parent-acceleration"),
        (True, np.nan, 0.0, "FAIL-parent-acceleration"),
        (True, 0.0, np.inf, "FAIL-parent-acceleration"),
    ),
)
def test_classification_uses_prospectively_frozen_boundaries(
    structural, linf, rms, expected,
):
    assert runner.classify_correction(structural, linf, rms) == expected


def test_endpoint_derivative_reproduces_degree_six_and_does_not_mutate():
    z = np.linspace(1.0, 2.0, 9)
    r = np.linspace(0.0, 1.0, 4)
    radial_field_factor = (
        1.0 + r[:, None] + 0.1 * np.arange(9, dtype=float)[None, :]
    )
    polynomial = z**6 - 2.0 * z**4 + 0.5 * z**2 - 3.0 * z + 1.0
    derivative = 6.0 * z**5 - 8.0 * z**3 + z - 3.0
    values = polynomial[:, None, None] * radial_field_factor[None, :, :]
    before = values.copy()

    found = runner._endpoint_derivative(values, z)
    expected = np.stack((
        derivative[0] * radial_field_factor,
        derivative[-1] * radial_field_factor,
    ))
    assert found.shape == (2, len(r), 9)
    np.testing.assert_allclose(found, expected, rtol=0.0, atol=2e-10)
    np.testing.assert_array_equal(values, before)


def test_row_implied_endpoint_derivative_matches_closed_synthetic_wall_rows():
    z, r, q, v, a, source, background = _flat_coupled_case(nr=9)
    coupled, coupled_record = solve_compact_wall_coupled_phi_normal_acceleration(
        q, v, a, source, source, source, z, r, background,
        capture_profiles=True,
    )
    assert coupled_record["passed"]
    normal_datum = np.stack((coupled[0, :, 6], coupled[-1, :, 6]))
    physical, _ = apply_compact_wall_acceleration(
        q, v, coupled, z, r, background, normal_datum, 7,
        fill_axis_after=False, impose_normal_tangential=False,
    )
    gauge = impose_compact_wall_normal_tangential_acceleration(physical)

    record = runner._row_implied_physical_endpoint_derivative(
        q, v, gauge, source, source, source, z, r, background,
    )
    endpoint_shape = (2, len(r), 9)
    for name in ("direct_reduced", "direct_physical", "row_implied_physical"):
        assert record[name].shape == endpoint_shape
        assert np.all(np.isfinite(record[name]))
    np.testing.assert_array_equal(
        record["row_defined_mask"],
        np.asarray((0, 0, 1, 1, 1, 1, 1, 1, 1), dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        record["row_implied_physical"][:, :, 0:2],
        record["direct_physical"][:, :, 0:2],
    )

    scored = record["row_defined_mask"].astype(bool)[None, None, :]
    denominator = np.maximum.reduce((
        np.ones(endpoint_shape),
        np.abs(record["direct_physical"]),
        np.abs(record["row_implied_physical"]),
    ))
    scaled = np.where(
        scored,
        np.abs(
            record["direct_physical"] - record["row_implied_physical"]
        ) / denominator,
        0.0,
    )
    assert record["maximum_scaled_difference"] == pytest.approx(
        np.max(scaled), rel=0.0, abs=0.0,
    )
    assert record["maximum_scaled_difference"] < 1e-10

    normal = record["normal_profiles"]
    assert np.isfinite(normal["maximum"])
    assert normal["maximum"] < 1e-10
    assert [wall["wall"] for wall in normal["walls"]] == ["lower", "upper"]
    for wall in normal["walls"]:
        profiles = wall["profiles"]
        assert np.asarray(profiles["terms"]).shape == (4, len(r))
        for name in ("terms", "residual", "scale", "normalized"):
            assert np.all(np.isfinite(profiles[name]))
        assert 0 <= profiles["maximum_index"] < len(r)
        assert profiles["maximum_radius"] == pytest.approx(
            r[profiles["maximum_index"]]
        )


def test_coupled_capture_profiles_have_complete_radial_records():
    z, r, q, v, a, source, background = _flat_coupled_case()
    solved, record = solve_compact_wall_coupled_phi_normal_acceleration(
        q, v, a, source, source, source, z, r, background,
        capture_profiles=True,
    )
    assert solved.shape == q.shape
    assert record["passed"]
    profiles = record["profiles"]
    expected_names = {
        "rank",
        "equilibrated_condition",
        "raw_condition",
        "pivot_strength",
        "normalized_linear_residual",
        "maximum_absolute_endpoint_correction",
    }
    assert set(profiles) == expected_names
    assert all(np.asarray(profile).shape == (len(r),) for profile in profiles.values())
    assert np.asarray(profiles["rank"]).dtype.kind in "iu"
    assert record["minimum_rank"] == int(np.min(profiles["rank"]))
    assert record["maximum_condition"] == pytest.approx(
        np.max(profiles["equilibrated_condition"]), rel=0.0, abs=0.0,
    )
    assert record["maximum_raw_condition"] == pytest.approx(
        np.max(profiles["raw_condition"]), rel=0.0, abs=0.0,
    )
    assert record["minimum_pivot_strength"] == pytest.approx(
        np.min(profiles["pivot_strength"]), rel=0.0, abs=0.0,
    )
    assert record["maximum_normalized_linear_residual"] == pytest.approx(
        np.max(profiles["normalized_linear_residual"]), rel=0.0, abs=0.0,
    )
    assert record["maximum_absolute_endpoint_correction"] == pytest.approx(
        np.max(profiles["maximum_absolute_endpoint_correction"]),
        rel=0.0, abs=0.0,
    )

    _, summary_only = solve_compact_wall_coupled_phi_normal_acceleration(
        q, v, a, source, source, source, z, r, background,
        capture_profiles=False,
    )
    assert "profiles" not in summary_only


def test_coupled_algebraic_rejection_has_structured_gate_diagnostics():
    z, r, q, v, a, source, background = _flat_coupled_case()
    with pytest.raises(CompactWallCoupledAlgebraicGateError) as caught:
        solve_compact_wall_coupled_phi_normal_acceleration(
            q, v, a, source, source, source, z, r, background,
            minimum_pivot_strength=0.5,
        )
    error = caught.value
    assert isinstance(error, RuntimeError)
    assert error.radial_index == 0
    assert error.gate == "rank_condition_pivot"
    assert error.diagnostics["rank"] == 4
    assert error.diagnostics["pivot_strength"] < 0.5
    assert error.diagnostics["minimum_allowed_pivot_strength"] == 0.5


def test_correction_stage_classifies_only_structured_coupled_gate(
    monkeypatch, tmp_path,
):
    parent = _correction_stage_parent()
    shape = parent["q"].shape[:2] + (3,)
    source_path = tmp_path / "synthetic-source.npz"
    zeros = np.zeros(shape)
    np.savez(
        source_path,
        source=zeros,
        source_time=zeros,
        source_second_time=zeros,
        memory=zeros,
    )
    gate_error = CompactWallCoupledAlgebraicGateError(
        "synthetic algebraic rejection",
        radial_index=2,
        gate="rank_condition_pivot",
        diagnostics={"rank": 3},
    )

    def reject(*args, **kwargs):
        raise gate_error

    classified = []

    def classify(parent_arg, source_arg, error_arg):
        classified.append((parent_arg, source_arg, error_arg))
        return {
            "classification": "FAIL-parent-acceleration",
            "failure": str(error_arg),
            "failure_stage": "coupled_Phi_gzz_algebraic_gate",
        }

    monkeypatch.setattr(
        runner, "solve_compact_wall_coupled_phi_normal_acceleration", reject,
    )
    monkeypatch.setattr(runner, "_write_coupled_gate_failure", classify)
    index = _NoCacheIndex()
    with pytest.raises(runner.PhaseA2ScientificGateFailure) as caught:
        runner._correction_stage(index, parent, source_path)
    assert caught.value.__cause__ is gate_error
    assert caught.value.result["classification"] == "FAIL-parent-acceleration"
    assert classified == [(parent, source_path, gate_error)]
    assert index.failed == []
    assert index.running == ["phase_a2/native_P11_correction"]
    assert len(index.completed) == 1
    stage_id, terminal_path, elapsed, terminal_metadata = index.completed[0]
    assert stage_id == "phase_a2/native_P11_correction"
    assert terminal_path == runner.RESULT
    assert elapsed >= 0.0
    assert terminal_metadata == {
        "classification": "FAIL-parent-acceleration",
        "terminal_scientific_result": True,
        "failure_stage": "coupled_Phi_gzz_algebraic_gate",
    }

    def crash(*args, **kwargs):
        raise RuntimeError("unexpected implementation failure")

    classified.clear()
    monkeypatch.setattr(
        runner, "solve_compact_wall_coupled_phi_normal_acceleration", crash,
    )
    crash_index = _NoCacheIndex()
    with pytest.raises(RuntimeError, match="unexpected implementation failure"):
        runner._correction_stage(crash_index, parent, source_path)
    assert classified == []
    assert crash_index.completed == []
    assert len(crash_index.failed) == 1
    assert crash_index.failed[0][0] == "phase_a2/native_P11_correction"
    assert "RuntimeError: unexpected implementation failure" in crash_index.failed[0][1]


def test_cached_terminal_scientific_result_reopens_without_coupled_solve(
    monkeypatch, tmp_path,
):
    parent = _correction_stage_parent()
    shape = parent["q"].shape[:2] + (3,)
    source_path = tmp_path / "synthetic-source.npz"
    zeros = np.zeros(shape)
    np.savez(
        source_path,
        source=zeros,
        source_time=zeros,
        source_second_time=zeros,
        memory=zeros,
    )
    cached_result = {
        "protocol_sha256": runner.PROTOCOL_SHA256,
        "classification": "FAIL-parent-acceleration",
        "failure_stage": "coupled_Phi_gzz_algebraic_gate",
        "failure": "cached structured scientific rejection",
        "source_artifact_sha256": runner.sha256_file(source_path),
    }
    cached_path = tmp_path / "cached-terminal-result.json"
    cached_path.write_text(json.dumps(cached_result))
    calls = []

    def forbidden_solver(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("cached terminal result invoked coupled solver")

    monkeypatch.setattr(
        runner,
        "solve_compact_wall_coupled_phi_normal_acceleration",
        forbidden_solver,
    )
    index = _NoCacheIndex(cached=cached_path)
    with pytest.raises(runner.PhaseA2ScientificGateFailure) as caught:
        runner._correction_stage(index, parent, source_path)
    assert caught.value.result == cached_result
    assert calls == []
    assert index.running == []
    assert index.completed == []
    assert index.failed == []


def test_run_scientific_catch_does_not_swallow_unexpected_runtime_error(
    monkeypatch,
):
    monkeypatch.setattr(runner, "validate_authorization", lambda: None)
    monkeypatch.setattr(runner, "recovery_index", lambda: object())
    monkeypatch.setattr(runner, "_load_parent", lambda: {"synthetic": True})
    monkeypatch.setattr(runner, "_source_stage", lambda index, parent: "source")

    def crash(index, parent, source):
        raise RuntimeError("unexpected implementation failure")

    monkeypatch.setattr(runner, "_correction_stage", crash)
    with pytest.raises(RuntimeError, match="unexpected implementation failure"):
        runner.run()


def test_normal_gauge_capture_profiles_are_unbuffered_and_self_consistent():
    z, r, q, v, a, source, background = _flat_coupled_case()
    solved, _ = solve_compact_wall_coupled_phi_normal_acceleration(
        q, v, a, source, source, source, z, r, background,
    )
    record = compact_wall_normal_gauge_acceleration_residuals(
        q, v, solved, source, source, source, z, r, background,
        radial_buffer=0, capture_profiles=True,
    )
    assert [wall["wall"] for wall in record["walls"]] == ["lower", "upper"]
    for wall in record["walls"]:
        profiles = wall["profiles"]
        terms = np.asarray(profiles["terms"])
        residual = np.asarray(profiles["residual"])
        scale = np.asarray(profiles["scale"])
        normalized = np.asarray(profiles["normalized"])
        assert terms.shape == (4, len(r))
        assert residual.shape == scale.shape == normalized.shape == (len(r),)
        np.testing.assert_allclose(residual, np.sum(terms, axis=0), atol=0.0)
        np.testing.assert_allclose(
            scale, np.maximum(1.0, np.sum(np.abs(terms), axis=0)), atol=0.0,
        )
        np.testing.assert_allclose(normalized, np.abs(residual) / scale, atol=0.0)
        maximum_index = int(np.argmax(normalized))
        assert profiles["maximum_index"] == maximum_index
        assert profiles["maximum_radius"] == pytest.approx(r[maximum_index])
        assert wall["maximum_normalized"] == pytest.approx(normalized[maximum_index])
        assert wall["maximum_absolute"] == pytest.approx(np.max(np.abs(residual)))
    assert record["maximum"] == pytest.approx(max(
        wall["maximum_normalized"] for wall in record["walls"]
    ))


def test_provenance_revalidation_fails_before_norms_or_classification():
    assert runner._require_provenance_revalidation({
        "parent": True, "source": True,
    }) is None
    with pytest.raises(
        RuntimeError,
        match="recovery/provenance revalidation failed: source, masks",
    ):
        runner._require_provenance_revalidation({
            "parent": True, "source": False, "masks": False,
        })

    tree = ast.parse(Path(runner.__file__).read_text())
    audit = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_audit_stage"
    )

    def call_line(name):
        lines = []
        for node in ast.walk(audit):
            if not isinstance(node, ast.Call):
                continue
            called = None
            if isinstance(node.func, ast.Name):
                called = node.func.id
            elif isinstance(node.func, ast.Attribute):
                called = node.func.attr
            if called == name:
                lines.append(node.lineno)
        assert lines, f"missing {name} call in _audit_stage"
        return min(lines)

    provenance_line = call_line("_require_provenance_revalidation")
    assert provenance_line < call_line("correction_norms")
    assert provenance_line < call_line("classify_correction")


def test_coupled_failure_record_names_source_driver_and_evolution_rhs_separately(
    monkeypatch,
):
    written = []
    monkeypatch.setattr(runner, "sha256_file", lambda path: "synthetic-sha256")
    monkeypatch.setattr(
        runner, "atomic_write_json",
        lambda path, payload: written.append((path, payload)),
    )
    parent = {
        "z": np.linspace(1.0, 2.0, 9),
        "r": np.linspace(0.0, 1.0, 7),
    }
    error = CompactWallCoupledAlgebraicGateError(
        "synthetic algebraic rejection",
        radial_index=3,
        gate="rank_condition_pivot",
        diagnostics={"rank": 3, "equilibrated_condition": 1e13},
    )
    result = runner._write_coupled_gate_failure(
        parent, Path("synthetic-source.npz"), error,
    )
    assert written == [(runner.RESULT, result)]
    assert result["classification"] == "FAIL-parent-acceleration"
    assert result["coupled_gate_failure"] == {
        "gate": "rank_condition_pivot",
        "radial_index": 3,
        "radius": 0.5,
        "diagnostics": {"rank": 3, "equilibrated_condition": 1e13},
    }
    assert result["evolution_RHS_or_RK_called"] is False
    assert result["source_driver_rhs_used_for_frozen_source_reconstruction"] is True
    assert "RHS_or_RK_called" not in result


def test_all_new_result_paths_are_scoped_under_recovery_root():
    paths = (
        runner.VALIDATION,
        runner.MANIFEST,
        runner.SOURCE_ARTIFACT,
        runner.CORRECTION_ARTIFACT,
        runner.RESULT,
    )
    assert all(path.parent == runner.RECOVERY_ROOT for path in paths)
    assert len(set(paths)) == len(paths)
    assert runner.RESULT.name == "result.json"


def test_validation_is_exact_scope_and_fails_closed(monkeypatch, tmp_path):
    validation = _patch_sealed_authorization_inputs(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError):
        runner.validate_authorization()

    authorization = {
        "native_P11_compatible_acceleration": "authorized",
        "target_projection": "not authorized",
        "RHS_or_RK": "not authorized",
        "full_matrix": "not authorized",
        "new_interface_physics": "not authorized",
    }
    candidate_paths = (
        Path(runner.__file__),
        Path("src/bhps/nonlinear_regular_so3_evolution.py"),
        Path("tests/test_phase_a2_parent_compatibility.py"),
    )
    valid_record = {
        "protocol_sha256": runner.PROTOCOL_SHA256,
        "authorization": authorization,
        "candidate_file_sha256": {
            str(path): runner.sha256_file(path) for path in candidate_paths
        },
        "immutable_input_mtime_ns": {
            str(path): path.stat().st_mtime_ns
            for path in (runner.PARENT, runner.PHASE_A, runner.DECOMPOSITION)
        },
    }
    validation.write_text(json.dumps(valid_record))
    assert runner.validate_authorization()["authorization"] == authorization

    bad = dict(authorization)
    bad["target_projection"] = "authorized"
    bad_record = {**valid_record, "authorization": bad}
    validation.write_text(json.dumps(bad_record))
    with pytest.raises(RuntimeError, match="authorization scope mismatch"):
        runner.validate_authorization()

    bad = dict(authorization)
    bad["extra_scope"] = "not authorized"
    bad_record = {**valid_record, "authorization": bad}
    validation.write_text(json.dumps(bad_record))
    with pytest.raises(RuntimeError, match="authorization scope mismatch"):
        runner.validate_authorization()

    validation.write_text(json.dumps({
        **valid_record, "protocol_sha256": "wrong-protocol",
    }))
    with pytest.raises(RuntimeError, match="does not name Protocol 123"):
        runner.validate_authorization()

    missing_candidate = {
        **valid_record,
        "candidate_file_sha256": dict(valid_record["candidate_file_sha256"]),
    }
    missing_candidate["candidate_file_sha256"].pop(str(candidate_paths[-1]))
    validation.write_text(json.dumps(missing_candidate))
    with pytest.raises(RuntimeError, match="candidate hash mismatch"):
        runner.validate_authorization()

    missing_mtime = {
        **valid_record,
        "immutable_input_mtime_ns": dict(valid_record["immutable_input_mtime_ns"]),
    }
    missing_mtime["immutable_input_mtime_ns"].pop(str(runner.PARENT))
    validation.write_text(json.dumps(missing_mtime))
    with pytest.raises(RuntimeError, match="immutable input mtime mismatch"):
        runner.validate_authorization()


def test_runner_structure_has_no_projection_outer_overwrite_or_evolution_rhs():
    tree = ast.parse(Path(runner.__file__).read_text())
    module_docstring = ast.get_docstring(tree)
    assert "source data" in module_docstring
    assert "evolution RHS call" in module_docstring
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    calls = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)

    prohibited = {
        "NativeRegularSO3RHS",
        "apply_outer_sommerfeld_acceleration",
        "build_mode_neutral_case",
        "project_parent",
        "run_phase_b",
    }
    assert prohibited.isdisjoint(imported)
    assert prohibited.isdisjoint(calls)

    run_source = ast.parse(Path(runner.__file__).read_text())
    run_node = next(
        node for node in run_source.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    run_calls = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(run_node)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert {"_source_stage", "_correction_stage", "_audit_stage"} <= run_calls
    assert prohibited.isdisjoint(run_calls)

    source_stage = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_source_stage"
    )
    source_stage_calls = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(source_stage)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert "source_driver_rhs" in source_stage_calls
    assert "NativeRegularSO3RHS" not in source_stage_calls
