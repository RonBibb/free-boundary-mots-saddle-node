import json

import numpy as np

import run_A790_matched_staged_continuum as runner

from bhps.matched_staged_continuum import (
    FIELD_COUNT,
    REDUCED_FIELD_ORDER,
    BOUNDARY_MODES,
    ProjectedJetField,
    axis_even_crossfit_audit,
    build_mode_neutral_case,
    second_wall_closure_audit,
)


def _flat_state(nz=9, nr=25):
    z = np.linspace(1.0, 2.0, nz)
    r = np.linspace(0.0, 1.0, nr)
    shape = (nz, nr, FIELD_COUNT)
    position = np.zeros(shape)
    position[:, :, 2] = -1.0
    position[:, :, 3] = 1.0
    position[:, :, 6] = 1.0
    velocity = np.zeros(shape)
    acceleration = np.zeros(shape)
    background = {
        "wall_stiffness": 0.0,
        "v0": 0.0,
        "v1": 0.0,
        "beta_a": 0.0,
        "beta_b": 0.0,
        "wall_potential_a": 0.0,
        "wall_potential_b": 0.0,
    }
    return position, velocity, acceleration, z, r, background


def _flat_mode_neutral_bundle():
    position, _, acceleration, z, r, background = _flat_state(nz=9, nr=31)
    jet = ProjectedJetField(
        z, r, position,
        np.zeros((3, *position.shape)),
        np.zeros((3, 3, *position.shape)),
    )
    return build_mode_neutral_case({
        "name": "phase-b-flat-structural-control",
        "z": z,
        "r": r,
        "jet_field": jet,
        "mass_squared": 0.0,
        "background": background,
    }, "phase-b-flat-structural-control")


def test_axis_crossfit_recovers_exact_even_cubic_jet_across_windows():
    position, _, acceleration, _, r, _ = _flat_state(nz=5, nr=81)
    radius_squared = r**2
    for z_index in range(position.shape[0]):
        for field in range(FIELD_COUNT):
            coefficients = np.array((
                0.01*(1+z_index+field),
                -0.02*(1+field),
                0.003*(1+z_index),
                -0.0004*(1+field),
            ))
            acceleration[z_index, :, field] = sum(
                coefficient*radius_squared**power
                for power, coefficient in enumerate(coefficients)
            )

    audit = axis_even_crossfit_audit(position, acceleration, r)

    assert audit["finite"]
    assert audit["polynomial_coordinate"] == "r^2"
    assert audit["degree"] == 3
    assert audit["maximum_scaled_spread"] < 2e-11
    assert tuple(audit["by_field_maximum_scaled_spread"]) == tuple(
        REDUCED_FIELD_ORDER
    )
    assert audit["point_count_minimum_by_width"] == sorted(
        audit["point_count_minimum_by_width"]
    )


def test_axis_crossfit_detects_unresolved_even_eighth_order_content():
    position, _, acceleration, _, r, _ = _flat_state(nz=4, nr=101)
    acceleration[:, :, 2] = 20.0*r[None, :]**8

    audit = axis_even_crossfit_audit(position, acceleration, r)

    assert audit["finite"]
    assert audit["maximum_scaled_spread"] > 1e-3
    assert audit["by_field_maximum_scaled_spread"]["h00"] > 1e-3
    for name, spread in audit["by_field_maximum_scaled_spread"].items():
        if name != "h00":
            assert spread == 0.0


def test_axis_crossfit_rejects_nonpositive_projected_radial_metric():
    position, _, acceleration, _, r, _ = _flat_state(nz=3, nr=51)
    position[1, 7, 3] = -1.0

    with np.testing.assert_raises_regex(ValueError, "radial metric is not positive"):
        axis_even_crossfit_audit(position, acceleration, r)


def test_second_wall_closure_is_exact_for_static_flat_state():
    position, velocity, acceleration, z, r, background = _flat_state()

    audit = second_wall_closure_audit(
        position, velocity, acceleration, z, r, background,
    )

    assert audit["finite"]
    assert audit["radial_buffer"] == 0
    assert audit["combined_normalized_RMS"] == 0.0
    assert audit["combined_normalized_Linf"] == 0.0
    for wall in ("lower", "upper"):
        assert audit["walls"][wall]["finite"]
        assert audit["walls"][wall]["combined_normalized_RMS"] == 0.0
        assert audit["walls"][wall]["combined_normalized_Linf"] == 0.0


def test_second_wall_closure_buffer_excludes_only_outer_radial_collar():
    position, velocity, acceleration, z, r, background = _flat_state(nr=31)
    outer_collar = np.zeros_like(r)
    outer_collar[-2:] = 1.0
    acceleration[:, :, 2] = z[:, None]*outer_collar[None, :]
    acceleration[:, :, 7] = z[:, None]*outer_collar[None, :]
    acceleration[:, :, 8] = z[:, None]*outer_collar[None, :]

    full = second_wall_closure_audit(
        position, velocity, acceleration, z, r, background,
    )
    retained = second_wall_closure_audit(
        position, velocity, acceleration, z, r, background, radial_buffer=2,
    )

    assert full["finite"] and retained["finite"]
    assert full["combined_normalized_Linf"] > 0.99
    assert full["combined_normalized_RMS"] > 0.0
    assert retained["radial_buffer"] == 2
    assert retained["combined_normalized_RMS"] == 0.0
    assert retained["combined_normalized_Linf"] == 0.0


def test_sequence_convergence_recovers_nonuniform_fourth_order():
    intervals = np.array((96.0, 112.0, 128.0))
    values = 0.25+3.0/intervals**4

    audit = runner._sequence_convergence(values, intervals, 3.0)

    assert audit["monotone_nonincreasing"]
    assert not audit["floor_resolved"]
    assert abs(audit["generalized_order"]-4.0) < 2e-7
    assert audit["minimum_order"] == 3.0
    assert audit["passes"]


def test_sequence_convergence_rejects_nonmonotone_values():
    audit = runner._sequence_convergence(
        (2.0e-4, 1.0e-4, 1.2e-4), (96, 112, 128), 1.5,
    )

    assert not audit["monotone_nonincreasing"]
    assert not audit["floor_resolved"]
    assert not audit["passes"]


def test_sequence_convergence_accepts_exact_and_near_floor_sequences():
    exact = runner._sequence_convergence(
        (0.125, 0.125, 0.125), (96, 112, 128), 8.0,
    )
    near_floor = runner._sequence_convergence(
        (0.125, 0.125-4.0e-13, 0.125-8.0e-13),
        (96, 112, 128), 8.0,
    )

    for audit in (exact, near_floor):
        assert audit["floor_resolved"]
        assert audit["passes"]
    assert exact["generalized_order"] is None


def test_expected_stage_signatures_encode_frozen_mode_order():
    owner = runner._expected_stage_signature("wall_owner_last_experimental")
    legacy = runner._expected_stage_signature("legacy_wall_axis_outer")

    assert owner == [
        (name, None) for name in runner.MANDATORY_LANDMARKS[
            "wall_owner_last_experimental"
        ]
    ]
    assert legacy[:2] == [
        ("bulk_positive_radius", None), ("initial_axis_fill", None),
    ]
    assert legacy[2:14] == [
        (name, iteration)
        for iteration in range(4)
        for name in (
            "normal_iteration_wall_endpoint_solve",
            "normal_iteration_post_wall_axis_fill",
            "normal_iteration_gzz_solve",
        )
    ]
    assert legacy[14:] == [
        ("final_compact_wall_endpoint_solve", None),
        ("final_compact_post_wall_axis_fill", None),
        ("pre_outer", None), ("post_outer", None),
        ("post_axis_operator_repair", None),
    ]


def test_scientific_npz_accepts_mixed_unicode_and_finite_numeric(tmp_path):
    path = tmp_path/"mixed-valid.npz"
    np.savez(
        path,
        protocol=np.asarray("sealed-protocol"),
        values=np.arange(6.0).reshape(2, 3),
        count=np.asarray(4),
    )

    record = runner.validate_scientific_npz(
        path, {"values": (2, 3)}, embedded={"protocol": "sealed-protocol"},
    )

    assert set(record["keys"]) == {"protocol", "values", "count"}
    assert record["byte_count"] > 0


def test_scientific_npz_rejects_nonfinite_numeric_in_mixed_archive(tmp_path):
    path = tmp_path/"mixed-nonfinite.npz"
    np.savez(
        path,
        protocol=np.asarray("sealed-protocol"),
        values=np.asarray((1.0, np.nan)),
    )

    with np.testing.assert_raises_regex(
        ValueError, "nonfinite numeric NPZ arrays.*values",
    ):
        runner.validate_scientific_npz(
            path, embedded={"protocol": "sealed-protocol"},
        )


def test_scientific_npz_rejects_embedded_identity_mismatch(tmp_path):
    path = tmp_path/"mixed-wrong-identity.npz"
    np.savez(
        path,
        protocol=np.asarray("different-protocol"), values=np.ones(2),
    )

    with np.testing.assert_raises_regex(
        ValueError, "embedded NPZ identity mismatch for protocol",
    ):
        runner.validate_scientific_npz(
            path, embedded={"protocol": "sealed-protocol"},
        )


def test_archive_stage_audit_requires_preservation_and_passing_gates(tmp_path):
    passing = {
        "staged_preservation": {"finite": True},
        "technical_audit": {"gates": {"all_technical_gates_pass": True}},
    }
    failing = {
        "staged_preservation": {"finite": True},
        "technical_audit": {"gates": {"all_technical_gates_pass": False}},
    }
    pass_path = tmp_path/"passing-stage-metadata.npz"
    fail_path = tmp_path/"failing-stage-metadata.npz"
    np.savez(
        pass_path,
        step_002_metadata_json=np.asarray(json.dumps(passing)),
        step_001_metadata_json=np.asarray(json.dumps(passing)),
    )
    np.savez(
        fail_path,
        step_001_metadata_json=np.asarray(json.dumps(passing)),
        step_002_metadata_json=np.asarray(json.dumps(failing)),
    )

    passed = runner._archive_stage_audit(pass_path)
    failed = runner._archive_stage_audit(fail_path)

    assert passed["record_count"] == 2
    assert passed["metadata_keys"] == [
        "step_001_metadata_json", "step_002_metadata_json",
    ]
    assert passed["all_staged_preservation_present"]
    assert passed["all_technical_audits_present"]
    assert passed["all_technical_gates_pass"]
    assert failed["all_staged_preservation_present"]
    assert failed["all_technical_audits_present"]
    assert not failed["all_technical_gates_pass"]


def test_technical_stage_audit_has_complete_inventory_for_both_modes():
    bundle = _flat_mode_neutral_bundle()
    expected_gates = {
        "all_values_and_diagnostics_finite",
        "lorentzian_signature_with_1e_8_margin",
        "mandatory_stage_order_and_full_shapes",
        "staged_causal_identity_at_most_1e_12",
        "velocity_hessian_stage_change_bitwise_zero",
        "outer_acceleration_residual_below_1e_10",
        "outer_source_residual_below_1e_10",
        "native_boundary_closure_valid",
        "owner_open_face_bitwise_unchanged",
        "owner_coupled_block_valid",
        "owner_full_normal_GH_below_1e_10",
        "required_raw_second_wall_rows_below_1e_10",
        "owner_reconciliation_valid",
        "native_axis_operator_valid",
        "all_technical_gates_pass",
    }

    for mode in BOUNDARY_MODES:
        _, record = runner.trace_observational_check(bundle, mode)
        audit = runner._technical_stage_audit(bundle, mode, record)

        assert set(audit["gates"]) == expected_gates
        assert all(isinstance(value, bool) for value in audit["gates"].values())
        assert audit["actual_stage_signature"] == audit["expected_stage_signature"]
        assert all(audit["shape_checks"].values())
        assert audit["gates"]["mandatory_stage_order_and_full_shapes"]
        assert audit["gates"]["all_values_and_diagnostics_finite"]
        assert audit["gates"]["required_raw_second_wall_rows_below_1e_10"]
        assert audit["axis_crossfit_observational_only"]["finite"]
        assert audit["axis_crossfit"] == audit["axis_crossfit_observational_only"]
        assert audit["axis_operator_repair"]["passes"]
