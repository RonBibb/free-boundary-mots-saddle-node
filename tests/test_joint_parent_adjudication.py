from __future__ import annotations

import numpy as np

from bhps.joint_parent_adjudication import (
    PARENT_GATE_GROUPS,
    TWO_PARENT_GATE_GROUPS,
    classify_protocol125_gate_records,
    sampling_order_from_errors,
    score_q4_q5_derivative_images_on_v_meshes,
    score_group_arrays,
    score_precomputed_groups_on_v_meshes,
    score_sampling_order,
    score_source_triplet_arrays,
    score_state_pair_on_v_meshes,
)
from bhps.joint_parent_refinement_diagnostics import frozen_validation_meshes


class _TrigonometricState:
    def __init__(self, offset=0.0):
        self.offset = float(offset)

    def evaluate_coordinate_components(self, z, r, *, z_order=0, r_order=0):
        z = np.asarray(z, dtype=float)[:, None]
        r = np.asarray(r, dtype=float)[None, :]
        kz = 4.0
        kr = 1.1
        if (z_order, r_order) == (0, 0):
            scalar = np.sin(kz*z)+np.cos(kr*r)+self.offset
        elif (z_order, r_order) == (1, 0):
            scalar = kz*np.cos(kz*z)+np.zeros_like(r)
        elif (z_order, r_order) == (0, 1):
            scalar = -kr*np.sin(kr*r)+np.zeros_like(z)
        elif (z_order, r_order) == (2, 0):
            scalar = -(kz**2)*np.sin(kz*z)+np.zeros_like(r)
        elif (z_order, r_order) == (1, 1):
            scalar = np.zeros((z.shape[0], r.shape[1]))
        elif (z_order, r_order) == (0, 2):
            scalar = -(kr**2)*np.cos(kr*r)+np.zeros_like(z)
        else:
            raise ValueError("unsupported derivative")
        scales = np.linspace(0.02, 0.10, 9)
        return scalar[:, :, None]*scales[None, None, :]


class _ReducedImageState:
    def __init__(self, state_name, *, q4_offset=0.0, axis_only_offset=0.0):
        self.state_name = str(state_name)
        self.q4_offset = float(q4_offset)
        self.axis_only_offset = float(axis_only_offset)

    def evaluate_reduced(self, z, r, *, z_order=0, r_order=0):
        z = np.asarray(z, dtype=float)[:, None]
        r = np.asarray(r, dtype=float)[None, :]
        result = np.zeros((z.shape[0], r.shape[1], 9))
        if self.state_name == "acceleration":
            if (z_order, r_order) == (0, 0):
                result[:, :, 4] = 0.2+0.01*z+0.002*r**2
                result[:, :, 5] = -0.1+0.003*z+0.004*r**2
                result[:, 0, 4] += self.axis_only_offset
            return result
        if (z_order, r_order) == (0, 0):
            result[:, :, 4] = 0.2+0.01*z+0.002*r**2+self.q4_offset
            result[:, :, 5] = -0.1+0.003*z+0.004*r**2
        elif (z_order, r_order) == (1, 0):
            result[:, :, 4] = 0.01
            result[:, :, 5] = 0.003
        elif (z_order, r_order) == (0, 1):
            result[:, :, 4] = 0.004*r
            result[:, :, 5] = 0.008*r
        elif (z_order, r_order) == (0, 2):
            result[:, :, 4] = 0.004
            result[:, :, 5] = 0.008
        return result


def _v_meshes():
    found = frozen_validation_meshes()
    return {name: found[name] for name in ("V0", "V1", "V2")}


def _gate_record(passed=True, *, complete=True, provenance_valid=True):
    return {
        "complete": bool(complete),
        "provenance_valid": bool(provenance_valid),
        "passed": bool(passed),
    }


def _master_records():
    parents = {
        parent: {name: _gate_record() for name in PARENT_GATE_GROUPS}
        for parent in ("N0", "N1")
    }
    two_parent = {name: _gate_record() for name in TWO_PARENT_GATE_GROUPS}
    return parents, two_parent


def _freeze_record():
    return {
        "status": "FROZEN",
        "protocol_sha256": "a"*64,
        "adjudicator_sha256": "b"*64,
        "frozen_before_parent_data": True,
        "independent_review_passed": True,
        "scientific_candidates_absent_at_freeze": True,
    }


def _assert_constituents_recombine(pooled, constituents):
    records = constituents["records"] if "records" in constituents else constituents
    assert pooled["sample_count"] == sum(
        record["sample_count"] for record in records.values()
    )
    assert pooled["E_inf"] == max(record["E_inf"] for record in records.values())
    expected_rms = np.sqrt(sum(
        record["sample_count"]*record["E_RMS"]**2
        for record in records.values()
    )/pooled["sample_count"])
    assert np.isclose(pooled["E_RMS"], expected_rms)


def test_scaled_group_score_uses_elementwise_denominator_and_flat_rms():
    left = np.asarray((0.0, 4.0, -2.0))
    right = np.asarray((0.5, 2.0, 2.0))
    score = score_group_arrays(left, right, component_order=("x", "y", "z"))
    expected = np.asarray((0.5, 0.5, 2.0))
    np.testing.assert_allclose(score["elementwise_error"], expected)
    assert np.isclose(score["E_RMS"], np.sqrt(np.mean(expected**2)))
    assert score["E_inf"] == 2.0


def test_q53_q33_representation_scorer_gates_every_v_mesh_and_group():
    passed = score_state_pair_on_v_meshes(
        _TrigonometricState(),
        _TrigonometricState(offset=0.5e-8),
        _v_meshes(),
        comparison_kind="Q53_Q33",
    )
    assert passed["passed"]
    assert len(passed["gates"]) == 9
    position_constituents = passed["records"]["V2"]["groups"]["position"][
        "constituents"
    ]
    assert position_constituents["direction_order"] == ("value",)
    assert len(position_constituents["records"]) == 9
    assert position_constituents["records"]["h_z0:value"]["E_inf"] > 0.0
    _assert_constituents_recombine(
        passed["records"]["V2"]["groups"]["position"],
        position_constituents,
    )
    first_constituents = passed["records"]["V2"]["groups"]["first_spatial"][
        "constituents"
    ]
    assert first_constituents["direction_order"] == ("z", "r")
    assert len(first_constituents["records"]) == 18

    failed = score_state_pair_on_v_meshes(
        _TrigonometricState(),
        _TrigonometricState(offset=2e-7),
        _v_meshes(),
        comparison_kind="Q53_Q33",
    )
    assert not failed["passed"]
    assert not failed["gates"]["V2_position"]
    assert failed["gates"]["V2_first_spatial"]


def test_source_triplet_is_concatenated_in_h_ht_htt_order():
    zeros = np.zeros((4, 5, 3))
    left = {
        "source": zeros,
        "source_time": zeros,
        "source_second_time": zeros,
    }
    right = {name: value.copy() for name, value in left.items()}
    right["source_second_time"][..., 2] = 2e-7
    score = score_source_triplet_arrays(left, right, comparison_kind="Q53_Q33")
    assert score["triplet_order"] == (
        "source", "source_time", "source_second_time",
    )
    assert np.isclose(score["E_inf"], 2e-7)
    assert score["constituent_order"][:3] == (
        "source:H_z", "source:H_0", "source:H_r/r",
    )
    assert np.isclose(
        score["constituents"]["source_second_time:H_r/r"]["E_inf"],
        2e-7,
    )
    assert score["constituents"]["source:H_r/r"]["E_inf"] == 0.0
    _assert_constituents_recombine(score, score["constituents"])
    assert not score["passed"]


def test_source_triplet_elementwise_archive_is_triplet_then_component_ordered():
    left = {
        name: np.zeros((1, 1, 3))
        for name in ("source", "source_time", "source_second_time")
    }
    right = {name: value.copy() for name, value in left.items()}
    right["source"][0, 0] = np.asarray((1.0, 2.0, 3.0))*1e-9
    right["source_time"][0, 0] = np.asarray((4.0, 5.0, 6.0))*1e-9
    right["source_second_time"][0, 0] = np.asarray((7.0, 8.0, 9.0))*1e-9
    score = score_source_triplet_arrays(left, right, comparison_kind="Q53_Q33")
    expected = np.arange(1.0, 10.0).reshape(3, 3)*1e-9
    np.testing.assert_allclose(score["elementwise_error"][:, :, 0, 0], expected)
    assert tuple(score["constituents"]) == score["constituent_order"]


def test_sealed_legacy_holdout_scores_explicit_original_grouping():
    meshes = _v_meshes()
    left = {}
    right = {}
    for name, mesh in meshes.items():
        shape = (len(mesh["z"]), len(mesh["r"]), 9)
        left[name] = {
            "position": np.zeros(shape),
            "first_spatial": np.zeros(shape+(2,)),
            "second_spatial": np.zeros(shape+(3,)),
        }
        right[name] = {group: values.copy() for group, values in left[name].items()}
    passed = score_precomputed_groups_on_v_meshes(
        left,
        right,
        meshes,
        comparison_kind="legacy_Q33_Q55",
        required_groups=("position", "first_spatial", "second_spatial"),
    )
    assert passed["passed"]
    right["V1"]["first_spatial"][0, 0, 0, 0] = 2e-7
    failed = score_precomputed_groups_on_v_meshes(
        left,
        right,
        meshes,
        comparison_kind="legacy_Q33_Q55",
        required_groups=("position", "first_spatial", "second_spatial"),
    )
    assert not failed["passed"]
    assert not failed["gates"]["V1_first_spatial"]


def test_sampling_order_rule_handles_floor_pass_order_pass_and_failure():
    floor = sampling_order_from_errors((8e-13, 7e-13, 6e-13))
    assert floor["classification"] == "sampling-floor-resolved"
    assert floor["passed"]

    counts = np.asarray((104.0, 128.0, 152.0))
    fourth = tuple((1.0/counts)**4)
    resolved = sampling_order_from_errors(fourth)
    assert resolved["passed"]
    assert np.isclose(resolved["order"], 4.0, rtol=1e-8)

    failed = sampling_order_from_errors((1e-4, 2e-4, 1e-5))
    assert not failed["passed"]


def test_sampling_order_scorer_uses_frozen_meshes_and_fd7():
    score = score_sampling_order(_TrigonometricState())
    assert score["passed"]
    assert score["gates"]["V2_second_spatial_RMS"]
    assert tuple(score["records"]) == ("V0", "V1", "V2")
    for record in score["records"].values():
        first = record["first_spatial_constituents"]
        second = record["second_spatial_constituents"]
        assert first["direction_order"] == ("z", "r")
        assert second["direction_order"] == ("zz", "zr", "rr")
        assert len(first["records"]) == 18
        assert len(second["records"]) == 27
        assert np.isfinite(first["records"]["h_z0:z"]["E_RMS"])
        assert np.isfinite(second["records"]["chi:rr"]["E_inf"])
        _assert_constituents_recombine(
            {
                "sample_count": record["first_spatial_sample_count"],
                "E_inf": record["first_spatial_Linf"],
                "E_RMS": record["first_spatial_RMS"],
            },
            first,
        )
        _assert_constituents_recombine(
            {
                "sample_count": record["second_spatial_sample_count"],
                "E_inf": record["second_spatial_Linf"],
                "E_RMS": record["second_spatial_RMS"],
            },
            second,
        )


def test_q4_q5_position_images_apply_exact_product_rules_and_gate_second_image():
    passed = score_q4_q5_derivative_images_on_v_meshes(
        _ReducedImageState("position"),
        _ReducedImageState("position"),
        _v_meshes(),
        comparison_kind="Q53_Q33",
        state_name="position",
    )
    assert passed["passed"]
    assert passed["product_rule_images_used"]
    assert passed["axis_images_explicit"]
    second = passed["records"]["V2"]["groups"]["second_spatial"]
    assert len(second["constituents"]["records"]) == 6
    _assert_constituents_recombine(second, second["constituents"])

    failed = score_q4_q5_derivative_images_on_v_meshes(
        _ReducedImageState("position"),
        _ReducedImageState("position", q4_offset=6e-6),
        _v_meshes(),
        comparison_kind="Q53_Q33",
        state_name="position",
    )
    assert not failed["passed"]
    assert not failed["gates"]["V2_second_spatial"]


def test_q4_q5_acceleration_axis_images_cannot_hide_in_coordinate_zeros():
    failed = score_q4_q5_derivative_images_on_v_meshes(
        _ReducedImageState("acceleration"),
        _ReducedImageState("acceleration", axis_only_offset=6e-6),
        _v_meshes(),
        comparison_kind="Q53_Q33",
        state_name="acceleration",
    )
    assert not failed["passed"]
    record = failed["records"]["V2"]["groups"]["acceleration"]
    assert record["axis_images_concatenated_into_gate"]
    assert record["axis_image_sample_count"] > 0
    assert record["constituent_order"] == (
        "N_tt:full", "T_tt:full", "N_tt:axis_rr", "T_tt:axis_r",
    )
    _assert_constituents_recombine(record, record["constituents"])


def test_master_classifier_is_exhaustive_and_fail_closed():
    parents, two_parent = _master_records()
    passed = classify_protocol125_gate_records(parents, two_parent, _freeze_record())
    assert passed["classification"] == "PASS-native-joint-parent"
    assert passed["phase_a_authorized"]
    assert not passed["rhs_rk_phase_b_full_matrix_authorized"]

    parents["N0"]["native_position_tangent"] = _gate_record(False)
    position = classify_protocol125_gate_records(parents, two_parent, _freeze_record())
    assert position["classification"] == "FAIL-parent-position"

    parents, two_parent = _master_records()
    parents["N1"]["dense_boundary_audit"] = _gate_record(False)
    dense_position = classify_protocol125_gate_records(
        parents, two_parent, _freeze_record(),
    )
    assert dense_position["classification"] == "FAIL-parent-position"

    parents, two_parent = _master_records()
    parents["N1"]["bulk_prerequisite"] = _gate_record(False)
    bulk = classify_protocol125_gate_records(parents, two_parent, _freeze_record())
    assert bulk["classification"] == "FAIL-parent-bulk"

    parents, two_parent = _master_records()
    two_parent["correction_refinement"] = _gate_record(False)
    acceleration = classify_protocol125_gate_records(
        parents, two_parent, _freeze_record(),
    )
    assert acceleration["classification"] == "FAIL-acceleration"

    parents, two_parent = _master_records()
    del parents["N0"]["signature_union"]
    invalid = classify_protocol125_gate_records(parents, two_parent, _freeze_record())
    assert invalid["classification"] == "INVALID-audit"
    assert invalid["invalid_reasons"]


def test_invalid_provenance_dominates_a_numerical_failure():
    parents, two_parent = _master_records()
    parents["N0"]["native_position_tangent"] = _gate_record(False)
    parents["N1"]["position_representation"] = _gate_record(
        False, provenance_valid=False,
    )
    result = classify_protocol125_gate_records(parents, two_parent, _freeze_record())
    assert result["classification"] == "INVALID-audit"
    assert not result["phase_a_authorized"]


def test_master_classifier_cannot_pass_while_protocol_is_draft():
    parents, two_parent = _master_records()
    result = classify_protocol125_gate_records(
        parents,
        two_parent,
        {
            **_freeze_record(),
            "status": "DRAFT — INVALID-specification",
        },
    )
    assert result["classification"] == "INVALID-audit"
    assert result["invalid_reasons"] == ("protocol-is-not-prospectively-frozen",)
    assert not result["phase_a_authorized"]
