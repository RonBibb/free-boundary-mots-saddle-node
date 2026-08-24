from __future__ import annotations

from dataclasses import replace

import numpy as np

from bhps.joint_parent_boundary_contracts import (
    NativeNormalizedCompactWallContract,
    derive_protocol125_outer_derivative_bundle,
)
from bhps.joint_parent_bulk_reference import FiniteWallReferenceHermitePair
from bhps.joint_parent_endpoint_audits import (
    convert_native_fd_acceleration_z_comparator,
    convert_row_implied_acceleration_z,
)
from bhps.joint_parent_final_matrix import (
    INPUT_HASH_KEYS,
    REQUIRED_FINAL_MATRIX_LANES,
    Protocol125FinalMatrixInputs,
    capture_protocol125_final_matrix_provenance,
    evaluate_protocol125_final_representation_matrix,
)
from bhps.joint_parent_legacy_holdout import (
    build_protocol125_legacy_holdout_inputs,
)
from bhps.joint_parent_position_audits import (
    bind_protocol125_position_audit_meshes,
)
from bhps.joint_parent_position_state import (
    derive_joint_parent_position_outer_contract,
)
from bhps.joint_parent_representation import (
    NATIVE_CHANNEL_ORDER,
    RadialFirstConstrainedHermitePair,
    SEALED_ADVERSE_COMPARATOR_NAMES,
)
from bhps.joint_parent_refinement_diagnostics import frozen_validation_meshes
from bhps.matched_staged_continuum import hash_arrays


def _endpoint_mapping(values):
    return {
        name: values[:, :, index]
        for index, name in enumerate(NATIVE_CHANNEL_ORDER)
    }


def _manufactured_flat_inputs():
    z = np.linspace(1.0, np.e, 9)
    r = np.linspace(0.0, 12.0, 17)
    shape = (len(z), len(r))
    zeros = np.zeros(shape)
    reduced_position = np.zeros((*shape, 9))
    reduced_position[:, :, 2] = -1.0
    reduced_position[:, :, 3] = 1.0
    reduced_position[:, :, 6] = 1.0
    selector_q = np.broadcast_to((1.0-z)[:, None], shape).copy()
    parent = {
        "z": z,
        "r": r,
        "position": reduced_position,
        "selector_q": selector_q,
        "psi_selector": np.ones(shape),
        "reference_q": selector_q.copy(),
        "reference_phi": zeros.copy(),
        "chi": zeros.copy(),
        "chi_r": zeros.copy(),
        **{
            f"shape_{name}": zeros.copy()
            for name in ("a", "b", "c", "a_r", "b_r", "c_r")
        },
    }
    position_fields = {
        "h00": -np.ones(shape),
        "h_perp": np.ones(shape),
        "h_rr": np.ones(shape),
        "h_zz": np.ones(shape),
        "Phi": zeros.copy(),
        "chi": zeros.copy(),
        "v_z": zeros.copy(),
        "v_0": zeros.copy(),
    }
    acceleration_fields = {
        name: zeros.copy() for name in NATIVE_CHANNEL_ORDER
    }
    background = {
        "wall_stiffness": 0.0,
        "v0": 0.0,
        "v1": 0.0,
        "beta_a": 0.0,
        "beta_b": 0.0,
        "wall_potential_a": 0.0,
        "wall_potential_b": 0.0,
    }
    position_stack = np.stack(
        tuple(position_fields[name] for name in NATIVE_CHANNEL_ORDER), axis=-1,
    )
    acceleration_stack = np.stack(
        tuple(acceleration_fields[name] for name in NATIVE_CHANNEL_ORDER), axis=-1,
    )
    compact = NativeNormalizedCompactWallContract.build(
        r,
        background,
        position_stack[[0, -1]],
        np.zeros((2, len(r))),
        np.zeros((2, len(r))),
    )
    position_endpoint = compact.z_first_s_jets(
        state_name="position",
        radius=r,
        wall_value_s_jets=(position_stack[[0, -1]],),
    )[0]
    acceleration_endpoint = compact.z_first_s_jets(
        state_name="acceleration",
        radius=r,
        wall_value_s_jets=(acceleration_stack[[0, -1]],),
    )[0]
    position_outer = derive_joint_parent_position_outer_contract(parent)
    shared_outer = derive_protocol125_outer_derivative_bundle(
        position_outer, acceleration_fields,
    )
    final_pair = RadialFirstConstrainedHermitePair.build(
        z,
        r,
        position_fields,
        acceleration_fields,
        _endpoint_mapping(position_endpoint),
        _endpoint_mapping(acceleration_endpoint),
        compact_wall_contract=compact,
        outer_open_face_contract=shared_outer,
        parent_r_max=12.0,
    )
    reference = FiniteWallReferenceHermitePair.build(
        z, r, selector_q, zeros,
    )
    audit_meshes = bind_protocol125_position_audit_meshes(final_pair)
    frozen = frozen_validation_meshes()
    v_meshes = {name: frozen[name] for name in ("V0", "V1", "V2")}
    Q53_source = {}
    Q33_source = {}
    legacy_Q33 = {}
    legacy_Q55 = {}
    for name, mesh in v_meshes.items():
        source_shape = (len(mesh["z"]), len(mesh["r"]), 3)
        Q53_source[name] = {
            lane: np.zeros(source_shape)
            for lane in ("source", "source_time", "source_second_time")
        }
        Q33_source[name] = {
            lane: value.copy() for lane, value in Q53_source[name].items()
        }
        group_shapes = {
            "position": (*source_shape[:2], 9),
            "first_spatial": (*source_shape[:2], 9, 2),
            "second_spatial": (*source_shape[:2], 9, 3),
            "acceleration": (*source_shape[:2], 9),
        }
        legacy_Q33[name] = {
            group: np.zeros(group_shape)
            for group, group_shape in group_shapes.items()
        }
        legacy_Q55[name] = {
            group: value.copy() for group, value in legacy_Q33[name].items()
        }
    fields = tuple(f"legacy_q{index}" for index in range(9))
    component_orders = {
        "position": fields,
        "first_spatial": tuple(
            f"{field}:{direction}"
            for field in fields for direction in ("z", "r")
        ),
        "second_spatial": tuple(
            f"{field}:{direction}"
            for field in fields for direction in ("zz", "zr", "rr")
        ),
        "acceleration": fields,
    }
    source_endpoint = acceleration_stack[[0, -1]]
    source_endpoint_s = np.zeros_like(source_endpoint)
    row_source = convert_row_implied_acceleration_z(
        source_endpoint,
        source_endpoint,
        source_endpoint_s,
        r,
        parent_r_max=12.0,
    )
    direct_source = convert_native_fd_acceleration_z_comparator(
        source_endpoint,
        source_endpoint,
        source_endpoint_s,
        r,
        parent_r_max=12.0,
    )
    dense_r = frozen["dense_wall"]["r"]
    dense_endpoint = np.zeros((2, len(dense_r), len(NATIVE_CHANNEL_ORDER)))
    row_dense = convert_row_implied_acceleration_z(
        dense_endpoint,
        dense_endpoint,
        dense_endpoint,
        dense_r,
        parent_r_max=12.0,
    )
    return Protocol125FinalMatrixInputs(
        final_pair=final_pair,
        reference_pair=reference,
        v_meshes=v_meshes,
        position_audit_meshes=audit_meshes,
        Q53_source_triplets_by_mesh=Q53_source,
        Q33_source_triplets_by_mesh=Q33_source,
        velocity_endpoint_z=np.zeros((2, len(r), 8)),
        row_implied_acceleration_z_source=row_source,
        row_implied_acceleration_z_dense=row_dense,
        direct_Dz7_acceleration_z_source=direct_source,
        shared_representation_fingerprint="f"*64,
        legacy_Q33_by_mesh=legacy_Q33,
        legacy_Q55_by_mesh=legacy_Q55,
        legacy_Q33_source_triplets_by_mesh={
            name: {
                lane: value.copy()
                for lane, value in Q53_source[name].items()
            }
            for name in Q53_source
        },
        legacy_Q55_source_triplets_by_mesh={
            name: {
                lane: value.copy()
                for lane, value in Q53_source[name].items()
            }
            for name in Q53_source
        },
        legacy_component_orders=component_orders,
    )


def test_final_matrix_composes_every_required_lane_without_authorizing_execution():
    inputs = _manufactured_flat_inputs()
    provenance = capture_protocol125_final_matrix_provenance(
        inputs, parent_label="manufactured-flat-matrix",
    )
    result = evaluate_protocol125_final_representation_matrix(inputs, provenance)
    assert result["classification"] == "PASS-final-representation-matrix"
    assert result["complete"]
    assert result["provenance_valid"]
    assert result["passed"]
    assert tuple(result["lanes"]) == REQUIRED_FINAL_MATRIX_LANES
    assert all(record["passed"] for record in result["lanes"].values())
    assert result["inputs_stable_while_scoring"]
    assert result["input_hashes_before"] == result["input_hashes_after"]
    assert not result["phase_a_authorized"]
    assert not result["scientific_execution_authorized"]
    assert not result["artifact_written"]
    wall = result["lanes"]["independent_dense_wall_position"]
    outer = result["lanes"]["independent_dense_outer_position"]
    assert not wall["stored_residual_or_target_array_comparison_used"]
    assert not outer["contract_target_query_called"]
    for lane in (
        "Q53_position_outer_derivative",
        "Q33_position_outer_derivative",
        "Q53_acceleration_outer_derivative",
        "Q33_acceleration_outer_derivative",
    ):
        assert not result["lanes"][lane]["contract_target_query_called"]


def test_final_matrix_rejects_missing_or_reordered_hash_provenance():
    inputs = _manufactured_flat_inputs()
    provenance = capture_protocol125_final_matrix_provenance(
        inputs, parent_label="manufactured-flat-matrix",
    )
    incomplete_hashes = dict(provenance["input_hashes"])
    incomplete_hashes.pop(INPUT_HASH_KEYS[-1])
    incomplete = {
        **dict(provenance),
        "input_hashes": incomplete_hashes,
    }
    result = evaluate_protocol125_final_representation_matrix(inputs, incomplete)
    assert result["classification"] == "INVALID-audit"
    assert not result["complete"]
    assert not result["provenance_valid"]
    assert not result["lanes"]
    assert not result["scientific_execution_authorized"]


def test_final_matrix_rejects_post_capture_tamper_and_missing_lane_before_scoring():
    inputs = _manufactured_flat_inputs()
    provenance = capture_protocol125_final_matrix_provenance(
        inputs, parent_label="manufactured-flat-matrix",
    )
    inputs.legacy_Q55_by_mesh["V0"]["position"][0, 0, 0] = 1e-4
    tampered = evaluate_protocol125_final_representation_matrix(inputs, provenance)
    assert tampered["classification"] == "INVALID-audit"
    assert "input hash mismatch" in tampered["invalid_reasons"][0]
    assert not tampered["lanes"]

    missing = _manufactured_flat_inputs()
    missing_provenance = capture_protocol125_final_matrix_provenance(
        missing, parent_label="manufactured-flat-matrix",
    )
    del missing.Q53_source_triplets_by_mesh["V2"]["source_second_time"]
    rejected = evaluate_protocol125_final_representation_matrix(
        missing, missing_provenance,
    )
    assert rejected["classification"] == "INVALID-audit"
    assert "source triplet lanes are incomplete" in rejected["invalid_reasons"][0]
    assert tuple(SEALED_ADVERSE_COMPARATOR_NAMES) == tuple(
        provenance["legacy_comparator_names"]
    )


def test_endpoint_Dz7_mismatch_is_a_classified_numerical_failure():
    inputs = _manufactured_flat_inputs()
    changed = dict(inputs.direct_Dz7_acceleration_z_source)
    changed_physical = np.asarray(changed["physical"]).copy()
    changed_physical[0, 1, 2] = 2e-12
    changed["physical"] = changed_physical
    # This is a prospectively captured, internally hashed comparator that
    # disagrees numerically with the row-owned route, not a post-capture
    # constituent-hash tamper.
    changed["physical_sha256"] = hash_arrays(changed_physical)
    mismatched = replace(inputs, direct_Dz7_acceleration_z_source=changed)
    provenance = capture_protocol125_final_matrix_provenance(
        mismatched, parent_label="manufactured-flat-matrix",
    )
    result = evaluate_protocol125_final_representation_matrix(
        mismatched, provenance,
    )
    assert result["complete"] and result["provenance_valid"]
    assert not result["passed"]
    assert result["classification"] == "FAIL-final-representation-matrix"
    assert "source_Dz7_vs_row_implied_acceleration_endpoint_z" in (
        result["failed_lanes"]
    )
    assert "Q53_acceleration_endpoint_conversion" in result["failed_lanes"]
    assert "Q33_acceleration_endpoint_conversion" in result["failed_lanes"]


def test_sealed_legacy_adapter_builds_original_groups_and_source_triplets(monkeypatch):
    inputs = _manufactured_flat_inputs()
    def manufactured_source_triplet(member, z, r, background):
        zeros = np.zeros((len(z), len(r), 3))
        return {
            "source": zeros,
            "source_time": zeros.copy(),
            "source_second_time": zeros.copy(),
        }

    monkeypatch.setattr(
        "bhps.joint_parent_legacy_holdout."
        "initial_driver_source_triplet_from_acceleration",
        manufactured_source_triplet,
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
    found = build_protocol125_legacy_holdout_inputs(
        inputs.final_pair,
        background,
        parent_identity="a"*64,
        v_meshes=inputs.v_meshes,
    )
    assert found["complete"]
    assert found["provenance_valid"]
    assert found["passed"]
    assert tuple(found["comparator_names"]) == SEALED_ADVERSE_COMPARATOR_NAMES
    assert tuple(found["legacy_Q33_by_mesh"]) == ("V0", "V1", "V2")
    assert tuple(found["legacy_Q55_by_mesh"]) == ("V0", "V1", "V2")
    assert tuple(found["component_orders"]) == (
        "position", "first_spatial", "second_spatial", "acceleration",
    )
    assert len(found["fingerprint"]) == 64
    assert not found["artifact_written"]
    assert not found["scientific_execution_authorized"]
