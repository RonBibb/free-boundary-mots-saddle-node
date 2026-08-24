from __future__ import annotations

from dataclasses import replace

import numpy as np

import bhps.joint_parent_acceleration as acceleration_module
from bhps.joint_parent_endpoint_audits import (
    ACCELERATION_ENDPOINT_CONVERSION_LANES,
    DENSE_WALL_PROFILE_DERIVATIVE_RECIPE,
    DENSE_WALL_SOURCE_RECIPE,
    SOURCE_WALL_PROFILE_DERIVATIVE_RECIPE,
    WALL_PROFILE_EVIDENCE_PROTOCOL_IDENTIFIER,
    WALL_PROFILE_INPUT_HASH_KEYS,
    score_normalized_wall_profiles,
    wall_profile_evidence_fingerprint,
)
from bhps.joint_parent_final_matrix import (
    INPUT_HASH_KEYS as FINAL_MATRIX_INPUT_HASH_KEYS,
    PROTOCOL_IDENTIFIER as FINAL_MATRIX_PROTOCOL_IDENTIFIER,
    REQUIRED_FINAL_MATRIX_LANES,
)
from bhps.joint_parent_gate_ledger import Protocol125GateLedger
from bhps.joint_parent_ordered_adjudicator import (
    adjudicate_protocol125_ordered,
)
from bhps.joint_parent_postacceleration import (
    ACCELERATION_FAILURE_INPUT_HASH_KEYS,
    ENDPOINT_DERIVATIVE_LANES,
    FINAL_REPRESENTATION_LANES,
    POST_ACCELERATION_GROUPS,
    Protocol125PostAccelerationFailureInputs,
    Protocol125PostAccelerationInputs,
    capture_protocol125_acceleration_failure_provenance,
    capture_protocol125_bulk_sampler_provenance,
    capture_protocol125_postacceleration_provenance,
    compose_protocol125_acceleration_failure_records,
    compose_protocol125_postacceleration_records,
)
from bhps.joint_parent_preacceleration import (
    INPUT_HASH_KEYS as PRE_ACCELERATION_INPUT_HASH_KEYS,
    PRE_ACCELERATION_GROUPS,
    PROTOCOL_IDENTIFIER as PRE_ACCELERATION_PROTOCOL_IDENTIFIER,
    _gate_record as _preacceleration_gate_record,
)
from bhps.joint_parent_protocol125_sampling_lineage import (
    PositionPayloadSnapshot,
    Protocol125BulkAccelerationSampler,
    validate_append_only_position_lineage,
)
from bhps.joint_parent_refinement_diagnostics import (
    DENSE_OUTER_SHA256,
    DENSE_WALL_SHA256,
    axis_acceleration_derivative_image_profile,
    correction_profile,
    frozen_validation_meshes,
)
from bhps.matched_staged_continuum import DriverConfiguration, hash_arrays


PARENT_IDENTITY = "a"*64


def _sha(index):
    return f"{int(index):064x}"


def _pre_acceleration_result():
    context = {"label": "N0", "identity": PARENT_IDENTITY}
    groups = {
        name: _preacceleration_gate_record(
            name,
            context,
            complete=True,
            provenance_valid=True,
            passed=True,
            details={"manufactured": name},
        )
        for name in PRE_ACCELERATION_GROUPS
    }
    hashes = {
        name: _sha(index+1)
        for index, name in enumerate(PRE_ACCELERATION_INPUT_HASH_KEYS)
    }
    return {
        "protocol_identifier": PRE_ACCELERATION_PROTOCOL_IDENTIFIER,
        "classification": "PASS-single-parent-pre-acceleration",
        "complete": True,
        "provenance_valid": True,
        "passed": True,
        "parent_label": "N0",
        "parent_identity": PARENT_IDENTITY,
        "required_group_order": PRE_ACCELERATION_GROUPS,
        "groups": groups,
        "invalid_reasons": (),
        "input_hashes_before": hashes,
        "input_hashes_after": dict(hashes),
        "inputs_stable_while_scoring": True,
        "single_parent_only": True,
        "second_parent_and_common_V2_still_required": True,
        "acceleration_evaluated": False,
        "acceleration_authorized": False,
        "scientific_execution_authorized": False,
        "artifact_written": False,
    }


def _coupled_record(nr):
    profiles = {
        "rank": [4]*nr,
        "equilibrated_condition": [2.0]*nr,
        "raw_condition": [2.0]*nr,
        "pivot_strength": [0.5]*nr,
        "normalized_linear_residual": [0.0]*nr,
        "maximum_absolute_endpoint_correction": [0.0]*nr,
    }
    return {
        "method": "direct_radial_4x4_both_walls_Phi_gzz",
        "maximum_allowed_condition": 1e12,
        "minimum_allowed_pivot_strength": 1e-10,
        "maximum_allowed_normalized_linear_residual": 1e-12,
        "maximum_condition": 2.0,
        "maximum_raw_condition": 2.0,
        "minimum_pivot_strength": 0.5,
        "worst_condition_radial_index": nr-1,
        "worst_condition_radius": 12.0,
        "weakest_pivot_radial_index": nr-1,
        "weakest_pivot_radius": 12.0,
        "maximum_normalized_linear_residual": 0.0,
        "minimum_rank": 4,
        "maximum_absolute_endpoint_correction": 0.0,
        "relative_correction": 0.0,
        "passed": True,
        "profiles": profiles,
    }


def _selective_field(nr):
    return {
        "required_rank": 2,
        "maximum_allowed_condition": 1e12,
        "minimum_allowed_normalized_pivot": 1e-10,
        "maximum_allowed_normalized_linear_residual": 1e-12,
        "minimum_rank": 2,
        "maximum_equilibrated_condition": 2.0,
        "maximum_raw_condition": 2.0,
        "minimum_normalized_pivot": 0.5,
        "maximum_normalized_linear_residual": 0.0,
        "profiles": {
            "rank": np.full(nr, 2, dtype=int),
            "equilibrated_condition": np.full(nr, 2.0),
            "raw_condition": np.full(nr, 2.0),
            "normalized_pivot": np.full(nr, 0.5),
            "normalized_linear_residual": np.zeros(nr),
        },
        "passed": True,
    }


def _selective_wall(name, nr):
    components = {}
    for component in ("tt", "sphere", "rr", "tr"):
        components[component] = {
            "residual": np.zeros(nr),
            "scale": np.ones(nr),
            "normalized": np.zeros(nr),
            "maximum_normalized": 0.0,
            "terms": np.zeros((4, nr)),
            "direct_physical_a_z": np.zeros(nr),
            "row_implied_physical_a_z": np.zeros(nr),
        }
    return {
        "wall": name,
        "radial_indices": np.arange(nr),
        "components": components,
        "chi": {
            "residual": np.zeros(nr),
            "scale": np.ones(nr),
            "normalized": np.zeros(nr),
            "maximum_normalized": 0.0,
            "contributions": np.zeros((3, nr)),
            "direct_physical_a_z": np.zeros(nr),
            "row_implied_physical_a_z": np.zeros(nr),
        },
        "maximum_metric_normalized": 0.0,
        "maximum_chi_normalized": 0.0,
    }


def _selective_record(nr):
    fields = {
        name: _selective_field(nr)
        for name in ("h_00", "h_perp", "h_rr", "h_0r", "chi")
    }
    return {
        "method": "direct_normalized_time_symmetric_DX2J_plus_chi",
        "time_symmetric": True,
        "maximum_allowed_normalized_residual": 1e-12,
        "maximum_metric_normalized_residual": 0.0,
        "maximum_chi_normalized_residual": 0.0,
        "minimum_tangential_endpoint_rank": 2,
        "maximum_tangential_endpoint_condition": 2.0,
        "weakest_tangential_endpoint_pivot": 0.5,
        "chi_endpoint_rank": 2,
        "per_field_algebraic_evidence": {
            "field_order": ("h_00", "h_perp", "h_rr", "h_0r", "chi"),
            "fields": fields,
            "each_field_gated_separately": True,
            "chi_credited_only_with_chi_block": True,
            "passed": True,
        },
        "protected_0_1_6_7_bitwise": True,
        "q4_q5_axis_bitwise": True,
        "walls": [_selective_wall("lower", nr), _selective_wall("upper", nr)],
        "direct_physical_a_z": np.zeros((2, nr, 9)),
        "row_implied_physical_a_z": np.zeros((2, nr, 9)),
        "row_defined_mask": np.asarray(
            (False, False, True, True, True, True, False, False, True),
        ),
        "maximum_row_implied_scaled_defect": 0.0,
        "passed": True,
    }


def _axis_record():
    return {
        "method": "native-seven-point-physical-numerator-axis-parity",
        "stencil_width": 7,
        "parent_radius": 12.0,
        "anisotropy_axis_positive_zero": True,
        "time_radial_axis_positive_zero": True,
        "positive_radius_reassembly_bitwise": True,
        "only_q4_q5_axis_changed_bitwise": True,
        "polynomial_fit_applied": False,
    }


def _junction_wall(name, nr):
    component_keys = (
        "metric", "metric_t", "metric_tt", "normal_derivative",
        "normal_derivative_t", "normal_derivative_tt", "robin_residual",
        "DX_robin_residual", "DX2_robin_residual", "J", "DXJ",
        "DJ_acceleration", "D2J_velocity_velocity", "DX2J",
        "DX2J_raw_robin_form",
    )
    components = {
        component: {
            **{key: np.zeros(nr) for key in component_keys},
            "second_form_maximum_absolute_defect": 0.0,
        }
        for component in ("tt", "rr", "sphere", "tr")
    }
    zeros = np.zeros((nr, 4, 4))
    separate_rows = {
        key: np.zeros(nr)
        for key in (
            "Phi_robin", "DX_Phi_robin", "DJ_Phi_robin_acceleration",
            "D2_Phi_robin_velocity_velocity", "DX2_Phi_robin", "chi_neumann",
            "DX_chi_neumann", "DJ_chi_neumann_acceleration",
            "D2_chi_neumann_velocity_velocity", "DX2_chi_neumann",
        )
    }
    source = {
        key: (
            np.asarray(0.0) if key == "beta_phiphi"
            else np.ones(nr) if key == "sqrt_gzz"
            else np.zeros(nr)
        )
        for key in (
            "beta", "beta_phi", "beta_phiphi", "beta_t", "beta_tt",
            "sqrt_gzz", "sqrt_gzz_t", "sqrt_gzz_tt",
        )
    }
    return {
        "wall": name,
        "orientation": -1.0 if name == "lower" else 1.0,
        "scope": "manufactured fixed-grid tangent",
        "components": components,
        "metric_tensor": zeros.copy(),
        "J_tensor": zeros.copy(),
        "DXJ_tensor": zeros.copy(),
        "DJ_acceleration_tensor": zeros.copy(),
        "D2J_velocity_velocity_tensor": zeros.copy(),
        "DX2J_tensor": zeros.copy(),
        "decomposition_maximum_absolute_defect": 0.0,
        "raw_vs_cancellation_exposed_maximum_absolute_defect": 0.0,
        "separate_rows": separate_rows,
        "source": source,
        "finite": True,
    }


def _normal_gauge_record():
    return {
        "walls": [
            {"wall": "lower", "maximum_normalized": 0.0, "maximum_absolute": 0.0},
            {"wall": "upper", "maximum_normalized": 0.0, "maximum_absolute": 0.0},
        ],
        "maximum": 0.0,
    }


def _fixed_point_record(nz=3, nr=3):
    coupled = _coupled_record(nr)
    selective = _selective_record(nr)
    axis = _axis_record()
    history = [
        {
            "map": index,
            "acceleration_scaled_Linf_change": 0.0,
            "source_triplet_scaled_Linf_change": 0.0,
            "consecutive_converged_maps": index,
            "coupled": coupled,
            "selective": selective,
            "normal_tangential_correction_scaled_Linf": 0.0,
            "axis_reconciliation_scaled_Linf": 0.0,
            "axis_reconciliation": axis,
        }
        for index in (1, 2)
    ]
    zeros = np.zeros((nz, nr, 3))
    source = {
        "source": zeros.copy(),
        "source_time": zeros.copy(),
        "source_second_time": zeros.copy(),
        "memory": zeros.copy(),
        "memory_time": zeros.copy(),
        "target": zeros.copy(),
        "advection": zeros.copy(),
        "raw_geometric_source": zeros.copy(),
        "normal_wall_completion": {
            "normal_gauge": {"maximum": 0.0},
            "ownership_pass": True,
            "changed_value_count": 0,
            "finite": True,
        },
        "Hdot_reassembly_scaled_Linf": 0.0,
        "difference_step": 1e-6,
        "driver": DriverConfiguration().public(),
        "outer_source_overwrite_applied": False,
        "memory_carried_from_previous_iterate": False,
    }
    return {
        "method": "Protocol-125-full-update-eight-map-fixed-point",
        "history": history,
        "maps_used": 2,
        "consecutive_converged_maps": 2,
        "source_triplet": source,
        "coupled": coupled,
        "selective": selective,
        "axis_reconciliation": axis,
        "normal_gauge": _normal_gauge_record(),
        "wall_second_tangent": {
            "lower": _junction_wall("lower", nr),
            "upper": _junction_wall("upper", nr),
        },
        "delta_acceleration": np.zeros((nz, nr, 9)),
        "outer_overwrite_applied": False,
        "generic_axis_fill_applied": False,
        "endpoint_history_carried": False,
    }


def _normalized_wall_score(nr=1025):
    zeros = np.zeros((2, nr))
    position = {
        "Phi": zeros.copy(),
        "Phi_z": zeros.copy(),
        "chi_z": zeros.copy(),
        "G": np.ones((2, nr)),
        "G_z": zeros.copy(),
        "H_z": zeros.copy(),
    }
    acceleration = {
        name: zeros.copy()
        for name in ("a_Phi", "a_Phi_z", "a_chi_z", "a_G", "a_G_z", "H_ztt")
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
    return score_normalized_wall_profiles(position, acceleration, background)


def _normalized_wall_evidence(*, fail_source_gate=None):
    z = np.linspace(1.0, np.e, 9)
    r = np.linspace(0.0, 12.0, 17)
    dense_r = frozen_validation_meshes()["dense_wall"]["r"]
    source = _normalized_wall_score(len(r))
    dense = _normalized_wall_score(len(dense_r))
    if fail_source_gate is not None:
        # Re-score a real isolated source-row failure so the record remains
        # internally self-consistent rather than flipping a summary bit.
        zeros = np.zeros((2, len(r)))
        position = {
            "Phi": zeros.copy(),
            "Phi_z": zeros.copy(),
            "chi_z": zeros.copy(),
            "G": np.ones((2, len(r))),
            "G_z": zeros.copy(),
            "H_z": zeros.copy(),
        }
        acceleration = {
            name: zeros.copy()
            for name in (
                "a_Phi", "a_Phi_z", "a_chi_z", "a_G", "a_G_z", "H_ztt"
            )
        }
        if fail_source_gate == "source_acceleration_chi_lower":
            acceleration["a_chi_z"][0, 3] = 2e-10
        else:
            raise ValueError("unsupported manufactured source wall gate")
        source = score_normalized_wall_profiles(
            position,
            acceleration,
            {
                "wall_stiffness": 0.0,
                "v0": 0.0,
                "v1": 0.0,
                "beta_a": 0.0,
                "beta_b": 0.0,
                "wall_potential_a": 0.0,
                "wall_potential_b": 0.0,
            },
        )
    velocity_shape = (len(z), len(r), 9)
    dense_velocity_shape = (2, len(dense_r), 9)
    source_velocity_sha = hash_arrays(np.zeros(velocity_shape))
    dense_velocity_sha = hash_arrays(np.zeros(dense_velocity_shape))
    walls = z[[0, -1]]
    input_hashes = {
        name: _sha(500+index)
        for index, name in enumerate(WALL_PROFILE_INPUT_HASH_KEYS)
    }
    input_hashes["completed_velocity_sha256"] = source_velocity_sha
    mesh_scores = {"source": source, "dense": dense}
    named_order = tuple(
        f"{mesh}_{stage}_{row}_{wall}"
        for mesh in ("source", "dense")
        for stage in ("position", "acceleration")
        for row in ("Phi", "chi", "normal_GH")
        for wall in ("lower", "upper")
    )
    named = {
        f"{mesh}_{name}": bool(mesh_scores[mesh]["gates"][name])
        for mesh in ("source", "dense")
        for name in mesh_scores[mesh]["gates"]
    }
    named_passed = all(named.values())
    record = {
        "protocol_identifier": WALL_PROFILE_EVIDENCE_PROTOCOL_IDENTIFIER,
        "parent_label": "N0",
        "parent_identity": PARENT_IDENTITY,
        "source_fingerprint": _sha(300),
        "endpoint_fingerprint": _sha(301),
        "wall_order": ("lower", "upper"),
        "row_order": ("Phi", "chi", "normal_GH"),
        "mesh_order": ("source", "dense"),
        "coordinates": {
            "source_z": z,
            "source_r": r,
            "dense_r": dense_r,
            "source_z_sha256": hash_arrays(z),
            "source_r_sha256": hash_arrays(r),
            "source_pair_sha256": hash_arrays(z, r),
            "source_wall_coordinate_sha256": hash_arrays(walls, r),
            "dense_r_sha256": hash_arrays(dense_r),
            "dense_wall_coordinate_sha256": hash_arrays(walls, dense_r),
        },
        "derivative_recipes": {
            "source_recipe": SOURCE_WALL_PROFILE_DERIVATIVE_RECIPE,
            "source_stencil_width": 7,
            "dense_recipe": DENSE_WALL_PROFILE_DERIVATIVE_RECIPE,
            "dense_source_recipe": DENSE_WALL_SOURCE_RECIPE,
        },
        "input_hashes": input_hashes,
        "time_symmetry": {
            "source_velocity_shape": velocity_shape,
            "source_velocity_sha256": source_velocity_sha,
            "source_positive_zero_reference_sha256": source_velocity_sha,
            "source_bitwise_positive_zero": True,
            "dense_velocity_shape": dense_velocity_shape,
            "dense_velocity_sha256": dense_velocity_sha,
            "dense_positive_zero_reference_sha256": dense_velocity_sha,
            "dense_bitwise_positive_zero": True,
            "dense_velocity_recipe": "exact-positive-zero-time-symmetric-extension",
            "passed": True,
        },
        "live_compact_context": {
            "contract_identifier": "native-normalized-compact-wall-shared-v1:test",
            "contract_fingerprint": _sha(600),
            "position_and_acceleration_share_live_contract": True,
            "source_normal_context_present": True,
            "source_second_normal_context_present": True,
            "source_position_reproduction_scaled_Linf": 0.0,
            "source_acceleration_reproduction_scaled_Linf": 0.0,
            "source_normal_context_reproduction_scaled_Linf": 0.0,
            "source_second_normal_context_reproduction_scaled_Linf": 0.0,
            "dense_source_normal_sha256": _sha(601),
            "dense_source_second_normal_sha256": _sha(602),
            "passed": True,
        },
        "meshes": mesh_scores,
        "named_gate_order": named_order,
        "named_row_wall_gates": named,
        "named_row_wall_passed": named_passed,
        "constituent_logical_AND": True,
        "complete": True,
        "provenance_valid": True,
        "passed": named_passed,
        "scientific_execution_authorized": False,
        "artifact_written": False,
    }
    record["fingerprint"] = wall_profile_evidence_fingerprint(record)
    return record


def _final_matrix_result(*, failed_lane=None):
    lanes = {}
    endpoint_states = {
        "Q53_position_endpoint_z": "position",
        "Q33_position_endpoint_z": "position",
        "Q53_acceleration_endpoint_z": "acceleration",
        "Q33_acceleration_endpoint_z": "acceleration",
    }
    outer_lanes = {
        "Q53_position_outer_derivative",
        "Q33_position_outer_derivative",
        "Q53_acceleration_outer_derivative",
        "Q33_acceleration_outer_derivative",
    }
    for index, name in enumerate(REQUIRED_FINAL_MATRIX_LANES):
        passed = name != failed_lane
        lane = {"passed": passed}
        if name in endpoint_states:
            lane.update({
                "state_name": endpoint_states[name],
                "dense_r_sha256": DENSE_WALL_SHA256,
            })
        elif name == "time_symmetric_velocity_endpoint_z":
            lane.update({"bitwise_positive_zero": True, "fingerprint": _sha(100)})
        elif name in outer_lanes:
            lane.update({
                "complete": True,
                "provenance_valid": True,
                "compact_endpoints_excluded_from_score": True,
                "contract_target_query_called": False,
                "fresh_degree_five_target_reconstruction": True,
                "dense_outer_sha256": DENSE_OUTER_SHA256,
                "fingerprint": _sha(101+index),
            })
        elif name == "independent_dense_wall_position":
            lane["stored_residual_or_target_array_comparison_used"] = False
        elif name == "independent_dense_outer_position":
            lane["contract_target_query_called"] = False
        lanes[name] = lane
    failed = tuple(name for name in REQUIRED_FINAL_MATRIX_LANES if not lanes[name]["passed"])
    hashes = {
        name: _sha(200+index)
        for index, name in enumerate(FINAL_MATRIX_INPUT_HASH_KEYS)
    }
    passed = not failed
    return {
        "protocol_identifier": FINAL_MATRIX_PROTOCOL_IDENTIFIER,
        "parent_label": "N0",
        "classification": (
            "PASS-final-representation-matrix" if passed
            else "FAIL-final-representation-matrix"
        ),
        "complete": True,
        "provenance_valid": True,
        "passed": passed,
        "required_lane_order": REQUIRED_FINAL_MATRIX_LANES,
        "lanes": lanes,
        "failed_lanes": failed,
        "invalid_reasons": (),
        "source_fingerprint": _sha(300),
        "endpoint_fingerprint": _sha(301),
        "input_hashes_before": hashes,
        "input_hashes_after": dict(hashes),
        "inputs_stable_while_scoring": True,
        "constituent_logical_AND": True,
        "phase_a_authorized": False,
        "scientific_execution_authorized": False,
        "artifact_written": False,
    }


def _lineage_result():
    groups = {
        "source": {"q": np.asarray((1.0, 2.0))},
        "compact": {"ownership": np.asarray((True, False))},
        "outer": {"target": np.asarray((0.1, 0.2))},
        "coefficients": {"c": np.asarray((3.0, 4.0))},
    }
    position = PositionPayloadSnapshot.capture_position_only(
        groups,
        compact_identifier="compact-position-only",
        compact_fingerprint=_sha(400),
        outer_identifier="outer-position-only",
        outer_fingerprint=_sha(401),
        archive_fingerprint=_sha(402),
    )
    shared = PositionPayloadSnapshot.capture_shared(
        groups,
        parent=position,
        appended_children={"acceleration": np.asarray((5.0, 6.0))},
        compact_identifier="compact-shared",
        compact_fingerprint=_sha(403),
        outer_identifier="outer-shared",
        outer_fingerprint=_sha(404),
        archive_fingerprint=_sha(405),
    )
    return validate_append_only_position_lineage(position, shared)


def _correction_and_axis(amplitude=0.0):
    meshes = frozen_validation_meshes()
    r = meshes["dense_wall"]["r"]
    position = np.zeros((3, len(r), 9))
    position[:, :, 2] = -1.0
    position[:, :, 3] = 1.0
    position[:, :, 6] = 1.0
    bulk = np.zeros_like(position)
    compatible = np.zeros_like(position)
    compatible[[0, -1], :, 6] = float(amplitude)
    correction = correction_profile(position, bulk, compatible, r)
    z = meshes["V2"]["z"]
    axis = axis_acceleration_derivative_image_profile(
        np.zeros((len(z), 9)), np.zeros((len(z), 9)), z,
    )
    return correction, axis


def _bulk_sampler():
    z = np.linspace(1.0, np.e, 9)
    r = np.linspace(0.0, 12.0, 9)
    return Protocol125BulkAccelerationSampler.build(z, r, np.zeros((9, 9, 9)))


def _inputs(amplitude=0.0):
    correction, axis = _correction_and_axis(amplitude)
    sampler_provenance = capture_protocol125_bulk_sampler_provenance(
        _bulk_sampler(),
        parent_label="N0",
        parent_identity=PARENT_IDENTITY,
        correction_profile=correction,
        axis_image_profile=axis,
    )
    return Protocol125PostAccelerationInputs(
        pre_acceleration_result=_pre_acceleration_result(),
        fixed_point_record=_fixed_point_record(),
        normalized_wall_profile_score=_normalized_wall_evidence(),
        final_representation_matrix=_final_matrix_result(),
        append_only_lineage=_lineage_result(),
        correction_profile=correction,
        axis_image_profile=axis,
        bulk_sampler_provenance=sampler_provenance,
    )


class _FailurePositionState:
    def fingerprint(self):
        return _sha(900)


def _scientific_acceleration_failure(*, wall=False):
    z = np.linspace(1.0, 2.0, 7)
    r = np.linspace(0.0, 1.0, 7)
    acceleration = np.zeros((len(z), len(r), 9))
    (
        label,
        identity,
        fixed,
        input_provenance,
        attempt,
    ) = acceleration_module._input_provenance(
        _FailurePositionState(),
        np.zeros_like(acceleration),
        acceleration,
        z,
        r,
        {"mass_squared": 0.0},
        parent_label="N0",
        parent_identity=PARENT_IDENTITY,
    )
    source = {
        "source": np.zeros((len(z), len(r), 3)),
        "source_time": np.zeros((len(z), len(r), 3)),
        "source_second_time": np.zeros((len(z), len(r), 3)),
    }
    if wall:
        history = ()
        event = {
            "owner": "coupled",
            "failed_map": 1,
            "exception_type": "CompactWallCoupledAlgebraicGateError",
            "message": "manufactured coupled wall rejection",
            "gate": "rank_condition_pivot",
            "radial_index": 2,
            "radius": float(r[2]),
            "field": None,
            "diagnostics": {"rank": 3, "required_rank": 4},
        }
        return acceleration_module._acceleration_failure_record(
            parent_label=label,
            parent_identity=identity,
            attempt_fingerprint=attempt,
            fixed_settings=fixed,
            input_provenance=input_provenance,
            failure_group="wall_algebra",
            failure_reason="coupled_wall_algebraic_gate_failure",
            history=history,
            consecutive_converged_maps=0,
            current=acceleration,
            source_triplet=source,
            failure_event=event,
            failed_stage_input=acceleration,
        )
    history = tuple({
        "map": index,
        "acceleration_scaled_Linf_change": 0.1,
        "source_triplet_scaled_Linf_change": 0.2,
        "consecutive_converged_maps": 0,
        "coupled": {"map": index, "passed": True},
        "selective": {"map": index, "passed": True},
        "normal_tangential_correction_scaled_Linf": 0.0,
        "axis_reconciliation_scaled_Linf": 0.0,
        "axis_reconciliation": {"map": index, "passed": True},
    } for index in range(1, 9))
    event = {
        "owner": "fixed_point",
        "failed_map": None,
        "exception_type": None,
        "message": "manufactured fixed-point nonconvergence",
        "gate": "two_consecutive_map_convergence",
        "radial_index": None,
        "radius": None,
        "field": None,
        "diagnostics": {
            "maximum_maps": 8,
            "required_consecutive": 2,
            "observed_consecutive": 0,
            "convergence_tolerance": 1e-12,
            "final_acceleration_scaled_Linf_change": 0.1,
            "final_source_triplet_scaled_Linf_change": 0.2,
        },
    }
    return acceleration_module._acceleration_failure_record(
        parent_label=label,
        parent_identity=identity,
        attempt_fingerprint=attempt,
        fixed_settings=fixed,
        input_provenance=input_provenance,
        failure_group="acceleration_closure",
        failure_reason="fixed_point_nonconvergence",
        history=history,
        consecutive_converged_maps=0,
        current=acceleration,
        source_triplet=source,
        failure_event=event,
        coupled=history[-1]["coupled"],
        selective=history[-1]["selective"],
        axis_reconciliation=history[-1]["axis_reconciliation"],
    )


def test_postacceleration_composer_emits_exact_immutable_ledger_ready_pass_set():
    inputs = _inputs()
    provenance = capture_protocol125_postacceleration_provenance(inputs)
    result = compose_protocol125_postacceleration_records(inputs, provenance)
    assert result["classification"] == "PASS-single-parent-post-acceleration"
    assert result["complete"] and result["provenance_valid"] and result["passed"]
    assert tuple(result["groups"]) == POST_ACCELERATION_GROUPS
    assert all(record["passed"] for record in result["groups"].values())
    assert tuple(FINAL_REPRESENTATION_LANES)+tuple(ENDPOINT_DERIVATIVE_LANES) != (
        REQUIRED_FINAL_MATRIX_LANES
    )
    assert set(FINAL_REPRESENTATION_LANES) | set(ENDPOINT_DERIVATIVE_LANES) == set(
        REQUIRED_FINAL_MATRIX_LANES
    )
    wall_details = result["groups"]["wall_algebra"]["details"]
    assert wall_details["source_normalized_wall_profiles"]["passed"]
    assert wall_details["dense_normalized_wall_profiles"]["passed"]
    assert wall_details["wall_profile_evidence"]["named_row_wall_gate_count"] == 24
    ledger = Protocol125GateLedger(
        {"status": "DRAFT"},
        {"N0": PARENT_IDENTITY, "N1": "b"*64},
    )
    for name in POST_ACCELERATION_GROUPS:
        ledger = ledger.append_parent_gate("N0", name, result["groups"][name])
    assert tuple(ledger.parent_records["N0"]) == POST_ACCELERATION_GROUPS
    assert not result["phase_a_authorized"]
    assert not result["scientific_execution_authorized"]
    assert not result["artifact_written"]
    with np.testing.assert_raises(TypeError):
        result["groups"]["correction_size"]["details"]["new"] = True


def test_large_but_well_formed_correction_is_scientific_failure_not_invalid_audit():
    inputs = _inputs(amplitude=0.06)
    provenance = capture_protocol125_postacceleration_provenance(inputs)
    result = compose_protocol125_postacceleration_records(inputs, provenance)
    assert result["classification"] == "FAIL-single-parent-post-acceleration"
    assert result["complete"]
    assert result["provenance_valid"]
    assert not result["passed"]
    assert result["failed_groups"] == ("correction_size",)
    correction = result["groups"]["correction_size"]
    assert correction["complete"] and correction["provenance_valid"]
    assert not correction["passed"]
    assert not correction["invalid_reasons"]
    assert correction["details"]["dense_wall_correction"]["full_physical_Linf"] == 0.06


def test_source_wall_failure_survives_a_passing_dense_analytic_score():
    inputs = _inputs()
    source_failed = replace(
        inputs,
        normalized_wall_profile_score=_normalized_wall_evidence(
            fail_source_gate="source_acceleration_chi_lower",
        ),
    )
    provenance = capture_protocol125_postacceleration_provenance(source_failed)
    result = compose_protocol125_postacceleration_records(source_failed, provenance)
    assert result["classification"] == "FAIL-single-parent-post-acceleration"
    assert result["complete"] and result["provenance_valid"]
    assert result["failed_groups"] == ("wall_algebra",)
    wall = result["groups"]["wall_algebra"]
    assert not wall["passed"]
    assert not wall["details"]["source_normalized_wall_profiles"]["passed"]
    assert wall["details"]["dense_normalized_wall_profiles"]["passed"]
    assert not wall["details"]["wall_profile_evidence"]["named_row_wall_passed"]


def test_acceleration_endpoint_conversion_failure_reaches_endpoint_group():
    inputs = _inputs()
    failed_lane = ACCELERATION_ENDPOINT_CONVERSION_LANES[0]
    changed = replace(
        inputs,
        final_representation_matrix=_final_matrix_result(failed_lane=failed_lane),
    )
    provenance = capture_protocol125_postacceleration_provenance(changed)
    result = compose_protocol125_postacceleration_records(changed, provenance)
    assert result["classification"] == "FAIL-single-parent-post-acceleration"
    assert result["complete"] and result["provenance_valid"]
    assert result["failed_groups"] == ("endpoint_derivatives",)
    endpoint = result["groups"]["endpoint_derivatives"]
    assert not endpoint["passed"]
    assert not endpoint["details"]["lane_pass"][failed_lane]


def test_missing_or_post_capture_tampered_evidence_is_invalid_audit():
    inputs = _inputs()
    missing_wall = dict(inputs.normalized_wall_profile_score)
    missing_wall.pop("fingerprint")
    missing_inputs = replace(inputs, normalized_wall_profile_score=missing_wall)
    missing_provenance = capture_protocol125_postacceleration_provenance(missing_inputs)
    missing = compose_protocol125_postacceleration_records(
        missing_inputs, missing_provenance,
    )
    assert missing["classification"] == "INVALID-audit"
    assert not missing["complete"]
    assert not missing["provenance_valid"]
    assert tuple(missing["groups"]) == POST_ACCELERATION_GROUPS
    assert all(not record["complete"] for record in missing["groups"].values())
    assert "schema" in missing["invalid_reasons"][0]

    absent_source = _normalized_wall_evidence()
    absent_source["meshes"] = dict(absent_source["meshes"])
    absent_source["meshes"].pop("source")
    absent_source_inputs = replace(
        inputs, normalized_wall_profile_score=absent_source,
    )
    absent_source_provenance = capture_protocol125_postacceleration_provenance(
        absent_source_inputs,
    )
    absent = compose_protocol125_postacceleration_records(
        absent_source_inputs, absent_source_provenance,
    )
    assert absent["classification"] == "INVALID-audit"
    assert "mesh inventory" in absent["invalid_reasons"][0]

    tampered_inputs = _inputs()
    tampered_provenance = capture_protocol125_postacceleration_provenance(
        tampered_inputs,
    )
    tampered_inputs.correction_profile["signed_normalized_correction"][0, 4, 2] = 1e-7
    tampered = compose_protocol125_postacceleration_records(
        tampered_inputs, tampered_provenance,
    )
    assert tampered["classification"] == "INVALID-audit"
    assert "input hash mismatch" in tampered["invalid_reasons"][0]
    assert all(not record["provenance_valid"] for record in tampered["groups"].values())


def test_wrong_fixed_point_owner_or_evidence_identity_fails_closed():
    inputs = _inputs()
    bad_fixed = dict(inputs.fixed_point_record)
    bad_fixed["outer_overwrite_applied"] = True
    owner_inputs = replace(inputs, fixed_point_record=bad_fixed)
    owner_provenance = capture_protocol125_postacceleration_provenance(owner_inputs)
    owner = compose_protocol125_postacceleration_records(owner_inputs, owner_provenance)
    assert owner["classification"] == "INVALID-audit"
    assert "owner prohibition" in owner["invalid_reasons"][0]

    bad_matrix = dict(inputs.final_representation_matrix)
    bad_matrix["parent_label"] = "N1"
    identity_inputs = replace(inputs, final_representation_matrix=bad_matrix)
    identity_provenance = capture_protocol125_postacceleration_provenance(identity_inputs)
    identity = compose_protocol125_postacceleration_records(
        identity_inputs, identity_provenance,
    )
    assert identity["classification"] == "INVALID-audit"
    assert "mislabeled" in identity["invalid_reasons"][0]


def test_scientific_acceleration_stop_composes_failed_closure_and_not_reached():
    failure = _scientific_acceleration_failure()
    inputs = Protocol125PostAccelerationFailureInputs(
        pre_acceleration_result=_pre_acceleration_result(),
        acceleration_failure_record=failure,
    )
    provenance = capture_protocol125_acceleration_failure_provenance(inputs)
    assert tuple(provenance["input_hashes"]) == (
        ACCELERATION_FAILURE_INPUT_HASH_KEYS
    )
    result = compose_protocol125_acceleration_failure_records(
        inputs, provenance,
    )
    assert result["classification"] == "FAIL-acceleration"
    assert result["complete"] and result["provenance_valid"]
    assert not result["passed"]
    assert result["failed_groups"] == ("acceleration_closure",)
    assert result["not_reached_groups"] == POST_ACCELERATION_GROUPS[1:]
    closure = result["groups"]["acceleration_closure"]
    assert closure["complete"] and closure["provenance_valid"]
    assert not closure["passed"]
    assert "not_reached" not in closure
    assert closure["details"]["scientific_failure_record"][
        "fingerprint"
    ] == failure["fingerprint"]
    for name in POST_ACCELERATION_GROUPS[1:]:
        gate = result["groups"][name]
        assert gate["complete"] and gate["provenance_valid"]
        assert not gate["passed"] and gate["not_reached"]
        assert gate["blocked_by"] == "acceleration_closure"
    with np.testing.assert_raises(TypeError):
        result["groups"]["wall_algebra"]["details"]["new"] = True


def test_measured_wall_abort_records_both_consequences_before_ordered_stop():
    failure = _scientific_acceleration_failure(wall=True)
    inputs = Protocol125PostAccelerationFailureInputs(
        pre_acceleration_result=_pre_acceleration_result(),
        acceleration_failure_record=failure,
    )
    provenance = capture_protocol125_acceleration_failure_provenance(inputs)
    result = compose_protocol125_acceleration_failure_records(inputs, provenance)
    assert result["classification"] == "FAIL-acceleration"
    assert result["failed_groups"] == (
        "acceleration_closure", "wall_algebra",
    )
    assert result["not_reached_groups"] == POST_ACCELERATION_GROUPS[2:]
    wall = result["groups"]["wall_algebra"]
    assert wall["complete"] and wall["provenance_valid"]
    assert not wall["passed"] and "not_reached" not in wall
    assert wall["details"]["direct_measured_wall_gate_failure"]
    assert wall["details"]["failure_event"]["gate"] == (
        "rank_condition_pivot"
    )
    for name in POST_ACCELERATION_GROUPS[2:]:
        assert result["groups"][name]["blocked_by"] == (
            "acceleration_closure"
        )


def test_composed_scientific_stops_are_valid_ordered_adjudicator_inputs():
    identities = {"N0": PARENT_IDENTITY, "N1": "b"*64}
    pre = {
        parent: {
            name: {
                "complete": True,
                "provenance_valid": True,
                "passed": True,
                "fingerprint": _sha(1000 + index),
                "parent_label": parent,
                "parent_identity": identities[parent],
            }
            for index, name in enumerate(PRE_ACCELERATION_GROUPS)
        }
        for parent in ("N0", "N1")
    }
    n1_post = {
        name: {
            "complete": True,
            "provenance_valid": True,
            "passed": True,
            "fingerprint": _sha(1100 + index),
            "parent_label": "N1",
            "parent_identity": identities["N1"],
        }
        for index, name in enumerate(POST_ACCELERATION_GROUPS)
    }
    freeze = {
        "status": "FROZEN",
        "frozen_before_parent_data": True,
        "independent_review_passed": True,
        "scientific_candidates_absent_at_freeze": True,
        "protocol_sha256": _sha(1200),
        "adjudicator_sha256": _sha(1201),
    }
    for wall in (False, True):
        inputs = Protocol125PostAccelerationFailureInputs(
            pre_acceleration_result=_pre_acceleration_result(),
            acceleration_failure_record=_scientific_acceleration_failure(
                wall=wall,
            ),
        )
        provenance = capture_protocol125_acceleration_failure_provenance(inputs)
        composed = compose_protocol125_acceleration_failure_records(
            inputs, provenance,
        )
        adjudicated = adjudicate_protocol125_ordered(
            pre,
            parent_identities=identities,
            protocol_freeze_record=freeze,
            post_acceleration_records={
                "N0": composed["groups"],
                "N1": n1_post,
            },
        )
        assert adjudicated["classification"] == "FAIL-acceleration"
        assert adjudicated["invalid_reasons"] == ()
        assert adjudicated["downstream_absence_is_ordered_stop"]


def test_failure_composer_rejects_tamper_identity_and_failed_prerequisite():
    mutable_failure = dict(_scientific_acceleration_failure())
    inputs = Protocol125PostAccelerationFailureInputs(
        pre_acceleration_result=_pre_acceleration_result(),
        acceleration_failure_record=mutable_failure,
    )
    provenance = capture_protocol125_acceleration_failure_provenance(inputs)
    mutable_failure["maps_completed"] = 7
    tampered = compose_protocol125_acceleration_failure_records(
        inputs, provenance,
    )
    assert tampered["classification"] == "INVALID-audit"
    assert not tampered["complete"] and not tampered["provenance_valid"]
    assert "input hash mismatch" in tampered["invalid_reasons"][0]

    wrong_identity = dict(_scientific_acceleration_failure())
    wrong_identity["parent_identity"] = "b"*64
    identity_inputs = Protocol125PostAccelerationFailureInputs(
        pre_acceleration_result=_pre_acceleration_result(),
        acceleration_failure_record=wrong_identity,
    )
    identity_provenance = capture_protocol125_acceleration_failure_provenance(
        identity_inputs,
    )
    identity = compose_protocol125_acceleration_failure_records(
        identity_inputs, identity_provenance,
    )
    assert identity["classification"] == "INVALID-audit"

    failed_pre = dict(_pre_acceleration_result())
    failed_pre["passed"] = False
    failed_inputs = Protocol125PostAccelerationFailureInputs(
        pre_acceleration_result=failed_pre,
        acceleration_failure_record=_scientific_acceleration_failure(),
    )
    failed_provenance = capture_protocol125_acceleration_failure_provenance(
        failed_inputs,
    )
    failed = compose_protocol125_acceleration_failure_records(
        failed_inputs, failed_provenance,
    )
    assert failed["classification"] == "INVALID-audit"
    assert "successful prerequisite" in failed["invalid_reasons"][0]
