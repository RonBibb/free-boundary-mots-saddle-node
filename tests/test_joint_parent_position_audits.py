from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from bhps.joint_parent_bulk_reference import FiniteWallReferenceHermitePair
from bhps.joint_parent_adjudication import score_construction_reload_provenance
from bhps.joint_parent_boundary_contracts import (
    derive_protocol125_outer_derivative_bundle,
)
from bhps.joint_parent_endpoint_audits import (
    score_state_endpoint_z_reproduction,
    score_state_outer_derivative_reproduction,
)
from bhps.joint_parent_position_audits import (
    DENSE_OUTER_LINF_CEILING,
    DENSE_WALL_LINF_CEILING,
    SIGNATURE_MARGIN_MINIMUM,
    bind_protocol125_position_audit_meshes,
    evaluate_protocol125_dense_outer_delta_robin_audit,
    evaluate_protocol125_dense_wall_audit,
    evaluate_protocol125_signature_union,
)
from bhps.joint_parent_position_pair import PositionOnlyConstrainedHermitePair
from bhps.joint_parent_position_state import (
    build_joint_parent_position_state,
    derive_joint_parent_position_outer_contract,
)
from bhps.joint_parent_representation import (
    COORDINATE_COMPONENT_ORDER,
    NATIVE_CHANNEL_ORDER,
)
from bhps.joint_parent_refinement_diagnostics import frozen_validation_meshes


def _flat_position_pair(*, h00=-1.0):
    z = np.linspace(1.0, np.e, 9)
    r = np.linspace(0.0, 12.0, 17)
    shape = (len(z), len(r))
    position = np.zeros((*shape, 9))
    position[:, :, 2] = float(h00)
    position[:, :, 3] = 1.0
    position[:, :, 6] = 1.0
    selector_q = np.broadcast_to((1.0-z)[:, None], shape).copy()
    zeros = np.zeros(shape)
    parent = {
        "z": z,
        "r": r,
        "position": position,
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
    background = {
        "wall_stiffness": 0.0,
        "v0": 0.0,
        "v1": 0.0,
        "beta_a": 0.0,
        "beta_b": 0.0,
        "wall_potential_a": 0.0,
        "wall_potential_b": 0.0,
    }
    outer = derive_joint_parent_position_outer_contract(parent)
    state, _ = build_joint_parent_position_state(
        position,
        z,
        r,
        background,
        outer_open_face_contract=outer,
        parent_r_max=12.0,
    )
    pair = PositionOnlyConstrainedHermitePair.from_primary(state)
    reference = FiniteWallReferenceHermitePair.build(
        z, r, selector_q, zeros,
    )
    return pair, reference


class _AdversePrimary:
    def __init__(self, state, mode):
        self._state = state
        self._mode = str(mode)

    def __getattr__(self, name):
        return getattr(self._state, name)

    def evaluate_coordinate_components(self, z, r, *, z_order=0, r_order=0):
        values = np.asarray(self._state.evaluate_coordinate_components(
            z, r, z_order=z_order, r_order=r_order,
        )).copy()
        if self._mode == "wall" and int(z_order) == 1 and int(r_order) == 0:
            values[:, :, COORDINATE_COMPONENT_ORDER.index("h_perp")] += 2e-6
        return values

    def evaluate_physical_channels(self, z, r, *, z_order=0, r_order=0):
        values = np.asarray(self._state.evaluate_physical_channels(
            z, r, z_order=z_order, r_order=r_order,
        )).copy()
        if self._mode == "outer" and int(r_order) == 1 and int(z_order) == 0:
            values[:, :, NATIVE_CHANNEL_ORDER.index("Phi")] += 2e-6
        return values


class _AdversePair:
    def __init__(self, pair, mode):
        self._pair = pair
        self.primary = _AdversePrimary(pair.primary, mode)
        self.comparator = pair.comparator
        self.source_fingerprint = pair.source_fingerprint
        self.endpoint_fingerprint = pair.endpoint_fingerprint

    def fingerprint(self):
        return self._pair.fingerprint()


def test_signature_union_contains_every_required_domain_and_passes_flat_metric():
    pair, _ = _flat_position_pair()
    meshes = bind_protocol125_position_audit_meshes(pair)
    audit = evaluate_protocol125_signature_union(pair, meshes)
    assert audit["passed"]
    assert audit["domain_order"] == (
        "source",
        "source_cell_midpoint",
        "V0",
        "V1",
        "V2",
        "dense_wall_lower",
        "dense_wall_upper",
        "dense_outer",
    )
    assert audit["union_wrong_signature_sample_count"] == 0
    assert audit["union_minimum_dimensionless_margin"] == pytest.approx(1.0)
    assert audit["minimum_margin_threshold"] == SIGNATURE_MARGIN_MINIMUM
    assert audit["no_face_axis_collar_or_corner_exclusion"]


def test_signature_union_rejects_a_lorentzian_metric_with_too_small_margin():
    pair, _ = _flat_position_pair(h00=-1e-12)
    meshes = bind_protocol125_position_audit_meshes(pair)
    audit = evaluate_protocol125_signature_union(pair, meshes)
    assert not audit["passed"]
    assert audit["union_wrong_signature_sample_count"] == 0
    assert audit["union_minimum_dimensionless_margin"] < SIGNATURE_MARGIN_MINIMUM


def test_independent_dense_wall_audit_passes_and_catches_derivative_corruption():
    pair, _ = _flat_position_pair()
    meshes = bind_protocol125_position_audit_meshes(pair)
    audit = evaluate_protocol125_dense_wall_audit(pair, meshes)
    assert audit["passed"]
    assert audit["combined_metric_absolute_J_Linf"] < DENSE_WALL_LINF_CEILING
    assert not audit["stored_residual_or_target_array_comparison_used"]
    assert audit["live_compact_contract_used_by_candidate_evaluator"]

    adverse = _AdversePair(pair, "wall")
    adverse_meshes = bind_protocol125_position_audit_meshes(adverse)
    failed = evaluate_protocol125_dense_wall_audit(adverse, adverse_meshes)
    assert not failed["passed"]
    assert not failed["metric"]["h_perp"]["lower"]["passed"]
    assert not failed["metric"]["h_perp"]["upper"]["passed"]


def test_independent_dense_outer_delta_robin_passes_and_catches_phi_r_error():
    pair, reference = _flat_position_pair()
    meshes = bind_protocol125_position_audit_meshes(pair)
    audit = evaluate_protocol125_dense_outer_delta_robin_audit(
        pair, reference, meshes,
    )
    assert audit["passed"]
    assert audit["q"]["normalized_Linf"] < DENSE_OUTER_LINF_CEILING
    assert audit["Phi"]["normalized_Linf"] < DENSE_OUTER_LINF_CEILING
    assert not audit["contract_target_query_called"]
    assert audit["compact_endpoints_excluded_from_score"]

    adverse = _AdversePair(pair, "outer")
    adverse_meshes = bind_protocol125_position_audit_meshes(adverse)
    failed = evaluate_protocol125_dense_outer_delta_robin_audit(
        adverse, reference, adverse_meshes,
    )
    assert not failed["passed"]
    assert not failed["Phi"]["passed"]


def test_audits_fail_closed_on_missing_or_changed_mesh_provenance():
    pair, _ = _flat_position_pair()
    with pytest.raises(TypeError, match="complete"):
        evaluate_protocol125_signature_union(pair, None)
    meshes = bind_protocol125_position_audit_meshes(pair)
    changed_v2 = meshes.V2_r.copy()
    changed_v2[1] = np.nextafter(changed_v2[1], np.inf)
    with pytest.raises(ValueError, match="V2 mesh"):
        replace(meshes, V2_r=changed_v2)

    other, _ = _flat_position_pair(h00=-0.5)
    with pytest.raises(ValueError, match="identity changed"):
        evaluate_protocol125_signature_union(other, meshes)


def test_construction_provenance_reloads_reference_and_position_bitwise():
    pair, reference = _flat_position_pair()
    record = {
        "finite_wall_maximum_residual": 1e-12,
        "joint_hybrid_maximum_residual": 2e-12,
        "input_fingerprint_before": "input-hash",
        "input_fingerprint_after": "input-hash",
        "physical_normalization_identifier": "poincare-ads-normalization",
        "branch_identifier": "manufactured-flat-branch",
        "expected_parent_label": "synthetic",
        "actual_parent_label": "synthetic",
    }
    passed = score_construction_reload_provenance(record, reference, pair)
    assert passed["passed"]
    assert passed["provenance_valid"]
    assert passed["gates"]["reference_reload_bitwise"]
    assert passed["gates"]["position_reload_bitwise"]

    changed = {**record, "joint_hybrid_maximum_residual": 2e-9}
    failed = score_construction_reload_provenance(changed, reference, pair)
    assert not failed["passed"]
    assert failed["provenance_valid"]
    assert not failed["gates"]["joint_hybrid_residual"]


def test_position_endpoint_z_reproduction_closes_source_and_dense_wall():
    pair, _ = _flat_position_pair()
    dense_r = frozen_validation_meshes()["dense_wall"]["r"]
    found = score_state_endpoint_z_reproduction(pair.primary, dense_r)
    assert found["passed"]
    assert found["state_name"] == "position"
    assert all(found["gates"].values())
    assert found["dense_analytic_vs_live_contract"]["scaled_Linf"] <= 1e-10


def test_position_endpoint_z_reproduction_rejects_nonfrozen_dense_wall():
    pair, _ = _flat_position_pair()
    dense_r = frozen_validation_meshes()["dense_wall"]["r"].copy()
    dense_r[1] = np.nextafter(dense_r[1], np.inf)
    with pytest.raises(ValueError, match="frozen wall mesh"):
        score_state_endpoint_z_reproduction(pair.primary, dense_r)


def test_position_outer_derivative_reproduction_is_independent_and_all_channel():
    pair, _ = _flat_position_pair()
    dense_z = frozen_validation_meshes()["dense_outer"]["z"]
    found = score_state_outer_derivative_reproduction(pair.primary, dense_z)
    assert found["passed"]
    assert found["state_name"] == "position"
    assert not found["contract_target_query_called"]
    assert found["fresh_degree_five_target_reconstruction"]
    assert tuple(found["source_records"]) == NATIVE_CHANNEL_ORDER
    assert tuple(found["dense_records"]) == NATIVE_CHANNEL_ORDER
    assert all(found["gates"].values())


class _AnalyticAccelerationOuterState:
    state_name = "acceleration"

    def __init__(self, position_state):
        self.source_z = position_state.source_z
        self.source_r = position_state.source_r
        lanes = np.arange(len(NATIVE_CHANNEL_ORDER), dtype=float)[None, :]
        source_coefficient = (
            0.03+0.004*self.source_z[:, None]**2
        )*(1.0+0.01*lanes)
        source = source_coefficient[:, None, :]*self.source_r[None, :, None]
        mapping = {
            name: source[:, :, index]
            for index, name in enumerate(NATIVE_CHANNEL_ORDER)
        }
        self.outer_open_face_contract = derive_protocol125_outer_derivative_bundle(
            position_state.outer_open_face_contract, mapping,
        )

    def evaluate_physical_channels(self, z, r, *, z_order=0, r_order=0):
        z = np.asarray(z, dtype=float)
        r = np.asarray(r, dtype=float)
        lanes = np.arange(len(NATIVE_CHANNEL_ORDER), dtype=float)[None, :]
        if z_order == 0:
            coefficient = (0.03+0.004*z[:, None]**2)*(1.0+0.01*lanes)
        elif z_order == 1:
            coefficient = (0.008*z[:, None])*(1.0+0.01*lanes)
        elif z_order == 2:
            coefficient = np.broadcast_to(0.008*(1.0+0.01*lanes), (len(z), len(NATIVE_CHANNEL_ORDER)))
        else:
            coefficient = np.zeros((len(z), len(NATIVE_CHANNEL_ORDER)))
        if r_order == 0:
            return coefficient[:, None, :]*r[None, :, None]
        if r_order == 1:
            return np.broadcast_to(coefficient[:, None, :], (len(z), len(r), len(NATIVE_CHANNEL_ORDER))).copy()
        return np.zeros((len(z), len(r), len(NATIVE_CHANNEL_ORDER)))


class _AdverseAccelerationOuterState(_AnalyticAccelerationOuterState):
    def evaluate_physical_channels(self, z, r, *, z_order=0, r_order=0):
        found = super().evaluate_physical_channels(
            z, r, z_order=z_order, r_order=r_order,
        )
        if int(z_order) == 0 and int(r_order) == 1:
            found[:, :, NATIVE_CHANNEL_ORDER.index("Phi")] += 2e-6
        return found


def test_acceleration_outer_derivative_reproduction_catches_live_phi_error():
    pair, _ = _flat_position_pair()
    dense_z = frozen_validation_meshes()["dense_outer"]["z"]
    state = _AnalyticAccelerationOuterState(pair.primary)
    passed = score_state_outer_derivative_reproduction(state, dense_z)
    assert passed["passed"]
    assert passed["state_name"] == "acceleration"

    adverse = _AdverseAccelerationOuterState(pair.primary)
    failed = score_state_outer_derivative_reproduction(adverse, dense_z)
    assert not failed["passed"]
    assert not failed["source_records"]["Phi"]["passed"]
    assert not failed["dense_records"]["Phi"]["passed"]
