from __future__ import annotations

import numpy as np

from bhps.joint_parent_native_completion import complete_native_parent_position
from bhps.joint_parent_native_evidence import (
    build_protocol125_native_position_tangent_evidence,
)
from bhps.joint_parent_position_pair import PositionOnlyConstrainedHermitePair
from bhps.joint_parent_position_state import (
    build_joint_parent_position_state,
    derive_joint_parent_position_outer_contract,
)
from bhps.joint_parent_preacceleration import NATIVE_POSITION_TANGENT_LANES
from bhps.matched_staged_continuum import hash_arrays


def _case():
    z = np.linspace(1.0, np.e, 17)
    r = np.linspace(0.0, 12.0, 17)
    shape = (len(z), len(r))
    raw = np.zeros((*shape, 9))
    raw[:, :, 2] = -1.0
    raw[:, :, 3] = 1.0
    raw[:, :, 6] = 1.0
    background = {
        "wall_stiffness": 0.0,
        "v0": 0.0,
        "v1": 0.0,
        "beta_a": 0.0,
        "beta_b": 0.0,
        "wall_potential_a": 0.0,
        "wall_potential_b": 0.0,
    }
    completed, completion = complete_native_parent_position(
        raw, z, r, background, prerequisite_tolerance=1e-12,
    )
    selector = np.broadcast_to((1.0-z)[:, None], shape).copy()
    zero = np.zeros(shape)
    parent = {
        "label": "N0",
        "z": z,
        "r": r,
        "position": completed,
        "raw_position": raw,
        "selector_q": selector,
        "phi": zero.copy(),
        "reference_q": selector.copy(),
        "reference_phi": zero.copy(),
        "psi_selector": np.ones(shape),
        "chi": zero.copy(),
        "chi_r": zero.copy(),
        "background": background,
        "completion_record": completion,
        **{
            f"shape_{name}": zero.copy()
            for name in ("a", "b", "c", "a_r", "b_r", "c_r")
        },
    }
    parent["parent_identity"] = hash_arrays(
        np.asarray("N0"), z, r, completed, selector, zero, selector, zero,
    )
    outer = derive_joint_parent_position_outer_contract(parent)
    state, state_record = build_joint_parent_position_state(
        completed,
        z,
        r,
        background,
        outer_open_face_contract=outer,
        parent_r_max=12.0,
    )
    return parent, PositionOnlyConstrainedHermitePair.from_primary(state), state_record


def test_native_evidence_builds_every_independent_lane_and_passes_flat_case():
    parent, pair, state_record = _case()
    result = build_protocol125_native_position_tangent_evidence(
        parent, pair, state_record,
    )
    assert result["complete"]
    assert result["provenance_valid"]
    assert result["passed"]
    assert tuple(result["lanes"]) == NATIVE_POSITION_TANGENT_LANES
    assert all(record["passed"] for record in result["lanes"].values())
    assert all(len(record["fingerprint"]) == 64 for record in result["lanes"].values())
    assert not result["acceleration_evaluated"]
    assert not result["scientific_execution_authorized"]
    assert not result["artifact_written"]


def test_native_evidence_records_a_scientific_completion_failure_without_hiding_it():
    parent, pair, state_record = _case()
    parent = dict(parent)
    parent["completion_record"] = dict(parent["completion_record"])
    corrections = dict(parent["completion_record"]["completion_corrections"])
    corrections["chi_normalized_Linf"] = 2e-10
    parent["completion_record"]["completion_corrections"] = corrections
    result = build_protocol125_native_position_tangent_evidence(
        parent, pair, state_record,
    )
    lane = result["lanes"]["native_position_completion"]
    assert lane["complete"]
    assert lane["provenance_valid"]
    assert not lane["passed"]
    assert not lane["details"]["gates"]["recorded_corrections_reproduce_arrays"]
    assert not result["passed"]


def test_native_evidence_detects_undeclared_position_owner_change():
    parent, pair, state_record = _case()
    parent = dict(parent)
    parent["raw_position"] = parent["raw_position"].copy()
    parent["raw_position"][3, 4, 7] = 1e-30
    result = build_protocol125_native_position_tangent_evidence(
        parent, pair, state_record,
    )
    owner = result["lanes"]["source_node_ownership"]
    assert not owner["passed"]
    assert owner["details"]["unauthorized_changed_value_count"] == 1


def test_native_evidence_rejects_parent_identity_tamper():
    parent, pair, state_record = _case()
    parent = dict(parent)
    parent["parent_identity"] = "f"*64
    try:
        build_protocol125_native_position_tangent_evidence(parent, pair, state_record)
    except ValueError as error:
        assert "identity does not reproduce" in str(error)
    else:
        raise AssertionError("tampered parent identity was accepted")
