from __future__ import annotations

import inspect

import numpy as np
import pytest

import bhps.joint_parent_validate_only as validate_module
from bhps.joint_parent_bulk_reference import FiniteWallReferenceHermitePair
from bhps.joint_parent_validate_only import (
    PROTOCOL125_STATUS,
    Protocol125AuthorizationError,
    guard_protocol125_execution,
    run_protocol125_validate_only,
)


def _manufactured_flat(*, reference_radial_curvature=0.0):
    # Poincare-AdS manufactured datum.  Fifty-seven compact nodes make both
    # the source FD7 and analytic constrained-representation raw lanes clear
    # their unchanged bulk thresholds; this is not a reference-only balance.
    z = np.linspace(1.0, np.e, 57)
    r = np.linspace(0.0, 2.0, 17)
    shape = (len(z), len(r))
    psi = np.broadcast_to((1.0/z)[:, None], shape).copy()
    position = np.zeros((*shape, 9))
    position[:, :, 2] = -psi**2
    position[:, :, 3] = psi**2
    position[:, :, 6] = psi**2
    selector_q = np.zeros(shape)
    reference_q = selector_q+float(reference_radial_curvature)*r[None, :]**2
    zeros = np.zeros(shape)
    parent = {
        "label": "manufactured-flat-smoke",
        "z": z,
        "r": r,
        "position": position,
        "selector_q": selector_q,
        "psi_selector": psi,
        "reference_q": reference_q,
        "reference_phi": zeros.copy(),
        "chi": zeros.copy(),
        "chi_r": zeros.copy(),
        **{
            f"shape_{name}": zeros.copy()
            for name in ("a", "b", "c", "a_r", "b_r", "c_r")
        },
    }
    background = {
        "mass_squared": 0.0,
        "wall_stiffness": 0.0,
        "v0": 0.0,
        "v1": 0.0,
        "beta_a": 1.0,
        "beta_b": 1.0,
        "wall_potential_a": 0.0,
        "wall_potential_b": 0.0,
    }
    reference = FiniteWallReferenceHermitePair.build(
        z, r, reference_q, zeros, parent_r_max=2.0,
    )
    return parent, background, reference


@pytest.mark.parametrize(
    "scope,label,phase",
    (
        ("candidate", None, None),
        ("manufactured", "N0", None),
        ("synthetic", "N1", None),
        ("synthetic", "N0-candidate-alias", None),
        ("synthetic", "smoke", "Phase A"),
        ("manufactured", "smoke", "Phase-B"),
        ("manufactured", "smoke", "pre-Phase-A-pilot"),
    ),
)
def test_invalid_protocol_guard_rejects_candidate_n0_n1_and_phase_a_b(
    scope, label, phase,
):
    with pytest.raises(Protocol125AuthorizationError, match="Protocol 125"):
        guard_protocol125_execution(scope, parent_label=label, phase=phase)


def test_validate_only_guard_rejects_canonical_parent_coordinates_under_alias():
    z = np.linspace(1.0, np.e, 145)
    r = np.linspace(0.0, 12.0, 325)
    with pytest.raises(Protocol125AuthorizationError, match="canonical N0"):
        guard_protocol125_execution(
            "manufactured",
            parent_label="not-n0",
            source_z=z,
            source_r=r,
        )


def test_manufactured_pipeline_runs_in_order_and_never_authorizes_candidate():
    parent, background, reference = _manufactured_flat()
    before = {
        name: np.asarray(value).copy()
        for name, value in parent.items()
        if not isinstance(value, str)
    }
    result = run_protocol125_validate_only(
        parent,
        background,
        reference,
        execution_scope="manufactured",
    )
    assert result.classification == "VALIDATE-ONLY-SMOKE-PASS"
    assert result.validate_only_smoke_pass
    assert result.acceleration_started
    assert result.position_pair is not None
    assert result.final_pair is not None
    assert result.stage_order == (
        "pre_acceleration_construction",
        "position_only_representation",
        "bulk_prerequisite_smoke",
        "post_prerequisite_acceleration",
        "final_shared_representation",
        "validate_only_adjudication",
    )
    assert all(record["passed"] for record in result.stage_records.values())
    assert result.protocol_gate_complete is False
    assert result.scientific_candidate_authorized is False
    assert result.artifact_written is False
    assert result.missing_protocol_scorers
    assert PROTOCOL125_STATUS == "LEGACY-VALIDATE-ONLY-NON-AUTHORIZING"
    assert not result.bulk_acceleration.flags.writeable
    assert not result.compatible_acceleration.flags.writeable
    for name, value in before.items():
        np.testing.assert_array_equal(parent[name], value)
    signature = inspect.signature(run_protocol125_validate_only)
    assert not ({"path", "output", "writer", "resume"} & set(signature.parameters))


def test_failed_bulk_smoke_stops_before_first_acceleration_call(monkeypatch):
    parent, background, reference = _manufactured_flat(
        reference_radial_curvature=0.02,
    )

    def forbidden_acceleration(*args, **kwargs):
        raise AssertionError("acceleration ran before the bulk prerequisite passed")

    monkeypatch.setattr(
        validate_module,
        "bulk_acceleration_from_completed_position",
        forbidden_acceleration,
    )
    result = run_protocol125_validate_only(
        parent,
        background,
        reference,
        execution_scope="synthetic",
    )
    assert result.classification == "VALIDATE-ONLY-BLOCKED-BULK"
    assert not result.acceleration_started
    assert result.bulk_acceleration is None
    assert result.compatible_acceleration is None
    assert "post_prerequisite_acceleration" not in result.stage_records
    assert not result.stage_records["bulk_prerequisite_smoke"]["passed"]
