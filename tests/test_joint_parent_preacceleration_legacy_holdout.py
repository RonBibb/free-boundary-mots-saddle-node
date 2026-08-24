from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import bhps.joint_parent_legacy_holdout as legacy
from bhps.joint_parent_position_pair import PositionOnlyConstrainedHermitePair
from bhps.joint_parent_representation import (
    NATIVE_CHANNEL_ORDER,
    OuterOpenFaceDerivativeResult,
    RadialFirstConstrainedHermiteState,
    SEALED_ADVERSE_COMPARATOR_NAMES,
    SEALED_ADVERSE_COMPARATOR_RECIPES,
    sealed_reduced_q33_q55_adverse_projections,
)
from bhps.matched_staged_continuum import REDUCED_FIELD_ORDER, hash_arrays


class _ZeroCompactContract:
    identifier = "test-preacceleration-legacy-zero-compact"

    def z_first_s_jets(self, *, state_name, radius, wall_value_s_jets):
        assert state_name == "position"
        return tuple(np.zeros_like(value) for value in wall_value_s_jets)

    def coefficient_arrays(self):
        return {"recipe": np.asarray("exact-zero-position-z-first")}


class _ZeroOuterContract:
    identifier = "test-preacceleration-legacy-zero-outer"

    def r_first_z_jets(self, *, state_name, compact_coordinate, outer_value_z_jets):
        assert state_name == "position"
        return OuterOpenFaceDerivativeResult(
            tuple(np.zeros_like(value) for value in outer_value_z_jets),
            np.ones(len(NATIVE_CHANNEL_ORDER), dtype=bool),
        )

    def coefficient_arrays(self):
        return {
            "recipe": np.asarray("exact-zero-position-r-first"),
            "ownership": np.ones(len(NATIVE_CHANNEL_ORDER), dtype=bool),
        }


def _position_pair():
    z = np.linspace(1.0, np.e, 9)
    r = np.linspace(0.0, 12.0, 11)
    x = ((z-1.0)/(np.e-1.0))[:, None]
    s = (r/12.0)[None, :]**2
    shape = (len(z), len(r))
    h_perp = 1.0+0.02*x+0.01*s+0.003*x*s
    q4 = 0.004+0.002*x+0.001*s+0.0005*x**2*s
    fields = {
        "h00": -1.1+0.01*x+0.002*s+np.zeros(shape),
        "h_perp": h_perp.copy(),
        "h_rr": h_perp+r[None, :]**2*q4,
        "h_zz": 1.2-0.015*x+0.004*s+np.zeros(shape),
        "Phi": 0.03+0.007*x**2-0.002*s+np.zeros(shape),
        "chi": -0.02+0.005*x*s+np.zeros(shape),
        "v_z": 0.01*x+0.003*s+np.zeros(shape),
        "v_0": -0.006*x**2+0.002*s+np.zeros(shape),
    }
    endpoints = {
        name: np.zeros((2, len(r))) for name in NATIVE_CHANNEL_ORDER
    }
    state = RadialFirstConstrainedHermiteState.build_position(
        z,
        r,
        fields,
        endpoints,
        compact_wall_contract=_ZeroCompactContract(),
        outer_open_face_contract=_ZeroOuterContract(),
        parent_r_max=12.0,
        z_degree=5,
    )
    return PositionOnlyConstrainedHermitePair.from_primary(state)


def _small_frozen_meshes():
    specifications = {
        "V0": (7, 8),
        "V1": (8, 9),
        "V2": (9, 10),
    }
    result = {}
    for name, (nz, nr) in specifications.items():
        z = np.linspace(1.0, np.e, nz)
        r = np.linspace(0.0, 12.0, nr)
        result[name] = {"z": z, "r": r, "sha256": hash_arrays(z, r)}
    return result


def _expected_groups(projected):
    return {
        "position": np.asarray(projected.reduced_fields),
        "first_spatial": np.stack(
            (projected.reduced_first[1], projected.reduced_first[2]), axis=-1,
        ),
        "second_spatial": np.stack(
            (
                projected.reduced_second[1, 1],
                projected.reduced_second[1, 2],
                projected.reduced_second[2, 2],
            ),
            axis=-1,
        ),
    }


def test_position_only_adapter_exactly_reproduces_sealed_spatial_lanes(monkeypatch):
    pair = _position_pair()
    meshes = _small_frozen_meshes()
    monkeypatch.setattr(legacy, "frozen_validation_meshes", lambda: meshes)
    state = pair.primary
    z = np.asarray(state.source_z)
    r = np.asarray(state.source_r)
    q = state.evaluate_reduced(z, r)
    q_z = state.evaluate_reduced(z, r, z_order=1)
    first = np.zeros((3, *q.shape))
    second = np.zeros((3, 3, *q.shape))
    first[0] = 0.37+0.11*q
    first[1] = q_z
    second[0, 0] = -0.23+0.07*q
    second[0, 1] = second[1, 0] = 0.19-0.03*q
    source_jet = SimpleNamespace(
        z=z,
        r=r,
        reduced_fields=q,
        reduced_first=first,
        reduced_second=second,
    )
    identity = hash_arrays(z, r, q, q_z)
    found = legacy.build_protocol125_preacceleration_legacy_position_inputs(
        pair,
        parent_identity=identity,
        v_meshes=meshes,
    )

    for mesh_name, mesh in meshes.items():
        expected = sealed_reduced_q33_q55_adverse_projections(
            source_jet,
            z,
            r,
            mesh["z"],
            mesh["r"],
            parent_identity=identity,
        )
        for comparator in SEALED_ADVERSE_COMPARATOR_NAMES:
            expected_groups = _expected_groups(expected[comparator])
            actual_groups = found[
                "legacy_Q33_by_mesh"
                if comparator == SEALED_ADVERSE_COMPARATOR_NAMES[0]
                else "legacy_Q55_by_mesh"
            ][mesh_name]
            assert tuple(actual_groups) == (
                "position", "first_spatial", "second_spatial",
            )
            for group in actual_groups:
                np.testing.assert_allclose(
                    actual_groups[group], expected_groups[group], rtol=0.0, atol=0.0,
                )


def test_position_only_adapter_is_immutable_provenanced_and_never_enters_acceleration(
    monkeypatch,
):
    pair = _position_pair()
    meshes = _small_frozen_meshes()
    monkeypatch.setattr(legacy, "frozen_validation_meshes", lambda: meshes)

    def forbidden(*args, **kwargs):
        raise AssertionError("pre-acceleration adapter entered acceleration path")

    monkeypatch.setattr(legacy, "represented_position_jet", forbidden)
    monkeypatch.setattr(
        legacy, "sealed_reduced_q33_q55_adverse_projections", forbidden,
    )
    monkeypatch.setattr(
        legacy, "initial_driver_source_triplet_from_acceleration", forbidden,
    )
    identity = "a"*64
    before = pair.fingerprint()
    found = legacy.build_protocol125_preacceleration_legacy_position_inputs(
        pair,
        parent_identity=identity,
        v_meshes=meshes,
    )
    after = pair.fingerprint()

    assert before == after
    assert found["complete"] and found["provenance_valid"] and found["passed"]
    assert found["parent_identity"] == identity
    assert tuple(found["comparator_names"]) == SEALED_ADVERSE_COMPARATOR_NAMES
    assert found["comparator_recipes"] == SEALED_ADVERSE_COMPARATOR_RECIPES
    assert tuple(found["evaluated_groups"]) == (
        "position", "first_spatial", "second_spatial",
    )
    assert tuple(found["component_orders"]) == tuple(found["evaluated_groups"])
    assert tuple(found["component_orders"]["position"]) == tuple(
        REDUCED_FIELD_ORDER
    )
    assert len(found["component_orders"]["first_spatial"]) == 18
    assert len(found["component_orders"]["second_spatial"]) == 27
    assert len(found["fingerprint"]) == 64
    assert found["inputs_stable_while_scoring"]
    assert found["position_only"]
    assert not found["acceleration_evaluated"]
    assert not found["source_triplets_evaluated"]
    assert not found["artifact_written"]
    assert not found["phase_a_authorized"]
    assert not found["scientific_execution_authorized"]
    assert "legacy_Q33_source_triplets_by_mesh" not in found
    assert "legacy_Q55_source_triplets_by_mesh" not in found
    for comparator in ("legacy_Q33_by_mesh", "legacy_Q55_by_mesh"):
        for groups in found[comparator].values():
            assert "acceleration" not in groups
            assert all(not value.flags.writeable for value in groups.values())
    with pytest.raises(ValueError):
        found["legacy_Q33_by_mesh"]["V0"]["position"][0, 0, 0] = 1.0


def test_position_only_adapter_rejects_nonpair_and_nonfrozen_mesh(monkeypatch):
    pair = _position_pair()
    meshes = _small_frozen_meshes()
    monkeypatch.setattr(legacy, "frozen_validation_meshes", lambda: meshes)
    with pytest.raises(TypeError, match="position-only"):
        legacy.build_protocol125_preacceleration_legacy_position_inputs(
            SimpleNamespace(primary=pair.primary), parent_identity="b"*64,
            v_meshes=meshes,
        )
    corrupted = {
        name: {
            "z": values["z"].copy(),
            "r": values["r"].copy(),
            "sha256": values["sha256"],
        }
        for name, values in meshes.items()
    }
    corrupted["V1"]["r"][3] += 1e-6
    with pytest.raises(ValueError, match="differs from Protocol 125"):
        legacy.build_protocol125_preacceleration_legacy_position_inputs(
            pair, parent_identity="b"*64, v_meshes=corrupted,
        )

