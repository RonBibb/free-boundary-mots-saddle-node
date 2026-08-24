"""Canonical input adapter for the Protocol-125 final matrix.

The final representation scorer intentionally accepts explicit, precomputed
source-triplet and legacy arrays.  This module supplies their sole production
construction route from one already completed shared Q53/Q33 pair.  It does
not construct a parent, solve an acceleration, write an artifact, or authorize
scientific execution.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from bhps.joint_parent_acceleration import represented_position_jet
from bhps.joint_parent_bulk_reference import FiniteWallReferenceHermitePair
from bhps.joint_parent_final_matrix import (
    Protocol125FinalMatrixInputs,
    capture_protocol125_final_matrix_provenance,
    protocol125_final_matrix_input_hashes,
)
from bhps.joint_parent_legacy_holdout import (
    build_protocol125_legacy_holdout_inputs,
)
from bhps.joint_parent_position_audits import (
    bind_protocol125_position_audit_meshes,
)
from bhps.joint_parent_refinement_diagnostics import frozen_validation_meshes
from bhps.joint_parent_representation import (
    NATIVE_CHANNEL_ORDER,
    RadialFirstConstrainedHermitePair,
)
from bhps.joint_parent_shared_representation import (
    Protocol125SharedRepresentationBuild,
)
from bhps.joint_parent_source_closure import (
    initial_driver_source_triplet_from_acceleration,
)
from bhps.matched_staged_continuum import hash_arrays


PROTOCOL_IDENTIFIER = "Protocol-125-final-matrix-input-adapter-v1"
PARENT_LABELS = ("N0", "N1")
V_MESH_NAMES = ("V0", "V1", "V2")
SOURCE_TRIPLET_KEYS = ("source", "source_time", "source_second_time")


def _valid_sha256(value):
    value = str(value)
    return bool(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _immutable(value):
    array = np.ascontiguousarray(np.asarray(value))
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({str(name): _freeze(item) for name, item in value.items()})
    if isinstance(value, np.ndarray):
        return _immutable(value)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _fingerprint_tree(value):
    digest = hashlib.sha256()

    def visit(item, path):
        label = str(path).encode("utf-8")
        digest.update(len(label).to_bytes(8, "little"))
        digest.update(label)
        if isinstance(item, Mapping):
            for name in sorted(item):
                visit(item[name], f"{path}/{name}")
            return
        if isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                visit(child, f"{path}/{index}")
            return
        array = np.ascontiguousarray(np.asarray(item))
        if array.dtype == object:
            raise ValueError("final-input provenance cannot hash object arrays")
        if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
            raise ValueError("final-input provenance contains nonfinite data")
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(array.tobytes())

    visit(value, "final-input-adapter")
    return digest.hexdigest()


def _state_triplets(position_state, acceleration_state, meshes, background):
    records = {}
    for mesh_name in V_MESH_NAMES:
        mesh = meshes[mesh_name]
        z = np.asarray(mesh["z"], dtype=float)
        r = np.asarray(mesh["r"], dtype=float)
        acceleration = acceleration_state.evaluate_reduced(z, r)
        jet = represented_position_jet(
            position_state, z, r, acceleration,
        )
        triplet = initial_driver_source_triplet_from_acceleration(
            jet, z, r, background,
        )
        records[mesh_name] = {
            name: _immutable(triplet[name]) for name in SOURCE_TRIPLET_KEYS
        }
    return MappingProxyType(records)


@dataclass(frozen=True)
class Protocol125FinalMatrixInputBundle:
    """Complete final-matrix inputs plus their pre-score provenance."""

    inputs: Protocol125FinalMatrixInputs
    provenance: Mapping
    parent_label: str
    parent_identity: str
    adapter_record: Mapping

    def fingerprint(self):
        return str(self.adapter_record["fingerprint"])


def build_protocol125_final_matrix_inputs(
    shared_build,
    reference_pair,
    background,
    *,
    parent_label,
    parent_identity,
):
    """Build every explicit array required by the final matrix scorer."""
    label = str(parent_label)
    identity = str(parent_identity)
    if label not in PARENT_LABELS or not _valid_sha256(identity):
        raise ValueError("final-input parent label or identity is invalid")
    if not isinstance(shared_build, Protocol125SharedRepresentationBuild):
        raise TypeError("final-input adapter requires the complete shared-build record")
    final_pair = shared_build.final_pair
    if not isinstance(final_pair, RadialFirstConstrainedHermitePair):
        raise TypeError("final-input adapter requires the shared Q53/Q33 pair")
    if not isinstance(reference_pair, FiniteWallReferenceHermitePair):
        raise TypeError("final-input adapter requires the finite-wall reference pair")
    if not isinstance(background, Mapping):
        raise TypeError("final-input background must be an explicit mapping")

    states = (
        final_pair.primary.position,
        final_pair.primary.acceleration,
        final_pair.comparator.position,
        final_pair.comparator.acceleration,
    )
    source_z = np.asarray(states[0].source_z, dtype=float)
    source_r = np.asarray(states[0].source_r, dtype=float)
    if any(
        not (
            np.array_equal(np.asarray(state.source_z), source_z)
            and np.array_equal(np.asarray(state.source_r), source_r)
        )
        for state in states[1:]
    ):
        raise ValueError("final-input states do not share a source grid")
    for member in (reference_pair.primary, reference_pair.comparator):
        if not (
            np.array_equal(np.asarray(member.source_z), source_z)
            and np.array_equal(np.asarray(member.source_r), source_r)
        ):
            raise ValueError("final-input reference/source coordinates differ")

    frozen = frozen_validation_meshes()
    meshes = MappingProxyType({
        name: MappingProxyType({
            "z": _immutable(frozen[name]["z"]),
            "r": _immutable(frozen[name]["r"]),
            "sha256": str(frozen[name]["sha256"]),
        })
        for name in V_MESH_NAMES
    })
    q53_triplets = _state_triplets(
        final_pair.primary.position,
        final_pair.primary.acceleration,
        meshes,
        background,
    )
    q33_triplets = _state_triplets(
        final_pair.comparator.position,
        final_pair.comparator.acceleration,
        meshes,
        background,
    )
    legacy = build_protocol125_legacy_holdout_inputs(
        final_pair,
        background,
        parent_identity=identity,
        v_meshes=meshes,
    )
    if not (
        legacy["complete"]
        and legacy["provenance_valid"]
        and legacy["passed"]
        and str(legacy["parent_identity"]) == identity
    ):
        raise RuntimeError("sealed legacy final-input adapter failed")

    velocity_endpoint_z = np.zeros(
        (2, len(source_r), len(NATIVE_CHANNEL_ORDER)), dtype=float,
    )
    if np.any(np.signbit(velocity_endpoint_z)):
        raise AssertionError("velocity endpoint-z record lost IEEE positive zero")
    audit_meshes = bind_protocol125_position_audit_meshes(final_pair)
    inputs = Protocol125FinalMatrixInputs(
        final_pair=final_pair,
        reference_pair=reference_pair,
        v_meshes=meshes,
        position_audit_meshes=audit_meshes,
        Q53_source_triplets_by_mesh=q53_triplets,
        Q33_source_triplets_by_mesh=q33_triplets,
        velocity_endpoint_z=_immutable(velocity_endpoint_z),
        row_implied_acceleration_z_source=shared_build.row_implied_source,
        row_implied_acceleration_z_dense=shared_build.row_implied_dense,
        direct_Dz7_acceleration_z_source=shared_build.direct_fd_source,
        shared_representation_fingerprint=shared_build.fingerprint(),
        legacy_Q33_by_mesh=legacy["legacy_Q33_by_mesh"],
        legacy_Q55_by_mesh=legacy["legacy_Q55_by_mesh"],
        legacy_Q33_source_triplets_by_mesh=(
            legacy["legacy_Q33_source_triplets_by_mesh"]
        ),
        legacy_Q55_source_triplets_by_mesh=(
            legacy["legacy_Q55_source_triplets_by_mesh"]
        ),
        legacy_component_orders=legacy["component_orders"],
    )
    input_hashes = protocol125_final_matrix_input_hashes(inputs)
    provenance = capture_protocol125_final_matrix_provenance(
        inputs, parent_label=label,
    )
    payload = {
        "protocol_identifier": PROTOCOL_IDENTIFIER,
        "parent_label": label,
        "parent_identity": identity,
        "source_coordinate_sha256": hash_arrays(source_z, source_r),
        "V_mesh_coordinate_sha256": {
            name: meshes[name]["sha256"] for name in V_MESH_NAMES
        },
        "Q53_source_triplets_sha256": _fingerprint_tree(q53_triplets),
        "Q33_source_triplets_sha256": _fingerprint_tree(q33_triplets),
        "legacy_adapter_fingerprint": str(legacy["fingerprint"]),
        "shared_representation_fingerprint": shared_build.fingerprint(),
        "row_implied_acceleration_z_source_fingerprint": str(
            shared_build.row_implied_source["fingerprint"]
        ),
        "row_implied_acceleration_z_dense_fingerprint": str(
            shared_build.row_implied_dense["fingerprint"]
        ),
        "direct_Dz7_acceleration_z_source_fingerprint": str(
            shared_build.direct_fd_source["fingerprint"]
        ),
        "final_matrix_input_hashes": input_hashes,
        "velocity_endpoint_z_positive_zero": True,
        "source_triplet_recipe": (
            "analytic-position-and-acceleration-state-jets;"
            "native-initial-driver-source-triplet"
        ),
        "frozen_V0_V1_V2_only": True,
        "target_repairs_applied": False,
        "artifact_written": False,
        "scientific_execution_authorized": False,
    }
    return Protocol125FinalMatrixInputBundle(
        inputs,
        provenance,
        label,
        identity,
        _freeze({**payload, "fingerprint": _fingerprint_tree(payload)}),
    )
