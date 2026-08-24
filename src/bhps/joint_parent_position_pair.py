"""Position-only constrained Q53/Q33 pair for Protocol 125.

The acceleration-bearing constrained pair cannot exist at the bulk
prerequisite stage.  This module provides the corresponding position-only
container: it degree-switches only the compact Hermite basis while preserving
the exact radial coefficients, wall contract, outer contract, ownership, and
stored endpoint data.  It performs no parent construction or correction.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from bhps.joint_parent_representation import RadialFirstConstrainedHermiteState


def _bitwise_equal(left, right):
    left = np.ascontiguousarray(np.asarray(left))
    right = np.ascontiguousarray(np.asarray(right))
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes() == right.tobytes()
    )


def _update_digest(digest, name, value):
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(str(name).encode())
    digest.update(b"\0")
    digest.update(str(array.shape).encode())
    digest.update(array.dtype.str.encode())
    digest.update(array.tobytes())


def _fingerprint_arrays(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        _update_digest(digest, name, value)
    return digest.hexdigest()


def _source_arrays(state):
    return {
        "source_z": state.source_z,
        "source_r": state.source_r,
        **state.radial_channels.coefficient_arrays("radial_channels"),
        **state.radial_anisotropy_numerator.coefficient_arrays(
            "radial_anisotropy_numerator"
        ),
    }


def _endpoint_arrays(state):
    arrays = {
        "stored_z_first_endpoints": state.stored_z_first_endpoints,
        "outer_ownership_mask": state.outer_ownership_mask,
        "compact_wall_contract_fingerprint": np.asarray(
            state.compact_wall_contract_fingerprint
        ),
        "outer_open_face_contract_fingerprint": np.asarray(
            state.outer_open_face_contract_fingerprint
        ),
    }
    for prefix, record in (
        ("compact", state.compact_wall_contract_record),
        ("outer", state.outer_open_face_contract_record),
    ):
        for index, (name, value) in enumerate(record):
            arrays[f"{prefix}_{index}_{name}"] = value
    return arrays


@dataclass(frozen=True)
class PositionOnlyConstrainedHermitePair:
    """Authoritative position Q53 plus identical-input position Q33."""

    primary: RadialFirstConstrainedHermiteState
    comparator: RadialFirstConstrainedHermiteState

    def __post_init__(self):
        if self.primary.state_name != "position" or self.primary.z_degree != 5:
            raise ValueError("position-only primary must be a Q53 position state")
        if self.comparator.state_name != "position" or self.comparator.z_degree != 3:
            raise ValueError("position-only comparator must be a Q33 position state")
        primary_source = _source_arrays(self.primary)
        comparator_source = _source_arrays(self.comparator)
        primary_endpoint = _endpoint_arrays(self.primary)
        comparator_endpoint = _endpoint_arrays(self.comparator)
        if set(primary_source) != set(comparator_source) or not all(
            _bitwise_equal(primary_source[name], comparator_source[name])
            for name in primary_source
        ):
            raise ValueError("position-only Q53/Q33 source bundles differ bitwise")
        if set(primary_endpoint) != set(comparator_endpoint) or not all(
            _bitwise_equal(primary_endpoint[name], comparator_endpoint[name])
            for name in primary_endpoint
        ):
            raise ValueError("position-only Q53/Q33 endpoint bundles differ bitwise")
        object.__setattr__(
            self, "source_fingerprint", _fingerprint_arrays(primary_source),
        )
        object.__setattr__(
            self, "endpoint_fingerprint", _fingerprint_arrays(primary_endpoint),
        )

    @classmethod
    def from_primary(cls, primary):
        """Build Q33 by changing only the compact degree of a frozen Q53."""
        if not isinstance(primary, RadialFirstConstrainedHermiteState):
            raise TypeError("position-only primary must be a constrained Hermite state")
        comparator = RadialFirstConstrainedHermiteState(
            primary.source_z,
            primary.source_r,
            primary.radial_channels,
            primary.radial_anisotropy_numerator,
            primary.stored_z_first_endpoints,
            primary.compact_wall_contract,
            primary.outer_open_face_contract,
            primary.outer_ownership_mask,
            "position",
            3,
            primary.compact_wall_contract_record,
            primary.outer_open_face_contract_record,
        )
        return cls(primary, comparator)

    def coefficient_arrays(self, prefix="position_only_constrained_pair"):
        return {
            **self.primary.coefficient_arrays(f"{prefix}_primary"),
            **self.comparator.coefficient_arrays(f"{prefix}_comparator"),
            f"{prefix}_source_fingerprint": np.asarray(self.source_fingerprint),
            f"{prefix}_endpoint_fingerprint": np.asarray(self.endpoint_fingerprint),
        }

    def fingerprint(self):
        return _fingerprint_arrays(self.coefficient_arrays())

    @classmethod
    def from_arrays(
        cls,
        archive,
        prefix="position_only_constrained_pair",
        *,
        compact_wall_contract,
        outer_open_face_contract,
    ):
        pair = cls(
            RadialFirstConstrainedHermiteState.from_arrays(
                archive,
                f"{prefix}_primary",
                compact_wall_contract=compact_wall_contract,
                outer_open_face_contract=outer_open_face_contract,
            ),
            RadialFirstConstrainedHermiteState.from_arrays(
                archive,
                f"{prefix}_comparator",
                compact_wall_contract=compact_wall_contract,
                outer_open_face_contract=outer_open_face_contract,
            ),
        )
        if str(archive[f"{prefix}_source_fingerprint"]) != pair.source_fingerprint:
            raise ValueError("persisted position-only source fingerprint differs")
        if str(archive[f"{prefix}_endpoint_fingerprint"]) != pair.endpoint_fingerprint:
            raise ValueError("persisted position-only endpoint fingerprint differs")
        return pair
