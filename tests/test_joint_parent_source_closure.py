from __future__ import annotations

import numpy as np
import pytest

from bhps.joint_parent_source_closure import (
    SOURCE_SECOND_TIME_DIFFERENCE_STEP,
    initial_driver_source_triplet_from_acceleration,
)
from bhps.matched_staged_continuum import DriverConfiguration, ProjectedJetField


def _flat_parent_jet(nz=9, nr=17):
    z = np.linspace(1.0, 2.0, nz)
    r = np.linspace(0.0, 2.0, nr)
    q = np.zeros((nz, nr, 9))
    q[:, :, 2] = -1.0
    q[:, :, 3] = 1.0
    q[:, :, 6] = 1.0
    first = np.zeros((3, *q.shape))
    second = np.zeros((3, 3, *q.shape))
    zz, rr = np.meshgrid(z, r, indexing="ij")
    second[0, 0, :, :, 2] = 0.02+0.003*zz+0.001*rr**2
    second[0, 0, :, :, 3] = -0.01+0.002*zz
    second[0, 0, :, :, 6] = 0.015+0.001*rr**2
    return z, r, ProjectedJetField(z, r, q, first, second)


def _background():
    return {
        "wall_stiffness": 0.0,
        "v0": 0.0,
        "v1": 0.0,
        "beta_a": 0.0,
        "beta_b": 0.0,
        "wall_potential_a": 0.0,
        "wall_potential_b": 0.0,
    }


def test_source_triplet_reconstructs_frozen_driver_map_without_outer_write():
    z, r, jet = _flat_parent_jet()
    found = initial_driver_source_triplet_from_acceleration(
        jet, z, r, _background(),
    )
    shape = (len(z), len(r), 3)
    for name in (
        "source", "source_time", "source_second_time", "memory",
        "memory_time", "target", "advection",
    ):
        assert found[name].shape == shape
        assert np.all(np.isfinite(found[name]))
    assert found["Hdot_reassembly_scaled_Linf"] <= 1e-12
    assert found["normal_wall_completion"]["normal_gauge"]["maximum"] < 1e-12
    assert found["difference_step"] == SOURCE_SECOND_TIME_DIFFERENCE_STEP
    assert found["driver"] == DriverConfiguration().public()
    assert not found["outer_source_overwrite_applied"]
    assert not found["memory_carried_from_previous_iterate"]


def test_source_triplet_requires_positive_zero_velocity_and_exact_coordinates():
    z, r, jet = _flat_parent_jet()
    bad_first = np.asarray(jet.reduced_first).copy()
    bad_first[0, 3, 4, 2] = -0.0
    bad = ProjectedJetField(
        z, r, jet.reduced_fields, bad_first, jet.reduced_second,
    )
    with pytest.raises(ValueError, match="positive-zero velocity"):
        initial_driver_source_triplet_from_acceleration(
            bad, z, r, _background(),
        )
    with pytest.raises(ValueError, match="coordinates differ"):
        initial_driver_source_triplet_from_acceleration(
            jet, z.copy()+1e-15, r, _background(),
        )
