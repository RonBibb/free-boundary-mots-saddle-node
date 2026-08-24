from __future__ import annotations

import numpy as np
import pytest

from bhps.joint_parent_native_completion import (
    analytic_even_q4_limit,
    complete_native_parent_position,
    complete_normal_gauge_source_wall,
)
from bhps.junction_preservation_diagnostic import wall_junction_rows


def _background(beta=0.0):
    return {
        "wall_stiffness": 0.0,
        "v0": 0.0,
        "v1": 0.0,
        "beta_a": float(beta),
        "beta_b": float(beta),
        "wall_potential_a": 0.0,
        "wall_potential_b": 0.0,
    }


def _position(nz=11, nr=9):
    z = np.linspace(1.0, 2.0, nz)
    r = np.linspace(0.0, 1.0, nr)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    q = np.zeros((nz, nr, 9))
    q[:, :, 2] = -(1.0 + 0.03*zz + 0.01*zz**2)*(1.0+0.02*rr**2)
    q[:, :, 3] = 1.0
    q[:, :, 4] = (0.2 + 0.04*zz + 0.03*zz**2)*(1.0+0.1*rr**2)
    q[:, :, 6] = (1.1 + 0.02*zz)*(1.0+0.01*rr**2)
    q[:, :, 8] = (0.1+0.03*zz+0.02*zz**3)*(1.0+0.2*rr**2)
    return z, r, q


def test_native_completion_closes_owned_rows_and_preserves_open_parent():
    z, r, q = _position()
    before = q.copy()
    completed, record = complete_native_parent_position(
        q, z, r, _background(), prerequisite_tolerance=1e-12,
    )
    zero = np.zeros_like(completed)
    for wall in ("lower", "upper"):
        rows = wall_junction_rows(
            completed, zero, z, r, _background(), wall, 7,
        )
        for name in ("tt", "rr", "sphere", "tr"):
            assert np.max(np.abs(
                rows["components"][name]["robin_normalized"]
            )) < 2e-12
        assert np.max(np.abs(
            rows["separate_rows"]["chi_neumann"]
        )) < 2e-12

    np.testing.assert_array_equal(completed[1:-1, 1:, :], before[1:-1, 1:, :])
    np.testing.assert_array_equal(completed[:, :, 3], before[:, :, 3])
    np.testing.assert_array_equal(completed[:, :, 7], before[:, :, 7])
    assert record["ownership_pass"]
    assert record["sphere_and_Phi_bitwise"]
    assert record["final"]["metric_normalized_Linf"] < 2e-12
    assert record["final"]["chi_neumann_Linf"] < 2e-12
    assert record["completion_corrections"][
        "normal_tangential_positive_zero_noop"
    ]
    assert record["completion_corrections"][
        "anisotropy_hrr_normalized_Linf"
    ] > 0.0


def test_native_completion_rejects_unqualified_sphere_or_phi_rows():
    z, r, q = _position()
    bad_sphere = q.copy()
    bad_sphere[:, :, 3] += 0.1*z[:, None]
    with pytest.raises(RuntimeError, match="sphere/Phi"):
        complete_native_parent_position(
            bad_sphere, z, r, _background(), prerequisite_tolerance=1e-12,
        )

    bad_phi = q.copy()
    bad_phi[:, :, 7] = z[:, None]
    with pytest.raises(RuntimeError, match="sphere/Phi"):
        complete_native_parent_position(
            bad_phi, z, r, _background(), prerequisite_tolerance=1e-12,
        )


def test_native_completion_requires_positive_zero_normal_tangential_input():
    z, r, q = _position()
    nonzero = q.copy()
    nonzero[3, 2, 0] = 1e-30
    with pytest.raises(RuntimeError, match="positive zero"):
        complete_native_parent_position(
            nonzero, z, r, _background(), prerequisite_tolerance=1e-12,
        )

    negative_zero = q.copy()
    negative_zero[4, 3, 1] = -0.0
    with pytest.raises(RuntimeError, match="positive zero"):
        complete_native_parent_position(
            negative_zero, z, r, _background(), prerequisite_tolerance=1e-12,
        )


def test_q4_axis_limit_comes_from_physical_numerator_derivative():
    z = np.linspace(1.0, 2.0, 9)
    r = np.linspace(0.0, 2.0, 11)
    s = (r/r[-1])**2
    q = np.zeros((len(z), len(r), 9))
    factors = 0.4 + 0.1*z
    q[:, :, 4] = factors[:, None]*(1.0+0.3*s[None, :]+0.2*s[None, :]**2)
    q[:, 0, 4] = -999.0
    found = analytic_even_q4_limit(q, r, 7)
    np.testing.assert_allclose(found, factors, rtol=0.0, atol=2e-12)


def test_normal_gauge_source_completion_closes_only_owned_trace():
    z, r, q = _position()
    source = np.arange(len(z)*len(r)*3, dtype=float).reshape(len(z), len(r), 3)
    completed, record = complete_normal_gauge_source_wall(
        q, source, z, r, _background(beta=0.2),
    )
    np.testing.assert_array_equal(completed[1:-1], source[1:-1])
    np.testing.assert_array_equal(completed[[0, -1], :, 0], source[[0, -1], :, 0])
    np.testing.assert_array_equal(completed[[0, -1], :, 2], source[[0, -1], :, 2])
    assert record["normal_gauge"]["maximum"] < 2e-12
    assert record["ownership_pass"]
