from types import SimpleNamespace

import numpy as np

from bhps.matched_staged_continuum import ContinuousReducedParent
from bhps.phase_a2_diagnostics import (
    endpoint_data_matched_q53_parent,
    endpoint_trace_comparison,
)


def _polynomial_jet():
    z = np.linspace(1.0, 2.0, 11)
    r = np.linspace(0.0, 12.0, 12)
    s = (r/12.0)**2
    zz, ss = np.meshgrid(z, s, indexing="ij")
    factors = np.arange(1.0, 10.0)[None, None, :]
    q0 = (
        0.7 + 0.2*zz**5 + 0.3*zz**2*ss + 0.11*ss**3
    )[:, :, None]*factors
    velocity = (0.1*zz**4+0.03*ss**2)[:, :, None]*factors
    acceleration = (-0.08*zz**5+0.02*ss**3)[:, :, None]*factors
    qz = (zz**4+0.6*zz*ss)[:, :, None]*factors
    qzz = (4.0*zz**3+0.6*ss)[:, :, None]*factors
    qs = (0.3*zz**2+0.33*ss**2)[:, :, None]*factors
    qss = (0.66*ss)[:, :, None]*factors
    qzs = (0.6*zz)[:, :, None]*factors
    vz = (0.4*zz**3)[:, :, None]*factors
    vs = (0.06*ss)[:, :, None]*factors
    ds_dr = 2.0*r/12.0**2
    d2s_dr2 = 2.0/12.0**2
    first = np.zeros((3, *q0.shape))
    second = np.zeros((3, 3, *q0.shape))
    first[0] = velocity
    first[1] = qz
    first[2] = qs*ds_dr[None, :, None]
    second[0, 0] = acceleration
    second[0, 1] = second[1, 0] = vz
    second[0, 2] = second[2, 0] = vs*ds_dr[None, :, None]
    second[1, 1] = qzz
    second[1, 2] = second[2, 1] = qzs*ds_dr[None, :, None]
    second[2, 2] = (
        qss*ds_dr[None, :, None]**2 + qs*d2s_dr2
    )
    return SimpleNamespace(
        z=z, r=r, reduced_fields=q0,
        reduced_first=first, reduced_second=second,
    )


def test_q53_reproduces_mixed_degree_polynomial_through_second_derivatives():
    jet = _polynomial_jet()
    parent = endpoint_data_matched_q53_parent(jet, jet.z, jet.r)
    target_z = np.linspace(1.0, 2.0, 17)
    target_r = np.linspace(0.0, 10.0, 19)
    found = parent.project(target_z, target_r)
    s = (target_r/12.0)**2
    zz, ss = np.meshgrid(target_z, s, indexing="ij")
    factors = np.arange(1.0, 10.0)[None, None, :]
    expected_q = (
        0.7 + 0.2*zz**5 + 0.3*zz**2*ss + 0.11*ss**3
    )[:, :, None]*factors
    expected_qz = (zz**4+0.6*zz*ss)[:, :, None]*factors
    expected_qzz = (4.0*zz**3+0.6*ss)[:, :, None]*factors
    expected_a = (-0.08*zz**5+0.02*ss**3)[:, :, None]*factors
    assert np.max(np.abs(found.reduced_fields-expected_q)) < 2e-11
    assert np.max(np.abs(found.reduced_first[1]-expected_qz)) < 2e-10
    assert np.max(np.abs(found.reduced_second[1, 1]-expected_qzz)) < 2e-9
    assert np.max(np.abs(found.reduced_second[0, 0]-expected_a)) < 2e-11
    assert parent.position.z_degree == 5
    assert parent.position.s_degree == 3
    assert parent.position.z_boundary == "first-z-plus-omitted-edge-knots"
    assert parent.acceleration.z_boundary == "not-a-knot"


def test_q53_and_primary_share_explicit_wall_data_but_not_free_qzz():
    jet = _polynomial_jet()
    zz, ss = np.meshgrid(jet.z, (jet.r/12.0)**2, indexing="ij")
    factors = np.arange(1.0, 10.0)[None, None, :]
    jet.reduced_fields[:] = (
        np.sin(2.3*zz)+0.17*ss+0.04*zz*ss**2
    )[:, :, None]*factors
    jet.reduced_first[0] = (
        np.cos(1.7*zz)+0.02*ss
    )[:, :, None]*factors
    jet.reduced_first[1, 0] = (0.37+0.11*ss[0])[:, None]*factors[0]
    jet.reduced_first[1, -1] = (-0.29+0.07*ss[-1])[:, None]*factors[0]
    jet.reduced_second[0, 1, 0] = (0.19-0.03*ss[0])[:, None]*factors[0]
    jet.reduced_second[0, 1, -1] = (-0.13+0.02*ss[-1])[:, None]*factors[0]
    primary = ContinuousReducedParent.from_jet_field(
        jet, jet.z, jet.r, parent_identity="synthetic-primary",
    )
    q53 = endpoint_data_matched_q53_parent(jet, jet.z, jet.r)
    record = endpoint_trace_comparison(
        primary, q53, np.linspace(1.0, 2.0, 15),
        np.linspace(0.0, 10.0, 17),
    )
    for name in (
        "position", "position_r", "position_rr", "position_z",
        "position_zr", "velocity", "velocity_z", "acceleration",
    ):
        assert record[name] < 2e-9
    assert record["position_zz_free_outcome"] > 1e-4


def test_q53_is_deterministic_and_fails_closed_on_bad_inputs():
    jet = _polynomial_jet()
    first = endpoint_data_matched_q53_parent(jet, jet.z, jet.r)
    second = endpoint_data_matched_q53_parent(jet, jet.z, jet.r)
    assert first.fingerprint() == second.fingerprint()
    assert np.array_equal(first.position.z_knots, second.position.z_knots)
    assert not first.position.coefficients.flags.writeable
    bad_z = jet.z.copy()
    bad_z[-1] += 1e-3
    with np.testing.assert_raises_regex(ValueError, "coordinates differ"):
        endpoint_data_matched_q53_parent(jet, bad_z, jet.r)
    poisoned = _polynomial_jet()
    poisoned.reduced_first[1, 0, 0, 0] = np.nan
    with np.testing.assert_raises_regex(ValueError, "nonfinite"):
        endpoint_data_matched_q53_parent(poisoned, poisoned.z, poisoned.r)
