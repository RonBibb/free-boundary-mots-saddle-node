from __future__ import annotations

import numpy as np
import pytest

from bhps.anisotropic_initial_data import _raw_residual_and_jacobian
from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.joint_parent_bulk_audit import (
    SOURCE_STENCIL_WIDTH,
    open_anisotropic_bulk_terms_fd,
    open_anisotropic_bulk_terms_from_jets,
)
from bhps.joint_parent_bulk_reference import FiniteWallReferenceHermitePair
from bhps.joint_parent_fields import (
    native_position_from_primitives,
    reconstruct_native_spatial_ansatz,
)


def _jet(value, z, r, *, z_first=0.0, r_first=0.0, zz=0.0, rr=0.0):
    shape = (len(z), len(r))

    def field(item):
        return np.broadcast_to(np.asarray(item, dtype=float), shape).copy()

    return {
        "value": field(value),
        "z": field(z_first),
        "r": field(r_first),
        "zz": field(zz),
        "rr": field(rr),
    }


def _flat_manufactured_jets(z, r):
    zz, rr = np.meshgrid(z, r, indexing="ij")
    phi = zz**2+rr**2
    candidate = {
        "h_zz": _jet(1.0, z, r),
        "h_rr": _jet(1.0, z, r),
        "h_perp": _jet(1.0, z, r),
        "Phi": _jet(phi, z, r, z_first=2.0*zz, r_first=2.0*rr, zz=2.0, rr=2.0),
        "chi": _jet(zz*rr, z, r, z_first=rr, r_first=zz),
    }
    reference = {
        "q": 1.0-zz,
        "q_z": -np.ones_like(zz),
        "q_r": np.zeros_like(zz),
        "q_zz": np.zeros_like(zz),
        "q_rr": np.zeros_like(zz),
        "phi": phi,
        "phi_z": 2.0*zz,
        "phi_r": 2.0*rr,
        "phi_zz": 2.0*np.ones_like(zz),
        "phi_rr": 2.0*np.ones_like(zz),
    }
    return candidate, reference


def test_analytic_manufactured_flat_equations_and_common_normalization():
    z = np.linspace(1.0, 1.8, 7)
    r = np.linspace(0.2, 1.6, 8)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    mass = 0.4
    candidate, reference = _flat_manufactured_jets(z, r)
    record = open_anisotropic_bulk_terms_from_jets(
        candidate, reference, z, r, {"mass_squared": mass},
    )
    phi = zz**2+rr**2
    gradient = (2.0*zz)**2+(2.0*rr)**2+rr**2+zz**2
    expected_hamiltonian = 12.0-gradient-mass*phi**2
    expected_phi = 8.0-mass*phi
    np.testing.assert_allclose(
        record["raw"]["hamiltonian"], expected_hamiltonian,
        rtol=2e-14, atol=2e-14,
    )
    np.testing.assert_allclose(
        record["raw"]["Phi"], expected_phi,
        rtol=2e-14, atol=2e-14,
    )
    for equation in ("hamiltonian", "Phi"):
        np.testing.assert_allclose(
            record["defect"][equation], record["raw"][equation],
            rtol=2e-14, atol=2e-14,
        )
        np.testing.assert_allclose(
            record["balanced"][equation], 0.0,
            rtol=0.0, atol=4e-14,
        )
        expected_denominator = np.maximum(
            1.0,
            np.abs(record["raw"][equation])
            +np.abs(record["defect"][equation]),
        )
        np.testing.assert_array_equal(
            record["common_denominator"][equation], expected_denominator,
        )
        np.testing.assert_allclose(
            record["raw_normalized"][equation],
            np.abs(record["raw"][equation])/expected_denominator,
            rtol=0.0,
            atol=0.0,
        )
    assert record["reassembly_Linf"] == 0.0
    assert record["candidate_chi_jets_reused_in_defect"]
    assert not record["lapse_used"]


def test_analytic_constant_tracefree_shape_has_expected_curvature():
    z = np.linspace(1.0, 1.5, 4)
    r = np.linspace(0.3, 1.7, 6)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    a = 0.10
    b = -0.05
    c = -0.025
    candidate = {
        "h_zz": _jet(np.exp(2.0*a), z, r),
        "h_rr": _jet(np.exp(2.0*b), z, r),
        "h_perp": _jet(np.exp(2.0*c), z, r),
        "Phi": _jet(0.0, z, r),
        "chi": _jet(0.0, z, r),
    }
    reference = {
        "q": 1.0-zz,
        "q_z": -np.ones_like(zz),
        "q_r": np.zeros_like(zz),
        "q_zz": np.zeros_like(zz),
        "q_rr": np.zeros_like(zz),
        "phi": np.zeros_like(zz),
        "phi_z": np.zeros_like(zz),
        "phi_r": np.zeros_like(zz),
        "phi_zz": np.zeros_like(zz),
        "phi_rr": np.zeros_like(zz),
    }
    record = open_anisotropic_bulk_terms_from_jets(
        candidate, reference, z, r, {"mass_squared": 0.5},
    )
    scalar_bar = 2.0*(np.exp(-2.0*c)-np.exp(-2.0*b))/rr**2
    np.testing.assert_allclose(
        record["balanced"]["hamiltonian"], scalar_bar,
        rtol=2e-14,
        atol=2e-14,
    )
    np.testing.assert_allclose(record["balanced"]["Phi"], 0.0, atol=0.0)


def _fd_case():
    z = np.linspace(1.0, np.e, 9)
    r = np.linspace(0.0, 2.0, 11)
    zz, rr = np.meshgrid(z, r, indexing="ij")
    psi = 0.85+0.015*zz+0.004*rr**2
    a = 0.012*np.sin(np.pi*(zz-1.0)/(np.e-1.0))*np.exp(-0.3*rr**2)
    radial = 0.004*rr**2*np.cos(0.7*zz)*np.exp(-0.2*rr**2)
    b = (-a+2.0*radial)/3.0
    c = (-a-radial)/3.0
    alpha = psi*(1.0+0.03*(zz-1.0)**2)
    phi = 0.02*np.cos(0.9*zz)*np.exp(-0.15*rr**2)
    chi = 0.03*np.sin(0.8*zz)*np.exp(-0.25*rr**2)
    position = native_position_from_primitives(
        z, r, alpha, psi, a, b, c, phi, chi,
    )
    reference_q = 1.0/(0.9+0.01*zz)-zz
    reference_phi = 0.01*np.cos(0.4*zz)*np.ones_like(rr)
    background = {
        "mass_squared": 0.5,
        "wall_stiffness": 20.0,
        "v0": 0.0,
        "v1": 0.0,
        "beta_a": 0.1,
        "beta_b": -0.1,
        "wall_potential_a": 0.0,
        "wall_potential_b": 0.0,
    }
    return z, r, position, reference_q, reference_phi, background


def test_fd_open_rows_reassemble_and_match_established_legacy_functional():
    z, r, position, reference_q, reference_phi, background = _fd_case()
    record = open_anisotropic_bulk_terms_fd(
        position, z, r, reference_q, reference_phi, background,
    )
    fields = reconstruct_native_spatial_ansatz(position, r)
    candidate_q = 1.0/fields["psi"]-z[:, None]
    dz = derivative_matrix(z, 1, SOURCE_STENCIL_WIDTH)
    dr = derivative_matrix(r, 1, SOURCE_STENCIL_WIDTH)
    chi_z = dz @ fields["chi"]
    chi_r = (dr @ fields["chi"].T).T
    chi_r[:, 0] = 0.0
    arguments = (
        z, r, fields["a"], fields["b"], fields["c"], background,
        chi_r, chi_z, reference_q, reference_phi,
    )
    legacy_raw = _raw_residual_and_jacobian(
        candidate_q, fields["phi"], *arguments, SOURCE_STENCIL_WIDTH, False,
    )
    zeros = np.zeros_like(reference_q)
    legacy_defect = _raw_residual_and_jacobian(
        reference_q,
        reference_phi,
        z,
        r,
        zeros,
        zeros,
        zeros,
        background,
        chi_r,
        chi_z,
        reference_q,
        reference_phi,
        SOURCE_STENCIL_WIDTH,
        False,
    )
    shape = reference_q.shape
    count = reference_q.size
    legacy_raw = {
        "hamiltonian": legacy_raw[:count].reshape(shape),
        "Phi": legacy_raw[count:].reshape(shape),
    }
    legacy_defect = {
        "hamiltonian": legacy_defect[:count].reshape(shape),
        "Phi": legacy_defect[count:].reshape(shape),
    }
    open_rows = np.zeros(shape, dtype=bool)
    open_rows[1:-1, :-1] = True
    for equation in ("hamiltonian", "Phi"):
        np.testing.assert_array_equal(
            record["raw"][equation][open_rows], legacy_raw[equation][open_rows],
        )
        np.testing.assert_array_equal(
            record["defect"][equation][open_rows],
            legacy_defect[equation][open_rows],
        )
        np.testing.assert_array_equal(
            record["balanced"][equation],
            record["raw"][equation]-record["defect"][equation],
        )
        np.testing.assert_array_equal(record["reassembly_defect"][equation], 0.0)
    owned_rows = ~open_rows
    assert max(
        np.max(np.abs(record["raw"][equation][owned_rows]-legacy_raw[equation][owned_rows]))
        for equation in ("hamiltonian", "Phi")
    ) > 1e-6
    assert record["source_stencil_width"] == 7


def test_fd_bulk_audit_is_bitwise_lapse_invariant():
    z, r, position, reference_q, reference_phi, background = _fd_case()
    altered = position.copy()
    altered[:, :, 2] = -3.7*np.abs(position[:, :, 2])-0.2
    baseline = open_anisotropic_bulk_terms_fd(
        position, z, r, reference_q, reference_phi, background,
    )
    changed = open_anisotropic_bulk_terms_fd(
        altered, z, r, reference_q, reference_phi, background,
    )
    for lane in (
        "raw", "defect", "balanced", "common_denominator",
        "raw_normalized", "balanced_normalized", "reassembly_defect",
    ):
        for equation in ("hamiltonian", "Phi"):
            np.testing.assert_array_equal(
                baseline[lane][equation], changed[lane][equation],
            )


def test_analytic_axis_requires_explicit_regular_limit():
    z = np.linspace(1.0, 1.4, 4)
    r = np.linspace(0.0, 1.4, 8)
    candidate, reference = _flat_manufactured_jets(z, r)
    with pytest.raises(ValueError, match="regular_axis=True"):
        open_anisotropic_bulk_terms_from_jets(
            candidate, reference, z, r, {"mass_squared": 0.4},
        )
    record = open_anisotropic_bulk_terms_from_jets(
        candidate,
        reference,
        z,
        r,
        {"mass_squared": 0.4},
        regular_axis=True,
    )
    zz = z[:, None]
    expected_phi_axis = 8.0-0.4*zz**2
    np.testing.assert_allclose(
        record["raw"]["Phi"][:, 0], expected_phi_axis[:, 0],
        rtol=2e-14,
        atol=2e-14,
    )
    assert record["axis_treatment"] == "explicit_regular_even_limit"


def test_analytic_backend_accepts_immutable_reference_contract_mapping():
    source_z = np.linspace(1.0, np.e, 11)
    source_r = np.linspace(0.0, 12.0, 15)
    source_zz, source_rr = np.meshgrid(source_z, source_r, indexing="ij")
    source_s = (source_rr/12.0)**2
    source_q = 0.08+0.01*source_zz**4+0.02*source_s**2
    source_phi = 0.03*source_zz**3-0.01*source_s
    reference = FiniteWallReferenceHermitePair.build(
        source_z, source_r, source_q, source_phi,
    ).primary
    z = np.linspace(1.05, 2.65, 7)
    r = np.linspace(0.2, 11.8, 9)
    mapping = reference.evaluate(z, r).as_derivative_mapping()
    s = z[:, None]+mapping["q"]
    psi = 1.0/s
    s_z = 1.0+mapping["q_z"]
    s_r = mapping["q_r"]
    psi_z = -psi**2*s_z
    psi_r = -psi**2*s_r
    psi_zz = 2.0*psi**3*s_z**2-psi**2*mapping["q_zz"]
    psi_rr = 2.0*psi**3*s_r**2-psi**2*mapping["q_rr"]
    metric = {
        "value": psi**2,
        "z": 2.0*psi*psi_z,
        "r": 2.0*psi*psi_r,
        "zz": 2.0*(psi_z**2+psi*psi_zz),
        "rr": 2.0*(psi_r**2+psi*psi_rr),
    }
    candidate = {
        "h_zz": metric,
        "h_rr": metric,
        "h_perp": metric,
        "Phi": {
            "value": mapping["phi"],
            "z": mapping["phi_z"],
            "r": mapping["phi_r"],
            "zz": mapping["phi_zz"],
            "rr": mapping["phi_rr"],
        },
        "chi": _jet(0.0, z, r),
    }
    record = open_anisotropic_bulk_terms_from_jets(
        candidate, mapping, z, r, {"mass_squared": 0.5},
    )
    for equation in ("hamiltonian", "Phi"):
        np.testing.assert_allclose(
            record["balanced"][equation], 0.0, rtol=0.0, atol=3e-11,
        )
