from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import LinAlgError
from scipy.sparse import csr_matrix, diags

import bhps.protocol131_postmortem as protocol131
from bhps.protocol131_postmortem import (
    Protocol131AuditError,
    _parent_ill_conditioned,
    _interpolated_left_null_subspace,
    _stats,
    classify_protocol131,
    linear_range_analysis,
    power_two_ruiz,
    residual_localization,
    right_power_two_scaling,
)


def _parent(nz=17, nr=19):
    z = np.linspace(1.0, np.e, nz)
    r = np.linspace(0.0, 12.0, nr)
    return {
        "label": "N0",
        "z": z,
        "r": r,
        "q": np.full((nz, nr), 0.03),
        "phi": np.full((nz, nr), 0.1),
    }


def test_stats_top_entries_use_absolute_value_then_c_order():
    z = np.linspace(1.0, 2.0, 3)
    r = np.linspace(0.0, 1.0, 4)
    values = np.zeros((3, 4))
    values[2, 3] = 4.0
    values[0, 2] = -4.0
    values[1, 1] = 3.0
    mask = np.ones_like(values, dtype=bool)
    result = _stats(values, mask, float(np.sum(values**2)), z, r)
    assert result["count"] == 12
    assert result["Linf"] == 4.0
    assert result["argmax"]["flat_index"] == 2
    assert result["top"][1]["flat_index"] == 11
    assert result["L2_energy_fraction"] == 1.0


def test_localization_atoms_are_nonoverlapping_and_recombine_each_block():
    parent = _parent()
    nz, nr = parent["q"].shape
    residual = np.arange(2*nz*nr, dtype=float) - nz*nr
    result = residual_localization(parent, residual)
    for block in ("metric", "Phi"):
        atoms = [
            value for name, value in result["atoms"].items()
            if name.startswith(block + "/")
        ]
        assert sum(item["count"] for item in atoms) == nz*nr
        assert np.isclose(
            sum(item["L2_energy_fraction"] for item in atoms),
            result["blocks"][block]["L2_energy_fraction"],
        )
        assert max(item["Linf"] for item in atoms) == result["blocks"][block]["Linf"]
    assert np.isclose(
        sum(item["L2_energy_fraction"] for item in result["blocks"].values()),
        1.0,
    )
    assert np.isclose(np.sum(result["binned_energy_16x16"]), 1.0)
    assert result["wall_even"].shape == (2, nr)
    assert result["wall_odd"].shape == (2, nr)


def test_right_power_two_scaling_is_exact_and_column_only():
    matrix = csr_matrix(np.array([
        [1.0, 0.0, 8.0],
        [0.0, 2.0, 0.0],
        [1.0, 0.0, 8.0],
    ]))
    scaled, factors, exponents = right_power_two_scaling(matrix)
    np.testing.assert_array_equal(factors, np.ldexp(np.ones(3), exponents))
    np.testing.assert_array_equal(scaled.toarray(), matrix.toarray()*factors[None, :])
    norms = np.sqrt(np.asarray(scaled.power(2).sum(axis=0)).ravel())
    assert np.all(norms >= 2.0**-0.5)
    assert np.all(norms <= 2.0**0.5)


def test_power_two_ruiz_uses_only_exact_binary_scalings():
    rng = np.random.default_rng(131)
    matrix = csr_matrix(rng.normal(size=(24, 24)) + 3.0*np.eye(24))
    transformed, rows, columns, records = power_two_ruiz(matrix)
    assert len(records) == 4
    np.testing.assert_array_equal(
        transformed.toarray(), rows[:, None]*matrix.toarray()*columns[None, :]
    )
    for values in (rows, columns):
        mantissa, _ = np.frexp(values)
        np.testing.assert_array_equal(mantissa, np.full_like(mantissa, 0.5))


def test_linear_range_lane_resolves_well_conditioned_diagonal_system():
    parent = _parent(nz=15, nr=1)
    parent["z"] = np.asarray([1.0])
    parent["r"] = np.linspace(0.0, 12.0, 15)
    parent["q"] = np.full((1, 15), 0.03)
    parent["phi"] = np.full((1, 15), 0.1)
    diagonal = np.linspace(0.7, 1.3, 30)
    matrix = diags(diagonal, format="csr")
    solution = np.linspace(-0.2, 0.2, 30)
    residual = np.asarray(matrix @ solution)
    result = linear_range_analysis(parent, residual, matrix)
    assert result["lu"]["succeeded"] is True
    assert result["lu"]["direct_projected_Linf"] < 1e-13
    assert result["lsmr"]["projected_Linf"] < 1e-10
    assert result["lsqr"]["projected_Linf"] < 1e-10
    assert result["projection_numerically_zero_floor"] is True
    assert result["projection_accepted"] is True
    assert result["spectrum_certified"] is True
    assert all(mode["definitely_nonnull"] for mode in result["modes"])
    assert not any(mode["definitely_null"] for mode in result["modes"])
    assert result["high_nullity_unresolved"] is False


def test_propack_fallback_can_still_certify_spectrum(monkeypatch):
    parent = _parent(nz=1, nr=15)
    diagonal = np.linspace(0.7, 1.3, 30)
    matrix = diags(diagonal, format="csr")
    residual = np.asarray(matrix @ np.linspace(-0.2, 0.2, 30))

    def fail_inverse(*args, **kwargs):
        raise RuntimeError("manufactured inverse failure")

    monkeypatch.setattr(protocol131, "_inverse_singular_modes", fail_inverse)
    result = linear_range_analysis(parent, residual, matrix)
    assert result["mode_method"] == "propack"
    assert result["spectrum_attempts"]
    assert result["spectrum_errors"] == []
    assert result["spectrum_certified"] is True


def test_propack_linalg_failure_becomes_uncertified_spectrum(monkeypatch):
    parent = _parent(nz=1, nr=15)
    matrix = diags(np.linspace(0.7, 1.3, 30), format="csr")
    residual = np.asarray(matrix @ np.linspace(-0.2, 0.2, 30))

    monkeypatch.setattr(
        protocol131,
        "_inverse_singular_modes",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("manufactured inverse failure")
        ),
    )
    monkeypatch.setattr(
        protocol131,
        "_propack_singular_modes",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            LinAlgError("manufactured PROPACK nonconvergence")
        ),
    )
    result = linear_range_analysis(parent, residual, matrix)
    assert result["mode_method"] == "propack"
    assert result["spectrum_certified"] is False
    assert result["modes"] == []
    assert "LinAlgError" in result["spectrum_errors"][0]


def test_linalg_failure_in_k16_extension_becomes_uncertified(monkeypatch):
    parent = _parent(nz=1, nr=15)
    matrix = diags(np.linspace(0.7, 1.3, 30), format="csr")
    residual = np.asarray(matrix @ np.linspace(-0.2, 0.2, 30))
    calls = []

    def manufactured_inverse(matrix, lu, sigma_max, seed, k):
        calls.append(k)
        if k == protocol131.SINGULAR_K_EXTENSION:
            raise LinAlgError("manufactured extension nonconvergence")
        records = []
        vectors = []
        for index in range(k):
            vector = np.zeros(matrix.shape[0])
            vector[index] = 1.0
            records.append({
                "sigma": 1.0e-30,
                "relative_sigma": 1.0e-30,
                "left_triplet_defect": 0.0,
                "right_triplet_defect": 0.0,
                "relative_interval_lower": 0.0,
                "relative_interval_upper": 1.0e-30,
            })
            vectors.append((vector.copy(), vector.copy()))
        return records, vectors

    monkeypatch.setattr(
        protocol131, "_inverse_singular_modes", manufactured_inverse,
    )
    result = linear_range_analysis(parent, residual, matrix)
    assert calls == [protocol131.SINGULAR_K, protocol131.SINGULAR_K_EXTENSION]
    assert result["mode_extension_used"] is True
    assert result["spectrum_certified"] is False
    assert result["modes"] == []
    assert "LinAlgError" in result["spectrum_errors"][0]


def test_sigma_max_failure_becomes_uncertified_spectrum_record(monkeypatch):
    parent = _parent(nz=1, nr=15)
    matrix = diags(np.linspace(0.7, 1.3, 30), format="csr")
    residual = np.linspace(-0.2, 0.2, 30)

    def fail_sigma(*args, **kwargs):
        raise RuntimeError("manufactured sigma-max failure")

    monkeypatch.setattr(protocol131, "_largest_singular_value", fail_sigma)
    result = linear_range_analysis(parent, residual, matrix)
    assert result["analysis_complete"] is False
    assert result["spectrum_certified"] is False
    assert result["failure_stage"] == "largest-singular-value"
    assert "sigma-max" in result["spectrum_errors"][0]


@pytest.mark.parametrize("manufactured", [0.0, np.nan, np.inf])
def test_sigma_max_rejects_invalid_values(monkeypatch, manufactured):
    monkeypatch.setattr(
        protocol131, "svds",
        lambda *args, **kwargs: np.asarray([manufactured]),
    )
    with pytest.raises(Protocol131AuditError, match="nonpositive or nonfinite"):
        protocol131._largest_singular_value(csr_matrix(np.eye(2)), 131)


def test_jacobian_gate_rejects_any_invalid_frozen_step(monkeypatch):
    parent = {
        "label": "N0",
        "z": np.asarray([1.0]),
        "r": np.asarray([0.0]),
        "q": np.asarray([[-0.999]]),
        "phi": np.asarray([[0.0]]),
    }
    monkeypatch.setattr(
        protocol131, "_direction_masks",
        lambda value: {"manufactured": np.ones(2)},
    )
    monkeypatch.setattr(
        protocol131, "_fixed_direction",
        lambda value, name, mask: np.asarray([-1.0, 0.0]),
    )
    monkeypatch.setattr(
        protocol131, "_residual_arguments",
        lambda value, q, phi: (q, phi),
    )
    monkeypatch.setattr(
        protocol131, "joint_parent_residual",
        lambda q, phi: np.concatenate((q.ravel(), phi.ravel())),
    )
    result = protocol131.audit_analytic_jacobian(
        parent, np.zeros(2), csr_matrix(np.eye(2)),
    )
    assert result["passed"] is False
    samples = result["directions"]["manufactured"]["samples"]
    assert samples[0]["valid"] is False
    assert any(
        sample.get("valid") and sample["relative_Linf"] <= 1.0e-5
        for sample in samples[1:]
    )


def test_rank_collapsed_interpolated_null_basis_fails_closed():
    parent = _parent(nz=17, nr=19)
    size = 2 * parent["q"].size
    vector = np.linspace(1.0, 2.0, size)
    arrays = {
        "z": parent["z"],
        "r": parent["r"],
        "mode_left_vectors": np.column_stack((vector, vector)),
    }
    summary = {
        "linear": {
            "modes": [
                {"definitely_null": True},
                {"definitely_null": True},
            ],
        },
    }
    with pytest.raises(Protocol131AuditError, match="lost rank"):
        _interpolated_left_null_subspace(summary, arrays)


def _classification_fixture(*, precision_complete=True, spectrum_certified=True):
    z = np.asarray([1.0, np.e])
    r = np.asarray([0.0, 12.0])
    left = np.arange(1.0, 9.0)[:, None]
    linear = {
        "spectrum_certified": spectrum_certified,
        "modes": [{
            "definitely_null": True,
            "definitely_nonnull": False,
            "G_at_rho_linear": 2.0e-10,
            "sigma": 0.0,
        }],
        "projection_accepted": True,
        "lsmr": {"projected_Linf": 2.0e-10},
        "lsqr": {"projected_Linf": 2.0e-10},
        "ruiz": {
            "lsmr_physical_Linf": 2.0e-10,
            "lsqr_physical_Linf": 2.0e-10,
            "obstruction_certificate_complete": False,
        },
        "high_nullity_unresolved": False,
        "sigma_max": 1.0,
        "lu": {"succeeded": False},
    }
    summary = {
        "jacobian_audit": {"passed": True},
        "precision": {
            "complete": precision_complete,
            "mp_certified": precision_complete,
            "dual_certified": precision_complete,
            "eta_F": 1.0e-14,
            "arithmetic_max_below_target": False,
            "longdouble_maximum": 5.0e-10,
        },
        "replay": {"maximum": 5.0e-10},
        "linear": linear,
        "trust_radius": {"rho_linear": 2.0**-10},
        "merit_curve": {"samples": []},
        "localization": {
            "dominant_atom_by_Linf": "metric/interior/axis",
            "blocks": {
                "metric": {"L2_energy_fraction": 0.75},
                "Phi": {"L2_energy_fraction": 0.25},
            },
        },
    }
    arrays = {
        "z": z,
        "r": r,
        "mode_left_vectors": left,
        "binned_energy_16x16": np.ones((2, 16, 16)),
    }
    return summary, arrays


def test_classifier_orders_incomplete_precision_before_mechanism_labels():
    summary, arrays = _classification_fixture(precision_complete=False)
    record = classify_protocol131(
        {"N0": summary, "N1": summary}, {"N0": arrays, "N1": arrays},
    )
    assert record["classification"] == "INCONCLUSIVE-MIXED"
    assert "precision" in record["reason"]
    assert record["complete"] is True
    assert record["provenance_valid"] is True


def test_classifier_orders_invalid_jacobian_before_incomplete_precision():
    summary, arrays = _classification_fixture(precision_complete=False)
    summary["jacobian_audit"] = {"passed": False}
    record = classify_protocol131(
        {"N0": summary, "N1": summary}, {"N0": arrays, "N1": arrays},
    )
    assert record["classification"] == "INVALID-JACOBIAN"
    assert "Jacobian" in record["reason"]


def test_classifier_orders_unresolved_spectrum_before_mechanism_labels():
    summary, arrays = _classification_fixture(spectrum_certified=False)
    record = classify_protocol131(
        {"N0": summary, "N1": summary}, {"N0": arrays, "N1": arrays},
    )
    assert record["classification"] == "INCONCLUSIVE-MIXED"
    assert "spectrum" in record["reason"]


def test_incomplete_ruiz_certificate_cannot_support_obstruction():
    summary, arrays = _classification_fixture()
    record = classify_protocol131(
        {"N0": summary, "N1": summary}, {"N0": arrays, "N1": arrays},
    )
    assert record["classification"] == "INCONCLUSIVE-MIXED"
    assert record["classification"] != "DISCRETE-COMPATIBILITY-OBSTRUCTION"


def test_unavailable_merit_curve_cannot_select_nonlinear_class():
    summary, arrays = _classification_fixture()
    summary["linear"]["modes"] = []
    summary["linear"]["lsmr"]["projected_Linf"] = 5.0e-11
    summary["linear"]["lsqr"]["projected_Linf"] = 5.0e-11
    summary["merit_curve"] = {"available": False, "samples": []}
    record = classify_protocol131(
        {"N0": summary, "N1": summary}, {"N0": arrays, "N1": arrays},
    )
    assert record["classification"] == "INCONCLUSIVE-MIXED"
    assert record["classification"] != "NONLINEAR/GLOBALIZATION-UNRESOLVED"


def _ill_conditioning_fixture(*, eta=0.0, orthogonal=True):
    first = np.asarray([1.0, 0.0])
    second = np.asarray([0.0, 1.0]) if orthogonal else first.copy()
    summary = {
        "replay": {"maximum": 5.0e-10},
        "precision": {"eta_F": eta},
        "trust_radius": {"rho_linear": 1.0},
        "linear": {
            "sigma_max": 1.0,
            "modes": [{
                "sigma": 1.0e-16,
                "definitely_nonnull": True,
                "required_physical_dimensionless_correction": 0.0,
            }],
            "lu": {
                "succeeded": True,
                "direct_relative_backward_L2": 0.0,
            },
        },
    }
    arrays = {
        "column_scale": np.ones(2),
        "physical_variable_scale": np.ones(2),
        "lsmr_solution_scaled": first,
        "lsqr_solution_scaled": second,
    }
    return summary, arrays


def test_ill_conditioning_compares_correction_vectors_not_only_norms():
    summary, arrays = _ill_conditioning_fixture(orthogonal=True)
    assert _parent_ill_conditioned(summary, arrays) is True
    summary, arrays = _ill_conditioning_fixture(orthogonal=False)
    assert _parent_ill_conditioned(summary, arrays) is False


def test_ill_conditioning_includes_precision_sensitivity():
    summary, arrays = _ill_conditioning_fixture(
        eta=1.1e-11, orthogonal=False,
    )
    assert _parent_ill_conditioned(summary, arrays) is True
