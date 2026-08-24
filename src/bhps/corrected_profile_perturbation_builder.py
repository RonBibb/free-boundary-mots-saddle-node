"""Constraint-solved corrected slices with a varied scalar-pulse radial width."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RectBivariateSpline

from bhps.anisotropic_geometry import (
    anisotropic_hamiltonian_residual,
    anisotropic_metric_acceleration,
    anisotropic_scalar_acceleration,
    anisotropic_scalar_gradient_energy,
    anisotropic_spatial_junction_fields,
)
from bhps.anisotropic_initial_data import (
    anisotropic_initial_data_residual,
    solve_anisotropic_initial_data,
)
from bhps.axisymmetric_reduced_wave_evolution import axisymmetric_principal_coefficients
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice
from bhps.generalized_harmonic_jets import spatial_metric_acceleration_trace
from bhps.lapse_acceleration_corner import construct_localized_target_lapse_acceleration_completion
from bhps.physical_corner_corrector import (
    combine_shape_modes,
    physical_corner_state,
    tracefree_shape_basis,
)
from bhps.regular_so3_gh_reduction import RegularSO3BackgroundJetField
from bhps.scalar_pulse import scalar_pulse


AMPLITUDE = 7.90
SIGMA_Y = .2
CENTER_FRACTION = .9


def _interpolate(field, source_z, source_r, target_z, target_r):
    zz, rr = np.meshgrid(target_z, target_r, indexing="ij")
    return RectBivariateSpline(
        source_z, source_r, np.asarray(field), kx=3, ky=3, s=0,
    ).ev(zz.ravel(), rr.ravel()).reshape(len(target_z), len(target_r))


def _relative_norm(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return float(
        np.linalg.norm(left - right)
        / max(np.linalg.norm(left), np.linalg.norm(right), 1e-300)
    )


def profile_perturbation_norms(z, r, sigma_r):
    """Measure a width perturbation against the unperturbed pulse."""
    sigma_r = float(sigma_r)
    if sigma_r <= 0:
        raise ValueError("sigma_r must be positive")
    varied = scalar_pulse(
        z, r, AMPLITUDE, sigma_r, SIGMA_Y, CENTER_FRACTION,
    )
    baseline = scalar_pulse(
        z, r, AMPLITUDE, 1., SIGMA_Y, CENTER_FRACTION,
    )
    varied_gradient = np.stack(varied[1:])
    baseline_gradient = np.stack(baseline[1:])
    return {
        "sigma_r": sigma_r,
        "delta_sigma_r": sigma_r - 1.,
        "pulse_relative_L2_difference": _relative_norm(varied[0], baseline[0]),
        "gradient_relative_L2_difference": _relative_norm(
            varied_gradient, baseline_gradient,
        ),
    }


def _maximum_junction_residual(junction, radial_buffer=7):
    radial_slice = slice(None, -int(radial_buffer)) if radial_buffer else slice(None)
    return float(max(
        np.max(np.abs(wall[name][radial_slice]))
        for wall in junction["walls"] for name in ("radial", "transverse")
    ))


def build_profile_refined(
    coarse, nz, nr, name, sigma_r, selector_iterations=40,
    slice_iterations=270,
):
    """Build one fully re-solved corrected slice at fixed A and varied width."""
    sigma_r = float(sigma_r)
    if sigma_r <= 0:
        raise ValueError("sigma_r must be positive")
    nz = int(nz)
    nr = int(nr)
    reference = solve_finite_wall_high_order_slice(
        AMPLITUDE, nz=nz, nr=nr, r_max=8., wall_stiffness=20.,
        sigma_r=sigma_r, sigma_y=SIGMA_Y,
        center_fraction=CENTER_FRACTION, epsilon=.1, backreaction=.01,
        tolerance=1e-10, iterations=int(slice_iterations),
    )
    z = reference["z"]
    r = reference["r"]
    chi, chi_r, chi_z = scalar_pulse(
        z, r, AMPLITUDE, sigma_r, SIGMA_Y, CENTER_FRACTION,
    )
    archive = np.load("results/corrected_family_knot_A8_state.npz")
    coefficients = archive["coefficients"]
    modes = tracefree_shape_basis(
        z, r, 6, (.5, 1.), 8., ((7.5, 1.5), (7.5, 3.)),
    )["modes"]
    a, b, c = combine_shape_modes(coefficients, modes)
    coarse_q = 1. / np.asarray(coarse["psi"]) - np.asarray(coarse["z"])[:, None]
    initial_q = _interpolate(coarse_q, coarse["z"], coarse["r"], z, r)
    initial_phi = _interpolate(coarse["phi"], coarse["z"], coarse["r"], z, r)
    selected = solve_anisotropic_initial_data(
        z, r, reference["q"], reference["phi"], a, b, c,
        reference["background"], chi_r, chi_z,
        initial_q=initial_q, initial_phi=initial_phi, stencil_width=7,
        tolerance=1e-9, iterations=int(selector_iterations),
    )
    q = selected["q"]
    phi = selected["phi"]
    psi = 1. / (z[:, None] + q)
    mass = float(reference["background"]["mass_squared"])
    acceleration = anisotropic_metric_acceleration(
        z, r, psi, a, b, c, phi, chi_r, chi_z, mass, chi=chi,
        stencil_width=7, lapse=psi,
    )
    phi_tt = anisotropic_scalar_acceleration(
        z, r, psi, a, b, c, phi, mass, lapse=psi, stencil_width=7,
    )
    chi_tt = anisotropic_scalar_acceleration(
        z, r, psi, a, b, c, chi, 0., lapse=psi, stencil_width=7,
    )
    trace = spatial_metric_acceleration_trace(acceleration, psi, a, b, c)
    completion = construct_localized_target_lapse_acceleration_completion(
        z, acceleration, psi, psi, a, phi, reference["background"],
        phi_tt, .5 * trace, .15,
    )

    selector_residual = anisotropic_initial_data_residual(
        q, phi, z, r, a, b, c, reference["background"], chi_r, chi_z,
        reference["q"], reference["phi"], 7,
    )
    hamiltonian_residual = selector_residual[:q.size].reshape(q.shape)
    scalar_residual = selector_residual[q.size:].reshape(q.shape)
    junction = anisotropic_spatial_junction_fields(
        z, r, psi, a, b, c, phi, reference["background"], 7,
    )
    corner = physical_corner_state(
        z, r, q, phi, a, b, c, reference["background"], chi_r, chi_z,
        chi, None, 7, 7, True,
    )
    gradient_scale = max(
        float(np.max(np.abs(chi_r))), float(np.max(np.abs(chi_z))), 1e-300,
    )
    neumann_defect = float(max(
        np.max(np.abs(chi_z[0])), np.max(np.abs(chi_z[-1])),
    ) / gradient_scale)
    raw_hamiltonian = anisotropic_hamiltonian_residual(
        z, r, psi, a, b, c, phi, chi_r, chi_z, mass, chi=chi,
        stencil_width=7,
    )
    compatibility = {
        "finite_wall_converged": bool(reference["converged"]),
        "finite_wall_maximum_residual": float(reference["max_abs_residual"]),
        "anisotropic_selector_converged": bool(selected["converged"]),
        "anisotropic_selector_maximum_residual": float(selected["maximum_residual"]),
        "balanced_hamiltonian_maximum_residual": float(
            np.max(np.abs(hamiltonian_residual))
        ),
        "balanced_scalar_maximum_residual": float(np.max(np.abs(scalar_residual))),
        "raw_hamiltonian_retained_interior_maximum": float(
            np.max(np.abs(raw_hamiltonian[1:-1, :-7]))
        ),
        "zeroth_order_spatial_junction_maximum_residual": _maximum_junction_residual(
            junction, 7,
        ),
        "second_corner_maximum_intrinsic_residual": float(
            corner["maximum_intrinsic_residual"]
        ),
        "second_corner_maximum_fixed_scaled_residual": float(
            corner["maximum_fixed_scaled_residual"]
        ),
        "scalar_wall_Neumann_relative_defect": neumann_defect,
        "extrinsic_curvature_and_scalar_momenta_zero": True,
        "momentum_constraint_by_time_symmetry": 0.,
        "gradient_energy_dimensionless": anisotropic_scalar_gradient_energy(
            z, r, psi, a, b, c, chi_r, chi_z,
        ),
    }
    return {
        "name": str(name),
        "source_grid": [nz, nr],
        "fold_amplitude": AMPLITUDE,
        "sigma_r": sigma_r,
        "sigma_y": SIGMA_Y,
        "center_fraction": CENTER_FRACTION,
        "selector_maximum": float(selected["maximum_residual"]),
        "z": z,
        "r": r,
        "psi": psi,
        "a": a,
        "b": b,
        "c": c,
        "phi": phi,
        "chi": chi,
        "chi_r": chi_r,
        "chi_z": chi_z,
        "background": reference["background"],
        "mass_squared": mass,
        "principal": axisymmetric_principal_coefficients(psi, a, b, c),
        "compatibility": compatibility,
        "profile_perturbation": profile_perturbation_norms(z, r, sigma_r),
        "jet_field": RegularSO3BackgroundJetField(
            z, r, psi, psi, a, b, c, phi, chi, acceleration,
            completion["lapse_acceleration"], phi_tt, chi_tt, 7,
        ),
    }
