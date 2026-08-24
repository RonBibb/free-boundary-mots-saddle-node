"""Task-specific matched-spacing Rmax=10 builder for corrected A=7.90 data.

The physical shape coefficients were solved jointly on Rmax=8, 10, and 12
at the A=8 family knot.  This builder re-evaluates that domain-independent
basis on Rmax=10 and solves fresh A=7.90 constraint data.  It never stretches
or extrapolates an Rmax=8 solution into the added exterior.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

from bhps.anisotropic_geometry import (
    anisotropic_metric_acceleration,
    anisotropic_scalar_acceleration,
)
from bhps.anisotropic_initial_data import solve_anisotropic_initial_data
from bhps.axisymmetric_reduced_wave_evolution import (
    axisymmetric_principal_coefficients,
)
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice
from bhps.generalized_harmonic_jets import spatial_metric_acceleration_trace
from bhps.lapse_acceleration_corner import (
    construct_localized_target_lapse_acceleration_completion,
)
from bhps.physical_corner_corrector import (
    combine_shape_modes,
    tracefree_shape_basis,
)
from bhps.regular_so3_gh_reduction import RegularSO3BackgroundJetField
from bhps.scalar_pulse import scalar_pulse


AMPLITUDE = 7.90
R_MAX = 10.0
BASIS_RADIUS = 8.0
AXIS_WIDTHS = (0.5, 1.0)
ANNULAR_PROFILES = ((7.5, 1.5), (7.5, 3.0))
KNOT_STATE = Path("results/corrected_family_knot_A8_state.npz")


def matched_radial_count(reference_count, reference_rmax=8.0, target_rmax=R_MAX):
    """Return a target count preserving the reference interval spacing."""
    intervals = int(reference_count) - 1
    target_intervals = intervals * float(target_rmax) / float(reference_rmax)
    rounded = int(round(target_intervals))
    if not np.isclose(target_intervals, rounded, rtol=0.0, atol=1e-12):
        raise ValueError("target domain is not commensurate with reference spacing")
    return rounded + 1


def interpolate(field, source_z, source_r, target_z, target_r):
    zz, rr = np.meshgrid(target_z, target_r, indexing="ij")
    return RectBivariateSpline(
        source_z, source_r, np.asarray(field), kx=3, ky=3, s=0,
    ).ev(zz.ravel(), rr.ravel()).reshape(len(target_z), len(target_r))


def shape_fields(z, r, coefficients):
    modes = tracefree_shape_basis(
        z, r, 6, AXIS_WIDTHS, BASIS_RADIUS, ANNULAR_PROFILES,
    )["modes"]
    return combine_shape_modes(coefficients, modes)


def _assemble(reference, selected, shape, amplitude, name):
    z = np.asarray(reference["z"])
    r = np.asarray(reference["r"])
    a, b, c = shape
    chi, chi_r, chi_z = scalar_pulse(z, r, amplitude)
    psi = 1.0 / (z[:, None] + selected["q"])
    phi = selected["phi"]
    mass = float(reference["background"]["mass_squared"])
    acceleration = anisotropic_metric_acceleration(
        z, r, psi, a, b, c, phi, chi_r, chi_z, mass, chi=chi,
        stencil_width=7, lapse=psi,
    )
    phi_tt = anisotropic_scalar_acceleration(
        z, r, psi, a, b, c, phi, mass, lapse=psi, stencil_width=7,
    )
    chi_tt = anisotropic_scalar_acceleration(
        z, r, psi, a, b, c, chi, 0.0, lapse=psi, stencil_width=7,
    )
    trace = spatial_metric_acceleration_trace(acceleration, psi, a, b, c)
    completion = construct_localized_target_lapse_acceleration_completion(
        z, acceleration, psi, psi, a, phi, reference["background"],
        phi_tt, 0.5 * trace, 0.15,
    )
    return {
        "name": str(name),
        "source_grid": [len(z), len(r)],
        "fold_amplitude": float(amplitude),
        "radial_domain": [float(r[0]), float(r[-1])],
        "selector_maximum": float(selected["maximum_residual"]),
        "reference_maximum_residual": float(reference["max_abs_residual"]),
        "z": z,
        "r": r,
        "psi": psi,
        "a": a,
        "b": b,
        "c": c,
        "phi": phi,
        "background": reference["background"],
        "mass_squared": mass,
        "principal": axisymmetric_principal_coefficients(psi, a, b, c),
        "jet_field": RegularSO3BackgroundJetField(
            z, r, psi, psi, a, b, c, phi, chi, acceleration,
            completion["lapse_acceleration"], phi_tt, chi_tt, 7,
        ),
    }


def build_A790_R10_base(selector_iterations=35, slice_iterations=260):
    """Build G5-quality Rmax=10 data from its domain-qualified A=8 seed."""
    if not KNOT_STATE.exists():
        raise FileNotFoundError("the shared-domain A=8 family-knot state is required")
    archive = np.load(KNOT_STATE)
    reference = solve_finite_wall_high_order_slice(
        AMPLITUDE, nz=49, nr=91, r_max=R_MAX, wall_stiffness=20.0,
        epsilon=0.1, backreaction=0.01, tolerance=1e-10,
        iterations=int(slice_iterations),
    )
    coefficients = archive["coefficients"]
    shape = shape_fields(reference["z"], reference["r"], coefficients)
    _, chi_r, chi_z = scalar_pulse(reference["z"], reference["r"], AMPLITUDE)
    selected = solve_anisotropic_initial_data(
        reference["z"], reference["r"], reference["q"], reference["phi"],
        *shape, reference["background"], chi_r, chi_z,
        initial_q=archive["q_G5R10"], initial_phi=archive["phi_G5R10"],
        stencil_width=7, tolerance=1e-9, iterations=int(selector_iterations),
    )
    return _assemble(reference, selected, shape, AMPLITUDE, "G5A790R10")


def build_A790_R10_refined(
    coarse, nz, nr, name, selector_iterations=45, slice_iterations=300,
):
    """Refine within Rmax=10 using only a same-domain interpolated seed."""
    nz = int(nz)
    nr = int(nr)
    if not np.isclose(np.asarray(coarse["r"])[-1], R_MAX):
        raise ValueError("coarse seed must already cover Rmax=10")
    reference = solve_finite_wall_high_order_slice(
        AMPLITUDE, nz=nz, nr=nr, r_max=R_MAX, wall_stiffness=20.0,
        epsilon=0.1, backreaction=0.01, tolerance=1e-10,
        iterations=int(slice_iterations),
    )
    archive = np.load(KNOT_STATE)
    shape = shape_fields(reference["z"], reference["r"], archive["coefficients"])
    coarse_q = 1.0 / np.asarray(coarse["psi"]) - np.asarray(coarse["z"])[:, None]
    initial_q = interpolate(
        coarse_q, coarse["z"], coarse["r"], reference["z"], reference["r"],
    )
    initial_phi = interpolate(
        coarse["phi"], coarse["z"], coarse["r"], reference["z"], reference["r"],
    )
    _, chi_r, chi_z = scalar_pulse(reference["z"], reference["r"], AMPLITUDE)
    selected = solve_anisotropic_initial_data(
        reference["z"], reference["r"], reference["q"], reference["phi"],
        *shape, reference["background"], chi_r, chi_z,
        initial_q=initial_q, initial_phi=initial_phi, stencil_width=7,
        tolerance=1e-9, iterations=int(selector_iterations),
    )
    return _assemble(reference, selected, shape, AMPLITUDE, name)


def build_A790_R10_pair():
    base = build_A790_R10_base()
    g7 = build_A790_R10_refined(
        base, 81, matched_radial_count(121), "G7A790R10",
        selector_iterations=45, slice_iterations=300,
    )
    g8 = build_A790_R10_refined(
        g7, 97, matched_radial_count(145), "G8A790R10",
        selector_iterations=50, slice_iterations=320,
    )
    return g7, g8
