"""Helpers for the sealed finite-amplitude, domain-qualified basin map.

The scientific classification functions in this module are deliberately pure.
The Rmax=12 builders generalize the audited A=7.90 construction without
changing its shared-domain A=8 shape coefficients or extrapolating a smaller
domain into the exterior.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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


CANDIDATE_AMPLITUDES = (7.84, 7.86, 7.88, 7.90, 7.92)
ADVERSE_CONTROL_AMPLITUDE = 7.94
DOMAIN_ANCHORS = (7.84, 7.90, 7.92)
R_MAX = 12.0
BASIS_RADIUS = 8.0
AXIS_WIDTHS = (0.5, 1.0)
ANNULAR_PROFILES = ((7.5, 1.5), (7.5, 3.0))
KNOT_STATE = Path("results/corrected_family_knot_A8_state.npz")


def amplitude_tag(amplitude: float) -> str:
    """Return the repository's fixed two-decimal amplitude tag."""
    return f"A{int(round(100.0 * float(amplitude))):03d}"


def matched_radial_count(
    reference_count: int, reference_rmax: float = 8.0,
    target_rmax: float = R_MAX,
) -> int:
    """Return a radial count preserving the reference interval spacing."""
    intervals = int(reference_count) - 1
    target_intervals = intervals * float(target_rmax) / float(reference_rmax)
    rounded = int(round(target_intervals))
    if not np.isclose(target_intervals, rounded, rtol=0.0, atol=1e-12):
        raise ValueError("target domain is not commensurate with reference spacing")
    return rounded + 1


def persistent_late_pair(counts: Mapping[str, int]) -> bool:
    """Require exactly two branches on both grids at t=.003 and t=.004."""
    return all(int(counts.get(key, -1)) == 2 for key in (
        "G7_t0.003", "G8_t0.003", "G7_t0.004", "G8_t0.004",
    ))


def sampled_onset(times: Sequence[float], counts: Sequence[int]) -> dict:
    """Report a sampled persistent 0->2 onset without treating it as primary."""
    values = [int(value) for value in counts]
    sample_times = [float(value) for value in times]
    if len(values) != len(sample_times):
        raise ValueError("times and counts must have equal length")
    positives = [index for index, value in enumerate(values) if value > 0]
    if not positives:
        return {"classification": "no_pair_in_window", "bracket": None}
    first = positives[0]
    if any(value != 0 for value in values[:first]) or any(
        value != 2 for value in values[first:]
    ):
        return {"classification": "nonpersistent_or_extra_branch", "bracket": None}
    lower = 0.0 if first == 0 else sample_times[first - 1]
    return {
        "classification": "sampled_persistent_zero_to_two",
        "bracket": {"lower": lower, "upper": sample_times[first]},
    }


def monotone_onset_diagnostic(records: Mapping[str, Mapping]) -> dict:
    """Diagnose, but never gate on, amplitude ordering of sampled onsets."""
    ordered = []
    for amplitude in CANDIDATE_AMPLITUDES:
        record = records.get(amplitude_tag(amplitude), {})
        bracket = record.get("sampled_onset", {}).get("bracket")
        ordered.append(None if bracket is None else float(bracket["upper"]))
    defined = [value for value in ordered if value is not None]
    nonincreasing = all(
        right <= left + 1e-15 for left, right in zip(defined, defined[1:])
    )
    return {
        "amplitudes": list(CANDIDATE_AMPLITUDES),
        "first_detection_times": ordered,
        "nonincreasing_where_defined": bool(nonincreasing),
        "claim_role": "secondary_domain_and_foliation_qualified_diagnostic",
    }


def aggregate_basin_status(
    fixed_domain_records: Mapping[str, Mapping],
    domain_anchor_records: Mapping[str, Mapping],
    adverse_control_pass: bool,
) -> dict:
    """Adjudicate the finite map while keeping domain support explicit."""
    candidate_tags = [amplitude_tag(value) for value in CANDIDATE_AMPLITUDES]
    anchor_tags = [amplitude_tag(value) for value in DOMAIN_ANCHORS]
    fixed_complete = all(tag in fixed_domain_records for tag in candidate_tags)
    anchors_complete = all(tag in domain_anchor_records for tag in anchor_tags)
    hard_failure = any(
        bool(record.get("hard_failure"))
        for record in tuple(fixed_domain_records.values())
        + tuple(domain_anchor_records.values())
    ) or not adverse_control_pass
    fixed_pass = fixed_complete and all(
        bool(fixed_domain_records[tag].get("primary_pass"))
        for tag in candidate_tags
    )
    anchor_pass = anchors_complete and all(
        bool(domain_anchor_records[tag].get("primary_pass"))
        for tag in anchor_tags
    )
    if hard_failure:
        status = "fail"
        classification = "invalid_or_detector_control_failure"
    elif fixed_pass and anchor_pass:
        status = "pass"
        classification = "domain_qualified_five_point_sampled_basin"
    else:
        status = "review"
        classification = (
            "fixed_domain_sample_supported_domain_qualification_incomplete"
            if fixed_pass else "sampled_boundary_or_incomplete_basin"
        )
    return {
        "status": status,
        "classification": classification,
        "fixed_domain_complete": bool(fixed_complete),
        "fixed_domain_primary_pass": bool(fixed_pass),
        "domain_anchors_complete": bool(anchors_complete),
        "domain_anchor_primary_pass": bool(anchor_pass),
        "adverse_control_pass": bool(adverse_control_pass),
    }


def _interpolate(field, source_z, source_r, target_z, target_r):
    zz, rr = np.meshgrid(target_z, target_r, indexing="ij")
    return RectBivariateSpline(
        source_z, source_r, np.asarray(field), kx=3, ky=3, s=0,
    ).ev(zz.ravel(), rr.ravel()).reshape(len(target_z), len(target_r))


def _shape_fields(z, r, coefficients):
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
        "z": z, "r": r, "psi": psi, "a": a, "b": b, "c": c,
        "phi": phi, "background": reference["background"],
        "mass_squared": mass,
        "principal": axisymmetric_principal_coefficients(psi, a, b, c),
        "jet_field": RegularSO3BackgroundJetField(
            z, r, psi, psi, a, b, c, phi, chi, acceleration,
            completion["lapse_acceleration"], phi_tt, chi_tt, 7,
        ),
    }


def build_R12_base(amplitude, selector_iterations=35, slice_iterations=280):
    """Build fresh Rmax=12 data from the domain-qualified shared A=8 knot."""
    amplitude = float(amplitude)
    if not KNOT_STATE.exists():
        raise FileNotFoundError("the shared-domain A=8 family-knot state is required")
    with np.load(KNOT_STATE) as archive:
        required = ("coefficients", "q_G5R12", "phi_G5R12")
        missing = [name for name in required if name not in archive.files]
        if missing:
            raise KeyError(f"domain-qualified Rmax=12 seed is incomplete: {missing}")
        coefficients = np.asarray(archive["coefficients"])
        initial_q = np.asarray(archive["q_G5R12"])
        initial_phi = np.asarray(archive["phi_G5R12"])
    reference = solve_finite_wall_high_order_slice(
        amplitude, nz=49, nr=109, r_max=R_MAX, wall_stiffness=20.0,
        epsilon=0.1, backreaction=0.01, tolerance=1e-10,
        iterations=int(slice_iterations),
    )
    shape = _shape_fields(reference["z"], reference["r"], coefficients)
    _, chi_r, chi_z = scalar_pulse(reference["z"], reference["r"], amplitude)
    selected = solve_anisotropic_initial_data(
        reference["z"], reference["r"], reference["q"], reference["phi"],
        *shape, reference["background"], chi_r, chi_z,
        initial_q=initial_q, initial_phi=initial_phi, stencil_width=7,
        tolerance=1e-9, iterations=int(selector_iterations),
    )
    return _assemble(
        reference, selected, shape, amplitude,
        f"G5{amplitude_tag(amplitude)}R12-basin",
    )


def build_R12_refined(
    coarse, amplitude, nz, nr, name, selector_iterations=45,
    slice_iterations=320,
):
    """Refine an amplitude on Rmax=12 using only its same-domain seed."""
    amplitude = float(amplitude)
    nz, nr = int(nz), int(nr)
    if not np.isclose(np.asarray(coarse["r"])[-1], R_MAX):
        raise ValueError("coarse seed must already cover Rmax=12")
    reference = solve_finite_wall_high_order_slice(
        amplitude, nz=nz, nr=nr, r_max=R_MAX, wall_stiffness=20.0,
        epsilon=0.1, backreaction=0.01, tolerance=1e-10,
        iterations=int(slice_iterations),
    )
    with np.load(KNOT_STATE) as archive:
        shape = _shape_fields(
            reference["z"], reference["r"], archive["coefficients"],
        )
    coarse_q = 1.0 / np.asarray(coarse["psi"]) - np.asarray(coarse["z"])[:, None]
    initial_q = _interpolate(
        coarse_q, coarse["z"], coarse["r"], reference["z"], reference["r"],
    )
    initial_phi = _interpolate(
        coarse["phi"], coarse["z"], coarse["r"],
        reference["z"], reference["r"],
    )
    _, chi_r, chi_z = scalar_pulse(reference["z"], reference["r"], amplitude)
    selected = solve_anisotropic_initial_data(
        reference["z"], reference["r"], reference["q"], reference["phi"],
        *shape, reference["background"], chi_r, chi_z,
        initial_q=initial_q, initial_phi=initial_phi, stencil_width=7,
        tolerance=1e-9, iterations=int(selector_iterations),
    )
    return _assemble(reference, selected, shape, amplitude, name)


def build_R12_pair(amplitude):
    """Build the sealed matched-spacing G7/G8 Rmax=12 pair."""
    tag = amplitude_tag(amplitude)
    base = build_R12_base(amplitude)
    g7 = build_R12_refined(
        base, amplitude, 81, matched_radial_count(121), f"G7{tag}R12-basin",
        selector_iterations=45, slice_iterations=320,
    )
    g8 = build_R12_refined(
        g7, amplitude, 97, matched_radial_count(145), f"G8{tag}R12-basin",
        selector_iterations=50, slice_iterations=340,
    )
    return g7, g8
