"""Zero-step nonlinear scalar wall-acceleration compatibility."""

from __future__ import annotations

import numpy as np


def _validate_fields(dz, compact_scale, compact_metric_acceleration, phi, phi_acceleration, chi_acceleration):
    compact_scale = np.asarray(compact_scale, dtype=float)
    compact_metric_acceleration = np.asarray(compact_metric_acceleration, dtype=float)
    phi = np.asarray(phi, dtype=float)
    phi_acceleration = np.asarray(phi_acceleration, dtype=float)
    chi_acceleration = np.asarray(chi_acceleration, dtype=float)
    shape = compact_scale.shape
    if any(field.shape != shape for field in (compact_metric_acceleration, phi, phi_acceleration, chi_acceleration)):
        raise ValueError("scalar wall fields must share one z-r grid")
    dz = np.asarray(dz.toarray() if hasattr(dz, "toarray") else dz, dtype=float)
    if dz.shape != (shape[0], shape[0]) or np.any(compact_scale <= 0):
        raise ValueError("invalid compact derivative or scale")
    return dz, compact_scale, compact_metric_acceleration, phi, phi_acceleration, chi_acceleration


def scalar_wall_second_corner_fields(
    dz, compact_scale, compact_metric_acceleration, phi, phi_acceleration,
    chi_acceleration, background, radial_buffer=7,
):
    """Evaluate twice-time-differentiated scalar wall conditions.

    In the fixed compact coordinate, the stabilizer boundary row is

    ``Phi_z + s gamma (Phi-v) A/2 = 0``,

    where ``A=sqrt(g_zz)`` and ``s=(-1,+1)`` at the lower/upper wall.
    On time-symmetric data its second time derivative is

    ``Phi_tt,z + s gamma [A Phi_tt + (Phi-v) g_zz,tt/(2A)]/2``.

    The collapse scalar has reflecting ``chi_z=0`` data, hence
    ``chi_tt,z=0`` at the same corner.
    """
    dz, A, gzz_tt, phi, phi_tt, chi_tt = _validate_fields(
        dz, compact_scale, compact_metric_acceleration, phi,
        phi_acceleration, chi_acceleration,
    )
    gamma = float(background["wall_stiffness"])
    buffer = int(radial_buffer)
    retained = slice(None, -buffer) if buffer else slice(None)
    walls = []
    for wall, index, target, sign in (
        ("lower", 0, float(background["v0"]), -1.0),
        ("upper", -1, float(background["v1"]), 1.0),
    ):
        phi_terms = (
            (dz @ phi_tt)[index],
            sign * 0.5 * gamma * A[index] * phi_tt[index],
            sign * 0.25 * gamma * (phi[index] - target) * gzz_tt[index] / A[index],
        )
        phi_residual = sum(phi_terms)
        phi_scale = np.maximum(1.0, sum(np.abs(term) for term in phi_terms))
        chi_residual = (dz @ chi_tt)[index]
        chi_scale = np.maximum(1.0, np.abs(chi_residual))
        walls.append({
            "wall": wall,
            "phi_residual": np.asarray(phi_residual[retained]),
            "phi_scale": np.asarray(phi_scale[retained]),
            "chi_residual": np.asarray(chi_residual[retained]),
            "chi_scale": np.asarray(chi_scale[retained]),
        })
    return {"walls": walls, "radial_buffer": buffer}


def solve_scalar_wall_accelerations(
    dz, compact_scale, compact_metric_acceleration, phi, phi_acceleration,
    chi_acceleration, background,
):
    """Solve stabilizer Robin and collapse-scalar Neumann endpoint rows."""
    dz, A, gzz_tt, phi, phi_tt_input, chi_tt_input = _validate_fields(
        dz, compact_scale, compact_metric_acceleration, phi,
        phi_acceleration, chi_acceleration,
    )
    phi_tt = phi_tt_input.copy()
    chi_tt = chi_tt_input.copy()
    gamma = float(background["wall_stiffness"])
    corrections = []
    for wall, index, target, sign in (
        ("lower", 0, float(background["v0"]), -1.0),
        ("upper", -1, float(background["v1"]), 1.0),
    ):
        diagonal = float(dz[index, index])
        phi_without = (dz @ phi_tt)[index] - diagonal * phi_tt[index]
        phi_forcing = sign * 0.25 * gamma * (phi[index] - target) * gzz_tt[index] / A[index]
        phi_denominator = diagonal + sign * 0.5 * gamma * A[index]
        if np.any(np.abs(phi_denominator) < 1e-12) or abs(diagonal) < 1e-12:
            raise RuntimeError("degenerate scalar wall acceleration row")
        old_phi = phi_tt[index].copy()
        old_chi = chi_tt[index].copy()
        phi_tt[index] = -(phi_without + phi_forcing) / phi_denominator
        chi_without = (dz @ chi_tt)[index] - diagonal * chi_tt[index]
        chi_tt[index] = -chi_without / diagonal
        before = np.concatenate((old_phi, old_chi))
        after = np.concatenate((phi_tt[index], chi_tt[index]))
        corrections.append({
            "wall": wall,
            "relative_norm": float(
                np.linalg.norm(after - before)
                / max(np.linalg.norm(after), np.linalg.norm(before), 1e-300)
            ),
            "maximum_absolute": float(np.max(np.abs(after - before))),
        })
    return {
        "phi_acceleration": phi_tt,
        "chi_acceleration": chi_tt,
        "corrections": corrections,
    }
