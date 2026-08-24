"""Native field assembly for the joint A=7.90 parent construction."""

from __future__ import annotations

import numpy as np

from bhps.anisotropic_geometry import (
    anisotropic_metric_acceleration,
    anisotropic_scalar_acceleration,
)
from bhps.generalized_harmonic_jets import spatial_metric_acceleration_trace
from bhps.gw_slice_high_order_solver import derivative_matrix


FIELD_COUNT = 9


def native_position_from_primitives(
    z, r, alpha, psi, a, b, c, phi, chi,
):
    """Assemble one time-symmetric native reduced position array."""
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    arrays = tuple(np.asarray(value, dtype=float) for value in (
        alpha, psi, a, b, c, phi, chi,
    ))
    shape = (len(z), len(r))
    if any(value.shape != shape for value in arrays):
        raise ValueError("native primitive fields must share the z-r grid")
    alpha, psi, a, b, c, phi, chi = arrays
    if np.any(alpha <= 0.0) or np.any(psi <= 0.0):
        raise ValueError("native lapse and conformal factor must be positive")
    h_zz = psi**2*np.exp(2.0*a)
    h_rr = psi**2*np.exp(2.0*b)
    h_perp = psi**2*np.exp(2.0*c)
    result = np.zeros((*shape, FIELD_COUNT))
    result[:, :, 2] = -alpha**2
    result[:, :, 3] = h_perp
    result[:, 1:, 4] = (
        h_rr[:, 1:]-h_perp[:, 1:]
    )/r[None, 1:]**2
    s = (r/r[-1])**2
    ds = derivative_matrix(s, 1, 7)
    if hasattr(ds, "toarray"):
        ds = ds.toarray()
    result[:, 0, 4] = (ds @ (h_rr-h_perp).T).T[:, 0]/r[-1]**2
    result[:, :, 6] = h_zz
    result[:, :, 7] = phi
    result[:, :, 8] = chi
    return result


def reconstruct_native_spatial_ansatz(position, r):
    """Recover independent lapse and determinant-defined spatial primitives."""
    q = np.asarray(position, dtype=float)
    r = np.asarray(r, dtype=float)
    if q.ndim != 3 or q.shape[1:] != (len(r), FIELD_COUNT):
        raise ValueError("native position has the wrong shape")
    h00 = q[:, :, 2]
    h_perp = q[:, :, 3]
    h_rr = h_perp+r[None, :]**2*q[:, :, 4]
    h_zz = q[:, :, 6]
    if np.any(h00 >= 0.0) or np.any(
        np.stack((h_perp, h_rr, h_zz)) <= 0.0
    ):
        raise ValueError("native metric has invalid diagonal signs")
    alpha = np.sqrt(-h00)
    psi = (h_zz*h_rr*h_perp**2)**0.125
    a = 0.5*np.log(h_zz/psi**2)
    b = 0.5*np.log(h_rr/psi**2)
    c = 0.5*np.log(h_perp/psi**2)
    tracefree = a+b+2.0*c
    return {
        "alpha": alpha,
        "psi": psi,
        "a": a,
        "b": b,
        "c": c,
        "phi": q[:, :, 7],
        "chi": q[:, :, 8],
        "h_zz": h_zz,
        "h_rr": h_rr,
        "h_perp": h_perp,
        "tracefree_maximum_absolute": float(np.max(np.abs(tracefree))),
        "lapse_conformal_scaled_Linf": float(np.max(
            np.abs(alpha-psi)/np.maximum.reduce((
                np.ones_like(alpha), np.abs(alpha), np.abs(psi),
            ))
        )),
    }


def _regular_even_quotient(numerator, r, power, stencil_width=7):
    numerator = np.asarray(numerator, dtype=float)
    r = np.asarray(r, dtype=float)
    if numerator.ndim != 2 or numerator.shape[1] != len(r) or r[0] != 0.0:
        raise ValueError("invalid regular quotient input")
    result = np.empty_like(numerator)
    result[:, 1:] = numerator[:, 1:]/r[None, 1:]**int(power)
    if int(power) == 1:
        dr = derivative_matrix(r, 1, stencil_width)
        if hasattr(dr, "toarray"):
            dr = dr.toarray()
        result[:, 0] = (dr @ numerator.T).T[:, 0]
    elif int(power) == 2:
        s = (r/r[-1])**2
        ds = derivative_matrix(s, 1, stencil_width)
        if hasattr(ds, "toarray"):
            ds = ds.toarray()
        result[:, 0] = (ds @ numerator.T).T[:, 0]/r[-1]**2
    else:
        raise ValueError("regular quotient power must be one or two")
    return result


def _native_scalar_gradients(field, z, r, stencil_width=7):
    """Differentiate one completed scalar with the native finite differences."""
    field = np.asarray(field, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    if field.shape != (len(z), len(r)):
        raise ValueError("native scalar and z-r grid have incompatible shapes")
    dz = derivative_matrix(z, 1, stencil_width)
    dr = derivative_matrix(r, 1, stencil_width)
    if hasattr(dz, "toarray"):
        dz = dz.toarray()
    if hasattr(dr, "toarray"):
        dr = dr.toarray()
    z_first = dz @ field
    r_first = (dr @ field.T).T
    # The completed scalar is even at the axis.  Enforce the exact native
    # parity value, including an IEEE positive-zero sign, instead of retaining
    # one-sided stencil roundoff there.
    r_first[:, 0] = 0.0
    return z_first, r_first


def bulk_acceleration_from_completed_position(
    position,
    z,
    r,
    background,
    *,
    stencil_width=7,
):
    """Construct the deterministic, unclosed native bulk acceleration seed.

    This stage contains no wall or corner completion.  In particular, its
    lapse seed imposes the frozen bulk gauge target ``partial_t Gamma_0=0`` at
    time symmetry; the acceleration owner may subsequently replace endpoint
    values when it solves the coupled wall system.
    """
    q = np.asarray(position, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    expected = (len(z), len(r), FIELD_COUNT)
    if q.shape != expected:
        raise ValueError("bulk acceleration position has an incompatible shape")
    fields = reconstruct_native_spatial_ansatz(q, r)
    chi_z, chi_r = _native_scalar_gradients(
        fields["chi"], z, r, stencil_width,
    )
    mass = float(background["mass_squared"])
    metric = anisotropic_metric_acceleration(
        z,
        r,
        fields["psi"],
        fields["a"],
        fields["b"],
        fields["c"],
        fields["phi"],
        chi_r,
        chi_z,
        mass,
        chi=fields["chi"],
        stencil_width=stencil_width,
        lapse=fields["alpha"],
    )
    phi_tt = anisotropic_scalar_acceleration(
        z,
        r,
        fields["psi"],
        fields["a"],
        fields["b"],
        fields["c"],
        fields["phi"],
        mass,
        lapse=fields["alpha"],
        stencil_width=stencil_width,
    )
    chi_tt = anisotropic_scalar_acceleration(
        z,
        r,
        fields["psi"],
        fields["a"],
        fields["b"],
        fields["c"],
        fields["chi"],
        0.0,
        lapse=fields["alpha"],
        stencil_width=stencil_width,
    )
    trace = spatial_metric_acceleration_trace(
        metric,
        fields["psi"],
        fields["a"],
        fields["b"],
        fields["c"],
    )
    target_relative_lapse_acceleration = 0.5*trace
    lapse_acceleration = (
        fields["alpha"]*target_relative_lapse_acceleration
    )
    metric_tt_acceleration = (
        -2.0*fields["alpha"]*lapse_acceleration
    )
    result = np.zeros_like(q)
    result[:, :, 1] = _regular_even_quotient(
        metric["zr"], r, 1, stencil_width,
    )
    result[:, :, 2] = metric_tt_acceleration
    result[:, :, 3] = metric["transverse"]
    result[:, :, 4] = _regular_even_quotient(
        metric["radial"]-metric["transverse"], r, 2, stencil_width,
    )
    result[:, :, 6] = metric["zz"]
    result[:, :, 7] = phi_tt
    result[:, :, 8] = chi_tt
    if not np.all(np.isfinite(result)):
        raise RuntimeError("native bulk acceleration is nonfinite")
    return result, {
        "spatial_reconstruction": fields,
        "native_chi_gradients": {
            "z": chi_z,
            "r": chi_r,
            "stencil_width": int(stencil_width),
            "axis_radial_positive_zero": bool(
                np.all(chi_r[:, 0] == 0.0)
                and not np.any(np.signbit(chi_r[:, 0]))
            ),
        },
        "spatial_metric_acceleration": {
            name: np.asarray(metric[name], dtype=float)
            for name in ("zz", "radial", "transverse", "zr")
        },
        "lapse_seed": {
            "method": "unclosed_bulk_Gamma0_t_zero",
            "spatial_metric_acceleration_trace": trace,
            "target_relative_lapse_acceleration": (
                target_relative_lapse_acceleration
            ),
            "lapse_acceleration": lapse_acceleration,
            "metric_tt_acceleration": metric_tt_acceleration,
            "wall_completion_applied": False,
        },
        "lapse_is_independent_of_spatial_conformal_factor": True,
        "compact_wall_completion_applied": False,
    }
