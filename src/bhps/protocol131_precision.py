"""Independent extended-precision arithmetic for Protocol 131.

The production selector is intentionally a binary64 implementation.  This
module provides an arithmetic-only control: it promotes the *already formed*
binary64 width-seven finite-difference weights to :class:`numpy.longdouble`
and independently re-evaluates the complete joint-parent residual.  It does
not solve for, update, or project a parent state.

Keeping the derivative weights fixed is important.  The comparison isolates
roundoff in residual evaluation from a change of discretization.  On systems
where ``np.longdouble`` is not wider than binary64, callers must treat the
result as a replay rather than an extended-precision control; use
:func:`longdouble_capability` to record that fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from bhps.gw_slice_high_order_solver import derivative_matrix


STENCIL_WIDTH = 7
_LD = np.longdouble


def longdouble_capability():
    """Return machine-readable information about the local long-double type."""
    information = np.finfo(np.longdouble)
    binary64 = np.finfo(np.float64)
    return {
        "dtype": np.dtype(np.longdouble).name,
        "itemsize": int(np.dtype(np.longdouble).itemsize),
        "epsilon": np.longdouble(information.eps),
        "mantissa_bits": int(information.nmant),
        "wider_than_float64": bool(information.eps < binary64.eps),
    }


def compensated_dot(left, right):
    """Return a long-double dot product with Neumaier compensation.

    The arrays must have the same shape and are treated as flattened real
    vectors.  Products and the compensated accumulation are both performed in
    ``np.longdouble``.  The compensation corrects summation loss; it is not an
    error-free two-product algorithm.
    """
    left = np.asarray(left, dtype=np.longdouble)
    right = np.asarray(right, dtype=np.longdouble)
    if left.shape != right.shape:
        raise ValueError("compensated_dot inputs must have the same shape")
    total = _LD(0.0)
    correction = _LD(0.0)
    for first, second in zip(left.ravel(order="C"), right.ravel(order="C")):
        term = first * second
        updated = total + term
        if abs(total) >= abs(term):
            correction += (total - updated) + term
        else:
            correction += (term - updated) + total
        total = updated
    return _LD(total + correction)


def _compensated_sum(terms, axis=0):
    """Vectorized Neumaier accumulation along one short stencil axis."""
    terms = np.asarray(terms, dtype=np.longdouble)
    moved = np.moveaxis(terms, axis, 0)
    total = np.zeros(moved.shape[1:], dtype=np.longdouble)
    correction = np.zeros_like(total)
    for term in moved:
        updated = total + term
        correction += np.where(
            np.abs(total) >= np.abs(term),
            (total - updated) + term,
            (term - updated) + total,
        )
        total = updated
    return total + correction


@dataclass(frozen=True)
class _Stencil:
    indices: np.ndarray
    weights: np.ndarray


def _fixed_float64_stencil(coordinates, derivative_order, width):
    """Build the production binary64 weights, then promote without refitting."""
    coordinates64 = np.asarray(coordinates, dtype=np.float64)
    matrix = derivative_matrix(coordinates64, derivative_order, width).toarray()
    size = len(coordinates64)
    starts = np.minimum(
        np.maximum(np.arange(size) - width // 2, 0), size - width,
    )
    indices = starts[:, None] + np.arange(width)[None, :]
    weights = matrix[np.arange(size)[:, None], indices]
    return _Stencil(
        indices=np.asarray(indices, dtype=np.int64),
        weights=np.asarray(weights, dtype=np.longdouble),
    )


@dataclass(frozen=True)
class _Operators:
    z_first: _Stencil
    z_second: _Stencil
    r_first: _Stencil
    r_second: _Stencil
    r: np.ndarray

    def dz(self, field):
        field = np.asarray(field, dtype=np.longdouble)
        gathered = field[self.z_first.indices, :]
        return _compensated_sum(
            self.z_first.weights[:, :, None] * gathered, axis=1,
        )

    def dzz(self, field):
        field = np.asarray(field, dtype=np.longdouble)
        gathered = field[self.z_second.indices, :]
        return _compensated_sum(
            self.z_second.weights[:, :, None] * gathered, axis=1,
        )

    def dr(self, field):
        field = np.asarray(field, dtype=np.longdouble)
        gathered = field[:, self.r_first.indices]
        return _compensated_sum(
            self.r_first.weights[None, :, :] * gathered, axis=2,
        )

    def drr(self, field):
        field = np.asarray(field, dtype=np.longdouble)
        gathered = field[:, self.r_second.indices]
        return _compensated_sum(
            self.r_second.weights[None, :, :] * gathered, axis=2,
        )

    def radial(self, field):
        first = self.dr(field)
        second = self.drr(field)
        result = second.copy()
        result[:, 0] = _LD(3.0) * second[:, 0]
        result[:, 1:] = (
            second[:, 1:] + _LD(2.0) * first[:, 1:] / self.r[None, 1:]
        )
        return result

    def derivatives(self, field):
        first_z = self.dz(field)
        first_r = self.dr(field)
        second_z = self.dzz(field)
        second_r = self.drr(field)
        transverse = np.zeros_like(first_r)
        transverse[:, 1:] = first_r[:, 1:] / self.r[None, 1:]
        transverse[:, 0] = second_r[:, 0]
        # Match the left-to-right production expression Dz @ field @ Dr.T.
        mixed = self.dr(first_z)
        return {
            "z": first_z,
            "r": first_r,
            "zz": second_z,
            "rr": second_r,
            "zr": mixed,
            "transverse_hessian": transverse,
        }


def _operators(z, r, width):
    return _Operators(
        z_first=_fixed_float64_stencil(z, 1, width),
        z_second=_fixed_float64_stencil(z, 2, width),
        r_first=_fixed_float64_stencil(r, 1, width),
        r_second=_fixed_float64_stencil(r, 2, width),
        r=np.asarray(r, dtype=np.longdouble),
    )


def _validated_inputs(
    q,
    phi,
    z,
    r,
    a,
    b,
    c,
    background,
    chi_r,
    chi_z,
    reference_q,
    reference_phi,
    stencil_width,
):
    width = int(stencil_width)
    if width != stencil_width or width != STENCIL_WIDTH:
        raise ValueError("Protocol-131 precision control requires stencil_width=7")
    z64 = np.asarray(z, dtype=np.float64)
    r64 = np.asarray(r, dtype=np.float64)
    if z64.ndim != 1 or r64.ndim != 1:
        raise ValueError("z and r must be one-dimensional")
    if len(z64) < width or len(r64) < width:
        raise ValueError("both grids must support a seven-point stencil")
    if np.any(np.diff(z64) <= 0.0) or np.any(np.diff(r64) <= 0.0):
        raise ValueError("z and r must be strictly increasing")
    if not np.isclose(r64[0], 0.0):
        raise ValueError("the regular radial grid must begin at the axis")
    shape = (len(z64), len(r64))
    fields = tuple(
        np.asarray(value, dtype=np.longdouble)
        for value in (
            q, phi, a, b, c, chi_r, chi_z, reference_q, reference_phi,
        )
    )
    if any(value.shape != shape for value in fields):
        raise ValueError("all fields must share the z-r grid shape")
    if not all(np.all(np.isfinite(value)) for value in fields):
        raise ValueError("all fields must be finite")
    if not np.all(np.isfinite(z64)) or not np.all(np.isfinite(r64)):
        raise ValueError("grid coordinates must be finite")
    zld = np.asarray(z64, dtype=np.longdouble)
    rld = np.asarray(r64, dtype=np.longdouble)
    if np.any(zld[:, None] + fields[0] <= _LD(0.0)):
        raise ValueError("z + q must remain positive")
    required = (
        "mass_squared", "wall_stiffness", "v0", "v1", "beta_a",
        "beta_b", "wall_potential_a", "wall_potential_b",
    )
    missing = [name for name in required if name not in background]
    if missing:
        raise KeyError(f"background lacks required coefficients: {missing}")
    coefficients = {name: _LD(background[name]) for name in required}
    if not all(np.isfinite(value) for value in coefficients.values()):
        raise ValueError("background coefficients must be finite")
    if coefficients["wall_stiffness"] < _LD(0.0):
        raise ValueError("wall stiffness must be nonnegative")
    return (*fields, zld, rld, coefficients, width)


def _raw_geometry(psi, a, b, c, operators):
    """Warped-product geometry before the conformal well-balance correction."""
    radius = operators.r
    aa = psi * np.exp(a)
    bb = psi * np.exp(b)
    cc = psi * np.exp(c)
    rho = cc * radius[None, :]
    da = operators.derivatives(aa)
    db = operators.derivatives(bb)
    dc = operators.derivatives(cc)
    drho = operators.derivatives(rho)

    bz_over_a = db["z"] / aa
    ar_over_b = da["r"] / bb
    base_scalar = -_LD(2.0) * (
        operators.dz(bz_over_a) + operators.dr(ar_over_b)
    ) / (aa * bb)

    gamma_z_zz = da["z"] / aa
    gamma_z_zr = da["r"] / aa
    gamma_z_rr = -bb * db["z"] / aa**2
    gamma_r_zz = -aa * da["r"] / bb**2
    gamma_r_zr = db["z"] / bb
    gamma_r_rr = db["r"] / bb
    hess_rho_zz = (
        drho["zz"] - gamma_z_zz * drho["z"] - gamma_r_zz * drho["r"]
    )
    hess_rho_rr = (
        drho["rr"] - gamma_z_rr * drho["z"] - gamma_r_rr * drho["r"]
    )
    hess_rho_zr = (
        drho["zr"] - gamma_z_zr * drho["z"] - gamma_r_zr * drho["r"]
    )

    safe_rho = np.where(rho != _LD(0.0), rho, _LD(1.0))
    hzz_over_rho = hess_rho_zz / safe_rho
    hrr_over_rho = hess_rho_rr / safe_rho
    hzr_over_rho = hess_rho_zr / safe_rho
    hzz_over_rho[:, 0] = (
        dc["zz"][:, 0] / cc[:, 0]
        - da["z"][:, 0] * dc["z"][:, 0] / (aa[:, 0] * cc[:, 0])
        + aa[:, 0] * da["rr"][:, 0] / bb[:, 0] ** 2
    )
    hrr_over_rho[:, 0] = (
        _LD(3.0) * dc["rr"][:, 0] / cc[:, 0]
        + bb[:, 0] * db["z"][:, 0] * dc["z"][:, 0]
        / (aa[:, 0] ** 2 * cc[:, 0])
        - db["rr"][:, 0] / bb[:, 0]
    )
    hzr_over_rho[:, 0] = _LD(0.0)

    laplacian_rho = hess_rho_zz / aa**2 + hess_rho_rr / bb**2
    gradient_rho_squared = drho["z"] ** 2 / aa**2 + drho["r"] ** 2 / bb**2
    ricci_zz = _LD(0.5) * base_scalar * aa**2 - _LD(2.0) * hzz_over_rho
    ricci_rr = _LD(0.5) * base_scalar * bb**2 - _LD(2.0) * hrr_over_rho
    ricci_zr = -_LD(2.0) * hzr_over_rho
    fiber_numerator = (
        _LD(1.0) - rho * laplacian_rho - gradient_rho_squared
    )
    ricci_transverse = np.zeros_like(fiber_numerator)
    np.divide(
        fiber_numerator,
        radius[None, :] ** 2,
        out=ricci_transverse,
        where=radius[None, :] != _LD(0.0),
    )
    ricci_transverse[:, 0] = ricci_rr[:, 0]
    return {
        "aa": aa,
        "bb": bb,
        "cc": cc,
        "ricci_zz": ricci_zz,
        "ricci_rr": ricci_rr,
        "ricci_transverse": ricci_transverse,
        "ricci_zr": ricci_zr,
    }


def _axisymmetric_scalar_curvature(psi, a, b, c, operators):
    """Long-double mirror of ``axisymmetric_diagonal_geometry`` curvature."""
    raw = _raw_geometry(psi, a, b, c, operators)
    zeros = np.zeros_like(psi)
    baseline = _raw_geometry(psi, zeros, zeros, zeros, operators)
    w = operators.derivatives(np.log(psi))
    laplacian = w["zz"] + w["rr"] + _LD(2.0) * w["transverse_hessian"]
    gradient_squared = w["z"] ** 2 + w["r"] ** 2
    common = laplacian + _LD(2.0) * gradient_squared
    exact_zz = -_LD(2.0) * (w["zz"] - w["z"] ** 2) - common
    exact_rr = -_LD(2.0) * (w["rr"] - w["r"] ** 2) - common
    exact_transverse = -_LD(2.0) * w["transverse_hessian"] - common
    ricci_zz = raw["ricci_zz"] + exact_zz - baseline["ricci_zz"]
    ricci_rr = raw["ricci_rr"] + exact_rr - baseline["ricci_rr"]
    ricci_transverse = (
        raw["ricci_transverse"] + exact_transverse
        - baseline["ricci_transverse"]
    )
    return (
        ricci_zz / raw["aa"] ** 2
        + ricci_rr / raw["bb"] ** 2
        + _LD(2.0) * ricci_transverse / raw["cc"] ** 2
    )


def _shape_laplacian(field, a, b, operators):
    a_z = operators.dz(a)
    b_r = operators.dr(b)
    inverse_a = np.exp(-_LD(2.0) * a)
    inverse_b = np.exp(-_LD(2.0) * b)
    dz_field = operators.dz(field)
    dr_field = operators.dr(field)
    return (
        inverse_a * (operators.dzz(field) - _LD(2.0) * a_z * dz_field)
        + inverse_b * (operators.radial(field) - _LD(2.0) * b_r * dr_field)
    )


def _wall_coefficients(phi_row, background, upper):
    gamma = background["wall_stiffness"]
    if upper:
        orientation = _LD(1.0)
        target = background["v1"]
        delta = phi_row - target
        potential = _LD(0.5) * gamma * delta**2
        beta = background["beta_b"] - (
            potential - background["wall_potential_b"]
        ) / _LD(6.0)
    else:
        orientation = _LD(-1.0)
        target = background["v0"]
        delta = phi_row - target
        potential = _LD(0.5) * gamma * delta**2
        beta = background["beta_a"] + (
            potential - background["wall_potential_a"]
        ) / _LD(6.0)
    return orientation, target, delta, beta


def _raw_anisotropic_residual(
    q,
    phi,
    z,
    r,
    a,
    b,
    c,
    background,
    chi_r,
    chi_z,
    reference_q,
    reference_phi,
    operators,
):
    """Long-double mirror of the raw anisotropic residual."""
    psi = _LD(1.0) / (z[:, None] + q)
    phi_z = operators.dz(phi)
    phi_r = operators.dr(phi)
    inverse_a = np.exp(-_LD(2.0) * a)
    inverse_b = np.exp(-_LD(2.0) * b)
    gradient = (
        inverse_a * (phi_z**2 + chi_z**2)
        + inverse_b * (phi_r**2 + chi_r**2)
    )
    scalar_bar = _axisymmetric_scalar_curvature(
        np.ones_like(psi), a, b, c, operators,
    )
    mass = background["mass_squared"]
    cosmological_constant = _LD(-6.0)
    potential = mass * phi**2
    hamiltonian = (
        -_LD(6.0) * _shape_laplacian(psi, a, b, operators)
        + (scalar_bar - gradient) * psi
        - (_LD(2.0) * cosmological_constant + potential) * psi**3
    )
    log_psi_z = operators.dz(np.log(psi))
    log_psi_r = operators.dr(np.log(psi))
    scalar = (
        _shape_laplacian(phi, a, b, operators)
        + _LD(3.0) * (
            inverse_a * log_psi_z * phi_z
            + inverse_b * log_psi_r * phi_r
        )
        - mass * psi**2 * phi
    )

    gamma = background["wall_stiffness"]
    exp_a = np.exp(a)
    psi_z = operators.dz(psi)
    b_z = operators.dz(b)
    for index, upper in ((0, False), (-1, True)):
        orientation, _, delta, beta = _wall_coefficients(
            phi[index], background, upper,
        )
        hamiltonian[index, :-1] = (
            psi_z[index] + psi[index] * b_z[index]
            + beta * psi[index] ** 2 * exp_a[index]
        )[:-1]
        # The legacy scalar sign is equal to the outward orientation.
        scalar[index, :-1] = (
            phi_z[index]
            + orientation * _LD(0.5) * gamma * delta * psi[index] * exp_a[index]
        )[:-1]

    q_r = operators.dr(q)
    reference_delta_q = q - reference_q
    phi_r = operators.dr(phi)
    reference_delta_phi = phi - reference_phi
    hamiltonian[:, -1] = (
        q_r[:, -1] + reference_delta_q[:, -1] / r[-1]
    )
    scalar[:, -1] = (
        phi_r[:, -1] + reference_delta_phi[:, -1] / r[-1]
    )
    return np.concatenate((hamiltonian.ravel(order="C"), scalar.ravel(order="C")))


def joint_parent_residual_longdouble(
    q,
    phi,
    z,
    r,
    a,
    b,
    c,
    background,
    chi_r,
    chi_z,
    reference_q,
    reference_phi,
    stencil_width=STENCIL_WIDTH,
):
    """Evaluate the complete hybrid joint-parent residual in long double.

    The result has the same two-block, C-order flattening as
    :func:`bhps.joint_parent_builder.joint_parent_residual`.  The first block
    is the metric selector and the second is the Phi selector.  Open compact
    nodes contain the raw-minus-reference anisotropic residual.  Both compact
    walls, including the radial corners, are overwritten by the normalized
    native metric junction row and the absolute Phi Robin row.
    """
    (
        q,
        phi,
        a,
        b,
        c,
        chi_r,
        chi_z,
        reference_q,
        reference_phi,
        z,
        r,
        background,
        width,
    ) = _validated_inputs(
        q,
        phi,
        z,
        r,
        a,
        b,
        c,
        background,
        chi_r,
        chi_z,
        reference_q,
        reference_phi,
        stencil_width,
    )
    operators = _operators(z, r, width)
    raw = _raw_anisotropic_residual(
        q, phi, z, r, a, b, c, background, chi_r, chi_z,
        reference_q, reference_phi, operators,
    )
    zeros = np.zeros_like(a)
    defect = _raw_anisotropic_residual(
        reference_q, reference_phi, z, r, zeros, zeros, zeros, background,
        chi_r, chi_z, reference_q, reference_phi, operators,
    )
    residual = raw - defect
    size = q.size
    metric_rows = residual[:size].reshape(q.shape)
    phi_rows = residual[size:].reshape(q.shape)

    psi = _LD(1.0) / (z[:, None] + q)
    sphere_metric = psi**2 * np.exp(_LD(2.0) * c)
    normal_scale = psi * np.exp(a)
    derivative_metric = operators.dz(sphere_metric)
    derivative_phi = operators.dz(phi)
    gamma = background["wall_stiffness"]
    for index, upper in ((0, False), (-1, True)):
        orientation, _, delta, beta = _wall_coefficients(
            phi[index], background, upper,
        )
        numerator = (
            derivative_metric[index]
            + _LD(2.0) * beta * normal_scale[index] * sphere_metric[index]
        )
        metric_rows[index] = (
            orientation * numerator / (_LD(2.0) * normal_scale[index])
        )
        phi_rows[index] = (
            derivative_phi[index]
            + orientation * _LD(0.5) * gamma * delta * normal_scale[index]
        )
    if residual.dtype != np.dtype(np.longdouble):
        raise RuntimeError("long-double evaluation unexpectedly narrowed")
    return residual


# Descriptive alias for callers that naturally put the precision first.
longdouble_joint_parent_residual = joint_parent_residual_longdouble


def _mp_exact(value, mp):
    """Convert a finite binary floating value to mpmath without decimal loss."""
    scalar = np.asarray(value).reshape(()).item()
    numerator, denominator = scalar.as_integer_ratio()
    return mp.mpf(numerator) / mp.mpf(denominator)


def reevaluate_wall_row_mpmath(
    q,
    phi,
    z,
    a,
    c,
    background: Mapping[str, object],
    wall,
    radial_index,
    stencil_width=STENCIL_WIDTH,
    dps=80,
):
    """Re-evaluate one normalized wall pair with at least 80 decimal digits.

    The width-seven derivative weights are still the production binary64
    weights, converted exactly to ``mpmath`` numbers.  Thus the calculation
    diagnoses arithmetic cancellation in a fixed discrete row; it is not an
    independent high-precision finite-difference discretization.

    The returned mapping contains ``mpmath.mpf`` values for both final rows
    and their derivative/source/numerator terms.
    """
    try:
        import mpmath as mp
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("mpmath is required for the 80-digit wall audit") from error
    width = int(stencil_width)
    precision = int(dps)
    if width != stencil_width or width != STENCIL_WIDTH:
        raise ValueError("the mpmath wall control requires stencil_width=7")
    if precision != dps or precision < 80:
        raise ValueError("the mpmath wall control requires at least 80 digits")
    z64 = np.asarray(z, dtype=np.float64)
    q = np.asarray(q)
    phi = np.asarray(phi)
    a = np.asarray(a)
    c = np.asarray(c)
    shape = (len(z64), q.shape[1] if q.ndim == 2 else -1)
    if q.ndim != 2 or any(field.shape != q.shape for field in (phi, a, c)):
        raise ValueError("q, phi, a, and c must share a two-dimensional shape")
    if q.shape[0] != len(z64) or len(z64) < width:
        raise ValueError("field shape does not match the compact grid")
    if np.any(np.diff(z64) <= 0.0):
        raise ValueError("z must be strictly increasing")
    if not all(np.all(np.isfinite(field)) for field in (q, phi, a, c)):
        raise ValueError("wall fields must be finite")
    radial = int(radial_index)
    if radial != radial_index or radial < 0 or radial >= shape[1]:
        raise IndexError("radial_index is outside the wall row")
    if wall not in ("lower", "upper"):
        raise ValueError("wall must be 'lower' or 'upper'")
    upper = wall == "upper"
    wall_index = len(z64) - 1 if upper else 0
    stencil = _fixed_float64_stencil(z64, 1, width)
    indices = stencil.indices[wall_index]
    weights = np.asarray(stencil.weights[wall_index], dtype=np.float64)
    required = (
        "wall_stiffness", "v0", "v1", "beta_a", "beta_b",
        "wall_potential_a", "wall_potential_b",
    )
    missing = [name for name in required if name not in background]
    if missing:
        raise KeyError(f"background lacks required wall coefficients: {missing}")

    with mp.workdps(precision):
        gamma = _mp_exact(background["wall_stiffness"], mp)
        target = _mp_exact(background["v1" if upper else "v0"], mp)
        bare_beta = _mp_exact(background["beta_b" if upper else "beta_a"], mp)
        reference_potential = _mp_exact(
            background["wall_potential_b" if upper else "wall_potential_a"], mp,
        )
        orientation = mp.mpf(1 if upper else -1)
        local_phi = _mp_exact(phi[wall_index, radial], mp)
        delta = local_phi - target
        potential = mp.mpf("0.5") * gamma * delta**2
        if upper:
            beta = bare_beta - (potential - reference_potential) / 6
        else:
            beta = bare_beta + (potential - reference_potential) / 6

        metric_values = []
        phi_values = []
        for index in indices:
            compact_z = _mp_exact(z64[index], mp)
            local_q = _mp_exact(q[index, radial], mp)
            local_c = _mp_exact(c[index, radial], mp)
            psi = 1 / (compact_z + local_q)
            metric_values.append(psi**2 * mp.exp(2 * local_c))
            phi_values.append(_mp_exact(phi[index, radial], mp))
        exact_weights = [_mp_exact(value, mp) for value in weights]
        metric_derivative = mp.fsum(
            weight * value for weight, value in zip(exact_weights, metric_values)
        )
        phi_derivative = mp.fsum(
            weight * value for weight, value in zip(exact_weights, phi_values)
        )
        compact_z = _mp_exact(z64[wall_index], mp)
        local_q = _mp_exact(q[wall_index, radial], mp)
        local_a = _mp_exact(a[wall_index, radial], mp)
        local_c = _mp_exact(c[wall_index, radial], mp)
        local_psi = 1 / (compact_z + local_q)
        normal_scale = local_psi * mp.exp(local_a)
        sphere_metric = local_psi**2 * mp.exp(2 * local_c)
        metric_source = 2 * beta * normal_scale * sphere_metric
        metric_numerator = metric_derivative + metric_source
        metric_row = orientation * metric_numerator / (2 * normal_scale)
        phi_source = orientation * mp.mpf("0.5") * gamma * delta * normal_scale
        phi_row = phi_derivative + phi_source
        return {
            "dps": precision,
            "wall": wall,
            "wall_index": wall_index,
            "radial_index": radial,
            "metric_derivative": +metric_derivative,
            "metric_source": +metric_source,
            "metric_numerator": +metric_numerator,
            "metric_row": +metric_row,
            "phi_derivative": +phi_derivative,
            "phi_source": +phi_source,
            "phi_row": +phi_row,
        }


# Concise alias retained for diagnostic runners.
mpmath_wall_row = reevaluate_wall_row_mpmath


def _mpmath_dot80(left, right):
    """Return an exact-binary-input 80-decimal dot product as text."""
    try:
        import mpmath as mp
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("mpmath is required for the 80-digit dual audit") from error
    left = np.asarray(left, dtype=np.float64).ravel(order="C")
    right = np.asarray(right, dtype=np.float64).ravel(order="C")
    if left.shape != right.shape:
        raise ValueError("80-digit dot inputs must have the same shape")
    with mp.workdps(80):
        value = mp.fsum(
            _mp_exact(first, mp)*_mp_exact(second, mp)
            for first, second in zip(left, right)
        )
        return mp.nstr(value, 82)


def extended_precision_residual(
    parent, residual, localization, cancellation, linear,
):
    """Run the fixed full-residual and load-bearing 80-digit controls.

    Returns ``(summary, arrays)``.  The current frozen arm64 runtime may alias
    long double to binary64; in that case a load-bearing bulk row cannot be
    certified because the independent 80-digit implementation covers the
    compact-wall functional only.  The summary then sets ``complete=False``
    so Protocol 131 must classify the result as inconclusive rather than infer
    an arithmetic or compatibility mechanism.
    """
    required = (
        "q", "phi", "z", "r", "a", "b", "c", "background",
        "chi_r", "chi_z", "reference_q", "reference_phi",
    )
    if not isinstance(parent, Mapping) or any(name not in parent for name in required):
        raise ValueError("precision parent record is incomplete")
    binary = np.asarray(residual, dtype=np.float64)
    extended = joint_parent_residual_longdouble(
        parent["q"], parent["phi"], parent["z"], parent["r"],
        parent["a"], parent["b"], parent["c"], parent["background"],
        parent["chi_r"], parent["chi_z"], parent["reference_q"],
        parent["reference_phi"], STENCIL_WIDTH,
    )
    if extended.shape != binary.shape or not np.all(np.isfinite(extended)):
        raise RuntimeError("extended residual is incomplete or nonfinite")
    difference = np.abs(extended-np.asarray(binary, dtype=np.longdouble))
    capability = longdouble_capability()
    n = parent["q"].size
    nz, nr = parent["q"].shape
    failing = np.flatnonzero(np.abs(binary) >= np.float64(0.5e-10))
    maximum_index = int(np.argmax(np.abs(binary)))
    load_bearing = sorted(set(failing.tolist() + [maximum_index]))
    mp_records = []
    mp_values = {}
    unsupported = []
    try:
        import mpmath as mp
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("mpmath is required for Protocol-131 precision") from error
    for global_index in load_bearing:
        block, local = divmod(int(global_index), n)
        i, j = divmod(local, nr)
        if i not in (0, nz-1):
            unsupported.append(global_index)
            continue
        wall = "lower" if i == 0 else "upper"
        replay = reevaluate_wall_row_mpmath(
            parent["q"], parent["phi"], parent["z"], parent["a"],
            parent["c"], parent["background"], wall, j,
        )
        value = replay["metric_row" if block == 0 else "phi_row"]
        with mp.workdps(80):
            value_text = mp.nstr(value, 82)
            exact_difference = abs(value-_mp_exact(binary[global_index], mp))
            below_target = bool(
                abs(value) < _mp_exact(np.float64(1.0e-10), mp)
            )
        value_float = float(value)
        mp_values[global_index] = value_float
        mp_records.append({
            "global_index": global_index,
            "block": "metric" if block == 0 else "Phi",
            "i": i, "j": j, "wall": wall,
            "value_80digit": value_text,
            "value_float64_projection": value_float,
            "absolute_difference_from_binary64": float(exact_difference),
            "absolute_below_exact_binary64_target": below_target,
        })
    reconstructed = np.abs(binary).copy()
    for index, value in mp_values.items():
        reconstructed[index] = abs(value)
    mp_certified = bool(not unsupported and len(mp_values) == len(load_bearing))
    # JSON artifacts use strict finite numbers.  Absence of an 80-digit
    # certificate is represented semantically, not by a non-standard NaN.
    mp_maximum = float(np.max(reconstructed)) if mp_certified else None
    # eta_F is a classification-bearing uncertainty, so it may include only
    # rows certified by genuinely wider arithmetic.  On arm64, longdouble is
    # binary64 and its full-grid reorder difference remains diagnostic only;
    # unrelated uncertified bulk rows must not contaminate a wall-row gate.
    if capability["wider_than_float64"]:
        eta_f = float(np.max(difference))
        eta_scope = "complete-longdouble-residual"
    else:
        eta_f = max(
            (
                item["absolute_difference_from_binary64"]
                for item in mp_records
            ),
            default=0.0,
        )
        eta_scope = "80-digit-certified-load-bearing-wall-rows"

    mode_duals = []
    dual_certified = True
    left_vectors = np.asarray(linear.get("mode_left_vectors", np.empty((len(binary), 0))))
    for index, mode in enumerate(linear.get("modes", ())):
        if not mode.get("definitely_null"):
            continue
        if index >= left_vectors.shape[1]:
            dual_certified = False
            break
        left = left_vectors[:, index]
        compensated = compensated_dot(left, binary)
        exact_text = _mpmath_dot80(left, binary)
        mode_duals.append({
            "mode_index": index,
            "compensated_longdouble": str(compensated),
            "mpmath_80digit": exact_text,
        })
    complete = bool(capability["wider_than_float64"] or mp_certified)
    mp_load_bearing_below = bool(
        mp_certified
        and all(
            item["absolute_below_exact_binary64_target"]
            for item in mp_records
        )
    )
    arithmetic_below = bool(
        (
            capability["wider_than_float64"]
            and np.max(np.abs(extended)) < _LD(np.float64(1.0e-10))
        )
        or mp_load_bearing_below
    )
    summary = {
        "complete": complete,
        "capability": {
            "dtype": capability["dtype"],
            "itemsize": capability["itemsize"],
            "epsilon": str(capability["epsilon"]),
            "mantissa_bits": capability["mantissa_bits"],
            "wider_than_float64": capability["wider_than_float64"],
        },
        "longdouble_maximum": float(np.max(np.abs(extended))),
        "longdouble_rms": float(np.sqrt(np.mean(extended**2))),
        "maximum_absolute_longdouble_difference": float(np.max(difference)),
        "eta_F": eta_f,
        "eta_F_scope": eta_scope,
        "load_bearing_row_count": len(load_bearing),
        "unsupported_bulk_load_bearing_rows": unsupported,
        "mp_certified": mp_certified,
        "mp_reconstructed_maximum": mp_maximum,
        "mp_load_bearing_below_exact_target": mp_load_bearing_below,
        "mp_wall_rows": mp_records,
        "dual_certified": bool(dual_certified),
        "mode_duals": mode_duals,
        "arithmetic_max_below_target": arithmetic_below,
        "scope_note": (
            "full compensated long-double replay; 80-digit certification is "
            "available only for normalized compact-wall rows and null-mode dots"
        ),
    }
    arrays = {
        "longdouble_residual": np.asarray(extended),
        "longdouble_minus_binary64": np.asarray(
            extended-np.asarray(binary, dtype=np.longdouble)
        ),
    }
    return summary, arrays


__all__ = [
    "STENCIL_WIDTH",
    "compensated_dot",
    "joint_parent_residual_longdouble",
    "longdouble_capability",
    "longdouble_joint_parent_residual",
    "mpmath_wall_row",
    "reevaluate_wall_row_mpmath",
    "extended_precision_residual",
]
