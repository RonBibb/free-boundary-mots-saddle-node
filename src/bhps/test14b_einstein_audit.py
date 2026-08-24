"""Independent full-Einstein-tensor contractions for Test 14B."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, RectBivariateSpline

from bhps.linearized_gh_einstein_scalar import metric_geometry_from_jets
from bhps.nonlinear_regular_so3_evolution import reduced_state_jets
from bhps.regular_so3_gh_reduction import regular_so3_perturbation_jets


SQRT2 = np.sqrt(2.0)


def _interpolate(z, r, field, zcoord, radius):
    field = np.asarray(field, dtype=float)
    result = np.empty((len(zcoord), *field.shape[2:]))
    for index in np.ndindex(field.shape[2:]):
        result[(slice(None), *index)] = RectBivariateSpline(
            z, r, field[(slice(None), slice(None), *index)],
            kx=min(3, len(z) - 1), ky=min(3, len(r) - 1), s=0,
        ).ev(zcoord, radius)
    return result


def einstein_null_contractions(
    position, velocity, acceleration, z, r, profile, mass_squared,
    sample_nodes=129, stencil_width=7,
):
    """Return smooth ``G_ll`` and ``G_ln`` on the profile's full mesh.

    The full Ricci tensor is reconstructed from coordinate jets, including a
    finite-difference time derivative of the archived velocity.  This path is
    independent of the canonical scalar-stress evaluation used by the primary
    Test-14B balance.
    """
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    acceleration = np.asarray(acceleration, dtype=float)
    z = np.asarray(z, dtype=float)
    r = np.asarray(r, dtype=float)
    theta = np.asarray(profile["theta"], dtype=float)
    rho = np.asarray(profile["rho"], dtype=float)
    slope = np.asarray(profile["slope"], dtype=float)
    count = min(int(sample_nodes), len(theta))
    indices = np.unique(np.rint(np.linspace(0, len(theta) - 1, count)).astype(int))
    sampled_theta = theta[indices]
    sampled_rho = rho[indices]
    sampled_slope = slope[indices]
    sine = np.sin(sampled_theta)
    cosine = np.cos(sampled_theta)
    zcoord = z[-1] - sampled_rho * cosine
    radius = sampled_rho * sine

    first, second = reduced_state_jets(
        position, velocity, z, r, stencil_width=stencil_width,
    )
    second[0, 0] = acceleration
    values = _interpolate(z, r, position, zcoord, radius)
    sampled_first = _interpolate(
        z, r, np.moveaxis(first, 0, 2), zcoord, radius,
    )
    sampled_second = _interpolate(
        z, r, np.moveaxis(second, (0, 1), (2, 3)), zcoord, radius,
    )

    tangent_coordinate = np.stack((
        sampled_rho * sine - sampled_slope * cosine,
        sampled_rho * cosine + sampled_slope * sine,
    ), axis=1)

    g_ll = np.empty(len(indices))
    g_ln = np.empty(len(indices))
    scalar_ll = np.empty(len(indices))
    scalar_ln_total = np.empty(len(indices))
    ricci_scalar = np.empty(len(indices))
    null_l_norm = np.empty(len(indices))
    null_n_norm = np.empty(len(indices))
    null_cross = np.empty(len(indices))
    for local in range(len(indices)):
        jets = regular_so3_perturbation_jets(
            radius[local], values[local], sampled_first[local],
            sampled_second[local],
        )
        geometry = metric_geometry_from_jets(
            jets["metric"], jets["metric_first"], jets["metric_second"],
        )
        inverse_metric = geometry["inverse_metric"]
        scalar = float(np.einsum("ab,ab->", inverse_metric, geometry["ricci"]))
        einstein = geometry["ricci"] - 0.5 * scalar * jets["metric"]
        base_metric = jets["metric"][1:3, 1:3]
        base_inverse = np.linalg.inv(base_metric)
        beta_covector = jets["metric"][0, 1:3]
        shift = base_inverse @ beta_covector
        lapse = np.sqrt(-jets["metric"][0, 0] + beta_covector @ shift)
        local_normal_covector = np.array((
            -tangent_coordinate[local, 1], tangent_coordinate[local, 0],
        ))
        local_normal_covector /= np.sqrt(
            local_normal_covector @ base_inverse @ local_normal_covector
        )
        local_normal = base_inverse @ local_normal_covector
        u = np.array((
            1.0 / lapse, -shift[0] / lapse,
            -shift[1] / lapse, 0.0, 0.0,
        ))
        spatial_normal = np.array((
            0.0, local_normal[0], local_normal[1], 0.0, 0.0,
        ))
        outgoing = (u + spatial_normal) / SQRT2
        ingoing = (u - spatial_normal) / SQRT2
        null_l_norm[local] = np.einsum(
            "a,ab,b->", outgoing, jets["metric"], outgoing,
        )
        null_n_norm[local] = np.einsum(
            "a,ab,b->", ingoing, jets["metric"], ingoing,
        )
        null_cross[local] = np.einsum(
            "a,ab,b->", outgoing, jets["metric"], ingoing,
        )
        g_ll[local] = np.einsum("a,ab,b->", outgoing, einstein, outgoing)
        g_ln[local] = np.einsum("a,ab,b->", outgoing, einstein, ingoing)

        phi_gradient = jets["phi_first"]
        chi_gradient = jets["chi_first"]
        potential = -6.0 + 0.5 * float(mass_squared) * jets["phi"] ** 2
        gradient_norm = (
            np.einsum("a,ab,b->", phi_gradient, inverse_metric, phi_gradient)
            + np.einsum("a,ab,b->", chi_gradient, inverse_metric, chi_gradient)
        )
        stress = (
            np.outer(phi_gradient, phi_gradient)
            + np.outer(chi_gradient, chi_gradient)
            - jets["metric"] * (0.5 * gradient_norm + potential)
        )
        scalar_ll[local] = np.einsum(
            "a,ab,b->", outgoing, stress, outgoing,
        )
        scalar_ln_total[local] = np.einsum(
            "a,ab,b->", outgoing, stress, ingoing,
        )
        ricci_scalar[local] = scalar

    def full(values_sampled):
        return CubicSpline(sampled_theta, values_sampled)(theta)

    full_g_ll = full(g_ll)
    full_g_ln = full(g_ln)
    full_scalar_ll = full(scalar_ll)
    full_scalar_ln = full(scalar_ln_total)
    scale_ll = max(
        float(np.linalg.norm(full_g_ll)), float(np.linalg.norm(full_scalar_ll)),
        1e-300,
    )
    scale_ln = max(
        float(np.linalg.norm(full_g_ln)), float(np.linalg.norm(full_scalar_ln)),
        1e-300,
    )
    return {
        "T_ll": full_g_ll,
        "T_ln": full_g_ln,
        "sample_nodes": len(indices),
        "einstein_to_scalar_Tll_relative_L2": float(
            np.linalg.norm(full_g_ll - full_scalar_ll) / scale_ll
        ),
        "einstein_to_scalar_total_Tln_relative_L2": float(
            np.linalg.norm(full_g_ln - full_scalar_ln) / scale_ln
        ),
        "maximum_absolute_ricci_scalar": float(np.max(np.abs(ricci_scalar))),
        "maximum_absolute_l_squared": float(np.max(np.abs(null_l_norm))),
        "maximum_absolute_n_squared": float(np.max(np.abs(null_n_norm))),
        "maximum_absolute_l_dot_n_plus_one": float(np.max(np.abs(
            null_cross + 1.0
        ))),
        "finite": bool(
            np.all(np.isfinite(full_g_ll)) and np.all(np.isfinite(full_g_ln))
        ),
    }
