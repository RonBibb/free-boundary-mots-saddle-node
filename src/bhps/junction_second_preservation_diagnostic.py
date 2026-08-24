"""Second fixed-grid preservation tangent for compact-wall Israel rows.

For each independent SO(3) tangential metric component, the native wall row
is

``J[q] = s (D_z f + 2 beta(Phi) sqrt(g_zz) f) / (2 sqrt(g_zz))``.

This module evaluates its *coordinate-time, fixed-grid, semi-discrete* second
derivative along a state trajectory with ``q_t=v`` and ``q_tt=a``:

``D_X^2 J = D J[q].a + D^2 J[q](v,v)``.

The production seven-point ``D_z`` matrix is retained.  This is not a
covariant derivative along a horizon generator, a moving-surface material
derivative, or a Lie derivative; it contains no connection, moving-basis, or
surface-advection terms.  It also does not solve or alter any boundary row.
It only audits supplied position, velocity, and acceleration snapshots.
"""

from __future__ import annotations

import numpy as np

from bhps.junction_preservation_diagnostic import (
    COMPONENTS,
    WALLS,
    _assemble_so3_tensor,
    _component_fields,
    _dense_derivative,
    _maximum_record,
    _orthonormal_frames,
    _proper_profile_statistics,
    _validate_state,
    radial_zones,
    wall_source_coefficients,
)


def _validate_acceleration(acceleration, shape):
    value = np.asarray(acceleration, dtype=float)
    if value.shape != shape:
        raise ValueError("acceleration must share the position shape")
    if not np.all(np.isfinite(value)):
        raise ValueError("acceleration must be finite")
    return value


def wall_junction_second_tangent(
    position, velocity, acceleration, z, r, background, wall,
    stencil_width=7,
):
    """Return raw ``J``, ``D_XJ``, and ``D_X^2J`` wall components.

    ``D_X^2J`` is decomposed exactly into ``DJ[q].acceleration`` and
    ``D2J[q](velocity, velocity)``.  All quantities use the same fixed-grid
    semi-discrete wall functional and the same orientation as the production
    compact-wall Robin rows.
    """
    q, v, z, r = _validate_state(position, velocity, z, r)
    a = _validate_acceleration(acceleration, q.shape)
    if wall not in WALLS:
        raise ValueError("wall must be 'lower' or 'upper'")

    index = -1 if wall == "upper" else 0
    dz = _dense_derivative(z, 1, stencil_width)
    q_fields = _component_fields(q, r)
    v_fields = _component_fields(v, r)
    a_fields = _component_fields(a, r)

    source = wall_source_coefficients(q[index, :, 7], background, wall)
    orientation = source["orientation"]
    beta = source["beta"]
    beta_phi = source["beta_phi"]
    gamma = float(background["wall_stiffness"])
    beta_phiphi = (-gamma / 6.0) if wall == "upper" else (gamma / 6.0)
    phi_t = v[index, :, 7]
    phi_tt = a[index, :, 7]
    beta_t = beta_phi * phi_t
    beta_tt = beta_phi * phi_tt + beta_phiphi * phi_t**2

    A = np.sqrt(q[index, :, 6])
    A_t = v[index, :, 6] / (2.0 * A)
    A_tt = (
        a[index, :, 6] / (2.0 * A)
        - v[index, :, 6] ** 2 / (4.0 * A**3)
    )
    inverse_scale_rate = A_t / A
    inverse_scale_second = 2.0 * inverse_scale_rate**2 - A_tt / A

    components = {}
    second_form_defect = 0.0
    for name in COMPONENTS:
        field = q_fields[name]
        field_t = v_fields[name]
        field_tt = a_fields[name]
        value = field[index]
        value_t = field_t[index]
        value_tt = field_tt[index]
        value_z = (dz @ field)[index]
        value_zt = (dz @ field_t)[index]
        value_ztt = (dz @ field_tt)[index]

        robin = value_z + 2.0 * beta * A * value
        robin_t = value_zt + 2.0 * (
            beta_t * A * value
            + beta * A_t * value
            + beta * A * value_t
        )
        robin_tt = value_ztt + 2.0 * (
            beta_tt * A * value
            + beta * A_tt * value
            + beta * A * value_tt
            + 2.0 * beta_t * A_t * value
            + 2.0 * beta_t * A * value_t
            + 2.0 * beta * A_t * value_t
        )

        junction = orientation * robin / (2.0 * A)
        first_tangent = orientation / (2.0 * A) * (
            robin_t - inverse_scale_rate * robin
        )
        raw_second_tangent = orientation / (2.0 * A) * (
            robin_tt
            - 2.0 * inverse_scale_rate * robin_t
            + inverse_scale_second * robin
        )

        # Cancellation-exposed decomposition of
        #   J = s [ (D_z f)/(2 A) + beta f ].
        # It avoids recovering the Hessian by subtracting two potentially
        # acceleration-sized values.  Because f, Phi, and g_zz are linear
        # state coordinates, D2J[v,v] contains only the terms below.
        normal_acceleration = a[index, :, 6]
        normal_velocity = v[index, :, 6]
        dj_acceleration = orientation * (
            value_ztt / (2.0 * A)
            - value_z * normal_acceleration / (4.0 * A**3)
            + beta_phi * phi_tt * value
            + beta * value_tt
        )
        hessian_velocity = orientation * (
            - normal_velocity * value_zt / (2.0 * A**3)
            + 3.0 * normal_velocity**2 * value_z / (8.0 * A**5)
            + beta_phiphi * phi_t**2 * value
            + 2.0 * beta_phi * phi_t * value_t
        )
        second_tangent = dj_acceleration + hessian_velocity
        local_second_form_defect = float(np.max(np.abs(
            second_tangent - raw_second_tangent
        )))
        second_form_defect = max(second_form_defect, local_second_form_defect)

        components[name] = {
            "metric": value,
            "metric_t": value_t,
            "metric_tt": value_tt,
            "normal_derivative": value_z,
            "normal_derivative_t": value_zt,
            "normal_derivative_tt": value_ztt,
            "robin_residual": robin,
            "DX_robin_residual": robin_t,
            "DX2_robin_residual": robin_tt,
            "J": junction,
            "DXJ": first_tangent,
            "DJ_acceleration": dj_acceleration,
            "D2J_velocity_velocity": hessian_velocity,
            "DX2J": second_tangent,
            "DX2J_raw_robin_form": raw_second_tangent,
            "second_form_maximum_absolute_defect": local_second_form_defect,
        }

    tensor_keys = (
        "J", "DXJ", "DJ_acceleration", "D2J_velocity_velocity", "DX2J",
    )
    tensors = {
        key: _assemble_so3_tensor({
            name: components[name][key] for name in COMPONENTS
        }) for key in tensor_keys
    }
    decomposition_defect = (
        tensors["DX2J"]
        - tensors["DJ_acceleration"]
        - tensors["D2J_velocity_velocity"]
    )
    metric_tensor = _assemble_so3_tensor({
        name: components[name]["metric"] for name in COMPONENTS
    })

    # The stabilizer Robin row and reflecting-scalar Neumann row are separate
    # compatibility lanes.  They are not components of the Israel tensor.
    sign = orientation
    delta = q[index, :, 7] - source["target"]
    phi = q[:, :, 7]
    phi_velocity = v[:, :, 7]
    phi_acceleration = a[:, :, 7]
    chi = q[:, :, 8]
    chi_velocity = v[:, :, 8]
    chi_acceleration = a[:, :, 8]
    A_acceleration_direction = a[index, :, 6] / (2.0 * A)
    A_velocity_hessian = -v[index, :, 6] ** 2 / (4.0 * A**3)
    scalar_factor = sign * 0.5 * gamma
    phi_robin = (dz @ phi)[index] + scalar_factor * delta * A
    phi_robin_t = (dz @ phi_velocity)[index] + scalar_factor * (
        phi_t * A + delta * A_t
    )
    phi_robin_dj_acceleration = (
        (dz @ phi_acceleration)[index]
        + scalar_factor * (
            phi_tt * A + delta * A_acceleration_direction
        )
    )
    phi_robin_hessian_velocity = scalar_factor * (
        2.0 * phi_t * A_t + delta * A_velocity_hessian
    )
    phi_robin_tt = (
        phi_robin_dj_acceleration + phi_robin_hessian_velocity
    )
    chi_robin = (dz @ chi)[index]
    chi_robin_t = (dz @ chi_velocity)[index]
    chi_robin_tt = (dz @ chi_acceleration)[index]

    return {
        "wall": wall,
        "orientation": orientation,
        "scope": (
            "fixed-grid coordinate-time second derivative of the semi-discrete "
            "Israel row; noncovariant and without moving-surface terms"
        ),
        "components": components,
        "metric_tensor": metric_tensor,
        "J_tensor": tensors["J"],
        "DXJ_tensor": tensors["DXJ"],
        "DJ_acceleration_tensor": tensors["DJ_acceleration"],
        "D2J_velocity_velocity_tensor": tensors["D2J_velocity_velocity"],
        "DX2J_tensor": tensors["DX2J"],
        "decomposition_maximum_absolute_defect": float(
            np.max(np.abs(decomposition_defect))
        ),
        "raw_vs_cancellation_exposed_maximum_absolute_defect": (
            second_form_defect
        ),
        "separate_rows": {
            "Phi_robin": phi_robin,
            "DX_Phi_robin": phi_robin_t,
            "DJ_Phi_robin_acceleration": phi_robin_dj_acceleration,
            "D2_Phi_robin_velocity_velocity": phi_robin_hessian_velocity,
            "DX2_Phi_robin": phi_robin_tt,
            "chi_neumann": chi_robin,
            "DX_chi_neumann": chi_robin_t,
            "DJ_chi_neumann_acceleration": chi_robin_tt,
            "D2_chi_neumann_velocity_velocity": np.zeros_like(chi_robin_tt),
            "DX2_chi_neumann": chi_robin_tt,
        },
        "source": {
            "beta": beta,
            "beta_phi": beta_phi,
            "beta_phiphi": beta_phiphi,
            "beta_t": beta_t,
            "beta_tt": beta_tt,
            "sqrt_gzz": A,
            "sqrt_gzz_t": A_t,
            "sqrt_gzz_tt": A_tt,
        },
        "finite": bool(all(
            np.all(np.isfinite(tensor)) for tensor in tensors.values()
        ) and all(np.all(np.isfinite(value)) for value in (
            phi_robin, phi_robin_t, phi_robin_tt,
            chi_robin, chi_robin_t, chi_robin_tt,
        ))),
    }


def summarize_wall_second_tangent(record, r, buffer_points=7):
    """Return orthonormal ``D_X^2J`` statistics in three radial zones.

    The frame and proper spatial measure are constructed from the supplied
    position wall metric.  This does not convert the coordinate-time tangent
    into a covariant or moving-surface derivative.
    """
    r = np.asarray(r, dtype=float)
    metric = np.asarray(record["metric_tensor"], dtype=float)
    tangent = np.asarray(record["DX2J_tensor"], dtype=float)
    if metric.shape != (len(r), 4, 4) or tangent.shape != metric.shape:
        raise ValueError("wall record and radial grid are not aligned")
    frames, frame_defect = _orthonormal_frames(metric)
    tangent_hat = np.einsum(
        "nai,nab,nbj->nij", frames, tangent, frames,
    )
    frobenius = np.linalg.norm(tangent_hat, axis=(1, 2))
    maximum_component = np.max(np.abs(tangent_hat), axis=(1, 2))
    zones = {}
    for zone, indices in radial_zones(r, buffer_points).items():
        zones[zone] = {
            "DX2J_orthonormal_frobenius": _maximum_record(
                frobenius, indices, r,
            ),
            "DX2J_orthonormal_max_component": _maximum_record(
                maximum_component, indices, r,
            ),
            "proper_statistics": _proper_profile_statistics(
                frobenius, indices, r, metric,
            ),
        }
    return {
        "wall": record["wall"],
        "scope": (
            "position-metric orthonormal summary of a fixed-grid coordinate-time "
            "second tangent; noncovariant and without moving-surface terms"
        ),
        "zones": zones,
        "frame_defect_maximum": float(np.max(frame_defect)),
        "finite": bool(
            np.all(np.isfinite(frobenius))
            and np.all(np.isfinite(maximum_component))
            and np.all(np.isfinite(frame_defect))
        ),
    }
