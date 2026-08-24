"""Initial generalized-harmonic source jets for time-symmetric data.

The sign convention used here records the lower contracted Christoffel symbol
``Gamma_mu = g^{ab} Gamma_{mu ab}``.  A generalized-harmonic implementation
may define its constraint as either ``H_mu-Gamma_mu`` or ``H_mu+Gamma_mu``;
only the corresponding sign of the reported source has to be changed.
"""

from __future__ import annotations

import numpy as np


def spatial_metric_acceleration_trace(acceleration,psi,a,b,c):
    """Return ``gamma^{ij} gamma_ij,tt`` for the diagonal spatial metric."""
    psi=np.asarray(psi,dtype=float);a=np.asarray(a,dtype=float)
    b=np.asarray(b,dtype=float);c=np.asarray(c,dtype=float)
    return (
        np.asarray(acceleration["zz"])/(psi**2*np.exp(2*a))
        +np.asarray(acceleration["radial"])/(psi**2*np.exp(2*b))
        +2*np.asarray(acceleration["transverse"])/(psi**2*np.exp(2*c))
    )


def initial_contracted_christoffel_time_jet(
    acceleration,alpha,lapse_acceleration,psi,a,b,c,
):
    """Return ``partial_t Gamma_0`` at zero shift and vanishing first jets.

    For ``g_00=-alpha^2`` and ``partial_t g_{mu nu}=0``, direct contraction
    gives

    ``partial_t Gamma_0 = alpha_tt/alpha
                              - gamma^{ij} gamma_ij,tt/2``.
    """
    alpha=np.asarray(alpha,dtype=float)
    lapse_acceleration=np.asarray(lapse_acceleration,dtype=float)
    if alpha.shape!=lapse_acceleration.shape or np.any(alpha<=0):
        raise ValueError("alpha and its acceleration must share a positive grid")
    trace=spatial_metric_acceleration_trace(acceleration,psi,a,b,c)
    relative=lapse_acceleration/alpha
    return {
        "gamma_0_time_derivative":relative-.5*trace,
        "relative_lapse_acceleration":relative,
        "spatial_metric_acceleration_trace":trace,
    }


def initial_normal_contracted_christoffel(
    z,r,alpha,psi,a,b,c,stencil_width=7,
):
    """Return the lower normal-coordinate ``Gamma_z`` on the initial slice."""
    from bhps.adm_corner import _axisymmetric_derivatives

    alpha=np.asarray(alpha,dtype=float);psi=np.asarray(psi,dtype=float)
    potential=(
        np.log(psi)+np.asarray(a)-np.log(psi)-np.asarray(b)
        -2*(np.log(psi)+np.asarray(c))-np.log(alpha)
    )
    return _axisymmetric_derivatives(potential,z,r,stencil_width)["z"]


def diagonal_spatial_source_second_jets(
    z,r,acceleration,alpha,lapse_acceleration,psi,a,b,c,stencil_width=7,
):
    """Return ``Gamma_z,tt`` and ``Gamma_r,tt`` for zero-shift diagonal data.

    The result follows by differentiating the exact diagonal identities

    ``Gamma_z=d_z(log A-log B-2 log C-log alpha)`` and
    ``Gamma_r=d_r(log B-log A-2 log C-log alpha)-2/r``

    twice in time at a time-symmetric slice.  The static ``-2/r`` term drops
    out, avoiding an axis singularity in the second jet.
    """
    from bhps.adm_corner import _axisymmetric_derivatives

    trace_parts={
        "A":.5*np.asarray(acceleration["zz"])/(np.asarray(psi)*np.exp(a))**2,
        "B":.5*np.asarray(acceleration["radial"])/(np.asarray(psi)*np.exp(b))**2,
        "C":.5*np.asarray(acceleration["transverse"])/(np.asarray(psi)*np.exp(c))**2,
    }
    relative=np.asarray(lapse_acceleration)/np.asarray(alpha)
    z_potential=trace_parts["A"]-trace_parts["B"]-2*trace_parts["C"]-relative
    r_potential=trace_parts["B"]-trace_parts["A"]-2*trace_parts["C"]-relative
    dz=_axisymmetric_derivatives(z_potential,z,r,stencil_width)
    dr=_axisymmetric_derivatives(r_potential,z,r,stencil_width)
    return {
        "gamma_z_second_time_derivative":dz["z"],
        "gamma_r_second_time_derivative":dr["r"],
        "z_source_potential":z_potential,
        "r_source_potential":r_potential,
    }
