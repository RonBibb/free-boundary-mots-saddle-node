"""Gradient-topology diagnostics for localized stabilizer profiles."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from bhps.gw_slice_high_order_solver import derivative_matrix


def _spline_roots(spline,lower,upper,samples):
    points=np.linspace(lower,upper,int(samples));values=spline(points)
    roots=[]
    for left,right,vleft,vright in zip(points[:-1],points[1:],values[:-1],values[1:]):
        if vleft==0:roots.append(float(left))
        elif vleft*vright<0:roots.append(float(brentq(spline,left,right,xtol=1e-13)))
    if values[-1]==0:roots.append(float(points[-1]))
    unique=[]
    for root in roots:
        if not unique or abs(root-unique[-1])>1e-9*(upper-lower):unique.append(root)
    return unique


def localized_stabilizer_gradient_diagnostics(z,r,phi,psi,a,b):
    """Locate compact-direction turns and axis stationary points.

    ``a`` and ``b`` are the compact and radial logarithmic anisotropies, so
    the spatial metric factors are ``A=psi exp(a)`` and ``B=psi exp(b)``.
    Axis regularity sets ``Phi_r=0`` exactly at ``r=0``.
    """
    z=np.asarray(z,dtype=float);r=np.asarray(r,dtype=float)
    phi=np.asarray(phi,dtype=float);psi=np.asarray(psi,dtype=float)
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    expected=(len(z),len(r))
    if (
        z.ndim!=1 or r.ndim!=1 or len(z)<7 or len(r)<7
        or np.any(np.diff(z)<=0) or np.any(np.diff(r)<=0) or r[0]!=0
        or any(field.shape!=expected for field in (phi,psi,a,b))
        or np.any(psi<=0)
    ):
        raise ValueError("invalid localized stabilizer inputs")
    phi_z=derivative_matrix(z,1)@phi
    phi_r=phi@derivative_matrix(r,1).T
    phi_r[:,0]=0.
    compact_factor=psi*np.exp(a);radial_factor=psi*np.exp(b)
    invariant_gradient_squared=(phi_z/compact_factor)**2+(phi_r/radial_factor)**2

    turn_locations=np.full(len(r),np.nan)
    transverse_at_turn=np.full(len(r),np.nan)
    turn_counts=np.zeros(len(r),dtype=int)
    for j in range(len(r)):
        profile=CubicSpline(z,phi[:,j])
        roots=_spline_roots(profile.derivative(),z[0],z[-1],8*(len(z)-1)+1)
        turn_counts[j]=len(roots)
        if roots:
            turn_locations[j]=roots[0]
            transverse_at_turn[j]=float(CubicSpline(z,phi_r[:,j])(roots[0]))

    axis_profile=CubicSpline(z,phi[:,0]);axis_derivative=axis_profile.derivative()
    axis_roots=_spline_roots(axis_derivative,z[0],z[-1],8*(len(z)-1)+1)
    stationary=[]
    # Even-axis fit phi(r)=c0+c1 r^2+... gives Phi_rr(0)=2c1.
    radial_second=np.array([
        2*np.polynomial.polynomial.polyfit(r[:6]**2,phi[i,:6],3)[1]
        for i in range(len(z))
    ])
    radial_second_spline=CubicSpline(z,radial_second)
    for root in axis_roots:
        compact_second=float(axis_profile.derivative(2)(root))
        radial_second_value=float(radial_second_spline(root))
        if compact_second>0 and radial_second_value>0:kind="minimum"
        elif compact_second<0 and radial_second_value<0:kind="maximum"
        else:kind="saddle"
        stationary.append({
            "z":float(root),"r":0.,"phi":float(axis_profile(root)),
            "phi_zz":compact_second,"phi_rr":radial_second_value,
            "classification":kind,
        })

    finite_turns=np.isfinite(turn_locations)
    return {
        "phi_z":phi_z,"phi_r":phi_r,
        "invariant_gradient_squared":invariant_gradient_squared,
        "turn_locations":turn_locations,
        "transverse_gradient_at_turn":transverse_at_turn,
        "turn_count_per_ray":turn_counts,
        "rays_with_compact_turn":int(np.count_nonzero(finite_turns)),
        "fraction_of_rays_with_compact_turn":float(np.mean(finite_turns)),
        "multiple_turn_ray_count":int(np.count_nonzero(turn_counts>1)),
        "turn_z_range":None if not np.any(finite_turns) else [
            float(np.nanmin(turn_locations)),float(np.nanmax(turn_locations)),
        ],
        "axis_stationary_points":stationary,
        "minimum_sampled_invariant_gradient_magnitude":float(
            np.sqrt(np.min(invariant_gradient_squared))
        ),
        "maximum_abs_phi_r":float(np.max(np.abs(phi_r))),
        "maximum_abs_phi_z":float(np.max(np.abs(phi_z))),
        "divided_compact_gradient_variable_admissible":bool(not np.any(finite_turns)),
    }
