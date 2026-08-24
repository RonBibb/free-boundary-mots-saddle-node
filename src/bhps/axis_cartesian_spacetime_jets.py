"""Regular Cartesian spacetime jets from axisymmetric initial data.

The input spatial metric is

``gamma = A^2 dz^2 + B^2 dr^2 + C^2 r^2 dOmega_2^2``

with ``A=psi exp(a)``, ``B=psi exp(b)``, and ``C=psi exp(c)``.  Cylindrical
components are singular coordinates at ``r=0`` even when the geometry is
smooth.  This module converts their even-axis limits into ordinary Cartesian
jets in coordinates ``(t,z,x,y,w)``.  It is intended for pointwise covariant
operator audits at the symmetry axis.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def _validate_field(name,field,shape):
    value=np.asarray(field,dtype=float)
    if value.shape!=shape or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite field on the z-r grid")
    return value


def _even_axis_radial_second(z,r,field,fit_points=6,degree=3):
    """Estimate ``d_r^2 field(z,0)`` from an even polynomial in ``r^2``."""
    count=int(fit_points);degree=int(degree)
    if count<degree+1 or count>len(r):
        raise ValueError("invalid even-axis fit size")
    powers=np.asarray(r[:count],dtype=float)**2
    return np.array([
        2*np.polynomial.polynomial.polyfit(powers,row[:count],degree)[1]
        for row in np.asarray(field)
    ])


def _axis_jet(z,r,field,location,fit_points,degree):
    """Return value, compact derivatives, and radial Hessian on the axis."""
    axis=CubicSpline(z,np.asarray(field)[:,0])
    radial_second=CubicSpline(
        z,_even_axis_radial_second(z,r,field,fit_points,degree),
    )
    return {
        "value":float(axis(location)),
        "z":float(axis.derivative(1)(location)),
        "zz":float(axis.derivative(2)(location)),
        "rr":float(radial_second(location)),
    }


def _axis_value(z,field,location):
    return float(CubicSpline(z,np.asarray(field)[:,0])(location))


def construct_time_symmetric_axis_spacetime_jets(
    z,r,alpha,psi,a,b,c,phi,chi,
    metric_acceleration,lapse_acceleration,phi_acceleration,chi_acceleration,
    location,radial_fit_points=6,radial_polynomial_degree=3,
):
    """Construct regular five-dimensional jets at ``(z=location,r=0)``.

    First time derivatives, the shift, and its acceleration are taken to
    vanish.  ``metric_acceleration`` supplies the ADM values of
    ``partial_t^2 gamma`` in the compact, radial, transverse, and mixed
    directions.  The returned ``background`` dictionary can be passed
    directly to the pointwise generalized-harmonic Einstein--scalar kernel.
    """
    z=np.asarray(z,dtype=float);r=np.asarray(r,dtype=float)
    if (
        z.ndim!=1 or r.ndim!=1 or len(z)<4 or len(r)<4
        or np.any(np.diff(z)<=0) or np.any(np.diff(r)<=0) or r[0]!=0
    ):
        raise ValueError("invalid z-r grid")
    location=float(location)
    if not z[0]<=location<=z[-1]:raise ValueError("axis location lies outside grid")
    shape=(len(z),len(r))
    alpha=_validate_field("alpha",alpha,shape)
    psi=_validate_field("psi",psi,shape)
    a=_validate_field("a",a,shape);b=_validate_field("b",b,shape)
    c=_validate_field("c",c,shape);phi=_validate_field("phi",phi,shape)
    chi=_validate_field("chi",chi,shape)
    lapse_acceleration=_validate_field(
        "lapse_acceleration",lapse_acceleration,shape,
    )
    phi_acceleration=_validate_field("phi_acceleration",phi_acceleration,shape)
    chi_acceleration=_validate_field("chi_acceleration",chi_acceleration,shape)
    if np.any(alpha<=0) or np.any(psi<=0):raise ValueError("alpha and psi must be positive")
    accelerations={
        name:_validate_field(f"metric_acceleration[{name}]",metric_acceleration[name],shape)
        for name in ("zz","radial","transverse","zr")
    }

    compact=psi**2*np.exp(2*a)
    radial=psi**2*np.exp(2*b)
    transverse=psi**2*np.exp(2*c)
    metric_fields={
        "time":-alpha**2,"compact":compact,"transverse":transverse,
        "anisotropy":radial-transverse,
    }
    jets={
        name:_axis_jet(
            z,r,field,location,radial_fit_points,radial_polynomial_degree,
        )
        for name,field in metric_fields.items()
    }
    phi_jet=_axis_jet(
        z,r,phi,location,radial_fit_points,radial_polynomial_degree,
    )
    chi_jet=_axis_jet(
        z,r,chi,location,radial_fit_points,radial_polynomial_degree,
    )

    dimension=5;metric=np.zeros((dimension,dimension))
    first=np.zeros((dimension,dimension,dimension))
    second=np.zeros((dimension,dimension,dimension,dimension))
    metric[0,0]=jets["time"]["value"]
    metric[1,1]=jets["compact"]["value"]
    for index in range(2,dimension):metric[index,index]=jets["transverse"]["value"]
    for left,name in ((0,"time"),(1,"compact")):
        first[1,left,left]=jets[name]["z"]
        second[1,1,left,left]=jets[name]["zz"]
        for transverse_index in range(2,dimension):
            second[transverse_index,transverse_index,left,left]=jets[name]["rr"]
    for left in range(2,dimension):
        first[1,left,left]=jets["transverse"]["z"]
        second[1,1,left,left]=jets["transverse"]["zz"]
    # If F=C^2 and D=B^2-C^2=(D_rr/2) r^2+O(r^4), then
    # d_k d_l gamma_ij = F_rr delta_kl delta_ij
    #   +(D_rr/2)(delta_ik delta_jl+delta_il delta_jk).
    for derivative_left in range(2,dimension):
        for derivative_right in range(2,dimension):
            for metric_left in range(2,dimension):
                for metric_right in range(2,dimension):
                    value=0.
                    if derivative_left==derivative_right and metric_left==metric_right:
                        value+=jets["transverse"]["rr"]
                    if derivative_left==metric_left and derivative_right==metric_right:
                        value+=.5*jets["anisotropy"]["rr"]
                    if derivative_left==metric_right and derivative_right==metric_left:
                        value+=.5*jets["anisotropy"]["rr"]
                    second[derivative_left,derivative_right,metric_left,metric_right]=value

    alpha_value=float(np.sqrt(-metric[0,0]))
    second[0,0,0,0]=-2*alpha_value*_axis_value(
        z,lapse_acceleration,location,
    )
    second[0,0,1,1]=_axis_value(z,accelerations["zz"],location)
    transverse_acceleration=_axis_value(z,accelerations["transverse"],location)
    for index in range(2,dimension):second[0,0,index,index]=transverse_acceleration

    phi_first=np.zeros(dimension);phi_second=np.zeros((dimension,dimension))
    chi_first=np.zeros(dimension);chi_second=np.zeros((dimension,dimension))
    phi_first[1]=phi_jet["z"];chi_first[1]=chi_jet["z"]
    phi_second[0,0]=_axis_value(z,phi_acceleration,location)
    chi_second[0,0]=_axis_value(z,chi_acceleration,location)
    phi_second[1,1]=phi_jet["zz"];chi_second[1,1]=chi_jet["zz"]
    for index in range(2,dimension):
        phi_second[index,index]=phi_jet["rr"]
        chi_second[index,index]=chi_jet["rr"]

    radial_metric_value=_axis_value(z,radial,location)
    radial_acceleration=_axis_value(z,accelerations["radial"],location)
    mixed_acceleration=_axis_value(z,accelerations["zr"],location)
    regularity_scale=max(abs(radial_metric_value),abs(metric[2,2]),1e-300)
    acceleration_scale=max(abs(radial_acceleration),abs(transverse_acceleration),1e-300)
    background={
        "metric":metric,"metric_first":first,"metric_second":second,
        "phi":phi_jet["value"],"phi_first":phi_first,"phi_second":phi_second,
        "chi":chi_jet["value"],"chi_first":chi_first,"chi_second":chi_second,
    }
    return {
        "background":background,
        "location":{"z":location,"r":0.},
        "axis_field_jets":{"phi":phi_jet,"chi":chi_jet,**jets},
        "regularity":{
            "radial_minus_transverse_metric":radial_metric_value-metric[2,2],
            "relative_metric_mismatch":abs(radial_metric_value-metric[2,2])/regularity_scale,
            "radial_minus_transverse_acceleration":radial_acceleration-transverse_acceleration,
            "relative_acceleration_mismatch":abs(radial_acceleration-transverse_acceleration)/acceleration_scale,
            "mixed_zr_acceleration":mixed_acceleration,
        },
        "assumptions":[
            "time-symmetric first jets",
            "zero shift and zero shift acceleration",
            "smooth even axis with B=C",
        ],
    }
