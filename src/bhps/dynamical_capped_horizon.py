"""Outgoing-null expansion for donor-capped surfaces on evolved SO(3) slices.

The four-dimensional spatial metric is written as a two-dimensional base
``h_AB dx^A dx^B`` in ``(z,r)`` warped by a round two-sphere of squared radius
``r^2 g_perp``.  For a unit outward base normal ``s`` the expansion convention
used here is

``theta_+ = D_i s^i + K_ij s^i s^j - K``

with ``K_ij=(-gamma_ij,t + L_beta gamma_ij)/(2 alpha)``.  It reduces to the
existing minimal-surface equation when the shift and spatial metric velocity
vanish.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline,RectBivariateSpline
from scipy.optimize import least_squares

from bhps.adm_corner import _axisymmetric_derivatives


def _derivatives(field,z,r,stencil_width):
    return _axisymmetric_derivatives(
        np.asarray(field,dtype=float),np.asarray(z,dtype=float),
        np.asarray(r,dtype=float),int(stencil_width),
    )


def regular_so3_adm_slice(position,velocity,z,r,stencil_width=7):
    """Return lapse, base geometry, and extrinsic curvature on one slice."""
    q=np.asarray(position,dtype=float);v=np.asarray(velocity,dtype=float)
    z=np.asarray(z,dtype=float);r=np.asarray(r,dtype=float)
    expected=(len(z),len(r),9)
    if q.shape!=expected or v.shape!=expected or r[0]!=0:
        raise ValueError("invalid evolved regular SO(3) slice")
    radius=r[None,:]
    h=np.empty((len(z),len(r),2,2));hdot=np.empty_like(h)
    h[:,:,0,0]=q[:,:,6];h[:,:,0,1]=h[:,:,1,0]=radius*q[:,:,1]
    h[:,:,1,1]=q[:,:,3]+radius**2*q[:,:,4]
    hdot[:,:,0,0]=v[:,:,6];hdot[:,:,0,1]=hdot[:,:,1,0]=radius*v[:,:,1]
    hdot[:,:,1,1]=v[:,:,3]+radius**2*v[:,:,4]
    determinant=h[:,:,0,0]*h[:,:,1,1]-h[:,:,0,1]**2
    if np.any(determinant<=0) or np.any(q[:,:,3]<=0):
        raise RuntimeError("spatial slice is not positive definite")
    inverse=np.empty_like(h)
    inverse[:,:,0,0]=h[:,:,1,1]/determinant
    inverse[:,:,1,1]=h[:,:,0,0]/determinant
    inverse[:,:,0,1]=inverse[:,:,1,0]=-h[:,:,0,1]/determinant
    beta_cov=np.stack((q[:,:,0],radius*q[:,:,5]),axis=2)
    beta=np.einsum("...ab,...b->...a",inverse,beta_cov)
    lapse_squared=-q[:,:,2]+np.einsum("...a,...a->...",beta_cov,beta)
    if np.any(lapse_squared<=0):raise RuntimeError("coordinate-time normal is not timelike")
    lapse=np.sqrt(lapse_squared)

    derivatives={}
    for a,b,name in ((0,0,"zz"),(0,1,"zr"),(1,1,"rr")):
        derivatives[a,b]=derivatives[b,a]=_derivatives(
            h[:,:,a,b],z,r,stencil_width,
        )
    dh=np.empty((2,len(z),len(r),2,2))
    for direction,key in enumerate(("z","r")):
        for a in range(2):
            for b in range(2):dh[direction,:,:,a,b]=derivatives[a,b][key]
    connection=np.zeros((len(z),len(r),2,2,2))
    for upper in range(2):
        for left in range(2):
            for right in range(2):
                for contracted in range(2):
                    connection[:,:,upper,left,right]+=.5*inverse[:,:,upper,contracted]*(
                        dh[left,:,:,contracted,right]
                        +dh[right,:,:,contracted,left]
                        -dh[contracted,:,:,left,right]
                    )
    dbeta=np.empty((2,len(z),len(r),2))
    for component in range(2):
        local=_derivatives(beta_cov[:,:,component],z,r,stencil_width)
        dbeta[0,:,:,component]=local["z"];dbeta[1,:,:,component]=local["r"]
    covariant_beta=np.empty((len(z),len(r),2,2))
    for left in range(2):
        for right in range(2):
            covariant_beta[:,:,left,right]=dbeta[left,:,:,right]
            for contracted in range(2):
                covariant_beta[:,:,left,right]-=(
                    connection[:,:,contracted,left,right]*beta_cov[:,:,contracted]
                )
    extrinsic_base=(
        -hdot+covariant_beta+np.swapaxes(covariant_beta,-1,-2)
    )/(2*lapse[:,:,None,None])

    transverse=q[:,:,3];transverse_dot=v[:,:,3]
    transverse_derivatives=_derivatives(transverse,z,r,stencil_width)
    log_transverse_gradient=np.empty((len(z),len(r),2))
    log_transverse_gradient[:,:,0]=.5*transverse_derivatives["z"]/transverse
    log_transverse_gradient[:,:,1]=.5*transverse_derivatives["r"]/transverse
    log_radius_gradient=log_transverse_gradient.copy()
    log_radius_gradient[:,1:,1]+=1/r[None,1:]
    extrinsic_sphere=(
        -.5*transverse_dot/transverse
        +np.einsum("...a,...a->...",beta,log_radius_gradient)
    )/lapse
    return {
        "lapse":lapse,"shift_covector":beta_cov,"shift":beta,
        "base_metric":h,"base_inverse":inverse,"base_connection":connection,
        "log_transverse_scale_gradient":log_transverse_gradient,
        "log_sphere_radius_gradient":log_radius_gradient,
        "extrinsic_base":extrinsic_base,
        "extrinsic_sphere_eigenvalue":extrinsic_sphere,
        "minimum_spatial_determinant":float(np.min(determinant)),
        "minimum_lapse":float(np.min(lapse)),
    }


def _spline(z,r,field):
    return RectBivariateSpline(
        np.asarray(z,dtype=float),np.asarray(r,dtype=float),
        np.asarray(field,dtype=float),kx=min(3,len(z)-1),ky=min(3,len(r)-1),s=0,
    )


class PreparedCappedExpansionSlice:
    """Reusable interpolation data for repeated surface solves on one slice."""

    def __init__(self,position,velocity,z,r,stencil_width=7):
        self.z=np.asarray(z,dtype=float);self.r=np.asarray(r,dtype=float)
        self.stencil_width=int(stencil_width)
        self.adm=regular_so3_adm_slice(
            position,velocity,self.z,self.r,self.stencil_width,
        )
        self.splines={}
        for name in (
            "base_metric","base_inverse","base_connection",
            "log_transverse_scale_gradient","extrinsic_base",
            "extrinsic_sphere_eigenvalue",
        ):
            field=np.asarray(self.adm[name])
            if field.ndim==2:
                self.splines[(name,)]=_spline(self.z,self.r,field)
            else:
                for index in np.ndindex(field.shape[2:]):
                    self.splines[(name,*index)]=_spline(
                        self.z,self.r,field[(slice(None),slice(None),*index)],
                    )

    def sample(self,key,zcoord,radius):
        return self.splines[tuple(key)].ev(zcoord,radius)


def prepare_capped_expansion_slice(position,velocity,z,r,stencil_width=7):
    """Prepare one evolved slice for repeated capped-expansion evaluations."""
    return PreparedCappedExpansionSlice(position,velocity,z,r,stencil_width)


def capped_outgoing_expansion(
    position,velocity,z,r,profile,stencil_width=7,prepared=None,
):
    """Evaluate ``theta_+`` along one donor-capped profile.

    ``profile`` supplies arrays ``theta``, ``rho``, and ``slope=d rho/d theta``
    in the same star-shaped convention used by the static anisotropic finder.
    """
    theta=np.asarray(profile["theta"],dtype=float)
    rho=np.asarray(profile["rho"],dtype=float)
    slope=np.asarray(profile["slope"],dtype=float)
    if theta.ndim!=1 or rho.shape!=theta.shape or slope.shape!=theta.shape or len(theta)<5:
        raise ValueError("invalid capped profile")
    z=np.asarray(z,dtype=float);r=np.asarray(r,dtype=float);z_b=float(z[-1])
    radius=rho*np.sin(theta);zcoord=z_b-rho*np.cos(theta)
    if (
        np.min(radius)<=0 or np.max(radius)>r[-1]
        or np.min(zcoord)<z[0] or np.max(zcoord)>z[-1]
    ):
        raise ValueError("capped profile leaves the numerical domain")
    prepared=(
        prepare_capped_expansion_slice(position,velocity,z,r,stencil_width)
        if prepared is None else prepared
    )
    if not (
        np.array_equal(prepared.z,z) and np.array_equal(prepared.r,r)
        and prepared.stencil_width==int(stencil_width)
    ):
        raise ValueError("prepared capped slice uses a different grid")

    h=np.empty((len(theta),2,2));inverse=np.empty_like(h)
    connection=np.empty((len(theta),2,2,2));log_gradient=np.empty((len(theta),2))
    extrinsic=np.empty_like(h)
    for a in range(2):
        log_gradient[:,a]=prepared.sample(
            ("log_transverse_scale_gradient",a),zcoord,radius,
        )
        for b in range(2):
            h[:,a,b]=prepared.sample(("base_metric",a,b),zcoord,radius)
            inverse[:,a,b]=prepared.sample(("base_inverse",a,b),zcoord,radius)
            extrinsic[:,a,b]=prepared.sample(("extrinsic_base",a,b),zcoord,radius)
            for upper in range(2):
                connection[:,upper,a,b]=prepared.sample(
                    ("base_connection",upper,a,b),zcoord,radius,
                )
    extrinsic_sphere=prepared.sample(
        ("extrinsic_sphere_eigenvalue",),zcoord,radius,
    )
    # Interpolate only the smooth transverse scale and add the coordinate
    # sphere-radius derivative analytically. Interpolating a nodal 1/r field
    # across the regular axis would introduce a false spline boundary layer.
    log_gradient[:,1]+=1/radius
    tangent_coordinate=np.stack((
        rho*np.sin(theta)-slope*np.cos(theta),
        rho*np.cos(theta)+slope*np.sin(theta),
    ),axis=1)
    speed=np.sqrt(np.einsum("...a,...ab,...b->...",tangent_coordinate,h,tangent_coordinate))
    tangent=tangent_coordinate/speed[:,None]
    normal_covector=np.stack((-tangent_coordinate[:,1],tangent_coordinate[:,0]),axis=1)
    normal_norm=np.sqrt(np.einsum(
        "...a,...ab,...b->...",normal_covector,inverse,normal_covector,
    ))
    normal_covector/=normal_norm[:,None]
    normal=np.einsum("...ab,...b->...a",inverse,normal_covector)
    normal_covector_theta=np.gradient(normal_covector,theta,axis=0,edge_order=2)
    curve_divergence=(
        np.einsum("...a,...a->...",tangent,normal_covector_theta)/speed
        -np.einsum(
            "...a,...b,...cab,...c->...",tangent,tangent,connection,normal_covector,
        )
    )
    sphere_divergence=2*np.einsum("...a,...a->...",normal,log_gradient)
    mean_curvature=curve_divergence+sphere_divergence
    tangent_extrinsic=np.einsum("...a,...ab,...b->...",tangent,extrinsic,tangent)
    extrinsic_correction=-tangent_extrinsic-2*extrinsic_sphere
    expansion=mean_curvature+extrinsic_correction
    scale=np.maximum(1.,np.abs(mean_curvature)+np.abs(extrinsic_correction))
    dz=float(np.min(np.diff(z)));dr=float(np.min(np.diff(r)))
    interior=(radius>=2*dr)&(zcoord<=z_b-2*dz)
    if not np.any(interior):raise ValueError("capped profile has no two-cell interior")
    return {
        "theta":theta,"rho":rho,"z":zcoord,"r":radius,
        "outgoing_expansion":expansion,"mean_curvature":mean_curvature,
        "extrinsic_curvature_correction":extrinsic_correction,
        "maximum_absolute_expansion":float(np.max(np.abs(expansion))),
        "maximum_normalized_expansion":float(np.max(np.abs(expansion)/scale)),
        "minimum_expansion":float(np.min(expansion)),
        "maximum_expansion":float(np.max(expansion)),
        "two_cell_interior_mask":interior,
        "two_cell_interior_count":int(np.count_nonzero(interior)),
        "two_cell_interior_maximum_absolute":float(np.max(np.abs(expansion[interior]))),
        "two_cell_interior_maximum_normalized":float(np.max(np.abs(expansion[interior])/scale[interior])),
        "minimum_curve_speed":float(np.min(speed)),
        "finite":bool(np.all(np.isfinite(expansion))),
        "convention":"theta_plus=D_i s^i+K_ij s^i s^j-K",
    }


def _regularized_expansion(values,theta,radius,zcoord,r_grid,z_grid):
    """Fill coordinate endpoint collars from the regular interior limit."""
    result=np.asarray(values,dtype=float).copy();theta=np.asarray(theta,dtype=float)
    dr=float(np.min(np.diff(r_grid)));dz=float(np.min(np.diff(z_grid)))
    axis_safe=np.flatnonzero(radius>=2*dr)
    wall_safe=np.flatnonzero(zcoord<=z_grid[-1]-2*dz)
    if len(axis_safe)<6 or len(wall_safe)<6:
        raise ValueError("surface lacks a regular endpoint interpolation collar")
    first=int(axis_safe[0]);axis_indices=axis_safe[:min(8,len(axis_safe))]
    if first>0:
        coefficients=np.polynomial.polynomial.polyfit(
            theta[axis_indices]**2,result[axis_indices],3,
        )
        result[:first]=np.polynomial.polynomial.polyval(theta[:first]**2,coefficients)
    last=int(wall_safe[-1]);wall_indices=wall_safe[-min(8,len(wall_safe)):]
    if last<len(result)-1:
        distance=np.pi/2-theta
        coefficients=np.polynomial.polynomial.polyfit(
            distance[wall_indices],result[wall_indices],3,
        )
        result[last+1:]=np.polynomial.polynomial.polyval(
            distance[last+1:],coefficients,
        )
    return result


def solve_dynamical_capped_surface(
    position,velocity,z,r,initial,tolerance=5e-5,nodes=81,
    maximum_evaluations=300,stencil_width=7,
):
    """Solve ``theta_+=0`` with smooth-axis and orthogonal-wall endpoints.

    This nodal least-squares solver is independent of the static area BVP. It
    uses the full evolved-slice expansion and continues naturally from a prior
    capped profile.
    """
    z=np.asarray(z,dtype=float);r=np.asarray(r,dtype=float)
    theta=np.linspace(1e-4,np.pi/2,int(nodes))
    if np.isscalar(initial):rho0=np.full_like(theta,float(initial))
    else:
        rho0=np.interp(theta,np.asarray(initial["theta"]),np.asarray(initial["rho"]))
    prepared=prepare_capped_expansion_slice(
        position,velocity,z,r,stencil_width,
    )
    lower=max(2.05*(r[1]-r[0]),1e-4)
    upper=.98*min(z[-1]-z[0],r[-1]/np.sin(theta[-1]))

    def profile(values):
        curve=CubicSpline(theta,values)
        return {"theta":theta,"rho":values,"slope":curve(theta,1)}

    def residual(values):
        local=profile(values)
        try:
            expansion=capped_outgoing_expansion(
                position,velocity,z,r,local,stencil_width,prepared,
            )
            regular=_regularized_expansion(
                expansion["outgoing_expansion"],theta,expansion["r"],
                expansion["z"],r,z,
            )
        except (ValueError,RuntimeError,np.linalg.LinAlgError):
            return np.full(len(theta),1e3)
        scale=max(float(np.mean(values)),1e-6)
        result=regular.copy()
        result[0]=local["slope"][0]/scale**2
        result[-1]=local["slope"][-1]/scale**2
        return result

    solved=least_squares(
        residual,rho0,bounds=(lower,upper),xtol=1e-10,ftol=1e-10,gtol=1e-10,
        max_nfev=int(maximum_evaluations),x_scale="jac",
    )
    local=profile(solved.x)
    expansion=capped_outgoing_expansion(
        position,velocity,z,r,local,stencil_width,prepared,
    )
    regular=_regularized_expansion(
        expansion["outgoing_expansion"],theta,expansion["r"],expansion["z"],r,z,
    )
    scale=max(float(np.mean(solved.x)),1e-6)
    boundary=max(abs(local["slope"][0]),abs(local["slope"][-1]))
    maximum=float(np.max(np.abs(regular[1:-1])))
    in_domain=bool(
        np.min(expansion["z"])>z[0]+2*(z[1]-z[0])
        and np.max(expansion["r"])<.9*r[-1] and np.min(solved.x)>2*r[1]
    )
    return {
        "converged":bool(solved.success and in_domain and maximum<float(tolerance) and boundary/scale**2<float(tolerance)),
        "optimizer_success":bool(solved.success),"message":str(solved.message),
        "in_domain":in_domain,"function_evaluations":int(solved.nfev),
        "theta":theta,"rho":solved.x,"slope":local["slope"],
        "rho_axis":float(solved.x[0]),"rho_brane":float(solved.x[-1]),
        "boundary_slope_error":float(boundary),
        "regularized_expansion_maximum":maximum,
        "regularized_expansion_l2":float(np.linalg.norm(regular[1:-1])),
        "raw_two_cell_interior_maximum":expansion["two_cell_interior_maximum_absolute"],
        "minimum_expansion":float(np.min(regular[1:-1])),
        "maximum_expansion":float(np.max(regular[1:-1])),
    }


def solve_spectral_dynamical_capped_surface(
    position,velocity,z,r,initial,tolerance=5e-5,collocation_nodes=101,
    cosine_modes=16,maximum_evaluations=200,stencil_width=7,prepared=None,
):
    """Solve ``theta_+=0`` in a smooth endpoint-compatible cosine basis.

    ``rho=sum a_n cos(2 n theta)`` has zero derivative at both the symmetry
    axis and compact wall.  The mode coefficients are determined from an
    overdetermined set of expansion equations in the two-native-cell interior,
    avoiding the free endpoint-collar degrees of the nodal engineering solver.
    """
    z=np.asarray(z,dtype=float);r=np.asarray(r,dtype=float)
    count=int(collocation_nodes);modes=int(cosine_modes)
    if count<2*modes+5 or modes<4:
        raise ValueError("spectral cap solve needs at least twice as many collocation nodes as modes")
    theta=np.linspace(1e-4,np.pi/2,count);indices=np.arange(modes)
    basis=np.cos(2*theta[:,None]*indices[None,:])
    derivative=-2*indices[None,:]*np.sin(2*theta[:,None]*indices[None,:])
    initial_rho=(
        np.full_like(theta,float(initial)) if np.isscalar(initial) else
        np.interp(theta,np.asarray(initial["theta"]),np.asarray(initial["rho"]))
    )
    coefficients=np.linalg.lstsq(basis,initial_rho,rcond=None)[0]
    prepared=(
        prepare_capped_expansion_slice(position,velocity,z,r,stencil_width)
        if prepared is None else prepared
    )
    initial_radius=initial_rho*np.sin(theta)
    initial_z=z[-1]-initial_rho*np.cos(theta)
    fixed_mask=(
        (initial_radius>=2*np.min(np.diff(r)))
        &(initial_z<=z[-1]-2*np.min(np.diff(z)))
    )
    if np.count_nonzero(fixed_mask)<2*modes:
        raise ValueError("spectral cap has too few regular interior equations")

    def local_profile(values):
        return {
            "theta":theta,"rho":basis@values,"slope":derivative@values,
        }

    def residual(values):
        local=local_profile(values)
        try:
            expansion=capped_outgoing_expansion(
                position,velocity,z,r,local,stencil_width,prepared,
            )
            return expansion["outgoing_expansion"][fixed_mask]
        except (ValueError,RuntimeError,np.linalg.LinAlgError):
            return np.full(np.count_nonzero(fixed_mask),1e3)

    solved=least_squares(
        residual,coefficients,xtol=1e-11,ftol=1e-11,gtol=1e-11,
        max_nfev=int(maximum_evaluations),x_scale="jac",
    )
    local=local_profile(solved.x)
    expansion=capped_outgoing_expansion(
        position,velocity,z,r,local,stencil_width,prepared,
    )
    values=expansion["outgoing_expansion"][fixed_mask]
    maximum=float(np.max(np.abs(values)))
    endpoint_derivative=np.array((
        np.sum(-2*indices*np.sin(0)*solved.x),
        np.sum(-2*indices*np.sin(np.pi*indices)*solved.x),
    ))
    singular=np.linalg.svd(np.asarray(solved.jac),compute_uv=False)
    condition=float(singular[0]/max(singular[-1],1e-300))
    in_domain=bool(
        np.min(expansion["z"])>z[0]+2*(z[1]-z[0])
        and np.max(expansion["r"])<.9*r[-1]
        and np.min(local["rho"])>2*r[1]
    )
    return {
        "converged":bool(solved.success and in_domain and maximum<float(tolerance)),
        "optimizer_success":bool(solved.success),"message":str(solved.message),
        "in_domain":in_domain,"function_evaluations":int(solved.nfev),
        "collocation_nodes":count,"cosine_modes":modes,
        "theta":theta,"rho":local["rho"],"slope":local["slope"],
        "cosine_coefficients":solved.x,
        "rho_axis":float(np.sum(solved.x)),
        "rho_brane":float(np.sum(((-1.)**indices)*solved.x)),
        "boundary_slope_error":float(np.max(np.abs(endpoint_derivative))),
        "interior_expansion_maximum":maximum,
        "interior_expansion_l2":float(np.linalg.norm(values)),
        "interior_equation_count":int(np.count_nonzero(fixed_mask)),
        "minimum_jacobian_singular_value":float(singular[-1]),
        "maximum_jacobian_singular_value":float(singular[0]),
        "jacobian_condition_number":condition,
        "minimum_expansion":float(np.min(values)),
        "maximum_expansion":float(np.max(values)),
    }
