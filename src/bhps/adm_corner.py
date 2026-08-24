"""Time-symmetric ADM metric acceleration and spatial Israel corners.

The spatial dimension is four, the initial shift and extrinsic curvature
vanish, and the diagnostic lapse is ``alpha=psi``.  This audits the three
spatial tangential Israel rows and the mixed spatial wall-gauge row.  It does
not supply the time-time Israel row or a generalized-harmonic lapse evolution.
"""

from __future__ import annotations

import numpy as np

from bhps.gw_slice_high_order_solver import derivative_matrix


def _axisymmetric_derivatives(field,z,r,width):
    field=np.asarray(field,dtype=float);z=np.asarray(z,dtype=float);r=np.asarray(r,dtype=float)
    if field.shape!=(len(z),len(r)):raise ValueError("field shape does not match grid")
    dz=derivative_matrix(z,1,width);dzz=derivative_matrix(z,2,width)
    dr=derivative_matrix(r,1,width);drr=derivative_matrix(r,2,width)
    first_z=dz@field;first_r=field@dr.T
    second_z=dzz@field;second_r=field@drr.T;mixed=dz@field@dr.T
    transverse=np.divide(first_r,r[None,:],out=np.zeros_like(first_r),where=r[None,:]!=0)
    transverse[:,0]=second_r[:,0]
    return {
        "z":first_z,"r":first_r,"zz":second_z,"rr":second_r,
        "zr":mixed,"transverse_hessian":transverse,"Dz":dz,
    }


def time_symmetric_adm_metric_acceleration(
    z,r,psi,phi,chi_r,chi_z,m_phi_squared,m_chi_squared=0.,chi=None,
    cosmological_constant=-6.,kappa5_squared=1.,stencil_width=7,lapse=None,
):
    """Return ``partial_t^2 gamma_ij`` in a radial Cartesian frame.

    From the ``4+1`` ADM equation at ``K_ij=0``, zero shift, and
    ``alpha=psi``,

    ``gamma_tt = 2 alpha D_iD_j alpha - 2 alpha^2 R_ij
      +2 alpha^2 kappa^2[sum d_i f d_j f + gamma_ij V/3]
      +4 alpha^2 Lambda gamma_ij/3``.
    """
    psi=np.asarray(psi,dtype=float);phi=np.asarray(phi,dtype=float)
    chi_r=np.asarray(chi_r,dtype=float);chi_z=np.asarray(chi_z,dtype=float)
    if psi.shape!=phi.shape or psi.shape!=chi_r.shape or psi.shape!=chi_z.shape:
        raise ValueError("field shapes must agree")
    if np.any(psi<=0):raise ValueError("psi must be positive")
    alpha=psi if lapse is None else np.asarray(lapse,dtype=float)
    if alpha.shape!=psi.shape or np.any(alpha<=0):raise ValueError("lapse must be positive and match psi")
    omega_derivatives=_axisymmetric_derivatives(np.log(psi),z,r,stencil_width)
    lapse_derivatives=_axisymmetric_derivatives(alpha,z,r,stencil_width)
    phi_derivatives=_axisymmetric_derivatives(phi,z,r,stencil_width)
    wz,wr=omega_derivatives["z"],omega_derivatives["r"]
    wzz,wrr=omega_derivatives["zz"],omega_derivatives["rr"]
    wzr,wperp=omega_derivatives["zr"],omega_derivatives["transverse_hessian"]
    pz,pr=phi_derivatives["z"],phi_derivatives["r"]
    az,ar=lapse_derivatives["z"],lapse_derivatives["r"]
    azz,arr=lapse_derivatives["zz"],lapse_derivatives["rr"]
    azr,aperp=lapse_derivatives["zr"],lapse_derivatives["transverse_hessian"]
    laplacian=wzz+wrr+2*wperp
    gradient_squared=wz*wz+wr*wr
    conformal_ricci_common=laplacian+2*gradient_squared
    lapse_gradient_contraction=wz*az+wr*ar
    hessian_zz=azz-2*wz*az+lapse_gradient_contraction
    hessian_radial=arr-2*wr*ar+lapse_gradient_contraction
    hessian_transverse=aperp+lapse_gradient_contraction
    hessian_mixed=azr-wz*ar-wr*az
    ricci_zz=-2*(wzz-wz*wz)-conformal_ricci_common
    ricci_radial=-2*(wrr-wr*wr)-conformal_ricci_common
    ricci_transverse=-2*wperp-conformal_ricci_common
    ricci_mixed=-2*(wzr-wz*wr)
    potential=float(m_phi_squared)*phi*phi
    if float(m_chi_squared)!=0:
        if chi is None:raise ValueError("chi values required when m_chi_squared is nonzero")
        potential=potential+float(m_chi_squared)*np.asarray(chi,dtype=float)**2
    isotropic=alpha**2*psi**2*(
        2*float(kappa5_squared)*potential/3+4*float(cosmological_constant)/3
    )
    zz=2*alpha*hessian_zz-2*alpha**2*ricci_zz+2*alpha**2*float(kappa5_squared)*(pz*pz+chi_z*chi_z)+isotropic
    radial=2*alpha*hessian_radial-2*alpha**2*ricci_radial+2*alpha**2*float(kappa5_squared)*(pr*pr+chi_r*chi_r)+isotropic
    transverse=2*alpha*hessian_transverse-2*alpha**2*ricci_transverse+isotropic
    mixed=2*alpha*hessian_mixed-2*alpha**2*ricci_mixed+2*alpha**2*float(kappa5_squared)*(pz*pr+chi_z*chi_r)
    # The mixed radial-Cartesian tensor component is odd and vanishes on a
    # smooth symmetry axis.  One-sided scalar derivative stencils do not
    # preserve that tensor identity at the first node.
    mixed[:,0]=0.
    return {
        "zz":zz,"radial":radial,"transverse":transverse,"zr":mixed,
        "Dz":omega_derivatives["Dz"],
        "maximum_absolute_acceleration":float(max(
            np.max(np.abs(zz)),np.max(np.abs(radial)),np.max(np.abs(transverse)),np.max(np.abs(mixed))
        )),
        "lapse":alpha,
        "assumptions":"K_ij=0, zero shift, arbitrary positive initial lapse, four spatial dimensions",
    }


def time_symmetric_scalar_acceleration(
    z,r,psi,scalar,mass_squared,lapse=None,stencil_width=7,
):
    """Return ``partial_t^2 scalar`` at zero scalar momentum and shift."""
    psi=np.asarray(psi,dtype=float);scalar=np.asarray(scalar,dtype=float)
    alpha=psi if lapse is None else np.asarray(lapse,dtype=float)
    if scalar.shape!=psi.shape or alpha.shape!=psi.shape or np.any(psi<=0) or np.any(alpha<=0):
        raise ValueError("invalid scalar-acceleration inputs")
    w=_axisymmetric_derivatives(np.log(psi),z,r,stencil_width)
    f=_axisymmetric_derivatives(scalar,z,r,stencil_width)
    a=_axisymmetric_derivatives(alpha,z,r,stencil_width)
    flat_laplacian=f["zz"]+f["rr"]+2*f["transverse_hessian"]
    w_dot_f=w["z"]*f["z"]+w["r"]*f["r"]
    a_dot_f=a["z"]*f["z"]+a["r"]*f["r"]
    return (
        alpha**2*(flat_laplacian+2*w_dot_f)/psi**2
        +alpha*a_dot_f/psi**2-alpha**2*float(mass_squared)*scalar
    )


def shift_acceleration_correction(z,r,psi,shift_z_acceleration,shift_r_acceleration,stencil_width=7):
    """Return ``L_v gamma`` for an axisymmetric shift acceleration ``v``."""
    psi=np.asarray(psi,dtype=float)
    vz=np.asarray(shift_z_acceleration,dtype=float)
    vr=np.asarray(shift_r_acceleration,dtype=float)
    if psi.shape!=vz.shape or psi.shape!=vr.shape:raise ValueError("shift fields must match psi")
    w=_axisymmetric_derivatives(np.log(psi),z,r,stencil_width)
    dz=derivative_matrix(np.asarray(z,dtype=float),1,stencil_width)
    dr=derivative_matrix(np.asarray(r,dtype=float),1,stencil_width)
    vz_z=dz@vz;vz_r=vz@dr.T;vr_z=dz@vr;vr_r=vr@dr.T
    advection=2*psi**2*(vz*w["z"]+vr*w["r"])
    radial_ratio=np.divide(vr,np.asarray(r)[None,:],out=np.zeros_like(vr),where=np.asarray(r)[None,:]!=0)
    radial_ratio[:,0]=vr_r[:,0]
    return {
        "zz":advection+2*psi**2*vz_z,
        "radial":advection+2*psi**2*vr_r,
        "transverse":advection+2*psi**2*radial_ratio,
        "zr":psi**2*(vr_z+vz_r),
        "Dz":dz,
    }


def add_metric_accelerations(first,second):
    """Add two acceleration dictionaries while preserving their derivative."""
    return {
        name:np.asarray(first[name])+np.asarray(second[name])
        for name in ("zz","radial","transverse","zr")
    }|{"Dz":first["Dz"]}


def spatial_israel_second_corner_audit(
    acceleration,psi,phi,background,stabilizer_acceleration=None,
    radial_buffer=None,
):
    """Audit the spatial tangential Israel and mixed wall-gauge rows."""
    psi=np.asarray(psi,dtype=float);phi=np.asarray(phi,dtype=float)
    shape=psi.shape
    if phi.shape!=shape:raise ValueError("psi and phi shapes must agree")
    scalar_acceleration=(
        np.zeros(shape) if stabilizer_acceleration is None
        else np.asarray(stabilizer_acceleration,dtype=float)
    )
    if scalar_acceleration.shape!=shape:raise ValueError("invalid stabilizer acceleration shape")
    dz=acceleration["Dz"];gamma=float(background["wall_stiffness"])
    buffer=int(radial_buffer if radial_buffer is not None else min(7,shape[1]-1))
    radial_slice=slice(None,-buffer) if buffer>0 else slice(None)
    fields=spatial_israel_second_corner_residual_fields(
        acceleration,psi,phi,background,scalar_acceleration,buffer,
    )
    walls=[]
    for wall in fields["walls"]:
        components={}
        for name,item in wall["tangential_components"].items():
            local=np.abs(item["residual"]);normalized=local/item["scale"]
            maximum_index=int(np.argmax(normalized))
            components[name]={
                "maximum_absolute_residual":float(np.max(local)),
                "maximum_normalized_residual":float(np.max(normalized)),
                "maximum_index_within_retained_radial_grid":maximum_index,
            }
        mixed=np.abs(wall["mixed_zr_residual"]);mixed_scale=wall["mixed_zr_scale"]
        walls.append({
            "wall":wall["wall"],"tangential_components":components,
            "mixed_zr_dirichlet_acceleration_max":float(np.max(mixed)),
            "mixed_zr_maximum_normalized_acceleration":float(np.max(mixed/mixed_scale)),
        })
    return {
        "walls":walls,
        "maximum_tangential_normalized_residual":float(max(
            item["maximum_normalized_residual"]
            for wall in walls for item in wall["tangential_components"].values()
        )),
        "maximum_mixed_zr_acceleration":float(max(
            wall["mixed_zr_dirichlet_acceleration_max"] for wall in walls
        )),
        "maximum_mixed_zr_normalized_acceleration":float(max(
            wall["mixed_zr_maximum_normalized_acceleration"] for wall in walls
        )),
        "scope":"three spatial tangential Israel rows plus mixed spatial wall gauge",
    }


def spatial_israel_second_corner_residual_fields(
    acceleration,psi,phi,background,stabilizer_acceleration=None,
    radial_buffer=None,
):
    """Return unsummarized residual and scale arrays for covariance audits."""
    psi=np.asarray(psi,dtype=float);phi=np.asarray(phi,dtype=float)
    shape=psi.shape
    scalar_acceleration=(
        np.zeros(shape) if stabilizer_acceleration is None
        else np.asarray(stabilizer_acceleration,dtype=float)
    )
    if phi.shape!=shape or scalar_acceleration.shape!=shape:raise ValueError("invalid field shapes")
    dz=acceleration["Dz"];gamma=float(background["wall_stiffness"])
    buffer=int(radial_buffer if radial_buffer is not None else min(7,shape[1]-1))
    radial_slice=slice(None,-buffer) if buffer>0 else slice(None)
    walls=[]
    for wall_index,index,target in (
        (0,0,float(background["v0"])),(1,-1,float(background["v1"])),
    ):
        wall_potential=.5*gamma*(phi[index]-target)**2
        if wall_index==0:
            beta=float(background["beta_a"])+(wall_potential-float(background["wall_potential_a"]))/6
            beta_phi=gamma*(phi[index]-target)/6
        else:
            beta=float(background["beta_b"])-(wall_potential-float(background["wall_potential_b"]))/6
            beta_phi=-gamma*(phi[index]-target)/6
        components={}
        for name in ("radial","transverse"):
            value=np.asarray(acceleration[name])[index]
            derivative=(dz@np.asarray(acceleration[name]))[index]
            terms=(
                derivative,
                2*beta*psi[index]*value,
                beta*psi[index]*np.asarray(acceleration["zz"])[index],
                2*beta_phi*scalar_acceleration[index]*psi[index]**3,
            )
            residual=sum(terms)
            scale=np.maximum(1.,sum(np.abs(term) for term in terms))
            components[name]={
                "residual":np.asarray(residual[radial_slice]),
                "scale":np.asarray(scale[radial_slice]),
            }
        mixed=np.asarray(acceleration["zr"])[index,radial_slice]
        mixed_scale=np.maximum(1.,
            np.abs(np.asarray(acceleration["zz"])[index,radial_slice])
            +np.abs(np.asarray(acceleration["radial"])[index,radial_slice])
            +np.abs(np.asarray(acceleration["transverse"])[index,radial_slice])
        )
        walls.append({
            "wall":"lower" if wall_index==0 else "upper",
            "tangential_components":components,
            "mixed_zr_residual":np.asarray(mixed),
            "mixed_zr_scale":np.asarray(mixed_scale),
        })
    return {"walls":walls,"radial_buffer":buffer}
