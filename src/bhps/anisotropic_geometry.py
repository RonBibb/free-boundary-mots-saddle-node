"""Axisymmetric non-conformal geometry for time-symmetric 4+1 initial data.

The four-dimensional spatial metric is

``gamma = A^2 dz^2 + B^2 dr^2 + rho^2 dOmega_2^2``

with ``A=psi exp(a)``, ``B=psi exp(b)``, and
``rho=r psi exp(c)``.  The implementation uses the two-dimensional warped-
product identities and returns radial-Cartesian components, so the transverse
metric component is ``rho^2/r^2`` away from the axis.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson

from bhps.adm_corner import _axisymmetric_derivatives


def _differentiate(field,z,r,width):
    return _axisymmetric_derivatives(np.asarray(field,dtype=float),z,r,width)


def axisymmetric_diagonal_geometry(z,r,psi,a,b,c,stencil_width=7,_well_balanced=True):
    """Return metric, Ricci, and warped-product auxiliary fields.

    The raw warped-product discretization is corrected by its conformal
    discrete defect.  Thus ``a=b=c=0`` reproduces the established conformal
    Ricci stencil exactly even when the input is a discrete elliptic solution
    whose high derivatives are not spectrally smooth.
    """
    z=np.asarray(z,dtype=float);r=np.asarray(r,dtype=float)
    psi=np.asarray(psi,dtype=float);a=np.asarray(a,dtype=float)
    b=np.asarray(b,dtype=float);c=np.asarray(c,dtype=float)
    if not (psi.shape==a.shape==b.shape==c.shape==(len(z),len(r))):
        raise ValueError("metric fields must share the z-r grid shape")
    if np.any(psi<=0):raise ValueError("psi must be positive")
    aa=psi*np.exp(a);bb=psi*np.exp(b);cc=psi*np.exp(c)
    rho=cc*r[None,:]
    da,db,dc,drho=(_differentiate(item,z,r,stencil_width) for item in (aa,bb,cc,rho))

    # Scalar curvature of the orthogonal two-metric A^2 dz^2+B^2 dr^2.
    bz_over_a=db["z"]/aa;ar_over_b=da["r"]/bb
    d_bz_over_a=_differentiate(bz_over_a,z,r,stencil_width)["z"]
    d_ar_over_b=_differentiate(ar_over_b,z,r,stencil_width)["r"]
    base_scalar=-2*(d_bz_over_a+d_ar_over_b)/(aa*bb)

    gamma_z_zz=da["z"]/aa;gamma_z_zr=da["r"]/aa
    gamma_z_rr=-bb*db["z"]/aa**2
    gamma_r_zz=-aa*da["r"]/bb**2
    gamma_r_zr=db["z"]/bb;gamma_r_rr=db["r"]/bb
    hess_rho_zz=drho["zz"]-gamma_z_zz*drho["z"]-gamma_r_zz*drho["r"]
    hess_rho_rr=drho["rr"]-gamma_z_rr*drho["z"]-gamma_r_rr*drho["r"]
    hess_rho_zr=drho["zr"]-gamma_z_zr*drho["z"]-gamma_r_zr*drho["r"]

    rho_safe=np.where(rho!=0,rho,1.)
    hzz_over_rho=hess_rho_zz/rho_safe
    hrr_over_rho=hess_rho_rr/rho_safe
    hzr_over_rho=hess_rho_zr/rho_safe
    # Smooth-axis limits for even A,B,C with B=C at r=0.
    hzz_over_rho[:,0]=(
        dc["zz"][:,0]/cc[:,0]
        -da["z"][:,0]*dc["z"][:,0]/(aa[:,0]*cc[:,0])
        +aa[:,0]*da["rr"][:,0]/bb[:,0]**2
    )
    hrr_over_rho[:,0]=(
        3*dc["rr"][:,0]/cc[:,0]
        +bb[:,0]*db["z"][:,0]*dc["z"][:,0]/(aa[:,0]**2*cc[:,0])
        -db["rr"][:,0]/bb[:,0]
    )
    hzr_over_rho[:,0]=0.

    laplacian_rho=hess_rho_zz/aa**2+hess_rho_rr/bb**2
    gradient_rho_squared=drho["z"]**2/aa**2+drho["r"]**2/bb**2
    ricci_zz=.5*base_scalar*aa**2-2*hzz_over_rho
    ricci_rr=.5*base_scalar*bb**2-2*hrr_over_rho
    ricci_zr=-2*hzr_over_rho
    fiber_numerator=1-rho*laplacian_rho-gradient_rho_squared
    ricci_transverse=np.divide(
        fiber_numerator,r[None,:]**2,out=np.zeros_like(fiber_numerator),
        where=r[None,:]!=0,
    )
    ricci_transverse[:,0]=ricci_rr[:,0]
    if _well_balanced:
        baseline=axisymmetric_diagonal_geometry(
            z,r,psi,np.zeros_like(psi),np.zeros_like(psi),np.zeros_like(psi),
            stencil_width,_well_balanced=False,
        )
        w=_differentiate(np.log(psi),z,r,stencil_width)
        laplacian=w["zz"]+w["rr"]+2*w["transverse_hessian"]
        gradient_squared=w["z"]**2+w["r"]**2
        common=laplacian+2*gradient_squared
        exact={
            "zz":-2*(w["zz"]-w["z"]**2)-common,
            "radial":-2*(w["rr"]-w["r"]**2)-common,
            "transverse":-2*w["transverse_hessian"]-common,
            "zr":-2*(w["zr"]-w["z"]*w["r"]),
        }
        ricci_zz+=exact["zz"]-baseline["ricci_zz"]
        ricci_rr+=exact["radial"]-baseline["ricci_radial"]
        ricci_transverse+=exact["transverse"]-baseline["ricci_transverse"]
        ricci_zr+=exact["zr"]-baseline["ricci_zr"]
    scalar_curvature=(
        ricci_zz/aa**2+ricci_rr/bb**2
        +2*ricci_transverse/cc**2
    )
    return {
        "metric_zz":aa**2,"metric_radial":bb**2,"metric_transverse":cc**2,
        "A":aa,"B":bb,"C":cc,"rho":rho,
        "ricci_zz":ricci_zz,"ricci_radial":ricci_rr,
        "ricci_transverse":ricci_transverse,"ricci_zr":ricci_zr,
        "scalar_curvature":scalar_curvature,"base_scalar_curvature":base_scalar,
        "rho_laplacian":laplacian_rho,"rho_gradient_squared":gradient_rho_squared,
        "derivatives":{"A":da,"B":db,"C":dc,"rho":drho},
    }


def anisotropic_hamiltonian_residual(
    z,r,psi,a,b,c,phi,chi_r,chi_z,m_phi_squared,
    m_chi_squared=0.,chi=None,cosmological_constant=-6.,kappa5_squared=1.,
    stencil_width=7,
):
    """Evaluate ``R-2 Lambda-kappa^2[(D f)^2+m^2 f^2]``."""
    geometry=axisymmetric_diagonal_geometry(z,r,psi,a,b,c,stencil_width)
    phi=np.asarray(phi,dtype=float);chi_r=np.asarray(chi_r,dtype=float)
    chi_z=np.asarray(chi_z,dtype=float)
    dphi=_differentiate(phi,z,r,stencil_width)
    gradient=(dphi["z"]**2+chi_z**2)/geometry["metric_zz"]
    gradient+=(dphi["r"]**2+chi_r**2)/geometry["metric_radial"]
    potential=float(m_phi_squared)*phi**2
    if float(m_chi_squared)!=0:
        if chi is None:raise ValueError("chi values required for nonzero chi mass")
        potential+=float(m_chi_squared)*np.asarray(chi,dtype=float)**2
    return (
        geometry["scalar_curvature"]-2*float(cosmological_constant)
        -float(kappa5_squared)*(gradient+potential)
    )


def anisotropic_scalar_gradient_energy(z,r,psi,a,b,c,scalar_r,scalar_z):
    """Return the time-symmetric massless-scalar gradient energy.

    For ``gamma=A^2 dz^2+B^2 dr^2+(Cr)^2 dOmega_2^2``, integration over the
    two-sphere gives

    ``E=2 pi integral r^2 C^2[(B/A) scalar_z^2+(A/B) scalar_r^2] dz dr``.
    """
    z=np.asarray(z,dtype=float);r=np.asarray(r,dtype=float)
    psi=np.asarray(psi,dtype=float);a=np.asarray(a,dtype=float)
    b=np.asarray(b,dtype=float);c=np.asarray(c,dtype=float)
    scalar_r=np.asarray(scalar_r,dtype=float);scalar_z=np.asarray(scalar_z,dtype=float)
    if not all(field.shape==psi.shape for field in (a,b,c,scalar_r,scalar_z)):
        raise ValueError("all energy fields must share the z-r grid shape")
    A=psi*np.exp(a);B=psi*np.exp(b);C=psi*np.exp(c)
    density=2*np.pi*r[None,:]**2*C**2*(B/A*scalar_z**2+A/B*scalar_r**2)
    return float(simpson(simpson(density,x=r,axis=1),x=z))


def anisotropic_metric_acceleration(
    z,r,psi,a,b,c,phi,chi_r,chi_z,m_phi_squared,
    m_chi_squared=0.,chi=None,cosmological_constant=-6.,kappa5_squared=1.,
    stencil_width=7,lapse=None,
):
    """Return ``partial_t^2 gamma_ij`` at ``K_ij=0`` and zero shift."""
    geometry=axisymmetric_diagonal_geometry(z,r,psi,a,b,c,stencil_width)
    psi=np.asarray(psi,dtype=float);phi=np.asarray(phi,dtype=float)
    alpha=psi if lapse is None else np.asarray(lapse,dtype=float)
    if alpha.shape!=psi.shape or np.any(alpha<=0):raise ValueError("invalid lapse")
    dalpha=_differentiate(alpha,z,r,stencil_width)
    dphi=_differentiate(phi,z,r,stencil_width)
    da,db,drho=(geometry["derivatives"][name] for name in ("A","B","rho"))
    aa,bb,rho=geometry["A"],geometry["B"],geometry["rho"]
    gamma_z_zz=da["z"]/aa;gamma_z_zr=da["r"]/aa
    gamma_z_rr=-bb*db["z"]/aa**2
    gamma_r_zz=-aa*da["r"]/bb**2
    gamma_r_zr=db["z"]/bb;gamma_r_rr=db["r"]/bb
    hess_zz=dalpha["zz"]-gamma_z_zz*dalpha["z"]-gamma_r_zz*dalpha["r"]
    hess_rr=dalpha["rr"]-gamma_z_rr*dalpha["z"]-gamma_r_rr*dalpha["r"]
    hess_zr=dalpha["zr"]-gamma_z_zr*dalpha["z"]-gamma_r_zr*dalpha["r"]
    hess_transverse=np.divide(
        rho*(drho["z"]*dalpha["z"]/aa**2+drho["r"]*dalpha["r"]/bb**2),
        np.asarray(r)[None,:]**2,out=np.zeros_like(rho),where=np.asarray(r)[None,:]!=0,
    )
    hess_transverse[:,0]=hess_rr[:,0]
    # The same well-balancing used for Ricci is required for the lapse
    # Hessian because discrete derivatives do not obey an exact product rule.
    base_a=base_b=psi;base_rho=psi*np.asarray(r)[None,:]
    dbase=_differentiate(psi,z,r,stencil_width);dbase_rho=_differentiate(base_rho,z,r,stencil_width)
    base_gz_zz=dbase["z"]/psi;base_gz_zr=dbase["r"]/psi
    base_gz_rr=-dbase["z"]/psi;base_gr_zz=-dbase["r"]/psi
    base_gr_zr=dbase["z"]/psi;base_gr_rr=dbase["r"]/psi
    raw_base_zz=dalpha["zz"]-base_gz_zz*dalpha["z"]-base_gr_zz*dalpha["r"]
    raw_base_rr=dalpha["rr"]-base_gz_rr*dalpha["z"]-base_gr_rr*dalpha["r"]
    raw_base_zr=dalpha["zr"]-base_gz_zr*dalpha["z"]-base_gr_zr*dalpha["r"]
    raw_base_transverse=np.divide(
        base_rho*(dbase_rho["z"]*dalpha["z"]+dbase_rho["r"]*dalpha["r"])/psi**2,
        np.asarray(r)[None,:]**2,out=np.zeros_like(base_rho),where=np.asarray(r)[None,:]!=0,
    )
    raw_base_transverse[:,0]=raw_base_rr[:,0]
    w=_differentiate(np.log(psi),z,r,stencil_width)
    w_dot_a=w["z"]*dalpha["z"]+w["r"]*dalpha["r"]
    exact_zz=dalpha["zz"]-2*w["z"]*dalpha["z"]+w_dot_a
    exact_rr=dalpha["rr"]-2*w["r"]*dalpha["r"]+w_dot_a
    exact_transverse=dalpha["transverse_hessian"]+w_dot_a
    exact_zr=dalpha["zr"]-w["z"]*dalpha["r"]-w["r"]*dalpha["z"]
    hess_zz+=exact_zz-raw_base_zz;hess_rr+=exact_rr-raw_base_rr
    hess_transverse+=exact_transverse-raw_base_transverse;hess_zr+=exact_zr-raw_base_zr
    potential=float(m_phi_squared)*phi**2
    if float(m_chi_squared)!=0:
        if chi is None:raise ValueError("chi values required for nonzero chi mass")
        potential+=float(m_chi_squared)*np.asarray(chi,dtype=float)**2
    isotropic_factor=(
        2*float(kappa5_squared)*potential/3+4*float(cosmological_constant)/3
    )*alpha**2
    metric={
        "zz":geometry["metric_zz"],"radial":geometry["metric_radial"],
        "transverse":geometry["metric_transverse"],
    }
    hessian={"zz":hess_zz,"radial":hess_rr,"transverse":hess_transverse,"zr":hess_zr}
    ricci={
        "zz":geometry["ricci_zz"],"radial":geometry["ricci_radial"],
        "transverse":geometry["ricci_transverse"],"zr":geometry["ricci_zr"],
    }
    gradients={
        "zz":dphi["z"]**2+np.asarray(chi_z)**2,
        "radial":dphi["r"]**2+np.asarray(chi_r)**2,
        "transverse":np.zeros_like(psi),
        "zr":dphi["z"]*dphi["r"]+np.asarray(chi_z)*np.asarray(chi_r),
    }
    result={}
    for name in ("zz","radial","transverse"):
        result[name]=(
            2*alpha*hessian[name]-2*alpha**2*ricci[name]
            +2*alpha**2*float(kappa5_squared)*gradients[name]
            +isotropic_factor*metric[name]
        )
    result["zr"]=(
        2*alpha*hessian["zr"]-2*alpha**2*ricci["zr"]
        +2*alpha**2*float(kappa5_squared)*gradients["zr"]
    )
    # This radial-Cartesian mixed component is odd under axis reflection.
    result["zr"][:,0]=0.
    result["Dz"]=_differentiate(np.log(psi),z,r,stencil_width)["Dz"]
    result["geometry"]=geometry;result["lapse"]=alpha
    return result


def anisotropic_scalar_acceleration(
    z,r,psi,a,b,c,scalar,mass_squared,lapse=None,stencil_width=7,
):
    """Scalar acceleration for the diagonal geometry at zero momentum."""
    geometry=axisymmetric_diagonal_geometry(z,r,psi,a,b,c,stencil_width)
    scalar=np.asarray(scalar,dtype=float);alpha=psi if lapse is None else np.asarray(lapse,dtype=float)
    df=_differentiate(scalar,z,r,stencil_width);dalpha=_differentiate(alpha,z,r,stencil_width)
    aa,bb,rho=geometry["A"],geometry["B"],geometry["rho"]
    da,db,drho=(geometry["derivatives"][name] for name in ("A","B","rho"))
    gamma_z_zz=da["z"]/aa;gamma_z_rr=-bb*db["z"]/aa**2
    gamma_r_zz=-aa*da["r"]/bb**2;gamma_r_rr=db["r"]/bb
    hzz=df["zz"]-gamma_z_zz*df["z"]-gamma_r_zz*df["r"]
    hrr=df["rr"]-gamma_z_rr*df["z"]-gamma_r_rr*df["r"]
    rho_term=np.divide(
        2*(drho["z"]*df["z"]/aa**2+drho["r"]*df["r"]/bb**2),
        rho,out=np.zeros_like(rho),where=rho!=0,
    )
    rho_term[:,0]=2*hrr[:,0]/bb[:,0]**2
    laplacian=hzz/aa**2+hrr/bb**2+rho_term
    # Well-balance against the established conformal scalar Laplacian.
    dpsi=_differentiate(psi,z,r,stencil_width);base_rho=psi*np.asarray(r)[None,:]
    dbase_rho=_differentiate(base_rho,z,r,stencil_width)
    base_hzz=df["zz"]-dpsi["z"]*df["z"]/psi+dpsi["r"]*df["r"]/psi
    base_hrr=df["rr"]+dpsi["z"]*df["z"]/psi-dpsi["r"]*df["r"]/psi
    base_rho_term=np.divide(
        2*(dbase_rho["z"]*df["z"]+dbase_rho["r"]*df["r"])/psi**2,
        base_rho,out=np.zeros_like(base_rho),where=base_rho!=0,
    )
    base_rho_term[:,0]=2*base_hrr[:,0]/psi[:,0]**2
    raw_base=base_hzz/psi**2+base_hrr/psi**2+base_rho_term
    w=_differentiate(np.log(psi),z,r,stencil_width)
    exact_base=(df["zz"]+df["rr"]+2*df["transverse_hessian"]+2*(w["z"]*df["z"]+w["r"]*df["r"]))/psi**2
    laplacian+=exact_base-raw_base
    lapse_gradient=dalpha["z"]*df["z"]/aa**2+dalpha["r"]*df["r"]/bb**2
    return alpha**2*(laplacian-float(mass_squared)*scalar)+alpha*lapse_gradient


def anisotropic_spatial_israel_second_corner_fields(
    acceleration,psi,a,b,c,phi,background,scalar_acceleration=None,
    radial_buffer=7,
):
    """Return differentiated spatial Israel rows for a diagonal metric."""
    psi=np.asarray(psi,dtype=float);a=np.asarray(a,dtype=float)
    b=np.asarray(b,dtype=float);c=np.asarray(c,dtype=float);phi=np.asarray(phi,dtype=float)
    scalar_acceleration=np.zeros_like(psi) if scalar_acceleration is None else np.asarray(scalar_acceleration,dtype=float)
    dz=acceleration["Dz"];buffer=int(radial_buffer)
    radial_slice=slice(None,-buffer) if buffer else slice(None)
    sqrt_zz=psi*np.exp(a);metric_radial=psi**2*np.exp(2*b);metric_transverse=psi**2*np.exp(2*c)
    gamma=float(background["wall_stiffness"]);walls=[]
    for wall_index,index,target in ((0,0,float(background["v0"])),(1,-1,float(background["v1"]))):
        wall_potential=.5*gamma*(phi[index]-target)**2
        if wall_index==0:
            beta=float(background["beta_a"])+(wall_potential-float(background["wall_potential_a"]))/6
            beta_phi=gamma*(phi[index]-target)/6
        else:
            beta=float(background["beta_b"])-(wall_potential-float(background["wall_potential_b"]))/6
            beta_phi=-gamma*(phi[index]-target)/6
        components={}
        for name,metric_component in (("radial",metric_radial),("transverse",metric_transverse)):
            value=np.asarray(acceleration[name])[index]
            terms=(
                (dz@np.asarray(acceleration[name]))[index],
                2*beta*sqrt_zz[index]*value,
                beta*metric_component[index]*np.asarray(acceleration["zz"])[index]/sqrt_zz[index],
                2*beta_phi*scalar_acceleration[index]*sqrt_zz[index]*metric_component[index],
            )
            residual=sum(terms);scale=np.maximum(1.,sum(np.abs(term) for term in terms))
            components[name]={"residual":residual[radial_slice],"scale":scale[radial_slice]}
        mixed=np.asarray(acceleration["zr"])[index,radial_slice]
        mixed_scale=np.maximum(
            1.,
            np.abs(np.asarray(acceleration["zz"])[index,radial_slice])
            +np.abs(np.asarray(acceleration["radial"])[index,radial_slice])
            +np.abs(np.asarray(acceleration["transverse"])[index,radial_slice]),
        )
        walls.append({
            "wall":"lower" if index==0 else "upper",
            "tangential_components":components,
            "mixed_zr_residual":np.asarray(mixed),
            "mixed_zr_scale":np.asarray(mixed_scale),
        })
    return {"walls":walls,"radial_buffer":buffer}


def anisotropic_spatial_junction_fields(z,r,psi,a,b,c,phi,background,stencil_width=7):
    """Return zeroth-order radial and transverse Israel residual arrays."""
    psi=np.asarray(psi,dtype=float);a=np.asarray(a,dtype=float)
    b=np.asarray(b,dtype=float);c=np.asarray(c,dtype=float);phi=np.asarray(phi,dtype=float)
    dz=_differentiate(np.log(psi),z,r,stencil_width)["Dz"]
    log_psi=np.log(psi);sqrt_zz=psi*np.exp(a);gamma=float(background["wall_stiffness"])
    walls=[]
    for wall_index,index,target in ((0,0,float(background["v0"])),(1,-1,float(background["v1"]))):
        wall_potential=.5*gamma*(phi[index]-target)**2
        beta=(
            float(background["beta_a"])+(wall_potential-float(background["wall_potential_a"]))/6
            if wall_index==0 else
            float(background["beta_b"])-(wall_potential-float(background["wall_potential_b"]))/6
        )
        radial=(dz@(log_psi+b))[index]+beta*sqrt_zz[index]
        transverse=(dz@(log_psi+c))[index]+beta*sqrt_zz[index]
        walls.append({
            "wall":"lower" if index==0 else "upper",
            "radial":radial,"transverse":transverse,
        })
    return {"walls":walls}
