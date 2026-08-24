"""Independent equilibrium-GW constraint solver in ``u=log(psi/psi_GW)``.

Unlike :mod:`bhps.gw_slice_solver`, this formulation never introduces the
inverse-conformal variable ``q``.  Its nonlinear bulk and brane equations are
obtained after analytically subtracting the one-dimensional GW background.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from bhps.gw_background import solve_gw_background
from bhps.initial_data import make_grid
from bhps.scalar_pulse import scalar_pulse


def gw_log_residual(u,z,r,background,chi_r,chi_z):
    u=np.asarray(u).reshape(len(z),len(r));nz,nr=u.shape
    dz,dr=z[1]-z[0],r[1]-r[0];out=np.zeros_like(u)
    warp=background["psi"];warp_log_z=background["psi_z"]/warp
    potential_coefficient=warp**2*(-2+background["mass_squared"]*background["phi"]**2/6)
    collapse_gradient=(chi_r**2+chi_z**2)/6
    uzz=(u[2:,1:-1]-2*u[1:-1,1:-1]+u[:-2,1:-1])/dz**2
    urr=(u[1:-1,2:]-2*u[1:-1,1:-1]+u[1:-1,:-2])/dr**2
    uz=(u[2:,1:-1]-u[:-2,1:-1])/(2*dz)
    ur=(u[1:-1,2:]-u[1:-1,:-2])/(2*dr)
    out[1:-1,1:-1]=(
        uzz+urr+2*ur/r[None,1:-1]+uz**2+ur**2
        +2*warp_log_z[1:-1,None]*uz+collapse_gradient[1:-1,1:-1]
        +potential_coefficient[1:-1,None]*np.expm1(2*u[1:-1,1:-1])
    )
    uzz_axis=(u[2:,0]-2*u[1:-1,0]+u[:-2,0])/dz**2
    radial_axis=6*(u[1:-1,1]-u[1:-1,0])/dr**2
    uz_axis=(u[2:,0]-u[:-2,0])/(2*dz)
    out[1:-1,0]=(
        uzz_axis+radial_axis+uz_axis**2+2*warp_log_z[1:-1]*uz_axis
        +collapse_gradient[1:-1,0]+potential_coefficient[1:-1]*np.expm1(2*u[1:-1,0])
    )
    out[0,:-1]=(
        (-3*u[0,:-1]+4*u[1,:-1]-u[2,:-1])/(2*dz)
        +background["beta_a"]*warp[0]*np.expm1(u[0,:-1])
    )
    out[-1,:-1]=(
        (3*u[-1,:-1]-4*u[-2,:-1]+u[-3,:-1])/(2*dz)
        +background["beta_b"]*warp[-1]*np.expm1(u[-1,:-1])
    )
    # Exact transform of delta-q_r+delta-q/r=0 under
    # delta q=(exp(-u)-1)/psi_GW.
    out[:,-1]=(3*u[:,-1]-4*u[:,-2]+u[:,-3])/(2*dr)+np.expm1(u[:,-1])/r[-1]
    return out.ravel()


def gw_log_jacobian(u,z,r,background,chi_r,chi_z):
    u=np.asarray(u).reshape(len(z),len(r));nz,nr=u.shape
    dz,dr=z[1]-z[0],r[1]-r[0]
    warp=background["psi"];warp_log_z=background["psi_z"]/warp
    coefficient=warp**2*(-2+background["mass_squared"]*background["phi"]**2/6)
    matrix=lil_matrix((nz*nr,nz*nr));index=lambda i,j:i*nr+j
    for i in range(nz):
        for j in range(nr):
            row=index(i,j)
            if j==nr-1:
                matrix[row,index(i,j-2)]=1/(2*dr);matrix[row,index(i,j-1)]=-2/dr
                matrix[row,row]=3/(2*dr)+np.exp(u[i,j])/r[-1]
            elif i==0:
                matrix[row,index(0,j)]=-3/(2*dz)+background["beta_a"]*warp[0]*np.exp(u[0,j])
                matrix[row,index(1,j)]=2/dz;matrix[row,index(2,j)]=-1/(2*dz)
            elif i==nz-1:
                matrix[row,index(i,j)]=3/(2*dz)+background["beta_b"]*warp[-1]*np.exp(u[-1,j])
                matrix[row,index(i-1,j)]=-2/dz;matrix[row,index(i-2,j)]=1/(2*dz)
            else:
                uz=(u[i+1,j]-u[i-1,j])/(2*dz);z_advection=warp_log_z[i]+uz
                matrix[row,index(i-1,j)]=1/dz**2-z_advection/dz
                matrix[row,index(i+1,j)]=1/dz**2+z_advection/dz
                if j==0:
                    matrix[row,index(i,1)]=6/dr**2
                    matrix[row,row]=-2/dz**2-6/dr**2+2*coefficient[i]*np.exp(2*u[i,0])
                else:
                    ur=(u[i,j+1]-u[i,j-1])/(2*dr)
                    matrix[row,index(i,j-1)]=1/dr**2-1/(r[j]*dr)-ur/dr
                    matrix[row,index(i,j+1)]=1/dr**2+1/(r[j]*dr)+ur/dr
                    matrix[row,row]=-2/dz**2-2/dr**2+2*coefficient[i]*np.exp(2*u[i,j])
    return matrix.tocsr()


def solve_gw_log_slice(
    amplitude,sigma_r=1.,sigma_y=.2,center_fraction=.9,d_over_ell=1.,r_max=8.,
    nz=25,nr=37,epsilon=.1,backreaction=.01,tolerance=1e-9,iterations=100,initial=None,
):
    z,r=make_grid(d_over_ell,r_max,nz,nr);background=solve_gw_background(z,epsilon,backreaction)
    if not background["converged"]:raise RuntimeError(background["message"])
    _,chi_r,chi_z=scalar_pulse(z,r,amplitude,sigma_r,sigma_y,center_fraction,d_over_ell)
    u=np.zeros((nz,nr)) if initial is None else np.asarray(initial).copy();history=[]
    for _ in range(iterations):
        residual=gw_log_residual(u,z,r,background,chi_r,chi_z)
        norm=float(np.max(np.abs(residual)));history.append(norm)
        if norm<tolerance:break
        step=spsolve(gw_log_jacobian(u,z,r,background,chi_r,chi_z),-residual).reshape(u.shape)
        damping=1.;accepted=False
        while damping>=2**-16:
            candidate=u+damping*step
            if np.max(np.abs(candidate))<50 and np.max(np.abs(gw_log_residual(candidate,z,r,background,chi_r,chi_z)))<norm:
                u=candidate;accepted=True;break
            damping*=.5
        if not accepted:break
    final=float(np.max(np.abs(gw_log_residual(u,z,r,background,chi_r,chi_z))))
    psi=background["psi"][:,None]*np.exp(u)
    density=2*np.pi*r[None,:]**2*psi**2*(chi_r**2+chi_z**2)
    energy_simpson=float(simpson(simpson(density,x=r,axis=1),x=z))
    energy_trapezoid=float(np.trapezoid(np.trapezoid(density,x=r,axis=1),x=z))
    return {
        "converged":final<tolerance,"z":z,"r":r,"u":u,"psi":psi,"background":background,
        "energy_dimensionless":energy_simpson,"energy_dimensionless_trapezoid":energy_trapezoid,
        "energy_quadrature_relative_difference":abs(energy_simpson-energy_trapezoid)/max(abs(energy_simpson),1e-300),
        "max_abs_residual":final,"min_psi":float(np.min(psi)),
        "max_relative_deformation":float(np.max(np.abs(np.expm1(u)))),"history":history,
        "outer_boundary":"u_r+(exp(u)-1)/r=0; exact transform of the q asymptotic condition",
        "initial_data_interpretation":"Phi on equilibrium background with zero momentum; independent log-conformal constraint formulation",
    }
