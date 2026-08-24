"""Well-balanced metric constraint with equilibrium Goldberger--Wise data.

This constructs a valid time-symmetric initial slice with ``Phi`` set to its
stabilized background profile and zero scalar momentum.  The metric is free to
respond through ``q=1/psi-z``.  It is not a quasi-static solve for the
localized perturbation of ``Phi``.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from bhps.gw_background import solve_gw_background
from bhps.initial_data import make_grid
from bhps.scalar_pulse import scalar_pulse


def _raw_residual(q,z,r,background,chi_r,chi_z):
    q=np.asarray(q).reshape(len(z),len(r));nz,nr=q.shape
    dz,dr=z[1]-z[0],r[1]-r[0];out=np.zeros_like(q);s=z[:,None]+q
    gradient=(background["phi_z"][:,None]**2+chi_r**2+chi_z**2)/6
    potential=background["mass_squared"]*background["phi"][:,None]**2/6
    qzz=(q[2:,1:-1]-2*q[1:-1,1:-1]+q[:-2,1:-1])/dz**2
    qrr=(q[1:-1,2:]-2*q[1:-1,1:-1]+q[1:-1,:-2])/dr**2
    qz=(q[2:,1:-1]-q[:-2,1:-1])/(2*dz)
    qr=(q[1:-1,2:]-q[1:-1,:-2])/(2*dr)
    lap=qzz+qrr+2*qr/r[None,1:-1];local=s[1:-1,1:-1]
    out[1:-1,1:-1]=-local*lap+4*qz+2*(qz**2+qr**2)+gradient[1:-1,1:-1]*local**2+potential[1:-1]
    qzz_axis=(q[2:,0]-2*q[1:-1,0]+q[:-2,0])/dz**2
    radial_axis=6*(q[1:-1,1]-q[1:-1,0])/dr**2
    qz_axis=(q[2:,0]-q[:-2,0])/(2*dz);local_axis=s[1:-1,0]
    out[1:-1,0]=-local_axis*(qzz_axis+radial_axis)+4*qz_axis+2*qz_axis**2+gradient[1:-1,0]*local_axis**2+potential[1:-1,0]
    out[0,:-1]=(-3*q[0,:-1]+4*q[1,:-1]-q[2,:-1])/(2*dz)-(background["beta_a"]-1)
    out[-1,:-1]=(3*q[-1,:-1]-4*q[-2,:-1]+q[-3,:-1])/(2*dz)-(background["beta_b"]-1)
    background_q=1/background["psi"]-z
    delta=q-background_q[:,None]
    out[:,-1]=(3*q[:,-1]-4*q[:,-2]+q[:,-3])/(2*dr)+delta[:,-1]/r[-1]
    return out


def gw_slice_residual(q,z,r,background,chi_r,chi_z):
    q=np.asarray(q).reshape(len(z),len(r));raw=_raw_residual(q,z,r,background,chi_r,chi_z)
    background_q=np.repeat((1/background["psi"]-z)[:,None],len(r),axis=1)
    zeros=np.zeros_like(q);defect=_raw_residual(background_q,z,r,background,zeros,zeros)
    return (raw-defect).ravel()


def gw_slice_jacobian(q,z,r,background,chi_r,chi_z):
    q=np.asarray(q).reshape(len(z),len(r));nz,nr=q.shape
    dz,dr=z[1]-z[0],r[1]-r[0]
    gradient=(background["phi_z"][:,None]**2+chi_r**2+chi_z**2)/6;s=z[:,None]+q
    matrix=lil_matrix((nz*nr,nz*nr));index=lambda i,j:i*nr+j
    for i in range(nz):
        for j in range(nr):
            row=index(i,j)
            if j==nr-1:
                matrix[row,index(i,j-2)]=1/(2*dr)
                matrix[row,index(i,j-1)]=-2/dr
                matrix[row,row]=3/(2*dr)+1/r[-1]
            elif i==0:
                matrix[row,index(0,j)]=-3/(2*dz);matrix[row,index(1,j)]=2/dz;matrix[row,index(2,j)]=-1/(2*dz)
            elif i==nz-1:
                matrix[row,index(i,j)]=3/(2*dz);matrix[row,index(i-1,j)]=-2/dz;matrix[row,index(i-2,j)]=1/(2*dz)
            else:
                local=s[i,j];qz=(q[i+1,j]-q[i-1,j])/(2*dz)
                if j==0:
                    lap=(q[i+1,0]-2*q[i,0]+q[i-1,0])/dz**2+6*(q[i,1]-q[i,0])/dr**2
                    matrix[row,index(i-1,0)]=-local/dz**2-(2+2*qz)/dz
                    matrix[row,index(i+1,0)]=-local/dz**2+(2+2*qz)/dz
                    matrix[row,index(i,1)]=-6*local/dr**2
                    matrix[row,row]=local*(2/dz**2+6/dr**2)-lap+2*gradient[i,0]*local
                else:
                    qr=(q[i,j+1]-q[i,j-1])/(2*dr)
                    lap=(q[i+1,j]-2*q[i,j]+q[i-1,j])/dz**2+(q[i,j+1]-2*q[i,j]+q[i,j-1])/dr**2+(q[i,j+1]-q[i,j-1])/(r[j]*dr)
                    minus=1/dr**2-1/(r[j]*dr);plus=1/dr**2+1/(r[j]*dr)
                    matrix[row,index(i-1,j)]=-local/dz**2-(2+2*qz)/dz
                    matrix[row,index(i+1,j)]=-local/dz**2+(2+2*qz)/dz
                    matrix[row,index(i,j-1)]=-local*minus-2*qr/dr
                    matrix[row,index(i,j+1)]=-local*plus+2*qr/dr
                    matrix[row,row]=local*(2/dz**2+2/dr**2)-lap+2*gradient[i,j]*local
    return matrix.tocsr()


def solve_gw_slice(
    amplitude,sigma_r=1.,sigma_y=.2,center_fraction=.9,d_over_ell=1.,r_max=8.,
    nz=25,nr=37,epsilon=.1,backreaction=.01,tolerance=1e-9,iterations=100,initial=None,
):
    z,r=make_grid(d_over_ell,r_max,nz,nr);background=solve_gw_background(z,epsilon,backreaction)
    if not background["converged"]:
        raise RuntimeError(background["message"])
    _,chi_r,chi_z=scalar_pulse(z,r,amplitude,sigma_r,sigma_y,center_fraction,d_over_ell)
    background_q=np.repeat((1/background["psi"]-z)[:,None],nr,axis=1)
    q=background_q.copy() if initial is None else np.asarray(initial).copy();history=[]
    for _ in range(iterations):
        residual=gw_slice_residual(q,z,r,background,chi_r,chi_z)
        norm=float(np.max(np.abs(residual)));history.append(norm)
        if norm<tolerance:break
        step=spsolve(gw_slice_jacobian(q,z,r,background,chi_r,chi_z),-residual).reshape(q.shape)
        damping=1.;accepted=False
        while damping>=2**-16:
            candidate=q+damping*step
            if np.min(z[:,None]+candidate)>0 and np.max(np.abs(gw_slice_residual(candidate,z,r,background,chi_r,chi_z)))<norm:
                q=candidate;accepted=True;break
            damping*=.5
        if not accepted:break
    final_grid=gw_slice_residual(q,z,r,background,chi_r,chi_z).reshape(nz,nr)
    final=float(np.max(np.abs(final_grid)))
    psi=1/(z[:,None]+q);base=np.repeat(background["psi"][:,None],nr,axis=1)
    density=2*np.pi*r[None,:]**2*psi**2*(chi_r**2+chi_z**2)
    energy_simpson=float(simpson(simpson(density,x=r,axis=1),x=z))
    energy_trapezoid=float(np.trapezoid(np.trapezoid(density,x=r,axis=1),x=z))
    return {
        "converged":final<tolerance,"z":z,"r":r,"q":q,"psi":psi,"background":background,
        "energy_dimensionless":energy_simpson,"energy_dimensionless_trapezoid":energy_trapezoid,
        "energy_quadrature_relative_difference":abs(energy_simpson-energy_trapezoid)/max(abs(energy_simpson),1e-300),
        "max_abs_residual":final,"min_psi":float(np.min(psi)),
        "residual_l2":float(np.sqrt(np.mean(final_grid**2))),
        "bulk_residual_max":float(np.max(np.abs(final_grid[1:-1,:-1]))),
        "junction_residual_max":float(max(np.max(np.abs(final_grid[0,:-1])),np.max(np.abs(final_grid[-1,:-1])))),
        "outer_boundary_residual_max":float(np.max(np.abs(final_grid[:,-1]))),
        "max_relative_deformation":float(np.max(np.abs(psi/base-1))),"history":history,
        "initial_data_interpretation":"Phi on equilibrium background with zero momentum; no quasi-static delta-Phi solve",
    }
