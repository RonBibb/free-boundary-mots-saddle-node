"""Hamiltonian solver on the weak-backreaction GW background control."""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from bhps.gw_background import solve_gw_background
from bhps.initial_data import make_grid
from bhps.scalar_pulse import scalar_pulse


def _raw_residual(psi,z,r,phi,phi_z,mphi2,beta_a,beta_b,chi_r,chi_z):
    psi=np.asarray(psi).reshape(len(z),len(r));nz,nr=psi.shape
    dz,dr=z[1]-z[0],r[1]-r[0]
    out=np.zeros_like(psi)
    scalar_gradient=(phi_z[:,None]**2+chi_r**2+chi_z**2)/6
    scalar_mass=mphi2*phi[:,None]**2/6
    out[1:-1,1:-1]=(psi[2:,1:-1]-2*psi[1:-1,1:-1]+psi[:-2,1:-1])/dz**2+(psi[1:-1,2:]-2*psi[1:-1,1:-1]+psi[1:-1,:-2])/dr**2+(psi[1:-1,2:]-psi[1:-1,:-2])/(r[None,1:-1]*dr)-2*psi[1:-1,1:-1]**3+scalar_gradient[1:-1,1:-1]*psi[1:-1,1:-1]+scalar_mass[1:-1]*psi[1:-1,1:-1]**3
    out[1:-1,0]=(psi[2:,0]-2*psi[1:-1,0]+psi[:-2,0])/dz**2+6*(psi[1:-1,1]-psi[1:-1,0])/dr**2-2*psi[1:-1,0]**3+scalar_gradient[1:-1,0]*psi[1:-1,0]+scalar_mass[1:-1,0]*psi[1:-1,0]**3
    out[0,:-1]=(-3*psi[0,:-1]+4*psi[1,:-1]-psi[2,:-1])/(2*dz)+beta_a*psi[0,:-1]**2
    out[-1,:-1]=(3*psi[-1,:-1]-4*psi[-2,:-1]+psi[-3,:-1])/(2*dz)+beta_b*psi[-1,:-1]**2
    return out


def stabilized_residual(psi,z,r,background,chi_r,chi_z):
    psi=np.asarray(psi).reshape(len(z),len(r))
    raw=_raw_residual(psi,z,r,background["phi"],background["phi_z"],background["mass_squared"],background["beta_a"],background["beta_b"],chi_r,chi_z)
    base=np.repeat(background["psi"][:,None],len(r),axis=1)
    zeros=np.zeros_like(base)
    defect=_raw_residual(base,z,r,background["phi"],background["phi_z"],background["mass_squared"],background["beta_a"],background["beta_b"],zeros,zeros)
    raw[:,-1]=psi[:,-1]-background["psi"]
    defect[:,-1]=0
    return (raw-defect).ravel()


def stabilized_jacobian(psi,z,r,background,chi_r,chi_z):
    psi=np.asarray(psi).reshape(len(z),len(r));nz,nr=psi.shape
    dz,dr=z[1]-z[0],r[1]-r[0]
    gradient=(background["phi_z"][:,None]**2+chi_r**2+chi_z**2)/6
    mass=background["mass_squared"]*background["phi"][:,None]**2/6
    matrix=lil_matrix((nz*nr,nz*nr));index=lambda i,j:i*nr+j
    for i in range(nz):
        for j in range(nr):
            row=index(i,j)
            if j==nr-1:
                matrix[row,row]=1
            elif i==0:
                matrix[row,index(0,j)]=-3/(2*dz)+2*background["beta_a"]*psi[0,j]
                matrix[row,index(1,j)]=2/dz
                matrix[row,index(2,j)]=-1/(2*dz)
            elif i==nz-1:
                matrix[row,index(i,j)]=3/(2*dz)+2*background["beta_b"]*psi[i,j]
                matrix[row,index(i-1,j)]=-2/dz
                matrix[row,index(i-2,j)]=1/(2*dz)
            elif j==0:
                matrix[row,index(i-1,0)]=1/dz**2
                matrix[row,index(i+1,0)]=1/dz**2
                matrix[row,index(i,1)]=6/dr**2
                matrix[row,row]=-2/dz**2-6/dr**2-6*psi[i,0]**2+gradient[i,0]+3*mass[i,0]*psi[i,0]**2
            else:
                matrix[row,index(i-1,j)]=matrix[row,index(i+1,j)]=1/dz**2
                matrix[row,index(i,j-1)]=1/dr**2-1/(r[j]*dr)
                matrix[row,index(i,j+1)]=1/dr**2+1/(r[j]*dr)
                matrix[row,row]=-2/dz**2-2/dr**2-6*psi[i,j]**2+gradient[i,j]+3*mass[i,0]*psi[i,j]**2
    return matrix.tocsr()


def solve_stabilized(
    amplitude,
    sigma_r=1.,
    sigma_y=.2,
    center_fraction=.9,
    d_over_ell=1.,
    r_max=8.,
    nz=25,
    nr=37,
    epsilon=.1,
    backreaction=.01,
    tolerance=1e-9,
    iterations=80,
    initial=None,
):
    z,r=make_grid(d_over_ell,r_max,nz,nr)
    background=solve_gw_background(z,epsilon,backreaction)
    if not background["converged"]:
        raise RuntimeError(background["message"])
    chi,chi_r,chi_z=scalar_pulse(z,r,amplitude,sigma_r,sigma_y,center_fraction,d_over_ell)
    base=np.repeat(background["psi"][:,None],nr,axis=1)
    psi=base.copy() if initial is None else np.asarray(initial).copy()
    history=[]
    for _ in range(iterations):
        residual=stabilized_residual(psi,z,r,background,chi_r,chi_z)
        norm=float(np.max(np.abs(residual)));history.append(norm)
        if norm<tolerance:
            break
        step=spsolve(stabilized_jacobian(psi,z,r,background,chi_r,chi_z),-residual).reshape(psi.shape)
        damping=1.;accepted=False
        while damping>=2**-16:
            candidate=psi+damping*step
            if np.min(candidate)>0 and np.max(np.abs(stabilized_residual(candidate,z,r,background,chi_r,chi_z)))<norm:
                psi=candidate;accepted=True;break
            damping*=.5
        if not accepted:
            break
    final=float(np.max(np.abs(stabilized_residual(psi,z,r,background,chi_r,chi_z))))
    density=2*np.pi*r[None,:]**2*psi**2*(chi_r**2+chi_z**2)
    energy=float(simpson(simpson(density,x=r,axis=1),x=z))
    return {
        "converged":final<tolerance,
        "z":z,
        "r":r,
        "psi":psi,
        "chi":chi,
        "background":background,
        "energy_dimensionless":energy,
        "max_abs_residual":final,
        "min_psi":float(np.min(psi)),
        "max_relative_deformation":float(np.max(np.abs(psi/base-1))),
        "history":history,
    }
