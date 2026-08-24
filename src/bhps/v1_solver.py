"""Production-control Hamiltonian solver with a smooth bulk scalar pulse."""

from __future__ import annotations
import numpy as np
from scipy.integrate import simpson
from scipy.sparse.linalg import spsolve
from bhps.initial_data import make_grid,jacobian as vacuum_jacobian
from bhps.scalar_pulse import scalar_pulse


def v1_residual(psi,z,r,chi_r,chi_z,mchi=0.,outer_boundary="dirichlet"):
    from bhps.initial_data import residual as vacuum_residual
    psi=np.asarray(psi).reshape(len(z),len(r)); source_grad=(chi_r**2+chi_z**2)/6
    source_mass=mchi**2*(chi_r*0+1)/6  # coefficient populated below from reconstructed chi when nonzero
    out=vacuum_residual(psi,z,r,0,.75).reshape(psi.shape)
    out[1:-1,:-1]+=source_grad[1:-1,:-1]*psi[1:-1,:-1]
    if outer_boundary=="asymptotic_radion":
        background=np.repeat((1/z)[:,None],len(r),axis=1)
        delta=psi-background;dr=r[1]-r[0]
        out[:,-1]=(3*delta[:,-1]-4*delta[:,-2]+delta[:,-3])/(2*dr)+delta[:,-1]/r[-1]
    elif outer_boundary!="dirichlet":
        raise ValueError("outer_boundary must be 'dirichlet' or 'asymptotic_radion'")
    # V1 production begins massless. Refuse a silent incorrect massive solve.
    if mchi!=0: raise NotImplementedError("massive chi requires chi values in residual")
    return out.ravel()


def v1_jacobian(psi,z,r,chi_r,chi_z,outer_boundary="dirichlet"):
    """Exact discrete Jacobian for the massless V1 Hamiltonian residual."""
    psi=np.asarray(psi).reshape(len(z),len(r));nz,nr=psi.shape
    source=(chi_r**2+chi_z**2)/6
    matrix=vacuum_jacobian(psi,z,r,0,.75).tolil()
    for i in range(1,nz-1):
        for j in range(nr-1):matrix[i*nr+j,i*nr+j]+=source[i,j]
    if outer_boundary=="asymptotic_radion":
        dr=r[1]-r[0]
        for i in range(nz):
            row=i*nr+nr-1
            matrix.rows[row]=[row-2,row-1,row]
            matrix.data[row]=[1/(2*dr),-2/dr,3/(2*dr)+1/r[-1]]
    elif outer_boundary!="dirichlet":
        raise ValueError("outer_boundary must be 'dirichlet' or 'asymptotic_radion'")
    return matrix.tocsr()


def solve_v1(amplitude,sigma_r=1.,sigma_y=.2,center_fraction=.9,d_over_ell=1.,r_max=8.,nz=25,nr=37,tolerance=1e-9,iterations=80,initial=None,outer_boundary="dirichlet"):
    z,r=make_grid(d_over_ell,r_max,nz,nr);chi,chi_r,chi_z=scalar_pulse(z,r,amplitude,sigma_r,sigma_y,center_fraction,d_over_ell)
    background=np.repeat((1/z)[:,None],nr,axis=1);psi=background.copy() if initial is None else np.asarray(initial).copy();history=[]
    source=(chi_r**2+chi_z**2)/6
    for _ in range(iterations):
        f=v1_residual(psi,z,r,chi_r,chi_z,outer_boundary=outer_boundary);norm=float(np.max(np.abs(f)));history.append(norm)
        if norm<tolerance:break
        step=spsolve(v1_jacobian(psi,z,r,chi_r,chi_z,outer_boundary),-f).reshape(psi.shape);damping=1.;accepted=False
        while damping>=2**-16:
            candidate=psi+damping*step
            if np.min(candidate)>0 and np.max(np.abs(v1_residual(candidate,z,r,chi_r,chi_z,outer_boundary=outer_boundary)))<norm:psi=candidate;accepted=True;break
            damping*=.5
        if not accepted:break
    final=float(np.max(np.abs(v1_residual(psi,z,r,chi_r,chi_z,outer_boundary=outer_boundary))))
    energy_density=2*np.pi*r[None,:]**2*psi**2*(chi_r**2+chi_z**2)
    energy=float(simpson(simpson(energy_density,x=r,axis=1),x=z))
    return {"converged":final<tolerance,"z":z,"r":r,"psi":psi,"chi":chi,"energy_dimensionless":energy,"max_abs_residual":final,"min_psi":float(np.min(psi)),"max_relative_deformation":float(np.max(np.abs(psi/background-1))),"history":history}


def continue_v1(amplitudes,**kwargs):
    """Follow a single solution branch, stopping at the first rejected point."""
    previous=None;accepted=[];rejected=None
    for amplitude in amplitudes:
        solved=solve_v1(float(amplitude),initial=previous,**kwargs)
        record={key:value for key,value in solved.items() if key not in {"z","r","psi","chi","history"}}
        record["amplitude"]=float(amplitude);record["newton_iterations"]=len(solved["history"])
        if not solved["converged"]:
            rejected=record;break
        accepted.append(record);previous=solved["psi"]
    return {"accepted":accepted,"rejected":rejected}
