"""Independent high-order discretization of equilibrium-GW metric data.

The physical inverse-conformal variable is retained because it exactly
represents the light radion direction, but every first/second derivative is
formed from independently generated five-point polynomial weights rather than
the three-point stencils in :mod:`bhps.gw_slice_solver`.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.integrate import simpson
from scipy.sparse import csr_matrix,diags,eye,kron,lil_matrix
from scipy.sparse.linalg import spsolve

from bhps.gw_background import solve_gw_background
from bhps.initial_data import make_grid
from bhps.scalar_pulse import scalar_pulse


def derivative_matrix(x,order,width=5):
    """Polynomial finite-difference matrix on an arbitrary one-dimensional grid."""
    x=np.asarray(x,dtype=float);size=len(x)
    if width>size or width<=order:raise ValueError("invalid stencil width")
    matrix=lil_matrix((size,size))
    for index in range(size):
        start=min(max(index-width//2,0),size-width);indices=np.arange(start,start+width)
        offsets=x[indices]-x[index]
        vandermonde=np.vstack([offsets**power for power in range(width)])
        target=np.zeros(width);target[order]=math.factorial(order)
        weights=np.linalg.solve(vandermonde,target)
        matrix[index,indices]=weights
    return matrix.tocsr()


def _operators(z,r):
    dz=derivative_matrix(z,1);dzz=derivative_matrix(z,2)
    dr=derivative_matrix(r,1);drr=derivative_matrix(r,2)
    radial=lil_matrix(drr)
    radial[0,:]=3*drr.getrow(0)
    for j in range(1,len(r)):
        radial[j,:]=drr.getrow(j)+(2/r[j])*dr.getrow(j)
    identity_z=eye(len(z),format="csr");identity_r=eye(len(r),format="csr")
    return {
        "Dz":kron(dz,identity_r,format="csr"),
        "Dr":kron(identity_z,dr,format="csr"),
        "Lap":kron(dzz,identity_r,format="csr")+kron(identity_z,radial.tocsr(),format="csr"),
        "dz_1d":dz,"dr_1d":dr,
    }


def _raw(q,z,r,background,chi_r,chi_z,operators):
    q=np.asarray(q).reshape(len(z),len(r));flat=q.ravel();s=(z[:,None]+q).ravel()
    qz=operators["Dz"]@flat;qr=operators["Dr"]@flat;lap=operators["Lap"]@flat
    gradient=((background["phi_z"][:,None]**2+chi_r**2+chi_z**2)/6).ravel()
    potential=np.repeat(background["mass_squared"]*background["phi"]**2/6,len(r))
    out=(-s*lap+4*qz+2*(qz**2+qr**2)+gradient*s**2+potential).reshape(q.shape)
    qz_grid=qz.reshape(q.shape);qr_grid=qr.reshape(q.shape)
    out[0,:-1]=qz_grid[0,:-1]-(background["beta_a"]-1)
    out[-1,:-1]=qz_grid[-1,:-1]-(background["beta_b"]-1)
    background_q=1/background["psi"]-z;delta=q-background_q[:,None]
    out[:,-1]=qr_grid[:,-1]+delta[:,-1]/r[-1]
    return out.ravel()


def gw_high_order_residual(q,z,r,background,chi_r,chi_z,operators=None):
    operators=_operators(z,r) if operators is None else operators
    q=np.asarray(q).reshape(len(z),len(r));raw=_raw(q,z,r,background,chi_r,chi_z,operators)
    background_q=np.repeat((1/background["psi"]-z)[:,None],len(r),axis=1);zeros=np.zeros_like(q)
    return raw-_raw(background_q,z,r,background,zeros,zeros,operators)


def gw_high_order_jacobian(q,z,r,background,chi_r,chi_z,operators=None):
    operators=_operators(z,r) if operators is None else operators
    q=np.asarray(q).reshape(len(z),len(r));flat=q.ravel();s=(z[:,None]+q).ravel()
    qz=operators["Dz"]@flat;qr=operators["Dr"]@flat;lap=operators["Lap"]@flat
    gradient=((background["phi_z"][:,None]**2+chi_r**2+chi_z**2)/6).ravel()
    matrix=(
        diags(-lap+2*gradient*s)-diags(s)@operators["Lap"]
        +diags(4+4*qz)@operators["Dz"]+diags(4*qr)@operators["Dr"]
    ).tolil();nr=len(r);nz=len(z)
    for i in range(nz):
        for j in range(nr):
            row=i*nr+j
            if j==nr-1:
                matrix[row,:]=operators["Dr"].getrow(row);matrix[row,row]+=1/r[-1]
            elif i in (0,nz-1):
                matrix[row,:]=operators["Dz"].getrow(row)
    return matrix.tocsr()


def solve_gw_high_order_slice(
    amplitude,sigma_r=1.,sigma_y=.2,center_fraction=.9,d_over_ell=1.,r_max=8.,
    nz=25,nr=37,epsilon=.1,backreaction=.01,tolerance=1e-9,iterations=100,initial=None,
):
    z,r=make_grid(d_over_ell,r_max,nz,nr);background=solve_gw_background(z,epsilon,backreaction)
    if not background["converged"]:raise RuntimeError(background["message"])
    _,chi_r,chi_z=scalar_pulse(z,r,amplitude,sigma_r,sigma_y,center_fraction,d_over_ell)
    background_q=np.repeat((1/background["psi"]-z)[:,None],nr,axis=1)
    q=background_q.copy() if initial is None else np.asarray(initial).copy();operators=_operators(z,r);history=[]
    for _ in range(iterations):
        residual=gw_high_order_residual(q,z,r,background,chi_r,chi_z,operators)
        norm=float(np.max(np.abs(residual)));history.append(norm)
        if norm<tolerance:break
        step=spsolve(gw_high_order_jacobian(q,z,r,background,chi_r,chi_z,operators),-residual).reshape(q.shape)
        damping=1.;accepted=False
        while damping>=2**-16:
            candidate=q+damping*step
            if np.min(z[:,None]+candidate)>0 and np.max(np.abs(gw_high_order_residual(candidate,z,r,background,chi_r,chi_z,operators)))<norm:
                q=candidate;accepted=True;break
            damping*=.5
        if not accepted:break
    final_grid=gw_high_order_residual(q,z,r,background,chi_r,chi_z,operators).reshape(nz,nr)
    final=float(np.max(np.abs(final_grid)))
    psi=1/(z[:,None]+q);base=background["psi"][:,None]
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
        "discretization":"five-point polynomial differentiation matrices",
    }
