"""Independent five-point discretization of the coupled finite-wall slice."""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson
from scipy.sparse import bmat,diags,eye,kron,lil_matrix
from scipy.sparse.linalg import spsolve

from bhps.gw_background import solve_gw_background
from bhps.gw_slice_high_order_solver import derivative_matrix
from bhps.initial_data import make_grid
from bhps.scalar_pulse import scalar_pulse


def _operators(z,r,stencil_width=5):
    stencil_width=int(stencil_width)
    dz=derivative_matrix(z,1,stencil_width);dzz=derivative_matrix(z,2,stencil_width)
    dr=derivative_matrix(r,1,stencil_width);drr=derivative_matrix(r,2,stencil_width)
    radial=lil_matrix(drr);radial[0,:]=3*drr.getrow(0)
    for j in range(1,len(r)):radial[j,:]=drr.getrow(j)+(2/r[j])*dr.getrow(j)
    iz=eye(len(z),format="csr");ir=eye(len(r),format="csr")
    return (
        kron(dz,ir,format="csr"),kron(iz,dr,format="csr"),
        kron(dzz,ir,format="csr")+kron(iz,radial.tocsr(),format="csr"),
    )


def _background_fields(z,r,background):
    return (
        np.repeat((1/background["psi"]-z)[:,None],len(r),axis=1),
        np.repeat(background["phi"][:,None],len(r),axis=1),
    )


def _raw(q,phi,z,r,background,chi_r,chi_z,operators):
    q=np.asarray(q).reshape(len(z),len(r));phi=np.asarray(phi).reshape(q.shape);nz,nr=q.shape
    dzop,drop,lap=operators;qv=q.ravel();pv=phi.ravel();s=(z[:,None]+q).ravel()
    qz=dzop@qv;qr=drop@qv;lq=lap@qv;pz=dzop@pv;pr=drop@pv;lp=lap@pv
    mass=background["mass_squared"];chi2=(chi_r**2+chi_z**2).ravel()
    h=(-s*lq+4*qz+2*(qz**2+qr**2)+(pz**2+pr**2+chi2)*s**2/6+mass*pv**2/6).reshape(q.shape)
    f=(lp-3*((1+qz)*pz+qr*pr)/s-mass*pv/s**2).reshape(q.shape)
    q0,phi0=_background_fields(z,r,background);qzg=qz.reshape(q.shape);pzg=pz.reshape(q.shape);sg=s.reshape(q.shape)
    gamma=background["wall_stiffness"];v0=background["wall_target_v0"];v1=background["wall_target_v1"]
    h[:,-1]=(drop@qv).reshape(q.shape)[:,-1]+(q[:,-1]-q0[:,-1])/r[-1]
    f[:,-1]=(drop@pv).reshape(q.shape)[:,-1]+(phi[:,-1]-phi0[:,-1])/r[-1]
    ua=.5*gamma*(phi[0,:-1]-v0)**2;ub=.5*gamma*(phi[-1,:-1]-v1)**2
    beta_a=background["beta_a"]+(ua-background["wall_potential_a"])/6
    beta_b=background["beta_b"]-(ub-background["wall_potential_b"])/6
    h[0,:-1]=qzg[0,:-1]-(beta_a-1);h[-1,:-1]=qzg[-1,:-1]-(beta_b-1)
    f[0,:-1]=pzg[0,:-1]-.5*gamma*(phi[0,:-1]-v0)/sg[0,:-1]
    f[-1,:-1]=pzg[-1,:-1]+.5*gamma*(phi[-1,:-1]-v1)/sg[-1,:-1]
    return np.concatenate((h.ravel(),f.ravel()))


def finite_wall_high_order_residual(
    q,phi,z,r,background,chi_r,chi_z,operators=None,stencil_width=5,
):
    operators=(
        _operators(z,r,stencil_width) if operators is None else operators
    )
    raw=_raw(q,phi,z,r,background,chi_r,chi_z,operators)
    q0,phi0=_background_fields(z,r,background);zeros=np.zeros_like(q0)
    return raw-_raw(q0,phi0,z,r,background,zeros,zeros,operators)


def finite_wall_high_order_jacobian(
    q,phi,z,r,background,chi_r,chi_z,operators=None,stencil_width=5,
):
    operators=(
        _operators(z,r,stencil_width) if operators is None else operators
    )
    q=np.asarray(q).reshape(len(z),len(r));phi=np.asarray(phi).reshape(q.shape);nz,nr=q.shape;n=q.size
    dzop,drop,lap=operators;qv=q.ravel();pv=phi.ravel();s=(z[:,None]+q).ravel()
    qz=dzop@qv;qr=drop@qv;lq=lap@qv;pz=dzop@pv;pr=drop@pv
    mass=background["mass_squared"];gradient=(pz**2+pr**2+(chi_r**2+chi_z**2).ravel())/6
    a=(1+qz)*pz+qr*pr
    hq=(diags(-lq+2*gradient*s)-diags(s)@lap+diags(4+4*qz)@dzop+diags(4*qr)@drop).tolil()
    hp=(diags(s**2*pz/3)@dzop+diags(s**2*pr/3)@drop+diags(mass*pv/3)).tolil()
    fq=(-3*diags(pz/s)@dzop-3*diags(pr/s)@drop+diags(3*a/s**2+2*mass*pv/s**3)).tolil()
    fp=(lap-3*diags((1+qz)/s)@dzop-3*diags(qr/s)@drop-diags(np.full(n,mass)/s**2)).tolil()

    def replace(block,row,source):
        source=source.tocsr().getrow(row);block.rows[row]=source.indices.tolist();block.data[row]=source.data.tolist()

    zero=lil_matrix((n,n));outer=drop+diags(np.tile(np.r_[np.zeros(nr-1),1/r[-1]],nz))
    for i in range(nz):
        row=i*nr+nr-1;replace(hq,row,outer);replace(hp,row,zero);replace(fq,row,zero);replace(fp,row,outer)
    gamma=background["wall_stiffness"]
    for i,upper,target in ((0,False,background["wall_target_v0"]),(nz-1,True,background["wall_target_v1"])):
        for j in range(nr-1):
            row=i*nr+j;local=s[row];delta=pv[row]-target
            replace(hq,row,dzop);hp.rows[row]=[row];hp.data[row]=[(1 if upper else -1)*gamma*delta/6]
            fq.rows[row]=[row];fq.data[row]=[(-1 if upper else 1)*gamma*delta/(2*local**2)]
            replace(fp,row,dzop);fp[row,row]+=(1 if upper else -1)*gamma/(2*local)
    return bmat(((hq.tocsr(),hp.tocsr()),(fq.tocsr(),fp.tocsr())),format="csr")


def solve_finite_wall_high_order_slice(
    amplitude,wall_stiffness=20.,sigma_r=1.,sigma_y=.2,center_fraction=.9,
    d_over_ell=1.,r_max=8.,nz=25,nr=37,epsilon=.1,backreaction=.01,
    tolerance=1e-9,iterations=100,initial=None,stencil_width=5,
):
    z,r=make_grid(d_over_ell,r_max,nz,nr)
    background=solve_gw_background(z,epsilon,backreaction,wall_stiffness=wall_stiffness)
    if not background["converged"]:raise RuntimeError(background["message"])
    _,chi_r,chi_z=scalar_pulse(z,r,amplitude,sigma_r,sigma_y,center_fraction,d_over_ell)
    q0,phi0=_background_fields(z,r,background)
    if initial is None:q=q0.copy();phi=phi0.copy()
    elif isinstance(initial,dict):q=np.asarray(initial["q"]).copy();phi=np.asarray(initial["phi"]).copy()
    else:q=np.asarray(initial[0]).copy();phi=np.asarray(initial[1]).copy()
    stencil_width=int(stencil_width)
    operators=_operators(z,r,stencil_width);history=[]
    for _ in range(iterations):
        residual=finite_wall_high_order_residual(q,phi,z,r,background,chi_r,chi_z,operators)
        norm=float(np.max(np.abs(residual)));history.append(norm)
        if norm<tolerance:break
        step=spsolve(finite_wall_high_order_jacobian(q,phi,z,r,background,chi_r,chi_z,operators),-residual)
        dq,dphi=step[:q.size].reshape(q.shape),step[q.size:].reshape(phi.shape);damping=1.;accepted=False
        while damping>=2**-16:
            cq=q+damping*dq;cp=phi+damping*dphi
            if np.min(z[:,None]+cq)>0:
                trial=finite_wall_high_order_residual(cq,cp,z,r,background,chi_r,chi_z,operators)
                if np.max(np.abs(trial))<norm:q,phi=cq,cp;accepted=True;break
            damping*=.5
        if not accepted:break
    final_grid=finite_wall_high_order_residual(q,phi,z,r,background,chi_r,chi_z,operators)
    final=float(np.max(np.abs(final_grid)));psi=1/(z[:,None]+q)
    density=2*np.pi*r[None,:]**2*psi**2*(chi_r**2+chi_z**2)
    energy_simpson=float(simpson(simpson(density,x=r,axis=1),x=z))
    energy_trapezoid=float(np.trapezoid(np.trapezoid(density,x=r,axis=1),x=z))
    return {
        "converged":final<tolerance,"z":z,"r":r,"q":q,"phi":phi,"psi":psi,"background":background,
        "energy_dimensionless":energy_simpson,"energy_dimensionless_trapezoid":energy_trapezoid,
        "energy_quadrature_relative_difference":abs(energy_simpson-energy_trapezoid)/max(abs(energy_simpson),1e-300),
        "max_abs_residual":final,"residual_l2":float(np.sqrt(np.mean(final_grid**2))),
        "min_psi":float(np.min(psi)),"max_stabilizer_deformation":float(np.max(np.abs(phi-phi0))),
        "history":history,
        "stencil_width":stencil_width,
        "discretization":(
            f"{stencil_width}-point polynomial differentiation matrices "
            "for both coupled fields"
        ),
    }
