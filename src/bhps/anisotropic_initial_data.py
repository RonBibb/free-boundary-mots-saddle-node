"""Nonlinear conformal--scalar solve for a fixed trace-free shape metric."""

from __future__ import annotations

import numpy as np
from scipy.sparse import bmat,diags,eye,kron,lil_matrix
from scipy.sparse.linalg import spsolve

from bhps.anisotropic_geometry import axisymmetric_diagonal_geometry
from bhps.gw_slice_high_order_solver import derivative_matrix


def _shape_operators(z,r,a,b,stencil_width=7):
    z=np.asarray(z);r=np.asarray(r);nz,nr=len(z),len(r)
    dz1=derivative_matrix(z,1,stencil_width);dzz=derivative_matrix(z,2,stencil_width)
    dr1=derivative_matrix(r,1,stencil_width);drr=derivative_matrix(r,2,stencil_width)
    radial=lil_matrix(drr);radial[0,:]=3*drr.getrow(0)
    for j in range(1,nr):radial[j,:]=drr.getrow(j)+(2/r[j])*dr1.getrow(j)
    iz=eye(nz,format="csr");ir=eye(nr,format="csr")
    dz=kron(dz1,ir,format="csr");dr=kron(iz,dr1,format="csr")
    zz=kron(dzz,ir,format="csr");radial=kron(iz,radial.tocsr(),format="csr")
    av=np.asarray(a).ravel();bv=np.asarray(b).ravel()
    az=dz@av;br=dr@bv
    inverse_a=np.exp(-2*av);inverse_b=np.exp(-2*bv)
    lap=(
        diags(inverse_a)@(zz-2*diags(az)@dz)
        +diags(inverse_b)@(radial-2*diags(br)@dr)
    ).tocsr()
    return {"Dz":dz,"Dr":dr,"Lap":lap,"inverse_a":inverse_a,"inverse_b":inverse_b,"a_z":az,"b_r":br}


def _raw_residual_and_jacobian(
    q,phi,z,r,a,b,c,background,chi_r,chi_z,reference_q,reference_phi,
    stencil_width=7,need_jacobian=False,
):
    q=np.asarray(q).reshape(len(z),len(r));phi=np.asarray(phi).reshape(q.shape)
    a=np.asarray(a).reshape(q.shape);b=np.asarray(b).reshape(q.shape);c=np.asarray(c).reshape(q.shape)
    nz,nr=q.shape;n=q.size;operators=_shape_operators(z,r,a,b,stencil_width)
    dz,dr,lap=operators["Dz"],operators["Dr"],operators["Lap"]
    qv=q.ravel();pv=phi.ravel();psi=1/(np.asarray(z)[:,None]+q);psiv=psi.ravel()
    phiz=dz@pv;phir=dr@pv;chiz=np.asarray(chi_z).ravel();chir=np.asarray(chi_r).ravel()
    inverse_a,inverse_b=operators["inverse_a"],operators["inverse_b"]
    gradient=inverse_a*(phiz**2+chiz**2)+inverse_b*(phir**2+chir**2)
    mass=float(background["mass_squared"]);cosmological_constant=-6.
    ones=np.ones_like(psi);zero=np.zeros_like(psi)
    scalar_bar=axisymmetric_diagonal_geometry(
        z,r,ones,a,b,c,stencil_width,
    )["scalar_curvature"].ravel()
    potential=mass*pv**2
    h=-6*(lap@psiv)+(scalar_bar-gradient)*psiv-(2*cosmological_constant+potential)*psiv**3
    wz=dz@np.log(psiv);wr=dr@np.log(psiv)
    f=lap@pv+3*(inverse_a*wz*phiz+inverse_b*wr*phir)-mass*psiv**2*pv
    h=h.reshape(q.shape);f=f.reshape(q.shape)
    gamma=float(background["wall_stiffness"]);ea=np.exp(a)
    for wall_index,index,target in ((0,0,float(background["v0"])),(1,-1,float(background["v1"]))):
        wall_potential=.5*gamma*(phi[index]-target)**2
        beta=(
            float(background["beta_a"])+(wall_potential-float(background["wall_potential_a"]))/6
            if wall_index==0 else float(background["beta_b"])-(wall_potential-float(background["wall_potential_b"]))/6
        )
        psi_z=(dz@psiv).reshape(q.shape)[index]
        b_z=(dz@b.ravel()).reshape(q.shape)[index]
        h[index,:-1]=(psi_z+psi[index]*b_z+beta*psi[index]**2*ea[index])[:-1]
        phi_z=phiz.reshape(q.shape)[index]
        sign=-1 if wall_index==0 else 1
        f[index,:-1]=(phi_z+sign*.5*gamma*(phi[index]-target)*psi[index]*ea[index])[:-1]
    h[:,-1]=(dr@qv).reshape(q.shape)[:,-1]+(q[:,-1]-np.asarray(reference_q)[:,-1])/r[-1]
    f[:,-1]=(dr@pv).reshape(q.shape)[:,-1]+(phi[:,-1]-np.asarray(reference_phi)[:,-1])/r[-1]
    residual=np.concatenate((h.ravel(),f.ravel()))
    if not need_jacobian:return residual

    dpsi_dq=-psiv**2
    h_psi=(-6*lap+diags(scalar_bar-gradient-3*(2*cosmological_constant+potential)*psiv**2)).tocsr()
    hq=(h_psi@diags(dpsi_dq)).tolil()
    hp=(-2*diags(psiv*inverse_a*phiz)@dz-2*diags(psiv*inverse_b*phir)@dr
        -diags(2*mass*pv*psiv**3)).tolil()
    fp=(lap+3*diags(inverse_a*wz)@dz+3*diags(inverse_b*wr)@dr-diags(mass*psiv**2)).tolil()
    # delta log(psi) = -psi delta q, including derivatives of psi.
    fq=(
        (3*diags(inverse_a*phiz)@dz+3*diags(inverse_b*phir)@dr)@diags(-psiv)
        +diags(2*mass*psiv**3*pv)
    ).tolil()

    def replace(block,row,source):
        source=source.tocsr().getrow(row);block.rows[row]=source.indices.tolist();block.data[row]=source.data.tolist()

    def replace_with_row(block,row,row_source):
        source=row_source.tocsr();block.rows[row]=source.indices.tolist();block.data[row]=source.data.tolist()

    empty=lil_matrix((n,n));outer=dr+diags(np.tile(np.r_[np.zeros(nr-1),1/r[-1]],nz))
    bz=(dz@b.ravel()).reshape(q.shape)
    for wall_index,i,target in ((0,0,float(background["v0"])),(1,nz-1,float(background["v1"]))):
        index=0 if i==0 else -1;delta=phi[index]-target
        wall_potential=.5*gamma*delta**2
        beta=(
            float(background["beta_a"])+(wall_potential-float(background["wall_potential_a"]))/6
            if wall_index==0 else float(background["beta_b"])-(wall_potential-float(background["wall_potential_b"]))/6
        )
        beta_phi=(gamma*delta/6) if wall_index==0 else (-gamma*delta/6)
        scalar_sign=-1 if wall_index==0 else 1
        for j in range(nr-1):
            row=i*nr+j;local_psi=psi[index,j];local_ea=ea[index,j]
            # Construct local-row forms explicitly to avoid an auxiliary full diagonal.
            hpsi=dz.getrow(row).tolil();hpsi[0,row]+=bz[index,j]+2*beta[j]*local_psi*local_ea
            replace_with_row(hq,row,hpsi.tocsr()@diags(dpsi_dq))
            hp.rows[row]=[row];hp.data[row]=[beta_phi[j]*local_psi**2*local_ea]
            fq.rows[row]=[row];fq.data[row]=[-scalar_sign*.5*gamma*delta[j]*local_ea*local_psi**2]
            replace(fp,row,dz)
            fp[row,row]+=scalar_sign*.5*gamma*local_psi*local_ea
    for i in range(nz):
        row=i*nr+nr-1
        replace(hq,row,outer);replace(hp,row,empty)
        replace(fq,row,empty);replace(fp,row,outer)
    return residual,bmat(((hq.tocsr(),hp.tocsr()),(fq.tocsr(),fp.tocsr())),format="csr")


def anisotropic_initial_data_residual(
    q,phi,z,r,a,b,c,background,chi_r,chi_z,reference_q,reference_phi,
    stencil_width=7,
):
    """Well-balanced Hamiltonian plus momentarily stationary scalar residual."""
    raw=_raw_residual_and_jacobian(
        q,phi,z,r,a,b,c,background,chi_r,chi_z,reference_q,reference_phi,
        stencil_width,False,
    )
    zero=np.zeros_like(a)
    defect=_raw_residual_and_jacobian(
        reference_q,reference_phi,z,r,zero,zero,zero,background,chi_r,chi_z,
        reference_q,reference_phi,stencil_width,False,
    )
    # Retain the complete reference-slice discretization defect.  Shape
    # dependence of the physical source remains active in ``raw``.
    return raw-defect


def anisotropic_initial_data_jacobian(
    q,phi,z,r,a,b,c,background,chi_r,chi_z,reference_q,reference_phi,
    stencil_width=7,
):
    return _raw_residual_and_jacobian(
        q,phi,z,r,a,b,c,background,chi_r,chi_z,reference_q,reference_phi,
        stencil_width,True,
    )[1]


def solve_anisotropic_initial_data(
    z,r,reference_q,reference_phi,a,b,c,background,chi_r,chi_z,
    initial_q=None,initial_phi=None,stencil_width=7,tolerance=1e-9,iterations=20,
):
    """Newton solve for ``q,phi`` with a prescribed trace-free shape metric."""
    q=np.asarray(reference_q).copy() if initial_q is None else np.asarray(initial_q).copy()
    phi=np.asarray(reference_phi).copy() if initial_phi is None else np.asarray(initial_phi).copy()
    history=[]
    for _ in range(int(iterations)):
        residual=anisotropic_initial_data_residual(
            q,phi,z,r,a,b,c,background,chi_r,chi_z,reference_q,reference_phi,stencil_width,
        )
        norm=float(np.max(np.abs(residual)));history.append(norm)
        if norm<tolerance:break
        jacobian=anisotropic_initial_data_jacobian(
            q,phi,z,r,a,b,c,background,chi_r,chi_z,reference_q,reference_phi,stencil_width,
        )
        step=spsolve(jacobian,-residual);size=q.size
        dq=step[:size].reshape(q.shape);dphi=step[size:].reshape(phi.shape)
        damping=1.;accepted=False
        while damping>=2**-16:
            candidate_q=q+damping*dq;candidate_phi=phi+damping*dphi
            if np.min(np.asarray(z)[:,None]+candidate_q)>0:
                trial=anisotropic_initial_data_residual(
                    candidate_q,candidate_phi,z,r,a,b,c,background,chi_r,chi_z,
                    reference_q,reference_phi,stencil_width,
                )
                if np.max(np.abs(trial))<norm:
                    q,phi=candidate_q,candidate_phi;accepted=True;break
            damping*=.5
        if not accepted:break
    final=anisotropic_initial_data_residual(
        q,phi,z,r,a,b,c,background,chi_r,chi_z,reference_q,reference_phi,stencil_width,
    )
    return {
        "q":q,"phi":phi,"psi":1/(np.asarray(z)[:,None]+q),
        "converged":bool(np.max(np.abs(final))<tolerance),
        "maximum_residual":float(np.max(np.abs(final))),
        "residual_l2":float(np.sqrt(np.mean(final**2))),"history":history,
    }
