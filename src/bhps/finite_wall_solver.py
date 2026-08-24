"""Coupled finite-wall metric--stabilizer initial-data solver.

The time-symmetric Hamiltonian constraint is solved together with a selected
quasi-static Goldberger--Wise response.  The latter uses the lapse ``N=psi``;
it is therefore a defined elliptic initial-data subfamily, not a claim that
the resulting slice is a complete static spacetime.

Units are ``ell=kappa5=1`` and ``q=1/psi-z`` is used for conditioning.  The
brane scalar equations use the one-sided Z2 half-Robin coefficients.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson
from scipy.sparse import bmat, diags, eye, kron, lil_matrix
from scipy.sparse.linalg import spsolve

from bhps.gw_background import solve_gw_background
from bhps.initial_data import make_grid
from bhps.scalar_pulse import scalar_pulse


def _operators(z, r):
    """Second-order axisymmetric operators in C-order flattening."""
    z=np.asarray(z);r=np.asarray(r);nz,nr=len(z),len(r)
    dz,dr=z[1]-z[0],r[1]-r[0]
    dz1=lil_matrix((nz,nz));dzz=lil_matrix((nz,nz))
    dz1[0,0:3]=[-3/(2*dz),2/dz,-1/(2*dz)]
    dz1[-1,-3:]=[1/(2*dz),-2/dz,3/(2*dz)]
    for i in range(1,nz-1):
        dz1[i,i-1]=-1/(2*dz);dz1[i,i+1]=1/(2*dz)
        dzz[i,i-1:i+2]=[1/dz**2,-2/dz**2,1/dz**2]
    dr1=lil_matrix((nr,nr));lrad=lil_matrix((nr,nr))
    lrad[0,0]=-6/dr**2;lrad[0,1]=6/dr**2
    dr1[-1,-3:]=[1/(2*dr),-2/dr,3/(2*dr)]
    for j in range(1,nr-1):
        dr1[j,j-1]=-1/(2*dr);dr1[j,j+1]=1/(2*dr)
        lrad[j,j-1]=1/dr**2-1/(r[j]*dr)
        lrad[j,j]=-2/dr**2
        lrad[j,j+1]=1/dr**2+1/(r[j]*dr)
    iz=eye(nz,format="csr");ir=eye(nr,format="csr")
    dzop=kron(dz1.tocsr(),ir,format="csr")
    drop=kron(iz,dr1.tocsr(),format="csr")
    lap=kron(dzz.tocsr(),ir,format="csr")+kron(iz,lrad.tocsr(),format="csr")
    return dzop,drop,lap


def _background_fields(z, r, background):
    q0=np.repeat((1/background["psi"]-z)[:,None],len(r),axis=1)
    phi0=np.repeat(background["phi"][:,None],len(r),axis=1)
    return q0,phi0


def _raw_residual(q, phi, z, r, background, chi_r, chi_z, operators=None):
    q=np.asarray(q).reshape(len(z),len(r));phi=np.asarray(phi).reshape(q.shape)
    nz,nr=q.shape;n=q.size
    dzop,drop,lap=_operators(z,r) if operators is None else operators
    qv=q.ravel();pv=phi.ravel();s=(z[:,None]+q).ravel()
    qz=dzop@qv;qr=drop@qv;lq=lap@qv
    pz=dzop@pv;pr=drop@pv;lp=lap@pv
    mass=background["mass_squared"]
    chi2=(chi_r**2+chi_z**2).ravel()
    h=-s*lq+4*qz+2*(qz**2+qr**2)+(pz**2+pr**2+chi2)*s**2/6+mass*pv**2/6
    a=(1+qz)*pz+qr*pr
    f=lp-3*a/s-mass*pv/s**2
    h=h.reshape(nz,nr);f=f.reshape(nz,nr)
    q0,phi0=_background_fields(z,r,background)
    gamma=background["wall_stiffness"]
    v0,v1=background["wall_target_v0"],background["wall_target_v1"]
    ua0,ub0=background["wall_potential_a"],background["wall_potential_b"]
    qz2=qz.reshape(nz,nr);pz2=pz.reshape(nz,nr);ss=s.reshape(nz,nr)
    # Outer radial boundary takes precedence at the two corners.
    h[:,-1]=(drop@qv).reshape(nz,nr)[:,-1]+(q[:,-1]-q0[:,-1])/r[-1]
    f[:,-1]=(drop@pv).reshape(nz,nr)[:,-1]+(phi[:,-1]-phi0[:,-1])/r[-1]
    ua=.5*gamma*(phi[0,:-1]-v0)**2
    ub=.5*gamma*(phi[-1,:-1]-v1)**2
    beta_a=background["beta_a"]+(ua-ua0)/6
    beta_b=background["beta_b"]-(ub-ub0)/6
    h[0,:-1]=qz2[0,:-1]-(beta_a-1)
    h[-1,:-1]=qz2[-1,:-1]-(beta_b-1)
    f[0,:-1]=pz2[0,:-1]-.5*gamma*(phi[0,:-1]-v0)/ss[0,:-1]
    f[-1,:-1]=pz2[-1,:-1]+.5*gamma*(phi[-1,:-1]-v1)/ss[-1,:-1]
    return np.concatenate((h.ravel(),f.ravel()))


def finite_wall_residual(q, phi, z, r, background, chi_r, chi_z, selector_source=None):
    """Well-balanced coupled residual ordered as ``[H, KG]``."""
    operators=_operators(z,r)
    raw=_raw_residual(q,phi,z,r,background,chi_r,chi_z,operators)
    q0,phi0=_background_fields(z,r,background);zeros=np.zeros_like(q0)
    defect=_raw_residual(q0,phi0,z,r,background,zeros,zeros,operators)
    result=raw-defect
    if selector_source is not None:
        source=np.asarray(selector_source,dtype=float).reshape(q0.shape).copy()
        source[0,:]=0.;source[-1,:]=0.;source[:,-1]=0.
        result[q0.size:]-=source.ravel()
    return result


def finite_wall_jacobian(q, phi, z, r, background, chi_r, chi_z):
    """Exact block Jacobian of :func:`finite_wall_residual`."""
    q=np.asarray(q).reshape(len(z),len(r));phi=np.asarray(phi).reshape(q.shape)
    nz,nr=q.shape;n=q.size
    dzop,drop,lap=_operators(z,r)
    qv=q.ravel();pv=phi.ravel();s=(z[:,None]+q).ravel()
    qz=dzop@qv;qr=drop@qv;lq=lap@qv
    pz=dzop@pv;pr=drop@pv
    mass=background["mass_squared"]
    chi2=(chi_r**2+chi_z**2).ravel()
    gradient=(pz**2+pr**2+chi2)/6
    a=(1+qz)*pz+qr*pr
    hq=(-diags(s)@lap+diags(4+4*qz)@dzop+diags(4*qr)@drop
        +diags(-lq+2*gradient*s)).tolil()
    hp=(diags(s**2*pz/3)@dzop+diags(s**2*pr/3)@drop
        +diags(mass*pv/3)).tolil()
    fq=(-3*diags(pz/s)@dzop-3*diags(pr/s)@drop
        +diags(3*a/s**2+2*mass*pv/s**3)).tolil()
    fp=(lap-3*diags((1+qz)/s)@dzop-3*diags(qr/s)@drop
        -diags(np.full(n,mass)/s**2)).tolil()

    def replace_row(block, row, source):
        source=source.tocsr().getrow(row)
        block.rows[row]=source.indices.tolist();block.data[row]=source.data.tolist()

    q0,phi0=_background_fields(z,r,background)
    gamma=background["wall_stiffness"];v0=background["wall_target_v0"];v1=background["wall_target_v1"]
    outer_q=drop+diags(np.tile(np.r_[np.zeros(nr-1),1/r[-1]],nz))
    outer_p=outer_q
    zero=lil_matrix((n,n))
    for i in range(nz):
        row=i*nr+nr-1
        replace_row(hq,row,outer_q);replace_row(hp,row,zero)
        replace_row(fq,row,zero);replace_row(fp,row,outer_p)
    for i,upper,target in ((0,False,v0),(nz-1,True,v1)):
        for j in range(nr-1):
            row=i*nr+j;local=s[row];delta=pv[row]-target
            replace_row(hq,row,dzop)
            hp.rows[row]=[row];hp.data[row]=[(1 if upper else -1)*gamma*delta/6]
            fq.rows[row]=[row];fq.data[row]=[(-1 if upper else 1)*gamma*delta/(2*local**2)]
            replace_row(fp,row,dzop)
            fp[row,row]+= (1 if upper else -1)*gamma/(2*local)
    return bmat(((hq.tocsr(),hp.tocsr()),(fq.tocsr(),fp.tocsr())),format="csr")


def solve_finite_wall_slice(
    amplitude,wall_stiffness=20.,sigma_r=1.,sigma_y=.2,center_fraction=.9,
    d_over_ell=1.,r_max=8.,nz=25,nr=37,epsilon=.1,backreaction=.01,
    tolerance=1e-9,iterations=100,initial=None,stabilizer_forcing_amplitude=0.,
    stabilizer_forcing_mode=1,stabilizer_forcing_sigma_r=1.,stabilizer_forcing_profile="sin",
):
    """Solve the coupled finite-wall selected initial-data subfamily."""
    z,r=make_grid(d_over_ell,r_max,nz,nr)
    background=solve_gw_background(z,epsilon,backreaction,wall_stiffness=wall_stiffness)
    if not background["converged"]:
        raise RuntimeError(background["message"])
    _,chi_r,chi_z=scalar_pulse(z,r,amplitude,sigma_r,sigma_y,center_fraction,d_over_ell)
    if int(stabilizer_forcing_mode)<1:
        raise ValueError("stabilizer_forcing_mode must be a positive integer")
    compact=(z-z[0])/(z[-1]-z[0])
    phase=int(stabilizer_forcing_mode)*np.pi*compact
    if stabilizer_forcing_profile=="sin":compact_profile=np.sin(phase)
    elif stabilizer_forcing_profile=="sin_squared":compact_profile=np.sin(phase)**2
    else:raise ValueError("stabilizer_forcing_profile must be 'sin' or 'sin_squared'")
    selector_source=(
        float(stabilizer_forcing_amplitude)*background["v0"]*compact_profile[:,None]
        *np.exp(-r[None,:]**2/(2*float(stabilizer_forcing_sigma_r)**2))
    )
    q0,phi0=_background_fields(z,r,background)
    if initial is None:
        q=q0.copy();phi=phi0.copy()
    elif isinstance(initial,dict):
        q=np.asarray(initial["q"]).copy();phi=np.asarray(initial["phi"]).copy()
    else:
        q=np.asarray(initial[0]).copy();phi=np.asarray(initial[1]).copy()
    history=[]
    for _ in range(iterations):
        residual=finite_wall_residual(q,phi,z,r,background,chi_r,chi_z,selector_source)
        norm=float(np.max(np.abs(residual)));history.append(norm)
        if norm<tolerance:break
        step=spsolve(finite_wall_jacobian(q,phi,z,r,background,chi_r,chi_z),-residual)
        dq,dphi=step[:q.size].reshape(q.shape),step[q.size:].reshape(phi.shape)
        damping=1.;accepted=False
        while damping>=2**-16:
            candidate_q=q+damping*dq;candidate_phi=phi+damping*dphi
            if np.min(z[:,None]+candidate_q)>0:
                trial=finite_wall_residual(candidate_q,candidate_phi,z,r,background,chi_r,chi_z,selector_source)
                if np.max(np.abs(trial))<norm:
                    q,phi=candidate_q,candidate_phi;accepted=True;break
            damping*=.5
        if not accepted:break
    final_grid=finite_wall_residual(q,phi,z,r,background,chi_r,chi_z,selector_source)
    h_grid=final_grid[:q.size].reshape(q.shape);f_grid=final_grid[q.size:].reshape(q.shape)
    final=float(np.max(np.abs(final_grid)));psi=1/(z[:,None]+q)
    density=2*np.pi*r[None,:]**2*psi**2*(chi_r**2+chi_z**2)
    energy_simpson=float(simpson(simpson(density,x=r,axis=1),x=z))
    energy_trapezoid=float(np.trapezoid(np.trapezoid(density,x=r,axis=1),x=z))
    return {
        "converged":final<tolerance,"z":z,"r":r,"q":q,"phi":phi,"psi":psi,
        "background":background,"energy_dimensionless":energy_simpson,
        "energy_dimensionless_trapezoid":energy_trapezoid,
        "energy_quadrature_relative_difference":abs(energy_simpson-energy_trapezoid)/max(abs(energy_simpson),1e-300),
        "max_abs_residual":final,"hamiltonian_residual_max":float(np.max(np.abs(h_grid))),
        "stabilizer_residual_max":float(np.max(np.abs(f_grid))),
        "bulk_hamiltonian_residual_max":float(np.max(np.abs(h_grid[1:-1,:-1]))),
        "bulk_stabilizer_residual_max":float(np.max(np.abs(f_grid[1:-1,:-1]))),
        "metric_junction_residual_max":float(max(np.max(np.abs(h_grid[0,:-1])),np.max(np.abs(h_grid[-1,:-1])))),
        "scalar_wall_residual_max":float(max(np.max(np.abs(f_grid[0,:-1])),np.max(np.abs(f_grid[-1,:-1])))),
        "outer_boundary_residual_max":float(max(np.max(np.abs(h_grid[:,-1])),np.max(np.abs(f_grid[:,-1])))),
        "residual_l2":float(np.sqrt(np.mean(final_grid**2))),"min_psi":float(np.min(psi)),
        "max_relative_metric_deformation":float(np.max(np.abs(psi/np.repeat(background["psi"][:,None],nr,axis=1)-1))),
        "max_stabilizer_deformation":float(np.max(np.abs(phi-phi0))),"history":history,
        "stabilizer_forcing_amplitude":float(stabilizer_forcing_amplitude),
        "stabilizer_forcing_mode":int(stabilizer_forcing_mode),
        "stabilizer_forcing_profile":stabilizer_forcing_profile,
        "stabilizer_forcing_max":float(np.max(np.abs(selector_source))),
        "initial_data_interpretation":"Hamiltonian plus sourced N=psi stabilizer selector; zero source is momentarily stationary, nonzero source represents controlled initial scalar acceleration; not a complete static-spacetime solve",
    }
