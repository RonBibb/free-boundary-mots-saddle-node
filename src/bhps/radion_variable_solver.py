"""Well-balanced C3 solver using the exact radion variable q=1/psi-z."""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from bhps.initial_data import make_grid
from bhps.scalar_pulse import scalar_pulse


def _interpolated_axis_extremum(relative,z):
    axis=np.asarray(relative)[:,0];y=np.log(z)
    index=int(np.argmax(np.abs(axis)));value=float(axis[index]);location=float(y[index])
    if 0<index<len(y)-1:
        sign=1. if axis[index]>=0 else -1.
        coefficients=np.polyfit(y[index-1:index+2],sign*axis[index-1:index+2],2)
        if coefficients[0]<0:
            vertex=float(-coefficients[1]/(2*coefficients[0]))
            if y[index-1]<=vertex<=y[index+1]:
                value=float(sign*np.polyval(coefficients,vertex));location=vertex
    return value,location


def q_residual(q,z,r,chi_r,chi_z,outer_boundary="asymptotic_radion"):
    q=np.asarray(q).reshape(len(z),len(r));nz,nr=q.shape
    dz,dr=z[1]-z[0],r[1]-r[0]
    out=np.zeros_like(q);source=(chi_r**2+chi_z**2)/6
    s=z[:,None]+q
    qzz=(q[2:,1:-1]-2*q[1:-1,1:-1]+q[:-2,1:-1])/dz**2
    qrr=(q[1:-1,2:]-2*q[1:-1,1:-1]+q[1:-1,:-2])/dr**2
    qz=(q[2:,1:-1]-q[:-2,1:-1])/(2*dz)
    qr=(q[1:-1,2:]-q[1:-1,:-2])/(2*dr)
    radial=qrr+2*qr/r[None,1:-1]
    local=s[1:-1,1:-1]
    out[1:-1,1:-1]=-local*(qzz+radial)+4*qz+2*(qz**2+qr**2)+source[1:-1,1:-1]*local**2
    qzz_axis=(q[2:,0]-2*q[1:-1,0]+q[:-2,0])/dz**2
    radial_axis=6*(q[1:-1,1]-q[1:-1,0])/dr**2
    qz_axis=(q[2:,0]-q[:-2,0])/(2*dz)
    local_axis=s[1:-1,0]
    out[1:-1,0]=-local_axis*(qzz_axis+radial_axis)+4*qz_axis+2*qz_axis**2+source[1:-1,0]*local_axis**2
    out[0,:-1]=(-3*q[0,:-1]+4*q[1,:-1]-q[2,:-1])/(2*dz)
    out[-1,:-1]=(3*q[-1,:-1]-4*q[-2,:-1]+q[-3,:-1])/(2*dz)
    if outer_boundary=="asymptotic_radion":
        out[:,-1]=(3*q[:,-1]-4*q[:,-2]+q[:,-3])/(2*dr)+q[:,-1]/r[-1]
    elif outer_boundary=="dirichlet":
        out[:,-1]=q[:,-1]
    else:
        raise ValueError("outer_boundary must be 'dirichlet' or 'asymptotic_radion'")
    return out.ravel()


def q_jacobian(q,z,r,chi_r,chi_z,outer_boundary="asymptotic_radion"):
    q=np.asarray(q).reshape(len(z),len(r));nz,nr=q.shape
    dz,dr=z[1]-z[0],r[1]-r[0]
    source=(chi_r**2+chi_z**2)/6;s=z[:,None]+q
    matrix=lil_matrix((nz*nr,nz*nr));index=lambda i,j:i*nr+j
    for i in range(nz):
        for j in range(nr):
            row=index(i,j)
            if j==nr-1:
                if outer_boundary=="dirichlet":
                    matrix[row,row]=1
                elif outer_boundary=="asymptotic_radion":
                    matrix[row,index(i,j-2)]=1/(2*dr)
                    matrix[row,index(i,j-1)]=-2/dr
                    matrix[row,row]=3/(2*dr)+1/r[-1]
                else:
                    raise ValueError("outer_boundary must be 'dirichlet' or 'asymptotic_radion'")
            elif i==0:
                matrix[row,index(0,j)]=-3/(2*dz)
                matrix[row,index(1,j)]=2/dz
                matrix[row,index(2,j)]=-1/(2*dz)
            elif i==nz-1:
                matrix[row,index(i,j)]=3/(2*dz)
                matrix[row,index(i-1,j)]=-2/dz
                matrix[row,index(i-2,j)]=1/(2*dz)
            else:
                local=s[i,j]
                qz=(q[i+1,j]-q[i-1,j])/(2*dz)
                if j==0:
                    lap=(q[i+1,0]-2*q[i,0]+q[i-1,0])/dz**2+6*(q[i,1]-q[i,0])/dr**2
                    matrix[row,index(i-1,0)]=-local/dz**2-(2+2*qz)/dz
                    matrix[row,index(i+1,0)]=-local/dz**2+(2+2*qz)/dz
                    matrix[row,index(i,1)]=-6*local/dr**2
                    matrix[row,row]=local*(2/dz**2+6/dr**2)-lap+2*source[i,0]*local
                else:
                    qr=(q[i,j+1]-q[i,j-1])/(2*dr)
                    lap=(q[i+1,j]-2*q[i,j]+q[i-1,j])/dz**2+(q[i,j+1]-2*q[i,j]+q[i,j-1])/dr**2+(q[i,j+1]-q[i,j-1])/(r[j]*dr)
                    minus=1/dr**2-1/(r[j]*dr);plus=1/dr**2+1/(r[j]*dr)
                    matrix[row,index(i-1,j)]=-local/dz**2-(2+2*qz)/dz
                    matrix[row,index(i+1,j)]=-local/dz**2+(2+2*qz)/dz
                    matrix[row,index(i,j-1)]=-local*minus-2*qr/dr
                    matrix[row,index(i,j+1)]=-local*plus+2*qr/dr
                    matrix[row,row]=local*(2/dz**2+2/dr**2)-lap+2*source[i,j]*local
    return matrix.tocsr()


def solve_q(
    amplitude,
    sigma_r=1.,
    sigma_y=.2,
    center_fraction=.9,
    d_over_ell=1.,
    r_max=8.,
    nz=25,
    nr=37,
    tolerance=1e-9,
    iterations=80,
    initial=None,
    outer_boundary="asymptotic_radion",
):
    z,r=make_grid(d_over_ell,r_max,nz,nr)
    chi,chi_r,chi_z=scalar_pulse(z,r,amplitude,sigma_r,sigma_y,center_fraction,d_over_ell)
    q=np.zeros((nz,nr)) if initial is None else np.asarray(initial).copy()
    history=[]
    for _ in range(iterations):
        residual=q_residual(q,z,r,chi_r,chi_z,outer_boundary)
        norm=float(np.max(np.abs(residual)));history.append(norm)
        if norm<tolerance:
            break
        step=spsolve(q_jacobian(q,z,r,chi_r,chi_z,outer_boundary),-residual).reshape(q.shape)
        damping=1.;accepted=False
        while damping>=2**-16:
            candidate=q+damping*step
            if np.min(z[:,None]+candidate)>0 and np.max(np.abs(q_residual(candidate,z,r,chi_r,chi_z,outer_boundary)))<norm:
                q=candidate;accepted=True;break
            damping*=.5
        if not accepted:
            break
    final=float(np.max(np.abs(q_residual(q,z,r,chi_r,chi_z,outer_boundary))))
    psi=1/(z[:,None]+q);background=np.repeat((1/z)[:,None],nr,axis=1)
    relative=psi/background-1
    interpolated,interpolated_y=_interpolated_axis_extremum(relative,z)
    density=2*np.pi*r[None,:]**2*psi**2*(chi_r**2+chi_z**2)
    energy=float(simpson(simpson(density,x=r,axis=1),x=z))
    return {
        "converged":final<tolerance,
        "z":z,"r":r,"q":q,"psi":psi,"chi":chi,
        "energy_dimensionless":energy,
        "max_abs_residual":final,
        "min_psi":float(np.min(psi)),
        "max_relative_deformation":float(np.max(np.abs(relative))),
        "interpolated_axis_extremum":interpolated,
        "interpolated_axis_extremum_y":interpolated_y,
        "history":history,
    }


def continue_q(amplitudes,**kwargs):
    previous=None;accepted=[];rejected=None
    for amplitude in amplitudes:
        solved=solve_q(float(amplitude),initial=previous,**kwargs)
        record={key:value for key,value in solved.items() if key not in {"z","r","q","psi","chi","history"}}
        record["amplitude"]=float(amplitude);record["newton_iterations"]=len(solved["history"])
        if not solved["converged"]:
            rejected=record;break
        accepted.append(record);previous=solved["q"]
    return {"accepted":accepted,"rejected":rejected}
