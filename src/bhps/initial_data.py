"""Independent finite-difference initial-data solver for BHPS."""

from __future__ import annotations

import math
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def make_grid(d_over_ell=1.0, r_max=8.0, nz=17, nr=25):
    return np.linspace(1, math.exp(d_over_ell), nz), np.linspace(0, r_max, nr)


def residual(psi, z, r, amplitude, width):
    psi=np.asarray(psi).reshape(len(z),len(r)); nz,nr=psi.shape
    dz,dr=z[1]-z[0],r[1]-r[0]; out=np.zeros_like(psi)
    out[1:-1,1:-1]=(psi[2:,1:-1]-2*psi[1:-1,1:-1]+psi[:-2,1:-1])/dz**2+(psi[1:-1,2:]-2*psi[1:-1,1:-1]+psi[1:-1,:-2])/dr**2+(psi[1:-1,2:]-psi[1:-1,:-2])/(r[None,1:-1]*dr)-2*psi[1:-1,1:-1]**3
    out[1:-1,0]=(psi[2:,0]-2*psi[1:-1,0]+psi[:-2,0])/dz**2+6*(psi[1:-1,1]-psi[1:-1,0])/dr**2-2*psi[1:-1,0]**3
    out[0,:-1]=(-3*psi[0,:-1]+4*psi[1,:-1]-psi[2,:-1])/(2*dz)+psi[0,:-1]**2
    q=amplitude*np.exp(-(r[:-1]/width)**2)
    out[-1,:-1]=(3*psi[-1,:-1]-4*psi[-2,:-1]+psi[-3,:-1])/(2*dz)+(1-q)*psi[-1,:-1]**2
    out[:,-1]=psi[:,-1]-1/z
    background=np.repeat((1/z)[:,None],nr,axis=1)
    # Subtract the raw finite-grid AdS defect explicitly.
    defect=np.zeros_like(background)
    defect[1:-1,1:-1]=(background[2:,1:-1]-2*background[1:-1,1:-1]+background[:-2,1:-1])/dz**2-2*background[1:-1,1:-1]**3
    defect[1:-1,0]=(background[2:,0]-2*background[1:-1,0]+background[:-2,0])/dz**2-2*background[1:-1,0]**3
    defect[0,:-1]=(-3*background[0,:-1]+4*background[1,:-1]-background[2,:-1])/(2*dz)+background[0,:-1]**2
    defect[-1,:-1]=(3*background[-1,:-1]-4*background[-2,:-1]+background[-3,:-1])/(2*dz)+background[-1,:-1]**2
    return (out-defect).ravel()


def jacobian(psi,z,r,amplitude,width):
    psi=np.asarray(psi).reshape(len(z),len(r)); nz,nr=psi.shape; dz,dr=z[1]-z[0],r[1]-r[0]
    J=lil_matrix((nz*nr,nz*nr)); ix=lambda i,j:i*nr+j; q=amplitude*np.exp(-(r/width)**2)
    for i in range(nz):
        for j in range(nr):
            row=ix(i,j)
            if j==nr-1:J[row,row]=1
            elif i==0:
                J[row,ix(0,j)]=-3/(2*dz)+2*psi[0,j];J[row,ix(1,j)]=2/dz;J[row,ix(2,j)]=-1/(2*dz)
            elif i==nz-1:
                J[row,ix(i,j)]=3/(2*dz)+2*(1-q[j])*psi[i,j];J[row,ix(i-1,j)]=-2/dz;J[row,ix(i-2,j)]=1/(2*dz)
            elif j==0:
                J[row,ix(i-1,0)]=1/dz**2;J[row,ix(i+1,0)]=1/dz**2;J[row,ix(i,1)]=6/dr**2;J[row,row]=-2/dz**2-6/dr**2-6*psi[i,0]**2
            else:
                J[row,ix(i-1,j)]=J[row,ix(i+1,j)]=1/dz**2;J[row,ix(i,j-1)]=1/dr**2-1/(r[j]*dr);J[row,ix(i,j+1)]=1/dr**2+1/(r[j]*dr);J[row,row]=-2/dz**2-2/dr**2-6*psi[i,j]**2
    return J.tocsr()


def solve(amplitude, width=.75, nz=17, nr=25, initial=None, tolerance=1e-9, iterations=80):
    z,r=make_grid(nz=nz,nr=nr); background=np.repeat((1/z)[:,None],nr,axis=1); psi=background.copy() if initial is None else np.asarray(initial).copy()
    history=[]
    for _ in range(iterations):
        f=residual(psi,z,r,amplitude,width); norm=float(np.max(np.abs(f))); history.append(norm)
        if norm<tolerance:return {"converged":True,"z":z,"r":r,"psi":psi,"max_abs_residual":norm,"history":history}
        step=spsolve(jacobian(psi,z,r,amplitude,width),-f).reshape(psi.shape); damping=1.; accepted=False
        while damping>=2**-16:
            candidate=psi+damping*step
            if np.min(candidate)>0 and np.max(np.abs(residual(candidate,z,r,amplitude,width)))<norm:psi=candidate;accepted=True;break
            damping*=.5
        if not accepted:break
    return {"converged":False,"z":z,"r":r,"psi":psi,"max_abs_residual":float(np.max(np.abs(residual(psi,z,r,amplitude,width)))),"history":history}
