"""Manufactured-solution and near-null-mode diagnostics for V1."""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigs

from bhps.initial_data import make_grid, residual as vacuum_residual
from bhps.scalar_pulse import scalar_pulse
from bhps.v1_solver import solve_v1, v1_jacobian


def manufactured_stencil_errors(nz=25,nr=37,r_max=8.,alpha=.01):
    """Compare each discrete stencil class with an analytic continuum value."""
    z,r=make_grid(1.,r_max,nz,nr);R=r_max
    g=(1-(r/R)**2)**2
    gp=-4*r/R**2*(1-(r/R)**2)
    radial_lap=-12/R**2+20*r**2/R**4
    psi=1/z[:,None]+alpha*z[:,None]**2*g[None,:]
    continuum=2/z[:,None]**3+alpha*(2*g[None,:]+z[:,None]**2*radial_lap[None,:])-2*psi**3
    discrete=vacuum_residual(psi,z,r,0,.75).reshape(nz,nr)
    boundary_continuum=(-1/z[:,None]**2+2*alpha*z[:,None]*g[None,:])+psi**2
    return {
        "dz":float(z[1]-z[0]),"dr":float(r[1]-r[0]),
        "interior_max_error":float(np.max(np.abs(discrete[1:-1,1:-1]-continuum[1:-1,1:-1]))),
        "axis_max_error":float(np.max(np.abs(discrete[1:-1,0]-continuum[1:-1,0]))),
        "brane_A_max_error":float(np.max(np.abs(discrete[0,:-1]-boundary_continuum[0,:-1]))),
        "brane_B_max_error":float(np.max(np.abs(discrete[-1,:-1]-boundary_continuum[-1,:-1]))),
        "outer_boundary_max_error":float(np.max(np.abs(discrete[:,-1]))),
    }


def near_null_diagnostic(amplitude=.5,nz=25,nr=37,r_max=8.):
    solved=solve_v1(amplitude,nz=nz,nr=nr,r_max=r_max,tolerance=1e-9)
    _,chi_r,chi_z=scalar_pulse(solved["z"],solved["r"],amplitude)
    matrix=v1_jacobian(solved["psi"],solved["z"],solved["r"],chi_r,chi_z)
    values,vectors=eigs(matrix,k=1,sigma=0,which="LM")
    vector=vectors[:,0].real
    background=np.repeat((1/solved["z"])[:,None],nr,axis=1)
    deformation=(solved["psi"]-background).ravel()
    correlation=abs(np.dot(deformation,vector))/(np.linalg.norm(deformation)*np.linalg.norm(vector))
    rel=solved["psi"]/background-1
    maximum=np.unravel_index(np.argmax(np.abs(rel)),rel.shape)
    return {
        "converged":solved["converged"],
        "near_zero_eigenvalue":float(values[0].real),
        "mode_deformation_correlation":float(correlation),
        "max_deformation":float(rel[maximum]),
        "max_index":[int(maximum[0]),int(maximum[1])],
        "max_y":float(np.log(solved["z"][maximum[0]])),
        "max_r":float(solved["r"][maximum[1]]),
    }
