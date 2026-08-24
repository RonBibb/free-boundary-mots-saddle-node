"""Independent nodal solver for anisotropic donor-capped surfaces."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import root

from bhps.anisotropic_capped_surface import _splines,anisotropic_rho_second


def anisotropic_capped_fd_residual(rho,theta,z_b,splines):
    rho=np.asarray(rho);theta=np.asarray(theta);spacing=theta[1]-theta[0]
    residual=np.zeros_like(rho)
    residual[0]=(-3*rho[0]+4*rho[1]-rho[2])/(2*spacing)
    residual[-1]=(3*rho[-1]-4*rho[-2]+rho[-3])/(2*spacing)
    slope=(rho[2:]-rho[:-2])/(2*spacing)
    second=(rho[2:]-2*rho[1:-1]+rho[:-2])/spacing**2
    residual[1:-1]=second-anisotropic_rho_second(
        theta[1:-1],rho[1:-1],slope,z_b,splines,
    )
    return residual


def solve_anisotropic_capped_surface_fd(
    z,r,psi,a,b,c,initial,nodes=81,tolerance=1e-10,max_evaluations=6000,
):
    z=np.asarray(z);r=np.asarray(r);theta=np.linspace(1e-4,np.pi/2,nodes)
    z_b=float(z[-1]);splines=_splines(z,r,psi,a,b,c)
    if np.isscalar(initial):guess=np.full(nodes,float(initial))
    else:guess=np.interp(theta,np.asarray(initial["theta"]),np.asarray(initial["rho"]))
    solved=root(
        lambda values:anisotropic_capped_fd_residual(values,theta,z_b,splines),
        guess,method="hybr",options={"xtol":tolerance,"maxfev":max_evaluations},
    )
    rho=solved.x
    discrete=float(np.max(np.abs(anisotropic_capped_fd_residual(rho,theta,z_b,splines))))
    profile=CubicSpline(theta,rho,bc_type=((1,0.),(1,0.)))
    dense=np.linspace(theta[0],theta[-1],500);dense_rho=profile(dense)
    dense_slope=profile(dense,1);dense_second=profile(dense,2)
    radius=dense_rho*np.sin(dense);zcoord=z_b-dense_rho*np.cos(dense)
    rhs=anisotropic_rho_second(dense,dense_rho,dense_slope,z_b,splines)
    in_domain=bool(
        np.min(dense_rho)>2*r[1] and np.min(zcoord)>z[0]+2*(z[1]-z[0])
        and np.max(radius)<.9*r[-1]
    )
    return {
        "converged":bool(solved.success and in_domain and discrete<max(1e-8,100*tolerance)),
        "solver_success":bool(solved.success),"message":str(solved.message),
        "in_domain":in_domain,"theta":dense,"rho":dense_rho,"slope":dense_slope,
        "nodal_theta":theta,"nodal_rho":rho,
        "rho_axis":float(dense_rho[0]),"rho_brane":float(dense_rho[-1]),
        "rho_min":float(np.min(dense_rho)),"rho_max":float(np.max(dense_rho)),
        "discrete_residual_max":discrete,
        "continuous_defect_interior_max":float(np.max(np.abs(dense_second-rhs)[2:-2])),
        "function_evaluations":int(solved.nfev),"nodes":int(nodes),
    }
