"""Independent finite-difference/Newton solver for donor-capped surfaces."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline,RectBivariateSpline
from scipy.optimize import root

from bhps.capped_surface import _rho_second


def capped_fd_residual(rho,theta,spline,z_b):
    """Second-order nodal residual with one-sided Neumann boundaries."""
    rho=np.asarray(rho);theta=np.asarray(theta);spacing=theta[1]-theta[0]
    residual=np.zeros_like(rho)
    residual[0]=(-3*rho[0]+4*rho[1]-rho[2])/(2*spacing)
    residual[-1]=(3*rho[-1]-4*rho[-2]+rho[-3])/(2*spacing)
    slope=(rho[2:]-rho[:-2])/(2*spacing)
    second=(rho[2:]-2*rho[1:-1]+rho[:-2])/spacing**2
    angle=theta[1:-1];local=rho[1:-1]
    radius=local*np.sin(angle);zcoord=z_b-local*np.cos(angle)
    psi=spline.ev(zcoord,radius)
    psi_z=spline.ev(zcoord,radius,dx=1,dy=0)
    psi_r=spline.ev(zcoord,radius,dx=0,dy=1)
    residual[1:-1]=second-_rho_second(angle,local,slope,psi,psi_r,psi_z)
    return residual


def solve_capped_surface_fd(z,r,psi,initial,nodes=81,tolerance=1e-10,max_evaluations=5000):
    """Solve the cap equation without ``solve_bvp`` collocation."""
    z=np.asarray(z);r=np.asarray(r);theta=np.linspace(1e-4,np.pi/2,nodes);z_b=float(z[-1])
    spline=RectBivariateSpline(z,r,psi,kx=min(3,len(z)-1),ky=min(3,len(r)-1))
    if np.isscalar(initial):
        guess=np.full(nodes,float(initial))
    else:
        guess=np.interp(theta,np.asarray(initial["theta"]),np.asarray(initial["rho"]))
    solved=root(
        lambda values:capped_fd_residual(values,theta,spline,z_b),guess,
        method="hybr",options={"xtol":tolerance,"maxfev":max_evaluations},
    )
    rho=solved.x;discrete=float(np.max(np.abs(capped_fd_residual(rho,theta,spline,z_b))))
    profile=CubicSpline(theta,rho,bc_type=((1,0.),(1,0.)))
    dense=np.linspace(theta[0],theta[-1],400)
    dense_rho=profile(dense);dense_slope=profile(dense,1);dense_second=profile(dense,2)
    radius=dense_rho*np.sin(dense);zcoord=z_b-dense_rho*np.cos(dense)
    sampled_psi=spline.ev(zcoord,radius)
    psi_z=spline.ev(zcoord,radius,dx=1,dy=0);psi_r=spline.ev(zcoord,radius,dx=0,dy=1)
    rhs=_rho_second(dense,dense_rho,dense_slope,sampled_psi,psi_r,psi_z)
    in_domain=bool(
        np.min(dense_rho)>2*r[1] and np.min(zcoord)>z[0]+2*(z[1]-z[0])
        and np.max(radius)<.9*r[-1]
    )
    return {
        "converged":bool(solved.success and in_domain and discrete<max(1e-8,100*tolerance)),
        "solver_success":bool(solved.success),"message":str(solved.message),"in_domain":in_domain,
        "theta":dense,"rho":dense_rho,"slope":dense_slope,
        "nodal_theta":theta,"nodal_rho":rho,
        "rho_axis":float(dense_rho[0]),"rho_brane":float(dense_rho[-1]),
        "rho_min":float(np.min(dense_rho)),"rho_max":float(np.max(dense_rho)),
        "discrete_residual_max":discrete,
        "continuous_defect_interior_max":float(np.max(np.abs(dense_second-rhs)[2:-2])),
        "function_evaluations":int(solved.nfev),"nodes":int(nodes),
    }


def capped_fd_jacobian_diagnostic(z,r,psi,solution,relative_step=1e-6):
    """Return the smallest singular direction of the discrete cap residual."""
    z=np.asarray(z);r=np.asarray(r);theta=np.asarray(solution["nodal_theta"])
    rho=np.asarray(solution["nodal_rho"]);z_b=float(z[-1])
    spline=RectBivariateSpline(z,r,psi,kx=min(3,len(z)-1),ky=min(3,len(r)-1))
    step=relative_step*max(1.,float(np.mean(rho)));jacobian=np.empty((len(rho),len(rho)))
    for column in range(len(rho)):
        shift=np.zeros_like(rho);shift[column]=step
        jacobian[:,column]=(
            capped_fd_residual(rho+shift,theta,spline,z_b)
            -capped_fd_residual(rho-shift,theta,spline,z_b)
        )/(2*step)
    _,singular_values,right=np.linalg.svd(jacobian,full_matrices=False)
    vector=right[-1]
    return {
        "smallest_singular_value":float(singular_values[-1]),
        "next_singular_value":float(singular_values[-2]),
        "largest_singular_value":float(singular_values[0]),
        "relative_smallest_singular_value":float(singular_values[-1]/singular_values[0]),
        "null_residual":float(np.linalg.norm(jacobian@vector)),
        "relative_step":float(step),"right_singular_vector":vector,
    }
