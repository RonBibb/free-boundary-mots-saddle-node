"""Independent finite-difference/Newton finder for closed bulk S3 surfaces."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline,RectBivariateSpline
from scipy.optimize import root

from bhps.capped_surface import _rho_second


def closed_fd_residual(rho,theta,spline,z_center):
    rho=np.asarray(rho);theta=np.asarray(theta);spacing=theta[1]-theta[0]
    residual=np.zeros_like(rho)
    residual[0]=(-3*rho[0]+4*rho[1]-rho[2])/(2*spacing)
    residual[-1]=(3*rho[-1]-4*rho[-2]+rho[-3])/(2*spacing)
    slope=(rho[2:]-rho[:-2])/(2*spacing)
    second=(rho[2:]-2*rho[1:-1]+rho[:-2])/spacing**2
    angle=theta[1:-1];local=rho[1:-1]
    radius=local*np.sin(angle);zcoord=float(z_center)-local*np.cos(angle)
    psi=spline.ev(zcoord,radius);psi_z=spline.ev(zcoord,radius,dx=1,dy=0);psi_r=spline.ev(zcoord,radius,dx=0,dy=1)
    residual[1:-1]=second-_rho_second(angle,local,slope,psi,psi_r,psi_z)
    return residual


def solve_closed_surface_fd(z,r,psi,z_center,initial,nodes=121,tolerance=1e-10,max_evaluations=6000):
    z=np.asarray(z);r=np.asarray(r);theta=np.linspace(1e-4,np.pi-1e-4,nodes)
    spline=RectBivariateSpline(z,r,psi,kx=min(3,len(z)-1),ky=min(3,len(r)-1))
    guess=np.full(nodes,float(initial)) if np.isscalar(initial) else np.interp(theta,np.asarray(initial["theta"]),np.asarray(initial["rho"]))
    solved=root(
        lambda values:closed_fd_residual(values,theta,spline,z_center),guess,
        method="hybr",options={"xtol":tolerance,"maxfev":max_evaluations},
    )
    rho=solved.x;discrete=float(np.max(np.abs(closed_fd_residual(rho,theta,spline,z_center))))
    profile=CubicSpline(theta,rho,bc_type=((1,0.),(1,0.)));dense=np.linspace(theta[0],theta[-1],500)
    dense_rho=profile(dense);slope=profile(dense,1);second=profile(dense,2)
    radius=dense_rho*np.sin(dense);zcoord=float(z_center)-dense_rho*np.cos(dense)
    sampled=spline.ev(zcoord,radius);psi_z=spline.ev(zcoord,radius,dx=1,dy=0);psi_r=spline.ev(zcoord,radius,dx=0,dy=1)
    rhs=_rho_second(dense,dense_rho,slope,sampled,psi_r,psi_z)
    dz=z[1]-z[0];dr=r[1]-r[0]
    in_domain=bool(
        np.min(dense_rho)>2*dr and np.min(zcoord)>z[0]+2*dz and np.max(zcoord)<z[-1]-2*dz
        and np.max(radius)<.9*r[-1]
    )
    return {
        "converged":bool(solved.success and in_domain and discrete<max(1e-8,100*tolerance)),
        "solver_success":bool(solved.success),"in_domain":in_domain,"message":str(solved.message),
        "z_center":float(z_center),"theta":dense,"rho":dense_rho,"slope":slope,
        "z_lower_tip":float(zcoord[0]),"z_upper_tip":float(zcoord[-1]),"radius_max":float(np.max(radius)),
        "rho_min":float(np.min(dense_rho)),"rho_max":float(np.max(dense_rho)),
        "discrete_residual_max":discrete,"continuous_defect_interior_max":float(np.max(np.abs(second-rhs)[2:-2])),
        "function_evaluations":int(solved.nfev),"nodes":int(nodes),
    }


def find_closed_surfaces_fd(z,r,psi,centers=None,guesses=None,nodes=121,tolerance=1e-10):
    z=np.asarray(z);r=np.asarray(r);dz=z[1]-z[0]
    if centers is None:centers=np.linspace(z[0]+4*dz,z[-1]-4*dz,7)
    if guesses is None:
        maximum=min(.45*(z[-1]-z[0]),.5*r[-1]);guesses=np.linspace(max(3*r[1],.08),maximum,10)
    trials=[];accepted=[]
    for center in centers:
        for guess in guesses:
            item=solve_closed_surface_fd(z,r,psi,float(center),float(guess),nodes=nodes,tolerance=tolerance)
            item["guess"]=float(guess);trials.append(item)
            if item["converged"]:
                signature=np.array((item["z_lower_tip"],item["z_upper_tip"],item["radius_max"]))
                if not any(np.linalg.norm(signature-np.array((x["z_lower_tip"],x["z_upper_tip"],x["radius_max"])))<5e-3 for x in accepted):accepted.append(item)
    return {
        "closed_surface_found":bool(accepted),"accepted":accepted,"trial_count":len(trials),
        "successful_trials":sum(item["solver_success"] for item in trials),
        "in_domain_successful_trials":sum(item["solver_success"] and item["in_domain"] for item in trials),
        "trials":trials,
    }
