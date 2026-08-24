"""Closed bulk S3 minimal-surface finder in the SO(3)-symmetric slice."""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson,solve_bvp
from scipy.interpolate import RectBivariateSpline

from bhps.capped_surface import capped_surface_equation


def _smooth_axis_boundaries(left,right):
    return np.array((left[1],right[1]))


def solve_closed_profile(z,r,psi,z_center,initial,tolerance=1e-7,nodes=220):
    """Solve a star-shaped closed surface about a trial compact center."""
    z=np.asarray(z);r=np.asarray(r)
    spline=RectBivariateSpline(z,r,psi,kx=min(3,len(z)-1),ky=min(3,len(r)-1))
    theta=np.linspace(1e-4,np.pi-1e-4,nodes)
    if np.isscalar(initial):
        state=np.vstack((np.full_like(theta,float(initial)),np.zeros_like(theta)))
    else:
        state=np.vstack((
            np.interp(theta,np.asarray(initial["theta"]),np.asarray(initial["rho"])),
            np.interp(theta,np.asarray(initial["theta"]),np.asarray(initial["slope"])),
        ))
    solved=solve_bvp(
        lambda angle,values:capped_surface_equation(spline,float(z_center),angle,values),
        _smooth_axis_boundaries,theta,state,tol=tolerance,max_nodes=8000,
    )
    dense=np.linspace(theta[0],theta[-1],500);values=solved.sol(dense)
    derivative=solved.sol(dense,1);rhs=capped_surface_equation(spline,float(z_center),dense,values)
    rho,slope=values;radius=rho*np.sin(dense);zcoord=float(z_center)-rho*np.cos(dense)
    sampled=spline.ev(zcoord,radius)
    dz=z[1]-z[0];dr=r[1]-r[0]
    in_domain=bool(
        np.min(rho)>2*dr and np.min(zcoord)>z[0]+2*dz and np.max(zcoord)<z[-1]-2*dz
        and np.max(radius)<.9*r[-1]
    )
    area=float(simpson(4*np.pi*sampled**3*radius**2*np.sqrt(rho**2+slope**2),x=dense))
    return {
        "converged":bool(solved.success and in_domain),"solver_success":bool(solved.success),"in_domain":in_domain,
        "z_center":float(z_center),"theta":dense,"rho":rho,"slope":slope,
        "z_lower_tip":float(zcoord[0]),"z_upper_tip":float(zcoord[-1]),
        "radius_max":float(np.max(radius)),"rho_min":float(np.min(rho)),"rho_max":float(np.max(rho)),
        "area":area,"boundary_slope_error":float(max(abs(slope[0]),abs(slope[-1]))),
        "surface_residual_max":float(np.max(np.abs(derivative-rhs))),
    }


def find_closed_surfaces(z,r,psi,centers=None,guesses=None,tolerance=2e-5):
    """Search closed bulk surfaces over polar centers and radius seeds."""
    z=np.asarray(z);r=np.asarray(r);dz=z[1]-z[0]
    if centers is None:centers=np.linspace(z[0]+4*dz,z[-1]-4*dz,7)
    if guesses is None:
        maximum=min(.45*(z[-1]-z[0]),.5*r[-1])
        guesses=np.linspace(max(3*r[1],.08),maximum,10)
    trials=[];accepted=[]
    for center in centers:
        for guess in guesses:
            item=solve_closed_profile(z,r,psi,float(center),float(guess),tolerance=tolerance)
            item["guess"]=float(guess);trials.append(item)
            if item["converged"] and item["boundary_slope_error"]<10*tolerance:
                signature=np.array((item["z_lower_tip"],item["z_upper_tip"],item["radius_max"]))
                if not any(np.linalg.norm(signature-np.array((x["z_lower_tip"],x["z_upper_tip"],x["radius_max"])))<5e-3 for x in accepted):
                    accepted.append(item)
    return {
        "closed_surface_found":bool(accepted),"accepted":accepted,"trial_count":len(trials),
        "successful_trials":sum(item["solver_success"] for item in trials),
        "in_domain_successful_trials":sum(item["solver_success"] and item["in_domain"] for item in trials),
        "trials":trials,
    }
