"""Independent finite-difference/Newton spanning-surface finder."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline,RectBivariateSpline
from scipy.optimize import root


def spanning_fd_residual(radius,z,spline):
    radius=np.asarray(radius);spacing=z[1]-z[0];out=np.zeros_like(radius)
    out[0]=(-3*radius[0]+4*radius[1]-radius[2])/(2*spacing)
    out[-1]=(3*radius[-1]-4*radius[-2]+radius[-3])/(2*spacing)
    slope=(radius[2:]-radius[:-2])/(2*spacing);second=(radius[2:]-2*radius[1:-1]+radius[:-2])/spacing**2
    local=radius[1:-1];psi=spline.ev(z[1:-1],local)
    psi_z=spline.ev(z[1:-1],local,dx=1,dy=0);psi_r=spline.ev(z[1:-1],local,dx=0,dy=1)
    out[1:-1]=second-(1+slope**2)*(2/local+3*(psi_r-slope*psi_z)/psi)
    return out


def solve_spanning_surface_fd(z,r,psi,guess,nodes=81,tolerance=1e-10,max_evaluations=5000):
    z=np.asarray(z);r=np.asarray(r);mesh=np.linspace(z[0],z[-1],nodes)
    spline=RectBivariateSpline(z,r,psi,kx=min(3,len(z)-1),ky=min(3,len(r)-1))
    initial=np.full(nodes,float(guess));solved=root(
        lambda values:spanning_fd_residual(values,mesh,spline),initial,
        method="hybr",options={"xtol":tolerance,"maxfev":max_evaluations},
    )
    radius=solved.x;discrete=float(np.max(np.abs(spanning_fd_residual(radius,mesh,spline))))
    in_domain=bool(np.min(radius)>2*r[1] and np.max(radius)<.9*r[-1])
    profile=CubicSpline(mesh,radius,bc_type=((1,0.),(1,0.)));dense=np.linspace(z[0],z[-1],400)
    values=profile(dense);slope=profile(dense,1);second=profile(dense,2)
    sampled=spline.ev(dense,values);psi_z=spline.ev(dense,values,dx=1,dy=0);psi_r=spline.ev(dense,values,dx=0,dy=1)
    rhs=(1+slope**2)*(2/values+3*(psi_r-slope*psi_z)/sampled)
    return {
        "converged":bool(solved.success and in_domain and discrete<max(1e-8,100*tolerance)),
        "solver_success":bool(solved.success),"in_domain":in_domain,"message":str(solved.message),
        "radius_A":float(radius[0]),"radius_B":float(radius[-1]),
        "radius_min":float(np.min(radius)),"radius_max":float(np.max(radius)),
        "discrete_residual_max":discrete,"continuous_defect_interior_max":float(np.max(np.abs(second-rhs)[2:-2])),
        "nodes":int(nodes),"function_evaluations":int(solved.nfev),
    }


def find_spanning_surfaces_fd(z,r,psi,guesses=None,nodes=81,tolerance=1e-10):
    if guesses is None:guesses=np.linspace(max(3*r[1],.1),.75*r[-1],18)
    trials=[];accepted=[]
    for guess in guesses:
        item=solve_spanning_surface_fd(z,r,psi,float(guess),nodes,tolerance);item["guess"]=float(guess);trials.append(item)
        if item["converged"]:
            signature=np.array([item["radius_A"],item["radius_B"],item["radius_min"],item["radius_max"]])
            if not any(np.linalg.norm(signature-np.array([old["radius_A"],old["radius_B"],old["radius_min"],old["radius_max"]]))<5e-3 for old in accepted):
                accepted.append(item)
    return {"spanning_surface_found":bool(accepted),"accepted":accepted,"trials":trials}
