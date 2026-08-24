"""Profile-following utilities for the capped-surface pair."""

from __future__ import annotations

import numpy as np
from scipy.integrate import simpson,solve_bvp
from scipy.interpolate import RectBivariateSpline

from bhps.capped_surface import _smooth_free_boundaries,capped_surface_equation


def solve_capped_profile(z,r,psi,initial,tolerance=1e-7,nodes=160):
    spline=RectBivariateSpline(z,r,psi,kx=min(3,len(z)-1),ky=min(3,len(r)-1))
    theta=np.linspace(1e-4,np.pi/2,nodes);z_b=float(z[-1])
    if np.isscalar(initial):
        state=np.vstack((np.full_like(theta,float(initial)),np.zeros_like(theta)))
    else:
        old_theta=np.asarray(initial["theta"]);state=np.vstack((
            np.interp(theta,old_theta,np.asarray(initial["rho"])),
            np.interp(theta,old_theta,np.asarray(initial["slope"])),
        ))
    solved=solve_bvp(
        lambda angle,values:capped_surface_equation(spline,z_b,angle,values),
        _smooth_free_boundaries,theta,state,tol=tolerance,max_nodes=6000,
    )
    dense=np.linspace(theta[0],theta[-1],400);values=solved.sol(dense)
    derivative=solved.sol(dense,1);rhs=capped_surface_equation(spline,z_b,dense,values)
    rho,slope=values;radius=rho*np.sin(dense);zcoord=z_b-rho*np.cos(dense)
    sampled_psi=spline.ev(zcoord,radius)
    area=float(simpson(4*np.pi*sampled_psi**3*radius**2*np.sqrt(rho**2+slope**2),x=dense))
    in_domain=bool(np.min(rho)>2*r[1] and np.min(zcoord)>z[0]+2*(z[1]-z[0]) and np.max(radius)<.9*r[-1])
    return {
        "converged":bool(solved.success and in_domain),
        "solver_success":bool(solved.success),
        "in_domain":in_domain,
        "theta":dense,
        "rho":rho,
        "slope":slope,
        "rho_axis":float(rho[0]),
        "rho_brane":float(rho[-1]),
        "rho_min":float(np.min(rho)),
        "rho_max":float(np.max(rho)),
        "area":area,
        "surface_residual_max":float(np.max(np.abs(derivative-rhs))),
        "boundary_slope_error":float(max(abs(slope[0]),abs(slope[-1]))),
    }


def fit_fold_normal_form(records,tail=12):
    """Fit (rho_B,out-rho_B,in)^2 = slope*(A-A_fold)."""
    usable=[item for item in records if item["pair_converged"] and item["radius_separation"]>1e-5]
    usable=sorted(usable,key=lambda item:item["amplitude"])[:tail]
    if len(usable)<3:
        raise ValueError("at least three paired points are required")
    amplitude=np.array([item["amplitude"] for item in usable])
    squared=np.array([item["radius_separation"]**2 for item in usable])
    slope,intercept=np.polyfit(amplitude,squared,1)
    prediction=slope*amplitude+intercept
    ss_res=float(np.sum((squared-prediction)**2));ss_tot=float(np.sum((squared-np.mean(squared))**2))
    return {
        "fold_amplitude":float(-intercept/slope),
        "normal_form_slope":float(slope),
        "fit_r_squared":float(1-ss_res/ss_tot if ss_tot else 1),
        "fit_point_count":len(usable),
        "fit_amplitude_min":float(np.min(amplitude)),
        "fit_amplitude_max":float(np.max(amplitude)),
    }
