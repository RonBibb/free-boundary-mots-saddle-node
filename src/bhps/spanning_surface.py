"""Interval-spanning S2 x I minimal-surface finder."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.integrate import solve_bvp
from scipy.interpolate import RectBivariateSpline


def spanning_surface_equation(spline,z,state):
    radius,slope=state
    z_knots,r_knots=spline.get_knots()
    sample_r=np.clip(radius,r_knots[0],r_knots[-1])
    psi=spline.ev(z,sample_r)
    psi_z=spline.ev(z,sample_r,dx=1,dy=0)
    psi_r=spline.ev(z,sample_r,dx=0,dy=1)
    safe_radius=np.maximum(radius,1e-10)
    second=(1+slope**2)*(2/safe_radius+3*(psi_r-slope*psi_z)/psi)
    return np.vstack((slope,second))


def _orthogonal_boundaries(left,right):
    return np.array((left[1],right[1]))


def solve_spanning_profile(z,r,psi,initial,tolerance=1e-7,nodes=160):
    """Solve one spanning profile, optionally seeded by a previous profile."""
    z=np.asarray(z);r=np.asarray(r);spline=RectBivariateSpline(z,r,psi,kx=min(3,len(z)-1),ky=min(3,len(r)-1))
    mesh=np.linspace(z[0],z[-1],nodes)
    if np.isscalar(initial):state=np.vstack((np.full_like(mesh,float(initial)),np.zeros_like(mesh)))
    else:
        old_z=np.asarray(initial["z"]);state=np.vstack((
            np.interp(mesh,old_z,np.asarray(initial["radius"])),
            np.interp(mesh,old_z,np.asarray(initial["slope"])),
        ))
    solved=solve_bvp(
        lambda compact,values:spanning_surface_equation(spline,compact,values),
        _orthogonal_boundaries,mesh,state,tol=tolerance,max_nodes=6000,
    )
    dense=np.linspace(z[0],z[-1],400);values=solved.sol(dense);derivative=solved.sol(dense,1)
    rhs=spanning_surface_equation(spline,dense,values);radius,slope=values;sampled=spline.ev(dense,radius)
    in_domain=bool(np.min(radius)>2*r[1] and np.max(radius)<.9*r[-1])
    area=float(np.trapezoid(4*np.pi*sampled**3*radius**2*np.sqrt(1+slope**2),x=dense))
    return {
        "converged":bool(solved.success and in_domain),"solver_success":bool(solved.success),"in_domain":in_domain,
        "z":dense,"radius":radius,"slope":slope,"radius_A":float(radius[0]),"radius_B":float(radius[-1]),
        "radius_min":float(np.min(radius)),"radius_max":float(np.max(radius)),"area":area,
        "boundary_slope_error":float(max(abs(slope[0]),abs(slope[-1]))),
        "surface_residual_max":float(np.max(np.abs(derivative-rhs))),
    }


def spanning_area_stability(spline,solution,nodes=41,relative_step=2.5e-5,maximum_angular_mode=3):
    """Discrete second variation and angular-mode inertia of a spanning surface."""
    knots_z,_=spline.get_knots();z=np.linspace(knots_z[0],knots_z[-1],nodes);radius=solution.sol(z)[0]
    mid=.5*(z[:-1]+z[1:]);spacing=z[1]-z[0]
    def area(values):
        local=.5*(values[:-1]+values[1:]);slope=(values[1:]-values[:-1])/spacing
        psi=spline.ev(mid,local)
        return float(np.sum(4*np.pi*spacing*psi**3*local**2*np.sqrt(1+slope**2)))
    step=relative_step*max(1.,float(np.mean(radius)));base=area(radius);hessian=np.zeros((nodes,nodes))
    for i in range(nodes):
        shift=np.zeros(nodes);shift[i]=step
        hessian[i,i]=(area(radius+shift)-2*base+area(radius-shift))/step**2
        for j in range(i):
            other=np.zeros(nodes);other[j]=step
            value=(area(radius+shift+other)-area(radius+shift-other)-area(radius-shift+other)+area(radius-shift-other))/(4*step**2)
            hessian[i,j]=hessian[j,i]=value
    local=.5*(radius[:-1]+radius[1:]);slope=(radius[1:]-radius[:-1])/spacing;psi=spline.ev(mid,local)
    segment_mass=4*np.pi*spacing*psi**5*local**2/np.sqrt(1+slope**2)
    mass=np.zeros(nodes);mass[:-1]+=.5*segment_mass;mass[1:]+=.5*segment_mass;mass_matrix=np.diag(mass)
    angular=[]
    for mode in range(maximum_angular_mode+1):
        addition=np.zeros(nodes)
        if mode:
            segment=segment_mass*mode*(mode+1)/(psi**2*local**2)
            addition[:-1]+=.5*segment;addition[1:]+=.5*segment
        values=eigh(hessian+np.diag(addition),mass_matrix,eigvals_only=True)
        angular.append({
            "angular_mode":mode,"lowest_normalized_eigenvalue":float(values[0]),
            "negative_mode_count":int(np.sum(values<-1e-8*max(float(np.max(np.abs(values))),1.))),
        })
    return {"area":base,"area_hessian_nodes":nodes,"area_hessian_step":step,"angular_mode_spectrum":angular}


def find_spanning_surfaces(z,r,psi,guesses=None,tolerance=2e-5,stability_nodes=41,stability_step=2.5e-5):
    """Search for surfaces r=R(z) meeting both branes orthogonally."""
    spline=RectBivariateSpline(z,r,psi,kx=min(3,len(z)-1),ky=min(3,len(r)-1))
    if guesses is None:
        guesses=tuple(np.linspace(max(3*r[1],.1),.75*r[-1],18))
    mesh=np.linspace(z[0],z[-1],120);trials=[];accepted=[]
    for guess in guesses:
        initial=np.vstack((np.full_like(mesh,guess),np.zeros_like(mesh)))
        solved=solve_bvp(
            lambda compact,state:spanning_surface_equation(spline,compact,state),
            _orthogonal_boundaries,mesh,initial,tol=tolerance,max_nodes=4000,
        )
        radius=solved.y[0];slope=solved.y[1]
        in_domain=bool(np.min(radius)>2*r[1] and np.max(radius)<.9*r[-1])
        dense=np.linspace(z[0],z[-1],400);dense_state=solved.sol(dense);dense_derivative=solved.sol(dense,1)
        dense_rhs=spanning_surface_equation(spline,dense,dense_state)
        item={
            "guess":float(guess),
            "solver_success":bool(solved.success),
            "in_domain":in_domain,
            "radius_A":float(radius[0]),
            "radius_B":float(radius[-1]),
            "radius_min":float(np.min(radius)),
            "radius_max":float(np.max(radius)),
            "boundary_slope_error":float(max(abs(slope[0]),abs(slope[-1]))),
            "surface_residual_max":float(np.max(np.abs(dense_derivative-dense_rhs))),
        }
        trials.append(item)
        if solved.success and in_domain and item["boundary_slope_error"]<10*tolerance:
            signature=np.array((item["radius_A"],item["radius_B"],item["radius_min"],item["radius_max"]))
            if not any(np.linalg.norm(signature-np.array((x["radius_A"],x["radius_B"],x["radius_min"],x["radius_max"])))<5e-3 for x in accepted):
                item.update(spanning_area_stability(spline,solved,stability_nodes,stability_step))
                accepted.append(item)
    return {
        "spanning_surface_found":bool(accepted),
        "accepted":accepted,
        "trial_count":len(trials),
        "successful_trials":sum(item["solver_success"] for item in trials),
        "in_domain_successful_trials":sum(item["solver_success"] and item["in_domain"] for item in trials),
        "trials":trials,
    }


def spanning_maximum_principle_diagnostic(z,r,psi,radial_margin_points=2,refinement=4):
    """Evaluate a sufficient pointwise obstruction to spanning surfaces.

    At a global maximum of a positive Neumann profile ``R(z)``, its equation
    gives ``R''=2/R+3 partial_R(log psi)``.  Strict positivity of this quantity
    on the admissible domain contradicts ``R''<=0`` and therefore excludes a
    finite spanning solution.  The returned result is a sampled numerical
    diagnostic; the minimum and sampling spacings make its margin auditable.
    """
    z=np.asarray(z);r=np.asarray(r);spline=RectBivariateSpline(z,r,psi,kx=min(3,len(z)-1),ky=min(3,len(r)-1))
    dense_z=np.linspace(z[0],z[-1],refinement*(len(z)-1)+1)
    lower=radial_margin_points*r[1];upper=.9*r[-1]
    dense_r=np.linspace(lower,upper,refinement*(len(r)-1)+1)
    value=spline(dense_z,dense_r);derivative=spline(dense_z,dense_r,dx=0,dy=1)
    diagnostic=2/dense_r[None,:]+3*derivative/value;index=np.unravel_index(np.argmin(diagnostic),diagnostic.shape)
    minimum=float(diagnostic[index])
    return {
        "strict_obstruction_on_sampled_domain":bool(minimum>0),
        "minimum_2_over_R_plus_3_dR_logpsi":minimum,
        "minimum_z":float(dense_z[index[0]]),"minimum_r":float(dense_r[index[1]]),
        "sample_dz":float(dense_z[1]-dense_z[0]),"sample_dr":float(dense_r[1]-dense_r[0]),
        "admissible_r_min":float(lower),"admissible_r_max":float(upper),
    }
