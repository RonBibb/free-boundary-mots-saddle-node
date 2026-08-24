"""Capped minimal surfaces in a diagonal anisotropic time-symmetric slice."""

from __future__ import annotations

import math
import numpy as np
from scipy.linalg import eigh
from scipy.integrate import simpson,solve_bvp
from scipy.interpolate import RectBivariateSpline


def _splines(z,r,psi,a,b,c):
    order_z=min(3,len(z)-1);order_r=min(3,len(r)-1)
    return {
        name:RectBivariateSpline(z,r,np.asarray(field),kx=order_z,ky=order_r)
        for name,field in (
            ("A",np.asarray(psi)*np.exp(a)),
            ("B",np.asarray(psi)*np.exp(b)),
            ("C",np.asarray(psi)*np.exp(c)),
        )
    }


def anisotropic_rho_second(theta,rho,slope,z_b,splines):
    """Euler--Lagrange solve for ``rho''`` from the anisotropic area."""
    theta=np.asarray(theta);rho=np.asarray(rho);slope=np.asarray(slope)
    st=np.sin(theta);ct=np.cos(theta);radius=rho*st;zcoord=z_b-rho*ct
    values={}
    for name,spline in splines.items():
        values[name]=spline.ev(zcoord,radius)
        values[name+"z"]=spline.ev(zcoord,radius,dx=1,dy=0)
        values[name+"r"]=spline.ev(zcoord,radius,dx=0,dy=1)
    aa=values["A"]**2;bb=values["B"]**2;cc=values["C"]
    aaz=2*values["A"]*values["Az"];aar=2*values["A"]*values["Ar"]
    bbz=2*values["B"]*values["Bz"];bbr=2*values["B"]*values["Br"]
    aaq=-ct*aaz+st*aar;bbq=-ct*bbz+st*bbr
    aat=rho*(st*aaz+ct*aar);bbt=rho*(st*bbz+ct*bbr)

    h=aa*ct**2+bb*st**2
    j=rho*st*ct*(bb-aa)
    k=rho**2*(aa*st**2+bb*ct**2)
    hq=aaq*ct**2+bbq*st**2
    jq=st*ct*(bb-aa)+rho*st*ct*(bbq-aaq)
    kq=2*rho*(aa*st**2+bb*ct**2)+rho**2*(aaq*st**2+bbq*ct**2)
    ht=aat*ct**2+bbt*st**2+2*st*ct*(bb-aa)
    jt=rho*((ct**2-st**2)*(bb-aa)+st*ct*(bbt-aat))
    kt=rho**2*(aat*st**2+bbt*ct**2+2*st*ct*(aa-bb))

    fiber=cc**2*radius**2
    fiber_z=2*cc*values["Cz"]*radius**2
    fiber_r=2*cc*values["Cr"]*radius**2+2*cc**2*radius
    fiber_q=-ct*fiber_z+st*fiber_r
    fiber_t=rho*(st*fiber_z+ct*fiber_r)

    energy=h*slope**2+2*j*slope+k
    speed=np.sqrt(np.maximum(energy,1e-300));moment=h*slope+j
    energy_q=hq*slope**2+2*jq*slope+kq
    energy_t=ht*slope**2+2*jt*slope+kt
    moment_q=hq*slope+jq;moment_t=ht*slope+jt
    p_t=fiber_t*moment/speed+fiber*moment_t/speed-fiber*moment*energy_t/(2*speed**3)
    p_q=fiber_q*moment/speed+fiber*moment_q/speed-fiber*moment*energy_q/(2*speed**3)
    lagrangian_q=fiber_q*speed+fiber*energy_q/(2*speed)
    p_p=fiber*(h/speed-moment**2/speed**3)
    numerator=lagrangian_q-p_t-p_q*slope
    return np.divide(
        numerator,p_p,out=np.full_like(numerator,np.nan),where=np.abs(p_p)>1e-300,
    )


def anisotropic_capped_equation(splines,z_b,theta,state):
    return np.vstack((
        state[1],
        anisotropic_rho_second(theta,state[0],state[1],z_b,splines),
    ))


def _free_boundaries(left,right):
    return np.array([left[1],right[1]])


def find_anisotropic_donor_capped_surfaces(
    z,r,psi,a,b,c,guesses=None,tolerance=2e-5,
):
    """Search for half-S3 minimal surfaces attached to the upper wall."""
    z=np.asarray(z);r=np.asarray(r);splines=_splines(z,r,psi,a,b,c);z_b=float(z[-1])
    maximum_rho=min(z_b-z[0],.85*r[-1])
    if guesses is None:guesses=tuple(np.linspace(.08*maximum_rho,.9*maximum_rho,12))
    theta=np.linspace(1e-4,math.pi/2,120);trials=[];accepted=[]
    for guess in guesses:
        initial=np.vstack((np.full_like(theta,float(guess)),np.zeros_like(theta)))
        solved=solve_bvp(
            lambda angle,state:anisotropic_capped_equation(splines,z_b,angle,state),
            _free_boundaries,theta,initial,tol=tolerance,max_nodes=4000,
        )
        dense=np.linspace(theta[0],theta[-1],400);state=solved.sol(dense)
        derivative=solved.sol(dense,1);rhs=anisotropic_capped_equation(splines,z_b,dense,state)
        local_rho,local_slope=state;radius=local_rho*np.sin(dense)
        zcoord=z_b-local_rho*np.cos(dense)
        aa=splines["A"].ev(zcoord,radius);bb=splines["B"].ev(zcoord,radius)
        cc=splines["C"].ev(zcoord,radius)
        rp=local_slope*np.sin(dense)+local_rho*np.cos(dense)
        zp=-local_slope*np.cos(dense)+local_rho*np.sin(dense)
        area=4*np.pi*simpson((cc*radius)**2*np.sqrt(aa**2*zp**2+bb**2*rp**2),x=dense)
        in_domain=bool(
            np.min(local_rho)>2*r[1] and np.min(zcoord)>z[0]+2*(z[1]-z[0])
            and np.max(radius)<.9*r[-1]
        )
        item={
            "guess":float(guess),"solver_success":bool(solved.success),
            "message":str(solved.message),"in_domain":in_domain,
            "rho_axis":float(local_rho[0]),"rho_brane":float(local_rho[-1]),
            "rho_min":float(np.min(local_rho)),"rho_max":float(np.max(local_rho)),
            "z_tip":float(zcoord[0]),"brane_radius":float(radius[-1]),
            "boundary_slope_error":float(max(abs(local_slope[0]),abs(local_slope[-1]))),
            "surface_residual_max":float(np.max(np.abs(derivative-rhs))),
            "area":float(area),
        }
        trials.append(item)
        if solved.success and in_domain and item["boundary_slope_error"]<10*tolerance:
            signature=np.array([item["rho_axis"],item["rho_brane"]])
            if not any(np.linalg.norm(signature-np.array([old["rho_axis"],old["rho_brane"]]))<5e-3 for old in accepted):
                accepted.append(item)
    return {
        "capped_surface_found":bool(accepted),"accepted":accepted,
        "trial_count":len(trials),"successful_trials":sum(x["solver_success"] for x in trials),
        "in_domain_successful_trials":sum(x["solver_success"] and x["in_domain"] for x in trials),
        "trials":trials,
    }


def solve_anisotropic_capped_profile(
    z,r,psi,a,b,c,initial,tolerance=2e-6,nodes=160,max_nodes=6000,
):
    """Solve one anisotropic cap from a scalar or prior profile seed."""
    z=np.asarray(z);r=np.asarray(r);splines=_splines(z,r,psi,a,b,c);z_b=float(z[-1])
    theta=np.linspace(1e-4,math.pi/2,int(nodes))
    if np.isscalar(initial):
        state=np.vstack((np.full_like(theta,float(initial)),np.zeros_like(theta)))
    else:
        source=np.asarray(initial["theta"])
        state=np.vstack((
            np.interp(theta,source,np.asarray(initial["rho"])),
            np.interp(theta,source,np.asarray(initial["slope"])),
        ))
    solved=solve_bvp(
        lambda angle,values:anisotropic_capped_equation(splines,z_b,angle,values),
        _free_boundaries,theta,state,tol=tolerance,max_nodes=int(max_nodes),
    )
    dense=np.linspace(theta[0],theta[-1],400);values=solved.sol(dense)
    derivative=solved.sol(dense,1);rhs=anisotropic_capped_equation(splines,z_b,dense,values)
    rho,slope=values;radius=rho*np.sin(dense);zcoord=z_b-rho*np.cos(dense)
    aa=splines["A"].ev(zcoord,radius);bb=splines["B"].ev(zcoord,radius)
    cc=splines["C"].ev(zcoord,radius)
    rp=slope*np.sin(dense)+rho*np.cos(dense)
    zp=-slope*np.cos(dense)+rho*np.sin(dense)
    area=float(4*np.pi*simpson(
        (cc*radius)**2*np.sqrt(aa**2*zp**2+bb**2*rp**2),x=dense,
    ))
    in_domain=bool(
        np.min(rho)>2*r[1] and np.min(zcoord)>z[0]+2*(z[1]-z[0])
        and np.max(radius)<.9*r[-1]
    )
    return {
        "converged":bool(solved.success and in_domain),
        "solver_success":bool(solved.success),"in_domain":in_domain,
        "message":str(solved.message),"theta":dense,"rho":rho,"slope":slope,
        "rho_axis":float(rho[0]),"rho_brane":float(rho[-1]),
        "rho_min":float(np.min(rho)),"rho_max":float(np.max(rho)),
        "area":area,"surface_residual_max":float(np.max(np.abs(derivative-rhs))),
        "boundary_slope_error":float(max(abs(slope[0]),abs(slope[-1]))),
    }


def anisotropic_capped_area_stability(
    z,r,psi,a,b,c,solution,nodes=41,relative_step=2.5e-5,
    maximum_angular_mode=3,
):
    """Discrete anisotropic area Hessian and full-angular mode counts."""
    splines=_splines(z,r,psi,a,b,c);z_b=float(np.asarray(z)[-1])
    theta=np.linspace(1e-4,np.pi/2,int(nodes))
    rho=np.interp(theta,np.asarray(solution["theta"]),np.asarray(solution["rho"]))
    mid=.5*(theta[:-1]+theta[1:]);delta=theta[1]-theta[0]

    def fields(values):
        local=.5*(values[:-1]+values[1:]);slope=(values[1:]-values[:-1])/delta
        radius=local*np.sin(mid);zcoord=z_b-local*np.cos(mid)
        aa=splines["A"].ev(zcoord,radius);bb=splines["B"].ev(zcoord,radius)
        cc=splines["C"].ev(zcoord,radius)
        rp=slope*np.sin(mid)+local*np.cos(mid)
        zp=-slope*np.cos(mid)+local*np.sin(mid)
        speed=np.sqrt(aa**2*zp**2+bb**2*rp**2)
        return local,radius,aa,bb,cc,speed

    def area(values):
        _,radius,_,_,cc,speed=fields(values)
        return float(np.sum(4*np.pi*delta*(cc*radius)**2*speed))

    step=float(relative_step)*max(1.,float(np.mean(rho)));base=area(rho)
    hessian=np.zeros((nodes,nodes))
    for i in range(nodes):
        shift=np.zeros(nodes);shift[i]=step
        hessian[i,i]=(area(rho+shift)-2*base+area(rho-shift))/step**2
        for j in range(i):
            other=np.zeros(nodes);other[j]=step
            value=(
                area(rho+shift+other)-area(rho+shift-other)
                -area(rho-shift+other)+area(rho-shift-other)
            )/(4*step**2)
            hessian[i,j]=hessian[j,i]=value
    eigenvalues=np.linalg.eigvalsh(hessian)
    local,radius,aa,bb,cc,speed=fields(rho)
    segment_mass=4*np.pi*delta*(cc*radius)**2*aa**2*bb**2*local**2/speed
    mass=np.zeros(nodes);mass[:-1]+=.5*segment_mass;mass[1:]+=.5*segment_mass
    mass_matrix=np.diag(mass);normalized=eigh(hessian,mass_matrix,eigvals_only=True)
    angular=[]
    for angular_mode in range(int(maximum_angular_mode)+1):
        if angular_mode==0:values=normalized
        else:
            angular_segment=segment_mass*angular_mode*(angular_mode+1)/(cc*radius)**2
            lumped=np.zeros(nodes);lumped[:-1]+=.5*angular_segment;lumped[1:]+=.5*angular_segment
            values=eigh(
                (hessian+np.diag(lumped))[1:,1:],mass_matrix[1:,1:],eigvals_only=True,
            )
        scale=max(float(np.max(np.abs(values))),1.)
        angular.append({
            "angular_mode":angular_mode,
            "lowest_normalized_eigenvalue":float(values[0]),
            "negative_mode_count":int(np.sum(values < -1e-8*scale)),
        })
    scale=max(float(np.max(np.abs(eigenvalues))),1.)
    return {
        "area_hessian_nodes":int(nodes),"area_hessian_step":step,
        "lowest_area_hessian_eigenvalue":float(eigenvalues[0]),
        "second_area_hessian_eigenvalue":float(eigenvalues[1]),
        "negative_mode_count":int(np.sum(eigenvalues < -1e-8*scale)),
        "lowest_normalized_jacobi_eigenvalue":float(normalized[0]),
        "second_normalized_jacobi_eigenvalue":float(normalized[1]),
        "angular_mode_spectrum":angular,
    }
