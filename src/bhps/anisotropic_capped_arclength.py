"""Pseudo-arclength continuation for anisotropic capped surfaces."""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid,solve_bvp
from scipy.interpolate import RectBivariateSpline

from bhps.anisotropic_capped_surface import anisotropic_rho_second


class _InterpolatedSpline:
    def __init__(self,family,amplitude,name):
        self.family=family;self.amplitude=float(amplitude);self.name=name

    def ev(self,zcoord,radius,dx=0,dy=0):
        return self.family.evaluate(self.name,self.amplitude,zcoord,radius,dx,dy)


class AnisotropicMetricFamily:
    """Piecewise-linear amplitude family of physical scale factors A,B,C."""
    def __init__(self,amplitudes,z,r,psi_values,a_values,b_values,c_values):
        self.amplitudes=np.asarray(amplitudes,dtype=float)
        self.z=np.asarray(z,dtype=float);self.r=np.asarray(r,dtype=float)
        if len(self.amplitudes)<2 or np.any(np.diff(self.amplitudes)<=0):
            raise ValueError("amplitudes must be strictly increasing")
        shape=(len(self.amplitudes),len(self.z),len(self.r))
        inputs={
            "A":np.asarray(psi_values)*np.exp(np.asarray(a_values)),
            "B":np.asarray(psi_values)*np.exp(np.asarray(b_values)),
            "C":np.asarray(psi_values)*np.exp(np.asarray(c_values)),
        }
        if any(value.shape!=shape for value in inputs.values()):
            raise ValueError("metric family fields have the wrong shape")
        self.splines={name:[
            RectBivariateSpline(
                self.z,self.r,value,kx=min(3,len(self.z)-1),ky=min(3,len(self.r)-1),
            ) for value in values
        ] for name,values in inputs.items()}

    def evaluate(self,name,amplitude,zcoord,radius,dx=0,dy=0):
        amplitude=float(amplitude)
        if amplitude<self.amplitudes[0] or amplitude>self.amplitudes[-1]:
            raise ValueError("pseudo-arclength step left the metric-family range")
        upper=int(np.searchsorted(self.amplitudes,amplitude,side="right"))
        upper=min(max(upper,1),len(self.amplitudes)-1);lower=upper-1
        weight=(amplitude-self.amplitudes[lower])/(self.amplitudes[upper]-self.amplitudes[lower])
        zcoord=np.clip(zcoord,self.z[0],self.z[-1]);radius=np.clip(radius,self.r[0],self.r[-1])
        left=self.splines[name][lower].ev(zcoord,radius,dx=dx,dy=dy)
        right=self.splines[name][upper].ev(zcoord,radius,dx=dx,dy=dy)
        return (1-weight)*left+weight*right

    def at(self,amplitude):
        return {name:_InterpolatedSpline(self,amplitude,name) for name in ("A","B","C")}


def _profile_on(profile,theta):
    return np.interp(theta,np.asarray(profile["theta"]),np.asarray(profile["rho"]))


def pseudo_arclength_step(
    family,previous,current,step,tolerance=2e-6,nodes=180,max_nodes=8000,
):
    theta=np.linspace(1e-4,np.pi/2,int(nodes));length=theta[-1]-theta[0]
    rho_previous=_profile_on(previous,theta);rho_current=_profile_on(current,theta)
    delta_rho=rho_current-rho_previous
    delta_amplitude=float(current["amplitude"]-previous["amplitude"])
    norm=float(np.sqrt(np.trapezoid(delta_rho**2,x=theta)/length+delta_amplitude**2))
    if not np.isfinite(norm) or norm<=0:raise ValueError("continuation points must be distinct")
    tangent_rho=delta_rho/norm;tangent_amplitude=delta_amplitude/norm
    predicted_rho=rho_current+step*tangent_rho
    predicted_amplitude=float(current["amplitude"]+step*tangent_amplitude)
    predicted_slope=np.gradient(predicted_rho,theta,edge_order=2)
    auxiliary=np.concatenate(([0.],cumulative_trapezoid(
        (predicted_rho-rho_current)*tangent_rho/length,x=theta,
    )))
    state=np.vstack((predicted_rho,predicted_slope,auxiliary))

    def equation(angle,values,parameter):
        rho,slope=values[:2]
        second=anisotropic_rho_second(
            angle,rho,slope,family.z[-1],family.at(parameter[0]),
        )
        reference=np.interp(angle,theta,rho_current)
        tangent=np.interp(angle,theta,tangent_rho)
        return np.vstack((slope,second,(rho-reference)*tangent/length))

    def boundaries(left,right,parameter):
        return np.array([
            left[1],right[1],left[2],
            right[2]+tangent_amplitude*(parameter[0]-current["amplitude"])-step,
        ])

    try:
        solved=solve_bvp(
            equation,boundaries,theta,state,p=np.array([predicted_amplitude]),
            tol=tolerance,max_nodes=int(max_nodes),
        )
    except ValueError as error:
        return {"converged":False,"message":str(error),"amplitude":predicted_amplitude}
    dense=np.linspace(theta[0],theta[-1],400);values=solved.sol(dense)
    rho,slope=values[:2];radius=rho*np.sin(dense)
    zcoord=family.z[-1]-rho*np.cos(dense)
    in_domain=bool(
        np.min(rho)>2*family.r[1]
        and np.min(zcoord)>family.z[0]+2*(family.z[1]-family.z[0])
        and np.max(radius)<.9*family.r[-1]
    )
    return {
        "converged":bool(solved.success and in_domain),
        "solver_success":bool(solved.success),"in_domain":in_domain,
        "message":str(solved.message),"amplitude":float(solved.p[0]),
        "theta":dense,"rho":rho,"slope":slope,
        "rho_axis":float(rho[0]),"rho_brane":float(rho[-1]),
        "rho_min":float(np.min(rho)),"rho_max":float(np.max(rho)),
        "boundary_slope_error":float(max(abs(slope[0]),abs(slope[-1]))),
        "arclength_residual":float(abs(
            values[2,-1]+tangent_amplitude*(solved.p[0]-current["amplitude"])-step
        )),
        "mesh_nodes":int(len(solved.x)),
    }


def continue_capped_arclength(family,first,second,step,count,**kwargs):
    points=[first,second]
    for _ in range(int(count)):
        candidate=pseudo_arclength_step(family,points[-2],points[-1],step,**kwargs)
        points.append(candidate)
        if not candidate["converged"]:break
    return points
