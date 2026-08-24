"""Pseudo-arclength continuation for donor-capped marginal surfaces.

The metric is represented by a set of independently solved amplitude slices.
Linear interpolation in amplitude is deliberately used so its error can be
audited by halving the metric-family spacing.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_bvp
from scipy.interpolate import RectBivariateSpline

from bhps.capped_surface import _rho_second


class MetricFamily:
    """Piecewise-linear-in-amplitude family of spatial conformal factors."""

    def __init__(self, amplitudes, z, r, psi_values):
        self.amplitudes=np.asarray(amplitudes,dtype=float)
        self.z=np.asarray(z,dtype=float);self.r=np.asarray(r,dtype=float)
        values=np.asarray(psi_values,dtype=float)
        if values.shape!=(len(self.amplitudes),len(self.z),len(self.r)):
            raise ValueError("psi_values has the wrong shape")
        if len(self.amplitudes)<2 or np.any(np.diff(self.amplitudes)<=0):
            raise ValueError("amplitudes must be strictly increasing")
        self.splines=[RectBivariateSpline(
            self.z,self.r,value,kx=min(3,len(self.z)-1),ky=min(3,len(self.r)-1)
        ) for value in values]

    def evaluate(self, amplitude, zcoord, radius):
        amplitude=float(amplitude)
        if amplitude<self.amplitudes[0] or amplitude>self.amplitudes[-1]:
            raise ValueError("pseudo-arclength step left the metric-family range")
        upper=int(np.searchsorted(self.amplitudes,amplitude,side="right"))
        upper=min(max(upper,1),len(self.amplitudes)-1);lower=upper-1
        width=self.amplitudes[upper]-self.amplitudes[lower]
        weight=(amplitude-self.amplitudes[lower])/width
        zcoord=np.clip(zcoord,self.z[0],self.z[-1]);radius=np.clip(radius,self.r[0],self.r[-1])
        samples=[]
        for dx,dy in ((0,0),(1,0),(0,1)):
            left=self.splines[lower].ev(zcoord,radius,dx=dx,dy=dy)
            right=self.splines[upper].ev(zcoord,radius,dx=dx,dy=dy)
            samples.append((1-weight)*left+weight*right)
        return tuple(samples)


def _profile_on(profile,theta):
    source_theta=np.asarray(profile["theta"])
    return np.interp(theta,source_theta,np.asarray(profile["rho"]))


def pseudo_arclength_step(family,previous,current,step,tolerance=2e-6,nodes=180,max_nodes=8000):
    """Take one secant-predictor pseudo-arclength step.

    ``previous`` and ``current`` contain ``amplitude``, ``theta``, and ``rho``.
    The inner product is mean(profile product) plus the amplitude product, so
    the step has a transparent scale in the dimensionless variables used here.
    """
    theta=np.linspace(1e-4,np.pi/2,nodes);length=theta[-1]-theta[0]
    rho_previous=_profile_on(previous,theta);rho_current=_profile_on(current,theta)
    delta_rho=rho_current-rho_previous
    delta_amplitude=float(current["amplitude"]-previous["amplitude"])
    norm=float(np.sqrt(np.trapezoid(delta_rho**2,x=theta)/length+delta_amplitude**2))
    if not np.isfinite(norm) or norm<=0:
        raise ValueError("continuation points must be distinct")
    tangent_rho=delta_rho/norm;tangent_amplitude=delta_amplitude/norm
    predicted_rho=rho_current+step*tangent_rho
    predicted_amplitude=float(current["amplitude"]+step*tangent_amplitude)
    predicted_slope=np.gradient(predicted_rho,theta,edge_order=2)
    integrand=(predicted_rho-rho_current)*tangent_rho/length
    auxiliary=np.concatenate(([0.],cumulative_trapezoid(integrand,x=theta)))
    state=np.vstack((predicted_rho,predicted_slope,auxiliary))

    def equation(angle,values,parameter):
        rho,slope=values[:2];st=np.sin(angle);ct=np.cos(angle)
        radius=rho*st;zcoord=family.z[-1]-rho*ct
        psi,psi_z,psi_r=family.evaluate(parameter[0],zcoord,radius)
        second=_rho_second(angle,rho,slope,psi,psi_r,psi_z)
        reference=np.interp(angle,theta,rho_current)
        tangent=np.interp(angle,theta,tangent_rho)
        arclength_density=(rho-reference)*tangent/length
        return np.vstack((slope,second,arclength_density))

    def boundaries(left,right,parameter):
        return np.array([
            left[1],right[1],left[2],
            right[2]+tangent_amplitude*(parameter[0]-current["amplitude"])-step,
        ])

    try:
        solved=solve_bvp(
            equation,boundaries,theta,state,p=np.array([predicted_amplitude]),
            tol=tolerance,max_nodes=max_nodes,verbose=0,
        )
    except ValueError as error:
        return {"converged":False,"message":str(error),"amplitude":predicted_amplitude}
    dense=np.linspace(theta[0],theta[-1],400);values=solved.sol(dense)
    rho,slope=values[:2];radius=rho*np.sin(dense);zcoord=family.z[-1]-rho*np.cos(dense)
    in_domain=bool(
        np.min(rho)>2*family.r[1]
        and np.min(zcoord)>family.z[0]+2*(family.z[1]-family.z[0])
        and np.max(radius)<.9*family.r[-1]
    )
    return {
        "converged":bool(solved.success and in_domain),
        "solver_success":bool(solved.success),
        "in_domain":in_domain,
        "message":solved.message,
        "amplitude":float(solved.p[0]),
        "theta":dense,"rho":rho,"slope":slope,
        "rho_axis":float(rho[0]),"rho_brane":float(rho[-1]),
        "rho_min":float(np.min(rho)),"rho_max":float(np.max(rho)),
        "boundary_slope_error":float(max(abs(slope[0]),abs(slope[-1]))),
        "arclength_residual":float(abs(values[2,-1]+tangent_amplitude*(solved.p[0]-current["amplitude"])-step)),
        "mesh_nodes":int(len(solved.x)),
    }


def continue_capped_arclength(family,first,second,step,count,**kwargs):
    """Return a pseudo-arclength sequence starting with two known profiles."""
    points=[first,second]
    for _ in range(count):
        candidate=pseudo_arclength_step(family,points[-2],points[-1],step,**kwargs)
        points.append(candidate)
        if not candidate["converged"]:
            break
    return points
