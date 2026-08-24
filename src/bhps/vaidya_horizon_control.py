"""Analytic and numerical horizon controls in ingoing Vaidya collapse."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def thin_shell_mass(v,final_mass=1.,shell_time=0.):
    """Mass function for an ideal ingoing null shell."""
    v=np.asarray(v,dtype=float)
    return np.where(v<float(shell_time),0.,float(final_mass))


def thin_shell_apparent_horizon(v,final_mass=1.,shell_time=0.):
    """Positive-radius marginal surface; NaN means none before the shell."""
    v=np.asarray(v,dtype=float);radius=np.full(v.shape,np.nan)
    radius[v>=float(shell_time)]=2*float(final_mass)
    return radius


def thin_shell_event_horizon(v,final_mass=1.,shell_time=0.):
    """Exact event-horizon generator for null-shell collapse.

    Before the shell it is an outgoing Minkowski null ray.  It reaches
    ``r=2M`` at the shell and remains there in the final Schwarzschild region.
    Values before its regular center origin are returned as NaN.
    """
    v=np.asarray(v,dtype=float);mass=float(final_mass);shell=float(shell_time)
    birth=shell-4*mass;radius=np.full(v.shape,np.nan)
    before=(v>=birth)&(v<shell);radius[before]=2*mass+.5*(v[before]-shell)
    radius[v>=shell]=2*mass
    return radius


def outgoing_null_rhs(v,r,mass):
    """``dr/dv`` for outgoing radial null curves in ingoing Vaidya."""
    radius=np.asarray(r,dtype=float);return .5*(1-2*np.asarray(mass,dtype=float)/radius)


def outgoing_expansion(radius,mass):
    """Outgoing expansion in the normalization ``l=partial_v+f partial_r/2``."""
    radius=np.asarray(radius,dtype=float);mass=np.asarray(mass,dtype=float)
    with np.errstate(divide="ignore",invalid="ignore"):
        return (1-2*mass/radius)/radius


def smootherstep_mass(v,final_mass=1.,start=0.,duration=1.):
    """C2 compact-transition mass profile obeying the null energy condition."""
    v=np.asarray(v,dtype=float);x=np.clip((v-float(start))/float(duration),0.,1.)
    smooth=x**3*(10-15*x+6*x**2)
    return float(final_mass)*smooth


def trace_smooth_event_horizon(
    final_mass=1.,start=0.,duration=1.,past_margin=None,rtol=1e-11,atol=1e-12,max_step=None,
):
    """Trace the event horizon backward from the final stationary region."""
    final_mass=float(final_mass);start=float(start);duration=float(duration)
    if final_mass<=0 or duration<=0:raise ValueError("mass and duration must be positive")
    final=start+duration
    past_margin=6*final_mass if past_margin is None else float(past_margin)
    maximum=duration/200 if max_step is None else float(max_step)

    def rhs(v,state):
        mass=float(smootherstep_mass(v,final_mass,start,duration))
        return np.array((outgoing_null_rhs(v,state[0],mass),))

    def center(_,state):return state[0]
    center.terminal=True;center.direction=0
    solved=solve_ivp(
        rhs,(final,start-past_margin),np.array((2*final_mass,)),events=center,
        rtol=rtol,atol=atol,max_step=maximum,dense_output=True,
    )
    if solved.t_events[0].size:
        birth=float(solved.t_events[0][0]);lower=birth
    else:
        birth=float("nan");lower=float(solved.t[-1])
    sample=np.linspace(lower,final,1001);radius=np.maximum(solved.sol(sample)[0],0.)
    mass=smootherstep_mass(sample,final_mass,start,duration)
    apparent=2*mass;apparent[mass==0]=np.nan
    return {
        "converged":bool(solved.success),"message":solved.message,"v":sample,"event_radius":radius,
        "apparent_radius":apparent,"mass":mass,"birth_time":birth,"collapse_start":start,
        "collapse_end":final,"final_mass":final_mass,"function_evaluations":int(solved.nfev),
        "minimum_event_minus_apparent":float(np.nanmin(radius-apparent)),
    }
