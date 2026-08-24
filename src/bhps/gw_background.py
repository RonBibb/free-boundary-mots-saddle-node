"""Weak-backreaction Goldberger--Wise background control.

Units are ``ell=kappa5=1``.  The lower brane fixes the conformal normalization
and retains its tuned Robin coefficient.  The upper coefficient is reported as
the tension retuning required by the scalar backreaction.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_bvp, simpson


def solve_gw_background(
    z,
    epsilon=0.1,
    backreaction=0.01,
    v1_over_v0=None,
    tolerance=1e-10,
    wall_stiffness=None,
):
    z=np.asarray(z,dtype=float)
    if np.any(np.diff(z)<=0) or z[0]<=0:
        raise ValueError("z must be a strictly increasing positive grid")
    if epsilon<0 or backreaction<0:
        raise ValueError("epsilon and backreaction must be nonnegative")
    d=float(np.log(z[-1]/z[0]))
    ratio=np.exp(-epsilon*d) if v1_over_v0 is None else float(v1_over_v0)
    mass_squared=epsilon*(epsilon+4)
    v0=float(np.sqrt(backreaction))
    v1=v0*ratio
    phi_guess=v0*(z/z[0])**(-epsilon)
    psi_guess=1/z
    state=np.vstack((
        psi_guess,
        -1/z**2,
        phi_guess,
        -epsilon*phi_guess/z,
    ))

    def ode(_,values):
        psi,psi_z,phi,phi_z=values
        return np.vstack((
            psi_z,
            2*psi**3-psi*phi_z**2/6-psi**3*mass_squared*phi**2/6,
            phi_z,
            -3*psi_z*phi_z/psi+psi**2*mass_squared*phi,
        ))

    def boundary(left,right):
        if wall_stiffness is not None:
            gamma=float(wall_stiffness)
            if gamma<=0:
                raise ValueError("wall_stiffness must be positive")
            return np.array((
                left[0]-1/z[0],
                left[1]+left[0]**2,
                left[3]/left[0]-.5*gamma*(left[2]-v0),
                right[3]/right[0]+.5*gamma*(right[2]-v1),
            ))
        return np.array((
            left[0]-1/z[0],
            left[1]+left[0]**2,
            left[2]-v0,
            right[2]-v1,
        ))

    solution=solve_bvp(ode,boundary,z,state,tol=tolerance,max_nodes=20000)
    sampled=solution.sol(z)
    psi,psi_z,phi,phi_z=sampled
    beta_b=float(-psi_z[-1]/psi[-1]**2)
    gamma=None if wall_stiffness is None else float(wall_stiffness)
    wall_u_a=0. if gamma is None else .5*gamma*(phi[0]-v0)**2
    wall_u_b=0. if gamma is None else .5*gamma*(phi[-1]-v1)**2
    return {
        "converged":bool(solution.success),
        "message":solution.message,
        "z":z,
        "psi":psi,
        "psi_z":psi_z,
        "phi":phi,
        "phi_z":phi_z,
        "epsilon":float(epsilon),
        "mass_squared":float(mass_squared),
        "backreaction":float(backreaction),
        "v0":v0,
        "v1":v1,
        "beta_a":1.0,
        "beta_b":beta_b,
        "wall_stiffness":gamma,
        "wall_target_v0":v0,"wall_target_v1":v1,
        "wall_potential_a":float(wall_u_a),"wall_potential_b":float(wall_u_b),
        "retuned_bare_tension_a":float(6-wall_u_a),
        "retuned_bare_tension_b":float(-6*beta_b-wall_u_b),
        "proper_separation":float(simpson(psi,x=z)),
        "max_ads_relative_deformation":float(np.max(np.abs(psi*z-1))),
        "boundary_residual_max":float(np.max(np.abs(boundary(sampled[:,0],sampled[:,-1])))),
    }
