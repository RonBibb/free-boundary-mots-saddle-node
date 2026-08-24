"""Gauge-invariant coupled scalar--radion spectrum on a warped interval.

The master equation follows the scalar sector of Boos et al.,
Mod. Phys. Lett. A 21 (2006) 1431, arXiv:hep-th/0511185v4.  Their
``g = exp(-2 A) h_44`` already contains the scalar metric perturbation and,
through the linearized constraint, the bulk-scalar perturbation.  It is not a
fixed-metric Klein--Gordon mode.

We use conformal coordinates

    ds^2 = psi(z)^2 (eta_ab dx^a dx^b + dz^2),   dy = psi dz,

and the action normalization ``R/(2 kappa5_squared)``.  The weak form is

    K[g,v] = mu^2 M[g,v],

with

    K = integral dz [s g_z v_z + 2 kappa5_squared/(3 psi) g v],
    M = integral dz s g v + b_0 g(0)v(0) + b_1 g(1)v(1),
    s = 1/(psi phi_z^2).

For finite quadratic wall potentials the boundary weights ``b_i`` encode the
eigenvalue-dependent scalar Israel conditions.  Positive weights, together
with the positive bulk forms, make ``mu^2 > 0`` manifest at the discrete
variational level.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar
from scipy.integrate import simpson


def _wall_pair(wall_stiffness):
    if wall_stiffness is None:
        return None
    values=np.asarray(wall_stiffness,dtype=float)
    if values.ndim==0:
        values=np.repeat(values,2)
    if values.shape!=(2,) or np.any(values<=0):
        raise ValueError("wall_stiffness must be positive or a positive pair")
    return values


def frozen_wentzell_boundary_symbol(
    laplace_real,
    laplace_imag,
    tangential_wavenumber,
    bulk_gradient_weight,
    boundary_kinetic_weight,
    bulk_mass_squared=0.0,
):
    """Frozen scalar--radion determinant for a dynamical wall condition.

    In an inward half-space coordinate the decaying mode is
    ``exp(-rho x)``.  The outward condition is
    ``p d_n g = b Box_4 g``, hence
    ``D=p rho+b(s^2+k^2)`` with
    ``rho^2=s^2+k^2+m^2``.  Positive ``p`` and ``b`` supply positive bulk and
    wall kinetic energies.
    """
    sigma=complex(float(laplace_real),float(laplace_imag))
    wave=float(tangential_wavenumber);p=float(bulk_gradient_weight)
    b=float(boundary_kinetic_weight);mass_squared=float(bulk_mass_squared)
    if sigma.real<=0 or wave<0 or p<=0 or b<0 or mass_squared<0:
        raise ValueError("require Re(s)>0, k>=0, p>0, b>=0, and mass^2>=0")
    frequency=sigma*sigma+wave*wave
    decay=np.sqrt(frequency+mass_squared)
    if decay.real<0 or (decay.real==0 and decay.imag<0):decay=-decay
    determinant=p*decay+b*frequency
    scale=max(1.,abs(p*decay),abs(b*frequency))
    return {
        "s":sigma,"decay_rate":decay,"boundary_determinant":determinant,
        "normalized_determinant_magnitude":float(abs(determinant)/scale),
        "unstable_root":bool(abs(determinant)<=1e-12*scale),
    }


def coupled_scalar_radion_energy(
    z,
    psi,
    phi_z,
    g,
    g_t,
    g_z,
    tangential_gradient_squared=0.0,
    boundary_mass_weights=(0.0,0.0),
    kappa5_squared=1.0,
):
    """Positive quadratic bulk-plus-wall energy of the master field."""
    z=np.asarray(z,dtype=float);psi=np.asarray(psi,dtype=float)
    phi_z=np.asarray(phi_z,dtype=float);g=np.asarray(g,dtype=float)
    g_t=np.asarray(g_t,dtype=float);g_z=np.asarray(g_z,dtype=float)
    tangent=np.asarray(tangential_gradient_squared,dtype=float)
    if tangent.ndim==0:tangent=np.full(z.shape,float(tangent))
    if any(array.shape!=z.shape for array in (psi,phi_z,g,g_t,g_z,tangent)):
        raise ValueError("energy arrays must match z")
    boundary=np.asarray(boundary_mass_weights,dtype=float)
    if boundary.shape!=(2,) or np.any(boundary<0) or np.any(psi<=0) or np.any(phi_z==0):
        raise ValueError("invalid positive energy weights")
    s=1/(psi*phi_z**2);q=2*float(kappa5_squared)/(3*psi)
    density=.5*(s*(g_t**2+g_z**2+tangent)+q*g**2)
    bulk=float(simpson(density,x=z))
    wall=float(.5*(boundary[0]*(g_t[0]**2+tangent[0])+boundary[1]*(g_t[-1]**2+tangent[-1])))
    return {"bulk_energy":bulk,"boundary_energy":wall,"total_energy":bulk+wall,"density":density}


def coupled_scalar_radion_spectrum(
    z,
    psi,
    psi_z,
    phi,
    phi_z,
    mass_squared,
    wall_stiffness=None,
    count=8,
    kappa5_squared=1.0,
):
    """Return the coupled scalar--radion eigenvalues and diagnostic matrices.

    ``wall_stiffness=None`` is the stiff-potential limit, in which the master
    variable has natural Neumann data.  A finite scalar or pair supplies
    ``U_i''`` in the doubled-cover convention used by ``MODEL_V1.md``.
    """
    z=np.asarray(z,dtype=float);psi=np.asarray(psi,dtype=float)
    psi_z=np.asarray(psi_z,dtype=float);phi=np.asarray(phi,dtype=float)
    phi_z=np.asarray(phi_z,dtype=float);mass_squared=float(mass_squared)
    kappa5_squared=float(kappa5_squared);wall_pair=_wall_pair(wall_stiffness)
    size=len(z)
    if (
        z.ndim!=1 or size<3 or psi.shape!=(size,) or psi_z.shape!=(size,)
        or phi.shape!=(size,) or phi_z.shape!=(size,) or np.any(np.diff(z)<=0)
        or np.any(psi<=0) or mass_squared<0 or kappa5_squared<=0
    ):
        raise ValueError("invalid warped-background inputs")
    derivative_floor=1e-13*max(1.,float(np.max(np.abs(phi_z))))
    if np.any(np.abs(phi_z)<=derivative_floor):
        raise ValueError("phi_z must remain nonzero for this master variable")
    if np.any(phi_z[:-1]*phi_z[1:]<0):
        raise ValueError("phi_z changes sign; the divided master variable is singular")

    # Use the exact background scalar equation rather than differentiating a
    # sampled profile: phi_zz=-3(psi_z/psi)phi_z+psi^2 m^2 phi.
    phi_zz=-3*psi_z*phi_z/psi+psi**2*mass_squared*phi
    proper_log_derivative=(phi_zz/phi_z-psi_z/psi)/psi

    stiffness=np.zeros((size,size));mass_matrix=np.zeros((size,size))
    sl_weight=1/(psi*phi_z**2)
    potential_weight=2*kappa5_squared/(3*psi)
    element=np.array(((2.,1.),(1.,2.)))
    derivative=np.array(((1.,-1.),(-1.,1.)))
    for i in range(size-1):
        spacing=z[i+1]-z[i]
        s=.5*(sl_weight[i]+sl_weight[i+1])
        q=.5*(potential_weight[i]+potential_weight[i+1])
        stiffness[i:i+2,i:i+2]+=s/spacing*derivative+q*spacing/6*element
        mass_matrix[i:i+2,i:i+2]+=s*spacing/6*element

    if wall_pair is None:
        alphas=np.array((np.inf,np.inf));boundary_weights=np.zeros(2)
    else:
        # Lower/upper signs follow the one-sided orbifold conditions:
        # alpha_0=U_0''/2-phi_yy/phi_y and
        # alpha_1=U_1''/2+phi_yy/phi_y.
        alphas=np.array((
            wall_pair[0]/2-proper_log_derivative[0],
            wall_pair[1]/2+proper_log_derivative[-1],
        ))
        boundary_weights=np.array((
            1/(alphas[0]*psi[0]**2*phi_z[0]**2),
            1/(alphas[1]*psi[-1]**2*phi_z[-1]**2),
        ))
        mass_matrix[0,0]+=boundary_weights[0]
        mass_matrix[-1,-1]+=boundary_weights[1]

    if np.any(~np.isfinite(mass_matrix)) or np.any(np.linalg.eigvalsh(mass_matrix)<=0):
        raise ValueError("coupled scalar-radion kinetic form is not positive")
    number=min(max(1,int(count)),size)
    values,vectors=eigh(stiffness,mass_matrix,subset_by_index=(0,number-1))
    return {
        "mu_squared":values,
        "eigenvectors":vectors,
        "minimum_mu_squared":float(values[0]),
        "all_positive":bool(values[0]>0),
        "wall_alphas":alphas,
        "boundary_mass_weights":boundary_weights,
        "positive_wall_weights":bool(np.all(boundary_weights>=0)),
        "proper_phi_log_derivative_at_walls":np.array((proper_log_derivative[0],proper_log_derivative[-1])),
        "bulk_stiffness_minimum_eigenvalue":float(np.linalg.eigvalsh(stiffness)[0]),
        "master_variable":"g=exp(-2A) h_yy; delta-Phi is fixed by the linearized constraint",
        "boundary_condition":"eigenvalue-dependent scalar Israel condition represented as boundary kinetic weight",
    }


def shoot_lowest_scalar_radion_mode(
    z,
    psi,
    psi_z,
    phi,
    phi_z,
    mass_squared,
    wall_stiffness=None,
    eigenvalue_hint=None,
    kappa5_squared=1.0,
    tolerance=1e-10,
):
    """Independently reproduce the lowest master eigenvalue by shooting."""
    z=np.asarray(z,dtype=float);psi=np.asarray(psi,dtype=float)
    psi_z=np.asarray(psi_z,dtype=float);phi=np.asarray(phi,dtype=float)
    phi_z=np.asarray(phi_z,dtype=float);mass_squared=float(mass_squared)
    kappa5_squared=float(kappa5_squared);wall_pair=_wall_pair(wall_stiffness)
    if any(array.shape!=z.shape for array in (psi,psi_z,phi,phi_z)):
        raise ValueError("background arrays must match z")
    derivative_floor=1e-13*max(1.,float(np.max(np.abs(phi_z))))
    if np.any(np.abs(phi_z)<=derivative_floor) or np.any(phi_z[:-1]*phi_z[1:]<0):
        raise ValueError("phi_z must remain nonzero with fixed sign for shooting")
    phi_zz=-3*psi_z*phi_z/psi+psi**2*mass_squared*phi
    proper_log_derivative=(phi_zz/phi_z-psi_z/psi)/psi
    if wall_pair is None:
        alphas=np.array((np.inf,np.inf))
    else:
        alphas=np.array((
            wall_pair[0]/2-proper_log_derivative[0],
            wall_pair[1]/2+proper_log_derivative[-1],
        ))
        if np.any(alphas<=0):
            raise ValueError("wall alpha must be positive for the audited branch")

    log_s_derivative=CubicSpline(z,-psi_z/psi-2*phi_zz/phi_z)
    phi_z_spline=CubicSpline(z,phi_z)
    psi_spline=CubicSpline(z,psi)

    def determinant(eigenvalue,return_solution=False):
        eigenvalue=float(eigenvalue)
        lower_slope=0. if wall_pair is None else -eigenvalue/(alphas[0]*psi[0])

        def equation(location,state):
            g,g_z=state
            return (
                g_z,
                -float(log_s_derivative(location))*g_z
                +(2*kappa5_squared*float(phi_z_spline(location))**2/3-eigenvalue)*g,
            )

        integrated=solve_ivp(
            equation,(z[0],z[-1]),(1.,lower_slope),method="DOP853",
            rtol=tolerance,atol=tolerance*1e-3,dense_output=return_solution,
        )
        if not integrated.success:
            raise RuntimeError(integrated.message)
        g,g_z=integrated.y[:,-1]
        upper_term=0. if wall_pair is None else eigenvalue/(alphas[1]*float(psi_spline(z[-1])))
        value=float(g_z-upper_term*g)
        return (value,integrated) if return_solution else value

    if eigenvalue_hint is None:
        finite_element=coupled_scalar_radion_spectrum(
            z,psi,psi_z,phi,phi_z,mass_squared,wall_stiffness,count=1,
            kappa5_squared=kappa5_squared,
        )
        eigenvalue_hint=finite_element["minimum_mu_squared"]
    hint=float(eigenvalue_hint)
    if hint<=0:raise ValueError("eigenvalue_hint must be positive")
    lower=0.;upper=max(2*hint,1e-14)
    lower_value=determinant(lower);upper_value=determinant(upper)
    for _ in range(40):
        if lower_value*upper_value<=0:break
        upper*=2;upper_value=determinant(upper)
    else:
        raise RuntimeError("could not bracket the lowest shooting root")
    root=root_scalar(
        determinant,bracket=(lower,upper),xtol=max(tolerance*hint,1e-15),rtol=tolerance,
    )
    residual,integrated=determinant(root.root,return_solution=True)
    return {
        "mu_squared":float(root.root),"boundary_residual":float(residual),
        "iterations":int(root.iterations),"converged":bool(root.converged),
        "wall_alphas":alphas,"integration_steps":int(len(integrated.t)),
        "method":"adaptive DOP853 shooting plus bracketing root solve",
    }
