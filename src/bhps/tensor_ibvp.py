"""Linear transverse-traceless tensor-sector boundary diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh


def frozen_neumann_boundary_symbol(laplace_real,laplace_imag,tangential_wavenumber):
    """Return the decaying-wave Neumann determinant for a frozen half-space.

    For ``u_tt=u_nn+Delta_T u`` and ``u_n=0``, a mode
    ``exp(s t+i k.x-lambda n)`` has ``lambda^2=s^2+|k|^2``.  The square-root
    branch is chosen with positive real part.  A zero determinant with
    ``Re(s)>0`` would be a boundary instability.
    """
    s=complex(float(laplace_real),float(laplace_imag));k=float(tangential_wavenumber)
    if s.real<=0 or k<0:raise ValueError("require Re(s)>0 and k>=0")
    decay=np.sqrt(s*s+k*k)
    if decay.real<0 or (decay.real==0 and decay.imag<0):decay=-decay
    return {"s":s,"decay_rate":decay,"boundary_determinant":-decay,"unstable_root":bool(abs(decay)<1e-12)}


def weighted_neumann_tensor_spectrum(z,psi,count=8,tangential_wavenumber=0.):
    """Linear-FE spectrum of ``-(p f')'=p(omega^2-k^2)f``, ``p=psi^3``.

    Natural Neumann rows implement the TT part of the linearized Israel
    condition.  The symmetric generalized eigenproblem makes positivity
    auditable without a coordinate-dependent strong-form stencil.
    """
    z=np.asarray(z,dtype=float);psi=np.asarray(psi,dtype=float)
    if z.ndim!=1 or psi.shape!=z.shape or np.any(np.diff(z)<=0) or np.any(psi<=0):
        raise ValueError("z and positive psi must be matching ordered one-dimensional arrays")
    size=len(z);stiffness=np.zeros((size,size));mass=np.zeros((size,size))
    for i in range(size-1):
        spacing=z[i+1]-z[i];weight=.5*(psi[i]**3+psi[i+1]**3)
        stiffness[i:i+2,i:i+2]+=weight/spacing*np.array(((1.,-1.),(-1.,1.)))
        mass[i:i+2,i:i+2]+=weight*spacing/6*np.array(((2.,1.),(1.,2.)))
    stiffness+=float(tangential_wavenumber)**2*mass
    values,vectors=eigh(stiffness,mass,subset_by_index=(0,min(int(count),size)-1))
    constant=np.ones(size);constant/=np.sqrt(constant@mass@constant)
    zero_overlap=float(abs(vectors[:,0]@mass@constant))
    return {
        "omega_squared":values,"minimum_omega_squared":float(values[0]),
        "first_positive_omega_squared":float(values[1]) if len(values)>1 else None,
        "zero_mode_constant_overlap":zero_overlap,
        "all_within_roundoff_nonnegative":bool(values[0]>-1e-10*max(1.,float(values[-1]))),
        "weight":"psi^3","boundary_condition":"natural Neumann from TT Israel sector",
    }


def tensor_energy_density(psi,h_t,h_z,tangential_gradient_squared=0.):
    """Positive quadratic density and radial wall flux for a TT polarization."""
    psi=np.asarray(psi,dtype=float);h_t=np.asarray(h_t,dtype=float);h_z=np.asarray(h_z,dtype=float)
    density=.5*psi**3*(h_t**2+h_z**2+np.asarray(tangential_gradient_squared,dtype=float))
    flux=psi**3*h_t*h_z
    return {"density":density,"normal_flux":flux}


def frozen_positive_robin_boundary_symbol(laplace_real,laplace_imag,tangential_wavenumber,mass,wall_stiffness):
    """Frozen scalar boundary determinant for positive quadratic wall energy."""
    s=complex(float(laplace_real),float(laplace_imag));k=float(tangential_wavenumber)
    mass=float(mass);gamma=float(wall_stiffness)
    if s.real<=0 or k<0 or mass<0 or gamma<0:raise ValueError("invalid half-space parameters")
    decay=np.sqrt(s*s+k*k+mass*mass)
    if decay.real<0 or (decay.real==0 and decay.imag<0):decay=-decay
    # In inward half-space coordinate x, n_out=-d_x and the decaying mode is
    # exp(-lambda x).  B=n_out.d phi+gamma phi/2=(lambda+gamma/2)phi.
    determinant=decay+gamma/2
    return {"decay_rate":decay,"boundary_determinant":determinant,"unstable_root":bool(abs(determinant)<1e-12)}


def weighted_robin_scalar_spectrum(z,psi,mass_squared,wall_stiffness,count=8,tangential_wavenumber=0.):
    """Positive fixed-metric scalar spectrum with Z2 half-Robin walls."""
    z=np.asarray(z,dtype=float);psi=np.asarray(psi,dtype=float);mass_squared=float(mass_squared);gamma=float(wall_stiffness)
    if z.ndim!=1 or psi.shape!=z.shape or np.any(np.diff(z)<=0) or np.any(psi<=0) or mass_squared<0 or gamma<0:
        raise ValueError("invalid scalar spectrum inputs")
    size=len(z);stiffness=np.zeros((size,size));mass_matrix=np.zeros((size,size))
    for i in range(size-1):
        spacing=z[i+1]-z[i];p=.5*(psi[i]**3+psi[i+1]**3);v=.5*(psi[i]**5+psi[i+1]**5)*mass_squared
        stiffness[i:i+2,i:i+2]+=p/spacing*np.array(((1.,-1.),(-1.,1.)))
        stiffness[i:i+2,i:i+2]+=v*spacing/6*np.array(((2.,1.),(1.,2.)))
        mass_matrix[i:i+2,i:i+2]+=p*spacing/6*np.array(((2.,1.),(1.,2.)))
    # Coordinate Robin data are phi_z=+psi gamma phi/2 at the lower wall and
    # phi_z=-psi gamma phi/2 at the upper wall, yielding positive weak terms.
    stiffness[0,0]+=psi[0]**4*gamma/2;stiffness[-1,-1]+=psi[-1]**4*gamma/2
    stiffness+=float(tangential_wavenumber)**2*mass_matrix
    values=eigh(stiffness,mass_matrix,eigvals_only=True,subset_by_index=(0,min(int(count),size)-1))
    return {
        "omega_squared":values,"minimum_omega_squared":float(values[0]),
        "all_positive":bool(values[0]>0),"bulk_weight":"psi^3",
        "wall_energy_coefficients":[float(psi[0]**4*gamma/2),float(psi[-1]**4*gamma/2)],
        "boundary_condition":"one-sided Z2 positive half-Robin",
    }
