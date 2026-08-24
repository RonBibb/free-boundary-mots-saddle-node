"""Regularized lapse-only repair of spatial Israel second corners."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from bhps.adm_corner import spatial_israel_second_corner_audit,time_symmetric_adm_metric_acceleration,time_symmetric_scalar_acceleration


def lapse_log_basis(z,r,compact_modes=5,radial_modes=10):
    """Cosine basis with zero compact normal derivative at both walls."""
    z=np.asarray(z,dtype=float);r=np.asarray(r,dtype=float)
    compact=(z-z[0])/(z[-1]-z[0]);radial=r/r[-1]
    basis=[];labels=[]
    # Omit the constant-constant rescaling, which cannot repair shape data.
    for n in range(int(compact_modes)):
        compact_factor=np.cos(n*np.pi*compact)[:,None]
        for m in range(int(radial_modes)):
            if n==0 and m==0:continue
            # Half-integer radial cosines are even at the axis and vanish at
            # the finite radial boundary.
            radial_factor=np.cos((m+.5)*np.pi*radial)[None,:]
            basis.append(compact_factor*radial_factor);labels.append([n,m])
    return {"basis":np.asarray(basis),"labels":labels}


def repair_spatial_corners_with_lapse(
    z,r,psi,phi,chi_r,chi_z,background,chi=None,
    compact_modes=5,radial_modes=10,stencil_width=7,radial_buffer=7,
    regularization=1e-3,coefficient_bound=.35,maximum_function_evaluations=10,
):
    """Fit ``alpha=psi*exp(sum c_A basis_A)`` to spatial corner rows.

    The compact cosine basis makes ``partial_z log(alpha/psi)=0`` at both
    walls, so the zeroth-order time-time Israel/lapse Robin condition is
    retained to stencil accuracy.
    """
    z=np.asarray(z,dtype=float);r=np.asarray(r,dtype=float)
    psi=np.asarray(psi,dtype=float);phi=np.asarray(phi,dtype=float)
    chi_r=np.asarray(chi_r,dtype=float);chi_z=np.asarray(chi_z,dtype=float)
    expansion=lapse_log_basis(z,r,compact_modes,radial_modes)
    basis=expansion["basis"];count=len(basis);buffer=int(radial_buffer)
    retained=len(r)-buffer
    # The elliptic selector defines the alpha=psi continuum acceleration as
    # zero.  Subtract its high-order diagnostic defect to remain well balanced.
    selector_defect=time_symmetric_scalar_acceleration(
        z,r,psi,phi,background["mass_squared"],lapse=psi,
        stencil_width=stencil_width,
    )

    def calculate(coefficients):
        log_correction=np.tensordot(coefficients,basis,axes=(0,0))
        lapse=psi*np.exp(log_correction)
        acceleration=time_symmetric_adm_metric_acceleration(
            z,r,psi,phi,chi_r,chi_z,background["mass_squared"],chi=chi,
            stencil_width=stencil_width,lapse=lapse,
        )
        scalar_acceleration=(
            time_symmetric_scalar_acceleration(
                z,r,psi,phi,background["mass_squared"],lapse=lapse,
                stencil_width=stencil_width,
            )-selector_defect
        )
        audit=spatial_israel_second_corner_audit(
            acceleration,psi,phi,background,
            stabilizer_acceleration=scalar_acceleration,radial_buffer=buffer,
        )
        rows=[];scales=[]
        for wall in audit["walls"]:
            index=0 if wall["wall"]=="lower" else -1
            gamma=float(background["wall_stiffness"]);target=background["v0"] if index==0 else background["v1"]
            potential=.5*gamma*(phi[index]-target)**2
            beta=(background["beta_a"]+(potential-background["wall_potential_a"])/6) if index==0 else (background["beta_b"]-(potential-background["wall_potential_b"])/6)
            for name in ("radial","transverse"):
                value=acceleration[name][index];derivative=(acceleration["Dz"]@acceleration[name])[index]
                beta_phi=(gamma*(phi[index]-target)/6) if index==0 else (-gamma*(phi[index]-target)/6)
                terms=(
                    derivative,2*beta*psi[index]*value,
                    beta*psi[index]*acceleration["zz"][index],
                    2*beta_phi*scalar_acceleration[index]*psi[index]**3,
                )
                residual=sum(terms)[:retained];scale=np.maximum(1.,sum(np.abs(term) for term in terms)[:retained])
                rows.append(residual);scales.append(scale)
            mixed=acceleration["zr"][index,:retained]
            mixed_scale=np.maximum(1.,
                np.abs(acceleration["zz"][index,:retained])
                +np.abs(acceleration["radial"][index,:retained])
                +np.abs(acceleration["transverse"][index,:retained])
            )
            rows.append(mixed);scales.append(mixed_scale)
        return lapse,log_correction,acceleration,audit,np.concatenate(rows),np.concatenate(scales)

    zero=np.zeros(count)
    initial_calculation=calculate(zero);baseline_scales=initial_calculation[-1]

    def evaluate(coefficients,full=False):
        lapse,log_correction,acceleration,audit,physical,_=calculate(coefficients)
        residual=physical/baseline_scales
        if regularization>0:
            weights=np.array([1+n*n+m*m for n,m in expansion["labels"]],dtype=float)
            residual=np.concatenate((residual,np.sqrt(regularization)*weights*coefficients))
        if full:return lapse,log_correction,acceleration,audit,residual
        return residual

    initial=evaluate(zero)
    solved=least_squares(
        evaluate,zero,bounds=(-float(coefficient_bound),float(coefficient_bound)),
        max_nfev=int(maximum_function_evaluations),xtol=1e-9,ftol=1e-9,gtol=1e-9,
        verbose=0,
    )
    lapse,log_correction,acceleration,audit,final=evaluate(solved.x,full=True)
    physical_row_count=6*retained
    return {
        "success":bool(solved.success),"message":solved.message,
        "function_evaluations":int(solved.nfev),"coefficient_count":count,
        "coefficients":solved.x,"basis_labels":expansion["labels"],
        "lapse":lapse,"log_lapse_correction":log_correction,
        "minimum_lapse":float(np.min(lapse)),"maximum_lapse":float(np.max(lapse)),
        "maximum_absolute_log_lapse_correction":float(np.max(np.abs(log_correction))),
        "initial_physical_residual_l2":float(np.linalg.norm(initial[:physical_row_count])),
        "final_physical_residual_l2":float(np.linalg.norm(final[:physical_row_count])),
        "initial_maximum_fixed_scaled_residual":float(np.max(np.abs(initial[:physical_row_count]))),
        "final_maximum_fixed_scaled_residual":float(np.max(np.abs(final[:physical_row_count]))),
        "corner_audit":audit,
        "regularization":float(regularization),
        "zeroth_order_lapse_robin_preserved_by_basis":True,
        "maximum_discrete_wall_derivative_of_log_correction":float(max(
            np.max(np.abs((acceleration["Dz"]@log_correction)[0])),
            np.max(np.abs((acceleration["Dz"]@log_correction)[-1])),
        )),
    }
