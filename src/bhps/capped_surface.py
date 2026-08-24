"""Donor-capped apparent-horizon finder on a time-symmetric slice.

The meridional curve is written in polar form about brane B,

    r=rho(theta) sin(theta),
    z=z_B-rho(theta) cos(theta),   0 <= theta <= pi/2.

It closes smoothly on the symmetry axis and meets brane B orthogonally.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.linalg import eigh
from scipy.integrate import solve_bvp
from scipy.integrate import simpson
from scipy.interpolate import RectBivariateSpline


def _rho_second(theta, rho, slope, psi, psi_r, psi_z):
    """Solve H_flat+3 n.grad(log psi)=0 for rho''.

    The outward unit normal in the (r,z) plane is (z',-r')/speed.
    Algebra is evaluated directly to keep the geometric convention visible.
    """
    st, ct = np.sin(theta), np.cos(theta)
    radius = rho * st
    rp = slope * st + rho * ct
    zp = -slope * ct + rho * st
    speed2 = rho**2 + slope**2
    speed = np.sqrt(speed2)

    # r''=rho'' sin+2rho' cos-rho sin
    # z''=-rho'' cos+2rho' sin+rho cos
    # The numerator r'z''-z'r'' is affine in rho''.
    rpp_without = 2 * slope * ct - rho * st
    zpp_without = 2 * slope * st + rho * ct
    numerator_without = rp * zpp_without - zp * rpp_without
    numerator_coefficient = -rp * ct - zp * st  # equals -rho

    safe_radius = np.maximum(radius, 1e-10)
    rotational = 2 * zp / (speed * safe_radius)
    conformal = 3 * (zp * psi_r - rp * psi_z) / (speed * psi)
    target_numerator = -(rotational + conformal) * speed**3
    return (target_numerator - numerator_without) / numerator_coefficient


def capped_surface_equation(spline, z_b, theta, state):
    rho, slope = state
    st, ct = np.sin(theta), np.cos(theta)
    radius = rho * st
    zcoord = z_b - rho * ct
    z_knots, r_knots = spline.get_knots()
    sample_z = np.clip(zcoord, z_knots[0], z_knots[-1])
    sample_r = np.clip(radius, r_knots[0], r_knots[-1])
    psi = spline.ev(sample_z, sample_r)
    psi_z = spline.ev(sample_z, sample_r, dx=1, dy=0)
    psi_r = spline.ev(sample_z, sample_r, dx=0, dy=1)
    second = _rho_second(theta, rho, slope, psi, psi_r, psi_z)
    return np.vstack((slope, second))


def _smooth_free_boundaries(left, right):
    return np.array([left[1], right[1]])


def capped_area_stability(spline,z_b,solution,nodes=41,relative_step=2.5e-5,maximum_angular_mode=3):
    """Return the inertia of a discrete second variation of capped area.

    The polar-graph node values are varied freely, including at the brane;
    orthogonal intersection is the natural boundary condition of the area
    functional.  Eigenvalue magnitudes depend on this discretization, while
    the negative-mode count is the primary diagnostic.
    """
    theta=np.linspace(0,np.pi/2,nodes)
    rho=np.asarray(solution.sol(theta)[0])
    mid=.5*(theta[:-1]+theta[1:]);delta=theta[1]-theta[0]

    def area(values):
        rho_mid=.5*(values[:-1]+values[1:])
        slope=(values[1:]-values[:-1])/delta
        radius=rho_mid*np.sin(mid);zcoord=z_b-rho_mid*np.cos(mid)
        psi=spline.ev(zcoord,radius)
        return float(np.sum(4*np.pi*delta*psi**3*radius**2*np.sqrt(rho_mid**2+slope**2)))

    step=relative_step*max(1.,float(np.mean(rho)));base=area(rho)
    hessian=np.zeros((nodes,nodes))
    for i in range(nodes):
        shift=np.zeros(nodes);shift[i]=step
        hessian[i,i]=(area(rho+shift)-2*base+area(rho-shift))/step**2
        for j in range(i):
            other=np.zeros(nodes);other[j]=step
            value=(area(rho+shift+other)-area(rho+shift-other)-area(rho-shift+other)+area(rho-shift-other))/(4*step**2)
            hessian[i,j]=hessian[j,i]=value
    eigenvalues=np.linalg.eigvalsh(hessian)
    rho_mid=.5*(rho[:-1]+rho[1:]);slope=(rho[1:]-rho[:-1])/delta
    radius=rho_mid*np.sin(mid);zcoord=z_b-rho_mid*np.cos(mid)
    psi=spline.ev(zcoord,radius);speed=np.sqrt(rho_mid**2+slope**2)
    # If eta=delta rho, its physical normal displacement in gamma=psi^2 delta
    # is f=psi*(rho/speed)*eta.  Lump the corresponding integral of f^2 dA.
    segment_mass=4*np.pi*delta*psi**5*radius**2*rho_mid**2/speed
    mass=np.zeros(nodes);mass[:-1]+=.5*segment_mass;mass[1:]+=.5*segment_mass
    # The polar axis has vanishing continuum measure.  The first segment still
    # supplies a positive lumped mass, so the generalized problem is regular.
    mass_matrix=np.diag(mass)
    normalized=eigh(hessian,mass_matrix,eigvals_only=True)
    angular_spectra=[]
    for angular_mode in range(maximum_angular_mode+1):
        if angular_mode==0:
            values=normalized
        else:
            # For f(theta)Y_lm, the angular gradient adds
            # l(l+1)/(psi*r)^2 to the Jacobi quadratic form.  A regular
            # non-spherical perturbation vanishes where the S2 orbit closes,
            # so remove the polar-axis degree of freedom for l>0.
            angular_segment=segment_mass*angular_mode*(angular_mode+1)/(psi**2*radius**2)
            angular_lumped=np.zeros(nodes)
            angular_lumped[:-1]+=.5*angular_segment;angular_lumped[1:]+=.5*angular_segment
            stiffness=hessian+np.diag(angular_lumped)
            values=eigh(stiffness[1:,1:],mass_matrix[1:,1:],eigvals_only=True)
        angular_spectra.append({
            "angular_mode":angular_mode,
            "lowest_normalized_eigenvalue":float(values[0]),
            "negative_mode_count":int(np.sum(values < -1e-8*max(float(np.max(np.abs(values))),1.))),
        })
    scale=max(float(np.max(np.abs(eigenvalues))),1.)
    negative=int(np.sum(eigenvalues < -1e-8*scale))
    near_zero=int(np.sum(np.abs(eigenvalues)<=1e-8*scale))
    return {
        "area_hessian_nodes":int(nodes),
        "area_hessian_step":float(step),
        "lowest_area_hessian_eigenvalue":float(eigenvalues[0]),
        "second_area_hessian_eigenvalue":float(eigenvalues[1]),
        "negative_mode_count":negative,
        "near_zero_mode_count":near_zero,
        "lowest_normalized_jacobi_eigenvalue":float(normalized[0]),
        "second_normalized_jacobi_eigenvalue":float(normalized[1]),
        "normalized_negative_mode_count":int(np.sum(normalized < -1e-8*max(float(np.max(np.abs(normalized))),1.))),
        "angular_mode_spectrum":angular_spectra,
    }


def find_donor_capped_surfaces(z, r, psi, guesses=None, tolerance=2e-5,stability_nodes=41,stability_step=2.5e-5):
    """Search for half-S3 marginal surfaces attached only to brane B."""
    spline = RectBivariateSpline(z, r, psi, kx=min(3, len(z)-1), ky=min(3, len(r)-1))
    z_b = float(z[-1])
    maximum_rho = min(z_b-z[0], 0.85*r[-1])
    if guesses is None:
        guesses = tuple(np.linspace(0.08*maximum_rho, 0.9*maximum_rho, 12))
    theta = np.linspace(1e-4, math.pi/2, 100)
    trials, accepted = [], []
    for guess in guesses:
        initial = np.vstack((np.full_like(theta, guess), np.zeros_like(theta)))
        solved = solve_bvp(
            lambda angle, state: capped_surface_equation(spline, z_b, angle, state),
            _smooth_free_boundaries,
            theta,
            initial,
            tol=tolerance,
            max_nodes=3000,
        )
        rho = solved.y[0]
        slope = solved.y[1]
        rr = rho*np.sin(solved.x)
        zz = z_b-rho*np.cos(solved.x)
        in_domain = bool(np.min(rho) > 2*r[1] and np.min(zz) > z[0]+2*(z[1]-z[0]) and np.max(rr) < .9*r[-1])
        dense=np.linspace(theta[0],theta[-1],300)
        dense_state=solved.sol(dense);dense_derivative=solved.sol(dense,1)
        dense_rhs=capped_surface_equation(spline,z_b,dense,dense_state)
        dense_rho,dense_slope=dense_state
        dense_r=dense_rho*np.sin(dense);dense_z=z_b-dense_rho*np.cos(dense)
        dense_psi=spline.ev(dense_z,dense_r)
        area_integrand=4*np.pi*dense_psi**3*dense_r**2*np.sqrt(dense_rho**2+dense_slope**2)
        item = {
            "guess": float(guess), "solver_success": bool(solved.success),
            "in_domain": in_domain, "rho_axis": float(rho[0]),
            "rho_brane": float(rho[-1]), "rho_min": float(np.min(rho)),
            "rho_max": float(np.max(rho)), "z_tip": float(zz[0]),
            "brane_radius": float(rr[-1]),
            "boundary_slope_error": float(max(abs(slope[0]), abs(slope[-1]))),
            "surface_residual_max":float(np.max(np.abs(dense_derivative-dense_rhs))),
            "area":float(simpson(area_integrand,x=dense)),
        }
        trials.append(item)
        if solved.success and in_domain and item["boundary_slope_error"] < 10*tolerance:
            item.update(capped_area_stability(spline,z_b,solved,stability_nodes,stability_step))
            signature = np.array([item["rho_axis"], item["rho_brane"]])
            if not any(np.linalg.norm(signature-np.array([x["rho_axis"],x["rho_brane"]])) < 5e-3 for x in accepted):
                accepted.append(item)
    return {
        "capped_surface_found": bool(accepted),
        "accepted": accepted,
        "trial_count": len(trials),
        "successful_trials": sum(x["solver_success"] for x in trials),
        "in_domain_successful_trials": sum(x["solver_success"] and x["in_domain"] for x in trials),
        "trials": trials,
    }
