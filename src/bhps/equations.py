"""Symbolic identities for Model V1 constraints and energy."""

from __future__ import annotations

import sympy as sp


def conformal_hamiltonian_identities():
    r,z,ell,kappa=sp.symbols("r z ell kappa5",positive=True)
    m_phi,m_chi=sp.symbols("m_Phi m_chi",nonnegative=True)
    psi=sp.Function("psi")(r,z); phi=sp.Function("Phi")(r,z); chi=sp.Function("chi")(r,z)
    lap=sp.diff(psi,z,2)+sp.diff(psi,r,2)+2*sp.diff(psi,r)/r
    grad_phi=sp.diff(phi,z)**2+sp.diff(phi,r)**2
    grad_chi=sp.diff(chi,z)**2+sp.diff(chi,r)**2
    residual=sp.simplify(lap-2*psi**3/ell**2+kappa**2*psi*(grad_phi+grad_chi)/6+kappa**2*psi**3*(m_phi**2*phi**2+m_chi**2*chi**2)/6)
    background=ell/z
    ads_residual=sp.simplify(residual.subs({psi:background,phi:0,chi:0}).doit())
    rho_chi=sp.Rational(1,2)*(grad_chi/psi**2+m_chi**2*chi**2)
    energy_integrand=sp.simplify(4*sp.pi*r**2*psi**4*rho_chi)
    return {"symbols":{"r":r,"z":z,"ell":ell,"kappa5":kappa},"residual":residual,"ads_residual":ads_residual,"rho_chi":rho_chi,"energy_integrand":energy_integrand}


def momentum_time_symmetry_identity():
    pi_phi,pi_chi=sp.symbols("Pi_Phi Pi_chi")
    dphi_i,dchi_i=sp.symbols("D_i_Phi D_i_chi")
    current=-(pi_phi*dphi_i+pi_chi*dchi_i)
    return {"scalar_momentum_density":current,"time_symmetric_value":sp.simplify(current.subs({pi_phi:0,pi_chi:0}))}


def scalar_nec_identity():
    k_dot_grad_phi,k_dot_grad_chi=sp.symbols("k_grad_Phi k_grad_chi",real=True)
    contraction=sp.expand(k_dot_grad_phi**2+k_dot_grad_chi**2)
    return {"Tkk":contraction,"sum_of_squares":True}


def unstabilized_radion_family_identity():
    """Exact conformal family exposing the C3 radion zero direction."""
    z,z0,z1,ell,c=sp.symbols("z z0 z1 ell c",positive=True)
    psi=ell/(z+c)
    bulk=sp.simplify(sp.diff(psi,z,2)-2*psi**3/ell**2)
    robin=sp.simplify(sp.diff(psi,z)+psi**2/ell)
    separation=sp.integrate(psi,(z,z0,z1))
    separation_slope=sp.simplify(sp.diff(separation,c))
    tangent=sp.simplify(sp.diff(psi,c).subs(c,0))
    return {
        "psi":psi,
        "bulk_residual":bulk,
        "robin_residual":robin,
        "proper_separation":separation,
        "proper_separation_slope":separation_slope,
        "zero_mode_tangent":tangent,
    }


def leading_gw_radion_mass_identity():
    """Derive the stiff-wall, probe-GW radion curvature symbolically."""
    w,ell,kappa,epsilon,v0,v1=sp.symbols(
        "w ell kappa5 epsilon v0 v1",positive=True
    )
    potential=4*w**4*(v1-v0*w**epsilon)**2/ell
    canonical_scale_squared=6*ell/kappa**2
    curvature=sp.diff(potential,w,2)/canonical_scale_squared
    at_minimum=sp.simplify(curvature.subs(v0,v1/w**epsilon))
    expected=4*kappa**2*epsilon**2*v1**2*w**2/(3*ell**2)
    return {
        "potential":potential,
        "canonical_scale_squared":canonical_scale_squared,
        "mass_squared_at_minimum":at_minimum,
        "expected_mass_squared":expected,
        "difference":sp.simplify(at_minimum-expected),
    }


def orbifold_ads_junction_identity():
    """Check outward-normal Z2 junction signs against exact conformal AdS."""
    z,ell,kappa=sp.symbols("z ell kappa5",positive=True);psi=ell/z
    coordinate_curvature=sp.simplify(sp.diff(psi,z)/psi**2)
    lower_curvature=-coordinate_curvature;upper_curvature=coordinate_curvature
    lower_tension=6/(kappa**2*ell);upper_tension=-6/(kappa**2*ell)
    # For S_mu_nu=-lambda h_mu_nu in four brane dimensions,
    # S_mu_nu-S h_mu_nu/3=(lambda/3)h_mu_nu.
    lower_israel=kappa**2*lower_tension/6;upper_israel=kappa**2*upper_tension/6
    return {
        "coordinate_curvature":coordinate_curvature,
        "lower_outward_curvature":lower_curvature,"upper_outward_curvature":upper_curvature,
        "lower_israel_curvature":sp.simplify(lower_israel),"upper_israel_curvature":sp.simplify(upper_israel),
        "lower_difference":sp.simplify(lower_curvature-lower_israel),
        "upper_difference":sp.simplify(upper_curvature-upper_israel),
    }


def scalar_robin_corner_identity():
    """Time-symmetric first and scalar-only second Robin compatibility."""
    alpha,gamma,pi=sp.symbols("alpha gamma Pi",real=True)
    normal_d_alpha_pi=sp.symbols("nD_alphaPi",real=True)
    acceleration,normal_d_acceleration=sp.symbols("A_Phi nD_A_Phi",real=True)
    first=normal_d_alpha_pi+sp.Rational(1,2)*gamma*alpha*pi
    first_time_symmetric=sp.simplify(first.subs({pi:0,normal_d_alpha_pi:0}))
    second_scalar_only=normal_d_acceleration+sp.Rational(1,2)*gamma*acceleration
    return {
        "first_compatibility":first,
        "first_time_symmetric_value":first_time_symmetric,
        "second_scalar_only_compatibility":second_scalar_only,
        "geometry_acceleration_remainder_required":True,
    }


def scalar_wall_energy_flux_identity():
    """Show conservative cancellation of bulk flux by half wall energy."""
    phi_t,u_prime=sp.symbols("Phi_t U_prime",real=True)
    outward_normal_derivative=-u_prime/2
    bulk_flux=phi_t*outward_normal_derivative
    half_wall_energy_rate=u_prime*phi_t/2
    return {
        "bulk_flux":bulk_flux,"half_wall_energy_rate":half_wall_energy_rate,
        "total_rate":sp.simplify(bulk_flux+half_wall_energy_rate),
    }


def israel_scalar_codazzi_identity():
    """Boundary momentum constraint from Israel plus scalar half-Robin data.

    The timelike brane is four-dimensional.  For
    ``S_ab=-(lambda+U)h_ab``, the project's oriented Israel convention gives
    ``K_ab=c h_ab`` with ``c=kappa5^2(lambda+U)/6``.  Codazzi then exactly
    matches the normal-tangential scalar stress.
    """
    kappa,u_prime,d_a_phi,n_phi=sp.symbols("kappa5 U_prime D_a_Phi n_Phi",real=True)
    d_a_c=kappa**2*u_prime*d_a_phi/6
    codazzi_momentum=sp.simplify(-3*d_a_c)
    matter_momentum=kappa**2*n_phi*d_a_phi
    robin_matter=sp.simplify(matter_momentum.subs(n_phi,-u_prime/2))
    return {
        "codazzi_momentum":codazzi_momentum,
        "scalar_normal_tangential_stress":robin_matter,
        "difference":sp.simplify(codazzi_momentum-robin_matter),
        "boundary_dimension":4,
    }


def adm_four_spatial_scalar_projection_identity():
    """Matter projection entering the 4+1 ADM ``K_ij`` evolution."""
    gradient_squared,potential,directional=sp.symbols(
        "gradient_squared potential directional_gradient_product",real=True
    )
    rho=sp.Rational(1,2)*(gradient_squared+potential)
    spatial_trace=-gradient_squared-2*potential
    spatial_tensor=directional-sp.Rational(1,2)*(gradient_squared+potential)
    projected=sp.simplify(
        spatial_tensor-sp.Rational(1,3)*(spatial_trace-rho)
    )
    expected=directional+potential/3
    return {
        "rho":rho,"spatial_trace":spatial_trace,
        "adm_matter_projection":projected,"expected":expected,
        "difference":sp.simplify(projected-expected),
    }


def israel_second_corner_gauge_covariance_identity():
    """Schematic covariance identity for a compatible Israel corner.

    Let ``C`` denote the pullback of the Israel defect tensor to a timelike
    wall and ``T`` a wall-tangent evolution vector.  A change of lapse and a
    wall-preserving shift replace it at the corner by ``T'=f T+V``.  If
    ``C=0`` and its first tangential derivatives vanish there, then

    ``(T')^2 C = f^2 T^2 C``.

    Consequently a positive lapse and tangential shift cannot turn a nonzero
    geometric second-corner tensor into zero.  The symbols representing the
    derivatives along ``V`` are kept independent before imposing the corner
    hypotheses so that the reduction itself is checked rather than assumed.
    """
    f,t_f,v_f=sp.symbols("f T_f V_f",real=True)
    t_c,v_c,t2_c,tv_c,vt_c,v2_c=sp.symbols(
        "T_C V_C T2_C T_V_C V_T_C V2_C",real=True
    )
    transformed=sp.expand(
        f*t_f*t_c+f**2*t2_c+f*tv_c+v_f*t_c+f*vt_c+v2_c
    )
    compatible=sp.simplify(transformed.subs({
        t_c:0,v_c:0,tv_c:0,vt_c:0,v2_c:0,
    }))
    expected=f**2*t2_c
    return {
        "transformed_second_derivative":transformed,
        "compatible_corner_value":compatible,
        "expected":expected,
        "difference":sp.simplify(compatible-expected),
        "hypotheses":(
            "C=0 on the wall corner, T_C=0, and V is tangent to the wall; "
            "therefore all displayed V derivatives of C and T_C vanish"
        ),
    }
