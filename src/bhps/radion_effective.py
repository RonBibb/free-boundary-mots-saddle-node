"""Leading effective Goldberger--Wise radion diagnostics.

This is limited to stiff boundary potentials, small scalar backreaction, and
the leading small-epsilon potential.  It is not a coupled 5D mode solver.
"""

from __future__ import annotations

import math


def epsilon_from_mass(mphi_ell):
    if mphi_ell<0:
        raise ValueError("mphi_ell must be nonnegative")
    return math.sqrt(4+mphi_ell**2)-2


def leading_radion_mass(epsilon,b1,separation_over_ell):
    """Return ``mu_rad*ell`` for ``b1=kappa5^2*v1^2``."""
    if epsilon<0 or b1<0 or separation_over_ell<0:
        raise ValueError("epsilon, b1, and separation must be nonnegative")
    return 2/math.sqrt(3)*epsilon*math.sqrt(b1)*math.exp(-separation_over_ell)


def finite_interval_weak_radion_mass_squared(epsilon,b0,separation_over_ell):
    """First-order stiff-wall mass without the large-warp approximation.

    This is the Rayleigh shift of the unstabilized constant master mode on
    pure AdS for ``Phi=v0*z**(-epsilon)``.  It retains both interval endpoints;
    at large separation and small epsilon it reduces to the usual leading
    Goldberger--Wise expression.
    """
    if epsilon<=0 or b0<0 or separation_over_ell<=0:
        raise ValueError("require epsilon>0, b0>=0, and positive separation")
    d=float(separation_over_ell);z1=math.exp(d)
    numerator=epsilon**2*b0*(z1**2-1)*(2*epsilon+4)
    denominator=3*(z1**(2*epsilon+4)-1)
    return numerator/denominator


def gw_control_point(epsilon,b0,separation_over_ell=1.,source_width_over_ell=1.):
    if b0<0 or source_width_over_ell<=0:
        raise ValueError("b0 must be nonnegative and source width positive")
    b1=b0*math.exp(-2*epsilon*separation_over_ell)
    mu_ell=leading_radion_mass(epsilon,b1,separation_over_ell)
    return {
        "epsilon":float(epsilon),
        "b0":float(b0),
        "b1":float(b1),
        "separation_over_ell":float(separation_over_ell),
        "source_width_over_ell":float(source_width_over_ell),
        "mu_rad_times_ell":float(mu_ell),
        "mu_rad_times_source_width":float(mu_ell*source_width_over_ell),
        "necessary_heavy_condition_mu_sigma_gt_1":bool(mu_ell*source_width_over_ell>1),
        "probe_backreaction_nominal":bool(max(b0,b1)<=.1),
        "small_epsilon_nominal":bool(epsilon<=.1),
    }
