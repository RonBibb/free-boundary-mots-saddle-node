"""Orbifold vector-gauge reachability on a warped interval.

For the stabilized ``S1/Z2`` background, ``h_{mu y}`` and ``xi_y`` are odd,
whereas ``xi_mu`` is even.  The linear gauge transformation is

    h'_{mu y}=h_{mu y}-(partial_mu xi_y+partial_y xi_mu+2 A_y xi_mu).

Writing ``F_mu=exp(2A) xi_mu`` turns gauge reachability into the elementary
first-order equation ``F_mu'=exp(2A)(h_{mu y}-partial_mu xi_y)``.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def gauge_away_mixed_component(y,warp_exponent,h_mu_y,d_mu_xi_y,lower_tangent_gauge=0.0):
    """Construct ``xi_mu`` that sends one component of ``h_{mu y}`` to zero."""
    y=np.asarray(y,dtype=float);warp=np.asarray(warp_exponent,dtype=float)
    mixed=np.asarray(h_mu_y,dtype=float);normal_gauge_gradient=np.asarray(d_mu_xi_y,dtype=float)
    if y.ndim!=1 or len(y)<3 or any(array.shape!=y.shape for array in (warp,mixed,normal_gauge_gradient)) or np.any(np.diff(y)<=0):
        raise ValueError("matching arrays on an ordered one-dimensional grid are required")
    source=np.exp(2*warp)*(mixed-normal_gauge_gradient)
    source_spline=CubicSpline(y,source)
    primitive=source_spline.antiderivative()
    weighted_parameter=float(lower_tangent_gauge)+primitive(y)-primitive(y[0])
    xi_mu=np.exp(-2*warp)*weighted_parameter
    # Evaluate the transformation with the derivative defined by the same
    # interpolant.  This avoids replacing the gauge identity by a stencil test.
    warp_y=CubicSpline(y,warp).derivative()(y)
    xi_mu_y=np.exp(-2*warp)*(source-2*warp_y*weighted_parameter)
    transformed=mixed-normal_gauge_gradient-xi_mu_y-2*warp_y*xi_mu
    return {
        "xi_mu":xi_mu,"xi_mu_y":xi_mu_y,"transformed_h_mu_y":transformed,
        "maximum_transformed_residual":float(np.max(np.abs(transformed))),
        "source_at_walls":np.array((source[0],source[-1])),
        "wall_preserving_normal_gauge":bool(abs(normal_gauge_gradient[0])<1e-12 and abs(normal_gauge_gradient[-1])<1e-12),
        "residual_tangent_gauge":"xi_mu=exp(-2A) epsilon_mu(x)",
    }

