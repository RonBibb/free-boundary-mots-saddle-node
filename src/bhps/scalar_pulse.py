"""Smooth Neumann-compatible scalar pulses on an interval."""

from __future__ import annotations
import numpy as np


def neumann_image_profile(y, center, width, interval, images=5):
    """Gaussian plus reflected periodic images on ``[0, interval]``.

    The normalization is evaluated analytically at ``center`` rather than at
    the largest sampled grid value.  Consequently the represented physical
    profile is independent of resolution.
    """
    y=np.asarray(y); value=np.zeros_like(y,dtype=float); derivative=np.zeros_like(y,dtype=float)
    normalization=0.0
    for n in range(-images,images+1):
        for image_center in (center+2*n*interval,-center+2*n*interval):
            delta=y-image_center; term=np.exp(-delta**2/(2*width**2)); value+=term; derivative+=-delta/width**2*term
            normalization+=np.exp(-(center-image_center)**2/(2*width**2))
    value/=normalization
    derivative/=normalization
    return value,derivative


def scalar_pulse(z,r,amplitude=1.,sigma_r=1.,sigma_y=.2,center_fraction=.9,d_over_ell=1.,images=5):
    y=np.log(z); center=center_fraction*d_over_ell
    transverse,dtrans_dy=neumann_image_profile(y,center,sigma_y,d_over_ell,images)
    radial=np.exp(-r**2/(2*sigma_r**2)); drad=-r/sigma_r**2*radial
    chi=amplitude*transverse[:,None]*radial[None,:]
    chi_r=amplitude*transverse[:,None]*drad[None,:]
    chi_z=amplitude*(dtrans_dy/z)[:,None]*radial[None,:]
    return chi,chi_r,chi_z
