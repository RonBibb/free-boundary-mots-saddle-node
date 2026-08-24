"""Gauge-covariance diagnostics for the nonlinear Israel second corner."""

from __future__ import annotations

import numpy as np

from bhps.adm_corner import spatial_israel_second_corner_residual_fields


def maximum_tangential_residual(fields):
    """Return the largest intrinsic tangential residual in supplied scales."""
    return float(max(
        np.max(np.abs(component["residual"])/component["scale"])
        for wall in fields["walls"]
        for component in wall["tangential_components"].values()
    ))


def compare_tangential_residual_fields(reference,candidate,wall_multiplier=1.):
    """Compare residual arrays using the reference row scales.

    ``wall_multiplier`` may be a scalar or one radial array for each wall.
    It represents the covariance prediction multiplying the reference
    second-corner tensor, for example ``f**2`` under ``alpha -> f alpha``.
    """
    if np.isscalar(wall_multiplier):
        multipliers=[float(wall_multiplier)]*len(reference["walls"])
    else:
        multipliers=list(wall_multiplier)
    if len(multipliers)!=len(reference["walls"]):
        raise ValueError("one multiplier is required for each wall")
    maxima=[];l2_numerator=0.;l2_denominator=0.
    for ref_wall,new_wall,multiplier in zip(reference["walls"],candidate["walls"],multipliers):
        for name in ("radial","transverse"):
            ref=ref_wall["tangential_components"][name]
            new=new_wall["tangential_components"][name]
            predicted=np.asarray(multiplier)*ref["residual"]
            difference=new["residual"]-predicted
            scaled=difference/ref["scale"]
            maxima.append(np.max(np.abs(scaled)))
            l2_numerator+=float(np.dot(scaled,scaled))
            baseline=ref["residual"]/ref["scale"]
            l2_denominator+=float(np.dot(baseline,baseline))
    return {
        "maximum_fixed_scaled_covariance_defect":float(max(maxima)),
        "covariance_defect_l2_over_reference_l2":float(
            np.sqrt(l2_numerator/max(l2_denominator,np.finfo(float).tiny))
        ),
    }


def compare_mixed_residual_fields(reference,candidate,wall_multiplier=1.):
    """Compare the auxiliary mixed wall-gauge row in reference scales."""
    if np.isscalar(wall_multiplier):
        multipliers=[float(wall_multiplier)]*len(reference["walls"])
    else:
        multipliers=list(wall_multiplier)
    maxima=[]
    for ref_wall,new_wall,multiplier in zip(reference["walls"],candidate["walls"],multipliers):
        difference=(
            new_wall["mixed_zr_residual"]
            -np.asarray(multiplier)*ref_wall["mixed_zr_residual"]
        )
        maxima.append(np.max(np.abs(difference)/ref_wall["mixed_zr_scale"]))
    return {"maximum_fixed_scaled_mixed_response":float(max(maxima))}


def corner_fields(acceleration,psi,phi,background,scalar_acceleration=None,radial_buffer=7):
    """Small public wrapper keeping covariance callers on one row ordering."""
    return spatial_israel_second_corner_residual_fields(
        acceleration,psi,phi,background,
        stabilizer_acceleration=scalar_acceleration,
        radial_buffer=radial_buffer,
    )
