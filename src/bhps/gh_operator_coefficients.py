"""Extract 17-field lower-order coefficients from the covariant GH kernel.

The field order matches the wall-adapted boundary system: four
normal--tangential metric components, ten tangential symmetric components,
the normal--normal metric component, and the two scalars.  At a static
zero-shift background, multiplying the metric equations by ``-2`` gives the
same scalar coordinate-wave principal symbol for all seventeen fields.
"""

from __future__ import annotations

import numpy as np

from bhps.linearized_gh_einstein_scalar import (
    linearized_reduced_einstein_two_scalar_residual,metric_geometry_from_jets,
)


_METRIC_FIELDS=(
    ("h_z0",(1,0)),("h_zx",(1,2)),("h_zy",(1,3)),("h_zw",(1,4)),
    ("h00",(0,0)),("hxx",(2,2)),("hyy",(3,3)),("hww",(4,4)),
    ("h0x",(0,2)),("h0y",(0,3)),("h0w",(0,4)),
    ("hxy",(2,3)),("hxw",(2,4)),("hyw",(3,4)),
    ("h_zz",(1,1)),
)
FIELD_ORDER=tuple(name for name,_ in _METRIC_FIELDS)+("delta_Phi","delta_chi")


def _zero_perturbation():
    return {
        "metric":np.zeros((5,5)),
        "metric_first":np.zeros((5,5,5)),
        "metric_second":np.zeros((5,5,5,5)),
        "phi":0.,"phi_first":np.zeros(5),"phi_second":np.zeros((5,5)),
        "chi":0.,"chi_first":np.zeros(5),"chi_second":np.zeros((5,5)),
    }


def _set_metric_component(array,pair,value):
    left,right=pair;array[left,right]=value;array[right,left]=value


def pack_wall_adapted_residual(metric_residual,phi_residual,chi_residual):
    """Pack a symmetric metric residual and two scalar rows into 17 fields."""
    metric=np.asarray(metric_residual)
    if metric.shape!=(5,5):raise ValueError("metric residual must be 5 by 5")
    return np.array(
        [metric[pair] for _,pair in _METRIC_FIELDS]+[phi_residual,chi_residual],
        dtype=np.result_type(metric,phi_residual,chi_residual),
    )


def _linear_column(background,perturbation,mass_squared,potential_offset,kappa5_squared):
    result=linearized_reduced_einstein_two_scalar_residual(
        background,perturbation,mass_squared=mass_squared,
        potential_offset=potential_offset,kappa5_squared=kappa5_squared,
    )
    return pack_wall_adapted_residual(
        result["metric_residual"],result["phi_residual"],result["chi_residual"],
    )


def frozen_source_gh_coefficient_matrices(
    background,mass_squared=0.,potential_offset=-6.,kappa5_squared=1.,
):
    """Return zero- and first-order matrices for the normalized GH system.

    ``coordinate_first_matrices[mu]`` multiplies ``partial_mu u`` in the
    coordinate-principal equation

    ``g^ab partial_a partial_b u + B^mu partial_mu u + C u = 0``.

    ``scalar_wave_adjusted_first_matrices`` instead uses the scalar covariant
    wave operator as the common principal block.  For a diagonal static
    zero-shift metric, ``evolution_reaction_matrix=-alpha^2 C`` is the matrix
    assembled with a minus sign by the existing wave solver, while
    ``evolution_first_matrices=alpha^2 (B+Gamma^mu I)`` supplies the strong
    first-derivative acceleration terms.
    """
    geometry=metric_geometry_from_jets(
        background["metric"],background["metric_first"],background["metric_second"],
    )
    metric=np.asarray(background["metric"],dtype=float)
    if metric.shape!=(5,5) or abs(metric[0,0])<1e-15:
        raise ValueError("background must contain a nondegenerate five-metric")
    if np.max(np.abs(metric[0,1:]))>1e-12 or np.max(np.abs(metric[1:,0]))>1e-12:
        raise ValueError("evolution normalization currently requires zero shift")
    row_normalization=np.r_[np.full(15,-2.),np.ones(2)]
    zero=np.zeros((17,17));first=np.zeros((5,17,17))
    for column in range(17):
        perturbation=_zero_perturbation()
        if column<15:
            _set_metric_component(
                perturbation["metric"],_METRIC_FIELDS[column][1],1.,
            )
        elif column==15:perturbation["phi"]=1.
        else:perturbation["chi"]=1.
        zero[:,column]=row_normalization*_linear_column(
            background,perturbation,mass_squared,potential_offset,kappa5_squared,
        )
        for derivative in range(5):
            perturbation=_zero_perturbation()
            if column<15:
                pair=_METRIC_FIELDS[column][1]
                _set_metric_component(
                    perturbation["metric_first"][derivative],pair,1.,
                )
            elif column==15:perturbation["phi_first"][derivative]=1.
            else:perturbation["chi_first"][derivative]=1.
            first[derivative,:,column]=row_normalization*_linear_column(
                background,perturbation,mass_squared,potential_offset,kappa5_squared,
            )
    gamma_upper=np.asarray(geometry["contracted_christoffel_upper"],dtype=float)
    adjusted=first+gamma_upper[:,None,None]*np.eye(17)[None,:,:]
    alpha_squared=-1/metric[0,0]
    principal=np.asarray(geometry["inverse_metric"],dtype=float)
    return {
        "field_order":FIELD_ORDER,
        "metric_pairs":tuple(pair for _,pair in _METRIC_FIELDS),
        "row_normalization":row_normalization,
        "inverse_metric_principal":principal,
        "zero_order_matrix":zero,
        "coordinate_first_matrices":first,
        "contracted_christoffel_upper":gamma_upper,
        "scalar_wave_adjusted_first_matrices":adjusted,
        "evolution_reaction_matrix":-alpha_squared*zero,
        "evolution_first_matrices":alpha_squared*adjusted,
        "finite":bool(
            np.all(np.isfinite(zero)) and np.all(np.isfinite(first))
            and np.all(np.isfinite(principal))
        ),
        "assumptions":[
            "frozen generalized-harmonic source during linearization",
            "static zero-shift background for evolution normalization",
            "no constraint damping or source-driver perturbation",
        ],
    }
