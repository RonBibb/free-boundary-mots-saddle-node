#!/usr/bin/env python3
"""Audit the pointwise covariant GH Einstein--two-scalar kernel."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.linearized_gh_einstein_scalar import (
    linearized_reduced_einstein_two_scalar_residual,
    metric_geometry_from_jets,
    reduced_einstein_two_scalar_residual,
)


def zero_jets(metric,phi=0.,chi=0.):
    n=len(metric)
    return {
        "metric":np.asarray(metric,dtype=float),
        "metric_first":np.zeros((n,n,n)),"metric_second":np.zeros((n,n,n,n)),
        "phi":float(phi),"phi_first":np.zeros(n),"phi_second":np.zeros((n,n)),
        "chi":float(chi),"chi_first":np.zeros(n),"chi_second":np.zeros((n,n)),
    }


eta=np.diag((-1.,1.,1.,1.,1.));rng=np.random.default_rng(20260815)

# Exact flat principal-part control.
flat=zero_jets(eta);flat_perturbation=zero_jets(np.zeros((5,5)))
second=rng.normal(size=(5,5,5,5));second=.5*(second+second.swapaxes(0,1));second=.5*(second+second.swapaxes(2,3))
flat_perturbation["metric_second"]=second
flat_linear=linearized_reduced_einstein_two_scalar_residual(
    flat,flat_perturbation,potential_offset=0.,
)
flat_expected=-.5*np.einsum("cd,cdab->ab",eta,second)
flat_error=float(np.max(np.abs(flat_linear["metric_residual"]-flat_expected)))

# Exact Poincare-AdS background control.
coordinate=1.7;ads=zero_jets(eta/coordinate**2);normal=4
ads["metric_first"][normal]=-2*eta/coordinate**3
ads["metric_second"][normal,normal]=6*eta/coordinate**4
ads_geometry=metric_geometry_from_jets(ads["metric"],ads["metric_first"],ads["metric_second"])
ads_residual=reduced_einstein_two_scalar_residual(
    ads["metric"],ads["metric_first"],ads["metric_second"],
    ads["phi"],ads["phi_first"],ads["phi_second"],
    ads["chi"],ads["chi_first"],ads["chi_second"],
    ads_geometry["contracted_christoffel_covector"],
    ads_geometry["contracted_christoffel_covector_first"],potential_offset=-6.,
)
ads_error=float(np.max(np.abs(ads_residual["metric_residual"])))

# Independent centered-difference replication on a generic regular jet.
metric=np.diag((-1.2,1.05,1.1,.95,1.08));background=zero_jets(metric,phi=.08,chi=.03)
first=.015*rng.normal(size=(5,5,5));first=.5*(first+first.swapaxes(1,2))
second_background=.01*rng.normal(size=(5,5,5,5));second_background=.5*(second_background+second_background.swapaxes(0,1));second_background=.5*(second_background+second_background.swapaxes(2,3))
background["metric_first"]=first;background["metric_second"]=second_background
background["phi_first"]=.02*rng.normal(size=5);background["phi_second"]=.02*rng.normal(size=(5,5));background["phi_second"]+=background["phi_second"].T
background["chi_first"]=.02*rng.normal(size=5);background["chi_second"]=.02*rng.normal(size=(5,5));background["chi_second"]+=background["chi_second"].T
perturbation=zero_jets(.03*rng.normal(size=(5,5)),phi=.04,chi=-.02)
perturbation["metric"]=.5*(perturbation["metric"]+perturbation["metric"].T)
perturbation["metric_first"]=.03*rng.normal(size=(5,5,5));perturbation["metric_first"]=.5*(perturbation["metric_first"]+perturbation["metric_first"].swapaxes(1,2))
perturbation["metric_second"]=.03*rng.normal(size=(5,5,5,5));perturbation["metric_second"]=.5*(perturbation["metric_second"]+perturbation["metric_second"].swapaxes(0,1));perturbation["metric_second"]=.5*(perturbation["metric_second"]+perturbation["metric_second"].swapaxes(2,3))
perturbation["phi_first"]=.03*rng.normal(size=5);perturbation["phi_second"]=.03*rng.normal(size=(5,5));perturbation["phi_second"]+=perturbation["phi_second"].T
perturbation["chi_first"]=.03*rng.normal(size=5);perturbation["chi_second"]=.03*rng.normal(size=(5,5));perturbation["chi_second"]+=perturbation["chi_second"].T
generic_geometry=metric_geometry_from_jets(background["metric"],background["metric_first"],background["metric_second"])
source=generic_geometry["contracted_christoffel_covector"];source_first=generic_geometry["contracted_christoffel_covector_first"]
complex_linear=linearized_reduced_einstein_two_scalar_residual(
    background,perturbation,mass_squared=.41,potential_offset=-6.,
)

def direct(sign,step):
    values={key:np.asarray(background[key])+sign*step*np.asarray(perturbation[key]) for key in background}
    return reduced_einstein_two_scalar_residual(
        values["metric"],values["metric_first"],values["metric_second"],
        values["phi"],values["phi_first"],values["phi_second"],
        values["chi"],values["chi_first"],values["chi_second"],
        source,source_first,mass_squared=.41,potential_offset=-6.,
    )

finite_difference_records=[]
for step in (1e-2,5e-3,2.5e-3):
    plus=direct(1.,step);minus=direct(-1.,step)
    metric_fd=(plus["metric_residual"]-minus["metric_residual"])/(2*step)
    phi_fd=(plus["phi_residual"]-minus["phi_residual"])/(2*step)
    chi_fd=(plus["chi_residual"]-minus["chi_residual"])/(2*step)
    numerator=np.sqrt(
        np.sum((metric_fd-complex_linear["metric_residual"])**2)
        +(phi_fd-complex_linear["phi_residual"])**2
        +(chi_fd-complex_linear["chi_residual"])**2
    )
    scale=max(1.,np.sqrt(
        np.sum(complex_linear["metric_residual"]**2)
        +complex_linear["phi_residual"]**2+complex_linear["chi_residual"]**2
    ))
    finite_difference_records.append({"step":step,"relative_error":float(numerator/scale)})

fold=json.loads(Path("results/corrected_fold_stabilizer_gradient_audit.json").read_text())
stationary=[]
for case in fold["cases"]:
    point=case["axis_stationary_points"][0]
    stationary.append({
        "name":case["name"],"z":point["z"],"phi":point["phi"],
        "orthonormal_compact_hessian":point["orthonormal_compact_hessian"],
        "orthonormal_radial_hessian":point["orthonormal_radial_hessian"],
        "metric_residual_per_delta_phi_diagonal":point["metric_residual_per_delta_phi_diagonal"],
        "scalar_residual_per_diagonal_covariant_h":point["scalar_residual_per_diagonal_covariant_h"],
        "regular_mixing_finite":point["regular_mixing_finite"],
        "mixing_depends_on_inverse_scalar_gradient":point["mixing_depends_on_inverse_scalar_gradient"],
    })

acceptance={
    "flat_principal_maximum_error_below_1e_12":flat_error<1e-12,
    "ads_background_residual_below_1e_12":ads_error<1e-12,
    "generic_centered_difference_converges_quadratically":finite_difference_records[-1]["relative_error"]<finite_difference_records[-2]["relative_error"]/3.5,
    "generic_finest_centered_difference_error_below_1e_8":finite_difference_records[-1]["relative_error"]<1e-8,
    "both_corrected_stationary_mixings_finite":all(item["regular_mixing_finite"] for item in stationary),
    "no_corrected_stationary_mixing_divides_by_gradient":all(not item["mixing_depends_on_inverse_scalar_gradient"] for item in stationary),
}
payload={
    "status":"pass" if all(acceptance.values()) else "review",
    "scope":"pointwise covariant frozen-source generalized-harmonic Einstein--two-scalar linearization kernel",
    "flat_principal_maximum_error":flat_error,
    "ads_background_maximum_residual":ads_error,
    "generic_centered_difference_replication":finite_difference_records,
    "corrected_fold_stationary_mixing":stationary,
    "acceptance":acceptance,
    "limitations":[
        "pointwise jet kernel rather than assembled axisymmetric evolution",
        "generalized-harmonic source is frozen during linearization",
        "constraint damping and evolved gauge driver are not included",
        "corrected-fold entry presently supplies the regular stationary matter-mixing block, not its complete curvature block",
    ],
}
Path("results/linearized_gh_operator_kernel_audit.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps(payload,indent=2,sort_keys=True))
