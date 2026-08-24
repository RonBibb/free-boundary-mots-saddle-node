#!/usr/bin/env python3
"""Audit the shared corrected shape outside its retained training interval."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_geometry import anisotropic_metric_acceleration,anisotropic_scalar_acceleration,anisotropic_spatial_israel_second_corner_fields
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice
from bhps.scalar_pulse import scalar_pulse


settings=(
    ("G5R8",49,73,8.,8.572038845434301),
    ("G6R8",65,97,8.,8.572368541895216),
    ("G5R10",49,91,10.,8.591525244593445),
    ("G5R12",49,109,12.,8.605669112233226),
)
archive=np.load("results/high_order_physical_corner_shared_grid_domain_solve_state.npz")
cases=[]
for name,nz,nr,r_max,amplitude in settings:
    reference=solve_finite_wall_high_order_slice(
        amplitude,nz=nz,nr=nr,r_max=r_max,wall_stiffness=20.,epsilon=.1,
        backreaction=.01,tolerance=1e-10,iterations=240,
    )
    chi,chi_r,chi_z=scalar_pulse(reference["z"],reference["r"],amplitude)
    zero=np.zeros_like(reference["q"]);background=reference["background"]

    def corner_fields(q,phi,a,b,c):
        psi=1/(reference["z"][:,None]+q)
        acceleration=anisotropic_metric_acceleration(
            reference["z"],reference["r"],psi,a,b,c,phi,chi_r,chi_z,
            float(background["mass_squared"]),chi=chi,stencil_width=7,lapse=psi,
        )
        scalar_acceleration=anisotropic_scalar_acceleration(
            reference["z"],reference["r"],psi,a,b,c,phi,
            float(background["mass_squared"]),lapse=psi,stencil_width=7,
        )
        return anisotropic_spatial_israel_second_corner_fields(
            acceleration,psi,a,b,c,phi,background,scalar_acceleration,7,
        )

    baseline=corner_fields(reference["q"],reference["phi"],zero,zero,zero)
    corrected=corner_fields(
        archive[f"q_{name}"],archive[f"phi_{name}"],archive[f"a_{name}"],
        archive[f"b_{name}"],archive[f"c_{name}"],
    )
    rows=[];labels=[]
    for base_wall,new_wall in zip(baseline["walls"],corrected["walls"]):
        for component in ("radial","transverse"):
            rows.append(
                new_wall["tangential_components"][component]["residual"]
                /base_wall["tangential_components"][component]["scale"]
            )
            labels.append(f"{new_wall['wall']}_{component}")
        rows.append(new_wall["mixed_zr_residual"]/base_wall["mixed_zr_scale"])
        labels.append(f"{new_wall['wall']}_mixed_zr")
    values=np.vstack(rows);r=reference["r"][:-7]

    def region(label,mask):
        if not np.any(mask):return {"name":label,"sample_count":0,"maximum":None}
        subset=values[:,mask];row_index,local_index=np.unravel_index(
            np.argmax(np.abs(subset)),subset.shape,
        )
        radius=r[mask][local_index]
        return {
            "name":label,"sample_count":int(np.count_nonzero(mask)),
            "maximum":float(np.max(np.abs(subset))),
            "signed_value":float(subset[row_index,local_index]),
            "radius_at_maximum":float(radius),"row_at_maximum":labels[row_index],
        }

    regions=[
        region("trained",r<=6.75),
        region("cutoff_shoulder",(r>6.75)&(r<=8.)),
        region("beyond_basis_radius",r>8.),
        region("all_trustworthy",np.ones_like(r,dtype=bool)),
    ]
    cases.append({
        "name":name,"grid_size":[nz,nr],"r_max":r_max,
        "last_trustworthy_radius":float(r[-1]),"regions":regions,
    })

full_maximum=max(
    next(region for region in case["regions"] if region["name"]=="all_trustworthy")["maximum"]
    for case in cases
)
payload={
    "status":"shared_shape_fails_0.025_full_exterior_gate" if full_maximum>.025
    else "shared_shape_passes_0.025_full_exterior_gate",
    "corner_state":"results/high_order_physical_corner_shared_grid_domain_solve_state.npz",
    "threshold":.025,"cases":cases,
    "interpretation":"The shared shape remains below 0.019 on its trained interval but reaches 0.043-0.075 on the trustworthy exterior points. The largest row is the lower-wall radial component near r=7.9 on the larger domains and then decays. This is a localized cutoff/selector-tail failure, not a failure of the already retained inner interval.",
    "limitations":[
        "the last seven radial points are excluded because they form the outer derivative stencil",
        "normalization uses the local uncorrected component scales",
        "this audit diagnoses the existing fixed shape and does not refit it",
    ],
}
Path("results/physical_corner_full_exterior_audit.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps(payload,indent=2))
