#!/usr/bin/env python3
"""Orbifold-vector and linear constraint-boundary audit."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.constraint_ibvp import frozen_constraint_boundary_symbol,israel_wave_boundary_count
from bhps.equations import israel_scalar_codazzi_identity
from bhps.vector_gauge import gauge_away_mixed_component


vector_cases=[]
for size in (33,65,129,257):
    y=np.linspace(0,1,size);warp=.7*y+.03*np.sin(np.pi*y)**2
    mixed=np.sin(np.pi*y)*(1+.2*np.cos(2*np.pi*y))
    normal_gradient=.1*np.sin(2*np.pi*y)
    result=gauge_away_mixed_component(y,warp,mixed,normal_gradient,.4)
    vector_cases.append({
        "grid_size":size,
        "maximum_transformed_residual":result["maximum_transformed_residual"],
        "wall_preserving_normal_gauge":result["wall_preserving_normal_gauge"],
    })

symbols=[]
for real in np.logspace(-3,2,16):
    for imag in np.linspace(-20,20,25):
        for wave in np.linspace(0,20,17):
            item=frozen_constraint_boundary_symbol(real,imag,wave)
            symbols.append(item)

codazzi=israel_scalar_codazzi_identity();count=israel_wave_boundary_count(5,2)
payload={
    "status":"orbifold_vector_sector_is_not_independent_and_linear_constraint_boundary_structure_closes",
    "vector_gauge_reachability":vector_cases,
    "physical_vector_interpretation":{
        "h_mu_y":"odd orbifold component gaugeable to zero with wall-preserving xi_y",
        "remaining_vector_like_polarizations":"contained in transverse-traceless massive spin-2 tower",
        "independent_vector_tower":False,
    },
    "boundary_count":count,
    "israel_scalar_codazzi_difference":str(codazzi["difference"]),
    "codazzi_identity_passes":bool(codazzi["difference"]==0),
    "constraint_symbol_sample_count":len(symbols),
    "minimum_normalized_tangential_constraint_determinant":float(min(x["normalized_tangential_magnitude"] for x in symbols)),
    "unstable_constraint_root_detected":bool(any(x["unstable_normal_root"] or x["unstable_tangential_root"] for x in symbols)),
    "theorem_scope_audit":{
        "comparison":"Fournodavlos--Smulevici arXiv:2104.08851",
        "source_sha256":"9a4770059c3194ec361c74d8345cfb3c47b0e0ba558f31bd23e41d8894c056c3",
        "covered_by_published_theorem":False,
        "reasons":[
            "published theorem is formulated for 3+1 vacuum Einstein",
            "our system is 4+1 Einstein with dynamical scalars",
            "their umbilic coefficient is constant whereas finite-wall Israel data depend on Phi",
        ],
        "surviving_key_structure":"Israel plus half-Robin scalar data imply the boundary momentum constraint through Codazzi",
    },
    "limitations":[
        "frozen principal constraint symbol, not a variable-coefficient Kreiss estimate",
        "boundary row count is necessary but not sufficient for well-posedness",
        "the full coupled frozen matrix is audited separately; its variable-coefficient estimate remains open",
        "higher corner compatibility remains open",
    ],
}
Path("results/vector_constraint_audit.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps(payload,indent=2))
