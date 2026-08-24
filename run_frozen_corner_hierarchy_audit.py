#!/usr/bin/env python3
"""Manufactured higher-corner audit for the frozen principal wave system."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.corner_hierarchy import frozen_principal_corner_hierarchy,stabilizer_acceleration_corner_residual
from bhps.gw_background import solve_gw_background
from bhps.israel_wave_matrix import coupled_robin_matrix


rng=np.random.default_rng(20260812)
cases=[]
for gamma in (20.,100.):
    background=solve_gw_background(
        np.linspace(1,np.e,257),epsilon=.1,backreaction=.01,
        wall_stiffness=gamma,tolerance=1e-11,
    )
    walls=[]
    for wall_index,index,target in (
        (0,0,background["v0"]),(1,-1,background["v1"]),
    ):
        uprime=float(gamma*(background["phi"][index]-target))
        c=float(1. if wall_index==0 else -background["beta_b"])
        robin=coupled_robin_matrix(c,uprime/6,uprime/2,gamma)["matrix"]
        seed=.01*rng.normal(size=13)
        hierarchies=[
            frozen_principal_corner_hierarchy(robin,seed,wave,10)
            for wave in (0.,.3,1.,3.)
        ]
        flat=stabilizer_acceleration_corner_residual(robin,0.,0.)
        nonflat=stabilizer_acceleration_corner_residual(robin,0.,.2)
        walls.append({
            "wall":"lower" if wall_index==0 else "upper",
            "tangential_wavenumbers":[0.,.3,1.,3.],
            "maximum_time_derivative_order":10,
            "maximum_boundary_residual_norm":float(max(
                item["maximum_boundary_residual_norm"] for item in hierarchies
            )),
            "maximum_normalized_boundary_residual":float(max(
                item["maximum_normalized_boundary_residual"] for item in hierarchies
            )),
            "maximum_propagator_commutator_norm":float(max(
                item["propagator_commutator_norm"] for item in hierarchies
            )),
            "wall_flat_pure_stabilizer_acceleration_passes":flat["passes"],
            "nonflat_control_maximum_residual":nonflat["maximum_absolute_residual"],
        })
    cases.append({"wall_stiffness":gamma,"walls":walls})

walls=[wall for case in cases for wall in case["walls"]]
payload={
    "status":"frozen_principal_higher_corner_hierarchy_manufactured_through_order_ten",
    "cases":cases,
    "all_manufactured_hierarchies_pass":bool(all(
        wall["maximum_normalized_boundary_residual"]<1e-12
        and wall["maximum_propagator_commutator_norm"]<1e-10 for wall in walls
    )),
    "all_wall_flat_acceleration_jets_pass":bool(all(
        wall["wall_flat_pure_stabilizer_acceleration_passes"] for wall in walls
    )),
    "interpretation":[
        "The frozen principal 17-field wave system admits explicit time-symmetric data satisfying every tested corner order.",
        "The previously used wall-flat stabilizer acceleration clears the complete frozen second-corner matrix because both its value and normal derivative vanish at each wall.",
        "This construction does not show that the constrained nonlinear initial data satisfy the lower-order metric corner equations.",
    ],
    "limitations":[
        "constant-coefficient principal wave operator",
        "manufactured local collar jets rather than global constrained initial data",
        "lower-order curvature, matter, and gauge-source terms omitted",
        "nonlinear higher corners remain open",
    ],
}
Path("results/frozen_corner_hierarchy_audit.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps(payload,indent=2))
