#!/usr/bin/env python3
"""Sealed controls and corrected-fold t=0 reduction of dynamic theta-plus."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.dynamical_capped_horizon import (
    capped_outgoing_expansion,regular_so3_adm_slice,
)
from run_corrected_fold_dynamical_horizon_t0_engineering import evaluate
from run_corrected_fold_nonlinear_rhs_G7_axis_refinement import build_refined
from run_corrected_fold_regular_so3_runtime import build_geometry


OUTPUT=Path("results/corrected_fold_dynamical_horizon_t0_reduction.json")


def relative(left,right):
    return float(abs(left-right)/max(abs(left),abs(right),1e-300))


def analytic_controls():
    z=np.linspace(0,4,49);r=np.linspace(0,3,65)
    q=np.zeros((len(z),len(r),9));q[:,:,2]=-1.;q[:,:,3]=1.;q[:,:,6]=1.
    radius=1.2;theta=np.linspace(1e-3,np.pi/2,301)
    profile={"theta":theta,"rho":np.full_like(theta,radius),"slope":np.zeros_like(theta)}
    zero=capped_outgoing_expansion(q,np.zeros_like(q),z,r,profile)
    k=.17;velocity=np.zeros_like(q);velocity[:,:,3]=-2*k;velocity[:,:,6]=-2*k
    curved=capped_outgoing_expansion(q,velocity,z,r,profile)
    shift=.21;shifted=q.copy();shifted[:,:,0]=shift;shifted[:,:,2]=-1+shift**2
    adm=regular_so3_adm_slice(shifted,np.zeros_like(q),z,r)
    keep=slice(3,-3)
    return {
        "flat_half_sphere_maximum_error":float(np.max(np.abs(
            zero["outgoing_expansion"][keep]-3/radius
        ))),
        "isotropic_extrinsic_maximum_error":float(np.max(np.abs(
            curved["outgoing_expansion"][keep]-(3/radius-3*k)
        ))),
        "constant_shift_lapse_maximum_error":float(np.max(np.abs(adm["lapse"]-1))),
        "constant_shift_extrinsic_maximum":float(max(
            np.max(np.abs(adm["extrinsic_base"])),
            np.max(np.abs(adm["extrinsic_sphere_eigenvalue"])),
        )),
    }


def main():
    controls=analytic_controls()
    print("building corrected G6/G7 A=7.94 states",flush=True)
    fold=build_geometry("G6");seed={**fold,"fold_amplitude":7.94}
    g6=build_refined(seed,65,97,"G6A794",selector_iterations=35,slice_iterations=260)
    g7=build_refined(g6,81,121,"G7A794",selector_iterations=40,slice_iterations=270)
    records=[evaluate(g6,"G6"),evaluate(g7,"G7")]
    radius_transfer={
        name:relative(
            records[0]["static_profile"][name],records[1]["static_profile"][name],
        ) for name in ("rho_axis","rho_brane")
    }
    acceptance={
        "analytic_expansion_controls_below_5e_5":bool(max(
            controls["flat_half_sphere_maximum_error"],
            controls["isotropic_extrinsic_maximum_error"],
        )<5e-5),
        "constant_shift_control_below_5e_10":bool(max(
            controls["constant_shift_lapse_maximum_error"],
            controls["constant_shift_extrinsic_maximum"],
        )<5e-10),
        "static_caps_converge_below_1e_6":bool(all(
            record["static_profile"]["surface_residual_max"]<1e-6
            and record["static_profile"]["boundary_slope_error"]<1e-8
            for record in records
        )),
        "time_symmetric_extrinsic_corrections_below_1e_12":bool(max(
            record["dynamical_expansion"]["maximum_extrinsic_correction"]
            for record in records
        )<1e-12),
        "two_cell_interior_expansions_below_5e_4":bool(max(
            record["dynamical_expansion"]["two_native_cell_interior_maximum"]
            for record in records
        )<5e-4),
        "G6_G7_cap_radii_transfer_below_0_2_percent":bool(max(
            radius_transfer.values()
        )<.002),
        "all_expansions_finite":bool(all(
            record["dynamical_expansion"]["finite"] for record in records
        )),
    }
    summary={
        "analytic_controls":controls,"radius_transfer":radius_transfer,
        "interior_expansion_residuals":{
            record["label"]:record["dynamical_expansion"]["two_native_cell_interior_maximum"]
            for record in records
        },
        "maximum_extrinsic_correction":max(
            record["dynamical_expansion"]["maximum_extrinsic_correction"]
            for record in records
        ),
    }
    payload={
        "status":"pass" if all(acceptance.values()) else "review",
        "scope":"sealed analytic controls and corrected G6/G7 t=0 reduction of the full dynamical capped-surface outgoing expansion",
        "protocol":"notes/59_dynamical_horizon_t0_reduction_protocol.md",
        "records":records,"summary":summary,"acceptance":acceptance,
        "limitations":[
            "time-symmetric validation only",
            "endpoint expansion equation is replaced by smooth-axis and orthogonal-wall boundary conditions",
            "no dynamical marginal-surface solve or increased-duration evolution",
        ],
    }
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":payload["status"],"summary":summary,"acceptance":acceptance},indent=2),flush=True)


if __name__=="__main__":main()
