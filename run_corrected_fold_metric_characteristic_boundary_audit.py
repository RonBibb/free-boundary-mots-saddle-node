#!/usr/bin/env python3
"""Audit the incoming metric-characteristic constraint projector on the fold."""

import json,sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.constraint_ibvp import (
    RegularSO3ConstraintBoundaryFeedback,
    RegularSO3OuterMetricConstraintFeedback,
    gh_incoming_metric_constraint_extraction,
    gh_incoming_metric_constraint_lift,
    gh_metric_characteristic_projectors,
    regular_so3_boundary_characteristic_count,
    regular_so3_metric_characteristic_projector_matrices,
)
from bhps.gh_source_driver import AxisymmetricDrivenGHWaveIBVP
from run_corrected_fold_free_constraint_pulse import (
    interpolate_constraint_coefficients,sample_constraint_coefficients,
)
from run_corrected_fold_gh_driver_runtime import (
    CONSTRAINT_DAMPING,DRIVER_ETA,DRIVER_MU,
    interpolate_source_coefficients,sample_source_coefficients,
)
from run_corrected_fold_regular_so3_runtime import (
    build_geometry,manufactured_problem,sample_coefficients,
)


RADII=np.array((.125,.25,.5,1.,2.,3.,4.))


def boundary_samples(geometry):
    """Return compact-wall and outer-radial boundary sample points."""
    z0=float(geometry["z"][0]);z1=float(geometry["z"][-1])
    samples=[]
    for radius in RADII:
        samples.append(("compact_lower",z0,float(radius),"compact",-1.))
        samples.append(("compact_upper",z1,float(radius),"compact",1.))
    for z_value in np.geomspace(z0,z1,5):
        samples.append(("radial_outer",float(z_value),4.,"radial",1.))
    return samples


def sample_projectors(geometry):
    records=[];matrices={}
    for boundary,z_value,radius,direction,sign in boundary_samples(geometry):
        metric=geometry["jet_field"].at(z_value,radius)["metric"]
        result=regular_so3_metric_characteristic_projector_matrices(
            metric,radius,direction,sign,
        )
        key=(boundary,z_value,radius)
        matrices[key]=np.stack(tuple(result[name] for name in (
            "gauge","constraint","physical",
        )))
        records.append({
            "boundary":boundary,"z":z_value,"r":radius,
            "regular_ranks":result["ranks"],
            "full_ranks":result["full_projector_diagnostics"]["full_ranks"],
            "regular_completeness_defect":result["completeness_defect"],
            "regular_idempotence_defect":result["idempotence_defect"],
            "regular_orthogonality_defect":result["orthogonality_defect"],
            "full_completeness_defect":result["full_projector_diagnostics"]["completeness_defect"],
            "full_idempotence_defect":result["full_projector_diagnostics"]["idempotence_defect"],
            "full_orthogonality_defect":result["full_projector_diagnostics"]["orthogonality_defect"],
            "projector_frobenius_norm":float(np.linalg.norm(matrices[key])),
        })
    return records,matrices


def transfer_audit(g5_matrices,g6_matrices):
    differences=[]
    for key in g6_matrices:
        difference=np.linalg.norm(g5_matrices[key]-g6_matrices[key])
        differences.append(float(difference/max(np.linalg.norm(g6_matrices[key]),1e-300)))
    return {
        "sample_count":len(differences),
        "maximum_g5_g6_relative_difference":max(differences),
        "rms_g5_g6_relative_difference":float(np.sqrt(np.mean(np.square(differences)))),
    }


def outer_domain_transfer_audit(g5,g6):
    """Compare full outer projectors across fold and radial-domain controls."""
    z_values=np.geomspace(g6["z"][0],g6["z"][-1],5);projectors={}
    for name,geometry in (("G5",g5),("G6",g6)):
        for radius in (4.,6.):
            values=[]
            for z_value in z_values:
                metric=geometry["jet_field"].at(float(z_value),radius)["metric"]
                inverse=np.linalg.inv(metric);lapse=1/np.sqrt(-inverse[0,0])
                time=np.array((-lapse,0.,0.,0.,0.));normal=np.zeros(5)
                normal[2]=1/np.sqrt(inverse[2,2])
                result=gh_metric_characteristic_projectors(metric,time,normal)
                values.append(np.stack(tuple(
                    result[key] for key in ("gauge","constraint","physical")
                )))
            projectors[name,radius]=np.asarray(values)
    relative=lambda left,right:float(
        np.linalg.norm(left-right)/max(np.linalg.norm(right),1e-300)
    )
    return {
        "basis":"full five-dimensional coordinate tensor projectors",
        "g5_g6_relative_difference_r4":relative(
            projectors["G5",4.],projectors["G6",4.],
        ),
        "g5_g6_relative_difference_r6":relative(
            projectors["G5",6.],projectors["G6",6.],
        ),
        "g6_r4_r6_outer_location_sensitivity":relative(
            projectors["G6",4.],projectors["G6",6.],
        ),
    }


def representative_geometry(geometry):
    z_value=float(np.sqrt(geometry["z"][0]*geometry["z"][-1]));radius=4.
    metric=geometry["jet_field"].at(z_value,radius)["metric"]
    inverse=np.linalg.inv(metric);lapse=float(1/np.sqrt(-inverse[0,0]))
    t=np.array((-lapse,0.,0.,0.,0.));n=np.zeros(5)
    n[2]=1/np.sqrt(inverse[2,2])
    return z_value,radius,metric,t,n,lapse


def lift_identity_audit(geometry):
    z_value,radius,metric,t,n,lapse=representative_geometry(geometry)
    projectors=gh_metric_characteristic_projectors(metric,t,n)
    rng=np.random.default_rng(260814);records=[]
    for _ in range(8):
        constraint=rng.normal(size=5)
        correction=gh_incoming_metric_constraint_lift(
            metric,t,n,constraint,lapse,0.,
        )
        extracted=gh_incoming_metric_constraint_extraction(metric,t,n,correction)
        projected={
            name:np.einsum("abcd,cd->ab",projectors[name],correction)
            for name in ("gauge","constraint","physical")
        }
        records.append({
            "removal_identity_relative_defect":float(
                np.linalg.norm(extracted+lapse*constraint)
                /max(lapse*np.linalg.norm(constraint),1e-300)
            ),
            "constraint_sector_relative_defect":float(
                np.hypot(np.linalg.norm(projected["gauge"]),np.linalg.norm(projected["physical"]))
                /max(np.linalg.norm(correction),1e-300)
            ),
        })
    return {
        "z":z_value,"r":radius,"lapse":lapse,"records":records,
        "maximum_removal_identity_relative_defect":max(
            item["removal_identity_relative_defect"] for item in records
        ),
        "maximum_constraint_sector_relative_defect":max(
            item["constraint_sector_relative_defect"] for item in records
        ),
    }


def rk4_step(state,time_step,rhs):
    k1=rhs(state);k2=rhs(state+.5*time_step*k1)
    k3=rhs(state+.5*time_step*k2);k4=rhs(state+time_step*k3)
    return state+time_step*(k1+2*k2+2*k3+k4)/6


def frozen_bjorhus_audit(geometry):
    z_value,radius,metric,t,n,lapse=representative_geometry(geometry)
    constraint0=np.array((.17,-.11,.07,.13,-.09))
    state0=-gh_incoming_metric_constraint_lift(metric,t,n,constraint0,1.,0.)
    initial_extracted=gh_incoming_metric_constraint_extraction(metric,t,n,state0)
    final_time=.5/lapse;records=[]

    def rhs(state):
        constraint=gh_incoming_metric_constraint_extraction(metric,t,n,state)
        return gh_incoming_metric_constraint_lift(metric,t,n,constraint,lapse,0.)

    for steps in (1,2,4,8,16):
        state=state0.copy();time_step=final_time/steps
        for _ in range(steps):state=rk4_step(state,time_step,rhs)
        constraint=gh_incoming_metric_constraint_extraction(metric,t,n,state)
        exact=constraint0*np.exp(-lapse*final_time)
        records.append({
            "steps":steps,"time_step":time_step,
            "constraint_relative_error":float(
                np.linalg.norm(constraint-exact)/np.linalg.norm(exact)
            ),
            "constraint_l2_ratio":float(np.linalg.norm(constraint)/np.linalg.norm(constraint0)),
        })
    errors=np.array([item["constraint_relative_error"] for item in records])
    return {
        "z":z_value,"r":radius,"lapse":lapse,"final_time":final_time,
        "initial_extraction_relative_defect":float(
            np.linalg.norm(initial_extracted-constraint0)/np.linalg.norm(constraint0)
        ),
        "exact_constraint_l2_ratio":float(np.exp(-lapse*final_time)),
        "records":records,
        "convergence_rates":[
            float(value) for value in np.log(errors[:-1]/errors[1:])/np.log(2.)
        ],
    }


def corrected_fold_runtime_interface_audit(geometry):
    """Apply live outer-face feedback inside the driven wave RHS."""
    print("sampling live corrected-fold runtime coefficients",flush=True)
    wave_coefficients=sample_coefficients(
        geometry,constraint_damping=CONSTRAINT_DAMPING,
    )
    source_coefficients=sample_source_coefficients(geometry,CONSTRAINT_DAMPING)
    constraint_coefficients=sample_constraint_coefficients(geometry)
    wave,_,base_values,_,base_left,base_right=manufactured_problem(
        geometry,wave_coefficients,9,13,
    )
    source_zero,source_first=interpolate_source_coefficients(
        source_coefficients,wave.z,wave.r,
    )
    driver=AxisymmetricDrivenGHWaveIBVP(
        wave,source_zero,source_first,DRIVER_MU,DRIVER_ETA,
        radial_first_is_scaled=True,
    )
    constraint_zero,constraint_first=interpolate_constraint_coefficients(
        constraint_coefficients,wave.z,wave.r,
    )
    live_constraint=RegularSO3ConstraintBoundaryFeedback(
        wave.z,wave.r,constraint_zero,constraint_first,5,
        radial_first_is_scaled=True,
    )
    metrics=np.asarray([
        geometry["jet_field"].at(z_value,wave.r[-1])["metric"]
        for z_value in wave.z
    ])
    outer=RegularSO3OuterMetricConstraintFeedback(
        live_constraint,metrics,include_corners=False,
    )
    zz,rr=np.meshgrid(wave.z,wave.r,indexing="ij")
    position=.01*base_values;velocity=np.zeros_like(position)
    source=np.stack((
        .003*(1+.1*zz)*(1-.05*rr**2),
        -.002*(1-.07*zz)*(1+.03*rr**2),
        .0015*(1+.04*zz-.02*rr**2),
    ),axis=2)
    memory=np.zeros_like(source);target=np.zeros_like(source)
    left=lambda time,rq:.01*base_left(time,rq)
    right=lambda time,rq:.01*base_right(time,rq)
    baseline=driver.rhs(
        0.,position,velocity,source,memory,target,None,left,right,
    )
    corrected=driver.rhs(
        0.,position,velocity,source,memory,target,None,left,right,
        metric_boundary_constraint_feedback=outer,
    )
    boundary=outer.evaluate(position,velocity,source)
    mask=boundary["incoming_mask"]
    expected=-boundary["lapse"][:,None,None]*boundary["characteristic_correction"]
    observed=corrected[1]-baseline[1]
    active_defect=float(
        np.linalg.norm(observed[mask]-expected[mask])
        /max(np.linalg.norm(expected[mask]),1e-300)
    )
    outside=np.array(observed,copy=True);outside[mask]=0.
    outside_defect=float(np.max(np.abs(outside)))
    sector_defect=0.
    for i in range(1,len(wave.z)-1):
        projector=regular_so3_metric_characteristic_projector_matrices(
            metrics[i],wave.r[-1],"radial",1.,
        )
        correction=boundary["characteristic_correction"][i,-1,:7]
        sector_defect=max(sector_defect,float(np.hypot(
            np.linalg.norm(projector["gauge"]@correction),
            np.linalg.norm(projector["physical"]@correction),
        )/max(np.linalg.norm(correction),1e-300)))
    setup={
        "driver":driver,"outer":outer,"position":position,"velocity":velocity,
        "source":source,"memory":memory,"target":target,"left":left,"right":right,
    }
    return setup,{
        "grid_size":[len(wave.z),len(wave.r)],
        "active_outer_node_count":int(np.sum(mask)),
        "expected_active_outer_node_count":len(wave.z)-2,
        "live_constraint_l2":float(np.linalg.norm(boundary["constraint"][mask])),
        "rhs_active_relative_defect":active_defect,
        "rhs_outside_maximum_defect":outside_defect,
        "constraint_sector_relative_defect":sector_defect,
        "lower_corner_active":bool(mask[0,-1]),
        "upper_corner_active":bool(mask[-1,-1]),
        "corner_policy":boundary["corner_policy"],
    }


def corrected_fold_runtime_timestep_audit(setup,final_time=.03):
    driver=setup["driver"]
    initial=(setup["position"],setup["velocity"],setup["source"],setup["memory"])
    arguments=(
        setup["target"],final_time,None,setup["left"],setup["right"],
    )

    def evolve(steps):
        return driver.integrate(
            *initial,arguments[0],arguments[1],final_time/steps,
            arguments[2],arguments[3],arguments[4],
            metric_boundary_constraint_feedback=setup["outer"],
        )

    print("running live outer-feedback timestep refinement",flush=True)
    reference=evolve(256);reference_state=tuple(
        reference[name] for name in ("position","velocity","source","memory")
    )
    records=[]
    scale=np.sqrt(sum(np.linalg.norm(value)**2 for value in reference_state))
    for steps in (8,16,32,64):
        result=evolve(steps);state=tuple(
            result[name] for name in ("position","velocity","source","memory")
        )
        error=np.sqrt(sum(
            np.linalg.norm(value-exact)**2
            for value,exact in zip(state,reference_state)
        ))
        boundary=setup["outer"].evaluate(
            result["position"],result["velocity"],result["source"],final_time,
        )
        mask=boundary["incoming_mask"]
        records.append({
            "steps":steps,"time_step":result["time_step"],
            "combined_relative_error":float(error/max(scale,1e-300)),
            "outer_constraint_l2":float(np.linalg.norm(boundary["constraint"][mask])),
            "corner_position_maximum":float(max(
                np.max(np.abs(result["position"][0,-1])),
                np.max(np.abs(result["position"][-1,-1])),
            )),
        })
    errors=np.array([item["combined_relative_error"] for item in records])
    return {
        "final_time":final_time,"reference_steps":reference["steps"],
        "records":records,"convergence_rates":[
            float(value) for value in np.log(errors[:-1]/errors[1:])/np.log(2.)
        ],
    }


def main():
    print("building corrected G5 geometry",flush=True);g5=build_geometry("G5")
    print("building corrected G6 geometry",flush=True);g6=build_geometry("G6")
    print("sampling boundary characteristic projectors",flush=True)
    records5,matrices5=sample_projectors(g5);records6,matrices6=sample_projectors(g6)
    transfer=transfer_audit(matrices5,matrices6)
    outer_transfer=outer_domain_transfer_audit(g5,g6)
    lift=lift_identity_audit(g6);bjorhus=frozen_bjorhus_audit(g6)
    runtime_setup,runtime_interface=corrected_fold_runtime_interface_audit(g6)
    runtime_time=corrected_fold_runtime_timestep_audit(runtime_setup)
    boundary_count=regular_so3_boundary_characteristic_count(2)
    all_records=records5+records6
    maximum_algebra_defect=max(
        max(item[key] for item in all_records)
        for key in (
            "regular_completeness_defect","regular_idempotence_defect",
            "regular_orthogonality_defect","full_completeness_defect",
            "full_idempotence_defect","full_orthogonality_defect",
        )
    )
    acceptance={
        "all_full_ranks_are_5_5_5":all(
            item["full_ranks"]=={"gauge":5,"constraint":5,"physical":5}
            for item in all_records
        ),
        "all_regular_ranks_are_3_3_1":all(
            item["regular_ranks"]=={"gauge":3,"constraint":3,"physical":1}
            for item in all_records
        ),
        "all_projector_entries_are_finite":all(
            np.isfinite(item["projector_frobenius_norm"]) for item in all_records
        ),
        "maximum_projector_algebra_defect_below_1e_10":maximum_algebra_defect<1e-10,
        "g5_g6_transfer_difference_below_2_percent":transfer["maximum_g5_g6_relative_difference"]<.02,
        "outer_projector_g5_g6_transfer_below_2_percent_at_r4_r6":max(
            outer_transfer["g5_g6_relative_difference_r4"],
            outer_transfer["g5_g6_relative_difference_r6"],
        )<.02,
        "outer_location_sensitivity_below_5_percent":outer_transfer["g6_r4_r6_outer_location_sensitivity"]<.05,
        "lift_removal_identity_defect_below_1e_12":lift["maximum_removal_identity_relative_defect"]<1e-12,
        "lift_is_constraint_sector_only_below_1e_12":lift["maximum_constraint_sector_relative_defect"]<1e-12,
        "initial_frozen_extraction_defect_below_1e_12":bjorhus["initial_extraction_relative_defect"]<1e-12,
        "all_frozen_rk4_rates_above_3p8":min(bjorhus["convergence_rates"])>3.8,
        "finest_frozen_constraint_error_below_1e_7":bjorhus["records"][-1]["constraint_relative_error"]<1e-7,
        "finest_frozen_decay_ratio_matches_exact":abs(
            bjorhus["records"][-1]["constraint_l2_ratio"]-bjorhus["exact_constraint_l2_ratio"]
        )<1e-7,
        "regular_physical_and_outer_face_counts_close":boundary_count["both_face_counts_close"],
        "live_outer_constraint_is_nonzero":runtime_interface["live_constraint_l2"]>1e-5,
        "runtime_rhs_correction_matches_bjorhus_map_below_1e_11":runtime_interface["rhs_active_relative_defect"]<1e-11,
        "runtime_rhs_changes_only_outer_mask_below_1e_12":runtime_interface["rhs_outside_maximum_defect"]<1e-12,
        "runtime_correction_is_constraint_sector_only_below_1e_11":runtime_interface["constraint_sector_relative_defect"]<1e-11,
        "physical_wall_corners_retain_precedence":not runtime_interface["lower_corner_active"] and not runtime_interface["upper_corner_active"],
        "all_live_runtime_rk4_rates_above_3p5":min(runtime_time["convergence_rates"])>3.5,
        "finest_live_runtime_error_below_1e_8":runtime_time["records"][-1]["combined_relative_error"]<1e-8,
    }
    payload={
        "status":"pass" if all(acceptance.values()) else "review",
        "scope":"five-dimensional incoming metric-wave constraint projector and live outer-radial RHS correction on the corrected fold",
        "dimension":5,"transverse_trace_factor":"1/(D-2)=1/3",
        "project_sign_convention":"C_a=Gamma_a-H_a; E27 removes incoming c^(0-)_a",
        "sample_count_per_geometry":len(records6),
        "maximum_projector_algebra_defect":maximum_algebra_defect,
        "g5_records":records5,"g6_records":records6,"transfer":transfer,
        "outer_domain_transfer":outer_transfer,
        "lift_identity":lift,"frozen_bjorhus_evolution":bjorhus,
        "boundary_count":boundary_count,"live_runtime_interface":runtime_interface,
        "live_runtime_timestep_convergence":runtime_time,
        "acceptance":acceptance,
        "limitations":[
            "compact-wall projectors are algebra diagnostics, not extra boundary conditions",
            "static zero-shift corrected-fold background",
            "outer-face correction is an additive principal Bjorhus term in the second-order linear runtime",
            "complementary outer gauge/physical data retain the existing homogeneous treatment",
            "linear corrected-fold system rather than nonlinear Einstein evolution",
        ],
    }
    Path("results/corrected_fold_metric_characteristic_boundary_audit.json").write_text(
        json.dumps(payload,indent=2,sort_keys=True)+"\n"
    )
    print(json.dumps({
        "status":payload["status"],"maximum_projector_algebra_defect":maximum_algebra_defect,
        "transfer":transfer,"outer_domain_transfer":outer_transfer,"lift_identity":lift,
        "frozen_bjorhus_evolution":bjorhus,"runtime_interface":runtime_interface,
        "runtime_timestep":runtime_time,"acceptance":acceptance,
    },indent=2))


if __name__=="__main__":main()
