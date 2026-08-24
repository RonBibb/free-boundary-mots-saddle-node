"""Shared-shape nonlinear corner repair across grids or radial domains."""

from __future__ import annotations

import numpy as np

from bhps.anisotropic_initial_data import solve_anisotropic_initial_data
from bhps.physical_corner_corrector import (
    combine_shape_modes,
    physical_corner_state,
    relinearized_projected_corner_jacobian,
    tracefree_shape_basis,
)


def solve_shared_physical_corner(
    cases,initial_coefficients,radial_modes=6,axis_widths=(),basis_radius=None,
    annular_profiles=(),
    stencil_width=7,finite_difference_step=2e-4,maximum_iterations=3,
    corner_tolerance=.025,shape_bound=.5,initial_trust_radius=.04,
    regularization=1.,selector_tolerance=1e-9,difference_scheme="forward",
    maximum_row_weight=6.,include_mixed=True,verbose=False,
):
    """Fit one physical shape coefficient vector to several selected slices.

    Every case supplies ``z,r,reference_q,reference_phi,background,chi_r,chi_z``
    and ``radial_buffer``; ``chi`` and initial selector fields are optional.
    The nonlinear selector is solved separately in each domain, while the
    anisotropic shape coefficients are shared exactly.
    """
    if not cases:raise ValueError("at least one case is required")
    prepared=[];labels=None
    for source in cases:
        case=dict(source);z=np.asarray(case["z"]);r=np.asarray(case["r"])
        reference_q=np.asarray(case["reference_q"]);reference_phi=np.asarray(case["reference_phi"])
        basis=tracefree_shape_basis(
            z,r,radial_modes,axis_widths,basis_radius,annular_profiles,
        )
        if labels is None:labels=basis["labels"]
        elif basis["labels"]!=labels:raise ValueError("case shape bases do not share labels")
        zero=np.zeros_like(reference_q)
        base=physical_corner_state(
            z,r,reference_q,reference_phi,zero,zero,zero,case["background"],
            case["chi_r"],case["chi_z"],case.get("chi"),None,stencil_width,
            int(case["radial_buffer"]),include_mixed,
        )
        case.update(
            z=z,r=r,reference_q=reference_q,reference_phi=reference_phi,
            modes=basis["modes"],scales=base["scales"],base_state=base,
        )
        prepared.append(case)
    count=len(labels);coefficients=np.asarray(initial_coefficients,dtype=float).copy()
    if coefficients.shape!=(count,):raise ValueError("invalid shared coefficient shape")
    trust=float(initial_trust_radius);reg=float(regularization)

    def solve_case(case,candidate,initial_q=None,initial_phi=None):
        a,b,c=combine_shape_modes(candidate,case["modes"])
        selected=solve_anisotropic_initial_data(
            case["z"],case["r"],case["reference_q"],case["reference_phi"],
            a,b,c,case["background"],case["chi_r"],case["chi_z"],
            initial_q=initial_q,initial_phi=initial_phi,stencil_width=stencil_width,
            tolerance=selector_tolerance,iterations=30,
        )
        if not selected["converged"]:return None,selected
        state=physical_corner_state(
            case["z"],case["r"],selected["q"],selected["phi"],a,b,c,
            case["background"],case["chi_r"],case["chi_z"],case.get("chi"),
            case["scales"],stencil_width,int(case["radial_buffer"]),include_mixed,
        )
        return state,selected

    states=[];selectors=[]
    for case in prepared:
        state,selected=solve_case(
            case,coefficients,case.get("initial_q"),case.get("initial_phi"),
        )
        if state is None:raise RuntimeError(f"initial selector failed for {case.get('name','case')}")
        states.append(state);selectors.append(selected)
    history=[];linearizations=[]
    for iteration in range(int(maximum_iterations)+1):
        maxima=[state["maximum_fixed_scaled_residual"] for state in states]
        l2s=[state["fixed_scaled_residual_l2"] for state in states]
        maximum_shape=float(max(
            max(np.max(np.abs(state[name])) for name in ("a","b","c"))
            for state in states
        ))
        history.append({
            "iteration":iteration,"worst_case_maximum":float(max(maxima)),
            "combined_residual_l2":float(np.sqrt(sum(value*value for value in l2s))),
            "case_maxima":[float(value) for value in maxima],
            "case_l2":[float(value) for value in l2s],
            "case_selector_maxima":[float(item["maximum_residual"]) for item in selectors],
            "coefficient_l2":float(np.linalg.norm(coefficients)),
            "maximum_shape_logarithm":maximum_shape,"trust_radius":trust,
            "regularization":reg,
        })
        if max(maxima)<corner_tolerance or iteration==int(maximum_iterations):break

        local=[]
        for case,state in zip(prepared,states):
            local.append(relinearized_projected_corner_jacobian(
                case["z"],case["r"],state["q"],state["phi"],state["a"],
                state["b"],state["c"],case["modes"],case["reference_q"],
                case["reference_phi"],case["background"],case["chi_r"],
                case["chi_z"],case.get("chi"),case["scales"],stencil_width,
                int(case["radial_buffer"]),finite_difference_step,
                difference_scheme,include_mixed,
            ))
        matrix=np.vstack([item["matrix"] for item in local])
        residual=np.concatenate([state["vector"] for state in states])
        singular=np.linalg.svd(matrix,compute_uv=False)
        threshold=singular[0]*max(matrix.shape)*np.finfo(float).eps
        rank=int(np.count_nonzero(singular>threshold))
        linearizations.append({
            "iteration":iteration,"rank":rank,
            "largest_singular_value":float(singular[0]),
            "smallest_retained_singular_value":float(singular[rank-1]),
            "case_projection_ratios":[float(item["maximum_projection_ratio"]) for item in local],
        })
        global_max=max(maxima)
        weights=np.sqrt(1+float(maximum_row_weight)*(
            np.abs(residual)/max(global_max,1e-300)
        )**4)
        old_merit=float(np.dot(residual,residual)+maximum_row_weight*global_max**2)
        accepted=None
        for local_reg in (reg,reg*10,reg/10):
            augmented=np.vstack((weights[:,None]*matrix,np.sqrt(local_reg)*np.eye(count)))
            rhs=np.concatenate((-weights*residual,np.zeros(count)))
            step=np.linalg.lstsq(augmented,rhs,rcond=1e-10)[0]
            norm=np.linalg.norm(step)
            if norm>trust:step*=trust/norm
            for fraction in (1.,.5,.25):
                candidate=coefficients+fraction*step
                candidate_states=[];candidate_selectors=[];valid=True
                for case,state,linear in zip(prepared,states,local):
                    dq=sum(value*response["dq"] for value,response in zip(step,linear["responses"]))
                    dphi=sum(value*response["dphi"] for value,response in zip(step,linear["responses"]))
                    new_state,new_selector=solve_case(
                        case,candidate,state["q"]+fraction*dq,state["phi"]+fraction*dphi,
                    )
                    if new_state is None or max(
                        np.max(np.abs(new_state["a"])),np.max(np.abs(new_state["b"])),
                        np.max(np.abs(new_state["c"])),
                    )>shape_bound:
                        valid=False;break
                    candidate_states.append(new_state);candidate_selectors.append(new_selector)
                if not valid:continue
                candidate_max=max(item["maximum_fixed_scaled_residual"] for item in candidate_states)
                candidate_residual=np.concatenate([item["vector"] for item in candidate_states])
                merit=float(np.dot(candidate_residual,candidate_residual)+maximum_row_weight*candidate_max**2)
                if merit<old_merit and candidate_max<max(global_max*1.02,corner_tolerance):
                    accepted=(candidate,candidate_states,candidate_selectors,local_reg,fraction,merit/old_merit)
                    break
            if accepted is not None:break
        if accepted is None:
            trust*=.5;reg*=10;history[-1]["step_accepted"]=False
            if trust<1e-3:break
            continue
        coefficients,states,selectors,reg,fraction,ratio=accepted
        trust=min(trust*(1.35 if fraction==1 else .8),.2)
        history[-1].update(
            step_accepted=True,accepted_fraction=float(fraction),merit_ratio=float(ratio),
        )
        if verbose:print({
            "iteration":iteration+1,
            "worst_case_maximum":max(item["maximum_fixed_scaled_residual"] for item in states),
            "case_maxima":[item["maximum_fixed_scaled_residual"] for item in states],
            "selector_maxima":[item["maximum_residual"] for item in selectors],
        },flush=True)
    maxima=[state["maximum_fixed_scaled_residual"] for state in states]
    return {
        "converged":bool(max(maxima)<corner_tolerance),"coefficients":coefficients,
        "labels":labels,"states":states,"selectors":selectors,"history":history,
        "linearizations":linearizations,"final_case_maxima":[float(x) for x in maxima],
        "final_worst_case_maximum":float(max(maxima)),
        "settings":{
            "case_count":len(prepared),"mode_count":count,"radial_modes":int(radial_modes),
            "axis_widths":[float(x) for x in axis_widths],
            "basis_radius":None if basis_radius is None else float(basis_radius),
            "annular_profiles":[[float(center),float(width)] for center,width in annular_profiles],
            "corner_tolerance":float(corner_tolerance),"shape_bound":float(shape_bound),
            "maximum_iterations":int(maximum_iterations),"difference_scheme":difference_scheme,
        },
    }
