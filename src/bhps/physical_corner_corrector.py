"""Linear authority audit for physical anisotropic corner correctors."""

from __future__ import annotations

import math
import numpy as np
from scipy.sparse import diags,eye,kron,lil_matrix
from scipy.sparse.linalg import splu

from bhps.adm_corner import _axisymmetric_derivatives
from bhps.anisotropic_geometry import anisotropic_hamiltonian_residual,anisotropic_metric_acceleration,anisotropic_scalar_acceleration,anisotropic_spatial_israel_second_corner_fields,anisotropic_spatial_junction_fields,axisymmetric_diagonal_geometry
from bhps.anisotropic_initial_data import anisotropic_initial_data_jacobian,anisotropic_initial_data_residual,solve_anisotropic_initial_data
from bhps.gw_slice_high_order_solver import derivative_matrix


def compact_double_zero_hermite_basis(z):
    """Profiles with zero value/first derivative and independent wall 2/3-jets."""
    z=np.asarray(z,dtype=float);x=(z-z[0])/(z[-1]-z[0])
    matrix=np.zeros((8,8));conditions=[]
    row=0
    for endpoint in (0.,1.):
        for order in range(4):
            for power in range(order,8):
                matrix[row,power]=math.factorial(power)/math.factorial(power-order)*endpoint**(power-order)
            conditions.append((endpoint,order));row+=1
    profiles=[];labels=[]
    for endpoint,order,label in ((0.,2,"lower_second"),(0.,3,"lower_third"),(1.,2,"upper_second"),(1.,3,"upper_third")):
        target=np.zeros(8);target[conditions.index((endpoint,order))]=1.
        coefficients=np.linalg.solve(matrix,target)
        values=np.polynomial.polynomial.polyval(x,coefficients)
        normalization=np.max(np.abs(values));values/=normalization
        profiles.append(values);labels.append(label)
    return {"profiles":np.asarray(profiles),"labels":labels}


def radial_corrector_basis(r,count=10,spherical=False,basis_radius=None):
    """Smooth even radial bumps with a fourth-order zero at the outer edge."""
    r=np.asarray(r,dtype=float);radius=r[-1] if basis_radius is None else float(basis_radius)
    if radius<=0 or radius>r[-1]:raise ValueError("basis radius must be positive and lie in the grid")
    x=r/radius;count=int(count)
    centers=np.linspace(0.,.82*radius,count)
    width=max(1.25*radius/max(count-1,1),.35)
    taper=np.maximum(0.,1-x*x)**4
    profiles=[]
    for center in centers:
        values=(np.exp(-((r-center)/width)**2)+np.exp(-((r+center)/width)**2))*taper
        if spherical:values*=x*x
        maximum=np.max(np.abs(values))
        if maximum>0:values/=maximum
        profiles.append(values)
    return {"profiles":np.asarray(profiles),"centers":centers,"spherical":bool(spherical)}


def tracefree_shape_basis(
    z,r,radial_modes=10,axis_widths=(),basis_radius=None,annular_profiles=(),
):
    """Return regular unit-determinant compact and spherical shape modes."""
    compact=compact_double_zero_hermite_basis(z)
    common=radial_corrector_basis(r,radial_modes,False,basis_radius)
    spherical=radial_corrector_basis(r,radial_modes,True,basis_radius)
    modes=[];labels=[]
    for jet,profile_z in zip(compact["labels"],compact["profiles"]):
        for radial_index,profile_r in enumerate(common["profiles"]):
            shape=profile_z[:,None]*profile_r[None,:]
            modes.append((3*shape,-shape,-shape))
            labels.append({"family":"compact_vs_radial_space","wall_jet":jet,"radial_mode":radial_index})
        for radial_index,profile_r in enumerate(spherical["profiles"]):
            shape=profile_z[:,None]*profile_r[None,:]
            modes.append((np.zeros_like(shape),2*shape,-shape))
            labels.append({"family":"radial_vs_transverse","wall_jet":jet,"radial_mode":radial_index})
    radius=np.asarray(r)[-1] if basis_radius is None else float(basis_radius)
    x=np.asarray(r)/radius
    taper=np.maximum(0.,1-x*x)**4
    for jet,profile_z in zip(compact["labels"],compact["profiles"]):
        for width in axis_widths:
            width=float(width)
            if width<=0:raise ValueError("axis widths must be positive")
            localized=np.exp(-(np.asarray(r)/width)**2)*taper
            localized/=np.max(np.abs(localized))
            shape=profile_z[:,None]*localized[None,:]
            modes.append((3*shape,-shape,-shape))
            labels.append({"family":"compact_vs_radial_space_axis_localized","wall_jet":jet,"axis_width":width})
            spherical_localized=localized*x*x
            spherical_localized/=np.max(np.abs(spherical_localized))
            shape=profile_z[:,None]*spherical_localized[None,:]
            modes.append((np.zeros_like(shape),2*shape,-shape))
            labels.append({"family":"radial_vs_transverse_axis_localized","wall_jet":jet,"axis_width":width})
    # Domain-independent Gaussian tails repair selector responses beyond the
    # compact basis radius.  Analytic normalization keeps the same physical
    # mode on every truncated radial domain.
    for jet,profile_z in zip(compact["labels"],compact["profiles"]):
        for center,width in annular_profiles:
            center=float(center);width=float(width)
            if center<=0 or width<=0:
                raise ValueError("annular centers and widths must be positive")
            annular=(
                np.exp(-((np.asarray(r)-center)/width)**2)
                +np.exp(-((np.asarray(r)+center)/width)**2)
            )/(1+np.exp(-(2*center/width)**2))
            shape=profile_z[:,None]*annular[None,:]
            modes.append((3*shape,-shape,-shape))
            labels.append({
                "family":"compact_vs_radial_space_annular","wall_jet":jet,
                "center":center,"width":width,
            })
            spherical=annular*(np.asarray(r)/center)**2
            shape=profile_z[:,None]*spherical[None,:]
            modes.append((np.zeros_like(shape),2*shape,-shape))
            labels.append({
                "family":"radial_vs_transverse_annular","wall_jet":jet,
                "center":center,"width":width,
            })
    return {"modes":modes,"labels":labels}


def radial_buffer_for_cutoff(r,retained_r_max):
    """Return a point buffer that retains the same physical radial interval."""
    r=np.asarray(r,dtype=float);cutoff=float(retained_r_max)
    if not r[0]<=cutoff<r[-1]:raise ValueError("cutoff must lie inside the radial grid")
    return int(np.count_nonzero(r>cutoff))


def combine_shape_modes(coefficients,modes):
    """Combine trace-free shape basis tuples into ``a,b,c`` arrays."""
    coefficients=np.asarray(coefficients,dtype=float)
    if len(coefficients)!=len(modes):raise ValueError("coefficient and mode counts differ")
    if not modes:raise ValueError("at least one shape mode is required")
    return tuple(sum(value*mode[index] for value,mode in zip(coefficients,modes)) for index in range(3))


def _flatten_corner_fields(fields,scales=None,include_mixed=False):
    values=[];local_scales=[]
    for wall in fields["walls"]:
        for name in ("radial","transverse"):
            component=wall["tangential_components"][name]
            values.append(component["residual"]);local_scales.append(component["scale"])
        if include_mixed:
            values.append(wall["mixed_zr_residual"])
            local_scales.append(wall["mixed_zr_scale"])
    values=np.concatenate(values)
    local_scales=np.concatenate(local_scales) if scales is None else np.asarray(scales)
    return values,local_scales


def physical_corner_state(
    z,r,q,phi,a,b,c,background,chi_r,chi_z,chi=None,scales=None,
    stencil_width=7,radial_buffer=7,include_mixed=False,
):
    """Evaluate the fixed-scale spatial Israel corner vector for one state."""
    q=np.asarray(q);phi=np.asarray(phi);psi=1/(np.asarray(z)[:,None]+q)
    zero=np.zeros_like(psi);mass=float(background["mass_squared"])
    acceleration=anisotropic_metric_acceleration(
        z,r,psi,a,b,c,phi,chi_r,chi_z,mass,chi=chi,
        stencil_width=stencil_width,lapse=psi,
    )
    scalar_acceleration=anisotropic_scalar_acceleration(
        z,r,psi,a,b,c,phi,mass,lapse=psi,stencil_width=stencil_width,
    )
    fields=anisotropic_spatial_israel_second_corner_fields(
        acceleration,psi,a,b,c,phi,background,scalar_acceleration,radial_buffer,
    )
    raw,local_scales=_flatten_corner_fields(fields,scales,include_mixed)
    vector=raw/local_scales
    intrinsic=max(
        np.max(np.abs(component["residual"])/component["scale"])
        for wall in fields["walls"] for component in wall["tangential_components"].values()
    )
    if include_mixed:
        intrinsic=max(intrinsic,max(
            np.max(np.abs(wall["mixed_zr_residual"])/wall["mixed_zr_scale"])
            for wall in fields["walls"]
        ))
    return {
        "q":q,"phi":phi,"psi":psi,"a":np.asarray(a),"b":np.asarray(b),"c":np.asarray(c),
        "fields":fields,"raw":raw,"scales":local_scales,"vector":vector,
        "maximum_fixed_scaled_residual":float(np.max(np.abs(vector))),
        "fixed_scaled_residual_l2":float(np.linalg.norm(vector)),
        "maximum_intrinsic_residual":float(intrinsic),
    }


def relinearized_projected_corner_jacobian(
    z,r,current_q,current_phi,current_a,current_b,current_c,shape_modes,
    reference_q,reference_phi,background,chi_r,chi_z,chi,scales,
    stencil_width=7,radial_buffer=7,finite_difference_step=2e-4,
    difference_scheme="central",include_mixed=False,
):
    """Rebuild ``d corner/d shape`` along the nonlinear selector manifold."""
    current_q=np.asarray(current_q);current_phi=np.asarray(current_phi)
    step=float(finite_difference_step);zero=np.zeros_like(current_q)
    selector_jacobian=anisotropic_initial_data_jacobian(
        current_q,current_phi,z,r,current_a,current_b,current_c,background,
        chi_r,chi_z,reference_q,reference_phi,stencil_width,
    ).tocsc()
    factorization=splu(selector_jacobian)
    if difference_scheme not in ("central","forward"):
        raise ValueError("difference_scheme must be central or forward")
    base_selector=anisotropic_initial_data_residual(
        current_q,current_phi,z,r,current_a,current_b,current_c,background,
        chi_r,chi_z,reference_q,reference_phi,stencil_width,
    )
    base_corner=physical_corner_state(
        z,r,current_q,current_phi,current_a,current_b,current_c,background,
        chi_r,chi_z,chi,scales,stencil_width,radial_buffer,include_mixed,
    )["vector"]
    columns=[];responses=[];projection_ratios=[]
    for a_mode,b_mode,c_mode in shape_modes:
        selector_plus=anisotropic_initial_data_residual(
            current_q,current_phi,z,r,current_a+step*a_mode,current_b+step*b_mode,
            current_c+step*c_mode,background,chi_r,chi_z,reference_q,reference_phi,
            stencil_width,
        )
        if difference_scheme=="central":
            selector_minus=anisotropic_initial_data_residual(
                current_q,current_phi,z,r,current_a-step*a_mode,current_b-step*b_mode,
                current_c-step*c_mode,background,chi_r,chi_z,reference_q,reference_phi,
                stencil_width,
            )
            shape_source=(selector_plus-selector_minus)/(2*step)
        else:
            shape_source=(selector_plus-base_selector)/step
        response=factorization.solve(-shape_source)
        dq=response[:current_q.size].reshape(current_q.shape)
        dphi=response[current_q.size:].reshape(current_q.shape)
        responses.append({"dq":dq,"dphi":dphi})
        corner=[];selector=[]
        signs=(1.,-1.) if difference_scheme=="central" else (1.,)
        for sign in signs:
            local_q=current_q+sign*step*dq;local_phi=current_phi+sign*step*dphi
            local_a=current_a+sign*step*a_mode
            local_b=current_b+sign*step*b_mode
            local_c=current_c+sign*step*c_mode
            state=physical_corner_state(
                z,r,local_q,local_phi,local_a,local_b,local_c,background,
                chi_r,chi_z,chi,scales,stencil_width,radial_buffer,include_mixed,
            )
            corner.append(state["vector"])
            selector.append(anisotropic_initial_data_residual(
                local_q,local_phi,z,r,local_a,local_b,local_c,background,
                chi_r,chi_z,reference_q,reference_phi,stencil_width,
            ))
        if difference_scheme=="central":
            columns.append((corner[0]-corner[1])/(2*step))
            selector_derivative=(selector[0]-selector[1])/(2*step)
        else:
            columns.append((corner[0]-base_corner)/step)
            selector_derivative=(selector[0]-base_selector)/step
        projection_ratios.append(float(
            np.linalg.norm(selector_derivative)/max(np.linalg.norm(shape_source),1e-300)
        ))
    matrix=np.column_stack(columns);singular=np.linalg.svd(matrix,compute_uv=False)
    threshold=singular[0]*max(matrix.shape)*np.finfo(float).eps
    rank=int(np.count_nonzero(singular>threshold))
    return {
        "matrix":matrix,"responses":responses,"rank":rank,"singular_values":singular,
        "maximum_projection_ratio":float(max(projection_ratios)),
        "median_projection_ratio":float(np.median(projection_ratios)),
        "difference_scheme":difference_scheme,
    }


def solve_relinearized_physical_corner(
    z,r,reference_q,reference_phi,background,chi_r,chi_z,chi=None,
    radial_modes=6,stencil_width=7,radial_buffer=7,finite_difference_step=2e-4,
    maximum_iterations=5,corner_tolerance=.02,shape_bound=.5,
    initial_trust_radius=.08,regularization=.1,selector_tolerance=1e-9,
    difference_scheme="forward",maximum_row_weight=4.,verbose=False,
    initial_coefficients=None,initial_q=None,initial_phi=None,include_mixed=False,
    axis_widths=(),basis_radius=None,
):
    """Regularized nonlinear variable-projection solve for physical corners.

    At every accepted shape iterate, ``q,Phi`` are solved nonlinearly and the
    constraint-projected corner Jacobian is rebuilt on that corrected state.
    """
    z=np.asarray(z);r=np.asarray(r);reference_q=np.asarray(reference_q)
    reference_phi=np.asarray(reference_phi);shape_basis=tracefree_shape_basis(
        z,r,radial_modes,axis_widths,basis_radius,
    )
    modes=shape_basis["modes"];count=len(modes)
    coefficients=(
        np.zeros(count) if initial_coefficients is None
        else np.asarray(initial_coefficients,dtype=float).copy()
    )
    if coefficients.shape!=(count,):raise ValueError("invalid initial coefficient shape")
    zero=np.zeros_like(reference_q)
    current_q=reference_q.copy() if initial_q is None else np.asarray(initial_q).copy()
    current_phi=reference_phi.copy() if initial_phi is None else np.asarray(initial_phi).copy()
    base_state=physical_corner_state(
        z,r,reference_q,reference_phi,zero,zero,zero,background,chi_r,chi_z,chi,
        None,stencil_width,radial_buffer,include_mixed,
    )
    scales=base_state["scales"].copy();trust_radius=float(initial_trust_radius)
    reg=float(regularization);history=[];linearizations=[]

    def solve_state(candidate_coefficients,initial_q,initial_phi):
        a,b,c=combine_shape_modes(candidate_coefficients,modes)
        selector=solve_anisotropic_initial_data(
            z,r,reference_q,reference_phi,a,b,c,background,chi_r,chi_z,
            initial_q=initial_q,initial_phi=initial_phi,stencil_width=stencil_width,
            tolerance=selector_tolerance,iterations=25,
        )
        if not selector["converged"]:return None,selector
        state=physical_corner_state(
            z,r,selector["q"],selector["phi"],a,b,c,background,chi_r,chi_z,
            chi,scales,stencil_width,radial_buffer,include_mixed,
        )
        return state,selector

    if np.max(np.abs(coefficients))==0 and initial_q is None and initial_phi is None:
        current_state=base_state;current_selector={"converged":True,"maximum_residual":0.,"history":[]}
    else:
        current_state,current_selector=solve_state(coefficients,current_q,current_phi)
        if current_state is None:raise RuntimeError("initial continuation state does not solve the selector")
    for iteration in range(int(maximum_iterations)+1):
        maximum_shape=float(max(
            np.max(np.abs(current_state["a"])),np.max(np.abs(current_state["b"])),
            np.max(np.abs(current_state["c"])),
        ))
        history.append({
            "iteration":iteration,"maximum_fixed_scaled_corner_residual":current_state["maximum_fixed_scaled_residual"],
            "corner_residual_l2":current_state["fixed_scaled_residual_l2"],
            "maximum_intrinsic_corner_residual":current_state["maximum_intrinsic_residual"],
            "coefficient_l2":float(np.linalg.norm(coefficients)),
            "maximum_absolute_coefficient":float(np.max(np.abs(coefficients))),
            "maximum_shape_logarithm":maximum_shape,"trust_radius":trust_radius,
            "regularization":reg,"selector_maximum_residual":current_selector["maximum_residual"],
        })
        if current_state["maximum_fixed_scaled_residual"]<corner_tolerance:
            break
        if iteration==int(maximum_iterations):break
        linearization=relinearized_projected_corner_jacobian(
            z,r,current_state["q"],current_state["phi"],current_state["a"],
            current_state["b"],current_state["c"],modes,reference_q,reference_phi,
            background,chi_r,chi_z,chi,scales,stencil_width,radial_buffer,
            finite_difference_step,
            difference_scheme,include_mixed,
        )
        matrix=linearization["matrix"];residual=current_state["vector"]
        linearizations.append({
            "iteration":iteration,"rank":linearization["rank"],
            "maximum_projection_ratio":linearization["maximum_projection_ratio"],
            "median_projection_ratio":linearization["median_projection_ratio"],
            "largest_singular_value":float(linearization["singular_values"][0]),
            "smallest_retained_singular_value":float(linearization["singular_values"][linearization["rank"]-1]),
            "difference_scheme":linearization["difference_scheme"],
        })
        accepted=None
        row_weights=np.sqrt(1+float(maximum_row_weight)*(
            np.abs(residual)/max(np.max(np.abs(residual)),1e-300)
        )**4)
        old_objective=(
            current_state["fixed_scaled_residual_l2"]**2
            +float(maximum_row_weight)*current_state["maximum_fixed_scaled_residual"]**2
        )
        # Try several damping strengths before shrinking the trust region.
        for local_reg in (reg,reg*10,reg/10):
            augmented=np.vstack((row_weights[:,None]*matrix,np.sqrt(local_reg)*np.eye(count)))
            # Regularize the step, not the accumulated physical solution.
            rhs=np.concatenate((-row_weights*residual,np.zeros(count)))
            step_coefficients=np.linalg.lstsq(augmented,rhs,rcond=1e-10)[0]
            step_norm=np.linalg.norm(step_coefficients)
            if step_norm>trust_radius:step_coefficients*=trust_radius/step_norm
            dq=sum(value*response["dq"] for value,response in zip(step_coefficients,linearization["responses"]))
            dphi=sum(value*response["dphi"] for value,response in zip(step_coefficients,linearization["responses"]))
            for fraction in (1.,.5,.25):
                candidate_coefficients=coefficients+fraction*step_coefficients
                a,b,c=combine_shape_modes(candidate_coefficients,modes)
                candidate_shape=max(np.max(np.abs(a)),np.max(np.abs(b)),np.max(np.abs(c)))
                if candidate_shape>shape_bound:continue
                candidate_state,candidate_selector=solve_state(
                    candidate_coefficients,current_state["q"]+fraction*dq,
                    current_state["phi"]+fraction*dphi,
                )
                if candidate_state is None:continue
                new_objective=(
                    candidate_state["fixed_scaled_residual_l2"]**2
                    +float(maximum_row_weight)*candidate_state["maximum_fixed_scaled_residual"]**2
                )
                if new_objective<old_objective and candidate_state["maximum_fixed_scaled_residual"]<max(
                    current_state["maximum_fixed_scaled_residual"]*1.02,corner_tolerance
                ):
                    accepted=(candidate_coefficients,candidate_state,candidate_selector,local_reg,fraction,old_objective,new_objective)
                    break
            if accepted is not None:break
        if accepted is None:
            trust_radius*=.5;reg*=10
            history[-1]["step_accepted"]=False
            if trust_radius<1e-3:break
            continue
        coefficients,current_state,current_selector,used_reg,fraction,old_merit,new_merit=accepted
        trust_radius=min(trust_radius*(1.35 if fraction==1 else .8),.2)
        reg=used_reg
        history[-1].update({
            "step_accepted":True,"accepted_fraction":fraction,
            "accepted_regularization":used_reg,"merit_ratio":new_merit/old_merit,
        })
        if verbose:
            print({
                "iteration":iteration+1,
                "corner_maximum":current_state["maximum_fixed_scaled_residual"],
                "corner_l2":current_state["fixed_scaled_residual_l2"],
                "selector_residual":current_selector["maximum_residual"],
                "trust_radius":trust_radius,"regularization":reg,
            },flush=True)

    return {
        "converged":bool(current_state["maximum_fixed_scaled_residual"]<corner_tolerance),
        "coefficients":coefficients,"labels":shape_basis["labels"],
        "q":current_state["q"],"phi":current_state["phi"],"psi":current_state["psi"],
        "a":current_state["a"],"b":current_state["b"],"c":current_state["c"],
        "history":history,"linearizations":linearizations,
        "final_maximum_fixed_scaled_corner_residual":current_state["maximum_fixed_scaled_residual"],
        "final_corner_residual_l2":current_state["fixed_scaled_residual_l2"],
        "final_maximum_intrinsic_corner_residual":current_state["maximum_intrinsic_residual"],
        "final_selector_maximum_residual":current_selector["maximum_residual"],
        "maximum_shape_logarithm":float(max(
            np.max(np.abs(current_state["a"])),np.max(np.abs(current_state["b"])),np.max(np.abs(current_state["c"]))
        )),
        "maximum_log_conformal_change":float(np.max(np.abs(np.log(current_state["psi"]/(1/(z[:,None]+reference_q)))))),
        "maximum_stabilizer_change":float(np.max(np.abs(current_state["phi"]-reference_phi))),
        "settings":{
            "radial_modes":int(radial_modes),"mode_count":count,
            "maximum_iterations":int(maximum_iterations),"corner_tolerance":float(corner_tolerance),
            "shape_bound":float(shape_bound),"initial_trust_radius":float(initial_trust_radius),
            "initial_regularization":float(regularization),"finite_difference_step":float(finite_difference_step),
            "difference_scheme":difference_scheme,"maximum_row_weight":float(maximum_row_weight),
            "include_mixed":bool(include_mixed),
            "axis_widths":[float(value) for value in axis_widths],
            "basis_radius":None if basis_radius is None else float(basis_radius),
            "initial_coefficient_l2":float(np.linalg.norm(
                np.zeros(count) if initial_coefficients is None else np.asarray(initial_coefficients)
            )),
        },
    }


def _hamiltonian_conformal_operator(z,r,psi,phi,chi_r,chi_z,mass_squared,background,stencil_width):
    """Linearized geometric Hamiltonian operator for ``delta log psi``."""
    z=np.asarray(z);r=np.asarray(r);nz,nr=len(z),len(r);n=nz*nr
    dz1=derivative_matrix(z,1,stencil_width);dzz=derivative_matrix(z,2,stencil_width)
    dr1=derivative_matrix(r,1,stencil_width);drr=derivative_matrix(r,2,stencil_width)
    radial=lil_matrix(drr);radial[0,:]=3*drr.getrow(0)
    for j in range(1,nr):radial[j,:]=drr.getrow(j)+(2/r[j])*dr1.getrow(j)
    iz=eye(nz,format="csr");ir=eye(nr,format="csr")
    dzop=kron(dz1,ir,format="csr");drop=kron(iz,dr1,format="csr")
    flat_lap=kron(dzz,ir,format="csr")+kron(iz,radial.tocsr(),format="csr")
    w=_axisymmetric_derivatives(np.log(psi),z,r,stencil_width)
    lap_gamma=diags((1/psi**2).ravel())@(
        flat_lap+2*diags(w["z"].ravel())@dzop+2*diags(w["r"].ravel())@drop
    )
    geometry=axisymmetric_diagonal_geometry(
        z,r,psi,np.zeros_like(psi),np.zeros_like(psi),np.zeros_like(psi),stencil_width,
    )
    dphi=_axisymmetric_derivatives(phi,z,r,stencil_width)
    matter_gradient=(dphi["z"]**2+dphi["r"]**2+chi_z**2+chi_r**2)/psi**2
    operator=(-6*lap_gamma+diags((-2*geometry["scalar_curvature"]+2*matter_gradient).ravel())).tolil()
    gamma=float(background["wall_stiffness"])
    for i,index,target in ((0,0,float(background["v0"])),(nz-1,-1,float(background["v1"]))):
        potential=.5*gamma*(phi[index]-target)**2
        beta=(
            float(background["beta_a"])+(potential-float(background["wall_potential_a"]))/6
            if index==0 else float(background["beta_b"])-(potential-float(background["wall_potential_b"]))/6
        )
        for j in range(nr-1):
            row=i*nr+j;operator[row,:]=dzop.getrow(row);operator[row,row]+=beta[j]*psi[index,j]
    for i in range(nz):
        row=i*nr+nr-1;operator[row,:]=drop.getrow(row);operator[row,row]+=1/r[-1]
    return {"operator":operator.tocsc(),"factorization":splu(operator.tocsc())}


def prepare_physical_corrector(
    z,r,psi,phi,chi_r,chi_z,background,chi=None,radial_modes=10,
    stencil_width=7,radial_buffer=7,finite_difference_step=2e-4,
):
    """Build constraint-projected physical modes and their corner response."""
    z=np.asarray(z);r=np.asarray(r);psi=np.asarray(psi);phi=np.asarray(phi)
    zero=np.zeros_like(psi);mass=float(background["mass_squared"]);step=float(finite_difference_step)
    shape_basis=tracefree_shape_basis(z,r,radial_modes)
    reference_q=1/psi-z[:,None]
    selector_jacobian=anisotropic_initial_data_jacobian(
        reference_q,phi,z,r,zero,zero,zero,background,chi_r,chi_z,
        reference_q,phi,stencil_width,
    ).tocsc()
    selector_factorization=splu(selector_jacobian)
    baseline_acceleration=anisotropic_metric_acceleration(
        z,r,psi,zero,zero,zero,phi,chi_r,chi_z,mass,chi=chi,
        stencil_width=stencil_width,lapse=psi,
    )
    reference=anisotropic_spatial_israel_second_corner_fields(
        baseline_acceleration,psi,zero,zero,zero,phi,background,zero,radial_buffer,
    )

    def flatten(fields,scales=None):
        values=[];local_scales=[]
        for wall in fields["walls"]:
            for name in ("radial","transverse"):
                component=wall["tangential_components"][name]
                values.append(component["residual"])
                local_scales.append(component["scale"])
        values=np.concatenate(values)
        local_scales=np.concatenate(local_scales) if scales is None else np.asarray(scales)
        return values,local_scales

    target_raw,scales=flatten(reference);target=target_raw/scales
    projected_modes=[];columns=[];constraint_ratios=[]
    for a_mode,b_mode,c_mode in shape_basis["modes"]:
        plus=anisotropic_initial_data_residual(
            reference_q,phi,z,r,step*a_mode,step*b_mode,step*c_mode,background,
            chi_r,chi_z,reference_q,phi,stencil_width,
        )
        minus=anisotropic_initial_data_residual(
            reference_q,phi,z,r,-step*a_mode,-step*b_mode,-step*c_mode,background,
            chi_r,chi_z,reference_q,phi,stencil_width,
        )
        shape_source=(plus-minus)/(2*step)
        response=selector_factorization.solve(-shape_source)
        dq=response[:psi.size].reshape(psi.shape)
        dphi=response[psi.size:].reshape(psi.shape)
        u=-psi*dq
        projected_modes.append({"dq":dq,"dphi":dphi,"u":u,"a":a_mode,"b":b_mode,"c":c_mode})

        corner_vectors=[];constraint_derivatives=[]
        for sign in (1.,-1.):
            local_q=reference_q+sign*step*dq
            local_psi=1/(z[:,None]+local_q)
            local_phi=phi+sign*step*dphi
            local_a=sign*step*a_mode;local_b=sign*step*b_mode;local_c=sign*step*c_mode
            acceleration=anisotropic_metric_acceleration(
                z,r,local_psi,local_a,local_b,local_c,local_phi,chi_r,chi_z,mass,
                chi=chi,stencil_width=stencil_width,lapse=local_psi,
            )
            fields=anisotropic_spatial_israel_second_corner_fields(
                acceleration,local_psi,local_a,local_b,local_c,local_phi,background,
                zero,radial_buffer,
            )
            corner_vectors.append(flatten(fields,scales)[0])
            constraint_derivatives.append(anisotropic_initial_data_residual(
                local_q,local_phi,z,r,local_a,local_b,local_c,background,
                chi_r,chi_z,reference_q,phi,stencil_width,
            ))
        columns.append((corner_vectors[0]-corner_vectors[1])/(2*step)/scales)
        total_derivative=(constraint_derivatives[0]-constraint_derivatives[1])/(2*step)
        constraint_ratios.append(float(np.linalg.norm(total_derivative)/max(np.linalg.norm(shape_source),1e-300)))
    matrix=np.column_stack(columns)
    return {
        "matrix":matrix,"target":target,"target_raw":target_raw,"scales":scales,
        "modes":projected_modes,"labels":shape_basis["labels"],
        "reference_fields":reference,
        "maximum_constraint_projection_ratio":float(max(constraint_ratios)),
        "median_constraint_projection_ratio":float(np.median(constraint_ratios)),
        "settings":{"radial_modes":int(radial_modes),"stencil_width":int(stencil_width),"radial_buffer":int(radial_buffer),"finite_difference_step":step},
    }


def solve_linear_corner_correction(prepared,regularization=1e-6):
    """Solve the fixed-scale linear least-squares corner cancellation."""
    matrix=prepared["matrix"];target=prepared["target"]
    regularization=float(regularization)
    augmented=np.vstack((matrix,np.sqrt(regularization)*np.eye(matrix.shape[1])))
    rhs=np.concatenate((-target,np.zeros(matrix.shape[1])))
    coefficients=np.linalg.lstsq(augmented,rhs,rcond=1e-10)[0]
    residual=target+matrix@coefficients
    singular=np.linalg.svd(matrix,compute_uv=False)
    threshold=singular[0]*max(matrix.shape)*np.finfo(float).eps
    rank=int(np.count_nonzero(singular>threshold))
    dq=sum(coefficient*mode["dq"] for coefficient,mode in zip(coefficients,prepared["modes"]))
    dphi=sum(coefficient*mode["dphi"] for coefficient,mode in zip(coefficients,prepared["modes"]))
    u=sum(coefficient*mode["u"] for coefficient,mode in zip(coefficients,prepared["modes"]))
    a=sum(coefficient*mode["a"] for coefficient,mode in zip(coefficients,prepared["modes"]))
    b=sum(coefficient*mode["b"] for coefficient,mode in zip(coefficients,prepared["modes"]))
    c=sum(coefficient*mode["c"] for coefficient,mode in zip(coefficients,prepared["modes"]))
    return {
        "coefficients":coefficients,"dq":dq,"dphi":dphi,"u":u,"a":a,"b":b,"c":c,
        "initial_l2":float(np.linalg.norm(target)),"final_linear_l2":float(np.linalg.norm(residual)),
        "initial_maximum":float(np.max(np.abs(target))),"final_linear_maximum":float(np.max(np.abs(residual))),
        "coefficient_l2":float(np.linalg.norm(coefficients)),"maximum_absolute_coefficient":float(np.max(np.abs(coefficients))),
        "matrix_rank":rank,"row_count":matrix.shape[0],"column_count":matrix.shape[1],
        "smallest_retained_singular_value":float(singular[rank-1]),"largest_singular_value":float(singular[0]),
        "regularization":regularization,
    }


def evaluate_physical_corner_candidate(
    prepared,solution,z,r,psi,phi,chi_r,chi_z,background,chi=None,
    correction_scale=1.,stencil_width=7,radial_buffer=7,
):
    """Evaluate a linear corrector as an actual finite nonlinear geometry."""
    scale=float(correction_scale);psi=np.asarray(psi);phi=np.asarray(phi)
    zero=np.zeros_like(psi);mass=float(background["mass_squared"])
    reference_q=1/psi-np.asarray(z)[:,None]
    local_q=reference_q+scale*solution["dq"]
    local_psi=1/(np.asarray(z)[:,None]+local_q)
    local_phi=phi+scale*solution["dphi"]
    local_a=scale*solution["a"];local_b=scale*solution["b"];local_c=scale*solution["c"]
    acceleration=anisotropic_metric_acceleration(
        z,r,local_psi,local_a,local_b,local_c,local_phi,chi_r,chi_z,mass,
        chi=chi,stencil_width=stencil_width,lapse=local_psi,
    )
    fields=anisotropic_spatial_israel_second_corner_fields(
        acceleration,local_psi,local_a,local_b,local_c,local_phi,background,
        zero,radial_buffer,
    )
    values=[]
    for wall in fields["walls"]:
        for name in ("radial","transverse"):
            values.append(wall["tangential_components"][name]["residual"])
    values=np.concatenate(values);fixed_scaled=values/prepared["scales"]

    baseline_h=anisotropic_hamiltonian_residual(
        z,r,psi,zero,zero,zero,phi,chi_r,chi_z,mass,chi=chi,
        stencil_width=stencil_width,
    )
    candidate_h=anisotropic_hamiltonian_residual(
        z,r,local_psi,local_a,local_b,local_c,local_phi,chi_r,chi_z,mass,chi=chi,
        stencil_width=stencil_width,
    )
    h_change=candidate_h-baseline_h
    retained_h=h_change[1:-1,:-1]
    baseline_junction=anisotropic_spatial_junction_fields(
        z,r,psi,zero,zero,zero,phi,background,stencil_width,
    )
    candidate_junction=anisotropic_spatial_junction_fields(
        z,r,local_psi,local_a,local_b,local_c,local_phi,background,stencil_width,
    )
    junction_changes=[]
    for base_wall,new_wall in zip(baseline_junction["walls"],candidate_junction["walls"]):
        for name in ("radial","transverse"):
            junction_changes.append(new_wall[name][:-1]-base_wall[name][:-1])
    junction_changes=np.concatenate(junction_changes)
    return {
        "correction_scale":scale,
        "maximum_fixed_scaled_corner_residual":float(np.max(np.abs(fixed_scaled))),
        "corner_residual_l2":float(np.linalg.norm(fixed_scaled)),
        "maximum_absolute_hamiltonian_change":float(np.max(np.abs(retained_h))),
        "hamiltonian_change_l2":float(np.sqrt(np.mean(retained_h**2))),
        "maximum_absolute_junction_change":float(np.max(np.abs(junction_changes))),
        "maximum_absolute_log_conformal_change":float(np.max(np.abs(np.log(local_psi/psi)))),
        "maximum_absolute_stabilizer_change":float(np.max(np.abs(scale*solution["dphi"]))),
        "maximum_absolute_shape_logarithm":float(max(
            np.max(np.abs(local_a)),np.max(np.abs(local_b)),np.max(np.abs(local_c))
        )),
        "minimum_psi":float(np.min(local_psi)),
        "maximum_unit_determinant_defect":float(np.max(np.abs(local_a+local_b+2*local_c))),
    }
