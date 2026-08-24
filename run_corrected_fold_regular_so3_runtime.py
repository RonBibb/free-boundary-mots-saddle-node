#!/usr/bin/env python3
"""Run the regular nine-field lower-order GH system on the corrected G6 fold."""

import json,sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_geometry import anisotropic_metric_acceleration,anisotropic_scalar_acceleration
from bhps.anisotropic_initial_data import solve_anisotropic_initial_data
from bhps.axisymmetric_reduced_wave_evolution import AxisymmetricVariableReducedWaveIBVP,axisymmetric_principal_coefficients
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice
from bhps.generalized_harmonic_jets import spatial_metric_acceleration_trace
from bhps.lapse_acceleration_corner import construct_localized_target_lapse_acceleration_completion
from bhps.physical_corner_corrector import combine_shape_modes,tracefree_shape_basis
from bhps.regular_so3_gh_reduction import FIELD_ORDER,RegularSO3BackgroundJetField,regular_so3_gh_coefficient_matrices,regular_so3_robin_matrix
from bhps.scalar_pulse import scalar_pulse


R_MAX=4.;OMEGA=1.2
SUPPORT_Z_COUNT=5
SUPPORT_R=np.array((.125,.25,.5,1.,2.,3.,4.))
GRID_SIZES=((9,13),(17,25),(33,49),(49,73))


def spline(field,z,r):return RectBivariateSpline(z,r,np.asarray(field),kx=3,ky=3,s=0)


def build_geometry(resolution="G6"):
    resolution=str(resolution).upper()
    if resolution=="G6":
        fold_path="results/corrected_anisotropic_arclength_G6.json";name="G6R8";nz,nr=65,97
    elif resolution=="G5":
        fold_path="results/corrected_anisotropic_arclength.json";name="G5R8";nz,nr=49,73
    else:raise ValueError("resolution must be G5 or G6")
    fold=json.loads(Path(fold_path).read_text())
    amplitude=float(fold["summary"]["fine_fold_amplitude"])
    archive=np.load("results/corrected_family_knot_A8_state.npz");coefficients=archive["coefficients"]
    reference=solve_finite_wall_high_order_slice(
        amplitude,nz=nz,nr=nr,r_max=8.,wall_stiffness=20.,epsilon=.1,
        backreaction=.01,tolerance=1e-10,iterations=240,
    )
    chi,chi_r,chi_z=scalar_pulse(reference["z"],reference["r"],amplitude)
    modes=tracefree_shape_basis(
        reference["z"],reference["r"],6,(.5,1.),8.,((7.5,1.5),(7.5,3.0)),
    )["modes"]
    a,b,c=combine_shape_modes(coefficients,modes)
    selected=solve_anisotropic_initial_data(
        reference["z"],reference["r"],reference["q"],reference["phi"],a,b,c,
        reference["background"],chi_r,chi_z,
        initial_q=archive[f"q_{name}"],initial_phi=archive[f"phi_{name}"],
        stencil_width=7,tolerance=1e-9,iterations=30,
    )
    z=reference["z"];r=reference["r"];phi=selected["phi"]
    psi=1/(z[:,None]+selected["q"]);mass=float(reference["background"]["mass_squared"])
    acceleration=anisotropic_metric_acceleration(
        z,r,psi,a,b,c,phi,chi_r,chi_z,mass,chi=chi,stencil_width=7,lapse=psi,
    )
    phi_tt=anisotropic_scalar_acceleration(z,r,psi,a,b,c,phi,mass,lapse=psi,stencil_width=7)
    chi_tt=anisotropic_scalar_acceleration(z,r,psi,a,b,c,chi,0.,lapse=psi,stencil_width=7)
    trace=spatial_metric_acceleration_trace(acceleration,psi,a,b,c)
    completion=construct_localized_target_lapse_acceleration_completion(
        z,acceleration,psi,psi,a,phi,reference["background"],phi_tt,.5*trace,.15,
    )
    jet_field=RegularSO3BackgroundJetField(
        z,r,psi,psi,a,b,c,phi,chi,acceleration,completion["lapse_acceleration"],
        phi_tt,chi_tt,7,
    )
    return {
        "name":name,"source_grid":[nz,nr],"fold_amplitude":amplitude,
        "selector_maximum":float(selected["maximum_residual"]),
        "z":z,"r":r,"psi":psi,"a":a,"b":b,"c":c,"phi":phi,
        "background":reference["background"],"mass_squared":mass,
        "principal":axisymmetric_principal_coefficients(psi,a,b,c),
        "jet_field":jet_field,
    }


def even_axis_extension(values,r_support):
    values=np.asarray(values);extended=np.empty((values.shape[0],len(r_support)+1,*values.shape[2:]))
    extended[:,1:]=values;s=r_support**2;flat=values.reshape(values.shape[0],len(r_support),-1)
    axis=np.empty((values.shape[0],flat.shape[-1]));axis_alt=np.empty_like(axis)
    for i in range(values.shape[0]):
        for j in range(flat.shape[-1]):
            axis[i,j]=np.polynomial.polynomial.polyfit(s[:4],flat[i,:4,j],2)[0]
            axis_alt[i,j]=np.polynomial.polynomial.polyfit(s[:3],flat[i,:3,j],2)[0]
    extended[:,0]=axis.reshape((values.shape[0],*values.shape[2:]))
    scale=max(float(np.linalg.norm(axis)),1e-300)
    return extended,float(np.linalg.norm(axis-axis_alt)/scale)


def sample_coefficients(geometry,constraint_damping=0.,support_r=SUPPORT_R):
    support_r=np.asarray(support_r,dtype=float)
    z_support=np.geomspace(geometry["z"][0],geometry["z"][-1],SUPPORT_Z_COUNT)
    reaction=np.empty((len(z_support),len(support_r),9,9))
    first=np.empty((3,len(z_support),len(support_r),9,9))
    principal_defect=0.
    for i,z_value in enumerate(z_support):
        print(f"coefficient support z {i+1}/{len(z_support)}",flush=True)
        for j,r_value in enumerate(support_r):
            coefficients=regular_so3_gh_coefficient_matrices(
                geometry["jet_field"].at(z_value,r_value),r_value,
                mass_squared=geometry["mass_squared"],potential_offset=-6.,
                constraint_damping=constraint_damping,
            )
            reaction[i,j]=coefficients["evolution_reaction_matrix"]
            first[:,i,j]=coefficients["evolution_first_matrices"]
            principal_defect=max(principal_defect,coefficients["principal_identity_maximum_defect"])
    reaction_ext,reaction_axis_difference=even_axis_extension(reaction,support_r)
    time_ext,time_axis_difference=even_axis_extension(first[0],support_r)
    z_ext,z_axis_difference=even_axis_extension(first[1],support_r)
    scaled_radial=first[2]*support_r[None,:,None,None]
    radial_ext,radial_axis_difference=even_axis_extension(scaled_radial,support_r)
    return {
        "z":z_support,"r":np.r_[0.,support_r],"reaction":reaction_ext,
        "first":np.stack((time_ext,z_ext,radial_ext)),
        "radial_first_is_scaled":True,"principal_defect":principal_defect,
        "constraint_damping_rate":float(constraint_damping),
        "axis_extrapolation_relative_differences":{
            "reaction":reaction_axis_difference,"time_first":time_axis_difference,
            "z_first":z_axis_difference,"scaled_radial_first":radial_axis_difference,
        },
    }


def interpolate_tensor(values,z_source,r_source,z_target,r_target):
    zz,rr=np.meshgrid(z_target,r_target,indexing="ij");result=np.empty((len(z_target),len(r_target),*values.shape[2:]))
    for index in np.ndindex(values.shape[2:]):
        result[(slice(None),slice(None))+index]=RectBivariateSpline(
            z_source,r_source,values[(slice(None),slice(None))+index],kx=3,ky=3,s=0,
        ).ev(zz.ravel(),rr.ravel()).reshape(len(z_target),len(r_target))
    return result


def wall_matrix(geometry,r_values,wall_index):
    background=geometry["background"];gamma=float(background["wall_stiffness"])
    zi=0 if wall_index==0 else -1
    phi=spline(geometry["phi"],geometry["z"],geometry["r"])(geometry["z"][zi],r_values,grid=True).reshape(-1)
    target=float(background["v0"] if wall_index==0 else background["v1"])
    delta=phi-target;potential=.5*gamma*delta**2
    if wall_index==0:c=float(background["beta_a"])+(potential-float(background["wall_potential_a"]))/6
    else:c=-(float(background["beta_b"])-(potential-float(background["wall_potential_b"]))/6)
    return np.asarray([
        regular_so3_robin_matrix(value,uprime/6,uprime/2,gamma)["matrix"]
        for value,uprime in zip(c,gamma*delta)
    ])


def sampled_system(geometry,coefficients,nz,nr,r_max=R_MAX,outer_dirichlet=True):
    z=np.linspace(geometry["z"][0],geometry["z"][-1],nz);r=np.linspace(0,float(r_max),nr)
    zz,rr=np.meshgrid(z,r,indexing="ij");principal={}
    for key in ("mass_weight","z_gradient_weight","r_gradient_weight","z_boundary_weight","A"):
        principal[key]=spline(geometry["principal"][key],geometry["z"],geometry["r"]).ev(
            zz.ravel(),rr.ravel(),
        ).reshape(nz,nr)
    principal["z_speed"]=np.sqrt(principal["z_gradient_weight"]/principal["mass_weight"])
    principal["r_speed"]=np.sqrt(principal["r_gradient_weight"]/principal["mass_weight"])
    reaction=interpolate_tensor(coefficients["reaction"],coefficients["z"],coefficients["r"],z,r)
    first=np.empty((3,nz,nr,9,9))
    for direction in range(3):
        first[direction]=interpolate_tensor(coefficients["first"][direction],coefficients["z"],coefficients["r"],z,r)
    left=wall_matrix(geometry,r,0);right=wall_matrix(geometry,r,1)
    system=AxisymmetricVariableReducedWaveIBVP(
        z,r,principal["mass_weight"],principal["z_gradient_weight"],principal["r_gradient_weight"],
        left,right,principal["z_boundary_weight"][0],principal["z_boundary_weight"][-1],
        dirichlet_fields=2,coupled_reaction_matrices=reaction,
        evolution_first_matrices=first,radial_first_is_scaled=True,
        outer_dirichlet=outer_dirichlet,
    )
    return system,principal,reaction,first,left,right


def manufactured_fields(z,r):
    zz,rr=np.meshgrid(z,r,indexing="ij");x=(zz-z[0])/(z[-1]-z[0]);length=z[-1]-z[0]
    kr=np.pi/(2*R_MAX);g=np.cos(kr*rr);gr=-kr*np.sin(kr*rr);grr=-kr**2*g
    values=np.zeros((len(z),len(r),9));uz=np.zeros_like(values);uzz=np.zeros_like(values)
    rng=np.random.default_rng(926);polynomial=.035*rng.normal(size=(4,7))
    for field in range(2):
        wave=(field+1)*np.pi;f=np.sin(wave*x)*(1+.05*(field+1)*x)
        fz=(wave*np.cos(wave*x)*(1+.05*(field+1)*x)+.05*(field+1)*np.sin(wave*x))/length
        fzz=(-wave**2*np.sin(wave*x)*(1+.05*(field+1)*x)+.1*(field+1)*wave*np.cos(wave*x))/length**2
        values[:,:,field]=f*g;uz[:,:,field]=fz*g;uzz[:,:,field]=fzz*g
    powers=np.stack((np.ones_like(x),x,x**2,x**3),axis=-1)
    f=np.einsum("ijm,mk->ijk",powers,polynomial)
    fz=np.einsum("ijm,mk->ijk",np.stack((0*x,1+0*x,2*x,3*x**2),axis=-1),polynomial)/length
    fzz=np.einsum("ijm,mk->ijk",np.stack((0*x,0*x,2+0*x,6*x),axis=-1),polynomial)/length**2
    values[:,:,2:]=f*g[:,:,None];uz[:,:,2:]=fz*g[:,:,None];uzz[:,:,2:]=fzz*g[:,:,None]
    ur=values/g[:,:,None]*gr[:,:,None];urr=values/g[:,:,None]*grr[:,:,None]
    # The value vanishes at the outer Dirichlet boundary, but its radial
    # derivative does not.  Preserve ur there because the volume source is
    # integrated over the adjacent finite element.
    values[:,-1]=0.;uz[:,-1]=0.;uzz[:,-1]=0.
    return values,uz,uzz,ur,urr


def manufactured_problem(geometry,coefficients,nz,nr):
    system,principal,reaction,first,left,right=sampled_system(geometry,coefficients,nz,nr)
    values,uz,uzz,ur,urr=manufactured_fields(system.z,system.r)
    zz,rr=np.meshgrid(system.z,system.r,indexing="ij")
    pz=spline(geometry["principal"]["z_gradient_weight"],geometry["z"],geometry["r"])
    pr=spline(geometry["principal"]["r_gradient_weight"],geometry["z"],geometry["r"])
    pz_z=pz.ev(zz.ravel(),rr.ravel(),dx=1,dy=0).reshape(nz,nr)
    pr_r=pr.ev(zz.ravel(),rr.ravel(),dx=0,dy=1).reshape(nz,nr)
    divergence=pz_z[:,:,None]*uz+principal["z_gradient_weight"][:,:,None]*uzz
    divergence+=pr_r[:,:,None]*ur+principal["r_gradient_weight"][:,:,None]*urr
    divergence[:,1:]+=2*principal["r_gradient_weight"][:,1:,None]*ur[:,1:]/rr[:,1:,None]
    divergence[:,0]=pz_z[:,0,None]*uz[:,0]+principal["z_gradient_weight"][:,0,None]*uzz[:,0]+3*principal["r_gradient_weight"][:,0,None]*urr[:,0]
    radial_lower=np.empty_like(values)
    radial_lower[:,1:]=np.einsum("ijab,ijb->ija",first[2,:,1:]/rr[:,1:,None,None],ur[:,1:])
    radial_lower[:,0]=np.einsum("iab,ib->ia",first[2,:,0],urr[:,0])
    spatial_lower=np.einsum("ijab,ijb->ija",first[1],uz)+radial_lower
    reaction_term=np.einsum("ijab,ijb->ija",reaction,values)
    def source(time,zq,rq):
        cosine=np.cos(OMEGA*time);sine=np.sin(OMEGA*time)
        velocity=-OMEGA*sine*values
        time_lower=np.einsum("ijab,ijb->ija",first[0],velocity)
        return -OMEGA**2*cosine*values-cosine*divergence/principal["mass_weight"][:,:,None]-cosine*spatial_lower-time_lower+cosine*reaction_term
    left_data=np.array([-uz[0,j,2:]/principal["A"][0,j]-left[j]@values[0,j,2:] for j in range(nr)])
    right_data=np.array([uz[-1,j,2:]/principal["A"][-1,j]-right[j]@values[-1,j,2:] for j in range(nr)])
    lower=lambda time,rq:np.cos(OMEGA*time)*left_data
    upper=lambda time,rq:np.cos(OMEGA*time)*right_data
    return system,principal,values,source,lower,upper


def grid_audit(geometry,coefficients,final_time=.012):
    records=[]
    for nz,nr in GRID_SIZES:
        print(f"runtime grid {nz} x {nr}",flush=True)
        system,principal,values,source,lower,upper=manufactured_problem(geometry,coefficients,nz,nr)
        speed=max(np.max(principal["z_speed"]),np.max(principal["r_speed"]))
        semidiscrete=system.acceleration(
            0.,values,source,lower,upper,velocity=np.zeros_like(values),
        )
        exact_acceleration=-OMEGA**2*values
        acceleration_error=system.l2_norm(semidiscrete-exact_acceleration)
        acceleration_scale=system.l2_norm(exact_acceleration)
        per_field_acceleration=[]
        for field in range(9):
            error_field=np.zeros_like(values);scale_field=np.zeros_like(values)
            error_field[:,:,field]=semidiscrete[:,:,field]-exact_acceleration[:,:,field]
            scale_field[:,:,field]=exact_acceleration[:,:,field]
            per_field_acceleration.append(float(
                system.l2_norm(error_field)/max(system.l2_norm(scale_field),1e-300)
            ))
        result=system.integrate(values,np.zeros_like(values),final_time,.025/speed,source,lower,upper)
        exact=np.cos(OMEGA*final_time)*values;exact_v=-OMEGA*np.sin(OMEGA*final_time)*values
        error=np.hypot(system.l2_norm(result["position"]-exact),system.l2_norm(result["velocity"]-exact_v))
        scale=np.hypot(system.l2_norm(exact),system.l2_norm(exact_v))
        records.append({
            "grid_size":[nz,nr],
            "maximum_grid_spacing":float(max(np.max(np.diff(system.z)),np.max(np.diff(system.r)))),
            "semidiscrete_acceleration_relative_error":float(acceleration_error/acceleration_scale),
            "semidiscrete_acceleration_relative_error_by_field":dict(zip(FIELD_ORDER,per_field_acceleration)),
            "combined_relative_error":float(error/scale),"time_step":result["time_step"],
        })
    errors=np.array([item["combined_relative_error"] for item in records])
    acceleration_errors=np.array([item["semidiscrete_acceleration_relative_error"] for item in records])
    spacing=np.array([item["maximum_grid_spacing"] for item in records])
    return {
        "records":records,
        "convergence_rates":[float(value) for value in np.log(errors[:-1]/errors[1:])/np.log(spacing[:-1]/spacing[1:])],
        "semidiscrete_acceleration_convergence_rates":[
            float(value) for value in np.log(acceleration_errors[:-1]/acceleration_errors[1:])/np.log(spacing[:-1]/spacing[1:])
        ],
    }


def timestep_audit(geometry,coefficients,final_time=.1,nz=13,nr=17):
    system,_,values,source,lower,upper=manufactured_problem(geometry,coefficients,nz,nr)
    spacing=min(np.min(np.diff(system.z)),np.min(np.diff(system.r)));solutions=[];records=[]
    for steps in (8,16,32,64):
        courant=final_time/(steps*spacing)*1.000000001
        result=system.integrate(values,np.zeros_like(values),final_time,courant,source,lower,upper)
        solutions.append(result["position"])
        records.append({"steps":result["steps"],"time_step":result["time_step"]})
    differences=np.array([system.l2_norm(solutions[i]-solutions[i+1]) for i in range(3)])
    return {
        "records":records,"successive_solution_differences":[float(value) for value in differences],
        "convergence_rates":[float(value) for value in np.log(differences[:-1]/differences[1:])/np.log(2.)],
    }


def main():
    geometry=build_geometry()
    print("sampling covariant lower-order coefficients",flush=True)
    coefficients=sample_coefficients(geometry)
    grid=grid_audit(geometry,coefficients)
    time=timestep_audit(geometry,coefficients)
    acceptance={
        "selector_below_1e_8":geometry["selector_maximum"]<1e-8,
        "support_principal_defect_below_1e_10":coefficients["principal_defect"]<1e-10,
        "axis_extrapolation_variants_all_below_5_percent":max(coefficients["axis_extrapolation_relative_differences"].values())<.05,
        "all_grid_errors_finite":all(np.isfinite(item["combined_relative_error"]) for item in grid["records"]),
        "fine_grid_rate_above_1p6":grid["convergence_rates"][-1]>1.6,
        "both_timestep_rates_above_3p5":min(time["convergence_rates"])>3.5,
    }
    payload={
        "status":"pass" if all(acceptance.values()) else "review",
        "scope":"regular nine-field coupled lower-order Q1/RK4 manufactured runtime on the corrected G6 fold",
        "field_order":list(FIELD_ORDER),"source_grid":geometry["source_grid"],
        "fold_amplitude":geometry["fold_amplitude"],"selector_maximum":geometry["selector_maximum"],
        "coefficient_support":{"z":[float(value) for value in coefficients["z"]],"r":[float(value) for value in coefficients["r"]]},
        "axis_extrapolation_relative_differences":coefficients["axis_extrapolation_relative_differences"],
        "support_principal_identity_maximum_defect":coefficients["principal_defect"],
        "grid_convergence":grid,"timestep_convergence":time,"acceptance":acceptance,
        "limitations":[
            "manufactured linear evolution rather than physical perturbation spectrum or nonlinear collapse",
            "G6 corrected fold; G5/G6 coefficient transfer is covered by the separate sampled operator audit",
            "five-by-seven covariant coefficient support with even axis extrapolation",
            "frozen generalized-harmonic source, no constraint damping or evolved source driver",
            "homogeneous outer radial perturbation boundary at r=4",
        ],
    }
    Path("results/corrected_fold_regular_so3_runtime.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":payload["status"],"acceptance":acceptance,"axis":coefficients["axis_extrapolation_relative_differences"],"grid":grid,"time":time},indent=2))


if __name__=="__main__":main()
