#!/usr/bin/env python3
"""Exact massive fixed-background scalar evolution on corrected folds."""

import json,sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.sparse.linalg import eigsh

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_initial_data import solve_anisotropic_initial_data
from bhps.axisymmetric_reduced_wave_evolution import AxisymmetricVariableReducedWaveIBVP,axisymmetric_principal_coefficients
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice
from bhps.physical_corner_corrector import combine_shape_modes,tracefree_shape_basis
from bhps.scalar_pulse import scalar_pulse


R_MAX=4.;GRIDS=((13,17),(25,33),(49,65),(97,129));OMEGA=1.25


def spline(field,z,r):return RectBivariateSpline(z,r,np.asarray(field),kx=3,ky=3,s=0)


def corrected_geometry(name,nz,nr,amplitude,archive,coefficients):
    reference=solve_finite_wall_high_order_slice(
        amplitude,nz=nz,nr=nr,r_max=8.,wall_stiffness=20.,epsilon=.1,
        backreaction=.01,tolerance=1e-10,iterations=240,
    )
    _,chi_r,chi_z=scalar_pulse(reference["z"],reference["r"],amplitude)
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
    psi=1/(reference["z"][:,None]+selected["q"])
    return {
        "z":reference["z"],"r":reference["r"],"psi":psi,
        "principal":axisymmetric_principal_coefficients(psi,a,b,c),
        "mass_squared":float(reference["background"]["mass_squared"]),
        "wall_stiffness":float(reference["background"]["wall_stiffness"]),
        "selector_maximum":selected["maximum_residual"],
    }


def sampled_system(geometry,nz,nr):
    z=np.linspace(geometry["z"][0],geometry["z"][-1],nz);r=np.linspace(0,R_MAX,nr)
    zz,rr=np.meshgrid(z,r,indexing="ij");principal={}
    for key in ("mass_weight","z_gradient_weight","r_gradient_weight","z_boundary_weight","A","lapse"):
        principal[key]=spline(geometry["principal"][key],geometry["z"],geometry["r"]).ev(
            zz.ravel(),rr.ravel(),
        ).reshape(nz,nr)
    principal["z_coordinate_speed"]=np.sqrt(principal["z_gradient_weight"]/principal["mass_weight"])
    principal["r_coordinate_speed"]=np.sqrt(principal["r_gradient_weight"]/principal["mass_weight"])
    gamma=geometry["wall_stiffness"];robin=np.full((nr,1,1),-gamma/2)
    reaction=(
        principal["mass_weight"]*principal["lapse"]**2*geometry["mass_squared"]
    )[:,:,None]
    system=AxisymmetricVariableReducedWaveIBVP(
        z,r,principal["mass_weight"],principal["z_gradient_weight"],principal["r_gradient_weight"],
        robin,robin,principal["z_boundary_weight"][0],principal["z_boundary_weight"][-1],
        dirichlet_fields=0,reaction_weights=reaction,
    )
    return system,principal


def spectrum_audit(geometry,count=3):
    records=[];spectra=[]
    for nz,nr in GRIDS:
        system,_=sampled_system(geometry,nz,nr);operator=system.scalar_spatial_operator(0)
        values=eigsh(
            operator["operator"],k=count,M=operator["mass"],sigma=0.,which="LM",
            return_eigenvectors=False,
        )
        values=np.sort(values);spectra.append(values)
        records.append({"grid_size":[nz,nr],"omega_squared":[float(value) for value in values]})
    spectra=np.asarray(spectra);rates=[]
    for mode in range(count):
        differences=np.abs(np.diff(spectra[:,mode]))
        rates.append([float(value) for value in np.log(differences[:-1]/differences[1:])/np.log(2.)])
    return {
        "records":records,"mode_convergence_rates":rates,
        "all_sampled_positive":bool(np.min(spectra)>0),
        "finest_minimum_omega_squared":float(spectra[-1,0]),
    }


def manufactured_audit(geometry,grids=GRIDS[:3],final_time=.018,courant=.035):
    records=[];kz=1.1;kr=np.pi/(2*R_MAX)
    pz_source=spline(geometry["principal"]["z_gradient_weight"],geometry["z"],geometry["r"])
    pr_source=spline(geometry["principal"]["r_gradient_weight"],geometry["z"],geometry["r"])
    for nz,nr in grids:
        system,principal=sampled_system(geometry,nz,nr);zz,rr=np.meshgrid(system.z,system.r,indexing="ij")
        x=(zz-system.z[0])/(system.z[-1]-system.z[0]);length=system.z[-1]-system.z[0]
        f=1+.2*x+.1*x**2;fz=(.2+.2*x)/length;fzz=.2/length**2
        g=np.cos(kr*rr);gr=-kr*np.sin(kr*rr);grr=-kr**2*g
        spatial=(f*g)[:,:,None];uz=(fz*g)[:,:,None];uzz=(fzz*g)[:,:,None]
        ur=(f*gr)[:,:,None];urr=(f*grr)[:,:,None]
        pz_z=pz_source.ev(zz.ravel(),rr.ravel(),dx=1,dy=0).reshape(nz,nr)
        pr_r=pr_source.ev(zz.ravel(),rr.ravel(),dx=0,dy=1).reshape(nz,nr)
        divergence=(pz_z[:,:,None]*uz+principal["z_gradient_weight"][:,:,None]*uzz)
        divergence+=pr_r[:,:,None]*ur+principal["r_gradient_weight"][:,:,None]*urr
        divergence[:,1:]+=2*principal["r_gradient_weight"][:,1:,None]*ur[:,1:]/rr[:,1:,None]
        divergence[:,0]=(
            pz_z[:,0,None]*uz[:,0]+principal["z_gradient_weight"][:,0,None]*uzz[:,0]
            +3*principal["r_gradient_weight"][:,0,None]*urr[:,0]
        )
        strong_source=(
            -OMEGA**2*spatial-divergence/principal["mass_weight"][:,:,None]
            +geometry["mass_squared"]*principal["lapse"][:,:,None]**2*spatial
        )
        gamma=geometry["wall_stiffness"]
        lower_data=-uz[0]/principal["A"][0,:,None]+gamma*spatial[0]/2
        upper_data=uz[-1]/principal["A"][-1,:,None]+gamma*spatial[-1]/2
        source=lambda time,zq,rq:np.cos(OMEGA*time)*strong_source
        lower=lambda time,rq:np.cos(OMEGA*time)*lower_data
        upper=lambda time,rq:np.cos(OMEGA*time)*upper_data
        speed=max(np.max(principal["z_coordinate_speed"]),np.max(principal["r_coordinate_speed"]))
        result=system.integrate(spatial,np.zeros_like(spatial),final_time,courant/speed,source,lower,upper)
        exact_q=np.cos(OMEGA*final_time)*spatial;exact_v=-OMEGA*np.sin(OMEGA*final_time)*spatial
        error=float(np.hypot(system.l2_norm(result["position"]-exact_q),system.l2_norm(result["velocity"]-exact_v)))
        scale=float(np.hypot(system.l2_norm(exact_q),system.l2_norm(exact_v)))
        records.append({"grid_size":[nz,nr],"combined_relative_error":error/scale,"time_step":result["time_step"]})
    errors=np.array([item["combined_relative_error"] for item in records])
    return {"records":records,"convergence_rates":[float(value) for value in np.log(errors[:-1]/errors[1:])/np.log(2.)]}


def eigenmode_runtime_audit(geometry,nz=49,nr=65,final_time=1.):
    system,_=sampled_system(geometry,nz,nr);operator=system.scalar_spatial_operator(0)
    values,vectors=eigsh(operator["operator"],k=1,M=operator["mass"],sigma=0.,which="LM")
    frequency=float(np.sqrt(values[0]));mode=vectors[:,0]
    initial=np.zeros((nz,nr,1));initial.reshape(system.nodes,1)[operator["free_indices"],0]=mode
    spatial=operator["operator"];mass=operator["mass"]
    initial_energy=.5*float(mode@spatial@mode)
    records=[]
    for courant in (.4,.2,.1):
        result=system.integrate(initial,np.zeros_like(initial),final_time,courant)
        exact=np.cos(frequency*final_time)*initial
        error=system.l2_norm(result["position"]-exact)/system.l2_norm(exact)
        q=result["position"].reshape(system.nodes,1)[operator["free_indices"],0]
        v=result["velocity"].reshape(system.nodes,1)[operator["free_indices"],0]
        final_energy=.5*float(v@mass@v+q@spatial@q)
        records.append({
            "courant":courant,"time_step":result["time_step"],"steps":result["steps"],
            "relative_solution_error":error,
            "relative_endpoint_energy_drift":abs(final_energy-initial_energy)/abs(initial_energy),
        })
    errors=np.array([item["relative_solution_error"] for item in records])
    return {
        "omega_squared":float(values[0]),"records":records,
        "time_convergence_rates":[float(value) for value in np.log(errors[:-1]/errors[1:])/np.log(2.)],
    }


g5=json.loads(Path("results/corrected_anisotropic_arclength.json").read_text())
g6=json.loads(Path("results/corrected_anisotropic_arclength_G6.json").read_text())
archive=np.load("results/corrected_family_knot_A8_state.npz");coefficients=archive["coefficients"]
specifications=(
    ("G5R8",49,73,float(g5["summary"]["fine_fold_amplitude"])),
    ("G6R8",65,97,float(g6["summary"]["fine_fold_amplitude"])),
)
cases=[]
for name,nz,nr,amplitude in specifications:
    print(f"building {name} corrected geometry",flush=True)
    geometry=corrected_geometry(name,nz,nr,amplitude,archive,coefficients)
    print(f"running {name} scalar spectrum",flush=True)
    spectrum=spectrum_audit(geometry)
    print(f"running {name} manufactured evolution",flush=True)
    manufactured=manufactured_audit(geometry)
    print(f"running {name} timestep and energy audit",flush=True)
    runtime=eigenmode_runtime_audit(geometry)
    cases.append({
        "name":name,"source_grid":[nz,nr],"fold_amplitude":amplitude,
        "selector_maximum":geometry["selector_maximum"],
        "mass_squared":geometry["mass_squared"],"wall_stiffness":geometry["wall_stiffness"],
        "spectrum":spectrum,"manufactured":manufactured,
        "eigenmode_runtime":runtime,
    })

acceptance={
    "all_spectra_positive":all(case["spectrum"]["all_sampled_positive"] for case in cases),
    "all_fine_spectral_rates_above_1p7":all(min(rates[-1] for rates in case["spectrum"]["mode_convergence_rates"])>1.7 for case in cases),
    "all_fine_manufactured_rates_above_1p8":all(case["manufactured"]["convergence_rates"][-1]>1.8 for case in cases),
    "all_time_rates_above_3p5":all(min(case["eigenmode_runtime"]["time_convergence_rates"])>3.5 for case in cases),
    "all_energy_drifts_below_1e_9":all(max(item["relative_endpoint_energy_drift"] for item in case["eigenmode_runtime"]["records"])<1e-9 for case in cases),
}
payload={
    "status":"pass" if all(acceptance.values()) else "review",
    "scope":"exact massive fixed-background stabilizer perturbation operator on corrected G5/G6 folds",
    "radial_domain":[0.,R_MAX],"cases":cases,"acceptance":acceptance,
    "limitations":[
        "fixed-background scalar sector; metric perturbations and physical scalar-radion mixing omitted",
        "homogeneous outer r=4 perturbation boundary",
        "linear wall stiffness and time-independent corrected background",
        "not the full generalized-harmonic Einstein-scalar lower-order operator",
    ],
}
Path("results/corrected_fold_massive_scalar_evolution.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
print(json.dumps({
    "status":payload["status"],"acceptance":acceptance,
    "cases":[{
        "name":case["name"],"minimum_omega_squared":case["spectrum"]["finest_minimum_omega_squared"],
        "fine_spectral_rates":[rates[-1] for rates in case["spectrum"]["mode_convergence_rates"]],
        "manufactured_rates":case["manufactured"]["convergence_rates"],
        "time_rates":case["eigenmode_runtime"]["time_convergence_rates"],
        "maximum_energy_drift":max(item["relative_endpoint_energy_drift"] for item in case["eigenmode_runtime"]["records"]),
    } for case in cases],
},indent=2))
