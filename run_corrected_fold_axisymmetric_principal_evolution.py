#!/usr/bin/env python3
"""Coupled (z,r) principal evolution audit on the corrected folds."""

import json,sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.sparse.linalg import eigsh

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_initial_data import solve_anisotropic_initial_data
from bhps.axisymmetric_reduced_wave_evolution import AxisymmetricVariableReducedWaveIBVP,axisymmetric_principal_coefficients
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice
from bhps.israel_wave_matrix import analytic_robin_symmetrizer
from bhps.physical_corner_corrector import combine_shape_modes,tracefree_shape_basis
from bhps.scalar_pulse import scalar_pulse


GRID_SIZES=((13,17),(25,33),(49,65));R_MAX=4.;OMEGA=1.3


def corrected_geometry(name,nz,nr,amplitude,archive,coefficients):
    reference=solve_finite_wall_high_order_slice(
        amplitude,nz=nz,nr=nr,r_max=8.,wall_stiffness=20.,epsilon=.1,
        backreaction=.01,tolerance=1e-10,iterations=240,
    )
    _,chi_r,chi_z=scalar_pulse(reference["z"],reference["r"],amplitude)
    modes=tracefree_shape_basis(
        reference["z"],reference["r"],6,(.5,1.),8.,
        ((7.5,1.5),(7.5,3.0)),
    )["modes"]
    a,b,c=combine_shape_modes(coefficients,modes)
    selected=solve_anisotropic_initial_data(
        reference["z"],reference["r"],reference["q"],reference["phi"],a,b,c,
        reference["background"],chi_r,chi_z,
        initial_q=archive[f"q_{name}"],initial_phi=archive[f"phi_{name}"],
        stencil_width=7,tolerance=1e-9,iterations=30,
    )
    psi=1/(reference["z"][:,None]+selected["q"])
    principal=axisymmetric_principal_coefficients(psi,a,b,c)
    return {
        "z":reference["z"],"r":reference["r"],"psi":psi,"phi":selected["phi"],
        "a":a,"b":b,"c":c,"principal":principal,"background":reference["background"],
        "selector_maximum":selected["maximum_residual"],
    }


def spline(field,z,r):
    return RectBivariateSpline(z,r,np.asarray(field),kx=3,ky=3,s=0)


def wall_fields(geometry,r_values,wall_index):
    background=geometry["background"];gamma=float(background["wall_stiffness"])
    zi=0 if wall_index==0 else -1
    phi_spline=spline(geometry["phi"],geometry["z"],geometry["r"])
    phi=np.asarray(phi_spline(geometry["z"][zi],r_values,grid=True)).reshape(-1)
    target=float(background["v0"] if wall_index==0 else background["v1"])
    delta=phi-target;potential=.5*gamma*delta**2
    if wall_index==0:
        beta=float(background["beta_a"])+(potential-float(background["wall_potential_a"]))/6
        c_values=beta
    else:
        beta=float(background["beta_b"])-(potential-float(background["wall_potential_b"]))/6
        c_values=-beta
    matrices=[];symmetrizers=[];conditions=[];eigenvalues=[]
    for c,uprime in zip(c_values,gamma*delta):
        audit=analytic_robin_symmetrizer(c,uprime/6,uprime/2,gamma)
        matrices.append(audit["matrix"]);symmetrizers.append(audit["symmetrizer"])
        conditions.append(audit["condition_number"])
        eigenvalues.append(np.max(np.linalg.eigvals(audit["matrix"]).real))
    return {
        "matrix":np.asarray(matrices),"symmetrizer":np.asarray(symmetrizers),
        "c":np.asarray(c_values),"uprime":gamma*delta,
        "condition":np.asarray(conditions),"maximum_robin_eigenvalue":np.asarray(eigenvalues),
    }


def sampled_system(geometry,nz,nr,robin=True):
    z=np.linspace(geometry["z"][0],geometry["z"][-1],nz);r=np.linspace(0,R_MAX,nr)
    zz,rr=np.meshgrid(z,r,indexing="ij");principal={}
    for key in ("mass_weight","z_gradient_weight","r_gradient_weight","z_boundary_weight","A"):
        principal[key]=spline(
            geometry["principal"][key],geometry["z"],geometry["r"],
        ).ev(zz.ravel(),rr.ravel()).reshape(nz,nr)
    principal["z_coordinate_speed"]=np.sqrt(principal["z_gradient_weight"]/principal["mass_weight"])
    principal["r_coordinate_speed"]=np.sqrt(principal["r_gradient_weight"]/principal["mass_weight"])
    if robin:
        left=wall_fields(geometry,r,0);right=wall_fields(geometry,r,1)
    else:
        left={"matrix":np.zeros((nr,13,13)),"symmetrizer":np.tile(np.eye(13),(nr,1,1))}
        right={"matrix":np.zeros((nr,13,13)),"symmetrizer":np.tile(np.eye(13),(nr,1,1))}
    system=AxisymmetricVariableReducedWaveIBVP(
        z,r,principal["mass_weight"],principal["z_gradient_weight"],principal["r_gradient_weight"],
        left["matrix"],right["matrix"],principal["z_boundary_weight"][0],principal["z_boundary_weight"][-1],
    )
    return system,principal,left,right


def manufactured_fields(z,r,coefficients):
    zz,rr=np.meshgrid(z,r,indexing="ij");x=(zz-z[0])/(z[-1]-z[0]);length=z[-1]-z[0]
    radial_wave=np.pi/(2*R_MAX);g=np.cos(radial_wave*rr)
    gr=-radial_wave*np.sin(radial_wave*rr);grr=-radial_wave**2*g
    values=np.zeros((len(z),len(r),17));uz=np.zeros_like(values);uzz=np.zeros_like(values)
    ur=np.zeros_like(values);urr=np.zeros_like(values)
    for field in range(4):
        wave=(field+1)*np.pi;slope=.08*(field+1)
        f=np.sin(wave*x)*(1+slope*x)
        fz=(wave*np.cos(wave*x)*(1+slope*x)+slope*np.sin(wave*x))/length
        fzz=(-wave**2*np.sin(wave*x)*(1+slope*x)+2*slope*wave*np.cos(wave*x))/length**2
        values[:,:,field]=f*g;uz[:,:,field]=fz*g;uzz[:,:,field]=fzz*g
        ur[:,:,field]=f*gr;urr[:,:,field]=f*grr
    xp=np.stack((np.ones_like(x),x,x**2,x**3),axis=-1)
    f=np.einsum("ijm,mk->ijk",xp,coefficients)
    fz=np.einsum(
        "ijm,mk->ijk",np.stack((np.zeros_like(x),np.ones_like(x),2*x,3*x**2),axis=-1),coefficients,
    )/length
    fzz=np.einsum(
        "ijm,mk->ijk",np.stack((np.zeros_like(x),np.zeros_like(x),2*np.ones_like(x),6*x),axis=-1),coefficients,
    )/length**2
    values[:,:,4:]=f*g[:,:,None];uz[:,:,4:]=fz*g[:,:,None];uzz[:,:,4:]=fzz*g[:,:,None]
    ur[:,:,4:]=f*gr[:,:,None];urr[:,:,4:]=f*grr[:,:,None]
    values[:,-1,:]=0.;uz[:,-1,:]=0.;uzz[:,-1,:]=0.
    return values,uz,uzz,ur,urr


def manufactured_audit(geometry,final_time=.018,courant=.035):
    rng=np.random.default_rng(20260814);coefficients=.025*rng.normal(size=(4,13));records=[]
    for nz,nr in GRID_SIZES:
        system,principal,left,right=sampled_system(geometry,nz,nr,True)
        values,uz,uzz,ur,urr=manufactured_fields(system.z,system.r,coefficients)
        zz,rr=np.meshgrid(system.z,system.r,indexing="ij")
        pz_spline=spline(geometry["principal"]["z_gradient_weight"],geometry["z"],geometry["r"])
        pr_spline=spline(geometry["principal"]["r_gradient_weight"],geometry["z"],geometry["r"])
        pz_z=pz_spline.ev(zz.ravel(),rr.ravel(),dx=1,dy=0).reshape(nz,nr)
        pr_r=pr_spline.ev(zz.ravel(),rr.ravel(),dx=0,dy=1).reshape(nz,nr)
        z_term=pz_z[:,:,None]*uz+principal["z_gradient_weight"][:,:,None]*uzz
        radial_term=pr_r[:,:,None]*ur+principal["r_gradient_weight"][:,:,None]*urr
        radial_term[:,1:]+=2*principal["r_gradient_weight"][:,1:,None]*ur[:,1:]/rr[:,1:,None]
        radial_term[:,0]=3*principal["r_gradient_weight"][:,0,None]*urr[:,0]
        strong_source=-OMEGA**2*values-(z_term+radial_term)/principal["mass_weight"][:,:,None]
        left_data=np.empty((nr,13));right_data=np.empty((nr,13))
        for j in range(nr):
            left_data[j]=-uz[0,j,4:]/principal["A"][0,j]-left["matrix"][j]@values[0,j,4:]
            right_data[j]=uz[-1,j,4:]/principal["A"][-1,j]-right["matrix"][j]@values[-1,j,4:]
        source=lambda time,zq,rq:np.cos(OMEGA*time)*strong_source
        lower=lambda time,rq:np.cos(OMEGA*time)*left_data
        upper=lambda time,rq:np.cos(OMEGA*time)*right_data
        speed=max(np.max(principal["z_coordinate_speed"]),np.max(principal["r_coordinate_speed"]))
        result=system.integrate(
            values,np.zeros_like(values),final_time,courant/speed,source,lower,upper,
        )
        exact_q=np.cos(OMEGA*final_time)*values;exact_v=-OMEGA*np.sin(OMEGA*final_time)*values
        error=float(np.hypot(system.l2_norm(result["position"]-exact_q),system.l2_norm(result["velocity"]-exact_v)))
        scale=float(np.hypot(system.l2_norm(exact_q),system.l2_norm(exact_v)))
        records.append({
            "grid_size":[nz,nr],"time_step":result["time_step"],"combined_relative_error":error/scale,
            "fixed_boundary_maximum":float(max(
                np.max(np.abs(result["position"][[0,-1],:,:4])),
                np.max(np.abs(result["position"][:,-1,:])),
            )),
        })
    errors=np.array([item["combined_relative_error"] for item in records])
    rates=np.log(errors[:-1]/errors[1:])/np.log(2.)
    return {"records":records,"convergence_rates":[float(value) for value in rates]}


def constraint_audit(geometry,final_time=.018,courant=.035):
    constraint_grids=GRID_SIZES+((97,129),);records=[];normal_spectra=[];tangential_spectra=[]
    for nz,nr in constraint_grids:
        system,principal,_,_=sampled_system(geometry,nz,nr,False)
        mg=system.mass[system.gauge_free][:,system.gauge_free]
        kg=system.stiffness[system.gauge_free][:,system.gauge_free]
        mr=system.mass[system.robin_free][:,system.robin_free]
        kr=system.stiffness[system.robin_free][:,system.robin_free]
        normal_values,normal_vectors=eigsh(kg,k=1,M=mg,sigma=0.,which="LM")
        tangential_values,tangential_vectors=eigsh(kr,k=4,M=mr,sigma=0.,which="LM")
        normal_order=np.argsort(normal_values);tangential_order=np.argsort(tangential_values)
        normal_values=normal_values[normal_order];normal_vectors=normal_vectors[:,normal_order]
        tangential_values=tangential_values[tangential_order];tangential_vectors=tangential_vectors[:,tangential_order]
        initial=np.zeros((nz,nr,17));flat=initial.reshape(system.nodes,17)
        flat[system.gauge_free,0]=normal_vectors[:,0]
        for field in range(4):flat[system.robin_free,field+4]=tangential_vectors[:,field]
        speed=max(np.max(principal["z_coordinate_speed"]),np.max(principal["r_coordinate_speed"]))
        result=system.integrate(initial,np.zeros_like(initial),final_time,courant/speed)
        exact=np.zeros_like(initial);exact_flat=exact.reshape(system.nodes,17)
        exact_flat[system.gauge_free,0]=np.cos(np.sqrt(normal_values[0])*final_time)*normal_vectors[:,0]
        for field in range(4):
            exact_flat[system.robin_free,field+4]=np.cos(np.sqrt(tangential_values[field])*final_time)*tangential_vectors[:,field]
        error=system.l2_norm(result["position"]-exact)/system.l2_norm(exact)
        normal_spectra.append(normal_values.copy());tangential_spectra.append(tangential_values.copy())
        records.append({
            "grid_size":[nz,nr],"normal_eigenvalue":float(normal_values[0]),
            "tangential_eigenvalues":[float(value) for value in tangential_values],
            "semidiscrete_solution_relative_error":error,
            "normal_dirichlet_endpoint_maximum":float(np.max(np.abs(result["position"][[0,-1],:,0]))),
        })
    normal_spectra=np.asarray(normal_spectra).reshape(-1)
    tangential_spectra=np.asarray(tangential_spectra)
    normal_rates=np.log(np.abs(np.diff(normal_spectra[:-1]))/np.abs(np.diff(normal_spectra[1:])))/np.log(2.)
    tangent_rates=[]
    for mode in range(4):
        values=tangential_spectra[:,mode]
        tangent_rates.append(np.log(np.abs(np.diff(values[:-1]))/np.abs(np.diff(values[1:])))/np.log(2.))
    return {
        "construction":"source-free generalized eigenmodes satisfy every semidiscrete differentiated wall condition",
        "records":records,"normal_eigenvalue_convergence_rates":[float(value) for value in normal_rates],
        "tangential_eigenvalue_convergence_rates":[[float(value) for value in rates] for rates in tangent_rates],
    }


def compatible_hermite_state(system,principal,left,right,seed):
    x=(system.z-system.z[0])/(system.z[-1]-system.z[0]);length=system.z[-1]-system.z[0]
    g=np.cos(np.pi*system.r/(2*R_MAX));left_value=g[:,None]*seed
    right_value=.7*g[:,None]*seed[::-1]
    left_derivative=np.array([-principal["A"][0,j]*(left["matrix"][j]@left_value[j]) for j in range(system.nr)])
    right_derivative=np.array([principal["A"][-1,j]*(right["matrix"][j]@right_value[j]) for j in range(system.nr)])
    h00=2*x**3-3*x**2+1;h10=x**3-2*x**2+x;h01=-2*x**3+3*x**2;h11=x**3-x**2
    values=(
        h00[:,None,None]*left_value[None,:,:]+h10[:,None,None]*length*left_derivative[None,:,:]
        +h01[:,None,None]*right_value[None,:,:]+h11[:,None,None]*length*right_derivative[None,:,:]
    )
    state=np.zeros((system.nz,system.nr,17));state[:,:,4:]=values;state[:,-1,:]=0.
    return state


def energy_audit(geometry,final_time=.006,courant=.025,shift=500.):
    records=[];rng=np.random.default_rng(8128);seed=.004*rng.normal(size=13)
    for nz,nr in GRID_SIZES:
        system,principal,left,right=sampled_system(geometry,nz,nr,True)
        initial=compatible_hermite_state(system,principal,left,right,seed)
        diagnostic=lambda t,q,v:system.interpolated_symmetrizer_energy(
            q,v,left["symmetrizer"],right["symmetrizer"],shift,
        )
        speed=max(np.max(principal["z_coordinate_speed"]),np.max(principal["r_coordinate_speed"]))
        result=system.integrate(
            initial,np.zeros_like(initial),final_time,courant/speed,diagnostic=diagnostic,
        )
        energies=np.array([item["shifted"] for item in result["diagnostics"]])
        powers=np.array([item["predicted_shifted_energy_power"] for item in result["diagnostics"]])
        integral=np.trapezoid(powers,dx=result["time_step"])
        ledger=energies[-1]-energies[0]-integral
        scale=max(abs(energies[0]),abs(energies[-1]),abs(integral),1e-300)
        records.append({
            "grid_size":[nz,nr],"time_step":result["time_step"],
            "minimum_shifted_energy":float(np.min(energies)),
            "maximum_energy_amplification":float(np.max(energies)/energies[0]),
            "normalized_energy_ledger_residual":float(abs(ledger)/scale),
        })
    errors=np.array([item["normalized_energy_ledger_residual"] for item in records])
    rates=np.log(errors[:-1]/errors[1:])/np.log(2.)
    return {"energy_shift":shift,"records":records,"ledger_convergence_rates":[float(value) for value in rates]}


g5=json.loads(Path("results/corrected_anisotropic_arclength.json").read_text())
g6=json.loads(Path("results/corrected_anisotropic_arclength_G6.json").read_text())
archive=np.load("results/corrected_family_knot_A8_state.npz");coefficients=archive["coefficients"]
specifications=(
    ("G5R8",49,73,float(g5["summary"]["fine_fold_amplitude"])),
    ("G6R8",65,97,float(g6["summary"]["fine_fold_amplitude"])),
)
cases=[]
for name,nz,nr,amplitude in specifications:
    geometry=corrected_geometry(name,nz,nr,amplitude,archive,coefficients)
    cases.append({
        "name":name,"source_grid":[nz,nr],"fold_amplitude":amplitude,
        "selector_maximum":geometry["selector_maximum"],
        "manufactured":manufactured_audit(geometry),
        "constraint_propagation":constraint_audit(geometry),
        "energy_ledger":energy_audit(geometry),
    })

acceptance={
    "all_fine_manufactured_rates_above_1p8":all(case["manufactured"]["convergence_rates"][-1]>1.8 for case in cases),
    "all_fixed_boundaries_exact":all(max(item["fixed_boundary_maximum"] for item in case["manufactured"]["records"])<1e-13 for case in cases),
    "all_fine_constraint_eigenvalue_rates_above_1p7":all(
        case["constraint_propagation"]["normal_eigenvalue_convergence_rates"][-1]>1.7
        and min(rates[-1] for rates in case["constraint_propagation"]["tangential_eigenvalue_convergence_rates"])>1.7
        for case in cases
    ),
    "all_constraint_semidiscrete_propagation_errors_below_1e_8":all(
        max(item["semidiscrete_solution_relative_error"] for item in case["constraint_propagation"]["records"])<1e-8
        for case in cases
    ),
    "all_constraint_dirichlet_endpoints_exact":all(max(item["normal_dirichlet_endpoint_maximum"] for item in case["constraint_propagation"]["records"])<1e-13 for case in cases),
    "all_shifted_energies_positive":all(min(item["minimum_shifted_energy"] for item in case["energy_ledger"]["records"])>0 for case in cases),
    "all_energy_ledgers_converge":all(case["energy_ledger"]["ledger_convergence_rates"][-1]>1.4 and case["energy_ledger"]["records"][-1]["normalized_energy_ledger_residual"]<2e-3 for case in cases),
}
payload={
    "status":"pass" if all(acceptance.values()) else "review",
    "scope":"coupled axisymmetric (z,r) variable-principal 17-field evolution on the corrected G5/G6 folds",
    "radial_domain":[0.,R_MAX],"nested_grids":[list(item) for item in GRID_SIZES],
    "cases":cases,"acceptance":acceptance,
    "limitations":[
        "principal diagonal wave operator only; connection, curvature, matter, and generalized-harmonic lower-order couplings omitted",
        "outer r=4 perturbation Dirichlet boundary used for a short-time source-region test",
        "wall matrices depend on radius but are frozen in time",
        "constraint audit evolves compatible eigenmodes of the principal induced propagation system",
        "no nonlinear apparent-horizon or branch-selection evolution",
    ],
}
Path("results/corrected_fold_axisymmetric_principal_evolution.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps({
    "status":payload["status"],"acceptance":acceptance,
    "cases":[{
        "name":case["name"],"manufactured_rates":case["manufactured"]["convergence_rates"],
        "normal_constraint_rates":case["constraint_propagation"]["normal_eigenvalue_convergence_rates"],
        "tangential_constraint_rates":case["constraint_propagation"]["tangential_eigenvalue_convergence_rates"],
        "energy_rates":case["energy_ledger"]["ledger_convergence_rates"],
        "finest_energy_residual":case["energy_ledger"]["records"][-1]["normalized_energy_ledger_residual"],
    } for case in cases],
},indent=2))
