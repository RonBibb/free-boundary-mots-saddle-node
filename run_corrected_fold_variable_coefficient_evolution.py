#!/usr/bin/env python3
"""Variable-coefficient reduced-wave collar audit on the corrected fold."""

import json,sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.anisotropic_initial_data import solve_anisotropic_initial_data
from bhps.finite_wall_high_order_solver import solve_finite_wall_high_order_slice
from bhps.israel_wave_matrix import analytic_robin_symmetrizer
from bhps.physical_corner_corrector import combine_shape_modes,tracefree_shape_basis
from bhps.reduced_wave_evolution import VariableCoefficientReducedWaveIBVP,anisotropic_wave_principal_coefficients
from bhps.scalar_pulse import scalar_pulse


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
        reference["z"],reference["r"],reference["q"],reference["phi"],
        a,b,c,reference["background"],chi_r,chi_z,
        initial_q=archive[f"q_{name}"],initial_phi=archive[f"phi_{name}"],
        stencil_width=7,tolerance=1e-9,iterations=30,
    )
    psi=1/(reference["z"][:,None]+selected["q"])
    return {
        "z":reference["z"],"r":reference["r"],"psi":psi,"phi":selected["phi"],
        "a":a,"b":b,"c":c,"background":reference["background"],
        "selector_maximum":selected["maximum_residual"],
    }


def wall_audit(geometry,radial_index,wall_index):
    background=geometry["background"];gamma=float(background["wall_stiffness"])
    z_index=0 if wall_index==0 else -1
    target=float(background["v0"] if wall_index==0 else background["v1"])
    phi=float(geometry["phi"][z_index,radial_index]);delta=phi-target
    potential=.5*gamma*delta**2
    if wall_index==0:
        beta=float(background["beta_a"])+(potential-float(background["wall_potential_a"]))/6
        c=beta
    else:
        beta=float(background["beta_b"])-(potential-float(background["wall_potential_b"]))/6
        c=-beta
    uprime=gamma*delta
    audit=analytic_robin_symmetrizer(c,uprime/6,uprime/2,gamma)
    return {
        "c":c,"wall_scalar":phi,"wall_potential_derivative":uprime,
        "matrix":audit["matrix"],"symmetrizer":audit["symmetrizer"],
        "minimum_symmetrizer_eigenvalue":audit["minimum_eigenvalue"],
        "condition_number":audit["condition_number"],
        "symmetry_defect":audit["symmetry_defect"],
        "maximum_robin_eigenvalue":float(np.max(np.linalg.eigvals(audit["matrix"]).real)),
    }


def ray_coefficients(geometry,radial_index):
    principal=anisotropic_wave_principal_coefficients(
        geometry["psi"][:,radial_index],geometry["a"][:,radial_index],
        geometry["b"][:,radial_index],geometry["c"][:,radial_index],
    )
    left=wall_audit(geometry,radial_index,0);right=wall_audit(geometry,radial_index,1)
    return principal,left,right


def spatial_manufactured(points,coefficient_matrix,z0,z1):
    points=np.asarray(points);x=(points-z0)/(z1-z0);length=z1-z0
    values=np.zeros((len(points),17));first=np.zeros_like(values);second=np.zeros_like(values)
    for field in range(4):
        wave=(field+1)*np.pi;slope=.1*(field+1)
        values[:,field]=np.sin(wave*x)*(1+slope*x)
        first[:,field]=(wave*np.cos(wave*x)*(1+slope*x)+slope*np.sin(wave*x))/length
        second[:,field]=(-wave**2*np.sin(wave*x)*(1+slope*x)+2*slope*wave*np.cos(wave*x))/length**2
    powers=np.column_stack((np.ones_like(x),x,x**2,x**3))
    values[:,4:]=powers@coefficient_matrix
    first[:,4:]=(coefficient_matrix[1]+2*x[:,None]*coefficient_matrix[2]+3*x[:,None]**2*coefficient_matrix[3])/length
    second[:,4:]=(2*coefficient_matrix[2]+6*x[:,None]*coefficient_matrix[3])/length**2
    values[[0,-1],:4]=0.
    return values,first,second


def manufactured_ray_audit(geometry,radial_index,sizes=(25,49,97),final_time=.06,courant=.14):
    principal,left,right=ray_coefficients(geometry,radial_index)
    z_native=geometry["z"];z0=z_native[0];z1=z_native[-1]
    w_spline=CubicSpline(z_native,principal["mass_weight"],bc_type="natural")
    p_spline=CubicSpline(z_native,principal["gradient_weight"],bc_type="natural")
    p_derivative=p_spline.derivative();a_spline=CubicSpline(z_native,principal["A"],bc_type="natural")
    boundary_spline=CubicSpline(z_native,principal["boundary_weight"],bc_type="natural")
    rng=np.random.default_rng(9100+radial_index)
    coefficients=.04*rng.normal(size=(4,13));omega=1.4;records=[]
    for size in sizes:
        points=np.linspace(z0,z1,size);w=w_spline(points);p=p_spline(points)
        system=VariableCoefficientReducedWaveIBVP(
            points,w,p,left["matrix"],right["matrix"],
            boundary_spline(z0),boundary_spline(z1),
        )
        spatial,spatial_first,spatial_second=spatial_manufactured(
            points,coefficients,z0,z1,
        )
        def source(time,x):
            values,first,second=spatial_manufactured(x,coefficients,z0,z1)
            return np.cos(omega*time)*(
                -omega**2*values
                -(p_derivative(x)[:,None]*first+p_spline(x)[:,None]*second)/w_spline(x)[:,None]
            )
        left_data=lambda time:np.cos(omega*time)*(
            -spatial_first[0,4:]/a_spline(z0)-left["matrix"]@spatial[0,4:]
        )
        right_data=lambda time:np.cos(omega*time)*(
            spatial_first[-1,4:]/a_spline(z1)-right["matrix"]@spatial[-1,4:]
        )
        maximum_speed=float(np.max(np.sqrt(p/w)))
        result=system.integrate(
            spatial,np.zeros_like(spatial),final_time,courant/maximum_speed,
            source,left_data,right_data,
        )
        exact_q=np.cos(omega*final_time)*spatial
        exact_p=-omega*np.sin(omega*final_time)*spatial
        q_error=system.l2_norm(result["position"]-exact_q)
        p_error=system.l2_norm(result["velocity"]-exact_p)
        scale=float(np.hypot(system.l2_norm(exact_q),system.l2_norm(exact_p)))
        records.append({
            "grid_points":size,"time_step":result["time_step"],
            "combined_relative_error":float(np.hypot(q_error,p_error)/scale),
            "gauge_endpoint_maximum":float(np.max(np.abs(result["position"][[0,-1],:4]))),
        })
    errors=np.array([item["combined_relative_error"] for item in records])
    rates=np.log(errors[:-1]/errors[1:])/np.log(2.)
    return {
        "radius":float(geometry["r"][radial_index]),"radial_index":radial_index,
        "mass_weight_range":[float(np.min(principal["mass_weight"])),float(np.max(principal["mass_weight"]))],
        "gradient_weight_range":[float(np.min(principal["gradient_weight"])),float(np.max(principal["gradient_weight"]))],
        "coordinate_speed_range":[float(np.min(principal["coordinate_speed"])),float(np.max(principal["coordinate_speed"]))],
        "left_wall":{"c":left["c"],"uprime":left["wall_potential_derivative"],"condition":left["condition_number"]},
        "right_wall":{"c":right["c"],"uprime":right["wall_potential_derivative"],"condition":right["condition_number"],"maximum_robin_eigenvalue":right["maximum_robin_eigenvalue"]},
        "records":records,"convergence_rates":[float(value) for value in rates],
    }


def hermite_robin_state(points,left,right,a_left,a_right,seed):
    z0=points[0];z1=points[-1];length=z1-z0;x=(points-z0)/length
    left_value=np.asarray(seed);right_value=.7*np.asarray(seed)[::-1]
    left_derivative=-a_left*(left["matrix"]@left_value)
    right_derivative=a_right*(right["matrix"]@right_value)
    h00=2*x**3-3*x**2+1;h10=x**3-2*x**2+x
    h01=-2*x**3+3*x**2;h11=x**3-x**2
    values=(
        h00[:,None]*left_value+h10[:,None]*length*left_derivative
        +h01[:,None]*right_value+h11[:,None]*length*right_derivative
    )
    state=np.zeros((len(points),17));state[:,4:]=values
    return state


def energy_ledger_audit(geometry,radial_index,sizes=(25,49,97),final_time=.025,courant=.10,shift=500.):
    principal,left,right=ray_coefficients(geometry,radial_index);z=geometry["z"]
    splines={key:CubicSpline(z,principal[key],bc_type="natural") for key in ("mass_weight","gradient_weight","boundary_weight","A")}
    rng=np.random.default_rng(18000+radial_index);seed=.01*rng.normal(size=13);records=[]
    for size in sizes:
        points=np.linspace(z[0],z[-1],size);w=splines["mass_weight"](points);p=splines["gradient_weight"](points)
        system=VariableCoefficientReducedWaveIBVP(
            points,w,p,left["matrix"],right["matrix"],
            splines["boundary_weight"](z[0]),splines["boundary_weight"](z[-1]),
        )
        initial=hermite_robin_state(
            points,left,right,float(splines["A"](z[0])),float(splines["A"](z[-1])),seed,
        )
        diagnostic=lambda t,q,v:system.interpolated_symmetrizer_energy(
            q,v,left["symmetrizer"],right["symmetrizer"],shift,
        )
        maximum_speed=float(np.max(np.sqrt(p/w)))
        result=system.integrate(
            initial,np.zeros_like(initial),final_time,courant/maximum_speed,
            diagnostic=diagnostic,
        )
        energies=np.array([item["shifted"] for item in result["diagnostics"]])
        powers=np.array([item["predicted_shifted_energy_power"] for item in result["diagnostics"]])
        integral=np.trapezoid(powers,dx=result["time_step"])
        ledger=float(energies[-1]-energies[0]-integral)
        scale=max(abs(energies[0]),abs(energies[-1]),abs(integral),1e-300)
        records.append({
            "grid_points":size,"time_step":result["time_step"],
            "initial_shifted_energy":float(energies[0]),"minimum_shifted_energy":float(np.min(energies)),
            "maximum_shifted_energy_amplification":float(np.max(energies)/energies[0]),
            "normalized_energy_ledger_residual":abs(ledger)/scale,
        })
    errors=np.array([item["normalized_energy_ledger_residual"] for item in records])
    rates=np.log(errors[:-1]/errors[1:])/np.log(2.)
    return {
        "radius":float(geometry["r"][radial_index]),"radial_index":radial_index,
        "energy_shift":shift,"records":records,"ledger_convergence_rates":[float(value) for value in rates],
    }


def constraint_self_convergence(geometry,radial_index,sizes=(25,49,97,193),final_time=.04,courant=.12):
    principal,_,_=ray_coefficients(geometry,radial_index);z=geometry["z"]
    splines={key:CubicSpline(z,principal[key],bc_type="natural") for key in ("mass_weight","gradient_weight","boundary_weight")}
    solutions=[]
    for size in sizes:
        points=np.linspace(z[0],z[-1],size);x=(points-z[0])/(z[-1]-z[0])
        w=splines["mass_weight"](points);p=splines["gradient_weight"](points);zero=np.zeros((4,4))
        system=VariableCoefficientReducedWaveIBVP(
            points,w,p,zero,zero,splines["boundary_weight"](z[0]),splines["boundary_weight"](z[-1]),
            dirichlet_fields=1,
        )
        initial=np.zeros((size,5));initial[:,0]=np.sin(np.pi*x)
        for field in range(4):initial[:,field+1]=np.cos((field+1)*np.pi*x)
        maximum_speed=float(np.max(np.sqrt(p/w)))
        result=system.integrate(initial,np.zeros_like(initial),final_time,courant/maximum_speed)
        solutions.append((system,result["position"]))
    reference=solutions[-1][1];errors=[];records=[]
    intervals=sizes[-1]-1
    for (size,(system,solution)) in zip(sizes[:-1],solutions[:-1]):
        stride=intervals//(size-1);sampled=reference[::stride]
        error=system.l2_norm(solution-sampled)/system.l2_norm(sampled);errors.append(error)
        records.append({
            "grid_points":size,"relative_to_193_l2_error":error,
            "normal_dirichlet_endpoint_maximum":float(np.max(np.abs(solution[[0,-1],0]))),
        })
    rates=np.log(np.array(errors[:-1])/np.array(errors[1:]))/np.log(2.)
    return {
        "radius":float(geometry["r"][radial_index]),"records":records,
        "self_convergence_rates":[float(value) for value in rates],
    }


arclength_g5=json.loads(Path("results/corrected_anisotropic_arclength.json").read_text())
arclength_g6=json.loads(Path("results/corrected_anisotropic_arclength_G6.json").read_text())
archive=np.load("results/corrected_family_knot_A8_state.npz");coefficients=archive["coefficients"]
specifications=(
    ("G5R8",49,73,float(arclength_g5["summary"]["fine_fold_amplitude"])),
    ("G6R8",65,97,float(arclength_g6["summary"]["fine_fold_amplitude"])),
)
cases=[]
for name,nz,nr,amplitude in specifications:
    geometry=corrected_geometry(name,nz,nr,amplitude,archive,coefficients)
    targets=(0.,.5,1.,2.,4.,7.)
    indices=sorted(set(int(np.argmin(abs(geometry["r"]-target))) for target in targets))
    manufactured=[manufactured_ray_audit(geometry,index) for index in indices]
    ledger_indices=(indices[0],indices[min(2,len(indices)-1)],indices[-1])
    ledgers=[energy_ledger_audit(geometry,index) for index in ledger_indices]
    constraints=[constraint_self_convergence(geometry,index) for index in (indices[0],indices[min(2,len(indices)-1)])]
    cases.append({
        "name":name,"grid_size":[nz,nr],"amplitude":amplitude,
        "selector_maximum":geometry["selector_maximum"],
        "manufactured_rays":manufactured,"energy_ledgers":ledgers,
        "constraint_self_convergence":constraints,
    })

all_manufactured=[ray for case in cases for ray in case["manufactured_rays"]]
all_ledgers=[ray for case in cases for ray in case["energy_ledgers"]]
all_constraints=[ray for case in cases for ray in case["constraint_self_convergence"]]
acceptance={
    "all_manufactured_rates_above_1p8":all(min(ray["convergence_rates"])>1.8 for ray in all_manufactured),
    "all_gauge_endpoints_exact":all(max(item["gauge_endpoint_maximum"] for item in ray["records"])<1e-14 for ray in all_manufactured),
    "all_shifted_energies_positive":all(min(item["minimum_shifted_energy"] for item in ray["records"])>0 for ray in all_ledgers),
    "all_energy_ledgers_converge":all(ray["ledger_convergence_rates"][-1]>1.5 and ray["records"][-1]["normalized_energy_ledger_residual"]<2e-3 for ray in all_ledgers),
    "all_constraint_self_convergence_rates_above_1p7":all(min(ray["self_convergence_rates"])>1.7 for ray in all_constraints),
    "all_constraint_dirichlet_endpoints_exact":all(max(item["normal_dirichlet_endpoint_maximum"] for item in ray["records"])<1e-14 for ray in all_constraints),
}
payload={
    "status":"pass" if all(acceptance.values()) else "review",
    "scope":"variable-coefficient one-normal-dimensional principal reduced-wave collars extracted from corrected G5/G6 folds",
    "cases":cases,"acceptance":acceptance,
    "limitations":[
        "independent radial collars omit radial derivatives and inter-ray coupling",
        "bulk connection, curvature, matter, and evolved generalized-harmonic lower-order terms are omitted",
        "the wall matrices vary with radius but are frozen in time",
        "constraint test advances the induced principal propagation model rather than the nonlinear Einstein constraints",
        "not a nonlinear horizon or branch-selection evolution",
    ],
}
Path("results/corrected_fold_variable_coefficient_evolution.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
summary={
    "status":payload["status"],"acceptance":acceptance,
    "manufactured_rate_range":[min(min(ray["convergence_rates"]) for ray in all_manufactured),max(max(ray["convergence_rates"]) for ray in all_manufactured)],
    "finest_energy_ledger_residual_range":[min(ray["records"][-1]["normalized_energy_ledger_residual"] for ray in all_ledgers),max(ray["records"][-1]["normalized_energy_ledger_residual"] for ray in all_ledgers)],
    "constraint_rate_range":[min(min(ray["self_convergence_rates"]) for ray in all_constraints),max(max(ray["self_convergence_rates"]) for ray in all_constraints)],
}
print(json.dumps(summary,indent=2))
