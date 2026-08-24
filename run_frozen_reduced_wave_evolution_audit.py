#!/usr/bin/env python3
"""Runtime convergence audit for the frozen 17-field reduced-wave IBVP."""

import json,sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.gw_background import solve_gw_background
from bhps.israel_wave_matrix import analytic_robin_symmetrizer
from bhps.reduced_wave_evolution import FrozenReducedWaveIBVP,endpoint_robin_residual


def wall_system(background,gamma,wall_index):
    index=0 if wall_index==0 else -1
    target=background["v0"] if wall_index==0 else background["v1"]
    uprime=float(gamma*(background["phi"][index]-target))
    coefficient=float(1. if wall_index==0 else -background["beta_b"])
    audit=analytic_robin_symmetrizer(
        coefficient,uprime/6,uprime/2,gamma,
    )
    return {
        "name":"lower" if wall_index==0 else "upper",
        "c":coefficient,"wall_potential_derivative":uprime,
        "matrix":audit["matrix"],"symmetrizer":audit["symmetrizer"],
        "symmetry_defect":audit["symmetry_defect"],
    }


def convergence_rates(errors):
    errors=np.asarray(errors,dtype=float)
    return [float(value) for value in np.log(errors[:-1]/errors[1:])/np.log(2.)]


def manufactured_audit(left,right,sizes=(25,49,97),final_time=.18,courant=.18):
    rng=np.random.default_rng(20260813)
    coefficients=.08*rng.normal(size=(4,13))
    omega=1.7

    def spatial_fields(points):
        points=np.asarray(points);fields=np.zeros((len(points),17));second=np.zeros_like(fields)
        for field in range(4):
            wave=(field+1)*np.pi
            factor=1+.1*(field+1)*points
            fields[:,field]=np.sin(wave*points)*factor
            second[:,field]=-wave*wave*np.sin(wave*points)*factor+2*.1*(field+1)*wave*np.cos(wave*points)
        powers=np.column_stack((np.ones_like(points),points,points**2,points**3))
        fields[:,4:]=powers@coefficients
        second[:,4:]=2*coefficients[2]+6*points[:,None]*coefficients[3]
        fields[[0,-1],:4]=0.
        return fields,second

    left_value=coefficients[0]
    left_derivative=coefficients[1]
    right_value=np.sum(coefficients,axis=0)
    right_derivative=coefficients[1]+2*coefficients[2]+3*coefficients[3]
    records=[]
    for size in sizes:
        points=np.linspace(0,1,size);system=FrozenReducedWaveIBVP(
            points,left["matrix"],right["matrix"],
        )
        spatial,spatial_second=spatial_fields(points)
        source=lambda t,x,spatial=spatial,second=spatial_second:np.cos(omega*t)*(-omega*omega*spatial-second)
        left_data=lambda t:np.cos(omega*t)*(-left_derivative-left["matrix"]@left_value)
        right_data=lambda t:np.cos(omega*t)*(right_derivative-right["matrix"]@right_value)
        result=system.integrate(
            spatial,np.zeros_like(spatial),final_time,courant,
            source,left_data,right_data,
        )
        exact_position=np.cos(omega*final_time)*spatial
        exact_velocity=-omega*np.sin(omega*final_time)*spatial
        position_error=system.l2_norm(result["position"]-exact_position)
        velocity_error=system.l2_norm(result["velocity"]-exact_velocity)
        combined=float(np.hypot(position_error,velocity_error))
        exact_combined=float(np.hypot(system.l2_norm(exact_position),system.l2_norm(exact_velocity)))
        records.append({
            "grid_points":size,"time_step":result["time_step"],"steps":result["steps"],
            "position_l2_error":position_error,"velocity_l2_error":velocity_error,
            "combined_relative_error":combined/exact_combined,
            "maximum_gauge_dirichlet_endpoint":float(np.max(np.abs(result["position"][[0,-1],:4]))),
        })
    errors=[record["combined_relative_error"] for record in records]
    return {"records":records,"combined_error_rates":convergence_rates(errors)}


def positive_growth_audit(upper,sizes=(33,65,129),final_time=.08,courant=.15):
    values,vectors=np.linalg.eig(upper["matrix"])
    index=int(np.argmax(values.real));robin_value=float(values[index].real)
    vector=np.real(vectors[:,index]);vector/=np.sqrt(vector@upper["symmetrizer"]@vector)
    growth=float(brentq(lambda value:value*np.tanh(value/2)-robin_value,1e-10,2*robin_value+10))
    records=[]
    for size in sizes:
        points=np.linspace(0,1,size);system=FrozenReducedWaveIBVP(
            points,upper["matrix"],upper["matrix"],
        )
        profile=np.cosh(growth*(points-.5))/np.cosh(growth/2)
        initial=np.zeros((size,17));initial[:,4:]=profile[:,None]*vector
        result=system.integrate(initial,np.zeros_like(initial),final_time,courant)
        exact=np.cosh(growth*final_time)*initial
        relative_error=system.l2_norm(result["position"]-exact)/system.l2_norm(exact)
        mass=system.mass
        projection=float(np.sum((mass@result["position"][:,4:])*(initial[:,4:]@upper["symmetrizer"])))
        normalization=float(np.sum((mass@initial[:,4:])*(initial[:,4:]@upper["symmetrizer"])))
        amplification=projection/normalization
        measured=float(np.arccosh(max(1.,amplification))/final_time)
        endpoint=endpoint_robin_residual(
            points,result["position"][:,4:],upper["matrix"],upper["matrix"],
        )
        records.append({
            "grid_points":size,"time_step":result["time_step"],
            "relative_solution_error":relative_error,
            "measured_growth_rate":measured,
            "relative_growth_rate_error":abs(measured-growth)/growth,
            "endpoint_normalized_robin_residual":endpoint["normalized_l2"],
        })
    return {
        "robin_eigenvalue":robin_value,"continuum_growth_rate":growth,
        "continuum_amplification":float(np.cosh(growth*final_time)),
        "records":records,
        "solution_error_rates":convergence_rates([x["relative_solution_error"] for x in records]),
        "growth_error_rates":convergence_rates([x["relative_growth_rate_error"] for x in records]),
        "interpretation":"finite lower-order mirrored-wall coordinate growth; convergence toward a finite rate, not grid-frequency instability",
    }


def stable_energy_audit(lower,size=129,final_time=.4,courant=.15):
    values,vectors=np.linalg.eig(lower["matrix"])
    target=-2*lower["c"]
    index=int(np.argmin(np.abs(values.real-target)))
    robin_value=float(values[index].real)
    vector=np.real(vectors[:,index]);vector/=np.sqrt(vector@lower["symmetrizer"]@vector)
    wave=float(brentq(lambda value:value*np.tan(value/2)+robin_value,1e-10,np.pi-1e-10))
    points=np.linspace(0,1,size);system=FrozenReducedWaveIBVP(
        points,lower["matrix"],lower["matrix"],
    )
    profile=np.cos(wave*(points-.5));initial=np.zeros((size,17));initial[:,4:]=profile[:,None]*vector
    energy=lambda t,q,p:system.symmetrized_energy(q,p,lower["symmetrizer"])["total"]
    result=system.integrate(
        initial,np.zeros_like(initial),final_time,courant,diagnostic=energy,
    )
    energies=np.asarray(result["diagnostics"]);baseline=energies[0]
    exact=np.cos(wave*final_time)*initial
    return {
        "robin_eigenvalue":robin_value,"continuum_frequency":wave,
        "time_step":result["time_step"],"steps":result["steps"],
        "maximum_relative_energy_drift":float(np.max(np.abs(energies-baseline))/abs(baseline)),
        "relative_solution_error":system.l2_norm(result["position"]-exact)/system.l2_norm(exact),
        "minimum_energy":float(np.min(energies)),
    }


def constraint_propagation_audit(sizes=(25,49,97),final_time=.2,courant=.18):
    records=[]
    for size in sizes:
        points=np.linspace(0,1,size);zero=np.zeros((4,4))
        system=FrozenReducedWaveIBVP(points,zero,zero,dirichlet_fields=1)
        initial=np.zeros((size,5));initial[:,0]=np.sin(np.pi*points)
        for field in range(4):initial[:,field+1]=np.cos((field+1)*np.pi*points)
        result=system.integrate(initial,np.zeros_like(initial),final_time,courant)
        exact=np.zeros_like(initial);exact[:,0]=np.cos(np.pi*final_time)*np.sin(np.pi*points)
        for field in range(4):
            wave=(field+1)*np.pi
            exact[:,field+1]=np.cos(wave*final_time)*np.cos(wave*points)
        error=system.l2_norm(result["position"]-exact)/system.l2_norm(exact)
        endpoint=endpoint_robin_residual(points,result["position"][:,1:],zero,zero)
        records.append({
            "grid_points":size,"relative_l2_error":error,
            "normal_dirichlet_endpoint_maximum":float(np.max(np.abs(result["position"][[0,-1],0]))),
            "tangential_neumann_endpoint_normalized_residual":endpoint["normalized_l2"],
        })
    return {"records":records,"error_rates":convergence_rates([x["relative_l2_error"] for x in records])}


gamma=20.;background=solve_gw_background(
    np.linspace(1,np.e,257),epsilon=.1,backreaction=.01,
    wall_stiffness=gamma,tolerance=1e-11,
)
lower=wall_system(background,gamma,0);upper=wall_system(background,gamma,1)
manufactured=manufactured_audit(lower,upper)
growth=positive_growth_audit(upper)
energy=stable_energy_audit(lower)
constraints=constraint_propagation_audit()

passes={
    "manufactured_second_order":min(manufactured["combined_error_rates"])>1.8,
    "gauge_dirichlet_exact":max(x["maximum_gauge_dirichlet_endpoint"] for x in manufactured["records"])<1e-14,
    "finite_growth_converges":growth["records"][-1]["relative_growth_rate_error"]<2e-3 and min(growth["growth_error_rates"])>1.7,
    "stable_energy_conserved":energy["maximum_relative_energy_drift"]<1e-7,
    "constraint_model_second_order":min(constraints["error_rates"])>1.8,
    "constraint_boundary_pattern":max(x["normal_dirichlet_endpoint_maximum"] for x in constraints["records"])<1e-14,
}
payload={
    "status":"pass" if all(passes.values()) else "review",
    "scope":"executable frozen one-normal-dimensional principal reduced-wave IBVP; not nonlinear Einstein evolution",
    "field_count":17,
    "field_decomposition":{"gauge_dirichlet":4,"coupled_israel_scalar_robin":13},
    "production_background":{"wall_stiffness":gamma,"lower_c":lower["c"],"upper_c":upper["c"]},
    "wall_symmetry_defects":{"lower":lower["symmetry_defect"],"upper":upper["symmetry_defect"]},
    "manufactured_full_system":manufactured,
    "mirrored_upper_wall_finite_growth":growth,
    "mirrored_lower_wall_energy":energy,
    "frozen_constraint_propagation":constraints,
    "acceptance":passes,
    "limitations":[
        "flat frozen bulk principal part with one resolved normal dimension",
        "mirrored single-wall tests isolate each wall symmetrizer; the physical two-wall manufactured test uses both actual matrices",
        "does not evolve variable coefficients, nonlinear generalized-harmonic sources, Hamiltonian/momentum constraints, or horizons",
        "does not yet ingest the corrected two-dimensional fold as a nonlinear spacetime state",
    ],
}
Path("results/frozen_reduced_wave_evolution_audit.json").write_text(
    json.dumps(payload,indent=2,sort_keys=True)+"\n"
)
print(json.dumps(payload,indent=2))
