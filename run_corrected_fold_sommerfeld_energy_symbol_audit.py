#!/usr/bin/env python3
"""Energy and frozen-symbol audit of the weak homogeneous outer flux."""

import json,sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))

from bhps.constraint_ibvp import (
    frozen_regular_so3_sommerfeld_symbol,regular_so3_sommerfeld_energy_audit,
)
from run_corrected_fold_regular_so3_runtime import build_geometry


OUTPUT=Path("results/corrected_fold_sommerfeld_energy_symbol_audit.json")
RADII=(4.,6.)
PENALTIES=(.25,.5,1.,2.,4.)


def spline(field,geometry):
    return RectBivariateSpline(
        geometry["z"],geometry["r"],np.asarray(field),kx=3,ky=3,s=0,
    )


def actual_nodal_damping_audit(z,radius,surface):
    """Energy sign of the nodal-flux interpolation used by the runtime."""
    z=np.asarray(z,dtype=float);surface=np.asarray(surface,dtype=float)
    size=len(z);face=np.zeros((size,size));gauss=(-1/np.sqrt(3),1/np.sqrt(3))
    for i,spacing in enumerate(np.diff(z)):
        for xi in gauss:
            basis=np.array(((1-xi)/2,(1+xi)/2))
            face[i:i+2,i:i+2]+=spacing/2*radius**2*np.outer(basis,basis)
    active=np.arange(1,size-1)
    damping=(face@np.diag(surface))[np.ix_(active,active)]
    symmetric=.5*(damping+damping.T);values=np.linalg.eigvalsh(symmetric)
    exact=np.zeros((size,size))
    for i,spacing in enumerate(np.diff(z)):
        for xi in gauss:
            basis=np.array(((1-xi)/2,(1+xi)/2))
            local_surface=float(basis@surface[i:i+2])
            exact[i:i+2,i:i+2]+=spacing/2*radius**2*local_surface*np.outer(basis,basis)
    exact=exact[np.ix_(active,active)]
    return {
        "active_nodes":int(len(active)),
        "minimum_actual_symmetric_damping_eigenvalue":float(values[0]),
        "maximum_actual_symmetric_damping_eigenvalue":float(values[-1]),
        "actual_damping_condition_number":float(values[-1]/values[0]),
        "relative_difference_from_exact_weighted_face_matrix":float(
            np.linalg.norm(damping-exact)/max(np.linalg.norm(exact),1e-300)
        ),
        "strictly_dissipative_on_active_trace":bool(values[0]>0),
    }


def geometry_audit(name,geometry):
    z=np.linspace(geometry["z"][0],geometry["z"][-1],57)
    mass=spline(geometry["principal"]["mass_weight"],geometry)
    radial=spline(geometry["principal"]["r_gradient_weight"],geometry)
    records=[]
    for radius in RADII:
        surface=np.sqrt(
            mass.ev(z,np.full_like(z,radius))*radial.ev(z,np.full_like(z,radius))
        )
        nodal=actual_nodal_damping_audit(z,radius,surface)
        for z_value in np.geomspace(geometry["z"][0],geometry["z"][-1],7):
            metric=geometry["jet_field"].at(float(z_value),radius)["metric"]
            inverse=np.linalg.inv(metric);lapse=1/np.sqrt(-inverse[0,0])
            speed=lapse*np.sqrt(inverse[2,2])
            energy=regular_so3_sommerfeld_energy_audit(metric,radius)
            records.append({
                "fold":name,"radius":radius,"z":float(z_value),
                "coordinate_wave_speed":float(speed),
                "equal_rate_identity_defect":energy["equal_rate_identity_defect"],
                "projector_completeness_defect":energy["projector_completeness_defect"],
                "minimum_boundary_dissipation_eigenvalue":energy[
                    "minimum_boundary_dissipation_eigenvalue"
                ],
                "maximum_boundary_dissipation_eigenvalue":energy[
                    "maximum_boundary_dissipation_eigenvalue"
                ],
                "strictly_energy_dissipative":energy["strictly_energy_dissipative"],
                "nodal_trace":nodal,
            })
    return records


def symbol_scan(speeds):
    records=[];minimum=np.inf;unstable=0
    for speed in (min(speeds),float(np.median(speeds)),max(speeds)):
        for real in (1e-4,.01,.2,1.,10.):
            for imag in (-30.,-2.,0.,2.,30.):
                for wave in (0.,.1,1.,20.):
                    for penalty in PENALTIES:
                        result=frozen_regular_so3_sommerfeld_symbol(
                            real,imag,wave,speed,penalty,penalty,penalty,
                        )
                        gap=result["minimum_normalized_gap"];minimum=min(minimum,gap)
                        unstable+=int(result["unstable_root"])
                        records.append({
                            "speed":speed,"laplace_real":real,"laplace_imag":imag,
                            "tangential_wavenumber":wave,"penalty":penalty,
                            "normalized_gap":gap,"unstable_root":result["unstable_root"],
                        })
    return {
        "sample_count":len(records),"minimum_normalized_gap":float(minimum),
        "unstable_root_count":unstable,"records":records,
        "analytic_exclusion":"Re(s)>0 and positive rate imply Re(s+mu*c*lambda)>0 on the Re(lambda)>0 decay branch",
    }


def main():
    print("building corrected G5/G6 geometries",flush=True)
    records=[]
    for name in ("G5","G6"):
        records.extend(geometry_audit(name,build_geometry(name)))
    symbol=symbol_scan([item["coordinate_wave_speed"] for item in records])
    acceptance={
        "all_equal_rate_operators_are_identity_below_1e_12":bool(max(
            item["equal_rate_identity_defect"] for item in records
        )<1e-12),
        "all_projector_completeness_defects_below_1e_12":bool(max(
            item["projector_completeness_defect"] for item in records
        )<1e-12),
        "all_continuum_boundary_forms_strictly_dissipative":bool(all(
            item["strictly_energy_dissipative"] for item in records
        )),
        "all_actual_nodal_trace_forms_strictly_dissipative":bool(all(
            item["nodal_trace"]["strictly_dissipative_on_active_trace"]
            for item in records
        )),
        "frozen_symbol_scan_has_no_growing_root":bool(symbol["unstable_root_count"]==0),
        "frozen_symbol_minimum_normalized_gap_above_1e_6":bool(
            symbol["minimum_normalized_gap"]>1e-6
        ),
    }
    payload={
        "status":"pass" if all(acceptance.values()) else "review",
        "scope":"continuum boundary-energy sign, actual nodal trace form, and frozen half-space symbol for the complete homogeneous regular 3+3+1 Sommerfeld flux",
        "radii":list(RADII),"penalties":list(PENALTIES),
        "interpretation":"at equal rates the complete 3+3+1 projector sum is exactly the identity, so the passing candidate is a componentwise Sommerfeld absorber rather than an E27 constraint-preserving condition",
        "geometry_records":records,"symbol_scan":symbol,"acceptance":acceptance,
        "limitations":[
            "frozen static zero-shift principal symbol",
            "the symbol scan supports an analytic positive-real-part exclusion but is not a variable-coefficient Kreiss estimate",
            "lower-order GH, driver, and compact-wall couplings are absent from the half-space symbol",
            "nonlinear stability remains open",
        ],
    }
    OUTPUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "status":payload["status"],"geometry_samples":len(records),
        "maximum_identity_defect":max(item["equal_rate_identity_defect"] for item in records),
        "minimum_nodal_damping_eigenvalue":min(
            item["nodal_trace"]["minimum_actual_symmetric_damping_eigenvalue"]
            for item in records
        ),
        "minimum_symbol_gap":symbol["minimum_normalized_gap"],
        "acceptance":acceptance,
    },indent=2),flush=True)


if __name__=="__main__":main()
