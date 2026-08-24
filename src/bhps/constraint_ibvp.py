"""Boundary and regular generalized-harmonic constraint diagnostics.

The project uses ``C_a=Gamma_a-H_a``.  In the SO(3)-invariant sector the
five-dimensional covector has three regular coefficient fields,

``C_0``, ``C_z``, and ``c_r=C_x/x=C_r/r``.

The last quantity is evaluated as a regular vector coefficient rather than by
storing the vanishing Cartesian component at the symmetry axis.
"""

from __future__ import annotations

import numpy as np

from bhps.linearized_gh_einstein_scalar import metric_geometry_from_jets


REGULAR_SO3_CONSTRAINT_ORDER=("C_0","C_z","c_r=C_r/r")


def _pack_regular_so3_metric_tensor(tensor,radius):
    """Pack a symmetric Cartesian tensor into the seven regular metric fields."""
    value=np.asarray(tensor,dtype=float);radius=float(radius)
    if value.shape!=(5,5) or radius<=0:
        raise ValueError("invalid regular metric tensor")
    transverse=.5*(value[3,3]+value[4,4])
    return np.array((
        value[1,0],value[1,2]/radius,value[0,0],transverse,
        (value[2,2]-transverse)/radius**2,value[0,2]/radius,value[1,1],
    ))


def regular_so3_radial_metric_derivative_matrices(radius):
    """Map regular values/derivatives to packed Cartesian ``partial_r h``.

    The value map contains derivatives of the radius-dependent tensor basis;
    it is essential for the regular vector and anisotropy coefficients.
    """
    from bhps.regular_so3_gh_reduction import regular_so3_perturbation_jets

    radius=float(radius)
    if radius<=0:raise ValueError("radial metric derivative map requires r>0")
    value=np.zeros((7,7));derivative=np.zeros((7,7))
    for column in range(7):
        basis=np.zeros(9);basis[column]=1.
        tensor=regular_so3_perturbation_jets(radius,basis)["metric_first"][2]
        value[:,column]=_pack_regular_so3_metric_tensor(tensor,radius)
        first=np.zeros((3,9));first[2,column]=1.
        tensor=regular_so3_perturbation_jets(
            radius,np.zeros(9),first,
        )["metric_first"][2]
        derivative[:,column]=_pack_regular_so3_metric_tensor(tensor,radius)
    return {"value_matrix":value,"derivative_matrix":derivative}


def gh_metric_characteristic_projectors(
    metric,time_normal_covector,boundary_normal_covector,
):
    """Return the gauge, constraint, and physical ``u^(1-)`` projectors.

    This is the dimension-general form of Lindblom--Szilagyi Eqs. (E12)--
    (E14).  Their coefficient ``1/2`` is ``1/(D-2)`` here because the
    transverse boundary section has dimension ``D-2``; it is ``1/3`` in the
    present five-dimensional spacetime.
    """
    g=np.asarray(metric,dtype=float);t=np.asarray(time_normal_covector,dtype=float)
    n=np.asarray(boundary_normal_covector,dtype=float);dimension=len(t)
    if g.shape!=(dimension,dimension) or n.shape!=(dimension,) or dimension<3:
        raise ValueError("invalid metric characteristic geometry")
    inverse=np.linalg.inv(g)
    t_norm=float(t@inverse@t);n_norm=float(n@inverse@n);orthogonal=float(t@inverse@n)
    if not np.isclose(t_norm,-1.,atol=1e-11) or not np.isclose(n_norm,1.,atol=1e-11) or abs(orthogonal)>1e-11:
        raise ValueError("characteristic normals must be orthonormal")
    k=(t-n)/np.sqrt(2);ell=(t+n)/np.sqrt(2)
    k_upper=inverse@k;ell_upper=inverse@ell
    transverse=g+np.outer(t,t)-np.outer(n,n)
    transverse_mixed=transverse@inverse;transverse_upper=inverse@transverse@inverse
    delta=np.eye(dimension);cross_dimension=dimension-2
    gauge=np.zeros((dimension,dimension,dimension,dimension))
    constraint=np.zeros_like(gauge);physical=np.zeros_like(gauge)
    for a,b,c,d in np.ndindex((dimension,)*4):
        gauge[a,b,c,d]=-(
            k[a]*k[b]*ell_upper[c]*ell_upper[d]
            +.5*k[a]*(delta[b,c]*ell_upper[d]+delta[b,d]*ell_upper[c])
            +.5*k[b]*(delta[a,c]*ell_upper[d]+delta[a,d]*ell_upper[c])
        )
        constraint[a,b,c,d]=(
            transverse[a,b]*transverse_upper[c,d]/cross_dimension
            -.5*(
                ell[a]*transverse_mixed[b,c]*k_upper[d]
                +ell[b]*transverse_mixed[a,c]*k_upper[d]
                +ell[a]*transverse_mixed[b,d]*k_upper[c]
                +ell[b]*transverse_mixed[a,d]*k_upper[c]
            )+ell[a]*ell[b]*k_upper[c]*k_upper[d]
        )
        physical[a,b,c,d]=(
            .5*(
                transverse_mixed[a,c]*transverse_mixed[b,d]
                +transverse_mixed[a,d]*transverse_mixed[b,c]
            )-transverse[a,b]*transverse_upper[c,d]/cross_dimension
        )
    identity=np.zeros_like(gauge)
    for a,b,c,d in np.ndindex((dimension,)*4):
        identity[a,b,c,d]=.5*(delta[a,c]*delta[b,d]+delta[a,d]*delta[b,c])
    projectors={"gauge":gauge,"constraint":constraint,"physical":physical}
    idempotence=max(float(np.max(np.abs(
        np.einsum("abef,efcd->abcd",value,value)-value
    ))) for value in projectors.values())
    orthogonality=max(float(np.max(np.abs(
        np.einsum("abef,efcd->abcd",left,right)
    ))) for left_name,left in projectors.items() for right_name,right in projectors.items() if left_name!=right_name)
    return {
        **projectors,"identity":identity,"time_normal_covector":t,
        "boundary_normal_covector":n,"ingoing_null_covector":k,
        "outgoing_null_covector":ell,
        "transverse_metric":transverse,
        "completeness_defect":float(np.max(np.abs(gauge+constraint+physical-identity))),
        "idempotence_defect":idempotence,"orthogonality_defect":orthogonality,
        "full_ranks":{
            name:int(round(np.einsum("abab",value)))
            for name,value in projectors.items()
        },
    }


def gh_incoming_metric_constraint_extraction(
    metric,time_normal_covector,boundary_normal_covector,incoming_normal_derivative,
):
    """Return the principal incoming constraint ``c^(0-)_a`` (E26)."""
    geometry=gh_metric_characteristic_projectors(
        metric,time_normal_covector,boundary_normal_covector,
    )
    inverse=np.linalg.inv(np.asarray(metric,dtype=float));k=geometry["ingoing_null_covector"]
    k_upper=inverse@k;x=np.asarray(incoming_normal_derivative,dtype=float)
    dimension=len(k)
    if x.shape!=(dimension,dimension):raise ValueError("incoming tensor has the wrong shape")
    result=np.empty(dimension)
    for a in range(dimension):
        result[a]=np.sqrt(2)*(
            np.einsum("c,d,cd->",k_upper,np.eye(dimension)[a],x)
            -.5*k[a]*np.einsum("cd,cd->",inverse,x)
        )
    return result


def gh_incoming_metric_constraint_lift(
    metric,time_normal_covector,boundary_normal_covector,constraint_covector,
    lapse,normal_shift=0.,
):
    """Return the E27 correction to ``P(C) partial_t u^(1-)``.

    The returned tensor extracts to ``-(N+n_i N^i)c^(0-)_a`` and has no
    gauge or physical projection.  Thus it removes rather than injects the
    incoming constraint at principal order.
    """
    geometry=gh_metric_characteristic_projectors(
        metric,time_normal_covector,boundary_normal_covector,
    )
    g=np.asarray(metric,dtype=float);inverse=np.linalg.inv(g)
    ell=geometry["outgoing_null_covector"];k=geometry["ingoing_null_covector"]
    ell_upper=inverse@ell;k_upper=inverse@k
    transverse=geometry["transverse_metric"];transverse_mixed=transverse@inverse
    c=np.asarray(constraint_covector,dtype=float);dimension=len(c);cross_dimension=dimension-2
    lapse=float(lapse);normal_shift=float(normal_shift)
    if c.shape!=(dimension,) or lapse<=0 or lapse+normal_shift<=0:
        raise ValueError("invalid incoming-constraint lift data")
    lift=np.zeros((dimension,dimension,dimension))
    for a,b,d in np.ndindex(dimension,dimension,dimension):
        lift[a,b,d]=(
            .5*(ell[a]*transverse_mixed[b,d]+ell[b]*transverse_mixed[a,d])
            -transverse[a,b]*ell_upper[d]/cross_dimension
            -.5*ell[a]*ell[b]*k_upper[d]
        )
    return np.sqrt(2)*(lapse+normal_shift)*np.einsum("abd,d->ab",lift,c)


def regular_so3_metric_characteristic_projector_matrices(
    background_metric,radius,boundary_direction,outward_sign=1.,
):
    """Restrict the five-dimensional characteristic projectors to 7 fields."""
    from bhps.regular_so3_gh_reduction import regular_so3_perturbation_jets

    metric=np.asarray(background_metric,dtype=float);radius=float(radius)
    direction=str(boundary_direction);sign=float(outward_sign)
    if metric.shape!=(5,5) or radius<=0 or direction not in ("compact","radial") or sign not in (-1.,1.):
        raise ValueError("invalid regular projector geometry")
    inverse=np.linalg.inv(metric)
    if inverse[0,0]>=0 or np.max(np.abs(metric[0,1:]))>1e-12:
        raise ValueError("regular projector currently requires a static zero-shift background")
    lapse=1/np.sqrt(-inverse[0,0]);t=np.array((-lapse,0.,0.,0.,0.))
    coordinate=1 if direction=="compact" else 2
    n=np.zeros(5);n[coordinate]=sign/np.sqrt(inverse[coordinate,coordinate])
    full=gh_metric_characteristic_projectors(metric,t,n)

    matrices={}
    for name in ("gauge","constraint","physical"):
        matrix=np.empty((7,7))
        for column in range(7):
            values=np.zeros(9);values[column]=1.
            tensor=regular_so3_perturbation_jets(radius,values)["metric"]
            projected=np.einsum("abcd,cd->ab",full[name],tensor)
            matrix[:,column]=_pack_regular_so3_metric_tensor(projected,radius)
        matrices[name]=matrix
    identity=np.eye(7)
    return {
        **matrices,"lapse":float(lapse),"time_normal_covector":t,
        "boundary_normal_covector":n,
        "ranks":{name:int(np.linalg.matrix_rank(value,tol=1e-10)) for name,value in matrices.items()},
        "completeness_defect":float(np.max(np.abs(sum(matrices.values())-identity))),
        "idempotence_defect":max(float(np.max(np.abs(value@value-value))) for value in matrices.values()),
        "orthogonality_defect":max(float(np.max(np.abs(left@right))) for left_name,left in matrices.items() for right_name,right in matrices.items() if left_name!=right_name),
        "full_projector_diagnostics":{
            key:full[key] for key in (
                "completeness_defect","idempotence_defect","orthogonality_defect","full_ranks",
            )
        },
    }


def regular_so3_sommerfeld_energy_audit(
    background_metric,radius,boundary_direction="radial",outward_sign=1.,
    gauge_rate=1.,constraint_rate=1.,physical_rate=1.,
):
    """Audit the Euclidean wave-energy sign of projected Sommerfeld data.

    The reduced second-order runtime uses the same positive principal weight
    for each of its seven metric coefficients.  Its outer energy flux is
    therefore controlled by the symmetric part of
    ``mu_g P_g + mu_c P_c + mu_p P_p``.  Equal rates collapse to a multiple
    of the identity by projector completeness, independently of whether the
    individual reduced projectors are Euclidean self-adjoint.
    """
    rates={
        "gauge":float(gauge_rate),"constraint":float(constraint_rate),
        "physical":float(physical_rate),
    }
    if any(value<=0 or not np.isfinite(value) for value in rates.values()):
        raise ValueError("Sommerfeld rates must be finite and positive")
    projectors=regular_so3_metric_characteristic_projector_matrices(
        background_metric,radius,boundary_direction,outward_sign,
    )
    operator=sum(rates[name]*projectors[name] for name in rates)
    symmetric=.5*(operator+operator.T)
    eigenvalues=np.linalg.eigvalsh(symmetric)
    equal=max(rates.values())-min(rates.values())<=1e-14*max(rates.values())
    expected=rates["gauge"]*np.eye(7) if equal else None
    return {
        "rates":rates,"operator":operator,"symmetric_part":symmetric,
        "symmetric_eigenvalues":eigenvalues,
        "minimum_boundary_dissipation_eigenvalue":float(eigenvalues[0]),
        "maximum_boundary_dissipation_eigenvalue":float(eigenvalues[-1]),
        "strictly_energy_dissipative":bool(eigenvalues[0]>1e-12),
        "equal_rate_identity_defect":None if expected is None else float(
            np.max(np.abs(operator-expected))
        ),
        "projector_completeness_defect":projectors["completeness_defect"],
    }


def frozen_regular_so3_sommerfeld_symbol(
    laplace_real,laplace_imag,tangential_wavenumber,wave_speed=1.,
    gauge_rate=1.,constraint_rate=1.,physical_rate=1.,
):
    """Frozen half-space symbol for projected regular Sommerfeld data.

    For an inward-decaying mode and outward normal, each sector determinant
    is ``s + mu c lambda`` with
    ``lambda=sqrt((s/c)^2+|k|^2)`` on the positive-real branch.  A zero with
    ``Re(s)>0`` would be a boundary instability.
    """
    sigma=complex(float(laplace_real),float(laplace_imag))
    wave=float(tangential_wavenumber);speed=float(wave_speed)
    rates={
        "gauge":float(gauge_rate),"constraint":float(constraint_rate),
        "physical":float(physical_rate),
    }
    if (
        sigma.real<=0 or wave<0 or speed<=0 or not np.isfinite(speed)
        or any(value<=0 or not np.isfinite(value) for value in rates.values())
    ):
        raise ValueError("invalid frozen Sommerfeld symbol data")
    decay=np.sqrt((sigma/speed)**2+wave**2)
    if decay.real<0 or (decay.real==0 and decay.imag<0):decay=-decay
    determinants={name:sigma+rate*speed*decay for name,rate in rates.items()}
    scale=max(1.,abs(sigma),speed*abs(decay))
    gaps={name:float(abs(value)/scale) for name,value in determinants.items()}
    return {
        "laplace_frequency":sigma,"decay_rate":decay,
        "sector_determinants":determinants,"normalized_sector_gaps":gaps,
        "minimum_normalized_gap":min(gaps.values()),
        "unstable_root":bool(min(gaps.values())<=1e-12),
        "multiplicities":{"gauge":3,"constraint":3,"physical":1},
    }


def regular_so3_incoming_metric_constraint_lift(
    background_metric,radius,boundary_direction,constraint_values,
    outward_sign=1.,normal_shift=0.,
):
    """Lift three regular constraints into seven ``partial_t u^(1-)`` fields.

    ``constraint_values`` are ``(C_0,C_z,C_r/r)`` at the regular Cartesian
    representative ``x=r,y=w=0``.  The output follows ``FIELD_ORDER[:7]`` and
    is the dimension-correct E27 constraint-sector correction for one face.
    """
    from bhps.regular_so3_gh_reduction import FIELD_ORDER

    metric=np.asarray(background_metric,dtype=float);radius=float(radius)
    direction=str(boundary_direction);sign=float(outward_sign)
    constraint=np.asarray(constraint_values,dtype=float)
    if (
        metric.shape!=(5,5) or radius<=0 or constraint.shape!=(3,)
        or direction not in ("compact","radial") or sign not in (-1.,1.)
    ):
        raise ValueError("invalid regular incoming-constraint lift data")
    inverse=np.linalg.inv(metric)
    if inverse[0,0]>=0 or np.max(np.abs(metric[0,1:]))>1e-12:
        raise ValueError("regular lift currently requires a static zero-shift background")
    lapse=1/np.sqrt(-inverse[0,0]);t=np.array((-lapse,0.,0.,0.,0.))
    coordinate=1 if direction=="compact" else 2
    n=np.zeros(5);n[coordinate]=sign/np.sqrt(inverse[coordinate,coordinate])
    full_constraint=np.array((constraint[0],constraint[1],radius*constraint[2],0.,0.))
    correction=gh_incoming_metric_constraint_lift(
        metric,t,n,full_constraint,lapse,normal_shift,
    )
    reduced=_pack_regular_so3_metric_tensor(correction,radius)
    return {
        "correction":reduced,"field_order":FIELD_ORDER[:7],"lapse":float(lapse),
        "time_normal_covector":t,"boundary_normal_covector":n,
        "full_constraint_covector":full_constraint,
    }


def regular_so3_source_covector(radius,source_values):
    """Expand regular source coefficients ``(H_0,H_z,h_r)`` at ``x=r``."""
    radius=float(radius);values=np.asarray(source_values,dtype=float)
    if radius<0 or values.shape!=(3,):
        raise ValueError("invalid regular SO(3) source data")
    return np.array((values[0],values[1],radius*values[2],0.,0.))


def linearized_gauge_constraint(
    background_metric,background_first,background_second,
    perturbation_metric,perturbation_first,perturbation_second,
    source_perturbation=None,complex_step=1e-30,
):
    """Return ``delta Gamma_a-delta H_a`` from complete metric jets."""
    metric=np.asarray(background_metric);dimension=metric.shape[0]
    source=np.zeros(dimension) if source_perturbation is None else np.asarray(source_perturbation)
    if source.shape!=(dimension,):
        raise ValueError("source perturbation has incompatible shape")
    step=float(complex_step)
    if step<=0:raise ValueError("complex_step must be positive")
    geometry=metric_geometry_from_jets(
        metric.astype(complex)+1j*step*np.asarray(perturbation_metric),
        np.asarray(background_first,dtype=complex)+1j*step*np.asarray(perturbation_first),
        np.asarray(background_second,dtype=complex)+1j*step*np.asarray(perturbation_second),
    )
    delta_gamma=np.imag(geometry["contracted_christoffel_covector"])/step
    return delta_gamma-source


def linearized_regular_so3_gauge_constraint(
    background,perturbation,radius,source_values=None,complex_step=1e-30,
):
    """Project the pointwise linearized GH constraint onto three regular rows."""
    radius=float(radius)
    if radius<=0:
        raise ValueError("extract at r>0; use the regular flat formula at the axis")
    source_values=np.zeros(3) if source_values is None else np.asarray(source_values,dtype=float)
    constraint=linearized_gauge_constraint(
        background["metric"],background["metric_first"],background["metric_second"],
        perturbation["metric"],perturbation["metric_first"],perturbation["metric_second"],
        regular_so3_source_covector(radius,source_values),complex_step,
    )
    return np.array((constraint[0],constraint[1],constraint[2]/radius))


def regular_so3_constraint_coefficient_matrices(background,radius):
    """Extract value and first-derivative maps into the three GH constraints.

    The metric inputs use the nine-field regular SO(3) order.  The source map
    is exactly minus the three-by-three identity in the matching regular
    covector basis.
    """
    from bhps.regular_so3_gh_reduction import regular_so3_perturbation_jets

    radius=float(radius)
    if radius<=0:raise ValueError("extract off axis and take a regular parity limit")
    zero=np.zeros((3,9));first=np.zeros((3,3,9))
    for column in range(9):
        values=np.zeros(9);values[column]=1.
        zero[:,column]=linearized_regular_so3_gauge_constraint(
            background,regular_so3_perturbation_jets(radius,values),radius,
        )
        for direction in range(3):
            reduced_first=np.zeros((3,9));reduced_first[direction,column]=1.
            first[direction,:,column]=linearized_regular_so3_gauge_constraint(
                background,regular_so3_perturbation_jets(
                    radius,np.zeros(9),first=reduced_first,
                ),radius,
            )
    return {
        "constraint_order":REGULAR_SO3_CONSTRAINT_ORDER,
        "zero_matrix":zero,"first_matrices":first,
        "source_matrix":-np.eye(3),
        "finite":bool(np.all(np.isfinite(zero)) and np.all(np.isfinite(first))),
    }


def evaluate_regular_so3_constraint_field(
    z,r,values,velocity,source,zero_matrices,first_matrices,stencil_width=5,
    radial_first_is_scaled=False,
):
    """Evaluate the three linearized GH constraint fields on a nodal grid."""
    from bhps.gw_slice_high_order_solver import derivative_matrix

    z=np.asarray(z,dtype=float);r=np.asarray(r,dtype=float)
    values=np.asarray(values,dtype=float);velocity=np.asarray(velocity,dtype=float)
    source=np.asarray(source,dtype=float);zero=np.asarray(zero_matrices,dtype=float)
    first=np.asarray(first_matrices,dtype=float);shape=(len(z),len(r))
    if (
        values.shape!=(*shape,9) or velocity.shape!=values.shape
        or source.shape!=(*shape,3) or zero.shape!=(*shape,3,9)
        or first.shape!=(3,*shape,3,9) or r[0]!=0
    ):
        raise ValueError("invalid gridded regular constraint data")
    width=min(int(stencil_width),len(z),len(r))
    if width<3:raise ValueError("constraint diagnostic grid is too short")
    dz=derivative_matrix(z,1,width);dr=derivative_matrix(r,1,width)
    drr=derivative_matrix(r,2,width) if radial_first_is_scaled else None
    z_first=np.empty_like(values);r_first=np.empty_like(values)
    r_second=np.empty_like(values) if radial_first_is_scaled else None
    for field in range(9):
        z_first[:,:,field]=dz@values[:,:,field]
        r_first[:,:,field]=values[:,:,field]@dr.T
        if radial_first_is_scaled:r_second[:,:,field]=values[:,:,field]@drr.T
    # Every stored reduced coefficient is even at the symmetry axis.
    r_first[:,0]=0.
    radial_term=np.empty((*shape,3))
    if radial_first_is_scaled:
        radial_term[:,1:]=np.einsum(
            "ijab,ijb->ija",first[2,:,1:]/r[None,1:,None,None],r_first[:,1:],
        )
        radial_term[:,0]=np.einsum("iab,ib->ia",first[2,:,0],r_second[:,0])
    else:radial_term=np.einsum("ijab,ijb->ija",first[2],r_first)
    constraint=(
        np.einsum("ijab,ijb->ija",zero,values)
        +np.einsum("ijab,ijb->ija",first[0],velocity)
        +np.einsum("ijab,ijb->ija",first[1],z_first)
        +radial_term-source
    )
    return {
        "constraint":constraint,"z_first":z_first,"r_first":r_first,
        "r_second":r_second,
        "finite":bool(np.all(np.isfinite(constraint))),
    }


class RegularSO3ConstraintBoundaryFeedback:
    """Evaluate ``C_a=Gamma_a-H_a`` for live source-boundary feedback."""

    def __init__(
        self,z,r,zero_matrices,first_matrices,stencil_width=5,
        radial_first_is_scaled=False,
    ):
        self.z=np.asarray(z,dtype=float);self.r=np.asarray(r,dtype=float)
        self.zero=np.asarray(zero_matrices,dtype=float)
        self.first=np.asarray(first_matrices,dtype=float)
        self.stencil_width=int(stencil_width)
        self.radial_first_is_scaled=bool(radial_first_is_scaled)

    def evaluate(self,position,velocity,source,time=0.):
        del time
        return evaluate_regular_so3_constraint_field(
            self.z,self.r,position,velocity,source,self.zero,self.first,
            self.stencil_width,self.radial_first_is_scaled,
        )["constraint"]


class RegularSO3OuterMetricConstraintFeedback:
    """Map live regular constraints to outer-face ``partial_t u^(1-)`` data.

    The artificial radial face receives the three constraint-sector
    characteristics.  Compact-wall corner nodes are excluded by default so
    that their complete Israel/gauge boundary system keeps precedence.
    """

    def __init__(self,constraint_feedback,background_metrics,include_corners=False):
        self.constraint_feedback=constraint_feedback
        self.z=np.asarray(constraint_feedback.z,dtype=float)
        self.r=np.asarray(constraint_feedback.r,dtype=float)
        self.background_metrics=np.asarray(background_metrics,dtype=float)
        self.include_corners=bool(include_corners)
        if (
            self.r.ndim!=1 or len(self.r)<2 or self.r[-1]<=0
            or self.background_metrics.shape!=(len(self.z),5,5)
        ):
            raise ValueError("invalid outer metric-constraint feedback data")
        self.lift_matrices=np.zeros((len(self.z),7,3));self.lapses=np.zeros(len(self.z))
        for i in range(1 if not self.include_corners else 0,len(self.z)-(1 if not self.include_corners else 0)):
            for column in range(3):
                basis=np.zeros(3);basis[column]=1.
                lifted=regular_so3_incoming_metric_constraint_lift(
                    self.background_metrics[i],self.r[-1],"radial",basis,1.,
                )
                self.lift_matrices[i,:,column]=lifted["correction"]
                self.lapses[i]=lifted["lapse"]

    def evaluate(self,position,velocity,source,time=0.):
        constraint=np.asarray(self.constraint_feedback.evaluate(
            position,velocity,source,time,
        ),dtype=float)
        if constraint.shape!=(len(self.z),len(self.r),3):
            raise ValueError("live constraint field has the wrong shape")
        correction=np.zeros((len(self.z),len(self.r),9))
        mask=np.zeros((len(self.z),len(self.r)),dtype=bool)
        indices=range(len(self.z)) if self.include_corners else range(1,len(self.z)-1)
        for i in indices:
            correction[i,-1,:7]=self.lift_matrices[i]@constraint[i,-1]
            mask[i,-1]=True
        return {
            "characteristic_correction":correction,"incoming_mask":mask,
            "constraint":constraint,"lapse":self.lapses,
            "corner_policy":"outer_excluded_wall_precedence" if not self.include_corners else "outer_included",
        }


class RegularSO3OuterMetricCharacteristicFeedback:
    """Complete regular ``3+3+1`` outer metric characteristic feedback.

    Constraint-sector data use the live E27 lift.  The complementary gauge
    and physical incoming characteristics are relaxed toward zero.  This is
    a linear, static-zero-shift outer closure, not a nonlinear radiation
    boundary condition.
    """

    def __init__(
        self,constraint_feedback,background_metrics,gauge_rate=1.,physical_rate=1.,
        gamma2=0.,stencil_width=5,include_corners=False,
    ):
        from bhps.gw_slice_high_order_solver import derivative_matrix

        self.constraint=RegularSO3OuterMetricConstraintFeedback(
            constraint_feedback,background_metrics,include_corners,
        )
        self.z=self.constraint.z;self.r=self.constraint.r
        self.background_metrics=self.constraint.background_metrics
        self.gauge_rate=float(gauge_rate);self.physical_rate=float(physical_rate)
        self.gamma2=float(gamma2);self.stencil_width=int(stencil_width)
        if self.gauge_rate<=0 or self.physical_rate<=0 or self.gamma2<0:
            raise ValueError("invalid outer characteristic feedback rates")
        width=min(self.stencil_width,len(self.r))
        if width<3:raise ValueError("outer characteristic stencil is too short")
        derivative_row=derivative_matrix(self.r,1,width)[-1]
        if hasattr(derivative_row,"toarray"):derivative_row=derivative_row.toarray()
        self.radial_derivative_row=np.asarray(derivative_row,dtype=float).ravel()
        radial_map=regular_so3_radial_metric_derivative_matrices(self.r[-1])
        self.radial_value_matrix=radial_map["value_matrix"]
        self.radial_first_matrix=radial_map["derivative_matrix"]
        self.projectors=[];self.normal_upper_r=np.zeros(len(self.z))
        for i,metric in enumerate(self.background_metrics):
            result=regular_so3_metric_characteristic_projector_matrices(
                metric,self.r[-1],"radial",1.,
            )
            self.projectors.append(result)
            inverse=np.linalg.inv(metric)
            self.normal_upper_r[i]=np.sqrt(inverse[2,2])

    def evaluate(self,position,velocity,source,time=0.):
        q=np.asarray(position,dtype=float);v=np.asarray(velocity,dtype=float)
        expected=(len(self.z),len(self.r),9)
        if q.shape!=expected or v.shape!=expected:
            raise ValueError("outer characteristic wave state has the wrong shape")
        result=self.constraint.evaluate(q,v,source,time)
        correction=np.array(result["characteristic_correction"],copy=True)
        radial_first=np.einsum("j,zjf->zf",self.radial_derivative_row,q)
        incoming=np.zeros_like(correction);gauge=np.zeros_like(correction)
        physical=np.zeros_like(correction);mask=result["incoming_mask"]
        for i in np.flatnonzero(mask[:,-1]):
            # Packing is linear at a fixed nonzero radius, so the regular
            # characteristic is obtained directly from regular coefficient
            # fields.  This is algebraically identical to expanding three
            # full Cartesian tensors and packing their combination.
            incoming[i,-1,:7]=(
                -v[i,-1,:7]/result["lapse"][i]
                -self.normal_upper_r[i]*(
                    self.radial_value_matrix@q[i,-1,:7]
                    +self.radial_first_matrix@radial_first[i,:7]
                )
                -self.gamma2*q[i,-1,:7]
            )
            projector=self.projectors[i]
            gauge[i,-1,:7]=-self.gauge_rate*(
                projector["gauge"]@incoming[i,-1,:7]
            )
            physical[i,-1,:7]=-self.physical_rate*(
                projector["physical"]@incoming[i,-1,:7]
            )
            correction[i,-1]+=gauge[i,-1]+physical[i,-1]
        return {
            **result,"characteristic_correction":correction,
            "incoming_characteristic":incoming,"gauge_correction":gauge,
            "physical_correction":physical,"gauge_rate":self.gauge_rate,
            "physical_rate":self.physical_rate,"gamma2":self.gamma2,
            "closure":"constraint_E27_plus_zero_incoming_gauge_physical",
        }


class RegularSO3OuterMetricWeakFluxFeedback:
    """Mass-lift an outer characteristic correction as a weak face flux.

    The wrapped feedback supplies the seven regular metric-characteristic
    corrections.  For a static zero-shift background, the conversion from a
    covariant incoming correction ``B`` to the reduced radial flux is
    ``+N sqrt(w p_r) B`` with the outward radial orientation used here.  The
    sign is fixed by requiring the wrapped ``B=-mu u^-`` gauge/physical
    relaxation to be dissipative.  The Q1 wave system subsequently integrates this
    coefficient against the outer-face test functions and applies the
    inverse mass matrix.  This is a weak numerical candidate, not yet an
    analytic maximally dissipative or continuum well-posedness result.
    """

    def __init__(
        self,feedback,outer_surface_weight,penalty=1.,
        constraint_projectors=None,constraint_sommerfeld=False,
    ):
        self.feedback=feedback
        self.z=np.asarray(feedback.z,dtype=float);self.r=np.asarray(feedback.r,dtype=float)
        self.outer_surface_weight=np.asarray(outer_surface_weight,dtype=float)
        self.penalty=float(penalty)
        self.constraint_sommerfeld=bool(constraint_sommerfeld)
        self.constraint_projectors=constraint_projectors
        if (
            self.outer_surface_weight.shape!=(len(self.z),)
            or np.any(self.outer_surface_weight<=0) or not np.all(np.isfinite(self.outer_surface_weight))
            or self.penalty<=0 or not np.isfinite(self.penalty)
        ):
            raise ValueError("invalid weak outer-flux data")
        if self.constraint_sommerfeld:
            if self.constraint_projectors is None or len(self.constraint_projectors)!=len(self.z):
                raise ValueError("constraint Sommerfeld data require one projector per z node")
            for item in self.constraint_projectors:
                if np.asarray(item["constraint"]).shape!=(7,7):
                    raise ValueError("invalid constraint Sommerfeld projector")

    def evaluate(self,position,velocity,source,time=0.):
        result=self.feedback.evaluate(position,velocity,source,time)
        correction=np.asarray(result["characteristic_correction"],dtype=float)
        mask=np.asarray(result["incoming_mask"],dtype=bool)
        lapse=np.asarray(result["lapse"],dtype=float)
        expected=(len(self.z),len(self.r),9)
        if correction.shape!=expected or mask.shape!=expected[:2] or lapse.shape!=(len(self.z),):
            raise ValueError("wrapped outer feedback has incompatible shape")
        flux=(
            self.penalty*lapse[:,None]*self.outer_surface_weight[:,None]
            *correction[:,-1]
        )
        policy="E27_constraint_weak_flux"
        q=np.asarray(position,dtype=float);v=np.asarray(velocity,dtype=float)
        if self.constraint_sommerfeld:
            flux=np.zeros((len(self.z),9))
            for i in np.flatnonzero(mask[:,-1]):
                projector=self.constraint_projectors[i]["constraint"]
                flux[i,:7]=(
                    -self.penalty*self.outer_surface_weight[i]
                    *(projector@v[i,-1,:7])
                )
            policy="homogeneous_constraint_Sommerfeld_weak_flux"
        if "gauge_correction" in result and "physical_correction" in result:
            # E27 supplies the constraint-sector datum.  Gauge and physical
            # sectors instead use their energy-dissipative Sommerfeld flux;
            # converting their pointwise Bjorhus time derivatives as though
            # they were E27 data has the opposite energy sign.
            gauge=np.asarray(result["gauge_correction"],dtype=float)
            physical=np.asarray(result["physical_correction"],dtype=float)
            constraint_correction=correction-gauge-physical
            if not self.constraint_sommerfeld:
                flux=(
                    self.penalty*lapse[:,None]*self.outer_surface_weight[:,None]
                    *constraint_correction[:,-1]
                )
            for i in np.flatnonzero(mask[:,-1]):
                projector=self.feedback.projectors[i]
                complement=(
                    self.feedback.gauge_rate*projector["gauge"]@v[i,-1,:7]
                    +self.feedback.physical_rate*projector["physical"]@v[i,-1,:7]
                )
                if self.feedback.gamma2:
                    complement+=lapse[i]*self.feedback.gamma2*(
                        (projector["gauge"]+projector["physical"])@q[i,-1,:7]
                    )
                flux[i,:7]-=(
                    self.penalty*self.outer_surface_weight[i]*complement
                )
            policy=(
                "homogeneous_3+3+1_Sommerfeld_weak_flux"
                if self.constraint_sommerfeld else
                "E27_constraint_plus_Sommerfeld_gauge_physical_weak_flux"
            )
        flux[~mask[:,-1]]=0.
        return {
            **result,"weak_radial_flux":flux,
            "application":"q1_outer_face_weak_flux",
            "penalty":self.penalty,
            "outer_surface_weight":self.outer_surface_weight,
            "weak_flux_policy":policy,
        }


def flat_regular_so3_gauge_constraint(
    radius,values,first,second=None,source_values=None,
):
    """Independent closed form for the three regular flat-space constraints.

    ``values`` and ``first`` use the nine-field order from
    :mod:`bhps.regular_so3_gh_reduction` and derivative order ``(t,z,r)``.
    ``second`` is needed only at the axis to evaluate even radial quotients.
    """
    radius=float(radius);values=np.asarray(values,dtype=float);first=np.asarray(first,dtype=float)
    source=np.zeros(3) if source_values is None else np.asarray(source_values,dtype=float)
    if radius<0 or values.shape!=(9,) or first.shape!=(3,9) or source.shape!=(3,):
        raise ValueError("invalid regular SO(3) constraint jets")
    h_z0,v_z,h00,p,d,v_0,h_zz=values[:7]
    del h_z0,h00,p,h_zz  # Values enter only through the regular vector/tensor terms below.
    c0=(
        -.5*first[0,2]+first[1,0]+3*v_0+radius*first[2,5]
        -.5*first[0,6]-1.5*first[0,3]-.5*radius**2*first[0,4]
        -source[0]
    )
    cz=(
        -first[0,0]+.5*first[1,6]+.5*first[1,2]-1.5*first[1,3]
        -.5*radius**2*first[1,4]+3*v_z+radius*first[2,1]-source[1]
    )
    if radius>0:
        radial_quotient=(first[2,2]-first[2,6]-first[2,3])/radius
    else:
        if second is None:raise ValueError("axis evaluation requires second radial jets")
        second=np.asarray(second,dtype=float)
        if second.shape!=(3,3,9):raise ValueError("invalid second jets")
        radial_quotient=second[2,2,2]-second[2,2,6]-second[2,2,3]
    cr=(
        -first[0,5]+first[1,1]+.5*radial_quotient
        +.5*radius*first[2,4]+3*d-source[2]
    )
    return np.array((c0,cz,cr))


def frozen_constraint_mode_spectrum(
    wavenumber,damping_rate,ricci_mixed_eigenvalue=0.0,spatial_components=2,
):
    """Frozen ``rho=0`` subsidiary-system spectrum in an orthonormal frame.

    ``ricci_mixed_eigenvalue`` is the eigenvalue of ``R_a^b``.  It is zero in
    Minkowski and ``-4/ell**2`` in AdS5.  The returned SO(3) spectrum contains
    one temporal pair and ``spatial_components`` spatial pairs.  The formula
    is the five-dimensional counterpart of Gundlach et al. Eqs. (20)--(21).
    """
    wave=float(wavenumber);damping=float(damping_rate);ricci=float(ricci_mixed_eigenvalue)
    components=int(spatial_components)
    if wave<0 or damping<0 or components<1:
        raise ValueError("invalid frozen constraint-mode parameters")
    frequency_squared=wave**2-ricci
    temporal=np.roots((1.,2*damping,frequency_squared)).astype(complex)
    spatial=np.roots((1.,damping,frequency_squared)).astype(complex)
    spectrum=np.concatenate((temporal,*([spatial]*components)))
    tolerance=1e-12*max(1.,wave,damping,np.sqrt(abs(frequency_squared)))
    nonconstant_or_curved=frequency_squared>tolerance**2
    return {
        "temporal_roots":temporal,"spatial_roots":spatial,"spectrum":spectrum,
        "frequency_squared":float(frequency_squared),
        "maximum_real_part":float(np.max(spectrum.real)),
        "strictly_damped":bool(damping>0 and nonconstant_or_curved and np.max(spectrum.real)<-tolerance),
        "constant_mode_exception":bool(abs(frequency_squared)<=tolerance**2),
    }


def israel_wave_boundary_count(spacetime_dimension=5,scalar_fields=1):
    """Count reduced-wave boundary rows in a wall-adapted coordinate gauge.

    In ``D`` spacetime dimensions, the metric has ``D(D+1)/2`` reduced wave
    equations.  A timelike boundary has ``D-1`` mixed normal-tangent gauge
    rows, ``(D-1)D/2`` Israel rows, and one normal wave-coordinate row.
    """
    dimension=int(spacetime_dimension);scalars=int(scalar_fields)
    if dimension<3 or scalars<0:raise ValueError("invalid field count")
    metric_fields=dimension*(dimension+1)//2
    mixed_gauge=dimension-1
    israel=(dimension-1)*dimension//2
    normal_gauge=1
    metric_rows=mixed_gauge+israel+normal_gauge
    total_fields=metric_fields+scalars;total_rows=metric_rows+scalars
    return {
        "metric_wave_fields":metric_fields,"scalar_wave_fields":scalars,
        "mixed_normal_tangent_gauge_rows":mixed_gauge,
        "israel_rows":israel,"normal_wave_coordinate_rows":normal_gauge,
        "scalar_wall_rows":scalars,"metric_boundary_rows":metric_rows,
        "total_wave_fields":total_fields,"total_boundary_rows":total_rows,
        "count_closes":bool(total_fields==total_rows),
    }


def regular_so3_boundary_characteristic_count(scalar_fields=2):
    """Count independent regular boundary rows on physical and outer faces.

    The three outer constraint characteristics replace three incoming metric
    data; they are not additional equations at the already closed brane wall.
    """
    scalars=int(scalar_fields)
    if scalars<0:raise ValueError("scalar field count must be nonnegative")
    compact={
        "mixed_normal_tangent_gauge":2,"tangential_israel":4,
        "normal_wave_coordinate":1,
    }
    outer={"gauge_characteristics":3,"constraint_characteristics":3,"physical_characteristics":1}
    return {
        "regular_metric_fields":7,"scalar_fields":scalars,
        "compact_wall_metric_rows":compact,
        "compact_wall_total_rows":sum(compact.values())+scalars,
        "outer_radial_metric_rows":outer,
        "outer_radial_total_rows":sum(outer.values())+scalars,
        "constraint_rows_replace_incoming_data":True,
        "both_face_counts_close":bool(
            sum(compact.values())==7 and sum(outer.values())==7
        ),
    }


def frozen_constraint_boundary_symbol(laplace_real,laplace_imag,tangential_wavenumber):
    """Frozen symbols for normal-Dirichlet/tangential-Neumann gauge constraints."""
    sigma=complex(float(laplace_real),float(laplace_imag));wave=float(tangential_wavenumber)
    if sigma.real<=0 or wave<0:raise ValueError("require Re(s)>0 and k>=0")
    decay=np.sqrt(sigma*sigma+wave*wave)
    if decay.real<0 or (decay.real==0 and decay.imag<0):decay=-decay
    normal_dirichlet=1+0j;tangential_neumann=-decay
    scale=max(1.,abs(decay))
    return {
        "decay_rate":decay,"normal_constraint_determinant":normal_dirichlet,
        "tangential_constraint_determinant":tangential_neumann,
        "normalized_tangential_magnitude":float(abs(tangential_neumann)/scale),
        "unstable_normal_root":False,
        "unstable_tangential_root":bool(abs(tangential_neumann)<=1e-12*scale),
    }
