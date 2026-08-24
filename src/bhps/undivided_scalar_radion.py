"""Regular mixed scalar--radion equations on a warped interval.

The usual gauge-invariant master equation eliminates the scalar fluctuation
``f`` using ``f proportional to g_z/Phi_z``.  That representation is useful
only while ``Phi_z`` has fixed nonzero sign.  Here the constraint is retained
as an equation and no coefficient is divided by ``Phi_z``.

For conformal-coordinate backgrounds

    ds^2 = psi(z)^2 (eta_ab dx^a dx^b + dz^2),

the separated mode with eigenvalue ``mu_squared`` obeys

    g_z - K psi^2 Phi_z f = 0,

    K psi^2 [Phi_z f_z
      - (Phi_zz - Phi_z psi_z/psi) f]
      - K Phi_z^2 g/2 + mu_squared g = 0,

where ``K=4 kappa5_squared/3``.  This pair is algebraically equivalent to the
divided master equation wherever ``Phi_z != 0`` and has a finite continuous
extension through a simple turning point.  At the turn it is a
differential--algebraic system, not an ordinary first-order system.

The staggered eigensolver below is intentionally a control solver.  It places
``g`` at grid nodes and ``f`` at cell centers, uses centered second-order
differences, and retains the singular mass matrix of the
differential--algebraic problem.  Staggering removes the checkerboard branch
of a colocated first-derivative discretization.  Finite eigenvalues are
selected from the generalized spectrum; grid convergence and agreement with
the divided master solver are required before any physical interpretation.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eig
from scipy.integrate import simpson,solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar


def _wall_pair(wall_stiffness):
    values=np.asarray(wall_stiffness,dtype=float)
    if values.ndim==0:
        values=np.repeat(values,2)
    if values.shape!=(2,) or np.any(values<=0):
        raise ValueError("wall_stiffness must be positive or a positive pair")
    return values


def undivided_scalar_radion_kinetic_norm(z,psi,g,f,kappa5_squared=1.0):
    """Return the positive regular scalar-sector four-dimensional norm.

    Substitution of the undivided constraint into the second-variation scalar
    kinetic form gives, in conformal coordinates,

    ``integral dz [(3/(2 psi)) |g|^2 + 4 kappa5_squared psi^3 |f|^2]``.

    Unlike the divided master norm, this expression remains finite when the
    background stabilizer turns.
    """
    z=np.asarray(z,dtype=float);psi=np.asarray(psi,dtype=float)
    g=np.asarray(g);f=np.asarray(f);kappa5_squared=float(kappa5_squared)
    if (
        z.ndim!=1 or any(array.shape!=z.shape for array in (psi,g,f))
        or np.any(np.diff(z)<=0) or np.any(psi<=0) or kappa5_squared<=0
    ):
        raise ValueError("invalid kinetic-norm inputs")
    metric=float(simpson(1.5*np.abs(g)**2/psi,x=z))
    stabilizer=float(simpson(4*kappa5_squared*psi**3*np.abs(f)**2,x=z))
    return {
        "metric_contribution":metric,
        "stabilizer_contribution":stabilizer,
        "total":metric+stabilizer,
        "positive":bool(metric+stabilizer>0),
        "density":"3 |g|^2/(2 psi) + 4 kappa5_squared psi^3 |f|^2",
    }


def _three_point_weights(nodes,evaluation):
    """First-derivative Lagrange weights for three distinct nodes."""
    nodes=np.asarray(nodes,dtype=float);evaluation=float(evaluation)
    weights=np.empty(3)
    for j in range(3):
        others=[index for index in range(3) if index!=j]
        denominator=(nodes[j]-nodes[others[0]])*(nodes[j]-nodes[others[1]])
        weights[j]=(2*evaluation-nodes[others[0]]-nodes[others[1]])/denominator
    return weights


def second_order_derivative_matrix(z):
    """Return a three-point first-derivative matrix on a monotone grid."""
    z=np.asarray(z,dtype=float)
    if z.ndim!=1 or len(z)<3 or np.any(np.diff(z)<=0):
        raise ValueError("z must be a strictly increasing one-dimensional grid")
    size=len(z);derivative=np.zeros((size,size))
    derivative[0,:3]=_three_point_weights(z[:3],z[0])
    derivative[-1,-3:]=_three_point_weights(z[-3:],z[-1])
    for i in range(1,size-1):
        derivative[i,i-1:i+2]=_three_point_weights(z[i-1:i+2],z[i])
    return derivative


def chebyshev_lobatto_grid_and_derivative(lower,upper,size):
    """Return an increasing Lobatto grid and its spectral derivative."""
    lower=float(lower);upper=float(upper);size=int(size)
    if size<5 or not lower<upper:
        raise ValueError("require size >= 5 and lower < upper")
    index=np.arange(size)
    coordinate=np.cos(np.pi*index/(size-1))
    factors=np.ones(size);factors[[0,-1]]=2
    factors*=(-1.)**index
    differences=coordinate[:,None]-coordinate[None,:]
    derivative=(factors[:,None]/factors[None,:])/(differences+np.eye(size))
    derivative-=np.diag(np.sum(derivative,axis=1))
    z=.5*(lower+upper)+.5*(upper-lower)*coordinate
    derivative*=2/(upper-lower)
    return z[::-1],derivative[::-1,::-1]


def undivided_scalar_radion_spectral_matrices(
    z,
    psi,
    psi_z,
    phi,
    phi_z,
    mass_squared,
    wall_stiffness,
    kappa5_squared=1.0,
):
    """Independent Lobatto collocation matrices for the regular mixed pair.

    ``z`` must be the increasing grid returned by
    :func:`chebyshev_lobatto_grid_and_derivative`.  This independent dense
    derivative is used to test the staggered finite-difference result, not as
    its production replacement.
    """
    z=np.asarray(z,dtype=float);psi=np.asarray(psi,dtype=float)
    psi_z=np.asarray(psi_z,dtype=float);phi=np.asarray(phi,dtype=float)
    phi_z=np.asarray(phi_z,dtype=float);mass_squared=float(mass_squared)
    walls=_wall_pair(wall_stiffness);kappa5_squared=float(kappa5_squared)
    size=len(z)
    reference_z,derivative=chebyshev_lobatto_grid_and_derivative(z[0],z[-1],size)
    if (
        any(array.shape!=(size,) for array in (psi,psi_z,phi,phi_z))
        or not np.allclose(z,reference_z,rtol=0,atol=2e-13*max(1.,abs(z[-1])))
        or np.any(psi<=0) or mass_squared<0 or kappa5_squared<=0
    ):
        raise ValueError("inputs must lie on an increasing Chebyshev--Lobatto grid")
    phi_zz=-3*psi_z*phi_z/psi+psi**2*mass_squared*phi
    proper_phi_y=phi_z/psi
    proper_phi_yy=phi_zz/psi**2-phi_z*psi_z/psi**3
    coupling=4*kappa5_squared/3
    matrix=np.zeros((2*size,2*size));mass=np.zeros_like(matrix)
    matrix[:size,:size]=derivative
    matrix[:size,size:]=-np.diag(coupling*psi**2*phi_z)
    for i in range(1,size-1):
        row=size+i
        matrix[row,size:]+=coupling*psi[i]**2*phi_z[i]*derivative[i]
        matrix[row,size+i]-=coupling*psi[i]**2*(phi_zz[i]-phi_z[i]*psi_z[i]/psi[i])
        matrix[row,i]-=.5*coupling*phi_z[i]**2
        mass[row,i]=-1.
    lower=coupling*psi[0]**4*(walls[0]*proper_phi_y[0]/2-proper_phi_yy[0])
    upper=coupling*psi[-1]**4*(walls[1]*proper_phi_y[-1]/2+proper_phi_yy[-1])
    matrix[size,size]=-lower;mass[size,0]=1.
    matrix[-1,-1]=upper;mass[-1,size-1]=1.
    return {"matrix":matrix,"mass_matrix":mass,"derivative_matrix":derivative}


def _finite_real_spectrum(matrix,mass,imaginary_tolerance):
    values,vectors=eig(matrix,mass,right=True)
    finite=np.isfinite(values)
    real=np.abs(values.imag)<=float(imaginary_tolerance)*(1+np.abs(values.real))
    selected=np.flatnonzero(finite & real)
    ordered=selected[np.argsort(values.real[selected])]
    near_zero=selected[np.argsort(np.abs(values.real[selected]))]
    return values,vectors,finite,ordered,near_zero


def undivided_scalar_radion_spectral_control(
    z,
    psi,
    psi_z,
    phi,
    phi_z,
    mass_squared,
    wall_stiffness,
    count=8,
    kappa5_squared=1.0,
    imaginary_tolerance=1e-7,
):
    """Return the finite near-zero modes of the Lobatto control."""
    assembled=undivided_scalar_radion_spectral_matrices(
        z,psi,psi_z,phi,phi_z,mass_squared,wall_stiffness,kappa5_squared,
    )
    values,vectors,finite,ordered,near_zero=_finite_real_spectrum(
        assembled["matrix"],assembled["mass_matrix"],imaginary_tolerance,
    )
    number=min(max(1,int(count)),len(near_zero));indices=near_zero[:number]
    if number==0:raise RuntimeError("no finite real generalized eigenvalues found")
    return {
        **assembled,
        "mu_squared":values.real[indices],
        "eigenvectors":vectors[:,indices],
        "closest_to_zero_mu_squared":float(values.real[indices[0]]),
        "algebraic_minimum_finite_real_mu_squared":float(np.min(values.real[ordered])),
        "finite_eigenvalue_count":int(np.count_nonzero(finite)),
        "finite_real_eigenvalue_count":int(len(ordered)),
        "negative_finite_real_eigenvalue_count":int(np.count_nonzero(values.real[ordered]<0)),
        "all_finite_real_mu_squared":values.real[ordered],
        "formulation":"Chebyshev--Lobatto control for regular undivided mixed DAE",
    }


def undivided_scalar_radion_matrices(
    z,
    psi,
    psi_z,
    phi,
    phi_z,
    mass_squared,
    wall_stiffness,
    kappa5_squared=1.0,
):
    """Assemble ``A x = mu_squared B x`` for ``x=(g,f)``.

    The bulk scalar equation is used to evaluate ``Phi_zz`` without
    differentiating sampled data.  Finite wall stiffness is currently
    required because the soft-wall turning profiles are the target of this
    audit; the stiff-potential limit is already covered by the divided master
    solver.
    """
    z=np.asarray(z,dtype=float);psi=np.asarray(psi,dtype=float)
    psi_z=np.asarray(psi_z,dtype=float);phi=np.asarray(phi,dtype=float)
    phi_z=np.asarray(phi_z,dtype=float);mass_squared=float(mass_squared)
    kappa5_squared=float(kappa5_squared);walls=_wall_pair(wall_stiffness)
    size=len(z)
    if (
        z.ndim!=1 or size<5 or any(array.shape!=(size,) for array in (psi,psi_z,phi,phi_z))
        or np.any(np.diff(z)<=0) or np.any(psi<=0) or mass_squared<0
        or kappa5_squared<=0
    ):
        raise ValueError("invalid warped-background inputs")

    derivative=second_order_derivative_matrix(z)
    phi_zz=-3*psi_z*phi_z/psi+psi**2*mass_squared*phi
    proper_phi_y=phi_z/psi
    proper_phi_yy=phi_zz/psi**2-phi_z*psi_z/psi**3
    coupling=4*kappa5_squared/3

    # g has ``size`` nodal values; f has ``size-1`` cell-centered values.
    cell_centers=.5*(z[:-1]+z[1:]);spacing=np.diff(z)
    unknown_count=2*size-1
    matrix=np.zeros((unknown_count,unknown_count))
    mass=np.zeros_like(matrix)

    # Regular linearized constraint at every cell center.
    constraint_coefficient=.5*((psi**2*phi_z)[:-1]+(psi**2*phi_z)[1:])
    for cell in range(size-1):
        matrix[cell,cell]=-1/spacing[cell]
        matrix[cell,cell+1]=1/spacing[cell]
        matrix[cell,size+cell]=-coupling*constraint_coefficient[cell]

    # The undivided Einstein--scalar equation at interior g nodes.  Linear
    # interpolation supplies f at the node, while adjacent cell centers give
    # its centered derivative.
    for i in range(1,size-1):
        row=size-1+i
        left_center=cell_centers[i-1];right_center=cell_centers[i]
        center_distance=right_center-left_center
        left_weight=(right_center-z[i])/center_distance
        right_weight=(z[i]-left_center)/center_distance
        factor=coupling*psi[i]**2
        matrix[row,size+i-1]+=factor*(
            -phi_z[i]/center_distance
            -(phi_zz[i]-phi_z[i]*psi_z[i]/psi[i])*left_weight
        )
        matrix[row,size+i]+=factor*(
            phi_z[i]/center_distance
            -(phi_zz[i]-phi_z[i]*psi_z[i]/psi[i])*right_weight
        )
        matrix[row,i]-=.5*coupling*phi_z[i]**2
        mass[row,i]=-1.

    # Written without Phi_y'/Phi_y.  Signs are the lower/upper orbifold signs
    # of the Wentzell conditions used by the divided master formulation.
    lower_coefficient=coupling*psi[0]**4*(walls[0]*proper_phi_y[0]/2-proper_phi_yy[0])
    upper_coefficient=coupling*psi[-1]**4*(walls[1]*proper_phi_y[-1]/2+proper_phi_yy[-1])
    lower_row=size-1
    lower_distance=cell_centers[1]-cell_centers[0]
    lower_weights=np.array((
        (cell_centers[1]-z[0])/lower_distance,
        (z[0]-cell_centers[0])/lower_distance,
    ))
    upper_distance=cell_centers[-1]-cell_centers[-2]
    upper_weights=np.array((
        (cell_centers[-1]-z[-1])/upper_distance,
        (z[-1]-cell_centers[-2])/upper_distance,
    ))
    matrix[lower_row,size:size+2]=-lower_coefficient*lower_weights
    mass[lower_row,0]=1.
    matrix[-1,-2:]=upper_coefficient*upper_weights
    mass[-1,size-1]=1.

    return {
        "matrix":matrix,
        "mass_matrix":mass,
        "derivative_matrix":derivative,
        "cell_centers":cell_centers,
        "phi_zz":phi_zz,
        "proper_phi_y":proper_phi_y,
        "proper_phi_yy":proper_phi_yy,
        "minimum_abs_phi_z":float(np.min(np.abs(phi_z))),
        "phi_z_sign_change":bool(np.any(phi_z[:-1]*phi_z[1:]<0)),
        "finite_coefficients":bool(np.all(np.isfinite(matrix)) and np.all(np.isfinite(mass))),
        "field_layout":"g at z nodes; f at cell centers",
    }


def undivided_scalar_radion_spectrum(
    z,
    psi,
    psi_z,
    phi,
    phi_z,
    mass_squared,
    wall_stiffness,
    count=8,
    kappa5_squared=1.0,
    imaginary_tolerance=1e-7,
):
    """Return the selected finite real spectrum of the mixed DAE control."""
    assembled=undivided_scalar_radion_matrices(
        z,psi,psi_z,phi,phi_z,mass_squared,wall_stiffness,kappa5_squared,
    )
    values,vectors,finite,ordered,near_zero=_finite_real_spectrum(
        assembled["matrix"],assembled["mass_matrix"],imaginary_tolerance,
    )
    number=min(max(1,int(count)),len(near_zero))
    if number==0:
        raise RuntimeError("no finite real generalized eigenvalues found")
    indices=near_zero[:number]
    modes=vectors[:,indices]
    residuals=[]
    for column,index in enumerate(indices):
        vector=modes[:,column];value=values[index].real
        scale=max(1.,np.linalg.norm(assembled["matrix"]@vector),abs(value)*np.linalg.norm(assembled["mass_matrix"]@vector))
        residuals.append(float(np.linalg.norm(assembled["matrix"]@vector-value*assembled["mass_matrix"]@vector)/scale))
    return {
        **assembled,
        "mu_squared":values.real[indices],
        "eigenvectors":modes,
        "closest_to_zero_mu_squared":float(values.real[indices[0]]),
        "algebraic_minimum_finite_real_mu_squared":float(np.min(values.real[ordered])),
        "negative_finite_real_eigenvalue_count":int(np.count_nonzero(values.real[ordered]<0)),
        "all_finite_real_mu_squared":values.real[ordered],
        "generalized_residuals":np.asarray(residuals),
        "finite_eigenvalue_count":int(np.count_nonzero(finite)),
        "finite_real_eigenvalue_count":int(len(ordered)),
        "positive_finite_real_eigenvalue_count":int(np.count_nonzero(values.real[ordered]>0)),
        "maximum_discarded_finite_imaginary_part":float(np.max(np.abs(values.imag[finite]))) if np.any(finite) else np.inf,
        "formulation":"regular undivided mixed differential-algebraic control in (g,f)",
    }


def shoot_turning_scalar_radion_mode(
    z,
    psi,
    psi_z,
    phi,
    phi_z,
    mass_squared,
    wall_stiffness,
    eigenvalue_hint,
    kappa5_squared=1.0,
    turn_offset_fraction=1e-5,
    tolerance=1e-10,
):
    """Shoot independently from a simple ``Phi_z=0`` turning point.

    At the turn the mixed equations imply ``g_z=0`` and

    ``f=mu_squared*g/(K*psi^2*Phi_zz)``.

    The first derivative of ``f`` remains free.  For fixed ``mu_squared`` two
    linear basis shots determine that derivative from the lower scalar wall
    condition; the upper scalar wall residual is then the scalar root
    determinant.  Integrations start symmetrically on either side using the
    local regular series, so the ODE is never divided by zero.
    """
    z=np.asarray(z,dtype=float);psi=np.asarray(psi,dtype=float)
    psi_z=np.asarray(psi_z,dtype=float);phi=np.asarray(phi,dtype=float)
    phi_z=np.asarray(phi_z,dtype=float);mass_squared=float(mass_squared)
    walls=_wall_pair(wall_stiffness);kappa5_squared=float(kappa5_squared)
    hint=float(eigenvalue_hint);offset_fraction=float(turn_offset_fraction)
    if (
        z.ndim!=1 or len(z)<33 or any(array.shape!=z.shape for array in (psi,psi_z,phi,phi_z))
        or np.any(np.diff(z)<=0) or np.any(psi<=0) or mass_squared<0
        or kappa5_squared<=0 or hint==0 or not 0<offset_fraction<.01
    ):
        raise ValueError("invalid turning-point shooting inputs")
    changes=np.flatnonzero(phi_z[:-1]*phi_z[1:]<0)
    if len(changes)!=1:
        raise ValueError("shooting requires exactly one simple Phi_z sign change")

    psi_spline=CubicSpline(z,psi);psi_z_spline=CubicSpline(z,psi_z)
    phi_spline=CubicSpline(z,phi);phi_z_spline=CubicSpline(z,phi_z)
    left_index=int(changes[0])
    turn=root_scalar(
        phi_z_spline,bracket=(z[left_index],z[left_index+1]),xtol=1e-14,rtol=1e-14,
    ).root
    interval=z[-1]-z[0];offset=offset_fraction*interval
    if not z[0]<turn-offset<turn+offset<z[-1]:
        raise ValueError("turn offset does not remain inside the interval")
    coupling=4*kappa5_squared/3

    def coefficients(location):
        psi_value=float(psi_spline(location));psi_z_value=float(psi_z_spline(location))
        phi_value=float(phi_spline(location));phi_z_value=float(phi_z_spline(location))
        phi_zz_value=(
            -3*psi_z_value*phi_z_value/psi_value
            +psi_value**2*mass_squared*phi_value
        )
        return psi_value,psi_z_value,phi_z_value,phi_zz_value

    psi_turn,psi_z_turn,_,phi_zz_turn=coefficients(turn)
    if abs(phi_zz_turn)<=1e-12:
        raise ValueError("Phi_z turning point is not simple")
    coupling_turn=coupling*psi_turn**2
    coupling_turn_z=2*coupling*psi_turn*psi_z_turn
    # At Phi_z=0, differentiating the exact background scalar equation gives
    # Phi_zzz=-3(psi_z/psi)Phi_zz+2 psi psi_z m^2 Phi.
    phi_zzz_turn=(
        -3*psi_z_turn*phi_zz_turn/psi_turn
        +2*psi_turn*psi_z_turn*mass_squared*float(phi_spline(turn))
    )

    def ode(location,state,eigenvalue):
        g,f=state
        psi_value,psi_z_value,gradient,gradient_z=coefficients(location)
        mixed_gradient=gradient_z-gradient*psi_z_value/psi_value
        coefficient=coupling*psi_value**2
        return np.array((
            coefficient*gradient*f,
            mixed_gradient*f/gradient
            +(.5*coupling*gradient**2-eigenvalue)*g/(coefficient*gradient),
        ))

    def initial_state(eigenvalue,f_slope,side):
        distance=float(side)*offset
        f_at_turn=eigenvalue/(coupling_turn*phi_zz_turn)
        g_third=(
            (2*coupling_turn_z*phi_zz_turn+coupling_turn*phi_zzz_turn)*f_at_turn
            +2*coupling_turn*phi_zz_turn*f_slope
        )
        return np.array((
            1+.5*eigenvalue*distance**2+g_third*distance**3/6,
            f_at_turn+f_slope*distance,
        ))

    def integrate(eigenvalue,f_slope,side):
        start=turn+float(side)*offset
        end=z[-1] if side>0 else z[0]
        solution=solve_ivp(
            lambda location,state:ode(location,state,eigenvalue),
            (start,end),initial_state(eigenvalue,f_slope,side),method="DOP853",
            rtol=tolerance,atol=tolerance*1e-4,
        )
        if not solution.success:raise RuntimeError(solution.message)
        return solution.y[:,-1],len(solution.t)

    def wall_residual(eigenvalue,state,wall):
        location=z[0] if wall==0 else z[-1]
        derivative=ode(location,state,eigenvalue)[1]
        psi_value=float(psi_spline(location));gradient=float(phi_z_spline(location))
        g,f=state
        if wall==0:
            return float(derivative-.5*walls[0]*psi_value*f-gradient*g/(2*psi_value**2))
        return float(derivative+.5*walls[1]*psi_value*f-gradient*g/(2*psi_value**2))

    def basis_residuals(eigenvalue):
        records=[]
        for slope in (0.,1.):
            left,steps_left=integrate(eigenvalue,slope,-1)
            right,steps_right=integrate(eigenvalue,slope,1)
            records.append((
                wall_residual(eigenvalue,left,0),
                wall_residual(eigenvalue,right,1),
                steps_left+steps_right,
            ))
        lower_delta=records[1][0]-records[0][0]
        if abs(lower_delta)<=1e-14*max(1.,abs(records[0][0]),abs(records[1][0])):
            raise RuntimeError("lower wall does not determine the free turn slope")
        slope=-records[0][0]/lower_delta
        upper=records[0][1]+slope*(records[1][1]-records[0][1])
        return float(upper),float(slope),records

    def determinant(eigenvalue):
        return basis_residuals(float(eigenvalue))[0]

    # A local secant iteration avoids treating the zero-eigenvalue gauge limit
    # as the target when the physical signed mode lies close to zero.
    root=root_scalar(
        determinant,x0=hint,x1=hint*(1+1e-3),method="secant",
        xtol=max(abs(hint)*tolerance,1e-15),rtol=tolerance,maxiter=30,
    )
    if not root.converged:
        raise RuntimeError("turning-point shooting root did not converge")
    upper,slope,records=basis_residuals(root.root)
    left,steps_left=integrate(root.root,slope,-1)
    right,steps_right=integrate(root.root,slope,1)
    lower=wall_residual(root.root,left,0)
    return {
        "mu_squared":float(root.root),
        "turn_location":float(turn),
        "turn_phi_zz":float(phi_zz_turn),
        "turn_f":float(root.root/(coupling_turn*phi_zz_turn)),
        "turn_f_z":float(slope),
        "lower_wall_residual":float(lower),
        "upper_wall_residual":float(upper),
        "iterations":int(root.iterations),
        "converged":bool(root.converged),
        "integration_steps":int(steps_left+steps_right),
        "turn_offset_fraction":offset_fraction,
        "method":"two-sided DOP853 shooting from regular simple-turn series",
    }
