"""Executable frozen reduced-wave initial-boundary value problem.

This module advances the principal one-normal-dimensional wave system used by
the Israel--two-scalar boundary audit.  It is deliberately narrower than a
nonlinear generalized-harmonic Einstein solver: the bulk principal part is
the flat vector wave operator, while the complete audited lower-order Robin
matrix is retained at both walls.

Piecewise-linear finite elements make the Robin rows natural boundary terms.
The four mixed normal--tangent gauge fields are imposed strongly as homogeneous
Dirichlet data.  A method-of-lines RK4 integrator then provides a small runtime
test of the boundary algebra, its finite growth shift, and the corresponding
constraint-propagation model.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def linear_finite_element_matrices(points):
    """Return consistent mass and stiffness matrices on an ordered 1-D grid."""
    points=np.asarray(points,dtype=float)
    if points.ndim!=1 or len(points)<3 or np.any(np.diff(points)<=0):
        raise ValueError("points must be a strictly ordered one-dimensional grid")
    size=len(points);mass=np.zeros((size,size));stiffness=np.zeros((size,size))
    for index,spacing in enumerate(np.diff(points)):
        mass[index:index+2,index:index+2]+=spacing/6*np.array(((2.,1.),(1.,2.)))
        stiffness[index:index+2,index:index+2]+=1/spacing*np.array(((1.,-1.),(-1.,1.)))
    return mass,stiffness


def variable_linear_finite_element_matrices(points,mass_weight,gradient_weight):
    """Return P1 matrices for ``w u_tt-(p u_x)_x``.

    Both positive coefficients are represented by their continuous linear
    interpolants.  The mass integrals are exact for that representation.
    """
    points=np.asarray(points,dtype=float)
    mass_weight=np.asarray(mass_weight,dtype=float)
    gradient_weight=np.asarray(gradient_weight,dtype=float)
    if (
        points.ndim!=1 or len(points)<3 or np.any(np.diff(points)<=0)
        or mass_weight.shape!=points.shape or gradient_weight.shape!=points.shape
        or np.any(mass_weight<=0) or np.any(gradient_weight<=0)
    ):
        raise ValueError("invalid variable-coefficient finite-element inputs")
    size=len(points);mass=np.zeros((size,size));stiffness=np.zeros((size,size))
    for index,spacing in enumerate(np.diff(points)):
        w0,w1=mass_weight[index:index+2]
        p0,p1=gradient_weight[index:index+2]
        mass[index:index+2,index:index+2]+=spacing/12*np.array(
            ((3*w0+w1,w0+w1),(w0+w1,w0+3*w1))
        )
        stiffness[index:index+2,index:index+2]+=(p0+p1)/(2*spacing)*np.array(
            ((1.,-1.),(-1.,1.))
        )
    return mass,stiffness


def anisotropic_wave_principal_coefficients(psi,a,b,c,lapse=None):
    """Normal-wave weights for a diagonal anisotropic time-symmetric slice.

    For ``ds^2=-alpha^2 dt^2+A^2 dz^2+B^2 dr^2+(Cr)^2dOmega_2``, a field
    with only normal derivatives obeys

    ``w u_tt - partial_z(p u_z)=0`` with
    ``w=A B C^2/alpha`` and ``p=alpha B C^2/A``.

    The coefficient converting the orthonormal Robin row into the weak
    coordinate boundary term is ``s=alpha B C^2``.
    """
    psi=np.asarray(psi,dtype=float);a=np.asarray(a,dtype=float)
    b=np.asarray(b,dtype=float);c=np.asarray(c,dtype=float)
    alpha=psi if lapse is None else np.asarray(lapse,dtype=float)
    if not (psi.shape==a.shape==b.shape==c.shape==alpha.shape) or np.any(psi<=0) or np.any(alpha<=0):
        raise ValueError("metric and lapse arrays must be matching and positive")
    aa=psi*np.exp(a);bb=psi*np.exp(b);cc=psi*np.exp(c)
    return {
        "mass_weight":aa*bb*cc**2/alpha,
        "gradient_weight":alpha*bb*cc**2/aa,
        "boundary_weight":alpha*bb*cc**2,
        "coordinate_speed":alpha/aa,
        "A":aa,"B":bb,"C":cc,"lapse":alpha,
    }


class FrozenReducedWaveIBVP:
    """Finite-element method-of-lines solver for Dirichlet plus Robin waves.

    The field order is ``dirichlet_fields`` homogeneous Dirichlet variables
    followed by the variables satisfying

    ``partial_n u = R_left/right u + boundary_data``.

    Volume forcing is specified in the strong wave equation
    ``u_tt = u_xx + source``.  Boundary data are outward-normal residuals.
    """

    def __init__(self,points,left_robin,right_robin,dirichlet_fields=4):
        self.points=np.asarray(points,dtype=float)
        self.mass,self.stiffness=linear_finite_element_matrices(self.points)
        self.left_robin=np.asarray(left_robin,dtype=float)
        self.right_robin=np.asarray(right_robin,dtype=float)
        if (
            self.left_robin.ndim!=2
            or self.left_robin.shape[0]!=self.left_robin.shape[1]
            or self.right_robin.shape!=self.left_robin.shape
        ):
            raise ValueError("left and right Robin matrices must be matching square arrays")
        self.dirichlet_fields=int(dirichlet_fields)
        if self.dirichlet_fields<0:
            raise ValueError("dirichlet_fields must be nonnegative")
        self.robin_fields=self.left_robin.shape[0]
        self.field_count=self.dirichlet_fields+self.robin_fields
        self._mass_factor=cho_factor(self.mass,check_finite=False)
        self._interior_mass_factor=(
            cho_factor(self.mass[1:-1,1:-1],check_finite=False)
            if self.dirichlet_fields else None
        )

    def _volume_source(self,source,time):
        if source is None:
            return np.zeros((len(self.points),self.field_count))
        values=np.asarray(source(float(time),self.points),dtype=float)
        if values.shape!=(len(self.points),self.field_count):
            raise ValueError("volume source has the wrong shape")
        return values

    def _boundary_source(self,source,time):
        if source is None:return np.zeros(self.robin_fields)
        values=np.asarray(source(float(time)),dtype=float)
        if values.shape!=(self.robin_fields,):
            raise ValueError("Robin boundary source has the wrong shape")
        return values

    def acceleration(
        self,time,position,source=None,left_boundary_data=None,right_boundary_data=None,
    ):
        """Return the Galerkin acceleration at one method-of-lines state."""
        position=np.asarray(position,dtype=float)
        if position.shape!=(len(self.points),self.field_count):
            raise ValueError("position has the wrong shape")
        load=-self.stiffness@position+self.mass@self._volume_source(source,time)
        first=self.dirichlet_fields
        robin=position[:,first:]
        load[0,first:]+=(
            self.left_robin@robin[0]+self._boundary_source(left_boundary_data,time)
        )
        load[-1,first:]+=(
            self.right_robin@robin[-1]+self._boundary_source(right_boundary_data,time)
        )
        acceleration=np.zeros_like(position)
        if first:
            acceleration[1:-1,:first]=cho_solve(
                self._interior_mass_factor,load[1:-1,:first],check_finite=False,
            )
        acceleration[:,first:]=cho_solve(
            self._mass_factor,load[:,first:],check_finite=False,
        )
        return acceleration

    def integrate(
        self,initial_position,initial_velocity,final_time,courant=.2,
        source=None,left_boundary_data=None,right_boundary_data=None,
        diagnostic=None,
    ):
        """Advance with RK4 and return the final state and optional diagnostics."""
        position=np.asarray(initial_position,dtype=float).copy()
        velocity=np.asarray(initial_velocity,dtype=float).copy()
        expected=(len(self.points),self.field_count)
        if position.shape!=expected or velocity.shape!=expected:
            raise ValueError("initial arrays have the wrong shape")
        if self.dirichlet_fields:
            if np.max(np.abs(position[[0,-1],:self.dirichlet_fields]))>1e-13:
                raise ValueError("Dirichlet positions must vanish at both endpoints")
            if np.max(np.abs(velocity[[0,-1],:self.dirichlet_fields]))>1e-13:
                raise ValueError("Dirichlet velocities must vanish at both endpoints")
        duration=float(final_time);courant=float(courant)
        if duration<0 or courant<=0:raise ValueError("invalid integration controls")
        steps=max(1,int(np.ceil(duration/(courant*np.min(np.diff(self.points))))))
        dt=duration/steps if duration else 0.
        records=[]

        def rhs(time,q,p):
            return p,self.acceleration(
                time,q,source,left_boundary_data,right_boundary_data,
            )

        if diagnostic is not None:records.append(diagnostic(0.,position,velocity))
        time=0.
        for _ in range(steps):
            k1q,k1p=rhs(time,position,velocity)
            k2q,k2p=rhs(time+dt/2,position+dt*k1q/2,velocity+dt*k1p/2)
            k3q,k3p=rhs(time+dt/2,position+dt*k2q/2,velocity+dt*k2p/2)
            k4q,k4p=rhs(time+dt,position+dt*k3q,velocity+dt*k3p)
            position+=dt*(k1q+2*k2q+2*k3q+k4q)/6
            velocity+=dt*(k1p+2*k2p+2*k3p+k4p)/6
            time+=dt
            if self.dirichlet_fields:
                position[[0,-1],:self.dirichlet_fields]=0.
                velocity[[0,-1],:self.dirichlet_fields]=0.
            if diagnostic is not None:records.append(diagnostic(time,position,velocity))
        return {
            "time":time,"position":position,"velocity":velocity,
            "steps":steps,"time_step":dt,"diagnostics":records,
        }

    def l2_norm(self,values):
        """Consistent-mass spatial L2 norm summed over fields."""
        values=np.asarray(values,dtype=float)
        if values.shape!=(len(self.points),self.field_count):
            raise ValueError("values have the wrong shape")
        return float(np.sqrt(max(0.,np.sum((self.mass@values)*values))))

    def symmetrized_energy(self,position,velocity,robin_symmetrizer):
        """Semidiscrete energy when one symmetrizer applies at both walls.

        For homogeneous data this energy is conserved by the semidiscrete
        equations when ``W R_left`` and ``W R_right`` are symmetric.  It can
        be indefinite for positive Robin eigenvalues, exactly reflecting the
        finite lower-order growth shift of the frozen reduced system.
        """
        position=np.asarray(position,dtype=float);velocity=np.asarray(velocity,dtype=float)
        if position.shape!=(len(self.points),self.field_count) or velocity.shape!=position.shape:
            raise ValueError("state arrays have the wrong shape")
        weight=np.asarray(robin_symmetrizer,dtype=float)
        if weight.shape!=(self.robin_fields,self.robin_fields):
            raise ValueError("Robin symmetrizer has the wrong shape")
        first=self.dirichlet_fields
        qd=position[:,:first];pd=velocity[:,:first]
        qr=position[:,first:];pr=velocity[:,first:]
        kinetic=.5*np.sum((self.mass@pd)*pd)
        gradient=.5*np.sum((self.stiffness@qd)*qd)
        kinetic+=.5*np.sum((self.mass@pr)*(pr@weight))
        gradient+=.5*np.sum((self.stiffness@qr)*(qr@weight))
        boundary=-.5*float(qr[0]@weight@self.left_robin@qr[0])
        boundary-=.5*float(qr[-1]@weight@self.right_robin@qr[-1])
        return {
            "total":float(kinetic+gradient+boundary),
            "kinetic":float(kinetic),"gradient":float(gradient),
            "boundary":float(boundary),
        }


class VariableCoefficientReducedWaveIBVP(FrozenReducedWaveIBVP):
    """Frozen-field vector waves on a variable diagonal metric collar."""

    def __init__(
        self,points,mass_weight,gradient_weight,left_robin,right_robin,
        left_boundary_weight,right_boundary_weight,dirichlet_fields=4,
    ):
        self.points=np.asarray(points,dtype=float)
        self.mass_weight=np.asarray(mass_weight,dtype=float)
        self.gradient_weight=np.asarray(gradient_weight,dtype=float)
        self.mass,self.stiffness=variable_linear_finite_element_matrices(
            self.points,self.mass_weight,self.gradient_weight,
        )
        self.physical_left_robin=np.asarray(left_robin,dtype=float)
        self.physical_right_robin=np.asarray(right_robin,dtype=float)
        if (
            self.physical_left_robin.ndim!=2
            or self.physical_left_robin.shape[0]!=self.physical_left_robin.shape[1]
            or self.physical_right_robin.shape!=self.physical_left_robin.shape
        ):
            raise ValueError("left and right Robin matrices must be matching square arrays")
        self.left_boundary_weight=float(left_boundary_weight)
        self.right_boundary_weight=float(right_boundary_weight)
        if self.left_boundary_weight<=0 or self.right_boundary_weight<=0:
            raise ValueError("boundary weights must be positive")
        # These are the natural weak matrices p u_z, including conversion
        # from the orthonormal outward derivative to the coordinate endpoint.
        self.left_robin=self.left_boundary_weight*self.physical_left_robin
        self.right_robin=self.right_boundary_weight*self.physical_right_robin
        self.dirichlet_fields=int(dirichlet_fields)
        if self.dirichlet_fields<0:raise ValueError("dirichlet_fields must be nonnegative")
        self.robin_fields=self.physical_left_robin.shape[0]
        self.field_count=self.dirichlet_fields+self.robin_fields
        self._mass_factor=cho_factor(self.mass,check_finite=False)
        self._interior_mass_factor=(
            cho_factor(self.mass[1:-1,1:-1],check_finite=False)
            if self.dirichlet_fields else None
        )

    def acceleration(
        self,time,position,source=None,left_boundary_data=None,right_boundary_data=None,
    ):
        """Return acceleration with physical orthonormal Robin source data."""
        position=np.asarray(position,dtype=float)
        if position.shape!=(len(self.points),self.field_count):
            raise ValueError("position has the wrong shape")
        load=-self.stiffness@position+self.mass@self._volume_source(source,time)
        first=self.dirichlet_fields;robin=position[:,first:]
        load[0,first:]+=(
            self.left_robin@robin[0]
            +self.left_boundary_weight*self._boundary_source(left_boundary_data,time)
        )
        load[-1,first:]+=(
            self.right_robin@robin[-1]
            +self.right_boundary_weight*self._boundary_source(right_boundary_data,time)
        )
        acceleration=np.zeros_like(position)
        if first:
            acceleration[1:-1,:first]=cho_solve(
                self._interior_mass_factor,load[1:-1,:first],check_finite=False,
            )
        acceleration[:,first:]=cho_solve(
            self._mass_factor,load[:,first:],check_finite=False,
        )
        return acceleration

    def interpolated_symmetrizer_energy(
        self,position,velocity,left_symmetrizer,right_symmetrizer,shift=0.,
    ):
        """Energy with a linear positive collar extension of the wall weights.

        Its continuum balance has the lower-order volume term
        ``-integral p u_t^T W_x u_x``.  ``power`` returns that term, permitting
        a direct energy-ledger audit when the wall symmetrizers differ.
        """
        position=np.asarray(position,dtype=float);velocity=np.asarray(velocity,dtype=float)
        if position.shape!=(len(self.points),self.field_count) or velocity.shape!=position.shape:
            raise ValueError("state arrays have the wrong shape")
        left=np.asarray(left_symmetrizer,dtype=float);right=np.asarray(right_symmetrizer,dtype=float)
        if left.shape!=(self.robin_fields,self.robin_fields) or right.shape!=left.shape:
            raise ValueError("symmetrizers have the wrong shape")
        shift=float(shift);length=self.points[-1]-self.points[0]
        gradient_w=(right-left)/length
        # Three-point Gauss integration is exact for all polynomial products
        # generated by linearly interpolated fields, coefficients, and W.
        gauss=(-np.sqrt(3/5),0.,np.sqrt(3/5));weights=(5/9,8/9,5/9)
        first=self.dirichlet_fields
        kinetic=gradient=mass_shift=power=shift_power=0.
        qd=position[:,:first];pd=velocity[:,:first]
        # The strongly Dirichlet gauge block uses its natural identity energy.
        kinetic+=.5*float(np.sum((self.mass@pd)*pd))
        gradient+=.5*float(np.sum((self.stiffness@qd)*qd))
        qr=position[:,first:];pr=velocity[:,first:]
        for index,spacing in enumerate(np.diff(self.points)):
            qx=(qr[index+1]-qr[index])/spacing
            for xi,weight in zip(gauss,weights):
                n0=(1-xi)/2;n1=(1+xi)/2
                coordinate=n0*self.points[index]+n1*self.points[index+1]
                fraction=(coordinate-self.points[0])/length
                symmetrizer=(1-fraction)*left+fraction*right
                w=n0*self.mass_weight[index]+n1*self.mass_weight[index+1]
                p=n0*self.gradient_weight[index]+n1*self.gradient_weight[index+1]
                q=n0*qr[index]+n1*qr[index+1]
                v=n0*pr[index]+n1*pr[index+1]
                jacobian=spacing/2*weight
                kinetic+=.5*jacobian*w*float(v@symmetrizer@v)
                gradient+=.5*jacobian*p*float(qx@symmetrizer@qx)
                mass_shift+=.5*shift*jacobian*w*float(q@symmetrizer@q)
                power-=jacobian*p*float(v@gradient_w@qx)
                shift_power+=shift*jacobian*w*float(q@symmetrizer@v)
        boundary=-.5*float(qr[0]@left@self.left_robin@qr[0])
        boundary-=.5*float(qr[-1]@right@self.right_robin@qr[-1])
        base=kinetic+gradient+boundary
        return {
            "base":float(base),"shifted":float(base+mass_shift),
            "kinetic":float(kinetic),"gradient":float(gradient),
            "boundary":float(boundary),"mass_shift":float(mass_shift),
            "predicted_base_energy_power":float(power),
            "predicted_shifted_energy_power":float(power+shift_power),
        }

def endpoint_robin_residual(points,values,left_robin,right_robin):
    """Fourth-order one-sided diagnostic of homogeneous Robin residuals."""
    points=np.asarray(points,dtype=float);values=np.asarray(values,dtype=float)
    left=np.asarray(left_robin,dtype=float);right=np.asarray(right_robin,dtype=float)
    if len(points)<5 or values.shape!=(len(points),left.shape[0]) or right.shape!=left.shape:
        raise ValueError("invalid endpoint residual inputs")
    spacing=np.diff(points)
    if np.max(np.abs(spacing-spacing[0]))>1e-12*max(1.,abs(spacing[0])):
        raise ValueError("endpoint residual diagnostic currently requires a uniform grid")
    h=spacing[0]
    derivative_left=(-25*values[0]+48*values[1]-36*values[2]+16*values[3]-3*values[4])/(12*h)
    derivative_right=(25*values[-1]-48*values[-2]+36*values[-3]-16*values[-4]+3*values[-5])/(12*h)
    residual_left=-derivative_left-left@values[0]
    residual_right=derivative_right-right@values[-1]
    scale=max(
        1.,float(np.linalg.norm(derivative_left)),float(np.linalg.norm(left@values[0])),
        float(np.linalg.norm(derivative_right)),float(np.linalg.norm(right@values[-1])),
    )
    return {
        "left":residual_left,"right":residual_right,
        "maximum_absolute":float(max(np.max(np.abs(residual_left)),np.max(np.abs(residual_right)))),
        "normalized_l2":float(np.hypot(np.linalg.norm(residual_left),np.linalg.norm(residual_right))/scale),
    }
