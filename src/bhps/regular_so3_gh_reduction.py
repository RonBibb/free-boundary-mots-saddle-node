"""Regular SO(3)-symmetric reduction of the 5D GH perturbation operator.

The three brane-space Cartesian coordinates are ``x_i``.  A regular
SO(3)-invariant metric perturbation is parameterized by seven scalar
functions,

``h_0i=v0 x_i``, ``h_zi=vz x_i``, and
``h_ij=p delta_ij+d x_i x_j``,

together with ``h_z0``, ``h_00``, and ``h_zz``.  The variables ``v0``,
``vz``, and ``d`` remain finite and even at the axis.  Adding the stabilizer
and collapse scalar gives a nine-field system.  This is the physical
spherical-on-the-brane sector of the 17-field wall system; treating all
seventeen Cartesian components as unrelated radial scalars would omit the
tensorial angular derivatives.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RectBivariateSpline

from bhps.adm_corner import _axisymmetric_derivatives
from bhps.linearized_gh_einstein_scalar import linearized_reduced_einstein_two_scalar_residual,metric_geometry_from_jets


FIELD_ORDER=(
    "h_z0","v_z=h_zr/r","h00","h_perp",
    "d=(h_rr-h_perp)/r^2","v_0=h_0r/r","h_zz","delta_Phi","delta_chi",
)
REDUCED_DIRECTIONS=("t","z","r")


def regular_so3_robin_matrix(
    umbilic_coefficient,umbilic_phi_derivative,inward_phi_derivative,
    wall_stiffness,
):
    """Return the seven-field Robin block in ``FIELD_ORDER[2:]``.

    The matrix is the exact SO(3)-invariant restriction of the full thirteen
    field Israel--two-scalar block.  In particular, subtracting the radial and
    transverse diagonal rows gives ``partial_n d=-2 c d`` without a division
    by radius.
    """
    c=float(umbilic_coefficient);cp=float(umbilic_phi_derivative)
    phix=float(inward_phi_derivative);gamma=float(wall_stiffness)
    if gamma<0:raise ValueError("wall stiffness must be nonnegative")
    matrix=np.zeros((7,7))
    # h00, h_perp, d, v0, h_zz, delta Phi, delta chi.
    matrix[0,0]=-2*c;matrix[0,4]=c;matrix[0,5]=2*cp
    matrix[1,1]=-2*c;matrix[1,4]=-c;matrix[1,5]=-2*cp
    matrix[2,2]=-2*c;matrix[3,3]=-2*c
    matrix[4,4]=-12*c;matrix[4,5]=-8*cp
    matrix[5,4]=-.5*phix;matrix[5,5]=-gamma/2
    return {"matrix":matrix,"field_order":FIELD_ORDER[2:]}


def _add_symmetric(array,left,right,value):
    array[left,right]+=value
    if left!=right:array[right,left]+=value


def _radial_unit_data(radius):
    radius=float(radius);n=np.array((1.,0.,0.));identity=np.eye(3)
    if radius<=0:return n,identity,np.zeros((3,3)),None
    projector=identity-n[:,None]*n[None,:]
    n_first=projector/radius
    n_second=np.empty((3,3,3))
    for a in range(3):
        for i in range(3):
            for j in range(3):
                n_second[a,i,j]=-(
                    identity[a,j]*n[i]+identity[a,i]*n[j]+n[a]*identity[i,j]
                    -3*n[a]*n[i]*n[j]
                )/radius**2
    return n,identity,n_first,n_second


def _scalar_cartesian_jets(radius,value,first,second):
    """Map a reduced scalar jet in (t,z,r) to (t,z,x,y,w)."""
    radius=float(radius);value=float(value);first=np.asarray(first);second=np.asarray(second)
    full_first=np.zeros(5);full_second=np.zeros((5,5))
    full_first[:2]=first[:2];full_second[:2,:2]=second[:2,:2]
    if radius>0:
        n=np.array((1.,0.,0.));projector=np.eye(3)-n[:,None]*n[None,:]
        full_first[2:]=first[2]*n
        for mu in range(2):
            full_second[mu,2:]=second[mu,2]*n
            full_second[2:,mu]=second[2,mu]*n
        full_second[2:,2:]=second[2,2]*n[:,None]*n[None,:]+first[2]*projector/radius
    else:
        # Even regular variables have zero radial first derivative at r=0.
        full_second[2:,2:]=second[2,2]*np.eye(3)
    return value,full_first,full_second


def regular_so3_perturbation_jets(radius,values,first=None,second=None):
    """Expand nine reduced field jets into full Cartesian perturbation jets."""
    radius=float(radius);values=np.asarray(values,dtype=float)
    first=np.zeros((3,9)) if first is None else np.asarray(first,dtype=float)
    second=np.zeros((3,3,9)) if second is None else np.asarray(second,dtype=float)
    if values.shape!=(9,) or first.shape!=(3,9) or second.shape!=(3,3,9):
        raise ValueError("invalid reduced perturbation jets")
    if radius<0:raise ValueError("radius must be nonnegative")
    result={
        "metric":np.zeros((5,5)),"metric_first":np.zeros((5,5,5)),
        "metric_second":np.zeros((5,5,5,5)),
        "phi":0.,"phi_first":np.zeros(5),"phi_second":np.zeros((5,5)),
        "chi":0.,"chi_first":np.zeros(5),"chi_second":np.zeros((5,5)),
    }

    # Scalar metric blocks h_z0, h00, and h_zz.
    for field,pair in ((0,(1,0)),(2,(0,0)),(6,(1,1))):
        value,gradient,hessian=_scalar_cartesian_jets(
            radius,values[field],first[:,field],second[:,:,field],
        )
        _add_symmetric(result["metric"],*pair,value)
        for derivative in range(5):
            _add_symmetric(result["metric_first"][derivative],*pair,gradient[derivative])
        for left in range(5):
            for right in range(5):
                _add_symmetric(result["metric_second"][left,right],*pair,hessian[left,right])

    n,identity,n_first,n_second=_radial_unit_data(radius)
    x=radius*n
    # Vector blocks h_zi=v_z x_i and h_0i=v_0 x_i.
    for field,fixed in ((1,1),(5,0)):
        value=values[field];field_first=first[:,field];field_second=second[:,:,field]
        for a in range(3):
            pair=(fixed,a+2)
            _add_symmetric(result["metric"],*pair,value*x[a])
            for mu in range(2):
                _add_symmetric(result["metric_first"][mu],*pair,field_first[mu]*x[a])
            for i in range(3):
                derivative=value*identity[a,i]
                if radius>0:derivative+=field_first[2]*n[i]*x[a]
                _add_symmetric(result["metric_first"][i+2],*pair,derivative)
            for mu in range(2):
                for nu in range(2):
                    _add_symmetric(result["metric_second"][mu,nu],*pair,field_second[mu,nu]*x[a])
                for i in range(3):
                    cross=field_first[mu]*identity[a,i]
                    if radius>0:cross+=field_second[mu,2]*n[i]*x[a]
                    _add_symmetric(result["metric_second"][mu,i+2],*pair,cross)
                    _add_symmetric(result["metric_second"][i+2,mu],*pair,cross)
            if radius>0:
                projector=identity-n[:,None]*n[None,:]
                for i in range(3):
                    for j in range(3):
                        spatial=(
                            field_second[2,2]*n[i]*n[j]*x[a]
                            +field_first[2]*projector[i,j]*x[a]/radius
                            +field_first[2]*(n[i]*identity[a,j]+n[j]*identity[a,i])
                        )
                        _add_symmetric(result["metric_second"][i+2,j+2],*pair,spatial)
            # At the axis the second Cartesian derivative of an odd regular
            # vector is zero; the mixed t/z--Cartesian derivatives above are
            # finite and nonzero.

    # Spatial tensor h_ij=p delta_ij+d x_i x_j.
    p=values[3];d=values[4];pf=first[:,3];df=first[:,4]
    ps=second[:,:,3];ds=second[:,:,4]
    for a in range(3):
        for b in range(3):
            pair=(a+2,b+2);delta=identity[a,b]
            result["metric"][pair]+=p*delta+d*x[a]*x[b]
            for mu in range(2):
                result["metric_first"][mu][pair]+=pf[mu]*delta+df[mu]*x[a]*x[b]
            for i in range(3):
                spatial_first=0.
                if radius>0:
                    spatial_first=pf[2]*n[i]*delta+df[2]*n[i]*x[a]*x[b]
                spatial_first+=d*(identity[a,i]*x[b]+x[a]*identity[b,i])
                result["metric_first"][i+2][pair]+=spatial_first
            for mu in range(2):
                for nu in range(2):
                    result["metric_second"][mu,nu][pair]+=ps[mu,nu]*delta+ds[mu,nu]*x[a]*x[b]
                for i in range(3):
                    cross=df[mu]*(identity[a,i]*x[b]+x[a]*identity[b,i])
                    if radius>0:cross+=ps[mu,2]*n[i]*delta+ds[mu,2]*n[i]*x[a]*x[b]
                    result["metric_second"][mu,i+2][pair]+=cross
                    result["metric_second"][i+2,mu][pair]+=cross
            if radius>0:
                projector=identity-n[:,None]*n[None,:]
                for i in range(3):
                    for j in range(3):
                        spatial=(
                            ps[2,2]*n[i]*n[j]*delta+pf[2]*projector[i,j]*delta/radius
                            +ds[2,2]*n[i]*n[j]*x[a]*x[b]
                            +df[2]*projector[i,j]*x[a]*x[b]/radius
                            +df[2]*n[i]*(identity[a,j]*x[b]+x[a]*identity[b,j])
                            +df[2]*n[j]*(identity[a,i]*x[b]+x[a]*identity[b,i])
                            +d*(identity[a,i]*identity[b,j]+identity[a,j]*identity[b,i])
                        )
                        result["metric_second"][i+2,j+2][pair]+=spatial
            else:
                for i in range(3):
                    for j in range(3):
                        result["metric_second"][i+2,j+2][pair]+=(
                            ps[2,2]*identity[i,j]*delta
                            +d*(identity[a,i]*identity[b,j]+identity[a,j]*identity[b,i])
                        )

    for field,prefix in ((7,"phi"),(8,"chi")):
        value,gradient,hessian=_scalar_cartesian_jets(
            radius,values[field],first[:,field],second[:,:,field],
        )
        result[prefix]=value;result[f"{prefix}_first"]=gradient
        result[f"{prefix}_second"]=hessian
    return result


def _even_quotient_axis(field,r,power):
    """Fill the removable axis value of ``field/r**power`` by an even fit."""
    field=np.asarray(field,dtype=float);r=np.asarray(r,dtype=float);result=np.empty_like(field)
    result[:,1:]=field[:,1:]/r[None,1:]**power
    degree=3;count=min(7,len(r))
    if count<4:raise ValueError("radial grid is too short for an axis quotient")
    for index,row in enumerate(result):
        result[index,0]=np.polynomial.polynomial.polyfit(
            r[1:count]**2,row[1:count],degree,
        )[0]
    return result


class RegularSO3BackgroundJetField:
    """Interpolated regular Cartesian background jets on a z-r grid."""

    def __init__(
        self,z,r,alpha,psi,a,b,c,phi,chi,metric_acceleration,
        lapse_acceleration,phi_acceleration,chi_acceleration,stencil_width=7,
    ):
        self.z=np.asarray(z,dtype=float);self.r=np.asarray(r,dtype=float)
        shape=(len(self.z),len(self.r))
        fields=[np.asarray(item,dtype=float) for item in (alpha,psi,a,b,c,phi,chi)]
        if any(item.shape!=shape for item in fields) or self.r[0]!=0:
            raise ValueError("invalid regular background fields")
        alpha,psi,a,b,c,phi,chi=fields
        A2=psi**2*np.exp(2*a);B2=psi**2*np.exp(2*b);C2=psi**2*np.exp(2*c)
        anisotropy=_even_quotient_axis(B2-C2,self.r,2)
        zero=np.zeros(shape)
        self.reduced_fields=np.stack((zero,zero,-alpha**2,C2,anisotropy,zero,A2,phi,chi),axis=2)
        self.reduced_first=np.zeros((3,*shape,9));self.reduced_second=np.zeros((3,3,*shape,9))
        for field in range(9):
            derivatives=_axisymmetric_derivatives(
                self.reduced_fields[:,:,field],self.z,self.r,stencil_width,
            )
            self.reduced_first[1,:,:,field]=derivatives["z"]
            self.reduced_first[2,:,:,field]=derivatives["r"]
            self.reduced_second[1,1,:,:,field]=derivatives["zz"]
            self.reduced_second[1,2,:,:,field]=derivatives["zr"]
            self.reduced_second[2,1,:,:,field]=derivatives["zr"]
            self.reduced_second[2,2,:,:,field]=derivatives["rr"]
        lapse_tt=np.asarray(lapse_acceleration,dtype=float)
        phi_tt=np.asarray(phi_acceleration,dtype=float);chi_tt=np.asarray(chi_acceleration,dtype=float)
        accelerations={name:np.asarray(metric_acceleration[name],dtype=float) for name in ("zz","radial","transverse","zr")}
        if any(item.shape!=shape for item in (lapse_tt,phi_tt,chi_tt,*accelerations.values())):
            raise ValueError("invalid background acceleration fields")
        self.reduced_second[0,0,:,:,2]=-2*alpha*lapse_tt
        self.reduced_second[0,0,:,:,3]=accelerations["transverse"]
        self.reduced_second[0,0,:,:,4]=_even_quotient_axis(
            accelerations["radial"]-accelerations["transverse"],self.r,2,
        )
        self.reduced_second[0,0,:,:,1]=_even_quotient_axis(
            accelerations["zr"],self.r,1,
        )
        self.reduced_second[0,0,:,:,6]=accelerations["zz"]
        self.reduced_second[0,0,:,:,7]=phi_tt
        self.reduced_second[0,0,:,:,8]=chi_tt

    def _interpolate(self,array,z_value,r_value):
        result=np.empty(array.shape[2:])
        for index in np.ndindex(result.shape):
            result[index]=RectBivariateSpline(
                self.z,self.r,array[(slice(None),slice(None))+index],kx=3,ky=3,s=0,
            ).ev(z_value,r_value).item()
        return result

    def at(self,z_value,r_value):
        """Return complete Cartesian background jets at a physical point."""
        z_value=float(z_value);r_value=float(r_value)
        if not self.z[0]<=z_value<=self.z[-1] or not self.r[0]<=r_value<=self.r[-1]:
            raise ValueError("requested point lies outside the background grid")
        values=self._interpolate(self.reduced_fields,z_value,r_value)
        first=self._interpolate(
            np.moveaxis(self.reduced_first,0,2),z_value,r_value,
        )
        second_interpolation=np.moveaxis(self.reduced_second,(0,1),(2,3))
        second=self._interpolate(second_interpolation,z_value,r_value)
        return regular_so3_perturbation_jets(r_value,values,first,second)


def pack_regular_so3_residual(metric_residual,phi_residual,chi_residual,radius):
    """Project a Cartesian residual onto the nine regular reduced rows."""
    radius=float(radius);metric=np.asarray(metric_residual)
    if radius<=0:raise ValueError("pointwise regular-row projection requires r>0")
    transverse=.5*(metric[3,3]+metric[4,4])
    return np.array((
        metric[1,0],metric[1,2]/radius,metric[0,0],transverse,
        (metric[2,2]-transverse)/radius**2,metric[0,2]/radius,metric[1,1],
        phi_residual,chi_residual,
    ))


def _linear_column(
    background,perturbation,radius,mass_squared,potential_offset,kappa5_squared,
    constraint_damping=0.,constraint_damping_rho=0.,
):
    result=linearized_reduced_einstein_two_scalar_residual(
        background,perturbation,mass_squared=mass_squared,
        potential_offset=potential_offset,kappa5_squared=kappa5_squared,
        constraint_damping=constraint_damping,
        constraint_damping_rho=constraint_damping_rho,
    )
    return pack_regular_so3_residual(
        result["metric_residual"],result["phi_residual"],result["chi_residual"],radius,
    )


def regular_so3_gh_coefficient_matrices(
    background,radius,mass_squared=0.,potential_offset=-6.,kappa5_squared=1.,
    constraint_damping=0.,constraint_damping_rho=0.,
):
    """Extract the nine-field value, first-, and pure-second-jet matrices."""
    radius=float(radius)
    if radius<=0:raise ValueError("extract at r>0 and take a regular parity limit to the axis")
    normalization=np.r_[np.full(7,-2.),np.ones(2)]
    zero=np.zeros((9,9));first=np.zeros((3,9,9));pure_second=np.zeros((3,9,9))
    for column in range(9):
        values=np.zeros(9);values[column]=1.
        zero[:,column]=normalization*_linear_column(
            background,regular_so3_perturbation_jets(radius,values),radius,
            mass_squared,potential_offset,kappa5_squared,
            constraint_damping,constraint_damping_rho,
        )
        for derivative in range(3):
            reduced_first=np.zeros((3,9));reduced_first[derivative,column]=1.
            first[derivative,:,column]=normalization*_linear_column(
                background,regular_so3_perturbation_jets(
                    radius,np.zeros(9),first=reduced_first,
                ),radius,mass_squared,potential_offset,kappa5_squared,
                constraint_damping,constraint_damping_rho,
            )
            reduced_second=np.zeros((3,3,9));reduced_second[derivative,derivative,column]=1.
            pure_second[derivative,:,column]=normalization*_linear_column(
                background,regular_so3_perturbation_jets(
                    radius,np.zeros(9),second=reduced_second,
                ),radius,mass_squared,potential_offset,kappa5_squared,
                constraint_damping,constraint_damping_rho,
            )
    inverse=np.linalg.inv(np.asarray(background["metric"],dtype=float))
    expected=np.array((inverse[0,0],inverse[1,1],inverse[2,2]))
    principal_defect=max(
        float(np.max(np.abs(pure_second[index]-expected[index]*np.eye(9))))
        for index in range(3)
    )
    alpha_squared=-1/np.asarray(background["metric"])[0,0]
    geometry=metric_geometry_from_jets(
        background["metric"],background["metric_first"],background["metric_second"],
    )
    contracted=np.asarray(geometry["contracted_christoffel_upper"],dtype=float)
    scalar_wave_first=np.array((
        -contracted[0],-contracted[1],2*inverse[3,3]/radius-contracted[2],
    ))
    lower_first=first-scalar_wave_first[:,None,None]*np.eye(9)[None,:,:]
    return {
        "field_order":FIELD_ORDER,"row_normalization":normalization,
        "zero_order_matrix":zero,"first_matrices":first,
        "pure_second_matrices":pure_second,
        "expected_scalar_principal":expected,
        "principal_identity_maximum_defect":principal_defect,
        "scalar_wave_first_coefficients":scalar_wave_first,
        "lower_first_matrices":lower_first,
        "evolution_reaction_matrix":-alpha_squared*zero,
        "evolution_first_matrices":alpha_squared*lower_first,
        "constraint_damping_rate":float(constraint_damping),
        "constraint_damping_rho":float(constraint_damping_rho),
        "finite":bool(np.all(np.isfinite(zero)) and np.all(np.isfinite(first))),
        "limitations":[
            "coefficient extraction at r>0; regular even parity supplies the axis limit",
            "frozen generalized-harmonic source",
            "evolved source driver absent",
        ],
    }
