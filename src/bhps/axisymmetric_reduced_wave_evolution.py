"""Axisymmetric variable-principal reduced-wave evolution.

The solver is a staged runtime audit for the corrected braneworld folds.  It
couples the normal and radial principal derivatives on a diagonal static
slice, while retaining the complete radius-dependent Israel--scalar Robin
matrix at both compact walls.  Coupled square lower-order fields and
rectangular source-driver maps are supported.  The runtime remains a
linearized audit and does not include nonlinear Einstein evolution.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu


def axisymmetric_principal_coefficients(psi,a,b,c,lapse=None):
    """Return reduced coefficients for the axisymmetric scalar wave operator.

    After the two-sphere integral the actual weak weights are ``r^2`` times

    ``w=A B C^2/alpha``, ``p_z=alpha B C^2/A``, and
    ``p_r=alpha A C^2/B``.
    """
    psi=np.asarray(psi,dtype=float);a=np.asarray(a,dtype=float)
    b=np.asarray(b,dtype=float);c=np.asarray(c,dtype=float)
    alpha=psi if lapse is None else np.asarray(lapse,dtype=float)
    if not (psi.shape==a.shape==b.shape==c.shape==alpha.shape) or psi.ndim!=2:
        raise ValueError("metric fields must be matching two-dimensional arrays")
    if np.any(psi<=0) or np.any(alpha<=0):raise ValueError("psi and lapse must be positive")
    aa=psi*np.exp(a);bb=psi*np.exp(b);cc=psi*np.exp(c)
    return {
        "mass_weight":aa*bb*cc**2/alpha,
        "z_gradient_weight":alpha*bb*cc**2/aa,
        "r_gradient_weight":alpha*aa*cc**2/bb,
        "z_boundary_weight":alpha*bb*cc**2,
        "z_coordinate_speed":alpha/aa,
        "r_coordinate_speed":alpha/bb,
        "A":aa,"B":bb,"C":cc,"lapse":alpha,
    }


def axisymmetric_bilinear_finite_element_matrices(
    z,r,mass_weight,z_gradient_weight,r_gradient_weight,
):
    """Assemble Q1 mass and stiffness matrices with the radial ``r^2`` measure."""
    z=np.asarray(z,dtype=float);r=np.asarray(r,dtype=float)
    w=np.asarray(mass_weight,dtype=float);pz=np.asarray(z_gradient_weight,dtype=float)
    pr=np.asarray(r_gradient_weight,dtype=float);shape=(len(z),len(r))
    if (
        z.ndim!=1 or r.ndim!=1 or len(z)<3 or len(r)<3
        or np.any(np.diff(z)<=0) or np.any(np.diff(r)<=0) or r[0]!=0
        or w.shape!=shape or pz.shape!=shape or pr.shape!=shape
        or np.any(w<=0) or np.any(pz<=0) or np.any(pr<=0)
    ):
        raise ValueError("invalid axisymmetric finite-element inputs")
    nz,nr=shape;nodes=nz*nr
    mass=lil_matrix((nodes,nodes));stiffness=lil_matrix((nodes,nodes))
    gauss=(-1/np.sqrt(3),1/np.sqrt(3))
    for i,dz in enumerate(np.diff(z)):
        for j,dr in enumerate(np.diff(r)):
            indices=np.array((i*nr+j,i*nr+j+1,(i+1)*nr+j,(i+1)*nr+j+1))
            local_mass=np.zeros((4,4));local_stiffness=np.zeros((4,4))
            nodal_w=w[np.ix_((i,i+1),(j,j+1))].reshape(-1)
            nodal_pz=pz[np.ix_((i,i+1),(j,j+1))].reshape(-1)
            nodal_pr=pr[np.ix_((i,i+1),(j,j+1))].reshape(-1)
            for xi in gauss:
                nz0=(1-xi)/2;nz1=(1+xi)/2
                for eta in gauss:
                    nr0=(1-eta)/2;nr1=(1+eta)/2
                    basis=np.array((nz0*nr0,nz0*nr1,nz1*nr0,nz1*nr1))
                    dz_basis=np.array((-nr0,-nr1,nr0,nr1))/dz
                    dr_basis=np.array((-nz0,nz0,-nz1,nz1))/dr
                    radius=nr0*r[j]+nr1*r[j+1];jacobian=dz*dr/4
                    local_w=float(basis@nodal_w)*radius**2
                    local_pz=float(basis@nodal_pz)*radius**2
                    local_pr=float(basis@nodal_pr)*radius**2
                    local_mass+=jacobian*local_w*np.outer(basis,basis)
                    local_stiffness+=jacobian*(
                        local_pz*np.outer(dz_basis,dz_basis)
                        +local_pr*np.outer(dr_basis,dr_basis)
                    )
            for row,global_row in enumerate(indices):
                for column,global_column in enumerate(indices):
                    mass[global_row,global_column]+=local_mass[row,column]
                    stiffness[global_row,global_column]+=local_stiffness[row,column]
    return mass.tocsr(),stiffness.tocsr()


def axisymmetric_bilinear_reaction_matrix(z,r,reaction_weight):
    """Assemble ``integral r^2 reaction_weight u v dz dr`` with Q1 fields."""
    z=np.asarray(z,dtype=float);r=np.asarray(r,dtype=float)
    reaction=np.asarray(reaction_weight,dtype=float);shape=(len(z),len(r))
    if (
        z.ndim!=1 or r.ndim!=1 or len(z)<3 or len(r)<3
        or np.any(np.diff(z)<=0) or np.any(np.diff(r)<=0) or r[0]!=0
        or reaction.shape!=shape or not np.all(np.isfinite(reaction))
    ):
        raise ValueError("invalid axisymmetric reaction inputs")
    nz,nr=shape;matrix=lil_matrix((nz*nr,nz*nr));gauss=(-1/np.sqrt(3),1/np.sqrt(3))
    for i,dz in enumerate(np.diff(z)):
        for j,dr in enumerate(np.diff(r)):
            indices=np.array((i*nr+j,i*nr+j+1,(i+1)*nr+j,(i+1)*nr+j+1))
            nodal=reaction[np.ix_((i,i+1),(j,j+1))].reshape(-1);local=np.zeros((4,4))
            for xi in gauss:
                nz0=(1-xi)/2;nz1=(1+xi)/2
                for eta in gauss:
                    nr0=(1-eta)/2;nr1=(1+eta)/2
                    basis=np.array((nz0*nr0,nz0*nr1,nz1*nr0,nz1*nr1))
                    radius=nr0*r[j]+nr1*r[j+1]
                    local+=dz*dr/4*radius**2*float(basis@nodal)*np.outer(basis,basis)
            for row,global_row in enumerate(indices):
                for column,global_column in enumerate(indices):
                    matrix[global_row,global_column]+=local[row,column]
    return matrix.tocsr()


def axisymmetric_wall_reaction_matrix(r,left_weight,right_weight):
    """Assemble diagonal-in-field endpoint reaction weights over ``r``."""
    r=np.asarray(r,dtype=float);left=np.asarray(left_weight,dtype=float)
    right=np.asarray(right_weight,dtype=float)
    if r.ndim!=1 or len(r)<3 or r[0]!=0 or np.any(np.diff(r)<=0) or left.shape!=r.shape or right.shape!=r.shape:
        raise ValueError("invalid wall reaction inputs")
    nr=len(r);matrix=lil_matrix((2*nr,2*nr));gauss=(-1/np.sqrt(3),1/np.sqrt(3))
    for wall,weights in enumerate((left,right)):
        offset=wall*nr
        for j,dr in enumerate(np.diff(r)):
            local=np.zeros((2,2))
            for eta in gauss:
                n0=(1-eta)/2;n1=(1+eta)/2
                basis=np.array((n0,n1));radius=n0*r[j]+n1*r[j+1]
                local+=dr/2*radius**2*float(n0*weights[j]+n1*weights[j+1])*np.outer(basis,basis)
            for row in range(2):
                for column in range(2):matrix[offset+j+row,offset+j+column]+=local[row,column]
    return matrix.tocsr()


def axisymmetric_coupled_lower_order_matrices(
    z,r,mass_weight,reaction_matrix,evolution_first_matrices,
    radial_first_is_scaled=False,
):
    """Assemble coupled value, velocity, z-, and r-derivative weak matrices.

    ``reaction_matrix`` is the pointwise matrix ``M`` in
    ``u_tt=principal-Mu+...``.  ``evolution_first_matrices`` has leading
    direction order ``(t,z,r)`` and supplies the remaining acceleration terms.
    The common scalar-wave connection is already contained in the divergence
    principal operator and must not be included again.
    """
    from scipy.sparse import coo_matrix

    z=np.asarray(z,dtype=float);r=np.asarray(r,dtype=float);weight=np.asarray(mass_weight,dtype=float)
    reaction=np.asarray(reaction_matrix,dtype=float);first=np.asarray(evolution_first_matrices,dtype=float)
    shape=(len(z),len(r));fields=reaction.shape[-1] if reaction.ndim==4 else 0
    if (
        weight.shape!=shape or reaction.shape!=(len(z),len(r),fields,fields)
        or first.shape!=(3,len(z),len(r),fields,fields) or fields<1
        or not np.all(np.isfinite(reaction)) or not np.all(np.isfinite(first))
    ):
        raise ValueError("invalid coupled lower-order coefficient fields")
    nodes=len(z)*len(r);size=nodes*fields
    row_entries=[[] for _ in range(4)];column_entries=[[] for _ in range(4)]
    data_entries=[[] for _ in range(4)]
    gauss=(-1/np.sqrt(3),1/np.sqrt(3))
    for i,dz in enumerate(np.diff(z)):
        for j,dr in enumerate(np.diff(r)):
            node_indices=np.array((i*len(r)+j,i*len(r)+j+1,(i+1)*len(r)+j,(i+1)*len(r)+j+1))
            nodal_weight=weight[np.ix_((i,i+1),(j,j+1))].reshape(4)
            nodal_reaction=reaction[np.ix_((i,i+1),(j,j+1))].reshape(4,fields,fields)
            nodal_first=first[:,i:i+2,j:j+2].reshape(3,4,fields,fields)
            local_matrices=np.zeros((4,4,4,fields,fields))
            for xi in gauss:
                nz0=(1-xi)/2;nz1=(1+xi)/2
                for eta in gauss:
                    nr0=(1-eta)/2;nr1=(1+eta)/2
                    basis=np.array((nz0*nr0,nz0*nr1,nz1*nr0,nz1*nr1))
                    dz_basis=np.array((-nr0,-nr1,nr0,nr1))/dz
                    dr_basis=np.array((-nz0,nz0,-nz1,nz1))/dr
                    radius=nr0*r[j]+nr1*r[j+1];jacobian=dz*dr/4*radius**2
                    local_weight=float(basis@nodal_weight)
                    coefficients=[
                        np.einsum("n,nab->ab",basis,nodal_reaction),
                        np.einsum("n,nab->ab",basis,nodal_first[0]),
                        np.einsum("n,nab->ab",basis,nodal_first[1]),
                        np.einsum("n,nab->ab",basis,nodal_first[2]),
                    ]
                    if radial_first_is_scaled:
                        coefficients[3]=coefficients[3]/radius
                    trial_shapes=(basis,basis,dz_basis,dr_basis)
                    for test_local,test_node in enumerate(node_indices):
                        for trial_local,trial_node in enumerate(node_indices):
                            for matrix_index,(coefficient,trial_shape) in enumerate(zip(coefficients,trial_shapes)):
                                local_matrices[matrix_index,test_local,trial_local]+=jacobian*local_weight*basis[test_local]*trial_shape[trial_local]*coefficient
            for test_local,test_node in enumerate(node_indices):
                rows=test_node*fields+np.arange(fields)
                for trial_local,trial_node in enumerate(node_indices):
                    columns=trial_node*fields+np.arange(fields)
                    for matrix_index in range(4):
                        row_entries[matrix_index].append(np.repeat(rows,fields))
                        column_entries[matrix_index].append(np.tile(columns,fields))
                        data_entries[matrix_index].append(
                            local_matrices[matrix_index,test_local,trial_local].ravel()
                        )
    matrices=[]
    for rows,columns,data in zip(row_entries,column_entries,data_entries):
        matrices.append(coo_matrix((
            np.concatenate(data),(np.concatenate(rows),np.concatenate(columns)),
        ),shape=(size,size)).tocsr())
    return {
        "reaction":matrices[0],"time_first":matrices[1],
        "z_first":matrices[2],"r_first":matrices[3],
    }


def axisymmetric_rectangular_lower_order_matrices(
    z,r,mass_weight,reaction_matrix,evolution_first_matrices,
    radial_first_is_scaled=False,
):
    """Assemble value/first maps between different input and output fields.

    This is the rectangular counterpart of
    :func:`axisymmetric_coupled_lower_order_matrices`.  It is used for the
    ``9 <- 3`` generalized-harmonic source coupling into the wave equations.
    """
    from scipy.sparse import coo_matrix

    z=np.asarray(z,dtype=float);r=np.asarray(r,dtype=float);weight=np.asarray(mass_weight,dtype=float)
    reaction=np.asarray(reaction_matrix,dtype=float);first=np.asarray(evolution_first_matrices,dtype=float)
    shape=(len(z),len(r));outputs=reaction.shape[-2] if reaction.ndim==4 else 0
    inputs=reaction.shape[-1] if reaction.ndim==4 else 0
    if (
        weight.shape!=shape or reaction.shape!=(len(z),len(r),outputs,inputs)
        or first.shape!=(3,len(z),len(r),outputs,inputs) or outputs<1 or inputs<1
        or not np.all(np.isfinite(reaction)) or not np.all(np.isfinite(first))
    ):
        raise ValueError("invalid rectangular lower-order coefficient fields")
    row_entries=[[] for _ in range(4)];column_entries=[[] for _ in range(4)]
    data_entries=[[] for _ in range(4)];gauss=(-1/np.sqrt(3),1/np.sqrt(3))
    for i,dz in enumerate(np.diff(z)):
        for j,dr in enumerate(np.diff(r)):
            node_indices=np.array((i*len(r)+j,i*len(r)+j+1,(i+1)*len(r)+j,(i+1)*len(r)+j+1))
            nodal_weight=weight[np.ix_((i,i+1),(j,j+1))].reshape(4)
            nodal_reaction=reaction[np.ix_((i,i+1),(j,j+1))].reshape(4,outputs,inputs)
            nodal_first=first[:,i:i+2,j:j+2].reshape(3,4,outputs,inputs)
            local_matrices=np.zeros((4,4,4,outputs,inputs))
            for xi in gauss:
                nz0=(1-xi)/2;nz1=(1+xi)/2
                for eta in gauss:
                    nr0=(1-eta)/2;nr1=(1+eta)/2
                    basis=np.array((nz0*nr0,nz0*nr1,nz1*nr0,nz1*nr1))
                    dz_basis=np.array((-nr0,-nr1,nr0,nr1))/dz
                    dr_basis=np.array((-nz0,nz0,-nz1,nz1))/dr
                    radius=nr0*r[j]+nr1*r[j+1];jacobian=dz*dr/4*radius**2
                    local_weight=float(basis@nodal_weight)
                    coefficients=[
                        np.einsum("n,nab->ab",basis,nodal_reaction),
                        np.einsum("n,nab->ab",basis,nodal_first[0]),
                        np.einsum("n,nab->ab",basis,nodal_first[1]),
                        np.einsum("n,nab->ab",basis,nodal_first[2]),
                    ]
                    if radial_first_is_scaled:coefficients[3]=coefficients[3]/radius
                    trial_shapes=(basis,basis,dz_basis,dr_basis)
                    for test_local in range(4):
                        for trial_local in range(4):
                            for matrix_index,(coefficient,trial_shape) in enumerate(zip(coefficients,trial_shapes)):
                                local_matrices[matrix_index,test_local,trial_local]+=jacobian*local_weight*basis[test_local]*trial_shape[trial_local]*coefficient
            for test_local,test_node in enumerate(node_indices):
                rows=test_node*outputs+np.arange(outputs)
                for trial_local,trial_node in enumerate(node_indices):
                    columns=trial_node*inputs+np.arange(inputs)
                    for matrix_index in range(4):
                        row_entries[matrix_index].append(np.repeat(rows,inputs))
                        column_entries[matrix_index].append(np.tile(columns,outputs))
                        data_entries[matrix_index].append(local_matrices[matrix_index,test_local,trial_local].ravel())
    matrices=[]
    for rows,columns,data in zip(row_entries,column_entries,data_entries):
        matrices.append(coo_matrix((
            np.concatenate(data),(np.concatenate(rows),np.concatenate(columns)),
        ),shape=(len(z)*len(r)*outputs,len(z)*len(r)*inputs)).tocsr())
    return {
        "reaction":matrices[0],"time_first":matrices[1],
        "z_first":matrices[2],"r_first":matrices[3],
    }


class AxisymmetricVariableReducedWaveIBVP:
    """Q1/RK4 axisymmetric principal evolution for the complete wave block."""

    def __init__(
        self,z,r,mass_weight,z_gradient_weight,r_gradient_weight,
        left_robin,right_robin,left_boundary_weight,right_boundary_weight,
        dirichlet_fields=4,reaction_weights=None,
        coupled_reaction_matrices=None,evolution_first_matrices=None,
        radial_first_is_scaled=False,outer_dirichlet=True,velocity_diffusion=0.,
    ):
        self.z=np.asarray(z,dtype=float);self.r=np.asarray(r,dtype=float)
        self.nz=len(self.z);self.nr=len(self.r);self.nodes=self.nz*self.nr
        self.mass_weight=np.asarray(mass_weight,dtype=float)
        self.z_gradient_weight=np.asarray(z_gradient_weight,dtype=float)
        self.r_gradient_weight=np.asarray(r_gradient_weight,dtype=float)
        self.mass,self.stiffness=axisymmetric_bilinear_finite_element_matrices(
            self.z,self.r,self.mass_weight,self.z_gradient_weight,self.r_gradient_weight,
        )
        self.lumped_mass=np.asarray(self.mass.sum(axis=1)).ravel()
        if np.any(self.lumped_mass<=0):raise ValueError("lumped mass must be positive")
        self.left_robin=np.asarray(left_robin,dtype=float)
        self.right_robin=np.asarray(right_robin,dtype=float)
        self.left_boundary_weight=np.asarray(left_boundary_weight,dtype=float)
        self.right_boundary_weight=np.asarray(right_boundary_weight,dtype=float)
        if (
            self.left_robin.ndim!=3 or self.left_robin.shape[0]!=self.nr
            or self.left_robin.shape[1]!=self.left_robin.shape[2]
            or self.right_robin.shape!=self.left_robin.shape
            or self.left_boundary_weight.shape!=(self.nr,)
            or self.right_boundary_weight.shape!=(self.nr,)
            or np.any(self.left_boundary_weight<=0) or np.any(self.right_boundary_weight<=0)
        ):
            raise ValueError("invalid radius-dependent wall data")
        self.dirichlet_fields=int(dirichlet_fields);self.robin_fields=self.left_robin.shape[1]
        self.field_count=self.dirichlet_fields+self.robin_fields
        if self.dirichlet_fields<0:raise ValueError("dirichlet_fields must be nonnegative")
        if reaction_weights is None:
            self.reaction_weights=np.zeros((self.nz,self.nr,self.field_count))
        else:
            self.reaction_weights=np.asarray(reaction_weights,dtype=float)
            if self.reaction_weights.shape!=(self.nz,self.nr,self.field_count):
                raise ValueError("reaction weights have the wrong shape")
        self.reaction_matrices=[]
        for field in range(self.field_count):
            local=self.reaction_weights[:,:,field]
            self.reaction_matrices.append(
                None if np.max(np.abs(local))==0 else
                axisymmetric_bilinear_reaction_matrix(self.z,self.r,local)
            )
        if coupled_reaction_matrices is None and evolution_first_matrices is None:
            self.coupled_lower_order=None
        else:
            if coupled_reaction_matrices is None or evolution_first_matrices is None:
                raise ValueError("both coupled reaction and first-derivative fields are required")
            coupled=np.asarray(coupled_reaction_matrices,dtype=float)
            first_matrices=np.asarray(evolution_first_matrices,dtype=float)
            if coupled.shape!=(self.nz,self.nr,self.field_count,self.field_count):
                raise ValueError("coupled reaction field has the wrong shape")
            self.coupled_lower_order=axisymmetric_coupled_lower_order_matrices(
                self.z,self.r,self.mass_weight,coupled,first_matrices,
                radial_first_is_scaled=radial_first_is_scaled,
            )
        self.outer_dirichlet=bool(outer_dirichlet)
        self.velocity_diffusion=float(velocity_diffusion)
        if not np.isfinite(self.velocity_diffusion) or self.velocity_diffusion<0:
            raise ValueError("velocity diffusion must be finite and nonnegative")
        indices=np.arange(self.nodes).reshape(self.nz,self.nr)
        outer=np.zeros((self.nz,self.nr),dtype=bool);outer[:,-1]=True
        compact=np.zeros((self.nz,self.nr),dtype=bool);compact[[0,-1],:]=True
        gauge_fixed=compact | outer if self.outer_dirichlet else compact
        robin_fixed=outer if self.outer_dirichlet else np.zeros_like(outer)
        self.gauge_free=indices[~gauge_fixed].ravel()
        self.robin_free=indices[~robin_fixed].ravel()
        self.gauge_fixed=np.flatnonzero(gauge_fixed.ravel())
        self.robin_fixed=np.flatnonzero(robin_fixed.ravel())
        self._gauge_mass=splu(self.mass[self.gauge_free][:,self.gauge_free].tocsc()) if self.dirichlet_fields else None
        self._robin_mass=splu(self.mass[self.robin_free][:,self.robin_free].tocsc())

    def _source(self,source,time):
        if source is None:return np.zeros((self.nodes,self.field_count))
        zz,rr=np.meshgrid(self.z,self.r,indexing="ij")
        values=np.asarray(source(float(time),zz,rr),dtype=float)
        if values.shape!=(self.nz,self.nr,self.field_count):
            raise ValueError("volume source has the wrong shape")
        return values.reshape(self.nodes,self.field_count)

    def _boundary_values(self,source,time):
        if source is None:return np.zeros((self.nr,self.robin_fields))
        values=np.asarray(source(float(time),self.r),dtype=float)
        if values.shape!=(self.nr,self.robin_fields):
            raise ValueError("wall source has the wrong shape")
        return values

    def _wall_load(self,position,time,left_data,right_data):
        q=position.reshape(self.nz,self.nr,self.field_count)[:,:,self.dirichlet_fields:]
        load=np.zeros((self.nodes,self.robin_fields))
        gauss=(-1/np.sqrt(3),1/np.sqrt(3))
        sources=(self._boundary_values(left_data,time),self._boundary_values(right_data,time))
        for wall_index,(matrix,weight,data) in enumerate((
            (self.left_robin,self.left_boundary_weight,sources[0]),
            (self.right_robin,self.right_boundary_weight,sources[1]),
        )):
            z_index=0 if wall_index==0 else self.nz-1
            for j,spacing in enumerate(np.diff(self.r)):
                local=np.zeros((2,self.robin_fields))
                for eta in gauss:
                    n0=(1-eta)/2;n1=(1+eta)/2
                    radius=n0*self.r[j]+n1*self.r[j+1]
                    local_matrix=n0*matrix[j]+n1*matrix[j+1]
                    local_weight=n0*weight[j]+n1*weight[j+1]
                    local_q=n0*q[z_index,j]+n1*q[z_index,j+1]
                    local_data=n0*data[j]+n1*data[j+1]
                    flux=radius**2*local_weight*(local_matrix@local_q+local_data)
                    local[0]+=spacing/2*n0*flux;local[1]+=spacing/2*n1*flux
                load[z_index*self.nr+j]+=local[0]
                load[z_index*self.nr+j+1]+=local[1]
        return load

    def outer_radial_flux_load(self,flux,active=None):
        """Assemble a Q1 weak load on the artificial radial face.

        ``flux`` is the physical radial flux coefficient multiplying the
        reduced ``r^2`` surface measure.  The optional nodal ``active`` mask
        removes corner test rows so the compact-wall equations retain
        precedence there.
        """
        values=np.asarray(flux,dtype=float)
        if values.shape!=(self.nz,self.field_count) or not np.all(np.isfinite(values)):
            raise ValueError("outer radial flux has the wrong shape")
        if active is None:mask=np.ones(self.nz,dtype=bool)
        else:
            mask=np.asarray(active,dtype=bool)
            if mask.shape!=(self.nz,):raise ValueError("outer radial flux mask has the wrong shape")
        load=np.zeros((self.nodes,self.field_count));radius=float(self.r[-1])
        gauss=(-1/np.sqrt(3),1/np.sqrt(3))
        for i,spacing in enumerate(np.diff(self.z)):
            local=np.zeros((2,self.field_count))
            for xi in gauss:
                n0=(1-xi)/2;n1=(1+xi)/2
                local_flux=n0*values[i]+n1*values[i+1]
                local[0]+=spacing/2*n0*radius**2*local_flux
                local[1]+=spacing/2*n1*radius**2*local_flux
            load[i*self.nr+self.nr-1]+=local[0]
            load[(i+1)*self.nr+self.nr-1]+=local[1]
        outer=np.arange(self.nodes).reshape(self.nz,self.nr)[:,-1]
        load[outer[~mask]]=0.
        return load

    def acceleration(
        self,time,position,source=None,left_boundary_data=None,right_boundary_data=None,
        velocity=None,
    ):
        position=np.asarray(position,dtype=float)
        expected=(self.nz,self.nr,self.field_count)
        if position.shape!=expected:raise ValueError("position has the wrong shape")
        q=position.reshape(self.nodes,self.field_count)
        load=-self.stiffness@q+self.mass@self._source(source,time)
        for field,reaction in enumerate(self.reaction_matrices):
            if reaction is not None:load[:,field]-=reaction@q[:,field]
        flat_v=None
        if self.coupled_lower_order is not None:
            if velocity is None:raise ValueError("velocity is required by coupled lower-order terms")
            velocity=np.asarray(velocity,dtype=float)
            if velocity.shape!=(self.nz,self.nr,self.field_count):
                raise ValueError("velocity has the wrong shape")
            flat_q=q.ravel();flat_v=velocity.reshape(self.nodes,self.field_count).ravel()
            lower=(
                -self.coupled_lower_order["reaction"]@flat_q
                +self.coupled_lower_order["time_first"]@flat_v
                +self.coupled_lower_order["z_first"]@flat_q
                +self.coupled_lower_order["r_first"]@flat_q
            )
            load+=lower.reshape(self.nodes,self.field_count)
        if self.velocity_diffusion:
            if velocity is None:raise ValueError("velocity is required by velocity diffusion")
            velocity=np.asarray(velocity,dtype=float)
            if velocity.shape!=(self.nz,self.nr,self.field_count):
                raise ValueError("velocity has the wrong shape")
            nodal_velocity=velocity.reshape(self.nodes,self.field_count)
            load-=self.velocity_diffusion*(self.stiffness@nodal_velocity)
        load[:,self.dirichlet_fields:]+=self._wall_load(
            position,time,left_boundary_data,right_boundary_data,
        )
        result=np.zeros_like(q)
        if self.dirichlet_fields:
            result[self.gauge_free,:self.dirichlet_fields]=self._gauge_mass.solve(
                load[self.gauge_free,:self.dirichlet_fields]
            )
        result[self.robin_free,self.dirichlet_fields:]=self._robin_mass.solve(
            load[self.robin_free,self.dirichlet_fields:]
        )
        return result.reshape(expected)

    def integrate(
        self,initial_position,initial_velocity,final_time,courant=.06,
        source=None,left_boundary_data=None,right_boundary_data=None,diagnostic=None,
    ):
        q=np.asarray(initial_position,dtype=float).copy();v=np.asarray(initial_velocity,dtype=float).copy()
        expected=(self.nz,self.nr,self.field_count)
        if q.shape!=expected or v.shape!=expected:raise ValueError("initial state has the wrong shape")
        flat_q=q.reshape(self.nodes,self.field_count);flat_v=v.reshape(self.nodes,self.field_count)
        if self.dirichlet_fields and (
            np.max(np.abs(flat_q[self.gauge_fixed,:self.dirichlet_fields]))>1e-13
            or np.max(np.abs(flat_v[self.gauge_fixed,:self.dirichlet_fields]))>1e-13
        ):raise ValueError("gauge Dirichlet data must vanish")
        if self.robin_fixed.size and (
            np.max(np.abs(flat_q[self.robin_fixed,self.dirichlet_fields:]))>1e-13
            or np.max(np.abs(flat_v[self.robin_fixed,self.dirichlet_fields:]))>1e-13
        ):raise ValueError("outer Robin-block Dirichlet data must vanish")
        duration=float(final_time);courant=float(courant)
        if duration<0 or courant<=0:raise ValueError("invalid integration controls")
        spacing=min(np.min(np.diff(self.z)),np.min(np.diff(self.r)))
        steps=max(1,int(np.ceil(duration/(courant*spacing))))
        dt=duration/steps if duration else 0.;records=[];time=0.
        rhs=lambda t,x,y:(y,self.acceleration(
            t,x,source,left_boundary_data,right_boundary_data,velocity=y,
        ))
        if diagnostic is not None:records.append(diagnostic(time,q,v))
        for _ in range(steps):
            k1q,k1v=rhs(time,q,v)
            k2q,k2v=rhs(time+dt/2,q+dt*k1q/2,v+dt*k1v/2)
            k3q,k3v=rhs(time+dt/2,q+dt*k2q/2,v+dt*k2v/2)
            k4q,k4v=rhs(time+dt,q+dt*k3q,v+dt*k3v)
            q+=dt*(k1q+2*k2q+2*k3q+k4q)/6
            v+=dt*(k1v+2*k2v+2*k3v+k4v)/6;time+=dt
            flat_q=q.reshape(self.nodes,self.field_count);flat_v=v.reshape(self.nodes,self.field_count)
            if self.dirichlet_fields:
                flat_q[self.gauge_fixed,:self.dirichlet_fields]=0.
                flat_v[self.gauge_fixed,:self.dirichlet_fields]=0.
            flat_q[self.robin_fixed,self.dirichlet_fields:]=0.
            flat_v[self.robin_fixed,self.dirichlet_fields:]=0.
            if diagnostic is not None:records.append(diagnostic(time,q,v))
        return {"time":time,"position":q,"velocity":v,"steps":steps,"time_step":dt,"diagnostics":records}

    def l2_norm(self,values):
        values=np.asarray(values,dtype=float)
        if values.shape!=(self.nz,self.nr,self.field_count):raise ValueError("values have the wrong shape")
        flat=values.reshape(self.nodes,self.field_count)
        return float(np.sqrt(max(0.,np.sum((self.mass@flat)*flat))))

    def interpolated_symmetrizer_energy(
        self,position,velocity,left_symmetrizer,right_symmetrizer,shift=0.,
    ):
        """Return the two-dimensional energy and its predicted collar power."""
        q=np.asarray(position,dtype=float);v=np.asarray(velocity,dtype=float)
        if q.shape!=(self.nz,self.nr,self.field_count) or v.shape!=q.shape:
            raise ValueError("state has the wrong shape")
        wl=np.asarray(left_symmetrizer,dtype=float);wr=np.asarray(right_symmetrizer,dtype=float)
        if wl.shape!=(self.nr,self.robin_fields,self.robin_fields) or wr.shape!=wl.shape:raise ValueError("symmetrizers have the wrong shape")
        first=self.dirichlet_fields;flat_q=q.reshape(self.nodes,self.field_count);flat_v=v.reshape(self.nodes,self.field_count)
        kinetic=.5*float(np.sum((self.mass@flat_v[:,:first])*flat_v[:,:first]))
        gradient=.5*float(np.sum((self.stiffness@flat_q[:,:first])*flat_q[:,:first]))
        mass_shift=power=shift_power=0.;qr=q[:,:,first:];vr=v[:,:,first:]
        gauss=(-1/np.sqrt(3),1/np.sqrt(3));length=self.z[-1]-self.z[0]
        for i,dz in enumerate(np.diff(self.z)):
            for j,dr in enumerate(np.diff(self.r)):
                qn=qr[np.ix_((i,i+1),(j,j+1))].reshape(4,self.robin_fields)
                vn=vr[np.ix_((i,i+1),(j,j+1))].reshape(4,self.robin_fields)
                wn=self.mass_weight[np.ix_((i,i+1),(j,j+1))].reshape(4)
                pzn=self.z_gradient_weight[np.ix_((i,i+1),(j,j+1))].reshape(4)
                prn=self.r_gradient_weight[np.ix_((i,i+1),(j,j+1))].reshape(4)
                for xi in gauss:
                    nz0=(1-xi)/2;nz1=(1+xi)/2
                    for eta in gauss:
                        nr0=(1-eta)/2;nr1=(1+eta)/2
                        basis=np.array((nz0*nr0,nz0*nr1,nz1*nr0,nz1*nr1))
                        dz_basis=np.array((-nr0,-nr1,nr0,nr1))/dz
                        dr_basis=np.array((-nz0,nz0,-nz1,nz1))/dr
                        radius=nr0*self.r[j]+nr1*self.r[j+1]
                        coordinate=nz0*self.z[i]+nz1*self.z[i+1]
                        fraction=(coordinate-self.z[0])/length
                        local_wl=nr0*wl[j]+nr1*wl[j+1]
                        local_wr=nr0*wr[j]+nr1*wr[j+1]
                        weight=(1-fraction)*local_wl+fraction*local_wr
                        weight_z=(local_wr-local_wl)/length
                        weight_r=(
                            (1-fraction)*(wl[j+1]-wl[j])/dr
                            +fraction*(wr[j+1]-wr[j])/dr
                        )
                        local_q=basis@qn;local_v=basis@vn
                        qz=dz_basis@qn;qr_derivative=dr_basis@qn
                        w=float(basis@wn);pz=float(basis@pzn);pr=float(basis@prn)
                        jacobian=dz*dr/4*radius**2
                        kinetic+=.5*jacobian*w*float(local_v@weight@local_v)
                        gradient+=.5*jacobian*(
                            pz*float(qz@weight@qz)+pr*float(qr_derivative@weight@qr_derivative)
                        )
                        mass_shift+=.5*float(shift)*jacobian*w*float(local_q@weight@local_q)
                        power-=jacobian*(
                            pz*float(local_v@weight_z@qz)
                            +pr*float(local_v@weight_r@qr_derivative)
                        )
                        shift_power+=float(shift)*jacobian*w*float(local_q@weight@local_v)
        boundary=0.
        for wall_index,(matrix,surface,weights) in enumerate((
            (self.left_robin,self.left_boundary_weight,wl),
            (self.right_robin,self.right_boundary_weight,wr),
        )):
            zi=0 if wall_index==0 else -1
            for j,dr in enumerate(np.diff(self.r)):
                for eta in gauss:
                    n0=(1-eta)/2;n1=(1+eta)/2
                    radius=n0*self.r[j]+n1*self.r[j+1]
                    local_q=n0*qr[zi,j]+n1*qr[zi,j+1]
                    local_matrix=n0*matrix[j]+n1*matrix[j+1]
                    local_surface=n0*surface[j]+n1*surface[j+1]
                    local_weight=n0*weights[j]+n1*weights[j+1]
                    boundary-=.5*dr/2*radius**2*local_surface*float(
                        local_q@local_weight@local_matrix@local_q
                    )
        base=kinetic+gradient+boundary
        reaction_energy=0.
        for field,reaction in enumerate(self.reaction_matrices):
            if reaction is not None:
                local=flat_q[:,field]
                reaction_energy+=.5*float(local@reaction@local)
        base+=reaction_energy
        return {
            "base":float(base),"shifted":float(base+mass_shift),
            "kinetic":float(kinetic),"gradient":float(gradient),"boundary":float(boundary),
            "mass_shift":float(mass_shift),"predicted_base_energy_power":float(power),
            "predicted_shifted_energy_power":float(power+shift_power),
            "reaction":float(reaction_energy),
        }

    def scalar_spatial_operator(self,field_index):
        """Return the symmetric spatial operator for one uncoupled field.

        This helper requires diagonal wall matrices for the selected field and
        is used by fixed-background scalar spectral audits.
        """
        field=int(field_index)
        if not 0<=field<self.field_count:raise ValueError("invalid field index")
        if field<self.dirichlet_fields:
            free=self.gauge_free
        else:
            free=self.robin_free;local_field=field-self.dirichlet_fields
        operator=self.stiffness.copy().tolil()
        if self.reaction_matrices[field] is not None:operator+=self.reaction_matrices[field]
        if field>=self.dirichlet_fields:
            off_left=self.left_robin.copy();off_right=self.right_robin.copy()
            for array in (off_left,off_right):
                array[:,local_field,local_field]=0.
            if np.max(np.abs(off_left[:,:,local_field]))>1e-13 or np.max(np.abs(off_left[:,local_field,:]))>1e-13:
                raise ValueError("selected field is coupled at the left wall")
            if np.max(np.abs(off_right[:,:,local_field]))>1e-13 or np.max(np.abs(off_right[:,local_field,:]))>1e-13:
                raise ValueError("selected field is coupled at the right wall")
            left=-self.left_boundary_weight*self.left_robin[:,local_field,local_field]
            right=-self.right_boundary_weight*self.right_robin[:,local_field,local_field]
            boundary=axisymmetric_wall_reaction_matrix(self.r,left,right)
            indices=np.concatenate((np.arange(self.nr),(self.nz-1)*self.nr+np.arange(self.nr)))
            for row,global_row in enumerate(indices):
                for pointer in range(boundary.indptr[row],boundary.indptr[row+1]):
                    operator[global_row,indices[boundary.indices[pointer]]]+=boundary.data[pointer]
        operator=operator.tocsr()
        return {"operator":operator[free][:,free],"mass":self.mass[free][:,free],"free_indices":free}
