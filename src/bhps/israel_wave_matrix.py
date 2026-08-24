"""Frozen wall-adapted Einstein--two-scalar boundary matrix.

The construction generalizes the umbilic wave-gauge boundary rows of
Fournodavlos--Smulevici to the principal/lower-order frozen linearization
needed by this project.  It is an audit model, not a nonlinear theorem.
"""

from __future__ import annotations

import numpy as np


def coupled_robin_matrix(
    umbilic_coefficient,
    umbilic_phi_derivative,
    inward_phi_derivative,
    wall_stiffness,
    kappa5_squared=1.0,
):
    """Return ``R`` in ``partial_n u=R u`` for 10 tangential metric
    components, the normal-normal component, the stabilizer, and collapse
    scalar.

    The local frame is orthonormal and the boundary has four dimensions.
    The final collapse-scalar row is homogeneous Neumann.
    """
    c=float(umbilic_coefficient);c_phi=float(umbilic_phi_derivative)
    phi_x=float(inward_phi_derivative);gamma=float(wall_stiffness)
    kappa=float(kappa5_squared)
    if gamma<0 or kappa<=0:raise ValueError("require gamma>=0 and kappa5_squared>0")
    # Independent symmetric tangential pairs, with diagonal entries first.
    pairs=((0,0),(1,1),(2,2),(3,3),(0,1),(0,2),(0,3),(1,2),(1,3),(2,3))
    eta=np.array((-1.,1.,1.,1.));matrix=np.zeros((13,13))
    for row,(a,b) in enumerate(pairs):
        matrix[row,row]=-2*c
        if a==b:
            matrix[row,10]=-c*eta[a]
            matrix[row,11]=-2*c_phi*eta[a]
    # Wave-coordinate normal row and linearized scalar half-Robin row.
    matrix[10,10]=-12*c;matrix[10,11]=-8*c_phi
    matrix[11,10]=-.5*phi_x;matrix[11,11]=-gamma/2
    # Collapse scalar chi has reflecting Neumann data: R_chi=0.
    matrix[12,12]=0.
    return {"matrix":matrix,"field_order":[
        "h00","h11","h22","h33","h01","h02","h03","h12","h13","h23",
        "h_nn","delta_Phi","delta_chi",
    ]}


def matrix_spectral_audit(matrix,tolerance=1e-9):
    """Diagnose real diagonalizability and construct a positive symmetrizer."""
    matrix=np.asarray(matrix,dtype=float)
    if matrix.ndim!=2 or matrix.shape[0]!=matrix.shape[1]:raise ValueError("matrix must be square")
    values,vectors=np.linalg.eig(matrix)
    real=bool(np.max(np.abs(values.imag))<tolerance)
    rank=int(np.linalg.matrix_rank(vectors,tol=tolerance))
    diagonalizable=bool(rank==len(values))
    if not (real and diagonalizable):
        return {"eigenvalues":values,"all_real":real,"diagonalizable":diagonalizable,"symmetrizer":None}
    vectors=np.real(vectors);inverse=np.linalg.inv(vectors)
    symmetrizer=inverse.T@inverse
    defect=symmetrizer@matrix-matrix.T@symmetrizer
    symmetrizer_values=np.linalg.eigvalsh(symmetrizer)
    return {
        "eigenvalues":np.real(values),"all_real":True,"diagonalizable":True,
        "maximum_positive_eigenvalue":float(max(0.,np.max(np.real(values)))),
        "symmetrizer":symmetrizer,
        "symmetrizer_minimum_eigenvalue":float(symmetrizer_values[0]),
        "symmetrizer_condition_number":float(symmetrizer_values[-1]/symmetrizer_values[0]),
        "symmetry_defect":float(np.linalg.norm(defect,ord=2)),
    }


def analytic_robin_symmetrizer(
    umbilic_coefficient,
    umbilic_phi_derivative,
    inward_phi_derivative,
    wall_stiffness,
    kappa5_squared=1.0,
    separation_tolerance=1e-10,
):
    """Construct a smooth positive symmetrizer for the 13-field Robin block.

    The ten tangential components share eigenvalue ``a=-2c``.  A shear removes
    their coupling to the ``(h_nn,delta_Phi)`` block whenever ``a`` is
    spectrally separated from that block.  The physical Israel/scalar
    coefficients obey ``b/d=16*kappa5_squared/3``, giving a constant positive
    diagonal symmetrizer for the remaining two-by-two block.
    """
    kappa=float(kappa5_squared)
    if kappa<=0:raise ValueError("kappa5_squared must be positive")
    c_phi=float(umbilic_phi_derivative);phi_x=float(inward_phi_derivative)
    relation_defect=3*c_phi-kappa*phi_x
    relation_scale=max(1.,abs(3*c_phi),abs(kappa*phi_x))
    if abs(relation_defect)>1e-10*relation_scale:
        raise ValueError("coefficients do not satisfy the Israel/scalar variational relation")
    result=coupled_robin_matrix(
        umbilic_coefficient,c_phi,phi_x,
        wall_stiffness,kappa5_squared=kappa,
    )
    matrix=result["matrix"]
    a=-2*float(umbilic_coefficient)
    coupling=matrix[:10,10:12]
    reduced=matrix[10:12,10:12]
    separation_matrix=a*np.eye(2)-reduced
    separation=float(np.linalg.svd(separation_matrix,compute_uv=False)[-1])
    if separation<=separation_tolerance:
        raise ValueError("tangential and normal-scalar Robin blocks are not separated")

    # X(aI-D)=-B makes T^{-1}RT block diagonal for T=[[I,X],[0,I]].
    shear=-np.linalg.solve(separation_matrix.T,coupling.T).T
    transform=np.eye(13)
    transform[:10,10:12]=shear
    transform_inverse=np.eye(13)
    transform_inverse[:10,10:12]=-shear
    block_symmetrizer=np.eye(13)
    block_symmetrizer[11,11]=16*kappa/3
    symmetrizer=transform_inverse.T@block_symmetrizer@transform_inverse
    defect=symmetrizer@matrix-matrix.T@symmetrizer
    eigenvalues=np.linalg.eigvalsh(symmetrizer)
    transformed=transform_inverse@matrix@transform
    off_block=np.array(transformed,copy=True)
    off_block[:10,:10]=0;off_block[10:12,10:12]=0;off_block[12,12]=0
    return {
        "matrix":matrix,
        "symmetrizer":symmetrizer,
        "shear":shear,
        "normal_scalar_block":reduced,
        "spectral_separation_singular_value":separation,
        "minimum_eigenvalue":float(eigenvalues[0]),
        "maximum_eigenvalue":float(eigenvalues[-1]),
        "condition_number":float(eigenvalues[-1]/eigenvalues[0]),
        "symmetry_defect":float(np.linalg.norm(defect,ord=2)),
        "block_diagonalization_defect":float(np.linalg.norm(off_block,ord=2)),
    }


def physical_block_separation_determinant(
    umbilic_coefficient,wall_potential_derivative,wall_stiffness,
    kappa5_squared=1.0,
):
    """Exact determinant of ``a I-D`` for physical wall coefficients."""
    c=float(umbilic_coefficient);uprime=float(wall_potential_derivative)
    gamma=float(wall_stiffness);kappa=float(kappa5_squared)
    return float(5*c*gamma-20*c*c-kappa*uprime*uprime/3)


def frozen_full_boundary_symbol(laplace_real,laplace_imag,tangential_wavenumber,robin_matrix):
    """Return the 17-by-17 frozen boundary matrix and normalized singular gap."""
    sigma=complex(float(laplace_real),float(laplace_imag));wave=float(tangential_wavenumber)
    robin=np.asarray(robin_matrix,dtype=float)
    if sigma.real<=0 or wave<0 or robin.shape!=(13,13):raise ValueError("invalid frozen-symbol inputs")
    decay=np.sqrt(sigma*sigma+wave*wave)
    if decay.real<0 or (decay.real==0 and decay.imag<0):decay=-decay
    # Four mixed normal-tangent metric components obey Dirichlet conditions.
    boundary=np.zeros((17,17),dtype=complex);boundary[:4,:4]=np.eye(4)
    boundary[4:,4:]=decay*np.eye(13)-robin
    singular=np.linalg.svd(boundary,compute_uv=False)
    scale=max(1.,float(np.linalg.norm(boundary,ord=2)))
    return {
        "decay_rate":decay,"matrix":boundary,
        "minimum_singular_value":float(singular[-1]),
        "normalized_singular_gap":float(singular[-1]/scale),
        "unstable_root":bool(singular[-1]<=1e-12*scale),
    }
