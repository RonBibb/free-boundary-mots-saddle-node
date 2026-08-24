import sys,unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.adm_corner import spatial_israel_second_corner_residual_fields,time_symmetric_adm_metric_acceleration,time_symmetric_scalar_acceleration
from bhps.anisotropic_geometry import anisotropic_hamiltonian_residual,anisotropic_metric_acceleration,anisotropic_scalar_acceleration,anisotropic_scalar_gradient_energy,anisotropic_spatial_israel_second_corner_fields,axisymmetric_diagonal_geometry
from bhps.gauge_corner import compare_tangential_residual_fields


class AnisotropicGeometryTests(unittest.TestCase):
    @staticmethod
    def _fields(size=65):
        z=np.linspace(1,np.e,size);r=np.linspace(0,6,size+8)
        zz,rr=np.meshgrid(z,r,indexing="ij")
        psi=np.exp(.08*np.cos(1.3*zz)*np.exp(-(rr/2.5)**2))/zz
        phi=.1*np.cos(.7*zz)*np.exp(-(rr/3)**2)
        chi=.2*np.sin(.8*zz)*np.exp(-(rr/2)**2)
        # Use the same numerical derivatives in both implementations.
        from bhps.adm_corner import _axisymmetric_derivatives
        dchi=_axisymmetric_derivatives(chi,z,r,9)
        return z,r,psi,phi,chi,dchi["r"],dchi["z"]

    def test_flat_geometry_has_zero_curvature(self):
        z=np.linspace(1,2,33);r=np.linspace(0,4,41);one=np.ones((len(z),len(r)));zero=np.zeros_like(one)
        geometry=axisymmetric_diagonal_geometry(z,r,one,zero,zero,zero,9)
        self.assertLess(np.max(np.abs(geometry["scalar_curvature"])),2e-10)

    def test_mixed_metric_acceleration_enforces_axis_regularity(self):
        z,r,psi,phi,chi,chi_r,chi_z=self._fields();zero=np.zeros_like(psi)
        acceleration=anisotropic_metric_acceleration(
            z,r,psi,zero,zero,zero,phi,chi_r,chi_z,.41,
            stencil_width=9,lapse=psi,
        )
        np.testing.assert_array_equal(acceleration["zr"][:,0],0.)

    def test_conformal_geometry_matches_existing_adm_acceleration(self):
        z,r,psi,phi,chi,chi_r,chi_z=self._fields()
        zero=np.zeros_like(psi)
        old=time_symmetric_adm_metric_acceleration(
            z,r,psi,phi,chi_r,chi_z,.41,m_chi_squared=.2,chi=chi,
            stencil_width=9,lapse=psi,
        )
        new=anisotropic_metric_acceleration(
            z,r,psi,zero,zero,zero,phi,chi_r,chi_z,.41,
            m_chi_squared=.2,chi=chi,stencil_width=9,lapse=psi,
        )
        for name in ("zz","radial","transverse","zr"):
            self.assertLess(np.max(np.abs(old[name]-new[name])),2e-5)

    def test_conformal_scalar_acceleration_matches_existing_formula(self):
        z,r,psi,phi,chi,chi_r,chi_z=self._fields();zero=np.zeros_like(psi)
        old=time_symmetric_scalar_acceleration(z,r,psi,phi,.41,lapse=psi,stencil_width=9)
        new=anisotropic_scalar_acceleration(z,r,psi,zero,zero,zero,phi,.41,lapse=psi,stencil_width=9)
        self.assertLess(np.max(np.abs(old-new)),2e-6)

    def test_conformal_hamiltonian_matches_constraint_normalization(self):
        z,r,psi,phi,chi,chi_r,chi_z=self._fields();zero=np.zeros_like(psi)
        residual=anisotropic_hamiltonian_residual(
            z,r,psi,zero,zero,zero,phi,chi_r,chi_z,.41,
            m_chi_squared=.2,chi=chi,stencil_width=9,
        )
        # Independently reconstruct the known conformal Hamiltonian equation.
        from bhps.adm_corner import _axisymmetric_derivatives
        dp=_axisymmetric_derivatives(psi,z,r,9);df=_axisymmetric_derivatives(phi,z,r,9)
        lap=dp["zz"]+dp["rr"]+2*dp["transverse_hessian"]
        known=-6*lap/psi**3+12-(df["z"]**2+df["r"]**2+chi_z**2+chi_r**2)/psi**2-.41*phi**2-.2*chi**2
        self.assertLess(np.max(np.abs(residual-known)),2e-5)

    def test_scalar_gradient_energy_reduces_to_conformal_formula(self):
        z,r,psi,phi,chi,chi_r,chi_z=self._fields();zero=np.zeros_like(psi)
        energy=anisotropic_scalar_gradient_energy(
            z,r,psi,zero,zero,zero,chi_r,chi_z,
        )
        from scipy.integrate import simpson
        expected=float(simpson(simpson(
            2*np.pi*r[None,:]**2*psi**2*(chi_r**2+chi_z**2),x=r,axis=1,
        ),x=z))
        self.assertAlmostEqual(energy,expected,places=11)

    def test_general_corner_reduces_to_conformal_corner(self):
        z,r,psi,phi,chi,chi_r,chi_z=self._fields();zero=np.zeros_like(psi)
        background={"wall_stiffness":0.,"v0":0.,"v1":0.,"beta_a":1.,"beta_b":1.,"wall_potential_a":0.,"wall_potential_b":0.}
        acceleration=time_symmetric_adm_metric_acceleration(z,r,psi,phi,chi_r,chi_z,.41,stencil_width=9,lapse=psi)
        old=spatial_israel_second_corner_residual_fields(acceleration,psi,phi,background,zero,7)
        new=anisotropic_spatial_israel_second_corner_fields(acceleration,psi,zero,zero,zero,phi,background,zero,7)
        comparison=compare_tangential_residual_fields(old,new)
        self.assertLess(comparison["maximum_fixed_scaled_covariance_defect"],1e-12)
        for old_wall,new_wall in zip(old["walls"],new["walls"]):
            self.assertLess(np.max(np.abs(
                old_wall["mixed_zr_residual"]-new_wall["mixed_zr_residual"]
            )),1e-12)
            self.assertLess(np.max(np.abs(
                old_wall["mixed_zr_scale"]-new_wall["mixed_zr_scale"]
            )),1e-12)


if __name__=="__main__":unittest.main()
