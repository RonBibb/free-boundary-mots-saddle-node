import sys,unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.adm_corner import add_metric_accelerations,shift_acceleration_correction,spatial_israel_second_corner_audit,time_symmetric_adm_metric_acceleration,time_symmetric_scalar_acceleration
from bhps.gauge_corner import compare_tangential_residual_fields,corner_fields,maximum_tangential_residual
from bhps.gw_background import solve_gw_background


class AdmCornerTests(unittest.TestCase):
    def test_exact_ads_acceleration_converges_to_zero(self):
        maxima=[]
        for size in (33,65):
            z=np.linspace(1,np.e,size);r=np.linspace(0,8,size+8)
            psi=np.repeat((1/z)[:,None],len(r),axis=1);zero=np.zeros_like(psi)
            result=time_symmetric_adm_metric_acceleration(
                z,r,psi,zero,zero,zero,0.,stencil_width=9,
            )
            maxima.append(result["maximum_absolute_acceleration"])
        self.assertLess(maxima[1],maxima[0]/20)
        self.assertLess(maxima[1],3e-6)

    def test_zero_acceleration_clears_spatial_corner(self):
        z=np.linspace(1,np.e,17);r=np.linspace(0,4,21)
        background=solve_gw_background(z,wall_stiffness=20.)
        psi=np.repeat(background["psi"][:,None],len(r),axis=1)
        phi=np.repeat(background["phi"][:,None],len(r),axis=1)
        zero=np.zeros_like(psi)
        audit=spatial_israel_second_corner_audit(
            {"zz":zero,"radial":zero,"transverse":zero,"zr":zero,"Dz":__import__('bhps.gw_slice_high_order_solver',fromlist=['derivative_matrix']).derivative_matrix(z,1,7)},
            psi,phi,background,
        )
        self.assertEqual(audit["maximum_tangential_normalized_residual"],0.)
        self.assertEqual(audit["maximum_mixed_zr_acceleration"],0.)

    def test_explicit_psi_lapse_matches_default(self):
        z=np.linspace(1,np.e,25);r=np.linspace(0,6,31)
        psi=np.repeat((1/z)[:,None],len(r),axis=1);zero=np.zeros_like(psi)
        default=time_symmetric_adm_metric_acceleration(z,r,psi,zero,zero,zero,0.,stencil_width=7)
        explicit=time_symmetric_adm_metric_acceleration(z,r,psi,zero,zero,zero,0.,stencil_width=7,lapse=psi)
        for component in ("zz","radial","transverse","zr"):
            self.assertLess(np.max(np.abs(default[component]-explicit[component])),1e-12)

    def test_constant_scalar_has_mass_acceleration_only(self):
        z=np.linspace(1,np.e,17);r=np.linspace(0,4,21)
        psi=np.repeat((1/z)[:,None],len(r),axis=1);field=np.full_like(psi,.2)
        result=time_symmetric_scalar_acceleration(z,r,psi,field,3.,lapse=psi,stencil_width=7)
        self.assertLess(np.max(np.abs(result+3*psi**2*field)),1e-10)

    def test_zero_shift_has_zero_correction(self):
        z=np.linspace(1,np.e,17);r=np.linspace(0,4,21)
        psi=np.repeat((1/z)[:,None],len(r),axis=1);zero=np.zeros_like(psi)
        result=shift_acceleration_correction(z,r,psi,zero,zero,stencil_width=7)
        for component in ("zz","radial","transverse","zr"):
            self.assertEqual(np.max(np.abs(result[component])),0.)

    @staticmethod
    def _manufactured_nonzero_corner(size):
        z=np.linspace(1,np.e,size);r=np.linspace(0,8,size+16)
        zz,rr=np.meshgrid(z,r,indexing="ij")
        psi=1/zz;phi=np.zeros_like(psi)
        compact=(zz-1)/(np.e-1)
        chi=.4*np.cos(np.pi*compact)*np.exp(-(rr/2)**2)
        chi_z=-.4*np.pi/(np.e-1)*np.sin(np.pi*compact)*np.exp(-(rr/2)**2)
        chi_r=-rr*chi/2
        background={
            "wall_stiffness":0.,"v0":0.,"v1":0.,"beta_a":1.,"beta_b":1.,
            "wall_potential_a":0.,"wall_potential_b":0.,
        }
        acceleration=time_symmetric_adm_metric_acceleration(
            z,r,psi,phi,chi_r,chi_z,0.,m_chi_squared=.3,chi=chi,
            stencil_width=9,lapse=psi,
        )
        return z,r,zz,rr,psi,phi,chi,chi_r,chi_z,background,acceleration

    def test_variable_lapse_scales_geometric_corner_defect(self):
        errors=[]
        for size in (33,65):
            z,r,zz,rr,psi,phi,chi,chi_r,chi_z,background,base=self._manufactured_nonzero_corner(size)
            reference=corner_fields(base,psi,phi,background,np.zeros_like(psi),7)
            factor=np.exp(.25*np.cos(np.pi*rr/16))
            changed=time_symmetric_adm_metric_acceleration(
                z,r,psi,phi,chi_r,chi_z,0.,m_chi_squared=.3,chi=chi,
                stencil_width=9,lapse=factor*psi,
            )
            candidate=corner_fields(changed,psi,phi,background,np.zeros_like(psi),7)
            comparison=compare_tangential_residual_fields(
                reference,candidate,[factor[0,:-7]**2,factor[-1,:-7]**2],
            )
            errors.append(comparison["maximum_fixed_scaled_covariance_defect"])
            self.assertGreater(maximum_tangential_residual(reference),.02)
        self.assertLess(errors[1],errors[0]/20)
        self.assertLess(errors[1],5e-8)

    def test_tangential_shift_is_null_on_geometric_corner_defect(self):
        errors=[]
        for size in (33,65):
            z,r,zz,rr,psi,phi,chi,chi_r,chi_z,background,base=self._manufactured_nonzero_corner(size)
            reference=corner_fields(base,psi,phi,background,np.zeros_like(psi),7)
            compact=(zz-1)/(np.e-1)
            shift_r=.1*np.cos(2*np.pi*compact)*np.sin(np.pi*rr/8)
            correction=shift_acceleration_correction(
                z,r,psi,np.zeros_like(psi),shift_r,stencil_width=9,
            )
            changed=add_metric_accelerations(base,correction)
            candidate=corner_fields(changed,psi,phi,background,np.zeros_like(psi),7)
            comparison=compare_tangential_residual_fields(reference,candidate)
            errors.append(comparison["maximum_fixed_scaled_covariance_defect"])
        self.assertLess(errors[1],errors[0]/20)
        self.assertLess(errors[1],5e-8)


if __name__=="__main__":unittest.main()
