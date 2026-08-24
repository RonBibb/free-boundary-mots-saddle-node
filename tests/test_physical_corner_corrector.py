import sys,unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.physical_corner_corrector import combine_shape_modes,compact_double_zero_hermite_basis,radial_buffer_for_cutoff,radial_corrector_basis,tracefree_shape_basis


class PhysicalCornerCorrectorTests(unittest.TestCase):
    def test_compact_profiles_have_required_double_zeros(self):
        z=np.linspace(1,np.e,65);basis=compact_double_zero_hermite_basis(z)
        from bhps.gw_slice_high_order_solver import derivative_matrix
        dz=derivative_matrix(z,1,9)
        for profile in basis["profiles"]:
            self.assertLess(max(abs(profile[0]),abs(profile[-1])),2e-12)
            self.assertLess(max(abs((dz@profile)[0]),abs((dz@profile)[-1])),2e-8)

    def test_radial_profiles_are_axis_regular_and_outer_flat(self):
        r=np.linspace(0,8,81);basis=radial_corrector_basis(r,6)
        from bhps.gw_slice_high_order_solver import derivative_matrix
        dr=derivative_matrix(r,1,9)
        for profile in basis["profiles"]:
            self.assertLess(abs((dr@profile)[0]),2e-7)
            self.assertLess(abs(profile[-1]),1e-14)
            self.assertLess(abs((dr@profile)[-1]),2e-5)

    def test_tracefree_modes_and_axis_regularity(self):
        z=np.linspace(1,np.e,33);r=np.linspace(0,8,49)
        basis=tracefree_shape_basis(z,r,3)
        for (a,b,c),label in zip(basis["modes"],basis["labels"]):
            self.assertLess(np.max(np.abs(a+b+2*c)),1e-14)
            self.assertLess(np.max(np.abs(b[:,0]-c[:,0])),1e-14)

    def test_shape_combination_preserves_trace_and_axis(self):
        z=np.linspace(1,np.e,17);r=np.linspace(0,4,25);basis=tracefree_shape_basis(z,r,2)
        coefficients=np.linspace(-.03,.04,len(basis["modes"]))
        a,b,c=combine_shape_modes(coefficients,basis["modes"])
        self.assertLess(np.max(np.abs(a+b+2*c)),1e-14)
        self.assertLess(np.max(np.abs(b[:,0]-c[:,0])),1e-14)

    def test_axis_localized_modes_are_regular_and_appended(self):
        z=np.linspace(1,np.e,17);r=np.linspace(0,8,25)
        base=tracefree_shape_basis(z,r,2);extended=tracefree_shape_basis(z,r,2,(.5,1.))
        for old,new in zip(base["modes"],extended["modes"]):
            for old_component,new_component in zip(old,new):
                self.assertLess(np.max(np.abs(old_component-new_component)),1e-14)
        self.assertEqual(len(extended["modes"]),len(base["modes"])+16)
        for a,b,c in extended["modes"]:
            self.assertLess(np.max(np.abs(a+b+2*c)),1e-14)
            self.assertLess(np.max(np.abs(b[:,0]-c[:,0])),1e-14)

    def test_physical_radial_buffer_scales_with_grid(self):
        self.assertEqual(radial_buffer_for_cutoff(np.linspace(0,8,49),6.75),8)
        self.assertEqual(radial_buffer_for_cutoff(np.linspace(0,8,97),6.75),15)

    def test_annular_modes_are_domain_invariant_and_axis_regular(self):
        z=np.exp(np.linspace(0,1,17));r8=np.linspace(0,8,73);r12=np.linspace(0,12,109)
        small=tracefree_shape_basis(
            z,r8,radial_modes=1,annular_profiles=((7.5,1.5),(7.5,3.)),
        )
        large=tracefree_shape_basis(
            z,r12,radial_modes=1,annular_profiles=((7.5,1.5),(7.5,3.)),
        )
        annular_count=4*2*2
        self.assertEqual(len(small["modes"]),len(large["modes"]))
        for small_mode,large_mode in zip(
            small["modes"][-annular_count:],large["modes"][-annular_count:],
        ):
            for small_field,large_field in zip(small_mode,large_mode):
                np.testing.assert_allclose(small_field,large_field[:,:len(r8)])
            np.testing.assert_allclose(small_mode[1][:,0],small_mode[2][:,0],atol=1e-12)

    def test_fixed_radius_basis_is_domain_invariant(self):
        z=np.linspace(1,np.e,17);r8=np.linspace(0,8,73);r10=np.linspace(0,10,91)
        b8=tracefree_shape_basis(z,r8,3,(.5,1.),basis_radius=8.)["modes"]
        b10=tracefree_shape_basis(z,r10,3,(.5,1.),basis_radius=8.)["modes"]
        for mode8,mode10 in zip(b8,b10):
            for component8,component10 in zip(mode8,mode10):
                self.assertLess(np.max(np.abs(component8-component10[:,:len(r8)])),1e-13)
                self.assertLess(np.max(np.abs(component10[:,len(r8):])),1e-14)


if __name__=="__main__":unittest.main()
