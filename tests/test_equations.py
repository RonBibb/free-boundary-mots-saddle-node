import sys,unittest
from pathlib import Path
import sympy as sp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from bhps.equations import adm_four_spatial_scalar_projection_identity,conformal_hamiltonian_identities,israel_scalar_codazzi_identity,israel_second_corner_gauge_covariance_identity,leading_gw_radion_mass_identity,momentum_time_symmetry_identity,orbifold_ads_junction_identity,scalar_nec_identity,scalar_robin_corner_identity,scalar_wall_energy_flux_identity,unstabilized_radion_family_identity

class EquationTests(unittest.TestCase):
    def test_ads_solves_vacuum_constraint(self):
        self.assertEqual(conformal_hamiltonian_identities()["ads_residual"],0)
    def test_time_symmetry_solves_scalar_momentum_constraint(self):
        self.assertEqual(momentum_time_symmetry_identity()["time_symmetric_value"],0)
    def test_scalar_nec_is_sum_of_squares(self):
        self.assertTrue(scalar_nec_identity()["sum_of_squares"])
    def test_massless_energy_has_expected_conformal_power(self):
        identity=conformal_hamiltonian_identities(); text=str(identity["energy_integrand"])
        self.assertIn("psi(r, z)**2",text)
    def test_unstabilized_control_has_radion_family(self):
        identity=unstabilized_radion_family_identity()
        self.assertEqual(identity["bulk_residual"],0)
        self.assertEqual(identity["robin_residual"],0)
        self.assertNotEqual(identity["proper_separation_slope"],0)
        self.assertEqual(str(identity["zero_mode_tangent"]),"-ell/z**2")
    def test_leading_gw_radion_mass_derivation_closes(self):
        self.assertEqual(leading_gw_radion_mass_identity()["difference"],0)
    def test_orbifold_junction_orientation_reproduces_ads(self):
        identity=orbifold_ads_junction_identity()
        self.assertEqual(identity["lower_difference"],0)
        self.assertEqual(identity["upper_difference"],0)
    def test_time_symmetry_satisfies_first_scalar_robin_compatibility(self):
        identity=scalar_robin_corner_identity()
        self.assertEqual(identity["first_time_symmetric_value"],0)
        self.assertTrue(identity["geometry_acceleration_remainder_required"])
    def test_scalar_wall_flux_is_conservative(self):
        self.assertEqual(scalar_wall_energy_flux_identity()["total_rate"],0)
    def test_israel_and_scalar_robin_imply_boundary_momentum_constraint(self):
        self.assertEqual(israel_scalar_codazzi_identity()["difference"],0)
    def test_four_spatial_adm_scalar_projection(self):
        self.assertEqual(adm_four_spatial_scalar_projection_identity()["difference"],0)
    def test_israel_second_corner_is_gauge_covariant(self):
        identity=israel_second_corner_gauge_covariance_identity()
        self.assertEqual(identity["difference"],0)

if __name__=="__main__":unittest.main()
