import sys,unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.gh_operator_coefficients import (
    FIELD_ORDER,frozen_source_gh_coefficient_matrices,pack_wall_adapted_residual,
)
from bhps.linearized_gh_einstein_scalar import linearized_reduced_einstein_two_scalar_residual


def flat_background(phi=0.,chi=0.):
    return {
        "metric":np.diag((-1.,1.,1.,1.,1.)),
        "metric_first":np.zeros((5,5,5)),"metric_second":np.zeros((5,5,5,5)),
        "phi":phi,"phi_first":np.zeros(5),"phi_second":np.zeros((5,5)),
        "chi":chi,"chi_first":np.zeros(5),"chi_second":np.zeros((5,5)),
    }


class GeneralizedHarmonicOperatorCoefficientTests(unittest.TestCase):
    def test_field_order_matches_four_dirichlet_plus_thirteen_robin_blocks(self):
        self.assertEqual(len(FIELD_ORDER),17)
        self.assertEqual(FIELD_ORDER[:4],("h_z0","h_zx","h_zy","h_zw"))
        self.assertEqual(FIELD_ORDER[-3:],("h_zz","delta_Phi","delta_chi"))

    def test_flat_massless_coefficients_vanish(self):
        result=frozen_source_gh_coefficient_matrices(
            flat_background(),potential_offset=0.,
        )
        np.testing.assert_allclose(result["zero_order_matrix"],0.,atol=2e-13)
        np.testing.assert_allclose(result["coordinate_first_matrices"],0.,atol=2e-13)
        np.testing.assert_allclose(result["scalar_wave_adjusted_first_matrices"],0.,atol=2e-13)
        self.assertTrue(result["finite"])

    def test_flat_massive_scalar_has_positive_evolution_reaction(self):
        mass=.41
        result=frozen_source_gh_coefficient_matrices(
            flat_background(),mass_squared=mass,potential_offset=0.,
        )
        phi=FIELD_ORDER.index("delta_Phi")
        self.assertAlmostEqual(result["evolution_reaction_matrix"][phi,phi],mass)
        off=result["evolution_reaction_matrix"].copy();off[phi,phi]=0.
        np.testing.assert_allclose(off,0.,atol=2e-13)

    def test_extracted_value_and_first_matrices_reconstruct_direct_linearization(self):
        rng=np.random.default_rng(12);background=flat_background(phi=.2,chi=-.1)
        background["metric_first"]=.02*rng.normal(size=(5,5,5))
        background["metric_first"]=.5*(background["metric_first"]+background["metric_first"].swapaxes(1,2))
        background["metric_second"]=.01*rng.normal(size=(5,5,5,5))
        background["metric_second"]=.5*(background["metric_second"]+background["metric_second"].swapaxes(0,1))
        background["metric_second"]=.5*(background["metric_second"]+background["metric_second"].swapaxes(2,3))
        background["phi_first"]=.03*rng.normal(size=5);background["chi_first"]=.03*rng.normal(size=5)
        coefficients=frozen_source_gh_coefficient_matrices(
            background,mass_squared=.41,potential_offset=-6.,
        )
        values=.1*rng.normal(size=17);derivatives=.1*rng.normal(size=(5,17))
        perturbation=flat_background();perturbation["metric"][:]=0.
        perturbation["phi"]=values[15];perturbation["chi"]=values[16]
        pairs=coefficients["metric_pairs"]
        for column,pair in enumerate(pairs):
            perturbation["metric"][pair]=values[column]
            perturbation["metric"][pair[::-1]]=values[column]
            for derivative in range(5):
                perturbation["metric_first"][derivative][pair]=derivatives[derivative,column]
                perturbation["metric_first"][derivative][pair[::-1]]=derivatives[derivative,column]
        perturbation["phi_first"]=derivatives[:,15]
        perturbation["chi_first"]=derivatives[:,16]
        direct=linearized_reduced_einstein_two_scalar_residual(
            background,perturbation,mass_squared=.41,potential_offset=-6.,
        )
        packed=pack_wall_adapted_residual(
            direct["metric_residual"],direct["phi_residual"],direct["chi_residual"],
        )
        normalization=coefficients["row_normalization"]
        expected=coefficients["zero_order_matrix"]@values
        expected+=np.einsum("mij,mj->i",coefficients["coordinate_first_matrices"],derivatives)
        np.testing.assert_allclose(normalization*packed,expected,atol=2e-12)


if __name__=="__main__":unittest.main()
