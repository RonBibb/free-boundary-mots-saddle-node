import sys,unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))

from bhps.israel_wave_matrix import analytic_robin_symmetrizer,coupled_robin_matrix,frozen_full_boundary_symbol,matrix_spectral_audit,physical_block_separation_determinant


class IsraelWaveMatrixTests(unittest.TestCase):
    def test_matrix_contains_all_seventeen_boundary_rows(self):
        robin=coupled_robin_matrix(1.,-.01,-.03,20.)["matrix"]
        symbol=frozen_full_boundary_symbol(20.,0.,0.,robin)
        self.assertEqual(symbol["matrix"].shape,(17,17))
        self.assertFalse(symbol["unstable_root"])

    def test_actual_sign_pattern_is_symmetrizable(self):
        robin=coupled_robin_matrix(-1.,.002,.006,20.)["matrix"]
        audit=matrix_spectral_audit(robin)
        self.assertTrue(audit["all_real"])
        self.assertTrue(audit["diagonalizable"])
        self.assertGreater(audit["symmetrizer_minimum_eigenvalue"],0.)
        self.assertLess(audit["symmetry_defect"],1e-8)

    def test_shift_above_robin_spectral_abscissa_has_no_root(self):
        robin=coupled_robin_matrix(-1.,.002,.006,20.)["matrix"]
        growth=matrix_spectral_audit(robin)["maximum_positive_eigenvalue"]
        for imag in (-20.,-1.,0.,1.,20.):
            for wave in (0.,.2,2.,20.):
                result=frozen_full_boundary_symbol(growth+.1,imag,wave,robin)
                self.assertFalse(result["unstable_root"])

    def test_analytic_symmetrizer_is_positive_and_exact(self):
        audit=analytic_robin_symmetrizer(-1.0088,.00125,.00375,20.)
        self.assertGreater(audit["minimum_eigenvalue"],0.)
        self.assertLess(audit["symmetry_defect"],1e-10)
        self.assertLess(audit["block_diagonalization_defect"],1e-10)

    def test_analytic_symmetrizer_is_smooth_through_zero_wall_slope(self):
        for wall_slope in np.linspace(-.02,.02,17):
            audit=analytic_robin_symmetrizer(
                1.,wall_slope/6,wall_slope/2,20.,
            )
            self.assertGreater(audit["minimum_eigenvalue"],0.)
            self.assertLess(audit["symmetry_defect"],1e-10)

    def test_physical_separation_determinant(self):
        c=-1.0088;uprime=.0075;gamma=20.
        expected=5*c*gamma-20*c*c-uprime*uprime/3
        self.assertAlmostEqual(
            physical_block_separation_determinant(c,uprime,gamma),expected,
        )

    def test_nonvariational_coefficients_are_rejected(self):
        with self.assertRaises(ValueError):
            analytic_robin_symmetrizer(1.,.01,.01,20.)


if __name__=="__main__":unittest.main()
