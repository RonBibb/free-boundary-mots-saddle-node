import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.corrected_A790_R12_builder import (
    ANNULAR_PROFILES,
    AXIS_WIDTHS,
    BASIS_RADIUS,
    KNOT_STATE,
    build_A790_R12_refined,
    matched_radial_count,
    shape_fields,
)


class CorrectedA790R12BuilderTests(unittest.TestCase):
    def test_matched_counts_preserve_G7_and_G8_spacing(self):
        for original, extended in ((121, 181), (145, 217)):
            self.assertEqual(matched_radial_count(original), extended)
            self.assertAlmostEqual(8.0 / (original - 1), 12.0 / (extended - 1))

    def test_domain_independent_shape_agrees_on_common_nodes(self):
        archive = np.load(KNOT_STATE)
        coefficients = archive["coefficients"]
        z = np.linspace(np.exp(0.0), np.exp(1.0), 17)
        r8 = np.linspace(0.0, 8.0, 121)
        r12 = np.linspace(0.0, 12.0, 181)
        shape8 = shape_fields(z, r8, coefficients)
        shape12 = shape_fields(z, r12, coefficients)
        for left, right in zip(shape8, shape12):
            np.testing.assert_allclose(left, right[:, :121], rtol=0.0, atol=2e-14)

    def test_archive_has_genuine_R12_seed(self):
        archive = np.load(KNOT_STATE)
        self.assertEqual(archive["q_G5R12"].shape, (49, 109))
        self.assertEqual(archive["phi_G5R12"].shape, (49, 109))

    def test_refinement_rejects_smaller_domain_seed(self):
        with self.assertRaisesRegex(ValueError, "Rmax=12"):
            build_A790_R12_refined(
                {"r": np.linspace(0.0, 10.0, 7)}, 9, 13, "invalid",
            )

    def test_builder_uses_domain_qualified_basis(self):
        self.assertEqual(BASIS_RADIUS, 8.0)
        self.assertEqual(AXIS_WIDTHS, (0.5, 1.0))
        self.assertEqual(ANNULAR_PROFILES, ((7.5, 1.5), (7.5, 3.0)))


if __name__ == "__main__":
    unittest.main()

