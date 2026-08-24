import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.corrected_A790_R10_builder import (
    ANNULAR_PROFILES,
    AXIS_WIDTHS,
    BASIS_RADIUS,
    matched_radial_count,
    shape_fields,
)


class CorrectedA790R10BuilderTests(unittest.TestCase):
    def test_matched_counts_preserve_G7_and_G8_spacing(self):
        for original, extended in ((121, 151), (145, 181)):
            self.assertEqual(matched_radial_count(original), extended)
            self.assertAlmostEqual(8.0 / (original - 1), 10.0 / (extended - 1))

    def test_domain_independent_shape_agrees_on_common_nodes(self):
        archive = np.load("results/corrected_family_knot_A8_state.npz")
        coefficients = archive["coefficients"]
        z = np.linspace(np.exp(0.0), np.exp(1.0), 17)
        r8 = np.linspace(0.0, 8.0, 121)
        r10 = np.linspace(0.0, 10.0, 151)
        shape8 = shape_fields(z, r8, coefficients)
        shape10 = shape_fields(z, r10, coefficients)
        for left, right in zip(shape8, shape10):
            np.testing.assert_allclose(left, right[:, :121], rtol=0.0, atol=2e-14)

    def test_builder_uses_domain_qualified_basis(self):
        self.assertEqual(BASIS_RADIUS, 8.0)
        self.assertEqual(AXIS_WIDTHS, (0.5, 1.0))
        self.assertEqual(ANNULAR_PROFILES, ((7.5, 1.5), (7.5, 3.0)))


if __name__ == "__main__":
    unittest.main()
