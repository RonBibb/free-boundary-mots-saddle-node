import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.correlated_affine_spline import AffineForm, affine_sqrt


class AffineFormTests(unittest.TestCase):
    def test_shared_parameter_cancellation_is_exact_up_to_roundoff(self):
        x = AffineForm.parameter(2.0, 0.3)
        y = x - x
        self.assertTrue(y.range.contains(0.0))
        self.assertLess(y.range.width, 1e-14)

    def test_product_reciprocal_and_sqrt_enclose_dense_values(self):
        x = AffineForm.parameter(2.0, 0.2)
        expressions = (x * x, x.reciprocal(), affine_sqrt(x))
        functions = (lambda value: value**2, lambda value: 1.0/value, np.sqrt)
        for expression, function in zip(expressions, functions):
            for epsilon in np.linspace(-1.0, 1.0, 101):
                self.assertTrue(
                    expression.range.contains(function(2.0 + 0.2 * epsilon))
                )


if __name__ == "__main__":
    unittest.main()
