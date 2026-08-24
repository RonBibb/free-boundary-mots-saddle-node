import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.test14_quasilocal_charge import (
    OMEGA3,
    analytic_controls,
    charge_from_integrals,
)


class Test14QuasilocalChargeTests(unittest.TestCase):
    def test_flat_round_s3_cancels(self):
        radius = 1.3
        value = charge_from_integrals(
            OMEGA3 * radius**3,
            12 * math.pi**2 * radius,
            18 * math.pi**2 * radius,
            cosmological_constant=0.0,
        )
        self.assertAlmostEqual(
            value["raw_generalized_hawking_charge_kappa5_squared_E"], 0.0,
            places=12,
        )

    def test_round_schwarzschild_ads5_calibration(self):
        radius = 0.9
        value = charge_from_integrals(
            OMEGA3 * radius**3,
            12 * math.pi**2 * radius,
            cosmological_constant=-6.0,
        )
        expected = 3 * math.pi**2 * radius**2 * (1 + radius**2)
        self.assertAlmostEqual(
            value["generalized_hawking_ads_charge_kappa5_squared_E"],
            expected, places=12,
        )
        self.assertAlmostEqual(value["intrinsic_curvature_shape_factor"], 1.0)

    def test_sealed_analytic_controls_pass(self):
        controls = analytic_controls()
        self.assertTrue(controls["passed"])
        self.assertGreater(
            controls["reflected_seam"]["bulk_only_relative_error"], 0.1,
        )


if __name__ == "__main__":
    unittest.main()

