import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.capped_surface import find_donor_capped_surfaces


class CappedSurfaceTests(unittest.TestCase):
    def test_half_tangherlini_control(self):
        z = np.linspace(0.2, 2.0, 45)
        r = np.linspace(0.0, 3.0, 65)
        z_b = z[-1]
        strength = 0.36
        distance2 = r[None, :]**2 + (z[:, None]-z_b)**2
        # Smooth the puncture only outside the analytic horizon; interpolation
        # never samples the central singular point for the accepted solution.
        psi = 1 + strength/np.maximum(distance2, 1e-8)
        result = find_donor_capped_surfaces(
            z, r, psi, guesses=(0.45, 0.6, 0.8), tolerance=5e-5
        )
        self.assertTrue(result["capped_surface_found"])
        expected = np.sqrt(strength)
        found = result["accepted"][0]
        self.assertAlmostEqual(found["rho_axis"], expected, delta=3e-2)
        self.assertAlmostEqual(found["rho_brane"], expected, delta=3e-2)
        self.assertEqual(found["negative_mode_count"],found["normalized_negative_mode_count"])
        angular=found["angular_mode_spectrum"]
        self.assertEqual([item["angular_mode"] for item in angular],[0,1,2,3])
        self.assertGreater(angular[1]["lowest_normalized_eigenvalue"],angular[0]["lowest_normalized_eigenvalue"])

    def test_flat_control_has_no_finite_cap(self):
        z = np.linspace(0.2, 2.0, 31)
        r = np.linspace(0.0, 3.0, 41)
        psi = np.ones((len(z), len(r)))
        result = find_donor_capped_surfaces(z, r, psi, guesses=(0.4, 0.8, 1.2))
        self.assertFalse(result["capped_surface_found"])


if __name__ == "__main__":
    unittest.main()
