import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bhps.corrected_A790_test10e_boundary_corner_localization import (
    clipped_trapezoid,
    compact_wall_derivative_defect,
    cumulative_open_face_proper_coordinate,
    empirical_power_law,
    localize_face_enumeration,
    manufactured_controls,
)


class TestArchiveBoundaryLocalization(unittest.TestCase):
    def test_proper_coordinate_and_clipped_integral(self):
        z = np.linspace(1.0, 2.0, 9)[1:-1]
        proper = cumulative_open_face_proper_coordinate(z, np.full(len(z), 4.0))
        self.assertTrue(np.allclose(proper, 2.0 * (z - z[0])))
        value = clipped_trapezoid(proper, 1.0 + proper, 0.1, 1.1)
        exact = (1.1 + 0.5 * 1.1**2) - (0.1 + 0.5 * 0.1**2)
        self.assertAlmostEqual(value, exact, places=13)

    def test_constant_face_integral_and_phi_split(self):
        z = np.linspace(1.0, 2.0, 65)
        open_count = len(z) - 2
        q = np.ones(open_count)
        before = np.zeros((open_count, 2))
        after = np.column_stack((np.ones(open_count), 2.0 * np.ones(open_count)))
        result = localize_face_enumeration(z, 2.0, q, q, before, after)
        length = z[-2] - z[1]
        weight = 4.0 * math.pi * 4.0
        self.assertAlmostEqual(result["total_squared_l2"][0], weight * length)
        self.assertAlmostEqual(result["total_squared_l2"][1], 4.0 * weight * length)
        self.assertAlmostEqual(result["phi_squared_l2_fraction"], 0.2)
        self.assertAlmostEqual(result["combined_pointwise_maximum"], math.sqrt(5.0))
        for record in result["proper_collars"]:
            self.assertAlmostEqual(
                record["combined_union_squared_l2_fraction"],
                min(2.0 * record["width"] / length, 1.0),
                places=12,
            )
        for record in result["node_collars"]:
            self.assertAlmostEqual(
                record["combined_union_squared_l2_fraction"]
                + record["combined_central_squared_l2_fraction"],
                1.0,
                places=12,
            )

    def test_nonfinite_radius_and_nonfitting_collar_are_rejected(self):
        z = np.linspace(1.0, 2.0, 65)
        q = np.ones(len(z) - 2)
        before = np.zeros((len(q), 2))
        after = before.copy()
        with self.assertRaisesRegex(ValueError, "radius"):
            localize_face_enumeration(z, np.nan, q, q, before, after)
        after[4, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "not finite"):
            localize_face_enumeration(z, 2.0, q, q, before, after)
        after[4, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "does not fit"):
            localize_face_enumeration(
                z, 2.0, q, q, before, after, proper_widths=(0.50,),
            )

    def test_compatible_quartic_has_negligible_corner_defect(self):
        z = np.linspace(0.0, 1.0, 65)
        x = z[1:-1]
        delta = np.column_stack((x**2 * (1.0 - x) ** 2, np.zeros_like(x)))
        defect = compact_wall_derivative_defect(z, delta)
        self.assertLess(np.max(defect["absolute"]), 1e-10)

    def test_single_open_node_spike_has_corner_defect(self):
        z = np.linspace(0.0, 1.0, 65)
        delta = np.zeros((len(z) - 2, 2))
        delta[0, 0] = 1.0
        delta[-1, 0] = 1.0
        defect = compact_wall_derivative_defect(z, delta)
        self.assertGreater(np.min(defect["absolute"][:, 0]), 1.0)
        self.assertGreater(
            np.min(defect["derivative_cancellation_normalized"][:, 0]), 0.99,
        )

    def test_empirical_power_law(self):
        counts = np.asarray((80.0, 96.0, 112.0, 128.0))
        self.assertAlmostEqual(empirical_power_law(counts, 3.0 * counts**2), 2.0)
        self.assertIsNone(empirical_power_law(counts, np.zeros(4)))

    def test_self_and_adverse_controls(self):
        controls = manufactured_controls()
        self.assertTrue(controls["passed"], controls)
        self.assertTrue(all(controls["gates"].values()))


if __name__ == "__main__":
    unittest.main()
