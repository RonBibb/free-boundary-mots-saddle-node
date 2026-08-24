import math
import unittest

import numpy as np

from bhps.dynamical_capped_geometry import capped_surface_geometry


class DynamicalCappedGeometryTests(unittest.TestCase):
    def test_flat_hemisphere_has_expected_area_and_radius(self):
        z = np.linspace(0, 3, 21)
        r = np.linspace(0, 3, 25)
        position = np.zeros((len(z), len(r), 9))
        position[:, :, 2] = -1
        position[:, :, 3] = 1
        position[:, :, 6] = 1
        velocity = np.zeros_like(position)
        theta = np.linspace(0, math.pi / 2, 401)
        radius = 0.8
        result = capped_surface_geometry(
            position, velocity, z, r,
            {"theta": theta, "rho": np.full_like(theta, radius),
             "slope": np.zeros_like(theta)},
        )
        self.assertAlmostEqual(
            result["one_sided_cap_area"], math.pi**2 * radius**3, places=8,
        )
        self.assertAlmostEqual(result["equivalent_area_radius"], radius, places=8)
        self.assertAlmostEqual(
            result["proper_meridional_length"], math.pi * radius / 2, places=8,
        )
        self.assertAlmostEqual(result["endpoint_shape_ratio"], 1, places=12)
        self.assertAlmostEqual(result["meridional_shape_ratio"], 1, places=8)

    def test_rejects_profile_outside_grid(self):
        z = np.linspace(0, 1, 9)
        r = np.linspace(0, 1, 11)
        position = np.zeros((len(z), len(r), 9))
        position[:, :, 2] = -1
        position[:, :, 3] = 1
        position[:, :, 6] = 1
        theta = np.linspace(0, math.pi / 2, 21)
        with self.assertRaises(ValueError):
            capped_surface_geometry(
                position, np.zeros_like(position), z, r,
                {"theta": theta, "rho": np.full_like(theta, 2),
                 "slope": np.zeros_like(theta)},
            )


if __name__ == "__main__":
    unittest.main()
